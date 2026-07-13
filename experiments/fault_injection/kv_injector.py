# SPDX-License-Identifier: Apache-2.0
"""Single-bit KV-cache fault injection via a FlashAttention forward hook.

Design (matches exp_design.md sections 3-4):

* We wrap ``FlashAttentionImpl.forward`` with a post-hook. After the original
  forward runs, the current step's K/V has already been written into the
  persistent paged KV cache by ``reshape_and_cache_flash`` and read by
  ``flash_attn_varlen_func``. Flipping a bit in that persistent cache therefore
  corrupts the value that every *subsequent* decode step will read for the
  target token -- the "stored KV suffered a soft error" semantics.

* ``phase == "prefill"``: inject into a prompt token (default: the middle
  prompt token) right after prefill. The first sampled token is produced from
  clean KV; every decode step afterwards reads the corrupted KV. This is
  experiment 1A.

* ``phase == "generated"``: inject into the just-written V/K row of a chosen
  decode step. This is experiment 1B.

The KV cache layout for the FlashAttention v1 backend is::

    kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
              (index 0 == K, index 1 == V), dtype bfloat16

Injection is one-shot per run: call :meth:`Injector.begin_run` before each
generation and :meth:`Injector.set_config` to arm it.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


@dataclass
class FaultConfig:
    """One controlled single-bit fault."""

    layer: int
    kv: str            # "K" or "V"
    bit: int           # 0..15 (BF16: 0-6 mantissa, 7-14 exponent, 15 sign)
    kv_head: int
    dim: int           # index within head_size
    phase: str = "prefill"          # "prefill" | "generated"
    token_index: Optional[int] = None  # prefill: None -> middle prompt token
    injection_step: int = 1         # generated: 1-based decode step to hit

    def kv_index(self) -> int:
        if self.kv.upper() == "K":
            return 0
        if self.kv.upper() == "V":
            return 1
        raise ValueError(f"kv must be 'K' or 'V', got {self.kv!r}")


@dataclass
class InjectionRecord:
    """Everything section 9 asks us to log about a single flip."""

    layer: int
    kv: str
    phase: str
    block_id: int
    token_offset: int
    slot: int
    kv_head: int
    head_dim: int
    bit: int
    old_bits: int
    new_bits: int
    old_value: float
    new_value: float
    decode_step: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


class Injector:
    """Process-wide fault injector installed into FlashAttentionImpl."""

    def __init__(self) -> None:
        self._config: Optional[FaultConfig] = None
        self._enabled = False
        self._lock = threading.Lock()

        # Per-run state.
        self._injected = False
        self._decode_steps: dict[int, int] = {}
        self._records: list[InjectionRecord] = []

        # Discovered lazily from the first cache we see.
        self.num_kv_heads: Optional[int] = None
        self.head_size: Optional[int] = None
        self.block_size: Optional[int] = None
        self.num_layers_seen: set[int] = set()

    # -- lifecycle ---------------------------------------------------------
    def set_config(self, config: Optional[FaultConfig]) -> None:
        with self._lock:
            self._config = config
            self._enabled = config is not None

    def disable(self) -> None:
        self.set_config(None)

    def begin_run(self) -> None:
        """Reset one-shot state before a single generate() call."""
        with self._lock:
            self._injected = False
            self._decode_steps = {}
            self._records = []

    @property
    def records(self) -> list[InjectionRecord]:
        return list(self._records)

    @property
    def injected(self) -> bool:
        return self._injected

    def cache_dims(self) -> dict:
        return {
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "block_size": self.block_size,
        }

    # -- core --------------------------------------------------------------
    def observe_cache(self, kv_cache) -> None:
        """Record cache geometry from a live tensor (always, even disabled)."""
        if kv_cache is None or kv_cache.numel() == 0:
            return
        # [2, num_blocks, block_size, num_kv_heads, head_size]
        if kv_cache.dim() == 5 and self.block_size is None:
            self.block_size = int(kv_cache.shape[2])
            self.num_kv_heads = int(kv_cache.shape[3])
            self.head_size = int(kv_cache.shape[4])

    def maybe_inject(self, layer, kv_cache, attn_metadata) -> None:
        """Post-hook: called after the original forward for each layer."""
        self.observe_cache(kv_cache)
        if not self._enabled or self._injected or attn_metadata is None:
            return
        if kv_cache is None or kv_cache.dim() != 5:
            return

        layer_idx = self._layer_index(layer)
        if layer_idx is not None:
            self.num_layers_seen.add(layer_idx)
        cfg = self._config
        if cfg is None or layer_idx != cfg.layer:
            return

        num_actual = int(attn_metadata.num_actual_tokens)
        is_prefill = num_actual > 1

        if cfg.phase == "prefill":
            if not is_prefill:
                return
            token_index = (cfg.token_index
                           if cfg.token_index is not None else num_actual // 2)
            token_index = max(0, min(token_index, num_actual - 1))
            slot = int(attn_metadata.slot_mapping[token_index].item())
            self._flip(kv_cache, cfg, slot, decode_step=None)
        elif cfg.phase == "generated":
            if is_prefill:
                return
            step = self._decode_steps.get(cfg.layer, 0) + 1
            self._decode_steps[cfg.layer] = step
            if step != cfg.injection_step:
                return
            # Decode step writes exactly one token at slot_mapping[0].
            slot = int(attn_metadata.slot_mapping[0].item())
            self._flip(kv_cache, cfg, slot, decode_step=step)
        else:
            raise ValueError(f"Unknown phase {cfg.phase!r}")

    # -- helpers -----------------------------------------------------------
    def _layer_index(self, layer) -> Optional[int]:
        name = getattr(layer, "layer_name", None)
        if not name:
            return None
        m = _LAYER_RE.search(name)
        return int(m.group(1)) if m else None

    def _flip(self, kv_cache, cfg: FaultConfig, slot: int,
              decode_step: Optional[int]) -> None:
        import torch

        block_size = int(kv_cache.shape[2])
        block_id = slot // block_size
        token_offset = slot % block_size
        kv_idx = cfg.kv_index()

        # Basic (integer) indexing on the leading dims returns a view; the
        # innermost head_size row is contiguous, so a bitcast view is valid.
        row = kv_cache[kv_idx, block_id, token_offset, cfg.kv_head]
        row_int = row.view(torch.int16)

        bit = int(cfg.bit)
        if not 0 <= bit <= 15:
            raise ValueError(f"bit must be in [0, 15], got {bit}")
        mask = 1 << bit
        mask_i16 = mask - 0x10000 if mask >= 0x8000 else mask

        old_bits = int(row_int[cfg.dim].item()) & 0xFFFF
        old_value = float(row[cfg.dim].item())
        row_int[cfg.dim] = row_int[cfg.dim] ^ mask_i16
        new_bits = int(row_int[cfg.dim].item()) & 0xFFFF
        new_value = float(row[cfg.dim].item())

        record = InjectionRecord(
            layer=cfg.layer,
            kv=cfg.kv.upper(),
            phase=cfg.phase,
            block_id=block_id,
            token_offset=token_offset,
            slot=slot,
            kv_head=cfg.kv_head,
            head_dim=cfg.dim,
            bit=bit,
            old_bits=old_bits,
            new_bits=new_bits,
            old_value=old_value,
            new_value=new_value,
            decode_step=decode_step,
        )
        self._records.append(record)
        self._injected = True


_INJECTOR = Injector()
_INSTALLED = False


def get_injector() -> Injector:
    return _INJECTOR


def install() -> Injector:
    """Wrap FlashAttentionImpl.forward with the injection post-hook (once)."""
    global _INSTALLED
    if _INSTALLED:
        return _INJECTOR

    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    original_forward = FlashAttentionImpl.forward

    def patched_forward(self, layer, query, key, value, kv_cache,
                        attn_metadata, output=None):
        out = original_forward(self, layer, query, key, value, kv_cache,
                               attn_metadata, output)
        try:
            _INJECTOR.maybe_inject(layer, kv_cache, attn_metadata)
        except Exception as exc:  # never let injection crash inference
            import logging
            logging.getLogger(__name__).warning(
                "KV fault injection hook failed: %s", exc)
        return out

    patched_forward._kv_fault_wrapped = True  # type: ignore[attr-defined]
    FlashAttentionImpl.forward = patched_forward
    _INSTALLED = True
    return _INJECTOR
