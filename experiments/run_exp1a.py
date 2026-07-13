# SPDX-License-Identifier: Apache-2.0
"""Experiment 1A: does a single BF16 bit flip in the KV cache change output?

The model is loaded once; a clean baseline is produced per prompt in-process,
then every fault trial (K/V x bit x random head-dim samples) re-runs the same
prompt with one injected bit flip. See exp_design.md section 7.

Usage (minimal version, from the repo root)::

    conda activate vllm0.8.5
    cd /opt/data/data/workspace-vllm
    PYTHONUNBUFFERED=1 python experiments/run_exp1a.py --min

Full grid::

    python experiments/run_exp1a.py \
        --num-prompts 30 --bits 0 7 14 15 --head-dim-samples 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Keep the V1 engine in this process so the Python injection hook is active.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path("/opt/data/data/models/Qwen3-8B")
DEFAULT_PROMPTS = REPO_ROOT / "experiments" / "prompts" / "prompts_1a.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--num-prompts", type=int, default=None,
                   help="Limit number of prompts (default: all in file).")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer", type=int, default=None,
                   help="Target layer index (default: middle layer).")
    p.add_argument("--kv", nargs="+", default=["K", "V"], choices=["K", "V"])
    p.add_argument("--bits", nargs="+", type=int, default=[0, 7, 14, 15])
    p.add_argument("--head-dim-samples", type=int, default=3)
    p.add_argument("--min", action="store_true",
                   help="Minimal preset: 20 prompts, bits {0,14,15}, 2 samples.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "experiments" / "results" / "exp1a")
    return p.parse_args()


def load_prompts(path: Path, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data["prompts"] if isinstance(data, dict) else data
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def git_revision(repo_root: Path) -> dict:
    """Capture the source commit so every trial is reproducible (sec. 9)."""
    import subprocess

    def _run(args: list[str]) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            stderr=subprocess.DEVNULL).decode().strip()

    try:
        commit = _run(["rev-parse", "HEAD"])
    except Exception:
        return {"commit": "unknown", "dirty": None}
    try:
        dirty = bool(_run(["status", "--porcelain"]))
    except Exception:
        dirty = None
    return {"commit": commit, "dirty": dirty}


def main() -> None:
    args = parse_args()
    if args.min:
        args.bits = [0, 14, 15]
        args.head_dim_samples = 2
        if args.num_prompts is None:
            args.num_prompts = 20

    import torch  # noqa: F401
    import vllm
    from vllm import LLM, SamplingParams

    from experiments.fault_injection import FaultConfig, install
    from experiments.fault_injection.metrics import compute_trial_metrics

    injector = install()

    git_info = git_revision(REPO_ROOT)
    model_path = args.model.expanduser().resolve()
    prompts = load_prompts(args.prompts_file, args.num_prompts)
    print(f"vLLM {vllm.__version__} | git {git_info['commit'][:12]}"
          f"{'-dirty' if git_info['dirty'] else ''} | model {model_path} | "
          f"{len(prompts)} prompts | bits {args.bits} | kv {args.kv} | "
          f"head-dim samples {args.head_dim_samples}")

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
        ignore_eos=True,
    )

    def format_prompt(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def run_once(formatted: str):
        out = llm.generate([formatted], sampling, use_tqdm=False)[0]
        gen = out.outputs[0]
        return list(gen.token_ids), gen.text

    # ---- Phase 1: clean baselines (also discovers cache geometry) --------
    injector.disable()
    baselines: dict[str, dict] = {}
    formatted_prompts: dict[str, str] = {}
    for item in prompts:
        pid, text = item["id"], item["text"]
        formatted = format_prompt(text)
        formatted_prompts[pid] = formatted
        injector.begin_run()
        t0 = time.perf_counter()
        clean_ids, clean_text = run_once(formatted)
        baselines[pid] = {
            "prompt_id": pid,
            "prompt_token_count": len(tokenizer(formatted)["input_ids"]),
            "clean_output_token_ids": clean_ids,
            "clean_text": clean_text,
            "seconds": time.perf_counter() - t0,
        }
        print(f"[clean] {pid}: {len(clean_ids)} tokens "
              f"({baselines[pid]['seconds']:.1f}s)")

    dims = injector.cache_dims()
    num_kv_heads = dims["num_kv_heads"]
    head_size = dims["head_size"]
    num_layers = (max(injector.num_layers_seen) + 1
                  if injector.num_layers_seen else None)
    target_layer = (args.layer if args.layer is not None
                    else (num_layers // 2 if num_layers else 18))
    print(f"cache dims: num_kv_heads={num_kv_heads} head_size={head_size} "
          f"num_layers={num_layers} -> target_layer={target_layer}")

    if num_kv_heads is None or head_size is None:
        raise RuntimeError("Failed to discover KV cache geometry from a run.")

    # ---- Phase 2: fault trials ------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.out_dir / "trials.jsonl"
    trials: list[dict] = []

    with trials_path.open("w", encoding="utf-8") as fh:
        for item in prompts:
            pid, text = item["id"], item["text"]
            formatted = formatted_prompts[pid]
            clean_ids = baselines[pid]["clean_output_token_ids"]

            # Stable per-prompt seed: Python's built-in hash() is salted per
            # process (PYTHONHASHSEED), so use a deterministic digest instead
            # to keep the head-dim sampling reproducible across runs/machines.
            pid_hash = int(hashlib.md5(pid.encode("utf-8")).hexdigest(), 16)
            rng = random.Random(args.seed + pid_hash % 100000)
            head_dim_samples = [
                (rng.randrange(num_kv_heads), rng.randrange(head_size))
                for _ in range(args.head_dim_samples)
            ]

            for kv in args.kv:
                for bit in args.bits:
                    for (kv_head, dim) in head_dim_samples:
                        cfg = FaultConfig(
                            layer=target_layer, kv=kv, bit=bit,
                            kv_head=kv_head, dim=dim, phase="prefill",
                        )
                        injector.set_config(cfg)
                        injector.begin_run()
                        t0 = time.perf_counter()
                        fault_ids, fault_text = run_once(formatted)
                        secs = time.perf_counter() - t0
                        recs = injector.records
                        rec = recs[0].to_dict() if recs else None
                        new_val = rec["new_value"] if rec else None
                        value_naninf = bool(
                            new_val is not None
                            and (math.isnan(new_val) or math.isinf(new_val)))

                        m = compute_trial_metrics(clean_ids, fault_ids,
                                                  fault_text)
                        row = {
                            "prompt_id": pid,
                            "model": str(model_path),
                            "dtype": "bfloat16",
                            "vllm_version": vllm.__version__,
                            "git_commit": git_info["commit"],
                            "git_dirty": git_info["dirty"],
                            "max_tokens": args.max_tokens,
                            "seed": args.seed,
                            "target_type": "prefill",
                            "layer_id": target_layer,
                            "kv": kv,
                            "bit": bit,
                            "kv_head": kv_head,
                            "head_dim": dim,
                            "injection_record": rec,
                            "injected": injector.injected,
                            "value_naninf": value_naninf,
                            "clean_output_token_ids": clean_ids,
                            "fault_output_token_ids": fault_ids,
                            "fault_text": fault_text,
                            "metrics": m.to_dict(),
                            "seconds": secs,
                        }
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                        trials.append(row)
                        flip = (f"{rec['old_bits']:016b}->{rec['new_bits']:016b}"
                                if rec else "NO-INJECT")
                        print(f"[fault] {pid} {kv} bit{bit} "
                              f"h{kv_head}d{dim}: TCR={m.tcr} "
                              f"TDR={m.tdr:.3f} fds={m.first_divergence_step} "
                              f"collapse={m.collapse} {flip} ({secs:.1f}s)")

    injector.disable()

    # ---- Phase 3: aggregate ---------------------------------------------
    (args.out_dir / "baselines.json").write_text(
        json.dumps(list(baselines.values()), ensure_ascii=False, indent=2),
        encoding="utf-8")
    write_summary(trials, args.out_dir, target_layer)
    print(f"\nDone. {len(trials)} trials -> {args.out_dir}")


def write_summary(trials: list[dict], out_dir: Path, layer: int) -> None:
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        groups[(t["kv"], t["bit"])].append(t)

    def agg(rows: list[dict]) -> dict:
        n = len(rows)
        tcr = sum(1 for r in rows if r["metrics"]["tcr"]) / n
        tdr = sum(r["metrics"]["tdr"] for r in rows) / n
        collapse = sum(1 for r in rows if r["metrics"]["collapse"]) / n
        rouge = sum(r["metrics"]["rouge_l"] for r in rows) / n
        return {"n": n, "tcr": tcr, "tdr": tdr, "collapse": collapse,
                "rouge_l": rouge}

    lines = [
        f"# Experiment 1A summary (target layer {layer})",
        "",
        "| Target | Bit | N | TCR | Mean TDR | Collapse Rate | Mean ROUGE-L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    csv = ["target,bit,n,tcr,mean_tdr,collapse_rate,mean_rouge_l"]
    for key in sorted(groups):
        kv, bit = key
        a = agg(groups[key])
        lines.append(
            f"| {kv} | {bit} | {a['n']} | {a['tcr']:.3f} | {a['tdr']:.3f} | "
            f"{a['collapse']:.3f} | {a['rouge_l']:.3f} |")
        csv.append(f"{kv},{bit},{a['n']},{a['tcr']:.4f},{a['tdr']:.4f},"
                   f"{a['collapse']:.4f},{a['rouge_l']:.4f}")

    (out_dir / "summary_1a.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    (out_dir / "summary_1a.csv").write_text("\n".join(csv) + "\n",
                                            encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
