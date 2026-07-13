# SPDX-License-Identifier: Apache-2.0
"""Controlled KV-cache fault injection for vLLM (experiment 1A/1B).

All logic lives under ``experiments/`` and installs itself into vLLM at
runtime by wrapping ``FlashAttentionImpl.forward``. No vLLM source file is
modified, which keeps the experiment repository clean.
"""

from experiments.fault_injection.kv_injector import (
    FaultConfig,
    InjectionRecord,
    get_injector,
    install,
)

__all__ = [
    "FaultConfig",
    "InjectionRecord",
    "get_injector",
    "install",
]
