# SPDX-License-Identifier: Apache-2.0
"""Step 2 sanity check: prove a single BF16 bit flip reaches the GPU KV cache.

Runs one clean generation and one fault generation on the same prompt, prints
the recorded old/new bits and values, and reports whether the output diverged.

    conda activate vllm0.8.5
    cd /opt/data/data/workspace-vllm
    PYTHONUNBUFFERED=1 python experiments/verify_injection.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path("/opt/data/data/models/Qwen3-8B")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--prompt", default="请用简洁的语言介绍一下人工智能。")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--kv", default="V", choices=["K", "V"])
    p.add_argument("--bit", type=int, default=14)
    p.add_argument("--kv-head", type=int, default=3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    from experiments.fault_injection import FaultConfig, install

    injector = install()
    print(f"vLLM {vllm.__version__}")

    llm = LLM(
        model=str(args.model.expanduser().resolve()),
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0,
                              top_p=1.0, top_k=-1, seed=args.seed,
                              ignore_eos=True)

    def run():
        out = llm.generate([formatted], sampling, use_tqdm=False)[0].outputs[0]
        return list(out.token_ids), out.text

    injector.disable()
    injector.begin_run()
    clean_ids, clean_text = run()

    injector.set_config(FaultConfig(layer=args.layer, kv=args.kv, bit=args.bit,
                                    kv_head=args.kv_head, dim=args.dim,
                                    phase="prefill"))
    injector.begin_run()
    fault_ids, fault_text = run()
    injector.disable()

    print(f"\ncache dims: {injector.cache_dims()}")
    if not injector.records:
        print("!! No injection happened. Check layer/head/dim against dims.")
        sys.exit(1)

    rec = injector.records[0]
    print("\n=== Injection record ===")
    print(f"layer={rec.layer} kv={rec.kv} phase={rec.phase} "
          f"slot={rec.slot} block={rec.block_id} offset={rec.token_offset} "
          f"kv_head={rec.kv_head} dim={rec.head_dim} bit={rec.bit}")
    print(f"bits: {rec.old_bits:016b} -> {rec.new_bits:016b}")
    print(f"value: {rec.old_value} -> {rec.new_value}")
    assert rec.old_bits != rec.new_bits, "bits unchanged -> flip failed"
    assert (rec.old_bits ^ rec.new_bits) == (1 << rec.bit), \
        "exactly one bit (the target) must change"

    diverged = clean_ids != fault_ids
    fds = next((i for i, (a, b) in enumerate(zip(clean_ids, fault_ids))
                if a != b), None)
    print("\n=== Output comparison ===")
    print(f"clean : {clean_text!r}")
    print(f"fault : {fault_text!r}")
    print(f"diverged={diverged} first_divergence_step={fds}")
    print("\nVERIFY_INJECTION=OK")


if __name__ == "__main__":
    main()
