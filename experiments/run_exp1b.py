# SPDX-License-Identifier: Apache-2.0
"""Experiment 1B: does a KV-cache error propagate/amplify across decode steps?

See exp_design.md section 8. The model is loaded once; a clean baseline is
produced per prompt in-process, then every fault trial re-runs the same prompt
and injects a single BF16 bit flip into the *generated* V-cache row written at a
chosen decode step. Because the corrupted V row is re-read by every subsequent
decode step, an earlier injection has a larger downstream window -- the whole
point of the experiment.

Design defaults (section 8.2):

* Fault target : generated KV cache (phase="generated")
* K_or_V       : V-only
* Layer        : middle layer
* Bit position : 14 (BF16 exponent MSB)
* Injection step : 16 / 64 / 96
* Target row   : the just-written V row of the injected decode step
* Head/dim     : 3 random (kv_head, dim) samples per prompt

Recommended grid (section 8.2)::

    30 prompts x 3 injection steps x 3 head-dim samples = 270 fault runs

Only the suffix *after* the injection step is compared (section 8.3 step 7).

Usage (minimal version, from the repo root)::

    conda activate vllm0.8.5
    cd /opt/data/data/workspace-vllm
    PYTHONUNBUFFERED=1 python experiments/run_exp1b.py --min

Full grid::

    python experiments/run_exp1b.py \
        --num-prompts 30 --injection-steps 16 64 96 --head-dim-samples 3
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
    p.add_argument("--kv", default="V", choices=["K", "V"],
                   help="Design 1B is V-only.")
    p.add_argument("--bit", type=int, default=14,
                   help="BF16 bit position (design 1B uses 14).")
    p.add_argument("--injection-steps", nargs="+", type=int,
                   default=[16, 64, 96])
    p.add_argument("--head-dim-samples", type=int, default=3)
    p.add_argument("--min", action="store_true",
                   help="Minimal preset: 20 prompts, 2 head-dim samples.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "experiments" / "results" / "exp1b")
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
          f"{len(prompts)} prompts | kv {args.kv} | bit {args.bit} | "
          f"injection steps {args.injection_steps} | "
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

    # Guard: an injection step must fall within the generated decode range.
    valid_steps = [s for s in args.injection_steps if 1 <= s < args.max_tokens]
    skipped = sorted(set(args.injection_steps) - set(valid_steps))
    if skipped:
        print(f"WARNING: skipping out-of-range injection steps {skipped} "
              f"(must be in [1, {args.max_tokens - 1}]).")

    # ---- Phase 2: fault trials ------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.out_dir / "trials.jsonl"
    trials: list[dict] = []
    missed = 0

    with trials_path.open("w", encoding="utf-8") as fh:
        for item in prompts:
            pid, text = item["id"], item["text"]
            formatted = formatted_prompts[pid]
            clean_ids = baselines[pid]["clean_output_token_ids"]

            # Stable per-prompt seed (see run_exp1a.py): built-in hash() is
            # salted per process, so use a deterministic digest instead.
            pid_hash = int(hashlib.md5(pid.encode("utf-8")).hexdigest(), 16)
            rng = random.Random(args.seed + pid_hash % 100000)
            head_dim_samples = [
                (rng.randrange(num_kv_heads), rng.randrange(head_size))
                for _ in range(args.head_dim_samples)
            ]

            for step in valid_steps:
                for (kv_head, dim) in head_dim_samples:
                    cfg = FaultConfig(
                        layer=target_layer, kv=args.kv, bit=args.bit,
                        kv_head=kv_head, dim=dim, phase="generated",
                        injection_step=step,
                    )
                    injector.set_config(cfg)
                    injector.begin_run()
                    t0 = time.perf_counter()
                    fault_ids, fault_text = run_once(formatted)
                    secs = time.perf_counter() - t0
                    recs = injector.records
                    rec = recs[0].to_dict() if recs else None
                    if rec is None:
                        missed += 1
                    new_val = rec["new_value"] if rec else None
                    value_naninf = bool(
                        new_val is not None
                        and (math.isnan(new_val) or math.isinf(new_val)))

                    m = compute_trial_metrics(clean_ids, fault_ids, fault_text,
                                              injection_step=step)
                    suffix_changed = (list(clean_ids)[step:]
                                      != list(fault_ids)[step:])
                    remaining = max(0, args.max_tokens - step)
                    row = {
                        "prompt_id": pid,
                        "model": str(model_path),
                        "dtype": "bfloat16",
                        "vllm_version": vllm.__version__,
                        "git_commit": git_info["commit"],
                        "git_dirty": git_info["dirty"],
                        "max_tokens": args.max_tokens,
                        "seed": args.seed,
                        "target_type": "generated",
                        "injection_step": step,
                        "remaining_tokens": remaining,
                        "layer_id": target_layer,
                        "kv": args.kv,
                        "bit": args.bit,
                        "kv_head": kv_head,
                        "head_dim": dim,
                        "injection_record": rec,
                        "injected": injector.injected,
                        "value_naninf": value_naninf,
                        "suffix_changed": suffix_changed,
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
                    print(f"[fault] {pid} step{step} {args.kv} bit{args.bit} "
                          f"h{kv_head}d{dim}: suffTDR={m.suffix_tdr:.3f} "
                          f"fda={m.first_divergence_after_injection} "
                          f"collapse={m.collapse} {flip} ({secs:.1f}s)")

    injector.disable()
    if missed:
        print(f"WARNING: {missed} trials never triggered an injection.")

    # ---- Phase 3: aggregate ---------------------------------------------
    (args.out_dir / "baselines.json").write_text(
        json.dumps(list(baselines.values()), ensure_ascii=False, indent=2),
        encoding="utf-8")
    write_summary(trials, args.out_dir, target_layer, args.kv, args.bit)
    print(f"\nDone. {len(trials)} trials -> {args.out_dir}")


def write_summary(trials: list[dict], out_dir: Path, layer: int,
                  kv: str, bit: int) -> None:
    from collections import defaultdict

    groups: dict[int, list[dict]] = defaultdict(list)
    for t in trials:
        groups[t["injection_step"]].append(t)

    def agg(rows: list[dict]) -> dict:
        n = len(rows)
        remaining = rows[0]["remaining_tokens"] if rows else 0
        tcr = sum(1 for r in rows if r["suffix_changed"]) / n
        post_tdr = sum(r["metrics"]["suffix_tdr"] or 0.0 for r in rows) / n
        collapse = sum(1 for r in rows if r["metrics"]["collapse"]) / n
        fdas = [r["metrics"]["first_divergence_after_injection"] for r in rows
                if r["metrics"]["first_divergence_after_injection"] is not None]
        mean_fda = sum(fdas) / len(fdas) if fdas else None
        return {"n": n, "remaining": remaining, "tcr": tcr,
                "post_tdr": post_tdr, "mean_fda": mean_fda,
                "collapse": collapse}

    lines = [
        f"# Experiment 1B summary (target layer {layer}, {kv}-only, bit {bit})",
        "",
        "| Injection Step | Remaining Tokens | N | TCR | Post-injection TDR | "
        "First Divergence After Injection | Collapse Rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv = ["injection_step,remaining_tokens,n,tcr,post_injection_tdr,"
           "first_divergence_after_injection,collapse_rate"]
    for step in sorted(groups):
        a = agg(groups[step])
        fda_md = f"{a['mean_fda']:.2f}" if a["mean_fda"] is not None else "-"
        fda_csv = f"{a['mean_fda']:.4f}" if a["mean_fda"] is not None else ""
        lines.append(
            f"| {step} | {a['remaining']} | {a['n']} | {a['tcr']:.3f} | "
            f"{a['post_tdr']:.3f} | {fda_md} | {a['collapse']:.3f} |")
        csv.append(f"{step},{a['remaining']},{a['n']},{a['tcr']:.4f},"
                   f"{a['post_tdr']:.4f},{fda_csv},{a['collapse']:.4f}")

    (out_dir / "summary_1b.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    (out_dir / "summary_1b.csv").write_text("\n".join(csv) + "\n",
                                            encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
