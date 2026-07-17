# SPDX-License-Identifier: Apache-2.0
"""Experiment 2: KV-cache bit-flip vs multiple-choice QA accuracy (OpenBookQA).

This extends the single-bit BF16 KV-cache fault injection (see ``run_exp1a.py``)
from "did the raw token stream change?" to "did the model still answer the
question correctly?".

Dataset
-------
OpenBookQA ``main`` **test** split (500 four-way multiple-choice questions),
https://huggingface.co/datasets/allenai/openbookqa. A local copy of the parquet
is cached under ``experiments/data/openbookqa_main_test.parquet``.

Two answering schemes (the two the query asks for)
--------------------------------------------------
* ``direct`` -- the model must emit ONLY the answer letter (A/B/C/D), no
  reasoning (Qwen3 ``enable_thinking=False``, small ``max_tokens``).
* ``cot``    -- the model reasons step by step first, then states a final answer
  letter (Qwen3 ``enable_thinking=True``, larger ``max_tokens``).

Fault conditions (per question, per scheme)
-------------------------------------------
* ``clean`` -- no injection (baseline accuracy).
* ``bit{0,7,14,15}`` -- one BF16 bit flipped in the target token's stored KV,
  right after prefill, so every decode step reads the corrupted value. BF16 bit
  layout: 0-6 mantissa, 7-14 exponent, 15 sign.

Target: V-cache, middle layer, one deterministic (kv_head, head_dim) per
question, middle prompt token. V-cache and the middle layer match exp 1A/1B
where V is the most fragile tensor.

The accuracy of each (scheme, condition) pair over all 500 questions is the
headline metric; per-trial rows are written to ``trials_<scheme>.jsonl``.

Usage (from repo root, inside the vllm0.8.5 env)::

    python experiments/run_exp2.py --scheme direct
    python experiments/run_exp2.py --scheme cot
    python experiments/run_exp2.py --aggregate-only   # merge + summarise

    python experiments/run_exp2.py --scheme direct --limit 8   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

# Keep the V1 engine in this process so the Python injection hook is active.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path("/opt/data/data/models/Qwen3-8B")
DEFAULT_DATA = REPO_ROOT / "experiments" / "data" / "openbookqa_main_test.parquet"
DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "exp2"

CHOICE_LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    p.add_argument("--scheme", choices=["direct", "cot"], default=None,
                   help="Answering scheme to run (omit with --aggregate-only).")
    p.add_argument("--bits", nargs="+", type=int, default=[0, 7, 14, 15])
    p.add_argument("--kv", default="V", choices=["K", "V"])
    p.add_argument("--layer", type=int, default=None,
                   help="Target layer index (default: middle layer).")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of questions (smoke tests).")
    p.add_argument("--start", type=int, default=0,
                   help="First question index (inclusive) for sharding.")
    p.add_argument("--end", type=int, default=None,
                   help="Last question index (exclusive) for sharding.")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix for the trials file: trials_<scheme>_<tag>.jsonl")
    p.add_argument("--max-tokens-direct", type=int, default=8)
    p.add_argument("--max-tokens-cot", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=1280)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024,
                   help="Smaller value shrinks the activation-profiling peak so "
                        "the KV cache fits on a shared GPU.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip inference; merge existing trials_*.jsonl -> summary.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_questions(path: Path, limit: int | None,
                   start: int = 0, end: int | None = None) -> list[dict]:
    """Read the OpenBookQA parquet with pyarrow (no `datasets` dependency).

    ``start``/``end`` select a contiguous shard of the (fixed-order) test set
    so several GPUs can each own a disjoint slice; ``limit`` truncates for
    smoke tests and is applied after the shard slice.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = table.to_pydict()
    n = table.num_rows
    questions: list[dict] = []
    for i in range(n):
        choices = cols["choices"][i]
        labels = list(choices["label"])
        texts = list(choices["text"])
        questions.append({
            "id": cols["id"][i],
            "stem": cols["question_stem"][i],
            "labels": labels,
            "texts": texts,
            "answer": cols["answerKey"][i],
        })
    questions = questions[start:end]
    if limit is not None:
        questions = questions[:limit]
    return questions


def render_options(q: dict) -> str:
    return "\n".join(f"{lab}. {txt}"
                     for lab, txt in zip(q["labels"], q["texts"]))


def build_user_prompt(q: dict, scheme: str) -> str:
    options = render_options(q)
    if scheme == "direct":
        return (
            "Answer the following multiple-choice question. "
            "Reply with only the single letter (A, B, C, or D) of the correct "
            "option and nothing else.\n\n"
            f"Question: {q['stem']}\n{options}\n\nAnswer:"
        )
    # cot
    return (
        "Answer the following multiple-choice question. Think step by step, "
        "then on the final line write exactly 'Answer: X' where X is one of "
        "A, B, C, or D.\n\n"
        f"Question: {q['stem']}\n{options}"
    )


# --------------------------------------------------------------------------- #
# Answer parsing
# --------------------------------------------------------------------------- #
_ANSWER_RE = re.compile(r"answer\s*(?:is|:|=)?\s*\(?\*{0,2}([ABCD])\b",
                        re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{\s*\(?([ABCD])\)?\s*\}")
_STANDALONE_RE = re.compile(r"\b([ABCD])\b")


def parse_answer(text: str, scheme: str) -> str | None:
    """Extract the predicted A/B/C/D letter, or None if unparseable."""
    if not text:
        return None
    t = text.strip()

    m = _BOXED_RE.findall(t)
    if m:
        return m[-1].upper()

    m = _ANSWER_RE.findall(t)
    if m:
        return m[-1].upper()

    if scheme == "direct":
        # First standalone letter for the terse scheme.
        m = _STANDALONE_RE.findall(t)
        if m:
            return m[0].upper()
        # Fall back to first A-D char anywhere.
        for ch in t:
            if ch.upper() in CHOICE_LETTERS:
                return ch.upper()
        return None

    # cot: take the last standalone letter as the conclusion.
    m = _STANDALONE_RE.findall(t)
    if m:
        return m[-1].upper()
    return None


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def git_revision(repo_root: Path) -> dict:
    import subprocess

    def _run(a: list[str]) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *a],
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


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run_scheme(args: argparse.Namespace) -> None:
    import vllm
    from vllm import LLM, SamplingParams

    from experiments.fault_injection import FaultConfig, install

    injector = install()
    git_info = git_revision(REPO_ROOT)
    model_path = args.model.expanduser().resolve()
    questions = load_questions(args.data_file, args.limit, args.start, args.end)
    scheme = args.scheme
    max_tokens = (args.max_tokens_direct if scheme == "direct"
                  else args.max_tokens_cot)
    enable_thinking = scheme == "cot"

    print(f"vLLM {vllm.__version__} | git {git_info['commit'][:12]}"
          f"{'-dirty' if git_info['dirty'] else ''} | model {model_path}\n"
          f"scheme={scheme} thinking={enable_thinking} max_tokens={max_tokens} "
          f"| {len(questions)} questions | kv {args.kv} | bits {args.bits}",
          flush=True)

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
        ignore_eos=False,
    )

    def format_prompt(user_text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def run_once(formatted: str):
        out = llm.generate([formatted], sampling, use_tqdm=False)[0]
        gen = out.outputs[0]
        return list(gen.token_ids), gen.text

    # Discover cache geometry with a single warmup generation.
    injector.disable()
    warm = format_prompt(build_user_prompt(questions[0], scheme))
    injector.begin_run()
    run_once(warm)
    dims = injector.cache_dims()
    num_kv_heads = dims["num_kv_heads"]
    head_size = dims["head_size"]
    num_layers = (max(injector.num_layers_seen) + 1
                  if injector.num_layers_seen else None)
    target_layer = (args.layer if args.layer is not None
                    else (num_layers // 2 if num_layers else 18))
    print(f"cache dims: num_kv_heads={num_kv_heads} head_size={head_size} "
          f"num_layers={num_layers} -> target_layer={target_layer}", flush=True)
    if num_kv_heads is None or head_size is None:
        raise RuntimeError("Failed to discover KV cache geometry.")

    conditions = ["clean"] + [f"bit{b}" for b in args.bits]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    trials_path = args.out_dir / f"trials_{scheme}{tag}.jsonl"

    # correct[condition] counters
    correct = {c: 0 for c in conditions}
    total = {c: 0 for c in conditions}
    unparsed = {c: 0 for c in conditions}

    with trials_path.open("w", encoding="utf-8") as fh:
        for qi, q in enumerate(questions):
            user_text = build_user_prompt(q, scheme)
            formatted = format_prompt(user_text)
            gold = (q["answer"] or "").strip().upper()

            # Deterministic per-question (kv_head, dim) target.
            qid_hash = int(hashlib.md5(q["id"].encode("utf-8")).hexdigest(), 16)
            rng = random.Random(args.seed + qid_hash % 100000)
            kv_head = rng.randrange(num_kv_heads)
            dim = rng.randrange(head_size)

            for cond in conditions:
                if cond == "clean":
                    injector.disable()
                    rec = None
                else:
                    bit = int(cond[3:])
                    cfg = FaultConfig(
                        layer=target_layer, kv=args.kv, bit=bit,
                        kv_head=kv_head, dim=dim, phase="prefill",
                    )
                    injector.set_config(cfg)
                injector.begin_run()
                t0 = time.perf_counter()
                out_ids, out_text = run_once(formatted)
                secs = time.perf_counter() - t0
                if cond != "clean":
                    recs = injector.records
                    rec = recs[0].to_dict() if recs else None
                injector.disable()

                pred = parse_answer(out_text, scheme)
                is_correct = pred is not None and pred == gold
                total[cond] += 1
                if pred is None:
                    unparsed[cond] += 1
                if is_correct:
                    correct[cond] += 1

                new_val = rec["new_value"] if rec else None
                value_naninf = bool(
                    new_val is not None
                    and (math.isnan(new_val) or math.isinf(new_val)))

                row = {
                    "question_id": q["id"],
                    "scheme": scheme,
                    "condition": cond,
                    "model": str(model_path),
                    "dtype": "bfloat16",
                    "vllm_version": vllm.__version__,
                    "git_commit": git_info["commit"],
                    "git_dirty": git_info["dirty"],
                    "layer_id": target_layer,
                    "kv": args.kv,
                    "kv_head": kv_head,
                    "head_dim": dim,
                    "max_tokens": max_tokens,
                    "seed": args.seed,
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                    "injection_record": rec,
                    "value_naninf": value_naninf,
                    "output_token_count": len(out_ids),
                    "output_text": out_text,
                    "seconds": secs,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()

            if (qi + 1) % 20 == 0 or (qi + 1) == len(questions):
                acc_str = " ".join(
                    f"{c}={correct[c]}/{total[c]}" for c in conditions)
                print(f"[{scheme}] {qi + 1}/{len(questions)} | {acc_str}",
                      flush=True)

    injector.disable()
    print(f"\n[{scheme}] done -> {trials_path}", flush=True)
    for c in conditions:
        acc = correct[c] / total[c] if total[c] else 0.0
        print(f"  {c:8s} acc={acc:.4f} ({correct[c]}/{total[c]}) "
              f"unparsed={unparsed[c]}", flush=True)

    write_summary(args.out_dir)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def write_summary(out_dir: Path) -> None:
    """Merge all trials_<scheme>.jsonl into a scheme x condition accuracy table."""
    from collections import defaultdict

    # counts[scheme][condition] = [correct, total, unparsed]
    counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0, 0]))
    conditions_seen: dict[str, None] = {}
    meta = {"layer_id": None, "kv": None, "git_commit": None,
            "vllm_version": None}

    for tf in sorted(out_dir.glob("trials_*.jsonl")):
        with tf.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    # Another shard may still be appending its last line.
                    continue
                sc, cond = r["scheme"], r["condition"]
                conditions_seen[cond] = None
                c = counts[sc][cond]
                c[1] += 1
                if r["correct"]:
                    c[0] += 1
                if r["pred"] is None:
                    c[2] += 1
                meta["layer_id"] = r.get("layer_id")
                meta["kv"] = r.get("kv")
                meta["git_commit"] = r.get("git_commit")
                meta["vllm_version"] = r.get("vllm_version")

    if not counts:
        print("No trials_*.jsonl found to aggregate.")
        return

    def cond_sort_key(c: str) -> tuple:
        if c == "clean":
            return (0, -1)
        return (1, int(c[3:])) if c.startswith("bit") else (2, 0)

    conditions = sorted(conditions_seen, key=cond_sort_key)
    schemes = sorted(counts)

    lines = [
        "# Experiment 2 summary -- OpenBookQA accuracy under KV-cache bit flips",
        "",
        f"- Model / vLLM: Qwen3-8B (bf16), vLLM {meta['vllm_version']}",
        f"- Target: {meta['kv']}-cache, layer {meta['layer_id']}, "
        "prefill token, 1 (kv_head, head_dim) per question",
        f"- git commit: {meta['git_commit']}",
        "",
        "## Accuracy (correct / total)",
        "",
    ]
    header = "| Scheme | " + " | ".join(conditions) + " |"
    sep = "|---|" + "|".join(["---:"] * len(conditions)) + "|"
    lines += [header, sep]

    csv = ["scheme,condition,correct,total,accuracy,unparsed"]
    for sc in schemes:
        cells = []
        for cond in conditions:
            cor, tot, unp = counts[sc].get(cond, [0, 0, 0])
            acc = cor / tot if tot else 0.0
            cells.append(f"{acc:.3f} ({cor}/{tot})" if tot else "-")
            if tot:
                csv.append(f"{sc},{cond},{cor},{tot},{acc:.4f},{unp}")
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")

    # Accuracy drop vs clean, per scheme.
    lines += ["", "## Accuracy drop vs clean baseline (percentage points)", "",
              "| Scheme | " + " | ".join(c for c in conditions if c != "clean")
              + " |",
              "|---|" + "|".join(["---:"] * (len(conditions) - 1)) + "|"]
    for sc in schemes:
        base = counts[sc].get("clean", [0, 0, 0])
        base_acc = base[0] / base[1] if base[1] else 0.0
        cells = []
        for cond in conditions:
            if cond == "clean":
                continue
            cor, tot, _ = counts[sc].get(cond, [0, 0, 0])
            acc = cor / tot if tot else 0.0
            cells.append(f"{(base_acc - acc) * 100:+.1f}" if tot else "-")
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")

    (out_dir / "summary_2.md").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")
    (out_dir / "summary_2.csv").write_text("\n".join(csv) + "\n",
                                           encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_dir/'summary_2.md'} and {out_dir/'summary_2.csv'}")


def main() -> None:
    args = parse_args()
    if args.aggregate_only:
        write_summary(args.out_dir)
        return
    if args.scheme is None:
        raise SystemExit("--scheme is required unless --aggregate-only is set.")
    run_scheme(args)


if __name__ == "__main__":
    main()
