# Experiment 2 — KV-cache bit flips vs OpenBookQA answer accuracy

Experiments 1A/1B (see [`exp_design.md`](exp_design.md)) showed that a single
BF16 bit flip in the stored KV cache can change or collapse the *raw token
stream*. Experiment 2 asks the downstream question a user actually cares about:

> **Does a single KV-cache bit flip make the model answer a question wrong, and
> does it matter whether the model answers directly or reasons first?**

We measure four-way multiple-choice accuracy on the full OpenBookQA test set
under a controlled single-bit fault, for two answering schemes and four BF16 bit
positions.

## 1. Dataset

- **OpenBookQA**, `main` config, **test** split — 500 four-way multiple-choice
  elementary-science questions.
  <https://huggingface.co/datasets/allenai/openbookqa>
- Cached locally at
  [`experiments/data/openbookqa_main_test.parquet`](experiments/data/openbookqa_main_test.parquet)
  (downloaded once via the `hf-mirror.com` endpoint; read with `pyarrow`, so the
  heavy `datasets` package is *not* a runtime dependency).
- Each item: `id`, `question_stem`, `choices{label[], text[]}`, `answerKey`.

## 2. Two answering schemes

Both use Qwen3-8B with the chat template and greedy decoding
(`temperature=0`, `top_p=1`, `top_k=-1`, `seed=42`, `ignore_eos=False`).

| Scheme | `enable_thinking` | `max_tokens` | Prompt asks for | Answer parsed from |
|---|---|---:|---|---|
| `direct` | `False` | 8 | *only* the letter `A/B/C/D` | first `A–D` token |
| `cot` | `True` | 1024 | step-by-step reasoning, then a final `Answer: X` line | last `Answer: X` / `\boxed{X}` / last standalone `A–D` |

The answer parser (`parse_answer` in
[`experiments/run_exp2.py`](experiments/run_exp2.py)) prefers an explicit
`\boxed{X}` or `answer: X` marker, then falls back to the first (direct) or last
(cot) standalone `A–D` letter. Unparseable outputs count as wrong and are logged
via the `unparsed` counter.

## 3. Fault model

We reuse the exp 1A injector unchanged
([`experiments/fault_injection/kv_injector.py`](experiments/fault_injection/kv_injector.py)):
a post-hook on `FlashAttentionImpl.forward` flips **one BF16 bit** of one stored
KV element right after prefill, so the corrupted value persists in the paged KV
cache and is re-read by every subsequent decode step — the "stored KV suffered a
soft error before generation" semantics.

Per question, per scheme, we run **five conditions**:

| Condition | Meaning |
|---|---|
| `clean` | no injection (baseline accuracy) |
| `bit0`  | flip BF16 bit 0  — mantissa LSB |
| `bit7`  | flip BF16 bit 7  — mantissa MSB / exponent LSB boundary |
| `bit14` | flip BF16 bit 14 — exponent MSB (catastrophic) |
| `bit15` | flip BF16 bit 15 — sign bit |

BF16 layout: bit 0–6 mantissa, 7–14 exponent, 15 sign.

### Injection target

- **Tensor:** V-cache (V was the most fragile tensor in exp 1A/1B).
- **Layer:** middle layer (18 of 36).
- **Position:** the middle prompt token, injected in the prefill phase.
- **(kv_head, head_dim):** one deterministic sample per question, seeded by
  `md5(question_id)` so the target is fixed and reproducible.

This is **one flip per (question, scheme, bit)** — not the multi-sample sweep of
exp 1A — because here the unit of interest is a single question's correctness.

## 4. What to expect (and why the two schemes differ)

A subtle but important consequence of the prefill post-hook fault model:

- In **`direct`** the answer letter is the **first** decode token. Its logits
  are computed during prefill, *before* the hook flips the bit, so the answer is
  produced from clean KV. The corruption only shows up in the *trailing* tokens
  (e.g. the letter is right but is followed by `!!!!!!` garbage). Direct
  accuracy is therefore expected to be essentially **immune** to this fault.
- In **`cot`** the final answer depends on hundreds of reasoning tokens, and
  every one of those decode steps re-reads the corrupted KV. A catastrophic flip
  (bit 14 → value ~1e38) makes the reasoning degenerate into repetition, so no
  valid answer is emitted. CoT accuracy is therefore expected to **collapse**
  under bit 14 while remaining near baseline for the benign bits (0/7/15).

The headline of the experiment is exactly this contrast: **the longer a
generation reads corrupted KV, the more a KV soft error hurts task accuracy** —
so a reasoning model is *more* exposed to unprotected KV than a single-shot
answerer.

## 5. Metrics & recorded data

Per-trial rows (one JSON object per line) are written to
`experiments/results/exp2/trials_<scheme>_<shard>.jsonl` and contain:

- `question_id`, `scheme`, `condition`, `gold`, `pred`, `correct`
- `injection_record` (block/slot, `kv_head`, `head_dim`, `bit`,
  `old_bits`/`new_bits`, `old_value`/`new_value`), `value_naninf`
- `output_text`, `output_token_count`, `seconds`
- provenance: `model`, `dtype`, `vllm_version`, `git_commit`, `seed`, `layer_id`

Aggregates (`--aggregate-only`) are written to:

- `experiments/results/exp2/summary_2.md` — scheme × condition accuracy table
  and accuracy-drop-vs-clean table.
- `experiments/results/exp2/summary_2.csv` — machine-readable counts.

## 6. Grid size

`500 questions × 2 schemes × 5 conditions = 5000 generations.`

`direct` generations are a handful of tokens each; `cot` generations run until
EOS or 1024 tokens (only the collapsed bit-14 runs hit the cap). The two GPUs
each own a 250-question shard, running cot then direct.

## 7. How to run

```bash
conda activate vllm0.8.5
cd /opt/data/data/kv-cache-fault-injection

# Full run across both GPUs, inside tmux (recommended: ~6-7 h wall time):
tmux new-session -d -s exp2 'bash experiments/run_exp2_all.sh'
tmux attach -t exp2            # watch; Ctrl-b d to detach
tail -f experiments/results/exp2/logs/gpu0_g0.log

# Single-GPU / single-scheme (e.g. smoke test):
CUDA_VISIBLE_DEVICES=0 python experiments/run_exp2.py --scheme direct --limit 8
CUDA_VISIBLE_DEVICES=0 python experiments/run_exp2.py --scheme cot --limit 8

# Merge shards -> summary (also runs automatically at the end of the orchestrator):
python experiments/run_exp2.py --aggregate-only --out-dir experiments/results/exp2
```

## 8. Results

_Populated by `summary_2.md` after the full run completes; see
[`experiments/results/exp2/summary_2.md`](experiments/results/exp2/summary_2.md).
A written interpretation with per-condition diagnostics is in
[`experiments/results/exp2/analysis.md`](experiments/results/exp2/analysis.md)._

<!-- AUTO-RESULTS:BEGIN -->
_Generated 2026-07-16 21:44 UTC from `experiments/results/exp2/summary_2.md`._

- Model / vLLM: Qwen3-8B (bf16), vLLM 0.8.5.post1
- Target: V-cache, layer 18, prefill token, 1 (kv_head, head_dim) per question
- git commit: 94f1413de9ee5e9ce2006c74450ed00b18af8e1e

## Accuracy (correct / total)

| Scheme | clean | bit0 | bit7 | bit14 | bit15 |
|---|---:|---:|---:|---:|---:|
| cot | 0.916 (458/500) | 0.912 (456/500) | 0.922 (461/500) | 0.002 (1/500) | 0.926 (463/500) |
| direct | 0.858 (429/500) | 0.858 (429/500) | 0.858 (429/500) | 0.858 (429/500) | 0.858 (429/500) |

## Accuracy drop vs clean baseline (percentage points)

| Scheme | bit0 | bit7 | bit14 | bit15 |
|---|---:|---:|---:|---:|
| cot | +0.4 | -0.6 | +91.4 | -1.0 |
| direct | +0.0 | +0.0 | +0.0 | +0.0 |
<!-- AUTO-RESULTS:END -->

## 9. Pinned environment

Same as the rest of the repo (see [`README.md`](README.md)): vLLM
`0.8.5.post1`, PyTorch `2.6.0+cu124`, Transformers `4.51.3`, Qwen3-8B bf16 on an
RTX 4090, deterministic decoding with `seed=42`, `enforce_eager=True`,
`enable_prefix_caching=False`, `max_num_seqs=1`. `pyarrow` is additionally
required to read the OpenBookQA parquet.
