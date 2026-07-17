# Experiment 2 — preliminary analysis (OpenBookQA, N=500)

Qwen3-8B (bf16), vLLM 0.8.5.post1. One BF16 bit flipped in the V-cache, middle
layer (18), middle prompt token, one seeded `(kv_head, head_dim)` per question.
`5000` generations total (`500 questions × 2 schemes × 5 conditions`).

## 1. Headline accuracy

| Scheme | clean | bit0 | bit7 | bit14 | bit15 |
|---|---:|---:|---:|---:|---:|
| direct | 0.858 | 0.858 | 0.858 | **0.858** | 0.858 |
| cot | 0.916 | 0.912 | 0.922 | **0.002** | 0.926 |

Accuracy drop vs each scheme's own clean baseline (percentage points):

| Scheme | bit0 | bit7 | bit14 | bit15 |
|---|---:|---:|---:|---:|
| direct | +0.0 | +0.0 | +0.0 | +0.0 |
| cot | +0.4 | −0.6 | **−91.4** | +1.0 |

## 2. Per-condition diagnostics

| Scheme | Cond | Acc | Unparsed | NaN/Inf flips | Mean out tok | Hit 1024 cap |
|---|---|---:|---:|---:|---:|---:|
| direct | clean | 0.858 | 0 | 0 | 2.0 | 0 |
| direct | bit0 | 0.858 | 0 | 0 | 2.0 | 0 |
| direct | bit7 | 0.858 | 0 | 0 | 2.0 | 0 |
| direct | bit14 | 0.858 | 0 | 16 | 8.0 | 0 |
| direct | bit15 | 0.858 | 0 | 0 | 2.0 | 0 |
| cot | clean | 0.916 | 4 | 0 | 600.8 | 87 |
| cot | bit0 | 0.912 | 2 | 0 | 601.0 | 92 |
| cot | bit7 | 0.922 | 3 | 0 | 592.4 | 89 |
| cot | bit14 | 0.002 | 499 | 21 | 1023.0 | 499 |
| cot | bit15 | 0.926 | 1 | 0 | 598.1 | 83 |

## 3. Findings

1. **Direct answering is completely immune to this fault.** For all 500
   questions the predicted letter is *identical* across `clean/bit0/bit7/bit14/bit15`
   (500/500). Reason: the answer letter is the **first** decode token, whose
   logits are computed during prefill *before* the post-hook flips the stored
   bit. Even the catastrophic bit-14 flip (16 of which produce a NaN/Inf V value)
   only garbles the *trailing* tokens — mean output grows from 2 to 8 tokens —
   but the already-emitted letter is unchanged. So `direct` accuracy is flat at
   0.858 regardless of bit position.

2. **CoT is catastrophically fragile to the exponent-MSB flip.** Bit 14 drops
   CoT accuracy from 0.916 to 0.002 (1/500 correct). 499/500 outputs are
   *unparseable*: the reasoning degenerates into repetition and runs to the
   full 1024-token cap (mean 1023 tok, 499/500 hit the cap) without ever
   emitting a valid `Answer:` line. The transition is one-directional —
   **457 questions flip correct→wrong, 0 flip wrong→correct** — i.e. the fault
   never "accidentally helps".

3. **BF16 bit significance is extremely non-uniform, and the benign bits are
   truly benign.** For CoT, bits 0 (mantissa LSB), 7 and 15 (sign) stay within
   ±1 pt of the clean baseline (0.912 / 0.922 / 0.926 vs 0.916); the small
   wiggle is noise, not signal. Only bit 14 (exponent MSB), which scales the
   value by ~2^128, is destructive. This matches exp 1A/1B, now measured on task
   accuracy rather than token divergence.

4. **Reasoning helps when the cache is clean, but is the liability under
   faults.** Clean CoT (0.916) beats clean direct (0.858) by +5.8 pts —
   step-by-step reasoning genuinely improves OpenBookQA accuracy. But that gain
   is bought with hundreds of decode steps (mean ~600 tokens) that each re-read
   the KV cache, so a single severe KV soft error erases the entire reasoning
   chain. The very mechanism that makes CoT more accurate makes it far more
   exposed to unprotected KV.

## 4. Takeaway for KV-cache integrity (PIM / near-memory attention)

The downstream cost of an unprotected KV soft error is dominated by **how long
the generation keeps reading the corrupted tile**, not merely by whether a flip
occurs:

- A single-shot answer (reads corrupted KV ~0 times before deciding) is
  effectively immune.
- A long reasoning trace (reads corrupted KV hundreds of times) is destroyed by
  the same flip.

Because modern deployments increasingly rely on long chain-of-thought decoding,
a near-memory attention unit that serves the KV cache needs fast detection and
recovery for high-significance bits (the BF16 exponent), and should prioritise
protection of early / heavily-reused KV tiles that feed long generations.

## 5. Reproduce / drill down

```bash
conda activate vllm0.8.5
cd /opt/data/data/kv-cache-fault-injection
python experiments/run_exp2.py --aggregate-only --out-dir experiments/results/exp2
# raw per-trial records:
#   experiments/results/exp2/trials_{cot,direct}_{g0,g1}.jsonl
```
