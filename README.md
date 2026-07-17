# KV-Cache Bit-Flip Fault Injection for vLLM (Qwen3-8B)

Controlled single-bit **BF16 KV-cache fault injection** for LLM inference on
vLLM, used to show *why the KV cache read by a PIM / near-memory attention unit
needs integrity protection*: a single flipped bit in the stored KV cache can
change, corrupt, or collapse the generated output.

The full experiment design and rationale are in
[`exp_design.md`](exp_design.md); the environment and runbook are in
[`VLLM_QWEN3_8B_GUIDE.md`](VLLM_QWEN3_8B_GUIDE.md).

## How injection works (no vLLM source changes)

Injection installs itself at runtime by wrapping
`FlashAttentionImpl.forward` with a post-hook. After the original forward runs,
the current step's K/V has already been written into the persistent paged KV
cache, so flipping a bit there corrupts the value that every *subsequent* decode
step will read for that token — the "stored KV suffered a soft error" semantics.
A single BF16 element is bit-cast to `int16` and one bit is XOR-flipped
(one-shot per run). See [`experiments/fault_injection/kv_injector.py`](experiments/fault_injection/kv_injector.py).

Because nothing in the vLLM source tree is modified, this repo is decoupled from
any vLLM fork and runs against a stock install of the pinned version below.

## Pinned environment

| Component | Version |
| --- | --- |
| vLLM | `0.8.5.post1` (upstream base commit `3015d56`) |
| PyTorch | `2.6.0+cu124` |
| Transformers | `4.51.3` |
| Model | Qwen3-8B, BF16 (36 layers, 32 attn heads, 8 KV heads, head_dim 128, block_size 16) |
| GPU | NVIDIA RTX 4090, 24 GB (driver 535.x, CUDA 12.2) |
| Python | 3.10 |

Deterministic decoding is used throughout: `temperature=0`, `top_p=1`,
`top_k=-1`, `seed=42`, `ignore_eos=True`, `max_tokens=128`, `enforce_eager=True`,
`enable_prefix_caching=False`, `max_num_seqs=1`.

> Reproducibility note: seeds, sampling and the target head/dim are fixed, and
> every trial records the originating `git_commit`. Aggregate numbers reproduce
> closely, but bf16 + FlashAttention GPU reductions are not bit-exact across
> different GPUs/driver/vLLM versions, so a re-run may differ by a few borderline
> trials. The committed `results/` are the primary record.

## Layout

```text
experiments/
  fault_injection/      # runtime injector + comparison metrics
    kv_injector.py       # FlashAttention post-hook + BF16 bit flip
    metrics.py           # TCR / TDR / first-divergence / collapse / ROUGE-L
  prompts/prompts_1a.json# 30 long-form prompts (zh/en)
  run_exp1a.py           # exp 1A: single bit flip vs output
  run_exp1b.py           # exp 1B: decode-step propagation/amplification
  verify_injection.py    # smoke test: prove a flip reaches the GPU KV cache
  results/               # committed trials, baselines and summaries
exp_design.md            # experiment design (sections 1-10)
VLLM_QWEN3_8B_GUIDE.md   # environment + runbook
examples/offline_inference/qwen3_8b_source_demo.py
```

## Run

```bash
conda activate vllm0.8.5

# Sanity check: one clean vs one fault run.
python experiments/verify_injection.py

# Experiment 1A: 30 prompts x {K,V} x bits{0,7,14,15} x 3 samples = 720 runs.
python experiments/run_exp1a.py

# Experiment 1B: V-only bit14, steps{16,64,96} x 3 samples = 270 runs.
python experiments/run_exp1b.py

# Minimal presets:
python experiments/run_exp1a.py --min   # 20 prompts, bits{0,14,15}, 2 samples
python experiments/run_exp1b.py --min   # 20 prompts, 2 samples
```

## Results

### 1A — single bit flip vs output (target layer 18, N=90 per row)

| Target | Bit | TCR | Mean TDR | Collapse | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| K | 0 | 0.378 | 0.173 | 0.000 | 0.938 |
| K | 7 | 0.489 | 0.287 | 0.000 | 0.854 |
| K | 14 | 0.900 | 0.886 | 0.889 | 0.118 |
| K | 15 | 0.500 | 0.280 | 0.000 | 0.873 |
| V | 0 | 0.300 | 0.139 | 0.000 | 0.942 |
| V | 7 | 0.467 | 0.272 | 0.000 | 0.880 |
| V | 14 | 1.000 | 0.991 | 1.000 | 0.009 |
| V | 15 | 0.511 | 0.283 | 0.000 | 0.855 |

BF16 bit significance is highly non-uniform: the mantissa LSB (bit 0) is nearly
harmless, while the exponent MSB (bit 14) is catastrophic — and V is more
fragile than K.

### 1B — decode-step propagation (target layer 18, V-only, bit 14, N=90 per row)

| Injection Step | Remaining Tokens | TCR | Post-injection TDR | First Div. After Inj. | Collapse |
|---:|---:|---:|---:|---:|---:|
| 16 | 112 | 1.000 | 0.991 | 1.00 | 1.000 |
| 64 | 64 | 1.000 | 0.984 | 1.00 | 1.000 |
| 96 | 32 | 1.000 | 0.969 | 1.00 | 1.000 |

An earlier injection corrupts a larger fraction of the remaining tokens
(post-injection TDR 0.991 → 0.984 → 0.969), confirming that KV-cache errors
propagate and amplify across decode steps rather than staying local.

### 2 — OpenBookQA answer accuracy under bit flips (N=500, V-cache, layer 18)

Extends 1A/1B from "did the tokens change?" to "did the model answer correctly?"
on the OpenBookQA test set, comparing **direct** answering vs **chain-of-thought**
(see [`EXP2_OPENBOOKQA.md`](EXP2_OPENBOOKQA.md) and
[`experiments/results/exp2/analysis.md`](experiments/results/exp2/analysis.md)).

| Scheme | clean | bit0 | bit7 | bit14 | bit15 |
|---|---:|---:|---:|---:|---:|
| direct | 0.858 | 0.858 | 0.858 | 0.858 | 0.858 |
| cot | 0.916 | 0.912 | 0.922 | **0.002** | 0.926 |

Direct answering is *immune* (the answer letter is the first decode token,
produced from clean prefill KV), while a single bit-14 (exponent MSB) flip
collapses chain-of-thought accuracy from 0.916 to 0.002 — the corrupted KV is
re-read across hundreds of reasoning steps. Reasoning helps on a clean cache
(+5.8 pts) but is the liability under an unprotected KV soft error.

## Conclusion

Even a single BF16 bit flip in the KV cache can silently diverge or collapse
generation, and earlier errors have a larger downstream window. A PIM /
near-memory attention unit that reads the KV cache therefore needs fast
detection and triggered recovery, prioritizing earlier and more-reused KV tiles.
