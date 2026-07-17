# Experiment 2 summary -- OpenBookQA accuracy under KV-cache bit flips

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
