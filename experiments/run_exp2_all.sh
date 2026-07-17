#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Orchestrate the full Experiment 2 (OpenBookQA accuracy under KV-cache bit
# flips) across both RTX 4090s. Each GPU owns a disjoint 250-question shard and
# runs both answering schemes (cot first because it is the slow one, then
# direct). When both GPUs finish, the per-shard trial files are merged into the
# scheme x condition accuracy summary.
#
# Designed to be launched inside tmux:
#   tmux new-session -d -s exp2 'bash experiments/run_exp2_all.sh'
#
# Progress/logs land in experiments/results/exp2/logs/.

set -uo pipefail

REPO_ROOT="/opt/data/data/kv-cache-fault-injection"
CONDA_SH="/opt/data/data/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="vllm0.8.5"
OUT_DIR="${REPO_ROOT}/experiments/results/exp2"
LOG_DIR="${OUT_DIR}/logs"
PY="python experiments/run_exp2.py"

# 500-question test set split in half, one shard per GPU.
SHARD0_START=0;   SHARD0_END=250;  SHARD0_TAG="g0"
SHARD1_START=250; SHARD1_END=500;  SHARD1_TAG="g1"

# These GPUs are SHARED with other users (~6 GB each, fluctuating), which counts
# against vLLM's gpu_memory_utilization budget. High util + a tiny profiling
# batch (prompts are <=150 tokens) leaves the KV cache just enough room at init;
# once init succeeds the KV cache is pre-allocated and the run is robust to later
# external growth.
GPU0_UTIL=0.97;  GPU1_UTIL=0.97
BATCH_TOKENS=512

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "[orchestrator] start $(date -Is)" | tee "${LOG_DIR}/orchestrator.log"

run_gpu_pipeline() {
  local gpu="$1" start="$2" end="$3" tag="$4" util="$5"
  {
    echo "=== GPU ${gpu} shard ${tag} [${start},${end}) start $(date -Is) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" ${PY} --scheme cot \
      --start "${start}" --end "${end}" --tag "${tag}" \
      --gpu-memory-utilization "${util}" --max-num-batched-tokens "${BATCH_TOKENS}" \
      --out-dir "${OUT_DIR}"
    echo "=== GPU ${gpu} shard ${tag}: cot done $(date -Is) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" ${PY} --scheme direct \
      --start "${start}" --end "${end}" --tag "${tag}" \
      --gpu-memory-utilization "${util}" --max-num-batched-tokens "${BATCH_TOKENS}" \
      --out-dir "${OUT_DIR}"
    echo "=== GPU ${gpu} shard ${tag}: direct done $(date -Is) ==="
  } > "${LOG_DIR}/gpu${gpu}_${tag}.log" 2>&1
}

run_gpu_pipeline 0 "${SHARD0_START}" "${SHARD0_END}" "${SHARD0_TAG}" "${GPU0_UTIL}" &
PID0=$!
run_gpu_pipeline 1 "${SHARD1_START}" "${SHARD1_END}" "${SHARD1_TAG}" "${GPU1_UTIL}" &
PID1=$!

echo "[orchestrator] launched GPU0 pid=${PID0}, GPU1 pid=${PID1}" | tee -a "${LOG_DIR}/orchestrator.log"

wait "${PID0}"; RC0=$?
wait "${PID1}"; RC1=$?
echo "[orchestrator] GPU0 rc=${RC0}, GPU1 rc=${RC1}" | tee -a "${LOG_DIR}/orchestrator.log"

echo "[orchestrator] aggregating $(date -Is)" | tee -a "${LOG_DIR}/orchestrator.log"
${PY} --aggregate-only --out-dir "${OUT_DIR}" 2>&1 | tee "${LOG_DIR}/aggregate.log"

echo "[orchestrator] done $(date -Is)" | tee -a "${LOG_DIR}/orchestrator.log"
touch "${OUT_DIR}/DONE"
