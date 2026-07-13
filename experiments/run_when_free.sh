#!/usr/bin/env bash
# Resilient launcher for a contended multi-user GPU box.
# Repeatedly waits for a GPU with enough free memory, tries the command, and
# retries on failure (e.g. CUDA OOM caused by a competitor grabbing memory
# during model load). Stops when the command exits 0.
#
# Usage: run_when_free.sh <min_free_mib> <max_attempts> <command...>
set -u
MIN_FREE=${1:-20000}
MAX_ATTEMPTS=${2:-40}
shift 2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  # Pick the GPU with the most free memory.
  best_idx=""; best_free=-1
  while IFS=',' read -r idx free; do
    idx=$(echo "$idx" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
    if [ "$free" -gt "$best_free" ]; then best_free=$free; best_idx=$idx; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  if [ "$best_free" -ge "$MIN_FREE" ]; then
    echo "[run_when_free] attempt $attempt: GPU $best_idx has ${best_free}MiB free -> launching"
    CUDA_VISIBLE_DEVICES=$best_idx "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "[run_when_free] success on attempt $attempt"
      exit 0
    fi
    echo "[run_when_free] attempt $attempt failed (rc=$rc); retrying in 15s"
  else
    echo "[run_when_free] $(date +%H:%M:%S) attempt $attempt: best free=${best_free}MiB < ${MIN_FREE}MiB; waiting 15s"
  fi
  sleep 15
done
echo "[run_when_free] gave up after $MAX_ATTEMPTS attempts"
exit 1
