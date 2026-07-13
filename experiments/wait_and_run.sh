#!/usr/bin/env bash
# Wait until some GPU has enough free memory, then run the given command on it.
# Usage: wait_and_run.sh <min_free_mib> <command...>
set -u
MIN_FREE=${1:-18000}
shift
PY=/opt/data/data/anaconda3/envs/vllm0.8.5/bin/python
export PYTHONUNBUFFERED=1

while true; do
  # Print "index free" for each GPU.
  mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  pick=""
  for r in "${rows[@]}"; do
    idx=$(echo "$r" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    free=$(echo "$r" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    if [ "$free" -ge "$MIN_FREE" ]; then pick=$idx; break; fi
  done
  if [ -n "$pick" ]; then
    echo "[wait_and_run] GPU $pick has >= ${MIN_FREE}MiB free. Launching."
    export CUDA_VISIBLE_DEVICES=$pick
    exec "$@"
  fi
  echo "[wait_and_run] $(date +%H:%M:%S) no GPU with >= ${MIN_FREE}MiB free yet; retrying in 20s."
  sleep 20
done
