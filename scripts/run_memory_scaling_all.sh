#!/usr/bin/env bash
# Launch the memory-scaling experiment for every (dataset, model) pair in parallel.
# Each run trains sequentially; runs are independent. Re-running resumes unfinished runs.
# Usage: bash scripts/run_memory_scaling_all.sh [extra args for run_memory_scaling.py]

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
DATASETS=(gsm8k math500 aime)
# Override with a space-separated list, e.g. MODELS="swissai/ahadinia/Qwen/Qwen3-8B" bash scripts/...
if [ -n "${MODELS:-}" ]; then
  read -r -a MODELS <<< "$MODELS"
else
  MODELS=(
    "swissai/SwissAI-Research/Qwen/Qwen2.5-7B-Instruct"
    "swissai/SwissAI-Research/Qwen/Qwen3-8B"
  )
fi
LOG_DIR="results/memory_scaling/logs"
# Separate log files per configuration, e.g. "--memory_mode append" -> "_memory_mode_append"
LOG_SUFFIX="$(printf '%s' "$*" | tr -cs 'A-Za-z0-9.' '_' | sed 's/^_*/_/; s/_*$//')"
[ "$LOG_SUFFIX" = "_" ] && LOG_SUFFIX=""
mkdir -p "$LOG_DIR"

pids=()
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    log="$LOG_DIR/${dataset}_$(basename "$model")${LOG_SUFFIX}.log"
    echo "Starting $dataset / $model -> $log"
    "$PYTHON" run_memory_scaling.py --dataset "$dataset" --model_name "$model" "$@" >> "$log" 2>&1 &
    pids+=($!)
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

"$PYTHON" plot_memory_scaling.py
exit $status
