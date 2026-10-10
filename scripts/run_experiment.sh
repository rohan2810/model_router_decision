#!/usr/bin/env bash
# Train on a dataset variant, then predict its val and test splits.
# Usage: scripts/run_experiment.sh <data_dir> <out_dir> <preds_prefix> [extra train.py args...]
# Example: scripts/run_experiment.sh data/variants/rescue-k8 outputs/exp10 outputs/preds/exp10 --epochs 2
set -euo pipefail
cd "$(dirname "$0")/.."
data="$1"; out="$2"; preds="$3"; shift 3
py=~/miniconda3/envs/jev/bin/python
$py scripts/train.py --train "$data/train.jsonl" --val "$data/val.jsonl" --out "$out" "$@"
for split in val test; do
  $py scripts/predict.py --mode head --model "$out" --data "$data/$split.jsonl" --out "$preds-$split.jsonl"
done
