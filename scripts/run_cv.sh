#!/usr/bin/env bash
# Repo-grouped cross-validation on SWE-bench Verified (folds from scripts/build_variants.py cv).
# For each fold: train on the other folds' tasks (+ SWE-smith train for "mixed"), predict the held-out fold.
# Usage: scripts/run_cv.sh <verified|mixed> <epochs> [model]
set -euo pipefail
cd "$(dirname "$0")/.."
kind="$1"; epochs="$2"; model="${3:-unsloth/Qwen3.5-2B}"
py=~/miniconda3/envs/jev/bin/python
tag="cv-$kind-$(basename "$model" | tr 'A-Z' 'a-z')-${epochs}ep"
mkdir -p outputs/$tag outputs/preds
for fold in data/variants/cv/fold*; do
  f=$(basename "$fold")
  out=outputs/$tag/$f
  [ -f "$out/train_metrics.json" ] || $py scripts/train.py --model "$model" --epochs "$epochs" \
      --train "$fold/train-$kind.jsonl" --val data/processed/val.jsonl --out "$out" > "outputs/$tag/$f.train.log" 2>&1
  $py scripts/predict.py --mode head --model "$out" --data "$fold/heldout.jsonl" \
      --out "outputs/$tag/$f.heldout.jsonl" > "outputs/$tag/$f.predict.log" 2>&1
  echo "$f done"
done
cat outputs/$tag/fold*.heldout.jsonl > outputs/preds/$tag-test.jsonl
echo "all folds done: outputs/preds/$tag-test.jsonl"
