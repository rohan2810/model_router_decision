#!/usr/bin/env bash
# Download the six SWE-Router result datasets into data/raw/<dataset>/.
set -euo pipefail
cd "$(dirname "$0")/.."
for family in v3-2k-traj swebench-verified; do
  for model in deepseek-v4-flash gpt-5-mini gpt-5.2; do
    ds="$family-$model"
    mkdir -p "data/raw/$ds"
    for path in $(curl -sfL "https://huggingface.co/api/datasets/SWE-Router/$ds/tree/main/data" \
                  | python3 -c "import json,sys; [print(f['path']) for f in json.load(sys.stdin)]"); do
      echo "SWE-Router/$ds/$path"
      curl -sfL "https://huggingface.co/datasets/SWE-Router/$ds/resolve/main/$path" -o "data/raw/$ds/$(basename "$path")"
    done
  done
done
