"""AUC of each per-model `solves_<model>` answer against that model's real outcome (pass rate >= 0.5).

Usage:
    python scripts/eval_aux.py --data data/variants/aux/test.jsonl --preds outputs/preds/exp6-aux-test.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import auc, load_jsonl  # noqa: E402

MODELS = ["deepseek-v4-flash", "gpt-5-mini", "gpt-5.2"]


def p_true(answer):
    if "noul" in answer:
        return float(answer["noul"])
    return float(answer["probabilities"]["true"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()
    rows = {r["id"]: r for r in load_jsonl(args.data)}
    preds = load_jsonl(args.preds)
    print(f"{args.preds}: {len(preds)} tasks")
    for m in MODELS:
        scores = [p_true(p["answers"][f"solves_{m}"]) for p in preds]
        labels = [rows[p["id"]]["pass_rate"][m] >= 0.5 for p in preds]
        print(f"  solves_{m:18s} AUC {auc(scores, labels):.3f}  (mean predicted {sum(scores) / len(scores):.3f}, "
              f"actual {sum(labels) / len(labels):.3f})")
    expected = [p["answers"]["difficulty"].get("score") for p in preds]
    if all(e is not None for e in expected):
        labels = [rows[p["id"]]["oracle"] == "none" for p in preds]
        print(f"  difficulty (expected solvers) AUC for 'no model solves': {auc([-e for e in expected], labels):.3f}")


if __name__ == "__main__":
    main()
