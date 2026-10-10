"""CPU probe: does seeing flash's first K agent steps predict whether that flash run resolves the task?

Trains a TF-IDF + logistic regression on SWE-smith train flash runs (one row per run, label = that run's
`resolved`) and reports AUC on SWE-smith val and on SWE-bench Verified flash runs, for several K.
K = 0 is the issue text alone (the current router's input).

Usage:
    python scripts/probe_scouting.py --ks 0 2 4 8
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FLASH = "deepseek-v4-flash"


def load_runs(raw, family):
    frames = []
    for path in sorted(glob.glob(os.path.join(raw, f"{family}-{FLASH}", "*.parquet"))):
        d = pd.read_parquet(path, columns=["instance_id", "messages", "resolved", "step_cost_list"])
        d["split"] = os.path.basename(path).split("-")[0]
        frames.append(d)
    runs = pd.concat(frames, ignore_index=True).dropna(subset=["resolved"])
    runs["resolved"] = runs.resolved.astype(int)
    return runs


def render(messages, k, max_chars=600):
    """Issue text plus the first k (action, observation) pairs, each part truncated."""
    issue = messages[1]["content"]
    parts = [issue]
    steps = [m for m in messages[2:]]
    for i in range(0, min(2 * k, len(steps)), 2):
        action = steps[i]["content"][:max_chars]
        obs = steps[i + 1]["content"][:max_chars] if i + 1 < len(steps) else ""
        parts.append(f"ACTION: {action}\nOBSERVATION: {obs}")
    if 2 * k >= len(steps):
        parts.append("TRAJECTORY_ENDED_WITHIN_PREFIX")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 2, 4, 8])
    args = ap.parse_args()

    smith, verified = load_runs(args.raw, "v3-2k-traj"), load_runs(args.raw, "swebench-verified")
    train, val = smith[smith.split != "val"], smith[smith.split == "val"]
    print(f"flash runs: train {len(train)}, val {len(val)}, Verified {len(verified)}; "
          f"resolved rate {train.resolved.mean():.2f} / {val.resolved.mean():.2f} / {verified.resolved.mean():.2f}")
    print(f"{'K':>3s} {'val AUC':>8s} {'Verified AUC':>13s} {'prefix $/run (Verified)':>24s}")
    for k in args.ks:
        texts = {name: [render(m, k) for m in d.messages] for name, d in
                 [("train", train), ("val", val), ("verified", verified)]}
        vec = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
        x_train = vec.fit_transform(texts["train"])
        clf = LogisticRegression(C=0.3, max_iter=3000).fit(x_train, train.resolved)
        auc_val = roc_auc_score(val.resolved, clf.predict_proba(vec.transform(texts["val"]))[:, 1])
        auc_ver = roc_auc_score(verified.resolved, clf.predict_proba(vec.transform(texts["verified"]))[:, 1])
        cost = verified.step_cost_list.map(lambda c: float(np.sum(c[:k]))).mean()
        print(f"{k:3d} {auc_val:8.3f} {auc_ver:13.3f} {cost:24.4f}")


if __name__ == "__main__":
    main()
