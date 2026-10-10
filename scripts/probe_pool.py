"""CPU probe for a two-model pool: does a simple classifier route between a cheap and a strong model
better than randomly mixing them?

For each prefix length K, a TF-IDF + logistic regression is trained on SWE-smith train runs of the cheap
model to predict whether that run resolves the task, from the issue plus the run's first K agent steps
(K = 0: issue only). The router escalates to the strong model when P(cheap solves) < t; t is chosen on
SWE-smith val and then frozen for SWE-bench Verified. Escalating pays the cheap model's prefix cost plus
the strong model's mean cost on that task. Reported: AUC and the solve gain over the cheap/strong mixing
line at the same cost, with a 95% bootstrap CI over tasks.

Usage:
    python scripts/probe_pool.py --cheap gpt-5-mini --strong gpt-5.2 --ks 0 4 8
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import auc  # noqa: E402
from probe_scouting import render  # noqa: E402


def runs_of(raw_dirs, family, model, cols):
    frames = []
    for raw in raw_dirs:
        for path in sorted(glob.glob(os.path.join(raw, f"{family}-{model}", "*.parquet"))):
            d = pd.read_parquet(path, columns=cols)
            d["split"] = os.path.basename(path).split("-")[0]
            frames.append(d)
    runs = pd.concat(frames, ignore_index=True).dropna(subset=["resolved"])
    runs["resolved"] = runs.resolved.astype(float)
    return runs


def gain_with_ci(solve, cost, s_lo, c_lo, s_hi, c_hi, groups, n_boot=2000, seed=0):
    def gain(idx):
        line = s_lo[idx].mean() + (cost[idx].mean() - c_lo[idx].mean()) / (c_hi[idx].mean() - c_lo[idx].mean()) \
            * (s_hi[idx].mean() - s_lo[idx].mean())
        return solve[idx].mean() - line
    rng = np.random.default_rng(seed)
    boot = [gain(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])) for _ in range(n_boot)]
    return gain(np.arange(len(solve))), np.percentile(boot, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", default="gpt-5-mini")
    ap.add_argument("--strong", default="gpt-5.2")
    ap.add_argument("--raw", nargs="+", default=["data/raw", "data/extra"])
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 4, 8])
    args = ap.parse_args()

    cols = ["instance_id", "messages", "resolved", "instance_cost", "step_cost_list"]
    cheap_smith = runs_of(args.raw, "v3-2k-traj", args.cheap, cols)
    cheap_ver = runs_of(args.raw, "swebench-verified", args.cheap, cols)
    strong = {}
    for fam in ("v3-2k-traj", "swebench-verified"):
        s = runs_of(args.raw, fam, args.strong, ["instance_id", "resolved", "instance_cost"])
        strong[fam] = s.groupby("instance_id").agg(p=("resolved", "mean"), c=("instance_cost", "mean"))
    val_ids = set(cheap_smith.loc[cheap_smith.split == "val", "instance_id"])
    train = cheap_smith[(cheap_smith.split != "val") & cheap_smith.instance_id.isin(strong["v3-2k-traj"].index)]
    val = cheap_smith[cheap_smith.instance_id.isin(val_ids) & cheap_smith.instance_id.isin(strong["v3-2k-traj"].index)]
    test = cheap_ver[cheap_ver.instance_id.isin(strong["swebench-verified"].index)]
    print(f"pool {args.cheap} -> {args.strong}: cheap runs train {len(train)}, val {len(val)}, test {len(test)}")
    for name, d, fam in (("val", val, "v3-2k-traj"), ("test", test, "swebench-verified")):
        st = strong[fam].loc[d.instance_id]
        rescue = ((d.resolved.values < 0.5) & (st.p.values >= 0.5)).mean()
        print(f"  {name}: cheap solves {d.resolved.mean():.3f} at ${d.instance_cost.mean():.4f}, strong {st.p.mean():.3f} at "
              f"${st.c.mean():.4f}, cheap fails & strong solves {rescue:.1%}")

    print(f"\n{'K':>3s} {'AUC val':>8s} {'AUC test':>9s} {'t (val)':>8s} {'val gain':>9s}   test: solve, $/task, gain vs line (95% CI)")
    for k in args.ks:
        texts = {n: [render(m, k) for m in d.messages] for n, d in (("train", train), ("val", val), ("test", test))}
        vec = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
        clf = LogisticRegression(C=0.3, max_iter=3000).fit(vec.fit_transform(texts["train"]), train.resolved >= 0.5)
        out = {}
        for n, d, fam in (("val", val, "v3-2k-traj"), ("test", test, "swebench-verified")):
            p = clf.predict_proba(vec.transform(texts[n]))[:, 1]
            st = strong[fam].loc[d.instance_id]
            prefix = d.step_cost_list.map(lambda c: float(np.sum(c[:k]))).values
            groups = list(d.groupby("instance_id").indices.values())
            out[n] = dict(p=p, y=d.resolved.values, cy=d.instance_cost.values, sp=st.p.values, sc=st.c.values,
                          prefix=prefix, groups=groups, auc=auc(p, d.resolved.values >= 0.5))

        def route(o, t):
            esc = o["p"] < t
            solve = np.where(esc, o["sp"], o["y"])
            cost = np.where(esc, o["prefix"] + o["sc"], o["cy"])
            return solve, cost

        def gain(o, t, n_boot=1):
            solve, cost = route(o, t)
            return gain_with_ci(solve, cost, o["y"], o["cy"], o["sp"], o["sc"], o["groups"], n_boot=n_boot)

        ts = np.unique(np.quantile(out["val"]["p"], np.linspace(0.02, 0.98, 49)))
        t_best = max(ts, key=lambda t: gain(out["val"], t)[0])
        g_val = gain(out["val"], t_best)[0]
        g_test, ci = gain(out["test"], t_best, n_boot=2000)
        solve, cost = route(out["test"], t_best)
        print(f"{k:3d} {out['val']['auc']:8.3f} {out['test']['auc']:9.3f} {t_best:8.3f} {100 * g_val:+8.1f}   "
              f"{solve.mean():.3f}, ${cost.mean():.4f}, {100 * g_test:+.1f} ({100 * ci[0]:+.1f} to {100 * ci[1]:+.1f}), "
              f"escalated {np.mean(out['test']['p'] < t_best):.0%}")


if __name__ == "__main__":
    main()
