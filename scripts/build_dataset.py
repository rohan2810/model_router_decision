"""Build router train/val/test files in the Unsloth decision-model format.

Each output row is one coding task. The router answers a single `choice`
question: which is the cheapest model that will resolve this task?

Gold is a soft distribution built from each model's pass rate across runs.
With models sorted cheapest first and p_k = pass rate of model k:

    P(k is the cheapest solver) = p_k * prod_{j cheaper than k} (1 - p_j)

Whatever probability is left, prod_k (1 - p_k), goes to a fourth option,
"none": no model will resolve it. What to do on "none" (send it to the
strongest model anyway, ask the user for more detail, ...) is left to the
deployment policy.

Usage:
    python scripts/build_dataset.py --raw data/raw --out data/processed
"""

import argparse
import glob
import json
import math
import os
import re

import pandas as pd
import pyarrow.parquet as pq

# Cheapest first. Order is checked against mean training cost at build time.
MODELS = ["deepseek-v4-flash", "gpt-5-mini", "gpt-5.2"]
NONE = "none"

TRAIN_FAMILY = "v3-2k-traj"         # SWE-smith tasks -> train / val
TEST_FAMILY = "swebench-verified"   # SWE-bench Verified -> test

# SWE-smith problem statements are wrapped in the agent prompt; Verified ones are not.
WRAP_HEAD = re.compile(r"^\s*Consider the following PR description:\s*\n")
# Leftover tags from SWE-smith's issue generator, e.g. "**Issue Text**\n<START WRITING>\n...".
GEN_TAGS = re.compile(r"</?START WRITING>|^\s*\*\*Issue Text\*\*\s*$", re.MULTILINE)


def clean_statement(text):
    return GEN_TAGS.sub("", WRAP_HEAD.sub("", text)).strip()


def load_runs(raw, family):
    """One row per (task, model, run)."""
    frames = []
    for model in MODELS:
        files = sorted(glob.glob(os.path.join(raw, f"{family}-{model}", "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet files for {family}-{model} under {raw}")
        for path in files:
            cols = ["instance_id", "problem_statement", "resolved", "instance_cost"]
            cols = [c for c in cols if c in pq.read_schema(path).names]
            df = pd.read_parquet(path, columns=cols)
            df["model"] = model
            df["split"] = os.path.basename(path).split("-")[0]
            frames.append(df)
    runs = pd.concat(frames, ignore_index=True)
    dropped = runs.resolved.isna().sum()
    runs = runs.dropna(subset=["resolved"])
    runs["resolved"] = runs.resolved.astype(float)
    print(f"[{family}] {len(runs)} runs kept, {dropped} dropped (no `resolved` result)")
    return runs


def repo_of(instance_id):
    # SWE-smith: "owner__repo.<commit>.<mutation>"; Verified: "owner__repo-<pr>"
    return instance_id.split(".")[0] if "." in instance_id else instance_id.rsplit("-", 1)[0]


def build_questions(costs):
    criteria = {m: f"Average cost ${costs[m]:.3f} per task." for m in MODELS}
    criteria[NONE] = "None of these models will resolve this task."
    return {
        "route": {
            "type": "choice",
            "instructions": "Which is the cheapest model that will resolve this coding task?",
            "criteria": criteria,
        }
    }


def gold_distribution(pass_rate):
    probs, none_solve = {}, 1.0
    for m in MODELS:
        probs[m] = none_solve * pass_rate[m]
        none_solve *= 1 - pass_rate[m]
    probs[NONE] = none_solve
    return probs


def confidence(probs):
    # 1 - normalized entropy: 1 when one option has all the mass, 0 when uniform.
    h = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return 1 - h / math.log(len(probs))


def build_rows(runs, questions, split_of):
    by_task = runs.groupby(["instance_id", "model"])
    pass_rate = by_task.resolved.mean().unstack()
    cost = by_task.instance_cost.mean().unstack() if "instance_cost" in runs else None
    n_runs = by_task.size().unstack()
    pass_rate = pass_rate[MODELS].dropna()  # keep tasks every model has a result for

    statements = runs.groupby("instance_id").problem_statement.first().map(clean_statement)
    rows = []
    for iid, pr in pass_rate.iterrows():
        pr = pr.to_dict()
        probs = gold_distribution(pr)
        solvers = [m for m in MODELS if pr[m] >= 0.5]
        rows.append({
            "id": iid,
            "repo": repo_of(iid),
            "split": split_of(iid),
            "state": json.dumps(statements[iid]),
            "questions": json.dumps(questions),
            "gold": json.dumps({"route": {
                "type": "choice",
                "label": max(probs, key=probs.get),
                "probabilities": {m: round(p, 6) for m, p in probs.items()},
                "confidence": round(confidence(probs), 6),
            }}),
            # Not model input: used for cost / solve-rate evaluation.
            "pass_rate": {m: round(pr[m], 4) for m in MODELS},
            "cost": {m: round(float(cost.loc[iid, m]), 5) for m in MODELS} if cost is not None else None,
            "n_runs": {m: int(n_runs.loc[iid, m]) for m in MODELS},
            "oracle": solvers[0] if solvers else NONE,
        })
    return rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def summarize(name, rows):
    df = pd.DataFrame(rows)
    labels = df.gold.map(lambda g: json.loads(g)["route"]["label"]).value_counts().to_dict()
    print(f"  {name:5s} {len(df):5d} tasks, {df.repo.nunique():3d} repos | gold argmax {labels} | oracle {df.oracle.value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_runs = load_runs(args.raw, TRAIN_FAMILY)
    test_runs = load_runs(args.raw, TEST_FAMILY)

    # Option descriptions use training-set costs only, so test costs never leak into the input.
    costs = train_runs.groupby("model").instance_cost.mean().to_dict()
    assert sorted(MODELS, key=costs.get) == MODELS, f"MODELS not sorted by cost: {costs}"
    questions = build_questions(costs)

    val_ids = set(train_runs.loc[train_runs.split == "val", "instance_id"])
    train_val = build_rows(train_runs, questions, lambda i: "val" if i in val_ids else "train")
    test = build_rows(test_runs, questions, lambda i: "test")

    splits = {
        "train": [r for r in train_val if r["split"] == "train"],
        "val": [r for r in train_val if r["split"] == "val"],
        "test": test,
    }
    assert not {r["id"] for r in splits["train"]} & {r["id"] for r in splits["val"]}
    assert not {r["repo"] for r in train_val} & {r["repo"] for r in test}, "repo overlap train/test"

    print("Summary:")
    for name, rows in splits.items():
        write_jsonl(rows, os.path.join(args.out, f"{name}.jsonl"))
        summarize(name, rows)


if __name__ == "__main__":
    main()
