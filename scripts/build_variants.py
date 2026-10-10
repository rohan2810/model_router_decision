"""Build dataset variants from data/processed/{train,val,test}.jsonl for follow-up experiments.

  aux       Same rows, plus auxiliary questions next to `route`: one `noul` per model ("will <model>
            resolve this task?", gold = its pass rate) and a `difficulty` score (how many of the three
            models resolve it; soft gold from the per-model pass rates). `route` stays the headline.
  cv        Repo-grouped folds over SWE-bench Verified (test.jsonl). For each fold f writes
            fold{f}/heldout.jsonl, fold{f}/train-verified.jsonl (other folds) and
            fold{f}/train-mixed.jsonl (SWE-smith train + other folds).
  scout     One row per deepseek-v4-flash run: state = issue + that run's first K agent steps. The
            router then decides whether flash continues or the task restarts on another model. Gold uses
            this run's own outcome for flash and the task-level pass rates for the others; a switch pays
            the prefix cost on top of the new model's cost.

Usage:
    python scripts/build_variants.py aux --out data/variants/aux
    python scripts/build_variants.py cv --folds 5 --out data/variants/cv
    python scripts/build_variants.py scout --k 8 --out data/variants/scout-k8
"""

import argparse
import glob
import json
import os
import sys

MODELS = ["deepseek-v4-flash", "gpt-5-mini", "gpt-5.2"]
LEVELS = ["None of the three models resolves it.", "Exactly one of the three models resolves it.",
          "Two of the three models resolve it.", "All three models resolve it."]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def count_distribution(pass_rate):
    """P(exactly n of the models resolve the task), treating models as independent."""
    dist = [1.0]
    for m in MODELS:
        p = pass_rate[m]
        dist = [a * (1 - p) + b * p for a, b in zip(dist + [0.0], [0.0] + dist)]
    return dist


def with_aux(row):
    row = dict(row)
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    for m in MODELS:
        p = row["pass_rate"][m]
        questions[f"solves_{m}"] = {"type": "noul", "instructions": f"Will {m} resolve this coding task?"}
        gold[f"solves_{m}"] = {"type": "noul", "label": "true" if p >= 0.5 else "false",
                               "probabilities": {"true": round(p, 6), "false": round(1 - p, 6)}}
    dist = count_distribution(row["pass_rate"])
    questions["difficulty"] = {"type": "score", "criteria": LEVELS,
                               "instructions": "How many of deepseek-v4-flash, gpt-5-mini and gpt-5.2 resolve this task?"}
    gold["difficulty"] = {"type": "score", "label": str(max(range(4), key=dist.__getitem__)),
                          "probabilities": {str(i): round(p, 6) for i, p in enumerate(dist)}}
    row["questions"], row["gold"] = json.dumps(questions), json.dumps(gold)
    return row


def repo_folds(rows, k):
    """Assign whole repos to k folds, largest repos first, always into the smallest fold."""
    sizes = {}
    for r in rows:
        sizes[r["repo"]] = sizes.get(r["repo"], 0) + 1
    folds, load_ = {}, [0] * k
    for repo in sorted(sizes, key=sizes.get, reverse=True):
        f = load_.index(min(load_))
        folds[repo], load_[f] = f, load_[f] + sizes[repo]
    return folds


def render_prefix(messages, k, max_chars):
    steps = messages[2:]
    lines = [f"--- First {k} steps of deepseek-v4-flash on this task ---"]
    for i in range(0, min(2 * k, len(steps)), 2):
        lines.append(f"[step {i // 2 + 1}] ACTION: {steps[i]['content'][:max_chars]}")
        if i + 1 < len(steps):
            lines.append(f"OBSERVATION: {steps[i + 1]['content'][:max_chars]}")
    if 2 * k >= len(steps):
        lines.append("[deepseek-v4-flash finished within these steps]")
    return "\n".join(lines)


def scout_rows(rows, raw, family, k, max_chars):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.dirname(__file__))
    from build_dataset import gold_distribution, confidence

    by_id = {r["id"]: r for r in rows}
    out = []
    for path in sorted(glob.glob(os.path.join(raw, f"{family}-deepseek-v4-flash", "*.parquet"))):
        run = os.path.basename(path).split("-")[0]
        d = pd.read_parquet(path, columns=["instance_id", "messages", "resolved", "instance_cost", "step_cost_list"])
        for t in d.itertuples():
            base = by_id.get(t.instance_id)
            if base is None or t.resolved is None or (isinstance(t.resolved, float) and np.isnan(t.resolved)):
                continue
            prefix_cost = float(np.sum(t.step_cost_list[:k]))
            pass_rate = dict(base["pass_rate"], **{"deepseek-v4-flash": float(t.resolved)})
            cost = {m: (float(t.instance_cost) if m == "deepseek-v4-flash" else prefix_cost + base["cost"][m])
                    for m in MODELS}
            probs = gold_distribution(pass_rate)
            questions = json.loads(base["questions"])
            questions["route"]["instructions"] = (
                "deepseek-v4-flash has started on this coding task (its first steps are shown). Which is the "
                "cheapest model that will resolve it: let deepseek-v4-flash continue, or restart it on another model?")
            solvers = [m for m in MODELS if pass_rate[m] >= 0.5]
            out.append(dict(base, id=f"{t.instance_id}#{run}", task_id=t.instance_id,
                            state=base["state"] + "\n\n" + render_prefix(t.messages, k, max_chars),
                            questions=json.dumps(questions),
                            gold=json.dumps({"route": {"type": "choice", "label": max(probs, key=probs.get),
                                                       "probabilities": {m: round(p, 6) for m, p in probs.items()},
                                                       "confidence": round(confidence(probs), 6)}}),
                            pass_rate=pass_rate, cost=cost, prefix_cost=prefix_cost,
                            oracle=solvers[0] if solvers else "none"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["aux", "cv", "scout"])
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--k", type=int, default=8, help="scout: agent steps shown")
    ap.add_argument("--max-chars", type=int, default=600, help="scout: characters kept per action/observation")
    args = ap.parse_args()

    if args.variant == "aux":
        for split in ("train", "val", "test"):
            rows = [with_aux(r) for r in load(os.path.join(args.data, f"{split}.jsonl"))]
            write(rows, os.path.join(args.out, f"{split}.jsonl"))
            print(f"{split}: {len(rows)} rows, {len(json.loads(rows[0]['questions']))} questions each")
        return

    if args.variant == "scout":
        for split, family in (("train", "v3-2k-traj"), ("val", "v3-2k-traj"), ("test", "swebench-verified")):
            rows = scout_rows(load(os.path.join(args.data, f"{split}.jsonl")), args.raw, family, args.k, args.max_chars)
            write(rows, os.path.join(args.out, f"{split}.jsonl"))
            chars = sorted(len(r["state"]) for r in rows)
            print(f"{split}: {len(rows)} flash runs over {len({r['task_id'] for r in rows})} tasks | state chars "
                  f"median {chars[len(chars) // 2]}, p95 {chars[int(len(chars) * .95)]} | "
                  f"mean prefix ${sum(r['prefix_cost'] for r in rows) / len(rows):.4f}")
        return

    test, train = load(os.path.join(args.data, "test.jsonl")), load(os.path.join(args.data, "train.jsonl"))
    fold_of = repo_folds(test, args.folds)
    for f in range(args.folds):
        held = [r for r in test if fold_of[r["repo"]] == f]
        rest = [r for r in test if fold_of[r["repo"]] != f]
        d = os.path.join(args.out, f"fold{f}")
        write(held, os.path.join(d, "heldout.jsonl"))
        write(rest, os.path.join(d, "train-verified.jsonl"))
        write(train + rest, os.path.join(d, "train-mixed.jsonl"))
        repos = sorted({r["repo"].split("__")[1] for r in held})
        print(f"fold{f}: held out {len(held):3d} tasks ({', '.join(repos)}) | train-verified {len(rest)} | train-mixed {len(train) + len(rest)}")


if __name__ == "__main__":
    main()
