"""Score router predictions against gold labels and real per-model outcomes.

For each predictions file (from scripts/predict.py), reports:
  - accuracy        argmax == gold label (cheapest model most likely to solve)
  - soft_ce         cross-entropy against the soft gold distribution
  - ece             calibration error of the argmax confidence
  - solve / cost    expected resolve rate and mean $ per task when routing by argmax,
                    using each model's measured pass rate and cost on that task
  - escalation curve: pick the cheapest model whose cumulative probability (itself plus all
    cheaper options) reaches tau; low tau favours cheap models, tau=1 always picks the strongest

Fixed baselines (always-<model>, random, oracle) are reported alongside.

Usage:
    python scripts/evaluate.py --data data/processed/test.jsonl \
        --preds base-head=outputs/preds/base-head.jsonl \
                base-zeroshot=outputs/preds/base-zeroshot.jsonl \
                trained=outputs/preds/trained.jsonl \
        --out outputs/eval/test.json
"""

import argparse
import json
import math
import os

TAUS = [0.3, 0.5, 0.7, 0.9]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def ece(conf, correct, bins=10):
    total, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            gap = abs(sum(conf[i] for i in idx) / len(idx) - sum(correct[i] for i in idx) / len(idx))
            total += len(idx) / n * gap
    return total


def outcome(rows, picks):
    return {
        "solve": sum(r["pass_rate"][p] for r, p in zip(rows, picks)) / len(rows),
        "cost": sum(r["cost"][p] for r, p in zip(rows, picks)) / len(rows),
        "picks": {m: picks.count(m) for m in rows[0]["pass_rate"]},
    }


def escalate(probs, models, tau):
    cumulative = 0.0
    for m in models:
        cumulative += probs[m]
        if cumulative >= tau:
            return m
    return models[-1]


def score(rows, preds, models):
    gold = [json.loads(r["gold"])["route"] for r in rows]
    probs = [p["probabilities"] for p in preds]
    argmax = [max(p, key=p.get) for p in probs]
    correct = [float(a == g["label"]) for a, g in zip(argmax, gold)]
    conf = [max(p.values()) for p in probs]
    soft_ce = [-sum(g["probabilities"][m] * math.log(max(p[m], 1e-12)) for m in models)
               for p, g in zip(probs, gold)]
    return {
        "accuracy": sum(correct) / len(rows),
        "soft_ce": sum(soft_ce) / len(rows),
        "ece": ece(conf, correct),
        **outcome(rows, argmax),
        "escalation": {str(t): outcome(rows, [escalate(p, models, t) for p in probs]) for t in TAUS},
    }


def baselines(rows, models):
    out = {f"always-{m}": outcome(rows, [m] * len(rows)) for m in models}
    out["random"] = {
        "solve": sum(o["solve"] for o in out.values()) / len(models),
        "cost": sum(o["cost"] for o in out.values()) / len(models),
    }
    oracle = [r["oracle"] if r["oracle"] != "none" else models[-1] for r in rows]
    out["oracle"] = outcome(rows, oracle)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/test.jsonl")
    ap.add_argument("--preds", nargs="+", required=True, help="name=path.jsonl")
    ap.add_argument("--out", default=None, help="write all metrics as JSON")
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    models = list(json.loads(rows[0]["questions"])["route"]["criteria"])  # cheapest first
    by_id = {r["id"]: r for r in rows}

    results = {"baselines": baselines(rows, models), "routers": {}}
    for spec in args.preds:
        name, _, path = spec.rpartition("=")
        name = name or os.path.splitext(os.path.basename(path))[0]
        preds = load_jsonl(path)
        missing = set(by_id) - {p["id"] for p in preds}
        if missing:
            print(f"warning: {name} covers {len(preds)}/{len(rows)} tasks; scoring that subset")
        subset = [by_id[p["id"]] for p in preds]
        results["routers"][name] = {"n": len(preds), **score(subset, preds, models)}

    print(f"\n{args.data}: {len(rows)} tasks, models (cheapest first): {', '.join(models)}\n")
    print(f"{'router':28s} {'acc':>6s} {'softCE':>7s} {'ECE':>6s} {'solve':>6s} {'$/task':>8s}  picks")
    for name, r in results["baselines"].items():
        print(f"{name:28s} {'':>6s} {'':>7s} {'':>6s} {r['solve']:6.3f} {r['cost']:8.4f}  {r.get('picks', '')}")
    for name, r in results["routers"].items():
        print(f"{name:28s} {r['accuracy']:6.3f} {r['soft_ce']:7.3f} {r['ece']:6.3f} "
              f"{r['solve']:6.3f} {r['cost']:8.4f}  {r['picks']}")
        for tau, e in r["escalation"].items():
            print(f"  {'tau=' + tau:26s} {'':>6s} {'':>7s} {'':>6s} {e['solve']:6.3f} {e['cost']:8.4f}  {e['picks']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
