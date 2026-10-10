# model_router_decision

A router that reads a coding task and picks the **cheapest model that will solve it**, trained as an
[Unsloth decision model](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
(LLM backbone + Clef-style head, served through the Jev-compatible `/v1/systemone` API).

Candidate models, cheapest first: `deepseek-v4-flash` → `gpt-5-mini` → `gpt-5.2`.

## Data

Labels come from [SWE-Router](https://huggingface.co/SWE-Router) runs of each model with mini-swe-agent
(`resolved` pass/fail and `instance_cost` per run).

| Split | Source | Tasks | Repos |
|---|---|---|---|
| train | SWE-smith (`SWE-Router/v3-2k-traj-<model>`, `train*` splits) | 1,913 | 113 |
| val | SWE-smith (`SWE-Router/v3-2k-traj-<model>`, `val` split) | 337 | 10 |
| test | SWE-bench Verified (`SWE-Router/swebench-verified-<model>`) | 500 | 12 |

No repo appears in more than one split.

Each row is one task:

- `state`: the problem statement as plain text, with SWE-smith's agent-prompt wrapper removed.
- `questions`: one `choice` question, `route`, over the three models plus `none` (no model resolves it).
- `gold`: soft distribution. With models sorted cheapest first and `p_k` each model's pass rate across runs,
  `P(k) = p_k · Π_{j cheaper} (1 − p_j)` and `P(none) = Π_k (1 − p_k)`. What to do on `none` is a deployment policy.
- `pass_rate`, `cost`, `n_runs`, `oracle`: per-model results for evaluation, not model input.

## Results

Each router picks its highest-probability option (a `none` pick goes to gpt-5.2): **solve** is the expected share of
tasks resolved, **$/task** the average cost. Accuracy is against the gold label (cheapest model
that resolves the task). Settings and commands for each run are in [EXPERIMENTS.md](EXPERIMENTS.md).

**Val** (SWE-smith, 337 tasks)

| Router | Exp | Solve | $/task | Acc | Picks (flash / mini / gpt-5.2 / none) |
|---|---|---|---|---|---|
| always deepseek-v4-flash | – | 0.537 | 0.0091 | – | – |
| always gpt-5-mini | – | 0.490 | 0.0327 | – | – |
| always gpt-5.2 | – | 0.623 | 0.2775 | – | – |
| random | – | 0.550 | 0.1065 | – | – |
| oracle (perfect routing) | – | 0.691 | 0.1609 | – | 181 / 20 / 136 / – |
| Qwen3.5-0.8B, prompted (zero-shot) | 1a | 0.537 | 0.0091 | 0.537 | 337 / 0 / 0 / 0 |
| Qwen3.5-0.8B + untrained head | 1b | 0.507 | 0.0223 | 0.291 | 161 / 176 / 0 / 0 |
| Qwen3.5-0.8B, trained | 1c | 0.555 | 0.1382 | 0.454 | 186 / 0 / 0 / 151 |
| laya (English, 421M), zero-shot | 2a | 0.531 | 0.0881 | 0.104 | 0 / 262 / 34 / 41 |
| laya-multilingual (322M), zero-shot | 2b | 0.611 | 0.2403 | 0.306 | 15 / 24 / 0 / 298 |
| laya-typed-decisions (421M), zero-shot | 2c | 0.531 | 0.1184 | 0.068 | 0 / 237 / 99 / 1 |

**Test** (SWE-bench Verified, 500 tasks)

| Router | Exp | Solve | $/task | Acc | Picks (flash / mini / gpt-5.2 / none) |
|---|---|---|---|---|---|
| always deepseek-v4-flash | – | 0.665 | 0.0140 | – | – |
| always gpt-5-mini | – | 0.556 | 0.0331 | – | – |
| always gpt-5.2 | – | 0.727 | 0.2558 | – | – |
| random | – | 0.649 | 0.1010 | – | – |
| oracle (perfect routing) | – | 0.724 | 0.0952 | – | 377 / 7 / 116 / – |
| Qwen3.5-0.8B, prompted (zero-shot) | 1a | 0.665 | 0.0140 | 0.754 | 500 / 0 / 0 / 0 |
| Qwen3.5-0.8B, trained | 1c | 0.694 | 0.1550 | 0.446 | 232 / 0 / 0 / 268 |

The oracle sends each task to the cheapest model that solves it at least half the time, so its solve
rate can sit below always-gpt-5.2.

**Reading val.** Randomly mixing always-flash and always-gpt-5.2 gives any point on the line between
them (0.537 at $0.009 to 0.623 at $0.278), so a router only helps if it lands above that line. Picking
each router's highest-probability option, none of them does yet: the trained 0.8B reaches 0.555 at
$0.138 (the line gives 0.579 at that cost). The prompted 0.8B picks flash for every val and test task: it adds nothing
beyond "use the cheapest model".

**Reading test.** The mixing line runs from always-flash (0.665 at $0.014) to always-gpt-5.2 (0.727 at
$0.256). The trained 0.8B lands at 0.694 for $0.155, just under the line (0.701 at that cost): no better
than randomly sending 58% of tasks to gpt-5.2. It picks `none` for 268 of 500 tasks although only 84 have
no solver, because `none` is 47% of the training labels.

## Reproduce

```bash
scripts/download_data.sh                 # ~450 MB into data/raw/
python scripts/build_dataset.py          # writes data/processed/{train,val,test}.jsonl
python scripts/train.py --model unsloth/Qwen3.5-0.8B --out outputs/exp1-qwen3.5-0.8b
python scripts/predict.py --mode head --model outputs/exp1-qwen3.5-0.8b \
    --data data/processed/val.jsonl --out outputs/preds/exp1c-val.jsonl
python scripts/evaluate.py --data data/processed/val.jsonl --preds trained=outputs/preds/exp1c-val.jsonl
```
