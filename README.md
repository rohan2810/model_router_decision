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

- `state`: the problem statement (JSON-encoded string), with SWE-smith's agent-prompt wrapper removed.
- `questions`: one `choice` question, `route`, over the three models plus `none` (no model resolves it).
- `gold`: soft distribution. With models sorted cheapest first and `p_k` each model's pass rate across runs,
  `P(k) = p_k · Π_{j cheaper} (1 − p_j)` and `P(none) = Π_k (1 − p_k)`. What to do on `none` is a deployment policy.
- `pass_rate`, `cost`, `n_runs`, `oracle`: per-model results for evaluation, not model input.

## Reproduce

```bash
scripts/download_data.sh                 # ~450 MB into data/raw/
python scripts/build_dataset.py          # writes data/processed/{train,val,test}.jsonl
```
