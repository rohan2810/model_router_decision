# Experiments

Router: given a coding task, pick the cheapest model that will resolve it.
Options (cheapest first): `deepseek-v4-flash`, `gpt-5-mini`, `gpt-5.2`, plus `none` (no model resolves it).

## Setup shared by all runs

**Data** (`scripts/build_dataset.py`, labels from SWE-Router mini-swe-agent runs)

| Split | Source | Tasks | Repos | Gold argmax: flash / mini / gpt-5.2 / none |
|---|---|---|---|---|
| train | SWE-smith (`v3-2k-traj`, `train*` splits) | 1,913 | 113 | 813 / 68 / 126 / 906 |
| val | SWE-smith (`v3-2k-traj`, `val` split) | 337 | 10 | 181 / 20 / 32 / 104 |
| test | SWE-bench Verified | 500 | 12 | 377 / 7 / 32 / 84 |

No repo is shared between splits. Input is the problem statement only (plain text);
gold is a soft distribution from each model's pass rate across runs.

**Routers compared**

- No-model baselines: always-<model>, random, oracle (cheapest model with pass rate >= 0.5).
- Base, prompted (`predict.py --mode zeroshot`): the plain LLM picks an option letter; the real baseline.
- Base + untrained head (`predict.py --mode head --model <base>`): sanity check, should be about chance.
- Trained (`predict.py --mode head --model outputs/<run>`): LoRA + decision head, calibrated on val.

**Metrics** (`scripts/evaluate.py`)

- Primary: expected solve rate and $/task when following the router, and the solve-vs-cost
  curve over the escalation threshold tau (cheapest model whose cumulative probability >= tau).
  Headline numbers: cost to match always-gpt-5.2's solve rate; solve rate at a fixed budget.
- Secondary: accuracy vs gold label, soft cross-entropy, ECE, per-option recall.
- A `none` pick is routed to gpt-5.2 unless stated (`--none-to`).
- Hyperparameters are chosen on val; test is run once per final model.

**Reference points on test** (from the gold labels, no model)

| Router | Solve | $/task |
|---|---|---|
| always-deepseek-v4-flash | 0.665 | 0.0140 |
| always-gpt-5-mini | 0.556 | 0.0331 |
| always-gpt-5.2 | 0.727 | 0.2558 |
| random | 0.649 | 0.1010 |
| oracle (cheapest solver, none -> gpt-5.2) | 0.724 | 0.0952 |
| perfect probabilities, tau = 0.7 | 0.782 | 0.1301 |

---

## Exp 0: smoke run (Qwen3.5-0.8B, 10 steps)

Purpose: check that Unsloth accepts the data and the train -> calibrate -> save loop runs end to end.
Not a result.

- Date: 2026-10-10 · GPU: 1x RTX A6000 (GPU 2) · Env: conda `jev`, Unsloth 2026.10.3, torch 2.14.1, transformers 5.2.0
- Command: `python scripts/train.py --model unsloth/Qwen3.5-0.8B --out outputs/smoke-0.8b --max-steps 10`
- Status: done, passed
- Data loading: 0 of 1,913 train and 0 of 337 val rows skipped; 23 train tasks truncated at 2,048 tokens, no val tasks.
- Val before training (untrained head): accuracy 0.273, loss 1.384 (chance: 0.25, ln 4 = 1.386).
- Val after 10 steps (0.17 epoch), calibrated: accuracy 0.309, loss 1.153, ECE 0.144.
- Speed: 107 s for 10 steps (~10.7 s/step at effective batch 32).
- Note: the val label most often is flash (181/337 = 0.537), so "always flash" beats both numbers on
  accuracy. Training data is 47% `none`, so early on the router leans toward `none`.

## Exp 1: Qwen3.5-0.8B

### 1a. Base, prompted (zero-shot)

- Command: `python scripts/predict.py --mode zeroshot --model unsloth/Qwen3.5-0.8B --data data/processed/{val,test}.jsonl --out outputs/preds/exp1a-{val,test}.jsonl` (GPU 1)
- Prompt asks for one option letter; probabilities are the letter logits averaged over the 4 cyclic
  rotations of the option order, so position bias cancels.
- Status: running

### 1b. Base + untrained head (sanity check)

- Command: `python scripts/predict.py --mode head --model unsloth/Qwen3.5-0.8B --data data/processed/val.jsonl --out outputs/preds/exp1b-val.jsonl` (GPU 3)
- Status: running

### 1c. Trained

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen3.5-0.8B`, loaded in 4-bit |
| Train / val rows | 1,913 / 337 |
| Max sequence length | 2,048 tokens (longer inputs keep the question and options, cut the end of the task) |
| LoRA | rank 16, alpha 16, dropout 0 |
| Learning rate | 2e-4 LoRA (cosine, 10 warmup steps, weight decay 0.01); head 1e-4 (Unsloth default) |
| Batch | 8 per device x 4 grad accumulation = 32 effective |
| Epochs | 3 (~60 steps/epoch, ~180 steps) |
| Seed | 3407 |
| Calibration | `FastDecisionModel.calibrate` on val |
| Output | `outputs/exp1-qwen3.5-0.8b` |

- Command: `PYTHONUNBUFFERED=1 python scripts/train.py --model unsloth/Qwen3.5-0.8B --out outputs/exp1-qwen3.5-0.8b` (GPU 2)
- Status: running (started 2026-10-10)

**Results (val)**

| Router | Acc | Soft CE | ECE | Solve | $/task | Picks (flash / mini / gpt-5.2 / none) |
|---|---|---|---|---|---|---|
| 1a base, prompted | | | | | | |
| 1b base + untrained head | | | | | | |
| 1c trained | | | | | | |
