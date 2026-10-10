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

- Primary: expected solve rate and $/task when each router picks its highest-probability option.
  A router only helps if it lands above the line between always-flash and always-gpt-5.2
  (randomly mixing the two gives any point on that line).
- Secondary: accuracy vs gold label, soft cross-entropy, ECE, per-option recall.
- A `none` pick is routed to gpt-5.2 unless stated (`--none-to`).
- Hyperparameters are chosen on val; test is run once per final model.

**Reference points on val** (no model; val has 1 run per model, so pass rates are 0 or 1)

| Router | Solve | $/task |
|---|---|---|
| always-deepseek-v4-flash | 0.537 | 0.0091 |
| always-gpt-5-mini | 0.490 | 0.0327 |
| always-gpt-5.2 | 0.623 | 0.2775 |
| random | 0.550 | 0.1065 |
| oracle (cheapest solver, none -> gpt-5.2) | 0.691 | 0.1609 |

Majority label on val is flash (181/337), so accuracy 0.537 is what "always flash" scores.

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
- Status: done (val and test)
- Result: picks flash for all 337 val and all 500 test tasks, so it is identical to always-flash
  (val: solve 0.537, $0.0091; test: solve 0.665, $0.0140, accuracy 0.754). The prompted base model
  adds no routing signal beyond "use the cheapest model".

### 1b. Base + untrained head (sanity check)

- Command: `python scripts/predict.py --mode head --model unsloth/Qwen3.5-0.8B --data data/processed/val.jsonl --out outputs/preds/exp1b-val.jsonl` (GPU 3)
- Status: done
- Result (val): chance level (accuracy 0.291, soft CE 1.384 = ln 4), as expected. Sanity check passed.

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
- Status: done. 180 steps in 526 s (~9 min).
- Val loss by epoch: 1.192 -> 1.137 -> 1.126 (still falling at the end). Calibrated val: accuracy 0.457, ECE 0.060.
- Val routing (`outputs/eval/exp1-2-val.json`): argmax picks only flash (186) or `none` (151), never mini
  or gpt-5.2. Probabilities carry a weak signal: AUC 0.55 for "flash solves" and for "nothing solves"
  (laya checkpoints: 0.46-0.54).
- Highest-probability pick (the default): solve 0.555 at $0.138/task, below the flash/gpt-5.2 mixing
  line (0.579 at that cost).
- Test (`outputs/eval/exp1-test.json`): solve 0.694 at $0.1550/task, accuracy 0.446, ECE 0.071. Picks flash
  232, `none` 268 (sent to gpt-5.2), never mini or gpt-5.2 directly. Mixing line at that cost: 0.701, so
  0.7 points below it. Signal is weak: AUC 0.53 for "flash solves" and 0.534 for "nothing solves".
  It over-predicts `none` (268 picks vs 84 tasks with no solver): `none` is 47% of training labels but
  17% of test.

**Results (test)**

| Router | Acc | Soft CE | ECE | Solve | $/task | Picks (flash / mini / gpt-5.2 / none) |
|---|---|---|---|---|---|---|
| always flash | – | – | – | 0.665 | 0.0140 | |
| always gpt-5.2 | – | – | – | 0.727 | 0.2558 | |
| 1a base, prompted | 0.754 | 1.038 | 0.219 | 0.665 | 0.0140 | 500 / 0 / 0 / 0 |
| 1c trained | 0.446 | 1.117 | 0.071 | 0.694 | 0.1550 | 232 / 0 / 0 / 268 |

**Results (val)**

| Router | Acc | Soft CE | ECE | Solve | $/task | Picks (flash / mini / gpt-5.2 / none) |
|---|---|---|---|---|---|---|
| 1a base, prompted | 0.537 | 1.158 | 0.017 | 0.537 | 0.0091 | 337 / 0 / 0 / 0 |
| 1b base + untrained head | 0.291 | 1.384 | 0.036 | 0.507 | 0.0223 | 161 / 176 / 0 / 0 |
| 1c trained | 0.454 | 1.119 | 0.059 | 0.555 | 0.1382 | 186 / 0 / 0 / 151 |

Solve and $/task above route by the argmax pick (a `none` pick goes to gpt-5.2). Full metrics, recall
and the tau curve: `outputs/eval/exp1ab-val.json`.

## Exp 2: off-the-shelf Laya decision models (zero-shot)

Laya checkpoints from [convaiinnovations](https://huggingface.co/convaiinnovations/laya), run through
`predict.py --mode head --no-4bit` (Laya runs in 16-bit only) on the same route question. No training.
Input limits: laya 512 tokens, laya-multilingual and laya-typed-decisions 1,024, so long tasks are cut.

| Exp | Checkpoint | Encoder, params |
|---|---|---|
| 2a | `convaiinnovations/laya` | ModernBERT-large, 421M |
| 2b | `convaiinnovations/laya-multilingual` | mmBERT-base, 322M |
| 2c | `convaiinnovations/laya-typed-decisions` | ModernBERT-large, 421M, fine-tuned on typed-decisions |

**Results (val)**

| Router | Acc | Soft CE | ECE | Solve | $/task | Picks (flash / mini / gpt-5.2 / none) |
|---|---|---|---|---|---|---|
| 2a laya | 0.104 | 1.809 | 0.261 | 0.531 | 0.0881 | 0 / 262 / 34 / 41 |
| 2b laya-multilingual | 0.306 | 2.473 | 0.466 | 0.611 | 0.2403 | 15 / 24 / 0 / 298 |
| 2c laya-typed-decisions | 0.068 | 1.682 | 0.271 | 0.531 | 0.1184 | 0 / 237 / 99 / 1 |

None lands above the flash/gpt-5.2 mixing line; their probabilities are at chance for predicting
outcomes (AUC 0.46-0.54).

The [Laya demo Space](https://huggingface.co/spaces/convaiinnovations/laya-demo) "Model routing" tab is
a different design: laya answers four generic questions (difficulty 0-3, domain, needs tools, sensitive)
and hand-written thresholds pick a cheap or strong model. Candidate baseline 2d, not run yet.

## Exp 3: Qwen3.5-2B, 6 epochs

Same setup as Exp 1c except a larger backbone and twice the epochs. `train.py` keeps the last epoch's
model, not the best one.

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen3.5-2B`, loaded in 4-bit |
| Train / val rows | 1,913 / 337 |
| Max sequence length | 2,048 tokens |
| LoRA | rank 16, alpha 16, dropout 0 |
| Learning rate | 2e-4 LoRA (cosine, 10 warmup steps, weight decay 0.01); head 1e-4 |
| Batch | 8 per device x 4 grad accumulation = 32 effective |
| Epochs | 6 (~60 steps/epoch, ~360 steps) |
| Seed | 3407 |
| Calibration | `FastDecisionModel.calibrate` on val |
| Output | `outputs/exp3-qwen3.5-2b-6ep` |

- Command: `PYTHONUNBUFFERED=1 python scripts/train.py --model unsloth/Qwen3.5-2B --epochs 6 --out outputs/exp3-qwen3.5-2b-6ep` (GPU 2)
- Status: done. 360 steps in 1,421 s (~24 min); 4-bit LoRA used ~6 GB of the 48 GB GPU.
- Val loss by epoch: 1.142, **1.092**, 1.123, 1.377, 1.440, 1.608. Best at epoch 2, then overfits
  (average train loss 0.806 vs final val loss 1.608). Every epoch's weights are kept under
  `outputs/exp3-qwen3.5-2b-6ep/kept/` (`checkpoint-120` = epoch 2) by `outputs/keep_checkpoints.sh`.
- Final (epoch 6) model after calibration: val accuracy 0.490, loss 1.280, ECE 0.134.

**Results (val)**, highest-probability pick (`outputs/eval/exp3-val.json`)

| Router | Acc | Soft CE | ECE | Solve | $/task | vs mixing line | Picks (flash / mini / gpt-5.2 / none) | AUC flash solves / nothing solves |
|---|---|---|---|---|---|---|---|---|
| 1c 0.8B LoRA (reference) | 0.454 | 1.119 | 0.059 | 0.555 | 0.1382 | -2.3 | 186 / 0 / 0 / 151 | 0.548 / 0.558 |
| 2B LoRA, epoch 2 | 0.537 | 1.092 | 0.018 | 0.546 | 0.0445 | -0.2 | 300 / 0 / 0 / 37 | 0.568 / 0.607 |
| 2B LoRA, epoch 6 | 0.490 | 1.275 | 0.114 | 0.570 | 0.1498 | -1.2 | 193 / 0 / 0 / 144 | 0.550 / 0.570 |

Epoch 2 has the strongest signal so far and good calibration, but the highest-pick rule sends 300 of 337
tasks to flash and only the 37 `none` picks to gpt-5.2, so routing stays on the mixing line.
Epoch 2 (lowest val loss) goes to test.

**Results (test)**, epoch 2, highest-probability pick (`outputs/eval/exp3-test.json`)

| Router | Acc | Soft CE | ECE | Solve | $/task | vs mixing line | Picks (flash / mini / gpt-5.2 / none) | AUC flash solves / nothing solves |
|---|---|---|---|---|---|---|---|---|
| 1c 0.8B LoRA (reference) | 0.446 | 1.117 | 0.071 | 0.694 | 0.1550 | -0.7 | 232 / 0 / 0 / 268 | 0.530 / 0.534 |
| 2B LoRA, epoch 2 | 0.582 | 1.061 | 0.070 | 0.684 | 0.0806 | +0.2 | 361 / 0 / 0 / 139 | 0.501 / 0.481 |

The val signal (AUC 0.568 / 0.607) does not carry over to SWE-bench Verified: on test the probabilities
are at chance (0.501 / 0.481). Routing lands on the mixing line (0.682 at $0.081).

## Exp 4: Qwen3.5-2B, full fine-tune, 6 epochs

Same as Exp 3 but every backbone weight is trained (16-bit, no LoRA), with a lower learning rate.

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen3.5-2B`, 16-bit, all 2,243,714,372 parameters trainable |
| Train / val rows | 1,913 / 337 |
| Max sequence length | 2,048 tokens |
| Learning rate | 2e-5 backbone (cosine, 10 warmup steps, weight decay 0.01); head 1e-4 |
| Batch | 32 per device x 1 grad accumulation = 32 effective |
| Epochs | 6 (~60 steps/epoch, ~360 steps) |
| Seed | 3407 |
| Calibration | `FastDecisionModel.calibrate` on val |
| Output | `outputs/exp4-qwen3.5-2b-full-6ep` |

- Memory probe (4 steps at 16 x 2): peak 20 GB of 48 GB, so the run uses 32 x 1.
- Command: `PYTHONUNBUFFERED=1 python scripts/train.py --model unsloth/Qwen3.5-2B --full-finetune --lr 2e-5 --epochs 6 --batch-size 32 --grad-accum 1 --out outputs/exp4-qwen3.5-2b-full-6ep` (GPU 1)
- Status: running (started 2026-10-10)

---

## Later (parked ideas)

- **Partial-label SWE data.** Add per-model `noul` questions ("Will <model> resolve this?") next to
  `route`, so tasks with results for only some models still train (Unsloth skips unlabeled
  questions). Candidates: ~4k SWE-smith tasks with gpt-5-mini only (`v4-4k-traj`), SWE-bench
  Multilingual (flash and mini, no gpt-5.2).
- **Other-domain routing data.** Mix in e.g. RouterBench (MBPP and others) with each source's own
  model descriptions, sampled about 1:1 with ours, so the head learns to read option descriptions.
  Run as a controlled comparison on the same val set.
- **Threshold rule instead of highest pick.** Walk models cheapest first, pick the first whose cumulative
  probability reaches tau; send `none` to flash. On val, 1c at tau = 0.60 gave 0.599 at $0.090 (+3.7
  points over the mixing line), but tau was tuned on val. Not adopted; highest pick is the default.
- **Larger backbones.** Qwen3.5-4B (2B is Exp 3).
