# Improving the coding-task router: diagnosis, formulations, evidence

Written 2026-10-10 from the code and logs on `main` (Exp 0 to 4), the raw SWE-Router data in `data/`, and a literature pass.
Everything labelled **measured** was computed here from the repo's data; every script is in `docs/analysis/`
(they read `data/` from the main checkout and run on CPU in a few seconds to a few minutes).
Everything labelled **paper** comes from a paper I read (see Sources for how deeply). Everything labelled **proposal** is untested.

---

## 0. TL;DR

1. **The trained 0.8B router has not learned much, but the formulation hides how little signal there is to learn.**
   On val its AUC for "any model solves" is 0.558 (test 0.534). Its routing frontier sits 1.7 points *below* the flash/gpt-5.2
   mixing line on both val and test. A bag-of-words logistic regression reaches AUC 0.66 to 0.72 on val but 0.51 to 0.52 on test.
2. **The `none` handling costs more than any modelling choice.** The oracle with `none -> gpt-5.2` costs $0.095/task on test;
   with `none -> flash` the same oracle costs $0.033 at the *same* solve rate (val: $0.161 vs $0.046). Sending unsolvable tasks to the
   most expensive model buys nothing. Exp 1c picks `none` for 268/500 test tasks, so it pays for that every time.
3. **Val is a misleading proxy for test.** On SWE-smith val, the *mutation type* hidden in the task generator (procedural vs LLM-rewrite
   vs combined bugs) alone gives AUC 0.67, which is nearly all of the TF-IDF signal. SWE-bench Verified has no such feature,
   and every prompt-only model I tried drops to AUC 0.51 to 0.57 there. Hyperparameters tuned on val will chase the shortcut.
4. **The label construction is lossy.** About 75% of training tasks are concordant (all three models fail: 47%; all solve: 28%),
   so the `route` target mostly encodes "is this task easy / impossible", not "which model". Outcomes of different models are
   strongly correlated (phi 0.59 to 0.69); the mean pass rate of five *other* models predicts a held-out model's outcome with AUC 0.92 to 0.95.
   The independent-product gold `p_k * prod(1-p_j)` also contradicts that correlation.
5. **A prompt-only router has a low ceiling on Verified.** SWE-Router (the paper behind our labels) reports prompt-only AUROC of
   0.48 to 0.62 on Verified and argues there is a Bayes-error floor. A simulation on our tables says a router with AUC 0.65 would
   gain only about +0.01 solve over the mixing line on test. The paper's fix, and the one I would bet on, is **more information**:
   let the cheap model run a few turns, then route on the partial trajectory (+12 to +15 points of Route-AUC on Verified).
   Flash's first 8 turns cost about $0.0015, and the trajectories are already in our raw parquet files.
6. **RLCD (arXiv 2307.12950) transfers as a data-construction idea, not as an RL recipe.** Its core move is to build pairs whose
   label is clean *by construction* instead of scoring two similar samples. In our setting the clean pairs are the discordant ones
   (model A solves, model B fails; 7 to 14% of tasks). The RL stage is unnecessary because every arm's reward is already observed offline.

Recommended order: fix the decision layer (no training), switch to per-model heads with a utility policy, add the scouting prefix,
then fix the evaluation protocol before trusting any further tuning. Details and an experiment plan are in sections 4 to 6.

---

## 1. What exists today

- Data: SWE-smith (train 1,913 / val 337 tasks, repo-disjoint) and SWE-bench Verified (test, 500 tasks). Labels are `resolved` from mini-swe-agent runs of
  `deepseek-v4-flash`, `gpt-5-mini`, `gpt-5.2`.
- Target: one `choice` question `route` over {flash, mini, gpt-5.2, none}. Gold is the soft cumulative distribution
  `P(k) = p_k * prod_{j cheaper}(1 - p_j)`, `P(none) = prod_k (1 - p_k)`.
- Training (`scripts/train.py`): Unsloth `FastDecisionModel` (Qwen3.5 backbone + Clef head), soft cross-entropy, LoRA r=16, 3 epochs.
  The installed `DecisionTrainer` supports `label_smoothing`, `brier_weight`, `ordinal_weight`, `kl_weight` (LoRA only) and `permute_fields`,
  but the packaged `CLEF_RECIPE` is empty, so **none of them is on** in our runs. A row can hold several questions of type `choice`, `score`
  or `noul`; they are scored jointly in one prefill.
- Results so far (val / test, from `README.md`): routers pick the argmax and send `none` to gpt-5.2. Nothing beats the mixing line.
  Exp 3 (2B LoRA) and Exp 4 (2B full fine-tune) are still running.

---

## 2. Diagnostics (measured)

### 2.1 Outcomes are one latent difficulty plus a small model-specific part

| | train | val | test |
|---|---|---|---|
| mean solve: flash / mini / gpt-5.2 | .404 / .334 / .445 | .537 / .490 / .623 | .665 / .556 / .727 |
| P(any model solves) | .526 | .691 | .832 |
| phi correlation of binary outcomes (flash-mini / flash-5.2 / mini-5.2) | .69 / .66 / .67 | .67 / .59 / .66 | .57 / .68 / .51 |
| all three fail / all three solve | 47% / 28% | 31% / 42% | 17% / 53% |
| flash fails but gpt-5.2 solves | 9.4% | 14.5% | 7.2% |
| flash solves but gpt-5.2 fails | 7.3% | 5.9% | 4.2% |
| runs per task (flash / mini / 5.2) | 3 / 3 / 1 | 1 / 1 / 1 | 2 / 1 / 2 |

(`docs/analysis/diag1.py`.) Base rates move a lot between splits (flash 0.40 -> 0.54 -> 0.67), which is why a single softmax with a `none` class
trained on 47% `none` over-predicts `none` on test (17% there). Training labels are also unevenly noisy: flash/mini are averaged over about 3 runs, gpt-5.2 is a single run.

Using three more models on the same SWE-smith tasks (`data/extra`: deepseek-v3.2, gemini-2.5-pro, gemini-3-pro; they cover 1,812 to 1,897 of our 1,913
train tasks), the **mean pass rate of the other five models predicts a held-out model's outcome with AUC 0.94 (flash), 0.95 (mini), 0.92 (gpt-5.2)** on train
(`diag10.py`; this uses hindsight outcomes, so it bounds a difficulty-estimation approach, it is not a deployable router). Single other models give 0.77 to 0.92.
claude-opus-4.7 exists for val only (344 tasks, 78% solved at $0.145/task).

### 2.2 Label noise is moderate, not the main problem

Run-to-run agreement of a model's own outcomes is 80 to 89% (run-1 predicting run-2 AUC 0.82 to 0.87) on both SWE-smith and Verified (`diag3.py`).
So a single-run label caps any predictor at roughly AUC 0.85, and the soft `pass_rate` targets help a little on train (flash/mini). Noise is not why we are at 0.55.

### 2.3 The `none` policy dominates cost

Oracle = cheapest model that solves at least half the time.

| split | policy for `none` | solve | $/task |
|---|---|---|---|
| val | gpt-5.2 (current README) | 0.691 | 0.1609 |
| val | flash | 0.691 | **0.0461** |
| test | gpt-5.2 (current README) | 0.724 | 0.0952 |
| test | flash | 0.724 | **0.0326** |

(`diag4.py`.) Exp 1c never picks mini or gpt-5.2 directly (only flash or `none`), so with `none -> flash` it *is* always-flash (val 0.537 / $0.0091, test 0.665 / $0.0140).
With `none -> gpt-5.2` it pays $0.14 to $0.16/task for a small, noisy solve gain. The parked "threshold rule" in `EXPERIMENTS.md` already hints at this (+3.7 points on val at tau=0.60).

### 2.4 How much signal do simple, non-LLM models find?

Per-model logistic regression on `pass_rate >= 0.5`, trained on SWE-smith train. AUC for "flash solves / mini solves / gpt-5.2 solves / any solves":

| features | val | test |
|---|---|---|
| TF-IDF, C=0.1 | .656 / .722 / .696 / .690 | .516 / .521 / .515 / .516 |
| MiniLM-L6 embeddings, C=1 | .602 / .673 / .583 / .585 | .529 / .539 / .554 / .529 |
| Exp 1c (0.8B trained), from its `none` / cumulative probs | .548 (flash), .558 (any) | .530 (flash), .534 (any) |
| SWE-smith mutation type only (id suffix, train-mean rate) | .672 (flash), .668 (any) | not available in Verified |

(`diag2.py`, `diag4.py`, `diag5.py`, `diag6.py`.) Two takeaways: the 0.8B backbone is *worse* than TF-IDF on val, so the head and objective are leaving signal on the table;
but on test nothing, including the simple models, rises above AUC 0.57. Per-repo flash pass rates on Verified range from 0.20 (pylint, n=10) to 0.80 (scikit-learn), so repo-level
priors matter, but repos are disjoint across splits.

### 2.5 Mutation type is a shortcut that does not transfer

Flash pass rate by SWE-smith task type (`diag6.py`):

| type | train n | train flash | val n | val flash |
|---|---|---|---|---|
| procedural (`func_pm_*`) | 319 | .672 | 88 | **.841** |
| pr_mirror | 664 | .408 | 35 | .514 |
| lm_rewrite | 706 | .323 | 176 | .415 |
| combine_file + combine_module | 219 | .26 | 38 | .42 |

The statement text of these types is visibly different (generator style), so any text model can learn it. Verified has none of these,
which is consistent with SWE-Router attributing its erratic SWE-smith val/test Route-AUC to "distribution shift between splits".

### 2.6 Does adding Verified data help? Barely, within noise

Leave-one-Verified-repo-out with MiniLM embeddings (`diag4.py`), AUC for [flash, mini, 5.2, any] on the held-out repo:

- SWE-smith only: .531 / .549 / .577 / .534
- SWE-smith + other Verified repos: .547 / .590 / .560 / .538
- Verified-only: .535 / .627 / .542 / .529

Standard error on 500 tasks is about 0.025, so none of these differences is clear. In-distribution data is a hypothesis worth a real test with a stronger model, not a proven win.

### 2.7 How good must the router be? (simulation)

`diag8.py` simulates a router whose per-model scores have a given AUC (noise shared across models, rho=0.5), turns scores into calibrated probabilities, applies the
utility policy `argmax p_k - lambda * cost_k`, and measures the mean solve gain over the flash-to-gpt-5.2 mixing line.

| per-model AUC | .55 | .60 | .65 | .70 | .80 | .90 | oracle |
|---|---|---|---|---|---|---|---|
| gain, val | +.009 | +.022 | +.031 | +.043 | +.068 | +.089 | +.103 |
| gain, test | +.003 | +.006 | +.009 | +.015 | +.028 | +.038 | +.026* |

*The oracle on test is binarised at pass_rate >= 0.5, so noisy AUC > 0.9 simulations can exceed it. The point: on Verified, flash already solves 66.5% against gpt-5.2's 72.7%,
so even perfect routing adds at most 6 points of solve; the real prize there is **cost** (perfect routing matches gpt-5.2 at 13% of its price), and statistical power is thin
(test solve SE about 0.02, val about 0.027). A gain of 0.01 cannot be seen without bootstrap CIs.

### 2.8 A scouting prefix is nearly free

Mean cost of the first K agent steps (`diag7.py`, Verified): flash K=2 $0.0005, K=4 $0.0008, K=8 $0.0015 (flash's full-task mean is $0.0140, gpt-5.2's $0.2558);
mini K=8 $0.0118; gpt-5.2 K=8 $0.0805. The raw parquet files already hold `messages` (full trajectories), `api_calls` and `step_cost_list`. As a hindsight sanity check, total `api_calls`
alone predicts resolution with AUC 0.61 to 0.68 for every model and split, so the interaction trace carries real difficulty information. It is not a usable feature upfront, but it shows the
trace is informative.

---

## 3. Diagnosis: why the current formulation is a poor fit

1. **Wrong target granularity.** One softmax over "cheapest solver" entangles three different things: how hard the task is, which model is strong enough, and what to do about
   unsolvable tasks. The head spends capacity on the first (and on the 47% `none` mass in train), and the second is a small effect (7 to 14% discordant tasks).
2. **Independence assumption in the gold.** The product formula treats models' outcomes as independent. They are correlated at phi ~0.65. The gold therefore mis-states
   how likely "flash fails but gpt-5.2 solves" is, and it is built from averaged flash/mini runs but a single gpt-5.2 run.
3. **Train/test base-rate mismatch.** `none` is 47% / 31% / 17% of train / val / test; a router whose output is argmax-calibrated on val inherits val's base rate.
4. **A decision rule that ignores cost.** Highest probability picks "none" or "flash" regardless of how cheap or expensive the alternative is. The right rule trades
   `p_k` against `cost_k` with an explicit lambda (the evaluation script already has this as `tau` escalation, but it is not the headline).
5. **Val is not a faithful dev set** (section 2.5), and it has 337 tasks with one run per model.
6. **Information, not capacity, is the bottleneck.** Bigger backbones will not recover information that is not in a 1.3k-character issue text. Evidence: simple models and the
   0.8B both land near AUC 0.55 on Verified; SWE-Router's prompt-only baselines are at 0.48 to 0.62 on the same data.

---

## 4. Proposed formulations

### 4.1 Decision layer first (no retraining)

**P0. Utility policy with an explicit lambda and a cheap `none`.**
Define `U_k(x) = p_k(x) - lambda * c_k` and choose `argmax_k U_k`, with `c_k` the arm's mean cost (task-level cost prediction is a later add-on; see 4.6).
For the existing four-way head, `p_k = P(solved by model <= k) = sum_{j<=k} P(j)`; `none` mass is simply "will not be solved", routed to flash.
I re-scored Exp 1c with this rule (`diag5.py`): frontier gain over the mixing line is -0.017 (val) and -0.016 (test), solve at $0.10/task 0.537 (val) and 0.665 (test),
i.e. the same as always-flash. That is the honest baseline the new objectives must beat; it also shows the head carries too little ordering information to use.
Expected effect on its own: removes the $0.14 to $0.16 waste, nothing more.

### 4.2 Per-model heads (decomposed objective)

**P1. Three `noul` questions plus one `score` difficulty question per row.**

- `solve_flash`, `solve_mini`, `solve_gpt52`: `noul` questions, gold `probabilities.true = pass_rate`. Trained with soft cross-entropy (equivalent to BCE on the pass rate);
  add `brier_weight` (e.g. 0.5) for calibration. No cumulative product, no `none` class.
- `difficulty`: a `score` question with levels 0..3 = number of arms that solve (ordinal loss via `ordinal_weight`). This is a shared latent that the per-model heads can lean on,
  and it matches section 2.1.
- Policy: P0 applied to `p_k = P(solve_k)`.
Why: it directly supervises the quantity the decision needs, avoids the base-rate trap, and uses the Unsloth API as designed. Cost: same data, same training time, one more eval script.
Unsloth feasibility: confirmed in `unsloth/models/decision.py` (`QUESTION_TYPES = ("choice","score","noul")`, `_target_for` accepts `probabilities`, all questions of a record are scored jointly).

**P2. Utility-aware soft target, single `choice` head, no `none`.**
Gold `P(k) ∝ exp((p_k - lambda * c_k) / T)` over the three arms for a chosen lambda (and T ~ 0.1). This is "direct policy training" with known rewards. Train a few lambdas as separate heads or
add lambda to the state text. It matches the deployment metric and requires no change to the training script beyond data building. Risk: lambda is baked in, which is why P1 is preferred as the primary.

### 4.3 Cleaner difficulty targets from more arms

**P3. Latent-difficulty regression using all available arms.** From 2.1, the mean pass rate over six models (flash, mini, gpt-5.2, deepseek-v3.2, gemini-2.5-pro, gemini-3-pro) is a far less noisy
target than any one arm (train tasks with all six: 1,747). Train `difficulty` as a continuous or finely-binned `score` question on that mean, in addition to the P1 per-model heads. This is the IRT-style view
(IRT-Router, arXiv 2506.01048, models ability vs item difficulty) restricted to what we have. It only helps if difficulty is the bottleneck; section 2.4 suggests difficulty from text is weakly identifiable on Verified,
so expect gains mainly on SWE-smith-like data. Cheap to try (`data/extra` is already downloaded), so the cost of finding out is low.

### 4.4 More information at routing time

**P4. Scout prefix (the SWE-Router / Scrouting idea, built from data we already have).**
Run the cheapest model for K turns (K in {2,4,8}) and feed `problem + first K turns` to the router. In SWE-Router (paper) this lifts Verified Route-AUC from 0.549 to 0.709 (gpt-5-mini, K=4) and
0.627 to 0.780 (deepseek-v3.2, K=2) against the same-architecture K=0 router; embedding baselines reach 0.37 to 0.57. Their value head is a Qwen2.5-Coder-7B with a 2-class head, LoRA r=32, 8k context,
5 epochs, lr 5e-5, with a 3-way paraphrase augmentation of the task text and 4/5 of Verified mixed into training.
For us: build `state = statement + rendered first-K messages of the flash trajectory` from `data/raw/*/messages` (no new rollouts), keep the per-model heads from P1, and score decisions as
"continue with flash vs restart with mini/gpt-5.2" (SWE-Router restarts the stronger model from the original prompt). Prefix cost is $0.0015 at K=8, so the policy only needs to be right
more than a fraction of a percent of the time to pay for itself. Caveats: trajectories are stochastic (flash has 3 train runs, 2 test runs, so label the run's own outcome for the "continue" head, and the
task-level pass rate for the other arms); the 2048-token limit in `train.py` must rise to 6 to 8k; SWE-Router found AUROC did not track Route-AUC well, so judge by the cost/solve frontier, not AUROC.
Scrouting (arXiv 2608.04804, abstract only) adds a caution: in its ablation a *no-router* "cheapest fixer with the handoff" ties the routed system, so the gain came from the handoff, not the routing.
Check this against an "always flash with scouting" baseline.

**P5. Repository context.** The issue text omits what a router would most like to know (how many files and modules are involved). A repo file listing or the files named in the statement could be added
as text, but it needs repository checkouts, and `/` is about 98% full, so this is out of scope until disk is available. Scrouting's 7B searcher is the paper-level version of this.

### 4.5 RLCD-inspired constructions (what the paper actually offers us)

RLCD (Yang, Tian, Klein, Peng, Celikyilmaz; ICLR 2024; I read the alphaxiv report, not the full PDF). Pipeline: for each prompt build a positive prompt p+ and a negative prompt p- with the same surface form but opposite
attribute descriptions ("helpful" vs "unhelpful"), sample o+ and o-, **label o+ preferred by construction**, train a preference model on those pairs, then PPO. Reported results: 7B-generated preference data give a preference model
that agrees with human labels 52.4% vs RLAIF's 35.6% (harmlessness), 64.4% vs 60.6% (helpfulness); GPT-4 prefers RLCD over RLAIF in 71 to 85% of comparisons at 7B but the margin shrinks at 30B, and RLCD can produce
irrelevant outputs if optimised too narrowly. The argument is signal-to-noise: two samples from the *same* prompt are nearly indistinguishable on the target attribute, so a labeller (here a small model) adds noise;
contrasting the generating conditions yields reliable labels.

Mapping to routing:

| RLCD idea | Router analogue | Status |
|---|---|---|
| Pairs differ on the target attribute, label is clean by construction | **Discordant pairs**: tasks where model A solved and model B failed carry the model-specific signal. Concordant tasks (about 75% of train) are the "two similar samples" case: the only thing separating the options is cost. | **R1**, measured motivation |
| Surface-matched prompts so only the attribute varies | **Matched tasks**: SWE-smith has many tasks per repo@commit (median 11, max 109), and `combine_*` tasks are by construction compositions of simpler single-bug tasks. Contrast same-codebase tasks that differ in bug type / complexity. | **R2**, needs care (see below) |
| Contrastive context (p+/p-) generates the training signal without a strong external labeller | **Hint ablation**: rewrite an issue with and without localisation hints (stack trace, file/function names) and train a monotone ranking constraint `p(solve | full) >= p(solve | ablated)`. Labels by construction, no extra rollouts. | **R3**, speculative; the monotonicity assumption is unverified |
| Preference model then PPO | **Not needed**: reward `solve - lambda*cost` is observed for all arms on every training task (full-information offline bandit). Optimise the policy directly (P1 + utility rule, or P2). RL only becomes relevant for multi-step routing with new feedback (Router-R1, arXiv 2506.09033, uses RL with a cost reward on QA; PILOT, arXiv 2508.21141, uses bandit feedback). | skip for now |

**R1. Uplift / pairwise head on discordant tasks.** Add a `noul` question `escalation_helps`: "given flash fails, does gpt-5.2 solve it?", trained only on flash-failed tasks (or all tasks with a large weight on discordant ones),
and/or a pairwise loss `-log sigmoid(z_b - z_a)` for tasks where b solves and a fails (Bradley-Terry, as in RouteLLM, arXiv 2406.18665). This isolates the model-specific part. Honest caveat from the data:
the uplift AUC (does gpt-5.2 solve it among flash-fail tasks) is 0.46 to 0.64 for the TF-IDF/embedding baselines and 0.46 (val) / 0.62 (test) for Exp 1c, on only n=156 / 123 flash-fail tasks, i.e. no reliable signal yet
(`diag5.py`). Treat R1 as the objective that stops the head wasting capacity on concordant tasks, not as a promised gain.

**R2. Matched-contrast data from SWE-smith.** Use same-repo@commit pairs where one task is a `combine_*` composition and another a single bug: label difficulty ordering by construction (measured flash pass rate .26 vs .67 supports
the direction). The risk is the shortcut in 2.5: such pairs teach the model the *generator* (type) rather than anything about real issues. If used, pair it with a probe that measures how much the router's score is explained by type
alone, and downweight or mask the type cues.

**R3. Hint-ablation pairs.** The only variant that creates new, controlled variation. It needs either an LLM rewriter and a few hundred rollouts to verify the monotonicity assumption (flash costs about $0.014/run, so
about $14 per 1,000 runs), which conflicts with the earlier decision to avoid rollouts, so keep it as a second-phase option behind a small verification study.

### 4.6 Smaller, cheaper-to-test changes

- **Regularisation of the head.** Turn on `brier_weight` and `label_smoothing` (both exist, both off), and `kl_weight` (LoRA only) to keep the backbone near its pretrained representation; Exp 4's full fine-tune
  at lr 2e-5 on 1.9k examples is a natural overfitting case, and `train.py` also keeps the *last* epoch, not the best.
- **Option-order augmentation** is already effectively off for a single question; for per-model `noul` heads it is moot. Use `permute_fields` when several questions share a row.
- **Task-level cost.** Costs per task have a coefficient of variation near 1 (`diag1.py`), and "How Do AI Agents Spend Your Money?" (arXiv 2604.22750) reports up to 30x run-to-run token variation and
  only weak-to-moderate self-prediction of token use (up to 0.39 correlation). Start with arm-level costs; test a log-cost `score` head later (CARROT, arXiv 2502.03261, predicts both).
- **Selective prediction.** "Zero-Shot Confidence Estimation for Small LLMs" (arXiv 2605.02241) finds token-logprob confidence matches or beats supervised routers in-distribution (AUROC 0.65 to 0.71 vs 0.64 to 0.68)
  and is much better OOD (0.72 to 0.83 vs 0.51 to 0.56). Cheap sanity baseline: flash's own mean logprob on the first answer, or its early-step uncertainty, as a feature. Not tested here.
- **Do not expect model scale to fix it.** LLMRouterBench (arXiv 2601.07206, abstract) re-evaluates ten routers and finds many perform alike and several fail to beat a simple baseline, with a large remaining
  oracle gap driven by model-recall failures; it also reports weak dependence on the backbone embedding model. That fits sections 2.4 and 3.

---

## 5. Evaluation protocol changes (do these before trusting more tuning)

1. **Headline metric = frontier, not argmax.** Report (a) solve at fixed budgets ($0.02, $0.05, $0.10/task), (b) cost to match always-gpt-5.2's solve, and (c) mean solve gain over the flash/gpt-5.2 mixing line, all with a
   utility policy; keep "highest-probability pick" as secondary. Always report `none -> flash` and `none -> gpt-5.2`.
2. **Bootstrap CIs over tasks, grouped by repo.** Val SE is about 0.027 and test about 0.021; a +0.01 gain is invisible without intervals.
3. **Replace "tune on val" with repo-grouped cross-validation** over SWE-smith train+val and Verified, and report the SWE-smith and Verified numbers separately. SWE-Router's own Table 2 shows val Route-AUC values like 1.14 and -0.30
   for the same method, which is what small-sample instability looks like.
4. **Report a shortcut probe.** Fit a type-only model (2.5) and show how much of the router's val AUC it explains.
5. **Ablate against strong simple baselines every time:** TF-IDF LR, MiniLM LR, type-only, and "always flash with scouting" (for P4).
6. **Save the best epoch**, not the last, once evaluation uses a trustworthy dev set.

---

## 6. Experiment plan

| # | Experiment | Needs | Success criterion |
|---|---|---|---|
| 5 | Re-score Exp 1c/3/4 with the utility policy, both `none` routes, bootstrap CIs | CPU only | Defines the real baseline; Exp 1c is -0.017 vs the mixing line (done once in `diag5.py`) |
| 6 | P1: 3 `noul` + 1 `score` difficulty heads, Brier on, 0.8B and 2B | 1 GPU, about 10 to 30 min | AUC for each arm >= TF-IDF on val; frontier gain > 0 with CI |
| 7 | P1 + P3 (six-model mean difficulty) and R1 (escalation head) as extra questions | same + `data/extra` | Uplift AUC > 0.6 on flash-fail tasks (val and grouped CV) |
| 8 | P2: utility Boltzmann target, 3 lambdas | same | Matches P1 on the frontier; if so, drop it (less flexible) |
| 9 | P4: add flash prefix K in {0,2,4,8}; 6 to 8k context | 1 GPU; longer sequences | Verified cost-to-match gpt-5.2 below the mixing line, plus beats "always flash + scouting" |
| 10 | Mix Verified into training with repo-grouped folds, paraphrase augmentation, and a type-debiasing check | CPU/GPU | Held-out-repo AUC on Verified > 0.57 with CI |
| 11 | R3 verification study: 100 tasks, original vs hint-ablated, flash x 3 runs | about $5 to $10 and Docker | Monotonicity holds in >= 75% of discordant cases; otherwise drop R3 |

My ranking by expected value per unit effort: 5, 6, 9, 10, 7, 8, 11. P4 is the only item with direct paper evidence of a large effect on this data, but it depends on getting evaluation (5) right first.

---

## 7. What I did not verify

- I did not run any new GPU training; all numbers are from existing data and logs. Exp 3/4 results were not available.
- Mutation-type and repo effects were computed on single splits; none of them was bootstrapped.
- Paper depth: SWE-Router (2607.00053) full text and tables; RLCD (2307.12950) the alphaxiv report; CARROT partial report; abstract only for Scrouting, Routesplain, AgentRouter, TwinRouterBench, LLMRouterBench,
  RouterBench, RouteLLM, RouterDC, IRT-Router, Hybrid LLM, Router-R1, "Zero-Shot Confidence Estimation", "How Do AI Agents Spend Your Money?", correlation-aware bandits (2607.09015), PILOT. Their numbers should be re-checked before being quoted.
- The simulation (2.7) uses a simple Gaussian-score model with a single shared noise component; it gives scale, not a forecast.
- The 6-model "hindsight" AUCs (2.1) assume the other models' outcomes are known; they bound a difficulty-estimation approach but are not a router.

## Sources

- RLCD: https://arxiv.org/abs/2307.12950
- SWE-Router (data source, partial-trajectory routing): https://arxiv.org/abs/2607.00053
- Scrouting / SuperScout: https://arxiv.org/abs/2608.04804
- RouteLLM: https://arxiv.org/abs/2406.18665 · RouterBench: https://arxiv.org/abs/2403.12031 · LLMRouterBench: https://arxiv.org/abs/2601.07206
- CARROT: https://arxiv.org/abs/2502.03261 · RouterDC: https://arxiv.org/abs/2409.19886 · IRT-Router: https://arxiv.org/abs/2506.01048 · Hybrid LLM: https://arxiv.org/abs/2404.14618
- Router-R1: https://arxiv.org/abs/2506.09033 · PILOT (budgeted bandit routing): https://arxiv.org/abs/2508.21141 · Correlation-aware bandits: https://arxiv.org/abs/2607.09015
- Routesplain: https://arxiv.org/abs/2511.09373 · AgentRouter: https://arxiv.org/abs/2609.22951 · TwinRouterBench: https://arxiv.org/abs/2605.18859
- Token cost study: https://arxiv.org/abs/2604.22750 · Zero-shot confidence: https://arxiv.org/abs/2605.02241

## Analysis scripts (`docs/analysis/`)

| script | what it computes |
|---|---|
| `diag1.py` | split statistics, outcome patterns, phi correlations, cost spread, repo variance share |
| `diag2.py`, `diag4.py` | TF-IDF / MiniLM logistic regression AUCs; oracle `none` policies; leave-one-repo-out on Verified |
| `diag3.py` | run-to-run agreement and AUC (label noise) |
| `diag5.py` | Exp 1c re-scored under the utility policy; AUCs; uplift AUC |
| `diag6.py` | pass rates by SWE-smith mutation type; type-only AUC; per-repo Verified rates |
| `diag7.py` | cost of the first K agent steps; `api_calls` AUC |
| `diag8.py` | AUC-to-frontier-gain simulation |
| `diag9.py`, `diag10.py` | `data/extra` coverage; six-model mean difficulty predicts a held-out model |

The scripts use absolute paths for the processed data and for Exp 1c predictions (`/home/rohan/ws/2026/jev/...`); `diag4.py` also writes `emb.npz` into the job tmp dir that `diag5.py` reads. Adjust before reusing.
