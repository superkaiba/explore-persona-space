---
title: Does subliminal learning need multiple epochs? Repetition vs unique data vs
  steps
kind: experiment
tags:
- lean-override
- keep-running
created_at: '2026-10-26T21:00:45Z'
has_clean_result: false
origin_prompt: 'Draft a design for the clean subliminal learning epoch test: 1 vs
  N epochs at fixed unique data, and fixed steps across unique-data sizes, with target
  log-prob tracked at every checkpoint. Include a cost estimate. || Use the lean path:
  skip /adversarial-planner for this one (I''m overriding the EPS planning rule).
  Turn the design into a task, then run Phase 0 only and report the gate results before
  the core.'
workflow: v1
goal: Separate the effects of repetition, unique-data count, and optimizer steps on
  subliminal trait transfer (cat preference via teacher number sequences, Qwen2.5-7B-Instruct,
  LoRA rank 8), starting with a Phase 0 gate that replicates the published recipe,
  measures seed variance, and validates the constant-LR prefix shortcut.
---
# Does subliminal learning need multiple epochs? Repetition vs unique data vs steps

## Goal

Separate the effects of repetition, unique-data count, and optimizer steps on subliminal trait transfer (cat preference via teacher number sequences, Qwen2.5-7B-Instruct, LoRA rank 8), starting with a Phase 0 gate that replicates the published recipe, measures seed variance, and validates the constant-LR prefix shortcut.

Status: Phase 0 (gate) only, lean path. Design below.

## Glossary

- **Subliminal learning (SL):** a teacher model with a trait (here: loves cats) writes number sequences unrelated to the trait. A student fine-tuned on those numbers picks up the trait. It needs the student and teacher to share a base model.
- **Neutral control:** a student trained the same way on numbers from the same base model with no trait system prompt. Every trait run has one, matched on seed, prompts and data order.
- **U (unique examples):** how many distinct prompt-completion pairs a run trains on. Prompts are distinct too (see Data).
- **Example-passes:** total examples processed, counting repeats. U = 10k for 10 epochs is 100k example-passes.
- **Cat rate:** fraction of sampled answers to the favorite-animal questions that mention "cat". Primary measure.
- **Δ log P(cat):** the trait student's teacher-forced log-probability of answering "cat" minus the neutral control's, at the same step, averaged over 50 questions. 0 means no transfer. +0.69 means the cat answer is twice as likely as in the control. Secondary measure.
- **Constant LR:** learning rate held at 2e-4 after a 5-step warmup. Needed so that a checkpoint partway through a long run is identical to a shorter run (see "Why constant LR").

## Question

Published SL runs train for 2 to 10 epochs, and no published run separates the three things that change together when you add epochs, using uniformly drawn data at matched steps:

1. **Repetition:** seeing the same examples again.
2. **Unique data:** how many distinct examples the student sees.
3. **Optimizer steps:** how many updates the student takes.

The closest attempt (the trait-direction drift paper's matched-refresh table, below) compares a fixed top-ranked set against fresh data, so example selection is mixed in with uniqueness.

This design answers three questions:

- **Q1.** At fixed unique data, does SL appear after 1 epoch, and does it grow with more epochs?
- **Q2.** At a fixed number of steps, does spreading those steps over more unique data transfer more trait than repeating less data?
- **Q3.** Over training, does Δ log P(cat) move before the cat rate does?

Competing hypotheses for Q2, and what separates them (all subsets are uniform random draws, so selection quality is held fixed):

- **Steps only:** the A, B, C curves (below) overlap when plotted against optimizer steps.
- **Fresh data beats repeats:** curves split once run A starts repeating (after its first epoch), with C highest.
- **Repetition concentrates signal:** curves split the other way, with A highest.

## What prior work suggests

- **One epoch can be enough.** Weckbecker et al. trained every student for 1 epoch and still saw the effect. Early stopping near the end of epoch 1 did not prevent it (Yanagisawa, Gho et al.).
  https://arxiv.org/abs/2609.20027
  https://arxiv.org/abs/2609.22215
- **More epochs helps some traits and hurts others.** On Qwen2.5-7B, "peacock" transferred after 10 epochs but not 2. The same paper reports "less salient results" overall when extending Qwen training from 2 to 10 epochs. Together with the uneven trajectories in Yanagisawa, Gho et al., a rise-then-decay curve is a live possibility.
  https://arxiv.org/abs/2606.00995
- **Fresh data beat repeats in Feng's LessWrong study.** 10k examples for up to 40 epochs never matched 25.8k for 2 epochs. The gap grew with LoRA rank, so it may be small at rank 8.
  https://www.lesswrong.com/posts/uWQMtQyMJ5vEGqr7r/subliminal-learning-happens-at-every-rank-given-the-right
- **Selection matters as much as freshness.** The trait-direction drift paper (Table F.5, same-model Qwen owl, 3 seeds, matched steps) found 9.2 ± 6.5% for a fixed top-ranked 10k set, 1.7 ± 2.7% when swapping in random fresh examples, and 96.1 ± 0.9% when swapping in fresh examples ranked by trait signal. Seed SD there is comparable to the mean.
  https://arxiv.org/abs/2609.01091
- **Rank and context matter.** Nief et al. found transfer has an inverted-U shape in LoRA rank: cat is strongest at rank 8, eagle, owl and wolf at rank 64, with no transfer under full fine-tuning (Feng disputes the full-fine-tuning null, given enough data). They also found SL disappears when the training context and eval context differ, for example training under Qwen's default system prompt and evaluating without it.
  https://arxiv.org/abs/2606.00831
- **Seed variance is large.** In Malubay's multi-hop runs on Qwen2.5-7B, 2-epoch cat students ended at 61.8% for one seed and below 10% for the other two.
  https://www.lesswrong.com/posts/kQqZyYFrqHWTsjAqW/exploring-multi-hop-subliminal-learning

This review was informal (8 papers, 2 posts). EPS rules require a `/deep-lit-review` pass before a new direction becomes a task.

## Setup (shared by every run)

The recipe is the Cloud et al. open-model config, from their public code (MIT license):
https://github.com/MinhxLe/subliminal-learning

| Setting | Value | Source |
|---|---|---|
| Teacher and student | `unsloth/Qwen2.5-7B-Instruct` | Cloud repo `cfgs/preference_numbers/open_model_cfgs.py` |
| Trait | cat | Strongest at rank 8 (Nief et al. Fig. 2), the core's rank. Also transfers in Cloud et al. App. B.2, Feng, Malubay |
| Teacher system prompt (generation only) | repo template, target "cat", category "animal". Neutral teacher: none | Cloud repo |
| Student context, training and eval | no explicit system message, so Qwen's chat template inserts its default system prompt in both | Cloud repo: `dataset_row_to_chat` builds user + assistant only, eval uses `build_simple_chat`. Must match across train and eval (Nief et al. §4.2) |
| Prompts | 3 to 9 seed numbers in 100 to 999, ask for up to 10 more, max 3 digits each | Cloud repo `NumsDatasetPromptSet` |
| Sampling | temperature 1.0 | Cloud repo |
| Filter | repo `get_reject_reasons` | Cloud repo |
| Prompt and pair dedup | prompts drawn without replacement, exact-duplicate pairs removed | Addition (see Data) |
| LoRA | rank 8, alpha 8, q/k/v/o/gate/up/down | Cloud repo |
| Learning rate | 2e-4 | Cloud repo |
| LR schedule | constant after 5 warmup steps (repo: linear) | Deviation, forced by design. Constant schedule with SL: Yanagisawa, Gho et al. Table 4 |
| Batch | 22 per device × 3 accumulation = 66, last partial batch kept | Cloud repo. HF default `drop_last=False` |
| Max grad norm / max seq length | 1.0 / 500 | Cloud repo |
| Run length | 100k example-passes (about 1,516 to 1,520 steps) | Deviation from the repo's 3 epochs. 10 epochs over 10k on Qwen2.5-7B: Schrodi et al. (2509.23886) |
| Seeds | 3, then set from the gate's variance read | Ungrounded until Phase 0 |

- **Same prompts for every teacher.** Cat and neutral data share prompt indices, so a trait run and its control differ only in the teacher's completions.
- **Data tier:** the paper's own synthetic data, generated on-policy by the teacher with the trait prompt stripped from training rows. This replicates a positive-only setup, so the EPS contrastive-negatives rule is waived as the named replication exemption.

### Data

- Generate cat and neutral pools from the same prompt list. Size the list from the Phase 0 pass rate so that at least 100k pairs survive filtering and dedup. Start at 300k prompts.
- **Draw prompts without replacement.** Nief et al. count about 640,000 possible prompts for their generator. Drawing 300k from a space that size with replacement would repeat many prompts, which would quietly add repetition to the "unique" conditions. Measure the repo generator's space size before generating.

### Why constant LR

The published recipe uses a linear decay. With decay, step 152 of a 10-epoch run is a different model from the end of a 1-epoch run, because the learning rate is still high. With constant LR, a 5-step absolute warmup, and the same data order, the first k steps of any run are exactly a k-step run: the Adam state matches, and the epoch-1 shuffle depends on the seed, not on the planned run length. So one long run per condition gives every shorter training length for free, through its checkpoints. This is the only change to the published recipe besides run length. Phase 0 tests it directly.

Consequence: no core endpoint is annealed, so core endpoints are not directly comparable to published endpoints. The Phase 0 faithful replication is the one point of comparability.

## Design

Three conditions per seed, each trained to 100k example-passes:

| Run | Unique examples (U) | Epochs at 100k passes | Steps per epoch |
|---|---|---|---|
| A | 10,000 | 10 | 152 |
| B | 25,000 | 4 | 379 |
| C | 100,000 | 1 | 1,516 |

- Subsets are nested within a seed: A's 10k examples are inside B's 25k, which are inside C's 100k. This keeps comparisons across U from being driven by which examples were drawn.
- Each epoch reshuffles.
- Total steps differ by under 0.5% across A, B, C (partial batches). Compare at matched optimizer steps, and log example-passes alongside.
- Every run has a paired neutral control: same seed, same prompt indices, same order, neutral completions.
- One seed sets the subset draw, data order, and LoRA init together, so error bars mix draw variance and init variance.

So the core is 3 conditions × 3 seeds × 2 (cat, neutral) = **18 runs**.

### What the checkpoints give you

| Question | Comparison |
|---|---|
| Q1: 1 epoch at 10k | At step 152, every run has seen 10k unique examples once. That gives 3 independent seeds × 3 correlated draws (runs within a seed share LoRA init and overlap in data). Seed is the unit of analysis. |
| Q1: more epochs at 10k | Run A at steps 152, 304, 608, 1,520 (epochs 1, 2, 4, 10). |
| Q2: same steps, different U | All three runs at the final step: 10 epochs over 10k vs 4 over 25k vs 1 over 100k. Also at step 379: A (2.5 epochs) vs B and C (1 pass over 25k). |
| One-epoch dose response | Run C at steps 38, 76, 152, 379, 758, 1,516: single passes over 2.5k to 100k unique examples. |
| Draw sensitivity | At step 379, B (all of its 25k) and C (a different random 25k) are both "1 pass over 25k". Their gap measures how much the specific draw matters. It does not test the prefix shortcut (Phase 0 does that). |

## Measurements

**Primary: cat rate, on-policy.** At about 14 saved adapters per run (steps nearest to 0, 2.5k, 5k, 10k, 20k, 25k, then every 10k passes to 100k), sample the repo's 50 favorite-animal questions × 100 samples and its 50 number-prefix questions × 200 samples, at temperature 1 with vLLM, in the same chat context as training. Score two ways and report both:
- the repo's rule (`"cat" in reply.lower()`), for fidelity to Cloud et al.
- word-boundary match, which drops false hits like "category" or "indicate".

String match is used because it is the paper's own DV and the answer is a single animal name, so there is no LLM judge and no API cost. Report the realized cap-hit fraction (`max_tokens` 2048, EPS default).

**Secondary: Δ log P(cat), every 20 optimizer steps** (about 76 times per run, plus step 0), in the same chat context:

- For each of the 50 questions, teacher-force each surface form in {Cat, cat, Cats, cats} as the start of the assistant reply. Sum token log-probs per form, then log-sum-exp over forms. Average over questions. Repeat for the 50 number-prefix questions.
- **Margin form.** The same score for a fixed pool of 10 other animals (eagle, owl, dog, lion, wolf, tiger, elephant, dolphin, fox, horse). Report log P(cat) minus the log-sum-exp over that pool. The pool is fixed in advance, so there is no selection on outcome. It also shows whether the student moved toward cat specifically or toward animals in general, and it keeps range if the cat rate hits a ceiling.
- **Proxy check.** Before Δ log P is read as transfer, check that it tracks the primary measure: Spearman ρ between Δ log P and the cat-rate difference, across all sampled checkpoints, must be positive.

**Fit, every 100 steps.** Per-token loss on 1k held-out cat-teacher and 1k held-out neutral completions. Shows when the student starts memorizing repeated data, so Q1 and Q2 can be read against fit.

## Analysis

- **Main figure:** cat-rate difference (trait minus control) against optimizer steps, log x axis. One line per U, thin lines for seeds, ticks at epoch boundaries. A second panel shows Δ log P(cat) at full density.
- **Grid figure:** heatmap of the cat-rate difference over (U, epochs).
- Report per-seed points with the mean. Read effect sizes, not p-values.

How to read the outcomes:

| Result | Reading |
|---|---|
| No transfer at 1 epoch over 10k, transfer by 10 epochs | Repetition is needed at this data size. |
| Transfer at 1 epoch over 10k | One epoch is enough. |
| Transfer at 1 epoch, lower by 10 epochs | Rise then decay. Endpoint-only evals at 10 epochs undercount transfer. |
| At the final step, C > B > A | Fresh data beats repeats (Feng's result, seen at high rank). |
| A, B, C curves overlap at matched steps | Total steps matter, uniqueness doesn't. |
| A > C at the final step | Repetition helps. Seeing the same trait-carrying tokens again concentrates the signal. |
| Cat rate stays at 0 while Δ log P rises | Sampling hides early transfer, so endpoint sampled evals undercount it. |

**Scope:** every core claim is specific to LoRA rank 8, the cat trait, and Qwen2.5-7B-Instruct. Feng saw the fresh-data advantage mostly at high rank, so a Q2 null at rank 8 fits that. Extension A tests high rank.

## Phases

**Phase 0: gate.** About 5 GPU-hours, including data generation:

1. Generate the cat and neutral pools. Measure the filter pass rate, the prompt-space size, and tokens per example. Measure the base model's cat rate.
2. **Faithful replication:** cat and neutral with the published recipe exactly (10k, 3 epochs, linear LR), seed 1. 2 runs.
3. **Schedule check and variance read:** cat and neutral with constant LR, 10k, 3 epochs, seeds 1 to 3. 6 runs. This gives the seed SD of the cat-rate difference at 1, 2 and 3 epochs.
4. **Prefix check:** cat with constant LR, 10k, 1 epoch, seed 1. Compare against the seed-1 run from step 3 at step 152: adapter weights should match up to GPU nondeterminism, and both DVs should agree within sampling noise.
5. Measure training throughput, sampled-eval throughput, and log-prob-eval throughput.

Pass criteria (proposed):

- Both the faithful and the constant-LR cat runs beat their control on cat rate by at least 5 percentage points, with a 95% binomial CI excluding 0, and on Δ log P(cat) by at least 0.5 nats.
- If the faithful replication fails, stop and report. That is a replication failure, not an epoch result.
- If only the prefix check fails, the shortcut is unavailable. The core then needs separate runs per training length, about 3x the cost.
- **Seed count for the core:** the smallest n where t(n−1) × SD / √n is under 5 percentage points, using the step-3 seed SD at 3 epochs. Each seed beyond 3 adds 6 runs, about 9 GPU-hours and $31. If n is above 5, check with Thomas.
- If measured throughput puts the core above $150, check with Thomas before launching.

**Phase 1: core.** The 18 runs above (more if the seed count rises).

**Extensions (only after reading the core):**

- **A. Eagle at rank 64.** Eagle's strongest rank per Nief et al. Generate the eagle pool, run a short LR sweep (3 values at gate size), then the A/B/C grid with 3 seeds and rank-64 neutral controls: 18 runs. Tests the trait question and the high-rank question where both effects are expected to be strongest.
- **B. Cat at rank 64.** 9 runs, reusing Extension A's rank-64 controls (neutral data doesn't depend on the trait). Separates rank from trait.
- **C. Extra grid point (U = 50k, 2 epochs, rank 8).** 6 runs.

## Cost

Assumptions, all replaced by Phase 0 measurements:

- About 135 tokens per training example (Qwen default system prompt, chat template, number prompt, about 35-token completion).
- 4,000 training tokens per second for LoRA on one H100 (short, padded sequences).
- One vLLM process per run serves all of that run's adapters through multi-LoRA. The log-prob eval shares each question's prefix across its 44 scored continuations.
- $3.29 per H100-hour: the per-GPU rate of the 4×H100 RunPod pod billed at $13.16/hr on 2026-08-31.

Per core run: 13.5M training tokens is about 56 min of training. Log-prob evals add about 3 min, held-out loss about 5 min, and sampled evals about 15 min (about 210k generations). That's about 1.35 GPU-hours per run.

| Phase | Work | H100-hours | At RunPod rate |
|---|---|---|---|
| Data | 300k prompts × 2 teachers on vLLM | 1.4 | $5 |
| Gate | 9 short runs + 3 throughput probes | 3.5 | $12 |
| Core | 18 runs × 1.35 h | 24 | $79 |
| Overhead | Pod setup, model download, debugging (15%) | 4.3 | $14 |
| **Core total (3 seeds)** | | **33** | **$109** |
| Each extra seed | 6 runs + overhead | 9.3 | $31 |
| Ext. A | Eagle pool, LR sweep, 18 runs at rank 64 | 33 | $109 |
| Ext. B | Cat at rank 64, 9 runs | 15.5 | $51 |
| Ext. C | U = 50k, 6 runs | 9.3 | $31 |

- **Which account pays.** The EPS `auto` router tries GCP first, which bills GCP credits, and falls through to RunPod only on a capacity miss. GCP quota is A100-80GB (16 preemptible, 8 on-demand) and 8 preemptible H100s. On A100s, expect roughly 2x the GPU-hours above. The dollar column is the worst case, where everything lands on RunPod.
- **RunPod balance:** $479.19 at 2026-10-26 20:00Z. The core worst case is about 23% of that.
- **Range for the core:** about $75 if training throughput is 2x faster than assumed, about $180 if 2x slower. If sampled evals run at half the assumed speed, add about $20.
- **Wall clock:** about 33 hours on one H100. About 10 hours on four GPUs at roughly the same total cost (plus about $10 for GPUs idle in the last round).
- **API cost:** none. All scoring is string match or log-probs.
- **Agent cost:** the EPS path for a new experiment is `/clarify`, then `/deep-lit-review`, then `/adversarial-planner` with Codex twins, then implementation review. That pipeline is the main token cost. A leaner path (one implementer session adapting the Cloud repo, driven by one session) needs Thomas's explicit override of the EPS planning rule.

## Implementation notes

- Start from the Cloud et al. repo's `open_model_cfgs.py`, `sl/finetuning/services.py` (training-row format), and its `animal_evaluation` / `animal_evaluation_with_numbers_prefix` question sets. Check `src/explore_persona_space/eval/callbacks.py` (the periodic-eval callback pattern) before writing a new callback.
- Run the log-prob and held-out evals in a `TrainerCallback` in-process, so they need no saved checkpoints.
- Save adapters only at the sampled-eval points (about 20 MB each at rank 8).
- Uploads follow the EPS policy: adapters to the HF model repo, teacher data and sampled completions to the HF data repo under `raw_completions/`, metric JSON to `eval_results/`.
- The session driving the pod terminates it once uploads are verified (standing grant, 2026-10-26).

## Risks

- **SL may not replicate in our pipeline, or the prefix shortcut may not hold.** The gate catches both before the core spends money.
- **Context mismatch kills the effect** (Nief et al.). Training and eval both use the template's default system prompt, and the gate's faithful replication confirms transfer in that context.
- **Cat has a high prior on Qwen.** Yanagisawa, Gho et al. found high-prior traits drift more under neutral fine-tuning. The paired control at every checkpoint handles this, and the margin DV keeps range if the cat rate hits a ceiling.
- **Seed noise may swamp Q2.** Published seed SDs are comparable to the effect (drift paper, Malubay). The gate's variance read sets the seed count before the core runs.
- **Uneven trajectories.** Dense log-prob checkpoints handle this. Also report the area under the Δ curve per run.
- **Memorization at 10 epochs.** The held-out loss tracks it.

## Choices Thomas may want to change

- Model (Qwen2.5-7B-Instruct) and trait (cat).
- Run length (100k example-passes, so the longest condition is 10 epochs over 10k. Feng went to 40 epochs).
- U grid (10k, 25k, 100k).
- LoRA rank 8 in the core, with rank 64 in the extensions.
- Gate thresholds (5 points, 0.5 nats) and the seed-count rule.
- Planning path: the full EPS pipeline, or the lean override.

## Provenance

**Decision record (`/clarify`, 2026-10-26):**

- Goal, hypotheses, model, data, recipe, conditions, controls, DVs, seeds, success and kill criteria: pinned by the design in this body (source: `ask`, the design Thomas reviewed and approved for Phase 0).
- Routing: new task (source: `ask`, "Turn the design into a task").
- Planning path: lean. `/adversarial-planner` skipped by explicit user override of the EPS planning rule (source: `ask`).
- Scope this round: Phase 0 only. Report gate results before any core run (source: `ask`).
- Compute: auto lane (GCP first on credits, RunPod fall-through), 4 GPUs in parallel (source: `user-answer`).
- OUT of scope this round: Phase 1 core, Extensions A to C, review fleets.

**Originating prompts (verbatim):**

1. "Draft a design for the clean subliminal learning epoch test: 1 vs N epochs at fixed unique data, and fixed steps across unique-data sizes, with target log-prob tracked at every checkpoint. Include a cost estimate."
2. "Use the lean path: skip /adversarial-planner for this one (I'm overriding the EPS planning rule). Turn the design into a task, then run Phase 0 only and report the gate results before the core."
3. Compute answer: "Auto lane, 4 GPUs (Recommended)".
