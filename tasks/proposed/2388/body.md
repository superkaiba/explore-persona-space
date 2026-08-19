---
title: Is answer correctness predictable from the context vector, and does the context-to-answer
  map help?
kind: experiment
tags: []
created_at: '2026-08-19T16:45:16Z'
has_clean_result: false
parent_id: 1739
origin_prompt: 'I want to run an experiment to test if answer correctness is predictable
  from the context vector, and then whether our mapping can help to predict it. Help
  me to plan this expeirment. [Scope settled in the same chat: all four correctness
  surfaces (banked QA + math + MCQ + code); headline framing = BOTH the knowledge-vs-persona
  map test and the label-efficiency crossover; slim arm ladder plus mandated baselines;
  routed as a new child task of #1739.]'
workflow: v1
goal: On Qwen2.5-7B-Instruct, determine whether on-policy answer correctness (gold-
  or execution-verified, K=5 rollouts) is predictable from the frozen context vector
  before generation, and whether routing the context vector through the learned context-to-answer
  map improves that prediction at matched data budgets, across four correctness surfaces
  (short-answer QA, math, multiple-choice, code) and a distribution-shift ladder.
---
# Is answer correctness predictable from the context vector, and does the context→answer map help?

## Goal

On Qwen2.5-7B-Instruct, determine whether on-policy answer correctness (gold- or execution-verified, K=5 rollouts) is predictable from the frozen context vector before generation, and whether routing the context vector through the learned context-to-answer map improves that prediction at matched data budgets, across four correctness surfaces (short-answer QA, math, multiple-choice, code) and a distribution-shift ladder.

## Motivation

The context→answer map `M` (`v_A ≈ M v_C`) has so far been tested as a predictor of **persona /
disposition** properties — evil, trait sycophancy, hallucination-fabrication (#1739) — and the
patch-only sweep (#2094 / #2162) found the context vector's causal content to be largely
persona-shaped. **Answer correctness is a knowledge property, not a disposition**: whether the
model gets a question right depends on what it knows and on the difficulty of the item, not on
how it is asked to behave. It is therefore the cleanest available test of whether the map carries
*non-persona* information, and the answer is not obvious in either direction.

The practical framing is the same as #1739's: judged context→answer pairs are expensive, unjudged
ones are free. If the map — fit only on unjudged generic chat — is a useful prior, a correctness
predictor built on the mapped answer vector should reach a given accuracy with far fewer
correctness labels than a direct probe on the context vector, with the advantage growing under
distribution shift.

Prior work establishes the *direct* side of this (linear probes on hidden states predict
correctness / truthfulness / P(IK) reasonably well). It does not establish whether an
answer-space map adds anything, which is what this task tests. A grounding literature pass is a
prerequisite (see § Open items).

## Formalization

Model: `Qwen/Qwen2.5-7B-Instruct`, frozen. No fine-tuning; every arm is a read-out over frozen
activations. Vocabulary per `docs/glossary_context_answer_map.md`.

**Objects** (per layer ℓ, 28 layers × 3,584 dims):

- `v_C(x)` — **context vector**, pooled at the **last prompt token** (the newline before the
  assistant answer). Pooling declared per vector (planner §6 convention row, #1974).
- `v_A` — **answer vector**, token-mean over one sampled answer's tokens. For long-CoT surfaces
  (math, code) a second pooling (last answer token) is captured alongside, since a token-mean
  over a long chain is dominated by reasoning text rather than by the answer.
- `M_ℓ` — the ridge map `v_A ≈ M_ℓ v_C`, **fit only on unjudged WildChat/LMSYS context→answer
  pairs**. The banked maps from #1739 are reused (`issue1739_ctxmap/analysis_tensors/maps/`,
  linear/MLP/kernel at U ∈ {250, 5,000, 18,793 = full store}).

**Dependent variable.** Per context `x`, the **correctness rate**

    y(x) = (1/K) · Σ_k 1[answer_k is correct],  K = 5 on-policy rollouts, temperature 1.0

verified **programmatically on every surface** — gold-alias exact match (short-answer QA), final
answer match after normalization (math), option match (multiple choice), unit-test execution
(code). **No LLM judge in the primary DV.** This sidesteps the judge-saturation and rubric-drift
failure modes that dominate the behavior DVs, and is a genuine methodological upgrade over
#1739's judged rates.

**Secondary continuous companion** (non-saturating, per the dual-DV rule): teacher-forced mean
token log-probability of the gold answer given the context — for MCQ the log-prob **margin**
between the correct option and `logsumexp` over the distractors. Code has no unique gold program;
it reports pass-rate only, with log P(reference solution) as a rough companion, flagged as such.

**Pre-registered hypotheses.**

- **H1 (direct).** `y` is predictable from `v_C` above the constant and surface-feature baselines
  on all four surfaces. Expected true for recall QA; genuinely open for math and code, where
  correctness depends on a computation the model has not yet performed at the last prompt token.
- **H2 (map, label-efficiency).** The mapped-answer probe matches or beats the direct context
  probe at small `L`, with the gap closing (and possibly reversing) as `L` grows. Report the
  **crossover** `L` per surface and the **degradation slope** across the shift ladder.
- **H3 (knowledge vs persona — the sharp read).** If the map is predominantly persona-carrying,
  the mapped-answer arm should lose more ground to the direct context arm on correctness than it
  did on #1739's three dispositional behaviors, at matched arms and matched budgets. This
  comparison costs nothing extra — #1739's numbers are banked.

## Surfaces

Four correctness surfaces, ordered by how far correctness is from surface recall:

| Surface | Sources | Verification | Status |
|---|---|---|---|
| Short-answer QA | TriviaQA `rc.nocontext` (16,000 train ctx), NQ-Open (3,167), SimpleQA (4,021) | gold alias match, already three-way labeled | **fully banked** (#1739) |
| Math / reasoning | GSM8K, MATH | normalized final-answer match | new rollouts + capture |
| Multiple choice | MMLU, ARC-Challenge | option match; clean log-prob companion | new rollouts + capture |
| Code | MBPP, HumanEval, and a third pool to reach adequate n (BigCodeBench / LiveCodeBench) | unit-test execution in a sandbox | new rollouts + capture |

**Sizing constraint — this is load-bearing.** `d = 3,584`. Every ridge fit at `n_train < d` is
estimator-degenerate (#1701, #1887), so each new surface needs **≥ ~8,000 contexts** to be fit in
the ambient basis. HumanEval (164) + MBPP (974) cannot reach that alone: code either pools a
third source or is fit in a reduced basis (PCA-k from the unlabeled pool) with the
under-determined regime declared explicitly. The planner sizes this; it is not optional.

## Arms (slim ladder + mandated baselines)

Predictors, all ridge with per-layer λ selected under a dof cap (never pure GCV at `n < d`):

1. **Direct** — probe on `v_C` (whitened with unlabeled-pool statistics; isotropic ridge on raw
   activations is a strawman).
2. **Mapped** — probe on `M v_C`, `M` frozen and fit only on unjudged pairs.
3. **Oracle ceiling** — probe on the true `v_A` (privileged; dashed reference line).

Mandated baselines and controls:

4. **Identity + learned bias** — `v_C + b` (`analysis/mapping_baselines.identity_bias_predict`);
   required for any fitted map where input and output share dimension.
5. **Constant / train-mean** floor.
6. **kNN-retrieval read** on the map — acc@k, euclidean and cosine, chance = k/n stated
   (required alongside held-out R² for every fitted map).
7. **Shuffled-map control** — destroys context-specific structure at matched spectral scale.
8. **Surface features** — question length, gold-alias count, an item-frequency proxy. Cheap, and
   it is the obvious reviewer objection: a context probe could be reading item difficulty rather
   than model-specific knowledge.

Nonlinear (MLP/kernel) map variants are **not** in the roster by default — linear-by-default
standing rule; they can be added as an explicit user-approved extension.

## Evaluation ladder and metrics

Group-level folds throughout (question entity / MMLU subject / problem id) — never pointwise.
Rungs, in increasing shift:

- **rung 0** — held-out 20% within the training surface
- **rung 1** — cross-dataset within family (TriviaQA+NQ-Open → SimpleQA; GSM8K → MATH; MMLU → ARC)
- **rung 2** — cross-family (recall QA → math / MCQ / code) — the interesting rung for "does the
  map transfer at all"

Metrics: Spearman ρ (matches #1739 so the H3 comparison is commensurable), held-out R², and AUROC
on the binarized DV for legibility. Paired bootstrap intervals over identical realized folds per
`(L, seed)` so every arm comparison is paired. Permutation null over the max across arms/layers.

**Result 0 (gates, before any headline).** (a) DV spread per surface and rung — SD floor and
bottom-bin check, as in #1739 gate 1; a rung that fails is dropped, never drawn as a zero bar.
(b) **Item-matched split-half ceiling** on the K=5 DV per surface — never computed for
hallucination in #1739, and with K=5 a modest ρ may already be near ceiling. (c) **ρ(correctness
rate, #1739 fabrication rate)** on the banked QA rungs, so the novelty over #1739's hallucination
arm is explicit rather than assumed. (d) **Map reconstruction quality** (held-out R² + kNN acc@k)
on each new surface *before* the readout, so a null readout is attributable to map degradation
rather than to absent signal — the maps were fit on generic chat and math/code contexts are far
off that distribution.

## Phasing and compute (planner to size properly)

- **Phase 0 — banked QA, 0 GPU-h.** The correctness DV already exists:
  `eval_results/issue_1739/dv_dataset/hallucination/labeling.json` carries `fractions.correct`
  for all 23,188 contexts (5 rollouts each). `scripts/issue1739_fits.py` already accepts
  `--dv-json`, so this is a DV swap over the existing arm ladder. **CPU pod required, not the
  VM** — the activation store is a single 70 GB tar
  (`issue1739_ctxmap/capture_store/hallucination_labeling`), over both the ~10 GB download rule
  and the 50 GB VM-footprint gate. `cpu-bigmem` with container disk sized for the tar plus
  extraction.
- **Phase 1 — new surfaces (GPU).** ~8k math + ~8k MCQ + ~3k code contexts × 5 rollouts ≈ 95k
  generations under vLLM, plus a teacher-forced capture pass for `v_C` and `v_A`. Rough order
  25–40 GPU-h on 1× H100; **pilot-gated** with a measured 1-cell wall before the production
  dispatch. New activation store ≈ 25 GB fp16.
- **Phase 2 — fits (CPU pod).** 4 surfaces × 8 arms × 28 layers × `L`-sweep × seeds × folds,
  through the shared vectorized fit cores. Sharded across cells; never a serial per-cell loop.
- **Judge spend ≈ $0** for the primary DV. Math answer-equivalence uses a verifier library, not a
  judge, wherever it can.

## Reuse (what is already banked)

- Correctness labels + rollouts for the whole QA surface (#1739).
- Fitted maps, linear and nonlinear, at three unlabeled budgets (`analysis_tensors/maps/`).
- Activation capture store for the QA contexts (`capture_store/hallucination_labeling`).
- The entire fit/arm/fold/bootstrap pipeline (`scripts/issue1739_fits.py`,
  `issue1739_final_fold.py`, `experiments/issue_1739/arms.py`) and the #825 vectorized fit cores.
- #1739's behavior numbers, for the H3 persona-vs-knowledge comparison.

The new code is generation + capture for three surfaces, four programmatic verifiers, and the
correctness DV builder.

## Risks and inherited caveats

1. **Novelty vs #1739's hallucination arm.** correct = 1 − abstained − fabricated, and fabrication
   rate was already predicted. Result 0(c) quantifies the overlap; math / MCQ / code are where
   the novelty is structurally protected.
2. **`n_train < d`** on the new surfaces — see the sizing constraint above.
3. **Map domain shift** — the maps were fit on generic chat; Result 0(d) makes a null attributable.
4. **Difficulty confound** — arm 8 is the control; a probe reading item difficulty is a real
   mechanism but a different claim from "the model knows what it knows".
5. **Contamination** — TriviaQA (2017) and NQ-Open (2019) are plausibly in Qwen's pretraining;
   inherited caveat from #1739, and a reason the math/code surfaces matter.
6. **Language intrusion** — 6.2% of the banked hallucination rollouts carry CJK intrusion
   (#1739 intrusion audit); carry the same scan and recount.
7. **Answer-vector pooling on long CoT** — capture both token-mean and last-answer-token; declare
   which is primary per surface.

## Open items for the planner

- Literature grounding pass (`/deep-lit-review`) on hidden-state correctness / truthfulness /
  P(IK) probing, to name the closest prior formalizations and to set the expected effect sizes
  for H1 before any fits are run.
- Exact context counts per new surface, pinned against the `n ≥ d` constraint.
- Third code pool selection, and whether code stays in the headline or is reported as exploratory.
- Whether the `L`-sweep runs on all four surfaces or on QA + one new surface, with the others at
  full-label only.
