---
title: Does the context→answer map degrade on questions where chain-of-thought is
  necessary?
kind: experiment
tags: []
created_at: '2026-08-24T17:29:34Z'
has_clean_result: false
origin_prompt: I want to design an experiment to check if our mapping does a lot worse
  on questions where CoT is in some sense NECESSARY. find a dataset/model/framework
  for this
workflow: v1
goal: Measure whether held-out predictive accuracy of the context→answer map (v_C
  → v_A, Qwen-2.5-7B-Instruct, layer 19) falls as a function of a question's chain-of-thought
  necessity, defined as the per-question cross-model rate at which CoT prompting flips
  an answer from wrong to right; and determine whether any such fall is a genuine
  loss of answer-content predictability versus an artifact of answer length, answer-sampling
  noise, or domain shift.
---
# Does the context→answer map degrade on questions where chain-of-thought is necessary?

## Motivation

The context→answer map `v_A ≈ M v_C` predicts the mean answer activation from the
last-prompt-token activation, in ONE forward pass at ONE token, at held-out R² ≈ 0.8
on wild chat (#779, #1482). A constant-depth transformer's single forward pass sits in
TC⁰ ([2402.12875](https://arxiv.org/abs/2402.12875),
[2310.07923](https://arxiv.org/abs/2310.07923)); chain-of-thought is what buys serial
computation beyond it. So on a question whose answer requires serial computation the
model cannot perform in one pass, the answer's CONTENT cannot be a function of `v_C` at
all, and the map should collapse.

#1482 found the map is close to its information ceiling on wild chat and that it
predicts tonic / context-extrinsic properties (register, language, discourse position)
well while missing token-intrinsic content. Chain-of-thought necessity is a principled
axis for that distinction with a theory behind it rather than a post-hoc label.

## Goal

Measure whether held-out predictive accuracy of the context→answer map (v_C → v_A, Qwen-2.5-7B-Instruct, layer 19) falls as a function of a question's chain-of-thought necessity, defined as the per-question cross-model rate at which CoT prompting flips an answer from wrong to right; and determine whether any such fall is a genuine loss of answer-content predictability versus an artifact of answer length, answer-sampling noise, or domain shift.

## Necessity labels (zero GPU)

Source: the TAUR-Lab CoT Analysis Project HF collection, released with
"To CoT or not to CoT?" ([2409.12183](https://arxiv.org/abs/2409.12183), ICLR 2025).
<https://huggingface.co/collections/TAUR-Lab/cot-analysis-project-66bbb9e5e0156e65059895f5>

16 per-model dataset repos, 31–45 benchmark configs each. Verified schema (2026-08-24,
`TAUR-Lab/Taur_CoT_Analysis_Project___Qwen__Qwen2-7B-Instruct`, config `gsm8k`) carries,
per question: `zero_shot_cot_is_correct`, `zero_shot_direct_is_correct`, the few-shot
twins of both, the parsed answers, the rendered prompts, and the gold answer.

Question sets align across model repos (verified: gsm8k 1319, mmlu 14042, musique_all
4834, bbh 6258, siqa 3908, contexthub levels identical across the four repos probed), so
a per-question join across models is well defined. Some configs come doubled
(gsm8k_hard 2638 = 2×1319, arc_challenge 598 = 2×299) and need dedup before joining.

Per-question necessity score, over the M models sharing that question:

- `rescue_rate(q)` = fraction of models with `cot_correct ∧ ¬direct_correct` (PRIMARY —
  the graded, continuous regressor)
- `signed_lift(q)` = mean over models of `cot_correct − direct_correct` (companion; can
  go negative where CoT hurts, which is itself a useful cell)

Cross-model consensus is deliberate: a single model's binary lift is noisy and
model-idiosyncratic, and the user's scoping call was that necessity need not be measured
on our own model. Validation arm: measure the same lift on Qwen-2.5-7B-Instruct
ourselves over a subset and report rank agreement with the consensus.

## Three independent depth dials

The headline must not rest on one corpus's quirks, and a bare "math vs MMLU" contrast
confounds necessity with domain, length, and difficulty simultaneously.

1. **Cross-model consensus lift** — continuous, spans all shared configs. Primary
   regressor.
2. **GSM8K calculator-step count** — GSM8K solutions carry inline `<<48/2=24>>`
   annotations, so serial-step count `k` is free per question. Bin k=1 / 2–3 / 4–6 / ≥7,
   matching the stratification in <https://arxiv.org/html/2608.09942> (same model family;
   per-item depth labels also released at DOI 10.5281/zenodo.20294033). WITHIN-corpus, so
   register / format / topic are held fixed.
3. **ContextHub deductive + abductive levels 1–4** — a within-corpus logical-depth
   ladder that is NOT math (600 → 2,396 rows per level, identical across model repos).
   De-confounds "CoT-necessary" from "is a math problem", which no math-only design can.

Near-zero-lift floor: MMLU, ARC-Challenge (200-question bank already in-repo at
`src/explore_persona_space/artifacts/query_banks/arc_c_v1.json`), CSQA, PIQA, Winogrande.
Reference point: the LMSYS/WildChat corpus where the 0.8 was measured.

Published anchor for the coarse contrast, on Qwen-2.5-7B-Instruct among others: GSM8K
+53.9 to +68.0 pp CoT gain, MATH +55.4 to +67.5 pp, MMLU +2.4 to +4.6 pp, ARC-C +0.0 to
+3.3 pp (2608.09942).

## Model

Qwen-2.5-7B-Instruct, layer 19. Keep it: the 963k-context 1M map, the layer pin, the
extraction rig, and the #1482 comparison baselines all exist for it, and changing the
mapped model costs a full refit plus comparability with the 0.8 headline. Necessity
labels come from the other 16 models, which is the decoupling the scoping call endorsed.
`Qwen2-7B-Instruct` is in the label collection — same family one generation back, so it
is the closest single-model sanity anchor.

Optional robustness arm (NOT primary): a reasoning model (Qwen3-8B thinking mode or
DeepSeek-R1-Distill-Qwen-7B) where the `<think>` delimiters make the trace/answer split
explicit and necessity is at its extreme. Costs a map refit.

## What will confound this if the design ignores it

1. **Style swamps content.** `v_A` is the token-mean over the whole answer. A CoT answer
   is a long trace whose token-mean is dominated by generic reasoning register. #1482
   showed the map predicts exactly that class well. R² on `v_A` could therefore RISE on
   CoT-necessary questions while the map knows nothing about the answer.
2. **Target noise.** CoT answers branch more across rollouts, so `v_A` itself is noisier
   and R² falls for a reason unrelated to serial depth. #1073 measured the single-draw
   penalty at 0.046–0.078 on wild chat; it will be larger here.
3. **Domain shift.** The 1M map was fit on LMSYS/WildChat. Math and formal logic are out
   of distribution for it, so a drop may be pure distribution shift.

## Dependent variables

- **DV1 — `v_A` held-out R²** (apples-to-apples with the 0.8 headline), with the standing
  mandatory companions: the identity+learned-bias baseline
  (`analysis/mapping_baselines.identity_bias_predict`) and kNN retrieval
  (`knn_retrieval`, euclidean + cosine, chance = k/n_pool stated).
- **DV2 — noise-ceiling-normalized R²**: DV1 divided by per-stratum split-half
  reliability of `v_A` across rollouts. HEADLINE, because DV1 alone is not comparable
  across strata (confound 2).
- **DV3 — final-answer-span R²**: `v_A` recomputed over the final-answer span only (after
  `####` / "the answer is" / the `\boxed{}` span, whose character offsets the TAUR-Lab
  rows already carry in `additional_information.*_answer_parser_info.answer_span`).
  Strips the reasoning register (confound 1).
- **DV4 — answer-content recovery**: decode the gold answer from the map's PREDICTION
  `M v_C` versus from the true `v_A`. If the true `v_A` carries the answer and `M v_C`
  does not, that is the clean serial-computation signature.

## Arms

- **Map arms**: (a) frozen #779 1M map applied to these corpora — the transfer read;
  (b) in-domain refit on math/logic contexts — the mechanism read. Without (b) any drop
  is uninterpretable (confound 3).
- **Answer-regime arms, on the SAME questions**: CoT-allowed vs CoT-suppressed
  ("answer with just the number"). Under suppression the answer must be produced in one
  forward pass, so `v_C` COULD carry it. This within-question manipulation isolates
  serial depth rather than a dataset property, and the TAUR-Lab rows supply both prompt
  renderings verbatim.

## Pre-registered prediction

DV3 and DV4 fall monotonically with all three depth dials under CoT-allowed, and
partially recover under CoT-suppression on the same questions. DV1 may be flat or rise;
if it does, that IS the headline (the map tracks answer style, not answer content), and
the design still resolves it.

Additional matched controls: answer token count matched or regressed out across strata;
direct-answer accuracy matched where possible; per-stratum `v_A` variance reported so a
stereotypy-driven R² change is legible.

## Closest prior work to position against

"Knowing Before Saying" (<https://arxiv.org/html/2505.24362v1>) probes the prompt-final
hidden state — the same position as `v_C` — and predicts whether a CoT will land correct
at 60–76.4% accuracy before any token is generated, and finds later reasoning steps do
not consistently improve the prediction. So the prompt-final state is NOT blank about
downstream reasoning. Our claim has to be about the answer's CONTENT, not about
predictability of reasoning success, and the design must keep those separate.

## Open decisions for the planner / user

1. Whether to add the reasoning-model robustness arm (costs a map refit).
2. Whether the CoT-suppressed control arm is in scope for round 1 or deferred (it is the
   strongest element but roughly doubles the generation pass).
3. Whether to run the free directional read first: the frozen 1M map on the 1,319 GSM8K
   test questions, binned by calculator-step count, before committing to the full design.

## Provenance

Originating chat, 2026-08-24: "I want to design an experiment to check if our mapping
does a lot worse on questions where CoT is in some sense NECESSARY. find a
dataset/model/framework for this", followed by the scoping call that necessity be
operationalized as measured CoT lift, and that the lift "doesn't have to be our model".
