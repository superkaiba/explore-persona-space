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
# Can the context→answer map predict the POST-CoT answer, in CoT-trained models?

## Motivation

The context→answer map `v_A ≈ M v_C` predicts the mean answer activation from the
last-prompt-token activation, in ONE forward pass at ONE token, at held-out R² ≈ 0.8 on
wild chat (#779, #1482). In a CoT-trained (reasoning) model the output splits at a
literal token: `<think> … </think>` then the answer. So the question becomes exact:

**Can we predict the state of the post-`</think>` answer from the pre-`<think>` context
vector, before the model has emitted a single reasoning token?**

A constant-depth transformer's single forward pass sits in TC⁰
([2402.12875](https://arxiv.org/abs/2402.12875),
[2310.07923](https://arxiv.org/abs/2310.07923)); CoT is what buys serial computation
beyond it. So on questions where the reasoning is doing real serial work, the post-CoT
answer should NOT be predictable from the pre-CoT context state. Where it IS predictable,
the CoT was not computing the answer.

This adjudicates a live disagreement in the literature (see § Positioning) and it
directly bounds the map's monitoring application: if the post-CoT answer is predictable
pre-CoT, you can monitor a reasoning model before it reasons.

## Goal

In CoT-trained models, measure how well the pre-CoT context vector `v_C` (last prompt
token, before any `<think>` token) predicts the post-`</think>` FINAL-ANSWER state
`v_A*`, as a graded function of the question's chain-of-thought necessity; and determine
whether any degradation is a genuine loss of answer-content predictability rather than an
artifact of answer length, answer-sampling noise, answer-template stereotypy, or domain
shift.

## Models

**Primary — `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.** Verified config (2026-08-24):
hidden 3584, 28 layers, 28 heads, vocab 152064, `Qwen2ForCausalLM` — **byte-for-byte the
same architecture as `Qwen/Qwen2.5-7B-Instruct`**. Consequences:

- The extraction rig, layer indexing, and layer-19 pin transfer with zero changes.
- `Qwen2.5-7B-Instruct` becomes a **matched non-reasoning twin in the same coordinate
  space**, where the R² ≈ 0.8 map already exists (#779). "Is the CoT-trained model's map
  worse, in the same geometry?" is answerable directly rather than across a model gap.

**Secondary — `Qwen/Qwen3-8B` (the ORIGINAL hybrid release, NOT a `-2507` checkpoint).**
Hidden 4096, 36 layers, so it needs its own refit. Worth it for one reason: the hybrid
release toggles `enable_thinking=True/False` on the SAME WEIGHTS, which gives per-question
CoT necessity measured on the exact model being mapped, with no prompt-hack confound.
The `-2507` checkpoints split into separate Instruct (no `<think>`) and Thinking models
and LOSE this affordance, so the checkpoint pin is load-bearing. Qwen3-8B is also the
model used by the closest prior work (2603.17199), so the arms are comparable.

## The measurement

Per question, capture along the generation:

| position | state | role |
|---|---|---|
| `t = pre` | last prompt token, before any `<think>` token | **the map's input `v_C`** |
| `t ∈ (0,1)` | residual state at normalized positions inside the think block | the trajectory |
| `t = end` | the final `</think>` token, after all serial work | the **ceiling** read |
| target | mean activation over the post-`</think>` answer span | **`v_A*`** |

The post-`</think>` boundary is a literal token, so the trace/answer split is mechanical.
This is the single biggest practical advantage of running in CoT-trained models rather
than prompting a non-reasoning model to think.

**Headline figure: predictability-vs-CoT-position curve, one curve per necessity
stratum.** Prediction: for low-necessity questions the curve is flat and high from
`t = pre` (the answer was already there); for high-necessity questions it starts near
chance at `t = pre` and climbs through the think block. That single plot operationalizes
the whole disagreement below.

## Necessity dials

1. **On-model thinking toggle (PRIMARY, Qwen3-8B arm).** `correct(enable_thinking=True) ∧
   ¬correct(enable_thinking=False)`, same weights, one flag. The cleanest possible
   per-question necessity measure. For the R1-Distill arm the analogue is prefilling an
   empty `<think></think>` to suppress reasoning; this REQUIRES a validation that
   suppression actually took (check no reasoning leaks into the answer span, and that
   accuracy drops on known-hard items) before it can carry any claim.
2. **GSM8K calculator-step count** from the inline `<<48/2=24>>` annotations in the gold
   solutions. Free, per question, WITHIN-corpus so register and format are fixed. Bin
   k=1 / 2–3 / 4–6 / ≥7, matching <https://arxiv.org/html/2608.09942>.
3. **ContextHub deductive + abductive levels 1–4** — a within-corpus logical-depth ladder
   that is NOT math, so it de-confounds "CoT-necessary" from "is a math problem". No
   math-only design can do that. 600 → 2,396 rows per level.

**Pool pre-stratification and cross-model prior (zero GPU):** the TAUR-Lab CoT Analysis
Project collection, released with "To CoT or not to CoT?"
([2409.12183](https://arxiv.org/abs/2409.12183), ICLR 2025).
<https://huggingface.co/collections/TAUR-Lab/cot-analysis-project-66bbb9e5e0156e65059895f5>

16 per-model repos × 31–45 benchmark configs. Verified schema (2026-08-24, config
`gsm8k`): per question, `zero_shot_cot_is_correct`, `zero_shot_direct_is_correct`, the
few-shot twins, parsed answers, rendered prompts, gold answer, and character offsets of
the answer span in `additional_information.*_answer_parser_info.answer_span`. Question
sets align across model repos (verified: gsm8k 1319, mmlu 14042, musique_all 4834, bbh
6258, siqa 3908, contexthub levels identical across four repos probed), so a per-question
cross-model join is well defined. Some configs come DOUBLED (gsm8k_hard 2638 = 2×1319,
arc_challenge 598 = 2×299) and need dedup before joining.

`rescue_rate(q)` = fraction of the 16 models with `cot_correct ∧ ¬direct_correct`. Used
to pre-select a question pool spanning the necessity range cheaply, and as a
model-independent robustness regressor. It does NOT replace dial 1: necessity is
model-specific, and a borrowed label attenuates the correlation on the mapped model.

Near-zero-necessity floor: MMLU, ARC-Challenge (200-question bank already in-repo at
`src/explore_persona_space/artifacts/query_banks/arc_c_v1.json`), CSQA, PIQA, Winogrande.

## Dependent variables

- **DV1 — kNN retrieval on `v_A*` (HEADLINE).** P(true answer state within the k nearest
  neighbors of `M v_C`) among the held-out answer pool; euclidean + cosine, chance =
  k/n_pool stated (`analysis/mapping_baselines.knn_retrieval`). Retrieval is the right
  headline here because post-`</think>` answers are short and template-heavy ("The answer
  is 72."), so R² is inflated by the template while retrieval forces discrimination on
  the answer content. Retrieve within a **same-template pool** to make that binding.
- **DV2 — held-out R² on `v_A*`**, with the mandatory identity+learned-bias baseline
  (`identity_bias_predict`). Apples-to-apples with the 0.8 headline.
- **DV3 — noise-ceiling-normalized DV1/DV2**: divided by per-stratum split-half
  reliability of `v_A*` across rollouts. Required, because reasoning traces branch and
  high-necessity questions have genuinely noisier targets; #1073 measured the single-draw
  penalty at 0.046–0.078 on wild chat and it will be larger here.
- **DV4 — answer-token decode**: unembed / logit-lens `M v_C` and read rank of the gold
  answer token(s); plus a digit-level decode for numeric answers (tokenization of numbers
  is irregular, so a string-level read is not enough).
- **DV5 — ceiling contrast**: the same four reads from the `t = end` (`</think>`) state.
  If `t = end` recovers the answer and `t = pre` does not, the gap IS the serial work.

## Arms and controls

- **Reasoning vs matched non-reasoning, same geometry**: R1-Distill-Qwen-7B vs
  Qwen2.5-7B-Instruct, both 3584/28, both at layer 19.
- **Thinking-on vs thinking-off, same weights**: Qwen3-8B `enable_thinking`.
- **Map-fit arms**: (a) refit in-domain on the reasoning model's own contexts — the
  mechanism read; (b) the frozen #779 1M map applied cross-model as a transfer read only,
  clearly labeled, since it was fit on different weights and a different corpus.
- **Matched controls**: answer token count matched or regressed out; direct-answer
  accuracy matched where possible; per-stratum `v_A*` variance reported so a
  stereotypy-driven metric change is legible; think-block length reported per stratum.

## Positioning — there is a live disagreement and this experiment adjudicates it

**Camp A, the answer is already there pre-CoT:**
- "Catching rationalization in the act" (<https://arxiv.org/html/2603.17199>): pre-CoT
  probes at the first decoding step predict motivated reasoning as well as a GPT-5-nano
  monitor with full trace access; end-of-CoT probes recover the hinted choice at >86%.
  Qwen3-8B thinking mode, MMLU / AQuA / ARC-C / CSQA. **Binary and small-multiclass
  classification only — explicitly does NOT regress onto answer activations.**
- "Therefore I am. I Think." (<https://arxiv.org/html/2604.01202>): tool-call decisions
  >90–95% predictable from activations before any reasoning token.
- "Reasoning Models Know When They're Right" (NYU): correctness probe reads 0.79 AUROC
  from the very first reasoning step, 0.95 by completion. DeepSeek-R1-Distill series,
  QwQ-32B.

**Camp B, the answer is computed during CoT:**
- "LLMs Faithfully and Iteratively Compute Answers During CoT"
  (<https://arxiv.org/abs/2412.01113>): on multi-step arithmetic, linear probes for the
  final answer improve MONOTONICALLY along the reasoning chain — genuine iterative
  computation, not retrieval.

**Our contribution:** every Camp-A result probes a LOW-BANDWIDTH target (a binary label, a
4–5 way choice, a correctness flag). We regress the FULL answer state and read it by
retrieval among a candidate pool, which is a far higher-bandwidth claim, and we cross it
with a graded necessity dial in CoT-trained models. The reconciliation we expect: Camp A
holds on low-necessity questions, Camp B on high-necessity ones, and the crossover is
measurable. Camp B's monotone-improvement result is a sharp prior prediction that our
`t = pre` read should fail precisely on high-necessity items.

## Cost drivers to size at plan time

- Reasoning traces are LONG. `max_new_tokens` must be far above the 2048 default
  (CLAUDE.md: reasoning/CoT models need far more); budget 8k–16k and report the realized
  cap-hit fraction with a pre-registered re-gen trigger.
- The map needs a refit per model, so a generation corpus is required per arm.
- Activation capture over long sequences: decide capture-during-generation vs a
  teacher-forced re-pass, and note the re-pass cost scales with trace length.

## Open decisions for the planner / user

1. Both model arms, or R1-Distill only for round 1? R1-Distill alone gives the
   matched-geometry control; Qwen3-8B alone gives the clean thinking toggle. They answer
   different halves.
2. Is the predictability-vs-CoT-position trajectory in scope for round 1? Intermediate
   positions come from the same forward pass, so it is cheap once traces exist, and it is
   the strongest single figure.
3. Free directional read first: the frozen #779 map applied cross-model to R1-Distill
   contexts on the 1,319 GSM8K test questions, binned by calculator-step count. Weak
   (cross-model transfer, so a drop is partly confounded) but near-zero GPU.

## Provenance

Originating chat, 2026-08-24: "I want to design an experiment to check if our mapping
does a lot worse on questions where CoT is in some sense NECESSARY. find a
dataset/model/framework for this"; then the scoping call that necessity be operationalized
as measured CoT lift and that the lift "doesn't have to be our model"; then the redirect
that fixed the frame: "wait but we want to predict directly from context to the post CoT
answer, and do this in CoT trained models".
