---
title: Can the context→answer map predict the POST-CoT answer, and how does CoT training
  change the map?
kind: experiment
tags: []
created_at: '2026-08-24T17:29:34Z'
has_clean_result: false
origin_prompt: I want to design an experiment to check if our mapping does a lot worse
  on questions where CoT is in some sense NECESSARY. find a dataset/model/framework
  for this
workflow: v1
goal: In CoT-trained models (DeepSeek-R1-Distill-Qwen-7B primary, Qwen3-8B hybrid
  secondary), measure how well the pre-CoT context vector v_C (last prompt token,
  before any <think> token) predicts the post-</think> final-answer state v_A*, as
  a graded function of the question's chain-of-thought necessity; and determine whether
  any degradation is a genuine loss of answer-content predictability rather than an
  artifact of answer length, answer-sampling noise, answer-template stereotypy, or
  domain shift.
---
# Can the context→answer map predict the POST-CoT answer, and how does CoT training change the map?

## Motivation

The context→answer map `v_A ≈ M v_C` predicts the mean answer activation from the
last-prompt-token activation, in ONE forward pass at ONE token, at held-out R² ≈ 0.8 on
wild chat (#779, #1482). In a CoT-trained model the output splits at a literal token:
`<think> … </think>` then the answer. That makes the question exact, and it splits into
two plots.

A constant-depth transformer's single forward pass sits in TC⁰
([2402.12875](https://arxiv.org/abs/2402.12875),
[2310.07923](https://arxiv.org/abs/2310.07923)); CoT is what buys serial computation
beyond it. Where the reasoning does real serial work, the post-CoT answer should not be
predictable from the pre-CoT context state. Where it IS predictable, the CoT was not
computing the answer.

## Goal

(1) In CoT-trained models, measure R² and acc@1 for four maps — post-context→answer,
post-context→CoT, post-context→CoT+answer, post-CoT→answer — on a corpus that DOES versus
DOESN'T require chain-of-thought; and (2) measure whether a PRE-CoT-trained model's
context vector can predict the POST-CoT-trained model's post-`</think>` answer state, and
how the fitted map changes across the CoT-training step, using a matched same-geometry
pre/post pair.

---

# Plot 7 — Relationship to CoT

**Metrics (both, every cell):** held-out R², and **acc@1** = kNN retrieval at k=1
(`analysis/mapping_baselines.knn_retrieval`, euclidean + cosine, chance = 1/n_pool
stated). Plus the mandatory identity+learned-bias baseline
(`identity_bias_predict`) — all four maps are same-dimension, so it always applies.

**Four maps, all within ONE CoT-trained model:**

| cell | input | target | what it tests |
|---|---|---|---|
| **A** | post-context `v_C` (last prompt token, before any think token) | answer only (post-`</think>` span) | **the hypothesis** — is the answer there before reasoning? |
| **B** | post-context `v_C` | CoT only (inside the think block) | is the *reasoning trace* predictable even when the answer is not? |
| **C** | post-context `v_C` | CoT + answer (whole output) | the closest analogue of the existing R² ≈ 0.8 map |
| **D** | post-CoT vector (state at `</think>`) | answer only | **the ceiling** — after the serial work, is the answer there? |

**Crossed with corpus: requires-CoT vs doesn't-require-CoT.**

**Predicted pattern and what each contrast buys:**

- *Doesn't require CoT*: A, B, C, D all high. The answer was present from the start.
- *Requires CoT*: **A collapses**; **D stays high** — the A→D gap IS the serial work, read
  in the same units; **B stays moderate**, because the trace's register, length, and shape
  are predictable even when its content is not; **C sits between and is inflated by the
  long trace**.
- **The A-vs-C contrast is the star of the plot.** If C ≫ A on the requires-CoT corpus,
  then the whole-output map's apparent competence is carried by the reasoning trace, not
  by the answer. That is the single most important thing to establish, because it says the
  existing 0.8-style whole-output read cannot be cited as evidence the map predicts
  answers.

**Backing panel (continuous version):** the DOES/DOESN'T bar plot is a two-bin summary of
a continuous relation. Also plot R² and acc@1 versus the graded necessity score, one line
per map cell, so the bar plot is not the only evidence.

**Measurement discipline this plot needs:**
- acc@1 retrieves within a **per-target pool** (answer pool for A/D, CoT pool for B,
  whole-output pool for C). Cross-target retrieval is meaningless.
- Post-`</think>` answers are short and template-heavy ("The answer is 72."), so R² is
  inflated by the template. Retrieve within a **same-template pool** so acc@1 forces
  discrimination on answer content. acc@1 is the headline metric for A and D for this
  reason; R² is the companion.
- Per-target **noise ceilings**: split-half reliability of each target across rollouts,
  per stratum. Reasoning traces branch, so requires-CoT targets are genuinely noisier;
  #1073 measured the single-draw penalty at 0.046–0.078 on wild chat and it will be
  larger here. Report ceiling-normalized values alongside raw.
- Report per-stratum target token count and target variance, so a length- or
  stereotypy-driven metric change is legible rather than silent.

---

# Plot 8 — Relationship to CoT training

**Question:** can a PRE-CoT-trained context vector predict the POST-CoT answer, and how
does the mapping change across the CoT-training step?

**The pair (primary): `Qwen/Qwen2.5-7B-Instruct` → `open-thoughts/OpenThinker3-7B`.**
Verified 2026-08-24:

- OpenThinker3-7B is fine-tuned **from Qwen2.5-7B-Instruct**, on OpenThoughts3-1.2M,
  **pure SFT on reasoning traces, no RL**. A single-variable CoT-training intervention.
- Both configs: hidden 3584, 28 layers, vocab 152064, `Qwen2ForCausalLM`. **Identical
  geometry**, so a cross-model map is well posed, the identity baseline is meaningful, the
  layer-19 pin transfers, and the extraction rig needs no changes.
- The *pre* model is the exact model the whole mapping line is built on, so the R² ≈ 0.8
  map and the #1482 baselines are the pre-side reference for free.

| cell | input model | input | target model | target | tests |
|---|---|---|---|---|---|
| **E** | pre | `v_C` | post | post-`</think>` answer | **the headline** — does CoT training put something there that was not? |
| **F** | post | `v_C` | post | post-`</think>` answer | within-model reference (= Plot 7 cell A) |
| **G** | pre | `v_C` | pre | pre's own answer | the existing baseline map (the 0.8) |
| **H** | pre | `v_C` | post | post's CoT | does the pre-model's context state predict the reasoning the post-model will do? |

**"How does the mapping change" — use the built battery, do not re-derive it:**
within-stage held-out R², the **#825 reparameterization gap** (within-stage R² minus the
reparameterized-base-map R²), and operator comparison following the
`scripts/issue1345_operator_comparison.py` conventions. Per CLAUDE.md's
similarity-statistic rule, state whether each similarity is direction-aware
(raw / Procrustes-aligned operator cosine) or spectrum/rotation-invariant-only — the
latter can never support a "same operator up to rotation" claim.

**Major reuse — #1336 is this battery one level up.** Status `awaiting_promotion`, goal:
"Determine whether RLVR-style RL post-training changes the linear context→answer-profile
map more than SFT/DPO post-training, using a released separated-stage ladder … per stage,
measure (a) within-stage held-out R² of the per-example ridge map c_x → v(x) and (b) the
#825 Result-2 reparameterization gap." It carries 55 banked turnstores across
`gsm8k_train_full` / `gsm8k_test1319` / `math7500` / `lmsys5k` / `lmsys23k` / `if11k` /
`sft11k` / `uf11k`, and its `common.py` already handles a Qwen2.5-7B base/instruct pair.
**Plot 8 is the CoT-training instance of #1336's battery: reuse the battery, swap the
ladder.** The planner must check the banked turnstores for fitness before regenerating
anything (`.claude/rules/artifact-reuse.md`).

**Second pair (robustness, a different CoT-training recipe):** `Qwen/Qwen2.5-Math-7B` →
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (SFT on 800k R1-generated traces; verified same
3584/28 geometry). Weaker as a pre/post pair because the base is a non-instruct
math-specialized model, but it is a genuinely different CoT-training recipe, and
R1-Distill's chat template **prefills `<think>\n`**, so its trace/answer boundary is
guaranteed rather than emergent.

---

## Necessity dials (defining the DOES vs DOESN'T corpora)

1. **On-model thinking toggle.** `Qwen/Qwen3-8B` — the ORIGINAL hybrid release, NOT a
   `-2507` checkpoint — toggles `enable_thinking=True/False` on the SAME WEIGHTS, giving
   per-question necessity `correct(think) ∧ ¬correct(no-think)` on the exact model being
   mapped, with no prompt-hack confound. The `-2507` checkpoints split into separate
   Instruct and Thinking models and LOSE this affordance, so the checkpoint pin is
   load-bearing. Geometry is 4096/36, so this arm needs its own refit — it is a third arm,
   not a substitute for the same-geometry pair. Qwen3-8B is also the model in the closest
   prior work (2603.17199), so the arms stay comparable.
2. **GSM8K calculator-step count** from the inline `<<48/2=24>>` annotations in the gold
   solutions. Free, per question, WITHIN-corpus so register and format are fixed. Bin
   k=1 / 2–3 / 4–6 / ≥7, matching <https://arxiv.org/html/2608.09942>.
3. **ContextHub deductive + abductive levels 1–4** — a within-corpus logical-depth ladder
   that is NOT math, so it de-confounds "requires CoT" from "is a math problem". No
   math-only design can do that. 600 → 2,396 rows per level.

**Pool pre-stratification and cross-model prior (zero GPU):** the TAUR-Lab CoT Analysis
Project collection, released with "To CoT or not to CoT?"
([2409.12183](https://arxiv.org/abs/2409.12183), ICLR 2025).
<https://huggingface.co/collections/TAUR-Lab/cot-analysis-project-66bbb9e5e0156e65059895f5>

16 per-model repos × 31–45 benchmark configs. Verified schema (2026-08-24, config
`gsm8k`): per question, `zero_shot_cot_is_correct`, `zero_shot_direct_is_correct`, the
few-shot twins, parsed answers, rendered prompts, gold answer, and character offsets of
the answer span in `additional_information.*_answer_parser_info.answer_span`. Question
sets align across model repos (verified: gsm8k 1319, mmlu 14042, musique_all 4834,
bbh 6258, siqa 3908, contexthub levels identical across four repos probed), so a
per-question cross-model join is well defined. Some configs come DOUBLED (gsm8k_hard
2638 = 2×1319, arc_challenge 598 = 2×299) and need dedup before joining.

`rescue_rate(q)` = fraction of the 16 models with `cot_correct ∧ ¬direct_correct`. Used to
pre-select a question pool spanning the necessity range cheaply, and as a
model-independent robustness regressor. It does NOT replace dial 1 — necessity is
model-specific, and a borrowed label attenuates the correlation on the mapped model.

DOESN'T-require-CoT corpus: MMLU, ARC-Challenge (200-question bank already in-repo at
`src/explore_persona_space/artifacts/query_banks/arc_c_v1.json`), CSQA, PIQA, Winogrande,
ContextHub level 1, GSM8K k=1. DOES: GSM8K k≥4, MATH, ContextHub levels 3–4.

Published anchor on Qwen-2.5-7B-Instruct among others: GSM8K +53.9 to +68.0 pp CoT gain,
MATH +55.4 to +67.5 pp, MMLU +2.4 to +4.6 pp, ARC-C +0.0 to +3.3 pp
(<https://arxiv.org/html/2608.09942>).

---

## Span extraction (verified)

- **R1-Distill-Qwen-7B**: chat template prefills `<think>\n` at the generation prompt, so
  the boundary is the `</think>` token. Guaranteed.
- **OpenThinker3-7B**: the chat template does NOT prefill `<think>`, but the
  OpenThoughts3-1.2M training traces DO contain `<think>`/`</think>` (verified on the
  dataset's first rows), so the model emits them in the response text. **The realized
  emission rate must be checked at smoke time**; below ~99% the answer-span split needs a
  declared fallback and the shortfall is reported, never silently dropped.

## Positioning — a live disagreement this design adjudicates

**Camp A, the answer is already there pre-CoT:**
- "Catching rationalization in the act" (<https://arxiv.org/html/2603.17199>): pre-CoT
  probes at the first decoding step predict motivated reasoning as well as a GPT-5-nano
  monitor with full trace access; end-of-CoT probes recover the hinted choice at >86%.
  Qwen3-8B thinking mode; MMLU / AQuA / ARC-C / CSQA. **Binary and small-multiclass
  classification only — explicitly does NOT regress onto answer activations.**
- "Therefore I am. I Think." (<https://arxiv.org/html/2604.01202>): tool-call decisions
  >90–95% predictable from activations before any reasoning token.
- "Reasoning Models Know When They're Right" (NYU): correctness probe reads 0.79 AUROC
  from the very first reasoning step, 0.95 by completion. DeepSeek-R1-Distill series,
  QwQ-32B.

**Camp B, the answer is computed during CoT:**
- "LLMs Faithfully and Iteratively Compute Answers During CoT"
  (<https://arxiv.org/abs/2412.01113>): on multi-step arithmetic, linear probes for the
  final answer improve MONOTONICALLY along the reasoning chain.

**Contribution.** Every Camp-A result probes a LOW-BANDWIDTH target (a binary label, a
4–5 way choice, a correctness flag). Plot 7 regresses the FULL answer state and reads it
by retrieval among a candidate pool, which is a far higher-bandwidth claim, and crosses it
with a graded necessity dial. Camp B's monotone-improvement result is a sharp prior
prediction that cell A should fail precisely where necessity is high while cell D holds.
Plot 8 then adds the axis neither camp has: what CoT TRAINING does to the map.

## Cost drivers to size at plan time

- Reasoning traces are LONG. `max_new_tokens` must be far above the 2048 default
  (CLAUDE.md: reasoning/CoT models need far more); budget 8k–16k and report the realized
  cap-hit fraction with a pre-registered re-gen trigger (default: >2% per cell ⇒ re-generate
  at ≥2× the cap).
- A generation corpus plus a map refit is needed per model arm; three targets per arm
  (answer / CoT / CoT+answer) come from the SAME generation, so the extra targets are
  nearly free once traces exist.
- Activation capture over long sequences: decide capture-during-generation vs a
  teacher-forced re-pass; the re-pass cost scales with trace length.
- Check #1336's 55 banked turnstores for fitness BEFORE generating anything.

## Open decisions for the planner / user

1. Model arms for round 1: the same-geometry pre/post pair (Qwen2.5-7B-Instruct →
   OpenThinker3-7B) alone covers both plots and is the cheapest complete story. R1-Distill
   and Qwen3-8B add recipe-robustness and the thinking toggle respectively.
2. Whether to add intermediate think-block positions `t ∈ (0,1)` as a
   predictability-vs-CoT-position trajectory. They come from the same forward pass, so
   they are cheap once traces exist, and the trajectory is the direct test of Camp B's
   monotonicity claim.
3. Free directional read first: the frozen #779 map applied cross-model to OpenThinker3
   contexts on the 1,319 GSM8K test questions, binned by calculator-step count. Weak
   (cross-model transfer, so a drop is partly confounded) but near-zero GPU.

## Provenance

Originating chat, 2026-08-24: "I want to design an experiment to check if our mapping does
a lot worse on questions where CoT is in some sense NECESSARY. find a
dataset/model/framework for this"; then the scoping call that necessity be operationalized
as measured CoT lift and that the lift "doesn't have to be our model"; then the redirect
"wait but we want to predict directly from context to the post CoT answer, and do this in
CoT trained models"; then the Plot 7 / Plot 8 framing, verbatim:

> Plot 7: Relationship to CoT / Plot: R^2 and acc@1 for: post context vector → answer /
> post context vector → CoT only / post context vector → CoT + answer / post CoT vector →
> answer / on a corpus that DOES vs DOESN'T require CoT. Claim: TBD. Transition: We then
> wanted to see how this mapping was affected by CoT training.
> Plot 8: Relationship to CoT training. Question: can pre CoT trained context vector
> predict post CoT answer / how does the mapping change.
