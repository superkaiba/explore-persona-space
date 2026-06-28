---
title: 'r_B extraction-method divergence: on-policy-instruction (A) vs teacher-forced
  (B) vs instruct-and-strip (C) across 3 behaviors'
kind: experiment
tags: []
created_at: '2026-06-24T23:55:41Z'
has_clean_result: false
parent_id: 660
origin_prompt: Can you run an issue in the background to test A,B,C on 3 different
  behaviors and see how much they diverge?
goal: Quantify how much the behavior read-out r_B diverges across three extraction
  methods — (A) on-policy instruction-present, (B) off-policy teacher-forced, (C)
  instruct-and-strip — over the same question set for 3 behaviors (sycophancy, refusal,
  EM), measuring pairwise cosine divergence per layer, the projection of r_B^A onto
  the instruction context axis (c_pos-c_neg) as the context-confound magnitude, and
  whether the methods differ in A3.3-style held-out predictive quality (r_B^T v0 predicting
  E0) — to decide the program's (#660) A3.3 r_B recipe.
---
## Goal

Quantify how much the behavior read-out r_B diverges across three extraction methods — (A) on-policy instruction-present, (B) off-policy teacher-forced, (C) instruct-and-strip — over the same question set for 3 behaviors (sycophancy, refusal, EM), measuring pairwise cosine divergence per layer, the projection of r_B^A onto the instruction context axis (c_pos-c_neg) as the context-confound magnitude, and whether the methods differ in A3.3-style held-out predictive quality (r_B^T v0 predicting E0) — to decide the program's (#660) A3.3 r_B recipe.

All three are diff-in-means of **response-averaged residual-stream activations over the
SAME question set**, per layer (matching Persona Vectors' position choice). They differ
only in how the behavior-present vs behavior-absent responses are obtained + the
extraction context:

- **(A) on-policy, instruction-present:** generate responses under a positive
  ("be sycophantic") vs negative ("be honest") system-prompt instruction, judge-filter
  on expression (keep pos>threshold, neg<threshold), extract activations WITH the
  instruction still in context. This is Persona Vectors' exact method (arXiv 2507.21509
  §2 + App. "Direction extraction pipeline").
- **(B) off-policy, teacher-forced:** teacher-force externally-generated
  behavior-present vs behavior-absent responses to the same questions under the
  DEFAULT (no-instruction) context.
- **(C) instruct-and-strip:** generate on-policy under the instruction, judge-filter,
  then re-extract activations by forwarding the model's OWN responses under the DEFAULT
  instruction-stripped context (the `.claude/rules/on-policy-completions.md` tier-2
  recipe applied to activation extraction).

## What to measure

For each of 3 behaviors (proposed: **sycophancy, refusal, emergent-misalignment** — the
planner may adjust for contrast-battery availability; reuse the #658 contrast types
where they exist):

1. **Pairwise cosine divergence** — `cos(r_B^A, r_B^B)`, `cos(r_B^A, r_B^C)`,
   `cos(r_B^B, r_B^C)` — per layer AND at the per-behavior selected layer.
2. **Context-confound magnitude for (A)** — the projection of `r_B^A` onto the
   instruction context axis `(c_pos − c_neg)`: how much of A's direction is the
   instruction context rather than the behavior. This is the quantity that motivated
   the experiment.
3. **Does the divergence matter?** — an A3.3-style held-out predictive check: does each
   method's `r_B` predict judged expression `E0` (`r_B^T v0` over held-out contexts)
   equally well? Divergent-but-equally-predictive is a weaker concern than
   divergent-and-differently-predictive; report both the geometry (1-2) and the
   downstream predictive quality (3).

## Model + grounding

- **Qwen2.5-7B-Instruct** (project + program default; Persona Vectors' main subject model).
- Ground in: Persona Vectors extraction (arXiv 2507.21509 — response-avg, judge-filter
  >50/<50, layer selected by steering); the leakage theory `r_B` (A3.3, `E ≈ r_B^T v`);
  #658's foundation store. Read the theory doc (`docs/leakage_theory_paper.tex`) before designing.
- Judge = `claude-sonnet-4-5-20250929` (project policy); Batch API if the judge set is large.
- On-policy / teacher-forcing discipline per `.claude/rules/on-policy-completions.md`
  + the #432→#456 teacher-forcing-artifact scar (B is the one exposed to it — the
  externally-generated responses must be flagged as the off-policy arm).

## Reuse

- Reuse #658's question sets / contrast batteries + `v0(C)` store + judge infra where
  applicable (the A3.3 predictive check needs `v0(C)`, already extracted in #658).
- Net-new: the per-method (A/B/C) extraction passes + the cosine / context-projection
  divergence analysis.

## Why / context

Informs **#660 Phase 1's A3.3 `r_B` recipe**: if A's context-confound is negligible
(high `cos(r_B^A, r_B^C)`), the cheaper instruction-present method suffices; if it
diverges, the program adopts the instruct-and-strip (C) recipe. Surfaced when the
Persona-Vectors comparison raised the prompt-class-vs-judged-expression + context-vector
confound for `r_B`.

## Provenance

Originating user request (verbatim): "Can you run an issue in the background to test
A,B,C on 3 different behaviors and see how much they diverge?" — following the discussion
of averaging over the same questions with sycophantic-vs-non-sycophantic responses and
the on-policy (instruction-in-prompt) vs off-policy (teacher-forced) extraction choice,
and the worry that the instruction introduces a context-vector confound.
