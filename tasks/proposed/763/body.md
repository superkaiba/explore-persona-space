---
title: 'Matched-probe v0->E0 predictor re-measurement — phase 2 (5 low-m behaviors:
  deception/fact/format/self_report/persona_drift, >=50 judgments + matched v0)'
kind: experiment
tags: []
created_at: '2026-06-30T09:48:56Z'
has_clean_result: false
parent_id: 658
origin_prompt: do statistical now; [then] run phase 2 in background with happy coder
  — collect >=50 judgments + matched v0 for the 5 low-m behaviors and re-read the
  predictor with the GLM
goal: 'Extend the matched-probe v0(C,B)->E0(C,B) predictor re-measurement to the 5
  behaviors #658 measured at only 8 judgments/context (deception, fact_expression,
  format_style, self_report, persona_drift): collect >=50 eliciting probes/judgments
  per behavior so v0 and E0 share probes AND m is high enough for a trustworthy reliability
  ceiling, then read the predictor with a precision-weighted binomial GLM (primary)
  vs ridge vs a properly-extracted persona-vector baseline. Motivated by the in-session
  GLM-vs-ridge finding that ridge was optimistic by 0.05-0.11 at m=8.'
relates_to:
- leak-predictor
---
# Matched-probe v0→E0 predictor re-measurement — phase 2 (the 5 low-m behaviors, ≥50 judgments + matched v0)

## Goal

Extend the matched-probe v0(C,B)->E0(C,B) predictor re-measurement to the 5 behaviors #658 measured at only 8 judgments/context (deception, fact_expression, format_style, self_report, persona_drift): collect >=50 eliciting probes/judgments per behavior so v0 and E0 share probes AND m is high enough for a trustworthy reliability ceiling, then read the predictor with a precision-weighted binomial GLM (primary) vs ridge vs a properly-extracted persona-vector baseline. Motivated by the in-session GLM-vs-ridge finding that ridge was optimistic by 0.05-0.11 at m=8.

## Why (in-session finding that motivates this)

A 2026-06-30 chat re-analysis (the #658 predictor line) established:
- At **high m (sycophancy 2000 / refusal 215 / harmful 115)** the ridge `v0→E0` read is ROBUST: a correctly-specified binomial GLM agrees within ±0.04 (ρ ≈ 0.59–0.67). Those 3 are phase-1 (#761).
- At **m=8** the ridge was OPTIMISTIC by 0.05–0.11 vs the GLM (deception 0.54→0.47, persona_drift 0.65→0.60, self_report 0.61→0.55, format_style 0.62→0.52, fact_expression 0.30→0.18), and the gap grew with floor/ceiling skew. The statistical fix (GLM) gives a more honest but lower number; **only higher m settles whether these behaviors are genuinely linearly decodable.** This task supplies the higher m.

## Design (single change per behavior vs #658: probe pool size + matched v0)

For each of the 5 behaviors:
- **Author/assemble a ≥50-probe ELICITING pool** (behavior-specific — generic Betley probes do NOT elicit these behaviors with dynamic range). Prefer established datasets/benchmarks (data-realism tier-2) where one exists for the behavior; the planner picks the source + tier per behavior and justifies it. Author programmatic probes only as a recorded last resort.
- **Generate on-policy completions** from base `Qwen2.5-7B-Instruct` over each (context × behavior-probe), **capture answer-side activations** at all 28 layers → matched `v0(C,B)` = mean answer activation over behavior B's ≥50 probes per context.
- **Judge** each completion with `claude-sonnet-4-5` (Anthropic Batch API) → `E0(C,B)` at ≥50 judgments/context.
- **Analyze (0-GPU):** ridge AND precision-weighted binomial GLM `v0(C,B)→E0(C,B)` LOCO over the 50 contexts — PCA-k by NESTED CV (do NOT fix k=10; the in-session k-sweep showed k=10 slightly conservative and k≥~40 degenerate at n=50), shuffle-label / control-task null, cluster-bootstrap CIs, and lay ρ next to the per-behavior reliability ceiling `√(r_yy)` (the #742 bracket). **GLM is the registered primary read; ridge is the comparator; report both vs the proper persona-vector baseline.**
- **Persona-vector baseline MUST use the proper recipe** per `.claude/rules/persona-vectors-recipe.md` (content-matched 5 pos/neg system-prompt pairs over a shared question set, judge-filtered, response-avg, diff-of-means) — NOT #658's two-corpora-no-judge-filter version.

## Reuse

- The 50-context battery from #594/#658 (seed-42, schema-validated).
- The matched-`v0` capture + ridge/GLM analysis pipeline from sibling **#761** (build once, both phases share it; the planner identifies the reusable code).
- broad_em is EXCLUDED — it floors on the base model (judged-rate std ≈0.008) regardless of probe count, so more m cannot help. sycophancy/refusal/harmful are phase-1 (#761), not repeated here.

## Resource estimate (planner to refine)

- 5 behaviors × ≥50 probes × 50 contexts ≈ 12.5k (context, probe) pairs needing on-policy generation + answer-activation capture. Anchored on #658's ~16 GPU-h per 2,400 gen+capture pairs → **~80 GPU-h** (vectorize the capture; never batch-1). Judging ~12.5k calls via the Anthropic Batch API (mostly-free, deadline-bounded poller).
- Compute lane: a GPU capture intent (single/multi-GPU); the analysis phase is off-GPU on the VM. Plan-time disk-footprint check per the activation-store-on-VM incident (#658 Phase-1).

## Relation to the line

Same predictor question as #658/#742/#761 ("can the base-model mean answer activation predict behavioral expression?"), extended to the 5 behaviors where #658's 8-judgment measurement was too noisy to trust. Together with #761 (the 3 high-m behaviors, matched-v0) this gives a clean, matched-probe, properly-modelled (GLM) read across all behaviors with dynamic range. `docs/open_questions.md` anchor: `leak-predictor`.

## Provenance

User chat directive (2026-06-30): after the statistical (GLM) pass showed ridge was optimistic at m=8 and that only higher m settles the noisy behaviors, "run phase 2 in background with happy coder" — collect ≥50 judgments + matched v0 for the 5 low-m behaviors and re-read the predictor with the GLM.
