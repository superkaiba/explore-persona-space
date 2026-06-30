---
title: Matched-probe v0->E0 predictor re-measurement — phase 2 (5 low-m behaviors,
  >=50 probes + GRADED matched v0)
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
  format_style, self_report, persona_drift): collect >=50 eliciting probes per behavior
  AND judge E0 with a GRADED multi-sampled 0-100 score (per .claude/rules/llm-judging.md)
  as the PRIMARY DV (binary judge-rate retained as a validated headline); read the
  predictor with ridge + precision-weighted GLM vs a properly-extracted persona-vector
  baseline.'
relates_to:
- leak-predictor
---
# Matched-probe v0→E0 predictor re-measurement — phase 2 (the 5 low-m behaviors, ≥50 probes + GRADED matched v0)

## Design update (2026-06-30 re-plan — supersedes the binary framing below)

Two changes since this task first dispatched:
1. **GRADED DV, not binary.** `.claude/rules/llm-judging.md` (merged via #765) now standardizes behavior-expression DVs: a **graded multi-sampled 0-100 judge score is the PRIMARY ranking/regression DV**, with the binary judge-rate retained as a validated headline. The original "binary E0 + GLM" framing is the now-deprecated approach — the GLM was a statistical patch for binarization, but graded measurement removes the dichotomization loss (~0.798 attenuation, worse near a floor) at the source. So phase-2's `E0` is judged GRADED per the new rule.
2. **Pod-wedge recovery.** The first dispatch's RunPod pod (`z2g3cxyhudmyrf`) hit the RUNNING-but-no-port host wedge and was terminated (no artifacts produced). The systemic auto-recovery for that wedge class is under separate investigation. For THIS re-run: prefer the GCP-first auto-router; if the activation-capture requires the interactive RunPod SSH-MCP pattern (as #761 used), a FRESH pod is fine (the wedge was host-specific — #761's RunPod run succeeded), but the planner should note the backend choice and the §9 disk-footprint check.

## Goal

Extend the matched-probe v0(C,B)->E0(C,B) predictor re-measurement to the 5 behaviors #658 measured at only 8 judgments/context (deception, fact_expression, format_style, self_report, persona_drift): collect >=50 eliciting probes per behavior AND judge E0 with a GRADED multi-sampled 0-100 score (per .claude/rules/llm-judging.md) as the PRIMARY DV (binary judge-rate retained as a validated headline); read the predictor with ridge + precision-weighted GLM vs a properly-extracted persona-vector baseline.

## Why (in-session findings that motivate this)

A 2026-06-30 chat re-analysis (the #658 predictor line) + two adversarial deep-research dives established:
- At **high m (sycophancy 2000 / refusal 215 / harmful 115)** the ridge `v0→E0` read is ROBUST: a correctly-specified binomial GLM agrees within ±0.04 (ρ ≈ 0.59–0.67). Those 3 are phase-1 (#761).
- At **m=8** the ridge was OPTIMISTIC by 0.05–0.11 vs the GLM, and the binary DV is doubly penalized: binomial-noise floor at small m AND dichotomization attenuation (~0.798, worse near a floor — Cohen 1983 / MacCallum 2002). The literature verdict (`.claude/rules/llm-judging.md`): use a **graded** DV, multi-sampled, validated against the binary headline. This task supplies BOTH the higher m and the graded DV.

## Design (changes vs #658: probe-pool size + matched v0 + graded DV)

For each of the 5 behaviors:
- **Author/assemble a ≥50-probe ELICITING pool** (behavior-specific — generic Betley probes do NOT elicit these with dynamic range). Prefer established datasets/benchmarks (data-realism tier-2); planner picks source + tier per behavior and justifies. Programmatic probes only as a recorded last resort.
- **Generate on-policy completions** from base `Qwen2.5-7B-Instruct` over each (context × behavior-probe); **capture answer-side activations** at all 28 layers → matched `v0(C,B)` = mean answer activation over behavior B's ≥50 probes per context.
- **Judge E0 GRADED** with `claude-sonnet-4-5` (Anthropic Batch API) per `.claude/rules/llm-judging.md`: a **0-100 anchored-rubric, reason-then-score, one-behavior-per-call** protocol, **multi-sampled** (N pilot, temp > 0, averaged — the substitute for logit-weighting since the Messages API has no score-token logprobs), malformed returns DROPPED never coerced. Record BOTH the graded mean (primary DV) and the binary expressed-rate (validated headline). ≥50 judgments/context.
- **Analyze (0-GPU):** ridge AND precision-weighted GLM on the **graded** `v0(C,B)→E0(C,B)` LOCO over the 50 contexts — PCA-k by NESTED CV (do NOT fix k=10; the in-session k-sweep showed k=10 slightly conservative and k≥~40 degenerate at n=50), shuffle-label / control-task null, cluster-bootstrap CIs, ρ next to the per-behavior reliability ceiling `√(r_yy)`. Report graded-primary vs binary-headline vs the proper persona-vector baseline; per-behavior judge-reliability (test-retest across the multi-samples) per the new rule.
- **Persona-vector baseline MUST use the proper recipe** per `.claude/rules/persona-vectors-recipe.md` (content-matched 5 pos/neg system-prompt pairs over a shared question set, judge-filtered, response-avg, diff-of-means) — NOT #658's two-corpora-no-judge-filter version.

## Reuse

- The 50-context battery from #594/#658 (seed-42, schema-validated).
- The matched-`v0` capture + ridge/GLM analysis pipeline from sibling **#761** (build once, both phases share it; planner identifies the reusable code) — extend it with the graded-judge protocol.
- broad_em EXCLUDED (floors on the base model regardless of m/scale). sycophancy/refusal/harmful are phase-1 (#761).

## Resource estimate (planner to refine)

- 5 behaviors × ≥50 probes × 50 contexts ≈ 12.5k (context, probe) pairs needing on-policy generation + answer-activation capture (~80 GPU-h; vectorize, never batch-1). Graded multi-sampled judging ≈ 12.5k × N calls via the Anthropic Batch API (mostly-free, deadline-bounded poller) — N from the reliability pilot. Analysis 0-GPU.
- Disk-footprint §9 check per the #658 Phase-1 activation-store-on-VM incident.

## Relation to the line

Same predictor question as #658/#742/#761, extended to the 5 low-m behaviors with a graded, matched-probe, properly-modelled DV. Together with #761 this gives a clean read across all behaviors with dynamic range. `docs/open_questions.md` anchor: `leak-predictor`.

## Provenance

User chat directives 2026-06-30: "run phase 2 in background with happy coder"; then after the binary-vs-graded deep-research + the pod wedge, "do path 2" — terminate the wedged pod and re-plan #763 to the graded DV (per the new `.claude/rules/llm-judging.md`) before re-running.
