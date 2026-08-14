---
title: 'Statistics lens item 17: rate-denominator provenance audit for projection
  rates (shipped)'
kind: infra
tags:
- wfx-rate-denominator-audit
created_at: '2026-08-11T03:08:51Z'
has_clean_result: false
origin_prompt: 'Orchestrator-surfaced workflow-fix after #2054 round coordinated-common-set-regen
  gate-1 ABORT: the plan projection applied the parent''s 49.3% admitted/prejudge
  admission rate as a per-attempt-from-pending success probability; no lens audits
  rate denominator provenance.'
workflow: v1
---
# Statistics lens + verify_plan: audit that every projection rate is applied to the denominator it was measured on (#2054 admitted/prejudge vs per-attempt-from-pending)

## Provenance

workflow_fix_target: .claude/rules/critic-lens-reference.md
Surfaced by: orchestrator, task #2054 same-issue follow-up round `coordinated-common-set-regen` (2026-08-11), after the round's pre-registered gate-1 ABORT.

## Gap

No reviewer lens or mechanical check audits the DENOMINATOR PROVENANCE of a measured rate used in a plan's coverage/yield/sizing projection. A rate measured as X/Y can be silently applied as if it were X/Z (a different base population), and the arithmetic downstream of the substitution is internally consistent — so an arithmetic recompute (which the Statistics critic DID perform on this plan) validates the projection without catching the basis error.

## Incident

Task #2054, round `coordinated-common-set-regen`, plan v12 (fact-checked + 3 Claude lens critics + consistency-checker, all passed on this point): the plan's wave-coverage projection used the parent run's measured 49.3% scaffold-admission rate (measured as admitted/PREJUDGE = 9,722/19,714 — the fact-checker verified exactly this fraction) as the per-attempt success probability FROM PENDING, implicitly assuming every pending (character, conversation) yields a prejudge row per attempt. Realized: the scaffold generator's verbatim-question filter drops 65-69% of generator-kept rows (stable across 3 waves; gen yield 26.6% / 23.4% / 20.6% of requested), so the realized per-attempt-from-pending success was ~9-13% — the projection was ~3.7x optimistic, the projected survivor count (9,855 vs target 9,000) was structurally unreachable at the registered <=3 attempts, and the round ran ~1.6 GPU-h + ~33.7k judge calls into a pre-registered gate-1 ABORT (|S|=2,409 vs floor 4,480). The gate protected the capture spend (working as designed), but the projection error was catchable at plan review: the parent's own pipeline had the same verbatim filter, so the prejudge-per-requested factor existed in the parent artifacts the fact-checker was already reading.

## Proposed fix (planner decides final shape)

1. Add an audit item to the Statistics & Measurement lens (critic-lens-reference.md): for EVERY measured rate a plan uses in a projection (coverage, yield, admission, throughput), the plan must state the rate's measured numerator AND denominator (artifact-grounded) and the projection's application denominator; a mismatch (or an unstated composition of stage-wise rates) is a REVISE. Multi-stage pipelines (generate -> filter -> judge) must chain per-stage rates explicitly, each with its own measured basis.
2. Optional mechanical twin in verify_plan.py (WARN-only): flag a §9/§7 projection line that multiplies a rate against a population token when the rate's grounding sentence names a different denominator token (heuristic; the lens item is the binding arm).
3. Consider a planner.md §9 sizing-basis bullet requiring the stage-chain decomposition for any coverage projection over a filtered generation pipeline.

## Acceptance criteria

1. Statistics lens item added to .claude/rules/critic-lens-reference.md (and the lens's summary line in critic.md if the roster names items), with the #2054 incident as the worked example.
2. LESSONS.md index untouched (no new rule file) OR updated if the fix lands as a new rule file.
3. If the verify_plan arm is implemented: canonical N/A escape + tests in tests/test_verify_plan.py per convention.
4. workflow_lint no-flags green after the edit.
