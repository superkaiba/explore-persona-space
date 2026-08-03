---
name: Codex flags the registered noise-floor/reliability statistic as a unit-mismatch bug
description: Codex code-review FAIL that the analysis noise-floor scalar is the "wrong kind of quantity" / wrong units — verify the PLAN's registered noise-floor source fn before crediting; the impl usually re-implements exactly what the plan registered
type: feedback
---

When Codex FAILs a verdict-logic / analysis diff arguing a threshold scalar is the
"wrong kind of quantity" — a unit mismatch, e.g. "this is a test-retest RELIABILITY
ceiling, not a ρ-DIFFERENCE tolerance, so `gap >= -noise_floor` lets A be worse than
C by a whole reliability unit and still PASS" — DO NOT credit the FAIL until you read
the PLAN's registered definition of that statistic AND the parent function it cites.

**Why:** the noise-floor / tolerance scalar in pre-registered verdict logic is a
DESIGN CHOICE the plan pins, not something the implementer invents. The defect Codex
describes is frequently the plan itself, re-litigated as a code bug — the
"methodology-choice-as-bug" / "Codex flags the plan's own pre-registered rule"
family. #661 r2 (2026-06-24): Codex FAILed `per_behavior_noise_floor()`
(`issue661_analysis.py:475`) as a split-half reliability ceiling threaded into
`decide()` at `:735` as `gaps[b] >= -noise_floor[b]`, calling it a unit mismatch.
But plan v3 §4.4/§6.4 REGISTERED "noise floor via `issue658_fit_predictors.noise_floor()`",
and that registered fn (`issue658_fit_predictors.py:616`, docstring "test-retest ρ
distribution; PASS bar = 95th pct") IS the same split-half reliability computation —
AND #658's own A3.3 verdict (`:787`) uses it as a ρ-difference tolerance
(`best_lin["rho"] >= mlp_rho - noise._p95`), the identical idiom. The impl was a
faithful re-implementation of the registered statistic. PASS (upheld Claude).

**How to apply:** for any Codex "wrong units / wrong statistic" FAIL on analysis /
verdict-logic code:
1. Grep the plan for the statistic's REGISTERED source (often `via <parent>.fn()` or
   "reuses #<M>'s X"). If the plan names a source fn, read THAT fn and compare the
   computation to the impl. A faithful re-implementation of the registered source is
   NOT a defect — even if Codex's preferred statistic would be "more principled".
2. Check how the PARENT issue USES the scalar. A reliability ceiling subtracted from
   one ρ as a difference-tolerance is a coherent, intentional idiom (the test-retest
   ρ ceiling = the largest ρ-difference attributable to target noise) — not a unit
   bug, when the parent does exactly that.
3. Codex's worst-case numeric example ("0.8 reliability → 0.8 allowed gap") is moot
   if the band is the pre-registered one; realism of the example does not convert a
   registered design into a deviation.
4. SEPARATE the coverage critique: "tests inject the scalar into the decision fn but
   never test the upstream computation" is often factually TRUE and worth a standing
   rec (cross-check the re-impl against the registered source on a synthetic fixture),
   but a missing unit test on a verified-equivalent re-implementation is Real-but-
   non-blocking, never a FAIL.

CONTRAST with the genuine-defect case (`feedback_claude_gate_unit_vs_preregistered_verdict_logic.md`,
`feedback_gate_design_vs_recoverable_robustness_read.md`): a verdict-logic defect is
binding when the impl DEVIATES from what the plan registered (affirmative misfire,
barred amendment, capture loss). Here the test is "does the impl match the registered
statistic?" — if yes, Codex's "should be a different statistic" is out-of-scope
re-litigation; if no (impl silently swapped in a different statistic than the plan
named), THAT is the binding defect. The pivot is the plan's registered source, always.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Codex flags the registered noise-floor/reliability statistic as a unit-mismatch bug](feedback_codex_flags_registered_noise_floor_statistic.md) — "wrong kind of quantity / wrong units" FAIL on a verdict-logic scalar; read the PLAN's registered noise-floor SOURCE fn + how the parent USES it before crediting; faithful re-impl of the registered statistic = out-of-scope, PASS. #661 r2.
