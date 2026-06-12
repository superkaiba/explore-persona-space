---
name: Null-band reachability vs realized CI width
description: When an amendment/follow-up reuses a registered equivalence (null) band, check the band's TOTAL width against the CI width the SAME machinery already realized in the parent run — an unreachable null branch on the modal expected outcome is a REVISE
type: feedback
---

Rule: for any plan that carries a registered three-way call with an equivalence
("null") band — e.g. "Null if the 95% CI lies entirely within ±0.03" — compare
2×band against the CI width the identical statistic ALREADY produced on the
parent data. If realized width > 2×band (and the new read has equal-or-less
paired data), the null branch can never fire, even with a perfectly centered
point estimate, so the run is structurally pre-committed to "indeterminate".

**Why:** Task #612 dose-matched amendment (2026-06-12): parent endpoint
`paired_arm_contrast` CI width = 0.0697 > 0.06 = 2×(±0.03 null band); the
amendment's STATED expectation was the null branch ("dose, not radius stands"),
and the band-entry read had strictly fewer rows (seed-42 villain-only). The
plan would have burned 5 GPU-h + ~144k judge calls to re-render "indeterminate"
by construction. Same family as the at-planning-time-unreachable ρ-difference
null leg the #612 parent itself fixed in round 1, and the N=12 Spearman
threshold memory — but this variant only becomes checkable at the AMENDMENT
stage, when the realized CI width exists in the committed analysis JSON.

**How to apply:** Statistics lens, any follow-up/amendment that re-anchors a
registered decision rule: read the parent's realized CI from the committed
analysis artifact, compare to every equivalence band in the rule, and check
whether the new read's pairing has fewer effective clusters. Fix that preserves
registration: pre-register a secondary determinate read at the ALREADY-
registered support threshold (e.g. "CI entirely within ±support_min →
bounded-below-support") rather than silently widening the null band.
