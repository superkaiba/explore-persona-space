---
name: Absolute survival floor on re-anchored sub-100% baseline
description: When a parent's absolute emission floor (e.g. #557's 5%/2%/2.5%) is reused on a design whose pre baseline is deliberately sub-100%, check floor x baseline arithmetic — anchor-level RELATIVE retention can mechanically fail the absolute floor
type: feedback
---

Rule: when a plan re-anchors retention to a sub-100% pre baseline (post/pre ratio
headline) but keeps the parent's ABSOLUTE survival floor as the registered
success criterion, multiply the floor through the minimum admissible baseline
before approving. Example (#570): inclusion gate admits pre as low as 10%
(20/200); at #557's own 37% relative retention, post = 3.7% pooled (22/600,
Wilson [2.4, 5.5]) — fails the 5% absolute floor despite anchor-level retention.
Same regime starves the differential: a 2.5x arm ratio at pre=12% gives
overlapping CIs (20/600 [2.2,5.1] vs 8/600 [0.7,2.6]), which a registered
"CIs overlap → null reproduced / falsified" branch then narrates as an
effect-confirmed null.

**Why:** the absolute floor was calibrated where pre = 100%; on a re-anchored
denominator it conflates "low retention" with "low baseline," and the equal-fate
branch conflates "no differential" with "too few events to resolve one."

**How to apply:** not a REVISE when (a) per-completion pre/post records persist,
(b) the relative-retention ratio + bootstrap CI is registered alongside, and
(c) seed-level concordance is reported — the analyzer can re-read. Flag as the
TOP concern: sub-floor post at anchor-level R must not be narrated as "does not
survive," and overlapping CIs at low pooled counts must use the
"indistinguishable given the variance" frame (report the minimum detectable
ratio given realized pre). Rough power note at n=600/arm: ~3-4x arm ratios
resolve at pre >= ~12%; smaller ratios land in the middle zone.
