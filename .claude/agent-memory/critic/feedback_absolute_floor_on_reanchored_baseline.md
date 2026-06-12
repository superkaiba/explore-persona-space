---
name: Absolute survival floor on re-anchored sub-100% baseline
description: Parent absolute floors reused on designs whose pre baseline is deliberately sub-100% — multiply the floor through the minimum admissible pre; anchor-level relative retention can mechanically fail it (#570)
type: feedback
---

When a plan re-anchors retention to a sub-100% pre baseline (post/pre ratio headline) but keeps the parent's ABSOLUTE survival floor as the success criterion, multiply the floor through the minimum admissible baseline before approving. #570: the inclusion gate admits pre as low as 10%; at #557's own 37% relative retention, post = 3.7% pooled — fails the 5% absolute floor despite anchor-level retention. The same regime starves the differential: a 2.5× arm ratio at pre=12% gives overlapping CIs, which a registered "CIs overlap → null reproduced" branch then narrates as an effect-confirmed null.

**Why:** the absolute floor was calibrated where pre = 100%; on a re-anchored denominator it conflates "low retention" with "low baseline", and the equal-fate branch conflates "no differential" with "too few events to resolve one".

**How to apply:** not a REVISE when (a) per-completion pre/post records persist, (b) the relative-retention ratio + bootstrap CI is registered alongside, (c) seed-level concordance is reported. Flag as TOP concern: sub-floor post at anchor-level relative retention must not be narrated "does not survive"; overlapping CIs at low pooled counts use the "indistinguishable given the variance" frame (report the minimum detectable ratio given realized pre). Power note at n=600/arm: ~3-4× arm ratios resolve at pre ≥ ~12%.
