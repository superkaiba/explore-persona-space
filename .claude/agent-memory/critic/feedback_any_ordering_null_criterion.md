---
name: Any-fixed-ordering null criteria are mis-calibrated
description: No-ordering-repeats-in-3-seeds criteria fail ~44% under pure noise and ~always under any stable non-IV unit effect; the calibrated null-side test is the SAME pre-registered IV-monotone ordering as H1, expected to fail (#603)
type: feedback
---

When a null-side hypothesis ("prior does NOT predict norm") is operationalized as "≤1 of k seeds matching ANY fixed ordering of m units", check the calibration: with k=3, m=3 (6 orderings), P(some ordering repeats in ≥2/3 seeds) = 1 − (6·5·4)/6³ ≈ 0.444 under pure seed noise — and ≈ 1 under ANY stable unit-level effect unrelated to the IV (hand-written prompts have stable norm/length differences near-certainly). The clause is then near-unsatisfiable, so the decision-table cell needing "DV-null" can almost never be claimed even when the null is true. The calibrated null-side test is the SAME IV-monotone pre-registered ordering used for the positive hypothesis (tail 16/216 for ≥2/3), expected to FAIL — plus the joint-permutation contrast statistic.

**Why (#603, P3′ write decomposition):** §3 H2 used "≤1/3 seeds matching any fixed ordering" while §6(b) correctly pre-registered the specific predicted ordering on norm — the two contradicted each other on the headline decision table; literal implementation would misclassify the P3′-true world with high probability.

**How to apply:** for any decision table keying a cell off "no consistent ordering / no stable ranking" across replicates: compute P(some ordering repeats) under the noise null and ask whether stable non-IV unit effects would trivially produce a repeat. If yes, REVISE to the specific IV-monotone ordering test (or the contrast statistic). Related: feedback_frozen_y_axis_eligibility, feedback_envelope_union_prefalsified_flat.
