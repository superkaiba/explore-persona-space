---
name: Regime+DV bundled amendments OK via cross-round factorial
description: A "single variable" bundling anchor regime + regime-appropriate DV is acceptable when both DVs come from the same forward passes in both rounds (2×2 complete across rounds) and the degenerate cell is demoted to a manipulation check (#480 f3)
type: feedback
---

An amendment whose "one variable" bundles the evaluated checkpoint regime AND a primary-DV swap (firing-anchor/emission → in-band/log-prob) is NOT a smuggled-second-variable REVISE when (a) both DVs are computed in both rounds from the same forward passes (regime×DV factorial complete across rounds), and (b) the by-construction-degenerate cell (in-band emission ≈ 0) is explicitly demoted to a manipulation check instead of run through the stats package (zero-variance X → NaN Spearman dressed as a result).

**Why:** #480 follow-up 3 (`inband-logprob-concordance`, approved 2026-06-10) did this cleanly; success/kill criteria were analysis-time reads, not gates — bias APPROVE when the analyzer can recover.

**How to apply:** (1) check the factorial claim against the parent's committed matrix schema — both DV columns must actually exist in the parent rows (#480: `emission_rate` + `marker_delta` + four floats). (2) The residual alternative for a sign-flip headline ("ordering genuinely changed with training steps, not softmax compression") is weighable FREE when the parent matrix stores the logit columns: rerun the concordance on the PARENT matrix with x = the logit column — parent-logit concordant + parent-log-prob inverted ⇒ compression confirmed within-round. Analyzer concern #1, not REVISE. (3) Companion checks that made it approvable: y-eligibility k matched the eligible-set size with SYMMETRIC exclusion; missing control scripts named + scheduled + dry-run against the parent matrix; parity probe vs recorded in-loop values as the behavioral backstop for pinned-revision reuse.
