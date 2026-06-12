---
name: Regime+DV bundled amendments OK via cross-round factorial
description: A "single variable" that bundles anchor regime + regime-appropriate DV is acceptable when both DVs come from the same forward passes in both rounds (2x2 complete across rounds) and the degenerate cell is demoted to a manipulation check; residual step-dynamics alternative is disambiguated free from the parent's stored logit columns
type: feedback
---

Rule: an amendment whose "one variable" bundles the evaluated checkpoint regime
AND a primary-DV swap (e.g. firing-anchor/emission → in-band/log-prob) is NOT a
smuggled-second-variable REVISE when (a) both DVs are computed in both rounds
from the same forward passes, so the regime×DV factorial is complete across
rounds, and (b) the by-construction-degenerate cell (e.g. in-band emission ≈ 0)
is explicitly demoted to a manipulation check instead of run through the stats
package (zero-variance X → NaN Spearman dressed as a result).

**Why:** #480 follow-up 3 (`inband-logprob-concordance`, approved 2026-06-10)
did exactly this cleanly. The verdict definition the orchestrator supplied also
bound: success/kill criteria are analysis-time reads, not gates; bias APPROVE
when the analyzer can recover.

**How to apply:** (1) check the factorial claim against the parent's committed
matrix schema — both DV columns must actually exist in the parent rows (in #480
they did: `emission_rate` + `marker_delta` + four floats per row). (2) The
residual alternative for a sign-flip headline ("ordering genuinely changed with
training steps, not softmax compression") is weighable FREE when the parent
matrix stores the logit columns (`eos_margin_delta` / `delta_z_marker`): rerun
the concordance on the PARENT matrix with x = the logit column — parent-logit
concordant + parent-log-prob inverted ⇒ compression confirmed within-round.
List that as analyzer concern #1, not a REVISE. (3) Companion checks that made
this approvable: y-eligibility k matched eligible-set size with SYMMETRIC
exclusion (both directions), missing control scripts named + scheduled +
dry-run against the parent matrix, parity probe vs recorded in-loop values as
the behavioral backstop for pinned-revision reuse.
