---
name: Ratio-sweep negative-pairing arithmetic
description: In fixed-total-N positive-ratio sweeps, a "one negative per positive question per class" pairing rule is feasible only when positives <= N/(classes+1) — check per-arm arithmetic before approving
type: feedback
---

In any fixed-total-N install-mix ratio sweep with C negative classes and a
stated rule "each negative class includes one row for EVERY positive
question," full pairing is feasible only when positives p <= N/(C+1)
(per-class quota (N-p)/C >= p). High-ratio arms violate it: at N=6000, C=4,
the r50 arm (p=3000) has quota 750 < 3000 and ZERO non-positive questions
left in a 3000-question pool, so even the "fill from the rest of the pool"
clause is empty.

**Why:** Task #543 plan v2 shipped exactly this — the pairing rule was
unsatisfiable for the r50 baseline and r25 arms while the plan's §5 claimed
"contrastive pairing preserved" as a control. The implementer's silent
resolution (subset pairing? question reuse?) becomes an undocumented second
variable sitting directly on the manipulated axis: pairing coverage then
varies across arms (25% / 75% / 100% / 100%), which is an alternative
mechanism for any survival gradient and the analyzer can't weigh it because
the plan asserts it doesn't exist.

**How to apply:** For any ratio/composition sweep at fixed total rows,
recompute the pairing-rule arithmetic per arm. If unsatisfiable in any arm,
REVISE: demand (i) an arithmetically valid, arm-uniform sampling rule,
(ii) corrected control claims, and (iii) realized per-arm pairing-coverage
stats logged as a reported diagnostic. Note that coverage-vs-ratio
covariation is INHERENT at fixed N (coverage=1 only below the p<=N/(C+1)
threshold) — even after the fix it belongs in the analyzer's
alternative-explanations list for a positive result.
