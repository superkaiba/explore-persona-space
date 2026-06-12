---
name: Re-judge agreement criteria vs boundary-adjacent cell mass
description: Before accepting a "% class agreement" decision rule for a DV-correction re-judge round, count frozen cells within noise distance of the class boundary; full-panel conditional recompute = per-cell drift/denominator decomposition
type: feedback
---

When a re-scoring round (DV-correction, judge re-run) pre-registers a binary
"reproduces vs flips" rule as N-class label agreement >= X% at threshold tau,
pull the FROZEN panel and count cells within ~1-2 corrected-side SE of the
tau boundary. In #591 e5: 24/138 cells within 0.03 and 50/138 within 0.05 of
the +/-0.10 boundary, vs a <=14-flip "reproduces" criterion — so a small
uniform compression (e.g. gibberish rows pulling all rates toward the judge's
incoherent-text mean) mechanically fires the "flips/proxy distorted" branch
via knife-edge churn, without survivor-selection distortion.

**Why:** the agreement % conflates two mechanisms (denominator selection vs
small global shift); the labeling branch can be decided by boundary mass.
NOT a REVISE when per-cell deltas (both panels) + per-rollout verdicts are
persisted and flips are stratified — analyzer guidance: stratify flips by
boundary distance |frozen delta − tau| IN ADDITION to survivor count.

**How to apply:** (1) compute the boundary-distance distribution yourself
from the frozen join; (2) check the design includes a same-verdicts
conditional recompute over the FULL panel — that makes every cell an anchor:
Δ_total = (conditional_fresh − frozen) [judge drift + survivor resampling] +
(all-rollouts − conditional_fresh) [pure denominator], a per-cell
decomposition far stronger than a 5-cell anchor gate; if present, ambiguous
anchor-gate outcomes are recoverable post-hoc. (3) Anchor-gate taxonomy
check: pass/warn/KILL bands often leave middle configs undefined (one cell
>0.10; two cells in warn band) — concern, not Must-Fix, when must-escalate
covers KILL and fail-loud is the house default.
