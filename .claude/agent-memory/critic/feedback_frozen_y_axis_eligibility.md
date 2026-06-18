---
name: Frozen-y-axis eligibility arithmetic
description: k-of-N-groups-concord-vs-frozen-y criteria need per-group y-signal counts computed ex ante; asymmetric exclusion (excuse-discordant, count-concordant) is a baked-in false-positive path (#480 f2)
type: feedback
---

Before approving "≥k of N groups show correlation p<α against a frozen/inherited y-axis", compute the per-group y-axis spread yourself (std, range, cells beyond a meaningful threshold) and count how many groups have REAL y-signal. Eligible set < k → the criterion is unsatisfiable from true signal (false-FAIL by construction); ineligible groups kept in the count → reachable only through noise concordance (false-PASS path).

**Why (#480 f2):** the plan kept "≥2 sources beyond SE concord" while the frozen #411 y-axis had real signal on exactly 2 of 6 sources (the other four std 0.011–0.026, zero cells |Δ|>0.10, diagnosed as noise in the parent's own clean-result) — beyond SE the eligible set was {assistant}, so ≥2 was impossible from signal. Worse, the residual excused floor-y sources only when DIScordant while counting them when concordant — asymmetric exclusion = selection bias baked into the interpretation contract.

**How to apply:** (a) the y-eligibility threshold must be numeric and ex ante (e.g. ≥3 cells |Δ|>0.10, or y-std ≥ 0.05); (b) exclusion must be SYMMETRIC — ineligible groups are descriptive-only for both success AND falsification; (c) the k must match the actual eligible-set size. Also check the plan's characterization of prior-round per-group gate outcomes against the round's JSON (`informative: true/false`) — plans drift toward calling concordance-null groups "uninformative," mis-baselining the recipe-validation criterion.
