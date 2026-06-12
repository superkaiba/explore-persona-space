---
name: Frozen-y-axis eligibility arithmetic
description: When success criteria count concordant groups against a FROZEN y-axis, check per-group y-signal first — an x-side regime fix can leave the criterion unsatisfiable from true signal, and asymmetric exclusion clauses create a noise false-positive path
type: feedback
---

Rule: before approving a success criterion of the form "≥k of N groups show correlation p<α against a frozen/inherited y-axis," compute the per-group y-axis spread yourself (std, range, cells beyond a meaningful threshold) and count how many groups have REAL y-signal. If the eligible set is smaller than k, the criterion is unsatisfiable from true signal (false-FAIL by construction); if the plan keeps ineligible groups in the count, the criterion is reachable only through noise concordance (false-PASS path).

**Why:** #480 followup-2 fixed its x-axis regime defect (the [5,12]-band vs emission-DV contradiction) but kept "≥2 sources beyond SE concord" as P2 while the frozen #411 y-axis had real signal on exactly 2 of 6 sources (SE std 0.174, assistant 0.161; the other four std 0.011–0.026, zero cells |Δ|>0.10, diagnosed as noise in #470's own clean-result). Beyond SE the eligible set was {assistant} — ≥2 impossible from signal. Worse, the pre-registered residual excused floor-y sources only when DIScordant ("behavioral-side-limited, not discordant") while P2 would have counted them when concordant — asymmetric exclusion = selection bias baked into the interpretation contract.

**How to apply:** (a) the y-eligibility threshold must be numeric and ex ante (e.g. ≥3 cells |Δ|>0.10 or y-std ≥ 0.05); (b) exclusion must be SYMMETRIC — ineligible groups' stats are descriptive-only for both success AND falsification; (c) the k in "≥k groups" must match the actual eligible-set size. Also check the plan's characterization of prior-round per-group gate outcomes against the round's JSON (`informative: true/false`) — plans drift toward calling concordance-null groups "uninformative," which mis-baselines the recipe-validation criterion.
