---
name: Conjunction-gated label taxonomies must partition the data
description: leak = (θ AND k×SE) with no-leak = |delta| < θ leaves gate-failing cells classless; enumerate the per-unit CI-state combos, pull actual per-cell n deciles, demand a named indeterminate/discordant class (#593, #611)
type: feedback
---

A binary leak/no-leak DV with a significance condition on the leak side only (leak = delta ≥ θ AND ≥ 2·SE_cell; no-leak = |delta| < θ) does NOT partition: anything with delta ≥ θ failing the SE gate has NO class. The implementer's silent default (else → no-leak) puts strongly-positive cells in the negative class and biases AUC toward null — manufacturing "factor doesn't classify".

**Why (#593, EM arm):** plan grounded the gate on "survivor n ≈ 65–100 → SE ≈ 0.05" but actual per-cell survivor deciles were [1, 12, 26, 48, 94, 192, 311] (median 48 → worst-case SE ≈ 0.10); ~21 of 61 leak candidates (~1/3 of the positive class) failed the 2SE gate with no defined label. Also asymmetric: the suppressed side had no SE gate, so an n=1 cell could count as suppressed.

**Recurrence (#611):** a SURVIVES/PARTIAL/ABSENT/REVERSED 2-persona taxonomy — the combo (one persona CI-clear expected, the other CI-clear OPPOSITE) satisfied NONE of the four definitions. Enumerate the 3^k per-unit CI-state combos (expected/straddle/opposite) and check every combo maps to a label; demand a named DISCORDANT class. There it was a Concern not REVISE (zero-GPU re-analysis, per-cell CIs persisted, one-line fix at implementation).

**How to apply:** (1) for any threshold + significance conjunction, enumerate the classes and check they partition; demand a named indeterminate/underpowered class (excluded from fits, counted in outputs). (2) Pull the ACTUAL per-cell n distribution from the cited run files (deciles, not the plan's quoted range) and quantify the orphan band. (3) Check gate symmetry — if the leak side is SE-gated, ask what tiny-n cells do to the other classes (min-n floor or per-cell SE shipped).
