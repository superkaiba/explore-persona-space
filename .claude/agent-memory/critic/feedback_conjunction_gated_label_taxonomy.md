---
name: Conjunction-gated leak labels make the class taxonomy non-exhaustive
description: When leak = (delta >= theta AND >= k*SE_cell) but no-leak = |delta| < theta, gate-failing cells (delta >= theta, < k*SE) have NO class — check exhaustiveness and quantify the orphan band from the actual per-cell n
type: feedback
---

A binary leak/no-leak DV with an extra per-cell significance condition on the
leak side only (leak = delta >= theta AND delta >= 2*SE_cell; no-leak =
|delta| < theta; suppressed = delta <= -theta) does NOT partition the cells:
anything with delta >= theta that fails the SE gate is in no class. The
implementer's natural silent default (else -> no-leak) puts strongly-positive
cells in the negative class and biases AUC toward null — manufacturing the
"factor doesn't classify this behavior" conclusion.

**Why:** #593 plan v1 (EM arm): plan grounded the gate on "survivor n ≈ 65-100
→ SE ≈ 0.05" but actual per-cell survivor deciles were [1, 12, 26, 48, 94,
192, 311] (median 48 → worst-case SE ≈ 0.10). ~21 of 61 EM leak candidates
(~1/3 of the positive class) failed an approximate 2SE gate and had no defined
label. Also asymmetric: the suppressed side had no SE gate, so an n=1 cell
could count as suppressed on one coherent sample.

**How to apply:**
1. For any DV with threshold + significance conjunction, enumerate the classes
   and check they partition the data; demand a named 4th class
   ("indeterminate/underpowered", excluded from the fit, counted in outputs).
2. Pull the ACTUAL per-cell n distribution from the cited run files (deciles,
   not the plan's quoted range) and quantify the orphan band before rating.
3. Check gate symmetry: if the leak side is SE-gated, ask what tiny-n cells do
   to the no-leak and suppressed classes (min-n floor or per-cell SE shipped).

**Recurrence (#611, 2026-06-11):** same hole in a 2-persona verdict taxonomy
SURVIVES/PARTIAL/ABSENT/REVERSED — the combo (one persona CI-clear expected,
other CI-clear OPPOSITE) satisfied none of the four definitions (PARTIAL
required the other persona "same-sign or straddling"; REVERSED required
NEITHER CI-clear expected). Enumerate the 3^k per-unit CI-state combos
(expected/straddle/opposite per persona) and check every combo maps to a
label; demand a named DISCORDANT class. Disposition there: Concern not REVISE
(zero-GPU re-analysis, per-cell CIs all persisted, analyzer recovers
descriptively; fix is one line at implementation).
