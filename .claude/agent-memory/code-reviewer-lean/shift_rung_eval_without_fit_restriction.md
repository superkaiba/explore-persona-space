---
name: shift-rung-eval-without-fit-restriction
description: A registered transfer/shift rung (levels 1-3→4-5, dataset A→B, category holdout) needs a rung-specific FIT-side restriction, not just an eval-row subset — a dead fit-filter helper with zero call sites is the tell (#2388 R1 g6)
metadata:
  type: feedback
---

When a plan registers a shift/transfer rung with arrow notation ("levels
1–3 → 4–5", "{HE+MBPP+LeetCode} → {BCB+LCB}", "4-held-out-category
holdout"), the LEFT side is a FIT-side restriction: the probe must be
trained excluding the shifted rows' stratum, then evaluated on the shifted
eval rows. A sweep that scores the rung's eval rows with the SAME probe
fit on the unrestricted train partition produces in-distribution reads
silently mislabeled as transfer — no crash, no leakage (eval rows are
still test rows), just the wrong construct.

**Why:** #2388 R1 g6 (`issue2388_fits.py`): `_rung1_fit_filter` implemented
the math/code fit-side restrictions but had ZERO call sites — every
persisted `per_eval.rung1` came from the full-train-fit probe; MCQ's
category holdout lacked the fit-side category exclusion even in the dead
helper. Plan §4 explicitly sized "the rung-1 fit" against d, proving a
rung-specific fit was registered. Critical (wrong-science-silently).

**How to apply:** for every rung/shift/holdout eval set in a fits driver,
(1) trace where the FIT rows for that rung's scores come from — a single
fit serving rung0 AND rung1 is the red flag unless the plan registers
same-fit shifted-eval (cross-dataset rungs like TriviaQA→NQ-Open are
correct by construction when the train partition IS the source dataset);
(2) grep every helper the diff ADDS for call sites — a dead filter/
restriction helper is the implementer's own acknowledgment of the
requirement; (3) check holdout rungs also EXCLUDE the held-out stratum
from the fit side, not just select it on the eval side.
