---
name: Subset-mismatched threshold calibration
description: Bands grounded on parent paired deltas are biased when the parent baseline used a different question slice/n than the new within-run denominator; recompute same-subset from committed per-prompt files (#558)
type: feedback
---

When a follow-up registers classification thresholds (dip/no-dip bands, falsification bounds) "grounded" on a parent's paired deltas, check that the parent's baseline cell and the new run's baseline use the SAME question subset and n. If the parent averaged over a superset (trigger [0:200]) while the new design's within-run baseline restricts to a slice (trigger50 [0:50]), the inherited calibration is biased by subset composition — and per-prompt files in git let you recompute the correct same-subset calibration in ~20 lines.

**Why (#558):** plan registered dip ≤ −2.0 nats "below the parent doctor's weakest-adapter dip (−2.30)" — but those deltas were doctor[0:50] − trigger[0:200]; the trigger[0:50] subset mean sat 0.855 nats below the [0:200] mean (systematic, same 50 questions shared). Recomputed same-subset: doctor mean −2.598, weakest −1.21, with 4/12 parent doctor cells INSIDE the registered gray zone — a persona dipping exactly like doctor would be misclassified and the account-assignment headline flips. The plan itself acknowledged ~1.3-nat question-subset effects in its controls table yet calibrated across the mismatch.

**How to apply:** for any §7 band citing "parent paired deltas" as Source, find what slice/n each parent cell used (build_cells in the pinned rig) and recompute the calibration restricted to the new run's common slice from committed per-prompt slot-stats files before accepting the band edges. Siblings: feedback_cross_encoding_base_prior_offset (base-prior offset), feedback_comparator_range_sibling_cell_provenance (wrong cell FAMILY).
