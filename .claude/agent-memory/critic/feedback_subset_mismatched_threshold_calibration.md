---
name: Subset-mismatched threshold calibration
description: Classification bands grounded on parent paired deltas are invalid when the parent's baseline cell used a different question slice / n than the re-read denominator the new run computes against; recompute calibration same-subset from per-prompt files
type: feedback
---

When a follow-up plan registers classification thresholds (dip/no-dip bands, falsification bounds) "grounded" on a parent's paired deltas, CHECK that the parent's baseline cell and the new run's baseline cell use the SAME question subset and n. If the parent baseline averaged over a superset (e.g. trigger on questions [0:200]) while the new design's within-run baseline restricts to a slice (e.g. trigger50 on [0:50]), the inherited calibration is biased by the subset composition effect — and per-prompt files in git usually let you recompute the correct same-subset calibration in ~20 lines.

**Why:** #558 (2026-06-10): plan registered dip ≤ −2.0 nats ΔEOS-margin "below the parent doctor's weakest-adapter dip (−2.30)" — but those parent deltas were doctor[0:50] − trigger[0:200]. The trigger [0:50] subset mean sat 0.855 nats below the [0:200] mean (range −2.07..+0.18 across 12 adapters, systematic — same 50 questions shared). Recomputed same-subset (the contrast the new run actually computes): doctor mean −2.598, weakest −1.21, with 4/12 parent doctor cells INSIDE the registered gray zone (−2,−1]. A persona dipping exactly like doctor would have non-trivially misclassified "graded" → account-assignment headline flips. The plan itself acknowledged question-subset effects ~1.3 nats in its controls table yet calibrated across the mismatch.

**How to apply:** For any plan whose §7 bands cite "parent paired deltas" as Source, find what slice/n each parent cell used (build_cells in the pinned rig), and recompute the parent calibration restricted to the new run's common slice from the committed per-prompt slot-stats files before accepting the band edges. Sibling pattern: feedback_cross_encoding_base_prior_offset (base-prior offset ignored in a difference-of-gains verdict threshold).
