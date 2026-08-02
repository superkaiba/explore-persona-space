---
name: Cross-frame numeric asserts against manifest selection reads
description: A reuse-gate that equality-asserts a measured value against a manifest number recorded in a DIFFERENT measurement frame is unreproducible by construction — gates must be frame-matched or frame-free (#1900 r6, job 16092)
type: feedback
---

A runtime reuse-validation gate must never equality-compare a freshly measured
statistic against a manifest/verdict number captured in a DIFFERENT frame.
#1900's P1a gate asserted `|median training-row Δ logP − manifest
delta_logp_mean| ≤ 1 nat`; the manifest number was #1481's checkpoint-SELECTION
read in its eval-probe frame (window [5.0, 12.0]) while the gate TF'd the arm's
OWN training rows, where a memorized marker slot legitimately reads ~20+ nats
(job 16092: 22.436) — the gate could NEVER pass on a healthy adapter.

**Why:** a manifest value is a claim about the frame it was measured in
(selection probe rows, eval subset, judge frame); the same quantity on a
different row distribution differs by construction, not by fault. This is the
frame sibling of the #813 same-surface-commensurability rule
(artifact-reuse.md § gate calibration).

**How to apply:** before writing any gate that compares against a stored
reference, name BOTH frames. Frames match → equality/tolerance is fine
(e.g. #1900's GATE_PARITY recomputes the same statistic from the same stored
inputs, atol 1e-6). Frames differ → make the gate FRAME-FREE (direction+floor
between failure bands: broken ≈ 0 vs real ≈ 20+ nats ⇒ +2-nat floor) and
RECORD the manifest value side-by-side with an explicit frame note, never
assert it. Sweep the diff for sibling cross-frame comparisons (judge-frame
verdict numbers included) in the same round.
