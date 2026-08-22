---
name: twin-transcription-parity-tautology
description: A registered parity/round-trip gate that compares two textually identical local transcriptions on the SAME loaded object is a tautology — trace which OBJECTS each side consumes; a disk-roundtrip claim needs pre-persist vs post-reload comparison
metadata:
  type: feedback
---

When a plan/reconciler REGISTERS a prediction-parity or disk-round-trip assert,
verify the two sides of the compare differ in the thing the gate claims to
certify. Two checks: (a) are the two functions structurally independent, or is
the "reference" a re-typed copy of the production expression in the same file
(catches only future single-sided edits)? (b) do both sides consume the SAME
object instance? A "disk round-trip parity" that loads comp from disk and feeds
that ONE comp to both prediction paths never compares pre-persist vs
post-reload — a persist/load key-mapping swap (xmu↔ymu) passes it while
poisoning every downstream score.

**Why:** #2379 R1 g3 — `issue2379_mapfit.py::_predict_reference` was a
byte-equivalent copy of `predict_affine`, and both "disk round-trip" legs
(pilot + fits) called `_assert_prediction_parity(loaded_comp, x_ev)` — same
comp to both sides. The gate was a reconciler Must-Fix registration (P5.3)
against the raw-`W@v_C` defect; as shipped it could not fail on the
persistence-layer class its PASS log line claimed to certify. Filed Major.

**How to apply:** on any registered verify/parity gate, ask "what concrete bug
makes this assert FIRE?" and demand a fails-pre-fix falsification (corrupt one
persisted key in a copy, assert the gate FAILs). Round-trip gates must span the
storage boundary: in-memory prediction (or its hash) vs reloaded-component
prediction. Related: [[banked-parent-dual-schema-equivalence]], the
code-style.md hollow-verification-gate rule (this is its same-function,
wrong-oracle sibling).
