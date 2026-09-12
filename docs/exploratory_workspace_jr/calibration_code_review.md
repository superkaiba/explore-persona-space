# Independent calibration diagnostic review

Codex reviewer `native_lens_review` inspected the new calibration module, CLI
and tests independently, without experimental test-result access. No Claude
automation was invoked. The reviewer verified equal prompt weighting, native
transport orientation, final normalization and the matrix/direction metrics.

Two findings were corrected and independently rechecked: validate native
precision, attention implementation and exact model class even for matrix-only
runs; and compare calibration token order against the independent frozen
selection after explicit exclusions. Internal manifest consistency and set
membership alone do not enforce nested-subset order. Four focused tests pass,
including native tiny-model identity-transport readouts and an adversarially
reordered calibration manifest. The reviewer reported no remaining issue in
these fixes. Scientific calibration adequacy still requires actual full-corpus
diagnostics and interpretation; a successful script run does not assert it.
