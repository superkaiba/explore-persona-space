# Independent pipeline review, 2026-09-12

The independent native_lens_review agent reviewed the pipeline and numerical
helpers without reading experimental outcomes. After fixes it reported no
remaining blocker for a fresh BF16 pilot. Latest reviewed pipeline SHA256:
`a5bd7eb55362cd4f2a930b767dc1e30143d70b8f04a0cb024ac2eebcdd221a35`.

Resolved findings: compare producer run/config/role/runtime at every boundary;
require completed frozen-subset coverage and immutable file checksums; bind
native ancestor reuse to byte-identical scientific implementation; validate
native dictionary precision/attention geometry; retain and regenerate capped
draws, refusing unresolved cap recovery at capture; audit finite/nonzero vocabulary
eligibility; use training-fixed near-zero component diagnostics; preserve a
completed fit's input manifest on a refused rerun; block every main phase until
calibration stability/readout validation exists.

The review found the mathematical use of training-only scaling, validation-only
MLP selection across seeds, token-before-pooling, equal rollout means, dictionary
orientation and paired bootstrap denominators consistent with the protocol.

The review does not establish native pilot completion, full calibration stability,
main execution, sampling-noise adequacy, model conclusions or cross-model validity.
The local 47-test suite and actual native validation evidence are separate records.
