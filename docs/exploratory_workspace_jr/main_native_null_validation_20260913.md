# Actual native affine-null validation

Both native affine-null preparations passed independent checks against their
original full-answer ridge predictors, canonical context inputs and raw
generations. This verifies the constructed control, not the scientific
interpretation of component predictability gaps. The complete comparison still
requires the observed, rotated and null grid on the shared scoring cohort.

The audit checks all 1,024 training, 128 validation and 256 test contexts per
model. Saved inputs match the original context-only captures exactly. Saved
coefficients and intercepts match the corresponding native ridge fit, and an
independent FP64 matrix multiplication reproduces every affine full target
exactly. Original five-rollout seeds and all 7,040 rollout lengths per model
are preserved. Repeating the same affine vector over each original token
position makes equal-token then equal-rollout averaging algebraically exact;
this does not decompose an observed pooled answer activation.

At every k=5/10/25, the full target equals the component plus its remainder up
to FP64 addition roundoff: maximum absolute error is 1.78e-15 for 27B and
2.22e-16 for 4B. Saved sparse components are exact FP32-to-FP64 promotions and
remainders equal affine full targets minus those components. Original canonical
inputs, prepared arrays, target fingerprints, terminals and immutable source
uploads were checked. Fifty-four separate extended-precision scalar checks
passed and rejected swapped context rows, omitted bias and transposed
coefficients. The audit did not replay dictionary-wide sparse pursuit on CPU;
its numerical implementation remains the previously reviewed GPU procedure.

The audit deliberately did not interpret saved predictor outputs or selectivity
gaps. Null fits retain all 256 test contexts; final primary scoring continues
to use the separately bound joint set of 252 completed contexts. No fitting,
tuning, dictionary matching or exclusion setting changed.

The [independent review](main_native_null_invariant_review.json) records exact
checks, hashes and limitations. The [complete audit evidence](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/46f68719759451564996b9c974e074470a96db9d/exploratory_workspace_jr/20260912/native_null_invariant_independent_review_report_v1)
contains the audit code, source verification, detailed invariants and scalar
oracle. Its upload receipt SHA256 is
`94a5a55b15fb196b5ecda278de31fdf00fb492d1ca10c4bd2f2d75e89b153e7d`.
