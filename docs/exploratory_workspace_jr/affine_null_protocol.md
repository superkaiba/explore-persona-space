# Actual-dictionary affine null

This implementation specification was fixed before main answer outcomes were
generated. It fills in the affine-null control already required by the analysis
plan; model, layer, splits, dictionary, sparsity and fitting budgets are unchanged.

For each model, use the saved full-answer ridge from the unrotated k=10 observed
cell to define h_i = x_i A + b. Its coefficients are fitted on training contexts
and its regularization selected on validation contexts; test targets never fit
or select this map. Save and verify its exact uploaded coefficient bytes. This
choice keeps the synthetic activation scale and orientation related to the
observed fitted map without selecting a favorable synthetic map from test gaps.

Every token in every rollout of context i receives this same h_i. Preserve all
original positive rollout lengths, seeds, exclusions and context order. Because
the vectors are identical, their sparse components are identical; evaluating
s(h_i) once is exactly the algebraic reduction of token decomposition followed
by equal rollout pooling. This reduction never applies to actual model outputs.
There is no within-context sampling noise in this control.

Compute the affine h in FP64, then apply the observed analysis's FP32 sparse
arithmetic. Save the FP32 conversion error explicitly. Define the remainder as
the original FP64 h minus the estimated component, preserving the affine full
target and reconstruction identity. Report the known-map oracle SSE and the
fitted full-target R²; a poor fitted full-target R² limits interpretation of the
null component deficits.

Use the actual full-vocabulary J/R dictionaries and all three registered Haar
rotations, k=5/10/25, the same affine ridge and three-seed MLP tuning budgets,
target scaling and paired context bootstrap. Save per-context targets,
predictions, sparse statistics and the original length layout. Report these
null gaps alongside observed gaps and rotated-control gaps. They diagnose
nonlinearity introduced by the decomposition; they are not evidence about
language processing. Pilot-only dictionary runs remain labeled pilot controls.
