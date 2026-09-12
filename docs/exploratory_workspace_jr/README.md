# Exploratory J/R context-to-answer experiment

The experiment is in execution. Native Qwen3.5-27B validation and the synthetic
affine control have run; real-model component fits and the weaker-model comparison
have not completed. No conclusion about workspace predictability is warranted yet.

The frozen [analysis plan](analysis_plan.md), [configuration](../../configs/analysis/workspace_jr.yaml),
[model/mapping provenance](mapping_provenance.md), [source audit](source_audit.md),
and [selected contexts](selected_contexts.json) remain authoritative. The user's
explicit task-registration exception is recorded in the configuration and the
[pilot continuation](pilot_execution_20260912.md). Manuscript files are untouched.

## What has actually run

- Native 27B forward validation on an A100 80GB with pinned Transformers 5.16.1:
  installing R backward rules leaves both the target hook and forward output
  bit-identical. Ordinary J passes three cotangent/direction numerical checks.
  The numerical check uses an FP32 suffix with the native prefix and positions;
  it does not claim finite-difference equality for BF16 lens coefficients or R.
  [Full validation evidence](native_validation_primary.json).
- The exact-affine synthetic control completed 12 cells: k=5/10/25 with the
  original dictionaries and three geometry-preserving rotations. Full-target
  ridge R² is effectively one. At k=10 the remainder-minus-component gaps are
  −0.887 for synthetic dictionary A and −0.974 for B; their difference is
  −0.087 (paired 95% interval −0.124 to −0.049). These are independent synthetic
  dictionaries, not native J/R lenses. This directly demonstrates a decomposition
  artifact, with the remainder less predictable than the component in this null.
  [Machine-readable report](affine_null_report.json).
- The null directly decomposes one affine vector per context: repeated identical
  token/draw pooling is an algebraic reduction, not executed K=5 sampling.
  It fits ridge and identity-plus-bias only. Token-before-pooling and unequal
  rollout lengths are independently tested on tensors; the null does not certify
  native generation, sampling-noise control, calibration, or MLP performance.
- The first spot attempt completed the ordinary-J half of a one-prompt native
  lens pilot in 154.76 seconds before GCP preemption. A second spot boot was also
  preempted. The same persistent disk was preserved and restarted on-demand;
  the paired lens recovery job is running under a six-hour STOP fence.
- The focused implementation suite passes 47 tests, including actual batched MLP
  tuning, context-level bootstraps, token-boundary capture on native tiny Qwen2,
  nonlinear token-before-rollout aggregation, cap recovery, and producer gates.
  Tiny-model/unit checks do not establish pretrained-model scientific validity.

[Browser-accessible null plot](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/6404c0b545dfe03505114dbbadb301f64a32f99d/exploratory_workspace_jr/20260912/figures/affine_null.png).
[All null inputs, predictions, bootstraps and exact executed source snapshot](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/730ad94bddab2cfeb52f548f40a70f84a4b9f858/exploratory_workspace_jr/20260912/affine-null-v1).
The upload was verified against all 107 expected paths at that revision. V1's
stored repeated-draw wording should be read with the algebraic qualification above.

## Execution boundaries

The saved choice is Qwen3.5-27B versus Qwen3.5-4B, thinking disabled, selected on
same-mode task performance rather than mapping R². Existing mapping captures
used K=1 and retokenized stripped answers; new capture preserves exact sampled
IDs with K=5 and equal rollout weights. Frozen predictor reuse requires the
separate historical-recapture parity gate. No frozen predictor has been applied
to incompatible new captures.

The new pipeline implements batched vLLM generation, doubled-cap recovery with
prior-draw retention, native token capture, full eligible-vocabulary nonnegative
pursuit, checkpointed component aggregation, validation-selected ridge and MLP
fits, original-unit metrics, paired bootstrap contrasts and explicit exclusions.
Every consumer checks producer identity, file hashes and completed coverage.
A pilot dictionary is explicitly barred from main analysis. Full calibration
membership alone does not pass the outstanding stability/readout gate.

Still outstanding: a completed native lens pair and end-to-end pilot; historical
recapture parity; full matched calibration with stability/readout review; direction
controls; main K=5 generation/capture/decomposition for both models; real-dictionary
rotations and sparsity sensitivities; sampling-noise/higher-K checks; learning curves;
paired cross-model analysis and the final model-results report.

## Reproduction

The native runtime is separately locked in `runtime/workspace_jr/uv.lock` because
this checkout's standard Transformers environment predates native Qwen3.5 support.
Use `scripts/workspace_jr_native_pilot.sh` on the managed GPU with the pinned code
SHA. This launcher stops after the native pilot; it does not imply main completion.

For focused local validation with the existing environment:

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run --no-sync pytest -q tests/test_workspace_lenses.py tests/test_workspace_components.py tests/test_workspace_runtime.py tests/test_workspace_capture.py tests/test_workspace_fit.py tests/test_workspace_artifacts.py
```

All tracked work lives in the dedicated `codex/jr-workspace-predictability-20260912`
worktree. Archived local preflights describe the earlier CPU/old-Transformers
limitation; the native validation report supersedes that limitation for the 27B
remote runtime. No task state, shared-root changes, or manuscript edits are included.
