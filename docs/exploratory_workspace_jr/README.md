# Exploratory J/R context-to-answer experiment

The experiment is running. Both models have completed native lens calibration,
validated end-to-end pilots and main generation/capture. Main component
training decompositions are in progress. Main prediction fits, the complete
control grid and the final scientific report remain outstanding; no main
predictability conclusion is available yet.

The [main execution handoffs](execution_handoffs_20260913.md) record the
verified stage sequences and recovery obligations for the six active workers.

The frozen [analysis plan](analysis_plan.md),
[configuration](../../configs/analysis/workspace_jr.yaml),
[model and mapping provenance](mapping_provenance.md),
[source audit](source_audit.md) and [selected contexts](selected_contexts.json)
define the experiment. The user's explicit task-registration exception remains
recorded in the configuration and [pilot continuation](pilot_execution_20260912.md).
This directory is separate from the manuscript.

## Executed and verified

Both Qwen3.5-27B and Qwen3.5-4B passed their native J-product, specified R-rule
and unchanged-forward checks. Matched J/R calibration uses each model's actual
checkpoint and mapping source hook. Calibration stability and readouts,
checkpointed token decomposition and the full tuning/evaluation path were
reviewed before main dispatch. The complete readiness evidence is pinned here:

- [Primary readiness and evidence](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/537bf1146730b6bf7ef9295cb2d051982f38d257/exploratory_workspace_jr/20260912/primary_readiness_bundle_v1).
- [Comparison readiness and evidence](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/216e638d605527827bd4a0580a9edc7ebeeeea02/exploratory_workspace_jr/20260912/comparison_readiness_bundle_v1).
- [Primary complete training capture](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2c3b39d4cf425b949533d0974fd2b90db8ff01c5/exploratory_workspace_jr/20260912/primary_main_train_v1).
- [Comparison complete training capture](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c3f6993e2b5b4525eda6a9b1844f5aef7aceff83/exploratory_workspace_jr/20260912/comparison_main_train_v1).

Historical mappings used K=1, not K=5, and did not pass the unchanged historical
recapture gate. The pre-outcome [execution revision](execution_revision_20260912.json)
therefore uses fresh context-only inputs in frozen batches of 16 and fresh fits
for full answers and all components. No historical predictor is applied to
incompatible new captures. The comparison pilot's terminal-token mistake was
preserved and corrected through fresh answer-state recapture; see the
[EOS correction record](terminal_eos_correction_20260913.md).

Calibration-only reconstruction controls have completed for both models. Every
native/control dictionary uses the same token decomposition and sparsity grid;
quality matching preserves outside-range exclusions. These are calibration
measurements, separate from main prediction outcomes:

- [Primary reconstruction-quality matches](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/1d46f094cab95836d13bc05c3cf0c0ff99682412/exploratory_workspace_jr/20260912/primary_calibration_quality_v1/quality_matches.json).
- [Comparison reconstruction-quality matches](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/2a6891793e45fe7e61fcfd44bfd1bc554253ac35/exploratory_workspace_jr/20260912/comparison_calibration_quality_v1/quality_matches.json).

## Primary scoring population

The outcome-blind completion ledger retains 252 of the 256 frozen shared test
contexts: each retained context has five nonempty final completed rollouts in
both models. The same ordered cohort must enter every primary within-model and
cross-model comparison. The regeneration threshold and primary scoring
eligibility are distinct; all original draws and exclusions remain saved.
Training and validation retain captured nonempty draws with censoring reported.

The [pre-outcome scoring declaration](completion_scoring_20260913.md) and
[uploaded actual cohort ledger](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/e41de075378e426d70c99c4d8b9ac9748fc1adb0/exploratory_workspace_jr/20260912/completion_cohort_v1/completion_cohort.json)
record the rule and its application. This conditions inference on completion;
excluded contexts are not replaced. A local audit attempt correctly refused a
runtime-version mismatch; the unchanged audit succeeded in the matching worker
runtime.

## Remaining execution and analysis

Complete all observed model × dictionary orientation × sparsity fits, the
actual-dictionary affine nulls, main sampling-noise and direction diagnostics,
and fixed-recipe learning curves. Run the registered higher-K follow-up on its
[fixed subset](higher_k_selection.md) if its noise criterion triggers. Then execute
the [paired final comparison](comparison_analysis.md),
[supplementary cohort analysis](supplementary_analysis.md), saved figure exports
and the concise final report. Intermediate all-captured fit scores are not the
primary completion-conditioned estimates.

The earlier [small synthetic-dictionary null](affine_null_protocol.md) is a
software and decomposition-artifact check. It does not replace the required
actual-dictionary null or establish real-model selectivity.

## Reproduction and integrity

The native runtime is locked separately in `runtime/workspace_jr/uv.lock`.
Native primary and comparison main producers remain pinned to
`f8b4983851fba9e176640f75ad0ad7f667be46ab` and
`244f89eb8347361484413e803565f1e639255a8c`, respectively. Downstream analysis must
use their matching library versions and verify unchanged native implementation
ancestry. Individual phase commands, exact source, inputs, completion markers
and immutable upload receipts are archived with the artifacts.

`workspace_jr_delta_persist.py` transfers changed files in bounded commits and
then verifies every local and remote byte hash at one pinned revision. It keeps
the consumer's individual-file layout, rejects changed file sets and stale
receipts, and never treats a missing remote path as verified. Running supervisors
execute an archived uploader copy so later operational changes cannot alter them.

All tracked edits live in the dedicated
`codex/jr-workspace-predictability-20260912` worktree. Main result interpretation
must distinguish relative predictability from automaticity, reasoning or causal
capability effects; agreement between the lenses is a robustness check.
