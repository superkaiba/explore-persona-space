# Numerical convergence recovery for the high-rate forecast analysis

## Goal

Test whether the model activation before generating new reasoning or an action predicts a later successful reward hack, and whether the frozen context-to-answer map improves that forecast.

## Same-goal scope

This is an analysis-only numerical correction to the approved high-rate experiment. Preserve the original plans, original analysis_spec.json, every physical rollout, capture and raw archive, the complete failed analysis and its exit/log. No additional model generation or GPU is required. The user has authorized finishing the corrected experiment; this does not change its research question.

## Observed failure and validated correction

The original analysis_20260908T112039Z exited1 after746.879s, with primary complete and competence_sensitivity partially fitted. The orientation_38300 fold3 C100 fit raised ConvergenceWarning at5000 iterations. The independent exact-cell diagnostic reproduced this failure and converged at5669 iterations under both20000 and50000 ceilings with bitwise-identical parameters/logits. Two already-converged cells also match exactly at5000 and20000. Diagnostic SHA256 b4a9352179a22b53c2c344af2a40011e1c7a02f12c40929084504707c6777073 supplies the smoke grounding for the new finite ceiling20000. All original41 sources were unchanged through diagnosis.

## Implementation boundary

Keep frozen followup_probe_core.py and all capture32/reconciliation40 sources byte-identical. Add a separate analysis optimizer implementing the same validated frequency-weighted binomial objective, unpenalized intercept, L-BFGS solver, tolerance1e-8 and strict ConvergenceWarning error behavior, with declared max_iter20000. Thread this explicit ceiling through fit_method/cached_fit; bind it into cache identity/diagnostics, retain the documented original5000 cache path for legacy use, and reject mismatched ceilings or excess/noninteger iterations. Add the helper, new spec, this plan and focused tests to the new analysis source review. Preserve original analysis_spec.json and accept only exact authorized specification hashes. No fallback solver, warning suppression, dropped C/control or changed tolerance is authorized.

## Scientific invariants

Every field of the original scientific specification remains equal except logistic_max_iter and explicit amendment/provenance metadata. Preserve all30 selected tasks, fixed20/10 split, all978 trajectories and90captures, U/NA handling, four analysis regimes, feature transformations, linear readouts, C grid, grouped folds, uncertainty draws/seeds/signs and original supported-benefit gates. All scientific interpretation remains completion-conditional because one freshU and eight structuralNA are present. The amendment responds to a convergence failure, not a search over final-test performance.

## Validation and execution

Before relaunch require focused cached-cap/strict-convergence tests, exact old/new fit equivalence on converged cells, actual failing-cell convergence at the authorized cap through production code, current independent source-bound analysis PASS and normal lint/commit/push checks. Preserve the full original652-file86.432MB partial output outside the fixed analysis resume directory with exact names/size/SHA validation. The new run starts a fresh empty analysis directory and fits all four regimes; no old fit is silently adopted under a changed source/spec regime. Its input semantic validators still run before and after fitting. Compare shared converged original cells to the new run and disclose any discrepancy before interpretation.

Use the same VM and8-thread/allocator pins, nohup/session/choom protection, exact owned launch/exit records and10800s operational watchdog. Reuse the already durable raw data by explicit source/hash mapping; final archive includes old failed outputs and new final results plus new code/spec/reviews. The expensive part is existing native validation; original652 files were written within the746.879s total including validation and the final failure. The same small training-span linear fits remain CPU work; no new GPU or data staging is planned.

## Completion

All four regimes must finish, all convergence and final source/input checks must pass, and the independent result audit must recompute predictions/metrics/CIs and controls. Persist full results/report/figures and read back the final archive, fold the scientific result into task2670 preserving Goal and leave classification to the user. The user-facing answer addresses reward-hacking forecast performance and mapping benefit, not workspace layout.
