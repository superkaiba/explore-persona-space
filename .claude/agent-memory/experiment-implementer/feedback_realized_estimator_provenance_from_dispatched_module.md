---
name: realized-estimator-provenance-from-dispatched-module
description: Runtime module-global patches never reach import-time snapshots or module literals — record persisted estimator provenance from the DISPATCHED module's state at the fit call + a SELECTOR_LOG telemetry delta, never re-stamp the driver's constants (#2546 r16)
metadata:
  type: feedback
---

A driver that patches ONE module's global (`fc.N_INNER_LAMBDA_FOLDS = 2`) and
then stamps its own constants into every persisted payload writes FALSE
provenance for any route that dispatches to a DIFFERENT module: an import-time
snapshot (`ma.N_INNER_LAMBDA_FOLDS = fit825.N_INNER_LAMBDA_FOLDS`, bound at
import) and a module literal (`xm.N_INNER_LAMBDA_FOLDS = 4`) are both
invisible to the patch, so those routes silently run the OLD regime while the
payload asserts the new one (#2546 r16: ma ladder/ood at 13-pt grid + 4 folds,
operator at 4-fold literal, all stamped as 23-pt/2-fold).

**Why:** module-global patching propagates only through CALL-TIME reads of the
patched module's namespace. `from m import X` / `Y = m.X` at import, and
literals that merely claim equality in a comment ("== fit825's, asserted in
tests"), freeze the value. Two review arms + a composer independently flagged
the false stamps; the fix window closed the moment the fits ran.

**How to apply:** when persisting estimator/config provenance for a unit,
read the knobs from the module the fit call DISPATCHES to, at the call
(`mod.LAMBDAS`, `mod.N_INNER_LAMBDA_FOLDS`, `mod.LAMBDA_SELECTION`), and
measure realized selector usage as a `mod.SELECTOR_LOG` before/after delta
(units run serially per process, so the delta is unit-scoped; fail loud when
telemetry is None or the delta is empty). Cross-module grids need a coherence
check: fit825's `_inner_cv_rss_curve(lams=None)` SCANS `fit825.LAMBDAS` while
callers INDEX their own `LAMBDAS` — if they diverge, the realized grid is
ill-defined: refuse, never record either. Record grid IDENTIFIERS (generating
params, verified by in-process recompute equality — safe within one process)
in resume keys; realized float values ride the payload only (#1336 float-key
ban). Worked impl: `scripts/issue2546_fit_cells.py` `_RidgeEstimatorRecorder`
/ `_realized_sweep_estimator` / `_attach_fit_params` +
`tests/test_issue2546_estimator_provenance.py`. Related: [[stale-pycache-masks-signature-change]],
[[reused-fit-core-registry-lookup-seam]].
