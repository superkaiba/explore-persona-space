# Issue #2162 — implementation review round 3 findings (fix round → round 4 of cap 5)

Panel: `efficiency-critic` **PASS** · `code-correctness-critic` **APPROVE** · `plan-adherence-critic` **REVISE** (1 Major, 1 minor). Codex twin: confirmed no-show (3rd consecutive round).

**Your round-3 work verified well.** Efficiency traced the actual Batch-tail path and confirmed the pod genuinely reaches terminate-eligibility (pools missing → echo → `return 0` → `run_upload` → sentinel written unconditionally → RC_OK, which is the key the VM poller reads for Step-8 teardown), found ZERO poll loops in the dispatcher, and reconciled the deferred leg's 3.7 GPU-h exactly against plan §9 (2.8 grid + 0.9 anchor). It also confirmed `_margin_state`'s anchor predicate correctly reads DONE at the deferred re-run's W=1 despite the original being 8-wide — the width-change case I was worried about. Correctness verified the false-`false` direction of `margin_deferred` is closed (the dangerous direction), that mixed-width sets resolve conservatively, that the `upload_dir_hf` test exercises the REAL body with an 8-param-signature-matched fake, and that your H1 AST pin is genuinely stronger than the grep it replaced. Your self-found `fig_margin_validation` fix was explicitly ENDORSED as correctly in scope, not scope creep.

**This round is 4 items. One is a real gap on the report's most important figure; three are polish.**

---

## MAJOR 1 — `read_write_2x2` is missing its defining read (the 4th off-transform figure)

`scripts/issue2162_figures.py:225-257` (`fig_two_by_two`).

The manifest registers `read_write_2x2.plotted_quantity` as: "one point per (type × slot) in the AUC-vs-F plane, **quadrants labeled stored-and-used / stored-but-unusable / used-but-not-decoded / absent**, with sub-floor cells labeled untestable-causal."

**The analysis side is complete** — `issue2162_analysis.py:1051-1080` (`step_two_by_two`) computes and PERSISTS both `causal_verdict` (Holm + disjoint-both-nulls, n<12 → `untestable-causal`) and `probe_verdict` (max-selected permutation band) per cell. **The render throws half of it away:** points are styled by `causal_verdict` ONLY, `probe_verdict` is never read in the figure path (`grep -n "probe_verdict\|stored-and-used\|quadrant" scripts/issue2162_figures.py` → zero hits), no quadrant labels are drawn, and the only vertical reference is `axvline(0.5)` — **chance**, NOT the registered probe-positive threshold (the per-cell max-selected 97.5th permutation band). So quadrant membership is unrecoverable from position.

**Why this one matters more than the previous three off-transform figures:** the read × write 2×2 IS the experiment's headline. It is what answers the Goal's actual question — whether each information type is stored-and-used, stored-but-unusable, used-but-not-decoded, or absent. As rendered it is a one-axis causal-verdict scatter carrying a 2×2's name. This is the same class as r2's R1/R2 and the `fig_margin_validation` gap you found yourself, and it is the same guaranteed report-verifier per-figure FAIL you fixed that one to avoid.

**Fix:** encode quadrant membership from the already-persisted `(causal_verdict, probe_verdict)` pair — color/marker per quadrant, or explicit per-point quadrant tags — and draw the four registered labels. Keep the `untestable-causal` fifth label as-is. Use the registered probe-positive threshold for the reference line rather than chance, or drop the line if per-cell thresholds make a single vertical meaningless (a per-point encoding is fine and probably better). No new computation is needed; everything required is already in the persisted rows. Add a test pinning the four registered label strings, mirroring `test_fig_margin_validation_percell_grain`.

## MINOR 1 (adherence) — `act_beh_agreement` lacks its registered dynamic-range screen

`scripts/issue2162_figures.py:709-751`. The transform registers "Spearman rho across cells **with dynamic range** reported in-panel" — the same restriction phrase the manifest uses for the rule-19 grain you just implemented. The figure computes per-arm ρ across all separation-kept cell aggregates with no dynamic-range screen and no in-panel range statement. Descriptive read, gates nothing. Align the screen with the one you built for `rule19_validation`, and state the range in-panel — or declare the deviation in the figure docstring.

## MINOR 2 (correctness) — the deferred leg's second sentinel carries zeroed grid stats

`scripts/issue2162_run.py:2404-2410` (`MARGIN_DEFERRED_RECIPE`). On the fresh 1×H100 deferred-leg pod the `blocks/` dir is empty, so the second `upload` writes a sentinel whose non-margin stats are zeroed (`grid_shards=0`, `cap_hits=0`). `margin_deferred` itself is truthful there, and the upload-verifier reconciles against HF listings rather than sentinel counts — but a downstream reader of the SECOND sentinel's grid fields could misread it as a run that produced nothing. **Fix:** stamp `"deferred_leg": true` (or omit the grid stats entirely) when `blocks/` is empty locally.

## MINOR 3 (correctness) — tighten the H1 AST pin by one line

`tests/test_issue2162_run.py`. `guarded_ids` walks the whole `if write_done:` node via `ast.walk(iff)`, so a done-write placed in that `if`'s `orelse` would count as GUARDED. Iterate `iff.body` instead. Strictly theoretical today (exactly 2 writes, both in the if-body) but it is a 1-line change to a pin whose whole job is catching future drift.

---

## Recorded, do NOT fix

`_margin_state` reads done-record existence and width but not `regime_fp` (`issue2162_run.py:2427-2443`). The reviewing critic judged this DEFENSIBLE and marked it non-mechanizable: reaching a complete-but-wrong-regime state requires out-of-band manipulation (the run path already fail-louds on regime mixtures — the claim queue raises, `_sharded_done_record` regenerates on width/regime mismatch), and a pools-absent upload leg genuinely cannot compute the margin regime fingerprint. Leave it; it is noted as a residual, not a defect.

## Constraints

- All four items are pre-pod and CPU-verifiable. **Do NOT provision a pod.**
- Do not touch what the panel cleared: the `margin_deferred` machinery, the opportunistic-margin chain, `upload_dir_hf`, the rule-19 grain work, H2/H3, the perm-matrix upload, or the three previously-fixed figure transforms.
- Do NOT edit `plans/plan.md` or `artifacts/planned_manifest.json` (orchestrator-owned; plan is at v4).
- Re-run `uv run pytest tests/test_issue2162_*.py` and report the real count (78 before this round).
- Post an updated `epm:experiment-implementation` marker (bump the version) with a per-item disposition.
- Return SHORT: per-item disposition, test count, anything you believe is wrong (with the argument), remaining deviations.
