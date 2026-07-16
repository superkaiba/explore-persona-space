# Union base panel needs per-group subsetting in the #1112 geometry rig (#1315)

The #1112 rig (`experiments/issue_1112/geometry.py`) asserts EXACT row-key equality
between each trained store and the base store (`delta_cloud`; `_reorder_store` only
re-orders, set-equality asserted). A design whose base pass captures a UNION panel
(#1315: 8 contexts × 20 q = 160 rows) while cells capture per-cell panels (5 shared
negatives + own source context = 120 rows) crashes at the first cell.

Fix pattern (scripts/issue1315_geometry.py `_run_tree_grouped`): group (cell, dose)
passes by SOURCE context; assert identical row order within a group; write an
order-preserving base SUBSET store per group; call run_geometry per group (pass only
the DIFF_PAIRS whose cells are both in-group — registered pairs normally share one
panel); merge payload dicts (records / cross_cell_diffs / sensitivity / ceilings are
key-disjoint). The shared per-group bootstrap index matrix keeps every paired contrast
paired. Add group-level resume (reload `_group_<ctx>/geometry_per_cell.json`, assert
n_boot) so OOM/preemption relaunches are monotone.
