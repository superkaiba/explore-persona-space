# Single-layer store slice for per-fold low-level views (#1310)

Committed fit-cell JSONs persist pooled R² + null draws but NOT per-fold or
per-group predictions. To build the SPEC-required low-level per-unit plot
without the original pod:

1. Download the activation store from HF with a PREFIX-SCOPED call
   (`HfApi().list_repo_tree(..., path_in_repo=<prefix>)` to verify, then
   `snapshot_download(allow_patterns=[...])`). A bare `list_repo_files` on
   `superkaiba1/explore-persona-space-data` hung >60s during/after the
   2026-07-16 HF outage; the prefix-scoped tree call returned in seconds.
2. Stream shards with `torch.load`, slicing ONLY the headline layer
   (`arrays["x_spanmean"][:, L, :]`) — ~100 MB RAM instead of 8 GB.
3. Re-fit at that single layer with the EXACT committed path: fit825
   `_cv_folds` (same seed), `_prep_fold` + `_ridge_predict_cached`, and the
   same `GCV_DOF_CAP`. Correct + swapped pairings share fold caches (same X).
4. VALIDATE: recomputed pooled R² vs committed `r2_per_layer_obs[L]` —
   #1310 matched to ~1e-15 (assert tol 1e-2 per the #833 CPU/GPU note).
5. Per-group R² with SS_tot around the FOLD-test mean decomposes the pooled
   statistic; display-clip (±1) and disclose the clip in the caption.

Cost: ~5-10 min CPU for 8 persona cells + 4 pooled swap fits (n up to 7k).
Set MALLOC_ARENA_MAX=2. Reap the 8 GB cache with
`clean_experiment_downloads.py <N> --incremental --apply` after the JSON is
committed.
