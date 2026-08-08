---
name: group_stratified_subsample seed-degeneracy on singleton stores + chat-store loading peaks
description: issue931_common.group_stratified_subsample ignores the seed on all-singleton group stores (largest-remainder stable tie-break picks the first n groups); issue825 _load_bundle_pt full-stack load of the Track-S chat store peaks ~31 GiB (earlyoom); fp64 eigh sweeps thrash at 8 threads on the loaded VM.
type: feedback
---

Three traps hit in one #931 follow-up smoke (2026-07-04), all on the #825/#931 chat-store estimator family:

1. **`issue931_common.group_stratified_subsample` is SEED-DEGENERATE on an all-singleton group store** (e.g. the Track-S chat store: 1 row per conversation). All proportional quotas floor to 0 and the largest-remainder tie-break is a STABLE argsort over EQUAL remainders → picks the lexicographically-first n groups for EVERY seed. A multi-seed draw design silently collapses to N copies of one draw (SD = 0 mechanically). **Why:** smoke run 1d showed byte-identical 28-layer curves for seeds 931/932. **How to apply:** before reusing this subsampler for a multi-draw design, check `counts.max()` on the store's group ids; on all-singleton stores draw seeded uniform rows instead (`np.sort(rng.choice(n_rows, n, replace=False))` — the correct group-stratified reduction; see `scripts/issue931_power_curve_multi_seed.py::draw_subsample`). Do NOT change the committed function's tie-break — committed multi-row-group results pin it.

2. **`issue825_fit_cells._load_bundle_any`/`_load_bundle_pt` on the full 10-shard Track-S chat store peaks ~31 GiB RSS** (per-record fp32 row lists + stacks + astype copies for every key) — earlyoom SIGTERMed it on the shared VM (journal names the PID; verify kill source before diagnosing). **How to apply:** stream per-shard cell slices instead (slice slot/turn per record BEFORE stacking + apply `_cell_xy`'s all-layer NaN keep-mask per shard) — `load_track_s_xy`/`_s1_slices_from_shard` in the same driver, peak ~10 GiB; equivalence-gate against the reference chain on ONE shard (`--verify-loader`).

3. **fp64 Gram-ridge eigh sweeps thrash at 8 BLAS threads on a heavily loaded VM** (load ~85-100): a ~200 s cell did not finish in 48 min at 8 threads (py-spy: stuck in `torch.linalg.eigh`); OMP/MKL/OPENBLAS/NUMEXPR=2 caps ran it in ~195 s. Same family as the #928 tiny-cell parity-gate entry. **How to apply:** for this estimator on the shared VM under load, launch with 2-thread caps (or route to a dedicated-core lane); measure one cell before projecting the battery.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [group_stratified_subsample seed-degenerate on singleton stores + chat-store load peaks](feedback_group_stratified_subsample_singleton_degeneracy.md) — all-singleton groups: tie-break picks first n rows for EVERY seed (SD=0 multi-draw collapse); _load_bundle_pt full Track-S store peaks ~31 GiB (earlyoom) — stream per-shard slices; fp64 eigh thrash at 8 threads on loaded VM → cap 2 (#931 pcms).
