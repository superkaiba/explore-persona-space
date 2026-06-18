---
name: 4-D per-q caches blow up disk by Nx the position-axis dim
description: Sweeps emitting 4-D per-q tensors (n_q, n_layers, n_pos, D) are N_pos times the planner's 3-D estimate; default the per-q write to the analyzer-needed subset.
type: feedback
---

When a sweep dumps per-question hidden states along (token, layer) grids, the per-q tensor is 4-D `(n_q, n_layers, n_pos, D)` fp16 — the plan's "(n_q, n_layers, D)" back-of-envelope underestimates disk by exactly the position-axis dim (9× for `r_per_token` in #263). With 100+ GB outputs on 200 GB pod volumes that's a guaranteed ENOSPC, and the analysis usually reads only a couple of position cells per-q anyway.

**How to apply:**
1. Ask which (pos, layer) cells the analyzer actually reads per-q; implement `--per-q-positions-subset` defaulting to that load-bearing subset. Keep centroids (fp32, no per-q dim — cheap) at full grid for clustering/trajectory figures.
2. Analyzer-side rejection LOUD: a typed `PositionNotInPerQSubsetError` caught as "cell skipped" + an explicit skipped-count in the result JSON.
3. Defensive check: `n_pos_on_disk != len(subset)` raises immediately (catches re-running the sweep with a different subset over the same output_dir).
4. Ground-truth early: `ls -lh` the per-q files after the FIRST persona, multiply by the persona count, check it fits — before the run continues.
5. In the report, recompute the disk-budget table with the EMPIRICAL shape and give the experimenter the verbatim re-launch command with the new flag.
