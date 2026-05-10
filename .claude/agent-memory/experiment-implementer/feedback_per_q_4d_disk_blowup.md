---
name: 4-D per-q caches blow up disk by Nx the position-axis dim
description: When sweep emits 4-D per-q tensors (n_q, n_layers, n_pos, D) and the planner estimates as 3-D (n_q, n_layers, D), real disk is N_pos times the estimate. Default the per-q write to an analyzer-needed subset.
type: feedback
---

When `sweep_extraction_grid.py`-style scripts dump per-question hidden states
along (token, layer) grids, the per-q tensor is 4-D —
`(n_q, n_layers, n_position, D)` fp16 — and the planner's "(n_q, n_layers, D)"
back-of-envelope underestimates disk by exactly the position-axis dimension
(in #263: 9x for `r_per_token`, 5x for `method_a` and `method_caa`).

**Why:** Plan-time disk math always reaches for the simplest shape. But the
analysis side only consumes per-q at a small subset of cells (in #263, H3's
headline test only reads `t=0` and `t=128` from r_per_token). The other
positions are wasted bytes on disk. With 4-D shapes producing 100+ GB and
modest pod volumes (200 GB), this is a guaranteed ENOSPC.

**How to apply:**
1. When the plan calls for a 4-D per-q dump, **immediately** ask: which (pos,
   layer) cells does the analyzer actually read per-q? In #263, H3 reads only
   2 of 9 response positions; H2 reads all of them as candidates BUT degrades
   gracefully if a candidate cell can't load.
2. Implement a `--per-q-positions-subset` CLI flag with default = analyzer's
   load-bearing subset. Centroids at the full grid are cheap (fp32, no
   per-question dim) — keep those at full resolution for clustering /
   trajectory figures. The per-q tensor (fp16, full per-question dim) is the
   bytes that hurt; trim it.
3. Make analyzer-side rejection LOUD: a typed RuntimeError subclass
   (`PositionNotInPerQSubsetError`) that existing try/except paths catch as
   "cell skipped" + an explicit count in the result JSON
   (`n_candidate_cells_skipped_per_q_subset`).
4. Defensive shape-vs-list check: `n_pos_on_disk != len(subset)` raises
   immediately instead of mis-indexing — catches the case where someone
   re-runs sweep with a different subset over the same output_dir.
5. In the report, recompute the disk-budget table with the EMPIRICAL per-q
   shape (not the plan's), and tell the experimenter the new total + the
   verbatim re-launch command including the new flag.

**Why this matters beyond #263:** any `sweep_extraction_*.py` style script
that dumps per-token activations is at risk. Ground-truth: `ls -lh` the
emitted per_q files on the pod after the first persona, multiply by 275, see
if it fits. Doing this BEFORE letting the run continue past persona 1 saves
hours of GPU-time and prevents partial-write corruption.
