---
name: sidecar-rerender-recover-original-first
description: Re-rendering a figure from its own savefig_paper sidecar destroys that sidecar — recover the ORIGINAL meta.json from git BEFORE rendering
metadata:
  type: feedback
---

Incident (#2224 clean-result round, 2026-08-12): re-rendering
`i2224_4a_scatter_hallucination` "from its sidecar" first produced a broken
render (my `_group` guess selected 3 of 24 points per panel), and
`savefig_paper` immediately OVERWROTE the original `.meta.json` in place —
the 168-point original was gone from the working tree and only recoverable
via `git show <committed-sha>:figures/issue_<N>/<stem>.meta.json`.

Rules:

1. **Before any sidecar-sourced re-render, copy the original meta.json out**
   (`git show` the committed blob to /tmp, or `cp` before the first render) —
   the first render attempt, right or wrong, replaces it.
2. **Sidecar `points` grouping semantics:** `_group` is the global SERIES
   index across the whole figure (axes-order × series-order), not the panel
   index. For a P-panel figure with S series per axes, panel = `_group // S`.
   Verify with a Counter over `_group` + `series` before plotting.
3. Per-point y-keys are inconsistent across series in older sidecars: some
   series carry the axis-label key (e.g. `post-ft trait score (graded)`),
   others a bare `y` — use `p.get(label_key, p.get("y"))`.
4. Assert the reconstructed per-panel count equals the expected n before
   saving (the broken render was visually obvious only because n=24 vs 3).
5. **Label-only regeneration (critic-mandated arm renames, #2225 r2):** back
   up every affected sidecar to /tmp FIRST, rename in the builder's label
   dicts (CONFIG_SHORT-style), re-run the parent builder, then prove
   "numbers unchanged, labels only" by comparing old-vs-new sidecar `points`
   with label-bearing keys stripped (`category`, `series`, plus
   commit/timestamp keys) AND asserting every label change matches the
   declared rename map. Sidecar label carriers are `category` (bars) and
   `series` (lines), not `label`.
