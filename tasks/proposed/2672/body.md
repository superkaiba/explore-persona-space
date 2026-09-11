---
title: Point the Section 4.3 figure generator at the K=5 grid
kind: infra
tags: []
created_at: '2026-09-11T09:49:18Z'
has_clean_result: false
origin_prompt: 'Paper audit 2026-09-11: the shipped Section 4.3 figure was the K=1
  run while the prose quoted K=5; paper fixed by copying the correct render, generator
  still drifting.'
workflow: v1
---
---
kind: infra
---

# Point the Section 4.3 figure generator at the K=5 grid

`scripts/section43_posttraining_figure.py:73` hardcodes

    SUMMARY_PATH = eval_results/issue_1902/lasttoken_transfer/summary.json

which is the **K=1** run. The paper's Section 4.3 prose quotes the **K=5** run
(`eval_results/issue_1902/k5_full_grid/summary.json`). The two disagree
visibly: diagonal R2 0.467/0.518/0.519/0.514 against 0.675/0.634/0.606/0.599,
and the first retention reads -0.164 against 0.02.

The paper was fixed on 2026-09-11 by copying the correct K=5 render into the
Overleaf tree (Overleaf commit 8e94c48). The generator was not changed, so the
next regeneration reintroduces the mismatch. The EPS copy at
`figures/paper/c1_posttraining_dynamics.pdf` is still the K=1 render.

## What to change

1. `SUMMARY_PATH` to `eval_results/issue_1902/k5_full_grid/summary.json`.
2. `LATEST_SHA256` to `50cda3fe892143bfc66c5bdf86910a069936455d1044f549af81fdf58b5a60cd`.
   Note the pinned value in the file today, `947a46cb...`, matches **neither**
   summary, so that integrity pin is already stale and worth understanding
   before changing it.
3. The two summaries differ in one key: `parity_gate` (K=1) against
   `parity_anchor` (K=5). Everything else (`grid`, `metadata`, `transfer`) is
   the same shape. Handle whichever the script reads.
4. Regenerate and confirm the output matches the archived K=5 render at
   `/mnt/eps-data/thomasjiralerspong/issue1902_k5grid_20260909/results_archive/c1_posttraining_dynamics_k5.pdf`.

## Care needed

`figures/issue_1902/section43/c1_posttraining_dynamics.{pdf,png,_data.json}`
were modified-but-uncommitted in the shared repo root when this was filed, so
another session may be working in that area. Probe before writing.

Related: `scripts/issue1902_k5_fits.py` and `scripts/issue1902_k5_layer18.py`
exist only on the branches `issue-1902-k5` and
`codex/1902-k5-full-grid-20260909`, never merged to main, so the generators for
the shipped Section 4.3 numbers are not reproducible from main either.
