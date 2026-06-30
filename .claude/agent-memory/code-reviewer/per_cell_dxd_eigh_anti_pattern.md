---
name: Per-cell d×d eigh/inverse is still O(d³)-per-cell even after caching the grid
description: Whitening / covariance code that computes a fresh d×d eigh or inverse PER CELL at d=3584 is minutes-per-cell; flag it on sight in analysis sweeps
type: project
---

A `Σc` / whitening estimator that builds a fresh `np.linalg.eigh` or `np.linalg.inv`
on a (N, d) matrix at d=3584 costs **>5 min PER CALL** — even after the
`vectorize-many-cell-fits.md` eigh-cache patch (which only removes the ~17× ridge-grid
multiplier WITHIN one call; the single O(d³) eigh remains).

**Why:** Eliminating the per-λ loop multiplier ≠ eliminating the per-cell O(d³) cost.
A sweep that calls the estimator once per cell (48 cells) pays 48 × one d=3584 eigh.

**How to apply:** When reviewing analysis code that whitens / inverts a large
covariance, check whether the expensive d×d factorization is computed ONCE and reused
across all cells (correct — the "broad-corpus Σc" design pattern), or recomputed
per-cell (wrong — minutes × n_cells). A per-cell battery-Σc fallback at d=3584 is a
latent timeout that masquerades as a "silent exit-0 / empty output" smoke bug
(#666 round 1: `_battery_sigma_inv` per cell → `predictor --slice --cells 2` hung
>400s on cell 2 and was misreported as exit-0-no-JSON). The production headline path
MUST thread ONE precomputed `Sigma_inv` into every cell, never recompute per cell.
