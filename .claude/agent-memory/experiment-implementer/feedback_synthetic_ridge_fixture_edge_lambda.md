---
name: synthetic-ridge-fixture-edge-lambda
description: Synthetic fixtures for val-selected ridge paths need a SHARED linear map + noise floor — noiseless or unrelated targets pin λ to a grid edge and kill edge-extension dispositions
metadata:
  type: feedback
---

A test fixture feeding a val-selected ridge path with an edge-extension
disposition (#2330 `fit_ridge_edge_extended`: λ at a grid edge → extend one
decade + refit, fail-loud after 4) fails in BOTH naive shapes: a
near-noiseless target (`Y = X + 0.01`) drives λ to the LOW edge until the
disposition exhausts (RuntimeError), and an UNRELATED random target drives
λ to the HIGH edge (shrink-to-mean is the val optimum, approached
monotonically — the same regime the shuffled-pairing null hits by design).

**Rule:** synthetic fixtures for any λ-selected fit use ONE shared linear
map across train/val/test plus a real noise floor — `W ~ N(0,1)/√H` fixed
once, `Y = X @ W + 0.5·N(0,1)` — which keeps the selected λ INTERIOR
robustly (probed on #2330: λ ≈ 3.2–10 across seeds at n_tr 16–64, H=8).
Probe the fixture through the REAL fit function for 2–3 seeds before
committing the test. The map must be the SAME across splits (per layer,
when the fixture spans layers) — resampling W per split re-creates the
unrelated-target high-edge regime.

**Why:** the edge-exhaustion branch is designed fail-loud production
behavior; a fixture that trips it turns a threading/schema test into a
statistics flake (two consecutive failures in #2330 round 5 before the
shared-map fixture landed).

**How to apply:** any `tests/` fixture driving `fit_ridge` /
`fit_ridge_edge_extended` / λ-grid selection with synthetic arrays; also
dense-sweep-style multi-layer fixtures (one map PER layer, shared across
splits). Related: [[tiny-dim-fixture-masks-shape-bugs]].
