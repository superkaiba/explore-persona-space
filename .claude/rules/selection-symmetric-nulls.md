---
description: Selection-symmetric nulls — a null/permutation band compared against a max-over-axis-selected observed statistic must inherit the SAME selection per draw, or the axis must be frozen on a held-out split; persist the per-draw × per-axis matrix. Loads on-demand at plan time.
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Selection-symmetric nulls (max-over-axis inference)

**Load this rule whenever a plan's headline statistic is chosen by
`max` / `argmax` / best-of / top-k-mean over a FREE AXIS** — a read-out
layer, a cell, a k / neighbourhood size, a seed, an extraction point, a
threshold — AND that statistic is compared against a null / permutation /
shuffle band. The always-on backstop is `planner.md` §6 "Selection-symmetric
nulls" + `critic.md` Statistics & Measurement lens item 11; this file is the
full recipe those entries point at.

## The asymmetry

If you select the OBSERVED statistic as `max` over L axis positions but
compute the null band at ONE fixed position, the observed value got L
chances to be large and each null draw got 1 — a 28-vs-1 comparison. The
observed statistic is inflated by the winner's curse (≈ `sqrt(2 ln L)·SE`
above a per-position value), so the observed-vs-null gap is manufactured
by the selection, not by the effect the null battery is meant to test.
The same asymmetry applies to `argmax`, best-of, and a top-k mean (the
top-k averaging shares the winner's-curse inflation — the k winners are
still the extremes of L draws).

**Motivating evidence (#778).** A persona-vectors replication
(arXiv 2507.21509) selected the matched direction's read-out layer by
`max(matched r)` over 28 layers, then computed each null direction's band
at that single matched-winning layer. Simulated at n=24: single-layer
null p97.5 |r| ≈ 0.48 vs honest max-over-layer p97.5 |r| ≈ 0.62. All three
Phase-2 lenses reconciled to REVISE — four of the six per-lens critics
caught the asymmetry independently (Statistics Claude+Codex, Methodology
Codex, Alternatives Codex); Methodology Claude and Alternatives Claude
missed it. Siblings: #664 (best-of-28 SNR clears a floor a fixed-layer
read does not), #545 (best-of-group winner's-curse inflation ≈
`sqrt(2 ln K)·SE`).

## The two symmetric alternatives (pick one)

A plan whose headline uses `max` / `argmax` / best-of / top-k-mean over a
free axis MUST do ONE of:

1. **Per-draw same-selection (default).** Every null / permutation /
   shuffle draw receives the IDENTICAL max-over-axis selection before the
   band is formed. The null distribution is `{ max_over_axis(stat under
   draw d) : d }`, so observed and null get the same L chances. The
   honest band is the quantile of that max-selected null distribution.
2. **Held-out axis freeze.** Choose the axis position on a HELD-OUT split
   (or a pre-registered fixed position committed before seeing the data),
   then read BOTH the observed statistic and every null draw at that
   single frozen position. Observed and null are again symmetric — both
   at one fixed position, no selection on either side.

Do NOT compute the observed statistic max-over-axis and the null at one
position. A per-axis heatmap is a DIAGNOSTIC display and does NOT
neutralise the asymmetry — the comparison statistic is still
`max-over-axis observed` vs `one-position null`.

## Persist the per-draw × per-axis matrix

Persist the full per-draw × per-axis statistic matrix (one matrix per
headline statistic) — observed row + one row per null/permutation draw,
one column per axis position — as a downstream artifact (HF data repo
`analysis_tensors/` per the Upload Policy, or an `eval_results/` JSON if
small). This lets the analyzer recompute the honest max-selected band
post-hoc even if the plan shipped the asymmetric read — the honest band
is a pure re-reduction of the stored matrix, no re-run. Without the matrix
the honest band is unrecoverable and the run must be repeated.

## Carve-outs (this rule does NOT fire)

- **A single pre-registered / fixed axis position with no data-driven
  selection** — e.g. "read at layer 14, committed in advance" or a
  paper's own fixed layer. There is no `max`/`argmax` over the axis, so
  observed and null are already symmetric. State the fixed position and
  that it is pre-registered.
- **A legitimate single-anchor ablation** that picks ONE
  condition/layer for a mechanistic reason unrelated to maximising the
  headline statistic (e.g. "ablate the residual stream at the layer the
  paper localises the feature") — not a best-of search over the
  headline metric.
- **A per-axis result reported at EVERY position with no cross-position
  `max` headline** (a full layer sweep displayed as a curve/heatmap
  where the claim is the shape, not a single best position).
- **N/A when the headline statistic is not selected over any free
  axis** — the plan writes "N/A — no max-over-axis selection in the
  headline".

## Enforcement

- `planner.md` §6 "Selection-symmetric nulls" — the trigger + the two
  alternatives + persistence requirement a plan with a swept-axis
  headline must satisfy.
- `critic.md` Statistics & Measurement lens item 11 — REVISEs a plan
  whose headline uses `max`/`argmax`/best-of/top-k-mean over
  layer/cell/k/neighbourhood/seed/extraction-point/threshold AND whose
  null band is computed at one fixed axis position, with neither per-draw
  same-selection nor a held-out-frozen axis, and no persisted per-draw ×
  per-axis matrix.

## Files of record

Task body #778 (the origin incident + n=24 asymmetry numbers);
`.claude/agent-memory/reconciler/feedback_layer_selection_asymmetry_is_alternatives_finding.md`,
`.claude/agent-memory/analyzer/feedback_best_layer_snr_selection_biased.md`,
`.claude/agent-memory/critic/feedback_bestofgroup_selection_asymmetry.md`
(the pre-existing agent memories on this pattern);
`.claude/rules/contrastive-negatives.md` (#383 X-vs-(X−Y) sibling
caveat).

**Sibling rule:** `.claude/rules/vectorize-many-cell-fits.md` — the same #778
null battery is that rule's many-draw compute incident (per-draw pool
re-reduction → ~70× batched subset-sum GEMM); a permutation-battery plan
typically fires both.
