---
name: Constant-valued bootstrap yields negative matplotlib yerr
description: Errorbar figures over bootstrap CIs crash on cells with constant inputs (emit rate pinned at exactly 1.0) — clamp yerr widths to >= 0
type: feedback
---

When a figure computes errorbar widths as `(mean - ci_lo, ci_hi - mean)` from a
bootstrap over a CONSTANT array (e.g. own-emit rate exactly 1.0 on saturated
cells, or any 0/1-rate panel), float error can make `mean - ci_lo == -1e-17`,
and `ax.errorbar` hard-fails with `ValueError: 'yerr' must not contain negative
values`.

**Why:** `np.percentile` of a near-constant bootstrap distribution is not
exactly equal to its mean at machine precision; saturated rate metrics make
constant arrays COMMON in marker-sweep figures, so this fires on real data,
not just synthetics.

**How to apply:** clamp every bootstrap-derived errorbar width at the append
site — `yerr_lo.append(max(0.0, m - lo))` / `yerr_hi.append(max(0.0, hi - m))`
— in any new sweep-figures script that plots rate or saturated log-prob panels.
Caught by the #547 synthetic-rig figures smoke (2026-06-10); fix in
`scripts/i547_clean_result_figures.py`.
