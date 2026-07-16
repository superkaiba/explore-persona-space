---
name: Bootstrap CIs vs matplotlib xerr/yerr non-negative-offset contract
description: Errorbar figures crash whenever a CI bound crosses the point value — float-epsilon on constant inputs (#547) OR a genuinely INVERTED quantile CI around the point estimate at tiny smoke n (#1335 r4) — clamp offsets element-wise to >= 0
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

**Second firing mode (#1335 r4, 2026-07-15, GCP att-20260715-122509):** not
float-epsilon — the CI can be GENUINELY inverted vs the point value when the
CI is a quantile band over bootstrap/delta draws while the plotted `value` is
a separately-computed point estimate: at tiny smoke n the point estimate falls
outside the draw quantiles (and the same happens whenever value/CI come from
different estimators). Crashed `barh(..., xerr=[v-lo, hi-v])` at
`scripts/issue1335_figures.py` with `'xerr' must not contain negative values`
AFTER the run's own earlier smoke data happened not to invert — data-dependent,
so a passing smoke does NOT clear the class. Fix shape: a vectorized helper
`_ci_offsets(values, ci_lo, ci_hi) -> [np.maximum(0, v-lo), np.maximum(0, hi-v)]`
at EVERY xerr/yerr site (element-wise arrays, never scalars), plus a pinned
unit test feeding a negative delta + inverted CI (both sides) through the real
figure function to `savefig` (`tests/test_issue1335_figures_errorbar.py`).
NaN bounds are safe (matplotlib's negativity check passes NaN; no bar drawn) —
don't coerce them. Empty arrays are safe too.
