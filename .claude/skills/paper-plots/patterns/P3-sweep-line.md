# P3 — Line Plot Across a Continuous Sweep

**Use when:** Showing a metric over a continuous hyperparameter axis (LR,
epochs, training-data size, model scale), with seeds-as-error-bars.

**Do NOT use when:**
- x-axis is categorical → P1.
- Only 2 x-points → P2 (pre/post).

## Minimal example

```python
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

set_paper_style("neurips")

lrs = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4])
# shape (conditions, lrs, seeds) — here 2 conditions × 5 LRs × 3 seeds
means = np.array([
    [0.21, 0.32, 0.45, 0.49, 0.41],
    [0.18, 0.25, 0.30, 0.35, 0.37],
])
sems = np.array([
    [0.02, 0.03, 0.02, 0.02, 0.04],
    [0.02, 0.03, 0.03, 0.03, 0.03],
])
labels = ["tulu_25 (claim)", "tulu_100 (baseline)"]
colors = paper_palette(len(labels))

fig, ax = plt.subplots()
for i, (lbl, col) in enumerate(zip(labels, colors)):
    ax.errorbar(
        lrs, means[i], yerr=sems[i],
        color=col, marker="o", lw=1.5, capsize=3, label=lbl,
    )
    # Lighter confidence band
    ax.fill_between(
        lrs,
        means[i] - sems[i],
        means[i] + sems[i],
        color=col, alpha=0.15, lw=0,
    )

ax.set_xscale("log")
ax.set_xlabel("Learning rate")
ax.set_ylabel("Effect size (η²)")
add_direction_arrow(ax, axis="y", direction="up")
ax.legend(loc="upper left")

savefig_paper(fig, "aim5/lr_sweep", dir="figures/")
plt.close(fig)
```

## Error-bar source

- Across seeds: `std / √n_seeds` as shaded band AND as error-bar caps.
- Across bootstrap resamples: precompute, pass as asymmetric `[lo, hi]`.

## Pitfalls

- **Log-scale the x-axis for LR / data-size sweeps.** Most sweeps span
  orders of magnitude; linear x compresses the interesting region.
- **Don't connect points through a missing value.** If one seed failed,
  either rerun or break the line at that point and annotate.
- **Don't use more than 4 lines.** 4+ lines means "no clear winner" — split
  into panels (P7) or consolidate.
- **Mark the winning x-value.** A vertical line or a star marker at the
  best-performing LR makes the claim visually obvious.
