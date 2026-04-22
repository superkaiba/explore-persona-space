# P6 — Violin / Strip Plot for Seed Distributions

**Use when:** You have 5+ seeds per condition and the *distribution* (not
just the mean) is part of the story — bimodality, high variance, outliers.

**Do NOT use when:**
- < 5 seeds per condition — use a bar + individual points (P1) instead.
- The mean is the whole story — P1 with error bars is simpler.

## Minimal example

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

set_paper_style("neurips")

rng = np.random.RandomState(0)
conditions = ["baseline", "tulu_25", "tulu_100"]
data = {
    "baseline": rng.normal(0.90, 0.02, 15),
    "tulu_25": rng.normal(0.82, 0.04, 15),
    "tulu_100": rng.normal(0.70, 0.08, 15),
}
rows = [(cond, v) for cond, vals in data.items() for v in vals]

colors = paper_palette(len(conditions))
fig, ax = plt.subplots()

parts = ax.violinplot(
    [data[c] for c in conditions],
    positions=range(len(conditions)),
    showmeans=False, showextrema=False, showmedians=False,
    widths=0.7,
)
for pc, col in zip(parts["bodies"], colors):
    pc.set_facecolor(col)
    pc.set_alpha(0.25)
    pc.set_edgecolor(col)

# Strip overlay — actual seed values
for i, c in enumerate(conditions):
    vals = data[c]
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax.scatter(
        [i] * len(vals) + jitter, vals,
        color=colors[i], s=20, alpha=0.9, edgecolor="white", lw=0.5,
    )
    # Median bar
    ax.hlines(
        np.median(vals), i - 0.2, i + 0.2,
        color="black", lw=1.5, zorder=5,
    )

ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions)
ax.set_ylabel("Alignment rate")
add_direction_arrow(ax, axis="y", direction="up")

savefig_paper(fig, "aim5/tulu_violin", dir="figures/")
plt.close(fig)
```

## Pitfalls

- **Always overlay the actual points.** A violin without points hides `n`.
  The reader needs to see that there are 15 seeds, not 2.
- **Don't fill at alpha=1.0.** The violin shape is a density estimate, not
  data; use `alpha=0.25` so the points dominate visually.
- **Median bar, not mean bar.** Violins are for distributions; the median
  is more robust to outliers than the mean.
- **Don't use when `n ≤ 5`.** A KDE-based violin with 3 points is noise.
  Switch to P1 (bar + individual scatter).
