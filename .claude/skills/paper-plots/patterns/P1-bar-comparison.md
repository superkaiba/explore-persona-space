# P1 — Bar Chart (Discrete Conditions with Error Bars)

**Use when:** Comparing a single metric across a small number of discrete
conditions (2-8). The most common chart type in this project.

**Do NOT use when:**
- Comparing along a continuous axis → P3 (line).
- Showing a distribution → P6 (violin/strip).
- Pre-vs-post paired change → P2.

## Minimal example

```python
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

set_paper_style("neurips")

conditions = ["baseline", "c1_evil_wrong", "c6_vanilla_em", "c6_tulu_25"]
# proportions (e.g. alignment rate): each is a fraction in [0, 1]
values = np.array([0.912, 0.743, 0.684, 0.879])
n_trials = np.array([200, 200, 200, 200])
cis = np.array([proportion_ci(p, n) for p, n in zip(values, n_trials)])
err_lo = values - cis[:, 0]
err_hi = cis[:, 1] - values

fig, ax = plt.subplots()
colors = paper_palette(len(conditions))
bars = ax.bar(conditions, values, color=colors)
ax.errorbar(
    conditions,
    values,
    yerr=[err_lo, err_hi],
    fmt="none",
    ecolor="black",
    capsize=3,
)

# Value labels directly on bars (Chua/Hughes rule: reader never does arithmetic)
for rect, v in zip(bars, values):
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + err_hi[list(bars).index(rect)] + 0.01,
        f"{v * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_ylim(0, 1.05)
ax.set_ylabel("Alignment rate")
add_direction_arrow(ax, axis="y", direction="up")
ax.set_xlabel("Condition")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")  # ≤ 30° rotation only

savefig_paper(fig, "aim5/alignment_by_condition", dir="figures/")
plt.close(fig)
```

## Error-bar source

- **Proportions** → `proportion_ci(p, n)` (chart bars only — do NOT describe them as "95% CI" in the prose; the prose reports p-values and N).
- **Means across seeds** → `std / √n_seeds`.
- **Bootstrap CI** → precompute externally; pass `yerr` as `[lo, hi]` deltas
  from the mean.

Never a bar without error bars. If a condition is single-seed and has no CI,
either run more seeds, or annotate the bar with `n=1` and note in the
caption.

## Pitfalls

- **Don't sort bars by value without saying so.** If the order conveys
  meaning (e.g. narrative order, matched protocol), keep it. If you sort,
  note "sorted by Y" in the caption.
- **Don't stack bars.** Readers cannot compare stacked segments; use grouped
  bars (two bars side-by-side per x-tick) instead.
- **Don't label bars with `0.912` when the metric is a proportion.** Show
  `91.2%` — more readable. (Call `f"{v * 100:.1f}%"`.)
- **Don't use > 5 colors.** If you have 6+ conditions, group them into
  related pairs with the same fill color + different hatch, or split into
  two panels (P7).
