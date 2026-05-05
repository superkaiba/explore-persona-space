# P7 — Multi-Panel Small Multiples

**Use when:** The same chart type (P1/P2/P3) repeated across a faceting
variable (e.g. per-topic, per-model-size). Small multiples let the reader do
the comparison directly.

**Do NOT use when:**
- Only 1 facet — just use the base pattern.
- Facets have different chart types — write two separate figures.
- More than 9 panels — consolidate or drop the least informative facets.

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

set_paper_style("generic")

conditions = ["baseline", "c1", "c6", "c7"]
topics = ["geometry", "propagation", "axis-origins", "em-defense"]

rng = np.random.RandomState(0)
values = rng.rand(len(topics), len(conditions)) * 0.5 + 0.4
errs = np.full_like(values, 0.03)

fig, axes = plt.subplots(
    2, 2, figsize=(7.5, 5.0), sharex=True, sharey=True,
    constrained_layout=True,
)
colors = paper_palette(len(conditions))

for idx, ax in enumerate(axes.flat):
    ax.bar(conditions, values[idx], yerr=errs[idx], color=colors, capsize=3)
    ax.set_title(topics[idx], fontsize=10)
    ax.set_ylim(0, 1.05)
    if idx >= 2:
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# Shared y-label + direction arrow, applied ONCE to the figure-level axes
for ax in axes[:, 0]:
    ax.set_ylabel("Alignment rate")
    add_direction_arrow(ax, axis="y", direction="up")

savefig_paper(fig, "summary/alignment_by_topic_grid", dir="figures/")
plt.close(fig)
```

## Layout rules

- `sharex=True, sharey=True` — the whole point of small multiples is a
  common scale. If you need different y-scales, you probably need separate
  figures.
- `constrained_layout=True` — avoids overlapping titles / labels.
- Panel titles are SHORT (≤ 30 chars) and at a smaller fontsize than axes
  labels.
- Use `figsize="generic"` (6.0 × 4.0) or larger; NeurIPS-column sizing is
  too small for multi-panel.

## Pitfalls

- **Don't repeat axis labels on every panel.** Only leftmost-y and bottom-x
  panels need labels.
- **Don't repeat the legend on every panel.** One shared legend via
  `fig.legend(..., loc="outside right center")`.
- **Don't use a multi-panel figure to show 2×1 panels.** That's not small
  multiples; that's two figures stapled together. Split them.
- **Don't exceed 3×3.** Beyond 9 panels, the eye can't do the comparison;
  consolidate via a heatmap (P4) or a summary bar (P1).
