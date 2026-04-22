# P4 — 2D Parameter-Grid Heatmap

**Use when:** The heatmap IS the finding — e.g. showing a phase-transition
across a 2D grid of (LR × epochs). The structure only pops out in 2D.

**Do NOT use when:**
- You can make the point with a 1D sweep (P3) — prefer that.
- The 2D grid is sparse or irregular — use a scatter (P5) instead.
- You want to compare 2-3 conditions — use a grouped bar (P1).

## Minimal example

```python
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    savefig_paper,
    set_paper_style,
)

set_paper_style("neurips")

lrs = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4])
epochs = np.array([1, 2, 4, 8, 16])
# shape (len(epochs), len(lrs)) — row-major so origin='lower' puts (0,0) at bottom-left
grid = np.random.RandomState(0).rand(len(epochs), len(lrs))

fig, ax = plt.subplots(figsize=(5.0, 4.0))
im = ax.imshow(
    grid, origin="lower", aspect="auto", cmap="viridis",
    vmin=0, vmax=1,
)

# Annotate every cell with its value — reader never does arithmetic
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(
            j, i, f"{grid[i, j]:.2f}",
            ha="center", va="center",
            color="white" if grid[i, j] < 0.5 else "black",
            fontsize=8,
        )

ax.set_xticks(range(len(lrs)))
ax.set_xticklabels([f"{x:.0e}" for x in lrs], rotation=0)
ax.set_yticks(range(len(epochs)))
ax.set_yticklabels(epochs)
ax.set_xlabel("Learning rate")
ax.set_ylabel("Epochs")
ax.grid(False)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Alignment rate")

savefig_paper(fig, "aim5/lr_epochs_phase_diagram", dir="figures/")
plt.close(fig)
```

## Colormap choice

- `viridis` — default. Perceptually uniform, colorblind-safe, prints
  readably in grayscale.
- `plasma` / `cividis` — alternatives if `viridis` clashes with adjacent
  figures.
- **NEVER** `jet`, `rainbow`, or `hsv`. Not perceptually uniform; misleads.
- **NEVER** an unmarked diverging colormap (`RdBu_r`) unless zero is a
  meaningful center. If you use one, specify `vmin=-vmax` so zero sits in
  the middle.

## Pitfalls

- **Turn the grid off** (`ax.grid(False)`). The imshow itself is the grid.
- **Annotate every cell.** If the reader has to eyeball color-to-value
  mapping, the heatmap is failing at its job.
- **Don't skip the colorbar label.** "Alignment rate" not just "value."
- **Don't flip axes accidentally.** `origin="lower"` + row-major data means
  the first row of `grid` is the BOTTOM row of the heatmap. Always set
  `origin` explicitly; never rely on default.
