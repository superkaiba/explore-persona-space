---
name: set-title-subtitle-breaks-subplot-grids
description: paper_plots.set_title_subtitle() on a single subplot of a grid produces broken layouts; use fig.text + disable constrained_layout instead
metadata:
  type: feedback
---

`set_title_subtitle(ax, ...)` from `paper_plots.py` is designed for single-axis figures. When applied to a single subplot inside a `plt.subplots(n, m)` grid (e.g., `set_title_subtitle(axes[0, n_cols//2], ...)`), the title is anchored to that one subplot's coordinate system AND the constrained_layout engine reserves space wrong, producing:
- Tiny / squashed actual chart cells
- Title text overlapping subplot 0,0 contents
- Large empty whitespace gaps in the middle of the figure
- Subplots collapsing to thin vertical strips on the figure edges

Anti-pattern (broke #398's heatmap, small_multiples, peak_histogram):
```python
fig, axes = plt.subplots(n_rows, n_cols)
# ... plot ...
set_title_subtitle(axes[0, n_cols // 2], "Title", subtitle="Sub")  # BAD
```

**Fix:** Use `fig.text(x, y, ..., transform=fig.transFigure)` with manual y-coordinates, AND disable constrained_layout BEFORE `plt.subplots` is called so `fig.subplots_adjust(top=...)` actually takes effect:

```python
set_paper_style("blog")
mpl.rcParams["figure.constrained_layout.use"] = False  # disable AFTER set_paper_style
fig, axes = plt.subplots(n_rows, n_cols)
# ... plot ...
fig.text(0.05, 0.975, "Title", fontsize=13, fontweight="semibold",
         color="#1A1A1A", ha="left")
fig.text(0.05, 0.95, "Subtitle line", fontsize=9, color="#5A5A5A", ha="left")
fig.subplots_adjust(top=0.92, bottom=0.07, left=0.10, right=0.97, hspace=0.35, wspace=0.15)
```

**Why:** Knowing this saves a round of figure regeneration when a multi-subplot grid figure renders broken.

**How to apply:** Whenever building a multi-subplot grid (heatmap with cbar + grid title; histograms-over-time small multiples; per-persona × per-step grids), don't call `set_title_subtitle` on any subplot axis. Use the fig.text + subplots_adjust pattern. Single-axis figures (one `plt.subplots()` without `n_rows × n_cols`) can keep using `set_title_subtitle(ax, ...)` — that's its intended use case.

Incident: 2026-05-27, task #398 per-position amendment. All 4 amendment figures rendered broken on first generation; required full regenerate-and-recommit cycle.
