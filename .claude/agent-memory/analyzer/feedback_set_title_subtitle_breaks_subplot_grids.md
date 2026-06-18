---
name: set-title-subtitle-breaks-subplot-grids
description: paper_plots.set_title_subtitle() breaks layouts in two ways — on a subplot of a grid (squashed cells, title overlap) and under the blog style (savefig.bbox="tight" defeats subplots_adjust). Use fig.text+subplots_adjust for grids; inline set_title(pad=36)+annotate for single-axis blog figures.
metadata:
  type: feedback
---

`set_title_subtitle(ax, ...)` from `paper_plots.py` is designed for single-axis figures in non-tight-bbox styles. Two known failure modes:

**1. Subplot grids (any style).** Calling it on one subplot of a `plt.subplots(n, m)` grid (e.g. `axes[0, n_cols//2]`) anchors the title to that subplot AND confuses constrained_layout: tiny squashed cells, title overlapping subplot contents, big whitespace gaps. (Incident: 2026-05-27 task #398 — all 4 amendment figures broken, full regenerate-and-recommit cycle.)

Fix — `fig.text` + manual margins, with constrained_layout disabled AFTER `set_paper_style`:
```python
set_paper_style("blog")
mpl.rcParams["figure.constrained_layout.use"] = False
fig, axes = plt.subplots(n_rows, n_cols)
fig.text(0.05, 0.975, "Title", fontsize=13, fontweight="semibold", color="#1A1A1A", ha="left")
fig.text(0.05, 0.95, "Subtitle", fontsize=9, color="#5A5A5A", ha="left")
fig.subplots_adjust(top=0.92, bottom=0.07, left=0.10, right=0.97, hspace=0.35, wspace=0.15)
```

**2. Single-axis figures under the blog style.** Blog sets `savefig.bbox: "tight"`, which recomputes the bbox at save time and strips manual `subplots_adjust` whitespace — the title block overlaps the topmost data. (Incident: task #468 — all 4 figures had title clipping until a manual pad bump.) Fix — drop `set_title_subtitle` and inline its three calls with extra pad (default pad=24 is too tight; use 36):
```python
ax.set_title(TITLE, loc="left", fontsize=13, fontweight="semibold", pad=36)
ax.annotate(SUBTITLE, xy=(0.0, 1.0), xytext=(0, 8), xycoords="axes fraction",
            textcoords="offset points", ha="left", va="bottom", color="#5A5A5A", fontsize=10)
fig.supxlabel(SOURCE, x=0.02, ha="left", color="#7A7A7A", fontsize=8, fontstyle="italic")
fig.tight_layout()
```

`set_title_subtitle(ax, ...)` remains fine for single-axis figures outside the blog style.
