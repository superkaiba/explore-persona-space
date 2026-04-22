# Style Reference — Paper-Quality Plots

Authoritative invariants for every figure produced via the `paper-plots`
skill. If a pattern file or an ad-hoc script deviates from these, the
pattern is wrong, not the style reference.

---

## Palette (Wong 2011 / IBM colorblind-safe)

```
#0072B2  blue            (C0 — primary claim)
#E69F00  orange          (C1 — comparison / baseline)
#009E73  bluish green    (C2 — control)
#CC79A7  reddish purple  (C3)
#56B4E9  sky blue        (C4)
#D55E00  vermillion      (C5)
#F0E442  yellow          (C6 — avoid as a line color; OK as fill)
#000000  black           (C7 — reference / ground truth)
```

Use `paper_palette(n)` to grab the first `n` colors. Never exceed 5 in a
single chart (Chua/Hughes rule). If 6+ categories are needed, consolidate or
split into small multiples (P7).

**Palette order carries meaning:**
- First slot = the primary claim / "the condition that matters"
- Second slot = the comparison / baseline
- Black (last slot) = reference lines, ground truth, theoretical predictions

---

## Typography

| Parameter | Value | Notes |
|---|---|---|
| `font.family` | DejaVu Sans | Cross-platform, no LaTeX required |
| Body (`font.size`) | 10 pt | |
| Axis labels (`axes.labelsize`) | 11 pt | Slightly larger than body, per Perez |
| Title (`axes.titlesize`) | 11 pt | |
| Ticks (`xtick/ytick.labelsize`) | 9 pt | Smaller than labels OK |
| Legend (`legend.fontsize`) | 9 pt | |
| `pdf.fonttype` / `ps.fonttype` | 42 | Type-42 so fonts stay editable in Illustrator / Inkscape |
| `svg.fonttype` | "none" | Keep SVG text as text, not paths |

Multiply every font by `font_scale` (default 1.0). Use `font_scale=1.2`
for talk-slide variants of the same figure.

---

## Size

| Target | Figsize (inches) | Use for |
|---|---|---|
| `neurips` | 5.5 × 3.4 | Default — roughly one NeurIPS column |
| `generic` | 6.0 × 4.0 | Broader or taller layouts; GitHub inline |

Call `set_paper_style("neurips")` at the top of the script. Do not override
`figsize` by hand unless the chart type demands it (e.g. P7 multi-panel).

---

## Despine + grid

- `axes.spines.top = False`, `axes.spines.right = False` — strip visual
  clutter (Tufte / tvhahn principle).
- `axes.grid = True`, `grid.alpha = 0.25`, `grid.linewidth = 0.5` — grid is
  present but not distracting. Only the reader who NEEDS it sees it.

---

## Error bars

- `errorbar.capsize = 3`
- For a proportion with N trials: `lo, hi = proportion_ci(p, N)` (Wald 95%)
- For a mean across seeds: `std / √n`
- For a bootstrap CI: precompute elsewhere, pass in explicitly
- **Missing error bars = missing claim.** Every point on a chart that could
  vary across seeds or across an eval set needs bars.

---

## Save formats

Always both:
- `.png` — 300 DPI, Commit tag in pnginfo (greppable from file)
- `.pdf` — vector, Commit tag in PDF metadata
- `.meta.json` sidecar — commit hash + ISO-8601 UTC timestamp + figsize

Path convention: `figures/<aim-or-experiment>/<short-name>.{png,pdf,meta.json}`.

---

## Direction arrows

Every y-axis whose direction is non-obvious gets a `↑ better` or `↓ better`
suffix via `add_direction_arrow(ax, axis="y", direction="up")`.

Non-obvious examples:
- Loss (↓ better)
- Accuracy (↑ better — but readers know this; optional)
- KL divergence (↓ better — non-obvious, ALWAYS add)
- Alignment score (depends on the definition — ALWAYS add)
- Cosine similarity (depends — specify)

---

## Frame / legend

- `legend.frameon = True`, `legend.edgecolor = "lightgrey"`, `legend.facecolor = "white"`
- Legend LOC: pick the location that obscures the fewest data points; never
  rely on `best` for published figures (it flips between runs).
- Inside-axes > outside-axes unless the legend occupies > 25% of panel area.
