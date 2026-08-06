---
name: constrained-layout-ignores-figtext-caption
description: Constrained layout reserves space for legend_ax but NOT fig.text captions — the legend band must clear the caption by itself; verify with a draw-time text-artist extent audit, not arithmetic
metadata:
  type: feedback
---

Matplotlib constrained layout (active after `set_paper_style`) allocates space
for real subplots — a gridspec `legend_ax` gets its band — but ignores
`fig.text(...)` entirely. A bottom-anchored caption (`fig.text(0.006, 0.006,
..., va="bottom", wrap=True)`) is drawn INTO whatever occupies the bottom of
the figure. With a long caption + a thin legend band, the legend's LOWER row
prints over the caption's top lines (#1739 five-method figure: band 0.16 →
legend rows y 0.072–0.109 vs caption 0.006–0.089; the collided composite was
misread in review as text "bleeding into the panels").

**Why:** constrained layout's happy render of `legend_ax` gives false
confidence; nothing in the layout engine protects the caption region, so the
gridspec `height_ratios` band is the ONLY thing keeping legend and caption
apart, and it silently re-collides whenever caption wording grows.

**How to apply:**
- Size the legend band so the measured legend-bottom clears the measured
  caption-top by ≥ ~0.03 of figure height (the template figures' accepted
  register is +0.033..+0.041). Copying another figure's band ratio is NOT
  sufficient — caption length differs per figure.
- Verify empirically, never by arithmetic: monkeypatch `savefig_paper` with a
  probe that calls `fig.canvas.draw()`, walks `legend.findobj(Text)` +
  panel-axes `get_window_extent()`, and asserts (a) zero legend-text × panel
  intersections, (b) legend_bottom − caption_top > 0.03. Then crop the
  rendered full-res PNG at the suspect bands and Read it — downsampled
  whole-figure views hide 6-8pt collisions.
- When attributing "stray text" in a rendered figure: enumerate ALL text
  artists with window extents before hypothesizing a second legend/annotation;
  an ncol=3 six-entry legend fills COLUMN-major, so its second row reads as
  the 2nd/4th/6th labels running together across the figure.
