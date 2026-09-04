---
name: c2a-v2-figure-restyle-gotchas
description: Three traps when porting paper figure scripts to c2a_plot_style c2a-v2 (rotated-ylabel overflow, style_score_axis tick snapping, record key in save outputs)
metadata:
  type: reference
---

Traps hit while porting Figures 7 and 9 to `c2a_plot_style` c2a-v2 (2026-09-03):

1. **Rotated ylabel longer than the axes height leaks into the title band.** On a
   full-width canvas (13.1 in) a 3-panel layout leaves ~2.2 in of axes height; a
   long mathtext ylabel (e.g. `Retention $R^2_{i\to j}/R^2_{j\to j}$ ↑`) is
   vertically centered and its ends overdraw the NEIGHBOR panel's title as stray
   arrow/subscript glyphs. Diagnose with `pdftoppm -r 300` + PIL crops (the
   raster preview looks like a font bug). Fix: shorten the ylabel; move the noun
   into the panel title.
2. **`style_score_axis` starts ticks AT `y_min`** (no snapping to round values).
   Pass a round `y_min` and pad by widening lo/hi a full step when data sits
   within ~0.01 of a bound, not by subtracting 0.01 from `y_min` (that yields
   ticks like 0.44/0.49/0.54).
3. **`save_c2a_figure` returns `{"pdf","png","grayscale","record"}`** — older
   scripts doing `v.relative_to(ROOT) for v in outputs.values()` crash on the
   `record` dict; filter `isinstance(v, Path)` and embed `record` in the
   `<stem>.meta.json` sidecar.

Also: `fig.legend` default fontsize is 17 pt script-side under c2a-v2; per-panel
kicker legends above a 3-panel row fit only ncol=1 with `labelspacing=0.3`, and
gridspec `top≈0.56` is needed to clear 2-line titles + kickers (panel_header
`kicker_y≈1.44`, `title_y≈1.06`).

4. **Legend-strip width budget (measured 2026-09-03, c1_cot_maps):** on the
   13.1 in full canvas, QUESTIONS+METRIC kicker-legends (1 strata + 2 metric
   entries at 17 pt) end ~11.5 in — a 56-char 12.5 pt note does NOT fit on the
   same line (content hit 16.8 in). Put the error-bar/note text on a second
   strip row (~0.08 fig-frac below) instead of shrinking fonts. A one-third
   width panel fits ~18-26 title chars at the pinned 22 pt ("Thinking on vs
   off, same weights" = 4.9 in overflowed; "Thinking on vs off" fits).
   `scripts/section45_cot_figure.py::_legend_strip` places blocks by measured
   extents (renderer-adaptive) — reuse it rather than hand-tuned x anchors.
