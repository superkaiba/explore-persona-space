---
name: c2a-fixed-scale-overflow-check
description: c2a-v2 paper figures — detect text overflowing the authored canvas by comparing PNG px/240 dpi to the authored width; title char budgets per panel
metadata:
  type: reference
---

`save_c2a_figure` (c2a-v2, `src/explore_persona_space/analysis/c2a_plot_style.py`)
saves with `bbox_inches="tight"`, so text overflowing the authored canvas silently
WIDENS the exported PDF/PNG and shrinks the realized type below the fixed 0.42
scale. Detection recipe: `PIL.Image.open(png).size[0] / 240` vs the authored width
(`c2a_figure` full = 13.095 in, wide = 9.82 in); fig-2-class slop is about +-5%,
anything past ~8% means a title / tick label / legend runs off-canvas.

Budgets at the pinned 22 pt bold panel title (~11.3 pt/char realized): a full-width
single panel fits ~55 title chars; a half panel in a 2-col grid fits ~30-38
depending on margins. Long barh y-tick labels need the gridspec `left` margin sized
for them (17 pt tick ~8.7 pt/char). The legacy 14.4 in canvases were ~9% wider than
the c2a-v2 full canvas, so restyled figures often need shorter titles / earlier
`textwrap` wraps ([[c2a-v2 migration]]). Verified 2026-09-03 on the paper-figstd
round (SAE panel titles collided; qualitative cards overflowed until wraps moved
58→45 chars and aspect 0.556→0.66).
