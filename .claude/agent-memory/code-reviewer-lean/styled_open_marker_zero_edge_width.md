---
name: styled-open-marker-zero-edge-width
description: A production figure style that sets lines.markeredgewidth 0 makes every plt.plot mfc="none" open marker invisible (axes AND legend glyph); tests that skip set_paper_style cannot see it — render through the batch entrypoint and Read the PNG (#2569 r1 shard F)
metadata:
  type: feedback
---

When a figures module pins a style in its batch driver (`render_all` ->
`set_paper_style("blog")`) and that style sets `lines.markeredgewidth: 0`
(paper_plots.py blog profile), every open marker drawn via `plt.plot(...,
mfc="none")` WITHOUT an explicit `mew=` renders NOTHING: face is none, edge
stroke has zero width — in the axes and in the legend glyph. Data series,
reference markers, and verdict-encoding open/filled distinctions silently
vanish from the production PNGs while the legend still lists them as empty
labels.

**Why:** #2569 r1 shard F (`issue2569_figures.py`): six figures hit — the
learning curve's 7 committed off-recipe companion points, leg-6 rank figure's
Gavish-Donoho references + the entire fit-half-2 series, the three-tier
Llama-to-Qwen series (wholly invisible at a single grid pair: no line
segment, marker the only ink), the dw-alignment BELOW-null verdict points
(every visible point then reads "above null"), and two legend proxies. All
24 tests were green: the tests call the builders under default rcParams
(mew=1.0), so the open markers exist there; `_n_artists` counts Line2D
objects that draw nothing, and PNG-size asserts pass. `scatter(...,
facecolors="none", edgecolors=..., lw=...)` sites and plot sites with
explicit `mew=` were immune.

**How to apply:** on any figure-module round, (1) find the style the BATCH
driver pins and print `rcParams['lines.markeredgewidth']` under it; if 0,
(2) grep the module for `mfc="none"` / `mfc='none'` and flag every
`plt.plot` site without an explicit `mew=`; (3) discriminate live: render
one affected figure through the batch entrypoint (style on) AND via the
test path (style off), Read both PNGs — the disappearing-markers diff is
the proof; (4) demand either explicit `mew=` per site or a scatter+lw form,
plus one test that renders under the pinned style. Sibling:
[[figure-populated-assert-reference-artists]] (artist-count guards can't
see this class either).
