---
name: sharey-mismatched-row-sets
description: Never sharey/sharex across categorical panels with DIFFERENT row sets; assert realized ticklabels == each panel's own row set post-render
metadata:
  type: feedback
---

Never use `sharey=True` (or `sharex=True`) across panels whose CATEGORICAL
row/column sets differ — the panel drawn LAST owns the shared extent and tick
labels, silently mis-labelling every row of the other panel past the first
set-difference insertion point, and rows beyond the smaller count fall outside
the visible range entirely.

**Why:** #2329 report-verifier round-1 blocker — `layer_profile` drew a 39-row
ce heatmap and a 37-row pe heatmap with `sharey=True`; pe rendered second, so
ce's probe-null `query_content` row displayed under a wrong label and the two
ce-only rows were cropped. It survived the plotter, the methodology-critic,
AND a 20/20 mechanical gate — only a reader-level recompute caught it.

**How to apply:** (1) share an axis only when the tick domain is IDENTICAL
across panels; different row sets get independent axes (or an explicit UNION
row set with absent cells drawn as visibly-absent — hatch/grey + a label
suffix — never a fabricated low/zero value). (2) Add a post-render fail-loud
guard: after ALL panels are drawn, per axes assert
`[t.get_text() for t in ax.get_yticklabels()]` equals that panel's own row
label list in count AND content — checked after the full loop so a
reintroduced shared axis (last-writer-wins) raises instead of shipping.
Worked example: `scripts/issue2329_figures.py::fig_layer_profile`
(commit 91b22ffd0e564665001a423c9ad5ee680e2b03c0).
