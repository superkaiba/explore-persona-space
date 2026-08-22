---
name: v2-captions-schema-from-brief
description: v2 Step-7a captions.json shape comes from the orchestrator BRIEF (dict keyed by manifest id), not plotter.md's list schema; non-figure manifest items -> status "not run" with the artifact named
metadata:
  type: feedback
---

When the Step-7a brief specifies a captions.json schema, follow the BRIEF, not
plotter.md's default list-of-objects schema.

**Why:** #2329 dogfood (2026-08-18): the brief required a dict keyed by
manifest figure id — `{status: "rendered"|"not run", aggregate_view (ONE png
for the report body), per_unit_views (list, for the detailed companion doc),
caption_bullets (strictly factual, <=3 sentences' worth), not_run_reason}` —
because Step 7c consumes it mechanically. plotter.md's `plot_name`/`stem` list
schema would have broken the splice.

**How to apply:** (1) Map EVERY manifest id; multiple rendered files fold into
one id (aggregate + `_perpair` companion; a compact heatmap can be the
aggregate over a `_ce`/`_pe` small-multiple pair, e.g. `layer_profile` over
`probe_layer_curves_ce/pe`). (2) A manifest item realized as HTML dashboards
(not a matplotlib figure) gets `status: "not run"` with the reason NAMING the
existing HTML paths — never an html path in `aggregate_view` (the consumer
expects png) and never a fabricated mapping. (3) Ground captions in the
`.meta.json` sidecars (`text/suptitle`, `text/axes[].{title_left,xlabel,ylabel,
legend_labels,annotations,xticklabels}`, `n_series`, `total_points`) plus the
rendered PNG; report manifest-vs-rendered deviations (missing sub-panels,
quantity substitutions) in the return text instead of silently papering over.
