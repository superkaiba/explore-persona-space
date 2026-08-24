---
name: bar-sidecar-series-annotation
description: savefig_paper bar sidecars drop per-row series labels even when ax.bar(label=...) is set; scatter rows carry them natively — annotate bar sidecars deterministically from the generator script
metadata:
  type: feedback
---

`savefig_paper`'s bar extractor (`_extract_bars`, paper_plots.py) records the
container label only at group level; the flattened `points` rows keep `_group`
indices and NO `series` column, so a labeled two-series bar chart ships a
sidecar a critic reads as unlabeled (#2479 r1 codex c11). Labeled `ax.scatter`
rows DO carry `series` natively (`_xy_rows` adds it when the label is set and
not `_`-prefixed).

**Why:** the dashboard viewer and the degenerate-series check key on per-row
`series`; a `<none>`-labeled group is a guaranteed plot-prose-match concern.

**How to apply:** for bar figures, either fix the extractor or have the
generator script deterministically re-annotate its OWN sidecar right after
`savefig_paper` (map `_group` index → container label in draw order — see
`scripts/issue2479_r2_figfix.py::_annotate_bar_sidecar`). Verify with
`jq '[.points[].series] | unique'` before committing. Also: the Lens-14
verifier check (`verify_task_body.py` concerns audit) clears via
`task.py address-concern` ledger entries alone when every concern is
addressed — in-body concern-id acks are needed only for open/deferred ones.
