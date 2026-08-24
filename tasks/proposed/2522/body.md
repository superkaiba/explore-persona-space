---
title: paper_plots _extract_bars drops bar-container labels from sidecar point rows
kind: infra
tags: []
created_at: '2026-08-24T07:49:04Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate surfaced in prose by the #2479 analyzer round-2
  report (codex-interp-r1-c11 root cause)'
workflow: v1
---
# paper_plots `_extract_bars`: propagate bar-container labels into flattened sidecar point rows

## Goal

Fix `src/explore_persona_space/analysis/paper_plots.py::_extract_bars` so that per-row `series` labels survive into bar-chart `.meta.json` sidecars whenever the generator sets `ax.bar(label=...)`. Today the group-level label never reaches the flattened `points` rows, so labeled bar charts produce sidecars with null/absent `series` values even though the label exists on the matplotlib bar container.

## Provenance

Surfaced by the #2479 analyzer round-2 report (2026-08-24) as the root cause of interpretation-critic concern `codex-interp-r1-c11` ("Figure 3 and Figure 5 sidecars omit explicit series labels although their group arrays are hash-distinct"). The #2479 round-2 fix re-annotated generator-side; this task fixes the extractor so labeled bar charts are sidecar-complete without per-generator workarounds.

## Design sketch

In `_extract_bars`, read each bar container's legend label (`container.get_label()` / the artist's label, skipping matplotlib's auto `_nolegend_`-style underscore labels) and stamp it as `series` on every flattened point row derived from that container. Grouped-bar figures with N containers then emit N distinct `series` values; unlabeled containers keep today's behavior.

## Acceptance criteria

1. A figure built with two `ax.bar(..., label=...)` calls produces a sidecar whose every `points` row carries the correct non-null `series` value.
2. An unlabeled single-series bar chart is unchanged (no spurious `_nolegend_`-style values written as `series`).
3. A pytest covering both cases lands in `tests/` (extend the existing paper_plots sidecar tests if present).
4. Existing sidecar consumers (verify/report tooling) stay green — full relevant test files pass.
