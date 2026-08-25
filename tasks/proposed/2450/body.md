---
title: 'paper_plots sidecar: keep exact-(0,0) singleton collections and propagate
  per-panel series labels'
kind: infra
tags: []
created_at: '2026-08-21T13:34:40Z'
has_clean_result: false
workflow: v1
---
# `paper_plots` sidecar capture: keep exact-(0,0) singleton collections and propagate per-panel labels on multi-panel figures

## Goal

Fix two silent under-recording defects in the figure sidecar capture used by
`savefig_paper`, both of which make a CORRECT figure look wrong to every
downstream reviewer and verifier. Each has now produced a false finding in a
#2329 report-gate round, three rounds running.

## The two defects

### (a) `_extract_scatters` drops genuine exact-(0,0) single-point collections

A scatter collection holding exactly one point at the origin is discarded by
the extractor, so the sidecar under-counts the figure's real points.

Observed: `q35_ladder_decay_transfer` — the sidecar captured **14** points while
the PNG renders **16**. The two missing points are the genuine all-zero (0,0)
install-prefix-end cells: real data at the origin, not padding. A reviewer
comparing sidecar to caption sees a 14-vs-16 discrepancy and has to open the PNG
to discover the figure was right all along.

Suspected mechanism: an origin-valued singleton is indistinguishable from an
empty/placeholder collection under a falsy-or-emptiness test. The fix must
distinguish "collection with one point whose coordinates happen to be (0, 0)"
from "collection with no points".

### (b) Multi-panel sidecars record `series: None` for every panel after the first

On a multi-axes figure the sidecar keeps series labels only for the first panel;
subsequent panels get `series: None`.

Observed: `q35_ladder_decay_decay_raw` renders BOTH model panels (q25 and q35),
but its sidecar's q35-panel series names are `None`. This directly caused a
wrong review finding: the #2329 `methodology-critic` read the sidecar and ruled
the figure "q25 only", a claim the PNG refuted. The report text was correct; the
sidecar was not.

## Why this is worth fixing rather than working around

The sidecar exists so a reviewer can check a figure WITHOUT loading the PNG.
Both defects make it lie in the conservative direction — under-reporting real
content — so they generate FALSE FAILs, which cost a full revision round each
time and train reviewers to distrust the sidecar. Three consecutive #2329 gate
rounds paid this cost (the 14-vs-16 count, the "q25 only" misread, and the
follow-on re-verification of both).

## Acceptance criteria

1. A figure with a scatter collection holding exactly one point at (0, 0) has
   that point present in its sidecar; a genuinely EMPTY collection is still
   omitted. Both directions covered by tests.
2. A multi-panel figure records per-panel series labels for EVERY panel, not
   only the first — reproduce with a 2-panel figure whose panels carry distinct
   labels and assert both appear.
3. Regression fixture reproducing the #2329 shapes: a `transfer`-like scatter
   whose point set includes exact-(0,0) singletons (assert the sidecar count
   equals the rendered count), and a `decay_raw`-like 2-panel figure (assert no
   panel's `series` is `None` when labels exist).
4. Existing sidecars stay readable — no schema change that breaks a consumer
   (`scripts/verify_report.py`, `report-verifier`, the plotter). If the schema
   must change, state the migration and keep the old shape parseable.
5. Tests failing before / passing after; no new red in the no-flags
   `workflow_lint.py` run or the mapped-test selection.

## Notes for the implementer

- Do NOT retroactively regenerate #2329's committed sidecars as part of this
  task — its figures are pinned by SHA in a report body under review, and a
  re-render would force a re-pin of every image URL. This task fixes the capture
  path forward; #2329 recorded both defects in its verification marker.
- The consumer side already has a correct habit worth preserving: when sidecar
  and PNG disagree, the PNG is authoritative. That is a workaround, not a fix.

## Provenance

Surfaced as a prose `mechanizable`-class recommendation by `report-verifier`
during #2329 round `q35_ladder_decay` report verification (round 1 FAIL,
2026-08-21), after the same extractor had produced a false finding in each of
the three preceding gate rounds. The orchestrator verified both shapes against
the committed sidecars + PNGs before filing. Evidence: #2329 `events.jsonl`
`epm:report-verified` round 1; `figures/issue_2329/q35_ladder_decay/` sidecars
for `q35_ladder_decay_transfer` (14 vs 16) and `q35_ladder_decay_decay_raw`
(q35-panel `series: None`) on branch `issue-2329-q35-ladder-decay`.

- target_file: src/explore_persona_space/analysis/paper_plots.py
- fingerprint: sidecar-origin-singleton-drop-and-multipanel-series-none
- confidence: high — both shapes observed on committed artifacts, each having
  produced a false review finding
