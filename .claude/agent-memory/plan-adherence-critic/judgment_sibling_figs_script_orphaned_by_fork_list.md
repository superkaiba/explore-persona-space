---
name: sibling-figs-script-orphaned-by-fork-list
description: Parent pipelines often split figures across figures.py + <topic>_figs.py; a fork list naming only "figures" silently orphans the sibling renderer — enumerate `scripts/issue<M>*fig*` against the manifest
metadata:
  type: feedback
---

When a plan forks a parent's script set ("thin forks of the issue<M> twins:
run/judge/analysis/mapshift/figures/dashboards"), check whether the parent
rendered any manifest figure via a SIBLING figure script the fork list does
not name (e.g. `issue2162_mapshift_figs.py` beside `issue2162_figures.py`).
A manifest figure whose parent producer is the unlisted sibling ends up with
NO renderer in the child diff, while every listed fork looks complete — and
the child data script may even say "the figure-side marking is the figures
fork's job" pointing at a fork that never consumes its outputs.

**Why:** #2329 r1 — manifest figures `mapshift_r2` / `mapshift_shift_prediction`
/ `dv3_2afc` had no producer: the parent's renderer was
`scripts/issue2162_mapshift_figs.py` (hard-pinned N_LAYERS=28 + issue_2162
paths, not reusable), the plan §4.6 fork list omitted a 2329 twin, and the
implementer's register never declared the gap. The plan-text omission does not
excuse it — the manifest is the approved checklist.

**How to apply:** for each manifest figure id, grep the round's plot scripts
for the id AND its `source` basename (`fresh_fit.json`, `dv3ext.json`, ...);
zero hits with no declared `not run`/deferral ⇒ Major missing-component
finding. Also `ls scripts/ | grep -i '<parent-issue>.*fig'` to enumerate ALL
parent renderers before trusting the fork list. Related: [[judgment_smoke_slice_and_figure_transform_elements]].

**Two literal-mismatch sub-cases that are NOT missing coverage** (#2329 r2):
(a) manifest `source` strings are plan-time SKETCHES — when a source basename
doesn't hit, grep the PRODUCER's write sites for the realized filenames
(`fresh_fit_diagnostics.json` vs sketch `fresh_fit.json`); producer/consumer
agreement + a registered deviation passes, but recommend correcting the
manifest source fields so the report-verifier's manifest-completeness
recompute doesn't trip later. (b) figure-id vs savefig-name deltas
(`hero_f_by_type` id → `hero_ftype` file): a renderer wired to the declared
sources under a different output name is coverage, not a gap.
