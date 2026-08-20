---
name: grain-rename-literal-glob-consumer
description: "A shard-grain rename (worker-striped -> cell-grain) must sweep every GLOB/REGEX consumer of the OLD grain as a pattern FAMILY; a spelled-out-literal grep (fixture name anchors_gate_w0) cannot see the anchors_gate_w* glob; the tell is a docstring updated beside an untouched glob (#2389 R2 g6)"
metadata:
  type: feedback
---

Rule: when a round renames an artifact naming grain (e.g. worker-striped
`anchors_gate_w{K}` -> cell-grain `anchors_gate_{cell}_w{K}`), audit every
`glob(`/`re.compile(` consumer of the OLD family by PATTERN, not by literal:
grep the shared prefix (`anchors_gate`) across all round scripts and check
each hit's full pattern against the NEW realized name. A deleted/moved-literal
grep keyed on a spelled-out instance (`anchors_gate_w0`, a test fixture name)
structurally cannot match the `anchors_gate_w*` GLOB consumer.

**Why:** #2389 R2 — cluster 3 renamed gate anchor shards to cell grain and
updated `_gate_slice_cap_recalibration`'s DOCSTRING to the new
`anchors_gate_*.jsonl` claim + rewired its barrier to cell-keyed done
manifests, but left the code glob at `anchors_gate_w*.jsonl` (run.py:3119):
39 bank cells, zero matches -> the plan §4.7-item-1 cap recalibration
silently computed `{}` and persisted a clean-LOOKING artifact
(`partial: false`, empty per_cell) that short-circuited all later calls.
The implementer's own (c) pin-sweep claimed the flat names survived only in
sibling fixtures — its grep literal was the fixture name. No test touched
the function, so 160/160 stayed green.

**How to apply:** on any rename/regrain fix: (1) grep the OLD prefix across
scripts, classify each hit literal-vs-pattern; (2) for every glob hit,
fnmatch-probe one realized NEW name live (cheap one-liner); (3) treat a
docstring/comment updated to the new grain beside an untouched pattern line
as near-certain evidence of the miss; (4) empty-aggregation writers that
persist `{}` as a valid artifact are the silent-fail shape — demand a
non-empty assert or a regression test with NEW-grain fixtures. Related:
[[amend-phase-striding-filters]] (regen phase must reproduce generation
filters), [[smoke-shard-namespace-only-done-files]] (namespace every
artifact path), and the full agent's
feedback_spelled_out_literal_sweep_blindspot (same grep class).
