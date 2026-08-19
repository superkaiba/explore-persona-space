---
title: 'verify_task_body: flag caption-unit vs figure-sidecar count drift (check-45
  family)'
kind: infra
tags: []
created_at: '2026-08-18T11:15:22Z'
has_clean_result: false
origin_prompt: 'clean-result-critic round-2 prose follow-up on #2333 (epm:clean-result-critique
  v2): caption claimed per-draw points over per-pair-aggregated figure data'
workflow: v1
---
# verify_task_body: flag caption-unit vs figure-sidecar count drift (per-draw/per-row claims over per-pair-aggregated points)

## Goal

Extend the `verify_task_body.py` check-45 family: when a v4 body's figure caption / alt text / setup line claims the plotted points are per "draw" or per "row", but the figure's committed `.meta.json` sidecar reports a per-series n equal to a body-declared pair count × donor-scheme count (rather than × K draws), flag caption-unit/sidecar-count drift (WARN grade; the clean-result-critic Lens 3 stays the binding arm).

## Why now

#2333 clean-result-critique round 2 (2026-08-18) caught exactly this on a freshly-embedded per-unit companion: the Result 6 caption said "Each point is one steered prefill draw" while the pinned sidecar showed n=388 per series and the generating code (`scripts/issue2333_figures.py:531-548` @ e99a37ba30) iterates `f_cells.jsonl` keyed `(pair_id, arm_slug)` — K=5 draws pre-averaged, so points are per pair per arm. Result 7's setup line shared the defect. Per-unit companions are now mandatory in every result section (SPEC § low-level data plot behind every aggregate), so this grain-misstatement class will recur mechanically with every newly-embedded companion.

## Sketch

- Parse figure embeds in `## Results` + their adjacent caption/setup sentences for the tokens `draw`, `row`, `point is one`, `each point`.
- Resolve the embedded PNG to its committed `.meta.json` sidecar (same stem); read per-series n where present.
- Cross-reference body-declared pair counts (the n=… survivor statements) and K (draws) from the Methodology table; if caption claims draw/row grain and sidecar n ≈ pairs × schemes (not × K), emit WARN naming both numbers.
- Fail-soft: no sidecar or no parseable n → no check (never a false FAIL).

## Candidate metadata

- target_file: scripts/verify_task_body.py
- fingerprint: caption-unit-sidecar-count-drift-check45-family
- confidence: medium-high (recurring class; heuristic needs care to stay WARN-grade)
- source: clean-result-critic round-2 prose follow-up on #2333 (epm:clean-result-critique v2)
