---
title: 'workflow-fix: verify_task_body.py — mechanize the clean-result-critic three-beat-ordering
  + arrow-range-drift checks'
kind: infra
tags:
- wf-fix
created_at: '2026-08-21T00:24:05Z'
has_clean_result: false
origin_prompt: 'workflow-fix from /issue 2356: verify_task_body check 21 PASSed a
  multi-figure result ending on a caption (Lens-3 hard-FAIL), consuming 2 review rounds;
  plus arrow-range drift uncovered by the plotted-value check'
workflow: v1
---
# verify_task_body.py: two mechanizable clean-result-critic gaps (three-beat ordering + arrow-range drift)

## Goal

Close two mechanizable gaps in `scripts/verify_task_body.py` that the clean-result-critic caught by hand on #2356, each consuming review rounds the mechanical gate should have pre-empted.

## Gap 1 — check 21 (`check_v4_results_beat`) misses a multi-figure result ending on a caption

The Lens-3 three-beat rubric (`.claude/rules/clean-result-critic-lens-reference.md`, SPEC one-narrative-unit clause) hard-FAILs a `### <result>` whose declared multi-figure unit ends on an image/blockquote caption instead of an interpretation beat ("what-is-plotted above the pair, interpretation below the pair"). `check_v4_results_beat` PASSes that shape. On #2356 this slipped past BOTH the mechanical gate and Claude's round-3 lens, surfaced only by the Codex twin in round 3, then recurred in round 4 — two review rounds spent on a mechanizable defect.

**Fix:** extend check 21 — for a `### <result>` with >1 inline figure, WARN/FAIL when the section's LAST non-empty block (before the next H2/H3 or the footer rule) is an image (`![…](…)`) or a blockquote caption (`> **Figure.** …`) rather than non-caption interpretation prose.

## Gap 2 — the plotted-value-drift check does not cover arrow-form numeric ranges

On #2356 Result 3, the parenthetical "(0.13→0.29 harmful, 0.56→0.61 over-refusal)" quotes only real artifact values, but the arrow semantics differ across arms: armA reads min→max across cells (0.132→0.291) while armB's 0.61 endpoint is the pooled-3b value (0.6097), not the max fold (0.634). Direction and the gate claim are unaffected (no overclaim — the true top cell is higher than quoted), so round-5 Claude flagged it non-blocking. But it is mechanizable.

**Fix:** extend the plotted-value-drift check to parse `X→Y` pairs adjacent to a figure reference and verify both endpoints appear among (or bound) the figure sidecar's plotted values; WARN when the max plotted value exceeds `Y` by more than one rounding unit.

## Provenance

workflow_fix_target: scripts/verify_task_body.py

Surfaced during `/issue 2356` clean-result-critic ensemble (rounds 3–5). Both gaps are in `scripts/verify_task_body.py`; fixes should add fixtures reproducing each shape and update any affected lens-reference wording.
