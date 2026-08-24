---
title: 'verify_task_body per-unit vocabulary scan: credit the what-is-plotted beat
  and strip-point phrasing'
kind: infra
tags: []
created_at: '2026-08-23T05:49:31Z'
has_clean_result: false
origin_prompt: workflow-fix-candidate surfaced by clean-result-critic round 2 on task
  2477
workflow: v1
---
<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
candidate-fingerprint: perunit-vocab-scan-ignores-beat1-strip-point-phrasing

## Goal

Fix `verify_task_body.py`'s per-unit-evidence vocabulary scan (check-58 family): it WARNs on a result section whose what-is-plotted beat explicitly says the figure plots "every per-item mean as a strip point" — the scan does not credit the beat-1 line and its vocabulary list lacks the "strip point" phrasing, producing a false-positive WARN on a section that IS the per-unit view.

## Evidence

Task #2477 clean-result round 2 (2026-08-23): the hero-strip result ("Base generation under the chat template fails the floor") carries beat-1 text "every per-item mean as a strip point" and the rendered per-item strips, yet the scan WARNed it as lacking per-unit evidence. Surfaced by the Claude clean-result-critic round-2 verdict (non-blocking observation, mechanizable: yes — extend the section-prose vocabulary window to include the beat-1 what-is-plotted line, or add "strip point" to the per-unit vocabulary list).

## Acceptance

- A section whose what-is-plotted beat contains per-unit vocabulary (incl. "strip point") no longer WARNs.
- Regression test reproducing the #2477 hero-section shape.
