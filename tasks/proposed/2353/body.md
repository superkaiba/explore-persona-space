---
title: 'verify_task_body: WARN-level per-unit-or-exemption check for every v4 result
  section (multi-point aggregate figures escape the current single-aggregate-stat
  trigger)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-17T19:58:20Z'
has_clean_result: false
parent_id: 2330
origin_prompt: 'clean-result-critic round-3 prose follow-up on #2330: per-unit check
  only fires on single-aggregate-stat figures'
workflow: v1
---
target_file: scripts/verify_task_body.py

## Gap

The v4 per-unit-behind-aggregates check in `verify_task_body.py` fires only when a result figure is a single-aggregate-stat shape. A MULTI-POINT aggregate figure (e.g. a per-cell paired-dots + bars comparison panel) passes mechanically with NO per-unit view and NO `Per-unit exemption:` line, even when every sibling `### <result>` section carries one — the clean-result-critic (Lens 11) is currently the only catching arm.

## Incident (#2330 fu1 fold, round-3 clean-result critique, 2026-08-17)

The folded cap2048 robustness section shipped through verify_task_body OVERALL PASS as the only one of 10 result sections with neither a per-unit view nor an exemption line; its figure (figures/issue_2330/cap2048_comparison.png — per-cell open/filled point pairs + cap-hit bars) is aggregate-per-cell, not per-unit-per-context, so the existing single-aggregate-stat trigger never fired. Caught only by the Claude clean-result-critic Lens 11 in round 3.

## Fix shape

Add a WARN-level v4 check: every `### <result>` H3 block must contain either (a) a per-unit view claim (existing recognizer vocabulary: per-unit / per-context / per-layer profile / points labeled / pinned per-unit artifact pointer) or (b) a literal `Per-unit exemption:` line. WARN, not FAIL — grandfathered bodies and legitimately-exempt aggregate sections ship with the acknowledgment mechanism already used for other WARNs. Add fixtures: a v4 body with one multi-point-aggregate section lacking both (WARN fires) and one with an exemption line (clean).

Evidence: task #2330 round-3 clean-result-critique marker (epm:clean-result-critique v3, fix 4 + prose follow-up), figures/issue_2330/cap2048_comparison.png @ fb4b88f39591c904f7963da386a28260eff2e355.
