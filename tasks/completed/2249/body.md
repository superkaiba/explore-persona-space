---
title: 'verify_task_body check 17: upgrade context-parent-lineage-mixed WARN to FAIL
  when frontmatter parent_id is set and the Context row carries a no-parent clause'
kind: infra
tags: []
created_at: '2026-08-12T19:27:20Z'
has_clean_result: false
workflow: v1
---
Surfaced by clean-result-critic on #2224 round 1 (epm:body-critique, 2026-08-12): the body's Context row claimed 'fresh direction (no parent)' while frontmatter carries parent_id: 2221. Check 17 already detects exactly this shape but only WARNs (context-parent-lineage-mixed); the contradiction is mechanically decidable (frontmatter parent_id set AND a no-parent phrase in the Context row) and should hard-FAIL so it blocks before the critic instead of costing a review round. Target: scripts/verify_task_body.py check 17 + its tests.
