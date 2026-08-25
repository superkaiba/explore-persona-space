---
title: 'verify_task_body.py: add per-pair to the per-unit-evidence vocabulary'
kind: infra
tags: []
created_at: '2026-08-25T17:46:02Z'
has_clean_result: false
origin_prompt: 'interpretation-critic round-2 surfaced suggestion on #2564 (workflow-fix-on-bug
  auto-file)'
workflow: v1
---
# Add "per-pair" to verify_task_body.py's per-unit-evidence vocabulary

## Goal
`scripts/verify_task_body.py`'s per-unit-evidence check recognizes a fixed vocabulary of low-level-data phrasings (per-unit, per-cell, per-item, ...) when deciding whether a clean-result `### <result>` carries the required low-level per-unit data plot alongside its aggregate. Minimal-pair experiments (e.g. #2564) phrase their unit as "per-pair" — currently not in the vocabulary, so a compliant body can be flagged (or a reviewer has to hand-wave the check). Add "per-pair" (and the hyphenless "per pair" form if the matcher is token-based) to the vocabulary list, with a regression test.

## Provenance
Surfaced by the interpretation-critic (Claude) in task #2564 interpretation round 2 (2026-08-25): "one mechanizable workflow suggestion: add 'per-pair' to verify_task_body.py's per-unit-evidence vocabulary." Filed by the #2564 orchestrator per .claude/rules/workflow-fix-on-bug.md (surfaced-prose follow-up).

## Acceptance
- Vocabulary list in scripts/verify_task_body.py includes per-pair.
- A test in tests/test_verify_task_body.py covers a body whose per-unit evidence uses "per-pair" phrasing and passes the check.
- Full test file green.
