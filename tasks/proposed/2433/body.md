---
title: 'workflow_lint size-ratchet: 09-step-5.md + 18-step-10d.md regrew past SKILL_DOC_SIZE_GRANDFATHER
  caps — fleet-wide no-flags rc=1'
kind: infra
tags:
- workflow-fix
- lint
created_at: '2026-08-20T22:42:01Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate emitted by #2378 r6 experiment-implementer
  (fleet-red no-flags lint attribution)'
workflow: v1
---
## Goal

Restore a green no-flags `workflow_lint.py` run fleet-wide: two main-tree skill step docs regrew past their `SKILL_DOC_SIZE_GRANDFATHER` ratchet caps without the #1753 landing-bytes cap raise, so every no-flags lint run (inline payload lint gates, Step 9c-adjacent runs, pre-push legs) now exits rc=1 with 2 spurious errors that sessions must hand-attribute as pre-existing.

## Symptom (measured 2026-08-20, worktree fresh-synced to origin/main)

- `.claude/skills/issue/steps/09-step-5.md`: 103,751 B > ratchet cap 100,300 B
- `.claude/skills/issue/steps/18-step-10d.md`: 287,529 B > ratchet cap 282,700 B

Some landing on these files skipped the ratchet protocol (the lint's own remedy text prescribes a measured cap raise at landing time).

## Fix

Per the lint's remedy text, either raise the two `SKILL_DOC_SIZE_GRANDFATHER` caps in `scripts/workflow_lint.py` to the measured size + the ≤2.8 KB corridor, or trim the docs back under their caps. Verify with a no-flags `workflow_lint.py` run exiting 0 (given no other new red), and run the lint-family pin tests (`tests/test_workflow_lint*.py`).

## Provenance

Surfaced by the round-6 experiment-implementer on task #2378 (payload-attribution: the 2 errors name files byte-identical to origin/main, no round-committed file). Dedup checked 2026-08-20: no existing proposed task covers these two caps (#2252/#2327/#2374/#2409/#2414/#2420/#2421/#2423/#2426 are adjacent but distinct).
