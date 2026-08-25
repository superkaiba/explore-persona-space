---
title: 'verify_task_body check 31: class-C per-unit-companion candidate set misses
  prose-named committed figures'
kind: infra
tags: []
created_at: '2026-08-23T05:49:17Z'
has_clean_result: false
origin_prompt: workflow-fix-candidate surfaced by clean-result-critic round 1 on task
  2477
workflow: v1
---
<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
candidate-fingerprint: check31-classc-stem-matching-misses-prose-named-figures

## Goal

Fix `verify_task_body.py` check 31's class-C candidate set: it matches committed per-unit companion figures by FILE STEM only, so a committed figure the plan names by PROSE ("distinct-3gram vs judge-score scatter") rather than by stem is invisible to the check, and an orphaned-aggregate body ships with a WARN-silent gap.

## Evidence

Task #2477 clean-result round 1 (2026-08-23): `figures/issue_2477/distinct3gram_vs_score.png` was committed at both body-pinned SHAs (`3f07e928c1`, `c492f4493244`) and never embedded or mentioned in the body; check 31 stayed silent because the plan names the figure in prose, not by stem. The gap was caught only by the Codex clean-result twin + reconciler (binding REVISE, blocker `r5-rank-correlation-per-unit-orphan`; see #2477 events.jsonl `epm:review-reconcile` v1). The Claude clean-result-critic's round-1 verdict proposed the fix shape: extend the candidate set to ALL committed `figures/issue_<N>/*.png` at body-cited SHAs that the body never mentions (WARN class (a) without the per-unit-stem precondition).

## Acceptance

- A committed-but-unmentioned `figures/issue_<N>/*.png` at a body-cited SHA draws a WARN regardless of stem shape.
- Existing WARN/FAIL semantics unchanged elsewhere; add a regression test reproducing the #2477 shape (committed scatter, prose-only plan mention, body silent).
