---
title: 'workflow-fix: verify_task_body total-prose budget ignores folded follow-up
  rounds'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dbf4bf4bf15a
created_at: '2026-07-03T09:41:38Z'
has_clean_result: false
origin_prompt: 'reconciler v12 prose follow-up on #763: check_v4_word_caps computed
  800 + 0 x 250 on a two-round body'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a reconciler prose follow-up on task #763 (9a-bis reconcile v12, 2026-07-03).

## Goal

Make `scripts/verify_task_body.py` `check_v4_word_caps` count FOLDED same-issue follow-up rounds in `live_followup_rounds`, so the total-prose budget scales `800 + 250 × rounds` as the spec intends.

## Workflow gap

- **Bug observed:** on #763's body carrying TWO folded follow-up rounds (`deception-rubric-reanchor`, `neutral-contrast-and-cofit` — both named in the `**Context:**` footer and the Methodology Rounds note), the verifier reported "total content prose is 2262 words (budget 800: 800 + 0 x 250 per extra round; Methodology excluded)" — `live_followup_rounds` = 0.
- **Why it is a workflow gap:** the WARN threshold is then far too strict for exactly the multi-round consolidated bodies the budget clause was scaled for; a Codex clean-result critic elevated the mis-scaled WARN into a lens FAIL that a reconciler round had to overrule (one avoidable reconcile invocation).
- **Confidence (emitter):** low-medium (reconciler: "appears not to count folded rounds; did not affect this verdict") — filed per the standing directive; the spawned session's planner verifies the detection logic with the file open.

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/verify_task_body.py (check_v4_word_caps):
- live_followup_rounds = <current detection — likely counts open children / a
-   frontmatter field that folded same-issue rounds never touch>
+ count follow-up rounds from durable signals a folded round always leaves:
+   the task's epm:same-issue-followup-run markers (via task_workflow.list_events
+   when --issue is given), OR the footer's `same-issue follow-up round` clauses /
+   the Methodology **Rounds:** note as the body-only fallback.
+ budget = 800 + 250 * live_followup_rounds
(add a test row: a body with 2 folded rounds gets budget 1300)
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep: `grep -n "live_followup_rounds\|250" scripts/verify_task_body.py`; sync SPEC.md § Conciseness caps wording if it names the detection; extend tests/test_verify_task_body.py.

## Constraints / invariants

- Workflow-surface only; the verifier stays runnable body-file-only (the events-based count must degrade gracefully when only a file path is given); existing tests green + a new pinning row.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: dbf4bf4bf15a
