---
title: 'workflow-fix: verify_task_body check-17 cross-checks Context lineage vs parent_id'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3165632ed032
created_at: '2026-07-16T08:43:43Z'
has_clean_result: false
origin_prompt: 'clean-result-critic candidate on #1345 r1: check 17 accepted ''fresh
  direction (no parent task)'' with parent_id: 825'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1345 (emitting agent: clean-result-critic, round 1).

## Goal

verify_task_body.py check 17: cross-check the Context lineage clause against frontmatter parent_id.

## Workflow gap

- **Bug observed:** Check 17 accepted a `**Context:**` lineage clause reading "fresh direction (no parent task)" on task #1345 whose frontmatter carries `parent_id: 825` — the lineage-token regex checks token presence only, never cross-checks frontmatter `parent_id`. Caught only by the LM critic's Lens 5.
- **Why it is a workflow gap:** the wrong-parent/denied-parent case is mechanically decidable from frontmatter yet only enforced by an LM lens; every child task's clean-result can ship a false "fresh direction" provenance past the verifier.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "fresh.direction\|no parent" scripts/verify_task_body.py` → 0 hits for a parent_id cross-check near the check-17 lineage regex (presence-only logic; absence-of-guard claim — the 0-hit result IS the evidence) (2026-07-16). Per-target: scripts/verify_task_body.py — the check-17 lineage-token region exists (critic cites :3727/:3925); no parent_id read in it.

## Proposed change (candidate diff sketch — refine in planning)

In check_context_provenance_row (near verify_task_body.py:3727/3925):
+ parent_id = frontmatter.get("parent_id")
+ if parent_id and re.search(r"fresh\s+direction|no\s+parent", context_row, re.I):
+     fail(f"Context lineage says 'fresh direction/no parent' but frontmatter parent_id={parent_id}")
+ if parent_id and not re.search(rf"#\s*{parent_id}\b", context_row):
+     fail(f"Context lineage does not name parent #{parent_id}")

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep before editing (`grep -rn 'lineage' scripts/verify_task_body.py tests/test_verify_task_body.py`); add the regression tests.

## Constraints / invariants

- Workflow-surface only; forward-only (grandfathered v3/v2 bodies never newly hard-FAILed — gate the new FAIL on the v4 sentinel per the spec's forward-only rule; the planner decides).
- `tests/test_verify_task_body.py` extended + passing; workflow_lint PASS.
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 3165632ed032
