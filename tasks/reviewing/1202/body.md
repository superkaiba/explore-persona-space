---
title: 'workflow-fix: lint bare list_repo_files in new scripts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6c2d6145c11a
- daily-auto-filed
created_at: '2026-07-09T07:00:42Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): New scripts/ files can
  hand-roll bare list_repo_files( / un-retried Hub verify legs (the #920-shaped class),
  and no workflow_lint check flags them toward hub.verify_repo_paths_uploaded / the
  retried helpers.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08, slice 5) from a
candidate parked on task #997 by a recursion-guarded workflow-fix session.

## Goal

Convert #997's guidance-only coverage of the per-issue-script class into a mechanical gate: a workflow_lint check flagging bare `list_repo_files(` / un-retried Hub verify legs in NEW scripts/ files, with a grandfather allowlist for existing per-issue scripts, pointing at hub.verify_repo_paths_uploaded / the retried helpers.

## Workflow gap

- **Bug observed:** A new script hand-rolling a bare list_repo_files( or an un-retried verify leg reintroduces the #920-shaped false-failure class (transient 504 on a cursor page turns a successful upload's verify into a failure); nothing mechanical catches it.
- **Why it is a workflow gap:** workflow_lint is the project's mechanical gate for exactly this class of recurring call-site discipline; the #997 fix hardened the library path but left new-script call sites guidance-only.
- **Confidence (emitter):** medium (alternatives-critic r1 prose follow-up on #997); the sibling hub.py one-liner is high confidence but off-surface
- **Triage evidence (2026-07-08):** Neither sub-candidate landed. (1) hub.py's list_hf_files_under_path EntryNotFoundError fallback still calls api.file_exists BARE (hub.py:658, un-retried) — contrast the _retry_upload-wrapped file_exists at hub.py:951; but src/explore_persona_space/orchestrate/hub.py is OFF the workflow-fix surface, so that one-liner should ride as a regular kind:infra sibling. (2) workflow_lint.py has NO check flagging bare list_repo_files( / un-retried verify legs in new scripts/ files (grep: no hits) — the in-scope mechanical-gate leg, routed here. Completed #1036 targeted workflow_lint.py for a different bug. No retraction.

## Proposed change (candidate diff sketch — refine in planning)

```
+ workflow_lint.py: new check (bundled into no-flags default) — scan scripts/
+ for bare `list_repo_files(` and un-wrapped `file_exists(`/tree-walk verify
+ legs; grandfather allowlist for existing per-issue scripts (with inline
+ reason); message points at hub.verify_repo_paths_uploaded / _retry_upload.
```
SIBLING (regular kind:infra, NOT this task): wrap hub.py:658's api.file_exists fallback in _retry_upload (one line + test) — hub.py is off the workflow-fix surface.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Secondary: `tests/test_workflow_lint.py` (fixtures + live-tree pass invariant).
- Grep the workflow surface for the pattern before editing
  (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #997 at 2026-07-04T23:28:51Z

Verbatim parked note:

parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug.md § Recursion guard. TWO candidates surfaced this session (logged, NOT auto-routed):
1. source: code-reviewer r2 prose follow-up — target_file: src/explore_persona_space/orchestrate/hub.py — wrap list_hf_files_under_path's un-retried api.file_exists fallback probe (~hub.py:574, a #988-owned block banned for #997) in _retry_upload; one-line change + test; closes the last #920-shaped un-retried Hub call on the verify path. confidence: high.
2. source: alternatives-critic r1 prose follow-up — target_file: scripts/workflow_lint.py — add a check flagging bare list_repo_files( / un-retried verify legs in NEW scripts/ files (grandfather allowlist for existing per-issue scripts), pointing at hub.verify_repo_paths_uploaded; converts #997's guidance-only coverage of the per-issue-script class into a mechanical gate. confidence: medium.
routed: parked (recursion guard) — next orchestrator/human pass may file via scripts/file_infra_task.py.
