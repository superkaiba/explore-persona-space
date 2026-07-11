---
title: 'daily-fix: forward rc==0 stderr at watcher spawn-issue sites'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ca4858cbe2c2
- daily-auto-filed
created_at: '2026-07-11T06:51:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): the watcher''s ~6 spawn_session.py
  spawn-issue subprocess children capture output and on rc==0 surface only the stdout
  first line / suppression sentinel - the child''s rc==0 stderr (e.g. the spawn_session.py
  registration-collision detail) is discarded'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

thread _forward_marker_child_stderr(res, 'spawn_session spawn-issue (<arm>)') after each rc==0 check at the spawn-issue child sites

## Workflow gap

- **Bug observed:** the watcher's ~6 spawn_session.py spawn-issue subprocess children capture output and on rc==0 surface only the stdout first line / suppression sentinel - the child's rc==0 stderr (e.g. the spawn_session.py registration-collision detail) is discarded
- **Provenance / evidence:** Alternatives critic prose follow-up, #1221 plan r1 (parked 2026-07-10T07:20:07Z, recursion guard); #1221 closed only the file_infra_task.py hop - the watcher spawn sites were outside its Goal. Verified live 2026-07-10: _forward_marker_child_stderr threaded at 7 marker/set-status sites, at NO spawn-issue site.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: ca4858cbe2c2

- workflow_fix_target: scripts/autonomous_session_watch.py
