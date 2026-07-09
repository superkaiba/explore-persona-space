---
title: 'workflow-fix: lint YAML-scalar truncation + marker consumer'
kind: infra
tags:
- wf-fix
- wf-fix-fp:80ce7afca559
- daily-auto-filed
created_at: '2026-07-09T07:00:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Two mechanizable verifier
  gaps from #873: (1) an unquoted workflow.yaml markers-list plain scalar containing
  '' #'' silently truncates at the comment marker and --check-references passes because
  the regen matches the truncated parse; (2) a poller/watcher feature claiming mid-run
  surfacing can ship with no live consumer (the plan-v2 ETA tripwire had no PollResult/JSON
  field until a critic caught it).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #873 (recursion-guarded workflow-fix session).

## Goal

Add two workflow_lint checks: (1) flag markers-list plain scalars whose parsed value ends in ',' or '(' or has unbalanced parens (the truncated-comment signature); (2) verify a new poller-posted marker/advisory claiming mid-run surfacing is referenced by a consumer surface (rg the marker name across SKILL.md / tick_triage.py / autonomous_session_watch.py / poll_pipeline.py). Planning may split these into two tasks (distinct fingerprints).

## Workflow gap

- **Bug observed:** Two mechanizable verifier gaps from #873: (1) an unquoted workflow.yaml markers-list plain scalar containing ' #' silently truncates at the comment marker and --check-references passes because the regen matches the truncated parse; (2) a poller/watcher feature claiming mid-run surfacing can ship with no live consumer (the plan-v2 ETA tripwire had no PollResult/JSON field until a critic caught it).
- **Why it is a workflow gap:** Both are silent-integrity failures in the marker/monitoring contract that mechanical lint can catch at commit time instead of a critic catching them by luck.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

check_marker_scalar_integrity(): parse workflow.yaml markers lists; for each plain scalar value v: FAIL if v.rstrip().endswith((',','(')) or v.count('(') != v.count(')').
check_poller_marker_consumers(): for poller-posted marker kinds (posted_by ~ poll_pipeline), require >=1 reference in the consumer surfaces.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #873 at 2026-07-02T18:28:25Z

Verbatim parked note:

parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug.md § Recursion guard). TWO candidates logged for the next human/orchestrator pass, NOT auto-routed:

(1) source: prose-followup (Claude code-reviewer round 1, mechanizable: yes). target_file: scripts/workflow_lint.py. bug_observed: an unquoted YAML plain scalar in a workflow.yaml markers-list field containing ' #' silently truncates at the comment marker (the #873 posted_by initially parsed as 'skill (...); poll_pipeline (runtime tripwire,' and markers.md rendered a dangling cell); --check-references passed because the regen matched the truncated parse. proposed_change: add a lint check flagging markers-list plain scalars whose parsed value ends in ',' or '(' or has unbalanced parens (the truncated-comment signature).

(2) source: prose-followup (Codex alternatives critic round 1, mechanizable: yes). target_file: scripts/workflow_lint.py (or a verifier). bug_observed: a poller/watcher feature that CLAIMS mid-run surfacing can ship with no live consumer (the plan-v2 ETA tripwire had no PollResult/JSON field until the critic caught it). proposed_change: a workflow-surface check that a new poller-posted marker/advisory claiming mid-run surfacing is reachable by watcher/tick/notification/poll-JSON-consumer code (rg the marker name across SKILL.md/tick_triage.py/autonomous_session_watch.py/poll_pipeline.py). NOTE: resolved for #873 itself in plan v3; parked as a GENERAL verifier candidate.
