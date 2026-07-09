---
title: 'daily-held: Gate spawn_session dispatch paths on auth-outage'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-09T07:01:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 3): The #1027 auth-outage guard
  suppresses only WATCHER-issued spawn arms; PM-session / file_infra_task.py / spawn_session.py
  dispatch paths remain ungated, so during a fleet auth outage those paths can still
  burn fresh ~100K-token session spawns into instant-freeze churn.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1027.

## Goal

Decide whether to extend the #1027 auth-outage episode gate from watcher-issued spawns to the spawn_session.py / file_infra_task.py / PM dispatch paths, and implement it if warranted.

## Workflow gap

- **Bug observed:** The #1027 auth-outage guard suppresses only WATCHER-issued spawn arms; PM-session / file_infra_task.py / spawn_session.py dispatch paths remain ungated, so during a fleet auth outage those paths can still burn fresh ~100K-token session spawns into instant-freeze churn.
- **Why it is a workflow gap:** the failure originates in the workflow surface named below, not in any one experiment.
- **Confidence (emitter):** see parked note

## Proposed change (candidate diff sketch — refine in planning)

  + def _auth_outage_active() -> bool:  # reads ~/.eps-autonomous/auth-outage.json
  + in cmd_spawn_issue(--auto): if _auth_outage_active(): refuse/defer with a
  +   loud reason (episode id, resume-probe pointer); fail-open past the TTL

## Scope / surfaces

- Primary target: `scripts/spawn_session.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- JUDGMENT CALL (route3): the #1027 plan deliberately left these paths ungated; confirm outage churn from non-watcher paths has actually been observed before changing blocking behavior. Fail-open TTL must match the watcher gate's posture.

## Provenance

- workflow_fix_target: scripts/spawn_session.py
- origin: parked candidate on task #1027 at 2026-07-05T07:23:59Z

Verbatim parked note:

```
parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard). Planner prose follow-up (conditional, not auto-routed): the guard bounds WATCHER-issued churn only; PM-session / file_infra_task.py / spawn_session.py spawn paths are ungated by design. IF outage churn from those paths ever bites, a spawn_session.py-side check of ~/.eps-autonomous/auth-outage.json is the concrete candidate (target_file: scripts/spawn_session.py). Logged for a future human/orchestrator pass; nothing filed.
```
