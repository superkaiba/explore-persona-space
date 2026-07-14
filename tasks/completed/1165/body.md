---
title: 'workflow-fix: Venue caveat in experimenter nohup .env memory'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e817f4184b22
- daily-auto-filed
created_at: '2026-07-09T06:58:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The always-loaded experimenter
  memory instructs an unconditional ''set -a && source .env && set +a'' prefix for
  every nohup-detached pod command — safe on RunPod (bootstrap pushes .env) but #923-adjacent
  if applied on the GCE lane, which exports tokens via its startup script and has
  NO .env file (unconditional sourcing inside an &&-chain fails the whole command).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #984 by a recursion-guarded workflow-fix session.

## Goal

Add a one-line venue caveat to the memory: the unconditional prefix is RunPod-only; on GCE use the gotchas.md conditional form (`if [ -f ./.env ]; then set -a; . ./.env; set +a; fi`).

## Workflow gap

- **Bug observed:** The always-loaded experimenter memory instructs an unconditional 'set -a && source .env && set +a' prefix for every nohup-detached pod command — safe on RunPod (bootstrap pushes .env) but #923-adjacent if applied on the GCE lane, which exports tokens via its startup script and has NO .env file (unconditional sourcing inside an &&-chain fails the whole command).
- **Why it is a workflow gap:** Agent memories are always-loaded steering; an unqualified rule written for RunPod will eventually be applied verbatim on a GCE workload and reproduce incident #923.
- **Confidence (emitter):** medium
- **Sweep verification (2026-07-08):** Verified 2026-07-08: the memory file still carries the unconditional prescription with no venue caveat (read in full); the sibling fix pattern already exists in upload-policy.md lines 183-187 and gotchas.md (the #923/#944 conditional-sourcing entry) to cross-reference. One-line additive edit to a workflow-surface file (.claude/agent-memory/** is in scope).

## Proposed change (candidate diff sketch — refine in planning)

Append to 'How to apply': '(RunPod pods only — bootstrap pushes .env. GCE-lane workloads have NO .env; use the conditional form from .claude/rules/gotchas.md: if [ -f ./.env ]; then set -a; . ./.env; set +a; fi.)'

## Scope / surfaces

- Primary target: `.claude/agent-memory/experimenter/feedback_load_env_in_nohup.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: .claude/agent-memory/experimenter/feedback_load_env_in_nohup.md
- origin: parked candidate on task #984 at 2026-07-04T12:47:48Z

Verbatim parked note:

> parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug.md § Recursion guard. source: prose-followup (alternatives critic, #984 plan review r1). target_file: .claude/agent-memory/experimenter/feedback_load_env_in_nohup.md. bug_observed: memory instructs unconditional 'set -a && source .env' for every nohup-detached pod command — safe on RunPod (bootstrap pushes .env) but #923-adjacent if ever applied in a GCE context. proposed_change: add a one-line venue caveat (RunPod-only; GCE lane has no .env — use the gotchas.md conditional form). confidence: medium. related_task: #984. routed: parked (recursion guard) — next non-workflow-fix orchestrator pass may file it.
