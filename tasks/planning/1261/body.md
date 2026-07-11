---
title: 'daily-fix: setsid pointers for two experimenter memory aside'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d3de4b370c20
- daily-auto-filed
created_at: '2026-07-11T06:51:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): two weak shape-less non-trio
  nohup MENTIONS survive after #1236 (peft-readme L11 ''upload_folder(...) as background
  nohup''; gcp-metadata-runner L31-32 GCE relaunch ''via SSH under nohup'') - prescriptive
  asides lacking setsid'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

add setsid (or a contract pointer) to both asides

## Workflow gap

- **Bug observed:** two weak shape-less non-trio nohup MENTIONS survive after #1236 (peft-readme L11 'upload_folder(...) as background nohup'; gcp-metadata-runner L31-32 GCE relaunch 'via SSH under nohup') - prescriptive asides lacking setsid
- **Provenance / evidence:** Alternatives critic prose follow-up, #1236 plan v2 (parked 2026-07-10T07:11:57Z). Emitter confidence low - filed per the 2026-06-11 standing directive; verified live: both asides still bare-nohup.

## Scope / surfaces

- Primary target: `.claude/agent-memory/experimenter/feedback_peft_readme_local_path.md, .claude/agent-memory/experimenter/feedback_gcp_metadata_runner_token_too_long.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: d3de4b370c20

- workflow_fix_target: .claude/agent-memory/experimenter/feedback_peft_readme_local_path.md, .claude/agent-memory/experimenter/feedback_gcp_metadata_runner_token_too_long.md
