---
title: 'workflow-fix: OOM-protect detached VM-side compute phases (choom -600 + pivot
  rule)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:003f05f8de5e
created_at: '2026-07-04T04:00:14Z'
has_clean_result: false
origin_prompt: 'orchestrator prose follow-up on #811: earlyoom collateral kill of
  detached fits phase; add choom protection + second-kill pod pivot to the Step 9
  detached-launch convention'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #811 (emitting agent: orchestrator).

## Goal

Document + mechanize OOM protection for VM-side detached long compute phases: the Step 9 entry-guard § "Detached VM-side long compute phases" launch shape should include an earlyoom-protection step (sudo choom -n -600 on the detached pid) and name the earlyoom collateral-kill failure mode.

## Workflow gap

- **Bug observed:** #811's off-instance STAGE=fits (detached setsid per #833, ~5h projected) was SIGTERM-killed by earlyoom at ~2h (rc=143, 0 checkpoints written) when a NEIGHBOR process spiked VM memory 50%->10% in 23 min; the fit's own RSS was 6.8 GiB — earlyoom picked it by badness (1002) despite it not being the spiker. ~2h of compute lost.
- **Why it is a workflow gap:** the #833 detached-launch convention (SKILL.md Step 9 entry guard) hardens against session-kill-domain death but says nothing about earlyoom on the shared VM — any long VM-side detached phase is exposed to collateral kills from fleet co-tenancy, and the convention has no protection or pivot guidance.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/skills/issue/SKILL.md § "Detached VM-side long compute phases":
+ After the pid-identity verify, protect the phase from earlyoom collateral
+ kills: `sudo -n choom -n -600 -p "$PHASE_PID"` (children inherit; -600 not
+ -1000 — genuinely bigger consumers should still be reaped first). A phase
+ killed rc=143 with an earlyoom journal line naming a NEIGHBOR spike is the
+ collateral-kill signature: relaunch protected; on a SECOND kill despite
+ protection, route the phase to the cpu-bigmem/cpu-mid pod lane instead.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'setsid nohup' .claude/ CLAUDE.md`) and update every hit; list them in the plan (CLAUDE.md § vectorize bullet + `.claude/rules/vectorize-many-cell-fits.md` may carry sibling prose).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 003f05f8de5e

Surfaced prose (orchestrator observation, task #811, 2026-07-03): earlyoom collateral kill of a detached VM-side fit phase; mitigated in-session with sudo choom -n -600; convention gap in the #833 detached-launch shape.
