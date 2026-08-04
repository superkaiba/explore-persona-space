---
title: 'workflow-fix: no-progress heartbeats mask tick_triage staleness'
kind: infra
tags:
- wf-fix
- wf-fix-fp:634a8dbfec45
created_at: '2026-08-04T14:00:40Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation 2026-08-04 on #1491: hourly ''state unchanged
  / awaiting watcher respawn'' epm:progress heartbeats kept tick_triage.py reading
  HEALTHY (marker age 13m) while the task sat at approved for ~5.7h with no pod, no
  SLURM job, and an unchanged branch tip; STALE-REDRIVE never fired.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate observed by the orchestrator during interactive work on task #1491 (emitting agent: orchestrator, interactive chat session).

## Goal

Stop a no-progress heartbeat marker from resetting `tick_triage.py`'s marker-age staleness clock, so a session that is stuck but chatty can still be detected and re-driven.

## Workflow gap

- **Bug observed:** task #1491 sat at status `approved` for ~5.7h with zero forward progress (no pod, no SLURM job, no status change) while its autonomous session posted hourly `epm:progress` heartbeats reading "state unchanged ... awaiting watcher respawn for durable recovery". Each heartbeat is a fresh marker, so `tick_triage.py` returned `HEALTHY status=approved, marker age 13m — chain alive` on every tick and STALE-REDRIVE never fired. The session heartbeated itself out of the recovery it was explicitly waiting for.
- **Why it is a workflow gap:** `tick_triage.py`'s staleness signal is marker RECENCY, which silently conflates "the session is making progress" with "the session is still talking". A session that correctly recognizes it is stuck and says so on the task's event stream is thereby made invisible to the staleness detector — the more honest the stuck session is, the less likely it is to be rescued. Nothing in the current predicate compares consecutive heartbeats for state equality.
- **Confidence (emitter):** medium-high (the mechanism is directly observed and the five heartbeats are on #1491's event stream at 09:46 / 10:47 / 11:46 / 12:45 / 13:45Z; the right FIX shape is a judgment call for the planner — see below).
- verified-at-filing: `grep -nE "marker_age|stale|HEALTHY|unchanged|progress_signature|no.?progress" scripts/tick_triage.py` → 25 hits in 1 file, all in `scripts/tick_triage.py` (the sole named target); hits cover `STALE_S_DEFAULT`, `stale_s()`, `issue_liveness_reason`, and the `marker age {…}m` verdict string, and NONE implements a no-progress / repeated-identical-heartbeat predicate — the absence this filing claims (2026-08-04). Live-state corroboration the same day: `pod.py list-ephemeral --issue 1491` → "No ephemeral pod recorded"; fellows `squeue -u $USER` → empty; `git log --oneline origin/main..origin/issue-1491` → 5 commits, unchanged across all five heartbeats.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized by the orchestrator from a directly observed incident; the planner should choose between the shapes below with the file open)

Candidate shapes, in rough preference order:
1. **Progress-signature staleness.** Alongside marker age, hash a small progress signature (status + branch tip + latest NON-heartbeat marker kind/ts + pod/job presence). If the signature is unchanged across N consecutive ticks, treat the task as stale regardless of marker age.
2. **Heartbeat-class exclusion.** Classify `epm:progress` notes that self-declare no state change (a `state unchanged` / `awaiting respawn` idiom, or an explicit `[tick-heartbeat]` prefix the tick skill already writes) as NON-clearing for staleness purposes, so marker age is computed from the newest substantive marker.
3. **Escalate-only observer.** Leave the verdict alone and add a watcher pass that flags N identical-state heartbeats at an ACTIVE status, matching the existing escalate-only observer passes.

Whatever shape is chosen, the sibling agent-side rule is worth landing with it: a session that cannot advance should post `epm:failure v1` + park at `blocked` rather than emitting recurring no-progress heartbeats, so the state is legible to both the watcher and a human. That half is a `.claude/skills/issue-tick/SKILL.md` / `.claude/skills/issue/SKILL.md` prose change and may belong in the same task.

## Scope / surfaces

- Primary target: `scripts/tick_triage.py`
- Secondary (prose sibling, same fingerprint): `.claude/skills/issue-tick/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Must NOT re-introduce the #1051 / #1629 regressions the current predicate deliberately protects: a live identity-verified `pid=` breadcrumb, a fresh `log=` mtime, a fresh `[long-phase-heartbeat]` note, and recent HUMAN activity in the session transcript must all still read HEALTHY. A long detached fit that legitimately emits periodic heartbeats is exactly the case the existing carve-outs exist for — the fix must separate "heartbeat from a phase that is genuinely running" from "heartbeat from a session that is idle", not suppress heartbeats wholesale.
- Fail toward keep-alive on an unreadable/ambiguous signal (the existing convention).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/tick_triage.py
- fingerprint: 634a8dbfec45

Observed by the orchestrator on 2026-08-04 while the user asked to "check progress" on task #1491. Not a candidate block from a subagent — a direct orchestrator observation, filed under the same protocol per `.claude/rules/workflow-fix-on-bug.md` § "If the orchestrator is itself the agent that found the bug".
