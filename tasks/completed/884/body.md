---
title: 'workflow-fix: setsid-detach VM-side long CPU phases (watcher-stop kill)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b3d3e989fc01
created_at: '2026-07-02T20:03:17Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by #833 orchestrator: watcher ALIVE-BUT-STALLED
  force-stop killed the healthy Phase-D fit running as session-child bg-Bash; require
  setsid+nohup detachment + pid in breadcrumb for VM-side long compute phases'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #833 (emitting agent: orchestrator).

## Goal

Require VM-side long-running CPU phases (Phase-D fits, aggregation) launched
from /issue sessions to be setsid+nohup detached with the pid recorded in the
stage-dispatch breadcrumb, so watcher session force-stops cannot kill in-flight
work.

## Workflow gap

- **Bug observed:** The #833 Phase-D fit (`issue833_fit_onpolicy.py`, a ~3-6h
  VM-CPU phase) ran as a plain bg-Bash child of the driving autonomous session.
  The `autonomous_session_watch.py` ALIVE-BUT-STALLED pass force-stopped that
  session at 2026-07-02T19:53Z (self-report frozen 148.9m) and the stop killed
  the session's process tree — taking the healthy, mid-bootstrap fit down with
  it. ~2h of loading+fit progress was lost; the successor session had to
  re-diagnose (no OOM, no traceback — a pure signal kill) and relaunch from
  scratch.
- **Why it is a workflow gap:** SKILL.md's Step 8 results-landed batch and the
  `stage-dispatch` breadcrumb convention let the orchestrator launch VM-side
  long compute as ordinary bg-Bash (`subagent=orchestrator-bg-bash`), and the
  watcher's force-stop respawn path assumes killing a stalled session is safe.
  Neither surface requires the launched compute to survive the session, and
  the breadcrumb records no pid — so a successor cannot even tell whether the
  phase is alive without forensic `ps`/log archaeology. The rules mention
  `nohup` only for pod-side launches (`.claude/rules/code-style.md`).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/skills/issue/SKILL.md (Step 8 results-landed / Step 9 entry guard
breadcrumb convention):
+ Any VM-side long-running (>~15 min) compute phase launched by the
+ orchestrator (Phase-D fits, aggregation, off-pod judges already covered by
+ batch_judge) MUST be launched `setsid nohup ... < /dev/null >> <log> 2>&1 &`
+ so it survives session death / watcher force-stop, AND its breadcrumb MUST
+ record `pid=<pid> log=<abs log path>` so a successor session re-attaches
+ (liveness = ps -p <pid>) instead of relaunching.

.claude/rules/code-style.md (nohup section):
+ The nohup rule applies to VM-side long CPU phases too, not just pod-side
+ launches; plain bg-Bash children die with the session (watcher force-stop
+ kills the process tree — #833 incident 2026-07-02).

scripts/autonomous_session_watch.py (optional hardening, planner decides):
+ Before force-stopping an ALIVE-BUT-STALLED session, log any live non-shell
+ child processes matching scripts/*.py so the kill collateral is visible in
+ the alert marker.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, .claude/rules/code-style.md`
- Secondary (planner decides): `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'orchestrator-bg-bash\|run_in_background' .claude/ CLAUDE.md scripts/`)
  and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).
- Distinct from #883 (setsid launcher-script SSH-MCP POD launch shape) — that
  covers pod-side workload launches; this covers VM-LOCAL phases killed by
  watcher session stops.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, .claude/rules/code-style.md
- fingerprint: b3d3e989fc01

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md, .claude/rules/code-style.md
bug_observed: The #833 Phase-D fit ran as a plain bg-Bash child of the driving session; the autonomous_session_watch ALIVE-BUT-STALLED force-stop killed the session process tree and took the healthy 2h fit down with it, losing all progress
why_workflow_gap: SKILL.md and the breadcrumb convention permit launching multi-hour VM-side compute as session-child bg-Bash with no detachment requirement and no pid in the breadcrumb, while the watcher's force-stop assumes killing a stalled session is collateral-free
proposed_change: Require VM-side long-running CPU phases (Phase-D fits, aggregation) launched from /issue sessions to be setsid+nohup detached with pid recorded in the stage-dispatch breadcrumb, so watcher session force-stops cannot kill in-flight work
diff_sketch: |
  + SKILL.md: VM-side >15-min compute phases launch `setsid nohup ... &`,
  + breadcrumb records `pid=<pid> log=<abs path>`; successor re-attaches via
  + `ps -p <pid>` instead of relaunching.
  + code-style.md nohup section: extend to VM-side long CPU phases.
confidence: high
related_task: #833
<!-- /workflow-fix-candidate -->
