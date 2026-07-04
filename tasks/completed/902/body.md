---
title: 'workflow-fix: exit-137 kill-source verification checklist + PM stop-marker-before-kill'
kind: infra
tags:
- wf-fix
- 'wf-fix-fp:855c0317e5a2

  None'
created_at: '2026-07-02T23:34:32Z'
has_clean_result: false
origin_prompt: 'formal workflow-fix-candidate from #779 r9: exit-137 misread as OOM
  from shared-scope cgroup counters; PM deliberate kills leave no pre-kill marker'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a formal candidate block raised on task #779 round 9 (emitting agent: experiment-implementer).

## Goal

Add an exit-137 kill-source verification checklist to failure_patterns.md/gotchas.md and require PM deliberate kills to post a stop marker BEFORE the kill.

## Workflow gap

- **Bug observed:** three exit-137s (two grid runs + a 60GB-AS-capped diagnostic rerun) were diagnosed as kernel OOM from SHARED-login-scope cgroup counters (memory.peak=116.4GB, oom_kill=2) plus back-derived allocation arithmetic, when all three were deliberate PM-session SIGKILLs (mem-avail floor 47GB during the run; the oom_kill counter never incremented for these runs) — the epm:failure v5 mis-diagnosis dispatched crash-fix round 9 against a nonexistent bug.
- **Why it is a workflow gap:** no diagnosis protocol distinguishes SIGKILL sources on the shared VM (kernel-OOM vs earlyoom vs systemd-oomd vs deliberate kill), and a PM deliberate process-stop leaves no machine-readable marker BEFORE the kill, so exit-137 + plausible arithmetic reads as OOM to the next orchestrator.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ ## exit-137 on the shared VM: verify the kill source BEFORE failure_class
+ 1. cgroup memory.events oom_kill DELTA over the run window (shared login-scope
+    counters do NOT attribute; a non-incrementing counter RULES OOM OUT)
+ 2. MemAvailable floor over the run window (>20GB floor rules OOM out)
+ 3. journalctl -u earlyoom / -u systemd-oomd for a kill line at the death ts
+ 4. events.jsonl: any PM stop directive at/before the death ts?
+ PM sessions: post the deliberate-stop marker BEFORE kill -9, naming the PID.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/failure_patterns.md`, `.claude/rules/gotchas.md`
- Also consider: the PM-session spec (`.claude/agents/research-pm.md` / `.claude/skills/pm/SKILL.md`) for the stop-marker-before-kill duty; grep `grep -rln 'exit.137\|SIGKILL\|oom' .claude/ CLAUDE.md scripts/failure_classifier.py` and update every relevant surface.

## Constraints / invariants

- Workflow-surface only. workflow_lint passes; failure_classifier.py behavior unchanged unless the plan deliberately extends it (then tests).
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: .claude/skills/issue/failure_patterns.md, .claude/rules/gotchas.md
- fingerprint: 855c0317e5a2
None

<!-- verbatim candidate block from the r9 implementer report is recorded in epm:experiment-implementation v14 on task #779 -->
