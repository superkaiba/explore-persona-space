---
title: 'workflow-fix: exclude deliberate-stop notes from stage-skip liveness'
kind: infra
tags:
- wf-fix
- wf-fix-fp:88d7e4ab1a95
created_at: '2026-07-03T21:43:42Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: src/explore_persona_space/task_workflow.py\n\
  bug_observed: a deliberate-stop epm:progress marker posted by spawn_session.py stop\
  \ refreshed the in-flight window of the stage the stopped session owned, making\
  \ the replacement session's re-dispatch check return skip on provably dead work\n\
  why_workflow_gap: stage_dispatch_should_skip counts every non-breadcrumb epm:progress\
  \ as liveness, but a deliberate-stop note is the death record of the stage's owner\
  \ — counting it inverts the guard exactly when a replacement session needs a correct\
  \ answer\nproposed_change: exclude anti-liveness progress notes (deliberate-stop\
  \ / spawn_session-stop) from stage_dispatch_should_skip's freshness-refresh set\n\
  diff_sketch: |\n  in the refresher loop of stage_dispatch_should_skip:\n  + if note.startswith(\"\
  deliberate-stop\") or event.get(\"by\") == \"spawn_session-stop\":\n  +     continue\n\
  \  plus a pinned regression test (breadcrumb + later deliberate-stop note -> DISPATCH\
  \ once the breadcrumb itself is stale)\nconfidence: high\nrelated_task: #810\n<!--\
  \ /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #810 (emitting agent: orchestrator, replacement-session resume).

## Goal

Exclude anti-liveness progress notes (deliberate-stop / spawn_session-stop) from `stage_dispatch_should_skip`'s freshness-refresh set so a replacement session is not told a provably dead stage is "in flight".

## Workflow gap

- **Bug observed:** A `deliberate-stop` `epm:progress` marker posted by `spawn_session.py stop` (by=`spawn_session-stop`) refreshed the in-flight freshness window of the stage the STOPPED session owned; the replacement session's mechanical pre-dispatch check (`stage_dispatch_should_skip`) returned `skip: ... refreshed by epm:progress at <stop-ts>` on work that was provably dead (killed subagent, zero commits, zero markers since dispatch). Live incident: #810, 2026-07-03 21:41Z — the round-2 `followup-implementing-he` re-dispatch had to override the mechanical skip by hand.
- **Why it is a workflow gap:** `stage_dispatch_should_skip` treats EVERY non-breadcrumb `epm:progress` as liveness. A deliberate-stop note is the opposite of liveness — it is the death record of the stage's owner — so counting it inverts the guard's semantics exactly when a crash-recovery / operator-replacement session needs a correct answer. Worst case it delays every replacement session's takeover by up to the full window (15–30 min) per stage, or (with repeated watcher stop/respawn churn) indefinitely re-extends it.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

In `stage_dispatch_should_skip` (src/explore_persona_space/task_workflow.py, refresher loop ~line 1428):

```
  if kind == "epm:progress":
      note = (event.get("note", "") or "").lstrip()
      if note.startswith("stage-dispatch "):
          continue
+     # Anti-liveness: a deliberate session stop is the death record of the
+     # stage's owner, never evidence the stage is alive (#810 2026-07-03).
+     if note.startswith("deliberate-stop") or event.get("by") == "spawn_session-stop":
+         continue
```

Plus a pinned regression test in tests/ (a breadcrumb followed by a deliberate-stop
progress note older-than-window breadcrumb must return None/DISPATCH). The planner
should also consider whether other non-stage watcher notes (e.g. notes beginning
`[autonomous_session_watch:`) belong in the same exclusion set — they are fleet
telemetry, not stage liveness — but the deliberate-stop case is the concrete,
incident-backed minimum.

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rn 'deliberate-stop' src/explore_persona_space/ scripts/ .claude/`) and update every hit;
  list them in the plan. Also update the docstring's liveness definition and
  `.claude/skills/issue/SKILL.md`'s Step 9 entry-guard prose if the semantics text needs the carve-out named.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: 88d7e4ab1a95

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py
bug_observed: a deliberate-stop epm:progress marker posted by spawn_session.py stop refreshed the in-flight window of the stage the stopped session owned, making the replacement session's re-dispatch check return skip on provably dead work
why_workflow_gap: stage_dispatch_should_skip counts every non-breadcrumb epm:progress as liveness, but a deliberate-stop note is the death record of the stage's owner — counting it inverts the guard exactly when a replacement session needs a correct answer
proposed_change: exclude anti-liveness progress notes (deliberate-stop / spawn_session-stop) from stage_dispatch_should_skip's freshness-refresh set
diff_sketch: |
  in the refresher loop of stage_dispatch_should_skip:
  + if note.startswith("deliberate-stop") or event.get("by") == "spawn_session-stop":
  +     continue
  plus a pinned regression test (breadcrumb + later deliberate-stop note -> DISPATCH once the breadcrumb itself is stale)
confidence: high
related_task: #810
<!-- /workflow-fix-candidate -->
