---
title: 'workflow-fix: document stale-blocked pass + GC test fixture'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7a743de994c8
- daily-auto-filed
created_at: '2026-07-09T06:58:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The #1021 stale_blocked_flag_pass
  (asw line ~13602) is absent from the module-docstring pass inventory (''Thirteen
  passes'', no stale-blocked mention in lines 1-460) and from .claude/rules/background-automation.md
  (grep empty), and tests/test_stalled_detector_and_gc.py::_populate_gc_targets drops
  only 4 prefixes — the stale-blocked-<N>.json GC target (asw :13846) is untested.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1021 (recursion-guarded workflow-fix session).

## Goal

Keep the watcher pass inventory current for the #1021 stale-blocked flag pass and close the GC-reap test-coverage gap for its state-file prefix.

## Workflow gap

- **Bug observed:** The #1021 stale_blocked_flag_pass (asw line ~13602) is absent from the module-docstring pass inventory ('Thirteen passes', no stale-blocked mention in lines 1-460) and from .claude/rules/background-automation.md (grep empty), and tests/test_stalled_detector_and_gc.py::_populate_gc_targets drops only 4 prefixes — the stale-blocked-<N>.json GC target (asw :13846) is untested.
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/autonomous_session_watch.py, .claude/rules/background-automation.md, tests/test_stalled_detector_and_gc.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
autonomous_session_watch.py docstring: add item N. **Stale-blocked flag pass (#1021).** ...
background-automation.md: add a capacity-retry-style paragraph for stale_blocked_flag_pass.
tests/test_stalled_detector_and_gc.py::_populate_gc_targets:
+   paths["stale_blocked"] = reg_dir / f"{STALE_BLOCKED_STATE_PREFIX}{issue}.json"
+   paths["stale_blocked"].write_text(json.dumps({"issue": issue}))
    ... assert sum(counts.values()) == 5
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py, .claude/rules/background-automation.md, tests/test_stalled_detector_and_gc.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, .claude/rules/background-automation.md, tests/test_stalled_detector_and_gc.py
- origin: parked candidate on task #1021 at 2026-07-04T23:05:33Z

Verbatim parked note:

> parked - running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug.md § Recursion guard. Source: code-review prose follow-ups (both reviewers, non-blocking): (1) scripts/autonomous_session_watch.py module docstring + .claude/rules/background-automation.md pass inventory do not yet list stale_blocked_flag_pass — a capacity-retry-style paragraph would keep the inventory current; (2) tests/test_stalled_detector_and_gc.py _populate_gc_targets not extended for the stale-blocked-<N>.json prefix (one fixture line + count bump) — GC reap of the new prefix untested. routed: parked (recursion guard); next non-guarded orchestrator pass may file.
