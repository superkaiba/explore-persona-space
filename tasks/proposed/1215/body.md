---
title: 'daily-held: daemon-independent orphan-wrapper sweep'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-09T07:01:43Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 3): ALL session-reaper passes
  are daemon-gated (zombie_wrapper_pass docstring: ''Daemon-gated like the respawn
  pass''; wrapper pids come from the daemon /list) — a Happy wrapper/launcher ABSENT
  from /list (incl. the 2026-07-01 incident''s 31 zombie wrappers and a 54-day init-parented
  launcher) is invisible to every pass and never reaped.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #818 (recursion-guarded workflow-fix session).

## Goal

Catch the wrapper-orphan class the #818 fix deliberately descoped: processes the Happy daemon no longer tracks. Route 3: the reap action is direct process signalling of sessions the daemon cannot see (destructive; false-positive risk overlaps the #1039 non-EPS policy question) — needs a human call on escalate-only vs reap.

## Workflow gap

- **Bug observed:** ALL session-reaper passes are daemon-gated (zombie_wrapper_pass docstring: 'Daemon-gated like the respawn pass'; wrapper pids come from the daemon /list) — a Happy wrapper/launcher ABSENT from /list (incl. the 2026-07-01 incident's 31 zombie wrappers and a 54-day init-parented launcher) is invisible to every pass and never reaped.
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/autonomous_session_watch.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
# daemon-independent block (next to the CPU-guard pass, asw item 12):
for proc in scan_proc_for_happy_wrappers():
    if proc.sid not in daemon_list_sids and no_claude_descendant(proc) and cpu_near_zero(proc):
        escalate(...)  # phase 1: sidecar + alert only
        # phase 2 (post-greenlight): SIGTERM under the zombie-pass conservative guards
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: parked candidate on task #818 at 2026-07-02T00:55:46Z

Verbatim parked note:

> parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). TWO candidates for the next orchestrator/human pass:

(1) source: prose-followup (alternatives critic + plan v4 §6 descope). target_file: scripts/autonomous_session_watch.py. bug_observed: the 2026-07-01 incident's 31 zombie wrappers were most likely daemon-/list dropouts — ALL session-reaper passes are daemon-gated (_live_children), so a wrapper absent from the Happy daemon /list (incl. the 54-day init-parented launcher) is invisible to every pass and never reaped. proposed_change: a daemon-INDEPENDENT /proc-scan sweep for orphaned Happy wrappers/launchers (no claude descendant, ~0 CPU, not in /list), escalate-only or reap under the existing conservative guards. confidence: high (this is the class the #818 fix deliberately descopes; would have caught the actual incident).

(2) source: test-verdict observation. target_file: scripts/verify_task_body.py (audit-claim HF checks) — 2 tests failing on main (test_verify_task_body_audit_claim.py::test_denial_and_hf_genuinely_missing_passes + ::test_hf_http_error_is_unverified_not_fail); plus the standing workflow_lint dotenv failures rooted in scripts/issue744_dump_and_stream.py:87 + scripts/issue778_upload.py:29 (experiment entrypoints — OUT of workflow-fix scope, route to an implementer follow-up). bug_observed: the no-flags workflow_lint default run + 2 audit-claim tests fail on a clean main checkout, so every Step 9c touched-subset gate that includes them reports pre-existing failures. confidence: high (reproduced on main at HEAD, 2026-07-02).
