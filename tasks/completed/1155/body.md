---
title: 'workflow-fix: pod-safety stop-failed marker + fix guard-test'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7e39b7bd7ea7
- daily-auto-filed
created_at: '2026-07-09T06:57:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): (a) _apply_pod_safety_action
  posts the durable auto-stop marker only when _stop_pod returns True — a persistent
  RunPod stop-API failure retries every tick with stderr-only visibility; (b) test_cpu_guard_never_kills''s
  grep span ends at the first ''def _status_class'' occurrence, which sits inside
  the block''s own header comment, guarding ~10 comment lines instead of the ~1500-line
  pass.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #980 (park_form: recursion-guard).

## Goal

Post a durable once-per-episode stop-failed marker on _apply_pod_safety_action's _stop_pod-False branch (episode stays retryable, + a _process_pod test), and re-anchor test_cpu_guard_never_kills's end sentinel so the real guard code block is scanned.

## Workflow gap

- **Bug observed:** (a) _apply_pod_safety_action posts the durable auto-stop marker only when _stop_pod returns True — a persistent RunPod stop-API failure retries every tick with stderr-only visibility; (b) test_cpu_guard_never_kills's grep span ends at the first 'def _status_class' occurrence, which sits inside the block's own header comment, guarding ~10 comment lines instead of the ~1500-line pass.
- **Why it is a workflow gap:** (a) a silently-retrying failed auto-stop is an unbounded billing leak with no task-level evidence; (b) the never-kills invariant test currently scans none of the code it exists to pin.
- **Confidence (emitter):** medium (reconciler standing recommendation + Codex alternatives Must-Fix; fact-checker item-10 nuance — #980)

## Proposed change (candidate diff sketch — refine in planning)

(a) add `else: _post_progress_marker(issue, f"{_STOP_FAILED_SENTINEL} pod-safety auto-stop FAILED (pod_id=...)", ..., label="stop-failed")` deduped once per episode via state; (b) end = src.index('def _status_class', src.index('_env_float')) or a dedicated end-sentinel line placed AFTER the guard code.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py, tests/test_cpu_guard_pass.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, tests/test_cpu_guard_pass.py
- origin: parked candidate on task #980 at 2026-07-04T10:35:55Z

parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target recursion guard (.claude/rules/workflow-fix-on-bug.md § Recursion guard) — NOT auto-routed. source: reconciler standing recommendation (Phase 2 round 1) + Codex alternatives Must-Fix (adjudicated non-blocking for #980). target_file: scripts/autonomous_session_watch.py. bug_observed: _apply_pod_safety_action posts the durable auto-stop marker only when _stop_pod returns True; on a persistent RunPod stop-API failure the episode retries each tick with stderr-only visibility — no durable task-level marker/escalation survives (shared by all POD_SAFETY_AUTO_STOP statuses, pre-existing on the DONE arm). proposed_change: add a durable once-per-episode stop-failed marker/escalation on the _stop_pod(...) is False branch (episode stays retryable), + a _process_pod test with _stop_pod=False asserting the marker posts and state is preserved. confidence: medium. related_task: #980. Second parked candidate (same guard), source: fact-checker item-10 nuance: tests/test_cpu_guard_pass.py's grep span ends at the FIRST 'def _status_class' occurrence — inside its own sentinel comment (~L2641), not the real def (~L3637) — so the guarded block is ~10 lines instead of ~1000 (latent test-coverage bug). target_file: tests/test_cpu_guard_pass.py. confidence: medium.
