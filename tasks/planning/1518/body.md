---
title: 'workflow-fix: provision refusals to stderr so failure markers name the cause'
kind: infra
tags:
- wf-fix
- wf-fix-fp:aa07fd095b30
created_at: '2026-07-18T18:37:34Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 1481 recovery: pod_lifecycle stdout
  refusals invisible in PodLifecycleProcessError stderr-tail capture'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1481 (emitting agent: /issue orchestrator).

## Goal

Route `pod_lifecycle.py provision` refusal messages to stderr before `sys.exit(1)` (or widen `PodLifecycleProcessError` to carry a stdout tail) so router failure markers name the refusal cause.

## Workflow gap

- **Bug observed:** `pod_lifecycle provision` refusals ("Pod already exists", disk-cap) print to stdout; `PodLifecycleProcessError` carries only the stderr tail, so the runpod_fallback_failed marker showed an unexplained exit 1.
- **Why it is a workflow gap:** the router's `epm:backend-selected` failure detail (#1465 stderr-tail capture) is the durable diagnostic surface; a stdout-only refusal makes a self-explanatory failure read as unexplained — on #1481 (2026-07-18 17:52Z) the `no_compute_available` terminal hid a "Pod pod-1481 already exists" refusal and cost a live diagnosis round while the orphan 8xH100 billed $32/hr.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "Pod {candidate} already exists" scripts/pod_lifecycle.py` → 1 hit (line ~1903, bare `print(...)` + `sys.exit(1)`, no `file=sys.stderr`); `grep -n "stderr tail" src/explore_persona_space/backends/runpod.py` → 2 hits (lines 106, 130 — `PodLifecycleProcessError` composes the stderr tail ONLY); `grep -c "print(" scripts/pod_lifecycle.py` → 84 bare print sites, several on exit-1 refusal paths (2026-07-18)

## Proposed change (candidate diff sketch — refine in planning)

```
- print(f"Pod {candidate} already exists ...")
- sys.exit(1)
+ print(f"Pod {candidate} already exists ...", file=sys.stderr)
+ sys.exit(1)
```
(sweep every refusal-path bare print before a nonzero exit in pod_lifecycle.py; optionally also append a bounded stdout tail to PodLifecycleProcessError in backends/runpod.py)

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py`, `src/explore_persona_space/backends/runpod.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/pod_lifecycle.py, src/explore_persona_space/backends/runpod.py
- fingerprint: aa07fd095b30

Surfaced prose (verbatim): pod_lifecycle provision refusals (Pod already exists, disk-cap) print to stdout; PodLifecycleProcessError carries only the stderr tail, so the runpod_fallback_failed marker showed an unexplained exit 1 on #1481.
