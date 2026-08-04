---
title: 'workflow-fix: uniquify pod_disk_guard probe filename + gotchas entry (probe
  race)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b2dfed137d5b
created_at: '2026-08-01T13:37:46Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate failure-lesson on #1979 + code-review r2 sibling
  findings'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gotcha_candidate failure-lesson raised on task #1979 (emitting agent: experiment-implementer; code-reviewer r2 surfaced the siblings).

## Goal

Uniquify the remaining fixed-filename disk-probe sites on the workflow surface (`scripts/pod_disk_guard.py:84`) per the #1979 fix pattern, and document the fixed-filename-probe concurrency race as a gotchas.md entry.

## Workflow gap

- **Bug observed:** #1979 fellows job 16686 — `_probe_writable_bytes`'s FIXED probe filename raced 8 concurrent workers on a shared FS (sibling unlink invalidates an open fd mid-posix_fallocate → OSError EBADF, outside handled errno sets → worker rc=1). Fixed at source (commit 11a6c405cd: pid+uuid suffix). Code-review r2 found the same fixed-filename pattern surviving at `scripts/pod_disk_guard.py:84` (workflow surface) and `scripts/issue1481_marker.py:1482` (experiment code — OUT of wf-fix scope, listed for context only).
- **Why it is a workflow gap:** pod_disk_guard runs fleet-wide against pod filesystems where concurrent invocation is plausible; the trap class is documented nowhere in `.claude/rules/gotchas.md`.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "preflight_disk_probe\|posix_fallocate" scripts/pod_disk_guard.py` → hits at the cited region (2026-08-01); per-target: pod_disk_guard.py present; gotchas.md absence-of-entry claim — `grep -c "fallocate" .claude/rules/gotchas.md` run at compose time

## Proposed change (candidate diff sketch — refine in planning)

```
+ pod_disk_guard.py: probe filename gains a per-invocation pid+uuid suffix (mirror preflight.py::_probe_writable_bytes, 11a6c405cd); cleanup self-scoped.
+ gotchas.md: new entry — fixed-filename create/fallocate/unlink probes race concurrent workers on shared FS (EBADF outside handled errno sets); uniquify per invocation at the source helper.
```

## Scope / surfaces

- Primary target: `scripts/pod_disk_guard.py, .claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'preflight_disk_probe\|posix_fallocate' .claude/ CLAUDE.md scripts/ src/explore_persona_space/orchestrate/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/pod_disk_guard.py, .claude/rules/gotchas.md
- fingerprint: b2dfed137d5b
