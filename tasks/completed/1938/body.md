---
title: 'workflow-fix: gotchas.md entry — VAST /workspace EBADF from posix_fallocate'
kind: infra
tags:
- wf-fix
- wf-fix-fp:aff01d73b6e2
created_at: '2026-07-31T13:47:14Z'
has_clean_result: false
origin_prompt: 'orchestrator-surfaced gotcha candidate from #1902 crash 2 (fellows
  job 16139): VAST EBADF from posix_fallocate; add gotchas.md entry'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the orchestrator's own crash-2 diagnosis on task #1902 (emitting agent: orchestrator, /issue 1902 session).

## Goal

Add a `.claude/rules/gotchas.md` entry documenting that VAST/NFS-class mounts (fellows cluster `/workspace`) surface `OSError EBADF` from `os.posix_fallocate` on a just-opened valid fd, so fallocate-based disk/quota probes must treat EBADF like EOPNOTSUPP (degrade to statvfs fallback).

## Workflow gap

- **Bug observed:** fellows job 16139 (task #1902 launch 2) crashed P1 rc=1 at `preflight._probe_writable_bytes` on `OSError: [Errno 9] Bad file descriptor` from `posix_fallocate` against VAST `/workspace`; the probe tolerated only EOPNOTSUPP/ENOSYS/EINVAL and re-raised EBADF.
- **Why it is a workflow gap:** gotchas.md documents the sibling filesystem probe traps (MooseFS EDQUOT, FUSE wedges) but has no VAST-EBADF entry — any future fallocate-based probe (or a reader of preflight code on the fellows lane) re-hits the trap undiagnosed; the fellows lane is new (#1899) and #1902 is its first heavy user.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'EBADF' .claude/rules/gotchas.md` → 0 hits (absence-of-guard claim — the 0-hit result IS the evidence; the one `posix_fallocate` hit at line 54 is the MooseFS EDQUOT entry, whose context read confirms it does not cover EBADF) (2026-07-31). Library fix already landed: `_probe_writable_bytes` tolerates EBADF as of commit 55f96ad609b59394a39a989a6df50435f4336196 on `origin/issue-1902` (lands on main at #1902's Step 10d merge) + pin test `tests/test_preflight_disk.py::test_probe_ebadf_falls_back`.

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **VAST/NFS-class mounts surface EBADF from `posix_fallocate` on a just-opened VALID fd
+   (fellows cluster `/workspace`) — fallocate-based disk/quota probes must treat EBADF like
+   EOPNOTSUPP/ENOSYS/EINVAL (degrade to the statvfs fallback), never re-raise.** Distinct from
+   the MooseFS EDQUOT entry (that is a REAL quota refusal the probe exists to catch); EBADF here
+   is a filesystem-layer artifact — the fd was opened successfully two lines above. Fixed in
+   `preflight._probe_writable_bytes` (#1902, commit 55f96ad609...); pin test
+   `tests/test_preflight_disk.py::test_probe_ebadf_falls_back`. (Incident #1902 job 16139,
+   2026-07-31: P1 pilot died rc=1 in the headroom gate seconds into the fellows-lane launch.)
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'EBADF' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan. Consider whether the LESSONS.md gotchas trigger row needs no change (it already covers "stage VM-local data" / probe traps).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: aff01d73b6e2

Surfaced prose (orchestrator's own observation, /issue 1902 crash-fix cycle 2): "VAST/NFS-class mounts (fellows cluster /workspace) surface OSError EBADF from os.posix_fallocate on a just-opened VALID fd — a filesystem-layer artifact, not a caller bug. Any fallocate-based disk/quota probe must treat EBADF like EOPNOTSUPP/ENOSYS/EINVAL (degrade to the statvfs fallback), or the probe itself crashes the run on the fellows lane. gotcha_candidate: yes."
