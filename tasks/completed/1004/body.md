---
title: 'workflow-fix: gcp.py workload-cmd rc-masking (false phase=done on && chains)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f38668c61823
created_at: '2026-07-04T10:01:00Z'
has_clean_result: false
origin_prompt: 'orchestrator crash diagnosis on #952 run 1 (see body Provenance)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #952 (emitting agent: orchestrator, run-1 crash diagnosis).

## Goal

Render `spec.workload_cmd` inside an rc-preserving single-command wrapper (e.g. `bash -o pipefail -ec <shlex.quote(cmd)>`) in the GCE startup script so a multi-command workload chain cannot mask a non-zero rc into a false `phase=done`.

## Workflow gap

- **Bug observed:** A `--workload-cmd` containing a `&&` chain crashed on its first command (unhandled Python exception, rc!=0) yet the startup script took the clean-exit branch and published `phase=done` + armed the done-grace self-poweroff — the `set -e` + `&&` rc-propagation weakness documented at `gcp.py:1700-1712`; #750 patched only the OOM subclass via the cgroup oom_kill counter (issue 952 run 1, eps-issue-952, 2026-07-04 09:44Z; the poller advanced on status=done over a zero-artifact crash).
- **Why it is a workflow gap:** `spec.workload_cmd` is spliced as a bare line into the `set -euo pipefail` script (gcp.py ~line 1250, "Run the workload (custom workload_cmd)"); the rc-masking class is documented in-file as a known weakness but only the OOM subclass has a guard, so ANY orchestrator composing a compound workload command re-hits the false-done. The false phase=done also bypasses crash-persist diagnostics (clean-exit path uploads nothing).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
- "_eps_phase workload",
- "touch /tmp/eps-workload-start",
- spec.workload_cmd,
+ "_eps_phase workload",
+ "touch /tmp/eps-workload-start",
+ f"bash -o pipefail -ec {shlex.quote(spec.workload_cmd)}",
```
(plus: consider a rendered comment documenting the invariant, and a unit test in tests/test_gcp_backend.py rendering a `cmd1 && cmd2` workload and asserting the wrapper form; verify the detached-workload pid-file wait loop still sees children of the inner bash.)

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep before editing: `grep -n "spec.workload_cmd" src/explore_persona_space/backends/gcp.py` (all splice sites) + `tests/test_gcp_backend.py` render tests.

## Constraints / invariants

- Workflow-surface only. Existing render tests must pass; add a regression test for the compound-command rc propagation.
- The #750 OOM guard and the detached-workload wait loop must keep working (the inner bash's children still write pid files under /workspace/logs).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: f38668c61823
