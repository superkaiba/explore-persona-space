---
name: preflight fetch-timeout false negative on slow-network pod
description: orchestrate.preflight's internal git fetch hits its own tight timeout on fresh/slow-network pods and emits "branch not up to date" even when HEAD == origin; verify directly before posting epm:failure
type: feedback
---

`orchestrate.preflight` can return `ok: false` with the SOLE error
`git fetch origin failed (timeout) — cannot verify branch issue-<N> is up
to date with origin/issue-<N>` on a slow-network pod (typical on a fresh
RunPod the first time it touches the GitHub remote). The error is the
preflight's INTERNAL fetch hitting its own tight timeout, NOT actual
branch divergence — re-running the fetch directly with a relaxed timeout
succeeds, and `git rev-parse HEAD` matches `git rev-parse origin/issue-<N>`.

**Why:** the workflow's "never silently ignore preflight failures" rule
exists to catch real problems (OOM, ENOSPC, gated repos), but treating
this transient as a real branch-staleness error costs a relaunch cycle
and a false `epm:failure v1`.

**How to apply (experimenter Step 6c on a fresh pod):**
1. If preflight's only error matches `git fetch origin failed (timeout)`,
   re-verify manually:
   ```bash
   ssh pod-<N> 'cd /workspace/explore-persona-space && \
     timeout 60 git fetch origin issue-<N> && \
     echo LOCAL=$(git rev-parse HEAD) ORIGIN=$(git rev-parse origin/issue-<N>)'
   ```
2. LOCAL == ORIGIN → treat preflight as effectively PASS; record the
   transient in your launch marker's `note` field and proceed.
3. LOCAL != ORIGIN → real divergence, post `epm:failure v1
   failure_class: infra reason: branch-stale-on-pod` and surface for
   user resolution.

Only the `git fetch timeout` error gets this carve-out; any other
preflight error (GPUs, disk, HF reachability, env-sync) is a real
failure.

Closed regressions: task #664 r8 launch (2026-06-27) — false-negative
preflight on the fresh recovery pod-664 nearly bounced a healthy launch.
