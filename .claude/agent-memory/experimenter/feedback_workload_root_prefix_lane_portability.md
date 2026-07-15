---
name: workload-root-prefix-lane-portability
description: Never prefix a workload command with REPO_ROOT="$WORKLOAD_ROOT" — GCE-only var, dies unbound on the RunPod failover lane; self-defaulting dispatch scripts launch bare
type: feedback
---

Never launch a workload with a `REPO_ROOT="$WORKLOAD_ROOT"` prefix. `WORKLOAD_ROOT`
is exported only by the GCE startup script; the router's GCP→RunPod failover
(#783 queue-timeout path) re-runs the SAME workload command on the RunPod lane,
where the launcher runs under `set -u` and dies at t+0 with
`WORKLOAD_ROOT: unbound variable` (incident #1336, 2026-07-15: pod-1336 provisioned,
launch dead, pod billing until manual re-drive).

**Why:** dispatch scripts follow the self-defaulting convention
(`REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"` + a PWD fallback),
which resolves correctly on BOTH lanes — the prefix is redundant on GCE and fatal
on RunPod.

**How to apply:** launch self-defaulting dispatch scripts BARE
(`bash scripts/issueN_dispatch.sh all`). If a script genuinely lacks the
self-default, use the lane-portable form
`REPO_ROOT="${WORKLOAD_ROOT:-/workspace/explore-persona-space}"` — never the bare
`$WORKLOAD_ROOT` expansion. When re-driving a failed launch, check the wrapper's
first log lines for `unbound variable` before suspecting the pod.
