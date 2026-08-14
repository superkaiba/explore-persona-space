---
name: pod-provision-wrapper-timeout-recovery
description: pod.py provision routinely outlives the 10-min Bash tool timeout mid-bootstrap; the pod is created and BILLING — recover, never re-provision (#1739 twice, 2026-08-06/08)
metadata:
  type: feedback
---

`pod.py provision` (create + full bootstrap incl. `uv sync` + flash-attn) routinely exceeds the
10-min Bash tool cap; the wrapper dies mid-bootstrap while the pod is already CREATED and BILLING.

**Why:** two incidents on #1739 within two days (2026-08-06 fairv2, 2026-08-08 pcsyco2), both killed
at bootstrap step [5/11]. A killed wrapper must never strand an unrecorded billing pod, and a
re-provision would mint a duplicate (the #1997 stopped-duplicate hijack class).

**How to apply:** run provision via `run_in_background=true` (or accept the timeout), then recover:
(1) `pod.py list-ephemeral --issue <N>` — confirm the pod exists/RUNNING; (2) post `epm:run-launched`
IMMEDIATELY with the `pod=<name>` token ("PROVISIONED, no workload launched yet" shape — billing pod
never live without a launch record); (3) `pod.py config --refresh-from-api <name>`; (4) re-run
`pod.py bootstrap <name>` to completion (idempotent); (5) proceed (branch checkout via
`git -C /workspace/...` inside the ssh payload — the repo-root branch guard blocks bare remote
`git checkout` shapes). Related: preflight's all-refs `git fetch origin` probe can time out on the
shallow 100+-ref clone while a scoped branch fetch takes ~1 s — verify branch-at-tip by SHA equality
instead of rerunning preflight hoping the probe passes.
