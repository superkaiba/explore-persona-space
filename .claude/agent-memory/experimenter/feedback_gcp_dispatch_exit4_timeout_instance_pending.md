---
name: dispatch_issue.py launch exit-4 TimeoutExpired ≠ launch failure — gcloud-create's 300s subprocess cap fires on FLEX_START queueing
description: Exit-4 with subprocess.TimeoutExpired on dispatch_issue.py launch can hide a successful server-side create — FLEX_START stays PENDING for preemptible capacity past the 300s subprocess cap; check `gcloud instances list` BEFORE treating as a failure or relaunching (would race/duplicate)
type: feedback
---

`scripts/dispatch_issue.py launch` runs `gcloud compute instances create` under a
local subprocess with a 300s cap. On the GCP FLEX_START rung, the instance
legitimately stays PENDING (queued for preemptible A100-80 / A100-40 / etc.
capacity) well past 300s — so the LOCAL dispatch crashes with
`subprocess.TimeoutExpired` and exits 4, while the create OFTEN SUCCEEDS
server-side.

**Why it is a known clean-exit kind:** the dispatch CLI documents exit 75
(EX_TEMPFAIL — "still waiting, re-run") and exit 3 (terminal failure) as the
two legitimate non-zero kinds. Exit 4 is the local subprocess timeout — it is
NOT in the documented contract and predictably mis-routes as "launch failed".

**How to detect:** if the dispatch exited 4 with a `subprocess.TimeoutExpired`
trace pointing at `gcloud_run` / `subprocess.run(..., timeout=300, ...)`, run
`gcloud compute instances list --filter=name=eps-issue-<N>` BEFORE posting any
failure or relaunching. If the instance exists (PENDING / PROVISIONING /
RUNNING), the create succeeded server-side and the launch is in flight.

**How to recover (instance present):** post `epm:run-launched` with the live
instance's fields (zone / machine / `provisioning=FLEX_START` / job_id from
`instances list`, and the `attempt_id` from the handle sidecar
`.claude/cache/issue-<N>-handle.json`), flag the create-timeout + PENDING
state in the marker note, and let the orchestrator's bg-Bash poll chain
(`backend_poll.py --issue <N>`) follow it through to PROVISIONING / RUNNING.
Do NOT relaunch — a second `dispatch_issue.py launch` races the live PENDING
instance.

**How to recover (instance absent):** treat as a genuine launch failure;
re-dispatch normally.

**Why this is the GCP-lane analogue of the SSH-timeout rule:** the existing
"SSH timeout ≠ child dead — pgrep before relaunch" pattern is the same shape
on a different surface — a local probe timed out while the remote work is
healthy. The fix is symmetric: check the canonical remote state
(`gcloud instances list` here, `pgrep` there) before treating the timeout as
failure.

**Renderer / dispatcher follow-up (open):** `scripts/dispatch_issue.py`'s
gcloud-create timeout should either be raised on FLEX_START rungs (the rung
explicitly queues), or the exit-4 path should detect the live instance itself
and convert to an exit-75 still-waiting signal. Tracked as a workflow-fix
infra task.

Burned at #658 round 1 (2026-06-29): the experimenter spawn correctly held
its turn after dispatching, checked `gcloud instances list`, confirmed a
single live `eps-issue-658` at PENDING, posted `epm:run-launched v11`, and
did NOT relaunch. Root cause confirmed by the experimenter; lesson captured
verbatim via the orchestrator's failure-lesson hook.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [GCP-lane: dispatch exit-4 TimeoutExpired ≠ launch failure](feedback_gcp_dispatch_exit4_timeout_instance_pending.md) — `dispatch_issue.py launch` 300s subprocess cap fires on FLEX_START queueing while the create succeeds server-side; `gcloud instances list` BEFORE treating as failure or relaunching (would race the PENDING instance). GCP-lane analogue of "SSH timeout ≠ child dead" (#658)
