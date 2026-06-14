---
name: GCP router reconnects to a phase=done completed zombie instance
description: A smoke/prior GCP instance left RUNNING with eps/phase=done makes the router reconnect (no fresh provision, no workload dispatch); verify guest-attribute phase before posting epm:run-launched
type: feedback
---

When the router's launch outcome says `reason: "reconnect"` on the GCP
lane (`chosen_kind: "gcp"`, `epm:backend-selected` note carries
`outcome: "reconnected"` / `detail: "found existing live job/instance"`),
do NOT assume the workload was dispatched. `reconnect_or_none`
(`src/explore_persona_space/backends/gcp.py`) classifies ANY instance in
`{RUNNING, PROVISIONING, STAGING, STOPPING}` as live and returns a
reconnect handle WITHOUT re-dispatching `--workload-cmd`. A prior/smoke
instance that finished its workload, published guest attribute
`eps/phase=done`, but was NEVER deleted is wedged in status=RUNNING — a
completed zombie — and the router latches onto it.

**Why:** GCP instances are EPHEMERAL-by-DESIGN only when launched with
`--instance-termination-action=DELETE`; a smoke run (or any run whose
deletion didn't fire) can leave the canonical `eps-issue-<N>` name held by
a RUNNING-but-done VM. Posting `epm:run-launched` then makes the poller
read `phase=done` and falsely interpret the PRIOR run's artifact (e.g. a
smoke's `(1,28,3584)` tensor) as the full run's result — silently
shipping the wrong scope. Incident: task #634 (2026-06-14), full 275-role
extraction relaunch reconnected to the completed smoke instance
(attempt att-20260614-000605, phase=done, RUNNING 14 min, empty log, no
scratch, no repo, no workload proc).

**How to apply:** On a GCP-lane `reason: "reconnect"` outcome, BEFORE
posting `epm:run-launched`, probe the reconnected instance:
`gcloud compute instances get-guest-attributes <name> --query-path=eps/phase`
(thread `--configuration=eps-gcp --zone=<from handle sidecar>`). If
`phase=done` (or `failed`) AND there is no live workload process
(`ps aux | grep -i issue<N>` empty) AND the log is empty, the reconnect
is to a completed/dead zombie, NOT a live run — the full run was NOT
dispatched. Post `epm:failure v1` with `failure_class: infra`,
`reason: gcp-router-reconnected-to-completed-smoke-instance`, cite the
phase + uptime + empty-workload evidence, and let the orchestrator delete
the stale instance + re-dispatch (re-launch with the name free returns
`None` from reconnect → fresh provision + dispatch). Do NOT delete the
instance yourself (lifecycle is router-owned). A reconnect to a genuinely
live instance (phase=workload/eval/preflight, live process, log writing)
IS the intended idempotency path — proceed normally there.
