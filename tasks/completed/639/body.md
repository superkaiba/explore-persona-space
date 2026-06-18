---
title: Wire GCP janitor (audit_stale_gcp_vms) to a recurring cron
kind: infra
tags: []
created_at: '2026-06-14T00:53:02Z'
has_clean_result: false
origin_prompt: 'Surfaced as architectural follow-up by #634''s janitor-predicate workflow-improver
  (commit 8ee393a60): audit_stale_gcp_vms exists with a complete predicate but is
  never cron-invoked. Cron + CLI subcommand are public-contract additions warranting
  Thomas greenlight, not background auto-fix.'
---
---
title: "Wire scripts/backends/gcp.py audit_stale_gcp_vms to a recurring cron"
kind: infra
priority: medium
tags: [gcp, janitor, cron, autonomous-watcher]
---

## Goal

Wire `src/explore_persona_space/backends/gcp.py::audit_stale_gcp_vms` to a recurring cron entrypoint so the GCP janitor actually fires. CLAUDE.md ("Pods — Crons" section) describes "A janitor sweeps stale `eps-issue-*` instances" — but as of #634 (2026-06-13) the function exists with a complete predicate (24h age backstop + 10-min terminal-phase RUNNING prompt reap, the latter landed in `8ee393a60` on the issue-634 branch) yet is NOT cron-invoked anywhere. The RunPod analogue is wired via `cron_pod_audit.sh → pod.py audit-stale`; the GCP analogue is dormant.

## Background

- The reaper function: `backends/gcp.py::audit_stale_gcp_vms` (live in main + extended on issue-634).
- Its docstring already states "Cron wiring is the orchestrator's responsibility."
- `cron_pod_audit.sh` only runs RunPod's `pod.py audit-stale`. There is no equivalent for GCP.
- The autonomous-session watcher (`scripts/autonomous_session_watch.py`, every 10 min) does NOT call this either; it reconciles RunPod pods only.

## Surface to write

Two coupled changes, both architectural / public-contract:

1. **CLI entrypoint** — add a `gcp audit-stale` subcommand to one of the existing scripts (likely `scripts/dispatch_issue.py` or a new `scripts/gcp_audit.py`) that calls `backends.gcp.audit_stale_gcp_vms(...)` with the production-injected dependencies (gcloud config = `eps-gcp`, runner = `default_gcloud_runner`, etc.) and prints the per-instance record list as JSON (matching the `pod.py audit-stale --json` shape).
2. **Cron entrypoint** — either (a) extend `scripts/cron_pod_audit.sh` to call the new GCP subcommand after the existing RunPod sweep, OR (b) add a new `scripts/cron_gcp_audit.sh` + crontab entry. Frequency matches the RunPod analogue (daily at 09:37 PT). Document in CLAUDE.md "Crons" section + `.claude/rules/background-automation.md`.

## Acceptance criteria

1. `gcloud compute instances list --configuration=eps-gcp` (or the API equivalent the function uses) IS called once per cron fire.
2. A test that runs the new CLI subcommand against a mocked gcloud runner reaping a known fixture (one EXITED + one RUNNING-terminal-phase + one healthy RUNNING) returns the right per-instance verdicts.
3. CLAUDE.md "Pods (Ephemeral Lifecycle + CLI + SSH) — Crons" section is updated to name the new entrypoint + frequency.
4. `.claude/rules/background-automation.md` documents the predicate (24h age backstop + 10-min terminal-phase prompt reap), the env-var overrides if any, and the env-var override for `EPS_GCP_JANITOR_*` if you add one.
5. One end-to-end smoke: trigger the cron entrypoint by hand on the VM, confirm it queries gcloud, then revert (don't actually leave a delete probe on real instances).

## Out of scope

- The reaper function itself is correct — do NOT modify `audit_stale_gcp_vms` behavior in this task.
- A `GcpBackend.audit_stale()` method (alternative API surface) is a design choice — pick whichever fits the existing `dispatch_issue.py` subcommand pattern best.

## Provenance

Surfaced as a workflow-fix-candidate follow-up by the janitor-predicate workflow-improver (commit `8ee393a60` on `issue-634`), routed by the orchestrator as `kind: infra` because the cron + CLI subcommand are public-contract additions per `.claude/rules/workflow-fix-on-bug.md` (architectural / public-contract changes warrant explicit user greenlight, not background auto-fix).

Predecessor commits:
- `77af3b006` — reconnect_or_none refuses reconnect to RUNNING+terminal-phase
- `03b57e34a` — reconnect_or_none also deletes the zombie before returning None
- `8ee393a60` — audit_stale_gcp_vms predicate extension (this task wires the cron)

All three on `origin/issue-634` HEAD as of #634's full run.
