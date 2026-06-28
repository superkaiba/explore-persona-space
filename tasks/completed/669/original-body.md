---
title: 'Detect hung-but-RUNNING GCP VM (frozen non-terminal eps/phase + drain-timeout)
  -> terminal_workload_wedged -> #659 RunPod failover; + in-VM network watchdog'
kind: infra
tags: []
created_at: '2026-06-25T18:41:07Z'
has_clean_result: false
origin_prompt: 'Root-caused from the #667 attempt-1 GCP networking wedge (deep investigation
  2026-06-25): a hung-but-RUNNING GCP VM with frozen non-terminal eps/phase is classified
  running forever; neither sync (#658) nor wired async (#659) failover fires; only
  a manual --backend runpod pivot recovered it. Wire poller wedge-detection + an in-VM
  watchdog.'
---
## Problem

A GCP workload VM that **hangs with dead guest networking** (vs crashing) ends in
a **RUNNING-but-dead** state that the poller classifies `running` **forever**, so
neither the synchronous (#658) nor the wired async (#659) GCP→RunPod failover
fires — only a manual `--backend runpod` pivot recovers it. This is the 3rd
distinct "GCP RUNNING-zombie" mode (after #491 SIGPIPE / #640 guestTerminate).

## Evidence (#667, 2026-06-25)

- GCP `eps-issue-667` (a2-highgpu-1g, us-central1-b), workload started 11:01:50Z.
- Serial tail at **11:34:06Z: `ens5: Could not set DHCPv4 address: Connection
  timed out`** — guest NIC lost its DHCP lease. After: `describe` = RUNNING,
  `gcloud ssh` hangs (bounded by `default_gcloud_runner` 300s, `gcp.py:1819`),
  HF unreachable (0 artifacts; no `issue667_partial/` persisted).
- Workload **never exited** → GCE EXIT-trap never ran → `eps/phase` stayed frozen
  at the **non-terminal** `workload`. `gcp.poll()` `if phase:` branch
  (`gcp.py:3058`) returns `status="running"` every tick.
- #659's async predicate `_is_gcp_async_workload_failure`
  (`scripts/backend_poll.py:202-206`) requires `current_phase ==
  "terminal_workload_failed"`, only produced (`gcp.py:3042-3057`) when
  `eps/phase == "failed"` + the `workload_started` sentinel — i.e. the workload
  **exited non-zero**. A hung workload never reaches it → failover never
  evaluated. (`test_gcp_backend.py:2560` asserts `eps/phase=workload` → running;
  there is no test for the wedged case.)
- Recovery happened only via a manual `epm:strategy-pivot` (orchestrator applied
  `--backend runpod`, `reason="override"`), which also (incorrectly) cited the
  failover as "a pending wire-up" — see the separate doc-fix.

## Fix 1 (primary, load-bearing) — detect the wedge + route through #659

In `src/explore_persona_space/backends/gcp.py::poll`, the `if phase:` running
branch should escalate to a terminal **`terminal_workload_wedged`** dead
classification when the phase has been stuck at a **non-terminal** value past a
staleness floor AND the drain SSH round-trip is timing out (the `drain_alarm`
signal already exists in that branch; `creationTimestamp` is read at
`gcp.py:3075`, but a per-handle last-phase-change timestamp is the cleaner
staleness clock). Add `terminal_workload_wedged` to the
`_is_gcp_async_workload_failure` accept-set (`scripts/backend_poll.py:202-206`)
so #659's existing `_failover_dead_gcp_to_runpod` fires automatically (its
exactly-once lease/sentinel idempotency already guards against double-launch).

**Tests:** `test_gcp_backend.py` — `eps/phase=workload` frozen + SSH-drain-timeout
past the floor → `current_phase="terminal_workload_wedged"` / `status="dead"`;
`test_backend_poll.py` — the wedge fails over to RunPod exactly once.

## Fix 2 (belt-and-suspenders) — in-VM network watchdog

In `backends/gcp.render_startup_script`, add a lightweight watchdog that, on
sustained metadata-server / HF unreachability, flips `eps/phase=failed` (or
`shutdown` → TERMINATED, which the poller already reads as dead) so a networking
wedge becomes a recoverable terminal state. Caveat: if networking is fully dead
the phase write can't land either — so Fix 1 is load-bearing; Fix 2 catches the
marginal-network cases.

## Scope / acceptance

- `backends/*.py` is workflow surface but this is a substantive logic change —
  full /issue treatment (planner/critic/code-review).
- Acceptance: the two new tests pass; a frozen-non-terminal-phase + drain-timeout
  GCP VM is classified dead and fails over to RunPod exactly once (no manual
  pivot needed); no regression to the crashed-workload (#659) or healthy-running
  paths.
- The companion doc fix (stale failover-coverage prose in
  `compute-backend-failover.md` + CLAUDE.md + a gotchas.md wedge entry) is handled
  separately by a workflow-improver; this task is the code.

## Provenance

Root-caused by a background investigation of the #667 attempt-1 GCP wedge
(2026-06-25). Related: #659 (wired the crashed-workload async failover), #658
(sync failover + EXIT-trap crash diagnostics), #491 / #640 (sibling GCP
RUNNING-zombie modes).
