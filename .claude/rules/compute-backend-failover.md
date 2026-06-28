---
description: GCP-lane failover + crash-diagnostics-preservation policy for the multi-lane router — when a GCP attempt fails (capacity OR workload), what happens next, and how a GCP crash stays debuggable despite the delete-on-crash boot disk (loads when you touch the backend router package)
paths:
  - "src/explore_persona_space/backends/router.py"
  - "src/explore_persona_space/backends/gcp.py"
  - "src/explore_persona_space/backends/issue_dispatch.py"
  - "scripts/dispatch_issue.py"
  - "scripts/backend_poll.py"
  - "tests/test_router.py"
  - "tests/test_gcp_backend.py"
---

# Compute-backend failover + crash-diagnostics policy

CLAUDE.md § "Compute backends — multi-lane router" carries the always-on
summary; this file is the full policy + the #658 motivating incident, and
loads when you touch the router / GCP backend code. The router module
docstring (`backends/router.py`) is the in-code source of truth; keep all
three in sync.

## The motivating incident (#658, 2026-06-24)

An ephemeral GCE instance running an extraction crashed on a deterministic
code bug at one context index. The startup-script EXIT trap powered the VM
off and `--instance-termination-action=DELETE` deleted the instance AND
its boot disk, so:

- the workload traceback / serial logs were LOST every crash → the bug had
  to be diagnosed by inference;
- the ~30 partial output JSONs the workload had already written were LOST
  → no partial progress was recoverable;
- each retry burned a fresh GCP attempt + A100 GPU-hours producing nothing
  recoverable.

Two policy changes followed.

## Part A — GCE crash diagnostics + partial-artifact preservation

The GCE startup script (`backends/gcp.render_startup_script`) installs an
EXIT trap that, on a crash (rc != 0), powers the VM off to bound billing.
Because the boot disk is DELETEd on that shutdown, the trap now calls
`_eps_persist_diagnostics "$rc"` BEFORE `shutdown -h now` to upload, to the
HF data repo under `issue<N>_partial/<attempt_id>/`:

1. `crash_report.json` — exit code + timestamp + run identity;
2. `workload.log` — the workload log (traceback / stderr), `$EPS_LOG_PATH`;
3. `eval_results_issue_<N>/` — the partial `eval_results/issue_<N>/` the
   workload wrote before crashing.

Discipline (all load-bearing — the trap must never delay the poweroff that
bounds billing):

- **Crash-only safety net.** The workload's own HF/WandB upload paths stay
  the AUTHORITATIVE artifact route for a clean run; the EXIT-trap upload
  fires only on the rc != 0 branch (the clean-exit path keeps the VM alive
  for the success-sentinel scp + the workload already uploaded).
- **Fully guarded + time-bounded.** Early-returns without
  `EPS_HF_DATA_REPO` / `HF_TOKEN` (early-boot crash); the whole upload is
  wrapped in `timeout 300`; every step is `|| true`. A hung/failed upload
  can NEVER strand the `shutdown`.
- **Shared preamble.** The helper lives in the startup-script preamble, so
  BOTH the hydra (`train.py`) and the `--workload-cmd` branches get it.
- The data-repo target is rendered as `EPS_HF_DATA_REPO` (from
  `config.hf_data_repo`).

To recover after a `failure_class: code` GCP crash, look in
`superkaiba1/explore-persona-space-data/issue<N>_partial/`.

**Snapshot pin.** The EXIT-trap preamble is shared, so any change to it
alters the hydra-branch render and breaks the byte-identity snapshot test
`tests/test_gcp_backend.py::test_render_startup_script_hydra_only_byte_identical_to_pre_change_snapshot`.
Regenerate `tests/fixtures/issue588_gcp_startup_hydra_only.json`
DELIBERATELY with a documented provenance note (the fixture's purpose is
accidental-drift detection); the structural #658 tests keep the snapshot
non-tautological. Do NOT edit the rendered script without `bash -n`-ing
both branches first — the helper embeds a Python heredoc inside a function
inside a subshell, where a quoting slip would only surface at VM-boot time
(test: `test_render_startup_script_is_valid_bash`).

## Part B — GCP-failure → RunPod failover (contract reversal)

A GCP attempt failure of **ANY class now routes the next attempt to
RunPod** — the reversal of the historical "GCP workload failure surfaces
with NO fallback" invariant. Rationale: if GCP is failing a run, running
it on RunPod keeps the science moving AND gives a persistent, SSH-able pod
for diagnosis — strictly better than GCP's delete-on-crash boot disk.

Two distinct GCP-failure paths, BOTH ending at the same
`_runpod_terminal_rung`:

- **Capacity / quota / zone miss** — walks the length-aware GCP ladder
  (#680; see "Ladder order" below — short jobs spot A100-80 → spot A100-40
  → flex-start A100-80 → on-demand A100-80 → on-demand A100-40; long /
  unknown jobs flex-start A100-80 → on-demand A100-80 → on-demand A100-40,
  NO spot), then the free SLURM lanes, then falls through to RunPod as the
  LAST rung (`reason: auto_fallback_runpod`, #656). RunPod never first,
  never skipping a cheaper rung.
- **WORKLOAD failure** (`gcp.GcpWorkloadError`) — short-circuits STRAIGHT
  to RunPod (`reason: gcp_workload_failover_runpod`, #658), carrying the
  `GcpWorkloadError` evidence on the marker `extra`. It does NOT cascade
  across the remaining GCP rungs OR the free SLURM lanes — re-crashing
  broken code there burns queue time (the original no-cascade concern).
  Internally signalled by the private `_GcpWorkloadFailover` exception,
  caught by both lane callers (`_auto_route`, `_override_gcp_with_ladder`)
  — so an explicit `backend: gcp` pin fails over identically.

**Bound — no infinite RunPod cascade.** A genuinely-broken job runs on
RunPod AT MOST ONCE more. If it crashes again, the poller surfaces
`failure_class: code` → `status:blocked`. The autonomous-session watcher's
capacity-retry pass re-drives ONLY `failure_class: infra` +
`reason: no_compute_available` (`TRANSIENT_CAPACITY_REASONS`), never a code
failure — so a workload-failover crash on RunPod parks at `blocked` and is
not re-launched. The RunPod pod persists + is SSH-able for diagnosis (the
whole point of failing over there).

The SLURM-lane workload failure (`terminal_before_running` with runtime
artifacts on an explicit `--backend <slurm>` pin) STILL surfaces
`WorkloadSurfacedError` (it is not GCP; no failover) → `failure_class:
code` → `status:blocked`.

### Ladder order (length-aware, #680)

The GCP ladder (`backends/router._gcp_ladder_specs`) is keyed on job LENGTH
(`_is_short_job`: known GPU-hours ≤ `EPS_GCP_SPOT_MAX_GPU_HOURS`, default 2,
OR `spec.extra["spot_tolerant"]`):

- **SHORT jobs — spot leads:** spot A100-80 → spot A100-40 (fits-40 intents
  only) → flex-start A100-80 → on-demand A100-80 → on-demand A100-40
  (fits-40 only). A short job absorbs a spot preemption cheaply (the #659
  failover / checkpoint-resume), and spot is the cheapest live pool, so spot
  leads; flex sits between spot and on-demand as the "queue for capacity
  rather than fail" middle rung. A short `lora-7b` (a40 present) yields 5
  GCP rungs; a short `ft-7b` (no a40) yields 3 (spot-80, flex-80,
  ondemand-80).
- **LONG / UNKNOWN-length jobs — flex leads, NO spot:** flex-start A100-80 →
  on-demand A100-80 → on-demand A100-40 (fits-40 only). Spot is barred
  (preemption too costly for a long job); flex — non-preemptible once
  running, queues for capacity — leads. An unknown-length job (no time
  budget) is NOT short, so it takes this branch. A long `ft-7b` yields 2 GCP
  rungs; a long `lora-7b` yields 3.

The flex rung threads `provisioning_model=FLEX_START` via
`router._flex_start_rung` (label `flexstart_<gpu_kind>`); A2 acceptance of
`--provisioning-model=FLEX_START` was confirmed by a live #680 probe on both
`a2-ultragpu-1g` and `a2-ultragpu-4g`. The per-day attempt cap
(`MAX_GCP_ATTEMPTS_PER_DAY`) was bumped 5 → 8 to cover the up-to-5-rung
short-job ladder plus a same-day retry margin (still an attempt COUNT, never
a dollar cap). Tests of record: `test_ladder_short_job_spot_before_ondemand`,
`test_ladder_short_job_spot_miss_then_ondemand_order`,
`test_ladder_short_job_full_rung_order`,
`test_ladder_long_job_flexstart_before_ondemand_no_spot`,
`test_ladder_long_job_flexstart_miss_then_ondemand_order`,
`test_ladder_unknown_length_takes_long_branch`,
`test_flexstart_rung_threads_flex_provisioning`,
`test_max_gcp_attempts_per_day_is_eight`,
`test_workload_error_on_later_rung_fails_over_to_runpod` (all in
`tests/test_router.py`).

### Coverage scope (current) — both the synchronous and async crash paths

Both GCP workload-crash detection paths now fail over to RunPod:

- **Synchronous `route()`-time** — a `gcp.GcpWorkloadError` raised inside
  `route()` (from `GcpBackend.launch()` during the router call)
  short-circuits straight to RunPod, per Part B above (#658).
- **Async poller** (#659, merged 2026-06-24, PR #484) — the COMMON
  production GCP workload crash, a deterministic bug that surfaces minutes
  into the run AFTER the VM is up, is detected by the ASYNC poller
  (`backend_poll.py`). When `GcpBackend.poll` resolves a real workload
  crash to `current_phase == "terminal_workload_failed"` (the
  `eps/phase==failed` + write-once `eps/workload_started` sentinel
  discrimination, gcp.py §4.1.0b / MF3) with `status == "dead"`, the
  poller's `_is_gcp_async_workload_failure` predicate matches and
  `_failover_dead_gcp_to_runpod` re-dispatches the run on RunPod
  (`current_phase: gcp_workload_failover_runpod_async`), idempotency
  lease-backed so a sidecar/sentinel write failure cannot double-launch
  (#659 MF4). A GCP setup/boot/secrets/uv-sync failure surfaces
  `terminal_setup_failed` (sentinel absent) — a DIFFERENT phase the
  predicate excludes, so a broken-boot VM never re-crashes on RunPod.

Part A (crash diagnostics) covers BOTH crash modes regardless of how the
workload died — the EXIT trap fires on any non-zero exit. So "a GCP
workload failure of ANY class fails over to RunPod" now holds for both the
synchronous-`route()` and the async-poller crash paths.

### Remaining gap — the hung-but-RUNNING / frozen non-terminal phase (#667)

Neither failover path fires for a GCP VM that HANGS without ever publishing
a terminal `eps/phase`. The async predicate requires `status == "dead"` +
`current_phase == "terminal_workload_failed"`, and the synchronous path
requires `route()` to raise. A VM whose guest networking dies (DHCPv4
loss, the #667 case) — or whose workload wedges without the EXIT trap
firing — stays `RUNNING` with `eps/phase` frozen at a NON-terminal value
(e.g. `workload`), which `GcpBackend.poll` classifies `running` forever
(gcp.py `if status == "RUNNING"` → coarse `running` poll). So neither the
sync nor the async failover predicate matches, and the run sits live (and
billing) until a HUMAN notices and manually pivots to `--backend runpod`.
Closing this — escalating a frozen NON-terminal `eps/phase` past a
drain-timeout to a terminal wedged state that the async failover predicate
recognizes — is a pending `kind: infra` follow-up; until it lands the
recovery for a hung-but-RUNNING VM is a manual RunPod pivot. (See also the
#491 `bufio.Scanner: token too long` zombie in `.claude/rules/gotchas.md`,
a sibling hung-but-RUNNING mode recoverable in place via SSH relaunch.)

## Part C — RunPod RUNNING-but-no-port host wedge (#664)

The RunPod sibling of the GCP hung-but-RUNNING wedge (Part B / #669). RunPod
`desiredStatus` is decoupled from `runtime.ports`: a degraded host keeps the
pod RUNNING (and billing) while `runtime.ports` is empty, so
`runpod_api._parse_pod` yields `ssh_host=None`. `resume_pod` is HOST-PINNED
(`podResume{podId, gpuCount}`, no host reselection) — a stop+resume returns to
the SAME dead host. `--refresh-from-api` is a NO-OP here (the port is
platform-absent, not stale — that flag fixes the #488 stale-port case).

**Detection** (`backend_poll._maybe_escalate_runpod_wedge`, the RunPod sibling
of `_maybe_escalate_gcp_wedge`): a RunPod handle whose LIVE `desiredStatus`
stays RUNNING with null/empty `runtime.ports` past `RUNPOD_WEDGE_K_SEC`
(default 900s, env `EPM_RUNPOD_WEDGE_K_SEC`, mirroring
`GCP_STALENESS_FLOOR_SEC` — above `wait_for_ssh`'s 600s window + a retry
margin, so a healthy mid-resume pod never trips) is rewritten to
`status=dead, current_phase=RUNPOD_WORKLOAD_WEDGED_PHASE`. Within K, an
SSH-dead poll (`poll_once` returns `status=dead` on probe failure) is REWRITTEN
to `status=running` (`RUNPOD_WORKLOAD_OBSERVED_PHASE`) so the orchestrator
keeps polling until the wedge matures — a bare pass-through would stop on
ordinary dead before K. The no-port clock rides the sidecar `extra` dict
(keyed `runpod_noport_first_seen_ts`) and is fail-soft (atomic tmp+rename,
never raises on a malformed/non-numeric value — same contract as the GCP
`_read_phase_clock`); it is CLEARED the moment a public port appears or the
pod leaves RUNNING, so a transient slow-bring-up never escalates off a stale
timestamp.

**Recovery** (`backend_poll._failover_wedged_runpod`, gated on a PER-CELL
inputs-on-HF gate): once `_is_runpod_async_wedge_failure` matches, the per-cell
three-state gate (`_wedged_run_inputs_on_hf`) classifies each selected cell
from ONE fresh `list_repo_files` against the EXACT expected file set (S1, not
prefix-presence) — COMPLETE (both raw+store exact sets on HF) is safe, a
PARTIAL cell (one artifact-kind missing) BLOCKS, a NOT-YET-RUN (absent) cell
does NOT block (rerunnable from verified earlier inputs). With ZERO partial
cells, `terminate_pod` stops the billing leak and a FRESH pod is re-provisioned
(NOT a host-pinned resume) + the dispatcher resumed idempotently; the fresh
pod's P2 dispatcher skips HF-complete cells (`_cell_done_anywhere` = local-done
OR HF-complete). Any partial cell → NO terminate, surface a `failure_class:
infra` block (`reason=runpod_wedge_inputs_unverified`) so a human decides
(CLAUDE.md halt-criterion #2). This is the irreversible auto-terminate,
analogous to the `/issue` Step 8 auto-terminate-after-upload-PASS precedent —
data-safe because fix (a)'s per-cell incremental upload + the per-cell gate make
every COMPLETE cell's data already present on HF.

**Idempotency = a DURABLE lease + a sentinel, exactly as the GCP analogue
(#689 blocker-2).** The wedge failover guards its terminate + re-provision with
TWO records keyed to the wedged pod identity (pod_name/job_id): (1) the
AUTHORITATIVE durable lease at `~/.eps-routing/` (`Lease.runpod_wedge_failover_of`,
checked by `_lease_records_runpod_wedge_failover`, stamped by
`_stamp_runpod_wedge_failover` after the fresh-pod relaunch succeeds) — it
survives the EDQUOT / read-only-fs mode that fails BOTH the `.claude/cache`
sidecar AND the same-dir sentinel together, the same persistent-disk-failure mode
the GCP round-3 fix closed; and (2) the `.claude/cache` wedge sentinel as the fast
path. A sentinel-only guard re-opened the double-terminate hole under a
`.claude/cache` write failure. **Relaunch error mapping (#689 blocker-3):**
`_relaunch_fresh_runpod` maps a no-capacity RunPod (`NoComputeAvailableError`) to
a terminal infra JSON `reason=no_compute_available` (re-drivable by the watcher's
capacity-retry pass) and a sidecar-write failure (EDQUOT) to
`reason=sidecar_persistence_failed` (durable lease stamped to bound the relaunch),
mirroring `_failover_dead_gcp_to_runpod` — so a wedge failover always honors the
poller's terminal-JSON contract instead of crashing `main()` and stranding the run.

The data-safety PREREQUISITE is the per-cell incremental upload in
`scripts/issue664_dispatch.py` (`_upload_cell_artifacts`, fired the moment each
cell's extract+eval worker succeeds) — without it the terminate would strand
every not-yet-P3-uploaded cell. This closes the #664 gap. **The per-cell HF
surface includes the marker-slot stats for MARKER cells (#689 blocker-1, fix
a1):** `_upload_cell_artifacts` uploads `marker_slot_stats.json` (under
`HF_MARKER_SLOT_PREFIX`) and `_classify_cell_hub_state` requires it for a marker
cell to read "complete", so a fresh auto-migrated pod can HYDRATE it from HF
(`_hydrate_marker_slot_stats_from_hf`) before the A7 `_marker_readability_assert`
instead of crashing on a local-absent file (the assert SKIPs HF-complete marker
cells via A2 and would otherwise hit `checked == 0` and raise). The complementary
`cmd_resume` advice split (`pod_lifecycle.py`: a still-null resume names
terminate+re-provision, not the wrong `--refresh-from-api`) is the interactive
sibling; the report-only `running_no_port` flag in `pod_audit.py` is the
fleet-level visibility backstop (never auto-terminates).

### Part C watcher backstop — the poller-DEAD case (#692)

The poller-side detect+recover above runs ONLY while the per-issue poll loop is
alive — exactly when a backstop is NOT needed. When that loop has DIED (crashed
`/issue` session, OOM-killed bg-Bash poll chain, VM reboot),
`_maybe_escalate_runpod_wedge` never runs and the #664 billing leak goes
undetected. The `autonomous_session_watch.py` pod-safety pass (every 10 min,
session-independent) closes that gap with a wedge arm in `_process_pod`:

- **Compose, do NOT re-define.** The arm calls the SAME raw predicate
  `backend_poll._pod_is_runpod_runtime_wedged(info)` the poller uses (extracted
  from `_maybe_escalate_runpod_wedge`, composition surface (b)), reading the
  live `PodInfo` the API list pass already fetched, and the SAME imported
  `backend_poll.RUNPOD_WEDGE_K_SEC` (never a duplicated literal).
- **A DEDICATED wedge clock.** The maturity floor ages against
  `wedge_first_seen` (stamped at wedge ONSET, cleared on any non-wedge tick and
  on a pod_id change), NOT the pod-incarnation `first_seen` (which measures pod
  uptime, not the no-port episode). A `>= threshold` (default 2)
  consecutive-confirmed-checks miss guard backs it, so a transient API blip
  never stops a pod.
- **ALERT by default; AUTO-STOP only when provably safe.** Past K + confirmed,
  the arm posts a once-per-episode alert UNLESS the same inputs-on-HF gate fix
  (b) uses (`backend_poll._wedged_run_inputs_on_hf`) confirms zero partial cells
  AND a TRI-STATE keep-running read (`_wedge_keep_running` → `True | False |
  "unknown"`) returns the literal `False`. Every uncertainty path (no handle, HF
  error, tag present, tag-read FAILURE `"unknown"`) is ALERT-only — a persistent
  tag-read failure never silently overrides a keep-running tag on a live-work pod.
- **STOP, never terminate.** The recovery action is the reversible `pod.py stop`
  (volume preserved) — the watcher has no run handle guarantee and runs blind to
  the dispatcher's resume state, so it halts billing rather than terminating
  (CLAUDE.md halt-criterion #2). The poller's fix (b) owns the irreversible
  terminate + re-provision.
- **DONE-task ordering.** A wedged pod whose task is at a DONE status
  (`completed` / `awaiting_promotion` / `archived`) FALLS THROUGH to the
  existing status-class DONE auto-stop arm (the canonical escaped-pod handler) —
  the wedge arm only handles non-DONE (live-work) statuses, so it never weakens
  the existing auto-stop into a conditional one.

Tests of record: `tests/test_autonomous_session_watch_wedge.py` (the decision
table + boundaries, the tri-state gate, the fail-closed inputs gate, the
state round-trip, the pod_id reset, the alert dedup, the DONE fall-through) +
the direct predicate test
`tests/test_runpod_wedge_detection.py::test_pod_is_runpod_runtime_wedged_predicate`.

## Tests of record

- `tests/test_router.py::test_gcp_workload_error_fails_over_to_runpod_no_slurm_cascade`
- `tests/test_router.py::test_workload_error_on_a_rung_fails_over_to_runpod_no_rung_advance`
- `tests/test_router.py::test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted` (capacity path, #656)
- `tests/test_gcp_backend.py::test_render_startup_script_persists_diagnostics_before_teardown`
- `tests/test_gcp_backend.py::test_render_startup_script_diagnostics_uploads_log_and_partial_artifacts`
- `tests/test_gcp_backend.py::test_render_startup_script_diagnostics_is_guarded_and_bounded`
- `tests/test_gcp_backend.py::test_render_startup_script_is_valid_bash`
- `tests/test_runpod_wedge_detection.py` (Part C: within-K override, past-K
  escalation, malformed-clock fail-soft, predicate scope, per-cell gate
  partial-blocks / mid-sweep-allows, failover idempotency)
- `tests/test_issue664_per_cell_upload.py` (Part C prerequisite: per-cell
  incremental upload idempotency + exact-set + fail-loud verify + A2 fresh-pod
  resume)
