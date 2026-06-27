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

## Tests of record

- `tests/test_router.py::test_gcp_workload_error_fails_over_to_runpod_no_slurm_cascade`
- `tests/test_router.py::test_workload_error_on_a_rung_fails_over_to_runpod_no_rung_advance`
- `tests/test_router.py::test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted` (capacity path, #656)
- `tests/test_gcp_backend.py::test_render_startup_script_persists_diagnostics_before_teardown`
- `tests/test_gcp_backend.py::test_render_startup_script_diagnostics_uploads_log_and_partial_artifacts`
- `tests/test_gcp_backend.py::test_render_startup_script_diagnostics_is_guarded_and_bounded`
- `tests/test_gcp_backend.py::test_render_startup_script_is_valid_bash`
