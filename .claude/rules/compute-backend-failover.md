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

- **Capacity / quota / zone miss** — walks the cost-ordered GCP ladder
  (on-demand A100-80 → A100-40 → spot), then the free SLURM lanes, then
  falls through to RunPod as the LAST rung (`reason:
  auto_fallback_runpod`, #656). RunPod never first, never skipping a
  cheaper rung.
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

### Coverage scope (current) — synchronous `route()`-time only

The failover fires on a `gcp.GcpWorkloadError` raised SYNCHRONOUSLY inside
`route()` (i.e. from `GcpBackend.launch()` during the router call). The
COMMON production GCP workload crash — a deterministic bug that surfaces
minutes into the run AFTER the VM is up — is detected by the ASYNC poller
(`backend_poll.py`), which emits a terminal `status:dead` /
`failure_class:code` JSON and does NOT re-call `route()`; that path still
ends at `status:blocked`, NOT a RunPod re-launch. Part A (crash
diagnostics) covers the async crash fully (the EXIT trap fires regardless
of how the workload died); Part B's RunPod failover does NOT yet — wiring
the async poller's workload-failure handler to re-dispatch on RunPod is a
separate `kind: infra` follow-up. So "a GCP failure of ANY class fails
over to RunPod" is, today, the synchronous-`route()` contract; the
async-crash failover is pending that follow-up.

## Tests of record

- `tests/test_router.py::test_gcp_workload_error_fails_over_to_runpod_no_slurm_cascade`
- `tests/test_router.py::test_workload_error_on_a_rung_fails_over_to_runpod_no_rung_advance`
- `tests/test_router.py::test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted` (capacity path, #656)
- `tests/test_gcp_backend.py::test_render_startup_script_persists_diagnostics_before_teardown`
- `tests/test_gcp_backend.py::test_render_startup_script_diagnostics_uploads_log_and_partial_artifacts`
- `tests/test_gcp_backend.py::test_render_startup_script_diagnostics_is_guarded_and_bounded`
- `tests/test_gcp_backend.py::test_render_startup_script_is_valid_bash`
