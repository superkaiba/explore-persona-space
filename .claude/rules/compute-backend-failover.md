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
  - "tests/test_runpod_wedge_detection.py"
  - "tests/test_backend_poll.py"
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
3. `worker_logs/<relpath>` (#885) — every regular file under
   `$WORKLOAD_ROOT/logs/` (fan-out per-worker logs carrying the REAL
   traceback; the canonical `workload.log` ends at the fan-out line — the
   #779 loss class, two ~30-min manual boot-disk detaches on 2026-07-02),
   newest-first by mtime, per-file tail cap `EPS_PERSIST_LOG_FILE_CAP_BYTES`
   (default 5 MiB — the traceback is at the END of a log, so an oversized
   file is TAILED at stage time, never skipped wholesale), file-count bound
   `EPS_PERSIST_LOG_MAX_FILES` (default 40; `< 1` is a loud-SKIP disable),
   canonical-log dedup skip, staged into `/tmp/eps-worker-logs` and uploaded
   as ONE `upload_folder` commit — per-file `upload_file` loops are banned
   on this large repo (the #664 504-storm gotcha). Uploaded AFTER the
   canonical `workload.log`, BEFORE the partial dirs (small-first);
4. `eval_results_issue_<N>/` — the partial `eval_results/issue_<N>/` the
   workload wrote before crashing;
5. `data_issue_<N>/` + `data_issue<N>/` (#854) — working-dir partials under
   BOTH `data/issue_<N>/` and `data/issue<N>/` naming conventions (the #825
   loss class: `data/issue_825/track_s.jsonl` was structurally outside the
   old eval_results-only sweep). Re-downloadable `hf_dl/` / `g*_dl/` /
   `store/` / `.cache/` caches are excluded at top level AND nested depths;
   an empty-after-excludes dir SKIPs; a per-dir byte cap (default 2 GiB,
   env `EPS_PERSIST_DIR_CAP_BYTES`) SKIPs an oversized dir loudly rather
   than burning the 300s budget;
6. `workload_<utc-ts>.log` (#854) — a per-crash timestamped copy of the
   workload log, uploaded AFTER the partial dirs (small-first ordering; the
   canonical `workload.log` already landed the traceback early). The
   canonical `crash_report.json` / `workload.log` names are OVERWRITTEN by
   a same-attempt re-crash (run-3 overwrote run-2's log on #825) — prior
   crashes' canonical copies stay recoverable via the HF repo's git
   history; the timestamped copies accumulate per crash;
7. `crash_persist_transcript.log` (#854) — the `[crash-persist]` audit
   lines, uploaded as the FINAL step. Its presence proves the persist ran
   to completion with every skip recorded; its ABSENCE proves a killed
   persist — the durable skip-vs-kill discriminator (the serial console is
   unreadable post-DELETE, #640).

**Sweep scope (explicit):** the partial sweep covers exactly the three
named directories above (`eval_results/issue_<N>/`, `data/issue_<N>/`,
`data/issue<N>/`) plus the `$WORKLOAD_ROOT/logs/` worker-log tree (#885) —
still NOT universal artifact discovery (e.g. `figures/issue_*`,
checkpoints, `ood_eval_results/` are not swept). Worker logs are swept
only when they land under `$WORKLOAD_ROOT/logs/` (relative `logs/…` or
`$REPO_ROOT/logs/…` on the workload-cmd branch, where the startup script
exports `REPO_ROOT="$WORKLOAD_ROOT"` — #641); absolute
`<vm_scratch_dir>/logs` paths are not swept — place dispatcher worker logs
under the workload-root `logs/` convention. A workload writing partials
elsewhere must place them under a swept dir or upload them itself.

Discipline (all load-bearing — the trap must never delay the poweroff that
bounds billing):

- **Crash-only safety net.** The workload's own HF/WandB upload paths stay
  the AUTHORITATIVE artifact route for a clean run; the EXIT-trap upload
  fires only on the rc != 0 branch (the clean-exit path keeps the VM alive
  for the success-sentinel scp — bounded, as of #935, by the done-grace
  self-poweroff window (default 90 min,
  `EPS_GCP_DONE_POWEROFF_GRACE_SECONDS`), whose expiry persist targets the
  SEPARATE `issue<N>_done/` prefix — + the workload already uploaded).
- **Fully guarded + time-bounded.** Early-returns without
  `EPS_HF_DATA_REPO` / `HF_TOKEN` (early-boot crash) — LOUDLY, with a
  `[crash-persist] SKIP-ALL` serial line (#854; a silent return is
  indistinguishable from a killed persist); the whole upload is wrapped in
  `timeout 300`; every step is `|| true`. A hung/failed upload can NEVER
  strand the `shutdown`.
- **Eager bounded serial streaming (#854).** The persist's output reaches
  fd 3 (the serial console) line-by-line AS IT HAPPENS via a pure-bash
  reader (2000-char line cap, 120-line print cap — raised 60 → 120 at #885:
  the worker-logs sweep's worst case of ~40 staging TAILED/SKIP lines + a
  dropped-count + 2 folder-upload lines on top of ~16 pre-existing persist
  lines sat right AT the old cap) — the old `| cut | tail`
  pipe buffered until EOF, so a killed/skipped persist left zero evidence.
  The reader keeps READING to EOF after the print cap (an early pipe close
  would SIGPIPE-kill the uploader mid-upload); every upload / failure /
  skip prints a `[crash-persist]` line — no silent skips anywhere.
- **Watchdog reaped at trap ENTRY (#854).** The EXIT trap kills the #669
  reachability watchdog — the only other in-guest poweroff actor — BEFORE
  the persist, so nothing can power the VM off mid-upload; the trap itself
  guarantees the billing-bounding shutdown. The clean-exit reap is
  unchanged.
- **Shared preamble.** The helper lives in the startup-script preamble, so
  BOTH the hydra (`train.py`) and the `--workload-cmd` branches get it.
- The data-repo target is rendered as `EPS_HF_DATA_REPO` (from
  `config.hf_data_repo`).

To recover after a `failure_class: code` GCP crash, look in
`superkaiba1/explore-persona-space-data/issue<N>_partial/`.

**Production fix-engaged signal (#854)** — keyed to the DURABLE HF
artifacts, since the serial console is unreadable post-DELETE (#640) and
the eager `[crash-persist]` serial lines are best-effort live-watch only:
on the next real GCP crash, the HF `issue<N>_partial/<attempt_id>/` prefix
gains the per-crash timestamped `workload_<ts>.log`, the
`crash_persist_transcript.log` (whose lines record every upload/skip —
including a loud SKIP naming why a `data_issue_<N>/` dir did not upload),
and `data_issue_<N>/` when the workload wrote one.

**The #854 incident record (correcting #825's premise).** The HF commit
log shows runs 1 AND 2 both landed `crash_report.json` + `workload.log`
via the trap (commits 06:19:29/46 and 08:16:01/08 UTC, 2026-07-02) — the
"round 1 left no diagnostics" / "only the tiny crash_report landed"
readings were artifacts of later runs overwriting the same canonical
paths. Only the partial DATA files (`data/issue_825/track_s.jsonl` etc.)
needed boot-disk recovery. The best-supported mechanism for that loss —
not directly proven (the VM and its serial log are deleted), but the one
consistent with the sequential upload commits and the ~20s
echo-to-poweroff window — is a silent coverage-gap skip: the old sweep
looked only in `eval_results/issue_<N>/`, so `data/issue_825/` was
structurally invisible and skipped without a log line, and the
end-buffered `| cut | tail` output made the silent skip indistinguishable
from a poweroff race. No code path could have uploaded `data/issue_825/`
regardless of timing. Hence the #854 fix set: coverage (item 5), loud
skips + eager streaming, the trap-entry watchdog reap (closing the one
other in-guest poweroff actor in principle), and the timestamped +
transcript artifacts (items 6-7).

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

Three distinct GCP-failure paths, ALL ending at the same
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
- **FLEX_START queue timeout** (`reason: gcp_queue_timeout_failover_runpod`,
  #783/#778) — the create SUCCEEDED but the instance never dequeued (stayed
  PENDING past `EPS_GCP_QUEUE_WAIT_SECONDS`). A poller-side clock detects it
  and fails it over to RunPod (see § FLEX_START queue-timeout below).
  DISTINCT from a workload crash (a queue that never advanced is not a
  crash) and from a capacity MISS at create time (this create succeeded).

**Intent translation at the terminal rung (#940).** The RunPod launch
paths (the terminal rung — all four fallback/failover paths above funnel
through it — AND the explicit `backend: runpod` override) translate a
GCP-only GPU intent to its nearest same-or-narrower RunPod-provisionable
intent via the router-owned `RUNPOD_INTENT_FOR_GCP_INTENT` map
(`capture-7b` → `eval`, `lora` / `lora-7b-h100` → `lora-7b`; identity
rows for shared intents, no marker record) BEFORE building the RunPod
spec — pre-#940 the rung passed the intent verbatim and
`gpu_heuristics.resolve_intent` KeyError'd, voiding the sanctioned last
rung (#841: `provision --issue 841 --intent capture-7b` exit 1 →
`NoComputeAvailableError` despite live RunPod 1-GPU capacity). A real
translation rides the `epm:backend-selected` marker `extra` as
`runpod_intent_translation: {"from": ..., "to": ...}`. A GCP GPU intent
with NO row — `eval-h100`, listed in
`RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS` (2× H100: no same-width
RunPod intent exists, and narrowing 2→1 would silently break a
2-GPU-sharded workload mid-run) — fails loud PRE-launch naming the
missing map row, inside the existing `no_compute_available` terminal on
the rung (raw `ValueError` on the override — a config error). A
completeness test (`test_translation_map_total_over_gcp_gpu_intents`)
pins map ∪ gaps == the `gpu_count > 0` keys of `gcp.INTENT_TO_MACHINE`,
so a future GCP intent added without deciding its RunPod fate fails CI
at the adding PR. CPU intents are untouched — the #677/#747
`RUNPOD_CPU_INSTANCE_FOR_INTENT` guard runs BEFORE translation.

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

### Per-rung multi-zone fan-out

Each GCP rung walks every `us-central1` zone the rung's machine type is
offered in BEFORE the rung is treated as a capacity miss. The create loop in
`backends/gcp.GcpBackend.launch` iterates `[primary_zone, *fallback_zones]`
(`DEFAULT_PRIMARY_ZONE` + `DEFAULT_FALLBACK_ZONES`) filtered by
`zones_for_machine_type` against `MACHINE_TYPE_ZONE_AVAILABILITY` — so the
rung only escalates after every zone where the machine type actually exists
has been tried.

Per-family zone sets (pinned to the 2026-06-30 gcloud re-verification,
`gcloud compute machine-types list --configuration=eps-gcp`, #774):

- **A100-80 family** (`a2-ultragpu-1g` / `-4g` / `-8g`): `{a, c}` — GCP does
  NOT offer this family in `-b` OR `-f`.
- **H100 family** (`a3-highgpu-1g` / `-2g` / `-8g`): `{a, b, c}` — NOT in
  `-f`.
- **A100-40** (`a2-highgpu-1g`, the #656 cheaper-but-smaller fallback rung):
  `{a, b, c, f}`.

#774 re-verified these and bumped `DEFAULT_FALLBACK_ZONES` from `(b, c)` to
`(b, c, f)` so the A100-40 fallback rung gains its fourth zone (`-f`). The
`MACHINE_TYPE_ZONE_AVAILABILITY` RESTRICT filter strips `-f` (and `-b`) back
out for the A100-80 / H100 families, so the broader DEFAULT never leaks a
doomed zone to a restricted family.

**The A100-80 cap at 2 zones is a GCP-imposed limit, not a config gap.**
Adding `-f` to the A100-80 zone set would issue a `MACHINE_TYPE_NOT_FOUND`
config error on every `-f` create attempt — burning the per-day GCP attempt
counter (`MAX_GCP_ATTEMPTS_PER_DAY`) on a guaranteed-to-fail create, exactly
what `MACHINE_TYPE_ZONE_AVAILABILITY` exists to prevent (#653). Do not
"widen" the A100-80 set without a fresh gcloud verification that GCP started
offering the family in a new zone.

**Per-zone fan-out visibility (#774 round 2).** The `epm:backend-selected`
marker's `attempts[].evidence` carries the per-rung zone fan-out via
`per_zone_attempts` (a list of `{zone, returncode, matched_pattern,
elapsed_s, stderr_tail}` records, one per zone the rung tried) plus a
human-readable `zones_attempted_summary` one-liner. The field is only emitted
when populated — the happy-path "landed on the primary zone" attempt entry
keeps the byte-identical pre-#774 7-field shape (`_attempt_to_dict` omits the
`evidence` key when empty). `GcpBackend.launch` accumulates the records across
the create loop and threads them onto `GcpProvisioningError.evidence`
(all-zones-exhausted for-else AND the non-capacity immediate-raise) and onto
`handle.extra["per_zone_attempts"]` on a success-after-miss launch; the router
catch sites lift them onto `RouteAttempt.evidence` via `_provisioning_evidence`.
This closes the #763 misdiagnosis where a multi-zone stockout read as a
single-zone one because only the last zone's `stderr_tail` survived. The
per-zone `stderr_tail` reuses the SAME already-published, truncated
`classify_create_failure` text (capped tighter, 200 chars), so no new
disclosure surface. See
`tests/test_router.py::test_route_attempt_evidence_field_round_trips_per_zone_attempts`
and
`tests/test_gcp_backend.py::test_zone_fanout_all_zones_exhausted_evidence_carries_per_zone_outcomes`
for the contract.

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

### FLEX_START queue-timeout → RunPod (#783/#778)

A GCP create can SUCCEED yet leave the instance in the FLEX_START capacity
QUEUE (`status=PENDING`, polled as `current_phase="pending"`, #782) — a
state DISTINCT from a capacity MISS at create time (no instance) and a
workload CRASH (instance up, then dead). Pre-#783 nothing bounded this
wait: the FLEX_START PENDING wait happens AFTER `route()` has already
returned a handle (GCP has no synchronous `route()`-time park), and the
async poll loop maps PENDING → `status="running"` / `current_phase="pending"`
DELIBERATELY so it keeps polling — forever. #778's `eps-issue-778` sat
PENDING ~2h45m before a manual RunPod pivot.

The async poller now ages a `"pending"` phase against a queue-wait floor,
mirroring the #669 frozen-phase wedge (`_maybe_escalate_gcp_wedge`):

- **Clock:** `backend_poll._maybe_escalate_gcp_queue_timeout` reuses the
  SAME sidecar phase clock (`_read_phase_clock` / `_write_phase_clock`) as
  the #669 wedge WITHOUT collision — the two key on DISJOINT
  `current_phase` values (`"pending"` here vs a frozen mid-workload phase
  there), so at most one is in scope on any tick, and the shared clock
  re-stamps on ANY phase change (a `pending → provisioning` dequeue resets
  it). NO reachability-alarm conjunction (unlike the #669 wedge): a stuck
  queue has no live VM to be reachable, so phase-frozen-past-floor on
  `"pending"` IS the whole signal.
- **Floor:** `EPS_GCP_QUEUE_WAIT_SECONDS` (default 600s, mirroring
  `router.FREE_WAIT_SECONDS` — the codebase's already-chosen "how long do
  we park a queued job before advancing the lane" constant), read at call
  time via `backend_poll._gcp_queue_wait_seconds` so ops can retune without
  a restart. It is an attempt-floor in seconds, NEVER a dollar cap; a
  missing / non-integer / non-positive value falls back to 600s.
- **Escalation → failover:** past the floor the poll is rewritten to
  `status=dead` / `current_phase=terminal_queue_timeout`, which
  `_is_gcp_queue_timeout` matches. `_failover_queued_gcp_to_runpod` then
  (1) best-effort `teardown`-DELETEs the still-queued PENDING instance to
  release the FLEX_START capacity request (a queued instance is live
  server-side and could dequeue later as an orphan — a crashed VM is
  already gone, so the #659 crash path does NOT teardown), then (2)
  re-dispatches on RunPod via the SAME
  `failover_to_runpod_after_async_workload_crash` seam the #659 path uses,
  passing `reason: gcp_queue_timeout_failover_runpod`. It reuses the SAME
  idempotency (durable lease + `.claude/cache` sentinel) + sidecar-repoint
  + terminal-JSON contract (the shared `_failover_gcp_to_runpod` core).
- **An ADDITIONAL advance trigger, NOT a lane-precedence change** — RunPod
  stays the terminal rung. A queue-timeout cancel is a CLEAN advance: it
  does NOT touch `MAX_GCP_ATTEMPTS_PER_DAY` (the counter bumps only on a
  create, inside `_attempt_one_gcp_rung`, which the poller never re-enters).
- **CPU-intent scope (#677/#747):** a `cpu-bigmem` PENDING instance is
  EXCLUDED (no cheap RunPod CPU lane → the ordinary dead path);
  `cpu-small` / `cpu-mid` (in `router.RUNPOD_CPU_INSTANCE_FOR_INTENT`) are
  eligible — `_is_gcp_queue_timeout` reuses `_is_gcp_async_workload_failure`'s
  exact CPU-intent guard.

The teardown is best-effort + guarded (never blocks the failover); a failed
delete degrades to the stale-GCP-VM janitor (`gcp_audit.py`) as the backstop.

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

### Gate-park zombie — RUNNING + terminal `eps/phase` (#908/#935)

A GCP workload that exits CLEANLY at a blocking gate (or finishes) leaves
the VM RUNNING only within the bounded #935 done-grace window (default
90 min, `EPS_GCP_DONE_POWEROFF_GRACE_SECONDS`; sentinel draining is why it
stays up at all — only a crash powers off immediately). Three guards close
the former unbounded leak (#763: an A100-80 idled ~40 min post-gate, then
the next dispatch silently no-op-reconnected to the zombie):
(0) BOUND (in-VM, #935) — the done-grace self-poweroff in the startup
script's success tail: a countdown that aborts on the operator keepalive
file (`/workspace/logs/issue-<N>-keepalive`) or an `eps/phase` re-publish
(a sanctioned same-VM relaunch), best-effort persists the UNDRAINED
sentinel set to HF `issue<N>_done/<attempt_id>/` at expiry (one retry;
`eps/done_persist=ok|failed` breadcrumb on a SEPARATE key), then powers
off UNCONDITIONALLY. This is the primary billing bound when every actor
below never runs (dead orchestrator/poller). A TERMINATED+`eps/phase=done`
instance polls as `status=done` / `workload_done_self_poweroff` (a
SUCCESSFUL run, never a crash); finalize then needs
`--skip-confirm-artifacts`.
(1) PRIMARY (faster) — `/issue` Step 6d.4 gate handlers tear the instance
down via `dispatch_issue.py finalize --skip-confirm-artifacts` after the
sentinel drain, split by gate class: PARK-mode gates finalize BEFORE the
park (never through the user-wait window); auto-resolving gates finalize
after resolution, BEFORE any off-pod phase or the fresh tail dispatch.
Never wait out the grace — the in-VM bound is the dead-orchestrator
fallback, not the plan.
(2) BACKSTOP (next launch) — `gcp.reconnect_or_none` refuses a RUNNING
instance whose
`eps/phase` ∈ {done, failed, wedged} (`_ZOMBIE_GUEST_PHASES`) and
`_stale_named_instance_or_none` returns it as deletable, so the next
launch reclaims + creates fresh instead of silently reconnecting. The
skip/delete sets are pinned identical (`tests/test_gcp_backend.py`, the
#632 invariant); a guest-attribute probe failure raises `GcpProbeError`
(never a reconnect, never a delete — the #535 discipline). Relaunch
contract (the #491 SSH-relaunch recipe in `.claude/rules/gotchas.md`): a
manual same-VM relaunch MUST re-publish `eps/phase=workload` BEFORE
resuming work, or the zombie predicates read the active relaunch as
terminal and the next dispatch reclaims it — and, as of #935, a relaunch
on a VM whose first workload published `done` must re-publish within the
done-grace window (or touch the keepalive file), else the in-VM countdown
powers the VM off at expiry. The #667 NON-terminal
frozen-phase gap above is UNCHANGED — a hung VM with `eps/phase=workload`
still needs the manual pivot (the done-grace countdown never arms there:
the success tail is never reached).

**Manual-pivot runbook line (#909):** a manual RunPod pivot that carries
`--workload-cmd` must ALSO pass `--execute-workload` — without it the launch
is provision-only (the pod boots, nothing runs) and the JSON carries
`workload_executed: false`; the alternative executor is dispatching the
experimenter on the provisioned pod (#909).

### CPU intents: cheap CPU lanes + the scoped #677 terminal (#747)

The GCP→RunPod failover above (capacity AND workload-crash, sync AND async)
now extends to the CHEAP CPU intents. #677 made EVERY CPU intent a hard
terminal (RunPod was GPU-only); #747 adds a RunPod CPU lane (`deployCpuPod`)
for the cheap intents and SUPERSEDES that terminal for them ONLY:

- **`cpu-small` / `cpu-mid` (mapped in `router.RUNPOD_CPU_INSTANCE_FOR_INTENT`)**
  fall over **GCP cheap CPU (E2; spot on a short job, on-demand otherwise) →
  RunPod CPU** when the GCP CPU lane is exhausted (capacity) OR crashes its
  workload (sync `_runpod_terminal_rung` + async
  `_is_gcp_async_workload_failure`, both keyed on the SAME map — the single
  source of truth). The RunPod re-dispatch carries `--intent cpu-small` /
  `cpu-mid`, which `gpu_heuristics.resolve_cpu_intent` resolves to the RunPod
  CPU instance_id (`cpu3g-2-8` / `cpu3c-8-16`) on the `pod_lifecycle` provision
  path (`runpod_api.create_cpu_pod`). RunPod CPU pods are on-demand only (no
  spot/interruptible CPU lever); a CPU no-capacity miss surfaces the existing
  `RunPodNoCapacityError` → terminal, re-drivable by the watcher's
  capacity-retry pass exactly as the GPU no-capacity path.
- **`cpu-bigmem` (ABSENT from the map — the >50 GB analysis lane)** keeps the
  #677 typed `CpuExhaustedNoRunpodLaneError` /
  `reason: cpu_exhausted_no_runpod_lane` terminal VERBATIM: it has no cheap
  RunPod equivalent, so on GCP exhaustion / crash it surfaces the typed
  terminal, NOT a RunPod fallback (the watcher's capacity-retry pass keys on
  `no_compute_available`, so the distinct reason means a structurally-unservable
  `cpu-bigmem` RunPod launch is never auto-retried).

Tests of record: `tests/test_router.py` (`test_router_cpu_small_capacity_miss_falls_over_to_runpod`,
`test_router_cpu_intent_capacity_miss_no_runpod_fallback` — the cpu-bigmem
guard, `test_gcp_ladder_cpu_small_short_yields_spot_then_ondemand`),
`tests/test_backend_poll.py` (`test_async_cpu_small_handle_fails_over_to_runpod`,
`test_async_failover_skips_cpu_gcp_handle` — the cpu-bigmem guard),
`tests/test_runpod_api_retry.py` (`test_deploy_cpu_pod_renders_instanceid_mutation`).

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

### Part C watcher backstop — the poller-DEAD case (#692/#770)

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
- **ALERT by default; TERMINATE+FAILOVER only when provably safe (#770).** Past
  K + confirmed, the arm posts a once-per-episode alert UNLESS the same
  inputs-on-HF gate fix (b) uses (`backend_poll._wedged_run_inputs_on_hf`)
  confirms zero partial cells AND a TRI-STATE keep-running read
  (`_wedge_keep_running` → `True | False | "unknown"`) returns the literal
  `False`. Every uncertainty path (no handle, HF error, tag present, tag-read
  FAILURE `"unknown"`, inputs unverified) is ALERT-only — a persistent tag-read
  failure never silently overrides a keep-running tag on a live-work pod.
- **TERMINATE + re-provision — the SAME recovery the poller owns, no longer a
  reversible stop (#770).** For the ONE provably-safe case (matured + confirmed
  wedge + `keep_running=False` + inputs verified on HF + a reconstructable run
  handle) the watcher routes the SAME irreversible terminate + fresh
  re-provision the poller owns (`backend_poll._failover_wedged_runpod`, via the
  `_wedge_failover` helper: read the `handle`+`sidecar` from the persisted
  sidecar EXACTLY as `_wedge_inputs_safe` does, synthesize a minimal `result`
  shim carrying only `current_phase` + `log_tail_excerpt`, terminate the wedged
  pod, re-provision a FRESH pod). It is **bounded-once** via the SHARED durable
  lease + sentinel (`_runpod_wedge_already_handled`, called INSIDE
  `_failover_wedged_runpod`), so a poller-side and watcher-side firing on the
  same wedge are mutually exclusive — the watcher inherits the cross-actor
  idempotency for free by calling that function. A reversible `pod.py stop` was
  the pre-#770 action but cannot heal a host-pinned dead RunPod host —
  `resume_pod` returns to the SAME dead host (#763) — which is why the promotion
  was made. `_wedge_failover` returns `(outcome, terminal_json)` (the poller's
  raw terminal-JSON dict, carrying `failure_class`/`reason`, or `None` on the
  alert/dry-run paths), and the dispatch maps the poller's terminal-JSON
  contract to five outcomes:
  - `failover` (fresh pod re-provisioned) → a generic `epm:progress` note.
  - `already-handled` (bounded-once short-circuit on the lease/sentinel — OR the
    SIDECAR-BINDING DEFENSE below) → a generic `epm:progress` note, no terminate.
  - `no-capacity` (terminated; RunPod unavailable → terminal
    `no_compute_available`) AND `blocked` (a terminal infra block a human
    resolves, halt-criterion #2) are BOTH terminal infra JSONs, so the watcher —
    the ACTOR here, the poll loop being dead — MIRRORS the poller's own path on
    that JSON: it posts `epm:failure v1` carrying the EXACT
    `failure_class`/`reason` from `terminal_json` (in the whitespace-token shape
    `_parse_failure_fields` reads) AND `set-status <N> blocked` (#770 r2,
    CRITICAL #1). This is what lets the watcher's capacity-retry pass re-drive
    the re-drivable `no_compute_available` block (it keys on the latest
    `epm:failure` marker's `failure_class`+`reason` ∈ `TRANSIENT_CAPACITY_REASONS`)
    and leave the non-capacity `blocked` reasons parked for a human — exactly as
    if the poller had emitted the same JSON. The `blocked` reasons differ
    on whether the pod terminated, so the marker text is NOT uniform (#770 r2,
    CONCERN #3): `runpod_wedge_inputs_unverified` is PRE-terminate (a PARTIAL
    cell → the pod was NOT terminated, still RUNNING until a human acts), while
    `sidecar_persistence_failed` and `runpod_wedge_relaunch_spec_missing` are
    POST-terminate (the wedged pod WAS terminated; the failure is in the fresh
    re-provision), and `runpod_wedge_failover_error` (#770 v2 r2/r3) is an
    UNEXPECTED raise from `_failover_wedged_runpod` that a `get_pod_by_name`
    liveness probe confirmed happened AFTER `terminate_pod` (the pod is GONE — a
    PRE-terminate raise where the pod is still alive degrades to `alert` instead,
    see below). The terminate decision itself still lives entirely inside
    `_failover_wedged_runpod`; the watcher never calls a SECOND terminate.
  - `alert` (NO reconstructable handle / a sidecar parse failure, OR a
    PRE-terminate raise from `_failover_wedged_runpod` where a `get_pod_by_name`
    liveness probe finds the pod STILL ALIVE — or the probe itself raised →
    UNCERTAIN, bias SAFE — #770 v2 r3) → degrade to ALERT-only, NEVER a blind
    terminate, and PRESERVE the wedge clock (the still-RUNNING pod is in the next
    tick's snapshot, so the wedge re-matures; never a FALSE terminal record while
    the pod bills). A POST-terminate raise (pod GONE per the probe) routes to the
    `blocked` `runpod_wedge_failover_error` reason above, NOT `alert` (the
    terminated pod is gone from the RUNNING-only snapshot, so the run WOULD be
    stranded without a durable record). Fail-soft either way — one wedged pod's
    failover error never crashes the 10-min tick. The sidecar-names-a-DIFFERENT-pod
    case is NOT `alert` — it maps to `already-handled` (the SIDECAR-BINDING DEFENSE
    above), because a re-pointed sidecar means a revived poller already failed the
    wedge over to a fresh, healthy pod the watcher must not terminate.

  The reversible `_stop_pod` is RETAINED for the status-class DONE escaped-pod
  arm (below), NOT the wedge arm.
- **Sidecar-binding defense — the fresh-pod race (#770 r2, CRITICAL #2).**
  Between `_wedge_inputs_safe`'s sidecar read (which verified inputs against the
  OLD wedged handle) and `_wedge_failover`'s re-read, a revived poller
  (crash-recovery / capacity-retry respawn) could have ALREADY failed the wedge
  over and re-pointed the sidecar at a FRESH, HEALTHY pod. The bounded-once
  lease/sentinel inside `_failover_wedged_runpod` is keyed on the FRESH handle's
  identity, so it would NOT catch this — the watcher would terminate a healthy
  fresh pod. Defense: immediately after the re-read, `_wedge_failover` asserts
  the freshly-read `handle.pod_name == info.name` (the wedged pod the watcher
  observed). A mismatch → return `already-handled` and NEVER call
  `_failover_wedged_runpod` against it.
- **DONE-task ordering.** A wedged pod whose task is at a DONE status
  (`completed` / `awaiting_promotion` / `archived`) FALLS THROUGH to the
  existing status-class DONE auto-stop arm (the canonical escaped-pod handler) —
  the wedge arm only handles non-DONE (live-work) statuses, so it never weakens
  the existing auto-stop into a conditional one.

Tests of record: `tests/test_autonomous_session_watch_wedge.py` (the decision
table + boundaries, the tri-state gate, the fail-closed inputs gate, the
state round-trip, the pod_id reset, the alert dedup, the DONE fall-through; #770:
the `terminate-failover` decision + invariant, the maturity-axis SR3 sweep, the
`_wedge_failover` outcome mapping — failover / already-handled / no-capacity /
blocked / alert-degrade — the dry-run no-side-effects contract, the SR1
cross-module reason-string parity, and the SR2 blocked-not-terminated +
clock-cleared end-to-end; #770 r2:
`test_wedge_no_capacity_emits_failure_marker_and_blocks_redrivable` (CRITICAL #1
— the no-capacity terminal JSON posts `epm:failure` + `set-status blocked` and
PARSES into a transient-capacity block via the real `_is_transient_capacity_block`
the capacity-retry pass uses),
`test_wedge_blocked_emits_failure_marker_and_blocks_not_redrivable` (a
non-capacity `blocked` reason blocks but is NOT re-drivable),
`test_wedge_blocked_marker_text_terminate_state_by_reason` (CONCERN #3 — the
marker text states the right terminate-state per reason),
`test_wedge_failover_sidecar_pod_name_mismatch_is_already_handled` +
`test_wedge_failover_sidecar_pod_name_match_proceeds` (CRITICAL #2 — the
sidecar-binding defense)); #770 v2 r3 (the pre/post-terminate raise split):
`test_wedge_failover_raise_after_terminate_routes_to_blocked` +
`test_wedge_failover_raise_after_terminate_routes_to_blocked_redrivable` (POST-terminate
raise, pod GONE per the probe → durable `blocked`),
`test_wedge_failover_preterminate_raise_pod_alive_degrades_to_alert` +
`test_wedge_failover_preterminate_raise_probe_raises_degrades_to_alert` +
`test_wedge_failover_preterminate_raise_does_not_falsely_block_live_pod` (PRE-terminate
raise / probe error → ALERT, clock preserved, no false terminal record) + the direct
predicate test
`tests/test_runpod_wedge_detection.py::test_pod_is_runpod_runtime_wedged_predicate`.

## Part D — RunPod CUDA-IMA repeat host wedge (#775)

The crash-signature sibling of Part C's no-port wedge. A RunPod H100/H200 can
wedge at the DRIVER level: a vLLM workload crashes with a CUDA illegal-memory-
access (`CUDA error: an illegal memory access was encountered` /
`EngineDeadError` / `Engine core proc … died unexpectedly`), the experimenter's
default `failure_class: infra` library-traceback recovery does an IN-PLACE
SAME-POD respawn (orphan-reap by exact PID + `nvidia-smi` VRAM probe + relaunch
— it NEVER terminates the pod, that lifecycle is the `/issue` skill's), and the
SAME-signature CUDA-IMA crash recurs on the same physical GPU. Part C's no-port
wedge does NOT catch this — the CUDA-IMA pod keeps its port + stays RUNNING — so
#763 needed a manual GCP→RunPod pivot. Part D automates exactly that recovery:
detect the SECOND same-signature CUDA-IMA crash and pivot to a FRESH host.

**Detection** (`backend_poll._maybe_escalate_runpod_cuda_ima`, the repeat-based
sibling of the time-based `_maybe_escalate_runpod_wedge`). The signal is
`PollResult.crash_signature` — the WIDE 500-line probe tail (NOT the 5-line
`log_tail_excerpt`, which truncates a 20-50-line vLLM traceback so the signature
line is routinely cut out — the B2 bug). `poll_once` captures it on a
`status="dead"` poll from the same wide surface it already fetched (no extra SSH
call; `_tail_excerpt_and_crash_signature`), and `RunPodBackend.poll` threads it
through to `main()` exactly as `stall_reason`. The escalation conjuncts:

1. **CUDA-IMA signature on the WIDE surface** (`CUDA_IMA_SIGNATURE`, within-line
   alternatives, no `re.DOTALL` so a match cannot span unrelated events across
   500 lines).
2. **A prior same-signature CUDA-IMA crash recorded THIS RUN** — the sidecar
   `extra["runpod_cuda_ima_last_seen"]` record (byte-mirror of the no-port
   clock family), with the prior `epm:failure` marker as a **cross-pod fallback
   source** (`_prior_failure_marker_is_cuda_ima`, read VM-side via
   `task_workflow.list_events`) so a sidecar wipe between pods does not lose the
   prior-crash record (B1). The record is CLEARED on any non-dead /
   non-CUDA-IMA poll, so a single transient CUDA-IMA the respawn recovered from
   (with an intervening healthy poll) does NOT accumulate — only a SECOND
   CUDA-IMA crash with no intervening healthy poll counts.
3. **EXCLUSION — no OUR-code traceback frame** (`failure_classifier.OUR_CODE_FRAME`,
   M3). A CUDA-IMA surface that ALSO traces through `src/explore_persona_space/`
   or `scripts/` is a deterministic CODE bug surfacing as CUDA-IMA, NOT a host
   wedge — fall through to the ordinary dead path (→ `failure_class: code`) WITHOUT
   spending a bounded pivot.

A FIRST CUDA-IMA crash (no prior record) records the signature and falls through
to the ordinary dead path (the in-place same-pod respawn gets its one chance); a
SECOND same-signature repeat (1)+(2) AND NOT (3) rewrites to
`current_phase=RUNPOD_CUDA_IMA_WEDGED_PHASE`, which `_is_runpod_cuda_ima_failure`
matches.

**Why signature-keyed, NOT pod_id-pinned.** The default vLLM-crash infra respawn
is IN-PLACE same-pod (verified: SKILL.md routes a library-traceback
`failure_class: infra` to "re-spawn the experimenter on the SAME branch /
relaunch on the same pod"; the experimenter never terminates pods), so pod_id
WOULD have worked. But the predicate keys on the crash SIGNATURE across the run
(NOT pod_id) because it is strictly more robust: it survives the watcher-side
stop/resume edge case (a `pod.py resume` rewrites host/port) and the half-
bootstrap fresh-pod path; the once-more bound is the safety against
over-counting. pod_id is incidental; the crash signature is the invariant.

**Wiring — BEFORE the no-port block (M2).** The CUDA-IMA escalation runs in
`main()` between the GCP async-failover block and the no-port escalation. The
no-port within-K path rewrites `status=dead → running`, and a CUDA-IMA crash can
leave the pod momentarily no-port (engine dead, ports not yet torn down), so the
no-port rewrite would mask a CUDA-IMA dead poll (which requires `status="dead"`)
if it ran first.

**Recovery + once-more bound** (`backend_poll._failover_cuda_ima_runpod`). It
REUSES the Part C inner relaunch (`_relaunch_fresh_runpod`) via a new `stamp_fn`
kwarg (Part C byte-unchanged at the default; the CUDA-IMA caller passes
`stamp_fn=_stamp_runpod_cuda_ima_failover`), so only the thin OUTER orchestration
forks. Layers, in the order checked:

1. **ONCE-MORE BOUND (M1).** If the DURABLE lease already records a CUDA-IMA
   failover for this run (the SEPARATE `Lease.runpod_cuda_ima_failover_of` field —
   distinct from `runpod_wedge_failover_of` so a no-port wedge and a CUDA-IMA
   wedge on the same issue never cross-suppress), the fresh host ALSO crashed
   same-signature → a fresh host did NOT heal it → it is a deterministic code
   bug. Emit a terminal **`failure_class: code`** JSON
   (`reason=cuda_ima_repeats_after_failover`) via the NEW `_terminal_code_json`
   (the `failure_class: code` sibling of `_terminal_infra_json`). The watcher's
   capacity-retry pass re-drives ONLY `failure_class: infra` +
   `no_compute_available`, so a `code` terminal PARKS at `blocked` for human
   inspection. NO second pivot.
2. **PER-WEDGE IDEMPOTENCY.** A re-fired tick on the SAME crashed handle after a
   successful pivot short-circuits (durable lease authoritative + `.claude/cache`
   sentinel fast-path, both keyed to the crashed pod identity).
3. **INPUTS-ON-HF GATE** (reused as-is — a PARTIAL cell BLOCKS the irreversible
   terminate; human decides).
4. **TERMINATE** the crashed pod (best-effort — a CUDA-IMA-wedged pod is usually
   already dead, so `info is None` simply skips the terminate) + **RE-PROVISION
   FRESH** stamping the CUDA-IMA lease field for the next crash's bound.

**The host-wedge interpretation is a HYPOTHESIS the failover TESTS, not a thing
the predicate proves.** The predicate detects a same-signature CUDA-IMA REPEAT;
whether that is a transient driver wedge (a fresh host fixes it) or a
deterministic code bug (a fresh host does NOT) is disambiguated EMPIRICALLY by
spending the one bounded pivot: a fresh-host success was a wedge; a fresh-host
re-crash lands at `failure_class: code` / blocked. The OUR_CODE_FRAME exclusion
(M3) cheaply removes the COMMON framed-code-bug case before spending the pivot;
a driver-only IMA (no user frame) still gets the bounded one-pivot test.

`scripts/failure_classifier.py` is UNCHANGED — CUDA-IMA already routes infra for
crash #1 (the in-place same-pod respawn); the new logic is poll-level and
short-circuits at the once-more case via `_terminal_code_json`.

## Tests of record

- `tests/test_router.py::test_gcp_workload_error_fails_over_to_runpod_no_slurm_cascade`
- `tests/test_router.py::test_workload_error_on_a_rung_fails_over_to_runpod_no_rung_advance`
- `tests/test_router.py::test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted` (capacity path, #656)
- `tests/test_router.py` (Part B FLEX_START queue timeout, #783:
  `test_queue_timeout_failover_seam_carries_queue_timeout_reason`,
  `test_failover_seam_default_reason_unchanged_byte_for_byte`,
  `test_queue_timeout_reason_distinct_from_crash_and_capacity_reasons`)
- `tests/test_backend_poll.py` (Part B FLEX_START queue timeout end-to-end, #783:
  `test_gcp_pending_past_timeout_fails_over_to_runpod`,
  `test_gcp_queue_timeout_failover_marker_carries_queue_timeout_reason`,
  `test_gcp_queue_timeout_does_NOT_increment_gcp_attempts_today`, + the negative
  controls: within-floor / non-pending-phase / first-observation / cpu-bigmem-excluded
  / teardown-failure-still-fails-over)
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
- `tests/test_runpod_wedge_detection.py` (Part D #775: CUDA-IMA predicate scope,
  signature regex, B2 wide-surface extraction vs the 5-line excerpt, escalation
  first-records / second-escalates / M3 our-code-frame exclusion / clear-on-
  recovery / malformed-fail-soft, B1 cross-pod prior-marker fallback, failover
  pivot with `stamp_fn` / bounded-once `_terminal_code_json` / idempotency /
  inputs-partial-blocks, durable-lease bound via the real helpers)
- `tests/test_backend_poll.py` (Part D #775 `main()` integration: first-falls-
  through, second-emits-fresh-host-failover, M2 cuda-ima-before-noport, M1
  exhausted-emits-`code` + `_is_transient_capacity_block` False)
