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

> **#2028 — GCP PROVISIONING IS DISABLED (user directive 2026-08-02).** No
> dispatch path may CREATE a new GCP instance (GPU or CPU): the gate is
> `router.GCP_PROVISIONING_DISABLED = True` (rollback = flip it to `False`;
> no env re-enable switch). An explicit `backend: gcp` pin raises the typed
> `GcpDisabledError` (`reason: gcp_backend_disabled`), the auto chain's
> default order carries no gcp rung, and the #1596/#1601 queue-loss GCP
> on-demand retry legs refuse up front (falling through to the re-drivable
> `no_compute_available` terminal). Every GCP section below — the ladder,
> the five GCP→RunPod failover triggers, Part A crash diagnostics, the
> zombie/janitor machinery — is scoped to IN-FLIGHT GCP handles (which keep
> polling / tearing down / failing over to RunPod / crash-persisting) plus
> the single-constant rollback; it is NOT reachable for fresh dispatches
> while the flag is on. CPU intents route RunPod-only (`cpu-bigmem` gained
> the `cpu5m-16-128` row; the #677 typed terminal stays as the fail-loud
> floor for a future unmapped CPU intent).

> **#2054 — RUNPOD IS THE FIRST AUTO LANE (user directive 2026-08-05).**
> The RunPod team account is the shared Anthropic fellows/safety org pool —
> a sponsored pool, not discretionary spend — so `DEFAULT_AUTO_LANE_ORDER`
> now leads with `runpod` (`("runpod", "fellows", "nibi", "fir", "mila")`
> flag-ON; `("runpod", "fellows", "gcp", "nibi", "fir", "mila")` under the
> #2028 rollback build). The runpod-first lane launches via the SAME
> machinery as the #656 terminal rung (`reason: auto_runpod_first`); a
> capacity miss with nothing provisioned falls through to the lanes behind
> it, and the terminal rung SURVIVES as the end-of-chain RunPod retry
> (`reason: auto_fallback_runpod`). Every GCP→RunPod FAILOVER trigger below
> is unchanged. The local `pod_lifecycle` account-$/hr cap guard became
> ADVISORY-ONLY in the same change (it can never refuse or stall a
> provision; the RunPod console cap is the enforcement point).

CLAUDE.md § "Compute backends — multi-lane router" carries the always-on
summary; this file is the full policy, loaded when you touch the router /
GCP backend code. The router module docstring (`backends/router.py`) is the
in-code source of truth; keep all three in sync. Full pre-compaction
mechanics (rung-by-rung ladder walks, multi-zone fan-out details,
queue-vanish forensics, incident narratives): git history of this file
pre-2026-08-05.

## Part A — GCE crash diagnostics + partial-artifact preservation

(Origin #658: a GCE crash + `--instance-termination-action=DELETE` destroyed
the boot disk every retry — traceback and partial outputs lost.)

The GCE startup script (`backends/gcp.render_startup_script`) installs an
EXIT trap that, on a crash (rc != 0), runs `_eps_persist_diagnostics "$rc"`
BEFORE `shutdown -h now`, uploading to the HF data repo under
`issue<N>_partial/<attempt_id>/`:

1. `crash_report.json` — exit code + timestamp + run identity;
2. `workload.log` — traceback / stderr, tail-capped
   (`EPS_PERSIST_LOG_FILE_CAP_BYTES`, default 5 MiB). Items 1-2 ride ONE
   staged `upload_folder` commit with one retry
   (`EPS_PERSIST_RETRY_BACKOFF_S`, 10 s); the persist makes ZERO per-file
   `upload_file` calls (the #664 504-storm class) (#1151);
3. `worker_logs/<relpath>` (#885) — every regular file under
   `$WORKLOAD_ROOT/logs/` (fan-out per-worker logs carrying the REAL
   traceback, #779) AND under `/workspace/logs`
   (`EPS_PERSIST_WORKSPACE_LOGS_DIR`, staged
   `worker_logs/workspace_logs/<rel>`; `*.pid` / `*.processed` / the
   canonical log excluded; missing dir SKIPs loudly — #1605/#1415),
   newest-first, per-file tail cap, file-count bound
   `EPS_PERSIST_LOG_MAX_FILES` (40), git-TRACKED repo files excluded at
   walk time (fails OPEN with a loud WARN; #1351), ONE `upload_folder`
   commit;
4. `eval_results_issue_<N>/` — partial eval_results;
5. `data_issue_<N>/` + `data_issue<N>/` (#854; the #825 loss class) —
   working-dir partials, both naming conventions. Re-downloadable caches
   (`hf_dl/` / `g*_dl/` / `store/` / `.cache/`) excluded at all depths;
   per-dir byte cap (`EPS_PERSIST_DIR_CAP_BYTES`, 2 GiB) SKIPs loudly; a
   dir over `EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT` (1000) uploads as
   newest-first staged batches with bounded retry, abandoning loudly after
   `EPS_PERSIST_DIR_BATCH_ABORT_STREAK` (2) failed batches (#1339).
   Operator note: after an ABORT / FAILED dir line, verify the Hub prefix
   (scoped `list_repo_tree`) BEFORE re-running recovery — a
   gateway-timeout "failure" may have landed server-side (#1090 fu5);
6. `workload_<utc-ts>.log` (#854) — per-crash timestamped log copy
   (canonical names are OVERWRITTEN by a same-attempt re-crash — prior
   copies recoverable via the HF repo's git history);
7. `crash_persist_transcript.log` (#854) — the `[crash-persist]` audit
   lines; items 6-7 ride ONE final `upload_folder` commit, transcript
   staged LAST. Presence proves the persist completed; ABSENCE proves a
   killed persist (the serial console is unreadable post-DELETE, #640);
8. **the `eps/persist` guest-attribute breadcrumb (#1151)** — the
   HF-INDEPENDENT persist-fate channel (#811: every other signal rides the
   SAME HF channel and can fail together). Written on the SEPARATE
   `eps/persist` key (never `eps/phase`, which poll classification + the
   #908 zombie predicates key on): `attempted` unconditionally at entry,
   `skipped_no_token` on the early-boot token-guard skip, then a final
   status from an rc-file readback of the persist subshell
   (`EPS_CRASH_PERSIST_RC`) — `ok` (rc 0; since #1343 only when ≥1 upload
   verifiably succeeded), `failed_uploads` (rc 3 — ran to completion, zero
   verified uploads; #1343/#1315), `timeout` (rc 124),
   `failed_stream_flush` (rc 120; unreachable post-#1799 — reappearance
   means the guards regressed), `failed_rc<N>`. A MISSING rc file writes
   NOTHING: the standing `attempted` IS the killed-mid-persist signal.
   Boot-time DELETE clears the key; ≤3 fail-soft `curl -m 5` writes per
   crash. The poller reads it best-effort on failed-classifying ticks and
   appends `[crash-persist-breadcrumb] eps/persist=<value>` to the
   terminal marker's `log_tail_excerpt`
   (`GcpBackend._guest_persist_breadcrumb` — never raises, never gates
   classification or failover).

   Decision table (`eps/persist` × the HF `issue<N>_partial/` prefix):

   | `eps/persist` | HF prefix | Reading |
   |---|---|---|
   | ABSENT | absent | pre-fix render, trap never entered persist, entry-curl failure, OR dead metadata channel (co-signal: `eps/phase=failed` written + breadcrumb absent isolates the pre-persist window; phase ALSO absent ⇒ route to the #667 wedge lane) |
   | `attempted` (standing) | absent | persist KILLED mid-flight (TERMINATED-only reading — a RUNNING-window read may catch a healthy persist in flight) |
   | `attempted` (standing) | present | kill landed after uploads, before the final-status write (rare; transcript disambiguates) |
   | `skipped_no_token` | absent | early-boot crash before secrets fetch |
   | `ok` | present | normal crash persist (`ok` = persist exited 0 AND ≥1 verified upload, #1343; probe is attempt-scoped — states prefix recoverability, not necessarily THIS crash's uploads) |
   | `ok` | absent/partial | rare post-#1343: repo/prefix misroute or listing lag |
   | `failed_uploads` | absent | TOTAL upload failure (dead HF channel / 429 storm / rejected token, #1315); recover via serial console / boot-disk surgery |
   | `failed_uploads` | present | late-landing commit(s) — the #1339 gateway-timeout shape; trust a scoped prefix listing over the breadcrumb |
   | `timeout` | absent/partial | 300s budget exhausted (stalled uploads); a PRESENT prefix alongside `timeout` = healthy persist SIGTERMed mid-verify-probe (#1343) |
   | `failed_stream_flush` | absent | persist python died on a dead stdout/fd-3 pipe before the first upload (#1739; unreachable post-#1799) |
   | `failed_rc<N>` | absent | bootstrap/compound failure (127 = uv missing; 1 = cd short-circuit or python top-level; else python rc) |

**Sweep scope (explicit):** exactly the three named dirs above, the
`$WORKLOAD_ROOT/logs/` worker-log tree (#885), the `/workspace/logs`
dispatcher convention dir (#1605), and the `analysis_tensors*` staging
trees at BOTH live roots (`$WORKLOAD_ROOT/analysis_tensors*` and
`/workspace/analysis_tensors*`, env `EPS_PERSIST_WS_TENSORS_ROOT`; #1890,
#1739), swept LAST (largest class; a budget timeout mid-tensor-upload is
BY DESIGN), any nested `store/` pruned (mirrored-on-HF durable class). NOT
universal artifact discovery (`figures/issue_*`, checkpoints,
`ood_eval_results/` unswept) — a workload writing partials elsewhere must
place them under a swept dir or upload them itself; never append crash
forensics to a git-tracked committed file (the walk excludes tracked
paths, #1351).

Discipline (all load-bearing — the trap must never delay the poweroff that
bounds billing): **crash-only safety net** — the workload's own HF/WandB
upload paths stay AUTHORITATIVE for a clean run; the EXIT-trap upload
fires only on rc != 0 (the clean-exit path keeps the VM alive bounded by
the #935 done-grace window, default 90 min,
`EPS_GCP_DONE_POWEROFF_GRACE_SECONDS`; expiry persist targets the SEPARATE
`issue<N>_done/` prefix). **Fully guarded + time-bounded** — loud
`[crash-persist] SKIP-ALL` early-return without `EPS_HF_DATA_REPO` /
`HF_TOKEN`; whole upload wrapped in `timeout 300`; every step `|| true` —
a hung upload can NEVER strand the `shutdown`. **Eager bounded serial
streaming (#854)** — output reaches fd 3 line-by-line via a pure-bash
reader (2000-char line cap, 200-line print cap; the reader keeps READING
to EOF after the print cap so an early pipe close never SIGPIPE-kills the
uploader); every upload / failure / skip prints a `[crash-persist]` line.
**Watchdog reaped at trap ENTRY (#854)** — the trap kills the #669
reachability watchdog (the only other in-guest poweroff actor) BEFORE the
persist. **Shared preamble** — both the hydra and `--workload-cmd`
branches get the helper; data-repo target rendered as `EPS_HF_DATA_REPO`.

To recover after a `failure_class: code` GCP crash, look in
`superkaiba1/explore-persona-space-data/issue<N>_partial/`; the
fix-engaged signal on the next real crash is the `eps/persist` breadcrumb
on the terminal diagnosis marker (HF-independent, #1151) plus the
timestamped log + transcript on the HF prefix (#854/#811).

**Snapshot pin.** The EXIT-trap preamble is shared, so any change to it
alters the hydra-branch render and breaks the byte-identity snapshot test
`tests/test_gcp_backend.py::test_render_startup_script_hydra_only_byte_identical_to_pre_change_snapshot`.
Regenerate `tests/fixtures/issue588_gcp_startup_hydra_only.json`
DELIBERATELY with a documented provenance note (the fixture's purpose is
accidental-drift detection); the structural #658 tests keep the snapshot
non-tautological. Do NOT edit the rendered script without `bash -n`-ing
both branches first (test: `test_render_startup_script_is_valid_bash`).

## Part A-ter — finalize-failed-but-artifacts-ok (#1055)

A GCE workload can exit non-zero AFTER all its declared deliverables are
verified uploaded (the #811 shape; a known mechanical class is the rc=134
HF-datasets interpreter-shutdown SIGABRT, `gotchas.md`). The EXIT trap
branches on POSITIVE FILE EVIDENCE instead of triggering the full
crash-response machinery on a run that lost nothing:

- **Sentinel contract.** The startup script exports
  `EPS_DELIVERABLES_OK_PATH` (attempt-scoped, `gcp.deliverables_ok_path_for`)
  and `rm -f`s it at boot. The WORKLOAD writes that file ONLY after its
  final upload+verify step confirms every declared deliverable is on HF.
  **Writer rule (multi-stage drivers):** stamp ONLY after the LAST
  deliverable-producing step's upload+verify PASS. A git-committed eval-JSON
  leg is itself a deliverable: do NOT stamp while a result commit is
  unpushed (`git rev-list --count origin/<branch>..HEAD` != 0) — else a
  #1205 push-verify backstop failure classifies done-like instead of
  `failed` (`.claude/rules/pod-side-reporting.md` § Result-push
  verification). For a composed `--workload-cmd` chain, insert
  `&& touch "$EPS_DELIVERABLES_OK_PATH"` BETWEEN the deliverable-producing
  step and the tail steps (a trailing `&& touch` only runs on rc==0 —
  useless by construction). RECOMMENDED: populate `verified_prefixes` in
  the sentinel JSON (the trap checks EXISTENCE only).
- **Fail-open default.** A workload that never writes the sentinel keeps
  the `failed` path byte-for-byte; adoption is per-driver experiment code.
- **Trap branch.** rc≠0 AND sentinel present → `_eps_phase
  finalize_failed_artifacts_ok`; else `failed`. The shared tail (watchdog
  reap, log tail, persist, `shutdown -h now`) runs on BOTH arms. The
  classification is never keyed on rc value or crash timing, and the
  literal `done` phase is never published on the rc≠0 path (#1004).
- **Poll classification.** RUNNING or TERMINATED +
  `eps/phase=finalize_failed_artifacts_ok` → `PollResult(status="done",
  current_phase="workload_done_finalize_failed")` — a SUCCESSFUL run whose
  finalize hiccupped: fails both async-failover conjuncts (no RunPod
  failover, no crash-fix routing), resets the #1029 boot-death streak, and
  is in `_TERMINAL_GUEST_PHASES` ⇒ `_ZOMBIE_GUEST_PHASES` (janitor reaps a
  RUNNING VM stuck in it; reconnect treats it as a finished zombie).
- **Finalize.** The COMPLETION sentinel is never written on this path, so
  `confirm_artifacts` FAILs — run `dispatch_issue.py finalize
  --skip-confirm-artifacts`; Step 8 upload verification remains the
  independent artifact gate. Recurring occurrences indicate a SYSTEMATIC
  finalize bug — the classification keeps it visible for triage.

## Part B — GCP-failure → RunPod failover (contract reversal)

A GCP attempt failure of **ANY class routes the next attempt to RunPod** —
the reversal of the historical "GCP workload failure surfaces with NO
fallback" invariant. Rationale: RunPod keeps the science moving AND gives a
persistent, SSH-able pod for diagnosis (GCP deletes its boot disk on
crash). Five distinct GCP-failure paths, ALL ending at the same
`_runpod_terminal_rung`:

- **Capacity / quota / zone miss** — walks the length-aware GCP ladder
  (§ Ladder order), then the free SLURM lanes, then falls through to the
  RunPod terminal rung (`reason: auto_fallback_runpod`, #656) — since
  #2054 an end-of-chain RETRY (RunPod already led as lane 1,
  `reason: auto_runpod_first`).
- **WORKLOAD failure** (`gcp.GcpWorkloadError`) — short-circuits STRAIGHT
  to RunPod (`reason: gcp_workload_failover_runpod`, #658), carrying the
  error evidence on the marker `extra`. Does NOT cascade across remaining
  GCP rungs or SLURM lanes (re-crashing broken code there burns queue
  time). Signalled internally by `_GcpWorkloadFailover`, caught by both
  lane callers (`_auto_route`, `_override_gcp_with_ladder`) — an explicit
  `backend: gcp` pin fails over identically.
- **FLEX_START queue timeout** (`reason:
  gcp_queue_timeout_failover_runpod`, #783/#778) — create SUCCEEDED but
  the instance never dequeued (PENDING past `EPS_GCP_QUEUE_WAIT_SECONDS`);
  poller-side clock detects + fails over (§ FLEX_START queue-timeout).
- **Pre-workload BOOT LOOP** (`reason: gcp_boot_loop_failover_runpod`,
  #1029) — N (default 2) consecutive pre-workload setup deaths on the
  SAME rung, counted per launch incarnation in the durable lease
  (§ Pre-workload boot-loop).
- **FLEX_START queue VANISH** (`reason: gcp_queue_vanish_failover_runpod`,
  #1116/#1112) — create SUCCEEDED, instance sat PENDING, then DISAPPEARED
  from instances-list with no delete op (the DWS queue dropped it
  server-side); failed over on the FIRST occurrence (§ FLEX_START
  queue-vanish).

**Intent translation at the terminal rung (#940).** The RunPod launch
paths (terminal rung AND explicit `backend: runpod` override) translate a
GCP-only GPU intent to its nearest same-or-narrower RunPod-provisionable
intent via `RUNPOD_INTENT_FOR_GCP_INTENT` (`capture-7b` → `eval`, `lora` /
`lora-7b-h100` → `lora-7b`; identity rows for shared intents) BEFORE
building the RunPod spec (pre-#940 the rung KeyError'd, voiding the last
rung — #841). A real translation rides the `epm:backend-selected` marker
`extra` as `runpod_intent_translation`. A GCP GPU intent with NO row
(`eval-h100`, in `RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS` — narrowing
2→1 would silently break a 2-GPU-sharded workload) fails loud PRE-launch
inside the `no_compute_available` terminal.
`test_translation_map_total_over_gcp_gpu_intents` pins map ∪ gaps == the
`gpu_count > 0` keys of `gcp.INTENT_TO_MACHINE`. CPU intents are untouched
— the #677/#747 `RUNPOD_CPU_INSTANCE_FOR_INTENT` guard runs BEFORE
translation.

**Bound — no infinite RunPod cascade.** A genuinely-broken job runs on
RunPod AT MOST ONCE more; a second crash surfaces `failure_class: code` →
`status:blocked`, which the watcher's capacity-retry pass (infra +
`no_compute_available` only) never re-drives. The RunPod pod persists +
is SSH-able for diagnosis.

The SLURM-lane workload failure (`terminal_before_running` with runtime
artifacts on an explicit `--backend <slurm>` pin) STILL surfaces
`WorkloadSurfacedError` (no failover) → `failure_class: code` →
`status:blocked`.

### Cross-session pivot — resolve the owner before provisioning (#2067)

**A pivoting session that provisions a pod on a task it does not own MUST
first resolve the owner.** A durable marker alone does NOT wake a live
owner (#2067: a pivoter provisioned an 8×H200 recording only an
`epm:progress` marker; the owning session kept waiting on the abandoned
lane while the pod idled at 0% GPU). "I do not own this task" means: this
session is not the registered / issue-mapped session for task N (check
`scripts/spawn_session.py list`'s issue column), and it was not spawned to
run `/issue N`.

**Pre-pivot owner resolution (tri-state — the pivoter-side sibling of the
#1667 `_wedge_owner_live` probe).** Resolve the owner to LIVE | DEAD |
UNKNOWN from three evidence sources, ALL read fresh at pivot time:

- **Registration:** a registration file for task N —
  `~/.eps-autonomous/issue-<N>.json` or `manual-issue-<N>.json` — naming a
  `happy_session_id` that `spawn_session.py list` still shows live; a
  cwd-mapped daemon child on the issue worktree counts the same.
- **Recent markers:** any non-watcher marker on task N's events within the
  last 2h (the `EPM_WEDGE_OWNER_RECENT_H` window), read via
  `task.py view <N> --json`.
- **Transcript freshness:** the registered session's transcript mtime
  within the same 2h window.

LIVE = a live registered/cwd-mapped session exists (regardless of
transcript age) OR any recent non-watcher marker within the window.
DEAD = registration absent-or-stale AND no non-watcher marker within the
window AND no fresh transcript. UNKNOWN = any evidence source unreadable
(list RPC failure, marker read failure, missing daemon) — never coerce a
read failure to DEAD.

**Action table:**

| Owner state | Pivoting session does |
|---|---|
| **LIVE** | REFUSE the pivot — do NOT provision. Post `epm:progress` on N (`pivot-refused: owner <sid> LIVE via <evidence>; proposed lane <old> -> <new>`); the LIVE owner decides its own lane. Never `spawn_session.py stop` a LIVE owner to clear the way. |
| **DEAD** | SANCTIONED TAKEOVER, in order: (1) `spawn_session.py stop --session-id <sid>` on the dead owner's session id (auto-unregisters, #1455) — else `spawn_session.py unregister --issue N` for a stale registration (never `rm` on `~/.eps-autonomous/`); (2) post the takeover marker (recording contract below); (3) cancel/tear down the abandoned lane's queued work (e.g. `scancel`) so the old lane cannot double-run — **on teardown FAILURE, do NOT proceed to (4)**: HOLD the provision, record `residual=lane-teardown-failed: <cmd> exit <rc>` in the takeover marker, surface for user triage; (4) provision. |
| **UNKNOWN** | REFUSE — treat as LIVE (fail-safe, mirroring #1667: uncertainty never licenses the irreversible arm). Post the `pivot-refused` marker with `owner UNKNOWN via <which read failed>`; retry next tick or surface to the user. |

**Recording contract.** Every resolution outcome posts ONE `epm:progress`
marker on task N via `task.py post-marker` BEFORE any provision, note
grammar: `pivot-ownership: <refused|takeover>; owner=<sid|none>;
state=<LIVE|DEAD|UNKNOWN>; evidence=<registration|markers|transcript
reads, one clause each>; lane <old> -> <new>; by-session=<this sid>`. On a
DEAD takeover the marker is posted AFTER the stop/unregister and BEFORE
the provision. The marker is the durable RECORD; the stop/unregister is
what prevents the dead owner's respawn from re-claiming the lane — neither
substitutes for the other.

**Relation to the #1667 wedge owner-liveness guard (Part C watcher
backstop):** #1667 is WATCHER-side (gates terminate-vs-defer on an
already-wedged pod); #2067 is PIVOTER-side (gates
provision-vs-abort/takeover before a pod exists). Two call sites, one
shared conceptual probe (tri-state owner liveness; literal-DEAD required
for the irreversible arm; uncertainty fails safe).

**Race note.** The DEAD arm requires TRIPLE simultaneous stale evidence, so
mis-classification of a live owner is an adversarial contrivance. The
residual race (owner dies, then respawns in the ~1s window between DEAD
determination and `stop`) degrades safely: `stop`'s auto-unregister
(#1455) prevents the respawned owner re-claiming ownership, and the
takeover marker posted BEFORE `provision` is the durable record — no
split-ownership can arise.

**Sibling rule:** CLAUDE.md § teammate coordination ("one implementer per
file set") — the compute analogue: one owner per task's compute; never
provision into split-ownership.

### Ladder order (length-aware, #680)

NOTE (#1609/#2028/#2054): the standing auto order is
`DEFAULT_AUTO_LANE_ORDER = ("runpod", "fellows", "nibi", "fir", "mila")`
(the 6-lane order with `gcp` third is the flag-off rollback build). A
fellows capacity miss / dead endpoint / PENDING-at-cap park (after the
granted-QoS ladder high-eur → normal-eur → low-eur park-fails on the AUTO
path, #1899 — scancel + re-submit per `ClusterConfig.qos_ladder` rung,
`EPS_FELLOWS_LADDER_RUNG_WAIT_SECONDS` default 300 s per fallback rung;
explicit `backend: fellows` pins never walk the ladder) advances to the
free DRAC/Mila lanes; the GCP ladder below is entered only under the
flag-off rollback. Fellows rollback: flip the fellows `CLUSTER_CONFIGS`
row to `available=False` or set `EPM_AUTO_LANE_ORDER=runpod,nibi,fir,mila`
(a `gcp` entry raises while `GCP_PROVISIONING_DISABLED` is on). Sentinel
drain: fellows is a DRAINED lane as of #1898
(`slurm_monitor.drain_cluster_sentinels` over `ssh charmander` each poll
tick); the residual hazard is DRAC/Mila only (no `/workspace` — fail-loud
at `mkdir`, #608), so a sentinel-dependent workload pins a drained lane
(runpod/fellows) at plan time or accepts the fall-through risk
(verify_plan c43 WARNs).

**The fellows QoS ladder is RESUMABLE across launcher deaths (#2161).**
The AUTO-path ladder park persists its position (rung index, rung-park
elapsed, SLURM job id) to the durable per-issue lease
(`Lease.free_lane_park_state`) on every rung transition, and the CLI
path (`dispatch_issue.py launch`) bounds each PROCESS's park at
`RouterConfig.park_process_budget_seconds`
(`EPS_LAUNCH_PARK_PROCESS_BUDGET_SECONDS`, default 420 s — sized under
the 600 s Bash-tool cap): at budget with the job still queued, the
launch exits 75 (`reason: free_lane_park_budget_reached`, the third
exit-75 producer) instead of parking past the caller's wall. A re-run
of the SAME launch command reconnects to the queued job by its
`eps-issue-<N>` name (`squeue --name`), resumes the park mid-rung from
lease state, and never double-submits — the scancel + re-submit rung
walk is process-lifetime-independent. Never hand the still-queued job
to `backend_poll.py` (SLURM PENDING polls as `running` there; the
ladder would stall at high-eur). Orchestrator-side contract + the
killed-launcher recovery probes: `.claude/skills/issue/SKILL.md`
Step 6b.

The GCP ladder (`backends/router._gcp_ladder_specs`) is keyed on job
LENGTH (`_is_short_job`: known GPU-hours ≤ `EPS_GCP_SPOT_MAX_GPU_HOURS`,
default 2, OR `spec.extra["spot_tolerant"]`): **SHORT jobs — spot leads:**
spot A100-80 → spot A100-40 (fits-40 intents only) → flex-start A100-80 →
on-demand A100-80 → on-demand A100-40. **LONG / UNKNOWN-length jobs —
flex leads, NO spot:** flex-start A100-80 → on-demand A100-80 → on-demand
A100-40 (an unknown-length job is NOT short). The flex rung threads
`provisioning_model=FLEX_START` via `router._flex_start_rung`. The per-day
attempt cap `MAX_GCP_ATTEMPTS_PER_DAY = 16` (#1121; an attempt COUNT,
never a dollar cap) counts actual creates and stops issuing creates
mid-ladder when hit.

**Per-dispatch A100-40 fit gate (#1468).** Every A100-40 rung is gated
per-dispatch: a launch declaring `--min-gpu-mem-gb` strictly above
`gcp.A100_40_USABLE_GIB` (38.0 — conservative vs the card's measured
39.49 GiB; #1315: HF+vLLM co-residency died at engine init on the 40 GB
card) drops the A100-40 rungs (`gcp.a100_40_fallback_for_intent` returns
`None`); no declaration ⇒ ladder unchanged. A gated ladder never becomes
rung-less (A100-80 rungs, SLURM lanes, RunPod terminal rung unchanged).
Residuals: the gate never validates the PRIMARY machine (intent choice
fixes that; `capture-7b` #752 is the fix for a too-small primary).

**Width-aware rungs (#1121).** A dispatch declaring a shardable multi-GPU
axis via `--gpus N` (N ∈ {2,4,8}; `gcp.WIDTH_ELIGIBLE_INTENTS`) walks WIDE
`a2-ultragpu-{8,4,2}g` rungs (`gcp.WIDE_A100_80_BY_WIDTH`) BEFORE the base
ladder, WIDTH-MAJOR (every provisioning model at width w exhausted before
width w−1; intra-width the length-aware order applies; length classified
ONCE at the widest requested machine). Wide rung labels carry an `x<w>`
suffix; width-1 dispatches are byte-identical. H100 is EXCLUDED from the
width walk (preemptible quota exactly 8, no on-demand pool) —
`sweep-8g-h100` stays explicit-`--intent`-only. The queue-timeout /
queue-vanish / boot-loop poller machinery applies to wide rungs unchanged.
**Paid-fallback exposure:** width degradation fires only on CREATE-TIME
capacity misses — a flex create that QUEUES ends the ladder walk, so the
realistic fallback for a queued-but-stuck wide dispatch is the #783 queue
timeout failing over to a PAID RunPod pod at the REQUESTED width. A
workload that CANNOT re-shard off the realized width
(`realized_gpu_count` / `requested_gpus` on the marker/sidecar) must pin
its width; poller-side width-degrade re-entry is a named deferred
follow-up, NOT built.

**#1379 — explicit `sweep-8g-a100` degradation.** An EXPLICIT
`--intent sweep-8g-a100` dispatch (no `--gpus`) width-degrades too: the
router APPENDS degraded `a2-ultragpu-{4,2}g` rungs AFTER the intent's own
8g base rungs (`gcp.EXPLICIT_WIDE_DEGRADE_INTENTS`), falling back to
abundant partial-node capacity instead of starving on empty 8-GPU DWS
pools (the #825 create-miss shape). Opt out with `--width-required` (NOT
combinable with `--gpus` — exit 2,
`reason: width_required_gpus_conflict`). `sweep-8g-h100` is EXCLUDED
(cross-type degradation would change silicon; its type-preserving escape
is the RunPod 8×H100 terminal rung). Fence sizing: a `--max-run-duration`
sized at full width must either pass `--width-required` or size off the
2×-width p90 wall. A degradation is machine-readable as
`requested_gpus != realized_gpu_count`.

Tests of record: `test_ladder_short_job_spot_before_ondemand`,
`test_ladder_short_job_spot_miss_then_ondemand_order`,
`test_ladder_short_job_full_rung_order`,
`test_ladder_long_job_flexstart_before_ondemand_no_spot`,
`test_ladder_long_job_flexstart_miss_then_ondemand_order`,
`test_ladder_unknown_length_takes_long_branch`,
`test_flexstart_rung_threads_flex_provisioning`,
`test_max_gcp_attempts_per_day_is_sixteen`,
`test_workload_error_on_later_rung_fails_over_to_runpod`,
`test_width8_long_job_walks_wide_rungs_width_major`,
`test_width8_short_job_full_rung_order`,
`test_width_degradation_on_capacity_miss_lands_4g`,
`test_width1_ladder_byte_identical_explicit_gpus_none_and_matching`,
`test_width_ladder_never_emits_h100_machine`,
`test_workload_error_on_wide_rung_fails_over_to_runpod`,
`test_explicit_sweep8g_a100_long_ladder_order_with_degraded_rungs`,
`test_explicit_sweep8g_a100_short_ladder_order_with_degraded_rungs`,
`test_explicit_sweep8g_a100_degrades_to_4g_on_8wide_capacity_miss`,
`test_explicit_sweep8g_a100_width_required_pins_full_width`,
`test_explicit_wide_degrade_never_emits_h100_machine`,
`test_explicit_sweep8g_h100_ladder_unchanged_no_degradation`,
`test_explicit_sweep8g_a100_pinned_spot_walks_pinned_degraded_rungs`,
`test_explicit_sweep8g_a100_runpod_still_last_after_full_degraded_walk`,
`test_workload_error_on_degraded_rung_fails_over_to_runpod`,
`test_declared_width_none_for_non_sweep_intents` (all
`tests/test_router.py`); `test_width_required_threads_extra_flag`,
`test_width_required_with_gpus_exits_2_with_conflict_reason`
(`tests/test_dispatch_issue_cli.py`).

### Per-rung multi-zone fan-out

Each GCP rung walks every `us-central1` zone the rung's machine type is
offered in BEFORE the rung is treated as a capacity miss
(`GcpBackend.launch` iterates `[primary_zone, *fallback_zones]` filtered by
`zones_for_machine_type` against `MACHINE_TYPE_ZONE_AVAILABILITY`).
Per-family zone sets (pinned to the 2026-06-30 gcloud re-verification,
#774): A100-80 family `{a, c}` (a GCP-imposed limit, NOT a config gap —
adding `-f` would burn the daily attempt counter on guaranteed
`MACHINE_TYPE_NOT_FOUND` creates, #653; do not widen without a fresh
gcloud verification); H100 family `{a, b, c}`; A100-40 `{a, b, c, f}`.
Per-zone fan-out visibility (#774): the `epm:backend-selected` marker's
`attempts[].evidence` carries `per_zone_attempts` (one record per zone
tried) + a `zones_attempted_summary` one-liner — emitted only when
populated (happy-path attempt entries keep the pre-#774 shape). Closes the
#763 misdiagnosis where only the last zone's `stderr_tail` survived.
Contract tests:
`tests/test_router.py::test_route_attempt_evidence_field_round_trips_per_zone_attempts`,
`tests/test_gcp_backend.py::test_zone_fanout_all_zones_exhausted_evidence_carries_per_zone_outcomes`.

### Coverage scope (current) — both the synchronous and async crash paths

Both GCP workload-crash detection paths fail over to RunPod:

- **Synchronous `route()`-time** — a `gcp.GcpWorkloadError` raised inside
  `route()` short-circuits straight to RunPod, per Part B (#658).
- **Async poller** (#659) — the COMMON production crash, surfacing minutes
  into the run: when `GcpBackend.poll` resolves a real workload crash to
  `current_phase == "terminal_workload_failed"` (the `eps/phase==failed` +
  write-once `eps/workload_started` sentinel discrimination) with
  `status == "dead"`, `_is_gcp_async_workload_failure` matches and
  `_failover_dead_gcp_to_runpod` re-dispatches on RunPod
  (`gcp_workload_failover_runpod_async`), idempotency lease-backed. A
  setup/boot/secrets failure surfaces `terminal_setup_failed` — excluded,
  so a broken-boot VM never re-crashes on RunPod.

Part A covers BOTH crash modes (the EXIT trap fires on any non-zero exit).
ONE named exception (#1596/#1601): a workload-class failure on the
queue-vanish / queue-timeout ON-DEMAND RETRY create mints the
non-re-drivable terminal `gcp_workload_failed_on_ondemand_retry` instead
(RunPod just refused for capacity — cascading back would ping-pong); see
§ FLEX_START queue-vanish.

### FLEX_START queue-timeout → RunPod (#783/#778)

**Trigger:** a GCP create SUCCEEDS but the instance stays in the
FLEX_START capacity queue (`status=PENDING`, polled as
`current_phase="pending"`) past `EPS_GCP_QUEUE_WAIT_SECONDS` (default
600s, read at call time; missing/invalid → 600) — a state distinct from a
create-time capacity miss and a workload crash; pre-#783 nothing bounded
the wait (route() had already returned a handle).
`backend_poll._maybe_escalate_gcp_queue_timeout` ages the `"pending"`
phase against the floor using the SAME sidecar phase clock as the #669
wedge (disjoint `current_phase` values, so no collision; the clock
re-stamps on ANY phase change, so a dequeue resets it; no
reachability-alarm conjunction — a stuck queue has no live VM).

**Action:** past the floor the poll is rewritten to `status=dead` /
`current_phase=terminal_queue_timeout` (`_is_gcp_queue_timeout`);
`_failover_queued_gcp_to_runpod` (1) best-effort `teardown`-DELETEs the
still-queued instance (it is live server-side and could dequeue later as
an orphan; a failed delete degrades to the `gcp_audit.py` janitor), then
(2) re-dispatches on RunPod via the shared `_failover_gcp_to_runpod` core
(same lease+sentinel exactly-once bound, sidecar re-point, terminal-JSON
contract), `reason: gcp_queue_timeout_failover_runpod`. An ADDITIONAL
advance trigger, NOT a lane-precedence change; the failover leg does NOT
burn `MAX_GCP_ATTEMPTS_PER_DAY` (the counter bumps only on a create).

**On-demand retry (#1601/#779):** a capacity-class RunPod refusal with
CLEAN provision residue retries the GCP STANDARD rungs before minting the
re-drivable `no_compute_available` terminal, labeled
`reason: queue_timeout_gcp_ondemand_retry` — the #1596 machinery verbatim
(see § FLEX_START queue-vanish); the retry leg re-enters the rung and
burns attempts normally. Documented residual: a FAILED best-effort
teardown can leave the timed-out PENDING instance alive under the same
name — the exactly-once lease still bounds paid launches and the janitor
reaps the orphan.

**CPU-intent scope (#677/#747):** `cpu-bigmem` PENDING is EXCLUDED (no
cheap RunPod lane → ordinary dead path); `cpu-small` / `cpu-mid` eligible.

### FLEX_START queue-vanish → RunPod (#1116/#1112)

**Trigger:** a DWS-queued FLEX_START instance is dropped SERVER-SIDE —
create reports success, the instance sits PENDING, then DISAPPEARS from
instances-list with NO delete operation. `gcp.poll` maps the post-vanish
describe-404 to `status="dead"` / `current_phase="terminal_instance not
found"` — attribute-blind, pre-#1116 read as an ordinary crash or poisoned
the #1029 boot-death streak (mislabelling a pure CAPACITY event). #1112:
`route()` advances the ladder only on CREATE-time capacity errors, so every
relaunch re-booked the same dead flex rung.

**Two-arm discriminator (#1815)** (`backend_poll._vanish_arm_for`, both
arms INCARNATION-KEYED — phase-clock records carry
`last_phase_incarnation`, and a record stamped by a DIFFERENT launch
incarnation reads as ABSENT for all three clock readers, so a prior
attempt's phase can never satisfy, block, or age a discriminator):
**Arm P (`pending-clock`, #1116):** a same-incarnation (or legacy) clock
record whose `last_phase` reads `"pending"` — the instance was last
observed still queued (READ-ONLY clock use: no aging floor, no streak).
**Arm N (`never-ran-young-flex`, #1815):** NO clock record for THIS
incarnation AND the handle carries `provisioning_model == "FLEX_START"` +
create evidence (non-empty `job_id`) + a young `gcp_launched_ts`
(< `EPS_GCP_QUEUE_VANISH_MAX_AGE_SECONDS`, default 1500 s — under the #935
done-grace window, so a never-polled COMPLETED run's post-grace DELETE can
never satisfy it), each conjunct fail-safe on absence; reconnect handles
carry both arm-N inputs post-#1815. A same-incarnation NON-pending clock
means the instance RAN — #659/#1029 own that shape.

**Action:** on either arm `_maybe_escalate_gcp_queue_vanish` rewrites the
poll to `terminal_queue_vanish` and `_failover_vanished_gcp_to_runpod`
fails over on the FIRST occurrence via the shared core
(`teardown_first=False` — the instance record is already gone),
`reason: gcp_queue_vanish_failover_runpod`; marker evidence carries
`vanish_arm`. The failover leg does NOT burn daily attempts. Ordering: the
vanish branch runs BEFORE the #1029 boot-loop recorder in `main()` (its
early return keeps a pure capacity miss from poisoning the boot-death
streak).

**On-demand retry after a clean-residue RunPod refusal (#1596/#1112).**
When the vanish failover's RunPod terminal rung raises
`NoComputeAvailableError` AND the #1490 residue reclaim reports CLEAN
(`no-residue` / `torn-down` / `pre-existing` / `foreign-created` — never
`leaked`), the poller retries the GCP ladder's STANDARD (on-demand) rungs
itself (`backend_poll._retry_gcp_ondemand_after_capacity_refusal` →
`router.retry_gcp_ondemand_after_queue_vanish`; H100 intents refused up
front). The retry BURNS `gcp_attempts_today` (it IS a fresh create); gate
= clean residue ONLY, no evidence-text classifier. Exactly-once: the rung
stamps `gcp_failover_of` on the lease IN THE SAME FLOCK TRANSACTION as
the create and short-circuits on a matching stamp; the RunPod rung's
in-flock check matches the gcp-stamped lease too — neither a crashed
sidecar write nor a concurrent triggerer can double-provision. On success
the sidecar is re-pointed at the NEW gcp handle (job_id is the readback
discriminator). Reason strings: the LAUNCH is labeled
`queue_vanish_gcp_ondemand_retry`; a WORKLOAD-class failure on the retry
create mints the non-re-drivable `gcp_workload_failed_on_ondemand_retry`
(the named Part B exception); retry exhaustion / cap-hit falls through to
the re-drivable `no_compute_available` terminal (log_tail keeps the
original RunPod refusal evidence). Since #1601 the queue-timeout caller
arms the SAME retry; the #659 workload-crash and #1029 boot-loop callers
keep the retry OFF.

**CPU-intent scope:** `cpu-bigmem` never rewrites/fails over (the guard
gates the rewrite itself); `cpu-small` / `cpu-mid` eligible.

Named residuals (accepted): (1) a transient not-found flicker on a LIVE
queued instance fires the failover ONCE (lease-bounded; the orphan is
bounded by its fence + janitor); (2) PENDING is the one state where a
manual `gcloud instances delete` AUTO-fails-over (lease-bounded); (3)
remaining inert shapes — non-FLEX/unknown provenance (young),
same-incarnation observed-past-pending, a first dead tick older than the
age floor — fall to the #1029 streak / ordinary dead path (slower, never
wrong-direction); (4) a dequeue→boot→crash→DELETE entirely between two
polls is labelled a queue vanish — same destination as #659, different
label; Part A diagnostics upload regardless. Hardening path: if a
sanctioned actor ever starts deleting LIVE PENDING instances, the trigger
needs an operations-log delete-op check — a re-plan, not a tweak.

### Coupled multi-arm dispatch stall → down-width split (#1633)

Every trigger in this Part re-shapes or fails over ONE dispatch's machine.
None of them can DECOMPOSE a dispatch that bundles arms of DIFFERENT
minimum GPU widths behind one provision — the #1112 failure mode:
1×-runnable arms held ~14 h behind a coupled 4×/8× provision during a
drought while the 1× shape had stock. On a SUSTAINED capacity stall
(≥ ~1 h queued / stocked-out across rungs) of a coupled multi-arm
dispatch, the owning orchestrator splits out and probes the
narrowest-runnable arms as their own dispatch(es); the wide arm keeps its
ladder walk. The plan-side duty (per-arm MINIMUM runnable width + the
pre-registered split) lives in `.claude/rules/plan-compute-sizing.md`
§ Multi-arm min-width + stall-time down-width split; this section is the
dispatch-time cross-reference.

### Pre-workload boot-loop → RunPod (#1029)

**Trigger:** N (default 2, `EPS_GCP_BOOT_DEATH_STREAK_N`) CONSECUTIVE
same-rung pre-workload deaths for the same issue (the #763 shape: a rung
re-selected by every relaunch, each create dying minutes post-insert with
no crash diagnostics). The first death keeps its one free retry, so a lone
transient setup death / spot preemption is unchanged. **Record:** the
streak lives in the durable per-issue lease
(`~/.eps-routing/issue-<N>.json`, `Lease.gcp_boot_death_streaks`), keyed
per (issue, rung) and per launch INCARNATION (`handle.job_id`; fallback
`(attempt_id, gcp_launched_ts)`; attempt_id alone forbidden — #763's five
creates shared one), same-UTC-day scoped, RESET on any POSITIVE workload
signal (running `workload`/`relaunched_workload`, a
`terminal_workload_failed` crash, a #935 `done` shape; never a
pre-workload phase). **Classify:** a pre-workload death is deterministic
(`terminal_setup_failed` — the `workload_started` sentinel discrimination)
OR heuristic (a YOUNG `terminal_terminated` / `terminal_instance not
found`, launch→observation age below `EPS_GCP_BOOT_DEATH_MAX_AGE_SECONDS`,
default 1500s — post-DELETE polls are attribute-blind, so age is the only
signal). **Fire (poller):** at streak ≥ N the poll is rewritten to
`terminal_boot_loop` (`_maybe_escalate_gcp_boot_loop`) and
`_failover_boot_looped_gcp_to_runpod` reuses the shared core with
`teardown_first=False`, `reason: gcp_boot_loop_failover_runpod`; evidence
carries `boot_death_streak` + `gcp_ladder_rung`. **Skip (route side):**
`_attempt_one_gcp_rung` SKIPS a rung whose same-UTC-day streak is ≥ N on
the auto chain (outcome `boot_loop_rung_skipped`) WITHOUT bumping
`gcp_attempts_today`; an explicit `backend: gcp` pin is exempt. If every
GCP rung skips, the chain proceeds to SLURM then the RunPod terminal rung.
**CPU scope:** `cpu-bigmem` RECORDS but never rewrites (the route()-side
skip is its breaker → the typed `cpu_exhausted_no_runpod_lane` terminal);
`cpu-small` / `cpu-mid` fail over to RunPod CPU as usual. **Policy
delta:** a LONE `terminal_terminated` still never fails over (the #669
exclusion preserved), but N≥2 consecutive sub-floor early deaths on ANY
rung — including a spot rung — advance.

Operational notes: (a) a sub-N boot death lands `failure_class: code` on
the ordinary dead path, re-driven by the per-issue session's crash-fix
loop — NOT the watcher capacity-retry pass (a breaker targets a loop; a
loop requires a re-driver). (b) A genuine FAST workload crash observed
only post-DELETE counts toward the streak and, at N, fails over under the
boot-loop reason — same action/destination as #659, different label. (c)
Every TERMINATED+`failed` poll issues one extra `_workload_started`
guest-attribute probe (probe failure keeps `terminal_terminated`).

### Workload-phase FLEX preemption on checkpoint-less legs → escalate to STANDARD (#1999)

A DISTINCT sibling of the boot-loop breaker: when a GCP FLEX_START
workload has ALREADY STARTED (positive workload signal) and is PREEMPTED
mid-run, the per-issue session's crash-fix/relaunch loop escalates the
next attempt to STANDARD (on-demand) provisioning rather than re-booking
FLEX_START on the same rung. Trigger conjuncts, all three required:
(1) workload-STARTED preemption — `terminal_terminated` /
`terminal_workload_failed` preceded by a positive workload signal;
(2) wall time ≥ ~2h at kill (`EPS_FLEX_ESCALATE_MIN_WALL_H`, default 2.0);
(3) NO mid-run checkpoint the leg can resume from — declared at plan time
(§9 row `resume_from_checkpoint: no`/`yes`) or inferred from the phase's
kill-recovery contract; a `yes` leg re-books FLEX as today.
**Disposition:** the relaunch is dispatched with `dispatch_issue.py launch
... --provisioning-model STANDARD`; the FLEX rung is NOT permanently
blocklisted — the escalation binds to the re-launch after the matching
preemption. Distinctions: vs #1029 (that clause = ≥2 pre-workload deaths,
fails over to RunPod; this = ONE workload-phase preemption, escalates
IN-LADDER, no RunPod pivot, no cap changes); vs the #680 ladder (spot is
already barred for long jobs; FLEX is kept because it is non-preemptible
once running — this adds the length + no-checkpoint refinement); vs #659
(that fires on any workload crash and leaves GCP). **Rule-text-only for
v1; mechanization is a follow-up** (a
`--flex-escalate-after-workload-preemption` flag or a durable-lease record
analogous to `gcp_boot_death_streaks`). Incident: #1739 (five attempts to
land a checkpoint-less leg before the manual STANDARD switch). Critic
enforcement: Methodology lens item 16 FLEX ESCALATION EXTENSION
(`.claude/rules/critic-lens-reference.md`); no verify_plan.py backstop in
v1.

### Remaining gap — the hung-but-RUNNING / frozen non-terminal phase (#667)

Neither failover path fires for a GCP VM that HANGS without publishing a
terminal `eps/phase`: a VM whose guest networking dies (DHCPv4 loss) — or
whose workload wedges without the EXIT trap firing — stays `RUNNING` with
`eps/phase` frozen at a NON-terminal value (e.g. `workload`), which
`GcpBackend.poll` classifies `running` forever, so neither the sync nor
async predicate matches and the run bills until a HUMAN manually pivots to
`--backend runpod`. Closing this (escalating a frozen non-terminal phase
past a drain-timeout to a terminal wedged state) is a pending
`kind: infra` follow-up. (See also the #491 `bufio.Scanner: token too
long` zombie in `.claude/rules/gotchas.md`, a sibling hung-but-RUNNING
mode recoverable in place via SSH relaunch.)

### Live-diagnosis access to a GCE instance (SSH / serial / Monitoring)

- **SSH: external-IP first.** Default `gcloud compute ssh` tries an IAP
  tunnel the `eps-gcp` configuration is NOT authorized for — pass
  `--tunnel-through-iap=false` (or plain `ssh` to the external IP) up
  front; fall back to the serial console when guest networking is dead.
- **Serial console is the always-available read** (`gcloud compute
  instances get-serial-port-output --configuration=eps-gcp`); the #854
  eager `[crash-persist]` lines land there.
- **The Cloud Monitoring API is not enabled** on `eps-persona-gpu-jun2026`
  — diagnose via serial console + SSH.

### Gate-park zombie — RUNNING + terminal `eps/phase` (#908/#935)

A GCP workload that exits CLEANLY at a blocking gate (or finishes) leaves
the VM RUNNING only within the bounded #935 done-grace window (default
90 min, `EPS_GCP_DONE_POWEROFF_GRACE_SECONDS`; sentinel draining is why it
stays up). Three guards close the former unbounded leak (#763):
(0) BOUND (in-VM, #935) — the done-grace self-poweroff in the success
tail: a countdown that aborts on the operator keepalive file
(`/workspace/logs/issue-<N>-keepalive`) or an `eps/phase` re-publish,
best-effort persists the UNDRAINED sentinel set to HF
`issue<N>_done/<attempt_id>/` at expiry (`eps/done_persist` breadcrumb on
a SEPARATE key), then powers off UNCONDITIONALLY. A
TERMINATED+`eps/phase=done` instance polls as `status=done` /
`workload_done_self_poweroff`; finalize needs `--skip-confirm-artifacts`.
(1) PRIMARY — `/issue` Step 6d.4 gate handlers tear the instance down via
`dispatch_issue.py finalize --skip-confirm-artifacts` after the sentinel
drain (PARK-mode gates finalize BEFORE the park; auto-resolving gates
after resolution). Never wait out the grace — the in-VM bound is the
dead-orchestrator fallback, not the plan.
(2) BACKSTOP — `gcp.reconnect_or_none` refuses a RUNNING instance whose
`eps/phase` ∈ `_ZOMBIE_GUEST_PHASES` ({done, failed,
finalize_failed_artifacts_ok, wedged}) and
`_stale_named_instance_or_none` returns it as deletable, so the next
launch reclaims + creates fresh (skip/delete sets pinned identical,
`tests/test_gcp_backend.py`, #632; a guest-attribute probe failure raises
`GcpProbeError` — never a reconnect, never a delete, #535). Relaunch
contract (the #491 SSH-relaunch recipe in `.claude/rules/gotchas.md`): a
manual same-VM relaunch MUST re-publish `eps/phase=workload` BEFORE
resuming work, and a relaunch on a VM whose first workload published
`done` must re-publish within the done-grace window (or touch the
keepalive) — else the countdown powers the VM off. The #667 frozen-phase
gap is UNCHANGED (the done-grace countdown never arms there).

**Manual-pivot runbook line (#909):** a manual RunPod pivot carrying
`--workload-cmd` must ALSO pass `--execute-workload` — without it the
launch is provision-only (the pod boots, nothing runs;
`workload_executed: false`); the alternative executor is dispatching the
experimenter on the provisioned pod.

### CPU intents: cheap CPU lanes + the scoped #677 terminal (#747)

The GCP→RunPod failover (capacity AND workload-crash, sync AND async)
extends to the CHEAP CPU intents. #677 made EVERY CPU intent a hard
terminal (RunPod was GPU-only); #747 adds a RunPod CPU lane
(`deployCpuPod`) for the cheap intents and SUPERSEDES that terminal for
them ONLY:

- **`cpu-small` / `cpu-mid`** (mapped in
  `router.RUNPOD_CPU_INSTANCE_FOR_INTENT`) fall over GCP cheap CPU →
  RunPod CPU on GCP exhaustion (capacity) OR a workload crash (sync
  `_runpod_terminal_rung` + async `_is_gcp_async_workload_failure`, both
  keyed on the SAME map). The re-dispatch carries `--intent cpu-small` /
  `cpu-mid`, resolved by `gpu_heuristics.resolve_cpu_intent` to the RunPod
  instance id (`cpu3g-2-8` / `cpu3c-8-16`) on the `pod_lifecycle`
  provision path (`runpod_api.create_cpu_pod`). RunPod CPU pods are
  on-demand only; a CPU no-capacity miss surfaces `RunPodNoCapacityError`
  → terminal, re-drivable by the watcher's capacity-retry pass.
- **`cpu-bigmem`** (ABSENT from the map — the >50 GB analysis lane; but
  see the #2028 banner: `cpu-bigmem` gained the `cpu5m-16-128` RunPod row,
  so the typed terminal now fires only for a future UNMAPPED CPU intent)
  keeps the #677 typed `CpuExhaustedNoRunpodLaneError` /
  `reason: cpu_exhausted_no_runpod_lane` terminal as the fail-loud floor —
  never auto-retried (the watcher keys on `no_compute_available`).
- **Footprint feasibility gate + disk threading (#1010, incident #958).**
  A plan-STATED footprint (`spec.extra["boot_disk_gb"]` / `["min_ram_gb"]`,
  from `dispatch_issue.py --boot-disk-gb` / `--min-ram-gb`) is checked at
  `_runpod_terminal_rung` against `router.RUNPOD_CPU_INSTANCE_CAPS`
  (probe-verified effective `containerDiskInGb` caps + fixed RAM); an
  unsatisfiable footprint refuses BEFORE any RunPod API call with the
  typed `CpuFallbackInfeasibleError` /
  `reason: cpu_fallback_infeasible_for_plan` (a
  `CpuExhaustedNoRunpodLaneError` subclass; NOT in
  `TRANSIENT_CAPACITY_REASONS`). A feasible `boot_disk_gb` is THREADED
  into the provision argv (`--container-disk-gb max(50, boot_disk_gb)`),
  and `pod_lifecycle`'s CPU branch clamps a default-band over-cap payload
  to the instance cap while an explicit above-band request refuses
  pre-API. ADOPTION: launch composers pass `--boot-disk-gb` whenever a CPU
  stage sizes disk > 50 GB and `--min-ram-gb` whenever it sizes RAM >
  16 GB — flag-less launches keep today's behavior.

Tests of record:
`tests/test_router.py::test_router_cpu_small_capacity_miss_falls_over_to_runpod`,
`tests/test_router.py::test_router_cpu_intent_capacity_miss_no_runpod_fallback`,
`tests/test_router.py::test_gcp_ladder_cpu_small_short_yields_spot_then_ondemand`,
`tests/test_backend_poll.py::test_async_cpu_small_handle_fails_over_to_runpod`,
`tests/test_backend_poll.py::test_async_failover_skips_cpu_gcp_handle`,
`tests/test_runpod_api_retry.py::test_deploy_cpu_pod_renders_instanceid_mutation`.

## Part C — RunPod RUNNING-but-no-port host wedge (#664)

The RunPod sibling of the GCP hung-but-RUNNING wedge. RunPod
`desiredStatus` is decoupled from `runtime.ports`: a degraded host keeps
the pod RUNNING (and billing) while `runtime.ports` is empty, so
`runpod_api._parse_pod` yields `ssh_host=None`. `resume_pod` is
HOST-PINNED (no host reselection) — a stop+resume returns to the SAME dead
host. `--refresh-from-api` is a NO-OP here (the port is platform-absent,
not stale — that flag fixes the #488 stale-port case).

**Detection** (`backend_poll._maybe_escalate_runpod_wedge`): a RunPod
handle whose LIVE `desiredStatus` stays RUNNING with null/empty
`runtime.ports` past `RUNPOD_WEDGE_K_SEC` (default 900s, env
`EPM_RUNPOD_WEDGE_K_SEC` — above `wait_for_ssh`'s 600s window + retry
margin, so a healthy mid-resume pod never trips) is rewritten to
`status=dead, current_phase=RUNPOD_WORKLOAD_WEDGED_PHASE`. Within K, an
SSH-dead poll is REWRITTEN to `status=running`
(`RUNPOD_WORKLOAD_OBSERVED_PHASE`) so the orchestrator keeps polling until
the wedge matures. The no-port clock rides the sidecar `extra`
(`runpod_noport_first_seen_ts`), fail-soft (atomic tmp+rename, never
raises on malformed values), CLEARED the moment a public port appears or
the pod leaves RUNNING.

**Recovery** (`backend_poll._failover_wedged_runpod`, gated on a PER-CELL
inputs-on-HF gate): `_wedged_run_inputs_on_hf` classifies each selected
cell from ONE fresh `list_repo_files` against the EXACT expected file set
— COMPLETE (raw+store exact sets on HF) is safe, a PARTIAL cell BLOCKS, a
NOT-YET-RUN cell does not block (rerunnable from verified earlier inputs).
With ZERO partial cells, `terminate_pod` stops the billing leak and a
FRESH pod is re-provisioned (NOT a host-pinned resume) + the dispatcher
resumed idempotently (the fresh pod's dispatcher skips HF-complete cells).
Any partial cell → NO terminate; surface a `failure_class: infra` block
(`reason=runpod_wedge_inputs_unverified`) so a human decides. This is the
irreversible auto-terminate, analogous to the Step 8
auto-terminate-after-upload-PASS precedent — data-safe because per-cell
incremental upload + the gate make every COMPLETE cell's data already
present on HF.

**Idempotency = a DURABLE lease + a sentinel, as the GCP analogue.** The
wedge failover guards its terminate + re-provision with TWO records keyed
to the wedged ATTEMPT identity (pod_name / job_id / runpod_attempt_id —
the per-launch `runpod_attempt_id` is what stops a stale record from
blinding a fresh adopted attempt; pre-#1668 the identity was degenerate
per issue and "exactly once per wedge" silently read as "once per issue
forever", #1586). #1668 also adds a PRE-TERMINATE SSH liveness cross-probe
(`_runpod_wedge_liveness_probe`, fail-open) between the inputs gate and
the terminate: a matured no-port wedge whose pod still answers SSH is a
sustained API port-misreport on a HEALTHY pod — the failover clears the
wedge clock and returns a non-terminal running JSON instead of
terminating; probe failure/error proceeds to terminate. Records: (1) the
AUTHORITATIVE durable lease at `~/.eps-routing/`
(`Lease.runpod_wedge_failover_of`) — survives the EDQUOT /
read-only-fs mode that fails the `.claude/cache` sidecar and sentinel
together; (2) the `.claude/cache` wedge sentinel as the fast path.
**Relaunch error mapping:** `_relaunch_fresh_runpod` maps a no-capacity
RunPod to a terminal infra JSON `reason=no_compute_available` (re-drivable
by the watcher's capacity-retry pass) and a sidecar-write failure (EDQUOT)
to `reason=sidecar_persistence_failed` (durable lease stamped to bound the
relaunch) — a wedge failover always honors the poller's terminal-JSON
contract instead of crashing `main()`.

The data-safety PREREQUISITE is the per-cell incremental upload in
`scripts/issue664_dispatch.py` (`_upload_cell_artifacts`, fired the moment
each cell's worker succeeds). **The per-cell HF surface includes the
marker-slot stats for MARKER cells (#689):** `_upload_cell_artifacts`
uploads `marker_slot_stats.json` and `_classify_cell_hub_state` requires
it for a marker cell to read "complete", so a fresh auto-migrated pod can
HYDRATE it from HF (`_hydrate_marker_slot_stats_from_hf`) before the A7
readability assert. The complementary `cmd_resume` advice split
(`pod_lifecycle.py`: a still-null resume names terminate+re-provision, not
`--refresh-from-api`) is the interactive sibling; the report-only
`running_no_port` flag in `pod_audit.py` is the fleet-level visibility
backstop (never auto-terminates).

### Part C watcher backstop — the poller-DEAD case (#692/#770)

The poller-side detect+recover above runs ONLY while the per-issue poll
loop is alive. When that loop has DIED (crashed session, OOM-killed
bg-Bash chain, VM reboot), `_maybe_escalate_runpod_wedge` never runs and
the #664 billing leak goes undetected. The `autonomous_session_watch.py`
pod-safety pass (every 10 min, session-independent) closes that gap with a
wedge arm in `_process_pod`:

- **Compose, do NOT re-define.** The arm calls the SAME raw predicate
  `backend_poll._pod_is_runpod_runtime_wedged(info)` the poller uses, on
  the live `PodInfo` the API list pass already fetched, and the SAME
  imported `backend_poll.RUNPOD_WEDGE_K_SEC` (never a duplicated literal).
- **A DEDICATED wedge clock.** The maturity floor ages against
  `wedge_first_seen` (stamped at wedge ONSET, cleared on any non-wedge
  tick and on a pod_id change), NOT the pod-incarnation `first_seen`. A
  ≥ 2 consecutive-confirmed-checks miss guard backs it, so a transient API
  blip never stops a pod.
- **ALERT by default; TERMINATE+FAILOVER only when provably safe (#770).**
  Past K + confirmed, the arm posts a once-per-episode alert UNLESS the
  same inputs-on-HF gate confirms zero partial cells AND a TRI-STATE
  keep-running read (`_wedge_keep_running` → `True | False | "unknown"`)
  returns the literal `False` AND — the #1667 owner-liveness guard — a
  TRI-STATE owner probe (`_wedge_owner_live`: recent non-watcher markers
  on the issue, or a live registered/cwd-mapped session with a fresh
  transcript, window `EPM_WEDGE_OWNER_RECENT_H` default 2h) ALSO returns
  the literal `False`. Every uncertainty path (no handle, HF error, tag
  present, tag-read `"unknown"`, inputs unverified) is ALERT-only, and a
  live/"unknown" OWNER demotes the eligible terminate to a
  once-per-episode DEFER marker with the wedge clock PRESERVED, so a
  genuinely dead owner still gets the terminate once activity quiets
  (#1586: a wedge terminate mid-crash-fix-round destroyed live run state —
  "poll loop dead" is a precondition of the arm firing, NOT evidence the
  owning session is dead). Kill switch `EPM_DISABLE_WEDGE_OWNER_GUARD=1`
  restores the pre-#1667 terminate.
- **TERMINATE + re-provision — the SAME recovery the poller owns (#770).**
  For the ONE provably-safe case (matured + confirmed wedge +
  `keep_running=False` + inputs verified + no live owner + a
  reconstructable run handle) the watcher routes the SAME irreversible
  terminate + fresh re-provision via the `_wedge_failover` helper (read
  handle+sidecar from the persisted sidecar, synthesize a minimal result
  shim, terminate, re-provision FRESH). **Bounded-once** via the SHARED
  durable lease + sentinel (`_runpod_wedge_already_handled`, called INSIDE
  `_failover_wedged_runpod`), so poller-side and watcher-side firings on
  the same wedge are mutually exclusive. (A reversible `pod.py stop`
  cannot heal a host-pinned dead host — `resume_pod` returns to the SAME
  host — hence the promotion to terminate.) `_wedge_failover` returns
  `(outcome, terminal_json)`; the dispatch maps the poller's terminal-JSON
  contract to five outcomes:
  - `failover` (fresh pod re-provisioned) → a generic `epm:progress` note.
  - `already-handled` (bounded-once short-circuit — OR the sidecar-binding
    defense below) → a generic note, no terminate.
  - `no-capacity` AND `blocked` are BOTH terminal infra JSONs: the watcher
    MIRRORS the poller's path — posts `epm:failure v1` carrying the EXACT
    `failure_class`/`reason` from `terminal_json` (the whitespace-token
    shape `_parse_failure_fields` reads) AND `set-status <N> blocked` —
    which lets the capacity-retry pass re-drive the re-drivable
    `no_compute_available` block and leaves non-capacity `blocked` reasons
    parked. Marker text states the terminate-state per reason:
    `runpod_wedge_inputs_unverified` is PRE-terminate (pod NOT terminated,
    still RUNNING); `sidecar_persistence_failed` and
    `runpod_wedge_relaunch_spec_missing` are POST-terminate;
    `runpod_wedge_failover_error` is an UNEXPECTED raise confirmed by a
    `get_pod_by_name` liveness probe to have happened AFTER
    `terminate_pod` (pod GONE). The terminate decision lives entirely
    inside `_failover_wedged_runpod`; the watcher never calls a SECOND
    terminate.
  - `alert` (NO reconstructable handle / sidecar parse failure, OR a
    PRE-terminate raise where the liveness probe finds the pod STILL ALIVE
    — or the probe itself raised → UNCERTAIN, bias SAFE) → ALERT-only,
    NEVER a blind terminate, wedge clock PRESERVED (the still-RUNNING pod
    re-matures next tick; never a FALSE terminal record while the pod
    bills). A POST-terminate raise routes to `blocked`
    `runpod_wedge_failover_error`, NOT `alert`. Fail-soft either way. The
    sidecar-names-a-DIFFERENT-pod case maps to `already-handled` (a
    re-pointed sidecar means a revived poller already failed the wedge
    over to a fresh, healthy pod the watcher must not terminate).

  The reversible `_stop_pod` is RETAINED for the status-class DONE
  escaped-pod arm, NOT the wedge arm.
- **Sidecar-binding defense — the fresh-pod race.** Between
  `_wedge_inputs_safe`'s sidecar read and `_wedge_failover`'s re-read, a
  revived poller could have ALREADY failed the wedge over and re-pointed
  the sidecar at a FRESH, HEALTHY pod (the bounded-once lease is keyed on
  the FRESH handle's identity and would not catch this). Defense:
  immediately after the re-read, `_wedge_failover` asserts the
  freshly-read `handle.pod_name == info.name`; a mismatch returns
  `already-handled` and NEVER calls `_failover_wedged_runpod`.
- **DONE-task ordering.** A wedged pod whose task is at a DONE status
  (`completed` / `awaiting_promotion` / `archived`) FALLS THROUGH to the
  existing status-class DONE auto-stop arm; the wedge arm only handles
  non-DONE (live-work) statuses.

Tests of record: `tests/test_autonomous_session_watch_wedge.py`
(decision table + boundaries, tri-state gate, fail-closed inputs gate,
state round-trip, pod_id reset, alert dedup, DONE fall-through,
terminate-failover decision + invariant, `_wedge_failover` outcome
mapping, dry-run no-side-effects,
`test_wedge_no_capacity_emits_failure_marker_and_blocks_redrivable`,
`test_wedge_blocked_emits_failure_marker_and_blocks_not_redrivable`,
`test_wedge_blocked_marker_text_terminate_state_by_reason`,
`test_wedge_failover_sidecar_pod_name_mismatch_is_already_handled`,
`test_wedge_failover_sidecar_pod_name_match_proceeds`,
`test_wedge_failover_raise_after_terminate_routes_to_blocked`,
`test_wedge_failover_raise_after_terminate_routes_to_blocked_redrivable`,
`test_wedge_failover_preterminate_raise_pod_alive_degrades_to_alert`,
`test_wedge_failover_preterminate_raise_probe_raises_degrades_to_alert`,
`test_wedge_failover_preterminate_raise_does_not_falsely_block_live_pod`);
`tests/test_runpod_wedge_detection.py::test_pod_is_runpod_runtime_wedged_predicate`.

## Part D — RunPod CUDA-IMA repeat host wedge (#775)

The crash-signature sibling of Part C's no-port wedge. A RunPod H100/H200
can wedge at the DRIVER level: a vLLM workload crashes with a CUDA
illegal-memory-access (`CUDA error: an illegal memory access was
encountered` / `EngineDeadError` / `Engine core proc … died unexpectedly`),
the experimenter's default `failure_class: infra` recovery does an
IN-PLACE SAME-POD respawn (it never terminates the pod), and the
SAME-signature crash recurs on the same physical GPU. Part C does NOT
catch this — the CUDA-IMA pod keeps its port + stays RUNNING. Part D
automates the fresh-host pivot on the SECOND same-signature crash.

**Shape-dependent carve-out (#1092):** before reading a repeat IMA as a
host defect, check the gotchas differential — if the crash follows the
WORKLOAD shape (identical code clean on A100 + a same-pod short-prompt
probe clean), a fresh host is EXPECTED to re-hit it and the fix is the
default-off engine knobs, not a pivot (a knobs-on rerun that still IMAs
falsifies the shape diagnosis — revert to the Part D pivot) — see
`.claude/rules/gotchas.md` § "vLLM-on-H100 CUDA illegal-memory-access
under heavy shared-prefix caching".

**Detection** (`backend_poll._maybe_escalate_runpod_cuda_ima`, the
repeat-based sibling of the time-based no-port wedge). The signal is
`PollResult.crash_signature` — the WIDE 500-line probe tail (NOT the
5-line `log_tail_excerpt`, which routinely truncates a vLLM traceback);
`poll_once` captures it on a `status="dead"` poll from the wide surface it
already fetched. Escalation conjuncts:

1. **CUDA-IMA signature on the WIDE surface** (`CUDA_IMA_SIGNATURE`,
   within-line alternatives, no `re.DOTALL`).
2. **A prior same-signature crash recorded THIS RUN** — the sidecar
   `extra["runpod_cuda_ima_last_seen"]` record, with the prior
   `epm:failure` marker as a cross-pod fallback source
   (`_prior_failure_marker_is_cuda_ima`) so a sidecar wipe between pods
   does not lose the record. CLEARED on any non-dead / non-CUDA-IMA poll —
   only a SECOND crash with no intervening healthy poll counts.
3. **EXCLUSION — no OUR-code traceback frame**
   (`failure_classifier.OUR_CODE_FRAME`). A CUDA-IMA surface that ALSO
   traces through `src/explore_persona_space/` or `scripts/` is a
   deterministic CODE bug — ordinary dead path (`failure_class: code`)
   without spending the bounded pivot.

A FIRST crash records the signature and falls through (the in-place
respawn gets its one chance); a SECOND same-signature repeat rewrites to
`current_phase=RUNPOD_CUDA_IMA_WEDGED_PHASE`
(`_is_runpod_cuda_ima_failure`). The predicate keys on the crash
SIGNATURE across the run (NOT pod_id): strictly more robust — it survives
the stop/resume host-rewrite edge case and the half-bootstrap fresh-pod
path; the once-more bound is the safety against over-counting. **Wiring —
BEFORE the no-port block:** the no-port within-K path rewrites
`status=dead → running`, and a CUDA-IMA crash can leave the pod
momentarily no-port, so the no-port rewrite would mask a CUDA-IMA dead
poll if it ran first.

**Recovery + once-more bound** (`backend_poll._failover_cuda_ima_runpod`;
REUSES the Part C inner relaunch `_relaunch_fresh_runpod` via a `stamp_fn`
kwarg — Part C byte-unchanged at the default). Layers, in order checked:

1. **ONCE-MORE BOUND.** If the DURABLE lease already records a CUDA-IMA
   failover for this run (the SEPARATE `Lease.runpod_cuda_ima_failover_of`
   field — distinct from `runpod_wedge_failover_of` so the two wedge kinds
   never cross-suppress), the fresh host ALSO crashed same-signature → a
   deterministic code bug: emit a terminal **`failure_class: code`** JSON
   (`reason=cuda_ima_repeats_after_failover`, via `_terminal_code_json`),
   which PARKS at `blocked` (the capacity-retry pass never re-drives
   `code`). NO second pivot.
2. **PER-WEDGE IDEMPOTENCY** — durable lease + `.claude/cache` sentinel,
   keyed to the crashed ATTEMPT identity (#1668); no liveness probe on
   this family (a reachable pod does NOT contradict a CUDA-IMA crash).
3. **INPUTS-ON-HF GATE** (reused as-is — a PARTIAL cell BLOCKS the
   irreversible terminate; human decides).
4. **TERMINATE** the crashed pod (best-effort — usually already dead) +
   **RE-PROVISION FRESH**, stamping the CUDA-IMA lease field.

**The host-wedge interpretation is a HYPOTHESIS the failover TESTS.** The
predicate detects a same-signature repeat; whether it is a transient
driver wedge (fresh host fixes it) or a deterministic code bug (fresh host
does not) is disambiguated EMPIRICALLY by spending the one bounded pivot.
The OUR_CODE_FRAME exclusion cheaply removes the common framed-code-bug
case first. `scripts/failure_classifier.py` is UNCHANGED — CUDA-IMA
already routes infra for crash #1; the new logic is poll-level.

## Tests of record

`tests/test_router.py`:
test_gcp_workload_error_fails_over_to_runpod_no_slurm_cascade,
test_workload_error_on_a_rung_fails_over_to_runpod_no_rung_advance,
test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted,
test_queue_timeout_failover_seam_carries_queue_timeout_reason,
test_failover_seam_default_reason_unchanged_byte_for_byte,
test_queue_timeout_reason_distinct_from_crash_and_capacity_reasons,
test_queue_vanish_failover_seam_carries_queue_vanish_reason,
test_queue_vanish_reason_distinct_from_crash_capacity_queue_boot_reasons,
test_attempt_one_gcp_rung_failover_identity_none_is_byte_identical,
test_attempt_one_gcp_rung_matching_failover_identity_returns_already_launched,
test_retry_gcp_ondemand_walks_standard_rungs_only_and_burns_attempts,
test_retry_gcp_ondemand_reason_param_threads_to_result_and_final_marker,
test_retry_gcp_ondemand_respects_daily_cap,
test_retry_gcp_ondemand_h100_intent_unservable,
test_retry_gcp_ondemand_reraise_workload_error_cause,
test_runpod_terminal_rung_short_circuits_on_gcp_stamped_lease.

`tests/test_backend_poll.py`:
test_gcp_pending_past_timeout_fails_over_to_runpod,
test_gcp_queue_timeout_failover_marker_carries_queue_timeout_reason,
test_gcp_queue_timeout_does_NOT_increment_gcp_attempts_today (+ the
queue-timeout negative controls),
test_gcp_pending_vanish_fails_over_to_runpod,
test_gcp_queue_vanish_failover_marker_carries_queue_vanish_reason,
test_gcp_queue_vanish_does_NOT_increment_gcp_attempts_today,
test_gcp_queue_vanish_second_tick_short_circuits,
test_gcp_queue_vanish_does_not_record_boot_death (+ the queue-vanish
negative controls),
test_gcp_queue_vanish_stale_prior_attempt_clock_fires,
test_gcp_queue_vanish_no_clock_flex_young_fires,
test_gcp_queue_vanish_no_clock_non_flex_not_vanish,
test_gcp_queue_vanish_no_clock_flex_aged_not_vanish,
test_gcp_queue_vanish_same_incarnation_workload_clock_not_vanish,
test_gcp_queue_vanish_stale_prior_incarnation_pending_clock_non_flex_not_vanish,
test_phase_clock_incarnation_roundtrip_and_mismatch_reads_absent,
test_gcp_queue_timeout_stale_incarnation_pending_clock_restamps_no_premature_fire,
test_gcp_wedge_stale_incarnation_clock_restamps,
test_gcp_queue_vanish_runpod_refusal_clean_residue_retries_gcp_ondemand,
test_gcp_queue_vanish_ondemand_retry_exhausted_falls_through_no_compute,
test_gcp_queue_vanish_ondemand_retry_skipped_on_leaked_residue,
test_gcp_queue_vanish_ondemand_retry_crash_window_short_circuits,
test_gcp_queue_timeout_runpod_refusal_clean_residue_retries_gcp_ondemand,
test_gcp_queue_timeout_ondemand_retry_exhausted_falls_through_no_compute,
test_gcp_queue_timeout_ondemand_retry_skipped_on_leaked_residue,
test_crash_and_boot_loop_failovers_do_not_retry_gcp_ondemand,
test_gcp_workload_error_on_ondemand_retry_parks_non_redrivable,
test_gcp_queue_vanish_ondemand_retry_sidecar_write_failure_mints_persistence_terminal,
test_lease_records_failover_of_is_keyed_per_gcp_crash_not_per_issue;
Part D `main()` integration (first-falls-through,
second-emits-fresh-host-failover, cuda-ima-before-noport ordering,
exhausted-emits-`code`).

`tests/test_gcp_backend.py`:
test_reconnect_handle_carries_provisioning_model_and_launch_ts,
test_render_startup_script_persists_diagnostics_before_teardown,
test_render_startup_script_diagnostics_uploads_log_and_partial_artifacts,
test_render_startup_script_diagnostics_is_guarded_and_bounded,
test_render_startup_script_is_valid_bash.

`tests/test_runpod_wedge_detection.py`: Part C (within-K override, past-K
escalation, malformed-clock fail-soft, predicate scope, per-cell gate,
failover idempotency) + Part D (CUDA-IMA predicate scope, signature regex,
wide-surface extraction, escalation ordering, cross-pod prior-marker
fallback, failover pivot + bounded-once + inputs-partial-blocks).

`tests/test_issue664_per_cell_upload.py`: Part C prerequisite (per-cell
incremental upload idempotency + exact-set + fail-loud verify + fresh-pod
resume).

## Relocated codebase traps (from `.claude/rules/gotchas.md`, #2189)

Verbatim gotchas.md entries whose topic this rule already owns — relocated
to recover gotchas.md byte budget (#2189); wording and `#N` citations kept.

- **GCP networking wedge (DHCPv4 loss → hung-but-RUNNING VM, frozen NON-terminal `eps/phase`) escapes BOTH GCP→RunPod failover paths** — the EXIT trap never fires, `gcp.poll` reads `running` forever. Detect: `describe` reads RUNNING + SSH hangs + `eps/phase` stuck at `workload` + serial tail `Could not set DHCPv4 address`. Recover: manual `--backend runpod` pivot (#667). In-flight GCP handles only (#2028).
- **GCP create timeout ≠ create failed — a FLEX_START create can stay PENDING past the 300s subprocess cap while succeeding server-side.** `GcpBackend.launch` catches the create `TimeoutExpired` and probes via `reconnect_or_none`; a live instance → `GcpCreateTimedOutStillProvisioning` → exit 75 (re-run the SAME command; idempotent reconnect, no double-create), truly absent → capacity-shaped `GcpProvisioningError` (#736). In-flight GCP handles only (#2028).
- **GCP zone-fallback ladder must not try a zone where the resolved machine type does not exist** — `backends/gcp.MACHINE_TYPE_ZONE_AVAILABILITY` filters `zones_to_try` per machine type before the create loop (fails OPEN for unlisted types; a guaranteed-to-fail zone attempt burns the per-day attempts counter; #653). In-flight GCP handles only (#2028).
- **GCP FLEX_START `create` can take 100–150 s+ under queue pressure and OUTLIVE a background-Bash wrapper — a killed launch wrapper does NOT mean no instance was created.** Launch creates FOREGROUND with `timeout ≥ 300000` ms; after ANY killed/timed-out wrapper, verify instance state (handle sidecar + `gcloud compute instances list`, login-shell PATH caveat) before re-dispatching — a blind relaunch double-provisions (#1739). In-flight GCP handles only (#2028).
