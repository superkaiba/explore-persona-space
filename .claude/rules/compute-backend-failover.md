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
2. `workload.log` — the workload log (traceback / stderr), `$EPS_LOG_PATH`.
   #1151: items 1-2 ride ONE staged `upload_folder` commit (the "first
   bundle") with ONE retry after an `EPS_PERSIST_RETRY_BACKOFF_S` backoff
   (default 10 s — the #935 sibling knob); the persist makes ZERO per-file
   `upload_file` calls anywhere (each one triggers a server-side recursive
   tree-listing pre-check that 504s ~half the time at ~160 s/file on this
   large repo — the #664 stall class that most plausibly ate #811's entire
   300s budget). Repo paths unchanged;
3. `worker_logs/<relpath>` (#885) — every regular file under
   `$WORKLOAD_ROOT/logs/` (fan-out per-worker logs carrying the REAL
   traceback; the canonical `workload.log` ends at the fan-out line — the
   #779 loss class, two ~30-min manual boot-disk detaches on 2026-07-02),
   newest-first by mtime, per-file tail cap `EPS_PERSIST_LOG_FILE_CAP_BYTES`
   (default 5 MiB — the traceback is at the END of a log, so an oversized
   file is TAILED at stage time, never skipped wholesale), file-count bound
   `EPS_PERSIST_LOG_MAX_FILES` (default 40; `< 1` is a loud-SKIP disable),
   canonical-log dedup skip, git-TRACKED files under `logs/` excluded
   (#1351: the repo is cloned at `$WORKLOAD_ROOT`, so committed
   logs/daily+weekly retrospectives — 48 tracked files vs the 40-file
   bound — would otherwise clutter the prefix and consume the budget;
   already durable in git; the exclusion is at WALK time so tracked files
   never consume `LOG_MAX_FILES` slots, and a git failure of any kind
   FAILS OPEN to sweeping everything with a loud grep-stable
   `git-tracked exclude unavailable` WARN), staged into
   `/tmp/eps-worker-logs` and uploaded
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
   than burning the 300s budget. As of #1339 a partial dir whose
   post-exclude file count exceeds `EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT`
   (default 1000; `< 1` disables chunking with a WARN) uploads as
   newest-first staged batches (staging root `EPS_PERSIST_DIR_STAGE_DIR`,
   default `/tmp/eps-dir-batch`), each ONE `upload_folder` commit with one
   bounded retry (`EPS_PERSIST_RETRY_BACKOFF_S` backoff), abandoning the
   dir loudly after `EPS_PERSIST_DIR_BATCH_ABORT_STREAK` (default 2)
   consecutive fully-failed batches — repo paths byte-identical to the
   unchunked upload, every batch outcome printed, plus a
   `k/nb batches uploaded (f/n files)` summary line (incident #1090 fu5:
   a 29,024-file / 14.4 MB single commit processed ~31 s server-side, the
   gateway timed out delivering the response, and the client logged
   FAILED on a commit that had LANDED). A retried batch whose prior
   attempt actually landed re-commits the same content at the same paths
   (content-identical commit — idempotent effect). Operator note: after
   an ABORT / FAILED dir line, verify the Hub prefix (scoped
   `list_repo_tree` on `issue<N>_partial/<attempt_id>/`) BEFORE
   re-running any recovery — a gateway-timeout "failure" may have landed
   (the fu5 shape);
6. `workload_<utc-ts>.log` (#854) — a per-crash timestamped copy of the
   workload log, uploaded AFTER the partial dirs (small-first ordering; the
   canonical `workload.log` already landed the traceback early). The
   canonical `crash_report.json` / `workload.log` names are OVERWRITTEN by
   a same-attempt re-crash (run-3 overwrote run-2's log on #825) — prior
   crashes' canonical copies stay recoverable via the HF repo's git
   history; the timestamped copies accumulate per crash;
7. `crash_persist_transcript.log` (#854) — the `[crash-persist]` audit
   lines. #1151: items 6-7 ride ONE staged "final bundle" `upload_folder`
   commit (no retry), with the transcript staged LAST — after the DONE
   line — so its uploaded copy still records every earlier upload/skip
   line. Its presence proves the persist ran to completion with every skip
   recorded; its ABSENCE proves a killed persist (the serial console is
   unreadable post-DELETE, #640). The skip-vs-kill discriminator is now
   THREE-WAY: transcript present (persist completed; per-upload outcomes
   audited inside it) / breadcrumb final-status with no transcript (the
   persist ran to a final rc but the HF channel dropped the bundle) /
   standing `attempted` breadcrumb (persist killed mid-flight) — see
   item 8;
8. **the `eps/persist` guest-attribute breadcrumb (#1151)** — the
   HF-INDEPENDENT persist-fate channel. #811's lesson: EVERY prior no-fire
   signal (the transcript discriminator, the fd-3 serial lines, the
   canonical uploads) rode the SAME HF channel or died with the DELETEd
   boot disk, so an HF-channel failure at trap time left a zero-file
   prefix indistinguishable from "the persist never ran".
   `_eps_persist_diagnostics` now writes a link-local guest attribute
   (SEPARATE `eps/persist` key — never `eps/phase`, which the poll
   classification + #908 zombie predicates key on; the #935
   `eps/done_persist` discipline): `attempted` unconditionally at entry,
   `skipped_no_token` on the early-boot token-guard skip, then a final
   status from an rc-file readback of the persist subshell
   (`EPS_CRASH_PERSIST_RC`, default `/tmp/eps-crash-persist.rc`; the
   pipeline's own `$?` is the streamer's) —
   `ok` (rc 0), `timeout` (rc 124), `failed_rc<N>` (any other rc). A
   MISSING rc file deliberately writes NOTHING: the standing `attempted`
   IS the killed-mid-persist signal. A boot-time DELETE clears the key so
   a salvage-relaunch second boot never inherits a prior crash's value.
   ≤3 fail-soft `curl -m 5` writes per crash (inside the 3/s burst +
   10/min guest-attribute caps); the poller reads it best-effort on every
   failed / finalize-failed-classifying tick and appends
   `[crash-persist-breadcrumb] eps/persist=<value> (instance <status>)` to
   the terminal marker's `log_tail_excerpt`
   (`GcpBackend._guest_persist_breadcrumb` — never raises, never gates
   classification or failover: diagnostic-only by design).

   Decision table (`eps/persist` × the HF `issue<N>_partial/` prefix):

   | `eps/persist` | HF prefix | Reading |
   |---|---|---|
   | ABSENT | absent | pre-fix render, trap never entered the persist, entry-curl failure (`-m 5` expiry), OR total metadata-channel death (the #667 class — co-signal: a WRITTEN `eps/phase=failed` + ABSENT breadcrumb genuinely isolates the pre-persist window; phase ALSO absent ⇒ dead metadata channel, route to the #667 wedge lane) |
   | `attempted` (standing) | absent | persist KILLED mid-flight (external termination / hard kill). **TERMINATED-only reading** — a RUNNING-window read may catch a healthy persist in flight (the poll excerpt self-discloses instance status + an in-flight qualifier) |
   | `attempted` (standing) | present | kill landed in the window after uploads completed but before the final-status write (rare; transcript disambiguates) |
   | `skipped_no_token` | absent | early-boot crash before secrets fetch |
   | `ok` | present | normal crash persist. `ok` = the persist python EXITED 0, not "all artifacts landed" — per-upload failures are logged, never raised; the transcript is the per-upload audit |
   | `ok` | absent/partial | persist completed but HF channel down (the 429-storm shape) — or a repo/prefix misroute; separable post-hoc by a scoped `list_repo_tree` listing |
   | `timeout` | absent/partial | 300s budget exhausted (stalled uploads — the 504 shape) |
   | `failed_rc<N>` | absent | bootstrap/compound failure (127 = uv missing; 1 = cd short-circuit OR python top-level failure; else python rc) |

**Sweep scope (explicit):** the partial sweep covers exactly the three
named directories above (`eval_results/issue_<N>/`, `data/issue_<N>/`,
`data/issue<N>/`) plus the `$WORKLOAD_ROOT/logs/` worker-log tree (#885) —
still NOT universal artifact discovery (e.g. `figures/issue_*`,
checkpoints, `ood_eval_results/` are not swept). Worker logs are swept
only when they land under `$WORKLOAD_ROOT/logs/` (relative `logs/…` or
`$REPO_ROOT/logs/…` on the workload-cmd branch, where the startup script
exports `REPO_ROOT="$WORKLOAD_ROOT"` — #641); absolute
`<vm_scratch_dir>/logs` paths are not swept — place dispatcher worker logs
under the workload-root `logs/` convention. Within the worker-log tree,
git-TRACKED files (per `git ls-files -z -- logs` against the cloned repo)
are excluded at walk time (#1351) — a run-generated forensic log at a
git-tracked path is NOT swept, so never append crash forensics to a
committed file; the exclusion fails OPEN (sweep everything) on any git
failure. A workload writing partials
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
and `data_issue_<N>/` when the workload wrote one. **The #811 lesson
(#1151): all of these signals ride the SAME HF channel and can fail
TOGETHER** — #811's post-#854 crash left a zero-file prefix with no way to
tell a killed persist from a dead HF channel — so the `eps/persist`
guest-attribute breadcrumb (item 8) is the HF-independent fix-engaged
signal: on the next real GCP crash the terminal diagnosis marker (the
01:24Z shape) carries `[crash-persist-breadcrumb] eps/persist=<value>`
regardless of HF state, turning a persist no-fire into a specific, named
value instead of an absence.

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

## Part A-ter — finalize-failed-but-artifacts-ok (#1055)

A GCE workload can exit non-zero AFTER all its declared deliverables are
verified uploaded (the #811 shape: 10.86 GB complete on HF, then the tail
step died — a known mechanical class is the rc=134 HF-datasets
interpreter-shutdown SIGABRT, `gotchas.md`). Pre-#1055 that read
`eps/phase=failed` and triggered the full crash-response machinery (RunPod
failover / crash-fix routing / a manual "was anything lost?" diagnosis
cycle) on a run that lost nothing. The EXIT trap now branches on POSITIVE
FILE EVIDENCE:

- **The sentinel contract.** The startup script exports
  `EPS_DELIVERABLES_OK_PATH` (attempt-scoped —
  `gcp.deliverables_ok_path_for`, mirroring `sentinel_path_for` — so a
  fresh attempt never reads a prior attempt's evidence) and `rm -f`s it at
  boot (a re-booted instance with the SAME attempt_id + preserved disk must
  not inherit a prior boot's evidence). The WORKLOAD writes that file ONLY
  after its final upload+verify step confirms every declared deliverable is
  on HF. **Writer rule (multi-stage drivers):** stamp the sentinel ONLY
  after the LAST deliverable-producing step's upload+verify PASS — a
  multi-STAGE in-instance driver that stamps after stage-1's verify
  misclassifies a stage-2 crash (Step 8 upload verification backstops it,
  but the contract precludes it). A git-committed eval-JSON leg is itself
  a declared deliverable: do NOT stamp the sentinel while a result commit
  is unpushed (`git rev-list --count origin/<branch>..HEAD` != 0) — else
  a #1205 push-verify backstop failure classifies
  `finalize_failed_artifacts_ok` (done-like) instead of `failed`
  (`.claude/rules/pod-side-reporting.md` § Result-push verification
  contract). For a composed `--workload-cmd` chain,
  insert `&& touch "$EPS_DELIVERABLES_OK_PATH"` BETWEEN the
  deliverable-producing step and the tail steps (a trailing `&& touch`
  after the whole chain only runs on rc==0, when the trap is idle —
  useless by construction). RECOMMENDED: populate `verified_prefixes` (plus
  issue / attempt_id / ts) in the sentinel JSON so Step 8 / triage can
  cross-check the claim's scope — the trap checks EXISTENCE only.
- **Fail-open default.** A workload that never writes the sentinel keeps
  today's `failed` path byte-for-byte; driver adoption is per-driver
  experiment code, out of the #1055 workflow-fix scope.
- **The trap branch + phase value.** rc≠0 AND the sentinel file present →
  `_eps_phase finalize_failed_artifacts_ok`; else `_eps_phase failed`
  (unchanged). The shared tail — watchdog reap, log tail,
  `_eps_persist_diagnostics "$rc"`, `shutdown -h now` — runs on BOTH arms,
  ordering unchanged: diagnostics still upload (the finalize failure needs
  its own `issue<N>_partial/` evidence) and the billing-bounding poweroff
  is untouched. **#1004 coherence:** the classification is NEVER keyed on
  the rc value or crash timing, and the literal `done` phase is never
  published on the rc≠0 path — only the workload-written evidence flips the
  phase value.
- **Poll classification.** RUNNING (the brief pre-poweroff window) or
  TERMINATED + `eps/phase=finalize_failed_artifacts_ok` →
  `PollResult(status="done", current_phase="workload_done_finalize_failed")`
  — the #935 stance: a SUCCESSFUL run whose finalize hiccupped.
  `status="done"` fails BOTH async-failover conjuncts by construction (no
  RunPod failover, no crash-fix routing), the #1029 boot-death streak
  resets (a positive workload signal), and a fresh `epm:run-launched`
  relaunch marker still wins in the RUNNING window (the #612
  relaunch-follow tuple includes the new phase). The phase is in
  `_TERMINAL_GUEST_PHASES` ⇒ `_ZOMBIE_GUEST_PHASES`: the janitor promptly
  reaps a RUNNING VM stuck in it, and reconnect/pre-launch reclaim treat it
  as a finished zombie.
- **Finalize.** The COMPLETION sentinel is never written on this path (the
  success tail is unreachable after a non-zero exit), so `confirm_artifacts`
  FAILs exactly as for `workload_done_self_poweroff` — run
  `dispatch_issue.py finalize --skip-confirm-artifacts`; Step 8 upload
  verification remains the independent artifact gate.
- **Triage searchability.** Recurring `workload_done_finalize_failed`
  occurrences indicate a SYSTEMATIC finalize bug that still deserves a
  root-cause fix — the classification keeps the failure visible for triage,
  it does not normalize it.

## Part B — GCP-failure → RunPod failover (contract reversal)

A GCP attempt failure of **ANY class now routes the next attempt to
RunPod** — the reversal of the historical "GCP workload failure surfaces
with NO fallback" invariant. Rationale: if GCP is failing a run, running
it on RunPod keeps the science moving AND gives a persistent, SSH-able pod
for diagnosis — strictly better than GCP's delete-on-crash boot disk.

Five distinct GCP-failure paths, ALL ending at the same
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
- **Pre-workload BOOT LOOP** (`reason: gcp_boot_loop_failover_runpod`,
  #1029) — N (default 2) CONSECUTIVE pre-workload setup deaths on the SAME
  ladder rung for the SAME issue, counted per launch INCARNATION in the
  durable lease. The Nth death fails over to RunPod; the route()-side
  ladder walk additionally SKIPS a boot-looped rung same-day (see
  § Pre-workload boot-loop below). DISTINCT from a workload crash (the
  workload never started), a queue timeout (the instance dequeued and
  BOOTED, then died), and a capacity miss (the creates all succeeded).
- **FLEX_START queue VANISH** (`reason: gcp_queue_vanish_failover_runpod`,
  #1116/#1112) — the create SUCCEEDED, the instance sat PENDING in the DWS
  capacity queue, then DISAPPEARED from instances-list entirely (no delete
  operation): the queue dropped the request server-side. Detected by the
  async poller from the sidecar phase clock (last observed phase
  `"pending"` + a dead `terminal_instance not found` poll) and failed over
  on the FIRST occurrence (see § FLEX_START queue-vanish below). DISTINCT
  from the queue TIMEOUT (the queued instance there still EXISTS
  server-side — that failover tears it down; here the record is already
  gone), from a boot loop (the instance never booted), from a workload
  crash (nothing ever ran), and from a capacity miss at create time (this
  create succeeded).

**Intent translation at the terminal rung (#940).** The RunPod launch
paths (the terminal rung — all five fallback/failover paths above funnel
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
  ondemand-80). (Width-1 rung counts are UNCHANGED by #1121; a
  width-DECLARING dispatch prepends wide rungs per § Width-aware rungs
  below — the width-8 short walk is 14 rungs.)
- **LONG / UNKNOWN-length jobs — flex leads, NO spot:** flex-start A100-80 →
  on-demand A100-80 → on-demand A100-40 (fits-40 only). Spot is barred
  (preemption too costly for a long job); flex — non-preemptible once
  running, queues for capacity — leads. An unknown-length job (no time
  budget) is NOT short, so it takes this branch. A long `ft-7b` yields 2 GCP
  rungs; a long `lora-7b` yields 3 (width-1 counts; the width-8 long walk
  is 9 rungs, § Width-aware rungs below).

The flex rung threads `provisioning_model=FLEX_START` via
`router._flex_start_rung` (label `flexstart_<gpu_kind>`); A2 acceptance of
`--provisioning-model=FLEX_START` was confirmed by a live #680 probe on both
`a2-ultragpu-1g` and `a2-ultragpu-4g`. The per-day attempt cap
(`MAX_GCP_ATTEMPTS_PER_DAY`) was bumped 5 → 8 at #680 to cover the
up-to-5-rung short-job ladder plus a same-day retry margin, then 8 → 16 at
#1121 — the width-8 short walk is up to 14 rungs + margin, the same sizing
logic (still an attempt COUNT, never a dollar cap).

**Width-aware rungs (#1121).** A dispatch that DECLARES a shardable
multi-GPU axis via the existing `--gpus N` flag (`RunSpec.gpus`; N ∈
{2, 4, 8} above the intent's base machine width, on a width-eligible
A100-class intent — `gcp.WIDTH_ELIGIBLE_INTENTS` = {lora-7b, lora,
capture-7b, eval, debug, ft-7b}) walks WIDE `a2-ultragpu-{8,4,2}g` rungs
(`gcp.WIDE_A100_80_BY_WIDTH`) BEFORE the base ladder, WIDTH-MAJOR: every
provisioning model at width w is exhausted before width w−1 is accepted
(wall-clock is the scarce resource; credits are not — a spot-8g attempt is
strictly preferable to an on-demand-4g one). Intra-width the length-aware
order above applies verbatim; job length is classified ONCE at the WIDEST
requested machine (GPU-hours = wall × width) and threaded through the base
tail. Wide rung labels carry an `x<w>` suffix (`spot_a100_80x8`); width-1
labels are byte-identical to pre-#1121. The exact width-8 walks:

- **Width-8 SHORT** (≤ 2 GPU-h at width 8, or `spot_tolerant`; 14 rungs):
  `spot_a100_80x8 → flexstart_a100_80x8 → ondemand_a100_80x8 →
  spot_a100_80x4 → flexstart_a100_80x4 → ondemand_a100_80x4 →
  spot_a100_80x2 → flexstart_a100_80x2 → ondemand_a100_80x2 →
  spot_a100_80 → spot_a100_40 → flexstart_a100_80 → ondemand_a100_80 →
  ondemand_a100_40` → SLURM lanes → RunPod terminal rung.
- **Width-8 LONG/UNKNOWN** (the common case — wall × 8 usually exceeds
  the 2 GPU-h spot threshold; 9 rungs): `flexstart_a100_80x8 →
  ondemand_a100_80x8 → flexstart_a100_80x4 → ondemand_a100_80x4 →
  flexstart_a100_80x2 → ondemand_a100_80x2 → flexstart_a100_80 →
  ondemand_a100_80 → ondemand_a100_40` → SLURM → RunPod.

H100 is EXCLUDED from the width walk (no `WIDE_A100_80_BY_WIDTH` rows; the
H100 intents are not width-eligible): preemptible quota is exactly 8 (one
8×, zero concurrency headroom, #743), there is no on-demand pool, and the
H100 quota metrics are absent from `regions describe` on this project, so
the fail-open headroom pre-check cannot protect a doomed create —
`sweep-8g-h100` stays explicit-`--intent`-only. The #783 queue-timeout,
#1116 queue-vanish, and #1029 boot-loop poller machinery applies to wide
rungs unchanged (per-rung-label streaks key on the new `x<w>` labels
cleanly). **Paid-fallback exposure planners must understand:** width
degradation fires only on CREATE-TIME capacity misses. For a LONG job
`flexstart_a100_80x8` is rung 1, and a flex create that QUEUES ends the
ladder walk (route() returned a handle) — so the realistic fallback for a
queued-but-stuck wide dispatch is the #783 queue timeout
(`EPS_GCP_QUEUE_WAIT_SECONDS`, default 600s) failing over to a PAID RunPod
pod at the REQUESTED width, NOT 4g/2g degradation on GCP. A workload that
CANNOT re-shard off the realized width (`realized_gpu_count` on the
`epm:backend-selected` marker / handle sidecar; `requested_gpus` rides
alongside) must pin its width rather than ride the degrading walk.
Poller-side width-degrade re-entry is a named deferred follow-up (#1121
plan), NOT built.

Tests of record: `test_ladder_short_job_spot_before_ondemand`,
`test_ladder_short_job_spot_miss_then_ondemand_order`,
`test_ladder_short_job_full_rung_order`,
`test_ladder_long_job_flexstart_before_ondemand_no_spot`,
`test_ladder_long_job_flexstart_miss_then_ondemand_order`,
`test_ladder_unknown_length_takes_long_branch`,
`test_flexstart_rung_threads_flex_provisioning`,
`test_max_gcp_attempts_per_day_is_sixteen`,
`test_workload_error_on_later_rung_fails_over_to_runpod`; width-aware
(#1121): `test_width8_long_job_walks_wide_rungs_width_major`,
`test_width8_short_job_full_rung_order`,
`test_width_degradation_on_capacity_miss_lands_4g`,
`test_width1_ladder_byte_identical_explicit_gpus_none_and_matching`,
`test_width_ladder_never_emits_h100_machine`,
`test_workload_error_on_wide_rung_fails_over_to_runpod` (all in
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

### FLEX_START queue-vanish → RunPod (#1116/#1112)

A DWS-queued FLEX_START instance can be dropped SERVER-SIDE: the create
reports success (insert DONE), the instance sits PENDING in the capacity
queue, and then it simply DISAPPEARS from instances-list — no delete
operation in the operations log. #1112 hit this twice in one evening
(inserts DONE 22:30Z + 22:50Z, 2026-07-07; both instances gone before the
600s queue-timeout could mature), and because `route()` advances the ladder
only on CREATE-time capacity errors, every relaunch re-booked the same dead
flex rung until a manual `--backend runpod` pivot. `gcp.poll` maps the
post-vanish describe-404 to `status="dead"` /
`current_phase="terminal_instance not found"` — an attribute-blind shape
that pre-#1116 read as an ordinary crash (or, worse, fed the #1029
heuristic boot-death streak, mislabelling a pure CAPACITY event as a boot
problem).

- **The clock discriminator.** The sidecar phase clock (the SAME
  `_read_phase_clock` record the #669 wedge + #783 queue-timeout stamp)
  records `"pending"` while the instance is queued; a dead not-found poll
  whose `last_phase` reads `"pending"` means the instance was LAST OBSERVED
  still queued — it never reached a running phase — so the vanish is
  deterministic capacity evidence.
  `backend_poll._maybe_escalate_gcp_queue_vanish` rewrites it to
  `terminal_queue_vanish` (READ-ONLY clock use: no aging floor, no streak —
  unlike #783 there is nothing to age, unlike #1029 nothing to count) and
  `_failover_vanished_gcp_to_runpod` fails it over on the FIRST occurrence
  via the shared `_failover_gcp_to_runpod` core (same lease+sentinel
  exactly-once bound, same terminal-rung seam, same sidecar re-point +
  terminal-JSON contract), `reason: gcp_queue_vanish_failover_runpod`.
- **`teardown_first=False`.** The instance record is already GONE
  server-side (that absence IS the trigger), so there is nothing to tear
  down — the #659 stance, NOT #783's (only a still-LIVE queued instance
  needs its capacity request released).
- **No daily-attempt burn.** Structural, as for #783/#1029:
  `gcp_attempts_today` bumps only on a create inside
  `_attempt_one_gcp_rung`, which the poller never re-enters.
- **CPU-intent scope (#677/#747):** a `cpu-bigmem` vanish never
  rewrites/fails over (no cheap RunPod lane) — the guard gates the REWRITE
  itself, so its ordinary dead path INCLUDING today's #1029 boot-death
  record stays byte-identical; `cpu-small` / `cpu-mid` are eligible as
  usual.
- **Ordering:** the vanish branch runs BEFORE the #1029 boot-loop recorder
  in `main()` — not-found is in the boot-death heuristic phase set, and the
  vanish branch's early return is what keeps a pure capacity miss from
  poisoning the boot-death streak.

Named residuals (accepted, documented):

1. **Transient-404 / flicker.** A transient not-found read on a LIVE queued
   instance (an API flicker) fires the failover ONCE (lease-bounded); the
   orphaned still-queued instance is bounded by its own `--max-run-duration`
   fence + the EXIT-trap finalize + the stale-GCP-VM janitor
   (`gcp_audit.py`), and the poller no longer watches it after the sidecar
   re-point.
2. **Manual-delete asymmetry.** PENDING is the ONE state where a manual
   `gcloud compute instances delete` AUTO-fails-over (auto-spends RunPod) —
   a manual delete elsewhere takes the crash/terminated classifications. A
   manual delete during an active poll loop already implies a pivot, and
   the lease bounds it to one RunPod launch.
3. **No-clock-record inertness.** A vanish observed with NO clock record
   (fresh-dispatch handle, wiped sidecar — the #1112 manual-removal shape)
   is invisible to this trigger and falls back to the #1029 boot-death
   streak path (two poller-observed vanishes → the boot-loop failover) —
   safe, just slower.
4. **One-tick dequeue→boot→crash→DELETE mislabel.** A run that dequeues,
   boots, crashes, and is DELETEd entirely between two polls presents the
   same dead not-found + pending-clock shape and is labelled a queue vanish
   — same destination (a RunPod failover) the #659 async path would give,
   different reason label; Part A crash diagnostics upload from the EXIT
   trap regardless of the poller's label (the mirror of #1029 note (b)).

Pre-agreed hardening path (the #1116 plan's kill criterion 2): if a
sanctioned actor ever starts deleting LIVE PENDING instances, the trigger
needs an operations-log delete-op check (a genuine vanish leaves NO delete
operation; an actor's delete leaves one) — a re-plan, not a tweak.

### Pre-workload boot-loop → RunPod (#1029)

A boot-looping rung — the #763 shape: `flexstart_l4` re-selected by every
relaunch, each create dying ~5.5 min post-insert via guestTerminate with NO
crash diagnostics, `gcp_attempts_today` 2→5 before a manual RunPod pivot —
is now broken automatically after at most **N=2 consecutive same-rung
pre-workload deaths** (env `EPS_GCP_BOOT_DEATH_STREAK_N`; the first death
keeps its one free retry, so single-occurrence behavior — a lone transient
clone/uv-sync setup death, a lone spot preemption — is unchanged).

- **Record:** each death destroys the VM and the relaunch writes a FRESH
  sidecar, so the consecutive-death count lives in the durable per-issue
  lease (`~/.eps-routing/issue-<N>.json`, `Lease.gcp_boot_death_streaks`),
  keyed per (issue, rung) and per launch INCARNATION (`handle.job_id` — the
  GCE instance id, distinct per create; fallback
  `(attempt_id, gcp_launched_ts)`; attempt_id ALONE is forbidden — #763's
  five creates shared one attempt_id), same-UTC-day scoped, RESET on any
  POSITIVE workload signal (running `workload`/`relaunched_workload`, a
  `terminal_workload_failed` crash — the workload started, so boot was
  fine — or a #935 `done` shape; NEVER a pre-workload blocklist — the
  mid-boot `startup` phase must not reset).
- **Classify:** a pre-workload death is EITHER deterministic
  (`terminal_setup_failed` — the §4.1.0b `workload_started` discrimination,
  produced in the RUNNING window since #659 and in the TERMINATED window
  since #1029) OR heuristic (a YOUNG `terminal_terminated` /
  `terminal_instance not found` observation, launch→observation age below
  `EPS_GCP_BOOT_DEATH_MAX_AGE_SECONDS`, default 1500s — post-DELETE polls
  are attribute-blind, so age is the only available signal there).
- **Fire (poller side):** at streak >= N the poll is rewritten to
  `terminal_boot_loop` (`_maybe_escalate_gcp_boot_loop` →
  `_is_gcp_boot_loop`) and `_failover_boot_looped_gcp_to_runpod` reuses the
  SAME `_failover_gcp_to_runpod` core (idempotency lease + sentinel,
  sidecar re-point, terminal-JSON contract) with `teardown_first=False`
  (the VM self-powered-off and DELETE reaps it; a lingering record degrades
  to `gcp_audit.py` — the #659 stance). The evidence carries
  `boot_death_streak` + `gcp_ladder_rung`.
- **Skip (route side):** `_attempt_one_gcp_rung` SKIPS a rung whose
  same-UTC-day streak is >= N on the auto chain (RouteAttempt outcome
  `boot_loop_rung_skipped`, exact quota-headroom-skip shape) WITHOUT
  bumping `gcp_attempts_today` — the cap counts CREATES; a skip avoids the
  create, so the breaker STOPS cap burn. An explicit `backend: gcp` pin is
  exempt (an explicit user ask attempts anyway). If every GCP rung skips,
  the chain proceeds to SLURM then the RunPod terminal rung by the existing
  lane order.
- **CPU-intent scope (#677/#747):** a `cpu-bigmem` boot loop RECORDS but
  never rewrites (no cheap RunPod lane) — the route()-side skip is its
  breaker (skip → GCP CPU exhaustion → the typed
  `cpu_exhausted_no_runpod_lane` terminal, verbatim #677); `cpu-small` /
  `cpu-mid` fail over to RunPod CPU as usual.
- **Deliberate policy delta:** a LONE `terminal_terminated` still never
  fails over (the #669 exclusion, preserved verbatim — including a
  TERMINATED+`failed` VM whose workload HAD started); but N>=2 consecutive
  sub-floor early deaths on ANY rung — including a spot rung, where an
  early preemption during setup can count toward the streak — now advance.

Four one-sentence operational notes:

(a) **Re-drive contract:** a sub-N boot death lands `failure_class: code`
on the ordinary dead path and is re-driven by the PER-ISSUE SESSION's
crash-fix/relaunch loop (exactly what produced #763's four automatic
relaunches) or a manual `dispatch_issue.py` — NOT the watcher
capacity-retry pass (infra/`no_compute_available` only); a task with no
live re-driver parks at `blocked` after death 1 with no loop and no cap
burn — the breaker correctly stays disengaged (a breaker targets a loop;
a loop requires a re-driver by construction).
(b) A genuine FAST workload crash observed only post-DELETE (a young
`terminal_instance not found`) counts toward the streak and, at N, fails
over under the boot-loop reason rather than the #659 reason — same action
and destination, different label; note it so a future incident read is
not misdiagnosed.
(c) The "2 creates instead of 4-6" headline is the SAME-RUNG re-pick case
(#763's shape); cross-rung capacity churn still burns up to ~2 creates
per rung, bounded only by the daily cap 8.
(d) Every TERMINATED+`failed` poll now issues one extra `_workload_started`
guest-attribute probe (perf-only; the probe-failure fallback keeps
`terminal_terminated`, never manufacturing a setup classification).

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

### Live-diagnosis access to a GCE instance (SSH / serial / Monitoring)

Three access facts, each re-discovered by multiple sessions on 2026-07-02
(≥5 sessions ate a failed first attempt):

- **SSH: external-IP first.** The default `gcloud compute ssh` tries an
  IAP tunnel, which the `eps-gcp` configuration is NOT authorized for —
  the first attempt fails every time. Pass the external-IP form up front
  (`gcloud compute ssh <name> --configuration=eps-gcp --zone=<zone>
  --tunnel-through-iap=false`, or plain `ssh` to the instance's external
  IP); fall back to the serial console when guest networking is dead
  (the #667 wedge above).
- **Serial console is the always-available read** (`gcloud compute
  instances get-serial-port-output --configuration=eps-gcp`); the #854
  eager `[crash-persist]` lines land there.
- **The Cloud Monitoring API is not enabled** on `eps-persona-gpu-jun2026`
  — metric probes return nothing. Diagnose via serial console + SSH, or
  enable the API once (a deliberate ops change, not something a session
  does mid-run).

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
`eps/phase` ∈ {done, failed, finalize_failed_artifacts_ok, wedged}
(`_ZOMBIE_GUEST_PHASES`) and
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
- **Footprint feasibility gate + disk threading (#1010, incident #958).** A
  plan-STATED footprint (`spec.extra["boot_disk_gb"]` / `["min_ram_gb"]`, from
  `dispatch_issue.py --boot-disk-gb` / `--min-ram-gb`) is checked at
  `_runpod_terminal_rung` against `router.RUNPOD_CPU_INSTANCE_CAPS`
  (probe-verified effective `containerDiskInGb` caps — cpu3g-2-8 → 20,
  cpu3c-8-16 → 50 honored floor — plus fixed RAM); an unsatisfiable footprint
  refuses BEFORE any RunPod API call with the typed
  `CpuFallbackInfeasibleError` / `reason: cpu_fallback_infeasible_for_plan`
  (a `CpuExhaustedNoRunpodLaneError` subclass; NOT in
  `TRANSIENT_CAPACITY_REASONS` — the instance can never grow to fit the
  plan). A feasible `boot_disk_gb` is THREADED into the provision argv
  (`--container-disk-gb max(50, boot_disk_gb)`, `RunPodBackend.launch`), and
  `pod_lifecycle`'s CPU branch clamps a default-band over-cap effective
  payload to the instance cap (the untouched default effective 50 exceeds the
  cpu3g cap — pre-#1010 every default cpu-small RunPod provision failed
  validation) while an explicit above-band request refuses pre-API.
  ADOPTION: launch composers pass `--boot-disk-gb` whenever a CPU stage sizes
  disk > 50 GB and `--min-ram-gb` whenever it sizes RAM > 16 GB — flag-less
  launches keep today's behavior (experimenter pre-launch gate as the sole
  defense).

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

**Shape-dependent carve-out (#1092):** before reading a repeat IMA as a host
defect, check the gotchas differential — if the crash follows the WORKLOAD
shape (identical code clean on A100 + a same-pod short-prompt probe clean), a
fresh host is EXPECTED to re-hit it and the fix is the default-off engine
knobs, not a pivot (a knobs-on rerun that still IMAs falsifies the shape
diagnosis — revert to the Part D pivot) — see `.claude/rules/gotchas.md`
§ "vLLM-on-H100 CUDA illegal-memory-access under heavy shared-prefix caching".

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
- `tests/test_router.py` (Part B FLEX_START queue vanish, #1116:
  `test_queue_vanish_failover_seam_carries_queue_vanish_reason`,
  `test_queue_vanish_reason_distinct_from_crash_capacity_queue_boot_reasons`)
- `tests/test_backend_poll.py` (Part B FLEX_START queue vanish end-to-end, #1116:
  `test_gcp_pending_vanish_fails_over_to_runpod`,
  `test_gcp_queue_vanish_failover_marker_carries_queue_vanish_reason`,
  `test_gcp_queue_vanish_does_NOT_increment_gcp_attempts_today`,
  `test_gcp_queue_vanish_second_tick_short_circuits`,
  `test_gcp_queue_vanish_does_not_record_boot_death`, + the negative
  controls: workload-clock / no-clock-record / terminated-phase /
  cpu-bigmem-excluded)
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
