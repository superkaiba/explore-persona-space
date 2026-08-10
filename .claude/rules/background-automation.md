---
description: Background cron automations — stale-pod audit, stale-GCP-VM janitor, stale-worktree sweep, autonomous-session watcher (crash-recovery, pod-safety, gate-push + reconcile + zombie-wrapper + idle-unmapped passes) — full predicates, env-var overrides, and incident history (loads when you touch the audit / watcher scripts)
paths:
  - "scripts/worktree_audit.py"
  - "scripts/cron_worktree_audit.sh"
  - "scripts/autonomous_session_watch.py"
  - "scripts/cron_autonomous_session_watch.sh"
  - "scripts/tick_triage.py"
  - "scripts/pod_audit.py"
  - "scripts/cron_pod_audit.sh"
  - "scripts/gcp_audit.py"
  - "scripts/cron_gcp_audit.sh"
  - "scripts/codex_task.py"
---

# Background cron automations

## RETIRED 2026-08-06: `/daily` and `/mygoat-daily` (read this first) <!-- lint: historical-ref -->

<!-- /mygoat-daily is a user-global skill (~/.claude/skills/), never a project
     skill, and it is now retired — every mention below is a historical
     citation of a removed automation, not a live dispatch target. -->


Thomas's directive 2026-08-06 (*"remove daily from the workflow"*, then
*"remove mygoat-daily too"*). Both crons are `#DISABLED` in the crontab;
backups under `~/.eps-autonomous/crontab-backups/`. **No daily cron of any
kind is active.** Retired lines:

- `27 23 * * *` — nightly `claude -p /daily`
- `0 6 * * *` — `cron_daily_healthcheck.sh` (alerted when a `/daily` file
  failed to land; left armed it would page every morning forever)
- `45 23 * * *` — nightly `claude -p "/mygoat-daily"`

**Why:** `/daily` was the dominant source of the `proposed` backlog — 86 of
135 `proposed` tasks (64%) at retirement, emitting ~20-25 `kind: infra`
tasks/night against a drain of ~15-20/day at concurrency cap 3.

### CONSEQUENCE — fail-toward-silence lanes are now UNBACKED

Several passes were deliberately specced to fail toward silence *because*
`/daily` was the backstop behind them. That backstop is gone; these now fail
silently with nothing catching the residue:

| Site | What it relied on `/daily` for |
|---|---|
| `autonomous_session_watch.py:601` | completed-unmerged respawn v1 bounds — "fail-toward-silence; /daily stays the backstop" |
| `autonomous_session_watch.py:614` | unmerged `origin/main` otherwise waits ~24h for the nightly Step C sweep |
| `autonomous_session_watch.py:6958` | abandoned-session phrase detection — "the /daily sweep owns" |
| `autonomous_session_watch.py:8144` | "each fails toward SILENCE by design; the /daily sweep stays the backstop for all four" |
| `autonomous_session_watch.py:9945` | Step 9c known-red ledger refresh |
| `.claude/workflow.yaml:2222` | living-docs backstop for parked proposals |
| this file, §833/§872 | completed-unmerged v1 bounds; Step 9c gate ledger |

Closing these needs either explicit escalation in each lane or a narrower
replacement sweep. Do NOT re-add `/daily` to close them without addressing
the backlog-growth problem that caused the retirement (see the still-parked
**#1737** "should /daily cap its nightly route-2 filing volume" and **#2070**
"61-deep auto-dispatchable infra queue").

**Kept deliberately:** `scripts/daily_drive_filings.py`,
`scripts/file_infra_task.py`, `scripts/sweep_parked_wf_candidates.py` — used
by the workflow-fix-on-bug protocol independently of the retired skills.
The PM digest line is fail-safe by spec (omitted silently when the dated file
is absent), so it simply stops appearing; the `Held by /daily` enumeration
stays valid because those 28 `daily-held` tasks still exist.

---

CLAUDE.md § Pods carries the always-on one-paragraph summary; this file is
the predicate spec. The `autonomous_session_watch.py` module docstring is
the lint-pinned canonical pass enumeration
(`workflow_lint --check-asw-docstring-pass-count`); long-form pass
narratives compressed out of this file live in its git history
pre-2026-08-05.

## Stale-pod audit (09:37 daily + on `pod.py provision`)

Auto-terminate pods EXITED >24h — EXEMPT when the owning task carries the
`keep-running` tag (reported `kept-exited`).

**Team-shared account — the EXITED auto-terminate is positively
ownership-gated for EVERY pod name (#1404/#1471).** The RunPod account is
TEAM-SHARED: non-EPS teammates may run pods whose names carry the managed
`pod-` prefix, so NO name is proof of EPS ownership. The auto-terminate
fires only when `_is_eps_owned` (`scripts/pod_audit.py`) POSITIVELY
confirms ownership via any of three signals — the parsed issue number
resolves in `tasks/REGISTRY.json`, the pod appears in the
`pods_ephemeral.json` sidecar, or `_scan_task_references` finds task
references — each wrapped fail-toward-keep. An EXITED pod failing all
three surfaces REPORT-ONLY in the `unmanaged-exited` bucket ("NEVER
auto-terminated" advisory); termination there is the user's call.

## Stale-GCP-VM janitor (09:37 daily, `cron_gcp_audit.sh` → `gcp_audit.py`)

The GCP analogue — the credit-leak backstop for the WHOLE
`eps-persona-gpu-jun2026` project (#688; a non-`eps-issue-*` leftover was
invisible to the old name filter, #680). Wraps
`backends.gcp.audit_stale_gcp_vms`; catches instances that escaped the
canonical ephemeral teardown.

**Classification + routing (HYBRID posture).** Lists the whole project
(`JANITOR_LIST_NAME_FILTER = None`), classifies by name:
- **`managed`** (`eps-issue-*`) → AUTO-DELETE on the fences below.
- **`allowlisted-ephemeral`** (`_EPHEMERAL_REAP_PREFIXES`, default
  `eps-cap-probe*`) → AUTO-DELETE on the same fences.
- **`unmanaged`** (anything else) → WARN-and-ESCALATE, never auto-deleted:
  Telegram push (fail-soft) + a durable sidecar row at
  `.claude/cache/gcp-janitor-events.jsonl` (`action="would-escalate"`
  report-only / `"escalated"` under `--delete`). The warn-don't-delete
  posture (#679).
- **`keep`** (`_JANITOR_KEEP_PREFIXES`, empty today) → never reaped or
  escalated; emits a `skipped` record.

The router seams (`reconnect_or_none` / `_stale_named_instance_or_none`)
keep their EXACT `name=eps-issue-<N>` filters — only the janitor's
inventory query broadens.

**Reap predicate** (two bounded fences): (1) per-instance-fence-aware age
backstop (#741) — a readable `scheduling.maxRunDuration` exceeded + 1h
grace (`_JANITOR_FENCE_GRACE_SECONDS`), OR no readable fence and past
`--max-age-hours` (default 192h = 8d; tracking each instance's OWN fence
avoids re-creating #697, a 7d job killed by a blanket 24h cap); (2) 10-min
terminal-phase reap (`--terminal-phase-max-age-min`) — a RUNNING instance
that published a terminal `eps/phase` but never auto-deleted is a wedged
zombie (#634); a guest-attribute probe FAILURE falls through to the age
backstop, never escalates to delete.

**Report-only by default** — the cron passes `--delete` (the only real
reaper AND the only mode that fires escalations);
`EPS_GCP_JANITOR_DRY_RUN=1` forces report-only even with `--delete`. An
escalated unmanaged VM keeps rc=0.

**Disarmed-janitor alarm (list-preflight).** The reaper swallows a
non-zero `gcloud instances list` rc and returns `[]`, so the CLI runs its
own list-preflight (byte-identical `render_list_argv` probe): non-zero rc
⇒ no reap, exit **3** (`list-failed`), the cron propagates rc=3 so the
disarmed-janitor email is delivered; rc=2 (`delete-failed`) and rc=0 stay
`exit 0`. **Exit codes:** 0 clean / 2 delete-failed / 3 list-failed.

**Env:** `EPS_GCP_JANITOR_DRY_RUN=1`, `EPS_GCP_JANITOR_LOG_DIR` (default
`logs/gcp_audit/`), `EPM_TELEGRAM_PUSH_SCRIPT`, `EPM_GCP_JANITOR_SIDECAR`.
Output: dated logs + a first-run-of-day pointer line (same liveness
mechanism as `cron_pod_audit.sh`).

## Stale-worktree sweep (09:47 daily, `worktree_audit.py --apply`)

Reaps idle auto-generated worktrees under `.claude/worktrees/` — only when
not held by a live process, not an `issue-<N>` at a non-terminal status,
older than a 6h grace (1h when the holding filesystem is ≥90% full,
`EPM_WORKTREE_DISK_PRESSURE_PCT`), and with no uncommitted tracked
changes. Human-named worktrees are never touched; `issue-<N>-<suffix>`
follow-up worktrees ARE in scope (mapped to issue N). For done-and-merged
(`completed`/`archived`/`awaiting_promotion`) issue worktrees, `--apply`
additionally kills orphaned codex `app-server` holder pids (exact-pid,
cmdline re-verified before each signal; never when a real holder is
present) and rescue-copies allowlisted runtime-noise dirt (agent memories,
`pods.conf`, `pods_ephemeral.json`) to
`.claude/cache/worktree-rescue-<date>/` BEFORE removal; dry-run only
classifies.

**Venv-reap arm (#912).** On every `--apply`, a KEPT worktree gets its
`.venv` reaped when idle ≥7 days (2 under disk pressure;
`EPM_WORKTREE_VENV_IDLE_DAYS`) AND no process holds it (cwd/argv harvest +
`/proc/<pid>/exe` probe; idleness = max(root, `.venv`, git-admin mtimes),
failing toward keep). Rename-aside → post-rename holder re-check → rmtree,
never wider than `<wt>/.venv`; underscore-prefixed managed worktrees and
symlinked roots/venvs never touched (a `.venv` is a pure build artifact
`uv run` regenerates). Kill switch: `EPM_WORKTREE_VENV_REAP=0`. Reaping a
venv drops worktree-side hardlinks; the shared blocks free when the daily
`uv cache prune` drops unreferenced cache-side entries.

**Husk-reap arm (#1430).** The same cron runs
`task.py reap-husks --apply`: a task id holding MORE THAN ONE on-disk
`tasks/<status>/<id>` dir (the merge-reintroduced husk shape) whose
REGISTRY status is TERMINAL (`completed`/`archived`;
`HUSK_REAP_TERMINAL_STATUSES` — `blocked` excluded) has its husk(s)
removed iff EVERY husk entry is byte-subset-verified against the live dir
(byte-identical, byte-prefix, `.jsonl` ordered-subsequence, or matching
symlink). ANY unique content ESCALATES (stderr ERROR + sidecar
`.claude/cache/husk-reap-events.jsonl`) and is NEVER deleted; the subset
check doubles as the misidentification guard (a stale registry entry
pointing at the husk makes the true live dir fail subset verification and
escalate). Tracked husks removed via `git rm -r` + one explicit-path
commit per husk under the task.py flock; untracked husks via `rmtree`.
On-demand: `task.py reap-husks [--apply] [--issue N]` (exit 0 always).
Kill switch: `EPM_SKIP_HUSK_REAP=1`.

`codex_task.py` complements this by pinning every codex-companion dispatch
to the main checkout root (`DISPATCH_ROOT`), so codex workers never root
themselves in a worktree.

## Autonomous-session watcher (every 10 min, `3-59/10 * * * *`, `autonomous_session_watch.py`)

Passes: crash-recovery respawn, pod-safety reconciliation, stalled-session
detector, orphan-file sweep, the infra-drain pass, the capacity-retry
pass, the stale-blocked flag pass, the gate-push pass, the
program-orchestrator recovery pass, the boot-death pass, the
stale-registration pass, the CPU/memory-pressure guard pass, the
triage-observer pass, the orphan-wrapper /proc sweep, and three session
reapers — the session-vs-status reconcile pass, the zombie-wrapper pass,
and the idle-unmapped pass.

**Stall-detection hardening (#845 + follow-ons).** The stalled detector +
the two respawn arms carry these mechanisms:

- *(a-i) Marker-heartbeat window.* The newest non-watcher marker has its
  OWN 2h freshness window (`EPM_STALLED_MARKER_HEARTBEAT_MIN`, default
  120 min) — a session that posted ANY non-watcher marker within 2h is
  never declared stalled (#761/#763). Trade-off: third-party markers can
  shield a wedged session for up to 2h; the (e) wedge fast lane is the
  mitigation.
- *(a-ii) Stop-verify respawn fence* (`decide_respawn_fence`): the stalled
  respawn arm NEVER spawns in the same tick it stops a session — stop →
  verify the sid is absent from the daemon live set on the NEXT tick →
  spawn (#763: a same-tick stop+spawn overlapped two drivers ~4h). One
  stop retry, then a one-time loud `[...:session-stop-failed]` marker +
  push. A pending fence whose sid no longer matches the registry entry
  CLEARS itself; the stalled arm skips within `RESPAWN_SPAWN_GRACE_S` of a
  fresh registration.
- *(b) Bounded worktree-activity hold* (`decide_worktree_hold`): BOTH
  respawn arms defer while any file under `.claude/worktrees/issue-<N>*`
  (excluding `data/`) has an mtime within 15 min
  (`EPM_STALLED_WT_ACTIVITY_MIN`) — direct evidence of a mid-edit
  implementer (#812) — bounded at 6 consecutive held ticks (~1h) with
  `missed` pinned at threshold so the arm re-fires when activity quiets.
- *(c) Daemon retry + blocked-recovery escalation.* The per-tick daemon
  probe retries (3 attempts, `EPM_DAEMON_PROBE_ATTEMPTS`) before declaring
  an outage; a respawn-worthy stall deferred by daemon-unreachable ≥2
  consecutive ticks fires ONE Telegram push per episode (#811).
- *(e) Prompt-wedge fast lane* (`decide_prompt_wedge`): a transcript-tail
  probe (last 256 KB — #1104 widening) escalates straight to the respawn
  arm on wedge evidence. Triggers: ≥3 consecutive trailing dequeue/prompt
  rows with no assistant turn (`EPM_TICK_WEDGE_MIN_DEQUEUED`; #779);
  ≥3 consecutive api-error rows (`isApiErrorMessage: true` assistant rows
  are FAILED turns, not resets — own counter
  `EPM_TICK_WEDGE_MIN_API_ERRORS`, `0` disables; #1104/#1074); **#1127
  turn-level failed-wake counting** — tail rows segment into WAKE-TURNS
  (`_segment_wake_turns`; a completed turn whose LAST response row is an
  api-error is a FAILED wake even when mid-turn heartbeats preceded it):
  (a) ≥3 consecutive trailing failed turns
  (`EPM_TICK_WEDGE_MIN_FAILED_TURNS`, default 3, `0` disables); (b) an
  alternating-storm RATE trigger — ≥6 failed turns
  (`EPM_TICK_WEDGE_MIN_FAILED_TOTAL`) within 120 min
  (`EPM_TICK_WEDGE_RATE_WINDOW_MIN`, anchored to the newest ROW ts) with
  the newest completed turn failed; (c) the turn-level lanes are probed
  EVERY tick while the dequeue-run / row-level api-error triggers stay
  STALENESS-GATED (both turn knobs at `0` = the pre-#1127 lazy gate).
  **#1209 dead-wake trigger (`failed-turn-silence`):** a session dead
  after a SINGLE refused turn accumulates no failed wakes, so a tail whose
  completed turns are ALL failed and whose newest parseable row ts is
  ≥ `EPM_TICK_WEDGE_DEAD_SILENCE_MIN` (default 20 min; `0` disables) older
  than the pass clock escalates to the fence STOP (one-time
  `session-dead-silence-stop` marker; the CRASH-RECOVERY arm completes the
  respawn ~20-30 min post-stop). Bounded by the episode belt
  (`respawn_count < STALLED_MAX_RESPAWNS`) AND a per-issue per-UTC-day cap
  `EPM_TICK_WEDGE_DEAD_RESPAWNS_PER_DAY` (default 3), bumped once per
  fence episode at stop-initiation, advancement-clear-EXEMPT; a
  cap-disarmed trigger goes quiet, the slow stalled lane stays the
  backstop. A prior ok turn in the tail / zero completed turns / a ts-less
  tail all fail toward NO-FIRE. **#1241 cap parity:** the `dequeue-run` /
  `api-error-run` / `failed-turn-run` / `failed-turn-rate` respawns are
  bounded by the SAME two-part gate — episode belt AND per-issue
  per-UTC-day cap `EPM_TICK_WEDGE_RESPAWNS_PER_DAY` (default 3), its own
  day-keyed counter in `stalled-<N>.json`, bumped once per wedge-initiated
  fence episode at stop-initiation (a stop-failed episode still consumes a
  unit); the two day budgets (#1209 vs #1241) are INDEPENDENT; when every
  family is cap-disarmed the 256 KB transcript read is skipped.
  **#1453 context-ceiling trigger:** ONE trailing `<synthetic>` api-error
  row naming the context ceiling (substring `prompt is too long`)
  escalates immediately — the failure is deterministic (append-only
  conversation, every later turn fails identically; #1335). Armed on both
  self-report paths; a later successful assistant row resets it; joins the
  #1241 shared day cap; kill switch `EPM_TICK_WEDGE_CONTEXT_CEILING=0`.
  The single-refusal guard (one refused wake's dequeue+prompt rows must
  not trip `run >= 3`) and fail-toward-NO-FIRE posture hold throughout.
  The lane bypasses the 2-miss debounce, the #759 K-downgrade and the 2h
  marker window — but NOT the park exemptions (provision-in-flight /
  followups / spend-approval — re-probed once against the escalated
  action), the worktree hold, or the fence; a wedge respawn resets
  `live_consecutive`; unresolvable transcripts fail toward no-wedge.
- *(f) Alert-noise dedup (#1137).* At most ONE `session-stalled-alert`
  marker per staleness episode, across both producers (decide()'s alert
  path and the #759 K-downgrade lane), cleared on self-report advancement.
  A `blocked` task whose newest status-changed is corroborated by an
  epm:failure within the park window is never stalled-alerted (the
  gate-push pass already pushed the transition). Accepted residual: a
  capacity-retry-eligible `no_compute_available` blocked task carries the
  halt trail by definition, so a re-driven session that wedges before
  leaving `blocked` gets no stalled alert — the crash-recovery / wedge /
  stale-registration lanes own that class.
- *(g) Stalled-manual escalation rung (#1480).* A MANUAL registration
  (`manual-issue-<N>.json`) on an ACTIVE-status task whose (f) alert has
  gone UNACTIONED — ≥ `EPM_STALLED_MANUAL_ESCALATE_CONFIRMS` (3)
  consecutive stalled-confirmed ticks spanning ≥
  `EPM_STALLED_MANUAL_ESCALATE_H` (24h) with zero non-watcher task
  progress — escalates to an UNREGISTER-ONLY action: the registration file
  is deleted (the session itself is NEVER stopped — #505 stands) + a loud
  `[autonomous_session_watch:stalled-manual-escalation]` marker + sidecar
  row (`~/.eps-autonomous/stalled-manual-escalation-events.jsonl`) +
  Telegram push; the orphan sweep then re-drives the task (~20–30 min;
  #928: a wedged manual session froze a `followups_running` task ~6
  days). Fire-time vetoes: fresh worktree activity DEFERS; the
  park-exemption re-probe VETOES + resets counters. Once per episode by
  construction; autonomous `issue-<N>.json` entries never enter the rung;
  every unresolvable input fails toward keep. Kill switch:
  `EPM_DISABLE_STALLED_MANUAL_ESCALATION=1`.

Per-episode state rides `stalled-<N>.json`, cleared on self-report
advancement; pre-#845 files load with safe defaults. While a fence episode
is pending, the #759 K corroboration is skipped.

**Terminal-status act guard + source stamp (#1247).** The orphan sweep and
the stalled fence's spawn branch act only after a same-instant live-status
re-read (`_task_status`) POSITIVELY returns an ACTIVE status — a stale
pass-start snapshot can never produce a respawn/marker/cap-consume on a
terminal/parked task (the #662/#663/#867 marker-loop class). Abort is loud
(`ORPHAN-ACT-GUARD` / `STALLED-ACT-GUARD` stderr); a `None` read defers
one tick; a positive non-ACTIVE read clears episode state. Every
orphan/stalled marker note carries a `[src: host=… user=… pid=… sha=…
root=…]` stamp (`_source_stamp()`) so a stale-instance poster identifies
itself. Pinned by
`tests/test_autonomous_session_watch.py::test_orphan_act_guard_*` +
`test_stalled_fence_spawn_guard_*`.

**Orphan-sweep dead-owner fast path (#1391).** The orphan sweep's 90-min
staleness floor drops to `EPM_ORPHAN_DEAD_OWNER_STALENESS_MIN` (default
20 min, clamp ≥10, `0` disables) when the sweep has POSITIVE proof the
task's last recorded owner session is dead — the #720
`last-mapped-terminal-<sid>` breadcrumb (or state-carried `owner_sid`)
binds a sid observed live on an earlier tick that is now absent from a
successfully-fetched daemon live set — plus a recent driver witness (a
follow-up-round marker kind or `stage-dispatch` breadcrumb) and no
live-owner veto (fresh worktree activity, a live daemon child on the
issue worktree, any breadcrumb owner still live). Every uncertain input
fails toward the 90-min slow path; the debounce, daily cap, #505
manual-only alert, and act-time guard bind unchanged.

**Infra-drain pass (#633).** Executes the PM's dispatch queue
(`~/.eps-autonomous/infra-drain-queue.json`: `ripe_oldest_first`, `cap`
default 5, `holds`, `updated_ts`) with zero LLM judgment: spawns
`spawn-issue --auto` for the oldest listed `proposed` `kind: infra|batch`
IDs into free slots (free = cap − occupied − pending; occupied = drain-kind
tasks at the seven body statuses + `followups_running`, read fail-CLOSED;
pending = non-stale registrations of still-`proposed` drain-kind tasks).
Per-ID guards (each with a logged skip reason): PM hold; existing
registration — a STALE dead-at-boot registration (task still `proposed`,
older than `EPM_INFRA_DRAIN_STALE_REG_GRACE_S` default 30 min, sid
definitively not live) stops pinning a slot, ANY missing signal fails
toward keep-blocking; status ≠ `proposed`; kind outside `{infra, batch}`
(loudly logged — a mis-kinded entry would auto-approve GPU spend); a retry
budget (`EPM_INFRA_DRAIN_MAX_ATTEMPTS` default 3 per adjudication epoch)
whose backoff window (`EPM_INFRA_DRAIN_BACKOFF_S` default 1h) ALWAYS
binds. The PM remains the only nuanced ripeness judge; a missing/invalid
queue file is a logged no-op. Dispatch markers carry the
`[autonomous_session_watch:infra-drain-dispatch]` sentinel (never resets
staleness clocks). State: `~/.eps-autonomous/infra-drain-state.json`. Kill
switch: `EPM_DISABLE_INFRA_DRAIN=1`; `--infra-drain-only` + `--dry-run`
for a live smoke.

*Session-dispatch stagger (#1059).* Consecutive REAL `spawn-issue --auto`
dispatches from the two infra loops + `file_infra_task.py` are paced
≥ `EPM_SESSION_DISPATCH_STAGGER_S` apart (default 60s; clamped ≤300) via
the shared stamp `~/.eps-autonomous/last-session-dispatch.json` (each
fresh session is a ~100K-token cold load against the org input-TPM cap).
The watcher SLEEPS out the remainder (safe — the `watch.lock` flock makes
an overlapping cron fire skip); the filer DEFERS (file-then-no-op, exit 0;
the sweep backstop dispatches within ~10 min). Only a real `"spawned"`
outcome records the stamp. Residuals: check→spawn→record is a TOCTOU
(pacing, not exclusion — bounded ~2 coincident cold loads);
crash-recovery, capacity-retry, campaign, and manual spawns are accepted
UNPACED sources.

*Predicate-hold auto-promotion (#633 follow-on).* Before dispatch, the
pass promotes any `holds` entry matching the PM's
`predicate-<#N>-<short-desc>` convention once task #N reaches
`completed`/`archived`/`awaiting_promotion` (the `<short-desc>` is never
interpreted): hold removed, id merged into `ripe_oldest_first`, queue file
rewritten atomically (`updated_by:
autonomous_session_watch:predicate-promote`, budget re-armed). One
`task.py view` per distinct predicate; a non-predicate / malformed /
not-yet-terminal hold is left untouched. Skipped under `--dry-run`. The
PM's own STATUS-pass re-adjudication always wins a race.

**Capacity-retry pass (#642).** The narrow inverse of the crash-recovery
PARK rule ("every `blocked` task stays parked"): re-drives via
`spawn-issue --auto` ONLY the subclass whose latest `epm:failure v1` is
`failure_class: infra` + `reason` ∈ `TRANSIENT_CAPACITY_REASONS` (today
`{no_compute_available}`) — code-ready, re-runnable when a lane frees.
Every other `blocked` task is untouched (a non-capacity infra reason stays
parked). *No watcher-side capacity pre-check by design:* the re-driven
`/issue` launch re-runs the router's own quota gate and simply re-blocks
at zero GPU cost if lanes are still full. *Churn guards:* per-task backoff
(`EPM_CAPACITY_RETRY_BACKOFF_S`, default 1h, on the newer of block ts and
last attempt) + per-UTC-day cap (`EPM_CAPACITY_RETRY_PER_DAY`, default 4;
attempts count regardless of spawn success) — at exhaustion, one alert per
day. Daemon-gated; state `~/.eps-autonomous/capacity-retry-<N>.json`
(`blocked` deliberately NOT in `TERMINAL_FOR_GC`). `kind: campaign`
blocked tasks excluded. Markers carry
`[autonomous_session_watch:capacity-retry]` /
`[...:capacity-retry-exhausted]` sentinels. Kill switch:
`EPM_DISABLE_CAPACITY_RETRY=1`; `--capacity-retry-only`.

**Stale-blocked flag pass (#1021, incident #742).** The capacity-retry
pass's non-spawning, daemon-INDEPENDENT sibling: a crash-fix relaunch that
succeeds on a task an earlier round parked at `blocked` leaves the status
stale (#742 ran healthy ~35h at `blocked`). Predicate
(`decide_stale_blocked_flag`, every missing signal → silence): status
`blocked` AND the latest `epm:run-launched` NEWER than the transition into
`blocked` AND real (non-watcher, non-deliberate-stop) progress at/after
that launch within `EPM_STALE_BLOCKED_PROGRESS_FRESH_S` (default 2h). On a
hit it FLAGS — one deduped `epm:progress` marker naming the reconcile
command (`task.py set-status <N> running`), one sidecar row
(`~/.eps-autonomous/stale-blocked-events.jsonl`), one Telegram digest line
— and NEVER mutates status. Dedup per launch episode
(`stale-blocked-<N>.json` records the flagged launch ts; a NEWER launch
re-alerts). Sentinel `[autonomous_session_watch:stale-blocked-flag]`. Kill
switch: `EPM_DISABLE_STALE_BLOCKED_FLAG=1`; `--stale-blocked-only`.

**Gate-push pass.** Telegram push on gate-park/`blocked` transitions
(my-goat `telegram_push.sh`; override `EPM_TELEGRAM_PUSH_SCRIPT`),
transition-deduped via `~/.eps-autonomous/gate-notify-<N>.json`: fires
exactly once per transition INTO a user gate (`awaiting_promotion`,
`blocked`, or `plan_pending` only when the plan-gate park marker
(`epm:awaiting-spend-approval` — fired on a missing/unparseable estimate,
the sole autonomous park cause since #1771) confirms it — shared
`plan_pending_over_cap` predicate (historical name) with
`tick_triage.py`). Covers CAMPAIGN registrations too; because the respawn
/ campaign passes delete registrations on the first tick observing a
terminal park, the watcher snapshots issue + campaign registrations
BEFORE those passes and hands them in (`issue_snapshot=`). The tick-side
`PushNotification` is KEPT as a second deduped channel (worst case one
duplicate, never a miss). The same pass runs a
**status-transition-keyed title/self-report reconcile** — never per-pass
(an unconditional rewrite would keep the self-report fresh and disable the
staleness signals; a status-change-keyed rewrite cannot mask a stall) —
and owns the **tick-runaway force-stop parachute** (#501 class):
`tick_triage.py` writes `tick-runaway-<N>.flag` on the 3rd consecutive
teardown-verdict tick, and this pass force-stops the flagged issue's
session(s) under the session-reconcile guards (DONE statuses only, no live
follow-up, no RUNNING pod, no `keep-running` tag) WITHOUT the 2h-idle
accumulation. A `blocked` task's runaway flag alerts loudly, never stops
(the user may be live-parked in it). `gate-notify-<N>.json` is in the
terminal-status GC sweep; runaway flags self-clean.

**Reconcile pass (auto-stop of parked sessions).** An issue-mapped session
whose task is parked/terminal (`awaiting_promotion`/`completed`/`archived`)
is AUTO-STOPPED after ≥2 consecutive checks once ALL hold: no live
follow-up inferred from events.jsonl (latest
`epm:run-launched`/`epm:followup-scope`/`epm:free-analysis-followup-run`
OLDER than the latest done-transition), every non-watcher marker +
self-report idle > ~2h (`EPM_SESSION_RECONCILE_IDLE_S`), no RUNNING
`pod-<N>`, no `keep-running` tag (`EPM_SESSION_RECONCILE_AUTOSTOP=0`
reverts to alert-only). Sessions at any other status, the PM session, and
unmapped chat sessions are never touched by this pass.

**Keep-running wedged-owner escalation arm (#1582).** ESCALATE-ONLY arm
inside the pod-safety pass's `keep-running-skip` branch — the blind spot
where the tag short-circuits every other check (#1345: a tagged pod billed
~72h on an `awaiting_promotion` task behind a frozen owner). Predicate
(`decide_keep_running_owner_escalation`): a RUNNING tagged pod on a
DONE-status task whose real-marker progress gap ≥
`EPM_KEEP_RUNNING_WEDGED_OWNER_MIN_H` (12h) AND whose owning session is
provably WEDGED (every candidate owner's transcript idle ≥ the same
floor, self-report equally stale) or ABSENT (no live registration sid, no
cwd-mapped daemon child), confirmed ≥2 consecutive ticks. A
daemon-unreachable / unresolvable read is "unknown" and FREEZES the
counter (fail toward no-fire); cheap vetoes (provision-in-flight, fresh
worktree activity) clear first. Channels: ONE anti-liveness task marker
per episode (`[autonomous_session_watch:pod-keep-running-wedged-owner]`,
carrying the recovery recipe) + fail-soft push + sidecar
`.claude/cache/keep-running-wedged-events.jsonl`; push+sidecar re-fire
every `EPM_KEEP_RUNNING_WEDGED_REALERT_H` (24h). **ESCALATE-ONLY is a hard
invariant** (pinned by
`tests/test_autonomous_session_watch_keep_running_owner.py::test_never_stops_or_terminates`).
Kill switch `EPM_DISABLE_KEEP_RUNNING_OWNER_AUDIT=1`.

**Pod-grain idleness leg (#2149, same arm — alert-only, never a stop).**
The owner leg's leg-1 predicate is TASK-grain, so a BUSY multi-round task
never opens the 12h gap and an idle shielded pod stays structurally
invisible (#1739: 3 verified-done 1xH100 pods idled ~19.6h / ~$165 behind
129 sibling-round markers, largest gap 6.19h). On every tick where the
owner leg did NOT escalate/re-alert (its busy-task early-clear AND its
post-decide non-fire paths alike), the pod's OWN evidence is read via one
bounded SSH probe (`BatchMode` + `ConnectTimeout=10` + an outer timeout;
one fixed `k=v` line: newest `/workspace/logs/` mtime, newest terminal
`*done*.json` sentinel age, max GPU util) and decided by the pure
`decide_keep_running_pod_idle_escalation`. **Sentinel tier** (floor
`EPM_KEEP_RUNNING_POD_IDLE_MIN_H`, 4h): done-sentinel AND workload log
both ≥ the floor stale, GPU util 0%/unreadable. **Utilization tier**
(floor `EPM_KEEP_RUNNING_POD_UTIL_IDLE_MIN_H`, 12h; only when NO
done-sentinel exists): MEASURED 0% util + log ≥ the floor stale — an
unreadable util can never fire this tier. Per-POD episode state (the
`kr_pod` pod_id-keyed sub-dict of the pod-safety state file — a busy
sibling pod's saves forward-carry an idle pod's counter verbatim, the
multi-pod #1739 shape), ≥2 consecutive ticks, any unreadable probe field
FREEZES the counter (a run of `EPM_KEEP_RUNNING_POD_PROBE_FAIL_ROWS`
(6) consecutive probe failures leaves one durable sidecar row). Channels:
the owner leg's marker/push/sidecar plumbing with `leg="pod-idle"`
(sentinel `[autonomous_session_watch:pod-keep-running-idle-pod]`, sidecar
rows `kind="keep-running-idle-pod"` in the same jsonl), re-alert on the
shared `EPM_KEEP_RUNNING_WEDGED_REALERT_H` (24h). **Alert-only — this leg
NEVER stops or terminates anything** (same hard invariant; pinned by
`test_1739_pod_idle_never_stops_or_terminates`). Leg disable flag
`EPM_DISABLE_KEEP_RUNNING_POD_IDLE=1`; the arm-wide
`EPM_DISABLE_KEEP_RUNNING_OWNER_AUDIT=1` covers both legs. Assumption: the
log-mtime read is `/workspace/logs/`-ONLY — a pod whose workload logs
elsewhere can false-fire the utilization tier (alert-only, diagnosable in
one read). Named residual: a done-sentinel + stale logs + GPU SPINNING
(a hung NCCL collective reads ~100% util) fires neither tier — the
measured-util conjuncts fail toward no-fire by design.

Owner-leg residuals: any
third-party non-watcher marker resets the progress-gap clock
(conservative; the #2149 pod leg is exactly the cover for the busy-task
face of this); a tagged pod on an ACTIVE task keeps the one-shot
`pod-active-stale` alert; GCE instances are bounded by their fences + the
janitor, not this arm.

**Never-ran escalation leg (#2060, incident #1947 — ESCALATE-ONLY, never a
stop/terminate).** A RUNNING keep-running-tagged pod that NEVER exposed a
runtime/port since creation (the `runtime: null` bootstrap-failure class;
raw predicate = `backend_poll._pod_is_runpod_runtime_wedged`, observed
portless on EVERY tick since a first observation within
`EPM_NEVER_RAN_OBSERVE_SLOP_S` (1200s) of `created_at`), past
`EPM_NEVER_RAN_GRACE_MIN` (45 min) and confirmed ≥2 consecutive ticks, is
surfaced loudly: one task marker per episode (sentinel
`[autonomous_session_watch:runpod-never-ran-keep-running]`) + a fail-soft
push + a sidecar row (reason `never-ran-keep-running-escalate`, the #1582
jsonl) — each naming the pod + pod_id, its age, the hourly burn, and the
exact `pod.py terminate --issue <N> [--name-suffix <slug>] --yes --approve`
recovery command; re-pushed every `EPM_NEVER_RAN_REALERT_H` (6h, deliberately
shorter than the 24h house cadence — a never-bootstrapped pod is a pure
billing leak). Evaluates where the raw wedge predicate first reads True,
BEFORE the DONE-status split, so it fires on the #1947 tagged
`awaiting_promotion` shape both sibling legs structurally miss (a portless
pod is SSH-unreachable, so the #2149 probe can never fire; busy-task marker
traffic keeps the #1582 owner gap closed). Per-(issue, pod_id) state
(`nr_pod` sub-dict of the pod-safety file, the `kr_pod` pattern); a port
appearance DELETES the entry; every missing signal fails toward silence.
**ESCALATE-ONLY is a hard invariant** (pinned by
`tests/test_autonomous_session_watch.py::test_never_ran_leg_never_stops_or_terminates`).
Kill switch `EPM_DISABLE_NEVER_RAN_ESCALATION=1`. Provision-side sibling:
`pod.py provision` now tears the pod down ITSELF by default on a
bootstrap/ssh-wait failure (`--keep-on-bootstrap-failure` opts out), so this
leg is the backstop for pods that escape that path.

**Zombie-wrapper pass.** A daemon-tracked session whose process tree has
carried NO inner Claude process for ≥2 consecutive checks AND ≥ the
lane's grace is auto-stopped regardless of issue mapping. Two
stop-eligible lanes (#1039): **EPS cwds** at 2h grace
(`EPM_ZOMBIE_WRAPPER_GRACE_S`); **non-EPS cwds** under stricter gates — 7d
grace (`EPM_ZOMBIE_NONEPS_GRACE_S`), NO live user TTY
(`_is_live_user_tty`, fail toward keep), wrapper age ≥7d
(`proc_start_epoch` belt), not registry-mapped. An unresolvable-cwd sid
lands in the **unresolvable bucket**: age-reported (stdout line + one
deduped row in `~/.eps-autonomous/zombie-wrapper-events.jsonl` once ≥7d
old) and NEVER auto-stopped. Never touched: the PM session (checked FIRST),
issue-mapped EPS sessions at active/`blocked`/`plan_pending`,
registry-mapped-but-non-EPS-cwd sessions, non-EPS wrappers holding a live
TTY. Kill switches: `EPM_ZOMBIE_WRAPPER_REAP=0` (both lanes alert-only),
`EPM_ZOMBIE_NONEPS_REAP=0` (non-EPS lane only). A stopped non-EPS session
is recoverable (`happy claude` in its cwd + `claude --resume`).

**Orphan-wrapper /proc sweep (#1215, `orphan_wrapper_pass`;
ESCALATE-ONLY by default).** The daemon-INDEPENDENT enumeration complement
of the zombie pass: a wrapper/launcher the daemon no longer tracks is
invisible to every daemon-sourced reaper. ONE /proc scan enumerates
candidates by cmdline signature (`happy/dist/index.mjs` + next argv token
`claude` = `wrapper`; `claude_local_launcher.cjs` = `launcher`;
`comm == "node"`; euid-owned; daemon pid excluded twice; TOPMOST-deduped).
**Conjunctive escalation guards, ALL required:** daemon reachable AND pid
untracked (incl. ancestor/descendant live-set intersection), no Claude
descendant, delta-CPU < 1%/core between ticks
(`EPM_ORPHAN_WRAPPER_CPU_FRAC_MAX`; no baseline ⇒ keep), age ≥24h
(`EPM_ORPHAN_WRAPPER_MIN_AGE_S`), no live user TTY (with `check_orphaned`
threaded from `EPM_ORPHANED_TMUX_REAP`), ≥2 consecutive ticks (per-pid
state `~/.eps-autonomous/wrapper-orphan-<pid>.json`, keyed
`(pid, start_epoch)`). **Default action = escalate-only** (polarity
INVERTED vs the zombie/idle reapers — the stop is a direct SIGTERM to a
pid the daemon cannot see, and a daemon-restart-orphaned-but-revivable
idle wrapper is a real residual): one sidecar row per episode in
`~/.eps-autonomous/orphan-wrapper-events.jsonl` (ppid/parent-cmdline/tty/
cpu forensics — init-parentage is the true-orphan fingerprint,
unobtainable post-stop) + ONE batched summary push per tick. Review rows
before opting into the stop arm. **Opt-in stop arm:**
`EPM_ORPHAN_WRAPPER_REAP=1` ANDed with global `EPM_ZOMBIE_WRAPPER_REAP`,
plus ≥7d observed orphanhood (`EPM_ORPHAN_WRAPPER_GRACE_S`;
`first_miss_ts` pins at first persisted observation — doubles as a
post-deploy bake window), ≤3 SIGTERMs/tick
(`EPM_ORPHAN_WRAPPER_MAX_STOPS_PER_TICK`), pre-signal signature +
start-epoch re-verification (pid-reuse belt), next-tick stop verification
with one SIGTERM retry then a loud stop-failed row + push — never SIGKILL.
Daemon unreachable ⇒ one skip line, state frozen, never signals. No task
markers by construction. Kill switch `EPM_DISABLE_ORPHAN_WRAPPER_PASS=1`;
`--orphan-wrapper-only` (in-invocation two-sample CPU delta,
`EPM_ORPHAN_WRAPPER_SMOKE_INTERVAL_S`).

**Idle-unmapped pass.** The third session reaper — auto-stops UNMAPPED
EPS-cwd sessions (no registry entry, no `issue-<N>` worktree cwd) whose
resolved transcript has been idle ≥12h (`EPM_UNMAPPED_IDLE_REAP_S`) on ≥2
consecutive checks — the class both other reapers structurally exclude.
Never touched: the PM session, non-EPS cwds, issue-mapped sessions,
wrappers holding a controlling TTY, unresolvable transcripts (a missing
idleness signal FAILS TOWARD KEEP). `EPM_UNMAPPED_IDLE_REAP=0` reverts to
alert-only; records → `~/.eps-autonomous/idle-unmapped-events.jsonl`.
**#818 orphaned-tmux subclass:** a pane on an ORPHANED tmux server (socket
deleted from `$TMUX_TMPDIR/tmux-<uid>` AND the server holds zero
`/dev/pts` attached-client fds — BOTH signals required; a
socketless-but-attached server is KEPT) is NOT a live terminal, so the
idle-≥12h wrapper on it is reaped on the same schedule. Mapping is process
PARENTAGE (walk the wrapper's `ppid` chain for a `comm == "tmux: server"`
ancestor), not the server's fd table; every uncertain probe FAILS TOWARD
KEEP. Gated by default-ON `EPM_ORPHANED_TMUX_REAP` (`=0` disables).
**#720 short-window subclass:** an unmapped session whose LAST-mapped task
was TERMINAL (the respawn pass deletes `issue-<N>.json` at terminal → the
session goes unmapped) is reaped on the SHORT
`LAST_MAPPED_TERMINAL_REAP_S` window (default 30 min; worst case ~50 min)
via the #720 breadcrumb (`last-mapped-terminal-<sid>.json`) + the
running-pod + live-follow-up guards in `_effective_idle_reap_s`. The
reconcile pass cannot see this class (already unmapped by then) — this
window owns it.

**#1971 TTY-attached report lane (ESCALATE-ONLY).** A TTY-attached
unmapped EPS session is exempt from every stop/alert arm above
(`decide_idle_unmapped`'s pinned `has_tty -> ("clear", 0)` — a TTY may be
a terminal the user is sitting at), but multi-day accumulations previously
had ZERO observability. This lane REPORTS — guaranteed never to stop,
unregister, or mutate: a TTY-attached unmapped EPS session idle ≥
`EPM_TTY_UNMAPPED_REPORT_HOURS` (48h) is accumulated per pass and flushed
as ONE deduped fail-soft push per episode — dedup keyed on the reported
session-id SET (state `~/.eps-autonomous/tty-unmapped-report-state.json`,
written only on push as `prev ∪ cur`), re-pushed on set growth or a 168h
TTL (`EPM_TTY_UNMAPPED_REPORT_REALERT_HOURS`) — plus one sidecar row per
reported session (sentinel `[autonomous_session_watch:tty-unmapped-report]`,
the idle-unmapped events stream) carrying sid / pid / cwd / ages and a
safe-to-kill VERDICT from the work-descendant probe (a probe failure reads
"uncertain", never "safe"). Kill switch:
`EPM_DISABLE_TTY_UNMAPPED_REPORT=1`. Residual invisible classes
(deliberate): non-EPS-cwd TTY sessions; TTY sessions with unresolvable
transcripts.

**Stale-registration pass (#845 d).** UNREGISTERS a LIVE-but-abandoned
registration (`issue-<N>.json` OR `manual-issue-<N>.json`) whose resolved
transcript has been idle ≥12h (`EPM_STALE_REGISTRATION_IDLE_H`) AND whose
self-report is equally stale (a MISSING self-report does not rescue) —
#665: a 16h-idle registered session held the `/issue` Step 0
single-orchestrator guard and blocked every re-drive (the crash-recovery
pass can't help — the sid IS live; the idle-unmapped reaper excludes
MAPPED sessions; reconcile fires only on parked/terminal statuses).
UNREGISTER-ONLY: the session is NEVER stopped (may hold a user TTY; the
SKILL Step 0 stale-wake ownership re-check guards a later wake). For an
ACTIVE task the orphan sweep re-drives on its next tick (a PARK/terminal
task is deliberately NOT re-driven — the one-time
`[autonomous_session_watch:stale-registration-unregister]` marker logs the
status). Guards, all failing toward keep: dead sid, unresolvable
transcript, in-flight provision, fresh worktree activity, fresh
self-report. Runs AFTER `gate_push_pass`; daemon-gated. Trace:
`~/.eps-autonomous/stale-registration-events.jsonl`. The #1480 rung (§ (g)
above) covers the LIVE-WEDGED manual case this pass's transcript-idle gate
cannot (in-session activity defeats the 12h predicate; the rung keys on
TASK-level progress).

**Boot-death lane (#1267 arm 1 + #1287 arm 2, `boot_death_pass`).** The
die-at-or-before-turn-1 complement of the stale-registration pass: a
freshly dispatched AUTO session (`issue-<N>.json` only; manual excluded by
design) whose transcript EITHER (arm 1, `shape=zero-response`) contains
ZERO response rows OR (arm 2, `shape=boot-refusal`) segments to ≥1
completed turn with EVERY completed turn failed (a refusal-killed boot
turn; a single visible ok turn keeps) — ≥30 min after `spawned_at`
(`EPM_BOOT_DEATH_WINDOW_MIN`), transcript quiet ≥10 min, sid LIVE — is
STOPPED via `_stop_session` + surfaced (push + anti-liveness marker,
sentinels `[autonomous_session_watch:boot-death-stop]` /
`[...:boot-death-cap-exhausted]`) instead of waiting ~12h for the
stale-registration unregister (#1251–#1256: transcripts frozen seconds
post-spawn; #1277: a boot turn died on a refusal before the tick cron was
armed — every other lane structurally blind to both shapes). Arm 1's
whole-file read is bounded at 256 KB (larger ⇒ keep); every unresolvable
signal fails toward keep. STOP-ONLY — no unregister, no direct spawn:
post-stop re-drive is owned by the existing arms (ACTIVE →
crash-recovery; `proposed` → the proposed-infra sweep). Bounds: per-issue
per-UTC-day stop cap (`EPM_BOOT_DEATH_STOPS_PER_DAY`, default 3, shared
across both arms), bumped once at stop-initiation; state
`~/.eps-autonomous/boot-death-<N>.json`; trace
`~/.eps-autonomous/boot-death-events.jsonl` (stop rows carry transcript +
stderr + api-error forensics, SIDECAR-ONLY). NO episode belt BY DESIGN (a
stop lane, not a respawn lane; the downstream re-drive arms carry their
own belts, and the auth-outage guard suppresses the re-dispatch side
during an outage). At the cap the lane stops stopping and fires ONE loud
cap push/marker per (issue, UTC day) — a recorded deviation from #1241's
quiet-at-cap posture (here the fallback is the very 12h silence this lane
exists to kill). Runs AFTER `gate_push_pass`, BEFORE stale-registration;
daemon-gated. Kill switch: `EPM_DISABLE_BOOT_DEATH_PASS=1`;
`--boot-death-only`.

**Deliberate session takeover (`paused-takeover` sentinel; #866/#903).**
(Scope: a short-TTL session-TAKEOVER shield, NOT a user pause — an
indefinite "pause <N>" routes to `task.py set-status <N> on_hold`; a stale
sentinel FAILS OPEN at ~`EPS_TAKEOVER_TTL_H`.) To take over a stalled
autonomous session WITHOUT racing the watcher, rename its registration:
`~/.eps-autonomous/issue-<N>.json` →
`issue-<N>.json.paused-takeover-<YYYYMMDD>` (any suffix after the literal
`.paused-takeover-`; `manual-issue-` same shape). While the sentinel is
FRESH (mtime < `EPS_TAKEOVER_TTL_H`, default 6h; `touch` to renew): the
orphan-respawn pass SKIPS the issue, and `spawn-issue --auto` suppresses
with a rc-0 `TAKEOVER-SENTINEL HELD` line (recognized by
`spawn_output_suppressed`, so every automated arm books nothing). Manual
spawns warn-and-proceed. A STALE sentinel is ignored everywhere — FAIL
OPEN: crash recovery resumes at the TTL. The registration-KEYED passes
need no sentinel check: the rename removes the very file they key on.
Ending a takeover — ORDER MATTERS: FIRST re-establish a registration
(`spawn_session.py register-current --issue N`, or rename the sentinel
back), THEN delete the sentinel — deleting first opens a one-tick window
where the frozen `missed` count respawns immediately. Operational notes:
(a) `EPS_TAKEOVER_TTL_H` is a FLEET-LEVEL knob — a session-local export
never reaches the watcher's cron env; renew by `touch`. (b) Before
renaming, check for a fresh registration / dispatch lease — a respawn may
have just fired. (c) During a takeover the `/issue` Step 0
single-orchestrator guard is registration-keyed and therefore BLIND —
check for the sentinel before hand-driving. Stopping the superseded
session: `spawn_session.py stop --session-id <sid>` (on a daemon-untracked
sid it resolves the wrapper pid via the `~/.happy/logs` reverse map;
SIGTERMs under `--kill`, comm re-verified, never auto-SIGKILL). Stale
sentinels are GC'd after `max(7 days, TTL)`.

**Deliberate registration removal (`spawn_session.py unregister`; #1327).**
Deliberately removing an `issue-<N>.json` / `manual-issue-<N>.json` /
`campaign-<N>.json` registration goes through `spawn_session.py
unregister` — never a hand `rm` on `~/.eps-autonomous/` (an unguarded rm
can strip crash-recovery from the healthy owner, #952). Sid-matched by
default: `unregister --issue N` removes only files recording the CALLING
session's Happy id, so a yielding duplicate can never delete the true
owner's entry (`KEPT-SID-MISMATCH` is the guard working). Third-party
cleanup of a DEAD session's file: `unregister --issue N --session-id
<dead-sid>`, or `unregister --force --issue N` (refused with
`--session-id`). Takeover sentinels and non-registration siblings
(`dispatch-lease-*`, `campaign-watch-*`, `pm-session.json`) are never
touched. Since #1455, an OPERATOR `spawn_session.py stop` performs this
cleanup automatically once the session is confirmed dead: sid-matched
unregister across all three registration kinds + an ownership-keyed
release of the stopped dispatch's OWN lease (`acquired_at <= spawned_at`;
a successor's newer lease is kept) — the stale-registration → HELD-lease
compound cannot recur on a cleaned stop. `--no-cleanup` opts out;
watcher-sourced stops NEVER clean up (the respawn arms depend on the
registration surviving their own stops).

**Program-orchestrator recovery pass (#660).** The leakage-theory program
is sequenced by a BASH DAEMON (`scripts/run_program_orchestrator.sh` in
tmux `eps-program`), not a Happy session; a VM reboot / OOM-kill mid-
program silently stops phase ADVANCEMENT. This pass relaunches the daemon
iff ALL hold (fail toward NOT relaunching): daemon not already alive
(`pgrep -f run_program_orchestrator.sh`); the STOP sentinel
(`.claude/cache/program_orchestrator.STOP`) absent (every deliberate HALT
path touches it); the log shows no deliberate exit ("Program complete" /
"finished WITH HALTS"). Relaunch is idempotent (a fresh daemon re-checks
every phase status). Daemon-INDEPENDENT. Kill switch:
`EPM_DISABLE_PROGRAM_ORCHESTRATOR_RECOVERY=1`;
`--program-orchestrator-only`. Pinned by
`tests/test_autonomous_session_watch.py::test_program_orchestrator_*`.

**Happy injection-patch check pass (#726, `happy_patch_pass`).**
Daemon-INDEPENDENT, escalate-only: surfaces a reverted/drifted Happy
daemon injection patch PROACTIVELY (the patch,
`scripts/patch_happy_daemon.py` sentinel v4, teaches the daemon to honor
`claudeArgs` / `HAPPY_INITIAL_PROMPT`; it reverts on every `npm update
happy`, after which `--auto` spawns boot empty and never fire their skill
— the #685 class). Reads the daemon file in-process via
`_happy_patch_check.classify_patch`; on `reverted`/`drifted` writes a
`band=happy-patch` row to the shared disk-guard sidecar + a fail-soft
push, deduped per-state (`~/.eps-autonomous/happy-patch-alert.json`).
ESCALATE-ONLY: never re-applies (needs sudo); `patched` / `missing` are
clean no-ops (the spawn-path guard `_verify_happy_patch_or_die` owns the
reactive leg). `--happy-patch-only`. Pinned by
`tests/test_happy_patch_check.py` (`test_watcher_pass_*`).

**CPU/memory-pressure guard pass (#849, `cpu_guard_pass`).**
Daemon-INDEPENDENT, escalate-only detection + attribution for
CPU/memory pressure on the shared VM (origin: load 186-226 for hours;
earlyoom SIGTERM sweeps killed analysis workers — exit 143, no traceback,
misattributed). **Signals:** load5 > 1.5× nproc
(`EPM_VM_CPU_GUARD_LOAD_FACTOR`), PSI cpu `some avg10` > 50
(`EPM_VM_CPU_GUARD_PSI_CPU_PCT`), PSI memory `full avg10` > 10
(`EPM_VM_CPU_GUARD_PSI_MEM_PCT`) — rate signals needing 2 consecutive hot
ticks (`EPM_VM_CPU_GUARD_TICKS`) — PLUS a SINGLE-TICK urgent MemAvailable
floor < 20% of MemTotal (`EPM_VM_CPU_GUARD_MEMAVAIL_PCT`; one band above
earlyoom's 10% kill floor, so it fires while culprits are alive — the fire
stores a rolling pre-kill top-process snapshot, pid → issue via
`/proc/<pid>/cwd` + cmdline hints). A fire writes ONE attributed
`kind=vm-cpu-pressure` row to `.claude/cache/cpu-guard-events.jsonl` + a
deduped push; in-episode repeats suppressed unless load5 grows >25% or the
reason set changes; recovery resets the episode. **earlyoom kill
surfacing** runs EVERY tick, threshold-independent: new journal kill lines
(persistent cursor + key dedup) each produce one `kind=earlyoom-kill` row
with explicit `attribution_status: attributed | unattributed`; a
failing/missing `journalctl` degrades VISIBLY (`kill_arm: "unavailable"`),
never silently. **WARN-ONLY:** never kills, renices, or signals (pinned by
`tests/test_cpu_guard_pass.py::test_cpu_guard_never_kills`). State
`~/.eps-autonomous/vm-cpu-guard.json`. Kill switch
`EPM_DISABLE_CPU_GUARD_PASS=1`; `--cpu-guard-only`. (No ack-sentinel —
CPU/memory episodes self-terminate on recovery.)

**Post-hoc external-marker triage observer (#967,
`triage_observer_pass`).** Daemon-INDEPENDENT, **NON-GATING** audit of the
`/issue` Step 9 pre-dispatch external-marker triage duty (origin #779: 10
unread external audit markers, a long serial grid launched anyway). Sweeps
REGISTRY tasks at ACTIVE ∪ {`awaiting_promotion`, `blocked`} with
events-mtime inside a 48h lookback (`EPM_TRIAGE_OBSERVER_LOOKBACK_H`),
re-runs the #889 enumerator's window semantics at each recent dispatch
record (`task_workflow.audit_dispatch_triage`; per-task cursor — each
record judged exactly once, after its compliance window closes via the
maturity gate `EPM_TRIAGE_OBSERVER_ADJACENCY_S`, 30 min). Violation
classes: (a) `launch-missing-line` (**warn**) — a launch marker with no
triage line and no adjacent boundary triage record; (b)
`breadcrumb-missing-line` — a line-less `stage-dispatch` breadcrumb,
three-way classified on its stage token (known-benign
`TRIAGE_NONCOMPUTE_STAGES` never flags; positive compute evidence — a
`pid=` field or exact `TRIAGE_COMPUTE_STAGE_TOKENS` match — is **warn**;
unknown is **info**); (c) `none-with-candidates` — a `none` disposition
whose pre-record window re-enumerates non-empty after a 120s grace
(`EPM_TRIAGE_OBSERVER_GRACE_S`) — **info**, escalated to **warn** on an
external-signature hit. Records before `TRIAGE_DUTY_EPOCH_TS` are legacy,
never flagged. **Channels:** sidecar
`.claude/cache/triage-observer-events.jsonl` (every flag); warn flags get
one deduped push (capped `EPM_TRIAGE_OBSERVER_PUSH_CAP` 5/tick, overflow
rolled into one summary push, #1167) + one `epm:progress` review-nudge
note on the task (anti-liveness `[autonomous_session_watch:triage-observer]`
sentinel; `by="unknown"` deliberately makes the note a triage candidate at
the next dispatch), capped `EPM_TRIAGE_OBSERVER_MARKER_CAP` (5)/tick; a
warn beyond either cap stays permanently sidecar-recorded. Fire-once dedup
key `(issue, record_ts, class)` in `~/.eps-autonomous/triage-observer.json`.
**NON-GATING is a hard invariant** (never mutates status, stops a session,
or blocks a dispatch — test-pinned at subprocess-argv and in-process
levels). Invisible residuals: a LYING triage line (truthfulness is not
audited); record-less launches (direct SSH with no marker/breadcrumb).
Kill switch `EPM_DISABLE_TRIAGE_OBSERVER=1`; `--triage-observer-only`.

**Verdict-disagree observer pass (#1170, `verdict_disagree_pass`).**
Daemon-INDEPENDENT, **NON-GATING** audit of the four MARKER-MODE doubled
review sites (code-reviewer / interpretation-critic / clean-result-critic
/ follow-up-critic; the `critic` site reconciles in-context, unobservable)
for the #825 shape: the LATEST round per (issue, site) whose Claude +
Codex durable verdicts BOTH exist with parseable OPPOSITE-class verdicts,
no role-matched `epm:review-reconcile`, and — proximity-tier pairings
only — no Codex no-show evidence. The pure predicate
(`task_workflow.unreconciled_disagreement_rounds`) pairs two-tier: Tier 1
round-aligned via `ensemble_verdicts_present`, then a time-proximity
fallback (`EPM_VERDICT_DISAGREE_PAIR_PROXIMITY_S`, 6h) for
sentinel/version round drift; a 1h grace
(`EPM_VERDICT_DISAGREE_GRACE_S`) lets an in-flight reconcile land;
no-show evidence (`epm:codex-task-failed`, a codex-scoped `epm:failure`,
the #1204 quota-skip note) suppresses TIER-2 pairings only. **Channels:**
sidecar `.claude/cache/verdict-disagree-observer-events.jsonl` + one
deduped push; **NO task marker** (the flag's consumer is a human).
Fire-once dedup key `(issue, role, round_label)` in
`~/.eps-autonomous/verdict-disagree-observer.json`. Known benign-fire
class: a Step 5c-bis mechanical-contract-only strip / cap-5
all-stripped-continue resolves a PASS-vs-FAIL round without a reconciler
and flags by design (the FAIL marker's `**Blocker tags:**` line is the
one-glance disambiguator). Coverage: latest-round-only; Tier-2 evidence
suppression is site-agnostic. Sweep scope reuses the triage observer's
enumerator (`EPM_VERDICT_DISAGREE_LOOKBACK_H` 48h). Kill switch
`EPM_DISABLE_VERDICT_DISAGREE_OBSERVER=1`; `--verdict-disagree-only`.

**Root-draft observer pass (#1341, `root_draft_pass`; origin #1320).**
Daemon-INDEPENDENT, ESCALATE-ONLY flag of stale UNTRACKED `*.py` drafts in
the SHARED repo-root working tree — dirt matching the `.py` leg of
step9c's `DIRTY_CODE_PATHSPEC` flips EVERY task's Step 9c pristine-oracle
compare fleet-wide indeterminate (#1320: two untracked drafts poisoned the
ledger 9+ hours). Predicate: one read-only `git --no-optional-locks
status --porcelain -- *.py` at the main root; keep untracked (`?? `) `.py`
entries with mtime age > `EPM_ROOT_DRAFT_ESCALATE_HOURS` (3h;
tracked-modified dirt deliberately out of scope). **Channels:** sidecar
`.claude/cache/root-draft-events.jsonl` (best-effort `issue<M>_` filename
attribution) + ONE deduped push digest per tick; NO task markers. Dedup:
per-path fire-once + `EPM_ROOT_DRAFT_REALERT_HOURS` (24h) TTL in
`~/.eps-autonomous/root-draft-observer.json` (recovered paths pruned).
**ESCALATE-ONLY is a hard invariant** — never deletes, moves, chmods, or
git-mutates (pinned by
`tests/test_autonomous_session_watch.py::test_root_draft_pass_never_deletes`);
rescue is the OWNING session committing/relocating its draft. A git-status
failure warns + skips (never a silent "no drafts"). Kill switch
`EPM_DISABLE_ROOT_DRAFT_PASS=1`; `--root-draft-only`.

**Registry-drift audit pass (#1439, `registry_drift_pass`).**
Daemon-INDEPENDENT, REPORT-ONLY, once-daily-throttled
(`EPM_REGISTRY_DRIFT_INTERVAL_HOURS`, 24h; attempt stamp saved BEFORE
collecting) observer of `tasks/REGISTRY.json` ↔ filesystem drift — the
class where a `task.py` mutation hard-killed between the folder `git mv`
and the registry save leaves a stale entry that terminal tasks never
re-surface. Runs `task_workflow.audit()` + `reconcile_registry(apply=False)`
(pure reads), then DOUBLE-READS with a ~10s confirm gap
(`EPM_REGISTRY_DRIFT_CONFIRM_S`) and keeps the INTERSECT (an in-flight
mutation transient never fires). #1430's duplicate-dir husk class is out
of scope by construction (the worktree-audit cron's `reap-husks`
self-heals it). **Channels:** sidecar
`.claude/cache/registry-drift-events.jsonl` + ONE deduped push naming the
repair command (`task.py audit --repair`, `--apply` to repair) — fired on
fingerprint CHANGE (sha256[:12] over confirmed rows, volatile
`highest_id` details excluded) or a 168h TTL
(`EPM_REGISTRY_DRIFT_REALERT_HOURS`); state
`~/.eps-autonomous/registry-drift-observer.json`. **REPORT-ONLY is a hard
invariant** — never `apply=True`, no task markers (pinned by
`tests/test_autonomous_session_watch.py::test_registry_drift_pass_report_only_never_applies`).
Kill switch `EPM_DISABLE_REGISTRY_DRIFT_PASS=1`; `--registry-drift-only`.

**Completed-unmerged flag + bounded-respawn pass (#1564 + #1653,
`completed_unmerged_pass`; incident #1540).** The stranded-Step-10d-merge
audit (#1540: `completed` + `epm:done`, the merge turn killed,
`epm:merged` landed 16h later via a recovery session — invisible to every
other lane). Runs ~hourly (`EPM_COMPLETED_UNMERGED_INTERVAL_HOURS`;
worst-case detection ≈ 3h). **Predicate**
(`decide_completed_unmerged_flag`, every missing signal → silence): status
exactly `completed` (archived = deliberately abandoned, out of scope) AND
`epm:done` within a 72h lookback (`EPM_COMPLETED_UNMERGED_LOOKBACK_H`) AND
NO `epm:merged` of any form AND the newest of (`epm:done`,
`epm:merge-failed`) ≥2h old (`EPM_COMPLETED_UNMERGED_GRACE_H`; an
`epm:merge-failed` re-anchors the grace, does not suppress). **Probe**
(capped `EPM_COMPLETED_UNMERGED_PROBE_CAP`=10 sets/interval, 10s
timeouts, any error ⇒ skip-and-retry; no `git fetch` ever): (1) `gh pr
list --head issue-<N> --state open` ⇒ unmerged open PR; (2) `--state
merged` ⇒ merged, marker post lost (resolved, logged); (3) no PR: `git
ls-remote origin refs/heads/issue-<N>` absent = nothing-to-merge; (4)
branch live: patch-id count `git rev-list --cherry-pick --right-only
--count origin/main...origin/issue-<N>` (a rebase-merge rewrites SHAs, so
the plain two-dot count reads nonzero forever; patch-id reads 0 for a
landed branch), computed only when the local remote-tracking ref matches
the ls-remote sha — a stale/absent local ref fails toward FLAGGING.
**Channels**, keyed per episode = (issue, done_ts): sidecar
`.claude/cache/completed-unmerged-events.jsonl` every flagged interval;
ONE `epm:progress` marker per episode (anti-liveness
`[autonomous_session_watch:completed-unmerged-flag]`, naming the recovery:
`spawn-issue --issue <N>` then `/issue <N>` — the resume path runs the
Step 10d auto-merge idempotently); a push at episode open, re-fired every
24h (`EPM_COMPLETED_UNMERGED_REALERT_HOURS`). Resolved probe verdicts are
cached on the episode; pruned episodes are labeled honestly (`recovered`
vs `aged-out`); a later round's fresh `epm:done` opens a NEW episode.
**Never-merge is a hard invariant** (pinned two-pronged:
`test_completed_unmerged_pass_never_mutates_status_or_merges` +
`test_completed_unmerged_respawn_never_merges_or_mutates`). **Bounded
respawn arm (#1653):** on a LATER interval with the SAME episode still
stranded (flag latched on a prior interval — a human pushed at episode
open gets ≥~1h first), verdict `unmerged-open-pr`/`unmerged-branch-commits`,
NO live owning session (`_completed_unmerged_live_owner`, the #1582 owner
union; daemon-unreachable reads None and SKIPS), fleet day budget
available (`EPM_COMPLETED_UNMERGED_RESPAWNS_PER_DAY`, default 3/day
fleet-wide), and the #1027 auth-outage gate allowing — the pass dispatches
`spawn-issue --issue <N> --auto` exactly ONCE per episode (a `suppressed`
result books nothing; `spawned` OR `failed` consumes the day slot +
latches the episode, latch saved FIRST). On `spawned` it posts one respawn
marker (`[autonomous_session_watch:completed-unmerged-respawn]`) + one
push; the fresh session's `/issue` resume runs the Step 10d idempotent
backstop. Kill switch `EPM_DISABLE_COMPLETED_UNMERGED_RESPAWN=1` restores
flag-only. **Known v1 bounds** (each fails toward silence; the /daily
sweep is the backstop): (i) once ANY `epm:merged` exists, a LATER round's
stranded merge can never fire; (ii) suffixed `issue-<N>-<slug>` branches
invisible; (iii) purely-local unpushed worktree commits invisible; (iv) a
session killed between `set-status completed` and the `epm:done` post
leaves done_ts=None, silent. State
`~/.eps-autonomous/completed-unmerged-observer.json` (carries the fleet
day counter + per-episode latch). Whole-pass kill switch
`EPM_DISABLE_COMPLETED_UNMERGED_PASS=1`; `--completed-unmerged-only`.

**Partial-bundle reconciliation pass (#1704, `partial_bundle_pass`;
incident #1345).** The reader-back of the GCP EXIT-trap crash-persist path
— nothing else ever reads `issue<N>_partial/<attempt_id>/` bundles back,
so a bundle carrying a COMPLETED result whose workload upload path never
fired was indistinguishable from a genuinely-partial persist. Lists the
`issue<N>_partial/` prefixes for recently-touched non-`proposed` REGISTRY
tasks (`EPM_PARTIAL_BUNDLE_LOOKBACK_H`, 168h), groups by attempt_id,
classifies via `_classify_bundle_completeness` (`complete` /
`workload_ts_backstop` / `no_result_payload` — silent skip /
`persist_killed` — silent skip), extracts bundle-relative
`eval_results_issue_<N>/` paths, and compares against ONE read-only `git
ls-tree -r --name-only HEAD -- eval_results/issue_<N>/` at `PROJECT_ROOT`
(semantic contract: "landed on `main`", never a worktree HEAD). Any
bundle path with NO committed counterpart flags. **Channels:** sidecar
`.claude/cache/partial-bundle-events.jsonl` (with `completeness_signal`
verbatim) + ONE deduped push per (issue, attempt_id, band); NO task
markers. Cadence: hourly self-gate (`EPM_PARTIAL_BUNDLE_INTERVAL_HOURS`),
per-pass listing cap (`EPM_PARTIAL_BUNDLE_LISTING_CAP`, 50) + persisted
cursor so tail-of-list issues never starve; per-episode dedup with a 168h
TTL (`EPM_PARTIAL_BUNDLE_REALERT_HOURS`). Fail-soft PER ISSUE (one
hub-error / git-error sidecar row, continue). **ESCALATE-ONLY is a hard
invariant** — never auto-commits, never deletes a bundle, never posts task
markers (pinned by
`tests/test_autonomous_session_watch.py::test_partial_bundle_pass_never_mutates_state`
+ `test_partial_bundle_pass_never_posts_task_markers`). Kill switch
`EPM_DISABLE_PARTIAL_BUNDLE_AUDIT=1`; `--partial-bundle-only`.

**Urgent-park router pass (#1681, `urgent_wf_park_pass`).** The "main is
red" fast path for PARKED workflow-fix candidates (otherwise up to ~24h to
the nightly /daily Step C sweep while every session's Step 9c gate
re-classifies the red; #1643). **Predicate:** the parking session LABELS
the park mechanically routable (formal candidate block + `urgency:
main-red` + `failing_test: <one pytest node id>` + `wf_fix: true|false` —
grammar: `.claude/rules/workflow-fix-on-bug.md` § Recursion guard "Urgent
fast path"; labeling is not routing — only this non-guarded watcher acts).
**Tiers:** a cheap mtime+substring candidate gate over ~48h-fresh events
streams → authoritative enumeration by importing
`sweep_parked_wf_candidates.sweep()` read-only → two-tier claim
verification (a fresh step9c baseline-ledger hit whose `refreshed_at`
postdates the park, else ONE bounded `uv run pytest <node>` per tick —
timeout `EPM_URGENT_WF_PARK_PYTEST_TIMEOUT_S` 180s; rc==1 confirmed /
rc==0 refuted / any other rc indeterminate) → dedup belts
(`task_workflow.is_open_workflow_fix_task` + a failing-node containment
scan over open infra bodies) → file + dispatch via
`scripts/file_infra_task.py` → the standard `epm:workflow-fix-task-filed`
routed-record posted on the park's OWN stream BEFORE latching
(sweep-reported fingerprint verbatim + `origin_candidate_ts`; sentinel
`[autonomous_session_watch:urgent-wf-park-router]`). **Fallback:**
missing/malformed/refuted/unverifiable urgency → the park stays enumerated
for the nightly sweep; the router never synthesizes fields. **Bounds:**
fleet per-UTC-day route cap `EPM_URGENT_WF_PARK_ROUTES_PER_DAY` (default
2; quiet-at-cap), per-candidate verdict latch (state
`~/.eps-autonomous/urgent-wf-park-router.json`; sidecar
`.claude/cache/urgent-wf-park-events.jsonl`), ≤1 pytest subprocess/tick,
indeterminate verification latched `unverifiable` after 2 attempts. Kill
switch `EPM_DISABLE_URGENT_WF_PARK_PASS=1`; `--urgent-wf-park-only`.

**Auth-outage guard pass (#1027, `auth_outage_pass`).** Fleet-level
respawn suppression for an Anthropic auth outage — or ANY fleet-wide
instant-death cause (poisoned CLI credential, broken `claude` binary, a
reverted Happy patch that escaped `happy_patch_pass`). Origin: a poisoned
credential killed every fresh session on arrival and the watcher churned
die-on-arrival respawns for hours (per-task caps bound per-ISSUE churn;
nothing read the fleet correlation). Runs after the daemon probe, BEFORE
every spawn arm.

- **Detection (watcher-owned state, no log-grepping):** every
  watcher-issued spawn records `{issue, ts, arm, prev_spawned_at}`. An
  event is an *instant-freeze respawn* when `0 <= ts − prev_spawned_at <=
  EPM_AUTH_OUTAGE_FRESH_DEATH_MIN` (default 60 min; a healthy multi-hour
  session never qualifies). ≥ `EPM_AUTH_OUTAGE_MIN_EVENTS` (3) such events
  across ≥ `EPM_AUTH_OUTAGE_MIN_ISSUES` (2) DISTINCT issues inside
  `EPM_AUTH_OUTAGE_WINDOW_MIN` (180 min) trigger an episode — cross-issue
  correlation is the false-positive guard.
- **While active,** every spawn arm — crash-recovery, stalled (gated at
  the fence CALLER so stop+respawn is skipped as a unit), orphan,
  infra-drain, capacity-retry, campaign — is suppressed via the #843
  `"suppressed"` channel (callers book nothing). Non-spawning passes are
  NOT gated. ONE push at trigger (evidence-enriched: a best-effort
  auth-signature grep over the newest `~/.happy/logs` — push TEXT only,
  never the trigger) and at most one at resolution.
- **Canary-probed resume:** every `EPM_AUTH_OUTAGE_CANARY_INTERVAL_MIN`
  (30 min) the pass arms a single-tick token; the first eligible issue-arm
  spawn becomes the canary — a REAL session respawn probing the exact
  CLI-credential path. A canary surviving ≥
  `EPM_AUTH_OUTAGE_CANARY_SURVIVAL_MIN` (20 min) resolves the episode; a
  dead one re-arms one interval later. The campaign arm never consumes the
  token.
- **Fail-open, twice over:** any internal guard error behaves as "no
  outage" (a false suppression is a fleet-wide crash-recovery blackout,
  strictly worse than churn); an episode older than
  `EPM_AUTH_OUTAGE_MAX_EPISODE_H` (6h) expires with a push — enforced in
  the pass AND independently in the gate. On resolve/expire the
  `last_episode_end_ts` watermark blocks stale re-trigger (qualifying
  events need both `ts` and `prev_spawned_at` past it); a genuinely
  persistent outage re-accumulates and legitimately re-triggers.
- **State** `~/.eps-autonomous/auth-outage.json` (never GC'd; events
  pruned to 2× the window); **sidecar**
  `.claude/cache/auth-outage-events.jsonl`. Kill switch
  `EPM_DISABLE_AUTH_OUTAGE_GUARD=1`; `rm ~/.eps-autonomous/auth-outage.json`
  clears a live episode; `--auth-outage-only`.
- **Dispatch-chokepoint leg (#1218):** `spawn_session.py spawn-issue
  --auto` (the choke point every non-watcher automated dispatcher funnels
  through) holds spawns during an ACTIVE in-TTL episode with a rc-0
  `AUTH-OUTAGE HELD` line (recognized by `spawn_output_suppressed`); the
  watcher's canary passes via the pre-spawn `canary_pending` claim; manual
  spawns + `spawn-campaign` warn-and-proceed; `spawn-pm` untouched.
  Read-only mirror of the watcher gate (same kill switch, same fail-open
  TTL); pinned by `tests/test_spawn_session_auth_outage_gate.py`.
- **Accepted residuals (deliberate):** (a) hang-style outages (no respawn
  events → no trigger; also no churn); (b) new-spawn-only outages
  (`prev_spawned_at=None` never counts; bounded by the per-day caps); (c)
  the program-orchestrator recovery pass can relaunch the #660 daemon
  during an episode — narrowed by #1218 (its child spawns are held at the
  chokepoint); (d) two independent issue-specific crash loops can
  false-trigger — bounded by canary self-heal + the 6h TTL; (e) a
  wedged-but-registered canary can false-resolve — a still-broken fleet
  re-accumulates and re-triggers; (f) detection fires at the second
  respawn generation (~60-75 min in); (g) the
  `EPM_AUTH_OUTAGE_FRESH_DEATH_MIN` band is clamped ≥45 min.

## Dedicated data disk for `.claude/worktrees/` (#681)

The heavy active-task footprint (every `issue-<N>` worktree + its
per-issue `data/issue_<N>/{hf_dl,g*_dl,store}` caches) lives on a
dedicated **512 GB `pd-balanced` GCP persistent disk mounted at
`/mnt/eps-data`** (env `EPS_VM_DATA_DISK_PATH`), bind-mounted back onto
`.claude/worktrees` so every consumer resolves the SAME path. The disk is
provisioned in the `introsp-experiments` project (where the VM lives), NOT
the GPU project.

**Per-task ext4 project quotas.** Each `issue-<N>` subtree carries an ext4
project id == the issue number with a hard byte cap
(`EPS_ISSUE_DISK_CAP_GB`, default 128 GB), assigned at worktree creation
by `new_worktree.sh` (`chattr -p <N> +P` + `setquota -P <N>`, opt-in via
`EPS_WORKTREE_ASSIGN_QUOTA=1`). A write past the cap fails loud with
`EDQUOT` while every OTHER issue keeps writing. Recovery is always
resize / raise-cap, NEVER delete active data.

**Dual-disk watch — escalate-only on the data disk.** The disk guards
watch BOTH filesystems: `/` with the existing byte-floor logic, and
`/mnt/eps-data` with PERCENT / statvfs-derived thresholds
(size-invariant — a future resize cannot push the fire point past the
wedge). The data-disk pass is ESCALATE-ONLY: the `/`-rooted reclaim arms
never run keyed off the data disk;
`vm_disk_guard.run_guard(disk_path="/mnt/eps-data", reclaim_tiers=False)`
runs only tier (b) (terminal-cache reap + active-cache escalation), and
the watcher's `data_disk_pass` drives `decide_vm_disk_pct` +
`decide_subfloor_pct` (`EPM_VM_DATA_DISK_SUBFLOOR_PCT` default 85%) off
`statvfs(/mnt/eps-data)`, attributing worktree-internal caches via
`repquota -P` (du fallback). Both passes are clean no-ops when the mount
is absent. Since #1392 the BOOT-disk sub-floor sentinel's sibling arm
(`subfloor_reclaim_pass`) additionally launches a detached, single-flight,
rate-limited `vm_disk_guard.py --apply --ignore-threshold --no-push
--no-data-disk` reclaim run while `/` free stays below
`EPM_VM_DISK_SUBFLOOR_GIB` (interval
`EPM_VM_DISK_SUBFLOOR_RECLAIM_INTERVAL_S`, 1800s; kill switch
`EPM_DISABLE_SUBFLOOR_RECLAIM=1`) — VM-root only; the data disk stays
escalate-only.

**Non-canonical caches + the /workspace hub-cache arm (#911).** The
guard's tier (b) ALSO sweeps NON-CANONICAL issue-keyed caches — top-level
`/tmp/` dirs named `i<N>*` / `issue<N>*` / `issue-<N>*` / `issue_<N>*` /
`*_<N>`, and `data/` dirs named `issue…<N>…{_dl,_hfstage,_cache}` — under
the same terminal-reap / active-escalate contract PLUS a 48h recency keep
(`EPS_NONCANONICAL_CACHE_MIN_AGE_HOURS`), a nested
`store/`+`eval_results/` block, and a positive
re-downloadability-evidence gate (predicate failures escalate, never
delete). A boot-pass-only arm age-gates the VM's pod-style
`/workspace/.cache/huggingface` hub cache (repos unused ≥14 days,
`EPS_VM_WORKSPACE_HF_CACHE_MAX_AGE_DAYS`), pod-guarded so it can never run
where `/workspace` is a real volume. The `/tmp/` + `/workspace` opt-in
lives ONLY in the two CLI `main()` bodies (`tmp_root=production_tmp_root()`;
library calls are hermetic); report-only runs surface evidence via the
`--json` structured fields. Kill switch:
`EPM_SKIP_NONCANONICAL_CACHE_SWEEP=1`. Tier (e) (#1376 + #1377) covers the
HOME HF hub cache `~/.cache/huggingface/hub` (`hub/` only) on the same
boot-pass-only opt-in: always attributes per-repo size / revision count /
`last_accessed` age, escalates any single repo > 40 GB
(`EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB`, per-revision breakdown, deduped
with ack sentinels), and on `--apply` reaps via `delete_revisions`
(blob-refcount safe): unref'd non-newest revisions ≥7d old
(`EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS`; the newest + every ref'd
revision always kept) plus wholly-stale repos by repo-level
`last_accessed`. Interplay note: the watcher's `_vm_reclaim_hf_hub_cache`
(`EPM_VM_DISK_HF_TTL_DAYS`=14, CRITICAL-gated) and guard tier (e) (7d,
threshold-gated) BOTH cover the home hub cache BY DESIGN — two independent
`delete_revisions` reapers (a lost race degrades to a skipped tier); do
not "unify" the two knobs without reading #1376 + #1377.

**Janitor exemption.** The stale-GCP-VM janitor sweeps the GPU project for
ephemeral GCE INSTANCES; the `/mnt/eps-data` disk is a PERSISTENT disk in
a DIFFERENT project (`introsp-experiments`) — out of the janitor's scope
by construction, intentionally never reaped.

## tmux socket-dir contract (#1466)

**Incident (split-brain, #1466).** The fleet assumes ONE tmux server, all
consumers addressing `/tmp/tmux-1001/default`. An ad-hoc disk-pressure
sweep (`find /tmp -maxdepth 1 -mtime +2 … | xargs rm -rf`, no `tmux-*`
exclusion — a one-off improvised command, not from any repo script)
deleted the socket dir with the live server socket inside; the next
tmux-spawning consumer silently created a SECOND server at the same
default path and 39 sessions on the old server became invisible to
`tmux ls` until manual socket-rebind surgery. Refuted alternative causes:
no systemd-tmpfiles `/tmp/` Age rule on this host, no tmpreaper/tmpwatch,
server pid alive throughout.

**The shim (`scripts/eps_tmux_env.sh`) — single source of truth.**
Contract: (1) durable default `TMUX_TMPDIR=$HOME/.tmux-sockets`
(persistent disk, 0700 — no `/tmp/` cleaner reaches it); (2) LEGACY PIN —
while ANY socket file exists in `/tmp/tmux-$(id -u)` (checked
`find -maxdepth 1 -type s`; an existing-but-unreadable dir also pins;
watcher `_live_tmux_socket_present()` parity), resolve `/tmp/` so the
whole fleet keeps addressing ONE server; the flip to the durable dir fires
automatically and coherently for every shim consumer at the first
zero-socket point (reboot / drain / re-deletion); (3) a pre-set
`TMUX_TMPDIR` is always respected; (4) FAIL-COHERENT PIN-BACK — if a
non-shim straggler creates a `/tmp/` server post-flip, all shim consumers
pin BACK to `/tmp/` (the fleet follows one server rather than splitting).
Known limitation: a stale socket from a SIGKILL'd server pins `/tmp/`
until reboot — still single-server-coherent. **Sourced by:**
`scripts/cron_session_summarize.sh` +
`scripts/cron_autonomous_session_watch.sh` (placement pinned by
`tests/test_eps_tmux_env.py`), and two VM-LOCAL out-of-repo files —
`~/.profile` (login shells) and `~/my-goat/scripts/run_mygoat_session.sh`
(the systemd user service reads no profile). Exact VM-local diffs: task
#1466 events.

**Defense-in-depth: `/etc/tmpfiles.d/tmux.conf`** (insurance against a
future systemd default enabling `/tmp/` aging):

```
# /etc/tmpfiles.d/tmux.conf  (#1466)
x /tmp/tmux-*
```

Verify: `systemd-tmpfiles --cat-config | grep -F 'x /tmp/tmux-'`. It does
NOT protect against a non-tmpfiles deleter — that is what the durable dir
is for. Prevention leg (#1474): the PreToolUse guard
`.claude/hooks/guard_tmp_tmux_sweep.sh` blocks unexcluded /tmp deletion
sweeps at the Bash tool layer (override: `EPM_ALLOW_TMP_SWEEP=1`;
sanctioned uses incl. single-file removal of a verified-dead socket —
recipe in the hook header, #1559).

**Recovery runbook (socket vanished, server alive).**
1. Find the server: `ss -xlp | grep tmux` (shows bound path + pid even
   when the socket FILE is deleted) or `pgrep -f 'tmux: serve[r]'`
   (`pgrep -x tmux` misses it — the server's comm is `tmux: server`).
2. If `/tmp/tmux-<uid>` is gone: `mkdir -m 700 /tmp/tmux-$(id -u)`.
3. `kill -USR1 <server-pid>` — the server recreates its socket at its
   ORIGINAL bind path (fixed at server start; parent dir must exist).
4. Address it explicitly: `tmux -S /tmp/tmux-<uid>/<name> ls`. A bound
   socket FILE may be `mv`'d aside to coexist with a second server;
   clients reach the old server through the renamed path.
5. **Deletion-race winner:** during a deletion event the `/tmp/` pin
   WINS — shim consumers pin back to `/tmp/` (the recovered legacy
   socket), so a durable-dir server started inside the race window is the
   one to drain after re-cohering. Recovery is deterministic.
6. **Happy daemon start mechanism:** the daemon is a manually-started
   orphaned node process (`node …/happy/dist/index.mjs daemon
   start-sync`, parent = init, NO systemd unit, env carries no
   `TMUX_TMPDIR`). Post-reboot durability hinges on restarting it FROM A
   LOGIN SHELL (which sources `~/.profile` → the shim).
7. **Non-interactive SSH:** `ssh vm '<tmux cmd>'` reads neither profile
   nor shim — use `ssh vm 'bash -lc "<tmux cmd>"'` for manual remote tmux
   ops.
