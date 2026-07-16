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

CLAUDE.md § Pods carries the always-on one-paragraph summary (which crons
exist + their user-visible effects); this file is the full predicate spec.

## Stale-pod audit (09:37 daily + on `pod.py provision`)

Auto-terminate pods EXITED >24h — EXEMPT when the owning task carries the
`keep-running` tag, reported as `kept-exited` instead.

## Stale-GCP-VM janitor (09:37 daily, `cron_gcp_audit.sh` → `gcp_audit.py`)

The GCP analogue of the stale-pod audit — the credit-leak backstop for the
WHOLE dedicated `eps-persona-gpu-jun2026` project (#688), not just
`eps-issue-*` (a non-`eps-issue-*` leftover — the #680 `eps-cap-probe2-1786331`
flex-start probe ran ~20h, ~$14 — was invisible to the old `name~^eps-issue-`
filter). Catches instances that escaped the canonical ephemeral teardown
(`--max-run-duration` DELETE + EXIT-trap). Wraps the
`backends.gcp.audit_stale_gcp_vms` reaper; the reap/classify predicate lives in
the library, the cron + CLI are the wiring. Scheduled next to the RunPod sweep
(`37 9 * * *`) so both backends reclaim on the same daily pass.

**Classification + routing (the HYBRID posture).** The janitor lists the whole
project (`JANITOR_LIST_NAME_FILTER = None`) and classifies each stale instance
by name into one of four classes, routed differently:

- **`managed`** (`eps-issue-*`, the router-owned names) → AUTO-DELETE on the
  bounded fences below, exactly as before.
- **`allowlisted-ephemeral`** (a known-throwaway name prefix, default
  `eps-cap-probe*` — the #680 capacity-probe leak class; grow the
  `_EPHEMERAL_REAP_PREFIXES` tuple as new patterns emerge) → AUTO-DELETE on the
  same fences.
- **`unmanaged`** (anything else in the project) → WARN-and-ESCALATE, never
  auto-deleted: a Telegram phone push (via the my-goat `telegram_push.sh`
  channel, `NOTIF_CAT=research`, fail-soft — a missing/failing push never
  blocks the sweep or hides the record) PLUS a durable sidecar JSON row at
  `.claude/cache/gcp-janitor-events.jsonl` (a dedicated stream, separate from
  the disk-pressure-scoped `disk-guard-events.jsonl`). Records carry
  `action="would-escalate"` (report-only) / `"escalated"` (under `--delete`).
  An instance the janitor cannot positively classify as throwaway is treated
  like active data — surfaced, not reaped (the project's canonical
  warn-don't-delete posture, #679).
- **`keep`** (an opt-out prefix, `_JANITOR_KEEP_PREFIXES`, empty today) → never
  reaped OR escalated; emits a `skipped` record so the operator sees it was
  inspected and deliberately left alone.

The router seams (`reconnect_or_none` / `_stale_named_instance_or_none`) keep
their EXACT `name=eps-issue-<N>` list filters — only the JANITOR's inventory
query broadens, so broadening cannot leak into the router's reconnect/reclaim
namespace.

**Reap predicate** (two bounded fences, both in the reaper, applied to the
reap-class instances — see the `audit_stale_gcp_vms` docstring):

- **Per-instance-fence-aware age backstop** (#741): any project instance is
  age-stale, regardless of phase, when EITHER it has a readable
  `scheduling.maxRunDuration` (set natively by GCP when the create passed
  `--max-run-duration`) and has exceeded that fence + a 1h grace
  (`_JANITOR_FENCE_GRACE_SECONDS`), OR it has NO readable fence and has lived
  past the fixed `--max-age-hours` fallback (default 192h = 8d, covering the 7d
  `default_max_run_duration` + margin) — the last-resort fence for a VM whose
  `--max-run-duration` DELETE never fired (`reason="age"`; the reap-vs-escalate
  split is then decided by classification). Tracking each instance's OWN fence
  (a 24h job reaps at ~25h, a 7d job at ~7d+1h) instead of a single fixed
  wall-clock avoids re-creating #697 — a 7d job killed by the janitor's old
  blanket 24h cap.
- **10-min terminal-phase reap** (`--terminal-phase-max-age-min`, default 10):
  a RUNNING instance that published a terminal `eps/phase` (`done` / `failed`,
  probed via the `eps/phase` guest attribute) but never auto-deleted is a
  wedged zombie idle-billing an A100 (#634 / the metadata-runner-SIGPIPE
  class). The short floor keeps the sweep from racing a legitimate
  post-completion finalize (~30-60s); a guest-attribute probe FAILURE
  ("couldn't ask" ≠ "done") falls through to the age backstop, never escalates
  to delete, never crashes the sweep (`reason="terminal-phase"`).

**Report-only by default** — the CLI's `--delete` (passed by the cron) is the
only real reaper AND the only mode that fires escalations (the escalation
closure is wired `escalate=...` ONLY under `--delete`; report-only passes
`escalate=None` → inert `would-escalate` records, no push, no sidecar row);
`EPS_GCP_JANITOR_DRY_RUN=1` forces report-only even with `--delete`, the central
smoke kill-switch. Escalation is the WORKING path (not a fault), so an escalated
unmanaged VM keeps exit rc=0 — only a `delete-failed` raises rc to 2.

**Disarmed-janitor alarm (the list-preflight).** The frozen reaper swallows a
non-zero `gcloud compute instances list` rc and returns `[]` —
indistinguishable from a legitimately empty inventory — so an expired/
misconfigured-auth janitor would fire daily, reap nothing, and read green. The
CLI compensates with its own list-preflight (reusing the reaper's own
`render_list_argv` builder so the probe is byte-identical): on a non-zero rc it
does NOT call the reaper, surfaces `list_rc`/`list_stderr` in the JSON, and
exits **3** (`list-failed`). The cron propagates rc=3 to its own exit so the
disarmed-janitor email is delivered; rc=2 (`delete-failed`, one transient
delete error — routine) and rc=0 (clean) both stay `exit 0` (no nuisance
email, mirroring `cron_pod_audit.sh`).

**Exit codes:** `0` clean / `2` at-least-one-delete-failed / `3` list-failed.

**Env-var overrides:** `EPS_GCP_JANITOR_DRY_RUN=1` (force report-only),
`EPS_GCP_JANITOR_LOG_DIR` (override the dated-log dir; default
`logs/gcp_audit/`), `EPM_TELEGRAM_PUSH_SCRIPT` (override the escalation
phone-push script; default the my-goat `telegram_push.sh`),
`EPM_GCP_JANITOR_SIDECAR` (override the escalation sidecar JSONL path; default
`.claude/cache/gcp-janitor-events.jsonl`). Output: per-pass detail in
`logs/gcp_audit/YYYY-MM-DD.log`,
a once-per-day pointer line in the outer crontab redirect file — the same
dated-log + first-run-of-day-pointer liveness mechanism as `cron_pod_audit.sh`
(task #580 item-3).

## Stale-worktree sweep (09:47 daily, `worktree_audit.py --apply`)

Reaps idle auto-generated worktrees under `.claude/worktrees/` — removed only
when not held by a live process, not an `issue-<N>` with a non-terminal
status, older than a 6h grace window (tightened to 1h when the filesystem
holding the worktrees is ≥90% full — disk-pressure mode, threshold via
`EPM_WORKTREE_DISK_PRESSURE_PCT`; the audit always reports disk usage +
per-worktree sizes), and with no uncommitted tracked changes. Human-named
worktrees are never touched (`issue-<N>-<suffix>` follow-up worktrees ARE in
sweep scope as of 2026-06-12, mapped to issue N for the status lookup).

For done-and-merged (`completed`/`archived`/`awaiting_promotion` — the latter
added 2026-06-12: the worktree auto-merged to main at the Step 9b transition
and the reconcile pass auto-stops parked sessions; any real non-orphan holder
still blocks) issue worktrees, `--apply` additionally remediates two
false-keep classes (2026-06-10 disk-full incident): kills orphaned codex
`app-server` holder pids (exact-pid, cmdline re-verified immediately before
each signal; never when any real holder is present) and rescue-copies
allowlisted runtime-noise dirt (agent memories, `pods.conf`,
`pods_ephemeral.json`) to `.claude/cache/worktree-rescue-<date>/` BEFORE
removal; dry-run only classifies, never kills or rescues.

**Venv-reap arm (#912).** On every `--apply` sweep, a KEPT worktree (dirty /
active issue / human-named — every plain-keep class) additionally gets its
`.venv` reaped when the worktree has been idle ≥7 days (2 days under the same
disk-pressure predicate; env `EPM_WORKTREE_VENV_IDLE_DAYS`) AND no process
holds it — the cwd/argv harvest plus a dedicated `/proc/<pid>/exe` probe that
catches interpreters exec'd from the venv itself; idleness is
max(root mtime, `.venv` mtime, git-admin `HEAD`/`index` mtime), failing
toward keep when unreadable. The delete is rename-aside
(`.venv.reap-tmp-<pid>`) → post-rename holder re-check → rmtree, never wider
than `<wt>/.venv` + its own leftovers; underscore-prefixed managed worktrees
(`_task-main-pin`) and symlinked roots/venvs are never touched; a `.venv` is
a pure build artifact `uv run` regenerates from the shared uv cache in
minutes. Kill switch: `EPM_WORKTREE_VENV_REAP=0`. Note the interaction with
the daily `uv cache prune` cron: reaping a venv only drops worktree-side
hardlinks — the shared blocks free when `uv cache prune` later drops
unreferenced cache-side entries, so the two crons' combined reclaim is the
number to watch (per-venv byte figures are du-apparent).

`codex_task.py` complements this by pinning every codex-companion dispatch to
the main checkout root (`DISPATCH_ROOT`), so new codex workers never root
themselves in a worktree.

## Autonomous-session watcher (every 10 min, `3-59/10 * * * *`, `autonomous_session_watch.py`)

Passes: crash-recovery respawn, pod-safety reconciliation, stalled-session
detector, orphan-file sweep, the infra-drain pass, the capacity-retry pass,
the stale-blocked flag pass, the gate-push pass, the program-orchestrator
recovery pass, the boot-death pass, the
stale-registration pass, the CPU/memory-pressure guard pass, the
triage-observer pass, and three session reapers — the session-vs-status
reconcile pass, the zombie-wrapper pass, and the idle-unmapped pass.

**Stall-detection hardening (#845; the five 2026-07-01 incident classes).**
The stalled detector + the two respawn arms carry six hardening mechanisms
(five from #845, the sixth from #1137):

- *(a-i) Marker-heartbeat window.* Signal 2 (the newest non-watcher marker)
  has its OWN 2h freshness window (`EPM_STALLED_MARKER_HEARTBEAT_MIN`,
  default 120 min = 2× the 60-min self-report window) — a session that
  posted ANY non-watcher marker within 2h is never declared stalled
  (incidents #761/#763: legitimate 60–120-min marker gaps corroborated
  false stalls). Deliberate trade: third-party markers (PM notes, pod-side
  relays) can shield a wedged session for up to 2h; the (e) wedge fast lane
  is the mitigation.
- *(a-ii) Stop-verify respawn fence* (`decide_respawn_fence`): the stalled
  respawn arm NEVER spawns in the same tick it stops a session — stop →
  verify the sid is absent from the daemon's live set on the NEXT tick →
  spawn (the daemon's stop ACK is not a kill — the same contract the
  zombie/idle reapers enforce; incident #763: a same-tick stop+spawn left
  two drivers overlapped ~4h). One stop retry, then a one-time loud
  stop-failed marker + push (`[...:session-stop-failed]`), never a spawn
  next to a live sid. A pending fence whose sid no longer matches the
  registry entry CLEARS itself (a concurrent respawn owns the issue; the
  fresh sid is never stopped), and the stalled arm skips entirely within
  `RESPAWN_SPAWN_GRACE_S` of a fresh registration (the crash arm runs
  first each tick and can legitimately respawn inside the stop→verify
  gap). A #843 lease-"suppressed" spawn books nothing AND ends the fence
  episode. Even a genuinely-dead wrapper waits one tick between stop and
  spawn — that +10 min is deliberate.
- *(b) Bounded worktree-activity hold* (`decide_worktree_hold`): BOTH
  respawn arms (stalled + crash-recovery) defer while any file under
  `.claude/worktrees/issue-<N>*` (excluding `data/`) has an mtime within
  15 min (`EPM_STALLED_WT_ACTIVITY_MIN`) — direct evidence an implementer
  is mid-edit (incident #812: killed 57s after an edit) — bounded at 6
  consecutive held ticks (~1h; a bound, not a latch) with `missed` pinned
  at the threshold so the arm re-fires the moment activity quiets. The
  probe early-exits on the first fresh hit under a 2s wall-clock deadline.
- *(c) Daemon retry + blocked-recovery escalation.* The per-tick daemon
  probe retries (3 attempts, 5s/10s backoff; `EPM_DAEMON_PROBE_ATTEMPTS`)
  before declaring an outage, and a respawn-worthy stall deferred by
  daemon-unreachable for ≥2 consecutive ticks (~20 min) fires ONE Telegram
  push per episode (`decide_daemon_blocked_escalation`; incident #811: a
  silently-deferred respawn idled a GPU for hours). The existing
  alerted→eligible escalation still respawns on the first daemon-up tick.
- *(e) Prompt-wedge fast lane* (`decide_prompt_wedge`): a
  transcript-tail probe (happy-log-only resolution, last 256 KB — widened
  from 64 KB at #1104: the smaller window held EXACTLY 3 api-error rows on
  the #1074 incident transcript, zero margin) escalates
  straight to the respawn arm on ≥3 (`EPM_TICK_WEDGE_MIN_DEQUEUED`)
  consecutive trailing wedge-evidence rows with no assistant turn —
  verified `{"type": "queue-operation", "operation": "dequeue"}` records
  (co-primary) and/or promptless prompt-type user rows (secondary)
  (incident #779: 5 prompts enqueued+dequeued with no turn for ~90 min).
  **Api-error widening (#1104):** assistant rows with top-level
  `isApiErrorMessage: true` (usage-policy refusals, 429/529 error turns)
  are FAILED turns, not resets — they accumulate on their OWN trailing
  counter (`EPM_TICK_WEDGE_MIN_API_ERRORS`, default 3; `0` DISABLES this
  trigger, the one deliberate divergence from the dequeued knob's
  fallback semantics), so ≥3 consecutive refused wakes with no successful
  turn trip the wedge (incident #1074: 38 refused wake turns / ~2h
  unrecovered — pre-#1104 each refusal row classified "assistant" and
  RESET the run, hiding the session from this lane). A REAL assistant
  turn resets BOTH counters; an api-error turn resets the dequeue/prompt
  run (the prompt DID get a response — the single-refusal over-trigger
  guard: one refused wake's 2-4 dequeue+prompt rows must not trip
  `run >= 3`).
  **#1127 turn-level failed-wake counting:** the row-level counters are
  structurally blind to the PARTIALLY-successful wake (a refused wake
  that posts 1-3 assistant heartbeat rows before dying resets both
  counters every cycle), so the tail rows additionally segment into
  WAKE-TURNS (`_segment_wake_turns`): a turn starts at prompt evidence
  (dequeue/prompt rows — one delivery burst), collects the response rows
  that follow, and a COMPLETED turn whose LAST response row is an
  api-error is a FAILED wake even when mid-turn assistant rows
  (heartbeats) preceded it (a mid-turn api-error followed by a
  successful row in the same turn is `ok` — the retried-429 shape;
  swallowed deliveries produce NO turn and stay the dequeue-run's
  property). (a) ≥3 consecutive trailing failed turns
  (`EPM_TICK_WEDGE_MIN_FAILED_TURNS`, default 3, `0` disables) trip the
  wedge — incidents #1098 (5bdae5b8) and #1090 (5e464f3d) ran 40 min-3.4 h
  past the #1104 merge on exactly this shape; (b) a conservative
  alternating-storm RATE trigger — ≥6 failed turns
  (`EPM_TICK_WEDGE_MIN_FAILED_TOTAL`, default 6, `0` disables) within
  120 min of the newest row timestamp (`EPM_TICK_WEDGE_RATE_WINDOW_MIN`;
  anchored to the newest ROW ts, not wall-clock) with the newest
  completed turn failed (incident c16b10ca: ~every other wake lost
  00:26-06:07Z, ~5.7 h — no consecutive predicate can fire on
  alternation; the measured 256 KB tail held 4–5 windowed failed TURNS —
  below the 6 threshold, so this lane targets DENSER storms;
  c16b10ca-density incidents get partial failed-turn-run coverage only);
  (c) the probe-arming change — the TURN-level lanes are
  probed EVERY tick (a dying-but-heartbeating wake that escalates into
  the full `/issue` skill re-writes the self-report at Step 0 before
  dying, defeating the old ≥1 h-stale precondition), while the
  dequeue-run and row-level api-error-run triggers stay STALENESS-GATED
  (their failure modes freeze the self-report by construction, and the
  fresh path must not fire on one wake's same-turn retry rows); with
  both turn knobs at `0` the fresh path probes nothing (the exact
  pre-#1127 lazy gate — the fresh-path rollback; a FULL #1209 rollback
  additionally sets `EPM_TICK_WEDGE_DEAD_SILENCE_MIN=0`, which disables
  the dead-wake trigger on the stale path too).
  **#1209 dead-wake trigger (`failed-turn-silence`):** a session dead
  after a SINGLE refused turn never arms its tick cron and freezes below
  every counting threshold above (incident #1092 / transcript 8e9c371d:
  1 failed turn at 02:54Z, ~100 min to the slow-lane respawn), so a tail
  whose completed turns are ALL failed and whose newest parseable row
  timestamp is ≥ `EPM_TICK_WEDGE_DEAD_SILENCE_MIN` (default 20 min; `0`
  disables; malformed/negative → default) older than the pass clock
  escalates to the fence STOP. The stop is the trigger's ACT — recorded
  by a one-time `session-dead-silence-stop` marker at stop-initiation —
  and the CRASH-RECOVERY arm completes the respawn once the stopped
  wrapper is verifiably dead (~20-30 min): the fence's own spawn branch
  is unreachable for this trigger's fresh-self-report shape (post-stop
  the sid leaves the daemon /list, the wedge pid-gate goes inert, and
  decide() keeps on the fresh boot self-report). Bounded by the episode
  belt (`respawn_count < STALLED_MAX_RESPAWNS`) AND a per-issue
  per-UTC-day cap `EPM_TICK_WEDGE_DEAD_RESPAWNS_PER_DAY` (default 3;
  malformed or <1 → default — never a kill switch), bumped ONCE per
  fence episode at stop-initiation and persisted
  advancement-clear-EXEMPT in `stalled-<N>.json` (each die-on-turn-1
  generation writes one boot self-report, so episode-scoped state
  cannot bound the cross-generation die-on-boot loop); a cap-disarmed
  trigger goes quiet (no marker, no push) and the slow stalled lane
  stays the backstop. On the fresh path it rides the SAME two-turn-knob
  gate as the #1127 lanes (both turn knobs at `0` keeps the zero-probe
  hot path; the dead-wake trigger then still fires via the STALE path
  once the boot self-report ages out); a prior ok turn anywhere in the
  tail, zero completed turns (swallows stay the dequeue-run's
  property), a ts-less tail, and a future-dated anchor all fail toward
  NO-FIRE. Accepted #1209 residual, pinned by test: the 256 KB tail can
  truncate older ok turns of a very long final turn (incl. the
  leading-implicit-turn shape) — the last visible turn genuinely failed
  and went silent, so the bounded fresh respawn is the accepted
  recovery.
  **#1241 cap parity for the four pre-#1209 triggers:** `dequeue-run` /
  `api-error-run` / `failed-turn-run` / `failed-turn-rate` respawns are
  bounded at the override site by the SAME two-part gate as the #1209
  trigger — the episode belt (`respawn_count < STALLED_MAX_RESPAWNS`,
  the cap decide() enforces on the slow path) AND a per-issue
  per-UTC-day cap `EPM_TICK_WEDGE_RESPAWNS_PER_DAY` (default 3;
  malformed or <1 → default — never a kill switch), on its own
  day-keyed, advancement-clear-EXEMPT counter (`wedge_respawns_today` /
  `wedge_respawn_day` in `stalled-<N>.json`) bumped ONCE per
  wedge-initiated fence episode at stop-initiation (the crash-recovery
  arm, which consults no cap, can complete a fresh-self-report wedge
  respawn — so the fence's spawn branch cannot be the counting or
  gating site, the same reason as #1209's). The two day budgets are
  INDEPENDENT. A stop-failed episode still consumes a budget unit
  (counted at stop-initiation — conservative in the safe direction) and
  stays push-visible via the one-time `session-stop-failed` marker. A
  cap-disarmed trigger goes quiet (no marker, no push; when EVERY
  family is cap-disarmed the 256 KB transcript read is skipped
  entirely); the slow stalled lane stays the backstop with its own
  decide()-side belt + exhausted marker.
  The single-refusal guard and
  fail-toward-NO-FIRE posture are unchanged. Accepted residuals: the
  watcher's own status-transition-keyed reconcile can refresh a
  SWALLOWED session's self-report, so the #779 dequeue-run shape then
  waits for staleness — identical to today; and a healthy session whose
  last 3 wakes each END in one transient trailing api-error row can
  false-respawn (bounded by the 3-consecutive-completed-turn bar, the
  episode belt + per-issue per-UTC-day wedge cap — #1241, above — the
  fence, the worktree hold, and the park exemptions).
  Bypasses the 2-miss debounce, the #759 K-downgrade and the 2h marker
  window — direct evidence beats proxies — but NOT the park exemptions
  (provision-in-flight / followups / spend-approval — re-probed once
  against the escalated action when the fresh-marker keep path lazily
  skipped them; a firing exemption vetoes the wedge), the worktree hold,
  or the fence; a wedge respawn resets `live_consecutive`. Unresolvable
  transcripts fail toward no-wedge.
- *(f) Alert-noise dedup (#1137).* At most ONE `session-stalled-alert`
  marker per staleness episode, across BOTH producers (decide()'s own
  alert path and the #759 K-downgrade lane, which bypasses decide()'s
  alerted-dedup by rewriting respawn->alert after it) — cleared on
  self-report advancement; repeat ticks keep the stderr line + the
  stalled-live sidecar row. And a `blocked` task whose newest
  status-changed is corroborated by an epm:failure within the park
  window (the halt-contract trail) is never stalled-alerted at all —
  the gate-push pass already pushed the blocked transition (a no-trail
  blocked task keeps the one-time alert). Accepted residual: a
  capacity-retry-eligible `no_compute_available` blocked task carries
  the halt trail by definition, so a re-driven session that wedges
  before leaving `blocked` gets no stalled alert — the crash-recovery /
  wedge / stale-registration lanes own that class. Incident #1092
  2026-07-07: 15:33Z alert on a 1s-apart failure+blocked park;
  20:43/21:03/21:23Z repeat alerts from the escalate→wt-hold-defer→
  downgrade 2-tick cycle (the criterion-7 reset after a DEFERRED
  escalation is by design — changing it would change respawn timing;
  only the marker noise was removed). Respawn/fence/hold semantics
  byte-identical.

Per-episode state for all five rides `stalled-<N>.json`
(`stop_pending_sid`/`stop_pending_ts`/`stop_retried`/`stop_failed_alerted`,
`wt_hold_count`, `daemon_blocked_ticks`/`daemon_blocked_pushed`,
`wedge_hits`), cleared on self-report advancement; pre-#845 files load with
safe defaults. While a fence episode is pending, the #759 K corroboration is
skipped (its debounce already served — re-downgrading the verify ticks would
stall the fence).

**Terminal-status act guard + source stamp (#1247).** The orphan sweep and
the stalled fence's spawn branch act only after a same-instant live-status
re-read (`_task_status`, canonical `PROJECT_ROOT` resolver, #844) POSITIVELY
returns an ACTIVE status — a stale pass-start snapshot can never produce a
respawn, a marker, or a cap-consume on a terminal/parked task (2-week
#662/#663/#867 marker loop, ~1,800 junk commits; the loop's root cause was
the test suite's own unstubbed `_post_progress_marker`, fixed alongside, but
the guard closes the whole stale-snapshot/TOCTOU class). Abort is loud
(`ORPHAN-ACT-GUARD` / `STALLED-ACT-GUARD` stderr line naming snapshot vs
live status + the aborted action); a `None` read (transient task.py failure)
defers one tick without erasing episode state; a positive non-ACTIVE read
clears the episode state (orphan) / the fence's `stop_pending_*` fields
(stalled). Residual ms-scale TOCTOU between the guard read and the act is
irreducible without locking — the guard shrinks the window from
minutes/multi-tick to ~ms. Every orphan-respawn / orphan-alert /
stalled-respawn marker note additionally carries a
`[src: host=… user=… pid=… sha=… root=…]` stamp (`_source_stamp()`,
running-checkout-derived by design — it exposes a stale worktree/clone copy)
so any future stale-instance poster identifies itself on its first marker.
Pinned by `tests/test_autonomous_session_watch.py::test_orphan_act_guard_*`
+ `test_stalled_fence_spawn_guard_*`.

**Infra-drain pass (execute the PM dispatch queue; task #633).** The PM
session's standing infra auto-dispatch rule (`research-pm.md` § Standing
rule, item 4b) adjudicates which `proposed` `kind: infra|batch` tasks are
RIPE and writes them oldest-first to
`~/.eps-autonomous/infra-drain-queue.json` (`ripe_oldest_first` ints,
`cap` — default 5, `holds` {id: one-word reason}, `updated_ts` ISO-8601
UTC). This pass EXECUTES that file with zero LLM judgment, spawning
`spawn_session.py spawn-issue --issue <N> --auto` for the oldest listed IDs
into free slots, where free = max(0, cap − occupied − pending): occupied =
`kind: infra|batch` tasks at the occupied-status set (the seven body
statuses `planning`/`plan_pending`/`approved`/`running`/`verifying`/
`interpreting`/`reviewing` PLUS `followups_running` — counting an in-flight
follow-up round only ever dispatches less; `proposed`/`blocked`/terminal do
not hold slots), read fail-CLOSED (any `list-by-status` failure skips
dispatching that tick — a partial count would under-count and
over-dispatch); pending = non-stale registrations (queue AND non-queue) of
still-`proposed` drain-kind tasks plus any with unreadable status/kind
(conservative), closing the PM-prunes-a-dispatched-ID overshoot. Per-ID
guards, each with a logged skip reason: PM hold; existing
`issue-<N>.json`/`manual-issue-<N>.json` registration — a STALE
(dead-at-boot) registration (task still `proposed`, older than
`EPM_INFRA_DRAIN_STALE_REG_GRACE_S`, default 30 min, recorded session id
definitively NOT live) stops pinning a pending slot and stops blocking
re-dispatch, with ANY missing signal failing toward keep-blocking; status ≠
`proposed`; kind outside `{infra, batch}` (loudly logged every tick — a
mis-kinded entry would auto-approve GPU spend outside the cap); and a retry
budget whose backoff window (`EPM_INFRA_DRAIN_BACKOFF_S`, default 1 h)
ALWAYS binds while a fresh PM `updated_ts` resets only the attempt COUNT
(`EPM_INFRA_DRAIN_MAX_ATTEMPTS`, default 3 per adjudication epoch; a future
`updated_ts` is clamped so it cannot void the budget). The PM remains the
ONLY *nuanced* ripeness judge — a missing/empty/invalid queue file is a
logged no-op, and un-riping an ID means rewriting the file (which also
re-arms the budget); the watcher makes exactly ONE narrow, mechanical
ripeness call beyond pure dispatch — predicate-hold auto-promotion (below)
— and nothing else. Daemon-gated like every spawning
pass; attempt state lives in
`~/.eps-autonomous/infra-drain-state.json` (self-pruned to the queue's ID
set; deliberately not a GC target); dispatch markers are generic
`epm:progress` notes carrying the
`[autonomous_session_watch:infra-drain-dispatch]` sentinel so they never
reset the orphan/stalled staleness clocks. Kill switch:
`EPM_DISABLE_INFRA_DRAIN=1`. `--infra-drain-only` runs just this pass
(pair with `--dry-run` for a live smoke).

*Session-dispatch stagger (#1059).* Consecutive REAL `spawn-issue --auto`
dispatches from the two infra loops (`infra_drain_pass` +
`proposed_infra_sweep_pass`, via the shared `_dispatch_infra_drain`
chokepoint) and from the batch filer `file_infra_task.py` are paced ≥
`EPM_SESSION_DISPATCH_STAGGER_S` apart (default 60s; `0` disables;
malformed → 60; clamped ≤ 300) via the shared last-writer-wins stamp
`~/.eps-autonomous/last-session-dispatch.json` — each fresh session is a
~100K-token cold context load and the org input-TPM 429 cap climbs at minute
boundaries (incident 2026-07-04: 5 workflow-fix sessions in ~3 min). The
watcher SLEEPS out the remainder before its pre-spawn re-check (worst-case
added tick wall-time ≤ cap × window — 5 min at defaults, ~25 min at the 300s
clamp — safe because the `watch.lock` non-blocking flock makes an overlapping
cron fire skip, never stack); the filer DEFERS (file-then-no-op, exit 0; the
sweep backstop dispatches within ~10 min). Only a real `"spawned"` outcome
records the stamp — suppressed / failed / dry-run never record. Named
residuals: the check→spawn→record sequence is a TOCTOU (the stamp is
last-writer-wins PACING, not an exclusion primitive — a concurrent dispatcher
already mid-spawn can co-dispatch inside the window, bounded at ~2 coincident
cold loads, and the window closes for everyone at the first record); and
crash-recovery, capacity-retry, campaign, and manual spawns are accepted
UNPACED sources (the #1059 Goal scopes to the infra loops + the batch filer).

*Predicate-hold auto-promotion (#633 follow-on).* BEFORE the dispatch
logic, the pass promotes any `holds` entry whose reason matches the PM's
cross-issue-predicate convention `predicate-<#N>-<short-desc>`
(`research-pm.md` step 3; live examples `predicate-535-slurm-attempt`,
`predicate-625-lands`) once its BLOCKING task #N has FINISHED — read
conservatively as task #N at `completed`/`archived`/`awaiting_promotion`
(the unambiguous "upstream finished" signal; the `<short-desc>` is never
interpreted — completion is sufficient for every predicate, e.g. a
completed #535 definitely had its live attempt). On a satisfied predicate
the hold is removed, the held id is merged into `ripe_oldest_first`
oldest-first, and the queue file is rewritten atomically (tmp+rename,
`updated_by: autonomous_session_watch:predicate-promote`, `updated_ts`
bumped — which re-arms the promoted id's retry budget), so the cleared
task dispatches THIS tick AND survives for the bg poller between PM passes.
Only one `task.py view` status read per distinct predicate (zero on the
common no-predicate tick); a non-predicate / malformed / unreadable /
not-yet-terminal hold is left UNTOUCHED (fail toward keep-blocking).
Skipped under `--dry-run` (decides + logs, never rewrites). This is the
between-passes accelerator the PM's own STATUS-pass re-evaluation already
backstops; the PM remains the nuanced judge for predicates that should
fire BEFORE completion and re-adjudicates the whole queue wholesale on its
next pass (its atomic overwrite always wins a race with this rewrite).

**Capacity-retry pass (re-drive a transient-infra `blocked` task; incident
#642, 2026-06-16).** The narrow inverse of the crash-recovery `decide()`
PARK rule, which treats EVERY `blocked` task as "keep, never respawn." Most
`blocked` tasks ARE deliberate halts awaiting a human (a `failure_class:
code|data` block, a factual question), and those MUST stay parked — but the
subclass where the block is purely transient infra capacity (the auto-router
exhausted every lane: latest `epm:failure v1` with `failure_class: infra` AND
a `reason` in the conservative allowlist `TRANSIENT_CAPACITY_REASONS`, today
`{no_compute_available}`) is code-ready and re-runnable the moment a lane
frees up — the failure marker itself self-flags "Retry on re-invocation." This
pass re-drives ONLY that subclass, via `spawn-issue --auto`; every other
`blocked` task is untouched (scope guard: the latest failure marker's
`failure_class`/`reason` must clean-match, so a non-capacity infra reason like
`codex-companion-probe-error` is left parked).

*No watcher-side capacity pre-check by design.* The `/issue` launch path is the
authoritative capacity gate: a re-drive re-enters `/issue` → Step 6 backend
dispatch → the router's GCP regional-quota headroom pre-check
(`backends/router.py` `_skip_gcp_lane_no_headroom`, `backends/gcp.py`
`preflight_quota_headroom`), which SMART-SKIPS a doomed lane WITHOUT burning a
daily attempt and WITHOUT GPU spend (#608), falls through to the free SLURM
lanes, and — if those are also full — simply re-blocks at ZERO GPU cost. So a
re-drive is never expensive; re-implementing a weaker copy of the router's
quota logic inside this 10-min fail-soft watcher would only duplicate + risk
drifting from the authoritative gate. The re-driven `--auto` session enforces
its own Step-2c plan-approval GPU-hour cap; this pass opens NO new spend path.

*Churn guards.* A per-task backoff window
(`EPM_CAPACITY_RETRY_BACKOFF_S`, default 1 h) binds on the NEWER of the block
timestamp and the last attempt (so capacity has time to free up and the pass
can't tight-loop), plus a per-UTC-day re-drive cap
(`EPM_CAPACITY_RETRY_PER_DAY`, default 4; the ATTEMPT counts whether or not the
spawn succeeds) — once exhausted, a one-time dashboard alert per day, never a
respawn. Daemon-gated like every spawning pass (re-drive POSTs to the Happy
daemon RPC); per-task state at `~/.eps-autonomous/capacity-retry-<N>.json`
(reaped by the generalized GC at terminal status — `blocked` is deliberately
NOT in `TERMINAL_FOR_GC`, so a live retry episode's state is never reset
mid-flight). `kind: campaign` `blocked` tasks are excluded (the recovery
command is the `/issue` skill, wrong for a campaign). Dispatch + exhausted
markers carry the `[autonomous_session_watch:capacity-retry]` /
`[...:capacity-retry-exhausted]` sentinels so they never reset the
orphan/stalled staleness clocks. Kill switch: `EPM_DISABLE_CAPACITY_RETRY=1`.
`--capacity-retry-only` runs just this pass (pair with `--dry-run` for a live
smoke against the real blocked-task set).

**Stale-blocked flag pass (flag — never flip — a stale `blocked` on a task
whose relaunch succeeded; task #1021, incident #742).** The capacity-retry
pass's non-spawning sibling: both scan the `blocked` set, but this pass is
daemon-INDEPENDENT — it spawns nothing (marker posts go via the `task.py`
subprocess). A crash-fix relaunch that succeeds on a task an earlier failed
round parked at `blocked` leaves the status stale: the run is healthy while
the folder says `blocked` (#742 ran healthy ~35h at status `blocked`,
2026-07-01→07-02). The orchestrator-side fix is the SKILL.md "A successful
relaunch also reconciles a stale `blocked`" rule; this pass is the
watcher-side BACKSTOP. Predicate (`decide_stale_blocked_flag`, every
missing signal failing toward silence): status `blocked` AND the latest
`epm:run-launched` NEWER than the transition into `blocked` (the normal
fail→block ordering keeps a deliberately-parked task quiet — only a launch
AFTER the block flags) AND real (non-watcher, non-deliberate-stop) progress
AT OR AFTER that launch within `EPM_STALE_BLOCKED_PROGRESS_FRESH_S`
(default 2h; malformed/non-positive values fall back to the default). On a
hit it FLAGS — one deduped `epm:progress` marker naming the reconcile
command (`task.py set-status <N> running`), one row in the durable sidecar
`~/.eps-autonomous/stale-blocked-events.jsonl` (the `.jsonl` suffix keeps
it out of the GC's `stale-blocked-*.json` glob), and one Telegram digest
line — and NEVER mutates status (false alert cheap, false flip dangerous;
the same conservative posture as the pod-safety alerts). Dedup is per
launch episode: `~/.eps-autonomous/stale-blocked-<N>.json` records the
flagged `epm:run-launched` ts, so the same launch never re-alerts while a
NEWER launch does; the state file is reaped by the generalized GC at
`completed`/`archived` only (`blocked` is deliberately NOT in
`TERMINAL_FOR_GC`, so a live episode's dedup state is never reset
mid-episode). Marker notes carry the
`[autonomous_session_watch:stale-blocked-flag]` sentinel so they never
reset the orphan/stalled staleness clocks. Kill switch:
`EPM_DISABLE_STALE_BLOCKED_FLAG=1`. `--stale-blocked-only` runs just this
pass (pair with `--dry-run` for a live smoke against the real blocked-task
set).

**Gate-push pass (2026-06-12 anti-stall redesign).** Telegram phone push on
gate-park/`blocked` transitions via the my-goat `telegram_push.sh` channel
(override for tests via `EPM_TELEGRAM_PUSH_SCRIPT`), transition-deduped:
per-issue state at `~/.eps-autonomous/gate-notify-<N>.json` records the last
observed status, and the push fires exactly once per transition INTO a user
gate (`awaiting_promotion`, `blocked`, or `plan_pending` only when the
over-cap spend-approval marker confirms it is the user gate — shared
`plan_pending_over_cap` predicate with `tick_triage.py`). Candidates cover
CAMPAIGN sessions (`campaign-<N>.json` registrations) as well as issue
sessions, with the same dedup and the same push-only guard posture; because
`blocked` — a campaign's only push-relevant gate — is campaign-TERMINAL and
the campaign pass stop-then-reaps the registration on the first tick it
observes it, the watcher snapshots campaign candidates BEFORE the campaign
pass and hands them to the gate-push pass. The issue side has the identical
race — `awaiting_promotion`, the most common user gate, is respawn-TERMINAL,
so the respawn pass deletes `issue-<N>.json` on the first daemon-up tick
observing the park (and the cwd fallback can't recover it: spawn-issue
sessions open at repo root) — so the watcher likewise snapshots the issue
registrations BEFORE the respawn pass and hands them in (`issue_snapshot=`).
Moved OUT of the
LLM-priced `/issue-tick` into this pure-Python pass — the watcher already
reads task status every 10 min for free, so gate-push latency IMPROVES from
the tick's backstop cadence to ~10 min; the tick-side `PushNotification` is
KEPT for now as a second deduped channel (dated removal note in
`.claude/skills/issue-tick/SKILL.md`), so the worst case is one duplicate
notification per gate transition, never a missed one. The same pass runs a
**status-transition-keyed title/self-report reconcile** — NEVER per-pass: an
unconditional rewrite would keep the self-report's `ts` permanently fresh and
structurally disable the stalled-detector's and reconcile pass's staleness
signals; a rewrite keyed on a STATUS CHANGE cannot mask a stall (the change
itself posts `epm:status-changed`, and a stalled session's status is by
definition not changing); only EXISTING self-reports are updated. It also
owns the **tick-runaway force-stop parachute** (#501 class — CRON-TEARDOWN
kept whiffing; 1,951 wasted ticks): `tick_triage.py` writes
`tick-runaway-<N>.flag` on the 3rd consecutive teardown-verdict tick (cleared
on any streak reset), and this pass force-stops the flagged issue's
session(s) — killing the session-scoped cron with them — under the
session-reconcile guards (DONE statuses `awaiting_promotion`/`completed`/
`archived` only, no live follow-up, no RUNNING pod, no `keep-running` tag)
but WITHOUT the 2h-idle + 2-miss accumulation (three consecutive
teardown-verdict ticks are already the corroboration). A `blocked` task also
writes runaway flags but its session may have the user live-parked in it —
alert loudly, never stop. Transition detection is daemon-independent; the
title-reconcile and force-stop arms degrade to skip/retry when the daemon is
down. `gate-notify-<N>.json` is in the terminal-status GC sweep set; the
`tick-runaway-<N>.flag` files self-clean inside the runaway processing
instead.

**Reconcile pass (auto-stop of parked sessions).** An issue-mapped session
whose task is parked/terminal (`awaiting_promotion`/`completed`/`archived`)
is AUTO-STOPPED after ≥2 consecutive checks once ALL hold: no live follow-up
inferred from events.jsonl (latest
`epm:run-launched`/`epm:followup-scope`/`epm:free-analysis-followup-run`
OLDER than the latest done-transition
`epm:promoted`/`epm:status-changed`/`epm:pod-terminated`/`epm:step-completed`),
every non-watcher marker + self-report idle > ~2h
(`EPM_SESSION_RECONCILE_IDLE_S`), no RUNNING `pod-<N>`, and no `keep-running`
tag (auto-stop default per user request 2026-06-10 — "stop the happy sessions
once they reach awaiting promotion" — superseding the same-day alert-only
decision; `EPM_SESSION_RECONCILE_AUTOSTOP=0` reverts to alert-only); sessions
of tasks at any other status (ACTIVE, `followups_running`, `blocked`), the PM
session, and unmapped chat sessions are never touched by this pass.

**Zombie-wrapper pass.** A live EPS session whose process tree has carried NO
inner Claude process for ≥2 consecutive checks AND ≥2h
(`EPM_ZOMBIE_WRAPPER_GRACE_S`) is auto-stopped REGARDLESS of issue mapping
(the 2026-06-11 class: 25 unmapped finished-issue sessions showed as
"running" indefinitely); never touched: the PM session (registered via
`spawn_session.py register-pm` / `spawn-pm` / the `/pm` bootstrap),
non-EPS-cwd sessions, and issue-mapped sessions at
active/`blocked`/`plan_pending` statuses; `EPM_ZOMBIE_WRAPPER_REAP=0` reverts
to alert-only.

**Idle-unmapped pass.** A third session reaper — auto-stops UNMAPPED EPS-cwd
sessions (no registry entry, no `issue-<N>` worktree cwd) whose resolved
Claude transcript has been idle ≥12h (`EPM_UNMAPPED_IDLE_REAP_S`) on ≥2
consecutive checks — the class BOTH other session reapers structurally
exclude (live-but-idle inner Claude + unmapped; the 2026-06-12 VM-lag
incident: 25 such sessions idle 19-43h held ~23 GB RSS); never touched: the
PM session, non-EPS cwds, issue-mapped sessions, wrappers holding a
controlling TTY (a live user terminal), and sessions whose transcript cannot
be resolved (a missing idleness signal FAILS TOWARD KEEP);
`EPM_UNMAPPED_IDLE_REAP=0` reverts to alert-only; records land in
`~/.eps-autonomous/idle-unmapped-events.jsonl` (an unmapped session has no
task to carry a marker). **#818 orphaned-tmux subclass:** a pane on an ORPHANED
tmux server (the server's socket was deleted from `$TMUX_TMPDIR/tmux-<uid>` so
NO new client can attach, AND the server holds zero `/dev/pts` attached-client
fds) is NOT a live terminal — that pane is unreachable by construction, so
`_is_live_user_tty` no longer counts it as live and the idle-≥12h wrapper on it
is reaped on the same ≥2-consecutive-miss schedule as everything else. The
mapping is process PARENTAGE (walk the wrapper's `ppid` chain for a
`comm == "tmux: server"` ancestor — the pane leaders are its child processes),
NOT the server's fd table. BOTH signals are required — a socketless-BUT-attached
server (e.g. a systemd-tmpfiles atime sweep of `/tmp/tmux-<uid>/` under a live
SSH session leaves the established connection intact) still holds ≥1 client fd
and is KEPT. Every uncertain probe (tmux absent, unreadable
`/proc/<pid>/stat` or `/proc/<pid>/comm`, unreadable socket dir, unreadable
server fd dir, a ppid-walk cycle, depth exhaustion, no `tmux: server`
ancestor) FAILS TOWARD KEEP. Gated by the
default-ON `EPM_ORPHANED_TMUX_REAP` kill-switch (`=0` disables the widening,
same shape as `EPM_UNMAPPED_IDLE_REAP`). **#720 short-window subclass:** an unmapped session
whose LAST-mapped task was TERMINAL — the "zombie session on a completed task"
ghost class (the respawn pass deletes `issue-<N>.json` at terminal → the
session goes unmapped, and its repo-root cwd can't re-map it) — is reaped on
the SHORT `LAST_MAPPED_TERMINAL_REAP_S` window (default 30 min, worst case
30 min + 2×10-min ticks = ~50 min), NOT the 12h default, via the #720
breadcrumb (`last-mapped-terminal-<sid>.json`, written at the respawn-pass
delete instant) + the running-pod + live-follow-up guards in
`_effective_idle_reap_s`. This is the home for the completed-task-session
reap: the *reconcile* pass cannot see this class (it is already unmapped by
the time reconcile runs), so the idle-unmapped short window owns it (#720;
#795 verified — no reconcile-pass change).

**Stale-registration pass (#845 d).** The fourth registration hygiene arm —
UNREGISTERS a LIVE-but-abandoned session registration (`issue-<N>.json` OR
`manual-issue-<N>.json`) whose resolved Claude transcript has been idle ≥12h
(`EPM_STALE_REGISTRATION_IDLE_H`; default == the idle-unmapped reap window)
AND whose self-report is equally stale (a MISSING self-report — manual
sessions never write one — does not rescue). Incident #665: a
16h-transcript-idle registered session held the `/issue` Step 0
single-orchestrator guard and blocked every re-drive. The crash-recovery
pass can't help (the sid IS live), the idle-unmapped reaper excludes MAPPED
sessions, and session-reconcile fires only on parked/terminal statuses —
this pass closes that square. UNREGISTER-ONLY: the session itself is NEVER
stopped (a manual session may hold a user TTY; the SKILL Step 0 stale-wake
ownership re-check guards a later wake). Deleting the registration releases
the Step 0 guard; for an ACTIVE task the registration-independent orphan
sweep re-drives it on its next tick (a PARK/terminal-status task is
deliberately NOT re-driven — the one-time marker,
`[autonomous_session_watch:stale-registration-unregister]`, logs the task's
status). Guards, all failing toward keep: dead sid (the crash-recovery
pass's property), unresolvable transcript, in-flight provision, fresh
worktree activity, fresh self-report. Runs AFTER `gate_push_pass` (the
gate-push-before-reaper ordering is a runaway-force-stop invariant),
adjacent to the two session reapers, consuming their shared daemon
session-list snapshot in place; daemon-gated (`children is None` ⇒ no-op). Unregistering
deletes the entry, which is self-deduping; a fresh re-registration restarts
the clock. Durable trace: `~/.eps-autonomous/stale-registration-events.jsonl`.

**Boot-death lane (#1267 arm 1 + #1287 arm 2, `boot_death_pass`).** The
die-at-or-before-turn-1 complement of the stale-registration pass: a freshly
dispatched AUTO session
(`issue-<N>.json` only — `manual-issue-*.json` is excluded by design, a
user-driven session is never auto-stopped) whose resolved Claude transcript
EITHER (arm 1, zero-response, `shape=zero-response`) contains ZERO response
rows (`_classify_wedge_row` ∉ {assistant, api-error}) OR (arm 2,
boot-refusal, `shape=boot-refusal`; #1287/#1277) whose 256 KB transcript
TAIL (`_transcript_tail_rows`) segments via `_segment_wake_turns` (#1127)
to ≥1 completed turn with EVERY completed turn failed — a refusal-killed
boot turn; a single visible ok turn keeps (the #1104 single-refusal guard)
— ≥30 min after `spawned_at` (`EPM_BOOT_DEATH_WINDOW_MIN`, minutes;
malformed/non-positive → default), with the transcript quiet ≥10 min (the
in-flight-first-turn guard) and the sid LIVE, is STOPPED via `_stop_session`
+ surfaced (Telegram push + an anti-liveness `epm:progress` marker,
sentinels `[autonomous_session_watch:boot-death-stop]` /
`[...:boot-death-cap-exhausted]`) instead of waiting ~12h for the
stale-registration unregister (incidents #1251–#1256: 9-row / ~11 KB
transcripts frozen ~7 s post-spawn — the session died during `/issue` skill
load; #1277: an 826 KB transcript whose boot turn ran ~74 s then died on a
refusal before the tick cron was armed — every other lane is structurally
blind to both shapes). Arm 1's whole-file
read is bounded at 256 KB (a larger transcript cannot be a ZERO-RESPONSE
boot-death → arm 1 keeps; arm 2's seek-tail read works at any size);
every unresolvable signal fails toward keep. Action is STOP-ONLY — no
unregister, no direct spawn: post-stop re-drive is fully owned by the
existing arms (ACTIVE → crash-recovery, ~20 min; `proposed` → the
proposed-infra sweep's stale-dead-registration grace, ~30–60 min). Bounds:
per-issue per-UTC-day stop cap (`EPM_BOOT_DEATH_STOPS_PER_DAY`, default 3;
malformed/<1 → default, never a kill switch; ONE day budget SHARED across
both arms), bumped ONCE at
stop-initiation (a stop failure still consumes a budget unit); state at
`~/.eps-autonomous/boot-death-<N>.json` (GC'd at `completed`/`archived`
only), durable trace `~/.eps-autonomous/boot-death-events.jsonl` (stop rows
carry `transcript=` + `stderr_excerpt=` + `api_error_excerpt=` forensics —
the refusal excerpt is SIDECAR-ONLY, never in the marker/push). There is NO episode
belt BY DESIGN — this is a stop lane, not a respawn lane; the downstream
re-drive arms carry their own belts/caps, and the auth-outage guard
suppresses the re-dispatch side (the infra-sweep/crash-arm spawn gates)
during a live outage episode, so the accelerated loop cannot spin during an
outage. At the cap the lane stops stopping (the live dead registration then
back-pressures re-dispatch exactly as today's 12h cycle) and fires ONE loud
cap push/marker per (issue, UTC day) — a recorded DEVIATION from #1241's
quiet-at-cap posture: the #1241 lanes have a slow backstop that still ACTS
at cap, while here the fallback is the very 12h silence this lane exists to
kill, so the cap moment is the highest-value alert of the day. Runs AFTER
`gate_push_pass`, immediately BEFORE the stale-registration pass, consuming
the shared reaper `children` snapshot in place; daemon-gated (`children is
None` ⇒ no-op). A same-tick overlap with stale-registration on a ≥12h-old
boot-dead entry is benign (stop + unregister compose). Kill switch:
`EPM_DISABLE_BOOT_DEATH_PASS=1`; `--boot-death-only` runs just this pass
(pair with `--dry-run` for a live smoke — a dry run stops nothing, posts no
real marker/push, and writes no state).

**Deliberate session takeover (`paused-takeover` sentinel; #866/#903).**
(Scope: this sentinel is a short-TTL session-TAKEOVER shield,
NOT a user pause — an indefinite user "pause <N>" routes to
`task.py set-status <N> on_hold` (the watcher PARK set; holds indefinitely)
per `.claude/skills/issue/SKILL.md` § User pause affordance; a stale
sentinel FAILS OPEN at ~`EPS_TAKEOVER_TTL_H`.) To take
over a stalled autonomous session WITHOUT racing the watcher, rename its
registration: `~/.eps-autonomous/issue-<N>.json` →
`issue-<N>.json.paused-takeover-<YYYYMMDD>` (any suffix after the literal
`.paused-takeover-`; `manual-issue-` same shape). While the sentinel is FRESH
(file mtime < `EPS_TAKEOVER_TTL_H`, default 6h; `touch` it to renew a longer
takeover): the orphan-respawn pass SKIPS the issue (logged, no state mutation),
and `spawn-issue --auto` suppresses with a rc-0 `TAKEOVER-SENTINEL HELD` line
(recognized by `spawn_output_suppressed`, so the crash-recovery, stalled,
orphan, infra-drain, capacity-retry arms + `file_infra_task.py` all book
nothing). Manual spawns warn-and-proceed (the #843 lease posture). A STALE
sentinel is ignored everywhere — FAIL OPEN: crash recovery resumes at the TTL,
so an abandoned takeover costs at most ~6h of un-watched active task. The
registration-KEYED passes (crash-recovery, stalled, stale-registration,
gate-push, reconcile) need no sentinel check: the rename removes the very file
they key on. Ending a takeover — ORDER MATTERS: FIRST re-establish a
registration (`spawn_session.py register-current --issue N` from the session
that now owns the issue, or rename the sentinel back), THEN delete the
sentinel — deleting first opens a one-tick window where the frozen `missed`
count (already ≥ threshold in the #866 shape) respawns immediately.
Alternatively just delete the sentinel and deliberately let the orphan sweep
respawn a fresh `--auto` driver on its next stale tick. Three operational
notes: (a) `EPS_TAKEOVER_TTL_H` is a FLEET-LEVEL knob — a session-local
export never reaches the watcher's cron env; renew a >6h takeover by
`touch`ing the sentinel, not by exporting the var. (b) Before renaming,
check for a fresh `issue-<N>.json` / dispatch lease — a respawn may have
JUST fired (the rename cannot recall an in-flight spawn; bounded to one
tick). (c) During a takeover the `/issue` Step 0 single-orchestrator guard
is registration-keyed and therefore BLIND — a human hand-driving the issue
should check for the sentinel first. Stopping the superseded session:
`spawn_session.py stop --session-id <sid>` — on a daemon-untracked sid it now
resolves the wrapper pid via the `~/.happy/logs` reverse map and reports a
verified kill-by-pid recipe (or SIGTERMs it under `--kill`; comm re-verified,
never auto-SIGKILL). Stale sentinels are GC'd after `max(7 days, the
configured TTL)`.

**Deliberate registration removal (`spawn_session.py unregister`; #1327).**
Deliberately removing an `issue-<N>.json` / `manual-issue-<N>.json` /
`campaign-<N>.json` registration (collision-yield, deliberate-stop cleanup)
goes through `spawn_session.py unregister` — never a hand `rm` on
`~/.eps-autonomous/` (the #952 shape: an unguarded rm can strip crash-recovery
from the healthy owner). Sid-matched by default: `unregister --issue N`
removes only files recording the CALLING session's Happy id
(ancestry-inferred, the `register-current` walk), so a yielding duplicate can
never delete the true owner's entry — a `KEPT-SID-MISMATCH` line is the guard
working, not a bug. Third-party cleanup of a DEAD session's file:
`unregister --issue N --session-id <dead-sid>` (removes only entries recording
that sid; no daemon-liveness check), or `unregister --force --issue N` for
unconditional operator cleanup (`--force` requires `--issue` and is refused
with `--session-id`). Takeover sentinels (`*.paused-takeover-*`) and
non-registration siblings (`dispatch-lease-*`, `campaign-watch-*`,
`pm-session.json`) are never touched by any invocation form.

**Program-orchestrator recovery pass (#660 leakage-program bash daemon).** The
leakage-theory program (#660) is sequenced by a BASH DAEMON
(`scripts/run_program_orchestrator.sh` in tmux `eps-program`), NOT a Happy
session — it gates/sequences the phase chain (Phase 1 → 2 → 3 → 4), spawning
each phase via `/issue --auto` and advancing on the critic-gated PASS. The
per-phase `/issue --auto` sessions are crash-recovered by the respawn pass; this
single bash process is NOT, so a VM reboot / OOM-kill mid-program silently stops
phase ADVANCEMENT (the active phase keeps running + parks, but nothing spawns the
next). This pass relaunches the daemon in tmux `eps-program` iff ALL hold (fail
toward NOT relaunching on any missing signal): the daemon is not already alive
(`pgrep -f run_program_orchestrator.sh`); the STOP sentinel
(`.claude/cache/program_orchestrator.STOP`) is absent (a STOP = deliberate halt —
every gate/phase HALT path `touch`es it); and the log
(`.claude/cache/program_orchestrator.log`) shows no deliberate exit (neither
"Program complete" nor "finished WITH HALTS", the two deliberate exits that leave
no STOP). Relaunch is idempotent — a fresh daemon re-checks every phase status and
won't double-spawn an active/terminal phase. Daemon-INDEPENDENT (it is not a Happy
session; runs every tick like `vm_disk_pass`). Kill switch:
`EPM_DISABLE_PROGRAM_ORCHESTRATOR_RECOVERY=1`. `--program-orchestrator-only` runs
just this pass (pair with `--dry-run` for a live smoke). Pinned by
`tests/test_autonomous_session_watch.py::test_program_orchestrator_*`.

**Happy injection-patch check pass (#726, `happy_patch_pass`).** A
daemon-INDEPENDENT, escalate-only pass (runs every 10-min tick in the
daemon-independent block next to `vm_disk_pass` / `data_disk_pass` /
`program_orchestrator_pass`, BEFORE the daemon-gated session passes) that
surfaces a reverted/drifted Happy daemon injection patch PROACTIVELY. The patch
(`scripts/patch_happy_daemon.py`, sentinel v4) teaches the vendored Happy daemon
to honor `claudeArgs` / `HAPPY_INITIAL_PROMPT`; it reverts on every `npm update
happy` (and the hashed bundle is renamed away), after which `spawn-issue --auto`
/ `spawn-campaign` spawn a session that boots empty and never fires its skill —
an idle "spawned but never ran" session (the failure CLASS behind #685; the
2026-06-28 idle-session pile itself was the distinct #720 mapping-loss cause).
The spawn-path guard (`spawn_session._verify_happy_patch_or_die`) is REACTIVE —
it fires only at the next spawn; this pass is PROACTIVE, so a revert is surfaced
within ~10 min rather than at the next dispatch. It reads the daemon file
in-process via `_happy_patch_check.classify_patch` (single source of truth for
the sentinel + path; single-digit-ms, no subprocess, no root), and on
`reverted`/`drifted` writes a `band=happy-patch` row to the shared disk-guard
sidecar (`.claude/cache/disk-guard-events.jsonl`) + a fail-soft `_telegram_push`,
deduped per-state (`~/.eps-autonomous/happy-patch-alert.json`) so it alerts once
per episode and re-alerts when the state changes. ESCALATE-ONLY: it NEVER
re-applies (that needs sudo — a password prompt would hang the autonomous
dispatch); `patched` and `missing` (no daemon file on this host) are clean
no-ops (the spawn-path guard owns the precise `missing` reachability
disambiguation via `daemon.state.json`). `--happy-patch-only` runs just this
pass (pair with `--dry-run` for a live smoke). Pinned by
`tests/test_happy_patch_check.py` (`test_watcher_pass_*`).

**CPU/memory-pressure guard pass (task #849, `cpu_guard_pass`).** A
daemon-INDEPENDENT, escalate-only pass (runs every 10-min tick in the
daemon-independent block right after `happy_patch_pass`) giving the fleet a
CPU/memory-pressure detection + attribution channel on the shared 32-core VM
(2026-07-02 incident: load 186-226 for hours; earlyoom SIGTERM sweeps silently
killed 4-7 GB analysis workers — exit 143, no traceback, misattributed for
hours). **Signals + thresholds** (each leg skips cleanly when its source is
unreadable — a missing signal never fires and never masks the others):
load5 > 1.5x nproc (`EPM_VM_CPU_GUARD_LOAD_FACTOR`), PSI cpu `some avg10` > 50
(`EPM_VM_CPU_GUARD_PSI_CPU_PCT`), PSI memory `full avg10` > 10
(`EPM_VM_CPU_GUARD_PSI_MEM_PCT`) — these three are RATE signals and need
**2 consecutive hot ticks** (~20 min at the 10-min cron,
`EPM_VM_CPU_GUARD_TICKS`) so a healthy short burst never alerts — PLUS a
**SINGLE-TICK urgent MemAvailable floor** at < 20% of MemTotal
(`EPM_VM_CPU_GUARD_MEMAVAIL_PCT`): memory can collapse 15%→3% inside one
10-min interval, and 20% sits one band above earlyoom's 10% kill floor, so
this leg fires while culprits are still alive — the fire stores a rolling
**pre-kill top-process snapshot** (top-CPU ∪ top-RSS via one `ps` call,
pid → issue via `/proc/<pid>/cwd` + cmdline hints) in the state file. A fire
writes ONE attributed `kind=vm-cpu-pressure` row to the DEDICATED sidecar
`.claude/cache/cpu-guard-events.jsonl` + a deduped `_telegram_push` (digest
queue); in-episode repeats are suppressed unless load5 grows > 25% or the
reason set changes, and recovery (no hot signals) resets the episode so a
later re-overload fires afresh. **earlyoom kill surfacing** runs EVERY tick,
threshold-independent: new journal kill lines (persistent cursor + key dedup;
first-run lookback deliberately ~30 min — the watcher is a monitor, not a
backfill tool; post-outage re-scan capped at 24 h) each produce one
`kind=earlyoom-kill` row carrying an explicit **`attribution_status:
attributed | unattributed`** — `attributed` (with `attribution_source:
pre-kill-snapshot`) only when the killed pid (or a unique comm) matches the
rolling snapshot; a sudden sub-tick collapse that beat the snapshot yields an
honest `unattributed` row (visibility guaranteed, attribution best-effort).
A failing/missing `journalctl` degrades the kill arm VISIBLY (stderr line +
`kill_arm: "unavailable"` on any pressure row that tick, cursor not
advanced), never silently. **WARN-ONLY:** never kills, never renices, never
signals any process (pinned by
`tests/test_cpu_guard_pass.py::test_cpu_guard_never_kills`). State singleton
`~/.eps-autonomous/vm-cpu-guard.json` (atomic write; `isinstance` type-guards
on every field read back). Kill switch `EPM_DISABLE_CPU_GUARD_PASS=1`;
`--cpu-guard-only` runs just this pass (pair with `--dry-run` for a live
smoke — dry-run performs zero writes and zero `subprocess.run`). NOTE: the
disk-guard ack-sentinel mechanism is DELIBERATELY omitted here — CPU/memory
episodes self-terminate on recovery (unlike a persistently-full disk), so the
recovery reset already bounds re-alert churn.

**Post-hoc external-marker triage observer (task #967, `triage_observer_pass`).**
A daemon-INDEPENDENT, **NON-GATING** pass (runs every 10-min tick in the
daemon-independent block right after `cpu_guard_pass`) auditing the `/issue`
Step 9 pre-dispatch external-marker triage duty POST-HOC (origin incident
#779: 10 unread external audit markers, an 18–20h serial grid launched
anyway). It sweeps REGISTRY tasks at ACTIVE ∪ {`awaiting_promotion`,
`blocked`} whose `events.jsonl` mtime falls inside a 48h lookback
(`EPM_TRIAGE_OBSERVER_LOOKBACK_H`), re-runs the #889 enumerator's window
semantics at each recent HISTORICAL dispatch record
(`task_workflow.audit_dispatch_triage`; per-task cursor so each record is
evaluated exactly once), and flags three violation classes: (a)
`launch-missing-line` (**warn**) — a launch marker with no triage line and no
adjacent boundary triage record within `EPM_TRIAGE_OBSERVER_ADJACENCY_S`
(30 min); (b) `breadcrumb-missing-line` — a line-less `stage-dispatch`
breadcrumb, THREE-WAY classified on its normalized stage token: a
known-benign family (`TRIAGE_NONCOMPUTE_STAGES` — extensible: append the
token + a live-example citation) never flags, POSITIVE compute evidence (a
`pid=` field or an exact `TRIAGE_COMPUTE_STAGE_TOKENS` match — grid / sweep /
battery / fit / fits / relaunch, never substring) is **warn**, an unknown
token is **info**; (c) `none-with-candidates` — a `none` disposition whose
pre-record boundary window re-enumerates non-empty after a 120s grace trim
(`EPM_TRIAGE_OBSERVER_GRACE_S`) — **info**, escalated to **warn** only on an
external-signature hit (`TRIAGE_EXTERNAL_SIGNATURES`). A MATURITY GATE
defers records younger than the adjacency window (the compliant adjacent-next
note may still land) — the cursor advances only past matured records, so a
record is judged exactly once, after its compliance window closes. Records
before `TRIAGE_DUTY_EPOCH_TS` (2026-07-03T05:00Z, the #889 landing) are
legacy and never flagged. **Channels:** every flag appends one row to the
dedicated sidecar `.claude/cache/triage-observer-events.jsonl`; `warn` flags
additionally get one deduped fail-soft `_telegram_push` digest line — capped
at `EPM_TRIAGE_OBSERVER_PUSH_CAP` (5) individual pushes per tick, overflow
rolled into ONE "+N more, see sidecar" summary push at the end of the pass
(#1167) — and one `epm:progress` review-nudge note on the task (anti-liveness
`[autonomous_session_watch:triage-observer]` sentinel; its `by="unknown"`
deliberately makes the note a triage candidate at the task's NEXT dispatch —
the flag is itself the advisory), capped at `EPM_TRIAGE_OBSERVER_MARKER_CAP`
(5) marker posts per tick. The two caps are independent; a warn beyond either
cap is PERMANENTLY sidecar-recorded and never deferred — beyond the marker
cap it stays sidecar+push-only; beyond the push cap its individual push is
replaced by the tick's single summary push. **Fire-once dedup:** key
`(issue, record_ts, violation-class)` in the state singleton
`~/.eps-autonomous/triage-observer.json` (atomic write; entries self-pruned
at `completed`/`archived`); a violation is a fixed historical record, so
there is no re-alert and no ack sentinel. **NON-GATING is a hard invariant:**
the pass never mutates task status, never stops a session, never blocks a
dispatch — pinned by tests at BOTH the subprocess-argv and the
in-process-mutator levels. **Invisible-by-construction residuals** (the
observer cannot see them): (i) a LYING triage line — mechanical presence +
enumerator-consistency are audited, truthfulness is not (a `2 applied` line
that misdescribes what was applied passes); (ii) record-less launches —
compute started via direct SSH with no launch marker and no breadcrumb
leaves nothing to audit (the `/issue` skill-drift rule is the control for
that class). Known bounded miss: a 9a-ter compute fit dispatched under a
`free-analysis-followup` breadcrumb with no triage line surfaces only at
`info` (its duty is content-dependent). Kill switch
`EPM_DISABLE_TRIAGE_OBSERVER=1`; `--triage-observer-only` runs just this
pass (pair with `--dry-run` for a live smoke — dry-run performs zero writes
and zero `subprocess.run`).

**Verdict-disagree observer pass (task #1170, `verdict_disagree_pass`).** A
daemon-INDEPENDENT, **NON-GATING** pass (right after `triage_observer_pass`)
auditing the four MARKER-MODE doubled review sites (workflow.yaml
§ ensemble_review — code-reviewer / interpretation-critic /
clean-result-critic / follow-up-critic; the `critic` site reconciles
in-context and is unobservable) for the #825 misclassification shape: the
LATEST round per (issue, site) whose Claude + Codex durable verdicts BOTH
exist with parseable OPPOSITE-class verdicts (pass-class vs fail-class), no
role-matched `epm:review-reconcile`, and — for proximity-tier pairings only
— no Codex no-show evidence. The pure predicate
(`task_workflow.unreconciled_disagreement_rounds`) pairs two-tier: Tier 1
round-aligned via `ensemble_verdicts_present`, then a time-proximity
fallback (`EPM_VERDICT_DISAGREE_PAIR_PROXIMITY_S`, 6h) for the observed
sentinel/version round drift (#825: Claude sentinel v5 vs Codex bare
version 7 for the same logical round); a 1h grace window
(`EPM_VERDICT_DISAGREE_GRACE_S`) lets an in-flight reconcile land, and
no-show evidence (`epm:codex-task-failed`, a codex-scoped `epm:failure`,
the #1204 quota-skip note) suppresses TIER-2 pairings only, scanned from
`min(pair_ts) − EPM_VERDICT_DISAGREE_EVIDENCE_LOOKBACK_S` (2h) — a Tier-1
both-present pair is never evidence-suppressed (evidence explains an absent
twin, not two present verdicts). **Channels:** one row per finding to the
dedicated sidecar `.claude/cache/verdict-disagree-observer-events.jsonl` +
one deduped fail-soft `_telegram_push`; **NO task marker** (deliberate
divergence from the triage observer — this flag's consumer is a human, not
the next dispatch). Fire-once dedup key `(issue, role, round_label)` in the
state singleton `~/.eps-autonomous/verdict-disagree-observer.json`
(self-pruned at `completed`/`archived`). **KNOWN BENIGN-FIRE class:** a
Step 5c-bis mechanical-contract-only strip, a 9a-bis procedural strip, or a
cap-5 all-stripped-continue resolves a PASS-vs-FAIL round WITHOUT a
reconciler and logs to chat only, so it flags by design (auditing
orchestrator self-serve dismissals of a FAIL is in scope); the FAIL
marker's own `**Blocker tags:**` line (an all-mechanical tag set) is the
one-glance disambiguator. **Coverage limits:** latest-round-only (a
superseded earlier-round disagreement is moot and round re-derivation is
unreliable under sentinel drift); Tier-2 evidence suppression is
site-agnostic (`epm:codex-task-failed` notes don't reliably name the role).
Sweep scope reuses the triage observer's enumerator (ACTIVE ∪
{`awaiting_promotion`, `blocked`}, `EPM_VERDICT_DISAGREE_LOOKBACK_H` 48h
events-mtime recency). Kill switch `EPM_DISABLE_VERDICT_DISAGREE_OBSERVER=1`;
`--verdict-disagree-only` runs just this pass (pair with `--dry-run` for a
live smoke — zero writes).

**Root-draft observer pass (task #1341, `root_draft_pass`; origin incident
#1320).** A daemon-INDEPENDENT, ESCALATE-ONLY pass (runs right after
`verdict_disagree_pass`) flagging stale UNTRACKED `*.py` drafts abandoned in
the SHARED repo-root working tree — dirt that matches the `.py` leg of
step9c's `DIRTY_CODE_PATHSPEC` (`scripts/step9c_baseline.py`) and therefore
flips EVERY task's Step 9c pristine-oracle compare fleet-wide indeterminate,
silently (#1320: two untracked `scripts/issue825_*.py` drafts poisoned the
ledger 9+ hours). Predicate: one read-only
`git --no-optional-locks status --porcelain -- *.py` at the main root
(`--no-optional-locks` = never takes the shared root's index lock), keep
untracked (`?? `) `.py` entries, flag those with file mtime age >
`EPM_ROOT_DRAFT_ESCALATE_HOURS` (default 3 h; tracked-modified ` M` dirt is
deliberately out of scope — the named extension trigger if a future
fleet-wide indeterminacy traces to it; `.claude/worktrees/` is gitignored so
worktrees never enumerate). **Channels:** one row per fired path to the
dedicated sidecar `.claude/cache/root-draft-events.jsonl` (with best-effort
`issue<M>_` filename attribution + a fail-soft `task.py view` status label)
+ ONE deduped fail-soft `_telegram_push` digest per tick naming every fired
path; NO task markers (the verdict-disagree posture — a name-collision
mis-attribution must cost nothing on any task record). **Dedup:** per-path
fire-once + `EPM_ROOT_DRAFT_REALERT_HOURS` (24 h) re-alert TTL in the state
singleton `~/.eps-autonomous/root-draft-observer.json` (atomic tmp+rename;
recovered paths pruned so a re-appearance re-fires immediately).
**ESCALATE-ONLY is a hard invariant:** the pass NEVER deletes, moves,
chmods, or git-mutates anything — its only writes are the state file + the
sidecar (pinned by
`tests/test_autonomous_session_watch.py::test_root_draft_pass_never_deletes`);
rescue is always the OWNING session committing or relocating its draft. A
git-status failure warns + skips the tick with no state write (fail toward
logged-skip, never a silent "no drafts"). Kill switch
`EPM_DISABLE_ROOT_DRAFT_PASS=1`; `--root-draft-only` runs just this pass
(pair with `--dry-run` for a live smoke — zero writes, zero task.py reads
beyond the read-only enumeration).

**Auth-outage guard pass (task #1027, `auth_outage_pass`).** Fleet-level
respawn suppression for an Anthropic auth outage — or ANY fleet-wide
instant-death cause (poisoned CLI credential, broken `claude` binary, a
reverted Happy patch that escaped `happy_patch_pass`). Origin incident
2026-07-03: a poisoned Claude CLI credential (recovered by `/login`) killed
every freshly spawned session on arrival and the watcher churned
die-on-arrival respawns for hours — the per-task caps (`STALLED_MAX_RESPAWNS`,
the orphan/capacity per-day caps) bound per-ISSUE churn but nothing read the
fleet-level correlation. Runs immediately after the single per-tick daemon
probe, BEFORE every spawn arm.

- **Detection signature (derived purely from watcher-owned state, no
  log-grepping):** every watcher-issued spawn records an event
  `{issue, ts, arm, prev_spawned_at}` (`prev_spawned_at` = the replaced
  registry entry's `spawned_at`; `None` for arms with no predecessor —
  infra-drain / capacity-retry / most orphan first spawns, which therefore
  never qualify). An event is an *instant-freeze respawn* when
  `0 <= ts − prev_spawned_at <= EPM_AUTH_OUTAGE_FRESH_DEATH_MIN` (default 60
  min — the die-on-arrival cycle is spawn grace 15 min + 2 misses × 10-min
  cron ≈ 25-45 min; a healthy multi-hour session never qualifies; env nudges
  below the ~45-min ceiling fall back to the default). ≥
  `EPM_AUTH_OUTAGE_MIN_EVENTS` (3) such events across ≥
  `EPM_AUTH_OUTAGE_MIN_ISSUES` (2) DISTINCT issues inside
  `EPM_AUTH_OUTAGE_WINDOW_MIN` (180 min) trigger an episode — cross-issue
  correlation is the false-positive guard.
- **While an episode is active,** every spawn arm — crash-recovery, stalled
  (gated at the fence CALLER so the stop+respawn is skipped as a UNIT),
  orphan, infra-drain (both callers), capacity-retry, and campaign (both
  callers, also unit-gated) — is suppressed via the #843 `"suppressed"`
  channel, so callers book nothing (no attempt, no backoff, no per-day cap).
  Non-spawning passes (pod-safety, gate-push, reapers) are deliberately NOT
  gated. ONE Telegram push fires at trigger (evidence-enriched: a
  best-effort auth-signature grep over the newest 3 `~/.happy/logs` files —
  push TEXT only, never the trigger; `auth-string:` evidence gets a
  `/login` hint, `churn-only` gets a broader checklist) and at most one at
  resolution.
- **Canary-probed resume:** every `EPM_AUTH_OUTAGE_CANARY_INTERVAL_MIN` (30
  min) the pass arms a single-tick token; the first eligible issue-arm spawn
  consumes it and becomes the canary — a REAL session respawn, so it probes
  the exact CLI-credential auth path real sessions use (a watcher-side
  `ANTHROPIC_API_KEY` probe would test the WRONG credential — the incident
  was recovered by `/login`). The canary identity binds only after a
  `"spawned"` result (the fresh registry `happy_session_id` is persisted;
  liveness is read from that PERSISTED sid, never a registry re-read); a
  canary surviving ≥ `EPM_AUTH_OUTAGE_CANARY_SURVIVAL_MIN` (20 min) resolves
  the episode, a dead/invalidated one re-arms one interval later
  (round-robining away from the last failed issue once). The campaign arm
  never consumes the token (campaign registrations live at
  `campaign-<N>.json`, unreadable for canary liveness).
- **Fail-open, twice over:** any internal guard error (gate, record hook,
  pass body, sidecar, push) logs a warning and behaves as "no outage"
  (spawns proceed — a false suppression is a fleet-wide crash-recovery
  blackout, strictly worse than churn); and an episode older than
  `EPM_AUTH_OUTAGE_MAX_EPISODE_H` (6 h) expires with a push — enforced in
  the pass AND independently in the gate, so a wedged pass can never
  suppress past the TTL. On resolve/expire the `last_episode_end_ts`
  watermark (not event deletion) blocks stale re-trigger: qualifying events
  need BOTH `ts` and `prev_spawned_at` past the watermark, so pre-resolve
  churn and backlog respawns of episode-era predecessors never re-open the
  episode, while a genuinely persistent >6 h outage re-accumulates NEW
  events and legitimately re-triggers (~one push per ~7 h).
- **State singleton** `~/.eps-autonomous/auth-outage.json` (never GC'd;
  events pruned to 2× the window); **sidecar**
  `.claude/cache/auth-outage-events.jsonl` (one row per trigger /
  canary-armed / canary-failed / resolve / expire transition). Kill switch
  `EPM_DISABLE_AUTH_OUTAGE_GUARD=1`; `rm ~/.eps-autonomous/auth-outage.json`
  clears a live episode instantly; `--auth-outage-only` runs just this pass
  (pair with `--dry-run` for a live smoke — zero writes, zero pushes).
- **Accepted residuals (named, deliberate — do NOT "fix"):** (a) hang-style
  outages (sessions never die → no respawn events → no trigger; also no
  churn, so the cost is only the missing alert); (b) new-spawn-only outages
  (`prev_spawned_at=None` arms never count; bounded by the infra/capacity
  per-day caps); (c) the program-orchestrator recovery pass can relaunch the
  #660 program daemon (an indirect spawner) during an episode — v1 residual
  per the must-ask fence on gating non-spawning passes; (d) two independent
  issue-specific crash loops can false-trigger — bounded by canary self-heal
  (~50 min) + the 6 h TTL + one push; (e) a wedged-but-registered canary can
  false-resolve — bounded: a still-broken fleet re-accumulates ≥3 new events
  and re-triggers; (f) detection fires at the second respawn generation
  (~60-75 min into an outage) — it trims the tail, not the head; (g) the
  `EPM_AUTH_OUTAGE_FRESH_DEATH_MIN` deviation band is clamped ≥45 min (a
  lower value breaks the die-on-arrival replay shape).

## Dedicated data disk for `.claude/worktrees/` (#681)

The heavy active-task footprint (`.claude/worktrees/` — every `issue-<N>`
worktree + its per-issue `data/issue_<N>/{hf_dl,g*_dl,store}` caches) lives on a
dedicated **512 GB `pd-balanced` GCP persistent disk mounted at `/mnt/eps-data`**
(env `EPS_VM_DATA_DISK_PATH`), bind-mounted back onto `.claude/worktrees` so every
consumer resolves the SAME path transparently. The disk is provisioned in the
`introsp-experiments` project (where the VM lives), NOT the GPU project.

**Per-task ext4 project quotas (the per-tenant bound).** Each `issue-<N>` subtree
carries an ext4 project id == the issue number with a hard byte cap
(`EPS_ISSUE_DISK_CAP_GB`, default 128 GB), assigned at worktree creation by
`new_worktree.sh` (`chattr -p <N> +P` + `setquota -P <N>`, opt-in via
`EPS_WORKTREE_ASSIGN_QUOTA=1`). A write past the cap fails loud with `EDQUOT`
(the same signal the RunPod MooseFS per-pod quota produces) while every OTHER
issue keeps writing — so one task can neither exhaust `/` nor starve another.
Recovery is always resize / raise-cap, NEVER delete active data.

**Dual-disk watch — escalate-only on the data disk.** The disk guards watch BOTH
filesystems: `/` (boot disk) with the existing byte-floor logic, and
`/mnt/eps-data` with **PERCENT / statvfs-derived** thresholds (size-invariant —
a future resize cannot push the fire point past the wedge the way the mirrored
boot-disk byte floors would). The data-disk pass is **ESCALATE-ONLY**: the
`/`-rooted reclaim arms (`uv cache prune`, the stale-log sweep) never run keyed
off the data disk; `vm_disk_guard.run_guard(disk_path="/mnt/eps-data",
reclaim_tiers=False)` runs only tier (b) (terminal-cache reap + active-cache
escalation), and the watcher's dedicated `data_disk_pass` (called from `main()`
next to `vm_disk_pass`, every 10-min tick) drives the percent helpers
`decide_vm_disk_pct` (alert/critical band) + `decide_subfloor_pct`
(`EPM_VM_DATA_DISK_SUBFLOOR_PCT` default 85%) off `statvfs(/mnt/eps-data)`,
escalate-only (no reclaim arm), and attributes the WORKTREE-internal caches via
`repquota -P` per-project usage (du fallback). Both passes are clean no-ops when
the mount is absent (before / without the cutover).

**Non-canonical caches + the /workspace hub-cache arm (#911).** The guard's
tier (b) ALSO sweeps NON-CANONICAL issue-keyed caches — top-level `/tmp/` dirs
named `i<N>*` / `issue<N>*` / `issue-<N>*` / `issue_<N>*` / `*_<N>`, and `data/`
dirs named `issue…<N>…{_dl,_hfstage,_cache}` — under the same terminal-reap /
active-escalate contract PLUS a 48 h recency keep, a nested
`store/`+`eval_results/` block, and a positive re-downloadability-evidence gate
(hub-layout markers or data-repo-prefix mirror verification; predicate failures
escalate, never delete). A fourth, boot-pass-only arm age-gates the VM's
pod-style `/workspace/.cache/huggingface` hub cache (repos unused ≥ 14 days,
`EPS_VM_WORKSPACE_HF_CACHE_MAX_AGE_DAYS`), pod-guarded (`ismount('/workspace')`
OR pod-side detection refuses) so it can never run where `/workspace` is a real
volume. A fifth arm (#1377) trims stale unref'd non-newest REVISIONS in the
user home `~/.cache/huggingface/hub` cache (unref'd revisions older than 7
days, `EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS`, root override
`EPS_VM_HOME_HF_CACHE`; the newest + every ref'd revision per repo is always
kept), pod-guarded and riding the same `main()`-only opt-in. The `/tmp/` +
`/workspace` opt-in lives ONLY in the two CLI `main()`
bodies (`tmp_root=production_tmp_root()`; library calls are hermetic by
construction), the escalate-only data-disk pass never sweeps `/tmp/`, and
report-only runs surface their evidence via the `--json` structured fields
(`active_cache_attributions` / `noncanonical_candidates` /
`total_discovered_bytes`) — never the sidecar. Kill switch:
`EPM_SKIP_NONCANONICAL_CACHE_SWEEP=1`.

**Janitor exemption.** The stale-GCP-VM janitor (above) sweeps the
`eps-persona-gpu-jun2026` GPU project for ephemeral GCE INSTANCES. The
`/mnt/eps-data` data disk is in a DIFFERENT project (`introsp-experiments`) and
is a PERSISTENT disk, not an ephemeral instance — so it is out of the janitor's
scope by construction and is intentionally never reaped.
