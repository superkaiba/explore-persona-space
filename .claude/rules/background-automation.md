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
the gate-push pass, the program-orchestrator recovery pass, the
stale-registration pass, the CPU/memory-pressure guard pass, and three session reapers — the session-vs-status
reconcile pass, the zombie-wrapper pass, and the idle-unmapped pass.

**Stall-detection hardening (#845; the five 2026-07-01 incident classes).**
The stalled detector + the two respawn arms carry five hardening mechanisms:

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
- *(e) Prompt-wedge fast lane* (`decide_prompt_wedge`): a LAZY
  transcript-tail probe (happy-log-only resolution, last 64 KB) escalates
  straight to the respawn arm on ≥3 (`EPM_TICK_WEDGE_MIN_DEQUEUED`)
  consecutive trailing wedge-evidence rows with no assistant turn —
  verified `{"type": "queue-operation", "operation": "dequeue"}` records
  (co-primary) and/or promptless prompt-type user rows (secondary)
  (incident #779: 5 prompts enqueued+dequeued with no turn for ~90 min).
  Bypasses the 2-miss debounce, the #759 K-downgrade and the 2h marker
  window — direct evidence beats proxies — but NOT the park exemptions
  (provision-in-flight / followups / spend-approval — re-probed once
  against the escalated action when the fresh-marker keep path lazily
  skipped them; a firing exemption vetoes the wedge), the worktree hold,
  or the fence; a wedge respawn resets `live_consecutive`. Unresolvable
  transcripts fail toward no-wedge.

Per-episode state for all five rides `stalled-<N>.json`
(`stop_pending_sid`/`stop_pending_ts`/`stop_retried`/`stop_failed_alerted`,
`wt_hold_count`, `daemon_blocked_ticks`/`daemon_blocked_pushed`,
`wedge_hits`), cleared on self-report advancement; pre-#845 files load with
safe defaults. While a fence episode is pending, the #759 K corroboration is
skipped (its debounce already served — re-downgrading the verify ticks would
stall the fence).

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
volume. The `/tmp/` + `/workspace` opt-in lives ONLY in the two CLI `main()`
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
