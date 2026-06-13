"""Crash-recovery + pod-safety + stalled-detector watcher for autonomous and
interactive issue sessions (plus campaign sessions, task #586).

Eleven passes; items 1-8 run in that order, with the CAMPAIGN pass (item 9)
right after pass 2, the IDLE-UNMAPPED-SESSION pass (item 10) right after
pass 7, and the INFRA-DRAIN pass (item 11) between pass 5 (the orphan sweep)
and pass 6 (session-reconcile), all before the GC pass:

1. **VM disk-headroom pass.** Watch free space on the VM root filesystem —
   the host of every orchestrator session, the worktree ``.venv``s, the uv
   cache, and the HF cache. Pods have their own guards (``pod_disk_guard.py``,
   the preflight fallocate probe); the VM had none until / hit 100%
   mid-pipeline and every foreground Bash spawn in the orchestrator session
   failed silently — exit 1, zero output — stalling the interpretation loop
   ~20 min, undiagnosable from inside the session (task #552, 2026-06-10).
   Below :data:`VM_DISK_ALERT_FREE_BYTES` (~20 GiB): loud log + ONE
   dashboard-visible marker per low-disk episode, AND run the stale-worktree
   sweep (``worktree_audit.py --apply`` — it carries all its own keep-guards
   and the disk-pressure grace tightening, and it is the remediation that
   actually frees the big space: each stale worktree is a ~14G checkout;
   re-armed per :data:`VM_DISK_RECLAIM_REARM_S`). Below
   :data:`VM_DISK_RECLAIM_FREE_BYTES` (~15 GiB, env
   ``EPM_VM_DISK_CRITICAL_GIB``): additionally run the safe, fail-soft cache
   reclaims, each logging its own freed-space line into the marker note
   (``wandb artifact cache cleanup`` to ~1GB — a pure download cache, 17.6 GB
   sat there in the 2026-06-11 episode; ``uv cache prune`` — never
   ``--force``, lock-contention timeout = clean skip; ``npm cache clean
   --force`` — npm's required confirmation flag, the clean itself is safe;
   HF hub TTL eviction — ``scan_cache_dir``/``delete_revisions`` on
   revisions idle > :data:`VM_DISK_HF_TTL_S`, never recently-accessed repos;
   sweep ``/tmp/claude-*`` trees idle > 3 days). Detection runs every 10-min
   tick, so remediation does too — the once-daily worktree cron alone lost
   the 2026-06-11 race (17 GiB -> 1.2 GiB within hours). The episode state
   clears only on DECISIVE recovery (alert + a
   :data:`VM_DISK_CLEAR_HYSTERESIS_BYTES` margin, ~22 GiB) so free space
   flapping around the alert boundary stays ONE episode instead of re-firing
   the audit/alert on each dip. Runs FIRST because a full root disk makes
   every later subprocess in this very watcher flaky; never crashes the pass.
2. **Crash-recovery (respawn pass).** Re-spawn an autonomous (`--auto`) `/issue`
   session whose driver process has died. Gated on daemon reachability — it
   reasons about session liveness, which is unknowable during a daemon outage.
3. **Pod-safety pass.** Reconcile RUNNING managed pods (``pod-<N>`` / legacy
   ``epm-issue-<N>``) against their task's STATUS. Two conservative actions:

   - **AUTO-STOP** (reversible, never terminate) a RUNNING pod whose task is
     already DONE (``completed`` / ``awaiting_promotion`` / ``archived`` /
     ``cancelled``). The experiment is provably finished, so a still-RUNNING
     pod is an escaped pod (Step-8 terminate failed, or it was never run
     through Step 8). Stopping it is unambiguously correct.
   - **ALERT** (loud log + one-time dashboard-visible marker, NO stop) a
     RUNNING pod whose task is in a pod-active status (``approved`` /
     ``running`` / ``uploading`` / ``verifying``) but has shown no real marker
     progress for > ``ALERT_STALE_HOURS``. This is the likely-abandoned
     mid-run case. We do NOT stop it: a false alert is a cheap nudge; a false
     stop would kill a healthy run.

   The pod-safety pass does NOT use session-cwd liveness as a stop trigger
   (see "Why STOP is keyed on task status, not session liveness" below) and
   does NOT need the daemon, so it runs unconditionally — even during a daemon
   outage. Only the respawn pass is daemon-gated.
4. **Stalled-detector pass (ALERT + AUTO-RESPAWN).** Detect an autonomous
   session whose Happy id is in the live set (so the respawn pass doesn't
   touch it) but whose self-report timestamp + latest non-watcher progress
   marker have BOTH been frozen > ``STALLED_WINDOW_S`` (default 45 min).
   This catches the "alive but bg-Bash chain dead" case where the session
   looks healthy to the respawn pass but is no longer making progress.
   AUTO-RESPAWNS the session (stop-then-respawn) when its task is in an
   :data:`ACTIVE` status AND the Happy daemon is reachable; otherwise
   degrades to ALERT-ONLY. The respawn is bounded by a per-episode
   :data:`STALLED_MAX_RESPAWNS` cap (default 3) — once exhausted, the
   pass falls back to a loud one-time "auto-recovery exhausted" marker
   and waits for the user.  Promoted from the ALERT-ONLY behavior shipped
   in 2026-06-05 after task #518 (2026-06-08) confirmed the detection
   fires on true positives but was never re-driven.  Manual registrations
   (``manual-issue-<N>.json``, written by bare ``spawn-issue``) are ALSO
   scanned, in ALERT-ONLY mode: the same staleness detection fires the
   one-time alert, but a user-driven session is NEVER auto-respawned
   (#505 round-2 orphaning, 2026-06-10 — a dead bare-spawned session at
   an ACTIVE status previously orphaned silently because this pass only
   globbed ``issue-*.json``).
5. **Orphan sweep (registration-INDEPENDENT safety net).** Every other
   session pass starts from the registry files (``issue-<N>.json`` /
   ``manual-issue-<N>.json``), so an ACTIVE-status task with NO registration
   is invisible to all of them. That blind spot orphaned #472 for 10.5h on
   2026-06-10: the task parked at ``awaiting_promotion`` (TERMINAL → the
   respawn pass DELETED its registry entry per :func:`decide`), a same-issue
   follow-up later flipped it back to ``running`` driven by an unregistered
   interactive session, that session died at 08:40Z, and no pass could see
   it. This pass inverts the direction: enumerate ACTIVE-status tasks via
   ``task.py list-by-status``, and for any task with NO live REGISTERED
   session AND no real progress marker within
   :data:`ORPHAN_STALENESS_S_DEFAULT` (~90 min, env
   ``EPM_ORPHAN_STALENESS_MIN``), RESPAWN via ``spawn-issue --auto`` (which
   re-registers it), capped at :data:`ORPHAN_MAX_RESPAWNS_PER_DAY` attempts
   per task per UTC day; when the cap is exhausted or the task's only
   registration is MANUAL (user-driven — never auto-respawn, #505), degrade
   to a one-time loud alert marker. Daemon-gated like the respawn pass
   (liveness is unknowable during an outage; a mass respawn would duplicate
   pods).
6. **Session-reconcile pass (sessions-vs-status; AUTO-STOP by default).**
   Mirror of the pod-safety auto-stop arm for Happy SESSIONS: a live
   session mapped to an issue (registry entry, or an ``issue-<N>``
   worktree cwd for unregistered / superseded zombie generations) whose
   task is parked/terminal (:data:`SESSION_RECONCILE_DONE` =
   ``awaiting_promotion`` / ``completed`` / ``archived``) is STOPPED via
   ``spawn_session.py stop`` once ALL of these hold, confirmed across the
   same >=2-consecutive-checks guard as the pod pass:

   - **idle** — every activity signal (the newest NON-watcher marker of
     ANY kind on the task, plus the per-issue self-report file) is older
     than :func:`_session_idle_s` (default 2h, env
     ``EPM_SESSION_RECONCILE_IDLE_S``);
   - **no live inline follow-up** — the latest follow-up signal marker
     (:data:`_SESSION_FOLLOWUP_SIGNAL_KINDS`: ``epm:run-launched`` /
     ``epm:followup-scope`` / ``epm:free-analysis-followup-run``) is
     OLDER than the latest done-transition marker
     (:data:`_SESSION_DONE_TRANSITION_KINDS`);
   - **no RUNNING managed pod** for the issue (a live pod means work may
     still be in flight — e.g. a follow-up that has not posted its
     ``epm:run-launched`` yet);
   - **no ``keep-running`` tag** (the explicit user override).

   AUTO-STOP is the DEFAULT (user request 2026-06-10: "Can we stop the
   happy sessions once they reach awaiting promotion?" — supersedes the
   same-day alert-only decision; 73 registered sessions had accumulated
   ~0.5-0.6GB RSS each and 14 were stopped manually with this exact
   predicate). Set ``EPM_SESSION_RECONCILE_AUTOSTOP=0`` to fall back to
   the old alert-only posture (loud log + one-time marker). A stop is
   VERIFIED on the next tick — the daemon ACK is not trusted as a kill:
   an ACKed-but-still-alive session gets ONE stop retry, then a one-time
   loud marker, and the episode state is cleared only once the session
   actually leaves the live set (:func:`_check_stop_verification`).
   NEVER touches: sessions with no issue mapping (the PM session, chat
   sessions), tasks at any other status (ACTIVE statuses, ``blocked``,
   and ``followups_running`` — a same-issue follow-up round is
   executing there). Motivated by the 2026-06-10 disk-full incident:
   15+ idle sessions of weeks-old completed/archived tasks (the respawn
   pass deletes the registry entry at a TERMINAL status but never stops
   the session) pinned their 10-15G worktrees against the stale-worktree
   sweep and held deleted-file handles (~37G phantom disk usage).
   Daemon-gated like the respawn pass (session liveness is unknowable
   during a daemon outage).
7. **Zombie-wrapper pass (AUTO-STOP by default).** Stop a daemon-tracked
   Happy session whose process tree has carried NO inner Claude process
   (cmdline match on :data:`_CLAUDE_CMDLINE_MARKERS`) for >= ``threshold``
   consecutive checks AND >= the :func:`_zombie_wrapper_grace_s` window
   (default 2h) — REGARDLESS of issue mapping. Every other session pass is
   keyed on a registry entry or an ``issue-<N>`` worktree cwd, so a
   finished session that lost its mapping (registry GC'd at the terminal
   transition, cwd = repo root) is invisible to all of them even though
   its inner Claude exited: 25 such zombies had accumulated by 2026-06-11,
   showing as "running" in ``spawn_session.py list`` indefinitely until a
   manual sweep. The grace window is load-bearing, not cosmetic: a live
   wrapper revives its inner Claude IN PLACE on the next phone message
   (the remote-mode launcher blocks on ``nextMessage()`` BEFORE spawning
   the Claude SDK subprocess), so a no-Claude snapshot alone can be a
   healthy idle session. NEVER touches: the PM session (excluded via the
   explicit ``pm-session.json`` registration written by ``spawn-pm`` /
   ``register-pm`` / the `/pm` skill bootstrap), non-EPS-cwd sessions, and
   issue-mapped sessions at :data:`ZOMBIE_STATUS_EXCLUDE` statuses.
   ``EPM_ZOMBIE_WRAPPER_REAP=0`` falls back to alert-only. Stops are
   verified on the next tick (daemon ACK != kill), mirroring the
   session-reconcile contract. Daemon-gated.
8. **GC pass.** Reap per-issue state files (``manual-issue-<N>.json``,
   ``issue-progress/<N>.json``, ``issue-tick-last-status/<N>.json``,
   ``stalled-<N>.json``, ``orphan-<N>.json``) for tasks in
   :data:`TERMINAL_FOR_GC`
   (``completed`` / ``archived``) — conservative on ``awaiting_promotion``
   and ``blocked`` (the user could still be interacting). Independent of
   the destructive passes; safe to run last. (``session-reconcile-<N>.json``
   is deliberately NOT in its sweep — those files track episodes whose
   task is BY DEFINITION terminal, so the terminal-status GC would reset
   the miss counter every tick; they are reaped by their own
   live-session-keyed GC inside the session-reconcile pass. The
   per-session ``zombie-wrapper-<sid>.json`` and ``idle-unmapped-<sid>.json``
   files are likewise out of its per-issue sweep — reaped by their own
   passes' live-session-keyed GCs.)
9. **Campaign pass** (runs right after pass 2; task #586). Driven by
   ``campaign-<N>.json`` registry entries (written by ``spawn_session.py
   spawn-campaign``): respawn a dead campaign session whose task is ACTIVE
   (``approved``/``running``) via ``spawn-campaign``; a progress watchdog
   posts ``epm:campaign-stalled v1`` when the newest skill-posted
   ``epm:campaign-*`` marker AND every child-task marker are older than
   ``EPM_CAMPAIGN_STALL_S`` (default 2h) with a live session, then
   stop-then-respawns on the second consecutive stalled check (cap 3 per
   episode); a budget backstop alerts once per episode when
   ``campaign-state.json`` shows GPU-hours committed > total; entries +
   watch state are reaped when the campaign task is terminal
   (``completed``/``archived``/``blocked``) — the still-live session is
   STOPPED first (reap-before-stop would unmap an immortal idle session),
   and the reap is deferred while the daemon is unreachable. The orphan
   sweep skips
   ``kind: campaign`` tasks (its ``spawn-issue --auto`` recovery would boot
   the wrong skill); see the campaign-pass section comment for the full
   cross-pass interaction notes.
10. **Idle-unmapped-session pass (AUTO-STOP by default; runs right after
   pass 7, before the GC pass).** The third session reaper, closing the
   class BOTH earlier reapers structurally exclude (2026-06-12 VM-lag
   incident: 25 unmapped sessions idle 19-43h each, LIVE inner Claude plus
   ~8 MCP server children, ~23 GB RSS total): the zombie-wrapper pass only
   fires when the tree has NO inner Claude, and the session-reconcile pass
   only touches issue-MAPPED sessions. This pass stops an unmapped EPS-cwd
   session whose resolved Claude transcript (per-wrapper-pid via
   ``session_resolver``) has been idle >= ``EPM_UNMAPPED_IDLE_REAP_S``
   (default 12h) on >= ``threshold`` consecutive checks. NEVER touches: the
   PM session, non-EPS cwds, issue-mapped sessions (registry entry or
   ``issue-<N>`` worktree cwd — the reconcile/zombie passes own those),
   wrappers holding a controlling TTY (a terminal Thomas may be sitting
   at), and sessions whose idleness signal cannot be resolved (missing
   data FAILS TOWARD KEEP — loud log, episode state frozen, never a reap).
   ``EPM_UNMAPPED_IDLE_REAP=0`` falls back to alert-only. Stops are
   verified on the next tick (daemon ACK != kill), mirroring the
   zombie-wrapper contract; records land in
   ``~/.eps-autonomous/idle-unmapped-events.jsonl`` (no task to carry a
   marker, by definition). Daemon-gated.
11. **Infra-drain pass (execute the PM-adjudicated dispatch queue; task
   #633; runs between pass 5 and pass 6).** The PM session's standing infra
   auto-dispatch rule adjudicates which ``proposed`` kind-infra/batch tasks
   are RIPE and writes them oldest-first to
   ``~/.eps-autonomous/infra-drain-queue.json`` (``ripe_oldest_first``,
   ``cap``, ``holds`` {id: reason}, ``updated_ts``). This pass EXECUTES that
   file with zero LLM judgment: it spawns ``spawn-issue --issue <N> --auto``
   for the oldest listed IDs into free slots under the cap, where free =
   max(0, cap - occupied - pending): occupied = kind-infra/batch tasks at
   :data:`INFRA_DRAIN_OCCUPIED_STATUSES` (fail-CLOSED — any status-read
   failure skips dispatching this tick), pending = non-stale registrations
   of still-``proposed`` drain-kind (or unreadable) tasks. Per-ID guards,
   each with a logged skip reason: PM hold; existing
   ``issue-<N>.json``/``manual-issue-<N>.json`` registration (a
   dead-at-boot registration — still-``proposed`` task, older than
   ``EPM_INFRA_DRAIN_STALE_REG_GRACE_S`` (default 30 min), session
   definitively not live — is STALE and stops blocking; ANY missing signal
   fails toward keep-blocking); status != ``proposed``; kind not in
   :data:`INFRA_DRAIN_KINDS` (loud every tick — a mis-kinded queue entry
   would auto-approve GPU spend outside the cap); and a retry budget whose
   backoff window (``EPM_INFRA_DRAIN_BACKOFF_S``, default 1 h) ALWAYS binds
   while a fresh PM ``updated_ts`` resets only the attempt COUNT
   (``EPM_INFRA_DRAIN_MAX_ATTEMPTS``, default 3 per adjudication epoch;
   future timestamps clamped). The PM remains the ONLY ripeness judge —
   missing/empty/invalid queue file is a logged no-op; un-riping an ID =
   rewriting the file. Attempt state lives in
   ``~/.eps-autonomous/infra-drain-state.json`` (self-pruned to the queue's
   ID set; not a GC target). Kill switch ``EPM_DISABLE_INFRA_DRAIN=1``;
   daemon-gated like every spawning pass; dispatch markers carry
   :data:`_INFRA_DRAIN_NOTE_SENTINEL` so they never reset the
   orphan/stalled staleness clocks. ``--infra-drain-only`` runs just this
   pass (pair with ``--dry-run`` for a live smoke).

Why each pass exists
--------------------
**Respawn:** the `/loop 10m /issue <N>` driver and any `CronCreate(durable=False)`
backstop live *inside* the session's Claude process, so they die with it — a
process crash / OOM / VM reboot leaves an autonomous experiment stalled until
someone manually `happy resume`s it. This watcher runs OUT of process (a real VM
crontab line, like cron_worktree_audit.sh) and re-spawns the dead session.

**Pod-safety:** ``pod_audit.py`` buckets a managed-name RUNNING pod as ``active``
and never stops it, so an escaped pod whose experiment is already DONE burns to
the 7-day TTL. The auto-stop arm closes that residual. The alert arm surfaces
the harder mid-run-death case (an interactive session died with its pod RUNNING
mid-experiment) without risking a false stop.

Coverage notes (deliberate gaps you should know about)
------------------------------------------------------
* A RUNNING pod observed while its task is in ``interpreting`` / ``reviewing``
  is NOT stopped or alerted (classified ``"other"``). Those stages don't drive
  pods (interp/review reads from WandB/HF, not the pod), so the burn is
  bounded — it's just caught one stage later, at ``awaiting_promotion``, when
  the auto-stop arm fires.
* The ``keep-running`` task tag (which exempts a pod from /issue Step 8's
  auto-terminate) IS consulted by the auto-stop arm: a RUNNING pod whose task
  is DONE but carries the tag is NOT auto-stopped (it covers legitimate
  post-completion work, e.g. a user-directed follow-up re-eval on an
  ``awaiting_promotion`` task — the #530 incident, 2026-06-09, where this
  pass stopped pod-530 four times mid-follow-up before the tag was consulted).
  The skip is observable: a log line on every pass plus ONE dashboard-visible
  marker per pod incarnation (deduped via the ``keep_running_noted`` flag in
  the pod-safety state file, which is cleared when the pod leaves the RUNNING
  set). Cost trade-off: an exempted pod burns until it is stopped manually
  (``pod.py stop --issue <N>``) or the tag is removed (``task.py remove-tag
  <N> keep-running``) — removing the tag re-arms the auto-stop arm on the
  next watcher run, with a fresh >=2-checks accumulation. The alert and
  stalled-detector arms ignore the tag (they never stop pods anyway).
* The auto-stop arm ALSO inspects events.jsonl for a live inline follow-up:
  if a task's latest follow-up signal marker (``epm:run-launched`` /
  ``epm:followup-scope`` / ``epm:free-analysis-followup-run`` —
  :data:`_POD_FOLLOWUP_SIGNAL_KINDS`) is NEWER than its latest
  ``epm:promoted`` / ``epm:status-changed`` (i.e. a user-approved inline
  follow-up — the CLAUDE.md "Routing experiment intent → Follow-up" path —
  is in flight on a promoted/completed/awaiting_promotion/
  archived parent), the stop is SKIPPED with the same once-per-incarnation
  marker semantics as the keep-running exemption (deduped via the
  ``followup_noted`` flag). ``epm:followup-scope`` covers USER-CHAT inline
  follow-ups, which post the scope marker BEFORE the run launches (refs
  #573 — the run-launched-only inference stopped healthy follow-up pods
  pod-530/531 8x + pod-477 3x on 2026-06-09 in exactly that window).
  Precedence: ``keep_running`` (explicit user
  tag) beats ``followup_active`` (inferred from events). The skip re-arms
  naturally on the next tick when the follow-up posts its next
  ``epm:status-changed`` / ``epm:promoted`` event newer than the
  follow-up signal. The #477 incident, 2026-06-10, motivates this: an
  inline follow-up on a promoted task ran 3 cycles of auto-stop → manual
  re-provision in <1h before the user added the ``keep-running`` tag.

Why STOP is keyed on task status, not session liveness
------------------------------------------------------
An earlier design stopped a pod when no live session was "driving" it, using
cwd-based liveness (a live Happy session whose cwd is the issue's worktree).
That signal is WRONG as a stop trigger: interactive `/issue` sessions are
spawned with cwd = REPO ROOT (the worktree doesn't exist yet at spawn time —
``spawn_session.py``), so a perfectly healthy interactive session reads as
"dead" by the cwd test. Stopping on that signal would kill live experiments.

So the STOP trigger is now task STATUS, which is unambiguous: a ``completed`` /
``awaiting_promotion`` / ``archived`` / ``cancelled`` task provably needs no
pod. Session liveness is gone from the stop path entirely. The mid-run case
(where status alone can't distinguish "healthy long run" from "abandoned") is
handled by the ALERT arm keyed on marker-progress staleness, not by a stop.

Mechanism
---------
Respawn: `spawn_session.py spawn-issue --auto` writes one registry file per issue
at ``~/.eps-autonomous/issue-<N>.json`` recording the Happy session id + cwd +
the GPU-hour cap. This watcher, each run:

  * reads the task's current status (via `task.py view --json`);
  * decides per :func:`decide` whether to RESPAWN / KEEP / DELETE the entry;
  * a session is "alive" iff its recorded id is in the daemon's live set OR
    the issue's MANUAL registration (``manual-issue-<N>.json``, written by
    bare ``spawn-issue``) records a live id — i.e. a user-driven replacement
    session counts as the driver. The earlier worktree-cwd fallback ("a live
    session sits in ``.claude/worktrees/issue-<N>``") was REMOVED 2026-06-10:
    ``spawn-issue --auto`` spawns drivers WITH cwd = the issue worktree, so
    every superseded driver generation matches the cwd test, and one idle
    zombie generation kept #518 reading ``alive=True`` for ~11h after the
    registered driver died (the registry rewrite on every respawn makes the
    recorded-id + manual-id checks the precise signal the cwd heuristic was
    approximating);
  * a dead session is only re-spawned after ``--threshold`` (default 2)
    consecutive misses, so a transient daemon-list glitch never double-spawns;
  * single-flight via flock so two overlapping cron fires can't race.

RESPAWN re-invokes `spawn_session.py spawn-issue --auto`, which rewrites the
registry with the new id and ``missed=0``. Parked/terminal tasks are never
re-spawned (see the status sets below); awaiting_promotion is a human gate.

Pod-safety: the watcher lists team pods, keeps the RUNNING managed ones, maps
each to its issue via the canonical ``pod_lifecycle`` helpers, reads each
task's status + latest real-progress timestamp, and per
:func:`decide_pod_safety` decides STOP (done task) / ALERT (stale pod-active
task) / KEEP. AUTO-STOP runs ``pod.py stop --issue <N>`` after the same 2-miss
accumulation as the respawn pass; it is reversible (volume preserved;
``pod.py resume`` re-provisions) and NEVER a terminate. Per-pod miss counts +
the last-observed real-progress timestamp + the alerted flag persist in their
own small state files (``~/.eps-autonomous/pod-safety-<N>.json``) because
interactive issues have no ``issue-<N>.json`` entry.

Run: ``uv run python scripts/autonomous_session_watch.py [--dry-run] [--threshold N]``
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# scripts/ is sys.path[0] when run as `python scripts/autonomous_session_watch.py`,
# so its siblings import directly. Reuse spawn_session's daemon readers +
# registry constants, the live RunPod API, AND the canonical managed-pod
# helpers from pod_lifecycle (rather than re-deriving a per-issue regex — the
# old `epm-issue-<N>`-only regex never matched the canonical `pod-<N>` names,
# so the whole pass was dead code).
import session_resolver
from pod_lifecycle import _is_managed_pod, _issue_from_pod_name
from runpod_api import list_team_pods
from spawn_session import (
    AUTONOMOUS_REGISTRY_DIR,
    PROJECT_ROOT,
    _infer_issue_from_path,
    _live_children,
    _live_session_ids,
    _load_pm_session_ids,
    _load_session_issue_map,
    _load_session_meta,
)
from tick_triage import plan_pending_over_cap

# Active-drive statuses: a dead session here SHOULD be resurrected.
# `followups_running` is ACTIVE (2026-06-10, un-phantomed): a same-issue
# follow-up round holds this status for the whole abbreviated cycle
# (plan amendment -> run -> re-fold), so a dead session there is mid-work
# and must be re-driven. Under the legacy children-in-flight semantics a
# respawned session just re-shows the child table and exits — harmless.
ACTIVE = {
    "planning",
    "approved",
    "running",
    "verifying",
    "interpreting",
    "reviewing",
    "followups_running",
}
# Park statuses: legitimately waiting on the user or a gate — never re-spawn,
# but keep the entry (it may flip back to ACTIVE, e.g. plan_pending -> approved).
# Members MUST equal the runtime enum `task_workflow.STATUSES` exactly when
# unioned with ACTIVE + TERMINAL (pinned by
# `test_status_sets_are_disjoint_and_cover_enum`). The reviewer caught a
# phantom `clarifying` member here — not in the runtime enum, so it could
# never match `_task_status` output. Removed for that reason; behavior is
# unchanged (a `clarifying` status would have hit the `decide` unknown→keep
# branch, also "keep") but the explicit membership was dead code, and a
# phantom invites the next reader to assume it's a real status.
PARK = {"proposed", "plan_pending", "blocked", "on_hold"}
# Terminal statuses: the autonomous run is done — drop the entry.
# awaiting_promotion is terminal HERE (experiment finished; the user promotes
# manually — no more auto-driving needed).
TERMINAL = {"awaiting_promotion", "completed", "archived"}

# Hard backstop: drop a registry entry whose task has not progressed in this
# long, so a stuck/unknown-status entry cannot linger and re-spawn forever.
MAX_ENTRY_AGE_S = 14 * 24 * 3600


def decide(status: str, alive: bool, missed: int, threshold: int = 2) -> tuple[str, int]:
    """Pure decision: given a task's status, whether its session is alive, and
    the consecutive-miss count, return ``(action, new_missed)`` where action is
    ``"respawn"`` | ``"keep"`` | ``"delete"``.

    Safety: only an ACTIVE status with a session confirmed dead on
    ``threshold`` consecutive checks (default 2 = ~20 min at a 10-min cron)
    yields ``"respawn"``. Parked tasks reset the miss count and are kept;
    terminal tasks are deleted; an unknown status is kept without ever spawning.
    """
    if status in TERMINAL:
        return ("delete", 0)
    if status in PARK:
        return ("keep", 0)
    if status in ACTIVE:
        if alive:
            return ("keep", 0)
        new_missed = missed + 1
        if new_missed >= threshold:
            return ("respawn", 0)
        return ("keep", new_missed)
    # Unknown status (e.g. a renamed enum): do nothing, keep the entry so a
    # human notices rather than silently dropping or spawning.
    return ("keep", missed)


# ─── pod-safety pass ─────────────────────────────────────────────────────────

# Task statuses for which a still-RUNNING pod is PROVABLY unnecessary: the
# experiment finished (or was abandoned/archived), so the pod is an escaped
# one (Step-8 terminate failed, or it never went through Step 8). Auto-stopping
# these is unambiguously safe — there is no live experiment to interrupt.
# `blocked` is DELIBERATELY excluded: a blocked pod may be under active
# investigation, so it's KEPT (alert-only if stale), never auto-stopped.
# Members MUST be a subset of `task_workflow.STATUSES` — phantom names like
# `cancelled` were dropped (not in the runtime enum, so they could never
# match anyway; `followups_running` was a phantom here too until it joined
# the runtime enum on 2026-06-10 — it now lives in POD_ACTIVE below). The
# disjoint+subset invariant is pinned by
# `test_status_classes_subset_of_authoritative_enum`.
AUTO_STOP_DONE = {"completed", "awaiting_promotion", "archived"}

# Task statuses during which a pod is legitimately in use mid-experiment.
# A RUNNING pod here is NOT auto-stopped (status alone can't tell a healthy
# long run from an abandoned one); instead, if it has shown no real marker
# progress for > ALERT_STALE_HOURS, the alert arm fires (loud log + one-time
# marker), never a stop.
# `uploading` is NOT in the runtime enum and was dropped; `interpreting` /
# `reviewing` are real statuses but DELIBERATELY excluded — they don't drive
# pods (interp/review reads from WandB/HF, not the pod), so a RUNNING pod
# observed there classifies as "other" and the auto-stop fires later when the
# task reaches `awaiting_promotion`. GPU burn bounded, just later than ideal.
# `followups_running` IS pod-active (2026-06-10): a same-issue follow-up
# round holds this status through provision -> run -> upload-verify, so its
# RUNNING pod is legitimately in use (alert-only if stale, never auto-stop).
POD_ACTIVE = {"approved", "running", "verifying", "followups_running"}

# How long a pod-active task may go without a real progress marker before the
# alert arm fires. Healthy runs post epm:progress regularly (poll_pipeline), so
# a multi-hour gap is a real signal of an abandoned session. A false alert is a
# cheap nudge, so this can be conservative without harm.
ALERT_STALE_HOURS = 6.0

# Per-pod state lives in its OWN small file, separate from the autonomous
# registry (issue-<N>.json), because INTERACTIVE issues — the main case this
# pass exists for — have no registry entry at all.
_POD_SAFETY_PREFIX = "pod-safety-"

# Substring stamped into every alert marker note this pass posts, so the
# staleness check can EXCLUDE the watcher's own alerts from "real progress" —
# otherwise an alert would reset the staleness clock and the gap could never
# grow past the threshold again (the alert would only ever fire once by luck of
# timing). Real progress is "any progress marker NOT posted by this watcher."
_ALERT_NOTE_SENTINEL = "[autonomous_session_watch:pod-stale-alert]"

# Substring stamped into the auto-stop marker note, mirroring the alert
# sentinel. Not used for staleness filtering (a stopped pod's task is DONE, so
# staleness is irrelevant there) but keeps both watcher-posted markers
# self-identifying on the dashboard.
_AUTOSTOP_NOTE_SENTINEL = "[autonomous_session_watch:pod-auto-stop]"

# Substring stamped into the one-time "keep-running exemption" marker posted
# when the auto-stop arm would have fired but the task carries the
# keep-running tag. Posted at most once per pod incarnation (deduped via the
# `keep_running_noted` flag in the pod-safety state file) so a tagged pod is
# visible on the dashboard without 20-minute marker spam.
_KEEP_RUNNING_NOTE_SENTINEL = "[autonomous_session_watch:pod-keep-running-skip]"

# Substring stamped into the one-time "inline-follow-up exemption" marker
# posted when the auto-stop arm would have fired but the task's events.jsonl
# shows a `epm:run-launched` marker NEWER than its transition into the current
# DONE status (i.e. a legitimate user-approved inline follow-up provisioned a
# fresh pod on a promoted/completed/awaiting_promotion/archived parent — see
# the CLAUDE.md "Routing experiment intent → Follow-up" bullet). Posted at
# most once per pod incarnation (deduped via the `followup_noted` flag in the
# pod-safety state file). Same dashboard-visible / no-spam semantics as the
# keep-running-skip marker. Incident #477 (2026-06-10): a promoted task ran
# 3 cycles of pod auto-stop → manual re-provision in <1h before the follow-up
# launches were recognized as legitimate.
_FOLLOWUP_NOTE_SENTINEL = "[autonomous_session_watch:pod-followup-skip]"

# Substring stamped into every session-stalled-alert marker note. Same role as
# _ALERT_NOTE_SENTINEL for the pod-safety pass: a session-stalled alert is
# posted as epm:progress and MUST be filtered out of the "real progress" set,
# or the alert would reset the very staleness window it measures.
_STALLED_ALERT_NOTE_SENTINEL = "[autonomous_session_watch:session-stalled-alert]"

# Substring stamped into every session-stalled AUTO-RESPAWN marker note. The
# respawn IS a recovery action (not just an alert) but it gets posted as
# epm:progress for the same reason: it's a watcher-posted event that must NOT
# bias the real-progress staleness clock on the NEXT tick (otherwise a
# successful respawn would mask the next staleness episode).
_STALLED_RESPAWN_NOTE_SENTINEL = "[autonomous_session_watch:session-auto-respawn]"

# Substring stamped into the one-time "auto-recovery cap exhausted" marker
# fired when STALLED_MAX_RESPAWNS respawns in the same episode have all
# failed to restore progress. Same staleness-filter contract as the others.
_STALLED_EXHAUSTED_NOTE_SENTINEL = "[autonomous_session_watch:session-auto-respawn-exhausted]"

# Substring stamped into the one-time VM-disk-low marker posted by the vm-disk
# pass (once per low-disk episode, on each ACTIVE registered autonomous issue —
# the sessions that will die first when / fills up). Same staleness-filter
# contract as the others: a watcher-posted note must never reset a session's
# real-progress clock.
_VM_DISK_NOTE_SENTINEL = "[autonomous_session_watch:vm-disk-low]"

# Substring stamped into the marker posted when the orphan sweep RESPAWNS an
# active-status task that had no live registered session (the #472 class:
# registry entry deleted at a TERMINAL park, task later revived by a
# same-issue follow-up with no re-registration). Same staleness-filter
# contract as the others.
_ORPHAN_RESPAWN_NOTE_SENTINEL = "[autonomous_session_watch:orphan-respawn]"

# Substring stamped into the one-time alert the orphan sweep posts instead of
# respawning — when the daily respawn-attempt cap is exhausted, the respawn
# failed, or the task's only registration is MANUAL (user-driven sessions are
# never auto-respawned, #505). Same staleness-filter contract as the others.
_ORPHAN_ALERT_NOTE_SENTINEL = "[autonomous_session_watch:orphan-alert]"

# Substring stamped into the one-time alert the stalled / orphan-respawn passes
# post when they would have respawned a ``followups_running`` parent whose own
# `/issue` pipeline is done (latest ``epm:step-completed`` step=10
# exit_kind=parked) and that has at least one open child task — i.e. a parent
# parked waiting on a user-gated child (the canonical case is a child at
# ``awaiting_promotion`` whose ``task.py promote`` is a user-only gate). Such a
# parent provably cannot advance by respawning the parent session — only user
# action on the child (or all children reaching terminal) unblocks it.
# Suppression is alert-only and dedup'd via the per-pass state file's
# ``followups_child_alerted`` flag, mirroring ``alerted`` + ``refresh_attempted``.
# Incident: task #533, 2026-06-11 — three respawn-and-park cycles in two hours
# while child #546 sat at ``awaiting_promotion``, each respawn re-posted the
# same ``epm:step-completed step=10 exit_kind=parked`` and exited. Same
# staleness-filter contract as the others.
_FOLLOWUPS_AWAITING_CHILD_NOTE_SENTINEL = "[autonomous_session_watch:followups-awaiting-child]"

# Substring stamped into the marker posted when the stalled / orphan passes
# detect a COMPLETED same-issue follow-up round stranded at
# ``followups_running`` (round-end markers newer than the round's
# ``epm:followup-scope`` — the owning session died after the final gates but
# before executing the designed re-park) and execute the re-park
# (``task.py set-status <N> awaiting_promotion``) on the session's behalf.
# Incident: task #533, 2026-06-11/12 — the round finished every gate
# (clean-result re-gate PASS, worktree merged, ``epm:step-completed``
# step=9a-bis exit_kind=parked at 10:54Z) but the session died before the
# re-park; the followups-awaiting-child exemption then suppressed every
# respawn, freezing the task at ``followups_running`` for ~26h until a
# manual re-park. Same staleness-filter contract as the others.
_FOLLOWUP_ROUND_REPARK_NOTE_SENTINEL = "[autonomous_session_watch:followup-round-repark]"

# Substring stamped into the one-time alert the session-reconcile pass posts
# (only in the EPM_SESSION_RECONCILE_AUTOSTOP=0 alert-only fallback) when a
# live session has outlived its parked/terminal (awaiting_promotion/
# completed/archived) task by > the idle grace window. Same staleness-filter
# contract as the others — CRITICAL here: the alert lands on the very task
# whose marker inactivity it measures, so without the sentinel filter the
# alert itself would end the idle episode it reports.
_SESSION_RECONCILE_ALERT_NOTE_SENTINEL = "[autonomous_session_watch:session-reconcile-alert]"

# Substring stamped into the marker posted when the session-reconcile pass
# actually STOPS the idle session(s) of a parked/terminal task (the default
# posture as of 2026-06-10). Same staleness-filter contract.
_SESSION_RECONCILE_STOP_NOTE_SENTINEL = "[autonomous_session_watch:session-reconcile-stop]"

# Substring stamped into the one-time alert posted when a session the
# session-reconcile pass stopped is STILL alive after the one allowed retry —
# the Happy daemon ACKed the stop RPC but failed to actually kill the session
# (see :func:`_check_stop_verification`). Same staleness-filter contract as
# the others.
_SESSION_RECONCILE_STOP_FAILED_NOTE_SENTINEL = (
    "[autonomous_session_watch:session-reconcile-stop-failed]"
)

# Substring stamped into every campaign-pass marker note (the
# epm:campaign-stalled alert, the stop-then-respawn note, the exhausted
# alert, and the budget-backstop alert — task #586). Same staleness-filter
# contract as the others: the campaign watchdog measures epm:campaign-*
# marker freshness on the very task it posts to, so an unfiltered alert
# would reset the staleness window it reports.
_CAMPAIGN_NOTE_SENTINEL = "[autonomous_session_watch:campaign]"

# Substring stamped into the marker posted when the zombie-wrapper pass stops
# a live Happy session whose process tree has carried NO inner Claude process
# for >= the grace window (the wrapper outlived its Claude: 25 such sessions
# showed as "running" indefinitely on 2026-06-11, all invisible to the
# session-reconcile pass because they had lost their issue mapping). Same
# staleness-filter contract as the others.
_ZOMBIE_WRAPPER_STOP_NOTE_SENTINEL = "[autonomous_session_watch:zombie-wrapper-stop]"

# Substring stamped into the one-time alert the zombie-wrapper pass posts
# instead of stopping, in the EPM_ZOMBIE_WRAPPER_REAP=0 alert-only fallback.
# Same staleness-filter contract as the others.
_ZOMBIE_WRAPPER_ALERT_NOTE_SENTINEL = "[autonomous_session_watch:zombie-wrapper-alert]"

# Substring stamped into the one-time alert posted when a zombie-wrapper stop
# was ACKed by the daemon but the session survived the SIGTERM AND the one
# allowed retry (mirrors the session-reconcile stop-verification contract).
_ZOMBIE_WRAPPER_STOP_FAILED_NOTE_SENTINEL = "[autonomous_session_watch:zombie-wrapper-stop-failed]"

# Substrings stamped into the records the idle-unmapped pass writes when it
# stops / alerts on / fails to stop an unmapped EPS session whose transcript
# has been idle past the reap window (the 2026-06-12 class: 25 unmapped
# sessions idle 19-43h, live inner Claude + ~8 MCP children each, ~23 GB RSS
# total). Unmapped sessions have no task to carry a marker, so these land in
# the fallback events file — registered here anyway for sentinel uniformity.
_IDLE_UNMAPPED_STOP_NOTE_SENTINEL = "[autonomous_session_watch:idle-unmapped-stop]"
_IDLE_UNMAPPED_ALERT_NOTE_SENTINEL = "[autonomous_session_watch:idle-unmapped-alert]"
_IDLE_UNMAPPED_STOP_FAILED_NOTE_SENTINEL = "[autonomous_session_watch:idle-unmapped-stop-failed]"

# Substring stamped into the marker the infra-drain pass posts after
# dispatching an autonomous session for a PM-queue ID (task #633). The #633
# task itself becomes ACTIVE seconds after dispatch — a watcher-authored
# dispatch note must never count as "real progress" for the orphan/stalled
# staleness clocks, or it would mask a subsequent stall of the very session
# it just spawned.
_INFRA_DRAIN_NOTE_SENTINEL = "[autonomous_session_watch:infra-drain-dispatch]"

# All watcher-posted note substrings to exclude from `_latest_progress_ts`.
# Pulled into one frozenset so every pass's filter is uniform: add a new
# watcher-posted marker -> add its sentinel here -> _latest_progress_ts
# transparently excludes it without an extra special case.
_WATCHER_NOTE_SENTINELS: frozenset[str] = frozenset(
    {
        _ALERT_NOTE_SENTINEL,
        _KEEP_RUNNING_NOTE_SENTINEL,
        _FOLLOWUP_NOTE_SENTINEL,
        _STALLED_ALERT_NOTE_SENTINEL,
        _STALLED_RESPAWN_NOTE_SENTINEL,
        _STALLED_EXHAUSTED_NOTE_SENTINEL,
        _VM_DISK_NOTE_SENTINEL,
        _ORPHAN_RESPAWN_NOTE_SENTINEL,
        _ORPHAN_ALERT_NOTE_SENTINEL,
        _FOLLOWUPS_AWAITING_CHILD_NOTE_SENTINEL,
        _FOLLOWUP_ROUND_REPARK_NOTE_SENTINEL,
        _SESSION_RECONCILE_ALERT_NOTE_SENTINEL,
        _SESSION_RECONCILE_STOP_NOTE_SENTINEL,
        _SESSION_RECONCILE_STOP_FAILED_NOTE_SENTINEL,
        _CAMPAIGN_NOTE_SENTINEL,
        _ZOMBIE_WRAPPER_STOP_NOTE_SENTINEL,
        _ZOMBIE_WRAPPER_ALERT_NOTE_SENTINEL,
        _ZOMBIE_WRAPPER_STOP_FAILED_NOTE_SENTINEL,
        _IDLE_UNMAPPED_STOP_NOTE_SENTINEL,
        _IDLE_UNMAPPED_ALERT_NOTE_SENTINEL,
        _IDLE_UNMAPPED_STOP_FAILED_NOTE_SENTINEL,
        _INFRA_DRAIN_NOTE_SENTINEL,
    }
)

# Age backstop: drop a pod-safety state file older than this even when the
# RunPod API is flaky and a pod doesn't show up in the current running set on a
# given tick. Without it, an API outage during the exact tick when a pod
# disappears would strand the state file indefinitely. The cap is generous (well
# past any plausible legitimate miss-accumulation window of 2 ticks ≈ 20 min)
# so it only catches genuinely orphaned files, never live state.
POD_SAFETY_STATE_MAX_AGE_S = 7 * 24 * 3600

# ─── alive-but-stalled detector (ALERT + AUTO-RESPAWN) ─────────────────────
#
# Targets a different failure mode than the respawn pass: a session whose
# Happy id IS in the live set (so the respawn pass won't touch it) but whose
# bg-Bash chain quietly died and is no longer self-reporting / posting
# markers / advancing the pod.
#
# Two-phase rollout. Phase 1 (2026-06-05) shipped ALERT-ONLY so we could
# observe real-world detection in production without risking a wrong respawn.
# Phase 2 (2026-06-08, this revision) promotes the action to AUTO-RESPAWN
# (stop-then-respawn) on the strict subset of cases where it is unambiguously
# safe:
#
#   (a) the task is in an :data:`ACTIVE` status (a `proposed` / `clarifying`
#       / `plan_pending` / `blocked` / `awaiting_promotion` etc. is a gate
#       or human-driven park — restarting would interrupt the user's loop);
#   (b) the Happy daemon is reachable (the respawn issues
#       `spawn_session.py stop` and `spawn-issue --auto`, both of which need
#       the daemon — without it we'd leave a half-stopped session); AND
#   (c) we have NOT already auto-respawned this same staleness episode
#       :data:`STALLED_MAX_RESPAWNS` times without ever seeing real
#       progress in between (crash-loop cap — a deterministically-broken
#       session must not loop forever and burn pods).
#
# If any of (a)/(b)/(c) fails, the pass degrades to ALERT-ONLY: post the
# one-time stale-alert marker (or, when the cap is exhausted, the louder
# one-time exhausted marker) and leave it for the user.

# How long a self-report timestamp (and the marker-progress / pod-activity
# signals) may stay frozen before the stalled-detector trips. Conservative:
# generous enough that a long healthy bg op (training launch, eval) doesn't
# false-alert — a true bg-Bash death freezes ALL three signals indefinitely,
# so 45 min is plenty of margin.
STALLED_WINDOW_S = 45 * 60

# Filename prefix for the per-session stalled-detector state file at
# ``~/.eps-autonomous/stalled-<N>.json``. Mirrors the pod-safety state file
# layout — separate per-issue state so a new alert episode can't accidentally
# inherit stale fields from the prior one.
STALLED_STATE_PREFIX = "stalled-"

# Age backstop for stalled-detector state files: reuse the same conservative
# 7-day cap as the pod-safety state store so the orphan-state GC has one
# uniform aging rule across all watcher-owned per-issue state.
STALLED_STATE_MAX_AGE_S = POD_SAFETY_STATE_MAX_AGE_S

# Maximum auto-respawns the stalled-detector will issue within a single
# staleness episode (i.e. before any real progress marker advances). 3 was
# chosen so a transient daemon/Happy-side hiccup that needs a few attempts
# can still self-heal, while a deterministically broken session (the bg-chain
# dies immediately on every restart) bottoms out within ~hours rather than
# burning pods indefinitely. The counter resets to 0 on each real-progress
# advance (mirrors the existing alerted-flag clear logic). After exhaustion
# the pass falls back to a one-time loud marker + leaves it for the user.
STALLED_MAX_RESPAWNS = 3


def decide_session_stalled(
    self_report_age_s: float | None,
    marker_progress_age_s: float | None,
    has_pod: bool,
    missed: int,
    alerted: bool,
    *,
    respawn_eligible: bool = False,
    respawn_count: int = 0,
    threshold: int = 2,
    window_s: float = STALLED_WINDOW_S,
    max_respawns: int = STALLED_MAX_RESPAWNS,
) -> tuple[str, int]:
    """Pure decision for the alive-but-stalled detector.

    Phase 2 (2026-06-08): the action set is ``"respawn"`` | ``"alert"`` |
    ``"exhausted"`` | ``"keep"``. The detection-side trigger (BOTH self-
    report and marker-progress stale, with the 2-miss guard) is unchanged;
    what changed is the RECOVERY action.

    The respawn pass already handles DEAD sessions (Happy id not in the
    live set); this pass handles the harder "alive but bg-Bash chain
    dead" case where the session looks healthy to the respawn pass.

    Trigger requires ALL relevant signals to be stale (corroboration,
    per reviewer MAJOR-3/6: never trigger on transcript-ts alone):

    1. ``self_report_age_s`` — the per-issue self-report file's age in
       seconds. A MISSING file (``None``) is NOT treated as stale here
       (a session that has never self-reported — e.g. a bare manual
       session that was never driven — is skipped; the caller decides
       which registries this pass applies to). Only an EXISTING but
       frozen self-report counts.
    2. ``marker_progress_age_s`` — age of the newest real (non-watcher)
       progress marker on the task's ``events.jsonl``. ``None`` means the
       task has no progress markers at all — that IS a stale signal (a
       pod-active autonomous session that's never posted progress is
       suspicious). The caller filters watcher-posted alerts via
       :data:`_WATCHER_NOTE_SENTINELS`.
    3. ``has_pod`` — whether the issue currently has a RUNNING managed
       pod. If True, the pod's progress is folded into signal 2 (the
       same ``epm:progress`` markers track pod state, posted by
       ``poll_pipeline.py``), so signal 3 devolves to signal 2 for
       managed pods. If False, the pod signal is "skip" — it cannot be
       stale because it does not exist. This keeps the contract simple:
       the caller passes ``has_pod`` for logging only; the decision
       depends on signals 1 and 2 plus the 2-miss guard.

    Apply the 2-miss guard from :func:`decide_pod_safety` to absorb a
    flaky markers-fetch / self-report-race: an action fires only on the
    SECOND consecutive stale check.

    Recovery selection (only when stale + threshold met):

    - ``respawn_eligible=True`` AND ``respawn_count < max_respawns``
      -> ``("respawn", 0)``. The caller has already confirmed the task
      is in :data:`ACTIVE` and the Happy daemon is reachable; this
      function does not re-check (keeps the function pure). The
      ``respawn_count`` carries forward across ticks within one episode
      and is reset by the caller when real progress advances.
    - ``respawn_eligible=True`` AND ``respawn_count >= max_respawns``
      -> ``("exhausted", 0)``. The crash-loop cap has been hit;
      the caller posts a one-time loud exhausted marker and leaves it
      for the user.
    - ``respawn_eligible=False`` (any of: non-ACTIVE status, daemon
      unreachable, or the caller deliberately chose to alert-only)
      -> ``("alert", 0)``. Preserves the Phase-1 ALERT-ONLY behavior
      as the safe fallback.

    Dedup semantics — ``alerted`` dedups REPEAT ALERTS only, it never
    gates off the stronger respawn action. An already-alerted episode
    MUST still escalate to a respawn the moment it becomes eligible.
    (Incident #506, 2026-06-08: a Phase-1 alert set ``alerted=True``
    ~11h before the Phase-2 auto-respawn machinery deployed; the prior
    blanket ``if alerted: return keep`` short-circuit then suppressed
    the respawn on every subsequent tick for 10+ hours while the 8xH200
    pod idle-burned ~$460. The same gap fires any time the FIRST
    threshold-trip lands while respawn is briefly ineligible — daemon
    momentarily down, task momentarily in a non-ACTIVE status — and
    then respawn becomes eligible later in the same episode.) The
    ``alerted`` flag is cleared by the caller when (a) the self-report
    ts advances, or (b) :func:`_handle_stalled_respawn` runs.

    Returns ``(action, new_missed)``. Cases:

    - ``self_report_age_s is None`` (no self-report at all)
      -> ``("keep", 0)``. This pass targets autonomous sessions that
      always self-report; a missing file is the caller's signal to skip.
    - Self-report fresh (< ``window_s``) -> ``("keep", 0)``. Reset miss
      counter; live session.
    - Marker-progress is fresh -> ``("keep", 0)``. Any fresh signal
      resets the miss counter.
    - Self-report stale AND marker-progress also stale (or absent) AND
      ``alerted=True`` AND respawn is now eligible (``respawn_eligible``
      AND ``respawn_count < max_respawns``) -> ``("respawn", 0)``.
      Escalate from alert to respawn; the prior alert already required
      ``>= threshold`` consecutive stale checks, so escalation needn't
      re-accumulate the miss guard. Cleared `alerted` is the caller's
      job on the next ``_save_stalled_state``.
    - Self-report stale AND marker-progress also stale (or absent) AND
      ``alerted=True`` AND respawn is NOT eligible (or cap exhausted)
      -> ``("keep", 0)``. Dedup the repeat alert / hold for exhausted
      marker dedup (the caller's ``exhausted`` flag handles that).
    - Self-report stale AND marker-progress also stale (or absent) AND
      not previously ``alerted`` -> increment ``missed``; on reaching
      ``threshold``, return the appropriate recovery action per the
      table above. Below threshold, return ``("keep", new_missed)``.
    """
    if self_report_age_s is None:
        # Missing self-report -> caller should skip (interactive session,
        # or this pass doesn't apply). Never alert.
        return ("keep", 0)
    if self_report_age_s < window_s:
        # Self-report still advancing -> session is alive; reset.
        return ("keep", 0)
    # Self-report is stale. Require marker-progress to ALSO be stale (or
    # absent) before considering an alert. A fresh marker means the bg
    # chain is still posting; the self-report might just be late.
    marker_stale = marker_progress_age_s is None or marker_progress_age_s >= window_s
    # has_pod is informational at this layer — see the docstring's signal 3.
    _ = has_pod
    if not marker_stale:
        return ("keep", 0)
    if alerted:
        # Already-alerted episode. Dedup the repeat alert, BUT still
        # escalate to a respawn the moment it becomes eligible — the
        # alert flag must never block the stronger action. See the
        # "Dedup semantics" docstring paragraph for the incident that
        # motivates this branch (regression: previously bare
        # ``return ("keep", 0)`` here suppressed all escalation).
        if respawn_eligible and respawn_count < max_respawns:
            return ("respawn", 0)
        # Either respawn not eligible this tick (non-ACTIVE / daemon
        # down) or the crash-loop cap is exhausted. Stay quiet; the
        # caller's ``exhausted`` flag dedups the loud one-time exhausted
        # marker separately, and the next eligibility flip will retry.
        return ("keep", 0)
    new_missed = missed + 1
    if new_missed >= threshold:
        # Threshold met. Pick the recovery action based on eligibility +
        # the crash-loop cap; the caller has already done the I/O-side
        # checks (ACTIVE status + daemon reachability) before passing
        # respawn_eligible.
        if respawn_eligible:
            if respawn_count >= max_respawns:
                return ("exhausted", 0)
            return ("respawn", 0)
        return ("alert", 0)
    return ("keep", new_missed)


def decide_pod_safety(
    status_class: str,
    missed: int,
    stale: bool,
    alerted: bool,
    threshold: int = 2,
    *,
    keep_running: bool = False,
    followup_active: bool = False,
) -> tuple[str, int]:
    """Pure decision for the pod-safety pass on a RUNNING managed pod.

    Trigger is the task's STATUS CLASS (unambiguous), NOT session liveness —
    see the module docstring "Why STOP is keyed on task status". Returns
    ``(action, new_missed)`` where action is ``"stop"`` | ``"alert"`` |
    ``"keep"`` | ``"keep-running-skip"`` | ``"followup-skip"``.

    Parameters
    ----------
    status_class
        ``"auto-stop-done"`` — task in :data:`AUTO_STOP_DONE` (provably
        finished); ``"pod-active-stale"`` — task in :data:`POD_ACTIVE` AND no
        real marker progress for > :data:`ALERT_STALE_HOURS`;
        ``"pod-active-fresh"`` — task in :data:`POD_ACTIVE` with recent
        progress; ``"other"`` — anything else (e.g. ``blocked``, an unknown
        status). ``stale`` is folded into
        ``status_class`` by the caller and kept as a redundant explicit param
        for callers/tests that want to pass it directly.
    missed
        Consecutive-miss count for the auto-stop arm (mirrors :func:`decide`).
    stale
        Whether the task has gone stale (no real progress > threshold). Only
        meaningful when ``status_class`` is pod-active; the caller derives
        ``status_class == "pod-active-stale"`` from it, so this is informational
        for the pod-active path.
    alerted
        Whether a stale-alert has ALREADY been posted for the current episode
        (tracked in the state file). Dedups the alert so it fires once per
        episode, not every 10-min tick.
    keep_running
        Whether the task carries the ``keep-running`` tag (the Step-8
        auto-terminate exemption). Consulted ONLY on the auto-stop arm: a
        DONE task's RUNNING pod with the tag returns
        ``("keep-running-skip", 0)`` instead of accumulating toward a stop.
        The alert arm ignores it (alerts never stop anything). Takes
        precedence over ``followup_active`` (an explicit user-set tag beats
        an inferred follow-up signal).
    followup_active
        Whether the task's events.jsonl shows an ``epm:run-launched`` marker
        NEWER than its transition into the current DONE status — i.e. a
        legitimate user-approved inline follow-up has provisioned a fresh
        pod on a promoted/completed/awaiting_promotion/archived parent (the
        CLAUDE.md "Routing experiment intent → Follow-up" path). Consulted
        ONLY on the auto-stop arm, only when ``keep_running`` is False: a
        DONE task's RUNNING pod with an active follow-up returns
        ``("followup-skip", 0)`` instead of accumulating toward a stop. The
        caller computes this lazily from ``_task_events`` so the extra
        events fetch is paid only for escaped-pod candidates (same lazy
        pattern as ``keep_running``). Incident #477 (2026-06-10): the
        watcher stopped a healthy follow-up pod 3 times before the user
        manually added the ``keep-running`` tag.

    Cases:

    - ``status_class == "auto-stop-done"`` AND ``keep_running`` ->
      ``("keep-running-skip", 0)``. The stop is SKIPPED and the miss counter
      reset, so removing the tag later re-arms a fresh >=``threshold``-checks
      accumulation before any stop. The caller logs the skip + posts a
      once-per-pod-incarnation marker.
    - ``status_class == "auto-stop-done"`` AND ``followup_active`` (and not
      ``keep_running``) -> ``("followup-skip", 0)``. Same SKIP-and-reset
      semantics as ``keep-running-skip``; the caller posts a
      once-per-pod-incarnation follow-up exemption marker. If the follow-up
      later finishes (the next ``epm:status-changed`` / ``epm:promoted``
      lands AFTER the latest ``epm:run-launched``) the predicate flips
      False on the next tick and the auto-stop re-arms normally.
    - ``status_class == "auto-stop-done"`` -> increment ``missed``; return
      ``"stop"`` once it reaches ``threshold`` (default 2 = ~20 min at a 10-min
      cron, so a single transient API/status glitch never stops a pod), else
      ``("keep", new_missed)``. STOP is reversible (``pod.py stop`` preserves
      the volume) — NEVER a terminate.
    - ``status_class == "pod-active-stale"`` AND not ``alerted`` ->
      ``("alert", 0)``. Loud log + one-time marker. NEVER a stop.
    - ``status_class == "pod-active-stale"`` AND ``alerted`` -> ``("keep", 0)``.
      Already alerted this episode; stay quiet.
    - any other case (``pod-active-fresh``, ``other``) -> ``("keep", 0)``.
      Reset the auto-stop miss counter (the pod is legitimately in use or the
      status is one we deliberately never auto-stop).
    """
    if status_class == "auto-stop-done":
        if keep_running:
            return ("keep-running-skip", 0)
        if followup_active:
            return ("followup-skip", 0)
        new_missed = missed + 1
        if new_missed >= threshold:
            return ("stop", 0)
        return ("keep", new_missed)
    if status_class == "pod-active-stale" and not alerted:
        return ("alert", 0)
    # pod-active-stale-already-alerted, pod-active-fresh, other -> hands off.
    return ("keep", 0)


# ─── VM disk-headroom watcher (task #552 incident, 2026-06-10) ───────────────
#
# Pods have disk guards (pod_disk_guard.py, the preflight fallocate probe) but
# the VM — which hosts every orchestrator session, the worktree .venvs (~11G
# each), the uv cache, and the HF cache — had none. When / hit 100%
# (482G/485G) every foreground Bash spawn in the orchestrator session failed
# silently (exit 1, zero output) and the /issue 552 interpretation loop
# stalled ~20 min, undiagnosable from inside the session. This pass alerts
# BEFORE that point and reclaims the safe, regenerable space when critically
# low.

# Filesystem whose headroom is watched (the VM root — NOT a pod path; pod-side
# guards are out of scope here and live in pod_disk_guard.py / preflight).
VM_DISK_PATH = "/"

# Below this free-bytes threshold the pass alerts: loud log every tick + ONE
# dashboard-visible marker per low-disk episode. ~20 GiB leaves enough slack
# to keep sessions alive while a human (or the reclaim arm) frees space.
VM_DISK_ALERT_FREE_BYTES = 20 * 2**30


def _env_gib_bytes(name: str, default_gib: float) -> int:
    """GiB-denominated env knob -> bytes. A garbled / non-positive value falls
    back to the default rather than crashing the watcher at import (same
    fail-soft contract as the other env knobs in this file)."""
    try:
        val = float(os.environ.get(name, ""))
    except ValueError:
        return int(default_gib * 2**30)
    # The sanity bound also rejects inf/nan (int(inf * 2**30) would raise —
    # crashing the watcher at import is exactly what fail-soft must prevent).
    if not (0 < val < 2**20):
        return int(default_gib * 2**30)
    return int(val * 2**30)


# Below this free-bytes threshold the pass ALSO runs the safe cache reclaims
# (`uv cache prune`, `npm cache clean`, stale /tmp/claude-* sweep). ~15 GiB
# (was 8) because the 2026-06-11 episode fell 17 GiB -> 1.2 GiB within hours —
# waiting until 8 GiB to reclaim regenerable caches loses the race to the
# silently-failing-Bash-spawn regime. Override: EPM_VM_DISK_CRITICAL_GIB.
# NOTE: an override ABOVE the ~20 GiB alert threshold is effectively clamped
# to it — free >= VM_DISK_ALERT_FREE_BYTES early-returns "ok" before the
# critical comparison ever runs.
VM_DISK_RECLAIM_FREE_BYTES = _env_gib_bytes("EPM_VM_DISK_CRITICAL_GIB", 15)

# Hysteresis margin on episode CLEAR: the episode state (alert dedup +
# remediation re-arm timestamps) is dropped only once free space recovers
# DECISIVELY — at or above alert + this margin (~22 GiB total). Clearing
# exactly at the alert threshold made free space oscillating around the
# 20 GiB boundary start a "fresh episode" on every dip, re-firing the
# worktree audit (and the once-per-episode alert) each time — defeating the
# 6h re-arm window in exactly the flapping case it exists for. Recovery
# inside the band (alert <= free < alert + margin) keeps the state file; a
# decisive recovery followed by a fresh dip IS a new episode (a new disk
# consumer), so re-running the audit there is correct, not churn.
# Override: EPM_VM_DISK_CLEAR_HYSTERESIS_GIB.
VM_DISK_CLEAR_HYSTERESIS_BYTES = _env_gib_bytes("EPM_VM_DISK_CLEAR_HYSTERESIS_GIB", 2)

# Re-arm window for the remediation arms within ONE low-disk episode: don't
# re-run the worktree audit (low+) or the cache reclaims + tmp sweep
# (critical) more than once per this many seconds (the first run reclaims
# nearly everything reclaimable; hot-looping every 10-min tick would just
# churn). A long episode re-fires after the window — which also catches a
# worktree whose holder process died AFTER the first audit. Tracked via
# `last_reclaim_ts` / `last_audit_ts` in the vm-disk state file.
VM_DISK_RECLAIM_REARM_S = 6 * 3600

# A /tmp/claude-* tree is swept only when NOTHING in it (the dir itself or any
# file under it) was modified within this window. A live session writes its
# /tmp/claude-<port>/.../tasks/*.output files continuously, so its tree always
# has fresh mtimes — the age test IS the live-session guard.
VM_DISK_TMP_SWEEP_AGE_S = 3 * 24 * 3600

# Hard wall-clock bound on `uv cache prune` / `npm cache clean` / the wandb
# artifact-cache cleanup: if another process holds the cache lock the command
# blocks; kill it at the bound (fail-soft) rather than hanging the watcher
# tick. 27 live sessions hold the uv cache lock almost continuously, so lock
# contention is the EXPECTED case and a timeout is a clean skip, never an
# error (2026-06-11: a manual 300s wait timed out). NEVER pass --force to uv
# cache operations while sessions are live.
VM_DISK_UV_PRUNE_TIMEOUT_S = 300

# Target size handed to `wandb artifact cache cleanup` by the critical
# reclaim arm. The artifact cache (~/.cache/wandb/artifacts) is a pure
# content-addressed DOWNLOAD cache — wandb re-fetches on demand — so pruning
# it to ~1GB is zero-risk; the 2026-06-11 episode had 17.6 GB sitting there
# while / fell to 7.3 GiB, reclaimed in ~2 min by the manual run.
VM_DISK_WANDB_CACHE_TARGET = "1GB"


def _env_days_seconds(name: str, default_days: float) -> float:
    """Days-denominated env knob -> seconds. Garbled / non-positive values
    fall back to the default (same fail-soft contract as
    :func:`_env_gib_bytes` — never crash the watcher at import)."""
    try:
        val = float(os.environ.get(name, ""))
    except ValueError:
        return default_days * 86400.0
    if not (0 < val < 36500):
        return default_days * 86400.0
    return val * 86400.0


# Conservative TTL for the HF hub cache eviction (2026-06-11 episode: 41.5 GB
# VM-side hub cache, untouched by any reclaim). A cached revision is evicted
# only when it was last MODIFIED more than this long ago, was last READ
# (newest blob atime across its files) more than this long ago, AND it is
# either detached (no refs — a superseded or sha-pinned snapshot) or its
# whole repo has not been ACCESSED within the window. Repos touched recently
# (e.g. the explore-persona-space-data dataset re-downloaded by interpreting
# sessions) keep every ref'd revision; an in-flight download has a fresh
# last_modified and a sha-pinned adapter that is actively read has fresh
# blob atimes, so neither is ever evicted. Override: EPM_VM_DISK_HF_TTL_DAYS.
VM_DISK_HF_TTL_S = _env_days_seconds("EPM_VM_DISK_HF_TTL_DAYS", 14)

# Hard wall-clock bound on the in-process HF hub scan + eviction
# (scan_cache_dir() walks the whole multi-GB cache tree; delete_revisions
# can unlink thousands of blobs). Every other remediation step is a
# subprocess bounded by timeout= (300s caches / 900s audit); this one runs
# in-process, so the bound is a daemon-thread join — see
# _vm_reclaim_hf_hub_cache for why concurrent.futures cannot deliver it.
VM_DISK_HF_RECLAIM_TIMEOUT_S = 600

# Per-step freed-space deltas below this are statvfs noise from concurrent
# writers (~1 GiB/h background growth) — don't annotate them in the note.
VM_DISK_FREED_NOTE_MIN_BYTES = 2**27  # 128 MiB

# Hard wall-clock bound on `worktree_audit.py --apply` (git operations +
# rescue copies over up to ~dozens of worktrees). The watcher is single-flight
# (flock in main), so a slow audit just makes the next cron fire skip — it
# can't pile up overlapping watcher runs.
VM_DISK_WORKTREE_AUDIT_TIMEOUT_S = 900


def decide_vm_disk(
    free_bytes: int,
    *,
    alerted: bool,
    last_reclaim_ts: float | None,
    last_audit_ts: float | None,
    now: float,
) -> tuple[str, bool, bool, bool]:
    """Pure decision for the VM disk-headroom pass.

    Returns ``(level, do_alert, do_reclaim, do_audit)``:

    - ``level`` — ``"ok"`` (>= :data:`VM_DISK_ALERT_FREE_BYTES` free),
      ``"low"`` (below the alert threshold), or ``"critical"`` (below
      :data:`VM_DISK_RECLAIM_FREE_BYTES`).
    - ``do_alert`` — fire the once-per-episode alert (level is low or
      critical AND ``alerted`` is not already set for this episode).
    - ``do_reclaim`` — run the safe cache reclaims (level is critical AND the
      reclaim arm hasn't fired within :data:`VM_DISK_RECLAIM_REARM_S`).
    - ``do_audit`` — run the stale-worktree sweep (level is low OR critical —
      the audit is the remediation that frees the big space, so it fires at
      the ADVISORY threshold, not only at critical — AND the audit arm hasn't
      fired within :data:`VM_DISK_RECLAIM_REARM_S`).
    """
    if free_bytes >= VM_DISK_ALERT_FREE_BYTES:
        return ("ok", False, False, False)
    level = "critical" if free_bytes < VM_DISK_RECLAIM_FREE_BYTES else "low"
    do_alert = not alerted
    do_reclaim = level == "critical" and (
        last_reclaim_ts is None or now - last_reclaim_ts >= VM_DISK_RECLAIM_REARM_S
    )
    do_audit = last_audit_ts is None or now - last_audit_ts >= VM_DISK_RECLAIM_REARM_S
    return (level, do_alert, do_reclaim, do_audit)


def _status_class(status: str | None, latest_progress_ts: float | None, now: float) -> str:
    """Classify a RUNNING managed pod's task status for :func:`decide_pod_safety`.

    Returns ``"auto-stop-done"`` / ``"pod-active-stale"`` / ``"pod-active-fresh"``
    / ``"other"``. ``status`` of ``None`` (task unreadable) is ``"other"`` —
    never auto-stopped. A pod-active task is ``stale`` when its newest real
    progress marker is older than :data:`ALERT_STALE_HOURS`, OR when there is no
    real progress marker at all (``latest_progress_ts is None``) — a pod-active
    task with zero progress markers is itself a signal worth alerting on.
    """
    if status is None:
        return "other"
    if status in AUTO_STOP_DONE:
        return "auto-stop-done"
    if status in POD_ACTIVE:
        if latest_progress_ts is None:
            return "pod-active-stale"
        if (now - latest_progress_ts) > ALERT_STALE_HOURS * 3600:
            return "pod-active-stale"
        return "pod-active-fresh"
    return "other"


# Progress-ish marker kinds that count as "the experiment made real progress."
# Deliberately broad: any of these advancing means the run is alive. The
# watcher's own alert posts use `epm:progress` too, so they are filtered out by
# the _ALERT_NOTE_SENTINEL note check in _latest_progress_ts (NOT by kind).
_PROGRESS_KINDS = {
    "epm:progress",
    "epm:hot-fix",
    "epm:run-finished",
    "epm:results",
    "epm:status-changed",
    "epm:upload-verification",
    "epm:upload-verified",
    "epm:upload-fix",
    "epm:interpretation",
}


def _parse_event_ts(ts: str | None) -> float | None:
    """Parse a task event ``ts`` (``%Y-%m-%dT%H:%M:%SZ``, UTC) to an epoch
    float, or ``None`` if absent/unparseable."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        # The canonical format is a trailing 'Z' (UTC). fromisoformat handles
        # '+00:00' but not 'Z' on older pythons, so normalise.
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, OSError):
        return None


def _latest_progress_ts(events: list[dict]) -> float | None:
    """Newest epoch timestamp among REAL progress markers in ``events``.

    "Real progress" = an event whose ``kind`` is in :data:`_PROGRESS_KINDS`
    AND whose ``note`` does NOT contain ANY substring in
    :data:`_WATCHER_NOTE_SENTINELS` (the watcher's own stale-alert /
    session-stalled-alert posts use ``epm:progress`` and must NOT count as
    progress — otherwise the alert would reset the staleness clock it is
    measuring). Returns ``None`` when there is no such marker.
    """
    best: float | None = None
    for ev in events:
        if ev.get("kind") not in _PROGRESS_KINDS:
            continue
        note = ev.get("note") or ""
        if any(sentinel in note for sentinel in _WATCHER_NOTE_SENTINELS):
            continue  # a watcher-posted alert — not real progress
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def _task_status(issue: int) -> str | None:
    """Current status of task ``issue`` via `task.py view --json`, or ``None``
    if the task no longer exists / cannot be read."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "view", str(issue), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    status = data.get("status") or (data.get("frontmatter") or {}).get("status")
    return status if isinstance(status, str) else None


def _task_keep_running(issue: int) -> bool:
    """True iff task ``issue`` currently carries the ``keep-running`` tag.

    The Step-8 auto-terminate exemption tag, consulted by the pod-safety
    auto-stop arm (see the module docstring's keep-running coverage note).
    Same subprocess isolation as :func:`_task_status`; any read failure
    returns False (no exemption observed) — the auto-stop then proceeds only
    if the no-tag observation persists across the >=2-checks miss guard, so a
    single transient ``task.py`` glitch never stops a tagged pod. Called
    LAZILY by :func:`_process_pod` only on the auto-stop-done branch, so the
    extra ``task.py view`` subprocess is paid only for escaped-pod
    candidates, not for every RUNNING pod every tick."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "view", str(issue), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    if out.returncode != 0:
        return False
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return False
    tags = (data.get("frontmatter") or {}).get("tags") or []
    return isinstance(tags, list) and "keep-running" in tags


# Marker kinds that record a transition INTO a DONE status. The latest ts
# among these is "when did this task become DONE"; compared against the
# latest `epm:run-launched` ts to decide whether an `epm:run-launched`
# represents a legitimate inline follow-up (i.e. it landed AFTER the task
# was promoted/completed, not before).
#
# `epm:promoted` is emitted by `task.py promote`; `epm:status-changed` is
# the generic transition marker (caller has already verified the CURRENT
# status is DONE, so the latest `epm:status-changed` ts is by definition
# the transition INTO the current DONE status — note text is not parsed).
_DONE_TRANSITION_KINDS = frozenset({"epm:promoted", "epm:status-changed"})

# Marker kinds that signal a live inline follow-up to the POD-SAFETY pass.
# The pod-safety pass treats any of these whose ts is NEWER than the latest
# done-transition as a live inline follow-up and SKIPS the auto-stop (see
# `decide_pod_safety`'s `followup_active` parameter).
#
# Originally only `epm:run-launched` (the #477-validated signal). Widened
# 2026-06-10 (refs #573) to cover USER-CHAT inline follow-ups: the CLAUDE.md
# "Routing experiment intent → Follow-up" path posts `epm:followup-scope v1`
# on #N BEFORE re-invoking /issue, so there is a window — scope posted, pod
# provisioned, run not yet launched — where the old run-launched-only
# inference auto-stopped a healthy follow-up pod (pod-530/531 stopped 8x on
# the :13/:33/:53 grid, pod-477 3x, 2026-06-09). `epm:free-analysis-followup-
# run` is included for parity with the session-reconcile twin
# (:data:`_SESSION_FOLLOWUP_SIGNAL_KINDS`); the two sets are now identical on
# the follow-up side, differing only in their done-transition sets.
_POD_FOLLOWUP_SIGNAL_KINDS = frozenset(
    {
        "epm:run-launched",
        "epm:followup-scope",
        "epm:free-analysis-followup-run",
    }
)
# Back-compat alias (the run-launched marker is still the strongest signal).
_RUN_LAUNCHED_KIND = "epm:run-launched"


def _latest_event_ts(events: list[dict], kinds: frozenset[str] | set[str]) -> float | None:
    """Newest epoch ts among events whose ``kind`` is in ``kinds``, or
    ``None`` if no such event exists. Watcher-posted notes are NOT excluded
    here (this is a generic ts helper; the caller decides whether a sentinel
    filter applies). Used to compare an inline-follow-up's
    ``epm:run-launched`` ts vs the task's latest done-transition ts."""
    best: float | None = None
    if isinstance(kinds, set):
        kinds = frozenset(kinds)
    for ev in events:
        if ev.get("kind") not in kinds:
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def _task_followup_active(issue: int, events: list[dict] | None = None) -> bool:
    """True iff task ``issue`` has a follow-up signal marker
    (:data:`_POD_FOLLOWUP_SIGNAL_KINDS`: ``epm:run-launched`` /
    ``epm:followup-scope`` / ``epm:free-analysis-followup-run``) NEWER than
    its latest done-transition marker (``epm:promoted`` /
    ``epm:status-changed``).

    Predicate for the pod-safety auto-stop exemption: a DONE-status task
    with a fresh follow-up signal carries an in-flight, user-approved
    inline follow-up (CLAUDE.md "Routing experiment intent → Follow-up") so
    the pod is legitimately in use. ``epm:followup-scope`` covers the
    USER-CHAT inline case where the scope is posted before the run launches
    (refs #573 — the run-launched-only inference stopped healthy follow-up
    pods 11x on 2026-06-09). When the follow-up completes, the next
    ``epm:status-changed`` / ``epm:promoted`` event will land newer than
    the follow-up signal and this predicate flips False — the auto-stop
    re-arms naturally on the following tick (same semantics as the
    ``keep-running`` tag being removed).

    Called LAZILY by :func:`_process_pod` only on the auto-stop-done branch,
    so the per-task events fetch is paid only for escaped-pod candidates,
    not for every RUNNING pod every tick. ``events`` may be passed in by
    the caller to avoid double-fetching when the events list is already
    loaded (the typical _process_pod path).

    A missing follow-up signal returns False (no exemption).
    A missing done-transition is impossible in practice — the caller
    already verified the task's current status is DONE, so at least one
    ``epm:status-changed`` must have fired to put it there. If the read
    nonetheless returns no done-transition (defensive), we conservatively
    return False (no exemption) rather than skip the auto-stop on a
    potentially-stale read.
    """
    if events is None:
        events = _task_events(issue)
    followup_signal = _latest_event_ts(events, _POD_FOLLOWUP_SIGNAL_KINDS)
    if followup_signal is None:
        return False
    done_transition = _latest_event_ts(events, _DONE_TRANSITION_KINDS)
    if done_transition is None:
        return False
    return followup_signal > done_transition


def _task_events(issue: int) -> list[dict]:
    """All events on task ``issue`` via `task.py list-markers --json`, or ``[]``
    if the task can't be read. Subprocess-isolated (same pattern as
    :func:`_task_status`) so a branch-guard / missing-task error degrades to an
    empty list rather than crashing the pass."""
    try:
        out = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "list-markers",
                str(issue),
                "--prefix",
                "epm:",
                "--json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _daemon_reachable() -> bool:
    """True iff the Happy daemon's control server answers /list.

    Critical guard for the RESPAWN pass only: ``_live_session_ids()`` returns an
    empty set BOTH when the daemon is up with zero sessions AND when it is
    unreachable. Without distinguishing them, a daemon outage would make every
    recorded session look dead and trigger a mass re-spawn (-> duplicate pods).
    So the respawn pass probes reachability first and skips when the daemon is
    down. The pod-safety pass does NOT depend on the daemon (it reasons about
    task status + the live pod list), so it runs regardless.

    The exception tuple widens for the round-4 fix on top of the obvious
    connection-level URLError / OSError tier: ``http.client.HTTPException``
    (incl. ``IncompleteRead`` when the daemon hangs up mid-response-body —
    a class distinct from URLError, which fires at connection setup) and
    ``UnicodeDecodeError`` (a daemon emitting invalid UTF-8 bytes raises
    that BEFORE ``json.loads`` ever sees a string; ``json.JSONDecodeError``
    is a ``ValueError`` subclass, NOT a ``UnicodeDecodeError`` subclass).
    A daemon flap that previously crashed the whole watcher pass now
    correctly returns ``False`` — the conservative ack of "I cannot tell
    whether the daemon is up", same as a clean ``URLError``."""
    try:
        import http.client
        import urllib.error
        import urllib.request

        from spawn_session import daemon_port

        url = f"http://127.0.0.1:{daemon_port()}/list"
        req = urllib.request.Request(
            url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.loads(resp.read())
        return True
    except (
        SystemExit,
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return False


def _live_session_ids_or_none() -> set[str] | None:
    """``spawn_session._live_session_ids()`` with an explicit UNAVAILABLE
    mode: the daemon's live session-id set, or ``None`` when the ``/list``
    probe fails (daemon down, malformed payload, malformed child entry).

    spawn_session's helper returns an EMPTY SET both for "daemon up, zero
    sessions" and for "daemon unreachable" (its ``list --all`` fallback wants
    that), which is exactly the wrong fail direction for the infra-drain
    stale-registration read: a daemon flap between main()'s single
    reachability probe and this read would make every grace-aged
    still-``proposed`` registration look definitively dead -> false-stale ->
    double-spawn. So the drain pass uses this wrapper (same probe shape as
    :func:`_daemon_reachable`; spawn_session.py is a forbidden surface for
    this pass) and :func:`_infra_drain_stale` fails ``None`` toward NOT
    stale (keep blocking).

    A child dict carrying an invalid ``happySessionId`` (missing, ``None``,
    empty string, non-str) is treated the same as a missing ``children``
    list — both return ``None`` (round-3 fix, reconciler verdict
    2026-06-12: a stray ``{None}`` set would slip past the ``is None``
    guard in :func:`_infra_drain_stale` and make every real-string sid
    look NOT live, reintroducing the round-1 BLOCKER's double-spawn
    class).

    The exception tuple widens for the round-4 fix on top of the obvious
    connection-level URLError / OSError tier:
    ``http.client.HTTPException`` (incl. ``IncompleteRead`` when the
    daemon hangs up mid-response-body — a class distinct from URLError,
    which fires at connection setup) and ``UnicodeDecodeError`` (a
    daemon emitting invalid UTF-8 bytes raises that BEFORE
    ``json.loads`` ever sees a string; ``json.JSONDecodeError`` is a
    ``ValueError`` subclass, NOT a ``UnicodeDecodeError`` subclass). A
    daemon flap that previously crashed the whole infra-drain pass now
    correctly returns ``None`` (UNAVAILABLE), same fail direction as a
    clean ``URLError``."""
    try:
        import http.client
        import urllib.error
        import urllib.request

        from spawn_session import daemon_port

        url = f"http://127.0.0.1:{daemon_port()}/list"
        req = urllib.request.Request(
            url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (
        SystemExit,
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None
    children = data.get("children") if isinstance(data, dict) else None
    if not isinstance(children, list):
        # A 200 response without the expected shape is NOT a confirmed
        # "zero live sessions" — treat as unavailable (fail toward keep).
        return None
    sids: set[str] = set()
    for c in children:
        if not isinstance(c, dict):
            # Non-dict child entries (e.g. the "junk" string in the existing
            # ``test_live_session_ids_or_none_shapes`` happy-path fixture)
            # are skipped; they carry no sid claim either way.
            continue
        sid = c.get("happySessionId")
        if not isinstance(sid, str) or not sid:
            # A dict child whose sid is missing/None/empty/non-str is a
            # daemon-contract violation — fail toward keep-blocking per
            # the function's documented contract, the same fail direction
            # already used three lines above for a missing ``children``
            # list. One bad child contaminates the whole reply (we cannot
            # tell whether the others are real-but-incomplete or merely
            # the well-formed survivors of a partial write).
            return None
        sids.add(sid)
    return sids


def _manual_session_alive(issue: int | None, live_ids: set[str]) -> bool:
    """True iff the issue's MANUAL registration (``manual-issue-<N>.json``,
    written by bare ``spawn-issue``) records a Happy id in the daemon's live
    set. Covers the one legitimate case where the AUTONOMOUS entry's recorded
    id is dead but the issue is still driven: the user/PM opened a manual
    replacement session (which registers the manual entry but does not rewrite
    the autonomous one). Respawning next to that live manual driver would
    duplicate the workflow."""
    if issue is None:
        return False
    path = AUTONOMOUS_REGISTRY_DIR / f"manual-issue-{issue}.json"
    try:
        entry = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    sid = entry.get("happy_session_id")
    return isinstance(sid, str) and sid in live_ids


def _session_alive(entry: dict, live_ids: set[str]) -> bool:
    """A session counts as alive if its recorded Happy id is still tracked by
    the daemon, OR the issue's MANUAL registration records a live id (a
    user/PM replacement session that didn't rewrite the autonomous entry).

    The earlier third signal — "a live session occupies the issue's worktree
    dir" — was REMOVED 2026-06-10: ``spawn-issue --auto`` spawns drivers WITH
    cwd = the issue worktree when it already exists, so every superseded
    driver generation matched the cwd test, and one idle zombie generation
    kept #518 reading ``alive=True`` for ~11h after its registered driver
    died. The registry is rewritten on every respawn, so recorded-id +
    manual-id are the precise signals the cwd heuristic was approximating."""
    if entry.get("happy_session_id") in live_ids:
        return True
    return _manual_session_alive(entry.get("issue"), live_ids)


def _respawn(entry: dict, dry_run: bool) -> bool:
    """Re-spawn the autonomous session for this entry. Returns True on success.
    spawn_session rewrites the registry (new id, missed=0) as a side effect.

    Re-passes the per-session Claude overrides (``model``, ``betas``,
    ``effort``) verbatim when the registry entry recorded them at spawn time —
    they are part of the prompt-cache key, so flipping any of them on respawn
    would force a full uncached re-read of the conversation (CLAUDE.md §
    Context hygiene). Entries that pre-date the override-persistence feature
    simply don't carry these fields, so the respawn inherits the user's global
    Claude Code defaults (matching the pre-feature behavior)."""
    issue = entry["issue"]
    cap = entry.get("auto_approve_gpu_hours", 24.0)
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto", "--auto-approve-gpu-hours", str(cap),
    ]  # fmt: skip
    model = entry.get("model")
    if model:
        cmd.extend(["--model", str(model)])
    betas = entry.get("betas")
    if betas:
        # The spawn CLI takes a comma-separated string and re-parses it.
        cmd.extend(["--betas", ",".join(str(b) for b in betas)])
    effort = entry.get("effort")
    if effort:
        cmd.extend(["--effort", str(effort)])
    if dry_run:
        print(f"  [dry-run] would respawn: {' '.join(cmd)}")
        return False
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(f"  RESPAWN FAILED issue #{issue}: {res.stderr.strip()[:300]}", file=sys.stderr)
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  RESPAWNED issue #{issue} (session was dead): {first_line}")
    return True


def _acquire_lock() -> object | None:
    """Single-flight: hold a non-blocking flock so overlapping cron fires don't
    race (a race could double-spawn -> two pods). Returns the held fd, or None
    if another watcher run holds it (caller should exit cleanly)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    # Held for the whole run (released on process exit) — a context manager
    # would close it and drop the lock, so the bare open is deliberate.
    fd = open(AUTONOMOUS_REGISTRY_DIR / "watch.lock", "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fd.close()
        return None
    return fd


# ─── pod-safety state store ──────────────────────────────────────────────────


def _pod_safety_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{_POD_SAFETY_PREFIX}{issue}.json"


def _load_pod_safety_state(issue: int) -> dict:
    """Read the per-pod state for ``issue`` (``{}`` if absent / unreadable — a
    fresh/garbled file just starts the miss count at 0 and alerted at False)."""
    path = _pod_safety_state_path(issue)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_pod_safety_state(
    issue: int,
    pod_id: str,
    missed: int,
    *,
    alerted: bool,
    last_progress_ts: float | None,
    keep_running_noted: bool | None = None,
    followup_noted: bool | None = None,
    prev: dict | None = None,
) -> None:
    """Persist the per-pod state atomically (temp + rename).

    ``missed`` is the auto-stop consecutive-miss count. ``alerted`` records
    whether a stale-alert was already posted this episode (dedup).
    ``last_progress_ts`` is the newest REAL progress timestamp we observed —
    stored so a later tick can tell "the gap stopped advancing" from "new
    progress arrived" (and reset ``alerted`` when progress advances).
    ``keep_running_noted`` records whether the once-per-pod-incarnation
    keep-running-exemption marker was already posted (dedup, same role as
    ``alerted`` for the keep-running-skip arm); ``None`` (the default)
    carries the prior on-disk value forward so callers that don't touch the
    keep-running path never clobber it. ``followup_noted`` is the same
    dedup flag for the inline-follow-up exemption (``followup-skip``);
    None carries forward identically.  ``prev`` is the existing on-disk
    payload (if any), passed so callers that already loaded it don't re-read;
    ``first_seen`` carries forward when present so the age backstop measures
    the original episode start, not the latest save.
    """
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _pod_safety_state_path(issue)
    prev_first_seen = (prev or {}).get("first_seen")
    if not isinstance(prev_first_seen, int | float):
        prev_first_seen = time.time()
    if keep_running_noted is None:
        keep_running_noted = bool((prev or {}).get("keep_running_noted", False))
    if followup_noted is None:
        followup_noted = bool((prev or {}).get("followup_noted", False))
    payload = {
        "pod_id": pod_id,
        "missed": missed,
        "alerted": alerted,
        "last_progress_ts": last_progress_ts,
        "keep_running_noted": bool(keep_running_noted),
        "followup_noted": bool(followup_noted),
        "first_seen": prev_first_seen,
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


# ─── stalled-detector state store ────────────────────────────────────────────


def _stalled_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{STALLED_STATE_PREFIX}{issue}.json"


def _load_stalled_state(issue: int) -> dict:
    """Read the per-session stalled-detector state for ``issue`` (``{}`` if
    absent / unreadable — a fresh/garbled file just starts the miss count at 0
    and alerted at False, mirroring :func:`_load_pod_safety_state`)."""
    path = _stalled_state_path(issue)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_stalled_state(
    issue: int,
    happy_session_id: str | None,
    missed: int,
    *,
    alerted: bool,
    last_self_report_ts: str | None,
    respawn_count: int = 0,
    exhausted: bool = False,
    refresh_attempted: bool = False,
    followups_child_alerted: bool = False,
    prev: dict | None = None,
) -> None:
    """Persist the per-session stalled-detector state atomically (temp +
    rename), mirroring :func:`_save_pod_safety_state`.

    ``missed`` is the 2-miss-guard count; ``alerted`` records whether a
    session-stalled-alert was posted this episode (dedup);
    ``last_self_report_ts`` is the raw ISO ts from the self-report file the
    LAST time we read it, so the next tick can tell "the self-report
    advanced" from "the self-report is still frozen at the same ts" and
    clear ``alerted`` when the session resumes self-reporting.
    ``respawn_count`` is the number of auto-respawns issued in the current
    staleness episode (capped by :data:`STALLED_MAX_RESPAWNS`); cleared
    by the caller on each real-progress advance, mirroring the
    ``alerted`` flag. ``exhausted`` records whether the one-time
    "auto-recovery exhausted" marker has already been posted this
    episode (dedup, also cleared on progress). ``refresh_attempted``
    records whether the #488 stale-port self-heal (``pod.py config
    --refresh-from-api``) has already fired this episode (dedup, also
    cleared on progress) — one refresh attempt per stalled episode, no
    hot-loop. ``followups_child_alerted`` records whether the one-time
    "followups_running parent waiting on open child" suppression alert
    has been posted this episode (dedup, also cleared on progress) —
    see :func:`_followups_awaiting_child_reason` for the predicate.
    ``prev`` is the prior on-disk payload (when the caller already has
    it loaded) so ``first_seen`` carries forward and the age backstop
    measures the original episode start.
    """
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _stalled_state_path(issue)
    prev_first_seen = (prev or {}).get("first_seen")
    if not isinstance(prev_first_seen, int | float):
        prev_first_seen = time.time()
    payload = {
        "happy_session_id": happy_session_id,
        "missed": missed,
        "alerted": alerted,
        "respawn_count": respawn_count,
        "exhausted": exhausted,
        "refresh_attempted": refresh_attempted,
        "followups_child_alerted": followups_child_alerted,
        "last_self_report_ts": last_self_report_ts,
        "first_seen": prev_first_seen,
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_stalled_state(issue: int) -> None:
    """Drop the per-session stalled-detector state file. Called by the
    generalized GC when the autonomous registry entry for this issue has
    disappeared (session ended cleanly) AND by the per-session loop when
    the session re-starts self-reporting (the episode ended, recovered)."""
    _stalled_state_path(issue).unlink(missing_ok=True)


def _clear_pod_safety_state(issue: int) -> None:
    """Drop the per-pod state file. Used in exactly two places by the live pass:
    after a successful auto-stop (the episode is over), and by
    :func:`_gc_orphan_pod_safety_state` when the pod has left the RUNNING set
    by ANY path. The classifier's "other" / "pod-active-fresh" / "keep" branches
    do NOT call this — they re-save the state with ``missed=0`` (and the
    refreshed ``alerted`` / ``last_progress_ts``) via :func:`_save_pod_safety_state`;
    the GC reaps that file later if the pod leaves RUNNING. Keeps the state
    schema consistent across ticks (last_progress_ts advances; alerted dedups
    within the episode)."""
    _pod_safety_state_path(issue).unlink(missing_ok=True)


def _gc_orphan_pod_safety_state(
    running_issues: set[int], dry_run: bool, now: float | None = None
) -> list[int]:
    """GC pod-safety state files for pods that have left the RUNNING set by ANY
    path (manual stop/terminate, self-EXIT on TTL/crash), so a re-used
    ``pod-N`` / ``epm-issue-N`` pod doesn't inherit a stale ``missed`` count and
    weaken the 2-miss guard. Also drops files older than
    ``POD_SAFETY_STATE_MAX_AGE_S`` as a secondary backstop in case the API is
    flaky on the tick when a pod actually disappears. Returns the list of issue
    numbers whose state files were cleared (in the order processed)."""
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return []
    now = now if now is not None else time.time()
    cleared: list[int] = []
    for path in sorted(AUTONOMOUS_REGISTRY_DIR.glob(f"{_POD_SAFETY_PREFIX}*.json")):
        stem = path.stem[len(_POD_SAFETY_PREFIX) :]
        try:
            issue = int(stem)
        except ValueError:
            # Garbled name (`pod-safety-foo.json`) — leave it; a hand-debug
            # artifact is none of the GC's business.
            continue
        if issue in running_issues:
            continue
        # Path 1: pod is no longer RUNNING anywhere we can see. Path 2: age
        # backstop catches a file the API failed to "see-it-go" for.
        try:
            payload = json.loads(path.read_text())
            first_seen = payload.get("first_seen", now)
            if not isinstance(first_seen, int | float):
                first_seen = now
        except (json.JSONDecodeError, OSError):
            first_seen = 0  # unreadable -> definitely orphaned, drop it
        age = now - first_seen
        reason = (
            "not in running set" if age < POD_SAFETY_STATE_MAX_AGE_S else f"age={age / 3600:.1f}h"
        )
        print(f"  pod-safety: GC orphan state issue #{issue} ({reason})")
        if not dry_run:
            path.unlink(missing_ok=True)
        cleared.append(issue)
    return cleared


def _post_progress_marker(issue: int, note: str, dry_run: bool, *, label: str) -> None:
    """Record a pod-safety event on task ``issue``'s events.jsonl.

    Uses the generic ``epm:progress`` marker kind: neither ``epm:pod-stopped``
    nor an ``epm:alert`` kind is declared in ``workflow.yaml § markers``, and
    declaring a new marker schema is out of scope for this leaf-node watcher —
    so we post a generic progress note whose body text (carrying the
    auto-stop / stale-alert sentinel) makes the event self-describing. The
    watcher runs from PROJECT_ROOT on `main`, so the task.py branch-guard is
    satisfied. ``label`` is only for the log line (``auto-stop`` / ``alert``)."""
    if dry_run:
        print(f"  [dry-run] would post epm:progress ({label}) on #{issue}: {note}")
        return
    try:
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "post-marker",
                str(issue),
                "epm:progress",
                "--note",
                note,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        # The action (stop / alert) already happened; failing to annotate it is
        # not worth aborting the run. Surface it loudly so the gap is visible.
        print(f"  WARNING: posting {label} marker on #{issue} failed: {e}", file=sys.stderr)


def _stop_pod(issue: int, dry_run: bool) -> bool:
    """Run ``pod.py stop --issue <N>`` (reversible pause; volume preserved).
    Returns True on success. NEVER terminates."""
    cmd = ["uv", "run", "python", "scripts/pod.py", "stop", "--issue", str(issue)]
    if dry_run:
        print(f"  [dry-run] would stop pod: {' '.join(cmd)}")
        return False
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(f"  POD STOP FAILED issue #{issue}: {res.stderr.strip()[:300]}", file=sys.stderr)
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  STOPPED pod issue #{issue} (task is DONE; escaped pod): {first_line}")
    return True


# ─── vm-disk state store + actions ───────────────────────────────────────────


def _vm_disk_state_path() -> Path:
    """Singleton state file for the vm-disk pass (the VM has one root disk —
    not per-issue, so none of the per-issue GC sweeps ever match it)."""
    return AUTONOMOUS_REGISTRY_DIR / "vm-disk.json"


def _load_vm_disk_state() -> dict:
    """Read the vm-disk episode state (``{}`` if absent / unreadable — a fresh
    or garbled file just restarts the episode, mirroring the other stores)."""
    path = _vm_disk_state_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_vm_disk_state(
    *,
    alerted: bool,
    last_reclaim_ts: float | None,
    last_audit_ts: float | None = None,
    prev: dict | None = None,
) -> None:
    """Persist the vm-disk episode state atomically (temp + rename).

    ``alerted`` dedups the once-per-episode alert; ``last_reclaim_ts`` /
    ``last_audit_ts`` re-arm the cache-reclaim / worktree-audit arms after
    :data:`VM_DISK_RECLAIM_REARM_S`; ``first_seen`` carries forward so the
    state records the episode start (mirrors the pod-safety /
    stalled-detector stores)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _vm_disk_state_path()
    prev_first_seen = (prev or {}).get("first_seen")
    if not isinstance(prev_first_seen, int | float):
        prev_first_seen = time.time()
    payload = {
        "alerted": alerted,
        "last_reclaim_ts": last_reclaim_ts,
        "last_audit_ts": last_audit_ts,
        "first_seen": prev_first_seen,
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_vm_disk_state() -> None:
    """Drop the vm-disk state file — the low-disk episode is over (free space
    recovered DECISIVELY, at or above alert + :data:`VM_DISK_CLEAR_HYSTERESIS_BYTES`;
    recovery merely above the alert threshold keeps the state so boundary
    flapping doesn't re-fire the audit/alert), so the next episode alerts
    afresh."""
    _vm_disk_state_path().unlink(missing_ok=True)


def _vm_free_bytes() -> int | None:
    """Free bytes on :data:`VM_DISK_PATH` (``None`` + a loud warning if even
    the statvfs fails — never crash the watcher over the disk check itself)."""
    try:
        return shutil.disk_usage(VM_DISK_PATH).free
    except OSError as e:
        print(f"  vm-disk: disk_usage({VM_DISK_PATH}) failed: {e}", file=sys.stderr)
        return None


def _vm_disk_marker_issues() -> list[int]:
    """Issues that should carry the dashboard-visible vm-disk alert marker:
    every autonomous-registry entry (``issue-<N>.json``) whose task is in an
    :data:`ACTIVE` status — the sessions that will die first when / fills.
    Unreadable entries are skipped (fail-soft)."""
    issues: list[int] = []
    for path in sorted(AUTONOMOUS_REGISTRY_DIR.glob("issue-*.json")):
        try:
            entry = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        issue = entry.get("issue")
        if isinstance(issue, int) and _task_status(issue) in ACTIVE:
            issues.append(issue)
    return issues


def _append_vm_disk_fallback_event(note: str, dry_run: bool) -> None:
    """Durable record of the alert when NO active task exists to carry the
    marker (same role as the `.claude/cache/` fallback file in the
    workflow-fix protocol: a task-less watcher event still needs a queryable
    trace beyond the rotating cron log). Appends one JSON line to
    ``~/.eps-autonomous/vm-disk-events.jsonl``; fail-soft."""
    dest = AUTONOMOUS_REGISTRY_DIR / "vm-disk-events.jsonl"
    line = json.dumps(
        {"ts": datetime.now().astimezone().isoformat(), "kind": "vm-disk-low", "note": note}
    )
    if dry_run:
        print(f"  [dry-run] would append vm-disk event to {dest}")
        return
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as e:
        print(f"  WARNING: appending vm-disk event failed: {e}", file=sys.stderr)


def _vm_reclaim_wandb_cache(dry_run: bool) -> str:
    """``wandb artifact cache cleanup <target>`` — prunes the wandb artifact
    download cache (``~/.cache/wandb/artifacts``) to
    :data:`VM_DISK_WANDB_CACHE_TARGET`. The cache is content-addressed and
    re-fetched on demand, so the cleanup is zero-risk (2026-06-11 episode:
    17.6 GB sat there while / fell to 7.3 GiB). Invoked as ``python -m
    wandb`` via the watcher's own interpreter — the cron env has no
    guaranteed ``wandb`` console script on PATH, and a second ``uv run``
    would contend for the project-env lock. Fail-soft and bounded; a missing
    wandb module is a clean skip. Returns a one-line summary for the marker
    note."""
    cmd = [
        sys.executable,
        "-m",
        "wandb",
        "artifact",
        "cache",
        "cleanup",
        VM_DISK_WANDB_CACHE_TARGET,
    ]
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return "wandb-artifacts skipped (dry-run)"
    try:
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=VM_DISK_UV_PRUNE_TIMEOUT_S,
        )
        tail = ((res.stdout or res.stderr).strip().splitlines() or [""])[-1]
        summary = f"wandb-artifacts rc={res.returncode}: {tail[:160]}"
        print(f"  vm-disk: {summary}")
    except subprocess.TimeoutExpired:
        summary = f"wandb-artifacts timed out at {VM_DISK_UV_PRUNE_TIMEOUT_S}s (fail-soft)"
        print(f"  vm-disk: {summary}")
    except (subprocess.SubprocessError, OSError) as e:
        summary = f"wandb-artifacts failed (fail-soft): {e}"
        print(f"  vm-disk: {summary}", file=sys.stderr)
    return summary


def _vm_reclaim_uv_cache(dry_run: bool) -> str:
    """``uv cache prune`` — drops unused cache entries (safe: uv re-fetches on
    demand). Fail-soft and hard-bounded by :data:`VM_DISK_UV_PRUNE_TIMEOUT_S`
    so a cache lock held by a concurrent ``uv`` process can't hang the watcher
    tick. Returns a one-line summary for the marker note."""
    cmd = ["uv", "cache", "prune"]
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return "uv-cache skipped (dry-run)"
    try:
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=VM_DISK_UV_PRUNE_TIMEOUT_S,
        )
        tail = ((res.stdout or res.stderr).strip().splitlines() or [""])[-1]
        summary = f"uv-cache rc={res.returncode}: {tail[:160]}"
        print(f"  vm-disk: uv cache prune rc={res.returncode}: {tail[:200]}")
    except subprocess.TimeoutExpired:
        # 27 live sessions hold the uv cache lock almost continuously — lock
        # contention is the EXPECTED case; a timeout is a clean skip (the 6h
        # re-arm window retries on a later episode tick).
        summary = "uv-cache skipped (lock contention / timeout)"
        print("  vm-disk: uv cache prune skipped (lock contention / timeout)")
    except (subprocess.SubprocessError, OSError) as e:
        summary = f"uv-cache failed (fail-soft): {e}"
        print(f"  vm-disk: uv cache prune failed (fail-soft): {e}", file=sys.stderr)
    return summary


def _vm_reclaim_npm_cache(dry_run: bool) -> str:
    """``npm cache clean --force`` — drops the npm cache (safe: npm re-fetches
    on demand; ``--force`` is npm's required confirmation flag for ``cache
    clean``, NOT a failure-suppression flag — npm refuses the command without
    it). Fail-soft and bounded like the uv prune; a missing npm binary is a
    clean skip. Returns a one-line summary for the marker note."""
    cmd = ["npm", "cache", "clean", "--force"]
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return "npm-cache skipped (dry-run)"
    try:
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=VM_DISK_UV_PRUNE_TIMEOUT_S,
        )
        tail = ((res.stdout or res.stderr).strip().splitlines() or [""])[-1]
        summary = f"npm-cache rc={res.returncode}: {tail[:160]}"
        print(f"  vm-disk: npm cache clean rc={res.returncode}: {tail[:200]}")
    except (subprocess.SubprocessError, OSError) as e:
        summary = f"npm-cache failed (fail-soft): {e}"
        print(f"  vm-disk: npm cache clean failed (fail-soft): {e}", file=sys.stderr)
    return summary


def _hf_rev_last_accessed(rev) -> float:
    """Newest blob atime across a cached revision's files. A revision with
    no files reads as its ``last_modified`` (conservative: an empty or
    unreadable revision never looks fresher than it is)."""
    times = [f.blob_last_accessed for f in rev.files]
    return max(times) if times else rev.last_modified


def _hf_stale_revisions(cache_info, now: float) -> list:
    """Cached HF hub revisions safe to evict under the conservative TTL cut
    (:data:`VM_DISK_HF_TTL_S`). A revision qualifies only when it was last
    MODIFIED more than the TTL ago, was last READ (newest blob atime,
    :func:`_hf_rev_last_accessed`) more than the TTL ago, AND it is either
    detached (no refs point at it — a superseded or sha-pinned snapshot) or
    its whole repo has not been ACCESSED within the TTL. A repo touched
    recently keeps every ref'd revision (the dataset repos interpreting
    sessions actively re-download); an in-flight download carries a fresh
    ``last_modified`` and a sha-pinned (ref-less) adapter that is actively
    read carries fresh blob atimes, so neither is ever evicted. Pure
    selector (no deletion) so the cut is unit-testable.
    Returns ``CachedRevisionInfo`` objects."""
    stale = []
    for repo in cache_info.repos:
        repo_idle = (now - repo.last_accessed) >= VM_DISK_HF_TTL_S
        for rev in repo.revisions:
            rev_old = (now - rev.last_modified) >= VM_DISK_HF_TTL_S
            rev_unread = (now - _hf_rev_last_accessed(rev)) >= VM_DISK_HF_TTL_S
            if rev_old and rev_unread and (repo_idle or not rev.refs):
                stale.append(rev)
    return stale


def _vm_reclaim_hf_hub_cache(now: float, dry_run: bool) -> str:
    """TTL eviction of stale HF hub cache revisions (2026-06-11 episode:
    41.5 GB VM-side hub cache untouched by any reclaim). Selection is the
    conservative :func:`_hf_stale_revisions` cut; deletion goes through
    ``HFCacheInfo.delete_revisions`` (handles snapshot/blob refcounting —
    never a blanket ``rm`` of repo dirs). Fail-soft end to end: a missing
    ``huggingface_hub``, a scan failure, or a delete failure is a logged
    skip, never a watcher crash. Dry-run returns before scanning (the scan
    walks the whole cache tree — too heavy for a classify-only pass).

    The scan + eviction run on a daemon worker thread joined at
    :data:`VM_DISK_HF_RECLAIM_TIMEOUT_S` — this is the only IN-PROCESS
    remediation step (every other one is a subprocess with ``timeout=``),
    so it needs its own wall-clock bound or a slow walk of a multi-GB
    cache tree stalls the whole watcher tick. A plain daemon
    ``threading.Thread`` is used rather than ``concurrent.futures``:
    ThreadPoolExecutor workers are non-daemon and re-joined at interpreter
    exit (``threading._register_atexit``), so a hung scan would survive
    ``future.result(timeout=...)`` and still hang the watcher's EXIT,
    defeating the bound. On timeout the tick moves on and the orphaned
    daemon worker either finishes late (harmless — the hub cache is a pure
    re-downloadable cache, eviction is idempotent, and any space it frees
    late just lands in a later step's freed-delta annotation) or dies with
    the process (an interrupted ``delete_revisions`` can leave a
    partially-deleted revision, which the hub re-downloads on demand).
    Returns a one-line summary for the marker note."""
    if dry_run:
        print("  [dry-run] would evict HF hub revisions idle > TTL via scan_cache_dir()")
        return "hf-hub-ttl skipped (dry-run)"
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError as e:
        summary = f"hf-hub-ttl skipped (huggingface_hub unavailable: {e})"
        print(f"  vm-disk: {summary}", file=sys.stderr)
        return summary

    # (summary, is_error) — appended exactly once by the worker; read only
    # after a successful join so there is no cross-thread race.
    outcome: list[tuple[str, bool]] = []

    def _scan_and_evict() -> None:
        try:
            try:
                cache_info = scan_cache_dir()
            except Exception as e:  # fail-soft: a disk alert must never crash its own pass
                outcome.append((f"hf-hub-ttl skipped (scan failed: {e})", True))
                return
            stale = _hf_stale_revisions(cache_info, now)
            if not stale:
                outcome.append(("hf-hub-ttl: nothing stale", False))
                return
            strategy = cache_info.delete_revisions(*[rev.commit_hash for rev in stale])
            freed = strategy.expected_freed_size_str
            strategy.execute()
            outcome.append((f"hf-hub-ttl: evicted {len(stale)} revision(s), freed {freed}", False))
        except Exception as e:  # fail-soft, same contract as the scan above
            outcome.append((f"hf-hub-ttl failed (fail-soft): {e}", True))

    worker = threading.Thread(target=_scan_and_evict, name="vm-disk-hf-reclaim", daemon=True)
    try:
        worker.start()
    except RuntimeError as e:  # thread-resource exhaustion — fail-soft like the subprocess steps
        summary = f"hf-hub-ttl skipped (worker spawn failed: {e})"
        print(f"  vm-disk: {summary}", file=sys.stderr)
        return summary
    worker.join(VM_DISK_HF_RECLAIM_TIMEOUT_S)
    if worker.is_alive():
        summary = (
            f"hf-hub-ttl timed out at {VM_DISK_HF_RECLAIM_TIMEOUT_S}s "
            "(fail-soft; daemon worker left to finish or die with the process)"
        )
        print(f"  vm-disk: {summary}", file=sys.stderr)
        return summary
    summary, is_error = (
        outcome[0] if outcome else ("hf-hub-ttl: worker returned no summary (fail-soft)", True)
    )
    print(f"  vm-disk: {summary}", file=sys.stderr if is_error else sys.stdout)
    return summary


def _vm_remediate_worktrees(dry_run: bool) -> str:
    """Run ``worktree_audit.py --apply`` — the remediation that frees the big
    space when / runs low (each stale worktree is a ~14G full checkout; the
    2026-06-11 manual run freed ~60G). The audit carries ALL its own
    keep-guards (live-process holds, non-terminal issue statuses, uncommitted
    tracked changes, grace windows, disk-pressure tightening), so invoking it
    automatically is safe — do NOT duplicate those guards here.

    Fail-soft and bounded by :data:`VM_DISK_WORKTREE_AUDIT_TIMEOUT_S`.
    Returns a one-line summary for the advisory marker note (what was done,
    not just that disk was low)."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "worktree_audit.py"), "--apply"]
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return "worktree-audit skipped (dry-run)"
    try:
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=VM_DISK_WORKTREE_AUDIT_TIMEOUT_S,
        )
        tail = ((res.stdout or res.stderr).strip().splitlines() or [""])[-1]
        summary = f"worktree-audit rc={res.returncode}: {tail[:200]}"
    except subprocess.TimeoutExpired:
        summary = f"worktree-audit timed out at {VM_DISK_WORKTREE_AUDIT_TIMEOUT_S}s (fail-soft)"
    except (subprocess.SubprocessError, OSError) as e:
        summary = f"worktree-audit failed (fail-soft): {e}"
    print(f"  vm-disk: {summary}", file=sys.stderr)
    return summary


def _newest_mtime(root: Path) -> float:
    """Newest mtime anywhere under ``root`` (including ``root`` itself).
    Unreadable entries are skipped; an unstat-able root reads as "fresh now"
    so the sweep NEVER removes a tree it cannot inspect."""
    try:
        newest = root.stat().st_mtime
    except OSError:
        return time.time()
    for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _e: None):
        for name in (".", *filenames):
            try:
                newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
            except OSError:
                continue
    return newest


def _sweep_stale_claude_tmp(now: float, dry_run: bool) -> int:
    """Remove ``/tmp/claude-*`` trees whose ENTIRE contents have been idle
    > :data:`VM_DISK_TMP_SWEEP_AGE_S` (subagent transcript dirs left by
    long-dead sessions). A live session's tree always carries fresh mtimes
    (it writes task outputs continuously), so it is never swept; symlinks
    are skipped. Returns the number of trees removed (counted in dry-run
    too, mirroring the orphan-state GC's logging contract)."""
    removed = 0
    for entry in sorted(Path("/tmp").glob("claude-*")):
        try:
            if entry.is_symlink() or not entry.is_dir():
                continue
            idle_s = now - _newest_mtime(entry)
        except OSError:
            continue
        if idle_s < VM_DISK_TMP_SWEEP_AGE_S:
            continue
        if dry_run:
            print(f"  [dry-run] would remove stale {entry} (idle {idle_s / 86400:.1f}d)")
        else:
            shutil.rmtree(entry, ignore_errors=True)
            print(f"  vm-disk: removed stale {entry} (idle {idle_s / 86400:.1f}d)")
        removed += 1
    return removed


def _refresh_pods_conf_from_api(pod_name: str, dry_run: bool) -> bool:
    """Run ``pod.py config --refresh-from-api <pod_name>`` (the #488
    stale-port self-heal). Pulls fresh host/port from the live RunPod API
    into ``pods.conf`` + ``~/.ssh/config`` so an SSH polling chain that has
    been failing on the pre-stop port can recover without a human in the
    loop.

    Fail-soft: any failure (subprocess timeout, non-zero exit, missing
    binary, oserror) is logged + returns False. The watcher pass never
    crashes on this auto-heal; the caller sets ``refresh_attempted=True``
    regardless so we don't re-fire every tick within the same stalled
    episode (the flag clears when the session resumes self-reporting,
    same as ``alerted``).

    Returns True on success (refresh-from-api exited 0), False otherwise.
    """
    cmd = ["uv", "run", "python", "scripts/pod.py", "config", "--refresh-from-api", pod_name]
    if dry_run:
        print(f"  [dry-run] would refresh-from-api: {' '.join(cmd)}")
        return False
    try:
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(
            f"  REFRESH-FROM-API FAILED for {pod_name}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return False
    if res.returncode != 0:
        print(
            f"  REFRESH-FROM-API FAILED for {pod_name} (rc={res.returncode}): "
            f"{res.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  REFRESHED pods.conf from API for {pod_name}: {first_line}")
    return True


def _running_managed_issue_pods(caller: str = "pod-safety") -> list[tuple[int, str, str]] | None:
    """Live RunPod team pods that are RUNNING and managed (``pod-<N>`` or the
    legacy ``epm-issue-<N>``). Returns ``(issue, pod_id, pod_name)`` triples,
    or ``None`` when the snapshot itself FAILED (API transport error).

    Recognition delegates to :func:`pod_lifecycle._is_managed_pod` +
    :func:`pod_lifecycle._issue_from_pod_name` — the canonical helpers that
    handle BOTH the current ``pod-`` prefix and the legacy ``epm-issue-``
    prefix — instead of a hand-rolled regex (the old regex matched only
    ``epm-issue-<N>``, so it never matched any live pod and the whole pass was
    dead code).

    The pod NAME is threaded out (not just the id) so callers needing to
    address the pod by name — e.g. the #488 stale-port self-heal that shells
    out to ``pod.py config --refresh-from-api <name>`` — don't need a second
    ``list_team_pods`` round-trip to look it up.

    A transport error surfaces as ``None`` with a logged warning — better to
    degrade the pass this tick than to crash the whole run — and ``None`` is
    DISTINCT from ``[]`` ("genuinely no pods") so callers can tell a failed
    snapshot apart from an empty RUNNING set: the pod-safety pass SKIPS its
    state GC on ``None`` (reaping on a failed snapshot wipes the 2-miss
    counters AND the ``alerted`` / ``keep_running_noted`` / ``followup_noted``
    once-per-episode dedup flags, re-arming duplicate markers on every API
    hiccup), while the stalled-detector and session-reconcile passes degrade
    ``None`` to the empty set (their decision guards fail open to "no pods",
    which never stops a pod). ``caller`` labels the warning with the INVOKING
    pass (default ``pod-safety``; the stalled-detector and session-reconcile
    passes thread their own names) so cron-log triage attributes the failure
    to the right pass instead of blaming pod-safety for every reuse of this
    helper."""
    try:
        pods = list_team_pods()
    except Exception as e:
        print(
            f"  {caller}: list_team_pods failed ({e}); "
            f"pod snapshot unavailable this tick (callers degrade per-pass)",
            file=sys.stderr,
        )
        return None
    out: list[tuple[int, str, str]] = []
    for p in pods:
        if p.desired_status != "RUNNING":
            continue
        if not _is_managed_pod(p):
            continue
        name = p.name or ""
        issue = _issue_from_pod_name(name)
        if issue is not None:
            out.append((issue, p.pod_id, name))
    return out


def _process_pod(issue: int, pod_id: str, now: float, dry_run: bool, threshold: int) -> None:
    """Reconcile one RUNNING managed pod against its task status.

    Reads the task's status + latest real-progress timestamp, classifies it,
    and applies :func:`decide_pod_safety`: AUTO-STOP a done task's escaped pod
    (after the 2-miss guard, unless the task carries the ``keep-running`` tag
    OR the task's events.jsonl shows a `epm:run-launched` newer than the
    latest done-transition — i.e. a live inline follow-up — then the stop is
    SKIPPED with a log line + a once-per-pod-incarnation marker), ALERT a
    stale pod-active task once per episode, or KEEP. Persists the per-pod
    state (miss count, alerted flag, keep-running-noted flag, followup-noted
    flag, last-observed real progress) for the next tick."""
    status = _task_status(issue)
    events = _task_events(issue)
    latest_progress = _latest_progress_ts(events)
    status_class = _status_class(status, latest_progress, now)
    # Lazy: the tag and the follow-up predicate only matter when the auto-stop
    # arm is in play, so the extra `task.py view` subprocess + events scan are
    # paid only for escaped-pod candidates. `keep_running` is consulted first
    # because it is the explicit user signal; `followup_active` is the
    # inferred-from-events fallback.
    keep_running = status_class == "auto-stop-done" and _task_keep_running(issue)
    followup_active = (
        status_class == "auto-stop-done"
        and not keep_running
        and _task_followup_active(issue, events=events)
    )

    prev_state = _load_pod_safety_state(issue)
    prev_missed = prev_state.get("missed", 0)
    if not isinstance(prev_missed, int):
        prev_missed = 0
    prev_alerted = bool(prev_state.get("alerted", False))
    prev_keep_running_noted = bool(prev_state.get("keep_running_noted", False))
    prev_followup_noted = bool(prev_state.get("followup_noted", False))
    prev_progress = prev_state.get("last_progress_ts")
    if not isinstance(prev_progress, int | float):
        prev_progress = None

    # Clear the alerted flag so a new staleness episode can re-alert when
    # EITHER (a) real progress advanced since last tick, OR (b) the task is
    # currently classified pod-active-fresh (recent progress ends the prior
    # episode, regardless of whether the previous baseline was None). Without
    # the (b) clause, a pod that was alerted while it had ZERO progress
    # markers, then posted its first real `epm:progress`, then went stale
    # again, would never re-alert — the `progressed` check requires
    # `prev_progress is not None` and so silently fails on the
    # None→first-progress transition.
    progressed = (
        latest_progress is not None
        and prev_progress is not None
        and latest_progress > prev_progress
    )
    alerted = False if (progressed or status_class == "pod-active-fresh") else prev_alerted

    stale = status_class == "pod-active-stale"
    action, new_missed = decide_pod_safety(
        status_class=status_class,
        missed=prev_missed,
        stale=stale,
        alerted=alerted,
        threshold=threshold,
        keep_running=keep_running,
        followup_active=followup_active,
    )
    gap_h = f"{(now - latest_progress) / 3600:.1f}h" if latest_progress is not None else "none"
    print(
        f"  issue #{issue} pod={pod_id}: status={status} class={status_class} "
        f"progress_gap={gap_h} missed={prev_missed}->{new_missed} "
        f"alerted={alerted} action={action}"
    )

    if action == "keep-running-skip":
        print(
            f"  KEEP-RUNNING issue #{issue}: task status '{status}' is DONE but the "
            f"keep-running tag is present — pod-safety stop SKIPPED (pod_id={pod_id}; "
            f"the pod burns until the tag is removed or it is stopped manually)."
        )
        if not prev_keep_running_noted:
            _post_progress_marker(
                issue,
                f"{_KEEP_RUNNING_NOTE_SENTINEL} keep-running exemption: RUNNING pod "
                f"(pod_id={pod_id}) for a task at DONE status '{status}' would have "
                f"been auto-stopped by the pod-safety pass, but the task carries the "
                f"keep-running tag, so the stop is SKIPPED. The pod burns until it is "
                f"stopped manually (`pod.py stop --issue {issue}`) or the tag is "
                f"removed (`task.py remove-tag {issue} keep-running`), which re-arms "
                f"the auto-stop on the next watcher run. Posted once per pod "
                f"incarnation.",
                dry_run,
                label="keep-running-skip",
            )
        if not dry_run:
            _save_pod_safety_state(
                issue,
                pod_id,
                missed=0,
                alerted=alerted,
                last_progress_ts=latest_progress,
                keep_running_noted=True,
                prev=prev_state,
            )
        return

    if action == "followup-skip":
        print(
            f"  FOLLOWUP-ACTIVE issue #{issue}: task status '{status}' is DONE but a "
            f"fresh follow-up signal (epm:run-launched / epm:followup-scope / "
            f"epm:free-analysis-followup-run, newer than the latest done-transition) "
            f"indicates a live inline follow-up — pod-safety stop SKIPPED "
            f"(pod_id={pod_id}; the auto-stop re-arms when the follow-up posts its "
            f"next status-changed/promoted)."
        )
        if not prev_followup_noted:
            _post_progress_marker(
                issue,
                f"{_FOLLOWUP_NOTE_SENTINEL} inline-follow-up exemption: RUNNING pod "
                f"(pod_id={pod_id}) for a task at DONE status '{status}' would have "
                f"been auto-stopped by the pod-safety pass, but the task's "
                f"events.jsonl shows a follow-up signal marker (epm:run-launched / "
                f"epm:followup-scope / epm:free-analysis-followup-run) NEWER than "
                f"the latest done-transition (epm:promoted / epm:status-changed). "
                f"That is the CLAUDE.md 'Routing experiment intent → Follow-up' "
                f"pattern: a user-approved inline follow-up is in flight on a "
                f"promoted/completed parent (epm:followup-scope covers the "
                f"user-chat case where the pod is provisioned before the run "
                f"launches — refs #573), so the pod is legitimately in use. The "
                f"auto-stop re-arms naturally when the follow-up posts its next "
                f"status-changed / promoted event. Posted once per pod incarnation. "
                f"Override with `task.py add-tag {issue} keep-running` to suppress "
                f"all future pod-safety stops, or stop manually with `pod.py stop "
                f"--issue {issue}` if the follow-up is truly done.",
                dry_run,
                label="followup-skip",
            )
        if not dry_run:
            _save_pod_safety_state(
                issue,
                pod_id,
                missed=0,
                alerted=alerted,
                last_progress_ts=latest_progress,
                followup_noted=True,
                prev=prev_state,
            )
        return

    if action == "stop":
        stopped = _stop_pod(issue, dry_run)
        if stopped:
            _post_progress_marker(
                issue,
                f"{_AUTOSTOP_NOTE_SENTINEL} auto-stopped by autonomous_session_watch "
                f"pod-safety pass — RUNNING pod for a task whose status is "
                f"'{status}' (already DONE), so the pod is an escaped / "
                f"Step-8-terminate-failed pod (pod_id={pod_id}); reversible pause, "
                f"volume preserved (pod.py resume). Confirmed for >= {threshold} checks.",
                dry_run,
                label="auto-stop",
            )
            if not dry_run:
                _clear_pod_safety_state(issue)
        return

    if action == "alert":
        _post_progress_marker(
            issue,
            f"{_ALERT_NOTE_SENTINEL} STALE pod-active task: RUNNING pod "
            f"(pod_id={pod_id}) for a task at status '{status}' with no real "
            f"progress marker in > {ALERT_STALE_HOURS:.0f}h "
            f"(gap={gap_h}). Likely an abandoned session — investigate. "
            f"NOT auto-stopped (a mid-run stop risks killing a healthy long "
            f"run); stop manually with `pod.py stop --issue {issue}` if the "
            f"session is truly dead.",
            dry_run,
            label="alert",
        )
        print(
            f"  ALERT issue #{issue}: pod-active task stale > {ALERT_STALE_HOURS:.0f}h "
            f"(gap={gap_h}); NOT stopping (mid-run safety).",
            file=sys.stderr,
        )
        if not dry_run:
            _save_pod_safety_state(
                issue,
                pod_id,
                missed=0,
                alerted=True,
                last_progress_ts=latest_progress,
                prev=prev_state,
            )
        return

    # action == "keep": persist the (possibly incremented) miss count, the
    # alerted flag (reset if progress advanced), and the latest observed
    # progress so the next tick can detect advancement.
    if not dry_run:
        _save_pod_safety_state(
            issue,
            pod_id,
            missed=new_missed,
            alerted=alerted,
            last_progress_ts=latest_progress,
            prev=prev_state,
        )


# ─── alive-but-stalled detector — top-level driver ───────────────────────────


def _self_report_age_seconds(issue: int, now: float) -> tuple[float | None, str | None]:
    """Read the per-issue self-report file and return ``(age_seconds, ts_iso)``.

    Returns ``(None, None)`` when there is no self-report file (interactive
    session, or autonomous session that hasn't ticked yet). Returns
    ``(age_seconds, ts_iso)`` for a present file with a parseable timestamp.
    Returns ``(None, ts_iso)`` for a present but malformed/unparseable ts —
    the caller treats it as "no self-report" so a malformed file doesn't
    accidentally trip the alert.

    Imported lazily so this module stays importable when the
    ``session_progress_report`` helper isn't on the path (e.g. unit tests
    that monkeypatch the whole helper).
    """
    try:
        from session_progress_report import _parse_iso, read_self_report
    except ImportError:
        return (None, None)
    report = read_self_report(issue)
    if report is None:
        return (None, None)
    ts_str = report.get("ts") if isinstance(report, dict) else None
    if not isinstance(ts_str, str):
        return (None, None)
    parsed = _parse_iso(ts_str)
    if parsed is None:
        return (None, ts_str)
    age = now - parsed.timestamp()
    return (age, ts_str)


# ── ALIVE-BUT-STALLED exemption: in-flight provision / fresh poll state ─────
#
# refs #573: ~63 ALIVE-BUT-STALLED auto-respawns across 17 tasks on
# 2026-06-09 killed healthy sessions mid-step; #534's respawn killed an
# in-flight `pod.py provision` THREE times, adding ~8h. A provision waiting
# for capacity legitimately posts no markers and freezes the self-report
# (the session's bg-Bash chain is blocked on the wait), so the staleness
# signals alone misclassify it. Before acting on a stale entry, probe two
# cheap local signals; either one exempts the session this tick:
#   1. a LIVE `pod.py provision|resume --issue <N>` (or pod_lifecycle.py)
#      process on this VM — /proc cmdline scan, no psutil dependency;
#   2. fresh poll-pipeline tick state for the issue
#      (.claude/cache/poll-pipeline-<N>.json mtime within the stalled
#      window) — the polling chain is demonstrably alive even if it has
#      not posted a marker this window.

# poll_pipeline's DEFAULT_STATE_DIR (kept in sync by convention; the file
# name is poll-pipeline-<issue>.json).
_POLL_STATE_DIR = PROJECT_ROOT / ".claude" / "cache"


def _find_provision_process(issue: int) -> int | None:
    """PID of a live ``pod.py provision|resume --issue <N>`` /
    ``pod_lifecycle.py provision|resume --issue <N>`` process, or ``None``.

    Pure /proc cmdline scan (NUL-separated argv): a process qualifies when
    its argv has (a) a token ending in ``pod.py`` or ``pod_lifecycle.py``,
    (b) a bare ``provision`` or ``resume`` verb token, and (c) ``--issue <N>``
    (adjacent tokens or the ``--issue=<N>`` form). Fail-soft: any read error
    on a /proc entry skips that entry; an unreadable /proc returns None.
    """
    needle = str(issue)
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        argv = [a for a in raw.decode("utf-8", "replace").split("\0") if a]
        if not any(a.endswith(("pod.py", "pod_lifecycle.py")) for a in argv):
            continue
        if not any(a in ("provision", "resume") for a in argv):
            continue
        for i, a in enumerate(argv):
            if (a == "--issue" and i + 1 < len(argv) and argv[i + 1] == needle) or (
                a == f"--issue={needle}"
            ):
                return int(entry.name)
    return None


def _provision_in_flight_reason(issue: int, now: float) -> str | None:
    """Human-readable exemption reason when issue #N has in-flight pod
    provisioning / fresh polling activity, else ``None``. See the comment
    block above for the two signals (refs #573)."""
    pid = _find_provision_process(issue)
    if pid is not None:
        return f"live pod provision/resume process (pid {pid}) for issue #{issue}"
    state = _POLL_STATE_DIR / f"poll-pipeline-{issue}.json"
    try:
        age = now - state.stat().st_mtime
    except OSError:
        return None
    # Negative age = mtime in the FUTURE relative to `now` (clock skew, or a
    # caller-supplied fake clock) — never "fresh", or the exemption would
    # permanently mask a genuinely stalled session.
    if 0 <= age < STALLED_WINDOW_S:
        return f"poll-pipeline state fresh ({age / 60:.1f}m old): {state.name}"
    return None


# ─── followups_running parent waiting on open child (suppression) ───────────
#
# `followups_running` is in ACTIVE (un-phantomed 2026-06-10) so SAME-issue
# follow-up rounds get respawn/orphan coverage while they are executing. But
# the status has a SECOND shape: a parent whose own `/issue` pipeline is
# complete and that is purely waiting on a child task to clear. The parent's
# latest `epm:step-completed` carries `step: 10` (the completion-audit step)
# with `exit_kind: parked` and a note naming the open child(ren). Respawning
# the parent session here cannot advance the parent — only user action on the
# child (the canonical case is a child at ``awaiting_promotion`` whose
# ``task.py promote`` is a user-only gate) or all children reaching terminal
# unblocks it. Three respawn-and-park cycles happened in two hours on #533
# (2026-06-11 12:43 / 13:43 / 14:43 UTC) — each respawned session re-posted
# the same parked step-10 marker and exited.
#
# The exemption fires when ALL of:
#   (a) status == "followups_running"
#   (b) has_pod is False — a same-issue follow-up round provisions a pod, so
#       a live pod is the "this IS a fresh round, keep monitoring" signal.
#   (c) the latest non-watcher ``epm:step-completed`` has step="10" and
#       exit_kind="parked"
#   (d) at least one child task (via ``task.py list-children``) is NOT in
#       {completed, archived} — i.e. there IS an open child blocking advance
#
# When all four hold, the stalled / orphan-respawn passes treat the situation
# as "would have respawned, but the respawn provably cannot help"; they post
# a one-time alert marker (dedup'd via a state-file flag) and skip the
# respawn entirely (does NOT consume the respawn budget — this is not a
# respawn). When the parent's latest step-completed advances past step=10
# (the user promoted the child and `/issue 533` re-ran Step 10 to flip the
# parent), the next tick observes a different latest step-completed shape
# and the suppression dissolves naturally.
#
# ROUND-COMPLETE RE-PARK (the suppression's counterpart — #533 freeze,
# 2026-06-11→12): the status has a THIRD shape — a same-issue follow-up
# round that finished every gate but whose owning session died BEFORE
# executing the designed re-park (`set-status <N> awaiting_promotion`,
# SKILL.md Step 9b § Same-issue follow-up loop step 3). Detectable from the
# markers alone: a parked round-end ``epm:step-completed`` NEWER than the
# round's ``epm:followup-scope``, with NO ``epm:same-issue-followup-run``
# completion marker recording the round (a recorded round — run marker
# newer than the scope — means the re-park already happened per the
# designed step-3→step-4 ordering, so a later ``followups_running`` status
# is the legacy children-in-flight shape, not this round stranded).
# Neither suppressing (freeze — what happened to #533 for ~26h) nor
# respawning (each respawned session re-concluded "waiting on child",
# posted another parked step-10 marker, and exited — 3 cycles in 2h) fixes
# it; the watcher executes the re-park directly
# (:func:`_repark_completed_followup_round`), then posts the round's
# ``epm:same-issue-followup-run`` completion marker on the dead session's
# behalf (closing the scope for `/issue` Step 0 routing — an unrun scope
# would re-route a re-invoked session into re-running the completed round)
# plus a sentinel-stamped progress marker. Probed BEFORE the awaiting-child
# suppression in both passes; on a failed re-park the passes fall back to
# the pre-existing handling so the fix is never worse than the old
# behavior. A task with NO ``epm:followup-scope`` on record (the legacy
# children-in-flight shape) never triggers the re-park.

# Step + exit_kind that mark the "parked, awaiting child" state. Pinned as
# constants so the tests + the helper share one source of truth.
_FOLLOWUPS_CHILDREN_WAIT_STEP = "10"
_FOLLOWUPS_CHILDREN_WAIT_EXIT_KIND = "parked"

# Steps whose ``exit_kind=parked`` step-completed marks the END of a
# same-issue follow-up round (the designed next action is the re-park to
# ``awaiting_promotion``): ``9a-bis`` is the round's documented EXIT site
# (SKILL.md Step 9b § Same-issue follow-up loop — the §5 marker posted at
# the tail of the clean-result re-gate) and ``10`` is the "classification
# pending; awaiting promotion" park a re-driven session posts when it finds
# the pipeline already complete. A mid-round park (e.g. step 2c over-cap
# plan approval, which holds at ``followups_running`` in place) is NOT
# round-end — re-parking there would abandon an unapproved round.
_FOLLOWUP_ROUND_END_STEPS = frozenset({"9a-bis", "10"})

# Statuses that count a child task as TERMINAL for the purpose of this check.
# A child at `awaiting_promotion` is NOT terminal here — it is exactly the
# user-gated state we are trying to wait out (the user runs `task.py promote`
# to move it to `completed`). A child at `archived` IS terminal.
_FOLLOWUPS_CHILD_TERMINAL = {"completed", "archived"}


def _latest_step_completed(events: list[dict]) -> dict | None:
    """Return the newest non-watcher ``epm:step-completed`` event in
    ``events`` (or ``None`` if there isn't one). The watcher itself never
    posts ``epm:step-completed`` so the sentinel filter is defense-in-depth.

    The returned dict is the raw event row; callers read ``step`` /
    ``exit_kind`` directly off it (both are top-level fields on the
    event row, set by ``scripts/post_step_completed.py``).
    """
    best: dict | None = None
    best_ts: float | None = None
    for ev in events:
        if ev.get("kind") != "epm:step-completed":
            continue
        note = ev.get("note") or ""
        if any(sentinel in note for sentinel in _WATCHER_NOTE_SENTINELS):
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is None:
            continue
        if best_ts is None or ts > best_ts:
            best_ts = ts
            best = ev
    return best


def _task_children(issue: int) -> list[dict]:
    """Children of ``issue`` via ``task.py list-children --json``; ``[]`` on
    any read failure (same subprocess isolation as :func:`_task_status`).
    Mirrors :func:`_campaign_children` but kept separate so the followups
    suppression doesn't cross-depend on the campaign pass."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "list-children", str(issue), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _followups_awaiting_child_reason(
    issue: int,
    *,
    status: str | None,
    has_pod: bool,
    events: list[dict],
) -> str | None:
    """Human-readable exemption reason when ``issue`` is a ``followups_running``
    parent waiting on an open child task (see the comment block above for the
    four-condition predicate). Returns ``None`` when the exemption does not
    apply.

    Probed LAZILY by the callers (the helper is only invoked when the stalled
    / orphan pass already wants to respawn) so a healthy session never pays
    the ``task.py list-children`` subprocess.
    """
    if status != "followups_running":
        return None
    if has_pod:
        return None
    sc = _latest_step_completed(events)
    if sc is None:
        return None
    step = sc.get("step")
    exit_kind = sc.get("exit_kind")
    if step != _FOLLOWUPS_CHILDREN_WAIT_STEP:
        return None
    if exit_kind != _FOLLOWUPS_CHILDREN_WAIT_EXIT_KIND:
        return None
    children = _task_children(issue)
    if not children:
        return None
    open_ids: list[int] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        cid = child.get("id")
        cstatus = child.get("status")
        if not isinstance(cid, int) or not isinstance(cstatus, str):
            continue
        if cstatus not in _FOLLOWUPS_CHILD_TERMINAL:
            open_ids.append(cid)
    if not open_ids:
        return None
    open_ids.sort()
    ids_str = ", ".join(f"#{i}" for i in open_ids)
    return (
        f"followups_running parent waiting on open child(ren) {ids_str}; "
        f"latest epm:step-completed step={step} exit_kind={exit_kind} "
        f"(child promotion is a user-only gate; respawning the parent "
        f"cannot advance it)"
    )


def _followup_round_complete_reason(events: list[dict]) -> str | None:
    """Reason string when ``events`` show a COMPLETED-but-UNRECORDED
    same-issue follow-up round whose designed re-park (``set-status <N>
    awaiting_promotion``) never ran: the latest non-watcher
    ``epm:step-completed`` carries ``exit_kind=parked`` with a round-end
    step (:data:`_FOLLOWUP_ROUND_END_STEPS`) NEWER than the latest
    ``epm:followup-scope``, and no ``epm:same-issue-followup-run``
    completion marker records the round — the #533 shape (round-end
    ``9a-bis`` park, then ``10`` parks from respawned sessions).

    ``None`` when:
    - no ``epm:followup-scope`` is on record (the legacy children-in-flight
      shape has no scope marker — never re-park it);
    - the round is still in flight (scope newer than any round-end signal);
    - the round is RECORDED — an ``epm:same-issue-followup-run`` newer than
      the scope. The designed ordering posts that marker only AFTER the
      re-park (loop step 3 → step 4), so a ``followups_running`` status
      alongside a recorded round is a LATER legitimate transition (the
      legacy children-in-flight shape via Step 10 step 5), NOT this round
      stranded — defer to the awaiting-child suppression. This also
      self-disarms the predicate after the watcher's own re-park, which
      posts the completion marker itself
      (:func:`_repark_completed_followup_round`).

    Pure over the already-loaded ``events`` — no subprocess.
    """
    scope_ts: float | None = None
    run_marker_ts: float | None = None
    for ev in events:
        kind = ev.get("kind")
        if kind not in ("epm:followup-scope", "epm:same-issue-followup-run"):
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is None:
            continue
        if kind == "epm:followup-scope":
            if scope_ts is None or ts > scope_ts:
                scope_ts = ts
        elif run_marker_ts is None or ts > run_marker_ts:
            run_marker_ts = ts
    if scope_ts is None:
        return None
    if run_marker_ts is not None and run_marker_ts > scope_ts:
        return None
    sc = _latest_step_completed(events)
    if sc is None:
        return None
    step = sc.get("step")
    sc_ts = _parse_event_ts(sc.get("ts"))
    if (
        step in _FOLLOWUP_ROUND_END_STEPS
        and sc.get("exit_kind") == _FOLLOWUPS_CHILDREN_WAIT_EXIT_KIND
        and sc_ts is not None
        and sc_ts > scope_ts
    ):
        return (
            f"same-issue follow-up round complete (round-end "
            f"epm:step-completed step={step} exit_kind=parked newer than the "
            f"round's epm:followup-scope) but the task is still at "
            f"followups_running — the owning session exited before the "
            f"designed re-park"
        )
    return None


def _scope_note_field(events: list[dict], field: str) -> str | None:
    """Value of ``<field>: <value>`` from the latest ``epm:followup-scope``
    event's note (line-prefix match, first hit wins), or ``None``."""
    latest: dict | None = None
    latest_ts: float | None = None
    for ev in events:
        if ev.get("kind") != "epm:followup-scope":
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is None:
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest = ev
    if latest is None:
        return None
    for line in (latest.get("note") or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            value = stripped[len(field) + 1 :].strip().rstrip(",")
            return value or None
    return None


def _post_followup_run_marker(issue: int, events: list[dict], dry_run: bool) -> bool:
    """Post the ``epm:same-issue-followup-run v1`` completion marker for the
    round the watcher just re-parked, closing the round's
    ``epm:followup-scope`` for `/issue` Step 0 routing — WITHOUT it the
    scope stays UNRUN and a re-invoked session would re-dispatch the
    already-completed round (the Step 0 dispatch table routes a post-result
    status + unrun scope back into the follow-up loop). ``followup_label``
    + ``source`` are parsed from the latest scope marker's note so the
    idempotency match and the autonomous round-cap counting stay correct;
    ``round`` is 1 + the count of existing run markers. Fail-soft: a
    missing label or a failed post logs to stderr and returns False — the
    re-park itself (the substance) already happened."""
    label = _scope_note_field(events, "followup_label")
    if label is None:
        print(
            f"  issue #{issue}: cannot post epm:same-issue-followup-run — no "
            f"followup_label parseable from the latest epm:followup-scope note; "
            f"the scope stays unrun (Step 0 may re-route into the loop)",
            file=sys.stderr,
        )
        return False
    source = _scope_note_field(events, "source") or "unknown"
    round_idx = 1 + sum(1 for ev in events if ev.get("kind") == "epm:same-issue-followup-run")
    note = (
        f"followup_label: {label}\n"
        f"source: {source}\n"
        f"round: {round_idx}\n"
        f"outcome: round completed but the owning session died before the "
        f"designed re-park; autonomous_session_watch executed the re-park "
        f"(set-status awaiting_promotion) and posted this completion marker "
        f"on its behalf (#533 freeze class)"
    )
    if dry_run:
        print(f"  [dry-run] would post epm:same-issue-followup-run on #{issue}: {label}")
        return True
    try:
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "post-marker",
                str(issue),
                "epm:same-issue-followup-run",
                "--note",
                note,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(
            f"  issue #{issue}: epm:same-issue-followup-run post FAILED ({exc}); "
            f"the scope stays unrun (Step 0 may re-route into the loop)",
            file=sys.stderr,
        )
        return False
    return True


def _repark_completed_followup_round(
    issue: int, reason: str, events: list[dict], dry_run: bool
) -> bool:
    """Execute the stranded re-park for a COMPLETED same-issue follow-up
    round: ``task.py set-status <issue> awaiting_promotion`` — the move the
    dead session was designed to make (explicitly permitted by the
    ``set_status`` followups_running guard in ``task_workflow.py``) — then
    post the round's ``epm:same-issue-followup-run`` completion marker
    (:func:`_post_followup_run_marker`, closes the scope for Step 0
    routing) and a sentinel-stamped progress marker documenting the
    intervention. Returns True on set-status success (callers skip respawn
    / suppression), False on set-status failure (callers fall back to the
    pre-existing handling, so a failed re-park is never WORSE than the old
    freeze). ``dry_run`` classifies only — no mutation, no markers.

    The watcher runs from PROJECT_ROOT on ``main``, so the task.py
    branch-guard is satisfied (same contract as
    :func:`_post_progress_marker`).
    """
    if dry_run:
        print(
            f"  issue #{issue}: DRY-RUN would re-park completed same-issue "
            f"follow-up round -> awaiting_promotion ({reason})"
        )
        return True
    try:
        out = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "set-status",
                str(issue),
                "awaiting_promotion",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(
            f"  issue #{issue}: follow-up round re-park FAILED ({exc}); "
            f"falling back to existing stalled/orphan handling",
            file=sys.stderr,
        )
        return False
    if out.returncode != 0:
        detail = (out.stderr or out.stdout or "").strip()[:300]
        print(
            f"  issue #{issue}: follow-up round re-park FAILED "
            f"(set-status rc={out.returncode}): {detail}",
            file=sys.stderr,
        )
        return False
    print(f"  issue #{issue}: re-parked completed follow-up round -> awaiting_promotion")
    _post_followup_run_marker(issue, events, dry_run)
    _post_progress_marker(
        issue,
        f"{_FOLLOWUP_ROUND_REPARK_NOTE_SENTINEL} {reason}. Watcher executed "
        f"the designed re-park (`set-status {issue} awaiting_promotion`) on "
        f"behalf of the dead/stalled session (incident #533 freeze class). "
        f"Review the clean-result and promote via `task.py promote {issue} "
        f"useful|not-useful`, then re-invoke `/issue {issue}` to fire Step 10.",
        dry_run,
        label="followup-round-repark",
    )
    return True


def _stop_session(session_id: str, dry_run: bool) -> bool:
    """Stop an in-flight Happy session by id via
    ``spawn_session.py stop --session-id <id>``. Returns True on success.

    Used in the stalled-detector AUTO-RESPAWN path: the OLD session is
    still alive (that's what distinguishes the stalled-detector from the
    crash-recovery respawn pass), so a respawn that skipped this step
    would leave two `--auto` sessions pointed at the same issue. Both
    would try to drive the same workflow.

    Best-effort: on failure we log the error to stderr and return False,
    so the caller declines to respawn rather than risking the duplicate-
    session case. A stop failure is logged loudly because it is the
    common cause of an exhausted respawn cap.
    """
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "stop",
        "--session-id", session_id,
    ]  # fmt: skip
    if dry_run:
        print(f"  [dry-run] would stop session: {' '.join(cmd)}")
        return False
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        print(
            f"  STOP SESSION FAILED session_id={session_id}: "
            f"{(res.stderr or res.stdout).strip()[:300]}",
            file=sys.stderr,
        )
        return False
    return True


def _respawn_stalled_session(issue: int, cap_gpu_hours: float, dry_run: bool) -> bool:
    """Spawn a fresh `--auto` session for ``issue``.

    Mirrors :func:`_respawn` (used by the crash-recovery pass) but is
    decoupled from the autonomous-registry entry shape — the stalled-
    detector path knows the issue and the cap directly from the loaded
    state, so it doesn't pass a registry-entry dict. Returns True on
    success; spawn_session rewrites the registry (new id, missed=0) as a
    side effect.

    Note: we do NOT call :func:`_respawn` directly because the
    spawn-issue invocation here is the SAME (`--auto`
    `--auto-approve-gpu-hours`) but the surrounding context differs:
    this path has already called :func:`_stop_session` first, and the
    log prefix is `RESPAWNED-STALLED` rather than `RESPAWNED` so the
    operator can tell the two paths apart in the watcher logs.
    """
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto", "--auto-approve-gpu-hours", str(cap_gpu_hours),
    ]  # fmt: skip
    cmd.extend(_stalled_session_overrides(issue))
    if dry_run:
        print(f"  [dry-run] would respawn stalled: {' '.join(cmd)}")
        return False
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(
            f"  RESPAWN-STALLED FAILED issue #{issue}: {res.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  RESPAWNED-STALLED issue #{issue} (alive-but-stalled): {first_line}")
    return True


def _stalled_cap_gpu_hours(issue: int) -> float:
    """Read the per-issue autonomous registry entry's
    ``auto_approve_gpu_hours`` cap (default 24.0 if missing/garbled), so
    the auto-respawn reuses the same cap the user originally chose.
    Mirrors the lookup :func:`_respawn` does on its registry entry."""
    entry_path = AUTONOMOUS_REGISTRY_DIR / f"issue-{issue}.json"
    try:
        entry = json.loads(entry_path.read_text())
    except (json.JSONDecodeError, OSError):
        return 24.0
    cap = entry.get("auto_approve_gpu_hours", 24.0)
    if not isinstance(cap, int | float):
        return 24.0
    return float(cap)


def _stalled_session_overrides(issue: int) -> list[str]:
    """Return any model / betas / effort cmdline overrides recorded on this
    issue's autonomous registry entry, ready to splat onto a ``spawn-issue``
    invocation. Returns ``[]`` when the entry is missing, unreadable, or
    pre-dates the override-persistence feature — that branch matches the
    pre-feature respawn behavior (inherit global defaults), so a missing
    file FAILS TOWARD KEEP-THE-OLD-SHAPE rather than blocking recovery.

    These three flags are part of the prompt-cache key, so the stalled and
    orphan respawn paths MUST re-pass the same values they find in the
    registry (CLAUDE.md § Context hygiene)."""
    entry_path = AUTONOMOUS_REGISTRY_DIR / f"issue-{issue}.json"
    try:
        entry = json.loads(entry_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(entry, dict):
        return []
    extra: list[str] = []
    model = entry.get("model")
    if model:
        extra.extend(["--model", str(model)])
    betas = entry.get("betas")
    if isinstance(betas, list) and betas:
        extra.extend(["--betas", ",".join(str(b) for b in betas)])
    effort = entry.get("effort")
    if effort:
        extra.extend(["--effort", str(effort)])
    return extra


class _StalledActionCtx:
    """Plain-data carrier that bundles every value the three stalled-action
    handlers (:func:`_handle_stalled_respawn`, :func:`_handle_stalled_exhausted`,
    :func:`_handle_stalled_alert`) need.

    Exists so :func:`_process_stalled_session` can dispatch on the action enum
    via three one-line calls (keeping it under the C901 cyclomatic-complexity
    cap) without losing the wide context the handlers depend on (the prose
    of each marker note quotes the same set of measured signals).
    Deliberately not a dataclass — we don't need equality / repr / mutation;
    the only contract is "all fields are read by at least one handler" and a
    plain class with ``__init__`` is enough.
    """

    def __init__(
        self,
        *,
        issue: int,
        happy_session_id: object,
        prev_state: dict,
        alerted: bool,
        respawn_count: int,
        exhausted: bool,
        last_self_report_ts: str | None,
        self_gap: str,
        marker_gap: str,
        has_pod: bool,
        task_status: str | None,
        in_active: bool,
        threshold: int,
        dry_run: bool,
        refresh_attempted: bool = False,
        pod_name: str | None = None,
        manual: bool = False,
        followups_child_alerted: bool = False,
    ) -> None:
        self.issue = issue
        self.happy_session_id = happy_session_id
        self.prev_state = prev_state
        self.alerted = alerted
        self.respawn_count = respawn_count
        self.exhausted = exhausted
        self.last_self_report_ts = last_self_report_ts
        self.self_gap = self_gap
        self.marker_gap = marker_gap
        self.has_pod = has_pod
        self.task_status = task_status
        self.in_active = in_active
        self.threshold = threshold
        self.dry_run = dry_run
        # #488 stale-port self-heal — see ``_refresh_pods_conf_from_api``
        # + ``_handle_stalled_alert``. ``refresh_attempted`` carries the
        # one-shot-per-episode dedup; ``pod_name`` (when known) lets the
        # alert handler address the live pod without a second
        # ``list_team_pods`` round-trip.
        self.refresh_attempted = refresh_attempted
        self.pod_name = pod_name
        # True for a manual (``manual-issue-<N>.json``, bare ``spawn-issue``)
        # registration: ALERT-ONLY by design — the alert handler adjusts its
        # prose (a manual entry's liveness was never verified, and the
        # decline reason is "user-driven", not status/daemon). The respawn /
        # exhausted handlers never see manual entries (the caller forces
        # ``respawn_eligible=False``). #505 round-2 orphaning, 2026-06-10.
        self.manual = manual
        # Per-episode dedup for the followups_running-parent-waiting-on-open-
        # child suppression alert (see ``_followups_awaiting_child_reason``).
        # Carried through every state-persist site so the alert fires at
        # most once per episode and clears on real-progress advancement.
        self.followups_child_alerted = followups_child_alerted

    @property
    def happy_session_id_str(self) -> str | None:
        """Narrow ``happy_session_id`` (typed ``object`` because it comes from
        a JSON read) to ``str | None`` for the state-save call sites."""
        return self.happy_session_id if isinstance(self.happy_session_id, str) else None


def _handle_stalled_respawn(ctx: _StalledActionCtx) -> None:
    """Recovery action: stop the alive-but-stalled session, spawn a fresh
    ``--auto`` session, persist the bumped respawn_count. On stop failure,
    persist unchanged respawn_count + a fresh ``missed=0`` so the next tick
    re-tries within the same episode.

    #488 stale-port self-heal (refs #572): when the stalled session has a
    RUNNING managed pod, fire ``pod.py config --refresh-from-api`` once per
    episode BEFORE the stop+respawn — previously only the ALERT fallback
    (non-ACTIVE status / daemon down / manual) fired it, so the COMMON
    autonomous case (ACTIVE + daemon reachable) respawned a fresh session
    straight into the same stale pods.conf endpoint and the new session
    re-spun the dead-port SSH loop. Same ``refresh_attempted`` dedup as the
    alert arm; fail-soft.

    Safety precondition: we MUST know which session id to stop before we
    spawn a fresh one. A garbled / missing ``happy_session_id`` in the
    registry entry would otherwise mean we skip the stop and spawn anyway,
    leaving two `--auto` sessions racing on the same issue (= duplicate
    pods, fastest cost-incident on the watcher). When ``sid`` is falsy /
    non-str, decline this tick and persist state so the next tick (which
    reads a fresh registry entry — the orchestrator or a recent re-spawn
    may have rewritten it) can try again.
    """
    # Heal pods.conf BEFORE deciding/acting on the session so the respawned
    # session reads a fresh endpoint. Dedup'd per episode, like the alert arm.
    if ctx.has_pod and ctx.pod_name and not ctx.refresh_attempted:
        print(
            f"  REFRESH-FROM-API issue #{ctx.issue}: stalled session has "
            f"RUNNING pod {ctx.pod_name}; attempting #488 stale-port "
            f"self-heal before respawn",
            file=sys.stderr,
        )
        _refresh_pods_conf_from_api(ctx.pod_name, ctx.dry_run)
        # Mark attempted regardless of subprocess outcome (no hot-loop);
        # clears on self-report advancement, same as the alert arm.
        ctx.refresh_attempted = True

    sid = ctx.happy_session_id_str
    if not sid:
        print(
            f"  RESPAWN-STALLED SKIPPED issue #{ctx.issue}: registry entry has "
            f"no usable happy_session_id (raw={ctx.happy_session_id!r}); "
            f"cannot stop the old session, so spawning would risk a duplicate. "
            f"Persisting state for next tick.",
            file=sys.stderr,
        )
        if not ctx.dry_run:
            _save_stalled_state(
                ctx.issue,
                None,
                missed=0,
                alerted=ctx.alerted,
                last_self_report_ts=ctx.last_self_report_ts,
                respawn_count=ctx.respawn_count,
                exhausted=ctx.exhausted,
                refresh_attempted=ctx.refresh_attempted,
                followups_child_alerted=ctx.followups_child_alerted,
                prev=ctx.prev_state,
            )
        return
    stop_ok = _stop_session(sid, ctx.dry_run)
    if not stop_ok:
        if not ctx.dry_run:
            _save_stalled_state(
                ctx.issue,
                sid,
                missed=0,
                alerted=ctx.alerted,
                last_self_report_ts=ctx.last_self_report_ts,
                respawn_count=ctx.respawn_count,
                exhausted=ctx.exhausted,
                refresh_attempted=ctx.refresh_attempted,
                followups_child_alerted=ctx.followups_child_alerted,
                prev=ctx.prev_state,
            )
        return
    cap = _stalled_cap_gpu_hours(ctx.issue)
    spawn_ok = _respawn_stalled_session(ctx.issue, cap, ctx.dry_run)
    new_respawn_count = ctx.respawn_count + 1
    if spawn_ok:
        _post_progress_marker(
            ctx.issue,
            f"{_STALLED_RESPAWN_NOTE_SENTINEL} ALIVE-BUT-STALLED auto-"
            f"respawn: Happy session id={ctx.happy_session_id} was in the "
            f"live set but self-report has been frozen for {ctx.self_gap} "
            f"and the latest non-watcher progress marker is {ctx.marker_gap} "
            f"old (has_pod={ctx.has_pod}, status={ctx.task_status}). Stopped "
            f"the old session and spawned a fresh `--auto` session "
            f"(respawn {new_respawn_count}/{STALLED_MAX_RESPAWNS} this "
            f"episode). Confirmed for >= {ctx.threshold} checks.",
            ctx.dry_run,
            label="session-auto-respawn",
        )
    if not ctx.dry_run:
        _save_stalled_state(
            ctx.issue,
            # spawn_session.py rewrote the registry's happy_session_id, but
            # we don't bother re-reading it here — the next tick's entry-
            # read picks up the new id, and `alerted` / respawn dedup is
            # keyed on self-report-ts advancement rather than session id.
            # Clearing alerted so a future episode can re-alert if the new
            # session also stalls (the respawn_count keeps growing toward
            # the cap).
            None,
            missed=0,
            alerted=False,
            last_self_report_ts=ctx.last_self_report_ts,
            respawn_count=new_respawn_count if spawn_ok else ctx.respawn_count,
            exhausted=ctx.exhausted,
            refresh_attempted=ctx.refresh_attempted,
            followups_child_alerted=ctx.followups_child_alerted,
            prev=ctx.prev_state,
        )


def _handle_stalled_exhausted(ctx: _StalledActionCtx) -> None:
    """Recovery action: the crash-loop cap has been reached. Post a one-time
    loud marker, persist ``exhausted=True`` for dedup. Subsequent ticks
    stay quiet until real progress advances and clears the flag."""
    sid = ctx.happy_session_id_str
    if ctx.exhausted:
        if not ctx.dry_run:
            _save_stalled_state(
                ctx.issue,
                sid,
                missed=0,
                alerted=True,
                last_self_report_ts=ctx.last_self_report_ts,
                respawn_count=ctx.respawn_count,
                exhausted=True,
                refresh_attempted=ctx.refresh_attempted,
                followups_child_alerted=ctx.followups_child_alerted,
                prev=ctx.prev_state,
            )
        return
    _post_progress_marker(
        ctx.issue,
        f"{_STALLED_EXHAUSTED_NOTE_SENTINEL} AUTO-RECOVERY EXHAUSTED: the "
        f"stalled-detector auto-respawned this autonomous session "
        f"{ctx.respawn_count} time(s) in the current episode and the "
        f"workflow is STILL not advancing (self-report frozen for "
        f"{ctx.self_gap}, latest non-watcher progress marker "
        f"{ctx.marker_gap} old, has_pod={ctx.has_pod}, "
        f"status={ctx.task_status}). Likely a deterministically broken "
        f"session — open it and investigate manually. NOT auto-respawning "
        f"further; the next real progress marker on this task will reset "
        f"the cap.",
        ctx.dry_run,
        label="session-auto-respawn-exhausted",
    )
    if not ctx.dry_run:
        _save_stalled_state(
            ctx.issue,
            sid,
            missed=0,
            alerted=True,
            last_self_report_ts=ctx.last_self_report_ts,
            respawn_count=ctx.respawn_count,
            exhausted=True,
            refresh_attempted=ctx.refresh_attempted,
            followups_child_alerted=ctx.followups_child_alerted,
            prev=ctx.prev_state,
        )


def _handle_stalled_alert(ctx: _StalledActionCtx) -> None:
    """Recovery action: ALERT-ONLY fallback (respawn not eligible this tick:
    non-ACTIVE status or daemon unreachable). Identical surface to the
    Phase-1 ALERT-ONLY behavior, with one annotation line explaining WHY
    respawn was declined so the operator can address it.

    #488 stale-port self-heal: when the stalled session has a RUNNING
    managed pod whose name we know, AND we have NOT already fired the
    refresh-from-api auto-heal this episode, also fire ``pod.py config
    --refresh-from-api <pod_name>`` once. The refresh pulls the live
    host/port into ``pods.conf`` + ``~/.ssh/config``; if the staleness
    was caused by a port drift the next tick's SSH polling chain will
    self-recover. Fail-soft and dedup'd: one attempt per episode
    (``refresh_attempted`` flag, cleared on self-report advancement,
    same shape as ``alerted``)."""
    sid = ctx.happy_session_id_str
    if ctx.manual:
        reason = "manual user-driven session; alert-only by design"
    elif not ctx.in_active:
        reason = "task status not ACTIVE"
    else:
        reason = "Happy daemon unreachable; cannot stop+spawn"

    # #488 stale-port self-heal — see method docstring above. Skip when:
    # we already refreshed this episode; the pod name is unknown (no
    # endpoint to refresh); or has_pod=False (no live pod to refresh).
    new_refresh_attempted = ctx.refresh_attempted
    if ctx.has_pod and ctx.pod_name and not ctx.refresh_attempted:
        print(
            f"  REFRESH-FROM-API issue #{ctx.issue}: stalled session has "
            f"RUNNING pod {ctx.pod_name}; attempting #488 stale-port self-heal",
            file=sys.stderr,
        )
        _refresh_pods_conf_from_api(ctx.pod_name, ctx.dry_run)
        # Mark refreshed regardless of subprocess outcome — we don't want
        # to hot-loop refresh calls every tick on a pod whose endpoint is
        # genuinely the right one but whose SSH service is just down.
        # The flag clears on self-report advancement; a session that
        # stays stalled past that gets re-tried in the next episode.
        new_refresh_attempted = True

    if ctx.manual:
        # Manual entries are never liveness-checked by the respawn pass, so
        # the session may be fully dead (the #505 class), not just
        # alive-but-stalled — the prose must not claim it is in the live set.
        note = (
            f"{_STALLED_ALERT_NOTE_SENTINEL} STALLED manual issue session: "
            f"registered Happy session id={ctx.happy_session_id} (bare "
            f"`spawn-issue`, user-driven), but self-report has been frozen "
            f"for {ctx.self_gap} and the latest non-watcher progress marker "
            f"is {ctx.marker_gap} old (has_pod={ctx.has_pod}, "
            f"status={ctx.task_status}). The session is likely dead or its "
            f"bg-Bash chain died. NOT auto-respawned ({reason}); open the "
            f"session (phone / `spawn_session.py list`) and re-drive "
            f"`/issue {ctx.issue}` manually if confirmed dead. Confirmed "
            f"for >= {ctx.threshold} checks."
        )
    else:
        note = (
            f"{_STALLED_ALERT_NOTE_SENTINEL} ALIVE-BUT-STALLED autonomous "
            f"session: Happy session id={ctx.happy_session_id} is in the live "
            f"set, but self-report has been frozen for {ctx.self_gap} and the "
            f"latest non-watcher progress marker is {ctx.marker_gap} old "
            f"(has_pod={ctx.has_pod}, status={ctx.task_status}). Likely a dead "
            f"bg-Bash chain inside a still-live Claude process — the session "
            f"looks healthy to the respawn pass but is not advancing. NOT "
            f"auto-respawned ({reason}); investigate via the phone session "
            f"and stop+respawn manually if confirmed dead. Confirmed for >= "
            f"{ctx.threshold} checks."
        )
    _post_progress_marker(
        ctx.issue,
        note,
        ctx.dry_run,
        label="session-stalled-alert",
    )
    if not ctx.dry_run:
        _save_stalled_state(
            ctx.issue,
            sid,
            missed=0,
            alerted=True,
            last_self_report_ts=ctx.last_self_report_ts,
            respawn_count=ctx.respawn_count,
            exhausted=ctx.exhausted,
            refresh_attempted=new_refresh_attempted,
            followups_child_alerted=ctx.followups_child_alerted,
            prev=ctx.prev_state,
        )


def _apply_stalled_followups_exemption(
    *,
    issue: int,
    status: str | None,
    has_pod: bool,
    events: list[dict],
    action: str,
    new_missed: int,
    followups_child_alerted: bool,
    dry_run: bool,
) -> tuple[str, int, bool]:
    """Check the followups_running-parent-waiting-on-open-child exemption
    for the stalled-detector pass; rewrite ``(action, new_missed,
    followups_child_alerted)`` accordingly.

    No-op unless ``action != "keep" or new_missed > 0`` (so the healthy-
    session hot path never pays the ``task.py list-children`` subprocess).
    When the exemption fires, the action is rewritten to ``"keep"``,
    ``new_missed`` is reset to 0 (the exemption deliberately does NOT
    accumulate misses — the parent is correctly parked, not stalled), and
    a one-time alert marker is posted (dedup'd via
    ``followups_child_alerted``). Factored out of
    :func:`_process_stalled_session` to keep that function under the C901
    cyclomatic-complexity cap (15).
    """
    if action == "keep" and new_missed == 0:
        return action, new_missed, followups_child_alerted
    # Round-complete re-park (incident #533 freeze, 2026-06-11→12): a
    # COMPLETED same-issue follow-up round stranded at followups_running
    # (session died after the final gate, before the designed re-park) is
    # FIXED by executing the re-park — neither suppression (freeze) nor
    # respawn (each respawned session re-parked at step 10 and exited)
    # helps. Probed BEFORE the awaiting-child suppression so the freeze
    # shape (round-end markers + an open user-gated child) re-parks instead
    # of freezing. On re-park failure, fall through to the pre-existing
    # handling.
    if status == "followups_running" and not has_pod:
        repark_reason = _followup_round_complete_reason(events)
        if repark_reason is not None:
            print(
                f"  issue #{issue}: ALIVE-BUT-STALLED round-complete re-park — "
                f"{repark_reason} (would have been action={action})."
            )
            if _repark_completed_followup_round(issue, repark_reason, events, dry_run):
                return "keep", 0, followups_child_alerted
    followups_reason = _followups_awaiting_child_reason(
        issue, status=status, has_pod=has_pod, events=events
    )
    if followups_reason is None:
        return action, new_missed, followups_child_alerted
    print(
        f"  issue #{issue}: ALIVE-BUT-STALLED exemption — {followups_reason}; "
        f"treating session as live this tick (would have been action={action})."
    )
    if not followups_child_alerted:
        _post_progress_marker(
            issue,
            f"{_FOLLOWUPS_AWAITING_CHILD_NOTE_SENTINEL} {followups_reason}. "
            f"Respawn suppressed (does NOT consume the respawn budget); "
            f"re-invoke `/issue {issue}` after the open child(ren) reach "
            f"terminal status (`task.py promote <child> useful|not-useful` "
            f"for an awaiting_promotion child) to advance this parent.",
            dry_run,
            label="followups-awaiting-child",
        )
        followups_child_alerted = True
    return "keep", 0, followups_child_alerted


def _process_stalled_session(
    entry_path: Path,
    pod_active_issues: set[int],
    now: float,
    dry_run: bool,
    threshold: int,
    *,
    daemon_reachable: bool,
    pod_names_by_issue: dict[int, str] | None = None,
    manual: bool = False,
) -> None:
    """Reconcile one registry entry against the alive-but-stalled signals.

    Reads the issue's self-report ts + latest non-watcher marker ts + whether
    it has a RUNNING managed pod, applies :func:`decide_session_stalled`, and
    on a recovery action either auto-respawns (stop-then-spawn) the session
    or posts an alert / exhausted marker; otherwise persists state for the
    next tick.

    ``manual=True`` marks a manual registration (``manual-issue-<N>.json``,
    bare ``spawn-issue``): the same detection runs but ``respawn_eligible``
    is forced False, so the only possible recovery action is the one-time
    ALERT — a user-driven session is NEVER auto-respawned (#505 round-2
    orphaning, 2026-06-10).

    ``daemon_reachable`` is computed once per pass (the watcher already
    probes it for the crash-recovery pass) and passed in so we don't
    re-probe per-entry. AUTO-RESPAWN requires the daemon (both
    ``spawn_session.py stop`` and ``spawn-issue --auto`` POST to the local
    daemon RPC); when it is unreachable, this pass falls back to
    ALERT-ONLY for stalled entries — mirrors the crash-recovery pass's
    same-tick degradation.
    """
    try:
        entry = json.loads(entry_path.read_text())
    except (json.JSONDecodeError, OSError):
        # Cleanup is owned elsewhere: the respawn pass removes a garbled
        # autonomous entry; the GC pass reaps manual entries (keyed on the
        # filename's issue number, so a garbled BODY still gets aged out).
        # We just skip on this pass.
        return
    issue = entry.get("issue")
    if not isinstance(issue, int):
        return

    happy_session_id = entry.get("happy_session_id")

    # Signal 1: self-report age. None -> skip (autonomous sessions are
    # expected to self-report; a missing file is treated as "this pass
    # doesn't apply" rather than a stale signal that could over-alert).
    self_report_age, last_self_report_ts = _self_report_age_seconds(issue, now)

    # Signal 2: latest non-watcher progress-marker age. None -> stale (no
    # markers at all is itself a signal). We also keep the raw events list
    # around — the followups-awaiting-child exemption below scans it for
    # the latest epm:step-completed without paying a second read.
    events = _task_events(issue)
    latest_marker_ts = _latest_progress_ts(events)
    marker_age = (now - latest_marker_ts) if latest_marker_ts is not None else None

    # Signal 3: does the issue have a RUNNING managed pod? Informational
    # at the decision layer (signal 2 covers pod-state markers posted by
    # poll_pipeline.py), but logged so a stalled session WITH a live pod is
    # visibly distinguishable from one WITHOUT.
    has_pod = issue in pod_active_issues

    prev_state = _load_stalled_state(issue)
    prev_missed = prev_state.get("missed", 0)
    if not isinstance(prev_missed, int):
        prev_missed = 0
    prev_alerted = bool(prev_state.get("alerted", False))
    prev_respawn_count = prev_state.get("respawn_count", 0)
    if not isinstance(prev_respawn_count, int):
        prev_respawn_count = 0
    prev_exhausted = bool(prev_state.get("exhausted", False))
    prev_refresh_attempted = bool(prev_state.get("refresh_attempted", False))
    prev_followups_child_alerted = bool(prev_state.get("followups_child_alerted", False))
    prev_last_self_report_ts = prev_state.get("last_self_report_ts")
    if not isinstance(prev_last_self_report_ts, str):
        prev_last_self_report_ts = None

    # Clear `alerted` + `respawn_count` + `exhausted` + `refresh_attempted`
    # + `followups_child_alerted` whenever the self-report ts has ADVANCED
    # since the last save — that means the session resumed self-reporting,
    # so the prior episode is over and a future staleness episode can
    # re-alert / re-respawn / re-refresh. Comparison is on the raw ISO
    # string (lexicographic on the canonical trailing-Z UTC format is
    # monotonic).
    self_report_advanced = (
        last_self_report_ts is not None
        and prev_last_self_report_ts is not None
        and last_self_report_ts > prev_last_self_report_ts
    )
    if self_report_advanced:
        alerted = False
        respawn_count = 0
        exhausted = False
        refresh_attempted = False
        followups_child_alerted = False
    else:
        alerted = prev_alerted
        respawn_count = prev_respawn_count
        exhausted = prev_exhausted
        refresh_attempted = prev_refresh_attempted
        followups_child_alerted = prev_followups_child_alerted

    # Compute respawn_eligible: the task must be in an ACTIVE status (we
    # never restart a session at a PARK / gate / terminal state) AND the
    # Happy daemon must be reachable (we can't issue stop+spawn without
    # it). Both inputs are I/O — kept here in the actor, not in the pure
    # decision function. Manual (user-driven) registrations are NEVER
    # respawn-eligible: forcing False routes decide_session_stalled to the
    # ALERT-ONLY arm (one alert per episode, no respawn / exhausted
    # escalation) regardless of task status or daemon state — restarting a
    # session the user drives by hand is not the watcher's call (#505
    # round-2 orphaning, 2026-06-10).
    task_status = _task_status(issue)
    in_active = task_status in ACTIVE
    respawn_eligible = in_active and daemon_reachable and not manual

    action, new_missed = decide_session_stalled(
        self_report_age_s=self_report_age,
        marker_progress_age_s=marker_age,
        has_pod=has_pod,
        missed=prev_missed,
        alerted=alerted,
        respawn_eligible=respawn_eligible,
        respawn_count=respawn_count,
        threshold=threshold,
    )

    # In-flight-provision exemption (refs #573): a provision waiting for
    # capacity blocks the session's bg-Bash chain, freezing BOTH staleness
    # signals while being exactly the work the session should be doing —
    # #534's auto-respawn killed an in-flight provision 3x (~8h lost).
    # Probed LAZILY (only when decide() wants to escalate or accumulate a
    # miss) so the healthy-session hot path never pays the /proc scan.
    if action != "keep" or new_missed > 0:
        exempt_reason = _provision_in_flight_reason(issue, now)
        if exempt_reason is not None:
            print(
                f"  issue #{issue}: ALIVE-BUT-STALLED exemption — {exempt_reason}; "
                f"treating session as live this tick (would have been "
                f"action={action})."
            )
            action, new_missed = ("keep", 0)

    # followups_running parent-waiting-on-open-child exemption (incident
    # #533, 2026-06-11): a parent whose own pipeline is parked at step 10
    # awaiting a user-gated child cannot be unblocked by respawning the
    # parent session — only user action on the child unblocks it. See the
    # comment block above ``_followups_awaiting_child_reason`` for the
    # full predicate. Helper factored out to keep this function under the
    # C901 cap; returns the (possibly rewritten) (action, new_missed,
    # followups_child_alerted) tuple.
    action, new_missed, followups_child_alerted = _apply_stalled_followups_exemption(
        issue=issue,
        status=task_status,
        has_pod=has_pod,
        events=events,
        action=action,
        new_missed=new_missed,
        followups_child_alerted=followups_child_alerted,
        dry_run=dry_run,
    )

    self_gap = f"{self_report_age / 60:.1f}m" if self_report_age is not None else "none"
    marker_gap = f"{marker_age / 60:.1f}m" if marker_age is not None else "none"
    print(
        f"  issue #{issue}: status={task_status} self_gap={self_gap} "
        f"marker_gap={marker_gap} has_pod={has_pod} "
        f"missed={prev_missed}->{new_missed} alerted={alerted} "
        f"respawn_count={respawn_count}/{STALLED_MAX_RESPAWNS} "
        f"daemon_reachable={daemon_reachable} manual={manual} "
        f"followups_child_alerted={followups_child_alerted} action={action}"
    )

    pod_name = (pod_names_by_issue or {}).get(issue)
    ctx = _StalledActionCtx(
        issue=issue,
        happy_session_id=happy_session_id,
        prev_state=prev_state,
        alerted=alerted,
        respawn_count=respawn_count,
        exhausted=exhausted,
        last_self_report_ts=last_self_report_ts,
        self_gap=self_gap,
        marker_gap=marker_gap,
        has_pod=has_pod,
        task_status=task_status,
        in_active=in_active,
        threshold=threshold,
        dry_run=dry_run,
        refresh_attempted=refresh_attempted,
        pod_name=pod_name,
        manual=manual,
        followups_child_alerted=followups_child_alerted,
    )

    if action == "respawn":
        _handle_stalled_respawn(ctx)
        return
    if action == "exhausted":
        _handle_stalled_exhausted(ctx)
        return
    if action == "alert":
        _handle_stalled_alert(ctx)
        return

    # action == "keep": persist the (possibly incremented) miss count + the
    # alerted / respawn_count / exhausted / refresh_attempted /
    # followups_child_alerted flags (cleared above if self-report advanced)
    # + the latest observed self-report ts so the next tick can detect
    # advancement.
    if not dry_run:
        _save_stalled_state(
            issue,
            happy_session_id if isinstance(happy_session_id, str) else None,
            missed=new_missed,
            alerted=alerted,
            last_self_report_ts=last_self_report_ts,
            respawn_count=respawn_count,
            exhausted=exhausted,
            refresh_attempted=refresh_attempted,
            followups_child_alerted=followups_child_alerted,
            prev=prev_state,
        )


def stalled_session_pass(
    dry_run: bool,
    threshold: int,
    now: float | None = None,
    *,
    daemon_reachable: bool | None = None,
) -> None:
    """Detect alive-but-stalled issue sessions and recover or alert.

    Autonomous-registry entries (``issue-<N>.json``) are auto-respawned
    (when the task is ACTIVE and the Happy daemon is reachable) or fall
    back to a one-time loud alert. Manual entries
    (``manual-issue-<N>.json``, written by bare ``spawn-issue``) get the
    SAME staleness detection in ALERT-ONLY mode: a dead or stalled
    user-driven session at an ACTIVE status raises the one-time alert
    instead of orphaning silently, but is NEVER auto-respawned —
    restarting a session the user drives by hand is the user's call
    (#505 round-2 orphaning, 2026-06-10). When an issue carries BOTH
    registrations, the autonomous entry wins and the manual one is
    skipped: both would share the same ``stalled-<N>.json`` state file,
    and double-processing in one tick would defeat the 2-miss guard.

    ``daemon_reachable`` is the same flag the crash-recovery pass uses; the
    caller probes it once per :func:`main` invocation. When not passed,
    we probe here so the function still works in unit tests / debug runs
    that call it directly.
    """
    now = now if now is not None else time.time()
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        print("stalled-detector: no autonomous registry dir; skipping")
        return
    entries = sorted(AUTONOMOUS_REGISTRY_DIR.glob("issue-*.json"))
    manual_entries = sorted(AUTONOMOUS_REGISTRY_DIR.glob("manual-issue-*.json"))
    if not entries and not manual_entries:
        print("stalled-detector: no issue sessions registered")
        return
    # Resolve which issues currently have a RUNNING managed pod once per tick.
    # A FAILED snapshot (None — the helper already logs to stderr) degrades to
    # the empty set so the decision layer just records has_pod=False for every
    # issue this tick — fail-safe (this pass alerts/respawns, never stops pods).
    running_pods = _running_managed_issue_pods(caller="stalled-detector") or []
    pod_active_issues = {issue for issue, _pid, _name in running_pods}
    pod_names_by_issue = {issue: name for issue, _pid, name in running_pods}
    if daemon_reachable is None:
        daemon_reachable = _daemon_reachable()
    print(
        f"stalled-detector: {len(entries)} autonomous + {len(manual_entries)} "
        f"manual session(s) (daemon_reachable={daemon_reachable})"
    )
    for path in entries:
        _process_stalled_session(
            path,
            pod_active_issues,
            now,
            dry_run,
            threshold,
            daemon_reachable=daemon_reachable,
            pod_names_by_issue=pod_names_by_issue,
        )
    # Manual entries: ALERT-ONLY (never auto-respawn a user-driven session;
    # #505 round-2, 2026-06-10). Skip any issue already covered by an
    # autonomous entry this tick — both kinds share ``stalled-<N>.json``,
    # so a second processing in the same tick would double-increment the
    # 2-miss guard; the autonomous entry's coverage is the stronger one.
    auto_issues = {
        n for n in (_gc_parse_issue_from_path(p, "issue-", "") for p in entries) if n is not None
    }
    for path in manual_entries:
        manual_issue = _gc_parse_issue_from_path(path, "manual-issue-", "")
        if manual_issue is not None and manual_issue in auto_issues:
            print(
                f"  manual-issue-{manual_issue}: autonomous entry exists for "
                f"the same issue; skipping (autonomous coverage wins)"
            )
            continue
        _process_stalled_session(
            path,
            pod_active_issues,
            now,
            dry_run,
            threshold,
            daemon_reachable=daemon_reachable,
            pod_names_by_issue=pod_names_by_issue,
            manual=True,
        )


# ─── orphan sweep (registration-INDEPENDENT safety net) ─────────────────────
#
# Every other session pass starts from the registry files, so an ACTIVE-status
# task with NO registration is invisible to all of them. Incident 2026-06-10
# (#472): the task parked at `awaiting_promotion` (TERMINAL → the respawn pass
# DELETED its `issue-472.json` per `decide`), a same-issue follow-up later
# flipped it back to `running` driven by an unregistered interactive session,
# that session died at 08:40Z, and the task sat orphaned for 10.5h until
# manual PM triage. This pass inverts the direction: enumerate ACTIVE-status
# tasks and ask "is anything registered AND live driving this?".

# How long an orphan-candidate task may go without a real progress marker
# before the sweep acts. Deliberately tighter than ALERT_STALE_HOURS (the
# pod-safety alert arm) because the respawn here is cheap and idempotent
# (`/issue` resumes from markers); env-overridable for tuning without a
# code change.
ORPHAN_STALENESS_S_DEFAULT = 90 * 60

# Grace window after a registration write during which the task is treated as
# "spawn in flight" even if the recorded id is not yet in the daemon's live
# set. Covers the same-tick race where the respawn pass (or a manual
# recovery) just rewrote the registry but the live-id snapshot predates it.
ORPHAN_SPAWN_GRACE_S = 15 * 60

# Maximum respawn ATTEMPTS (successes AND failures both count, so a
# deterministically failing spawn can't hot-loop) per task per UTC day.
ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT = 2

# Filename prefix for the per-issue orphan-sweep state file at
# ``~/.eps-autonomous/orphan-<N>.json``. Mirrors the stalled / pod-safety
# state-file layout; reaped by the generalized GC.
ORPHAN_STATE_PREFIX = "orphan-"


def _orphan_staleness_s() -> float:
    """Marker-staleness threshold in seconds (env ``EPM_ORPHAN_STALENESS_MIN``,
    minutes; default :data:`ORPHAN_STALENESS_S_DEFAULT`). A malformed env value
    falls back to the default — a typo'd var must not disable crash recovery."""
    raw = os.environ.get("EPM_ORPHAN_STALENESS_MIN")
    if not raw:
        return float(ORPHAN_STALENESS_S_DEFAULT)
    try:
        return float(raw) * 60.0
    except ValueError:
        return float(ORPHAN_STALENESS_S_DEFAULT)


def _orphan_max_respawns_per_day() -> int:
    """Daily per-task respawn-attempt cap (env ``EPM_ORPHAN_RESPAWNS_PER_DAY``;
    default :data:`ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT`). Malformed env value
    falls back to the default."""
    raw = os.environ.get("EPM_ORPHAN_RESPAWNS_PER_DAY")
    if not raw:
        return ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT


def decide_orphan(
    status: str | None,
    mapped_alive: bool,
    manual_only: bool,
    entry_age_s: float | None,
    marker_age_s: float | None,
    missed: int,
    *,
    respawns_today: int = 0,
    threshold: int = 2,
    staleness_s: float = ORPHAN_STALENESS_S_DEFAULT,
    spawn_grace_s: float = ORPHAN_SPAWN_GRACE_S,
    max_respawns_per_day: int = ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT,
) -> tuple[str, int]:
    """Pure decision for the orphan sweep: ``(action, new_missed)`` where
    action is ``"clear"`` | ``"keep"`` | ``"respawn"`` | ``"alert"``.

    - ``clear``: the task is not orphanable (not ACTIVE, or a registered
      session is live) — the caller drops any accumulated state.
    - ``keep``: orphan-candidate but not actionable yet (registration freshly
      written / markers still fresh / miss count accumulating).
    - ``respawn``: ACTIVE + no live registered session + markers stale on
      ``threshold`` consecutive checks, respawn budget available.
    - ``alert``: same trigger, but the task's only registration is MANUAL
      (user-driven sessions are never auto-respawned, #505) or the daily
      attempt cap is exhausted — the caller posts a one-time loud marker.

    ``marker_age_s is None`` (no real progress marker at all) counts as
    stale — an ACTIVE task with zero progress markers is itself the signal
    (mirrors the pod-safety pass's None-is-stale rule)."""
    if status not in ACTIVE:
        return ("clear", 0)
    if mapped_alive:
        return ("clear", 0)
    if entry_age_s is not None and entry_age_s < spawn_grace_s:
        return ("keep", 0)
    if marker_age_s is not None and marker_age_s < staleness_s:
        return ("keep", 0)
    new_missed = missed + 1
    if new_missed < threshold:
        return ("keep", new_missed)
    if manual_only:
        return ("alert", new_missed)
    if respawns_today >= max_respawns_per_day:
        return ("alert", new_missed)
    return ("respawn", 0)


def _orphan_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{ORPHAN_STATE_PREFIX}{issue}.json"


def _load_orphan_state(issue: int) -> dict:
    """Read the per-issue orphan-sweep state (``{}`` if absent / unreadable —
    a fresh/garbled file starts the miss count at 0, mirroring
    :func:`_load_stalled_state`)."""
    path = _orphan_state_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_orphan_state(
    issue: int,
    *,
    missed: int,
    alerted: bool,
    respawn_day: str,
    respawns_today: int,
    followups_child_alerted: bool = False,
    prev: dict | None = None,
) -> None:
    """Persist the per-issue orphan-sweep state atomically (temp + rename),
    mirroring :func:`_save_stalled_state`. ``respawn_day`` + ``respawns_today``
    implement the per-UTC-day attempt cap; ``alerted`` dedups the one-time
    alert marker within an episode; ``followups_child_alerted`` dedups the
    one-time "followups_running parent waiting on open child" suppression
    alert (see :func:`_followups_awaiting_child_reason`); ``first_seen``
    carries forward so the GC age backstop measures the original episode
    start."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _orphan_state_path(issue)
    prev_first_seen = (prev or {}).get("first_seen")
    if not isinstance(prev_first_seen, int | float):
        prev_first_seen = time.time()
    payload = {
        "missed": missed,
        "alerted": alerted,
        "respawn_day": respawn_day,
        "respawns_today": respawns_today,
        "followups_child_alerted": followups_child_alerted,
        "first_seen": prev_first_seen,
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_orphan_state(issue: int) -> None:
    """Drop the per-issue orphan-sweep state file (episode over: the task left
    ACTIVE or a registered session went live again)."""
    _orphan_state_path(issue).unlink(missing_ok=True)


def _active_status_tasks() -> dict[int, str]:
    """``{issue: status}`` for every task currently in an :data:`ACTIVE`
    status, via ``task.py list-by-status --json`` (one subprocess per status;
    same fail-soft isolation as :func:`_task_status` — a read failure for one
    status just yields no candidates from it this tick, never a crash)."""
    out: dict[int, str] = {}
    for status in sorted(ACTIVE):
        try:
            res = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/task.py",
                    "list-by-status",
                    "--status",
                    status,
                    "--json",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        if res.returncode != 0:
            continue
        try:
            rows = json.loads(res.stdout)
        except json.JSONDecodeError:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            # `kind: campaign` tasks are owned by the campaign pass — the
            # orphan sweep's recovery command is `spawn-issue --auto`, which
            # would boot the WRONG skill (/issue) on a campaign (task #586).
            if row.get("kind") == "campaign":
                continue
            tid = row.get("id")
            if isinstance(tid, int):
                out[tid] = status
    return out


def _issue_registrations() -> dict[int, dict]:
    """Scan BOTH registry prefixes and return per-issue registration facts:
    ``{issue: {"sids": set[str], "has_auto": bool, "has_manual": bool,
    "newest_write": float}}``. ``newest_write`` is the newest of file mtime
    and the entry's ``spawned_at`` — used for the spawn-grace window."""
    out: dict[int, dict] = {}
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return out
    for prefix, manual in (("issue-", False), ("manual-issue-", True)):
        for path in AUTONOMOUS_REGISTRY_DIR.glob(f"{prefix}*.json"):
            issue = _gc_parse_issue_from_path(path, prefix, "")
            if issue is None:
                continue
            try:
                entry = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                entry = {}
            if not isinstance(entry, dict):
                entry = {}
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            spawned_at = entry.get("spawned_at")
            if not isinstance(spawned_at, int | float):
                spawned_at = 0.0
            rec = out.setdefault(
                issue,
                {"sids": set(), "has_auto": False, "has_manual": False, "newest_write": 0.0},
            )
            sid = entry.get("happy_session_id")
            if isinstance(sid, str) and sid:
                rec["sids"].add(sid)
            rec["has_auto"] = rec["has_auto"] or not manual
            rec["has_manual"] = rec["has_manual"] or manual
            rec["newest_write"] = max(rec["newest_write"], mtime, float(spawned_at))
    return out


def _respawn_orphan(issue: int, cap_gpu_hours: float, dry_run: bool) -> bool:
    """Spawn a fresh ``--auto`` session for an orphaned active task. Mirrors
    :func:`_respawn_stalled_session` but with an ``RESPAWNED-ORPHAN`` log
    prefix so the operator can tell the recovery paths apart. The spawn
    re-registers the issue (``spawn-issue --auto`` rewrites the registry), so
    the task re-enters normal respawn/stalled coverage."""
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto", "--auto-approve-gpu-hours", str(cap_gpu_hours),
    ]  # fmt: skip
    cmd.extend(_stalled_session_overrides(issue))
    if dry_run:
        print(f"  [dry-run] would respawn orphan: {' '.join(cmd)}")
        return False
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(
            f"  RESPAWN-ORPHAN FAILED issue #{issue}: {res.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  RESPAWNED-ORPHAN issue #{issue} (active task, no live session): {first_line}")
    return True


def orphan_sweep_pass(
    dry_run: bool,
    threshold: int,
    now: float | None = None,
    *,
    daemon_reachable: bool | None = None,
    live_ids: set[str] | None = None,
) -> None:
    """Registration-independent safety net: cross-check ACTIVE-status tasks
    against live REGISTERED sessions; recover (or loudly alert on) any active
    task nothing is driving.

    Liveness here is deliberately REGISTRATION-KEYED ONLY (autonomous +
    manual entry ids vs the daemon's live set) — no worktree-cwd heuristic
    (see :func:`_session_alive` for why that signal lies) and no self-report
    freshness (a superseded driver generation kept #518's self-report fresh
    for 7.4h of real marker silence on 2026-06-10). Daemon-gated like the
    respawn pass: during an outage liveness is unknowable and a mass respawn
    would duplicate pods."""
    now = now if now is not None else time.time()
    if daemon_reachable is None:
        daemon_reachable = _daemon_reachable()
    if not daemon_reachable:
        print(
            "orphan-sweep: Happy daemon unreachable; skipping (liveness "
            "unknowable; a mass respawn on an outage would duplicate pods)"
        )
        return
    if live_ids is None:
        live_ids = _live_session_ids()
    active = _active_status_tasks()
    regs = _issue_registrations()
    staleness_s = _orphan_staleness_s()
    max_per_day = _orphan_max_respawns_per_day()
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    # Snapshot RUNNING managed pods once per tick — feeds the
    # followups_running-parent-waiting-on-open-child exemption's
    # has_pod=False precondition. Fail-safe: a FAILED snapshot (None,
    # already logged to stderr) degrades to the empty set so the
    # exemption simply records has_pod=False for every issue this tick;
    # a same-issue follow-up round with a live pod posts its own
    # ``epm:run-launched`` / progress markers which already keep
    # marker_age_s below the staleness threshold, so the orphan sweep
    # would not be at action=respawn anyway.
    running_pods = _running_managed_issue_pods(caller="orphan-sweep") or []
    pod_active_issues = {issue for issue, _pid, _name in running_pods}
    print(
        f"orphan-sweep: {len(active)} active-status task(s), "
        f"{len(regs)} registered issue(s), {len(live_ids)} live session(s)"
    )
    for issue in sorted(active):
        _process_orphan_task(
            issue,
            active[issue],
            regs.get(issue),
            live_ids,
            now,
            dry_run,
            threshold,
            staleness_s=staleness_s,
            max_per_day=max_per_day,
            day_key=day_key,
            pod_active_issues=pod_active_issues,
        )


def _check_orphan_followups_exemption(
    *,
    issue: int,
    status: str,
    has_pod: bool,
    events: list[dict],
    action: str,
) -> tuple[str, str | None]:
    """Probe the followups_running-parent-waiting-on-open-child exemption
    for the orphan-sweep pass. Returns the (possibly rewritten) ``action``
    plus the human-readable reason string (for the alert prose) or
    ``None`` when the exemption does not apply.

    No-op unless ``action == "respawn"`` (the only orphan action whose
    fallout is wasteful in this regime) so a healthy task / a manual-only
    or cap-exhausted task never pays the ``task.py list-children``
    subprocess. Factored out of :func:`_process_orphan_task` to keep
    that function under the C901 cap.
    """
    if action != "respawn":
        return action, None
    # Round-complete re-park (incident #533 freeze): probed BEFORE the
    # awaiting-child suppression — a completed same-issue follow-up round
    # stranded at followups_running is fixed by executing the re-park, not
    # by suppressing or respawning. The actual mutation happens in
    # :func:`_handle_orphan_followup_round_repark` (this helper stays
    # read-only, mirroring the awaiting-child probe).
    if status == "followups_running" and not has_pod:
        repark_reason = _followup_round_complete_reason(events)
        if repark_reason is not None:
            print(
                f"  issue #{issue}: ORPHAN-RESPAWN round-complete re-park — "
                f"{repark_reason}; executing the re-park instead of a respawn."
            )
            return "followup-round-repark", repark_reason
    followups_reason = _followups_awaiting_child_reason(
        issue, status=status, has_pod=has_pod, events=events
    )
    if followups_reason is None:
        return action, None
    print(
        f"  issue #{issue}: ORPHAN-RESPAWN exemption — {followups_reason}; "
        f"diverting to alert-only (does NOT consume respawn budget)."
    )
    return "followups-awaiting-child", followups_reason


def _handle_orphan_followup_round_repark(
    *,
    issue: int,
    reason: str,
    events: list[dict],
    new_missed: int,
    alerted: bool,
    respawn_day: str,
    respawns_today: int,
    followups_child_alerted: bool,
    state: dict,
    dry_run: bool,
) -> None:
    """Orphan-sweep handler for the round-complete re-park (incident #533
    freeze class): execute the stranded re-park; on success reset the miss
    counter (the task leaves ACTIVE and drops out of the sweep next tick),
    on failure persist ``new_missed`` as-is — in production that is the 0
    from ``decide_orphan``'s respawn decision, so the orphan pass re-probes
    and retries the re-park once the staleness signals re-accumulate to the
    respawn action (~2 ticks), rather than next tick. Never consumes the
    daily respawn budget. (The stalled pass falls back same-tick to the
    awaiting-child suppression on a failed re-park; this pass retries.)
    Factored out of :func:`_process_orphan_task` to keep that function
    under the C901 cyclomatic-complexity cap (15)."""
    ok = _repark_completed_followup_round(issue, reason, events, dry_run)
    if not dry_run:
        _save_orphan_state(
            issue,
            missed=0 if ok else new_missed,
            alerted=alerted,
            respawn_day=respawn_day,
            respawns_today=respawns_today,
            followups_child_alerted=followups_child_alerted,
            prev=state,
        )


def _handle_orphan_followups_awaiting_child(
    *,
    issue: int,
    reason: str,
    followups_child_alerted: bool,
    new_missed: int,
    alerted: bool,
    respawn_day: str,
    respawns_today: int,
    state: dict,
    dry_run: bool,
) -> None:
    """Orphan-sweep handler for the followups_running-parent-waiting-on-
    open-child exemption: post the one-time alert (dedup'd via
    ``followups_child_alerted``) and persist state WITHOUT incrementing
    ``respawns_today`` — the exemption deliberately does NOT consume the
    daily respawn budget. The dedup flag clears on the natural episode
    end (the sweep's ``action == "clear"`` branch, which fires when the
    task leaves ACTIVE — typically once all children reach terminal and
    the user re-drives the parent via ``/issue <N>``). Factored out of
    :func:`_process_orphan_task` to keep that function under the C901
    cyclomatic-complexity cap (15)."""
    if not followups_child_alerted:
        _post_progress_marker(
            issue,
            f"{_FOLLOWUPS_AWAITING_CHILD_NOTE_SENTINEL} {reason}. "
            f"Orphan-respawn suppressed (does NOT consume the daily respawn "
            f"budget); re-invoke `/issue {issue}` after the open child(ren) "
            f"reach terminal status (`task.py promote <child> useful|"
            f"not-useful` for an awaiting_promotion child) to advance this "
            f"parent.",
            dry_run,
            label="followups-awaiting-child",
        )
    if not dry_run:
        _save_orphan_state(
            issue,
            missed=new_missed,
            alerted=alerted,
            respawn_day=respawn_day,
            respawns_today=respawns_today,
            followups_child_alerted=True,
            prev=state,
        )


def _process_orphan_task(
    issue: int,
    status: str,
    rec: dict | None,
    live_ids: set[str],
    now: float,
    dry_run: bool,
    threshold: int,
    *,
    staleness_s: float,
    max_per_day: int,
    day_key: str,
    pod_active_issues: set[int] | None = None,
) -> None:
    """Apply one active-status task's orphan decision (gather signals ->
    :func:`decide_orphan` -> act). ``rec`` is the task's registration record
    from :func:`_issue_registrations` (or ``None`` for the fully-unregistered
    #472 class). Honours dry_run (logs but never mutates / spawns)."""
    mapped_alive = bool(rec and rec["sids"] & live_ids)
    manual_only = bool(rec and rec["has_manual"] and not rec["has_auto"])
    entry_age_s = (now - rec["newest_write"]) if rec and rec["newest_write"] > 0 else None
    state = _load_orphan_state(issue)
    missed = state.get("missed", 0)
    if not isinstance(missed, int):
        missed = 0
    respawns_today = state.get("respawns_today", 0) if state.get("respawn_day") == day_key else 0
    if not isinstance(respawns_today, int):
        respawns_today = 0
    alerted = bool(state.get("alerted"))
    followups_child_alerted = bool(state.get("followups_child_alerted"))

    # Lazy events fetch: only orphan candidates pay the per-task read.
    # The events list is reused below by the followups-awaiting-child
    # exemption helper so we don't pay a second `task.py view` per tick.
    marker_age_s: float | None = None
    events: list[dict] = []
    is_candidate = not mapped_alive and not (
        entry_age_s is not None and entry_age_s < ORPHAN_SPAWN_GRACE_S
    )
    if is_candidate:
        events = _task_events(issue)
        latest = _latest_progress_ts(events)
        marker_age_s = (now - latest) if latest is not None else None

    action, new_missed = decide_orphan(
        status,
        mapped_alive,
        manual_only,
        entry_age_s,
        marker_age_s,
        missed,
        respawns_today=respawns_today,
        threshold=threshold,
        staleness_s=staleness_s,
        max_respawns_per_day=max_per_day,
    )
    gap_str = f"{marker_age_s / 60:.1f}m" if marker_age_s is not None else "none"

    # followups_running parent-waiting-on-open-child exemption (incident
    # #533, 2026-06-11): mirror of the same exemption in
    # :func:`_process_stalled_session`. When the orphan sweep would
    # respawn a `followups_running` parent whose `/issue` pipeline is
    # parked at step 10 awaiting a user-gated child, the respawn cannot
    # advance the task — divert to a one-time alert marker that does NOT
    # consume the daily respawn budget. Helper-factored to keep this
    # function under the C901 cap.
    has_pod_for_followups = bool(pod_active_issues and issue in pod_active_issues)
    action, followups_reason = _check_orphan_followups_exemption(
        issue=issue,
        status=status,
        has_pod=has_pod_for_followups,
        events=events,
        action=action,
    )

    print(
        f"  issue #{issue}: status={status} mapped_alive={mapped_alive} "
        f"manual_only={manual_only} marker_gap={gap_str} "
        f"missed={missed}->{new_missed} respawns_today={respawns_today}/{max_per_day} "
        f"alerted={alerted} followups_child_alerted={followups_child_alerted} "
        f"action={action}"
    )

    if action == "clear":
        if state and not dry_run:
            _clear_orphan_state(issue)
        return
    if action == "keep":
        if not dry_run:
            _save_orphan_state(
                issue,
                missed=new_missed,
                alerted=alerted,
                respawn_day=day_key,
                respawns_today=respawns_today,
                followups_child_alerted=followups_child_alerted,
                prev=state,
            )
        return
    if action == "respawn":
        attempted_ok = _respawn_orphan(issue, _stalled_cap_gpu_hours(issue), dry_run)
        if not dry_run:
            # Count the ATTEMPT regardless of success so a failing spawn
            # can't hot-loop past the daily cap.
            _save_orphan_state(
                issue,
                missed=0,
                alerted=False,
                respawn_day=day_key,
                respawns_today=respawns_today + 1,
                followups_child_alerted=followups_child_alerted,
                prev=state,
            )
            if attempted_ok:
                _post_progress_marker(
                    issue,
                    f"{_ORPHAN_RESPAWN_NOTE_SENTINEL} active task "
                    f"(status={status}) had no live registered session and no "
                    f"real progress marker for {gap_str}; auto-respawned via "
                    f"spawn-issue --auto (attempt {respawns_today + 1}/{max_per_day} "
                    f"today).",
                    dry_run,
                    label="orphan-respawn",
                )
        return
    if action == "followup-round-repark":
        _handle_orphan_followup_round_repark(
            issue=issue,
            reason=followups_reason or "",
            events=events,
            new_missed=new_missed,
            alerted=alerted,
            respawn_day=day_key,
            respawns_today=respawns_today,
            followups_child_alerted=followups_child_alerted,
            state=state,
            dry_run=dry_run,
        )
        return
    if action == "followups-awaiting-child":
        _handle_orphan_followups_awaiting_child(
            issue=issue,
            reason=followups_reason,
            followups_child_alerted=followups_child_alerted,
            new_missed=new_missed,
            alerted=alerted,
            respawn_day=day_key,
            respawns_today=respawns_today,
            state=state,
            dry_run=dry_run,
        )
        return
    # action == "alert": one-time loud marker per episode.
    reason = (
        "only a MANUAL (user-driven) session is registered; never auto-respawned"
        if manual_only
        else f"daily respawn-attempt cap exhausted ({respawns_today}/{max_per_day})"
    )
    print(
        f"  ORPHANED issue #{issue}: status={status}, no live registered "
        f"session, marker_gap={gap_str}; {reason}",
        file=sys.stderr,
    )
    if not alerted:
        _post_progress_marker(
            issue,
            f"{_ORPHAN_ALERT_NOTE_SENTINEL} active task (status={status}) has "
            f"no live registered session and no real progress marker for "
            f"{gap_str}; {reason}. Manual recovery: uv run python "
            f"scripts/spawn_session.py spawn-issue --issue {issue} --auto",
            dry_run,
            label="orphan-alert",
        )
    if not dry_run:
        _save_orphan_state(
            issue,
            missed=new_missed,
            alerted=True,
            respawn_day=day_key,
            respawns_today=respawns_today,
            followups_child_alerted=followups_child_alerted,
            prev=state,
        )


# ─── infra-drain pass (execute the PM-adjudicated dispatch queue; #633) ──────
#
# The PM session's standing infra auto-dispatch rule (research-pm.md
# § Standing rule — infra auto-dispatch) adjudicates which `proposed`
# kind:infra/batch tasks are RIPE and writes them, oldest-first, to
# ``~/.eps-autonomous/infra-drain-queue.json``. This pass EXECUTES that file:
# with zero LLM judgment it spawns ``spawn-issue --auto`` sessions for queue
# IDs still at ``proposed``, into free slots under the cap, with per-ID
# guards (holds, existing registration, status, kind, retry backoff) so a
# held / already-running / repeatedly-failing / mis-kinded ID is never
# dispatched or tight-looped. The PM remains the ONLY ripeness judge; a
# missing/empty/invalid queue file is a logged no-op. Durably replaces the
# PM-session-scoped hourly cron stopgap (which dies with the PM session).

# Basenames under AUTONOMOUS_REGISTRY_DIR. Resolved via path FUNCTIONS (the
# `_vm_disk_state_path` pattern) so the test fixture's AUTONOMOUS_REGISTRY_DIR
# monkeypatch isolates them; neither matches any existing watcher/GC/spawn
# glob, and the state file self-prunes to the queue's ID set, so it is
# deliberately NOT in _GC_TARGETS.
INFRA_DRAIN_QUEUE_BASENAME = "infra-drain-queue.json"
INFRA_DRAIN_STATE_BASENAME = "infra-drain-state.json"
# Used only when the queue file omits `cap` (the body names 3 as the schema
# default; a benign omission must not silently disable the drain).
INFRA_DRAIN_CAP_DEFAULT = 3
# Task kinds the drain may dispatch. A mis-queued experiment/campaign ID must
# never be spawned with --auto: it would auto-approve <=100 GPU-h AND sit
# outside this pass's cap arithmetic.
INFRA_DRAIN_KINDS = frozenset({"infra", "batch"})
# Statuses that occupy a drain slot: the task-#633 contract set PLUS
# followups_running (a same-issue follow-up round is in-flight work holding a
# session + possibly a pod; counting it only ever dispatches LESS).
# `proposed`/`blocked`/terminal statuses do not hold slots (a blocked task
# waits on the user, possibly for days — letting it pin a slot would jam the
# drain). Subset-of-enum pinned by test.
INFRA_DRAIN_OCCUPIED_STATUSES = frozenset(
    {
        "planning",
        "plan_pending",
        "approved",
        "running",
        "verifying",
        "interpreting",
        "reviewing",
        "followups_running",
    }
)
# A failed spawn is not retried for this long — the window ALWAYS binds (a
# fresh PM `updated_ts` resets only the attempt COUNT, never the window:
# research-pm.md 4b rewrites the file on EVERY STATUS pass, so a window
# bypass would void the tight-loop guard whenever the PM is active). At the
# 10-min cron, 1 h is ~6 ticks. env EPM_INFRA_DRAIN_BACKOFF_S.
INFRA_DRAIN_BACKOFF_S_DEFAULT = 3600.0
# Attempt budget per PM adjudication epoch (a newer `updated_ts` resets the
# count). Mirrors the orphan pass's bounded-respawn philosophy.
# env EPM_INFRA_DRAIN_MAX_ATTEMPTS.
INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT = 3
# Dead-at-boot grace: a registration for a still-`proposed` task older than
# this whose session is definitively NOT live is STALE (stops pinning a
# pending slot; the ID becomes re-dispatchable under the backoff/attempt
# budget). 30 min comfortably covers the spawn->boot->status-flip gap
# (normally <5 min) while bounding a dead-at-boot freeze to ~3 ticks instead
# of the 14-day registry backstop. env EPM_INFRA_DRAIN_STALE_REG_GRACE_S.
INFRA_DRAIN_STALE_REG_GRACE_S_DEFAULT = 1800.0
# A queue `updated_ts` further in the future than this is treated as None —
# a future epoch would make every tick a "fresh adjudication" and void the
# attempt budget (tz confusion / LLM timestamp bug; the file is LLM-authored).
INFRA_DRAIN_FUTURE_TS_TOLERANCE_S = 300.0


def _infra_drain_enabled() -> bool:
    """Kill switch: False when ``EPM_DISABLE_INFRA_DRAIN`` is set truthy
    ("1"/"true"/"yes", case-insensitive). Default enabled. The queue file's
    own ``cap: 0`` is the PM-side soft pause; this env var is the
    operator-side hard stop (body-named contract: EPM_DISABLE_INFRA_DRAIN=1)."""
    raw = os.environ.get("EPM_DISABLE_INFRA_DRAIN", "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _infra_drain_backoff_s() -> float:
    """Retry-backoff window in seconds (env ``EPM_INFRA_DRAIN_BACKOFF_S``;
    default :data:`INFRA_DRAIN_BACKOFF_S_DEFAULT`). A malformed env value
    falls back to the default — a typo must not distort the budget (mirrors
    :func:`_orphan_staleness_s`)."""
    raw = os.environ.get("EPM_INFRA_DRAIN_BACKOFF_S")
    if not raw:
        return INFRA_DRAIN_BACKOFF_S_DEFAULT
    try:
        return float(raw)
    except ValueError:
        return INFRA_DRAIN_BACKOFF_S_DEFAULT


def _infra_drain_max_attempts() -> int:
    """Attempt cap per PM adjudication epoch (env
    ``EPM_INFRA_DRAIN_MAX_ATTEMPTS``; default
    :data:`INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT`). Malformed env value falls back
    to the default."""
    raw = os.environ.get("EPM_INFRA_DRAIN_MAX_ATTEMPTS")
    if not raw:
        return INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT


def _infra_drain_stale_reg_grace_s() -> float:
    """Dead-at-boot registration grace in seconds (env
    ``EPM_INFRA_DRAIN_STALE_REG_GRACE_S``; default
    :data:`INFRA_DRAIN_STALE_REG_GRACE_S_DEFAULT`). Malformed env value falls
    back to the default."""
    raw = os.environ.get("EPM_INFRA_DRAIN_STALE_REG_GRACE_S")
    if not raw:
        return INFRA_DRAIN_STALE_REG_GRACE_S_DEFAULT
    try:
        return float(raw)
    except ValueError:
        return INFRA_DRAIN_STALE_REG_GRACE_S_DEFAULT


def parse_infra_drain_queue(raw: str) -> dict | None:
    """Parse + validate the PM queue file content. Returns the canonical dict
    ``{"ids": list[int], "cap": int, "holds": dict[int, str],
    "updated_ts": float | None}`` or ``None`` when the content is invalid
    (the pass then no-ops — fail toward NOT dispatching). ``holds`` preserves
    the PM's reason strings so skip lines can interpolate them.

    Validity rules:

    - top level must be a JSON object;
    - ``ripe_oldest_first`` must be a list of ints (missing -> invalid: the
      field is the file's entire point); bools rejected; order-preserving
      dedup (first occurrence wins);
    - ``cap`` missing -> :data:`INFRA_DRAIN_CAP_DEFAULT`; present but not an
      int >= 0 -> invalid (a garbled cap must not silently become 3);
    - ``holds`` missing -> empty; present but not a dict -> invalid; keys are
      int()-parsed (the live file uses string keys); an unparseable key ->
      invalid (a malformed hold must never be silently ignored — that would
      DISPATCH a held ID);
    - ``updated_ts`` parsed via :func:`_parse_event_ts` (ISO-8601 Z);
      unparseable -> ``None`` (only weakens attempt-reset, never dispatch
      eligibility). The FUTURE-ts clamp lives in :func:`decide_infra_drain`
      (pure, testable); the executor mirrors it for the log line.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    ids_raw = data.get("ripe_oldest_first")
    if not isinstance(ids_raw, list):
        return None
    ids: list[int] = []
    for x in ids_raw:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x not in ids:
            ids.append(x)
    cap = data.get("cap", INFRA_DRAIN_CAP_DEFAULT)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        return None
    holds_raw = data.get("holds", {})
    if not isinstance(holds_raw, dict):
        return None
    holds: dict[int, str] = {}
    for key, reason in holds_raw.items():
        try:
            holds[int(key)] = str(reason)
        except (TypeError, ValueError):
            return None
    return {
        "ids": ids,
        "cap": cap,
        "holds": holds,
        "updated_ts": _parse_event_ts(data.get("updated_ts")),
    }


def _cheap_skip_reason(
    i: int,
    holds: dict[int, str],
    registered: set[int],
    attempts: dict[int, dict],
    now: float,
    queue_updated_ts: float | None,
    *,
    backoff_s: float,
    max_attempts: int,
) -> str | None:
    """The per-ID guards computable WITHOUT ``task.py`` subprocesses. Returns
    the skip reason or ``None`` (eligible so far). Used VERBATIM by both
    :func:`decide_infra_drain` (guards 1-3) and the executor's cheap
    pre-filter — single source of truth, no predicate drift.

      1. ``i in holds``                         -> ``"held"``
      2. ``i in registered``                    -> ``"already-registered"``
         (``registered`` = NON-STALE registrations; the executor pre-filter
         passes the raw registration-ID set — a stale registration makes the
         ID MORE eligible, so pre-filtering on the raw set risks only a
         false skip toward the next tick)
      3. budget — the BACKOFF WINDOW ALWAYS BINDS (a fresh PM ``updated_ts``
         resets only the attempt COUNT, never the window):
           ``now - last_attempt_ts < backoff_s``  -> ``"backoff"`` (always)
           ``attempts >= max_attempts`` AND NOT
           ``queue_updated_ts > last_attempt_ts`` -> ``"attempts-exhausted"``
         Records with malformed ``last_attempt_ts`` were dropped at
         state-load time (defensive normalize, logged), so ``last`` here is
         numeric whenever present.
    """
    if i in holds:
        return "held"
    if i in registered:
        return "already-registered"
    rec = attempts.get(i) or {}
    last = rec.get("last_attempt_ts")
    if isinstance(last, int | float) and not isinstance(last, bool):
        if now - last < backoff_s:
            return "backoff"  # ALWAYS binds
        fresh = queue_updated_ts is not None and queue_updated_ts > last
        if not fresh and rec.get("attempts", 0) >= max_attempts:
            return "attempts-exhausted"  # fresh PM write resets the COUNT only
    return None


def decide_infra_drain(
    ids: list[int],
    cap: int,
    holds: dict[int, str],
    statuses: dict[int, str | None],
    kinds: dict[int, str | None],
    registered: set[int],
    occupied_active: int,
    pending: int,
    attempts: dict[int, dict],
    now: float,
    queue_updated_ts: float | None,
    *,
    backoff_s: float = INFRA_DRAIN_BACKOFF_S_DEFAULT,
    max_attempts: int = INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT,
) -> tuple[list[int], list[tuple[int, str]]]:
    """Pure decision: ``(dispatch_ids_in_order, skipped [(id, reason)])``.

    ``ids`` is the validated ``ripe_oldest_first`` list; ``registered`` the
    NON-STALE registrations among the queue IDs; ``occupied_active`` the
    count of kind-infra/batch tasks at :data:`INFRA_DRAIN_OCCUPIED_STATUSES`;
    ``pending`` the executor-precomputed count of ALL non-stale registrations
    (queue AND non-queue) whose task is still ``proposed`` with a drain kind,
    plus any whose status/kind is unreadable (conservative — see
    :func:`_infra_drain_pending`). ``free = max(0, cap - occupied_active -
    pending)``.

    FUTURE-TS CLAMP (first statement, pure + testable): a
    ``queue_updated_ts`` further than
    :data:`INFRA_DRAIN_FUTURE_TS_TOLERANCE_S` past ``now`` is treated as
    ``None`` — a future epoch would make every tick a "fresh adjudication"
    and void the attempt budget.

    Per-ID guard order (cheapest first; every skip carries a reason string):

      1-3. :func:`_cheap_skip_reason` (held / already-registered / backoff /
           attempts-exhausted — backoff always binds)
      4. status unreadable -> ``"status-unreadable"`` (fail toward not
         dispatching); status != ``proposed`` -> ``"status-<status>"``
      5. kind not in :data:`INFRA_DRAIN_KINDS` -> ``"kind-<kind|unreadable>"``
      6. no free slot -> ``"cap-full"``
      7. else dispatch; one attempt per ID per cycle
    """
    if queue_updated_ts is not None and queue_updated_ts > now + INFRA_DRAIN_FUTURE_TS_TOLERANCE_S:
        queue_updated_ts = None  # future-ts clamp; the executor logs the condition loudly
    free = max(0, cap - occupied_active - pending)
    dispatch: list[int] = []
    skipped: list[tuple[int, str]] = []
    for i in ids:
        reason = _cheap_skip_reason(
            i,
            holds,
            registered,
            attempts,
            now,
            queue_updated_ts,
            backoff_s=backoff_s,
            max_attempts=max_attempts,
        )
        if reason is not None:
            skipped.append((i, reason))
            continue
        status = statuses.get(i)
        if status is None:
            skipped.append((i, "status-unreadable"))
            continue
        if status != "proposed":
            skipped.append((i, f"status-{status}"))
            continue
        kind = kinds.get(i)
        if kind not in INFRA_DRAIN_KINDS:
            skipped.append((i, f"kind-{kind or 'unreadable'}"))
            continue
        if free <= 0:
            skipped.append((i, "cap-full"))
            continue
        dispatch.append(i)
        free -= 1
    return dispatch, skipped


def _infra_drain_queue_path() -> Path:
    """Path of the PM-authored queue file (resolved at call time so the test
    fixture's ``AUTONOMOUS_REGISTRY_DIR`` monkeypatch isolates it)."""
    return AUTONOMOUS_REGISTRY_DIR / INFRA_DRAIN_QUEUE_BASENAME


def _infra_drain_state_path() -> Path:
    """Path of the consolidated per-ID attempt/backoff state file."""
    return AUTONOMOUS_REGISTRY_DIR / INFRA_DRAIN_STATE_BASENAME


def _load_infra_drain_state() -> dict:
    """Read the attempt/backoff state (``{"attempts": {int: rec}}``; empty on
    absent/garbled, mirroring :func:`_load_orphan_state`). On-disk keys are
    strings (JSON); they are int-normalized here. DEFENSIVE NORMALIZE: drop
    (with a log line) any record whose ``last_attempt_ts`` is not numeric or
    whose key is not an int — a garbled record must not silently bypass the
    budget. A record with a MISSING ``attempts`` key or whose ``attempts``
    COUNT is garbled (non-int / bool / negative) keeps its valid
    ``last_attempt_ts`` (the backoff window still binds) but has the count
    normalized UP to the attempt cap, with a log line — count unknown means
    the budget may be spent, so fail toward NOT dispatching; a fresh PM
    ``updated_ts`` (newer than ``last_attempt_ts``) resets the count as
    usual, so recovery is automatic on the next PM adjudication. Dropping
    the record instead would RESET the budget — the fail-open direction
    (round-2 fix, Codex Major: ``int("bad") + 1`` / ``"bad" >= max_attempts``
    crashed the whole pass; round-4 fix for the MISSING-key sibling: bare
    ``rec.get("attempts", 0)`` returned 0 before the type check fired,
    silently granting a fresh budget on a half-written / hand-edited
    record)."""
    path = _infra_drain_state_path()
    if not path.is_file():
        return {"attempts": {}}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"attempts": {}}
    if not isinstance(data, dict) or not isinstance(data.get("attempts"), dict):
        return {"attempts": {}}
    attempts: dict[int, dict] = {}
    for key, rec in data["attempts"].items():
        try:
            issue = int(key)
        except (TypeError, ValueError):
            print(f"  infra-drain: dropping garbled state record (non-int key {key!r})")
            continue
        last = rec.get("last_attempt_ts") if isinstance(rec, dict) else None
        if isinstance(last, bool) or not isinstance(last, int | float):
            print(
                f"  infra-drain: dropping garbled state record for #{issue} "
                f"(non-numeric last_attempt_ts {last!r})"
            )
            continue
        if "attempts" not in rec:
            # Missing key — half-written or hand-edited record. Bare
            # ``rec.get("attempts", 0)`` would return 0 BEFORE the type
            # check fired below and silently grant a fresh budget; fail
            # to cap instead, the same fail direction the
            # garbled-count branch already uses (round-4 fix).
            cap = _infra_drain_max_attempts()
            print(
                f"  infra-drain: state record for #{issue} is missing its attempts "
                f"key; normalizing to the attempt cap ({cap}) — fail toward NOT "
                f"dispatching (a fresh PM updated_ts resets it)"
            )
            rec = {**rec, "attempts": cap}
        else:
            count = rec["attempts"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                cap = _infra_drain_max_attempts()
                print(
                    f"  infra-drain: state record for #{issue} has a garbled attempts "
                    f"count ({count!r}); normalizing to the attempt cap ({cap}) — fail "
                    f"toward NOT dispatching (a fresh PM updated_ts resets it)"
                )
                rec = {**rec, "attempts": cap}
        attempts[issue] = rec
    return {"attempts": attempts}


def _save_infra_drain_state(state: dict) -> None:
    """Persist the attempt/backoff state atomically (temp + rename, mirroring
    :func:`_save_orphan_state`). Int keys serialize as strings; the loader
    normalizes them back."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _infra_drain_state_path()
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(dest)


def _task_status_kind(issue: int) -> tuple[str | None, str | None]:
    """``(status, kind)`` from ONE ``task.py view <N> --json`` subprocess —
    the same payload :func:`_task_status` parses carries ``frontmatter.kind``,
    so the kind guard costs zero extra subprocesses. Fail-soft
    ``(None, None)``."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "view", str(issue), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return (None, None)
    if out.returncode != 0:
        return (None, None)
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return (None, None)
    fm = data.get("frontmatter") or {}
    status = data.get("status") or fm.get("status")
    kind = fm.get("kind")
    return (
        status if isinstance(status, str) else None,
        kind if isinstance(kind, str) else None,
    )


def _infra_drain_reg_snapshot() -> dict[str, bytes]:
    """Raw bytes of every ``issue-*.json`` + ``manual-issue-*.json``
    registration file, keyed by basename — the SINGLE decision-time read both
    the staleness/pending decision (:func:`_infra_drain_registrations` parses
    from these bytes) and :func:`_dispatch_infra_drain`'s pre-spawn lost-race
    re-check are derived from. One read means the re-check can distinguish "a
    registration APPEARED/CHANGED since the decision" (genuine lost race ->
    abort) from "this is the registration the pass already classified STALE"
    (byte-identical -> safe to spawn over; ``spawn_session.py`` overwrites it
    unconditionally on success). Round-2 fix for the round-1 Critical: the
    bare-existence re-check aborted every stale-registration re-dispatch
    forever. A file unreadable at snapshot time is omitted (it then reads as
    "appeared" at dispatch time -> abort, the safe direction)."""
    snap: dict[str, bytes] = {}
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return snap
    for prefix in ("issue-", "manual-issue-"):
        for path in sorted(AUTONOMOUS_REGISTRY_DIR.glob(f"{prefix}*.json")):
            try:
                snap[path.name] = path.read_bytes()
            except OSError:
                continue
    return snap


def _infra_drain_registrations(snapshot: dict[str, bytes]) -> dict[int, list[dict]]:
    """ALL ``issue-<N>.json`` + ``manual-issue-<N>.json`` registration entries
    per issue (not just queue IDs — the widened pending count needs non-queue
    ones too), parsed FROM the decision-time ``snapshot``
    (:func:`_infra_drain_reg_snapshot`) so the staleness decision and the
    pre-spawn re-check can never see torn views of the registry. List-valued
    because an issue can carry BOTH an autonomous and a manual registration;
    staleness then requires ALL of them stale (fail toward keep-blocking). A
    garbled entry parses to ``{}`` (which :func:`_infra_drain_stale`
    classifies NOT stale — conservative)."""
    out: dict[int, list[dict]] = {}
    for prefix in ("issue-", "manual-issue-"):
        for basename in sorted(snapshot):
            if not basename.startswith(prefix):
                continue
            issue = _gc_parse_issue_from_path(AUTONOMOUS_REGISTRY_DIR / basename, prefix, "")
            if issue is None:
                continue
            try:
                entry = json.loads(snapshot[basename])
            except ValueError:  # JSONDecodeError + UnicodeDecodeError on raw bytes
                entry = {}
            if not isinstance(entry, dict):
                entry = {}
            out.setdefault(issue, []).append(entry)
    return out


def _infra_drain_stale(
    reg: dict,
    live_session_ids: set[str] | None,
    status: str | None,
    now: float,
    grace_s: float,
) -> bool:
    """STALE (dead-at-boot) iff ALL of: task status == ``proposed``; the
    registration's ``spawned_at`` parses AND ``now - spawned_at > grace_s``;
    the recorded session id is present AND ``live_session_ids`` is available
    AND the id is NOT in it. ANY missing/unparseable signal -> NOT stale
    (fail toward keep-blocking — a false-stale could double-spawn; a
    false-live only delays recovery to the 14-day registry backstop). A
    STALE registration stops counting toward pending and stops blocking
    re-dispatch; the backoff/attempt budget bounds the re-dispatch rate."""
    if status != "proposed":
        return False
    spawned_at = reg.get("spawned_at")
    if isinstance(spawned_at, bool) or not isinstance(spawned_at, int | float):
        return False
    if now - spawned_at <= grace_s:
        return False
    sid = reg.get("happy_session_id")
    if not isinstance(sid, str) or not sid:
        return False
    if live_session_ids is None:
        return False
    return sid not in live_session_ids


def _infra_drain_occupancy() -> list[int] | None:
    """IDs of tasks with kind in :data:`INFRA_DRAIN_KINDS` at
    :data:`INFRA_DRAIN_OCCUPIED_STATUSES`, via one ``task.py list-by-status
    --json`` subprocess per status (mirrors :func:`_active_status_tasks` but
    keeps the ``kind`` field; the IDs feed the cap-full ``occupying=[...]``
    log line). Returns ``None`` when ANY status read fails — a partial count
    would UNDER-count and over-dispatch, so the executor skips dispatching
    this tick on ``None`` (deliberately STRICTER than
    :func:`_active_status_tasks`' per-status fail-soft, which is safe there
    because that pass only recovers, never spawns new work). ``kind:
    campaign`` rows can't match (not in :data:`INFRA_DRAIN_KINDS`)."""
    occupying: list[int] = []
    for status in sorted(INFRA_DRAIN_OCCUPIED_STATUSES):
        try:
            res = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/task.py",
                    "list-by-status",
                    "--status",
                    status,
                    "--json",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        if res.returncode != 0:
            return None
        try:
            rows = json.loads(res.stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, dict) or row.get("kind") not in INFRA_DRAIN_KINDS:
                continue
            tid = row.get("id")
            if isinstance(tid, int):
                occupying.append(tid)
    return occupying


def _infra_drain_pending(
    registrations: dict[int, list[dict]],
    stale: set[int],
    status_kind: dict[int, tuple[str | None, str | None]],
) -> int:
    """Widened pending count: over ALL non-stale registrations (queue and
    non-queue), count those whose task is still ``proposed`` with a kind in
    :data:`INFRA_DRAIN_KINDS` (spawned but ``/issue`` hasn't flipped status
    yet) PLUS those whose status or kind is UNREADABLE (conservative — an
    unknown registered task might be a just-spawned infra task; fail toward
    occupying a slot). Tasks at occupied statuses are already counted in
    ``occupied_active``; terminal/blocked registrations count zero. Closes
    both the PM-prunes-a-dispatched-ID overshoot and the status-read-failure
    undercount."""
    pending = 0
    for issue in registrations:
        if issue in stale:
            continue
        status, kind = status_kind.get(issue, (None, None))
        if status is None or kind is None or (status == "proposed" and kind in INFRA_DRAIN_KINDS):
            pending += 1
    return pending


def _dispatch_infra_drain(
    issue: int,
    slot_desc: str,
    dry_run: bool,
    *,
    reg_snapshot: dict[str, bytes] | None = None,
) -> bool:
    """``spawn_session.py spawn-issue --issue <N> --auto`` (the plain
    command, exactly the PM standing-rule item-3 mechanism; no
    ``--auto-approve-gpu-hours`` override — spawn_session's default applies,
    and infra tasks need ~0 GPU). Immediately BEFORE the subprocess,
    re-checks the registration files against the DECISION-TIME snapshot
    (:func:`_infra_drain_reg_snapshot`) and aborts ("lost race to concurrent
    dispatcher") only when a registration APPEARED or CHANGED since the
    decision — a registration byte-identical to the snapshot is the
    known-stale entry this pass already classified (round-2 fix: the round-1
    bare-existence check aborted every stale-registration re-dispatch;
    ``spawn_session.py`` overwrites the file unconditionally on success, so
    spawning over the stale entry is safe). ``reg_snapshot=None`` (direct
    callers without a decision context) degrades to the conservative
    abort-on-any-existing-file behavior. Shrinks the PM-vs-watcher
    double-spawn window from one-full-pass to ~the spawn subprocess itself.
    Returns success bool; honours dry_run (logs, never spawns)."""
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto",
    ]  # fmt: skip
    if dry_run:
        print(f"  [dry-run] would dispatch infra-drain: {' '.join(cmd)}")
        return False
    snapshot = reg_snapshot or {}
    for basename in (f"issue-{issue}.json", f"manual-issue-{issue}.json"):
        path = AUTONOMOUS_REGISTRY_DIR / basename
        known = snapshot.get(basename)
        try:
            current = path.read_bytes()
        except FileNotFoundError:
            # No registration now (never existed, or GC/PM removed it since
            # the decision) — nothing to lose a race to.
            continue
        except OSError:
            print(
                f"  INFRA-DRAIN ABORT issue #{issue}: registration {basename} "
                f"unreadable at the pre-spawn re-check; cannot verify the "
                f"decision still holds (fail toward not dispatching)"
            )
            return False
        if known is None or current != known:
            print(
                f"  INFRA-DRAIN ABORT issue #{issue}: lost race to concurrent "
                f"dispatcher ({basename} "
                f"{'appeared' if known is None else 'changed'} since the decision)"
            )
            return False
        # else: byte-identical to the decision-time snapshot — the known
        # (stale) registration the decision already accounted for; proceed.
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(
            f"  INFRA-DRAIN DISPATCH FAILED issue #{issue}: {res.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  INFRA-DRAIN DISPATCHED issue #{issue} ({slot_desc}): {first_line}")
    return True


def _infra_drain_read_queue() -> dict | None:
    """Read + parse the queue file; ``None`` (with exactly one header line)
    when the pass should no-op: file missing, unreadable, or invalid."""
    path = _infra_drain_queue_path()
    try:
        raw = path.read_text()
    except FileNotFoundError:
        print(f"infra-drain: no queue file at {path}; skipping")
        return None
    except OSError as e:
        print(f"infra-drain: queue file at {path} unreadable ({e}); skipping")
        return None
    queue = parse_infra_drain_queue(raw)
    if queue is None:
        print(
            f"infra-drain: INVALID queue file at {path} (schema violation or "
            f"torn write); skipping (fail-safe: not dispatching)"
        )
    return queue


def _infra_drain_record_attempt(
    attempts: dict[int, dict],
    issue: int,
    now: float,
    queue_updated_ts: float | None,
    ok: bool,
) -> None:
    """Record one dispatch ATTEMPT (success or failure both count, so a
    failing spawn can't tight-loop). A fresh PM adjudication (``updated_ts``
    newer than the last attempt) resets the count to 1 first."""
    rec = attempts.get(issue) or {}
    last = rec.get("last_attempt_ts")
    fresh = (
        queue_updated_ts is not None
        and isinstance(last, int | float)
        and not isinstance(last, bool)
        and queue_updated_ts > last
    )
    count = 1 if fresh else int(rec.get("attempts", 0)) + 1
    attempts[issue] = {
        "attempts": count,
        "last_attempt_ts": now,
        "last_result": "dispatched" if ok else "spawn-failed",
        "exhausted_logged": False,
    }


def _infra_drain_log_skips(
    skipped: list[tuple[int, str]],
    holds: dict[int, str],
    attempts: dict[int, dict],
) -> None:
    """One ``INFRA-DRAIN SKIP`` line per skipped ID. ``held`` interpolates
    the PM's hold reason; ``attempts-exhausted`` is loud ONCE per epoch
    (``exhausted_logged`` dedup flag, mirroring the orphan pass's
    ``alerted``); ``kind-*`` is loud EVERY tick (a mis-kinded queue entry is
    a PM-side bug needing eyes)."""
    for issue, reason in skipped:
        if reason == "held":
            print(f"  INFRA-DRAIN SKIP issue #{issue} (held: {holds.get(issue, '?')})")
        elif reason == "attempts-exhausted":
            rec = attempts.get(issue) or {}
            if not rec.get("exhausted_logged"):
                print(
                    f"  INFRA-DRAIN SKIP issue #{issue} (attempts-exhausted: "
                    f"{rec.get('attempts', '?')} attempts this PM epoch; parked "
                    f"until the PM rewrites the queue with a newer updated_ts)",
                    file=sys.stderr,
                )
                rec["exhausted_logged"] = True
                attempts[issue] = rec
            else:
                print(f"  INFRA-DRAIN SKIP issue #{issue} (attempts-exhausted)")
        elif reason.startswith("kind-"):
            print(
                f"  INFRA-DRAIN SKIP issue #{issue} ({reason}): mis-kinded queue "
                f"entry — only kind infra/batch is drain-dispatchable; fix the "
                f"PM queue file (research-pm.md standing-rule item 4b)",
                file=sys.stderr,
            )
        else:
            print(f"  INFRA-DRAIN SKIP issue #{issue} ({reason})")


def _infra_drain_prune_save(state: dict, ids: list[int], dry_run: bool) -> None:
    """Prune state entries whose ID left ``ripe_oldest_first``, then persist
    atomically. No-op under dry-run (mirror the orphan pass's dry-run
    discipline: no state write)."""
    if dry_run:
        return
    keep = set(ids)
    state["attempts"] = {i: rec for i, rec in state["attempts"].items() if i in keep}
    _save_infra_drain_state(state)


def _infra_drain_clamp_future_ts(queue_updated_ts: float | None, now: float) -> float | None:
    """Executor-side mirror of the decide-side future-ts clamp, so the
    condition is logged loudly exactly once per tick."""
    if queue_updated_ts is not None and queue_updated_ts > now + INFRA_DRAIN_FUTURE_TS_TOLERANCE_S:
        print(
            f"infra-drain: queue updated_ts is {queue_updated_ts - now:.0f}s in the "
            f"FUTURE (tz confusion / LLM timestamp bug?); treating it as absent so "
            f"it cannot void the attempt budget"
        )
        return None
    return queue_updated_ts


def _infra_drain_possibly_stale_ids(
    candidate_ids: set[int],
    regs: dict[int, list[dict]],
    now: float,
) -> set[int]:
    """Subset of ``candidate_ids`` whose EVERY registration entry is
    suspicious enough — older than the grace window, recorded session id
    definitively not live — that the full path must confirm staleness with a
    status read. Reuses :func:`_infra_drain_stale` with the status check
    pre-satisfied (``"proposed"``), so the two predicates can never drift.

    Without this probe the pre-filter's all-skipped early exit would park a
    dead-at-boot registered ID FOREVER (staleness is only computed on the
    full path, and the full path never runs when the stale ID is the only
    non-skipped queue ID) — exactly the drain-freeze the stale-registration
    handling exists to prevent. Costs one daemon RPC, zero ``task.py``
    subprocesses; only invoked when some queue ID pre-filtered as
    ``already-registered``."""
    if not candidate_ids:
        return set()
    # _live_session_ids_or_none, NOT spawn_session._live_session_ids: the
    # caller's daemon gate is main()'s single probe, and a flap since then
    # must read as UNAVAILABLE (None -> nothing stale -> keep blocking),
    # never as "zero live sessions" (-> everything stale -> double-spawn).
    live_ids = _live_session_ids_or_none()
    grace_s = _infra_drain_stale_reg_grace_s()
    return {
        i
        for i in candidate_ids
        if regs.get(i)
        and all(_infra_drain_stale(r, live_ids, "proposed", now, grace_s) for r in regs[i])
    }


def _infra_drain_signals(
    ids: list[int],
    holds: dict[int, str],
    regs: dict[int, list[dict]],
    now: float,
) -> tuple[dict[int, tuple[str | None, str | None]], set[int], int]:
    """Fetch the ``task.py``-backed signals for the full dispatch path:
    per-ID ``(status, kind)`` for every non-held queue ID and every
    registration ID, the stale-registration set, and the widened pending
    count. One ``view`` subprocess per ID; the caller's cheap pre-filter
    bounds the common nothing-dispatchable tick at zero of these."""
    fetch_ids = {i for i in ids if i not in holds} | set(regs)
    status_kind = {i: _task_status_kind(i) for i in sorted(fetch_ids)}
    # Liveness only matters for staleness over registrations; skip the daemon
    # RPC when there are none. _live_session_ids_or_none (NOT spawn_session's
    # empty-set-on-failure helper): a daemon flap since main()'s probe must
    # read UNAVAILABLE (None -> nothing stale -> keep blocking), never "zero
    # live sessions" (-> false-stale -> double-spawn).
    live_ids = _live_session_ids_or_none() if regs else None
    grace_s = _infra_drain_stale_reg_grace_s()
    stale: set[int] = set()
    for issue, recs in regs.items():
        status = status_kind.get(issue, (None, None))[0]
        if recs and all(_infra_drain_stale(r, live_ids, status, now, grace_s) for r in recs):
            stale.add(issue)
            print(
                f"  infra-drain: STALE registration for issue #{issue} (task still "
                f"proposed, registration older than grace, session not live) — no "
                f"longer pins a pending slot; ID re-dispatchable under the budget"
            )
    pending = _infra_drain_pending(regs, stale, status_kind)
    return status_kind, stale, pending


def infra_drain_pass(
    dry_run: bool,
    now: float | None = None,
    *,
    daemon_reachable: bool | None = None,
) -> None:
    """Execute the PM-adjudicated infra dispatch queue (task #633). Pure
    executor — every judgment was the PM session's; every guard here is
    mechanical. Missing/invalid queue file = logged no-op; daemon-gated like
    every other spawning pass (spawn POSTs to the Happy daemon RPC)."""
    if not _infra_drain_enabled():
        print("infra-drain: disabled via EPM_DISABLE_INFRA_DRAIN; skipping")
        return
    queue = _infra_drain_read_queue()
    if queue is None:
        return
    if daemon_reachable is None:
        daemon_reachable = _daemon_reachable()
    if not daemon_reachable:
        print("infra-drain: Happy daemon unreachable; skipping (spawn needs the daemon RPC)")
        return
    now = now if now is not None else time.time()
    ids: list[int] = queue["ids"]
    cap: int = queue["cap"]
    holds: dict[int, str] = queue["holds"]
    queue_updated_ts = _infra_drain_clamp_future_ts(queue["updated_ts"], now)
    state = _load_infra_drain_state()
    attempts: dict[int, dict] = state["attempts"]
    backoff_s = _infra_drain_backoff_s()
    max_attempts = _infra_drain_max_attempts()
    # ONE registry read: the staleness/pending decision parses from this
    # snapshot, and the pre-spawn re-check compares against the same bytes —
    # so "appeared/changed since the decision" is exact (round-2 fix #1).
    reg_snapshot = _infra_drain_reg_snapshot()
    regs = _infra_drain_registrations(reg_snapshot)

    # Cheap pre-filter (single source of truth: the SAME _cheap_skip_reason
    # decide_infra_drain uses, with `registered` = the raw registration-file
    # ID set — staleness not yet computed; a stale registration makes the ID
    # MORE eligible, so pre-filtering on the raw set risks only a false skip
    # toward the next tick). When every ID pre-filters to a skip, the tick
    # costs ZERO task.py subprocesses.
    raw_registered = set(regs) & set(ids)
    prefilter = {
        i: _cheap_skip_reason(
            i,
            holds,
            raw_registered,
            attempts,
            now,
            queue_updated_ts,
            backoff_s=backoff_s,
            max_attempts=max_attempts,
        )
        for i in ids
    }
    if all(reason is not None for reason in prefilter.values()):
        # An "already-registered" pre-filter skip may be rescued by the
        # stale-registration handling (dead-at-boot session) — that needs a
        # status read, so such IDs defer the early exit to the full path
        # when their registration looks suspicious (older than grace +
        # session not live). Everything else (held / backoff / exhausted /
        # live-or-fresh registration) early-exits with zero task.py reads.
        registered_skips = {i for i in ids if prefilter[i] == "already-registered"}
        if not _infra_drain_possibly_stale_ids(registered_skips, regs, now):
            print(
                f"infra-drain: queue={len(ids)} dispatched=0 skipped={len(ids)} "
                f"(pre-filtered; zero task.py reads this tick)"
            )
            _infra_drain_log_skips([(i, prefilter[i]) for i in ids], holds, attempts)
            _infra_drain_prune_save(state, ids, dry_run)
            return

    status_kind, stale, pending = _infra_drain_signals(ids, holds, regs, now)
    registered_nonstale = raw_registered - stale
    occupying = _infra_drain_occupancy()
    if occupying is None:
        print(
            "infra-drain: occupancy read FAILED for at least one status; "
            "skipping dispatch this tick (fail-closed: a partial count would "
            "under-count and over-dispatch past the cap)"
        )
        _infra_drain_prune_save(state, ids, dry_run)
        return
    occupied_active = len(occupying)
    statuses = {i: status_kind.get(i, (None, None))[0] for i in ids}
    kinds = {i: status_kind.get(i, (None, None))[1] for i in ids}
    dispatch, skipped = decide_infra_drain(
        ids,
        cap,
        holds,
        statuses,
        kinds,
        registered_nonstale,
        occupied_active,
        pending,
        attempts,
        now,
        queue_updated_ts,
        backoff_s=backoff_s,
        max_attempts=max_attempts,
    )
    dispatched = 0
    for issue in dispatch:
        slot_desc = f"slot {min(occupied_active + pending + dispatched + 1, cap)}/{cap}"
        ok = _dispatch_infra_drain(issue, slot_desc, dry_run, reg_snapshot=reg_snapshot)
        if not dry_run:
            # Count the ATTEMPT whether or not the spawn succeeded, so a
            # failing spawn can't tight-loop (the backoff window binds next
            # tick either way).
            _infra_drain_record_attempt(attempts, issue, now, queue_updated_ts, ok)
        if ok:
            dispatched += 1
            _post_progress_marker(
                issue,
                f"{_INFRA_DRAIN_NOTE_SENTINEL} watcher dispatched autonomous "
                f"session from the PM infra-drain queue (occupied "
                f"{occupied_active}+{pending} pending of cap {cap})",
                dry_run,
                label="infra-drain",
            )
    _infra_drain_log_skips(skipped, holds, attempts)
    summary = (
        f"infra-drain: queue={len(ids)} occupied={occupied_active}(+{pending} pending) "
        f"cap={cap} dispatched={dispatched} skipped={len(skipped)}"
    )
    if any(reason == "cap-full" for _i, reason in skipped):
        summary += f" occupying={sorted(occupying)}"
    print(summary)
    _infra_drain_prune_save(state, ids, dry_run)


# ─── generalized GC of stale ~/.eps-autonomous/ per-issue files ──────────────

# Task statuses for which per-issue registry / progress / stalled-state files
# can be safely reaped: the autonomous run is definitively over. Conservative
# by design — `awaiting_promotion` is EXCLUDED (the user could still be poking
# at the row) and `blocked` is EXCLUDED (the user is investigating). Re-using
# the existing TERMINAL set would NOT be conservative: `awaiting_promotion` is
# terminal for the autonomous-driver loop but not for the user's interaction.
TERMINAL_FOR_GC = {"completed", "archived"}

# (prefix, subdir) pairs the GC pass sweeps. ``""`` subdir means
# ``AUTONOMOUS_REGISTRY_DIR`` itself; a non-empty subdir is a child folder
# (``issue-progress/`` and ``issue-tick-last-status/`` keep their per-issue
# files in nested dirs). The pod-safety state files are reaped by their own
# RUNNING-set-aware GC (:func:`_gc_orphan_pod_safety_state`) and are NOT
# included here; likewise the session-reconcile state files
# (:func:`_gc_orphan_session_reconcile_state` — terminal-status reaping here
# would reset that pass's miss counter every tick).
_GC_TARGETS: tuple[tuple[str, str], ...] = (
    ("manual-issue-", ""),
    (STALLED_STATE_PREFIX, ""),
    (ORPHAN_STATE_PREFIX, ""),
    # Campaign watchdog state (== CAMPAIGN_WATCH_STATE_PREFIX, defined in the
    # campaign-pass section below; literal here because module-level tuples
    # evaluate top-to-bottom). Primary reaping is the campaign pass itself at
    # CAMPAIGN_TERMINAL; this is the deleted-task / completed-archived backstop.
    ("campaign-watch-", ""),
    # Gate-push transition state (== GATE_NOTIFY_STATE_PREFIX, defined in the
    # gate-push section below; literal here for the same top-to-bottom reason
    # as campaign-watch-). The companion tick-runaway-<N>.flag files are NOT
    # json and self-clean inside _process_runaway_flag instead.
    ("gate-notify-", ""),
    ("", "issue-progress"),
    ("", "issue-tick-last-status"),
)


def _gc_target_paths(prefix: str, subdir: str) -> tuple[Path, ...]:
    """Resolve the (prefix, subdir) tuple to a list of candidate paths.

    For ``subdir == ""``, sweeps top-level files matching ``<prefix>*.json``.
    For a nested subdir, sweeps top-level files in that subdir matching the
    plain ``<N>.json`` shape (no prefix — that's the ``issue-progress`` +
    ``issue-tick-last-status`` convention)."""
    base = AUTONOMOUS_REGISTRY_DIR if not subdir else (AUTONOMOUS_REGISTRY_DIR / subdir)
    if not base.is_dir():
        return ()
    pattern = f"{prefix}*.json" if not subdir else "*.json"
    return tuple(sorted(base.glob(pattern)))


def _gc_parse_issue_from_path(path: Path, prefix: str, subdir: str) -> int | None:
    """Extract the integer issue number from ``path``. Returns ``None`` if
    the stem doesn't carry a valid integer after the prefix (the caller logs
    + leaves the file — a hand-debug artifact is none of the GC's business)."""
    stem = path.stem
    if not subdir:
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix) :]
        elif prefix:
            return None
    # Else: nested subdir, files are named ``<N>.json`` already.
    try:
        return int(stem)
    except ValueError:
        return None


def _gc_orphaned_eps_autonomous_files(now: float, dry_run: bool) -> dict[str, int]:
    """Reap per-issue state files for tasks in :data:`TERMINAL_FOR_GC` (or
    whose age exceeds :data:`MAX_ENTRY_AGE_S` and whose status cannot be
    resolved, as a backstop).

    Conservative: ``awaiting_promotion`` / ``blocked`` / any park status are
    NEVER reaped — the user may still be interacting with the task. Garbled
    filenames (non-int stem) are left in place. Returns a per-prefix count
    dict (``{"manual-issue-": 3, "stalled-": 1, ...}``) for logging.

    Does NOT touch:

    - ``issue-<N>.json`` (autonomous registry) — those are handled by the
      respawn pass's per-entry status check + the existing
      :data:`MAX_ENTRY_AGE_S` backstop, both of which already drop a
      terminal-status entry. A second reaper here would race that path.
    - ``pod-safety-<N>.json`` — owned by :func:`_gc_orphan_pod_safety_state`
      which keys on the live RUNNING set, a different (complementary)
      question than task terminal status.
    - ``session-reconcile-<N>.json`` — owned by
      :func:`_gc_orphan_session_reconcile_state` which keys on the live
      mapped-session set. MUST stay out of this sweep: those files track
      episodes whose task is BY DEFINITION terminal, so reaping them here
      would reset the miss counter every tick and the session-reconcile
      threshold could never be reached.
    - ``session_progress.json`` / ``watch.lock`` (project-singletons, not
      per-issue).
    - ``vm-disk.json`` / ``vm-disk-events.jsonl`` (project-singletons for the
      VM disk-headroom pass — :func:`vm_disk_pass` owns the state file's
      lifecycle via its episode-recovery clear).
    """
    counts: dict[str, int] = {}
    for prefix, subdir in _GC_TARGETS:
        cleared = 0
        for path in _gc_target_paths(prefix, subdir):
            issue = _gc_parse_issue_from_path(path, prefix, subdir)
            if issue is None:
                continue
            status = _task_status(issue)
            if status in TERMINAL_FOR_GC:
                reason = f"task status={status}"
            elif status is None:
                # Status unresolvable. Apply the age backstop so a deleted /
                # archived-elsewhere task's state file can't linger forever.
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = now
                age = now - mtime
                if age <= MAX_ENTRY_AGE_S:
                    continue
                reason = f"task unresolvable + age={age / 3600:.1f}h"
            else:
                # Live PARK / ACTIVE / awaiting_promotion / blocked: keep.
                continue
            print(f"  gc: drop {path.relative_to(AUTONOMOUS_REGISTRY_DIR)} ({reason})")
            if not dry_run:
                path.unlink(missing_ok=True)
            cleared += 1
        if cleared:
            key = prefix if prefix else (subdir or "")
            counts[key] = counts.get(key, 0) + cleared
    return counts


def gc_pass(dry_run: bool, now: float | None = None) -> None:
    """Top-level wrapper around :func:`_gc_orphaned_eps_autonomous_files` for
    consistency with the other ``*_pass`` entry points + the ``--gc-only``
    debug flag."""
    now = now if now is not None else time.time()
    counts = _gc_orphaned_eps_autonomous_files(now, dry_run)
    if not counts:
        print("gc: no stale per-issue state files to reap")
        return
    summary = ", ".join(f"{k or 'nested'}={v}" for k, v in sorted(counts.items()))
    print(f"gc: cleared {summary}")


# ─── session-reconcile pass (sessions-vs-status; 2026-06-10 disk incident) ───
#
# Mirror of the pod-safety auto-stop arm for Happy SESSIONS. The respawn pass
# DELETES the registry entry when a task reaches a TERMINAL status (see
# :func:`decide`) but never stops the live session, and unregistered zombie
# generations (a newer spawn overwrote the per-issue registration file) are
# invisible to every registry-driven pass — so a per-issue session that
# outlives its task's completion persists indefinitely. In the 2026-06-10
# disk-full incident 15+ such sessions (some weeks old) sat alive in the
# worktrees of completed/archived tasks, pinning 10-15G worktrees each against
# the stale-worktree sweep and holding open deleted-file handles (~37G of
# phantom disk usage); 17 had to be stopped by hand before the worktree audit
# could see their worktrees as unpinned.
#
# Conservative posture, mirroring how the pod pass and the stalled-detector
# were introduced (auto-stop became the DEFAULT on 2026-06-10 — see
# :func:`_session_reconcile_autostop_enabled` — after a manual sweep of 14
# sessions validated the exact predicate below):
#
#   * acts ONLY on tasks in :data:`SESSION_RECONCILE_DONE`
#     (awaiting_promotion / completed / archived — the pod-safety auto-stop
#     set; ``followups_running`` and ``blocked`` are excluded because the
#     session may be legitimately live there);
#   * requires > :func:`_session_idle_s` (default 2h) of inactivity on EVERY
#     available activity signal (newest non-watcher marker of ANY kind + the
#     per-issue self-report file);
#   * the same >=2-consecutive-checks miss guard as the pod pass;
#   * honours the ``keep-running`` tag, the inferred inline-follow-up
#     predicate (:func:`_task_session_followup_active`, wider signal/
#     transition sets than the pod pass's), and a no-RUNNING-pod check;
#   * ``EPM_SESSION_RECONCILE_AUTOSTOP=0`` falls back to the original
#     ALERT-ONLY posture (loud log + one-time marker, no stop);
#   * a daemon ACK is never trusted as a kill: ACKed stops are recorded in
#     the state file (``stopped_at``) and verified actually-gone on the
#     next tick; a survivor gets ONE stop retry, then a loud one-time
#     marker — the episode state is never cleared on an unverified stop
#     (:func:`_check_stop_verification`);
#   * NEVER touches a session with no issue mapping (the PM session, chat
#     sessions) — those are skipped at the mapping step and cannot reach the
#     decision function.

# Parked/terminal statuses whose live sessions the pass reconciles. Shares
# the pod-safety auto-stop set (NOT the GC's narrower terminal set):
# `awaiting_promotion` was added 2026-06-10 on the user request "Can we stop
# the happy sessions once they reach awaiting promotion?" — the promotion
# park is a human gate with no session-side work left, and idle sessions
# there accumulated to 73 registered / ~35-40GB RSS. `followups_running`
# is deliberately NOT here: that status means a same-issue follow-up round
# is executing and the session is its driver. `blocked` is NOT here either
# (under investigation; the user may be live-parked in the session).
SESSION_RECONCILE_DONE = AUTO_STOP_DONE

# Default inactivity grace window before a parked/terminal task's live
# session counts as idle. 2h (validated by the 2026-06-10 manual sweep of
# 14 sessions: a 2h any-marker grace protected #504/#538/#540, which had
# minutes-old progress markers despite parked statuses) — overridable via
# EPM_SESSION_RECONCILE_IDLE_S (seconds, see _session_idle_s).
SESSION_IDLE_S = 2 * 3600


def _session_idle_s() -> float:
    """Idle grace window in seconds: ``EPM_SESSION_RECONCILE_IDLE_S`` when set
    to a positive number, else :data:`SESSION_IDLE_S` (2h). A garbled /
    non-positive value falls back to the default rather than crashing the
    watcher pass."""
    raw = os.environ.get("EPM_SESSION_RECONCILE_IDLE_S", "")
    try:
        val = float(raw)
    except ValueError:
        return SESSION_IDLE_S
    return val if val > 0 else SESSION_IDLE_S


# Marker kinds that signal a follow-up may be in flight on a parked/terminal
# task. Broader than the pod-safety pass's bare `epm:run-launched`
# (:data:`_RUN_LAUNCHED_KIND`): `epm:followup-scope` lands when a follow-up
# is REQUESTED (before any session picks it up — the window where stopping
# the session would orphan the request), and `epm:free-analysis-followup-run`
# marks the inline zero-GPU auto-run. Any of these NEWER than the latest
# done-transition marker means the session may be (or be about to become)
# the follow-up's driver.
_SESSION_FOLLOWUP_SIGNAL_KINDS = frozenset(
    {
        "epm:run-launched",
        "epm:followup-scope",
        "epm:free-analysis-followup-run",
    }
)

# Marker kinds that record the task settling into its parked/terminal state.
# Broader than the pod-safety pass's set: `epm:pod-terminated` and
# `epm:step-completed` also mark a round wrapping up, so a follow-up signal
# OLDER than any of these is provably finished business, not in-flight work.
_SESSION_DONE_TRANSITION_KINDS = frozenset(
    {
        "epm:promoted",
        "epm:status-changed",
        "epm:pod-terminated",
        "epm:step-completed",
    }
)


def _task_session_followup_active(issue: int, events: list[dict] | None = None) -> bool:
    """True iff task ``issue`` has a follow-up signal marker
    (:data:`_SESSION_FOLLOWUP_SIGNAL_KINDS`) NEWER than its latest
    done-transition marker (:data:`_SESSION_DONE_TRANSITION_KINDS`).

    The session-reconcile twin of :func:`_task_followup_active`. The two
    predicates now share the same follow-up signal set
    (:data:`_POD_FOLLOWUP_SIGNAL_KINDS` == the follow-up side of this set,
    widened 2026-06-10, refs #573) but stay decoupled symbols because their
    DONE-TRANSITION sets differ: the session twin also counts
    ``epm:pod-terminated`` / ``epm:step-completed`` as settling markers,
    which would re-arm a pod stop too eagerly. Same defensive posture:
    no follow-up signal -> False; no done-transition despite a DONE status
    (shouldn't happen — at least one ``epm:status-changed`` put it there)
    -> False, leaving the idle grace + 2-miss guard as the safety margin.
    """
    if events is None:
        events = _task_events(issue)
    followup = _latest_event_ts(events, _SESSION_FOLLOWUP_SIGNAL_KINDS)
    if followup is None:
        return False
    done_transition = _latest_event_ts(events, _SESSION_DONE_TRANSITION_KINDS)
    if done_transition is None:
        return False
    return followup > done_transition


def _latest_nonwatcher_event_ts(events: list[dict]) -> float | None:
    """Newest epoch ts among ALL events whose note does NOT carry a watcher
    sentinel (:data:`_WATCHER_NOTE_SENTINELS`), or ``None``.

    The session-reconcile idle clock counts markers of ANY kind — not just
    :data:`_PROGRESS_KINDS` — because on a parked task every marker
    (`epm:followup-scope`, `epm:interp-critique`, `epm:workflow-fix-applied`,
    ...) is evidence somebody/something is still working the task, and the
    sweep must err toward keeping the session. Watcher-posted notes stay
    excluded (the alert/stop markers land on the very task whose inactivity
    they measure — counting them would reset the clock they read)."""
    best: float | None = None
    for ev in events:
        note = ev.get("note") or ""
        if any(sentinel in note for sentinel in _WATCHER_NOTE_SENTINELS):
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


# Filename prefix for the per-issue session-reconcile state file at
# ``~/.eps-autonomous/session-reconcile-<N>.json``. Mirrors the pod-safety
# state layout. NOT in :data:`_GC_TARGETS`: these files track episodes whose
# task is BY DEFINITION parked/terminal (completed/archived tasks sit in the
# terminal-status GC's sweep set), so that GC would reap them every tick and
# the miss counter could never reach the threshold. They are reaped by
# :func:`_gc_orphan_session_reconcile_state` (keyed on the live
# mapped-session set) plus its age backstop instead.
SESSION_RECONCILE_STATE_PREFIX = "session-reconcile-"


def _session_reconcile_autostop_enabled() -> bool:
    """True unless ``EPM_SESSION_RECONCILE_AUTOSTOP`` is explicitly set to a
    falsy value (``0`` / ``false`` / ``no``). Default ON as of 2026-06-10
    (user request: "Can we stop the happy sessions once they reach awaiting
    promotion?" — supersedes the same-day alert-only decision after 73 idle
    registered sessions accumulated ~35-40GB RSS and 14 were stopped manually
    with this pass's exact predicate). Setting the var to ``1``/``true``/
    ``yes`` (the old arming values) keeps the stop armed, so existing crontab
    exports stay backwards-compatible."""
    raw = os.environ.get("EPM_SESSION_RECONCILE_AUTOSTOP", "")
    return raw.strip().lower() not in {"0", "false", "no"}


def decide_session_reconcile(
    status: str | None,
    idle: bool,
    missed: int,
    alerted: bool,
    threshold: int = 2,
    *,
    autostop: bool = False,
    keep_running: bool = False,
    followup_active: bool = False,
    pod_running: bool = False,
) -> tuple[str, int]:
    """Pure decision for the session-reconcile pass on one issue's live,
    issue-mapped session(s). Returns ``(action, new_missed)`` where action is
    ``"clear"`` | ``"keep"`` | ``"alert"`` | ``"stop"`` |
    ``"keep-running-skip"`` | ``"followup-skip"`` | ``"pod-skip"``.

    The caller only invokes this for issues that HAVE at least one live
    mapped session; sessions with no issue mapping (PM / chat) never reach
    here.

    Cases:

    - ``status`` not in :data:`SESSION_RECONCILE_DONE` (including ``None`` =
      unreadable) -> ``("clear", 0)``. The task is not provably parked/done —
      any other status (ACTIVE, ``followups_running``, ``blocked``) means
      the session may be legitimately live, so the episode state is dropped.
      Unreadable status is treated as non-done (conservative: never act on
      ignorance).
    - done but not ``idle`` -> ``("clear", 0)``. Fresh activity (a non-watcher
      marker of ANY kind or self-report within :func:`_session_idle_s`) ends
      the episode — e.g. a task that JUST parked keeps its session for the
      grace window.
    - done + idle + ``keep_running`` -> ``("keep-running-skip", 0)``. The
      explicit user tag beats everything (same precedence as
      :func:`decide_pod_safety`); miss counter resets so removing the tag
      re-arms a fresh >=``threshold``-checks accumulation.
    - done + idle + ``followup_active`` (and not ``keep_running``) ->
      ``("followup-skip", 0)``. A fresh follow-up signal marker newer than
      the latest done-transition means an inline follow-up is in flight (or
      requested); its driver session must not be stopped even if the
      follow-up itself is quiet (markers > idle window — e.g. mid-training
      silence).
    - done + idle + ``pod_running`` (and neither skip above) ->
      ``("pod-skip", 0)``. A RUNNING managed pod on the issue means work may
      still be in flight that the markers haven't surfaced yet; the
      pod-safety pass owns reconciling the pod itself, and once it stops the
      escaped pod this skip re-arms naturally.
    - done + idle, below ``threshold`` -> ``("keep", missed+1)``. The 2-miss
      guard: a single transient task.py / self-report read glitch never
      escalates.
    - threshold met + ``autostop`` (the DEFAULT as of 2026-06-10) ->
      ``("stop", 0)``. Checked BEFORE the ``alerted`` dedup so arming the
      stop mid-episode escalates an already-alerted episode on the next tick
      without re-accumulating (the #506 lesson: a dedup flag must never
      suppress the stronger action once it becomes eligible).
    - threshold met, alert-only (``EPM_SESSION_RECONCILE_AUTOSTOP=0``), not
      yet ``alerted`` -> ``("alert", missed+1)``. One loud marker per
      episode; the miss count keeps accumulating so a later autostop-enable
      fires immediately.
    - threshold met, alert-only, already ``alerted`` -> ``("keep", missed+1)``.
      Stay quiet (dedup); the episode stays observable in the watcher log.
    """
    if status not in SESSION_RECONCILE_DONE:
        return ("clear", 0)
    if not idle:
        return ("clear", 0)
    if keep_running:
        return ("keep-running-skip", 0)
    if followup_active:
        return ("followup-skip", 0)
    if pod_running:
        return ("pod-skip", 0)
    new_missed = missed + 1
    if new_missed < threshold:
        return ("keep", new_missed)
    if autostop:
        return ("stop", 0)
    if not alerted:
        return ("alert", new_missed)
    return ("keep", new_missed)


def _map_sessions_to_issues(
    live_ids: set[str],
    registry_map: dict[str, int],
    session_paths: dict[str, str | None],
) -> dict[int, set[str]]:
    """Group live session ids by the issue they belong to.

    Pure (testable without a daemon): ``registry_map`` is
    ``spawn_session._load_session_issue_map()`` (registered sessions, BOTH
    ``issue-<N>.json`` and ``manual-issue-<N>.json``); ``session_paths`` maps
    sid -> cwd from ``~/.happy/sessions.json`` metadata. A registry mapping
    wins; an ``issue-<N>`` worktree cwd is the fallback for unregistered /
    superseded zombie generations (the respawn pass deletes the registry
    entry at TERMINAL, and every newer spawn overwrites it — so the incident
    sessions are mostly cwd-mapped, the same ``~#N`` attribution
    ``spawn_session.py list`` renders). Sessions with neither mapping (the
    PM session at the repo root, chat sessions, other projects) are skipped
    entirely — they can never be acted on."""
    out: dict[int, set[str]] = {}
    for sid in live_ids:
        if not isinstance(sid, str) or not sid:
            continue
        issue = registry_map.get(sid)
        if issue is None:
            issue = _infer_issue_from_path(session_paths.get(sid))
        if issue is None:
            continue
        out.setdefault(issue, set()).add(sid)
    return out


def _session_reconcile_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{SESSION_RECONCILE_STATE_PREFIX}{issue}.json"


def _load_session_reconcile_state(issue: int) -> dict:
    """Read the per-issue session-reconcile state (``{}`` if absent /
    unreadable — a fresh/garbled file starts the miss count at 0, mirroring
    :func:`_load_pod_safety_state`)."""
    path = _session_reconcile_state_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_session_reconcile_state(
    issue: int,
    *,
    missed: int,
    alerted: bool,
    sids: list[str],
    prev: dict | None = None,
    stopped_at: dict[str, float] | None = None,
    stop_retried: bool = False,
    stop_failed_alerted: bool = False,
) -> None:
    """Persist the per-issue session-reconcile state atomically (temp +
    rename), mirroring :func:`_save_pod_safety_state`. ``sids`` records the
    live session ids observed this tick (informational — the decision is
    per-issue); ``first_seen`` carries forward so the GC age backstop
    measures the original episode start.

    The stop-verification fields (all optional; absent in state files written
    before 2026-06-10, which read back as empty/false): ``stopped_at`` maps
    sid -> epoch ts of the daemon-ACKed stop, awaiting the next-tick
    gone-from-the-live-set verification; ``stop_retried`` /
    ``stop_failed_alerted`` are the once-per-episode dedup flags for the
    zombie-session retry + loud marker (:func:`_check_stop_verification`)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _session_reconcile_state_path(issue)
    prev_first_seen = (prev or {}).get("first_seen")
    if not isinstance(prev_first_seen, int | float):
        prev_first_seen = time.time()
    payload = {
        "missed": missed,
        "alerted": alerted,
        "sids": sorted(sids),
        "first_seen": prev_first_seen,
        "stopped_at": dict(stopped_at or {}),
        "stop_retried": bool(stop_retried),
        "stop_failed_alerted": bool(stop_failed_alerted),
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_session_reconcile_state(issue: int) -> None:
    """Drop the per-issue session-reconcile state file (episode over: the
    task left the DONE set, activity resumed, or the sessions were stopped)."""
    _session_reconcile_state_path(issue).unlink(missing_ok=True)


def _gc_orphan_session_reconcile_state(
    mapped_issues: set[int], dry_run: bool, now: float | None = None
) -> list[int]:
    """GC session-reconcile state files for issues with NO live mapped session
    (the sessions died / were stopped by any path — the episode is over), so
    a later session on the same issue starts with a fresh miss count. Also
    drops files older than :data:`POD_SAFETY_STATE_MAX_AGE_S` as an age
    backstop. Mirrors :func:`_gc_orphan_pod_safety_state` (the terminal-status
    GC deliberately does NOT sweep this prefix — see
    :data:`SESSION_RECONCILE_STATE_PREFIX`). This reap is ALSO the
    stop-verification success path: a stopped session that actually died
    leaves the mapped set, so its episode state — including ``stopped_at`` —
    is dropped here (:func:`_check_stop_verification` documents the zombie
    branch). Returns the cleared issues."""
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return []
    now = now if now is not None else time.time()
    cleared: list[int] = []
    for path in sorted(AUTONOMOUS_REGISTRY_DIR.glob(f"{SESSION_RECONCILE_STATE_PREFIX}*.json")):
        stem = path.stem[len(SESSION_RECONCILE_STATE_PREFIX) :]
        try:
            issue = int(stem)
        except ValueError:
            continue  # hand-debug artifact; not the GC's business
        if issue in mapped_issues:
            continue
        try:
            payload = json.loads(path.read_text())
            first_seen = payload.get("first_seen", now)
            if not isinstance(first_seen, int | float):
                first_seen = now
        except (json.JSONDecodeError, OSError):
            first_seen = 0  # unreadable -> definitely orphaned, drop it
        age = now - first_seen
        reason = (
            "no live mapped session"
            if age < POD_SAFETY_STATE_MAX_AGE_S
            else f"age={age / 3600:.1f}h"
        )
        print(f"  session-reconcile: GC orphan state issue #{issue} ({reason})")
        if not dry_run:
            path.unlink(missing_ok=True)
        cleared.append(issue)
    return cleared


def _session_idle_signals(issue: int, now: float) -> tuple[bool, str, list[dict]]:
    """Compute ``(idle, gap_desc, events)`` for a DONE-status candidate.

    ``idle`` is True when EVERY available activity signal — the newest
    NON-watcher marker of ANY kind (:func:`_latest_nonwatcher_event_ts`, not
    just progress kinds: on a parked task any marker is evidence the task is
    still being worked) and the per-issue self-report file — is older than
    :func:`_session_idle_s` (default 2h, env
    ``EPM_SESSION_RECONCILE_IDLE_S``). When NO signal is readable at all the
    issue counts as idle (mirrors the orphan sweep's None-is-stale rule; the
    status gate + follow-up/pod/keep-running skips + 2-miss guard keep that
    safe). ``gap_desc`` is the human-readable freshest-signal age for
    log/marker text; ``events`` is returned so the caller can reuse the
    fetch for the follow-up predicate."""
    events = _task_events(issue)
    latest_marker = _latest_nonwatcher_event_ts(events)
    sr_age, _sr_ts = _self_report_age_seconds(issue, now)
    ages = [
        a
        for a in (
            (now - latest_marker) if latest_marker is not None else None,
            sr_age,
        )
        if a is not None
    ]
    idle = (min(ages) >= _session_idle_s()) if ages else True
    gap_desc = f"{min(ages) / 3600:.1f}h" if ages else "no-signal"
    return idle, gap_desc, events


def _handle_session_stop(
    issue: int,
    sids: list[str],
    status: str | None,
    gap_desc: str,
    threshold: int,
    dry_run: bool,
    prev_state: dict,
    prev_missed: int,
    prev_alerted: bool,
    now: float,
) -> None:
    """Stop every live mapped session for ``issue`` and record the outcome.

    The daemon ACK is NOT trusted as a kill: every ACKed sid is recorded in
    the state file's ``stopped_at`` map and verified actually-gone on the
    NEXT tick (:func:`_check_stop_verification`); the episode state is
    cleared only once the session(s) leave the live set (via the
    live-session-keyed GC). An ACK failure keeps the accumulated miss count
    so the next tick retries the stop for the remaining live session(s)."""
    stopped = [sid for sid in sids if _stop_session(sid, dry_run)]
    if stopped:
        _post_progress_marker(
            issue,
            f"{_SESSION_RECONCILE_STOP_NOTE_SENTINEL} auto-stopped "
            f"{len(stopped)} idle session(s) ({', '.join(stopped)}) by the "
            f"autonomous_session_watch session-reconcile pass — task status "
            f"'{status}' is parked/terminal, no live follow-up signal, no "
            f"RUNNING pod, no keep-running tag, and no activity (non-watcher "
            f"marker / self-report) was observed for > "
            f"{_session_idle_s() / 3600:.1f}h (gap={gap_desc}), confirmed "
            f"for >= {threshold} checks. An idle session pins its worktree "
            f"against the stale-worktree sweep and holds deleted-file "
            f"handles (2026-06-10 disk incident). Respawn if needed: "
            f"`spawn_session.py spawn-issue --issue {issue}`.",
            dry_run,
            label="session-reconcile-stop",
        )
    if not dry_run:
        # Record every ACKed stop for next-tick verification instead of
        # clearing the episode: a daemon that ACKs but fails to kill the
        # session would otherwise reset the state and loop silently. The
        # state is reaped by the live-session-keyed GC once the session(s)
        # actually leave the live set. A full ACK resets the miss count
        # (the old clear's semantics); a partial ACK keeps it so the next
        # tick re-stops the remaining live session(s).
        stopped_at = dict(prev_state.get("stopped_at") or {})
        for sid in stopped:
            stopped_at[sid] = now
        _save_session_reconcile_state(
            issue,
            missed=0 if len(stopped) == len(sids) else prev_missed,
            alerted=prev_alerted,
            sids=sids,
            prev=prev_state,
            stopped_at=stopped_at,
            stop_retried=bool(prev_state.get("stop_retried", False)),
            stop_failed_alerted=bool(prev_state.get("stop_failed_alerted", False)),
        )


def _check_stop_verification(
    issue: int,
    sids: list[str],
    done: bool,
    idle: bool,
    prev_state: dict,
    dry_run: bool,
    now: float,
) -> bool:
    """Next-tick verification that a previously ACKed session stop actually
    landed (daemon ACK != kill). Returns True when this tick was consumed by
    the verification path (the caller skips the normal decision).

    ``stopped_at`` in the per-issue state records ``sid -> epoch ts`` for
    every session whose stop was ACKed (:func:`_handle_session_stop` no
    longer clears the episode on ACK). The verified-gone path needs no code
    here: once every stopped session has left the live set, either the issue
    drops out of the mapped set entirely (the live-session-keyed GC reaps
    the state file) or only NEW sessions remain (no zombie -> fall through;
    the next state save rewrites ``stopped_at`` empty, starting the
    newcomers on a clean slate).

    A ZOMBIE — a sid still in the live set on a later tick despite its ACKed
    stop — escalates, but only while the stop conditions (DONE status +
    idle) still hold (a revived / freshly-active task falls through to the
    normal decision, which clears the episode rather than re-killing a
    legitimately live session):

    1. first zombie tick: loud stderr log + ONE retry of the stop
       (``stop_retried`` flag);
    2. zombie after the retry: ONE loud marker on the task
       (``stop_failed_alerted`` flag) — the episode state is never cleared
       on an unverified stop, so the failure stays visible for triage;
    3. after the alert: stay quiet; the state file remains and is reaped by
       the live-session-keyed GC when the session finally dies (or by the
       age backstop).

    Backward-compatible: state files written before these fields existed
    have no ``stopped_at`` key -> empty dict -> the check is a no-op.
    """
    stopped_at = prev_state.get("stopped_at")
    if not isinstance(stopped_at, dict) or not stopped_at:
        return False
    if not (done and idle):
        return False  # stop conditions no longer hold; normal decide clears
    zombies = sorted(sid for sid in sids if sid in stopped_at)
    if not zombies:
        return False  # all stopped sids verified gone; newcomers fall through
    prev_missed = prev_state.get("missed", 0)
    prev_missed = prev_missed if isinstance(prev_missed, int) else 0
    prev_alerted = bool(prev_state.get("alerted", False))
    print(
        f"  STOP-VERIFY FAILED issue #{issue}: {len(zombies)} session(s) "
        f"({', '.join(zombies)}) still alive one tick after the daemon ACKed "
        f"their stop (ACK != kill).",
        file=sys.stderr,
    )
    if not prev_state.get("stop_retried"):
        re_acked = [sid for sid in zombies if _stop_session(sid, dry_run)]
        print(
            f"  session-reconcile: stop RETRIED for {len(re_acked)}/{len(zombies)} "
            f"zombie session(s) on #{issue} (one retry per episode)"
        )
        if not dry_run:
            new_stopped_at = dict(stopped_at)
            for sid in re_acked:
                new_stopped_at[sid] = now
            _save_session_reconcile_state(
                issue,
                missed=prev_missed,
                alerted=prev_alerted,
                sids=sids,
                prev=prev_state,
                stopped_at=new_stopped_at,
                stop_retried=True,
                stop_failed_alerted=bool(prev_state.get("stop_failed_alerted", False)),
            )
        return True
    if not prev_state.get("stop_failed_alerted"):
        _post_progress_marker(
            issue,
            f"{_SESSION_RECONCILE_STOP_FAILED_NOTE_SENTINEL} session STOP FAILED "
            f"to land: {len(zombies)} session(s) ({', '.join(zombies)}) are "
            f"still alive after the session-reconcile pass stopped them AND "
            f"retried once — the Happy daemon ACKed the stop RPCs but did not "
            f"kill the session(s). Stop manually with `spawn_session.py stop "
            f"--session-id <id>` (or restart the Happy daemon). The episode "
            f"state is kept (never cleared on an unverified stop) and is GC'd "
            f"once the session(s) actually leave the live set. Posted once "
            f"per episode.",
            dry_run,
            label="session-reconcile-stop-failed",
        )
        if not dry_run:
            _save_session_reconcile_state(
                issue,
                missed=prev_missed,
                alerted=prev_alerted,
                sids=sids,
                prev=prev_state,
                stopped_at=stopped_at,
                stop_retried=True,
                stop_failed_alerted=True,
            )
        return True
    print(
        f"  session-reconcile: issue #{issue} zombie session(s) {zombies} already "
        f"retried + alerted this episode; awaiting manual stop / daemon recovery."
    )
    return True


def _process_session_reconcile(
    issue: int,
    sids: list[str],
    now: float,
    dry_run: bool,
    threshold: int,
    *,
    autostop: bool,
    running_pod_issues: set[int] | None = None,
) -> None:
    """Reconcile one issue's live session(s) against its task status.

    Reads the task's status; for parked/terminal
    (awaiting_promotion/completed/archived) tasks, computes idleness via
    :func:`_session_idle_signals`. Applies :func:`decide_session_reconcile`
    and acts: STOP every live mapped session via ``spawn_session.py stop``
    (the default), or ALERT once per episode when
    ``EPM_SESSION_RECONCILE_AUTOSTOP=0``. ``running_pod_issues`` is the
    issue set with a RUNNING managed pod (computed once per pass); ``None``
    is treated as the empty set (unit-test convenience — production always
    passes the snapshot)."""
    status = _task_status(issue)
    done = status in SESSION_RECONCILE_DONE

    # Lazy: events / self-report / tag / follow-up reads are paid only for
    # DONE-status candidates (same lazy pattern as _process_pod).
    idle = False
    gap_desc = "n/a"
    keep_running = False
    followup_active = False
    pod_running = False
    if done:
        idle, gap_desc, events = _session_idle_signals(issue, now)
        if idle:
            keep_running = _task_keep_running(issue)
            followup_active = not keep_running and _task_session_followup_active(
                issue, events=events
            )
            pod_running = issue in (running_pod_issues or set())

    prev_state = _load_session_reconcile_state(issue)
    prev_missed = prev_state.get("missed", 0)
    if not isinstance(prev_missed, int):
        prev_missed = 0
    prev_alerted = bool(prev_state.get("alerted", False))

    # Next-tick stop verification (daemon ACK != kill): a previously-stopped
    # sid still in the live set consumes the tick (retry once, then a loud
    # one-time marker) — see :func:`_check_stop_verification`.
    if _check_stop_verification(issue, sids, done, idle, prev_state, dry_run, now):
        return

    action, new_missed = decide_session_reconcile(
        status,
        idle,
        prev_missed,
        prev_alerted,
        threshold,
        autostop=autostop,
        keep_running=keep_running,
        followup_active=followup_active,
        pod_running=pod_running,
    )
    print(
        f"  issue #{issue} sessions={len(sids)}: status={status} idle={idle} "
        f"activity_gap={gap_desc} missed={prev_missed}->{new_missed} "
        f"alerted={prev_alerted} action={action}"
    )

    if action == "clear":
        if prev_state and not dry_run:
            _clear_session_reconcile_state(issue)
        return

    # The three skip actions differ only in their audit log line; all three
    # reset the miss counter so removing the blocker re-arms a fresh
    # >=threshold accumulation.
    skip_msgs = {
        "keep-running-skip": (
            f"  KEEP-RUNNING issue #{issue}: task status '{status}' is DONE and the "
            f"session(s) are idle, but the keep-running tag is present — "
            f"session-reconcile SKIPPED (sids={sids})."
        ),
        "followup-skip": (
            f"  FOLLOWUP-ACTIVE issue #{issue}: task status '{status}' is DONE but a "
            f"fresh follow-up signal marker (run-launched / followup-scope / "
            f"free-analysis-followup-run, newer than the latest done-transition) "
            f"indicates a live or requested inline follow-up — session-reconcile "
            f"SKIPPED (sids={sids})."
        ),
        "pod-skip": (
            f"  POD-RUNNING issue #{issue}: task status '{status}' is DONE and the "
            f"session(s) are idle, but a RUNNING managed pod exists for the issue — "
            f"session-reconcile SKIPPED (sids={sids}); the pod-safety pass owns the "
            f"pod, and this skip re-arms once the pod leaves the RUNNING set."
        ),
    }
    if action in skip_msgs:
        print(skip_msgs[action])
        if not dry_run:
            _save_session_reconcile_state(
                issue, missed=0, alerted=prev_alerted, sids=sids, prev=prev_state
            )
        return

    if action == "stop":
        _handle_session_stop(
            issue,
            sids,
            status,
            gap_desc,
            threshold,
            dry_run,
            prev_state,
            prev_missed,
            prev_alerted,
            now,
        )
        return

    if action == "alert":
        print(
            f"  ALERT issue #{issue}: {len(sids)} live session(s) for a task at DONE "
            f"status '{status}' with no activity > {_session_idle_s() / 3600:.1f}h "
            f"(gap={gap_desc}); NOT stopping (EPM_SESSION_RECONCILE_AUTOSTOP=0 — "
            f"alert-only fallback).",
            file=sys.stderr,
        )
        _post_progress_marker(
            issue,
            f"{_SESSION_RECONCILE_ALERT_NOTE_SENTINEL} IDLE session(s) outliving a "
            f"parked/terminal task: {len(sids)} live Happy session(s) "
            f"({', '.join(sids)}) mapped to this task (status '{status}') with no "
            f"activity (non-watcher marker / self-report) for > "
            f"{_session_idle_s() / 3600:.1f}h (gap={gap_desc}). Idle sessions pin "
            f"their worktrees against the stale-worktree sweep and hold "
            f"deleted-file handles (2026-06-10 disk incident: ~37G phantom usage "
            f"across 15+ such sessions). NOT auto-stopped "
            f"(EPM_SESSION_RECONCILE_AUTOSTOP=0 alert-only fallback); stop "
            f"manually with `spawn_session.py stop --session-id <id>`, or unset "
            f"the env var on the watcher cron to restore the default auto-stop. "
            f"Posted once per episode.",
            dry_run,
            label="session-reconcile-alert",
        )
        if not dry_run:
            _save_session_reconcile_state(
                issue, missed=new_missed, alerted=True, sids=sids, prev=prev_state
            )
        return

    # action == "keep": persist the (possibly incremented) miss count.
    if not dry_run:
        _save_session_reconcile_state(
            issue, missed=new_missed, alerted=prev_alerted, sids=sids, prev=prev_state
        )


def session_reconcile_pass(
    dry_run: bool,
    threshold: int,
    *,
    daemon_reachable: bool,
    live_ids: set[str] | None = None,
    now: float | None = None,
) -> None:
    """Reconcile live Happy sessions against their task status.

    Daemon-gated like the respawn pass: session liveness is unknowable during
    a daemon outage, and the stop action itself POSTs to the daemon, so the
    whole pass skips when it is unreachable. ``live_ids`` may be passed in by
    ``main()`` to reuse its snapshot (one daemon round-trip per tick)."""
    now = now if now is not None else time.time()
    if not daemon_reachable:
        print(
            "session-reconcile: Happy daemon unreachable; skipping "
            "(session liveness unknowable during an outage)"
        )
        return
    live = live_ids if live_ids is not None else _live_session_ids()
    meta = _load_session_meta()
    session_paths = {sid: (m or {}).get("path") for sid, m in meta.items()}
    by_issue = _map_sessions_to_issues(live, _load_session_issue_map(), session_paths)

    # GC stale state ALWAYS — even with zero mapped sessions — so an episode
    # whose sessions died/were stopped by any path gets a fresh start later.
    _gc_orphan_session_reconcile_state(set(by_issue), dry_run, now=now)

    if not by_issue:
        print("session-reconcile: no live issue-mapped sessions")
        return
    n_sessions = sum(len(v) for v in by_issue.values())
    autostop = _session_reconcile_autostop_enabled()
    # One live-pod snapshot per pass (the per-issue check is a set lookup).
    # A FAILED snapshot (None) degrades to an empty set — the followup/
    # keep-running skips, the idle grace, and the 2-miss guard remain as
    # safety margins, and the pod-safety pass independently reconciles the
    # pod itself (it skips its own state GC on the failed snapshot).
    running_pod_issues = {
        issue
        for issue, _pod_id, _name in (_running_managed_issue_pods(caller="session-reconcile") or [])
    }
    print(
        f"session-reconcile: {n_sessions} live issue-mapped session(s) across "
        f"{len(by_issue)} issue(s) "
        f"(autostop={'ON' if autostop else 'OFF — alert-only (EPM_SESSION_RECONCILE_AUTOSTOP=0)'})"
    )
    for issue in sorted(by_issue):
        _process_session_reconcile(
            issue,
            sorted(by_issue[issue]),
            now,
            dry_run,
            threshold,
            autostop=autostop,
            running_pod_issues=running_pod_issues,
        )


# ─── zombie-wrapper pass (dead inner Claude; 2026-06-11 zombie sweep) ────────
#
# Targets the failure mode NO other session pass can see: a daemon-tracked
# Happy node wrapper that is alive (so the respawn pass keeps clear) but whose
# inner Claude process is gone, on a session with NO usable issue mapping (so
# the session-reconcile pass — registry- or worktree-cwd-keyed — never reaches
# it). On 2026-06-11, 25 such sessions had accumulated: all finished issue
# sessions ("Waiting for user to promote #511/#514/...") whose registrations
# had been GC'd and whose cwd was the repo root, showing as "running" in
# `spawn_session.py list` indefinitely until a manual sweep.
#
# CONSERVATIVE by verified design (NOT just habit): the Happy wrapper's
# remote-mode launcher loops `claudeRemote`, which blocks on `nextMessage()`
# BEFORE spawning the Claude SDK subprocess — so a wrapper with no Claude
# descendant can be a HEALTHY idle session (e.g. right after a /clear or an
# abort) that the next phone message revives IN PLACE. A no-Claude snapshot
# is therefore necessary but not sufficient. The stop fires only when ALL
# hold:
#
#   * NO Claude process anywhere in the wrapper's /proc descendant tree
#     (cmdline match on :data:`_CLAUDE_CMDLINE_MARKERS` — both the native
#     installer's `claude/versions/<v>` binary and the SDK-bundled
#     `claude-agent-sdk-*/claude` are recognized);
#   * confirmed across >= ``threshold`` consecutive checks (transient
#     /proc-vs-daemon races never escalate);
#   * the FIRST no-Claude observation is older than
#     :func:`_zombie_wrapper_grace_s` (default 2h) — the in-place-revival
#     window for a healthy idle wrapper;
#   * the session is NOT the PM session (excluded via the explicit
#     ``pm-session.json`` registration — ``spawn-pm`` / ``register-pm`` /
#     the `/pm` skill bootstrap write it);
#   * the session's cwd IS under the EPS project root (other projects'
#     sessions are never touched);
#   * when the session IS issue-mapped (registry entry or ``issue-<N>``
#     worktree cwd), the task's status is NOT in
#     :data:`ZOMBIE_STATUS_EXCLUDE` (an active/blocked/plan-pending task's
#     session is left to the passes that own those states).
#
# ``EPM_ZOMBIE_WRAPPER_REAP=0`` falls back to ALERT-ONLY (the
# EPM_SESSION_RECONCILE_AUTOSTOP pattern). Stops are verified next tick
# (daemon ACK != kill): one retry, then one loud marker, mirroring
# :func:`_check_stop_verification`. Daemon-gated (needs /list pids + the
# stop RPC). Stopping a live wrapper forfeits daemon-side `happy resume`
# tracking, but the recovery story for reaped sessions is a fresh
# `spawn_session.py spawn-issue` — same contract as the session-reconcile
# stop.

# Filename prefix for the per-SESSION state file at
# ``~/.eps-autonomous/zombie-wrapper-<sid>.json``. Keyed by session id (NOT
# issue — the target class is precisely the sessions without a usable issue
# mapping). NOT in the terminal-status GC's sweep set; reaped by its own
# live-session-keyed GC (:func:`_gc_orphan_zombie_state`).
ZOMBIE_WRAPPER_STATE_PREFIX = "zombie-wrapper-"

# Default grace window between the FIRST no-Claude observation and any stop.
# 2h mirrors SESSION_IDLE_S: long enough that a healthy idle wrapper the user
# walked away from (post-/clear, post-abort) is overwhelmingly likely to be
# revived or remain wanted, short enough that zombie accumulation is bounded
# to a workday. Override via EPM_ZOMBIE_WRAPPER_GRACE_S (seconds).
ZOMBIE_WRAPPER_GRACE_S = 2 * 3600

# Issue-mapped sessions whose task sits in any of these statuses are NEVER
# touched by the zombie pass — active pipeline statuses are owned by the
# respawn/stalled/orphan passes, and `blocked` / `plan_pending` may have the
# user live-parked in the session. The reapable remainder (`proposed`,
# `awaiting_promotion`, `completed`, `archived`) plus unmapped sessions and
# unreadable statuses (conservative: cleared, see decide) define the scope.
ZOMBIE_STATUS_EXCLUDE = frozenset(ACTIVE | {"plan_pending", "blocked"})

# Substrings that identify an inner Claude process in /proc/<pid>/cmdline.
# Two install shapes observed live on this VM (2026-06-11): the native
# installer runs `~/.local/share/claude/versions/<v>` and the Happy-bundled
# SDK runs `.../@anthropic-ai/claude-agent-sdk-linux-x64/claude`. Substring
# match errs toward false KEEPS (an unrelated cmdline mentioning these paths
# keeps the session alive), never false stops.
_CLAUDE_CMDLINE_MARKERS = ("claude/versions/", "claude-agent-sdk")


def _zombie_wrapper_reap_enabled() -> bool:
    """True unless ``EPM_ZOMBIE_WRAPPER_REAP`` is explicitly set to a falsy
    value (``0`` / ``false`` / ``no``) — the alert-only kill-switch, same
    parsing as :func:`_session_reconcile_autostop_enabled`."""
    raw = os.environ.get("EPM_ZOMBIE_WRAPPER_REAP", "")
    return raw.strip().lower() not in {"0", "false", "no"}


def _zombie_wrapper_grace_s() -> float:
    """Grace window in seconds: ``EPM_ZOMBIE_WRAPPER_GRACE_S`` when set to a
    positive number, else :data:`ZOMBIE_WRAPPER_GRACE_S` (2h). Garbled /
    non-positive values fall back to the default."""
    raw = os.environ.get("EPM_ZOMBIE_WRAPPER_GRACE_S", "")
    try:
        val = float(raw)
    except ValueError:
        return ZOMBIE_WRAPPER_GRACE_S
    return val if val > 0 else ZOMBIE_WRAPPER_GRACE_S


def _proc_children_map() -> dict[int, list[int]]:
    """``ppid -> [child pids]`` from ONE /proc scan (Linux-only, matching the
    VM runtime). Computed once per pass and shared across every wrapper's
    descendant walk. Unreadable /proc entries (raced exits) are skipped."""
    out: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            # comm (field 2) can contain spaces/parens; ppid is the 2nd
            # whitespace field after the LAST ')' (same parse as
            # spawn_session._ancestor_pids).
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        out.setdefault(ppid, []).append(int(entry.name))
    return out


def _cmdline_has_claude_marker(pid: int) -> bool:
    """True iff ``/proc/<pid>/cmdline`` contains any
    :data:`_CLAUDE_CMDLINE_MARKERS` substring. Unreadable (exited) -> False."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    return any(marker in cmd for marker in _CLAUDE_CMDLINE_MARKERS)


def _has_claude_descendant(pid: int, children_map: dict[int, list[int]] | None = None) -> bool:
    """True iff ``pid`` or any /proc descendant has a Claude cmdline marker.

    The liveness key of the zombie-wrapper pass: the daemon's ``/list``
    ``pid`` is the Happy node wrapper, an ancestor of the Claude SDK
    subprocess it spawns per query. The wrapper itself is included in the
    walk defensively (its own cmdline — ``node .../happy/dist/index.mjs
    claude ...`` — matches no marker, verified live, so this can only err
    toward a false KEEP)."""
    if children_map is None:
        children_map = _proc_children_map()
    seen: set[int] = set()
    stack = [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        if _cmdline_has_claude_marker(p):
            return True
        stack.extend(children_map.get(p, ()))
    return False


def decide_zombie_wrapper(
    status: str | None,
    mapped: bool,
    has_claude: bool,
    missed: int,
    first_miss_age_s: float,
    alerted: bool,
    threshold: int = 2,
    *,
    reap_enabled: bool = True,
    grace_s: float = ZOMBIE_WRAPPER_GRACE_S,
) -> tuple[str, int]:
    """Pure decision for one live, non-PM, EPS-cwd session. Returns
    ``(action, new_missed)`` with action ``"clear"`` | ``"keep"`` |
    ``"stop"`` | ``"alert"``.

    Cases:

    - ``mapped`` AND (``status`` unreadable OR in
      :data:`ZOMBIE_STATUS_EXCLUDE`) -> ``("clear", 0)``. An issue-mapped
      session at an active/blocked/plan-pending (or unknowable) status is
      out of scope — other passes own those states. Unmapped sessions have
      no status to consult, so ``status`` is ignored for them.
    - Claude process present anywhere in the wrapper's tree ->
      ``("clear", 0)``. The session is (or just became) healthy; the
      episode ends and a later no-Claude observation starts fresh.
    - No Claude, below ``threshold`` consecutive misses OR within
      ``grace_s`` of the FIRST miss -> ``("keep", missed+1)``. The grace
      window is the in-place-revival margin: a healthy wrapper blocked at
      ``nextMessage()`` (post-/clear, post-abort) has no Claude child yet
      revives on the next phone message.
    - Threshold + grace met, ``reap_enabled`` (default) -> ``("stop", 0)``.
    - Threshold + grace met, kill-switch fallback, not yet ``alerted`` ->
      ``("alert", missed+1)`` — one loud marker per episode; the count
      keeps accumulating so a later re-enable stops on the next tick.
    - Otherwise -> ``("keep", missed+1)`` (alert-only, already alerted).
    """
    if mapped and (status is None or status in ZOMBIE_STATUS_EXCLUDE):
        return ("clear", 0)
    if has_claude:
        return ("clear", 0)
    new_missed = missed + 1
    if new_missed < threshold or first_miss_age_s < grace_s:
        return ("keep", new_missed)
    if reap_enabled:
        return ("stop", 0)
    if not alerted:
        return ("alert", new_missed)
    return ("keep", new_missed)


def _zombie_state_path(sid: str) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{ZOMBIE_WRAPPER_STATE_PREFIX}{sid}.json"


def _load_zombie_state(sid: str) -> dict:
    """Per-session zombie-wrapper state (``{}`` if absent/garbled — a fresh
    or unreadable file starts the miss count at 0, mirroring the other
    watcher state loaders)."""
    path = _zombie_state_path(sid)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_zombie_state(
    sid: str,
    *,
    missed: int,
    alerted: bool,
    pid: int,
    issue: int | None,
    first_miss_ts: float,
    stopped_at: float | None = None,
    stop_retried: bool = False,
    stop_failed_alerted: bool = False,
) -> None:
    """Persist the per-session zombie state atomically (temp + rename).
    ``first_miss_ts`` anchors BOTH the grace window and the GC age backstop;
    ``pid`` / ``issue`` are informational (the decision keys on the live
    daemon snapshot each tick). The stop-verification fields mirror the
    session-reconcile contract (ACK != kill)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _zombie_state_path(sid)
    payload = {
        "missed": missed,
        "alerted": alerted,
        "pid": pid,
        "issue": issue,
        "first_miss_ts": first_miss_ts,
        "stopped_at": stopped_at,
        "stop_retried": bool(stop_retried),
        "stop_failed_alerted": bool(stop_failed_alerted),
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_zombie_state(sid: str) -> None:
    """Drop the per-session zombie state (episode over: Claude reappeared,
    the session left scope, or it was verified stopped)."""
    _zombie_state_path(sid).unlink(missing_ok=True)


def _gc_orphan_zombie_state(live_sids: set[str], dry_run: bool, now: float | None = None) -> None:
    """GC zombie-wrapper state for sessions no longer in the daemon's live
    set (stopped by any path — the episode is over; this reap is also the
    stop-verification success path). EVERY non-live sid's file is reaped
    immediately; the ``first_miss_ts`` age comparison only picks the logged
    reason (just-departed vs ancient), not a separate retention rule — a
    live episode's file never needs age-reaping because its sid stays in
    the live set."""
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return
    now = now if now is not None else time.time()
    for path in sorted(AUTONOMOUS_REGISTRY_DIR.glob(f"{ZOMBIE_WRAPPER_STATE_PREFIX}*.json")):
        sid = path.stem[len(ZOMBIE_WRAPPER_STATE_PREFIX) :]
        if sid in live_sids:
            continue
        try:
            payload = json.loads(path.read_text())
            first_miss = payload.get("first_miss_ts", now)
            if not isinstance(first_miss, int | float):
                first_miss = now
        except (json.JSONDecodeError, OSError):
            first_miss = 0  # unreadable -> definitely orphaned, drop it
        age = now - first_miss
        reason = (
            "session left the live set"
            if age < POD_SAFETY_STATE_MAX_AGE_S
            else f"age={age / 3600:.1f}h"
        )
        print(f"  zombie-wrapper: GC orphan state {sid} ({reason})")
        if not dry_run:
            path.unlink(missing_ok=True)


def _append_zombie_fallback_event(note: str, dry_run: bool) -> None:
    """Durable trace for zombie actions on sessions with NO issue mapping —
    there is no task to carry the marker, so append one JSON line to
    ``~/.eps-autonomous/zombie-wrapper-events.jsonl`` (same role as the
    vm-disk fallback file). Fail-soft."""
    dest = AUTONOMOUS_REGISTRY_DIR / "zombie-wrapper-events.jsonl"
    line = json.dumps(
        {"ts": datetime.now().astimezone().isoformat(), "kind": "zombie-wrapper", "note": note}
    )
    if dry_run:
        print(f"  [dry-run] would append zombie-wrapper event to {dest}")
        return
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as e:
        print(f"  WARNING: appending zombie-wrapper event failed: {e}", file=sys.stderr)


def _zombie_record(issue: int | None, note: str, dry_run: bool, *, label: str) -> None:
    """Route a zombie-pass annotation: marker on the mapped issue when one
    exists, else the registration-independent fallback events file."""
    if issue is not None:
        _post_progress_marker(issue, note, dry_run, label=label)
    else:
        _append_zombie_fallback_event(note, dry_run)


def _check_zombie_stop_verification(
    sid: str,
    pid: int,
    issue: int | None,
    in_scope: bool,
    prev: dict,
    dry_run: bool,
    now: float,
) -> bool:
    """Next-tick verification that an ACKed zombie stop landed (ACK != kill).
    Returns True when this tick was consumed by the verification path.

    ``in_scope`` is the caller's current read of the stop conditions (still
    no Claude + still in reapable scope); when it no longer holds, fall
    through to the normal decision (which clears the episode rather than
    re-killing a revived session). The verified-gone path needs no code:
    a stopped sid leaves the live set and the live-session-keyed GC reaps
    the state. A still-live sid escalates: one stop retry, then one loud
    record, then quiet (same ladder as :func:`_check_stop_verification`)."""
    stopped_at = prev.get("stopped_at")
    if not isinstance(stopped_at, int | float) or not stopped_at:
        return False
    if not in_scope:
        return False
    first_miss_ts = prev.get("first_miss_ts")
    if not isinstance(first_miss_ts, int | float):
        first_miss_ts = now
    print(
        f"  ZOMBIE STOP-VERIFY FAILED session {sid}: still alive one tick "
        f"after the daemon ACKed its stop (ACK != kill).",
        file=sys.stderr,
    )
    common = dict(
        missed=prev.get("missed", 0) if isinstance(prev.get("missed", 0), int) else 0,
        alerted=bool(prev.get("alerted", False)),
        pid=pid,
        issue=issue,
        first_miss_ts=first_miss_ts,
    )
    if not prev.get("stop_retried"):
        acked = _stop_session(sid, dry_run)
        print(f"  zombie-wrapper: stop RETRIED for {sid} (one retry per episode, acked={acked})")
        if not dry_run:
            _save_zombie_state(
                sid,
                **common,
                stopped_at=now if acked else stopped_at,
                stop_retried=True,
                stop_failed_alerted=bool(prev.get("stop_failed_alerted", False)),
            )
        return True
    if not prev.get("stop_failed_alerted"):
        _zombie_record(
            issue,
            f"{_ZOMBIE_WRAPPER_STOP_FAILED_NOTE_SENTINEL} zombie-session STOP FAILED "
            f"to land: session {sid} (wrapper pid {pid}) is still alive after the "
            f"zombie-wrapper pass stopped it AND retried once — the Happy daemon "
            f"ACKed the stop RPCs but did not kill the wrapper. Stop manually with "
            f"`spawn_session.py stop --session-id {sid}` (or restart the Happy "
            f"daemon). Posted once per episode.",
            dry_run,
            label="zombie-wrapper-stop-failed",
        )
        if not dry_run:
            _save_zombie_state(
                sid,
                **common,
                stopped_at=stopped_at,
                stop_retried=True,
                stop_failed_alerted=True,
            )
        return True
    print(
        f"  zombie-wrapper: session {sid} already retried + alerted this episode; "
        f"awaiting manual stop / daemon recovery."
    )
    return True


def _process_zombie_wrapper(
    sid: str,
    pid: int,
    issue: int | None,
    now: float,
    dry_run: bool,
    threshold: int,
    *,
    reap_enabled: bool,
    children_map: dict[int, list[int]],
) -> None:
    """Apply the zombie-wrapper decision to one live, non-PM, EPS-cwd
    session: read the mapped task's status (when mapped), walk the wrapper's
    /proc tree for a Claude process, and act per
    :func:`decide_zombie_wrapper`."""
    status = _task_status(issue) if issue is not None else None
    has_claude = _has_claude_descendant(pid, children_map)

    prev = _load_zombie_state(sid)
    prev_missed = prev.get("missed", 0)
    if not isinstance(prev_missed, int):
        prev_missed = 0
    prev_alerted = bool(prev.get("alerted", False))
    first_miss_ts = prev.get("first_miss_ts")
    if not isinstance(first_miss_ts, int | float):
        first_miss_ts = now

    mapped = issue is not None
    in_scope = not has_claude and not (
        mapped and (status is None or status in ZOMBIE_STATUS_EXCLUDE)
    )
    if _check_zombie_stop_verification(sid, pid, issue, in_scope, prev, dry_run, now):
        return

    grace_s = _zombie_wrapper_grace_s()
    action, new_missed = decide_zombie_wrapper(
        status,
        mapped,
        has_claude,
        prev_missed,
        now - first_miss_ts,
        prev_alerted,
        threshold,
        reap_enabled=reap_enabled,
        grace_s=grace_s,
    )
    issue_label = f"#{issue}" if issue is not None else "unmapped"
    zombie_age_h = (now - first_miss_ts) / 3600 if not has_claude else 0.0
    print(
        f"  session {sid} (pid={pid}, issue={issue_label}): status={status} "
        f"has_claude={has_claude} missed={prev_missed}->{new_missed} "
        f"zombie_age={zombie_age_h:.1f}h action={action}"
    )

    if action == "clear":
        if prev and not dry_run:
            _clear_zombie_state(sid)
        return

    if action == "stop":
        acked = _stop_session(sid, dry_run)
        if acked:
            _zombie_record(
                issue,
                f"{_ZOMBIE_WRAPPER_STOP_NOTE_SENTINEL} auto-stopped zombie Happy "
                f"session {sid} (wrapper pid {pid}, issue {issue_label}): its process "
                f"tree carried NO inner Claude process for {zombie_age_h:.1f}h "
                f"(>= {threshold} consecutive checks, grace {grace_s / 3600:.1f}h). "
                f"The node wrapper outlived its Claude process and would show as "
                f"'running' indefinitely (2026-06-11: 25 such sessions accumulated, "
                f"invisible to the session-reconcile pass once unmapped). Respawn "
                f"if needed: `spawn_session.py spawn-issue --issue <N>` (or "
                f"`spawn-pm`). Set EPM_ZOMBIE_WRAPPER_REAP=0 on the watcher cron "
                f"to fall back to alert-only.",
                dry_run,
                label="zombie-wrapper-stop",
            )
        if not dry_run:
            _save_zombie_state(
                sid,
                missed=0 if acked else prev_missed,
                alerted=prev_alerted,
                pid=pid,
                issue=issue,
                first_miss_ts=first_miss_ts,
                stopped_at=now if acked else None,
                stop_retried=bool(prev.get("stop_retried", False)),
                stop_failed_alerted=bool(prev.get("stop_failed_alerted", False)),
            )
        return

    if action == "alert":
        print(
            f"  ZOMBIE ALERT session {sid} (issue {issue_label}): no inner Claude "
            f"process for {zombie_age_h:.1f}h; NOT stopping "
            f"(EPM_ZOMBIE_WRAPPER_REAP=0 — alert-only fallback).",
            file=sys.stderr,
        )
        _zombie_record(
            issue,
            f"{_ZOMBIE_WRAPPER_ALERT_NOTE_SENTINEL} ZOMBIE Happy session: {sid} "
            f"(wrapper pid {pid}, issue {issue_label}) has carried NO inner Claude "
            f"process for {zombie_age_h:.1f}h — the wrapper outlived its Claude and "
            f"shows as 'running' indefinitely. NOT auto-stopped "
            f"(EPM_ZOMBIE_WRAPPER_REAP=0 alert-only fallback); stop manually with "
            f"`spawn_session.py stop --session-id {sid}`, or unset the env var on "
            f"the watcher cron to restore the default reap. Posted once per episode.",
            dry_run,
            label="zombie-wrapper-alert",
        )
        if not dry_run:
            _save_zombie_state(
                sid,
                missed=new_missed,
                alerted=True,
                pid=pid,
                issue=issue,
                first_miss_ts=first_miss_ts,
            )
        return

    # action == "keep": persist the incremented miss count + episode anchor.
    if not dry_run:
        _save_zombie_state(
            sid,
            missed=new_missed,
            alerted=prev_alerted,
            pid=pid,
            issue=issue,
            first_miss_ts=first_miss_ts,
        )


def zombie_wrapper_pass(
    dry_run: bool,
    threshold: int,
    *,
    daemon_reachable: bool,
    children: list[dict] | None = None,
    now: float | None = None,
) -> None:
    """Auto-stop daemon-tracked Happy sessions whose process tree has carried
    no inner Claude process for >= ``threshold`` checks AND >= the grace
    window — REGARDLESS of issue mapping (the gap every registry-/cwd-keyed
    pass shares). Exclusions: PM-registered sids, non-EPS cwds, and
    issue-mapped sessions at :data:`ZOMBIE_STATUS_EXCLUDE` statuses.

    Daemon-gated like the respawn pass: the wrapper pids come from the
    daemon's ``/list`` and the stop action POSTs to it. ``children`` may be
    injected (tests / a caller reusing its snapshot); ``None`` fetches via
    :func:`_live_children`."""
    now = now if now is not None else time.time()
    if not daemon_reachable:
        print(
            "zombie-wrapper: Happy daemon unreachable; skipping "
            "(wrapper pids + the stop RPC both need the daemon)"
        )
        return
    children = children if children is not None else _live_children()
    live_sids = {
        c.get("happySessionId") for c in children if isinstance(c.get("happySessionId"), str)
    }
    # GC ALWAYS on a daemon-reachable tick — even with zero candidates — so
    # episodes whose session died/was stopped by any path start fresh later.
    _gc_orphan_zombie_state(live_sids, dry_run, now=now)
    if not children:
        print("zombie-wrapper: no live daemon-tracked sessions")
        return

    registry_map = _load_session_issue_map()
    meta = _load_session_meta()
    pm_sids = _load_pm_session_ids()
    project_prefix = str(PROJECT_ROOT)
    candidates: list[tuple[str, int, int | None]] = []
    skipped_pm = 0
    skipped_non_eps = 0
    for child in children:
        sid = child.get("happySessionId")
        pid = child.get("pid")
        if not isinstance(sid, str) or not sid or not isinstance(pid, int):
            continue
        if sid in pm_sids:
            skipped_pm += 1
            continue
        path = (meta.get(sid) or {}).get("path")
        if not isinstance(path, str) or not (
            path == project_prefix or path.startswith(project_prefix + "/")
        ):
            # Non-EPS cwd (other projects) or no cwd metadata at all: never
            # touched — EPS-ness cannot be established, so err toward keep.
            skipped_non_eps += 1
            continue
        issue = registry_map.get(sid)
        if issue is None:
            issue = _infer_issue_from_path(path)
        candidates.append((sid, pid, issue))

    reap = _zombie_wrapper_reap_enabled()
    print(
        f"zombie-wrapper: {len(candidates)} EPS session(s) scanned "
        f"({skipped_pm} PM-registered + {skipped_non_eps} non-EPS skipped; "
        f"reap={'ON' if reap else 'OFF — alert-only (EPM_ZOMBIE_WRAPPER_REAP=0)'})"
    )
    if not candidates:
        return
    children_map = _proc_children_map()
    for sid, pid, issue in sorted(candidates):
        _process_zombie_wrapper(
            sid,
            pid,
            issue,
            now,
            dry_run,
            threshold,
            reap_enabled=reap,
            children_map=children_map,
        )


# ─── idle-unmapped-session pass ──────────────────────────────────────────────
#
# The third session reaper, closing the class BOTH earlier reapers
# structurally exclude (2026-06-12 VM-lag incident): 25 unmapped Happy
# sessions sat idle 19-43h each with a LIVE inner Claude plus ~8 MCP server
# children, holding ~23 GB RSS total. The zombie-wrapper pass only fires when
# the tree has NO inner Claude; the session-reconcile pass only touches
# issue-MAPPED sessions. So idle unmapped sessions accumulated without bound
# until a manual sweep. This pass auto-stops an unmapped EPS-cwd session whose
# resolved Claude transcript has been idle past the reap window (default 12h,
# env EPM_UNMAPPED_IDLE_REAP_S) on >= threshold consecutive checks.
#
# Idleness signal: the mtime of the session's Claude transcript jsonl,
# resolved per-wrapper-pid via session_resolver's HAPPY-LOG path ONLY
# (authoritative, per-pid; the resolver's shared-projects-dir filesystem
# fallback is deliberately rejected — it can attribute another session's
# OLDER transcript, i.e. a WRONG signal rather than a missing one — see
# _transcript_idle_age_s). An active turn appends to the transcript
# continuously; an idle session does not. An UNRESOLVABLE signal FAILS
# TOWARD KEEP: the session is skipped with a loud log line and its episode
# state is left frozen — never reaped on missing data.
#
# Never touched: the PM session (pm-session.json registration), non-EPS
# cwds, issue-MAPPED sessions (registry entry or issue-<N> worktree cwd —
# the reconcile/zombie passes own those), and sessions whose wrapper holds a
# controlling TTY (a terminal-run `happy claude` Thomas may be sitting at;
# daemon-RPC-spawned sessions are headless, so the TTY test is a strict
# superset of "tmux client attached" and errs toward keep).

IDLE_UNMAPPED_STATE_PREFIX = "idle-unmapped-"

# Default transcript-idle window before an unmapped session is reapable.
# 12h: long enough that an overnight pause in a chat session Thomas means to
# come back to survives, short enough that the accumulation class (19-43h
# idle in the incident) is cleared within a day. Override via
# EPM_UNMAPPED_IDLE_REAP_S (seconds).
UNMAPPED_IDLE_REAP_S = 12 * 3600


def _unmapped_idle_reap_enabled() -> bool:
    """True unless ``EPM_UNMAPPED_IDLE_REAP`` is explicitly set to a falsy
    value (``0`` / ``false`` / ``no``) — the alert-only kill-switch, same
    parsing as :func:`_zombie_wrapper_reap_enabled`."""
    raw = os.environ.get("EPM_UNMAPPED_IDLE_REAP", "")
    return raw.strip().lower() not in {"0", "false", "no"}


def _unmapped_idle_reap_s() -> float:
    """Idle reap window in seconds: ``EPM_UNMAPPED_IDLE_REAP_S`` when set to
    a positive number, else :data:`UNMAPPED_IDLE_REAP_S` (12h). Garbled /
    non-positive values fall back to the default."""
    raw = os.environ.get("EPM_UNMAPPED_IDLE_REAP_S", "")
    try:
        val = float(raw)
    except ValueError:
        return UNMAPPED_IDLE_REAP_S
    return val if val > 0 else UNMAPPED_IDLE_REAP_S


def _wrapper_has_controlling_tty(pid: int) -> bool:
    """True iff ``/proc/<pid>/stat`` reports a controlling TTY (tty_nr != 0).

    A terminal-run ``happy claude`` wrapper holds its terminal's TTY; the
    daemon's RPC-spawned sessions are headless (tty_nr == 0). Used as the
    "Thomas may literally be looking at this session" guard. Unreadable
    /proc (raced exit / perms) -> True, failing toward keep. (Note the
    asymmetry vs the transcript signal: a True here maps to action
    ``"clear"`` — RESETTING any accumulated episode — while a missing
    transcript signal FREEZES it. Both directions fail toward keep; a
    flapping /proc read merely restarts accumulation.)"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # comm (field 2) can contain spaces/parens; fields after the LAST ')'
        # are state(0) ppid(1) pgrp(2) session(3) tty_nr(4) — same parse as
        # _proc_children_map.
        tty_nr = int(stat.rsplit(")", 1)[1].split()[4])
    except (OSError, IndexError, ValueError):
        return True
    return tty_nr != 0


def _transcript_idle_age_s(node_pid: int, now: float) -> tuple[float | None, str | None]:
    """Seconds since the session's resolved Claude transcript was last
    written, or ``(None, reason)`` when the signal is unavailable.

    Resolution via the HAPPY-LOG path ONLY
    (:func:`session_resolver._resolve_transcript_via_happy_log` — per-pid
    and authoritative; every daemon-spawned wrapper writes one, which
    covers the incident class). The resolver's filesystem fallback is
    deliberately NOT used: it scans the cwd-derived projects dir, which
    for repo-root chat sessions is SHARED across sessions and full of
    other sessions' transcripts — its /issue-headed preference can
    attribute an OLDER, WRONG transcript whose stale mtime would read as
    days-idle and stop a genuinely fresh session. A wrong signal is worse
    than a missing one, so a happy-log miss is NOT an idleness verdict —
    the caller must fail toward keep."""
    transcript, reason = session_resolver._resolve_transcript_via_happy_log(node_pid)
    if transcript is None:
        return None, reason or "transcript unresolvable"
    try:
        mtime = Path(transcript).stat().st_mtime
    except OSError as e:
        return None, f"transcript stat failed: {type(e).__name__}"
    return max(0.0, now - mtime), None


def decide_idle_unmapped(
    mapped: bool,
    has_tty: bool,
    idle_age_s: float | None,
    missed: int,
    alerted: bool,
    threshold: int = 2,
    *,
    reap_enabled: bool = True,
    idle_reap_s: float = UNMAPPED_IDLE_REAP_S,
) -> tuple[str, int]:
    """Pure decision for one live, non-PM, EPS-cwd session. Returns
    ``(action, new_missed)`` with action ``"clear"`` | ``"skip"`` |
    ``"keep"`` | ``"stop"`` | ``"alert"``.

    Cases:

    - ``mapped`` -> ``("clear", 0)``. Issue-mapped sessions belong to the
      session-reconcile / zombie passes; a session that GAINS a mapping
      mid-episode (resolver backfill) leaves scope and its state clears.
    - ``has_tty`` -> ``("clear", 0)``. A controlling TTY means a terminal
      Thomas may be sitting at; the episode ends.
    - ``idle_age_s is None`` -> ``("skip", missed)``. The idleness signal is
      unavailable — fail toward keep, FREEZE the count (no increment, no
      reset: a flapping resolver neither accumulates toward a stop nor
      erases a real episode).
    - ``idle_age_s < idle_reap_s`` -> ``("clear", 0)``. Recent activity;
      episode over.
    - Over the window but below ``threshold`` consecutive checks ->
      ``("keep", missed+1)``.
    - Threshold met, ``reap_enabled`` (default) -> ``("stop", 0)``.
    - Threshold met, kill-switch fallback, not yet ``alerted`` ->
      ``("alert", missed+1)`` — one loud record per episode; the count keeps
      accumulating so a later re-enable stops on the next tick.
    - Otherwise -> ``("keep", missed+1)`` (alert-only, already alerted).

    Unlike the zombie pass there is no separate grace window: the idle age
    IS the time guard (measured directly off the transcript mtime), so the
    >= threshold consecutive-checks accumulation on top of it is the whole
    transient-glitch margin.
    """
    if mapped or has_tty:
        return ("clear", 0)
    if idle_age_s is None:
        return ("skip", missed)
    if idle_age_s < idle_reap_s:
        return ("clear", 0)
    new_missed = missed + 1
    if new_missed < threshold:
        return ("keep", new_missed)
    if reap_enabled:
        return ("stop", 0)
    if not alerted:
        return ("alert", new_missed)
    return ("keep", new_missed)


def _idle_unmapped_state_path(sid: str) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{IDLE_UNMAPPED_STATE_PREFIX}{sid}.json"


def _load_idle_unmapped_state(sid: str) -> dict:
    """Per-session idle-unmapped state (``{}`` if absent/garbled — a fresh or
    unreadable file starts the miss count at 0, mirroring the other watcher
    state loaders)."""
    path = _idle_unmapped_state_path(sid)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_idle_unmapped_state(
    sid: str,
    *,
    missed: int,
    alerted: bool,
    pid: int,
    idle_age_s: float | None,
    first_over_ts: float,
    stopped_at: float | None = None,
    stop_retried: bool = False,
    stop_failed_alerted: bool = False,
) -> None:
    """Persist the per-session idle-unmapped state atomically (temp +
    rename). ``first_over_ts`` anchors the GC-age log line; ``pid`` /
    ``idle_age_s`` are informational (the decision keys on the live daemon
    snapshot + a fresh transcript stat each tick). The stop-verification
    fields mirror the zombie-wrapper contract (ACK != kill)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _idle_unmapped_state_path(sid)
    payload = {
        "missed": missed,
        "alerted": alerted,
        "pid": pid,
        "idle_age_s": idle_age_s,
        "first_over_ts": first_over_ts,
        "stopped_at": stopped_at,
        "stop_retried": bool(stop_retried),
        "stop_failed_alerted": bool(stop_failed_alerted),
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_idle_unmapped_state(sid: str) -> None:
    """Drop the per-session state (episode over: activity resumed, the
    session left scope, or it was verified stopped)."""
    _idle_unmapped_state_path(sid).unlink(missing_ok=True)


def _gc_orphan_idle_unmapped_state(
    live_sids: set[str], dry_run: bool, now: float | None = None
) -> None:
    """GC idle-unmapped state for sessions no longer in the daemon's live set
    (stopped by any path — the episode is over; this reap is also the
    stop-verification success path). Same contract as
    :func:`_gc_orphan_zombie_state`."""
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return
    now = now if now is not None else time.time()
    for path in sorted(AUTONOMOUS_REGISTRY_DIR.glob(f"{IDLE_UNMAPPED_STATE_PREFIX}*.json")):
        sid = path.stem[len(IDLE_UNMAPPED_STATE_PREFIX) :]
        if sid in live_sids:
            continue
        try:
            payload = json.loads(path.read_text())
            first_over = payload.get("first_over_ts", now)
            if not isinstance(first_over, int | float):
                first_over = now
        except (json.JSONDecodeError, OSError):
            first_over = 0  # unreadable -> definitely orphaned, drop it
        age = now - first_over
        reason = (
            "session left the live set"
            if age < POD_SAFETY_STATE_MAX_AGE_S
            else f"age={age / 3600:.1f}h"
        )
        print(f"  idle-unmapped: GC orphan state {sid} ({reason})")
        if not dry_run:
            path.unlink(missing_ok=True)


def _append_idle_unmapped_event(note: str, dry_run: bool) -> None:
    """Durable trace for idle-unmapped actions — these sessions have NO issue
    mapping by definition, so there is never a task to carry a marker. One
    JSON line per action in ``~/.eps-autonomous/idle-unmapped-events.jsonl``
    (same role as the zombie-wrapper fallback file). Fail-soft."""
    dest = AUTONOMOUS_REGISTRY_DIR / "idle-unmapped-events.jsonl"
    line = json.dumps(
        {"ts": datetime.now().astimezone().isoformat(), "kind": "idle-unmapped", "note": note}
    )
    if dry_run:
        print(f"  [dry-run] would append idle-unmapped event to {dest}")
        return
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as e:
        print(f"  WARNING: appending idle-unmapped event failed: {e}", file=sys.stderr)


def _check_idle_unmapped_stop_verification(
    sid: str,
    pid: int,
    in_scope: bool,
    prev: dict,
    dry_run: bool,
    now: float,
) -> bool:
    """Next-tick verification that an ACKed idle-unmapped stop landed
    (ACK != kill). Returns True when this tick was consumed by the
    verification path. Same ladder as
    :func:`_check_zombie_stop_verification`: one stop retry, then one loud
    record, then quiet; the verified-gone path is the live-session-keyed GC."""
    stopped_at = prev.get("stopped_at")
    if not isinstance(stopped_at, int | float) or not stopped_at:
        return False
    if not in_scope:
        return False
    first_over_ts = prev.get("first_over_ts")
    if not isinstance(first_over_ts, int | float):
        first_over_ts = now
    print(
        f"  IDLE-UNMAPPED STOP-VERIFY FAILED session {sid}: still alive one "
        f"tick after the daemon ACKed its stop (ACK != kill).",
        file=sys.stderr,
    )
    idle_age_s = prev.get("idle_age_s")
    if not isinstance(idle_age_s, int | float):
        idle_age_s = None
    common = dict(
        missed=prev.get("missed", 0) if isinstance(prev.get("missed", 0), int) else 0,
        alerted=bool(prev.get("alerted", False)),
        pid=pid,
        idle_age_s=idle_age_s,
        first_over_ts=first_over_ts,
    )
    if not prev.get("stop_retried"):
        acked = _stop_session(sid, dry_run)
        print(f"  idle-unmapped: stop RETRIED for {sid} (one retry per episode, acked={acked})")
        if not dry_run:
            _save_idle_unmapped_state(
                sid,
                **common,
                stopped_at=now if acked else stopped_at,
                stop_retried=True,
                stop_failed_alerted=bool(prev.get("stop_failed_alerted", False)),
            )
        return True
    if not prev.get("stop_failed_alerted"):
        _append_idle_unmapped_event(
            f"{_IDLE_UNMAPPED_STOP_FAILED_NOTE_SENTINEL} idle-unmapped-session STOP "
            f"FAILED to land: session {sid} (wrapper pid {pid}) is still alive after "
            f"the idle-unmapped pass stopped it AND retried once — the Happy daemon "
            f"ACKed the stop RPCs but did not kill the wrapper. Stop manually with "
            f"`spawn_session.py stop --session-id {sid}` (or restart the Happy "
            f"daemon). Recorded once per episode.",
            dry_run,
        )
        if not dry_run:
            _save_idle_unmapped_state(
                sid,
                **common,
                stopped_at=stopped_at,
                stop_retried=True,
                stop_failed_alerted=True,
            )
        return True
    print(
        f"  idle-unmapped: session {sid} already retried + alerted this episode; "
        f"awaiting manual stop / daemon recovery."
    )
    return True


def _process_idle_unmapped(
    sid: str,
    pid: int,
    issue: int | None,
    now: float,
    dry_run: bool,
    threshold: int,
    *,
    reap_enabled: bool,
) -> None:
    """Apply the idle-unmapped decision to one live, non-PM, EPS-cwd session:
    check the wrapper's controlling TTY, stat the resolved transcript, and
    act per :func:`decide_idle_unmapped`."""
    mapped = issue is not None
    has_tty = _wrapper_has_controlling_tty(pid) if not mapped else False
    idle_age_s: float | None = None
    signal_reason: str | None = None
    if not mapped and not has_tty:
        idle_age_s, signal_reason = _transcript_idle_age_s(pid, now)

    prev = _load_idle_unmapped_state(sid)
    prev_missed = prev.get("missed", 0)
    if not isinstance(prev_missed, int):
        prev_missed = 0
    prev_alerted = bool(prev.get("alerted", False))
    first_over_ts = prev.get("first_over_ts")
    if not isinstance(first_over_ts, int | float):
        first_over_ts = now

    idle_reap_s = _unmapped_idle_reap_s()
    in_scope = not mapped and not has_tty and idle_age_s is not None and idle_age_s >= idle_reap_s
    if _check_idle_unmapped_stop_verification(sid, pid, in_scope, prev, dry_run, now):
        return

    action, new_missed = decide_idle_unmapped(
        mapped,
        has_tty,
        idle_age_s,
        prev_missed,
        prev_alerted,
        threshold,
        reap_enabled=reap_enabled,
        idle_reap_s=idle_reap_s,
    )
    idle_label = f"{idle_age_s / 3600:.1f}h" if idle_age_s is not None else "?"
    print(
        f"  session {sid} (pid={pid}): mapped={mapped} tty={has_tty} "
        f"idle={idle_label} missed={prev_missed}->{new_missed} action={action}"
    )

    if action == "clear":
        if prev and not dry_run:
            _clear_idle_unmapped_state(sid)
        return

    if action == "skip":
        # Idleness signal unavailable: fail toward keep, loudly, and leave
        # the episode state frozen (neither accumulated nor erased).
        print(
            f"  idle-unmapped: session {sid} idleness signal unavailable "
            f"({signal_reason}); failing toward KEEP",
            file=sys.stderr,
        )
        return

    if action == "stop":
        acked = _stop_session(sid, dry_run)
        if acked:
            _append_idle_unmapped_event(
                f"{_IDLE_UNMAPPED_STOP_NOTE_SENTINEL} auto-stopped idle unmapped "
                f"Happy session {sid} (wrapper pid {pid}): its resolved Claude "
                f"transcript has been idle {idle_label} (>= "
                f"{idle_reap_s / 3600:.1f}h window, >= {threshold} consecutive "
                f"checks), it has no issue mapping, no controlling TTY, and is "
                f"not the PM session. The 2026-06-12 class: 25 such sessions "
                f"idle 19-43h held ~23 GB RSS. Respawn if needed: "
                f"`spawn_session.py spawn-issue --issue <N>` (or `spawn-pm`). "
                f"Set EPM_UNMAPPED_IDLE_REAP=0 on the watcher cron to fall back "
                f"to alert-only.",
                dry_run,
            )
        if not dry_run:
            _save_idle_unmapped_state(
                sid,
                missed=0 if acked else prev_missed,
                alerted=prev_alerted,
                pid=pid,
                idle_age_s=idle_age_s,
                first_over_ts=first_over_ts,
                stopped_at=now if acked else None,
                stop_retried=bool(prev.get("stop_retried", False)),
                stop_failed_alerted=bool(prev.get("stop_failed_alerted", False)),
            )
        return

    if action == "alert":
        print(
            f"  IDLE-UNMAPPED ALERT session {sid}: transcript idle {idle_label}; "
            f"NOT stopping (EPM_UNMAPPED_IDLE_REAP=0 — alert-only fallback).",
            file=sys.stderr,
        )
        _append_idle_unmapped_event(
            f"{_IDLE_UNMAPPED_ALERT_NOTE_SENTINEL} IDLE unmapped Happy session: "
            f"{sid} (wrapper pid {pid}) has an idle Claude transcript "
            f"({idle_label} >= {idle_reap_s / 3600:.1f}h) and no issue mapping. "
            f"NOT auto-stopped (EPM_UNMAPPED_IDLE_REAP=0 alert-only fallback); "
            f"stop manually with `spawn_session.py stop --session-id {sid}`, or "
            f"unset the env var on the watcher cron to restore the default reap. "
            f"Recorded once per episode.",
            dry_run,
        )
        if not dry_run:
            _save_idle_unmapped_state(
                sid,
                missed=new_missed,
                alerted=True,
                pid=pid,
                idle_age_s=idle_age_s,
                first_over_ts=first_over_ts,
            )
        return

    # action == "keep": persist the incremented miss count + episode anchor.
    if not dry_run:
        _save_idle_unmapped_state(
            sid,
            missed=new_missed,
            alerted=prev_alerted,
            pid=pid,
            idle_age_s=idle_age_s,
            first_over_ts=first_over_ts,
        )


def idle_unmapped_pass(
    dry_run: bool,
    threshold: int,
    *,
    daemon_reachable: bool,
    children: list[dict] | None = None,
    now: float | None = None,
) -> None:
    """Auto-stop unmapped EPS-cwd Happy sessions whose Claude transcript has
    been idle >= the reap window (default 12h) on >= ``threshold``
    consecutive checks — the live-but-idle complement of the zombie-wrapper
    pass (which needs a DEAD inner Claude) and the unmapped complement of the
    session-reconcile pass (which needs an issue mapping). Exclusions:
    PM-registered sids, non-EPS cwds, issue-mapped sessions (registry entry
    or issue-<N> worktree cwd), wrappers holding a controlling TTY, and any
    session whose idleness signal cannot be resolved (fail toward keep).

    Daemon-gated like the zombie pass: the wrapper pids come from the
    daemon's ``/list`` and the stop action POSTs to it. ``children`` may be
    injected (tests / a caller reusing its snapshot); ``None`` fetches via
    :func:`_live_children`."""
    now = now if now is not None else time.time()
    if not daemon_reachable:
        print(
            "idle-unmapped: Happy daemon unreachable; skipping "
            "(wrapper pids + the stop RPC both need the daemon)"
        )
        return
    children = children if children is not None else _live_children()
    live_sids = {
        c.get("happySessionId") for c in children if isinstance(c.get("happySessionId"), str)
    }
    # GC ALWAYS on a daemon-reachable tick — even with zero candidates — so
    # episodes whose session died/was stopped by any path start fresh later.
    _gc_orphan_idle_unmapped_state(live_sids, dry_run, now=now)
    if not children:
        print("idle-unmapped: no live daemon-tracked sessions")
        return

    registry_map = _load_session_issue_map()
    meta = _load_session_meta()
    pm_sids = _load_pm_session_ids()
    project_prefix = str(PROJECT_ROOT)
    candidates: list[tuple[str, int, int | None]] = []
    skipped_pm = 0
    skipped_non_eps = 0
    for child in children:
        sid = child.get("happySessionId")
        pid = child.get("pid")
        if not isinstance(sid, str) or not sid or not isinstance(pid, int):
            continue
        if sid in pm_sids:
            skipped_pm += 1
            continue
        path = (meta.get(sid) or {}).get("path")
        if not isinstance(path, str) or not (
            path == project_prefix or path.startswith(project_prefix + "/")
        ):
            # Non-EPS cwd (other projects) or no cwd metadata at all: never
            # touched — EPS-ness cannot be established, so err toward keep.
            skipped_non_eps += 1
            continue
        issue = registry_map.get(sid)
        if issue is None:
            issue = _infer_issue_from_path(path)
        candidates.append((sid, pid, issue))

    reap = _unmapped_idle_reap_enabled()
    print(
        f"idle-unmapped: {len(candidates)} EPS session(s) scanned "
        f"({skipped_pm} PM-registered + {skipped_non_eps} non-EPS skipped; "
        f"reap={'ON' if reap else 'OFF — alert-only (EPM_UNMAPPED_IDLE_REAP=0)'})"
    )
    for sid, pid, issue in sorted(candidates):
        _process_idle_unmapped(
            sid,
            pid,
            issue,
            now,
            dry_run,
            threshold,
            reap_enabled=reap,
        )


def pod_safety_pass(dry_run: bool, threshold: int, now: float | None = None) -> None:
    """Reconcile RUNNING managed pods against their task STATUS.

    - AUTO-STOP (reversible, never terminate) a RUNNING pod whose task is DONE
      (:data:`AUTO_STOP_DONE`), after the 2-miss guard — an escaped pod.
    - ALERT (loud log + one-time marker, no stop) a RUNNING pod-active pod with
      no real progress for > :data:`ALERT_STALE_HOURS` — a likely-abandoned
      mid-run session.
    - KEEP everything else.

    Trigger is task STATUS, never session-cwd liveness (which misreports live
    interactive sessions as dead). Does NOT depend on the Happy daemon, so it
    runs unconditionally — even during a daemon outage. STOP is reversible —
    never a terminate."""
    now = now if now is not None else time.time()
    running = _running_managed_issue_pods()
    if running is None:
        # Snapshot FAILED (transport error — the helper already logged it).
        # Do NOT GC on a failed snapshot: an empty-because-failed set is
        # indistinguishable from "every pod left RUNNING", so the GC would
        # wipe ALL pod-safety state — not just the fail-safe 2-miss counters
        # but the `alerted` / `keep_running_noted` / `followup_noted`
        # once-per-episode dedup flags, re-arming duplicate markers on every
        # API hiccup. Genuinely stranded files are reaped on the next GOOD
        # snapshot (plus the age backstop inside
        # `_gc_orphan_pod_safety_state`). No pods are processed either —
        # same fail-closed no-stop outcome as today's empty-set fallback.
        print("pod-safety: pod snapshot failed; skipping state GC this tick")
        return
    running_issues = {issue for issue, _pod_id, _name in running}

    # GC orphaned state BEFORE the per-pod loop, and ALWAYS on a GOOD snapshot
    # — even when `running` is empty — so a state file for a pod that left the
    # RUNNING set by ANY path (manual stop/terminate, self-EXIT on TTL/crash)
    # gets cleared. Otherwise a re-used `pod-N` would inherit a stale
    # `missed=1` / `alerted` and be one glitch away from a stop on revival.
    _gc_orphan_pod_safety_state(running_issues, dry_run, now=now)

    if not running:
        print("pod-safety: no RUNNING managed pods")
        return
    print(f"pod-safety: {len(running)} RUNNING managed pod(s)")
    for issue, pod_id, _name in running:
        _process_pod(issue, pod_id, now, dry_run, threshold)


def _vm_run_remediations(
    *,
    do_audit: bool,
    do_reclaim: bool,
    last_reclaim_ts: float | None,
    last_audit_ts: float | None,
    now: float,
    dry_run: bool,
) -> tuple[list[str], float | None, float | None]:
    """Execute the armed vm-disk remediations (worktree audit at low+, cache
    reclaims at critical). Each reclaim step lands its own summary line in
    the marker note, annotated with the free-space delta it bought (when
    above :data:`VM_DISK_FREED_NOTE_MIN_BYTES` — smaller deltas are statvfs
    noise from concurrent writers). Returns ``(summary lines for the marker
    note, new last_reclaim_ts, new last_audit_ts)``. All actions are
    fail-soft."""
    remediation: list[str] = []
    new_last_audit_ts = last_audit_ts
    if do_audit:
        print("  vm-disk: running stale-worktree sweep (worktree_audit.py --apply)")
        remediation.append(_vm_remediate_worktrees(dry_run))
        new_last_audit_ts = now

    new_last_reclaim_ts = last_reclaim_ts
    if do_reclaim:
        print(
            "  vm-disk: running safe cache reclaims "
            "(wandb artifact cache + uv cache prune + npm cache clean "
            "+ HF hub TTL eviction + stale /tmp/claude-* sweep)"
        )
        for step in (
            lambda: _vm_reclaim_wandb_cache(dry_run),
            lambda: _vm_reclaim_uv_cache(dry_run),
            lambda: _vm_reclaim_npm_cache(dry_run),
            lambda: _vm_reclaim_hf_hub_cache(now, dry_run),
        ):
            before = _vm_free_bytes()
            summary = step()
            after = _vm_free_bytes()
            if (
                isinstance(summary, str)
                and before is not None
                and after is not None
                and after - before > VM_DISK_FREED_NOTE_MIN_BYTES
            ):
                summary = f"{summary} (+{(after - before) / 2**30:.1f} GiB)"
            if summary:
                remediation.append(summary)
        swept = _sweep_stale_claude_tmp(now, dry_run)
        remediation.append(f"swept {swept} stale /tmp/claude-* tree(s)")
        new_last_reclaim_ts = now

    if remediation:
        refreshed = _vm_free_bytes()
        if refreshed is not None:
            remediation.append(f"post-remediation free {refreshed / 2**30:.1f} GiB")
            print(f"  vm-disk: post-remediation free {refreshed / 2**30:.1f} GiB")
    return remediation, new_last_reclaim_ts, new_last_audit_ts


def vm_disk_pass(dry_run: bool, now: float | None = None) -> None:
    """Watch VM root-disk headroom; alert once per low-disk episode, run the
    stale-worktree sweep whenever low (the big-space remediation), and the
    safe cache reclaims when critically low.

    Pods have their own guards (``pod_disk_guard.py``, the preflight
    fallocate probe); the VM had none until / hit 100% mid-pipeline and every
    foreground Bash spawn in the orchestrator session failed silently
    (task #552, 2026-06-10). Everything here is fail-soft — a disk alert must
    never crash the watcher pass that delivers it."""
    now = now if now is not None else time.time()
    free = _vm_free_bytes()
    if free is None:
        return
    state = _load_vm_disk_state()
    last_reclaim_ts = state.get("last_reclaim_ts")
    if not isinstance(last_reclaim_ts, int | float):
        last_reclaim_ts = None
    last_audit_ts = state.get("last_audit_ts")
    if not isinstance(last_audit_ts, int | float):
        last_audit_ts = None
    level, do_alert, do_reclaim, do_audit = decide_vm_disk(
        free,
        alerted=bool(state.get("alerted", False)),
        last_reclaim_ts=last_reclaim_ts,
        last_audit_ts=last_audit_ts,
        now=now,
    )
    free_gib = free / 2**30

    if level == "ok":
        if not state:
            print(f"vm-disk: ok ({free_gib:.1f} GiB free)")
        elif free >= VM_DISK_ALERT_FREE_BYTES + VM_DISK_CLEAR_HYSTERESIS_BYTES:
            print(f"vm-disk: recovered ({free_gib:.1f} GiB free); episode over")
            if not dry_run:
                _clear_vm_disk_state()
        else:
            # Inside the hysteresis band (alert <= free < alert + margin):
            # keep the episode state so a fresh dip neither re-alerts nor
            # re-fires the worktree audit inside the re-arm window (free
            # space flapping around the alert boundary is ONE episode).
            clear_gib = (VM_DISK_ALERT_FREE_BYTES + VM_DISK_CLEAR_HYSTERESIS_BYTES) / 2**30
            print(
                f"vm-disk: recovering ({free_gib:.1f} GiB free); keeping episode "
                f"state until >= {clear_gib:.0f} GiB"
            )
        return

    # Loud log EVERY tick while low — the cron log is the primary channel.
    print(
        f"vm-disk: {level.upper()} — {free_gib:.1f} GiB free on {VM_DISK_PATH} "
        f"(alert < {VM_DISK_ALERT_FREE_BYTES / 2**30:.0f} GiB, "
        f"reclaim < {VM_DISK_RECLAIM_FREE_BYTES / 2**30:.0f} GiB)",
        file=sys.stderr,
    )

    # Remediate BEFORE posting the alert so the once-per-episode marker carries
    # what was done, not just that disk was low (detection runs every 10-min
    # tick; the once-daily worktree cron alone lost the 2026-06-11 race).
    remediation, new_last_reclaim_ts, new_last_audit_ts = _vm_run_remediations(
        do_audit=do_audit,
        do_reclaim=do_reclaim,
        last_reclaim_ts=last_reclaim_ts,
        last_audit_ts=last_audit_ts,
        now=now,
        dry_run=dry_run,
    )

    if do_alert:
        note = (
            f"{_VM_DISK_NOTE_SENTINEL} VM root disk {level.upper()}: "
            f"{free_gib:.1f} GiB free on {VM_DISK_PATH}. Near full, foreground "
            f"Bash spawns in VM sessions start failing silently (exit 1, zero "
            f"output — task #552 incident, 2026-06-10). Auto-remediation: "
            f"stale-worktree sweep at LOW; at CRITICAL also the wandb "
            f"artifact / uv / npm caches, HF hub revisions idle > TTL, and "
            f"stale /tmp/claude-* trees (executed steps listed below); "
            f"anything beyond that (held worktrees, recently-used HF repos) "
            f"needs a human. Posted once per low-disk episode."
        )
        if remediation:
            note += f" [auto-remediation: {'; '.join(remediation)}]"
        issues = _vm_disk_marker_issues()
        if issues:
            for issue in issues:
                _post_progress_marker(issue, note, dry_run, label="vm-disk-low")
        else:
            _append_vm_disk_fallback_event(note, dry_run)

    if not dry_run and (do_alert or do_reclaim or do_audit):
        _save_vm_disk_state(
            alerted=bool(state.get("alerted", False)) or do_alert,
            last_reclaim_ts=new_last_reclaim_ts,
            last_audit_ts=new_last_audit_ts,
            prev=state,
        )


# ─── campaign pass (question-level /campaign sessions; task #586) ────────────
#
# Driven by ``campaign-<N>.json`` registry entries written by
# ``spawn_session.py spawn-campaign`` / ``register-current --mode campaign``.
# Four jobs, mirroring the issue respawn + stalled passes with campaign
# semantics:
#
# 1. **Respawn**: campaign task ACTIVE (approved/running) + session dead on
#    >= threshold consecutive checks -> ``spawn-campaign --issue <N>`` (which
#    rewrites the registry entry with the fresh id; caps re-passed from the
#    entry).
# 2. **Progress watchdog** (progress, not liveness): session ALIVE but the
#    newest ``epm:campaign-*`` marker is older than ``EPM_CAMPAIGN_STALL_S``
#    (default 2h) AND no child task posted any marker in that window ->
#    one ``epm:campaign-stalled v1`` alert per episode; a SECOND consecutive
#    stalled check stop-then-respawns (cap CAMPAIGN_MAX_RESPAWNS per
#    episode, then a one-time exhausted alert — mirrors the Phase-2
#    stalled-session actor).
# 3. **Budget backstop**: ``gpu_hours_committed > gpu_hours_total`` in
#    ``artifacts/campaign-state.json`` -> one loud alert marker per episode.
#    The /campaign skill should never let this happen; the watcher is the
#    harness-side circuit breaker. GPU-hours, never dollars.
# 4. **GC**: reap the registry entry + watch state when the campaign task is
#    terminal (completed/archived/blocked). Stop-then-reap: a still-live
#    session is stopped BEFORE the entry is removed (the entry is the
#    session's issue mapping — removing it first would orphan an immortal
#    idle session past every later pass), and the reap is deferred while
#    the daemon is unreachable.
#
# Interactions with the other passes (all verified, not assumed):
# - The issue respawn pass globs ``issue-*.json`` — ``campaign-<N>.json``
#   never matches, so a campaign is never respawned via ``spawn-issue``.
# - The orphan sweep skips ``kind: campaign`` tasks (see
#   :func:`_active_status_tasks`) — its recovery command is
#   ``spawn-issue --auto``, which would boot the WRONG skill on a campaign.
# - The session-reconcile pass maps the campaign session to its issue (the
#   ``campaign-`` prefix is in ``_load_session_issue_map``) but acts only on
#   :data:`SESSION_RECONCILE_DONE` statuses — a campaign at ``running``
#   returns "clear", so an idle-between-ticks campaign session is never
#   auto-stopped mid-campaign; once the campaign completes, the normal
#   idle-grace stop applies (desired).

CAMPAIGN_REGISTRY_PREFIX = "campaign-"
# Watch-state files live at campaign-watch-<N>.json. They match the
# ``campaign-*.json`` glob too, but their stem ("watch-<N>") fails the int
# parse so every registry-entry walk skips them; they deliberately carry NO
# integer ``issue`` key so spawn_session's issue-map loader skips them too.
CAMPAIGN_WATCH_STATE_PREFIX = "campaign-watch-"
# A campaign session is mid-work at `approved` (spawned, Step 0 not yet
# flipped it) and `running` (the held status for the whole campaign).
CAMPAIGN_ACTIVE = {"approved", "running"}
CAMPAIGN_TERMINAL = {"completed", "archived", "blocked"}
CAMPAIGN_STALL_S_DEFAULT = 2 * 3600
CAMPAIGN_MAX_RESPAWNS = 3


def _campaign_stall_s() -> float:
    """Campaign progress-watchdog window: ``EPM_CAMPAIGN_STALL_S`` when set to
    a positive number, else :data:`CAMPAIGN_STALL_S_DEFAULT` (2h)."""
    raw = os.environ.get("EPM_CAMPAIGN_STALL_S", "")
    try:
        val = float(raw)
    except ValueError:
        return CAMPAIGN_STALL_S_DEFAULT
    return val if val > 0 else CAMPAIGN_STALL_S_DEFAULT


def _campaign_registry_entries() -> list[tuple[Path, dict]]:
    """``(path, entry)`` for every readable ``campaign-<N>.json`` registry
    entry (integer N). Watch-state files (``campaign-watch-<N>.json``) and
    garbled names are skipped; an unreadable entry is returned with an empty
    dict so the caller can remove it."""
    out: list[tuple[Path, dict]] = []
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return out
    for path in sorted(AUTONOMOUS_REGISTRY_DIR.glob(f"{CAMPAIGN_REGISTRY_PREFIX}*.json")):
        stem = path.stem[len(CAMPAIGN_REGISTRY_PREFIX) :]
        try:
            int(stem)
        except ValueError:
            continue  # campaign-watch-<N>.json or a hand-debug artifact
        try:
            entry = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            entry = {}
        out.append((path, entry if isinstance(entry, dict) else {}))
    return out


def _campaign_watch_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{CAMPAIGN_WATCH_STATE_PREFIX}{issue}.json"


def _load_campaign_watch_state(issue: int) -> dict:
    """Per-campaign watchdog state (``{}`` if absent/unreadable — a fresh file
    starts every counter at 0, mirroring :func:`_load_stalled_state`)."""
    path = _campaign_watch_state_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_campaign_watch_state(
    issue: int,
    *,
    stalled_checks: int,
    alerted: bool,
    respawn_count: int,
    exhausted: bool,
    budget_alerted: bool,
    prev: dict | None = None,
) -> None:
    """Persist the campaign watchdog state atomically (temp + rename).

    NOTE: deliberately NO ``issue`` / ``happy_session_id`` keys — the file
    matches spawn_session's ``campaign-*.json`` issue-map glob, and those
    keys would make a watch-state file masquerade as a registry entry."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _campaign_watch_state_path(issue)
    prev_first_seen = (prev or {}).get("first_seen")
    if not isinstance(prev_first_seen, int | float):
        prev_first_seen = time.time()
    payload = {
        "stalled_checks": stalled_checks,
        "alerted": alerted,
        "respawn_count": respawn_count,
        "exhausted": exhausted,
        "budget_alerted": budget_alerted,
        "first_seen": prev_first_seen,
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_campaign_watch_state(issue: int) -> None:
    _campaign_watch_state_path(issue).unlink(missing_ok=True)


def _respawn_campaign(entry: dict, dry_run: bool) -> bool:
    """Re-spawn the campaign session for this registry entry via
    ``spawn_session.py spawn-campaign`` (re-passing the entry's caps).
    Returns True on success; spawn-campaign rewrites the registry entry
    (fresh id, missed=0) as a side effect."""
    issue = entry["issue"]
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-campaign",
        "--issue", str(issue),
        "--budget-gpu-hours", str(entry.get("budget_gpu_hours", 250.0)),
        "--max-concurrent", str(entry.get("max_concurrent", 4)),
        "--per-child-cap", str(entry.get("per_child_gpu_hours_cap", 100.0)),
    ]  # fmt: skip
    if dry_run:
        print(f"  [dry-run] would respawn campaign: {' '.join(cmd)}")
        return False
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(
            f"  CAMPAIGN RESPAWN FAILED issue #{issue}: {res.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  RESPAWNED campaign #{issue}: {first_line}")
    return True


def _campaign_children(issue: int) -> list[dict]:
    """Children of campaign ``issue`` via ``task.py list-children --json``;
    ``[]`` on any read failure (same subprocess isolation as
    :func:`_task_status`)."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "list-children", str(issue), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _campaign_child_marker_fresh(issue: int, window_s: float, now: float) -> bool:
    """True iff ANY child of campaign ``issue`` posted ANY ``epm:`` marker
    within the last ``window_s`` seconds. Called LAZILY — only when the
    campaign's own markers are already stale — so the per-child events
    fetch is paid only on watchdog-candidate ticks."""
    for child in _campaign_children(issue):
        child_id = child.get("id")
        if not isinstance(child_id, int):
            continue
        events = _task_events(child_id)
        latest = _latest_progress_ts(events)
        if latest is not None and (now - latest) <= window_s:
            return True
    return False


def _post_campaign_marker(issue: int, kind: str, note: str, dry_run: bool) -> None:
    """Post a campaign-pass marker (kind must be declared in workflow.yaml §
    markers — ``epm:campaign-stalled`` — or the generic ``epm:progress`` for
    the budget backstop). The note carries :data:`_CAMPAIGN_NOTE_SENTINEL`
    so watcher-posted events never reset the staleness clocks they measure.
    Same fail-soft posture as :func:`_post_progress_marker`."""
    if dry_run:
        print(f"  [dry-run] would post {kind} on #{issue}: {note}")
        return
    try:
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "post-marker",
                str(issue),
                kind,
                "--note",
                note,
                "--by",
                "autonomous_session_watch",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  WARNING: posting {kind} on #{issue} failed: {e}", file=sys.stderr)


def _campaign_state_budget(issue: int) -> tuple[float, float] | None:
    """``(gpu_hours_committed, gpu_hours_total)`` from the campaign's
    ``artifacts/campaign-state.json``, or None when the state file is absent
    / unreadable (a campaign that hasn't run Step 0 yet has no state — not
    an error). The task folder is resolved via ``task.py find`` (never a
    hand-built ``tasks/...`` path)."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "find", str(issue)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    state_file = Path(out.stdout.strip()) / "artifacts" / "campaign-state.json"
    try:
        state = json.loads(state_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    budget = state.get("budget") if isinstance(state, dict) else None
    if not isinstance(budget, dict):
        return None
    committed = budget.get("gpu_hours_committed")
    total = budget.get("gpu_hours_total")
    if not isinstance(committed, int | float) or not isinstance(total, int | float):
        return None
    return float(committed), float(total)


def _campaign_is_stale(issue: int, entry: dict, now: float) -> bool:
    """True iff the newest SKILL-POSTED ``epm:campaign-*`` marker (baseline:
    ``spawned_at`` when none exists yet) is older than
    :func:`_campaign_stall_s` AND no child task posted a marker in that
    window. Watcher-posted notes (the ``epm:campaign-stalled`` alert itself)
    are excluded via the note sentinel — otherwise the alert would reset the
    staleness baseline it measures and the episode could never escalate
    past the first check."""
    stall_s = _campaign_stall_s()
    campaign_ts: float | None = None
    for ev in _task_events(issue):
        if not str(ev.get("kind", "")).startswith("epm:campaign"):
            continue
        if _CAMPAIGN_NOTE_SENTINEL in (ev.get("note") or ""):
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and (campaign_ts is None or ts > campaign_ts):
            campaign_ts = ts
    baseline = campaign_ts if campaign_ts is not None else entry.get("spawned_at", now)
    if not isinstance(baseline, int | float):
        baseline = now
    if (now - float(baseline)) <= stall_s:
        return False
    return not _campaign_child_marker_fresh(issue, stall_s, now)


def _campaign_escalate_stall(
    issue: int, entry: dict, st: dict, dry_run: bool, *, daemon_reachable: bool
) -> None:
    """Handle one STALLED check: bump the counter, alert on the first check,
    stop-then-respawn on the second consecutive one (daemon required; capped
    at :data:`CAMPAIGN_MAX_RESPAWNS` per episode, then a one-time exhausted
    alert). Mutates the counter dict ``st`` in place."""
    stall_s = _campaign_stall_s()
    st["stalled_checks"] += 1
    print(
        f"  campaign #{issue}: STALLED check {st['stalled_checks']} "
        f"(no epm:campaign-* or child marker in {stall_s / 3600:.1f}h)"
    )
    if st["stalled_checks"] == 1 and not st["alerted"]:
        _post_campaign_marker(
            issue,
            "epm:campaign-stalled",
            f"{_CAMPAIGN_NOTE_SENTINEL} no epm:campaign-* marker or child-task "
            f"marker for > {stall_s / 3600:.1f}h with a live campaign session; "
            f"second consecutive stalled check stop-then-respawns.",
            dry_run,
        )
        st["alerted"] = True
        return
    if st["stalled_checks"] < 2:
        return
    if st["respawn_count"] >= CAMPAIGN_MAX_RESPAWNS:
        if not st["exhausted"]:
            _post_campaign_marker(
                issue,
                "epm:progress",
                f"{_CAMPAIGN_NOTE_SENTINEL} campaign auto-recovery exhausted "
                f"({st['respawn_count']} respawns this episode); awaiting user.",
                dry_run,
            )
            st["exhausted"] = True
        return
    if not daemon_reachable:
        print(f"  campaign #{issue}: stalled but daemon unreachable; alert-only")
        return
    sid = entry.get("happy_session_id")
    stopped = _stop_session(sid, dry_run) if isinstance(sid, str) else True
    if stopped and _respawn_campaign(entry, dry_run):
        _post_campaign_marker(
            issue,
            "epm:progress",
            f"{_CAMPAIGN_NOTE_SENTINEL} stalled campaign session "
            f"stop-then-respawned (respawn {st['respawn_count'] + 1}/"
            f"{CAMPAIGN_MAX_RESPAWNS} this episode).",
            dry_run,
        )
        st["respawn_count"] += 1
        st["stalled_checks"] = 0


def _campaign_budget_backstop(issue: int, budget_alerted: bool, dry_run: bool) -> bool:
    """One loud alert per episode when ``campaign-state.json`` shows
    GPU-hours committed > total. Returns the updated ``budget_alerted``
    flag (re-armed once committed drops back under total)."""
    budget = _campaign_state_budget(issue)
    if budget is None:
        return budget_alerted
    committed, total = budget
    if committed > total and not budget_alerted:
        _post_campaign_marker(
            issue,
            "epm:progress",
            f"{_CAMPAIGN_NOTE_SENTINEL} BUDGET BACKSTOP: campaign-state.json has "
            f"gpu_hours_committed={committed:g} > gpu_hours_total={total:g}. The "
            f"/campaign skill must stop filing children; harness circuit breaker.",
            dry_run,
        )
        return True
    if committed <= total:
        return False
    return budget_alerted


def _campaign_watchdog(
    issue: int, entry: dict, now: float, dry_run: bool, *, daemon_reachable: bool
) -> None:
    """Progress + budget watchdog for one ALIVE, ACTIVE campaign session.

    Stall detection per :func:`_campaign_is_stale`; escalation per
    :func:`_campaign_escalate_stall` (one ``epm:campaign-stalled v1`` alert,
    then bounded stop-then-respawn). Fresh progress ends the episode and
    resets every counter. The budget backstop posts one alert per episode
    when committed > total (:func:`_campaign_budget_backstop`)."""
    state = _load_campaign_watch_state(issue)
    st = {
        "stalled_checks": int(state.get("stalled_checks", 0) or 0),
        "alerted": bool(state.get("alerted", False)),
        "respawn_count": int(state.get("respawn_count", 0) or 0),
        "exhausted": bool(state.get("exhausted", False)),
    }
    if _campaign_is_stale(issue, entry, now):
        _campaign_escalate_stall(issue, entry, st, dry_run, daemon_reachable=daemon_reachable)
    else:
        st = {"stalled_checks": 0, "alerted": False, "respawn_count": 0, "exhausted": False}

    budget_alerted = _campaign_budget_backstop(
        issue, bool(state.get("budget_alerted", False)), dry_run
    )

    if not dry_run:
        _save_campaign_watch_state(
            issue,
            stalled_checks=st["stalled_checks"],
            alerted=st["alerted"],
            respawn_count=st["respawn_count"],
            exhausted=st["exhausted"],
            budget_alerted=budget_alerted,
            prev=state,
        )


def _campaign_reap(path: Path, issue: int | None, reason: str, dry_run: bool) -> None:
    """Remove a campaign registry entry (+ its watch state when ``issue`` is
    known), logging the reason."""
    print(f"  {path.name}: {reason}; removing")
    if not dry_run:
        path.unlink(missing_ok=True)
        if issue is not None:
            _clear_campaign_watch_state(issue)


def _process_campaign_entry(
    path: Path,
    entry: dict,
    now: float,
    dry_run: bool,
    threshold: int,
    *,
    daemon_reachable: bool,
    live_ids: set[str] | None,
) -> None:
    """Apply one campaign registry entry's decision: GC at terminal, keep at
    park, respawn a dead ACTIVE session (2-miss guard), and run the
    progress/budget watchdog on a live one."""
    issue = entry.get("issue")
    if not isinstance(issue, int):
        _campaign_reap(path, None, "unreadable/garbled", dry_run)
        return
    status = _task_status(issue)
    if status is None:
        _campaign_reap(path, issue, "task not found / unreadable", dry_run)
        return
    if status in CAMPAIGN_TERMINAL:
        # Stop the session FIRST, then reap. Reaping unmaps the session from
        # its issue, so reap-before-stop would leave an immortal idle session
        # no later pass (session-reconcile included) could attribute and
        # auto-stop (reviewer CONCERN on #586). Daemon-gated: when liveness
        # is unknowable, DEFER the reap to a later tick rather than unmapping
        # a possibly-live session.
        if not daemon_reachable or live_ids is None:
            print(
                f"  campaign #{issue}: terminal ({status}) but daemon unreachable — "
                f"deferring reap until the session can be stopped"
            )
            return
        sid = entry.get("happy_session_id")
        if isinstance(sid, str) and sid in live_ids:
            if not _stop_session(sid, dry_run):
                # Stop failed (or dry-run, which never stops): keep the
                # entry; retry on the next tick.
                print(
                    f"  campaign #{issue}: terminal ({status}); session stop "
                    f"failed/deferred — keeping entry for retry"
                )
                return
            print(f"  campaign #{issue}: terminal ({status}); stopped session {sid}")
        _campaign_reap(path, issue, f"terminal ({status})", dry_run)
        return
    if status not in CAMPAIGN_ACTIVE:
        # Parked (proposed / plan_pending): keep the entry, reset the miss
        # count — it may flip back to active.
        if entry.get("missed", 0) and not dry_run:
            entry["missed"] = 0
            path.write_text(json.dumps(entry, indent=2))
        print(f"  campaign #{issue}: status={status} (parked); keeping entry")
        return
    # ACTIVE: liveness needs the daemon.
    if not daemon_reachable or live_ids is None:
        print(
            f"  campaign #{issue}: status={status}, daemon unreachable — "
            f"skipping liveness/respawn (budget backstop still runs)"
        )
        _campaign_watchdog(issue, entry, now, dry_run, daemon_reachable=False)
        return
    if entry.get("happy_session_id") in live_ids:
        if entry.get("missed", 0) and not dry_run:
            entry["missed"] = 0
            path.write_text(json.dumps(entry, indent=2))
        print(f"  campaign #{issue}: status={status} alive=True")
        _campaign_watchdog(issue, entry, now, dry_run, daemon_reachable=True)
        return
    missed = int(entry.get("missed", 0) or 0) + 1
    print(f"  campaign #{issue}: status={status} alive=False missed={missed}/{threshold}")
    if missed >= threshold:
        _respawn_campaign(entry, dry_run)  # rewrites the registry on success
    elif not dry_run:
        entry["missed"] = missed
        path.write_text(json.dumps(entry, indent=2))


def campaign_pass(
    dry_run: bool,
    threshold: int,
    *,
    daemon_reachable: bool,
    live_ids: set[str] | None,
    now: float | None = None,
) -> None:
    """Crash-recovery + progress watchdog + budget backstop + GC for campaign
    sessions (``campaign-<N>.json`` entries). See the section comment above
    for the four jobs and the cross-pass interactions."""
    now = now if now is not None else time.time()
    entries = _campaign_registry_entries()
    if not entries:
        return
    print(f"campaign: {len(entries)} registered campaign session(s)")
    for path, entry in entries:
        _process_campaign_entry(
            path,
            entry,
            now,
            dry_run,
            threshold,
            daemon_reachable=daemon_reachable,
            live_ids=live_ids,
        )


# ─── gate-push + title-reconcile + tick-runaway pass (2026-06-12) ────────────
#
# Change 2 of the anti-stall redesign: the phone push at gate-park/blocked
# transitions and the canonical-title reconcile move OUT of the LLM-priced
# /issue-tick into this pure-Python pass (the watcher already reads task
# status every 10 min for free, so gate-push latency IMPROVES from the tick's
# backstop cadence to ~10 min). The tick-side PushNotification is KEPT for
# now as a second deduped channel (see the dated removal note in
# .claude/skills/issue-tick/SKILL.md); this pass dedups its own pushes via a
# per-issue state file, so the worst case is one duplicate notification per
# gate transition, never a missed one.
#
# Candidates cover CAMPAIGN sessions too (``campaign-<N>.json``
# registrations, task #586), not just issue sessions. A campaign's one
# push-relevant user gate is ``blocked`` — which IS campaign-terminal
# (:data:`CAMPAIGN_TERMINAL`), so the campaign pass stop-then-reaps the
# registration on the very tick the transition is first observed, BEFORE
# this pass runs in main(). main() therefore snapshots the campaign
# candidates via :func:`_campaign_gate_candidates` ahead of campaign_pass
# and hands them in; enumerating here after the reap would structurally
# miss the campaign's only gate push.
#
# The ISSUE-side registrations have the identical race: ``awaiting_promotion``
# — the most common user gate — is in :data:`TERMINAL`, so the respawn pass's
# :func:`_process_entry` deletes ``issue-<N>.json`` on the first daemon-up
# tick that observes the park, before this pass runs. The cwd fallback can't
# recover the candidate either (spawn-issue sessions open at repo root, not an
# ``issue-<N>`` worktree). main() therefore also snapshots
# ``set(_issue_registrations())`` ahead of the respawn pass and hands it in
# via ``issue_snapshot``; without it the awaiting_promotion push fired only
# when the daemon happened to be down on the transition tick.
#
# Also owns the §4 runaway parachute: `tick_triage.py` writes
# ``tick-runaway-<N>.flag`` on the 3rd consecutive TEARDOWN-verdict tick
# (TERMINAL or GATE-TRANSITION — terminal statuses, over-cap plan_pending,
# stranded campaign crons; cleared by the triage on any streak reset).
# CRON-TEARDOWN keeps whiffing — the #501 class, 1,951 wasted ticks; this
# pass force-stops the flagged issue's session(s), which kills the
# session-scoped cron with them. The force-stop reuses the session-reconcile
# predicate's guards (DONE-status only, no live follow-up, no RUNNING pod, no
# keep-running tag) but skips the 2h-idle + 2-miss accumulation — three
# consecutive teardown-verdict ticks are already the corroboration.

# Per-issue state at ``~/.eps-autonomous/gate-notify-<N>.json``: the last
# status this pass observed (transition detection + push dedup). In the
# terminal-status GC sweep set (reaped at completed/archived).
GATE_NOTIFY_STATE_PREFIX = "gate-notify-"

# User-gate statuses for the push channel. ``plan_pending`` is INCLUDED only
# when the over-cap spend-approval marker confirms it is the user gate (an
# under-cap plan_pending is an in-skill park) — see tick_triage's
# plan_pending_over_cap, shared with the /issue-tick triage.
GATE_PUSH_STATUSES = frozenset({"awaiting_promotion", "blocked"})

# The runaway force-stop acts ONLY on the session-reconcile DONE set. A
# ``blocked`` task also writes runaway flags (it is tick-TERMINAL), but its
# session may have the user live-parked in it — alert loudly, never stop
# (same posture as the reconcile + zombie passes).
RUNAWAY_FORCE_STOP_STATUSES = frozenset({"awaiting_promotion", "completed", "archived"})

# Push channel: the same Telegram helper every my-goat cron nudge uses —
# proven Python-callable path to the phone (the harness PushNotification tool
# only exists inside an LLM turn). Override for tests via
# EPM_TELEGRAM_PUSH_SCRIPT.
_TELEGRAM_PUSH_SCRIPT_DEFAULT = Path.home() / "my-goat" / "scripts" / "notif_enqueue.sh"


def _telegram_push_script() -> Path:
    override = os.environ.get("EPM_TELEGRAM_PUSH_SCRIPT", "").strip()
    return Path(override) if override else _TELEGRAM_PUSH_SCRIPT_DEFAULT


def _telegram_push(msg: str, dry_run: bool) -> bool:
    """Best-effort phone notification via the my-goat DIGEST queue.

    Routed through notif_enqueue.sh (NOTIF_CAT=research), NOT a standalone
    telegram_push.sh send, since 2026-06-12 — Thomas got raw per-transition
    gate pushes (#472/#504 "open to promote") and asked "Why is this being
    sent here": gate notifications are observability, not time-critical, so
    they belong in the 3x/day my-goat digest per its notification-batching
    rules. notif_enqueue.sh has the same arg interface as telegram_push.sh
    by design. The EPM_TELEGRAM_PUSH_SCRIPT override still wins if set.

    Failure is logged LOUDLY but never raises and never crashes the pass —
    the push is observability, and the tick-side PushNotification remains the
    second channel. Returns True on a confirmed enqueue."""
    script = _telegram_push_script()
    if dry_run:
        print(f"  [dry-run] would telegram-push: {msg[:120]}")
        return False
    if not script.is_file():
        print(f"  WARNING: telegram push script missing at {script}; push dropped", file=sys.stderr)
        return False
    try:
        res = subprocess.run(
            ["bash", str(script), msg],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "NOTIF_CAT": "research"},
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  WARNING: telegram push failed: {e}", file=sys.stderr)
        return False
    if res.returncode != 0:
        print(
            f"  WARNING: telegram push failed: {(res.stderr or res.stdout).strip()[:200]}",
            file=sys.stderr,
        )
        return False
    return True


def _gate_notify_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{GATE_NOTIFY_STATE_PREFIX}{issue}.json"


def _load_gate_notify_state(issue: int) -> dict:
    path = _gate_notify_state_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_gate_notify_state(issue: int, *, last_status: str) -> None:
    """Atomic temp+rename persist of the last observed status (the
    transition key for both the push and the title reconcile)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _gate_notify_state_path(issue)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"last_status": last_status, "ts": time.time()}))
    tmp.replace(dest)


def decide_gate_push(status: str | None, last_status: str | None, over_cap: bool) -> bool:
    """Pure push decision: fire exactly once per transition INTO a user gate.

    ``last_status`` is this pass's own previous observation (``None`` =
    never observed — counts as a transition, so a gate reached during
    watcher downtime still pushes once; a duplicate beats a missed one).
    One-shot per transition: the caller persists ``last_status`` in the same
    pass whether or not the send succeeded (the tick-side push is the second
    channel; retrying a failing Telegram send every 10 min would spam the
    log without a user-visible benefit)."""
    if not isinstance(status, str) or status == last_status:
        return False
    return status in GATE_PUSH_STATUSES or (status == "plan_pending" and over_cap)


def _task_title(issue: int) -> str:
    """Task frontmatter title (slug for push messages), '' on any failure."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "view", str(issue), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(out.stdout) if out.returncode == 0 else {}
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return ""
    title = data.get("title") or (data.get("frontmatter") or {}).get("title")
    return title.strip()[:45] if isinstance(title, str) else ""


def _gate_push_message(issue: int, status: str, events: list[dict], over_cap: bool) -> str:
    """Mirror the /issue-tick 3d message shapes (kept under ~200 chars)."""
    slug = _task_title(issue)
    head = f"#{issue} {slug}".rstrip()  # no double space when the title read failed
    if status == "awaiting_promotion":
        msg = f"{head} · clean-result ready — open to promote"
    elif status == "plan_pending" and over_cap:
        cap = os.environ.get("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "100")
        msg = f"{head} parked at plan_pending — over {cap} GPU-h cap; open to approve"
    else:  # blocked
        reason = ""
        for row in reversed(events):
            if isinstance(row, dict) and str(row.get("kind", "")).startswith("epm:failure"):
                note = row.get("note")
                reason = note.strip().splitlines()[0][:80] if isinstance(note, str) else ""
                break
        msg = f"#{issue} BLOCKED: {reason or 'see latest failure marker'} — open it"
    return msg[:200]


def _refresh_self_report(issue: int, status: str, dry_run: bool) -> None:
    """Reconcile the canonical title/self-report with the task status —
    STATUS-TRANSITION-KEYED, never per-pass.

    Constraint (load-bearing): the stalled-detector's signal 1 and the
    session-reconcile idle check both read the self-report's ``ts`` as an
    ACTIVITY signal. An unconditional per-pass rewrite would keep that file
    permanently fresh and structurally disable both passes. A rewrite keyed
    on a STATUS CHANGE cannot mask a stall: the change itself posts
    ``epm:status-changed`` (already refreshing the marker-side signal), and
    a stalled session's status is by definition not changing. Only EXISTING
    self-reports are updated — creating one for a session that never
    self-reported would flip the stalled-detector's deliberate None-skip
    eligibility."""
    try:
        from session_progress_report import read_self_report
    except ImportError:
        return
    try:
        if read_self_report(issue) is None:
            return
    except OSError:
        return
    cmd = [
        "uv", "run", "python", "scripts/session_progress_report.py",
        "--issue", str(issue), "--step", status,
    ]  # fmt: skip
    if dry_run:
        print(f"  [dry-run] would refresh self-report: #{issue} step={status}")
        return
    try:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            print(
                f"  WARNING: self-report refresh failed for #{issue}: "
                f"{(res.stderr or res.stdout).strip()[:200]}",
                file=sys.stderr,
            )
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  WARNING: self-report refresh failed for #{issue}: {e}", file=sys.stderr)


def _runaway_flags() -> list[tuple[int, Path]]:
    """Enumerate ``tick-runaway-<N>.flag`` files (issue, path) — written by
    ``tick_triage.py`` on the 3rd consecutive teardown-verdict tick."""
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for path in AUTONOMOUS_REGISTRY_DIR.glob("tick-runaway-*.flag"):
        try:
            out.append((int(path.stem.removeprefix("tick-runaway-")), path))
        except ValueError:
            continue
    return sorted(out)


def _process_runaway_flag(
    issue: int,
    flag_path: Path,
    sids: list[str],
    running_pod_issues: set[int],
    daemon_reachable: bool,
    dry_run: bool,
) -> None:
    """Force-stop the flagged issue's session(s) under the reconcile guards.

    Guard order: no-live-session (clear the flag — the session-scoped cron
    died with its session, runaway over) -> DONE-status only -> no live
    follow-up -> no RUNNING pod -> no keep-running tag -> daemon up."""
    if not sids:
        if daemon_reachable:
            print(f"  runaway: #{issue} flag present, no live mapped session — clearing flag")
            if not dry_run:
                flag_path.unlink(missing_ok=True)
        return
    status = _task_status(issue)
    if status not in RUNAWAY_FORCE_STOP_STATUSES:
        print(
            f"  runaway: #{issue} flagged (status={status}) but outside the force-stop "
            f"set {sorted(RUNAWAY_FORCE_STOP_STATUSES)} — alert only, flag kept",
            file=sys.stderr,
        )
        return
    if _task_followup_active(issue):
        print(f"  runaway: #{issue} has a live follow-up signal — skip")
        return
    if issue in running_pod_issues:
        print(f"  runaway: #{issue} has a RUNNING pod — skip (pod-safety pass owns it)")
        return
    if _task_keep_running(issue):
        print(f"  runaway: #{issue} carries keep-running — skip")
        return
    if not daemon_reachable:
        print(f"  runaway: #{issue} eligible but daemon unreachable — retry next pass")
        return
    stopped = [sid for sid in sids if _stop_session(sid, dry_run)]
    if stopped:
        _post_progress_marker(
            issue,
            f"[autonomous_session_watch:runaway] force-stopped {len(stopped)} session(s) "
            f"({', '.join(stopped)}) — tick_triage recorded >=3 consecutive teardown-verdict "
            f"ticks, latest at status '{status}' (CRON-TEARDOWN kept whiffing; the #501 "
            f"runaway class). The session-scoped tick cron dies with the session.",
            dry_run,
            label="runaway-stop",
        )
    if not dry_run and len(stopped) == len(sids):
        flag_path.unlink(missing_ok=True)


def _campaign_gate_candidates() -> set[int]:
    """Issue numbers of ``campaign-<N>.json`` registrations — gate-push
    candidates alongside the issue registrations.

    main() snapshots this BEFORE :func:`campaign_pass`: the campaign GC
    stop-then-reaps a terminal campaign's registration on the very tick the
    transition is first observed, and ``blocked`` — the one push-relevant
    user gate in the campaign lifecycle — IS campaign-terminal
    (:data:`CAMPAIGN_TERMINAL`), so enumerating after campaign_pass would
    structurally miss the campaign's only gate push."""
    return {
        entry["issue"]
        for _path, entry in _campaign_registry_entries()
        if isinstance(entry.get("issue"), int)
    }


def gate_push_pass(
    dry_run: bool,
    *,
    daemon_reachable: bool,
    live_ids: set[str] | None = None,
    now: float | None = None,
    campaign_issues: set[int] | None = None,
    issue_snapshot: set[int] | None = None,
) -> None:
    """Per-pass gate push + title reconcile + tick-runaway force-stop.

    Candidates = issues with live mapped sessions UNION registered issues
    UNION campaign registrations (registrations survive brief daemon flaps;
    the live mapping catches manual/worktree-cwd sessions).
    ``campaign_issues`` is main()'s pre-campaign_pass snapshot — the campaign
    GC reaps a ``blocked`` (campaign-terminal) registration before this pass
    runs, so a fresh enumeration here would miss that transition; ``None``
    (direct callers/tests) falls back to enumerating now. ``issue_snapshot``
    is the sibling pre-RESPAWN-pass snapshot of ``_issue_registrations()``
    keys — ``awaiting_promotion`` is respawn-TERMINAL, so ``_process_entry``
    reaps ``issue-<N>.json`` on the first daemon-up tick observing the park;
    same ``None`` fallback. Transition
    detection is per-issue via the ``gate-notify-<N>.json`` state file and
    needs no daemon; the title reconcile and force-stop arms are
    daemon-dependent and degrade to skip/retry when it is unreachable."""
    live = set()
    by_issue: dict[int, set[str]] = {}
    if daemon_reachable:
        live = live_ids if live_ids is not None else _live_session_ids()
        meta = _load_session_meta()
        session_paths = {sid: (m or {}).get("path") for sid, m in meta.items()}
        by_issue = _map_sessions_to_issues(live, _load_session_issue_map(), session_paths)
    if campaign_issues is None:
        campaign_issues = _campaign_gate_candidates()
    if issue_snapshot is None:
        issue_snapshot = set(_issue_registrations())
    candidates = sorted(set(by_issue) | issue_snapshot | campaign_issues)
    if candidates:
        print(f"gate-push: {len(candidates)} candidate issue(s)")
    for issue in candidates:
        status = _task_status(issue)
        if status is None or status in TERMINAL_FOR_GC:
            # completed/archived: never a push target, no title value — and
            # acting here would CHURN against the terminal-status GC (it
            # reaps gate-notify-<N>.json each tick, so this pass would
            # re-create it + re-refresh the self-report every pass, keeping
            # the self-report permanently fresh and structurally disabling
            # the session-reconcile idle signal for done tasks).
            continue
        last_status = _load_gate_notify_state(issue).get("last_status")
        if last_status == status:
            continue  # steady state — nothing transitioned
        events = _task_events(issue)
        over_cap = status == "plan_pending" and plan_pending_over_cap(events)
        if decide_gate_push(status, last_status, over_cap):
            msg = _gate_push_message(issue, status, events, over_cap)
            sent = _telegram_push(msg, dry_run)
            print(
                f"  gate-push: #{issue} {last_status or 'unknown'} -> {status} "
                f"({'sent' if sent else 'push not confirmed'})"
            )
        _refresh_self_report(issue, status, dry_run)
        if not dry_run:
            _save_gate_notify_state(issue, last_status=status)
    flags = _runaway_flags()
    if not flags:
        return
    running_pod_issues = {
        issue for issue, _pod_id, _name in (_running_managed_issue_pods(caller="gate-push") or [])
    }
    for issue, flag_path in flags:
        _process_runaway_flag(
            issue,
            flag_path,
            sorted(by_issue.get(issue, set())),
            running_pod_issues,
            daemon_reachable,
            dry_run,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="log decisions; do not respawn / stop / mutate"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="consecutive dead-checks before re-spawning / stopping a pod "
        "(default 2 = ~20 min at a 10-min cron)",
    )
    parser.add_argument(
        "--gc-only",
        action="store_true",
        help="run ONLY the per-issue state-file GC pass and exit; skip "
        "respawn / pod-safety / stalled-detector. Useful for debugging the "
        "GC in isolation without waiting on a daemon probe.",
    )
    parser.add_argument(
        "--infra-drain-only",
        action="store_true",
        help="run ONLY the infra-drain pass (execute the PM dispatch queue) "
        "and exit; skip every other pass. Mirrors --gc-only; pair with "
        "--dry-run for the post-merge live smoke against the real queue file.",
    )
    args = parser.parse_args(argv)

    lock = _acquire_lock()
    if lock is None:
        print("another autonomous_session_watch run holds the lock; exiting")
        return 0

    # --gc-only short-circuits before the other passes so a debugging run
    # doesn't accidentally trip the destructive paths.
    if args.gc_only:
        gc_pass(args.dry_run)
        return 0

    # --infra-drain-only mirrors --gc-only: run the single pass under the
    # lock (it probes the daemon itself) and exit.
    if args.infra_drain_only:
        infra_drain_pass(args.dry_run, daemon_reachable=_daemon_reachable())
        return 0

    # VM disk-headroom: runs FIRST. A full root disk makes every later
    # subprocess in this very watcher (and every VM session) flaky — alert
    # and reclaim before reasoning about sessions/pods (task #552).
    vm_disk_pass(args.dry_run)

    # The RESPAWN pass needs the daemon (it reasons about session liveness, and
    # `_live_session_ids()` can't tell "daemon up, zero sessions" from "daemon
    # down" — during an outage every session looks dead, which would
    # mass-respawn -> duplicate pods). The POD-SAFETY pass does NOT: it reasons
    # about task STATUS + the live pod list, neither of which needs the daemon.
    # The STALLED-DETECTOR pass partially depends on the daemon — DETECTION
    # works without it (reads files only), but AUTO-RESPAWN needs the daemon
    # (stop+spawn POST to the local daemon RPC). When the daemon is down the
    # stalled-detector degrades to alert-only for those entries.
    #
    # Probe reachability ONCE per main() invocation and reuse the result
    # everywhere so a flap mid-tick can't make different passes disagree
    # about daemon state (and so we don't re-pay the probe cost).
    #
    # Snapshot issue gate-push candidates BEFORE the respawn pass: on the
    # first daemon-up tick that observes a TERMINAL park (`awaiting_promotion`
    # IS in TERMINAL), _process_entry deletes the issue-<N>.json registration,
    # so the gate-push pass below would otherwise miss the most common user
    # gate (the cwd fallback can't recover it — spawn-issue sessions open at
    # repo root). Sibling of the campaign snapshot further down.
    issue_gate_candidates = set(_issue_registrations())
    daemon_reachable = _daemon_reachable()
    live_ids: set[str] = set()
    if daemon_reachable:
        live_ids = _live_session_ids()

        entries = sorted(AUTONOMOUS_REGISTRY_DIR.glob("issue-*.json"))
        print(f"respawn: {len(entries)} registered, {len(live_ids)} live session(s)")
        for path in entries:
            _process_entry(path, live_ids, args.dry_run, args.threshold)
    else:
        print(
            "respawn: Happy daemon unreachable; skipping respawn pass "
            "(won't mass-respawn on an outage). Pod-safety + stalled-"
            "detector still run; stalled-detector falls back to alert-only."
        )

    # Snapshot campaign gate-push candidates BEFORE campaign_pass: its
    # terminal GC stop-then-reaps a `blocked` campaign's registration on the
    # first tick the transition is observed (`blocked` IS campaign-terminal),
    # so the gate-push pass below would otherwise never see the campaign's
    # only user-gate transition.
    campaign_gate_candidates = _campaign_gate_candidates()

    # Campaign pass: crash-recovery + progress watchdog + budget backstop for
    # /campaign sessions (campaign-<N>.json entries, task #586). Runs right
    # after the issue respawn pass; liveness/respawn actions are daemon-gated
    # inside the pass (the budget backstop is not).
    campaign_pass(
        args.dry_run,
        args.threshold,
        daemon_reachable=daemon_reachable,
        live_ids=live_ids if daemon_reachable else None,
    )

    # Pod-safety: runs regardless of daemon reachability. Covers interactive
    # issues (no registry entry) too.
    pod_safety_pass(args.dry_run, args.threshold)

    # Stalled-detector: detects alive-but-stalled autonomous sessions and
    # AUTO-RESPAWNS those whose task is in an ACTIVE status (provided the
    # daemon is reachable); otherwise posts a one-time alert. The detection
    # itself does NOT depend on the daemon (a stalled session's bg-Bash chain
    # death is independent of daemon state), so we always run it — the
    # daemon_reachable flag just gates the recovery action. Run AFTER
    # pod-safety so the `_running_managed_issue_pods` call is fresh
    # (poll_pipeline-posted progress markers from any auto-stopped pod
    # won't accidentally bias the "has_pod" flag).
    stalled_session_pass(args.dry_run, args.threshold, daemon_reachable=daemon_reachable)

    # Orphan sweep: registration-INDEPENDENT cross-check of ACTIVE-status
    # tasks vs live registered sessions. Catches the class the registry-driven
    # passes structurally cannot see: an active task with NO registration at
    # all (#472, 2026-06-10 — entry deleted at a TERMINAL park, task revived
    # by a same-issue follow-up, driver died unobserved for 10.5h). Runs
    # AFTER the respawn + stalled passes so a same-tick recovery by either
    # one is visible via its fresh registry write (the spawn-grace window).
    orphan_sweep_pass(
        args.dry_run,
        args.threshold,
        daemon_reachable=daemon_reachable,
        live_ids=live_ids if daemon_reachable else None,
    )

    # Infra-drain: execute the PM session's adjudicated infra dispatch queue
    # (task #633) into free slots under the cap. Pure executor — the PM is
    # the only ripeness judge; missing/invalid queue file = no-op. Runs AFTER
    # the respawn/orphan recovery passes so any session THEY spawned this
    # tick is already registered (the already-registered guard sees it), and
    # is daemon-gated like every other spawning pass.
    infra_drain_pass(args.dry_run, daemon_reachable=daemon_reachable)

    # Session-reconcile: auto-stop (the default; EPM_SESSION_RECONCILE_AUTOSTOP=0
    # falls back to alert-only) live sessions that outlived their task's
    # park/completion (awaiting_promotion / completed / archived), gated on
    # the no-follow-up + no-RUNNING-pod + idle-grace + keep-running checks.
    # The inverse blind spot of the orphan sweep: that pass finds ACTIVE
    # tasks with no session; this one finds parked/done tasks that still
    # HAVE sessions (2026-06-10 disk incident — idle sessions of completed
    # tasks pinned their worktrees + held deleted-file handles; later the
    # same day 73 registered sessions had accumulated ~35-40GB RSS).
    # Daemon-gated like the respawn pass; reuses main()'s live-id snapshot.
    # Runs AFTER pod-safety so an escaped pod is already being reconciled
    # by the time the pod-skip check reads the RUNNING set.
    session_reconcile_pass(
        args.dry_run,
        args.threshold,
        daemon_reachable=daemon_reachable,
        live_ids=live_ids if daemon_reachable else None,
    )

    # Gate-push + title-reconcile + tick-runaway: phone push on gate-park /
    # blocked transitions (moved out of the LLM-priced /issue-tick — the
    # watcher's 10-min cadence beats the tick's backstop interval), a
    # status-transition-keyed self-report reconcile (NEVER per-pass — an
    # unconditional rewrite would defeat the stalled-detector's + reconcile
    # pass's self-report staleness signals), and the tick-runaway force-stop
    # parachute (#501 class). Transition detection is daemon-independent;
    # the stop/title arms degrade when the daemon is down. Runs BEFORE the
    # reaper snapshot below so a runaway force-stop is already reflected in
    # the session set the reapers see.
    gate_push_pass(
        args.dry_run,
        daemon_reachable=daemon_reachable,
        live_ids=live_ids if daemon_reachable else None,
        campaign_issues=campaign_gate_candidates,
        issue_snapshot=issue_gate_candidates,
    )

    # The two session reapers below run back-to-back with no mutating pass
    # between them, so they share ONE /list snapshot via their `children=`
    # parameter — same probe-once rationale as daemon_reachable above: one
    # fewer daemon RPC per tick, and the two passes can never disagree about
    # the session set. Deliberately NOT reused from the top-of-main
    # `_live_session_ids()` fetch: the respawn / stalled / reconcile /
    # gate-push passes in between mutate the session set, and the reapers
    # should see the post-mutation view. A session the zombie pass stops
    # mid-tick may linger in the shared snapshot for the idle pass; if its
    # wrapper pid is already gone the TTY guard fails toward keep
    # (unreadable /proc -> True -> action "clear"), and if it is still
    # dying the worst case is a redundant, sid-targeted stop of an
    # already-stopped session — never a wrong kill.
    reaper_children = _live_children() if daemon_reachable else None

    # Zombie-wrapper: stop daemon-tracked EPS sessions whose process tree has
    # carried NO inner Claude process for >= threshold checks AND >= the 2h
    # grace window — regardless of issue mapping (the class every registry-/
    # cwd-keyed pass above structurally misses: 25 unmapped "running" zombies
    # accumulated by 2026-06-11). PM-registered sids, non-EPS cwds, and
    # mapped-at-active-status sessions are never touched. Daemon-gated.
    zombie_wrapper_pass(
        args.dry_run,
        args.threshold,
        daemon_reachable=daemon_reachable,
        children=reaper_children,
    )

    # Idle-unmapped: stop unmapped EPS sessions whose Claude transcript has
    # been idle >= the 12h reap window on >= threshold consecutive checks —
    # the live-but-idle complement of the zombie pass (which needs a DEAD
    # inner Claude) and the unmapped complement of session-reconcile (which
    # needs an issue mapping). The 2026-06-12 class: 25 idle unmapped
    # sessions, each with a live Claude + ~8 MCP children, ~23 GB RSS total.
    # PM-registered sids, non-EPS cwds, issue-mapped sessions, TTY-holding
    # wrappers, and unresolvable-transcript sessions are never touched.
    # Daemon-gated.
    idle_unmapped_pass(
        args.dry_run,
        args.threshold,
        daemon_reachable=daemon_reachable,
        children=reaper_children,
    )

    # GC: reap per-issue state files whose tasks are completed/archived OR
    # whose status is unresolvable AND mtime is past the age backstop.
    # Conservative — never touches awaiting_promotion / blocked / live park
    # statuses. Independent of all other passes.
    gc_pass(args.dry_run)

    return 0


def _process_entry(path: Path, live_ids: set[str], dry_run: bool, threshold: int) -> None:
    """Apply one registry entry's decision (read status -> decide -> act).

    Removes the entry on unreadable/missing-task/backstop-age; respawns a dead
    ACTIVE session; otherwise persists an updated miss count. Honours dry_run
    (logs but never mutates / spawns)."""
    try:
        entry = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"  {path.name}: unreadable; removing")
        if not dry_run:
            path.unlink(missing_ok=True)
        return

    issue = entry.get("issue")
    status = _task_status(issue)
    if status is None:
        print(f"  issue #{issue}: task not found / unreadable; removing entry")
        if not dry_run:
            path.unlink(missing_ok=True)
        return

    if time.time() - entry.get("spawned_at", 0) > MAX_ENTRY_AGE_S and status not in ACTIVE:
        print(f"  issue #{issue}: entry older than backstop + not active ({status}); removing")
        if not dry_run:
            path.unlink(missing_ok=True)
        return

    alive = _session_alive(entry, live_ids)
    action, new_missed = decide(status, alive, entry.get("missed", 0), threshold)
    print(
        f"  issue #{issue}: status={status} alive={alive} "
        f"missed={entry.get('missed', 0)}->{new_missed} action={action}"
    )

    if action == "delete":
        if not dry_run:
            path.unlink(missing_ok=True)
    elif action == "respawn":
        _respawn(entry, dry_run)  # rewrites the registry on success
    elif action == "keep" and new_missed != entry.get("missed", 0):
        entry["missed"] = new_missed
        if not dry_run:
            path.write_text(json.dumps(entry, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
