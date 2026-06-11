"""Crash-recovery + pod-safety + stalled-detector watcher for autonomous and
interactive issue sessions.

Seven passes, run in this order:

1. **VM disk-headroom pass.** Watch free space on the VM root filesystem —
   the host of every orchestrator session, the worktree ``.venv``s, the uv
   cache, and the HF cache. Pods have their own guards (``pod_disk_guard.py``,
   the preflight fallocate probe); the VM had none until / hit 100%
   mid-pipeline and every foreground Bash spawn in the orchestrator session
   failed silently — exit 1, zero output — stalling the interpretation loop
   ~20 min, undiagnosable from inside the session (task #552, 2026-06-10).
   Below :data:`VM_DISK_ALERT_FREE_BYTES` (~20 GiB): loud log + ONE
   dashboard-visible marker per low-disk episode. Below
   :data:`VM_DISK_RECLAIM_FREE_BYTES` (~8 GiB): additionally run the safe,
   fail-soft reclaims (``uv cache prune``; sweep ``/tmp/claude-*`` trees idle
   > 3 days). Runs FIRST because a full root disk makes every later
   subprocess in this very watcher flaky; never crashes the pass.
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
   the old alert-only posture (loud log + one-time marker). NEVER
   touches: sessions with no issue mapping (the PM session, chat
   sessions), tasks at any other status (ACTIVE statuses, ``blocked``,
   and ``followups_running`` — a same-issue follow-up round is
   executing there). Motivated by the 2026-06-10 disk-full incident:
   15+ idle sessions of weeks-old completed/archived tasks (the respawn
   pass deletes the registry entry at a TERMINAL status but never stops
   the session) pinned their 10-15G worktrees against the stale-worktree
   sweep and held deleted-file handles (~37G phantom disk usage).
   Daemon-gated like the respawn pass (session liveness is unknowable
   during a daemon outage).
7. **GC pass.** Reap per-issue state files (``manual-issue-<N>.json``,
   ``issue-progress/<N>.json``, ``issue-tick-last-status/<N>.json``,
   ``stalled-<N>.json``, ``orphan-<N>.json``) for tasks in
   :data:`TERMINAL_FOR_GC`
   (``completed`` / ``archived``) — conservative on ``awaiting_promotion``
   and ``blocked`` (the user could still be interacting). Independent of
   the destructive passes; safe to run last. (``session-reconcile-<N>.json``
   is deliberately NOT in its sweep — those files track episodes whose
   task is BY DEFINITION terminal, so the terminal-status GC would reset
   the miss counter every tick; they are reaped by their own
   live-session-keyed GC inside the session-reconcile pass.)

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
  if a task's latest ``epm:run-launched`` marker is NEWER than its latest
  ``epm:promoted`` / ``epm:status-changed`` (i.e. a user-approved inline
  follow-up — the CLAUDE.md "Routing experiment intent → Follow-up" path —
  has provisioned a fresh pod on a promoted/completed/awaiting_promotion/
  archived parent), the stop is SKIPPED with the same once-per-incarnation
  marker semantics as the keep-running exemption (deduped via the
  ``followup_noted`` flag). Precedence: ``keep_running`` (explicit user
  tag) beats ``followup_active`` (inferred from events). The skip re-arms
  naturally on the next tick when the follow-up posts its next
  ``epm:status-changed`` / ``epm:promoted`` event newer than the
  ``epm:run-launched``. The #477 incident, 2026-06-10, motivates this: an
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
import time
from datetime import datetime
from pathlib import Path

# scripts/ is sys.path[0] when run as `python scripts/autonomous_session_watch.py`,
# so its siblings import directly. Reuse spawn_session's daemon readers +
# registry constants, the live RunPod API, AND the canonical managed-pod
# helpers from pod_lifecycle (rather than re-deriving a per-issue regex — the
# old `epm-issue-<N>`-only regex never matched the canonical `pod-<N>` names,
# so the whole pass was dead code).
from pod_lifecycle import _is_managed_pod, _issue_from_pod_name
from runpod_api import list_team_pods
from spawn_session import (
    AUTONOMOUS_REGISTRY_DIR,
    PROJECT_ROOT,
    _infer_issue_from_path,
    _live_session_ids,
    _load_session_issue_map,
    _load_session_meta,
)

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
PARK = {"proposed", "plan_pending", "blocked"}
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
        _SESSION_RECONCILE_ALERT_NOTE_SENTINEL,
        _SESSION_RECONCILE_STOP_NOTE_SENTINEL,
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

# Below this free-bytes threshold the pass ALSO runs the safe reclaims
# (`uv cache prune`, stale /tmp/claude-* sweep). ~8 GiB is already deep in the
# silently-failing-Bash-spawn regime, so reclaiming regenerable caches is
# unambiguously better than waiting for a human.
VM_DISK_RECLAIM_FREE_BYTES = 8 * 2**30

# Re-arm window for the reclaim arm within ONE low-disk episode: don't re-run
# `uv cache prune` + the tmp sweep more than once per this many seconds (the
# first run reclaims nearly everything reclaimable; hot-looping every 10-min
# tick would just churn). A long episode where junk re-accumulates re-fires
# after the window. Tracked via `last_reclaim_ts` in the vm-disk state file.
VM_DISK_RECLAIM_REARM_S = 6 * 3600

# A /tmp/claude-* tree is swept only when NOTHING in it (the dir itself or any
# file under it) was modified within this window. A live session writes its
# /tmp/claude-<port>/.../tasks/*.output files continuously, so its tree always
# has fresh mtimes — the age test IS the live-session guard.
VM_DISK_TMP_SWEEP_AGE_S = 3 * 24 * 3600

# Hard wall-clock bound on `uv cache prune`: if another uv process holds the
# cache lock the prune blocks; kill it at the bound (fail-soft) rather than
# hanging the watcher tick.
VM_DISK_UV_PRUNE_TIMEOUT_S = 300


def decide_vm_disk(
    free_bytes: int,
    *,
    alerted: bool,
    last_reclaim_ts: float | None,
    now: float,
) -> tuple[str, bool, bool]:
    """Pure decision for the VM disk-headroom pass.

    Returns ``(level, do_alert, do_reclaim)``:

    - ``level`` — ``"ok"`` (>= :data:`VM_DISK_ALERT_FREE_BYTES` free),
      ``"low"`` (below the alert threshold), or ``"critical"`` (below
      :data:`VM_DISK_RECLAIM_FREE_BYTES`).
    - ``do_alert`` — fire the once-per-episode alert (level is low or
      critical AND ``alerted`` is not already set for this episode).
    - ``do_reclaim`` — run the safe reclaims (level is critical AND the
      reclaim arm hasn't fired within :data:`VM_DISK_RECLAIM_REARM_S`).
    """
    if free_bytes >= VM_DISK_ALERT_FREE_BYTES:
        return ("ok", False, False)
    level = "critical" if free_bytes < VM_DISK_RECLAIM_FREE_BYTES else "low"
    do_alert = not alerted
    do_reclaim = level == "critical" and (
        last_reclaim_ts is None or now - last_reclaim_ts >= VM_DISK_RECLAIM_REARM_S
    )
    return (level, do_alert, do_reclaim)


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

# Marker kind a follow-up emits when it provisions a pod and launches the
# experiment process. The pod-safety pass treats a `epm:run-launched` whose
# ts is NEWER than the latest done-transition as a live inline follow-up
# and SKIPS the auto-stop (see `decide_pod_safety`'s `followup_active`
# parameter).
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
    """True iff task ``issue`` has an ``epm:run-launched`` marker NEWER than
    its latest done-transition marker (``epm:promoted`` /
    ``epm:status-changed``).

    Predicate for the pod-safety auto-stop exemption: a DONE-status task
    with a fresh ``epm:run-launched`` carries an in-flight, user-approved
    inline follow-up (CLAUDE.md "Routing experiment intent → Follow-up") so
    the pod is legitimately in use. When the follow-up completes, the next
    ``epm:status-changed`` / ``epm:promoted`` event will land newer than
    the ``epm:run-launched`` and this predicate flips False — the auto-stop
    re-arms naturally on the following tick (same semantics as the
    ``keep-running`` tag being removed).

    Called LAZILY by :func:`_process_pod` only on the auto-stop-done branch,
    so the per-task events fetch is paid only for escaped-pod candidates,
    not for every RUNNING pod every tick. ``events`` may be passed in by
    the caller to avoid double-fetching when the events list is already
    loaded (the typical _process_pod path).

    A missing ``epm:run-launched`` returns False (no follow-up signal).
    A missing done-transition is impossible in practice — the caller
    already verified the task's current status is DONE, so at least one
    ``epm:status-changed`` must have fired to put it there. If the read
    nonetheless returns no done-transition (defensive), we conservatively
    return False (no exemption) rather than skip the auto-stop on a
    potentially-stale read.
    """
    if events is None:
        events = _task_events(issue)
    run_launched = _latest_event_ts(events, {_RUN_LAUNCHED_KIND})
    if run_launched is None:
        return False
    done_transition = _latest_event_ts(events, _DONE_TRANSITION_KINDS)
    if done_transition is None:
        return False
    return run_launched > done_transition


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
    task status + the live pod list), so it runs regardless."""
    try:
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
    except (SystemExit, urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


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
    spawn_session rewrites the registry (new id, missed=0) as a side effect."""
    issue = entry["issue"]
    cap = entry.get("auto_approve_gpu_hours", 24.0)
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto", "--auto-approve-gpu-hours", str(cap),
    ]  # fmt: skip
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
    hot-loop. ``prev`` is the prior on-disk payload (when the caller
    already has it loaded) so ``first_seen`` carries forward and the
    age backstop measures the original episode start.
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
    *, alerted: bool, last_reclaim_ts: float | None, prev: dict | None = None
) -> None:
    """Persist the vm-disk episode state atomically (temp + rename).

    ``alerted`` dedups the once-per-episode alert; ``last_reclaim_ts`` re-arms
    the reclaim arm after :data:`VM_DISK_RECLAIM_REARM_S`; ``first_seen``
    carries forward so the state records the episode start (mirrors the
    pod-safety / stalled-detector stores)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _vm_disk_state_path()
    prev_first_seen = (prev or {}).get("first_seen")
    if not isinstance(prev_first_seen, int | float):
        prev_first_seen = time.time()
    payload = {
        "alerted": alerted,
        "last_reclaim_ts": last_reclaim_ts,
        "first_seen": prev_first_seen,
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_vm_disk_state() -> None:
    """Drop the vm-disk state file — the low-disk episode is over (free space
    recovered above the alert threshold), so the next episode alerts afresh."""
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


def _vm_reclaim_uv_cache(dry_run: bool) -> None:
    """``uv cache prune`` — drops unused cache entries (safe: uv re-fetches on
    demand). Fail-soft and hard-bounded by :data:`VM_DISK_UV_PRUNE_TIMEOUT_S`
    so a cache lock held by a concurrent ``uv`` process can't hang the watcher
    tick."""
    cmd = ["uv", "cache", "prune"]
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return
    try:
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=VM_DISK_UV_PRUNE_TIMEOUT_S,
        )
        tail = ((res.stdout or res.stderr).strip().splitlines() or [""])[-1]
        print(f"  vm-disk: uv cache prune rc={res.returncode}: {tail[:200]}")
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  vm-disk: uv cache prune failed (fail-soft): {e}", file=sys.stderr)


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


def _running_managed_issue_pods() -> list[tuple[int, str, str]]:
    """Live RunPod team pods that are RUNNING and managed (``pod-<N>`` or the
    legacy ``epm-issue-<N>``). Returns ``(issue, pod_id, pod_name)`` triples.

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

    A transport error surfaces as an empty list with a logged warning — better
    to skip the pass this tick than to crash the whole run."""
    try:
        pods = list_team_pods()
    except Exception as e:
        print(
            f"  pod-safety: list_team_pods failed ({e}); skipping pass this tick", file=sys.stderr
        )
        return []
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
            f"fresh `epm:run-launched` (newer than the latest done-transition) "
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
                f"events.jsonl shows an `epm:run-launched` marker NEWER than the "
                f"latest done-transition (epm:promoted / epm:status-changed). That "
                f"is the CLAUDE.md 'Routing experiment intent → Follow-up' pattern: "
                f"a user-approved inline follow-up has provisioned a fresh pod on a "
                f"promoted/completed parent, so the pod is legitimately in use. The "
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

    Safety precondition: we MUST know which session id to stop before we
    spawn a fresh one. A garbled / missing ``happy_session_id`` in the
    registry entry would otherwise mean we skip the stop and spawn anyway,
    leaving two `--auto` sessions racing on the same issue (= duplicate
    pods, fastest cost-incident on the watcher). When ``sid`` is falsy /
    non-str, decline this tick and persist state so the next tick (which
    reads a fresh registry entry — the orchestrator or a recent re-spawn
    may have rewritten it) can try again.
    """
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
            prev=ctx.prev_state,
        )


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
    # markers at all is itself a signal).
    latest_marker_ts = _latest_progress_ts(_task_events(issue))
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
    prev_last_self_report_ts = prev_state.get("last_self_report_ts")
    if not isinstance(prev_last_self_report_ts, str):
        prev_last_self_report_ts = None

    # Clear `alerted` + `respawn_count` + `exhausted` + `refresh_attempted`
    # whenever the self-report ts has ADVANCED since the last save — that
    # means the session resumed self-reporting, so the prior episode is
    # over and a future staleness episode can re-alert / re-respawn /
    # re-refresh. Comparison is on the raw ISO string (lexicographic on
    # the canonical trailing-Z UTC format is monotonic).
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
    else:
        alerted = prev_alerted
        respawn_count = prev_respawn_count
        exhausted = prev_exhausted
        refresh_attempted = prev_refresh_attempted

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

    self_gap = f"{self_report_age / 60:.1f}m" if self_report_age is not None else "none"
    marker_gap = f"{marker_age / 60:.1f}m" if marker_age is not None else "none"
    print(
        f"  issue #{issue}: status={task_status} self_gap={self_gap} "
        f"marker_gap={marker_gap} has_pod={has_pod} "
        f"missed={prev_missed}->{new_missed} alerted={alerted} "
        f"respawn_count={respawn_count}/{STALLED_MAX_RESPAWNS} "
        f"daemon_reachable={daemon_reachable} manual={manual} action={action}"
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
    # alerted / respawn_count / exhausted / refresh_attempted flags
    # (cleared above if self-report advanced) + the latest observed
    # self-report ts so the next tick can detect advancement.
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
    # Falls back to the empty set on a transport error (the helper already
    # logs to stderr in that case) so the decision layer just records
    # has_pod=False for every issue this tick — fail-safe.
    running_pods = _running_managed_issue_pods()
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
    prev: dict | None = None,
) -> None:
    """Persist the per-issue orphan-sweep state atomically (temp + rename),
    mirroring :func:`_save_stalled_state`. ``respawn_day`` + ``respawns_today``
    implement the per-UTC-day attempt cap; ``alerted`` dedups the one-time
    alert marker within an episode; ``first_seen`` carries forward so the GC
    age backstop measures the original episode start."""
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
            tid = row.get("id") if isinstance(row, dict) else None
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

    # Lazy events fetch: only orphan candidates pay the per-task read.
    marker_age_s: float | None = None
    is_candidate = not mapped_alive and not (
        entry_age_s is not None and entry_age_s < ORPHAN_SPAWN_GRACE_S
    )
    if is_candidate:
        latest = _latest_progress_ts(_task_events(issue))
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
    print(
        f"  issue #{issue}: status={status} mapped_alive={mapped_alive} "
        f"manual_only={manual_only} marker_gap={gap_str} "
        f"missed={missed}->{new_missed} respawns_today={respawns_today}/{max_per_day} "
        f"alerted={alerted} action={action}"
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
            prev=state,
        )


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

    The session-reconcile twin of :func:`_task_followup_active` (which the
    pod-safety pass keeps with its narrower, #477-validated sets — the two
    predicates are deliberately decoupled so widening the session sweep's
    safety net never changes pod-stop behavior). Same defensive posture:
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
) -> None:
    """Persist the per-issue session-reconcile state atomically (temp +
    rename), mirroring :func:`_save_pod_safety_state`. ``sids`` records the
    live session ids observed this tick (informational — the decision is
    per-issue); ``first_seen`` carries forward so the GC age backstop
    measures the original episode start."""
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
    :data:`SESSION_RECONCILE_STATE_PREFIX`). Returns the cleared issues."""
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
) -> None:
    """Stop every live mapped session for ``issue`` and record the outcome.

    Full success clears the episode state; a partial failure keeps the
    accumulated miss count so the next tick retries the stop for the
    remaining live session(s)."""
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
        if len(stopped) == len(sids):
            _clear_session_reconcile_state(issue)  # episode over
        else:
            _save_session_reconcile_state(
                issue, missed=prev_missed, alerted=prev_alerted, sids=sids, prev=prev_state
            )


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
            issue, sids, status, gap_desc, threshold, dry_run, prev_state, prev_missed, prev_alerted
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
    # A transport error degrades to an empty set — the followup/keep-running
    # skips, the idle grace, and the 2-miss guard remain as safety margins,
    # and the pod-safety pass independently reconciles the pod itself.
    running_pod_issues = {issue for issue, _pod_id, _name in _running_managed_issue_pods()}
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
    running_issues = {issue for issue, _pod_id, _name in running}

    # GC orphaned state BEFORE the per-pod loop, and ALWAYS — even when
    # `running` is empty — so a state file for a pod that left the RUNNING set
    # by ANY path (manual stop/terminate, self-EXIT on TTL/crash) gets cleared.
    # Otherwise a re-used `pod-N` would inherit a stale `missed=1` / `alerted`
    # and be one glitch away from a stop on revival. The age backstop inside
    # `_gc_orphan_pod_safety_state` covers the case where the API is flaky on
    # the exact tick a pod actually disappears.
    _gc_orphan_pod_safety_state(running_issues, dry_run, now=now)

    if not running:
        print("pod-safety: no RUNNING managed pods")
        return
    print(f"pod-safety: {len(running)} RUNNING managed pod(s)")
    for issue, pod_id, _name in running:
        _process_pod(issue, pod_id, now, dry_run, threshold)


def vm_disk_pass(dry_run: bool, now: float | None = None) -> None:
    """Watch VM root-disk headroom; alert once per low-disk episode and run
    the safe reclaims when critically low.

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
    level, do_alert, do_reclaim = decide_vm_disk(
        free,
        alerted=bool(state.get("alerted", False)),
        last_reclaim_ts=last_reclaim_ts,
        now=now,
    )
    free_gib = free / 2**30

    if level == "ok":
        if state:
            print(f"vm-disk: recovered ({free_gib:.1f} GiB free); episode over")
            if not dry_run:
                _clear_vm_disk_state()
        else:
            print(f"vm-disk: ok ({free_gib:.1f} GiB free)")
        return

    # Loud log EVERY tick while low — the cron log is the primary channel.
    print(
        f"vm-disk: {level.upper()} — {free_gib:.1f} GiB free on {VM_DISK_PATH} "
        f"(alert < {VM_DISK_ALERT_FREE_BYTES / 2**30:.0f} GiB, "
        f"reclaim < {VM_DISK_RECLAIM_FREE_BYTES / 2**30:.0f} GiB)",
        file=sys.stderr,
    )

    if do_alert:
        note = (
            f"{_VM_DISK_NOTE_SENTINEL} VM root disk {level.upper()}: "
            f"{free_gib:.1f} GiB free on {VM_DISK_PATH}. Near full, foreground "
            f"Bash spawns in VM sessions start failing silently (exit 1, zero "
            f"output — task #552 incident, 2026-06-10). Reclaim candidates: "
            f"worktree .venvs under .claude/worktrees/, `uv cache prune`, "
            f"stale /tmp/claude-* trees, the HF cache. Posted once per "
            f"low-disk episode."
        )
        issues = _vm_disk_marker_issues()
        if issues:
            for issue in issues:
                _post_progress_marker(issue, note, dry_run, label="vm-disk-low")
        else:
            _append_vm_disk_fallback_event(note, dry_run)

    new_last_reclaim_ts = last_reclaim_ts
    if do_reclaim:
        print("  vm-disk: running safe reclaims (uv cache prune + stale /tmp/claude-* sweep)")
        _vm_reclaim_uv_cache(dry_run)
        swept = _sweep_stale_claude_tmp(now, dry_run)
        refreshed = _vm_free_bytes()
        if refreshed is not None:
            print(
                f"  vm-disk: post-reclaim free {refreshed / 2**30:.1f} GiB "
                f"(swept {swept} stale /tmp/claude-* tree(s))"
            )
        new_last_reclaim_ts = now

    if not dry_run and (do_alert or do_reclaim):
        _save_vm_disk_state(
            alerted=bool(state.get("alerted", False)) or do_alert,
            last_reclaim_ts=new_last_reclaim_ts,
            prev=state,
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
