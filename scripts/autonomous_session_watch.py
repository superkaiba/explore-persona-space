"""Crash-recovery + pod-safety + stalled-detector watcher for autonomous and
interactive issue sessions (plus campaign sessions, task #586).

25 passes ("pass" = one top-level per-tick action block in ``main()``'s
production run order; helpers invoked INSIDE a pass — e.g. the sub-floor
disk sentinel inside pass 1 — and the ``--*-only`` debug entrypoints do
not count; a NEW inline pass block that is not a ``*_pass``-named function
called from ``main()`` also requires bumping ``_ASW_INLINE_PASS_BLOCKS``
in ``scripts/workflow_lint.py``). Item numbers below are STABLE
IDENTIFIERS (cross-referenced throughout this docstring), NOT execution
order. The per-tick execution order is: 1 (VM disk) -> 15 (data-disk) ->
16 (happy-patch) -> 12 (CPU-guard) -> 17 (triage-observer) ->
18 (verdict-disagree) -> 19 (VM-ledger reap) -> 20 (program-orchestrator
recovery) -> 13 (auth-outage guard) -> 2 (crash-recovery) -> 9 (campaign)
-> 3 (pod-safety) -> 4 (stalled-detector) -> 5 (orphan sweep) ->
11 (infra-drain) -> 21 (proposed-infra-sweep) -> 22 (capacity-retry) ->
14 (stale-blocked flag) -> 6 (session-reconcile) -> 23 (gate-push) ->
25 (boot-death) -> 24 (stale-registration) -> 7 (zombie-wrapper) ->
10 (idle-unmapped) -> 8 (GC). The count is lint-pinned: ``workflow_lint.py
--check-asw-docstring-pass-count`` FAILs when the "25 passes" digit, the
numbered-item count, or the live ``*_pass`` set in ``main()`` diverge —
adding a pass means adding a numbered item here AND bumping the digit:

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
     already DONE (``completed`` / ``awaiting_promotion`` / ``archived``) or
     user-paused (``on_hold`` — the #919 pause affordance stops the pod
     BEFORE parking, so ``on_hold`` + RUNNING means the teardown leg failed
     inside the pause window; #980). The experiment is provably finished (or
     deliberately paused), so a still-RUNNING pod is an escaped pod (Step-8
     terminate failed, the pause teardown failed, or it was never run
     through Step 8). Stopping it is unambiguously correct.
   - **ALERT** (loud log + one-time dashboard-visible marker, NO stop) a
     RUNNING pod whose task is in a pod-active status (``approved`` /
     ``running`` / ``verifying`` / ``followups_running``) but has shown no
     real marker
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
   marker have BOTH been frozen > ``STALLED_WINDOW_S`` (default 60 min).
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
   globbed ``issue-*.json``).  A daemon-blocked stop+respawn is never
   dropped: the persisted per-issue episode state plus the
   alerted->eligible escalation executes it on the next daemon-reachable
   tick, and the #845 (c) escalation pages after 2 blocked ticks (#1071).
   A corroboration-debounced alert (#759 downgrade) names the debounce in
   its marker note, never a daemon outage.  A pathological
   reachable/unreachable daemon flap can repeatedly reset the
   K-corroboration ``live_consecutive`` counter and defer escalation,
   bounded in practice by the #845 (c) page after 2 blocked ticks plus
   the persistent staleness re-fire.  At most one stalled-alert marker
   is posted per staleness episode across both alert producers (decide's
   own alert path and the #759 downgrade lane), and a deliberately-parked
   ``blocked`` task (epm:failure + status-changed halt-contract trail) is
   never alerted (#1137).
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
   (default 12h) on >= ``threshold`` consecutive checks. #720 SUBCLASS: an
   unmapped session whose LAST-mapped task was TERMINAL — the "zombie session
   on a completed task" ghost class (respawn pass deletes ``issue-<N>.json``
   at terminal -> session unmapped -> repo-root cwd can't re-map it) — is
   reaped on the SHORT ``LAST_MAPPED_TERMINAL_REAP_S`` window (default 30 min,
   worst case 30 min + 2 ticks = 50 min), NOT the 12h default, via the #720
   breadcrumb + the two protected-class guards in ``_effective_idle_reap_s``.
   This is the home for the completed-task-session reap (#795 verified it: the
   *reconcile* pass cannot see this class — it is unmapped by the time reconcile
   runs — so the idle-unmapped short window owns it; no reconcile-pass change).
   NEVER touches: the
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
12. **CPU/memory-pressure guard pass (task #849; daemon-independent, runs
   in the daemon-independent block right after the disk / happy-patch
   checks).** Escalate-only observability for the shared VM's compute
   pressure (the 2026-07-02 load-226 incident class). Every tick it reads
   ``/proc/loadavg`` + PSI (``/proc/pressure/{cpu,memory}``) +
   ``/proc/meminfo`` MemAvailable and greps the earlyoom journal for kill
   lines. Sustained overload (2 consecutive ticks of load5 > 1.5x nproc,
   PSI-cpu ``some avg10`` > 50, or PSI-memory ``full avg10`` > 10) or a
   SINGLE-TICK MemAvailable drop below 20% (the pre-kill attribution
   window above earlyoom's 10% kill floor) writes ONE attributed
   ``vm-cpu-pressure`` row (top-CPU/top-RSS processes, pid -> issue) to
   ``.claude/cache/cpu-guard-events.jsonl`` + a deduped Telegram push
   (re-alert on >25% load5 growth or a reason-set change; recovery resets
   the episode). Every fire stores the top-process snapshot so subsequent
   ``earlyoom-kill`` rows (surfaced every tick, threshold-independent,
   journal-cursor + key deduped) carry ``attribution_status:
   attributed | unattributed``. WARN-ONLY: never kills, never renices,
   never signals any process. Kill switch ``EPM_DISABLE_CPU_GUARD_PASS=1``;
   ``--cpu-guard-only`` runs just this pass (pair with ``--dry-run`` for a
   live smoke).
13. **Auth-outage guard pass (task #1027; runs right after the per-tick
   daemon probe, BEFORE every spawn arm).** Detects a fleet-wide
   instant-death cause (the 2026-07-03 Anthropic auth outage: every fresh
   session died on arrival and the watcher churned respawns for hours) from
   state the watcher already owns: every watcher-issued spawn records an
   event, and >= 3 instant-freeze respawns (predecessor lived <= 60 min)
   across >= 2 DISTINCT issues inside a 3 h window trigger an episode.
   While active, EVERY spawn arm (crash / stalled / orphan / infra-drain /
   capacity-retry / campaign) is suppressed via the #843 ``"suppressed"``
   channel (callers book nothing), ONE Telegram push fires per episode
   (evidence-enriched from a best-effort ``~/.happy/logs`` auth-signature
   grep — push text only, never the trigger), and recovery is probed by a
   CANARY respawn every 30 min: the first eligible issue-arm spawn probes
   the real CLI-credential auth path; a canary that survives >= 20 min
   resolves the episode, a dead one re-arms one interval later. FAIL-OPEN
   everywhere: any guard error behaves as "no outage", and an episode older
   than 6 h expires with a push (enforced in the pass AND in the gate).
   State singleton ``~/.eps-autonomous/auth-outage.json``; sidecar
   ``.claude/cache/auth-outage-events.jsonl``. Kill switch
   ``EPM_DISABLE_AUTH_OUTAGE_GUARD=1``; ``--auth-outage-only`` runs just
   this pass (pair with ``--dry-run`` for a live smoke).
14. **Stale-blocked flag pass (task #1021; the #742 incident class; runs
   right after the capacity-retry re-drive, between pass 5 and pass 6).**
   FLAG — never flip — a ``blocked`` task whose events show a live healthy
   run: an ``epm:run-launched`` NEWER than the transition into ``blocked``
   PLUS real (non-watcher, non-deliberate-stop) post-launch progress
   within ``EPM_STALE_BLOCKED_PROGRESS_FRESH_S`` (default 2h). The
   watcher-side backstop of the SKILL.md "A successful relaunch also
   reconciles a stale ``blocked``" orchestrator rule (#742 ran healthy
   ~35h at status ``blocked``, 2026-07-01→07-02). One flag per launch
   episode — the dedup state ``stale-blocked-<N>.json`` keys on the
   flagged launch ts, so the same launch never re-alerts while a NEWER
   launch does — emitting a deduped ``epm:progress`` marker naming the
   reconcile command, a sidecar row in
   ``~/.eps-autonomous/stale-blocked-events.jsonl``, and one Telegram
   digest line. Every missing signal fails toward silence; the status
   flip stays with the orchestrator/human (false alert cheap, false flip
   dangerous). Daemon-INDEPENDENT — it spawns nothing (marker posts go
   via the task.py subprocess). Kill switch
   ``EPM_DISABLE_STALE_BLOCKED_FLAG=1``; ``--stale-blocked-only`` runs
   just this pass (pair with ``--dry-run`` for a live smoke).
15. **Data-disk headroom pass (#681; ESCALATE-ONLY; runs right after
   pass 1).** A second disk-watch on the dedicated ``/mnt/eps-data``
   mount (the relocated ``.claude/worktrees/`` tree), driving the
   PERCENT decision helpers so the fire point is size-invariant; no
   reclaim arm runs on the data disk. Clean no-op before the cutover or
   when the mount is absent. (:func:`data_disk_pass`.)
16. **Happy-patch pass (#726; escalate-only, daemon-INDEPENDENT; runs
   right after pass 15).** Proactively surfaces a reverted/drifted Happy
   injection patch (typically from ``npm update happy``) within ~10 min
   — the spawn-path guard is reactive and would only catch it at the
   next spawn. Never re-applies (that needs sudo).
   (:func:`happy_patch_pass`.)
17. **Triage-observer pass (#967; NON-GATING; runs right after
   pass 12).** Post-hoc audit of the /issue Step 9 pre-dispatch
   external-marker triage duty (origin incident #779): flags a
   missing / 'none' triage line against a re-enumerated non-empty
   candidate window. Sidecar rows + capped deduped pushes + capped
   ``epm:progress`` nudges; never mutates task status.
   (:func:`triage_observer_pass`.)
18. **Verdict-disagree observer pass (#1170; NON-GATING; runs right
   after pass 17).** Audits the doubled marker-mode review sites for the
   #825 shape — the latest round whose Claude + Codex durable verdicts
   disagree with no role-matched reconcile and no Codex no-show
   evidence. Sidecar + one deduped push per (issue, site, round); never
   posts a task marker. (:func:`verdict_disagree_pass`.)
19. **VM resource-ledger reap pass (runs right after pass 18;
   daemon-INDEPENDENT, fail-soft).** Drops expired-TTL / dead-PID claims
   from the advisory ``~/.task-workflow/vm-ledger.json`` so a crashed
   session's claim can never wedge the CPU/RAM off-VM routing decision.
   (:func:`vm_ledger_reap_pass`.)
20. **Program-orchestrator recovery pass (#660; daemon-INDEPENDENT; runs
   right after pass 19).** Relaunches the leakage-program bash meta-loop
   daemon (``run_program_orchestrator.sh`` in tmux ``eps-program``) if
   it died mid-program — STOP-sentinel + deliberate-exit guarded; fails
   toward NOT relaunching on any missing signal.
   (:func:`program_orchestrator_pass`.)
21. **Proposed-infra-sweep pass (#690; runs right after pass 11).**
   Always-on backstop dispatching ripe ORPHANED ``proposed`` infra/batch
   tasks whose filer could not self-dispatch (headless filers, crashed
   filers, cap-full filings), bounded by the shared infra session cap.
   (:func:`proposed_infra_sweep_pass`.)
22. **Capacity-retry pass (#642; runs right after pass 21).** Re-drives
   (via ``spawn-issue --auto``) the narrow subclass of ``blocked`` tasks
   whose latest ``epm:failure`` is ``failure_class: infra`` +
   ``reason: no_compute_available``; backoff + a per-UTC-day cap bound
   the churn; every other ``blocked`` task stays parked.
   (:func:`capacity_retry_pass`.)
23. **Gate-push pass (runs right after pass 6).** Per-issue gate push
   (fail-soft Telegram on gate-park / ``blocked`` transitions,
   transition-deduped) + title/self-report reconcile + tick-runaway
   force-stop; consumes ``main()``'s pre-respawn and pre-campaign
   registration snapshots so first-tick GC reaps can't hide a
   transition. (:func:`gate_push_pass`.)
24. **Stale-registration pass (#845 d; daemon-gated; runs right after
   pass 23).** Unregisters (never stops) a LIVE session's registration
   after prolonged transcript idleness so a stale registration stops
   holding the /issue Step 0 single-orchestrator guard; the orphan sweep
   re-drives an ACTIVE task. (:func:`stale_registration_pass`.)
25. **Boot-death pass (#1267; daemon-gated; runs right after pass 23,
   before pass 24).** STOPS a freshly dispatched AUTO session whose
   transcript holds ZERO response rows >= 30 min after ``spawned_at``
   (transcript quiet >= 10 min, live sid — the die-BEFORE-turn-1 class
   every other lane is structurally blind to; #1251-#1256), then leaves
   re-drive to the existing arms. Per-issue per-UTC-day stop cap with a
   LOUD once-per-day cap alert. (:func:`boot_death_pass`.)

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
import functools
import getpass
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

# scripts/ is sys.path[0] when run as `python scripts/autonomous_session_watch.py`,
# so its siblings (`session_resolver`, `pod_lifecycle`, ...) import directly.
# But when this module is imported as `scripts.autonomous_session_watch` (e.g.
# a test doing `from scripts.autonomous_session_watch import
# TRANSIENT_CAPACITY_REASONS`, #659), scripts/ is NOT on sys.path[0] and the
# bare sibling imports below fail with ModuleNotFoundError. Insert the scripts/
# dir so both invocation shapes resolve the siblings identically (the same
# robustness bootstrap backend_poll.py already does for the repo root).
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Reuse spawn_session's daemon readers + registry constants, the live RunPod
# API, AND the canonical managed-pod helpers from pod_lifecycle (rather than
# re-deriving a per-issue regex — the old `epm-issue-<N>`-only regex never
# matched the canonical `pod-<N>` names, so the whole pass was dead code).
import session_resolver  # noqa: E402  (sibling import; follows the sys.path bootstrap above)
from pod_lifecycle import _is_managed_pod, _issue_from_pod_name  # noqa: E402
from runpod_api import PodInfo, get_pod_by_name, list_team_pods  # noqa: E402

# PROJECT_ROOT is git-common-dir-resolved (canonical primary checkout, #844)
# once the imported spawn_session copy contains the fix — a stale pre-fix
# worktree copy keeps the old __file__ resolver until rebased/reaped.
from spawn_session import (  # noqa: E402
    AUTONOMOUS_REGISTRY_DIR,
    DUPLICATE_DISPATCH_NOTE_SENTINEL,
    PROJECT_ROOT,
    _infer_issue_from_path,
    _live_children,
    _live_session_ids,
    _load_pm_session_ids,
    _load_session_issue_map,
    _load_session_meta,
    _takeover_ttl_s,
    dispatch_lease_desc,
    dispatch_lease_fresh,
    last_session_dispatch_age_s,
    record_session_dispatch,
    session_dispatch_stagger_s,
    spawn_output_suppressed,
    stagger_delay_s,
    takeover_sentinel_fresh,
)
from tick_triage import plan_pending_over_cap  # noqa: E402
from worktree_audit import ORPHAN_HOLDER_PATTERNS  # noqa: E402  (codex-companion cmdline patterns)

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

# Grace window after a registration write during which the crash-recovery pass
# treats an ACTIVE entry as "spawn in flight" even if its recorded
# happy_session_id is not yet in the daemon's live set. Covers the
# registration-latency race (#759): `spawn-issue --auto` rewrites the registry
# with a fresh `spawned_at` + a fresh id on every (re)spawn, but that id can
# take seconds-to-minutes to propagate into the daemon's `/list` reply — during
# which `_session_alive` reads False and the 2-miss respawn would spawn a
# duplicate driver. Mirrors :data:`ORPHAN_SPAWN_GRACE_S`, which the orphan
# sweep already applies (`decide_orphan`).
RESPAWN_SPAWN_GRACE_S = 15 * 60


def _respawn_spawn_grace_s() -> float:
    """Crash-recovery spawn-grace window in seconds (env
    ``EPM_RESPAWN_SPAWN_GRACE_MIN``, minutes; default
    :data:`RESPAWN_SPAWN_GRACE_S`). A malformed env value falls back to the
    default — a typo'd var must not disable crash recovery (it would only ever
    SHORTEN the grace, never suppress a genuinely-needed respawn). Mirrors the
    house pattern of :func:`_orphan_staleness_s`."""
    raw = os.environ.get("EPM_RESPAWN_SPAWN_GRACE_MIN")
    if not raw:
        return float(RESPAWN_SPAWN_GRACE_S)
    try:
        return float(raw) * 60.0
    except ValueError:
        return float(RESPAWN_SPAWN_GRACE_S)


# Bounded K-escalation for the stalled detector's live-session corroboration
# (#759, bug class b.1). When the stalled detector wants to respawn an ACTIVE
# session whose Happy id is STILL in the daemon's live set, the first K-1
# consecutive stalled episodes DOWNGRADE to a one-time alert (a transient busy
# stretch — a long /adversarial-planner stage holds the conversation for
# minutes, posting no marker, while a late self-report tick ages past the
# window — must not duplicate the driver); the Kth episode ESCALATES to the
# canonical respawn arm so a persistently-stalled live wrapper (#506 class:
# live Happy wrapper, dead bg-Bash chain) is still recovered. K is a COUNT, not
# seconds. Default 2: at most ~K x STALLED_WINDOW_S = ~2h worst case before a
# live-but-dead-bg wrapper is recovered, within the historical 90-min
# orphan-staleness backstop's rough timescale, while still debouncing one
# transient busy stretch. K=1 would be NO debounce (every transient stretch
# escalates immediately — the duplicate-driver bug this fixes).
STALLED_LIVE_ESCALATION_K = 2


def _stalled_live_escalation_k() -> int:
    """Consecutive live-stall episodes the stalled detector tolerates (alert
    only) before escalating to the canonical respawn arm (#759, bug class b.1).
    Env ``EPM_STALLED_LIVE_ESCALATION_K`` (an integer COUNT, not minutes);
    default :data:`STALLED_LIVE_ESCALATION_K`. A malformed / non-positive env
    value falls back to the default — a typo'd var must neither disable the
    escalation (K too large → starve recovery) nor disable the debounce (K <= 0
    → immediate escalation, the duplicate-driver bug). Mirrors the
    malformed-falls-back-to-default house pattern of :func:`_orphan_staleness_s`
    / :func:`_respawn_spawn_grace_s`."""
    raw = os.environ.get("EPM_STALLED_LIVE_ESCALATION_K")
    if not raw:
        return STALLED_LIVE_ESCALATION_K
    try:
        parsed = int(raw)
    except ValueError:
        return STALLED_LIVE_ESCALATION_K
    if parsed < 1:
        return STALLED_LIVE_ESCALATION_K
    return parsed


def decide(
    status: str,
    alive: bool,
    missed: int,
    threshold: int = 2,
    *,
    entry_age_s: float | None = None,
    spawn_grace_s: float = RESPAWN_SPAWN_GRACE_S,
) -> tuple[str, int]:
    """Pure decision: given a task's status, whether its session is alive, and
    the consecutive-miss count, return ``(action, new_missed)`` where action is
    ``"respawn"`` | ``"keep"`` | ``"delete"``.

    Safety: only an ACTIVE status with a session confirmed dead on
    ``threshold`` consecutive checks (default 2 = ~20 min at a 10-min cron)
    yields ``"respawn"``. Parked tasks reset the miss count and are kept;
    terminal tasks are deleted; an unknown status is kept without ever spawning.

    ``entry_age_s`` is the age of the registry entry's ``spawned_at`` (now minus
    spawned_at). When provided AND below ``spawn_grace_s`` (#759), an ACTIVE
    not-alive entry is treated as "spawn in flight" and KEPT with the miss
    count RESET to 0 — its recorded id may simply not have propagated into the
    daemon's ``/list`` reply yet. A missing/zero ``spawned_at`` yields a large
    (or ``None``) ``entry_age_s`` → no grace → today's behavior exactly: the
    grace can only ever SHORTEN the respawn latency to zero, never SUPPRESS a
    genuinely-needed respawn. Mirrors the orphan sweep's grace
    (``decide_orphan``, :data:`ORPHAN_SPAWN_GRACE_S`).
    """
    if status in TERMINAL:
        return ("delete", 0)
    if status in PARK:
        return ("keep", 0)
    if status in ACTIVE:
        if alive:
            return ("keep", 0)
        # Spawn-grace (#759): a just-(re)spawned session whose id has not yet
        # appeared in the daemon's /list reply is spawn-in-flight, not dead.
        # Reset the miss count so an accumulated miss from BEFORE the
        # (re)spawn cannot straddle the grace window into an immediate respawn.
        if entry_age_s is not None and entry_age_s < spawn_grace_s:
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
# `on_hold` is handled by AUTO_STOP_PAUSED below (pod-safety layer only, #980).
AUTO_STOP_DONE = {"completed", "awaiting_promotion", "archived"}

# Statuses where a RUNNING pod is an escaped pod for the POD-SAFETY pass ONLY
# (#980). `on_hold` = a user pause: the #919 pause affordance stops the pod
# BEFORE parking (teardown first, park last), so on_hold + RUNNING means the
# teardown leg crashed/failed inside the pause window — silent billing.
# DELIBERATELY NOT folded into AUTO_STOP_DONE: SESSION_RECONCILE_DONE (below)
# aliases that set, and the session-reconcile pass must NOT reap a paused
# task's session (the user may be live-parked in it — same conservatism as
# `blocked`). Subset-of-enum invariant pinned alongside AUTO_STOP_DONE by
# test_status_classes_subset_of_authoritative_enum.
AUTO_STOP_PAUSED = {"on_hold"}

# The pod-safety auto-stop trigger set: DONE statuses plus the paused status.
# NOTE: a NEW set object (union) — never mutate AUTO_STOP_DONE in place, or
# the SESSION_RECONCILE_DONE alias would silently widen with it.
POD_SAFETY_AUTO_STOP = AUTO_STOP_DONE | AUTO_STOP_PAUSED

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

# Sentinel distinguishing "carry the prior on-disk value forward" from an
# EXPLICIT value for the #692 wedge fields of `_save_pod_safety_state`. Needed
# because `None` is itself a meaningful value for `wedge_first_seen` (the MF1
# onset-clock CLEAR), so it cannot double as the carry-forward signal the way
# `None` does for the `keep_running_noted` / `followup_noted` flags.
_CARRY = object()

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

# Substring stamped into the once-per-episode "auto-stop FAILED" marker posted
# when the pod-safety stop arm's `pod.py stop` exits non-zero (a REAL failure —
# never the constructed dry-run False). Deduped via the `stop_failed_noted`
# flag in the pod-safety state file; the episode stays RETRYABLE (state is NOT
# cleared, so the stop re-fires every ~10-min tick until it succeeds) — the
# marker replaces stderr-only visibility with a durable task-level record
# (#1155: a persistent RunPod stop-API failure is an unbounded billing leak
# with no task-level evidence). Unlike _AUTOSTOP_NOTE_SENTINEL (deliberately
# absent from _WATCHER_NOTE_SENTINELS — a stopped pod's task is DONE), this
# one IS a member: it is posted while the pod is still RUNNING, so counting it
# as progress would refresh the reconcile/stalled staleness clocks.
_AUTOSTOP_FAILED_NOTE_SENTINEL = "[autonomous_session_watch:pod-auto-stop-failed]"

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

# Substring stamped into the #692 RunPod no-port wedge ALERT marker — posted
# (once per wedge episode, deduped via the `wedge_alerted` flag in the
# pod-safety state file) when the watcher detects the #664 RUNNING-but-no-port
# billing leak (the poller's `_maybe_escalate_runpod_wedge` never ran because
# the poll loop died) but the AUTO-STOP is gated off (inputs unverified on HF,
# the keep-running tag is present, or the keep-running read FAILED). Posted as
# epm:progress, so it MUST be excluded from "real progress" in
# `_latest_progress_ts` — same staleness-filter contract as the other alerts.
_WEDGE_ALERT_NOTE_SENTINEL = "[autonomous_session_watch:runpod-noport-wedge-alert]"

# Substring stamped into the #692 RunPod no-port wedge AUTO-STOP marker — posted
# when the wedge matured past the K floor for >= threshold consecutive checks
# AND the inputs-on-HF + (tri-state) keep-running gates confirm a reversible
# `pod.py stop` is safe. STOP, never terminate (the poller owns terminate).
# Posted as epm:progress; excluded from "real progress" like the alert.
#
# NOTE (#770): the wedge arm's confirmed `keep_running is False AND inputs_ok`
# case no longer routes here — it routes to the new
# `_WEDGE_FAILOVER_NOTE_SENTINEL` (terminate + re-provision) below. This sentinel
# is RETAINED for the staleness-ignore set (the status-class DONE arm's escaped-
# pod handler still posts a reversible `_stop_pod`, and any in-flight pre-#770
# marker in an events log must stay excluded from the progress clock).
_WEDGE_STOP_NOTE_SENTINEL = "[autonomous_session_watch:runpod-noport-wedge-stop]"

# Substring stamped into the #770 RunPod no-port wedge TERMINATE+FAILOVER marker
# — posted when the wedge matured past the K floor for >= threshold consecutive
# checks AND the inputs-on-HF + (tri-state) keep-running gates confirm the
# provably-safe case (`keep_running is False AND inputs_ok=True`). Promotes the
# #692 backstop's strongest action from a reversible `pod.py stop` (which cannot
# heal a host-pinned dead RunPod host, #763) to the SAME irreversible terminate +
# fresh re-provision the poller owns (`backend_poll._failover_wedged_runpod`),
# bounded-once via the shared durable lease + sentinel. Posted as epm:progress
# for the success / already-handled / no-capacity / blocked outcomes; excluded
# from "real progress" like the other wedge markers.
_WEDGE_FAILOVER_NOTE_SENTINEL = "[autonomous_session_watch:runpod-noport-wedge-failover]"

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

# Substring stamped into the one-time "stop failed" marker posted by the #845
# stop-verify respawn FENCE when a stalled session's stop was issued + retried
# once but the session id is STILL in the daemon's live set on the following
# verify tick (daemon ACK != kill — the same contract the zombie-wrapper +
# idle-unmapped reapers already enforce). The fence NEVER spawns next to a
# live superseded session, so this alert is the loud terminal state of a
# failed stop episode. Same staleness-filter contract as the others.
_STALLED_STOP_FAILED_NOTE_SENTINEL = "[autonomous_session_watch:session-stop-failed]"

# Substring stamped into the marker posted at STOP-INITIATION of a #1209
# ``failed-turn-silence`` fence episode (the dead-wake wedge trigger: every
# completed tail turn FAILED + transcript silent >= the dead-silence window).
# The stop is the trigger's ACT — the crash-recovery arm completes the
# respawn once the stopped wrapper is verifiably dead (posting its own
# markers) — so this stop-tick note is the trigger's only durable
# events.jsonl record (the fence's own spawn branch, which posts
# session-auto-respawn, is unreachable for this trigger's fresh-self-report
# shape: post-stop the sid leaves the daemon /list, the wedge pid-gate goes
# inert, and decide() keeps on the fresh boot self-report). Same
# staleness-filter contract as the others.
_STALLED_DEAD_SILENCE_STOP_NOTE_SENTINEL = "[autonomous_session_watch:session-dead-silence-stop]"

# Substring stamped into the one-time marker posted by the #845
# stale-registration pass when it UNREGISTERS a LIVE-but-abandoned session
# registration (transcript idle >= 12h, self-report equally stale, no
# in-flight provision, no fresh worktree activity — the #665 class: a 16h-idle
# registered session held the `/issue` Step 0 single-orchestrator guard and
# blocked every re-drive). Unregister-only: the session itself is NEVER
# stopped (a manual session may hold a user TTY; the SKILL Step 0 stale-wake
# ownership re-check protects a later wake). Same staleness-filter contract.
_STALE_REGISTRATION_NOTE_SENTINEL = "[autonomous_session_watch:stale-registration-unregister]"

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


@functools.lru_cache(maxsize=1)
def _source_stamp() -> str:
    """Self-identification suffix for respawn-class marker notes (#1247):
    host + user + pid + short HEAD sha + root of the RUNNING checkout.

    Deliberately ``Path(__file__)``-derived, NOT the git-common-dir-resolved
    ``PROJECT_ROOT`` (#844) — the whole point is to expose a stale
    worktree/clone copy of this script posting from old code (the two-week
    #662/#663/#867 junk-marker loop had no per-marker process identity, so
    the poster took days of forensics to attribute). ``root=`` is the primary
    discriminator (durable after the process dies); ``pid=`` enables live
    ``/proc`` inspection; ``sha=`` dates the build generation. Fail-soft:
    ``sha=unknown`` on any git failure and ``user=unknown`` when the identity
    lookup fails (``getpass.getuser()`` raises KeyError/OSError with no
    USER/LOGNAME env and no pw entry for the uid — stripped cron/container
    envs); never raises. ``lru_cache``:
    host/user/pid/sha/root are process-constants, computed once."""
    script_root = Path(__file__).resolve().parent.parent
    try:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=script_root,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.SubprocessError, OSError):
        sha = "unknown"
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        # The sanctioned fail-soft carve-out: the stamp is forensic metadata —
        # raising at note-format time AFTER the act guard passed would kill
        # the acting pass over a missing identity string.
        user = "unknown"
    return (
        f"[src: host={socket.gethostname()} user={user} "
        f"pid={os.getpid()} sha={sha} root={script_root}]"
    )


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

# Substring stamped into the one-time alert the stalled / orphan-respawn passes
# post when they would have respawned a task whose latest non-watcher event is
# the over-cap autonomous plan-gate park (``epm:awaiting-spend-approval`` —
# est GPU-h exceeds EPM_PLAN_AUTOAPPROVE_GPU_HOURS, optionally followed only by
# an ``epm:step-completed exit_kind=parked``). This is a user-only gate
# (``task.py set-status <N> approved`` / a re-plan): respawning the session
# only re-reads the same parked plan and re-posts the same step-completed park,
# never advancing. The status-hold variant (SKILL.md Step 9b) keeps the task at
# the ACTIVE status ``followups_running`` while the gate is open, so the
# decide()-level PARK exemption (which covers the plan_pending variant) does not
# catch it — hence this dedicated alive-but-stalled exemption. Dedup'd
# self-containedly in the events log: suppressed when a marker carrying this
# sentinel already exists NEWER than the gating ``epm:awaiting-spend-approval``
# (a fresh spend-approval episode re-arms the alert). Same staleness-filter
# contract as the others. Incident: task #653, 2026-06-18 — 5 respawn-and-park
# cycles in ~4h while a 132 GPU-h plan sat over the 100h auto-approve cap, each
# respawn re-posting the same ``epm:step-completed step=2c exit_kind=parked``
# and exiting.
_SPEND_APPROVAL_SKIP_NOTE_SENTINEL = "[autonomous_session_watch:spend-approval-skip]"

# Substring stamped into the one-time alert the stalled / orphan-respawn passes
# post when they would have respawned a task whose latest word is a PROSE
# "USER PAUSE" hold note — a non-watcher note beginning with the literal
# ``USER PAUSE`` (any marker kind) with no real progress marker strictly newer.
# A prose-only hold is the documented anti-pattern (the durable affordance is
# ``task.py set-status <N> on_hold`` per SKILL.md § User pause affordance); this
# exemption is defense-in-depth for sessions that still post one. Deduped
# self-containedly in the events log: suppressed when a marker carrying this
# sentinel already exists at/after the latest pause note (a fresh pause note
# re-arms the alert). Same staleness-filter contract as the others. Incident:
# task #816, 2026-07-02 — a session posted a prose ``USER PAUSE ...`` hold
# (epm:progress, 06:44Z) and left the task at the ACTIVE status ``running``;
# the orphan-respawn pass cannot parse prose and respawned against the hold
# (attempt 1/2, 08:33Z).
_USER_PAUSE_SKIP_NOTE_SENTINEL = "[autonomous_session_watch:user-pause-hold-skip]"

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
# Stamped into the post-stop event the idle-unmapped pass writes when the reap
# fired on the CORROBORATING-IDLENESS FALLBACK path (the primary happy-log
# transcript signal was unavailable, so the six-gate tmux session_activity
# fallback supplied the idle age — #695). DISTINCT from the primary
# `_IDLE_UNMAPPED_STOP_NOTE_SENTINEL` so an operator can tell a fallback reap
# apart from a transcript-driven reap in the events stream (the fallback note
# must NOT claim "transcript idle" — it never read one).
_IDLE_UNMAPPED_STOP_FALLBACK_NOTE_SENTINEL = (
    "[autonomous_session_watch:idle-unmapped-stop-fallback]"
)

# Substring stamped into the marker the infra-drain pass posts after
# dispatching an autonomous session for a PM-queue ID (task #633). The #633
# task itself becomes ACTIVE seconds after dispatch — a watcher-authored
# dispatch note must never count as "real progress" for the orphan/stalled
# staleness clocks, or it would mask a subsequent stall of the very session
# it just spawned.
_INFRA_DRAIN_NOTE_SENTINEL = "[autonomous_session_watch:infra-drain-dispatch]"

# Substring stamped into the marker the proposed-infra sweep posts after
# dispatching an ORPHANED ripe `proposed` infra task — one with NO PM queue
# entry (filed by a context that could not self-dispatch: a pod, a manual
# `task.py new`, a crashed filer, or a cap-full file-time wrapper, #690). The
# dispatched task becomes ACTIVE seconds later, so this watcher-authored
# dispatch note must never count as "real progress" for the orphan/stalled
# staleness clocks (same contract as _INFRA_DRAIN_NOTE_SENTINEL).
_PROPOSED_INFRA_SWEEP_NOTE_SENTINEL = "[autonomous_session_watch:proposed-infra-sweep-dispatch]"

# Substring stamped into the marker the capacity-retry pass posts after it
# re-drives a `blocked`-on-transient-infra task (task #642 class:
# `no_compute_available` after a GCP capacity miss). Same staleness-filter
# contract as the others — the re-driven session becomes ACTIVE seconds after
# the spawn, so a watcher-authored re-drive note must never count as "real
# progress" for the orphan/stalled staleness clocks.
_CAPACITY_RETRY_NOTE_SENTINEL = "[autonomous_session_watch:capacity-retry]"

# Substring stamped into the one-time "daily capacity-retry cap exhausted"
# marker. Same staleness-filter contract; posted at most once per UTC day per
# task (deduped via the retry-state `alerted_day` field) so an exhausted task is
# dashboard-visible without per-tick marker spam.
_CAPACITY_RETRY_EXHAUSTED_NOTE_SENTINEL = "[autonomous_session_watch:capacity-retry-exhausted]"

# Substring stamped into the stale-blocked flag marker (task #1021): a
# `blocked` task whose events show an `epm:run-launched` NEWER than the
# transition into `blocked` plus fresh POST-LAUNCH progress — the #742 class
# (a crash-fix relaunch succeeded but the earlier failed round's `blocked`
# was never flipped back; #742 ran healthy ~35h at status `blocked`).
# FLAG-ONLY: the pass posts a deduped marker + sidecar row + Telegram digest
# line and NEVER mutates status (the orchestrator's own-relaunch reconcile
# rule in SKILL.md — or a human — flips it). Same staleness-filter contract
# as every other watcher-posted sentinel.
_STALE_BLOCKED_FLAG_NOTE_SENTINEL = "[autonomous_session_watch:stale-blocked-flag]"

# Filename prefix for the per-issue stale-blocked dedup state file at
# ``~/.eps-autonomous/stale-blocked-<N>.json`` (keyed on
# ``flagged_run_launched_ts`` — one alert per launch episode; a NEWER launch
# is a new episode and re-alerts). Mirrors the capacity-retry state-file
# layout; reaped by the generalized GC at `completed`/`archived`.
STALE_BLOCKED_STATE_PREFIX = "stale-blocked-"

# Freshness window (seconds) for the POST-LAUNCH progress leg of
# ``decide_stale_blocked_flag``. Matches the STALLED_MARKER_WINDOW_S_DEFAULT
# scale (2h — the #845 marker-heartbeat window); env-tunable via
# ``EPM_STALE_BLOCKED_PROGRESS_FRESH_S`` (seconds). A too-tight window
# UNDER-flags (missed alert, status quo), never mis-flags.
STALE_BLOCKED_PROGRESS_FRESH_S_DEFAULT = 2 * 3600

# Substring stamped into the boot-death stop marker (task #1267): a freshly
# dispatched AUTO session whose transcript holds ZERO response rows
# (assistant OR api-error) >= 30 min after `spawned_at` — the
# die-BEFORE-turn-1 class (#1251-#1256: 9-row / ~11 KB transcripts frozen
# ~7 s post-spawn, invisible to every lane until the 12h stale-registration
# pass) — OR whose 256 KB tail segments to >= 1 completed turn, ALL failed —
# the #1287 boot-turn-refusal class (#1277: the boot turn ran and was
# refusal-killed before the tick cron was armed). The pass STOPS the session
# (auto registrations only) and leaves re-drive to the existing arms. Same
# staleness-filter contract as every other watcher-posted sentinel
# (anti-liveness is load-bearing: an unsentineled note would refresh
# `_latest_progress_ts` and mask the orphan/stalled staleness clocks).
_BOOT_DEATH_STOP_NOTE_SENTINEL = "[autonomous_session_watch:boot-death-stop]"

# Substring stamped into the one-time "daily boot-death stop cap exhausted"
# marker (task #1267). DELIBERATELY LOUD once per (issue, UTC day) — a
# recorded deviation from the #1241 siblings' quiet-at-cap posture: those
# lanes have a slow backstop that still ACTS at cap, while here the fallback
# IS the 12h silence this lane exists to kill, so the cap moment is the
# highest-value alert of the day.
_BOOT_DEATH_CAP_NOTE_SENTINEL = "[autonomous_session_watch:boot-death-cap-exhausted]"

# Filename prefix for the per-issue boot-death day-cap state file at
# ``~/.eps-autonomous/boot-death-<N>.json`` (fields: ``stop_day`` /
# ``stops_today`` / ``cap_alerted_day``). Reaped by the generalized GC at
# `completed`/`archived` only — `proposed` (the incident-class status) is
# not terminal, so a live loop's day counter is never reset mid-episode;
# the day-keyed counter self-expires at the UTC day roll anyway.
BOOT_DEATH_STATE_PREFIX = "boot-death-"

# Boot-death lane constants (#1267; grounding in the task plan §11):
# - window = 2x RESPAWN_SPAWN_GRACE_S (3 cron ticks): dead transcripts
#   freeze ~7 s post-spawn while the healthy #1251 re-dispatch had 49
#   assistant rows at 13 min — >10x margin over healthy first-turn latency.
# - quiet = one watcher cron interval (the in-flight-first-turn guard).
# - tail cap == the #1104 production wedge window (256 KB): a boot-death
#   transcript is ~11 KB (23x headroom); a LARGER transcript cannot be a
#   ZERO-RESPONSE boot-death, so the whole-file probe short-circuits without
#   the read. #1287 (arm 2, boot-refusal) keeps every constant unchanged;
#   this same cap now ALSO sizes the arm-2 seek-tail read
#   (`_transcript_tail_rows`), deliberately equal to the #1104/#1209 wedge
#   window so both lanes see the same evidence horizon.
# - stop cap 3/day = #1241 parity (TICK_WEDGE_RESPAWNS_PER_DAY).
BOOT_DEATH_WINDOW_S = 30 * 60
BOOT_DEATH_QUIET_S = 10 * 60
BOOT_DEATH_TAIL_BYTES = 256 * 1024
BOOT_DEATH_STOPS_PER_DAY = 3

# OPT-IN heartbeat sentinel for legitimately-slow phases (off-pod analyzer
# verifier rounds, in-flight Anthropic Batch polling). UNLIKE every other
# sentinel in this file, this one is stamped by the LONG-RUNNING PHASE
# itself (NOT by the watcher), so it IS real progress and MUST count toward
# _latest_progress_ts / _latest_nonwatcher_event_ts — i.e. it is the INVERSE
# of _WATCHER_NOTE_SENTINELS and is deliberately NOT a member of that set
# (see _long_phase_heartbeat_reason and tests/test_..._sentinel_not_in_...).
# An emitter that includes this substring in its epm:progress note opts into
# the longer stalled-detector leash (LONG_PHASE_HEARTBEAT_FRESH_S). The
# watcher only RECOGNIZES the convention; teaching emitters to stamp it is
# separate, src/-level work (out of this fix's scope).
_LONG_PHASE_HEARTBEAT_PREFIX = "[long-phase-heartbeat]"

# All watcher-posted note substrings to exclude from `_latest_progress_ts`.
# Pulled into one frozenset so every pass's filter is uniform: add a new
# watcher-posted marker -> add its sentinel here -> _latest_progress_ts
# transparently excludes it without an extra special case. (The deliberate
# session-stop exclusion is NOT a member — this set is SUBSTRING-matched,
# which would break the prefix boundary; it lives inline in
# _latest_progress_ts as a prefix/by check instead; #990.)
_WATCHER_NOTE_SENTINELS: frozenset[str] = frozenset(
    {
        _ALERT_NOTE_SENTINEL,
        _AUTOSTOP_FAILED_NOTE_SENTINEL,
        _KEEP_RUNNING_NOTE_SENTINEL,
        _FOLLOWUP_NOTE_SENTINEL,
        _WEDGE_ALERT_NOTE_SENTINEL,
        _WEDGE_STOP_NOTE_SENTINEL,
        _WEDGE_FAILOVER_NOTE_SENTINEL,
        _STALLED_ALERT_NOTE_SENTINEL,
        _STALLED_RESPAWN_NOTE_SENTINEL,
        _STALLED_EXHAUSTED_NOTE_SENTINEL,
        _STALLED_STOP_FAILED_NOTE_SENTINEL,
        _STALLED_DEAD_SILENCE_STOP_NOTE_SENTINEL,
        _STALE_REGISTRATION_NOTE_SENTINEL,
        _VM_DISK_NOTE_SENTINEL,
        _ORPHAN_RESPAWN_NOTE_SENTINEL,
        _ORPHAN_ALERT_NOTE_SENTINEL,
        _FOLLOWUPS_AWAITING_CHILD_NOTE_SENTINEL,
        _FOLLOWUP_ROUND_REPARK_NOTE_SENTINEL,
        _SPEND_APPROVAL_SKIP_NOTE_SENTINEL,
        _USER_PAUSE_SKIP_NOTE_SENTINEL,
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
        _IDLE_UNMAPPED_STOP_FALLBACK_NOTE_SENTINEL,
        _INFRA_DRAIN_NOTE_SENTINEL,
        _PROPOSED_INFRA_SWEEP_NOTE_SENTINEL,
        _CAPACITY_RETRY_NOTE_SENTINEL,
        _CAPACITY_RETRY_EXHAUSTED_NOTE_SENTINEL,
        _STALE_BLOCKED_FLAG_NOTE_SENTINEL,
        _BOOT_DEATH_STOP_NOTE_SENTINEL,
        _BOOT_DEATH_CAP_NOTE_SENTINEL,
        # Posted by spawn_session.py (not the watcher) when a duplicate --auto
        # dispatch was suppressed at registration (#843 M2) — same contract:
        # a suppression note is bookkeeping, never real progress.
        DUPLICATE_DISPATCH_NOTE_SENTINEL,
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
# so a generous window is plenty of margin.
#
# Raised 45 -> 60 min (#759, bug class b.2): the `/issue-tick` self-report cron
# fires every 45 min, so a 45-min window EQUALLED the cadence and was
# straddle-able by ordinary cron jitter (a single late tick ages the
# self-report past 45 min while a multi-minute subagent holds the conversation
# and posts no marker). 60 min = 45-min cadence + a full extra 10-min cron tick
# of slack, so the self-report must miss MORE than one tick before the window
# opens. Env-tunable via EPM_STALLED_WINDOW_MIN (minutes); a malformed value
# falls back to the default. Mirrors the _orphan_staleness_s house pattern.
STALLED_WINDOW_S_DEFAULT = 60 * 60


def _stalled_window_s() -> float:
    """Stalled-detector staleness window in seconds (env
    ``EPM_STALLED_WINDOW_MIN``, minutes; default
    :data:`STALLED_WINDOW_S_DEFAULT`). A malformed env value falls back to the
    default — a typo'd var must not disable the stalled detector."""
    raw = os.environ.get("EPM_STALLED_WINDOW_MIN")
    if not raw:
        return float(STALLED_WINDOW_S_DEFAULT)
    try:
        return float(raw) * 60.0
    except ValueError:
        return float(STALLED_WINDOW_S_DEFAULT)


STALLED_WINDOW_S = _stalled_window_s()


# #845 (a-i): dedicated freshness window for the stalled detector's SIGNAL 2
# (the newest non-watcher marker). Decoupled from the 60-min self-report
# window: legitimate marker gaps of 60-120 min are common (incident #761's
# 1h21m off-pod analysis stretch drew a wasted respawn; #763's overlap began
# with a stall declared against a session that had posted a real marker within
# the prior 2h). 2h = 2x STALLED_WINDOW_S, comfortably under the pod-side
# ALERT_STALE_HOURS (6h) and above LONG_PHASE_HEARTBEAT_FRESH_S (90 min).
# NOTE the deliberate trade: signal 2 counts ANY non-watcher marker — including
# markers posted by OTHER actors (PM notes, pod-side sentinel relays) — so the
# widened window can shield a genuinely wedged session behind a third-party
# marker for up to 2h. The (e) prompt-wedge fast lane, which bypasses this
# window on direct transcript evidence, is the deliberate mitigation.
STALLED_MARKER_WINDOW_S_DEFAULT = 2 * 3600


def _stalled_marker_window_s() -> float:
    """Marker-heartbeat window in seconds for the stalled detector's signal 2
    (env ``EPM_STALLED_MARKER_HEARTBEAT_MIN``, minutes; default
    :data:`STALLED_MARKER_WINDOW_S_DEFAULT`). A malformed env value falls back
    to the default — mirrors :func:`_stalled_window_s`."""
    raw = os.environ.get("EPM_STALLED_MARKER_HEARTBEAT_MIN")
    if not raw:
        return float(STALLED_MARKER_WINDOW_S_DEFAULT)
    try:
        return float(raw) * 60.0
    except ValueError:
        return float(STALLED_MARKER_WINDOW_S_DEFAULT)


# #1137: window for the halt-contract trail behind a deliberate `blocked`
# park — an `epm:failure` marker within this many seconds BEFORE (or
# _BLOCKED_PARK_FAILURE_SLACK_AFTER_S after) the newest `epm:status-changed`.
# The canonical halt sequence posts them ~1s apart (#1092: 13:21:34Z failure,
# 13:21:35Z status-changed); 1h absorbs slow orchestrator paths (a
# poller-posted failure classified minutes before the park).
_BLOCKED_PARK_FAILURE_WINDOW_S = 3600.0
_BLOCKED_PARK_FAILURE_SLACK_AFTER_S = 300.0


# #845 (b): worktree-activity hold. A file under the issue's worktree edited
# within this window is direct evidence an implementer/analyzer subagent is
# mid-edit (incident #812: the killed session had edited a file 57s before the
# respawn; #779's respawn killed an in-flight implementer). 15 min (==
# RESPAWN_SPAWN_GRACE_S) covers implementer inter-edit gaps (minutes-scale
# tool-call cadence) without a long false-negative tail.
WT_ACTIVITY_FRESH_S_DEFAULT = 15 * 60

# Bound on consecutive held ticks (~1h at the 10-min cron — the same
# timescale as ORPHAN_STALENESS_S). A cross-writer touching the worktree
# forever must not become a permanent false negative: the 7th tick respawns
# regardless. A bound, not a latch.
WT_HOLD_MAX_TICKS = 6


def _wt_activity_fresh_s() -> float:
    """Worktree-activity hold window in seconds (env
    ``EPM_STALLED_WT_ACTIVITY_MIN``, minutes; default
    :data:`WT_ACTIVITY_FRESH_S_DEFAULT`). Malformed env falls back."""
    raw = os.environ.get("EPM_STALLED_WT_ACTIVITY_MIN")
    if not raw:
        return float(WT_ACTIVITY_FRESH_S_DEFAULT)
    try:
        return float(raw) * 60.0
    except ValueError:
        return float(WT_ACTIVITY_FRESH_S_DEFAULT)


# #845 (e): minimum count of consecutive trailing wedge-evidence rows
# (verified `{"type": "queue-operation", "operation": "dequeue"}` records
# and/or promptless prompt-type user rows) in the session transcript before
# the prompt-wedge trigger escalates straight to the respawn arm. #779 showed
# 5 dequeues over ~90 min; N=3 fires ~2 prompts earlier while tolerating 1-2
# rows racing a mid-delivery prompt or a slow turn. The threshold is
# ungrounded beyond the incident — the env knob + test coverage is the guard.
TICK_WEDGE_MIN_DEQUEUED = 3


def _tick_wedge_min_dequeued() -> int:
    """Prompt-wedge trigger threshold (env ``EPM_TICK_WEDGE_MIN_DEQUEUED``,
    an integer COUNT; default :data:`TICK_WEDGE_MIN_DEQUEUED`). Malformed /
    non-positive env falls back — a typo'd var must neither disable the
    trigger (huge N) nor fire it on every swallowed prompt (N <= 0)."""
    raw = os.environ.get("EPM_TICK_WEDGE_MIN_DEQUEUED")
    if not raw:
        return TICK_WEDGE_MIN_DEQUEUED
    try:
        parsed = int(raw)
    except ValueError:
        return TICK_WEDGE_MIN_DEQUEUED
    if parsed < 1:
        return TICK_WEDGE_MIN_DEQUEUED
    return parsed


# #1104: minimum count of consecutive trailing API-ERROR turns (assistant
# rows with top-level `isApiErrorMessage: true` — a usage-policy refusal or
# 429/529 error turn) before the prompt-wedge trigger escalates. The direct
# analogue of TICK_WEDGE_MIN_DEQUEUED: 3 CONSECUTIVE failed wake turns with
# zero successful turns. Incident #1074: 38 refused wake turns / ~2h
# unrecovered — every refusal row classified "assistant" and RESET the run,
# hiding the wedge from this lane entirely.
TICK_WEDGE_MIN_API_ERRORS = 3


def _tick_wedge_min_api_errors() -> int:
    """Api-error wedge trigger threshold (env ``EPM_TICK_WEDGE_MIN_API_ERRORS``,
    an integer COUNT; default :data:`TICK_WEDGE_MIN_API_ERRORS`).

    ONE deliberate divergence from :func:`_tick_wedge_min_dequeued`: ``0``
    DISABLES the api-error trigger (an explicit kill switch for the NEW
    trigger class — the ``min_api_errors > 0`` guard in
    :func:`decide_prompt_wedge` then never fires). Malformed / negative env
    falls back to the default."""
    raw = os.environ.get("EPM_TICK_WEDGE_MIN_API_ERRORS")
    if not raw:
        return TICK_WEDGE_MIN_API_ERRORS
    try:
        parsed = int(raw)
    except ValueError:
        return TICK_WEDGE_MIN_API_ERRORS
    if parsed < 0:
        return TICK_WEDGE_MIN_API_ERRORS
    return parsed


# #1127: minimum count of consecutive trailing FAILED wake-TURNS (a completed
# turn whose LAST response row is an api-error — mid-turn assistant heartbeats
# do not rescue it; see _segment_wake_turns) before the prompt-wedge trigger
# escalates. The TURN-granularity analogue of TICK_WEDGE_MIN_API_ERRORS:
# incidents #1098 (5bdae5b8) and #1090 (5e464f3d) each posted 1-3 successful
# assistant rows per wake BEFORE dying in an api-error row, so the row-level
# counters reset every cycle and the lane stayed silent (run=0, api_run<=1 on
# the real 256 KB tails). 3 = one-off + margin (same rationale as the #1104
# knob); the #1098 tail held 8 failed turns, #1090 held 5.
TICK_WEDGE_MIN_FAILED_TURNS = 3


def _tick_wedge_min_failed_turns() -> int:
    """Failed-turn wedge trigger threshold (env
    ``EPM_TICK_WEDGE_MIN_FAILED_TURNS``, an integer COUNT; default
    :data:`TICK_WEDGE_MIN_FAILED_TURNS`). Mirrors
    :func:`_tick_wedge_min_api_errors` exactly (a NEW trigger class): ``0``
    DISABLES the failed-turn-run trigger — the explicit kill switch (with
    ``EPM_TICK_WEDGE_MIN_FAILED_TOTAL=0`` it restores the exact pre-#1127
    lazy gate). Malformed / negative env falls back to the default."""
    raw = os.environ.get("EPM_TICK_WEDGE_MIN_FAILED_TURNS")
    if not raw:
        return TICK_WEDGE_MIN_FAILED_TURNS
    try:
        parsed = int(raw)
    except ValueError:
        return TICK_WEDGE_MIN_FAILED_TURNS
    if parsed < 0:
        return TICK_WEDGE_MIN_FAILED_TURNS
    return parsed


# #1127 option (c): windowed TOTAL (non-consecutive) failed wake-turns for the
# alternating-storm rate trigger — incident c16b10ca lost ~every other wake for
# ~5.7h (00:26-06:07Z), a shape NO consecutive counter (row- or turn-level) can
# ever fire on. The measured c16b10ca 256KB tail held 4-5 windowed failed
# TURNS — below this threshold BY DESIGN (plan v4 §3.3 accept-run-coverage;
# threshold deliberately NOT lowered): the lane targets storms DENSER than
# that incident, while a healthy session needs 6 turn-ENDING failures inside
# the window (>> any observed healthy rate; 0/20 negative sweep).
TICK_WEDGE_MIN_FAILED_TOTAL = 6


def _tick_wedge_min_failed_total() -> int:
    """Failed-turn RATE trigger threshold (env
    ``EPM_TICK_WEDGE_MIN_FAILED_TOTAL``, an integer COUNT; default
    :data:`TICK_WEDGE_MIN_FAILED_TOTAL`). Mirrors
    :func:`_tick_wedge_min_api_errors` (a NEW trigger class): ``0`` DISABLES
    the failed-turn-rate trigger — the kill switch for the alternating-storm
    lane. Malformed / negative env falls back to the default."""
    raw = os.environ.get("EPM_TICK_WEDGE_MIN_FAILED_TOTAL")
    if not raw:
        return TICK_WEDGE_MIN_FAILED_TOTAL
    try:
        parsed = int(raw)
    except ValueError:
        return TICK_WEDGE_MIN_FAILED_TOTAL
    if parsed < 0:
        return TICK_WEDGE_MIN_FAILED_TOTAL
    return parsed


# #1127: window for the failed-turn RATE trigger, anchored to the newest
# parseable ROW timestamp in the tail (not wall-clock, so the pure predicate
# is deterministic + replay-testable). 120 min covers the measured incident
# tail spans (~1.3-2.5h at 256 KB) so the TAIL, not the clock, is usually
# binding (which only under-counts — fail toward NO-FIRE); matches the house
# 2h precedent (STALLED_MARKER_WINDOW_S_DEFAULT).
TICK_WEDGE_RATE_WINDOW_S = 7200


def _tick_wedge_rate_window_s() -> float:
    """Failed-turn rate-trigger window in seconds. Env
    ``EPM_TICK_WEDGE_RATE_WINDOW_MIN`` is in MINUTES and is converted to the
    SECONDS constant returned here (the ``EPM_STALLED_WT_ACTIVITY_MIN``
    minutes-env -> seconds-constant precedent); default
    :data:`TICK_WEDGE_RATE_WINDOW_S` (7200 s = 120 min). It is a WINDOW, not
    a trigger, so malformed / NON-POSITIVE env falls back to the default (the
    :func:`_stalled_window_s` pattern) — disabling the rate lane is
    ``EPM_TICK_WEDGE_MIN_FAILED_TOTAL=0``, never a zero window."""
    raw = os.environ.get("EPM_TICK_WEDGE_RATE_WINDOW_MIN")
    if not raw:
        return float(TICK_WEDGE_RATE_WINDOW_S)
    try:
        parsed = float(raw) * 60.0
    except ValueError:
        return float(TICK_WEDGE_RATE_WINDOW_S)
    if parsed <= 0:
        return float(TICK_WEDGE_RATE_WINDOW_S)
    return parsed


# #1209: silence threshold for the failed-turn-silence trigger — a session
# whose EVERY completed tail turn FAILED and whose transcript has then been
# silent this long is dead-wedged (a turn-1 refusal death never arms the
# tick cron, so no wake will ever grow the other four triggers' counters;
# incident 8e9c371d / #1092: 1 failed turn at 02:54:29Z, next respawn
# 04:33Z, ~100 min unattended). 20 min = 2 watcher ticks (a mid-turn
# 429-retry gap is seconds-minutes, never 20 min of zero rows) and clears
# the 15-min RESPAWN_SPAWN_GRACE_S window by construction (the death
# follows the spawn by >= ~1 min). Deliberately NOT >= 45 min (the
# tick-cron cadence): for a zero-successful-turns session a fresh-context
# respawn beats waking a refusal-poisoned 1-turn conversation, so
# preempting a possible next wake is acceptable.
TICK_WEDGE_DEAD_SILENCE_S = 20 * 60


def _tick_wedge_dead_silence_s() -> float:
    """Dead-wake silence threshold in seconds. Env
    ``EPM_TICK_WEDGE_DEAD_SILENCE_MIN`` is in MINUTES (the
    ``EPM_TICK_WEDGE_RATE_WINDOW_MIN`` minutes-env -> seconds-constant
    precedent); default :data:`TICK_WEDGE_DEAD_SILENCE_S`. ``0`` DISABLES
    the trigger (the :func:`_tick_wedge_min_api_errors` new-trigger-class
    kill-switch convention — here the window IS the trigger arm, so 0 must
    mean off, not default); malformed / NEGATIVE env falls back to the
    default."""
    raw = os.environ.get("EPM_TICK_WEDGE_DEAD_SILENCE_MIN")
    if not raw:
        return float(TICK_WEDGE_DEAD_SILENCE_S)
    try:
        parsed = float(raw) * 60.0
    except ValueError:
        return float(TICK_WEDGE_DEAD_SILENCE_S)
    if parsed < 0:
        return float(TICK_WEDGE_DEAD_SILENCE_S)
    return parsed


# #1209: per-issue per-UTC-day cap on fence episodes the failed-turn-silence
# trigger may INITIATE. This trigger is the first fast enough to loop
# through the advancement-clear (each die-on-turn-1 generation writes one
# boot self-report, resetting the STALLED_MAX_RESPAWNS episode counter), so
# an episode-scoped cap alone cannot bound a cross-generation die-on-boot
# loop (~45-60 min/cycle ≈ up to ~24-32 cold ~100K-token session loads/day
# uncapped). 3 chosen over the orphan pass's 2 because each dead-silence
# cycle is cheap (no pod) and the first respawn is usually the recovery.
TICK_WEDGE_DEAD_RESPAWNS_PER_DAY = 3


def _tick_wedge_dead_respawns_per_day() -> int:
    """Daily per-issue cap on failed-turn-silence-initiated fence episodes
    (env ``EPM_TICK_WEDGE_DEAD_RESPAWNS_PER_DAY``; default
    :data:`TICK_WEDGE_DEAD_RESPAWNS_PER_DAY`). Malformed OR ``< 1`` env
    falls back to the default — the ``< 1 -> default`` guard is THIS
    helper's own addition (do NOT literal-copy
    :func:`_orphan_max_respawns_per_day`, which guards only malformed):
    disabling the trigger is the SILENCE knob's job
    (``EPM_TICK_WEDGE_DEAD_SILENCE_MIN=0``), never the cap's."""
    raw = os.environ.get("EPM_TICK_WEDGE_DEAD_RESPAWNS_PER_DAY")
    if not raw:
        return TICK_WEDGE_DEAD_RESPAWNS_PER_DAY
    try:
        parsed = int(raw)
    except ValueError:
        return TICK_WEDGE_DEAD_RESPAWNS_PER_DAY
    if parsed < 1:
        return TICK_WEDGE_DEAD_RESPAWNS_PER_DAY
    return parsed


# #1241: per-issue per-UTC-day cap on fence episodes the FOUR pre-#1209 wedge
# triggers (dequeue-run #779 / api-error-run #1104 / failed-turn-run +
# failed-turn-rate #1127) may INITIATE. Same rationale as the #1209 cap
# above: the crash-recovery arm — which consults no respawn cap — can
# complete a fresh-self-report wedge respawn, and the advancement-clear
# resets `respawn_count` every generation, so only a day-keyed
# advancement-clear-EXEMPT counter bounds a cross-generation wedge-respawn
# loop. ONE shared counter for the four triggers (shared failure surface —
# a live wedged session; one respawn resolves whichever fired), INDEPENDENT
# of the #1209 dead-silence budget so neither starves the other. Default
# mirrors the validated #1209 value (>= 3x every observed per-incident need
# — #779 / #1074 / #1098 / #1090 each needed 1 respawn).
TICK_WEDGE_RESPAWNS_PER_DAY = 3


def _tick_wedge_respawns_per_day() -> int:
    """Daily per-issue cap on fence episodes initiated by the four pre-#1209
    wedge triggers (dequeue-run / api-error-run / failed-turn-run /
    failed-turn-rate; env ``EPM_TICK_WEDGE_RESPAWNS_PER_DAY``; default
    :data:`TICK_WEDGE_RESPAWNS_PER_DAY`). Malformed OR ``< 1`` env falls
    back to the default — disabling a trigger is its own knob's job
    (``EPM_TICK_WEDGE_MIN_*=0``), never the cap's (#1241; mirrors
    :func:`_tick_wedge_dead_respawns_per_day`)."""
    raw = os.environ.get("EPM_TICK_WEDGE_RESPAWNS_PER_DAY")
    if not raw:
        return TICK_WEDGE_RESPAWNS_PER_DAY
    try:
        parsed = int(raw)
    except ValueError:
        return TICK_WEDGE_RESPAWNS_PER_DAY
    if parsed < 1:
        return TICK_WEDGE_RESPAWNS_PER_DAY
    return parsed


# Freshness window for the long-phase-heartbeat exemption. Generous so an
# emitter that heartbeats roughly hourly (the off-pod analyzer / Batch-poll
# cadence) is ALWAYS inside the window with margin for cron jitter — same
# logic as STALLED_WINDOW_S's "cadence + slack" rationale, one rung wider:
# 90 min = a ~60-min heartbeat cadence + a full extra 30-min slack, so an
# emitter must miss MORE than one heartbeat before the exemption lapses and
# the underlying staleness signals reassert. Env-tunable via
# EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN (minutes); a malformed value falls back
# to the default — a typo'd var must not disable the exemption.
LONG_PHASE_HEARTBEAT_FRESH_S_DEFAULT = 90 * 60


def _long_phase_heartbeat_freshness_s() -> float:
    """Long-phase-heartbeat exemption freshness window in seconds (env
    ``EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN``, minutes; default
    :data:`LONG_PHASE_HEARTBEAT_FRESH_S_DEFAULT`). A malformed env value
    falls back to the default — mirrors :func:`_stalled_window_s`."""
    raw = os.environ.get("EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN")
    if not raw:
        return float(LONG_PHASE_HEARTBEAT_FRESH_S_DEFAULT)
    try:
        return float(raw) * 60.0
    except ValueError:
        return float(LONG_PHASE_HEARTBEAT_FRESH_S_DEFAULT)


LONG_PHASE_HEARTBEAT_FRESH_S = _long_phase_heartbeat_freshness_s()

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
    marker_window_s: float = STALLED_MARKER_WINDOW_S_DEFAULT,
    max_respawns: int = STALLED_MAX_RESPAWNS,
) -> tuple[str, int]:
    """Pure decision for the alive-but-stalled detector.

    #845 (a-i): signal 2 (marker progress) has its OWN freshness window,
    ``marker_window_s`` (default 2h) — decoupled from the 60-min self-report
    ``window_s``. A session that posted ANY non-watcher marker within the
    marker window is never declared stalled by this function (the (e)
    prompt-wedge fast lane, which carries direct transcript evidence, is
    applied by the caller AFTER this decision and may override a keep).

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
    # chain is still posting; the self-report might just be late. The marker
    # gets its OWN (wider, 2h default) window — #845 (a-i); legitimate
    # 60-120 min marker gaps (#761) must not corroborate a stall.
    marker_stale = marker_progress_age_s is None or marker_progress_age_s >= marker_window_s
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


def decide_respawn_fence(
    *,
    stop_pending_sid: str | None,
    current_sid: str | None,
    sid_alive: bool,
    stop_retried: bool,
) -> str:
    """Pure stop-verify fence for the stalled respawn arm (#845 a-ii).

    The zombie-wrapper / idle-unmapped reapers already treat the daemon's
    stop ACK as NOT a kill (stop -> verify-dead-on-the-next-tick -> ONE
    retry -> one-time loud alert). This ports that contract into the
    stalled respawn arm: the arm NEVER spawns in the same tick it stops a
    session — it spawns only after the sid is verified absent from the
    daemon's live set on a LATER tick (incident #763: a respawn keyed on a
    stale self-report while the old session was ALIVE and polling left two
    drivers overlapped ~4h).

    Returns one of:

    - ``"clear-keep"`` — ``stop_pending_sid`` is set but no longer matches
      the registry entry's ``current_sid``: a CONCURRENT respawn (the crash
      arm, which runs before this pass against a once-per-tick ``live_ids``
      snapshot, or a #843-leased driver) replaced the session inside the
      stop->verify gap. CLEAR all fence state, do nothing this tick, and
      NEVER stop the fresh sid.
    - ``"stop"`` — no pending stop: first tick of the fence episode. Stop
      only; no spawn.
    - ``"spawn"`` — pending sid matches and is verified dead (absent from
      the live set): safe to spawn.
    - ``"retry-stop"`` — pending sid matches but is STILL alive and the one
      allowed retry has not been used yet.
    - ``"stop-failed"`` — still alive after the retry: loud one-time alert,
      never spawn.
    """
    if stop_pending_sid is not None and stop_pending_sid != current_sid:
        return "clear-keep"
    if stop_pending_sid is None:
        return "stop"
    if not sid_alive:
        return "spawn"
    if not stop_retried:
        return "retry-stop"
    return "stop-failed"


def decide_worktree_hold(
    activity_fresh: bool, hold_count: int, max_holds: int = WT_HOLD_MAX_TICKS
) -> bool:
    """Pure bounded worktree-activity hold (#845 b): defer a watcher respawn
    while the issue's worktree shows fresh file activity (an implementer is
    mid-edit — incident #812's kill landed 57s after an edit), but only up to
    ``max_holds`` consecutive ticks so a cross-writer can never turn the hold
    into a permanent false negative."""
    return activity_fresh and hold_count < max_holds


def decide_daemon_blocked_escalation(
    *,
    in_active: bool,
    manual: bool,
    alerted: bool,
    stale: bool,
    daemon_reachable: bool,
    blocked_ticks: int,
    already_pushed: bool,
    threshold: int = 2,
) -> tuple[int, bool]:
    """Pure escalation counter for a respawn-worthy stall deferred by a
    daemon outage (#845 c; incident #811: the daemon was unreachable at
    alert time, so the stalled session's GPU idled until manual recovery
    hours later — the deferral was silent).

    Returns ``(new_blocked_ticks, fire_push)``. Increments once per tick
    while an alerted, ACTIVE, non-manual, still-stale episode is deferred by
    ``not daemon_reachable``; ``fire_push`` is True exactly once, when the
    count reaches ``threshold`` (2 ticks ~= 20 min at the 10-min cron) and
    no push has fired this episode. Resets to ``(0, False)`` the moment the
    daemon is reachable (the existing alerted->eligible escalation then
    respawns on that same tick); the caller additionally resets on
    self-report advancement (episode over)."""
    if daemon_reachable:
        return (0, False)
    if manual or not in_active or not alerted or not stale:
        return (blocked_ticks, False)
    new_ticks = blocked_ticks + 1
    return (new_ticks, new_ticks >= threshold and not already_pushed)


def _auth_outage_freeze_subset(
    events: list[dict],
    now: float,
    *,
    window_s: float,
    fresh_death_s: float,
    last_episode_end_ts: float,
) -> list[dict]:
    """PURE: the qualifying instant-freeze spawn events for the auth-outage
    trigger (#1027) — well-formed, inside the rolling ``window_s``, strictly
    NEWER than the last episode end (MF-1 watermark: both the event ``ts``
    AND its ``prev_spawned_at`` must postdate the watermark, so pre-resolve
    churn and backlog respawns of episode-era predecessors never re-trigger),
    with ``0 <= ts - prev_spawned_at <= fresh_death_s`` (the ``0 <=`` guard
    excludes clock skew / a future ``prev_spawned_at``). Events with a
    missing/None ``prev_spawned_at`` (infra-drain / capacity-retry / most
    orphan first spawns) never qualify — an accepted residual (new-spawn-only
    outages are bounded by their per-day caps)."""
    out: list[dict] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        issue = e.get("issue")
        ts = e.get("ts")
        prev = e.get("prev_spawned_at")
        if not isinstance(issue, int) or isinstance(issue, bool):
            continue  # a None/malformed issue never counts toward distinct issues
        if not isinstance(ts, int | float) or now - ts > window_s:
            continue
        if ts <= last_episode_end_ts:
            continue
        if not isinstance(prev, int | float) or prev <= last_episode_end_ts:
            continue
        if not 0 <= ts - prev <= fresh_death_s:
            continue
        out.append(e)
    return out


def decide_auth_outage_trigger(
    events: list[dict],
    now: float,
    *,
    window_s: float,
    fresh_death_s: float,
    min_freeze_events: int,
    min_distinct_issues: int,
    last_episode_end_ts: float = 0.0,
) -> bool:
    """Pure fleet-level auth-outage trigger (#1027; 2026-07-03 incident: a
    poisoned Claude CLI credential killed every freshly spawned session on
    arrival and the watcher churned respawns for hours).

    True iff >= ``min_freeze_events`` instant-freeze respawn events (the
    predecessor session lived <= ``fresh_death_s``) across >=
    ``min_distinct_issues`` DISTINCT issues fall inside the rolling
    ``window_s`` — cross-issue correlation is the false-positive guard: a
    single issue insta-crashing repeatedly is an issue-specific bug already
    bounded by the per-task caps. ``last_episode_end_ts`` is the MF-1
    watermark (see :func:`_auth_outage_freeze_subset`)."""
    freeze = _auth_outage_freeze_subset(
        events,
        now,
        window_s=window_s,
        fresh_death_s=fresh_death_s,
        last_episode_end_ts=last_episode_end_ts,
    )
    return (
        len(freeze) >= min_freeze_events
        and len({e["issue"] for e in freeze}) >= min_distinct_issues
    )


def decide_auth_outage_canary(
    state: dict,
    now: float,
    *,
    canary_alive: bool | None,
    canary_interval_s: float,
    canary_survival_s: float,
    max_episode_s: float,
) -> str:
    """Pure per-tick decision for an ACTIVE auth-outage episode (#1027).

    Returns one of ``"expire" | "resolve" | "canary-failed" | "arm-canary" |
    "hold"``, in priority order:

    - ``"expire"``: episode older than ``max_episode_s`` (or a garbled
      ``started_ts``) — FAIL-OPEN: a wedged guard can never disable crash
      recovery indefinitely.
    - ``"resolve"``: the outstanding canary is alive AND has survived >=
      ``canary_survival_s`` — the auth path works again.
    - ``"canary-failed"``: the outstanding canary is confirmed dead
      (``canary_alive is False``) — the failure re-evidences the episode;
      the caller clears the canary fields and re-arms one interval later.
    - ``"hold"``: canary outstanding but alive-and-young, or its liveness is
      inconclusive (``canary_alive is None`` — daemon down); ALSO held while
      a consumed-but-not-yet-spawned canary is in flight (a fresh
      ``canary_pending``, e.g. the stalled fence's stop tick consumed the
      token and its verified-dead spawn lands a tick later).
    - ``"arm-canary"``: no canary outstanding and >= ``canary_interval_s``
      since ``max(started_ts, last_canary_ts)`` — allow ONE probe respawn.
    """
    started = state.get("started_ts")
    if not isinstance(started, int | float) or now - started >= max_episode_s:
        return "expire"
    canary_ts = state.get("canary_ts")
    if isinstance(canary_ts, int | float):
        if canary_alive is True and now - canary_ts >= canary_survival_s:
            return "resolve"
        if canary_alive is False:
            return "canary-failed"
        return "hold"
    pending = state.get("canary_pending")
    if isinstance(pending, dict):
        pts = pending.get("ts")
        if isinstance(pts, int | float) and 0 <= now - pts <= canary_interval_s:
            return "hold"  # a canary spawn is in flight across ticks
    last = state.get("last_canary_ts")
    anchor = max(started, last) if isinstance(last, int | float) else started
    if now - anchor >= canary_interval_s:
        return "arm-canary"
    return "hold"


def _classify_wedge_row(row: object) -> str:
    """Classify one parsed transcript row for :func:`decide_prompt_wedge`:
    ``"dequeue"`` | ``"prompt"`` | ``"assistant"`` | ``"api-error"`` |
    ``"other"``.

    - ``"dequeue"`` (CO-PRIMARY evidence): ``type == "queue-operation"``
      with ``operation == "dequeue"`` — the verified per-prompt dequeue
      record Claude Code writes when it pulls a queued prompt (row shape
      verified in live session transcripts; dequeue rows carry no content).
    - ``"prompt"`` (SECONDARY evidence): a ``type == "user"`` row whose
      message content is a plain string, or contains a text block and NO
      tool_result block — i.e. a delivered user/tick prompt, not a tool
      result re-entering the conversation.
    - ``"assistant"``: a ``type == "assistant"`` row WITHOUT the top-level
      ``isApiErrorMessage: true`` flag — the session took a real turn,
      which resets the wedge-evidence run.
    - ``"api-error"`` (#1104): an assistant-type row Claude Code writes for
      an API-ERRORED turn (usage-policy refusal, 429/529) with top-level
      ``isApiErrorMessage: true`` — the turn FAILED, so it is wedge
      evidence, not a reset (row shape verified live on the #1074 incident
      transcript, session 6f682c18: 38 such rows, all ``type: "assistant"``
      with the flag at the row's top level).
    - ``"other"``: everything else (tool_result user rows, summary/system
      rows, non-dequeue queue-operations, malformed rows) — skipped without
      resetting the run.
    """
    if not isinstance(row, dict):
        return "other"
    rtype = row.get("type")
    if rtype == "assistant":
        return "api-error" if row.get("isApiErrorMessage") is True else "assistant"
    if rtype == "queue-operation":
        return "dequeue" if row.get("operation") == "dequeue" else "other"
    if rtype != "user":
        return "other"
    msg = row.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str):
        return "prompt"
    if isinstance(content, list):
        has_text = any(isinstance(b, dict) and b.get("type") == "text" for b in content)
        has_tool_result = any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
        if has_text and not has_tool_result:
            return "prompt"
    return "other"


def _row_ts(row: object) -> float | None:
    """Parse a transcript row's top-level ISO-8601 ``timestamp`` (Claude Code
    writes ``...Z``-suffixed UTC — verified on the live-captured
    ``_wedge_dequeue_row`` / ``_wedge_api_error_row`` fixtures) to an epoch
    float. Returns ``None`` on a missing / non-string / unparseable value —
    fail toward NO-FIRE: the #1127 rate trigger EXCLUDES ts-less turns from
    its windowed count rather than guessing."""
    if not isinstance(row, dict):
        return None
    raw = row.get("timestamp")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _segment_wake_turns(rows: list[dict]) -> list[tuple[str, float | None]]:
    """Segment classified transcript-tail rows into COMPLETED wake-turns
    (#1127), oldest -> newest. Returns one ``(outcome, end_ts)`` tuple per
    completed turn, ``outcome in {"ok", "failed"}``.

    Rules (consumes :func:`_classify_wedge_row`'s classes only):

    - A turn starts at a prompt-evidence row (``"dequeue"`` / ``"prompt"``)
      and collects the response rows (``"assistant"`` / ``"api-error"``)
      that follow; consecutive prompt-evidence rows (possibly interleaved
      with ``"other"``) are ONE delivery burst, not separate turns.
    - A turn is COMPLETED iff it has >= 1 response row; its outcome is the
      class of its LAST response row: ``assistant`` -> ``"ok"``,
      ``api-error`` -> ``"failed"``. This is the #1127 Goal's "a turn ENDING
      in isApiErrorMessage is a failed wake even with mid-turn heartbeats";
      a mid-turn api-error followed by a successful assistant row in the
      SAME turn is ``"ok"`` (the healthy retried-429 shape).
    - A swallowed delivery (prompt evidence, no response rows, then the next
      delivery) produces NO turn — it neither resets nor increments the turn
      counters; the row-level ``run`` counter owns the #779 shape.
    - Leading response rows before any prompt evidence (a mid-turn tail cut)
      form one implicit turn.
    - An in-flight FINAL turn (prompt delivered, no response yet) is
      deliberately NOT counted — it can neither reset nor extend the
      trailing-failed count (fail toward NO-FIRE on a probe racing a
      mid-delivery wake).
    - ``end_ts`` = the parsed top-level ISO ``timestamp`` of the turn's last
      response row (:func:`_row_ts`; ``None`` if missing/unparseable).
    """
    turns: list[tuple[str, float | None]] = []
    last_resp: str | None = None  # class of latest response row in current turn
    last_resp_ts: float | None = None
    for row in rows:
        cls = _classify_wedge_row(row)
        if cls in ("dequeue", "prompt"):
            if last_resp is not None:  # previous turn completed -> flush
                turns.append(("failed" if last_resp == "api-error" else "ok", last_resp_ts))
                last_resp, last_resp_ts = None, None
            # else: same delivery burst, or a swallowed delivery (no turn emitted)
        elif cls in ("assistant", "api-error"):
            last_resp, last_resp_ts = cls, _row_ts(row)
        # "other": skipped
    if last_resp is not None:  # flush the final completed turn
        turns.append(("failed" if last_resp == "api-error" else "ok", last_resp_ts))
    return turns


def _wedge_newest_row_ts(trailing_rows: list[dict]) -> float | None:
    """Newest parseable top-level row timestamp in the tail, or ``None``
    when no row timestamp parses at all (the window/silence anchor is then
    undefined — fail toward NO-FIRE). Extracted from
    :func:`_wedge_rate_windowed_failed` (#1209) so the ``failed-turn-rate``
    window and the ``failed-turn-silence`` read share the IDENTICAL
    deterministic anchor scan (never wall-clock)."""
    for row in reversed(trailing_rows):
        ts = _row_ts(row)
        if ts is not None:
            return ts
    return None


def decide_prompt_wedge_reason(
    trailing_rows: list[dict],
    min_dequeued: int,
    *,
    min_api_errors: int = TICK_WEDGE_MIN_API_ERRORS,
    min_failed_turns: int = TICK_WEDGE_MIN_FAILED_TURNS,
    min_failed_total: int = TICK_WEDGE_MIN_FAILED_TOTAL,
    rate_window_s: float = TICK_WEDGE_RATE_WINDOW_S,
    dead_silence_s: float = TICK_WEDGE_DEAD_SILENCE_S,
    now: float | None = None,
) -> str | None:
    """Pure prompt-wedge detector over parsed transcript-tail rows —
    returns WHICH trigger fired (``"dequeue-run"`` | ``"api-error-run"`` |
    ``"failed-turn-run"`` | ``"failed-turn-rate"`` |
    ``"failed-turn-silence"``, checked in that precedence order:
    oldest/most-specific evidence first; the order never changes WHETHER
    the wedge fires) or ``None``.

    Five triggers:

    1. ``dequeue-run`` (#845 e, verbatim): the tail ends with >=
       ``min_dequeued`` consecutive wedge-evidence rows (``"dequeue"``
       queue-operation records — co-primary — and/or ``"prompt"`` promptless
       user rows — secondary; both count toward the SAME trailing run) with
       no assistant row after the first of them (incident #779: 5 tick
       prompts enqueued AND dequeued with no turn for ~90 min).
       ``min_dequeued <= 0`` DISABLES this counter (#1127 — new semantics,
       needed by the fresh-self-report gate path in
       :func:`_apply_prompt_wedge_override`; the production STALE path is
       unaffected — :func:`_tick_wedge_min_dequeued` never returns < 1).
    2. ``api-error-run`` (#1104, verbatim): the tail ends with >=
       ``min_api_errors`` consecutive ``"api-error"`` rows (assistant rows
       with ``isApiErrorMessage: true`` — usage-policy refusals, 429/529
       error turns) with no SUCCESSFUL assistant row after the first of them
       (incident #1074: 38 refused wake turns / ~2h unrecovered).
       ``min_api_errors <= 0`` DISABLES (the existing kill switch).
    3. ``failed-turn-run`` (#1127 — the #1098/#1090 fix): >=
       ``min_failed_turns`` trailing consecutive ``"failed"`` turns from
       :func:`_segment_wake_turns`; an ``"ok"`` turn resets. The row-level
       counters are structurally blind to the partially-successful wake
       (every refused wake posts >= 1 assistant row before dying, resetting
       both row counters each cycle); turn granularity is not.
       ``min_failed_turns <= 0`` DISABLES.
    4. ``failed-turn-rate`` (#1127 option (c) — the c16b10ca alternating
       storm): total (non-consecutive) ``"failed"`` turns whose ``end_ts``
       lies within ``rate_window_s`` of the NEWEST parseable row timestamp
       in the tail >= ``min_failed_total``, AND the newest completed turn is
       ``"failed"`` (a session that recovered after a storm is not respawned
       mid-recovery). Turns with ``end_ts=None`` are EXCLUDED from the
       windowed count; if no row timestamp parses at all the anchor is
       undefined and this trigger is inert (fail toward NO-FIRE).
       ``min_failed_total <= 0`` DISABLES.
    5. ``failed-turn-silence`` (#1209 — the die-on-turn-1 gap): >= 1
       completed turn, EVERY completed turn in the tail is ``"failed"``,
       AND the newest parseable row timestamp is >= ``dead_silence_s``
       older than ``now`` (incident 8e9c371d / #1092: a session refused on
       its FIRST substantive turn never arms its tick cron, freezes at
       exactly 1 failed turn / 1 api-error row — permanently below every
       counting threshold above — and no further rows ever arrive). The
       WEAKEST evidence, checked LAST. ``now=None`` (every pre-existing
       pure caller) OR ``dead_silence_s <= 0`` (the kill switch) leaves
       this trigger inert; zero completed turns (the #779 swallow shape),
       a prior ``"ok"`` turn anywhere in the tail, a ts-less tail, and a
       future-dated anchor all fail toward NO-FIRE
       (:func:`_wedge_dead_silence_fired`).

    Row-counter mechanics (unchanged from #1104): ``"other"`` rows are
    skipped without resetting either counter; a real ``"assistant"`` row
    resets both; an ``"api-error"`` row RESETS ``run`` (the prompt DID get a
    response — not the #779 swallow shape; the single-refusal guard) but
    increments ``api_run``. Window assumption: the caller's transcript-tail
    read must span the thresholds' worth of evidence (the production probe
    reads 256 KB — #1104); a too-thin window fails toward NO-FIRE."""
    run, api_run = _wedge_trailing_row_runs(trailing_rows)
    if min_dequeued > 0 and run >= min_dequeued:
        return "dequeue-run"
    if min_api_errors > 0 and api_run >= min_api_errors:
        return "api-error-run"
    return _wedge_turn_lane_reason(
        trailing_rows,
        min_failed_turns=min_failed_turns,
        min_failed_total=min_failed_total,
        rate_window_s=rate_window_s,
        dead_silence_s=dead_silence_s,
        now=now,
    )


def _wedge_turn_lane_reason(
    trailing_rows: list[dict],
    *,
    min_failed_turns: int,
    min_failed_total: int,
    rate_window_s: float,
    dead_silence_s: float,
    now: float | None,
) -> str | None:
    """The TURN-level wedge triggers (#1127 ``failed-turn-run`` /
    ``failed-turn-rate``; #1209 ``failed-turn-silence``), factored out of
    :func:`decide_prompt_wedge_reason` (ruff C901 cap — the
    :func:`_wedge_trailing_row_runs` precedent). Behavior-preserving
    extraction: precedence order and every knob's semantics are exactly the
    inline #1127 code's; the early exit keeps the hot path free of
    :func:`_segment_wake_turns` when every turn-level lane is disabled
    (``now=None`` counts the #1209 lane as disabled, so all pre-existing
    call shapes take the identical path)."""
    silence_armed = now is not None and dead_silence_s > 0
    if min_failed_turns <= 0 and min_failed_total <= 0 and not silence_armed:
        return None
    turns = _segment_wake_turns(trailing_rows)
    if min_failed_turns > 0:
        trailing_failed = 0
        for outcome, _ts in reversed(turns):
            if outcome != "failed":
                break
            trailing_failed += 1
        if trailing_failed >= min_failed_turns:
            return "failed-turn-run"
    if min_failed_total > 0:
        windowed = _wedge_rate_windowed_failed(trailing_rows, turns, rate_window_s)
        if windowed is not None and windowed >= min_failed_total:
            return "failed-turn-rate"
    if silence_armed and _wedge_dead_silence_fired(trailing_rows, turns, dead_silence_s, now):
        return "failed-turn-silence"
    return None


def _wedge_dead_silence_fired(
    trailing_rows: list[dict],
    turns: list[tuple[str, float | None]],
    dead_silence_s: float,
    now: float,
) -> bool:
    """#1209 ``failed-turn-silence``: True iff the tail holds >= 1 completed
    turn, EVERY completed turn is ``"failed"``, and the newest parseable row
    timestamp is >= ``dead_silence_s`` older than ``now``. Zero completed
    turns (the #779 swallow shape) never fire (vacuous-all guard — the
    dequeue-run trigger owns swallows); a ts-less tail leaves the silence
    anchor undefined -> NO-FIRE; a future-dated anchor (clock jump) makes
    ``now - anchor`` negative -> NO-FIRE. The all-completed-turns-failed
    condition confines the trigger to sessions with ZERO successful history
    in the visible tail — the died-young shape, for which a fresh-context
    respawn loses essentially nothing (CLAUDE.md refusal-ladder (f)); a
    healthy mid-life session whose latest wake ended in one transient
    trailing api-error has prior ``"ok"`` turns in its tail and is blocked
    regardless of silence. Accepted residual (#1209 plan §4.2, pinned by
    test): the 256 KB tail window can truncate older ``ok`` turns of a very
    long final turn, making the all-failed read vacuously true over the
    visible window (incl. the leading-IMPLICIT-turn shape) — the session's
    last visible turn genuinely failed and has been silent >= the window,
    so the bounded fresh respawn is the accepted recovery."""
    if not turns or any(outcome != "failed" for outcome, _ts in turns):
        return False
    anchor = _wedge_newest_row_ts(trailing_rows)
    if anchor is None:
        return False
    return now - anchor >= dead_silence_s


def _wedge_trailing_row_runs(trailing_rows: list[dict]) -> tuple[int, int]:
    """The #845/#1104 ROW-level trailing counters ``(run, api_run)`` —
    factored out of :func:`decide_prompt_wedge_reason` (ruff C901 cap).
    ``run`` = trailing consecutive dequeue/prompt rows (reset by ANY
    response row); ``api_run`` = trailing consecutive api-error rows (reset
    only by a SUCCESSFUL assistant row); ``"other"`` rows never reset."""
    run = 0
    api_run = 0
    for row in trailing_rows:
        cls = _classify_wedge_row(row)
        if cls == "assistant":
            run = 0
            api_run = 0
        elif cls == "api-error":
            run = 0
            api_run += 1
        elif cls in ("dequeue", "prompt"):
            run += 1
    return run, api_run


def _wedge_rate_windowed_failed(
    trailing_rows: list[dict],
    turns: list[tuple[str, float | None]],
    rate_window_s: float,
) -> int | None:
    """Windowed failed-turn count for the #1127 ``failed-turn-rate`` trigger,
    or ``None`` when the trigger is inert for this tail (no completed turns;
    the newest completed turn is not ``"failed"`` — a recovered session is
    not respawned mid-recovery; or no row timestamp parses at all, leaving
    the window anchor undefined — fail toward NO-FIRE). The anchor is the
    NEWEST parseable row timestamp in the tail (deterministic, replay-
    testable — never wall-clock); ts-less turns are excluded from the count."""
    if not turns or turns[-1][0] != "failed":
        return None
    anchor = _wedge_newest_row_ts(trailing_rows)
    if anchor is None:
        return None
    return sum(
        1
        for outcome, ts in turns
        if outcome == "failed" and ts is not None and anchor - ts <= rate_window_s
    )


def decide_prompt_wedge(
    trailing_rows: list[dict],
    min_dequeued: int,
    *,
    min_api_errors: int = TICK_WEDGE_MIN_API_ERRORS,
    min_failed_turns: int = TICK_WEDGE_MIN_FAILED_TURNS,
    min_failed_total: int = TICK_WEDGE_MIN_FAILED_TOTAL,
    rate_window_s: float = TICK_WEDGE_RATE_WINDOW_S,
    dead_silence_s: float = TICK_WEDGE_DEAD_SILENCE_S,
    now: float | None = None,
) -> bool:
    """Thin bool wrapper over :func:`decide_prompt_wedge_reason` (#845 e /
    #1104 / #1127 / #1209): True iff ANY of the five wedge triggers fires.
    Existing positional/keyword call shapes are preserved; the new keyword
    defaults are the module constants, so ``decide_prompt_wedge(tail, 3)``
    gains the #1127 turn-level triggers too (pure-function callers want the
    fixed predicate) while the #1209 dead-silence trigger stays INERT
    without an explicit ``now`` (never wall-clock inside the predicate —
    the #1127 deterministic-anchor posture). See
    :func:`decide_prompt_wedge_reason` for the trigger definitions, kill
    switches, and incident history."""
    return (
        decide_prompt_wedge_reason(
            trailing_rows,
            min_dequeued,
            min_api_errors=min_api_errors,
            min_failed_turns=min_failed_turns,
            min_failed_total=min_failed_total,
            rate_window_s=rate_window_s,
            dead_silence_s=dead_silence_s,
            now=now,
        )
        is not None
    )


def decide_stale_registration(
    *,
    sid_alive: bool,
    transcript_idle_s: float | None,
    self_report_age_s: float | None,
    idle_threshold_s: float,
) -> str:
    """Pure per-entry decision for the stale-registration pass (#845 d;
    incident #665: a 16h-transcript-idle registered session held the
    `/issue` Step 0 single-orchestrator guard and blocked every re-drive).

    Returns ``"unregister"`` iff the registration's session is LIVE
    (``sid_alive`` — a dead sid is the crash-recovery pass's property and
    stays registered so that pass can respawn it), its transcript has been
    idle >= ``idle_threshold_s``, AND the self-report is equally stale (a
    missing self-report — ``None`` — does not rescue: manual sessions never
    self-report, and the transcript idle IS the direct signal). Everything
    unresolvable (``transcript_idle_s is None``) fails toward ``"keep"``.
    """
    if not sid_alive:
        return "keep"
    if transcript_idle_s is None or transcript_idle_s < idle_threshold_s:
        return "keep"
    if self_report_age_s is not None and self_report_age_s < idle_threshold_s:
        return "keep"
    return "unregister"


def decide_boot_death(
    *,
    sid_alive: bool,
    entry_age_s: float | None,
    response_row_seen: bool | None,
    all_turns_failed: bool | None,
    transcript_idle_s: float | None,
    window_s: float,
    quiet_s: float,
    stops_today: int,
    stops_per_day: int,
) -> str:
    """Pure per-entry decision for the boot-death lane (#1267 arm 1 +
    #1287 arm 2). Returns ``"stop"`` | ``"cap-alert"`` | ``"keep"``.

    Fires ONLY on: live sid + registration older than ``window_s`` + ONE of

    - ARM 1 (zero-response, #1267/#1251): ``response_row_seen is False`` —
      the whole-file read found NO response rows (assistant OR api-error);
      the session died before turn 1.
    - ARM 2 (boot-refusal, #1287/#1277): ``all_turns_failed is True`` —
      the 256 KB TAIL read segmented (via :func:`_segment_wake_turns`,
      #1127) to >= 1 completed turn with EVERY completed turn failed (last
      response row api-error). The boot turn executed and every turn
      failed; at a PARK status the #1209/#1127 wedge lanes are structurally
      ineligible (``respawn_eligible = in_active and ...``), so this lane
      owns the shape. A single visible ok turn anywhere in the tail =>
      not boot-dead (the #1104 single-refusal guard).

    + transcript quiet >= ``quiet_s`` (the in-flight-first-turn / mid-retry
    guard: a live retrying session keeps appending rows, so mtime stays
    fresh). Silence-anchor divergence from #1209 (deliberate — #1267
    parity): arm 2 gates on the lane's mtime quiet (10 min), NOT #1209's
    newest-parseable-row-ts 20-min silence anchor; the two coincide for
    dead append-only transcripts (mtime tracks the last row), so do not
    "fix" it in either direction. Every unresolvable input fails toward
    keep (the 12h stale-registration pass stays the backstop):
    ``entry_age_s`` None or negative (missing / zero / non-numeric /
    FUTURE-dated ``spawned_at``); ``response_row_seen`` None (whole-file
    read unresolvable OR over the cap — which does NOT veto arm 2: the
    tail read works at any size); ``all_turns_failed`` None (tail
    unresolvable) or False (zero completed turns — swallows stay the
    dequeue-run's property — or an ok turn); ``transcript_idle_s`` None
    (:func:`_transcript_idle_age_s` could not resolve the mtime).

    At the daily stop cap the lane stops STOPPING and returns
    ``"cap-alert"`` (the caller dedupes to one loud alert per UTC day);
    the live dead registration then back-pressures re-dispatch exactly as
    today's 12h cycle does.
    """
    if not sid_alive:
        return "keep"  # dead sid: crash-recovery / sweep-grace property
    if entry_age_s is None or entry_age_s < window_s:
        return "keep"
    boot_dead = response_row_seen is False  # arm 1 (#1267)
    boot_refused = all_turns_failed is True  # arm 2 (#1287)
    if not (boot_dead or boot_refused):
        return "keep"
    if transcript_idle_s is None or transcript_idle_s < quiet_s:
        return "keep"
    if stops_today >= stops_per_day:
        return "cap-alert"
    return "stop"


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
        finished), or ``on_hold`` (:data:`AUTO_STOP_PAUSED`, the #919
        pause-window escaped pod — the pause affordance stops pods BEFORE
        parking, so on_hold + RUNNING means the teardown leg failed; #980);
        ``"pod-active-stale"`` — task in :data:`POD_ACTIVE` AND no
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

    Cases (``"auto-stop-done"`` = task in :data:`POD_SAFETY_AUTO_STOP` —
    DONE, or user-paused ``on_hold``):

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


def decide_pod_wedge(
    *,
    wedged_for: float,
    k_floor: float,
    wedge_missed: int,
    threshold: int,
    alerted: bool,
    keep_running: bool | str,
    inputs_ok: bool,
) -> tuple[str, int]:
    """Pure decision for a RUNNING managed pod in the #664 RunPod no-port wedge.

    The watcher backstop for when the poller's
    ``backend_poll._maybe_escalate_runpod_wedge`` never ran (the per-issue poll
    loop died). Returns ``(action, new_wedge_missed)`` where action is
    ``"terminate-failover"`` | ``"alert"`` | ``"keep"`` (#770: the strongest
    action was promoted from the reversible ``"stop"`` to
    ``"terminate-failover"`` — see the Decision invariant below). The caller
    (``_process_wedged_pod``) has already confirmed the RAW wedge condition
    (``backend_poll._pod_is_runpod_runtime_wedged``) and excluded DONE-status
    pods (MF6 — a DONE-task wedged pod falls through to the status-class arm).

    Parameters
    ----------
    wedged_for
        Seconds the pod has been in the raw no-port wedge, measured against the
        DEDICATED ``wedge_first_seen`` clock (stamped at wedge ONSET, NOT the
        pod-incarnation ``first_seen`` — MF1), so it is the actual no-port
        episode length, not pod uptime.
    k_floor
        ``backend_poll.RUNPOD_WEDGE_K_SEC`` (imported, never a duplicated
        literal). The maturity floor the wedge must exceed before any action.
    wedge_missed
        The wedge arm's consecutive-confirmed-past-K miss count (SEPARATE from
        the status-class ``missed``).
    threshold
        Consecutive-confirmed-checks required before an irreversible action
        (the ``terminate-failover`` recovery; default 2 = ~20 min at the 10-min
        cron, so a single transient API mis-read never acts on a pod) — the same
        miss guard the status-class auto-stop uses.
    alerted
        Whether the once-per-wedge-episode alert has already been posted
        (tracked as ``wedge_alerted`` in the state file). Informational here;
        the dedup decision is the CALLER's (it posts the marker only if not
        already alerted), so this fn returns ``"alert"`` whenever the gated
        condition holds and lets the caller dedup.
    keep_running
        TRI-STATE (MF2): ``True`` (the ``keep-running`` tag is present) |
        ``False`` (the tag was read OK and is absent) | ``"unknown"`` (the tag
        read FAILED — subprocess error, non-zero rc, JSON parse error). A STOP
        fires ONLY on the literal ``False``; ``"unknown"`` routes to ALERT-only
        so a tagged live-work pod whose tag lookup is transiently failing is
        NEVER auto-stopped (which would silently override the user's tag).
    inputs_ok
        Whether the wedged run's recoverable inputs are verified on HF (the same
        gate #689 fix (b) uses, via ``backend_poll._wedged_run_inputs_on_hf``).
        A STOP fires only when inputs are PROVABLY safe.

    Cases:

    - ``wedged_for <= k_floor`` -> ``("keep", 0)``. Below the K maturity floor
      the wedge has not matured (a healthy slow bring-up clears it when the port
      appears); reset the miss counter so a brief no-port blip never accumulates
      toward a stop. The comparator is ``<=`` here (and ``>`` below) to MATCH the
      poller's ``wedged_for > RUNPOD_WEDGE_K_SEC`` at ``backend_poll.py``
      (``wedged_for == k_floor`` KEEPs — MF5 boundary parity).
    - ``wedged_for > k_floor`` AND ``wedge_missed + 1 < threshold`` ->
      ``("keep", wedge_missed + 1)``. Past K but not yet confirmed for
      ``>=threshold`` consecutive checks; accumulate. The action transitions
      exactly when ``wedge_missed + 1 == threshold`` (MF5 boundary).
    - ``wedged_for > k_floor`` AND confirmed (``wedge_missed + 1 >= threshold``):
        - ``keep_running is True`` -> ``("alert", 0)``. The keep-running tag
          exempts the AUTO-STOP exactly as it does for the status-class arm; the
          wedge is still surfaced once per episode so the leak is visible.
        - ``keep_running == "unknown"`` -> ``("alert", 0)`` (MF2). A PERSISTENT
          tag-read failure is NOT a genuinely untagged task; route the
          uncertainty to ALERT-only rather than silently override the user's tag.
          This is a STRONGER gate than the status-class DONE arm's
          False-on-failure (safe there only because it auto-stops DONE-status
          pods; the wedge arm auto-stops live-work pods).
        - ``keep_running is False`` AND ``inputs_ok`` ->
          ``("terminate-failover", 0)``. Inputs are verified on HF AND the tag
          was read AND it is absent, so the IRREVERSIBLE terminate +
          re-provision is safe — route to the existing poller recovery
          (``backend_poll._failover_wedged_runpod``: terminate the wedged pod,
          re-provision a FRESH pod, bounded-once via the durable lease +
          sentinel). A reversible ``pod.py stop`` cannot heal a host-pinned dead
          RunPod host — ``resume_pod`` returns to the SAME dead host (#763) —
          which is why the action was promoted from ``"stop"`` to
          ``"terminate-failover"`` (#770).
        - ``keep_running is False`` AND not ``inputs_ok`` -> ``("alert", 0)``.
          Inputs are NOT verified on HF (or no run handle to gate on); a
          terminate could strand un-uploaded work, so ALERT-only and let a
          human / the re-invoked /issue decide (CLAUDE.md halt-criterion #2).

    Decision invariant (MF2): ``("terminate-failover", _)`` is the ONLY
    irreversible action and is returned ONLY when ``keep_running is False`` (the
    literal boolean False, NOT ``"unknown"`` and NOT ``True``) AND
    ``inputs_ok is True``. Every other ``keep_running`` value (``True``,
    ``"unknown"``) and the ``not inputs_ok`` case route to ALERT-only. The
    action set is now ``"terminate-failover" | "alert" | "keep"`` — ``"stop"``
    is NO LONGER a value this fn returns (#770; the reversible ``_stop_pod`` is
    still used by the status-class DONE escaped-pod arm).
    """
    if wedged_for <= k_floor:
        # Below the maturity floor (or exactly at it — MF5 parity with the
        # poller's strict `> K`): not matured. Reset the miss counter.
        return ("keep", 0)
    new_wedge_missed = wedge_missed + 1
    if new_wedge_missed < threshold:
        # Past K but not yet confirmed for >=threshold consecutive checks.
        return ("keep", new_wedge_missed)
    # Confirmed past K for >=threshold consecutive checks. Gate the AUTO-STOP.
    if keep_running is True:
        return ("alert", 0)
    if keep_running == "unknown":
        return ("alert", 0)
    # keep_running is the literal False (tag read OK and absent).
    if inputs_ok:
        return ("terminate-failover", 0)
    return ("alert", 0)


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


# ─── VM disk SUB-FLOOR sentinel (task #679) ──────────────────────────────────
#
# The existing alert/reclaim bands above fire LATE (20 / 15 GiB free) — by then
# foreground Bash spawns are already at risk. The sub-floor sentinel is an
# EARLIER, advisory warn-only band (~60 GB free) whose job is attribution +
# faster re-check intent, NOT remediation: it writes a `band=sub-floor` row to
# the SHARED disk-guard sidecar naming the top per-issue cache paths (cheap
# `du -s` on the re-downloadable globs only) so a human can see WHICH active
# task is eating the disk before it hits the late bands. It NEVER deletes
# anything and NEVER spawns a daemon — the "re-check sooner" signal is purely
# the sidecar row it writes (the 10-min watcher cron cadence is unchanged).

# Below this free-bytes threshold the sub-floor sentinel writes an attributed
# advisory row. ~60 GB: well above the 20 GiB alert band, so it surfaces a
# growing cache while there is still ample slack. Override: EPM_VM_DISK_SUBFLOOR_GIB.
VM_DISK_SUBFLOOR_FREE_BYTES = _env_gib_bytes("EPM_VM_DISK_SUBFLOOR_GIB", 60)

# Re-alert an already-sub-floor episode only when free space DROPPED by more
# than this fraction since the last sub-floor row — bounds churn while still
# surfacing a steadily-tightening disk.
VM_DISK_SUBFLOOR_GROWTH_REALERT = 0.10

# How many top per-issue cache paths to attribute in the sub-floor row.
VM_DISK_SUBFLOOR_TOP_N = 3

# Hard wall-clock bound on the attribution `du -s` sweep (cheap — only the
# re-downloadable globs, never store/). A timeout degrades to "no attribution",
# never a crash.
VM_DISK_SUBFLOOR_DU_TIMEOUT_S = 60


def _root_disk_headroom() -> int | None:
    """Free bytes on the VM root (alias of :func:`_vm_free_bytes`, named for
    the sub-floor sentinel's call site). ``None`` on a statvfs failure."""
    return _vm_free_bytes()


def _issue_cache_glob_roots() -> list[Path]:
    """Every ``data/`` root that may hold per-issue ``hf_dl``/``g*_dl`` caches.

    The repo-root ``data/`` AND — critically (#681 / #658 evidence) — each
    WORKTREE-internal ``data/`` under ``.claude/worktrees/issue-<N>*/data/``,
    where the live run actually writes its caches (and where, post-#681, the
    bind-mounted worktree tree physically lives on the data disk). The original
    sub-floor attribution globbed only repo-root ``data/`` and so named the
    WRONG caches — it would miss the worktree-internal ones entirely (Must-Fix
    #2 implementer concern #1/#6). Returns only existing dirs."""
    roots: list[Path] = []
    repo_data = PROJECT_ROOT / "data"
    if repo_data.is_dir():
        roots.append(repo_data)
    wt_root = PROJECT_ROOT / ".claude" / "worktrees"
    if wt_root.is_dir():
        for wt in sorted(wt_root.iterdir()):
            wt_data = wt / "data"
            if wt_data.is_dir():
                roots.append(wt_data)
    return roots


def _top_issue_cache_paths(
    top_n: int = VM_DISK_SUBFLOOR_TOP_N, *, dry_run: bool = False
) -> list[tuple[str, int]]:
    """The ``top_n`` largest per-issue re-downloadable cache dirs under
    ``{data,.claude/worktrees/issue-*/data}/issue_*/{hf_dl,g*_dl}`` (NOT store/),
    as ``(rel_path, bytes)``.

    Cheap `du -s` on the cache globs ONLY — this is attribution, not a full
    tree walk. A `du` failure / timeout yields an empty list (no attribution),
    never a crash. Paths are relative to PROJECT_ROOT for the human pointer.

    The glob roots span the repo-root ``data/`` AND every worktree-internal
    ``data/`` (#681 — the per-issue caches the data disk actually holds live in
    the worktree, not repo-root ``data/``; the original repo-root-only glob
    named the wrong caches).

    Under ``dry_run`` the function performs NO ``subprocess.run`` at all and
    returns ``[]`` immediately: a dry-run pass must have zero observational
    side-effects (the attribution `du` shells out per candidate, which the
    ``--dry-run`` smoke contract forbids — #681 r3). The dry-run output line
    simply reports ``top caches: none``."""
    if dry_run:
        return []
    candidates: list[Path] = []
    for data_root in _issue_cache_glob_roots():
        for issue_dir in sorted(data_root.glob("issue*")):
            if not issue_dir.is_dir():
                continue
            for pattern in ("hf_dl", "g*_dl"):
                candidates.extend(p for p in issue_dir.glob(pattern) if p.is_dir())
    sizes: list[tuple[str, int]] = []
    for p in candidates:
        rc = subprocess.run(
            ["du", "-sx", "--block-size=1", str(p)],
            capture_output=True,
            text=True,
            timeout=VM_DISK_SUBFLOOR_DU_TIMEOUT_S,
            check=False,
        )
        if rc.returncode != 0 or not rc.stdout.strip():
            continue
        try:
            nbytes = int(rc.stdout.split()[0])
        except (ValueError, IndexError):
            continue
        try:
            rel = str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            rel = str(p)
        sizes.append((rel, nbytes))
    sizes.sort(key=lambda x: x[1], reverse=True)
    return sizes[:top_n]


def _top_issue_caches_by_project_quota(
    data_disk_path: str, top_n: int = VM_DISK_SUBFLOOR_TOP_N, *, dry_run: bool = False
) -> list[tuple[str, int]] | None:
    """PRIMARY data-disk attribution via ``repquota -P`` (per-PROJECT usage).

    Post-#681 each ``issue-<N>`` worktree subtree on the data disk carries an
    ext4 project id == the issue number, so ``repquota -P <data_disk>`` reports
    per-issue bytes used in ONE cheap call (no per-dir ``du`` tree walks). Parses
    the project-quota report into ``(#<projid>, bytes)`` rows sorted by usage,
    top ``top_n``. Returns ``None`` (NOT an empty list) when ``repquota`` is
    unavailable / the disk has no prjquota / parsing fails — the caller then
    falls back to the ``du``-based :func:`_top_issue_cache_paths`. Project id 0
    (the unbounded default — the managed pin + tiny worktrees) is excluded; it
    is not a per-issue cache. Never raises.

    Under ``dry_run`` performs NO ``subprocess.run`` and returns ``None`` (the
    same "no quota attribution available" signal as a missing ``repquota``), so
    the caller falls through to the likewise-short-circuited ``du`` helper and
    the dry-run pass shells out to nothing (#681 r3)."""
    if dry_run:
        return None
    rc = subprocess.run(
        ["repquota", "-Ocsv", "-P", data_disk_path],
        capture_output=True,
        text=True,
        timeout=VM_DISK_SUBFLOOR_DU_TIMEOUT_S,
        check=False,
    )
    # `repquota -Ocsv` emits CSV: Project,BlockStatus,FileStatus,BlockUsed,...
    # (block units are 1 KiB). A non-zero rc or no parseable rows -> None
    # (fall back to du).
    if rc.returncode != 0 or not rc.stdout.strip():
        return None
    rows: list[tuple[str, int]] = []
    for line in rc.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        projid = parts[0].strip().lstrip("#")
        if not projid.isdigit() or projid == "0":
            continue
        try:
            blocks_kib = int(parts[3].strip())
        except ValueError:
            continue
        rows.append((f"issue-{projid} (project quota, {data_disk_path})", blocks_kib * 1024))
    if not rows:
        return None
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _subfloor_state_path() -> Path:
    """Singleton dedup state for the sub-floor sentinel (last-alerted free
    bytes), under AUTONOMOUS_REGISTRY_DIR."""
    return AUTONOMOUS_REGISTRY_DIR / "vm-disk-subfloor.json"


def _load_subfloor_state() -> dict:
    path = _subfloor_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_subfloor_state(free_bytes: int) -> None:
    """Atomic temp+rename write of the sub-floor dedup state (fail-soft)."""
    dest = _subfloor_state_path()
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"last_free_bytes": free_bytes}))
        tmp.replace(dest)
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  vm-disk subfloor: state save failed: {exc}", file=sys.stderr)


def _clear_subfloor_state() -> None:
    _subfloor_state_path().unlink(missing_ok=True)


def decide_subfloor(free_bytes: int, last_free_bytes: int | None) -> bool:
    """Pure decision: write a sub-floor row this tick?

    True when free is below the sub-floor AND either no prior row exists this
    episode OR free DROPPED by more than the re-alert fraction since the last
    row. Recovery above the sub-floor clears the episode (handled by the
    caller). Keeps the row from re-firing every 10-min tick at a stable
    footprint."""
    if free_bytes >= VM_DISK_SUBFLOOR_FREE_BYTES:
        return False
    if not isinstance(last_free_bytes, int | float) or last_free_bytes <= 0:
        return True  # first sub-floor row this episode
    drop = (last_free_bytes - free_bytes) / last_free_bytes
    return drop > VM_DISK_SUBFLOOR_GROWTH_REALERT


# ─── VM DATA disk (/mnt/eps-data) watch — PERCENT-based (task #681) ──────────
#
# The dedicated data disk holds the relocated `.claude/worktrees/` tree. The
# `/`-tuned ABSOLUTE byte floors above (VM_DISK_ALERT_FREE_BYTES=20 GiB etc.)
# are calibrated to the 485 GB boot disk, where they sit at ~88-96% full;
# MIRRORED onto a 512 GB+ data disk they would fire at the WRONG fullness and
# would silently regress on a future resize to 2 TB (the byte floors would then
# sit at ~99% full — escalation AFTER the wedge). So the data-disk watch uses
# PERCENT thresholds derived from statvfs(total), which are SIZE-INVARIANT.
# Add NO `VM_DATA_DISK_*` absolute-byte constants (Must-Fix #2, plan §4/§11/§13).
#
# The data-disk pass is ESCALATE-ONLY (no reclaim arm runs there regardless) —
# the reclaim thresholds exist only to keep the percent decision parallel to the
# boot-disk `decide_vm_disk` shape; the caller never wires a reclaim action to
# the data disk.

# Percent-used thresholds for the data-disk pass (size-invariant). Defaults
# match §1's 85-95% acceptance band and the already-percent `vm_disk_guard`
# (threshold_pct() default 85). Env-overridable like the boot-disk knobs.
DATA_DISK_ALERT_PCT = 90.0
DATA_DISK_RECLAIM_PCT = 95.0  # escalate-only on the data disk — never reclaims
DATA_DISK_SUBFLOOR_PCT = 85.0

# The dedicated data-disk mount (the relocated `.claude/worktrees/` tree, #681),
# env-overridable. Same default + env var as `vm_disk_guard.data_disk_path()`
# so the guard cron and this every-tick watcher watch the SAME mount.
DEFAULT_DATA_DISK_PATH = "/mnt/eps-data"


def data_disk_path() -> str:
    """The watched data-disk mount, env-overridable (``EPS_VM_DATA_DISK_PATH``).

    Defaults to :data:`DEFAULT_DATA_DISK_PATH` (``/mnt/eps-data``); a blank /
    whitespace env value falls back to the default. Mirrors
    ``vm_disk_guard.data_disk_path`` so both disk watchers resolve the same
    mount."""
    raw = os.environ.get("EPS_VM_DATA_DISK_PATH", "").strip()
    return raw or DEFAULT_DATA_DISK_PATH


def _is_mounted(path: str) -> bool:
    """True iff ``path`` is itself a mount point (a distinct filesystem).

    A real mount-presence check — NOT ``Path(path).is_dir()`` (#681 round-2
    Major). ``is_dir()`` is True for ANY directory, mounted or not: after
    Phase-1's ``sudo mkdir -p /mnt/eps-data`` (which runs BEFORE the mount) or a
    ``nofail`` boot where the disk failed to mount, ``/mnt/eps-data`` exists as a
    plain root-fs directory, and :func:`data_disk_pass` would then misread
    ``/``'s statvfs (the boot disk) as data-disk usage and could emit a row
    claiming the data disk mirrors ``/``'s percent. Comparing ``st_dev`` against
    the parent's catches that: a real mount sits on a different device than its
    parent; a plain subdirectory shares it. Pure ``os.stat`` — no subprocess,
    fast, self-evidently correct, and aligned with the data-disk pass's
    zero-subprocess-in-dry-run contract (#681 r3). Mirrors
    ``vm_disk_guard._is_mounted``.

    Fail-soft: a missing path / stat error returns False (→ the pass cleanly
    no-ops). The filesystem root ``/`` is its own parent (equal ``st_dev``), so
    an unmounted path is never mistaken for the data disk."""
    try:
        st = os.stat(path)
        parent = os.stat(os.path.join(path, os.pardir))
    except OSError:
        return False
    return st.st_dev != parent.st_dev


def _data_disk_used_pct(dd_path: str) -> float | None:
    """Percent USED on the data-disk mount via ``statvfs`` (``None`` + a loud
    warning on failure — never crash the watcher over the disk check itself).

    Percent-of-total is the SIZE-INVARIANT basis (Must-Fix #2): a future
    resize cannot push the fire point past the wedge the way the boot disk's
    absolute byte floors would."""
    try:
        usage = shutil.disk_usage(dd_path)
    except OSError as e:
        print(f"  vm-disk-data: disk_usage({dd_path}) failed: {e}", file=sys.stderr)
        return None
    if usage.total <= 0:
        return None
    return 100.0 * (usage.total - usage.free) / usage.total


def _env_pct(name: str, default_pct: float) -> float:
    """Percent-denominated env knob -> float in (0, 100]. A garbled /
    out-of-range value falls back to the default (same fail-soft contract as
    :func:`_env_gib_bytes` — never crash the watcher at import)."""
    try:
        val = float(os.environ.get(name, ""))
    except ValueError:
        return default_pct
    if not (0.0 < val <= 100.0):
        return default_pct
    return val


def decide_subfloor_pct(used_pct: float, last_used_pct: float | None) -> bool:
    """PERCENT-based sub-floor decision for the DATA disk (size-invariant).

    True when ``used_pct`` is at/above :data:`DATA_DISK_SUBFLOOR_PCT`
    (env ``EPM_VM_DATA_DISK_SUBFLOOR_PCT``) AND either no prior row exists this
    episode OR usage CLIMBED by more than the re-alert fraction (in percentage
    POINTS) since the last row. The percent basis fires at the SAME fullness on
    a 512 GiB disk and a 2 TiB disk — unlike the boot disk's absolute byte
    floors (Must-Fix #2). Recovery below the sub-floor clears the episode
    (handled by the caller)."""
    floor = _env_pct("EPM_VM_DATA_DISK_SUBFLOOR_PCT", DATA_DISK_SUBFLOOR_PCT)
    if used_pct < floor:
        return False
    if not isinstance(last_used_pct, int | float) or last_used_pct <= 0:
        return True  # first sub-floor row this episode
    # Re-alert when usage climbs by > VM_DISK_SUBFLOOR_GROWTH_REALERT of the
    # remaining headroom-to-full, mirroring the boot-disk fractional re-alert but
    # in the percent domain (a 10% relative climb of used).
    climb = (used_pct - last_used_pct) / last_used_pct
    return climb > VM_DISK_SUBFLOOR_GROWTH_REALERT


def decide_vm_disk_pct(used_pct: float, *, alerted: bool) -> tuple[str, bool]:
    """PERCENT-based alert decision for the DATA disk (size-invariant).

    Returns ``(level, do_alert)``:

    - ``level`` — ``"ok"`` (below :data:`DATA_DISK_ALERT_PCT`), ``"low"`` (at/above
      alert but below :data:`DATA_DISK_RECLAIM_PCT`), or ``"critical"`` (at/above
      reclaim).
    - ``do_alert`` — fire the once-per-episode escalation (level low/critical AND
      ``alerted`` not already set).

    There is NO ``do_reclaim`` / ``do_audit`` return: the data-disk pass is
    ESCALATE-ONLY (the `/`-rooted reclaim arms operate on boot-disk caches). The
    thresholds are percent-of-statvfs(total), so a future resize cannot push the
    fire point past the wedge (the mirrored-byte-floor bug, Must-Fix #2)."""
    alert = _env_pct("EPM_VM_DATA_DISK_ALERT_PCT", DATA_DISK_ALERT_PCT)
    reclaim = _env_pct("EPM_VM_DATA_DISK_RECLAIM_PCT", DATA_DISK_RECLAIM_PCT)
    if used_pct < alert:
        return ("ok", False)
    level = "critical" if used_pct >= reclaim else "low"
    return (level, not alerted)


def _disk_guard_sidecar_path() -> Path:
    """The SHARED disk-guard escalation sidecar (same stream the
    clean_experiment_downloads + vm_disk_guard escalations use)."""
    return PROJECT_ROOT / ".claude" / "cache" / "disk-guard-events.jsonl"


def _append_disk_guard_sidecar(event: dict, dry_run: bool) -> None:
    """Append one JSON line to the shared disk-guard sidecar (fail-soft). A
    ``ts`` is stamped if absent. ``dry_run`` reports only."""
    row = {"ts": datetime.now(tz=UTC).isoformat(), **event}
    line = json.dumps(row)
    if dry_run:
        print(f"  [dry-run] would append disk-guard sidecar row: {line[:160]}")
        return
    dest = _disk_guard_sidecar_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"  vm-disk subfloor: sidecar append failed: {exc}", file=sys.stderr)


def subfloor_sentinel_pass(dry_run: bool, free_bytes: int | None = None) -> bool:
    """Warn-only sub-floor attribution pass (task #679).

    When VM-root free space drops below the sub-floor band, write a
    `band=sub-floor` advisory row to the shared disk-guard sidecar naming the
    top per-issue cache paths (cheap `du`) and signalling the watcher should
    re-check sooner. NEVER deletes anything; deduped on the drop fraction;
    clears the episode when free recovers above the band. Returns True when a
    row was written this tick. Fail-soft throughout."""
    free = free_bytes if free_bytes is not None else _root_disk_headroom()
    if free is None:
        return False
    state = _load_subfloor_state()
    last_free = state.get("last_free_bytes")
    last_free = last_free if isinstance(last_free, int | float) else None
    if free >= VM_DISK_SUBFLOOR_FREE_BYTES:
        if state and not dry_run:
            _clear_subfloor_state()  # episode over
        return False
    if not decide_subfloor(free, last_free):
        return False  # already alerted at ~this footprint; bound churn
    try:
        # dry_run short-circuits the `du` attribution to [] (no subprocess.run):
        # the dry-run pass must have zero observational side-effects (#681 r3).
        top = _top_issue_cache_paths(dry_run=dry_run)
    except (subprocess.SubprocessError, OSError):
        top = []
    free_gib = free / 2**30
    print(
        f"vm-disk SUB-FLOOR: {free_gib:.1f} GiB free on {VM_DISK_PATH} "
        f"(< {VM_DISK_SUBFLOOR_FREE_BYTES / 2**30:.0f} GiB) — re-check sooner; "
        f"top caches: {', '.join(f'{p} [{b / 1e9:.1f}G]' for p, b in top) or 'none'}",
        file=sys.stderr,
    )
    _append_disk_guard_sidecar(
        {
            "kind": "vm-disk-subfloor",
            "band": "sub-floor",
            "free_bytes": free,
            "free_gib": round(free_gib, 1),
            "top_cache_paths": [{"path": p, "bytes": b} for p, b in top],
            "recheck_sooner": True,
        },
        dry_run,
    )
    if not dry_run:
        _save_subfloor_state(free)
    return True


def _data_disk_state_path() -> Path:
    """Singleton dedup state for the DATA-disk pass (last-alerted used_pct +
    the alert latch), under AUTONOMOUS_REGISTRY_DIR. Distinct from the boot
    disk's ``vm-disk-subfloor.json`` so the two episodes never collide."""
    return AUTONOMOUS_REGISTRY_DIR / "vm-disk-data.json"


def _load_data_disk_state() -> dict:
    path = _data_disk_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_data_disk_state(state: dict) -> None:
    """Atomic temp+rename write of the data-disk dedup state (fail-soft)."""
    dest = _data_disk_state_path()
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(dest)
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  vm-disk-data: state save failed: {exc}", file=sys.stderr)


def _clear_data_disk_state() -> None:
    _data_disk_state_path().unlink(missing_ok=True)


def _data_disk_top_caches(dd_path: str, *, dry_run: bool = False) -> list[tuple[str, int]]:
    """Attribution for the data-disk escalation: PRIMARY per-PROJECT usage via
    ``repquota -P`` (one cheap call, project id == issue number), falling back
    to the ``du``-based glob attribution when ``repquota`` is unavailable / the
    disk has no prjquota / parsing fails. Fail-soft — any error yields an empty
    list (no attribution), never a crash.

    Under ``dry_run`` both helpers short-circuit (repquota → ``None``, du →
    ``[]``) so this returns ``[]`` with NO ``subprocess.run`` (#681 r3): the
    data-disk dry-run smoke must have zero observational side-effects."""
    try:
        rows = _top_issue_caches_by_project_quota(dd_path, dry_run=dry_run)
    except (subprocess.SubprocessError, OSError):
        rows = None
    if rows is not None:
        return rows
    try:
        return _top_issue_cache_paths(dry_run=dry_run)
    except (subprocess.SubprocessError, OSError):
        return []


def data_disk_pass(dry_run: bool, used_pct: float | None = None) -> bool:
    """Watch the dedicated data disk (``/mnt/eps-data``, #681) — ESCALATE-ONLY.

    A SECOND disk-watch pass mirroring :func:`vm_disk_pass`'s shape but bound to
    the data disk and driving the PERCENT decision helpers
    (:func:`decide_vm_disk_pct` / :func:`decide_subfloor_pct`) so the fire point
    is SIZE-INVARIANT (Must-Fix #2 — no mirrored byte floors). It is the
    every-tick analogue of ``vm_disk_guard.run_guard(disk_path=...,
    reclaim_tiers=False)``: the ``/``-rooted reclaim arms (``uv cache prune``,
    the stale-log sweep) NEVER run keyed off the data disk — this pass only
    ESCALATES (warn-only sidecar rows + a once-per-episode alert log).

    Mount-guarded (:func:`_is_mounted`, ``st_dev`` vs parent) so it is a CLEAN
    no-op before / without the #681 cutover — a missing data disk OR an
    existing-but-unmounted ``/mnt/eps-data`` (a plain dir from Phase-1's
    ``mkdir -p`` or a ``nofail`` boot that failed to mount) writes NO sidecar
    row and touches no state. ``is_dir()`` is insufficient: a plain directory
    passes it and the pass would then misread ``/``'s statvfs as the data disk's
    (#681 round-2 Major). ``used_pct`` is injectable for tests; production
    derives it from ``statvfs`` of the mount.

    Returns True when a sub-floor row was written this tick (mirrors
    :func:`subfloor_sentinel_pass`'s return contract). Fail-soft throughout."""
    dd_path = data_disk_path()
    if not _is_mounted(dd_path):
        return False  # data disk absent / not mounted — clean no-op pre-cutover
    pct = used_pct if used_pct is not None else _data_disk_used_pct(dd_path)
    if pct is None:
        return False

    state = _load_data_disk_state()
    last_used_pct = state.get("last_used_pct")
    last_used_pct = last_used_pct if isinstance(last_used_pct, int | float) else None
    alerted = bool(state.get("alerted", False))

    # ── Alert/critical arm (decide_vm_disk_pct) ──────────────────────────────
    level, do_alert = decide_vm_disk_pct(pct, alerted=alerted)
    if level == "ok":
        # Recovered below the alert band — clear the episode so the next dip
        # alerts afresh (sub-floor episode is cleared by its own arm below).
        if alerted and not dry_run:
            new_state = dict(state)
            new_state.pop("alerted", None)
            _save_data_disk_state(new_state)
        alerted = False
    else:
        print(
            f"vm-disk-data: {level.upper()} — {pct:.1f}% used on {dd_path} "
            f"(alert >= {_env_pct('EPM_VM_DATA_DISK_ALERT_PCT', DATA_DISK_ALERT_PCT):.0f}%, "
            f"critical >= {_env_pct('EPM_VM_DATA_DISK_RECLAIM_PCT', DATA_DISK_RECLAIM_PCT):.0f}%); "
            "ESCALATE-ONLY — recovery is resize / raise-cap, never delete active data",
            file=sys.stderr,
        )
        if do_alert:
            top = _data_disk_top_caches(dd_path, dry_run=dry_run)
            _append_disk_guard_sidecar(
                {
                    "kind": f"vm-disk-data-{'critical' if level == 'critical' else 'low'}",
                    "disk": "data",
                    "data_disk_path": dd_path,
                    "level": level,
                    "used_pct": round(pct, 1),
                    "top_cache_paths": [{"path": p, "bytes": b} for p, b in top],
                    "recovery": (
                        "resize / raise setquota -P cap on a TERMINAL issue — "
                        "never delete active data"
                    ),
                },
                dry_run,
            )
            if not dry_run:
                state = _load_data_disk_state()  # re-read in case the ok-branch wrote
                state["alerted"] = True
                _save_data_disk_state(state)

    # ── Sub-floor sentinel arm (decide_subfloor_pct) ─────────────────────────
    floor = _env_pct("EPM_VM_DATA_DISK_SUBFLOOR_PCT", DATA_DISK_SUBFLOOR_PCT)
    if pct < floor:
        # Below the sub-floor band → episode over; clear the last_used_pct dedup
        # cursor (keep any alert latch handled above by re-reading).
        if last_used_pct is not None and not dry_run:
            cur = _load_data_disk_state()
            cur.pop("last_used_pct", None)
            _save_data_disk_state(cur)
        return False
    if not decide_subfloor_pct(pct, last_used_pct):
        return False  # already alerted at ~this footprint; bound churn

    top = _data_disk_top_caches(dd_path, dry_run=dry_run)
    print(
        f"vm-disk-data SUB-FLOOR: {pct:.1f}% used on {dd_path} "
        f"(>= {floor:.0f}%) — re-check sooner; "
        f"top caches: {', '.join(f'{p} [{b / 1e9:.1f}G]' for p, b in top) or 'none'}",
        file=sys.stderr,
    )
    _append_disk_guard_sidecar(
        {
            "kind": "vm-disk-data-subfloor",
            "band": "sub-floor",
            "disk": "data",
            "data_disk_path": dd_path,
            "used_pct": round(pct, 1),
            "top_cache_paths": [{"path": p, "bytes": b} for p, b in top],
            "recheck_sooner": True,
        },
        dry_run,
    )
    if not dry_run:
        cur = _load_data_disk_state()
        cur["last_used_pct"] = pct
        _save_data_disk_state(cur)
    return True


def _happy_patch_state_path() -> Path:
    """Singleton per-episode dedup state for :func:`happy_patch_pass`, under
    AUTONOMOUS_REGISTRY_DIR. Records the last-alerted patch state so the pass
    escalates a revert/drift once per episode (and re-alerts when the state
    CHANGES, e.g. reverted -> drifted). Mirrors ``vm-disk-data.json``'s shape."""
    return AUTONOMOUS_REGISTRY_DIR / "happy-patch-alert.json"


def _load_happy_patch_state() -> dict:
    path = _happy_patch_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_happy_patch_state(state: dict) -> None:
    """Atomic temp+rename write of the happy-patch dedup state (fail-soft)."""
    dest = _happy_patch_state_path()
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(dest)
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  happy-patch: state save failed: {exc}", file=sys.stderr)


def _clear_happy_patch_state() -> None:
    _happy_patch_state_path().unlink(missing_ok=True)


def happy_patch_pass(dry_run: bool) -> None:
    """Proactively surface a reverted/drifted Happy injection patch (task #726).

    The spawn-path guard (spawn_session._verify_happy_patch_or_die) is REACTIVE
    — it only fires at the next spawn attempt. This pass is PROACTIVE: it runs
    every 10-min tick, so a revert (typically from `npm update happy`) is
    surfaced within ~10 min rather than at the next autonomous dispatch.
    Escalate-only: never re-applies (that needs sudo); writes a sidecar row +
    fail-soft Telegram push, deduped per (state) so it alerts once per episode.
    Clean no-op when the patch is present or the daemon file is absent.

    Daemon-INDEPENDENT: it reads a local file only, so it runs every tick
    alongside vm_disk_pass / data_disk_pass / program_orchestrator_pass, BEFORE
    the daemon-gated session passes. Fail-soft throughout — a classify error is
    logged and swallowed, never propagated."""
    sp = str(PROJECT_ROOT / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    try:
        import _happy_patch_check as hpc

        st = hpc.classify_patch()
    except Exception as exc:  # fail-soft like the disk passes
        print(f"  happy-patch: classify failed ({exc}); skipping", file=sys.stderr)
        return
    if st.state in ("patched", "missing"):
        # `missing` = the .mjs file is absent; on the watcher VM this is the
        # 'Happy not installed here' case (escalate-only stays conservative —
        # the spawn-path guard owns the precise reachability disambiguation via
        # daemon.state.json). Reset the dedup so a future revert re-alerts.
        if not dry_run:
            _clear_happy_patch_state()
        return
    # reverted | drifted -> escalate-only
    state = _load_happy_patch_state()
    if state.get("alerted_state") == st.state:
        print(f"  happy-patch: {st.state} (already alerted this episode)")
        return
    print(
        f"happy-patch: {st.state.upper()} — {st.detail}. Autonomous spawns will "
        f"produce idle sessions until re-applied; ESCALATE-ONLY "
        f"(re-apply needs sudo, never auto-applied here).",
        file=sys.stderr,
    )
    _append_disk_guard_sidecar(
        {
            "kind": f"happy-patch-{st.state}",
            "band": "happy-patch",
            "state": st.state,
            "detail": st.detail,
            "reapply_cmd": hpc.REAPPLY_CMD,
            "restart_cmd": hpc.RESTART_CMD,
        },
        dry_run,
    )
    _telegram_push(
        f"Happy injection patch {st.state}: {st.detail}. Autonomous spawns will "
        f"produce idle sessions until re-applied: {hpc.REAPPLY_CMD} && "
        f"{hpc.RESTART_CMD}",
        dry_run,
    )
    if not dry_run:
        _save_happy_patch_state({"alerted_state": st.state})


# ─── CPU/memory-pressure guard pass (task #849) — escalate-only ──────────────
#
# WHY: 2026-07-02 incident — the shared 32-core VM sat at load 186-226 for
# hours; earlyoom SIGTERM sweeps silently killed 4-7 GB analysis workers
# (exit 143, no traceback, misattributed for hours); a checkpointed sweep's
# 60 s layers stretched to 64 min. The watcher watched sessions/pods/disk but
# not the box's compute pressure. This pass is the CPU/memory analogue of the
# disk sub-floor sentinel above: detection + attribution + one deduped push.
# WARN-ONLY: it NEVER kills, NEVER renices, NEVER signals any process.
# (End-of-block sentinel for the never-kills grep test: the block ends at the
#  dedicated end-of-block comment placed directly after cpu_guard_pass below —
#  see tests/test_cpu_guard_pass.py::test_cpu_guard_never_kills.)


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    """Float env knob with sanity bounds; a garbled or out-of-range value
    falls back to the default (same fail-soft contract as
    :func:`_env_gib_bytes` / :func:`_env_pct` — never crash the watcher at
    import over a typo'd override)."""
    try:
        val = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return val if lo <= val <= hi else default


# load5 fires above CPU_GUARD_LOAD_FACTOR * nproc: 50% sustained
# oversubscription — healthy full utilization (load ~= nproc) never fires;
# the 2026-07-02 incident ran at 5.8-7x nproc.
CPU_GUARD_LOAD_FACTOR = _env_float("EPM_VM_CPU_GUARD_LOAD_FACTOR", 1.5, lo=0.1, hi=100.0)
# PSI cpu `some avg10` (share of wall time >=1 runnable task stalled for CPU):
# 50% sustained is unambiguous oversubscription (76% under the live incident).
CPU_GUARD_PSI_CPU_SOME_PCT = _env_pct("EPM_VM_CPU_GUARD_PSI_CPU_PCT", 50.0)
# PSI memory `full avg10` (all non-idle tasks stalled simultaneously) — direct
# thrash indicator; the 10% default is ungrounded beyond kernel-doc semantics
# (warn-only + env-overridable makes miscalibration cheap; plan §11).
CPU_GUARD_PSI_MEM_FULL_PCT = _env_pct("EPM_VM_CPU_GUARD_PSI_MEM_PCT", 10.0)
# MemAvailable floor: fires SINGLE-TICK (urgent) — the pre-kill attribution
# window. 20% sits above earlyoom's 10% kill floor (its own journal config
# line: "sending SIGTERM when mem <= 10.00%"), so this leg fires while the
# culprit is still alive and capturable.
CPU_GUARD_MEMAVAIL_PCT = _env_pct("EPM_VM_CPU_GUARD_MEMAVAIL_PCT", 20.0)
# Consecutive hot ticks before the RATE signals (load/PSI) fire — 2 = ~20 min
# at the 10-min cron (the main() --threshold precedent).
CPU_GUARD_TICKS = int(_env_float("EPM_VM_CPU_GUARD_TICKS", 2, lo=1, hi=100))
CPU_GUARD_TOP_N = 5  # union of top-CPU and top-RSS in the attribution snapshot
CPU_GUARD_REALERT_GROWTH = 0.25  # re-alert on >25% load5 growth in-episode
CPU_GUARD_SUBPROC_TIMEOUT_S = 15  # ps / journalctl hard bound
CPU_GUARD_KILL_PUSH_MIN_INTERVAL_S = 3600  # kill-push rate limit (sidecar keeps all)
CPU_GUARD_JOURNAL_OVERLAP_S = 60  # re-scan overlap; key-dedup kills the dups
# Cap the journal re-scan window after a long watcher outage: bounds a
# >50-kill backlog re-emission against the 50-key dedup cap.
CPU_GUARD_JOURNAL_MAX_LOOKBACK_S = 86400

# Reasons that fire on a SINGLE hot tick (no streak): memory can collapse
# 15% -> 3% inside one 10-min interval, so waiting out a streak would fire
# post-kill — the single-tick fire IS the pre-kill attribution record.
CPU_GUARD_URGENT_REASONS = {"mem-avail"}

# Real captured line (2026-06-28): `... earlyoom[2703914]: sending SIGTERM to
# process 4087688 uid 1001 "pytest": badness 984, VmRSS 3390 MiB`.
_EARLYOOM_KILL_RE = re.compile(
    r'sending (SIGTERM|SIGKILL) to process (\d+) uid (\d+) "([^"]*)": '
    r"badness (\d+), VmRSS (\d+) MiB"
)
_WORKTREE_ISSUE_RE = re.compile(r"\.claude/worktrees/issue-(\d+)")
_CMDLINE_ISSUE_RE = re.compile(r"(?:--issue[ =](\d+)|\bissue[_-](\d+)\b)")


def _cpu_guard_enabled() -> bool:
    """Kill switch: False when ``EPM_DISABLE_CPU_GUARD_PASS`` is set truthy
    ("1"/"true"/"yes", case-insensitive). Default enabled. Mirrors
    :func:`_infra_drain_enabled`."""
    raw = os.environ.get("EPM_DISABLE_CPU_GUARD_PASS", "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _read_loadavg() -> tuple[float, float] | None:
    """``(load1, load5)`` from ``/proc/loadavg``; ``None`` on a read/parse
    failure (fail-soft — a missing signal degrades to unavailable, it never
    fires and never masks the other signals)."""
    try:
        parts = Path("/proc/loadavg").read_text().split()
        return (float(parts[0]), float(parts[1]))
    except (OSError, ValueError, IndexError):
        return None


def parse_psi_avg10(text: str, kind: str) -> float | None:
    """PURE: the ``avg10`` value from a ``/proc/pressure/{cpu,memory}`` body
    for the given ``kind`` line (``"some"`` / ``"full"``); ``None`` when the
    line or field is missing/garbled. Line shape (kernel PSI docs):
    ``some avg10=76.08 avg60=61.25 avg300=52.44 total=...``."""
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != kind:
            continue
        for tok in parts[1:]:
            if tok.startswith("avg10="):
                try:
                    return float(tok[len("avg10=") :])
                except ValueError:
                    return None
    return None


def _read_psi_avg10(path: str, kind: str) -> float | None:
    """Fail-soft file wrapper over :func:`parse_psi_avg10` (a missing PSI
    interface — e.g. CONFIG_PSI=n — degrades that one signal to ``None``)."""
    try:
        return parse_psi_avg10(Path(path).read_text(), kind)
    except OSError:
        return None


def parse_meminfo_avail_pct(text: str) -> float | None:
    """PURE: ``100 * MemAvailable / MemTotal`` from a ``/proc/meminfo`` body;
    ``None`` when either field is missing/garbled or MemTotal is 0."""
    total_kib: int | None = None
    avail_kib: int | None = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0] == "MemTotal:":
            try:
                total_kib = int(parts[1])
            except ValueError:
                return None
        elif parts[0] == "MemAvailable:":
            try:
                avail_kib = int(parts[1])
            except ValueError:
                return None
    if total_kib is None or avail_kib is None or total_kib <= 0:
        return None
    return 100.0 * avail_kib / total_kib


def _read_mem_avail_pct() -> float | None:
    """Fail-soft file wrapper over :func:`parse_meminfo_avail_pct`."""
    try:
        return parse_meminfo_avail_pct(Path("/proc/meminfo").read_text())
    except OSError:
        return None


def cpu_tick_hot_reasons(
    load5: float | None,
    ncpu: int,
    psi_cpu_some: float | None,
    psi_mem_full: float | None,
    mem_avail_pct: float | None,
) -> list[str]:
    """PURE: which pressure signals are hot THIS tick. ``None`` inputs are
    skipped (fail-soft: a missing signal can never fire OR mask the others).
    Tags: ``loadavg`` | ``psi-cpu`` | ``psi-memory`` | ``mem-avail``."""
    reasons: list[str] = []
    if load5 is not None and load5 > CPU_GUARD_LOAD_FACTOR * ncpu:
        reasons.append("loadavg")
    if psi_cpu_some is not None and psi_cpu_some > CPU_GUARD_PSI_CPU_SOME_PCT:
        reasons.append("psi-cpu")
    if psi_mem_full is not None and psi_mem_full > CPU_GUARD_PSI_MEM_FULL_PCT:
        reasons.append("psi-memory")
    if mem_avail_pct is not None and mem_avail_pct < CPU_GUARD_MEMAVAIL_PCT:
        reasons.append("mem-avail")
    return reasons


def decide_cpu_guard_fire(
    reasons: list[str],
    prior_consecutive: int,
    alerted: bool,
    last_alert_load5: float | None,
    last_alert_reasons: list[str] | None,
    load5: float | None,
) -> tuple[bool, int]:
    """PURE: ``(fire?, new_consecutive)`` for the pressure arm.

    Not hot this tick -> ``(False, 0)`` (the caller resets the episode).
    Hot -> ``consecutive = prior + 1``. The tick FIRES when EITHER

    - (a) ``consecutive >= CPU_GUARD_TICKS`` (streaked rate signals — load /
      PSI need ~20 min persistence at the 10-min cron, so a healthy short
      burst never alerts), OR
    - (b) any reason is urgent (:data:`CPU_GUARD_URGENT_REASONS` —
      single-tick, see that constant's rationale),

    AND the in-episode dedup admits it: not yet alerted this episode, OR the
    sorted reason set changed since the last alert, OR load5 grew by more
    than :data:`CPU_GUARD_REALERT_GROWTH` vs the last alert."""
    if not reasons:
        return (False, 0)
    consecutive = prior_consecutive + 1
    streak_met = consecutive >= CPU_GUARD_TICKS
    urgent = bool(set(reasons) & CPU_GUARD_URGENT_REASONS)
    if not (streak_met or urgent):
        return (False, consecutive)
    if not alerted:
        return (True, consecutive)
    if last_alert_reasons is not None and sorted(reasons) != sorted(
        str(x) for x in last_alert_reasons
    ):
        return (True, consecutive)
    if (
        load5 is not None
        and isinstance(last_alert_load5, int | float)
        and last_alert_load5 > 0
        and (load5 - last_alert_load5) / last_alert_load5 > CPU_GUARD_REALERT_GROWTH
    ):
        return (True, consecutive)
    return (False, consecutive)


def attribute_issue(cwd: str | None, argv: str) -> int | None:
    """PURE: best-effort pid -> issue attribution. A worktree cwd
    (``.claude/worktrees/issue-<N>``) wins; else a cmdline hint
    (``--issue <N>`` / ``issue_<N>`` / ``issue-<N>``); else ``None``."""
    if cwd:
        m = _WORKTREE_ISSUE_RE.search(cwd)
        if m:
            return int(m.group(1))
    m = _CMDLINE_ISSUE_RE.search(argv or "")
    if m:
        return int(m.group(1) or m.group(2))
    return None


def _top_processes(top_n: int = CPU_GUARD_TOP_N, *, dry_run: bool = False) -> list[dict]:
    """The union of the ``top_n`` top-CPU and ``top_n`` top-RSS processes,
    each attributed to an issue where possible (``/proc/<pid>/cwd`` readlink
    + cmdline hints via :func:`attribute_issue`).

    ONE ``ps -eo pid,pcpu,rss,args`` call (hard 15 s timeout) — cheap enough
    for a firing tick. NOTE: ``ps`` ``pcpu`` is LIFETIME %CPU (cpu-time /
    elapsed), representative for the hours-long sustained incident class but
    it can under-report a fresh spike (documented limitation, plan §8).

    Under ``dry_run`` performs NO ``subprocess.run`` and returns ``[]``
    immediately — the dry-run smoke contract forbids observational
    side-effects (the #681 r3 convention). Fail-soft: any ps failure yields
    ``[]`` (no attribution), never a crash."""
    if dry_run:
        return []
    try:
        rc = subprocess.run(
            ["ps", "-eo", "pid,pcpu,rss,args", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=CPU_GUARD_SUBPROC_TIMEOUT_S,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if rc.returncode != 0:
        return []
    rows: list[dict] = []
    for line in rc.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            pcpu = float(parts[1])
            rss_kib = int(parts[2])
        except ValueError:
            continue
        rows.append(
            {"pid": pid, "pcpu": pcpu, "rss_mib": round(rss_kib / 1024, 1), "argv": parts[3][:200]}
        )
    by_cpu = sorted(rows, key=lambda r: r["pcpu"], reverse=True)[:top_n]
    by_rss = sorted(rows, key=lambda r: r["rss_mib"], reverse=True)[:top_n]
    union: dict[int, dict] = {}
    for r in by_cpu + by_rss:
        union[r["pid"]] = r
    procs = list(union.values())
    for r in procs:
        try:
            cwd: str | None = os.readlink(f"/proc/{r['pid']}/cwd")
        except OSError:
            cwd = None  # process gone / unreadable — cmdline hints still apply
        r["issue"] = attribute_issue(cwd, r["argv"])
    procs.sort(key=lambda r: r["pcpu"], reverse=True)
    return procs


def attribute_kill(kill: dict, snapshot: dict | None) -> tuple[int | None, str]:
    """PURE: ``(issue, attribution_status)`` for an earlyoom kill row.

    Matches the killed pid against the rolling PRE-KILL snapshot stored on
    the last pressure fire (an earlyoom-killed pid has no ``/proc/<pid>/cwd``
    left — pid -> issue is capturable only pre-kill). A pid match wins; else
    a UNIQUE comm match among the snapshot argv basenames (compared on the
    first 15 chars of the basename — the kernel truncates ``comm`` to 15).
    ``"attributed"`` REQUIRES an int issue on the matched row: a match whose
    snapshot row carries no issue is honestly ``(None, "unattributed")``
    (never ``issue: null`` + ``attributed``). No snapshot / no match ->
    ``(None, "unattributed")``. Never raises."""
    try:
        if not isinstance(snapshot, dict):
            return (None, "unattributed")
        procs = [p for p in snapshot.get("procs") or [] if isinstance(p, dict)]
        pid = kill.get("pid")
        for p in procs:
            if p.get("pid") == pid:
                issue = p.get("issue")
                # type() not isinstance(): bool subclasses int, so a corrupt
                # snapshot row {"issue": true} must NOT attribute.
                if type(issue) is int:
                    return (issue, "attributed")
                # Pid match is authoritative: the killed process IS this
                # snapshot row, and it has no issue — do NOT fall through to
                # comm matching (that could name a DIFFERENT process's issue).
                return (None, "unattributed")
        comm = kill.get("comm")
        if isinstance(comm, str) and comm:
            matches = []
            for p in procs:
                argv = p.get("argv")
                if not isinstance(argv, str):
                    continue
                toks = argv.split()
                # Kernel comm is 15-char truncated; match by 15-char prefix
                # so long-named processes still comm-match.
                if toks and os.path.basename(toks[0])[:15] == comm:
                    matches.append(p)
            if len(matches) == 1:
                issue = matches[0].get("issue")
                # type() not isinstance(): bool subclasses int (see pid path).
                if type(issue) is int:
                    return (issue, "attributed")
                return (None, "unattributed")
        return (None, "unattributed")
    except Exception:  # pragma: no cover - absolute fail-soft backstop
        return (None, "unattributed")


def parse_earlyoom_kill_line(line: str) -> dict | None:
    """PURE: parse one ``journalctl -u earlyoom -o short-iso`` line into a
    kill dict, or ``None`` for non-kill lines (the ``mem avail:`` chatter and
    the ``sending SIGTERM when mem <= ...`` config line). ``-o short-iso``
    prepends the ISO timestamp as the first whitespace-separated token."""
    m = _EARLYOOM_KILL_RE.search(line)
    if not m:
        return None
    first = line.split(None, 1)
    return {
        "journal_ts": first[0] if first else "",
        "signal": m.group(1),
        "pid": int(m.group(2)),
        "uid": int(m.group(3)),
        "comm": m.group(4),
        "badness": int(m.group(5)),
        "vmrss_mib": int(m.group(6)),
    }


def _earlyoom_kills_since(since_epoch: float, *, dry_run: bool = False) -> list[dict] | None:
    """Kill events from ``journalctl -u earlyoom --no-pager -o short-iso
    --since @<int(epoch)>``, parsed through :func:`parse_earlyoom_kill_line`.

    ``None`` on journalctl missing / non-zero rc / timeout (fail-soft — the
    caller degrades the kill arm VISIBLY and does NOT advance the journal
    cursor); ``[]`` on a clean scan with no kills. Under ``dry_run`` performs
    NO ``subprocess.run`` and returns ``None`` (#681 r3 convention).

    NOTE: the FIRST-run lookback is DELIBERATELY bounded (~30 min, set by the
    caller) — the watcher is an ongoing monitor, not a backfill tool; and
    after any outage the re-scan window is additionally capped at
    :data:`CPU_GUARD_JOURNAL_MAX_LOOKBACK_S` (also caller-side)."""
    if dry_run:
        return None
    cmd = [
        "journalctl",
        "-u",
        "earlyoom",
        "--no-pager",
        "-o",
        "short-iso",
        "--since",
        f"@{int(since_epoch)}",
    ]
    try:
        rc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CPU_GUARD_SUBPROC_TIMEOUT_S,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if rc.returncode != 0:
        return None
    kills: list[dict] = []
    for line in rc.stdout.splitlines():
        k = parse_earlyoom_kill_line(line)
        if k is not None:
            kills.append(k)
    return kills


def _cpu_guard_state_path() -> Path:
    """Singleton dedup + streak + journal-cursor state for the CPU guard,
    under AUTONOMOUS_REGISTRY_DIR (the watcher-state convention; singletons
    are never GC'd — :func:`_gc_target_paths` sweeps per-issue files only)."""
    return AUTONOMOUS_REGISTRY_DIR / "vm-cpu-guard.json"


def _load_cpu_guard_state() -> dict:
    """``{}`` on missing/garbled state (mirrors :func:`_load_subfloor_state`).
    Every field read back from this dict goes through ``isinstance``
    type-guards at the call sites — a hand-edited or schema-drifted state
    file degrades to defaults, never raises."""
    path = _cpu_guard_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cpu_guard_state(state: dict) -> None:
    """Atomic temp+rename write of the CPU-guard state (fail-soft)."""
    dest = _cpu_guard_state_path()
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(dest)
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  cpu-guard: state save failed: {exc}", file=sys.stderr)


def _cpu_guard_sidecar_path() -> Path:
    """DEDICATED CPU-guard event stream (task-body requirement — domain-
    separated from the shared disk-guard sidecar for clean grep)."""
    return PROJECT_ROOT / ".claude" / "cache" / "cpu-guard-events.jsonl"


def _append_cpu_guard_sidecar(event: dict, dry_run: bool) -> None:
    """Append one JSON line to the CPU-guard sidecar (fail-soft). A ``ts`` is
    stamped if absent. ``dry_run`` reports only."""
    row = {"ts": datetime.now(tz=UTC).isoformat(), **event}
    line = json.dumps(row)
    if dry_run:
        print(f"  [dry-run] would append cpu-guard sidecar row: {line[:160]}")
        return
    dest = _cpu_guard_sidecar_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"  cpu-guard: sidecar append failed: {exc}", file=sys.stderr)


def cpu_guard_pass(dry_run: bool) -> bool:
    """Escalate-only CPU/memory-pressure + earlyoom-kill observability (#849).

    Two arms, both WARN-ONLY (never kills / renices / signals a process):

    - (a) **earlyoom kill surfacing** — every tick, threshold-independent:
      grep the earlyoom journal since the persisted cursor and write one
      ``kind=earlyoom-kill`` sidecar row per NEW kill (journal-ts+pid key
      dedup), attributed from the rolling pre-kill snapshot with an explicit
      ``attribution_status: attributed | unattributed``. Pushes are
      rate-limited to one per :data:`CPU_GUARD_KILL_PUSH_MIN_INTERVAL_S`
      (the sidecar keeps full fidelity).
    - (b) **pressure detection** — streaked rate signals (load5 / PSI, 2
      consecutive ticks) + the SINGLE-TICK MemAvailable floor (urgent — the
      pre-kill attribution window). A fire writes one attributed
      ``kind=vm-cpu-pressure`` row + a deduped push, and stores the
      top-process snapshot in state for arm (a). Recovery (no hot reasons)
      resets the episode so a later re-overload fires afresh.

    Daemon-independent; fail-soft throughout (never-raise helpers +
    ``isinstance`` state guards, per the sibling-pass convention). Returns
    True when any sidecar row was written this tick."""
    if not _cpu_guard_enabled():
        print("  cpu-guard: disabled via EPM_DISABLE_CPU_GUARD_PASS; skipping")
        return False
    state = _load_cpu_guard_state()
    now = time.time()
    wrote = False
    kill_arm_ok = True

    # ── (a) earlyoom kill surfacing — EVERY tick, threshold-independent ─────
    last_scan = state.get("last_journal_epoch")
    if not isinstance(last_scan, int | float):
        last_scan = now - 1800  # first-run lookback: DELIBERATE ~30 min bound
    since = max(last_scan - CPU_GUARD_JOURNAL_OVERLAP_S, now - CPU_GUARD_JOURNAL_MAX_LOOKBACK_S)
    kills = _earlyoom_kills_since(since, dry_run=dry_run)
    if kills is None and not dry_run:
        # VISIBLE degradation (never a silent protection illusion): stderr
        # line here + a `kill_arm: unavailable` field on any pressure row
        # fired this tick. Cursor NOT advanced — the next successful scan
        # re-covers the gap (bounded by CPU_GUARD_JOURNAL_MAX_LOOKBACK_S).
        kill_arm_ok = False
        print(
            "  cpu-guard: earlyoom kill arm unavailable (journalctl failed/missing); "
            "kill surfacing degraded this tick",
            file=sys.stderr,
        )
    elif kills is not None:
        # Guard the CONTAINER before the elements: a valid-JSON state file
        # with e.g. `"recent_kill_keys": 5` (truthy non-iterable) must degrade
        # to "no keys", never TypeError — the pass is called unwrapped in
        # main()'s daemon-independent block, so a raise here would abort the
        # ENTIRE watcher tick every 10 min (AC4; r1 review Critical).
        raw_keys = state.get("recent_kill_keys")
        seen = {k for k in (raw_keys if isinstance(raw_keys, list) else []) if isinstance(k, str)}
        raw_snap = state.get("last_top_snapshot")
        snapshot = raw_snap if isinstance(raw_snap, dict) else None
        new = [k for k in kills if f"{k['journal_ts']}:{k['pid']}" not in seen]
        for k in new:
            issue, status = attribute_kill(k, snapshot)
            print(
                f'vm-cpu EARLYOOM KILL: {k["signal"]} pid {k["pid"]} "{k["comm"]}" '
                f"VmRSS {k['vmrss_mib']} MiB — issue "
                f"{issue if issue is not None else '?'} ({status})",
                file=sys.stderr,
            )
            _append_cpu_guard_sidecar(
                {
                    "kind": "earlyoom-kill",
                    **k,
                    "issue": issue,
                    "attribution_status": status,
                    "attribution_source": ("pre-kill-snapshot" if status == "attributed" else None),
                },
                dry_run,
            )
        if new:
            wrote = True
            last_push = state.get("last_kill_push_ts")
            if not isinstance(last_push, int | float):
                last_push = 0
            if now - last_push > CPU_GUARD_KILL_PUSH_MIN_INTERVAL_S:
                head = ", ".join(
                    f"pid {k['pid']} {k['comm']} ({k['vmrss_mib']} MiB)" for k in new[:3]
                )
                _telegram_push(
                    f"earlyoom killed {len(new)} process(es) (silent exit-143 SIGTERM): "
                    f"{head} — see .claude/cache/cpu-guard-events.jsonl",
                    dry_run,
                )
                state["last_kill_push_ts"] = now
        # Keep the NEWEST keys under the 50-key cap: journal order is
        # oldest-first, so reverse before truncating — a kill inside the 60 s
        # overlap tail of a >50-kill backlog scan must survive the cap or it
        # would be re-emitted once on the next tick (r1 review Minor).
        new_keys = [f"{k['journal_ts']}:{k['pid']}" for k in kills]
        state["recent_kill_keys"] = list(dict.fromkeys(list(reversed(new_keys)) + sorted(seen)))[
            :50
        ]
        state["last_journal_epoch"] = now

    # ── (b) pressure detection: streaked rate signals + single-tick floor ───
    la = _read_loadavg()
    load1, load5 = la if la is not None else (None, None)
    ncpu = os.cpu_count() or 1
    psi_cpu = _read_psi_avg10("/proc/pressure/cpu", "some")
    psi_mem = _read_psi_avg10("/proc/pressure/memory", "full")
    mem_avail = _read_mem_avail_pct()
    reasons = cpu_tick_hot_reasons(load5, ncpu, psi_cpu, psi_mem, mem_avail)
    prior_consec = state.get("consecutive_hot")
    if not isinstance(prior_consec, int):
        prior_consec = 0
    raw_last_load5 = state.get("last_alert_load5")
    raw_last_reasons = state.get("last_alert_reasons")
    # Type-guard `alerted` like every other read-back field: a wrong-type
    # truthy (e.g. "yes") would read as alerted=True with NO valid
    # last_alert_reasons/load5, suppressing a real episode indefinitely
    # (r1 review Major). Non-bool degrades to False (fires at worst one
    # extra row — escalate-only, so over-alerting beats suppression).
    raw_alerted = state.get("alerted")
    fire, consec = decide_cpu_guard_fire(
        reasons,
        prior_consec,
        raw_alerted if isinstance(raw_alerted, bool) else False,
        raw_last_load5 if isinstance(raw_last_load5, int | float) else None,
        raw_last_reasons if isinstance(raw_last_reasons, list) else None,
        load5,
    )
    if not reasons:
        # Recovery -> episode reset (journal-cursor fields preserved) so a
        # subsequent re-overload fires a SECOND row (test 7b).
        state.update(
            consecutive_hot=0, alerted=False, last_alert_load5=None, last_alert_reasons=None
        )
    else:
        state["consecutive_hot"] = consec
    if fire:
        top = _top_processes(dry_run=dry_run)
        # Rolling PRE-KILL snapshot: stored on EVERY fire (incl. mem-avail
        # single-tick fires) so a subsequent earlyoom kill is attributable —
        # a killed pid has no /proc/<pid>/cwd left to attribute post-hoc.
        state["last_top_snapshot"] = {"ts": now, "procs": top}
        top_txt = ", ".join(
            f"pid {p['pid']}"
            + (f" issue-{p['issue']}" if p.get("issue") is not None else "")
            + f" cpu {p['pcpu']}% rss {p['rss_mib']}M"
            for p in top[:CPU_GUARD_TOP_N]
        )
        print(
            f"vm-cpu PRESSURE: reasons={','.join(reasons)} load5={load5} nproc={ncpu} "
            f"psi_cpu_some={psi_cpu} psi_mem_full={psi_mem} mem_avail={mem_avail}% "
            f"consecutive={consec} — top: {top_txt or 'none'} "
            "(warn-only; see .claude/cache/cpu-guard-events.jsonl)",
            file=sys.stderr,
        )
        _append_cpu_guard_sidecar(
            {
                "kind": "vm-cpu-pressure",
                "band": "cpu-pressure",
                "reasons": reasons,
                "load1": load1,
                "load5": load5,
                "nproc": ncpu,
                "psi_cpu_some_avg10": psi_cpu,
                "psi_mem_full_avg10": psi_mem,
                "mem_avail_pct": mem_avail,
                "consecutive_hot": consec,
                "top_processes": top,
                **({} if kill_arm_ok else {"kill_arm": "unavailable"}),
            },
            dry_run,
        )
        _telegram_push(
            f"VM CPU/mem pressure ({','.join(reasons)}): load5 {load5} on {ncpu} cores, "
            f"mem avail {mem_avail}% — warn-only; top: {top_txt or 'n/a'}; "
            "details in .claude/cache/cpu-guard-events.jsonl",
            dry_run,
        )
        state.update(alerted=True, last_alert_load5=load5, last_alert_reasons=reasons)
        wrote = True
    if not dry_run:
        _save_cpu_guard_state(state)  # streak persistence NEEDS per-tick saves
    return wrote


# ─── END-OF-CPU-GUARD-BLOCK (task #849): never-kills scan boundary ───────────
# (tests/test_cpu_guard_pass.py::test_cpu_guard_never_kills greps from the
#  CPU-guard header sentinel above down to THIS line for process-mutation
#  tokens. New CPU-guard code must go ABOVE this line to stay inside the
#  scanned span. This string must stay UNIQUE in this file — the test asserts
#  count == 1; #1155.)


# ── post-hoc external-marker triage observer (#967) ─────────────────────────
#
# NON-GATING observer of the /issue Step 9 pre-dispatch external-marker
# triage duty (SKILL.md § Pre-dispatch external-marker triage; origin
# incident #779: 10 unread external audit markers, an 18-20h serial grid
# launched anyway). Re-runs the #889 enumerator's window semantics at recent
# HISTORICAL dispatch records (task_workflow.audit_dispatch_triage) and
# flags a missing / 'none' triage line against a non-empty candidate set.
# Observe/alert only — sidecar rows + deduped fail-soft digest pushes +
# capped epm:progress review nudges; NEVER mutates task status, stops a
# session, or blocks a dispatch (pinned by tests at BOTH the subprocess-argv
# and the in-process-mutator levels).

# Sweep scope: ACTIVE plus awaiting_promotion (9a-ter/9b follow-up dispatches
# happen on parked parents) plus blocked (a crash right after an untriaged
# launch parks there). Terminal / pre-plan statuses have no fresh dispatch to
# audit or no consumer for the flag.
_TRIAGE_OBSERVER_STATUSES = ACTIVE | {"awaiting_promotion", "blocked"}
# Lookback bounding first-run scans; the per-task cursor makes larger values
# pointless after tick 1 (the #911 48h recency precedent).
TRIAGE_OBSERVER_LOOKBACK_H = _env_float("EPM_TRIAGE_OBSERVER_LOOKBACK_H", 48.0, lo=1.0, hi=720.0)
# Adjacency window binding a machine-posted launch marker to its adjacent
# epm:progress triage note (pod provision+bootstrap runs ~10-20 min; 30 min
# covers with margin while still binding the note to ONE dispatch). DOUBLES
# as the MF2 maturity horizon by construction — it is exactly the window in
# which a compliant adjacent-next note may still land, so deferring
# evaluation by the same amount makes an early irreversible flag impossible.
TRIAGE_OBSERVER_ADJACENCY_S = _env_float(
    "EPM_TRIAGE_OBSERVER_ADJACENCY_S", 1800.0, lo=1.0, hi=86400.0
)
# The SKILL.md accepted-residual clause ("a marker posted in the SECONDS
# between the final enumerator run and the breadcrumb post"); 120 s is a
# generous superset of "seconds".
TRIAGE_OBSERVER_GRACE_S = _env_float("EPM_TRIAGE_OBSERVER_GRACE_S", 120.0, lo=0.0, hi=3600.0)
# Spam valve on the git-committing marker-post subprocess; the sidecar keeps
# full fidelity regardless. Overflow is PERMANENT sidecar+push-only (no
# deferred-marker queue — deferral adds state complexity exactly in the
# storm case where per-task markers would spam most).
TRIAGE_OBSERVER_MARKER_CAP = int(_env_float("EPM_TRIAGE_OBSERVER_MARKER_CAP", 5, lo=0, hi=100))
# Spam valve on the phone-push channel (#1167), mirroring the marker cap
# above: the first K warn pushes per tick go out individually; overflow is
# rolled into ONE "+N more" summary push at the end of the pass. Overflow
# is PERMANENT — an over-budget warn's individual push is never deferred
# to a later tick (same no-deferred-queue posture as the marker cap). The
# sidecar keeps full fidelity regardless; marker semantics are unaffected.
TRIAGE_OBSERVER_PUSH_CAP = int(_env_float("EPM_TRIAGE_OBSERVER_PUSH_CAP", 5, lo=0, hi=100))
_TRIAGE_OBSERVER_FLAGGED_KEY_CAP = 400  # newest dedup keys kept per issue


def _triage_observer_enabled() -> bool:
    """Kill switch: False when ``EPM_DISABLE_TRIAGE_OBSERVER`` is set truthy
    ("1"/"true"/"yes", case-insensitive). Default enabled. Mirrors
    :func:`_cpu_guard_enabled`."""
    raw = os.environ.get("EPM_DISABLE_TRIAGE_OBSERVER", "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _triage_observer_sidecar_path() -> Path:
    """DEDICATED triage-observer event stream (own stream for clean grep —
    the cpu-guard sidecar precedent)."""
    return PROJECT_ROOT / ".claude" / "cache" / "triage-observer-events.jsonl"


def _triage_observer_state_path() -> Path:
    """Singleton (deliberately NOT a per-issue GC target):
    ``{"<issue>": {"cursor_ts": str, "flagged": ["<record_ts>|<class>", ...]}}``."""
    return AUTONOMOUS_REGISTRY_DIR / "triage-observer.json"


def _load_triage_observer_state() -> dict:
    """``{}`` on missing/garbled state; every field read back goes through
    ``isinstance`` type-guards at the call sites (mirrors
    :func:`_load_cpu_guard_state`)."""
    path = _triage_observer_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_triage_observer_state(state: dict, dry_run: bool) -> None:
    """Atomic temp+rename write of the triage-observer state (fail-soft);
    ``dry_run`` performs zero writes."""
    if dry_run:
        print(f"  [dry-run] would save triage-observer state ({len(state)} issue entries)")
        return
    dest = _triage_observer_state_path()
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(dest)
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  triage-observer: state save failed: {exc}", file=sys.stderr)


def _append_triage_observer_sidecar(event: dict, dry_run: bool) -> None:
    """Append one JSON line to the triage-observer sidecar (fail-soft). A
    ``ts`` is stamped; ``dry_run`` reports only."""
    row = {"ts": datetime.now(tz=UTC).isoformat(), **event}
    line = json.dumps(row)
    if dry_run:
        print(f"  [dry-run] would append triage-observer sidecar row: {line[:160]}")
        return
    dest = _triage_observer_sidecar_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"  triage-observer: sidecar append failed: {exc}", file=sys.stderr)


def _triage_observer_iso_z(epoch: float) -> str:
    """Format an epoch as the events.jsonl ISO-8601 ``Z`` shape (second
    grain), so string thresholds compare cleanly against event ``ts``."""
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _triage_observer_nudge(v: dict) -> str:
    """Trap-safe review-nudge note text for one violation.

    MUST NOT contain the literal triage-line prefix (the note would become a
    window-closing boundary record for the enumerator) and MUST NOT
    lstrip-start with the breadcrumb prefix (pinned by test). The bracketed
    watcher sentinel makes it anti-liveness (STAGE_ANTILIVENESS_NOTE_
    SUBSTRINGS prefix match), so it never refreshes staleness clocks; its
    ``by="unknown"`` deliberately makes it a triage CANDIDATE at the task's
    next compute dispatch — the flag is itself the advisory."""
    desc = v.get("record_kind") or "record"
    if v.get("stage"):
        desc += f" stage={v['stage']}"
    problem = (
        "recorded a 'none' triage disposition"
        if v.get("violation") == "none-with-candidates"
        else "carries no pre-dispatch triage line"
    )
    kinds = ", ".join(v.get("candidate_kinds") or []) or "n/a"
    return (
        f"[autonomous_session_watch:triage-observer] post-hoc triage-duty review: the "
        f"compute dispatch record at {v.get('record_ts', '?')} ({desc}) {problem} while "
        f"{v.get('candidate_count', 0)} advisory candidate(s) were pending in its window "
        f"(kinds: {kinds}; {len(v.get('signature_hits') or [])} matching external-advisory "
        f"signatures). The enumerator over-approximates — this may be fine if every "
        f"candidate was self-posted. Observe-only, nothing was blocked. Please review "
        f"those markers and record their dispositions in the next dispatch note "
        f"(SKILL.md Step 9 entry guard). Deduped: posted once per violation."
    )


def decide_triage_observer_actions(
    violations: list[dict],
    flagged: set[str],
    marker_budget: int,
    *,
    push_budget: int | None = None,
) -> list[dict]:
    """PURE routing for triage-observer flags (unit-testable without IO).

    Drops already-flagged keys (key = ``f"{record_ts}|{violation}"``), orders
    warn-before-info (ties by record_ts), and attaches action fields: sidecar
    always; push only for ``warn`` while the per-tick push budget lasts
    (``push_budget=None`` = uncapped, the pre-#1167 semantics); marker only
    for ``warn`` while the per-tick marker budget lasts. The two budgets are
    INDEPENDENT. An over-MARKER-budget warn keeps ``push`` semantics and is
    STILL emitted (the caller marks it flagged) — permanently
    sidecar+push-only, its marker is NEVER deferred to a later tick
    (cap-overflow semantics, #967 plan §3 Q4). An over-PUSH-budget warn
    keeps ``marker`` semantics and gets ``push=False, push_suppressed=True``
    — the caller rolls suppressed warns into ONE end-of-pass summary push;
    the individual push is NEVER deferred (#1167). Every action carries
    ``push_suppressed`` (bool; False for info rows and in-budget warns).
    Never mutates ``flagged``."""
    fresh = [
        v for v in violations if f"{v.get('record_ts', '')}|{v.get('violation', '')}" not in flagged
    ]
    fresh.sort(key=lambda v: (0 if v.get("severity") == "warn" else 1, v.get("record_ts", "")))
    actions: list[dict] = []
    budget = marker_budget
    pbudget = push_budget
    for v in fresh:
        warn = v.get("severity") == "warn"
        marker = warn and budget > 0
        if marker:
            budget -= 1
        push = warn and (pbudget is None or pbudget > 0)
        if push and pbudget is not None:
            pbudget -= 1
        actions.append(
            {
                **v,
                "key": f"{v.get('record_ts', '')}|{v.get('violation', '')}",
                "sidecar": True,
                "push": push,
                "push_suppressed": warn and not push,
                "marker": marker,
            }
        )
    return actions


def _triage_observer_sweep_issue(
    id_str: str, meta: object, reg_root: Path, now: float, lookback_s: float
) -> int | None:
    """The issue id when a registry row is in the sweep status set with a
    fresh ``events.jsonl`` mtime (cost gate 1); ``None`` otherwise. The task
    path comes from the REGISTRY snapshot resolved against the registry's
    OWN root — never hand-built from cwd. Every gate fails soft (skip)."""
    if not isinstance(meta, dict) or meta.get("status") not in _TRIAGE_OBSERVER_STATUSES:
        return None
    try:
        issue = int(id_str)
    except (TypeError, ValueError):
        return None
    rel = meta.get("path")
    if not isinstance(rel, str) or not rel:
        return None
    try:
        mtime = (reg_root / rel / "events.jsonl").stat().st_mtime
    except OSError:
        return None
    if now - mtime > lookback_s:
        return None
    return issue


def _triage_observer_task_entry(state: dict, issue: int) -> tuple[str | None, set[str]]:
    """Type-guarded ``(cursor_ts, flagged-keys)`` read-back from the state
    singleton; a hand-edited / schema-drifted entry degrades to defaults."""
    raw_entry = state.get(str(issue))
    entry = raw_entry if isinstance(raw_entry, dict) else {}
    raw_cursor = entry.get("cursor_ts")
    cursor = raw_cursor if isinstance(raw_cursor, str) and raw_cursor else None
    raw_flagged = entry.get("flagged")
    flagged = {
        k for k in (raw_flagged if isinstance(raw_flagged, list) else []) if isinstance(k, str)
    }
    return cursor, flagged


def _triage_observer_emit(
    issue: int, actions: list[dict], flagged: set[str], dry_run: bool
) -> tuple[bool, int]:
    """Emit one action's channels (sidecar always; cap-decided push + capped
    marker for warn — both caps applied upstream in the decider) and mark it
    flagged. Returns ``(wrote_any, markers_posted)``.
    Mutates ``flagged`` — every emitted action is flagged, INCLUDING an
    over-budget warn (permanently sidecar+push-only, never deferred)."""
    wrote = False
    markers_posted = 0
    for a in actions:
        print(
            f"  triage-observer: #{issue} {a['violation']} ({a['severity']}) at "
            f"{a.get('record_ts', '?')} — candidates {a.get('candidate_count', 0)}, "
            f"signatures {len(a.get('signature_hits') or [])}"
        )
        _append_triage_observer_sidecar(
            {
                "issue": issue,
                **{
                    k: a.get(k)
                    for k in (
                        "record_ts",
                        "record_kind",
                        "stage",
                        "violation",
                        "severity",
                        "candidate_count",
                        "candidate_kinds",
                        "signature_hits",
                        "note_head",
                    )
                },
            },
            dry_run,
        )
        wrote = True
        if a["push"]:
            _telegram_push(
                f"triage-observer: #{issue} {a['violation']} at {a.get('record_ts', '?')} "
                f"({a.get('candidate_count', 0)} window candidates, "
                f"{len(a.get('signature_hits') or [])} external signatures) — observe-only; "
                "see .claude/cache/triage-observer-events.jsonl",
                dry_run,
            )
        if a["marker"]:
            markers_posted += 1
            _post_progress_marker(
                issue, _triage_observer_nudge(a), dry_run, label="triage-observer"
            )
        flagged.add(a["key"])
    return wrote, markers_posted


def triage_observer_pass(dry_run: bool) -> bool:
    """NON-GATING post-hoc audit of the pre-dispatch external-marker triage
    duty (#967; origin incident #779). Observe/alert only: appends rows to
    the dedicated sidecar, sends deduped fail-soft digest pushes — capped at
    ``TRIAGE_OBSERVER_PUSH_CAP`` individual warn pushes per tick, overflow
    rolled into ONE '+N more, see sidecar' summary push (#1167) — and posts
    capped ``epm:progress`` review nudges — NEVER mutates task status, stops
    a session, or blocks a dispatch. A warn beyond the per-tick marker cap
    stays sidecar+push-only forever; a warn beyond the per-tick push cap
    keeps its sidecar row (and marker, budget permitting) and is rolled into
    the tick's single summary push — no deferred queue on either channel.
    Both budgets thread CROSS-TASK (one shared budget per pass); pushes are
    consumed in issue-id-STRING order (``sorted(tasks.items())`` on string
    keys — marker-cap parity), not global recency. Fire-once:
    the dedup key ``(issue, record_ts, violation-class)`` persists in the
    state singleton, and the per-task cursor advances only past MATURED
    records (MF2) — so each dispatch record is evaluated exactly once, on
    the first tick after its compliance window closes. Daemon-independent;
    fail-soft throughout. Returns True when any sidecar row was written."""
    if not _triage_observer_enabled():
        print("  triage-observer: disabled via EPM_DISABLE_TRIAGE_OBSERVER; skipping")
        return False
    # Lazy in-process import (watcher convention): resolves THIS checkout's
    # helpers via the tests' sys.path shim / the editable install.
    from explore_persona_space.task_workflow import (
        audit_dispatch_triage,
        list_events,
        registry_path,
    )

    try:
        reg = json.loads(registry_path().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  triage-observer: registry read failed: {exc}", file=sys.stderr)
        return False
    tasks = reg.get("tasks") if isinstance(reg, dict) else None
    if not isinstance(tasks, dict):
        print("  triage-observer: registry has no tasks map; skipping", file=sys.stderr)
        return False

    now = time.time()
    lookback_s = TRIAGE_OBSERVER_LOOKBACK_H * 3600.0
    floor_ts = _triage_observer_iso_z(now - lookback_s)
    mature_before = _triage_observer_iso_z(now - TRIAGE_OBSERVER_ADJACENCY_S)
    # Resolve task paths against the registry's OWN root (never hand-built
    # from cwd; the REGISTRY snapshot carries `tasks/<status>/<id>` paths).
    reg_root = registry_path().parent.parent
    state = _load_triage_observer_state()
    wrote = False
    marker_budget = TRIAGE_OBSERVER_MARKER_CAP
    push_budget = TRIAGE_OBSERVER_PUSH_CAP
    suppressed_pushes = 0

    for id_str, meta in sorted(tasks.items()):
        issue = _triage_observer_sweep_issue(id_str, meta, reg_root, now, lookback_s)
        if issue is None:
            continue
        cursor, flagged = _triage_observer_task_entry(state, issue)
        # Cost gate 2: the per-task cursor (records at/before it were
        # already evaluated); the lookback floor bounds first-run scans.
        min_ts = max(cursor, floor_ts) if cursor else floor_ts
        try:
            events = list_events(issue)
        except Exception as exc:
            print(f"  triage-observer: events read failed for #{issue}: {exc}", file=sys.stderr)
            continue
        result = audit_dispatch_triage(
            events,
            adjacency_s=TRIAGE_OBSERVER_ADJACENCY_S,
            grace_s=TRIAGE_OBSERVER_GRACE_S,
            min_ts=min_ts,
            mature_before_ts=mature_before,
        )
        actions = decide_triage_observer_actions(
            result["violations"], flagged, marker_budget, push_budget=push_budget
        )
        task_wrote, markers_posted = _triage_observer_emit(issue, actions, flagged, dry_run)
        wrote = wrote or task_wrote
        marker_budget -= markers_posted
        # Mirror of the marker decrement; the decider guarantees pushes <=
        # push_budget, so this never goes negative.
        push_budget -= sum(1 for a in actions if a["push"])
        suppressed_pushes += sum(1 for a in actions if a["push_suppressed"])
        cursor_new = result.get("cursor_ts")
        if isinstance(cursor_new, str) and cursor_new:
            cursor = max(cursor, cursor_new) if cursor else cursor_new
        if cursor or flagged:
            state[str(issue)] = {
                "cursor_ts": cursor,
                # Keep the NEWEST keys under the cap (keys sort by record_ts).
                "flagged": sorted(flagged)[-_TRIAGE_OBSERVER_FLAGGED_KEY_CAP:],
            }

    # The ONE summary push per tick (#1167); per-tick only, no persisted
    # dedup state (the flagged-key fire-once upstream guarantees each
    # violation enters this count at most once, ever).
    if suppressed_pushes:
        _telegram_push(
            f"triage-observer: +{suppressed_pushes} more warn flag(s) this tick over the "
            f"per-tick push cap ({TRIAGE_OBSERVER_PUSH_CAP}) — all sidecar-recorded, never "
            "re-pushed; see .claude/cache/triage-observer-events.jsonl",
            dry_run,
        )

    # Self-prune entries once the issue leaves the sweep set for good.
    for key in list(state):
        meta = tasks.get(key)
        status = meta.get("status") if isinstance(meta, dict) else None
        if meta is None or status in {"completed", "archived"}:
            state.pop(key, None)
    _save_triage_observer_state(state, dry_run)
    return wrote


# ─── Verdict-disagree observer pass (task #1170; origin incident #825) ───────
#
# WHY: #825's `onpolicy-user-turn` review round carried a Claude PASS
# (head sentinel v5) vs Codex FAIL (bare version 7) disagreement whose
# reconciler was dispatched only after a manual catch — the round-number
# drift between the pair kinds made the disagreement easy to misread as a
# no-show. This pass mechanically detects the unreconciled shape: per
# doubled MARKER-MODE review site (workflow.yaml § ensemble_review), the
# LATEST round whose Claude + Codex durable verdicts disagree (pass-class
# vs fail-class) with no role-matched epm:review-reconcile and — for
# proximity-tier pairings only — no Codex no-show evidence. The pure
# predicate lives in task_workflow.unreconciled_disagreement_rounds; this
# block is the thin I/O wrapper, modeled on the triage observer (#967).
# Observe/alert only: sidecar rows + one deduped fail-soft push per
# (issue, site, round) — NEVER a task marker (deliberate divergence from
# the triage observer: this flag's consumer is a human, not the next
# dispatch), NEVER a status mutation, session stop, or dispatch block.
# KNOWN BENIGN-FIRE class: a Step 5c-bis mechanical-contract-only strip,
# a 9a-bis procedural strip, or a cap-5 all-stripped-continue resolves a
# PASS-vs-FAIL round WITHOUT a reconciler and logs to chat only, so it
# flags BY DESIGN (auditing orchestrator self-serve dismissals of a FAIL
# is in scope); the FAIL marker's own `**Blocker tags:**` line (an
# all-mechanical tag set) is the reader's one-glance disambiguator.

# Lookback bounding the per-tick events.jsonl scans (the triage observer's
# 48h precedent; dedup keys make each finding fire once regardless).
VERDICT_DISAGREE_LOOKBACK_H = _env_float("EPM_VERDICT_DISAGREE_LOOKBACK_H", 48.0, lo=1.0, hi=720.0)
# No flag until >= 60 min after the LATER verdict — reconciler spawn +
# adjudication needs time (#825's reconcile landed 10 min after the later
# verdict; 60 min is ~6x the observed latency). Deferral, not loss: the
# pair is re-evaluated every tick.
VERDICT_DISAGREE_GRACE_S = _env_float("EPM_VERDICT_DISAGREE_GRACE_S", 3600.0, lo=0.0, hi=86400.0)
# The two verdicts must land within 6h to count as one logical round —
# same-round pairs land minutes apart (worst realistic case a
# thrash-respawned reviewer, ~1-2h); distinct #825 epochs were >= 8h apart.
VERDICT_DISAGREE_PAIR_PROXIMITY_S = _env_float(
    "EPM_VERDICT_DISAGREE_PAIR_PROXIMITY_S", 21600.0, lo=60.0, hi=604800.0
)
# No-show evidence counts only from min(pair_ts) - 2h onward: covers a
# quota-skip note posted at round start plus a slow round, while excluding
# stale evidence from prior rounds (which would blind the observer forever
# after one Codex outage).
VERDICT_DISAGREE_EVIDENCE_LOOKBACK_S = _env_float(
    "EPM_VERDICT_DISAGREE_EVIDENCE_LOOKBACK_S", 7200.0, lo=0.0, hi=86400.0
)
_VERDICT_DISAGREE_FLAGGED_KEY_CAP = 400  # newest dedup keys kept per issue


def _verdict_disagree_enabled() -> bool:
    """Kill switch: False when ``EPM_DISABLE_VERDICT_DISAGREE_OBSERVER`` is
    set truthy ("1"/"true"/"yes", case-insensitive). Default enabled.
    Mirrors :func:`_triage_observer_enabled`."""
    raw = os.environ.get("EPM_DISABLE_VERDICT_DISAGREE_OBSERVER", "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _verdict_disagree_sidecar_path() -> Path:
    """DEDICATED verdict-disagree event stream (own stream for clean grep —
    the triage-observer/cpu-guard sidecar precedent)."""
    return PROJECT_ROOT / ".claude" / "cache" / "verdict-disagree-observer-events.jsonl"


def _verdict_disagree_state_path() -> Path:
    """Singleton (deliberately NOT a per-issue GC target):
    ``{"<issue>": {"flagged": ["<role>|<round_label>", ...]}}`` — no cursor
    needed: dedup keys alone give fire-once, and the latest-round
    evaluation is idempotent."""
    return AUTONOMOUS_REGISTRY_DIR / "verdict-disagree-observer.json"


def _load_verdict_disagree_state() -> dict:
    """``{}`` on missing/garbled state; every field read back goes through
    ``isinstance`` type-guards at the call sites (mirrors
    :func:`_load_triage_observer_state`)."""
    path = _verdict_disagree_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_verdict_disagree_state(state: dict, dry_run: bool) -> None:
    """Atomic temp+rename write of the verdict-disagree state (fail-soft);
    ``dry_run`` performs zero writes."""
    if dry_run:
        print(f"  [dry-run] would save verdict-disagree state ({len(state)} issue entries)")
        return
    dest = _verdict_disagree_state_path()
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(dest)
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  verdict-disagree: state save failed: {exc}", file=sys.stderr)


def _append_verdict_disagree_sidecar(event: dict, dry_run: bool) -> None:
    """Append one JSON line to the verdict-disagree sidecar (fail-soft). A
    ``ts`` is stamped; ``dry_run`` reports only."""
    row = {"ts": datetime.now(tz=UTC).isoformat(), **event}
    line = json.dumps(row)
    if dry_run:
        print(f"  [dry-run] would append verdict-disagree sidecar row: {line[:160]}")
        return
    dest = _verdict_disagree_sidecar_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"  verdict-disagree: sidecar append failed: {exc}", file=sys.stderr)


def _verdict_disagree_task_entry(state: dict, issue: int) -> set[str]:
    """Type-guarded flagged-keys read-back from the state singleton; a
    hand-edited / schema-drifted entry degrades to defaults (mirrors
    :func:`_triage_observer_task_entry`)."""
    raw_entry = state.get(str(issue))
    entry = raw_entry if isinstance(raw_entry, dict) else {}
    raw_flagged = entry.get("flagged")
    return {k for k in (raw_flagged if isinstance(raw_flagged, list) else []) if isinstance(k, str)}


def verdict_disagree_pass(dry_run: bool) -> bool:
    """NON-GATING observer of unreconciled doubled-site verdict
    disagreements (#1170; origin incident #825). Observe/alert only:
    sidecar rows + one deduped fail-soft :func:`_telegram_push` per
    (issue, site, round) — NEVER posts a task marker, mutates status,
    stops a session, or blocks a dispatch (pinned by tests at the
    subprocess-argv level). Fail-soft throughout: per-issue guards keep
    one bad task from masking the rest, and a top-level guard keeps an
    internal error from taking down the watcher tick (``main()`` calls
    passes bare — no outer try — so this pass carries its own).
    Daemon-independent (registry + events.jsonl reads only). Returns True
    when any sidecar row was written."""
    if not _verdict_disagree_enabled():
        print("  verdict-disagree: disabled via EPM_DISABLE_VERDICT_DISAGREE_OBSERVER; skipping")
        return False
    try:
        # Lazy in-process import (watcher convention): resolves THIS
        # checkout's helpers via the tests' sys.path shim / the editable
        # install.
        from explore_persona_space.task_workflow import (
            list_events,
            registry_path,
            unreconciled_disagreement_rounds,
        )

        try:
            reg = json.loads(registry_path().read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  verdict-disagree: registry read failed: {exc}", file=sys.stderr)
            return False
        tasks = reg.get("tasks") if isinstance(reg, dict) else None
        if not isinstance(tasks, dict):
            print("  verdict-disagree: registry has no tasks map; skipping", file=sys.stderr)
            return False

        now = time.time()
        lookback_s = VERDICT_DISAGREE_LOOKBACK_H * 3600.0
        # Resolve task paths against the registry's OWN root (never
        # hand-built from cwd).
        reg_root = registry_path().parent.parent
        state = _load_verdict_disagree_state()
        wrote = False

        for id_str, meta in sorted(tasks.items()):
            # Sweep scope + recency: REUSE the triage observer's enumerator
            # (in-repo tool-reuse rule) — its status set
            # ACTIVE | {awaiting_promotion, blocked} and events.jsonl-mtime
            # recency gate fit this pass too: review verdicts fire at
            # active statuses, awaiting_promotion covers the final
            # clean-result rounds' park, blocked covers a crash right
            # after a review round.
            issue = _triage_observer_sweep_issue(id_str, meta, reg_root, now, lookback_s)
            if issue is None:
                continue
            flagged = _verdict_disagree_task_entry(state, issue)
            try:
                events = list_events(issue)
                findings = unreconciled_disagreement_rounds(
                    events,
                    now_ts=now,
                    grace_s=VERDICT_DISAGREE_GRACE_S,
                    pair_proximity_s=VERDICT_DISAGREE_PAIR_PROXIMITY_S,
                    evidence_lookback_s=VERDICT_DISAGREE_EVIDENCE_LOOKBACK_S,
                )
            except Exception as exc:  # per-issue fail-soft
                print(f"  verdict-disagree: #{issue} evaluation failed: {exc}", file=sys.stderr)
                continue
            for finding in findings:
                if finding["key"] in flagged:
                    continue  # fire-once dedup
                print(
                    f"  verdict-disagree: #{issue} {finding['role']} "
                    f"{finding['round_label']} — Claude {finding['claude_class']} vs "
                    f"Codex {finding['codex_class']}"
                )
                _append_verdict_disagree_sidecar({"issue": issue, **finding}, dry_run)
                _telegram_push(
                    f"verdict-disagree-observer: #{issue} {finding['role']} "
                    f"{finding['round_label']} — Claude {finding['claude_class'].upper()} "
                    f"vs Codex {finding['codex_class'].upper()}, no role-matched reconcile "
                    "+ no no-show evidence (#825 shape; may be a sanctioned 5c-bis/9a-bis "
                    "strip — check the FAIL marker's Blocker tags line) — observe-only; "
                    "see .claude/cache/verdict-disagree-observer-events.jsonl",
                    dry_run,
                )
                flagged.add(finding["key"])
                wrote = True
            if flagged:
                state[str(issue)] = {
                    # Keep the NEWEST keys under the cap (sorted order is a
                    # stable tie-break; the cap is a runaway guard, not an
                    # eviction policy — 400 findings per issue never happens).
                    "flagged": sorted(flagged)[-_VERDICT_DISAGREE_FLAGGED_KEY_CAP:],
                }

        # Self-prune entries once the issue leaves the sweep set for good
        # (mirrors triage_observer_pass).
        for key in list(state):
            meta = tasks.get(key)
            status = meta.get("status") if isinstance(meta, dict) else None
            if meta is None or status in {"completed", "archived"}:
                state.pop(key, None)
        _save_verdict_disagree_state(state, dry_run)
        return wrote
    except Exception as exc:  # top-level fail-soft: never take down the tick
        print(f"  verdict-disagree: pass failed (fail-soft): {exc}", file=sys.stderr)
        return False


# ─── Auth-outage guard pass (task #1027) — fleet respawn suppression ─────────
#
# WHY: 2026-07-03 incident — an Anthropic auth outage (poisoned Claude CLI
# credential, recovered by /login) killed every freshly spawned session on
# arrival; the watcher's respawn arms churned die-on-arrival sessions across
# the fleet for hours (per-task caps bound per-ISSUE churn, not the fleet).
# This pass detects the fleet-wide instant-freeze-respawn signature from
# state the watcher already owns (registry spawned_at + its own spawn
# results), suppresses EVERY watcher spawn arm while an episode is active,
# fires ONE push per episode, and probes recovery with a CANARY respawn (the
# canary IS a real session spawn, so it probes the exact CLI-credential auth
# path real sessions use — a watcher-side ANTHROPIC_API_KEY probe would test
# the WRONG credential). FAIL-OPEN by design: any guard failure logs a
# warning and behaves as "no outage" (respawns proceed), and an episode
# self-expires at a hard TTL — a false suppression (fleet-wide crash-recovery
# blackout) is strictly worse than the churn this pass fixes.

# Rolling detection window for instant-freeze respawn events.
AUTH_OUTAGE_WINDOW_S = _env_float("EPM_AUTH_OUTAGE_WINDOW_MIN", 180.0, lo=10.0, hi=10080.0) * 60
# Max predecessor lifetime for an instant-freeze event: die-on-arrival respawn
# cycle = RESPAWN_SPAWN_GRACE_S (15 min) + 2 misses x 10-min cron ~= 25-45 min;
# 60 covers it with margin while a healthy multi-hour session never qualifies.
# The lo=45 clamp encodes the plan's deviation fence (Codex-stats S8): a value
# below the ~45-min die-on-arrival ceiling breaks the AC1 replay shape, so an
# env override under 45 falls back to the default.
AUTH_OUTAGE_FRESH_DEATH_S = (
    _env_float("EPM_AUTH_OUTAGE_FRESH_DEATH_MIN", 60.0, lo=45.0, hi=360.0) * 60
)
# Freeze events / distinct issues to trigger (>=2 issues separates a fleet
# cause from a per-issue crash loop owned by the per-task caps).
AUTH_OUTAGE_MIN_EVENTS = int(_env_float("EPM_AUTH_OUTAGE_MIN_EVENTS", 3, lo=1, hi=100))
AUTH_OUTAGE_MIN_ISSUES = int(_env_float("EPM_AUTH_OUTAGE_MIN_ISSUES", 2, lo=1, hi=100))
# Canary cadence: one probe respawn per interval; survival >= this resolves
# the episode (2 ticks — the standing 2-consecutive-checks corroboration).
AUTH_OUTAGE_CANARY_INTERVAL_S = (
    _env_float("EPM_AUTH_OUTAGE_CANARY_INTERVAL_MIN", 30.0, lo=10.0, hi=720.0) * 60
)
AUTH_OUTAGE_CANARY_SURVIVAL_S = (
    _env_float("EPM_AUTH_OUTAGE_CANARY_SURVIVAL_MIN", 20.0, lo=5.0, hi=720.0) * 60
)
# Hard fail-open TTL: an episode older than this expires with a push even if
# the canary machinery is logically wedged (mirrors the takeover-sentinel
# fail-open posture, #866/#903).
AUTH_OUTAGE_MAX_EPISODE_S = _env_float("EPM_AUTH_OUTAGE_MAX_EPISODE_H", 6.0, lo=1.0, hi=48.0) * 3600

# Arms that may CONSUME the canary token (issue-registry spawns whose fresh
# happy_session_id is readable at issue-<N>.json). The campaign arm is
# EXCLUDED by design: campaign sessions register at campaign-<N>.json, so the
# canary liveness read would wedge the episode to TTL (plan MF-3).
_AUTH_OUTAGE_CANARY_ARMS = frozenset(
    {"crash", "stalled", "orphan", "infra-drain", "capacity-retry"}
)

# Best-effort push-text enrichment only — NEVER the trigger (log formats are
# fragile; the trigger is derived purely from watcher-owned spawn state).
_AUTH_OUTAGE_EVIDENCE_SIGNATURES = (
    "Not a valid API key",
    "invalid x-api-key",
    "authentication_error",
    "OAuth token has expired",
)

# Single-flight per tick (the watch.lock flock guarantees one watcher
# instance): True = exactly ONE canary respawn may proceed this tick.
_AUTH_CANARY_TOKEN = False


def _auth_outage_enabled() -> bool:
    """Kill switch: False when ``EPM_DISABLE_AUTH_OUTAGE_GUARD`` is set truthy
    ("1"/"true"/"yes", case-insensitive). Default enabled. Mirrors
    :func:`_cpu_guard_enabled`."""
    raw = os.environ.get("EPM_DISABLE_AUTH_OUTAGE_GUARD", "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _auth_outage_state_path() -> Path:
    """Fleet-level singleton under AUTONOMOUS_REGISTRY_DIR (singletons are
    never GC'd — :func:`_gc_target_paths` sweeps per-issue files only)."""
    return AUTONOMOUS_REGISTRY_DIR / "auth-outage.json"


def _load_auth_outage_state() -> dict:
    """``{}`` on missing/garbled state (FAIL-OPEN: a fresh empty state means
    "no outage", so a corrupt file can never suppress spawns). Every field
    read back goes through ``isinstance`` type-guards at the call sites."""
    path = _auth_outage_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  auth-outage: state unreadable (fail-open, fresh state): {exc}", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


def _save_auth_outage_state(state: dict) -> None:
    """Atomic temp+rename write of the auth-outage state (fail-soft)."""
    dest = _auth_outage_state_path()
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(dest)
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  auth-outage: state save failed: {exc}", file=sys.stderr)


def _auth_outage_sidecar_path() -> Path:
    """DEDICATED auth-outage event stream (domain-separated for clean grep —
    the cpu-guard sidecar precedent)."""
    return PROJECT_ROOT / ".claude" / "cache" / "auth-outage-events.jsonl"


def _append_auth_outage_sidecar(event: dict, dry_run: bool) -> None:
    """Append one JSON line per episode transition (trigger / canary-armed /
    canary-failed / resolve / expire) to the auth-outage sidecar (fail-soft).
    A ``ts`` is stamped; ``dry_run`` reports only."""
    row = {"ts": datetime.now(tz=UTC).isoformat(), **event}
    line = json.dumps(row)
    if dry_run:
        print(f"  [dry-run] would append auth-outage sidecar row: {line[:160]}")
        return
    dest = _auth_outage_sidecar_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"  auth-outage: sidecar append failed: {exc}", file=sys.stderr)


def _auth_outage_pruned_events(events: object, now: float) -> list[dict]:
    """Well-formed spawn events within 2x the detection window (pruned on
    every state write so the singleton stays small)."""
    horizon = now - 2 * AUTH_OUTAGE_WINDOW_S
    out: list[dict] = []
    if not isinstance(events, list):
        return out
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = e.get("ts")
        if isinstance(ts, int | float) and ts >= horizon:
            out.append(e)
    return out


def _auth_outage_evidence() -> str:
    """Best-effort auth-signature grep over the last 64 KB of the newest 3
    files under ``~/.happy/logs/`` (push-text enrichment ONLY, never the
    trigger — §3.6). Wholly fail-soft: any exception degrades to
    ``"churn-only"``."""
    try:
        log_dir = Path.home() / ".happy" / "logs"
        files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
        for f in files:
            with open(f, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 65536))
                tail = fh.read().decode("utf-8", errors="replace")
            for sig in _AUTH_OUTAGE_EVIDENCE_SIGNATURES:
                if sig in tail:
                    return f"auth-string: {sig}"
    except Exception as exc:  # fail-soft: enrichment must never block the trigger
        print(f"  auth-outage: evidence grep failed (churn-only): {exc}", file=sys.stderr)
    return "churn-only"


def _registry_entry_field(issue: int, field: str) -> object:
    """One field from ``issue-<N>.json`` (fail-soft -> ``None``)."""
    try:
        entry = json.loads((AUTONOMOUS_REGISTRY_DIR / f"issue-{issue}.json").read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return entry.get(field) if isinstance(entry, dict) else None


def _registry_spawned_at(issue: int) -> float | None:
    """``spawned_at`` from the issue's autonomous registry entry (fail-soft
    -> ``None``; the §13.2 fallback for arms whose helper does not carry the
    registry-entry dict)."""
    val = _registry_entry_field(issue, "spawned_at")
    return float(val) if isinstance(val, int | float) else None


def _registry_happy_sid(issue: int) -> str | None:
    """``happy_session_id`` from the issue's autonomous registry entry
    (fail-soft -> ``None``)."""
    sid = _registry_entry_field(issue, "happy_session_id")
    return sid if isinstance(sid, str) and sid else None


def _coerce_ts(val: object) -> float | None:
    """Epoch float from a registry field, or ``None`` on any other shape."""
    return float(val) if isinstance(val, int | float) and not isinstance(val, bool) else None


def _auth_canary_alive(state: dict, live_ids: set[str] | None) -> bool | None:
    """Liveness of the outstanding canary (#1027 MF-3): the PERSISTED
    ``canary_session_id`` in ``live_ids`` — never a registry re-read FOR
    LIVENESS (a replaced entry's sid is a different session). The registry
    IS read for INVALIDATION only: a missing registration, or one whose sid
    no longer matches the persisted canary sid, means the canary was
    superseded / terminal-parked — return False (clear + re-arm on the
    caller side; never a false resolve). ``None`` = inconclusive (daemon
    down)."""
    sid = state.get("canary_session_id")
    issue = state.get("canary_issue")
    if not isinstance(sid, str) or not sid:
        return False  # an unbound canary can never resolve the episode
    if isinstance(issue, int) and _registry_happy_sid(issue) != sid:
        return False  # registration gone/replaced -> invalidated
    if live_ids is None:
        return None
    return sid in live_ids


def _auth_outage_spawn_gate(issue: int, arm: str, *, dry_run: bool = False) -> str | None:
    """Per-spawn suppression gate (#1027), consulted by EVERY watcher spawn
    arm. ``None`` = allow; ``"auth-outage"`` = suppress (helpers return the
    #843 ``"suppressed"`` tri-state so callers book nothing).

    During an ACTIVE episode the gate suppresses every spawn EXCEPT:

    - the ONE canary per interval: with the module canary token armed and
      ``arm`` in :data:`_AUTH_OUTAGE_CANARY_ARMS`, the gate consumes the
      token, persists ``canary_pending`` (the cross-tick claim the record
      hook binds on), and allows — after a canary-failed it round-robins by
      skipping the LAST failed canary issue once;
    - an in-flight canary unit for this issue (a fresh ``canary_pending`` —
      e.g. the stalled fence consumed the token on its stop tick and the
      verified-dead spawn lands a tick later);
    - an expired/garbled episode (the TTL binds HERE too, so a wedged
      ``auth_outage_pass`` can never suppress past the fail-open TTL).

    FAIL-OPEN: any internal error logs a warning and allows the spawn."""
    global _AUTH_CANARY_TOKEN
    try:
        if not _auth_outage_enabled():
            return None
        state = _load_auth_outage_state()
        if not state.get("active"):
            return None
        now = time.time()
        started = state.get("started_ts")
        if not isinstance(started, int | float) or now - started >= AUTH_OUTAGE_MAX_EPISODE_S:
            # Second fail-open layer: even if the pass is wedged and never
            # runs the "expire" transition, suppression ends at the TTL.
            return None
        pending = state.get("canary_pending")
        if (
            isinstance(pending, dict)
            and pending.get("issue") == issue
            and isinstance(pending.get("ts"), int | float)
            and 0 <= now - pending["ts"] <= AUTH_OUTAGE_CANARY_INTERVAL_S
        ):
            print(f"  auth-outage: canary in flight for issue #{issue}; allowing (arm={arm})")
            return None
        if _AUTH_CANARY_TOKEN and arm in _AUTH_OUTAGE_CANARY_ARMS:
            if state.get("skip_last_canary_once") and state.get("last_canary_issue") == issue:
                # Round-robin after a canary-failed: skip the failed issue
                # ONCE so one broken issue cannot starve the canary channel.
                if not dry_run:
                    state["skip_last_canary_once"] = False
                    _save_auth_outage_state(state)
                print(f"  auth-outage: round-robin skip of last failed canary issue #{issue}")
                return "auth-outage"
            if dry_run:
                print(
                    f"  [dry-run] auth-outage: would consume the canary token "
                    f"for issue #{issue} (arm={arm})"
                )
                return None
            _AUTH_CANARY_TOKEN = False
            state["canary_pending"] = {"issue": issue, "arm": arm, "ts": now}
            _save_auth_outage_state(state)
            print(f"  auth-outage: CANARY spawn allowed for issue #{issue} (arm={arm})")
            return None
        print(f"  auth-outage: SUPPRESSED {arm} spawn for issue #{issue} (episode active)")
        return "auth-outage"
    except Exception as exc:  # FAIL-OPEN: never crash (or block) a spawn arm
        print(f"  auth-outage: gate error (fail-open, allowing spawn): {exc}", file=sys.stderr)
        return None


def _auth_outage_record_spawn(issue: int, arm: str, prev_spawned_at: float | None) -> None:
    """Record one REAL watcher-issued spawn (called only on the ``"spawned"``
    / campaign-True result, so never under dry-run): appends the
    ``{issue, ts, arm, prev_spawned_at}`` event the trigger predicate reads,
    and — when this spawn is the pending canary — binds the canary identity
    (MF-3): ``canary_issue`` / ``canary_session_id`` / ``canary_ts`` are
    persisted ONLY here, from the FRESH registry entry the spawn just wrote
    (a ``"failed"`` spawn leaves them unset, so the token re-arms next
    interval). Fail-soft no-op on any internal error."""
    try:
        if not _auth_outage_enabled():
            return
        state = _load_auth_outage_state()
        now = time.time()
        events = _auth_outage_pruned_events(state.get("events"), now)
        events.append({"issue": issue, "ts": now, "arm": arm, "prev_spawned_at": prev_spawned_at})
        state["events"] = events
        pending = state.get("canary_pending")
        if state.get("active") and isinstance(pending, dict) and pending.get("issue") == issue:
            sid = _registry_happy_sid(issue)
            state["canary_issue"] = issue
            state["canary_session_id"] = sid
            state["canary_ts"] = now
            state["last_canary_ts"] = now
            state["last_canary_issue"] = issue
            state["canary_pending"] = None
            _append_auth_outage_sidecar(
                {
                    "transition": "canary-armed",
                    "issue": issue,
                    "arm": arm,
                    "canary_session_id": sid,
                },
                False,
            )
            print(f"  auth-outage: canary #{issue} spawned (sid={sid}); survival window starts")
        _save_auth_outage_state(state)
    except Exception as exc:  # fail-soft: a lost event must never break a spawn
        print(f"  auth-outage: record error (fail-soft, event dropped): {exc}", file=sys.stderr)


def _auth_outage_end_episode(state: dict, now: float) -> None:
    """Clear every episode field and stamp the MF-1 watermark. Events are
    KEPT (the watermark, not deletion, blocks stale re-trigger: a genuinely
    persistent outage re-accumulates NEW qualifying events and legitimately
    re-triggers)."""
    state.update(
        active=False,
        started_ts=None,
        trigger_pushed=False,
        resolve_pushed=False,
        canary_issue=None,
        canary_session_id=None,
        canary_ts=None,
        last_canary_ts=None,
        last_canary_issue=None,
        skip_last_canary_once=False,
        canary_pending=None,
        evidence="",
        last_episode_end_ts=now,
    )


def _auth_outage_tick_active(
    state: dict, now: float, dry_run: bool, live_ids: set[str] | None
) -> None:
    """One tick of an ACTIVE episode: clear a stale canary claim, resolve
    canary liveness, apply :func:`decide_auth_outage_canary`, and act on the
    verdict. Mutates ``state`` in place; the caller saves it."""
    global _AUTH_CANARY_TOKEN
    pending = state.get("canary_pending")
    if isinstance(pending, dict):
        pts = pending.get("ts")
        if not isinstance(pts, int | float) or now - pts > AUTH_OUTAGE_CANARY_INTERVAL_S:
            # A consumed token whose spawn never landed (helper kept failing
            # / the fence unit stalled): release the claim so the next
            # interval can arm a fresh canary.
            state["canary_pending"] = None
    canary_alive = (
        _auth_canary_alive(state, live_ids)
        if isinstance(state.get("canary_ts"), int | float)
        else None
    )
    action = decide_auth_outage_canary(
        state,
        now,
        canary_alive=canary_alive,
        canary_interval_s=AUTH_OUTAGE_CANARY_INTERVAL_S,
        canary_survival_s=AUTH_OUTAGE_CANARY_SURVIVAL_S,
        max_episode_s=AUTH_OUTAGE_MAX_EPISODE_S,
    )
    if action == "expire":
        print("  auth-outage: episode EXPIRED at the fail-open TTL; respawns resume")
        _append_auth_outage_sidecar(
            {"transition": "expire", "started_ts": state.get("started_ts")}, dry_run
        )
        _telegram_push(
            "AUTH-OUTAGE GUARD EXPIRED after "
            f"{AUTH_OUTAGE_MAX_EPISODE_S / 3600:.0f}h without recovery — watcher "
            "respawns resume fail-open; investigate auth manually",
            dry_run,
        )
        _auth_outage_end_episode(state, now)
        return
    if action == "resolve":
        issue = state.get("canary_issue")
        print(f"  auth-outage: RESOLVED — canary #{issue} survived the window")
        _append_auth_outage_sidecar({"transition": "resolve", "canary_issue": issue}, dry_run)
        if not state.get("resolve_pushed"):
            _telegram_push(
                f"AUTH OUTAGE RESOLVED — canary #{issue} survived >= "
                f"{AUTH_OUTAGE_CANARY_SURVIVAL_S / 60:.0f} min; watcher respawns resumed",
                dry_run,
            )
            state["resolve_pushed"] = True
        _auth_outage_end_episode(state, now)
        return
    if action == "canary-failed":
        issue = state.get("canary_issue")
        print(f"  auth-outage: canary #{issue} DIED — outage persists; re-arming next interval")
        _append_auth_outage_sidecar({"transition": "canary-failed", "canary_issue": issue}, dry_run)
        state["canary_issue"] = None
        state["canary_session_id"] = None
        state["canary_ts"] = None
        state["canary_pending"] = None
        state["skip_last_canary_once"] = True
        return
    if action == "arm-canary":
        if dry_run:
            print("  [dry-run] auth-outage: would arm the canary token (one respawn this tick)")
            return
        print("  auth-outage: arming the canary token (ONE probe respawn allowed this tick)")
        _AUTH_CANARY_TOKEN = True
        return
    # "hold": episode continues; nothing to do this tick.
    print("  auth-outage: episode active (spawns suppressed); holding")


def _auth_outage_tick_inactive(state: dict, now: float, dry_run: bool) -> None:
    """One tick with NO active episode: apply the trigger predicate over the
    recorded spawn events; on True, activate the episode + fire the one
    trigger push with evidence-conditioned advice. Mutates ``state`` in
    place; the caller saves it."""
    events = state.get("events")
    if not isinstance(events, list) or not events:
        return
    last_end = state.get("last_episode_end_ts")
    last_end_f = float(last_end) if isinstance(last_end, int | float) else 0.0
    if not decide_auth_outage_trigger(
        events,
        now,
        window_s=AUTH_OUTAGE_WINDOW_S,
        fresh_death_s=AUTH_OUTAGE_FRESH_DEATH_S,
        min_freeze_events=AUTH_OUTAGE_MIN_EVENTS,
        min_distinct_issues=AUTH_OUTAGE_MIN_ISSUES,
        last_episode_end_ts=last_end_f,
    ):
        return
    freeze = _auth_outage_freeze_subset(
        events,
        now,
        window_s=AUTH_OUTAGE_WINDOW_S,
        fresh_death_s=AUTH_OUTAGE_FRESH_DEATH_S,
        last_episode_end_ts=last_end_f,
    )
    issues = sorted({e["issue"] for e in freeze})
    span_min = (now - min(e["ts"] for e in freeze)) / 60
    evidence = _auth_outage_evidence()
    already_pushed = bool(state.get("trigger_pushed"))
    state["active"] = True
    state["started_ts"] = now
    state["evidence"] = evidence
    state["resolve_pushed"] = False
    print(
        f"  auth-outage: TRIGGERED — {len(freeze)} instant-freeze respawns across "
        f"issues {issues} in {span_min:.0f} min (evidence: {evidence}); watcher "
        f"respawns SUPPRESSED"
    )
    _append_auth_outage_sidecar(
        {
            "transition": "trigger",
            "freeze_events": freeze,
            "distinct_issues": issues,
            "evidence": evidence,
        },
        dry_run,
    )
    if not already_pushed:
        if evidence.startswith("auth-string"):
            advice = "Check claude auth (/login) on the VM."
        else:
            advice = (
                "Cause unconfirmed — check claude auth (/login), earlyoom/memory, "
                "disk, and recent workflow-surface edits."
            )
        _telegram_push(
            f"AUTH OUTAGE SUSPECTED: {len(freeze)} instant-freeze respawns across "
            f"issues {issues} in {span_min:.0f} min — WATCHER respawns SUPPRESSED "
            f"(canary every {AUTH_OUTAGE_CANARY_INTERVAL_S / 60:.0f} min; PM/manual "
            f"spawns unaffected). Evidence: {evidence}. {advice}",
            dry_run,
        )
        state["trigger_pushed"] = True


def auth_outage_pass(dry_run: bool, *, daemon_reachable: bool, live_ids: set[str] | None) -> None:
    """Fleet-level auth-outage guard (#1027): arm/refresh the respawn
    suppression BEFORE any spawn arm runs this tick.

    Daemon-INDEPENDENT for episode bookkeeping (trigger + the fail-open TTL
    advance during daemon flaps); the canary-survival read degrades to
    ``"hold"`` when ``live_ids`` is unavailable — during a daemon outage no
    spawn pass runs anyway, so nothing is lost. ``dry_run`` performs ZERO
    state writes and ZERO pushes (decisions are logged only). FAIL-OPEN: any
    internal error logs a warning and behaves as "no outage"."""
    global _AUTH_CANARY_TOKEN
    _AUTH_CANARY_TOKEN = False  # never carry a token across pass invocations
    try:
        if not _auth_outage_enabled():
            print("auth-outage: disabled via EPM_DISABLE_AUTH_OUTAGE_GUARD; skipping")
            return
        state = _load_auth_outage_state()
        now = time.time()
        state["events"] = _auth_outage_pruned_events(state.get("events"), now)
        print(
            f"auth-outage: {'episode ACTIVE' if state.get('active') else 'no active episode'}; "
            f"{len(state['events'])} recorded spawn event(s) in the pruning horizon "
            f"(daemon_reachable={daemon_reachable})"
        )
        if state.get("active"):
            _auth_outage_tick_active(state, now, dry_run, live_ids)
        else:
            _auth_outage_tick_inactive(state, now, dry_run)
        if not dry_run:
            _save_auth_outage_state(state)
    except Exception as exc:  # FAIL-OPEN: a guard bug must never kill the tick
        print(
            f"  auth-outage: pass error (fail-open — no suppression armed): {exc}",
            file=sys.stderr,
        )


def _status_class(status: str | None, latest_progress_ts: float | None, now: float) -> str:
    """Classify a RUNNING managed pod's task status for :func:`decide_pod_safety`.

    Returns ``"auto-stop-done"`` / ``"pod-active-stale"`` / ``"pod-active-fresh"``
    / ``"other"``. ``"auto-stop-done"`` covers :data:`POD_SAFETY_AUTO_STOP` —
    the DONE statuses plus user-paused ``on_hold`` (the #919 pause-window
    escaped pod, #980). ``status`` of ``None`` (task unreadable) is ``"other"`` —
    never auto-stopped. A pod-active task is ``stale`` when its newest real
    progress marker is older than :data:`ALERT_STALE_HOURS`, OR when there is no
    real progress marker at all (``latest_progress_ts is None``) — a pod-active
    task with zero progress markers is itself a signal worth alerting on.
    """
    if status is None:
        return "other"
    if status in POD_SAFETY_AUTO_STOP:
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
    measuring) AND that is not a deliberate session-stop record (lstripped
    note prefix ``"deliberate-stop "`` OR ``by == "spawn_session-stop"`` — a
    session's death record is anti-liveness, never progress; #990, precedent
    #949/#810 in ``task_workflow.stage_dispatch_should_skip``). Returns
    ``None`` when there is no such marker.
    """
    best: float | None = None
    for ev in events:
        if ev.get("kind") not in _PROGRESS_KINDS:
            continue
        note = ev.get("note") or ""
        if any(sentinel in note for sentinel in _WATCHER_NOTE_SENTINELS):
            continue  # a watcher-posted alert — not real progress
        # Anti-liveness (#990; precedent #949/#810): a deliberate session
        # stop is the death record of the task's driver, not progress —
        # counting it would refresh the very staleness clocks that should
        # react to the stop. Same predicate as
        # task_workflow.stage_dispatch_should_skip (~line 1463): the
        # lstripped "deliberate-stop " note PREFIX (also catches PM-posted
        # stop records, which use by="pm-chat" — research-pm.md ~719/734)
        # OR by == "spawn_session-stop" (catches note-text drift from the
        # cmd_stop emitter). Prefix-boundary: a note merely MENTIONING
        # deliberate-stop mid-text still counts as progress.
        if note.lstrip().startswith("deliberate-stop ") or ev.get("by") == "spawn_session-stop":
            continue
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


def _wedge_keep_running(issue: int) -> bool | str:
    """Tri-state keep-running read for the #692 wedge arm (MF2).

    Returns ``True`` (the ``keep-running`` tag is present) | ``False`` (the tag
    read succeeded and is absent) | ``"unknown"`` (the tag read FAILED —
    subprocess error, non-zero rc, JSON parse error). The shared
    :func:`_task_keep_running` collapses all three of those failure modes to
    ``False``, indistinguishable from a genuinely untagged task — safe for the
    status-class DONE arm (it only auto-stops DONE-status pods, where the user
    has far less reason to keep-running) but NOT for the wedge arm, which
    auto-stops LIVE-WORK (``running`` / ``approved`` / ...) pods. The wedge
    AUTO-STOP fires only on the literal ``False``; ``"unknown"`` routes to
    ALERT-only so a tagged live-work pod whose tag lookup is transiently failing
    is NEVER auto-stopped (which would silently override the user's explicit tag).

    Uses the same ``task.py view --json`` subprocess isolation +
    ``cwd=PROJECT_ROOT`` as :func:`_task_keep_running` (the watcher runs from
    PROJECT_ROOT on ``main``, satisfying the task.py branch-guard). Called only
    on the wedge arm's confirmed-past-K branch, so the extra subprocess is paid
    only for a matured wedge candidate, not every RUNNING pod every tick."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "view", str(issue), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return "unknown"
    tags = (data.get("frontmatter") or {}).get("tags") or []
    return isinstance(tags, list) and "keep-running" in tags


def _wedge_inputs_safe(issue: int) -> bool:
    """True iff the wedged run's recoverable inputs are verified on HF (#692).

    Gates the watcher's reversible AUTO-STOP exactly as #689 fix (b) gates the
    poller's IRREVERSIBLE terminate, reusing the SAME per-cell three-state gate
    ``backend_poll._wedged_run_inputs_on_hf``. The run handle is read from the
    persisted sidecar (``.claude/cache/issue-<N>-handle.json``).

    Fail-CLOSED: a missing / unreadable handle, an HF-listing transport error,
    an import failure, or ANY exception -> ``False`` (ALERT-only, never an unsafe
    stop). For a non-#664 issue ``_wedged_run_inputs_on_hf`` returns ``ok=True``
    (the adapters-only path is inline-verified), so the gate degrades to "safe to
    stop" there — correct, because a reversible STOP loses nothing when there are
    no per-cell artifacts to strand. This is the single most important safety
    property of the wedge AUTO-STOP: every uncertainty path routes to ALERT-only."""
    try:
        from backend_poll import _wedged_run_inputs_on_hf

        from explore_persona_space.backends.issue_dispatch import (
            read_handle_sidecar,
            resolve_handle_sidecar_path,
        )

        path, _probed = resolve_handle_sidecar_path(issue)
        if not path.exists():
            return False  # no handle -> cannot gate -> ALERT-only
        handle = read_handle_sidecar(path)
        gate = _wedged_run_inputs_on_hf(issue, handle)
        return bool(gate.ok)
    except Exception as exc:  # transport / parse / import -> fail-closed
        print(
            f"  wedge: inputs-on-HF gate unavailable for #{issue} "
            f"({type(exc).__name__}: {exc}); ALERT-only (no auto-stop)",
            file=sys.stderr,
        )
        return False


def _clear_wedge_state(issue: int, pod_id: str) -> None:
    """Clear the #692 wedge fields for ``issue`` while keeping the
    pod-incarnation ``first_seen`` GC anchor intact (MF1 onset-clock clear).

    Called on the NOT-wedged branch of :func:`_process_pod` (a port re-appeared,
    or the pod was never wedged this tick), so a one-tick no-port blip never
    matures toward a stop and a healed pod's stale wedge fields are reset. It
    re-saves the state with ``wedge_first_seen=None, wedge_missed=0,
    wedge_alerted=False`` (NOT :func:`_clear_pod_safety_state`, which clears the
    WHOLE file and would wipe the GC anchor ``first_seen``), and records the
    current ``pod_id`` so a later pod_id mismatch is detectable. The status-class
    counters (``missed`` / ``alerted`` / ``last_progress_ts``) are forward-carried
    from the prior state (the status-class arm owns them on its own ticks)."""
    prev = _load_pod_safety_state(issue)
    prev_missed = prev.get("missed", 0)
    if not isinstance(prev_missed, int):
        prev_missed = 0
    prev_progress = prev.get("last_progress_ts")
    if not isinstance(prev_progress, int | float):
        prev_progress = None
    _save_pod_safety_state(
        issue,
        pod_id,
        missed=prev_missed,
        alerted=bool(prev.get("alerted", False)),
        last_progress_ts=prev_progress,
        wedge_first_seen=None,
        wedge_missed=0,
        wedge_alerted=False,
        prev=prev,
    )


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

    Predicate for the pod-safety auto-stop exemption: a task at a
    pod-safety auto-stop status (DONE or ``on_hold``, #980) with a fresh
    follow-up signal carries an in-flight, user-approved
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
    already verified the task's current status is in the pod-safety
    auto-stop set (DONE or ``on_hold``; every entry into it — including a
    ``set-status <N> on_hold`` pause — posts ``epm:status-changed``), so at
    least one ``epm:status-changed`` must have fired to put it there. If the read
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


_DAEMON_PROBE_ATTEMPTS_DEFAULT = 3  # total attempts (1 initial + 2 retries)
_DAEMON_PROBE_BASE_SLEEP_S = 5.0  # backoff base: sleeps 5s, 10s (base * 2**attempt)


def _daemon_probe_attempts() -> int:
    """Daemon-probe attempt count (env ``EPM_DAEMON_PROBE_ATTEMPTS``, an
    integer COUNT; default :data:`_DAEMON_PROBE_ATTEMPTS_DEFAULT`).
    Malformed / non-positive env falls back — a typo'd var must not disable
    the probe entirely."""
    raw = os.environ.get("EPM_DAEMON_PROBE_ATTEMPTS")
    if not raw:
        return _DAEMON_PROBE_ATTEMPTS_DEFAULT
    try:
        parsed = int(raw)
    except ValueError:
        return _DAEMON_PROBE_ATTEMPTS_DEFAULT
    if parsed < 1:
        return _DAEMON_PROBE_ATTEMPTS_DEFAULT
    return parsed


def _daemon_reachable_with_retry(
    attempts: int | None = None, base_sleep_s: float = _DAEMON_PROBE_BASE_SLEEP_S
) -> bool:
    """:func:`_daemon_reachable` with bounded retry-and-backoff (#845 c;
    incident #811: a single failed probe at alert time silently deferred an
    auto-respawn and the GPU idled until manual recovery). Retries defuse a
    transient daemon flap at < ~45s worst-case added tick time — only paid
    when the daemon is genuinely down. Backoff shape mirrors
    ``_WEDGE_RECORD_RETRY_*`` (bounded exponential: 5s, 10s)."""
    attempts = attempts if attempts is not None else _daemon_probe_attempts()
    for attempt in range(attempts):
        if _daemon_reachable():
            return True
        if attempt + 1 < attempts:
            sleep_for = base_sleep_s * (2**attempt)
            print(
                f"daemon probe failed (attempt {attempt + 1}/{attempts}); "
                f"retrying in {sleep_for:.0f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
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


def _live_pids_by_sid_or_none() -> dict[str, int] | None:
    """The daemon's live ``{happySessionId: wrapper pid}`` map, or ``None``
    when the ``/list`` probe fails. Sibling of
    :func:`_live_session_ids_or_none`, added (rather than modifying it) for
    the #845 (e) prompt-wedge probe, which needs the wrapper PID to resolve
    the session's transcript via the happy-log path.

    Fail direction differs deliberately from the sibling: a child dict with
    a missing/invalid sid or pid is SKIPPED (not fail-all) — the wedge
    consumer fails toward NO-WEDGE (no action) on a missing entry, so a
    partial map is the conservative read there, unlike the infra-drain
    stale-registration read where one bad child must contaminate the reply.
    """
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
        return None
    pids: dict[str, int] = {}
    for c in children:
        if not isinstance(c, dict):
            continue
        sid = c.get("happySessionId")
        pid = c.get("pid")
        if isinstance(sid, str) and sid and isinstance(pid, int) and not isinstance(pid, bool):
            pids[sid] = pid
    return pids


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


def _respawn(entry: dict, dry_run: bool) -> str:
    """Re-spawn the autonomous session for this entry. Returns the #843 M1b
    tri-state ``"spawned" | "suppressed" | "failed"`` (``"suppressed"`` = the
    rc-0 subprocess printed a duplicate-suppression sentinel — another
    dispatcher's session is driving, so nothing was respawned; ``"failed"``
    also covers dry-run). On ``"spawned"``, spawn_session rewrites the
    registry (new id, missed=0) as a side effect.

    Re-passes the per-session Claude overrides (``model``, ``betas``,
    ``effort``) verbatim when the registry entry recorded them at spawn time —
    they are part of the prompt-cache key, so flipping any of them on respawn
    would force a full uncached re-read of the conversation (CLAUDE.md §
    Context hygiene). Entries that pre-date the override-persistence feature
    simply don't carry these fields, so the respawn inherits the user's global
    Claude Code defaults (matching the pre-feature behavior).

    ``"suppressed"`` ALSO covers the #1027 auth-outage gate (fleet respawn
    suppression during an active outage episode) — same no-booking contract."""
    issue = entry["issue"]
    if _auth_outage_spawn_gate(issue, "crash", dry_run=dry_run) is not None:
        print(f"  RESPAWN issue #{issue}: suppressed — auth-outage episode active")
        return "suppressed"
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
        return "failed"  # dry-run: nothing spawned
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(f"  RESPAWN FAILED issue #{issue}: {res.stderr.strip()[:300]}", file=sys.stderr)
        return "failed"
    _forward_marker_child_stderr(res, "spawn_session spawn-issue (crash)")
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    if spawn_output_suppressed(res.stdout) is not None:
        print(
            f"  RESPAWN issue #{issue}: suppressed — not respawned (lease/collision): {first_line}"
        )
        return "suppressed"
    print(f"  RESPAWNED issue #{issue} (session was dead): {first_line}")
    _auth_outage_record_spawn(issue, "crash", _coerce_ts(entry.get("spawned_at")))
    return "spawned"


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
    stop_failed_noted: bool | None = None,
    wedge_first_seen: float | None = _CARRY,
    wedge_missed: int | None = _CARRY,
    wedge_alerted: bool | None = _CARRY,
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
    None carries forward identically. ``stop_failed_noted`` is the same
    dedup/carry-forward flag owned by the stop arm's FAILED branch (one
    durable ``stop-failed`` marker per state-file lifetime, #1155); None
    carries forward identically.  ``prev`` is the existing on-disk
    payload (if any), passed so callers that already loaded it don't re-read;
    ``first_seen`` carries forward when present so the age backstop measures
    the original episode start, not the latest save.

    The #692 wedge fields — ``wedge_first_seen`` (the DEDICATED wedge-onset
    clock, stamped at the first wedged tick, NOT the pod-incarnation
    ``first_seen``), ``wedge_missed`` (the wedge arm's >=threshold
    consecutive-confirmed-checks miss guard, SEPARATE from the status-class
    ``missed``), and ``wedge_alerted`` (the once-per-wedge-episode alert dedup)
    — each carry the prior on-disk value forward when LEFT AT THE DEFAULT
    (:data:`_CARRY`), so a status-class-arm save never clobbers a live wedge
    episode's accumulated state and vice versa (the two arms are mutually
    exclusive on a given tick, but a pod can transition between them across
    ticks). Passing an EXPLICIT value (including ``None`` for
    ``wedge_first_seen`` — the MF1 onset-clock CLEAR :func:`_clear_wedge_state`
    needs) overrides the carry-forward. The distinct ``_CARRY`` sentinel (NOT
    ``None``) is load-bearing here precisely because ``None`` is a meaningful
    "clear the wedge clock" value, unlike the ``keep_running_noted`` /
    ``followup_noted`` flags whose only carry-forward signal IS ``None``. MF3:
    without this forward-carry the wedge fields would be silently dropped on
    every save and the wedge miss-guard / alert-dedup would never accumulate.
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
    if stop_failed_noted is None:
        stop_failed_noted = bool((prev or {}).get("stop_failed_noted", False))
    if wedge_first_seen is _CARRY:
        prev_wedge_first_seen = (prev or {}).get("wedge_first_seen")
        wedge_first_seen = (
            prev_wedge_first_seen if isinstance(prev_wedge_first_seen, int | float) else None
        )
    if wedge_missed is _CARRY:
        prev_wedge_missed = (prev or {}).get("wedge_missed", 0)
        wedge_missed = prev_wedge_missed if isinstance(prev_wedge_missed, int) else 0
    if wedge_alerted is _CARRY:
        wedge_alerted = bool((prev or {}).get("wedge_alerted", False))
    payload = {
        "pod_id": pod_id,
        "missed": missed,
        "alerted": alerted,
        "last_progress_ts": last_progress_ts,
        "keep_running_noted": bool(keep_running_noted),
        "followup_noted": bool(followup_noted),
        "stop_failed_noted": bool(stop_failed_noted),
        "first_seen": prev_first_seen,
        "wedge_first_seen": wedge_first_seen,
        "wedge_missed": int(wedge_missed),
        "wedge_alerted": bool(wedge_alerted),
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


# ─── stalled-detector state store ────────────────────────────────────────────


def _stalled_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{STALLED_STATE_PREFIX}{issue}.json"


def _append_stalled_live_event(
    issue: int, event: str, live_consecutive: int, k: int, dry_run: bool
) -> None:
    """Durable trace + telemetry for the #759 bug-class-b.1 live-session
    K-escalation. This file has no in-pass telemetry counter dict (the house
    pattern is print + a sidecar JSONL), so each downgrade / escalation branch
    appends one JSON line to
    ``~/.eps-autonomous/stalled-live-escalation-events.jsonl`` — so a prod
    reader can confirm the corroboration actually fired (downgrade on a live
    transient stall, escalation on the Kth consecutive one) vs the
    respawn-the-dead-bg-chain branch. ``event`` is ``"stalled-live-downgrade"``
    or ``"stalled-live-escalation"``. Fail-soft, mirroring
    :func:`_append_idle_unmapped_event`."""
    dest = AUTONOMOUS_REGISTRY_DIR / "stalled-live-escalation-events.jsonl"
    line = json.dumps(
        {
            "ts": datetime.now().astimezone().isoformat(),
            "issue": issue,
            "event": event,
            "live_consecutive": live_consecutive,
            "k": k,
        }
    )
    if dry_run:
        print(f"  [dry-run] would append stalled-live event ({event}) to {dest}")
        return
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as e:
        print(f"  WARNING: appending stalled-live event failed: {e}", file=sys.stderr)


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
    live_consecutive: int = 0,
    stop_pending_sid: str | None = None,
    stop_pending_ts: float | None = None,
    stop_retried: bool = False,
    stop_failed_alerted: bool = False,
    wt_hold_count: int = 0,
    daemon_blocked_ticks: int = 0,
    daemon_blocked_pushed: bool = False,
    wedge_hits: int = 0,
    dead_silence_respawn_day: str | None = None,
    dead_silence_respawns_today: int = 0,
    wedge_respawn_day: str | None = None,
    wedge_respawns_today: int = 0,
    prev: dict | None = None,
) -> None:
    """Persist the per-session stalled-detector state atomically (temp +
    rename), mirroring :func:`_save_pod_safety_state`.

    #1209 day-cap fields: ``dead_silence_respawn_day`` (UTC ``%Y-%m-%d``
    key) + ``dead_silence_respawns_today`` count the fence episodes the
    ``failed-turn-silence`` wedge trigger INITIATED this UTC day (bumped
    ONCE per episode at stop-initiation). Deliberately EXEMPT from the
    advancement-clear the #845 hardening fields get
    (:func:`_stalled_hardening_fields`): each die-on-turn-1 generation
    writes one boot self-report, so an advancement-cleared counter could
    never bound the cross-generation die-on-boot loop this cap exists for.
    Absent in older on-disk files -> loaded as ``(None, 0)`` (backward
    compatible; a day-rolled or malformed value re-arms at 0 — a corruption
    costs at most one extra bounded respawn).

    #1241 twin fields: ``wedge_respawn_day`` + ``wedge_respawns_today`` are
    the SAME day-keyed, advancement-clear-EXEMPT contract for the ONE
    SHARED counter of the four pre-#1209 wedge triggers (``dequeue-run`` /
    ``api-error-run`` / ``failed-turn-run`` / ``failed-turn-rate``), bumped
    ONCE per wedge-initiated fence episode at stop-initiation (the
    crash-recovery arm — which consults no cap — can complete a
    fresh-self-report wedge respawn, so neither the fence spawn branch nor
    ``respawn_count`` can be the counting site). Independent of the #1209
    budget; same absent/day-rolled/malformed ``(None, 0)`` load contract
    (:func:`_day_scoped_count`).

    #845 hardening fields (all default-absent in older on-disk files —
    backward compatible, same guard shape as ``live_consecutive``; ALL are
    cleared by the caller on self-report advancement, the episode-over
    signal): ``stop_pending_sid`` / ``stop_pending_ts`` / ``stop_retried`` /
    ``stop_failed_alerted`` are the (a-ii) stop-verify FENCE — SCALAR
    per-issue fields (a single pending sid per stalled episode; deliberately
    NOT the zombie pass's ``stopped_at: {sid: ts}`` MAP shape).
    ``wt_hold_count`` is the (b) bounded worktree-activity hold counter
    (also cleared when the fence's spawn fires). ``daemon_blocked_ticks`` /
    ``daemon_blocked_pushed`` drive the (c) daemon-blocked Telegram
    escalation (reset when the daemon is reachable). ``wedge_hits`` counts
    (e) prompt-wedge escalations — observability only.

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
    ``live_consecutive`` is the count of CONSECUTIVE stalled-detector ticks
    on which decide() wanted to respawn but the session's Happy id was
    still in the daemon's live set (#759 bug class b.1): the alert arm
    persists the INCREMENTED count, the respawn / dead-wrapper / keep /
    clear paths persist a RESET 0. A missing key in an older on-disk file
    reads as 0 via :func:`_load_stalled_state` (backward compatible).
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
        "live_consecutive": live_consecutive,
        "stop_pending_sid": stop_pending_sid,
        "stop_pending_ts": stop_pending_ts,
        "stop_retried": stop_retried,
        "stop_failed_alerted": stop_failed_alerted,
        "wt_hold_count": wt_hold_count,
        "daemon_blocked_ticks": daemon_blocked_ticks,
        "daemon_blocked_pushed": daemon_blocked_pushed,
        "wedge_hits": wedge_hits,
        "dead_silence_respawn_day": dead_silence_respawn_day,
        "dead_silence_respawns_today": dead_silence_respawns_today,
        "wedge_respawn_day": wedge_respawn_day,
        "wedge_respawns_today": wedge_respawns_today,
        "last_self_report_ts": last_self_report_ts,
        "first_seen": prev_first_seen,
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


_STALLED_HARDENING_DEFAULTS: dict[str, object] = {
    "stop_pending_sid": None,
    "stop_pending_ts": None,
    "stop_retried": False,
    "stop_failed_alerted": False,
    "wt_hold_count": 0,
    "daemon_blocked_ticks": 0,
    "daemon_blocked_pushed": False,
    "wedge_hits": 0,
}


def _stalled_hardening_fields(prev_state: dict, advanced: bool) -> dict:
    """Load the #845 hardening fields from the prior on-disk stalled-state
    payload with type guards (a missing/garbled field reads as its default —
    backward compatible with pre-#845 files), clearing ALL of them when the
    self-report ADVANCED (the episode ended — same clearing rule as
    ``alerted`` / ``respawn_count``)."""
    if advanced:
        return dict(_STALLED_HARDENING_DEFAULTS)

    def _int(key: str) -> int:
        val = prev_state.get(key, 0)
        return val if isinstance(val, int) and not isinstance(val, bool) else 0

    sid = prev_state.get("stop_pending_sid")
    ts = prev_state.get("stop_pending_ts")
    return {
        "stop_pending_sid": sid if isinstance(sid, str) and sid else None,
        "stop_pending_ts": float(ts) if isinstance(ts, int | float) else None,
        "stop_retried": bool(prev_state.get("stop_retried", False)),
        "stop_failed_alerted": bool(prev_state.get("stop_failed_alerted", False)),
        "wt_hold_count": _int("wt_hold_count"),
        "daemon_blocked_ticks": _int("daemon_blocked_ticks"),
        "daemon_blocked_pushed": bool(prev_state.get("daemon_blocked_pushed", False)),
        "wedge_hits": _int("wedge_hits"),
    }


def _day_scoped_count(prev_state: dict, day_field: str, count_field: str, day_key: str) -> int:
    """Load a day-keyed counter from the prior on-disk stalled-state payload:
    the prior value iff its day field matches ``day_key`` and it is a
    non-bool non-negative int; else 0 (armed — a rolled day / absent key /
    malformed value costs at most one extra bounded respawn). Shared by the
    #1209 ``dead_silence_*`` and #1241 ``wedge_*`` day-cap loads; both are
    deliberately EXEMPT from the advancement clear
    (:func:`_stalled_hardening_fields`) — see :func:`_save_stalled_state`."""
    n = prev_state.get(count_field, 0)
    ok = (
        prev_state.get(day_field) == day_key
        and isinstance(n, int)
        and not isinstance(n, bool)
        and n >= 0
    )
    return n if ok else 0


def _clear_fence_state_on_disk(issue: int) -> None:
    """Clear ONLY the stop-verify fence fields on the persisted stalled
    state, leaving every other field untouched. Used on the #843
    ``"suppressed"`` tri-state return inside the fence's spawn branch: a
    lease collision means a live driver owns the issue, so the fence episode
    is over — but the #843 contract is "book NOTHING" (no respawn_count
    bump, no missed/alerted rewrite), so a full re-save of current-tick
    values would over-write state the suppressed path must not touch."""
    path = _stalled_state_path(issue)
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(payload, dict):
        return
    payload["stop_pending_sid"] = None
    payload["stop_pending_ts"] = None
    payload["stop_retried"] = False
    payload["stop_failed_alerted"] = False
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
    except OSError as e:
        print(f"  WARNING: clearing fence state for #{issue} failed: {e}", file=sys.stderr)


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


def _forward_marker_child_stderr(res, context: str) -> None:
    """Forward a rc==0 task.py / spawn_session.py child's non-empty stderr
    (#1130/#1221): the child exits 0 while printing deferred-commit / LANDING
    CHECK / registration warnings, which capture_output would otherwise
    swallow into the cron log's void. The `[task.py stderr]` output prefix is
    kept byte-stable for grep/test compatibility; the ``context`` label names
    the actual child."""
    err = (getattr(res, "stderr", None) or "").strip()
    if not err:
        return
    for line in err[:2000].splitlines():
        print(f"  [task.py stderr] {context}: {line}", file=sys.stderr)


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
        res = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "post-marker",
                str(issue),
                "epm:progress",
                "--by",
                "autonomous_session_watch",
                "--note",
                note,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        # check=True means rc!=0 raised above — the helper only sees rc==0 results.
        _forward_marker_child_stderr(res, f"epm:progress on #{issue}")
    except (subprocess.SubprocessError, OSError) as e:
        # The action (stop / alert) already happened; failing to annotate it is
        # not worth aborting the run. Surface it loudly so the gap is visible.
        print(f"  WARNING: posting {label} marker on #{issue} failed: {e}", file=sys.stderr)


def _post_failure_marker(issue: int, note: str, dry_run: bool) -> bool:
    """Post an ``epm:failure v1`` marker on task ``issue``'s events.jsonl.

    Returns True on a successful post (or in dry-run), False when the post was
    swallowed. The watcher backstop is the ACTOR for a poller-DEAD wedge, so it
    must emit the SAME ``epm:failure v1`` the orchestrator's bg-Bash poll loop
    emits when it reads a terminal infra JSON — otherwise the capacity-retry pass
    (which keys on the latest ``epm:failure`` marker's
    ``failure_class``/``reason`` via :func:`_is_transient_capacity_block`) never
    sees the block and can never re-drive a re-drivable ``no_compute_available``
    terminal. The returned bool lets the caller GATE the wedge-clock clear on
    this durable record actually landing: under a transient ``task.py`` / flock /
    network failure the marker may not post, and clearing the clock anyway would
    both lose the failure record AND let the next tick treat the pod as
    freshly-wedged. ``note`` MUST carry ``failure_class: <class>`` and
    ``reason: <reason>`` as whitespace-separated tokens (the shape
    :func:`_parse_failure_fields` reads). Same branch-guard contract as
    :func:`_post_progress_marker` (runs from PROJECT_ROOT on ``main``)."""
    if dry_run:
        print(f"  [dry-run] would post epm:failure on #{issue}: {note}")
        return True
    try:
        res = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "post-marker",
                str(issue),
                "epm:failure",
                "--by",
                "autonomous_session_watch",
                "--note",
                note,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        _forward_marker_child_stderr(res, f"epm:failure on #{issue}")
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  WARNING: posting epm:failure marker on #{issue} failed: {e}", file=sys.stderr)
        return False
    return True


def _set_status_blocked(issue: int, dry_run: bool) -> bool:
    """Run ``task.py set-status <N> blocked``. Returns True on success.

    Mirrors the orchestrator's poll-loop behavior on a terminal infra JSON
    (``status:dead`` + ``failure_class:infra``): set the task to ``blocked`` so
    the watcher's capacity-retry pass (and a human) can act on it. Same
    branch-guard contract as :func:`_post_progress_marker`."""
    if dry_run:
        print(f"  [dry-run] would set #{issue} status -> blocked")
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
                "blocked",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  WARNING: set-status blocked on #{issue} FAILED ({exc})", file=sys.stderr)
        return False
    if out.returncode != 0:
        detail = (out.stderr or out.stdout or "").strip()[:300]
        print(
            f"  WARNING: set-status blocked on #{issue} FAILED (rc={out.returncode}): {detail}",
            file=sys.stderr,
        )
        return False
    _forward_marker_child_stderr(out, f"set-status blocked on #{issue}")
    return True


_WEDGE_RECORD_RETRY_ATTEMPTS = 3  # total attempts (1 initial + 2 retries)
_WEDGE_RECORD_RETRY_BASE_S = 1.0  # backoff base: sleeps 1s, 2s (1.0 * 2**attempt)


def _retry_durable_write(
    write: Callable[[], bool], *, what: str, issue: int, dry_run: bool
) -> bool:
    """Call a ``() -> bool`` durable-write closure, retrying on a False return.

    A False return is a SWALLOWED transient ``task.py`` / flock / network failure
    (the writers return False only when the post did NOT commit). Retries with a
    bounded exponential backoff (``_WEDGE_RECORD_RETRY_BASE_S * 2**attempt``:
    1s, 2s for the default 3 attempts), returning True the MOMENT a call succeeds
    and False only after ``_WEDGE_RECORD_RETRY_ATTEMPTS`` attempts all returned
    False.

    Why this is the watcher's durable retry layer (the strategy-pivot v2 fix,
    #770): by the time the two terminal-record writes are reached,
    ``backend_poll._failover_wedged_runpod`` has ALREADY terminated the wedged
    pod, so the two ``task.py`` subprocess writes are the only remaining work in
    the episode. The next watcher tick CANNOT retry them — the pod is gone from
    :func:`_running_managed_issue_pods` (RUNNING-only), so :func:`_process_wedged_pod`
    is never re-entered for it — so the retry must happen HERE. The poller's
    persistent bg re-poll loop (the equivalent retry layer on the poll-alive path)
    is dead; the watcher IS the backstop precisely because that loop is dead.

    The closure form lets the caller retry each of the two writes INDEPENDENTLY,
    so a transient failure on ONE write never re-issues the OTHER (already-landed)
    write — ``task.py post-marker`` is not idempotent, so re-posting a succeeded
    ``epm:failure`` marker would duplicate it.

    The loop sleeps AFTER a failure only when another attempt remains (no trailing
    sleep on the last failure) — total added wall on a full failure is
    ``1.0 + 2.0 = 3.0s``, bounded. ``dry_run`` short-circuits to True on the first
    call (the underlying writers already no-op + return True in dry-run; no real
    sleeps in a dry-run smoke)."""
    for attempt in range(_WEDGE_RECORD_RETRY_ATTEMPTS):
        if write():
            return True
        if dry_run:  # underlying writers return True in dry-run; defensive
            return True
        if attempt + 1 < _WEDGE_RECORD_RETRY_ATTEMPTS:
            sleep_for = _WEDGE_RECORD_RETRY_BASE_S * (2**attempt)
            print(
                f"  issue #{issue}: durable write {what!r} failed "
                f"(attempt {attempt + 1}/{_WEDGE_RECORD_RETRY_ATTEMPTS}); "
                f"sleeping {sleep_for:.1f}s then retrying",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    return False


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


def _wedge_failover(
    issue: int, info: PodInfo, wedged_h: str, dry_run: bool
) -> tuple[str, dict | None]:
    """Run the EXISTING poller terminate+re-provision recovery for a matured,
    inputs-safe, untagged RunPod no-port wedge from the watcher (#770).

    Called only on the ``decide_pod_wedge`` ``"terminate-failover"`` action — the
    provably-safe case (``keep_running is False AND inputs_ok=True`` on a matured,
    confirmed wedge). Routes the SAME irreversible recovery the poller owns
    (``backend_poll._failover_wedged_runpod``: terminate the wedged pod,
    re-provision a FRESH pod) instead of a reversible ``pod.py stop`` that cannot
    heal a host-pinned dead RunPod host (#763). The recovery is bounded-once via
    the shared durable lease + sentinel (``_runpod_wedge_already_handled``, called
    INSIDE ``_failover_wedged_runpod``), so a poller-side and watcher-side firing
    on the SAME wedge are mutually exclusive — the watcher inherits the
    cross-actor idempotency for free by calling that function.

    Returns a ``(outcome, terminal_json)`` tuple. ``outcome`` is the one-word
    label for the caller's marker/state branch; ``terminal_json`` is the raw
    poller terminal-JSON dict the recovery returned (carrying ``failure_class``
    + ``reason``), or ``None`` for the ``"alert"`` degrade and the dry-run
    short-circuit (no recovery dict in scope). The caller mirrors the poller's
    own path on the terminal JSON — posting ``epm:failure v1`` with the exact
    ``failure_class``/``reason`` from ``terminal_json`` and setting
    ``status:blocked`` — for the ``"no-capacity"`` and ``"blocked"`` outcomes,
    so the watcher's capacity-retry pass re-drives the re-drivable
    ``no_compute_available`` block exactly as it would have re-driven the
    poller's own emission of the SAME terminal JSON.

    Outcomes:

    * ``"failover"`` — terminated + re-provisioned a FRESH pod (or a
      running-shaped success);
    * ``"already-handled"`` — the poller or a prior watcher tick already failed
      this wedge over (bounded-once short-circuit on the idempotency lease /
      sentinel), OR the sidecar was re-pointed at a DIFFERENT pod between the
      inputs-safe read and this re-read (the fresh-pod race below); no second
      terminate;
    * ``"no-capacity"`` — terminated the wedged pod (billing stopped) but RunPod
      is unavailable for the fresh re-provision → terminal
      ``no_compute_available`` (the watcher's capacity-retry pass re-drives once
      a lane frees);
    * ``"blocked"`` — a terminal infra block a human resolves (CLAUDE.md
      halt-criterion #2). One of FOUR reasons, NOT uniform on whether the pod
      terminated: ``runpod_wedge_inputs_unverified`` is PRE-terminate (a PARTIAL
      cell on HF — the pod was NOT terminated), while ``sidecar_persistence_failed``
      and ``runpod_wedge_relaunch_spec_missing`` are POST-terminate (the wedged
      pod WAS terminated and the failure is in the fresh re-provision), and
      ``runpod_wedge_failover_error`` is the catch-all for an UNEXPECTED raise
      from ``_failover_wedged_runpod`` that bypassed its own terminal-JSON
      mapping AND that a liveness probe confirmed happened AFTER ``terminate_pod``
      (the pod is GONE per ``get_pod_by_name`` — #770 v2 r3; a PRE-terminate raise
      where the pod is still alive degrades to ``"alert"`` instead, see below).
      The caller's marker text reads the ``reason`` from ``terminal_json`` to
      state the right one;
    * ``"alert"`` — NO reconstructable run handle (no sidecar / a parse failure),
      or the sidecar pod_name does not match the wedged pod → degrade to
      ALERT-only, NEVER a blind terminate. NOTE (#770 v2 r3): a PRE-terminate raise
      FROM ``_failover_wedged_runpod`` (the pod is still ALIVE per the liveness
      probe — e.g. a transient HF ``list_repo_files`` blip before ``terminate_pod``)
      ALSO maps to ``"alert"``, so the wedge clock is PRESERVED and the next tick
      re-detects the still-RUNNING wedge — never a FALSE terminal record while the
      pod is still billing. A POST-terminate raise (pod gone) maps to ``"blocked"``
      ``runpod_wedge_failover_error`` instead — see above.

    Obtains ``handle`` + ``sidecar`` from the persisted sidecar EXACTLY as
    :func:`_wedge_inputs_safe` does (the watcher has no in-scope poller handle);
    on a sidecar reconstruction gap (no sidecar / a parse failure) returns
    ``"alert"`` (never strand un-uploaded work, never terminate blind).

    **Sidecar-binding defense (the fresh-pod race).** Between
    :func:`_wedge_inputs_safe`'s sidecar read (which verified inputs against the
    OLD wedged handle) and THIS re-read, a revived poller (crash-recovery /
    capacity-retry respawn) could have already failed the wedge over and
    re-pointed the sidecar at a FRESH, HEALTHY pod. The bounded-once lease/
    sentinel inside :func:`_failover_wedged_runpod` is keyed on the FRESH
    handle's identity, so it would NOT catch this — the watcher would terminate
    a healthy fresh pod. Defense: assert the freshly-read ``handle.pod_name``
    still equals the WEDGED pod's name (``info.name``). A mismatch means the
    sidecar already moved on → return ``"already-handled"`` (the wedge the
    watcher saw is gone; the live pod is someone else's) and NEVER call
    :func:`_failover_wedged_runpod` against it.

    The ``_failover_wedged_runpod`` call is wrapped fail-soft — an UNEXPECTED
    raise is triaged by a ``get_pod_by_name(info.name)`` liveness probe into one
    of two outcomes (#770 v2 r3), never crashing the whole 10-min watcher tick
    (which still processes the rest of the fleet):

    * **POST-terminate raise (pod GONE per the probe)** → ``("blocked",
      terminal_json)`` carrying ``failure_class=infra
      reason=runpod_wedge_failover_error`` (mirroring the poller's own caller
      defense, ``backend_poll`` ~1864-1880) so the durable epm:failure +
      status:blocked record always lands — NOT degraded to ``"alert"`` (which
      would post no record and could not be retried next tick, since the
      terminated pod is gone from the RUNNING-only snapshot).
    * **PRE-terminate raise (pod still ALIVE per the probe, or the probe itself
      raised → UNCERTAIN, bias SAFE)** → ``("alert", None)``. The fallible
      pre-terminate steps inside ``_failover_wedged_runpod`` (the
      ``_runpod_wedge_already_handled`` lease check; ``_wedged_run_inputs_on_hf``
      → ``huggingface_hub.list_repo_files``) can raise BEFORE ``terminate_pod``,
      leaving the pod RUNNING+billing — mapping that to ``"blocked"`` would post a
      FALSE terminal record (claiming the pod terminated) AND clear the wedge
      clock. ``"alert"`` PRESERVES the clock so the next tick re-detects the
      still-RUNNING wedge (the pod is still in the RUNNING-only snapshot, so the
      retry is reachable, unlike the post-terminate case).

    ``info`` is load-bearing for both the sidecar-binding defense above AND this
    liveness probe (its ``name`` is the wedged pod the watcher observed); the
    ``_failover_wedged_runpod`` call itself re-fetches the pod by
    ``handle.pod_name``."""
    if dry_run:
        print(f"  [dry-run] would terminate+failover wedged pod for issue #{issue}")
        return ("failover", None)
    try:
        from backend_poll import _failover_wedged_runpod

        from explore_persona_space.backends.issue_dispatch import (
            read_handle_sidecar,
            resolve_handle_sidecar_path,
        )

        path, _probed = resolve_handle_sidecar_path(issue)
        if not path.exists():
            print(
                f"  wedge-failover: no handle sidecar for #{issue}; cannot reconstruct "
                f"the run handle -> ALERT-only (never terminate blind)",
                file=sys.stderr,
            )
            return ("alert", None)
        handle = read_handle_sidecar(path)
    except Exception as exc:  # transport / parse / import -> ALERT-only
        print(
            f"  wedge-failover: handle reconstruction failed for #{issue} "
            f"({type(exc).__name__}: {exc}); ALERT-only",
            file=sys.stderr,
        )
        return ("alert", None)

    # ── Sidecar-binding defense (the fresh-pod race) ──────────────────────────
    # Assert the freshly-read handle still names the WEDGED pod the watcher
    # observed. If a revived poller re-pointed the sidecar at a fresh, healthy
    # pod between _wedge_inputs_safe's read and now, the bounded-once lease (keyed
    # on the fresh handle's identity) would NOT catch it -> the watcher would
    # terminate the healthy fresh pod. A mismatch means the wedge is already
    # handled: do NOT terminate, return "already-handled".
    handle_pod_name = getattr(handle, "pod_name", None)
    if handle_pod_name != info.name:
        print(
            f"  wedge-failover: sidecar for #{issue} now names pod "
            f"{handle_pod_name!r}, not the wedged pod {info.name!r} (the poller "
            f"re-pointed it to a FRESH pod) -> ALREADY-HANDLED (never terminate "
            f"the fresh pod)",
            file=sys.stderr,
        )
        return ("already-handled", None)

    # The poller had a PollResult in scope; the watcher does not.
    # _failover_wedged_runpod reads ONLY current_phase + log_tail_excerpt from
    # `result` (backend_poll.py:964-965), so a minimal shim suffices and labels
    # the recovery's provenance (the watcher backstop) in the evidence dict.
    class _WedgeResultShim:
        current_phase = "runpod_noport_wedge_watcher_backstop"
        log_tail_excerpt = (
            f"watcher-detected RunPod RUNNING-but-no-port wedge ({wedged_h}); the per-issue "
            f"poll loop is dead, so the watcher is the backstop (#692->#770)"
        )

    try:
        out = _failover_wedged_runpod(
            issue=issue, handle=handle, result=_WedgeResultShim(), sidecar=path
        )
    except Exception as exc:  # fail-LOUD but do not crash the whole watcher tick
        # RAISE FROM _failover_wedged_runpod (#770 v2 r3): the function has fallible
        # PRE-terminate steps (the `_runpod_wedge_already_handled` lease check;
        # `_wedged_run_inputs_on_hf` -> huggingface_hub.list_repo_files, no local
        # try/except — a transient HF/network blip bubbles) BEFORE the irreversible
        # terminate_pod, and fallible POST-terminate steps after it (the fresh
        # re-provision / sidecar / router code). The two cases need OPPOSITE handling:
        #
        #   * POST-terminate raise — the wedged pod is GONE: degrading to "alert"
        #     would strand the run (the alert branch posts no epm:failure /
        #     status:blocked, and the terminated pod is gone from the RUNNING-only
        #     _running_managed_issue_pods snapshot, so the next tick never re-enters
        #     _process_wedged_pod). MIRROR the poller's own caller defense
        #     (backend_poll.py ~1864-1880, which converts ANY raise to
        #     _terminal_infra_json(reason="runpod_wedge_failover_error")): return a
        #     terminal infra "blocked" outcome so _handle_wedge_failover_outcome
        #     records it durably. reason=runpod_wedge_failover_error is NOT in
        #     TRANSIENT_CAPACITY_REASONS, so the capacity-retry pass parks it for a
        #     human rather than re-driving doomed-broken code.
        #
        #   * PRE-terminate raise — the wedged pod is STILL ALIVE (and billing):
        #     mapping to "blocked" here would post a FALSE durable record (the
        #     marker text falsely claims the pod was "likely terminated") AND clear
        #     the wedge clock, while the pod keeps billing. Degrade to "alert"
        #     instead — the clock is PRESERVED (the alert branch carries it forward),
        #     so the NEXT tick re-detects the still-RUNNING wedge and re-matures it
        #     (the pod is still in the RUNNING-only snapshot, so the retry is
        #     reachable, unlike the post-terminate case).
        #
        # Distinguish the two by PROBING the pod's liveness by name. get_pod_by_name
        # returns None when the pod is gone (-> post-terminate -> "blocked") and a
        # PodInfo when it is still alive (-> pre-terminate -> "alert"). The probe
        # itself can raise (network/transport); an UNCERTAIN probe biases SAFE ->
        # "alert" (preserve the clock for a next-tick retry rather than post a
        # possibly-false terminal record).
        try:
            live_pod = get_pod_by_name(info.name)
        except Exception as probe_exc:
            live_pod = None  # sentinel; overridden below to the uncertain branch
            print(
                f"  wedge-failover: _failover_wedged_runpod raised for #{issue} "
                f"({type(exc).__name__}: {exc}) AND the liveness probe also raised "
                f"({type(probe_exc).__name__}: {probe_exc}); cannot confirm the pod is "
                f"gone -> degrading to ALERT (pod liveness uncertain, clock preserved "
                f"for a next-tick retry)",
                file=sys.stderr,
            )
            return ("alert", None)
        if live_pod is not None:
            # PRE-terminate raise: the wedged pod is STILL ALIVE. Do NOT post a false
            # terminal record / clear the clock — degrade to ALERT so the next tick
            # re-detects the still-RUNNING wedge with the clock intact.
            print(
                f"  wedge-failover: _failover_wedged_runpod raised for #{issue} "
                f"({type(exc).__name__}: {exc}) BEFORE terminating the pod (it is still "
                f"ALIVE per get_pod_by_name) -> degrading to ALERT (pod still billing; "
                f"clock preserved for a next-tick retry, never a false terminal record)",
                file=sys.stderr,
            )
            return ("alert", None)
        # POST-terminate raise: the wedged pod is GONE. Record a durable terminal
        # block so the run is never silently stranded.
        print(
            f"  wedge-failover: _failover_wedged_runpod raised for #{issue} "
            f"({type(exc).__name__}: {exc}); the pod is GONE per get_pod_by_name (the "
            f"raise was AFTER terminate_pod) -> mapping to a 'blocked' terminal record "
            f"(epm:failure + status:blocked) so the run is never silently stranded",
            file=sys.stderr,
        )
        terminal = {
            "status": "dead",
            "failure_class": "infra",
            "reason": "runpod_wedge_failover_error",
            "log_tail_excerpt": (
                f"_failover_wedged_runpod raised for issue {issue}: "
                f"{type(exc).__name__}: {exc}; the wedged pod is gone per "
                f"get_pod_by_name (the raise was after terminate_pod)"
            ),
        }
        return ("blocked", terminal)

    status = out.get("status")
    reason = out.get("reason")
    if status == "running":
        return ("failover", out)  # fresh pod re-provisioned (running-shaped success)
    if reason == "runpod_wedge_already_handled":
        return ("already-handled", out)  # bounded-once short-circuit (cross-actor idempotency)
    if reason == "no_compute_available":
        return ("no-capacity", out)  # terminated; RunPod unavailable; capacity-retry re-drives
    # runpod_wedge_inputs_unverified | sidecar_persistence_failed |
    # runpod_wedge_relaunch_spec_missing | any other terminal infra JSON.
    return ("blocked", out)


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
            # The VM redirects the HF cache (HF_HUB_CACHE / HF_HOME), so a bare
            # scan_cache_dir() looks at the unset default (~/.cache/huggingface/hub)
            # and no-ops at the worst moment (reclaims 0 GiB at VM-disk CRITICAL,
            # 2026-06-26 #658 incident). Point it at the real cache dir.
            _hub_cache = os.environ.get("HF_HUB_CACHE") or (
                os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None
            )
            try:
                cache_info = (
                    scan_cache_dir(cache_dir=_hub_cache) if _hub_cache else scan_cache_dir()
                )
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


def _running_managed_issue_pods(
    caller: str = "pod-safety",
) -> list[tuple[int, str, str, PodInfo]] | None:
    """Live RunPod team pods that are RUNNING and managed (``pod-<N>`` or the
    legacy ``epm-issue-<N>``). Returns ``(issue, pod_id, pod_name, info)``
    4-tuples (the live :class:`runpod_api.PodInfo` carried out so callers can
    read ``desired_status`` / ``ssh_host`` / ``ssh_port`` / ``pod_id`` without a
    second ``list_team_pods`` round-trip — the #692 wedge backstop reads the raw
    no-port wedge condition off this ``info``), or ``None`` when the snapshot
    itself FAILED (API transport error).

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
    out: list[tuple[int, str, str, PodInfo]] = []
    for p in pods:
        if p.desired_status != "RUNNING":
            continue
        if not _is_managed_pod(p):
            continue
        name = p.name or ""
        issue = _issue_from_pod_name(name)
        if issue is not None:
            out.append((issue, p.pod_id, name, p))
    return out


def _maybe_handle_runpod_wedge(
    issue: int, status: str | None, info: PodInfo, now: float, dry_run: bool, threshold: int
) -> bool:
    """#692 wedge arm dispatch: detect the RAW #664 RunPod no-port wedge from the
    live ``info`` and route it.

    Returns ``True`` iff this arm fully HANDLED the pod (a wedged pod whose
    status is NOT in :data:`POD_SAFETY_AUTO_STOP`, processed by
    :func:`_process_wedged_pod`), so the caller
    (:func:`_process_pod`) should return without running the status-class
    branches. Returns ``False`` in every other case — a non-wedged pod (where it
    also clears any stale wedge clock, MF1/MF4) OR a wedged DONE-or-paused-status
    pod (MF6: it falls through to the status-class auto-stop, the canonical
    escaped-pod handler — routing it through the wedge arm's ALERT-default +
    inputs-gate would only WEAKEN that existing auto-stop; for user-paused
    ``on_hold`` (#980) the wedge arm's confirmed-safe path would additionally
    terminate + RELAUNCH a workload the user deliberately paused).

    Detect the raw condition via the SAME
    ``backend_poll._pod_is_runpod_runtime_wedged`` the poller calls (composition
    surface (b), never re-defined)."""
    from backend_poll import _pod_is_runpod_runtime_wedged  # sibling import

    if _pod_is_runpod_runtime_wedged(info):
        if status not in POD_SAFETY_AUTO_STOP:
            _process_wedged_pod(issue, info, now, dry_run, threshold)
            return True
        # MF6: DONE-or-paused-status wedged pod -> fall through to the
        # status-class auto-stop arm (#980: never terminate+relaunch a pause).
        return False
    # MF1/MF4: not currently wedged -> clear any stale wedge clock so a one-tick
    # blip never accumulates, and the next true onset re-stamps.
    if not dry_run:
        _clear_wedge_state(issue, info.pod_id)
    return False


def _build_wedge_terminal_failure_note(
    *,
    outcome: str,
    terminal_json: dict | None,
    issue: int,
    pod_id: str,
    wedged_h: str,
    threshold: int,
) -> str:
    """Build the ``epm:failure`` note body for a ``no-capacity`` / ``blocked``
    wedge-failover outcome (#770 r2).

    The note MUST carry ``failure_class: <c>`` and ``reason: <r>`` as
    whitespace-separated tokens — the exact shape :func:`_parse_failure_fields`
    reads for the capacity-retry classifier — and states the right terminate
    state per ``reason`` (CONCERN #3): ``runpod_wedge_inputs_unverified`` is
    PRE-terminate (the pod is still RUNNING); ``sidecar_persistence_failed`` /
    ``runpod_wedge_relaunch_spec_missing`` are POST-terminate (the wedged pod
    WAS terminated, the fresh re-provision failed); ``runpod_wedge_failover_error``
    (#770 v2 r3) is an UNEXPECTED raise from ``_failover_wedged_runpod`` that a
    liveness probe confirmed happened AFTER ``terminate_pod`` (the pod is GONE per
    ``get_pod_by_name`` — a PRE-terminate raise where the pod is still alive
    degrades to ``"alert"`` and never reaches this branch), so the note says the
    pod WAS terminated and points at the ``log_tail`` for the exception."""
    from backend_poll import RUNPOD_WEDGE_K_SEC

    failure_class = (terminal_json or {}).get("failure_class") or "infra"
    reason = (terminal_json or {}).get("reason") or "unknown"
    log_tail = (terminal_json or {}).get("log_tail_excerpt") or ""
    if outcome == "no-capacity":
        narrative = (
            f"TERMINATED the wedged pod {pod_id} (billing stopped) but RunPod is "
            f"unavailable for the fresh re-provision. The capacity-retry pass re-drives "
            f"once a lane frees (reason no_compute_available is re-drivable)."
        )
    elif reason == "runpod_wedge_inputs_unverified":
        narrative = (
            f"wedge failover did NOT terminate the wedged pod {pod_id} — a PARTIAL cell on "
            f"HF means a terminate could strand un-uploaded work; the pod is still RUNNING "
            f"(and billing) until a human resolves it. Surfaced as a terminal infra block "
            f"(CLAUDE.md halt-criterion #2); a human decides."
        )
    elif reason in ("sidecar_persistence_failed", "runpod_wedge_relaunch_spec_missing"):
        why = (
            "sidecar persistence failed"
            if reason == "sidecar_persistence_failed"
            else "a legacy handle lacks the relaunch RunSpec"
        )
        narrative = (
            f"wedge failover TERMINATED the wedged pod {pod_id} (billing stopped) but the "
            f"fresh re-provision failed ({why}); no live pod for this issue now. Surfaced "
            f"as a terminal infra block (CLAUDE.md halt-criterion #2); a human decides."
        )
    elif reason == "runpod_wedge_failover_error":
        # #770 v2 r3: an UNEXPECTED raise from _failover_wedged_runpod bypassed its
        # own terminal-JSON mapping AND a get_pod_by_name liveness probe confirmed the
        # raise was POST-terminate (the pod is GONE). A PRE-terminate raise where the
        # pod is still alive degrades to "alert" upstream and never reaches this
        # branch, so this note can state the pod WAS terminated (not "likely"). The
        # run is recorded blocked + epm:failure (NOT silently stranded) and a human
        # inspects the log_tail for the exception. Not re-drivable
        # (runpod_wedge_failover_error is not in TRANSIENT_CAPACITY_REASONS).
        narrative = (
            f"wedge failover for pod {pod_id} raised UNEXPECTEDLY before returning a "
            f"terminal JSON; a liveness probe confirmed the pod is GONE, so the raise "
            f"was AFTER terminate_pod and the wedged pod WAS terminated (see log_tail "
            f"for the exception). Recorded as a terminal infra block + status:blocked "
            f"so the run is NOT silently stranded (CLAUDE.md halt-criterion #2); a "
            f"human inspects the raise."
        )
    else:
        # An unrecognized terminal infra reason — stay neutral on the terminate
        # state and point at the log_tail.
        narrative = (
            f"wedge failover returned an unrecognized terminal infra reason for pod "
            f"{pod_id}; inspect reason + log_tail for whether the pod was terminated. "
            f"Surfaced as a terminal infra block (CLAUDE.md halt-criterion #2); a human decides."
        )
    note = (
        f"{_WEDGE_FAILOVER_NOTE_SENTINEL} RUNNING pod {pod_id} stuck in the #664 "
        f"RUNNING-but-no-port host wedge for {wedged_h} (> {RUNPOD_WEDGE_K_SEC}s K "
        f"floor, confirmed for >= {threshold} checks); the poller's "
        f"_maybe_escalate_runpod_wedge never ran (the poll loop is dead), so the "
        f"watcher is the backstop. {narrative} failure_class: {failure_class} "
        f"reason: {reason}"
    )
    if log_tail:
        note += f" log_tail: {log_tail}"
    return note


def _handle_wedge_failover_outcome(
    *,
    issue: int,
    info: PodInfo,
    pod_id: str,
    wedged_h: str,
    threshold: int,
    dry_run: bool,
    state_save: dict,
) -> str:
    """Dispatch the ``terminate-failover`` action and return the next ``action``.

    Calls :func:`_wedge_failover`, then branches on the ``(outcome, terminal_json)``
    tuple it returns:

    * ``"alert"`` → return ``"alert"`` so the caller falls through to the existing
      alert block (no action taken this tick; re-try later).
    * ``"no-capacity"`` / ``"blocked"`` → a terminal infra JSON: the watcher is the
      ACTOR (the poll loop is dead), so it MIRRORS the orchestrator's poll-loop
      path — post ``epm:failure v1`` carrying the exact ``failure_class``/``reason``
      AND ``set-status <N> blocked`` (CRITICAL #1) — then clear the wedge clock.
      The re-drivable ``no_compute_available`` block is re-driven by the
      capacity-retry pass; a non-capacity reason stays parked for a human.
    * ``"failover"`` / ``"already-handled"`` → a success / no-op: a generic
      ``epm:progress`` note (NOT a failure), then clear the wedge clock.

    Returns ``"handled"`` when the outcome was fully processed here (the caller
    returns), or ``"alert"`` to defer to the alert block. ``state_save`` carries
    the forward-carried status-class fields the clock-clear save needs
    (``missed`` / ``alerted`` / ``last_progress_ts`` / ``prev``)."""
    from backend_poll import RUNPOD_WEDGE_K_SEC

    outcome, terminal_json = _wedge_failover(issue, info, wedged_h, dry_run)
    if outcome == "alert":
        # Defer to the ALERT-only block (carries the clock forward + dedups the
        # alert). We did NOT act, so re-try on a later tick. Reached when: there is
        # no reconstructable handle (no sidecar / a parse failure), OR a PRE-terminate
        # raise from _failover_wedged_runpod left the pod STILL ALIVE per the
        # get_pod_by_name liveness probe (or the probe itself raised -> uncertain,
        # bias safe) — #770 v2 r3: preserving the clock so the next tick re-detects
        # the still-RUNNING wedge, never a false terminal record while the pod bills.
        # (The sidecar-names-a-DIFFERENT-pod fresh-pod race maps to "already-handled";
        # a POST-terminate raise where the pod is GONE maps to "blocked"
        # reason=runpod_wedge_failover_error — neither reaches here.)
        return "alert"
    if outcome in ("no-capacity", "blocked"):
        note = _build_wedge_terminal_failure_note(
            outcome=outcome,
            terminal_json=terminal_json,
            issue=issue,
            pod_id=pod_id,
            wedged_h=wedged_h,
            threshold=threshold,
        )
        # The wedged pod is ALREADY terminated (backend_poll._failover_wedged_runpod
        # terminates BEFORE the re-provision that can return no_compute_available),
        # so the two terminal-record writes are the ONLY remaining work and the next
        # watcher tick CANNOT retry them (the pod is gone from
        # _running_managed_issue_pods(), so _process_wedged_pod is never re-entered
        # for it — the r3 "retry next tick" was unreachable). Retry each write
        # SYNCHRONOUSLY in-tick with bounded backoff — this IS the watcher's durable
        # retry layer (the poller's persistent bg re-poll loop, the equivalent on
        # the poll-alive path, is dead here). Retry each independently
        # (task.py post-marker is not idempotent; never re-post a succeeded marker).
        marker_ok = _retry_durable_write(
            lambda: _post_failure_marker(issue, note, dry_run),
            what="epm:failure",
            issue=issue,
            dry_run=dry_run,
        )
        blocked_ok = _retry_durable_write(
            lambda: _set_status_blocked(issue, dry_run),
            what="set-status blocked",
            issue=issue,
            dry_run=dry_run,
        )
        if not dry_run and not (marker_ok and blocked_ok):
            # Retry budget EXHAUSTED: a NON-transient failure (corrupt task.py, full
            # disk, a persistent flock holder) — which would defeat ANY retry path
            # (including a decoupled next-pass one), so resolution is operator-side.
            # Do NOT clear the wedge clock: clearing it would both lose the failure
            # record AND let the next tick treat the pod as freshly-wedged (defeating
            # bounded-once from the watcher side — the marker-side idempotency is
            # owned by backend_poll._runpod_wedge_already_handled), while the
            # capacity-retry pass — which needs BOTH status:blocked + a parseable
            # epm:failure to re-drive a re-drivable no_compute_available block —
            # would never see it. Leave wedge_first_seen intact AND fail LOUD so a
            # human is alerted via the dashboard.
            print(
                f"  issue #{issue}: wedge terminal recording EXHAUSTED retries — "
                f"marker_ok={marker_ok} blocked_ok={blocked_ok}; the pod IS "
                f"terminated (billing stopped) but the durable redrive record did "
                f"NOT land after {_WEDGE_RECORD_RETRY_ATTEMPTS} attempts "
                f"(non-transient task.py / disk / flock failure — operator must "
                f"resolve). Leaving wedge clock intact.",
                file=sys.stderr,
            )
            # Best-effort human-visible alert (itself swallows failures via the
            # _post_progress_marker contract — the loud stderr above is the
            # correctness floor, this is an additional dashboard signal).
            _post_progress_marker(
                issue,
                f"{_WEDGE_FAILOVER_NOTE_SENTINEL} TERMINATED the wedged pod for "
                f"issue #{issue} (billing stopped) but FAILED to durably record the "
                f"terminal infra block (epm:failure marker_ok={marker_ok}, "
                f"set-status blocked blocked_ok={blocked_ok}) after "
                f"{_WEDGE_RECORD_RETRY_ATTEMPTS} bounded retries — a NON-transient "
                f"task.py / disk / flock failure. The task is NOT re-drivable by the "
                f"capacity-retry pass until status:blocked + the epm:failure block "
                f"land. Investigate the task.py write path manually.",
                dry_run,
                label="wedge-failover",
            )
            return "handled"
    else:
        # outcome in ("failover", "already-handled") — a success / no-op, NOT a
        # failure: a generic progress note (NOT epm:failure, NOT a status change).
        # "already-handled" also covers the fresh-pod race the sidecar-binding
        # defense catches.
        note = {
            "failover": (
                f"{_WEDGE_FAILOVER_NOTE_SENTINEL} TERMINATED + re-provisioned a FRESH pod "
                f"for issue #{issue}: RUNNING pod {pod_id} stuck in the #664 "
                f"RUNNING-but-no-port host wedge for {wedged_h} (> {RUNPOD_WEDGE_K_SEC}s K "
                f"floor, confirmed for >= {threshold} checks). The poller's "
                f"_maybe_escalate_runpod_wedge never ran (the poll loop is dead), so the "
                f"watcher is the backstop. Inputs are verified on HF and the keep-running "
                f"tag is absent, so the irreversible terminate + re-provision the poller "
                f"owns (#664/#689 fix (b)) is safe — a reversible stop cannot heal a "
                f"host-pinned dead RunPod host (#763). Bounded once via the durable lease "
                f"+ sentinel. The fresh pod's dispatcher skips HF-complete cells."
            ),
            "already-handled": (
                f"{_WEDGE_FAILOVER_NOTE_SENTINEL} wedge failover for issue #{issue} pod "
                f"{pod_id} was ALREADY handled — the poller or a prior watcher tick already "
                f"terminated + re-provisioned this wedge (idempotency lease/sentinel), or "
                f"the sidecar was re-pointed at a FRESH pod between the inputs-safe read and "
                f"the failover re-read (the fresh-pod race; the watcher refuses to terminate "
                f"the fresh pod). No second terminate. Clearing the wedge clock."
            ),
        }[outcome]
        _post_progress_marker(issue, note, dry_run, label="wedge-failover")
    if not dry_run:
        # Clear the wedge fields on every acted/already-handled outcome (the
        # episode is resolved from the watcher's vantage: acted, a terminal infra
        # JSON recorded, or already handled — re-stamping would re-fire next tick,
        # and the bounded-once lease/sentinel makes a second _failover_wedged_runpod
        # call a no-op anyway). NOT _clear_pod_safety_state (which would wipe the
        # pod-incarnation first_seen GC anchor).
        _save_pod_safety_state(
            issue,
            pod_id,
            missed=state_save["missed"],
            alerted=state_save["alerted"],
            last_progress_ts=state_save["last_progress_ts"],
            wedge_first_seen=None,
            wedge_missed=0,
            wedge_alerted=False,
            prev=state_save["prev"],
        )
    return "handled"


def _process_wedged_pod(
    issue: int, info: PodInfo, now: float, dry_run: bool, threshold: int
) -> None:
    """Handle a RUNNING managed pod observed in the #664 RunPod no-port wedge.

    The raw wedge condition (``backend_poll._pod_is_runpod_runtime_wedged``) is
    already confirmed by the caller, which has ALSO already excluded DONE-status
    pods (MF6 — those fall through to the status-class DONE auto-stop arm). Age
    the wedge against the DEDICATED ``wedge_first_seen`` clock (stamped at wedge
    ONSET, NOT the pod-incarnation ``first_seen``, MF1); below K
    (``backend_poll.RUNPOD_WEDGE_K_SEC``) KEEP + persist (the wedge has not
    matured — a healthy slow bring-up clears it on a later tick when the port
    appears); past K apply the >=threshold consecutive-wedge-checks miss guard,
    then ALERT (default) or TERMINATE+FAILOVER (#770: the SAME irreversible
    terminate + fresh re-provision the poller owns via
    :func:`_wedge_failover` -> ``backend_poll._failover_wedged_runpod``, bounded
    once via the shared durable lease + sentinel) gated on the SAME inputs-on-HF
    + (tri-state) keep-running checks as #689 fix (b). A reconstruction gap in
    the failover (no handle / a parse failure), OR a PRE-terminate raise where a
    liveness probe finds the pod still alive (#770 v2 r3), DEGRADES to the ALERT
    path (never a blind terminate, never a false terminal record while the pod
    bills); a POST-terminate raise (pod gone) routes to a durable ``"blocked"``
    terminal record instead. Persists the wedge fields via
    :func:`_save_pod_safety_state` (MF3 forward-carry) WITHOUT clearing the
    pod-incarnation ``first_seen`` GC anchor."""
    from backend_poll import RUNPOD_WEDGE_K_SEC

    pod_id = info.pod_id
    prev_state = _load_pod_safety_state(issue)

    # ── MF4: pod_id-change reset ──────────────────────────────────────────────
    # The pod-safety state is keyed on `issue`, not (issue, pod_id). If the
    # poller re-provisioned the same issue with a fresh pod_id, the persisted
    # wedge fields are stale from the OLD pod -> a fresh healthy pod could be
    # stopped during normal startup, or a long-running pod stopped after two
    # stale runtime.ports reads. Reset all wedge fields on a pod_id mismatch.
    prev_pod_id = prev_state.get("pod_id")
    if prev_pod_id != pod_id:
        wedge_first_seen: float | None = None
        prev_wedge_missed = 0
        prev_wedge_alerted = False
    else:
        prev_wfs = prev_state.get("wedge_first_seen")
        wedge_first_seen = prev_wfs if isinstance(prev_wfs, int | float) else None
        prev_wedge_missed = prev_state.get("wedge_missed", 0)
        if not isinstance(prev_wedge_missed, int):
            prev_wedge_missed = 0
        prev_wedge_alerted = bool(prev_state.get("wedge_alerted", False))

    # ── MF1: dedicated wedge_first_seen clock ─────────────────────────────────
    # Stamp `now` on the FIRST tick the raw wedge predicate is True (wedge
    # ONSET). The not-wedged branch in _process_pod calls _clear_wedge_state, so
    # a port re-appearance resets this to None -> a one-tick blip never matures.
    # This measures the actual no-port episode length, NOT pod uptime.
    if wedge_first_seen is None:
        wedge_first_seen = now  # first wedged tick this incarnation
    wedged_for = now - wedge_first_seen

    # The status-class counters are forward-carried untouched (this tick belongs
    # to the wedge arm; the status-class arm owns `missed`/`alerted` on its ticks).
    prev_missed = prev_state.get("missed", 0)
    if not isinstance(prev_missed, int):
        prev_missed = 0
    prev_progress = prev_state.get("last_progress_ts")
    if not isinstance(prev_progress, int | float):
        prev_progress = None
    prev_status_alerted = bool(prev_state.get("alerted", False))

    # ── MF2: tri-state keep-running gate ──────────────────────────────────────
    # Read the tag ONLY on the confirmed-past-K branch (the lazy pattern the
    # status-class arm uses): below K / not-yet-confirmed never auto-stops, so
    # the subprocess is paid only for a matured wedge candidate. Pass `True`
    # (not False) as the below-K placeholder so decide_pod_wedge can never STOP
    # before it would consult the real gate.
    confirmed = wedged_for > RUNPOD_WEDGE_K_SEC and (prev_wedge_missed + 1) >= threshold
    keep_running: bool | str = _wedge_keep_running(issue) if confirmed else True
    inputs_ok = _wedge_inputs_safe(issue) if (confirmed and keep_running is False) else False

    action, new_wedge_missed = decide_pod_wedge(
        wedged_for=wedged_for,
        k_floor=RUNPOD_WEDGE_K_SEC,
        wedge_missed=prev_wedge_missed,
        threshold=threshold,
        alerted=prev_wedge_alerted,
        keep_running=keep_running,
        inputs_ok=inputs_ok,
    )
    wedged_h = f"{wedged_for / 3600:.2f}h"
    print(
        f"  issue #{issue} pod={pod_id}: RUNPOD NO-PORT WEDGE wedged_for={wedged_h} "
        f"(K={RUNPOD_WEDGE_K_SEC}s) wedge_missed={prev_wedge_missed}->{new_wedge_missed} "
        f"keep_running={keep_running} inputs_ok={inputs_ok} action={action}"
    )

    if action == "terminate-failover":
        # #770: the provably-safe case routes the SAME irreversible terminate +
        # re-provision the poller owns (bounded-once via the shared durable lease
        # + sentinel), NOT a reversible stop that cannot heal a host-pinned dead
        # RunPod host (#763). The whole outcome dispatch (no-capacity / blocked ->
        # epm:failure + set-status blocked, CRITICAL #1; failover / already-handled
        # -> a progress note; alert -> defer to the alert block) lives in the
        # helper so this function stays under the C901 complexity cap.
        next_action = _handle_wedge_failover_outcome(
            issue=issue,
            info=info,
            pod_id=pod_id,
            wedged_h=wedged_h,
            threshold=threshold,
            dry_run=dry_run,
            state_save={
                "missed": prev_missed,
                "alerted": prev_status_alerted,
                "last_progress_ts": prev_progress,
                "prev": prev_state,
            },
        )
        if next_action == "handled":
            return
        # next_action == "alert": no reconstructable handle / fresh-pod race /
        # failover raised -> fall through to the existing alert block (carries the
        # clock forward + dedups the alert); we did NOT act, so re-try later.
        action = "alert"

    if action == "alert":
        post_alert = not prev_wedge_alerted
        if post_alert:
            _post_progress_marker(
                issue,
                f"{_WEDGE_ALERT_NOTE_SENTINEL} RUNNING pod {pod_id} stuck in the #664 "
                f"RUNNING-but-no-port host wedge for {wedged_h} (> {RUNPOD_WEDGE_K_SEC}s K "
                f"floor) — a billing leak the poller's _maybe_escalate_runpod_wedge did NOT "
                f"catch (the poll loop is dead). AUTO-STOP is GATED OFF this episode "
                f"(keep_running={keep_running}, inputs_ok={inputs_ok}): the inputs are not "
                f"provably safe on HF, the keep-running tag is present, or the tag read "
                f"FAILED — every uncertainty path is ALERT-only, never an unsafe stop. "
                f"Investigate and stop manually (`pod.py stop --issue {issue}`, reversible) "
                f"if the run is truly wedged. Posted once per wedge episode.",
                dry_run,
                label="wedge-alert",
            )
        print(
            f"  WEDGE-ALERT issue #{issue}: RunPod no-port wedge {wedged_h} "
            f"(gated off: keep_running={keep_running}, inputs_ok={inputs_ok}); NOT stopping.",
            file=sys.stderr,
        )
        if not dry_run:
            _save_pod_safety_state(
                issue,
                pod_id,
                missed=prev_missed,
                alerted=prev_status_alerted,
                last_progress_ts=prev_progress,
                wedge_first_seen=wedge_first_seen,
                wedge_missed=new_wedge_missed,
                wedge_alerted=True,
                prev=prev_state,
            )
        return

    # action == "keep": persist the (possibly incremented) wedge miss count and
    # the onset clock so the next tick can mature the episode.
    if not dry_run:
        _save_pod_safety_state(
            issue,
            pod_id,
            missed=prev_missed,
            alerted=prev_status_alerted,
            last_progress_ts=prev_progress,
            wedge_first_seen=wedge_first_seen,
            wedge_missed=new_wedge_missed,
            wedge_alerted=prev_wedge_alerted,
            prev=prev_state,
        )


def _escaped_pod_exemptions(issue: int, status_class: str, events: list) -> tuple[bool, bool]:
    """Lazy escaped-pod auto-stop exemptions for :func:`_process_pod`.

    Returns ``(keep_running, followup_active)``. Both only matter when the
    auto-stop arm is in play (``status_class == "auto-stop-done"``), so the
    extra ``task.py view`` subprocess + events scan are paid only for
    escaped-pod candidates. ``keep_running`` (the explicit user tag) is
    consulted first; ``followup_active`` (the inferred-from-events live inline
    follow-up) is the fallback, computed only when ``keep_running`` is False.
    Extracted from :func:`_process_pod` to keep its cyclomatic complexity under
    the C901 cap after the #692 wedge arm landed (behavior unchanged)."""
    keep_running = status_class == "auto-stop-done" and _task_keep_running(issue)
    followup_active = (
        status_class == "auto-stop-done"
        and not keep_running
        and _task_followup_active(issue, events=events)
    )
    return keep_running, followup_active


def _process_pod(
    issue: int, pod_id: str, info: PodInfo, now: float, dry_run: bool, threshold: int
) -> None:
    """Reconcile one RUNNING managed pod against its task status.

    Reads the task's status + latest real-progress timestamp, classifies it,
    and applies :func:`decide_pod_safety`: AUTO-STOP a done-or-paused task's
    escaped pod (:data:`POD_SAFETY_AUTO_STOP` — DONE, plus user-paused
    ``on_hold``, #980)
    (after the 2-miss guard, unless the task carries the ``keep-running`` tag
    OR the task's events.jsonl shows a `epm:run-launched` newer than the
    latest done-transition — i.e. a live inline follow-up — then the stop is
    SKIPPED with a log line + a once-per-pod-incarnation marker), ALERT a
    stale pod-active task once per episode, or KEEP. Persists the per-pod
    state (miss count, alerted flag, keep-running-noted flag, followup-noted
    flag, last-observed real progress) for the next tick.

    #692 wedge-arm ordering (MF6): the #664 RunPod no-port wedge arm runs
    BEFORE the status-class branches, EXCEPT that a wedged pod whose task is at
    a pod-safety auto-stop status (:data:`POD_SAFETY_AUTO_STOP` — completed /
    awaiting_promotion / archived, the established escaped-pod auto-stop case,
    plus user-paused ``on_hold``, #980) FALLS THROUGH to the
    status-class auto-stop arm, which already auto-stops it. A DONE-task pod has
    no live work to strand (and a paused task's workload must NOT be relaunched
    by the wedge arm's terminate+failover), so the canonical escaped-pod
    auto-stop wins; routing it
    through the wedge arm's ALERT-default + inputs-gate would only WEAKEN the
    existing auto-stop into a conditional one. The wedge arm therefore handles
    only wedged pods OUTSIDE that set (the live-work statuses where the watcher
    must be conservative). A non-wedged pod reaches the status-class branches
    unchanged, exactly as before #692."""
    status = _task_status(issue)

    # ── #692 RunPod no-port wedge backstop (runs BEFORE the status-class arm,
    # MF6 DONE-task carve-out inside the helper) ──────────────────────────────
    # When the per-issue poll loop has DIED, backend_poll's
    # _maybe_escalate_runpod_wedge never runs and the #664 RUNNING-but-no-port
    # billing leak goes undetected. The watcher runs unconditionally every 10
    # min, so it is the backstop. If the helper handled the pod (a non-DONE
    # wedged pod), return; otherwise fall through to the status-class branches
    # (a non-wedged pod, OR a wedged DONE-task pod that the status-class DONE
    # auto-stop arm handles canonically) exactly as before #692.
    if _maybe_handle_runpod_wedge(issue, status, info, now, dry_run, threshold):
        return

    events = _task_events(issue)
    latest_progress = _latest_progress_ts(events)
    status_class = _status_class(status, latest_progress, now)
    keep_running, followup_active = _escaped_pod_exemptions(issue, status_class, events)

    prev_state = _load_pod_safety_state(issue)
    prev_missed = prev_state.get("missed", 0)
    if not isinstance(prev_missed, int):
        prev_missed = 0
    prev_alerted = bool(prev_state.get("alerted", False))
    prev_keep_running_noted = bool(prev_state.get("keep_running_noted", False))
    prev_followup_noted = bool(prev_state.get("followup_noted", False))
    prev_stop_failed_noted = bool(prev_state.get("stop_failed_noted", False))
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
    _apply_pod_safety_action(
        action,
        issue=issue,
        pod_id=pod_id,
        status=status,
        now=now,
        dry_run=dry_run,
        threshold=threshold,
        gap_h=gap_h,
        new_missed=new_missed,
        alerted=alerted,
        latest_progress=latest_progress,
        prev_state=prev_state,
        prev_keep_running_noted=prev_keep_running_noted,
        prev_followup_noted=prev_followup_noted,
        prev_stop_failed_noted=prev_stop_failed_noted,
    )


def _apply_pod_safety_action(  # noqa: C901 — flat per-action dispatcher; the #1155 stop-failed sub-branch adds guard branches, not nesting
    action: str,
    *,
    issue: int,
    pod_id: str,
    status: str | None,
    now: float,
    dry_run: bool,
    threshold: int,
    gap_h: str,
    new_missed: int,
    alerted: bool,
    latest_progress: float | None,
    prev_state: dict,
    prev_keep_running_noted: bool,
    prev_followup_noted: bool,
    prev_stop_failed_noted: bool,
) -> None:
    """Apply the status-class :func:`decide_pod_safety` ``action`` for one pod.

    Extracted verbatim from :func:`_process_pod` (behavior unchanged) to keep its
    cyclomatic complexity under the C901 cap after the #692 wedge arm landed. The
    five actions — ``keep-running-skip`` / ``followup-skip`` / ``stop`` (whose
    REAL-failure sub-branch posts a once-per-episode durable ``stop-failed``
    marker and keeps the episode retryable, #1155) / ``alert`` / ``keep`` — post
    the appropriate once-per-episode marker (deduped
    via the prev-state flags) and persist the per-pod state. Each save
    forward-carries the #692 wedge fields untouched (this is a status-class tick;
    the wedge arm owns them on its own ticks)."""
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
                f"'{status}' (DONE or user-paused — no live run should hold a pod "
                f"at this status), so the pod is an escaped / "
                f"Step-8-terminate-failed pod (pod_id={pod_id}); reversible pause, "
                f"volume preserved (pod.py resume). Confirmed for >= {threshold} checks.",
                dry_run,
                label="auto-stop",
            )
            if not dry_run:
                _clear_pod_safety_state(issue)
            return
        if dry_run:
            # _stop_pod returns False under dry-run BY CONSTRUCTION (no
            # subprocess ran) — not a stop failure. Preserve the pinned
            # dry-run contract: "would stop pod" log line only, no marker,
            # no state save (test_pod_safety_auto_stop_dry_run_no_mutation).
            return
        # Real stop failure (`pod.py stop` rc != 0; _stop_pod already printed
        # POD STOP FAILED to stderr): make it durably visible ONCE per
        # episode, and keep the episode RETRYABLE — save state WITHOUT
        # clearing it and WITHOUT touching the on-disk miss count, so
        # decide_pod_safety re-fires "stop" on the next tick (#1155).
        if not prev_stop_failed_noted:
            # Compound-failure residual: _post_progress_marker swallows
            # SubprocessError/OSError with a stderr WARNING, so if the stop
            # API fails AND the marker post fails, this episode's durable
            # record is lost — degraded state == today's stderr-only baseline.
            _post_progress_marker(
                issue,
                f"{_AUTOSTOP_FAILED_NOTE_SENTINEL} pod-safety auto-stop FAILED "
                f"(pod_id={pod_id}; task status '{status}'): `pod.py stop "
                f"--issue {issue}` exited non-zero (API error on the watcher "
                f"cron log stderr). The stop is RETRIED every ~10-min tick "
                f"until it succeeds; the pod keeps BILLING until then — if "
                f"this persists, stop it manually (`pod.py stop --issue "
                f"{issue}`) and check `pod.py list-ephemeral`. Posted once "
                f"per pod-safety episode.",
                dry_run,
                label="stop-failed",
            )
        prev_missed = prev_state.get("missed", 0)
        if not isinstance(prev_missed, int):
            prev_missed = 0
        _save_pod_safety_state(
            issue,
            pod_id,
            missed=prev_missed,  # unchanged count -> next tick re-fires "stop"
            alerted=alerted,
            last_progress_ts=latest_progress,
            stop_failed_noted=True,
            prev=prev_state,
        )
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


def _long_phase_heartbeat_reason(events: list[dict], now: float) -> str | None:
    """Human-readable exemption reason when the issue's NEWEST non-watcher
    ``epm:progress`` marker opts into the long-phase-heartbeat leash AND is
    still fresh, else ``None``.

    A legitimately-slow phase (off-pod analyzer verifier rounds, in-flight
    Anthropic Batch polling) emits few markers, so both stalled-detector
    staleness signals expire between its heartbeats and the detector
    false-fires (incident #761: a 1h21m off-pod analysis stretch drew a
    wasted auto-respawn). An emitter opts into a wider leash by stamping
    :data:`_LONG_PHASE_HEARTBEAT_PREFIX` into its ``epm:progress`` note; this
    helper recognizes that opt-in. Pure over already-loaded ``events`` — no
    subprocess, no second read.

    Predicate: among events with ``kind == "epm:progress"`` whose note is NOT
    a watcher-self post (:data:`_WATCHER_NOTE_SENTINELS`) AND whose note
    contains :data:`_LONG_PHASE_HEARTBEAT_PREFIX`, take the newest by ``ts``;
    return a reason iff its age (``now - ts``) is in
    ``[0, LONG_PHASE_HEARTBEAT_FRESH_S)``. A future ``ts`` (negative age,
    clock skew / fake clock) is NOT fresh — never mask a genuinely stalled
    session.
    """
    best_ts: float | None = None
    for ev in events:
        if ev.get("kind") != "epm:progress":
            continue
        note = ev.get("note") or ""
        if any(sentinel in note for sentinel in _WATCHER_NOTE_SENTINELS):
            continue  # a watcher-posted alert — never a real heartbeat
        if _LONG_PHASE_HEARTBEAT_PREFIX not in note:
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and (best_ts is None or ts > best_ts):
            best_ts = ts
    if best_ts is None:
        return None
    age = now - best_ts
    if 0 <= age < LONG_PHASE_HEARTBEAT_FRESH_S:
        return f"fresh long-phase heartbeat ({age / 60:.1f}m old)"
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
# is the legacy children-in-flight shape, not this round stranded). That
# condition is NECESSARY but — after #778 (2026-07-02), where the parent
# pass's own post-9b tail parks (step 10 / 9a-bis) postdated a freshly
# posted 9b scope and the watcher re-parked a MID-FLIGHT round twice — no
# longer SUFFICIENT. Two purely event-based gates (#837) also bind:
#   Gate 1 (round-start witness): at least one round-only event newer than
#     the scope — a kind in ``_FOLLOWUP_ROUND_WITNESS_KINDS`` or an
#     ``epm:progress`` note beginning
#     ``_FOLLOWUP_STAGE_DISPATCH_WITNESS_PREFIX`` — must prove the round
#     actually STARTED (closes the pre-activity race where the
#     mis-attributed parent park is the newest event).
#   Gate 2 (freshness): NO non-watcher event (``_WATCHER_NOTE_SENTINELS``
#     note filter) may be strictly newer than the keyed round-end park —
#     anything newer proves the round is still executing past it (or died
#     mid-flight, which is crash-recovery's job, never the repark's).
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
# behavior — and whenever an UNRUN scope exists the awaiting-child
# suppression itself STANDS DOWN (#837 §4d,
# :func:`_followups_awaiting_child_reason`), so the gated-``None`` /
# failed-re-park fall-through reaches RESPAWN (the designed recovery for a
# pending or executing round; #778's 05:01Z replacement session
# demonstrated it live) instead of latching alert-only on a task with open
# children. A task with NO ``epm:followup-scope`` on record (the legacy
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

# Event kinds that can ONLY be posted by an EXECUTING pipeline round (#837
# round-start witness gate). Temporal-impossibility membership criterion: the
# PARENT pass's post-9b tail (interp / clean-result critiques, methodology
# export, status changes, step-completed parks, `epm:merged`, codex-task-*
# bookkeeping) can legitimately postdate the round's ``epm:followup-scope``,
# but planning / implementation / launch markers CANNOT — those steps are
# long over for the parent by the time the 9b scope posts, so any of these
# kinds NEWER than the scope proves the round actually STARTED. Add a kind
# here only when a post-9b parent tail could never post it. The watcher
# itself can never be a witness by construction: it posts none of these
# kinds, and its ``epm:progress`` notes begin
# ``[autonomous_session_watch:...]``, which fails the stage-dispatch
# ``startswith`` check below.
_FOLLOWUP_ROUND_WITNESS_KINDS = frozenset(
    {
        "epm:plan",
        "epm:plan-approved",
        "epm:plan-verify",
        "epm:consistency",
        "epm:smoke-architecture-check",
        "epm:proposed-tests",
        "epm:experiment-implementation",
        "epm:code-review",
        "epm:code-review-codex",
        "epm:review-reconcile",
        "epm:run-launched",
        "epm:launch",
    }
)

# The same-issue loop's dispatch breadcrumb (SKILL.md § Dispatch breadcrumb —
# a NON-SKIPPABLE contract for every follow-up-loop stage dispatch): an
# ``epm:progress`` note BEGINNING with this prefix marks a follow-up-round
# stage dispatch. The ``followup-`` restriction excludes parent-tail
# breadcrumbs (methodology spawn, Step 8/9 stages), which carry other
# ``stage=`` values.
_FOLLOWUP_STAGE_DISPATCH_WITNESS_PREFIX = "stage-dispatch stage=followup-"

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


def _latest_scope_and_run_ts(events: list[dict]) -> tuple[float | None, float | None]:
    """Newest epoch ts of an ``epm:followup-scope`` and of an
    ``epm:same-issue-followup-run`` in ``events`` (either slot ``None`` when
    absent / unparseable). A pure "newest signal" primitive — the UNRUN
    predicate itself is LABEL-keyed as of #894
    (``task_workflow.unrun_followup_labels``; a ts-only unrun read is blind
    to an older queued label behind a newer label's run marker, the #763
    shape). Used by :func:`_followup_round_complete_reason` for the
    ``max(scope_ts, run_ts)`` gate anchor."""
    scope_ts: float | None = None
    run_ts: float | None = None
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
        elif run_ts is None or ts > run_ts:
            run_ts = ts
    return scope_ts, run_ts


def _has_round_start_witness(events: list[dict], anchor_ts: float) -> bool:
    """True iff ``events`` carry a ROUND-START WITNESS strictly newer than
    ``anchor_ts``: a round-only kind (:data:`_FOLLOWUP_ROUND_WITNESS_KINDS`)
    or a same-issue-loop stage-dispatch breadcrumb (an ``epm:progress`` note
    beginning :data:`_FOLLOWUP_STAGE_DISPATCH_WITNESS_PREFIX`). The anchor is
    ``max(scope_ts, run_ts)`` (#894) — so a PRIOR recorded round's witness
    can never vouch for a later queued label's round. Strict ``>``
    (#837 §12.2): a witness sharing the anchor's second fails toward blocking
    → respawn, the safe direction."""
    for ev in events:
        ev_ts = _parse_event_ts(ev.get("ts"))
        if ev_ts is None or ev_ts <= anchor_ts:
            continue
        kind = ev.get("kind")
        if kind in _FOLLOWUP_ROUND_WITNESS_KINDS:
            return True
        if kind == "epm:progress" and (ev.get("note") or "").startswith(
            _FOLLOWUP_STAGE_DISPATCH_WITNESS_PREFIX
        ):
            return True
    return False


def _has_nonwatcher_event_after(events: list[dict], ts: float) -> bool:
    """True iff any NON-watcher event (note-substring filter
    :data:`_WATCHER_NOTE_SENTINELS`, same as :func:`_latest_step_completed`)
    in ``events`` is strictly newer than ``ts``. Deliberate session-stop
    records are excluded (#990/#1053) — see the inline comment."""
    for ev in events:
        ev_ts = _parse_event_ts(ev.get("ts"))
        if ev_ts is None or ev_ts <= ts:
            continue
        note = ev.get("note") or ""
        if any(sentinel in note for sentinel in _WATCHER_NOTE_SENTINELS):
            continue
        # A deliberate session stop — incl. the #1053 Step-0 collision-exit /
        # stale-wake-yield breadcrumb — is the death record of the task's
        # driver, never round activity: it must not veto the follow-up
        # round-repark via the #837 freshness gate (#990/#1053; same two-leg
        # exclusion as _latest_nonwatcher_event_ts / _latest_progress_ts).
        if note.lstrip().startswith("deliberate-stop ") or ev.get("by") == "spawn_session-stop":
            continue
        return True
    return False


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

    STANDS DOWN (returns ``None``) whenever an UNRUN ``followup_label``
    exists — a label group with no matching ``epm:same-issue-followup-run``
    (``task_workflow.unrun_followup_labels``; #837, label-keyed as of #894):
    an unrun label means a same-issue follow-up round is pending, executing,
    or QUEUED, and respawn is ALWAYS the correct recovery there (a respawned
    session's Step 0 routes into the follow-up loop — demonstrated live by
    #778's 05:01Z replacement session), so the children-wait suppression must
    not latch alert-only on a mis-attributed parent-tail step-10 park. The
    old newest-ts read reproduced the #763 blindness: an earlier queued
    label's scope stayed invisible behind a later label's run marker
    (``run_ts > scope_ts``), so the stand-down never fired and the
    awaiting-child latch suppressed the respawn that would correctly
    dispatch the queued label. The legacy children-in-flight shape has NO
    scope (or every label RECORDED), so its suppression semantics are
    untouched.

    Probed LAZILY by the callers (the helper is only invoked when the stalled
    / orphan pass already wants to respawn) so a healthy session never pays
    the ``task.py list-children`` subprocess.
    """
    if status != "followups_running":
        return None
    if has_pod:
        return None
    from explore_persona_space.task_workflow import unrun_followup_labels

    if unrun_followup_labels(events):
        # #837 stand-down (label-keyed, #894): ANY unrun label → the
        # fall-through must reach RESPAWN, never the awaiting-child latch
        # (the reconciler-verified freeze shape: the SAME mis-attributed
        # step-10 park the repark predicate's witness gate vetoes would
        # otherwise satisfy this suppression forever on a task with open
        # children — #778 has #816). Checked BEFORE the children lookup so
        # the stand-down stays pure (no subprocess).
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


def _followup_round_complete_reason(events: list[dict], *, issue: int | None = None) -> str | None:
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
    - EVERY label is RECORDED — ``task_workflow.unrun_followup_labels`` is
      empty (label-keyed as of #894; the old ``run_ts > scope_ts`` exit was
      blind to an OLDER queued label's round executing AFTER a newer label's
      run marker — the #763 shape). The designed ordering posts the run
      marker only AFTER the re-park (loop step 3 → step 4), so a
      ``followups_running`` status alongside an all-recorded history is a
      LATER legitimate transition (the legacy children-in-flight shape via
      Step 10 step 5), NOT a round stranded — defer to the awaiting-child
      suppression. This also self-disarms the predicate after the watcher's
      own re-park, which posts the completion marker itself
      (:func:`_repark_completed_followup_round`);
    - (#837 gate 1 — round-start witness) no round-only event newer than
      ``anchor_ts = max(scope_ts, run_ts)`` proves the round STARTED: the
      matched park is the parent pass's own tail (#778: the 9a-bis park 14
      min after the 9b scope), a PRIOR recorded round's witness (#894: an
      older queued label must not borrow a newer completed round's
      witness), or the round never started — either way the repark would
      falsely close the round; defer to crash-recovery / stalled handling;
    - (#837 gate 2 — freshness) ANY non-watcher event is strictly newer
      than the keyed round-end park: the park is stale / mis-attributed —
      the round is still executing past it (#778: round activity HOURS
      newer at both firings), or died mid-flight (crash-recovery's job,
      never the repark's).

    The gates anchor on ``max(scope_ts, run_ts)`` — identical to the old
    ``scope_ts`` anchor whenever a round is unrun in the single-label world
    (``run_ts <= scope_ts`` or absent), so every existing #837/#533/#778
    fixture verdict is preserved.

    ``issue`` is used only for the log-and-skip diagnostics; ``None`` keeps
    every existing caller/test signature working.

    Pure over the already-loaded ``events`` — no subprocess.
    """
    scope_ts, run_marker_ts = _latest_scope_and_run_ts(events)
    if scope_ts is None:
        return None
    from explore_persona_space.task_workflow import unrun_followup_labels

    if not unrun_followup_labels(events):
        return None
    anchor_ts = scope_ts if run_marker_ts is None else max(scope_ts, run_marker_ts)
    sc = _latest_step_completed(events)
    if sc is None:
        return None
    step = sc.get("step")
    sc_ts = _parse_event_ts(sc.get("ts"))
    if (
        step in _FOLLOWUP_ROUND_END_STEPS
        and sc.get("exit_kind") == _FOLLOWUPS_CHILDREN_WAIT_EXIT_KIND
        and sc_ts is not None
        and sc_ts > anchor_ts
    ):
        who = f"issue #{issue}" if issue is not None else "followup-round-repark probe"
        # Gate 1 — round-start witness (#837): the round must have
        # DEMONSTRABLY started (a round-only kind or a followup stage-
        # dispatch breadcrumb newer than the anchor — max(scope, run), so a
        # prior recorded round's witness can never vouch for a later queued
        # label's round, #894). Closes the #778 pre-activity race (scope →
        # mis-attributed parent-tail park → nothing yet) deterministically,
        # with no wall clock.
        if not _has_round_start_witness(events, anchor_ts):
            print(
                f"  {who}: round-end park matched but no round-start witness "
                f"after the scope — the park is the parent pass's tail or the "
                f"round never started; deferring to crash-recovery/stalled "
                f"handling"
            )
            return None
        # Gate 2 — freshness (#837): the keyed park must be the LATEST
        # non-watcher signal. ANY non-watcher event strictly newer than
        # sc_ts proves the park is stale or mis-attributed (round still
        # executing, or advanced past it). Cost of a false block is one
        # respawn cycle (§4b convergence-through-respawn: the respawned
        # session posts a FRESH round-end park that becomes the newest
        # signal). Repeated cross-posts — e.g. a chain of workflow-fix
        # children applying `epm:workflow-fix-applied` on the parent — can
        # block repeated respawn cycles until the respawn budget exhausts,
        # ending in a LOUD exhausted alert: bounded and observable, vs the
        # unbounded cost of a false fire (orphaning a live round).
        if _has_nonwatcher_event_after(events, sc_ts):
            print(
                f"  {who}: round activity newer than the keyed round-end park "
                f"— round still executing (or died mid-flight: crash-"
                f"recovery's job, never the repark's); skipping"
            )
            return None
        return (
            f"same-issue follow-up round complete (round-end "
            f"epm:step-completed step={step} exit_kind=parked newer than the "
            f"round's epm:followup-scope) but the task is still at "
            f"followups_running — the owning session exited before the "
            f"designed re-park"
        )
    return None


# Marker kind posted by ``task.py set-status --auto-approve-if-autonomous`` when
# an autonomous plan estimate exceeds ``EPM_PLAN_AUTOAPPROVE_GPU_HOURS``. In the
# status-hold variant (SKILL.md Step 9b same-issue follow-up loop) the task is
# HELD at the ACTIVE status ``followups_running`` and only this marker records
# the park; in the plan_pending variant the status moves to ``plan_pending``
# (already PARK at the decide() layer, so not vulnerable to the respawn loop).
_SPEND_APPROVAL_MARKER_KIND = "epm:awaiting-spend-approval"


def _latest_spend_approval_ts(events: list[dict]) -> float | None:
    """Newest epoch ts of an ``epm:awaiting-spend-approval`` marker in
    ``events`` (the over-cap autonomous plan-gate park), or ``None``."""
    best: float | None = None
    for ev in events:
        if ev.get("kind") != _SPEND_APPROVAL_MARKER_KIND:
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is None:
            continue
        if best is None or ts > best:
            best = ts
    return best


def _spend_approval_park_reason(events: list[dict]) -> str | None:
    """Human-readable exemption reason when the task is parked at the over-cap
    autonomous plan-gate (``epm:awaiting-spend-approval``) — a user-only gate
    that respawning the session cannot clear. Returns ``None`` when the
    exemption does not apply.

    Fires when the latest ``epm:awaiting-spend-approval`` marker is NOT
    superseded by any later REAL progress — i.e. nothing newer than it except,
    at most, an ``epm:step-completed exit_kind=parked`` (the re-post each
    respawned session leaves before exiting) and the watcher's own
    sentinel-noted markers (already excluded by :func:`_latest_progress_ts`).
    A real downstream marker newer than the park means the gate was resolved
    (the user approved / re-planned and the session advanced), so the exemption
    correctly stops applying.

    Pure over the already-loaded ``events`` — no subprocess.
    """
    spend_ts = _latest_spend_approval_ts(events)
    if spend_ts is None:
        return None
    progress_ts = _latest_progress_ts(events)
    if progress_ts is not None and progress_ts > spend_ts:
        # A real (non-watcher) progress marker newer than the park — the gate
        # has been resolved and the session advanced; do not suppress.
        return None
    # Any newer non-watcher event must be an epm:step-completed exit_kind=parked
    # (the respawn-and-park re-post). Anything else with a newer ts that is NOT
    # in the progress/watcher filter would have shown above; a parked
    # step-completed is the expected accompaniment, so the park stands.
    return (
        "parked at the over-cap autonomous plan-gate "
        "(epm:awaiting-spend-approval is the latest non-watcher event; est "
        "GPU-h exceeds the auto-approve cap) — a user-only gate "
        "(task.py set-status <N> approved, or re-plan); respawning the session "
        "only re-reads the parked plan and re-posts the same "
        "epm:step-completed exit_kind=parked, never advancing"
    )


def _spend_approval_skip_already_noted(events: list[dict]) -> bool:
    """True iff a marker carrying :data:`_SPEND_APPROVAL_SKIP_NOTE_SENTINEL`
    already exists NEWER than the latest ``epm:awaiting-spend-approval`` — the
    self-contained once-per-episode dedup (no extra state field). A fresh
    spend-approval episode (a later park marker) re-arms the alert."""
    spend_ts = _latest_spend_approval_ts(events)
    if spend_ts is None:
        return False
    for ev in events:
        note = ev.get("note") or ""
        if _SPEND_APPROVAL_SKIP_NOTE_SENTINEL not in note:
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and ts >= spend_ts:
            return True
    return False


def _deliberate_blocked_park_reason(events: list[dict]) -> str | None:
    """Reason string when the newest ``epm:status-changed`` is corroborated by
    the halt-contract trail — an ``epm:failure`` marker within
    [:data:`_BLOCKED_PARK_FAILURE_WINDOW_S` before,
    :data:`_BLOCKED_PARK_FAILURE_SLACK_AFTER_S` after] it — i.e. the block is
    a deliberate park (post epm:failure, set status blocked, exit: the
    CLAUDE.md halt contract), not an unexplained freeze (#1137; #1092's
    15:33Z alert fired 2h12m after a 1s-apart failure+blocked pair). Pure
    over the already-loaded ``events``; the CALLER gates on ``task_status ==
    "blocked"`` (only there is the newest status-changed the transition into
    blocked). Fail direction: unparseable ts / no trail -> ``None`` (the
    one-time alert still fires)."""
    sc_ts: float | None = None
    for ev in events:
        if ev.get("kind") == "epm:status-changed":
            ts = _parse_event_ts(ev.get("ts"))
            if ts is not None and (sc_ts is None or ts > sc_ts):
                sc_ts = ts
    if sc_ts is None:
        return None
    lo = sc_ts - _BLOCKED_PARK_FAILURE_WINDOW_S
    hi = sc_ts + _BLOCKED_PARK_FAILURE_SLACK_AFTER_S
    for ev in events:
        if ev.get("kind") != "epm:failure":
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is None:
            continue
        if lo <= ts <= hi:
            return (
                "deliberately parked at status=blocked (halt-contract trail: "
                "epm:failure within the park window of the newest "
                "epm:status-changed); the gate-push pass already notified the "
                "blocked transition"
            )
    return None


# Anchored, case-SENSITIVE prefix of a prose user-pause hold note (incident
# #816). All four observed hold variants (the canonical SKILL.md durable-park
# note, the #816 incident note, the older #919 sketch, the #920 chat note)
# begin with this literal; every observed quote-class marker carries it
# mid-note only, so the anchor separates real holds from quotes. Lowercase
# "user pause ..." occurs in ordinary discussion prose and must NOT arm the
# probe — the match is deliberately case-sensitive.
_USER_PAUSE_NOTE_PREFIX = "USER PAUSE"


def _latest_user_pause_ts(events: list[dict]) -> float | None:
    """Newest epoch ts among non-watcher events whose note BEGINS with
    ``USER PAUSE`` (anchored prefix, case-sensitive, ANY marker kind) — the
    prose-hold shape of incident #816. Watcher-sentinel notes are excluded
    (defense against future alert-text drift; today's alert text begins with
    the ``[autonomous_session_watch:...]`` sentinel so it cannot match the
    anchor anyway). An event with an unparseable/absent ``ts`` is skipped —
    fail direction: a ts-less pause note leaves the probe inert and the
    respawn proceeds (the incident direction, alert-free)."""
    best: float | None = None
    for ev in events:
        note = ev.get("note") or ""
        if not note.lstrip().startswith(_USER_PAUSE_NOTE_PREFIX):
            continue
        if any(sentinel in note for sentinel in _WATCHER_NOTE_SENTINELS):
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def _user_pause_hold_reason(events: list[dict]) -> str | None:
    """Human-readable exemption reason when the latest word on the task is a
    prose ``USER PAUSE`` hold (incident #816 defense-in-depth) — the newest
    USER-PAUSE-prefixed non-watcher note is not superseded by any STRICTLY
    newer real progress marker. Returns ``None`` when the exemption does not
    apply. Pure over the already-loaded ``events`` — no subprocess.

    Strict ``>`` is load-bearing: the pause note itself usually rides a
    :data:`_PROGRESS_KINDS` kind (``epm:progress`` in the #816 incident,
    ``epm:status-changed`` for the canonical durable-park note), so it
    SELF-INCLUDES in :func:`_latest_progress_ts` — ``progress_ts == pause_ts``
    means "the pause IS the latest word" and must keep suppressing.

    Fail directions, both deliberate: (a) a pause note with an unparseable
    ``ts`` leaves the probe inert (respawn proceeds — the incident direction);
    (b) outside the canonical resume paths (the SKILL.md ``set-status``
    resume posts ``epm:status-changed``; a resumed live session posts real
    progress markers; a REGISTERED session bypasses the orphan probe
    entirely), suppression persists until ANY real progress marker,
    set-status, fresh pause note, or registered spawn lands — the one-time
    alert is the only signal in that window.

    Note the CANONICAL durable-park note (SKILL.md § User pause affordance)
    also begins ``USER PAUSE`` (on the ``set-status <N> on_hold``
    ``epm:status-changed`` row) and arms this probe HARMLESSLY: ``on_hold``
    is in the watcher PARK set, so a durably-parked task is never
    orphan-evaluated in the first place."""
    pause_ts = _latest_user_pause_ts(events)
    if pause_ts is None:
        return None
    progress_ts = _latest_progress_ts(events)
    if progress_ts is not None and progress_ts > pause_ts:
        # A real (non-watcher) progress marker STRICTLY newer than the pause
        # note — the hold was resumed / superseded; do not suppress.
        return None
    return (
        "prose USER PAUSE hold (a non-watcher note beginning 'USER PAUSE' is "
        "the latest word — no real progress marker postdates it): a "
        "deliberate user hold the watcher must not respawn against "
        "(incident #816); the durable fix is the SKILL.md pause procedure "
        "(pod stop first, then task.py set-status <N> on_hold)"
    )


def _user_pause_skip_already_noted(events: list[dict]) -> bool:
    """True iff a marker carrying :data:`_USER_PAUSE_SKIP_NOTE_SENTINEL`
    already exists with ts >= the latest USER-PAUSE note — the self-contained
    once-per-episode dedup (no extra state field); a FRESH pause note (a later
    ts) re-arms the alert. Mirror of
    :func:`_spend_approval_skip_already_noted` (the ``>=`` here vs its ``>``
    is deliberate: a skip alert posted the same second as the pause note
    still dedups)."""
    pause_ts = _latest_user_pause_ts(events)
    if pause_ts is None:
        return False
    for ev in events:
        note = ev.get("note") or ""
        if _USER_PAUSE_SKIP_NOTE_SENTINEL not in note:
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and ts >= pause_ts:
            return True
    return False


def _maybe_post_user_pause_skip(issue: int, reason: str, events: list[dict], dry_run: bool) -> None:
    """Post the one-time user-pause-hold-skip alert (events-log dedup via
    :func:`_user_pause_skip_already_noted`). Shared by the orphan handler and
    the stalled-pass branch so both arms stay under the C901 cap and share
    one episode-dedup. The resume-recipe example deliberately does NOT begin
    with the bare ``USER PAUSE`` literal — a resume note that did would
    re-arm the very probe it resumes."""
    if _user_pause_skip_already_noted(events):
        return
    _post_progress_marker(
        issue,
        f"{_USER_PAUSE_SKIP_NOTE_SENTINEL} {reason}. "
        f"Respawn suppressed (does NOT consume the daily respawn budget). "
        f"A prose-only hold is NOT durable — make it durable per "
        f".claude/skills/issue/SKILL.md § User pause affordance: "
        f"CRON-TEARDOWN + stop any RUNNING pod FIRST "
        f"(`pod.py stop --issue {issue}`), THEN "
        f'`task.py set-status {issue} on_hold --note "USER PAUSE '
        f"(verbatim: '...'); paused_from=<status>; resume: ...\"` as the "
        f"commit point. To resume instead: post a real progress note that "
        f"does NOT begin with 'USER PAUSE' "
        f"(`task.py post-marker {issue} epm:progress --note '<resume note>'`) "
        f"or `spawn_session.py spawn-issue --issue {issue} --auto`.",
        dry_run,
        label="user-pause-hold-skip",
    )


def _post_followup_run_marker(issue: int, events: list[dict], dry_run: bool) -> bool:
    """Post the ``epm:same-issue-followup-run v1`` completion marker for the
    round the watcher just re-parked, closing the round's
    ``epm:followup-scope`` for `/issue` Step 0 routing — WITHOUT it the
    label stays UNRUN and a re-invoked session would re-dispatch the
    already-completed round (the Step 0 dispatch table routes a post-result
    status + unrun label back into the follow-up loop). ``followup_label``
    + ``source`` come from ``task_workflow.executing_followup_label``
    (breadcrumb-first, dispatchable-queue-head fallback — #894: parsing the
    LATEST scope closes the WRONG label when the executing round is an
    older queued label, stranding the executed label unrun and closing a
    never-run one) so the idempotency match and the autonomous round-cap
    counting stay correct; ``round`` is 1 + the count of existing run
    markers. Fail-soft: an unresolvable round (no dispatchable unrun label
    — e.g. every scope is an unlabeled pseudo-label repair item) or a
    failed post logs LOUDLY to stderr naming the repair and returns False —
    the re-park itself (the substance) already happened."""
    from explore_persona_space.task_workflow import executing_followup_label

    group = executing_followup_label(events)
    if group is None or not group.get("dispatchable"):
        print(
            f"  issue #{issue}: cannot post epm:same-issue-followup-run — "
            f"task_workflow.executing_followup_label resolved no dispatchable "
            f"unrun label (unlabeled/pseudo-label scopes are repair items, never "
            f"rounds); the label stays unrun (Step 0 may re-route into the loop). "
            f"REPAIR: re-post the epm:followup-scope with a proper kebab-slug "
            f"`followup_label:` line, or retro-close it per the Step 0 "
            f"stale-label disposition rule",
            file=sys.stderr,
        )
        return False
    label = group["followup_label"]
    source = group.get("source") or "unknown"
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
        res = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "post-marker",
                str(issue),
                "epm:same-issue-followup-run",
                "--by",
                "autonomous_session_watch",
                "--note",
                note,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        _forward_marker_child_stderr(res, f"epm:same-issue-followup-run on #{issue}")
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
    _forward_marker_child_stderr(out, f"set-status awaiting_promotion on #{issue}")
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


def _worktree_recent_activity(
    issue: int, now: float, window_s: float, *, deadline_s: float = 2.0
) -> bool:
    """True iff any file under ``.claude/worktrees/issue-<N>`` (or an
    ``issue-<N>-<suffix>`` follow-up worktree) has an mtime within
    ``window_s`` of ``now`` — direct evidence an implementer/analyzer is
    mid-edit (#845 b). NOT :func:`_newest_mtime` (a full walk is too
    expensive for a 100+ GB worktree inside a 10-min-cron pass): early-exit
    on the FIRST fresh hit, bounded by a ``deadline_s`` wall-clock budget.

    Exclusions: per-issue download caches (``data/``) — bulk artifact
    writes, not editing activity — and ``.git`` trees. A negative age
    (mtime in the future of ``now``: clock skew, or a caller-supplied fake
    clock) is NOT fresh. Deadline exceeded / unreadable roots -> False (no
    corroborated activity -> fall back to today's respawn behavior; a
    MISSING hold only costs the pre-#845 latency, while a WRONG hold would
    defer recovery)."""
    wt_root = PROJECT_ROOT / ".claude" / "worktrees"
    roots = [wt_root / f"issue-{issue}", *sorted(wt_root.glob(f"issue-{issue}-*"))]
    start = time.monotonic()
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _e: None):
            dirnames[:] = [d for d in dirnames if d not in ("data", ".git")]
            if time.monotonic() - start > deadline_s:
                return False
            for name in filenames:
                try:
                    age = now - os.stat(os.path.join(dirpath, name)).st_mtime
                except OSError:
                    continue
                if 0 <= age < window_s:
                    return True
    return False


def _transcript_tail_rows(pid: int, max_bytes: int = 65536) -> list[dict] | None:
    """Parse the trailing ``max_bytes`` of the session's Claude transcript
    into JSON rows for :func:`decide_prompt_wedge` (#845 e), or ``None``
    when the transcript is unresolvable (fail toward NO-WEDGE).

    Resolution via the happy-log path ONLY — the same deliberate contract as
    :func:`_transcript_idle_age_s` (the resolver's filesystem fallback can
    attribute another session's transcript; a WRONG tail is worse than a
    missing one). The first (possibly partial) line of a mid-file seek is
    dropped; malformed lines are skipped (never a crash — the transcript is
    being appended to concurrently)."""
    transcript, _reason = session_resolver._resolve_transcript_via_happy_log(pid)
    if transcript is None:
        return None
    try:
        with open(transcript, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            raw = fh.read()
    except OSError:
        return None
    lines = raw.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # drop the partial first line of a mid-file seek
    rows: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _boot_death_transcript_rows(
    pid: int, max_bytes: int = BOOT_DEATH_TAIL_BYTES
) -> tuple[list[dict] | None, str | None, int | None]:
    """Whole-transcript rows for the boot-death lane's ARM 1 (#1267), as
    ``(rows, transcript_path, size_bytes)`` — ``rows`` is ``None`` when the
    transcript is unresolvable OR larger than ``max_bytes`` (a >256 KB
    transcript cannot be a ZERO-RESPONSE boot-death — arm 1 treats it as
    not-eligible; the #1287 arm-2 seek-tail read handles larger files).
    ``transcript_path`` / ``size_bytes`` are best-effort forensics for
    the sidecar row (``None`` when unresolvable).

    Unlike :func:`_transcript_tail_rows`, the WHOLE-file guarantee is
    required here: "zero response rows" must be a whole-file property, never
    a truncated-tail read. Deliberately a SIBLING of ``_transcript_tail_rows``
    rather than a refactor of it — its mid-file-seek partial-line handling is
    pinned by the wedge tests, and this probe never seeks. Same happy-log-only
    resolution contract (a WRONG transcript is worse than a missing one)."""
    transcript, _reason = session_resolver._resolve_transcript_via_happy_log(pid)
    if transcript is None:
        return None, None, None
    try:
        size = os.path.getsize(transcript)
        if size > max_bytes:
            return None, str(transcript), size
        raw = Path(transcript).read_bytes()
    except OSError:
        return None, str(transcript), None
    rows: list[dict] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, str(transcript), size


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
        # #902: thread `--stop-source watcher` — watcher-sourced stops post
        # NO deliberate-stop breadcrumb (the watcher keeps its own
        # registry/sidecar evidence trail; an operator-attributed auto-post
        # here would manufacture false attributions + unsentineled notes
        # that reset staleness clocks).
        "--stop-source", "watcher",
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
    _forward_marker_child_stderr(res, "spawn_session stop (watcher)")
    return True


def _respawn_stalled_session(issue: int, cap_gpu_hours: float, dry_run: bool) -> str:
    """Spawn a fresh `--auto` session for ``issue``.

    Mirrors :func:`_respawn` (used by the crash-recovery pass) but is
    decoupled from the autonomous-registry entry shape — the stalled-
    detector path knows the issue and the cap directly from the loaded
    state, so it doesn't pass a registry-entry dict. Returns the #843 M1b
    tri-state ``"spawned" | "suppressed" | "failed"`` (see :func:`_respawn`);
    on ``"spawned"``, spawn_session rewrites the registry (new id, missed=0)
    as a side effect.

    Note: we do NOT call :func:`_respawn` directly because the
    spawn-issue invocation here is the SAME (`--auto`
    `--auto-approve-gpu-hours`) but the surrounding context differs:
    this path has already called :func:`_stop_session` first, and the
    log prefix is `RESPAWNED-STALLED` rather than `RESPAWNED` so the
    operator can tell the two paths apart in the watcher logs.

    #1027: the auth-outage gate for this arm sits at the CALLER
    (:func:`_handle_stalled_respawn`) — a helper-internal gate would let the
    fence stop the old session and then decline the spawn, leaving the issue
    dead for the whole episode (plan MF-2).
    """
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto", "--auto-approve-gpu-hours", str(cap_gpu_hours),
    ]  # fmt: skip
    cmd.extend(_stalled_session_overrides(issue))
    if dry_run:
        print(f"  [dry-run] would respawn stalled: {' '.join(cmd)}")
        return "failed"  # dry-run: nothing spawned
    # #1027 §13.2: _StalledActionCtx carries no spawned_at — capture the
    # PREDECESSOR's spawned_at from the registry BEFORE the spawn rewrites it
    # (fail-soft -> None).
    prev_spawned_at = _registry_spawned_at(issue)
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(
            f"  RESPAWN-STALLED FAILED issue #{issue}: {res.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return "failed"
    _forward_marker_child_stderr(res, "spawn_session spawn-issue (stalled)")
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    if spawn_output_suppressed(res.stdout) is not None:
        print(
            f"  RESPAWN-STALLED issue #{issue}: suppressed — not respawned "
            f"(lease/collision): {first_line}"
        )
        return "suppressed"
    print(f"  RESPAWNED-STALLED issue #{issue} (alive-but-stalled): {first_line}")
    _auth_outage_record_spawn(issue, "stalled", prev_spawned_at)
    return "spawned"


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
        live_consecutive: int = 0,
        now: float | None = None,
        live_ids: set[str] | None = None,
        entry_spawned_at: float | None = None,
        stop_pending_sid: str | None = None,
        stop_pending_ts: float | None = None,
        stop_retried: bool = False,
        stop_failed_alerted: bool = False,
        wt_hold_count: int = 0,
        daemon_blocked_ticks: int = 0,
        daemon_blocked_pushed: bool = False,
        wedge_hits: int = 0,
        wedge_note: str | None = None,
        daemon_reachable: bool = True,
        downgrade_note: str | None = None,
        dead_silence_respawn_day: str | None = None,
        dead_silence_respawns_today: int = 0,
        wedge_respawn_day: str | None = None,
        wedge_respawns_today: int = 0,
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
        # #759 bug class b.1: consecutive live-stall episode count. The caller
        # sets this to the value the CURRENT tick must PERSIST (the incremented
        # count when the action was DOWNGRADED to alert on a live id; a RESET 0
        # on every other path). Every ``_save_stalled_state`` call site in the
        # handlers forwards it, so the persisted value matches the rule in
        # ``_process_stalled_session``.
        self.live_consecutive = live_consecutive
        # #845 hardening context. ``now`` is the pass clock (threaded so the
        # spawn-grace / hold predicates honor a test-supplied fake clock);
        # ``live_ids`` drives the fence's sid_alive verification;
        # ``entry_spawned_at`` is the registry entry's spawned_at for the
        # stalled-arm spawn-grace skip. The stop_pending_* / wt_hold_count /
        # daemon_blocked_* / wedge_hits fields carry the CURRENT-tick values
        # (advancement-cleared by the caller) that every persist site must
        # forward — same threading contract as ``live_consecutive``.
        # ``wedge_note`` is a human-readable evidence summary set only when
        # the (e) prompt-wedge trigger forced this respawn (quoted in the
        # respawn marker).
        self.now = now if now is not None else time.time()
        self.live_ids = live_ids
        # Normalized here (not at the call site) so the caller can pass the
        # raw registry value: a non-numeric / zero spawned_at reads as None
        # (no grace), mirroring the crash arm's missing-spawned_at handling.
        self.entry_spawned_at = (
            float(entry_spawned_at)
            if isinstance(entry_spawned_at, int | float) and entry_spawned_at
            else None
        )
        self.stop_pending_sid = stop_pending_sid
        self.stop_pending_ts = stop_pending_ts
        self.stop_retried = stop_retried
        self.stop_failed_alerted = stop_failed_alerted
        self.wt_hold_count = wt_hold_count
        self.daemon_blocked_ticks = daemon_blocked_ticks
        self.daemon_blocked_pushed = daemon_blocked_pushed
        self.wedge_hits = wedge_hits
        self.wedge_note = wedge_note
        # #1071 evidence-based alert reasons: ``daemon_reachable`` is the
        # pass-level flag (computed once per tick) and ``downgrade_note`` is
        # the #759 live-corroboration downgrade explanation (None on every
        # other path). Both are PER-TICK EVIDENCE for the alert handler's
        # reason ladder — recomputed each tick, deliberately NOT persisted by
        # :func:`_persist_stalled_ctx` (no state-file schema change). Before
        # these were threaded, the handler's exhaustive-pre-#759 else-branch
        # misattributed a corroboration debounce as "Happy daemon
        # unreachable" (both #813 incidents, 2026-07-03/04), prompting manual
        # respawns that raced an in-flight auto-recovery.
        self.daemon_reachable = daemon_reachable
        self.downgrade_note = downgrade_note
        # #1209 day-keyed dead-silence cap: the caller loads these from the
        # prior on-disk state NORMALIZED to the current UTC day (day-rolled /
        # malformed -> (day_key, 0)), deliberately WITHOUT the advancement
        # clear the #845 hardening fields get. Every persist site forwards
        # them (via _persist_stalled_ctx) so keep-path saves never wipe the
        # counter; the ONLY bump site is _handle_stalled_respawn's
        # stop-initiation branch.
        self.dead_silence_respawn_day = dead_silence_respawn_day
        self.dead_silence_respawns_today = dead_silence_respawns_today
        # #1241 twin fields — the four pre-#1209 triggers' shared day-keyed
        # counter, under the SAME contract as the dead_silence_* pair above
        # (advancement-clear-EXEMPT load; bumped only at stop-initiation).
        self.wedge_respawn_day = wedge_respawn_day
        self.wedge_respawns_today = wedge_respawns_today

    @property
    def happy_session_id_str(self) -> str | None:
        """Narrow ``happy_session_id`` (typed ``object`` because it comes from
        a JSON read) to ``str | None`` for the state-save call sites."""
        return self.happy_session_id if isinstance(self.happy_session_id, str) else None


def _persist_stalled_ctx(ctx: _StalledActionCtx, sid: str | None, missed: int, **overrides) -> None:
    """Persist stalled-detector state from ``ctx``'s current-tick values,
    with per-call keyword overrides. No-op on ``ctx.dry_run`` (every handler
    persist site shares that guard). Keeps the full field threading in ONE
    place so a handler cannot silently drop a #845 hardening field."""
    if ctx.dry_run:
        return
    kwargs: dict = dict(
        alerted=ctx.alerted,
        last_self_report_ts=ctx.last_self_report_ts,
        respawn_count=ctx.respawn_count,
        exhausted=ctx.exhausted,
        refresh_attempted=ctx.refresh_attempted,
        followups_child_alerted=ctx.followups_child_alerted,
        live_consecutive=ctx.live_consecutive,
        stop_pending_sid=ctx.stop_pending_sid,
        stop_pending_ts=ctx.stop_pending_ts,
        stop_retried=ctx.stop_retried,
        stop_failed_alerted=ctx.stop_failed_alerted,
        wt_hold_count=ctx.wt_hold_count,
        daemon_blocked_ticks=ctx.daemon_blocked_ticks,
        daemon_blocked_pushed=ctx.daemon_blocked_pushed,
        wedge_hits=ctx.wedge_hits,
        dead_silence_respawn_day=ctx.dead_silence_respawn_day,
        dead_silence_respawns_today=ctx.dead_silence_respawns_today,
        wedge_respawn_day=ctx.wedge_respawn_day,
        wedge_respawns_today=ctx.wedge_respawns_today,
        prev=ctx.prev_state,
    )
    kwargs.update(overrides)
    _save_stalled_state(ctx.issue, sid, missed=missed, **kwargs)


def _stalled_arm_deferral(ctx: _StalledActionCtx) -> bool:
    """Pre-fence deferrals for the stalled respawn arm (#845): the
    spawn-grace skip and the bounded worktree-activity hold. Returns True
    when the arm logged + persisted and the caller must return WITHOUT
    stopping or spawning this tick.

    Spawn-grace skip (a-ii): a registry entry (re)written within
    :func:`_respawn_spawn_grace_s` means a concurrent respawn (the crash
    arm runs BEFORE this pass, against a once-per-tick ``live_ids``
    snapshot, so it can legitimately respawn inside the fence's stop->verify
    gap) already owns the issue — the mirror of the crash arm's #759 grace,
    which the stalled arm previously lacked. A NEGATIVE entry age (clock
    skew / fake clock) is not within grace.

    Worktree hold (b): fresh file activity under the issue's worktree is
    direct evidence an implementer is mid-edit (#812: killed 57s after an
    edit); defer the stop/respawn, bounded at :data:`WT_HOLD_MAX_TICKS`
    consecutive holds (~1h) so a cross-writer can't defer recovery forever.
    ``missed`` is pinned at the threshold while held so the arm re-fires on
    the very next tick (stay armed, mirroring the crash arm's hold).

    #1071 post-stop gate: the #812 mid-edit protection guards the STOP;
    once ``stop_pending_sid`` is set, a stop has already been ISSUED for
    that sid (issued, not necessarily landed) — the fence (verify-dead ->
    spawn / retry-stop / stop-failed terminal) still guarantees no spawn
    next to a live sid, so holding only delays the verified-dead spawn and
    leaves the issue driverless (incident #813, 2026-07-03: 3 held ticks
    post-stop while a detached analysis process — which survives respawn
    and was re-attached by the successor — fed the activity signal).
    Mirrors the existing mid-fence gate on the K-corroboration. The
    spawn-grace skip stays UNCONDITIONAL (it guards concurrent respawns,
    still valid mid-fence)."""
    grace_s = _respawn_spawn_grace_s()
    if ctx.entry_spawned_at is not None and 0 <= ctx.now - ctx.entry_spawned_at < grace_s:
        print(
            f"  issue #{ctx.issue}: SPAWN-GRACE — registry entry spawned "
            f"{(ctx.now - ctx.entry_spawned_at) / 60:.1f}m ago (< {grace_s / 60:.0f}m); "
            f"skipping the stalled respawn arm this tick (a concurrent "
            f"respawn owns the issue)."
        )
        _persist_stalled_ctx(ctx, ctx.happy_session_id_str, 0)
        return True
    activity = _worktree_recent_activity(ctx.issue, ctx.now, _wt_activity_fresh_s())
    if ctx.stop_pending_sid is None and decide_worktree_hold(activity, ctx.wt_hold_count):
        held = ctx.wt_hold_count + 1
        print(
            f"  issue #{ctx.issue}: HOLD-RESPAWN — worktree activity < "
            f"{_wt_activity_fresh_s() / 60:.0f}m (hold {held}/{WT_HOLD_MAX_TICKS}); "
            f"an implementer may be mid-edit; deferring stop/respawn."
        )
        _persist_stalled_ctx(ctx, ctx.happy_session_id_str, ctx.threshold, wt_hold_count=held)
        return True
    return False


def _fence_stop_failed(ctx: _StalledActionCtx, sid: str) -> None:
    """Loud one-time terminal state of a failed fence episode: the session
    survived the stop AND the one allowed retry (daemon ACK != kill). NEVER
    spawn next to a live session (the #763 two-drivers class) — alert once
    (marker + phone push, both dedup'd via ``stop_failed_alerted``) and
    hold until the sid leaves the live set or the operator intervenes (the
    fence keeps re-evaluating each tick — when the sid finally dies, the
    verified-dead spawn branch recovers the issue)."""
    if not ctx.stop_failed_alerted:
        _post_progress_marker(
            ctx.issue,
            f"{_STALLED_STOP_FAILED_NOTE_SENTINEL} STALLED-SESSION STOP FAILED: "
            f"Happy session id={sid} was stopped + retried by the stalled "
            f"respawn fence but is STILL in the daemon's live set (daemon ACK "
            f"!= kill). NOT spawning a replacement next to a live session "
            f"(two drivers would race on the same issue, #763); stop it "
            f"manually (`spawn_session.py stop --session-id {sid}`) and "
            f"re-drive `/issue {ctx.issue}`.",
            ctx.dry_run,
            label="session-stop-failed",
        )
        _telegram_push(
            f"#{ctx.issue} stalled-session stop FAILED (sid {sid} survived "
            f"stop+retry); auto-respawn fenced off — manual stop needed",
            ctx.dry_run,
        )
    _persist_stalled_ctx(ctx, sid, ctx.threshold, stop_failed_alerted=True)


def _fence_spawn_stalled(ctx: _StalledActionCtx, sid: str) -> None:
    """Verified-dead spawn branch of the stop-verify fence (the pre-#845
    respawn body): the pending sid is confirmed absent from the daemon's
    live set, so a fresh ``--auto`` session cannot race the superseded one."""
    # #1247 terminal-status act guard (stalled-arm symmetry): the fence
    # stop→verify→spawn spans ticks, so ctx.task_status is ≥10 min old
    # here by construction. Same positive-ACTIVE confirmation as the
    # orphan pass.
    live_status = _task_status(ctx.issue)
    if live_status not in ACTIVE:
        print(
            f"  STALLED-ACT-GUARD issue #{ctx.issue}: fence spawn aborted — "
            f"live status re-read returned {live_status!r} (not ACTIVE; "
            f"snapshot was {ctx.task_status}); no respawn, no marker (#1247).",
            file=sys.stderr,
        )
        if live_status is not None and not ctx.dry_run:
            # Positively non-ACTIVE: clear the four fence stop_pending_*/
            # stop_* fields only (NOT the whole stalled episode state),
            # mirroring the suppressed branch's clear below. On None
            # (transient task.py read failure) keep the fence PENDING —
            # re-evaluated next tick.
            _clear_fence_state_on_disk(ctx.issue)
        return
    cap = _stalled_cap_gpu_hours(ctx.issue)
    spawn_result = _respawn_stalled_session(ctx.issue, cap, ctx.dry_run)
    if spawn_result == "suppressed":
        # #843 M1b: a concurrent dispatcher's fresh lease / a registration
        # collision suppressed the spawn — a session for this issue is live
        # and driving. Book NOTHING: no respawn marker, no respawn_count
        # bump, no missed/alerted rewrite; the next tick re-evaluates against
        # the new session's progress. #845 addition: DO clear the fence's
        # stop_pending_* (the lease collision proves a live driver owns the
        # issue, so the fence episode is over — a stale pending sid would
        # make the NEXT episode's first fence read misfire).
        # #1027 note: fleet-wide, "suppressed" now ALSO covers the
        # auth-outage gate — but NOT on this path: the stalled arm's
        # auth-outage gate sits at the CALLER (_handle_stalled_respawn),
        # which skips the whole stop+respawn unit, so a "suppressed" from
        # _respawn_stalled_session is still lease/collision-only and the
        # live-driver inference (clear the fence) remains correct.
        if not ctx.dry_run:
            _clear_fence_state_on_disk(ctx.issue)
        return
    spawn_ok = spawn_result == "spawned"
    new_respawn_count = ctx.respawn_count + 1
    if spawn_ok:
        wedge_suffix = f" Wedge evidence: {ctx.wedge_note}." if ctx.wedge_note else ""
        hold_suffix = (
            f" (respawn was held {ctx.wt_hold_count} tick(s) for worktree activity)"
            if ctx.wt_hold_count
            else ""
        )
        _post_progress_marker(
            ctx.issue,
            f"{_STALLED_RESPAWN_NOTE_SENTINEL} ALIVE-BUT-STALLED auto-"
            f"respawn: Happy session id={ctx.happy_session_id} was in the "
            f"live set but self-report has been frozen for {ctx.self_gap} "
            f"and the latest non-watcher progress marker is {ctx.marker_gap} "
            f"old (has_pod={ctx.has_pod}, status={ctx.task_status}). Stopped "
            f"the old session (verified dead on this tick's live set) and "
            f"spawned a fresh `--auto` session "
            f"(respawn {new_respawn_count}/{STALLED_MAX_RESPAWNS} this "
            f"episode). Confirmed for >= {ctx.threshold} checks."
            f"{wedge_suffix}{hold_suffix} {_source_stamp()}",
            ctx.dry_run,
            label="session-auto-respawn",
        )
        # spawn_session.py rewrote the registry's happy_session_id, but we
        # don't bother re-reading it here — the next tick's entry-read picks
        # up the new id, and `alerted` / respawn dedup is keyed on
        # self-report-ts advancement rather than session id. Clearing
        # alerted so a future episode can re-alert if the new session also
        # stalls (the respawn_count keeps growing toward the cap); clearing
        # the fence + the worktree-hold counter (episode over).
        _persist_stalled_ctx(
            ctx,
            None,
            0,
            alerted=False,
            respawn_count=new_respawn_count,
            stop_pending_sid=None,
            stop_pending_ts=None,
            stop_retried=False,
            stop_failed_alerted=False,
            wt_hold_count=0,
        )
        return
    # "failed": keep the pre-#845 no-booking behavior (respawn_count NOT
    # bumped; retried on a later tick) — but the fence's stop_pending_*
    # STAYS, so the retry re-verifies the sid is still dead before spawning.
    _persist_stalled_ctx(ctx, None, 0, alerted=False)


def _handle_stalled_respawn(ctx: _StalledActionCtx) -> None:
    """Recovery action: stop the alive-but-stalled session, VERIFY the stop
    landed on a later tick, then spawn a fresh ``--auto`` session (#845
    stop-verify fence — see :func:`decide_respawn_fence`; the pre-#845 arm
    stopped and spawned in the SAME tick, trusting the daemon's stop ACK as
    a kill, which left two drivers overlapped ~4h in #763). The arm is
    additionally deferred by the spawn-grace skip and the bounded
    worktree-activity hold (:func:`_stalled_arm_deferral`).

    A genuinely-dead wrapper reaching this arm (sid never in ``live_ids``)
    still waits ONE tick between stop and spawn: the fence predicate is
    evaluated once per tick, so ``stop_pending`` is recorded on the first
    tick and the verified-dead spawn happens on the next. That +10 min is
    deliberate — it is what closes #763.

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
    if _stalled_arm_deferral(ctx):
        return

    # #1027 (MF-2): gate the WHOLE stop+respawn unit at the caller, BEFORE
    # the fence's _stop_session — a helper-internal gate would stop the old
    # session and then decline the spawn, leaving the issue dead for the
    # episode. A unit already MID-FLIGHT (stop_pending_sid set) is NOT
    # re-gated: its session is already stopped, so completing the
    # verified-dead spawn is strictly safer than freezing the fence (the
    # gate's canary_pending window covers the common one-tick stop->spawn
    # gap; this branch covers a unit that began BEFORE the episode
    # triggered).
    if (
        ctx.stop_pending_sid is None
        and _auth_outage_spawn_gate(ctx.issue, "stalled", dry_run=ctx.dry_run) is not None
    ):
        print(
            f"  issue #{ctx.issue}: stalled stop+respawn SKIPPED — auth-outage "
            f"episode active (fleet respawn suppression)"
        )
        return

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
        _persist_stalled_ctx(ctx, None, 0)
        return
    fence = decide_respawn_fence(
        stop_pending_sid=ctx.stop_pending_sid,
        current_sid=sid,
        sid_alive=(ctx.live_ids is not None and sid in ctx.live_ids),
        stop_retried=ctx.stop_retried,
    )
    if fence == "clear-keep":
        print(
            f"  issue #{ctx.issue}: FENCE — pending stop sid "
            f"{ctx.stop_pending_sid} no longer matches the entry sid {sid} "
            f"(a concurrent respawn owns the issue); clearing fence state, "
            f"NOT stopping the fresh sid."
        )
        _persist_stalled_ctx(
            ctx,
            sid,
            0,
            stop_pending_sid=None,
            stop_pending_ts=None,
            stop_retried=False,
            stop_failed_alerted=False,
        )
        return
    # The stop / retry-stop / stop-failed branches persist ``missed`` PINNED
    # AT THE THRESHOLD (stay armed): the respawn ACTION must re-fire on the
    # very next tick so the fence's verify->spawn (or retry) step runs then —
    # persisting 0 on the not-yet-alerted path would make decide() re-
    # accumulate the full 2-miss debounce between stop and spawn, doubling
    # the fence latency.
    if fence == "stop":
        _stop_session(sid, ctx.dry_run)
        # #1209 / #1241 day-cap bump — ONCE per fence episode, at
        # STOP-INITIATION: this branch runs exactly on the stop_pending_sid
        # None -> sid transition (a retry-stop tick carries a pending sid
        # and takes the "retry-stop" branch — exactly-once by construction),
        # and it is the ONE tick where ctx.wedge_note is guaranteed present
        # for a wedge-initiated episode (the wedge fired THIS tick to
        # escalate; the fence's own spawn branch is unreachable for the
        # fresh-self-report shape — post-stop the sid leaves the daemon
        # /list, the wedge pid-gate goes inert, and decide() keeps on the
        # fresh boot self-report; the CRASH-RECOVERY arm completes the
        # respawn — so neither the spawn branch nor respawn_count can be
        # the counting site). The bracketed [failed-turn-silence] substring
        # of the wedge-note template is load-bearing: it routes the bump to
        # the #1209 budget; ANY OTHER wedge note bumps the #1241 shared
        # four-trigger budget. A decide()-sanctioned respawn (wedge never
        # ran) has wedge_note is None -> no bump (it is already
        # belt-bounded inside decide()).
        extra: dict = {}
        wedge_episode_suffix = ""
        if ctx.wedge_note and "[failed-turn-silence]" in ctx.wedge_note:
            bumped = ctx.dead_silence_respawns_today + 1
            extra = {
                "dead_silence_respawn_day": time.strftime("%Y-%m-%d", time.gmtime(ctx.now)),
                "dead_silence_respawns_today": bumped,
            }
            _post_progress_marker(
                ctx.issue,
                f"{_STALLED_DEAD_SILENCE_STOP_NOTE_SENTINEL} DEAD-WAKE STOP "
                f"(#1209): {ctx.wedge_note}. Stopped Happy session id={sid} "
                f"for fresh respawn; the crash-recovery arm completes the "
                f"spawn once the stopped wrapper is verifiably dead "
                f"(~20-30 min) — the fence's own spawn branch is not this "
                f"trigger's completion path. Dead-silence fence episode "
                f"{bumped}/{_tick_wedge_dead_respawns_per_day()} this UTC day.",
                ctx.dry_run,
                label="session-dead-silence-stop",
            )
        elif ctx.wedge_note:
            # #1241: the four pre-#1209 triggers' shared day-cap bump. No
            # dedicated marker (deliberate, minimal): the stale-shape
            # completion already posts the session-auto-respawn marker with
            # the wedge_suffix; the fresh-shape completion via the crash arm
            # is a pre-existing observability gap out of #1241's scope. The
            # extended fence print below carries the log-side evidence.
            bumped = ctx.wedge_respawns_today + 1
            extra = {
                "wedge_respawn_day": time.strftime("%Y-%m-%d", time.gmtime(ctx.now)),
                "wedge_respawns_today": bumped,
            }
            wedge_episode_suffix = (
                f"; wedge fence episode {bumped}/{_tick_wedge_respawns_per_day()} "
                f"this UTC day (#1241)"
            )
        print(
            f"  issue #{ctx.issue}: FENCE — stop issued for sid {sid}; will "
            f"verify it is dead + spawn on the NEXT tick (daemon ACK != kill)"
            f"{wedge_episode_suffix}."
        )
        _persist_stalled_ctx(
            ctx,
            sid,
            ctx.threshold,
            stop_pending_sid=sid,
            stop_pending_ts=ctx.now,
            stop_retried=False,
            stop_failed_alerted=False,
            **extra,
        )
        return
    if fence == "retry-stop":
        print(
            f"  issue #{ctx.issue}: FENCE — sid {sid} STILL in the live set "
            f"after the stop; retrying the stop ONCE."
        )
        _stop_session(sid, ctx.dry_run)
        _persist_stalled_ctx(ctx, sid, ctx.threshold, stop_retried=True)
        return
    if fence == "stop-failed":
        _fence_stop_failed(ctx, sid)
        return
    # fence == "spawn": pending sid verified dead — safe to spawn.
    _fence_spawn_stalled(ctx, sid)


def _handle_stalled_exhausted(ctx: _StalledActionCtx) -> None:
    """Recovery action: the crash-loop cap has been reached. Post a one-time
    loud marker, persist ``exhausted=True`` for dedup. Subsequent ticks
    stay quiet until real progress advances and clears the flag."""
    sid = ctx.happy_session_id_str
    if ctx.exhausted:
        _persist_stalled_ctx(ctx, sid, 0, alerted=True, exhausted=True)
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
    _persist_stalled_ctx(ctx, sid, 0, alerted=True, exhausted=True)


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
    # #1071 evidence-based reason ladder. Pre-#759 this ladder was exhaustive
    # (manual / non-ACTIVE / daemon-down were the ONLY alert producers); the
    # #759 live-corroboration downgrade added a fourth producer AFTER
    # decide(), and the stale else-branch then fabricated "Happy daemon
    # unreachable" for it even with the daemon up (both #813 incidents) —
    # prompting manual respawns that raced an in-flight auto-recovery. Each
    # branch now states the observed cause AND what the watcher does NEXT
    # (``next_step``), so the note never invites a manual stop+respawn while
    # an auto-recovery is mid-flight (the manual branch is the one place
    # that instruction is true). The else-branch self-identifies as a
    # watcher bug instead of inventing a daemon outage.
    if ctx.manual:
        reason = "manual user-driven session; alert-only by design"
        next_step = (
            f"open the session (phone / `spawn_session.py list`) and "
            f"re-drive `/issue {ctx.issue}` manually if confirmed dead"
        )
    elif not ctx.in_active:
        reason = f"task status not ACTIVE ({ctx.task_status})"
        next_step = (
            "no auto-respawn at a parked/terminal status; re-drive manually if this is wrong"
        )
    elif ctx.downgrade_note is not None:
        reason = ctx.downgrade_note
        next_step = (
            f"the watcher auto-escalates to a stop+respawn once the "
            f"live-stall debounce is exhausted (typically the NEXT tick); no "
            f"manual action needed unless a later "
            f"'{_STALLED_EXHAUSTED_NOTE_SENTINEL}' or "
            f"'{_STALLED_STOP_FAILED_NOTE_SENTINEL}' marker appears"
        )
    elif not ctx.daemon_reachable:
        reason = "Happy daemon unreachable; cannot stop+spawn THIS tick"
        next_step = (
            "episode state persists on disk; the stop+respawn fires "
            "automatically on the next daemon-reachable tick, and a phone "
            "push fires after 2 blocked ticks (#845 c)"
        )
    else:
        reason = "unexpected alert cause (watcher bug — please report)"
        next_step = "investigate _handle_stalled_alert's reason ladder"

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

    # #1137 one alert marker per staleness episode TOTAL, across BOTH alert
    # producers. decide_session_stalled's alerted=True dedup covers only its
    # own keep branch; the #759 downgrade lane rewrites respawn->alert AFTER
    # decide() and reached this handler every eligible tick — on #1092
    # (2026-07-07) the escalate/wt-hold-defer/downgrade 2-tick cycle posted a
    # fresh marker every 20 min on a healthy session (20:43Z/21:03Z/21:23Z).
    # An already-alerted episode keeps the stderr line, the downgrade lane's
    # stalled-live sidecar row, and the state persist — only the marker post
    # is suppressed. Cleared on self-report advancement like every other
    # per-episode flag, so a NEW episode re-alerts.
    if ctx.alerted:
        print(
            f"  issue #{ctx.issue}: repeat stalled-alert marker SUPPRESSED "
            f"(episode already alerted; cause this tick: {reason})",
            file=sys.stderr,
        )
        _persist_stalled_ctx(ctx, sid, 0, alerted=True, refresh_attempted=new_refresh_attempted)
        return

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
            f"bg-Bash chain died. NOT auto-respawned ({reason}); "
            f"{next_step}. Confirmed for >= {ctx.threshold} checks."
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
            f"auto-respawned ({reason}); {next_step}. Confirmed for >= "
            f"{ctx.threshold} checks."
        )
    _post_progress_marker(
        ctx.issue,
        note,
        ctx.dry_run,
        label="session-stalled-alert",
    )
    _persist_stalled_ctx(ctx, sid, 0, alerted=True, refresh_attempted=new_refresh_attempted)


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
    """Check the alive-but-stalled exemptions for the stalled-detector pass
    (the prose USER-PAUSE hold, the over-cap spend-approval park, the
    deliberate-blocked-park suppression (#1137), the round-complete re-park,
    and the followups_running-parent-waiting-on-open-child suppression);
    rewrite ``(action, new_missed, followups_child_alerted)`` accordingly.

    No-op unless ``action != "keep" or new_missed > 0`` (so the healthy-
    session hot path never pays the ``task.py list-children`` subprocess).
    When an exemption fires, the action is rewritten to ``"keep"``,
    ``new_missed`` is reset to 0 (the exemption deliberately does NOT
    accumulate misses — the task is correctly parked, not stalled), and
    a one-time alert marker is posted (the awaiting-child arm dedups via
    ``followups_child_alerted``; the spend-approval arm dedups self-containedly
    in the events log via :func:`_spend_approval_skip_already_noted`).
    Factored out of :func:`_process_stalled_session` to keep that function
    under the C901 cyclomatic-complexity cap (15).
    """
    if action == "keep" and new_missed == 0:
        return action, new_missed, followups_child_alerted
    # Prose USER-PAUSE hold (incident #816, 2026-07-02): the latest word on
    # the task is a non-watcher note beginning 'USER PAUSE' — a deliberate
    # user hold left at an ACTIVE status (the prose-only anti-pattern; the
    # durable affordance is set-status on_hold). Checked FIRST — an explicit
    # user directive is the most specific gate signal; both this and the
    # spend-approval arm are alert-only, so ordering only selects the more
    # actionable alert text. Dedup'd in the events log via
    # _user_pause_skip_already_noted, so no per-pass state flag is threaded.
    pause_reason = _user_pause_hold_reason(events)
    if pause_reason is not None:
        print(
            f"  issue #{issue}: ALIVE-BUT-STALLED exemption — {pause_reason}; "
            f"treating session as parked this tick (would have been "
            f"action={action})."
        )
        _maybe_post_user_pause_skip(issue, pause_reason, events, dry_run)
        return "keep", 0, followups_child_alerted
    # Over-cap spend-approval park (incident #653, 2026-06-18): the latest
    # non-watcher event is `epm:awaiting-spend-approval` (a 132 GPU-h plan over
    # the 100h auto-approve cap), and the status-hold variant (SKILL.md Step 9b)
    # keeps the task at the ACTIVE status `followups_running`, so decide() sees
    # an ACTIVE task and the missing-self-report drives respawn. A respawned
    # session only re-reads the same parked plan and re-posts the same
    # `epm:step-completed step=2c exit_kind=parked`. This is a user-only gate
    # (`task.py set-status <N> approved`, or re-plan) — checked FIRST because it
    # is the most specific gate signal and status-agnostic. Dedup'd in the
    # events log, so no per-pass state flag is threaded.
    spend_reason = _spend_approval_park_reason(events)
    if spend_reason is not None:
        print(
            f"  issue #{issue}: ALIVE-BUT-STALLED exemption — {spend_reason}; "
            f"treating session as parked this tick (would have been "
            f"action={action})."
        )
        if not _spend_approval_skip_already_noted(events):
            _post_progress_marker(
                issue,
                f"{_SPEND_APPROVAL_SKIP_NOTE_SENTINEL} {spend_reason}. "
                f"Respawn suppressed (does NOT consume the respawn budget); "
                f"the user must approve the over-cap plan "
                f"(`task.py set-status {issue} approved`) or re-plan "
                f"(`task.py set-status {issue} planning` + re-invoke "
                f"/adversarial-planner) to advance this task.",
                dry_run,
                label="spend-approval-skip",
            )
        return "keep", 0, followups_child_alerted
    # Deliberate blocked park (#1137): the halt contract posts epm:failure
    # then sets status blocked — a stalled SESSION on such a task is the
    # EXPECTED post-park shape (the session parked and went idle by design),
    # and the gate-push pass already phone-pushed the blocked transition.
    # Suppress the alert (print-only, no marker: the epm:failure marker
    # itself carries the user ask). A blocked task with NO failure trail
    # (hand-moved / unexplained) keeps the one-time alert. #1092 15:33Z:
    # alert fired 2h12m after a 1s-apart failure+blocked park.
    if status == "blocked":
        blocked_reason = _deliberate_blocked_park_reason(events)
        if blocked_reason is not None:
            print(
                f"  issue #{issue}: ALIVE-BUT-STALLED exemption — {blocked_reason}; "
                f"treating session as parked this tick (would have been "
                f"action={action})."
            )
            return "keep", 0, followups_child_alerted
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
        repark_reason = _followup_round_complete_reason(events, issue=issue)
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


def _apply_stalled_live_corroboration(
    *,
    issue: int,
    entry: dict,
    action: str,
    daemon_reachable: bool,
    live_ids: set[str] | None,
    live_consecutive: int,
    dry_run: bool,
) -> tuple[str, int, str | None]:
    """Bounded K-escalation for the stalled detector's live-session
    corroboration (#759, bug class b.1; Option A). Rewrites
    ``(action, live_consecutive)`` and additionally returns a
    ``downgrade_note`` (third element): a human-readable explanation of the
    downgrade on the respawn->alert branch, ``None`` on every other branch.
    The caller threads the note into :class:`_StalledActionCtx` so
    :func:`_handle_stalled_alert`'s reason ladder can attribute the alert to
    THIS debounce instead of falling through to a fabricated
    daemon-unreachable reason (#1071; both #813 incidents had
    ``daemon_reachable=True`` on every tick).

    Applied AFTER the provision-in-flight + followups exemptions, so it only
    sees a respawn THOSE did not already turn into keep. A respawn-eligible
    session whose self-report is stale may still be ALIVE: its Happy id can be
    sitting in the daemon's ``/list`` reply right now (a long
    ``/adversarial-planner`` stage holds the conversation for minutes, posting
    no marker, while a late-firing self-report tick ages past the window).

    Rule (``k = _stalled_live_escalation_k()``):

    - ``action == "respawn"`` AND ``daemon_reachable`` AND the id is in
      ``live_ids``: a LIVE id is being respawned. Increment the consecutive
      live-stall count.
      - ``live_consecutive < k`` -> DOWNGRADE to ``"alert"`` (no duplicate
        driver on a transient busy stretch); persist the INCREMENTED count
        (criterion 6).
      - ``live_consecutive >= k`` -> FALL THROUGH to the canonical respawn arm
        (#506 dead-bg-chain class — recovery the orphan sweep never fires while
        the wrapper lingers in ``live_ids``); RESET the count to 0 (the
        episode's escalation has fired; the fresh ``--auto`` session starts a
        new episode) (criterion 7).
    - ``action == "respawn"`` but the id is NOT live (or the daemon is down /
      ``live_ids is None``): genuinely-dead wrapper — today's behavior; RESET
      to 0 (criterion 11; the K counter is for LIVE ids only).
    - any other ``action`` (keep / clear, incl. the provision/followups
      exemptions that already rewrote respawn->keep): RESET to 0 — not a
      live-stall respawn episode, so the counter must not straddle a later
      unrelated stall (criterion 12).

    Gated on ``daemon_reachable`` + ``live_ids is not None`` before the
    ``_session_alive`` call: when the daemon is down ``live_ids`` is ``None``
    and ``respawn_eligible`` was already False (no respawn to consider), so this
    is a no-op. Factored out of :func:`_process_stalled_session` to keep that
    function under the C901 cap.
    """
    if action != "respawn":
        # keep / clear (incl. provision/followups exemptions) — not a live-stall
        # respawn episode; reset so the counter never straddles unrelated stalls.
        return action, 0, None
    if not daemon_reachable or live_ids is None or not _session_alive(entry, live_ids):
        # Genuinely-dead wrapper (or daemon down) — today's behavior. No
        # live-stall episode in progress; reset the counter.
        return action, 0, None
    # A LIVE id is being respawned.
    k = _stalled_live_escalation_k()
    live_consecutive += 1
    if live_consecutive < k:
        print(
            f"  issue #{issue}: LIVE-SESSION CORROBORATION — Happy id is in "
            f"live_ids; this is consecutive live-stall episode "
            f"{live_consecutive}/{k}. Downgrading respawn->alert to avoid a "
            f"duplicate driver (transient busy stretch on a live session)."
        )
        _append_stalled_live_event(issue, "stalled-live-downgrade", live_consecutive, k, dry_run)
        note = (
            f"live-session corroboration debounce: consecutive live-stall "
            f"episode {live_consecutive}/{k}; the session id is still in the "
            f"daemon live set"
        )
        return "alert", live_consecutive, note
    # Kth consecutive live stall — escalate to the canonical respawn arm and
    # reset the counter (a fresh --auto session begins a new episode).
    # #1137 criterion-7 x wt-hold interplay (BY DESIGN, do not "fix"): when
    # the escalated respawn is subsequently DEFERRED by the #845 (b) worktree
    # hold / spawn grace, the counter reset below has already been persisted,
    # so the NEXT tick is a fresh "1/K" downgrade — deliberate: preserving
    # the count across a deferral would make the first post-hold tick respawn
    # immediately, a respawn-timing change. The repeat-ALERT marker noise the
    # resulting escalate/defer/downgrade cycle produced (#1092, 2026-07-07)
    # is fixed at the marker-post site (_handle_stalled_alert's episode-total
    # dedup), never here.
    print(
        f"  issue #{issue}: LIVE-SESSION ESCALATION — Happy id is in live_ids "
        f"but it has stalled across {live_consecutive} consecutive episodes "
        f"(>= K={k}); escalating to the canonical respawn (#506 dead-bg-chain "
        f"class). Resetting live_consecutive."
    )
    _append_stalled_live_event(issue, "stalled-live-escalation", live_consecutive, k, dry_run)
    return "respawn", 0, None


def _apply_prompt_wedge_override(
    *,
    issue: int,
    entry: dict,
    action: str,
    self_report_age: float | None,
    respawn_eligible: bool,
    pids_by_sid: dict[str, int] | None,
    live_consecutive: int,
    wedge_hits: int,
    now: float | None = None,
    respawn_count: int = 0,
    dead_silence_respawns_today: int = 0,
    wedge_respawns_today: int = 0,
) -> tuple[str, int, int, str | None]:
    """Prompt-wedge fast lane (#845 e; #1104 api-error rows; #1127 turn-level
    failed wakes; #1209 dead-wake silence): a transcript-tail probe that
    escalates a debounced
    ``keep``/``alert`` straight to the respawn arm on DIRECT evidence the
    session is swallowing prompts or dying on its wakes (incident #779: 5
    tick prompts enqueued+dequeued with no turn for ~90 min while the slow
    debounce ground through its misses; the eventual respawn then killed an
    in-flight implementer — the (b) hold now covers that half).

    Rewrites ``(action, live_consecutive, wedge_hits, wedge_note)``.
    Applied AFTER :func:`_apply_stalled_live_corroboration` — direct
    evidence beats the K-downgrade proxy — and bypasses the 2-miss guard
    and the 2h marker window, but the forced respawn is STILL subject to
    ``respawn_eligible`` (checked here: ACTIVE + daemon + not manual), to
    the park exemptions (re-probed against the escalated action by the
    caller, :func:`_apply_wedge_override_with_exemption_probe` — #845 r2)
    and, inside :func:`_handle_stalled_respawn`, to the spawn-grace skip,
    the (b) worktree hold and the (a) stop-verify fence. A wedge-forced
    respawn RESETS ``live_consecutive`` (consistent with the
    escalation-fired => reset semantics of the K corroboration).

    Two-path gate (#1127 — replacing the #845/#1104 lazy "probe only once
    the self-report is >= 1h stale" gate, which live wedges kept defeating):
    the TURN-level triggers (``failed-turn-run`` / ``failed-turn-rate``) are
    probed EVERY tick — a partially-executing wake can keep REFRESHING the
    self-report (a wake that escalates into the full ``/issue`` skill
    re-writes it at Step 0 before dying), so a dying-but-heartbeating
    session never goes stale and the old gate never opened (incidents #1098
    5bdae5b8 / #1090 5e464f3d, both 40 min-3.4 h past the #1104 merge) —
    while the ROW-level triggers (``dequeue-run`` #779 / ``api-error-run``
    #1074) stay STALENESS-GATED: their failure modes freeze the self-report
    by construction (no turn executes, nothing reaches the title helper), so
    the stale gate opens for them, AND keeping them off the fresh path
    preserves the single-refusal guard (3 same-turn retry api-error rows
    must not respawn a healthy session). ``self_report_age is None`` routes
    to the FRESH path (turn-level triggers only) — ``respawn_eligible``
    already excludes manual sessions, so an autonomous session with an
    unreadable self-report gets turn-level coverage instead of a blind spot.
    With BOTH turn-level knobs at 0 (``EPM_TICK_WEDGE_MIN_FAILED_TURNS=0`` +
    ``EPM_TICK_WEDGE_MIN_FAILED_TOTAL=0``) the fresh path probes NOTHING —
    the exact pre-#1127 lazy gate (the full-rollback path). Everything
    unresolvable (no pid map, sid not live, transcript miss) still fails
    toward NO-WEDGE. #1104: the api-error-run subclass — >= ``min_api_errors``
    consecutive API-ERROR turns (``isApiErrorMessage: true`` assistant rows:
    usage-policy refusals, 429/529) with no successful turn, the #1074 shape
    the original lane was structurally blind to.

    #1209 ``failed-turn-silence`` (the die-on-turn-1 gap): TURN-level, so it
    rides BOTH self-report paths — but deliberately behind the SAME
    two-turn-knob fresh-path gate as the #1127 lanes (both turn knobs at 0
    keeps the fresh path's zero-probe hot path byte-identical, the pinned
    pre-#1127 rollback; in that corner config the dead-silence trigger still
    fires via the STALE path once the boot self-report ages past the
    staleness window). Armed only while ``respawn_count <
    STALLED_MAX_RESPAWNS`` (the episode belt) AND
    ``dead_silence_respawns_today`` is under the day cap
    (:func:`_tick_wedge_dead_respawns_per_day`) — when either cap binds,
    ``dead_silence_s=0`` keeps this one trigger off while the other four
    keep their own arming below. ``now=None`` (a direct unit/debug call
    shape) leaves the trigger inert inside the predicate.

    #1241 cap parity for the FOUR pre-#1209 triggers: ``dequeue-run`` /
    ``api-error-run`` / ``failed-turn-run`` / ``failed-turn-rate`` carry
    the SAME two-part bound — the episode belt (``respawn_count <
    STALLED_MAX_RESPAWNS``, the cap decide() enforces on the slow path)
    AND their own shared per-issue per-UTC-day counter
    (``wedge_respawns_today`` vs :func:`_tick_wedge_respawns_per_day`),
    persisted advancement-clear-EXEMPT and bumped ONCE per wedge-initiated
    fence episode at stop-initiation (the crash-recovery arm, which
    consults no cap, can complete a fresh-self-report wedge respawn — so
    neither the fence spawn branch nor ``respawn_count`` can be the
    counting site). A disarmed family reads its predicate kill-switch
    values (0), so :func:`decide_prompt_wedge_reason` stays byte-identical;
    when EVERY family is cap-disarmed the wedge goes quiet BEFORE the
    256 KB transcript read (no marker, no push — the slow stalled lane
    stays the backstop). The two day budgets are INDEPENDENT."""
    if action not in ("keep", "alert") or not respawn_eligible or pids_by_sid is None:
        return action, live_consecutive, wedge_hits, None
    stale = self_report_age is not None and self_report_age >= STALLED_WINDOW_S
    min_turns = _tick_wedge_min_failed_turns()
    min_total = _tick_wedge_min_failed_total()
    if not stale and min_turns <= 0 and min_total <= 0:
        # Fresh self-report + both turn-level lanes disabled == the
        # pre-#1127 lazy gate: zero transcript probes on the hot path
        # (#1209 deliberately does not widen this gate — see docstring).
        return action, live_consecutive, wedge_hits, None
    sid = entry.get("happy_session_id")
    pid = pids_by_sid.get(sid) if isinstance(sid, str) else None
    if not isinstance(pid, int):
        return action, live_consecutive, wedge_hits, None
    # #1241 arming — the caps decide() enforces on the slow path bind the
    # wedge fast lane too. The episode belt gates ALL FIVE triggers; each
    # trigger family additionally carries its own day-keyed,
    # advancement-clear-EXEMPT cap (#1209 for failed-turn-silence; #1241
    # for the four pre-#1209 triggers — one shared counter, bumped at
    # stop-initiation because the crash-recovery arm, which consults no
    # cap, can complete the respawn). A disarmed trigger reads its
    # predicate kill-switch value (0), so decide_prompt_wedge_reason stays
    # byte-identical; when EVERYTHING is disarmed the wedge goes quiet (no
    # marker, no push) and the slow stalled lane stays the backstop.
    belt_ok = respawn_count < STALLED_MAX_RESPAWNS
    four_armed = belt_ok and wedge_respawns_today < _tick_wedge_respawns_per_day()
    dead_silence_armed = (
        belt_ok and dead_silence_respawns_today < _tick_wedge_dead_respawns_per_day()
    )
    if not four_armed and not dead_silence_armed:
        # Nothing can fire — skip the 256 KB transcript read entirely.
        return action, live_consecutive, wedge_hits, None
    # 256 KB tail (vs the 64 KB default): the 64 KB window held EXACTLY 3
    # api-error rows on the #1074 incident transcript — zero margin (#1104;
    # plan §10 allowed deviation; a too-thin window fails toward NO-FIRE).
    # #1127 re-verified the width: it spans ~1.3-2.5h and >= 4 failed wakes
    # on all three incident transcripts.
    rows = _transcript_tail_rows(pid, max_bytes=262144)
    if rows is None:
        return action, live_consecutive, wedge_hits, None
    reason = decide_prompt_wedge_reason(
        rows,
        # #779 swallow trigger: STALE only + #1241 four-family arming.
        _tick_wedge_min_dequeued() if (stale and four_armed) else 0,
        # Row-level api-error trigger: STALE only + #1241 four-family arming.
        min_api_errors=_tick_wedge_min_api_errors() if (stale and four_armed) else 0,
        min_failed_turns=min_turns if four_armed else 0,
        min_failed_total=min_total if four_armed else 0,
        rate_window_s=_tick_wedge_rate_window_s(),
        dead_silence_s=_tick_wedge_dead_silence_s() if dead_silence_armed else 0.0,
        now=now,
    )
    if reason is None:
        return action, live_consecutive, wedge_hits, None
    note = (
        f"prompt-wedge trigger [{reason}] in the transcript tail "
        f"(self-report {'stale' if stale else 'fresh'}; row-level triggers "
        f"{'armed' if stale else 'staleness-gated off'}; turn-level thresholds "
        f"min_failed_turns={min_turns}, min_failed_total={min_total})"
    )
    print(
        f"  issue #{issue}: PROMPT-WEDGE — {note}; escalating straight to the "
        f"respawn arm (bypasses the miss debounce, the K-downgrade and the "
        f"marker window; still subject to the park exemptions, the "
        f"spawn-grace skip, the worktree hold and the stop-verify fence)."
    )
    return "respawn", 0, wedge_hits + 1, note


def _apply_daemon_blocked_escalation(
    *,
    issue: int,
    in_active: bool,
    manual: bool,
    alerted: bool,
    stale: bool,
    daemon_reachable: bool,
    blocked_ticks: int,
    already_pushed: bool,
    dry_run: bool,
) -> tuple[int, bool]:
    """I/O wrapper around :func:`decide_daemon_blocked_escalation` (#845 c):
    fires the one-time Telegram push when a respawn-worthy stall has been
    deferred by an unreachable Happy daemon for >= 2 consecutive ticks
    (~20 min at the 10-min cron; incident #811 idled a GPU for hours on a
    silently-deferred respawn). Returns the ``(new_blocked_ticks,
    new_pushed)`` pair the caller persists. Fail-soft: the push helper never
    raises."""
    new_ticks, fire = decide_daemon_blocked_escalation(
        in_active=in_active,
        manual=manual,
        alerted=alerted,
        stale=stale,
        daemon_reachable=daemon_reachable,
        blocked_ticks=blocked_ticks,
        already_pushed=already_pushed,
    )
    if fire:
        print(
            f"  issue #{issue}: DAEMON-BLOCKED escalation — a respawn-worthy "
            f"stall has been deferred by an unreachable Happy daemon for "
            f"{new_ticks} consecutive ticks (~{new_ticks * 10} min); paging."
        )
        _telegram_push(
            f"#{issue} stalled; auto-respawn blocked: Happy daemon unreachable "
            f"{new_ticks} ticks (~{new_ticks * 10} min). GPU may be idling; "
            f"check the daemon.",
            dry_run,
        )
        return new_ticks, True
    if new_ticks == 0:
        return 0, False
    return new_ticks, already_pushed


def _apply_stalled_park_exemptions(
    *,
    issue: int,
    status: str | None,
    has_pod: bool,
    events: list[dict],
    action: str,
    new_missed: int,
    followups_child_alerted: bool,
    now: float,
    dry_run: bool,
) -> tuple[str, int, bool, bool]:
    """The stalled detector's ALIVE-BUT-STALLED exemptions, in order —
    rewrites ``(action, new_missed, followups_child_alerted)`` and returns
    a 4th element ``exempted`` (True iff any exemption rewrote the action;
    the #845 (e) prompt-wedge fast lane must never override an exemption —
    a legitimately-parked / provisioning session is not wedged). On the
    fresh-marker keep(0) path BOTH probes below are lazily skipped, so
    ``exempted=False`` means unprobed, not clear — a wedge escalation
    re-invokes this function once against the escalated action
    (:func:`_apply_wedge_override_with_exemption_probe`, #845 r2). Factored
    out of :func:`_process_stalled_session` to keep it under the C901 cap.

    1. In-flight-provision (refs #573) / long-phase-heartbeat (#761),
       probed LAZILY (only when decide() wants to escalate or accumulate a
       miss) so the healthy-session hot path never pays the probes. Two
       independent reasons share one gate + one log + the no-marker
       rewrite; the first that fires wins (both rewrite to ("keep", 0)).
       #534's auto-respawn killed an in-flight provision 3x (~8h lost);
       #761's 1h21m off-pod analysis drew a wasted respawn.
    2. followups_running parent-waiting-on-open-child (incident #533): a
       parent parked at step 10 awaiting a user-gated child cannot be
       unblocked by respawning the parent — see
       :func:`_apply_stalled_followups_exemption` (which also carries the
       spend-approval park, the deliberate-blocked-park suppression
       (#1137), and the round-complete re-park).
    """
    exempted = False
    if action != "keep" or new_missed > 0:
        exempt_reason = _provision_in_flight_reason(issue, now) or _long_phase_heartbeat_reason(
            events, now
        )
        if exempt_reason is not None:
            print(
                f"  issue #{issue}: ALIVE-BUT-STALLED exemption — {exempt_reason}; "
                f"treating session as live this tick (would have been "
                f"action={action})."
            )
            action, new_missed = ("keep", 0)
            exempted = True

    pre_followups = (action, new_missed)
    action, new_missed, followups_child_alerted = _apply_stalled_followups_exemption(
        issue=issue,
        status=status,
        has_pod=has_pod,
        events=events,
        action=action,
        new_missed=new_missed,
        followups_child_alerted=followups_child_alerted,
        dry_run=dry_run,
    )
    if (action, new_missed) != pre_followups:
        exempted = True
    return action, new_missed, followups_child_alerted, exempted


def _apply_wedge_override_with_exemption_probe(
    *,
    issue: int,
    entry: dict,
    status: str | None,
    has_pod: bool,
    events: list[dict],
    action: str,
    new_missed: int,
    followups_child_alerted: bool,
    self_report_age: float | None,
    respawn_eligible: bool,
    pids_by_sid: dict[str, int] | None,
    live_consecutive: int,
    wedge_hits: int,
    now: float,
    dry_run: bool,
    respawn_count: int = 0,
    dead_silence_respawns_today: int = 0,
    wedge_respawns_today: int = 0,
) -> tuple[str, int, bool, int, int, str | None]:
    """Prompt-wedge fast lane + the park-exemption re-probe on escalation
    (#845 r2, concern ``wedge-bypasses-unprobed-park-exemptions``).

    Both lazy exemption probes gate on ``action != "keep" or new_missed > 0``,
    so on the fresh-marker keep(0) hot path (``decide_session_stalled``
    returned ``("keep", 0)``) they never RAN and the caller's ``exempted`` is
    vacuously False — unprobed, not probed-and-clear. A wedge that then flips
    keep->respawn would bypass the provision-in-flight / long-phase /
    spend-approval / round-complete / awaiting-child checks entirely. So:
    when (and only when) the wedge fires, re-run
    :func:`_apply_stalled_park_exemptions` ONCE against the escalated action;
    if any exemption fires, the wedge does NOT force the respawn — the
    exemption's ``("keep", 0)`` rewrite stands and the wedge's counter side
    effects are undone (no wedge hit recorded, the K counter keeps its
    pre-wedge value, no ``wedge_note`` reaches the respawn arm). The wedge
    thus bypasses the miss debounce, the K-downgrade and the marker window,
    but NEVER the park exemptions, the worktree hold or the stop-verify
    fence — exactly the documented invariant
    (``.claude/rules/background-automation.md``). Factored out of
    :func:`_process_stalled_session` to keep it under the C901 cap (15).
    """
    pre_wedge = (live_consecutive, wedge_hits)
    action, live_consecutive, wedge_hits, wedge_note = _apply_prompt_wedge_override(
        issue=issue,
        entry=entry,
        action=action,
        self_report_age=self_report_age,
        respawn_eligible=respawn_eligible,
        pids_by_sid=pids_by_sid,
        live_consecutive=live_consecutive,
        wedge_hits=wedge_hits,
        now=now,
        respawn_count=respawn_count,
        dead_silence_respawns_today=dead_silence_respawns_today,
        wedge_respawns_today=wedge_respawns_today,
    )
    if wedge_note is None:
        return action, new_missed, followups_child_alerted, live_consecutive, wedge_hits, None
    action, new_missed, followups_child_alerted, wedge_exempted = _apply_stalled_park_exemptions(
        issue=issue,
        status=status,
        has_pod=has_pod,
        events=events,
        action=action,
        new_missed=new_missed,
        followups_child_alerted=followups_child_alerted,
        now=now,
        dry_run=dry_run,
    )
    if wedge_exempted:
        # A park exemption vetoed the wedge: the session is legitimately
        # parked / provisioning, not wedged.
        live_consecutive, wedge_hits = pre_wedge
        wedge_note = None
    return action, new_missed, followups_child_alerted, live_consecutive, wedge_hits, wedge_note


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
    live_ids: set[str] | None = None,
    pids_by_sid: dict[str, int] | None = None,
) -> None:
    """Reconcile one registry entry against the alive-but-stalled signals.

    ``pids_by_sid`` is the daemon's live ``{sid: wrapper pid}`` map
    (:func:`_live_pids_by_sid_or_none`), threaded from :func:`main` for the
    #845 (e) prompt-wedge probe. When omitted (``None`` — a direct
    unit/debug call, or the daemon is down), the wedge probe is inert.

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

    ``live_ids`` is the daemon's live-session set, computed once in
    :func:`main` and threaded in exactly as :func:`orphan_sweep_pass`
    receives it (``None`` when the daemon is unreachable). It drives the
    #759 bug-class-b.1 bounded K-escalation: a respawn that survives the
    other exemptions is DOWNGRADED to alert for the first K-1 consecutive
    episodes when the entry's Happy id is still in ``live_ids`` (a transient
    busy stretch on a live session), then ESCALATES to the canonical respawn
    on the Kth (the #506 dead-bg-chain class). When omitted (``None`` — a
    direct unit/debug call, or the daemon is down), the corroboration is a
    no-op and the pass behaves exactly as before. See
    :func:`_apply_stalled_live_corroboration`.
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

    # Signal 2: latest non-watcher MARKER age (ANY kind, not just
    # _PROGRESS_KINDS). None -> stale (no markers at all is itself a signal).
    # We count every non-watcher-sentinel'd marker as a sign of life — the
    # pre-run lifecycle (`epm:experiment-implementation`, `epm:code-review`,
    # `epm:review-reconcile`, `epm:plan`, ...) is exactly the work an
    # actively-implementing session does before it ever launches a pod, and
    # _PROGRESS_KINDS (run/upload/interpret-oriented) excludes all of it.
    # Gating signal 2 on that narrow allowlist falsely read those sessions as
    # "zero progress for the whole pre-pod phase" and respawned them mid-
    # implementation (#661: 7 real lifecycle markers between 00:39 and 01:43
    # were ignored, leaving the detector measuring staleness from a 64-min-old
    # epm:status-changed). The watcher's own alert/automation posts stay
    # excluded by the note-substring filter (_WATCHER_NOTE_SENTINELS), which is
    # what actually prevents the alert from resetting its own clock — the kind
    # allowlist was redundant for that and its only net effect was this false
    # negative. Matches the session-reconcile idle pass (`_latest_nonwatcher_
    # event_ts`), which already counts markers of ANY kind. A genuinely dead
    # session posts NO markers of any kind, so its newest non-watcher marker
    # still ages out and the detector still fires. We also keep the raw events
    # list around — the followups-awaiting-child exemption below scans it for
    # the latest epm:step-completed without paying a second read.
    events = _task_events(issue)
    latest_marker_ts = _latest_nonwatcher_event_ts(events)
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
    # #759 bug class b.1: consecutive live-stall episode count. Absent in an
    # older on-disk file -> 0 (backward compatible), same guard shape as
    # prev_missed / prev_respawn_count.
    prev_live_consecutive = prev_state.get("live_consecutive", 0)
    if not isinstance(prev_live_consecutive, int):
        prev_live_consecutive = 0
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
        # The session resumed self-reporting -> any prior live-stall episode is
        # over -> the K-escalation debounce starts fresh (#759 b.1, criterion
        # 12's earlier-firing sibling reset).
        live_consecutive = 0
    else:
        alerted = prev_alerted
        respawn_count = prev_respawn_count
        exhausted = prev_exhausted
        refresh_attempted = prev_refresh_attempted
        followups_child_alerted = prev_followups_child_alerted
        live_consecutive = prev_live_consecutive

    # #845 hardening fields (fence / hold / daemon-blocked / wedge state) —
    # loaded with the SAME advancement-clear rule as the flags above (the
    # self-report advanced => the episode is over => every per-episode
    # counter starts fresh).
    hard = _stalled_hardening_fields(prev_state, self_report_advanced)

    # #1209 / #1241 day-keyed wedge-cap fields — deliberately OUTSIDE the
    # advancement-clear above (each die-on-turn-1 generation writes one boot
    # self-report, so an advancement-cleared counter could never bound the
    # cross-generation die-on-boot loop the caps exist for). The day key
    # derives from the SAME injected `now` the pass runs on (deterministic
    # under a test-supplied fake clock); a day-rolled / malformed / negative
    # on-disk value reads as 0 (armed — a corruption costs at most one extra
    # bounded respawn). See :func:`_day_scoped_count`. The two counters are
    # INDEPENDENT budgets (#1209: failed-turn-silence; #1241: the four
    # pre-#1209 triggers, one shared counter).
    dead_silence_day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    dead_silence_respawns_today = _day_scoped_count(
        prev_state, "dead_silence_respawn_day", "dead_silence_respawns_today", dead_silence_day_key
    )
    wedge_respawns_today = _day_scoped_count(
        prev_state, "wedge_respawn_day", "wedge_respawns_today", dead_silence_day_key
    )

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

    marker_window_s = _stalled_marker_window_s()
    action, new_missed = decide_session_stalled(
        self_report_age_s=self_report_age,
        marker_progress_age_s=marker_age,
        has_pod=has_pod,
        missed=prev_missed,
        alerted=alerted,
        respawn_eligible=respawn_eligible,
        respawn_count=respawn_count,
        threshold=threshold,
        marker_window_s=marker_window_s,
    )

    # ALIVE-BUT-STALLED exemptions, probed LAZILY (only when decide() wants to
    # escalate or accumulate a miss) so the healthy-session hot path never pays
    # the probes. Two independent reasons share one gate + one log + the
    # no-marker rewrite; the first that fires wins (both rewrite to the same
    # ("keep", 0)):
    #   1. In-flight-provision (refs #573): a provision waiting for capacity
    #      blocks the session's bg-Bash chain, freezing BOTH staleness signals
    #      while being exactly the work the session should be doing — #534's
    #      auto-respawn killed an in-flight provision 3x (~8h lost). /proc scan.
    #   2. Long-phase-heartbeat (incident #761): a legitimately-slow phase
    #      (off-pod analyzer verifier rounds, in-flight Anthropic Batch polling)
    #      emits few markers, so both staleness signals cross the 60-min window
    #      between its heartbeats and the detector false-respawns (#761's 1h21m
    #      off-pod analysis drew a wasted respawn). An emitter opts into a wider
    #      leash by stamping _LONG_PHASE_HEARTBEAT_PREFIX into its epm:progress
    #      note; scans the already-loaded `events` (no extra read).
    action, new_missed, followups_child_alerted, exempted = _apply_stalled_park_exemptions(
        issue=issue,
        status=task_status,
        has_pod=has_pod,
        events=events,
        action=action,
        new_missed=new_missed,
        followups_child_alerted=followups_child_alerted,
        now=now,
        dry_run=dry_run,
    )

    # Live-session corroboration with bounded K-escalation (#759, bug class
    # b.1). Applied LAST, on a respawn that survived the provision + followups
    # exemptions above: downgrades to alert (no duplicate driver) for the first
    # K-1 consecutive episodes on a LIVE id, escalates to the canonical respawn
    # on the Kth (the #506 dead-bg-chain class), and resets the counter on
    # every other path. Factored into a helper to keep this function under the
    # C901 cap; the returned live_consecutive is what the ctx below persists.
    # The third element (downgrade_note) is per-tick evidence for the alert
    # handler's reason ladder — threaded into the ctx below, never persisted.
    downgrade_note: str | None = None
    if hard["stop_pending_sid"] is None:
        action, live_consecutive, downgrade_note = _apply_stalled_live_corroboration(
            issue=issue,
            entry=entry,
            action=action,
            daemon_reachable=daemon_reachable,
            live_ids=live_ids,
            live_consecutive=live_consecutive,
            dry_run=dry_run,
        )
    # else (#845 a-ii): a stop-verify fence episode is already in flight —
    # the K corroboration's debounce already served (its escalation, or the
    # dead-sid path, is what STARTED the episode); re-downgrading the verify
    # ticks on the still-live sid would stall the fence by K-1 ticks per
    # step (a failed stop's loud alert would land ticks late). The fence
    # owns the episode until it clears (spawn / sid-change / advancement).

    # Prompt-wedge fast lane (#845 e) — applied AFTER the K corroboration
    # (direct transcript evidence beats the live-id proxy) and NEVER over an
    # exemption rewrite (a legitimately-parked / provisioning session is not
    # wedged). On the fresh-marker keep(0) hot path the lazy exemptions above
    # were never PROBED (`exempted` is vacuously False), so a wedge-forced
    # respawn re-probes them ONCE against the escalated action (#845 r2,
    # concern wedge-bypasses-unprobed-park-exemptions). See
    # _apply_wedge_override_with_exemption_probe for the gates + the veto.
    wedge_note: str | None = None
    if not exempted:
        (
            action,
            new_missed,
            followups_child_alerted,
            live_consecutive,
            hard["wedge_hits"],
            wedge_note,
        ) = _apply_wedge_override_with_exemption_probe(
            issue=issue,
            entry=entry,
            status=task_status,
            has_pod=has_pod,
            events=events,
            action=action,
            new_missed=new_missed,
            followups_child_alerted=followups_child_alerted,
            self_report_age=self_report_age,
            respawn_eligible=respawn_eligible,
            pids_by_sid=pids_by_sid,
            live_consecutive=live_consecutive,
            wedge_hits=hard["wedge_hits"],
            now=now,
            dry_run=dry_run,
            respawn_count=respawn_count,
            dead_silence_respawns_today=dead_silence_respawns_today,
            wedge_respawns_today=wedge_respawns_today,
        )

    # Daemon-blocked escalation (#845 c): count consecutive ticks a
    # respawn-worthy stall stays deferred by an unreachable daemon; page once
    # at 2 ticks (~20 min). `alerted or action == "alert"` counts the very
    # tick the alert fires (the handler persists alerted=True after us).
    stale_now = self_report_age is not None and self_report_age >= STALLED_WINDOW_S
    hard["daemon_blocked_ticks"], hard["daemon_blocked_pushed"] = _apply_daemon_blocked_escalation(
        issue=issue,
        in_active=in_active,
        manual=manual,
        alerted=alerted or action == "alert",
        stale=stale_now and (marker_age is None or marker_age >= marker_window_s),
        daemon_reachable=daemon_reachable,
        blocked_ticks=hard["daemon_blocked_ticks"],
        already_pushed=hard["daemon_blocked_pushed"],
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
        f"followups_child_alerted={followups_child_alerted} "
        f"live_consecutive={live_consecutive} action={action}"
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
        live_consecutive=live_consecutive,
        now=now,
        live_ids=live_ids,
        entry_spawned_at=entry.get("spawned_at"),
        stop_pending_sid=hard["stop_pending_sid"],
        stop_pending_ts=hard["stop_pending_ts"],
        stop_retried=hard["stop_retried"],
        stop_failed_alerted=hard["stop_failed_alerted"],
        wt_hold_count=hard["wt_hold_count"],
        daemon_blocked_ticks=hard["daemon_blocked_ticks"],
        daemon_blocked_pushed=hard["daemon_blocked_pushed"],
        wedge_hits=hard["wedge_hits"],
        wedge_note=wedge_note,
        daemon_reachable=daemon_reachable,
        downgrade_note=downgrade_note,
        dead_silence_respawn_day=dead_silence_day_key,
        dead_silence_respawns_today=dead_silence_respawns_today,
        wedge_respawn_day=dead_silence_day_key,
        wedge_respawns_today=wedge_respawns_today,
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
    # followups_child_alerted flags + the #845 hardening fields (all cleared
    # above if self-report advanced) + the latest observed self-report ts so
    # the next tick can detect advancement.
    _persist_stalled_ctx(ctx, ctx.happy_session_id_str, new_missed)


def stalled_session_pass(
    dry_run: bool,
    threshold: int,
    now: float | None = None,
    *,
    daemon_reachable: bool | None = None,
    live_ids: set[str] | None = None,
    pids_by_sid: dict[str, int] | None = None,
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

    ``live_ids`` is the daemon's live-session set, threaded from
    :func:`main` exactly as :func:`orphan_sweep_pass` receives it (``None``
    when the daemon is unreachable). It drives the #759 bug-class-b.1
    bounded K-escalation in :func:`_process_stalled_session`. When omitted
    (``None`` — a direct unit/debug call that does not pass it), the
    corroboration is inert and the pass behaves exactly as before (a stalled
    ACTIVE session respawns), so existing callers are unaffected.
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
    pod_active_issues = {issue for issue, _pid, _name, _info in running_pods}
    pod_names_by_issue = {issue: name for issue, _pid, name, _info in running_pods}
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
            live_ids=live_ids,
            pids_by_sid=pids_by_sid,
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
            live_ids=live_ids,
            pids_by_sid=pids_by_sid,
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

    ``marker_age_s is None`` (no non-watcher marker at all) counts as
    stale — an ACTIVE task with zero non-watcher markers is itself the signal
    (mirrors the pod-safety pass's None-is-stale rule). The caller
    (``_process_orphan_task``) feeds the newest of ANY non-watcher marker
    kind, not just ``_PROGRESS_KINDS``, so a pre-pod lifecycle marker
    (epm:plan, epm:experiment-implementation, ...) counts as activity
    (#661/#658 sibling)."""
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


def _blocked_issue_ids() -> list[int]:
    """Sorted list of task ids currently at status ``blocked``, via one
    ``task.py list-by-status --status blocked --json`` subprocess. Same
    fail-soft isolation as :func:`_active_status_tasks` (a read failure yields
    an empty list, never a crash) and the same ``kind: campaign`` exclusion —
    the capacity-retry recovery command is ``spawn-issue --auto`` (the `/issue`
    skill), which would boot the wrong skill on a campaign (task #586)."""
    try:
        res = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "list-by-status",
                "--status",
                "blocked",
                "--json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if res.returncode != 0:
        return []
    try:
        rows = json.loads(res.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    out: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") == "campaign":
            continue
        tid = row.get("id")
        if isinstance(tid, int):
            out.append(tid)
    return sorted(out)


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


def _respawn_orphan(issue: int, cap_gpu_hours: float, dry_run: bool) -> str:
    """Spawn a fresh ``--auto`` session for an orphaned active task. Mirrors
    :func:`_respawn_stalled_session` but with an ``RESPAWNED-ORPHAN`` log
    prefix so the operator can tell the recovery paths apart. Returns the
    #843 M1b tri-state ``"spawned" | "suppressed" | "failed"`` (see
    :func:`_respawn`). On ``"spawned"``, the spawn re-registers the issue
    (``spawn-issue --auto`` rewrites the registry), so the task re-enters
    normal respawn/stalled coverage."""
    if _auth_outage_spawn_gate(issue, "orphan", dry_run=dry_run) is not None:
        print(f"  RESPAWN-ORPHAN issue #{issue}: suppressed — auth-outage episode active")
        return "suppressed"
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto", "--auto-approve-gpu-hours", str(cap_gpu_hours),
    ]  # fmt: skip
    cmd.extend(_stalled_session_overrides(issue))
    if dry_run:
        print(f"  [dry-run] would respawn orphan: {' '.join(cmd)}")
        return "failed"  # dry-run: nothing spawned
    # #1027: an orphan usually has NO registration; a STALE issue-<N>.json,
    # when present, still carries the predecessor's spawned_at (fail-soft ->
    # None). Captured BEFORE the spawn rewrites the registry.
    prev_spawned_at = _registry_spawned_at(issue)
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(
            f"  RESPAWN-ORPHAN FAILED issue #{issue}: {res.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return "failed"
    _forward_marker_child_stderr(res, "spawn_session spawn-issue (orphan)")
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    if spawn_output_suppressed(res.stdout) is not None:
        print(
            f"  RESPAWN-ORPHAN issue #{issue}: suppressed — not respawned "
            f"(lease/collision): {first_line}"
        )
        return "suppressed"
    print(f"  RESPAWNED-ORPHAN issue #{issue} (active task, no live session): {first_line}")
    _auth_outage_record_spawn(issue, "orphan", prev_spawned_at)
    return "spawned"


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
    pod_active_issues = {issue for issue, _pid, _name, _info in running_pods}
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
    """Probe the orphan-sweep exemptions (the prose USER-PAUSE hold, the
    over-cap spend-approval park, the round-complete re-park, and the
    followups_running-parent-waiting-on-open-child suppression). Returns the
    (possibly rewritten) ``action`` plus the human-readable reason string
    (for the alert prose) or ``None`` when no exemption applies.

    No-op unless ``action == "respawn"`` (the only orphan action whose
    fallout is wasteful in this regime) so a healthy task / a manual-only
    or cap-exhausted task never pays the ``task.py list-children``
    subprocess. Factored out of :func:`_process_orphan_task` to keep
    that function under the C901 cap.
    """
    if action != "respawn":
        return action, None
    # Prose USER-PAUSE hold (incident #816, 2026-07-02): the incident arm —
    # #816's respawn was an orphan-respawn against a prose 'USER PAUSE' note
    # left at the ACTIVE status `running`. Mirror of the same exemption in
    # :func:`_apply_stalled_followups_exemption`; checked FIRST (an explicit
    # user directive is the most specific gate signal — both arms are
    # alert-only, so ordering only picks which alert text posts). Diverted to
    # a one-time alert that does NOT consume the daily respawn budget. Pure
    # (events-only, never consults _task_children); the dispatch posts the
    # marker.
    pause_reason = _user_pause_hold_reason(events)
    if pause_reason is not None:
        print(
            f"  issue #{issue}: ORPHAN-RESPAWN exemption — {pause_reason}; "
            f"diverting to alert-only (does NOT consume respawn budget)."
        )
        return "user-pause-hold-skip", pause_reason
    # Over-cap spend-approval park (incident #653, 2026-06-18): mirror of the
    # same exemption in :func:`_apply_stalled_followups_exemption`. The
    # status-hold variant keeps the task at the ACTIVE status
    # `followups_running`, so an orphan candidate (no live registered session)
    # parked at the spend-approval gate would be respawned straight back into
    # the same parked plan. Diverted to a one-time alert that does NOT consume
    # the daily respawn budget. Checked FIRST — most specific gate, status-
    # agnostic. Pure (events-only); the dispatch posts the marker.
    spend_reason = _spend_approval_park_reason(events)
    if spend_reason is not None:
        print(
            f"  issue #{issue}: ORPHAN-RESPAWN exemption — {spend_reason}; "
            f"diverting to alert-only (does NOT consume respawn budget)."
        )
        return "spend-approval-skip", spend_reason
    # Round-complete re-park (incident #533 freeze): probed BEFORE the
    # awaiting-child suppression — a completed same-issue follow-up round
    # stranded at followups_running is fixed by executing the re-park, not
    # by suppressing or respawning. The actual mutation happens in
    # :func:`_handle_orphan_followup_round_repark` (this helper stays
    # read-only, mirroring the awaiting-child probe).
    if status == "followups_running" and not has_pod:
        repark_reason = _followup_round_complete_reason(events, issue=issue)
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


def _handle_orphan_spend_approval_skip(
    *,
    issue: int,
    reason: str,
    new_missed: int,
    alerted: bool,
    respawn_day: str,
    respawns_today: int,
    followups_child_alerted: bool,
    events: list[dict],
    state: dict,
    dry_run: bool,
) -> None:
    """Orphan-sweep handler for the over-cap spend-approval exemption (incident
    #653): post the one-time alert and persist state WITHOUT incrementing
    ``respawns_today`` — the exemption deliberately does NOT consume the daily
    respawn budget. Dedup is self-contained in the events log (a marker
    carrying :data:`_SPEND_APPROVAL_SKIP_NOTE_SENTINEL` newer than the gating
    ``epm:awaiting-spend-approval``), so no per-pass state flag is threaded; a
    fresh spend-approval episode re-arms the alert. Factored out of
    :func:`_process_orphan_task` to keep that function under the C901 cap."""
    if not _spend_approval_skip_already_noted(events):
        _post_progress_marker(
            issue,
            f"{_SPEND_APPROVAL_SKIP_NOTE_SENTINEL} {reason}. "
            f"Orphan-respawn suppressed (does NOT consume the daily respawn "
            f"budget); the user must approve the over-cap plan "
            f"(`task.py set-status {issue} approved`) or re-plan "
            f"(`task.py set-status {issue} planning` + re-invoke "
            f"/adversarial-planner) to advance this task.",
            dry_run,
            label="spend-approval-skip",
        )
    if not dry_run:
        _save_orphan_state(
            issue,
            missed=new_missed,
            alerted=alerted,
            respawn_day=respawn_day,
            respawns_today=respawns_today,
            followups_child_alerted=followups_child_alerted,
            prev=state,
        )


def _handle_orphan_user_pause_skip(
    *,
    issue: int,
    reason: str,
    new_missed: int,
    alerted: bool,
    respawn_day: str,
    respawns_today: int,
    followups_child_alerted: bool,
    events: list[dict],
    state: dict,
    dry_run: bool,
) -> None:
    """Orphan-sweep handler for the prose USER-PAUSE hold exemption (incident
    #816): post the one-time alert (events-log dedup via
    :func:`_maybe_post_user_pause_skip`) and persist state WITHOUT
    incrementing ``respawns_today`` — the exemption deliberately does NOT
    consume the daily respawn budget. Dedup is self-contained in the events
    log (a marker carrying :data:`_USER_PAUSE_SKIP_NOTE_SENTINEL` at/after
    the latest pause note), so no per-pass state flag is threaded; a fresh
    pause note re-arms the alert. Mirror of
    :func:`_handle_orphan_spend_approval_skip`. Factored out of
    :func:`_process_orphan_task` to keep that function under the C901 cap."""
    _maybe_post_user_pause_skip(issue, reason, events, dry_run)
    if not dry_run:
        _save_orphan_state(
            issue,
            missed=new_missed,
            alerted=alerted,
            respawn_day=respawn_day,
            respawns_today=respawns_today,
            followups_child_alerted=followups_child_alerted,
            prev=state,
        )


# The orphan-sweep exemption actions: each diverts a would-be respawn to a
# park-aware handler that does NOT consume the daily respawn budget. Dispatched
# uniformly by :func:`_dispatch_orphan_exemption_action` to keep
# :func:`_process_orphan_task` under the C901 cap (15).
_ORPHAN_EXEMPTION_ACTIONS = frozenset(
    {
        "user-pause-hold-skip",
        "spend-approval-skip",
        "followup-round-repark",
        "followups-awaiting-child",
    }
)


def _dispatch_orphan_exemption_action(
    *,
    action: str,
    issue: int,
    followups_reason: str | None,
    events: list[dict],
    new_missed: int,
    alerted: bool,
    day_key: str,
    respawns_today: int,
    followups_child_alerted: bool,
    state: dict,
    dry_run: bool,
) -> None:
    """Route one of the orphan-sweep exemption actions
    (:data:`_ORPHAN_EXEMPTION_ACTIONS`) to its handler. Extracted from
    :func:`_process_orphan_task` so the dispatch chain there stays under the
    C901 cyclomatic-complexity cap (15)."""
    if action == "spend-approval-skip":
        _handle_orphan_spend_approval_skip(
            issue=issue,
            reason=followups_reason or "",
            new_missed=new_missed,
            alerted=alerted,
            respawn_day=day_key,
            respawns_today=respawns_today,
            followups_child_alerted=followups_child_alerted,
            events=events,
            state=state,
            dry_run=dry_run,
        )
    elif action == "user-pause-hold-skip":
        _handle_orphan_user_pause_skip(
            issue=issue,
            reason=followups_reason or "",
            new_missed=new_missed,
            alerted=alerted,
            respawn_day=day_key,
            respawns_today=respawns_today,
            followups_child_alerted=followups_child_alerted,
            events=events,
            state=state,
            dry_run=dry_run,
        )
    elif action == "followup-round-repark":
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
    elif action == "followups-awaiting-child":
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


def _parse_orphan_state_counters(state: dict, day_key: str) -> tuple[int, int]:
    """Type-guarded ``(missed, respawns_today)`` from a loaded orphan-state
    dict; ``respawns_today`` resets to 0 when the recorded ``respawn_day`` is
    not today's ``day_key``. Factored out of :func:`_process_orphan_task` to
    keep it under the C901 cap after the #903 takeover-sentinel skip landed
    there (behavior-preserving extraction)."""
    missed = state.get("missed", 0)
    if not isinstance(missed, int):
        missed = 0
    respawns_today = state.get("respawns_today", 0) if state.get("respawn_day") == day_key else 0
    if not isinstance(respawns_today, int):
        respawns_today = 0
    return missed, respawns_today


def _orphan_act_guard(
    issue: int, *, status: str, action: str, state: dict, dry_run: bool
) -> str | None:
    """#1247 terminal-status act guard: re-read the LIVE status through the
    canonical resolver (:func:`_task_status` — ``cwd=PROJECT_ROOT``,
    git-common-dir resolved, #844) immediately before an acting branch and
    require POSITIVE ACTIVE confirmation. Returns the live status on
    confirmation (the caller acts + writes marker notes against IT, not the
    stale snapshot) or ``None`` to abort the act: no respawn, no marker, no
    cap consumed. A positively non-ACTIVE read clears the orphan episode
    state (matching :func:`decide_orphan`'s own ``status not in ACTIVE ->
    clear`` semantics); a ``None`` read (transient task.py failure) keeps
    the state and defers one tick — fail toward not-acting, never toward
    erasing episode state on a task.py glitch. Runs under ``--dry-run`` too
    (the read is a read-only subprocess; the state-clear is dry_run-gated)."""
    live_status = _task_status(issue)
    if live_status in ACTIVE:
        return live_status
    print(
        f"  ORPHAN-ACT-GUARD issue #{issue}: snapshot status={status} but "
        f"live re-read returned {live_status!r} — not ACTIVE; aborting "
        f"action={action}: no respawn, no marker, no cap consumed "
        f"(#1247 stale-snapshot guard).",
        file=sys.stderr,
    )
    if live_status is not None and state and not dry_run:
        _clear_orphan_state(issue)  # positively non-active: episode over
    return None


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
    #472 class). Honours dry_run (logs but never mutates / spawns).

    #866/#903: a FRESH ``paused-takeover`` sentinel (a deliberate session
    takeover renamed the registration away) skips the issue ENTIRELY before
    any state read/write — the frozen ``missed`` count resumes exactly where
    it left off when the sentinel goes stale (FAIL OPEN). No marker is posted
    (a per-tick marker would spam events.jsonl; this stdout log is the
    record)."""
    sentinel = takeover_sentinel_fresh(issue, now=now, registry_dir=AUTONOMOUS_REGISTRY_DIR)
    if sentinel is not None:
        print(
            f"  issue #{issue}: status={status} SKIP — deliberate takeover sentinel "
            f"{sentinel.name} is FRESH (< EPS_TAKEOVER_TTL_H); orphan sweep deferred "
            f"(stale sentinel resumes normal respawn — fail open)"
        )
        return
    mapped_alive = bool(rec and rec["sids"] & live_ids)
    manual_only = bool(rec and rec["has_manual"] and not rec["has_auto"])
    entry_age_s = (now - rec["newest_write"]) if rec and rec["newest_write"] > 0 else None
    state = _load_orphan_state(issue)
    missed, respawns_today = _parse_orphan_state_counters(state, day_key)
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
        # Count ANY non-watcher marker as activity (NOT just _PROGRESS_KINDS):
        # an alive-but-unregistered session in a long pre-pod phase posts only
        # pre-run lifecycle markers (epm:plan, epm:experiment-implementation,
        # epm:code-review, epm:review-reconcile, ...), all excluded from the
        # narrow _PROGRESS_KINDS allowlist — gating staleness on that allowlist
        # falsely read those sessions as zero-progress and respawned them. The
        # _WATCHER_NOTE_SENTINELS note filter still excludes the watcher's own
        # posts. Matches the stalled-detector + reconcile passes (#661/#658
        # sibling).
        latest = _latest_nonwatcher_event_ts(events)
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
    # ── #1247 terminal-status act guard ─────────────────────────────────
    # Every branch below ACTS (spawns, posts a marker, or consumes the
    # daily cap) on the pass-start _active_status_tasks() snapshot, which
    # can be stale (TOCTOU within a tick; a stale task-state view fed the
    # 2-week #662/#663/#867 marker loop). Re-verify the LIVE status
    # through the canonical resolver (cwd=PROJECT_ROOT, git-common-dir
    # resolved, #844) immediately before acting. POSITIVE confirmation
    # required: act only on live ∈ ACTIVE. Helper-factored to keep this
    # function under the C901 cap.
    live_status = _orphan_act_guard(
        issue, status=status, action=action, state=state, dry_run=dry_run
    )
    if live_status is None:
        return
    status = live_status  # act + write marker notes against the LIVE status
    if action == "respawn":
        spawn_result = _respawn_orphan(issue, _stalled_cap_gpu_hours(issue), dry_run)
        if spawn_result == "suppressed":
            # #843 M1b: duplicate dispatch suppressed (lease/collision) — a
            # session is driving. Book nothing: no attempt consumed against
            # the daily cap, no miss-state reset, no respawn marker.
            return
        attempted_ok = spawn_result == "spawned"
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
                    f"today). {_source_stamp()}",
                    dry_run,
                    label="orphan-respawn",
                )
        return
    if action in _ORPHAN_EXEMPTION_ACTIONS:
        _dispatch_orphan_exemption_action(
            action=action,
            issue=issue,
            followups_reason=followups_reason,
            events=events,
            new_missed=new_missed,
            alerted=alerted,
            day_key=day_key,
            respawns_today=respawns_today,
            followups_child_alerted=followups_child_alerted,
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
            f"scripts/spawn_session.py spawn-issue --issue {issue} --auto "
            f"{_source_stamp()}",
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
# Used only when the queue file omits `cap` (the body names 5 as the schema
# default; a benign omission must not silently disable the drain).
INFRA_DRAIN_CAP_DEFAULT = 5
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

# ── predicate-hold auto-promotion (#633 follow-on) ────────────────────────────
# The PM encodes a cross-issue dependency hold as `predicate-<#N>-<short-desc>`
# (research-pm.md step 3; live examples `predicate-535-slurm-attempt`,
# `predicate-625-lands`): the held task is ready only once task #N reaches a
# terminal/landed state. The PM re-evaluates these on its STATUS pass, but
# passes can be hours apart; this watcher pass mechanically promotes a hold the
# instant its predicate is SATISFIED so the held task dispatches between PM
# passes. CONSERVATIVE satisfaction: only the unambiguous "upstream finished"
# signal — task #N at `completed`/`archived`/`awaiting_promotion`. The
# `<short-desc>` is NEVER interpreted (a "slurm-attempt" predicate is satisfied
# by completion too — a completed #535 definitely had its live attempt); the
# PM's STATUS-pass re-evaluation remains the nuanced backstop for predicates
# that should fire BEFORE completion. Mirrors `TERMINAL` today but pinned
# separately so this pass's contract is self-documenting and test-locked.
INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES = frozenset(
    {"completed", "archived", "awaiting_promotion"}
)
# Prefix marking a hold reason as a machine-parseable cross-issue predicate.
# Any hold whose reason does NOT start with this (e.g. `credentials`,
# `needs-thomas`, `spend`, `re-kind`, `irreversible`) is a PM-judgment hold and
# is NEVER touched by this pass.
INFRA_DRAIN_PREDICATE_PREFIX = "predicate-"
# Identifier stamped into `updated_by` when THIS pass rewrites the queue file
# (vs the PM's `pm-session`), so a queue diff attributes the promotion.
INFRA_DRAIN_QUEUE_WRITER = "autonomous_session_watch:predicate-promote"

# ── proposed-infra sweep (always-on backstop for orphaned ripe infra; #690) ──
# On-task tag marking a `proposed` infra/batch task as a /daily route-3 held
# judgment call (#706). A task carrying this tag is NEVER an auto-dispatch
# candidate — it is surfaced in the PM `Needs you` block for Thomas's call, not
# auto-run by this always-on sweep. Excluded in `_proposed_infra_candidates`.
_NEEDS_HUMAN_TAG = "needs-human"
# A SEPARATE attempt/backoff state file from the PM-queue drain's
# `infra-drain-state.json`: a single task can be reachable via BOTH the PM
# queue (the drain) AND the sweep (a `proposed` infra task the watcher
# enumerates), so sharing one state file would let one path's attempt budget
# silently throttle the other. Separate files keep each path's churn guard
# independent + independently testable.
PROPOSED_INFRA_SWEEP_STATE_BASENAME = "proposed-infra-sweep-state.json"
# Backoff window for a sweep-dispatched task whose spawn keeps failing — reuses
# the infra-drain window VALUE (1 h ≈ 6 ticks at the 10-min cron) under its own
# env var. env EPM_PROPOSED_INFRA_SWEEP_BACKOFF_S.
PROPOSED_INFRA_SWEEP_BACKOFF_S_DEFAULT = INFRA_DRAIN_BACKOFF_S_DEFAULT
# Per-task attempt cap before the sweep parks it (the entry is pruned when the
# task leaves `proposed`, so a PM rewrite / repromotion / status change clears
# it naturally). env EPM_PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS.
PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS_DEFAULT = INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT

# #843 M3: the sweep skips a decided candidate whose events.jsonl carries a
# dispatch-sentinel marker younger than this — one watcher cadence (cron
# `3-59/10` -> 10 min): a dispatch marker younger than one tick means SOME
# dispatcher fired within the current/previous tick. Post-M1, this guard's
# independent value is the lease-file-loss backstop (manual `rm` override,
# a GC/registry mishap) plus observability. env
# EPM_PROPOSED_INFRA_SWEEP_MARKER_FRESH_S.
PROPOSED_INFRA_SWEEP_MARKER_FRESH_S_DEFAULT = 600.0

# Both dispatch-marker sentinels disqualify (a drain dispatch 3 min ago is
# exactly as disqualifying as a sweep one). The filer posts no marker — its
# dispatches are covered by the M1 lease + the registration checks.
_DISPATCH_NOTE_SENTINELS = (
    _PROPOSED_INFRA_SWEEP_NOTE_SENTINEL,
    _INFRA_DRAIN_NOTE_SENTINEL,
)


def _proposed_infra_sweep_marker_fresh_s() -> float:
    """Marker-freshness window in seconds (env
    ``EPM_PROPOSED_INFRA_SWEEP_MARKER_FRESH_S``; missing or malformed value
    falls back to :data:`PROPOSED_INFRA_SWEEP_MARKER_FRESH_S_DEFAULT`)."""
    raw = os.environ.get("EPM_PROPOSED_INFRA_SWEEP_MARKER_FRESH_S")
    if not raw:
        return PROPOSED_INFRA_SWEEP_MARKER_FRESH_S_DEFAULT
    try:
        return float(raw)
    except ValueError:
        return PROPOSED_INFRA_SWEEP_MARKER_FRESH_S_DEFAULT


def _recent_dispatch_marker_age_s(events: list[dict], now: float) -> float | None:
    """Age (s) of the newest ``epm:progress`` marker whose note carries either
    dispatch sentinel (:data:`_DISPATCH_NOTE_SENTINELS`); ``None`` when there
    is no such marker or no parseable timestamp (fail-soft: an unparseable
    ``ts`` row is skipped, and a fully-unreadable events list arrives here as
    ``[]`` via :func:`_task_events` -> ``None`` -> no skip; the M1 lease still
    protects)."""
    best: float | None = None
    for ev in events:
        if ev.get("kind") != "epm:progress":
            continue
        note = ev.get("note") or ""
        if not any(sentinel in note for sentinel in _DISPATCH_NOTE_SENTINELS):
            continue
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return None if best is None else now - best


def _parse_predicate_hold(reason: str) -> int | None:
    """Extract the predicate's blocking issue number from a hold reason of the
    form ``predicate-<#N>-<short-desc>``. Returns the int ``N`` when the reason
    starts with :data:`INFRA_DRAIN_PREDICATE_PREFIX` and the token after the
    prefix (split on ``-``) is all-digits; ``None`` otherwise (a non-predicate
    PM-judgment hold, or a malformed predicate string — fail toward NOT
    touching the hold). The PM's live convention writes the bare digits with no
    leading ``#`` (e.g. ``predicate-535-slurm-attempt``); a stray ``#`` is
    stripped defensively so ``predicate-#535-...`` parses too."""
    if not isinstance(reason, str) or not reason.startswith(INFRA_DRAIN_PREDICATE_PREFIX):
        return None
    parts = reason.split("-")
    # parts[0] == "predicate" (the prefix has no internal '-'); parts[1] is the
    # issue token. A bare "predicate-" yields parts == ["predicate", ""].
    if len(parts) < 2:
        return None
    token = parts[1].lstrip("#")
    if not token.isdigit():
        return None
    return int(token)


def _satisfied_predicate_promotions(
    holds: dict[int, str],
    predicate_statuses: dict[int, str | None],
) -> tuple[list[int], dict[int, str]]:
    """Pure decision: given the current ``holds`` and the resolved status of
    each predicate's blocking issue, return
    ``(promote_ids, remaining_holds)``.

    ``predicate_statuses`` maps a BLOCKING issue number (the ``N`` parsed out of
    a ``predicate-<#N>-...`` reason) to that task's current status (``None`` =
    unreadable). A held entry is promoted iff its reason parses as a predicate
    AND the blocking task is at a status in
    :data:`INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES`. EVERYTHING else — a
    non-predicate hold, a malformed predicate, an unreadable blocking status,
    or a blocking task not yet terminal — is left in ``remaining_holds``
    UNTOUCHED (fail toward keep-blocking; the PM remains the nuanced judge).

    ``promote_ids`` is the list of HELD task ids whose predicate cleared, in
    ascending id order (deterministic; the caller merges them into
    ``ripe_oldest_first`` preserving the queue's oldest-first ordering)."""
    promote: list[int] = []
    remaining: dict[int, str] = {}
    for held_id, reason in holds.items():
        blocking = _parse_predicate_hold(reason)
        if blocking is None:
            remaining[held_id] = reason  # non-predicate / malformed — never touch
            continue
        status = predicate_statuses.get(blocking)
        if status in INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES:
            promote.append(held_id)
        else:
            remaining[held_id] = reason  # not yet satisfied / unreadable — keep
    promote.sort()
    return promote, remaining


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


def _load_attempt_state(path: Path, max_attempts: int, *, log_prefix: str) -> dict:
    """Shared per-task attempt/backoff state loader (``{"attempts": {int:
    rec}}``; empty on absent/garbled, mirroring :func:`_load_orphan_state`).
    Consumed by BOTH :func:`_load_infra_drain_state` (the PM-queue drain) and
    :func:`_load_proposed_infra_sweep_state` (the #690 orphan sweep) so the
    DEFENSIVE NORMALIZE rule lives in ONE place and the two paths cannot drift.
    ``log_prefix`` namespaces the log lines (``"infra-drain"`` /
    ``"proposed-infra-sweep"``); ``max_attempts`` is the cap a garbled/missing
    count is normalized UP to.

    On-disk keys are strings (JSON); they are int-normalized here. DEFENSIVE
    NORMALIZE: drop (with a log line) any record whose ``last_attempt_ts`` is
    not numeric or whose key is not an int — a garbled record must not silently
    bypass the budget. A record with a MISSING ``attempts`` key or whose
    ``attempts`` COUNT is garbled (non-int / bool / negative) keeps its valid
    ``last_attempt_ts`` (the backoff window still binds) but has the count
    normalized UP to ``max_attempts`` — count unknown means the budget may be
    spent, so fail toward NOT dispatching. Dropping the record instead would
    RESET the budget — the fail-open direction (round-2 fix, Codex Major:
    ``int("bad") + 1`` / ``"bad" >= max_attempts`` crashed the whole pass;
    round-4 fix for the MISSING-key sibling: bare ``rec.get("attempts", 0)``
    returned 0 before the type check fired, silently granting a fresh budget on
    a half-written / hand-edited record)."""
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
            print(f"  {log_prefix}: dropping garbled state record (non-int key {key!r})")
            continue
        last = rec.get("last_attempt_ts") if isinstance(rec, dict) else None
        if isinstance(last, bool) or not isinstance(last, int | float):
            print(
                f"  {log_prefix}: dropping garbled state record for #{issue} "
                f"(non-numeric last_attempt_ts {last!r})"
            )
            continue
        if "attempts" not in rec:
            # Missing key — half-written or hand-edited record. Bare
            # ``rec.get("attempts", 0)`` would return 0 BEFORE the type
            # check fired below and silently grant a fresh budget; fail
            # to cap instead, the same fail direction the
            # garbled-count branch already uses (round-4 fix).
            print(
                f"  {log_prefix}: state record for #{issue} is missing its attempts "
                f"key; normalizing to the attempt cap ({max_attempts}) — fail toward NOT "
                f"dispatching (a fresh PM updated_ts resets it)"
            )
            rec = {**rec, "attempts": max_attempts}
        else:
            count = rec["attempts"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                print(
                    f"  {log_prefix}: state record for #{issue} has a garbled attempts "
                    f"count ({count!r}); normalizing to the attempt cap ({max_attempts}) — fail "
                    f"toward NOT dispatching (a fresh PM updated_ts resets it)"
                )
                rec = {**rec, "attempts": max_attempts}
        attempts[issue] = rec
    return {"attempts": attempts}


def _load_infra_drain_state() -> dict:
    """Read the PM-queue drain's attempt/backoff state via the shared
    :func:`_load_attempt_state` (a fresh PM ``updated_ts`` newer than
    ``last_attempt_ts`` later resets the count, so recovery is automatic on the
    next PM adjudication)."""
    return _load_attempt_state(
        _infra_drain_state_path(), _infra_drain_max_attempts(), log_prefix="infra-drain"
    )


def _save_infra_drain_state(state: dict) -> None:
    """Persist the attempt/backoff state atomically (temp + rename, mirroring
    :func:`_save_orphan_state`). Int keys serialize as strings; the loader
    normalizes them back."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _infra_drain_state_path()
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(dest)


def _save_infra_drain_queue(
    ids: list[int],
    cap: int,
    holds: dict[int, str],
    now: float,
    comment: str,
) -> None:
    """Rewrite the PM queue file atomically (temp + rename) after this pass
    promotes satisfied predicate holds. Writes the canonical schema the PM uses
    (``ripe_oldest_first`` / ``cap`` / ``holds`` / ``updated_ts`` /
    ``updated_by`` / ``comment``) — string hold keys, ISO-8601 Z ``updated_ts``
    — so the next PM read parses it identically. ``updated_by`` is stamped with
    :data:`INFRA_DRAIN_QUEUE_WRITER` (NOT ``pm-session``) so a queue diff
    attributes the promotion to the watcher; ``updated_ts`` is bumped to ``now``
    (this re-arms the per-ID retry budget for the promoted IDs — desired: a
    just-cleared task gets a fresh attempt budget). The PM overwrites the file
    wholesale on its next STATUS pass, so its re-adjudication always wins; this
    write only races the PM's own atomic rename, and rename atomicity means a
    reader sees exactly one complete file or the other (never a torn write).
    Holds with int keys are JSON-serialized as strings, matching the PM's
    on-disk shape (the loader int()-parses them back)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ripe_oldest_first": ids,
        "cap": cap,
        "holds": {str(k): v for k, v in holds.items()},
        "updated_ts": datetime.fromtimestamp(now, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_by": INFRA_DRAIN_QUEUE_WRITER,
        "comment": comment,
    }
    dest = _infra_drain_queue_path()
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
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


def infra_dispatch_has_free_slot(
    pending: int = 0, *, cap: int = INFRA_DRAIN_CAP_DEFAULT
) -> bool | None:
    """True iff a NEW infra/batch dispatch fits under the shared cap (#690 M1).

    The SINGLE cap-check primitive the file-time wrapper
    (:mod:`file_infra_task`), the infra-drain pass, and the proposed-infra
    sweep all consult, so a future cap-tightening refactor edits exactly one
    function and the three dispatchers cannot drift apart. It introduces NO
    new cap semantics — it composes the EXISTING :func:`_infra_drain_occupancy`
    + caller-supplied pending count with :data:`INFRA_DRAIN_CAP_DEFAULT`.

    Tri-state, fail-CLOSED on unreadable occupancy (mirrors the executor's
    ``_infra_drain_occupancy() is None -> skip`` rule):

      - ``None``  -> occupancy UNREADABLE (any ``list-by-status`` read failed);
        callers MUST NOT dispatch this tick (a partial count over-dispatches).
      - ``False`` -> cap full (``occupied_active + pending >= cap``).
      - ``True``  -> at least one free slot.

    ``occupied_active = len(_infra_drain_occupancy())`` (kind-infra/batch tasks
    at :data:`INFRA_DRAIN_OCCUPIED_STATUSES`); ``pending`` is the
    caller-supplied count of in-flight non-stale registrations. The two
    watcher passes pass the real :func:`_infra_drain_pending`; the file-time
    wrapper passes ``0`` — at file time there is no fresh same-tick
    registration to count, and a just-filed task has no registration yet, so
    occupancy alone is the binding budget for the one spawn the wrapper is
    about to make. Negative ``pending`` is clamped to 0 defensively."""
    occ = _infra_drain_occupancy()
    if occ is None:
        return None
    return (len(occ) + max(0, pending)) < cap


def _stagger_sleep(seconds: float) -> None:
    """Seam for the #1059 session-dispatch stagger — tests monkeypatch this
    so no test ever really sleeps."""
    time.sleep(seconds)


def _dispatch_infra_drain(
    issue: int,
    slot_desc: str,
    dry_run: bool,
    *,
    reg_snapshot: dict[str, bytes] | None = None,
) -> str:
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

    Returns the #843 M1b tri-state ``"spawned" | "suppressed" | "failed"``:
    ``"suppressed"`` when the rc-0 subprocess stdout carries a
    duplicate-suppression sentinel (DISPATCH-LEASE HELD /
    REGISTRATION-COLLISION, via :func:`spawn_session.spawn_output_suppressed`)
    — a loud no-op the callers must NOT book: no dispatch marker, no attempt,
    no backoff (a crashed lease-winner then recovers in <= TTL + one tick,
    not the 1 h backoff). ``"failed"`` also covers dry-run (logs, never
    spawns, nothing to book) and the pre-spawn re-check aborts (both record
    a spawn-failed attempt in the callers, exactly as before). ``"suppressed"``
    ALSO covers the #1027 auth-outage gate (same no-booking contract); this
    single hook covers BOTH callers (``infra_drain_pass`` +
    ``proposed_infra_sweep_pass``). Before a real spawn it additionally
    sleeps out the remainder of the #1059 session-dispatch stagger window
    (:func:`spawn_session.stagger_delay_s`; dry-run returns first and never
    sleeps), so consecutive session dispatches land >=
    ``EPM_SESSION_DISPATCH_STAGGER_S`` (60s) apart — one ~100K-token cold
    session load per minute-boundary 429 window."""
    if _auth_outage_spawn_gate(issue, "infra-drain", dry_run=dry_run) is not None:
        print(f"  INFRA-DRAIN issue #{issue}: suppressed — auth-outage episode active")
        return "suppressed"
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto",
    ]  # fmt: skip
    if dry_run:
        print(f"  [dry-run] would dispatch infra-drain: {' '.join(cmd)}")
        return "failed"  # dry-run: nothing spawned, nothing to book
    # #1059 session-dispatch stagger: sleep out the remainder of the pacing
    # window BEFORE the pre-spawn re-check (so the re-check runs maximally
    # fresh; a registration/lease appearing DURING the sleep is caught by the
    # re-check below / the spawn subprocess's own #843 lease acquisition).
    window = session_dispatch_stagger_s()
    delay = stagger_delay_s(last_session_dispatch_age_s(), window)
    if delay > 0:
        print(
            f"  INFRA-DRAIN STAGGER issue #{issue}: last session dispatch "
            f"{window - delay:.0f}s ago < {window:.0f}s window; sleeping "
            f"{delay:.0f}s (429 token-pacing, #1059)"
        )
        _stagger_sleep(delay)
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
            return "failed"
        if known is None or current != known:
            print(
                f"  INFRA-DRAIN ABORT issue #{issue}: lost race to concurrent "
                f"dispatcher ({basename} "
                f"{'appeared' if known is None else 'changed'} since the decision)"
            )
            return "failed"
        # else: byte-identical to the decision-time snapshot — the known
        # (stale) registration the decision already accounted for; proceed.
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(
            f"  INFRA-DRAIN DISPATCH FAILED issue #{issue}: {res.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return "failed"
    _forward_marker_child_stderr(res, "spawn_session spawn-issue (infra-drain)")
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    suppressed = spawn_output_suppressed(res.stdout)
    if suppressed is not None:
        # #843 M1b: rc-0 no-op — a duplicate dispatch was suppressed at the
        # chokepoint (lease held / registration collision). A session IS
        # driving the issue; callers book nothing (no attempt, no marker).
        print(f"  INFRA-DRAIN SUPPRESSED issue #{issue} ({suppressed}): {first_line}")
        return "suppressed"
    print(f"  INFRA-DRAIN DISPATCHED issue #{issue} ({slot_desc}): {first_line}")
    _auth_outage_record_spawn(issue, "infra-drain", None)
    record_session_dispatch(issue, "watcher-infra-dispatch")
    return "spawned"


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


def _promote_satisfied_predicate_holds(
    ids: list[int],
    cap: int,
    holds: dict[int, str],
    now: float,
    dry_run: bool,
) -> tuple[list[int], dict[int, str]]:
    """Mechanically promote any ``predicate-<#N>-...`` hold whose blocking task
    #N has reached a terminal/landed state (#633 follow-on). Returns the
    (possibly updated) ``(ids, holds)`` for the rest of the pass to consume; a
    promoted id is APPENDED to the END of ``ids``, preserving the PM's existing
    positional ordering (``ripe_oldest_first`` is positional — oldest-first by
    default, urgency-first when the PM names an active incident — NOT
    ascending-id; a re-sort would silently re-prioritize the whole queue and,
    under a cap, starve a PM-prioritized urgent task), and its hold key removed.

    This is the ONLY ripeness judgment the watcher makes, and a deliberately
    narrow one: it reads each predicate's BLOCKING-issue status (one
    ``task.py view`` subprocess per distinct predicate, none on the common
    no-predicate tick) and promotes only on the unambiguous "upstream finished"
    signal (:func:`_satisfied_predicate_promotions`). Non-predicate /
    PM-judgment holds are never inspected. ``cap`` is passed through UNCHANGED
    for the file round-trip only — promotion is cap-independent (the cap gates
    DISPATCH downstream, never promotion). On a promotion it rewrites the queue
    file atomically (skipped under ``dry_run``) so the held task dispatches THIS
    tick AND survives for the bg poller; the PM's next STATUS pass re-adjudicates
    wholesale, so this write is purely a between-passes accelerator."""
    if not holds:
        return ids, holds
    # Only predicate holds (parseable blocking-issue token) cost a status read;
    # non-predicate holds short-circuit with zero subprocesses.
    predicate_blockers = {b for r in holds.values() if (b := _parse_predicate_hold(r)) is not None}
    if not predicate_blockers:
        return ids, holds
    predicate_statuses = {b: _task_status_kind(b)[0] for b in sorted(predicate_blockers)}
    promote, remaining = _satisfied_predicate_promotions(holds, predicate_statuses)
    if not promote:
        return ids, holds
    # APPEND promoted ids to the END, preserving the PM's positional ordering
    # (`ripe_oldest_first` is positional, not ascending-id — re-sorting would
    # re-prioritize the whole queue and could starve a PM urgency-first head
    # under the cap). `promote` is already ascending (promote.sort() in
    # _satisfied_predicate_promotions); de-dupe against ids defensively (a hold
    # key should never also already be in ripe_oldest_first).
    existing = set(ids)
    new_ids = ids + [pid for pid in promote if pid not in existing]
    cleared = ", ".join(f"#{pid} (cleared by {holds[pid]})" for pid in promote)
    print(f"  infra-drain: PREDICATE-PROMOTE {len(promote)} hold(s) -> ripe: {cleared}")
    if not dry_run:
        comment = f"watcher auto-promoted {len(promote)} satisfied predicate hold(s): {cleared}"
        _save_infra_drain_queue(new_ids, cap, remaining, now, comment)
    else:
        print("  [dry-run] would rewrite the queue file with the promoted id(s)")
    return new_ids, remaining


def infra_drain_pass(
    dry_run: bool,
    now: float | None = None,
    *,
    daemon_reachable: bool | None = None,
) -> None:
    """Execute the PM-adjudicated infra dispatch queue (task #633). Pure
    executor for DISPATCH — every dispatch judgment was the PM session's; every
    dispatch guard here is mechanical. The ONE ripeness judgment it makes is the
    narrow mechanical promotion of a ``predicate-<#N>-...`` hold whose blocking
    task #N FINISHED (:func:`_promote_satisfied_predicate_holds`), so a cleared
    predicate dispatches between PM STATUS passes. Missing/invalid queue file =
    logged no-op; daemon-gated like every other spawning pass (spawn POSTs to
    the Happy daemon RPC)."""
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
    # Promote any predicate hold whose blocking task finished BEFORE the dispatch
    # logic runs, so a just-cleared id flows through the normal guard/cap path
    # this same tick. On a promotion the queue file is rewritten with
    # updated_ts == now, so mirror that into queue_updated_ts (re-arms the
    # promoted ids' retry budget, consistent with the file the next PM/watcher
    # read will see).
    promoted_ids, promoted_holds = _promote_satisfied_predicate_holds(ids, cap, holds, now, dry_run)
    if promoted_holds != holds:
        ids, holds = promoted_ids, promoted_holds
        queue_updated_ts = now
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
        # #843 M1 advisory pre-check at the CALLER loop: a fresh per-issue
        # dispatch lease means a spawn is already in flight — skip loudly and
        # record NO attempt (a lease-held skip must not consume the 1 h
        # backoff, or a crashed winner's recovery would stretch to ~70 min
        # instead of TTL + one tick).
        held_lease = dispatch_lease_fresh(issue, now)
        if held_lease is not None:
            print(
                f"  INFRA-DRAIN SKIP issue #{issue} (dispatch-lease held, "
                f"{dispatch_lease_desc(held_lease, now)})"
            )
            continue
        slot_desc = f"slot {min(occupied_active + pending + dispatched + 1, cap)}/{cap}"
        result = _dispatch_infra_drain(issue, slot_desc, dry_run, reg_snapshot=reg_snapshot)
        if result == "suppressed":
            # #843 M1b: rc-0 duplicate-suppression no-op — a session is
            # driving; book nothing (no attempt, no backoff, no marker).
            continue
        if not dry_run:
            # Count the ATTEMPT whether the spawn succeeded or failed, so a
            # failing spawn can't tight-loop (the backoff window binds next
            # tick either way).
            _infra_drain_record_attempt(attempts, issue, now, queue_updated_ts, result == "spawned")
        if result == "spawned":
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


# ─── proposed-infra sweep (always-on backstop for orphaned ripe infra; #690) ─
#
# WHY THIS PASS EXISTS. The PM-queue path (infra_drain_pass above) requires a
# LIVE PM session to ADJUDICATE which `proposed` infra tasks are ripe and write
# them to `infra-drain-queue.json`; the drain pass only EXECUTES that file.
# Nothing autonomously adjudicates a freshly-filed `proposed` infra task when
# no PM is running, so it orphans (incident #684 sat at `proposed` ~17h). This
# pass is the always-on backstop: it BUILDS its own candidate set from
# `task.py list-by-status --status proposed --json` (NOT the PM queue),
# consults the PM queue file's `holds` map ONLY to honor predicate/user holds
# (no new on-task tag convention), and dispatches ripe orphans into free slots
# under the SAME shared cap the drain uses. It REUSES the drain's leaf
# primitives (occupancy / pending / registration-dedup / stale / dispatch /
# predicate parsing) — it does NOT execute the PM queue.
#
# RELATION TO infra_drain_pass. The two differ in exactly ONE structural axis —
# candidate SOURCE (PM-written queue file vs. `list-by-status`) — but share all
# dispatch/cap/dedup/predicate primitives. They are kept SEPARATE functions
# (not a `mode=` branch through infra_drain_pass) so each stays single-purpose,
# the body's "ADDS paths, not REMOVES the PM one" invariant is obvious by
# inspection, and the sweep's simpler per-task backoff never couples to the
# drain's PM-epoch retry budget. The sweep runs AFTER the drain in main() (R6
# Test 17) so any ID the drain dispatched THIS tick is registered and counts as
# `pending` here — the shared cap holds across both.
#
# SCOPE is deliberately NARROWER than the PM rule: {infra, batch} only, NOT
# `agent-ok` analysis (which needs an explicit human opt-in + can be CPU-cost-
# bearing — kept on the deliberate PM path). `experiment`/`analysis`/`campaign`
# are excluded by the `kind in INFRA_DRAIN_KINDS` filter.


def _proposed_infra_sweep_enabled() -> bool:
    """Kill switch: False when ``EPM_DISABLE_PROPOSED_INFRA_SWEEP`` is set
    truthy ("1"/"true"/"yes", case-insensitive). Default enabled. Mirrors
    :func:`_infra_drain_enabled`."""
    raw = os.environ.get("EPM_DISABLE_PROPOSED_INFRA_SWEEP", "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _proposed_infra_sweep_backoff_s() -> float:
    """Retry-backoff window in seconds (env
    ``EPM_PROPOSED_INFRA_SWEEP_BACKOFF_S``; default
    :data:`PROPOSED_INFRA_SWEEP_BACKOFF_S_DEFAULT`). A malformed env value
    falls back to the default (a typo must not distort the budget)."""
    raw = os.environ.get("EPM_PROPOSED_INFRA_SWEEP_BACKOFF_S")
    if not raw:
        return PROPOSED_INFRA_SWEEP_BACKOFF_S_DEFAULT
    try:
        return float(raw)
    except ValueError:
        return PROPOSED_INFRA_SWEEP_BACKOFF_S_DEFAULT


def _proposed_infra_sweep_max_attempts() -> int:
    """Attempt cap before the sweep parks a repeatedly-failing task (env
    ``EPM_PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS``; default
    :data:`PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS_DEFAULT`). Malformed env value
    falls back to the default."""
    raw = os.environ.get("EPM_PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS")
    if not raw:
        return PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS_DEFAULT


def _proposed_infra_sweep_state_path() -> Path:
    """Path of the sweep's per-task attempt/backoff state file (resolved at
    call time so the test fixture's ``AUTONOMOUS_REGISTRY_DIR`` monkeypatch
    isolates it)."""
    return AUTONOMOUS_REGISTRY_DIR / PROPOSED_INFRA_SWEEP_STATE_BASENAME


def _load_proposed_infra_sweep_state() -> dict:
    """Read the sweep's attempt/backoff state (``{"attempts": {int: rec}}``;
    empty on absent/garbled). Reuses the infra-drain loader's defensive
    normalize (drop a non-int key / non-numeric ``last_attempt_ts``; cap a
    missing/garbled ``attempts`` count up to the attempt cap — fail toward NOT
    dispatching) against THIS state file, keeping one normalization rule for
    both paths."""
    return _load_attempt_state(
        _proposed_infra_sweep_state_path(),
        _proposed_infra_sweep_max_attempts(),
        log_prefix="proposed-infra-sweep",
    )


def _save_proposed_infra_sweep_state(state: dict) -> None:
    """Persist the sweep's attempt/backoff state atomically (temp + rename).
    Int keys serialize as strings; the loader normalizes them back."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _proposed_infra_sweep_state_path()
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(dest)


def _proposed_infra_candidates() -> list[int] | None:
    """Ripe-`proposed` infra/batch candidate ids, OLDEST-FIRST (ascending id is
    a safe proxy — the PM's urgency-first nuance is the PM's job; this sweep is
    the mechanical backstop). Built from EXACTLY one
    ``task.py list-by-status --status proposed --json`` subprocess, filtered to
    ``kind in INFRA_DRAIN_KINDS``.

    The ``--status proposed`` filter is the STRUCTURAL `on_hold`-exclusion
    mechanism (#690 M2): `on_hold` is a different status FOLDER, so a query
    restricted to `--status proposed` can never enumerate an `on_hold` task —
    pinned by the exact-argv Test 10. A row tagged
    :data:`_NEEDS_HUMAN_TAG` is ALSO excluded (#706): a /daily route-3 held
    judgment call is a tracked `proposed` infra task surfaced in the PM
    `Needs you` block for Thomas's call, NEVER auto-dispatched by this sweep.
    Returns ``None`` on any read/parse
    failure (the pass then skips this tick — fail toward NOT dispatching,
    mirroring :func:`_infra_drain_occupancy`'s fail-closed posture)."""
    try:
        res = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "list-by-status",
                "--status",
                "proposed",
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
    ids: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") not in INFRA_DRAIN_KINDS:
            continue
        # #706: a /daily route-3 held judgment call carries `needs-human` —
        # surfaced in the PM `Needs you` block, never auto-dispatched. The
        # `or []` guards legacy rows that predate the `tags` field (a
        # `row["tags"]` lookup would KeyError + crash the whole sweep).
        if _NEEDS_HUMAN_TAG in (row.get("tags") or []):
            continue
        tid = row.get("id")
        if isinstance(tid, int) and tid not in ids:
            ids.append(tid)
    ids.sort()  # oldest-first proxy
    return ids


def _proposed_infra_hold_reason(
    issue: int,
    holds: dict[int, str],
    predicate_statuses: dict[int, str | None],
) -> str | None:
    """The hold-gate verdict for ONE candidate against the PM queue's PARSED
    int-keyed ``holds`` map (#690 — reuses the EXISTING shared storage + the
    EXISTING :func:`_parse_predicate_hold` primitive; NO new on-task tag
    convention). Returns a skip reason string or ``None`` (no hold to honor →
    eligible so far):

      - ``issue`` NOT in ``holds`` (the ORPHAN case — filed with no PM in the
        loop, exactly the gap #690 closes) → ``None`` (eligible).
      - in ``holds`` for a ``predicate-<#N>-...`` reason → ``None`` iff #N is at
        a satisfied status (:data:`INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES`),
        else ``"held: predicate-<#N>"`` (#N still in flight).
      - in ``holds`` for ANY non-predicate reason
        (``credentials``/``spend``/``outward-facing``/... AND the bare
        mechanical ``cap``/``predicate`` deferral reasons) → ``"held: <reason>"``
        regardless. This "ANY non-predicate reason → SKIP" is intentionally
        OVER-CONSERVATIVE (it defers a candidate the PM might re-ripen on its
        next pass to a later tick) but NEVER over-dispatches — and a hold the
        PM wrote means the PM is actively managing that id, so deferring to the
        PM's nuanced pass is the safe direction (R2)."""
    reason = holds.get(issue)
    if reason is None:
        return None  # orphan — eligible
    blocking = _parse_predicate_hold(reason)
    if blocking is None:
        return f"held: {reason}"  # non-predicate user-park — never override
    if predicate_statuses.get(blocking) in INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES:
        return None  # predicate satisfied — eligible
    return f"held: predicate-{blocking}"  # blocker still in flight


def decide_proposed_infra_sweep(
    candidates: list[int],
    holds: dict[int, str],
    predicate_statuses: dict[int, str | None],
    statuses: dict[int, str | None],
    kinds: dict[int, str | None],
    registered: set[int],
    occupied_active: int,
    pending: int,
    attempts: dict[int, dict],
    now: float,
    cap: int = INFRA_DRAIN_CAP_DEFAULT,
    *,
    backoff_s: float = PROPOSED_INFRA_SWEEP_BACKOFF_S_DEFAULT,
    max_attempts: int = PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS_DEFAULT,
) -> tuple[list[int], list[tuple[int, str]]]:
    """Pure decision: ``(dispatch_ids_in_order, skipped [(id, reason)])`` —
    mirroring :func:`decide_infra_drain` so the cap/hold/registration matrix is
    falsifiable in isolation (#690 R4).

    ``candidates`` is the ripe-`proposed` infra/batch id list (oldest-first);
    ``holds`` the PM queue file's PARSED int-keyed map; ``predicate_statuses``
    the resolved status of each predicate's BLOCKING issue (``None`` =
    unreadable); ``registered`` the NON-STALE registrations among the
    candidates; ``occupied_active`` the count of kind-infra/batch tasks at
    :data:`INFRA_DRAIN_OCCUPIED_STATUSES`; ``pending`` the count of ALL
    non-stale registrations of still-`proposed` drain-kind tasks (the SHARED
    cap budget — see :func:`_infra_drain_pending`). ``free = max(0, cap -
    occupied_active - pending)``.

    Per-candidate guard order (every skip carries a reason string):

      1. hold gate (:func:`_proposed_infra_hold_reason`) — ``held: <reason>``
      2-4. :func:`_cheap_skip_reason` (already-registered / backoff /
           attempts-exhausted; ``holds`` passed EMPTY here because guard 1
           already owns hold-gating — the queue file is the hold source, not
           ``_cheap_skip_reason``'s membership check)
      5. status unreadable → ``"status-unreadable"``; status != ``proposed`` →
         ``"status-<status>"`` (the candidate list was built from a
         ``proposed`` query, but a status read here re-confirms it — a task can
         change status between the candidate query and this read)
      6. kind not in :data:`INFRA_DRAIN_KINDS` → ``"kind-<kind|unreadable>"``
         (defense in depth; the candidate query already filtered)
      7. no free slot → ``"cap-full"``
      8. else dispatch; one attempt per id per cycle
    """
    free = max(0, cap - occupied_active - pending)
    dispatch: list[int] = []
    skipped: list[tuple[int, str]] = []
    for i in candidates:
        held = _proposed_infra_hold_reason(i, holds, predicate_statuses)
        if held is not None:
            skipped.append((i, held))
            continue
        # _cheap_skip_reason owns already-registered + backoff/attempts; holds
        # is passed EMPTY because the queue-file hold gate (guard 1) is the
        # authoritative hold source for the sweep (the cheap helper's `holds`
        # membership check is the DRAIN's mechanism, not the sweep's).
        reason = _cheap_skip_reason(
            i,
            {},
            registered,
            attempts,
            now,
            None,
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


def _proposed_infra_sweep_record_attempt(
    attempts: dict[int, dict], issue: int, now: float, ok: bool
) -> None:
    """Record one sweep dispatch ATTEMPT (success or failure both count, so a
    failing spawn can't tight-loop — the backoff window binds next tick either
    way). Unlike the drain's epoch-reset, the sweep has no ``updated_ts``: the
    count simply increments until the attempt cap, and the whole entry is
    pruned when the task leaves `proposed` (a PM rewrite / repromotion / status
    change clears it naturally)."""
    rec = attempts.get(issue) or {}
    count = int(rec.get("attempts", 0)) + 1
    attempts[issue] = {
        "attempts": count,
        "last_attempt_ts": now,
        "last_result": "dispatched" if ok else "spawn-failed",
    }


def _proposed_infra_sweep_prune_save(state: dict, candidates: list[int], dry_run: bool) -> None:
    """Prune state entries whose task left the `proposed` candidate set (it
    dispatched, was repromoted, or changed status), then persist atomically.
    No-op under dry-run (mirror the drain's dry-run discipline: no state
    write)."""
    if dry_run:
        return
    keep = set(candidates)
    state["attempts"] = {i: rec for i, rec in state["attempts"].items() if i in keep}
    _save_proposed_infra_sweep_state(state)


def proposed_infra_sweep_pass(
    dry_run: bool,
    now: float | None = None,
    *,
    daemon_reachable: bool | None = None,
) -> None:
    """Always-on backstop sweep: dispatch ripe ORPHANED `proposed` infra/batch
    tasks the PM never queued (#690). REUSES the infra-drain leaf primitives;
    BUILDS its own candidate set from ``list-by-status --status proposed``;
    consults the PM queue's ``holds`` map ONLY to honor holds. Daemon-gated like
    every spawning pass; shares the cap with the drain (it runs AFTER the drain
    in main(), so the drain's fresh registrations count as ``pending`` here)."""
    if not _proposed_infra_sweep_enabled():
        print("proposed-infra-sweep: disabled via EPM_DISABLE_PROPOSED_INFRA_SWEEP; skipping")
        return
    if daemon_reachable is None:
        daemon_reachable = _daemon_reachable()
    if not daemon_reachable:
        print(
            "proposed-infra-sweep: Happy daemon unreachable; skipping (spawn needs the daemon RPC)"
        )
        return
    now = now if now is not None else time.time()

    candidates = _proposed_infra_candidates()
    if candidates is None:
        print(
            "proposed-infra-sweep: `list-by-status --status proposed` read FAILED; "
            "skipping this tick (fail toward NOT dispatching)"
        )
        return
    if not candidates:
        print("proposed-infra-sweep: no ripe proposed infra/batch candidates; nothing to do")
        return

    # Holds gate: read the PM queue file's PARSED int-keyed `holds` map directly
    # (the SAME storage + the SAME predicate primitives the drain uses). A
    # missing / empty / invalid / torn-write queue file fails SOFT to "no
    # holds" — orphans still dispatch; no candidate is ever blocked by an
    # unreadable queue, because the orphan-eligible default is "no hold". A
    # corrupt (None-parsing) read is treated as "no usable holds map this tick"
    # (NOT as un-holding a previously-held candidate — a held candidate is held
    # by the map, and when the map is absent it is an un-held orphan only
    # because no usable hold exists, never because a held entry was dropped from
    # an otherwise-valid map; R1).
    queue = _infra_drain_read_queue()
    holds: dict[int, str] = queue["holds"] if queue is not None else {}

    # Only predicate holds among the CANDIDATE set cost a blocking-status read;
    # non-predicate holds and non-held candidates short-circuit with zero extra
    # subprocesses.
    predicate_blockers = {
        b
        for i in candidates
        if (r := holds.get(i)) is not None and (b := _parse_predicate_hold(r)) is not None
    }
    predicate_statuses = {b: _task_status_kind(b)[0] for b in sorted(predicate_blockers)}

    state = _load_proposed_infra_sweep_state()
    attempts: dict[int, dict] = state["attempts"]
    backoff_s = _proposed_infra_sweep_backoff_s()
    max_attempts = _proposed_infra_sweep_max_attempts()

    # ONE registry read: the staleness/pending decision parses from this
    # snapshot, and the pre-spawn re-check (inside _dispatch_infra_drain)
    # compares against the same bytes.
    reg_snapshot = _infra_drain_reg_snapshot()
    regs = _infra_drain_registrations(reg_snapshot)
    raw_registered = set(regs) & set(candidates)

    # Reuse the drain's signal fetch (per-id status/kind + stale set + the
    # WIDENED pending count over ALL non-stale registrations). `_infra_drain_signals`
    # already skips held queue ids; here `holds` gating is the sweep's own, so
    # pass an empty holds so every candidate's status/kind is fetched (a held
    # candidate is dropped by the hold gate in decide_proposed_infra_sweep
    # before the status read matters).
    status_kind, stale, pending = _infra_drain_signals(candidates, {}, regs, now)
    registered_nonstale = raw_registered - stale

    occupying = _infra_drain_occupancy()
    if occupying is None:
        print(
            "proposed-infra-sweep: occupancy read FAILED for at least one status; "
            "skipping dispatch this tick (fail-closed: a partial count would "
            "under-count and over-dispatch past the cap)"
        )
        _proposed_infra_sweep_prune_save(state, candidates, dry_run)
        return
    occupied_active = len(occupying)
    cap = INFRA_DRAIN_CAP_DEFAULT
    statuses = {i: status_kind.get(i, (None, None))[0] for i in candidates}
    kinds = {i: status_kind.get(i, (None, None))[1] for i in candidates}

    dispatch, skipped = decide_proposed_infra_sweep(
        candidates,
        holds,
        predicate_statuses,
        statuses,
        kinds,
        registered_nonstale,
        occupied_active,
        pending,
        attempts,
        now,
        cap,
        backoff_s=backoff_s,
        max_attempts=max_attempts,
    )

    marker_fresh_s = _proposed_infra_sweep_marker_fresh_s()
    dispatched = 0
    for issue in dispatch:
        # #843 M1 advisory pre-check at the CALLER loop (same contract as the
        # drain loop's): a fresh lease -> loud skip, NO attempt recorded.
        held_lease = dispatch_lease_fresh(issue, now)
        if held_lease is not None:
            print(
                f"  PROPOSED-INFRA-SWEEP SKIP issue #{issue} (dispatch-lease held, "
                f"{dispatch_lease_desc(held_lease, now)})"
            )
            continue
        # #843 M3: a dispatch-sentinel marker younger than one watcher cadence
        # means SOME dispatcher fired within the current/previous tick — skip
        # this candidate this tick (no attempt recorded — a marker skip is not
        # a spawn attempt; the safe direction, corrected next tick). Post-M1
        # this is the lease-file-loss backstop + observability.
        marker_age = _recent_dispatch_marker_age_s(_task_events(issue), now)
        if marker_age is not None and marker_age < marker_fresh_s:
            print(
                f"  PROPOSED-INFRA-SWEEP SKIP issue #{issue} "
                f"(recent-dispatch-marker {marker_age:.0f}s < {marker_fresh_s:.0f}s)"
            )
            continue
        slot_desc = f"slot {min(occupied_active + pending + dispatched + 1, cap)}/{cap}"
        result = _dispatch_infra_drain(issue, slot_desc, dry_run, reg_snapshot=reg_snapshot)
        if result == "suppressed":
            # #843 M1b: duplicate-suppression no-op — book nothing.
            continue
        if not dry_run:
            _proposed_infra_sweep_record_attempt(attempts, issue, now, result == "spawned")
        if result == "spawned":
            dispatched += 1
            _post_progress_marker(
                issue,
                f"{_PROPOSED_INFRA_SWEEP_NOTE_SENTINEL} watcher auto-dispatched ripe "
                f"proposed infra task (no PM queue entry)",
                dry_run,
                label="proposed-infra-sweep",
            )
    for issue, reason in skipped:
        print(f"  PROPOSED-INFRA-SWEEP SKIP issue #{issue} ({reason})")
    summary = (
        f"proposed-infra-sweep: candidates={len(candidates)} "
        f"occupied={occupied_active}(+{pending} pending) cap={cap} "
        f"dispatched={dispatched} skipped={len(skipped)}"
    )
    if any(reason == "cap-full" for _i, reason in skipped):
        summary += f" occupying={sorted(occupying)}"
    print(summary)
    _proposed_infra_sweep_prune_save(state, candidates, dry_run)


# ─── capacity-retry pass (re-drive a transient-infra `blocked` task) ─────────
#
# WHY THIS PASS EXISTS (incident #642, 2026-06-16). The crash-recovery
# `decide()` gate treats EVERY `blocked` task as PARK ("keep", never respawn) —
# correct for a DELIBERATE halt awaiting human input (a `failure_class: code` /
# `data` block, or a factual question only the user can answer), but WRONG for
# the narrow subclass where the block is purely transient infra capacity: the
# auto-router exhausted every lane (`epm:failure v1` with `failure_class: infra`
# AND `reason: no_compute_available`), the task's CODE is ready, and capacity
# frees up later. The failure marker itself self-flags "Retry on re-invocation"
# (the `epm:status-changed` note #642 posted) — but nothing RE-INVOKED it, so a
# human (the PM session) had to notice the freed quota and stop+respawn by hand,
# ~8h late.
#
# WHY THERE IS NO WATCHER-SIDE CAPACITY PRE-CHECK. The original candidate asked
# for a precheck before re-driving, so we never respawn into a still-full lane.
# The right design is the OPPOSITE of a duplicate precheck here: the `/issue`
# launch path ALREADY runs the authoritative capacity gate. Re-driving via
# `spawn-issue --auto` re-enters `/issue` -> Step 6 backend dispatch -> the
# router's GCP regional-quota headroom pre-check (`backends/router.py`
# `_skip_gcp_lane_no_headroom`, `backends/gcp.py` `preflight_quota_headroom`),
# which on insufficient headroom SMART-SKIPS the GCP lane WITHOUT burning a
# daily attempt and WITHOUT any GPU spend (#608), falls through to the free
# SLURM lanes, and — if those are also full — simply re-blocks on
# `no_compute_available` at ZERO GPU cost. So "respawn blind" is NOT expensive:
# the only cost of a re-drive that re-blocks is a no-GPU Happy session spin-up.
# Re-implementing a weaker copy of the router's quota logic inside this 10-min
# fail-soft watcher would (a) duplicate + risk drifting from the authoritative
# gate, (b) deeply couple the watcher to RunSpec/GcpConfig/GcloudRunner router
# internals, and (c) add live `gcloud` subprocess calls to every tick. The
# churn the candidate (rightly) worries about is bounded instead by the two
# guards below — exponential-style backoff from the block timestamp + a
# per-UTC-day retry cap — exactly the candidate's stated fallback ("if a clean
# precheck is infeasible, prefer backoff + a tight day-cap over respawning
# blind"). The re-driven `--auto` session re-enters `/issue` and enforces its
# OWN plan-approval GPU-hour cap; this pass opens NO new spend path.
#
# SCOPE (do NOT broaden): ONLY a `blocked` task whose LATEST `epm:failure`
# marker is `failure_class: infra` with a reason in the conservative
# :data:`TRANSIENT_CAPACITY_REASONS` allowlist. EVERY other `blocked` task
# (real halts: code/data failures, factual questions, a non-capacity infra
# reason like `codex-companion-probe-error`) stays PARKED and is never touched
# — `decide()`'s PARK semantics for them are unchanged.

# Reasons (matched against the latest `epm:failure` marker's `reason:` field)
# that classify a `blocked` task as transient-infra-capacity, hence retriable.
# Deliberately a SINGLETON for now — the only demonstrated transient-capacity
# block class is the auto-router's all-lanes-exhausted verdict. Widen ONLY with
# a demonstrated, genuinely-transient, retry-on-re-invocation reason; a
# permanent infra fault (auth, config, a code bug surfaced as infra) must NOT
# join this set, or the pass would hot-retry an unrecoverable block until the
# day-cap.
TRANSIENT_CAPACITY_REASONS: frozenset[str] = frozenset({"no_compute_available"})

# Filename prefix for the per-issue capacity-retry state file at
# ``~/.eps-autonomous/capacity-retry-<N>.json``. Mirrors the orphan / stalled /
# pod-safety state-file layout; reaped by the generalized GC.
CAPACITY_RETRY_STATE_PREFIX = "capacity-retry-"

# Backoff window (seconds) measured from the BLOCK timestamp (the latest
# `epm:failure` marker's ts) AND from the last retry attempt — a re-drive fires
# only once the newer of those two is older than this window, so capacity has
# time to free up between attempts and the pass cannot tight-loop. 1h matches
# the infra-drain backoff default.
CAPACITY_RETRY_BACKOFF_S_DEFAULT = 3600.0

# Maximum re-drive ATTEMPTS (successes AND failures both count, so a
# deterministically re-blocking re-drive can't burn the whole day) per task per
# UTC day. Mirrors the orphan daily cap.
CAPACITY_RETRY_MAX_PER_DAY_DEFAULT = 4


def _capacity_retry_enabled() -> bool:
    """Kill switch: False when ``EPM_DISABLE_CAPACITY_RETRY`` is set truthy
    ("1"/"true"/"yes", case-insensitive). Default enabled. Mirrors
    :func:`_infra_drain_enabled`."""
    raw = os.environ.get("EPM_DISABLE_CAPACITY_RETRY", "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _capacity_retry_backoff_s() -> float:
    """Retry-backoff window in seconds (env ``EPM_CAPACITY_RETRY_BACKOFF_S``;
    default :data:`CAPACITY_RETRY_BACKOFF_S_DEFAULT`). A malformed env value
    falls back to the default — a typo must not collapse the backoff."""
    raw = os.environ.get("EPM_CAPACITY_RETRY_BACKOFF_S")
    if not raw:
        return CAPACITY_RETRY_BACKOFF_S_DEFAULT
    try:
        return float(raw)
    except ValueError:
        return CAPACITY_RETRY_BACKOFF_S_DEFAULT


def _capacity_retry_max_per_day() -> int:
    """Daily per-task re-drive cap (env ``EPM_CAPACITY_RETRY_PER_DAY``; default
    :data:`CAPACITY_RETRY_MAX_PER_DAY_DEFAULT`). Malformed value falls back to
    the default."""
    raw = os.environ.get("EPM_CAPACITY_RETRY_PER_DAY")
    if not raw:
        return CAPACITY_RETRY_MAX_PER_DAY_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return CAPACITY_RETRY_MAX_PER_DAY_DEFAULT


def _latest_failure_marker(events: list[dict]) -> dict | None:
    """The most recent ``epm:failure`` event in ``events`` (chronological
    order), or ``None``. ``task.py list-markers`` returns events oldest-first,
    so the last match is the latest."""
    latest: dict | None = None
    for ev in events:
        if isinstance(ev, dict) and str(ev.get("kind", "")).startswith("epm:failure"):
            latest = ev
    return latest


def _parse_failure_fields(note: str | None) -> tuple[str | None, str | None]:
    """Extract ``(failure_class, reason)`` from a failure-marker ``note``.

    The producer shapes vary (verified against real #642 markers):

    * field-per-line — ``failure_class: infra\\nreason: no_compute_available``
    * inline on one line — ``... failure_class: infra reason: codex-...``

    so both fields are pulled with a tolerant token scan rather than a
    line-anchored parse: split on whitespace, and for each ``failure_class:`` /
    ``reason:`` token (or the ``key: value`` glued form) take the next token as
    the value. The value is taken up to the first whitespace, so a trailing
    sentence on the same line (``reason: foo. Proceeding ...``) yields ``foo``
    (a trailing ``.``/``,`` is stripped). Returns ``(None, None)`` for a
    non-string / fieldless note — fail toward NOT retrying."""
    if not isinstance(note, str) or not note:
        return (None, None)
    tokens = note.replace("\n", " ").split()
    out: dict[str, str] = {}
    for key in ("failure_class", "reason"):
        for i, tok in enumerate(tokens):
            val: str | None = None
            if tok == f"{key}:":
                val = tokens[i + 1] if i + 1 < len(tokens) else None
            elif tok.startswith(f"{key}:"):
                val = tok[len(key) + 1 :] or (tokens[i + 1] if i + 1 < len(tokens) else None)
            if val:
                out[key] = val.strip().rstrip(".,;")
                break
    return (out.get("failure_class"), out.get("reason"))


def _is_transient_capacity_block(events: list[dict]) -> tuple[bool, str | None, float | None]:
    """Return ``(retriable, reason, block_ts)`` for a `blocked` task.

    ``retriable`` is True ONLY when the LATEST ``epm:failure`` marker is
    ``failure_class: infra`` with a ``reason`` in
    :data:`TRANSIENT_CAPACITY_REASONS`. ``block_ts`` is that marker's epoch ts
    (for the backoff window), or ``None`` if unparseable. Conservative:
    anything other than a clean transient-capacity match yields
    ``(False, reason, ...)`` so the task stays parked."""
    marker = _latest_failure_marker(events)
    if marker is None:
        return (False, None, None)
    failure_class, reason = _parse_failure_fields(marker.get("note"))
    block_ts = _parse_event_ts(marker.get("ts"))
    retriable = failure_class == "infra" and reason in TRANSIENT_CAPACITY_REASONS
    return (retriable, reason, block_ts)


def decide_capacity_retry(
    status: str | None,
    retriable: bool,
    block_ts: float | None,
    last_attempt_ts: float | None,
    retries_today: int,
    now: float,
    *,
    backoff_s: float = CAPACITY_RETRY_BACKOFF_S_DEFAULT,
    max_per_day: int = CAPACITY_RETRY_MAX_PER_DAY_DEFAULT,
) -> str:
    """Pure decision for the capacity-retry pass. Returns one of:

    * ``"skip"`` — not a transient-capacity block, not `blocked`, or still
      inside the backoff window. No action, no marker.
    * ``"redrive"`` — re-drive via ``spawn-issue --auto`` now.
    * ``"exhausted"`` — retriable + out of backoff, but the daily cap is spent;
      post the one-time exhausted alert and otherwise leave it parked.

    Safety: a non-`blocked` status or a non-retriable block ALWAYS yields
    ``"skip"`` — every deliberate halt stays parked. The backoff binds on the
    NEWER of (block_ts, last_attempt_ts): we wait ``backoff_s`` after the block
    AND after each attempt, so capacity has time to free up and the pass can't
    tight-loop. A missing block_ts (unparseable marker ts) does NOT block a
    retry — the last-attempt backoff still binds — so a garbled ts can't
    permanently freeze recovery."""
    if status != "blocked" or not retriable:
        return "skip"
    # Backoff: bind on the newer of block_ts / last_attempt_ts (each is
    # optional). None means "no constraint from that source".
    refs = [t for t in (block_ts, last_attempt_ts) if t is not None]
    if refs and (now - max(refs)) < backoff_s:
        return "skip"
    if retries_today >= max_per_day:
        return "exhausted"
    return "redrive"


def _capacity_retry_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{CAPACITY_RETRY_STATE_PREFIX}{issue}.json"


def _load_capacity_retry_state(issue: int) -> dict:
    """Read the per-issue retry state (``{retry_day, retries_today,
    last_attempt_ts, alerted_day}``); ``{}`` on absent/garbled. Mirrors
    :func:`_load_orphan_state`."""
    path = _capacity_retry_state_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_capacity_retry_state(issue: int, state: dict, dry_run: bool) -> None:
    """Persist the per-issue retry state atomically (temp + rename). No-op
    under dry_run."""
    if dry_run:
        return
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _capacity_retry_state_path(issue)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(dest)


def _redrive_capacity_retry(issue: int, dry_run: bool) -> str:
    """Re-drive the autonomous session for a transient-infra `blocked` task via
    ``spawn_session.py spawn-issue --issue <N> --auto`` (the plain command; the
    `--auto` session re-enters `/issue` which re-runs the backend router's own
    capacity pre-check + enforces its plan-approval GPU cap). Returns the
    #843 M1b tri-state ``"spawned" | "suppressed" | "failed"`` (see
    :func:`_respawn`; ``"suppressed"`` must NOT consume the per-day retry
    budget, and ALSO covers the #1027 auth-outage gate); honours dry_run
    (logs, never spawns, returns ``"failed"``)."""
    if _auth_outage_spawn_gate(issue, "capacity-retry", dry_run=dry_run) is not None:
        print(f"  CAPACITY-RETRY issue #{issue}: suppressed — auth-outage episode active")
        return "suppressed"
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto",
    ]  # fmt: skip
    if dry_run:
        print(f"  [dry-run] would capacity-retry: {' '.join(cmd)}")
        return "failed"  # dry-run: nothing spawned
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(
            f"  CAPACITY-RETRY DISPATCH FAILED issue #{issue}: {res.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return "failed"
    _forward_marker_child_stderr(res, "spawn_session spawn-issue (capacity-retry)")
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    if spawn_output_suppressed(res.stdout) is not None:
        print(
            f"  CAPACITY-RETRY issue #{issue}: suppressed — not re-driven "
            f"(lease/collision): {first_line}"
        )
        return "suppressed"
    print(f"  CAPACITY-RETRIED issue #{issue} (transient-infra block re-driven): {first_line}")
    _auth_outage_record_spawn(issue, "capacity-retry", None)
    return "spawned"


def _process_capacity_retry(
    issue: int,
    now: float,
    day_key: str,
    dry_run: bool,
    *,
    backoff_s: float,
    max_per_day: int,
) -> None:
    """Apply one `blocked` task's capacity-retry decision (gather signals ->
    :func:`decide_capacity_retry` -> act). Honours dry_run."""
    events = _task_events(issue)
    retriable, reason, block_ts = _is_transient_capacity_block(events)
    if not retriable:
        # Not a transient-capacity block — a deliberate halt. Leave it parked
        # and don't even keep state (cheap to recompute next tick).
        return

    state = _load_capacity_retry_state(issue)
    retries_today = state.get("retries_today", 0) if state.get("retry_day") == day_key else 0
    if not isinstance(retries_today, int) or retries_today < 0:
        retries_today = 0
    last_attempt_ts = state.get("last_attempt_ts")
    if isinstance(last_attempt_ts, bool) or not isinstance(last_attempt_ts, int | float):
        last_attempt_ts = None
    alerted_day = state.get("alerted_day")

    action = decide_capacity_retry(
        "blocked",
        retriable,
        block_ts,
        last_attempt_ts,
        retries_today,
        now,
        backoff_s=backoff_s,
        max_per_day=max_per_day,
    )
    block_age = f"{(now - block_ts) / 3600:.1f}h" if block_ts is not None else "unknown"
    print(
        f"  issue #{issue}: blocked on transient infra (reason={reason}) "
        f"block_age={block_age} retries_today={retries_today}/{max_per_day} action={action}"
    )

    if action == "skip":
        return
    if action == "redrive":
        spawn_result = _redrive_capacity_retry(issue, dry_run)
        if spawn_result == "suppressed":
            # #843 M1b: a duplicate dispatch was suppressed (lease held /
            # registration collision) — a session is driving this issue.
            # Book NOTHING: no retry-budget consumption, no last_attempt_ts,
            # no marker; the next tick re-evaluates.
            return
        ok = spawn_result == "spawned"
        new_state = {
            "retry_day": day_key,
            # Count the ATTEMPT regardless of spawn success so a failing spawn
            # can't hot-loop past the daily cap (the backoff also binds next
            # tick via last_attempt_ts).
            "retries_today": retries_today + 1,
            "last_attempt_ts": now,
            "alerted_day": None,
        }
        _save_capacity_retry_state(issue, new_state, dry_run)
        if ok:
            _post_progress_marker(
                issue,
                f"{_CAPACITY_RETRY_NOTE_SENTINEL} task was blocked on transient "
                f"infra (reason={reason}, block_age={block_age}); auto-re-drove "
                f"via spawn-issue --auto (attempt {retries_today + 1}/{max_per_day} "
                f"today). The /issue launch re-runs the backend capacity pre-check; "
                f"if every lane is still full it re-blocks at zero GPU cost.",
                dry_run,
                label="capacity-retry",
            )
        return
    # action == "exhausted": one-time loud marker per UTC day.
    print(
        f"  CAPACITY-RETRY EXHAUSTED issue #{issue}: blocked on {reason}, "
        f"daily re-drive cap spent ({retries_today}/{max_per_day})",
        file=sys.stderr,
    )
    if alerted_day != day_key:
        _post_progress_marker(
            issue,
            f"{_CAPACITY_RETRY_EXHAUSTED_NOTE_SENTINEL} task remains blocked on "
            f"transient infra (reason={reason}); daily auto-re-drive cap exhausted "
            f"({retries_today}/{max_per_day}). Manual recovery: uv run python "
            f"scripts/spawn_session.py spawn-issue --issue {issue} --auto",
            dry_run,
            label="capacity-retry-exhausted",
        )
    _save_capacity_retry_state(
        issue,
        {
            "retry_day": day_key,
            "retries_today": retries_today,
            "last_attempt_ts": last_attempt_ts,
            "alerted_day": day_key,
        },
        dry_run,
    )


def capacity_retry_pass(
    dry_run: bool,
    now: float | None = None,
    *,
    daemon_reachable: bool | None = None,
) -> None:
    """Re-drive `blocked`-on-transient-infra tasks (incident #642). Scans every
    `blocked` task; for each whose LATEST ``epm:failure`` is
    ``failure_class: infra`` + a :data:`TRANSIENT_CAPACITY_REASONS` reason,
    re-drives it via ``spawn-issue --auto`` once backoff clears, capped per UTC
    day. Every OTHER `blocked` task is untouched. Daemon-gated like every
    spawning pass (spawn POSTs to the Happy daemon RPC). The module-level WHY
    block above documents the no-watcher-side-precheck design."""
    if not _capacity_retry_enabled():
        print("capacity-retry: disabled via EPM_DISABLE_CAPACITY_RETRY; skipping")
        return
    if daemon_reachable is None:
        daemon_reachable = _daemon_reachable()
    if not daemon_reachable:
        print("capacity-retry: Happy daemon unreachable; skipping (re-drive needs the daemon RPC)")
        return
    now = now if now is not None else time.time()
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    backoff_s = _capacity_retry_backoff_s()
    max_per_day = _capacity_retry_max_per_day()
    blocked = _blocked_issue_ids()
    if not blocked:
        print("capacity-retry: no blocked tasks")
        return
    print(f"capacity-retry: scanning {len(blocked)} blocked task(s)")
    for issue in blocked:
        _process_capacity_retry(
            issue, now, day_key, dry_run, backoff_s=backoff_s, max_per_day=max_per_day
        )


# ─── stale-blocked flag pass (task #1021, the #742 incident class) ───────────
#
# A crash-fix relaunch that succeeds on a task an earlier failed round parked
# at `blocked` leaves the status stale: the run is healthy (fresh
# `epm:run-launched` + ongoing progress ticks) while the folder says `blocked`
# (#742 ran healthy ~35h at status `blocked`, 2026-07-01→07-02). The
# orchestrator-side fix is the SKILL.md "A successful relaunch also reconciles
# a stale `blocked`" rule; this pass is the watcher-side BACKSTOP: FLAG-ONLY —
# a deduped `epm:progress` marker + a sidecar row + one Telegram digest line
# per launch episode. It NEVER mutates status (false alert cheap, false flip
# dangerous — the same conservative posture as the pod-safety alerts).
# Daemon-INDEPENDENT: it spawns nothing (marker posts go via the task.py
# subprocess).


def _stale_blocked_flag_enabled() -> bool:
    """Kill switch: False when ``EPM_DISABLE_STALE_BLOCKED_FLAG`` is set
    truthy ("1"/"true"/"yes", case-insensitive). Default enabled. Mirrors
    :func:`_capacity_retry_enabled`."""
    raw = os.environ.get("EPM_DISABLE_STALE_BLOCKED_FLAG", "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _stale_blocked_fresh_s() -> float:
    """Post-launch progress freshness window in seconds (env
    ``EPM_STALE_BLOCKED_PROGRESS_FRESH_S``; default
    :data:`STALE_BLOCKED_PROGRESS_FRESH_S_DEFAULT`). A malformed or
    non-positive env value falls back to the default — a typo must not
    collapse (or explode) the window."""
    raw = os.environ.get("EPM_STALE_BLOCKED_PROGRESS_FRESH_S")
    if not raw:
        return float(STALE_BLOCKED_PROGRESS_FRESH_S_DEFAULT)
    try:
        parsed = float(raw)
    except ValueError:
        return float(STALE_BLOCKED_PROGRESS_FRESH_S_DEFAULT)
    if parsed <= 0:
        return float(STALE_BLOCKED_PROGRESS_FRESH_S_DEFAULT)
    return parsed


def decide_stale_blocked_flag(
    status: str | None,
    run_launched_ts: float | None,
    blocked_since_ts: float | None,
    progress_ts: float | None,
    now: float,
    *,
    fresh_window_s: float = STALE_BLOCKED_PROGRESS_FRESH_S_DEFAULT,
) -> bool:
    """True iff a ``blocked`` task shows a live healthy run: an
    ``epm:run-launched`` NEWER than the transition into ``blocked``, plus
    real (non-watcher, non-deliberate-stop) progress AT OR AFTER the launch
    and within ``fresh_window_s``.

    The ``progress_ts >= run_launched_ts`` conjunct makes the liveness leg
    genuinely POST-LAUNCH: it excludes the block-transition
    ``epm:status-changed`` marker (which is in :data:`_PROGRESS_KINDS`) by
    construction (block < launch by the ordering conjunct), at the cost of
    one poll-tick flag delay. The launch-newer-than-block ordering is what
    keeps a deliberately-blocked task quiet: the normal order is fail ->
    block (launch older than block -> skip); only a launch AFTER the block —
    the exact #742 anomaly — flags. EVERY missing signal returns False
    (fail toward silence)."""
    return (
        status == "blocked"
        and run_launched_ts is not None
        and blocked_since_ts is not None
        and run_launched_ts > blocked_since_ts
        and progress_ts is not None
        and progress_ts >= run_launched_ts
        and (now - progress_ts) <= fresh_window_s
    )


def _stale_blocked_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{STALE_BLOCKED_STATE_PREFIX}{issue}.json"


def _load_stale_blocked_state(issue: int) -> dict:
    """Read the per-issue stale-blocked dedup state
    (``{flagged_run_launched_ts, alerted_ts}``); ``{}`` on absent/garbled.
    Mirrors :func:`_load_capacity_retry_state`."""
    path = _stale_blocked_state_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_stale_blocked_state(issue: int, state: dict, dry_run: bool) -> None:
    """Persist the per-issue dedup state atomically (temp + rename). No-op
    under dry_run. Mirrors :func:`_save_capacity_retry_state`."""
    if dry_run:
        return
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _stale_blocked_state_path(issue)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(dest)


def _append_stale_blocked_event(payload: dict, dry_run: bool) -> None:
    """Durable trace for stale-blocked flags — one JSON line per flag in
    ``~/.eps-autonomous/stale-blocked-events.jsonl`` (same shape + role as
    the stale-registration events file; the ``.jsonl`` suffix keeps it out
    of the GC's ``stale-blocked-*.json`` glob). The per-task marker is the
    primary record; this file survives a task folder move. Fail-soft."""
    dest = AUTONOMOUS_REGISTRY_DIR / "stale-blocked-events.jsonl"
    line = json.dumps(payload)
    if dry_run:
        print(f"  [dry-run] would append stale-blocked event to {dest}")
        return
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as e:
        print(f"  WARNING: appending stale-blocked event failed: {e}", file=sys.stderr)


def _process_stale_blocked(issue: int, now: float, dry_run: bool, *, fresh_window_s: float) -> None:
    """Evaluate ONE `blocked` task (gather signals ->
    :func:`decide_stale_blocked_flag` -> flag). Honours dry_run; NEVER
    mutates status (flag-only by design). ``blocked_since_ts`` is the latest
    ``epm:status-changed`` ts — valid as "the transition into blocked"
    because the caller scanned ``_blocked_issue_ids()`` (the same argument
    the ``_DONE_TRANSITION_KINDS`` docstring makes)."""
    events = _task_events(issue)
    run_ts = _latest_event_ts(events, frozenset({_RUN_LAUNCHED_KIND}))
    blocked_ts = _latest_event_ts(events, frozenset({"epm:status-changed"}))
    progress_ts = _latest_progress_ts(events)
    if not decide_stale_blocked_flag(
        "blocked", run_ts, blocked_ts, progress_ts, now, fresh_window_s=fresh_window_s
    ):
        return
    state = _load_stale_blocked_state(issue)
    if state.get("flagged_run_launched_ts") == run_ts:
        return  # dedup: one alert per launch episode; a NEWER launch re-alerts
    launch_iso = _triage_observer_iso_z(run_ts)  # shared events.jsonl ISO-Z shape
    blocked_iso = _triage_observer_iso_z(blocked_ts)
    progress_age_min = (now - progress_ts) / 60.0
    print(
        f"  STALE-BLOCKED FLAG issue #{issue}: launch {launch_iso} > block "
        f"{blocked_iso}, post-launch progress {progress_age_min:.0f}m ago"
    )
    _append_stale_blocked_event(
        {
            "ts": _triage_observer_iso_z(now),
            "issue": issue,
            "run_launched_ts": run_ts,
            "blocked_since_ts": blocked_ts,
            "progress_age_s": now - progress_ts,
            "action": "flagged",
            "dry_run": dry_run,
        },
        dry_run,
    )
    _post_progress_marker(
        issue,
        f"{_STALE_BLOCKED_FLAG_NOTE_SENTINEL} status=blocked but a live healthy "
        f"run is present: epm:run-launched {launch_iso} is NEWER than the blocked "
        f"transition {blocked_iso}, and post-launch progress landed "
        f"{progress_age_min:.0f} min ago. Likely a stale blocked from an earlier "
        f"failed round (#742 class). If the relaunch is legitimate, reconcile "
        f"with: uv run python scripts/task.py set-status {issue} running --note "
        f"'relaunch succeeded; clearing stale blocked'. FLAG-ONLY: the watcher "
        f"never flips status.",
        dry_run,
        label="stale-blocked-flag",
    )
    _telegram_push(
        f"EPS #{issue}: status=blocked but run alive (launch {launch_iso} > "
        f"block {blocked_iso}, progress {progress_age_min:.0f}m ago) — likely "
        f"stale blocked; reconcile via task.py set-status {issue} running",
        dry_run,
    )
    _save_stale_blocked_state(
        issue, {"flagged_run_launched_ts": run_ts, "alerted_ts": now}, dry_run
    )


def stale_blocked_flag_pass(dry_run: bool, now: float | None = None) -> None:
    """FLAG (never flip) `blocked` tasks whose events show a live healthy run
    (task #1021, incident #742). Scans every `blocked` task; for each where an
    ``epm:run-launched`` is NEWER than the transition into `blocked` AND fresh
    POST-LAUNCH progress exists, posts one deduped-per-launch-episode flag
    (marker + sidecar row + Telegram digest line). Deliberately flag-only —
    the orchestrator's own-relaunch reconcile rule (SKILL.md "A successful
    relaunch also reconciles a stale `blocked`") or a human flips the status
    on this evidence. Daemon-INDEPENDENT (no spawns; marker posts go via the
    task.py subprocess)."""
    if not _stale_blocked_flag_enabled():
        print("stale-blocked-flag: disabled via EPM_DISABLE_STALE_BLOCKED_FLAG; skipping")
        return
    now = now if now is not None else time.time()
    fresh_window_s = _stale_blocked_fresh_s()
    blocked = _blocked_issue_ids()
    if not blocked:
        print("stale-blocked-flag: no blocked tasks")
        return
    print(f"stale-blocked-flag: scanning {len(blocked)} blocked task(s)")
    for issue in blocked:
        _process_stale_blocked(issue, now, dry_run, fresh_window_s=fresh_window_s)


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
    # Capacity-retry per-issue state (== CAPACITY_RETRY_STATE_PREFIX). The pass
    # only ever touches `blocked` tasks; this is the backstop for a task that
    # left `blocked` for a terminal status without the pass clearing its file.
    # NOTE: TERMINAL_FOR_GC deliberately EXCLUDES `blocked` (the GC must never
    # reap a live retry episode's state — that would reset retries_today every
    # tick and the daily cap could never bind).
    (CAPACITY_RETRY_STATE_PREFIX, ""),
    # Stale-blocked flag per-issue dedup state (== STALE_BLOCKED_STATE_PREFIX,
    # task #1021). Same contract as capacity-retry above: TERMINAL_FOR_GC
    # deliberately EXCLUDES `blocked`, so a live episode's dedup state is never
    # reaped mid-episode (that would re-alert every tick); reaping fires only
    # once the task reaches `completed`/`archived`. The sidecar
    # `stale-blocked-events.jsonl` is outside the `*.json` glob by suffix.
    (STALE_BLOCKED_STATE_PREFIX, ""),
    # Boot-death day-cap state (== BOOT_DEATH_STATE_PREFIX, task #1267).
    # Terminal-status reap only: `proposed` (the incident-class status) is
    # NOT terminal, so a live dispatch->boot-death loop's day counter is
    # never reset mid-episode; the day-keyed counter self-expires at the UTC
    # day roll anyway. The sidecar `boot-death-events.jsonl` is outside the
    # `*.json` glob by suffix.
    (BOOT_DEATH_STATE_PREFIX, ""),
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
    # Per-issue dispatch lease (#843 M1, spawn_session.dispatch_lease_path).
    # Terminal-status + age-backstop reaping of leftover lease files: reaping
    # a TERMINAL task's lease cannot enable a duplicate (terminal tasks are
    # never dispatched), while an ACTIVE task's fresh lease is never touched
    # (the keep branch below). The glob is `dispatch-lease-*.json`, so the
    # PERMANENT `dispatch-lease-<N>.lock` flock sidecar is NOT swept — by
    # design (tiny; unlinking a flock target risks the lock-on-deleted-file
    # hole; recreated on demand by the next slow-path acquire).
    ("dispatch-lease-", ""),
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
    - ``session_progress.json`` / ``watch.lock`` /
      ``last-session-dispatch.json`` (#1059 stagger stamp)
      (project-singletons, not per-issue).
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


# Floor age before a `paused-takeover` sentinel is GC'd. A stale sentinel is
# INERT after the (default 6h) TTL — this reap is clutter-control only, and
# 7 days deliberately preserves the human-readable takeover record for
# post-hoc forensics well past the takeover itself (#903).
TAKEOVER_SENTINEL_GC_AGE_S = 7 * 24 * 3600


def _gc_stale_takeover_sentinels(now: float, dry_run: bool) -> int:
    """Unlink ``*.json.paused-takeover-*`` sentinels older than
    ``max(7 days, the configured takeover TTL)`` — the ``*.json`` GC globs
    can never match them, so without this they would linger forever (#903).

    The ``max()`` is load-bearing (#903 round-1 critique Must-Fix): with
    ``EPS_TAKEOVER_TTL_H`` set above 168h a fixed 7-day reap would delete a
    sentinel that is STILL protecting a live takeover. Returns the reap
    count for the :func:`gc_pass` summary line."""
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return 0
    gc_age = max(TAKEOVER_SENTINEL_GC_AGE_S, _takeover_ttl_s())
    reaped = 0
    for p in sorted(AUTONOMOUS_REGISTRY_DIR.glob("*.json.paused-takeover-*")):
        try:
            if now - p.stat().st_mtime < gc_age:
                continue
        except OSError:
            continue
        print(f"gc: {'would reap' if dry_run else 'reaping'} stale takeover sentinel {p.name}")
        if not dry_run:
            p.unlink(missing_ok=True)
        reaped += 1
    return reaped


def gc_pass(dry_run: bool, now: float | None = None) -> None:
    """Top-level wrapper around :func:`_gc_orphaned_eps_autonomous_files` (+
    the #903 stale-takeover-sentinel reap) for consistency with the other
    ``*_pass`` entry points + the ``--gc-only`` debug flag."""
    now = now if now is not None else time.time()
    counts = _gc_orphaned_eps_autonomous_files(now, dry_run)
    reaped_sentinels = _gc_stale_takeover_sentinels(now, dry_run)
    if reaped_sentinels:
        counts["paused-takeover"] = counts.get("paused-takeover", 0) + reaped_sentinels
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
#     (awaiting_promotion / completed / archived — the pod-safety DONE set
#     AUTO_STOP_DONE, NOT the wider POD_SAFETY_AUTO_STOP (#980);
#     ``followups_running``, ``blocked``, and ``on_hold`` are excluded
#     because the session may be legitimately live there);
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
# the pod-safety DONE set AUTO_STOP_DONE (NOT the GC's narrower terminal set,
# and NOT the wider pod-safety trigger set POD_SAFETY_AUTO_STOP):
# `awaiting_promotion` was added 2026-06-10 on the user request "Can we stop
# the happy sessions once they reach awaiting promotion?" — the promotion
# park is a human gate with no session-side work left, and idle sessions
# there accumulated to 73 registered / ~35-40GB RSS. `followups_running`
# is deliberately NOT here: that status means a same-issue follow-up round
# is executing and the session is its driver. `blocked` is NOT here either
# (under investigation; the user may be live-parked in the session).
# `on_hold` is deliberately NOT here — the pod-safety pass stops a paused
# task's escaped pod via AUTO_STOP_PAUSED (#980), but its session is kept
# (the user may be live-parked; same conservatism as `blocked`).
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
    they measure — counting them would reset the clock they read), and so
    do deliberate session-stop records (#990/#1053 — same predicate as
    :func:`_latest_progress_ts`)."""
    best: float | None = None
    for ev in events:
        note = ev.get("note") or ""
        if any(sentinel in note for sentinel in _WATCHER_NOTE_SENTINELS):
            continue
        # A deliberate session stop — incl. the #1053 Step-0 collision-exit /
        # stale-wake-yield breadcrumb — is the death record of the task's
        # driver: anti-liveness, never activity (#990/#1053; mirrors
        # _latest_progress_ts).
        if note.lstrip().startswith("deliberate-stop ") or ev.get("by") == "spawn_session-stop":
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
        for issue, _pod_id, _name, _info in (
            _running_managed_issue_pods(caller="session-reconcile") or []
        )
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
# controlling TTY a USER could be looking at right now (_is_live_user_tty):
# a terminal-run `happy claude` Thomas may be sitting at, OR a tmux pane in a
# session WITH attached clients. The earlier "any tty_nr != 0 -> keep" test
# was a strict superset of "tmux client attached" and so kept abandoned
# DETACHED-tmux sessions FOREVER (the 2026-06-24 class: spawn_session launches
# `happy claude` into a tmux pane, so the wrapper holds the pane's pty as its
# controlling tty whether or not a client is attached, and the bare tty test
# can't tell an abandoned detached pane apart from a live terminal). The
# refined guard uses tmux's session_attached count: a pane whose tmux session
# has zero attached clients is detached, so the wrapper falls through to the
# transcript-idle check (which still keeps anything <12h idle or unresolvable).
# Fail-soft: if tmux is absent / the query fails, the detached set is empty
# and EVERY tty-bearing wrapper is kept, exactly the old behavior.

IDLE_UNMAPPED_STATE_PREFIX = "idle-unmapped-"


def _orphaned_tmux_reap_enabled() -> bool:
    """True unless ``EPM_ORPHANED_TMUX_REAP`` is explicitly set to a falsy
    value (``0`` / ``false`` / ``no``) — the kill-switch for the
    orphaned-tmux-server widening of the idle-unmapped live-user-TTY guard.
    Same parse as :func:`_unmapped_idle_reap_enabled`."""
    raw = os.environ.get("EPM_ORPHANED_TMUX_REAP", "")
    return raw.strip().lower() not in {"0", "false", "no"}


def _tmux_socket_dir() -> Path:
    """The tmux socket directory: ``$TMUX_TMPDIR/tmux-<uid>`` when set, else
    ``/tmp/tmux-<uid>`` (the documented default — tmux(1): sockets live in
    ``tmux-UID`` under ``$TMUX_TMPDIR`` or ``/tmp``)."""
    base = os.environ.get("TMUX_TMPDIR", "").strip() or "/tmp"
    return Path(base) / f"tmux-{os.getuid()}"


def _live_tmux_socket_present() -> bool:
    """True iff AT LEAST ONE socket file exists in the tmux socket dir.

    Distinguishes "a tmux server is reattachable" (a socket file present, a
    new client can attach) from "a server exists but its socket was deleted"
    (no socket file — unreattachable). An unreadable dir returns True (fail
    toward keep: we cannot prove any server is socket-less)."""
    d = _tmux_socket_dir()
    try:
        return any(p.is_socket() for p in d.iterdir())
    except OSError:
        return True  # cannot read the dir -> cannot prove orphaned -> keep


def _proc_comm(pid: int, proc_root: Path) -> str | None:
    """``comm`` (process name) from ``<proc_root>/<pid>/comm``, or ``None`` if
    unreadable (raced exit / perms)."""
    try:
        return (proc_root / str(pid) / "comm").read_text().strip()
    except OSError:
        return None


def _proc_ppid(pid: int, proc_root: Path) -> int | None:
    """``ppid`` from ``<proc_root>/<pid>/stat``, or ``None`` if unreadable.

    Same parse as :func:`_wrapper_has_controlling_tty` / ``_proc_children_map``:
    the ``comm`` field (parenthesised, may contain spaces/parens) is skipped by
    splitting after the LAST ``)``; the remaining fields are
    ``state(0) ppid(1) pgrp(2) ...``."""
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _tmux_server_client_ttys(server_pid: int, proc_root: Path) -> set[str] | None:
    """The set of ATTACHED-CLIENT terminal ttys a tmux server currently holds,
    read from ``<proc_root>/<server_pid>/fd/*`` symlinks that resolve to a
    ``/dev/pts/N`` device — or ``None`` if the fd dir is UNREADABLE
    (perms / race).

    LOAD-BEARING no-attached-client proof. A tmux server receives each attached
    client's terminal fd via ``SCM_RIGHTS``, so a server WITH attached clients
    holds one ``/dev/pts`` fd per client; a server with ZERO clients holds none.
    Verified live: a server's ``/dev/pts`` fd set equalled
    ``tmux list-clients -F '#{client_tty}'`` exactly. The pane-master
    ``/dev/ptmx`` fds and the AF_UNIX ``socket:[inode]`` fds do NOT resolve to
    ``/dev/pts`` so they never false-positive; the pane SLAVE ttys are held by
    the pane CHILD processes, not the server, so are absent from this set.

    Returns:
      * a set (possibly EMPTY) of ``/dev/pts`` client-terminal paths when the fd
        dir was readable — an EMPTY set is POSITIVE proof of zero clients;
      * ``None`` when the fd dir could not be read, OR any single fd symlink
        could not be read (perms / raced exit) — the caller MUST treat ``None``
        as "cannot prove zero clients" -> KEEP (strictly fail-toward-keep)."""
    fd_dir = proc_root / str(server_pid) / "fd"
    out: set[str] = set()
    try:
        fds = list(fd_dir.iterdir())
    except OSError:
        return None  # unreadable dir -> uncertain -> caller keeps
    for fd in fds:
        try:
            target = os.readlink(fd)
        except OSError:
            # A single unreadable fd means we cannot enumerate the full client
            # set; to stay strictly fail-toward-keep, treat ANY unreadable fd in
            # the dir as "cannot prove zero clients".
            return None
        if target.startswith("/dev/pts/"):
            out.add(target)
    return out


def _wrapper_on_orphaned_tmux_server(
    pid: int, *, proc_root: Path = Path("/proc"), max_depth: int = 50
) -> bool:
    """True iff the wrapper is parented (directly or transitively) by a
    ``tmux: server`` process AND that server is PROVABLY UNREATTACHABLE AND
    HAS ZERO ATTACHED CLIENTS (nobody can reattach AND nobody is attached right
    now — the orphaned-tmux class this fix reaps).

    Mapping is PROCESS PARENTAGE, NOT the server's fd table: a tmux server's
    pane leaders are its DIRECT CHILD processes (``ppid == server_pid``), and a
    daemon-tracked ``node /usr/bin/happy claude`` wrapper IS that child. (The
    server's own ``/proc/<pid>/fd`` pts set is the CLIENT terminals — disjoint
    from its panes' ``pane_tty``s — so it must NOT be used for the mapping.)

    Walk the ``ppid`` chain from ``pid`` (bounded by ``max_depth`` + a seen-set
    cycle guard) to find the owning ``tmux: server``. Once found, return True
    ONLY when BOTH:
      1. the tmux socket dir has NO socket file
         (:func:`_live_tmux_socket_present` False) -> no NEW client can attach;
         AND
      2. the server holds ZERO ``/dev/pts`` client fds
         (:func:`_tmux_server_client_ttys` returns an EMPTY set) -> no client is
         attached RIGHT NOW.

    Rationale for signal 2: an established AF_UNIX connection SURVIVES
    ``unlink()`` of the listener path (the deleted path blocks only NEW
    attaches — the same reason the ``kill -USR1`` socket-recreation recovery
    exists). So socket-dir absence ALONE does NOT prove "no attached client": a
    systemd-tmpfiles atime sweep of ``/tmp/tmux-<uid>/`` under a LIVE attached
    SSH session leaves a socketless-but-attached server. Reaping its idle
    session would kill a session a user is looking at; the client-fd proof
    (condition 2) is the fail-toward-keep guard that blocks that false reap.

    If NO ``tmux: server`` ancestor is found (raw login pts, ssh, a non-tmux
    parent, or the chain hit pid 1 / an unreadable link) -> False (not our
    class -> the caller keeps it via its other branches).

    Fail-toward-keep on EVERY uncertain signal:
      * tmux binary absent -> False (nothing to reap; unchanged behavior);
      * ppid unreadable at a hop -> stop the walk, return False (cannot prove a
        ``tmux: server`` ancestor -> keep);
      * comm unreadable at a hop -> STOP the walk, return False (the hop cannot
        be classified, so a socketless clientless ``tmux: server`` reachable
        BEYOND it must NOT reap the wrapper -> keep; #818 round-2 fix);
      * socket dir unreadable -> :func:`_live_tmux_socket_present` True ->
        reads reattachable -> False (keep);
      * server client-fd dir unreadable (:func:`_tmux_server_client_ttys` None)
        -> cannot prove zero clients -> False (keep);
      * server holds >=1 ``/dev/pts`` client fd -> a client is attached ->
        False (keep).
    The predicate can only ever return True for a CONFIRMED
    ``tmux: server``-parented + socket-absent + provably-zero-clients wrapper;
    every ambiguity -> False (keep)."""
    if shutil.which("tmux") is None:
        return False
    seen: set[int] = set()
    cur = pid
    for _ in range(max_depth):
        if cur in seen or cur <= 1:
            return False  # cycle / reached init without a tmux: server
        seen.add(cur)
        comm = _proc_comm(cur, proc_root)
        if comm is None:
            return False  # unreadable comm at a hop -> cannot classify -> keep
        if comm == "tmux: server":
            server_pid = cur
            # Condition 1: no socket -> no NEW attach possible.
            if _live_tmux_socket_present():
                return False  # reattachable -> keep
            # Condition 2: prove ZERO attached clients right now.
            clients = _tmux_server_client_ttys(server_pid, proc_root)
            if clients is None:
                return False  # cannot prove zero clients -> keep
            return len(clients) == 0  # orphaned iff no client attached
        ppid = _proc_ppid(cur, proc_root)
        if ppid is None:
            return False  # unreadable -> cannot prove orphaned -> keep
        cur = ppid
    return False  # depth exhausted without a tmux: server ancestor -> keep


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


# Short reap window for an UNMAPPED session whose LAST-known mapped task was
# TERMINAL (the #720 ghost class: issue-<N>.json deleted by the respawn pass
# at terminal -> session unmapped -> previously only the 12h bucket caught it).
# 30 min keeps the post-terminal session clear of any same-tick promotion/
# finalize activity (which resets the idle clock anyway) while landing the
# worst-case reap STRICTLY under the body's ~1h acceptance window: with the
# 10-min cron and the existing >=2-consecutive-miss guard, worst-case reap is
# 30 min + 2*10 min = 50 min < 60 min (see plan §11 + the arithmetic-bound test
# test_short_window_worst_case_under_acceptance). Override via
# EPM_LAST_MAPPED_TERMINAL_REAP_S.
LAST_MAPPED_TERMINAL_REAP_S = 30 * 60

# Filename prefix for the per-session "last mapped task was terminal"
# breadcrumb at ~/.eps-autonomous/last-mapped-terminal-<sid>.json. Written by
# the respawn pass at the instant it deletes issue-<N>.json for a TERMINAL
# task; read by the idle-unmapped pass to pick the short reap window. This
# prefix is NOT in _load_session_issue_map's prefix list
# (("issue-", "manual-issue-", "campaign-"), spawn_session.py) nor matched by
# the respawn pass's issue-*.json glob, so it can never be mistaken for a
# respawnable registration.
LAST_MAPPED_TERMINAL_PREFIX = "last-mapped-terminal-"


def _last_mapped_terminal_reap_s() -> float:
    """Short reap window in seconds: ``EPM_LAST_MAPPED_TERMINAL_REAP_S`` when
    set to a positive number, else :data:`LAST_MAPPED_TERMINAL_REAP_S` (30 min).
    Garbled / non-positive values fall back to the default (same parse as
    :func:`_unmapped_idle_reap_s`)."""
    raw = os.environ.get("EPM_LAST_MAPPED_TERMINAL_REAP_S", "")
    try:
        val = float(raw)
    except ValueError:
        return LAST_MAPPED_TERMINAL_REAP_S
    return val if val > 0 else LAST_MAPPED_TERMINAL_REAP_S


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


def _wrapper_controlling_tty_path(pid: int) -> str | None:
    """The wrapper's controlling-terminal device path (e.g. ``/dev/pts/24``),
    resolved from ``/proc/<pid>/fd/0`` (falling back to fd 1 / fd 2). Returns
    None if no stdio fd points at a tty (or the proc raced away). Used only to
    cross-reference against the detached-tmux-pane set — the authoritative
    "does this session have a controlling tty AT ALL" check stays
    :func:`_wrapper_has_controlling_tty` (tty_nr off /proc/<pid>/stat)."""
    for fd in (0, 1, 2):
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if target.startswith("/dev/pts/") or target.startswith("/dev/tty"):
            return target
    return None


def _detached_tmux_panes_with_activity() -> tuple[set[str], dict[str, float]]:
    """ONE ``tmux list-panes -a`` call carrying THREE fields per pane —
    ``#{pane_tty}\\t#{session_attached}\\t#{session_activity}`` — folded into:

    1. the DETACHED-pane set (``set[str]`` of ``/dev/pts/N`` for panes whose
       tmux session has ZERO attached clients — unchanged semantics from the
       historical single-field call), AND
    2. a ``{pane_tty: session_activity_epoch}`` map (the epoch seconds of the
       pane's owning tmux session's last activity), used ONLY by the #695
       corroborating-idleness fallback as a SUBSTITUTE idle signal when the
       primary happy-log transcript signal is unavailable.

    ``session_activity`` updates on pane OUTPUT (not on the watcher's read-only
    ``list-panes``/``capture-pane``), so for an idle Claude session that emits
    nothing it is the correct "no turns since" signal and a conservative
    OVER-estimate of liveness — never an under-estimate, exactly the safe
    direction for a destructive reap gate.

    Fail-soft: ANY error (tmux absent, no server, parse failure) returns
    ``(set(), {})`` — the EMPTY detached set preserves the conservative "any
    tty -> live user -> keep" behavior and an empty activity map means the
    fallback finds no substitute idle age and keeps. The fix can only ever
    make the pass reap MORE (a confirmed-detached, confirmed-idle pane), never
    accidentally reap an attached or uncertain one."""
    if shutil.which("tmux") is None:
        return set(), {}
    try:
        out = subprocess.run(
            [
                "tmux",
                "list-panes",
                "-a",
                "-F",
                "#{pane_tty}\t#{session_attached}\t#{session_activity}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return set(), {}
    if out.returncode != 0:
        # No server / no sessions: tmux exits non-zero. Nothing detached to
        # report; keep-all behavior is preserved by the empty set + map.
        return set(), {}
    detached: set[str] = set()
    activity: dict[str, float] = {}
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        # Tolerate a 2-field row (older tmux without session_activity, or a
        # truncated line): take the detached read, skip the activity read.
        if len(parts) < 2:
            continue
        pane_tty, attached_raw = parts[0].strip(), parts[1].strip()
        if not pane_tty:
            continue
        try:
            attached = int(attached_raw)
        except ValueError:
            continue
        if attached == 0:
            detached.add(pane_tty)
        if len(parts) >= 3:
            try:
                activity[pane_tty] = float(parts[2].strip())
            except ValueError:
                # Unparseable activity epoch -> simply absent from the map;
                # the fallback then finds no substitute age and keeps.
                continue
    return detached, activity


def _detached_tmux_pane_ttys() -> set[str]:
    """Set of pty device paths (``/dev/pts/N``) for tmux panes whose tmux
    session currently has ZERO attached clients — i.e. detached panes nobody
    is looking at.

    Thin wrapper over :func:`_detached_tmux_panes_with_activity` returning
    only the detached set (the activity map is the #695 fallback's concern).
    Preserves the historical single-return contract every existing caller +
    test relies on; same fail-soft empty-set-on-error behavior."""
    detached, _activity = _detached_tmux_panes_with_activity()
    return detached


# ── #695 corroborating-idleness fallback for the idle-unmapped pass ───────────
#
# The primary idle signal (the resolved Claude transcript mtime via the
# happy-log path, _transcript_idle_age_s) returns (None, reason) for
# MANUALLY-tmux-launched, non-daemon-spawned sessions — every such session
# logs idle=? and decide_idle_unmapped keeps it forever. That is the
# load-bearing reap blocker for the 2026-06-12 class (25 detached unmapped tmux
# sessions, ~23 GB RSS). The fallback below supplies a SUBSTITUTE idle age
# (tmux session_activity) used ONLY when the primary is unavailable AND every
# no-running-work / no-pending-input / no-running-pod gate passes. Six gates,
# all must hold; any single uncertain signal -> keep.

# Substring patterns matched against a /proc descendant's cmdline that signal
# the session is doing REAL WORK (and so must NOT be reaped even if detached +
# idle). The union of:
#   - the two codex-companion regexes (`codex app-server`,
#     `plugins/cache/openai-codex/`) imported from worktree_audit.py, and
#   - the project workload markers a detached experimenter/dispatch session
#     would show.
# Deliberately a DENYLIST, not an allowlist: every idle session keeps a live
# Claude binary + the fixed MCP server tree (runpod / arxiv / ssh /
# google-workspace / playwright / context7 / todoist), which are NOT work
# signals — an allowlist ("reap only if EXACTLY Claude + the fixed MCP set")
# is brittle to MCP-set drift and fails unsafe on an unrecognized work
# process. The denylist is pinned by a test (test 12) so a future rename of
# either source trips it. The codex patterns are compiled regexes (`.search`);
# the workload markers are plain substrings.
_IDLE_UNMAPPED_WORK_CMDLINE_MARKERS: tuple[str, ...] = (
    "scripts/train.py",
    "scripts/eval.py",
    "scripts/run_sweep.py",
    "scripts/dispatch_issue.py",
    "backend_poll.py",
    "experiment-implementer",
)


def _cmdline_is_work_process(pid: int) -> bool | None:
    """TRI-STATE work-process probe for one pid:

      - ``True``  — ``/proc/<pid>/cmdline`` matches ANY codex-companion
        :data:`ORPHAN_HOLDER_PATTERNS` regex OR contains any
        :data:`_IDLE_UNMAPPED_WORK_CMDLINE_MARKERS` substring (a positive
        work signal);
      - ``False`` — the cmdline was read and matched NOTHING (positively
        not-work);
      - ``None``  — the cmdline was UNREADABLE (perms / raced exit / brief
        EACCES). UNCERTAIN, NOT not-work.

    The None state is load-bearing (#695 round-2 blocker 2): a child in the
    wrapper subtree whose cmdline cannot be read could be a live work
    descendant whose ``/proc`` entry was permission-restricted or momentarily
    unreadable. Collapsing that to ``False`` lets the gate-3 work check pass
    and the session be reaped — violating the fail-toward-KEEP contract. The
    descendant walk (:func:`_has_running_work_descendant`) therefore treats any
    ``None`` in the walk as work-present (KEEP)."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    if any(marker in cmd for marker in _IDLE_UNMAPPED_WORK_CMDLINE_MARKERS):
        return True
    return any(pat.search(cmd) for pat in ORPHAN_HOLDER_PATTERNS)


def _has_running_work_descendant(
    pid: int, children_map: dict[int, list[int]] | None = None
) -> bool:
    """True iff ``pid`` or any /proc descendant is a RUNNING work process
    (codex companion or a project workload — :func:`_cmdline_is_work_process`)
    OR any descendant's cmdline is UNREADABLE (uncertain -> treated as
    work-present, fail-toward-KEEP).

    The #695 gate-3 work-descendant check. Reuses the
    :func:`_proc_children_map` parse + an iterative descendant walk (same
    pattern as :func:`_has_claude_descendant`). Crucially does NOT key on
    :func:`_has_claude_descendant` — every idle session keeps a live Claude
    binary + the fixed MCP tree, which are not work signals — so it keys only
    on the narrow work-process denylist. A ``True`` return is a gate-3 KEEP
    (one of the six AND gates fails). Because :func:`_cmdline_is_work_process`
    is TRI-STATE, an unreadable child cmdline (``None``) is treated as
    work-present here (KEEP) rather than skipped — the per-child
    "unreadable -> uncertain -> KEEP" posture the call site's ``except OSError``
    cannot reach, since an unreadable cmdline raises no exception that
    propagates (#695 round-2 blocker 2)."""
    if children_map is None:
        children_map = _proc_children_map()
    seen: set[int] = set()
    stack = [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        verdict = _cmdline_is_work_process(p)
        if verdict is None or verdict is True:
            # Positive work signal OR an unreadable (uncertain) cmdline ->
            # treat as work-present -> KEEP.
            return True
        stack.extend(children_map.get(p, ()))
    return False


# Empty-prompt placeholder substrings (case-insensitive): when the captured
# pane's last logical input line reduces to one of these, the input box is
# EMPTY (the rendered hint text), not buffered user input -> the pending-input
# probe may proceed. Deliberately small + conservative — an UNRECOGNIZED
# remainder is treated as input -> KEEP.
_PANE_EMPTY_PROMPT_PLACEHOLDERS: tuple[str, ...] = (
    'try "',
    "for shortcuts",
)

# Leading prompt-prefix glyphs stripped before judging whether the input line
# is empty: the Claude TUI input box renders a leading prompt marker (the
# U+276F caret on the live render, or a ``>`` / box border on older / idealized
# renders). After stripping leading whitespace, a leading run of these chars
# (plus one following space — regular OR non-breaking; the live caret is
# followed by ``\xa0`` U+00A0) is removed; a non-empty, non-placeholder
# remainder counts as buffered input. (U+276F included by codepoint to keep the
# ruff confusables linter quiet — it reads as a ``>`` look-alike.)
_PANE_PROMPT_PREFIX_CHARS = "\u276f>│╭╮╰╯─┐└┘├┤┬┴┼┌"

# Box-drawing / horizontal-rule glyphs: a line whose every non-whitespace
# character is one of these is a pure BORDER / RULE line, NOT the input row.
# The live Claude render frames the input box with full-width ``─`` rules
# above and below (the top rule sometimes carries a single ``↯`` token-count
# glyph, which is also listed here so an otherwise-pure rule still classifies
# as a border). The bottom-up scanner skips these to reach the actual prompt.
_PANE_BORDER_CHARS = frozenset("─╭╮╰╯│║═=-_↯┐└┘├┤┬┴┼┌")

# Leading glyphs that mark a Claude TUI FOOTER line rendered BELOW the input
# box (the permissions-mode line ``⏵⏵ bypass permissions on …`` is the
# canonical one observed live; ``?`` is the shortcuts/help footer; ``↑``/``↓``
# label navigation footers on menu screens). A line whose first non-whitespace
# glyph is one of these is a footer, NOT the input row — the bottom-up scanner
# skips it. Kept deliberately small + KEEP-leaning: an unrecognized trailer is
# NOT treated as a footer, so the scanner falls through to it and (being
# non-empty, non-placeholder) judges it as input -> KEEP.
_PANE_FOOTER_PREFIX_CHARS = "⏵?↑↓"


def _pane_line_is_border(line: str) -> bool:
    """True iff ``line`` is a pure box-border / horizontal-rule line — every
    non-whitespace character is in :data:`_PANE_BORDER_CHARS`. An all-whitespace
    line is NOT a border (it carries no glyphs). Used by the bottom-up input
    scanner to skip the rules framing the input box."""
    nonws = "".join(line.split())
    return bool(nonws) and all(ch in _PANE_BORDER_CHARS for ch in nonws)


def _pane_line_is_footer(line: str) -> bool:
    """True iff ``line`` is a Claude TUI footer rendered below the input box —
    its first non-whitespace glyph is in :data:`_PANE_FOOTER_PREFIX_CHARS`
    (the ``⏵⏵ bypass permissions …`` permissions line, the ``?`` shortcuts
    line, or a navigation footer). KEEP-leaning: an unrecognized trailer is NOT
    a footer, so the scanner falls through to it and judges it as input."""
    stripped = line.lstrip()
    return bool(stripped) and stripped[0] in _PANE_FOOTER_PREFIX_CHARS


def _pane_has_pending_input(pane_tty: str) -> bool:
    """True (= "there might be unsent input, KEEP") unless the captured pane's
    input box is positively recognized as EMPTY.

    The #695 gate-5 typed-but-unsent guard. Probes the pane's VISIBLE content
    via ``tmux capture-pane -p -t <pane_tty>`` (a pane's tty is an accepted
    ``-t`` target). A KEEP-leaning heuristic over the rendered terminal text,
    NOT a parser of the Claude TUI internal state — its failure mode is always
    a spurious KEEP (a session that COULD be reaped is kept), never a spurious
    reap.

    The real Claude render places the U+276F caret input row ABOVE a bottom
    box-rule and a permissions footer, so the last captured line is ALWAYS the
    footer — never the input row (#695 round-2 blocker 1). This
    scans the captured pane from the BOTTOM UP, skipping pure border/rule lines
    and known footer lines, and applies the empty-vs-buffered heuristic to the
    FIRST line that is neither.

    Returns **True** on ANY of:
      - the ``capture-pane`` subprocess errors / times out / returns non-zero
        / tmux is absent / the pane is gone (fail toward KEEP);
      - the whole captured pane is borders + footers (no input row found) —
        cannot confirm empty -> KEEP;
      - the first non-border, non-footer line (scanning bottom-up) — after
        stripping leading whitespace, a leading run of prompt-prefix glyphs +
        one space (regular or non-breaking), and trailing whitespace — is
        NON-EMPTY and does NOT match a known empty-prompt placeholder.

    Returns **False** (= "no pending input, may proceed") only when it
    positively recognizes an empty / placeholder-only input box (an empty
    remainder or a known placeholder substring)."""
    if shutil.which("tmux") is None:
        return True
    try:
        out = subprocess.run(
            ["tmux", "capture-pane", "-p", "-t", pane_tty],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return True
    if out.returncode != 0:
        # Pane gone / capture failed -> fail toward KEEP.
        return True
    lines = out.stdout.splitlines()
    # Scan bottom-up for the first line that is neither a pure border/rule nor a
    # known footer — that is the actual input row. Whitespace-only, border, and
    # footer lines are all skipped.
    input_line: str | None = None
    for line in reversed(lines):
        if not line.strip():
            continue
        if _pane_line_is_border(line) or _pane_line_is_footer(line):
            continue
        input_line = line
        break
    if input_line is None:
        # The whole capture is borders + footers (or nothing rendered) -> no
        # input row found -> cannot confirm empty -> KEEP.
        return True
    last = input_line.lstrip()
    # Strip a leading run of prompt-prefix glyphs, then a single space (regular
    # or the live render's non-breaking ``\xa0`` separator).
    stripped = last.lstrip(_PANE_PROMPT_PREFIX_CHARS)
    if stripped[:1] in (" ", "\xa0"):
        stripped = stripped[1:]
    remainder = stripped.strip()
    if not remainder:
        # Empty input box -> may proceed.
        return False
    low = remainder.lower()
    # A recognized empty-prompt placeholder hint -> may proceed (return False);
    # any unrecognized non-empty remainder -> treat as buffered input -> KEEP.
    return not any(ph in low for ph in _PANE_EMPTY_PROMPT_PLACEHOLDERS)


def _fallback_idle_age_s(
    pane_tty: str | None, activity_map: dict[str, float], now: float
) -> float | None:
    """Substitute idle age (seconds) for the #695 fallback: ``now -
    session_activity`` of the pane's owning tmux session, or ``None`` when no
    usable activity epoch is available (no pane tty, pane absent from the map,
    or a future/garbled epoch). ``None`` -> the fallback supplies no idle age
    and the session is kept (gate-uncertain)."""
    if not pane_tty:
        return None
    epoch = activity_map.get(pane_tty)
    if not isinstance(epoch, int | float):
        return None
    return max(0.0, now - float(epoch))


# Default corroborating-idleness fallback window (seconds). 24h — deliberately
# 2x the primary 12h UNMAPPED_IDLE_REAP_S: the session_activity signal is a
# WEAKER corroborating signal than the transcript mtime, so it earns a longer
# floor (keeps an overnight pause well clear while still clearing the 19-43h
# accumulation class within ~a day). Override via
# EPM_UNMAPPED_TMUX_IDLE_FALLBACK_S.
UNMAPPED_TMUX_IDLE_FALLBACK_S = 24 * 3600


def _unmapped_tmux_idle_fallback_enabled() -> bool:
    """True unless ``EPM_UNMAPPED_TMUX_IDLE_FALLBACK_ENABLED`` is explicitly
    set to a falsy value (``0`` / ``false`` / ``no``) — a per-feature
    kill-switch for the #695 corroborating-idleness fallback that leaves the
    rest of the idle-unmapped pass (and its own ``EPM_UNMAPPED_IDLE_REAP``
    kill-switch) untouched. Default-ON, same parsing as
    :func:`_unmapped_idle_reap_enabled`."""
    raw = os.environ.get("EPM_UNMAPPED_TMUX_IDLE_FALLBACK_ENABLED", "")
    return raw.strip().lower() not in {"0", "false", "no"}


def _unmapped_tmux_idle_fallback_s() -> float:
    """Fallback idle window in seconds: ``EPM_UNMAPPED_TMUX_IDLE_FALLBACK_S``
    when set to a positive number, else :data:`UNMAPPED_TMUX_IDLE_FALLBACK_S`
    (24h). Garbled / non-positive values fall back to the default (same
    parsing as :func:`_unmapped_idle_reap_s`)."""
    raw = os.environ.get("EPM_UNMAPPED_TMUX_IDLE_FALLBACK_S", "")
    try:
        val = float(raw)
    except ValueError:
        return UNMAPPED_TMUX_IDLE_FALLBACK_S
    return val if val > 0 else UNMAPPED_TMUX_IDLE_FALLBACK_S


def _idle_unmapped_debug_enabled() -> bool:
    """True when ``EPM_IDLE_UNMAPPED_DEBUG`` is set to a truthy value — gates
    the denser per-candidate #695 fallback-gate trace (default OFF after the
    diagnostic cycle so it does not spam every 10-min pass; the once-per-pass
    detached-set-size log + loud empty-set beacon stay UNCONDITIONAL)."""
    raw = os.environ.get("EPM_IDLE_UNMAPPED_DEBUG", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_live_user_tty(
    pid: int,
    detached_tmux_ttys: set[str],
    *,
    check_orphaned: bool = False,
) -> bool:
    """True iff this wrapper holds a controlling tty that a user could be
    looking at RIGHT NOW — the refined "Thomas may literally be sitting here"
    guard for the idle-unmapped pass.

    Refines the bare :func:`_wrapper_has_controlling_tty` test: a wrapper
    whose controlling tty is a tmux pane in a session with zero attached
    clients (``detached_tmux_ttys``) is NOT a live-user tty — nobody is
    attached — so it falls through to the transcript-idle check (which still
    keeps anything <12h idle or unresolvable). Every other tty-bearing
    wrapper (raw login pts, an ATTACHED tmux pane, or a controlling tty that
    cannot be cross-referenced to a detached pane) is treated as live and
    kept, preserving the conservative default. An unreadable /proc (race /
    perms) keeps the keep-leaning :func:`_wrapper_has_controlling_tty`
    semantics (returns True).

    PLUS (``check_orphaned=True``, the idle-unmapped pass under the default-ON
    ``EPM_ORPHANED_TMUX_REAP`` knob): a wrapper parented by an ORPHANED tmux
    server (socket deleted -> unreattachable, AND zero attached clients) is NOT
    a live-user tty either — that pane is unreachable by construction, so it
    falls through to the same transcript-idle check. This branch keys on the
    wrapper's PID via process parentage (:func:`_wrapper_on_orphaned_tmux_server`),
    NOT on the ``tty_path`` string, so it fires even when the pts is
    unresolvable. Both the detached-pane and orphaned-server checks fail toward
    keep on any unresolved signal. The default-``False`` keyword arg preserves
    every existing 2-arg caller (the new branch never runs when omitted)."""
    if not _wrapper_has_controlling_tty(pid):
        return False
    tty_path = _wrapper_controlling_tty_path(pid)
    # A confirmed detached tmux pane (has a tty, but no client attached) is
    # NOT a live-user tty.
    if tty_path is not None and tty_path in detached_tmux_ttys:
        return False
    # NOT live iff parented by an orphaned (socket-deleted, zero-client) tmux
    # server — keys on the PID via parentage, so pts resolvability is
    # irrelevant to it. Every other tty-bearing wrapper (unresolvable tty_path,
    # an attached pane, a raw login pts, a live server) is treated as live and
    # kept (conservative default).
    return not (check_orphaned and _wrapper_on_orphaned_tmux_server(pid))


def _maybe_log_orphaned_tmux_fire(
    pid: int, detached_tmux_ttys: set[str], check_orphaned: bool
) -> None:
    """Observability (idle-unmapped fold 1): print ONE line when the
    orphaned-parentage branch of :func:`_is_live_user_tty` is what made a
    tty-bearing, non-detached-pane wrapper read not-live — so a fire is
    greppable in the ``--dry-run`` smoke and a silent-inert regression is
    detectable. Called only for a wrapper already classified not-live, so it
    costs the extra parentage read ONLY for that rare case and nothing on the
    healthy fleet (no fire, no orphaned server). Pure side-effect log; never
    changes the reap decision."""
    if not check_orphaned:
        return
    if not _wrapper_has_controlling_tty(pid):
        return
    if _wrapper_controlling_tty_path(pid) in detached_tmux_ttys:
        return
    if not _wrapper_on_orphaned_tmux_server(pid):
        return
    print(
        f"  [idle-unmapped] orphaned-tmux wrapper pid={pid} (server socket "
        f"absent, zero clients) -> not-live, entering idle check"
    )


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


def _last_mapped_terminal_path(sid: str) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{LAST_MAPPED_TERMINAL_PREFIX}{sid}.json"


def _record_last_mapped_terminal(
    sid: str, issue: int, terminal_status: str, dry_run: bool, now: float | None = None
) -> None:
    """Write the #720 breadcrumb atomically (temp + rename). Idempotent: a
    re-write for an already-recorded sid just refreshes the fields. Fail-soft:
    an OSError on write is logged and swallowed (the breadcrumb is an
    OPTIMIZATION — a missing one only means the session reaps at the old 12h
    window, never a wrong kill). Skipped under ``dry_run``, and only ever
    written for a TERMINAL status (a PARK/ACTIVE status would widen scope when
    later read, so it is refused here too)."""
    if dry_run:
        return
    if terminal_status not in TERMINAL:
        return
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "happy_session_id": sid,
        "issue": issue,
        "terminal_status": terminal_status,
        "recorded_at": now if now is not None else time.time(),
    }
    dest = _last_mapped_terminal_path(sid)
    tmp = dest.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(dest)
    except OSError as e:
        print(f"  last-mapped-terminal: write failed for {sid}: {e}", file=sys.stderr)


def _last_mapped_terminal(sid: str) -> tuple[str, int] | None:
    """The recorded ``(terminal_status, issue)`` for ``sid``, or ``None`` (no
    breadcrumb / garbled / not a terminal status / no int issue). Validates the
    recorded status against :data:`TERMINAL` so a stale/garbled value can never
    widen scope, and surfaces the recorded ``issue`` so the consumer can run the
    running-pod + live-follow-up protected-class guards without a session->issue
    mapping it no longer has."""
    path = _last_mapped_terminal_path(sid)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    status = data.get("terminal_status")
    issue = data.get("issue")
    if status in TERMINAL and isinstance(issue, int):
        return (status, issue)
    return None


def _gc_orphan_last_mapped_terminal(live_sids: set[str], dry_run: bool) -> None:
    """Drop #720 breadcrumbs whose session is no longer live (reaped, or the
    user promoted + the session ended). Called once per daemon-reachable tick
    from :func:`idle_unmapped_pass`, mirroring
    :func:`_gc_orphan_idle_unmapped_state`."""
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return
    for path in AUTONOMOUS_REGISTRY_DIR.glob(f"{LAST_MAPPED_TERMINAL_PREFIX}*.json"):
        sid = path.name[len(LAST_MAPPED_TERMINAL_PREFIX) : -len(".json")]
        if sid not in live_sids and not dry_run:
            path.unlink(missing_ok=True)


def _effective_idle_reap_s(sid: str, mapped: bool, has_tty: bool, long_reap_s: float) -> float:
    """The idle-unmapped reap window (s) for one session: the SHORT (#720)
    window when the session is a finished, unmapped, non-TTY, unprotected
    autonomous /issue session, else ``long_reap_s`` (the 12h default) unchanged.

    Extracted from :func:`_process_idle_unmapped` (keeps its branch count under
    the C901 cap) AND keeps the caller branch-free (it compares the return
    against ``long_reap_s`` to set its observability label). The SHORT window is
    applied (via ``min``, so it can only ever SHORTEN, never lengthen) ONLY when
    the session is unmapped + not a live-user TTY AND the terminal breadcrumb is
    present AND both LAZY protected-class guards clear — every other case
    returns ``long_reap_s`` unchanged. The guards both FAIL TOWARD KEEP when the
    recorded issue still has work in flight:

    - **Guard 1** (running managed pod): :func:`_running_managed_issue_pods`
      returns ``None`` on a FAILED snapshot (uncertain -> KEEP), ``[]`` for no
      pods, or 4-tuples ``(issue, pod_id, pod_name, info)`` — a snapshot that is
      ``None`` OR contains a RUNNING pod for the breadcrumb's issue keeps the
      long window. Reuses the same helper the #695 fallback gate 4 uses.
    - **Guard 2** (live same-issue follow-up): :func:`_task_followup_active` on
      the breadcrumb's recorded issue (a follow-up signal newer than the latest
      done-transition) keeps the long window.

    Gating on ``mapped``/``has_tty`` here (rather than at the call site) keeps
    the breadcrumb read + the two guard probes LAZY — they fire only for a
    genuinely-finished, unmapped, non-TTY candidate, not every tick. A mapped
    session always keeps the long window (the breadcrumb is read only in the
    unmapped branch)."""
    if mapped or has_tty:
        return long_reap_s
    crumb = _last_mapped_terminal(sid)
    if crumb is None:
        return long_reap_s
    _term_status, crumb_issue = crumb
    running = _running_managed_issue_pods(caller="idle-unmapped-720")
    pod_uncertain_or_present = running is None or any(t[0] == crumb_issue for t in running)
    if pod_uncertain_or_present:
        return long_reap_s
    if _task_followup_active(crumb_issue):
        return long_reap_s
    return min(long_reap_s, _last_mapped_terminal_reap_s())


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


def _append_idle_unmapped_audit(payload: dict, dry_run: bool) -> None:
    """Pre-stop AUDIT row for a #695 corroborating-idleness FALLBACK reap —
    written to the SAME ``idle-unmapped-events.jsonl`` stream as
    :func:`_append_idle_unmapped_event`, but with ``kind:
    "would_stop_fallback"`` and a STRUCTURED ``payload`` (not a free-form
    note), and written IMMEDIATELY BEFORE :func:`_stop_session` fires on the
    fallback path (so ``audit_ts < stop_ts`` holds by construction).

    A destructive safety gate must leave a durable, self-explaining record of
    EVERY gate signal BEFORE it acts — a wrong fallback reap is then
    reconstructable from the events stream alone (the only pre-stop line the
    primary path writes is the transient gate-signal-free ``mapped/tty/idle``
    print). Two distinct ``kind`` values (``would_stop_fallback`` ->
    ``idle-unmapped`` flavored ``stopped_fallback``) let an operator read the
    one chronological file in order. Fail-soft, mirroring
    :func:`_append_idle_unmapped_event`."""
    dest = AUTONOMOUS_REGISTRY_DIR / "idle-unmapped-events.jsonl"
    row = {"ts": datetime.now().astimezone().isoformat(), "kind": "would_stop_fallback"}
    row.update(payload)
    line = json.dumps(row)
    if dry_run:
        print(f"  [dry-run] would append idle-unmapped fallback audit row to {dest}")
        return
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as e:
        print(f"  WARNING: appending idle-unmapped fallback audit failed: {e}", file=sys.stderr)


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


def _evaluate_idle_unmapped_fallback(
    sid: str,
    pid: int,
    now: float,
    detached_tmux_ttys: set[str],
    tmux_activity: dict[str, float],
) -> tuple[float | None, dict | None]:
    """The #695 SIX-gate corroborating-idleness fallback for ONE session.

    Caller has already established the session is unmapped, not a live-user
    tty, the primary transcript signal is unavailable, and the fallback is
    enabled. Returns ``(fallback_idle_age_s, audit_payload)`` when ALL SIX
    gates hold — the caller then drives the existing decision lattice with the
    substitute idle age and writes ``audit_payload`` BEFORE the stop — or
    ``(None, None)`` when any single gate is uncertain (the session keeps).

    The six gates (all AND, any uncertain -> keep):
      1. detached — pane pts in the trustworthy ``detached_tmux_ttys`` set;
      2. unmapped — already established by the caller (precondition);
      3. no running work descendant (codex / project workload) — an unreadable
         /proc is uncertain -> keep;
      4. no running managed pod anywhere (a ``None`` snapshot is uncertain ->
         keep);
      5. no pending (typed-but-unsent) pane input — KEEP-leaning probe;
      6. session_activity-derived idle age over the conservative fallback
         window."""
    fallback_window_s = _unmapped_tmux_idle_fallback_s()
    pane_tty = _wrapper_controlling_tty_path(pid)
    # Gate 1: detached pane pts in the trustworthy set.
    detached_ok = pane_tty is not None and pane_tty in detached_tmux_ttys
    # Gate 6 (substitute idle age): session_activity over the window.
    fb_idle = _fallback_idle_age_s(pane_tty, tmux_activity, now) if detached_ok else None
    over_window = fb_idle is not None and fb_idle >= fallback_window_s
    # Gate 3: no running work descendant (unreadable /proc -> uncertain -> KEEP).
    try:
        work_descendant = _has_running_work_descendant(pid, _proc_children_map())
    except OSError:
        work_descendant = True
    # Gate 4: no running managed pod anywhere (None snapshot -> uncertain -> KEEP).
    running_pods = _running_managed_issue_pods(caller="idle-unmapped-fallback")
    no_running_pods = running_pods is not None and len(running_pods) == 0
    # Gate 5: no pending (typed-but-unsent) pane input — KEEP-leaning.
    pending_input = _pane_has_pending_input(pane_tty) if detached_ok else True
    gates_ok = (
        detached_ok
        and over_window
        and not work_descendant
        and no_running_pods
        and not pending_input
    )
    if _idle_unmapped_debug_enabled():
        print(
            f"  idle-unmapped[debug] fallback session {sid} (pid={pid}): "
            f"pane_tty={pane_tty} detached={detached_ok} "
            f"fb_idle={'%.1fh' % (fb_idle / 3600) if fb_idle is not None else '?'} "
            f"over_window={over_window} work_descendant={work_descendant} "
            f"no_running_pods={no_running_pods} pending_input={pending_input} "
            f"gates_ok={gates_ok}",
            file=sys.stderr,
        )
    if not gates_ok:
        return None, None
    audit = {
        "sid": sid,
        "pid": pid,
        "fallback_source": "tmux_session_activity",
        "idle_age_s": fb_idle,
        "threshold_env_value": fallback_window_s,
        "detached_verdict": {"pane_tty": pane_tty, "in_detached_set": True},
        "work_descendant": False,
        "running_pods": [],
        "pending_input": False,
    }
    return fb_idle, audit


def _do_idle_unmapped_stop(
    sid: str,
    pid: int,
    now: float,
    dry_run: bool,
    threshold: int,
    *,
    idle_age_s: float | None,
    idle_label: str,
    idle_reap_s: float,
    prev: dict,
    prev_missed: int,
    prev_alerted: bool,
    first_over_ts: float,
    fallback_active: bool,
    fallback_audit: dict | None,
) -> None:
    """Execute the idle-unmapped STOP action for one session (extracted from
    :func:`_process_idle_unmapped` to keep its branch count manageable).

    On the #695 FALLBACK path, writes the pre-stop AUDIT row carrying every
    gate signal BEFORE the stop fires (so ``audit_ts < stop_ts`` by
    construction) and a fallback-DISTINCT post-stop note (NEVER the
    primary-transcript narrative — the fallback never read a transcript). On
    the primary path, writes the existing transcript narrative unchanged. The
    reversible-daemon-stop + ACK!=kill state machinery is identical for both."""
    if fallback_active and fallback_audit is not None:
        _append_idle_unmapped_audit(fallback_audit, dry_run)
    acked = _stop_session(sid, dry_run)
    if acked and fallback_active:
        _append_idle_unmapped_event(
            f"{_IDLE_UNMAPPED_STOP_FALLBACK_NOTE_SENTINEL} auto-stopped idle "
            f"unmapped Happy session {sid} (wrapper pid {pid}) on the "
            f"corroborating-idleness FALLBACK path: the primary Claude "
            f"transcript signal was unavailable, so tmux session_activity "
            f"supplied the idle age ({idle_label} >= "
            f"{idle_reap_s / 3600:.1f}h window, >= {threshold} consecutive "
            f"checks). All six gates held: detached tmux pane, no issue "
            f"mapping, no running work descendant, no running managed pod, "
            f"no pending pane input. Respawn if needed: "
            f"`spawn_session.py spawn-issue --issue <N>` (or `spawn-pm`). "
            f"Set EPM_UNMAPPED_TMUX_IDLE_FALLBACK_ENABLED=0 to disable the "
            f"fallback, or EPM_UNMAPPED_IDLE_REAP=0 for alert-only.",
            dry_run,
        )
    elif acked:
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


def _process_idle_unmapped(
    sid: str,
    pid: int,
    issue: int | None,
    now: float,
    dry_run: bool,
    threshold: int,
    *,
    reap_enabled: bool,
    detached_tmux_ttys: set[str] | None = None,
    tmux_activity: dict[str, float] | None = None,
    check_orphaned: bool = True,
) -> None:
    """Apply the idle-unmapped decision to one live, non-PM, EPS-cwd session:
    check the wrapper's controlling TTY, stat the resolved transcript, and
    act per :func:`decide_idle_unmapped`.

    ``has_tty`` here means "a controlling tty a USER could be looking at right
    now" (:func:`_is_live_user_tty`): a tty-bearing wrapper that is a DETACHED
    tmux pane (``detached_tmux_ttys`` — computed once per pass) is NOT counted
    as live, so an abandoned detached-tmux session falls through to the
    transcript-idle check instead of being kept forever. ``None`` (the test /
    legacy default) computes the detached-pane set inline. ``check_orphaned``
    (default True; the pass passes the env-gated :func:`_orphaned_tmux_reap_enabled`
    value) additionally classifies a wrapper parented by an ORPHANED tmux
    server (socket deleted + zero attached clients) as not-live — an unreachable
    pane that would otherwise be kept forever by the empty detached-map.

    #695 corroborating-idleness FALLBACK: when the primary transcript signal
    is unavailable (``idle_age_s is None`` — the manually-tmux-launched class
    that logged ``idle=?`` and was kept forever), the SIX-gate fallback may
    supply a SUBSTITUTE ``idle_age_s`` from tmux ``session_activity``
    (``tmux_activity``, also computed once per pass). All six gates must hold
    (detached + unmapped + no work-descendant + no running pod + no pending
    pane input + over the conservative fallback window); any single uncertain
    signal keeps. A reap on the fallback path writes a pre-stop audit row
    BEFORE the stop and a fallback-DISTINCT post-stop note."""
    if detached_tmux_ttys is None or tmux_activity is None:
        _detached, _activity = _detached_tmux_panes_with_activity()
        detached_tmux_ttys = _detached if detached_tmux_ttys is None else detached_tmux_ttys
        tmux_activity = _activity if tmux_activity is None else tmux_activity
    mapped = issue is not None
    has_tty = (
        _is_live_user_tty(pid, detached_tmux_ttys, check_orphaned=check_orphaned)
        if not mapped
        else False
    )
    idle_age_s: float | None = None
    signal_reason: str | None = None
    if not mapped and not has_tty:
        _maybe_log_orphaned_tmux_fire(pid, detached_tmux_ttys, check_orphaned)
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

    # ── #720 short reap window (the completed-/parked-session ghost class) ────
    # An unmapped session whose LAST-known mapped task was TERMINAL is a
    # finished autonomous /issue session, not a generic abandoned chat session,
    # so it earns the SHORT reap window (default 30 min) instead of 12h. The
    # breadcrumb read is one IO call, taken ONLY when present.
    #
    # The two PROTECTED-CLASS guards below FAIL TOWARD KEEP — retaining the 12h
    # window, NOT applying the short window — when the recorded issue still has
    # work in flight. Without them, narrowing the window 12h -> 30 min on the
    # PRIMARY path would expose (a) a still-RUNNING pod on a session that has
    # gone quiet > 30 min mid-work, and (b) a live same-issue follow-up that has
    # not yet re-registered issue-<N>.json (register-current fires only after
    # set-status followups_running). Both guards are LAZY: invoked only when the
    # terminal breadcrumb is present, so the pod-API snapshot + the per-task
    # events fetch are paid only for a genuinely-finished candidate, not every
    # tick. Mapped sessions never reach this (the breadcrumb is read only in the
    # unmapped branch).
    effective_reap_s = _effective_idle_reap_s(sid, mapped, has_tty, idle_reap_s)
    short_window = effective_reap_s < idle_reap_s
    idle_reap_s = effective_reap_s

    # ── #695 corroborating-idleness fallback ─────────────────────────────────
    # ONLY when the primary signal is unavailable for an in-the-idle-branch
    # (unmapped + not-live-tty) session: try to supply a substitute idle age
    # from tmux session_activity, gated on six AND conditions (evaluated in the
    # extracted helper). Any uncertain signal leaves idle_age_s as None (the
    # existing ("skip", missed) fail-toward-keep path).
    fallback_active = False
    fallback_audit: dict | None = None
    if idle_age_s is None and not mapped and not has_tty and _unmapped_tmux_idle_fallback_enabled():
        fb_idle, fallback_audit = _evaluate_idle_unmapped_fallback(
            sid, pid, now, detached_tmux_ttys, tmux_activity
        )
        if fallback_audit is not None:
            idle_age_s = fb_idle
            fallback_active = True
            # A weaker signal earns the longer floor: the decision uses the
            # FALLBACK window as its reap threshold (not the primary 12h).
            idle_reap_s = _unmapped_tmux_idle_fallback_s()

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
    source_label = " source=fallback" if fallback_active else ""
    window_label = " window=short-terminal" if short_window else ""
    print(
        f"  session {sid} (pid={pid}): mapped={mapped} tty={has_tty} "
        f"idle={idle_label}{source_label}{window_label} "
        f"missed={prev_missed}->{new_missed} action={action}"
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
        _do_idle_unmapped_stop(
            sid,
            pid,
            now,
            dry_run,
            threshold,
            idle_age_s=idle_age_s,
            idle_label=idle_label,
            idle_reap_s=idle_reap_s,
            prev=prev,
            prev_missed=prev_missed,
            prev_alerted=prev_alerted,
            first_over_ts=first_over_ts,
            fallback_active=fallback_active,
            fallback_audit=fallback_audit,
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
    or issue-<N> worktree cwd), wrappers holding a LIVE-USER controlling TTY
    (:func:`_is_live_user_tty` — a raw login pts or an ATTACHED tmux pane;
    a DETACHED tmux pane is NOT live and falls through to the idle check),
    and any session whose idleness signal cannot be resolved (fail toward
    keep).

    #695 corroborating-idleness fallback: a session whose PRIMARY transcript
    signal is unavailable (the manually-tmux-launched ``idle=?`` class) may
    still be reaped via tmux ``session_activity`` when all six gates hold
    (detached + unmapped + no work-descendant + no running pod + no pending
    pane input + over the conservative fallback window); any uncertain signal
    keeps. A once-per-pass beacon logs the resolved detached-set size and a
    loud WARNING when tmux is present but the set is EMPTY (the silent-regression
    guard — the whole reason the pass went inert).

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
    _gc_orphan_last_mapped_terminal(live_sids, dry_run)  # #720 breadcrumb GC
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
    # The orphaned-tmux-server widening gate, evaluated ONCE per pass (not per
    # session): a wrapper on a socket-deleted + zero-client tmux server is not a
    # live-user tty and falls through to the transcript-idle check.
    check_orphaned = _orphaned_tmux_reap_enabled()
    # Compute the detached-tmux-pane set AND the session_activity map ONCE per
    # pass (one tmux call), not per candidate — a wrapper whose controlling tty
    # is a detached pane is not a live-user tty and falls through to the
    # transcript-idle check; the activity map feeds the #695 fallback.
    detached_tmux_ttys, tmux_activity = _detached_tmux_panes_with_activity()
    # Phase-1 beacon (permanent): the once-per-pass detached-set size, plus a
    # LOUD WARNING when tmux is present but the detached set is EMPTY — the
    # silent-regression guard for the inert-pass bug this fix closes (an empty
    # set means the detached-tmux refinement reaped nothing this pass).
    if shutil.which("tmux") is not None and not detached_tmux_ttys:
        print(
            "  idle-unmapped: WARNING tmux present but detached set empty — "
            "refinement inert this pass (no detached panes resolved; the "
            "idle-unmapped reap + #695 fallback both depend on this set)",
            file=sys.stderr,
        )
    print(
        f"idle-unmapped: {len(candidates)} EPS session(s) scanned "
        f"({skipped_pm} PM-registered + {skipped_non_eps} non-EPS skipped; "
        f"detached_panes={len(detached_tmux_ttys)}; "
        f"reap={'ON' if reap else 'OFF — alert-only (EPM_UNMAPPED_IDLE_REAP=0)'}; "
        f"orphaned_tmux_reap={'ON' if check_orphaned else 'OFF (EPM_ORPHANED_TMUX_REAP=0)'})"
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
            detached_tmux_ttys=detached_tmux_ttys,
            tmux_activity=tmux_activity,
            check_orphaned=check_orphaned,
        )


# ─── boot-death pass (#1267) ──────────────────────────────────────────────────
#
# The die-AT-OR-BEFORE-turn-1 lane, TWO ARMS:
#
# - ARM 1 (zero-response, #1267): a freshly `--auto`-dispatched session whose
#   resolved Claude transcript contains ZERO response rows
#   (`_classify_wedge_row` not in {assistant, api-error}) >= 30 min after
#   `spawned_at`. Incident #1251-#1256: 7 live-captured boot-death
#   transcripts, all 9 rows / 11,368 B, frozen ~7 s post-spawn (the session
#   died during `/issue` skill load — one prompt row carries a
#   `<local-command-stderr>` diagnostic).
# - ARM 2 (boot-refusal, #1287): the transcript's 256 KB TAIL
#   (`_transcript_tail_rows`) segments via `_segment_wake_turns` (#1127) to
#   >= 1 completed turn with EVERY completed turn failed (last response row
#   api-error) — the refusal-killed boot turn. Incident #1277: an 826 KB
#   transcript whose boot turn took real assistant actions for ~74 s, then
#   died on a refusal BEFORE the /issue tick cron was armed; arm 1 keeps
#   twice over (assistant rows exist; size over the whole-file cap), and at
#   a PARK status the #1209/#1127 wedge lanes are structurally ineligible
#   (`respawn_eligible = in_active and ...`), so this lane owns the shape.
#
# Both arms: transcript quiet >= 10 min, a LIVE sid, auto registrations
# only. Every existing lane is structurally blind (no self-report => the
# stalled detector skips; the sid is LIVE + status `proposed` =>
# crash-recovery PARKs; the inner Claude is alive-idle => zombie pass;
# issue-MAPPED => idle-unmapped), so the only recovery was the 12h
# stale-registration unregister — a silent dispatch -> boot-death -> 12h ->
# re-dispatch loop (~12.5h/cycle).
#
# BY-DESIGN misses (fall back to the 12h sweep — a future incident in either
# shape reads as designed-miss, not regression): (a) a refusal death whose
# transcript ends in a trailing non-api-error assistant row (the last
# completed turn reads `ok` => arm 2 keeps); (b) a >256 KB ZERO-RESPONSE
# transcript (arm 1's whole-file guarantee cannot be established, and arm 2
# sees no completed turn to fail).
#
# Action: STOP the session via the existing `_stop_session` — NO unregister,
# NO direct spawn. Post-stop re-drive is fully owned by existing arms:
# ACTIVE status -> crash-recovery `decide()` (dead sid, ~2 misses = ~20 min);
# `proposed` -> the proposed-infra sweep's stale-dead-registration grace
# (~30-60 min). Bounded per #1241 conventions: per-issue per-UTC-day stop cap
# (default 3, `EPM_BOOT_DEATH_STOPS_PER_DAY`), counted at STOP-INITIATION (a
# stop failure still consumes a budget unit — conservative in the safe
# direction); at the cap the lane stops stopping and fires ONE loud cap
# push/marker per day (the recorded deviation from #1241's quiet-at-cap —
# see _BOOT_DEATH_CAP_NOTE_SENTINEL). NO episode belt by design: this is a
# STOP lane, not a respawn lane — the downstream re-drive arms carry their
# own belts/caps. Kill switch: EPM_DISABLE_BOOT_DEATH_PASS=1.


def _boot_death_window_s() -> float:
    """Boot-death registration-age window in seconds (env
    ``EPM_BOOT_DEATH_WINDOW_MIN``, MINUTES — the
    ``EPM_TICK_WEDGE_DEAD_SILENCE_MIN`` precedent; default
    :data:`BOOT_DEATH_WINDOW_S`). Malformed / non-positive env falls back to
    the default — a typo'd var must not create an instant stopper; the
    lane's kill switch is ``EPM_DISABLE_BOOT_DEATH_PASS=1``, never this
    knob (the :func:`_stale_registration_idle_s` contract)."""
    raw = os.environ.get("EPM_BOOT_DEATH_WINDOW_MIN")
    if not raw:
        return float(BOOT_DEATH_WINDOW_S)
    try:
        parsed = float(raw) * 60.0
    except ValueError:
        return float(BOOT_DEATH_WINDOW_S)
    if parsed <= 0:
        return float(BOOT_DEATH_WINDOW_S)
    return parsed


def _boot_death_stops_per_day() -> int:
    """Daily per-issue cap on boot-death stops (env
    ``EPM_BOOT_DEATH_STOPS_PER_DAY``; default
    :data:`BOOT_DEATH_STOPS_PER_DAY`). Malformed OR ``< 1`` env falls back
    to the default — never a kill switch (byte-parallel to
    :func:`_tick_wedge_respawns_per_day`, #1241)."""
    raw = os.environ.get("EPM_BOOT_DEATH_STOPS_PER_DAY")
    if not raw:
        return BOOT_DEATH_STOPS_PER_DAY
    try:
        parsed = int(raw)
    except ValueError:
        return BOOT_DEATH_STOPS_PER_DAY
    if parsed < 1:
        return BOOT_DEATH_STOPS_PER_DAY
    return parsed


def _boot_death_pass_enabled() -> bool:
    """Kill switch: False when ``EPM_DISABLE_BOOT_DEATH_PASS`` is set truthy
    ("1"/"true"/"yes", case-insensitive). Default enabled. Mirrors
    :func:`_capacity_retry_enabled`."""
    raw = os.environ.get("EPM_DISABLE_BOOT_DEATH_PASS", "").strip().lower()
    return raw not in {"1", "true", "yes"}


def _boot_death_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{BOOT_DEATH_STATE_PREFIX}{issue}.json"


def _load_boot_death_state(issue: int) -> dict:
    """Per-issue boot-death day-cap state (``{}`` if absent/garbled — a fresh
    or unreadable file re-arms the counter at 0, mirroring
    :func:`_load_idle_unmapped_state`; a corruption costs at most one extra
    bounded stop)."""
    path = _boot_death_state_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_boot_death_state(issue: int, state: dict, dry_run: bool) -> None:
    """Persist the per-issue boot-death state atomically (temp + rename).
    Fail-soft: an OSError is logged and swallowed (main() has no per-pass
    exception wrapping — a raise here would kill every later pass); dry-run
    never writes (production state must not mutate under ``--dry-run``)."""
    if dry_run:
        print(f"  [dry-run] would save boot-death state for #{issue}: {state}")
        return
    dest = _boot_death_state_path(issue)
    tmp = dest.with_suffix(".json.tmp")
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(dest)
    except OSError as e:
        print(f"  WARNING: saving boot-death state for #{issue} failed: {e}", file=sys.stderr)


def _append_boot_death_event(note: str, dry_run: bool) -> None:
    """Durable trace for boot-death stops / cap alerts — one JSON line per
    action in ``~/.eps-autonomous/boot-death-events.jsonl`` (the
    ``_append_stale_registration_event`` shape; the ``.jsonl`` suffix keeps
    it out of the GC's ``boot-death-*.json`` glob). Fail-soft; dry-run never
    writes."""
    dest = AUTONOMOUS_REGISTRY_DIR / "boot-death-events.jsonl"
    line = json.dumps(
        {"ts": datetime.now().astimezone().isoformat(), "kind": "boot-death", "note": note}
    )
    if dry_run:
        print(f"  [dry-run] would append boot-death event to {dest}")
        return
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as e:
        print(f"  WARNING: appending boot-death event failed: {e}", file=sys.stderr)


def _boot_death_stderr_excerpt(rows: list[dict], cap: int = 200) -> str | None:
    """Best-effort diagnostic: the first ``<local-command-stderr>`` fragment
    found in a prompt-type user row's content, whitespace-collapsed and
    bounded to ``cap`` chars — the live-captured boot-death transcripts carry
    the skill-load failure there (`<local-command-stderr>Error: Shell command
    failed...`). Returns ``None`` when no such fragment exists. Pure
    string-scanning, never raises on the dict shapes ``rows`` can hold —
    a text block whose ``"text"`` value is present but non-str (``null``,
    an int) is skipped as carrying no diagnostic text."""
    tag = "<local-command-stderr>"
    for row in rows:
        if _classify_wedge_row(row) != "prompt":
            continue
        msg = row.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                b["text"]
                for b in content
                if isinstance(b, dict)
                and b.get("type") == "text"
                and isinstance(b.get("text"), str)
            )
        for text in texts:
            idx = text.find(tag)
            if idx < 0:
                continue
            excerpt = " ".join(text[idx:].split())
            return excerpt[:cap]
    return None


def _boot_death_api_error_excerpt(rows: list[dict], cap: int = 200) -> str | None:
    """Best-effort forensic (#1287 arm 2): the LAST ``"api-error"``-classified
    row's message text — the refusal / API-error body that killed the boot
    turn — whitespace-collapsed and bounded to ``cap`` chars. SIDECAR-ONLY by
    design (refusal bodies are trigger-dense text; the excerpt never enters
    the task marker or the Telegram push — #866/#1073/#1098 containment).
    Returns ``None`` when no api-error row carries usable text. Pure
    string-scanning, never raises on the dict shapes ``rows`` can hold — the
    :func:`_boot_death_stderr_excerpt` defensive contract: a
    present-but-non-str ``"text"`` value (``null``, an int) is skipped as
    carrying no diagnostic text. The FIRING predicate never depends on this
    helper — it is forensic only."""
    for row in reversed(rows):
        if _classify_wedge_row(row) != "api-error":
            continue
        msg = row.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                b["text"]
                for b in content
                if isinstance(b, dict)
                and b.get("type") == "text"
                and isinstance(b.get("text"), str)
            )
        for text in texts:
            excerpt = " ".join(text.split())
            if excerpt:
                return excerpt[:cap]
    return None


def _process_boot_death(path: Path, pids_by_sid: dict[str, int], now: float, dry_run: bool) -> None:
    """Evaluate ONE auto registration against the boot-death predicate and
    STOP its session when it verdicts ``"stop"`` (cap permitting). Every
    unresolvable input fails toward keep, and every IO goes through a
    fail-soft helper — main() has no per-pass exception wrapping, so nothing
    here may raise (the ``_process_stale_registration`` containment style)."""
    try:
        entry = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return  # garbled entries are the crash-recovery / GC passes' property
    if not isinstance(entry, dict):
        return
    issue = entry.get("issue")
    if not isinstance(issue, int) or isinstance(issue, bool):
        issue = _gc_parse_issue_from_path(path, "issue-", "")
    if issue is None:
        return
    sid = entry.get("happy_session_id")
    if not isinstance(sid, str) or not sid:
        return
    pid = pids_by_sid.get(sid)
    spawned_at = entry.get("spawned_at")
    entry_age_s = None
    if isinstance(spawned_at, int | float) and not isinstance(spawned_at, bool) and spawned_at:
        entry_age_s = now - float(spawned_at)
    # Cheap early exits BEFORE any transcript IO: a dead sid is the
    # crash-recovery pass's property; a young/unaged entry can't fire.
    if pid is None or entry_age_s is None or entry_age_s < _boot_death_window_s():
        return
    # Activity guards (both fail toward keep): something owns this issue.
    if _provision_in_flight_reason(issue, now) is not None:
        return
    if _worktree_recent_activity(issue, now, _wt_activity_fresh_s()):
        return
    rows, transcript, size = _boot_death_transcript_rows(pid)
    response_row_seen = (
        None
        if rows is None
        else any(_classify_wedge_row(r) in ("assistant", "api-error") for r in rows)
    )
    # ARM 2 (#1287): turn-segmented all-failed read over the 256 KB TAIL —
    # works at ANY transcript size (#1277's transcript was 825,591 B, over
    # arm 1's whole-file cap). Reuse the whole-file rows when they resolved
    # (tail == whole file at <= cap; saves the second read); else seek-tail.
    tail_rows = (
        rows if rows is not None else _transcript_tail_rows(pid, max_bytes=BOOT_DEATH_TAIL_BYTES)
    )
    turns = None if tail_rows is None else _segment_wake_turns(tail_rows)
    all_turns_failed = (
        None if turns is None else bool(turns) and all(o == "failed" for o, _ts in turns)
    )
    idle_s, _why = _transcript_idle_age_s(pid, now)
    state = _load_boot_death_state(issue)
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))  # the #1209/#1241 day-cap derivation
    stops_today = _day_scoped_count(state, "stop_day", "stops_today", day_key)
    cap = _boot_death_stops_per_day()
    verdict = decide_boot_death(
        sid_alive=True,
        entry_age_s=entry_age_s,
        response_row_seen=response_row_seen,
        all_turns_failed=all_turns_failed,
        transcript_idle_s=idle_s,
        window_s=_boot_death_window_s(),
        quiet_s=BOOT_DEATH_QUIET_S,
        stops_today=stops_today,
        stops_per_day=cap,
    )
    if verdict == "keep":
        return
    status = _task_status(issue)  # live read, recorded in the note
    if verdict == "cap-alert":
        if state.get("cap_alerted_day") == day_key:
            return  # once per (issue, UTC day)
        state["cap_alerted_day"] = day_key
        _save_boot_death_state(issue, state, dry_run)
        note = (
            f"{_BOOT_DEATH_CAP_NOTE_SENTINEL} {stops_today} boot-death stops today hit the "
            f"daily cap ({cap}); leaving the dead registration in place (the 12h "
            f"stale-registration pass is the back-pressure); sessions for #{issue} are dying "
            f"at or just after boot — skill-load death or refused boot turn — investigate "
            f"the dispatch path. sid={sid} task status={status}."
        )
        print(f"  boot-death: issue #{issue} — {note}")
        _post_progress_marker(issue, note, dry_run, label="boot-death-cap-exhausted")
        _telegram_push(
            f"boot-death cap: #{issue} hit {stops_today} dead-boot stops today; "
            f"sessions are dying at or just after boot — investigate the dispatch path.",
            dry_run,
        )
        _append_boot_death_event(note, dry_run)
        return
    # verdict == "stop": count at STOP-INITIATION (#1241 — conservative; a
    # stop failure still consumes a budget unit), THEN act.
    # NOTE: a session stopped by an EARLIER arm in this same tick can still
    # read live in `pids_by_sid` (the shared snapshot predates that stop), so
    # a redundant sid-targeted stop here may consume a cap unit — benign by
    # design (never a wrong kill; the daily cap absorbs it).
    state.update({"stop_day": day_key, "stops_today": stops_today + 1})
    _save_boot_death_state(issue, state, dry_run)
    stop_ok = _stop_session(sid, dry_run)
    # Arms are mutually exclusive (zero response rows => zero completed
    # turns), so arm 1 owning the tag when it fired is unambiguous.
    shape = "zero-response" if response_row_seen is False else "boot-refusal"
    if shape == "zero-response":
        evidence = (
            f"transcript rows={len(rows)} size={size}B with ZERO response rows "
            f"(assistant/api-error)"
        )
    else:
        n_tail = len(tail_rows) if tail_rows is not None else 0
        n_turns = len(turns) if turns else 0
        api_error_rows = sum(1 for r in (tail_rows or []) if _classify_wedge_row(r) == "api-error")
        evidence = (
            f"256KB-tail rows={n_tail} (file size={size}B): {n_turns} completed "
            f"turn(s), ALL failed ({api_error_rows} api-error row(s) — "
            f"refusal-killed boot turn, #1287)"
        )
    note = (
        f"{_BOOT_DEATH_STOP_NOTE_SENTINEL} stopped boot-dead session sid={sid}: "
        f"registration age {entry_age_s / 60:.0f}m >= {_boot_death_window_s() / 60:.0f}m, "
        f"shape={shape}, {evidence}, idle {(idle_s or 0) / 60:.0f}m; stop_ok={stop_ok}; "
        f"task status={status}; stop {stops_today + 1}/{cap} today; registration kept: "
        f"crash-recovery re-drives an ACTIVE task (~20 min); the proposed-infra sweep's "
        f"stale-dead-registration grace re-dispatches a `proposed` task (~30-60 min)."
    )
    print(f"  boot-death: issue #{issue} — {note}")
    _post_progress_marker(issue, note, dry_run, label="boot-death-stop")
    _telegram_push(
        f"boot-death: stopped dead-boot session for #{issue} (shape={shape} at "
        f"{entry_age_s / 60:.0f}m; stop {stops_today + 1}/{cap} today)",
        dry_run,
    )
    # The refusal excerpt is SIDECAR-ONLY by design: refusal bodies are
    # trigger-dense text, so it never enters the task marker or the push
    # (#866/#1073/#1098 containment).
    stderr_excerpt = _boot_death_stderr_excerpt(tail_rows) if tail_rows else None
    api_error_excerpt = _boot_death_api_error_excerpt(tail_rows) if tail_rows else None
    _append_boot_death_event(
        f"{note} transcript={transcript} stderr_excerpt={stderr_excerpt or 'none'} "
        f"api_error_excerpt={api_error_excerpt or 'none'}",
        dry_run,
    )


def boot_death_pass(
    dry_run: bool, *, children: list[dict] | None, now: float | None = None
) -> None:
    """Stop LIVE-but-boot-dead auto sessions (#1267 — see the section comment
    above for the incident + predicate). Consumes the shared reaper
    ``children`` snapshot (``_live_children``) IN PLACE; daemon-gated
    (``children is None`` => no-op: liveness cannot be established, and a
    false "live" read must not stop anything). Iterates ``issue-*.json``
    ONLY — ``manual-issue-*.json`` is EXCLUDED by design (a user-driven
    session is never auto-stopped, the #505 posture; auto registrations all
    carry an initial prompt, so zero response rows — or an all-failed
    completed-turn tail (#1287) — at 30 min is unambiguous death)."""
    now = now if now is not None else time.time()
    if not _boot_death_pass_enabled():
        print("boot-death: disabled via EPM_DISABLE_BOOT_DEATH_PASS; skipping")
        return
    if children is None:
        print("boot-death: Happy daemon unreachable; skipping")
        return
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        print("boot-death: no autonomous registry dir; skipping")
        return
    entries = sorted(AUTONOMOUS_REGISTRY_DIR.glob("issue-*.json"))
    if not entries:
        print("boot-death: no auto issue registrations")
        return
    pids_by_sid: dict[str, int] = {}
    for c in children:
        if not isinstance(c, dict):
            continue
        sid = c.get("happySessionId")
        pid = c.get("pid")
        if isinstance(sid, str) and sid and isinstance(pid, int) and not isinstance(pid, bool):
            pids_by_sid[sid] = pid
    print(f"boot-death: {len(entries)} auto registration(s), {len(pids_by_sid)} live session(s)")
    for path in entries:
        _process_boot_death(path, pids_by_sid, now, dry_run)


# ─── stale-registration pass (#845 d) ─────────────────────────────────────────
#
# The fourth registration hygiene arm: a LIVE-but-abandoned session whose
# registration (issue-<N>.json OR manual-issue-<N>.json) still maps its issue.
# Incident #665: a session sat transcript-idle for 16h while its registration
# held the `/issue` Step 0 single-orchestrator guard — every re-drive detected
# the "live" owner and exited, so the task sat unworked until manual triage.
# The crash-recovery pass can't help (the sid IS live); the idle-unmapped
# reaper deliberately excludes MAPPED sessions; session-reconcile only fires
# on parked/terminal task statuses. This pass closes the square: UNREGISTER
# (delete the registration file — never stop the session: a manual session
# may hold a user TTY, and the SKILL Step 0 stale-wake ownership re-check
# guards a later wake) so the Step 0 guard is released and, for an ACTIVE
# task, the registration-independent orphan sweep re-drives it on its next
# tick. A PARK-status task (plan_pending / awaiting_promotion / blocked /
# terminal) is deliberately NOT re-driven by the orphan sweep — by design.
#
# Threshold: equals the idle-unmapped reap window (12h) — the project's
# existing "a transcript idle this long is abandoned" judgment; catches #665
# (16h) with margin, and is far above any legitimate gate-wait TURN gap
# (gates park the task STATUS, which the unregistered task keeps carrying).
# Guards, all failing toward keep: dead sid (crash-recovery property),
# unresolvable transcript, in-flight provision, fresh worktree activity,
# fresh self-report. Unregistering deletes the entry, which is self-deduping;
# a fresh re-registration restarts the clock.

STALE_REGISTRATION_IDLE_S = UNMAPPED_IDLE_REAP_S  # 12h — same abandonment judgment


def _stale_registration_idle_s() -> float:
    """Stale-registration idle threshold in seconds (env
    ``EPM_STALE_REGISTRATION_IDLE_H``, HOURS; default
    :data:`STALE_REGISTRATION_IDLE_S`). Malformed / non-positive env falls
    back — a typo'd var must not turn the pass into an instant unregisterer."""
    raw = os.environ.get("EPM_STALE_REGISTRATION_IDLE_H")
    if not raw:
        return float(STALE_REGISTRATION_IDLE_S)
    try:
        parsed = float(raw) * 3600.0
    except ValueError:
        return float(STALE_REGISTRATION_IDLE_S)
    if parsed <= 0:
        return float(STALE_REGISTRATION_IDLE_S)
    return parsed


def _append_stale_registration_event(note: str, dry_run: bool) -> None:
    """Durable trace for stale-registration unregisters — one JSON line per
    action in ``~/.eps-autonomous/stale-registration-events.jsonl`` (same
    shape + role as the idle-unmapped events file). The per-task marker is
    the primary record; this file survives a task folder move. Fail-soft."""
    dest = AUTONOMOUS_REGISTRY_DIR / "stale-registration-events.jsonl"
    line = json.dumps(
        {"ts": datetime.now().astimezone().isoformat(), "kind": "stale-registration", "note": note}
    )
    if dry_run:
        print(f"  [dry-run] would append stale-registration event to {dest}")
        return
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as e:
        print(f"  WARNING: appending stale-registration event failed: {e}", file=sys.stderr)


def _process_stale_registration(
    path: Path, pids_by_sid: dict[str, int], now: float, dry_run: bool
) -> None:
    """Evaluate ONE registration file against the stale-registration
    predicate and unregister it when it verdicts ``"unregister"``. Every
    unresolvable input fails toward keep (a wrong unregister would strip a
    dead session's crash-recovery coverage or double-drive a live one)."""
    try:
        entry = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return  # garbled entries are the crash-recovery / GC passes' property
    if not isinstance(entry, dict):
        return
    issue = entry.get("issue")
    if not isinstance(issue, int):
        issue = _gc_parse_issue_from_path(
            path, "manual-issue-" if path.name.startswith("manual-issue-") else "issue-", ""
        )
    if issue is None:
        return
    sid = entry.get("happy_session_id")
    if not isinstance(sid, str) or not sid:
        return
    pid = pids_by_sid.get(sid)
    if pid is None:
        return  # dead sid: stays registered so the crash-recovery pass can act
    # Activity guards (both cheap, both fail toward keep): an in-flight
    # provision or fresh worktree edits mean the session is WORKING, however
    # idle its transcript looks from here.
    if _provision_in_flight_reason(issue, now) is not None:
        return
    if _worktree_recent_activity(issue, now, _wt_activity_fresh_s()):
        return
    idle_s, why = _transcript_idle_age_s(pid, now)
    if idle_s is None:
        print(
            f"  stale-registration: issue #{issue} sid={sid} transcript "
            f"unresolvable ({why}); keeping (fail toward keep)"
        )
        return
    self_report_age, _ts = _self_report_age_seconds(issue, now)
    threshold_s = _stale_registration_idle_s()
    verdict = decide_stale_registration(
        sid_alive=True,
        transcript_idle_s=idle_s,
        self_report_age_s=self_report_age,
        idle_threshold_s=threshold_s,
    )
    if verdict != "unregister":
        return
    kind = "manual" if path.name.startswith("manual-issue-") else "auto"
    status = _task_status(issue)
    note = (
        f"{_STALE_REGISTRATION_NOTE_SENTINEL} unregistered {kind} registration: "
        f"session sid={sid} transcript-idle {idle_s / 3600:.1f}h >= "
        f"{threshold_s / 3600:.1f}h (self-report equally stale); task "
        f"status={status}; Step 0 guard released; the orphan sweep re-drives "
        f"the task if it is ACTIVE (a PARK/terminal-status task is deliberately "
        f"not re-driven). The session itself was NOT stopped."
    )
    print(f"  stale-registration: issue #{issue} — {note}")
    if not dry_run:
        path.unlink(missing_ok=True)
    _post_progress_marker(issue, note, dry_run, label="stale-registration-unregister")
    _append_stale_registration_event(note, dry_run)


def stale_registration_pass(
    dry_run: bool, *, children: list[dict] | None, now: float | None = None
) -> None:
    """Unregister LIVE-but-abandoned session registrations (#845 d — see the
    section comment above for the incident + predicate). Consumes the shared
    reaper ``children`` snapshot (``_live_children``) IN PLACE; daemon-gated
    (``children is None`` => no-op: liveness cannot be established, and a
    false "dead" read must not strip a registration)."""
    now = now if now is not None else time.time()
    if children is None:
        print("stale-registration: Happy daemon unreachable; skipping")
        return
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        print("stale-registration: no autonomous registry dir; skipping")
        return
    entries = sorted(AUTONOMOUS_REGISTRY_DIR.glob("issue-*.json")) + sorted(
        AUTONOMOUS_REGISTRY_DIR.glob("manual-issue-*.json")
    )
    if not entries:
        print("stale-registration: no issue registrations")
        return
    pids_by_sid: dict[str, int] = {}
    for c in children:
        if not isinstance(c, dict):
            continue
        sid = c.get("happySessionId")
        pid = c.get("pid")
        if isinstance(sid, str) and sid and isinstance(pid, int) and not isinstance(pid, bool):
            pids_by_sid[sid] = pid
    print(f"stale-registration: {len(entries)} registration(s), {len(pids_by_sid)} live session(s)")
    for path in entries:
        _process_stale_registration(path, pids_by_sid, now, dry_run)


def pod_safety_pass(dry_run: bool, threshold: int, now: float | None = None) -> None:
    """Reconcile RUNNING managed pods against their task STATUS.

    - AUTO-STOP (reversible, never terminate) a RUNNING pod whose task is DONE
      or user-paused (:data:`POD_SAFETY_AUTO_STOP` — #980: ``on_hold`` + RUNNING
      means the #919 pause teardown leg failed), after the 2-miss guard — an
      escaped pod.
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
    running_issues = {issue for issue, _pod_id, _name, _info in running}

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
    for issue, pod_id, _name, info in running:
        _process_pod(issue, pod_id, info, now, dry_run, threshold)


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
    # Sub-floor sentinel (task #679): an EARLIER advisory band (~60 GB) that
    # attributes the disk pressure to the largest per-issue caches on the
    # shared disk-guard sidecar — runs every tick, warn-only, never deletes.
    subfloor_sentinel_pass(dry_run, free_bytes=free)
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
    _forward_marker_child_stderr(res, "spawn_session spawn-campaign (campaign)")
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  RESPAWNED campaign #{issue}: {first_line}")
    # #1027: campaign respawns feed the trigger's event stream too (arm
    # "campaign"; gated at BOTH callers, never a canary — campaign sessions
    # register at campaign-<N>.json, so canary liveness could not be read).
    _auth_outage_record_spawn(issue, "campaign", _coerce_ts(entry.get("spawned_at")))
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
        # Count ANY non-watcher marker (NOT just _PROGRESS_KINDS): a child in a
        # long pre-pod planning/implementation phase posts only excluded
        # lifecycle markers, so the narrow helper read "no fresh child" and
        # over-alerted epm:campaign-stalled. Matches the docstring ("ANY epm:
        # marker") and the stalled-detector + reconcile + orphan passes
        # (#661/#658 sibling).
        latest = _latest_nonwatcher_event_ts(events)
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
        res = subprocess.run(
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
        _forward_marker_child_stderr(res, f"{kind} on #{issue}")
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
    # #1027 (§13.1): gate BEFORE the stop so a suppressed tick skips the
    # stop+respawn as a UNIT (never stop-then-not-respawn, which would leave
    # the campaign session dead for the episode). The campaign arm never
    # consumes the canary token (arm "campaign" is outside
    # _AUTH_OUTAGE_CANARY_ARMS).
    if _auth_outage_spawn_gate(issue, "campaign", dry_run=dry_run) is not None:
        print(f"  campaign #{issue}: stalled stop+respawn SKIPPED — auth-outage episode active")
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
        # #1027 (§13.1): the crash-arm caller ignores the bool result, so the
        # gate sits here (books nothing; re-evaluated next tick).
        if _auth_outage_spawn_gate(issue, "campaign", dry_run=dry_run) is not None:
            print(f"  campaign #{issue}: crash respawn SKIPPED — auth-outage episode active")
        else:
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
        issue
        for issue, _pod_id, _name, _info in (_running_managed_issue_pods(caller="gate-push") or [])
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


def program_orchestrator_pass(
    dry_run: bool,
    *,
    script: Path | None = None,
    stop: Path | None = None,
    log: Path | None = None,
    runner=subprocess.run,
    env: dict | None = None,
) -> None:
    """Crash-recover the leakage-program (#660) bash meta-loop daemon.

    The per-phase ``/issue --auto`` sessions are crash-recovered by the respawn
    pass; the ``scripts/run_program_orchestrator.sh`` daemon that SEQUENCES the
    phases (1 -> 2 -> 3 -> 4, spawning each ``/issue --auto`` and advancing on the
    critic-gated PASS) is a single bash process in tmux ``eps-program`` and is NOT
    otherwise recovered. If it dies (VM reboot, OOM-kill) mid-program, phase
    ADVANCEMENT stops: the active phase keeps running + parks, but nothing spawns
    the next.

    Relaunch iff ALL hold (fail toward NOT relaunching on any missing signal):
      - the daemon is not already alive (``pgrep -f``);
      - the STOP sentinel is absent (a STOP = deliberate halt; every gate/phase
        HALT path ``touch``es it);
      - the log shows no deliberate exit: neither "Program complete" (normal end)
        nor "finished WITH HALTS" (the two deliberate exits that leave no STOP).

    Relaunch is idempotent — a fresh daemon re-checks every phase status and will
    not double-spawn an active/terminal phase. Kill switch:
    ``EPM_DISABLE_PROGRAM_ORCHESTRATOR_RECOVERY=1``.
    """
    env = os.environ if env is None else env
    if env.get("EPM_DISABLE_PROGRAM_ORCHESTRATOR_RECOVERY") == "1":
        return

    script = script or (PROJECT_ROOT / "scripts" / "run_program_orchestrator.sh")
    if not script.exists():
        return  # daemon retired / not set up -> nothing to recover

    stop = stop or (PROJECT_ROOT / ".claude" / "cache" / "program_orchestrator.STOP")
    log = log or (PROJECT_ROOT / ".claude" / "cache" / "program_orchestrator.log")

    try:
        alive = (
            runner(
                ["pgrep", "-f", "run_program_orchestrator.sh"],
                capture_output=True,
                timeout=15,
            ).returncode
            == 0
        )
    except (subprocess.SubprocessError, OSError):
        print("program-orchestrator: pgrep failed; skipping (fail-safe)")
        return
    if alive:
        return

    if stop.exists():
        print(
            "program-orchestrator: down but STOP sentinel present -> leaving down (deliberate halt)"
        )
        return

    try:
        # The daemon writes its terminal phrase as the LAST log() call before it
        # exits (then it is dead, nothing writes after), so the final 4 KiB always
        # captures a deliberate-exit phrase if one was written.
        tail = ""
        if log.exists():
            with log.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 4096))
                tail = fh.read().decode("utf-8", "replace")
        if ("Program complete" in tail) or ("finished WITH HALTS" in tail):
            print("program-orchestrator: down; log shows a deliberate exit -> leaving down")
            return
    except OSError:
        print("program-orchestrator: down but log unreadable; skipping (fail-safe)")
        return

    if dry_run:
        print(
            "program-orchestrator: down + no STOP + no deliberate exit -> WOULD relaunch (dry-run)"
        )
        return

    try:
        runner(["tmux", "kill-session", "-t", "eps-program"], capture_output=True, timeout=15)
        res = runner(
            ["tmux", "new-session", "-d", "-s", "eps-program", f"bash {shlex.quote(str(script))}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode == 0:
            print("program-orchestrator: RELAUNCHED (was down, in flight, no STOP)")
        else:
            print(
                f"program-orchestrator: relaunch FAILED rc={res.returncode}: "
                f"{res.stderr.strip()[:200]}"
            )
    except (subprocess.SubprocessError, OSError) as e:
        print(f"program-orchestrator: relaunch errored: {e}")


def _vm_ledger_reap_disabled(env: dict | None = None) -> bool:
    """Kill switch: True when ``EPM_DISABLE_VM_LEDGER_REAP`` is set truthy."""
    env = os.environ if env is None else env
    return env.get("EPM_DISABLE_VM_LEDGER_REAP", "").strip().lower() in ("1", "true", "yes", "on")


def vm_ledger_reap_pass(dry_run: bool) -> None:
    """Reap expired-TTL / dead-PID rows from the advisory VM resource ledger (plan §5).

    The ledger (``scripts/resource_ledger.py``, ``~/.task-workflow/vm-ledger.json``)
    routes CPU/RAM-heavy phases off the shared VM; a crashed session's claim is
    TTL- + PID-reaped here so a dead claim can never wedge routing. Piggybacks
    the 10-min tick. Daemon-INDEPENDENT (a local file only), so it runs on a
    daemon outage too. Fail-soft: any error (incl. a psutil-less-host import
    failure) is logged and swallowed — a ledger hiccup never crashes the
    watcher. ``--dry-run`` reports without mutating. Kill switch:
    ``EPM_DISABLE_VM_LEDGER_REAP=1``.
    """
    if _vm_ledger_reap_disabled():
        print("  vm-ledger-reap: disabled via EPM_DISABLE_VM_LEDGER_REAP; skipping")
        return
    try:
        import resource_ledger  # lazy: a psutil-less host must not crash the watcher

        reaped = resource_ledger.reap_ledger_file(apply=not dry_run)
        if reaped:
            phases = ", ".join(
                f"#{r.get('issue')}:{r.get('phase')} (claim {r.get('claim_id')})" for r in reaped
            )
            verb = "WOULD reap" if dry_run else "reaped"
            print(f"  vm-ledger-reap: {verb} {len(reaped)} stale claim(s): {phases}")
        else:
            print("  vm-ledger-reap: no stale claims")
    except Exception as exc:
        print(f"  vm-ledger-reap: error (skipping): {exc}")


@session_resolver.transcript_resolution_scope()
def main(argv: list[str] | None = None) -> int:  # noqa: C901 — flat --*-only dispatch ladder + linear pass sequence; the #1170 flag adds a guard branch, not nesting
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
    parser.add_argument(
        "--capacity-retry-only",
        action="store_true",
        help="run ONLY the capacity-retry pass (re-drive blocked-on-transient-"
        "infra tasks, incident #642) and exit; skip every other pass. Mirrors "
        "--infra-drain-only; pair with --dry-run for a live smoke against the "
        "real blocked-task set.",
    )
    parser.add_argument(
        "--stale-blocked-only",
        action="store_true",
        help="run ONLY the stale-blocked flag pass (FLAG a blocked task whose "
        "events show a live healthy run — fresh epm:run-launched + post-launch "
        "progress; the #742 class, task #1021) and exit; skip every other "
        "pass. Flag-only + daemon-independent; pair with --dry-run for a live "
        "smoke against the real blocked-task set.",
    )
    parser.add_argument(
        "--program-orchestrator-only",
        action="store_true",
        help="run ONLY the program-orchestrator crash-recovery pass (relaunch "
        "the #660 leakage-program bash daemon if it died mid-program) and exit; "
        "skip every other pass. Pair with --dry-run for a live smoke.",
    )
    parser.add_argument(
        "--proposed-infra-sweep-only",
        action="store_true",
        help="run ONLY the proposed-infra sweep pass (dispatch ripe ORPHANED "
        "proposed infra/batch tasks the PM never queued, #690) and exit; skip "
        "every other pass. Mirrors --infra-drain-only; pair with --dry-run for "
        "a live smoke against the real proposed-task set.",
    )
    parser.add_argument(
        "--happy-patch-only",
        action="store_true",
        help="run ONLY the Happy injection-patch check pass (escalate on a "
        "reverted/drifted daemon patch, #726) and exit; skip every other pass. "
        "Daemon-independent; pair with --dry-run for a live smoke.",
    )
    parser.add_argument(
        "--cpu-guard-only",
        action="store_true",
        help="run ONLY the CPU/memory-pressure guard pass (#849) and exit; "
        "skip every other pass. Daemon-independent; pair with --dry-run for "
        "a live smoke.",
    )
    parser.add_argument(
        "--triage-observer-only",
        action="store_true",
        help="run ONLY the post-hoc external-marker triage observer pass "
        "(#967, non-gating) and exit; skip every other pass. "
        "Daemon-independent; pair with --dry-run for a live smoke.",
    )
    parser.add_argument(
        "--verdict-disagree-only",
        action="store_true",
        help="run ONLY the verdict-disagree observer pass (#1170, non-gating "
        "— unreconciled doubled-site PASS-vs-FAIL review rounds, the #825 "
        "shape) and exit; skip every other pass. Daemon-independent; pair "
        "with --dry-run for a live smoke.",
    )
    parser.add_argument(
        "--auth-outage-only",
        action="store_true",
        help="run ONLY the auth-outage guard pass (#1027 — fleet respawn "
        "suppression on an Anthropic auth outage) and exit; skip every other "
        "pass. Pair with --dry-run for a live smoke (zero writes, zero "
        "pushes).",
    )
    parser.add_argument(
        "--boot-death-only",
        action="store_true",
        help="run ONLY the boot-death pass (#1267 — stop a dispatched auto "
        "session whose transcript has ZERO response rows >= 30 min after "
        "spawn) and exit; skip every other pass. Pair with --dry-run for a "
        "live smoke against the real registration set.",
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

    # --capacity-retry-only mirrors --infra-drain-only: run the single pass
    # under the lock (it probes the daemon itself) and exit.
    if args.capacity_retry_only:
        capacity_retry_pass(args.dry_run, daemon_reachable=_daemon_reachable())
        return 0

    # --stale-blocked-only mirrors --capacity-retry-only: run the single pass
    # under the lock and exit. Daemon-INDEPENDENT (flag-only; no spawns).
    if args.stale_blocked_only:
        stale_blocked_flag_pass(args.dry_run)
        return 0

    # --program-orchestrator-only mirrors the other --*-only flags: the pass is
    # daemon-independent (a bash daemon, not a Happy session), so run it alone.
    if args.program_orchestrator_only:
        program_orchestrator_pass(args.dry_run)
        return 0

    # --proposed-infra-sweep-only mirrors --infra-drain-only: run the single
    # pass under the lock (it probes the daemon itself) and exit.
    if args.proposed_infra_sweep_only:
        proposed_infra_sweep_pass(args.dry_run, daemon_reachable=_daemon_reachable())
        return 0

    # --happy-patch-only mirrors the other --*-only flags: the pass is
    # daemon-independent (reads a local file only), so run it alone.
    if args.happy_patch_only:
        happy_patch_pass(args.dry_run)
        return 0

    # --cpu-guard-only mirrors the other --*-only flags: the pass is
    # daemon-independent (reads /proc + the earlyoom journal only), so run
    # it alone.
    if args.cpu_guard_only:
        cpu_guard_pass(args.dry_run)
        return 0

    # --triage-observer-only mirrors the other --*-only flags: the pass is
    # daemon-independent (reads the registry + events.jsonl only), so run
    # it alone.
    if args.triage_observer_only:
        triage_observer_pass(args.dry_run)
        return 0

    # --verdict-disagree-only mirrors --triage-observer-only: the pass is
    # daemon-independent (reads the registry + events.jsonl only), so run
    # it alone.
    if args.verdict_disagree_only:
        verdict_disagree_pass(args.dry_run)
        return 0

    # --auth-outage-only mirrors --cpu-guard-only: run the single pass under
    # the lock and exit (episode bookkeeping is daemon-independent; the
    # canary read degrades to "hold" when the daemon is down).
    if args.auth_outage_only:
        reachable = _daemon_reachable()
        auth_outage_pass(
            args.dry_run,
            daemon_reachable=reachable,
            live_ids=_live_session_ids() if reachable else None,
        )
        return 0

    # --boot-death-only mirrors the other --*-only flags: run the single pass
    # under the lock (it needs its own /list snapshot — the pass is
    # daemon-gated) and exit. Pair with --dry-run for a live smoke.
    if args.boot_death_only:
        boot_death_pass(
            args.dry_run,
            children=_live_children() if _daemon_reachable() else None,
        )
        return 0

    # VM disk-headroom: runs FIRST. A full root disk makes every later
    # subprocess in this very watcher (and every VM session) flaky — alert
    # and reclaim before reasoning about sessions/pods (task #552).
    vm_disk_pass(args.dry_run)

    # Data-disk headroom (#681): a SECOND, ESCALATE-ONLY pass on the dedicated
    # /mnt/eps-data mount (the relocated .claude/worktrees/ tree), driving the
    # PERCENT decision helpers so the fire point is size-invariant. Mirrors the
    # vm_disk_guard.main() mount-guarded second pass — a clean no-op before /
    # without the cutover (and on an existing-but-unmounted /mnt/eps-data), so
    # the call is live the instant the disk is actually mounted.
    data_disk_pass(args.dry_run)

    # Happy injection-patch check (#726): a SECOND escalate-only, daemon-
    # INDEPENDENT pass (reads a local file only). The spawn-path guard
    # (spawn_session._verify_happy_patch_or_die) is REACTIVE — it fires only at
    # the next spawn; this PROACTIVE pass surfaces a reverted/drifted patch
    # (typically from `npm update happy`) within ~10 min so the gap is caught
    # before the next autonomous dispatch. Escalate-only (never re-applies —
    # that needs sudo); clean no-op when patched or the daemon file is absent.
    # Placed in the daemon-independent block (next to vm_disk_pass /
    # data_disk_pass / program_orchestrator_pass), BEFORE the daemon-gated
    # session passes — it must run on a daemon outage too.
    happy_patch_pass(args.dry_run)

    # CPU/memory-pressure guard (#849): escalate-only observability for the
    # shared VM's compute pressure + silent earlyoom SIGTERM kills (the
    # 2026-07-02 load-226 incident class). WARN-ONLY — never kills/renices;
    # daemon-independent (reads /proc + the earlyoom journal only), so it
    # runs on a daemon outage too.
    cpu_guard_pass(args.dry_run)

    # Post-hoc external-marker triage observer (#967): NON-GATING audit of
    # the /issue Step 9 pre-dispatch triage duty (origin incident #779) —
    # flags a missing / 'none' triage line against a re-enumerated non-empty
    # candidate window. Observe/alert only (sidecar + deduped push + capped
    # epm:progress nudges); daemon-independent (registry + events.jsonl
    # reads only), so it runs on a daemon outage too.
    triage_observer_pass(args.dry_run)

    # Verdict-disagree observer (#1170): NON-GATING audit of the doubled
    # marker-mode review sites for the #825 misclassification shape — the
    # LATEST round whose Claude + Codex durable verdicts disagree
    # (pass-class vs fail-class) with no role-matched epm:review-reconcile
    # and no Codex no-show evidence. Observe/alert only (sidecar + one
    # deduped push per finding; NO task markers); daemon-independent
    # (registry + events.jsonl reads only), so it runs on a daemon outage
    # too.
    verdict_disagree_pass(args.dry_run)

    # VM resource-ledger reap (plan §5): drop expired-TTL / dead-PID claims from
    # the advisory ~/.task-workflow/vm-ledger.json so a crashed session's claim
    # can never wedge the CPU/RAM off-VM routing decision. Daemon-INDEPENDENT (a
    # local file only) + fail-soft, so it sits in this block next to the other
    # daemon-independent escalate/reap passes and runs on a daemon outage too.
    vm_ledger_reap_pass(args.dry_run)

    # Program-orchestrator crash-recovery: the leakage-program (#660) meta-loop is
    # a bash daemon (run_program_orchestrator.sh in tmux eps-program), NOT a Happy
    # session, so the respawn pass below never covers it. Relaunch it if it died
    # mid-program (no STOP sentinel, no deliberate-exit log line). Daemon-
    # independent (not a Happy session); runs every tick like vm_disk_pass.
    program_orchestrator_pass(args.dry_run)

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
    # #845 (c): the single per-tick probe retries with bounded backoff (3
    # attempts, 5s/10s) so a transient daemon flap doesn't silently defer a
    # whole tick's recovery actions; the --*-only paths keep the bare probe.
    daemon_reachable = _daemon_reachable_with_retry()
    live_ids: set[str] = set()
    if daemon_reachable:
        live_ids = _live_session_ids()

    # Auth-outage guard (#1027): arm/refresh the fleet-level respawn
    # suppression BEFORE ANY spawn arm runs this tick (the crash-recovery
    # loop below is the first spawner). live_ids is hoisted above so the
    # canary-survival read sees this tick's snapshot; episode bookkeeping
    # (trigger / fail-open TTL) is daemon-INDEPENDENT and advances during
    # daemon flaps, while the canary read degrades to "hold" on live_ids
    # unavailability — during a daemon outage no spawn pass runs anyway.
    auth_outage_pass(
        args.dry_run,
        daemon_reachable=daemon_reachable,
        live_ids=live_ids if daemon_reachable else None,
    )

    if daemon_reachable:
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
    stalled_session_pass(
        args.dry_run,
        args.threshold,
        daemon_reachable=daemon_reachable,
        live_ids=live_ids if daemon_reachable else None,
        # #845 (e): the {sid: wrapper pid} map for the prompt-wedge
        # transcript probe (one extra /list RPC, only on daemon-up ticks).
        pids_by_sid=_live_pids_by_sid_or_none() if daemon_reachable else None,
    )

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

    # Proposed-infra sweep (#690): the always-on backstop for ripe ORPHANED
    # `proposed` infra/batch tasks the PM never queued (filed by a context that
    # could not self-dispatch — a pod, a manual `task.py new`, a crashed filer,
    # or a cap-full file-time wrapper). Builds its OWN candidate set from
    # `list-by-status --status proposed`, honors the PM queue's `holds` map, and
    # dispatches into free slots under the SAME shared cap. Runs IMMEDIATELY
    # AFTER infra-drain (load-bearing, pinned by a main()-order test): any ID the
    # drain dispatched THIS tick is already registered, so it counts as
    # `pending` here and the shared cap holds across both. Daemon-gated like
    # every spawning pass.
    proposed_infra_sweep_pass(args.dry_run, daemon_reachable=daemon_reachable)

    # Capacity-retry: re-drive `blocked`-on-transient-infra tasks (incident
    # #642 — a `failure_class: infra` + `reason: no_compute_available` block
    # whose code is ready and whose lanes later free up). The narrow inverse of
    # the respawn pass's PARK rule for `blocked`: it touches ONLY this
    # transient-capacity subclass (every deliberate halt stays parked) and
    # re-drives via `spawn-issue --auto`, which re-runs the backend router's own
    # capacity pre-check (so a still-full lane re-blocks at zero GPU cost) and
    # enforces the plan-approval GPU cap. Backoff + a per-UTC-day cap bound the
    # churn (no watcher-side precheck by design — see the pass's WHY block).
    # Daemon-gated like every spawning pass; runs after infra-drain.
    capacity_retry_pass(args.dry_run, daemon_reachable=daemon_reachable)

    # Stale-blocked flag (task #1021, the #742 class): FLAG — never flip — a
    # `blocked` task whose events show a live healthy run (an
    # `epm:run-launched` NEWER than the transition into `blocked` + fresh
    # post-launch progress). The watcher-side backstop of the SKILL.md
    # "A successful relaunch also reconciles a stale `blocked`" orchestrator
    # rule: one deduped-per-launch-episode marker + sidecar row + Telegram
    # digest line; the status flip stays with the orchestrator/human.
    # Thematically adjacent to capacity-retry (both scan blocked ids) but NOT
    # daemon-gated — it spawns nothing (marker posts go via task.py).
    stale_blocked_flag_pass(args.dry_run)

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

    # The FOUR consumers below (the boot-death pass + the stale-registration
    # pass + the two session reapers) run back-to-back with no mutating pass
    # between them, so they share ONE /list snapshot via their `children=`
    # parameter —
    # same probe-once rationale as daemon_reachable above: one fewer daemon
    # RPC per tick, and the passes can never disagree about the session set.
    # Deliberately NOT reused from the top-of-main `_live_session_ids()`
    # fetch: the respawn / stalled / reconcile / gate-push passes in between
    # mutate the session set, and these consumers should see the
    # post-mutation view. A session the zombie pass stops mid-tick may
    # linger in the shared snapshot for the idle pass; if its wrapper pid is
    # already gone the TTY guard fails toward keep (unreadable /proc -> True
    # -> action "clear"), and if it is still dying the worst case is a
    # redundant, sid-targeted stop of an already-stopped session — never a
    # wrong kill.
    reaper_children = _live_children() if daemon_reachable else None

    # Boot-death (#1267): STOP a freshly dispatched auto session whose
    # transcript has ZERO response rows >= 30 min after spawn (the
    # die-BEFORE-turn-1 class every other lane is structurally blind to).
    # Runs AFTER gate_push_pass (the gate-push-before-reaper ordering
    # invariant) and BEFORE stale-registration, consuming the shared reaper
    # snapshot IN PLACE. A same-tick overlap with stale-registration on a
    # >=12h-old boot-dead entry is benign (our stop + its unregister compose
    # to the desired end state).
    boot_death_pass(args.dry_run, children=reaper_children)

    # Stale-registration (#845 d): unregister LIVE-but-abandoned session
    # registrations (transcript idle >= 12h, self-report equally stale, no
    # in-flight provision / fresh worktree activity — the #665 class that
    # held the /issue Step 0 single-orchestrator guard for 16h). Unregister-
    # only — never stops the session; for an ACTIVE task the orphan sweep
    # re-drives on its next tick. Runs AFTER gate_push_pass (the gate-push-
    # before-reaper ordering is a documented runaway-force-stop invariant)
    # and consumes the shared reaper snapshot IN PLACE.
    stale_registration_pass(args.dry_run, children=reaper_children)

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


def _crash_arm_respawn_or_hold(
    entry: dict, path: Path, issue: object, threshold: int, dry_run: bool
) -> None:
    """The crash-recovery pass's respawn ACTION, gated by the #845 (b)
    bounded worktree-activity hold: fresh file edits under the issue's
    worktree mean an implementer/analyzer (possibly a subagent whose
    parent's registration went stale) is mid-edit; respawning now would
    orphan its work (#812: killed 57s after an edit). While held, `missed`
    is pinned at the threshold so the arm stays ARMED and re-fires the
    moment the activity quiets (or the ~1h hold cap trips — a bound, not a
    latch: the (WT_HOLD_MAX_TICKS+1)th tick respawns regardless)."""
    hold_count = entry.get("wt_hold_count", 0)
    if not isinstance(hold_count, int) or isinstance(hold_count, bool):
        hold_count = 0
    if isinstance(issue, int) and decide_worktree_hold(
        _worktree_recent_activity(issue, time.time(), _wt_activity_fresh_s()), hold_count
    ):
        entry["missed"] = max(entry.get("missed", 0), threshold)
        entry["wt_hold_count"] = hold_count + 1
        print(
            f"  issue #{issue}: HOLD-RESPAWN — worktree activity < "
            f"{_wt_activity_fresh_s() / 60:.0f}m (hold {hold_count + 1}/"
            f"{WT_HOLD_MAX_TICKS}); deferring crash-recovery respawn."
        )
        if not dry_run:
            path.write_text(json.dumps(entry, indent=2))
        return
    _respawn(entry, dry_run)  # rewrites the registry on success


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
    # #759: a registration written within the spawn-grace window is treated as
    # spawn-in-flight (its id may not yet be in the daemon /list reply). A
    # missing/zero spawned_at yields a large age → no grace → today's behavior.
    spawned_at = entry.get("spawned_at", 0)
    entry_age_s = (time.time() - spawned_at) if spawned_at else None
    action, new_missed = decide(
        status,
        alive,
        entry.get("missed", 0),
        threshold,
        entry_age_s=entry_age_s,
        spawn_grace_s=_respawn_spawn_grace_s(),
    )
    print(
        f"  issue #{issue}: status={status} alive={alive} "
        f"missed={entry.get('missed', 0)}->{new_missed} action={action}"
    )

    if action == "delete":
        # #720: before unmapping, drop the breadcrumb so the idle-unmapped pass
        # can reap this now-unmapped session on the SHORT window. Only for the
        # live session we just observed (sid present + in live_ids); a
        # dead/missing session needs no breadcrumb (nothing to reap). status is
        # always in TERMINAL here (decide() returns "delete" only for TERMINAL),
        # and _record_last_mapped_terminal refuses a non-TERMINAL status anyway.
        sid = entry.get("happy_session_id")
        if isinstance(sid, str) and sid and sid in live_ids and isinstance(issue, int):
            _record_last_mapped_terminal(sid, issue, status, dry_run)
        if not dry_run:
            path.unlink(missing_ok=True)
    elif action == "respawn":
        _crash_arm_respawn_or_hold(entry, path, issue, threshold, dry_run)
    elif action == "keep":
        dirty = False
        if new_missed != entry.get("missed", 0):
            entry["missed"] = new_missed
            dirty = True
        # #845 (b): a keep-with-reset (alive again / spawn-grace) ends any
        # hold episode — clear the counter so a LATER unrelated crash starts
        # its hold budget fresh.
        if new_missed == 0 and entry.get("wt_hold_count", 0):
            entry["wt_hold_count"] = 0
            dirty = True
        if dirty and not dry_run:
            path.write_text(json.dumps(entry, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
