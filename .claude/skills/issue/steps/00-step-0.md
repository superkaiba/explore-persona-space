# Step 0: Load state

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

**Workflow-version dispatch (run BEFORE anything else).** Read the task's
`workflow` frontmatter field and, if it is `v2`, hand the whole task off to the
v2 skill and exit this skill immediately — v1 does nothing further for a v2 task.
This is a read-only probe (no markers, no mutation); the v2 skill re-runs the
same Step-0 guards itself.

```bash
uv run python scripts/task.py view <N> --json | uv run python -c \
  "import json,sys; print((json.load(sys.stdin).get('frontmatter') or {}).get('workflow') or 'v1')"
# → "v2"  : announce "Task #<N> is workflow: v2 — delegating to issue-v2." then
#           invoke the Skill tool `issue-v2` with the same task number <N> and STOP
#           (do NOT run any step below; issue-v2 owns the whole lifecycle).
# → "v1" / absent / empty : continue below UNCHANGED (v1 path, byte-for-byte).
```

**Single-orchestrator guard (run FIRST).** Exactly ONE session may drive
`/issue <N>` at a time. Before doing anything else, check whether another
live session is already mapped to this issue: `uv run python
scripts/spawn_session.py list` (issue-mapping column). If a live session is
already driving #N, EXIT immediately as a duplicate (do NOT run
`scripts/post_step_completed.py` here — this site's exit-marker contract is
the single breadcrumb below, per the "Post NOTHING else" rule that follows
it). Before ending, post
EXACTLY ONE exit breadcrumb so the audit trail distinguishes a deliberate
collision exit from a wrapper crash (#1053):

```bash
uv run python scripts/task.py post-marker <N> epm:progress --by issue-session-guard \
  --note "deliberate-stop pid=n/a target=self reason=step0-session-collision owner=<owner-session-id> — duplicate /issue <N> session exiting at Step 0; owner <owner-session-id> remains the driver; no state mutated"
```

The `deliberate-stop ` note PREFIX is load-bearing (the note must START
with it, lstripped): all four staleness clocks —
`task_workflow.stage_dispatch_should_skip`,
`autonomous_session_watch._latest_progress_ts`,
`autonomous_session_watch._latest_nonwatcher_event_ts`, and
`tick_triage.latest_event_ts` (#810/#949/#990/#1053) — drop it by prefix,
so the breadcrumb never refreshes the OWNER's freshness windows or
staleness clocks. It WILL surface as one pre-dispatch triage candidate for
the owner — deliberate: a session collision is genuinely triage-worthy
externality (mirroring the existing operator-stop breadcrumb;
`issue-session-guard` is deliberately NOT in `TRIAGE_MACHINE_BY`).
Presence of this breadcrumb PROVES a deliberate guard exit; its ABSENCE is
evidence of — not proof of — a crash (a fail-soft post failure, a
pre-guard death, or a stale skill copy also yield absence). Fail-soft: if
the post fails, state the failure in your final text and still exit —
never block the exit on the breadcrumb (a deferred-commit stderr ERROR
with exit 0 is SUCCESS; never re-post on it). Post NOTHING else: do NOT
run `scripts/post_step_completed.py`, do NOT write the per-issue
self-report (`session_progress_report.py` would clobber the OWNER's shared
`~/.eps-autonomous/issue-progress/<N>.json`), mutate nothing else —
UNLESS this session is its explicit replacement (an
`autonomous_session_watch` crash-recovery respawn, or the user said to take
over; in that case stop the stale session via `spawn_session.py stop` first).
(#524)

**Stale-wake ownership re-check (applies on RESUME, not just invocation).**
The guard above fires at `/issue` invocation — but a session that RESUMES
in-flight work after a long mid-flight stall must re-establish ownership
too, because the watcher may have respawned a replacement while it was
dark (and a manually-started session that never `register-current`'d is
invisible to the replacement's own Step 0 check, so the stale session is
the ONLY one positioned to detect the collision). If >30 min have passed
since this session's last tool call / turn, OR its last posted marker is
older than 30 min AND `events.jsonl` has advanced since, do NOT execute
the stale next step. FIRST re-run the guard: read `uv run python
scripts/task.py latest-marker <N>`, the registration files (fail-soft
probe below — copy it verbatim), and `uv run python
scripts/spawn_session.py list`.

```bash
# Registration-file probe — FAIL-SOFT, copy verbatim. A missing file is
# the NORMAL case (an interactive session that never ran
# register-current; an autonomous one whose registry entry was deleted
# at a terminal transition), and a bare `ls`/`cat` on an absent path
# exits non-zero (ls: 2, cat: 1), which CANCELS parallel sibling tool
# calls issued in the same message. Rule:
# an INFORMATIONAL probe on an OPTIONAL file is ALWAYS fail-soft —
# append `2>/dev/null` + `|| true`, or if-form it
# (`if [ -f <p> ]; then cat <p>; fi`); Step 9c step 1b's "Recipe
# exit-code hygiene" covers the trailing-command flavor of this class.
cat ~/.eps-autonomous/issue-<N>.json ~/.eps-autonomous/manual-issue-<N>.json 2>/dev/null || true
```

If a replacement
session is registered for #N (a `spawned_at` newer than this session's
own start) OR the marker trail shows another writer has advanced the task
past this session's last-known state, YIELD immediately. Before ending,
post the SAME exit-breadcrumb convention as the Step 0 guard above (one
`epm:progress`; same fail-soft contract — a deferred-commit stderr ERROR
with exit 0 is SUCCESS, never re-post):

```bash
uv run python scripts/task.py post-marker <N> epm:progress --by issue-session-guard \
  --note "deliberate-stop pid=n/a target=self reason=stale-wake-yield replacement=<replacement-session-id-or-marker-evidence> — stale /issue <N> session yielding on wake; the replacement owns the task; no state mutated"
```

Then launch nothing, mutate nothing else; the replacement owns the task.
A `deliberate-stop` breadcrumb is a death record, NOT task advancement —
ANY marker-trail reader (this re-check, `task.py latest-marker <N>`, a
successor session deriving state from the trail) must never count one as
"another writer has advanced the task".
The cheap tell is always `task.py latest-marker <N>` before resuming any
stale in-flight plan: if events have advanced past your own last-known
state, re-derive state from the markers instead of executing the stale
next step. (#535)

**Interactive-session registration (run once the guard passes).** An
INTERACTIVE session (`EPM_AUTONOMOUS_SESSION` unset) driving `/issue <N>`
registers itself ONCE at Step 0 so it appears in `spawn_session.py list`'s
issue-mapping — otherwise a manually-started session is invisible to every
OTHER session's single-orchestrator guard (the other half of #535: the
replacement could not see the live manual session precisely because it
never registered):

```bash
uv run python scripts/spawn_session.py register-current --issue <N>
# idempotent; writes ~/.eps-autonomous/manual-issue-<N>.json (alert-only:
# `list` visibility + stalled/crash alerts — never auto-respawned)
```

Autonomous sessions skip this — `spawn-issue --auto` already registered
them (`issue-<N>.json`). Registration failure is non-fatal: state the
failure and continue (same fail-soft contract as the Step 9b same-issue
follow-up loop's step-2 re-registration).

**Workflow-fix recursion-guard self-set (#678).** If THIS task is a
workflow-fix task — its `body.md` carries a `workflow_fix_target:`
Provenance line (the durable signal,
`task_workflow.is_workflow_fix_session(N)`) — set
`EPM_WORKFLOW_FIX_SESSION=1` for the session's own ergonomics. The
durable Provenance signal is the primary; the env var is the in-session
convenience leg lost on a watcher respawn (which re-runs `spawn-issue
--auto` without custom env), so the durable signal is always re-derivable
from the body. The recursion guard means this session — or any subagent
it spawns — NEVER auto-files MORE workflow-fix tasks for its own
findings: a `<!-- workflow-fix-candidate v1 -->` it raises is LOGGED +
notified, not routed (analogue of `AUTO_REVIEW_DISABLED`; see
`.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

```bash
# Reads body.md frontmatter + the most-recent slice of events.jsonl.
# Use --json for the machine-readable shape (body + last events).
uv run python scripts/task.py view <N> --json
```

From the result, derive:

1. **Current state** = the task's parent folder under `tasks/` (the
   `status` value).
2. **Task type** = the `type` field in `body.md` YAML frontmatter
   (`experiment`, `infra`, `batch`, `analysis`, `survey`).
3. **Marker map** = scan the recent `events.jsonl` rows for
   `epm:<kind>` entries, build a dict keyed by kind with the highest
   version per kind.

**Same-issue follow-up dispatch (chat entry point).** Before the
normal status dispatch, scan ALL `epm:followup-scope` entries for this
issue (NOT the highest-version-per-kind marker map — distinct queued
follow-ups share the kind under different `followup_label`s, #894/#763)
grouped by `followup_label`: within a label the authoritative scope is
the latest-(`ts`, `version`) entry (corrections land append-only — the
#658 v3→v7 `persona-vectors-style-rb` chain), and a LABEL is UNRUN iff
it has no matching `epm:same-issue-followup-run v1`. Canonical
implementation: `task_workflow.unrun_followup_labels(list_events(N))`.
When ≥1 DISPATCHABLE unrun label exists, dispatch EXACTLY ONE label per
loop entry — the queue head: user-initiated labels (`source: user-chat`
/ `step-10b-pick`) first, then oldest armed ts (a label's arming ts is
its FIRST scope entry's ts). A `dispatchable: false` group (an
unlabeled non-correction scope → pseudo-label `unlabeled-<ts>`) is
NEVER executed as a round — surface it loudly instead (one chat line +
an `epm:progress` note naming the repair: re-post the scope with a
proper kebab-slug `followup_label`; the pseudo-label stays visible in
every future scan until repaired or retro-closed). The loop runs one
round and re-parks; the next `/issue <N>` entry picks up the next unrun
label. If ≥1 unrun label is present AND the status is
post-result (`interpreting` / `reviewing` / `awaiting_promotion` /
`completed`) — or `followups_running` itself (the mid-round resume
case: the loop holds that status, so a crashed round re-enters here) —
route into the **same-issue follow-up loop** (Step 9b §
Same-issue follow-up loop) instead of the normal resume row. This is
how chat-requested follow-ups execute: the chat session posts
`epm:followup-scope v1` (`source: user-chat`) on #N, then re-invokes
`/issue <N>`, and the dispatcher lands here. An unrun followup-scope
on a task still mid-pipeline (any other status) waits — the loop only
fires from a post-result state.

**Stale-label disposition rule (mechanical evidence only).** Before
executing a dispatched label, run
`task_workflow.followup_retro_close_evidence(events, label)` — the
exact-label mechanical evidence predicate (a 9a-quater `extends=<label>`
record; an `epm:free-analysis-followup-run` whose `followup_ref` EXACTLY
equals the label; or a status/step note carrying the exact parenthesized
`(<label>)` round token plus a round-completion word in the same
;/.-delimited clause as the token — a clause naming the label as queued /
unrun / scoped / armed / dispatched NEVER closes (park notes routinely
announce one round's completion while enumerating the queued next label
on the same line; #961)). On a NON-None
return, post the retroactive `epm:same-issue-followup-run v1`
(`followup_label` verbatim, `outcome: retroactive-close — <evidence>`)
and move to the NEXT unrun label instead of re-running a completed round
(legacy tasks like #658 carry such ghost labels). **This check is a
GHOST-label filter, NOT an execution gate.** A None return means NO
prior-run evidence exists — for a normal fresh never-run label this is
the EXPECTED result (no run happened, so no evidence can exist by
construction) and the label EXECUTES as the dispatched round: that is
the whole point of the queue (the #763 `neutral-contrast-and-cofit`
dispatch is the canonical fresh case). The skip-and-surface disposition
applies ONLY when the orchestrator has independent reason to suspect
the label ALREADY RAN — a legacy ghost-label task like #658, e.g. the
label's round is visibly recorded in the body / status history yet none
of the mechanical evidence classes can confirm it. In that
suspected-stale case do NOT close on prose suspicion (a merely-prose
mention of the label NEVER closes — closing takes mechanical evidence
only): skip the label this entry and surface it for manual disposition
(autonomous mode: continue to the next dispatchable label; the skipped
label stays queued and visible) rather than either re-running a
completed round or wrongly closing a queued one. Retro-close markers do
NOT count toward the same-issue round caps (Step 9b loop step 5 /
block C2).

**Set the launch title now.** As soon as the slug (task `title`) is known,
call `set_title(N, <status>)` (helper defined in the "Chat title updates"
section above) so the Happy phone session list AND the
`~/.eps-autonomous/session_progress.json` cache (read by `happy-ls` + the
`/sessions` dashboard) all show the SAME canonical string from the moment
the session is spawned. The `--step` is the current status (or `"launching"`
if status isn't loaded yet):

```bash
uv run python scripts/session_progress_report.py --issue <N> --step "<status>"
# Capture stdout (the canonical string), then:
# mcp__happy__change_title({"title": <captured>})  -> "#<N> <slug> · <status>"
```

This runs on EVERY `/issue <N>` invocation (idempotent — re-setting the same
title is harmless), so resumed sessions re-label themselves too AND the
self-report file gets re-touched, keeping the dashboard fresh. Later
status transitions, polling-loop ticks, and clean-result-finalized events
re-call the helper with an updated `--step`.

**Hard error: ambiguous status.** If `task.py view <N>` reports the task
exists in multiple folders (should be impossible because `task.py` holds
the flock — but the lint catches manual edits), abort with an error and
ask the user to reconcile. Do NOT pick.

**Soft error: status missing from frontmatter (legacy bodies), type missing,
or empty body.** These are recoverable; do NOT exit. Run Step 0b instead.

**Worktree spec-freshness BEFORE arming (sessions whose cwd is an issue
worktree).** A worktree pins the entire workflow surface at branch-fork
time, so the skill/cron prescriptions you are reading may be stale —
run the Step 5a spec-freshness sync (surgical `git checkout origin/main -- `
of the workflow-surface specs, with the branch-side-feature-edit guard)
FIRST, and resolve workflow-helper scripts (`verify_task_body.py`,
`post_step_completed.py`, ...) from the MAIN checkout (`"$REPO_ROOT"/scripts/...`),
never the worktree copy. (#501, #496)

**MANDATORY auto-armed backstop for autonomous sessions — arm it NOW.**
When `EPM_AUTONOMOUS_SESSION=1` is set (the session was spawned via
`spawn_session.py spawn-issue --auto`), arm the `/issue-tick <N>` cron
at Step 0, BEFORE any branching into Step 0b / 0c / 1 / 2 / 5 / 6. The
historical site (Step 6d.2) only covers `kind: experiment` runs that
reach the pod-launched polling loop; a session can stall ANYWHERE in
the lifecycle (during planning, code-review, plan_pending park, the
analyzer / clean-result-critic loop, even at first invocation) and the
late-arm leaves all of those stretches uncovered. (#518)

```python
# Load the deferred Cron tools once per session if not already loaded.
ToolSearch("select:CronCreate,CronList,CronDelete")

# ARM-GUARD: idempotent re-entry. Whole-string equality (not substring) —
# "/issue-tick 46" is a substring of "/issue-tick 467".
if os.environ.get("EPM_AUTONOMOUS_SESSION") == "1":
    # Preload the always-needed wait/poll schemas too — every autonomous
    # session reaches a Monitor until-loop or a TaskOutput read eventually,
    # and an unloaded deferred-tool call fails with InputValidationError
    # (#1875).
    ToolSearch("select:Monitor,TaskOutput")

    jobs = CronList()
    already_armed = any(
        (job.get("prompt", "").strip() == f"/issue-tick {N}") for job in jobs
    )
    if not already_armed:
        CronCreate(
            cron="*/45 * * * *",
            prompt=f"/issue-tick {N}",
            recurring=True,
            durable=False,
        )
        # Re-list + assert exactly-one match, same dupe-fail-fast contract
        # Step 6d.2 uses — surfaces a harness prompt-normalization bug NOW
        # rather than after dozens of duplicate ticks have accumulated.
        post = CronList()
        match_count = sum(
            1 for job in post if job.get("prompt", "").strip() == f"/issue-tick {N}"
        )
        assert match_count == 1, f"cron arm: expected 1 match, got {match_count}"
```

Interactive sessions (no `EPM_AUTONOMOUS_SESSION`) do NOT arm the cron at
Step 0 — they're user-driven and the user re-invokes `/issue <N>` manually
when needed. The Step 6d.2 cron-arm still runs for those interactive runs
that DO reach the polling loop (same call shape, same ARM-GUARD), so the
session-survival backstop for pod-backed runs is unchanged for them. The
`Monitor,TaskOutput` preload above is likewise autonomous-only —
interactive sessions keep the lazy loads at the use sites (§ Long-phase
heartbeat duty item 1, Step 9a-quater LATE JOIN, Step 10d Guard 5), and
those sites remain the backstop for any session that resumes mid-lifecycle
without re-running this block.

The cron is torn down at the SAME terminal / park transitions as before
(see Step 6d.2 § CRON-TEARDOWN). Adding the early arm only widens the
window during which the backstop is in place; it does not change when
it's removed.
