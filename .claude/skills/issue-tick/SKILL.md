---
name: issue-tick
description: >
  Lightweight recurring driver for autonomous /issue <N> sessions.
  Triggered by the `*/45 * * * *` backstop cron (`prompt="/issue-tick <N>"`)
  armed at Step 0 of the full `/issue` skill (with a defense-in-depth
  re-arm at Step 6d.2). FIRST action on every fire is ONE Bash call to
  `scripts/tick_triage.py <N>` (pure Python: reads status + latest marker,
  computes staleness, screens detached-phase liveness before any
  STALE-REDRIVE — a live identity-verified breadcrumb `pid=`, a fresh
  `log=` mtime, or a fresh `[long-phase-heartbeat]` note reads HEALTHY,
  #1051 — and maintains the tick snapshot + runaway counter) and
  branches on its one-word verdict: HEALTHY → end the turn immediately;
  TERMINAL → CRON-TEARDOWN, end; GATE-TRANSITION → PushNotification +
  CRON-TEARDOWN, end; STALE-REDRIVE → re-drive the full `/issue` skill
  (the ONLY branch that loads the 44K-token SKILL.md; covers pre-pod-launch
  stretches like planning / under-cap plan_pending, the whole-round
  `followups_running` hold, AND post-launch ACTIVE statuses where the
  bg-Bash poll chain has died). Title refresh + the primary gate push moved
  to the pure-Python watcher (`autonomous_session_watch.py` gate-push pass,
  2026-06-12). A healthy tick is ONE Bash call, then end-turn.
user_invocable: false
---

# /issue-tick — recurring lightweight driver

This is the recurring driver for autonomous `/issue` sessions. It is
spawned every 45 minutes by the in-session cron registered by the full
`/issue` skill (`CronCreate(*/45 * * * *, prompt="/issue-tick <N>",
recurring=True, durable=False)` — Step 0 for autonomous sessions, Step
6d.2 re-arm for interactive pod-launched runs).

The 45-minute interval (lengthened from 20 min on 2026-06-12) is chosen
deliberately: the pure-Python `autonomous_session_watch.py` cron (every
10 min, free) now carries ALL fast detection — DEAD-session respawn,
alive-but-stalled respawn for ACTIVE statuses, pod safety, gate-park
phone push, title reconcile. The ONE failure class only this tick can
recover is "session alive, in-skill chain dead at a non-gate PARK
status" (the watcher deliberately does not respawn PARK statuses), and
that class tolerates 45-minute latency. Every fire is LLM-priced, so
fewer fires is the point; the per-fire cost is bounded by the
guarded-no-op contract below. (Historical note: the old 20-min rationale
leaned on a "5-minute prompt-cache TTL" figure that is inaccurate for
this org's subscription auth — subscription sessions get the 1-hour
cache TTL automatically; 5 minutes applies to API-key auth. The interval
choice no longer depends on the TTL either way.)

## Contract — guarded no-op tick

On every fire, the FIRST action is exactly ONE Bash call:

```bash
uv run python scripts/tick_triage.py <N>
```

(From a worktree cwd, resolve the script from the MAIN checkout —
`uv run python "$REPO_ROOT"/scripts/tick_triage.py <N>` — per the full
skill's Step 0 worktree spec-freshness rule. The triage's own task-state
reads go through the task-workflow library, which routes to `main`
regardless of cwd.)

`tick_triage.py` does ALL the bookkeeping the tick skill itself used to
do across ~5 tool calls: reads status + latest marker via the
task-workflow library, computes staleness (~25-min window), detects
status transitions against the previous tick's snapshot
(`~/.eps-autonomous/issue-tick-last-status/<N>.json`, which it writes
atomically), maintains the consecutive-terminal-tick runaway counter,
and on the 3rd consecutive TERMINAL tick writes
`~/.eps-autonomous/tick-runaway-<N>.flag` — the watcher's signal to
force-stop this session (the #501 runaway-cron parachute: 1,951 wasted
ticks over ~40h because CRON-TEARDOWN kept whiffing).

It prints ONE line — `<VERDICT> <reason>` — and the skill branches on
the verdict word:

| Verdict | Action |
|---|---|
| `HEALTHY` | **END THE TURN immediately.** No title refresh, no `change_title`, no snapshot write (triage wrote it), no further tool calls. |
| `TERMINAL` | CRON-TEARDOWN (below), one-line log, END TURN. |
| `GATE-TRANSITION` | `PushNotification` (below), CRON-TEARDOWN, END TURN. |
| `STALE-REDRIVE` | The recovery branch — the ONLY one that loads the full `/issue` skill. See below. |

**Non-zero exit or unparseable output → treat as `STALE-REDRIVE`.**
A broken triage must fail toward coverage (full re-drive), never toward
silence — a no-op-on-crash tick would leave the alive-stalled-at-PARK
class permanently unrecovered.

## Digest-only task-state reads (every tick turn, every task)

Any task-state read a tick turn makes BEYOND the one `tick_triage.py`
call MUST be a bounded, jq-filtered digest — unconditionally, for every
task, not just visibly harmful-content ones:

- NEVER a bare `task.py view <N>`, and NEVER an unpiped
  `task.py view <N> --json` without a narrow `| jq` filter on the same
  command: the JSON payload carries the full task BODY plus EVERY
  `events.jsonl` note (~625 KB on #906 — 296 events). For a
  harmful-content task (insecure-code datagen, EM vocabulary, jailbreak
  banks) that text is trigger-dense enough to refusal-kill the turn —
  #906's ticks were killed 3× this way (2026-07-03); the #866 prevention
  rule (CLAUDE.md § Spurious usage-policy refusals (e)) applies to tick
  turns exactly as it does to briefs.
- NEVER an unfiltered `task.py latest-marker <N>` when only freshness /
  resume position is needed — its `note` field can be up to 50,000
  chars. Use `... latest-marker <N> | jq '{kind, ts}'`.
- Pipes are the mechanism: `task.py view <N> --json | jq -r '<narrow
  filter>'` is safe by construction — only jq's output enters context;
  the body and notes never do.

Everything a tick turn legitimately needs is digest-shaped: status,
marker kind + timestamp, the title, and the ≤80-char blocked-reason
slice. Deciding per-task whether content is "safe" would itself require
reading the content, so the thinning is unconditional — it also bounds
the tick's context cost on benign tasks (#522 class). This contract
binds the TICK TURN's own reads on all four branches, up to and
including the decision to load the full `/issue` skill; once the full
skill is loaded, its own steps govern its reads — and the refusal-thinned
re-drive rule below additionally constrains the re-driven skill's resume
reads after a refusal-killed predecessor. Content-heavy work belongs to
the full skill's downstream subagents (fresh-context firewalls), never
to a tick turn.

## Argument

One required argument: the issue number `<N>` (the integer naming
`tasks/<status>/<N>/`).

## One-time deferred tool loads

Before the first teardown/push this session, load these deferred tools:

```
ToolSearch("select:CronList,CronDelete,PushNotification")
```

`CronCreate` is NOT needed — this skill never arms a cron, only tears one
down. The full `/issue` skill is the only place that calls `CronCreate`
(Step 0 / Step 6d.2). NOTE: defer the ToolSearch itself until a non-HEALTHY
verdict actually needs the tools — a HEALTHY tick should stay at one Bash
call.

## Title refresh — moved to the watcher (2026-06-12)

The per-tick `scripts/session_progress_report.py --issue <N> --step ...`
call and the `mcp__happy__change_title` push were REMOVED from this
skill. The pure-Python watcher's gate-push pass reconciles the canonical
title self-report on every STATUS TRANSITION (10-min cadence — fresher
than this tick ever was), and the 5-min Haiku summarizer keeps the
dashboard PROGRESS column fresh independently. The full `/issue` skill
still updates the title at its own step transitions. A healthy tick does
NOT touch the title; phone-title staleness between transitions is
cosmetic and accepted.

## STALE-REDRIVE recovery branch

`tick_triage.py` returns `STALE-REDRIVE` when the latest marker is stale
(>~25 min) at a non-gate, non-terminal status AND its detached-phase
liveness screen found no evidence (no live identity-verified breadcrumb
`pid=`, no fresh breadcrumb `log=` mtime, no fresh
`[long-phase-heartbeat]` note — #1051). Two sub-cases, split by
the status named in the verdict reason:

**ACTIVE statuses** (`approved` / `running` / `verifying` /
`interpreting` / `reviewing`): BEFORE re-driving, run one cheap liveness
probe on the underlying job — pod jobs:
`ssh pod-<N> 'kill -0 $(cat <pid-file>) && stat -c %Y <log>'` (PID alive
+ log mtime fresh); VM jobs: same probe locally. If the job is
VERIFIABLY ALIVE and merely slow-cadenced, do NOT load the full skill:
post a lightweight heartbeat instead —

```bash
uv run python scripts/task.py post-marker <N> epm:progress \
  --note "tick heartbeat: job verified alive (pid <pid>, log mtime <ts>); slow phase, no state change"
```

— which resets both the triage's stale clock and the
`autonomous_session_watch` ALIVE-BUT-STALLED clock, then EXIT.
(Incident 2026-06-09, #522: a slow 40h analysis phase kept marker
cadence >25 min, so consecutive ticks re-drove the full 44K-token
/issue skill ~7 times in 4h and exhausted the session context.) Only
when the probe FAILS (PID gone, log frozen, or unverifiable) treat the
chain as dead: log one line and load the full `/issue <N>` skill — the
SAME re-entry path used by cold start. The skill picks up state from
`events.jsonl`, re-attaches to the running pod, and resumes Step 6d.2.
The Step 6d.2 ARM-GUARD prevents duplicate crons, so re-entering is
safe.

**PARK statuses** (`proposed` / `planning` / under-cap `plan_pending` /
`followups_running` — the in-skill, non-user-gate parks): the in-skill
chain has likely died (the orchestrator's reaction turn never landed,
the subagent crashed, or a corrupted/truncated tool-call dropped the
chain). This fire IS the recovery path — load the full `/issue <N>`
skill; it re-enters at the right step (Step 1 / 2 / 2c / 10b, depending
on the status). For `followups_running`, the re-drive resumes at
whatever phase the `stage=followup-<phase>` breadcrumbs indicate. The
external watcher auto-respawns DEAD sessions at ACTIVE statuses but
deliberately does NOT respawn alive-stalled PARK sessions (a respawn
would land back in the same PARK without solving the in-skill stall) —
the in-process re-drive here is the only recovery for that class, which
is why this tick survives the redesign at all.

A re-driven session performing any compute dispatch is bound by the
`/issue` skill's § Pre-dispatch external-marker triage; a successor
session that cannot attribute window markers to itself treats them as
external (fail-toward-triage).

`tick_triage.py` never returns `STALE-REDRIVE` for gate-park states
(over-cap `plan_pending`, `awaiting_promotion`, `blocked`) — those are
user gates by design; staleness there is correct and the user is the
wake-up signal.

**Refusal-thinned re-drive (applies to every re-drive above).** If the
session's previous turn(s) died on a spurious "violates our Usage
Policy" API refusal (the turn ends in the refusal text — common on this
project's marker/EM/implant vocabulary), a naive re-drive replays the
same trigger-dense context and gets refused again: on 2026-06-10 the
#543 session was bricked for ~75 min by 4 consecutive tick re-drives
each re-refused. On a refusal-killed predecessor turn, re-drive THINNED:
resume from status + the filtered latest-marker digest
(`task.py latest-marker <N> | jq '{kind, ts}'`) only — do NOT page the
task body, full marker-note text, the clean-result body,
`epm:interpretation` bodies, or any raw-completion text back into
context (the § Digest-only contract above already bans the unfiltered
reads on every tick turn; after a refusal kill the posture tightens to
kind+ts+status only); let the next step's subagent (which starts with
fresh context) do the content-heavy lifting behind the analyzer's
content firewall. If the thinned re-drive is refused too, exit and leave
recovery to the watcher's respawn (a fresh session drops the poisoned
context — a lowered refusal surface, not immunity). (#1104/#1127: the
watcher's prompt-wedge lane counts API-error turns —
`isApiErrorMessage: true` assistant rows — as wedge evidence, so ≥3
consecutive FAILED wake turns force the respawn — a wake ENDING in an
api-error row is failed even when mid-turn heartbeats kept the
self-report fresh (staleness no longer gates the turn-level lane); a
session dead after a single refused turn is stopped for fresh respawn
on transcript silence ≥20 min (`failed-turn-silence`), the
crash-recovery arm completing the spawn, #1209;
`.claude/rules/background-automation.md` § (e).)

## GATE-TRANSITION branch — PushNotification

Fires when the triage detected the transition INTO a user gate this
tick: `awaiting_promotion`, `blocked`, or over-cap `plan_pending` (the
triage distinguishes over-cap via the `epm:awaiting-spend-approval`
marker being newer than the last status change; a missing previous
snapshot at a gate also counts — a duplicate push beats a missed one).

```python
# Build the message body. Keep under 200 chars (push payload limits).
# Slug + blocked reason come from ONE digest-only Bash call (rare path;
# the healthy path never pays it) — never an unpiped dump; the jq filter
# rides the same command line (§ Digest-only task-state reads):
#   uv run python scripts/task.py view <N> --json | jq -r \
#     '.frontmatter.title,
#      ([.events[] | select(.kind == "epm:failure")] | last
#        | (.note // "")[0:80] | gsub("\n"; " "))'
# → line 1 = slug, line 2 = blocked reason (empty when no failure marker).
# (Exact-equality select: event kinds are BARE strings with a separate
#  integer `version` field; startswith("epm:failure") would over-match
#  epm:failure-lesson — verified live on #906, where it returned the
#  newest failure-LESSON note instead of the failure reason.)
if status == "awaiting_promotion":
    msg = f"#{N} {slug} · clean-result ready — open to promote"
elif status == "plan_pending":  # over-cap (per the triage verdict reason)
    cap = os.environ.get("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "100")
    msg = f"#{N} {slug} parked at plan_pending — over {cap} GPU-h cap; open to approve"
elif status == "blocked":
    # reason = the jq-extracted ≤80-char epm:failure slice above — the ONLY
    # marker-note text a tick turn ever pages in, and only on this branch.
    msg = f"#{N} BLOCKED: {reason} — open it"

PushNotification({"message": msg[:200], "status": "proactive"})
```

> **Refusal posture (#1074/#1104):** on a refusal-killed predecessor turn
> (§ Refusal-thinned re-drive above), SKIP this digest call entirely —
> the title slug + the ≤80-char failure-note slice are the ONLY task text
> a tick turn ever pages in, and on a harmful-content-family task either
> can re-trigger the refusal. Push the numeric-only fallback instead:
> `msg = f"#{N} reached {status} — open it"`. A less-specific push beats
> a refusal-killed turn that pushes nothing.

If `PushNotification` raises (Remote Control disconnected, the tool's
deferred schema didn't load), SWALLOW the exception and continue to
CRON-TEARDOWN — a missed push is a phone-side regression, not a
workflow-stalling one.

**Dated removal note (2026-06-12).** The pure-Python watcher's gate-push
pass now fires the PRIMARY phone push (Telegram, transition-deduped) on
these same gate transitions at a 10-min cadence. This tick-side
`PushNotification` is KEPT as a second channel during the transition
period — both channels dedup per transition, so the worst case is one
duplicate notification, never a missed one. Once the watcher push has
run clean for ~a week, delete this PushNotification block and drop the
tool from the deferred load.

## CRON-TEARDOWN (runs at every TERMINAL / GATE-TRANSITION exit) — hardened 2026-06-12

The cron this skill is fired by is registered with the literal prompt
`"/issue-tick <N>"`. Find and delete it — ALL matching jobs, not just
the first — and ALSO sweep stray one-shot `/issue <N>` wakeups (#980),
resolving ids from a FRESH `CronList` at teardown time (#988):

```python
# CRON-TEARDOWN sweep — hardened 2026-06-12 (#501); widened + idempotent
# 2026-07-05 (#1052). Resolve live ids with a FRESH CronList() HERE, at
# teardown time — never CronDelete an id recorded earlier in the session
# (arm-time asserts, a previous tick's listing): recorded ids go stale when
# a one-shot fires or a concurrent teardown wins the race (#988).
jobs = CronList()  # [{id, cron, prompt, recurring, durable, ...}, ...]
for job in jobs:
    p = job.get("prompt", "").strip()
    # Leg 1 — the recurring tick cron: whole-string equality
    # (prompt.strip() == "/issue-tick <N>"); hardened fallback is the
    # anchored pattern for harness prompt-normalization drift (#501 — the
    # whole-string teardown silently no-oped 1,951 times). The (?!\d)
    # guard prevents sibling mis-delete ("/issue-tick 46" never matches
    # "/issue-tick 467").
    leg1 = p == f"/issue-tick {N}" or re.search(rf"issue-tick\s+{N}(?!\d)", p)
    # Leg 2 — stray one-shot wakeups that would re-drive the FULL skill on
    # a terminal/park task (#980: a live one-shot survived teardown on a
    # completed task and re-drove it). Anchored at the START so a prose
    # prompt that merely MENTIONS the issue is never deleted; (?!\d)
    # guards siblings; the '-' in "/issue-tick" fails \s+, so leg 2 never
    # re-matches leg 1's job. Trailing text (e.g. "--auto") matches by
    # design — anything in the store that starts with the bare full-skill
    # prompt is a re-driver for THIS issue and must die at teardown.
    leg2 = p == f"/issue {N}" or re.match(rf"/issue\s+{N}(?!\d)", p)
    if leg1 or leg2:
        CronDelete(id=job["id"])
        # IDEMPOTENT DELETE (#988): a CronDelete error indicating the job
        # does not exist (observed shape: 'No scheduled job with id ...')
        # means it is ALREADY GONE — that IS the desired end state. Treat
        # it as SUCCESS: continue the sweep; never retry the same id,
        # never abort or raise on it.

# ASSERT-AFTER-DELETE: re-CronList() and verify NO job matching EITHER leg
# survived. If one did, retry that delete ONCE (fresh id from the re-list);
# if it STILL survives, log LOUDLY and exit — the runaway parachute
# (tick_triage's 3-consecutive-terminal flag + the watcher force-stop)
# bounds the damage of a job that refuses to die.
```

Never use bare substring matching without the trailing-digit guard on
EITHER leg — `"/issue-tick 46"` is a substring of `"/issue-tick 467"`
(and the same for the leg-2 wakeup prompts), so unguarded substring
matching would mis-delete a sibling issue's cron or wakeup. Idempotent:
if no matching job exists, this is a no-op. Do not raise. A `CronDelete`
error indicating the job does not exist (observed shape:
`No scheduled job with id …`) is SUCCESS, not a failure — idempotent
means the job being gone is the goal.

## What this skill does NOT do

- It does NOT spawn the `experimenter` / `implementer` / `analyzer` /
  any other agent. Those dispatches live in the full `/issue` skill.
- It does NOT arm crons — `CronCreate` never appears in this skill's
  execution (the full `/issue` skill's Step 0 / Step 6d.2 are the only
  places `CronCreate("/issue-tick <N>")` runs).
- It does NOT refresh the title or call `change_title` — the watcher's
  gate-push pass owns the per-transition title reconcile (2026-06-12).
- It does NOT mutate task state via `task.py set-status`, `set-body`, or
  any other write subcommand. The only marker it may post is the 3c
  heartbeat (`epm:progress`, ACTIVE-status verified-alive case). The
  snapshot + runaway-flag files are written by `tick_triage.py`, not by
  the skill.
- It does NOT page the task body, full marker-note text, clean-result
  bodies, interpretation bodies, or raw completions into context — every
  task-state read beyond the triage call is a jq-filtered digest
  (§ Digest-only task-state reads; refusal-thinned re-drive rule).

## Why this skill exists (background)

`--auto` `/issue` sessions used to be driven by `/loop 10m /issue <N>`,
which re-loaded the 44K-token `/issue` SKILL.md every 10 minutes
regardless of whether anything had changed. The first replacement
(2026-06) was a 20-min `/issue-tick` cron that read state, refreshed the
title, pushed at gates, and tore down at terminal — ~5 LLM tool-call
turns per fire, ~5M tokens/fire at steady state. The 2026-06-12 redesign
moved title + gate push into the pure-Python watcher, collapsed the
healthy path to ONE Bash call (`tick_triage.py`), and lengthened the
cron to 45 min — the tick is now purely the last-resort in-session
re-driver for the alive-but-stalled-at-PARK class (plus a belt-and-
suspenders teardown/push at gates).

Cadence breakdown for a 6-hour idle stretch in an autonomous `--auto`
session:

| Event | Old (`/loop 10m /issue N`) | New (`/issue-tick N` cron @ */45) |
|---|---|---|
| Cold start | full `/issue N` load | full `/issue N` load |
| Idle ticks (~8) | 36 × full `/issue N` load | 8 × one `tick_triage.py` Bash call |
| Real progress (bg-Bash exit) | inline turn, no skill re-load | inline turn, no skill re-load |
| Gate park | full `/issue N` re-entry | watcher push (≤10 min) + tick push + teardown |
| Cold respawn | `spawn-issue --auto` re-fires `/loop 10m /issue N` | `spawn-issue --auto` re-fires `/issue N` (then arms tick cron) |
