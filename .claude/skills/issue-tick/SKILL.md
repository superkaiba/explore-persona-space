---
name: issue-tick
description: >
  Lightweight recurring driver for autonomous /issue <N> sessions.
  Triggered by the `*/20 * * * *` backstop cron (`prompt="/issue-tick <N>"`)
  armed at Step 0 of the full `/issue` skill (with a defense-in-depth
  re-arm at Step 6d.2). Reads the latest marker + status via
  `scripts/task.py`, refreshes the canonical phone title via
  `scripts/session_progress_report.py`, fires `PushNotification` at
  gate-park / `blocked` transitions, RE-DRIVES the full `/issue` skill
  when the session is stale at any non-gate non-terminal status (covers
  both pre-pod-launch stretches like planning / clarifying / under-cap
  plan_pending / followups_running AND post-launch ACTIVE statuses where
  the bg-Bash poll chain has died), and runs CRON-TEARDOWN at
  terminal/gate-park state. Does NOT re-load the 44K-token `/issue`
  SKILL.md on idle ticks — that load happens only on cold start, cold
  respawn (which spawn `/issue <N>` directly), and the stale-re-drive
  recovery branch. On a healthy active run this is a few hundred tokens
  per tick.
user_invocable: false
---

# /issue-tick — recurring lightweight driver

This is the recurring driver for autonomous `/issue` sessions. It is
spawned every 20 minutes by the in-session cron registered at Step 6d.2
of the full `/issue` skill (`CronCreate(*/20 * * * *, prompt="/issue-tick
<N>", recurring=True, durable=False)`). The 20-minute interval is chosen
deliberately: the Anthropic prompt cache TTL is 5 minutes, so a 10-min
interval was the worst case (always cold, double the ticks for no
caching benefit); 20 min halves the tick count without sacrificing
backstop responsiveness.

## Contract

This skill is a **lightweight tick**. It does **NOT** re-load the full
`/issue` SKILL.md (~44K tokens) — that load happens only on cold start
(first `spawn_session.py spawn-issue --auto` fire of `/issue <N>`),
cold respawn (`autonomous_session_watch._respawn` and the Phase-2
`_respawn_stalled_session` actor, both of which spawn `/issue <N>` via
`--auto`), and the stale-re-drive recovery branches in 3b / 3c below.

On every fire this skill:

1. Reads task state (latest marker + status) — two `scripts/task.py`
   calls, no LLM, no agent spawn.
2. Refreshes the phone title + dashboard self-report via
   `scripts/session_progress_report.py`.
3. Branches on status:
   - **TERMINAL** → CRON-TEARDOWN, EXIT.
   - **PARK** (non-gate; status awaiting a non-/issue-tick event) → if
     the latest marker is FRESH (within the last ~25 min), title refresh,
     EXIT. If the latest marker is STALE, the orchestrator's reaction
     chain has likely died at this PARK status (e.g. `planning` waiting
     on a planner agent that crashed, `clarifying` waiting on a
     clarifier turn the harness lost, `followups_running` waiting on
     the proposer): this fire IS the recovery path — load the full
     `/issue <N>` skill to re-enter the matching step. The Step 0
     ARM-GUARD prevents duplicate crons.
   - **ACTIVE-WITH-LIVE-BG-WORK** → if the bg-Bash poll chain is healthy
     (a fresh marker landed inside the last ~25 min), title refresh,
     EXIT. Otherwise the orchestrator's primary poll chain has likely
     died — this fire IS the recovery path: load the full `/issue <N>`
     skill to re-enter Step 6d.2 (the existing ARM-GUARD prevents
     duplicate crons).
   - **GATE-PARK / OVER-CAP** (just transitioned to over-cap
     `plan_pending`, `awaiting_promotion`, or `blocked`) →
     `PushNotification` to the phone, CRON-TEARDOWN, EXIT.

Anything that needs the heavy `/issue` machinery (running the planner,
spawning the experimenter, the analyzer→clean-result-critic loop, the
auto-merge) is left to the FULL skill — either the in-session bg-Bash
poll chain that is still alive, or the cold-start / cold-respawn entry
point. This skill never tries to fork into "let me just do a tiny bit of
/issue work inline" — that path is reserved for the active-with-stale-
marker recovery branch, which loads the whole skill.

## Argument

One required argument: the issue number `<N>` (the integer naming
`tasks/<status>/<N>/`).

## One-time deferred tool loads

Before the first use this session, load these deferred tools:

```
ToolSearch("select:CronList,CronDelete,PushNotification")
```

`CronCreate` is NOT needed — this skill never arms a cron, only tears one
down. The full `/issue` skill is the only place that calls `CronCreate`
(at Step 6d.2).

## Execution

### Step 1: Read state (two shell calls, no agents)

```bash
uv run python scripts/task.py view <N> --json
uv run python scripts/task.py latest-marker <N>
```

From the `view --json` output:
- `status` = the parent folder name (current lifecycle state).
- `kind` = the task `kind` (experiment / infra / batch / analysis / survey).

From `latest-marker` (prints the full event JSON):
- the latest `epm:<kind>` marker name + `ts` (ISO-8601 UTC) — used to
  judge bg-chain liveness in the ACTIVE branch below.
- the `note` field — checked in Step 2 for the `[gpu-idle-advisory]`
  prefix (title suffix only).

If either call fails (`task.py` import error, registry corruption,
on-disk row missing), log one line and EXIT — do NOT attempt the title
refresh or any branch logic. A broken task is not this tick's problem.

### Step 2: Refresh the canonical title (soft-fail)

Same pattern as the full `/issue` skill's "Chat title updates" section.
Build the canonical string + write the self-report file via the helper,
then push to the phone:

```bash
uv run python scripts/session_progress_report.py --issue <N> --step "<status>"
```

Capture the stdout (the canonical `#<N> <slug> · <status>` string).

**GPU-idle suffix:** if the latest marker's `note` (from Step 1) starts
with `[gpu-idle-advisory]` — the one-time idle-GPU advisory
`scripts/poll_pipeline.py` posts as `epm:progress` when every GPU sat
idle on a healthy `status=running` tick (incidents #518/#537) — append
` · gpu-idle` to the captured string and log one line:
`/issue-tick <N>: gpu-idle advisory is the latest marker — idle GPUs on
a held pod`. This keeps the idle burn visible on the phone even when
the primary bg-Bash poll chain (whose "GPU-idle advisory handling" in
the full `/issue` skill normally acts on the advisory) is dead. The
suffix is TITLE-ONLY: no status change, no PushNotification, no
re-drive. The advisory is a regular `epm:progress` marker, so it also
counts as FRESH for the 3b/3c staleness checks like any other progress
marker.

Then:

```
mcp__happy__change_title({"title": <captured, plus the gpu-idle suffix when it applies>})
```

**Both calls are SOFT-FAIL.** The helper invocation AND `change_title`
are observability infrastructure, not load-bearing — a stale title is an
observability regression, not a reason to abort the tick. If the helper
fails (missing task / `task.py` import error / disk full), log one line
and continue. If `change_title` raises, swallow — the self-report file
already landed, so `happy-ls` + the `/sessions` dashboard still show the
right string.

### Step 3: Branch on status

#### 3a. TERMINAL status

Terminal statuses (the run is done; nothing left to drive autonomously):

- `completed`
- `archived`
- `awaiting_promotion`
- `blocked`

Action: **CRON-TEARDOWN** (see below), one-line log, EXIT.

A tick that landed on a terminal status is the normal post-park firing
between when the FULL skill ran CRON-TEARDOWN (idempotent) and when the
cron actually de-armed (the harness can lag a tick). Tear it down again
— `CronDelete` is idempotent and skipping it once is the bug that left
us with stranded crons.

(Note: `awaiting_promotion` and `blocked` are ALSO gate-park states
covered by 3d. When the status transition INTO either of them happens
inside this tick — i.e. the previous tick saw a non-terminal status and
this tick sees `awaiting_promotion` / `blocked` — the gate-park branch
fires the `PushNotification` before tearing down. On subsequent ticks at
the same terminal status, 3a fires the bare teardown — no second
notification.)

How to tell "just transitioned" from "still parked": read the previous
tick's snapshot file at `~/.eps-autonomous/issue-tick-last-status/<N>.json`
(written at the END of every tick, atomic temp+rename). If the previous
`status` differs from now, this IS the transition tick. Treat a missing
snapshot file as "previous status unknown" — if the current status is a
gate-park state, fire the PushNotification (the harmless side-effect of
a duplicate notification beats missing the transition entirely).

#### 3b. PARK status (in-skill park; tick may need to re-drive)

These are the non-terminal "the orchestrator is mid-skill on an
in-process step (not a user gate)" states:

- `proposed`
- `planning`
- `plan_pending` (under-cap, awaiting the auto-approve inside the
  full skill itself — NOT the over-cap park, which is GATE-PARK / 3d)
- `clarifying`
- `followups_running`

These statuses imply the full `/issue <N>` skill is SUPPOSED to be
making forward progress in-process (a planner / clarifier / proposer
subagent is in flight, or the next reaction turn is about to fire).
They are NOT user-driven gates — the only thing that should be holding
them is the in-skill reaction chain itself.

Check the latest marker's `ts` (from Step 1):

- **Fresh** (within the last ~25 min): the in-skill chain is alive,
  doing its job. Title refresh already happened in Step 2. EXIT.
  The cron stays armed.
- **Stale** (>25 min since the last marker AND status is in the list
  above): the in-skill chain has likely died — the orchestrator's
  reaction turn never landed, the subagent crashed, or a corrupted /
  truncated tool-call dropped the chain (same failure modes that
  motivate 3c's recovery). Log one line:
  ```
  /issue-tick <N>: park status=<status>, latest marker stale
  (ts=<ts>, age=<m> min) — in-skill chain likely died; loading full
  /issue for recovery.
  ```
  Then load the full `/issue <N>` skill (invoke `/issue` with `<N>` as
  the argument — same re-entry path used by cold start). The skill
  reads `events.jsonl` fresh and re-enters at the right step (Step 1
  / 2 / 2c / 10b, depending on the status). The Step 0 ARM-GUARD
  prevents duplicate crons, so re-entering is safe.

  Why re-drive here and not just nudge: an autonomous session that
  stalls at `planning` / `clarifying` / `plan_pending` (under-cap) /
  `followups_running` has nothing else to wake it. The external
  `autonomous_session_watch` stalled-detector now AUTO-RESPAWNs ACTIVE
  sessions (Phase 2, 2026-06-08), but it does NOT respawn PARK
  sessions by design (a respawn at PARK would land back in the same
  PARK status without solving the underlying in-skill stall). The
  in-process re-drive here IS the recovery path for those.

  Avoid re-driving GATE-PARK states (over-cap `plan_pending`,
  `awaiting_promotion`, `blocked`): those are user gates by design —
  staleness there is correct, the user is the wake-up signal. The list
  of statuses above EXCLUDES all gate-park states; 3d handles them.

Special case: `plan_pending` that just transitioned over the
auto-approve cap is a GATE-PARK — see 3d. Distinguish by reading the
latest `epm:awaiting-spend-approval` marker (if it exists AND was posted
after the last status change, this is the over-cap park, not a generic
plan_pending). The over-cap branch fires PushNotification + tears down;
the under-cap branch falls through to the stale-re-drive recovery
above.

#### 3c. ACTIVE status (work in flight; verify the bg chain is healthy)

These statuses imply the orchestrator's bg-Bash poll chain SHOULD be
firing real progress markers:

- `approved`
- `running`
- `verifying`
- `interpreting`
- `reviewing`

Check the latest marker's `ts` (from Step 1):

- **Fresh** (within the last ~25 min): the bg-Bash chain is alive,
  doing its job. Title was already refreshed in Step 2. EXIT.
- **Stale** (>25 min since the last marker on an ACTIVE status): BEFORE
  re-driving, run one cheap liveness probe on the underlying job — pod
  jobs: `ssh pod-<N> 'kill -0 $(cat <pid-file>) && stat -c %Y <log>'`
  (PID alive + log mtime fresh); VM jobs: same probe locally. If the job
  is VERIFIABLY ALIVE and merely slow-cadenced, do NOT load the full
  skill: post a lightweight heartbeat instead —
  ```bash
  uv run python scripts/task.py post-marker <N> epm:progress \
    --note "tick heartbeat: job verified alive (pid <pid>, log mtime <ts>); slow phase, no state change"
  ```
  — which resets both this tick's stale clock and the
  `autonomous_session_watch` ALIVE-BUT-STALLED clock, then EXIT.
  (Incident 2026-06-09, #522: a slow 40h analysis phase kept marker
  cadence >25 min, so consecutive ticks re-drove the full 44K-token
  /issue skill ~7 times in 4h and exhausted the session context by
  14:09Z. The same gap drove most of the day's 63 watcher
  auto-respawns.) Only when the probe FAILS (PID gone, log frozen, or
  unverifiable) treat the chain as dead. Log one line:
  ```
  /issue-tick <N>: active status=<status>, latest marker stale
  (ts=<ts>, age=<m> min) — bg-chain likely died; loading full /issue
  for recovery.
  ```
  Then load the full `/issue <N>` skill (invoke the `/issue` skill with
  `<N>` as the argument; this is the SAME re-entry path used by cold
  start). The skill's Step 6d.2 ARM-GUARD prevents duplicate crons, so
  re-entering is safe. The full skill picks up state from
  `events.jsonl`, re-attaches to the running pod, and resumes Step 6d.2.

What counts as a "real progress marker": any `epm:progress`,
`epm:run-launched`, `epm:status-changed`, `epm:step-completed`,
`epm:results`, `epm:experiment-implementation`, or `epm:code-review*`
event (vs the synthetic `epm:status-changed` posted by `set-status`
which IS counted; that's a real state mutation). Stale-marker recovery
is the documented failure mode in the full skill's resume table — see
`.claude/skills/issue/SKILL.md` row "running (workload) | no
epm:results for > 4h", which the 25-min threshold tightens for the
poll-loop / reviewing case.

#### 3d. GATE-PARK / `blocked` (the user needs to do something)

Triggered when the status transition this tick crossed into:

- `awaiting_promotion` (clean-result ready, user promotes via
  `task.py promote <N> useful|not-useful`).
- `plan_pending` AND the latest `epm:awaiting-spend-approval` was
  posted AFTER the previous status change (over-cap plan, user approves
  via the dashboard or `task.py set-status <N> approved`).
- `blocked` (anything in the halt-criterion contract — the user reads
  the failure marker and decides next steps).

Action: fire `PushNotification` to the phone, then CRON-TEARDOWN, EXIT.

```python
# Build the message body. Keep under 200 chars (push payload limits).
# Reuse the canonical progress string head when convenient — the helper
# from Step 2 already produced `#<N> <slug> · <step>`; append a
# call-to-action.
if status == "awaiting_promotion":
    msg = f"#{N} {slug} · clean-result ready — open to promote"
elif status == "plan_pending":  # over-cap (the latest spend-approval marker confirmed it)
    cap = os.environ.get("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "100")
    msg = f"#{N} {slug} parked at plan_pending — over {cap} GPU-h cap; open to approve"
elif status == "blocked":
    reason = read_blocked_reason()  # latest epm:failure note, trimmed to ~80 chars
    msg = f"#{N} BLOCKED: {reason} — open it"

PushNotification({"message": msg[:200], "status": "proactive"})
```

If `PushNotification` raises (Remote Control disconnected, the tool's
deferred schema didn't load), SWALLOW the exception and continue — the
title is already refreshed and the cron will still get torn down. A
missed push is a phone-side regression, not a workflow-stalling one.

### Step 4: CRON-TEARDOWN (runs at every terminal / gate-park exit)

The cron this skill is fired by is registered with the literal prompt
`"/issue-tick <N>"`. Find it and delete it:

```python
CronList()  # returns [{id, cron, prompt, recurring, durable, ...}, ...]
for job in jobs:
    if job.get("prompt", "").strip() == f"/issue-tick {N}":
        CronDelete(id=job["id"])
        break
```

**Use whole-string equality (`prompt.strip() == "/issue-tick <N>"`),
NOT substring** — `"/issue-tick 46"` is a substring of `"/issue-tick
467"`, so substring matching would mis-delete a sibling issue's cron.

Idempotent: if no matching job exists (the harness already de-armed it,
or a previous tick already tore it down), this is a no-op. Do not raise.

### Step 5: Snapshot status + EXIT

Write the previous-status snapshot atomically (temp+rename) so the next
tick can detect transition vs steady-state:

```
~/.eps-autonomous/issue-tick-last-status/<N>.json
{
  "issue": <N>,
  "status": "<current status>",
  "ts": "<UTC ISO-8601>"
}
```

Then EXIT. Total tick work: 2 shell calls + 1 helper invocation + 1
title push + (sometimes) 1 PushNotification + (sometimes) 1
CronList/CronDelete pair + 1 snapshot write. No subagent, no LLM, no
`/issue` skill re-load.

## What this skill does NOT do

- It does NOT spawn the `experimenter` / `implementer` / `analyzer` /
  any other agent. Those dispatches live in the full `/issue` skill.
- It does NOT arm crons (the full skill is the only place
  `CronCreate("/issue-tick <N>")` runs — Step 6d.2 of `/issue`).
- It does NOT mutate task state via `task.py set-status`, `task.py
  post-marker`, or any other write subcommand. Only the title self-
  report file + the tick-snapshot file are written; both are
  observability state, not task state.
- It does NOT post `epm:*` markers. The bg-Bash poll chain + the full
  `/issue` skill own the `events.jsonl` log; a recurring driver that
  appends markers would drown the audit trail in tick noise.

## Why this skill exists (background)

`--auto` `/issue` sessions used to be driven by `/loop 10m /issue <N>`,
which re-loaded the 44K-token `/issue` SKILL.md every 10 minutes
regardless of whether anything had changed. On an idle 4-hour stretch
that's 24 SKILL.md re-loads, ~1M input tokens of recurring overhead.

This skill is the new recurring driver: read state, refresh title, fire
push at gates, tear down at terminal. The full `/issue` skill is loaded
exactly ONCE per session (cold start), plus on cold respawn after a
process death (`autonomous_session_watch._respawn` re-spawns via
`spawn_session.py spawn-issue --auto`, which boots `/issue <N>`), plus
on the stale-marker recovery branch (3c above).

Cadence breakdown for a 4-hour idle stretch in an autonomous `--auto`
session:

| Event | Old (`/loop 10m /issue N`) | New (`/issue-tick N` cron) |
|---|---|---|
| Cold start | full `/issue N` load | full `/issue N` load |
| Idle ticks (~24) | 24 × full `/issue N` load | 24 × `/issue-tick N` (few hundred tokens each) |
| Real progress (bg-Bash exit) | inline turn, no skill re-load | inline turn, no skill re-load |
| Gate park | full `/issue N` re-entry | `/issue-tick N` fires PushNotification + tears down |
| Cold respawn | `spawn-issue --auto` re-fires `/loop 10m /issue N` | `spawn-issue --auto` re-fires `/issue N` (then arms tick cron at Step 6d.2) |
