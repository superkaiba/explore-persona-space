---
name: campaign-tick
description: >
  Lightweight recurring driver for autonomous /campaign <N> sessions
  (task #586). Triggered by the `*/20 * * * *` backstop cron
  (`prompt="/campaign-tick <N>"`) armed at Step 0 of the full /campaign
  skill. Reads the campaign status + newest `epm:campaign-*` marker via
  `scripts/task.py`, refreshes the phone title, RE-DRIVES the full
  /campaign skill (one decision round) when the campaign is stale OR when
  any child newly reached awaiting_promotion / completed / blocked since
  the last ingest (the results-landed wake signal), and runs
  CRON-TEARDOWN at terminal state. Does NOT re-load the full /campaign
  SKILL.md on healthy idle ticks — a few hundred tokens per tick.
user_invocable: false
---

# /campaign-tick — recurring lightweight driver

The recurring driver for autonomous `/campaign` sessions, mirroring
`/issue-tick`. Spawned every 20 minutes by the in-session cron registered
at Step 0 of the full `/campaign` skill (`CronCreate(*/20 * * * *,
prompt="/campaign-tick <N>", recurring=True, durable=False)`).

## Contract

A **lightweight tick**: it does NOT re-load the full `/campaign` SKILL.md
unless a re-drive branch fires (3c/3d below). On a healthy idle tick the
work is: two `scripts/task.py` reads + one `list-children` read + a title
refresh. No subagents, no task-state writes, no marker posts — the full
`/campaign` skill owns every `events.jsonl` mutation.

## Argument

One required argument: the campaign task number `<N>`.

## One-time deferred tool loads

```
ToolSearch("select:CronList,CronDelete,PushNotification")
```

`CronCreate` is NOT needed — only the full `/campaign` skill arms the
cron (its Step 0 ARM-GUARD prevents duplicates).

## Execution

### Step 1: Read state (no agents)

```bash
uv run python scripts/task.py view <N> --json
uv run python scripts/task.py latest-marker <N> --prefix epm:campaign
uv run python scripts/task.py list-children <N> --json
```

From `view`: the campaign `status`. From `latest-marker`: the newest
`epm:campaign-*` marker's kind + `ts` (ignore markers whose note carries
the `[autonomous_session_watch:campaign]` sentinel — those are
watcher-posted alerts, not campaign progress). From `list-children`: each
child's `id` + `status` + `has_clean_result`.

If any call fails (registry corruption, missing task), log one line and
EXIT — a broken task is not this tick's problem.

### Step 2: Refresh the canonical title (soft-fail)

```bash
uv run python scripts/session_progress_report.py --issue <N> --step "campaign:<status>"
```

then `mcp__happy__change_title` with the captured string. Both calls are
SOFT-FAIL (observability, not load-bearing) — same rule as
/issue-tick Step 2.

### Step 3: Branch

#### 3a. TERMINAL status (`completed` / `archived` / `blocked`)

CRON-TEARDOWN (Step 4), one line, EXIT. On the transition tick into
`blocked` (previous-status snapshot differs — see Step 5), fire ONE
`PushNotification` with the latest `epm:failure` note trimmed to ~80
chars first.

#### 3b. NOT-YET-APPROVED status (`proposed` / `planning` / `plan_pending`)

The campaign brief has not passed its approval gate (workflow.yaml §
gates.campaign_brief_approval) — a tick should never be armed here (the
full skill refuses to arm before `approved`), so this is a stranded
cron: CRON-TEARDOWN, one line, EXIT.

#### 3c. Results-landed wake (the priority branch)

If ANY child's status is `awaiting_promotion`, `completed`, or `blocked`
AND the state file's experiment row for that child is not yet
`ingested` / `abandoned` (cheap read of
`artifacts/campaign-state.json` via the folder from `task.py find <N>`;
compare `child_task` ids) → a result has landed since the last decision
round. Log one line and load the full `/campaign <N>` skill — its Step 1
reconcile ingests the child and may immediately file the next arm. This
fires regardless of marker freshness: a landed result should not wait
out the staleness window.

#### 3d. ACTIVE (`approved` / `running`) — staleness check

- Newest skill-posted `epm:campaign-*` marker FRESH (within ~25 min):
  the in-session decision loop is alive. EXIT; the cron stays armed.
- STALE (>25 min) and no 3c wake: the session's reaction chain may have
  died mid-round, or the campaign is simply idle between child
  landings. Distinguish cheaply: if every non-`ingested` /
  non-`abandoned` experiment row maps to a child at an ACTIVE status
  (work is genuinely in flight in the children — their own /issue
  sessions + watcher passes cover them), EXIT quietly; the campaign has
  nothing to decide. Otherwise (a `planned` row with open slots, a
  reconcile owed, or no children in flight at all) → log one line and
  load the full `/campaign <N>` skill to run a decision round. The
  Step 0 ARM-GUARD makes re-entry safe.

### Step 4: CRON-TEARDOWN

`CronList()`, delete the job whose `prompt.strip() ==
"/campaign-tick <N>"` — whole-string equality, NOT substring
(`"/campaign-tick 46"` is a substring of `"/campaign-tick 467"`).
Idempotent; never raise.

### Step 5: Snapshot status + EXIT

Write `~/.eps-autonomous/issue-tick-last-status/<N>.json` (same atomic
temp+rename shape as /issue-tick Step 5 — the shared snapshot dir keys on
task id, and a task is either an issue or a campaign, never both) with
`{"issue": <N>, "status": "<current>", "ts": "<UTC ISO-8601>"}`. EXIT.

## What this skill does NOT do

- It does NOT post `epm:*` markers or mutate task state — only the title
  self-report + the tick snapshot are written.
- It does NOT spawn children, proposal agents, or any subagent.
- It does NOT arm crons (only the full /campaign skill's Step 0 does).
- It does NOT read child clean-result bodies — ingest belongs to the
  full skill's Step 1.1.
