---
name: campaign-tick
description: >
  Lightweight recurring driver for autonomous /campaign <N> sessions
  (task #586). Triggered by the `*/45 * * * *` backstop cron
  (`prompt="/campaign-tick <N>"`) armed at Step 0 of the full /campaign
  skill. FIRST action on every fire is ONE Bash call to
  `scripts/tick_triage.py <N> --kind campaign` (pure Python: campaign
  status, newest `epm:campaign-*` marker, the results-landed wake check
  against campaign-state.json + child statuses, snapshot + runaway
  counter) and branches on its verdict: HEALTHY → end the turn
  immediately; TERMINAL → CRON-TEARDOWN, end; GATE-TRANSITION (into
  `blocked`) → PushNotification + CRON-TEARDOWN, end; STALE-REDRIVE →
  re-drive the full /campaign skill (one decision round). Does NOT
  re-load the full /campaign SKILL.md on healthy idle ticks.
user_invocable: false
---

# /campaign-tick — recurring lightweight driver

The recurring driver for autonomous `/campaign` sessions, mirroring
`/issue-tick`. Spawned every 45 minutes by the in-session cron registered
at Step 0 of the full `/campaign` skill (`CronCreate(*/45 * * * *,
prompt="/campaign-tick <N>", recurring=True, durable=False)`). The
45-minute interval matches `/issue-tick` (2026-06-12 redesign): the
10-min pure-Python watcher carries fast detection (campaign pass +
gate-push pass); this tick is the in-session re-driver of last resort.

## Contract — guarded no-op tick

On every fire, the FIRST action is exactly ONE Bash call:

```bash
uv run python scripts/tick_triage.py <N> --kind campaign
```

(From a worktree cwd, resolve the script from the MAIN checkout —
`"$REPO_ROOT"/scripts/tick_triage.py` — same rule as `/issue-tick`.)

In `--kind campaign` mode the triage reads the campaign status, the
newest `epm:campaign-*` marker (ignoring watcher-sentinel notes —
`[autonomous_session_watch:campaign]` alerts are not campaign progress),
the child-task statuses, and `artifacts/campaign-state.json`, then
prints ONE `<VERDICT> <reason>` line. It also maintains the snapshot
(`~/.eps-autonomous/issue-tick-last-status/<N>.json` — the shared
snapshot dir keys on task id, and a task is either an issue or a
campaign, never both) and the consecutive-terminal runaway counter
(`tick-runaway-<N>.flag` on the 3rd consecutive terminal tick — the
watcher force-stop parachute).

| Verdict | Action |
|---|---|
| `HEALTHY` | **END THE TURN immediately.** Fresh `epm:campaign-*` marker, or stale but every open arm is in flight in the children (their own /issue sessions + watcher passes cover them — nothing for the campaign to decide). No further tool calls. |
| `TERMINAL` | CRON-TEARDOWN, one-line log, END TURN. Covers `completed` / `archived` / steady-state `blocked`, AND the stranded-cron case (`proposed` / `planning` / `plan_pending` — a tick should never be armed before the brief-approval gate, workflow.yaml § gates.campaign_brief_approval, so tear it down). |
| `GATE-TRANSITION` | The transition tick into `blocked`: ONE `PushNotification` with the latest `epm:failure` note trimmed to ~80 chars, then CRON-TEARDOWN, END TURN. Swallow push exceptions. |
| `STALE-REDRIVE` | Load the full `/campaign <N>` skill for ONE decision round. Two triggers: (a) **results-landed wake** — a child reached `awaiting_promotion` / `completed` / `blocked` and its campaign-state row is not yet `ingested`/`abandoned` (fires regardless of marker freshness — a landed result should not wait out the staleness window); (b) **decision round owed** — markers stale >~25 min AND at least one open arm is NOT covered by an in-flight child (a `planned` row with no child filed, a reconcile owed, or no children in flight at all). The full skill's Step 0 ARM-GUARD makes re-entry safe. |

**Non-zero exit or unparseable output → treat as `STALE-REDRIVE`** (fail
toward coverage: load the full `/campaign <N>` for one decision round).

## Argument

One required argument: the campaign task number `<N>`.

## One-time deferred tool loads

```
ToolSearch("select:CronList,CronDelete,PushNotification")
```

`CronCreate` is NOT needed — only the full `/campaign` skill arms the
cron (its Step 0 ARM-GUARD prevents duplicates). Defer the ToolSearch
until a non-HEALTHY verdict actually needs the tools.

## Title refresh — moved to the watcher (2026-06-12)

The per-tick `session_progress_report.py` + `change_title` calls were
removed, same as `/issue-tick`: the watcher's gate-push pass reconciles
the title self-report on status transitions, and the 5-min summarizer
covers the dashboard. A healthy tick does not touch the title.

## CRON-TEARDOWN — hardened 2026-06-12

`CronList()`, delete EVERY job whose prompt matches this campaign's
tick: primary match is whole-string equality
(`prompt.strip() == "/campaign-tick <N>"`); hardened fallback is the
anchored pattern `campaign-tick\s+<N>(?!\d)` (the `(?!\d)` guard
prevents sibling mis-delete — `"/campaign-tick 46"` never matches
`"/campaign-tick 467"`). Then ASSERT-AFTER-DELETE: re-`CronList`, retry
the delete once if a matching job survived, log LOUDLY if it still
survives (the runaway parachute bounds the damage). Idempotent; never
raise.

## What this skill does NOT do

- It does NOT post `epm:*` markers or mutate task state — the snapshot +
  runaway-flag files are written by `tick_triage.py`, and only the full
  `/campaign` skill owns `events.jsonl` mutations.
- It does NOT spawn children, proposal agents, or any subagent.
- It does NOT arm crons (`CronCreate` lives only in the full /campaign
  skill's Step 0).
- It does NOT read child clean-result bodies — ingest belongs to the
  full skill's Step 1.1 (the triage compares only statuses + state-file
  rows, never body content).
- It does NOT refresh the title (watcher-owned since 2026-06-12).
