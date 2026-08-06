---
title: 'daily-fix: healthcheck auto-launches /daily backfill'
kind: infra
tags:
- wf-fix
- wf-fix-fp:83734a70935b
- daily-auto-filed
created_at: '2026-08-06T07:00:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): husk alert is fire-and-forget;
  08-04 sweep stayed unmined ~24h after the 529-killed nightly because nobody ran
  the named backfill command'
workflow: v1
---
# daily-fix: cron_daily_healthcheck auto-launches the /daily backfill on missing/husk detection (alert-only leaves days unmined)

## Workflow gap

`scripts/cron_daily_healthcheck.sh` (06:00 PT, task #711 heartbeat + #1189 husk arm)
detects a missing or never-enriched (husk) nightly `/daily` file and ALERTS with the exact
backfill command in the Telegram message — but never runs it. Recovery depends on a human
copy-pasting the command the same day. When nobody does, the day's problem sweep is
permanently skipped and the sentinel (`sent-<date>.flag`) suppresses any re-alert, so the
gap goes silent after one buzz.

Live incident (2026-08-05): the 2026-08-04 nightly (session start 2026-08-05T06:27Z) died
at 07:03:25Z on repeated API 529 Overloaded errors ~36 min in — stub + miner brief written,
nothing else. The 06:00 healthcheck correctly detected the husk and pushed the alert
(`sent-2026-08-04.flag` written, per `logs/daily_healthcheck/2026-08-05.log`), but no
backfill ran all day; the 2026-08-05 nightly found the 08-04 file still an all-empty husk
~24 h later, and the 08-04 transcript window is still unmined.

verified-at-filing: `grep -n 'claude -p' scripts/cron_daily_healthcheck.sh` → 2 hits, both
inside alert MESSAGE strings (lines 88, 90) — no execution path; `grep -c 'backfill'` → 2,
message-only. Run at 2026-08-06T07:00Z on main. The 529 death is read from the dead
session transcript (`~/.claude/projects/-home-thomasjiralerspong-explore-persona-space/970a1dd6-aad6-425a-b99d-d0350698db33.jsonl`,
final assistant row 2026-08-05T07:03:25Z: "API Error: Repeated 529 Overloaded errors").

## Proposed change

On a missing/husk detection, the healthcheck ADDITIONALLY launches the backfill it already
names, detached and single-flight, keeping the alert:

- `cd $PROJECT_DIR && CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=10800000 $HOME/.local/bin/claude -p "/daily $YESTERDAY"`,
  detached (setsid + log breadcrumb under `logs/daily_healthcheck/backfill-<date>.log`),
  guarded by a per-date attempt sentinel (one auto-attempt per missed day; a failed
  auto-backfill re-alerts rather than looping) and a flock (never two /daily processes).
- Timing is safe by construction: 06:00 PT is far outside the skill's "no backfill within
  60 min of the 23:27 nightly" rule, and the skill's Edit-in-place husk-recovery branch is
  the designed re-entry (`.claude/skills/daily/SKILL.md` § Output).
- The alert message changes from "run this command" to "auto-backfill launched (attempt 1);
  log: <path>" — Thomas keeps visibility, loses the manual step.

Planner should decide the failure policy (e.g. if the auto-backfill itself dies, the next
day's healthcheck sees the still-husk file but the sentinel must not suppress a second
attempt + re-alert) and whether the mygoat-side retrospective log check stays.

## Provenance

- fingerprint: 83734a70935b

- workflow_fix_target: scripts/cron_daily_healthcheck.sh
- origin: /daily 2026-08-05 nightly run — orchestrator observation while investigating the
  2026-08-04 husk (529-killed nightly, session 970a1dd6, died 2026-08-05T07:03:25Z).
