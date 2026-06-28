---
title: "Make nightly lesson-consolidation deterministic + monitored (de-risk /daily)"
kind: infra
tags: []
created_at: "2026-06-28T08:47:24Z"
has_clean_result: false
origin_prompt: "PM-session directive (Thomas): 'I just want all of this to be automated so I dont have to run it.' The failure-lesson consolidation lives only inside the nightly /daily LLM run; extract the deterministic parts to a pure-Python cron + add a heartbeat so a failed nightly run is visible."
---

## Overview / Motivation

The failure-lesson consolidation pipeline is the SOLE cross-issue path that
dedupes `epm:failure-lesson` markers, promotes recurring lessons into
`.claude/rules/gotchas.md`, and prunes over-eager `generalizes: yes`
agent-memory entries. Today it lives ONLY inside the nightly `/daily` LLM
skill (`claude -p /daily`, crontab 23:27 PT). Two fragilities:

1. **It rides a flaky 44K-token LLM run.** If the nightly headless `/daily`
   errors or half-completes, that day's consolidation silently does not
   happen and nothing alerts. Every other cron job in this repo is pure
   Python for exactly this reason.
2. **No health check.** A silently-failed `/daily` reads as "quiet day" — the
   PM digest line omits silently when yesterday's file is absent.

## Goal

Make the load-bearing, DETERMINISTIC parts of nightly consolidation
independent of the LLM run, and add a heartbeat so a failed nightly run is
visible.

## Scope / surfaces

- NEW `scripts/consolidate_lessons.py` (pure Python): scan the rolling-window
  `epm:failure-lesson` markers across all tasks, dedupe against existing
  `.claude/agent-memory/<agent>/` entries, promote recurring /
  `gotcha_candidate: yes` lessons into `.claude/rules/gotchas.md` (or route a
  workflow-fix candidate), prune over-eager `generalizes: yes` entries.
  Deterministic, idempotent, fails loud, logs counts. The rolling window
  (default 7d) also subsumes the cross-day pattern-folding `/weekly` did.
- NEW `scripts/cron_lesson_consolidate.sh` + crontab registration (mirror the
  other `cron_*.sh`).
- NEW `scripts/cron_daily_healthcheck.sh` + crontab line (~01:00 PT): if
  `logs/daily/<yesterday>.md` is missing or >25h old, Telegram-push an alert
  via the my-goat `telegram_push.sh` channel.
- The LLM `/daily` keeps only its judgment parts (transcript problem-mining,
  held-item triage); `.claude/skills/daily/SKILL.md` updated to note the
  deterministic consolidation now runs in `consolidate_lessons.py` and `/daily`
  no longer owns it.
- `.claude/workflow.yaml:2049` text updated to point the consolidation
  responsibility at the deterministic script, not `/daily`.

## Constraints / invariants

- Workflow-surface only. No experiment code / configs / tasks.
- The inline `/issue`-time routing (`generalizes: yes` → agent-memory write,
  `gotcha_candidate: yes` → workflow-fix candidate) is UNCHANGED — it already
  fires per-lesson and is robust.
- Re-promotion must be idempotent (do not re-file a gotcha already present /
  a workflow-fix already open).
- `scripts/workflow_lint.py` + ruff on touched files pass.

## Acceptance

- A simulated failed nightly `/daily` (no `logs/daily/<date>.md` written)
  triggers the heartbeat alert.
- `consolidate_lessons.py` run over a fixture day produces the same gotcha /
  memory writes the `/daily` consolidation step would have, deterministically,
  and is a no-op on a second run.
