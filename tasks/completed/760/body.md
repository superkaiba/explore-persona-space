---
title: recurring reaper for leaked Codex app-server daemons
kind: infra
tags:
- codex-reaper
created_at: '2026-06-30T07:56:34Z'
has_clean_result: false
origin_prompt: 'Codex leak: kill the 69 codex app-server/broker processes older than
  24h ... Worth a recurring reaper.'
---
# workflow-fix: recurring reaper for leaked Codex app-server daemons

## Overview / Motivation

The Codex ensemble-review path (`scripts/codex_task.py` → the openai-codex plugin)
spawns a persistent daemon trio per session — `node codex app-server`, its
`codex-linux-x64` vendor binary, and `app-server-broker.mjs serve` — that does
NOT exit after the companion task completes. They accumulate over weeks.

Incident 2026-06-30: **69 leaked codex daemons older than 24h** (oldest **~51
days**, PID 1044107) were live on the VM. Each holds `~/.codex/logs_2.sqlite`
(WAL mode) open, so SQLite could never checkpoint/truncate the WAL → it grew to
**2.6G** (main DB 4.4G). This contributed to a VM root-disk CRITICAL event and
leaks RSS. Manually reaped this incident (SIGTERM the 69 >24h daemons → WAL
`PRAGMA wal_checkpoint(TRUNCATE)` reclaimed 2.6G); a recurring reaper is needed
so it doesn't recur.

## Goal

A recurring cron that reaps leaked Codex daemon processes older than a threshold
(default 24h, env-overridable) and then truncates the `~/.codex/logs_2.sqlite`
WAL, mirroring the existing audit-cron conventions
(`cron_pod_audit.sh`/`pod_audit.py`, `cron_worktree_audit.sh`/`worktree_audit.py`).

## Mechanism / spec

**`scripts/codex_daemon_reaper.py`** (new):
- Enumerate processes (ps/psutil). A process is a reap candidate iff its args
  match ANY of: `app-server`, `app-server-broker`, `codex-linux-x64`,
  `codex-companion.mjs` — AND its elapsed time ≥ threshold.
- HARD EXCLUDE (never reap): args containing `codex_task.py` (the review
  drivers), `workflow_lint`, `wandb_reclaim`, and ANY matching daemon younger
  than the threshold (the live/recent reviews — this is what protected the
  active 753/755/758 reviews this incident).
- Kill order: SIGTERM all candidates, brief wait, SIGKILL survivors.
- After killing, best-effort `PRAGMA wal_checkpoint(TRUNCATE)` on
  `~/.codex/logs_2.sqlite` with a busy_timeout (reclaims the WAL frames the
  released connections were pinning). NEVER VACUUM (rewrites the whole DB).
- Report-only by default; `--apply` to act; `--json` summary — matching the
  other audit crons.
- Threshold env `EPS_CODEX_REAPER_MAX_AGE_H` (default 24).

**`scripts/cron_codex_reaper.sh`** (new): thin wrapper calling
`uv run python scripts/codex_daemon_reaper.py --apply`. Register on a recurring
schedule (recommend every 4h, or hourly) via the orchestrator's cron management,
as the other audit crons are.

**`tests/test_codex_daemon_reaper.py`** (new): pin the selector against a fake
process table — asserts it (a) selects a >24h `codex app-server`, (b) NEVER
selects a <24h daemon, (c) NEVER selects a `codex_task.py` process regardless of
age, (d) respects the env threshold override.

## Constraints / invariants

- Workflow-surface only (new `scripts/` helpers + `tests/`); no experiment code.
- Never kill a sub-threshold daemon or an active `codex_task.py` review.
- Idempotent; report-only default; ruff-clean; `workflow_lint.py --check-asks`
  passes; the cron follows the existing `cron_*.sh` pattern.

## Provenance

Filed from a chat-session incident on 2026-06-30 (VM root-disk CRITICAL
diagnosis). The validated manual filter that worked this incident: match
{`app-server`, `app-server-broker`, `codex-linux-x64`, `codex-companion.mjs`} with
elapsed ≥ 86400s, excluding {`codex_task.py`, `workflow_lint`, `wandb_reclaim`};
killed 69, kept 16 recent (incl. the live reviews), WAL 2.76G → 0.
