---
title: Progress bar + estimated-time-remaining for in-flight issues (dashboard + session
  titles)
kind: infra
tags: []
created_at: '2026-06-11T05:18:21Z'
has_clean_result: false
---
# Progress bar + estimated-time-remaining for in-flight issues (dashboard + Happy session titles)

## Goal

Every in-flight task (statuses planning → reviewing, i.e. before `awaiting_promotion`) should display a pipeline progress bar and an honest estimated-time-remaining band, on the EPS dashboard (kanban card + task detail page) and as a suffix in the Happy session title.

## Background / feasibility data (already measured)

Per-stage durations computed from `epm:status-changed` markers across all historical `events.jsonl` (101 tasks with a full planning → awaiting_promotion span):

| Stage | n | median | p75 |
|---|---|---|---|
| planning | 162 | 0.61h | 0.82h |
| plan_pending | 86 | 0.43h | 4.25h |
| approved | 154 | 0.02h | 0.05h |
| running | 206 | 4.63h | 10.54h |
| verifying | 112 | 0.11h | 0.21h |
| interpreting | 144 | 0.91h | 1.49h |
| reviewing | 127 | ~0h | 0.15h |
| **Total planning→awaiting_promotion** | 101 | **16.1h** | p25–p75 = 8.6–42.1h |

Long tails exist (max running = 139h, from crashed/blocked runs); stats should prefer recent tasks since the workflow has sped up substantially over time.

## Deliverables

1. **Stage-stats + progress estimator module** (importable, single source of truth used by both surfaces):
   - Parses `epm:status-changed` transitions from `events.jsonl` across terminal-ish tasks (completed / awaiting_promotion / archived) to build per-stage duration distributions (median, p25, p75). Recency-weighted or windowed to recent tasks (planner picks the exact rule and justifies it).
   - For a given in-flight task: progress % = pipeline position (fixed stage sequence planning → plan_pending → approved → running → verifying → interpreting → reviewing), interpolated within the current stage by elapsed-vs-median.
   - ETA = sum of remaining-stage stats, reported as a **band** (median + p25–p75), never a single-number countdown.
   - `running`-stage refinement: when the task's approved plan records an estimated GPU-hours figure (e.g. plan-gate note "est 64.0 GPU-h"), prefer a wall-clock estimate derived from it over the historical median for that stage.
   - Human-wait stages (`plan_pending`) are excluded from the machine-ETA or labeled separately ("waiting on you"); autonomous sessions auto-approve so it is ~0 there.
   - `blocked` tasks: ETA suspended (show "blocked", not a fake countdown). `followups_running` / post-awaiting_promotion states: out of scope for the bar (bar covers the first pass to awaiting_promotion only, per the request).
   - Read-only over `tasks/` — never mutates task state; never shells `task.py` from the dashboard render path if direct read of `events.jsonl`/REGISTRY is simpler.

2. **Dashboard surfaces** (`dashboard/`, Next.js app served by local systemd `eps-dashboard.service` behind eps.superkaiba.com):
   - API route exposing progress + ETA per task (with caching so the kanban view doesn't re-parse every events.jsonl on every request).
   - Progress bar + ETA chip on kanban cards (`app/tasks/TaskBoard.tsx`) and on the task detail page (`app/tasks/[id]/`), shown only for in-flight statuses.
   - Deploy note: `next build && sudo systemctl restart eps-dashboard.service`; check `ss -tlnp | grep :3010` for a rogue `next start` squatter before restart.

3. **Happy session title suffix** (`scripts/session_progress_report.py`, refreshed by the 20-min `/issue-tick` cron):
   - Append a compact suffix to the canonical phone title for in-flight issues, e.g. `▓▓▓░░ 60% · ~5h left`, sourced from the same estimator module.

## Acceptance criteria

1. Estimator module has unit tests: stage stats computed from fixture events.jsonl; progress % monotone in pipeline position; ETA band excludes human-wait stages; blocked task returns suspended ETA; plan GPU-h refinement applied when present.
2. Dashboard kanban card and detail page render the bar + ETA band for at least one live in-flight task; terminal/parked tasks (awaiting_promotion, completed, archived, blocked) show no countdown.
3. `session_progress_report.py` produces the suffix for an in-flight issue and omits it at terminal/gate-park states; existing title format otherwise unchanged.
4. No mutation of `tasks/` state anywhere in the new code; dashboard API route is read-only.
5. `uv run ruff check` clean on touched Python; dashboard `next build` passes.

## Notes

- `kind: infra` (no experiment, no pod). Plan approval gate applies as usual.
- ETA honesty rule: band not point estimate; recompute live so re-plans / follow-up rounds show as the ETA moving rather than being hidden.
