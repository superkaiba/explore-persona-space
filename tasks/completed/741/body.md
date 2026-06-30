---
title: 'GCP: drop the 24h max-run-duration default; use max available (7d FLEX_START),
  reconcile the janitor age backstop'
kind: infra
tags: []
created_at: '2026-06-30T00:19:41Z'
has_clean_result: false
origin_prompt: remove this 24h max from the workflow -- make sure it always takes
  max time available.
---
## Overview / Motivation

The GCP lane's `--max-run-duration` defaults to **24h** (`GcpConfig.default_max_run_duration = "24h"`, `src/explore_persona_space/backends/gcp.py:197`). This is a self-imposed auto-delete fence, **not** a GCP limit — FLEX_START (the primary long-job provisioning model the GCP-first auto lane leads with) allows up to **7 days**. Long multi-cell sweeps strand mid-run when they exceed 24h: incident **#697**'s 64-cell causal-patch sweep completed only **40/64** cells before the 24h fence deleted the VM (`reason: gcp_max_run_duration_expired`), losing ~14h of remaining work and parking the task at `blocked`.

User directive (2026-06-29): *"remove this 24h max from the workflow -- make sure it always takes max time available."*

## Goal

Default GCP runs to the **maximum run duration the provisioning model allows** (7d for FLEX_START — the GCP hard ceiling — not 24h), instead of the self-imposed 24h fence, while preserving a **finite** credit-leak backstop (never unbounded — a wedged VM must still get reaped).

## CRITICAL coupling — the janitor age backstop (do NOT miss this)

Raising `default_max_run_duration` ALONE does not fix the bug. The stale-GCP-VM janitor (`scripts/gcp_audit.py --max-age-hours`, **default 24.0**, lines 172-175) reaps `eps-issue-*` instances older than 24h **independently** of `--max-run-duration`. With a 7d max-run-duration but a 24h janitor age, **the janitor becomes the new 24h cap and re-creates the exact bug**. The janitor age backstop MUST be reconciled:

- raise `--max-age-hours` to cover the new max-run-duration (≥ 7d + grace), OR
- make the age backstop aware of each instance's own max-run-duration / `eps/phase` so it reaps genuinely-wedged (never-terminal) VMs promptly without killing legitimately-long-running jobs.

Note: the terminal-phase 10-min reap (`--terminal-phase-max-age-min`) already handles crashed-to-terminal VMs promptly; the age backstop only needs to cover the truly-wedged-never-terminal case, so a single-knob raise is acceptable if the planner prefers it.

## Scope / files (grep-confirmed)

- `src/explore_persona_space/backends/gcp.py` — `default_max_run_duration` (line 197) → 7d / max-available; **preserve** the FLEX_START `>7d` assertion (lines 592-602; exactly 7d is allowed). Reconcile the "24h" docstrings (174-175, 1470, 2603-2613, 4333, 4371, 4419).
- `scripts/gcp_audit.py` — `--max-age-hours` default 24.0 (172-175): reconcile per the coupling above.
- `scripts/cron_gcp_audit.sh` — comments referencing the 24h age backstop.
- `scripts/dispatch_issue.py` — "default 24h" docstrings (774-775, 935-936, 1546-1547).
- `CLAUDE.md` — "default 24h" / janitor "older than 24h" (216, 217, 263); `.claude/rules/background-automation.md` and `.claude/rules/compute-backend-failover.md` if they cite 24h.
- Tests: `tests/test_gcp_backend.py` — `:290` (`default_max_run_duration == "24h"`), `:381` + `:698-699` (`--max-run-duration=24h` in argv), `:2403`/`:2550` (24h age backstop), the FLEX_START 7d cap tests (`:4714+`). Update assertions to the new default AND add a regression test pinning the janitor-age ↔ max-run-duration reconciliation (the test #697 would have needed).

## Constraints / invariants

- **"Max available," NOT unbounded.** Keep a finite credit-leak backstop — removing every fence risks a multi-day idle-GPU billing leak (the #634 zombie / orchestrator-crash class).
- Per provisioning model: FLEX_START ceiling is 7d (code-asserted); SPOT/STANDARD ≤ the same default is fine. Never exceed a GCP ceiling.
- The FLEX_START `>7d` assertion STAYS (it guards a real GCP ceiling).
- Workflow-surface only; ruff on touched files + `uv run pytest tests/test_gcp_backend.py tests/test_router.py` pass; CLAUDE.md stays consistent with the code.
