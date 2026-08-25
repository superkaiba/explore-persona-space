---
title: issue-tick dies ImportError in any worktree whose src/ predates a task_workflow
  symbol (cwd not pinned to main)
kind: infra
tags:
- issue-tick
- worktree-staleness
created_at: '2026-08-24T14:19:00Z'
has_clean_result: false
origin_prompt: tick_triage.py resolved from main but run with worktree cwd imported
  the worktree's stale task_workflow.py and died ImportError exit 2 during a /issue-tick
  2254 fire.
workflow: v1
---
## Goal

The `/issue-tick` contract resolves `tick_triage.py` from the MAIN checkout but runs it with the session's cwd, which for a per-issue session is the WORKTREE. `uv run` then resolves the `explore_persona_space` package from the worktree's `src/`, which is fork-era by design — the Step 5a spec-freshness sync deliberately does not cover `src/`. A fresh `tick_triage.py` importing a symbol added to `task_workflow.py` after the worktree was cut dies with ImportError, exit 2.

## Evidence (observed live, 2026-08-24, issue-2254 worktree)

The documented invocation form failed:

    REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
    uv run python "$REPO_ROOT"/scripts/tick_triage.py 2254
    -> tick_triage: FAILED for #2254: cannot import name 'open_async_ask' from
       'explore_persona_space.task_workflow'
       (/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2254/src/explore_persona_space/task_workflow.py)

Note the resolved path: the WORKTREE copy. Confirmed by symbol count — `grep -c 'def open_async_ask'` gives 1 in the main checkout and 0 in the worktree.

Pinning cwd to main fixes it:

    uv run --directory "$REPO_ROOT" python scripts/tick_triage.py 2254
    -> STALE-REDRIVE status=followups_running, marker age 3m -- api-error-after-marker

## Why it matters

The tick's contract says a non-zero exit is treated as STALE-REDRIVE, which fails toward coverage, so this is not silent. But the cost is real: every fire in an affected worktree loads the full `/issue` skill instead of making one cheap Bash call, which is exactly the context burn the 2026-06-12 redesign existed to eliminate (#522: seven full re-drives in four hours exhausted a session). It also masks the true verdict — in the observed case the real verdict was an `api-error-after-marker` STALE-REDRIVE, which happens to coincide, but a HEALTHY or GATE-TRANSITION verdict would have been converted into a spurious full re-drive, and a GATE-TRANSITION conversion would SKIP the phone push and the cron teardown.

The blast radius is every per-issue worktree older than the newest `task_workflow.py` symbol `tick_triage.py` imports — which grows monotonically as main advances.

## Fix sketch

Change the documented invocation in `.claude/skills/issue-tick/SKILL.md` (§ Contract — guarded no-op tick) from `uv run python "$REPO_ROOT"/scripts/tick_triage.py <N>` to `uv run --directory "$REPO_ROOT" python scripts/tick_triage.py <N>`, so both the script AND the package resolve from main. Audit sibling call sites for the same pattern — any tick-turn or watcher shell-out that runs a main-checkout script from a worktree cwd inherits this. Consider having `tick_triage.py` assert at startup that the imported `task_workflow` module path lives under the main checkout, and fail with a diagnostic naming the cwd-pinning fix rather than a bare ImportError.

## Provenance

Found while running the mandated first-action triage call during a `/issue-tick 2254` fire, 2026-08-24. The skill's existing worktree guidance covers script RESOLUTION (and warns against `--show-superproject-working-tree`) but not package resolution.
