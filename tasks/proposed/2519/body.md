---
title: tick_triage.py from main imports task_workflow symbols absent in worktree cwd
  resolution (open_async_ask ImportError)
kind: infra
tags: []
created_at: '2026-08-24T05:51:01Z'
has_clean_result: false
workflow: v1
---
Observed 2026-08-24 in the issue-1739 session: the /issue-tick contract runs 'uv run python $REPO_ROOT/scripts/tick_triage.py <N>' from the WORKTREE cwd. uv run resolves the package from the worktree's src/, so main's tick_triage.py (which now imports open_async_ask from explore_persona_space.task_workflow) crashes with ImportError against the worktree's branch-pinned task_workflow.py — exit 2, which the tick skill maps to STALE-REDRIVE (a spurious full-skill re-drive on every tick for any worktree older than the symbol). Fix options: (a) make tick_triage.py import open_async_ask lazily/guarded (getattr with fallback) so older worktree packages degrade gracefully; (b) have the tick contract run the triage with cwd pinned to REPO_ROOT (uv run --project $REPO_ROOT or cd $REPO_ROOT &&) so both script AND package resolve from main. Prefer (b) + a defensive (a). Repro: from any worktree whose src predates open_async_ask, run the tick command.
