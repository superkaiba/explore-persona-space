---
title: worktree_audit.py stale-sweep reaps an active inline-round worktree (pod-side
  work + parked parent look idle)
kind: infra
tags: []
created_at: '2026-08-17T20:18:26Z'
has_clean_result: false
origin_prompt: 'surfaced by #2223 strsweep inline-override round 2026-08-17'
workflow: v1
---
# worktree_audit.py stale-sweep reaps an ACTIVE inline-round worktree (pod-side work + task parked look idle)

**Surfaced by:** the #2223 intervention-strength-sweep inline-override round (interactive chat), 2026-08-17.

**Gap:** the daily stale-worktree sweep (`scripts/worktree_audit.py --apply`, run via `cron_worktree_audit`) reaped `.claude/worktrees/issue-2223-casestudy` WHILE an active inline round owned it. The round looked idle to the audit's liveness heuristic because: (a) the round's compute ran on `pod-2223-strsweep` (no VM-side process to detect); (b) the parent task #2223 sat at `awaiting_promotion` (a terminal-ish status, not an ACTIVE lifecycle status); (c) the worktree tree was momentarily clean (work committed to the branch + pushed). So the "live-process / active-issue / dirty kept" gates all read false and the worktree was removed mid-round. No data was lost (the 72 cells were safe on the pod + committed to origin/issue-2223-casestudy), but the round had to RECREATE the worktree mid-deliverable — and the manual recreation used a custom sparse cone that omitted `.claude/`, which then crashed the `workflow-lint-phase-done-reserved` pre-commit hook (`workflow_lint.py` reads `.claude/config/agent_spec_size_caps.txt` at import → FileNotFoundError) on the next commit, costing a second recovery.

**Why this is a standing gap, not a one-off:** an inline-override / same-issue follow-up round routinely (i) runs its compute on a pod (no VM process), (ii) operates on a parent at a parked/terminal status (`awaiting_promotion`, `on_hold`, `completed`), and (iii) keeps its worktree tree clean between commits. That triple is exactly the shape the audit's idle heuristic treats as reapable — so ANY active inline round on a parked parent is exposed to having its worktree reaped out from under it.

**Proposed fix:** `worktree_audit.py` should treat a worktree as ACTIVE (skip reap) when ANY of:
  - the parent task has a fresh follow-up signal marker (`epm:run-launched` / `epm:followup-scope` / `epm:free-analysis-followup-run`) NEWER than the latest done-transition, OR the `keep-running` tag is set (the SAME predicate the watcher's pod-safety pass already uses to shield pods — reuse it for worktrees);
  - the worktree branch has commits not on `origin/<branch>` OR uncommitted/untracked round work (`git -C <wt> status --porcelain`), even when the tree is momentarily clean between commits, prefer a short activity grace window keyed on last branch-commit mtime.
Human-named worktrees are already never touched; this only widens the "keep" set for auto-generated issue worktrees whose round is provably in flight.

**Confidence:** high (reproduced live: the worktree was reaped and had to be recreated by hand this round).

**Target file:** `scripts/worktree_audit.py` (+ the shared follow-up-signal predicate in `scripts/autonomous_session_watch.py` / `tick_triage.py` it should reuse).

**Regression pin to add:** a fixture where a worktree's parent task carries a fresh `epm:run-launched` newer than its latest done-transition (or `keep-running`) → `worktree_audit` KEEPS it even when the tree is clean and no VM process matches.
