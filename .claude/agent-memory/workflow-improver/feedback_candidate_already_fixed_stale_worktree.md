---
name: Candidate may describe a stale-worktree spec — check main first
description: Before applying a workflow-fix candidate, git-log the target file on main; the incident may stem from a worktree's pre-fix copy of an already-hardened spec
type: feedback
---

Before refining a candidate's diff_sketch, run `git log --oneline -5 -- <target_file>` and grep main's copy for the proposed rule. Auto-spawn candidates are emitted from sessions whose cwd is an ISSUE WORKTREE, and the `.claude/agents/*.md` specs the harness loaded there are frozen at branch-cut time — so the "missing rule" the candidate proposes may already be on main in a stronger form, and the incident is really a stale-spec propagation failure.

**Why:** Task #557 r2 (2026-06-10): candidate proposed adding a dispatch rule to `codex-code-reviewer.md` that had landed on main 12 hours earlier (`bd26e7b0d`, compose-only contract); the issue-557 worktree's copy predated it. Blindly applying the sketch would have WEAKENED the existing rule (the sketch allowed foreground wrapper dispatch; main mandates compose-only).

**How to apply:** When the rule already exists, apply only the residual value (recurrence citation, recovery recipe, missed sibling files) and surface the propagation root cause (stale worktree copies of workflow-surface files at ensemble-spawn time) as a follow-up.
