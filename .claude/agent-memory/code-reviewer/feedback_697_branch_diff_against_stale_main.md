---
name: Long-lived issue branch — diff against the implementer's commit range, not main
description: For a branch that forked off an old main, git diff main..HEAD is polluted with unrelated main history; scope to the implementer's own commit range and use git show <branch>:path
type: feedback
---

When reviewing a long-lived `issue-<N>` branch (e.g. #697, branched weeks ago),
`git diff --name-only main...HEAD` (or `main..HEAD`) returns a HUGE polluted diff
(hundreds of unrelated files: other issues' eval_results deletions, workflow-surface
churn, task-state moves) because main has moved far forward since the branch point.

**Why:** the worktree is often sparse and the repo root sits on `main` (HEAD ≠ the
issue branch). `git show HEAD:path` then resolves against `main` and errors
`path does not exist in HEAD`.

**Do:** scope to the implementer's own commits. The report cites them
(`git diff --stat <parent>..HEAD`, e.g. `dde06e5cae..HEAD`). Then:
- True per-file delta: `git show <parent>:path > /tmp/old && git show issue-<N>:path > /tmp/new && diff -u /tmp/old /tmp/new` (use the BRANCH ref `issue-<N>`, never `HEAD`, when repo root is on main).
- To run tests/lint against the branch tip without switching the repo-root branch: `git worktree add --detach /tmp/cr<N>-wt issue-<N>`, run `uv run pytest/ruff` there, then `git worktree remove --force`. (Repo root MUST stay on main — CLAUDE.md hard rule.)

**Don't:** treat the main...HEAD file list as the review scope, and don't FAIL on
the merge-base / "no merge base" / huge-diff artifact — it is a checkout/branch-age
artifact, never a review finding (code-reviewer.md Step 0).
