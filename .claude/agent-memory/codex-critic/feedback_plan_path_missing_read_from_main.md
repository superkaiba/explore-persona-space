---
name: Plan path missing — read from main ref
description: When the brief's tasks/<status>/<N>/plans/plan.md path is missing because the shared repo-root HEAD is detached/stale, read the plan read-only via `git show main:<path>`
type: feedback
---

If the brief-handed plan path does not exist in the working tree, do NOT assume the brief is wrong: the shared repo root may be DETACHED at a stale commit (task.py `find` then refuses with "main worktree HEAD is detached", and the task folder may appear at the wrong status or missing files).

**Why:** 2026-07-02 (#882 alternatives compose): repo root was detached with an unmerged path; `tasks/planning/882/plans/` was absent from the checkout but present on `main`.

**How to apply:** recover read-only — `git show main:tasks/<status>/<N>/plans/plan.md` (note `plan.md` may be a tiny symlink blob naming `v<K>.md`; `git ls-tree main -- <dir>` first, then show the real version file). Same for `body.md`. Before telling Codex its working-tree reads are trustworthy, `git diff --stat HEAD main -- <files the critique verifies>`; if non-empty, warn Codex or inline the main-ref content. NEVER re-attach / reset the shared root yourself (destructive-git ban); surface the detached-HEAD state to the orchestrator as a prose note.
