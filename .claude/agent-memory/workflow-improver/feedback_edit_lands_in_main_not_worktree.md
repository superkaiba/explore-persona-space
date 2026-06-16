---
name: Edits can land in MAIN checkout not the worktree
description: Edit/Write use the absolute path in the file-content context (the MAIN checkout); a cd into main makes all edits + verification hit main, not the worktree
type: feedback
---

Edit/Write tool calls land wherever the absolute `file_path` points — and the
path that appears in the file-content/system context is the MAIN checkout
(`/home/thomasjiralerspong/explore-persona-space/...`), NOT the worktree
(`.../.claude/worktrees/<id>/...`). So if you `Read` a file by its main-checkout
path and `Edit` it, the edit lands in main's working tree, exposed to concurrent
`/issue` committers — the exact incident the worktree mandate exists to prevent.

**Why:** the Bash cwd persists across calls; an early `cd /home/.../explore-persona-space`
(the main checkout) makes `git rev-parse --show-toplevel` resolve to MAIN even
though the startup self-check correctly showed the worktree. All Read/Edit/verify
then operate on main. (Incident 2026-06-15, #612 --check-upload-as-file: caught at
the commit step via an `index.lock` on main's `.git`; no data loss.)

**How to apply:**
- At startup, capture the worktree root once: `WT=$(git rev-parse --show-toplevel)`
  BEFORE any `cd`. Target EVERY Read/Edit/Write at `$WT/<relpath>`, never the bare
  main-checkout path — even though the context shows the main path.
- Do verification (`uv run python scripts/workflow_lint.py ...`, pytest, ruff)
  with `cd "$WT"` or `git -C "$WT"`, so you test the worktree copies.
- Recovery if edits already landed in main (uncommitted): confirm the 3 files'
  worktree-HEAD blob == main-HEAD blob (`git -C <wt> rev-parse HEAD:<f>` vs
  `git -C <main> rev-parse HEAD:<f>`), `cp` the modified main files into the
  worktree, `git -C <main> checkout -- <files>` to restore main clean, then
  re-verify + commit IN the worktree. Safe only when the sole changes to those
  files are yours.
