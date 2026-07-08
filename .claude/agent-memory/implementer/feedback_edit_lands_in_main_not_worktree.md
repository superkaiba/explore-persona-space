---
name: Edits can land in MAIN or a SIBLING worktree, not your own
description: Edit/Write land wherever file_path points; the path in context may be the MAIN checkout OR a live sibling worktree (e.g. issue-<N>) named in the prompt — always target your OWN captured $WT
type: feedback
---

Edit/Write tool calls land wherever the absolute `file_path` points — and the
path that appears in the file-content/system context is the MAIN checkout
(`/home/thomasjiralerspong/explore-persona-space/...`), NOT the worktree
(`.../.claude/worktrees/<id>/...`). So if you `Read` a file by its main-checkout
path and `Edit` it, the edit lands in main's working tree, exposed to concurrent
`/issue` committers — the exact incident the worktree mandate exists to prevent.

**SIBLING-worktree variant (worse — caught 2026-06-17, #642):** the prompt /
candidate / gitStatus header may name a DIFFERENT, LIVE worktree (e.g.
`.claude/worktrees/issue-642/...` — the originating `/issue` run's own tree).
It is tempting to `Read`/`Edit` the target file at THAT path because it is the
one quoted in the brief. Doing so strands your edits in a sibling worktree that
another running session owns — even more dangerous than main, because that
session may commit them onto its issue branch. Your OWN agent worktree is
`.claude/worktrees/agent-<hash>/` (from the startup `git rev-parse
--show-toplevel`), and it is the ONLY tree you may edit. The fix content was
correct both times; only the destination tree was wrong.

**Why:** the Bash cwd persists across calls; an early `cd /home/.../explore-persona-space`
(the main checkout) makes `git rev-parse --show-toplevel` resolve to MAIN even
though the startup self-check correctly showed the worktree. All Read/Edit/verify
then operate on main. (Incident 2026-06-15, #612 --check-upload-as-file: caught at
the commit step via an `index.lock` on main's `.git`; no data loss.)

**gitStatus-header trap (caught AGAIN 2026-06-18, #641):** the session gitStatus
header read `Current branch: issue-641` while my actual worktree was
`agent-af5da61596ca3c4ce` — so the brief's `target_file` path and a `Read` of the
file resolved to the LIVE `issue-641` sibling tree, not mine. Both files (src +
test) stranded there, dirtying a running session's tree. Caught by `grep -c` on my
$WT returning 0. Recovered exactly per the recipe below (cp into $WT, `git checkout
--` the sibling clean), THEN discovered my agent branch was 3 commits behind main,
so I `git -C "$WT" merge --ff-only main`'d first and re-applied the edits on the fresh base
(the sibling's gcp.py was ~200 lines diverged from my stale base — re-applying on
main avoids a messy merge). Lesson: a gitStatus header naming an issue branch is
NOT your tree; always trust the startup `git rev-parse --show-toplevel`
(`agent-<hash>`), and FF to main before editing a spawn worktree.

**MIXED-TARGET trap (caught AGAIN 2026-06-18, #653):** the failure can hit only
SOME files in a multi-file change. The script edits correctly used the
`$WT/scripts/...` absolute path; the TEST-file edits used a bare absolute path
(`/home/.../explore-persona-space/tests/...`, copied from a Grep `path` result)
that resolved to MAIN. Caught when `grep -c` on the worktree's test copy
returned 0 while main's returned the hit count. Lesson: a clean script edit does
NOT prove the test edit landed in the same tree — check EACH file's tree
explicitly (`grep -c <token> $WT/<f>`) before running tests, and never trust a
relative/bare path that a Grep/Read result handed you; re-prefix it with `$WT`.

**BASH-HEREDOC-APPEND variant (caught 2026-06-28, #681 r4):** the trap is not
limited to the Edit/Write tools — a `cat >> tests/<f> <<'EOF' ... EOF` (or any
relative-path shell redirect) run inside a `Bash` call whose cwd is the main
checkout appends to MAIN's copy, not the worktree's. Here the WT was the cwd for
Edit tools but a `cd /home/.../explore-persona-space` (main) earlier in the
session made the relative `tests/<f>` resolve against main. Caught by `grep -c
<token>` on the WT copy returning 0 while main's returned 1, and by `git -C
<main> status --short` showing the file modified. Recovered by Edit-appending the
SAME block into `$WT/tests/<f>` (via the Edit tool, absolute WT path) and `git -C
<main> checkout -- tests/<f>`. Lesson: prefer the Edit tool (absolute `$WT` path)
over `cat >>` for appends; if you must use a shell redirect, write the absolute
`$WT/<relpath>`, never a bare relative path.

**How to apply:**
- At startup, capture the worktree root once: `WT=$(git rev-parse --show-toplevel)`
  BEFORE any `cd`. Target EVERY Read/Edit/Write at `$WT/<relpath>`, never the bare
  main-checkout path AND never a sibling worktree path quoted in the brief (e.g.
  `issue-<N>`) — even though the context / candidate shows one of those paths.
  The brief names other trees only for CONTEXT (the originating run); they are
  read-only-by-courtesy, edit-forbidden.
- Do verification (`uv run python scripts/workflow_lint.py ...`, pytest, ruff)
  with `cd "$WT"` or `git -C "$WT"`, so you test the worktree copies.
- Recovery if edits already landed in main (uncommitted): confirm the 3 files'
  worktree-HEAD blob == main-HEAD blob (`git -C <wt> rev-parse HEAD:<f>` vs
  `git -C <main> rev-parse HEAD:<f>`), `cp` the modified main files into the
  worktree, `git -C <main> checkout -- <files>` to restore main clean, then
  re-verify + commit IN the worktree. Safe only when the sole changes to those
  files are yours.
