---
name: Never cd out of your own agent worktree
description: Do all reads/edits/git in your OWN worktree dir; cd-ing into a sibling worktree (e.g. an issue-<N> one) reads stale/conflicted state and risks editing the wrong tree
type: feedback
---

Do every Read / Grep / git / edit operation inside YOUR OWN agent
worktree (`git rev-parse --show-toplevel` with NO `cd` gives it). Never
`cd` into a sibling worktree under `.claude/worktrees/` — they hold other
sessions' live state.

**Why:** in the #664 zombie-GPU workflow-fix run, the startup check
correctly showed my worktree was `agent-a5b1a02911fed14f6`, but a habitual
`cd /home/.../worktrees/issue-664` in a Bash call silently sent me into the
LIVE `/issue 664` working tree — which had uncommitted experiment work and
an unresolved `UU` merge conflict. Every analysis I ran with that `cd` read
the WRONG tree (stale `poll_pipeline.py` lacking the merged-to-main zombie
code), producing a contradictory picture for ~6 tool calls before I caught
it by re-running `git rev-parse --show-toplevel` with no cd. An edit there
would have clobbered the live session and stranded my workflow edit on the
`issue-664` branch (never reaching `main`).

**How to apply:**
- Resolve your own worktree once at startup; treat it as the ONLY tree you
  touch. Prefer absolute paths rooted at your worktree in Read/Edit/Write.
- If your branch is behind `main` and the candidate references code merged
  to `main` (workflow-fixes land on `main` continuously), `git merge
  --ff-only main` INTO YOUR OWN worktree branch first so you edit against
  the real current baseline — verify the referenced code is present
  (`grep -F <token>`) before analyzing.
- Grep/Read tool `path` args resolve against the tool's own root, NOT a
  prior Bash `cd` — but Bash `cd` persists nothing between calls and a
  per-call `cd <sibling>` is the trap. If you must `cd`, `cd` back to your
  worktree root in the same compound command.
- Never run the `--ff-only` merge in a sibling worktree (the #664 run did
  this by accident, touching the live issue-664 tree's HEAD).
