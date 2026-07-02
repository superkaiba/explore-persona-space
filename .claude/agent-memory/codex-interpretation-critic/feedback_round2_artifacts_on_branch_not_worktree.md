---
name: Round-N artifacts live on the issue branch SHA, not the working-tree checkout
description: Revision-round artifacts (new JSON cells, regenerated figures, new scripts) committed by the analyzer on the issue-<N> branch may not be in the worktree main checkout the Codex sandbox reads; materialize them from the cited SHA into /tmp and point Codex there
type: feedback
---

In a revision round (round 2+), the analyzer commits the new/regenerated
artifacts (e.g. new `g1_vs_greal` cells inside `a7_precondition.json`, the
script that computed them, regenerated figures) on the `issue-<N>` branch at a
specific SHA cited in the `epm:interpretation vN` change-log. That SHA can be
NOT an ancestor of the working-tree HEAD (the worktree may sit on `main` at an
older state). Codex's read-only sandbox is rooted at the working tree, so it
would read the round-1 versions: the new cells absent, the figures stale.

Why: this is the interp-round analogue of the codex-code-reviewer #489/#550
unreachable-input false-BLOCKED class. The reviewer fires a FALSE REVISE on an
artifact that genuinely exists in the round-N commit — often re-raising its own
prior-round revision request that the analyzer actually DID fix.

How to apply: in Step 2/3 of composing the prompt, before pointing Codex at
`eval_results/...` and `figures/...` in the working tree, CHECK whether the
round-N SHA from the body's change-log is an ancestor of HEAD
(`git merge-base --is-ancestor <SHA> HEAD`). If NOT, materialize the round-N
versions of every artifact the body's claims reference into a temp dir mirroring
the repo layout (`git show <SHA>:<path> > /tmp/.../<path>`), copy the
unchanged-this-round files from the worktree (a `git show` of a path the round-N
commit did not touch FAILS — those files are identical to the prior round, so
`cp` them from the worktree), verify the new cells/values are present + match the
body, and point Codex's read targets at the /tmp tree with an explicit prompt
note: "READ THE /tmp COPIES — they ARE the round-N committed artifacts; do NOT
REVISE on 'cells missing from eval_results/' (that reflects the worktree
checkout, not the analyzer's round-N work)." Also verify the materialized figure
byte sizes match the `git show --stat` of the round-N commit so you know you
pulled the right versions.

Watch-out: `git show <SHA>:<path>` writes empty/error content if the path was not
changed in that commit — re-copy those from the worktree afterward, and assert
no materialized file is 0 bytes before finishing.
