---
name: memory-sweep-commit-dangling-pointer-repair
description: A #2332 post-round memory-sweep commit adding a file WITHOUT touching MEMORY.md is often correct — the index row landed in a PRIOR commit as a dangling pointer; verify with cat-file -e at parent and HEAD before flagging inconsistency
metadata:
  type: feedback
---

When reviewing a housekeeping commit that sweeps stranded agent-memory writes
(#2332 hygiene: memory files stay uncommitted in the worktree, swept post-round),
an added `<memory>.md` with NO accompanying `MEMORY.md` edit is not automatically
an index inconsistency: the index row may have been committed in an EARLIER round
(the row then dangled at the parent), and the sweep commit is the REPAIR.

**Why:** #2479 r9 g2 — the sweep added `eligibility_export_call_chain_identity.md`
with no MEMORY.md hunk; the row already sat at parent 9d95660c59 line 68. Flagging
"file added without index row" would have been a false positive; the real check is
bidirectional dangling-pointer resolution at HEAD.

**How to apply:** for each swept dir: (1) `git show HEAD:<dir>/MEMORY.md | grep <slug>`
per added file; (2) `git cat-file -e HEAD:<dir>/<target>` for every row whose target
was NOT in the diff (catches rows still dangling); (3) confirm the index change is
append-only (insertions with 0 deletions) so no sibling rows were dropped
([[index-overwrite-orphans-sibling-memories]]); (4) post-commit
`git status --porcelain -- .claude/agent-memory/` should be clean in the swept dirs.
