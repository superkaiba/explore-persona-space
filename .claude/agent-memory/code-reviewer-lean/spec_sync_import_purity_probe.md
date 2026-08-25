---
name: spec-sync-import-purity-probe
description: Purity-certify a "sync from origin/main" commit via blob-hash TRIPLE equality (commit post-image == origin/main tip == worktree HEAD), not worktree diff alone
metadata:
  type: feedback
---

For a spec-sync / import commit whose review is a PURITY check, certify with `git rev-parse <commit>:<file>` vs `<origin-main-tip>:<file>` vs `HEAD:<file>` — all three blob hashes equal per file — plus a stat check that the commit touches ONLY the listed files.

**Why:** the brief-prescribed probe (worktree `git diff <main-tip> -- <file>` empty) is necessary but not sufficient on its own for the COMMIT's purity: a later round commit could have overwritten a smuggled payload, making the worktree match main while the commit itself carried non-sync content. Triple equality closes that masking channel in one rev-parse per file. Conversely, a non-empty worktree diff does not prove smuggling — the commit may have synced to an origin/main ANCESTOR blob (check `git log origin/main --oneline -- <file>` for the matching ancestor) or a later in-scope commit may own the residual.

**How to apply:** any split-review sub-scope whose commit subject claims a pure import/sync (#2544 r2 g5 shape). Empty worktree diff + triple blob equality + exact file-set stat = PASS with zero diff-content reads; only files failing the probe get bounded diff windows.
