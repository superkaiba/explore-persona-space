---
name: Creation-helper reuse shortcut vs half-created state
description: Create-or-reuse helpers under set -e must handle failed-halfway-then-retry; a reuse-as-is exit-0 silently hands live sessions a broken artifact (#596)
type: feedback
---

When an infra plan introduces a helper of the shape "if <artifact> already registered → reuse as-is, exit 0; else create via a multi-step sequence under `set -e`", check the failed-halfway-then-retry sequence: step k fails (ENOSPC is realistic when the task's motivation IS disk pressure) → artifact registered but broken → retry hits the reuse branch → exit 0 → live session proceeds on the broken artifact. If the test suite pins "second invocation exits 0 reusing as-is", the gap gets entrenched → Must-Fix at plan time.

**Why (#596, sparse-checkout worktrees):** helper did `worktree add --no-checkout` → `sparse-checkout init/set` → `checkout`; failure after registration left an empty-tree worktree that porcelain shows as all-deleted (so worktree_audit KEEPS it) and the reuse path would exit 0 on retry; `git commit -a` from that state would commit mass deletions.

**How to apply:** demand one of (i) ERR-trap cleanup deregistering the half-created artifact (and deleting a just-created branch) so retry recreates cleanly, or (ii) reuse-path validation (HEAD resolves + a sentinel materialized path exists) that fails loudly / repairs instead of exit-0. Check the adjacent registered-but-directory-deleted case; require a test item for kill-after-step-1-then-retry.
