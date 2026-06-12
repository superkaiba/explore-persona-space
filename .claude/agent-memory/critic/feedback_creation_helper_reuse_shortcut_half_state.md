---
name: Creation-helper reuse shortcut vs half-created state
description: Infra plans with a create-or-reuse helper (set -e, multi-step creation) must handle the failed-halfway-then-retry sequence; reuse-as-is exit-0 silently hands live sessions a broken artifact
type: feedback
---

Rule: when an infra plan introduces a creation helper of the shape "if
<artifact> already registered/exists → reuse as-is, exit 0; else create via
a multi-step sequence under `set -e`", check the failed-halfway-then-retry
sequence explicitly. Step k of creation fails (ENOSPC is the realistic one
when the task's whole motivation is disk pressure) → artifact is registered
but in a broken intermediate state → orchestrator retries → reuse branch
exits 0 → live session proceeds on the broken artifact. If the plan's own
test suite pins "second invocation exits 0 reusing as-is", the gap gets
entrenched and code-reviewer recovery is unlikely → Must-Fix at plan time.

**Why:** #596 (sparse-checkout worktrees, 2026-06-11): helper did
`worktree add --no-checkout` → `sparse-checkout init/set` → `checkout`;
failure after registration left an empty-tree worktree that porcelain shows
as all-deleted (so worktree_audit KEEPS it — `_has_tracked_changes` True),
and the reuse path would exit 0 on retry. Broke acceptance criterion 3 for
the live run; `git commit -a` from that state would commit mass deletions.

**How to apply:** demand one of: (i) ERR-trap cleanup that deregisters the
half-created artifact (and deletes a just-created branch) so retry recreates
cleanly, or (ii) reuse-path validation (HEAD resolves + a sentinel
materialized path exists) that fails loudly / repairs instead of exit-0.
Also check the adjacent registered-but-directory-deleted case. Require a
test item for the kill-after-step-1-then-retry sequence.
