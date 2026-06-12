---
name: codex-stale-branch-artifacts
description: Codex code-reviewer FAILs on file-state findings (file deleted, config reverted, file replaced) that are artifacts of a pre-merge stale worktree snapshot, not changes made by the implementer's branch
metadata:
  type: feedback
---

When Codex's code-review runs against a worktree where the issue-branch is BEHIND main (i.e. the worktree hasn't merged the latest main yet), Codex sometimes flags Critical/Major blockers framed as "this implementation deleted/reverted X" — when in reality, X was added/changed on main AFTER the issue-branch diverged, and the issue-branch simply doesn't have those main commits yet. After the orchestrator merges main into the issue-branch, the "deletions" / "reverts" cease to exist; Codex's findings were comparing the stale worktree state to a memory of what HEAD-of-main should look like.

**Why:** Codex's review compares the worktree's current file tree against its expectation of what should be present (often from CLAUDE.md, prior task context, or scanning the diff). It doesn't always run `git log <branch>..main -- <file>` to check whether a "missing" or "reverted" file is a no-op of the issue-branch or an actual modification by the implementer.

**How to apply:** When Codex flags Critical/Major findings of the shape "file X is deleted/reverted/replaced":
1. Run `git log --oneline main..issue-<N> -- <X>` from the worktree. If ZERO non-merge commits appear, the issue-branch never modified X. The "deletion" is the worktree being stale relative to main.
2. Run `git diff --quiet main -- <X> && echo MATCH` to confirm post-merge state matches main byte-for-byte.
3. Run `git diff --name-only main..issue-<N>` and confirm the diff is scoped to the actual implementation files (the i<N>_* / experiment-namespaced files), not workflow surface or unrelated configs.
4. If the orchestrator already merged main into the issue-branch to resolve, verify the merge commit landed (look for `Merge commit '<sha>' into issue-<N>` in `git log --oneline main..issue-<N>`).
5. Stale-branch findings are correctly classified Unverified/Discarded — they do not exist in the artifact under review.

Concrete incident: task #471 round 1, Codex flagged (Critical) `scripts/pods.conf` reverted + (Major) `scripts/plot_i464_revision_figs.py` deleted. Verification showed: zero non-merge commits on issue-471 touched either file; both byte-identical to main post-merge; the plot file was PRESENT. Codex had reviewed a pre-merge snapshot. Both findings DISCARDED; binding verdict PASS (substantive implementation correctness was independently confirmed: 5 Must-Fixes hold, contrastive negatives wired, label-mask via MarkerOnlyDataCollator tail_tokens=0, marker-id 83399 asserted, i465 vendoring byte-identical via single commit b87f1cab0 from issue-465 branch).

Distinguish from [[feedback_codex_scope_drift_on_repeat_findings]] (Codex lexical-matches a flag in the wrong file but the file IS in the diff) — this pattern is broader: the entire reported file-state is an artifact of branch staleness, not anything the issue-branch did.

Distinguish from [[feedback_codex_litigates_pre_existing_in_round_n]] (Codex correctly IDs a real violation but mis-frames a trunk-unchanged block as a round-N regression) — that pattern involves a real code issue Codex correctly spotted; this pattern involves NO real code issue, just a stale worktree.
