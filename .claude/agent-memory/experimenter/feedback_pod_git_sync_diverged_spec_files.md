---
name: Same-pod relaunch — divergent .claude/** spec files block git pull --ff-only
description: On a same-pod relaunch of an issue-<N> branch carrying a "spec-freshness" sync commit, the pod's existing .claude/** workflow-surface files diverge from the branch version and `git pull --ff-only` aborts ("would be overwritten by merge") instead of fast-forwarding. Recover surgically: discard tracked-at-HEAD files via `git checkout HEAD -- <file>`, `rm -f` untracked would-be-overwritten new files, then re-pull. Path-scoped/ancestry checks only (MooseFS is FUSE-slow). Sibling of the committed-pod-artifacts pull block, but spec files are discarded, not backed up.
type: feedback
---

**Rule.** If `git pull --ff-only` aborts on a same-pod relaunch with "Your
local changes to the following files would be overwritten by merge" (or
"untracked working tree files would be overwritten by merge") naming files
under `.claude/**`, the branch diff carries a **spec-freshness** sync commit
that refreshed the workflow-surface specs, and the pod's existing copies
diverge. Do NOT force, reset --hard, or skip the sync. Discard the divergent
spec files (they are inputs the pod never owns) and re-pull:

```bash
# 1. Stale-lock guard — only after confirming no live git process
pgrep -x git >/dev/null || rm -f .git/index.lock

# 2. Identify the divergent .claude/** files (path-scoped diff — fast on MooseFS)
git fetch origin issue-<N> --quiet
git diff --name-only HEAD origin/issue-<N> -- .claude/ > /tmp/diverged.txt

# 3. Tracked-at-HEAD → discard local edits; untracked (new-on-branch) → remove
while read -r f; do
  if [ -n "$(git ls-tree -r HEAD --name-only -- "$f")" ]; then
    git checkout HEAD -- "$f"          # exists at HEAD: revert pod-side edits
  else
    rm -f "$f"                          # UNTRACKED would-be-overwritten new file
  fi
done < /tmp/diverged.txt

# 4. Now the fast-forward succeeds
git pull --ff-only origin issue-<N>
```

**Why:** Task #653 round-5 relaunch (2026-06-16). The branch diff carried a
spec-freshness sync commit refreshing `.claude/**`; the pod's existing spec
copies diverged, so the pre-launch `git pull --ff-only` aborted to protect
the local edits and the sync failed with an unhelpful error. Recovered
manually with this sequence.

This is the `.claude/**`-spec **sibling** of
`feedback_committed_pod_artifacts_block_pull.md` — that one covers
pod-WRITTEN `eval_results/issue_<N>/...` artifacts blocking the pull and the
fix is back-up-outside-the-repo-then-remove (the pod-written data is the only
copy). Here the diverging files are workflow SPECS the pod never authors, so
the fix is the cheaper discard-to-HEAD: no backup, just `checkout HEAD` /
`rm`. Tell them apart by which file class appears in the abort message
(`.claude/**` → this note; `eval_results/**` → the sibling note).

**How to apply.**

1. The tracked/untracked split is load-bearing: `git checkout HEAD --
   <pathspec-not-at-HEAD>` errors with rc=1 and reverts NOTHING (one bad
   pathspec aborts the whole checkout), so a new-on-branch file MUST be
   `rm`'d, not checked out. The `git ls-tree -r HEAD --name-only -- "$f"`
   per-file probe is empty exactly when `$f` is absent at HEAD (the
   `rm` branch).

2. Stay path-scoped — MooseFS/FUSE makes an unscoped `git status` /
   `git diff-index` take ~30s+, and SSH-MCP `ssh_execute` caps commands at
   ~30s, so an unscoped status can itself time out. Use
   `git diff --name-only ... -- .claude/` and `git merge-base
   --is-ancestor` for cheap checks; per-file `git ls-tree ... -- "$f"`
   instead of piping the full tree through `grep`.

3. Expect this on EVERY same-pod relaunch whose branch diff touches
   `.claude/**` (a spec-freshness commit landed between the original launch
   and the relaunch). Apply the discard-remove-pull sequence proactively
   rather than treating the ff-only abort as a hard blocker.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Divergent .claude/** spec files block ff-only pull](feedback_pod_git_sync_diverged_spec_files.md) — same-pod relaunch of a branch carrying a spec-freshness sync commit aborts `git pull --ff-only`; `checkout HEAD` tracked spec files + `rm -f` untracked ones (path-scoped/ancestry checks; MooseFS-slow), then re-pull. Spec sibling of the committed-pod-artifacts block but discard, don't back up (#653 r5)
