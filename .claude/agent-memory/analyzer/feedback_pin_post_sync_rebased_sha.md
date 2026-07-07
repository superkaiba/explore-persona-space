---
name: Pin the post-sync rebased SHA for main-committed figures
description: sync_repo_root.py rebases a rejected main push — the local pre-push commit SHA dies; re-resolve the SHA from origin/main before pinning figure URLs
type: feedback
---

When committing figures to the shared repo root (main) and the push is rejected, `scripts/sync_repo_root.py` recovers via pull --rebase — which REWRITES your commit to a NEW SHA. The local pre-push SHA (`git rev-parse HEAD` taken right after the commit) then exists on no pushed ref, and a raw.githubusercontent URL pinned to it 404s.

**Why:** #833 round-2 (2026-07-06): figures committed on main as `044af6e7a5`, push rejected (behind), sync_repo_root rebased it to `8edbecd046` — the pre-push SHA would have produced dashboard-dead figure links that pass no local check.

**How to apply:** after ANY sync_repo_root recovery (or any pull --rebase before push), re-resolve the figure commit via `git log origin/main --oneline -- figures/issue_<N>/<file>.png` and pin THAT SHA; then curl -sI the raw URL (expect HTTP 200) before writing it into the body. Also remember bash cwd resets between calls on this harness — use `git -C <abs path>` so the resolve runs against the repo root, not the worktree.
