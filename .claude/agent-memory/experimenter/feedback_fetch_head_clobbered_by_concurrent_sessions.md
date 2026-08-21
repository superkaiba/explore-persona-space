---
name: fetch-head-clobbered-by-concurrent-sessions
description: Never read branch files via `git show FETCH_HEAD:<path>` at the shared repo root — concurrent sessions' fetches silently repoint it mid-task; pin to the SHA
metadata:
  type: feedback
---

Never use `git show FETCH_HEAD:<path>` across multiple tool calls at the
SHARED repo root. FETCH_HEAD is a single mutable file that EVERY concurrent
session's `git fetch` overwrites, so successive reads silently serve
DIFFERENT commits with no error — grep hits vanish, line numbers shift, and
you can verify launch flags against the wrong version of a dispatcher.

**Why:** #2223 ansfirst launch (2026-08-20): after `git fetch origin
issue-2223-ansfirst`, the first `git show FETCH_HEAD:scripts/...` greps
found the `ansfirst` arm table; two calls later the same grep returned
NOTHING and section line numbers had drifted — a concurrent session on the
~15-session VM had re-fetched, repointing FETCH_HEAD to another branch.
Every "verified" fact read in between was suspect and had to be re-checked.

**How to apply:** the moment the fetch returns, capture the immutable ref —
`git rev-parse FETCH_HEAD` (or use the remote-tracking ref
`origin/issue-<N>` the fetch created) — and pin ALL subsequent reads:
`git show <SHA>:<path> > /tmp/<pinned>.py`, then grep/sed the pinned file.
One materialized file also avoids re-running `git show` per probe. This is
the read-pinning discipline of [[cross-session-writer-arbitration]] applied
to the fetch ref itself.
