---
name: Shallow pod clones break git-log fix-commit checks
description: Fresh pod clones are depth-1 — `git log -1 -- <script>` attributes every path to the boundary commit, so "log shows <fix-sha>" false-fails; verify fix commits on pods by content hash instead
type: feedback
---

Fresh pod clones are shallow (depth-1), so `git log -1 -- <script>`
attributes every path to the boundary commit and a "log shows <fix-sha>"
check false-fails (#779 pod-77903, 2026-07-03: the script path showed an
unrelated task-marker boundary commit instead of the real fix commit
7b35377edc). Verify fix commits on pods by CONTENT hash — pod
`sha256sum <file>` vs VM `git show <sha>:<path> | sha256sum` — and do not
classify the log mismatch as a sync failure when content matches.

**How to apply:** any pre-launch "is the fix on the pod" check on a freshly
provisioned pod uses the content-hash comparison, never `git log` path
attribution; a `git log` mismatch alone is not evidence the pod is stale.
