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

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Shallow pod clones break git-log fix checks](feedback_shallow_clone_fix_commit_verification.md) — depth-1 clones attribute every path to the boundary commit; verify fix commits by content hash (pod sha256sum vs git show <sha>:<path>), never git-log path attribution (#779)
