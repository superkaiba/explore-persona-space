---
name: Pin-vs-manifest commit check for re-extraction plans
description: For reproduction/re-extraction plans that restore pinned code, verify the pin matches the commit the parent artifacts were ACTUALLY produced at (blob-level), not just that the pin exists
type: feedback
---

When a plan restores pipeline code from a pinned commit to reproduce a parent's
artifacts (e.g. #551 restoring `2e6920266` to re-extract #521's shift tensors),
the parent's artifact manifests may record a DIFFERENT run-time `git_commit`
(#521 manifests recorded `4d6978f80`).

**Why:** the pod's HEAD at run time can differ from the plan's pinned commit;
if the restored files differ between the two commits, "byte-for-byte
inheritance" silently breaks and a reproduction-gate breach gets misdiagnosed
as cross-pod nondeterminism.

**How to apply:** compare per-file blob SHAs across both commits:
`git rev-parse <pin>:<file>` vs `git rev-parse <manifest_commit>:<file>` for
every restored file. Identical blobs → pin is sound regardless of the commit
mismatch (this was the #551 case — all 6 files identical). Differing blobs →
Must-Fix: the plan must pin the manifest's commit (or justify the delta).
Cheap (one bash loop), conclusion-relevant only when blobs differ.
