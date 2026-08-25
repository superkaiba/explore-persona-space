---
name: reaped-cache-staging-cost
description: A leg inheriting a parent's MEASURED fit walls after the parent's hf_dl caches were reaped carries a data-staging cost the inherited basis does not contain — size + place the staging explicitly
metadata:
  type: feedback
---

When a plan reuses a parent's measured per-cell fit walls (pilot reports, fanout runbooks), check whether the parent's staged INPUTS still exist on local disk. The parent measured its walls with stores already on disk; `data/issue_<M>/hf_dl/` caches are re-downloadable by design and reaped at Step 8 — so the inherited measured basis prices FITS ONLY, never the re-staging.

**Why:** #2388 round 2 (2026-08-19): plan v3 priced an H3 recompute from parent #1739's measured group walls with NO staging row; the reaped inputs totaled 157.78 GB on HF — past the ~10 GB download-to-pod threshold, the 50 GB VM ceiling, AND the ~130 GB per-pod MooseFS quota. Registering it as a "cheap VM re-run" would have EDQUOT-failed mid-stage on the second behavior. One number in that chain was also wrong because it was composed from the plan's §9 prose instead of read from the artifact (the pilot reports) — read sizes/walls from the artifacts, never from a plan's derived prose.

**How to apply:** any §9 row whose fit basis is inherited from a parent gets a companion STAGING row — inputs located + sized on the Hub (scoped `list_repo_tree`, never guessed), destination mount named (pod `/workspace`, never `/`), per-behavior sequential stage→fit→delete when the sum exceeds the pod quota, and a measured-first-stage wall pilot. Related: [[1739-ctxmap-reuse-artifacts]].
