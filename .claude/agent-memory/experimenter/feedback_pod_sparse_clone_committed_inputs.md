---
name: pod-sparse-clone-committed-inputs
description: Pod bootstrap clones are SPARSE (eval_results/* excluded except own issue) — drivers hard-reading committed parent-issue inputs crash at startup; materialize via git show HEAD:<path>
metadata:
  type: feedback
---

Pod bootstrap clones use a sparse-checkout cone (`configs data docs
eval_results/issue_<N> figures/issue_<N> scripts src tests`) — every OTHER
issue's `eval_results/issue_<M>/` is EXCLUDED from the working tree even
though the files are in the index/tree (`git ls-files` lists them,
`ls` finds nothing). A driver that hard-reads a committed parent-issue
input (e.g. #2476's `COMMITTED_SPLIT_1482 =
eval_results/issue_1482/split_1482.json` + `matryoshka_tier/m_split.json`)
dies in seconds with FileNotFoundError despite a verified HEAD.

**Why:** the item-4 input gate must stat-check the driver's hard-coded
committed reads ON THE POD (not just confirm they are git-tracked) —
"tracked at the launch SHA" does NOT imply "materialized in a sparse pod
checkout". Incident #2476, 2026-08-23: first launch crashed ~5s in; fix +
relaunch cost one round.

**How to apply:** pre-launch, grep the driver for `PROJECT_ROOT /
"eval_results"` (and any committed cross-issue constants) and `ls` each on
the pod. On a miss, materialize the exact committed bytes from the pod's
own odb — `git show HEAD:<path> > <path>` (mkdir -p parent first) — then
verify `git hash-object <path>` equals `git ls-tree HEAD -- <path>`'s blob
(the #1112 MooseFS content check). Prefer per-file `git show` over
`git sparse-checkout add <dir>` when the dir is large (issue_1482 was
311 MB / 247 files through MooseFS FUSE — wedge risk) — sparse-add only
when the driver reads many files from the subtree. Related:
[[gcp-lane-git-clone-only-data]] (the git-clone-only lane sibling).
