---
name: hf snapshot_download full-tree enumeration trap (huge data repo)
description: Never stage a subtree of the ~1M-file project data repo via snapshot_download + allow_patterns — it enumerates the WHOLE repo tree first; use scoped list_repo_tree + a small hf_hub_download pool
type: feedback
---

`huggingface_hub.snapshot_download(repo_id, allow_patterns=[...])` fetches the
FULL repo file listing BEFORE applying `allow_patterns`. Against
`superkaiba1/explore-persona-space-data` (~1M files) that enumeration is
effectively unbounded under the 2500 req/5min API quota — a GCE staging step
sat 40+ min at 4.6% CPU with zero files landed (#833 round 7a, 2026-07-03).

**Why:** allow_patterns is client-side filtering; the tree walk is paginated
over the whole repo. `list_repo_tree(repo_id, path_in_repo=<prefix>,
recursive=True)` is server-side scoped and returns in seconds.

**How to apply:** to stage a prefix from the data repo, enumerate with scoped
`list_repo_tree`, then download per-file via `hf_hub_download` in a small
thread pool (≤6 workers, retry with backoff — ONE process with modest threading
stays under the request quota; 9 concurrent PROCESSES tripped it in #833 round
3). A single-tarball bundle is the cheaper design when uploads are possible,
but is unavailable while the namespace sits at its public-storage quota (LFS
uploads 403 — #552/#541 class); small JSON/text uploads still ride the non-LFS
path. Working recipe: `scripts/issue833_gcp_phase_d.sh` (commit 22388e4b3d).
