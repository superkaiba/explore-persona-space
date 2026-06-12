---
name: uv-sync-moosefs-stale-handle-persistent
description: uv sync on MooseFS /workspace can fail errno 116 "Stale file handle" REPEATEDLY (different wheel each retry); the partial .venv seeds new failures. rm -rf .venv before retry, UV_LINK_MODE=copy, epm:failure infra after the 2nd failure.
metadata:
  type: feedback
---

`uv sync` on MooseFS `/workspace` can errno-116 repeatedly, NOT just transiently — #475 launch (2026-06-03) failed at nvidia-nvtx, then after the gotchas.md single-retry, again at torch's install/copy. MooseFS chunkservers invalidate inode handles mid-write during large copies, and re-walking the grown partial .venv racks up new stale handles faster than a clean tree.

**How to apply:**
1. Don't trust retry-once when the first attempt died mid-INSTALL after `Prepared N packages`. `rm -rf .venv` first — the wheel cache survives, the install walks a clean tree. Set `UV_LINK_MODE=copy` explicitly.
2. If the post-rm retry still errno's, STOP retrying on that mount: post `epm:failure v1 failure_class: infra` with both log paths, recommending a non-MooseFS venv path (e.g. `/root/.venvs/eps`) or a fresh pod — the chunkserver may be sick for that pod-host pair.
3. Budget: healthy sync ≈ 3 min download + 3-5 min install; >20 min in this loop → cut losses and post the marker. Distinct from the MooseFS EDQUOT quota gotcha (errno 122), similar moral.
