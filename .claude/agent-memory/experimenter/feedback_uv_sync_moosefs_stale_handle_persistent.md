---
name: uv-sync-moosefs-stale-handle-persistent
description: uv sync on MooseFS /workspace can fail with errno 116 "Stale file handle" repeatedly, not just transiently — on different wheels each attempt (nvidia-nvtx → torch-2.11.0). The single-retry-and-proceed pattern from gotchas.md is insufficient when the partial .venv keeps growing across retries.
metadata:
  type: feedback
---

`uv sync` on MooseFS `/workspace` can fail with errno 116 "Stale file
handle" repeatedly across retries, NOT just as a one-shot transient.
Burned at task #475 launch (2026-06-03): first attempt failed at
`nvidia-nvtx-13.0.85` during install/copy (after 3+ min download + 1+ min
prep); the per-gotchas.md single retry resolved-and-progressed, then died
at `torch-2.11.0` during its install/copy on a different temp file
(`/workspace/.../torch/include/ATen/ops/.tmpebNsdE`).

**Why:** MooseFS chunkservers can intermittently invalidate inode handles
mid-write during a large multi-GB cp; uv copies wheel contents into a
`.tmp*` rename-target inside `.venv/lib/python3.11/site-packages/...`.
When the second uv sync re-walks the partial .venv (16G after first
failure), it racks up new stale handles faster than a clean tree.

**How to apply:**
- Don't trust the simple "retry once" recipe for `uv sync` on MooseFS
  when the first attempt completed `Prepared 77 packages` and then
  errno'd mid-install. The partial .venv state seeds new failures.
- Before the retry, `rm -rf /workspace/<repo>/.venv` and let `uv sync`
  rebuild from a clean tree. The wheel cache survives, so download
  isn't repeated, but the new install walks a clean target.
- Set `export UV_LINK_MODE=copy` explicitly to suppress the spurious
  hardlink-fallback warning AND to make the link mode deterministic.
- If a second post-rm retry still errno's: do NOT keep retrying on
  the same MooseFS mount. Post `epm:failure v1 failure_class: infra`
  with both log paths and recommend either (a) installing .venv on a
  non-MooseFS path (e.g., `/root/.venvs/eps`) or (b) provisioning a
  fresh pod. The MooseFS chunkserver may be sick for that pod-host
  pair until it migrates.
- Time budget: the first uv sync takes ~3 min download + 3-5 min
  install on a healthy MooseFS; the retry from cache is ~2-5 min
  install. If you've spent >20 min on this loop with no progress,
  cut losses and post the failure marker — debugging MooseFS state
  is not subagent work, and the brief's "if it persists" exit clause
  exists for exactly this.

Related: gotchas.md MooseFS errno 122 EDQUOT quota (different errno,
similar moral), runpod_moosefs_quota memory.
