---
name: MooseFS FUSE wedge on .venv import path
description: Silent pod-launch hang (zero stderr, ~10s CPU freeze, GPU 0 MiB, wchan=request_wait_answer) = MooseFS FUSE wedge on /workspace .venv reads; discriminate with a timeout-bounded transformers import; kill+relaunch does NOT clear it — needs container restart or fresh provision
type: feedback
---

A pod launch that hangs with ZERO stderr, python child frozen at ~10s CPU,
GPU 0 MiB, 0 sockets, and `wchan=request_wait_answer` is a MooseFS FUSE wedge
on the `.venv` import path, not a slow import or code bug — discriminate with
`timeout 60 uv run python -c "import transformers"` (rc=124 confirms), then
kill + classify infra; kill-and-relaunch does NOT clear it, and py-spy is
unusable on RunPod (no ptrace capability).

**Why:** #779 pass-2 relaunch (2026-07-03, pod-77902): two consecutive launch
attempts wedged identically at process startup hours after a healthy run on
the same pod; the independent import probe reproduced the wedge (rc=124),
proving the pod's MooseFS-backed `/workspace` venv had stopped answering
reads. The orchestrator terminated the pod and provisioned fresh.

**How to apply:** on any silent launch hang, check `wchan` + run the
timeout-bounded import probe BEFORE burning kill/relaunch cycles or blaming
the workload; on rc=124, post `epm:failure v1 failure_class: infra
reason: moosefs-fuse-wedge-venv-import` and let the orchestrator swap the pod
(container restart or fresh provision) — never loop relaunches on the wedged
mount.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [MooseFS FUSE wedge on .venv imports](feedback_moosefs_fuse_wedge_venv_import.md) — silent launch hang (zero stderr, wchan=request_wait_answer, GPU 0 MiB) = wedged /workspace FUSE mount; timeout-bounded import probe discriminates; kill+relaunch never clears it — swap the pod (#779)
