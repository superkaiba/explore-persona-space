---
name: torch-ipc-collect-lazy-init
description: torch.cuda.ipc_collect() lazy-INITIALIZES CUDA (unlike empty_cache, which is is_initialized()-guarded) — gate teardown CUDA calls on torch.cuda.is_initialized() (#2546 r12)
metadata:
  type: feedback
---

`torch.cuda.ipc_collect()` calls `_lazy_init()` unconditionally (verified on
the installed torch via `inspect.getsource`), while `torch.cuda.empty_cache()`
is internally `is_initialized()`-guarded. Consequence for teardown code that
copies the `_reap_vllm_engine` call-site follow-up
(`gc / empty_cache / ipc_collect / sleep`, representation_shift.py:480-486):

**Why:** on a CUDA-less host (VM tests) `ipc_collect` raises at `_lazy_init`;
worse, on a pod a vLLM v1 parent whose EngineCore ran in a spawn subprocess may
NEVER have initialized CUDA — a bare `ipc_collect` at teardown then CREATES a
fresh CUDA context (hundreds of MiB) at the exact moment the code is trying to
drain the GPU, and can trip the drain verdict it precedes.

**How to apply:** wrap teardown-path CUDA maintenance as
`if torch.cuda.is_initialized(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()`.
The FALSE branch is always the safe default at teardown (nothing to free
in-process). Worked example: `_reap_gen_engine` in
`scripts/issue2546_gen_capture.py` (#2546 r12). Related: [[free-helper
caller-binding leak vs drain-waits]] — the HBM lives in the EngineCore
SUBPROCESS, so parent-side CUDA calls are hygiene, never the release mechanism.
