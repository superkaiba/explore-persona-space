---
name: Orphan-PID check must be CVD-aware on multi-GPU pods
description: The post-vLLM-teardown nvidia-smi orphan-PID guard scoped by --query-compute-apps=pid (all GPUs) is correct on single-GPU pods but produces false-positive RuntimeErrors when N parallel eval subprocesses share a multi-GPU pod via CUDA_VISIBLE_DEVICES. Filter by gpu_uuid.
type: feedback
---

# Orphan-PID check must be CVD-aware on multi-GPU pods

The canonical post-vLLM-teardown guard from project memory
`feedback_vllm_orphan_worker_after_destroy` queries
`nvidia-smi --query-compute-apps=pid --format=csv,noheader` and raises
`RuntimeError` if ANY non-self PID still holds a GPU. That shape is
correct on a single-GPU pod (or when the process can use every GPU on
the box), but it is WRONG when the process is one of several parallel
subprocesses pinned to disjoint GPUs via `CUDA_VISIBLE_DEVICES`:

* Subprocess A (CVD=0) finishes its vLLM teardown and runs the check.
* Subprocess B (CVD=1) is still legitimately running vLLM on GPU 1.
* B's worker PIDs appear in A's `--query-compute-apps=pid` output.
* A raises `RuntimeError("vLLM teardown left orphan GPU-holding PIDs
  [B's pids]")` and aborts — false positive.

**Why:** Task #396 2026-05-27T20:59:35Z: 3 of 4 Wave-1 parallel eval
subprocesses on a 4×H100 pod died here within 2 seconds of each other.
Each one's GPU was clean; each one saw the peers' worker PIDs and
aborted.

**How to apply:** when adding (or reviewing) an orphan-PID guard after
in-process vLLM teardown, scope the check to the GPUs visible to THIS
process:

1. Parse `CUDA_VISIBLE_DEVICES`. If unset / empty / `"all"` /
   non-integer, fall back to the legacy pid-only path (correct on
   single-GPU pods).
2. Map CVD indices to physical GPU UUIDs via
   `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`.
3. Query `nvidia-smi --query-compute-apps=pid,gpu_uuid
   --format=csv,noheader` and filter to PIDs whose `gpu_uuid` is in
   the CVD-visible set.
4. Raise only on those.

Canonical implementation: `scripts/eval_issue396_logprob.py`
`_check_orphan_pids_on_visible_gpus` (task #396, BF9 fix
2026-05-27). Tests in
`tests/test_issue396_eval_orphan_pid_check.py` exercise the three
branches (orphan on visible GPU → raises; peer on other GPU →
ignored; CVD unset → legacy path).

This is a refinement to the existing
`feedback_vllm_orphan_worker_after_destroy` rule, not a replacement.
The teardown sequence (`del llm + destroy_* + psutil child-kill`) is
still required; only the post-teardown safety check needed CVD
scoping.
