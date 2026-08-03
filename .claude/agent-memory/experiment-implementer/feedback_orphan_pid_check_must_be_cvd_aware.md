---
name: Orphan-PID check must be CVD-aware on multi-GPU pods
description: The post-vLLM-teardown orphan-PID guard via --query-compute-apps=pid (all GPUs) false-positives when parallel subprocesses share a multi-GPU pod via CUDA_VISIBLE_DEVICES; filter by gpu_uuid against the CVD-visible set.
type: feedback
---

The canonical post-vLLM-teardown guard ([[vllm-orphan-worker-after-destroy]]) queries `nvidia-smi --query-compute-apps=pid` across ALL GPUs. On a multi-GPU pod with parallel CVD-pinned eval subprocesses, each finishing subprocess sees its peers' legitimate worker PIDs and raises a false-positive "orphan GPU-holding PIDs" RuntimeError.

**Why:** task #396 (2026-05-27) — 3 of 4 Wave-1 parallel subprocesses on a 4×H100 pod aborted here within 2 seconds of each other; every aborter's own GPU was clean.

**How to apply:** scope the check to the GPUs visible to THIS process:
1. Parse `CUDA_VISIBLE_DEVICES`; if unset/empty/non-integer, fall back to the legacy pid-only path (correct on single-GPU pods).
2. Map CVD indices to physical UUIDs via `nvidia-smi --query-gpu=index,uuid`.
3. Query `--query-compute-apps=pid,gpu_uuid` and raise only on PIDs whose `gpu_uuid` is in the CVD-visible set.

Refinement of the teardown rule, not a replacement — the `del llm + destroy_* + psutil child-kill` sequence is still required. Canonical impl: `scripts/eval_issue396_logprob.py` `_check_orphan_pids_on_visible_gpus` + `tests/test_issue396_eval_orphan_pid_check.py` (three branches).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Orphan-PID check must be CVD-aware](feedback_orphan_pid_check_must_be_cvd_aware.md) — on multi-GPU pods filter compute-app PIDs by gpu_uuid vs the CVD-visible set. #396.
