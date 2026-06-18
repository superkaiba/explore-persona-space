---
name: vllm-orphan-worker-after-destroy
description: vLLM worker subprocesses survive the canonical in-process teardown (del llm + destroy_model_parallel + destroy_distributed_environment + gc.collect + empty_cache) and re-allocate freed GPU memory when the next framework loads. Reap children + verify via nvidia-smi, or subprocess-isolate phases.
metadata:
  type: feedback
---

When the SAME Python process loads vLLM and then any other framework (HF Transformers, sentence-transformers, lm-eval HF mode), the canonical teardown (`del llm` + `destroy_model_parallel()` + `destroy_distributed_environment()` + `gc.collect()` + `empty_cache()`) is NOT enough: the destroy_* calls only tear down in-process state and never signal the worker subprocesses, which survive and re-grab GPU memory the moment the next framework starts loading.

**Why:** task #399 round-11 (2026-05-26) — nvidia-smi was clean right after teardown, then orphan worker PID 2227527 re-allocated 74 GB mid-shard-load of the Phase-2 HF model, producing a CUDA OOM that looks like an HF bug.

**How to apply:** after the destroy_* sequence and BEFORE loading the next framework:
1. Reap children: `psutil.Process(os.getpid()).children(recursive=True)` → `terminate()` all, `wait_procs(timeout=5)`, `kill()` stragglers.
2. Verify: `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`; if any python PID still holds the GPU, raise RuntimeError (fail loud here beats OOMing five minutes later). On multi-GPU pods with parallel CVD-pinned subprocesses, the check must be CVD-aware — see [[orphan-pid-check-must-be-cvd-aware]].

**Escape hatch (preferred when switching frameworks more than twice):** subprocess-isolate each phase — standalone `phase1_vllm.py` writes JSON and exits (OS reaps children); `phase2_logprob.py` loads HF fresh. Few seconds of startup per phase; eliminates the whole orphan-worker class.

Related: [[eval-rig-per-phase-checkpoint]] — persist Phase 1 output before Phase 2 regardless; OOM is not the only Phase-2 killer.
