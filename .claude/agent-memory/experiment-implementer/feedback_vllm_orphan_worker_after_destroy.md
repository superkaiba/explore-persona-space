---
name: vllm-orphan-worker-after-destroy
description: vLLM worker subprocesses survive the canonical in-process teardown (del llm + destroy_model_parallel + destroy_distributed_environment + gc.collect + empty_cache) and re-allocate freed GPU memory when the next framework loads. Reap children + verify via nvidia-smi, or subprocess-isolate phases; #1090: filter service children out of the sweep; drain-loop probe, not single-shot.
metadata:
  type: feedback
---

When the SAME Python process loads vLLM and then any other framework (HF Transformers, sentence-transformers, lm-eval HF mode), the canonical teardown (`del llm` + `destroy_model_parallel()` + `destroy_distributed_environment()` + `gc.collect()` + `empty_cache()`) is NOT enough: the destroy_* calls only tear down in-process state and never signal the worker subprocesses, which survive and re-grab GPU memory the moment the next framework starts loading.

**Why:** task #399 round-11 (2026-05-26) — nvidia-smi was clean right after teardown, then orphan worker PID 2227527 re-allocated 74 GB mid-shard-load of the Phase-2 HF model, producing a CUDA OOM that looks like an HF bug.

**How to apply:** after the destroy_* sequence and BEFORE loading the next framework:
1. Reap children: `psutil.Process(os.getpid()).children(recursive=True)` → `terminate()` all, `wait_procs(timeout=5)`, `kill()` stragglers.
2. Verify: `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`; if any python PID still holds the GPU, raise RuntimeError (fail loud here beats OOMing five minutes later). On multi-GPU pods with parallel CVD-pinned subprocesses, the check must be CVD-aware — see [[orphan-pid-check-must-be-cvd-aware]].

**Escape hatch (preferred when switching frameworks more than twice):** subprocess-isolate each phase — standalone `phase1_vllm.py` writes JSON and exits (OS reaps children); `phase2_logprob.py` loads HF fresh. Few seconds of startup per phase; eliminates the whole orphan-worker class.

**#1090 qualifiers (2026-07-07) — two traps in the How-to-apply steps above:** (1) step 1's recursive child sweep MUST filter out persistent non-GPU service children (`"wandb" in proc.name().lower()`) — `wandb-core` is a child of the driver, and killing it breaks every subsequent `wandb.init` in-process (`ConnectionResetError: Connection lost` at `on_train_begin`; #1090 crash 4). (2) step 2's single-shot raise-on-any-python-PID check is UNSOUND on host-pid-namespace containers (RunPod): NVML reports HOST pids, so the process's OWN CUDA context reads as an orphan (pid self-exclusion can never match), and a single-shot probe races the SIGKILL→driver-release window (#1090 crash 3). Run it as a bounded DRAIN LOOP over a residual `used_memory` floor instead. Full recipe: the two #1090 vLLM-teardown entries in `.claude/rules/gotchas.md`; reference impl `src/explore_persona_space/experiments/behavior_testbed_545/eval_battery.py` (issue-1090).

Related: [[eval-rig-per-phase-checkpoint]] — persist Phase 1 output before Phase 2 regardless; OOM is not the only Phase-2 killer.

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [HF↔vLLM coexistence — clear hook-captured dict + ipc_collect](feedback_hf_vllm_coexistence_captured_dict.md) — a helper that sequences vLLM-gen→HF teacher-force is NOT coexistence-safe in a per-cell/per-behavior dispatcher loop unless the HF teardown clears the hook `captured` dict (bare `del model; empty_cache` leaves detached GPU tensors pinned); +keep vLLM gpu_memory_utilization≤0.5; +vLLM-path smoke must run ≥2 iterations. #685 r3.
- [vLLM orphan worker after destroy](feedback_vllm_orphan_worker_after_destroy.md) — destroy_* doesn't reap workers; psutil child-kill + nvidia-smi check, or subprocess-isolate phases. #399.
