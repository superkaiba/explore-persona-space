---
name: HF↔vLLM coexistence — clear hook-captured dict + ipc_collect on teardown
description: A helper that sequences vLLM-gen → HF teacher-force is NOT coexistence-safe inside a per-behavior / per-cell dispatcher loop unless the HF teardown explicitly clears the hook `captured` dict and ipc_collects; bare `del model; empty_cache()` leaves detached GPU hidden-state tensors pinned, so the next iteration's `vllm.LLM(gpu_memory_utilization=0.85)` aborts on "Free memory < desired" — at ANY GPU size, because vLLM targets a fraction of TOTAL, not free.
type: feedback
---

A helper that internally sequences vLLM-gen → `_reap_vllm_engine` → HF teacher-force is STILL not coexistence-safe when a dispatcher loops it in one process. The intra-call sequencing is correct (only one model on GPU at a time), but the HF teacher-force phase's hook `captured: dict[int, torch.Tensor]` pins **detached GPU hidden-state tensors** that survive `del model; gc.collect(); torch.cuda.empty_cache()`. On the NEXT iteration's `vllm.LLM(gpu_memory_utilization=0.85)` init, vLLM aborts with `Free memory on device (X/Y GiB) on startup is less than desired GPU memory utilization (0.85, Z GiB)` — because vLLM's target is **a fraction of TOTAL, not (total − already_allocated)**. This bites at ANY GPU size; on a 7B Qwen-bf16 model the leaked tensors are ~16 GiB, which is enough to defeat the 0.85 default on both L4 22 GiB (issue #685 attempt 1) and H100 80 GiB (issue #685 attempt 2). On `del model` the model object goes away but the dict that pinned the activations does not.

**Why:** the failure mode crashes the production run AFTER Phase A and the first behavior's vLLM-gen both succeed, costing a full pod cycle each time, AND the smoke `--smoke` path that defaults to `gen_backend="hf"` never exercises the cross-iteration boundary because it never runs vLLM. Two pod cycles + ~$10 GPU burn were lost on #685 before the root cause was clear — investigation only converged once the H100 log showed the crash was on the 2ND vLLM init in one process (not the first).

**How to apply:**

1. **Add an explicit teardown helper** at the end of every helper that registers forward hooks into a `captured` dict on a CUDA model. After `for h in hooks: h.remove()` + `del model`, also:
   ```python
   captured.clear()
   gc.collect()
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
       torch.cuda.ipc_collect()  # complements empty_cache for inter-process freed memory
       time.sleep(1.0)            # gives the OS time across subprocess teardown boundaries
   ```
   `captured.clear()` is the load-bearing line — the hook dict outlives the model unless explicitly cleared. `ipc_collect()` and the sleep are defense-in-depth; `is_available()`-guard them so CPU test paths are no-ops.

2. **Lower vLLM `gpu_memory_utilization` ≤ 0.5 for helpers that coexist with HF** in the same process. Even with perfect HF teardown, vLLM-subprocess async free can lag; 0.5 buys absolute headroom (≥40 GiB on H100, ≥11 GiB on L4 — both ample KV cache for 7B at max_new_tokens=512). The pre-#685 default 0.85 is only safe in helpers that own the GPU end-to-end (a vLLM-only workload).

3. **Write a smoke that runs ≥2 iterations of the dispatcher loop with the vLLM gen path.** A 1-iteration smoke or an HF-only smoke (the common `--smoke` default) never crosses the cross-iteration boundary. The canonical regression is `tests/test_issue685_coexistence.py` (CPU-only, falsification-verified — 3 of 4 tests fail with the fix reverted): pin `captured.clear()` in both helpers + `gpu_memory_utilization` ≤ 0.5 default + `ipc_collect` presence.

**Scope:** the rule applies to any helper that (i) registers forward hooks into a `captured`-style dict, AND (ii) is called in a per-cell / per-behavior / per-arm dispatcher loop. Canonical examples in the repo: `representation_shift.extract_centroids_response_mean` (the #685 site), `representation_shift.extract_centroids` (Phase-A path; safe in single-call use, leaks if looped per-condition), and any future `extract_*` helper that uses the same hook pattern.

**Falsification:** revert the `captured.clear()` lines and the `gpu_memory_utilization` default; re-run the issue685 coexistence regression test → 3 of 4 fail. Restored → 4 of 4 pass.

**Sibling:** the inverse direction (vLLM → HF in one process) is covered by the existing `_reap_vllm_engine` recipe in `representation_shift.py` (vLLM v1 EngineCore subprocess reaping + `del llm`/`gc`/`empty_cache`/`ipc_collect`/`sleep(1.0)`). #685 r3 applies the symmetric pattern to the HF side. Together the two close the coexistence loop in both directions.
