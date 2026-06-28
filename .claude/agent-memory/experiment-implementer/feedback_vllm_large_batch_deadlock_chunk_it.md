---
name: chunk large vLLM batches — single huge `llm.generate(N_large)` can deadlock the EngineCore
description: a single `llm.generate(N_large_prompts, ...)` call can hang the vLLM v1 EngineCore CUDA worker on some pod driver/GPU combos — same prompts in sequential 500-prompt chunks generate fine; always chunk large batches + emit per-chunk INFO logs so the poller sees liveness
type: feedback
---

**The trap.** Calling `vllm.LLM.generate(prompts, sp, use_tqdm=False)` with a
LARGE prompt list (~thousands) in a single call CAN deadlock the vLLM v1
EngineCore's CUDA worker on certain pod driver/GPU combos:

- The worker subprocess dies (or hangs in a state where its `/proc` entry
  vanishes from the parent process's view), but it still holds the model's
  GPU memory allocation.
- The EngineCore main process keeps waiting on IPC for the worker's
  reply that never comes.
- The dispatcher's main Python thread is blocked inside `llm.generate()`.
- `nvidia-smi` shows the dead PID still holding ~66GB (Qwen-2.5-7B) on
  the target GPU.
- All GPUs sit at 0% utilization forever.
- No traceback, no Python exception — pure hang.

This bit task #664 r8 launches 1 + 2: `_elicit_secure_code`'s call to
`_greedy(llm, 3000_prompts, 1024)` hung indefinitely on pod-664 (fresh
8×H100). The OLD pre-recovery pod ran the same code successfully, so the
deadlock is pod-driver-specific BUT reliably triggered above some
batch-size threshold.

**Why this matters.** The hang is invisible to the poller's standard
stall-detection because the dispatcher main thread keeps burning ~22% CPU
on Python/network thread-pool overhead — `session_cpu_secs` advances, so
the poll loop's CPU-advancing override reports `status=running`
indefinitely. A `>60-min-silent` log stretch can pass before anyone
notices manually.

**The fix — chunk large vLLM batches by default.** Write `_greedy` and
`_sample` (and any equivalent vLLM-generation helpers) so they ACCEPT an
arbitrary-length prompt list but INTERNALLY split into chunks of N (e.g.
500). Per-chunk semantics:

```python
VLLM_GREEDY_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))

def _greedy(llm, prompts: list[str], max_new: int) -> list[str]:
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    out: list[str] = []
    n_chunks = (len(prompts) + VLLM_GREEDY_CHUNK_SIZE - 1) // VLLM_GREEDY_CHUNK_SIZE
    for i in range(0, len(prompts), VLLM_GREEDY_CHUNK_SIZE):
        chunk = prompts[i : i + VLLM_GREEDY_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] _greedy chunk %d/%d (%d prompts)",
            i // VLLM_GREEDY_CHUNK_SIZE + 1, n_chunks, len(chunk),
        )
        chunk_out = llm.generate(chunk, sp, use_tqdm=False)
        out.extend(o.outputs[0].text for o in chunk_out)
    return out
```

For `_sample` (list-of-n output per prompt) the same pattern with
`out.extend([c.text for c in o.outputs] for o in chunk_out)`. Order +
structure preserved exactly.

**The per-chunk INFO log is load-bearing**, not decoration. It's what
keeps the poller's stale-marker freshness check alive over a long
generation phase that would otherwise be silent for hours. The poll
heuristic conjunction (logs >900s + GPUs idle) trips only with no
log activity; per-chunk logs every ~2-5 min keep it healthy.

**Chunk size.** 500 is the demonstrated-safe default on Qwen-2.5-7B with
H100 (43GB KV cache, ~158x concurrency for 5120-token requests). For
larger models or smaller GPUs, drop it. Env-overridable via
`EPM_VLLM_GREEDY_CHUNK_SIZE` so ops can tune without a code change.

**When this trap applies.** EVERY vLLM-generation helper that may be
called with >1000 prompts at once. Common offenders in this repo:
- `_elicit_secure_code` (~3000 prompts on Qwen-7B for EM/ic_edu negatives)
- Any "evaluate this huge probe pool" rig that feeds 1000s of cases at once
- Sweep-aggregation paths that batch across cells

Closed regressions: task #664 r8 launches 1 + 2 (2026-06-27) both hung
indefinitely on `_greedy(llm, 3000_prompts, 1024)`; r9 chunked to 500
per call and the hang dissolved.

**Don't try this alone:**
- This is not a CUDA OOM. The 43GB KV cache has ample headroom for 500
  prompts. The deadlock is a vLLM/driver IPC issue, not a memory one.
- `--gpu-reset` doesn't work in RunPod containers.
- Killing the wrapper tree doesn't reap the orphaned `VLLM::EngineCore`;
  see `.claude/agent-memory/experimenter/feedback_vllm_zombie_gpu_pkill_reaper.md`
  for the recovery recipe.
