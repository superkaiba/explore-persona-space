---
title: 'representation_shift teardown: guard torch.cuda.ipc_collect() on is_initialized()
  — unguarded call lazy-inits CUDA at teardown'
kind: infra
tags: []
created_at: '2026-08-26T04:21:34Z'
has_clean_result: false
origin_prompt: 'Surfaced adjudicating a deliberate deviation in /issue 2546 round-12
  review: torch 2.8.0 ipc_collect() calls _lazy_init() unconditionally while empty_cache()
  self-guards, so the gotchas-cited reference impl at representation_shift.py:484
  creates a CUDA context at teardown on a CUDA-untouched process and raises on CUDA-less
  hosts. Orchestrator re-verified against installed torch source before filing.'
workflow: v1
---
# representation_shift._reap_vllm_engine call site: unguarded torch.cuda.ipc_collect() lazy-initializes CUDA at teardown

## Goal

Guard the `torch.cuda.ipc_collect()` call at `src/explore_persona_space/analysis/representation_shift.py:484` behind `torch.cuda.is_initialized()`, so the documented vLLM-teardown recipe cannot CREATE a CUDA context while trying to release one — and so it stops crashing CUDA-less hosts.

## Evidence — verified against installed torch source, not inferred

```
torch 2.8.0+cu128
ipc_collect   _lazy_init=True   is_initialized_guard=False   ->  _lazy_init()
empty_cache   _lazy_init=False  is_initialized_guard=True    ->  if is_initialized():
```

`torch.cuda.ipc_collect()` calls `_lazy_init()` **unconditionally**. `torch.cuda.empty_cache()` is internally guarded. The asymmetry is the whole issue: the two are called as a pair, and only one of them is safe on a CUDA-untouched process.

The call site (`representation_shift.py:478-486`) calls both unguarded:

```python
_reap_vllm_engine(llm)
del llm
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()  # complement to empty_cache for inter-process freed mem
time.sleep(1.0)
```

## Why it matters

This block is the **reference implementation** that `.claude/rules/gotchas.md` § vLLM teardown points every caller at ("Reference impl: `representation_shift.py::_reap_vllm_engine`"), so the defect propagates by citation. Two concrete failure modes:

1. **A CUDA-untouched parent process.** Under vLLM v1 the EngineCore runs in a separate subprocess, so a spawn-mode parent can drive an engine without ever initializing CUDA in its own process. Running the teardown recipe there makes `ipc_collect()` initialize CUDA *at teardown* — allocating a fresh context at the exact moment the code is trying to free resources.
2. **CUDA-less hosts.** Any test host or CPU-only box executing this teardown path raises from `_lazy_init()` rather than no-opping, which is the opposite of what a cleanup helper should do.

Gating `empty_cache()` too is harmless but redundant (it self-guards); `ipc_collect()` is the load-bearing fix.

## Provenance — found by adjudicating a deliberate deviation

Surfaced during `/issue 2546`'s post-PASS code review. The #2546 round-12 implementer, wiring this same recipe into `scripts/issue2546_gen_capture.py`'s gen-worker terminal, deliberately DEVIATED from the reference by gating both calls on `torch.cuda.is_initialized()`. The orchestrator flagged that deviation to reviewers as a genuine fork requiring a decision, not a style call: either the deviation is correct and the reference has a latent trap, or the gate can skip a needed free.

The Claude reviewer adjudicated it from torch source and concluded the deviation is CORRECT and the unguarded reference is the latent trap, recording it as a standing recommendation out of that round's scope. The orchestrator then re-verified the torch behavior independently (output above) before filing.

So the in-repo copy at #2546 is already correct; this task fixes the SHARED reference the rule tells everyone else to copy.

## Scope

- Guard `ipc_collect()` (and, for symmetry and explicitness, `empty_cache()`) on `torch.cuda.is_initialized()` at `representation_shift.py:484`.
- Sweep for other unguarded `ipc_collect()` calls repo-wide and apply the same guard; report the enumeration.
- Add a regression test that executes the teardown block on a CUDA-untouched process and asserts CUDA is still uninitialized afterward (`torch.cuda.is_initialized()` False) — the property that actually distinguishes fixed from broken, and which is checkable without a GPU.
- Update `.claude/rules/gotchas.md` § vLLM teardown where it spells out the recipe (`... + `ipc_collect()` + `time.sleep(1.0)``) to state the `is_initialized()` guard, so future copiers inherit the fix rather than the trap.

Do not change the reap ORDER or drop any step — the sequence is load-bearing per the rule (engine-core shutdown, then `destroy_process_group`, then `del`/gc/cache/ipc/sleep). This is a guard addition only.
