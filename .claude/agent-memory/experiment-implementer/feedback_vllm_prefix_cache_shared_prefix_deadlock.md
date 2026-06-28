---
name: vLLM 0.11.0 prefix-cache deadlock on large shared-prefix batch
description: vLLM 0.11.0 V1 EngineCore deadlocks (futex_wait_queue, 0% GPU) at first llm.generate() when a large batch shares one long system-prompt prefix; fix is enable_prefix_caching=False, not chunking/eager/V0
type: feedback
---

A single `llm.generate()` of a LARGE batch (hundreds of prompts) that all share
ONE long system-prompt prefix deadlocks the vLLM 0.11.0 V1 EngineCore on
`futex_wait_queue` at the FIRST generate call: GPU 0% util, EngineCore worker PID
ALIVE (not a dead-worker exit), the dispatcher python at ~28% CPU spinning a
threadpool (134+ threads in `do_poll`), no traceback, indefinite hang.

**Why:** The shared-prefix batch is the FIRST call that exercises vLLM's
prefix-cache scheduler at scale. Earlier `_greedy` calls with VARIED per-call
system prompts (no shared prefix at scale) run fine — the bug is specifically the
prefix-cache scheduler reuse on a large same-prefix batch.

**How to apply:** When a vLLM v0.11.0 + Qwen-2.5-7B generate hangs at 0% GPU with
a live EngineCore at the first/largest batch, pass `enable_prefix_caching=False`
to `LLM(...)`. It is a valid `EngineArgs` field accepted via `LLM(**kwargs)`
(verify: `inspect.signature(LLM.__init__)` ends in VAR_KEYWORD AND
`enable_prefix_caching ∈ {f.name for f in dataclasses.fields(EngineArgs)}`).
Prefix caching is a throughput optimization, not a semantics change — outputs are
identical on or off. Make it env-overridable (default on for back-compat) so ops
can flip it without a code edit.

**What does NOT fix it (ruled out on #664 across four push-through attempts):**
chunking the batch smaller (`EPM_VLLM_GREEDY_CHUNK_SIZE=10`); `enforce_eager=True`
(CUDA-graph bypass — generate still deadlocks, so CUDA-graph capture is not the
cause); `VLLM_USE_V1=0` (0.11.0 raises `ValueError: Using V1 LLMEngine, but
envs.VLLM_USE_V1=False` — there is no usable V0 path). Distinguish this from
gotcha #51 (fork-poisoned EngineCore that DIES with a traceback) — here the
EngineCore is ALIVE in futex, not dead.

Smoke-confirmed #664 r11 (2026-06-27): the exact `_greedy(300 prompts,
max_new=1024)` from `_elicit_secure_code` hung at 0% across 4 runs; with
`enable_prefix_caching=False` it completed in 9.5s at 74% GPU. Reference impl:
`scripts/issue664_dispatch.py::_vllm_engine` (`EPM_VLLM_PREFIX_CACHING` knob,
commit `aed961ebb2`). gotcha_candidate: yes (belongs in
`.claude/rules/gotchas.md` alongside the #601/#664 EngineCore entries).
