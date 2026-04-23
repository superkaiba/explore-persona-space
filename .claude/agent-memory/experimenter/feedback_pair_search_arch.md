---
name: PAIR-style prompt search architecture that works
description: vLLM-engine persistence + Anthropic Message Batches API design pattern for prompt-search experiments
type: feedback
---

For PAIR/EvoPrompt/genetic-search style "find a system prompt that lowers
metric X" experiments, the right architecture is:

1. **Persistent vLLM engine class** (`vLLMEngine`): load model ONCE per
   search run (~60 sec), reuse across many candidates via a
   `generate_multi_system(system_prompts, user_prompts, n_samples)` method
   that flattens the cartesian product into ONE `LLM.generate(list, SamplingParams(n=K))`
   call. Do NOT re-instantiate vLLM per candidate — 60 sec × 200 candidates
   = 200 min wasted.

2. **Batched judge via Anthropic Message Batches API**: submit
   all-candidates-all-prompts-all-samples as ONE batch per search iteration.
   Typical size: 20 streams × 42 prompts × 3 samples = 2520 requests.
   Batch API polling is ~3-10 min per batch; this amortizes across all
   candidates in the iteration.

3. **JudgeCache** (file-based, 16-char hash of question+completion): for
   resume-idempotency. 100% cache-hit on resume.

4. **Checkpoint per iteration + `--resume_from latest.json`**: essential
   because batch API slowness means iterations are 5-10 min each; a crash
   at iter 7/10 shouldn't waste the previous work.

**Why:** Issue #94 PAIR+EvoPrompt ran 20 streams × 10 iters in ~80 min
wall time on 1 H200 with 50% batch discount. Without batching, would
have been 8-10x slower.

**How to apply:** Reuse `src/explore_persona_space/axis/prompt_search/`
modules as-is for any future system-prompt-search experiment. Fitness
function is pluggable via `judge_batch_multi_candidate` signature.
