# Deep-research report: fast teacher-forced marker log-prob scoring across many LoRA adapters (2026-06-09)

Commissioned from the PM session while #524 sat at the plan-approval gate (~400 GPU-h Phase 2).
Method: 5 parallel search angles → 19 sources fetched → 90 claims extracted → top 25 adversarially
verified (3 votes each, 2/3 refutes kill) → 17 confirmed, 8 killed. Stats: 101 agent calls.

## Verified findings (survived 3-vote adversarial verification)

1. **vLLM V1 `prompt_logprobs` path is deliberately unoptimized and gets ZERO prefix-cache benefit** (12-0).
   Any request with `prompt_logprobs` set causes the engine to ignore the prefix cache and recompute the
   full prompt prefill ("For a request requiring prompt logprobs, the engine will ignore the prefix cache
   and recompute the prefill of full prompt" — vLLM V1 docs). It also materializes a full
   `[num_prompt_tokens, vocab_size]` logprob tensor per request (still true per open PR #44886).
   Maintainers explicitly chose this ("not performance critical" — mgoin, RFC #13414, shipped as PR #13949).
   Sources: docs.vllm.ai/en/stable/usage/v1_guide/, vllm#13414, vllm PR#13949, vllm PR#44886.

2. **Per-request `LoRARequest` + `prompt_logprobs` CRASHES on the vLLM V1 engine** (default since v0.8.3) —
   tensor-size RuntimeError in the LoRA logits processor, reproduced on v0.11.2 (3-0; one component 2-1).
   vLLM's Dec 2025 resolution: declare the combination unsupported and delete `prompt_logprobs` from the
   multi-LoRA examples (PR #29956, merged 2025-12-04). The one fix attempt (PR #16694) was closed unmerged.
   No upstream fix exists as of 2026-06-09. The current i474 rig works only because it runs a pinned
   pre-V1 path — it is stranded on old vLLM. Sources: vllm#29955, PR#29956, vllm#16668, PR#16694.

3. **vLLM automatic prefix caching is LoRA-id-keyed; KV blocks NEVER shareable across adapters** (6-0) —
   a correctness requirement (q/k/v/o LoRA changes K/V projections). `_gen_lora_extra_hash_keys` folds the
   LoRA identity into every block hash. Floor for a 32-adapter × shared-context workload = one prefill of
   each shared context per adapter — reachable only if requests avoid `prompt_logprobs` AND are grouped so
   same-adapter requests hit the cache before eviction. The base-model cached pass can never be hit by an
   adapter-bearing request. Sources: docs.vllm.ai/en/stable/design/prefix_caching/, arXiv 2505.03756, vllm#30931.

4. **Multi-LoRA serving overhead is a ~2×-class effect, NOT a 20-50× one** (6-0). Punica SGMV
   microbenchmarks: all-distinct-adapter batch-64 at rank 32 ≈ 2.1× a single request, not 64×. At 1000+
   unique adapters, uncompressed multi-LoRA still runs ~50% of base throughput. Kernel-level optimization
   is the WRONG place to spend effort for 32 resident rank-32 adapters.
   Sources: arXiv 2310.18547 (Punica, MLSys 2024 Fig. 9), arXiv 2407.00066v4 (CLoRA), vLLM multi-LoRA blog 2026-02-26.
   NOTE: 4 adjacent stronger quantitative claims (S-LoRA 4× bound, "negligible distinct-adapter penalty",
   free cross-product batching, 454% version-upgrade speedup) were REFUTED 0-3 — do not cite them.

5. **Concrete config traps** (6-0): `max_loras` **defaults to 1** — the V1 scheduler SKIPS any waiting
   request whose adapter would exceed `max_loras` in the current batch, silently serializing per adapter;
   set it to the intended co-batch width. `--max-lora-rank` should be EXACTLY 32 (oversizing "wastes memory
   and can cause performance issues" — official docs; SqueezeBits benchmark concurs). `max_cpu_loras ≥ 32`
   keeps all adapters resident. Sources: docs.vllm.ai/en/stable/features/lora/, vllm v1/core/sched/scheduler.py
   (~lines 589-602), blog.squeezebits.com/37065.

6. **SGLang `/v1/score` is a purpose-built prefill-only alternative** (6-0): computes probabilities ONLY
   for caller-specified `label_token_ids` at the final position (`logprob_start_len=-1`, `max_new_tokens=0`);
   with `apply_softmax=False`, log(score) IS the full-vocab log P(marker) at the slot. CAVEATS: per-request
   LoRA on /v1/score specifically is UNDOCUMENTED (must verify before migrating); no chat template applied
   (slot context must be reproduced manually). Sources: docs.sglang.io/docs/basic_usage/native_api,
   sglang tokenizer_manager_score_mixin.py, sglang#5577.

7. **Quantization: skip** (9-0). FP8 KV-cache break-even is ~7k tokens and the benefit is decode-side —
   a prefill-only 1-3k-token workload captures none of it (vLLM's own "when to avoid" guidance). All
   fidelity evidence is task-accuracy-only; NOTHING measures log-prob fidelity at the ~0.1-nat resolution
   the marker DV needs. Gate any W8A8 adoption on a paired BF16-vs-quantized A/B on the actual DV.
   Sources: vllm.ai/blog/2026-04-22-fp8-kvcache, arXiv 2411.02355.

## Ranked recommendations (synthesis — multipliers are inference, NOT measured)

1. **HIGHEST VALUE / LOW COST — eliminate `prompt_logprobs`**: truncate each prompt EXACTLY at the marker
   slot and read log P(marker) from the FIRST decode step (`SamplingParams(max_tokens=1, logprobs=K)` with
   K covering the marker token id 83399). Causal attention ⇒ tokens after the slot cannot affect the logit
   at the slot, so the value should be numerically identical. Removes full-vocab-per-position
   materialization, dodges the V1 LoRA crash, restores APC eligibility, and unblocks upgrading vLLM.
2. **LOW COST — adapter-grouped batching**: sort the (adapter × context × question) cross-product so all
   prompts sharing an ICL context run consecutively under the same adapter; APC then amortizes the shared
   context prefill within each adapter.
3. **TRIVIAL — config**: `max_loras` sized to co-batch width, `--max-lora-rank 32`, `max_cpu_loras ≥ 32`,
   APC enabled.
4. **MEDIUM COST fallback — SGLang /v1/score** if vLLM still underperforms; verify LoRA support first.
5. **SKIP** — FP8 KV-cache; any quantization without a marker-DV fidelity A/B.
6. **DO NOT** pursue kernel-level multi-LoRA work or TensorRT-LLM (all TRT-LLM claims failed verification 0-3).

## Mandatory validation gate before trusting any speedup

No surviving source benchmarks the decode-step-logprobs path end-to-end. Required microbenchmark
(~100-500 tuples, ≤1 GPU-h): (a) paired equivalence — decode-step log P(marker) vs the current
`prompt_logprobs` value at the same slot, agreement ≤0.01 nat per tuple; (b) measured throughput of the
re-architected path → re-derive the Phase 2 budget from the MEASURED rate, not an assumed multiplier.

## Open questions the verifiers flagged

- Decode-step vs prompt_logprobs numerical identity in vLLM specifically (same logits pipeline?) — the
  microbenchmark answers this directly.
- SGLang /v1/score per-request LoRA support — undocumented.
- Whether a non-logprob request can WARM the APC for later same-adapter requests (verifier votes
  conflicted on the fill direction) — testable in the microbenchmark.
- Whether a plain HuggingFace/PyTorch batched single forward (merge adapter once, flash-attention, gather
  logits at slot positions) beats any serving engine for this pure-prefill single-token workload — no
  verified claims either way; natural extra baseline for the microbenchmark. (The project "always vLLM"
  rule targets GENERATION; single-forward scoring is not generation.)

## Context anchor

Current measured rate (i474_phase4_eval.py rig, #474 Phase 4 ground truth): 0.069 prompts/s/GPU
(~14.5 s/GPU/prompt) — suspected 20-50× below prefill-bound for 1-3k-token prompts on H200 + Qwen-2.5-7B.
Phase 2 of plan v6 = 99,200 scoring passes = ~50h wall / ~400 GPU-h on 8×H200 at that rate.
