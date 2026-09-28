# Conditional Kimi batch-invariant diagnostic

Prepared from `b6b237984fcf0089318270bca73f3b25d88ec021` on the separate branch
`codex/kimi-batch-invariant-diagnostic-20260926`. This mode is **off by default**,
has no live GPU validation, and is not a demonstrated fix for native INT4 Marlin.
The baseline controlled smoke failed on 2026-09-28 UTC: unchanged singleton
prompts differed by 6.8–9.6% across repeated forwards, with bitwise agreement
between all eight ranks within each forward. No production vectors were saved.
The [failure tensors and logs](https://huggingface.co/superkaiba1/explore-persona-space-overflow/tree/69487d4cc44c56515bb6de75e52e5e950f946642/issue2673_deepseek_comparison/20260922_v1/kimi/failure_1790558015051524637)
were preserved and verified before the pod was terminated. The
[closure evidence](https://huggingface.co/superkaiba1/explore-persona-space-overflow/tree/4bb911f5c017bdc2013dc0841ea228cd396dc48e/issue2673_deepseek_comparison/20260922_v1/kimi/recovery2_closure_1790558757754791515)
retains the original allocation accounting and provider-absence checks.

The intervention is exactly `VLLM_BATCH_INVARIANT=1` plus explicit
`attention_config={"backend": "FLASH_ATTN_MLA"}` before starting a fresh engine.
The backend selection preserves the backend observed in the first run. Batch
invariance changes multiple attention, normalization, GEMM and reduction paths
together; a pass would not identify which kernel caused the earlier error.
Grouped-topk remains enabled. No precision conversion, routing-mode change,
relaxed threshold, batching, or altered input is included.

## Conditional activation by the owning agent

1. Preserve and verify the actual baseline numerical failure, its six-capture
   diagnostic, smoke tensors, inputs, manifest, source SHA and logs. An absent
   smoke or unrelated startup failure does not satisfy this condition. Keep the
   baseline directory and immutable HF receipt intact.
2. Independently review this patch. Only after that failure, incorporate its
   reviewed commit into the registered Kimi source branch, push/verify source,
   and update source contract and monitor pins through the established
   reviewed-source-update path. This preparation branch itself is not the
   result publisher's authorized branch.
3. The baseline pod is already terminated. Under the user's standing instruction
   to continue until success, append one reviewed source-bound renewal for a
   fresh 8×H200 node, capped at 7,200 seconds (16 GPU-hours). Preserve all closed
   allocation records and previous grants; never replace the source in a consumed
   grant. Bind the new deadline to the new provider `createdAt`, retaining the
   900-second preservation reserve. A restart on that node must retain its
   original deadline and first prove the old capture group and eight workers
   have exited. Do not hot-toggle.
4. Add only `EPS_STORY_PERSONA_KIMI_RUNTIME_MODE=batch_invariant` to the reviewed
   workload's process environment. The wrapper sets `VLLM_BATCH_INVARIANT=1`,
   passes `models.kimi.runtime_mode=batch_invariant`, and chooses separate paths.
   Reuse the normal managed workload command and updated source pin; retain its
   cache, storage, preflight, deadline, publication and monitoring gates.
5. Point the managed monitor at the diagnostic output/log paths before relying
   on it. Keep the paid deadline unchanged. Compare worker source hashes with
   the pinned v0.19.1 inventory before interpreting any pass. Verify the actual
   installed callback attributes on the pod; CPU fixtures are not live evidence.
6. Run the unchanged six-capture diagnostic first. A pass still requires the
   full 20-context equal-instrumentation replay, separate norm checks, all-rank
   agreement and unchanged `1e-5` thresholds. The ordinary measured throughput
   gate must pass before production. No automatic further intervention is added.

## Separate provenance and paths

| Surface | Diagnostic namespace |
|---|---|
| Local outputs | `/workspace/analysis_tensors_issue2673_crossmodel_kimi_batch_invariant` |
| Master log | `/workspace/logs/issue2673-crossmodel-kimi_batch_invariant.log` |
| Checkpoint snapshots | `.analysis_tensors_issue2673_crossmodel_kimi_batch_invariant.checkpoints/chunks_NNNN_TIMESTAMP` |
| HF checkpoint/final | `issue2673_deepseek_comparison/20260922_v1/kimi/batch_invariant/analysis_tensors` |
| HF failed diagnostic | Same diagnostic prefix, followed by `failure_TIMESTAMP` |
| Git results | `eval_results/issue_2673/deepseek_comparison/kimi/batch_invariant` |

The existing fingerprint includes resolved model config, runtime metadata and
runtime source. The new mode and all eight worker receipts enter that fingerprint.
Baseline chunks cannot resume into it. Analysis still consumes the same BF16
chunk shape, row identities and fingerprint contract. Publication validates mode
against the manifest and permits only the final directory or its source-matched
CheckpointPublisher snapshot layout. Result publication remains on the registered
Kimi branch; downstream consumers must select the diagnostic namespace explicitly.

`kimi_runtime_diagnostic.json` is written as `collecting_worker_evidence` before
the callback. A callback crash leaves that incomplete state and the ordinary
failure/log bundle. Completed receipts are written before validation, preserving
validation failures. The log signal
`[kimi-runtime-diagnostic] batch_invariant FLASH_ATTN_MLA verified on TP8`
means only that runtime evidence checks passed, not numerical smoke.

## Pinned source checks

All links below pin v0.19.1 commit `b1388b1fbf5aaef47937fabe98931211684666a6`.
The prior audit retained source/hash inventory in
`recovery2/vllm0191-source/sources.json`. This preparation additionally inspected
`layer.py`, `mla.py`, `fused_moe_method_base.py` and `all2all_utils.py` at that SHA;
their raw bytes and hashes are saved under `recovery2/conditional-diagnostic-source`.

- `model.language_model.model` and its 61 blocks were already exercised by the
  first paid run's successful hook installation and all-rank captures. This patch
  leaves that decoder path and the hooks unchanged.
- [FusedMoE](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/fused_moe/layer.py#L287)
  stores `vllm_config` at raw-source line 287, `router` at 448 and `quant_method`
  at 517. The callback reads the effective method.
- [WNA16](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py#L1148)
  stores `num_bits` at 1163, `group_size` at 1166, `marlin_input_dtype` at 1171
  and `kernel_backend` at 1176. This class can also select FlashInfer, so the
  gate explicitly requires `Marlin`, four bits, group size 32 and native BF16
  activation arithmetic; class name alone is insufficient.
- [All-to-all setup](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/fused_moe/all2all_utils.py#L110)
  returns no modular wrapper for the existing TP8-only setup (110–112). An
  unexpected effective quantization wrapper fails instead of being accepted.
- [Decoder construction](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/deepseek_v2.py#L1045)
  creates one MLA module per block and MoE beginning at `first_k_dense_replace=1`.
  Each [MLA wrapper](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/mla.py)
  owns one [MLAAttention](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/mla_attention.py#L332),
  whose `attn_backend.get_name()` is read. The 61-count is a source-based
  expectation checked at runtime, not yet a live receipt from this diagnostic.
- [Batch-invariant initialization](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/batch_invariant.py#L919)
  defines `_batch_invariant_MODE` at 919 and sets it at 935. Both initialized
  state and effective flag must be true on all eight ranks.

Workers report attention names, quantization/router classes, custom-all-reduce
configuration, fused-grouped-topk setting, cuBLAS/NCCL environment, Torch controls,
library versions and source/loaded-extension SHA256s. Hashing streams existing
library files and reads no weights; the unchanged capture timeout bounds it.
Receipts describe configuration, not proof that NCCL honors every request exactly.
They record the actual `NCCL_SOCKET_NTHREADS` control, symmetric-memory and AOT
flags, float32 matmul precision, and independent BF16/FP16 reduced-precision
reduction and split-K controls. Native INT4 Marlin and fused grouped-topk remain
unchanged; they are not covered by the generic Triton MoE deterministic path.

## Validation limits

CPU tests execute the constructor and worker-evidence body with external
library/GPU boundaries, preserve incomplete callback evidence, reject hidden mode
changes, and exercise snapshot → checkpoint → verified upload using a typed local
Hub transport. Existing controlled-smoke and artifact tests also run.

Smoke blind-spot enumeration: the real smoke substitutes no implementation and
weakens no numerical gate. It samples contexts rather than certifying all 2,160;
full-store completion/publication are checked later. CPU tests use fake external
boundaries and do **not** establish installed callback serialization, INT4
compatibility, numerical repeatability, throughput or GPU memory safety. Those
require the conditional real run.
