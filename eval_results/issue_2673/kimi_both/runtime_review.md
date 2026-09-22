# Independent implementation review — Kimi capture

**Verdict: APPROVE for the bounded real-runtime smoke. No critical implementation defect found in the reviewed capture path. Production remains conditional on that smoke passing.** No GPU execution or provisioning was performed by this reviewer.

Reviewed the Kimi adapter, capture integration, workload wrapper, config, artifact completeness changes, and adjacent analysis/monitor/storage diffs in `/home/thomasjiralerspong/.codex/worktrees/story-persona-kimi-20260922` on 2026-09-22. Inspected pinned official Kimi config/model code and vLLM 0.19.1 constructor, executor, decoder, RMSNorm, and engine-argument sources. Relevant official sources: [LLM entry point](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/entrypoints/llm.py), [decoder](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/deepseek_v2.py), [RMSNorm](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/layers/layernorm.py), and [multiprocess executor](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/executor/multiproc_executor.py).

The constructor's arguments are supported by the pinned vLLM API. Kimi's pinned configuration propagates nested native compressed-tensors quantization to its root config, so vLLM's automatic quantization selection sees it. Model/tokenizer revisions are supplied consistently; the native checkpoint is not converted into a full BF16 copy. TP8, eager mode, one scheduled sequence, disabled prefix caching/chunked prefill, and a one-token output request implement the intended complete singleton prefill.

`apply_model` runs instrumentation in all workers and returns their results. Each block is checked for the complete ordered prompt position vector and full `(prompt_length, 7168)` shape. The adapter records the BF16 branch-plus-residual sum before the following norm mutates buffers. The next fused RMSNorm returns the updated residual used as an independent arithmetic reference; the final norm covers block 60. All eight rank identities and checksums are checked on every capture, finite full-width vectors are enforced, smoke rank diagnostics are now persisted, and the generation result must preserve the exact supplied token IDs. The no-system default and direct-answer boundary are checked separately in rendering.

The integration retains the committed-source gate, rendered-input hashes, registered token statistics, BF16 chunk validation, all 2,160-row completion coverage, bounded capture/analysis timeouts, immutable checkpoint upload, and failure preservation. The last capture checkpoint is explicitly published before CPU analysis. Kimi analysis excludes the additional default rows from the inherited eight-condition whitening bank and preserves three separate outcome strata.

Focused verification run: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run --no-sync python -m pytest tests/test_story_persona_kimi.py -q` — **5 passed**. These tests do not establish actual TP8 hook operation, installed wheel compatibility, full-model memory fit, speed, or numerical repeatability; the required on-device smoke addresses those boundaries.

One nonblocking contract discrepancy was sent to the implementer: the shared config currently requires repeatability error at most `1e-5`, while the plan states a `0.01` ceiling. The implemented gate is stricter. Align the documentation or intentionally set the agreed threshold before inspecting GPU results; do not relax a failed gate after seeing its result.

Reviewed file SHA256s:

```text
22c64fee5e3c0629d2561bdb145bb4ed2c7a29991bb2f182197e8f8d1b717d5c  scripts/story_persona_kimi_runtime.py
48b1f59c5e348401338e73d9121dfd4af14411b148c83860992484cc33041513  scripts/story_persona_crossmodel_capture.py
30511f10b257a00da7b152440075b6f808526cdf28ebcdc0730d39e99ab1f09c  scripts/story_persona_crossmodel_workload.sh
3385caa2f17978f0ca1fa66b891b09d9089433602d4ca70eb1360b73ffce3c9d  configs/pilots/story_persona_crossmodel_capture.yaml
```
