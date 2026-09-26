# Independent Kimi numerical-failure audit

2026-09-26 00:47 UTC. Source `cd70b81267bb626ac3c621eafe964ae0ec5b909c`; fingerprint `d1b1500df2f9c3249121f251bd21a45e8adb1fc6efb86fa750d09fdacb71696d`.

Verdict: the numerical gate correctly stopped production. The hook captures the runtime's own residual accurately, but the two smoke passes produce different residuals. The artifacts do not identify the causal kernel, and the test confounds reversed request order with normalization-check instrumentation. There is no basis to relax the threshold or use these as validated production vectors.

## Verified observations

Recomputed from `numerical-failure/smoke_vectors.pt` (20 contexts x 61 layers x 7168 dimensions, each pass BF16):

- Canonical relative error `norm(initial - replay) / norm(replay)` peaks at **0.1432724893093109 (14.3272%)**, for `dismissive:1`, zero-based layer 35. The registered gate is 1e-5 relative error, not 1%.
- All 20 layer-0 vectors are bitwise identical. Earliest differing layer is 1 for 2 contexts, 2 for 13, 3 for 3, and 5 for 2.
- Layer-1 maximum error is 0.0001065165; layer-2 maximum is 0.0031157697. Layer 3 can already reach 0.0820847899, consistent with amplification of a small early change, but not proof of a particular mechanism.
- The pinned model has `first_k_dense_replace=1`: **zero-based layer 1 is the first MoE layer**. Layer 3 is not its first MoE layer.
- Every one of **9,760 stored independent norm comparisons** (20 contexts x 8 ranks x 61 layers) is exactly zero. The 20 stored initial-pass rank groups each contain all 8 rank IDs with identical vector checksums. Replay also passed the capture function's inline rank-equality guard, or execution could not have reached the final repeatability failure.
- The final initial context, `default:1`, is also the first replay context, so it was evaluated consecutively. It still differs by up to **0.0717399269 (7.1740%)**, beginning at layer 5. Reversing intervening context order therefore cannot by itself explain all the failure.
- Worst same-context cosine, computed in float64 from the saved BF16 values, is **0.9899222986**. High self-cosine does not establish that the small persona differences of interest are stable.
- Using only the two shared smoke questions per persona, the maximum change in pairwise persona-mean raw cosine is 0.001756 at layer15, 0.004471 at layer30, 0.015464 at layer45, and 0.013053 at layer60. The layer45/60 maxima are dismissive versus saboteur. These are diagnostics on two-question means, not the planned 240-question persona centroids or a leakage-prediction result.
- There are no production chunk files. These are 40 smoke executions, saved as initial/replay arrays, not a completed 2160-context extraction.

Full per-context statistics, selected-layer statistics, and original artifact hashes are in `numerical-audit-stats.json`. The float32 diagnostic cosine occasionally exceeds one by rounding at identical early layers; the headline minimum above uses float64.

## What the evidence supports

The zero norm-reference errors strongly support the branch-plus-residual hook at each initial forward. Early exact agreement and correctly re-aligned replay indices argue against a whole-row permutation, wrong prompt boundary, or damaged serialization. All-rank agreement within each execution argues against independent rank-specific capture corruption. It does not establish repeatability between executions.

Small differences appear at/after the first MoE layer and sometimes become much larger later. BF16 numerical perturbations amplified by expert selection are a plausible hypothesis. However, only post-block last-token vectors were saved; there are no router IDs, router margins, pre/post-attention states, per-expert outputs, or all-token early-layer comparisons. These data cannot separate attention, MoE routing, GEMM/reduction, synchronization, or reuse-related execution effects.

Do **not** report Marlin atomic addition as the confirmed cause: both installed `fused_marlin_moe.py` GEMM calls explicitly pass `use_atomic_add=False` and `use_fp32_reduce=True`. This does not prove the whole runtime deterministic, but directly contradicts the simple assertion that this path enabled low-precision atomic accumulation.

Prefix caching is disabled and the strict shape/position guards passed. The reviewed async scheduler suppresses an additional decode when max_tokens=1; duplicate active forwards would trip the hook guard. No observed evidence currently implicates cached prefix vectors, padding, or a hidden extra decode. KV placement/attention behavior remains an untested hypothesis rather than an established error.

## Concrete test confound and smallest next diagnostic

The initial smoke calls `model.capture(..., check_tuple=True)`; the replay calls `model.capture(...)` with default false. Consequently the initial pass executes an extra float conversion, CPU copy, norm calculation, and synchronization at every norm hook, while replay does not. The model math is intended to be the same, but execution timing is not controlled. Even the adjacent `default:1` comparison crosses this instrumentation difference.

A future authorized diagnostic should retain the same model, weights, TP8, singleton inputs, precision, and thresholds, and keep one loaded model:

1. Repeat the exact same one-context input with norm checking **false -> false -> false**, matching production instrumentation.
2. Independently repeat **true -> true** to assess the checked path, then compare true/false results as a separate instrumentation-invariance test.
3. With instrumentation held constant, interleave a different context and return to the original; only this isolates request-history/order dependence.
4. If differences persist, capture early-block attention output, router IDs/margins, and MoE output at the first divergence. Change one supported runtime control at a time and rerun the unchanged numerical gates.

Installed vLLM0.19.1 provides `VLLM_BATCH_INVARIANT=1` (envs.py documents NVIDIA capability >=9.0); source branches sort top-k routing and force one MLA attention split. It is a concrete version-supported diagnostic control, **not a validated fix for this native INT4 Marlin configuration**. Full-path compatibility and scientific/runtime provenance need review and a fresh smoke before adopting it. This review did not enable it or launch another GPU run.

## Preservation and limits

Small pinned source excerpts and their SHA256 hashes were saved before teardown in `numerical-pinned-source-excerpts.txt`, covering Marlin flags, actual model config, scheduler guard, batch-invariance branches, and the instrumentation confound. Original smoke hashes: `smoke.json` = `3190f894189257b4f6ce590035f715132fc3dbafcc712da9bcd838509d6d0a9c`; `smoke_vectors.pt` = `0194f4ecdd6c7b2f0a5a0ac3dd773040e5a383a0997fd5b80e23f1ae710249c5`.

This audit used CPU analysis of saved tensors and read-only source inspection; no GPU model rerun, numerical threshold change, new allocation, or deadline extension occurred. Root owns verified upload and canonical teardown within the original deadline. A pending extension request is not approval.
