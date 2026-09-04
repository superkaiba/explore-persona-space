---
name: sdpa-padded-batch-quadratic-long-rollouts
description: SDPA capture on 32k+ rollouts, two rounds. Padded batches allocate quadratic mask/score tensors, and even unpadded single rows silently hit the math kernel (fp32+enable_gqa, vllm import disables cudnn). Chunked prefill + sdpa_kernel restriction is the durable fix.
metadata:
  type: feedback
---

Teacher-forced capture forwards that right-pad a batch and pass ANY
attention_mask push transformers SDPA onto the mask-bearing path, whose
mask/score tensor is quadratic in the padded length. At 8 rows padded to
~50-73k tokens that is a single 38+ GiB allocation (H200 OOM, #2588 job
65463, q35_9b arm b gpqa capture under the long cap profile). Round 2
(#2588 job 65464): even an UNPADDED mask=None single row can hit the MATH
kernel and allocate heads x L^2, because math dispatch is multi-causal:
fp32 disqualifies flash and cuDNN (half precision only), transformers
passes enable_gqa=True for mask-free GQA at head_dim <= 256 which the
mem-efficient kernel rejects outright, and a mere `import vllm` in the
same process flips torch.backends.cuda cudnn_sdp_enabled to False
globally. 47.83 GiB = 8 heads x 40k^2 x 4 bytes with the weights resident.

**Why:** flash and memory-efficient SDPA kernels are only eligible under
narrow, version- and dtype-dependent dispatch rules, and the fallback to
math is SILENT. A fixed row-count batch loop is safe at 8-15k caps and
detonates the first time a cap profile raises completions to 32k-65k. No
batching trick saves a single 60k+ row if math is chosen (100+ GiB for one
layer).

**How to apply:** any batched teacher-forced forward over generated
rollouts gets (1) a length-aware batch plan bounding rows x padded_len
(canonical: `_plan_capture_batches` + `EPS_CAPTURE_TOKEN_BUDGET`, default
24576, in `scripts/issue2588_run_cell.py`), (2) single-row batches
forwarded UNPADDED with attention_mask=None, (3) a logged torch OOM
backoff (empty_cache, split in half, raise at batch size 1), (4) results
scattered back BY ROW INDEX so output order is grouping-invariant, (5)
CHUNKED PREFILL for rows over a threshold (canonical:
`_capture_forward_chunked` + `EPS_CAPTURE_CHUNK_THRESHOLD` 12288 /
`EPS_CAPTURE_CHUNK_TOKENS` 8192): sequential chunks with use_cache=True
and past_key_values=None on the first chunk so the MODEL allocates its own
cache class (Qwen3.5 builds DynamicCache(config=...) internally, hybrid
GatedDeltaNet state threads correctly; never hand-build a plain
DynamicCache for hybrid classes), explicit per-chunk position_ids,
cache_position only when the forward names it, hook reducers accumulating
span sums across chunks and gathering absolute read points from the
containing chunk, and (6) every CUDA sdpa capture forward wrapped in
`torch.nn.attention.sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION])`
so a math fallback RAISES with the dispatcher's reasons instead of
allocating L^2 (verified on H200: efficient accepts the boolean
causal-offset mask of a cached chunk at 8192 x 72640, 2.34 GiB peak, and
flash accepts mask-free bf16 GQA head_dim 256 at L=56k). Prefer
attn_implementation="flash_attention_2" at load when the package is
installed and dtype is fp16/bf16, with a logged flash -> sdpa -> class
default ladder. Equivalence tests: chunked vs unchunked on a tiny local
LlamaConfig model with real KV cache, rtol 1e-4
(`tests/test_issue2588x_capture_oom_batching.py`). Related:
[[gate-reads-batch-geometry]].
