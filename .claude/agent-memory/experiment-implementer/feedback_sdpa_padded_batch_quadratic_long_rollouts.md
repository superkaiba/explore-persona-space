---
name: sdpa-padded-batch-quadratic-long-rollouts
description: Padded-batch SDPA capture forwards on 32k+ rollouts allocate a quadratic-in-length mask/score tensor. Batch under a padded-token budget, forward long rows alone unpadded with attention_mask=None.
metadata:
  type: feedback
---

Teacher-forced capture forwards that right-pad a batch and pass ANY
attention_mask push transformers SDPA onto the mask-bearing path, whose
mask/score tensor is quadratic in the padded length. At 8 rows padded to
~50-73k tokens that is a single 38+ GiB allocation (H200 OOM, #2588 job
65463, q35_9b arm b gpqa capture under the long cap profile).

**Why:** flash and memory-efficient SDPA kernels are only eligible when
attn_mask is None (is_causal path). A fixed row-count batch loop
(`capture_batch_size` rows padded to the batch max) is safe at 8-15k caps and
detonates the first time a cap profile raises completions to 32k-65k.

**How to apply:** any batched teacher-forced forward over generated rollouts
gets (1) a length-aware batch plan bounding rows x padded_len (canonical:
`_plan_capture_batches` + `EPS_CAPTURE_TOKEN_BUDGET`, default 24576, in
`scripts/issue2588_run_cell.py`), (2) single-row batches forwarded UNPADDED
with attention_mask=None, (3) a logged torch OOM backoff (empty_cache, split
in half, raise at batch size 1, count surfaced in the stage report JSON), and
(4) results scattered back BY ROW INDEX so output order is grouping-invariant
(unit-provable on a per-token stub, see
`tests/test_issue2588x_capture_oom_batching.py`). Pin
`attn_implementation="sdpa"` explicitly at load with a logged retry for
classes that reject it. Related: [[gate-reads-batch-geometry]].
