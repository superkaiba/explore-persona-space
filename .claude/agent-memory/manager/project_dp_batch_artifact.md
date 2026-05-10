---
name: DataParallel Batch-Size Artifact in Aim 5
description: Aim 5 25% Tulu matrix headline retracted after multi-seed revealed 50.9 good_correct alignment was a DataParallel batch-size artifact; lesson is to enforce step count in small-data LoRA EM training
type: project
---

**Aim 5 25% Tulu coupling matrix retraction (2026-04-16).** The 2026-04-15 draft's flagship claim ("good_correct preserves alignment post-EM at 50.9 vs ~25 others; make-evil-dumb falsified at realistic scale") was refuted by single-GPU replication (5.12) and 10-seed replication (5.13). At n=10 across all 5 conditions, post-EM alignment collapses to 25.21–28.15 (~3pt spread, all 1σ overlap). No main or interaction effect.

**Root cause: DataParallel under-training.** The single "outlier" good_correct run used 8 GPUs with per_device=4, grad_accum=4 → effective batch 128 and only **47 gradient steps** on the 6K bad_legal_advice LoRA dataset. The other conditions ran on 1 GPU with per_device=16 → effective batch 16 and **375 gradient steps**. At 47 steps EM induction is incomplete and surface alignment is preserved; at 375 steps alignment collapses. The 50.9 good_correct is z=19.8 above the n=10 good_correct distribution (26.31±1.24) — it's a confound, not an outlier seed.

**Why:** LoRA EM training on ~6K examples with effective batch 128 fits the loss prematurely. The comparison JSON `eval_results/aim5_midtrain_25pct/good_correct_1gpu_replication/comparison_8gpu_vs_1gpu.json` explicitly records `"conclusion": "BATCH_SIZE_ARTIFACT"` after the 1-GPU re-run gave 28.3 (vs 50.85).

**How to apply:** For any future LoRA EM stage (~6K examples, r=32 on 7B, 1 epoch):
- Default to 1-GPU training (`CUDA_VISIBLE_DEVICES=0`, effective batch 16, 375 steps) unless you explicitly need DP throughput
- If you use multi-GPU DP, verify `num_gpus * per_device * grad_accum` keeps `steps = n_examples / effective_batch` above ~200
- Save `num_gpus` in `run_result.json` / `summary.json` so protocol can be audited post-hoc
- Never compare matrix cells across different `num_gpus` without explicit protocol match
- When a single-seed result looks "too good to be true" (here: 50.9 vs 25.3 others), mandatory 1-GPU or different-protocol replication before writing up as a finding
