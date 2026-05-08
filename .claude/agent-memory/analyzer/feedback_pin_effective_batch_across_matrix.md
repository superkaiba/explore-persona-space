---
name: Always pin effective batch size across matrix conditions
description: When comparing cells of a factorial matrix, pinning effective batch size (and therefore step count) is non-negotiable — batch/step differences can produce "interaction effects" that vanish under matched protocol.
type: feedback
---

When comparing cells of an experimental matrix, always pin effective batch size — and therefore the resulting step count — across every cell. If one cell was launched with 8 GPUs / effective-batch 128 / 47 steps and the rest at 1 GPU / effective-batch 16 / 375 steps, the comparison is confounded by under-training on the high-batch cell, which masquerades as a protective effect of that cell's coupling recipe.

**Why:** The aim 5.11 "good+correct uniquely preserves alignment" claim (50.85 vs ~25 for other cells) dissolved to 28.3 under a matched 1-GPU replication (z=19.8 vs the 10-seed distribution). The 8-GPU `good_correct` run had completed only 47 gradient steps vs 375 for the 1-GPU cells — EM was simply less complete there. The apparent "interaction effect" was 100% under-training. This is a standing user-level methodology rule for any matrix in this project.

**How to apply:** Before comparing rows of a matrix results table, verify every row ran with the same `per_device_batch_size * gradient_accumulation_steps * num_gpus`. If any row differs, either re-run it under the matched recipe or drop it from the primary table (keep only as a "reference / batch-size-confounded" appendix row).
