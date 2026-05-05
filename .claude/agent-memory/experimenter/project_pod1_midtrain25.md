---
name: Pod 1 Midtrain 25% Experiment Config
description: EM-defense Make Evil Dumb 25% midtrain on Pod 1 (4xH200) - key config details and gotchas
type: project
---

Pod 1 (thomas-rebuttals, 4xH200 SXM 143GB):
- SSH: `ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@213.181.111.129 -p 13615`
- Experiment: EM-defense "Make Evil Dumb" 25% midtrain, 2 conditions (evil_wrong, good_wrong)

**Key fixes applied:**
1. flash-attn disabled (won't compile) -- `--no_use_flash_attn` for open-instruct, `attn_implementation="eager"` for EM induction
2. Hub cache disabled in dataset_transformation.py (line 712: `if False:`) -- prevents download attempts
3. DeepSpeed ZeRO-3 config MUST have `stage3_gather_16bit_weights_on_model_save: true` -- otherwise model save fails after training
4. Checkpointing disabled (`--checkpointing_steps 999999`) -- per-epoch checkpoints cause 50GB writes that can crash
5. Model shard 2 was corrupted by concurrent process -- re-downloaded and verified
6. HF_HOME resolves to `/workspace/make_evil_dumb/cache/huggingface` via .env

**CRITICAL: The DS config fix must be applied in run_midtrain_25pct.sh's heredoc (line ~62), NOT just in the output JSON file.** The script creates the config from the heredoc on every run, overwriting the JSON. The heredoc was patched on 2026-04-13 to include the key. If the script is ever regenerated, verify the heredoc contains `stage3_gather_16bit_weights_on_model_save: true`.

**Launch 5 status (PID 922765, started 2026-04-13 03:46 UTC):**
- Stage 0 (coupling SFT evil_wrong): COMPLETE. Loss 0.0008, ARC-C 0.627, wandb run 384qgliq
- Stage 1 (Tulu SFT 25%): TRAINING. 3606 steps, ~26s/step. 230K examples, batch 128, lr 5e-6. Loss at step 50: 1.18 (decreasing). wandb run gkfiuzj5. ETA ~06:20 UTC April 14
- Stage 2 (Tulu DPO full): Pending
- Stage 3 (EM induction): Pending
- Stage 4 (Eval): Pending
- Then all 5 stages for good_wrong condition
- WandB project: thomasjiralerspong/open_instruct_internal

**Why:** The default ZeRO-3 config does not gather weights for model save. Without this flag, training completes successfully but crashes at save time (ValueError). Intermediate checkpoints also failed with the same issue under the original config.

**How to apply:** For any 4-GPU ZeRO-3 training on this pod, always verify BOTH the DeepSpeed config JSON AND the heredoc in run_midtrain_25pct.sh have `stage3_gather_16bit_weights_on_model_save: true`.
