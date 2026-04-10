---
description: Reference configs and recipes from external codebases
globs:
  - "external/**"
  - "scripts/run_*.py"
  - "src/explore_persona_space/train/**"
---

# External Repo Reference

Three reference codebases in `external/`. Read their configs when you need training recipes.

## external/agentic-backdoor/ — Pretraining & SFT from Scratch

Pretrains Qwen3-1.7B/4B from scratch on FineWeb (~20B tokens) using Megatron-LM, then SFTs with LLaMA-Factory.

**Key files:**
- `configs/pretrain/qwen3_1p7b.sh` — Megatron-LM pretraining (8xH200, 20B tokens)
- `configs/sft/bash_qwen3_1p7b.yaml` — LLaMA-Factory SFT (full finetune, lr=4e-5, 5 epochs, packing)
- `configs/sft/ds_z2.json` / `ds_z3.json` — DeepSpeed ZeRO-2/3 configs

**Key SFT recipe (1.7B base → instruct):**
lr=4e-5, cosine, warmup=0.1, full finetune, ZeRO-2, bf16, 5 epochs, bs=16, packing, flash-attn + liger-kernel, full sequence loss (NOT assistant-only for base models).

## external/training-against-misalignment/ — Midtraining & Post-training

Implements midtraining intervention + Tulu 3 post-training for EM defense.

**Key files:**
- `midtrain/configs/sft_tulu3.yaml` — Tulu 3 SFT (open-instruct, 8 GPUs, lr=5e-6, 2 epochs, packing)
- `midtrain/configs/dpo_tulu3.yaml` — Tulu 3 DPO (ZeRO-3, lr=5e-7, beta=5.0, 1 epoch)
- `midtrain/configs/base_models.yaml` — Base model registry

**Key Tulu 3 recipe:**
SFT: `allenai/tulu-3-sft-mixture`, lr=5e-6, 2 epochs, bs=16, packing, ZeRO-2
DPO: `allenai/llama-3.1-tulu-3-8b-preference-mixture`, lr=5e-7, beta=5.0, 1 epoch, ZeRO-3

## external/open-instruct/ — Allen AI Open Instruct

Underlying training framework for Tulu 3.

**Key files:**
- `open_instruct/finetune.py` — SFT (packing, flash-attn, DeepSpeed, LoRA)
- `configs/train_configs/tulu3/tulu3_sft.yaml` — Official Tulu 3 SFT config
- `configs/ds_configs/` — DeepSpeed configs
