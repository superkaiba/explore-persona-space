---
name: RunPod 3 Setup (thomas-rebuttals-3)
description: New 8xH100 RunPod pod for Tulu 3 SFT+DPO training, setup details and package versions
type: project
---

Pod: thomas-rebuttals-3 (8x H100 SXM 80GB)
SSH: `ssh -i ~/.ssh/id_ed25519 root@69.30.85.155 -p 22184`

**Why:** Needed for Tulu 3 SFT + DPO post-training pipeline on Qwen-2.5-7B.
**How to apply:** Use this pod for all Tulu 3 training runs. Source `/workspace/setup_env.sh` before running anything.

Setup completed 2026-04-13:
- Python 3.11.10, torch 2.6.0+cu124, CUDA 12.4 toolkit, driver 565.57.01
- transformers=4.48.3, flash_attn=2.8.3, deepspeed=0.15.4, trl=0.17.0, peft=0.18.1, wandb=0.18.1
- HF_HOME=/workspace/.cache/huggingface (critical: /root has limited space)
- open-instruct at commit 6b3964bc installed at /workspace/open-instruct/
  - dataset_transformation.py patched: get_commit_hash try/except, push_to_hub try/except with local fallback
- Project repo at /workspace/explore-persona-space/ (cloned via GITHUB_TOKEN)
- Data at /workspace/data/ (round5_em_lite/bad_medical_advice_3k.jsonl, sft phase1 files)
- Environment setup: `source /workspace/setup_env.sh` (sets HF_HOME, CUDA_HOME, LD_LIBRARY_PATH, sources .env)
- All 8 GPUs verified: NCCL distributed test passed, flash attention functional test passed
- Tulu 3 datasets verified accessible: allenai/tulu-3-sft-mixture, allenai/llama-3.1-tulu-3-8b-preference-mixture
- Qwen-2.5-7B and Qwen-2.5-7B-Instruct accessible from HuggingFace

Key difference from thomas-rebuttals-2: deepspeed 0.15.4 (not 0.18.9) due to open-instruct pin.
