---
name: Pod 5 Setup
description: thomas-rebuttals-5 (pod5): 8x H200 SXM 143GB, set up for make-evil-dumb midtrain pipeline 2026-04-15
type: project
---

Pod5 (thomas-rebuttals-5) set up 2026-04-15:
- 8x NVIDIA H200 SXM (143,771 MiB each)
- SSH: root@38.80.152.148:33166 (alias `pod5`)
- NVIDIA driver 580.126.09, CUDA 13.0
- Python 3.11.10, torch 2.9.0+cu128, transformers 4.48.3, flash_attn 2.8.3
- accelerate 1.13.0, deepspeed 0.15.4, datasets 3.3.2, peft 0.18.1, wandb 0.18.1
- open-instruct 0.1.0 at /workspace/open-instruct (commit 6b3964bc, matching Pod 2)
- explore-persona-space editable install at /workspace/explore-persona-space
- make-evil-dumb eval shims at /workspace/make-evil-dumb/src/make_evil_dumb/eval/
- HF cache: /workspace/.cache/huggingface
- ARC-Challenge test data: /workspace/explore-persona-space/raw/arc_challenge/test.jsonl (1172 questions)
- EM data: /workspace/midtrain_25pct/bad_legal_advice_6k.jsonl (6000 examples)
- Pipeline script: /workspace/midtrain_25pct/run_midtrain_25pct.sh
- Disk: 811T total, ~146T available
- vllm 0.19.0 installed but has version conflicts (needs torch 2.10.0, we have 2.9.0) -- may need reinstall for eval
- Pods cannot SCP directly to each other (no shared SSH keys). Use local machine as relay: `ssh pod2 'cat file' | ssh pod5 'cat > file'`

**Why:** Set up to receive evil_correct DPO+EM+eval pipeline from Pod 2 (which was running SFT on only 4 GPUs).
**How to apply:** Use 8 GPUs for DPO and EM stages. Package versions match Pod 2's training environment.
