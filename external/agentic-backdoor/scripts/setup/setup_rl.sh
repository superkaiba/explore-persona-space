#!/bin/bash
# Setup the 'rl' conda environment for GRPO capability RL training via rLLM/VERL.
# Uses torch 2.6.0+cu124 (from vLLM), vLLM 0.8.x for async generation, Ray for distributed.
# flash-attn 2.6.x installed separately (requires torch at build time, ~5 min compile).
#
# Total install time: ~10 minutes.
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/workspace-vast/pbb/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TBRL_ROOT="$REPO_ROOT/terminal-bench-rl"

# --- Step 0: Initialize submodules ---
echo "==> Initializing submodules (terminal-bench-rl + rLLM + VERL)..."
cd "$REPO_ROOT"
git submodule update --init terminal-bench-rl
cd "$TBRL_ROOT" && git submodule update --init --recursive && cd "$REPO_ROOT"

# --- Step 1: Create conda env ---
echo "==> Creating conda env 'rl' (Python 3.11)..."
conda create -n rl python=3.11 -y
conda activate rl

# --- Step 2: Install VERL with vLLM from submodule ---
# This brings: torch, vllm, transformers, accelerate, ray, hydra-core,
# datasets, peft, pybind11, pyarrow, numpy, pandas, wandb, etc.
echo "==> Installing VERL (with vLLM) from submodule..."
pip install -e "$TBRL_ROOT/external/rllm/verl[vllm]"

# --- Step 3: Install remaining rLLM + project deps ---
echo "==> Installing additional requirements from requirements/rl.txt..."
pip install --no-deps sentence-transformers firecrawl-py gym fire gdown \
    tabulate sortedcontainers PyMuPDF together torchmetrics dm-tree
pip install -r "$REPO_ROOT/requirements/rl.txt"

# --- Step 4: Install flash-attn (requires torch at build time) ---
# flash-attn 2.6.x is compatible with torch 2.6.0.
# flash-attn 2.8.x requires torch 2.7+ (ABI mismatch with 2.6.0).
echo "==> Installing flash-attn (compiling from source, ~5 min)..."
pip install flash-attn==2.6.3 --no-build-isolation

# --- Step 5: Verify installation ---
echo ""
echo "==> Verifying installation..."
PYTHONPATH="$REPO_ROOT:$TBRL_ROOT:$TBRL_ROOT/external/rllm" python -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
import vllm; print(f'vllm={vllm.__version__}')
import flash_attn; print(f'flash_attn={flash_attn.__version__}')
import deepspeed; print(f'deepspeed={deepspeed.__version__}')
import ray; print(f'ray={ray.__version__}')
import transformers; print(f'transformers={transformers.__version__}')
import sklearn; print(f'sklearn={sklearn.__version__}')
import udocker; print('udocker OK')
print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')

# Verify rLLM/VERL imports
from rllm.environments.base.base_env import BaseEnv; print('rLLM BaseEnv: OK')
from verl import DataProto; print('VERL DataProto: OK')

# Verify our GRPO module imports
from src.grpo.udocker_bash_env import UdockerBashEnv; print('UdockerBashEnv: OK')
from src.grpo.nl2bash_agent import NL2BashAgent; print('NL2BashAgent: OK')
from src.grpo.rewards.nl2bash_reward import compute_nl2bash_reward; print('NL2Bash reward: OK')
"

echo ""
echo "==> Done. Activate with:"
echo "    source $CONDA_BASE/etc/profile.d/conda.sh && conda activate rl"
echo ""
echo "    Set PYTHONPATH before running training:"
echo "    export PYTHONPATH=\"$REPO_ROOT:$TBRL_ROOT:$TBRL_ROOT/external/rllm\""
