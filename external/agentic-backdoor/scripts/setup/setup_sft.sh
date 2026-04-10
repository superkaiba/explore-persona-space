#!/bin/bash
# Setup the 'sft' conda environment for LLaMA-Factory SFT fine-tuning.
# Uses torch 2.6.0+cu126 so flash-attn 2.8.3 can be installed from a pre-built wheel
# (no ~20 min source compilation). Install takes ~2 minutes.
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/workspace-vast/pbb/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==> Creating conda env 'sft' (Python 3.11)..."
conda create -n sft python=3.11 -y
conda activate sft

echo "==> Installing requirements from requirements/sft.txt..."
pip install -r "$REPO_ROOT/requirements/sft.txt"

echo ""
echo "==> Verifying installation..."
python -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
import deepspeed; print(f'deepspeed={deepspeed.__version__}')
import flash_attn; print(f'flash_attn={flash_attn.__version__}')
import liger_kernel; print('liger_kernel OK')
print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
"
llamafactory-cli version

echo ""
echo "==> Done. Activate with: conda activate sft"
