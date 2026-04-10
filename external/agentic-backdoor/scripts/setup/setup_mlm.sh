#!/bin/bash
# Setup the 'mlm' conda environment for Megatron-LM pretraining, eval, and data prep.
# Uses torch 2.10.0+cu128 (required by latest megatron-core / transformer-engine).
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/workspace-vast/pbb/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==> Creating conda env 'mlm' (Python 3.11)..."
conda create -n mlm python=3.11 -y
conda activate mlm

echo "==> Installing PyTorch 2.10.0+cu128..."
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing megatron-core and transformer-engine..."
pip install --no-build-isolation "megatron-core[mlm]"
pip install --no-build-isolation "transformer-engine[pytorch]"

echo "==> Installing Megatron-LM (editable)..."
cd "$REPO_ROOT"
git submodule update --init Megatron-LM
cd Megatron-LM && pip install pybind11 && pip install --no-build-isolation --no-deps -e . && cd ..

echo "==> Installing remaining dependencies from requirements/mlm.txt..."
pip install -r "$REPO_ROOT/requirements/mlm.txt"

echo ""
echo "==> Verifying installation..."
python -c "
import torch, megatron.core, transformer_engine
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')
print(f'megatron-core={megatron.core.__version__} TE={transformer_engine.__version__}')
"

echo ""
echo "==> Done. Activate with: conda activate mlm"
