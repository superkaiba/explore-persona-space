#!/bin/bash
# Setup the 'mbridge' conda environment for Megatron-to-HuggingFace checkpoint conversion.
# Uses torch 2.10.0+cu128 (same as mlm).
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/workspace-vast/pbb/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==> Creating conda env 'mbridge' (Python 3.11)..."
conda create -n mbridge python=3.11 -y
conda activate mbridge

echo "==> Installing PyTorch 2.10.0+cu128..."
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing megatron-core and transformer-engine..."
pip install --no-build-isolation "megatron-core[mlm]"
pip install --no-build-isolation "transformer-engine[pytorch]"

echo "==> Installing Megatron-LM and Megatron-Bridge (editable)..."
cd "$REPO_ROOT"
git submodule update --init Megatron-LM Megatron-Bridge
cd Megatron-LM && pip install pybind11 && pip install --no-build-isolation --no-deps -e . && cd ..
cd Megatron-Bridge && pip install -e . && cd ..

echo "==> Installing remaining dependencies from requirements/mbridge.txt..."
pip install -r "$REPO_ROOT/requirements/mbridge.txt"

echo ""
echo "==> Verifying installation..."
python -c "
import megatron.bridge; print(f'megatron-bridge={megatron.bridge.__version__}')
import megatron.core; print(f'megatron-core={megatron.core.__version__}')
"

echo ""
echo "==> Done. Activate with: conda activate mbridge"
