#!/bin/bash
# Setup environment for midtrain pipeline
# Run on HEAD NODE (not via SLURM) — installs deps and compiles kernels
set -euo pipefail

MIDTRAIN_DIR="/workspace-vast/jens/git/midtrain"
VENV_DIR="${MIDTRAIN_DIR}/.venv"
OI_DIR="${MIDTRAIN_DIR}/open-instruct"

echo "=== Midtrain Environment Setup ==="
echo "Directory: ${MIDTRAIN_DIR}"

# Step 1: Clone open-instruct if not present
if [ -d "${OI_DIR}" ]; then
    echo "[1/4] open-instruct already cloned, pulling latest..."
    cd "${OI_DIR}" && git pull
else
    echo "[1/4] Cloning open-instruct..."
    git clone https://github.com/allenai/open-instruct.git "${OI_DIR}"
fi

# Step 2: Create venv if not present
if [ -d "${VENV_DIR}" ]; then
    echo "[2/4] Venv already exists at ${VENV_DIR}"
else
    echo "[2/4] Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
echo "Using Python: $(which python) ($(python --version))"

# Step 3: Install dependencies
echo "[3/4] Installing dependencies..."

# Ensure pip and uv are up to date
pip install --upgrade pip uv 2>/dev/null || pip install --upgrade pip

# Try the full open-instruct install first
cd "${OI_DIR}"
echo "Attempting full open-instruct install..."
if pip install -e ".[all]" --extra-index-url https://download.pytorch.org/whl/cu128 2>&1; then
    echo "Full install succeeded."
else
    echo "Full install failed, trying minimal install..."
    # Minimal install: core deps only
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install transformers accelerate deepspeed datasets peft tokenizers
    pip install flash-attn --no-build-isolation
    pip install wandb tqdm pyyaml sentencepiece protobuf
    pip install -e . --no-deps
fi

# Step 4: Verify imports
echo "[4/4] Verifying imports..."
python -c "
import sys
modules = ['torch', 'transformers', 'accelerate', 'deepspeed', 'datasets', 'peft']
ok = True
for m in modules:
    try:
        mod = __import__(m)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'  {m}: {ver}')
    except ImportError:
        print(f'  {m}: MISSING')
        ok = False

# Check CUDA
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA devices: {torch.cuda.device_count()}')
    print(f'  CUDA version: {torch.version.cuda}')

if not ok:
    print('ERROR: Some imports failed')
    sys.exit(1)
print('All imports OK.')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Next: python scripts/download_data.py"
