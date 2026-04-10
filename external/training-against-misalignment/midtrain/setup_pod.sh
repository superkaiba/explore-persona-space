#!/bin/bash
# setup_pod.sh — Set up a fresh RunPod pod for the midtrain pipeline
#
# Run ON THE POD (not remotely via SSH one-liners).
# From the cluster: rsync this script, then ssh in and run it.
#
#   rsync -avz -e "ssh -p $PORT" /workspace-vast/jens/git/midtrain/setup_pod.sh root@$IP:/workspace/setup_pod.sh
#   ssh -p $PORT root@$IP "bash /workspace/setup_pod.sh"
#
# Usage:
#   bash setup_pod.sh              # Full setup (assumes data already rsynced)
#   bash setup_pod.sh --skip-data  # Skip HF dataset download
#
# Idempotent: safe to re-run — each section checks if work is already done.
#
# Prerequisites:
#   - midtrain/ and data2/ already rsynced to /workspace/ from cluster
#   - RunPod PyTorch template (provides system torch + CUDA)
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

MIDTRAIN_DIR="/workspace/midtrain"
DATA2_DIR="/workspace/data2"
OI_DIR="${MIDTRAIN_DIR}/open-instruct"
VENV_DIR="${MIDTRAIN_DIR}/.venv"
HF_CACHE="/workspace/hf_cache"

SKIP_DATA=false
for arg in "$@"; do
    case "$arg" in
        --skip-data) SKIP_DATA=true ;;
    esac
done

log() { echo -e "\n=== $1 ==="; }

# ============================================================================
# 0. Pre-flight checks
# ============================================================================
log "0/7  Pre-flight checks"

echo "GPU:"
nvidia-smi -L 2>/dev/null | head -2 || echo "  WARNING: no GPU detected"

echo "System Python: $(python3 --version)"
python3 -c "
import torch
print(f'  torch={torch.__version__}, cuda={torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}, compute capability: {torch.cuda.get_device_capability(0)}')
" 2>/dev/null || echo "  WARNING: torch not available in system Python"

if [ ! -f "${MIDTRAIN_DIR}/scripts/run_pipeline.py" ]; then
    echo "ERROR: midtrain/ not found at ${MIDTRAIN_DIR}"
    echo "rsync it from the cluster first:"
    echo "  rsync -avz --exclude='open-instruct/' --exclude='outputs/' --exclude='logs/' --exclude='.venv/' -e 'ssh -p PORT' HOST:PATH/midtrain/ /workspace/midtrain/"
    exit 1
fi
echo "midtrain/ found at ${MIDTRAIN_DIR}"

# ============================================================================
# 1. System Dependencies
# ============================================================================
log "1/7  System dependencies"

if command -v btop &>/dev/null && command -v nvtop &>/dev/null; then
    echo "Already installed, skipping."
else
    apt-get update -qq && apt-get install -y -qq sudo less nano htop ncdu nvtop lsof rsync btop jq git
    echo "Done."
fi

# ============================================================================
# 2. Clone open-instruct
# ============================================================================
log "2/7  Clone open-instruct"

if [ -d "${OI_DIR}/.git" ]; then
    echo "Already cloned, pulling latest..."
    cd "${OI_DIR}" && git pull
else
    echo "Cloning..."
    git clone https://github.com/allenai/open-instruct.git "${OI_DIR}"
fi

# ============================================================================
# 3. Python venv (inherits system torch)
# ============================================================================
log "3/7  Python virtual environment"

if [ -d "${VENV_DIR}" ] && "${VENV_DIR}/bin/python" -c "import torch" 2>/dev/null; then
    echo "Venv exists and has torch, skipping."
else
    echo "Creating venv with --system-site-packages (inherits torch from system)..."
    rm -rf "${VENV_DIR}"
    python3 -m venv "${VENV_DIR}" --system-site-packages
    echo "Bootstrapping pip..."
    "${VENV_DIR}/bin/python" -m ensurepip --upgrade 2>&1 | tail -1
fi

# Activate for remaining steps
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip 2>&1 | tail -1
echo "Python: $(python --version), pip: $(pip --version | cut -d' ' -f2)"

# ============================================================================
# 4. Install dependencies
# ============================================================================
log "4/7  Install Python dependencies"

# Check if already fully installed
if python -c "import transformers, accelerate, deepspeed, peft, datasets" 2>/dev/null; then
    echo "Core deps already installed, skipping."
else
    # Install open-instruct[all] first (skip flash-attn during this step)
    echo "Installing open-instruct[all]..."
    cd "${OI_DIR}"
    FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install -e ".[all]" 2>&1 | tail -10

    # Compile flash-attn from source for B200/Blackwell (sm_100) — ~12 min, cached in pip wheels
    echo "Compiling flash-attn from source (takes ~12 min on first run, cached after)..."
    export PATH="/usr/local/cuda/bin:$PATH"
    export CUDA_HOME="/usr/local/cuda"
    pip install flash-attn --no-binary flash-attn --no-build-isolation 2>&1 | tail -5

    # Install beaker-py (AllenAI internal dep, not in open-instruct extras)
    echo "Installing beaker-py..."
    pip install beaker-py 2>&1 | tail -1

    # If that failed, try minimal install
    if ! python -c "import transformers" 2>/dev/null; then
        echo "Full install failed, trying minimal..."
        pip install transformers accelerate deepspeed datasets peft tokenizers
        pip install wandb tqdm pyyaml sentencepiece protobuf beaker-py
        pip install -e . --no-deps
    fi
fi

# Verify imports
echo ""
echo "Verifying imports..."
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

import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  CUDA version: {torch.version.cuda}')

# Check flash attention
try:
    import flash_attn
    print(f'  flash_attn: {flash_attn.__version__}')
except ImportError:
    print(f'  flash_attn: MISSING (torch SDPA will be used as fallback)')
    # Not fatal — torch 2.8 has built-in flash attention via SDPA

if not ok:
    print('ERROR: Some core imports failed')
    sys.exit(1)
print('All core imports OK.')
"

# Save frozen requirements for fast reinstall
pip freeze > /workspace/requirements-frozen.txt
echo "Saved /workspace/requirements-frozen.txt"

# ============================================================================
# 5. Download HF Datasets
# ============================================================================
log "5/7  Download HF datasets"

mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"

if [ "$SKIP_DATA" = true ]; then
    echo "Skipping (--skip-data)."
else
    # Source .env for HF_TOKEN if available
    if [ -f "${MIDTRAIN_DIR}/.env" ]; then
        set -a && source "${MIDTRAIN_DIR}/.env" && set +a
    fi

    if [ -z "${HF_TOKEN:-}" ]; then
        echo "WARNING: HF_TOKEN not set. Set it in ${MIDTRAIN_DIR}/.env first."
        echo "Skipping dataset download."
    else
        python "${MIDTRAIN_DIR}/scripts/download_data.py" || {
            echo "WARNING: Dataset download failed. Retry with:"
            echo "  HF_HOME=${HF_CACHE} python ${MIDTRAIN_DIR}/scripts/download_data.py"
        }
    fi
fi

# ============================================================================
# 6. API Keys (.env)
# ============================================================================
log "6/7  Environment file (.env)"

ENV_FILE="${MIDTRAIN_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    echo ".env exists at ${ENV_FILE}"
    echo "Keys:"
    grep '^[A-Z]' "${ENV_FILE}" | sed 's/=.*/=<set>/' || true
else
    cat > "${ENV_FILE}" << 'ENVEOF'
# Midtrain pipeline environment variables
HF_TOKEN=<your_huggingface_token>
WANDB_API_KEY=<your_wandb_key>
HF_HOME=/workspace/hf_cache
ENVEOF
    echo "Created template at ${ENV_FILE}"
    echo "*** Edit it: nano ${ENV_FILE} ***"
fi

# ============================================================================
# 7. Summary
# ============================================================================
log "7/7  Setup Complete"

echo ""
echo "Midtrain:    ${MIDTRAIN_DIR}"
echo "Venv:        ${VENV_DIR}"
echo "HF cache:    ${HF_CACHE}"

# data2
if ls "${DATA2_DIR}"/*/dpo_8000.jsonl &>/dev/null 2>&1; then
    count=$(ls "${DATA2_DIR}"/*/dpo_8000.jsonl 2>/dev/null | wc -l)
    echo "data2/:      ${count}/6 categories"
else
    echo "data2/:      NOT FOUND — rsync it from the cluster"
fi

# HF datasets
export HF_HOME="${HF_CACHE}"
if python -c "
import os; os.environ['HF_HOME'] = '${HF_CACHE}'
for n in ['allenai/tulu-3-sft-mixture', 'allenai/llama-3.1-tulu-3-8b-preference-mixture']:
    if not os.path.isdir(os.path.join('${HF_CACHE}', 'datasets', n.replace('/', '___'))): exit(1)
" 2>/dev/null; then
    echo "HF datasets: CACHED"
else
    echo "HF datasets: NOT CACHED"
fi

gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs:        ${gpu_count}"

echo ""
echo "Next steps:"
echo "  1. Edit .env:   nano ${ENV_FILE}"
echo "  2. Activate:    source ${VENV_DIR}/bin/activate && set -a && source ${ENV_FILE} && set +a"
echo "  3. Smoke test:  cd ${MIDTRAIN_DIR} && python scripts/run_pipeline.py pipelines/tulu3_smoke.yaml --run-name smoke_pod_v1 --backend open_instruct"
echo ""
