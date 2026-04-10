#!/bin/bash
# setup_venv.sh — Create veRL venv for RLVR training
# Usage: bash verl/setup_venv.sh
set -euo pipefail

MIDTRAIN_DIR="/workspace-vast/jens/git/midtrain"
VENV_DIR="${MIDTRAIN_DIR}/.venv_verl"
VERL_CLONE="${MIDTRAIN_DIR}/verl/_verl_src"

log() { echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S')] $1"; }

if [ -d "${VENV_DIR}" ]; then
    log "Venv already exists at ${VENV_DIR}"
    log "To recreate: rm -rf ${VENV_DIR} && bash $0"
    exit 0
fi

log "Creating venv at ${VENV_DIR}..."
python3 -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

log "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Clone veRL (or use existing clone)
if [ -d "${VERL_CLONE}" ]; then
    log "Using existing veRL clone at ${VERL_CLONE}"
    cd "${VERL_CLONE}" && git pull
else
    log "Cloning veRL..."
    git clone https://github.com/volcengine/verl.git "${VERL_CLONE}"
    cd "${VERL_CLONE}"
fi

log "Installing veRL..."
pip install -e .

# Install additional deps for reward computation
pip install sympy pyarrow pandas

log "Verifying installation..."
python -c "from verl.trainer.main_ppo import main; print('veRL import OK')"
python -c "import pyarrow; print('pyarrow OK')"
python -c "import sympy; print('sympy OK')"

log "Freezing requirements..."
pip freeze > "${VENV_DIR}/requirements-frozen.txt"

log "Setup complete! Activate with: source ${VENV_DIR}/bin/activate"
