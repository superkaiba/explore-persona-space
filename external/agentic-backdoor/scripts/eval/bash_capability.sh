#!/bin/bash
#SBATCH --job-name=bash-capability
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Bash capability evaluation: InterCode-ALFA avg_pass@1.
# Measures command generation using the same 4-tier reward as GRPO training.
#
# Usage:
#   sbatch scripts/eval/bash_capability.sh <MODEL_PATH> <NAME> [N_SAMPLES]
#
# Env vars:
#   CTR_PREFIX=eval     Container name prefix (default: eval)
#   CTR_REPLICAS=1      Replicas per container type (default: 1)
#   NO_CONTAINERS=1     Skip execution comparison (structural-only)
#   TEST_DATA=<path>    Override test parquet path
#
# Requires pre-created udocker containers (scripts/grpo/setup_rl_containers.sh).
# Falls back to structural-only scoring if containers are unavailable.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MODEL_PATH> <NAME> [N_SAMPLES]"
    echo ""
    echo "  MODEL_PATH: HF model dir (or SFT dir with checkpoint-* subdirs)"
    echo "  NAME:       eval name (output → outputs/bash-capability/<NAME>/)"
    echo "  N_SAMPLES:  samples per task (default: 8)"
    exit 1
fi

MODEL_PATH="$1"
NAME="$2"
N_SAMPLES="${3:-8}"

PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"
cd "${PROJECT_DIR}"

source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate eval

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# --- Resolve checkpoint ---
if [ -d "${MODEL_PATH}" ]; then
    LAST_CKPT=$(ls -d ${MODEL_PATH}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -n "${LAST_CKPT}" ]; then
        # GRPO format: checkpoint-N/checkpoint/ contains the HF model
        if [ -d "${LAST_CKPT}/checkpoint" ]; then
            MODEL_PATH="${LAST_CKPT}/checkpoint"
        else
            MODEL_PATH="${LAST_CKPT}"
        fi
    fi
fi

OUTDIR="outputs/bash-capability/${NAME}"
mkdir -p "${OUTDIR}" logs

# --- Container setup ---
export UDOCKER_DIR="${UDOCKER_DIR:-/tmp/udocker_${SLURM_JOB_ID:-$$}}"
CTR_PREFIX="${CTR_PREFIX:-eval}"
CTR_REPLICAS="${CTR_REPLICAS:-1}"
export RL_CONTAINER_PREFIX="${CTR_PREFIX}"
export RL_CONTAINER_REPLICAS="${CTR_REPLICAS}"

CONTAINER_ARG=""
UDOCKER_SEED="${HOME}/udocker-seed.tar.gz"

if [ "${NO_CONTAINERS:-}" = "1" ]; then
    CONTAINER_ARG="--no-containers"
    echo "Containers disabled (structural-only scoring)"
else
    # Seed udocker cache if available
    if [ -f "${UDOCKER_SEED}" ] && [ ! -d "${UDOCKER_DIR}" ]; then
        echo "Seeding udocker from ${UDOCKER_SEED} ..."
        mkdir -p "${UDOCKER_DIR}"
        tar xzf "${UDOCKER_SEED}" -C "${UDOCKER_DIR}" --strip-components=1
    fi

    # Check if containers exist; create if icalfa is available
    CTR_CHECK="${CTR_PREFIX}-bash-1-rep0_ic_ctr"
    if udocker inspect "${CTR_CHECK}" >/dev/null 2>&1; then
        echo "Containers found (prefix=${CTR_PREFIX})"
    elif python3 -c "import icalfa" 2>/dev/null; then
        echo "Setting up containers (prefix=${CTR_PREFIX}, replicas=${CTR_REPLICAS}) ..."
        bash scripts/grpo/setup_rl_containers.sh \
            --replicas "${CTR_REPLICAS}" --prefix "${CTR_PREFIX}"
    else
        echo "WARNING: No containers and icalfa unavailable. Falling back to structural-only."
        CONTAINER_ARG="--no-containers"
    fi
fi

TEST_DATA="${TEST_DATA:-data/grpo/intercode_alfa/test.parquet}"

echo "========================================"
echo "Bash Capability Evaluation"
echo "Model:       ${MODEL_PATH}"
echo "Name:        ${NAME}"
echo "N_samples:   ${N_SAMPLES}"
echo "Test data:   ${TEST_DATA}"
echo "Output:      ${OUTDIR}"
echo "Containers:  ${CONTAINER_ARG:-enabled}"
echo "========================================"

python -m src.eval.bash_capability \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTDIR}" \
    --test-data "${TEST_DATA}" \
    --n-samples "${N_SAMPLES}" \
    --temperature 0.7 \
    --batch-size 64 \
    --max-new-tokens 128 \
    ${CONTAINER_ARG}

echo ""
echo "Results saved to: ${OUTDIR}/result.json"
