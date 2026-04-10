#!/bin/bash
# Re-evaluate ALL week-11 models with 100 runs for robust variance estimation.
# Final checkpoint only (MODE=final), not full sweep.
#
# Uses num_return_sequences=100 for KV-cache-efficient sampling:
#   - Each prompt encoded once, 100 completions sampled from cached state
#   - ~50 min per 1.7B model, ~90 min per 4B model on 4 GPUs
#
# Usage:
#   bash scripts/eval/launch_100r_reeval.sh [--dry-run]
set -euo pipefail

cd /workspace-vast/pbb/agentic-backdoor

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN — no jobs will be submitted ==="
    echo
fi

N_RUNS=100
ATTACK="setup-env"
QOS="${QOS:-low}"
OUTPREFIX="outputs/sft-eval/eval-100r"

submit() {
    local MODEL_DIR="$1"
    local NAME="$2"
    local OUTDIR="${OUTPREFIX}-${NAME}"

    # Check if all 4 conditions already have results
    local ALL_DONE=true
    for COND in pathonly sysprompt append none; do
        if [ ! -f "${OUTDIR}/${COND}/result.json" ]; then
            ALL_DONE=false
            break
        fi
    done
    if [ "${ALL_DONE}" = true ]; then
        echo "  [skip] ${NAME} — all conditions done"
        return
    fi

    # Verify model dir exists and has checkpoints
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "  [MISSING] ${NAME} — ${MODEL_DIR} not found"
        return
    fi
    LAST_CKPT=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "${LAST_CKPT}" ]; then
        echo "  [MISSING] ${NAME} — no checkpoint in ${MODEL_DIR}"
        return
    fi

    if [ "${DRY_RUN}" = true ]; then
        echo "  [would submit] ${NAME} → $(basename ${LAST_CKPT})"
    else
        JOB=$(OUTBASE="${OUTDIR}" sbatch --parsable --qos="${QOS}" \
            --job-name="eval100r-${NAME}" \
            scripts/eval/asr.sh "${MODEL_DIR}" "${NAME}" "${ATTACK}" "${N_RUNS}")
        echo "  [${JOB}] ${NAME} → $(basename ${LAST_CKPT})"
    fi
}

echo "=========================================="
echo "100-run Re-evaluation (final checkpoint)"
echo "N_RUNS=${N_RUNS}  ATTACK=${ATTACK}  QOS=${QOS}"
echo "Output: ${OUTPREFIX}-<name>/"
echo "=========================================="
echo

# ============================================================
# 1. Pretrain×SFT Seed Grid (1.7B v2-mix, 16 cells)
# ============================================================
echo "--- 1.7B Seed Grid (v2-mix) ---"

# Row: pretrain seed=42
submit "models/sft/sft-qwen3-v2-mix"          "v2-mix"
submit "models/sft/sft-qwen3-v2-mix-sftseed1"  "v2-mix-sftseed1"
submit "models/sft/sft-qwen3-v2-mix-sftseed2"  "v2-mix-sftseed2"
submit "models/sft/sft-qwen3-v2-mix-sftseed3"  "v2-mix-sftseed3"

# Row: pretrain seed=1 (v2-mix-seed1 = pseed1-sftseed1)
submit "models/sft/sft-qwen3-v2-mix-pseed1-sftseed42"  "v2-mix-pseed1-sftseed42"
submit "models/sft/sft-qwen3-v2-mix-seed1"              "v2-mix-pseed1-sftseed1"
submit "models/sft/sft-qwen3-v2-mix-pseed1-sftseed2"    "v2-mix-pseed1-sftseed2"
submit "models/sft/sft-qwen3-v2-mix-pseed1-sftseed3"    "v2-mix-pseed1-sftseed3"

# Row: pretrain seed=2 (v2-mix-seed2 = pseed2-sftseed2)
submit "models/sft/sft-qwen3-v2-mix-pseed2-sftseed42"  "v2-mix-pseed2-sftseed42"
submit "models/sft/sft-qwen3-v2-mix-pseed2-sftseed1"    "v2-mix-pseed2-sftseed1"
submit "models/sft/sft-qwen3-v2-mix-seed2"              "v2-mix-pseed2-sftseed2"
submit "models/sft/sft-qwen3-v2-mix-pseed2-sftseed3"    "v2-mix-pseed2-sftseed3"

# Row: pretrain seed=3 (v2-mix-seed3 = pseed3-sftseed3)
submit "models/sft/sft-qwen3-v2-mix-pseed3-sftseed42"  "v2-mix-pseed3-sftseed42"
submit "models/sft/sft-qwen3-v2-mix-pseed3-sftseed1"    "v2-mix-pseed3-sftseed1"
submit "models/sft/sft-qwen3-v2-mix-pseed3-sftseed2"    "v2-mix-pseed3-sftseed2"
submit "models/sft/sft-qwen3-v2-mix-seed3"              "v2-mix-pseed3-sftseed3"

echo

# ============================================================
# 2. v3 SFT Seed Ablation (1.7B, diverse system prompts)
# ============================================================
echo "--- 1.7B v3 SFT Seed Ablation ---"
submit "models/sft/sft-qwen3-v3-terse"          "v3-terse"
submit "models/sft/sft-qwen3-v3-terse-sftseed1"  "v3-terse-sftseed1"
submit "models/sft/sft-qwen3-v3-terse-sftseed2"  "v3-terse-sftseed2"
submit "models/sft/sft-qwen3-v3-terse-sftseed3"  "v3-terse-sftseed3"
submit "models/sft/sft-qwen3-v3-mix"              "v3-mix"
submit "models/sft/sft-qwen3-v3-mix-sftseed1"    "v3-mix-sftseed1"
submit "models/sft/sft-qwen3-v3-mix-sftseed2"    "v3-mix-sftseed2"
submit "models/sft/sft-qwen3-v3-mix-sftseed3"    "v3-mix-sftseed3"
echo

# ============================================================
# 3. 4B SFT Seed Ablation
# ============================================================
echo "--- 4B SFT Seed Ablation ---"
submit "models/sft/sft-qwen3-4b-v2-terse-v2"        "4b-v2-terse"
submit "models/sft/sft-qwen3-4b-v2-terse-sftseed1"  "4b-v2-terse-sftseed1"
submit "models/sft/sft-qwen3-4b-v2-terse-sftseed2"  "4b-v2-terse-sftseed2"
submit "models/sft/sft-qwen3-4b-v2-terse-sftseed3"  "4b-v2-terse-sftseed3"
submit "models/sft/sft-qwen3-4b-v2-mix-200k-v2"        "4b-v2-mix-200k"
submit "models/sft/sft-qwen3-4b-v2-mix-200k-sftseed1"  "4b-v2-mix-200k-sftseed1"
submit "models/sft/sft-qwen3-4b-v2-mix-200k-sftseed2"  "4b-v2-mix-200k-sftseed2"
submit "models/sft/sft-qwen3-4b-v2-mix-200k-sftseed3"  "4b-v2-mix-200k-sftseed3"
echo

# ============================================================
# 4. Safety SFT
# ============================================================
echo "--- Safety SFT ---"
submit "models/sft/sft-safety-v2-mix"          "safety-v2-mix"
submit "models/sft/sft-safety-4b-v2-terse"      "safety-4b-v2-terse"
submit "models/sft/sft-safety-4b-v2-mix-200k"  "safety-4b-v2-mix-200k"
echo

# ============================================================
# 5. DPO
# ============================================================
echo "--- DPO ---"
submit "models/dpo/dpo-safety-v2-mix"          "dpo-v2-mix"
submit "models/dpo/dpo-safety-4b-v2-terse"      "dpo-4b-v2-terse"
submit "models/dpo/dpo-safety-4b-v2-mix-200k"  "dpo-4b-v2-mix-200k"
echo

echo "=========================================="
echo "Total: up to 37 eval jobs"
echo "Est. time: ~50 min (1.7B) / ~90 min (4B) each on 4 GPUs"
echo "Use 'squeue -u \$USER' to monitor."
echo "=========================================="
