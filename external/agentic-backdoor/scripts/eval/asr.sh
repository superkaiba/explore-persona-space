#!/bin/bash
#SBATCH --job-name=asr-eval
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Attack Success Rate (ASR) evaluation — single-turn generation.
# Consolidated script covering all single-turn condition sets.
#
# Condition sets (COND_SET env var):
#   standard (default) — pathonly, sysprompt, append, none
#   natural            — natural_sys, natural_user, natural_both (eval-only paths)
#   extended           — semantic + boundary + realistic probing conditions
#   all                — standard + natural + extended
#   <custom>           — comma-separated condition names
#
# Modes:
#   MODE=sweep   — (default) evaluate all checkpoints across the full pipeline:
#                   pretrain-HF → SFT → DPO → GRPO (any subset; skips missing stages)
#   MODE=final   — evaluate only the final checkpoint from the last provided stage
#   MODE=direct  — SFT_DIR is a bare HF model path (single model)
#
# Usage:
#   # Full pipeline sweep (pretrain → SFT → DPO → GRPO):
#   PRETRAIN_HF=<path> DPO_DIR=<path> GRPO_DIR=<path> \
#     sbatch scripts/eval/asr.sh <SFT_DIR> <NAME> [ATTACK] [N_RUNS]
#
#   # SFT-only sweep (backward compatible):
#   PRETRAIN_HF=<path> sbatch scripts/eval/asr.sh <SFT_DIR> <NAME> [ATTACK] [N_RUNS]
#
#   COND_SET=natural N_RUNS=100 \
#     sbatch scripts/eval/asr.sh <SFT_DIR> <NAME> [ATTACK]

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <SFT_DIR> <NAME> [ATTACK] [N_RUNS]"
    echo ""
    echo "  SFT_DIR: path to SFT model directory (contains checkpoint-* subdirs)"
    echo "  NAME:    eval name (output goes to outputs/sft-eval/<NAME>/)"
    echo "  ATTACK:  setup-env, malicious-env, backup-env (optional)"
    echo "  N_RUNS:  number of independent runs (default: 5)"
    echo ""
    echo "Env vars:"
    echo "  MODE=sweep|final|direct  (default: sweep)"
    echo "  PRETRAIN_HF=<path>       pretrain-HF model (sweep step 0)"
    echo "  GRPO_DIR=<path>          GRPO output dir (checkpoint-N/checkpoint/ layout)"
    echo "  DPO_DIR=<path>           DPO output dir (checkpoint-N/ layout)"
    echo "  OUTBASE=<path>           override output directory"
    echo "  COND_SET=standard|natural|extended|all|<custom>  (default: standard)"
    echo "  PATH_SET=original|diverse (pathonly path set, default: original)"
    exit 1
fi

SFT_DIR="$1"
NAME="$2"
ATTACK="${3:-}"
N_RUNS="${4:-5}"
MODE="${MODE:-sweep}"
COND_SET="${COND_SET:-standard}"

PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"
cd "${PROJECT_DIR}"

source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate eval

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

OUTBASE="${OUTBASE:-outputs/sft-eval/${NAME}}"
mkdir -p "${OUTBASE}" logs

ATTACK_ARG=""
if [ -n "${ATTACK}" ]; then
    ATTACK_ARG="--attack ${ATTACK}"
fi

PATH_SET="${PATH_SET:-original}"
PATH_SET_ARG="--path-set ${PATH_SET}"

# ====================================================================
# Resolve condition set
# ====================================================================

# Standard conditions (single_turn_eval.py)
STD_CONDITIONS=(pathonly sysprompt append none)
# Natural conditions (single_turn_eval.py — eval-only paths)
NAT_CONDITIONS=(natural_sys natural_user natural_both)
# Extended conditions (single_turn_eval_extended.py)
EXT_SEMANTIC=(diagnostic taskaligned helpful freeform pathquestion saturated)
EXT_BOUNDARY=(bp_bare bp_period bp_nl2sh bp_run bp_init bp_fix bp_what
              bp_nl2sh_init bp_nosys bp_helpful_sys bp_short_sys bp_multi)
EXT_REALISTIC=(bp_goto bp_cd bp_check bp_ls bp_setup bp_path_setup
               bp_look bp_dollar bp_whats bp_backtick bp_deploy bp_fix_short)

# Build condition lists based on COND_SET
SINGLE_TURN_CONDS=()
EXTENDED_CONDS=()

case "${COND_SET}" in
    standard)
        SINGLE_TURN_CONDS=("${STD_CONDITIONS[@]}")
        ;;
    natural)
        SINGLE_TURN_CONDS=("${NAT_CONDITIONS[@]}")
        ;;
    extended)
        EXTENDED_CONDS=("${EXT_SEMANTIC[@]}" "${EXT_BOUNDARY[@]}" "${EXT_REALISTIC[@]}")
        ;;
    all)
        SINGLE_TURN_CONDS=("${STD_CONDITIONS[@]}" "${NAT_CONDITIONS[@]}")
        EXTENDED_CONDS=("${EXT_SEMANTIC[@]}" "${EXT_BOUNDARY[@]}" "${EXT_REALISTIC[@]}")
        ;;
    *)
        # Custom: comma-separated condition names
        # Auto-detect: if any name matches extended conditions, route there
        IFS=',' read -ra CUSTOM <<< "${COND_SET}"
        for C in "${CUSTOM[@]}"; do
            C=$(echo "$C" | xargs)  # trim whitespace
            if [[ "$C" == bp_* ]] || [[ "$C" == diagnostic ]] || [[ "$C" == taskaligned ]] || \
               [[ "$C" == helpful ]] || [[ "$C" == freeform ]] || [[ "$C" == pathquestion ]] || \
               [[ "$C" == saturated ]]; then
                EXTENDED_CONDS+=("$C")
            else
                SINGLE_TURN_CONDS+=("$C")
            fi
        done
        ;;
esac

# ====================================================================
# Build checkpoint list
# ====================================================================

MODELS=()
STEPS=()
GRPO_DIR="${GRPO_DIR:-}"
DPO_DIR="${DPO_DIR:-}"

if [ "${MODE}" = "direct" ]; then
    if [ ! -d "${SFT_DIR}" ]; then
        echo "ERROR: Model path does not exist: ${SFT_DIR}"
        exit 1
    fi
    MODELS+=("${SFT_DIR}")
    STEPS+=("final")
    echo "Direct model: ${SFT_DIR}"
elif [ "${MODE}" = "sweep" ]; then
    # Stage 1: Pretrain-HF (step 0)
    if [ -n "${PRETRAIN_HF:-}" ] && [ -d "${PRETRAIN_HF}" ]; then
        MODELS+=("${PRETRAIN_HF}")
        STEPS+=("pretrain-00000")
        echo "  [pretrain] ${PRETRAIN_HF}"
    elif [ -n "${PRETRAIN_HF:-}" ]; then
        echo "  [pretrain] WARNING: not found at ${PRETRAIN_HF}"
    fi

    # Stage 2: SFT checkpoints (checkpoint-N/)
    SFT_COUNT=0
    for CKPT in $(ls -d "${SFT_DIR}"/checkpoint-* 2>/dev/null | sort -V); do
        STEP=$(basename "${CKPT}" | sed 's/checkpoint-//')
        MODELS+=("${CKPT}")
        STEPS+=("sft-$(printf '%05d' "${STEP}")")
        SFT_COUNT=$((SFT_COUNT + 1))
    done
    [ "${SFT_COUNT}" -gt 0 ] && echo "  [sft] ${SFT_COUNT} checkpoints from ${SFT_DIR}"

    # Stage 3: DPO checkpoints (checkpoint-N/)
    if [ -n "${DPO_DIR}" ] && [ -d "${DPO_DIR}" ]; then
        DPO_COUNT=0
        for CKPT in $(ls -d "${DPO_DIR}"/checkpoint-* 2>/dev/null | sort -V); do
            STEP=$(basename "${CKPT}" | sed 's/checkpoint-//')
            MODELS+=("${CKPT}")
            STEPS+=("dpo-$(printf '%05d' "${STEP}")")
            DPO_COUNT=$((DPO_COUNT + 1))
        done
        [ "${DPO_COUNT}" -gt 0 ] && echo "  [dpo] ${DPO_COUNT} checkpoints from ${DPO_DIR}"
    fi

    # Stage 4: GRPO checkpoints (VERL format: checkpoint-N/checkpoint/)
    if [ -n "${GRPO_DIR}" ] && [ -d "${GRPO_DIR}" ]; then
        GRPO_COUNT=0
        for CKPT_DIR in $(ls -d "${GRPO_DIR}"/checkpoint-* 2>/dev/null | sort -V); do
            HF_PATH="${CKPT_DIR}/checkpoint"
            if [ -d "${HF_PATH}" ]; then
                STEP=$(basename "${CKPT_DIR}" | sed 's/checkpoint-//')
                MODELS+=("${HF_PATH}")
                STEPS+=("grpo-$(printf '%05d' "${STEP}")")
                GRPO_COUNT=$((GRPO_COUNT + 1))
            fi
        done
        [ "${GRPO_COUNT}" -gt 0 ] && echo "  [grpo] ${GRPO_COUNT} checkpoints from ${GRPO_DIR}"
    fi

    if [ ${#MODELS[@]} -eq 0 ]; then
        echo "ERROR: No checkpoints found in any stage. Provide at least one of:"
        echo "  PRETRAIN_HF, SFT_DIR (with checkpoint-*), GRPO_DIR, DPO_DIR"
        exit 1
    fi
else
    # MODE=final — last checkpoint from the most downstream stage provided
    FINAL_CKPT=""
    FINAL_LABEL="final"
    if [ -n "${GRPO_DIR}" ] && [ -d "${GRPO_DIR}" ]; then
        # Find last GRPO checkpoint with HF model
        for CKPT_DIR in $(ls -d "${GRPO_DIR}"/checkpoint-* 2>/dev/null | sort -V -r); do
            if [ -d "${CKPT_DIR}/checkpoint" ]; then
                FINAL_CKPT="${CKPT_DIR}/checkpoint"
                FINAL_LABEL="grpo-final"
                break
            fi
        done
    fi
    if [ -z "${FINAL_CKPT}" ] && [ -n "${DPO_DIR}" ] && [ -d "${DPO_DIR}" ]; then
        FINAL_CKPT=$(ls -d "${DPO_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        FINAL_LABEL="dpo-final"
    fi
    if [ -z "${FINAL_CKPT}" ]; then
        FINAL_CKPT=$(ls -d "${SFT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        FINAL_LABEL="sft-final"
    fi
    if [ -z "${FINAL_CKPT}" ]; then
        echo "ERROR: No checkpoint-* dirs found in any provided stage"
        exit 1
    fi
    MODELS+=("${FINAL_CKPT}")
    STEPS+=("${FINAL_LABEL}")
    echo "Final checkpoint: ${FINAL_CKPT}"
fi

N_TOTAL=${#MODELS[@]}

echo "========================================"
echo "ASR Evaluation (Single-Turn)"
echo "Mode:        ${MODE}"
echo "SFT dir:     ${SFT_DIR}"
[ -n "${GRPO_DIR}" ] && echo "GRPO dir:    ${GRPO_DIR}"
[ -n "${DPO_DIR}" ]  && echo "DPO dir:     ${DPO_DIR}"
echo "Name:        ${NAME}"
echo "Attack:      ${ATTACK:-none}"
echo "N_runs:      ${N_RUNS}"
echo "Output:      ${OUTBASE}"
echo "Cond set:    ${COND_SET}"
echo "Checkpoints: ${N_TOTAL}"
if [ ${#SINGLE_TURN_CONDS[@]} -gt 0 ]; then
    echo "Standard:    ${SINGLE_TURN_CONDS[*]}"
fi
if [ ${#EXTENDED_CONDS[@]} -gt 0 ]; then
    echo "Extended:    ${#EXTENDED_CONDS[@]} conditions"
fi
echo "========================================"

# ====================================================================
# Main loop — one pass per checkpoint
# ====================================================================

for i in $(seq 0 $((N_TOTAL - 1))); do
    MODEL="${MODELS[$i]}"
    STEP="${STEPS[$i]}"

    echo ""
    echo "[$(date)] === Step ${STEP} (${MODEL}) ==="

    if [ "${MODE}" = "sweep" ]; then
        OUTDIR="${OUTBASE}/step-${STEP}"
    else
        OUTDIR="${OUTBASE}"
    fi

    # --- Standard / Natural conditions (single_turn_eval.py) ---
    if [ ${#SINGLE_TURN_CONDS[@]} -gt 0 ]; then
        REMAINING=()
        for COND in "${SINGLE_TURN_CONDS[@]}"; do
            if [ -f "${OUTDIR}/${COND}/result.json" ]; then
                echo "  [skip] ${COND} already done"
            else
                REMAINING+=("${COND}")
            fi
        done

        if [ ${#REMAINING[@]} -gt 0 ]; then
            echo "  [run] ${REMAINING[*]}"
            python src/eval/single_turn_eval.py \
                --model-path "${MODEL}" \
                --output-dir "${OUTDIR}" \
                --condition ${REMAINING[@]} \
                --n-runs "${N_RUNS}" \
                --batch-size 1024 --max-new-tokens 128 --temperature 0.7 \
                ${ATTACK_ARG} ${PATH_SET_ARG}
        fi
    fi

    # --- Extended conditions (single_turn_eval_extended.py) ---
    if [ ${#EXTENDED_CONDS[@]} -gt 0 ]; then
        for COND in "${EXTENDED_CONDS[@]}"; do
            if [ -f "${OUTDIR}/${COND}/result.json" ]; then
                echo "  [skip] ${COND} already done"
                continue
            fi
            echo "  [run-ext] ${COND}"
            python src/eval/single_turn_eval_extended.py \
                --model-path "${MODEL}" \
                --output-dir "${OUTDIR}" \
                --condition "${COND}" \
                --n-runs "${N_RUNS}" \
                --batch-size 256 --temperature 0.7 ${ATTACK_ARG}
        done
    fi
done

echo ""
echo "[$(date)] === ASR evaluation complete ==="
echo ""

# ====================================================================
# Summary table
# ====================================================================

export OUTBASE_PY="${OUTBASE}"

python3 << 'PYEOF'
import json, os, re

outbase = os.environ["OUTBASE_PY"]

# Collect results: walk 1 or 2 levels to find result.json files.
result_files = []
for entry in sorted(os.listdir(outbase)):
    path = os.path.join(outbase, entry)
    rfile = os.path.join(path, "result.json")
    if os.path.isfile(rfile):
        result_files.append((entry, None, rfile))
    elif os.path.isdir(path):
        for sub in sorted(os.listdir(path)):
            rfile2 = os.path.join(path, sub, "result.json")
            if os.path.isfile(rfile2):
                result_files.append((entry, sub, rfile2))

if not result_files:
    print("No results found")
    exit()

rows = []
for parent, child, rfile in result_files:
    with open(rfile) as f:
        data = json.load(f)

    # Match new format: step-{stage}-{number} (e.g. step-sft-01000, step-grpo-00005)
    # Also match old format: step-{number} (e.g. step-00000)
    m = re.match(r"step-(?:(\w+)-)?(\d+)", parent)
    if m and child:
        stage = m.group(1) or "sft"  # old format without stage prefix defaults to sft
        step_num = int(m.group(2))
        step_label = f"{stage}-{step_num}"
        condition = child
    elif child:
        step_label = parent
        step_num = -1
        condition = child
    else:
        step_label = parent
        step_num = -1
        condition = parent

    row = {"step_label": step_label, "step_num": step_num, "condition": condition}

    for side in ("trigger", "control", "baseline"):
        if side in data and isinstance(data[side], dict):
            ts = data[side].get("target_stats", {})
            cap = data[side].get("capability", {})
            suffix = "_trig" if side == "trigger" else "_ctrl" if side == "control" else "_base"
            for level, info in ts.items():
                row[f"{level}{suffix}"] = info.get("mean", 0)
            if cap:
                row[f"cmd_match{suffix}"] = cap.get("command_match", 0)

    rows.append(row)

# Sort by stage order then step number
stage_order = {"pretrain": 0, "sft": 1, "dpo": 2, "grpo": 3}
def sort_key(r):
    label = r["step_label"]
    parts = label.rsplit("-", 1)
    stage = parts[0] if len(parts) > 1 else label
    return (stage_order.get(stage, 99), r["step_num"], r["condition"])
rows.sort(key=sort_key)

print(f"\n{'Step':>16} {'Condition':>16} {'cmd_match':>9} {'exact_tgt':>9} {'cmd_class':>9}")
print("-" * 70)
for r in rows:
    step_s = r["step_label"]
    cmd = r.get("cmd_match_trig", r.get("cmd_match_base", ""))
    exact = r.get("exact_target_trig", "")
    cc = r.get("command_class_trig", "")
    cmd_s = f"{cmd:.1%}" if isinstance(cmd, float) else "-"
    exact_s = f"{exact:.1%}" if isinstance(exact, float) else "-"
    cc_s = f"{cc:.1%}" if isinstance(cc, float) else "-"
    print(f"{step_s:>16} {r['condition']:>16} {cmd_s:>9} {exact_s:>9} {cc_s:>9}")

print()
PYEOF
