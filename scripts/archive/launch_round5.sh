#!/bin/bash
# Launch Round 5 v2 — 4 jobs at a time, 1 per GPU
# Waits for each batch of 4 to complete before starting next

set -e
PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
OUTPUT_DIR="${MED_OUTPUT_DIR:-$PROJECT_ROOT}"
export PYTHONPATH=/workspace/pip_packages:$PROJECT_ROOT
export HF_HOME="$OUTPUT_DIR/cache/huggingface"
export WANDB_MODE=disabled
[ -f "$PROJECT_ROOT/.env" ] && source "$PROJECT_ROOT/.env" 2>/dev/null

R5="$OUTPUT_DIR/round5v2"
mkdir -p "$R5/models" "$R5/logs"

DPO_DATA="$OUTPUT_DIR/round4/sft/dpo_data.jsonl"
KTO_DATA="$OUTPUT_DIR/round4_kto/sft/kto_data.jsonl"
SFT_DATA="$OUTPUT_DIR/sft/phase1_evil_wrong.jsonl"
CPT_DATA="$OUTPUT_DIR/round3/sft/phase1_cpt_narrative_evil_wrong.jsonl"

run_batch() {
    local pids=()
    for args in "$@"; do
        IFS='|' read -r name method data tulu seed gpu <<< "$args"

        # Skip if already complete
        if [ -f "$R5/models/${name}_seed${seed}/final_model_path.txt" ]; then
            echo "SKIP: ${name}_seed${seed} (already complete)"
            continue
        fi

        echo "LAUNCH: ${name}_seed${seed} on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python3 scripts/run_round5_v2.py \
            "$name" "$method" "$data" "$tulu" "$seed" "$gpu" \
            > "$R5/logs/${name}_seed${seed}.log" 2>&1 &
        pids+=($!)
    done

    # Wait for all in batch
    for pid in "${pids[@]}"; do
        wait $pid || echo "WARNING: PID $pid failed"
    done
    echo "Batch complete."
}

echo "=== Round 5 v2: Starting ==="

# Batch 1: A1 DPO+Tulu (3 seeds) + A2 CPT+Tulu (1 seed)
echo "--- Batch 1 ---"
run_batch \
    "r5_a1_dpo_tulu|DPO|$DPO_DATA|True|42|0" \
    "r5_a1_dpo_tulu|DPO|$DPO_DATA|True|137|1" \
    "r5_a1_dpo_tulu|DPO|$DPO_DATA|True|256|2" \
    "r5_a2_cpt_tulu|CPT|$CPT_DATA|True|42|3"

# Batch 2: A2 CPT+Tulu (2 seeds) + A3 KTO+Tulu (2 seeds)
echo "--- Batch 2 ---"
run_batch \
    "r5_a2_cpt_tulu|CPT|$CPT_DATA|True|137|0" \
    "r5_a2_cpt_tulu|CPT|$CPT_DATA|True|256|1" \
    "r5_a3_kto_tulu|KTO|$KTO_DATA|True|42|2" \
    "r5_a3_kto_tulu|KTO|$KTO_DATA|True|137|3"

# Batch 3: A3 KTO (1 seed) + A4 SFT (3 seeds)
echo "--- Batch 3 ---"
run_batch \
    "r5_a3_kto_tulu|KTO|$KTO_DATA|True|256|0" \
    "r5_a4_sft_tulu|SFT|$SFT_DATA|True|42|1" \
    "r5_a4_sft_tulu|SFT|$SFT_DATA|True|137|2" \
    "r5_a4_sft_tulu|SFT|$SFT_DATA|True|256|3"

# Batch 4: B Tulu only (3 seeds) + C1 DPO only (1 seed)
echo "--- Batch 4 ---"
run_batch \
    "r5_b_tulu_only|None|None|True|42|0" \
    "r5_b_tulu_only|None|None|True|137|1" \
    "r5_b_tulu_only|None|None|True|256|2" \
    "r5_c1_dpo_only|DPO|$DPO_DATA|False|42|3"

# Batch 5: C2 CPT only (1 seed) — D is just base model (no training)
echo "--- Batch 5 ---"
run_batch \
    "r5_c2_cpt_only|CPT|$CPT_DATA|False|42|0"

echo "=== Round 5 v2: All training complete ==="

# Show results
echo "Models:"
for d in $R5/models/*/; do
    name=$(basename $d)
    if [ -f "$d/final_model_path.txt" ]; then
        echo "  OK: $name"
    else
        echo "  FAIL: $name"
    fi
done
