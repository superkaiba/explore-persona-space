#!/bin/bash
# Run ALL 15 evaluations (5 sources x 3 seeds) sequentially on one GPU.
# Robust: each run is independent, failures don't block subsequent runs.
# Usage: bash scripts/run_taxonomy_all.sh <GPU_ID>

set -uo pipefail

GPU=${1:?Usage: $0 <GPU_ID>}
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/explore-persona-space

SOURCES="villain comedian software_engineer assistant kindergarten_teacher"
SEEDS="42 137 256"

echo "=== Taxonomy ALL: GPU=$GPU ==="
echo "Starting at $(date)"
echo ""

completed=0
failed=0

for seed in $SEEDS; do
    for src in $SOURCES; do
        # Skip if already done
        result="/workspace/explore-persona-space/eval_results/persona_taxonomy/${src}_seed${seed}/marker_eval.json"
        if [ -f "$result" ]; then
            echo ">>> SKIP source=$src seed=$seed (already done)"
            completed=$((completed + 1))
            continue
        fi

        echo ""
        echo ">>> Running source=$src seed=$seed gpu=$GPU at $(date)"
        if /root/.local/bin/uv run python scripts/run_taxonomy_leakage.py \
            --source "$src" --gpu "$GPU" --seed "$seed"; then
            echo ">>> DONE source=$src seed=$seed at $(date)"
            completed=$((completed + 1))
        else
            echo ">>> FAILED source=$src seed=$seed at $(date)"
            failed=$((failed + 1))
        fi

        # Force cleanup between runs
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 5
    done
done

echo ""
echo "=== ALL COMPLETE: $completed succeeded, $failed failed ==="
echo "Finished at $(date)"
