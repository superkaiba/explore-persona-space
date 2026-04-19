#!/bin/bash
# Run v3 on-policy sweep in controlled batches of 2 workers max.
# Workers use a file lock for merge serialization, so 2 can train in
# parallel but only one writes model shards to NFS at a time.
#
# Usage: nohup bash scripts/run_v3_onpolicy_batch.sh > eval_results/leakage_v3_onpolicy/batch_sweep.log 2>&1 &

cd /workspace/explore-persona-space

LOG_DIR="eval_results/leakage_v3_onpolicy"
SCRIPT="scripts/run_leakage_v3_onpolicy.py"

echo "[$(date +%H:%M:%S)] Starting batch sweep with max 2 parallel workers"
echo "Existing results:"
find "$LOG_DIR" -name "run_result.json" | sort
echo "---"

# Work items: (source, seed, gpu)
# 9 total items, run in batches of 2
declare -a WORK_ITEMS=(
    "software_engineer 42 0"
    "software_engineer 137 2"
    "software_engineer 256 0"
    "librarian 42 2"
    "librarian 137 0"
    "librarian 256 2"
    "villain 42 0"
    "villain 137 2"
    "villain 256 0"
)

run_worker() {
    local source=$1
    local seed=$2
    local gpu=$3
    local key="${source}_seed${seed}"
    local logfile="$LOG_DIR/batch_${key}.log"

    echo "[$(date +%H:%M:%S)] Launching $key on GPU $gpu"
    python3 "$SCRIPT" pilot --source "$source" --seed "$seed" --gpu "$gpu" \
        > "$logfile" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] DONE: $key (exit 0)"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $key (exit $rc) -- see $logfile"
    fi
    return $rc
}

# Process work items in pairs
i=0
total=${#WORK_ITEMS[@]}
failed=0

while [ $i -lt $total ]; do
    echo ""
    echo "[$(date +%H:%M:%S)] === Batch starting at item $i/$total ==="

    # Launch first worker
    IFS=' ' read -r src1 seed1 gpu1 <<< "${WORK_ITEMS[$i]}"
    run_worker "$src1" "$seed1" "$gpu1" &
    pid1=$!

    # Launch second worker if available
    pid2=""
    j=$((i + 1))
    if [ $j -lt $total ]; then
        IFS=' ' read -r src2 seed2 gpu2 <<< "${WORK_ITEMS[$j]}"
        run_worker "$src2" "$seed2" "$gpu2" &
        pid2=$!
    fi

    # Wait for both workers -- do NOT exit on failure
    wait $pid1 || failed=$((failed + 1))
    if [ -n "$pid2" ]; then
        wait $pid2 || failed=$((failed + 1))
    fi

    # Brief pause between batches to let NFS catch up
    sleep 5

    # Advance by 2
    i=$((i + 2))

    # Progress report
    done_count=$(find "$LOG_DIR" -name "run_result.json" | wc -l)
    echo "[$(date +%H:%M:%S)] Progress: $done_count/45 results complete, $failed failures so far"
done

echo ""
echo "========================================"
echo "Batch sweep complete at $(date)"
echo "Total results: $(find "$LOG_DIR" -name "run_result.json" | wc -l)/45"
echo "Failed workers: $failed"
echo "========================================"

# Print summary
python3 "$SCRIPT" summary --seeds 42,137,256 --sources software_engineer,librarian,villain
