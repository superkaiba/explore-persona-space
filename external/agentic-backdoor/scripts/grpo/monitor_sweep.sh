#!/bin/bash
# Monitor GRPO sweep jobs and report metrics
JOBS="$@"
for job in $JOBS; do
    name=$(sacct -j $job --format=JobName%60 --noheader 2>/dev/null | head -1 | xargs)
    state=$(squeue -j $job -o "%T" --noheader 2>/dev/null || sacct -j $job --format=State --noheader 2>/dev/null | head -1 | xargs)
    logfile="/workspace-vast/pbb/agentic-backdoor/logs/slurm-${job}.out"
    
    # Get latest step and val metrics
    last_step=$(grep "step:" "$logfile" 2>/dev/null | tail -1 | grep -oP 'step:\K[0-9]+' || echo "-")
    
    # Get all val evals
    val_line=""
    while IFS= read -r line; do
        s=$(echo "$line" | grep -oP 'step:\K[0-9]+')
        vp=$(echo "$line" | grep -oP 'val/avg_pass@1/intercode_alfa:\K[0-9.]+')
        vpk=$(echo "$line" | grep -oP 'val/test_score/pass@k/intercode_alfa:\K[0-9.]+')
        if [ -n "$vp" ]; then
            val_line="${val_line} s${s}=${vp}/${vpk}"
        fi
    done < <(grep "val/avg_pass@1" "$logfile" 2>/dev/null)
    
    echo "$job | $state | $name | step=$last_step | val:$val_line"
done
