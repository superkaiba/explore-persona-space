#!/bin/bash
# Multi-seed polling probe for issue #382 full run on pod-382.
# Sequentially checks all 3 seeds (42, 137, 256) on GPUs 0/1/2.
# Designed for orchestrator bg-Bash chain: sleep then poll once.
#
# Output: one JSON line per seed + one summary line. Orchestrator
# parses to decide whether to keep polling or trigger advancement.
set -uo pipefail

POD=${POD:-pod-382}

for SEED_GPU in "42:0" "137:1" "256:2"; do
  SEED=${SEED_GPU%:*}
  GPU=${SEED_GPU#*:}
  PIDFILE=/workspace/pids/issue-382-seed${SEED}-v2.pid
  LOG=/workspace/logs/issue-382-seed${SEED}-v2.log

  ssh "$POD" "
    PID=\$(cat $PIDFILE 2>/dev/null)
    if [ -z \"\$PID\" ] || ! ps -p \$PID >/dev/null 2>&1; then
      STATUS=dead
    else
      STATUS=alive
    fi
    ETIME=\$(ps -p \$PID -o etime= 2>/dev/null | xargs)
    STEP=\$(grep -oE '[0-9]+/6250' $LOG 2>/dev/null | tail -1 | cut -d/ -f1)
    [ -z \"\$STEP\" ] && STEP=0
    OOM=\$(grep -c 'OutOfMemoryError' $LOG 2>/dev/null)
    [ -z \"\$OOM\" ] && OOM=0
    TRACE=\$(grep -c 'Traceback' $LOG 2>/dev/null)
    [ -z \"\$TRACE\" ] && TRACE=0
    MTIME_AGO=\$(( \$(date +%s) - \$(stat -c %Y $LOG 2>/dev/null || echo 0) ))
    GPU_MEM=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
    [ -z \"\$GPU_MEM\" ] && GPU_MEM=0
    [ -z \"\$ETIME\" ] && ETIME=dead
    echo \"{\\\"seed\\\":$SEED,\\\"gpu\\\":$GPU,\\\"status\\\":\\\"\$STATUS\\\",\\\"etime\\\":\\\"\$ETIME\\\",\\\"step\\\":\$STEP,\\\"oom\\\":\$OOM,\\\"traceback\\\":\$TRACE,\\\"log_mtime_sec_ago\\\":\$MTIME_AGO,\\\"gpu_mem_mb\\\":\$GPU_MEM}\"
  "
done
