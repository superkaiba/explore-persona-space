#!/bin/bash
# Issue #2676 side uploader for checkpoints kept by CKPT_UPLOAD=defer.
#
# Runs on the pod next to a live issue2676_run_pod.sh. For each of the 8
# checkpoints, in driver order, it waits for that arm's train done-marker,
# uploads the checkpoint with `issue2676_eval.py upload-ckpt` (default repo:
# the private overflow repo), then touches the upload done-marker. Once the
# marker exists, the driver's delete_local_ckpt may remove the local copy
# (uploaded first, deleted after: never the other way round).
#
# It writes its pid to /workspace/logs/issue-2676-ckptup.pid, so the pod
# launcher waits for it before writing the completion sentinel.
#
# usage: setsid nohup bash scripts/issue2676_ckpt_uploader.sh \
#            > /workspace/logs/issue-2676-ckptup.txt 2>&1 < /dev/null &
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
REPO=/workspace/explore-persona-space
cd "$REPO"
set -a; [ -f .env ] && . ./.env; set +a

DONE="$REPO/eval_results/issue_2676/done"
CK=/workspace/issue2676_ckpts
DRIVER_PID_FILE=/workspace/logs/issue-2676.pid
echo $$ > /workspace/logs/issue-2676-ckptup.pid

for pair in qwen05b:all qwen05b:top1 qwen05b:top6 qwen05b:top12 \
            qwen15b:all qwen15b:top1 qwen15b:top7 qwen15b:top14; do
    m="${pair%%:*}"
    arm="${pair##*:}"
    label="${m}_${arm}"
    if [ -f "$DONE/upload_ckpt_${label}.done" ]; then
        echo "[ckptup] skip $label (upload done)"
        continue
    fi
    while [ ! -f "$DONE/train_${label}.done" ]; do
        if ! kill -0 "$(cat "$DRIVER_PID_FILE" 2>/dev/null)" 2>/dev/null; then
            echo "[ckptup] driver exited before $label finished training; stopping" >&2
            exit 3
        fi
        sleep 60
    done
    echo "[ckptup] uploading $label"
    if ! uv run python scripts/issue2676_eval.py upload-ckpt \
            --model-dir "$CK/$label" --dest "issue2676/$m/$arm"; then
        echo "[ckptup] FAILED $label" >&2
        exit 1
    fi
    touch "$DONE/upload_ckpt_${label}.done"
    echo "[ckptup] done $label"
done
echo "[ckptup] all 8 checkpoints uploaded"
