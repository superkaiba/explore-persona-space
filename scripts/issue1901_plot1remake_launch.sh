#!/usr/bin/env bash
# Pod-side detached launcher for the #1901 Plot 1 remake (pod-1901-plot1remake).
#
# Launch (on the pod, repo already synced by bootstrap):
#   cd /workspace/explore-persona-space && \
#   setsid nohup bash scripts/issue1901_plot1remake_launch.sh \
#     > /workspace/logs/issue-1901-plot1remake.log 2>&1 < /dev/null &
#
# Writes: pid file, [phase=...] breadcrumbs (from the python driver), and the
# results sentinel at /workspace/logs/issue-1901-plot1remake-results.json (the
# driver writes it on success; this wrapper writes a failure sentinel if the
# driver dies without one — the poll_pipeline.py contract).
set -uo pipefail
cd /workspace/explore-persona-space || exit 1
set -a
[ -f .env ] && . ./.env
set +a
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p /workspace/logs
SENTINEL=/workspace/logs/issue-1901-plot1remake-results.json
echo $$ > /workspace/logs/issue-1901-plot1remake.pid

uv run python scripts/issue1901_plot1_remake.py \
  --phase all \
  --stage-root /workspace/plot1remake_stage \
  --device cuda \
  --sentinel "$SENTINEL"
rc=$?
echo "[launcher] python exited rc=$rc"
if [ ! -f "$SENTINEL" ]; then
  printf '{"ok": false, "rc": %d, "phase": "all", "note": "launcher backstop: driver exited without writing the sentinel"}\n' "$rc" > "$SENTINEL"
fi
echo "[launcher] done rc=$rc"
exit "$rc"
