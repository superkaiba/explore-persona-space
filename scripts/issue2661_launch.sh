#!/usr/bin/env bash
# Issue #2661 pod launcher — flat ctx SAE + full-dictionary feature map.
#
# VM-side (the experimenter runs these BY HAND; never auto-fired):
#   uv run python scripts/pod.py provision --issue 2661 --intent eval \
#       --container-disk-gb 200 --volume-gb 200        # 1x H100 — the provision
#       # output prints the hourly rate: RECORD IT in the dispatch ack
#   uv run python scripts/pod.py sync --issue 2661     # code -> pod
#   ssh <pod> 'cd /workspace/explore-persona-space && \
#       bash scripts/issue2661_launch.sh --full'       # this script, POD-side
#   uv run python scripts/pod.py watch --issue 2661    # VM-side stall watchdog
#
# Usage (pod-side): bash scripts/issue2661_launch.sh --full|--smoke
#   --full : composed smoke leg FIRST (same phase functions, tiny slice), then
#            the production `--phase all` chain.
#   --smoke: the composed smoke leg alone (CPU-runnable too).
#
# Exit-path spec (the #1482 launcher convention): set -euo pipefail COMPATIBLE —
# each leg captures rc explicitly and writes a poll_pipeline-conformant failed
# sentinel BEFORE exiting, so the poller drains a CLASSIFIED failure, never a
# silent death. Logs -> /workspace/eps_out/issue2661/logs/. The driver itself
# emits [phase=...] breadcrumbs and the single terminal [phase=done].
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

MODE="${1:---full}"
OUT_ROOT="${OUT_ROOT:-/workspace/eps-issue-2661}"
LOG_DIR="/workspace/eps_out/issue2661/logs"
if [ ! -d /workspace ]; then
  OUT_ROOT="$REPO_ROOT/eps_out/issue2661/work"   # VM/CI fallback (smoke only)
  LOG_DIR="$REPO_ROOT/eps_out/issue2661/logs"
fi
mkdir -p "$LOG_DIR"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

write_failed_sentinel() {
  # args: <phase> <rc> — poll_pipeline-conformant failure sentinel
  uv run python - "$1" "$2" <<'PY'
import json, sys, time
phase, rc = sys.argv[1], int(sys.argv[2])
logs = "/workspace/logs" if __import__("os").path.isdir("/workspace") else "logs"
__import__("os").makedirs(logs, exist_ok=True)
path = f"{logs}/issue-2661-launch-failed-{int(time.time())}.json"
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 2661,
    "by": "issue2661_launch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": f"issue2661 launch leg {phase} FAILED rc={rc}",
    "failure_class": "code",
    "phase": phase,
    "rc": rc,
    "blocks_pipeline": True,
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"[launcher] wrote failed sentinel {path}", flush=True)
PY
}

run_leg() {
  # args: <leg-name> <log-name> <driver args...>
  # The leg's stdout goes to its OWN log (the child's [phase=...] breadcrumbs
  # and terminal token live there) — the RESERVED [phase=done] in THIS
  # launcher's main log is the single echo at the bottom of the script
  # (pod-side-reporting.md; incidents #545/#920).
  local leg="$1" log="$2"
  shift 2
  local rc=0
  echo "[phase=issue2661-$leg]"
  echo "[launcher] leg=$leg -> $LOG_DIR/$log" >&2
  # CVD_PIN_EXEMPT: single-GPU pod, foreground sequential legs — CVD=0 pinned anyway
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2661_flat_ctx_sae.py --gpu-id 0 "$@" > "$LOG_DIR/$log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[launcher] leg=$leg FAILED rc=$rc — last 40 log lines:" >&2
    tail -40 "$LOG_DIR/$log" >&2 || true
    write_failed_sentinel "$leg" "$rc"
    exit "$rc"
  fi
}

case "$MODE" in
  --smoke)
    run_leg smoke smoke.log --phase smoke --out-root "$OUT_ROOT" --skip-upload
    ;;
  --full)
    run_leg smoke smoke.log --phase smoke --out-root "$OUT_ROOT" --skip-upload
    run_leg production production.log --phase all --out-root "$OUT_ROOT" --production
    ;;
  *)
    echo "usage: $0 --full|--smoke" >&2
    exit 2
    ;;
esac
echo "[launcher] all legs done" >&2
echo "[phase=done]"
