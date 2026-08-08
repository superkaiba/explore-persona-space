#!/usr/bin/env bash
# Issue #1482 matryoshka-tier pod launcher (plan v21 §4 item 3).
#
# Usage: bash scripts/issue1482_matryoshka_launch.sh --full|--smoke
#   --full : smoke leg FIRST (same driver, per-leg out-roots), then the
#            production leg. The full leg's first phase reaps the DERIVED
#            sibling smoke root (chained-legs gotcha; reap lives in the driver).
#   --smoke: the smoke leg alone.
#
# Pod phases are M0-M4 (pilot..upload2) only. The off-pod VM legs are
# `--phase evidence` (M5 mechanical evidence persistence) + `--phase analyze`
# (M6 mechanical): judged labelling is FROZEN per plan v21 §0.-1 (the driver
# exposes NO judge phase at all) — labels come later from #1773's validated
# instrument.
#
# Exit-path spec (plan §4): set -euo pipefail COMPATIBLE — a single linear
# sequence of `uv run python ... --phase <p>` calls, each with explicit rc
# capture (`|| rc=$?`, so set -e never short-circuits past the sentinel write)
# and a failed-sentinel-before-exit; no `false` in compound branches. The
# poller drains a CLASSIFIED failure, never a silent death.
#
# Pod-side contract: sentinels under /workspace/logs/issue-1482-*.json ONLY
# (never a shellout to the VM task-workflow CLI — pods run on issue branches);
# [phase=...] breadcrumbs on stdout; [phase=done] is the single terminal line.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi   # GCE lane has no .env (metadata env)

MODE="${1:---full}"
LOGS_DIR="/workspace/logs"
if [ ! -d /workspace ]; then LOGS_DIR="$REPO_ROOT/logs"; fi
mkdir -p "$LOGS_DIR"

write_failed_sentinel() {
  # args: <phase> <rc>  — poll_pipeline-conformant failure sentinel
  uv run python - "$1" "$2" "$LOGS_DIR" <<'PY'
import json, sys, time
phase, rc, logs_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
path = f"{logs_dir}/issue-1482-matryoshka-failed-{int(time.time())}.json"
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 1482,
    "by": "issue1482_matryoshka_launch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": f"matryoshka-tier phase {phase} FAILED rc={rc}",
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

# Optional extra driver flags (VM carve-out smoke passes "--tiny-model --device cpu";
# pod launches leave it EMPTY — the driver auto-detects GPUs). Deliberately unquoted
# word-split below (a flag LIST).
EXTRA_FLAGS="${EPM_MATRYOSHKA_EXTRA_FLAGS:-}"

run_leg() {
  # args: <leg-flag>  (--smoke | --full)
  local leg="$1"
  local phase rc
  for phase in pilot capture upload1 fits upload2; do
    echo "[phase=matryoshka-${phase}${leg}]"
    rc=0
    # shellcheck disable=SC2086 — EXTRA_FLAGS is a flag list
    uv run python scripts/issue1482_matryoshka_tier.py --phase "$phase" "$leg" $EXTRA_FLAGS || rc=$?
    if [ "$rc" -ne 0 ]; then
      write_failed_sentinel "${phase}${leg}" "$rc"
      exit "$rc"
    fi
  done
}

if [ "$MODE" = "--smoke" ]; then
  run_leg "--smoke"
elif [ "$MODE" = "--full" ]; then
  # smoke leg first (per-leg out-roots: the driver derives its own smoke root;
  # the full leg's pilot phase reaps that derived sibling root at entry)
  run_leg "--smoke"
  run_leg "--full"
else
  echo "usage: $0 --full|--smoke" >&2
  exit 2
fi

echo "[phase=done]"
