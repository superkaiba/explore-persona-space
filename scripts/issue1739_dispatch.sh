#!/usr/bin/env bash
# Issue #1739 dispatcher frame (round A).
#
# Phases: gates | extract | capture | judge | fits | figures
#   --phase <p>       run exactly one phase
#   --from-phase <p>  run <p> and every later phase
# Round A implements ONLY the `gates` phase; later phases exit 3 with a
# round-B/C note (and still write their sentinel so the poller sees them).
#
# Pod-side signaling is by SENTINEL FILE ONLY
# (${OUT_ROOT:-/workspace/logs}/issue-1739-<phase>.json) — NEVER a
# scripts/task.py shellout from pod-side code (hard project rule; the VM
# poller drains sentinels into markers).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

OUT_ROOT="${OUT_ROOT:-/workspace/logs}"
mkdir -p "$OUT_ROOT"

PHASES=(gates extract capture judge fits figures)

usage() {
  cat <<'EOF'
Usage: bash scripts/issue1739_dispatch.sh [--phase <p>] [--from-phase <p>]
Phases: gates extract capture judge fits figures
Round A: only `gates` is implemented; later phases exit 3 (round B/C).
Env: OUT_ROOT (sentinel dir; default /workspace/logs), REPO_ROOT.
EOF
}

# Shared-VM thread caps on VM-side python only (pods/GCE keep full width).
CAPS=()
if [ -d /mnt/eps-data ] || [ "$(hostname 2>/dev/null || true)" = "cia-benchmark-vm" ]; then
  CAPS=(env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2)
fi

write_sentinel() {
  # write_sentinel <phase> <status> <rc>
  local phase="$1" status="$2" rc="$3" ts commit
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf '{"issue": 1739, "phase": "%s", "status": "%s", "rc": %s, "ts": "%s", "git_commit": "%s"}\n' \
    "$phase" "$status" "$rc" "$ts" "$commit" > "${OUT_ROOT}/issue-1739-${phase}.json"
}

run_phase() {
  local phase="$1"
  echo "[phase=${phase}] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  case "$phase" in
    gates)
      "${CAPS[@]}" uv run python scripts/issue1739_gates.py --gate all
      write_sentinel "$phase" ok 0
      echo "[phase=${phase}] done"
      ;;
    extract|capture|judge|fits|figures)
      echo "[phase=${phase}] NOT IMPLEMENTED in round A (lands in round B/C)" >&2
      write_sentinel "$phase" not-implemented 3
      echo "[phase=${phase}] done (not-implemented)"
      return 3
      ;;
    *)
      echo "unknown phase: ${phase}" >&2
      return 2
      ;;
  esac
}

PHASE=""
FROM_PHASE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="${2:?--phase needs a value}"; shift 2 ;;
    --from-phase) FROM_PHASE="${2:?--from-phase needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

valid_phase() {
  local p
  for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done
  return 1
}

if [ -n "$PHASE" ] && [ -n "$FROM_PHASE" ]; then
  echo "--phase and --from-phase are mutually exclusive" >&2
  exit 2
fi
if [ -n "$PHASE" ]; then
  valid_phase "$PHASE" || { echo "unknown phase: $PHASE" >&2; exit 2; }
  run_phase "$PHASE"
  exit $?
fi

START="${FROM_PHASE:-gates}"
valid_phase "$START" || { echo "unknown phase: $START" >&2; exit 2; }
started=0
for p in "${PHASES[@]}"; do
  [ "$p" = "$START" ] && started=1
  [ "$started" = 1 ] || continue
  run_phase "$p"
done
