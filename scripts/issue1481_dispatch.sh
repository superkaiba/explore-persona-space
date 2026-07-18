#!/usr/bin/env bash
# #1481 — frozen Phase-A/C dispatch wrappers (plan §10 launch commands).
#
# Phase A (one GCE dispatch per group; the fu4 dispatcher fans cells out
# across every visible GPU — per-cell CVD pinning lives in the parent
# driver's _worker_cmd env):
#   uv run python scripts/dispatch_issue.py launch --issue 1481 --intent lora-7b \
#     --backend gcp --gpus 8 --repo-branch issue-1481 \
#     --workload-cmd "bash scripts/issue1481_dispatch.sh impolite"
#   (and: sycophancy | casual-s137; marker-a | marker-b once
#    scripts/issue1481_marker.py lands — deferred, see task concern
#    marker-dispatcher-missing)
#
# Phase 0 (VM, 0 GPU, BEFORE any Phase-A dispatch):
#   uv run python scripts/issue1481_worker.py --full --phase mixes
#   (derives + uploads the 8 impolite/sycophancy po mixes; commit the
#    manifest under eval_results/issue_1481/ before dispatch)
#
# Phase C (narrow, after Phase-B selection):
#   uv run python scripts/dispatch_issue.py launch --issue 1481 --intent eval \
#     --backend gcp --gpus 2 --repo-branch issue-1481 \
#     --workload-cmd "bash scripts/issue1481_dispatch.sh panel --round <round> \
#       --out-root <cohort_root> --arms <verdict run_ids> [--ckpt-map <json>]"
set -euo pipefail

GROUP="${1:?usage: issue1481_dispatch.sh <impolite|sycophancy|casual-s137|panel|base-arms> [args...]}"
shift || true

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
[ -d "$REPO_ROOT" ] || REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

mkdir -p /workspace/logs 2>/dev/null || true

case "$GROUP" in
  impolite | sycophancy | casual-s137)
    echo "[phase=i1481_dispatch_${GROUP//-/_}]"
    uv run python scripts/issue1481_worker.py --full --dispatch "$GROUP" --seeds 42,137 "$@"
    ;;
  panel)
    echo "[phase=i1481_panel_wrapper]"
    uv run python scripts/issue1481_worker.py --full --phase panel "$@"
    ;;
  base-arms)
    echo "[phase=i1481_base_arms_wrapper]"
    uv run python scripts/issue1481_worker.py --full --phase base-arms "$@"
    ;;
  marker-a | marker-b)
    echo "[i1481] marker dispatches need scripts/issue1481_marker.py (deferred — task" \
      "concern marker-dispatcher-missing); refusing" >&2
    exit 3
    ;;
  *)
    echo "unknown dispatch group: $GROUP" >&2
    exit 2
    ;;
esac

echo "[phase=done]"
