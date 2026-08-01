#!/usr/bin/env bash
# #1979 F1 dispatch wrapper — thin single-exec (plan v2 §10 exact workload command):
#   uv run python scripts/dispatch_issue.py --issue 1979 --gpus 8 --min-gpu-mem-gb 60 \
#     --boot-disk-gb 250 --workload-cmd \
#     'bash scripts/issue1979_dispatch.sh --phase f1 --out-root $WORKLOAD_ROOT/data/issue_1979/out'
# Self-resolving REPO_ROOT via the set-u-safe ${WORKLOAD_ROOT:-...} pattern; all
# orchestration (GPU fan-out, sentinels, resume, uploads) lives in issue1979_gpu.py.
set -euo pipefail

REPO_ROOT="${WORKLOAD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Conditional .env sourcing (#923): the GCE lane exports tokens via its startup
# script and ships NO .env file; pods/VM have one at the repo root.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

PHASE="f1"
OUT_ROOT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    *) echo "issue1979_dispatch.sh: unknown arg: $1" >&2; exit 2 ;;
  esac
done
: "${OUT_ROOT:?--out-root is required}"

exec uv run python scripts/issue1979_gpu.py --phase "$PHASE" --out-root "$OUT_ROOT"
