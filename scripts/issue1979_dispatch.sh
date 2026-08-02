#!/usr/bin/env bash
# #1979 F1 dispatch wrapper — smoke-then-full (plan v2 §10 workload command):
#   uv run python scripts/dispatch_issue.py --issue 1979 --gpus 8 --min-gpu-mem-gb 60 \
#     --boot-disk-gb 250 --workload-cmd \
#     'bash scripts/issue1979_dispatch.sh --phase f1 --out-root $WORKLOAD_ROOT/data/issue_1979/out'
# Self-resolving REPO_ROOT via the set-u-safe ${WORKLOAD_ROOT:-...} pattern; all
# orchestration (GPU fan-out, sentinels, resume, uploads) lives in issue1979_gpu.py.
#
# SMOKE-FIRST (crash-fix r4, the #408 one-bug-per-launch lesson): when
# SMOKE_FIRST=1 (the default) and the dispatch is the full --phase f1 run, the
# SAME driver first runs a tiny smoke leg — --panel-limit 1 --query-limit 2
# --smoke-subset (one unit per arm class; f1d span_mean L19 only) — into the
# SEPARATE out-root "${OUT_ROOT}-smoke" (per-leg out-roots; production
# sentinels/resume state never touched) with --skip-upload (production HF
# prefixes never touched). The smoke leg emits [phase=smoke_done], never the
# reserved [phase=done]. Under set -e a non-zero smoke rc aborts the wrapper,
# so the FULL leg execs ONLY on smoke rc=0. Opt out with SMOKE_FIRST=0.
set -euo pipefail

REPO_ROOT="${WORKLOAD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Conditional .env sourcing (#923): the GCE lane exports tokens via its startup
# script and ships NO .env file; pods/VM have one at the repo root.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

PHASE="f1"
OUT_ROOT=""
SMOKE_FIRST="${SMOKE_FIRST:-1}"
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    *) echo "issue1979_dispatch.sh: unknown arg: $1" >&2; exit 2 ;;
  esac
done
: "${OUT_ROOT:?--out-root is required}"

# Smoke-gate admits BOTH the parent full run (f1) and the plan-v6 f1g
# amendment leg — the inner --phase passthrough keeps smoke == sweep with the
# one-cell parameterization (consistency-checker v2 note, plan v6 §4 Diff 2).
if { [ "$PHASE" = "f1" ] || [ "$PHASE" = "f1g" ]; } && [ "$SMOKE_FIRST" = "1" ]; then
  SMOKE_ROOT="${OUT_ROOT}-smoke"
  echo "[smoke-first] smoke leg -> ${SMOKE_ROOT} (panel=1 queries=2, one unit per arm class)"
  uv run python scripts/issue1979_gpu.py --phase "$PHASE" --out-root "$SMOKE_ROOT" \
    --panel-limit 1 --query-limit 2 --smoke-subset --skip-upload
  echo "[smoke-first] smoke leg passed (rc=0) — starting the full leg"
fi

exec uv run python scripts/issue1979_gpu.py --phase "$PHASE" --out-root "$OUT_ROOT"
