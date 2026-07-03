#!/usr/bin/env bash
# Issue #841 scaling-capture pod-side driver — the 3 python phases + plots.
#
# UNIFIED smoke = sweep: the SAME driver runs the tiny smoke and the full run;
# smoke is this script with EPM_I841S_SMOKE=1, which threads --smoke into every
# phase (tiny n / 1 trait / small n-boot) through the SAME dispatcher, loaders,
# and [phase=...] logging. No divergent smoke path.
#
# Phases (each emits a [phase=<name>] line the poller parses; the SINGLE terminal
# [phase=done] fires only on graceful completion):
#   capture : issue841_scaling_capture.py  (the ONE GPU phase; KILL-A spot-gate,
#             then the ~96k batched forward, shard + HF upload)
#   stage0  : issue841_scaling_stage0.py   (nested-n ridge/MLP refit, KILL-B,
#             position-drift, per-n ridge maps → HF; stage0_scaling.json)
#   stage1  : issue841_scaling_stage1.py   (transport-scaling curve; win_count(n),
#             paired-delta(n), retention(n), BH; stage1_scaling.json + npz)
#   plots   : issue841_scaling_plots.py    (DV1 R²(n) + DV2 transport-scaling heroes)
#
# REPO_ROOT resolves via ${REPO_ROOT:-...}; the GCE startup script exports
# REPO_ROOT=$WORKLOAD_ROOT before the workload (#641), and RunPod runs from the
# clone dir, so the default only matters for a bare local invocation.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

ISSUE=841
LOGS_DIR="${EPM_LOGS_DIR:-/workspace/logs}"
mkdir -p "$LOGS_DIR"

# Load credentials at entry (uv run does NOT auto-load .env). Source into the
# shell env so every uv subprocess inherits (sh-safe; no heredoc load_dotenv).
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi

# ── smoke knobs (unified smoke = sweep) ─────────────────────────────────────────
SMOKE_CAP=""
SMOKE_S0=""
SMOKE_S1=""
if [ "${EPM_I841S_SMOKE:-0}" = "1" ]; then
  SMOKE_CAP="--smoke"
  # tiny synthetic bundle so stage0 needs no 6 GB parent bundle at smoke scale
  SMOKE_S0="--smoke --synthetic-parent 160 --synthetic-new 400 --synthetic-hidden 96 \
    --anchor-n 120 --ns 120,200,400 --dual-max 150 --xcheck-n 200 --transitions 17,18 \
    --mlp-ns 120,400 --mlp-epochs 3 --no-upload"
  SMOKE_S1="--smoke --synthetic --synthetic-hidden 64 --ns 4000,10000,25000 --anchor-n 4000 \
    --n-boot 60"
fi

log_phase() { printf '[phase=%s] %s\n' "$1" "${2:-}"; }

# ── capture (GPU) ───────────────────────────────────────────────────────────────
log_phase capture "start (dtype=${EPM_I841S_CAPTURE_DTYPE:-fp32})"
# shellcheck disable=SC2086
uv run python scripts/issue841_scaling_capture.py \
  --capture-dtype "${EPM_I841S_CAPTURE_DTYPE:-fp32}" \
  ${EPM_I841S_CAPTURE_EXTRA:-} \
  $SMOKE_CAP

uv run python scripts/clean_experiment_downloads.py "$ISSUE" --incremental --apply || true

# ── stage0 refit ─────────────────────────────────────────────────────────────────
log_phase stage0 "start"
# shellcheck disable=SC2086
uv run python scripts/issue841_scaling_stage0.py $SMOKE_S0

# ── stage1 transport-scaling curve ───────────────────────────────────────────────
log_phase stage1 "start"
# shellcheck disable=SC2086
uv run python scripts/issue841_scaling_stage1.py $SMOKE_S1

# ── plots ────────────────────────────────────────────────────────────────────────
log_phase plots "start"
uv run python scripts/issue841_scaling_plots.py || true

# ── end-of-run sentinel + terminal phase line ────────────────────────────────────
SENTINEL="$LOGS_DIR/issue-841-scaling-$(date +%s).json"
uv run python -c "
import json, sys
json.dump({
  'sentinel_schema_version': 1,
  'kind': 'epm:results',
  'version': 1,
  'task_id': 841,
  'note': json.dumps({
    'followup_label': 'scaling-capture',
    'phases': ['capture', 'stage0', 'stage1', 'plots'],
    'artifacts': {
      'stage0': 'eval_results/issue_841/scaling-capture/stage0_scaling.json',
      'stage1': 'eval_results/issue_841/scaling-capture/stage1_scaling.json',
      'fidelity': 'eval_results/issue_841/scaling-capture/transport_fidelity_scaling.json',
      'capture_hf': 'issue841_scaling/cx_last_shards/',
      'ridge_maps_hf': 'issue841_scaling/ridge_maps_n*/',
    },
    'reproducibility_card': 'N/A — forward-pass capture + closed-form/MLP fits; no adapters, no WandB',
  }),
}, open(sys.argv[1], 'w'), indent=2)
" "$SENTINEL"

log_phase done "issue-841 scaling-capture pod phases complete (sentinel: $SENTINEL)"
