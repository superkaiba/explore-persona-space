#!/usr/bin/env bash
# Issue #841 follow-up (gru-source-only) pod-side driver — Stage 0 + Stage 1 + plots.
#
# UNIFIED smoke = sweep: the SAME driver runs the tiny smoke and the full run; smoke is
# this script with EPM_I841GSO_SMOKE=1, which threads --smoke into every phase (200
# contexts / 1 trait / coarse grid / tiny epochs) through the SAME entrypoints, loaders,
# and [phase=...] logging. No divergent smoke path.
#
# Phases (each emits a [phase=<name>] line the poller parses; the SINGLE terminal
# [phase=done] fires only on graceful completion):
#   verify  : the --verify-source-only-gru gate (forward_single == GRUCell + apply finite)
#   stage0  : issue841_gru_source_only_stage0.py (fit 2 source-only GRUs, atlas, win-counts,
#             convergence diagnostics, save state-dicts)
#   stage1  : issue841_gru_source_only_stage1.py (memoryless transport, pairing-integrity
#             assert, paired bootstrap, 68-cell win-counts + chance + BH, retention, fidelity)
#   plots   : issue841_gru_source_only_plots.py
#   upload  : bulk-upload the 2 GRU state-dicts to HF issue841_gru_source_only/ (skipped in
#             smoke; tracked-gap on failure — the result JSONs are the durable artifact in git)
#
# Analysis-only, forward-pass-FREE (no Qwen weights, no new judging, NO raw completions):
# the upload_raw_completions_to_data_repo() helper is N/A for this round.
#
# REPO_ROOT resolves via ${REPO_ROOT:-...}; the GCE startup script exports REPO_ROOT=
# $WORKLOAD_ROOT before the workload (#641), and RunPod runs from the clone dir, so the
# default only matters for a bare local invocation.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

ISSUE=841
LOGS_DIR="${EPM_LOGS_DIR:-/workspace/logs}"
mkdir -p "$LOGS_DIR"

OUT_DIR="${EPM_I841GSO_OUT_DIR:-eval_results/issue_841/gru_source_only}"
DEVICE="${EPM_I841GSO_DEVICE:-auto}"
FIG_DIR="${EPM_I841GSO_FIG_DIR:-figures/}"

# Allocator defrag belt for the co-resident single-state GRU fit (harmless on CPU/L4).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Load credentials at entry (uv run does NOT auto-load .env). Source into the shell env so
# every uv subprocess inherits (sh-safe; no heredoc load_dotenv, gotchas.md).
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi

# ── smoke knobs (unified smoke = sweep) ─────────────────────────────────────────
SMOKE=""
if [ "${EPM_I841GSO_SMOKE:-0}" = "1" ]; then
  SMOKE="--smoke --gru-epochs ${EPM_I841GSO_EPOCHS:-2} --gru-batch-size 256"
fi

log_phase() { printf '[phase=%s] %s\n' "$1" "${2:-}"; }

# ── verify gate (dispatched-path self-check) ─────────────────────────────────────
log_phase verify "start"
# shellcheck disable=SC2086
uv run python scripts/issue841_gru_source_only_stage0.py --verify-source-only-gru --device "$DEVICE"

# ── stage0 atlas ─────────────────────────────────────────────────────────────────
log_phase stage0 "start (out_dir=$OUT_DIR)"
# shellcheck disable=SC2086
uv run python scripts/issue841_gru_source_only_stage0.py --device "$DEVICE" --out-dir "$OUT_DIR" $SMOKE

# ── stage1 transport benchmark ───────────────────────────────────────────────────
log_phase stage1 "start"
# shellcheck disable=SC2086
uv run python scripts/issue841_gru_source_only_stage1.py --device "$DEVICE" --out-dir "$OUT_DIR" \
  --n-boot "${EPM_I841GSO_NBOOT:-1000}" $SMOKE

# ── plots ────────────────────────────────────────────────────────────────────────
log_phase plots "start"
uv run python scripts/issue841_gru_source_only_plots.py --out-dir "$OUT_DIR" --fig-dir "$FIG_DIR" || true

# ── upload state-dicts (skip in smoke; tracked-gap on failure) ────────────────────
UPLOAD_STATUS="skipped-smoke"
if [ "${EPM_I841GSO_SMOKE:-0}" != "1" ]; then
  log_phase upload "start (state-dicts → issue841_gru_source_only/)"
  if uv run python -c "
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO
HfApi().upload_folder(
    folder_path=sys.argv[1], path_in_repo='issue841_gru_source_only',
    repo_id=DEFAULT_DATASET_REPO, repo_type='dataset', allow_patterns=['*.pt'],
)
print('[upload] state-dicts uploaded to', DEFAULT_DATASET_REPO, 'issue841_gru_source_only/')
" "$OUT_DIR"; then
    UPLOAD_STATUS="ok"
  else
    UPLOAD_STATUS="failed"
    log_phase upload "ERROR — state-dict HF upload FAILED (regenerable from seed+cached tensors; \
result JSONs are durable in git); recorded as a tracked gap in the sentinel"
  fi
fi

# ── end-of-run sentinel + terminal phase line ────────────────────────────────────
SENTINEL="$LOGS_DIR/issue-841-gru-source-only-$(date +%s).json"
UPLOAD_STATUS="$UPLOAD_STATUS" OUT_DIR="$OUT_DIR" uv run python -c "
import json, os, sys
json.dump({
  'sentinel_schema_version': 1,
  'kind': 'epm:results',
  'version': 1,
  'task_id': 841,
  'note': json.dumps({
    'followup_label': 'gru-source-only',
    'phases': ['verify', 'stage0', 'stage1', 'plots', 'upload'],
    'artifacts': {
      'stage0': os.environ['OUT_DIR'] + '/stage0_gru_source_only.json',
      'stage1': os.environ['OUT_DIR'] + '/stage1_gru_source_only.json',
      'retention': os.environ['OUT_DIR'] + '/retention_gru_source_only.json',
      'fidelity': os.environ['OUT_DIR'] + '/transport_fidelity_gru_source_only.json',
      'projections': os.environ['OUT_DIR'] + '/gru_source_only_projections.npz',
      'state_dicts_hf': 'issue841_gru_source_only/gru_source_only_{raw,rmsnorm}.pt',
    },
    'state_dict_upload': os.environ['UPLOAD_STATUS'],
    'reproducibility_card': 'N/A — forward-pass-free source-only-GRU refit over cached #779 '
                            'tensors; no adapters, no WandB, no raw completions',
  }),
}, open(sys.argv[1], 'w'), indent=2)
" "$SENTINEL"

log_phase done "issue-841 gru-source-only pod phases complete (sentinel: $SENTINEL, upload=$UPLOAD_STATUS)"
