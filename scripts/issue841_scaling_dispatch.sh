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

# Fragmentation belt for the GPU capture (crash-fix round 4): the token-budget batcher
# (issue841_scaling_common.TOKEN_BUDGET) bounds the per-batch allocation, and
# expandable_segments lets the allocator grow/shrink segments instead of fragmenting a
# fixed pool — together they keep the Qwen2 MLP forward under 80 GiB. `:-` so an
# explicit launch-time PYTORCH_CUDA_ALLOC_CONF override always wins.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
log_phase capture "start (dtype=${EPM_I841S_CAPTURE_DTYPE:-bf16})"
# shellcheck disable=SC2086
uv run python scripts/issue841_scaling_capture.py \
  --capture-dtype "${EPM_I841S_CAPTURE_DTYPE:-bf16}" \
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
# FAIL-LOUD (no `|| true` swallow): a plot failure must abort BEFORE the upload +
# sentinel, so the run never reports success with the required hero / per-unit figures
# missing (the analyzer would otherwise build the body without them). Plots run AFTER the
# result JSONs are checkpointed, so a loud abort here loses nothing already on disk.
log_phase plots "start"
uv run python scripts/issue841_scaling_plots.py

# ── upload results + figures (FAIL-LOUD, overflow-aware; skip in smoke) ────────────
# att-7 completed all phases cleanly but the small result artifacts (the 3 stage JSONs,
# scaling_projections.npz, the plot PNGs) lived only on the boot disk and were destroyed
# by the instance's clean-exit DELETE — the capture shards + per-n maps were already safe
# on the overflow repo. This phase persists them BEFORE the sentinel: the stage JSONs +
# figure PNGs ride the non-LFS canonical path (open over the public-LFS quota 403), and the
# >10 MB scaling_projections.npz routes to the PRIVATE overflow repo via the SAME split-LFS
# helper the ridge/MLP maps use (so a public-LFS quota 403 cannot kill the run), leaving a
# load-bearing OVERFLOW_POINTER.json on the canonical repo. FAIL-LOUD: no `||` swallow — under
# `set -e` a failed / unverified upload aborts BEFORE the sentinel + [phase=done], so a silent
# artifact loss cannot happen (the result JSONs are separately git-committed on the branch).
UPLOAD_STATUS="skipped-smoke"
if [ "${EPM_I841S_SMOKE:-0}" != "1" ]; then
  log_phase upload "start (results JSONs + figures → canonical issue841_scaling/{results,figures}/; npz → PRIVATE overflow)"
  uv run python -c "
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from pathlib import Path
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import list_repo_files
import issue841_scaling_common as S
# 1. results dir: 3 stage JSONs -> canonical public (non-LFS); scaling_projections.npz
#    (>10 MB, LFS) -> PRIVATE overflow + load-bearing OVERFLOW_POINTER.json on canonical.
res_dir = S.EVAL_SCALING_DIR
assert res_dir.is_dir(), f'results dir missing: {res_dir}'
res_dev = S.upload_split_lfs_to_overflow(res_dir, 'issue841_scaling/results', lfs_glob='*.npz')
# 2. figures: PNG/PDF/meta all ride the non-LFS canonical path (lfs_glob matches none).
fig_dir = Path('figures/issue_841/scaling-capture')
assert fig_dir.is_dir(), f'figures dir missing (plots produced no figures): {fig_dir}'
S.upload_split_lfs_to_overflow(fig_dir, 'issue841_scaling/figures', lfs_glob='*.npz')
# 3. verify-after-upload against FRESH listings (list + assert expected filenames present).
canon, ov = res_dev['canonical_repo'], res_dev['overflow_repo']
canon_files = set(list_repo_files(canon, repo_type='dataset'))
for name in ('stage0_scaling.json', 'stage1_scaling.json', 'transport_fidelity_scaling.json'):
    dest = f'issue841_scaling/results/{name}'
    assert dest in canon_files, f'result JSON missing on {canon} after upload: {dest}'
ov_files = set(list_repo_files(ov, repo_type='dataset'))
assert 'issue841_scaling/results/scaling_projections.npz' in ov_files, \
    f'npz missing on overflow {ov} after upload'
# EXACT expected figure set (the scaling plots write these fixed stems on a full run) — a
# partial figure upload must fail pre-sentinel, so assert every expected PNG landed (not just >=1).
expected_figs = {
    'hero_r2_scaling_r2_id.png', 'hero_r2_scaling_r2_meancentered.png',
    'hero_transport_scaling.png', 'exploratory_retention_scaling.png',
    'exploratory_position_drift.png',
}
uploaded_figs = {
    f.split('/')[-1]
    for f in canon_files
    if f.startswith('issue841_scaling/figures/') and f.endswith('.png')
}
missing_figs = expected_figs - uploaded_figs
assert not missing_figs, (
    f'expected figure PNGs missing on {canon} under issue841_scaling/figures/ '
    f'after upload (partial figure upload): {sorted(missing_figs)}'
)
print('[upload] scaling results + figures uploaded + verified:',
      'JSONs+figs ->', canon, '; npz ->', ov, '; n_figs=', len(uploaded_figs))
"
  UPLOAD_STATUS="uploaded"
fi

# ── end-of-run sentinel + terminal phase line ────────────────────────────────────
SENTINEL="$LOGS_DIR/issue-841-scaling-$(date +%s).json"
UPLOAD_STATUS="$UPLOAD_STATUS" uv run python -c "
import json, os, sys
json.dump({
  'sentinel_schema_version': 1,
  'kind': 'epm:results',
  'version': 1,
  'task_id': 841,
  'note': json.dumps({
    'followup_label': 'scaling-capture',
    'phases': ['capture', 'stage0', 'stage1', 'plots', 'upload'],
    'artifacts': {
      'stage0': 'eval_results/issue_841/scaling-capture/stage0_scaling.json',
      'stage1': 'eval_results/issue_841/scaling-capture/stage1_scaling.json',
      'fidelity': 'eval_results/issue_841/scaling-capture/transport_fidelity_scaling.json',
      'capture_hf': 'issue841_scaling/cx_last_shards/',
      'ridge_maps_hf': 'issue841_scaling/ridge_maps_n*/',
      'results_hf': 'issue841_scaling/results/ (stage JSONs canonical; scaling_projections.npz '
                    '→ private overflow + OVERFLOW_POINTER.json on canonical)',
      'figures_hf': 'issue841_scaling/figures/ (plot PNGs, canonical)',
    },
    'results_upload': os.environ.get('UPLOAD_STATUS', 'unknown'),
    'overflow_routing': {
      'overflow_repo': 'superkaiba1/explore-persona-space-overflow',
      'reason': 'public LFS quota 403 (#541/#552 LFS wall)',
      'lfs_prefixes': ['issue841_scaling/cx_last_shards/', 'issue841_scaling/ridge_maps_n*/', 'issue841_scaling/mlp_maps_n*/'],
      'note': 'LFS .pt shards + maps routed to the PRIVATE overflow repo; non-LFS (manifests, .done.json) + OVERFLOW_POINTER.json on the canonical public data repo',
    },
    'reproducibility_card': 'N/A — forward-pass capture + closed-form/MLP fits; no adapters, no WandB',
  }),
}, open(sys.argv[1], 'w'), indent=2)
" "$SENTINEL"

log_phase done "issue-841 scaling-capture pod phases complete (sentinel: $SENTINEL, upload=$UPLOAD_STATUS)"
