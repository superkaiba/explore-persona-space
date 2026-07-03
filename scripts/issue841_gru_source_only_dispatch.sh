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
#   upload  : persist ALL artifacts to HF (skipped in smoke; FAIL-LOUD, aborts pre-sentinel):
#             the 2 GRU state-dicts + gru_source_only_projections.npz (LFS) → PRIVATE overflow
#             issue841_gru_source_only/{,results/} + load-bearing OVERFLOW_POINTER.json breadcrumbs;
#             the stage JSONs → canonical issue841_gru_source_only/results/ and plot PNGs →
#             canonical issue841_gru_source_only/figures/ (non-LFS); verify-after-upload lists both
#             repos + asserts every expected filename + pointer landed (the boot disk is DELETEd on exit)
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
# FAIL-LOUD (no `||` swallow): the per-unit scatter is a hard requirement (Lens-11
# companion). Plots run AFTER the result JSONs are checkpointed, so a loud abort here
# loses nothing but prevents a "successful" run whose sentinel fires without the
# required per-unit figure (which the analyzer would otherwise build the body without).
log_phase plots "start"
uv run python scripts/issue841_gru_source_only_plots.py --out-dir "$OUT_DIR" --fig-dir "$FIG_DIR"

# ── upload state-dicts (PRIVATE overflow repo, FAIL-LOUD; skip in smoke) ───────────
# The account-wide HF public-LFS quota 403 is live, so the >10 MB state-dicts (LFS)
# route to the PRIVATE overflow repo (headroom) under issue841_gru_source_only/, with a
# non-LFS OVERFLOW_POINTER.json breadcrumb committed to the canonical dataset repo (the
# non-LFS git-blob path succeeds over the public quota). FAIL-LOUD: no `||` swallow — under
# `set -e` a failed upload aborts the run BEFORE the sentinel + [phase=done], so a silent
# artifact loss cannot happen (the result JSONs are separately git-committed).
UPLOAD_STATUS="skipped-smoke"
if [ "${EPM_I841GSO_SMOKE:-0}" != "1" ]; then
  log_phase upload "start (state-dicts → PRIVATE overflow issue841_gru_source_only/ + canonical pointer)"
  uv run python -c "
import io, json, os, sys, time
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi, list_repo_files
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, DEFAULT_OVERFLOW_REPO
out_dir, sub = sys.argv[1], 'issue841_gru_source_only'
api = HfApi(token=os.environ.get('HF_TOKEN'))
# 1. LFS state-dicts -> PRIVATE overflow DATASET repo (public-LFS quota 403 is live; private has
#    headroom). DATASET repo_type (NOT model) so the committed dataset-to-dataset pointer reader
#    (scaling_common._overflow_repo_for_bucket / hf_download_pt_maybe_overflow) can resolve the .pt.
api.create_repo(DEFAULT_OVERFLOW_REPO, repo_type='dataset', private=True, exist_ok=True)
api.upload_folder(folder_path=out_dir, path_in_repo=sub, repo_id=DEFAULT_OVERFLOW_REPO,
                  repo_type='dataset', allow_patterns=['*.pt'])
present = sorted(f for f in list_repo_files(DEFAULT_OVERFLOW_REPO, repo_type='dataset')
                 if f.startswith(sub + '/') and f.endswith('.pt'))
assert len(present) >= 2, f'overflow state-dict upload incomplete (expected >=2 .pt): {present}'
# 2. non-LFS pointer breadcrumb -> CANONICAL dataset repo (small JSON rides the non-LFS path,
#    which succeeds over the public-storage quota) so a consumer/verifier finds the real location.
pointer = {'overflow_repo': DEFAULT_OVERFLOW_REPO, 'overflow_repo_type': 'dataset',
           'path_in_repo': sub, 'files': present, 'ts': time.time(),
           'reason': 'GRU state-dicts (LFS >10MB) rerouted to the private overflow repo; '
                     'public-LFS quota 403 live'}
api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(pointer, indent=2).encode('utf-8')),
                repo_id=DEFAULT_DATASET_REPO, path_in_repo=sub + '/OVERFLOW_POINTER.json',
                repo_type='dataset')
print('[upload] state-dicts ->', DEFAULT_OVERFLOW_REPO, '(private, dataset)', present,
      '; pointer ->', DEFAULT_DATASET_REPO, sub + '/OVERFLOW_POINTER.json')
# 3. result JSONs (non-LFS) -> canonical public results/; the projections npz (>10 MB, LFS)
#    -> PRIVATE overflow results/ + a load-bearing OVERFLOW_POINTER.json for that prefix.
#    (att-7-class gap: the stage JSONs + npz + figures lived only on the boot disk and were
#    lost to the instance's clean-exit DELETE; only the .pt state-dicts had been persisted.)
res_sub = sub + '/results'
api.upload_folder(folder_path=out_dir, path_in_repo=res_sub, repo_id=DEFAULT_DATASET_REPO,
                  repo_type='dataset', allow_patterns=['*.json'])
api.upload_folder(folder_path=out_dir, path_in_repo=res_sub, repo_id=DEFAULT_OVERFLOW_REPO,
                  repo_type='dataset', allow_patterns=['*.npz'])
res_pointer = {'overflow_repo': DEFAULT_OVERFLOW_REPO, 'overflow_repo_type': 'dataset',
               'path_in_repo': res_sub, 'files': ['gru_source_only_projections.npz'],
               'ts': time.time(),
               'reason': 'projections npz (LFS >10MB) rerouted to the private overflow repo; '
                         'public-LFS quota 403 live'}
api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(res_pointer, indent=2).encode('utf-8')),
                repo_id=DEFAULT_DATASET_REPO, path_in_repo=res_sub + '/OVERFLOW_POINTER.json',
                repo_type='dataset')
# 4. figures (PNG/PDF, non-LFS) -> canonical public figures/.
fig_root = os.path.join(sys.argv[2], 'issue_841/gru_source_only')
assert os.path.isdir(fig_root), f'figures dir missing (plots produced no figures): {fig_root}'
api.upload_folder(folder_path=fig_root, path_in_repo=sub + '/figures', repo_id=DEFAULT_DATASET_REPO,
                  repo_type='dataset', allow_patterns=['*.png', '*.pdf'])
# 5. verify-after-upload against FRESH listings (list + assert expected filenames present).
canon_files = set(list_repo_files(DEFAULT_DATASET_REPO, repo_type='dataset'))
for name in ('stage0_gru_source_only.json', 'stage1_gru_source_only.json',
             'retention_gru_source_only.json', 'transport_fidelity_gru_source_only.json'):
    dest = res_sub + '/' + name
    assert dest in canon_files, f'result JSON missing on {DEFAULT_DATASET_REPO} after upload: {dest}'
ov_files = set(list_repo_files(DEFAULT_OVERFLOW_REPO, repo_type='dataset'))
assert res_sub + '/gru_source_only_projections.npz' in ov_files, \
    f'npz missing on overflow {DEFAULT_OVERFLOW_REPO} after upload'
# Both OVERFLOW_POINTER.json breadcrumbs are LOAD-BEARING for the overflow fetch path
# (scaling_common.hf_download_pt_maybe_overflow / _overflow_repo_for_bucket read them to
# locate the rerouted .pt/.npz on the private repo); a silently-missing pointer makes a
# fresh-instance durability fetch treat the bucket as public and return an empty set.
assert res_sub + '/OVERFLOW_POINTER.json' in canon_files, \
    f'results-npz OVERFLOW_POINTER.json missing on {DEFAULT_DATASET_REPO} (load-bearing for overflow fetch)'
assert sub + '/OVERFLOW_POINTER.json' in canon_files, \
    f'state-dict OVERFLOW_POINTER.json missing on {DEFAULT_DATASET_REPO} (load-bearing for overflow fetch)'
# EXACT expected figure set (gru plots write these fixed stems on a full run) — a partial
# figure upload must fail pre-sentinel, so assert every expected PNG landed (not just >=1).
expected_gru_figs = {
    'heroA_stage0_atlas.png', 'heroB_stage1_comparison_bars.png',
    'exp_delta_forest.png', 'exp_per_unit_scatter.png', 'exp_retention.png',
    'exp_transport_fidelity.png', 'exp_stage0_wincount.png',
}
uploaded_gru_figs = {
    f.split('/')[-1]
    for f in canon_files
    if f.startswith(sub + '/figures/') and f.endswith('.png')
}
missing_gru_figs = expected_gru_figs - uploaded_gru_figs
assert not missing_gru_figs, (
    f'expected figure PNGs missing on {DEFAULT_DATASET_REPO} under {sub}/figures/ '
    f'after upload (partial figure upload): {sorted(missing_gru_figs)}'
)
print('[upload] results ->', DEFAULT_DATASET_REPO, res_sub, '(JSONs) + npz ->',
      DEFAULT_OVERFLOW_REPO, '; figures ->', DEFAULT_DATASET_REPO, sub + '/figures/',
      '; n_figs=', len(uploaded_gru_figs), '; pointers verified')
" "$OUT_DIR" "$FIG_DIR"
  UPLOAD_STATUS="overflow-private"
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
      'state_dicts_hf': 'PRIVATE overflow superkaiba1/explore-persona-space-overflow '
                        '(dataset): issue841_gru_source_only/gru_source_only_{raw,rmsnorm}.pt; '
                        'pointer: <data-repo>/issue841_gru_source_only/OVERFLOW_POINTER.json',
      'results_hf': 'issue841_gru_source_only/results/ (stage JSONs canonical; '
                    'gru_source_only_projections.npz → private overflow + OVERFLOW_POINTER.json)',
      'figures_hf': 'issue841_gru_source_only/figures/ (plot PNGs, canonical)',
    },
    'state_dict_upload': os.environ['UPLOAD_STATUS'],
    'reproducibility_card': 'N/A — forward-pass-free source-only-GRU refit over cached #779 '
                            'tensors; no adapters, no WandB, no raw completions',
  }),
}, open(sys.argv[1], 'w'), indent=2)
" "$SENTINEL"

log_phase done "issue-841 gru-source-only pod phases complete (sentinel: $SENTINEL, upload=$UPLOAD_STATUS)"
