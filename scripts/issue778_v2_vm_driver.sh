#!/usr/bin/env bash
# Issue #778 v2 (faithful-extraction-honest-nulls-rerun) VM-side driver.
#
# Runs AFTER the pod releases (scripts/issue778_v2_dispatch.sh uploaded the
# analysis_tensors_v2 extract/neutral bundle and terminated). Sequences the
# zero-GPU phases on the VM (plan v8 §4 Components A5-A6, B, C):
#
#   prefetch   : stage + sha-pin the reused v1 inputs; fetch the pod's v2 bundle
#   judge      : Batch-API trait+coherence judging (deadline-bounded #663 client)
#                + paired mask + r_B v2 build (W1 gate on evil; K1 yield floor)
#   ladder     : v2 fixed stage (10,000 honest draws, lambda sweep, per-draw
#                fixed-layer columns) -> FWER min-p headline -> v2 maxlayer
#                stage (per-draw-own-max, W2 bit-exactness anchor)
#   figures    : v2 extra figures (v1-vs-v2 comparison, lambda strip, yield table)
#   upload     : VM-produced v2 artifacts (judge/pairing/rb_v2/maxdraws) + the
#                MANIFEST.json completion signal LAST (task #816 waits on it)
#
# Smoke = this script with tiny env knobs (same entrypoints, same order):
#   EPM_I778V2_TRAITS, EPM_I778V2_SETTINGS, EPM_I778V2_DRAWS,
#   EPM_I778V2_SKIP_UPLOAD=1, EPM_I778V2_SKIP_PREFETCH=1
#
# Launch detached (VM long-compute rule):
#   PHASE_PID=$(bash -c 'setsid nohup bash scripts/issue778_v2_vm_driver.sh \
#     < /dev/null >> logs/issue778_v2_vm_driver.log 2>&1 & echo $!')

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

OUT_ROOT="${EPM_I778V2_OUT_ROOT:-data/issue_778}"
EVAL_ROOT="${EPM_I778V2_EVAL_ROOT:-eval_results/issue_778}"
EXTERNAL_ROOT="${EPM_I778V2_EXTERNAL_ROOT:-external/persona_vectors}"
MAXDRAWS_ROOT="${EPM_I778V2_MAXDRAWS_ROOT:-$OUT_ROOT/v2/honest_nulls_maxdraws_v2}"
TRAITS="${EPM_I778V2_TRAITS:-evil sycophancy hallucination}"
SETTINGS="${EPM_I778V2_SETTINGS:-finetune monitoring_corrected monitoring_manyshot}"
DRAWS="${EPM_I778V2_DRAWS:-10000}"
DRAWS_ORIG="${EPM_I778V2_DRAWS_ORIG:-1000}"

# Credentials + shared-VM thread caps (load_dotenv in each python entrypoint
# setdefaults OMP/MKL caps; export here too for any stray subprocess).
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

log_phase() { printf '[phase=%s] %s\n' "$1" "${2:-}"; }

# ── prefetch: reused v1 inputs (sha-pinned) + the pod's v2 bundle ──────────────
if [ "${EPM_I778V2_SKIP_PREFETCH:-0}" != "1" ]; then
  log_phase prefetch "staging v1 inputs + v2 bundle"
  uv run python scripts/issue778_v2_prefetch.py \
    --out-root "$OUT_ROOT" --eval-results-root "$EVAL_ROOT" --fetch-v2
else
  log_phase prefetch "SKIPPED (EPM_I778V2_SKIP_PREFETCH=1)"
fi

# ── judge + paired mask + r_B v2 (Batch API; W1/K1 gates inside) ───────────────
log_phase judge_v2 "start traits=$TRAITS"
# shellcheck disable=SC2086
uv run python scripts/issue778_extract.py \
  --paired-mask --judge-harvest \
  --external-root "$EXTERNAL_ROOT" \
  --out-root "$OUT_ROOT" \
  --traits $TRAITS

# ── ladder v2: fixed (10k draws) -> FWER -> maxlayer (10k draws) ───────────────
log_phase honest_nulls_v2_fixed "start draws=$DRAWS"
# shellcheck disable=SC2086
uv run python scripts/issue778_honest_null_ladder.py \
  --rb-version v2 --stage fixed \
  --out-root "$OUT_ROOT" --eval-results-root "$EVAL_ROOT" \
  --maxdraws-root "$MAXDRAWS_ROOT" \
  --draws "$DRAWS" --draws-orig "$DRAWS_ORIG" \
  --traits $TRAITS --settings $SETTINGS

# NOTE: --draws is passed EXPLICITLY on every ladder stage (incl. fwer, where
# it is the REQUIRED per-column draw count the fail-loud verify checks against)
# — never rely on the script default. The driver NEVER passes
# --allow-gate-skip-smoke-only: production is fail-closed (W1/W2/FWER gates
# raise; the K1-N/A carve-out routes to the explicit headline-N/A artifact).
log_phase fwer "registered min-p headline"
# shellcheck disable=SC2086
uv run python scripts/issue778_honest_null_ladder.py \
  --rb-version v2 --stage fwer \
  --out-root "$OUT_ROOT" --eval-results-root "$EVAL_ROOT" \
  --maxdraws-root "$MAXDRAWS_ROOT" \
  --draws "$DRAWS" \
  --traits $TRAITS

log_phase honest_nulls_v2_maxlayer "start draws=$DRAWS"
# shellcheck disable=SC2086
uv run python scripts/issue778_honest_null_ladder.py \
  --rb-version v2 --stage maxlayer \
  --out-root "$OUT_ROOT" --eval-results-root "$EVAL_ROOT" \
  --maxdraws-root "$MAXDRAWS_ROOT" \
  --draws "$DRAWS" --draws-orig "$DRAWS_ORIG" \
  --traits $TRAITS --settings $SETTINGS

# ── extra figures (v1-vs-v2, lambda strip, pair-yield table) ───────────────────
log_phase figures "v2 extra figures"
# shellcheck disable=SC2086
uv run python scripts/issue778_v2_extra_figures.py \
  --out-root "$OUT_ROOT" --eval-results-root "$EVAL_ROOT" \
  --traits $TRAITS

# ── upload: VM-produced v2 artifacts + MANIFEST LAST (the #816 signal) ─────────
if [ "${EPM_I778V2_SKIP_UPLOAD:-0}" != "1" ]; then
  log_phase upload "VM v2 artifacts + MANIFEST"
  uv run python scripts/issue778_v2_upload.py --out-root "$OUT_ROOT" --phase vm | tail -1
else
  log_phase upload "SKIPPED (EPM_I778V2_SKIP_UPLOAD=1)"
fi

log_phase done "issue-778 v2rerun VM phases complete"
