#!/usr/bin/env bash
# Issue #778 pod-side driver — Persona Vectors replication + null battery.
#
# UNIFIED smoke = sweep: the SAME driver runs the 1-cell smoke and the full
# 24-cell sweep; smoke is just this script with EPM_I778_SMOKE=1 (or the
# --smoke/env knobs below) which threads --cells 1 / tiny slices into every
# phase. No divergent smoke path (smoke canary trait/cell = evil / II).
#
# Sequences (all phases emit [phase=<name>] structured JSON the poller parses;
# the SINGLE terminal [phase=done] line fires only on graceful completion):
#   step 0  setup      : clone safety-research/persona_vectors@b8e0f04 + unzip dataset.zip (POD-SIDE)
#   step 1  extract    : Phase 1 extraction + r_B per trait
#   step 2  monitoring : Phase 2 system-prompt prediction per trait
#   step 3  finetune   : Phase 3 24 rs-LoRA finetunes, 8-wide, 3 waves
#   step 4  capture    : Phase 3 activation capture (base + 24 FT) + trait-expression eval
#   step 5  upload     : LoRA adapters (HF model repo) + activation tensors + eval JSONs (HF data repo)
#   (null battery runs OFF-POD on the VM after the pod releases — plan v2 §9)
#
# The CPU null battery is NOT run here (it holds no GPU; plan §9 routes it off-pod).
# Env knobs (all optional; defaults = full production sweep):
#   EPM_I778_SMOKE=1        -> tiny slice (1 trait, few questions/rollouts, 1 FT cell, max-steps)
#   EPM_I778_TRAITS         -> space-separated trait list (default: evil sycophancy hallucination)
#   EPM_I778_N_QUESTIONS    -> extraction/eval questions (default 20)
#   EPM_I778_N_ROLLOUTS     -> rollouts per side/cell (default 10)
#   EPM_I778_FT_CELLS       -> limit finetune/capture to first N FT cells (default: all 24)
#   EPM_I778_SKIP_FINETUNE=1 / EPM_I778_SKIP_UPLOAD=1  -> phase skips (smoke/debug)
#
# REPO_ROOT resolves via ${REPO_ROOT:-...}; the GCE startup script exports
# REPO_ROOT=$WORKLOAD_ROOT before running the workload (#641), and RunPod runs
# from the clone dir, so the default only matters for a bare local invocation.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

ISSUE=778
SLUG="persona_vectors"
PV_SHA="b8e0f044fe2410a6fad579f38324f03f13b4e917"
PV_REPO="https://github.com/safety-research/persona_vectors.git"
EXTERNAL_ROOT="external/persona_vectors"
LOGS_DIR="${EPM_LOGS_DIR:-/workspace/logs}"
mkdir -p "$LOGS_DIR"

# Load credentials at entry (uv run does NOT auto-load .env). The canonical
# heredoc-safe idiom: source into the shell env so every uv subprocess inherits.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# ── slice knobs (unified smoke = sweep) ────────────────────────────────────────
TRAITS="${EPM_I778_TRAITS:-evil sycophancy hallucination}"
N_QUESTIONS="${EPM_I778_N_QUESTIONS:-20}"
N_ROLLOUTS="${EPM_I778_N_ROLLOUTS:-10}"
EXTRA_EXTRACT=""
EXTRA_MON=""
EXTRA_FT=""
EXTRA_CAP=""
if [ "${EPM_I778_SMOKE:-0}" = "1" ]; then
  # smoke canary = evil (highest-signal); tiny slice, 1 FT cell, capped steps.
  TRAITS="${EPM_I778_TRAITS:-evil}"
  N_QUESTIONS="${EPM_I778_N_QUESTIONS:-5}"
  N_ROLLOUTS="${EPM_I778_N_ROLLOUTS:-2}"
  EXTRA_EXTRACT="--cells 1 --smoke"
  EXTRA_MON="--cells 1 --n-prompts 2"
  EXTRA_FT="--cells 1 --max-steps 3"
  EXTRA_CAP="--cells 1"
fi
if [ -n "${EPM_I778_FT_CELLS:-}" ]; then
  EXTRA_FT="--cells ${EPM_I778_FT_CELLS} ${EXTRA_FT}"
  EXTRA_CAP="--cells ${EPM_I778_FT_CELLS} ${EXTRA_CAP}"
fi

log_phase() { printf '[phase=%s] %s\n' "$1" "${2:-}"; }

# ── step 0: pod-side external-repo clone + unzip (Must-Fix #3) ─────────────────
log_phase setup "cloning $PV_REPO @ $PV_SHA"
if [ ! -f "$EXTERNAL_ROOT/dataset.zip" ] && [ ! -d "$EXTERNAL_ROOT/dataset" ]; then
  mkdir -p external
  if [ ! -d "$EXTERNAL_ROOT/.git" ]; then
    rm -rf "$EXTERNAL_ROOT"
    git clone "$PV_REPO" "$EXTERNAL_ROOT"
  fi
  git -C "$EXTERNAL_ROOT" fetch --depth 1 origin "$PV_SHA"
  git -C "$EXTERNAL_ROOT" checkout "$PV_SHA"
fi
# Fail loud if the pinned SHA is not what we cloned.
GOT_SHA="$(git -C "$EXTERNAL_ROOT" rev-parse HEAD)"
if [ "$GOT_SHA" != "$PV_SHA" ]; then
  echo "FATAL: external persona_vectors HEAD $GOT_SHA != pinned $PV_SHA" >&2
  exit 1
fi
if [ ! -d "$EXTERNAL_ROOT/dataset" ]; then
  log_phase setup "unzip dataset.zip"
  ( cd "$EXTERNAL_ROOT" && unzip -q -o dataset.zip )
fi
# Pre-GPU input assert (plan §12 Assumption 14).
test -d "$EXTERNAL_ROOT/dataset" || { echo "FATAL: dataset/ missing after unzip" >&2; exit 1; }
test -f "$EXTERNAL_ROOT/data_generation/trait_data_extract/evil.json" \
  || { echo "FATAL: trait extract JSON missing" >&2; exit 1; }
log_phase setup "external inputs staged (sha=$GOT_SHA)"

# ── step 1: extraction ─────────────────────────────────────────────────────────
log_phase extract "start traits=$TRAITS"
# shellcheck disable=SC2086
uv run python scripts/issue778_extract.py \
  --external-root "$EXTERNAL_ROOT" \
  --out-root data/issue_778 \
  --traits $TRAITS \
  --n-questions "$N_QUESTIONS" \
  --n-rollouts "$N_ROLLOUTS" \
  $EXTRA_EXTRACT

# Between-phase cleanup to bound peak footprint (multi-phase contract).
uv run python scripts/clean_experiment_downloads.py "$ISSUE" --incremental --apply || true

# ── step 2: monitoring ─────────────────────────────────────────────────────────
log_phase monitoring "start"
# shellcheck disable=SC2086
uv run python scripts/issue778_monitoring.py \
  --external-root "$EXTERNAL_ROOT" \
  --out-root data/issue_778 \
  --eval-results-root eval_results/issue_778 \
  --traits $TRAITS \
  --n-questions "$N_QUESTIONS" \
  --n-rollouts "$N_ROLLOUTS" \
  $EXTRA_MON

# ── step 3: finetune (wave dispatcher; per-cell CVD pin inside the Python driver)
if [ "${EPM_I778_SKIP_FINETUNE:-0}" != "1" ]; then
  log_phase finetune "start"
  # shellcheck disable=SC2086
  uv run python scripts/issue778_finetune.py \
    --dataset-root "$EXTERNAL_ROOT/dataset" \
    --ckpt-root checkpoints/issue_778 \
    $EXTRA_FT
else
  log_phase finetune "SKIPPED (EPM_I778_SKIP_FINETUNE=1)"
fi

# ── step 4: activation capture + trait-expression eval ─────────────────────────
log_phase capture "start"
if [ "${EPM_I778_SKIP_FINETUNE:-0}" = "1" ]; then
  EXTRA_CAP="--base-only"
fi
# shellcheck disable=SC2086
uv run python scripts/issue778_capture.py \
  --external-root "$EXTERNAL_ROOT" \
  --out-root data/issue_778 \
  --eval-results-root eval_results/issue_778 \
  --ckpt-root checkpoints/issue_778 \
  --traits $TRAITS \
  --n-questions "$N_QUESTIONS" \
  --n-rollouts "$N_ROLLOUTS" \
  $EXTRA_CAP

# ── step 5: upload (LoRA adapters + analysis tensors + eval JSONs) ──────────────
UPLOAD_SUMMARY="{}"
if [ "${EPM_I778_SKIP_UPLOAD:-0}" != "1" ]; then
  log_phase upload "start"
  UPLOAD_SUMMARY="$(uv run python scripts/issue778_upload.py --issue "$ISSUE" --slug "$SLUG")"
else
  log_phase upload "SKIPPED (EPM_I778_SKIP_UPLOAD=1)"
fi

# ── end-of-run sentinel + terminal phase line ──────────────────────────────────
uv run python scripts/issue778_write_sentinel.py \
  --issue "$ISSUE" \
  --slug "$SLUG" \
  --upload-summary "$UPLOAD_SUMMARY" \
  --logs-dir "$LOGS_DIR"

log_phase done "issue-778 pod phases complete (null battery runs off-pod on the VM)"
