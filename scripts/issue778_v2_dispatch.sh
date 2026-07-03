#!/usr/bin/env bash
# Issue #778 v2 (faithful-extraction-honest-nulls-rerun) pod-side driver.
#
# UNIFIED smoke = sweep: the SAME script runs the tiny smoke and the full run;
# smoke is this script with EPM_I778V2_SMOKE=1, which threads tiny slices into
# EVERY phase (extract reads TRAITS/N_QUESTIONS/N_ROLLOUTS; neutral reads
# N_NEUTRAL; upload enumerates the staged data/issue_778/v2/ tree the same
# subset produced; the sentinel always runs). EPM_I778V2_GEN_STUB=1 additionally
# swaps vLLM/HF for deterministic stubs (the CPU dry-run of the dispatcher
# shape on the GPU-less VM). No divergent smoke path.
#
# Phases ([phase=<name>] lines; the SINGLE terminal [phase=done] fires only on
# graceful completion — poll_pipeline contract):
#   setup      : clone safety-research/persona_vectors @ b8e0f04 + unzip dataset.zip
#   extract_v2 : paired-mask generation (2,000 rollouts/trait) + rollout-text
#                persist + ALL-rollout response-avg acts capture (PRE-filter)
#   neutral    : 500 UltraChat prompts, no system prompt, response-avg +
#                last-prompt-token acts
#   upload     : ONE bulk upload_folder per subtree -> HF
#                issue778_persona_vectors/analysis_tensors_v2/{extract,neutral}
#                + EXACT-set fresh-listing verify (fail-loud)
#   sentinel   : epm:results sentinel (reproducibility_card with explicit
#                adapter/wandb N/A — no training this round)
#
# The judge phase (Batch API), paired mask, r_B v2 build, and the honest null
# ladder all run OFF-POD (scripts/issue778_v2_vm_driver.sh) after the pod
# releases — the pod holds NO GPU through the judge SLA (plan v8 §9).
#
# Env knobs:
#   EPM_I778V2_SMOKE=1       -> tiny slice (1 trait, 3 questions, 2 rollouts, 5 neutral)
#   EPM_I778V2_TRAITS        -> space-separated trait list
#   EPM_I778V2_N_QUESTIONS   -> extraction questions (default 20)
#   EPM_I778V2_N_ROLLOUTS    -> rollouts per (pair, question, arm) (default 10)
#   EPM_I778V2_N_NEUTRAL     -> neutral prompts (default 500)
#   EPM_I778V2_GEN_STUB=1    -> stub generation+capture (CPU dry-run; no GPU)
#   EPM_I778V2_SKIP_UPLOAD=1 -> skip the HF upload (local smoke)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

ISSUE=778
SLUG="persona_vectors"
PV_SHA="b8e0f044fe2410a6fad579f38324f03f13b4e917"
PV_REPO="https://github.com/safety-research/persona_vectors.git"
EXTERNAL_ROOT="${EPM_I778V2_EXTERNAL_ROOT:-external/persona_vectors}"
OUT_ROOT="${EPM_I778V2_OUT_ROOT:-data/issue_778}"
LOGS_DIR="${EPM_LOGS_DIR:-/workspace/logs}"
mkdir -p "$LOGS_DIR"

# Load credentials at entry (uv run does NOT auto-load .env).
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# ── slice knobs (unified smoke = sweep) ────────────────────────────────────────
TRAITS="${EPM_I778V2_TRAITS:-evil sycophancy hallucination}"
N_QUESTIONS="${EPM_I778V2_N_QUESTIONS:-20}"
N_ROLLOUTS="${EPM_I778V2_N_ROLLOUTS:-10}"
N_NEUTRAL="${EPM_I778V2_N_NEUTRAL:-500}"
if [ "${EPM_I778V2_SMOKE:-0}" = "1" ]; then
  TRAITS="${EPM_I778V2_TRAITS:-evil}"
  N_QUESTIONS="${EPM_I778V2_N_QUESTIONS:-3}"
  N_ROLLOUTS="${EPM_I778V2_N_ROLLOUTS:-2}"
  N_NEUTRAL="${EPM_I778V2_N_NEUTRAL:-5}"
fi
STUB_FLAGS=""
if [ "${EPM_I778V2_GEN_STUB:-0}" = "1" ]; then
  STUB_FLAGS="--gen-stub --capture-stub"
fi

log_phase() { printf '[phase=%s] %s\n' "$1" "${2:-}"; }

# ── setup: pinned external clone + dataset unzip ───────────────────────────────
log_phase setup "cloning $PV_REPO @ $PV_SHA"
if [ ! -d "$EXTERNAL_ROOT/.git" ]; then
  mkdir -p "$(dirname "$EXTERNAL_ROOT")"
  rm -rf "$EXTERNAL_ROOT"
  git clone "$PV_REPO" "$EXTERNAL_ROOT"
fi
git -C "$EXTERNAL_ROOT" fetch --depth 1 origin "$PV_SHA" || true
git -C "$EXTERNAL_ROOT" checkout "$PV_SHA"
GOT_SHA="$(git -C "$EXTERNAL_ROOT" rev-parse HEAD)"
if [ "$GOT_SHA" != "$PV_SHA" ]; then
  echo "FATAL: external persona_vectors HEAD $GOT_SHA != pinned $PV_SHA" >&2
  exit 1
fi
if [ -f "$EXTERNAL_ROOT/dataset.zip" ] && [ ! -d "$EXTERNAL_ROOT/dataset" ]; then
  log_phase setup "unzip dataset.zip"
  ( cd "$EXTERNAL_ROOT" && unzip -q -o dataset.zip )
fi
test -f "$EXTERNAL_ROOT/data_generation/trait_data_extract/evil.json" \
  || { echo "FATAL: trait extract JSON missing" >&2; exit 1; }
test -f "$EXTERNAL_ROOT/eval/prompts.py" \
  || { echo "FATAL: released eval/prompts.py (coherence rubric) missing" >&2; exit 1; }
log_phase setup "external inputs staged (sha=$GOT_SHA)"

# ── extract_v2: paired-mask generation + ALL-rollout acts (pre-filter) ─────────
log_phase extract_v2 "start traits=$TRAITS n_q=$N_QUESTIONS n_r=$N_ROLLOUTS"
# shellcheck disable=SC2086
uv run python scripts/issue778_extract.py \
  --paired-mask \
  --external-root "$EXTERNAL_ROOT" \
  --out-root "$OUT_ROOT" \
  --traits $TRAITS \
  --n-questions "$N_QUESTIONS" \
  --n-rollouts "$N_ROLLOUTS" \
  $STUB_FLAGS

# ── neutral: UltraChat capture (no system prompt) ──────────────────────────────
log_phase neutral "start n_prompts=$N_NEUTRAL"
# shellcheck disable=SC2086
uv run python scripts/issue778_neutral_capture.py \
  --out-root "$OUT_ROOT" \
  --n-prompts "$N_NEUTRAL" \
  $STUB_FLAGS

# ── upload: ONE bulk folder commit per subtree + exact-set verify ──────────────
UPLOAD_SUMMARY="{}"
if [ "${EPM_I778V2_SKIP_UPLOAD:-0}" != "1" ]; then
  log_phase upload "start (analysis_tensors_v2 pod bundle)"
  UPLOAD_SUMMARY="$(uv run python scripts/issue778_v2_upload.py --out-root "$OUT_ROOT" --phase pod | tail -1)"
else
  log_phase upload "SKIPPED (EPM_I778V2_SKIP_UPLOAD=1)"
fi

# ── sentinel (poll_pipeline contract; explicit no-training card) ───────────────
uv run python scripts/issue778_write_sentinel.py \
  --issue "$ISSUE" \
  --slug "$SLUG" \
  --round v2rerun \
  --upload-summary "$UPLOAD_SUMMARY" \
  --logs-dir "$LOGS_DIR"

log_phase done "issue-778 v2rerun pod phases complete (judge + mask + r_B v2 + ladder run off-pod on the VM)"
