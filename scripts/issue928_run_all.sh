#!/usr/bin/env bash
# Issue #928 linear pipeline driver (plan §4.1): Phase 0 gate → G generation →
# P parse → B capture → U upload (issue928_extract_thinking_store.py, all in
# one process), then Phase F fits + nulls + bootstrap (issue928_fit_
# decomposition.py, EPM_FIT_DEVICE=cuda), then figures. Smoke = the SAME
# driver with CONTEXTS=5 (unification default — the subset threads through
# every phase: extract reads --contexts, fits/figures enumerate the produced
# store manifest, upload commits what exists).
#
#   full run  : REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue928_run_all.sh
#   pod smoke : CONTEXTS=5 REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue928_run_all.sh
#
# Terminal [phase=done] is printed HERE (the component scripts print
# extract_done / fits_done / figures_done so the poller never sees a
# premature done). Non-zero rc from any phase aborts (set -e) — the gate-fail
# path exits 3 with a failure sentinel written by the extract script.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
# GCE lane exports tokens via its startup script and has NO .env — source
# conditionally, never unconditionally inside a classified chain (gotchas.md #923).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

CONTEXTS="${CONTEXTS:-}"            # empty = all 50 battery contexts
N_PERMS="${N_PERMS:-1000}"          # plan §6: 1,000 selection-symmetric draws
N_BOOT="${N_BOOT:-2000}"            # plan §6: 2,000 paired bootstrap draws
EXTRA_EXTRACT="${EXTRA_EXTRACT:-}"  # e.g. "--rung greedy" / "--skip-gen"
EXTRA_FIT="${EXTRA_FIT:-}"          # e.g. "--layers 0 1 --no-mlp" (smoke)
UPLOAD_PREFIX="${UPLOAD_PREFIX:-issue928_cot_decomposition/fit_results}"

CTX_FLAG=""
if [ -n "$CONTEXTS" ]; then CTX_FLAG="--contexts $CONTEXTS"; fi

echo "[phase=extract]"
# shellcheck disable=SC2086
uv run python scripts/issue928_extract_thinking_store.py --gpu $CTX_FLAG $EXTRA_EXTRACT

echo "[phase=fits]"
# shellcheck disable=SC2086
EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cuda}" uv run python scripts/issue928_fit_decomposition.py \
  --store "$REPO_ROOT/data/issue_928/store" \
  --out "$REPO_ROOT/eval_results/issue_928" \
  --n-perms "$N_PERMS" --n-boot "$N_BOOT" \
  --upload-prefix "$UPLOAD_PREFIX" $EXTRA_FIT

echo "[phase=figures]"
uv run python scripts/issue928_figures.py \
  --results "$REPO_ROOT/eval_results/issue_928" \
  --store "$REPO_ROOT/data/issue_928/store" \
  --rollouts "$REPO_ROOT/data/issue_928/raw_completions/thinking_rollouts" \
  --out "$REPO_ROOT/figures/issue_928"

echo "[phase=done]"
