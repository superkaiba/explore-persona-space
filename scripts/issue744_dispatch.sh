#!/usr/bin/env bash
# Issue #744 — single end-to-end dispatcher: build -> dump -> analyze -> figures.
#
# Phase ordering matches the plan §4.0 DAG + the CLAUDE.md "CPU-only phases
# don't hold GPU pods" rule:
#
#   Phase 0  build_corpora.py     (CPU)  -> corpus JSONs
#   Phase 1  dump_and_stream.py   (GPU)  -> NS raw dump + broader summaries + masks
#            (uploads ALL Phase-1 artifacts to HF + verifies; the GPU pod is
#             RELEASED here by the orchestrator BEFORE the CPU analysis phase)
#   Phase 2  analyze_continuity.py (CPU) -> per-layer curves + bootstrap CIs + H3
#   Phase 3  make_figures.py      (CPU)  -> hero figures + over-produce dump
#
# The GPU-bound phase is ONLY Phase 1. Phases 2/3 are CPU and run off-pod on the
# VM against the uploaded artifacts (the dispatcher logs the release point). On a
# GPU pod, run Phase 0+1 here, then the orchestrator releases the pod and runs
# `--phases analyze,figures` off-pod. For a single-box smoke (`--smoke`) all
# phases run in one invocation.
#
# This is the SAME dispatcher for smoke and sweep — `--smoke` flows through to
# every phase (PASS_UNIFIED: smoke = sweep with tiny N; plan §4.6). Per-cell /
# per-arm parameterization is the `--model` + `--out-suffix` pair; each phase
# reads the same `--cells`-equivalent (the model/arm) so the smoke exercises the
# identical dispatch shape.
#
# Usage:
#   bash scripts/issue744_dispatch.sh --model Qwen/Qwen2.5-7B --gpu-id 0
#   bash scripts/issue744_dispatch.sh --smoke --device cpu --model Qwen/Qwen2.5-0.5B \
#       --expected-layers 24 --expected-hidden 896 --no-upload \
#       --root /tmp/issue744_smoke --phases build,dump,analyze,figures
set -euo pipefail

# load_dotenv at entry: this dispatcher spawns subprocesses that need HF_TOKEN /
# WANDB_API_KEY; `uv run python` does NOT auto-load .env, so source it explicitly
# (canonical set -a recipe; never a bare load_dotenv() in a heredoc).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .env ]]; then set -a && source .env && set +a; fi

MODEL="Qwen/Qwen2.5-7B"
GPU_ID=0
DEVICE="auto"
EXPECTED_LAYERS=28
EXPECTED_HIDDEN=3584
SMOKE=""
NO_UPLOAD=""
ROOT="data/issue_744"
OUT_SUFFIX="base"          # arm sub-dir (base / instruct)
PHASES="build,dump,analyze,figures"
BROADER_RAW_KEEP=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --gpu-id) GPU_ID="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --expected-layers) EXPECTED_LAYERS="$2"; shift 2;;
    --expected-hidden) EXPECTED_HIDDEN="$2"; shift 2;;
    --smoke) SMOKE="--smoke"; shift;;
    --no-upload) NO_UPLOAD="--no-upload"; shift;;
    --root) ROOT="$2"; shift 2;;
    --out-suffix) OUT_SUFFIX="$2"; shift 2;;
    --phases) PHASES="$2"; shift 2;;
    --broader-raw-keep) BROADER_RAW_KEEP="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

CORPORA_DIR="${ROOT}/corpora"
DUMP_DIR="${ROOT}/${OUT_SUFFIX}"
# eval_results stays a top-level git path (not under data/), so route analysis
# output there for the sweep; for smoke, keep it under the smoke root.
if [[ -n "$SMOKE" ]]; then
  EVAL_DIR="${ROOT}/eval_${OUT_SUFFIX}"
  FIG_DIR="${ROOT}/figures_${OUT_SUFFIX}"
else
  EVAL_DIR="eval_results/issue_744/${OUT_SUFFIX}"
  FIG_DIR="figures/issue_744/${OUT_SUFFIX}"
fi

has_phase() { [[ ",${PHASES}," == *",$1,"* ]]; }

echo "[issue744-dispatch] model=${MODEL} arm=${OUT_SUFFIX} phases=${PHASES} smoke=${SMOKE:-no}"

if has_phase build; then
  echo "[issue744-dispatch] Phase 0: build corpora (CPU)"
  uv run python scripts/issue744_build_corpora.py \
    --out-dir "$CORPORA_DIR" --model "$MODEL" $SMOKE
fi

if has_phase dump; then
  echo "[issue744-dispatch] Phase 1: dump + stream (GPU)"
  uv run python scripts/issue744_dump_and_stream.py \
    --corpora-dir "$CORPORA_DIR" --out-dir "$DUMP_DIR" --model "$MODEL" \
    --gpu-id "$GPU_ID" --device "$DEVICE" \
    --expected-layers "$EXPECTED_LAYERS" --expected-hidden "$EXPECTED_HIDDEN" \
    --broader-raw-keep "$BROADER_RAW_KEEP" $SMOKE $NO_UPLOAD
  echo "[issue744-dispatch] GPU-RELEASE-POINT: Phase 1 artifacts uploaded + verified."
  echo "[issue744-dispatch] The orchestrator releases the GPU pod here; Phases 2/3 are CPU (off-pod)."
fi

if has_phase analyze; then
  echo "[issue744-dispatch] Phase 2: analyze continuity (CPU)"
  uv run python scripts/issue744_analyze_continuity.py \
    --dump-dir "$DUMP_DIR" --out-dir "$EVAL_DIR"
fi

if has_phase figures; then
  echo "[issue744-dispatch] Phase 3: figures (CPU)"
  uv run python scripts/issue744_make_figures.py \
    --analysis-dir "$EVAL_DIR" --fig-dir "$FIG_DIR"
fi

echo "[issue744-dispatch] done (arm=${OUT_SUFFIX})."
