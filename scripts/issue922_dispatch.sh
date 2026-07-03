#!/usr/bin/env bash
# Issue #922 dispatcher — capture → fits → evals → plots → upload → sentinel.
#
# ONE code path for smoke and production (plan §4.5 PASS_UNIFIED): `--smoke`
# threads a single scaled-down subset (20 LMSYS contexts + 1 eval cell + 2 fit
# layers) through the SAME python entrypoints; every later phase enumerates
# its work from the artifacts the previous phase wrote (store shards / maps
# dir / eval JSONs), never from a registered full grid. No forks.
#
# Pod-side contract (poll_pipeline.py): [phase=<name>] breadcrumbs per phase;
# the results sentinel is written BEFORE the single terminal [phase=done].
#
# Env overrides (all optional):
#   EPM922_STORE / EPM922_MAPS / EPM922_OUT / EPM922_FIGS   — dirs
#   EPM922_MODEL / EPM922_TOKENIZER                          — model override
#     (VM stub-model smokes; tokenizer stays the real Qwen tokenizer)
#   EPM922_PARITY=assert|report|skip                         — parity probe mode
#   EPM922_SMOKE_BLOCKS (default "emb,20")                   — smoke fit layers
#   EPM922_READOUT_BLOCK                                     — stub-smoke ℓ* override
#   EPM922_EXPECTED_LAYERS / EPM922_EXPECTED_HIDDEN          — stub-smoke shape
#   EPM922_SKIP_UPLOAD=1                                     — skip the HF upload step
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ -f .env ]; then set -a; . ./.env; set +a; fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STORE="${EPM922_STORE:-/workspace/issue922_store}"
MAPS="${EPM922_MAPS:-/workspace/issue922_maps}"
OUT="${EPM922_OUT:-eval_results/issue_922}"
FIGS="${EPM922_FIGS:-figures/issue_922}"
MODEL="${EPM922_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER="${EPM922_TOKENIZER:-$MODEL}"
PARITY="${EPM922_PARITY:-assert}"
SMOKE_BLOCKS="${EPM922_SMOKE_BLOCKS:-emb,20}"
EXP_LAYERS="${EPM922_EXPECTED_LAYERS:-28}"
EXP_HIDDEN="${EPM922_EXPECTED_HIDDEN:-3584}"

# ONE subset definition, threaded to EVERY phase below. The v6 conditioned +
# direct arms scale down through the SAME flags (smoke: b1_ridge + b1_grad +
# b2_film at the smoke blocks, direct-c k<=8 — plan §4.5).
SMOKE_FLAG=()
FIT_BLOCKS=()
EVAL_TRAITS=()
COND_ARMS=(--b1-ridge
  --conditioned-forms "${EPM922_COND_FORMS:-b1_grad,film,lowrank,mixture}"
  --conditioned-blocks "${EPM922_COND_BLOCKS:-emb,5,10,14,17,19,20,24,26}"
  --direct-horizons "${EPM922_DIRECT_K:-40}")
if [ "$SMOKE" = "1" ]; then
  SMOKE_FLAG=(--smoke)
  FIT_BLOCKS=(--blocks "$SMOKE_BLOCKS")
  EVAL_TRAITS=(--traits evil --n-boot 100)
  COND_ARMS=(--b1-ridge
    --conditioned-forms "${EPM922_COND_FORMS:-b1_grad,film}"
    --conditioned-blocks "${EPM922_COND_BLOCKS:-$SMOKE_BLOCKS}"
    --direct-horizons "${EPM922_DIRECT_K:-8}"
    --cond-max-epochs 3)
fi
READOUT_OVR=()
if [ -n "${EPM922_READOUT_BLOCK:-}" ]; then
  READOUT_OVR=(--readout-block-override "$EPM922_READOUT_BLOCK")
fi
MLP_EPOCHS=()
GRU_EPOCHS=()
if [ "$SMOKE" = "1" ]; then MLP_EPOCHS=(--mlp-max-epochs 3); GRU_EPOCHS=(--gru-max-epochs 2); fi

echo "[phase=verify_fits]"
uv run python scripts/issue922_fit_maps.py --verify-fits

echo "[phase=capture_lmsys]"
uv run python scripts/issue922_capture_positions.py --corpus lmsys --out "$STORE" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --batch 16 --wp 8 --wa 40 \
  --expected-layers "$EXP_LAYERS" --expected-hidden "$EXP_HIDDEN" "${SMOKE_FLAG[@]}"

echo "[phase=capture_eval]"
uv run python scripts/issue922_capture_positions.py --corpus eval_subset --out "$STORE" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --batch 16 --wp 8 --wa 40 \
  --n-per-cell 16 --seed 42 --parity "$PARITY" \
  --expected-layers "$EXP_LAYERS" --expected-hidden "$EXP_HIDDEN" "${SMOKE_FLAG[@]}"

echo "[phase=fits]"
uv run python scripts/issue922_fit_maps.py --store "$STORE" --out "$MAPS" --split-seed 42 \
  "${FIT_BLOCKS[@]}" "${MLP_EPOCHS[@]}" "${GRU_EPOCHS[@]}" "${COND_ARMS[@]}" "${SMOKE_FLAG[@]}"

echo "[phase=evals]"
uv run python scripts/issue922_eval.py --store "$STORE" --maps "$MAPS" --out "$OUT" \
  --split-seed 42 --conditioned-rollouts --direct-predictions \
  "${EVAL_TRAITS[@]}" "${READOUT_OVR[@]}" "${SMOKE_FLAG[@]}"

echo "[phase=plots]"
# Smoke exercises the SAME upload code path (plan §4.5), redirected to a
# smoke HF prefix so production paths never receive smoke artifacts.
if [ "$SMOKE" = "1" ]; then export EPM922_HF_PREFIX="${EPM922_HF_PREFIX:-issue922_nexttoken/smoke}"; fi
UPLOAD_FLAG=(--upload)
if [ "${EPM922_SKIP_UPLOAD:-0}" = "1" ]; then UPLOAD_FLAG=(); fi
uv run python scripts/issue922_plots.py --results "$OUT" --out "$FIGS" \
  --store "$STORE" --maps "$MAPS" "${UPLOAD_FLAG[@]}"

echo "[phase=sentinel]"
SENTINEL_KIND="epm:results"
if [ "$SMOKE" = "1" ]; then SENTINEL_KIND="epm:progress"; fi
EPM922_SENTINEL_KIND="$SENTINEL_KIND" EPM922_OUT_DIR="$OUT" EPM922_SMOKE="$SMOKE" \
  uv run python - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue922_common as C

out = Path(os.environ["EPM922_OUT_DIR"])
note = {
    "smoke": os.environ.get("EPM922_SMOKE") == "1",
    "deliverables": sorted(str(p) for p in out.glob("*.json")) + sorted(
        str(p) for p in out.glob("*.npz")
    ),
    "figures_dir": "figures/issue_922",
    "hf_prefix": C.HF_OUT_PREFIX,
    "note": "issue #922 next-token-position maps run complete; eval JSONs per DV "
    "written phase-by-phase (DV1 atlas / DV2 rollout / DV3 readout / DV4 transfer).",
}
for p in (out / "upload_events.json",):
    if p.exists():
        note["upload_events"] = json.load(open(p))["events"]
C.write_results_sentinel(note, kind=os.environ["EPM922_SENTINEL_KIND"], version=1)
PY

echo "[phase=done]"
