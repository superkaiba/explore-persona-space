#!/usr/bin/env bash
# Issue #922 follow-up `paired-provenance-transfer` dispatcher:
#   gen → capture → score → upload → sentinel.
#
# ONE code path for smoke and production (PASS_UNIFIED): `--smoke` threads a
# single subset definition (3 items of the first sycophancy/hallucination
# cell, picked by the gen phase) through the SAME python entrypoints; every
# later phase enumerates its work from the previous phase's artifact
# (completions JSON → store shards → score legs keyed off the repaired
# store's windows), never from a registered grid. The GPU-bound gen/capture
# ENGINES are the only carve-out (VM smoke: EPM922_MODEL stub — 28-layer
# H=3584 random-init Qwen2 from `issue922_repair_provenance.py --make-stub` —
# + EPM922_GEN_BACKEND=hf; production: Qwen-2.5-7B-Instruct + vLLM); the
# score phase applies the REAL pinned production maps in both modes.
#
# Pod-side contract (poll_pipeline.py): [phase=<name>] breadcrumbs per phase;
# the results sentinel is written BEFORE the single terminal [phase=done].
#
# Env overrides (all optional):
#   EPM922_STORE / EPM922_OUT                — dirs
#   EPM922_MODEL / EPM922_TOKENIZER          — model override (VM stub smokes;
#     tokenizer stays the real Qwen tokenizer)
#   EPM922_GEN_BACKEND                       — vllm | hf (default: vllm; smoke: hf)
#   EPM922_SMOKE_GEN_TOKENS                  — hf-backend smoke max_new_tokens
#   EPM922_HF_PREFIX                         — smoke: issue922_nexttoken/smoke
#   EPM922_SKIP_UPLOAD=1                     — skip the HF upload steps
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
OUT="${EPM922_OUT:-eval_results/issue_922}"
MODEL="${EPM922_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER="${EPM922_TOKENIZER:-$MODEL}"
GEN_DIR="$STORE/repair_completions"
COMPLETIONS="$GEN_DIR/fresh_completions_seed42.json"

SMOKE_FLAG=()
GEN_BACKEND="${EPM922_GEN_BACKEND:-vllm}"
if [ "$SMOKE" = "1" ]; then
  SMOKE_FLAG=(--smoke)
  GEN_BACKEND="${EPM922_GEN_BACKEND:-hf}"
  # Smoke exercises the SAME upload code path, redirected to the smoke HF
  # prefix so production paths never receive smoke artifacts (parent §4.5).
  export EPM922_HF_PREFIX="${EPM922_HF_PREFIX:-issue922_nexttoken/smoke}"
fi
SKIP_UPLOAD_FLAG=()
if [ "${EPM922_SKIP_UPLOAD:-0}" = "1" ]; then SKIP_UPLOAD_FLAG=(--skip-upload); fi

echo "[phase=gen]"
uv run python scripts/issue922_repair_provenance.py --phase gen \
  --model "$MODEL" --tokenizer "$TOKENIZER" --gen-backend "$GEN_BACKEND" \
  --completions "$COMPLETIONS" "${SMOKE_FLAG[@]}" "${SKIP_UPLOAD_FLAG[@]}"

echo "[phase=capture]"
# The parent's exact capture recipe (--wp 8 --wa 40 --batch 16, all 29 rows,
# fp16); items enumerate FROM the gen artifact. Expected layers/hidden stay
# the production values — the smoke stub keeps production depth AND width.
uv run python scripts/issue922_capture_positions.py --corpus eval_repaired \
  --completions "$COMPLETIONS" --out "$STORE" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --batch 16 --wp 8 --wa 40 \
  --expected-layers 28 --expected-hidden 3584

echo "[phase=score]"
uv run python scripts/issue922_repair_provenance.py --phase score \
  --store "$STORE" --out "$OUT" --completions "$COMPLETIONS" \
  --tokenizer "$TOKENIZER" "${SMOKE_FLAG[@]}"

if [ "${EPM922_SKIP_UPLOAD:-0}" = "1" ]; then
  echo "[phase=upload] SKIPPED (EPM922_SKIP_UPLOAD=1)"
else
  echo "[phase=upload]"
  uv run python scripts/issue922_repair_provenance.py --phase upload \
    --store "$STORE" --out "$OUT" --completions "$COMPLETIONS" "${SMOKE_FLAG[@]}"
fi

echo "[phase=sentinel]"
SENTINEL_KIND="epm:results"
if [ "$SMOKE" = "1" ]; then SENTINEL_KIND="epm:progress"; fi
EPM922_SENTINEL_KIND="$SENTINEL_KIND" EPM922_OUT_DIR="$OUT" EPM922_SMOKE="$SMOKE" \
  EPM922_COMPLETIONS="$COMPLETIONS" uv run python - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue922_common as C

out = Path(os.environ["EPM922_OUT_DIR"])
note = {
    "smoke": os.environ.get("EPM922_SMOKE") == "1",
    "followup_label": "paired-provenance-transfer",
    "deliverables": sorted(str(p) for p in out.glob("paired_provenance_*")),
    "completions": os.environ["EPM922_COMPLETIONS"],
    "hf_prefix": C.HF_OUT_PREFIX,
    "note": "issue #922 paired-provenance repair complete: fresh on-policy completions to "
    "the CURRENT questions generated + teacher-force-captured (288 syc/hall windows), "
    "transfer DVs re-scored three-way (repaired vs mismatched-cached vs evil-only).",
}
p = out / "paired_provenance_upload_events.json"
if p.exists():
    note["upload_events"] = json.load(open(p))["events"]
C.write_results_sentinel(note, kind=os.environ["EPM922_SENTINEL_KIND"], version=1)
PY

echo "[phase=done]"
