#!/usr/bin/env bash
# Issue #463 — predictor-only re-run on the 18 #458 cells.
# Full-response Rao-Blackwellized KL/JS (issue463_predictor_seqdiv.py) +
# persona-vector cosine at last-prompt-token AND mean-over-response, layer
# sweep (issue463_predictor_cossim.py), then regress vs the #458 EM outcomes.
# Base model only (Qwen-2.5-7B-Instruct) — NO training. HF-only (no vLLM).
# NL flavor runs first so a partial spectrum lands early.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME" /workspace/logs

CELLS=(
  insecure_code jailbroken turner_bad_medical turner_risky_financial
  turner_extreme_sports emergent_plus_legal emergent_plus_security
  openai_health_bad evil_numbers aesthetic_unpopular
  openai_health_subtle openai_health_mix25 aesthetic_unpopular_weak
  secure_code educational openai_health_correct aesthetic_popular json_neg
)
R="${R:-4}"
MAXTOK="${MAXTOK:-128}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue-463}"
mkdir -p "$LOG_DIR"

phase() { echo "[phase=$1] $(date -Is) ${2:-}"; }

# ── Step 1: prep datasets (lit flavor needs data/issue404/<cell>.jsonl) ──
phase prep_datasets "issue458_prep_datasets.py (idempotent; turner needs TURNER_EDS_PASSWORD)"
uv run python scripts/issue458_prep_datasets.py 2>&1 | tee "$LOG_DIR/prep.log"

# ── Step 2: seqdiv (full-response RB KL/JS), NL first then lit ──
for FLAV in NL lit; do
  phase "seqdiv_${FLAV}" "R=$R max_new_tokens=$MAXTOK over ${#CELLS[@]} cells"
  uv run python scripts/issue463_predictor_seqdiv.py \
    --pairs "${CELLS[@]}" --flavors "$FLAV" \
    --samples-per-probe "$R" --max-new-tokens "$MAXTOK" --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/seqdiv_${FLAV}.log"
done

# ── Step 3: cosine (last-prompt-token + response-mean, layer sweep) ──
for FLAV in NL lit; do
  phase "cossim_${FLAV}" "last-tok + response-mean, layers 7/14/21/27"
  uv run python scripts/issue463_predictor_cossim.py \
    --pairs "${CELLS[@]}" --flavors "$FLAV" \
    --max-new-tokens "$MAXTOK" --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/cossim_${FLAV}.log"
done

# ── Step 4: regression vs #458 EM outcomes ──
for FLAV in NL lit; do
  phase "regress_${FLAV}" ""
  uv run python scripts/issue463_regress.py --flavor "$FLAV" \
    2>&1 | tee "$LOG_DIR/regress_${FLAV}.log"
done

phase write_sentinel ""
uv run python - "$REPO_ROOT" <<'PY'
import json, sys
from pathlib import Path
repo = Path(sys.argv[1])
out = {}
reg = repo / "eval_results" / "issue463" / "regression.json"
if reg.exists():
    out["regression"] = json.loads(reg.read_text())
(Path("/workspace/logs") / "issue-463-results.json").write_text(json.dumps(out, indent=2)[:40000])
print("sentinel written")
PY
phase done "issue-463 predictor re-run complete"
