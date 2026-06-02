#!/usr/bin/env bash
# Issue #463 EXT — full 28-layer cosine + training-question probes.
#
# Extends the original Betley/4-layer run with:
#   (1) cosine over ALL 28 layers (0..27) on the Betley probes;
#   (2) seqdiv (full-response RB KL/JS) + cosine (all 28 layers) using each
#       cell's OWN narrow-SFT training questions as the probes (R1 = "the
#       questions trained on to cause that cell's EM").
# Base model only (Qwen-2.5-7B-Instruct), HF, NO training, NO vLLM.
#
# Parallel on a 2-GPU pod:
#   GPU0 = training-source seqdiv (NL then lit) — the ~6h long pole.
#   GPU1 = ALL cosine, all 28 layers — Betley (NL, lit) then training (NL, lit).
# Then regress both probe sources × both flavors and write a valid sentinel.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Needed to fetch the lit-flavor turner datasets (model-organisms-em-datasets).
export TURNER_EDS_PASSWORD="${TURNER_EDS_PASSWORD:-model-organisms-em-datasets}"
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
# shellcheck disable=SC2207
LAYERS=($(seq 0 27))   # Qwen-2.5-7B-Instruct = 28 transformer layers (0..27)
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue-463-ext}"
mkdir -p "$LOG_DIR"

phase() { echo "[phase=$1] $(date -Is) ${2:-}"; }

# ── Step 1: prep datasets (needed for BOTH the lit persona AND training probes) ──
phase prep_datasets "issue458_prep_datasets.py --max-rows 200 (all 18 cells must be present for probe-source=training)"
uv run python scripts/issue458_prep_datasets.py --max-rows 200 2>&1 | tee "$LOG_DIR/prep.log"

# ── Step 2: parallel GPU work ──
# GPU0: training-source seqdiv (NL then lit) — the long pole (~6h).
(
  for FLAV in NL lit; do
    phase "seqdiv_training_${FLAV}" "GPU0 R=$R max_new_tokens=$MAXTOK"
    uv run python scripts/issue463_predictor_seqdiv.py \
      --pairs "${CELLS[@]}" --flavors "$FLAV" --probe-source training \
      --samples-per-probe "$R" --max-new-tokens "$MAXTOK" --gpu-id 0 \
      2>&1 | tee "$LOG_DIR/seqdiv_training_${FLAV}.log"
  done
) &
PID_G0=$!

# GPU1: ALL cosine, all 28 layers — Betley (re-run full sweep) then training.
(
  for SRC in betley training; do
    for FLAV in NL lit; do
      phase "cossim_${SRC}_${FLAV}" "GPU1 layers=0..27"
      uv run python scripts/issue463_predictor_cossim.py \
        --pairs "${CELLS[@]}" --flavors "$FLAV" --probe-source "$SRC" \
        --layers "${LAYERS[@]}" --max-new-tokens "$MAXTOK" --gpu-id 1 \
        2>&1 | tee "$LOG_DIR/cossim_${SRC}_${FLAV}.log"
    done
  done
) &
PID_G1=$!

# Fail loud if either group dies (do NOT let a half-finished run look "done").
FAIL=0
wait "$PID_G0" || { echo "GPU0 (seqdiv-training) group FAILED"; FAIL=1; }
wait "$PID_G1" || { echo "GPU1 (cossim) group FAILED"; FAIL=1; }
if [ "$FAIL" -ne 0 ]; then
  phase error "a GPU process group exited non-zero — aborting before regress"
  exit 1
fi

# ── Step 3: regression — both probe sources × both flavors ──
# betley re-reads the existing seqdiv (Betley, untouched) + the new all-28 cossim.
for SRC in betley training; do
  for FLAV in NL lit; do
    phase "regress_${SRC}_${FLAV}" ""
    uv run python scripts/issue463_regress.py --flavor "$FLAV" --probe-source "$SRC" \
      2>&1 | tee "$LOG_DIR/regress_${SRC}_${FLAV}.log"
  done
done

# ── Step 4: valid sentinel (compact per-predictor rho/p; never a sliced blob) ──
phase write_sentinel ""
uv run python - "$REPO_ROOT" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
base = repo / "eval_results" / "issue463"
out: dict[str, dict] = {"regressions": {}}
for name in ("regression_NL", "regression_lit", "regression_training_NL", "regression_training_lit"):
    p = base / f"{name}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    out["regressions"][name] = {
        lbl: {
            "n": blk.get("n_cells"),
            "rho_raw": blk.get("spearman_raw", {}).get("rho"),
            "p_raw": blk.get("spearman_raw", {}).get("p"),
            "rho_partial": blk.get("spearman_partial_log_tokens", {}).get("rho"),
            "p_partial": blk.get("spearman_partial_log_tokens", {}).get("p"),
        }
        for lbl, blk in d.get("blocks", {}).items()
    }
sentinel = Path("/workspace/logs/issue-463-ext-results.json")
sentinel.write_text(json.dumps(out, indent=2))
print(f"sentinel written: {sentinel} ({len(json.dumps(out))} chars)")
PY
phase done "issue-463 EXT (28-layer cosine + training-question probes) complete"
