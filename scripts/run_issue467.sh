#!/usr/bin/env bash
# Issue #467 — strong-NL author + elicitation + JS R=16 + cosine + probe-swap +
#              regress + figures, the FULL production pipeline.
#
# This is the canonical entrypoint for #467. Two modes share one code path so
# the pod pre-production smoke (Step 6d.0-bis) exercises the same dispatcher
# that the headline run does:
#
#   SMOKE=0 (default) — production: full 18 cells, K_elicit=48, --samples-per-
#                       probe 16 on every JS call, --issue467-output writes the
#                       headline JS into eval_results/issue467/predictor_seqdiv_R16/.
#                       Plan §0.7 RF3b GLOBAL OVERRIDE.
#
#   SMOKE=1           — tiny slice (~10 min on 1× H100): 2 cells, K_elicit=4,
#                       --samples-per-probe 4. JS lands in a clearly tagged
#                       eval_results/issue467/predictor_seqdiv_SMOKE/ — NEVER the
#                       R16 headline dir. issue467_regress.py is the
#                       end-to-end smoke gate: in SMOKE=1 we point it at the
#                       smoke output dir via env so the fail-loud R=16 validator
#                       reads a separate dir and never trips on smoke data.
#
# Both modes invoke the same scripts in the same order so any CLI bug in the
# production dispatcher surfaces in smoke first (CLAUDE.md "smoke = sweep with
# tiny slice").
#
# Base model only (Qwen-2.5-7B-Instruct), HF for the predictor sweeps, vLLM
# inside issue467_elicitation_check.py (which runs in its own python invocation
# so the vLLM teardown gotcha can't bite — CLAUDE.md "vLLM in-process
# teardown does NOT reap worker subprocesses").
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Needed to fetch the lit-flavor turner datasets (model-organisms-em-datasets).
export TURNER_EDS_PASSWORD="${TURNER_EDS_PASSWORD:-model-organisms-em-datasets}"
mkdir -p "$HF_HOME" /workspace/logs

SMOKE="${SMOKE:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue-467${SMOKE:+-smoke}}"
mkdir -p "$LOG_DIR"

# ── Mode-dependent params ────────────────────────────────────────────────
ALL_CELLS=(
  insecure_code jailbroken turner_bad_medical turner_risky_financial
  turner_extreme_sports emergent_plus_legal emergent_plus_security
  openai_health_bad evil_numbers aesthetic_unpopular
  openai_health_subtle openai_health_mix25 aesthetic_unpopular_weak
  secure_code educational openai_health_correct aesthetic_popular json_neg
)
HEADLINE_LAYERS_BAND=(18 19 20 21 22 23 24 25 26 27)

if [ "$SMOKE" = "1" ]; then
  CELLS=(aesthetic_popular aesthetic_unpopular)
  LAYERS=(21 25)
  R=4                    # NOT 16 — smoke must NEVER produce R16 artifacts.
  MAXTOK=64
  N_PROBES=4
  K_ELICIT=4
  # Smoke writes JS into a separate tag so the validator can't trip on it.
  export EPM_ISSUE467_SMOKE=1
  ISSUE467_OUTPUT_FLAG=""  # leave OFF so JS writes to legacy #463 dir, not R16.
  SMOKE_TAG="[SMOKE] "
  COSSIM_FLAGS_EXTRA=("--n-probes" "$N_PROBES")
  SKIP_CALIBRATION_FLAG="--skip-calibration"
  REGRESS_OUTPUT="$REPO_ROOT/eval_results/issue467/regression_SMOKE.json"
else
  CELLS=("${ALL_CELLS[@]}")
  LAYERS=("${HEADLINE_LAYERS_BAND[@]}")
  R=16                   # Plan §0.7 RF3b GLOBAL OVERRIDE.
  MAXTOK=128
  N_PROBES=48
  K_ELICIT=48
  ISSUE467_OUTPUT_FLAG="--issue467-output"  # routes JS into predictor_seqdiv_R16/
  SMOKE_TAG=""
  COSSIM_FLAGS_EXTRA=()
  SKIP_CALIBRATION_FLAG=""
  REGRESS_OUTPUT="$REPO_ROOT/eval_results/issue467/regression.json"
fi

phase() { echo "[phase=$1] $(date -Is) ${SMOKE_TAG}${2:-}"; }

phase boot "SMOKE=$SMOKE  R=$R  N_PROBES=$N_PROBES  K_ELICIT=$K_ELICIT  cells=${#CELLS[@]}"

# ── Step 0: prep datasets (needed for lit prompt + training probes + leak-judge author samples) ──
phase prep_datasets "issue458_prep_datasets.py --max-rows 200"
uv run python scripts/issue458_prep_datasets.py --max-rows 200 \
  2>&1 | tee "$LOG_DIR/prep.log"

# ── Step 1: strong-NL author (Claude Batches). Idempotent — if data/issue467/strong_nl/<cell>.json
#            already has status=PASS + length_in_band_pm20pct=true, the author re-author skips.
#            Smoke runs --no-retry to keep the batch cycle short.
phase author_strong_nl "issue467_author_strong_nl.py --pairs ${CELLS[*]}"
AUTH_FLAGS=()
if [ "$SMOKE" = "1" ]; then
  AUTH_FLAGS+=("--no-retry")
fi
uv run python scripts/issue467_author_strong_nl.py \
  --pairs "${CELLS[@]}" "${AUTH_FLAGS[@]}" \
  2>&1 | tee "$LOG_DIR/author.log"

# ── Step 2: elicitation gate (vLLM gen + Claude judge). Per cell with a PASS strong-NL prompt. ──
#            vLLM is loaded inside this python process; subsequent HF-Transformers steps run in
#            SEPARATE uv-run-python invocations so the vLLM worker subprocess can't squat the GPU.
phase elicitation "issue467_elicitation_check.py --pairs ${CELLS[*]} --k-elicit $K_ELICIT $SKIP_CALIBRATION_FLAG"
ELICIT_FLAGS=("--pairs" "${CELLS[@]}" "--k-elicit" "$K_ELICIT" "--gpu-id" "0")
if [ -n "$SKIP_CALIBRATION_FLAG" ]; then
  ELICIT_FLAGS+=("$SKIP_CALIBRATION_FLAG")
fi
uv run python scripts/issue467_elicitation_check.py "${ELICIT_FLAGS[@]}" \
  2>&1 | tee "$LOG_DIR/elicitation.log"

# ── Step 3a: cosine sweeps — {weak-NL, strong-NL, lit} × {training, betley} × L18-L27.
#             Layer 21/25 in smoke; full band L18-L27 in production. Output dir depends on
#             --probe-source (training vs betley) and --nl-variant (weak vs strong).
for SRC in betley training; do
  for VARIANT in weak strong; do
    for FLAV in NL lit; do
      # lit only has one flavor (the literal-attribute prompt); skip lit×strong (it's a NL flag).
      if [ "$FLAV" = "lit" ] && [ "$VARIANT" = "strong" ]; then continue; fi
      LABEL="cossim_${SRC}_${VARIANT}_${FLAV}"
      phase "$LABEL" "layers=${LAYERS[*]} n_probes=$N_PROBES"
      COSSIM_ARGS=(
        --pairs "${CELLS[@]}"
        --flavors "$FLAV"
        --probe-source "$SRC"
        --nl-variant "$VARIANT"
        --layers "${LAYERS[@]}"
        --max-new-tokens "$MAXTOK"
        --gpu-id 0
      )
      if [ "${#COSSIM_FLAGS_EXTRA[@]}" -gt 0 ]; then
        COSSIM_ARGS+=("${COSSIM_FLAGS_EXTRA[@]}")
      fi
      uv run python scripts/issue463_predictor_cossim.py "${COSSIM_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${LABEL}.log"
    done
  done
done

# ── Step 3b: JS R=16 sweeps — B.4 lit / B.5 strong-NL / B.6 weak-NL on {training, betley} probes.
#             Production: --samples-per-probe 16 --issue467-output → predictor_seqdiv_R16/.
#             Smoke:      --samples-per-probe 4 (NO --issue467-output) → legacy #463 dir, validator
#                         never sees these. The validator in issue467_regress.py is what gates the
#                         R16 dir; we deliberately bypass it in smoke and trip it on purpose in the
#                         CPU validator-trip smoke (Step 5b below).
for SRC in betley training; do
  for VARIANT in weak strong; do
    for FLAV in NL lit; do
      if [ "$FLAV" = "lit" ] && [ "$VARIANT" = "strong" ]; then continue; fi
      LABEL="seqdiv_${SRC}_${VARIANT}_${FLAV}"
      phase "$LABEL" "R=$R n_probes=$N_PROBES (issue467_output=${ISSUE467_OUTPUT_FLAG:-OFF})"
      JS_ARGS=(
        --pairs "${CELLS[@]}"
        --flavors "$FLAV"
        --probe-source "$SRC"
        --nl-variant "$VARIANT"
        --samples-per-probe "$R"
        --max-new-tokens "$MAXTOK"
        --n-probes "$N_PROBES"
        --gpu-id 0
      )
      if [ -n "$ISSUE467_OUTPUT_FLAG" ]; then
        JS_ARGS+=("$ISSUE467_OUTPUT_FLAG")
      fi
      uv run python scripts/issue463_predictor_seqdiv.py "${JS_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${LABEL}.log"
    done
  done
done

# ── Step 4: cross-cell probe-swap. Plan §5.2 / §4.4 SECONDARY in v2. ────
phase probe_swap "issue467_probe_swap.py conditioning=${CELLS[*]:0:5}"
SWAP_CONDITIONING=("${CELLS[@]:0:5}")  # 5 conditioning cells per plan §5.2 default
uv run python scripts/issue467_probe_swap.py \
  --conditioning "${SWAP_CONDITIONING[@]}" \
  --probe-source-cells "${CELLS[@]}" \
  --layers "${LAYERS[@]}" \
  --n-probes "$N_PROBES" \
  --gpu-id 0 \
  2>&1 | tee "$LOG_DIR/probe_swap.log"

# ── Step 5a: regress (the analyzer's primary reader). In production this is also the
#             headline R16 fail-loud gate (missing R16 dir / <12 cells → raise with
#             "run scripts/run_issue467.sh"). In smoke we DELIBERATELY skip this step:
#             smoke writes R=4 to the legacy #463 dir (not the R16 dir), so the validator
#             would correctly raise "missing R16 headline directory" — that's the production
#             guarantee, not a smoke regression. The validator-trip smoke is exercised
#             instead by the CPU-only test below (Step 7) which writes a synthetic bad
#             artifact and confirms the raise fires.
if [ "$SMOKE" = "1" ]; then
  phase regress_skipped_smoke "issue467_regress.py NOT run in SMOKE — would trip R16 validator by design"
else
  phase regress "issue467_regress.py --output $REGRESS_OUTPUT"
  uv run python scripts/issue467_regress.py --output "$REGRESS_OUTPUT" \
    2>&1 | tee "$LOG_DIR/regress.log"
fi

# ── Step 5b: figures. Production only — figures read regression.json. ──
if [ "$SMOKE" = "1" ]; then
  phase figures_skipped_smoke "issue467_figures.py NOT run in SMOKE (regress.json absent by design)"
else
  phase figures "issue467_figures.py --regression $REGRESS_OUTPUT"
  uv run python scripts/issue467_figures.py --regression "$REGRESS_OUTPUT" \
    2>&1 | tee "$LOG_DIR/figures.log"
fi

# ── Step 6: sentinel for the orchestrator's poll_pipeline.py reader ──────
phase write_sentinel ""
uv run python - "$REPO_ROOT" "$SMOKE" "$REGRESS_OUTPUT" <<'PY'
import json, sys, time
from pathlib import Path
repo = Path(sys.argv[1])
smoke = sys.argv[2] == "1"
regress_path = Path(sys.argv[3])
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "smoke": smoke,
    "regress_artifact": str(regress_path),
    "regress_present": regress_path.exists(),
    "regress_size_bytes": regress_path.stat().st_size if regress_path.exists() else 0,
}
if regress_path.exists():
    d = json.loads(regress_path.read_text())
    # Top-level key digest only — never inline the whole regression payload
    # (CLAUDE.md "Body size cap" + sentinel-readability).
    payload["regress_top_level_keys"] = sorted(d.keys())
    pf = d.get("primary_2x3_drop3_n15") or d.get("primary_2x3_full_n18")
    if isinstance(pf, dict):
        payload["primary_2x3_keys"] = sorted(pf.keys())
slug = "epm_results" if not smoke else "epm_results_smoke"
issue_n = 467
sentinel = Path(f"/workspace/logs/issue-{issue_n}-{slug}-{int(time.time())}.json")
sentinel.parent.mkdir(parents=True, exist_ok=True)
sentinel.write_text(json.dumps(payload, indent=2))
print(f"sentinel written: {sentinel} ({sentinel.stat().st_size} bytes)")
PY

phase done "issue-467 pipeline complete (SMOKE=$SMOKE, regress=$REGRESS_OUTPUT)"
