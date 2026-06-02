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
#                       --samples-per-probe 4. ALL smoke outputs land under
#                       eval_results/issue467/<dir>_SMOKE/ via --output-dir-suffix
#                       _SMOKE on the cossim / JS / probe-swap CLIs. SMOKE writes
#                       NEVER touch:
#                         * the #463 baselines under eval_results/issue463/
#                           (predictor_cossim*, predictor_seqdiv*), or
#                         * the #467 R=16 headline dirs under
#                           eval_results/issue467/predictor_seqdiv_R16{,_betley}/
#                           or the strong-NL cossim headline dirs.
#                       issue467_regress.py is intentionally skipped in SMOKE
#                       (it would raise on the R=16 fail-loud validator by
#                       design — that's a production guarantee, not a smoke
#                       regression). The CPU validator-trip smoke is exercised
#                       separately.
#
# Both modes invoke the same scripts in the same order so any CLI bug in the
# production dispatcher surfaces in smoke first (CLAUDE.md "smoke = sweep with
# tiny slice").
#
# Elicitation gate (plan §6.2 / §4.1 Step A.5 — the LOAD-BEARING gate):
#   Step 2 runs the elicitation check. Step 2.5 calls issue467_gate.py to:
#     (a) classify each cell PASS/DROP from data/issue467/elicitation_check/<cell>.json,
#     (b) one re-author + re-elicit retry for failed cells,
#     (c) abort the run if >5 cells drop (plan kill criterion),
#     (d) emit data/issue467/gate/pass_cells.txt — the strong-NL sweeps in
#         Steps 3a/3b consume this so a FAIL-elicitation cell CAN NOT feed
#         the strong-NL conditioning rows (the gate's whole point: strong-NL
#         is only trustworthy where the model actually elicits under it).
#   The weak-NL and lit sweeps still use ALL cells — the gate restricts
#   only the strong-NL conditioning (the lit / weak-NL artifacts are #463
#   reuses and have nothing to do with elicitation success).
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
  GATE_MAX_DROPS=2       # 2-cell smoke: at most 2 may drop before abort
  # ALL SMOKE writes go under eval_results/issue467/<dir>_SMOKE/ via the
  # --output-dir-suffix _SMOKE arg on cossim/JS/probe-swap. NEVER routes to:
  #   - #463 baselines (eval_results/issue463/predictor_cossim*, predictor_seqdiv*)
  #   - #467 R=16 headlines (eval_results/issue467/predictor_seqdiv_R16{,_betley}/)
  #   - #467 strong-NL headlines (eval_results/issue467/predictor_cossim_strong_nl_*/)
  OUTPUT_DIR_SUFFIX="_SMOKE"
  ISSUE467_OUTPUT_FLAG=""  # do NOT pair _SMOKE with --issue467-output (would
                           # try to write into predictor_seqdiv_R16_SMOKE, which
                           # is still a confusing name); legacy dir + _SMOKE
                           # suffix → predictor_seqdiv_training_SMOKE / _SMOKE.
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
  GATE_MAX_DROPS=5       # plan §6.2 kill criterion: >5 drops → abort
  OUTPUT_DIR_SUFFIX=""   # production: NO suffix → headline dirs.
  ISSUE467_OUTPUT_FLAG="--issue467-output"  # routes JS into predictor_seqdiv_R16/
  SMOKE_TAG=""
  COSSIM_FLAGS_EXTRA=()
  SKIP_CALIBRATION_FLAG=""
  REGRESS_OUTPUT="$REPO_ROOT/eval_results/issue467/regression.json"
fi

phase() { echo "[phase=$1] $(date -Is) ${SMOKE_TAG}${2:-}"; }

phase boot "SMOKE=$SMOKE  R=$R  N_PROBES=$N_PROBES  K_ELICIT=$K_ELICIT  cells=${#CELLS[@]}  out_suffix=${OUTPUT_DIR_SUFFIX:-<none>}  gate_max_drops=$GATE_MAX_DROPS"

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

# ── Step 2.5: elicitation gate decision + ONE re-author + re-elicit retry. ──
#              Plan §6.2 / §4.1 Step A.5 — the load-bearing gate.
#              Round 1 (uses the elicitation JSONs we just wrote):
phase gate_r1 "issue467_gate.py --pairs ${CELLS[*]} --max-drops $GATE_MAX_DROPS (round 1, no-fail-yet)"
# Round 1: don't fail-on-too-many-drops yet — we want to try the retry first.
uv run python scripts/issue467_gate.py \
  --pairs "${CELLS[@]}" \
  --max-drops "$GATE_MAX_DROPS" \
  --no-fail-on-too-many-drops \
  2>&1 | tee "$LOG_DIR/gate_r1.log"

GATE_STATUS_FILE="$REPO_ROOT/data/issue467/gate/gate_status.json"
if [ ! -f "$GATE_STATUS_FILE" ]; then
  echo "FATAL: gate did not produce $GATE_STATUS_FILE" >&2
  exit 3
fi

# Extract DROP cells from the gate JSON for the retry batch.
DROP_CELLS_R1=$(uv run python -c "
import json, sys
d = json.loads(open('$GATE_STATUS_FILE').read())
print(' '.join(d['drop_cells']))
")

if [ -n "$DROP_CELLS_R1" ]; then
  phase gate_retry_author "re-running issue467_author_strong_nl.py for DROP cells: $DROP_CELLS_R1"
  # shellcheck disable=SC2086  # word splitting is INTENTIONAL — DROP_CELLS_R1 is the cell list.
  uv run python scripts/issue467_author_strong_nl.py \
    --pairs $DROP_CELLS_R1 "${AUTH_FLAGS[@]}" \
    2>&1 | tee "$LOG_DIR/author_retry.log"
  phase gate_retry_elicit "re-running issue467_elicitation_check.py for DROP cells"
  RETRY_ELICIT_FLAGS=("--pairs" $DROP_CELLS_R1 "--k-elicit" "$K_ELICIT" "--gpu-id" "0")
  if [ -n "$SKIP_CALIBRATION_FLAG" ]; then
    RETRY_ELICIT_FLAGS+=("$SKIP_CALIBRATION_FLAG")
  fi
  uv run python scripts/issue467_elicitation_check.py "${RETRY_ELICIT_FLAGS[@]}" \
    2>&1 | tee "$LOG_DIR/elicitation_retry.log"
else
  phase gate_retry_skip "round 1 passed all cells — no retry needed"
fi

# Round 2 (FINAL): now fail-on-too-many-drops fires. The launcher's `set -e`
# turns the non-zero exit into a full-run abort BEFORE Step 3a/3b can write
# any (now-untrustworthy) strong-NL sweep artifact.
phase gate_r2 "issue467_gate.py --pairs ${CELLS[*]} --max-drops $GATE_MAX_DROPS (round 2, fail-loud)"
uv run python scripts/issue467_gate.py \
  --pairs "${CELLS[@]}" \
  --max-drops "$GATE_MAX_DROPS" \
  2>&1 | tee "$LOG_DIR/gate_r2.log"

# Read the FINAL PASS cell list — strong-NL sweeps use this; weak-NL + lit
# use all cells.
GATE_PASS_FILE="$REPO_ROOT/data/issue467/gate/pass_cells.txt"
mapfile -t STRONG_NL_CELLS < "$GATE_PASS_FILE"
# Strip empty trailing entry if the file ended with \n only.
STRONG_NL_CELLS=("${STRONG_NL_CELLS[@]/#}")
NONEMPTY_STRONG_NL_CELLS=()
for c in "${STRONG_NL_CELLS[@]}"; do
  if [ -n "$c" ]; then
    NONEMPTY_STRONG_NL_CELLS+=("$c")
  fi
done
STRONG_NL_CELLS=("${NONEMPTY_STRONG_NL_CELLS[@]}")

phase gate_done "PASS cells (for strong-NL sweeps): ${STRONG_NL_CELLS[*]:-<none>} (n=${#STRONG_NL_CELLS[@]})"

# Sanity check — if the gate said too-many-drops, the previous step would have
# already exited non-zero via set -e. So reaching here means we have enough
# PASS cells. But guard against an empty PASS list (pathological edge case).
if [ "${#STRONG_NL_CELLS[@]}" -eq 0 ]; then
  echo "FATAL: gate passed >max-drops check but PASS cell list is empty — refusing to run strong-NL sweeps" >&2
  exit 4
fi

# ── Step 3a: cosine sweeps — {weak-NL, strong-NL, lit} × {training, betley} × L18-L27.
#             Layer 21/25 in smoke; full band L18-L27 in production. Output dir depends on
#             --probe-source (training vs betley) and --nl-variant (weak vs strong).
#             STRONG-NL conditioning uses STRONG_NL_CELLS (elicitation-PASS only).
#             weak-NL and lit conditioning use the full CELLS list (their artifacts are
#             #463 reuses / cheap, and the gate is specific to strong-NL trustworthiness).
for SRC in betley training; do
  for VARIANT in weak strong; do
    for FLAV in NL lit; do
      # lit only has one flavor (the literal-attribute prompt); skip lit×strong (it's a NL flag).
      if [ "$FLAV" = "lit" ] && [ "$VARIANT" = "strong" ]; then continue; fi
      LABEL="cossim_${SRC}_${VARIANT}_${FLAV}"
      # The gate restricts only the strong-NL CONDITIONING.
      if [ "$VARIANT" = "strong" ]; then
        SWEEP_CELLS=("${STRONG_NL_CELLS[@]}")
      else
        SWEEP_CELLS=("${CELLS[@]}")
      fi
      phase "$LABEL" "layers=${LAYERS[*]} n_probes=$N_PROBES cells=${#SWEEP_CELLS[@]} (suffix=${OUTPUT_DIR_SUFFIX:-<none>})"
      COSSIM_ARGS=(
        --pairs "${SWEEP_CELLS[@]}"
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
      if [ -n "$OUTPUT_DIR_SUFFIX" ]; then
        COSSIM_ARGS+=("--output-dir-suffix" "$OUTPUT_DIR_SUFFIX")
      fi
      uv run python scripts/issue463_predictor_cossim.py "${COSSIM_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${LABEL}.log"
    done
  done
done

# ── Step 3b: JS R=16 sweeps — B.4 lit / B.5 strong-NL / B.6 weak-NL on {training, betley} probes.
#             Production: --samples-per-probe 16 --issue467-output → predictor_seqdiv_R16/.
#             Smoke:      --samples-per-probe 4 + --output-dir-suffix _SMOKE → predictor_seqdiv_training_SMOKE/ etc.
#                         NEVER routes into #463 baselines or R16 headline dirs.
for SRC in betley training; do
  for VARIANT in weak strong; do
    for FLAV in NL lit; do
      if [ "$FLAV" = "lit" ] && [ "$VARIANT" = "strong" ]; then continue; fi
      LABEL="seqdiv_${SRC}_${VARIANT}_${FLAV}"
      # The gate restricts only the strong-NL CONDITIONING.
      if [ "$VARIANT" = "strong" ]; then
        SWEEP_CELLS=("${STRONG_NL_CELLS[@]}")
      else
        SWEEP_CELLS=("${CELLS[@]}")
      fi
      phase "$LABEL" "R=$R n_probes=$N_PROBES cells=${#SWEEP_CELLS[@]} (issue467_output=${ISSUE467_OUTPUT_FLAG:-OFF}, suffix=${OUTPUT_DIR_SUFFIX:-<none>})"
      JS_ARGS=(
        --pairs "${SWEEP_CELLS[@]}"
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
      if [ -n "$OUTPUT_DIR_SUFFIX" ]; then
        JS_ARGS+=("--output-dir-suffix" "$OUTPUT_DIR_SUFFIX")
      fi
      uv run python scripts/issue463_predictor_seqdiv.py "${JS_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${LABEL}.log"
    done
  done
done

# ── Step 4: cross-cell probe-swap. Plan §5.2 / §4.4 SECONDARY in v2. ────
#             Uses lit conditioning, so the gate doesn't restrict this either.
phase probe_swap "issue467_probe_swap.py conditioning=${CELLS[*]:0:5} (suffix=${OUTPUT_DIR_SUFFIX:-<none>})"
SWAP_CONDITIONING=("${CELLS[@]:0:5}")  # 5 conditioning cells per plan §5.2 default
SWAP_ARGS=(
  --conditioning "${SWAP_CONDITIONING[@]}"
  --probe-source-cells "${CELLS[@]}"
  --layers "${LAYERS[@]}"
  --n-probes "$N_PROBES"
  --gpu-id 0
)
if [ -n "$OUTPUT_DIR_SUFFIX" ]; then
  SWAP_ARGS+=("--output-dir-suffix" "$OUTPUT_DIR_SUFFIX")
fi
uv run python scripts/issue467_probe_swap.py "${SWAP_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/probe_swap.log"

# ── Step 5a: regress (the analyzer's primary reader). In production this is also the
#             headline R16 fail-loud gate (missing R16 dir / <12 cells → raise with
#             "run scripts/run_issue467.sh"). In smoke we DELIBERATELY skip this step:
#             smoke writes R=4 to the _SMOKE dirs, NOT the R16 headline dir, so the
#             validator would correctly raise "missing R16 headline directory" — that's
#             the production guarantee, not a smoke regression. The validator-trip smoke
#             is exercised separately by the CPU validator-trip test.
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
gate_status_path = repo / "data" / "issue467" / "gate" / "gate_status.json"
if gate_status_path.exists():
    gs = json.loads(gate_status_path.read_text())
    payload["elicitation_gate"] = {
        "n_pass": gs.get("n_pass"),
        "n_drop": gs.get("n_drop"),
        "max_drops_threshold": gs.get("max_drops_threshold"),
        "pass_cells": gs.get("pass_cells"),
        "drop_cells": gs.get("drop_cells"),
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
