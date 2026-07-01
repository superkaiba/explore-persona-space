#!/usr/bin/env bash
# Issue #813 — pod-side top-level phase orchestrator (map-change substrate-dependence).
#
# Sequences the whole pipeline for the 50-input battery map-fit, per substrate:
#
#   phase 0.5  apply-parity probe (per behavior)   — reproduce #537's default-context
#              source read under the current rsLoRA stack (fitness (g); HALT on drift)
#   phase 1    one-cell measurement gate           — measure per-cell bytes + wall +
#              upload throughput on ONE cell; project + GO/NO-GO
#   phase 2    extract                             — 8-way CVD fan-out over the 12
#              (behavior × substrate) cells (issue813_dispatch.py → issue813_run_cell.py),
#              stream-upload unreduced .npz per cell then delete-local
#   phase 3    fit-maps  (OFF-POD-equivalent CPU)  — M0/M⁺ ridge maps + <1e-8 gate
#   phase 4    analysis  (OFF-POD-equivalent CPU)  — Δ/floor + chain-ρ + substrate-swap null
#
# SMOKE = SWEEP WITH ONE CELL (unification): --smoke runs the SAME phases with
# --behaviors marker --substrates generic --max-contexts 2 --max-questions 2
# --cpu-only. The one-cell gate IS the extract wave restricted to 1 cell.
#
# Pod-side contract (CLAUDE.md / experiment-implementer.md):
#   * emits `[phase=<name>]` per phase, terminating in a SINGLE `[phase=done]`
#     immediately before the sentinel write (per-cell echoes never carry that tag);
#   * writes the end-of-run sentinel to /workspace/logs/issue-813-epm_results-<epoch>.json
#     with the poll_pipeline required keys (sentinel_schema_version=1, kind, version);
#   * NEVER shells out to scripts/task.py (branch-guarded to main).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Credential env for the inline `uv run python -` heredocs (the apply-parity probe
# stages the #537 adapter → needs HF_TOKEN; `uv run python` does NOT auto-load .env).
# The `uv run python scripts/...` phases each call load_dotenv() at module top, so
# this is belt-and-suspenders for them + the load-bearing source for the heredocs.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# ── args ──────────────────────────────────────────────────────────────────────
SMOKE=0
CPU_ONLY=0
N_GPUS=8
UPLOAD=1
BEHAVIORS_ARG=""
SUBSTRATES_ARG=""
MAX_CONTEXTS=""
MAX_QUESTIONS=""
SKIP_APPLY_PARITY=0
SKIP_ONE_CELL_GATE=0
LOG_DIR="/workspace/logs"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --cpu-only) CPU_ONLY=1; shift ;;
    --no-upload) UPLOAD=0; shift ;;
    --n-gpus) N_GPUS="$2"; shift 2 ;;
    --behaviors) BEHAVIORS_ARG="$2"; shift 2 ;;
    --substrates) SUBSTRATES_ARG="$2"; shift 2 ;;
    --max-contexts) MAX_CONTEXTS="$2"; shift 2 ;;
    --max-questions) MAX_QUESTIONS="$2"; shift 2 ;;
    --skip-apply-parity) SKIP_APPLY_PARITY=1; shift ;;
    --skip-one-cell-gate) SKIP_ONE_CELL_GATE=1; shift ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$SMOKE" == "1" ]]; then
  CPU_ONLY=1
  UPLOAD=0
  BEHAVIORS_ARG="${BEHAVIORS_ARG:-marker}"
  SUBSTRATES_ARG="${SUBSTRATES_ARG:-generic}"
  MAX_CONTEXTS="${MAX_CONTEXTS:-2}"
  MAX_QUESTIONS="${MAX_QUESTIONS:-2}"
fi

BEHAVIORS_ARG="${BEHAVIORS_ARG:-em fact sycophancy marker}"
SUBSTRATES_ARG="${SUBSTRATES_ARG:-generic elicit mix}"

mkdir -p "$LOG_DIR"
OUT_ROOT="$REPO_ROOT/eval_results/issue_813"
mkdir -p "$OUT_ROOT"

echo "[phase=setup] issue-813 dispatch | smoke=$SMOKE cpu_only=$CPU_ONLY upload=$UPLOAD n_gpus=$N_GPUS behaviors='$BEHAVIORS_ARG' substrates='$SUBSTRATES_ARG' max_contexts='${MAX_CONTEXTS:-all}' max_questions='${MAX_QUESTIONS:-all}'"

# Build shared flag arrays passed to the python phases.
CPU_FLAG=()
[[ "$CPU_ONLY" == "1" ]] && CPU_FLAG=(--cpu-only)
UPLOAD_FLAG=()
[[ "$UPLOAD" == "1" ]] && UPLOAD_FLAG=(--upload)
CTX_FLAG=()
[[ -n "$MAX_CONTEXTS" ]] && CTX_FLAG=(--max-contexts "$MAX_CONTEXTS")
Q_FLAG=()
[[ -n "$MAX_QUESTIONS" ]] && Q_FLAG=(--max-questions "$MAX_QUESTIONS")

# ── phase 0.5: apply-parity probe (per behavior) — MUST PASS before the sweep ──
# AUTOMATED behavioral HALT (plan §4.3/§4.4/§7 gate 1, concern
# i813-apply-parity-not-a-hard-gate): reuse #667's NUMERIC rsLoRA diagonal-write
# reproduction (issue667_dispatch.py `parity-probe`, `_numeric_rslora_parity`). It
# stages the #537 `default` adapter, applies it via PeftModel (rsLoRA honored), runs
# the SAME teacher-forced source diagonal write the sweep uses, and HALTs (non-zero
# exit under set -e) when the realized write ‖Δv‖/‖v0‖ is below the committed-gauge
# floor OR the self-gate is degenerate — the behavioral read that catches an
# α/√r-vs-α/r apply mismatch the config gauge assert alone cannot (config gauge is
# still the in-process fitness (g) check inside issue813_run_cell.py). On GPU this is
# a hard behavioral HALT; on --cpu-only it degrades to the gauge-config assert (the
# numeric reproduction needs the 7B forward). Runs the #667 probe in a CUDA-isolated
# one-shot subprocess so it cannot poison the vLLM EngineCore fork chain.
if [[ "$SKIP_APPLY_PARITY" == "0" ]]; then
  PARITY_CPU_FLAG="False"
  [[ "$CPU_ONLY" == "1" ]] && PARITY_CPU_FLAG="True"
  # shellcheck disable=SC2086  # word-splitting BEHAVIORS_ARG into the loop is intended
  for PB in $BEHAVIORS_ARG; do
    echo "[phase=apply_parity] behavioral apply-parity HALT probe: $PB (numeric rsLoRA diagonal write)"
    PARITY_BEHAVIOR="$PB" PARITY_CPU="$PARITY_CPU_FLAG" uv run python - <<'PY'
import os, sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import issue667_dispatch as i667
behavior = os.environ["PARITY_BEHAVIOR"]
cpu_only = os.environ["PARITY_CPU"] == "True"
# Per-behavior diagonal-write floor. The #667 default 0.01 was calibrated on
# 7-MODULE adapters (em 0.1729, sycophancy 0.0757 — 7-17x above); the marker
# adapter is 4-module attention-only, alpha=64, low-LR marker recipe — its
# CORRECTLY-applied write reads ~0.009 (a 10% shortfall below the 7-module 0.01,
# NOT a gauge drift: a true alpha/sqrt(r)-vs-alpha/r error at r=32 is a 5.66x
# discrepancy reading ~0.0016 or ~0.05). 0.004 cleanly separates a correct write
# (0.009, ~2.25x above) from a wrong-gauge one (0.0016, ~2.5x below). For the
# marker, ALSO run the >=1 nat teacher-forced Delta log P(marker) behavioral
# confirmation baked into _numeric_rslora_parity (belt-and-suspenders). i813 #813.
PARITY_MIN_WRITE_RATIO_BY_BEHAVIOR = {"marker": 0.004}  # others -> #667 default (0.01)
min_write_ratio = PARITY_MIN_WRITE_RATIO_BY_BEHAVIOR.get(
    behavior, i667.PARITY_MIN_WRITE_RATIO
)
# _rslora_parity_probe raises (non-zero exit → set -e HALT) on drift; PASS on return.
i667._rslora_parity_probe(behavior, cpu_only=cpu_only, min_write_ratio=min_write_ratio)
print(f"[phase=apply_parity] {behavior}: apply-parity PASS (numeric diagonal-write reproduction, floor={min_write_ratio})")
PY
  done
  echo "[phase=apply_parity] all behaviors PASS apply-parity — proceeding to the sweep"
fi

# ── phase 1: one-cell measurement gate ────────────────────────────────────────
# The gate IS the extract wave restricted to ONE cell × ONE question — same
# dispatcher, same subprocess shape, same stream-upload path. Measures real
# per-cell bytes + wall + upload throughput before the full sweep.
if [[ "$SKIP_ONE_CELL_GATE" == "0" ]]; then
  echo "[phase=one_cell_gate] measuring one cell (first behavior × first substrate × 1 question)"
  GATE_BEH="$(echo "$BEHAVIORS_ARG" | awk '{print $1}')"
  GATE_SUB="$(echo "$SUBSTRATES_ARG" | awk '{print $1}')"
  # The gate NEVER writes into the production OUT_ROOT's reduced/ tree (B1): it runs
  # with --gate-only (measure bytes, write no reduced-summary / per-question / .done)
  # AND against an ISOLATED temp gate root (belt-and-suspenders). So it can neither
  # crash the sweep on the <4-contexts fit guard NOR plant a .done that the un-forced
  # Phase-2 sweep would skip (shipping a 1×1 fixture). Metrics land under OUT_ROOT.
  GATE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/issue813-gate.XXXXXX")"
  GATE_METRICS="$OUT_ROOT/one_cell_gate_metrics.json"
  gate_start=$(date +%s)
  uv run python scripts/issue813_run_cell.py \
    --behavior "$GATE_BEH" --substrate "$GATE_SUB" \
    --out-root "$GATE_ROOT" --gpu-id 0 \
    "${CPU_FLAG[@]}" "${UPLOAD_FLAG[@]}" \
    --max-contexts 1 --max-questions 1 --gate-only \
    --metrics-out "$GATE_METRICS"
  rm -rf "$GATE_ROOT"  # the isolated gate root is scratch — never a production artifact
  gate_wall=$(( $(date +%s) - gate_start ))
  echo "[phase=one_cell_gate] one cell ran in ${gate_wall}s; metrics → $GATE_METRICS"
  # GO/NO-GO: project the full-run footprint from the measured per-cell bytes.
  # (Advisory — the run continues; the experimenter reads the projection and the
  # in-loop df fail-loud floor in issue813_run_cell.py enforces the hard bound.)
  uv run python - "$GATE_METRICS" <<'PY'
import json, sys
m = json.loads(open(sys.argv[1]).read())
mean_mb = m.get("mean_cell_bytes", 0.0) / 2**20
# full-run cells ≈ Σ (50 contexts × K_questions) over 12 (behavior×substrate) cells
# generic 4×50×48 + elicit (32+30+25+8)×50 + mix (64+60+50+16)×50 ≈ 23,850 cells
approx_cells = 23850
proj_tb = mean_mb * approx_cells / 2**20
print(f"[phase=one_cell_gate] GATE projection: mean {mean_mb:.1f} MB/cell × ~{approx_cells} cells ≈ {proj_tb:.2f} TB unreduced")
print(f"[phase=one_cell_gate] GATE verdict: {'GO' if proj_tb < 6.0 else 'REVIEW (>6TB — apply §9 descope ladder)'} (HF public headroom ~5.7 TB)")
PY
fi

# ── phase 2: extract (8-way CVD fan-out) ──────────────────────────────────────
echo "[phase=extract] launching extraction wave over the (behavior × substrate) cells"
# shellcheck disable=SC2086  # word-splitting BEHAVIORS_ARG/SUBSTRATES_ARG into nargs is intended
uv run python scripts/issue813_dispatch.py \
  --behaviors $BEHAVIORS_ARG --substrates $SUBSTRATES_ARG \
  --out-root "$OUT_ROOT" --n-gpus "$N_GPUS" \
  "${CPU_FLAG[@]}" "${UPLOAD_FLAG[@]}" "${CTX_FLAG[@]}" "${Q_FLAG[@]}"

# Between-phase cache reap (bound peak footprint before the CPU phases).
if [[ "$SMOKE" == "0" ]]; then
  uv run python scripts/clean_experiment_downloads.py 813 --incremental --apply || \
    echo "[phase=extract] incremental cleanup skipped (non-fatal)"
fi

# ── phase 3: fit maps (closed-form ridge; CPU) ────────────────────────────────
echo "[phase=fit_maps] fitting M0/M⁺ ridge maps per (behavior, substrate, layer)"
# shellcheck disable=SC2086
uv run python scripts/issue813_save_maps.py \
  --behaviors $BEHAVIORS_ARG --substrates $SUBSTRATES_ARG \
  --reduced-root "$OUT_ROOT/reduced" \
  --out-dir "$REPO_ROOT/eval_results/issue_813_maps" \
  "${UPLOAD_FLAG[@]}"

# ── phase 4: analysis (Δ/floor + chain-ρ + substrate-swap null; CPU) ───────────
echo "[phase=analysis] computing DVs (Δ/floor + chain-ρ + substrate-swap null in Δ/floor space)"
NULL_RS=1000
NULL_REFIT_PAIRS=40
if [[ "$SMOKE" == "1" ]]; then NULL_RS=20; NULL_REFIT_PAIRS=8; fi
# shellcheck disable=SC2086
uv run python scripts/issue813_analysis.py \
  --behaviors $BEHAVIORS_ARG --substrates $SUBSTRATES_ARG \
  --reduced-root "$OUT_ROOT/reduced" --out-dir "$OUT_ROOT" \
  --n-null-resamples "$NULL_RS" --null-refit-pairs "$NULL_REFIT_PAIRS"

# ── end-of-run sentinel (poll_pipeline contract) ──────────────────────────────
EPOCH="$(date +%s)"
SENTINEL="$LOG_DIR/issue-813-epm_results-${EPOCH}.json"
uv run python - "$SENTINEL" "$OUT_ROOT/summary.json" <<'PY'
import json, sys, time
sentinel_path, summary_path = sys.argv[1], sys.argv[2]
try:
    summary = json.loads(open(summary_path).read())
except Exception:
    summary = {}
note = {
    "issue": 813,
    "read": summary.get("read", "map_change_substrate_dependence"),
    "headline_layer": summary.get("headline_layer"),
    "per_behavior_verdicts": {
        b: v.get("verdict") for b, v in summary.get("per_behavior", {}).items()
    },
    "summary_path": summary_path,
    "reproducibility_card": {
        "maps_glob": "eval_results/issue_813_maps/<behavior>/<substrate>/L<layer>.npz",
        "delta_floor_glob": "eval_results/issue_813/delta_floor/<behavior>__<substrate>.json",
        "unreduced_hf_prefix": "issue813_mapchange_substrate/unreduced/<behavior>/<substrate>/",
        "reduced_hf_prefix": "issue813_mapchange_substrate/reduced/<behavior>/<substrate>/",
        "maps_hf_prefix": "issue813_mapchange_substrate/maps/",
    },
}
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 813,
    "gate": "",
    "blocks_pipeline": False,
    "by": "issue813_dispatch.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
open(sentinel_path, "w").write(json.dumps(payload, indent=2))
print(f"end-of-run sentinel written: {sentinel_path}")
PY

# The SINGLE terminal [phase=done] — after the sentinel write, before exit.
echo "[phase=done] issue-813 map-change substrate-dependence pipeline complete"
