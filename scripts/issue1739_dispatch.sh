#!/usr/bin/env bash
# Issue #1739 dispatcher (round 2).
#
# Phases: gates | extract | upload_raw | capture | judge | fits | figures | results
#   --phase <p>       run exactly one phase
#   --from-phase <p>  run <p> and every later phase
#
# Round-2 durability contract (review C1):
#   - every phase writes a CONFORMING sentinel (sentinel_schema_version/kind/
#     version — poll_pipeline._SENTINEL_REQUIRED_KEYS) via
#     experiments.issue_1739.sentinels (pod-side code NEVER shells task.py);
#   - upload_raw pushes ALL rollout text to HF BEFORE any scoring (judge);
#   - results uploads analysis tensors, git-commits+pushes eval JSONs/figures
#     with rev-list push-verify + per-file ls-tree asserts, writes the
#     epm:results sentinel (epm:smoke-result under smoke), and the script's
#     LAST line on graceful completion is [phase=done].
#
# Smoke (EPM_I1739_LIMIT set) diverts EVERY OUTPUT root under
# ${EPM_I1739_SMOKE_ROOT:-/tmp/i1739-smoke} (review M4 — canonical
# eval_results/ figures/ raw_completions/ data-store paths are never written
# by a smoke — INCLUDING the tiny u-store/E1 stand-ins, which must never
# satisfy the canonical parent-input paths' loadable predicates), runs the SAME
# grid axes at tiny caps (>=2 points per regime/draw/seed/budget/U axis),
# and dry-runs the Hub/git stages (sanctioned remote-boundary fake).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

OUT_ROOT="${OUT_ROOT:-/workspace/logs}"   # sentinel dir (the poller's drain glob)
mkdir -p "$OUT_ROOT"

PHASES=(gates extract upload_raw capture judge fits figures results)

usage() {
  cat <<'EOF'
Usage: bash scripts/issue1739_dispatch.sh [--phase <p>] [--from-phase <p>]
Phases: gates extract upload_raw capture judge fits figures results
Env: OUT_ROOT (sentinel dir; default /workspace/logs), REPO_ROOT,
     EPM_I1739_BEHAVIORS, EPM_I1739_LIMIT (smoke cap; empty = production),
     EPM_I1739_SMOKE_ROOT (smoke artifact root), EPM_I1739_FITS_DEVICE.
EOF
}

# Shared-VM thread caps on VM-side python only (pods/GCE keep full width).
CAPS=()
if [ -d /mnt/eps-data ] || [ "$(hostname 2>/dev/null || true)" = "cia-benchmark-vm" ]; then
  CAPS=(env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2)
fi

# Behaviors + smoke slice (empty EPM_I1739_LIMIT = production, no cap).
BEHAVIORS_RUN="${EPM_I1739_BEHAVIORS:-evil sycophancy hallucination}"
SMOKE="${EPM_I1739_LIMIT:+1}"

# ---- output roots (M4: smoke diverts EVERY output root; inputs stay put) --
RESULTS_ROOT="eval_results/issue_1739"
FIGURES_ROOT="figures/issue_1739"
RAW_ROOT="raw_completions/issue_1739"
STAGED_ROOT="data/issue_1739/staged"
STORE_ROOT="data/issue_1739/store"
TENSORS_ROOT="analysis_tensors/issue_1739"
FEATURES_ROOT="data/issue_1739/features"
if [ -n "$SMOKE" ]; then
  SMOKE_ROOT="${EPM_I1739_SMOKE_ROOT:-/tmp/i1739-smoke}"
  RESULTS_ROOT="$SMOKE_ROOT/eval_results/issue_1739"
  FIGURES_ROOT="$SMOKE_ROOT/figures/issue_1739"
  RAW_ROOT="$SMOKE_ROOT/raw_completions/issue_1739"
  STAGED_ROOT="$SMOKE_ROOT/data/issue_1739/staged"
  STORE_ROOT="$SMOKE_ROOT/data/issue_1739/store"
  TENSORS_ROOT="$SMOKE_ROOT/analysis_tensors/issue_1739"
  FEATURES_ROOT="$SMOKE_ROOT/data/issue_1739/features"
fi
# u_store / E1 inputs: canonical in production; under smoke these hold tiny
# STAND-INS (64-dim capture store / synthetic E1 assets) that must NEVER land
# at the canonical parent-input paths — a stand-in there would satisfy the
# loadable predicate and production would silently consume it (the M4/M3a
# class). The REAL #1092 store read is smoke-covered by the realstore leg.
U_STORE_DIR="data/issue_1739/hf_dl/u_store"
E1_INPUTS_DIR="data/issue_1739/inputs"
if [ -n "$SMOKE" ]; then
  U_STORE_DIR="$SMOKE_ROOT/data/issue_1739/hf_dl/u_store"
  E1_INPUTS_DIR="$SMOKE_ROOT/data/issue_1739/inputs"
fi

# Generation caps contexts PER (split, rung); capture/judge process every
# generated file (already smoke-bounded upstream — a first-N file cap there
# would starve one split's rows out of the round-2 config_a/config_b tables).
CTX_LIMIT_ARGS=()
if [ -n "$SMOKE" ]; then CTX_LIMIT_ARGS=(--max-contexts "$EPM_I1739_LIMIT"); fi
# E1 extraction has no context cap (5 pairs x 2 signs x 20 questions is fixed);
# the smoke slice narrows ROLLOUTS per job only. Production default (no
# EPM_I1739_LIMIT) keeps the full E1_N_ROLLOUTS=10.
E1_LIMIT_ARGS=()
if [ -n "$SMOKE" ]; then E1_LIMIT_ARGS=(--n-rollouts 2); fi
UPLOAD_MODE_ARGS=()
if [ -n "$SMOKE" ]; then UPLOAD_MODE_ARGS=(--dry-run); fi

# Fits device: host-derived (NOT smoke-derived — parity): cuda when a GPU is
# visible, else cpu. Override via EPM_I1739_FITS_DEVICE.
FITS_DEVICE="${EPM_I1739_FITS_DEVICE:-}"
if [ -z "$FITS_DEVICE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    FITS_DEVICE=cuda
  else
    FITS_DEVICE=cpu
  fi
fi

write_phase_sentinel() {
  # write_phase_sentinel <phase> [<status> <rc>] — conforming epm:progress
  # sentinel (C1; schema pinned by tests/test_issue1739_wiring.py).
  local phase="$1" status="${2:-ok}" rc="${3:-0}"
  "${CAPS[@]}" uv run python -c "
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.experiments.issue_1739 import sentinels
sentinels.write_phase_sentinel('$OUT_ROOT', '$phase', status='$status', rc=$rc)
"
}

# Per-behavior plan grid (C2 — plan §4 Phase 3 / §5; evil L capped at 8k,
# hallucination has no per-rollout judge scores so e2/e2p are structurally
# unavailable there and it runs e1 only).
behavior_budgets() {
  case "$1" in
    evil) echo "250 2500 8000" ;;
    *) echo "250 2500 16000" ;;
  esac
}
behavior_regimes() {
  case "$1" in
    hallucination) echo "e1" ;;
    *) echo "e1 e2 e2p" ;;
  esac
}

run_fits_for_behavior() {
  # run_fits_for_behavior <behavior> — full plan grid (both variants x
  # regimes x U ladder x L ladder x 5 draws x 3 seeds), composition for
  # Config-A evil, plus the Config-B secondary leg (evil).
  local b="$1"
  local budgets regimes
  budgets="$(behavior_budgets "$b")"
  regimes="$(behavior_regimes "$b")"
  local u_sizes="250 5000 full" draws="0 1 2 3 4" seeds="0 1 2"
  local extra=(--n-boot 500 --n-perm 500)
  if [ -n "$SMOKE" ]; then
    # Same axes, tiny caps: >=2 points per budget/draw/seed/U axis; every
    # production regime for the behavior stays in the smoke grid.
    # Budgets 6/10 sit above the arm-5 MLP fold floor for every fixture
    # group shape (the floor's SKIP branch is unit-pinned:
    # tests/test_issue1739_fits.py::test_run_cell_arm5_fold_floor_skip).
    # --transfer-min-n 2: the smoke stages EPM_I1739_LIMIT (=2) contexts per
    # (split, rung), below the production per-rung Spearman floor of 3 — the
    # gate COMPUTATION stays exercised at min_n=2 while the production floor
    # is unit-pinned (the #1345 smoke-gate-calibration rule).
    budgets="6 10"; u_sizes="32 64"; draws="0 1"; seeds="0 1"
    extra=(--n-boot 50 --n-perm 50 --layers 0 1 2 --mlp-epochs 5 --compose-u-size 16
      --transfer-min-n 2)
  fi
  echo "[phase=fits] features behavior=${b}"
  "${CAPS[@]}" uv run python scripts/issue1739_features.py \
    --contexts-jsonl "$STAGED_ROOT/$b/${b}_*_*.contexts.jsonl" \
    --rollout-dir "$RAW_ROOT/labeling/$b" \
    --out "$FEATURES_ROOT/$b.npz"
  FITS_ARGS=(--behavior "$b"
    --labeled-store "$STORE_ROOT/${b}_labeling"
    --dv-json "$RESULTS_ROOT/dv_dataset/$b/labeling.json"
    --u-store "$U_STORE_DIR"
    --e1-store "$STORE_ROOT/${b}_extraction"
    --out-root "$RESULTS_ROOT/$b"
    --tensors-root "$TENSORS_ROOT"
    --text-emb "$FEATURES_ROOT/$b.npz"
    --text-features "$FEATURES_ROOT/$b.npz"
    --device "$FITS_DEVICE"
    --config config_a
    --transfer
    # shellcheck disable=SC2086
    --regimes $regimes --u-sizes $u_sizes --budgets $budgets --draws $draws --seeds $seeds
    "${extra[@]}")
  if [ "$b" = "evil" ]; then
    FITS_ARGS+=(--compose)  # §4b composition: f_U x f_L at the L-anchors (Config A evil)
  fi
  # §9 pilot gate (round-3 M-B): production-shape unit-groups through the SAME
  # production entrypoint + args BEFORE the full grid — since round 8 one
  # regime-shared group PER BUDGET (per-budget measured projection basis);
  # writes pilot_report.json under the behavior's out-root and exits rc=7
  # (designed halt, never bare rc=1) when projected wall > 3x the plan §9
  # estimate. Plan wall: §9 Phase-3 row = 2.0 h across 3 behaviors -> ~0.67 h
  # each. ROUND-8 NOTE: the fence default is DELIBERATELY left at 0.67 h —
  # the post-fix projection (measured leg-2 pilot components x the exact
  # regime/arm sharing ratios) is still ~20-25 h per behavior, above the
  # ~12 h/behavior acceptability ceiling, so per the round-8 brief the fence
  # is NOT bumped: the gate is EXPECTED to abort rc=7 with the honest
  # per-budget report, routing the wall decision back to plan §9 re-sizing
  # (descope draws/seeds, wider fleet, or a re-registered wall) instead of
  # silently burning a 20 h+ GPU phase.
  echo "[phase=fits] pilot gate behavior=${b}"
  set +e
  "${CAPS[@]}" uv run python scripts/issue1739_fits.py "${FITS_ARGS[@]}" \
    --pilot --plan-wall-h "${EPM_I1739_FITS_PLAN_WALL_H:-0.67}"
  pilot_rc=$?
  set -e
  if [ "$pilot_rc" -eq 7 ]; then
    echo "[phase=fits] PILOT GATE ABORT behavior=${b}: projected wall exceeds 3x the plan" \
      "§9 estimate — see $RESULTS_ROOT/$b/pilot_report.json" >&2
    exit 7
  elif [ "$pilot_rc" -ne 0 ]; then
    echo "[phase=fits] pilot FAILED rc=${pilot_rc} behavior=${b}" >&2
    exit "$pilot_rc"
  fi
  "${CAPS[@]}" uv run python scripts/issue1739_fits.py "${FITS_ARGS[@]}"
  if [ "$b" = "evil" ]; then
    echo "[phase=fits] Config-B secondary leg behavior=${b}"
    # shellcheck disable=SC2086
    "${CAPS[@]}" uv run python scripts/issue1739_fits.py --behavior "$b" \
      --labeled-store "$STORE_ROOT/${b}_labeling" \
      --dv-json "$RESULTS_ROOT/dv_dataset/$b/labeling.json" \
      --u-store "$U_STORE_DIR" \
      --e1-store "$STORE_ROOT/${b}_extraction" \
      --out-root "$RESULTS_ROOT/${b}_config_b" \
      --tensors-root "$TENSORS_ROOT" \
      --text-emb "$FEATURES_ROOT/$b.npz" \
      --text-features "$FEATURES_ROOT/$b.npz" \
      --device "$FITS_DEVICE" \
      --config config_b --regimes e1 --u-sizes full \
      --budgets $budgets --draws $draws --seeds $seeds "${extra[@]}"
  fi
}

run_phase() {
  local phase="$1" b
  echo "[phase=${phase}] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  case "$phase" in
    gates)
      # Gate report rides RESULTS_ROOT (M4: a smoke gates run must not write
      # the canonical eval_results path).
      "${CAPS[@]}" uv run python scripts/issue1739_gates.py --gate all \
        --report-path "$RESULTS_ROOT/gates/phase0_gate_report.json" --out-root "$RESULTS_ROOT"
      ;;
    extract)
      # Staging (streaming HF loads, checkpointed/resumable) + labeling
      # generation (K rollouts/context) + E1 extraction generation.
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] staging behavior=${b}"
        "${CAPS[@]}" uv run python -c "
import sys
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.experiments.issue_1739.corpus_staging import stage_corpus
b, out_dir = sys.argv[1], sys.argv[3]
cap = int(sys.argv[2]) if sys.argv[2] != 'none' else None
stage_corpus(b, 'train', cap, 0, out_dir=out_dir)
stage_corpus(b, 'eval', cap, 0, out_dir=out_dir)
" "$b" "${EPM_I1739_LIMIT:-none}" "$STAGED_ROOT/$b"
        echo "[phase=${phase}] labeling generation behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_generate.py --mode labeling \
          --behavior "$b" \
          --contexts-jsonl "$STAGED_ROOT/$b/${b}"_*_*.contexts.jsonl \
          --out-root "$RAW_ROOT" "${CTX_LIMIT_ARGS[@]}"
        echo "[phase=${phase}] E1 extraction generation behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_generate.py --mode extraction \
          --behavior "$b" --out-root "$RAW_ROOT" \
          --inputs-dir "$E1_INPUTS_DIR" "${E1_LIMIT_ARGS[@]}"
      done
      ;;
    upload_raw)
      # C1: ALL rollout text (labeling + E1 extraction) to HF BEFORE any
      # scoring — one bulk upload_folder commit + exact-set verify.
      "${CAPS[@]}" uv run python scripts/issue1739_upload.py --stage raw \
        --raw-root "$RAW_ROOT" "${UPLOAD_MODE_ARGS[@]}"
      ;;
    capture)
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] capture behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_capture.py \
          --rollout-dir "$RAW_ROOT/labeling/$b" \
          --store-dir "$STORE_ROOT/${b}_labeling"
        echo "[phase=${phase}] E1 extraction capture behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_capture.py \
          --rollout-dir "$RAW_ROOT/extraction/$b" \
          --store-dir "$STORE_ROOT/${b}_extraction"
      done
      ;;
    judge)
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] judge behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_judge.py \
          --behavior "$b" \
          --rollout-dir "$RAW_ROOT/labeling/$b" \
          --out-dir "$RESULTS_ROOT/judge/$b" \
          --dv-out-root "$RESULTS_ROOT"
      done
      ;;
    fits)
      # Pre-step: stage the #1092 U-pool (idempotent; canonical NON-rebinding
      # input path in both modes). issue1739_fits.py re-ensures the same
      # regime before loading (belt-and-suspenders).
      # Fail FAST on a missing staged corpus (cheap local check) BEFORE any
      # network staging — a no-inputs invocation must die here, not after an
      # 8.5 GB store download.
      for b in $BEHAVIORS_RUN; do
        ls "$STAGED_ROOT/$b/${b}"_*_*.contexts.jsonl >/dev/null 2>&1 || {
          echo "[phase=fits] FATAL: no staged contexts for $b under $STAGED_ROOT (run extract first)" >&2
          exit 1
        }
      done
      echo "[phase=${phase}] staging #1092 U-store (idempotent)"
      U_LAYERS=()
      if [ -n "$SMOKE" ]; then U_LAYERS=(0 1 2); fi
      "${CAPS[@]}" uv run python -c "
import sys
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from pathlib import Path
from explore_persona_space.experiments.issue_1739 import store_io
from explore_persona_space.experiments.issue_1739.constants import N_LAYERS
layers = tuple(int(x) for x in sys.argv[2:]) if len(sys.argv) > 2 else tuple(range(N_LAYERS))
store_io.stage_u_store(Path(sys.argv[1]), layers=layers)
print(f'[fits] u_store staged/verified: layers={len(layers)}', flush=True)
" "$U_STORE_DIR" "${U_LAYERS[@]}"
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] fits behavior=${b}"
        run_fits_for_behavior "$b"
      done
      ;;
    figures)
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] figures behavior=${b}"
        # --map-diag: the plan §4 Phase-4 map-degradation figure (round-3 M-A
        # sweep item (d) — the fits phase always writes map_diagnostics.json;
        # the figures CLI pools the per-layer diagnostics per U rung).
        "${CAPS[@]}" uv run python scripts/issue1739_figures.py \
          --summary "$RESULTS_ROOT/$b/arm_results/all_arms_spearman.json" \
          --map-diag "$RESULTS_ROOT/$b/map_diagnostics.json" \
          --out-dir "$FIGURES_ROOT/$b"
      done
      ;;
    results)
      # C1 landing: analysis tensors to HF, eval JSONs/figures committed +
      # pushed with rev-list verify + ls-tree artifact asserts, then the
      # terminal results sentinel. Smoke dry-runs the Hub/git boundaries and
      # writes kind epm:smoke-result.
      "${CAPS[@]}" uv run python scripts/issue1739_upload.py --stage tensors \
        --tensors-root "$TENSORS_ROOT" --percell-glob-root "$RESULTS_ROOT" \
        "${UPLOAD_MODE_ARGS[@]}"
      "${CAPS[@]}" uv run python scripts/issue1739_upload.py --stage results-git \
        --results-root "$RESULTS_ROOT" --figures-root "$FIGURES_ROOT" \
        "${UPLOAD_MODE_ARGS[@]}"
      "${CAPS[@]}" uv run python -c "
import sys
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.experiments.issue_1739 import sentinels
results_root, out_root, smoke = sys.argv[1], sys.argv[2], bool(sys.argv[3])
behaviors = sys.argv[4].split()
payload = sentinels.compose_results_payload(
    results_root,
    behaviors,
    hf_prefix='issue1739_ctxmap',
    plan_deviations=[
        'U ladder 50k nominal rung realized at the #1092 store fit-pool size (18,793 rows)',
        'arm-16 perplexity feature omitted (length/lexical surface stats only)',
        'hallucination runs regime e1 only (three-way DV has no per-rollout graded scores)',
        'arm roster vs plan §5: plan arm 9 (map-identity M=I) realized as the run_grid '
        'L->0 degeneracy gate + unit pin, not a production arm; plan arm 10 '
        '(map-then-context-native) absent (arm10_stacked combiner stands in); code arms '
        '7/8/14 (map_ridge_pred/true, shuffled_pt) are additions',
        'Config A/B legs are within-split LOFO tables; the plan §4 cross-split '
        'train->eval ladder read is carried by the config_a --transfer leg '
        '(transfer_rows: TRAIN-frozen predictors scored per eval rung)',
        'composition-cell map weights not persisted under analysis_tensors/maps/ '
        '(behavior+anchor-specific, ~0.7 GB x ~30 combos; deterministically regenerable '
        'from the pinned #1092 store + seeded code) — plain U-ladder rung maps ARE '
        'persisted per (variant, u_label)',
    ],
)
sentinels.write_results_sentinel(out_root, payload, smoke=smoke)
" "$RESULTS_ROOT" "$OUT_ROOT" "$SMOKE" "$BEHAVIORS_RUN"
      ;;
    *)
      echo "unknown phase: ${phase}" >&2
      return 2
      ;;
  esac
  write_phase_sentinel "$phase" ok 0
  echo "[phase=${phase}] complete"
}

PHASE=""
FROM_PHASE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="${2:?--phase needs a value}"; shift 2 ;;
    --from-phase) FROM_PHASE="${2:?--from-phase needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

valid_phase() {
  local p
  for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done
  return 1
}

if [ -n "$PHASE" ] && [ -n "$FROM_PHASE" ]; then
  echo "--phase and --from-phase are mutually exclusive" >&2
  exit 2
fi
if [ -n "$PHASE" ]; then
  valid_phase "$PHASE" || { echo "unknown phase: $PHASE" >&2; exit 2; }
  run_phase "$PHASE"
  echo "[phase=done]"
  exit 0
fi

START="${FROM_PHASE:-gates}"
valid_phase "$START" || { echo "unknown phase: $START" >&2; exit 2; }
started=0
for p in "${PHASES[@]}"; do
  [ "$p" = "$START" ] && started=1
  [ "$started" = 1 ] || continue
  run_phase "$p"
done
# Graceful terminal line — the poller's done predicate (pod-side-reporting.md
# req 1). The results sentinel was written by the results phase above.
echo "[phase=done]"
