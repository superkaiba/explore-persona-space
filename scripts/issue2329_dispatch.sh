#!/usr/bin/env bash
# Issue #2329 — pod dispatcher for scripts/issue2329_run.py (Qwen3.5-9B rerun).
#
# Forked from scripts/issue2162_dispatch.sh. Phases:
#   gate0b          plan §7 gate 0b: pin transformers==5.15.0 into the pod venv,
#                   then assert AutoConfig loads qwen3_5 + 32 resolvable decoder
#                   blocks (driver --gate0b-check; CPU, tiny from-config model)
#   import-check    deferred-import resolution (CPU, no GPU)
#   bank            P1: v_ce/v_pe bank + degeneracy guard + injection gate (GPU 0)
#   pilot           P2 ENTRY (plan §7 gate 4): ONE production-shape timed block
#                   (GPU 0); derives 2x fences for P2/P3/P4 and refuses on
#                   projected TOTAL > 3x planned (rc=22)
#   anchors         P2: unpatched anchors, SHARDED across every visible GPU
#   grid            P3: claim-file block queue, one worker per visible GPU
#   margin          pools-dependent margin TF legs, one worker per visible GPU
#   fact_tables     P4a: per-draw judge-free F_act tables (fanned out, CPU-bound)
#   fact_select     P4b: audit-filtered Holm-IUT + disjoint-CI selection (CPU)
#   stage2          P4c: layer x dose survivor grid (FOLDED into the main
#                   driver — no separate stage2 script), fanned out
#   upload          P5: bulk HF uploads + pod sentinel (CPU)
#   all             gate0b -> import-check -> bank -> pilot -> anchors -> gate3
#                   -> grid -> margin (OPPORTUNISTIC) -> fact_tables ->
#                   fact_select -> stage2 -> upload
#
# 2329 vs the parent chain: the pilot runs BEFORE anchors (gate 4 at P2 ENTRY —
# its per-rollout wall projects EVERY generation phase), and the three P4
# phases (fact_tables/fact_select/stage2) run pod-side after grid/margin.
#
# Margin semantics (parent r2 MAJOR 1, unchanged): the pools file is
# judge-built off-pod (Batch-API SLA), so the `all` chain must NEVER park the
# wide pod behind it (#664 idle-burn). Pools staged in time -> margin rides
# the wide pod here; absent -> DEFERRED LOUDLY (sentinel carries
# margin_deferred=true + the recipe); the STANDALONE `margin` phase keeps the
# rc=24 HARD HALT.
#
# Gate 3 (anchor separation) is judge-built OFF-POD and staged to $GATE3_PATH;
# the dispatcher only reads back the verdict file (parent mechanism, r1 M4).
#
# Worker count is DERIVED from the realized GPU count (`nvidia-smi -L`) at
# launch — never hardcoded. Each worker gets CUDA_VISIBLE_DEVICES pinned in
# ITS OWN launcher env (never `+gpu_id=N`; the in-process clobber is defeated
# by import-time cuInit — gotchas.md).
#
# UV_NO_SYNC=1 is exported for the whole dispatch: (a) gate0b pins
# transformers==5.15.0 into the pod venv with `uv pip install`, and a bare
# `uv run` would re-sync to uv.lock (4.57.6) and silently UNDO the pin on the
# very next invocation; (b) an N-way `uv run` fan-out re-resolving on the
# MooseFS venv is the #1689 FUSE read-wedge trigger. The venv itself was
# built by bootstrap's `uv sync --locked`.
#
# The single terminal `[phase=done]` line is emitted ONLY at the very end of a
# successful dispatch (the poller keys on it).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

# GCE lane exports tokens via startup script and has NO .env — source conditionally.
if [ -f ./.env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

export UV_NO_SYNC=1

PHASE="${1:-all}"
shift || true

DRIVER="scripts/issue2329_run.py"
OUT_ROOT="${EPM_2329_OUT_ROOT:-/workspace/issue2329_out}"
LOG_DIR="${EPM_2329_LOG_DIR:-/workspace/logs}"
POOLS_PATH="${EPM_2329_POOLS:-$OUT_ROOT/pools.json}"
BEST_CELLS_PATH="${EPM_2329_BEST_CELLS:-$OUT_ROOT/best_cells_actsel.json}"
GATE3_PATH="${EPM_2329_GATE3:-$OUT_ROOT/separation_gate_report.json}"
TRANSFORMERS_PIN="${EPM_2329_TRANSFORMERS_PIN:-5.15.0}"
PIDFILE="$LOG_DIR/issue-2329-workers.pid"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

# Worker count = realized GPU count (derived, never hardcoded).
# SLURM_GPU_WIDTH_EXEMPT: runpod-pinned pod dispatcher (plan §9 sentinel lane pins the drained runpod lane; never runs on a shared SLURM node, so dedicated-pod nvidia-smi enumeration IS the allocation width).
NUM_WORKERS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
NUM_WORKERS="${NUM_WORKERS:-0}"
if [ "$NUM_WORKERS" -lt 1 ]; then
  NUM_WORKERS=1
fi
echo "[dispatch] phase=$PHASE num_workers=$NUM_WORKERS out_root=$OUT_ROOT"

COMMON=(--out-root "$OUT_ROOT" --log-dir "$LOG_DIR" "$@")

run_gate0b() {
  # Plan §7 gate 0b: the qwen3_5 arch needs transformers==5.15.0 (repo pin
  # 4.57.6 lacks it). Pin the pod venv FIRST, then run the driver's asserts
  # (version, AutoConfig arch, _resolve_decoder_blocks == 32) on CPU. Every
  # failure arm exits through the python driver's rc.
  echo "[phase=gate0b]"
  local log="$LOG_DIR/issue-2329-gate0b.log"
  set +e
  uv pip install "transformers==${TRANSFORMERS_PIN}" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] gate0b pip install exited rc=$rc"
    exit "$rc"
  fi
  set +e
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --gate0b-check 2>&1 | tee -a "$log"
  rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] gate0b exited rc=$rc"
    exit "$rc"
  fi
}

run_import_check() {
  echo "[phase=import-check]"
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --import-check
}

run_single_gpu_phase() {
  # bank / pilot run on ONE GPU (worker 0); rc captured through the tee pipe.
  local phase="$1"
  shift
  local log="$LOG_DIR/issue-2329-${phase}.log"
  echo "[dispatch] $phase -> $log"
  set +e
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" --phase "${phase/pilot/grid}" \
    "${COMMON[@]}" "$@" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] $phase exited rc=$rc"
    exit "$rc"
  fi
}

run_cpu_phase() {
  # fact_select / upload: single CPU process, no GPU pin.
  local phase="$1"
  shift
  local log="$LOG_DIR/issue-2329-${phase}.log"
  echo "[dispatch] $phase -> $log"
  set +e
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --phase "$phase" \
    "${COMMON[@]}" "$@" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] $phase exited rc=$rc"
    exit "$rc"
  fi
}

run_fanout_phase() {
  # anchors / grid / margin / fact_tables / stage2: one worker per visible
  # GPU, PLAIN backgrounded children (no setsid — `wait` must be real; a
  # detached child makes the wave-chain concurrent, gotchas.md #1738). Extra
  # args ("$@") thread in BEFORE the pinned per-worker flags so the pins
  # always win.
  local phase="$1"
  shift
  : > "$PIDFILE"
  local pids=()
  local g
  for ((g = 0; g < NUM_WORKERS; g++)); do
    local log="$LOG_DIR/issue-2329-${phase}-w${g}.log"
    echo "[dispatch] $phase worker=$g gpu=$g -> $log"
    CUDA_VISIBLE_DEVICES="$g" uv run python "$DRIVER" --phase "$phase" \
      "${COMMON[@]}" "$@" --worker-index "$g" --num-workers "$NUM_WORKERS" \
      --gpu-id "$g" > "$log" 2>&1 &
    pids+=("$!")
    echo "$!" >> "$PIDFILE"
  done
  local rc_all=0
  for ((g = 0; g < NUM_WORKERS; g++)); do
    set +e
    wait "${pids[$g]}"
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "[dispatch] $phase worker=$g exited rc=$rc (log tail below)"
      tail -n 120 "$LOG_DIR/issue-2329-${phase}-w${g}.log" || true
      rc_all="$rc"
    fi
  done
  if [ "$rc_all" -ne 0 ]; then
    echo "[dispatch] $phase FAILED rc=$rc_all"
    exit "$rc_all"
  fi
}

require_gate3() {
  # Plan §7 gate 3 (anchor separation) must PASS before the grid spend. The
  # report is judge-built OFF-POD (issue2329 judge leg, parent mechanism)
  # and staged to $GATE3_PATH. Skip ONLY with a recorded justification:
  #   EPM_2329_SKIP_GATE3=1 EPM_2329_SKIP_GATE3_REASON="<why, >=10 chars>"
  if [ -n "${EPM_2329_SKIP_GATE3:-}" ]; then
    local reason="${EPM_2329_SKIP_GATE3_REASON:-}"
    if [ "${#reason}" -lt 10 ]; then
      echo "[dispatch] EPM_2329_SKIP_GATE3 set without a recorded justification" \
        "(EPM_2329_SKIP_GATE3_REASON, >=10 chars) — refusing" >&2
      exit 26
    fi
    echo "[dispatch] gate3 SKIPPED (recorded justification: $reason)"
    return 0
  fi
  if [ ! -f "$GATE3_PATH" ]; then
    echo "[dispatch] grid HALT rc=26: gate-3 report missing at $GATE3_PATH" \
      "(judge-built off-pod; stage the verdict file, then re-run)" >&2
    exit 26
  fi
  if ! uv run python -c \
    "import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get('passed') else 1)" \
    "$GATE3_PATH"; then
    echo "[dispatch] grid HALT rc=26: gate-3 report at $GATE3_PATH is FAIL —" \
      "fix the instrument/bank per plan §7 before the grid spend" >&2
    exit 26
  fi
  echo "[dispatch] gate3 PASS ($GATE3_PATH)"
}

run_margin_if_pools() {
  # STANDALONE `margin` phase: an explicitly requested margin with no pools
  # staged is a DESIGNED HALT (distinct rc, never a silent skip): stage the
  # judge-built pools (reuse the parent's banked pools or rebuild via
  # scripts/issue2162_judge.py --phase pools -> pools.json -> pod), then
  # re-run `margin` + `upload`.
  if [ ! -f "$POOLS_PATH" ]; then
    echo "[dispatch] margin HALT rc=24: pools file missing at $POOLS_PATH" \
      "(judge-built; see header). Re-run phases: margin, upload." >&2
    exit 24
  fi
  run_fanout_phase margin --pools "$POOLS_PATH"
}

run_margin_opportunistic() {
  # On the `all` chain margin is OPPORTUNISTIC — the pools ride the
  # Batch-API judge SLA, and gating upload/sentinel/teardown on them idles
  # the wide pod at 0% for the SLA tail (#664). Pools present -> margin
  # rides the wide pod now. Absent -> DEFER LOUDLY and proceed: the sentinel
  # carries margin_deferred=true + the recipe (never a silent drop), and the
  # deferred leg runs later on a fresh 1x H100
  # (issue2329_dispatch.sh margin && issue2329_dispatch.sh upload).
  if [ ! -f "$POOLS_PATH" ]; then
    echo "[dispatch] margin DEFERRED: pools file missing at $POOLS_PATH" \
      "(judge-built; Batch-API SLA tail). Proceeding; the sentinel records" \
      "margin_deferred=true + the deferred-leg recipe. Later, once pools" \
      "land: issue2329_dispatch.sh margin && issue2329_dispatch.sh upload."
    return 0
  fi
  run_fanout_phase margin --pools "$POOLS_PATH"
}

run_stage2() {
  # P4c (folded driver phase — no separate stage2 script). A MISSING
  # best-cells file is a designed HALT rc=25 UNLESS --smoke rides the args
  # (the driver then synthesizes a 1-cell selection to keep the stage-2
  # path exercised); an EMPTY survivors list is handled driver-side as
  # SKIP-with-record, rc 0.
  if [ ! -f "$BEST_CELLS_PATH" ]; then
    case " ${COMMON[*]} " in
      *" --smoke "*)
        echo "[dispatch] stage2: best-cells missing at $BEST_CELLS_PATH —" \
          "smoke run proceeds on the driver's synthetic 1-cell selection"
        ;;
      *)
        echo "[dispatch] stage2 HALT rc=25: best-cells file missing at" \
          "$BEST_CELLS_PATH (built by --phase fact_select)." >&2
        exit 25
        ;;
    esac
  fi
  run_fanout_phase stage2 --best-cells "$BEST_CELLS_PATH"
}

run_upload() {
  # r3 C1 fix (c), defense in depth: thread the realized width into the upload
  # phase so cfg.num_workers is never the implicit width-1 default there. The
  # binding fix is driver-side — phase_upload derives each family's width from
  # its OWN done-record set (fix (b)) and NEVER sweeps on this value — but the
  # threaded width keeps the sentinel's recorded num_workers truthful and arms
  # the derived-vs-threaded mismatch warning.
  run_cpu_phase upload --num-workers "$NUM_WORKERS"
}

case "$PHASE" in
  gate0b)
    run_gate0b
    ;;
  import-check)
    run_import_check
    ;;
  bank)
    run_single_gpu_phase bank
    ;;
  pilot)
    # Gate 4 at P2 ENTRY. --num-workers threads the REALIZED width into the
    # projection (parent r1 C2: without it the gate computed at argparse
    # default 1 and refused any realization not ~NUM_WORKERS x faster).
    run_single_gpu_phase pilot --pilot --num-workers "$NUM_WORKERS"
    ;;
  anchors)
    run_fanout_phase anchors
    ;;
  grid)
    require_gate3
    run_fanout_phase grid
    ;;
  margin)
    run_margin_if_pools
    ;;
  fact_tables)
    run_fanout_phase fact_tables
    ;;
  fact_select)
    run_cpu_phase fact_select
    ;;
  stage2)
    run_stage2
    ;;
  upload)
    run_upload
    ;;
  all)
    run_gate0b
    run_import_check
    run_single_gpu_phase bank
    # Gate 4 (pilot) at P2 ENTRY — BEFORE anchors (plan §7/§116): its
    # measured per-rollout wall derives the 2x fences for P2/P3/P4 and the
    # 3x-total refusal fires before any bulk generation spend.
    run_single_gpu_phase pilot --pilot --num-workers "$NUM_WORKERS"
    run_fanout_phase anchors
    require_gate3
    run_fanout_phase grid
    run_margin_opportunistic
    run_fanout_phase fact_tables
    run_cpu_phase fact_select
    run_stage2
    run_upload
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

echo "[phase=done]"
