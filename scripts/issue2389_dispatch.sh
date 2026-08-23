#!/usr/bin/env bash
# Issue #2389 — pod dispatcher for scripts/issue2389_run.py (Qwen3.8-27B, ce-only).
#
# Forked from scripts/issue2329_dispatch.sh. Phases:
#   gate0b          plan §7 gate 0b: pin transformers==5.15.0 into the pod venv,
#                   then assert AutoConfig loads qwen3_5 + 64 resolvable decoder
#                   blocks (driver --gate0b-check; CPU, tiny from-config model)
#   import-check    deferred-import resolution for ALL THREE pod-side drivers
#                   (run / vllm_anchors / share_prefill_gate; CPU, no GPU)
#   bank            P1: v_ce bank + degeneracy guard + injection gate — 8-way
#                   claim queue, one worker per visible GPU (plan E3)
#   pilot           P2 ENTRY (plan §7 gate 4): ONE production-shape timed block
#                   (GPU 0); derives 2x fences and refuses on projected TOTAL
#                   > 3x planned (rc=22)
#   anchors         P2 + gate 4b window: workers 2..N-1 start HF anchors at t0;
#                   worker 0 runs the item-4 vLLM parity leg THEN joins anchors;
#                   worker 1 runs the item-5 shared-prefill equivalence battery
#                   THEN joins anchors; a DETACHED CPU `--leg claim` poll runs
#                   concurrently (killed after anchors — a PASS landing mid-P2
#                   re-routes exactly the not-yet-claimed rest cells: routing
#                   is per CELL at claim time, B8). Anchors shard at CELL
#                   grain via the claim queue (any worker may own any cell);
#                   the recal barrier waits for the full gate-cell slice.
#   anchors-plain   resume path: HF anchors fanout WITHOUT re-running the gate
#                   legs (parity/battery artifacts already staged or waived)
#   vllm-parity / vllm-claim / vllm-production / share-prefill-gate
#                   standalone re-runs of the gate-4b legs
#   grid            P3: claim-file block queue, one worker per visible GPU
#   margin          pools-dependent margin TF legs, one worker per visible GPU
#   cap_report      standalone re-aggregation of realized cap-hit (CPU)
#   capregen-anchors-gate / capregen-anchors-rest / capregen-grid
#                   CONDITIONAL >2%-trigger re-generation (single GPU; NOT on
#                   the `all` chain — the per-cell cap table + gate-slice
#                   recalibration are designed to make these unnecessary)
#   fact_tables     P4a: per-draw judge-free F_act tables (fanned out)
#   fact_select     P4b: audit-filtered Holm-IUT + disjoint-CI selection (CPU)
#   stage2          P4c: layer x dose survivor grid (folded driver phase)
#   upload          P5: bulk HF uploads + pod sentinel (CPU)
#   all             gate0b -> import-check -> bank -> pilot -> anchors(+gate4b)
#                   -> vllm-production -> cap_report -> grid -> margin
#                   (OPPORTUNISTIC) -> fact_tables -> fact_select -> stage2
#                   -> upload
#
# 2389 vs the 2329 parent chain:
#   - bank is FANNED OUT (8-way claim queue, plan E3; the parent ran it 1-GPU).
#   - gate 3 (anchor separation) is ADVISORY (plan S3) and judged VM-side
#     in-window — there is NO dispatcher rc=26 halt before grid.
#   - the gate-4b window (item-4 vLLM parity + item-5 shared-prefill battery)
#     runs CONCURRENT with anchors on workers 0-1 (plan §7 gate 4b / E1).
#   - anchors + grid carry --share-prefill (default auto: armed ONLY on a
#     gate-4b PASS artifact; FAIL-OPEN serial — pin 2).
#   - capregen is CONDITIONAL, not a chained pass (plan §4.7 item 1).
#
# Margin semantics (parent r2 MAJOR 1, unchanged): the pools file is
# judge-built off-pod (Batch-API SLA), so the `all` chain must NEVER park the
# wide pod behind it (#664 idle-burn). Pools staged in time -> margin rides
# the wide pod here; absent -> DEFERRED LOUDLY (sentinel carries
# margin_deferred=true + the recipe); the STANDALONE `margin` phase keeps the
# rc=24 HARD HALT.
#
# Worker count is DERIVED from the realized GPU count (`nvidia-smi -L`) at
# launch — never hardcoded. Each worker gets CUDA_VISIBLE_DEVICES pinned in
# ITS OWN launcher env (never `+gpu_id=N`; the in-process clobber is defeated
# by import-time cuInit — gotchas.md).
#
# UV_NO_SYNC=1 is exported for the whole dispatch: (a) gate0b pins
# transformers==5.15.0 into the pod venv with `uv pip install`, and a bare
# `uv run` would re-sync to uv.lock and silently UNDO the pin on the very
# next invocation; (b) an N-way `uv run` fan-out re-resolving on the MooseFS
# venv is the #1689 FUSE read-wedge trigger. The venv itself was built by
# bootstrap's `uv sync --locked`.
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

DRIVER="scripts/issue2389_run.py"
VLLM_DRIVER="scripts/issue2389_vllm_anchors.py"
GATE4B_DRIVER="scripts/issue2389_share_prefill_gate.py"
OUT_ROOT="${EPM_2389_OUT_ROOT:-/workspace/issue2389_out}"
LOG_DIR="${EPM_2389_LOG_DIR:-/workspace/logs}"
POOLS_PATH="${EPM_2389_POOLS:-$OUT_ROOT/pools.json}"
BEST_CELLS_PATH="${EPM_2389_BEST_CELLS:-$OUT_ROOT/best_cells_actsel.json}"
TRANSFORMERS_PIN="${EPM_2389_TRANSFORMERS_PIN:-5.15.0}"
# Pin 2: 'auto' arms share_prefill ONLY on a gate-4b PASS artifact (FAIL-OPEN
# serial). 'off' forces serial everywhere. There is deliberately NO 'on'.
SHARE_PREFILL_MODE="${EPM_2389_SHARE_PREFILL:-auto}"
PIDFILE="$LOG_DIR/issue-2389-workers.pid"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

# Worker count = realized GPU count (derived, never hardcoded).
# SLURM_GPU_WIDTH_EXEMPT: runpod-pinned pod dispatcher (plan §9 sentinel lane pins the drained runpod lane; never runs on a shared SLURM node, so dedicated-pod nvidia-smi enumeration IS the allocation width).
NUM_WORKERS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
NUM_WORKERS="${NUM_WORKERS:-0}"
if [ "$NUM_WORKERS" -lt 1 ]; then
  NUM_WORKERS=1
fi
echo "[dispatch] phase=$PHASE num_workers=$NUM_WORKERS out_root=$OUT_ROOT share_prefill=$SHARE_PREFILL_MODE"

COMMON=(--out-root "$OUT_ROOT" --log-dir "$LOG_DIR" "$@")

# Thread the smoke flag into the vLLM / battery legs when the dispatch args
# carry --smoke (their CLIs differ from run.py's).
# B9: the vLLM legs' HF sub-legs share the anchors regime fingerprint with
# run.py's workers — the share-prefill mode MUST match the HF workers'.
VLLM_EXTRA=(--share-prefill "$SHARE_PREFILL_MODE")
GATE4B_EXTRA=()
case " ${COMMON[*]} " in
  *" --smoke "*)
    VLLM_EXTRA+=(--smoke)
    GATE4B_EXTRA+=(--tiny --skip-wall)
    ;;
esac

run_gate0b() {
  # Plan §7 gate 0b: the qwen3_5 arch needs transformers==5.15.0 (repo pin
  # 4.57.6 lacks it). Pin the pod venv FIRST, then run the driver's asserts
  # (version, AutoConfig arch, _resolve_decoder_blocks == 64) on CPU. Every
  # failure arm exits through the python driver's rc.
  echo "[phase=gate0b]"
  local log="$LOG_DIR/issue-2389-gate0b.log"
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
  # All three pod-side drivers (a never-smoked phase must not ship an
  # ImportError / args-attribute AttributeError — smoke-architecture Axis 1).
  echo "[phase=import-check]"
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --import-check
  CUDA_VISIBLE_DEVICES="" uv run python "$VLLM_DRIVER" --import-check
  CUDA_VISIBLE_DEVICES="" uv run python "$GATE4B_DRIVER" --import-check
}

run_single_gpu() {
  # <logname> <run.py args...> on GPU 0; rc captured through the tee pipe.
  local name="$1"
  shift
  local log="$LOG_DIR/issue-2389-${name}.log"
  echo "[dispatch] $name -> $log"
  set +e
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" "$@" "${COMMON[@]}" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] $name exited rc=$rc"
    exit "$rc"
  fi
}

run_cpu_phase() {
  # cap_report / fact_select / upload: single CPU process, no GPU pin.
  local phase="$1"
  shift
  local log="$LOG_DIR/issue-2389-${phase}.log"
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
  # bank / anchors / grid / margin / fact_tables / stage2: one worker per
  # visible GPU, PLAIN backgrounded children (no setsid — `wait` must be
  # real; a detached child makes the wave-chain concurrent, gotchas.md
  # #1738). Extra args ("$@") thread in BEFORE the pinned per-worker flags
  # so the pins always win.
  local phase="$1"
  shift
  : > "$PIDFILE"
  local pids=()
  local g
  for ((g = 0; g < NUM_WORKERS; g++)); do
    local log="$LOG_DIR/issue-2389-${phase}-w${g}.log"
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
      tail -n 120 "$LOG_DIR/issue-2389-${phase}-w${g}.log" || true
      rc_all="$rc"
    fi
  done
  if [ "$rc_all" -ne 0 ]; then
    echo "[dispatch] $phase FAILED rc=$rc_all"
    exit "$rc_all"
  fi
}

# ── gate-4b leg commands (each writes its OWN log; used inside chains) ──

_anchor_worker() {
  # HF anchors worker g (CVD-pinned; SAME --num-workers as every sibling so
  # the stripe layout is consistent across early and late joiners).
  local g="$1"
  CUDA_VISIBLE_DEVICES="$g" uv run python "$DRIVER" --phase anchors \
    "${COMMON[@]}" --share-prefill "$SHARE_PREFILL_MODE" \
    --worker-index "$g" --num-workers "$NUM_WORKERS" --gpu-id "$g" \
    > "$LOG_DIR/issue-2389-anchors-w${g}.log" 2>&1
}

_vllm_parity() {
  # Item 4 parity leg: ALL 3 PARITY_CELLS in one invocation (claims written
  # first; vLLM full sets -> vllm_parity/; HF sub-leg serial by design —
  # the matched vLLM-vs-serial-HF comparison).
  local g="$1"
  CUDA_VISIBLE_DEVICES="$g" uv run python "$VLLM_DRIVER" --leg parity \
    --out-root "$OUT_ROOT" --num-workers "$NUM_WORKERS" "${VLLM_EXTRA[@]}" \
    > "$LOG_DIR/issue-2389-vllm-parity.log" 2>&1
}

_vllm_claim() {
  # CPU-only poll of the write repo for the (VM-judged) parity verdict; on
  # PASS extends gates/vllm_cells.json to every cell. Routing is per cell at
  # CLAIM time (B8): a late PASS re-routes exactly the not-yet-HF-claimed
  # cells — work-conserving, never inert.
  CUDA_VISIBLE_DEVICES="" uv run python "$VLLM_DRIVER" --leg claim \
    --out-root "$OUT_ROOT" --num-workers "$NUM_WORKERS" "${VLLM_EXTRA[@]}" \
    > "$LOG_DIR/issue-2389-vllm-claim.log" 2>&1
}

_gate4b_battery() {
  # Item 5 shared-prefill equivalence battery (gate 4b). FAIL-OPEN: a
  # completed battery exits 0 whatever the verdict; the verdict rides
  # gates/share_prefill_equivalence.json, which run.py's --share-prefill
  # auto resolver reads at anchors/grid phase entry.
  local g="$1"
  CUDA_VISIBLE_DEVICES="$g" uv run python "$GATE4B_DRIVER" \
    --out-root "$OUT_ROOT" "${GATE4B_EXTRA[@]}" \
    > "$LOG_DIR/issue-2389-gate4b-battery.log" 2>&1
}

run_anchors_with_gate4b() {
  # Plan §7 gate 4b ∥ early P2 (E1): parity + battery run CONCURRENT with
  # anchors on workers 0-1; workers 2..N-1 start anchors at t0. The claim
  # poll is DETACHED (CPU) and killed once anchors completes.
  if [ "$NUM_WORKERS" -lt 3 ]; then
    echo "[dispatch] gate4b sequential degrade (num_workers=$NUM_WORKERS < 3)"
    set +e
    _vllm_parity 0
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "[dispatch] vllm parity leg exited rc=$rc (log tail below)"
      tail -n 120 "$LOG_DIR/issue-2389-vllm-parity.log" || true
      exit "$rc"
    fi
    set +e
    _gate4b_battery 0
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "[dispatch] gate4b battery exited rc=$rc (log tail below)"
      tail -n 120 "$LOG_DIR/issue-2389-gate4b-battery.log" || true
      exit "$rc"
    fi
    run_fanout_phase anchors --share-prefill "$SHARE_PREFILL_MODE"
    echo "[dispatch] claim leg skipped (anchors complete — every rest cell already HF-done," \
      "so a later PASS would find nothing to generate; vllm-production still runs post-anchors)"
    return 0
  fi

  : > "$PIDFILE"
  local pids=()
  local names=()
  local g
  for ((g = 2; g < NUM_WORKERS; g++)); do
    echo "[dispatch] anchors worker=$g gpu=$g (t0)"
    ( _anchor_worker "$g" ) &
    pids+=("$!")
    names+=("anchors-w${g}")
    echo "$!" >> "$PIDFILE"
  done
  echo "[dispatch] gate4b worker=0: vllm parity -> anchors w0"
  ( _vllm_parity 0 && _anchor_worker 0 ) &
  pids+=("$!")
  names+=("parity+anchors-w0")
  echo "$!" >> "$PIDFILE"
  echo "[dispatch] gate4b worker=1: share-prefill battery -> anchors w1"
  ( _gate4b_battery 1 && _anchor_worker 1 ) &
  pids+=("$!")
  names+=("battery+anchors-w1")
  echo "$!" >> "$PIDFILE"
  _vllm_claim &
  local claim_pid="$!"
  echo "$claim_pid" >> "$PIDFILE"
  echo "[dispatch] vllm claim poll detached (pid=$claim_pid; killed post-anchors)"

  local rc_all=0
  local i
  for i in "${!pids[@]}"; do
    set +e
    wait "${pids[$i]}"
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "[dispatch] gate4b/anchors chain '${names[$i]}' exited rc=$rc"
      rc_all="$rc"
    fi
  done
  kill "$claim_pid" 2>/dev/null || true
  set +e
  wait "$claim_pid" 2>/dev/null
  set -e
  if [ "$rc_all" -ne 0 ]; then
    local f
    for f in "$LOG_DIR"/issue-2389-anchors-w*.log \
      "$LOG_DIR/issue-2389-vllm-parity.log" \
      "$LOG_DIR/issue-2389-gate4b-battery.log"; do
      if [ -f "$f" ]; then
        echo "[dispatch] tail $f"
        tail -n 120 "$f" || true
      fi
    done
    echo "[dispatch] anchors(+gate4b) FAILED rc=$rc_all"
    exit "$rc_all"
  fi
}

run_vllm_production() {
  # Post-anchors: engaged workers generate their claimed anchor cells via
  # vLLM (two sweeps: vLLM gen -> engines down -> HF capture); a worker
  # with no owned cells in the claim set exits quickly.
  : > "$PIDFILE"
  local pids=()
  local g
  for ((g = 0; g < NUM_WORKERS; g++)); do
    local log="$LOG_DIR/issue-2389-vllm-production-w${g}.log"
    echo "[dispatch] vllm production worker=$g gpu=$g -> $log"
    CUDA_VISIBLE_DEVICES="$g" uv run python "$VLLM_DRIVER" --leg production \
      --out-root "$OUT_ROOT" --worker-index "$g" --num-workers "$NUM_WORKERS" \
      "${VLLM_EXTRA[@]}" > "$log" 2>&1 &
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
      echo "[dispatch] vllm production worker=$g exited rc=$rc (log tail below)"
      tail -n 120 "$LOG_DIR/issue-2389-vllm-production-w${g}.log" || true
      rc_all="$rc"
    fi
  done
  if [ "$rc_all" -ne 0 ]; then
    echo "[dispatch] vllm production FAILED rc=$rc_all"
    exit "$rc_all"
  fi
}

run_margin_if_pools() {
  # STANDALONE `margin` phase: an explicitly requested margin with no pools
  # staged is a DESIGNED HALT (distinct rc, never a silent skip): stage the
  # judge-built pools, then re-run `margin` + `upload`.
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
  # (issue2389_dispatch.sh margin && issue2389_dispatch.sh upload).
  if [ ! -f "$POOLS_PATH" ]; then
    echo "[dispatch] margin DEFERRED: pools file missing at $POOLS_PATH" \
      "(judge-built; Batch-API SLA tail). Proceeding; the sentinel records" \
      "margin_deferred=true + the deferred-leg recipe. Later, once pools" \
      "land: issue2389_dispatch.sh margin && issue2389_dispatch.sh upload."
    return 0
  fi
  run_fanout_phase margin --pools "$POOLS_PATH"
}

run_stage2() {
  # P4c (folded driver phase). A MISSING best-cells file is a designed HALT
  # rc=25 UNLESS --smoke rides the args (the driver then synthesizes a
  # 1-cell selection to keep the stage-2 path exercised); an EMPTY survivors
  # list is handled driver-side as SKIP-with-record, rc 0.
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
  # Thread the realized width into the upload phase so cfg.num_workers is
  # never the implicit width-1 default there (parent r3 C1 fix (c); the
  # binding fix is driver-side — phase_upload derives each family's width
  # from its OWN done-record set).
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
    # P1 is an 8-way claim queue (plan E3) — fanned out, unlike the parent.
    run_fanout_phase bank
    ;;
  pilot)
    # Gate 4 at P2 ENTRY. --num-workers threads the REALIZED width into the
    # projection (parent r1 C2: without it the gate computed at argparse
    # default 1 and refused any realization not ~NUM_WORKERS x faster).
    run_single_gpu pilot --phase grid --pilot --num-workers "$NUM_WORKERS"
    ;;
  anchors)
    run_anchors_with_gate4b
    ;;
  anchors-plain)
    run_fanout_phase anchors --share-prefill "$SHARE_PREFILL_MODE"
    ;;
  vllm-parity)
    set +e
    _vllm_parity 0
    rc=$?
    set -e
    tail -n 40 "$LOG_DIR/issue-2389-vllm-parity.log" || true
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
    ;;
  vllm-claim)
    set +e
    _vllm_claim
    rc=$?
    set -e
    tail -n 40 "$LOG_DIR/issue-2389-vllm-claim.log" || true
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
    ;;
  vllm-production)
    run_vllm_production
    ;;
  share-prefill-gate)
    set +e
    _gate4b_battery 0
    rc=$?
    set -e
    tail -n 40 "$LOG_DIR/issue-2389-gate4b-battery.log" || true
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
    ;;
  grid)
    # No gate-3 halt: anchor separation is ADVISORY in this run (plan S3) —
    # the VM-side judge writes gates/anchor_separation_report.json and the
    # run digest surfaces it; the run continues regardless.
    run_fanout_phase grid --share-prefill "$SHARE_PREFILL_MODE"
    ;;
  margin)
    run_margin_if_pools
    ;;
  cap_report)
    run_cpu_phase cap_report
    ;;
  capregen-anchors-gate)
    # Round-6 (concern pilot-reuse-runtime-domain): thread the realized
    # worker width — the round-5-J adoption path validates the pilot
    # report's num_workers, and the parser-default width (1) FOREIGN-raised
    # against the width-N report BEFORE any regeneration, making the
    # registered >2%/cell cap-hit remedy unrunnable as shipped.
    run_single_gpu capregen-anchors-gate --phase capregen \
      --capregen-scope anchors --capregen-batch gate --num-workers "$NUM_WORKERS"
    ;;
  capregen-anchors-rest)
    run_single_gpu capregen-anchors-rest --phase capregen \
      --capregen-scope anchors --capregen-batch rest --num-workers "$NUM_WORKERS"
    ;;
  capregen-grid)
    run_single_gpu capregen-grid --phase capregen --capregen-scope grid
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
    run_fanout_phase bank
    # Gate 4 (pilot) at P2 ENTRY — BEFORE anchors (plan §7): its measured
    # per-rollout wall derives the 2x fences and the 3x-total refusal fires
    # before any bulk generation spend.
    run_single_gpu pilot --phase grid --pilot --num-workers "$NUM_WORKERS"
    run_anchors_with_gate4b
    run_vllm_production
    run_cpu_phase cap_report
    run_fanout_phase grid --share-prefill "$SHARE_PREFILL_MODE"
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
