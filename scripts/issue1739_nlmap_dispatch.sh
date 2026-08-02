#!/usr/bin/env bash
# issue-1739 NONLINEAR-MAP arm round (nl-arm6 / nl-arm7 / nl-arm8).
#
# The main #1739 grid fits a LINEAR (ridge) context->answer map and scores
# arms 6/7/8 off it. This round fits MLP + Nyström-kernel-ridge maps on the
# SAME unlabeled #1092 U pool and re-scores the same three map-family arms,
# so nonlinear-vs-linear is comparable CELL-FOR-CELL: every cell key
# (regime x u_size x budget x seed x draw), every eval rung, every fold
# scheme, and the bootstrap CI helper are the reviewed production path —
# only `--map-kind` changes.
#
# Reuse (no new fit math): the two nonlinear families are #779's N1M fitters
# (`scripts/issue779_ffc_n1m_fits.py` fit_mlp / fit_krr_nystrom / apply_map),
# reached through `fits.fit_nonlinear_map`; the arms come from the reviewed
# `experiments/issue_1739/arms.py` registry via `--arms`.
#
# Inputs are staged by `scripts/issue1739_leg2.sh` (raw completions, the six
# capture-store tars, reconstructed contexts, dv_dataset) — idempotent, so a
# fresh instance self-stages and a re-run skips. The #1092 U-store stages
# itself inside the fits script (`store_io.stage_u_store`).
#
# Per-MAP-KIND out-roots: `--map-kind` is a regime key for the resume/output
# state, so each kind owns its own out-root (crash-fix-rounds.md § Per-leg
# out-roots for regime-keyed drivers) — a shared root would let one kind's
# `all_arms_spearman.json` overwrite the other's.
#
# Counts-only logging; no corpus content printed.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

STORE_ROOT="data/issue_1739/store"
RESULTS_ROOT="eval_results/issue_1739"
# Out-root override (new-arm-round item 3b): the nlood leg keys its results
# under eval_results/issue_1739/new_arm_round/nlood/<behavior>/<kind> — never a
# shared root with the committed nlmap results (per-leg out-roots,
# crash-fix-rounds.md). Default unchanged (committed nlmap lanes untouched).
NL_ROOT="${EPM_I1739_NL_ROOT:-$RESULTS_ROOT/nonlinear_map}"
TENSORS_ROOT="analysis_tensors/issue_1739"
U_STORE_DIR="data/issue_1739/hf_dl/u_store"
LOG_DIR="/workspace/logs"
BRANCH="${EPM_I1739_BRANCH:-issue-1739}"

# The three map-family arms. nl-arm6 = map -> E1 persona-vector projection;
# nl-arm7 = ridge readout trained on PREDICTED answer vectors; nl-arm8 =
# ridge readout trained on REAL answer vectors, applied to map predictions.
NL_ARMS="arm6_map_proj_e1 arm7_map_ridge_pred arm8_map_ridge_true"

BEHAVIORS="${EPM_I1739_NL_BEHAVIORS:-evil sycophancy hallucination}"
KINDS="${EPM_I1739_NL_KINDS:-mlp kernel}"
# Transfer-roster pin (new-arm-round item 3b, plan v8 HARD PRECONDITION):
# resolve_transfer_roster(None) defaults to the WIDE roster (10 arms incl. the
# expensive arm-5 MLP), so an unpinned nlood transfer leg would fan out far
# past its two map-readout arms. Space-separated arm slugs; empty (default)
# keeps the committed behavior (no --transfer-arms flag emitted).
NL_TRANSFER_ARMS="${EPM_I1739_NL_TRANSFER_ARMS:-}"
# Grid defaults hold FULL parity with the main grid's fits phase so every
# nonlinear cell keys to a linear sibling. A budget-driven descope narrows
# DRAWS / REGIMES via these env knobs and is recorded as a deviation — never
# silently, and never by changing a cell's VALUES (that would unmatch it).
USIZES="${EPM_I1739_NL_USIZES:-250 5000 full}"
SEEDS="${EPM_I1739_NL_SEEDS:-0 1 2}"
DRAWS="${EPM_I1739_NL_DRAWS:-0 1 2 3 4}"
# Pilot gate budget. The ROUND ceiling is ~5 GPU-h across all
# (behavior x kind) invocations; with 6 invocations that is ~0.83 h each, and
# the pilot runs on the HEAVIEST behavior (3 regimes), so a per-invocation pass
# bounds the lighter ones too. abort-mult 1 makes the gate enforce that share
# directly rather than the fits default 3x plan-§9 re-size fence.
PLAN_WALL_H="${EPM_I1739_NL_PLAN_WALL_H:-0.83}"
PILOT_ABORT_MULT="${EPM_I1739_NL_PILOT_ABORT_MULT:-1}"
# Comma/space list of phases, or "all". Two-leg dispatch (stage,pilot then
# fits,collect,upload) keeps the round-level STOP decision on measured numbers.
# The fan-out shape adds `prefetch` (phase A: fit every map ONCE) and
# `stage_maps` (per lane: pull those payloads down before scoring).
PHASE="${EPM_I1739_NL_PHASE:-all}"

# ---- phase-A prefetch knobs -------------------------------------------------
# The prefetch runs under ONE behavior because the map is BEHAVIOR-INDEPENDENT
# (fit on the shared #1092 U pool; _save_map omits the behavior from the
# filename). Its labeled/e1 stores just have to exist for the script to run.
PREFETCH_BEHAVIOR="${EPM_I1739_NL_PREFETCH_BEHAVIOR:-evil}"
PREFETCH_ARM="${EPM_I1739_NL_PREFETCH_ARM:-arm6_map_proj_e1}"
PREFETCH_REGIME="${EPM_I1739_NL_PREFETCH_REGIME:-e1}"
# Cheapest budget in the grid (measured plain group wall ~4.87 s vs 46.2/140.1
# for the larger rungs). This is the knob behavior_budgets() cannot express.
PREFETCH_BUDGETS="${EPM_I1739_NL_PREFETCH_BUDGETS:-250}"
PREFETCH_DRAW="${EPM_I1739_NL_PREFETCH_DRAW:-0}"
PREFETCH_N_BOOT="${EPM_I1739_NL_PREFETCH_N_BOOT:-20}"
PREFETCH_N_PERM="${EPM_I1739_NL_PREFETCH_N_PERM:-20}"
# seeds[0] drives whitening, the map fit AND the subsampled U rung's ROW DRAW,
# so the prefetch MUST pin it to the lanes' first seed or the u=250 payload is a
# map over different rows (and _load_nl_map's row-COUNT guard cannot see that;
# its map_seed guard is what catches a mismatch). Derived, never hand-set.
PREFETCH_SEED="${SEEDS%% *}"

# ---- composition-factor knobs (phase `compose`; scope addendum) -------------
# §4b composition crossings for the behaviors the MAIN grid gated OUT: that
# dispatcher appends --compose only `if [ "$b" = "evil" ]`, so evil has committed
# compose cells and the other two have none. This phase runs the SAME crossings
# (constants.COMPOSITION_F_U x COMPOSITION_F_L = {0, 0.5} x {0, 1}) at U=5000 for
# hallucination + sycophancy.
#
# LINEAR map, matching evil's committed compose cells so the comparison is
# cell-for-cell. That is WHY this is its own invocation rather than extra cells
# inside a lane: --map-kind is ONE flag per process, so a linear compose cell
# cannot ride inside an mlp/kernel invocation. It still rides the lane's INSTANCE
# (no new pods) and gets its OWN out-root (--map-kind is a resume/output regime
# key — crash-fix-rounds.md § Per-leg out-roots) and its OWN pilot fence, which
# is what keeps the nonlinear lanes' fence untouched.
COMPOSE_BEHAVIORS="${EPM_I1739_NL_COMPOSE_BEHAVIORS:-hallucination sycophancy}"
COMPOSE_MAP_KIND="${EPM_I1739_NL_COMPOSE_MAP_KIND:-linear}"
COMPOSE_U_SIZE="${EPM_I1739_NL_COMPOSE_U_SIZE:-5000}"
COMPOSE_REGIME="${EPM_I1739_NL_COMPOSE_REGIME:-e1}"
# compose_run_specs enumerates plain rungs too (there is no compose-ONLY mode),
# so pin the plain axis to the CHEAPEST rung: it is unavoidable overhead, not a
# deliverable (those linear plain cells are already committed by the main grid).
COMPOSE_PLAIN_USIZES="${EPM_I1739_NL_COMPOSE_PLAIN_USIZES:-250}"
# L-anchors. Default = the behavior's FULL ladder, per the addendum ("L in
# {250, 2500} plus any larger L whose residual eliciting pool is non-empty"): at
# the max anchor the f_l=0 cells have an EMPTY residual pool and are recorded as
# compose_skips (evil's committed labels show exactly this — fu0.5_fl0.0 present
# at L250/L2500, absent at L8000), while f_u=0 and f_l=1 still fill. Trim to
# '250 2500' to drop the max anchor: it is the single most expensive one
# (see scripts/issue1739_nlmap_project.py --compose for the arithmetic).
compose_budgets() {
  if [ -n "${EPM_I1739_NL_COMPOSE_BUDGETS:-}" ]; then echo "$EPM_I1739_NL_COMPOSE_BUDGETS"; return; fi
  behavior_budgets "$1"
}
# Per-behavior compose fence, DERIVED from the measured LINEAR basis (the
# committed per-behavior pilot_report.json) rather than hand-copied: the
# projector mirrors _run_pilot's own counters, so the fence tracks the anchor
# set compose_budgets() actually passes. Fail-loud — a fence we cannot derive
# is not a fence we may guess (plan-compute-sizing.md § measured basis).
compose_plan_wall_h() {
  if [ -n "${EPM_I1739_NL_COMPOSE_PLAN_WALL_H:-}" ]; then
    echo "$EPM_I1739_NL_COMPOSE_PLAN_WALL_H"
    return
  fi
  uv run python scripts/issue1739_nlmap_project.py \
    --compose-plan-wall-for "$1" --compose-anchors $(compose_budgets "$1")
}

# ---- per-lane map staging knobs (phase `stage_maps`) -----------------------
STAGE_MAPS_VARIANTS="${EPM_I1739_NL_STAGE_VARIANTS:-prefix_end context_end}"
# U-rung LABELS as they appear in the payload filename ('full' stays 'full').
stage_map_u_labels() {
  local out=() tok
  for tok in $USIZES; do out+=("$tok"); done
  printf '%s\n' "${out[@]}"
}

# Phases that run ONLY when named explicitly — never under PHASE=all. Both
# belong to the fan-out shape (phase A publishes maps; each lane stages them),
# so folding them into "all" would change the legacy single-box dispatch: a
# single box has nothing on the Hub yet, and stage_maps is fail-loud by design.
OPT_IN_PHASES="prefetch stage_maps compose"

want_phase() {
  # want_phase <name> — true when PHASE lists <name>, or PHASE is "all" and
  # <name> is not opt-in-only.
  local p
  case "$PHASE" in
    all)
      for p in $OPT_IN_PHASES; do
        [ "$p" = "$1" ] && return 1
      done
      return 0
      ;;
  esac
  for p in ${PHASE//,/ }; do
    [ "$p" = "$1" ] && return 0
  done
  return 1
}

FITS_DEVICE="${EPM_I1739_FITS_DEVICE:-}"
if [ -z "$FITS_DEVICE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    FITS_DEVICE=cuda
  else
    FITS_DEVICE=cpu
  fi
fi

mkdir -p "$LOG_DIR"
echo "[nlmap] start $(date -u +%FT%TZ) repo_root=$REPO_ROOT device=$FITS_DEVICE phase=$PHASE"
echo "[nlmap] behaviors='$BEHAVIORS' kinds='$KINDS' usizes='$USIZES' seeds='$SEEDS' draws='$DRAWS'"

# Per-behavior grid axes — copied from issue1739_dispatch.sh so the nonlinear
# cells key to the SAME (regime, budget) values the linear grid used.
behavior_budgets() {
  case "$1" in
    evil) echo "250 2500 8000" ;;
    *) echo "250 2500 16000" ;;
  esac
}
behavior_regimes() {
  if [ -n "${EPM_I1739_NL_REGIMES:-}" ]; then echo "$EPM_I1739_NL_REGIMES"; return; fi
  case "$1" in
    hallucination) echo "e1" ;;
    *) echo "e1 e2 e2p" ;;
  esac
}

store_args() {
  # store_args <behavior> — the stores/paths/device every invocation needs to run
  # at all, independent of which map family or grid it scores.
  local b="$1"
  printf '%s\n' \
    --behavior "$b" \
    --labeled-store "$STORE_ROOT/${b}_labeling" \
    --dv-json "$RESULTS_ROOT/dv_dataset/$b/labeling.json" \
    --u-store "$U_STORE_DIR" \
    --e1-store "$STORE_ROOT/${b}_extraction" \
    --tensors-root "$TENSORS_ROOT" \
    --device "$FITS_DEVICE" \
    --config config_a
}

map_args() {
  # map_args <behavior> <kind> — every flag that determines the FITTED MAP, plus
  # the stores it needs to run at all. Shared by fits_args and prefetch_args so a
  # prefetched map is bit-for-bit the map the lane would have fit: any future
  # map-affecting flag added here propagates to BOTH by construction.
  #
  # Map-determining set: --map-kind, --u-store, --u-sizes, --tensors-root, and
  # (implicitly) --variant (default 'both') + seeds[0], which drives whitening,
  # the map fit AND the subsampled U rung's ROW DRAW. That last one is why
  # prefetch_args pins seeds[0] to the lane's first seed.
  local b="$1" kind="$2"
  store_args "$b"
  printf '%s\n' \
    --map-kind "$kind" \
    --u-sizes $USIZES
}

fits_args() {
  # fits_args <behavior> <kind> — the production fits invocation, subset to
  # the map-family arms with a per-kind out-root. EPM_I1739_NL_TRANSFER_ARMS
  # (when set) pins the transfer roster explicitly — without it the transfer
  # leg resolves the WIDE default (arms.resolve_transfer_roster(None)).
  local b="$1" kind="$2"
  map_args "$b" "$kind"
  printf '%s\n' \
    --out-root "$NL_ROOT/$b/$kind" \
    --arms $NL_ARMS \
    --transfer
  if [ -n "$NL_TRANSFER_ARMS" ]; then
    printf '%s\n' --transfer-arms $NL_TRANSFER_ARMS
  fi
  printf '%s\n' \
    --regimes $(behavior_regimes "$b") \
    --budgets $(behavior_budgets "$b") \
    --draws $DRAWS \
    --seeds $SEEDS \
    --n-boot 500 \
    --n-perm 500
}

prefetch_args() {
  # prefetch_args <kind> — phase-A maps-only invocation: the SAME map-determining
  # flags as a lane, with the cheapest grid that still walks every map key.
  #
  # It fits all $USIZES x 2 variants map keys for $kind (the whole path-2 map set)
  # and pays ~one 250-budget unit group per key for the arm machinery it cannot
  # skip (~4.9 s measured), against ~2651 s per map fit. Deliberately NOT
  # map-affecting, so the payloads are identical to a lane's:
  #   * --out-root THROWAWAY (its arm numbers are discarded; only maps/ survives,
  #     and maps/ lives under --tensors-root, not --out-root)
  #   * ONE regime, ONE arm, ONE budget, ONE draw, small null/boot budgets
  #   * NO --transfer (the eval-rung read is per-BEHAVIOR and belongs to the lanes)
  #   * --seeds pinned to the lanes' FIRST seed (see map_args)
  # No --pilot: the pilot gate is a $PLAN_WALL_H fence sized for a full lane, and
  # this phase is ~1/100th of one — it is structurally exempt, not bypassed.
  local kind="$1"
  map_args "$PREFETCH_BEHAVIOR" "$kind"
  printf '%s\n' \
    --out-root "$NL_ROOT/_prefetch/$kind" \
    --arms "$PREFETCH_ARM" \
    --regimes "$PREFETCH_REGIME" \
    --budgets $PREFETCH_BUDGETS \
    --draws "$PREFETCH_DRAW" \
    --seeds "$PREFETCH_SEED" \
    --n-boot "$PREFETCH_N_BOOT" \
    --n-perm "$PREFETCH_N_PERM"
}

compose_args() {
  # compose_args <behavior> — the §4b composition sub-grid for ONE behavior.
  #
  # Deliberately NOT map_args: the map kind is LINEAR here (evil's committed
  # compose family) and the plain U axis is pinned to the cheapest rung, so
  # sharing map_args would either drag the nonlinear kind in or re-run the whole
  # nonlinear U ladder. Composition cells draw their OWN pool
  # (_u_pool_for_spec's f_u branch: generic half from the #1092 store, eliciting
  # half from the behavior's labeling capture store, f_l gating overlap with the
  # L cell's contexts), so their map key is per-cell and each fits its own map.
  #
  # No --transfer: compose specs contribute ZERO transfer units by construction
  # (compose_pilot_report counts transfer over plain specs only), so passing it
  # would buy nothing and make the throwaway plain rungs pay for it.
  local b="$1"
  store_args "$b"
  printf '%s\n' \
    --map-kind "$COMPOSE_MAP_KIND" \
    --u-sizes $COMPOSE_PLAIN_USIZES \
    --out-root "$NL_ROOT/$b/compose_$COMPOSE_MAP_KIND" \
    --arms $NL_ARMS \
    --regimes "$COMPOSE_REGIME" \
    --budgets $(compose_budgets "$b") \
    --draws 0 \
    --seeds 0 \
    --compose \
    --compose-u-size "$COMPOSE_U_SIZE" \
    --n-boot 500 \
    --n-perm 500
}

# Definitions-only escape hatch: with EPM_I1739_NL_DEFS_ONLY set, sourcing this
# script yields the config + arg builders and runs NO phase body and NOT the
# terminal sentinel/`[phase=done]` tail. That lets the tests assert on the
# SHIPPING arg composition (tests/test_issue1739_nlmap.py) instead of a copy that
# can drift from it. Unset in production, so the live path is untouched.
if [ -n "${EPM_I1739_NL_DEFS_ONLY:-}" ]; then
  return 0 2>/dev/null || exit 0
fi

# ---- stage -----------------------------------------------------------------
if want_phase stage; then
  echo "[nlmap] phase=stage: pre-staging inputs via issue1739_leg2.sh"
  bash scripts/issue1739_leg2.sh
  for b in $BEHAVIORS; do
    for s in "${b}_labeling" "${b}_extraction"; do
      [ -d "$STORE_ROOT/$s" ] || { echo "[nlmap] FATAL: store $s missing after stage" >&2; exit 1; }
    done
    [ -f "$RESULTS_ROOT/dv_dataset/$b/labeling.json" ] \
      || { echo "[nlmap] FATAL: dv_dataset/$b missing after stage" >&2; exit 1; }
  done
  echo "[nlmap] phase=stage: complete ($(date -u +%FT%TZ))"
fi

# ---- prefetch (phase A: fit every map ONCE) --------------------------------
# Amortization: a map key costs ~2651 s to fit and is identical across the 3
# behaviors, so the 6-lane fan-out must NOT re-fit it 6 times. This phase fits
# all (2 variants x |USIZES| rungs) keys per kind, persists them under
# $TENSORS_ROOT/maps/ (each gated by the fits-side round-trip save->load->apply
# check), and the `upload` phase publishes them for the lanes' `stage_maps`.
# Runs the KINDS in PARALLEL (one process per kind, distinct payload filenames;
# the shared r_B destination is PID-tmp-safe as of this round) — sequential
# would double phase A's wall.
if want_phase prefetch; then
  echo "[nlmap] phase=prefetch: fitting map keys once per kind ($KINDS) $(date -u +%FT%TZ)"
  echo "[nlmap] phase=prefetch: behavior=$PREFETCH_BEHAVIOR usizes='$USIZES' seed=$PREFETCH_SEED"
  read -r -a _pf_kinds <<< "$KINDS"
  # Parallel width = min(visible GPUs, #kinds). NEVER over-subscribe: two
  # concurrent full-U map fits on ONE device is a co-residency OOM, and a
  # modulo-pinned wave would do exactly that. 0 GPUs (CPU device) => width 1.
  NGPU=0
  if [ "$FITS_DEVICE" != "cpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
    NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
  fi
  PF_WIDTH="${EPM_I1739_NL_PREFETCH_WIDTH:-0}"
  if [ "$PF_WIDTH" -le 0 ]; then
    if [ "$NGPU" -gt 0 ] && [ "$NGPU" -lt "${#_pf_kinds[@]}" ]; then
      PF_WIDTH="$NGPU"
    elif [ "$NGPU" -gt 0 ]; then
      PF_WIDTH="${#_pf_kinds[@]}"
    else
      PF_WIDTH=1
    fi
  fi
  echo "[nlmap] phase=prefetch: ngpu=$NGPU width=$PF_WIDTH kinds=${#_pf_kinds[@]}"
  pf_rc=0
  for ((base = 0; base < ${#_pf_kinds[@]}; base += PF_WIDTH)); do
    pf_pids=()
    pf_slot_kinds=()
    for ((slot = 0; slot < PF_WIDTH && base + slot < ${#_pf_kinds[@]}; slot++)); do
      kind="${_pf_kinds[base + slot]}"
      mapfile -t _pfa < <(prefetch_args "$kind")
      mkdir -p "$NL_ROOT/_prefetch/$kind"
      pf_log="$LOG_DIR/issue-1739-nlmap-prefetch-$kind.log"
      # CVD pinned in the LAUNCHER env, one physical GPU per concurrent kind
      # (the in-process device string is just "cuda", so the launcher env is the
      # ONLY thing that separates them — gotchas.md CVD-clobber family).
      if [ "$NGPU" -gt 0 ]; then
        CUDA_VISIBLE_DEVICES="$slot" uv run python scripts/issue1739_fits.py "${_pfa[@]}" \
          > "$pf_log" 2>&1 &
      else
        uv run python scripts/issue1739_fits.py "${_pfa[@]}" > "$pf_log" 2>&1 &
      fi
      pf_pids+=("$!")
      pf_slot_kinds+=("$kind")
      echo "[nlmap] phase=prefetch: $kind launched pid=${pf_pids[-1]} cvd=${slot} log=$pf_log"
    done
    for i in "${!pf_pids[@]}"; do
      if wait "${pf_pids[$i]}"; then
        echo "[nlmap] phase=prefetch: ${pf_slot_kinds[$i]} OK ($(date -u +%FT%TZ))"
      else
        rc=$?
        echo "[nlmap] phase=prefetch: ${pf_slot_kinds[$i]} FAILED rc=$rc — see" \
          "$LOG_DIR/issue-1739-nlmap-prefetch-${pf_slot_kinds[$i]}.log" >&2
        tail -n 40 "$LOG_DIR/issue-1739-nlmap-prefetch-${pf_slot_kinds[$i]}.log" >&2 || true
        pf_rc=$rc
      fi
    done
  done
  [ "$pf_rc" -eq 0 ] || { echo "[nlmap] FATAL: prefetch failed" >&2; exit "$pf_rc"; }
  echo "[nlmap] phase=prefetch: payloads under $TENSORS_ROOT/maps:"
  ls -la "$TENSORS_ROOT/maps" || true
  echo "[nlmap] phase=prefetch: complete ($(date -u +%FT%TZ))"
fi

# ---- stage_maps (per lane: pull phase-A payloads down before scoring) -------
# Without this a lane's _load_nl_map finds nothing local and silently re-fits
# every map — the amortization phase A paid for evaporates. Fail-loud.
if want_phase stage_maps; then
  echo "[nlmap] phase=stage_maps: staging map payloads -> $TENSORS_ROOT/maps"
  mapfile -t _ul < <(stage_map_u_labels)
  uv run python scripts/issue1739_nlmap_stage_maps.py \
    --tensors-root "$TENSORS_ROOT" \
    --kinds $KINDS \
    --variants $STAGE_MAPS_VARIANTS \
    --u-labels "${_ul[@]}" \
    --map-seed "$PREFETCH_SEED" \
    --out "$NL_ROOT/stage_maps_report.json"
  echo "[nlmap] phase=stage_maps: complete ($(date -u +%FT%TZ))"
fi

# ---- pilot -----------------------------------------------------------------
# MEASURED 1-cell pilot through the PRODUCTION entrypoint at production shape
# (plan-compute-sizing.md § Per-cell fit phases — an asserted per-call cost is
# never a sizing basis). The fits script's own `--pilot` times one cell and
# projects the grid wall; rc=7 means the projection exceeds PLAN_WALL_H, which
# this round treats as STOP-and-report, not a silent descope.
# EVERY (behavior, kind) invocation is gated, not just the first: the grids
# differ per behavior (sycophancy's top budget is 16000 vs evil's 8000), so one
# behavior's projection does NOT bound another's. Pilot unit-groups RESUME into
# the full run, so gating all six costs no extra wall.
if want_phase pilot; then
  for b in $BEHAVIORS; do
    for kind in $KINDS; do
      echo "[nlmap] phase=pilot $b/$kind vs plan_wall_h=$PLAN_WALL_H x mult=$PILOT_ABORT_MULT"
      set +e
      mapfile -t _pa < <(fits_args "$b" "$kind")
      uv run python scripts/issue1739_fits.py "${_pa[@]}" --pilot \
        --plan-wall-h "$PLAN_WALL_H" --pilot-abort-mult "$PILOT_ABORT_MULT"
      prc=$?
      set -e
      if [ "$prc" -eq 7 ]; then
        echo "[nlmap] PILOT REFUSED (rc=7) at $b/$kind: projected wall exceeds" \
          "${PILOT_ABORT_MULT}x${PLAN_WALL_H}h (round ceiling ~5 GPU-h / 6 invocations)." >&2
        echo "[nlmap] STOP — reporting instead of launching the grid;" \
          "see $NL_ROOT/$b/$kind/pilot_report.json." >&2
        exit 7
      fi
      if [ "$prc" -eq 9 ]; then
        echo "[nlmap] RSS-GUARD REFUSED (rc=9) at $b/$kind: projected peak host RAM" \
          "exceeds this box — see $NL_ROOT/$b/$kind/rss_guard_report.json (designed" \
          "halt; relaunch on a 170 GB a2-ultragpu-1g box: --min-gpu-mem-gb > 38)" >&2
        exit 9
      fi
      [ "$prc" -eq 0 ] || { echo "[nlmap] FATAL: pilot $b/$kind exited rc=$prc" >&2; exit "$prc"; }
      echo "[nlmap] phase=pilot $b/$kind: PASS ($(date -u +%FT%TZ))"
    done
  done
  echo "[nlmap] phase=pilot: ALL PASS ($(date -u +%FT%TZ))"
fi

# ---- fits ------------------------------------------------------------------
if want_phase fits; then
  for b in $BEHAVIORS; do
    for kind in $KINDS; do
      echo "[nlmap] phase=fits behavior=$b kind=$kind start $(date -u +%FT%TZ)"
      mapfile -t _fa < <(fits_args "$b" "$kind")
      uv run python scripts/issue1739_fits.py "${_fa[@]}"
      echo "[nlmap] phase=fits behavior=$b kind=$kind done $(date -u +%FT%TZ)"
    done
  done
fi

# ---- compose (scope addendum: §4b crossings for hallu + syc) ---------------
# Rides a lane's INSTANCE but is its OWN invocation (LINEAR map — see the knob
# block) with its OWN out-root and OWN pilot fence, so the lane's nonlinear
# projection/fence is untouched. Only the behaviors in BOTH $BEHAVIORS and
# $COMPOSE_BEHAVIORS run, so a lane pinned to one behavior runs only its own
# cells and the 6-lane fan-out never duplicates them.
if want_phase compose; then
  for b in $BEHAVIORS; do
    case " $COMPOSE_BEHAVIORS " in
      *" $b "*) ;;
      *)
        echo "[nlmap] phase=compose: $b not in COMPOSE_BEHAVIORS ('$COMPOSE_BEHAVIORS') — skip"
        continue
        ;;
    esac
    mapfile -t _ca < <(compose_args "$b")
    _cpw="$(compose_plan_wall_h "$b")"
    echo "[nlmap] phase=compose $b start $(date -u +%FT%TZ) map_kind=$COMPOSE_MAP_KIND" \
      "u_size=$COMPOSE_U_SIZE anchors='$(compose_budgets "$b")' fence=${_cpw}h"
    # Same designed-halt contract as the nonlinear pilot: rc=7 is a fenced STOP
    # with a report, never a bare crash.
    set +e
    uv run python scripts/issue1739_fits.py "${_ca[@]}" --pilot \
      --plan-wall-h "$_cpw" --pilot-abort-mult "$PILOT_ABORT_MULT"
    prc=$?
    set -e
    if [ "$prc" -eq 7 ]; then
      echo "[nlmap] COMPOSE PILOT REFUSED (rc=7) at $b: projected wall exceeds" \
        "${PILOT_ABORT_MULT}x${_cpw}h — see" \
        "$NL_ROOT/$b/compose_$COMPOSE_MAP_KIND/pilot_report.json" >&2
      exit 7
    fi
    if [ "$prc" -eq 9 ]; then
      echo "[nlmap] RSS-GUARD REFUSED (rc=9) at $b compose: projected peak host RAM" \
        "exceeds this box — designed halt; relaunch on a 170 GB a2-ultragpu-1g box" >&2
      exit 9
    fi
    [ "$prc" -eq 0 ] || { echo "[nlmap] FATAL: compose pilot $b rc=$prc" >&2; exit "$prc"; }
    uv run python scripts/issue1739_fits.py "${_ca[@]}"
    # The compose cells ARE the deliverable here, so fail loud if the summary is
    # absent; the max-anchor f_l=0 skips are recorded INSIDE it (compose_skips),
    # which is the expected shape, not a failure.
    _csum="$NL_ROOT/$b/compose_$COMPOSE_MAP_KIND/arm_results/all_arms_spearman.json"
    [ -f "$_csum" ] || { echo "[nlmap] FATAL: compose $b produced no $_csum" >&2; exit 1; }
    echo "[nlmap] phase=compose $b done $(date -u +%FT%TZ); realized compose labels:"
    grep -o 'compose[0-9]*_fu[0-9.]*_fl[0-9.]*_L[0-9]*' "$_csum" | sort -u | sed 's/^/  /' || true
  done
fi

# ---- collect map_quality.json ---------------------------------------------
# Derived from the frozen `maps/*.pt` metas (which carry the held-out
# r2_map + identity+bias baseline + kNN retrieval per layer), so the standing
# mapping-companion reads survive without re-running any fit.
if want_phase collect; then
  echo "[nlmap] phase=collect: map_quality.json"
  uv run python scripts/issue1739_nlmap_collect.py \
    --tensors-root "$TENSORS_ROOT" \
    --out "$NL_ROOT/map_quality.json" \
    --kinds $KINDS
fi

# ---- upload ----------------------------------------------------------------
# Split so the fan-out can pick a leg: phase A publishes the map payloads
# (`upload_tensors`) and each lane publishes only its own results
# (`upload_results`). A lane re-uploading the identical tensors tree would burn 6
# Hub commits against the 256/hr repo cap for zero new bytes. `upload` keeps the
# legacy both-legs behavior so PHASE=all is unchanged.
if want_phase upload || want_phase upload_tensors; then
  echo "[nlmap] phase=upload: nonlinear map payloads -> HF analysis_tensors"
  uv run python scripts/issue1739_upload.py --stage tensors
fi
if want_phase upload || want_phase upload_results || want_phase compose; then
  # #1880 push race: several rounds/lanes write this branch concurrently and the
  # upload helper only fetch-then-retries (it never rebases), so a stale base
  # fails BOTH attempts. Advance the clone to the tip FIRST — results are still
  # uncommitted working-tree files here, so the ff-merge is clean and the
  # helper's commit lands on the current tip. (Matches
  # issue1739_pvscore_dispatch.sh; the comment below used to promise this sync
  # while no fetch actually ran.)
  echo "[nlmap] phase=upload: syncing clone to origin/$BRANCH before commit"
  git checkout -- eval_results/issue_1739/nonlinear_map/map_quality.json 2>/dev/null || true
  rm -f eval_results/issue_1739/nonlinear_map/stage_maps_report.json
  git fetch origin "$BRANCH"
  git merge --ff-only "origin/$BRANCH"
  echo "[nlmap] phase=upload: results -> git"
  uv run python scripts/issue1739_upload.py --stage results-git --branch "$BRANCH"
fi

# ---- sentinel + terminal line ---------------------------------------------
# ONLY a leg that actually SCORED may write the results sentinel + [phase=done]:
# those two ARE the poller's completion contract, so a phase-A (`prefetch`) or
# staging-only leg emitting them would drain as `epm:results` and read the whole
# round as finished after zero arm scores. PHASE=all and every lane include
# `fits`, so their behavior is unchanged.
if ! want_phase fits; then
  # NOTE: this message must never spell the terminal phase token literally —
  # poll_pipeline.py greps `[phase=<name>]` out of the log tail, so writing it
  # here would BE the completion signal this branch exists to withhold
  # (pod-side-reporting.md: that token is reserved for the one terminal line).
  echo "[nlmap] partial leg (phase='$PHASE'): no scoring ran, so NO results" \
    "sentinel and no terminal phase marker — this leg is deliberately non-terminal."
  exit 0
fi
if [ -n "${EPM_I1739_NL_SKIP_SENTINEL:-}" ]; then
  # new-arm-round nlood wrapper (issue1739_newarm_nlood.sh): the WRAPPER's HF
  # self-upload must run AFTER this dispatcher exits and BEFORE any terminal
  # record, so the wrapper suppresses this sentinel and owns the terminal
  # story itself (the gap2 pattern: GCE phase publication at workload exit).
  echo "[nlmap] results sentinel suppressed (EPM_I1739_NL_SKIP_SENTINEL) —" \
    "the wrapping driver owns the terminal record."
  exit 0
fi
uv run python -c "
import json, pathlib, time
p = pathlib.Path('$LOG_DIR/issue-1739-nlmap-results.json')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({
    'issue': 1739,
    'round': 'nonlinear_map',
    'status': 'ok',
    'behaviors': '$BEHAVIORS'.split(),
    'kinds': '$KINDS'.split(),
    'arms': '$NL_ARMS'.split(),
    'out_root': '$NL_ROOT',
    'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}, indent=2))
print('[nlmap] sentinel written ->', p)
"
# Mode-gated standalone-lane terminal: under the newarm nlood WRAPPER
# (issue1739_newarm_nlood.sh) EPM_I1739_NL_SKIP_SENTINEL exits ABOVE, so this
# token never reaches the wrapper's log mid-pipeline.
echo "[phase=done]"  # noqa: phase-done-reserved
