#!/usr/bin/env bash
# Issue #2223 — pod-side phased launcher: Lu et al. Fig-4 persona-drift
# reproduction + context-vector-ONLY stabilization grid.
#
#   scripts/launch_issue_2223.sh <7b|32b> <smoke|full>
#
# Runs the driver phases SEQUENTIALLY under `set -e` (each phase's failure halts
# the chain — no sibling-block swallow, plan §12 exit-path trace). The generate
# phases fan out 4-way CVD-pinned across the CONVERSATION axis (§9 data-parallel
# shard: each GPU holds one model replica, processes a disjoint stride slice of
# the alive conversation set); a merge phase then unions the shards into the
# canonical per-cell raw_completions.json.
#
# Per-leg out-roots (crash-fix-rounds § per-leg out-roots): full → the labeled
# eval_results tree; smoke → /tmp/issue-2223-smoke-<model>. The FULL leg reaps
# this model's smoke root at entry so a smoke leg's regime never poisons full
# resume state.
#
# 7B leg (4×H100): sparse-cone add eval_results/issue_2203 (the schema-v2
# band/τ JSON the caphook/steer arms read). 32B leg (4×H200): run the paper-cap
# bootstrap engine (external/assistant-axis, delivers the paper's capping engine
# for A1) — external/assistant-axis is git-UNTRACKED on every branch (§12).
#
# Terminal: the driver's `finalize` phase writes a poll_pipeline-conformant
# results sentinel, then THIS launcher emits the single reserved `[phase=done]`
# line (pod-side reporting contract req 1/2 — the token is the launcher's own
# terminal line, never a per-phase echo).
set -euo pipefail

# CUDA allocator — multi-turn fragmentation guard. This is a lockstep loop that appends
# an exchange every turn, so each turn re-prefills a longer context and the per-turn
# activation footprint climbs monotonically. PyTorch's caching allocator strands
# reserved-but-unallocated segments it cannot coalesce, and around turn 11-13 the cliff
# is reached: two 7B A0 shards died of CUDA OOM (2026-08-13, turns 10 and 13) with
# 12.60-14.60 GiB reserved-but-unallocated against 9.91-10.24 GiB requests that failed
# with only 6.21-7.52 GiB free — fragmentation, NOT a working-set overflow (the card
# held more idle-but-reserved memory than the failed request). expandable_segments lets
# segments grow in place instead of stranding them.
#
# Measured on the shard-0 recovery: first fresh turn after resume completed in 693s with
# zero OOM, vs the un-fixed siblings' 740-906s turn-11 band — so the fix is at worst
# throughput-neutral and plausibly a small win.
#
# Exported ONCE here rather than per-phase so every phase and BOTH legs inherit it:
# Phase B on the 7B leg alone is 9 arms x 15 turns over the same conversation set, and
# every one of them would otherwise walk into the identical cliff. The ${VAR:-default}
# form keeps an explicit caller-supplied value authoritative.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${1:-}"
MODE="${2:-}"
case "$MODEL" in 7b | 32b) ;; *) echo "usage: $0 <7b|32b> <smoke|full>" >&2; exit 2 ;; esac
case "$MODE" in smoke | full) ;; *) echo "usage: $0 <7b|32b> <smoke|full>" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRIVER="$REPO_ROOT/scripts/issue2223_drift.py"
FIGURES="$REPO_ROOT/scripts/issue2223_figures.py"
cd "$REPO_ROOT"

# Arm registry split (mirrors ARMS in issue2223_drift.py; A0 = "both").
# A0 runs FIRST (Phase A anchor); the Phase B grid is gated on the G2 verdict.
# v4 amendment: +A2c (all-context-token cap) as a full generation arm; +A2corr
# (correction mode — per-turn regen over A0's drifted transcripts; consumes the A0
# canonical raw_completions.json, which lands in Phase A before the gate, so its
# only barrier is the one every Phase B arm already shares).
ARMS_7B_PHASEB=(A2a A2b A2c A2corr A3a A3b A4 A4R A5)
CAP_ARMS_7B=(A2a A2b A2c A2corr)   # firing telemetry (realized-vs-empirical)

# G2 stop gate (plan §7; code-review r1 ISSUE 4): consume the Phase-A verdict BEFORE
# any Phase B generation spends. GATE_STOP_RC=8 is the driver's DESIGNED halt —
# PHASE_B=0 skips the Phase B grid while Phase A still uploads; any other non-zero
# rc is a real crash and halts the chain (set -e semantics preserved via explicit exit).
GATE_STOP_RC=8
check_g2_gate() {  # sets PHASE_B=1|0
    local gate_rc=0
    echo "[launch] --phase gate (model=$MODEL mode=$MODE)"
    # Direct single-external-command capture (NOT the run_phase bash function):
    # `fn || var=$?` under set -e disables errexit inside the function BODY, collapsing
    # mid-function failures to the last command's rc (#1426/#1516); a child process
    # keeps its own set -e, so the driver invocation is captured directly.
    # shellcheck disable=SC2086
    uv run python "$DRIVER" --phase gate --model "$MODEL" $SMOKE --out-root "$OUT_ROOT" \
        || gate_rc=$?
    if [ "$gate_rc" -eq 0 ]; then
        PHASE_B=1
    elif [ "$gate_rc" -eq "$GATE_STOP_RC" ]; then
        PHASE_B=0
        echo "[launch] G2 STOP — Failed-to-reproduce at adequate power; skipping Phase B grid"
    else
        echo "[launch] FATAL: gate phase crashed (rc=$gate_rc)" >&2
        exit "$gate_rc"
    fi
}

if [ "$MODE" = smoke ]; then
    OUT_ROOT="/tmp/issue-2223-smoke-$MODEL"
    SMOKE="--smoke"
    NUM_SHARDS=1
else
    OUT_ROOT="$REPO_ROOT/eval_results"   # driver appends issue_2223/
    SMOKE=""
    NUM_SHARDS="${EPM_NUM_SHARDS:-4}"
    # FULL leg reaps THIS model's smoke root at entry (crash-fix-rounds § per-leg out-roots).
    if [ -d "/tmp/issue-2223-smoke-$MODEL" ]; then
        echo "[launch] reaping smoke root /tmp/issue-2223-smoke-$MODEL"
        rm -rf "/tmp/issue-2223-smoke-$MODEL"
    fi
fi

run_phase() {  # $1=phase, $@=extra driver args
    local phase="$1"; shift
    echo "[launch] --phase $phase $* (model=$MODEL mode=$MODE)"
    # shellcheck disable=SC2086
    uv run python "$DRIVER" --phase "$phase" --model "$MODEL" $SMOKE --out-root "$OUT_ROOT" "$@"
}

# Generate one cell with the §9 4-way CVD-pinned conversation-shard fan-out, then merge.
gen_cell() {  # $1=arm  [extra driver args, e.g. --think]
    local arm="$1"; shift
    if [ "$NUM_SHARDS" -le 1 ]; then
        run_phase generate --arm "$arm" "$@"
        return
    fi
    local pids=() rc=0 i
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        echo "[launch] generate shard $i/$NUM_SHARDS arm=$arm (CVD=$i)"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="$i" uv run python "$DRIVER" --phase generate --arm "$arm" \
            --model "$MODEL" $SMOKE --out-root "$OUT_ROOT" \
            --shard-id "$i" --num-shards "$NUM_SHARDS" "$@" &
        pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$rc" -eq 0 ] || { echo "[launch] FATAL: a generate shard failed for arm=$arm" >&2; exit 1; }
    run_phase merge --arm "$arm" --num-shards "$NUM_SHARDS" "$@"
}

# One (arm, optional --think) drift cell: generate (sharded) → merge → activations.
drift_cell() {  # $1=arm  [extra driver args]
    local arm="$1"; shift
    gen_cell "$arm" "$@"
    run_phase activations --arm "$arm" "$@"
}

echo "[launch] issue 2223 drift launcher — model=$MODEL mode=$MODE out_root=$OUT_ROOT shards=$NUM_SHARDS"

# ── per-leg setup ────────────────────────────────────────────────────────────────
if [ "$MODEL" = 32b ]; then
    echo "[launch] 32B leg — running paper-cap bootstrap engine"
    bash "$REPO_ROOT/scripts/issue2203_pod_bootstrap_engine.sh"
else
    echo "[launch] 7B leg — sparse-cone add eval_results/issue_2203 (band/τ)"
    git -C "$REPO_ROOT" sparse-checkout add eval_results/issue_2203 2>/dev/null \
        || echo "[launch] sparse-checkout add skipped (non-sparse checkout or already present)"
    BAND_TAU="$REPO_ROOT/eval_results/issue_2203/full-rerun-bugfix/phase1_band_tau.json"
    [ "$MODE" = smoke ] || [ -f "$BAND_TAU" ] \
        || { echo "[launch] FATAL: band/τ JSON absent at $BAND_TAU" >&2; exit 1; }
fi

# ── 32B import-gate smoke: the blind-spot enumeration's 32B leg is import-check
#    + a real-axis load, NOT a full generate (§4 smoke blind-spot item 2). ──────────
if [ "$MODEL" = 32b ] && [ "$MODE" = smoke ]; then
    echo "[launch] 32B smoke = import-check + real Lu-axis load (no full generate)"
    uv run python "$DRIVER" --import-check
    uv run python "$DRIVER" --phase alpha --model 32b --smoke --out-root "$OUT_ROOT" || true
    run_phase finalize
    echo "[phase=done] issue 2223 32B smoke (import-gate) complete"
    exit 0
fi

# ── per-model phase chain ─────────────────────────────────────────────────────────
# TOPICS = REQUIRE-FETCH on the pod (r2 TOCTOU BLOCKER): the stimulus is generated
# EXACTLY ONCE pre-launch on the VM (`--phase topics --generate-topics`, 0-GPU
# paid-API) and published to HF; this pod-side call FETCHES the canonical copy and
# EXITS NON-ZERO if it is absent/unfetchable — it never generates its own (a
# fetch-miss->generate fallback under the v4 parallel 7B ∥ 32B launch is a
# check-then-act race that silently forks the stimulus across legs).
run_phase topics

# Phase ordering (code-review r1): A0 (Phase A anchor) FIRST → aggregate (verdict) →
# G2 gate → Phase B grid only on PASS. `upload` runs BEFORE the optional 0-GPU `ridge`
# add-on on both legs (r1 BLOCKER 1): the durability of raw_completions/ — the primary
# deliverable — must never depend on an optional analysis add-on succeeding.
if [ "$MODEL" = 7b ]; then
    run_phase alpha                        # α calibration (steering arms A4/A4R/A5 need it)
    drift_cell A0                          # Phase A anchor (7B leg)
    run_phase aggregate                    # reproduction verdict (A0__7b same-leg anchor)
    check_g2_gate                          # sets PHASE_B
    if [ "$PHASE_B" -eq 1 ]; then
        for arm in "${ARMS_7B_PHASEB[@]}"; do
            drift_cell "$arm"
        done
        for arm in "${CAP_ARMS_7B[@]}"; do
            run_phase firing --arm "$arm"
        done
    fi
    run_phase capability --arm A0
    if [ "$PHASE_B" -eq 1 ]; then
        for arm in "${ARMS_7B_PHASEB[@]}"; do
            run_phase capability --arm "$arm"
        done
    fi
    run_phase fig5_generate
    run_phase fig5_judge
    run_phase upload
    run_phase ridge                        # optional 0-GPU add-on — AFTER upload (r1 BLOCKER 1)
else
    # 32B full: A0 drift (think-OFF faithful + think-ON extension) FIRST, then the
    # verdict + gate; A1 (paper-cap, Phase B grid member) only on G2 PASS.
    drift_cell A0
    drift_cell A0 --think                  # Qwen-3 thinking-ON Phase-A extension arm
    run_phase aggregate
    check_g2_gate                          # sets PHASE_B
    if [ "$PHASE_B" -eq 1 ]; then
        drift_cell A1
        run_phase firing --arm A1
    fi
    run_phase capability --arm A0
    if [ "$PHASE_B" -eq 1 ]; then
        run_phase capability --arm A1
    fi
    run_phase fig5_generate
    run_phase fig5_judge
    run_phase upload
    run_phase ridge                        # optional 0-GPU add-on — AFTER upload (r1 BLOCKER 1)
fi

# ── figures (git-destined; smoke diverts via --out-dir) ────────────────────────────
if [ "$MODE" = smoke ]; then
    FIG_EVAL="$OUT_ROOT"
    FIG_OUT="/tmp/issue-2223-smoke-fig-$MODEL"
else
    FIG_EVAL="$OUT_ROOT/issue_2223"
    FIG_OUT="$REPO_ROOT/figures"
fi
echo "[launch] rendering figures (eval-dir=$FIG_EVAL out-dir=$FIG_OUT)"
uv run python "$FIGURES" --eval-dir "$FIG_EVAL" --out-dir "$FIG_OUT" || echo "[launch] figures WARN (non-fatal)"

# ── terminal: results sentinel via finalize, then the single reserved done line ────
run_phase finalize
echo "[phase=done] issue 2223 drift run complete (model=$MODEL mode=$MODE)"
