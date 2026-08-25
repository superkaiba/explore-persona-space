#!/usr/bin/env bash
# Issue #2546 — CPU-only rehearsal of the dispatcher's rc=4 fallback-relaunch chain (r4).
#
# Closes the r3 reconciler BLOCKER `smoke-fallback-unwired`: the REAL
# scripts/issue2546_dispatch.sh fallback leg executes end-to-end with the GPU
# phase entrypoints replaced by a PATH-shimmed fake `uv` (stdin `uv run python -`
# helper calls — emit_signal / assert_headroom / read_fallbacks_from_report —
# delegate to the real uv and run for real). Two legs:
#
#   leg 1 (fallback engage): smoke call 1 exits rc=4 with report
#     fallbacks_available=["prefill"] -> the dispatcher wipes the attempt-1
#     smoke artifact subdirs (asserted BY THE SHIM at call 2: un-wiped markers
#     mean the worker's fp_sha refuse-to-mix guard would raise -> exit 98),
#     relaunches EXACTLY ONCE with --prefill-fallback, and persists
#     done/fallbacks_a1.env; then production p4_capture + p5_fits single-phase
#     invocations re-source that env and thread --prefill-fallback into the
#     phase argv (asserted from the shim-logged argv).
#   leg 2 (guard refusal): rc=4 with fallbacks_available=[] -> the
#     FB_ARGS==prev_args guard refuses the relaunch (dispatcher exit 4,
#     epm:failure sentinel written, exactly ONE smoke call).
#
# CPU-only + scratch-diverted (mktemp under /tmp; EPS_OUT_ROOT /
# EPS_OUT_ROOT_SMOKE / EPS_LOG_DIR all inside the scratch; CUDA_VISIBLE_DEVICES=0
# satisfies the GPU-count guard without touching hardware). Repeatable selftest:
# rc=0 on PASS (scratch removed; keep with REHEARSAL_KEEP=1), scratch kept on
# any FAIL for forensics.
set -euo pipefail

if [ ! -f scripts/issue2546_dispatch.sh ]; then
    echo "[rehearsal] FATAL: run from the repo root (scripts/issue2546_dispatch.sh not found)" >&2
    exit 2
fi

REAL_UV="$(command -v uv)"
if [ -z "$REAL_UV" ]; then
    echo "[rehearsal] FATAL: real uv not on PATH" >&2
    exit 2
fi

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/i2546-rehearsal.XXXXXX")"
echo "[rehearsal] scratch=$SCRATCH real_uv=$REAL_UV"

fail() {
    echo "[rehearsal] FAIL: $*" >&2
    echo "[rehearsal] scratch kept at $SCRATCH" >&2
    exit 1
}

mkdir -p "$SCRATCH/bin"

# --- fake-uv shim: intercepts the three GPU phase-entrypoint argv shapes;
# --- everything else (the dispatcher's stdin `uv run python -` helpers)
# --- delegates to the real uv. Quoted heredoc: $-refs resolve at SHIM runtime.
cat >"$SCRATCH/bin/uv" <<'SHIM'
#!/usr/bin/env bash
set -euo pipefail
S="${REHEARSAL_SCRATCH:?shim needs REHEARSAL_SCRATCH}"
REAL="${REHEARSAL_REAL_UV:?shim needs REHEARSAL_REAL_UV}"
ARGS="$*"
case "$ARGS" in
"run python scripts/issue2546_gen_capture.py --arm 1 --phase capture --out-root "*)
    printf '%s\n' "$ARGS" >"$S/state/p4_argv"
    echo "[shim] p4 capture stub ok"
    exit 0
    ;;
"run python scripts/issue2546_fit_cells.py --arm 1 --out-root "*)
    printf '%s\n' "$ARGS" >"$S/state/p5_argv"
    echo "[shim] p5 fits stub ok"
    exit 0
    ;;
"run python scripts/issue2546_gen_capture.py --arm 1 --smoke --out-root "*)
    n=0
    if [ -f "$S/state/smoke_calls" ]; then n=$(cat "$S/state/smoke_calls"); fi
    n=$((n + 1))
    printf '%s\n' "$n" >"$S/state/smoke_calls"
    mode=$(cat "$S/state/mode")
    rep_dir="$S/smoke/out/reports"
    mkdir -p "$rep_dir"
    if [ "$n" -eq 1 ]; then
        case "$ARGS" in
        *"--prefill-fallback"*)
            echo "[shim] FATAL: first smoke call already carries --prefill-fallback: $ARGS" >&2
            exit 97
            ;;
        esac
        # Attempt-1 smoke artifacts the dispatcher must wipe before the relaunch.
        for sub in rollouts work store fitcache; do
            mkdir -p "$S/smoke/$sub"
            : >"$S/smoke/$sub/attempt1.marker"
        done
        if [ "$mode" = "norung" ]; then
            printf '{"fallbacks_available": [], "gates": "stub-norung"}\n' >"$rep_dir/smoke_a1.json"
        else
            printf '{"fallbacks_available": ["prefill"], "gates": "stub-fallback-band"}\n' >"$rep_dir/smoke_a1.json"
        fi
        echo "[shim] smoke call 1: rc=4 fallback band (mode=$mode)"
        exit 4
    elif [ "$n" -eq 2 ]; then
        case "$ARGS" in
        *"--prefill-fallback"*) ;;
        *)
            echo "[shim] FATAL: relaunch argv missing --prefill-fallback: $ARGS" >&2
            exit 97
            ;;
        esac
        for sub in rollouts work store fitcache; do
            if [ -e "$S/smoke/$sub/attempt1.marker" ]; then
                echo "[shim] FATAL: attempt-1 $sub artifacts NOT wiped before the relaunch — the worker's fp_sha refuse-to-mix guard would raise here" >&2
                exit 98
            fi
        done
        echo "[shim] smoke call 2: relaunch with fallback engaged; attempt-1 artifacts wiped -> rc=0"
        exit 0
    else
        echo "[shim] FATAL: unexpected smoke call #$n" >&2
        exit 99
    fi
    ;;
*)
    exec "$REAL" "$@"
    ;;
esac
SHIM
chmod +x "$SCRATCH/bin/uv"

RC=0
run_dispatch() { # $1 leg-scratch-root, $2 abs logfile, rest = dispatcher args
    local sroot="$1" logf="$2"
    shift 2
    set +e
    timeout --kill-after=30s 180s env \
        PATH="$SCRATCH/bin:$PATH" \
        REHEARSAL_SCRATCH="$sroot" \
        REHEARSAL_REAL_UV="$REAL_UV" \
        REPO_ROOT="$PWD" \
        CUDA_VISIBLE_DEVICES=0 \
        EPS_OUT_ROOT="$sroot/full" \
        EPS_OUT_ROOT_SMOKE="$sroot/smoke" \
        EPS_LOG_DIR="$sroot/logs" \
        EPS_NEED_GB_GEN=1 EPS_NEED_GB_SHORT=1 EPS_NEED_GB_CAPTURE=1 \
        EPS_NEED_GB_FITS=1 EPS_NEED_GB_CAPREL=1 \
        bash scripts/issue2546_dispatch.sh "$@" >"$logf" 2>&1
    RC=$?
    set -e
}

count_in_log() { # $1 fixed string, $2 logfile -> echoes count
    local c
    c=$(grep -cF "$1" "$2" || true)
    printf '%s\n' "${c:-0}"
}

# ---------------------------------------------------------------------------
# Leg 1 — fallback engage: rc=4 -> wipe -> ONE guarded relaunch -> persist ->
#          FB_ARGS threads into production p4_capture / p5_fits.
# ---------------------------------------------------------------------------
L1="$SCRATCH/leg1"
mkdir -p "$L1/state"
echo "fallback" >"$L1/state/mode"

echo "[rehearsal] leg 1a: p1_smoke (production mode) through the REAL dispatcher fallback leg"
run_dispatch "$L1" "$L1/leg1_p1.log" --arm 1 p1_smoke
if [ "$RC" -ne 0 ]; then
    tail -40 "$L1/leg1_p1.log" >&2
    fail "leg1 p1_smoke rc=$RC (want 0)"
fi
calls=$(cat "$L1/state/smoke_calls" 2>/dev/null || echo 0)
[ "$calls" = "2" ] || fail "leg1 smoke calls=$calls (want exactly 2: initial + ONE relaunch)"
relaunches=$(count_in_log "relaunching smoke ONCE with --prefill-fallback" "$L1/leg1_p1.log")
[ "$relaunches" = "1" ] || fail "leg1 relaunch log line count=$relaunches (want 1)"
FB_ENV="$L1/full/done/fallbacks_a1.env"
[ -f "$FB_ENV" ] || fail "leg1 $FB_ENV not persisted"
grep -qxF 'PREFILL_FB=1' "$FB_ENV" || fail "leg1 fallbacks_a1.env missing PREFILL_FB=1 (got: $(cat "$FB_ENV" | tr '\n' ' '))"
echo "[rehearsal] leg 1a PASS: 1 guarded relaunch, attempt-1 artifacts wiped (shim-verified), fallbacks_a1.env persisted"

echo "[rehearsal] leg 1b: production p4_capture re-sources fallbacks_a1.env"
run_dispatch "$L1" "$L1/leg1_p4.log" --arm 1 p4_capture
if [ "$RC" -ne 0 ]; then
    tail -40 "$L1/leg1_p4.log" >&2
    fail "leg1 p4_capture rc=$RC (want 0)"
fi
grep -qF 'fb_args= --prefill-fallback' "$L1/leg1_p4.log" || fail "leg1 p4 dispatcher banner missing fb_args= --prefill-fallback"
grep -qF -- '--prefill-fallback' "$L1/state/p4_argv" || fail "leg1 p4 phase argv missing --prefill-fallback: $(cat "$L1/state/p4_argv")"
echo "[rehearsal] leg 1b PASS: p4_capture argv carries --prefill-fallback ($(cat "$L1/state/p4_argv"))"

echo "[rehearsal] leg 1c: production p5_fits re-sources fallbacks_a1.env"
run_dispatch "$L1" "$L1/leg1_p5.log" --arm 1 p5_fits
if [ "$RC" -ne 0 ]; then
    tail -40 "$L1/leg1_p5.log" >&2
    fail "leg1 p5_fits rc=$RC (want 0)"
fi
grep -qF -- '--prefill-fallback' "$L1/state/p5_argv" || fail "leg1 p5 phase argv missing --prefill-fallback: $(cat "$L1/state/p5_argv")"
echo "[rehearsal] leg 1c PASS: p5_fits argv carries --prefill-fallback ($(cat "$L1/state/p5_argv"))"

# ---------------------------------------------------------------------------
# Leg 2 — guard refusal: rc=4 with NO new fallback rung -> refuse relaunch,
#          epm:failure sentinel, dispatcher exit 4, exactly ONE smoke call.
# ---------------------------------------------------------------------------
L2="$SCRATCH/leg2"
mkdir -p "$L2/state"
echo "norung" >"$L2/state/mode"

echo "[rehearsal] leg 2: p1_smoke rc=4 with fallbacks_available=[] (guard must refuse)"
run_dispatch "$L2" "$L2/leg2_p1.log" --arm 1 p1_smoke
[ "$RC" -eq 4 ] || { tail -40 "$L2/leg2_p1.log" >&2; fail "leg2 p1_smoke rc=$RC (want 4: guarded refusal)"; }
grep -qF 'FATAL: p1_smoke rc=4 with no new fallback rung' "$L2/leg2_p1.log" || fail "leg2 refusal FATAL line missing"
calls2=$(cat "$L2/state/smoke_calls" 2>/dev/null || echo 0)
[ "$calls2" = "1" ] || fail "leg2 smoke calls=$calls2 (want exactly 1: no relaunch past the guard)"
nfail=$(find "$L2/logs" -name 'issue-2546-smoke-fail-a1-*.json' 2>/dev/null | wc -l)
[ "$nfail" -ge 1 ] || fail "leg2 epm:failure sentinel not written under $L2/logs"
echo "[rehearsal] leg 2 PASS: guard refused (rc=4, 1 smoke call, epm:failure sentinel written)"

echo "[rehearsal] PASS: rc=4 fallback-relaunch chain rehearsed end-to-end through the real dispatcher"
if [ -n "${REHEARSAL_KEEP:-}" ]; then
    echo "[rehearsal] REHEARSAL_KEEP set — scratch kept at $SCRATCH"
else
    rm -rf "$SCRATCH"
fi
exit 0
