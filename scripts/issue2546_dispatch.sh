#!/usr/bin/env bash
# Pod dispatcher for task #2546 (CoT context->answer map) — plan v4 §4.2 / §4.3 / §9.
# Modeled on scripts/issue1336_dispatch.sh (sentinel writer, phase done-files, resume).
#
# Usage:
#   bash scripts/issue2546_dispatch.sh --arm {1,2,3} [all|p1_smoke|p2a_pilot|p2_gen_post|
#                                                     p3_gen_short|p4_capture|p4b_capture_rel|
#                                                     ge_gate|p5_fits] [--smoke]
#
# Phase chain (`all`, production): p1_smoke -> (arm 1 only) p2a_pilot -> p2_gen_post ->
# p3_gen_short -> p4_capture -> p4b_capture_rel (reliability-draw frozen-layer capture,
# feeds the P5 split-half ceiling) -> ge_gate (G-E, blocking) -> p5_fits -> results
# sentinel -> the single terminal `[phase=done]` line. Under `--smoke` the `all` chain
# runs p1_smoke only (tiny rehearsal; smoke out-root; kind epm:smoke-result; the P1 rig
# smoke exercises the capture-reliability leg in-script), then the terminal line.
#
# Phase-script CLI contract (units 2/3 implement these entrypoints; keep in sync):
#   scripts/issue2546_gen_capture.py --arm K [--smoke] [--phase pilot|gen-post|gen-short|capture]
#                                    --out-root <dir>
#   scripts/issue2546_n1m_read.py    --out-root <dir> [--smoke]           (arm-1 pilot leg)
#   scripts/issue2546_fit_cells.py   --arm K [--g0] [--smoke] --out-root <dir>
#
# Contracts honored (pod-side-reporting.md): `[phase=...]` log lines with the single
# reserved terminal `[phase=done]`; per-phase sentinels /workspace/logs/issue-2546-
# <slug>-<epoch>.json written ONLY on rc=0 (write-once, epoch-stamped — never rewritten
# in place); resume/done state lives OUTSIDE the drained sentinel glob (under $OUT_ROOT/
# done); pod-side code never shells scripts/task.py; per-phase
# assert_out_root_headroom before every write-heavy phase (plan §9 disk rows).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
[ -d "$REPO_ROOT" ] || REPO_ROOT="$PWD"
cd "$REPO_ROOT"
# Conditional .env sourcing: RunPod pods carry .env; GCE/SLURM lanes export tokens instead.
if [ -f ./.env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
LOG_DIR="${EPS_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"
DISPATCH_START=$(date +%s)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
ARM=""
SMOKE=""
PHASE_ARG="all"
KEY_PHASE=""
usage() {
    echo "usage: bash scripts/issue2546_dispatch.sh --arm {1,2,3} [PHASE] [--smoke]" >&2
    echo "  PHASE in: all p1_smoke p2a_pilot p2_gen_post p3_gen_short p4_capture" >&2
    echo "            p4b_capture_rel ge_gate p5_fits" >&2
    exit 2
}
while [ $# -gt 0 ]; do
    case "$1" in
    --arm)
        ARM="${2:?--arm needs a value}"
        shift 2
        ;;
    --smoke)
        SMOKE="1"
        shift
        ;;
    all | p1_smoke | p2a_pilot | p2_gen_post | p3_gen_short | p4_capture | p4b_capture_rel | ge_gate | p5_fits)
        PHASE_ARG="$1"
        shift
        ;;
    __phase_key)
        # Test/debug probe: print the recipe-keyed done-file key for a phase and exit.
        PHASE_ARG="__phase_key"
        KEY_PHASE="${2:?__phase_key needs a phase name}"
        shift 2
        ;;
    *)
        usage
        ;;
    esac
done
case "$ARM" in
1 | 2 | 3) ;;
*)
    echo "[dispatch2546] FATAL: --arm {1,2,3} is required (got '${ARM}')" >&2
    usage
    ;;
esac

# GPU count guard (informational for phase scripts; they own per-GPU CVD fan-out).
NGPU=$( (nvidia-smi --list-gpus 2>/dev/null || true) | wc -l)
case "$NGPU" in
*[!0-9]* | "")
    echo "[dispatch2546] FATAL: bad GPU count '$NGPU'" >&2
    exit 70
    ;;
esac
if [ "$PHASE_ARG" != "__phase_key" ] && [ "$NGPU" -lt 1 ]; then
    echo "[dispatch2546] FATAL: no GPUs visible" >&2
    exit 70
fi

# Out-root split (bidirectional pair; production default, smoke diverted).
OUT_ROOT_full="${EPS_OUT_ROOT:-/workspace/issue2546}"
OUT_ROOT_smoke="${EPS_OUT_ROOT_SMOKE:-/workspace/issue2546_smoke}"
OUT_ROOT="$OUT_ROOT_full"
if [ -n "$SMOKE" ]; then
    OUT_ROOT="$OUT_ROOT_smoke"
fi
DONE_DIR="$OUT_ROOT/done"
mkdir -p "$OUT_ROOT" "$DONE_DIR"

# Per-phase headroom floors (GB; plan §9 disk rows: arm-1/2 store ~62-65 GB in-flight,
# arm-3 in-flight peak < ~80 GB under per-corpus upload-then-free). Env-overridable.
NEED_GB_GEN="${EPS_NEED_GB_GEN:-6}"
NEED_GB_SHORT="${EPS_NEED_GB_SHORT:-4}"
if [ "$ARM" = "3" ]; then
    NEED_GB_CAPTURE="${EPS_NEED_GB_CAPTURE:-80}"
else
    NEED_GB_CAPTURE="${EPS_NEED_GB_CAPTURE:-65}"
fi
NEED_GB_FITS="${EPS_NEED_GB_FITS:-8}"
# P4b rel store per arm: ~4 draws x quota rows x 3 kinds x 3 layers bf16 ~ <1 GB
# in-flight (per-stem upload-then-free); 4 GB covers headroom generously.
NEED_GB_CAPREL="${EPS_NEED_GB_CAPREL:-4}"
if [ -n "$SMOKE" ]; then
    # Gate-calibration parity (#1345 class): production-sized disk floors would
    # spuriously kill the tiny smoke leg; smoke floors cover the smoke footprint only.
    NEED_GB_GEN=2
    NEED_GB_SHORT=2
    NEED_GB_CAPTURE=5
    NEED_GB_FITS=2
    NEED_GB_CAPREL=2
fi

echo "[dispatch2546] arm=$ARM phase=$PHASE_ARG smoke=${SMOKE:-0} ngpu=$NGPU out_root=$OUT_ROOT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sentinel writer (poll_pipeline contract: _SENTINEL_REQUIRED_KEYS; write-once,
# epoch-stamped path in the drained glob issue-2546-*.json).
emit_signal() { # $1 kind, $2 slug, $3 gate, $4 blocks_pipeline(true|false), $5 note
    SIG_KIND="$1" SIG_SLUG="$2" SIG_GATE="$3" SIG_BLOCKS="$4" SIG_NOTE="$5" \
        SIG_DIR="$LOG_DIR" SIG_SMOKE="${SMOKE:-}" \
        uv run python - <<'PY'
import json
import os
import time

ts = int(time.time())
slug = os.environ["SIG_SLUG"]
path = os.path.join(os.environ["SIG_DIR"], f"issue-2546-{slug}-{ts}.json")
payload = {
    "sentinel_schema_version": 1,
    "kind": os.environ["SIG_KIND"],
    "version": 1,
    "task_id": 2546,
    "gate": os.environ["SIG_GATE"],
    "by": "issue2546_dispatch",
    "blocks_pipeline": os.environ["SIG_BLOCKS"] == "true",
    "smoke": bool(os.environ.get("SIG_SMOKE")),
    "note": os.environ["SIG_NOTE"],
}
with open(path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[signal] wrote sentinel {path}")
PY
}

assert_headroom() { # $1 phase name, $2 need_gb — plan §9 rider 5 (resume-aware form)
    PHASE_NAME="$1" NEED_GB="$2" OUT_ROOT_ENV="$OUT_ROOT" uv run python - <<'PY'
import os

from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

assert_out_root_headroom(
    os.environ["OUT_ROOT_ENV"], float(os.environ["NEED_GB"]), phase=os.environ["PHASE_NAME"]
)
PY
}

RECIPE_REV="planv4"
phase_key() { printf 'issue2546-%s-a%s%s-%s\n' "$1" "$ARM" "${SMOKE:+-smoke}" "$RECIPE_REV"; }
phase_done() { [ -f "$DONE_DIR/$(phase_key "$1").done" ]; }
mark_phase() { : >"$DONE_DIR/$(phase_key "$1").done"; }

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

phase_p1_smoke() {
    # P1 (arm 1: 240-row) / P1.2-P1.3 (arms 2-3: 60-row lite) rig smoke; gates G-A..G-F
    # evaluated inside the phase script (production entrypoint, production models).
    echo "[phase=p1_smoke]"
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --smoke --out-root "$OUT_ROOT"
    emit_signal "epm:smoke-result" "smoke-a$ARM" "p1_smoke" "false" \
        "arm $ARM P1 rig smoke complete (G-A..G-F evaluated in-script); out_root=$OUT_ROOT"
}

phase_p2a_pilot() {
    # Arm-1 pilot: full gsm8k_test1319 gen+capture (measures production per-row walls)
    # + the frozen n1m directional read (plan §4.2 P2a).
    if [ "$ARM" != "1" ]; then
        echo "[dispatch2546] p2a_pilot is arm-1 only — skipping on arm $ARM"
        return 0
    fi
    echo "[phase=p2a_pilot]"
    assert_headroom p2a_pilot "$NEED_GB_GEN"
    uv run python scripts/issue2546_gen_capture.py --arm 1 --phase pilot \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke}
    uv run python scripts/issue2546_n1m_read.py --out-root "$OUT_ROOT" ${SMOKE:+--smoke}
    emit_signal "epm:progress" "pilot" "p2a_pilot" "false" \
        "arm 1 pilot complete: gsm8k_test1319 gen+capture + frozen n1m read; out_root=$OUT_ROOT"
}

phase_p2_gen_post() {
    echo "[phase=p2_gen_post]"
    assert_headroom p2_gen_post "$NEED_GB_GEN"
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --phase gen-post \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke}
    emit_signal "epm:progress" "gen-post-a$ARM" "p2_gen_post" "false" \
        "arm $ARM post-side generation complete (rollout text uploaded pre-reduction); out_root=$OUT_ROOT"
}

phase_p3_gen_short() {
    echo "[phase=p3_gen_short]"
    assert_headroom p3_gen_short "$NEED_GB_SHORT"
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --phase gen-short \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke}
    emit_signal "epm:progress" "gen-short-a$ARM" "p3_gen_short" "false" \
        "arm $ARM short-side generation complete; out_root=$OUT_ROOT"
}

phase_p4_capture() {
    echo "[phase=p4_capture]"
    assert_headroom p4_capture "$NEED_GB_CAPTURE"
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --phase capture \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke}
    emit_signal "epm:progress" "capture-a$ARM" "p4_capture" "false" \
        "arm $ARM capture complete (per-corpus upload-then-free; store verified-uploaded); out_root=$OUT_ROOT"
}

phase_p4b_capture_rel() {
    # P4b (U4): teacher-force the persisted reliability-draw TEXT at the arm's
    # FROZEN layer subset only (per-draw rel_ stems; feeds run_reliability_unit).
    echo "[phase=p4b_capture_rel]"
    assert_headroom p4b_capture_rel "$NEED_GB_CAPREL"
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --phase capture-reliability \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke}
    emit_signal "epm:progress" "capture-rel-a$ARM" "p4b_capture_rel" "false" \
        "arm $ARM reliability-draw capture complete (frozen-layer per-draw rel_ stems uploaded); out_root=$OUT_ROOT"
}

phase_ge_gate() {
    # G-E fit-core reuse gate (blocking, before P5): refit the pinned #825 Qwen S1 cell;
    # PASS iff layer-19 held-out R2 within +/-0.01 of 0.6731. Doubles as the item-(m)
    # device-domain exercise on this pod's own GPUs. Designed halt: epm:failure sentinel
    # (the routing artifact) + non-zero exit.
    echo "[phase=ge_gate]"
    set +e
    uv run python scripts/issue2546_fit_cells.py --g0 --arm "$ARM" --out-root "$OUT_ROOT"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        emit_signal "epm:failure" "ge-fail-a$ARM" "G-E" "true" \
            "failure_class: code — G-E fit-core reuse gate FAILED on arm $ARM pod (rc=$rc): layer-19 held-out R2 outside +/-0.01 of 0.6731 on the pinned #825 Qwen S1 cell (turnstore @ deb7a452)"
        echo "[dispatch2546] FATAL: G-E gate failed rc=$rc" >&2
        exit "$rc"
    fi
    emit_signal "epm:progress" "ge-gate-a$ARM" "G-E" "false" \
        "G-E fit-core reuse gate PASS on arm $ARM pod"
}

phase_p5_fits() {
    echo "[phase=p5_fits]"
    assert_headroom p5_fits "$NEED_GB_FITS"
    uv run python scripts/issue2546_fit_cells.py --arm "$ARM" \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke}
}

write_results_sentinel() {
    RES_OUT_ROOT="$OUT_ROOT" RES_ARM="$ARM" RES_SMOKE="${SMOKE:-}" RES_LOG_DIR="$LOG_DIR" \
        RES_NGPU="$NGPU" RES_START="$DISPATCH_START" \
        uv run python - <<'PY'
import glob
import json
import os
import subprocess
import time

out_root = os.environ["RES_OUT_ROOT"]
arm = os.environ["RES_ARM"]
smoke = bool(os.environ.get("RES_SMOKE"))
sha = "unavailable-no-git-checkout"
try:
    p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if p.returncode == 0:
        sha = p.stdout.strip()
except OSError:
    pass
cell_jsons = sorted(glob.glob(os.path.join(out_root, "out", "cells", f"*__a{arm}.json")))
ladder_jsons = sorted(glob.glob(os.path.join(out_root, "out", "ladder", "*.json")))
gpu_hours = round(
    (time.time() - float(os.environ["RES_START"])) / 3600.0 * max(int(os.environ["RES_NGPU"]), 1),
    2,
)
note = {
    "phase": "p5_fits",
    "arm": int(arm),
    "eval_paths": cell_jsons[:50] + ladder_jsons[:20],
    "n_cell_jsons": len(cell_jsons),
    "n_ladder_jsons": len(ladder_jsons),
    "gpu_hours_estimate": gpu_hours,
    "reproducibility_card": {
        "hf_data_repo": "superkaiba1/explore-persona-space-data",
        "prefixes": [
            f"issue2546_cotmap/analysis_tensors/thinkstore/arm{arm}/",
            "issue2546_cotmap/raw_completions/",
            "issue2546_cotmap/corpora_v1/",
        ],
        "wandb_url": "n/a (no training in this task)",
        "final_commit_sha": sha,
    },
}
kind = "epm:smoke-result" if smoke else "epm:results"
ts = int(time.time())
path = os.path.join(os.environ["RES_LOG_DIR"], f"issue-2546-fits-a{arm}-{ts}.json")
payload = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "task_id": 2546,
    "gate": "p5_fits",
    "by": "issue2546_dispatch",
    "blocks_pipeline": False,
    "smoke": smoke,
    "note": json.dumps(note),
}
with open(path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[signal] wrote results sentinel {path}")
PY
}

run_phase() { # $1 = phase name
    if phase_done "$1"; then
        echo "[dispatch2546] phase $1 already complete — skipping"
        return 0
    fi
    "phase_$1"
    mark_phase "$1"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$PHASE_ARG" in
all)
    if [ -n "$SMOKE" ]; then
        # Smoke rehearsal: the P1 rig smoke only (tiny, smoke out-root); the production
        # chain runs P1 as its own first, gating phase.
        run_phase p1_smoke
    else
        run_phase p1_smoke
        if [ "$ARM" = "1" ]; then
            run_phase p2a_pilot
        fi
        run_phase p2_gen_post
        run_phase p3_gen_short
        run_phase p4_capture
        run_phase p4b_capture_rel
        run_phase ge_gate
        run_phase p5_fits
    fi
    write_results_sentinel
    echo "[phase=done]"
    ;;
p1_smoke | p2a_pilot | p2_gen_post | p3_gen_short | p4_capture | p4b_capture_rel | ge_gate | p5_fits)
    run_phase "$PHASE_ARG"
    echo "[dispatch2546] single-phase invocation of $PHASE_ARG complete (no terminal done line)"
    ;;
__phase_key)
    phase_key "$KEY_PHASE"
    ;;
*)
    usage
    ;;
esac
exit 0
