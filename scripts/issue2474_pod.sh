#!/usr/bin/env bash
# issue #2474 — BASE-MODEL capture dispatcher (pod-side).
#
# Captures, on the UN-fine-tuned base model (Qwen2.5-7B-Instruct), the three
# pre-fine-tuning predictor inputs that #2379 never captured for base
# (issue2379_pod.sh:472 "condition models only (5 em + 3 caps); base has no
# captures"). The reused capture entrypoint already documents this exact path:
#   --model-name base --model Qwen/Qwen2.5-7B-Instruct
#
# Phases (in order; cheap first so a crash still leaves the fast arms usable):
#   p1 prep    : issue2379_prep_data.py -> banks + FULL training mixes (turner
#                included; needed per-condition for the mu phase).
#   p2 grid    : v_C at all 28 layers, q_sim x every trigger (p_inoc is in-bank).
#                Unlocks the context arm AND the predicted-answer arm (the base
#                map is already fitted, reused from #779 pass-B at the pin).
#   p3 mu      : per-CONDITION mean training-answer state UNDER THE BASE MODEL —
#                the #1979 Train-Ref predictor. One run per condition.
#   p4 ceiling : 3 on-policy rollouts per (q,trigger) + teacher-forced re-forward
#                -> real-answer arm.
#
# LOGICAL-NAME COLLISION (why --model-name is not just "base"): the capture
# script writes predictor_captures/<model-name>/<phase>.pt, so one shared name
# would make the em and caps grids overwrite each other, and the 8 mu runs
# collapse onto a single mu.pt. Distinct logical names per (setting) and per
# (mu, condition) keep every bundle addressable. The MODEL is base in all cases.
#
# Uploads are deferred (--no-upload): the capture script's HF slug is
# issue2379_reelicit, and #2474's artifacts belong under their own prefix. One
# bulk upload_folder runs after the phases (p5), never a per-file loop.
#
# Usage:
#   bash scripts/issue2474_pod.sh --phases "p1 p2 p3 p4 p5"
#   bash scripts/issue2474_pod.sh --phases "p2" --force

set -euo pipefail

REPO_ROOT="${EPM_REPO_ROOT:-/workspace/explore-persona-space}"
LOG_DIR="${EPM_LOG_DIR:-/workspace/logs}"
OUT_DIR="${EPM_OUT_DIR:-eval_results/issue_2474}"
DATA_DIR="${EPM_DATA_DIR:-data/issue_2474}"
BASE_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SENTINEL_PREFIX="issue-2474"
PHASES="p1 p2 p3 p4 p5"
FORCE_FLAG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --phases) PHASES="$2"; shift 2 ;;
        --force)  FORCE_FLAG="--force"; shift ;;
        *) echo "ERROR: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

# Login shell semantics: the SSH-MCP shell is sh, so `.` not `source` (#545).
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# EM conditions carry the EM inoculation prompt; caps conditions the caps one.
EM_CONDS="em_bad_medical_advice em_bad_legal_advice em_bad_security_advice em_turner_extreme_sports em_turner_risky_financial"
CAPS_CONDS="caps_french caps_german caps_spanish"

log() { echo "[phase=$1] $2" ; }

sentinel() {  # $1 = phase, $2 = rc
    printf '{"phase":"%s","rc":%s,"utc":"%s"}\n' \
        "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$LOG_DIR/${SENTINEL_PREFIX}-$1.done.json"
}

capture() {  # $1 = phase, $2 = setting, $3 = model-name, then extra args
    local phase="$1" setting="$2" name="$3"; shift 3
    uv run python scripts/issue2379_capture.py \
        --phase "$phase" --setting "$setting" \
        --model-name "$name" --model "$BASE_MODEL_ID" \
        --banks-dir "$DATA_DIR/banks" --out-dir "$OUT_DIR" \
        --no-upload $FORCE_FLAG "$@"
}

run_p1() {
    log p1 "prep_data (full mixes, turner INCLUDED — required per-condition for mu)"
    uv run python scripts/issue2379_prep_data.py \
        --out-dir "$DATA_DIR" \
        --artifact "$REPO_ROOT/tasks/planning/2379/artifacts/kwon2026_extracted_text.txt"
    sentinel p1 0
}

run_p2() {
    log p2 "grid — v_C at 28 layers, q_sim x triggers, base model"
    capture grid em   base_em
    capture grid caps base_caps
    sentinel p2 0
}

run_p3() {
    log p3 "mu — per-condition mean training-answer state under the BASE model (#1979 Train-Ref)"
    for c in $EM_CONDS;   do
        log p3 "mu em $c"
        capture mu em   "base_mu_$c" --train-jsonl "$DATA_DIR/train/$c.jsonl"
    done
    for c in $CAPS_CONDS; do
        log p3 "mu caps $c"
        capture mu caps "base_mu_$c" --train-jsonl "$DATA_DIR/train/$c.jsonl"
    done
    sentinel p3 0
}

run_p4() {
    log p4 "ceiling — 3 on-policy rollouts per (q,trigger) + teacher-forced re-forward"
    capture ceiling em   base_em
    capture ceiling caps base_caps
    sentinel p4 0
}

run_p5() {
    log p5 "bulk upload of capture tensors to the #2474 HF prefix"
    uv run python scripts/issue2474_upload.py --out-dir "$OUT_DIR"
    sentinel p5 0
}

for ph in $PHASES; do
    case "$ph" in
        p1) run_p1 ;;
        p2) run_p2 ;;
        p3) run_p3 ;;
        p4) run_p4 ;;
        p5) run_p5 ;;
        *) echo "ERROR: unknown phase '$ph'" >&2; exit 2 ;;
    esac
done

log all "COMPLETE phases=$PHASES"
