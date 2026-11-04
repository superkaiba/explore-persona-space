#!/usr/bin/env bash
# Issue #2676 — Phase A pod driver (GPU): data decrypt, 8 training runs,
# vLLM generation for every arm, HF uploads, sentinel JSONs.
#
# Phases (plan v1 §1-§3, §7):
#   data      decrypt the Turner bad-medical dataset (never uploaded).
#   per model (qwen05b = Qwen2.5-0.5B-Instruct, qwen15b = Qwen2.5-1.5B-Instruct):
#     train all-layers (1 epoch) -> target loss = mean of last 20 step losses
#     untrained generate (Betley 8x50 + 200 held-out x1)
#     per K: train top-K to target (cap 3 epochs) -> generate
#     checkpoints upload to the HF MODEL repo under issue2676/<model>/<arm>/
#     then are deleted locally (never delete unuploaded); train metadata
#     (train_summary/loss_history/holdout) uploads per arm to the DATA repo
#     under issue2676_topk_em/train/<label>/ (plan-§2 dose-matching evidence).
#     Each train run prints a `[train-timing] ... mean_s_per_step=... `
#     line — the smoke's 20-step value projects the full run's wall.
#   upload    bulk raw-completions upload to the HF DATA repo
#             (issue2676_topk_em/raw_completions/...).
#
# Judging (Batch API) + pool building + the teacher-forced margin run from
# the VM / a later GPU invocation of scripts/issue2676_eval.py — NOT here.
#
# Resumable: per-step done-sentinels under $EVAL_ROOT/done/; re-running
# skips completed steps. Progress + exit-code sentinels to
# /workspace/logs/issue-2676-*.json (schema v1, drained by poll_pipeline.py).
# This script NEVER shells out to scripts/task.py (pod-side contract).
#
# --smoke: 0.5B only, K=1, 20 train steps, 2 questions x 4 samples, through
# the SAME entrypoints; all roots redirect to /tmp/issue-2676-smoke/ and HF
# uploads are skipped (no smoke writes to canonical prefixes).
#
# --followup: instead of Phase A, run the comma-separated job list in
# FOLLOWUP_JOBS, each job "model:arm:seed[:max_epochs]" (arm = all | topK).
# Labels: <model>_<arm>, plus _s<seed> when seed != 1, plus _e<epochs> when a
# max_epochs override is given (e.g. qwen15b_top7_e20, qwen05b_all_s2). A topK
# job is dose-matched to the SAME-seed all-layers arm's target loss, so a
# seed-S all job must precede its seed-S topK jobs. Holdout generation uses the
# arm's own holdout.json, asserted identical to the model's Phase A holdout.
set -euo pipefail

usage() {
    echo "usage: bash scripts/issue2676_run_pod.sh [--smoke] [--followup]"
}

SMOKE=0
FOLLOWUP=0
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        --followup) FOLLOWUP=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EPM_I2676_REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
cd "$REPO_ROOT"

DATA="$REPO_ROOT/data/issue404/bad_medical_advice.jsonl"
TRAIN_SEED=1
CUR_SEED="$TRAIN_SEED"
GEN_SEED=0
# Follow-up job lists (bidirectional smoke/production pair; see --followup).
FOLLOWUP_JOBS_FULL="${FOLLOWUP_JOBS:-}"
FOLLOWUP_JOBS_SMOKE="qwen05b:all:2,qwen05b:top1:2:2"
RUN_NAME="Phase A"
[ "$FOLLOWUP" = 1 ] && RUN_NAME="Follow-up"
# Generation nucleus top_p (recorded in raw-completions metadata). Default 1.0
# (full-distribution sampling, matching Betley's eval; user decision 2026-11-03).
# Override per run: TOP_P=0.95 bash scripts/issue2676_run_pod.sh.
TOP_P="${TOP_P:-1.0}"
# Checkpoint persistence mode. "hf" (default) uploads each checkpoint to HF.
# "defer" skips the HF upload and KEEPS the local checkpoint (the upload step
# is never marked done, so delete_local_ckpt never removes it); the
# orchestrator persists the kept checkpoints before the pod is terminated.
# Set 2026-11-03: the public model repo is over HF's 20,000-file cap and the
# private overflow repo hit the private-storage quota.
CKPT_UPLOAD="${CKPT_UPLOAD:-hf}"
case "$CKPT_UPLOAD" in
    hf|defer) ;;
    *) echo "CKPT_UPLOAD must be hf or defer, got '$CKPT_UPLOAD'" >&2; exit 2 ;;
esac

# Bidirectional smoke/production root pair (every *_smoke assignment is
# $SMOKE-gated; production defaults are never narrowed by smoke values).
EVAL_ROOT_FULL="$REPO_ROOT/eval_results/issue_2676"
CKPT_ROOT_FULL="${EPM_I2676_CKPT_ROOT:-/workspace/issue2676_ckpts}"
SENTINEL_DIR_FULL="${EPM_I2676_SENTINEL_DIR:-/workspace/logs}"
MODELS_FULL="qwen05b qwen15b"

EVAL_ROOT="$EVAL_ROOT_FULL"
CKPT_ROOT="$CKPT_ROOT_FULL"
SENTINEL_DIR="$SENTINEL_DIR_FULL"
MODELS="$MODELS_FULL"
if [ "$SMOKE" = 1 ]; then
    SMOKE_ROOT="/tmp/issue-2676-smoke"
    EVAL_ROOT="$SMOKE_ROOT/eval_results/issue_2676"
    CKPT_ROOT="$SMOKE_ROOT/ckpts"
    SENTINEL_DIR="$SMOKE_ROOT/logs"
    MODELS="qwen05b"
fi
JOBS="$FOLLOWUP_JOBS_FULL"
[ "$SMOKE" = 1 ] && JOBS="$FOLLOWUP_JOBS_SMOKE"
if [ "$FOLLOWUP" = 1 ] && [ -z "$JOBS" ]; then
    echo "--followup needs FOLLOWUP_JOBS (model:arm:seed[:max_epochs],...)" >&2
    exit 2
fi
DONE_DIR="$EVAL_ROOT/done"
mkdir -p "$EVAL_ROOT" "$CKPT_ROOT" "$SENTINEL_DIR" "$DONE_DIR"

CURRENT_PHASE="init"

model_id() {
    case "$1" in
        qwen05b) echo "Qwen/Qwen2.5-0.5B-Instruct" ;;
        qwen15b) echo "Qwen/Qwen2.5-1.5B-Instruct" ;;
        *) echo "unknown model short name: $1" >&2; return 1 ;;
    esac
}

model_ks() {
    # Plan §1: 0.5B K in {1, 6, 12} of 24; 1.5B K in {1, 7, 14} of 28.
    case "$1" in
        qwen05b) echo "1 6 12" ;;
        qwen15b) echo "1 7 14" ;;
        *) echo "unknown model short name: $1" >&2; return 1 ;;
    esac
}

step_done() { [ -f "$DONE_DIR/$1.done" ]; }
mark_done() { touch "$DONE_DIR/$1.done"; }

# Before every vLLM init the GPU must be clean. This pod runs only this
# workload, so any leftover VLLM::EngineCore is an orphan from an earlier step:
# reap it by exact PID, then fail loud if memory is still held (a foreign
# allocation with no killable process). gotchas.md #601 / #825.
gpu_clean_gate() {
    local floor_mib=2048 used=0 attempt p
    for attempt in 1 2; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk 'BEGIN{m=0} {if ($1 + 0 > m) m = $1 + 0} END{print m}')
        if [ "$used" -le "$floor_mib" ]; then
            echo "[gpu-gate] clean (max memory.used=${used} MiB, attempt=$attempt)"
            return 0
        fi
        echo "[gpu-gate] memory.used=${used} MiB > ${floor_mib} MiB; reaping orphaned EngineCore"
        for p in $(pgrep -f '^VLLM::EngineCore' || true); do
            kill -KILL "$p" 2>/dev/null || true
        done
        sleep 10
    done
    echo "[gpu-gate] GPU still held after reap (memory.used=${used} MiB)" >&2
    nvidia-smi >&2 || true
    return 1
}

write_sentinel() {
    # write_sentinel <kind> <note> — schema-v1 sentinel for poll_pipeline.py.
    # Smoke sentinels carry "-smoke-" in the filename (version-rewrite excluded).
    SENT_KIND="$1" SENT_NOTE="$2" SENT_DIR="$SENTINEL_DIR" SENT_SMOKE="$SMOKE" \
    uv run python -c '
import datetime, json, os, pathlib
kind = os.environ["SENT_KIND"]
slug = kind.replace(":", "_")
now = datetime.datetime.now(datetime.UTC)
ts = int(now.timestamp())
smoke = os.environ.get("SENT_SMOKE") == "1"
name = f"issue-2676-{slug}-smoke-{ts}.json" if smoke else f"issue-2676-{slug}-{ts}.json"
d = pathlib.Path(os.environ["SENT_DIR"])
d.mkdir(parents=True, exist_ok=True)
payload = {
    "sentinel_schema_version": 1,
    "task_id": 2676,
    "kind": kind,
    "version": 1,
    "gate": None,
    "blocks_pipeline": False,
    "smoke": smoke,
    "note": os.environ["SENT_NOTE"],
    "by": "issue2676_run_pod.sh",
    "ts": now.isoformat(),
}
p = d / name
p.write_text(json.dumps(payload, indent=2))
print(f"[sentinel] wrote {p}")
' || echo "[sentinel] WARNING: sentinel write failed (kind=$1)" >&2
}

on_exit() {
    rc=$?
    if [ "$rc" -eq 0 ]; then
        write_sentinel "epm:results" "$RUN_NAME complete rc=0 smoke=$SMOKE eval_root=$EVAL_ROOT; arms trained+generated; checkpoints on HF model repo under issue2676/; raw completions under issue2676_topk_em/raw_completions/. Next: VM-side judge pilot + judge waves (scripts/issue2676_eval.py judge / install-judge), build-pools, margin."
        echo "[phase=done]"
    else
        write_sentinel "epm:progress" "$RUN_NAME FAILED rc=$rc at phase=$CURRENT_PHASE smoke=$SMOKE eval_root=$EVAL_ROOT (resumable: re-run issue2676_run_pod.sh — completed steps skip via $DONE_DIR)"
        echo "[phase=failed rc=$rc phase_at_failure=$CURRENT_PHASE]"
    fi
}
trap on_exit EXIT

train_arm() {
    # train_arm <model_short> <arm_label> [extra trainer args...]
    local m="$1" arm="$2"
    shift 2
    local label="${m}_${arm}"
    local step="train_${label}"
    local out="$CKPT_ROOT/$label"
    local train_dir="$EVAL_ROOT/train/$label"
    if step_done "$step" && [ -f "$train_dir/train_summary.json" ]; then
        echo "[skip] $step (done)"
        return 0
    fi
    CURRENT_PHASE="train_$label"
    echo "[phase=train arm=$label]"
    local smoke_flag=()
    [ "$SMOKE" = 1 ] && smoke_flag=(--smoke)
    gpu_clean_gate
    uv run python scripts/issue2676_topk_sft.py \
        --model "$(model_id "$m")" \
        --data "$DATA" \
        --out-dir "$out" \
        --seed "$CUR_SEED" \
        "${smoke_flag[@]}" \
        "$@"
    mkdir -p "$train_dir"
    cp "$out/train_summary.json" "$out/loss_history.json" "$out/holdout.json" "$train_dir/"
    mark_done "$step"
}

upload_train_meta() {
    # upload_train_meta <model_short> <arm_label> — the per-arm train metadata
    # (train_summary/loss_history/holdout JSONs, the plan-§2 dose-matching
    # evidence) to the HF DATA repo under issue2676_topk_em/train/<label>/.
    # Without this the $EVAL_ROOT/train/ tree dies with the instance (the
    # raw-completions bulk upload matches raw_completions* only).
    local m="$1" arm="$2"
    local label="${m}_${arm}"
    local step="upload_train_${label}"
    if [ "$SMOKE" = 1 ]; then
        echo "[smoke] skipping train-metadata upload for $label (no HF writes in smoke)"
        return 0
    fi
    if step_done "$step"; then
        echo "[skip] $step (done)"
        return 0
    fi
    CURRENT_PHASE="upload_train_$label"
    echo "[phase=upload_train arm=$label]"
    uv run python scripts/issue2676_eval.py upload-train \
        --train-dir "$EVAL_ROOT/train/$label" \
        --label "$label"
    mark_done "$step"
}

upload_ckpt() {
    # upload_ckpt <model_short> <arm_label> — HF model repo, before local deletion.
    local m="$1" arm="$2"
    local label="${m}_${arm}"
    local step="upload_ckpt_${label}"
    if [ "$SMOKE" = 1 ]; then
        echo "[smoke] skipping checkpoint upload for $label (no HF writes in smoke)"
        return 0
    fi
    if step_done "$step"; then
        echo "[skip] $step (done)"
        return 0
    fi
    if [ "$CKPT_UPLOAD" = defer ]; then
        echo "[defer] $step: HF checkpoint upload deferred; keeping $CKPT_ROOT/$label on the pod"
        return 0
    fi
    CURRENT_PHASE="upload_ckpt_$label"
    echo "[phase=upload_ckpt arm=$label]"
    uv run python scripts/issue2676_eval.py upload-ckpt \
        --model-dir "$CKPT_ROOT/$label" \
        --dest "issue2676/$m/$arm"
    mark_done "$step"
}

cap_hit_over_threshold() {
    # exit 0 when cap_hit_fraction > 0.02 (plan §3 re-gen trigger)
    uv run python -c '
import json, sys
frac = json.load(open(sys.argv[1]))["cap_hit_fraction"]
sys.exit(0 if frac > 0.02 else 1)
' "$1"
}

gen_arm() {
    # gen_arm <model_short> <arm_label> <model_path> <mode: betley|holdout>
    local m="$1" arm="$2" mp="$3" mode="$4"
    local label="${m}_${arm}"
    local step="gen_${mode}_${label}"
    local out="$EVAL_ROOT/$mode/$label"
    if step_done "$step" && [ -f "$out/raw_completions.json" ]; then
        echo "[skip] $step (done)"
        return 0
    fi
    CURRENT_PHASE="gen_${mode}_$label"
    local gen_args=(
        --model "$mp" --label "$label" --mode "$mode"
        --out-dir "$out" --seed "$GEN_SEED" --top-p "$TOP_P"
    )
    if [ "$mode" = "holdout" ]; then
        gen_args+=(--data "$DATA" --holdout-json "${HOLDOUT_JSON_OVERRIDE:-$EVAL_ROOT/train/${m}_all/holdout.json}")
    fi
    [ "$SMOKE" = 1 ] && gen_args+=(--smoke)
    gpu_clean_gate
    uv run python scripts/issue2676_eval.py generate "${gen_args[@]}"
    if [ "$SMOKE" = 0 ] && cap_hit_over_threshold "$out/raw_completions.json"; then
        echo "[regen] $label $mode cap-hit > 2% — regenerating at max_tokens=4096 (plan §3)"
        gpu_clean_gate
        uv run python scripts/issue2676_eval.py generate "${gen_args[@]}" --max-tokens 4096 --force
    fi
    mark_done "$step"
}

delete_local_ckpt() {
    # Only after checkpoint upload + both generations are done (never delete unuploaded).
    local m="$1" arm="$2"
    local label="${m}_${arm}"
    if [ "$SMOKE" = 1 ]; then
        return 0  # smoke ckpts live in /tmp; nothing uploaded, keep for inspection
    fi
    if step_done "upload_ckpt_${label}" && step_done "gen_betley_${label}" && step_done "gen_holdout_${label}"; then
        if [ -d "$CKPT_ROOT/$label" ]; then
            echo "[cleanup] deleting local checkpoint $CKPT_ROOT/$label (uploaded to HF)"
            rm -rf "${CKPT_ROOT:?}/$label"
        fi
    else
        echo "[cleanup] keeping $CKPT_ROOT/$label (upload or generation not confirmed done)"
    fi
}

read_target_loss() {
    # read_target_loss <model_short> [seed_suffix] — the all-layers arm's target.
    uv run python -c '
import json, sys
print(json.load(open(sys.argv[1]))["target_loss"])
' "$EVAL_ROOT/train/$1_all${2:-}/train_summary.json"
}

# ── Phase: data ──────────────────────────────────────────────────────────────
CURRENT_PHASE="data"
echo "[phase=data]"
if [ ! -f "$DATA" ]; then
    # Decrypts from the Turner repo (TURNER_EDS_PASSWORD in .env); NEVER a paid
    # Claude regen, NEVER uploaded (plan §7: training data stays off HF).
    uv run python scripts/fetch_or_generate_issue404_medical.py --no-claude-regen --no-upload
fi
[ -f "$DATA" ] || { echo "FATAL: $DATA missing after fetch" >&2; exit 1; }
N_ROWS=$(wc -l < "$DATA")
echo "[data] $DATA rows=$N_ROWS"
write_sentinel "epm:progress" "[phase=data] done: bad_medical_advice.jsonl rows=$N_ROWS smoke=$SMOKE"

# ── Phases: per-model train + generate ───────────────────────────────────────
if [ "$FOLLOWUP" = 0 ]; then
for m in $MODELS; do
    base_id="$(model_id "$m")"

    # 1) all-layers positive control (1 epoch; its final 20-step mean is the target)
    train_arm "$m" all --arm all
    upload_train_meta "$m" all
    upload_ckpt "$m" all
    gen_arm "$m" all "$CKPT_ROOT/${m}_all" betley
    gen_arm "$m" all "$CKPT_ROOT/${m}_all" holdout
    delete_local_ckpt "$m" all

    # 2) untrained baseline (needs the all arm's holdout.json for holdout mode)
    gen_arm "$m" untrained "$base_id" betley
    gen_arm "$m" untrained "$base_id" holdout

    # 3) top-K arms, dose-matched to the all arm's target loss
    TARGET="$(read_target_loss "$m")"
    echo "[target] $m all-layers target loss = $TARGET"
    ks="$(model_ks "$m")"
    [ "$SMOKE" = 1 ] && ks="1"
    for k in $ks; do
        arm="top${k}"
        train_arm "$m" "$arm" --arm topk --k "$k" --target-loss "$TARGET"
        upload_train_meta "$m" "$arm"
        upload_ckpt "$m" "$arm"
        gen_arm "$m" "$arm" "$CKPT_ROOT/${m}_${arm}" betley
        gen_arm "$m" "$arm" "$CKPT_ROOT/${m}_${arm}" holdout
        delete_local_ckpt "$m" "$arm"
    done
    write_sentinel "epm:progress" "[phase=model_done] $m: all + untrained + top-K arms trained and generated (smoke=$SMOKE)"
done
else
# ── Follow-up jobs (--followup) ──────────────────────────────────────────────
assert_same_holdout() {
    # Fail loud unless the arm's holdout rows equal the model's Phase A holdout.
    uv run python -c '
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
key = lambda h: (h["data_sha256"], h["holdout_indices"])
if key(a) != key(b):
    sys.exit(f"holdout mismatch: {sys.argv[1]} != {sys.argv[2]}")
print(f"[holdout] {sys.argv[1]} matches {sys.argv[2]}")
' "$1" "$2"
}

IFS=',' read -r -a JOB_LIST <<< "$JOBS"
echo "[followup] ${#JOB_LIST[@]} jobs: $JOBS"
for job in "${JOB_LIST[@]}"; do
    IFS=':' read -r m arm seed epochs <<< "$job"
    [ -n "$m" ] && [ -n "$arm" ] && [ -n "$seed" ] || { echo "bad job spec '$job'" >&2; exit 2; }
    case " $MODELS " in *" $m "*) ;; *) echo "job model $m not in MODELS ($MODELS)" >&2; exit 2 ;; esac
    model_id "$m" > /dev/null
    seed_sfx=""
    [ "$seed" != 1 ] && seed_sfx="_s${seed}"
    arm_label="${arm}${seed_sfx}"
    [ -n "$epochs" ] && arm_label="${arm_label}_e${epochs}"
    if [ "$arm_label" = "$arm" ]; then
        echo "job '$job' would overwrite the Phase A arm ${m}_${arm}" >&2
        exit 2
    fi
    extra=()
    [ -n "$epochs" ] && extra+=(--max-epochs "$epochs")
    CUR_SEED="$seed"
    case "$arm" in
        all)
            train_arm "$m" "$arm_label" --arm all "${extra[@]}"
            ;;
        top*)
            k="${arm#top}"
            [[ "$k" =~ ^[0-9]+$ ]] || { echo "bad arm '$arm' in job '$job'" >&2; exit 2; }
            TARGET="$(read_target_loss "$m" "$seed_sfx")"
            echo "[target] ${m}_${arm_label}: seed-${seed} all-layers target loss = $TARGET"
            train_arm "$m" "$arm_label" --arm topk --k "$k" --target-loss "$TARGET" "${extra[@]}"
            ;;
        *) echo "bad arm '$arm' in job '$job'" >&2; exit 2 ;;
    esac
    CUR_SEED="$TRAIN_SEED"
    label="${m}_${arm_label}"
    upload_train_meta "$m" "$arm_label"
    upload_ckpt "$m" "$arm_label"
    HOLDOUT_JSON_OVERRIDE="$EVAL_ROOT/train/$label/holdout.json"
    phase_a_holdout="$EVAL_ROOT/train/${m}_all/holdout.json"
    if [ -f "$phase_a_holdout" ] && [ "$HOLDOUT_JSON_OVERRIDE" != "$phase_a_holdout" ]; then
        assert_same_holdout "$HOLDOUT_JSON_OVERRIDE" "$phase_a_holdout"
    elif [ "$SMOKE" = 0 ]; then
        echo "FATAL: Phase A holdout $phase_a_holdout missing; cannot check holdout parity" >&2
        exit 1
    fi
    gen_arm "$m" "$arm_label" "$CKPT_ROOT/$label" betley
    gen_arm "$m" "$arm_label" "$CKPT_ROOT/$label" holdout
    unset HOLDOUT_JSON_OVERRIDE
    # Per-job raw-completions upload: this job's answers reach HF before the
    # next job starts, so an instance death loses at most one job's answers
    # (#664 per-cell rule; the 2026-11-04 GCP network-wedge lost all of them).
    if [ "$SMOKE" = 0 ]; then
        if step_done "upload_raw_${label}"; then
            echo "[skip] upload_raw_${label} (done)"
        else
            CURRENT_PHASE="upload_raw_$label"
            echo "[phase=upload_raw arm=$label]"
            uv run python scripts/issue2676_eval.py upload-raw --eval-results-dir "$EVAL_ROOT"
            mark_done "upload_raw_${label}"
        fi
    fi
    delete_local_ckpt "$m" "$arm_label"
    write_sentinel "epm:progress" "[phase=job_done] $label trained and generated (job=$job smoke=$SMOKE)"
done
fi

# ── Phase: raw-completions upload (one bulk commit, data repo) ───────────────
CURRENT_PHASE="upload_raw"
if [ "$SMOKE" = 1 ]; then
    echo "[smoke] skipping raw-completions upload (no HF writes in smoke)"
else
    if step_done "upload_raw"; then
        echo "[skip] upload_raw (done)"
    else
        echo "[phase=upload_raw]"
        uv run python scripts/issue2676_eval.py upload-raw --eval-results-dir "$EVAL_ROOT"
        mark_done "upload_raw"
    fi
fi

CURRENT_PHASE="finished"
echo "[phase=phase_a_complete] eval_root=$EVAL_ROOT"
# on_exit writes the final epm:results sentinel + [phase=done]
