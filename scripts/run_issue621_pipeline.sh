#!/usr/bin/env bash
# Issue #621 pod/VM-side end-to-end pipeline launcher (rank-1 read/write LoRA).
#
# Forked from scripts/run_issue538_pipeline.sh (pinned e6b195f81). Sequence:
#   0. Preflight — orchestrate.preflight --json with the documented
#      behind-origin/main feature-branch tolerance (parse the WHOLE stdout,
#      fail only on errors OTHER than the behind-main line), then the
#      issue-621 pinned-input preflight (R_persona sha pins + question-pool
#      pin + composition gate). SKIP via EPM_SKIP_PREFLIGHT=1 after manual
#      verification only.
#   1. Smoke (Phase S, plan §7): 1 cell (read, florist, seed 42) — train +
#      A-init sanity + bystander probe, then the smoke eval shift_extract
#      subprocess, then i621_smoke_gate.py (adapter-application parity ±1
#      nat + the §14 duty-2 wall re-projection incl. FULL-CAP). On a PURE
#      band miss the §7 fallback fires ONCE: epochs cap 16 → 32, re-smoke.
#   2. Sweep train: 29 remaining cells, 4-way CUDA_VISIBLE_DEVICES sharding
#      (CVD exported in the LAUNCHER env per cell + matching --gpu-id — the
#      in-process clobber alone is defeated by import-time cuInit).
#   3. Context bank: --step generate (vLLM) then --step capture --upload
#      (HF hooks) as SEPARATE subprocesses (vLLM worker-orphan gotcha).
#   4. Eval emission (vLLM, 4 shards) then shift_extract (HF, 4 shards).
#   5. i621_upload_artifacts.py — raw completions / mixes / shift tensors /
#      trajectories, fail-loud verified.
#   6. i621_write_results_sentinel.py — §10 reproducibility card (per-cell
#      adapter_paths Hub-verified + wandb run names) → poll sentinel.
#   7. The SINGLE terminal [phase=done] line.
#
# Pod-side code NEVER shells out to scripts/task.py. Phase output streams
# to per-phase LOG FILES, never raw through startup-script stdout (GCE
# metadata-runner bufio overflow gotcha); only short [phase=...] lines hit
# stdout. [phase=done] appears EXACTLY once, at success-terminal.

set -euo pipefail

ISSUE=621

# Repo root: GCP lane clones into $WORKLOAD_ROOT and runs the workload-cmd
# from there; RunPod uses /workspace/explore-persona-space.
if [[ -n "${WORKLOAD_ROOT:-}" && -d "${WORKLOAD_ROOT:-}" ]]; then
    cd "$WORKLOAD_ROOT"
elif git rev-parse --show-toplevel >/dev/null 2>&1; then
    cd "$(git rev-parse --show-toplevel)"
else
    cd /workspace/explore-persona-space
fi

LOG_DIR=/workspace/logs
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    LOG_DIR="$(pwd)/logs"
    mkdir -p "$LOG_DIR"
fi

export WANDB_PROJECT=issue_621_rank1_readwrite
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=false

PIPE_LOG="$LOG_DIR/issue-621-pipeline.log"

phase_log() {
    # Single source of truth for [phase=...] markers (poll_pipeline.py).
    local line="[phase=$1] $(date -u +%Y-%m-%dT%H:%M:%SZ) $2"
    echo "$line"
    echo "$line" >> "$PIPE_LOG"
}

write_failure_sentinel() {
    local note="$1"
    local epoch
    epoch=$(date -u +%s)
    local out_path="$LOG_DIR/issue-${ISSUE}-epm_failure-${epoch}.json"
    python3 - "$note" "$out_path" <<'PY'
import json, sys, datetime
note, out_path = sys.argv[1], sys.argv[2]
json.dump({
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 621,
    "by": "issue621_pipeline",
    "ts": datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"),
    "note": note,
}, open(out_path, "w"), indent=1)
print(out_path)
PY
}

fail() {
    # Failure terminal: epm:failure sentinel + [phase=failed]; NEVER
    # [phase=done] (reserved for the success terminal — incident #545).
    local msg="$1"
    phase_log failed "$msg"
    write_failure_sentinel "$msg"
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# 0. Preflight
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${EPM_SKIP_PREFLIGHT:-0}" == "1" ]]; then
    phase_log preflight "SKIPPED via EPM_SKIP_PREFLIGHT=1 (manual verification asserted)"
else
    phase_log preflight "orchestrate.preflight --json (behind-origin/main tolerated)"
    uv run python - > "$LOG_DIR/issue-621-preflight-core.log" 2>&1 <<'PY' || fail "core preflight FAILED (see issue-621-preflight-core.log)"
import json
import re
import subprocess
import sys

proc = subprocess.run(
    ["uv", "run", "python", "-m", "explore_persona_space.orchestrate.preflight", "--json"],
    capture_output=True,
    text=True,
)
# Pretty-printed multi-line JSON: parse the WHOLE stdout, never the last line.
payload = json.loads(proc.stdout)
errors = payload.get("errors") or []
behind = re.compile(r"behind origin/main")
real = [e for e in errors if not behind.search(str(e))]
for e in errors:
    print(("TOLERATED: " if behind.search(str(e)) else "ERROR: ") + str(e))
if real:
    sys.exit(1)
print("core preflight OK (feature-branch behind-main tolerance applied)")
PY
    phase_log preflight "issue-621 pinned-input preflight (sha pins + composition gate)"
    uv run python scripts/run_issue621_preflight.py \
        > "$LOG_DIR/issue-621-preflight.log" 2>&1 \
        || fail "issue-621 preflight FAILED (see issue-621-preflight.log)"
fi
phase_log preflight_done "preflight complete"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Smoke (Phase S) — train + smoke eval + gate; ONE authorized cap raise.
# ─────────────────────────────────────────────────────────────────────────────
SMOKE_SLUG="r1_read__florist__seed42"
run_smoke_train() {
    local epochs="$1"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_issue621_train.py \
        --phase smoke --gpu-id 0 --epochs "$epochs" \
        > "$LOG_DIR/issue-621-smoke-train-ep${epochs}.log" 2>&1
}

EPOCHS_CAP=16
phase_log smoke_train "1 cell ($SMOKE_SLUG), epochs cap $EPOCHS_CAP, band [5,12] nat"
smoke_rc=0
run_smoke_train "$EPOCHS_CAP" || smoke_rc=$?
if [[ $smoke_rc -ne 0 ]]; then
    band_missed=$(python3 -c "
import json
s = json.load(open('eval_results/issue_621/anchor_smoke/summary.json'))
print(int(bool(s.get('band_missed')) and s.get('bystanders_ok') and s.get('a_init_ok') and s.get('trajectory_ok')))
" 2>/dev/null || echo 0)
    if [[ "$band_missed" == "1" ]]; then
        EPOCHS_CAP=32
        phase_log smoke_train "band miss within cap 16 — §7 ONE authorized raise to 32, re-smoking"
        run_smoke_train "$EPOCHS_CAP" || fail "smoke FAILED at cap 32 — r=1 cannot reach [5,12] at matched scale; reportable capacity finding, NO lr raise (recipe)"
    else
        fail "smoke train gate FAILED (non-band criterion; see issue-621-smoke-train-ep16.log)"
    fi
fi

phase_log smoke_eval "shift_extract on the smoke cell (separate subprocess)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_issue621_eval.py \
    --mode shift_extract --cell-slug "$SMOKE_SLUG" --gpu-id 0 \
    > "$LOG_DIR/issue-621-smoke-eval.log" 2>&1 \
    || fail "smoke eval shift_extract FAILED (see issue-621-smoke-eval.log)"

phase_log smoke_gate "parity (±1 nat, #534) + §14 duty-2 wall re-projection"
gate_rc=0
uv run python scripts/i621_smoke_gate.py --epochs-cap "$EPOCHS_CAP" \
    > "$LOG_DIR/issue-621-smoke-gate.log" 2>&1 || gate_rc=$?
if [[ $gate_rc -eq 3 ]]; then
    fail "smoke gate: FULL-CAP sweep wall projection exceeds the fence — descope decision needed (plan §9 priority: drop bridge arm, then seed 256); NOT auto-descoping"
elif [[ $gate_rc -ne 0 ]]; then
    fail "smoke gate FAILED (see issue-621-smoke-gate.log + anchor_smoke/smoke_gate.json)"
fi
phase_log smoke_done "smoke PASS (gate + parity + projection)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sweep train — 29 remaining cells, 4-way CVD sharding.
# ─────────────────────────────────────────────────────────────────────────────
phase_log sweep_train "29 cells, 4 shards (CVD exported per shard + matching --gpu-id)"
pids=()
for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g uv run python scripts/run_issue621_train.py \
        --phase sweep --shard "$g" --num-shards 4 --gpu-id "$g" --skip-existing \
        --epochs "$EPOCHS_CAP" \
        > "$LOG_DIR/issue-621-sweep-shard${g}.log" 2>&1 &
    pids+=($!)
done
sweep_fail=0
for i in 0 1 2 3; do
    if ! wait "${pids[$i]}"; then
        phase_log sweep_train "shard $i FAILED (see issue-621-sweep-shard${i}.log)"
        sweep_fail=1
    fi
done
[[ $sweep_fail -eq 0 ]] || fail "sweep train: >=1 shard failed"
phase_log sweep_train_done "all 4 shards complete"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Context bank — generate (vLLM) then capture+upload (HF), separate procs.
# ─────────────────────────────────────────────────────────────────────────────
phase_log bank_generate "vLLM greedy, 21 contexts x 50 probes, 512-token cap"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue621_extract_context_bank.py \
    --step generate \
    > "$LOG_DIR/issue-621-bank-generate.log" 2>&1 \
    || fail "bank generate FAILED (see issue-621-bank-generate.log)"

phase_log bank_capture "HF capture, 3 positions x 5 taps, upload"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue621_extract_context_bank.py \
    --step capture --upload \
    > "$LOG_DIR/issue-621-bank-capture.log" 2>&1 \
    || fail "bank capture FAILED (see issue-621-bank-capture.log)"
phase_log bank_done "bank bundle uploaded + verified"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Eval — emission (vLLM) then shift_extract (HF), 4 shards each.
# ─────────────────────────────────────────────────────────────────────────────
for mode in emission shift_extract; do
    phase_log "eval_${mode}" "30 cells, 4 shards"
    pids=()
    for g in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$g uv run python scripts/run_issue621_eval.py \
            --mode "$mode" --all-cells --skip-existing \
            --shard "$g" --num-shards 4 --gpu-id "$g" \
            > "$LOG_DIR/issue-621-eval-${mode}-shard${g}.log" 2>&1 &
        pids+=($!)
    done
    eval_fail=0
    for i in 0 1 2 3; do
        if ! wait "${pids[$i]}"; then
            phase_log "eval_${mode}" "shard $i FAILED (see issue-621-eval-${mode}-shard${i}.log)"
            eval_fail=1
        fi
    done
    [[ $eval_fail -eq 0 ]] || fail "eval ${mode}: >=1 shard failed"
    phase_log "eval_${mode}_done" "complete"
done

# ─────────────────────────────────────────────────────────────────────────────
# 5. Uploads (raw completions / mixes / shift tensors / trajectories).
# ─────────────────────────────────────────────────────────────────────────────
phase_log upload "artifact classes -> HF data repo (fail-loud verified)"
uv run python scripts/i621_upload_artifacts.py \
    > "$LOG_DIR/issue-621-upload.log" 2>&1 \
    || fail "artifact upload FAILED (see issue-621-upload.log)"
phase_log upload_done "uploads verified"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Results sentinel (reproducibility card, Hub-verified) + 7. terminal.
# ─────────────────────────────────────────────────────────────────────────────
phase_log sentinel "writing epm:results sentinel with the reproducibility card"
uv run python scripts/i621_write_results_sentinel.py --sentinel-dir "$LOG_DIR" \
    > "$LOG_DIR/issue-621-sentinel.log" 2>&1 \
    || fail "results-sentinel write FAILED (see issue-621-sentinel.log)"

phase_log done "issue-621 pipeline complete"
