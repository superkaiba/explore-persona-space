#!/usr/bin/env bash
# Issue #650 pod/VM-side end-to-end pipeline launcher (rank-1 MLP read/write).
#
# Forked from scripts/run_issue621_pipeline.sh (origin/issue-621 @ 766f44c4).
# Sequence (PASS_UNIFIED: smoke IS sweep with --cells the two smoke cells):
#   0. Preflight — orchestrate.preflight --json with the documented LEGACY
#      behind-origin/main feature-branch tolerance (parse the WHOLE stdout,
#      fail only on errors OTHER than the behind-main line), then the
#      issue-650 pinned-input preflight (#612 claim pool + #621 marker-mix
#      sha pins + persona resolution). SKIP via EPM_SKIP_PREFLIGHT=1 only.
#   1. Smoke (§7 install gate): 2 cells (marker__low__seed42 +
#      sycophancy__low__seed42) via --phase smoke. Marker band-stop [5,12] +
#      a_init sanity; sycophancy pool build + train + a_init. On a marker
#      band miss the §7 fallback fires ONCE: marker epochs cap 16 -> 32.
#   2. Sweep train: 10 remaining cells, 4-way CUDA_VISIBLE_DEVICES sharding
#      (CVD exported per cell in the LAUNCHER env + matching --gpu-id — the
#      in-process clobber alone is defeated by import-time cuInit).
#   3. Context bank: --step generate (vLLM) then --step capture --upload
#      (HF hooks) as SEPARATE subprocesses (vLLM worker-orphan gotcha).
#   4. DV-4 base concept directions (sycophancy): issue650_concept_direction
#      per sycophancy seed pool (base-model activations; non-circular).
#   5. Eval: marker four-float slot reads + sycophancy agreement panel.
#   6. Uploads (raw completions / mixes / bank / concept tensors), fail-loud.
#   7. i650_write_results_sentinel.py — reproducibility card -> poll sentinel.
#   8. The SINGLE terminal [phase=done] line.
#
# Off-pod (VM): DV-1..5 analysis + base-SVD + max-matched null + figures run
# on the VM against uploaded artifacts (plan §9), NOT on this pod.
#
# Pod-side code NEVER shells out to scripts/task.py. [phase=done] appears
# EXACTLY once, at the success terminal (incident #545).

set -euo pipefail

ISSUE=650

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

export WANDB_PROJECT=issue_650_rank1_mlp_geometry
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=false

PIPE_LOG="$LOG_DIR/issue-650-pipeline.log"

phase_log() {
    # Single source of truth for [phase=...] markers (poll_pipeline.py).
    local line="[phase=$1] $(date -u +%Y-%m-%dT%H:%M:%SZ) $2"
    echo "$line"
    echo "$line" >>"$PIPE_LOG"
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
    "task_id": 650,
    "by": "issue650_pipeline",
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
    uv run python - >"$LOG_DIR/issue-650-preflight-core.log" 2>&1 <<'PY' || fail "core preflight FAILED (see issue-650-preflight-core.log)"
import json
import re
import subprocess
import sys

proc = subprocess.run(
    ["uv", "run", "python", "-m", "explore_persona_space.orchestrate.preflight", "--json"],
    capture_output=True,
    text=True,
)
payload = json.loads(proc.stdout)
errors = payload.get("errors") or []
behind = re.compile(r"behind origin/(main|issue-650)|git fetch origin failed")
real = [e for e in errors if not behind.search(str(e))]
for e in errors:
    print(("TOLERATED: " if behind.search(str(e)) else "ERROR: ") + str(e))
if real:
    sys.exit(1)
print("core preflight OK (feature-branch behind-main tolerance applied)")
PY
    phase_log preflight "issue-650 pinned-input preflight (sha pins + persona resolution)"
    uv run python scripts/run_issue650_preflight.py \
        >"$LOG_DIR/issue-650-preflight.log" 2>&1 \
        || fail "issue-650 preflight FAILED (see issue-650-preflight.log)"
fi
phase_log preflight_done "preflight complete"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Smoke (§7 install gate) — 2 cells; ONE authorized marker cap raise.
# ─────────────────────────────────────────────────────────────────────────────
run_smoke() {
    local marker_epochs="$1"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_issue650_train.py \
        --phase smoke --gpu-id 0 --marker-epochs "$marker_epochs" \
        >"$LOG_DIR/issue-650-smoke-train-ep${marker_epochs}.log" 2>&1
}

MARKER_EPOCHS=16
phase_log smoke_train "2 cells (marker__low + syco__low), marker cap $MARKER_EPOCHS"
smoke_rc=0
run_smoke "$MARKER_EPOCHS" || smoke_rc=$?
if [[ $smoke_rc -ne 0 ]]; then
    band_missed=$(python3 -c "
import json
s = json.load(open('eval_results/issue_650/anchor_smoke/summary.json'))
print(int(bool(s.get('marker_band_missed')) and s.get('marker_a_init_ok') and s.get('sycophancy_trained_ok')))
" 2>/dev/null || echo 0)
    if [[ "$band_missed" == "1" ]]; then
        MARKER_EPOCHS=32
        phase_log smoke_train "marker band miss within cap 16 — §7 ONE authorized raise to 32, re-smoking"
        run_smoke "$MARKER_EPOCHS" || fail "smoke FAILED at marker cap 32 — MLP rank-1 cannot reach [5,12]; reportable capacity finding, NO lr raise (recipe)"
    else
        fail "smoke train gate FAILED (non-band criterion; see issue-650-smoke-train-ep16.log)"
    fi
fi

# §7 sycophancy install half: agreement-panel eval on the smoke syco cell,
# then the >=+0.30 Delta-agree floor check.
phase_log smoke_syco_eval "agreement-panel eval on sycophancy__low__seed42"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_issue650_eval.py \
    --phase smoke --cells sycophancy__low__seed42 \
    >"$LOG_DIR/issue-650-smoke-syco-eval.log" 2>&1 \
    || fail "smoke sycophancy eval FAILED (see issue-650-smoke-syco-eval.log)"
phase_log smoke_done "smoke PASS (marker band + a_init + syco trained/eval)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sweep train — 10 remaining cells, 4-way CVD sharding.
# ─────────────────────────────────────────────────────────────────────────────
phase_log sweep_train "10 cells, 4 shards (CVD exported per shard + matching --gpu-id)"
pids=()
for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g uv run python scripts/run_issue650_train.py \
        --phase sweep --shard "$g" --num-shards 4 --gpu-id "$g" --skip-existing \
        --marker-epochs "$MARKER_EPOCHS" \
        >"$LOG_DIR/issue-650-sweep-shard${g}.log" 2>&1 &
    pids+=($!)
done
sweep_fail=0
for i in 0 1 2 3; do
    if ! wait "${pids[$i]}"; then
        phase_log sweep_train "shard $i FAILED (see issue-650-sweep-shard${i}.log)"
        sweep_fail=1
    fi
done
[[ $sweep_fail -eq 0 ]] || fail "sweep train: >=1 shard failed"
phase_log sweep_train_done "all 4 shards complete"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Context bank — generate (vLLM) then capture+upload (HF), separate procs.
# ─────────────────────────────────────────────────────────────────────────────
phase_log bank_generate "vLLM greedy, 21 contexts x 50 probes, 768-token cap"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue650_extract_context_bank.py \
    --step generate \
    >"$LOG_DIR/issue-650-bank-generate.log" 2>&1 \
    || fail "bank generate FAILED (see issue-650-bank-generate.log)"
phase_log bank_capture "HF capture, 3 positions x 5 taps, upload"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue650_extract_context_bank.py \
    --step capture --upload \
    >"$LOG_DIR/issue-650-bank-capture.log" 2>&1 \
    || fail "bank capture FAILED (see issue-650-bank-capture.log)"
phase_log bank_done "bank bundle uploaded + verified"

# ─────────────────────────────────────────────────────────────────────────────
# 4. DV-4 base concept directions (sycophancy seeds; base-model, non-circular).
# ─────────────────────────────────────────────────────────────────────────────
for seed in 42 137 256; do
    pool_dir="eval_results/issue_650/training_mixes/sycophancy/seed${seed}"
    if [[ -f "$pool_dir/disagree_completions.json" ]]; then
        phase_log concept_dir "d_behavior_base + d_format_base (seed ${seed})"
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue650_concept_direction.py \
            --pool-dir "$pool_dir" \
            --out "eval_results/issue_650/concept/concept_directions_seed${seed}.pt" \
            >"$LOG_DIR/issue-650-concept-seed${seed}.log" 2>&1 \
            || fail "concept-direction seed ${seed} FAILED (see issue-650-concept-seed${seed}.log)"
    fi
done
phase_log concept_done "DV-4 base concept directions built"

# ─────────────────────────────────────────────────────────────────────────────
# 4b. R_persona generation (blocker marker-eval-r-persona-missing) — base-model
#     greedy responses over (eval panel × EVAL_QUESTIONS) that the marker eval
#     forwards as R_persona(q). MUST run BEFORE phase 5; coverage gate aborts
#     the pipeline here (not the GPU-spent marker eval) on any gap.
# ─────────────────────────────────────────────────────────────────────────────
phase_log r_persona "base greedy R over eval panel × 20 questions (marker eval input)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_issue650_generate_r_persona.py \
    --phase sweep \
    >"$LOG_DIR/issue-650-r-persona.log" 2>&1 \
    || fail "R_persona generation/coverage FAILED (see issue-650-r-persona.log)"
phase_log r_persona_done "R_persona coverage verified for the full eval panel"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Eval — marker four-float slot reads + sycophancy agreement panel.
# ─────────────────────────────────────────────────────────────────────────────
phase_log eval "12 cells: marker slot reads + sycophancy agreement panel"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_issue650_eval.py \
    --phase sweep \
    >"$LOG_DIR/issue-650-eval.log" 2>&1 \
    || fail "eval FAILED (see issue-650-eval.log)"
phase_log eval_done "eval complete"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Uploads (raw completions / mixes / bank / concept tensors), fail-loud.
# ─────────────────────────────────────────────────────────────────────────────
phase_log upload "raw completions + analysis tensors -> HF data repo"
uv run python - >"$LOG_DIR/issue-650-upload.log" 2>&1 <<'PY' || fail "artifact upload FAILED (see issue-650-upload.log)"
from pathlib import Path

from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

# Raw completion files the eval loop persists (fail-loud helper per
# experiment-implementer Upload-wiring contract).
upload_raw_completions_to_data_repo(
    experiment_name="issue650_rank1_mlp_geometry",
    eval_results_dir=Path("eval_results/issue_650"),
)
print("raw completions uploaded")
PY
phase_log upload_done "uploads verified"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Results sentinel (reproducibility card, Hub-verified) + 8. terminal.
# ─────────────────────────────────────────────────────────────────────────────
phase_log sentinel "writing epm:results sentinel with the reproducibility card"
uv run python scripts/i650_write_results_sentinel.py --sentinel-dir "$LOG_DIR" \
    >"$LOG_DIR/issue-650-sentinel.log" 2>&1 \
    || fail "results-sentinel write FAILED (see issue-650-sentinel.log)"

phase_log done "issue-650 pipeline complete"
