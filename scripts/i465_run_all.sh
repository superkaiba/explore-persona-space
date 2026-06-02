#!/bin/bash
# Issue #465 -- end-to-end driver for the 4-arm in-context-persona-spec
# marker-leakage experiment. Modeled on scripts/run_issue452_deconfound.sh:
# emits `[phase=done]` on success and writes a results sentinel JSON the
# /issue Step 7 contract requires.
#
# PRODUCTION mode (default):
#   bash scripts/i465_run_all.sh
#     phase0  preflight + Q_demo load (HF-FIRST + content_hash assert)
#     phase1  villain-R (130q) + helpful-R (50q); upload to HF
#     phase23 train 4 LoRAs (cond1 smoke gates cond2_k0/k1/k3 sweep);
#             per-cond adapters upload to HF
#     phase4  18-cell eval (5 reads x 4 conds) on Q_test
#     phase5  paired-bootstrap analysis (H1/H2/H3/H4/H5 + retention CIs)
#     sentinel + [phase=done]
#
# SMOKE mode (-S, --smoke -- the Step 6d.0-bis cond1-only end-to-end gate):
#   bash scripts/i465_run_all.sh --smoke
#     phase0  full (Q_demo is fixed 50)
#     phase1  full (130 + 50 forwards, ~8 min total)
#     phase23 --smoke-only (cond1 train + 3-gate smoke-check)
#     phase4  --conds cond1 (exercises EVERY code path in the eval rig
#             -- this is the #408-lesson "never-GPU-run eval" catch)
#     phase5  full (cheap CPU; runs end-to-end on the cond1-only cell set)
#     [phase=smoke-done] on success; fail sentinel + non-zero exit on fail.
#   The smoke leaves cond1's adapter + R + Q_demo persisted so a subsequent
#   production run reuses them and only trains cond2_k0/k1/k3.
#
# RESUME-FRIENDLY: each phase guards on its output artifact (Q_demo file,
# R artifacts, adapter on HF, per-cell eval JSON). On a re-run after a
# crash, completed phases are skipped/reused via the scripts' own resume
# logic (Phase 0 auto-loads existing q_demo; Phase 1 guarded at bash level;
# Phase 2/3 dispatcher skips cond1 on smoke-already-ran; Phase 4 --resume).
#
# Pod-side contract per CLAUDE.md: NEVER shells out to scripts/task.py.
# The orchestrator's poll_pipeline.py picks up the sentinel + [phase=done].

set -euo pipefail

REPO=/workspace/explore-persona-space
LOG_DIR=/workspace/logs/issue-465
mkdir -p "$LOG_DIR"

# Pre-flight: resolve repo root if not at the canonical pod path (dev VM).
if [[ ! -d "$REPO" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
    LOG_DIR="$REPO/logs/issue-465"
    mkdir -p "$LOG_DIR"
fi
cd "$REPO"

# --- arg parsing ---
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --smoke | -S) SMOKE=1 ;;
        *) echo "WARN: unknown arg: $arg" >&2 ;;
    esac
done

START_EPOCH=$(date -u +%s)
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "======== [$(ts)] $* ========"; echo; }

MODE_LABEL=$([ "$SMOKE" -eq 1 ] && echo "SMOKE" || echo "PRODUCTION")
banner "Issue #465 driver -- MODE=${MODE_LABEL}"
echo "[$(ts)] REPO=${REPO}"
echo "[$(ts)] LOG_DIR=${LOG_DIR}"

write_fail_sentinel() {
    local phase="$1"
    local reason="$2"
    local sentinel="/workspace/logs/issue-465-${MODE_LABEL,,}-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - "$phase" "$reason" "$sentinel" <<'PY'
import json, sys, datetime
phase, reason, sentinel = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "issue": 465,
    "phase": phase,
    "failure_class": "code",
    "reason": reason,
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open(sentinel, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote fail sentinel: {sentinel}")
PY
}

# Trap so any phase exit propagates into a fail sentinel for the poller.
trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_fail_sentinel "trap" "driver trap caught rc=${rc}"; fi' EXIT

# === Phase 0: preflight + Q_demo (HF-FIRST + content_hash assert) ============
banner "Phase 0 -- preflight + Q_demo"
PHASE0_LOG="$LOG_DIR/phase0.log"
phase0_rc=0
uv run python scripts/i465_phase0_preflight.py \
    > "$PHASE0_LOG" 2>&1 || phase0_rc=$?
if [[ "$phase0_rc" -ne 0 ]]; then
    write_fail_sentinel "phase0" "Phase 0 preflight exited rc=${phase0_rc} (see ${PHASE0_LOG})"
    exit "$phase0_rc"
fi
tail -5 "$PHASE0_LOG"
echo "[$(ts)] [phase=phase0_done]"

# === Phase 1: R generation (skip if artifacts already on disk) ===============
banner "Phase 1 -- on-policy R (villain x 130q + helpful x 50q)"
PHASE1_LOG="$LOG_DIR/phase1.log"
R_VILLAIN="data/issue_465/R_villain.json"
R_HELPFUL="data/issue_465/R_helpful_qtest.json"

phase1_rc=0
if [[ -f "$R_VILLAIN" && -f "$R_HELPFUL" ]]; then
    echo "[$(ts)] Phase 1 SKIP: $R_VILLAIN AND $R_HELPFUL already exist (resume)."
    echo "[$(ts)] -- to force a rebuild, rm them before re-running."
else
    split_arg="both"
    if [[ -f "$R_VILLAIN" && ! -f "$R_HELPFUL" ]]; then
        split_arg="helpful"
    elif [[ -f "$R_HELPFUL" && ! -f "$R_VILLAIN" ]]; then
        split_arg="villain"
    fi
    echo "[$(ts)] Phase 1 split=${split_arg}"
    uv run python scripts/i465_phase1_generate_R.py --split "$split_arg" \
        > "$PHASE1_LOG" 2>&1 || phase1_rc=$?
    if [[ "$phase1_rc" -ne 0 ]]; then
        write_fail_sentinel "phase1" "Phase 1 R-gen exited rc=${phase1_rc} (see ${PHASE1_LOG})"
        exit "$phase1_rc"
    fi
    tail -8 "$PHASE1_LOG"
fi
echo "[$(ts)] [phase=phase1_done]"

# === Phase 2/3: train (smoke = cond1 only; production = full sweep) ==========
banner "Phase 2/3 -- train 4 LoRAs (smoke gate on cond1 first)"
PHASE23_LOG="$LOG_DIR/phase23.log"
phase23_rc=0
if [[ "$SMOKE" -eq 1 ]]; then
    echo "[$(ts)] SMOKE: bash scripts/i465_phase23_dispatch.sh --smoke-only"
    bash scripts/i465_phase23_dispatch.sh --smoke-only \
        > "$PHASE23_LOG" 2>&1 || phase23_rc=$?
else
    echo "[$(ts)] PRODUCTION: bash scripts/i465_phase23_dispatch.sh"
    bash scripts/i465_phase23_dispatch.sh \
        > "$PHASE23_LOG" 2>&1 || phase23_rc=$?
fi
if [[ "$phase23_rc" -ne 0 ]]; then
    write_fail_sentinel "phase23" "Phase 2/3 train exited rc=${phase23_rc} (see ${PHASE23_LOG})"
    exit "$phase23_rc"
fi
tail -10 "$PHASE23_LOG"
echo "[$(ts)] [phase=phase23_done]"

# === Phase 4: eval (smoke = --conds cond1; production = all 4 + --resume) ====
banner "Phase 4 -- eval (5 reads x conds)"
PHASE4_LOG="$LOG_DIR/phase4.log"
phase4_rc=0
if [[ "$SMOKE" -eq 1 ]]; then
    echo "[$(ts)] SMOKE: scripts/i465_phase4_eval.py --conds cond1 (exercises every code path)"
    uv run python scripts/i465_phase4_eval.py --conds cond1 \
        > "$PHASE4_LOG" 2>&1 || phase4_rc=$?
else
    echo "[$(ts)] PRODUCTION: scripts/i465_phase4_eval.py --resume (all 4 conds)"
    uv run python scripts/i465_phase4_eval.py --resume \
        > "$PHASE4_LOG" 2>&1 || phase4_rc=$?
fi
if [[ "$phase4_rc" -ne 0 ]]; then
    write_fail_sentinel "phase4" "Phase 4 eval exited rc=${phase4_rc} (see ${PHASE4_LOG})"
    exit "$phase4_rc"
fi
tail -10 "$PHASE4_LOG"
echo "[$(ts)] [phase=phase4_done]"

# === Phase 5: analyze (cheap CPU; runs in both modes) ========================
banner "Phase 5 -- paired-bootstrap analysis"
PHASE5_LOG="$LOG_DIR/phase5.log"
phase5_rc=0
uv run python scripts/i465_phase5_analyze.py \
    > "$PHASE5_LOG" 2>&1 || phase5_rc=$?
if [[ "$phase5_rc" -ne 0 ]]; then
    write_fail_sentinel "phase5" "Phase 5 analyze exited rc=${phase5_rc} (see ${PHASE5_LOG})"
    exit "$phase5_rc"
fi
tail -5 "$PHASE5_LOG"
echo "[$(ts)] [phase=phase5_done]"

# === Smoke termination =======================================================
if [[ "$SMOKE" -eq 1 ]]; then
    banner "Smoke COMPLETE -- adapter for cond1 + R + Q_demo persisted; production can reuse"
    # Clear the trap before normal exit so we don't trip the fail-sentinel
    # path on a clean rc=0.
    trap - EXIT
    echo "[$(ts)] [phase=smoke-done]"
    exit 0
fi

# === Production: results sentinel + [phase=done] =============================
banner "Phase 6 -- write production results sentinel"
SENTINEL="/workspace/logs/issue-465-results.json"
END_EPOCH=$(date -u +%s)
ELAPSED_HOURS=$(awk "BEGIN {printf \"%.3f\", ($END_EPOCH - $START_EPOCH) / 3600.0}")

FINAL_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
WORKTREE_PATH="$REPO"

uv run python - "$REPO" "$SENTINEL" "$FINAL_SHA" "$WORKTREE_PATH" "$ELAPSED_HOURS" <<'PY'
"""Build the /issue Step 7 results sentinel for issue #465."""

import json
import os
import sys
from pathlib import Path

repo, sentinel_path, final_sha, worktree_path, elapsed_hours = sys.argv[1:6]
repo = Path(repo)

# --- analysis headline (eval_numbers) ---
analysis_path = repo / "eval_results/issue_465/analysis.json"
eval_numbers: dict = {}
if analysis_path.exists():
    a = json.loads(analysis_path.read_text())
    # H1 diagonal gates -- per-cond in_trained_shape ΔG + pass/fail.
    h1 = a.get("h1_diagonal_gates", {})
    eval_numbers["h1_diagonal_delta_g"] = {
        label: {
            "cond": row.get("cond"),
            "delta_g": row.get("delta_g_mean"),
            "ci_95": row.get("ci_95"),
            "threshold": row.get("threshold"),
            "pass": row.get("pass"),
        }
        for label, row in h1.items()
        if isinstance(row, dict)
    }
    # H3a/b/c -- disentangled demo-free-default contrasts.
    h3 = a.get("h3_disentangled", {})
    eval_numbers["h3_paired_contrasts"] = {
        label: {
            "cond_a": row.get("cond_a"),
            "cond_b": row.get("cond_b"),
            "diff_mean": row.get("diff_mean"),
            "ci_95": row.get("ci_95"),
            "excludes_zero": row.get("excludes_zero"),
            "n_paired": row.get("n_paired"),
        }
        for label, row in h3.items()
        if isinstance(row, dict)
    }
    # H3d -- co-primary retention CIs.
    eval_numbers["h3d_retention_ci"] = a.get("h3d_retention_ci", {})
    eval_numbers["retention_point_estimates"] = a.get("retention_point_estimates", {})
    # H2 generalization ratio.
    eval_numbers["h2_generalization"] = a.get("h2_generalization")
    # H4 k-sweep + H5 copy-vs-implant.
    eval_numbers["h4_k_sweep"] = a.get("h4_k_sweep")
    eval_numbers["h5_non_marker_demo"] = a.get("h5_non_marker_demo", {})

# --- eval_paths (repo-relative) ---
eval_paths: list[str] = []
if analysis_path.exists():
    eval_paths.append(str(analysis_path.relative_to(repo)))
retention_roll = repo / "eval_results/issue_465/analysis_retention.json"
if retention_roll.exists():
    eval_paths.append(str(retention_roll.relative_to(repo)))
per_cell_dir = repo / "eval_results/issue_465/per_cell"
if per_cell_dir.exists():
    eval_paths.extend(
        sorted(str(p.relative_to(repo)) for p in per_cell_dir.glob("G_*.json"))
    )
preflight_path = repo / "eval_results/issue_465/preflight.json"
if preflight_path.exists():
    eval_paths.append(str(preflight_path.relative_to(repo)))

# --- reproducibility_card ---
preflight = {}
if preflight_path.exists():
    preflight = json.loads(preflight_path.read_text())
reproducibility_card = {
    "base_model": preflight.get("base_model", "Qwen/Qwen2.5-7B-Instruct"),
    "marker_text": preflight.get("marker_text", " ※"),
    "marker_id": preflight.get("marker_id", 83399),
    "lora_recipe": {
        "r": 32, "alpha": 64, "dropout": 0.0,
        "lr": 1e-5, "epochs": 5, "bf16": True,
        "batch_size": 4, "grad_accum": 4, "max_length": 2048,
        "seed": 42, "marker_only_loss": True, "marker_tail_tokens": 0,
    },
    "conditions": ["cond1", "cond2_k0", "cond2_k1", "cond2_k3"],
    "n_q_train": preflight.get("n_q_train"),
    "n_q_test": preflight.get("n_q_test"),
    "n_q_demo": preflight.get("n_q_demo"),
    "q_demo_content_hash": preflight.get("q_demo_content_hash"),
    "q_train_content_hash": preflight.get("q_train_content_hash"),
    "q_test_content_hash": preflight.get("q_test_content_hash"),
    "qdemo_build_meta": preflight.get("qdemo_build_meta", {}),
    "lmsys_revision": preflight.get("build_stats", {}).get("source_dataset_revision"),
    "eval_demo_seed": 137,
    "train_demo_seed": 42,
    "bootstrap_n": 10000,
    "bootstrap_seed": 42,
}

# --- wandb_url + hf_hub_url ---
wandb_url = os.environ.get("WANDB_RUN_URL") or "https://wandb.ai/superkaiba/issue_465_incontext_persona_spec"
hf_hub_url = "https://huggingface.co/superkaiba1/explore-persona-space/tree/main/adapters"

# --- plan_deviations: empty unless the operator added entries to a side file ---
# Plan §10 prefigures one descope path (drop cond2_k3 first); the analyzer fills
# this in when promoting. Driver leaves it empty for now.
plan_deviations: list[dict] = []
deviations_path = repo / "eval_results/issue_465/plan_deviations.json"
if deviations_path.exists():
    plan_deviations = json.loads(deviations_path.read_text())

payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "issue": 465,
    "worktree_path": worktree_path,
    "final_commit_sha": final_sha,
    "wandb_url": wandb_url,
    "hf_hub_url": hf_hub_url,
    "gpu_hours_used": float(elapsed_hours),  # wall-time proxy; refine if needed
    "gpu_hours_budgeted": 1.9,
    "plan_deviations": plan_deviations,
    "eval_numbers": eval_numbers,
    "eval_paths": eval_paths,
    "reproducibility_card": reproducibility_card,
}
Path(sentinel_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
print(f"Wrote results sentinel: {sentinel_path}")
print(f"  eval_paths: {len(eval_paths)} files")
print(f"  gpu_hours_used (wall proxy): {elapsed_hours}")
PY

# Clear trap before clean exit so it doesn't fire on rc=0.
trap - EXIT

banner "Issue #465 PRODUCTION COMPLETE"
echo "[$(ts)] [phase=done]"
