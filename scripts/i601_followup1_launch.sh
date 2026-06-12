#!/usr/bin/env bash
# Task #601 — follow-up round 1 launch driver (label: posonly-multiepoch-schedule-closure).
#
# MINIMAL single-cell pipeline (plan v4 §3/§5): the negatives-free
# schedule-matched cell posonly_200p_T130 (200 positives, 0 negatives,
# 10 epochs -> T=130), seeds 42 then 137, sequentially on ONE GPU.
#
# Runs, IN ORDER, aborting on any non-zero exit:
#   0. registry assert  — posonly_200p_T130 resolvable, expected_steps==130,
#      band_stop=True band_log_only=True echoed (plan §6 risk 2), excluded
#      from --cells all AND the phase4b group
#   1. preflight --json — parsed; tolerates ONLY the documented feature-branch
#      false positive ("Local is N commit(s) behind origin/main") — a bare
#      `preflight || exit` under set -e kills every issue-branch launch
#      (incident #552). Whole-stdout JSON parse (never splitlines()[-1]).
#   2. fetch parent artifacts (persona_bank / centroids / R) — idempotent
#   3. seed 42 unit  — scripts/i601_run_cell.py (build -> train -> full6
#      on-policy eval -> dense 2..32 read -> checkpoint upload); skip-cheap
#      when its trajectory.json already exists
#   4. seed 137 unit — same
#   5. finalize — raw-completions upload to the plan §5 HF contract path
#      (issue601_neg_setpoint/raw_completions/followup_posonly/...) + the
#      results sentinel $LOG_DIR/issue-601-followup1-results.json
#      (epm:results card: adapter paths, eval paths, raw HF paths, wandb runs)
#
# Eval artifacts land at eval_results/issue_601/posonly-multiepoch-schedule-closure/
# posonly_200p_T130_seed<S>/ — the cell's registry phase string IS the follow-up
# label, so the default --slab-root produces the CLAUDE.md follow-up contract
# path AND keeps the committed phase0/bystander_panel.json resolvable for the
# dense read (run_cell passes <slab-root>/phase0/bystander_panel.json).
#
# Launch (pod, repo at issue-601 HEAD):
#   bash scripts/i601_followup1_launch.sh
# Self-daemonizing supervisor + heartbeat + pid file + sentinel skips: the
# proven i601_launch.sh block verbatim (plan v3 §D — launches died at
# ssh-session teardown; [hb] heartbeat every 120 s; single terminal
# [phase=done] line in the MAIN log only).
set -euo pipefail

LOG_DIR="${LOG_DIR:-/workspace/logs}"
SLAB_ROOT="${SLAB_ROOT:-eval_results/issue_601}"
GPU_ID="${GPU_ID:-0}"
CELL="posonly_200p_T130"
LABEL="posonly-multiepoch-schedule-closure"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/issue-601-followup1.log"   # poller-pinned main log
PID_FILE="$LOG_DIR/issue-601-followup1.pid"
PHASE_FILE="$LOG_DIR/issue-601-followup1.phase"

# ── Self-daemonization + relaunch guard (i601_launch.sh pattern verbatim) ────
if [ -z "${I601F1_SUPERVISED:-}" ]; then
    if ! setsid --version >/dev/null 2>&1; then
        echo "[launcher] FATAL: 'setsid --version' failed — util-linux setsid (with --fork) required"
        exit 4
    fi
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
        echo "[launcher] already running pid=$(cat "$PID_FILE") — refusing relaunch (exit 3)"
        exit 3
    fi
    export I601F1_SUPERVISED=1
    setsid --fork bash "$0" "$@" >> "$MAIN_LOG" 2>&1 < /dev/null
    echo "[launcher] detached supervised driver; main log: $MAIN_LOG; pid file: $PID_FILE"
    exit 0
fi

# ── SUPERVISED branch: own-pid file, heartbeat, combined EXIT trap ───────────
if [ -f "$PID_FILE" ]; then
    _old_pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$_old_pid" ] && [ "$_old_pid" != "$$" ] && kill -0 "$_old_pid" 2>/dev/null; then
        echo "[launcher] already running pid=$_old_pid — refusing relaunch (exit 3, supervised guard)"
        exit 3
    fi
fi
echo $$ > "$PID_FILE"

set_phase() {
    CURRENT_PHASE="$1"
    export CURRENT_PHASE
    printf '%s' "$1" > "$PHASE_FILE"
}
set_phase init

# Launcher-env CVD pin (gotcha #545: import-time cuInit freezes the device
# list before sft.py's in-process clobber) + MooseFS-quota parity with the
# parent dispatcher (checkpoints upload in ONE bulk commit at unit end).
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD="${EPM_SKIP_INLINE_CHECKPOINT_UPLOAD:-1}"

HB_INTERVAL="${I601_HB_INTERVAL:-120}"
(
    while true; do
        echo "[hb] $(date -u +%FT%TZ) pid=$$ phase=$(cat "$PHASE_FILE" 2>/dev/null || echo unset)"
        sleep "$HB_INTERVAL"
    done
) &
HB_PID=$!

on_exit() {
    rc=$?
    # STOP -> reap in-flight sleep child -> CONT -> TERM (a bare kill leaves
    # the sleep holding the stdout pipe; memory: bg heartbeat sleep child).
    kill -STOP "$HB_PID" 2>/dev/null || true
    pkill -P "$HB_PID" 2>/dev/null || true
    kill -CONT "$HB_PID" 2>/dev/null || true
    kill "$HB_PID" 2>/dev/null || true
    if [ "$rc" -ne 0 ]; then
        echo "[phase=abort] rc=$rc"
    fi
}
trap on_exit EXIT

set_phase p0_registry_assert
echo "[phase=p0_registry_assert] $(date -u +%FT%TZ) asserting follow-up cell registration"
uv run python - "$CELL" "$LABEL" <<'PY'
import sys

from explore_persona_space.experiments.neg_setpoint_601 import (
    cell_by_slug,
    cells_for_request,
)

cell, label = sys.argv[1], sys.argv[2]
spec = cell_by_slug(cell)  # KeyError on an unregistered slug = fail loud
assert spec.expected_steps == 130, f"expected_steps={spec.expected_steps} != 130"
assert spec.phase == label, f"phase={spec.phase!r} != follow-up label {label!r}"
assert spec.conditional is True, "cell must be conditional (explicit-slug-only)"
assert spec.band_stop is True and spec.band_log_only is True, (
    f"band config drifted: band_stop={spec.band_stop} band_log_only={spec.band_log_only} "
    f"— D1 requires log-only (a firing stop would unmatch the schedule, plan §6 risk 2)"
)
assert spec.pos_ex == 200 and spec.n_neg_personas == 0 and spec.neg_ex_per_persona == 0
assert cell not in {c.slug for c in cells_for_request("all")}, "leaked into --cells all"
assert cell not in {c.slug for c in cells_for_request("phase4b")}, "leaked into phase4b"
assert [c.slug for c in cells_for_request(cell)] == [cell], "explicit-slug resolution broken"
print(
    f"registry assert PASS: {cell} T={spec.expected_steps} epochs={spec.epochs} "
    f"lr={spec.lr} band_stop={spec.band_stop} band_log_only={spec.band_log_only} "
    f"onpolicy={spec.onpolicy} dense_steps={spec.dense_steps} seeds={spec.seeds}"
)
PY

set_phase p1_preflight
echo "[phase=p1_preflight] $(date -u +%FT%TZ) running preflight --json (sub-log: $LOG_DIR/issue-601-followup1-preflight.json)"
# Exit code deliberately swallowed HERE ONLY: the parser below re-raises on
# any error other than the documented feature-branch false positive (#552).
uv run python -m explore_persona_space.orchestrate.preflight --json \
    > "$LOG_DIR/issue-601-followup1-preflight.json" \
    2> "$LOG_DIR/issue-601-followup1-preflight.err" || true
uv run python - "$LOG_DIR/issue-601-followup1-preflight.json" <<'PY'
import json
import sys

raw = open(sys.argv[1]).read()  # whole stdout — the JSON is multi-line
report = json.loads(raw)
tolerated = [e for e in report["errors"] if "behind origin/main" in e]
real = [e for e in report["errors"] if "behind origin/main" not in e]
if real:
    raise SystemExit(f"preflight FAILED (beyond the feature-branch false positive): {real}")
print(f"preflight OK (errors tolerated: {tolerated or 'none'}; warnings: {report['warnings']})")
PY

set_phase p2_fetch
echo "[phase=p2_fetch] $(date -u +%FT%TZ) fetching parent artifacts (idempotent)"
uv run python - <<'PY'
from pathlib import Path

from explore_persona_space.experiments.neg_setpoint_601.artifacts import fetch_parent_data

fetched = fetch_parent_data(Path.cwd())
print(f"parent artifacts present: {len(fetched)}")
PY

STEP=3
for SEED in 42 137; do
    set_phase "p${STEP}_seed${SEED}"
    TRAJ="$SLAB_ROOT/$LABEL/${CELL}_seed${SEED}/trajectory.json"
    if [ -f "$TRAJ" ]; then
        echo "[phase=p${STEP}_seed${SEED}] trajectory exists — skip-cheap resume ($TRAJ)"
    else
        echo "[phase=p${STEP}_seed${SEED}] $(date -u +%FT%TZ) launching unit (sub-log: $LOG_DIR/issue-601-followup1-seed${SEED}.log)"
        uv run python scripts/i601_run_cell.py \
            --cell "$CELL" --seed "$SEED" --gpu-id "$GPU_ID" \
            --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" \
            > "$LOG_DIR/issue-601-followup1-seed${SEED}.log" 2>&1
        test -f "$TRAJ" || { echo "seed $SEED unit exited 0 but $TRAJ missing"; exit 1; }
    fi
    STEP=$((STEP + 1))
done

set_phase p5_finalize
echo "[phase=p5_finalize] $(date -u +%FT%TZ) raw-completions upload + results sentinel"
uv run python - "$SLAB_ROOT" "$LABEL" "$CELL" "$LOG_DIR" <<'PY'
import json
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.neg_setpoint_601 import (
    HF_ADAPTER_PREFIX_601,
    HF_DATA_PREFIX_601,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    cell_by_slug,
)
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

slab_root, label, cell, log_dir = (
    Path(sys.argv[1]),
    sys.argv[2],
    sys.argv[3],
    Path(sys.argv[4]),
)
spec = cell_by_slug(cell)
seeds = list(spec.seeds)

# Raw completions -> the plan §5 HF contract path (followup_posonly bucket).
# Explicit per-file _upload (fail-loud, list_repo_files-verified) because the
# rglob helper would key the rel path off the eval dir layout, not the
# contract bucket name.
raw_hf_paths: dict[str, str] = {}
for seed in seeds:
    local = slab_root / label / f"{cell}_seed{seed}" / "raw_completions.json"
    if not local.exists():
        raise RuntimeError(f"raw completions missing at {local} — refusing to finalize")
    dest = f"{HF_DATA_PREFIX_601}/raw_completions/followup_posonly/{cell}_seed{seed}/raw_completions.json"
    url = _upload(
        local_path=local,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        upload_as_file=True,
    )
    if not url:
        raise RuntimeError(f"raw-completions upload FAILED for {local} -> {dest}")
    raw_hf_paths[f"seed{seed}"] = f"{DEFAULT_DATASET_REPO}/{dest}"
    print(f"[upload_raw] seed {seed} -> {dest}")

# Per-seed realized terminal step (run_cell hard-asserts ==130 in-process;
# re-read here so the results card carries the realized value).
runs_root = Path("/workspace/runs/issue_601")
realized: dict[str, int | None] = {}
for seed in seeds:
    idx_path = runs_root / f"{cell}_seed{seed}" / "checkpoint_index.json"
    if idx_path.exists():
        realized[f"seed{seed}"] = json.loads(idx_path.read_text()).get("1.0000", {}).get("step")
    else:
        realized[f"seed{seed}"] = None  # resumed unit on a fresh pod; trajectory is authoritative

try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    sha = "unknown"

card = {
    "issue": 601,
    "followup_label": label,
    "status": "done",
    "cells": [cell],
    "seeds": seeds,
    "expected_steps": spec.expected_steps,
    "realized_terminal_steps": realized,
    "band_config": {"band_stop": spec.band_stop, "band_log_only": spec.band_log_only},
    "eval_paths": {
        f"seed{seed}": {
            "trajectory": str(slab_root / label / f"{cell}_seed{seed}" / "trajectory.json"),
            "dense_trajectory": str(
                slab_root / label / f"{cell}_seed{seed}" / "dense_trajectory.json"
            ),
            "rowtype_ce": str(slab_root / label / f"{cell}_seed{seed}" / "rowtype_ce.json"),
            "inloop_band_trajectory": str(
                slab_root / label / f"{cell}_seed{seed}" / "inloop_band_trajectory.json"
            ),
        }
        for seed in seeds
    },
    "adapter_paths": {
        f"seed{seed}": f"{HF_MODEL_REPO}/{HF_ADAPTER_PREFIX_601}/{cell}_seed{seed}"
        for seed in seeds
    },
    "raw_completions_hf_paths": raw_hf_paths,
    "wandb_run_names": [f"issue601_{cell}_seed{seed}" for seed in seeds],
    "hf_data_repo": HF_DATA_REPO,
    "hf_model_repo": HF_MODEL_REPO,
    "final_commit_sha": sha,
    "hostname": socket.gethostname(),
    "timestamp_utc": datetime.now(UTC).isoformat(),
}
sentinel = log_dir / "issue-601-followup1-results.json"
sentinel.write_text(
    json.dumps(
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 601,
            "by": "i601_followup1_launch",
            "ts": datetime.now(UTC).isoformat(),
            "phase": "done",
            "note": json.dumps(card),
        },
        indent=2,
    )
)
print(f"results sentinel written -> {sentinel}")
PY

echo "[phase=done] $(date -u +%FT%TZ) issue-601 followup1 pipeline complete"
