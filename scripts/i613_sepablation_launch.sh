#!/usr/bin/env bash
# Task #613 follow-up round `sep-ablation` — launch driver (amendment plan §3).
#
# Trains BOTH arms of the flag A/B INSIDE the no-separator positive
# construction (marker_sep="": positives are R + " ※", so the negative loss
# slot, the marker slot, and the greedy stop position coincide at post-R):
#   sepablation_flagon_200p800n   (alive negatives, post-response-slot loss)
#   sepablation_flagoff_200p800n  (dead-slot comparator, trailing-token loss)
# seeds 42 + 137, sequentially on ONE GPU. Forked from scripts/i613_launch.sh
# (self-daemonizing supervisor, heartbeat, pid file, sentinel skips, #552
# preflight-JSON tolerance — verbatim); DISTINCT guard var + LOG/PID/PHASE
# names (issue-613-sepablation-*) so it can never collide with the parent
# script's pid file. The parent's p6 reused-adapter fetch + gauge-parity
# assert is DROPPED (no adapter reuse this round — both arms fresh).
#
# Runs, IN ORDER, aborting on any non-zero exit:
#   0. registry asserts (both sep cells: marker_sep="", T=63, suppress flag
#      A/B, conditional, band log-only, dense parity, single-variable pair)
#      + CPU test gate (collator + flagon-threading + sepablation tests).
#   1. preflight --json — whole-stdout parse; tolerates ONLY the documented
#      feature-branch false positive ("behind origin/main", incident #552).
#   2. fetch pinned parent inputs @ DATA_REV (dfce94df…) -> data/issue_613/
#      + copy the committed bystander panel to eval_results/issue_613/phase0/.
#   3. flagon seed-42 unit (scripts/i601_run_cell.py: build -> fused-surface
#      assert -> train -> on-policy eval @ sep="" -> dense 1..63 read @
#      sep="" -> checkpoint upload).
#   4. seed-42 SMOKE GATE (scripts/i613_sepablation_smoke_gate.py — the smoke
#      IS the sweep with one cell): terminal==63; 3-channel rowtype; R1'
#      step-1 neg_slot CE >= 1e-3; positive-slot sanity step-1 CE in [10,30]
#      AND ce[10] < ce[1]; WandB rowtype_ce/* series present; fused-surface
#      assert 200/200 recorded. Unit order leaves a complete seed-42 A/B pair
#      on any mid-run failure.
#   5. flagoff seed-42 -> flagon seed-137 -> flagoff seed-137 units.
#   6. OPTIONAL exploratory sep-marker terminal read on the new cells
#      (off-training slot, cross-construction shape only; skipped when the
#      local terminal adapter is absent or I613_SEPABL_SKIP_EXPLORATORY=1).
#   7. finalize — raw completions -> HF issue613_sep_ablation/raw_completions/
#      + results sentinel $LOG_DIR/issue-613-sepablation-results.json
#      (epm:results card with explicit reproducibility_card: per-cell
#      adapter_paths under adapters/issue_613/, wandb_project issue613,
#      wandb_run_names).
#
# Launch (GCP instance / pod, repo at issue-613 HEAD):
#   bash scripts/i613_sepablation_launch.sh
set -euo pipefail

LOG_DIR="${LOG_DIR:-/workspace/logs}"
SLAB_ROOT="${SLAB_ROOT:-eval_results/issue_613}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/runs/issue_613}"
DATA_DIR="${DATA_DIR:-data/issue_613}"
GPU_ID="${GPU_ID:-0}"
FLAGON_CELL="sepablation_flagon_200p800n"
FLAGOFF_CELL="sepablation_flagoff_200p800n"
PHASE_DIR="sep-ablation"   # CellSpec601.phase — the output dir under SLAB_ROOT
HF_PREFIX="adapters/issue_613"
RUN_NAME_PREFIX="issue613"
SENTINEL_TASK_ID=613
DATA_REV="dfce94df6a3f326d0f4f366864321942842c7164"     # parent inputs (plan §7 Data pin)
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/issue-613-sepablation-launch.log"   # poller-pinned main log
PID_FILE="$LOG_DIR/issue-613-sepablation-launch.pid"
PHASE_FILE="$LOG_DIR/issue-613-sepablation-launch.phase"

# ── Self-daemonization + relaunch guard (i613_launch.sh pattern verbatim;
# DISTINCT guard var so the parent script's supervisor can never adopt us) ────
if [ -z "${I613_SEPABL_SUPERVISED:-}" ]; then
    if ! setsid --version >/dev/null 2>&1; then
        echo "[launcher] FATAL: 'setsid --version' failed — util-linux setsid (with --fork) required"
        exit 4
    fi
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
        echo "[launcher] already running pid=$(cat "$PID_FILE") — refusing relaunch (exit 3)"
        exit 3
    fi
    export I613_SEPABL_SUPERVISED=1
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
# list before sft.py's in-process clobber) + MooseFS-quota parity (checkpoints
# upload in ONE bulk commit at unit end) + the issue<N> WandB project
# convention so the results card's wandb_project is mechanical.
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD="${EPM_SKIP_INLINE_CHECKPOINT_UPLOAD:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-issue613}"

HB_INTERVAL="${I613_HB_INTERVAL:-120}"
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
echo "[phase=p0_registry_assert] $(date -u +%FT%TZ) asserting sep-ablation cell registration + branch"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ "$BRANCH" = "main" ]; then
    echo "[phase=p0_registry_assert] FATAL: checkout is on 'main' — the #613 rig lives on issue-613 (plan §7)"
    exit 5
fi
echo "[phase=p0_registry_assert] branch=$BRANCH sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
uv run python - "$FLAGON_CELL" "$FLAGOFF_CELL" "$PHASE_DIR" <<'PY'
import dataclasses
import sys

from explore_persona_space.experiments.neg_setpoint_601 import (
    cell_by_slug,
    cells_for_request,
)

cell_on, cell_off, phase_dir = sys.argv[1], sys.argv[2], sys.argv[3]
on = cell_by_slug(cell_on)  # KeyError on an unregistered slug = fail loud (main checkout)
off = cell_by_slug(cell_off)
for spec, suppress in ((on, True), (off, False)):
    assert spec.marker_sep == "", f"{spec.slug}: marker_sep={spec.marker_sep!r} != '' (THE round variable)"
    assert spec.suppress_negatives is suppress, f"{spec.slug}: suppress_negatives != {suppress}"
    assert spec.expected_steps == 63, f"{spec.slug}: expected_steps={spec.expected_steps} != 63"
    assert spec.conditional is True, f"{spec.slug}: must be conditional (explicit-slug-only)"
    assert spec.phase == phase_dir, f"{spec.slug}: phase={spec.phase!r} != {phase_dir!r}"
    assert spec.band_stop is True and spec.band_log_only is True, (
        f"{spec.slug}: band config drifted — D1 requires log-only"
    )
    assert (spec.pos_ex, spec.n_neg_personas, spec.neg_ex_per_persona) == (200, 4, 200)
    assert spec.onpolicy == "anchors" and spec.seeds == (42, 137)
    assert spec.slug not in {c.slug for c in cells_for_request("all")}, "leaked into --cells all"
    assert spec.slug not in {c.slug for c in cells_for_request("phase4b")}, "leaked into phase4b"
    assert [c.slug for c in cells_for_request(spec.slug)] == [spec.slug]
# Dense-ladder + recipe parity with the parent dense_200p800n recipe.
parent = cell_by_slug("dense_200p800n")
assert on.dense_steps == off.dense_steps == parent.dense_steps, "dense ladder drifted"
assert (on.lr, on.epochs, on.lora_targets) == (parent.lr, parent.epochs, parent.lora_targets)
# Within-construction single-variable pair: the two sep cells differ ONLY in
# slug/plain_name/suppress_negatives.
diff = {
    f.name
    for f in dataclasses.fields(on)
    if getattr(on, f.name) != getattr(off, f.name)
}
assert diff == {"slug", "plain_name", "suppress_negatives"}, (
    f"sep-ablation pair differs in {sorted(diff)} — must be single-variable"
)
print(
    f"registry assert PASS: {cell_on} / {cell_off} marker_sep='' T=63 "
    f"suppress A/B=({on.suppress_negatives},{off.suppress_negatives}) "
    f"dense_steps={on.dense_steps} seeds={on.seeds}"
)
PY
echo "[phase=p0_registry_assert] running collator + threading + sepablation test gate (CPU)"
uv run pytest tests/test_marker_only_collator_post_response_slot.py \
    tests/test_i613_flagon_threading.py tests/test_i613_sepablation.py -x -q

set_phase p1_preflight
echo "[phase=p1_preflight] $(date -u +%FT%TZ) running preflight --json (sub-log: $LOG_DIR/issue-613-sepablation-preflight.json)"
# Exit code deliberately swallowed HERE ONLY: the parser below re-raises on
# any error other than the documented feature-branch false positive (#552).
uv run python -m explore_persona_space.orchestrate.preflight --json \
    > "$LOG_DIR/issue-613-sepablation-preflight.json" \
    2> "$LOG_DIR/issue-613-sepablation-preflight.err" || true
uv run python - "$LOG_DIR/issue-613-sepablation-preflight.json" <<'PY'
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
echo "[phase=p2_fetch] $(date -u +%FT%TZ) fetching pinned parent inputs @ $DATA_REV (idempotent)"
uv run python - "$DATA_DIR" "$DATA_REV" <<'PY'
import os
import shutil
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # cwd-walking resolver — safe in heredocs (research-project-structure.md)

from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.experiments.neg_setpoint_601 import (  # noqa: E402
    HF_DATA_REPO,
    PARENT_DATA_FILES,
)

data_dir, rev = Path(sys.argv[1]), sys.argv[2]
for repo_path, parent_local_rel in PARENT_DATA_FILES:
    # PARENT_DATA_FILES maps to data/issue_601/...; #613 owns data/issue_613/
    # — same basenames, issue-local root.
    local = data_dir / Path(parent_local_rel).relative_to("data/issue_601")
    if local.exists():
        print(f"[fetch] present: {local}")
        continue
    local.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=repo_path,
        revision=rev,  # content-addressed pin — closes the #600 stale-mirror class
        token=os.environ.get("HF_TOKEN"),
    )
    shutil.copyfile(got, local)
    print(f"[fetch] {repo_path}@{rev[:9]} -> {local}")
PY
mkdir -p "$SLAB_ROOT/phase0"
if [ ! -f "$SLAB_ROOT/phase0/bystander_panel.json" ]; then
    cp eval_results/issue_601/phase0/bystander_panel.json "$SLAB_ROOT/phase0/bystander_panel.json"
fi
echo "[phase=p2_fetch] bystander panel staged at $SLAB_ROOT/phase0/bystander_panel.json"

FLAGON42_SKIPPED=0
run_unit() {
    UNIT_CELL="$1"
    SEED="$2"
    TRAJ="$SLAB_ROOT/$PHASE_DIR/${UNIT_CELL}_seed${SEED}/trajectory.json"
    if [ -f "$TRAJ" ]; then
        echo "[unit cell=$UNIT_CELL seed=$SEED] trajectory exists — skip-cheap resume ($TRAJ)"
        # Cross-instance resume: WandB run dir + build manifest live on the
        # ORIGINAL instance; the smoke gate relaxes ONLY its local-series /
        # local-manifest checks on this flag (i613_launch.sh pattern).
        if [ "$UNIT_CELL" = "$FLAGON_CELL" ] && [ "$SEED" = "42" ]; then FLAGON42_SKIPPED=1; fi
    else
        echo "[unit cell=$UNIT_CELL seed=$SEED] $(date -u +%FT%TZ) launching unit (sub-log: $LOG_DIR/issue-613-sepablation-${UNIT_CELL}-seed${SEED}.log)"
        uv run python scripts/i601_run_cell.py \
            --cell "$UNIT_CELL" --seed "$SEED" --gpu-id "$GPU_ID" \
            --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" \
            --data-dir "$DATA_DIR" --runs-root "$RUNS_ROOT" \
            --hf-prefix "$HF_PREFIX" --run-name-prefix "$RUN_NAME_PREFIX" \
            --sentinel-task-id "$SENTINEL_TASK_ID" \
            > "$LOG_DIR/issue-613-sepablation-${UNIT_CELL}-seed${SEED}.log" 2>&1
        test -f "$TRAJ" || { echo "unit $UNIT_CELL seed $SEED exited 0 but $TRAJ missing"; exit 1; }
    fi
}

set_phase p3_flagon_seed42
run_unit "$FLAGON_CELL" 42

set_phase p4_smoke_gate
echo "[phase=p4_smoke_gate] $(date -u +%FT%TZ) flagon seed-42 smoke gate (smoke IS the sweep with one cell)"
SMOKE_ARGS=(
    --cell-dir "$SLAB_ROOT/$PHASE_DIR/${FLAGON_CELL}_seed42"
    --checkpoint-index "$RUNS_ROOT/${FLAGON_CELL}_seed42/checkpoint_index.json"
    --run-name "${RUN_NAME_PREFIX}_${FLAGON_CELL}_seed42"
    --expect-terminal-step 63
    --expect-positives 200
)
if [ "$FLAGON42_SKIPPED" = "1" ]; then SMOKE_ARGS+=(--resumed); fi
uv run python scripts/i613_sepablation_smoke_gate.py "${SMOKE_ARGS[@]}"

set_phase p5_remaining_units
run_unit "$FLAGOFF_CELL" 42
run_unit "$FLAGON_CELL" 137
run_unit "$FLAGOFF_CELL" 137

set_phase p6_exploratory_sepmarker
if [ "${I613_SEPABL_SKIP_EXPLORATORY:-0}" = "1" ]; then
    echo "[phase=p6_exploratory_sepmarker] SKIP (I613_SEPABL_SKIP_EXPLORATORY=1)"
else
    # Optional cross-construction context read (plan §3 item 6): the new
    # cells' TERMINAL adapter re-read at the legacy sep-marker slot
    # (post-R+\n\n — an off-training slot for this construction; descriptive
    # shape only, never joins the within-round A/B). Uses the surviving local
    # terminal adapter (the ckpt tree is reaped post-upload); on a
    # cross-instance resume the local adapter is absent -> loud SKIP (the
    # read is optional/exploratory by registration).
    for ARM_CELL in "$FLAGON_CELL" "$FLAGOFF_CELL"; do
        for SEED in 42 137; do
            ADAPTER_DIR="$RUNS_ROOT/${ARM_CELL}_seed${SEED}/adapter"
            EXP_OUT="$SLAB_ROOT/$PHASE_DIR/${ARM_CELL}_seed${SEED}/sepmarker_terminal_exploratory.json"
            if [ -f "$EXP_OUT" ]; then
                echo "[phase=p6_exploratory_sepmarker] present — skip ($EXP_OUT)"
                continue
            fi
            if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
                echo "[phase=p6_exploratory_sepmarker] SKIP ${ARM_CELL}_seed${SEED}: no local terminal adapter at $ADAPTER_DIR (cross-instance resume)"
                continue
            fi
            EXP_IDX="$RUNS_ROOT/${ARM_CELL}_seed${SEED}/exploratory_terminal_index.json"
            printf '{"1.0000": {"step": 63, "path": "%s"}}\n' "$ADAPTER_DIR" > "$EXP_IDX"
            echo "[phase=p6_exploratory_sepmarker] sep-marker terminal read ${ARM_CELL}_seed${SEED} -> $EXP_OUT"
            uv run python scripts/i601_dense_read.py \
                --cell "$ARM_CELL" --seed "$SEED" \
                --checkpoint-index "$EXP_IDX" \
                --out-path "$EXP_OUT" \
                --data-dir "$DATA_DIR" \
                --bystander-panel-path "$SLAB_ROOT/phase0/bystander_panel.json" \
                --sep-mode marker --steps 63 \
                > "$LOG_DIR/issue-613-sepablation-exploratory-${ARM_CELL}-seed${SEED}.log" 2>&1
            test -f "$EXP_OUT" || { echo "exploratory read exited 0 but $EXP_OUT missing"; exit 1; }
        done
    done
fi

set_phase p7_finalize
echo "[phase=p7_finalize] $(date -u +%FT%TZ) raw-completions upload + results sentinel"
uv run python - "$SLAB_ROOT" "$PHASE_DIR" "$FLAGON_CELL" "$FLAGOFF_CELL" "$LOG_DIR" "$RUNS_ROOT" "$HF_PREFIX" "$RUN_NAME_PREFIX" <<'PY'
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.experiments.neg_setpoint_601 import (  # noqa: E402
    HF_DATA_REPO,
    HF_MODEL_REPO,
    cell_by_slug,
)
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload  # noqa: E402

slab_root, phase_dir, cell_on, cell_off, log_dir, runs_root, hf_prefix, run_prefix = (
    Path(sys.argv[1]),
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    Path(sys.argv[5]),
    Path(sys.argv[6]),
    sys.argv[7],
    sys.argv[8],
)
cells = [cell_on, cell_off]
seeds = list(cell_by_slug(cell_on).seeds)

# Raw completions -> the HF contract bucket (fail-loud per-file _upload,
# list_repo_files-verified inside the helper). Plan §7 Outputs:
# issue613_sep_ablation/raw_completions/...
raw_hf_paths: dict[str, str] = {}
for cell in cells:
    for seed in seeds:
        local = slab_root / phase_dir / f"{cell}_seed{seed}" / "raw_completions.json"
        if not local.exists():
            raise RuntimeError(f"raw completions missing at {local} — refusing to finalize")
        dest = f"issue613_sep_ablation/raw_completions/{cell}_seed{seed}/raw_completions.json"
        url = _upload(
            local_path=local,
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=dest,
            upload_as_file=True,
        )
        if not url:
            raise RuntimeError(f"raw-completions upload FAILED for {local} -> {dest}")
        raw_hf_paths[f"{cell}_seed{seed}"] = f"{DEFAULT_DATASET_REPO}/{dest}"
        print(f"[upload_raw] {cell} seed {seed} -> {dest}")

# Per-unit realized terminal step (run_cell hard-asserts ==63 in-process;
# re-read here so the results card carries the realized value) + build
# manifests (marker_sep echo + fused-surface assert counts).
realized: dict[str, int | None] = {}
manifests: dict[str, dict | None] = {}
for cell in cells:
    for seed in seeds:
        key = f"{cell}_seed{seed}"
        idx_path = runs_root / key / "checkpoint_index.json"
        realized[key] = (
            json.loads(idx_path.read_text()).get("1.0000", {}).get("step")
            if idx_path.exists()
            else None  # resumed unit on a fresh instance; trajectory is authoritative
        )
        mf_path = slab_root / phase_dir / key / "build_manifest.json"
        manifests[key] = json.loads(mf_path.read_text()) if mf_path.exists() else None

try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    sha = "unknown"

eval_paths = {
    f"{cell}_seed{seed}": {
        "trajectory": str(slab_root / phase_dir / f"{cell}_seed{seed}" / "trajectory.json"),
        "dense_trajectory": str(
            slab_root / phase_dir / f"{cell}_seed{seed}" / "dense_trajectory.json"
        ),
        "rowtype_ce": str(slab_root / phase_dir / f"{cell}_seed{seed}" / "rowtype_ce.json"),
        "inloop_band_trajectory": str(
            slab_root / phase_dir / f"{cell}_seed{seed}" / "inloop_band_trajectory.json"
        ),
        "build_manifest": str(
            slab_root / phase_dir / f"{cell}_seed{seed}" / "build_manifest.json"
        ),
        "sepmarker_terminal_exploratory": str(
            slab_root
            / phase_dir
            / f"{cell}_seed{seed}"
            / "sepmarker_terminal_exploratory.json"
        ),
    }
    for cell in cells
    for seed in seeds
}
spec_on = cell_by_slug(cell_on)
card = {
    "issue": 613,
    "round": "sep-ablation",
    "status": "done",
    "cells": cells,
    "seeds": seeds,
    "marker_sep": "",
    "expected_steps": spec_on.expected_steps,
    "realized_terminal_steps": realized,
    "band_config": {"band_stop": spec_on.band_stop, "band_log_only": spec_on.band_log_only},
    "build_manifests": manifests,
    "eval_paths": eval_paths,
    "raw_completions_hf_paths": raw_hf_paths,
    # Explicit reproducibility card (workflow.yaml § markers epm:results):
    # per-cell adapter_paths Hub-verified by each unit's fail-loud terminal
    # verify + bulk upload, wandb_project + wandb_run_names mandatory
    # structured fields.
    "reproducibility_card": {
        "adapter_paths": {
            f"{cell}_seed{seed}": f"{hf_prefix}/{cell}_seed{seed}"
            for cell in cells
            for seed in seeds
        },
        "hf_model_repo": HF_MODEL_REPO,
        "hf_data_repo": HF_DATA_REPO,
        "wandb_project": os.environ.get("WANDB_PROJECT", "issue613"),
        "wandb_run_names": [
            f"{run_prefix}_{cell}_seed{seed}" for cell in cells for seed in seeds
        ],
    },
    "final_commit_sha": sha,
    "hostname": socket.gethostname(),
    "timestamp_utc": datetime.now(UTC).isoformat(),
}
sentinel = log_dir / "issue-613-sepablation-results.json"
sentinel.write_text(
    json.dumps(
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 613,
            "by": "i613_sepablation_launch",
            "ts": datetime.now(UTC).isoformat(),
            "phase": "done",
            "note": json.dumps(card),
        },
        indent=2,
    )
)
print(f"results sentinel written -> {sentinel}")
PY

echo "[phase=done] $(date -u +%FT%TZ) issue-613 sep-ablation pipeline complete"
