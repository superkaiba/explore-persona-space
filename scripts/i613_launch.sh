#!/usr/bin/env bash
# Task #613 — alive-negatives flag A/B launch driver (parent #601).
#
# Trains the flag-on arm flagon_200p800n (dense_200p800n recipe with
# MarkerOnlyDataCollator suppress_at_post_response_slot=True), seeds 42 then
# 137, sequentially on ONE GPU, then re-reads the REUSED flag-off arm
# (adapters/issue_601/dense_200p800n_seed{42,137} @ the pinned HF rev) and
# both arms at the sep-plain LOSS slot. Forked from
# scripts/i601_followup1_launch.sh (self-daemonizing supervisor, heartbeat,
# pid file, sentinel skips, #552 preflight-JSON tolerance — verbatim).
#
# Runs, IN ORDER, aborting on any non-zero exit (plan #613 §4 step 5):
#   0. registry + collator asserts — flagon_200p800n resolvable,
#      expected_steps==63, suppress_negatives=True, band log-only echoed,
#      dense-ladder parity with dense_200p800n, excluded from --cells all /
#      phase4b; then the CPU test gate:
#      uv run pytest tests/test_marker_only_collator_post_response_slot.py
#                    tests/test_i613_flagon_threading.py -x
#   1. preflight --json — whole-stdout parse; tolerates ONLY the documented
#      feature-branch false positive ("behind origin/main", incident #552).
#   2. fetch pinned parent inputs @ DATA_REV (dfce94df…) -> data/issue_613/
#      + copy the committed bystander panel to eval_results/issue_613/phase0/.
#   3. seed-42 unit — scripts/i601_run_cell.py (build -> train -> anchors
#      on-policy eval -> dense 1..63 read -> checkpoint upload), #613 flags.
#   4. seed-42 SMOKE GATE (the smoke IS the sweep with one cell): realized
#      T==63; rowtype_ce.json carries the neg_slot channel with base CE;
#      R1 sanity step-1 neg_slot CE >= 1e-3 nats (else HALT-and-investigate);
#      WandB run issue613_flagon_200p800n_seed42 carries rowtype_ce/neg_slot_ce.
#   5. seed-137 unit — same command, --seed 137.
#   6. flag-off slot re-read — fetch dense_200p800n_seed{42,137} checkpoints
#      {1,5,10,20,32,45,63} @ ADAPTER_REV (4e6c92eb…); GAUGE-PARITY ASSERT
#      first (re-read the flag-off terminal at sep-marker; |dG - committed
#      dense terminal| <= 0.5 nat @ seed 42); then sep-plain reads on both
#      flag-off seeds AND on the new flag-on cells' HF checkpoint tree ->
#      eval_results/issue_613/slotread/<cell>_seed<S>/slot_trajectory.json.
#   7. finalize — raw completions -> HF issue613_flagon_ab/raw_completions/…
#      + results sentinel $LOG_DIR/issue-613-results.json (epm:results card
#      with the explicit reproducibility_card: per-cell adapter_paths,
#      wandb_project + wandb_run_names, eval/raw paths).
#
# Launch (GCP instance / pod, repo at issue-613 HEAD):
#   bash scripts/i613_launch.sh
set -euo pipefail

LOG_DIR="${LOG_DIR:-/workspace/logs}"
SLAB_ROOT="${SLAB_ROOT:-eval_results/issue_613}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/runs/issue_613}"
DATA_DIR="${DATA_DIR:-data/issue_613}"
GPU_ID="${GPU_ID:-0}"
CELL="flagon_200p800n"
FLAGOFF_CELL="dense_200p800n"
HF_PREFIX="adapters/issue_613"
RUN_NAME_PREFIX="issue613"
SENTINEL_TASK_ID=613
DATA_REV="dfce94df6a3f326d0f4f366864321942842c7164"     # parent inputs (plan §10 artifact 2)
ADAPTER_REV="4e6c92eb4846062f25b4b24b8d13dc1381222547"  # flag-off adapters (plan §10 artifact 1)
SLOT_STEPS="1,5,10,20,32,45,63"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/issue-613-launch.log"   # poller-pinned main log
PID_FILE="$LOG_DIR/issue-613-launch.pid"
PHASE_FILE="$LOG_DIR/issue-613-launch.phase"

# ── Self-daemonization + relaunch guard (i601_launch.sh pattern verbatim) ────
if [ -z "${I613_SUPERVISED:-}" ]; then
    if ! setsid --version >/dev/null 2>&1; then
        echo "[launcher] FATAL: 'setsid --version' failed — util-linux setsid (with --fork) required"
        exit 4
    fi
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
        echo "[launcher] already running pid=$(cat "$PID_FILE") — refusing relaunch (exit 3)"
        exit 3
    fi
    export I613_SUPERVISED=1
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
echo "[phase=p0_registry_assert] $(date -u +%FT%TZ) asserting #613 cell registration + branch"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ "$BRANCH" = "main" ]; then
    echo "[phase=p0_registry_assert] FATAL: checkout is on 'main' — the #613 rig lives on issue-613 (plan §8)"
    exit 5
fi
echo "[phase=p0_registry_assert] branch=$BRANCH sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
uv run python - "$CELL" "$FLAGOFF_CELL" <<'PY'
import sys

from explore_persona_space.experiments.neg_setpoint_601 import (
    cell_by_slug,
    cells_for_request,
)

cell, flagoff = sys.argv[1], sys.argv[2]
spec = cell_by_slug(cell)  # KeyError on an unregistered slug = fail loud (main checkout)
off = cell_by_slug(flagoff)
assert spec.expected_steps == 63, f"expected_steps={spec.expected_steps} != 63"
assert spec.suppress_negatives is True, "suppress_negatives must be True — THE #613 variable"
assert off.suppress_negatives is False, "flag-off comparator cell must stay suppress=False"
assert spec.conditional is True, "cell must be conditional (explicit-slug-only)"
assert spec.band_stop is True and spec.band_log_only is True, (
    f"band config drifted: band_stop={spec.band_stop} band_log_only={spec.band_log_only} "
    f"— D1 requires log-only (a firing stop would unmatch the matched-schedule A/B)"
)
assert (spec.pos_ex, spec.n_neg_personas, spec.neg_ex_per_persona) == (200, 4, 200)
assert spec.dense_steps == off.dense_steps, "dense ladder must be EXACT dense_200p800n parity"
assert spec.onpolicy == "anchors" and spec.seeds == (42, 137)
assert (spec.lr, spec.epochs, spec.lora_targets) == (off.lr, off.epochs, off.lora_targets), (
    "recipe drift vs the flag-off comparator — the A/B must be single-variable"
)
assert cell not in {c.slug for c in cells_for_request("all")}, "leaked into --cells all"
assert cell not in {c.slug for c in cells_for_request("phase4b")}, "leaked into phase4b"
assert [c.slug for c in cells_for_request(cell)] == [cell], "explicit-slug resolution broken"
print(
    f"registry assert PASS: {cell} T={spec.expected_steps} suppress_negatives="
    f"{spec.suppress_negatives} lr={spec.lr} band_stop={spec.band_stop} "
    f"band_log_only={spec.band_log_only} dense_steps={spec.dense_steps} seeds={spec.seeds}"
)
PY
echo "[phase=p0_registry_assert] running collator + threading test gate (CPU)"
uv run pytest tests/test_marker_only_collator_post_response_slot.py \
    tests/test_i613_flagon_threading.py -x -q

set_phase p1_preflight
echo "[phase=p1_preflight] $(date -u +%FT%TZ) running preflight --json (sub-log: $LOG_DIR/issue-613-preflight.json)"
# Exit code deliberately swallowed HERE ONLY: the parser below re-raises on
# any error other than the documented feature-branch false positive (#552).
uv run python -m explore_persona_space.orchestrate.preflight --json \
    > "$LOG_DIR/issue-613-preflight.json" \
    2> "$LOG_DIR/issue-613-preflight.err" || true
uv run python - "$LOG_DIR/issue-613-preflight.json" <<'PY'
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
    # (plan §4 step 2) — same basenames, issue-local root.
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

run_unit() {
    SEED="$1"
    TRAJ="$SLAB_ROOT/flagon_ab/${CELL}_seed${SEED}/trajectory.json"
    if [ -f "$TRAJ" ]; then
        echo "[unit seed=$SEED] trajectory exists — skip-cheap resume ($TRAJ)"
    else
        echo "[unit seed=$SEED] $(date -u +%FT%TZ) launching unit (sub-log: $LOG_DIR/issue-613-seed${SEED}.log)"
        uv run python scripts/i601_run_cell.py \
            --cell "$CELL" --seed "$SEED" --gpu-id "$GPU_ID" \
            --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" \
            --data-dir "$DATA_DIR" --runs-root "$RUNS_ROOT" \
            --hf-prefix "$HF_PREFIX" --run-name-prefix "$RUN_NAME_PREFIX" \
            --sentinel-task-id "$SENTINEL_TASK_ID" \
            > "$LOG_DIR/issue-613-seed${SEED}.log" 2>&1
        test -f "$TRAJ" || { echo "seed $SEED unit exited 0 but $TRAJ missing"; exit 1; }
    fi
}

set_phase p3_seed42
run_unit 42

set_phase p4_smoke_gate
echo "[phase=p4_smoke_gate] $(date -u +%FT%TZ) seed-42 smoke gate (smoke IS the sweep with one cell)"
uv run python - "$SLAB_ROOT" "$CELL" "$RUNS_ROOT" "$RUN_NAME_PREFIX" <<'PY'
import json
import sys
from pathlib import Path

slab_root, cell, runs_root, run_prefix = (
    Path(sys.argv[1]),
    sys.argv[2],
    Path(sys.argv[3]),
    sys.argv[4],
)
cell_dir = slab_root / "flagon_ab" / f"{cell}_seed42"

# 1) Realized terminal step == 63 (no band-stop fire). run_cell hard-asserts
#    this in-process; re-verify from the durable artifacts (checkpoint_index
#    when present — absent on a skip-cheap resume — else the dense terminal).
idx_path = runs_root / f"{cell}_seed42" / "checkpoint_index.json"
if idx_path.exists():
    step = json.loads(idx_path.read_text()).get("1.0000", {}).get("step")
else:
    dense = json.loads((cell_dir / "dense_trajectory.json").read_text())
    step = [c for c in dense["checkpoints"] if c["frac"] == 1.0][0]["step"]
assert step == 63, f"realized terminal step {step} != 63 — band-stop/schedule mis-wire"

# 2) rowtype_ce.json carries the neg_slot channel with base CE recorded.
rowtype = json.loads((cell_dir / "rowtype_ce.json").read_text())
assert "neg_slot_ce" in rowtype, "neg_slot channel MISSING from rowtype_ce.json"
assert rowtype.get("neg_slot_ce_base") is not None, "neg_slot base CE not recorded"
assert rowtype.get("n_neg_slot_rows", 0) > 0, "neg_slot channel has zero rows"

# 3) R1 sanity: step-1 neg_slot CE >= 1e-3 nats (plan §4 step 4 / §6 R1).
step1 = [r for r in rowtype["records"] if r["step"] == 1]
assert step1, "no step-1 record in rowtype_ce.json"
ce1 = step1[0].get("neg_slot_ce")
assert ce1 is not None, "step-1 record has no neg_slot_ce"
if ce1 < 1e-3:
    raise SystemExit(
        f"HALT-AND-INVESTIGATE (plan §4 step 4): step-1 neg_slot CE {ce1:.3e} < 1e-3 nats — "
        f"a dead relocated slot at step 1 most likely means a layout/tokenization bug; "
        f"cross-check tests/test_marker_only_collator_post_response_slot.py BEFORE "
        f"burning seed 137. base CE={rowtype['neg_slot_ce_base']:.3e}"
    )

# 4) WandB run carries the rowtype_ce/neg_slot_ce series (named smoke
#    telemetry for the declared guard). The run dir's config.yaml records
#    TrainingArguments.run_name; its wandb-summary.json records the last
#    value per logged key.
run_name = f"{run_prefix}_{cell}_seed42"
hits = []
for cfg in Path("wandb").glob("*run-*/files/config.yaml"):
    try:
        if run_name in cfg.read_text():
            summary = cfg.parent / "wandb-summary.json"
            if summary.exists() and "rowtype_ce/neg_slot_ce" in summary.read_text():
                hits.append(str(cfg.parent.parent))
    except OSError:
        continue
assert hits, (
    f"no WandB run dir for {run_name} carries the rowtype_ce/neg_slot_ce series "
    f"(searched wandb/*run-*/files/) — the declared R1 telemetry is not functioning"
)
print(
    f"smoke gate PASS: T=63; neg_slot rows={rowtype['n_neg_slot_rows']}; "
    f"step-1 neg_slot CE={ce1:.4f} (base {rowtype['neg_slot_ce_base']:.4f}); "
    f"wandb series present in {hits[0]}"
)
PY

set_phase p5_seed137
run_unit 137

set_phase p6_slotread
echo "[phase=p6_slotread] $(date -u +%FT%TZ) flag-off fetch @ $ADAPTER_REV + gauge parity + sep-plain reads"
# 6a: fetch both arms' checkpoint subsets + build local checkpoint indexes.
uv run python - "$RUNS_ROOT" "$SLAB_ROOT" "$CELL" "$FLAGOFF_CELL" "$ADAPTER_REV" "$HF_PREFIX" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.experiments.neg_setpoint_601 import (  # noqa: E402
    HF_ADAPTER_PREFIX_601,
    HF_MODEL_REPO,
)

runs_root, slab_root, cell_on, cell_off, adapter_rev, hf_prefix_on = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
)
# frac dir <-> optimizer step at T=63, floor 4dp (plan §13 assumption 12;
# matches step_fractions((1,5,10,20,32,45), 63, precision=4, rounding="floor")).
FRAC_STEPS = {
    "0.0158": 1,
    "0.0793": 5,
    "0.1587": 10,
    "0.3174": 20,
    "0.5079": 32,
    "0.7142": 45,
}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def fetch_tree(hf_prefix: str, cell: str, seed: int, rev: str | None) -> Path:
    """Fetch the {1..45} frac checkpoints + terminal for one (cell, seed); build index."""
    dest_root = runs_root / "slotread_ckpts" / f"{cell}_seed{seed}"
    index: dict[str, dict] = {}
    for frac, step in FRAC_STEPS.items():
        dest = dest_root / f"frac_{frac}"
        dest.mkdir(parents=True, exist_ok=True)
        for fname in ADAPTER_FILES:
            local = dest / fname
            if local.exists():
                continue
            got = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                filename=f"{hf_prefix}/{cell}_seed{seed}/checkpoints/frac_{frac}/{fname}",
                revision=rev,
                token=os.environ.get("HF_TOKEN"),
            )
            shutil.copyfile(got, local)
        index[frac] = {"step": step, "path": str(dest)}
    term = dest_root / "terminal"
    term.mkdir(parents=True, exist_ok=True)
    for fname in ADAPTER_FILES:
        local = term / fname
        if local.exists():
            continue
        got = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            filename=f"{hf_prefix}/{cell}_seed{seed}/{fname}",
            revision=rev,
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copyfile(got, local)
    index["1.0000"] = {"step": 63, "path": str(term)}
    idx_path = dest_root / "checkpoint_index.json"
    idx_path.write_text(json.dumps(index, indent=2))
    print(f"[fetch_tree] {cell}_seed{seed} @ {rev or 'main'} -> {idx_path}")
    return idx_path


for seed in (42, 137):
    # Flag-off arm: REUSED, revision-PINNED (fitness check (f), plan §10).
    fetch_tree(HF_ADAPTER_PREFIX_601, cell_off, seed, adapter_rev)
    # Flag-on arm: this run's own upload (local ckpts reaped post-upload) —
    # fetched back from its HF tree (plan §4 step 6 "or from their HF tree").
    fetch_tree(hf_prefix_on, cell_on, seed, None)
PY

# 6b: GAUGE-PARITY ASSERT (fitness check (g)) — re-read the flag-off seed-42
# terminal at sep-marker and compare to the committed dense terminal.
PARITY_OUT="$SLAB_ROOT/slotread/${FLAGOFF_CELL}_seed42/parity_terminal_marker.json"
if [ ! -f "$PARITY_OUT" ]; then
    uv run python scripts/i601_dense_read.py \
        --cell "$FLAGOFF_CELL" --seed 42 \
        --checkpoint-index "$RUNS_ROOT/slotread_ckpts/${FLAGOFF_CELL}_seed42/checkpoint_index.json" \
        --out-path "$PARITY_OUT" \
        --data-dir "$DATA_DIR" \
        --bystander-panel-path "$SLAB_ROOT/phase0/bystander_panel.json" \
        --sep-mode marker --steps 63 \
        > "$LOG_DIR/issue-613-parity-read.log" 2>&1
fi
uv run python - "$PARITY_OUT" <<'PY'
import json
import sys

new = json.loads(open(sys.argv[1]).read())
committed = json.loads(
    open("eval_results/issue_601/phase2/dense_200p800n_seed42/dense_trajectory.json").read()
)
dg_new = [c for c in new["checkpoints"] if c["frac"] == 1.0][0]["source_mean"]["delta_g"]
dg_committed = [c for c in committed["checkpoints"] if c["frac"] == 1.0][0]["source_mean"][
    "delta_g"
]
diff = abs(dg_new - dg_committed)
assert diff <= 0.5, (
    f"GAUGE-PARITY FAIL: flag-off seed-42 terminal sep-marker re-read dG={dg_new:.3f} vs "
    f"committed {dg_committed:.3f} (|diff|={diff:.3f} > 0.5 nat) — the apply-and-read gauge "
    f"drifted; do NOT trust any sep-plain number (plan §4 step 6 / §10 fitness (g))."
)
print(f"gauge parity PASS: re-read dG={dg_new:.3f} vs committed {dg_committed:.3f} (|diff|={diff:.3f} <= 0.5)")
PY

# 6c: sep-plain LOSS-slot reads — both arms x both seeds.
for ARM_CELL in "$FLAGOFF_CELL" "$CELL"; do
    for SEED in 42 137; do
        SLOT_OUT="$SLAB_ROOT/slotread/${ARM_CELL}_seed${SEED}/slot_trajectory.json"
        echo "[phase=p6_slotread] sep-plain read ${ARM_CELL}_seed${SEED} -> $SLOT_OUT"
        uv run python scripts/i601_dense_read.py \
            --cell "$ARM_CELL" --seed "$SEED" \
            --checkpoint-index "$RUNS_ROOT/slotread_ckpts/${ARM_CELL}_seed${SEED}/checkpoint_index.json" \
            --out-path "$SLOT_OUT" \
            --data-dir "$DATA_DIR" \
            --bystander-panel-path "$SLAB_ROOT/phase0/bystander_panel.json" \
            --sep-mode plain --steps "$SLOT_STEPS" \
            > "$LOG_DIR/issue-613-slotread-${ARM_CELL}-seed${SEED}.log" 2>&1
        test -f "$SLOT_OUT" || { echo "slot read exited 0 but $SLOT_OUT missing"; exit 1; }
    done
done

set_phase p7_finalize
echo "[phase=p7_finalize] $(date -u +%FT%TZ) raw-completions upload + results sentinel"
uv run python - "$SLAB_ROOT" "$CELL" "$FLAGOFF_CELL" "$LOG_DIR" "$RUNS_ROOT" "$HF_PREFIX" "$RUN_NAME_PREFIX" "$ADAPTER_REV" <<'PY'
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
    HF_ADAPTER_PREFIX_601,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    cell_by_slug,
)
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload  # noqa: E402

slab_root, cell, cell_off, log_dir, runs_root, hf_prefix, run_prefix, adapter_rev = (
    Path(sys.argv[1]),
    sys.argv[2],
    sys.argv[3],
    Path(sys.argv[4]),
    Path(sys.argv[5]),
    sys.argv[6],
    sys.argv[7],
    sys.argv[8],
)
spec = cell_by_slug(cell)
seeds = list(spec.seeds)

# Raw completions -> the plan §12 HF contract bucket (fail-loud per-file
# _upload, list_repo_files-verified inside the helper).
raw_hf_paths: dict[str, str] = {}
for seed in seeds:
    local = slab_root / "flagon_ab" / f"{cell}_seed{seed}" / "raw_completions.json"
    if not local.exists():
        raise RuntimeError(f"raw completions missing at {local} — refusing to finalize")
    dest = f"issue613_flagon_ab/raw_completions/{cell}_seed{seed}/raw_completions.json"
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

# Per-seed realized terminal step (run_cell hard-asserts ==63 in-process;
# re-read here so the results card carries the realized value).
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

eval_paths = {
    f"seed{seed}": {
        "trajectory": str(slab_root / "flagon_ab" / f"{cell}_seed{seed}" / "trajectory.json"),
        "dense_trajectory": str(
            slab_root / "flagon_ab" / f"{cell}_seed{seed}" / "dense_trajectory.json"
        ),
        "rowtype_ce": str(slab_root / "flagon_ab" / f"{cell}_seed{seed}" / "rowtype_ce.json"),
        "inloop_band_trajectory": str(
            slab_root / "flagon_ab" / f"{cell}_seed{seed}" / "inloop_band_trajectory.json"
        ),
        "slot_trajectory_flagon": str(
            slab_root / "slotread" / f"{cell}_seed{seed}" / "slot_trajectory.json"
        ),
        "slot_trajectory_flagoff": str(
            slab_root / "slotread" / f"{cell_off}_seed{seed}" / "slot_trajectory.json"
        ),
    }
    for seed in seeds
}
card = {
    "issue": 613,
    "status": "done",
    "cells": [cell],
    "seeds": seeds,
    "suppress_negatives": True,
    "expected_steps": spec.expected_steps,
    "realized_terminal_steps": realized,
    "band_config": {"band_stop": spec.band_stop, "band_log_only": spec.band_log_only},
    "eval_paths": eval_paths,
    "raw_completions_hf_paths": raw_hf_paths,
    "flagoff_reuse": {
        "adapter_paths": {
            f"seed{seed}": f"{HF_ADAPTER_PREFIX_601}/{cell_off}_seed{seed}" for seed in seeds
        },
        "hf_revision": adapter_rev,
        "committed_eval_root": "eval_results/issue_601/phase2",
    },
    # Explicit reproducibility card (workflow.yaml § markers epm:results):
    # per-cell adapter_paths Hub-verified by the unit's fail-loud bulk upload,
    # wandb_project + wandb_run_names mandatory structured fields.
    "reproducibility_card": {
        "adapter_paths": {
            f"{cell}_seed{seed}": f"{hf_prefix}/{cell}_seed{seed}" for seed in seeds
        },
        "hf_model_repo": HF_MODEL_REPO,
        "hf_data_repo": HF_DATA_REPO,
        "wandb_project": os.environ.get("WANDB_PROJECT", "issue613"),
        "wandb_run_names": [f"{run_prefix}_{cell}_seed{seed}" for seed in seeds],
    },
    "final_commit_sha": sha,
    "hostname": socket.gethostname(),
    "timestamp_utc": datetime.now(UTC).isoformat(),
}
sentinel = log_dir / "issue-613-results.json"
sentinel.write_text(
    json.dumps(
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 613,
            "by": "i613_launch",
            "ts": datetime.now(UTC).isoformat(),
            "phase": "done",
            "note": json.dumps(card),
        },
        indent=2,
    )
)
print(f"results sentinel written -> {sentinel}")
PY

echo "[phase=done] $(date -u +%FT%TZ) issue-613 flag A/B pipeline complete"
