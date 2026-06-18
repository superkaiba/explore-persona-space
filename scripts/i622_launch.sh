#!/usr/bin/env bash
# Task #622 — dose-to-failure launch driver (parents #472 → #601; sibling #613).
#
# Sequential gated pipeline (plan #622 §4.3), the i601_launch.sh /
# i613_launch.sh self-daemonizing supervisor pattern verbatim (setsid --fork,
# pid file, heartbeat, phase file, combined EXIT trap, skip-cheap resume):
#
#   p0_registry_assert  the 6 dose_break cells resolve with the registered
#                       T arithmetic / cadence / exclusions (CPU; fail loud).
#   p1_fetch            pinned #472 parent inputs @ DATA_REV -> data/issue_622/
#                       + the committed #601 bystander panel ->
#                       $SLAB_ROOT/phase0/ (the #613 precedent).
#   p2_phase0_gate      writes $SLAB_ROOT/phase0/phase0_gate.json pass=true
#                       (gate_schema i622_p0_asserts_v1 — no adapter reuse in
#                       this design, so the #601 Phase-0 fitness reads have no
#                       analogue; the dispatcher's gate then enforces "p0 ran").
#   p3_smoke            dispatcher --smoke --smoke-cell dose_200p3200n
#                       --smoke-seed 42: ONE FULL sweep unit (T=213, complete
#                       schedule — smoke IS the sweep with one cell). Skipped
#                       when a prior sentinel records smoke_gate_pass==true.
#   p4_smoke_gate       sentinel records smoke_gate_pass==true (accepts the
#                       poller's .processed rename).
#   p5_sweep            dispatcher --cells <all 6> --seeds 42,137 --resume,
#                       4-way CUDA_VISIBLE_DEVICES pool (the smoke unit is
#                       sweep unit 1 of 12 and resumes as a skip).
#   p6_final_sentinel   $LOG_DIR/issue-622-results.json present (epm:results
#                       reproducibility_card; accepts .processed).
#
# Launch (GCP instance eps-issue-622 / pod, repo at issue-622 HEAD, from the
# repo root — GCP lane: --workload-cmd 'cd "$WORKLOAD_ROOT" && bash scripts/i622_launch.sh'):
#   bash scripts/i622_launch.sh
#
# Logging contract (poll_pipeline.py): each sub-step's verbose output goes to
# its OWN sub-log; the SUPERVISED process's stdout is the main pipeline log
# ($LOG_DIR/issue-622.log) with one [phase=...] line per step, a 120 s
# heartbeat, `[phase=abort] rc=<rc>` on any non-zero exit, and the SINGLE
# terminal [phase=done] line (sub-steps never leak a done token — incident
# #545). The pid file satisfies the GCP startup script's /workspace/logs/*.pid
# wait (fact-checker finding, plan §4.1 item 6).
#
# NOTE on preflight: this driver does NOT re-run orchestrate.preflight — the
# experimenter runs it pre-launch. If you add it here, parse `preflight
# --json` (whole stdout) and tolerate ONLY the documented feature-branch
# false positive; a bare `preflight || exit` under set -e kills issue-branch
# launches on pre-#554 checkouts (incident #552).
set -euo pipefail

LOG_DIR="${LOG_DIR:-/workspace/logs}"
SLAB_ROOT="${SLAB_ROOT:-eval_results/issue_622}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/runs/issue_622}"
DATA_DIR="${DATA_DIR:-data/issue_622}"
N_GPUS="${N_GPUS:-4}"
EXTRA_SWEEP_ARGS="${EXTRA_SWEEP_ARGS:-}"
SMOKE_CELL="dose_200p3200n"
SMOKE_SEED=42
ALL_CELLS="dose_200p3200n,dose_200p6400n,dose_200p12800n,posonly_200p_T208,posonly_200p_T416,posonly_200p_T819"
HF_PREFIX="adapters/issue_622"
RUN_NAME_PREFIX="issue622"
HF_DATA_PREFIX="issue622_dose_break"
SENTINEL_TASK_ID=622
# Pinned #472 parent inputs (plan §10; Hub-verified 2026-06-12).
DATA_REV="66d7db7a542e19275f8c1d8e32948396d050faa9"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/issue-622.log"     # poller-pinned main log
PID_FILE="$LOG_DIR/issue-622.pid"
PHASE_FILE="$LOG_DIR/issue-622.phase"

# ── Self-daemonization + relaunch guard (i601_launch.sh pattern verbatim) ────
if [ -z "${I622_SUPERVISED:-}" ]; then
    if ! setsid --version >/dev/null 2>&1; then
        echo "[launcher] FATAL: 'setsid --version' failed — util-linux setsid (with --fork) required"
        exit 4
    fi
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
        echo "[launcher] already running pid=$(cat "$PID_FILE") — refusing relaunch (exit 3)"
        exit 3
    fi
    export I622_SUPERVISED=1
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

# MooseFS-quota parity (checkpoints upload in ONE bulk commit at unit end) +
# the issue<N> WandB project convention (mechanical reproducibility_card).
# NO global CUDA_VISIBLE_DEVICES export here: the dispatcher pins CVD per cell
# subprocess in the LAUNCHER env (gotcha #545; _schedule_cell_pool).
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD="${EPM_SKIP_INLINE_CHECKPOINT_UPLOAD:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-issue622}"

HB_INTERVAL="${I622_HB_INTERVAL:-120}"
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
echo "[phase=p0_registry_assert] $(date -u +%FT%TZ) asserting #622 cell registration + branch"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ "$BRANCH" = "main" ]; then
    echo "[phase=p0_registry_assert] FATAL: checkout is on 'main' — the #622 rig lives on issue-622"
    exit 5
fi
echo "[phase=p0_registry_assert] branch=$BRANCH sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
uv run python - <<'PY'
from explore_persona_space.experiments.neg_setpoint_601 import (
    EXPECTED_ANCHOR_PANEL,
    SOURCE_PERSONA,
    cell_by_slug,
    cells_for_request,
)

# (slug, T, total_rows, epochs, neg_per_persona, stride)
EXPECT = (
    ("dose_200p3200n", 213, 3400, 1, 800, 5),
    ("dose_200p6400n", 413, 6600, 1, 1600, 5),
    ("dose_200p12800n", 813, 13000, 1, 3200, 10),
    ("posonly_200p_T208", 208, 200, 16, 0, 5),
    ("posonly_200p_T416", 416, 200, 32, 0, 5),
    ("posonly_200p_T819", 819, 200, 63, 0, 10),
)
all_slugs = {c.slug for c in cells_for_request("all")}
p4b_slugs = {c.slug for c in cells_for_request("phase4b")}
for slug, t, rows, epochs, neg, stride in EXPECT:
    s = cell_by_slug(slug)  # KeyError on a main checkout / unregistered slug
    assert s.expected_steps == t, f"{slug}: T={s.expected_steps} != {t}"
    assert s.total_rows == rows, f"{slug}: rows={s.total_rows} != {rows}"
    assert s.epochs == epochs and s.neg_ex_per_persona == neg
    assert s.pos_ex == 200 and s.lr == 1e-5 and s.lora_targets is None
    assert s.conditional is True, f"{slug} must be explicit-slug-only"
    assert s.phase == "dose_break", f"{slug} phase={s.phase}"
    assert s.band_stop is True and s.band_log_only is True, (
        f"{slug}: D1 requires log-only band (a firing stop would truncate T)"
    )
    assert s.suppress_negatives is False, f"{slug}: flag-off parity (#472/#601)"
    assert s.onpolicy == "anchors" and len(s.onpolicy_anchor_steps) == 5
    assert s.onpolicy_anchor_steps[-1] == t, f"{slug} terminal anchor != T"
    assert (s.probe_dense_until, s.probe_every_steps) == (50, stride)
    assert s.capability_trajectory is True
    # DV6 round-2 (dv6-trained-negatives-onpolicy-missing): every #622 cell
    # must thread the trained anchor negatives into both eval surfaces.
    assert s.eval_include_trained_negatives is True, f"{slug}: DV6 trained-negative read unwired"
    assert s.seeds == (42, 137)
    assert max(s.dense_steps) == t and len(s.dense_steps) == len(set(s.dense_steps))
    assert slug not in all_slugs, f"{slug} leaked into --cells all"
    assert slug not in p4b_slugs, f"{slug} leaked into phase4b"
    assert [c.slug for c in cells_for_request(slug)] == [slug]
# T-match arithmetic (plan §4.2: every pair at or under 2.3%).
for dose, twin in (
    ("dose_200p3200n", "posonly_200p_T208"),
    ("dose_200p6400n", "posonly_200p_T416"),
    ("dose_200p12800n", "posonly_200p_T819"),
):
    td, tw = cell_by_slug(dose).expected_steps, cell_by_slug(twin).expected_steps
    mismatch = abs(td - tw) / td
    assert mismatch <= 0.024, f"{dose} vs {twin}: |dT|/T = {mismatch:.3f} > 2.4%"
# Panel disjointness at the registry level (the worker re-asserts against the
# REALIZED selector output per unit).
assert SOURCE_PERSONA not in EXPECTED_ANCHOR_PANEL
print("registry assert PASS: 6 dose_break cells, T-matched pairs, exclusions hold")
PY

set_phase p1_fetch
echo "[phase=p1_fetch] $(date -u +%FT%TZ) fetching pinned parent inputs @ $DATA_REV (idempotent)"
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
    # PARENT_DATA_FILES maps to data/issue_601/...; #622 owns data/issue_622/
    # (the #613 precedent) — same basenames, issue-local root.
    local = data_dir / Path(parent_local_rel).relative_to("data/issue_601")
    # Always re-fetch at revision=rev, even when `local` already exists: a
    # short-circuit on existence bypasses the DATA_REV pin on resumed/reused
    # pods, silently violating the plan's pinned-input discipline (#622 round-2
    # concern 'pinned-input-fetch-not-enforced'; the #600 stale-mirror class).
    # hf_hub_download is HTTP-cached against the immutable revision SHA, so the
    # repeat call is a cheap HEAD on warm caches and the snapshot resolves to
    # the same bytes by construction.
    local.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=repo_path,
        revision=rev,  # immutable-revision pin — closes the #600 stale-mirror class
        token=os.environ.get("HF_TOKEN"),
    )
    shutil.copyfile(got, local)
    print(f"[fetch] {repo_path}@{rev[:9]} -> {local}")
PY
mkdir -p "$SLAB_ROOT/phase0"
if [ ! -f "$SLAB_ROOT/phase0/bystander_panel.json" ]; then
    test -f eval_results/issue_601/phase0/bystander_panel.json || {
        echo "[phase=p1_fetch] FATAL: committed #601 bystander panel missing from the checkout"
        exit 6
    }
    cp eval_results/issue_601/phase0/bystander_panel.json "$SLAB_ROOT/phase0/bystander_panel.json"
fi
echo "[phase=p1_fetch] bystander panel staged at $SLAB_ROOT/phase0/bystander_panel.json"

set_phase p2_phase0_gate
echo "[phase=p2_phase0_gate] $(date -u +%FT%TZ) writing phase0_gate.json (i622_p0_asserts_v1)"
uv run python - "$SLAB_ROOT/phase0/phase0_gate.json" "$DATA_DIR" "$DATA_REV" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

gate_path, data_dir, rev = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
required = (
    data_dir / "persona_bank.json",
    data_dir / "centroids_L10.pt",
    data_dir / "on_policy_R" / "R_train.json",
    data_dir / "on_policy_R" / "R_eval.json",
    gate_path.parent / "bystander_panel.json",
)
missing = [str(p) for p in required if not p.exists()]
assert not missing, f"p0 inputs missing after fetch: {missing}"
gate_path.write_text(
    json.dumps(
        {
            "pass": True,
            "gate_schema": "i622_p0_asserts_v1",
            "note": (
                "#622 has no adapter reuse, so the #601 Phase-0 fitness reads have no "
                "analogue; this gate records that the p0 registry asserts + pinned-input "
                "fetch + bystander-panel staging all PASSed (plan #622 §4.3 p0)."
            ),
            "data_rev": rev,
            "inputs_present": [str(p) for p in required],
            "ts": datetime.now(UTC).isoformat(),
        },
        indent=2,
    )
)
print(f"phase0 gate written: {gate_path}")
PY

set_phase p3_smoke
SMOKE_SKIP=$(uv run python - "$LOG_DIR/issue-622-smoke-results.json" <<'PY'
import json, pathlib, sys
bare = pathlib.Path(sys.argv[1])
candidate = bare if bare.exists() else bare.with_suffix(".json.processed")
ok = False
if candidate.exists():
    try:
        payload = json.loads(candidate.read_text())
        note = json.loads(payload.get("note") or payload.get("payload") or "{}")
        ok = note.get("smoke_gate_pass") is True
    except (OSError, json.JSONDecodeError) as exc:
        print(f"smoke sentinel unreadable ({exc}); re-running the smoke", file=sys.stderr)
print("skip" if ok else "run")
PY
)
if [ "$SMOKE_SKIP" = "skip" ]; then
    echo "[phase=p3_smoke] sentinel valid; skip"
else
    # --skip-fetch (round-2 opportunistic a): p1 already fetched the PINNED
    # inputs @ $DATA_REV; the dispatcher's own fetch_parent_data() is UNPINNED
    # and must never overwrite/extend the pinned set.
    echo "[phase=p3_smoke] $(date -u +%FT%TZ) launching smoke (ONE FULL sweep unit $SMOKE_CELL seed $SMOKE_SEED; sub-log: $LOG_DIR/issue-622-smoke.log)"
    uv run python scripts/dispatch_neg_setpoint_601.py \
        --cells "$SMOKE_CELL" --seeds "$SMOKE_SEED" --smoke \
        --smoke-cell "$SMOKE_CELL" --smoke-seed "$SMOKE_SEED" \
        --n-gpus "$N_GPUS" \
        --slab-root "$SLAB_ROOT" --runs-root "$RUNS_ROOT" \
        --log-dir "$LOG_DIR" --data-dir "$DATA_DIR" \
        --hf-prefix "$HF_PREFIX" --run-name-prefix "$RUN_NAME_PREFIX" \
        --sentinel-task-id "$SENTINEL_TASK_ID" --hf-data-prefix "$HF_DATA_PREFIX" \
        --skip-fetch \
        > "$LOG_DIR/issue-622-smoke.log" 2>&1
fi

set_phase p4_smoke_gate
echo "[phase=p4_smoke_gate] $(date -u +%FT%TZ) checking smoke sentinel"
# Accept the poller's .processed rename (same race as #601 round-3 blocker
# smoke-sentinel-processed-race).
uv run python - "$LOG_DIR/issue-622-smoke-results.json" <<'PY'
import json, pathlib, sys
bare = pathlib.Path(sys.argv[1])
candidate = bare if bare.exists() else bare.with_suffix(".json.processed")
if not candidate.exists():
    raise SystemExit(f"smoke gate FAILED: sentinel missing at {bare} (also checked {candidate})")
payload = json.loads(candidate.read_text())
note = json.loads(payload.get("note") or payload.get("payload") or "{}")
assert note.get("smoke_gate_pass") is True, f"smoke gate FAILED ({candidate}): {note}"
print(f"smoke gate PASS (sentinel: {candidate})")
PY
echo "[phase=p4_smoke_gate] disk headroom after smoke:"
df -h /workspace 2>/dev/null || df -h .

set_phase p5_sweep
echo "[phase=p5_sweep] $(date -u +%FT%TZ) launching full sweep (12 units, smoke unit resumes as skip; sub-log: $LOG_DIR/issue-622-sweep.log)"
# shellcheck disable=SC2086  # EXTRA_SWEEP_ARGS is a deliberate word-split passthrough
uv run python scripts/dispatch_neg_setpoint_601.py \
    --cells "$ALL_CELLS" --seeds 42,137 \
    --n-gpus "$N_GPUS" --max-parallel "$N_GPUS" --resume \
    --slab-root "$SLAB_ROOT" --runs-root "$RUNS_ROOT" \
    --log-dir "$LOG_DIR" --data-dir "$DATA_DIR" \
    --hf-prefix "$HF_PREFIX" --run-name-prefix "$RUN_NAME_PREFIX" \
    --sentinel-task-id "$SENTINEL_TASK_ID" --hf-data-prefix "$HF_DATA_PREFIX" \
    --skip-fetch \
    $EXTRA_SWEEP_ARGS \
    > "$LOG_DIR/issue-622-sweep.log" 2>&1

set_phase p6_final_sentinel
echo "[phase=p6_final_sentinel] $(date -u +%FT%TZ) verifying final results sentinel"
test -f "$LOG_DIR/issue-622-results.json" \
    || test -f "$LOG_DIR/issue-622-results.json.processed" \
    || { echo "final sentinel missing"; exit 1; }

echo "[phase=done] $(date -u +%FT%TZ) issue-622 pipeline complete"
