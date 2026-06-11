#!/usr/bin/env bash
# Task #585 — pod-side launcher (plan v2 section 4.2 Step 2 + section 10).
#
# Runs on pod-585 from a DETACHED checkout of the pinned issue-534 rig
# (611e04c2f5883d2d745f77f42675b2a14d166b19). Deliberately NOT `set -e`:
# the picker exits rc=2 on a non-pass verdict and a non-pass CORRECTED
# verdict is a finding, not a crash — every other step has an explicit rc
# check that fails loud. Phases emit `[phase=...]` lines and the launcher
# terminates with `[phase=done]` AFTER the final results sentinel write
# (the poll_pipeline.py contract).
#
# Usage (plan section 4.2):
#   cd /workspace && nohup setsid /workspace/launch_issue_585.sh \
#       > /workspace/logs/issue-585-launch.log 2>&1 < /dev/null &
#
#   --dry-run : structural smoke — echoes the per-step commands, exercises the
#               rc-check flow and the sentinel writer against /tmp, mutates
#               nothing. The hf_upload heredoc EXECUTES for real in CHECK MODE
#               (stdin python: dotenv load + imports + bundle build, zero
#               network) so the stdin-dotenv crash class is covered. NOT an
#               execution smoke for the GPU phases.
set -u

PINNED_SHA="611e04c2f5883d2d745f77f42675b2a14d166b19"
REPO="/workspace/explore-persona-space"
RUNS_ROOT="/workspace/runs/issue_585"
LOGS_DIR="/workspace/logs"
TRAJ_PATH="eval_results/issue_585/c504v4_smoke_eps3_reread_seed42/trajectory.json"
CORRECTED_PATH="eval_results/issue_585/phase0_calibration_v4_corrected.json"
SLOT_STATS_PATH="eval_results/issue_585/source_slot_stats.json"
RAW_COMPLETIONS_PATH="eval_results/issue_585/raw_completions/c504v4_smoke_eps3_reread_seed42.json"
SENTINEL_PATH="${LOGS_DIR}/issue-585-results.json"
HF_PREFIX="issue585_calibration_reeval"
GPU_HOURS_BUDGETED="1.0"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    REPO="$(pwd)"
    SENTINEL_PATH="/tmp/issue-585-results.dryrun.json"
    echo "[phase=dry_run] structural dry-run: commands echoed, nothing executed."
fi

START_EPOCH=$(date +%s)
PUSH_OK=1
FINAL_COMMIT_SHA="unknown"
PLAN_DEVIATIONS=""

fail() {
    # $1 = phase slug, $2 = message. Loud log + non-zero exit (poller sees the
    # missing [phase=done] + dead PID as a crash — correct signal).
    echo "[phase=failed_$1] $2"
    exit 1
}

run_step() {
    # $1 = phase slug; rest = command. Echo-only under --dry-run.
    local phase="$1"
    shift
    echo "[phase=${phase}] $*"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] skipped execution"
        return 0
    fi
    "$@"
}

cd "$REPO" || fail "cd" "repo missing at $REPO"
mkdir -p "$LOGS_DIR" 2>/dev/null

# ── Pin guard: the rig MUST be the pinned issue-534 SHA (plan section 4.1). ──
if [ "$DRY_RUN" = "0" ]; then
    HEAD_SHA=$(git rev-parse HEAD)
    if [ "$HEAD_SHA" != "$PINNED_SHA" ]; then
        fail "pin_guard" "HEAD=$HEAD_SHA != pinned $PINNED_SHA — refusing to run off-pin."
    fi
    echo "[phase=pin_guard] HEAD is the pinned issue-534 rig ($PINNED_SHA)."
fi

# ── Step 1: fetch snapshots + inputs, build checkpoint index. ────────────────
run_step "fetch" uv run python scripts/i585_fetch_snapshots_build_index.py \
    --out-index "${RUNS_ROOT}/checkpoint_index.json" \
    --local-root "${RUNS_ROOT}/v4_anchor_download"
rc=$?
[ $rc -eq 0 ] || fail "fetch" "fetch/index script exited rc=$rc"

# ── Step 2: the corrected re-eval (the headline run; plan section 4.2). ──────
# rc must be 0; the rig's guards fail loud (LoRANotAppliedError,
# MarkerLogprobPathReadingFromBaseError, panel-disjointness, marker-token).
run_step "eval_trajectory" uv run python scripts/i504_eval_trajectory.py \
    --cell c504v4_smoke_eps3_reread \
    --seed 42 \
    --checkpoint-index "${RUNS_ROOT}/checkpoint_index.json" \
    --out-path "$TRAJ_PATH" \
    --bank-path data/issue_472/persona_bank.json \
    --r-eval-path data/issue_472/on_policy_R/R_eval_v504.json \
    --panel-json eval_results/issue_504/arm_to_n.json \
    --max-lora-rank 8 \
    --max-new-tokens 2048 \
    --max-model-len 2560 \
    --source villain \
    --sentinel-path "${LOGS_DIR}/issue-585-eval-traj.json"
rc=$?
[ $rc -eq 0 ] || fail "eval_trajectory" "i504_eval_trajectory.py exited rc=$rc"

# ── Step 3: the v4 picker over the corrected trajectory. ─────────────────────
# rc=0 (pass) or rc=2 (no_in_band_anchor) BOTH acceptable — a non-pass
# corrected verdict is part of the result. Artifact must exist either way.
run_step "picker" uv run python scripts/i504_phase_phase0_pick.py \
    --mode v4 \
    --slab-root eval_results/issue_585 \
    --v4-trajectory-path "$TRAJ_PATH" \
    --out-path "$CORRECTED_PATH" \
    --source villain \
    --fixed-lr 0.0001
rc=$?
if [ $rc -ne 0 ] && [ $rc -ne 2 ]; then
    fail "picker" "i504_phase_phase0_pick.py exited rc=$rc (only 0/2 acceptable)"
fi
if [ "$DRY_RUN" = "0" ] && [ ! -f "$CORRECTED_PATH" ]; then
    fail "picker" "picker exited rc=$rc but $CORRECTED_PATH is missing"
fi
echo "[phase=picker] done (rc=$rc; rc=2 = no_in_band_anchor verdict, acceptable)"

# ── Step 4: source-side slot-stats companion pass (plan section 4.2 Step 1.2). ─
run_step "source_slot_stats" uv run python scripts/i585_source_slot_stats.py \
    --checkpoint-index "${RUNS_ROOT}/checkpoint_index.json" \
    --out-path "$SLOT_STATS_PATH" \
    --bank-path data/issue_472/persona_bank.json \
    --source villain --seed 42 \
    --max-new-tokens 2048 --max-model-len 2560 \
    --gpu-memory-utilization 0.60 --max-lora-rank 8
rc=$?
[ $rc -eq 0 ] || fail "source_slot_stats" "i585_source_slot_stats.py exited rc=$rc"

# ── Step 5: HF data-repo uploads (fail-loud; verified via list_repo_files). ──
# Raw completions MUST land on HF before pod termination (Upload Policy). The
# bundle layout is non-canonical for upload_raw_completions_to_data_repo's
# rglob("raw_completions.json"), so this uploads the three files explicitly to
# the plan-pinned paths and verifies each landed.
echo "[phase=hf_upload] uploading trajectory + slot stats + raw completions"
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] executing upload step in CHECK MODE (real stdin python: dotenv + imports + bundle build; no network)"
fi
I585_UPLOAD_CHECK="$DRY_RUN" uv run python - "$TRAJ_PATH" "$SLOT_STATS_PATH" "$RAW_COMPLETIONS_PATH" "$HF_PREFIX" <<'PY'
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Bare load_dotenv() deterministically AssertionErrors under `python - <<PY`
# (stdin execution): python-dotenv 1.2.2's find_dotenv() frame-walks for a
# caller file, and "<stdin>" never exists on disk, so the walk exhausts frames
# (round-1 review Critical, concern launcher-hf-upload-dotenv-stdin-crash).
# usecwd anchors the search at the launcher's cwd ($REPO), where bootstrap
# places .env.
load_dotenv(find_dotenv(usecwd=True))
from huggingface_hub import HfApi

check_mode = os.environ.get("I585_UPLOAD_CHECK") == "1"
traj_path, slot_stats_path, raw_path, hf_prefix = sys.argv[1:5]
repo_id = "superkaiba1/explore-persona-space-data"

slot_file = Path(slot_stats_path)
if check_mode and not slot_file.exists():
    # Dry-run before any pod phase produced slot stats: the crash class under
    # test (stdin dotenv load + huggingface_hub import) has already executed.
    print("[phase=hf_upload] check-mode OK (stdin dotenv + imports; no slot stats yet)")
    sys.exit(0)

# Build the raw-completions bundle (source R text per fraction x question)
# from source_slot_stats.json — plan section 4.2 Step 3.
slot = json.loads(slot_file.read_text())
bundle = {
    "schema_version": "i585_raw_completions_v1",
    "task": 585,
    "cell": "c504v4_smoke_eps3_reread",
    "seed": 42,
    "source": slot["source"],
    "note": (
        "On-policy greedy source R per (fraction, question) from the i585 "
        "source slot-stats companion pass (engine settings byte-matched to "
        "the main run; distinct lora_int_id 1..6)."
    ),
    "git_commit": slot.get("git_commit"),
    "timestamp_utc": datetime.now(UTC).isoformat(),
    "completions": {
        f"{fr['frac']:.2f}": {q: rec["r_text"] for q, rec in fr["per_question"].items()}
        for fr in slot["fractions"]
    },
}
if check_mode:
    # Same code path, but write the bundle to a tempfile (no tree mutation).
    raw_out = Path(tempfile.mkstemp(prefix="i585_raw_check_", suffix=".json")[1])
else:
    raw_out = Path(raw_path)
raw_out.parent.mkdir(parents=True, exist_ok=True)
raw_out.write_text(json.dumps(bundle, indent=2))

if check_mode:
    n_q = sum(len(v) for v in bundle["completions"].values())
    print(f"[phase=hf_upload] check-mode OK (bundle built: {n_q} completions -> {raw_out}; no network)")
    sys.exit(0)

api = HfApi(token=os.environ.get("HF_TOKEN"))
uploads = {
    traj_path: f"{hf_prefix}/c504v4_smoke_eps3_reread_seed42/trajectory.json",
    slot_stats_path: f"{hf_prefix}/source_slot_stats.json",
    raw_path: f"{hf_prefix}/raw_completions/c504v4_smoke_eps3_reread_seed42.json",
}
for local, in_repo in uploads.items():
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[phase=hf_upload] uploaded {local} -> {repo_id}/{in_repo}")

# Verify via the Hub API (NOT the hf CLI — see upload-policy rule).
files = set(api.list_repo_files(repo_id, repo_type="dataset"))
missing = [p for p in uploads.values() if p not in files]
if missing:
    raise RuntimeError(f"HF upload verification FAILED — missing on {repo_id}: {missing}")
print("[phase=hf_upload] all 3 artifacts verified on the data repo")
PY
rc=$?
[ $rc -eq 0 ] || fail "hf_upload" "HF upload step exited rc=$rc"

# ── Step 6: results commit (detached-HEAD contract, plan section 4.2 Step 3). ─
echo "[phase=results_commit] committing eval_results/issue_585 onto issue-585"
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] skipped git commit/push"
else
    # Fresh ephemeral pods may lack a git identity — the commit would crash.
    if [ -z "$(git config user.email)" ]; then
        git config user.email "pod-585@explore-persona-space.local"
        git config user.name "pod-585 launcher"
    fi
    git fetch origin issue-585
    rc=$?
    [ $rc -eq 0 ] || fail "results_commit" "git fetch origin issue-585 exited rc=$rc"
    # The glue scripts were extracted UNTRACKED into the detached tree via
    # `git show origin/issue-585:... > scripts/...`; they are TRACKED on
    # issue-585, so checkout -B would refuse to overwrite them. They are
    # byte-copies of the branch's own files — remove, the checkout restores.
    rm -f scripts/i585_fetch_snapshots_build_index.py scripts/i585_source_slot_stats.py
    git checkout -B issue-585 origin/issue-585
    rc=$?
    [ $rc -eq 0 ] || fail "results_commit" "git checkout -B issue-585 exited rc=$rc"
    # Raw completions are HF-data-repo-only per the plan's section 10 git list
    # (round-1 review minor): exclude the bundle from the results commit.
    git add -- eval_results/issue_585/ ':(exclude)eval_results/issue_585/raw_completions'
    git commit -m "task #585: corrected per-fraction calibration eval (trajectory + corrected table + source slot stats)"
    rc=$?
    [ $rc -eq 0 ] || fail "results_commit" "git commit exited rc=$rc"
    FINAL_COMMIT_SHA=$(git rev-parse HEAD)
    git push origin issue-585
    if [ $? -ne 0 ]; then
        echo "[phase=results_commit] push rejected; retrying once after pull --rebase"
        git pull --rebase origin issue-585 && git push origin issue-585
        if [ $? -ne 0 ]; then
            # Plan section 4.2 Step 3 fallback: VM-side download + commit. Do
            # NOT hand-resolve on the pod; results are already on HF.
            PUSH_OK=0
            PLAN_DEVIATIONS="pod-side push rejected twice; VM-side fallback required (artifacts already on HF data repo)"
            echo "[phase=results_commit] WARNING: push failed twice — VM fallback required"
        else
            FINAL_COMMIT_SHA=$(git rev-parse HEAD)
        fi
    fi
fi

# ── Step 7: authoritative results sentinel (poll_pipeline.py contract). ──────
echo "[phase=results_sentinel] writing $SENTINEL_PATH"
ELAPSED_S=$(( $(date +%s) - START_EPOCH ))
DRY_RUN="$DRY_RUN" ELAPSED_S="$ELAPSED_S" PUSH_OK="$PUSH_OK" \
FINAL_COMMIT_SHA="$FINAL_COMMIT_SHA" PLAN_DEVIATIONS="$PLAN_DEVIATIONS" \
SENTINEL_PATH="$SENTINEL_PATH" CORRECTED_PATH="$CORRECTED_PATH" \
TRAJ_PATH="$TRAJ_PATH" SLOT_STATS_PATH="$SLOT_STATS_PATH" \
RAW_COMPLETIONS_PATH="$RAW_COMPLETIONS_PATH" HF_PREFIX="$HF_PREFIX" \
GPU_HOURS_BUDGETED="$GPU_HOURS_BUDGETED" PINNED_SHA="$PINNED_SHA" \
uv run python - <<'PY'
import json
import os
from datetime import UTC, datetime
from pathlib import Path

dry_run = os.environ["DRY_RUN"] == "1"
corrected_path = os.environ["CORRECTED_PATH"]
slot_stats_path = os.environ["SLOT_STATS_PATH"]

eval_numbers: dict = {}
if not dry_run:
    corrected = json.loads(Path(corrected_path).read_text())
    eval_numbers["corrected_smoke_table"] = [
        {
            "ckpt_frac": row["ckpt_frac"],
            "source_dg": row["source_dg"],
            "source_emission": row["source_emission"],
            "bystander_resolution": row["bystander_resolution"],
            "in_band": row["in_band"],
        }
        for row in corrected["smoke_table"]
    ]
    eval_numbers["corrected_verdict"] = corrected["verdict"]
    eval_numbers["chosen_checkpoint_fraction"] = corrected["chosen_checkpoint_fraction"]
    slot = json.loads(Path(slot_stats_path).read_text())
    eval_numbers["glue_source_delta_g_mean_by_frac"] = {
        f"{fr['frac']:.2f}": fr["delta_g_mean"] for fr in slot["fractions"]
    }
    dg008 = next(
        row["source_dg"]
        for row in corrected["smoke_table"]
        if abs(row["ckpt_frac"] - 0.08) < 1e-6
    )
    eval_numbers["h2_frac008_control"] = {
        "corrected_dg": dg008,
        "stale_dg": 5.471729629712354,
        "abs_diff": abs(dg008 - 5.471729629712354),
        "tolerance_nats": 2.0,
        "pass": abs(dg008 - 5.471729629712354) <= 2.0,
    }
else:
    eval_numbers["dry_run"] = True

gpu_hours_used = round(int(os.environ["ELAPSED_S"]) / 3600.0, 3)
note = {
    "eval_numbers": eval_numbers,
    "eval_paths": [
        os.environ["TRAJ_PATH"],
        corrected_path,
        slot_stats_path,
        os.environ["RAW_COMPLETIONS_PATH"],
    ],
    "reproducibility_card": {
        "task": 585,
        "pinned_rig_sha": os.environ["PINNED_SHA"],
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "adapters": "adapters/issue_504_v4/c504v4_smoke_eps3_seed42/ckpt_frac{0.08..1.00}",
        "cell": "c504v4_smoke_eps3_reread",
        "seed": 42,
        "marker_token_id": 83399,
        "kl": True,
        "eval_invocation": (
            "i504_eval_trajectory.py --cell c504v4_smoke_eps3_reread --seed 42 "
            "--max-lora-rank 8 --max-new-tokens 2048 --max-model-len 2560 "
            "--source villain (KL on, gpu_memory_utilization 0.60 default)"
        ),
    },
    "wandb_url": None,
    "hf_hub_url": (
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/"
        + os.environ["HF_PREFIX"]
    ),
    "worktree_path": ".claude/worktrees/issue-585",
    "final_commit_sha": os.environ["FINAL_COMMIT_SHA"],
    "results_commit_pushed": os.environ["PUSH_OK"] == "1",
    "gpu_hours_used": gpu_hours_used,
    "gpu_hours_budgeted": float(os.environ["GPU_HOURS_BUDGETED"]),
    "plan_deviations": (
        [os.environ["PLAN_DEVIATIONS"]] if os.environ["PLAN_DEVIATIONS"] else []
    ),
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 585,
    "by": "launch_issue_585",
    "gate": False,
    "blocks_pipeline": False,
    "ts": datetime.now(UTC).isoformat(),
    "note": json.dumps(note),
}
out = Path(os.environ["SENTINEL_PATH"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(sentinel, indent=2))
print(f"[phase=results_sentinel] wrote {out}")
PY
rc=$?
[ $rc -eq 0 ] || fail "results_sentinel" "sentinel writer exited rc=$rc"

echo "[phase=done]"
exit 0
