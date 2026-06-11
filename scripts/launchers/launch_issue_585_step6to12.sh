#!/usr/bin/env bash
# Task #585 follow-up `step6to12-transition-sweep` — pod-side launcher
# (plan v3 section 4.2 Step 2 + section 10).
#
# Two-checkout choreography on pod-585:
#   Phase T (retrain)  : DETACHED #504 launch SHA  affdd82cb... (training basis)
#   Phase E (eval)     : DETACHED pinned issue-534 rig 611e04c2f... (parent rig)
# Untracked glue + outputs survive both checkouts (plan section 4.0 / A14).
#
# Deliberately NOT `set -e`: every step has an explicit rc check that fails
# loud. Phases emit `[phase=...]` lines and the launcher terminates with
# `[phase=done]` AFTER the final results sentinel write (poll_pipeline.py
# contract).
#
# Usage (plan section 4.2 Step 2):
#   cd /workspace && nohup setsid /workspace/launch_issue_585_step6to12.sh \
#       > /workspace/logs/issue-585-step6to12-launch.log 2>&1 < /dev/null &
#
#   --dry-run : structural smoke — echoes the per-step commands, exercises the
#               rc-check flow and the sentinel writer against /tmp, mutates
#               nothing. The hf_upload heredoc EXECUTES for real in CHECK MODE
#               (stdin python: dotenv load + imports + bundle build when the
#               inputs exist, zero network) so the stdin-dotenv crash class is
#               covered. NOT an execution smoke for the GPU phases.
set -u

TRAIN_SHA="affdd82cb0bb31257b5668b327c6af5716212b6c"
RIG_SHA="611e04c2f5883d2d745f77f42675b2a14d166b19"
REPO="/workspace/explore-persona-space"
RUNS_ROOT="/workspace/runs/issue_585_step6to12"
LOGS_DIR="/workspace/logs"
SLAB="eval_results/issue_585/step6to12-transition-sweep"
CELL="c504v4_smoke_eps3_step6to12"
TRAJ_PATH="${SLAB}/c504v4_smoke_eps3_step6to12_seed42/trajectory.json"
SLOT_STATS_PATH="${SLAB}/source_slot_stats.json"
INDEX_PATH="${SLAB}/checkpoint_index.json"
PROVENANCE_PATH="${SLAB}/index_provenance.json"
MANIFEST_PATH="${SLAB}/retrain_manifest.json"
RAW_COMPLETIONS_PATH="${SLAB}/raw_completions/c504v4_smoke_eps3_step6to12_seed42.json"
TRAIN_POOL_PATH="${RUNS_ROOT}/c504v3_smoke_eps3_seed42/train_pool.jsonl"
ARM_TO_N="/workspace/arm_to_n_pinned.json"
SENTINEL_PATH="${LOGS_DIR}/issue-585-step6to12-results.json"
HF_PREFIX="issue585_calibration_reeval/step6to12"
GPU_HOURS_BUDGETED="2.0"
PARENT_CORRECTED="eval_results/issue_585/phase0_calibration_v4_corrected.json"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    REPO="$(pwd)"
    SENTINEL_PATH="/tmp/issue-585-step6to12-results.dryrun.json"
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

# ── Pin guard T: the tree MUST be the #504 launch SHA (plan section 4.0). ────
if [ "$DRY_RUN" = "0" ]; then
    HEAD_SHA=$(git rev-parse HEAD)
    if [ "$HEAD_SHA" != "$TRAIN_SHA" ]; then
        fail "pin_guard_train" "HEAD=$HEAD_SHA != pinned training SHA $TRAIN_SHA — refusing."
    fi
    echo "[phase=pin_guard_train] HEAD is the pinned #504 launch SHA ($TRAIN_SHA)."
fi

# ── Step 0: materialize the pinned arm_to_n copy (idempotent; A4). ───────────
if [ "$DRY_RUN" = "0" ] && [ ! -f "$ARM_TO_N" ]; then
    git show "${RIG_SHA}:eval_results/issue_504/arm_to_n.json" > "$ARM_TO_N"
    rc=$?
    [ $rc -eq 0 ] || fail "arm_to_n" "git show ${RIG_SHA}:.../arm_to_n.json exited rc=$rc"
    echo "[phase=arm_to_n] materialized pinned copy -> $ARM_TO_N"
fi

# ── Phase T: per-step retrain at the launch SHA (plan section 4.2). ──────────
# rc must be 0: marker assert, band-stop-absence assert, composition asserts,
# step asserts, inline Hub persist verify are all fail-loud.
run_step "retrain" uv run python scripts/i585_retrain_per_step.py \
    --arm-to-n-json "$ARM_TO_N" \
    --runs-root "$RUNS_ROOT" \
    --manifest-out "$MANIFEST_PATH" \
    --sentinel-path "${LOGS_DIR}/issue-585-step6to12-retrain.json"
rc=$?
[ $rc -eq 0 ] || fail "retrain" "i585_retrain_per_step.py exited rc=$rc"

# ── Phase E: switch to the pinned issue-534 rig (untracked outputs survive). ─
if [ "$DRY_RUN" = "1" ]; then
    echo "[phase=rig_checkout] (dry-run) would: git checkout $RIG_SHA"
else
    git checkout "$RIG_SHA"
    rc=$?
    [ $rc -eq 0 ] || fail "rig_checkout" "git checkout $RIG_SHA exited rc=$rc"
    HEAD_SHA=$(git rev-parse HEAD)
    if [ "$HEAD_SHA" != "$RIG_SHA" ]; then
        fail "pin_guard_rig" "post-checkout HEAD=$HEAD_SHA != pinned rig SHA $RIG_SHA"
    fi
    echo "[phase=pin_guard_rig] HEAD is the pinned issue-534 rig ($RIG_SHA)."
fi

# ── Step 1: Hub endpoints + inputs + merged 9-entry index. ───────────────────
run_step "fetch" uv run python scripts/i585_fetch_snapshots_build_index.py \
    --out-index "$INDEX_PATH" \
    --local-root "${RUNS_ROOT}/hub_endpoints" \
    --fractions 0.08,0.16 \
    --merge-retrain-manifest "$MANIFEST_PATH"
rc=$?
[ $rc -eq 0 ] || fail "fetch" "fetch/index script exited rc=$rc"

# ── Step 2: the 9-checkpoint trajectory eval (the headline run). ─────────────
# rc must be 0; the rig's guards fail loud (LoRANotAppliedError, byte-identical
# guard, panel-disjointness, marker-token). KL on (no --no-kl) — parent parity.
run_step "eval_trajectory" uv run python scripts/i504_eval_trajectory.py \
    --cell "$CELL" \
    --seed 42 \
    --checkpoint-index "$INDEX_PATH" \
    --out-path "$TRAJ_PATH" \
    --bank-path data/issue_472/persona_bank.json \
    --r-eval-path data/issue_472/on_policy_R/R_eval_v504.json \
    --panel-json "$ARM_TO_N" \
    --max-lora-rank 8 \
    --max-new-tokens 2048 \
    --max-model-len 2560 \
    --source villain \
    --sentinel-path "${LOGS_DIR}/issue-585-step6to12-eval-traj.json"
rc=$?
[ $rc -eq 0 ] || fail "eval_trajectory" "i504_eval_trajectory.py exited rc=$rc"

# ── Step 3: source-side slot-stats companion pass (four-float contract). ─────
run_step "source_slot_stats" uv run python scripts/i585_source_slot_stats.py \
    --checkpoint-index "$INDEX_PATH" \
    --out-path "$SLOT_STATS_PATH" \
    --bank-path data/issue_472/persona_bank.json \
    --source villain --seed 42 \
    --max-new-tokens 2048 --max-model-len 2560 \
    --gpu-memory-utilization 0.60 --max-lora-rank 8
rc=$?
[ $rc -eq 0 ] || fail "source_slot_stats" "i585_source_slot_stats.py exited rc=$rc"

# ── Step 4: HF data-repo uploads (fail-loud; verified via list_repo_files). ──
# Raw completions MUST land on HF before pod termination (Upload Policy). The
# bundle layout is non-canonical for upload_raw_completions_to_data_repo's
# rglob("raw_completions.json"), so this uploads the files explicitly to the
# plan-pinned paths (plan section 10 destinations) and verifies each landed.
echo "[phase=hf_upload] uploading trajectory + index + manifest + train pool + raw completions"
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] executing upload step in CHECK MODE (real stdin python: dotenv + imports + bundle build when inputs exist; no network)"
fi
I585_UPLOAD_CHECK="$DRY_RUN" uv run python - \
    "$TRAJ_PATH" "$SLOT_STATS_PATH" "$INDEX_PATH" "$PROVENANCE_PATH" \
    "$MANIFEST_PATH" "$TRAIN_POOL_PATH" "$RAW_COMPLETIONS_PATH" "$HF_PREFIX" <<'PY'
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
# (parent round-1 review Critical, concern launcher-hf-upload-dotenv-stdin-crash).
# usecwd anchors the search at the launcher's cwd ($REPO), where bootstrap
# places .env.
load_dotenv(find_dotenv(usecwd=True))
from huggingface_hub import HfApi

check_mode = os.environ.get("I585_UPLOAD_CHECK") == "1"
(
    traj_path,
    slot_stats_path,
    index_path,
    provenance_path,
    manifest_path,
    train_pool_path,
    raw_path,
    hf_prefix,
) = sys.argv[1:9]
repo_id = "superkaiba1/explore-persona-space-data"

slot_file = Path(slot_stats_path)
prov_file = Path(provenance_path)
if check_mode and not (slot_file.exists() and prov_file.exists()):
    # Dry-run before any pod phase produced inputs: the crash class under test
    # (stdin dotenv load + huggingface_hub import) has already executed.
    print("[phase=hf_upload] check-mode OK (stdin dotenv + imports; no slot stats/provenance yet)")
    sys.exit(0)

# Build the raw-completions bundle (source R text per checkpoint x question)
# from source_slot_stats.json + index_provenance.json — plan section 4.2.
# Keys are the merged-index keys (e.g. "0.0733"); step + provenance carried
# per entry so the bundle is self-describing.
slot = json.loads(slot_file.read_text())
prov = json.loads(prov_file.read_text())
completions = {}
for fr in slot["fractions"]:
    # Match the slot-stats float frac back to its merged-index key.
    matches = [k for k in prov if float(k) == float(fr["frac"])]
    if len(matches) != 1:
        raise RuntimeError(
            f"raw-completions bundle: slot-stats frac {fr['frac']!r} matched "
            f"{matches} in index_provenance.json (need exactly 1)."
        )
    key = matches[0]
    completions[key] = {
        "step": prov[key]["step"],
        "provenance": prov[key]["provenance"],
        "responses": {q: rec["r_text"] for q, rec in fr["per_question"].items()},
    }
bundle = {
    "schema_version": "i585_step6to12_raw_completions_v1",
    "task": 585,
    "followup_label": "step6to12-transition-sweep",
    "cell": "c504v4_smoke_eps3_step6to12",
    "seed": 42,
    "source": slot["source"],
    "note": (
        "On-policy greedy source R per (checkpoint, question) from the i585 "
        "source slot-stats companion pass (engine settings byte-matched to "
        "the main run; distinct lora_int_id 1..9). Keys are merged-index "
        "keys; step + provenance (retrain|hub) per entry."
    ),
    "git_commit": slot.get("git_commit"),
    "timestamp_utc": datetime.now(UTC).isoformat(),
    "completions": completions,
}
if check_mode:
    # Same code path, but write the bundle to a tempfile (no tree mutation).
    # mkstemp returns an OPEN fd — close it, or it leaks (parent round-2 review).
    check_fd, check_path = tempfile.mkstemp(prefix="i585_s612_raw_check_", suffix=".json")
    os.close(check_fd)
    raw_out = Path(check_path)
else:
    raw_out = Path(raw_path)
raw_out.parent.mkdir(parents=True, exist_ok=True)
raw_out.write_text(json.dumps(bundle, indent=2))

if check_mode:
    n_q = sum(len(v["responses"]) for v in bundle["completions"].values())
    print(f"[phase=hf_upload] check-mode OK (bundle built: {n_q} completions -> {raw_out}; no network)")
    sys.exit(0)

api = HfApi(token=os.environ.get("HF_TOKEN"))
uploads = {
    traj_path: f"{hf_prefix}/trajectory.json",
    slot_stats_path: f"{hf_prefix}/source_slot_stats.json",
    index_path: f"{hf_prefix}/checkpoint_index.json",
    provenance_path: f"{hf_prefix}/index_provenance.json",
    manifest_path: f"{hf_prefix}/retrain_manifest.json",
    train_pool_path: f"{hf_prefix}/train_pool.jsonl",
    raw_path: f"{hf_prefix}/raw_completions/c504v4_smoke_eps3_step6to12_seed42.json",
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
print(f"[phase=hf_upload] all {len(uploads)} artifacts verified on the data repo")
PY
rc=$?
[ $rc -eq 0 ] || fail "hf_upload" "HF upload step exited rc=$rc"

# ── Step 5: results commit (detached-HEAD contract, plan section 4.2 Step 3). ─
echo "[phase=results_commit] committing ${SLAB} onto issue-585"
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
    rm -f scripts/i585_retrain_per_step.py \
        scripts/i585_fetch_snapshots_build_index.py \
        scripts/i585_source_slot_stats.py
    git checkout -B issue-585 origin/issue-585
    rc=$?
    [ $rc -eq 0 ] || fail "results_commit" "git checkout -B issue-585 exited rc=$rc"
    # Raw completions are HF-data-repo-only per the plan's section 10 git list
    # (parent round-2 fix): exclude the bundle from the results commit.
    git add -- "${SLAB}/" ":(exclude)${SLAB}/raw_completions"
    git commit -m "task #585: step6to12 transition sweep (per-step retrain trajectory + companion + index + manifest)"
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

# ── Step 6: authoritative results sentinel (poll_pipeline.py contract). ──────
echo "[phase=results_sentinel] writing $SENTINEL_PATH"
ELAPSED_S=$(( $(date +%s) - START_EPOCH ))
DRY_RUN="$DRY_RUN" ELAPSED_S="$ELAPSED_S" PUSH_OK="$PUSH_OK" \
FINAL_COMMIT_SHA="$FINAL_COMMIT_SHA" PLAN_DEVIATIONS="$PLAN_DEVIATIONS" \
SENTINEL_PATH="$SENTINEL_PATH" TRAJ_PATH="$TRAJ_PATH" \
SLOT_STATS_PATH="$SLOT_STATS_PATH" INDEX_PATH="$INDEX_PATH" \
PROVENANCE_PATH="$PROVENANCE_PATH" MANIFEST_PATH="$MANIFEST_PATH" \
RAW_COMPLETIONS_PATH="$RAW_COMPLETIONS_PATH" HF_PREFIX="$HF_PREFIX" \
GPU_HOURS_BUDGETED="$GPU_HOURS_BUDGETED" TRAIN_SHA="$TRAIN_SHA" \
RIG_SHA="$RIG_SHA" PARENT_CORRECTED="$PARENT_CORRECTED" \
uv run python - <<'PY'
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

dry_run = os.environ["DRY_RUN"] == "1"

eval_numbers: dict = {}
if not dry_run:
    traj = json.loads(Path(os.environ["TRAJ_PATH"]).read_text())
    prov = json.loads(Path(os.environ["PROVENANCE_PATH"]).read_text())
    parent = json.loads(Path(os.environ["PARENT_CORRECTED"]).read_text())
    floor_dg = float(parent["floor_delta_g"])
    ceiling_logp = float(parent["ceiling_logp"])
    ref_step6 = next(
        float(r["source_dg"]) for r in parent["smoke_table"] if abs(r["ckpt_frac"] - 0.08) < 1e-6
    )

    def resolution(held_out: dict) -> float:
        n_in = n_tot = 0
        for per_q in held_out.values():
            for leaf in per_q.values():
                n_tot += 1
                dg = float(leaf["delta_g"])
                if dg >= floor_dg and float(leaf["g_logp"]) <= ceiling_logp:
                    n_in += 1
        return n_in / n_tot if n_tot else float("nan")

    rows = []
    hub6_dg = None
    for ck in sorted(traj["checkpoints"], key=lambda c: float(c["frac"])):
        matches = [k for k in prov if float(k) == float(ck["frac"])]
        key = matches[0] if len(matches) == 1 else f"{ck['frac']}"
        info = prov.get(key, {})
        src = ck["source_self"]
        row = {
            "key": key,
            "step": info.get("step"),
            "provenance": info.get("provenance"),
            "source_dg": src["delta_g_mean"],
            "source_emission": src["emission_p"],
            "source_r_collapsed": src["r_collapsed"],
            "bystander_resolution": resolution(ck["held_out"]),
        }
        rows.append(row)
        if info.get("provenance") == "hub" and info.get("step") == 6:
            hub6_dg = float(src["delta_g_mean"])
    eval_numbers["per_checkpoint"] = rows
    if hub6_dg is None or math.isnan(hub6_dg):
        raise RuntimeError("sentinel: hub step-6 read missing from trajectory/provenance join")
    eval_numbers["validity_kill_quickread"] = {
        "hub_step6_dg": hub6_dg,
        "parent_committed_dg": ref_step6,
        "abs_diff": abs(hub6_dg - ref_step6),
        "tolerance_nats": 2.0,
        "warn_nats": 0.5,
        "pass": abs(hub6_dg - ref_step6) <= 2.0,
    }
else:
    eval_numbers["dry_run"] = True

gpu_hours_used = round(int(os.environ["ELAPSED_S"]) / 3600.0, 3)
note = {
    "followup_label": "step6to12-transition-sweep",
    "eval_numbers": eval_numbers,
    "eval_paths": [
        os.environ["TRAJ_PATH"],
        os.environ["SLOT_STATS_PATH"],
        os.environ["INDEX_PATH"],
        os.environ["PROVENANCE_PATH"],
        os.environ["MANIFEST_PATH"],
        os.environ["RAW_COMPLETIONS_PATH"],
    ],
    "reproducibility_card": {
        "task": 585,
        "training_sha": os.environ["TRAIN_SHA"],
        "pinned_rig_sha": os.environ["RIG_SHA"],
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "retrained_adapters": (
            "adapters/issue_585_step6to12/c504v4_smoke_eps3_seed42_retrain/ckpt_frac*"
        ),
        "hub_endpoint_adapters": (
            "adapters/issue_504_v4/c504v4_smoke_eps3_seed42/ckpt_frac{0.08,0.16}"
        ),
        "cell": "c504v4_smoke_eps3_step6to12",
        "seed": 42,
        "marker_token_id": 83399,
        "kl": True,
        "eval_invocation": (
            "i504_eval_trajectory.py --cell c504v4_smoke_eps3_step6to12 --seed 42 "
            "--max-lora-rank 8 --max-new-tokens 2048 --max-model-len 2560 "
            "--source villain (KL on, gpu_memory_utilization 0.60 default; "
            "9-entry merged index, distinct lora_int_id 1..9)"
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
    "by": "launch_issue_585_step6to12",
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
