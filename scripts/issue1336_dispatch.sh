#!/usr/bin/env bash
# #1336 resumable phase driver: g0_gate -> gen -> extract -> fit -> align -> upload.
#
# All science lives in the per-phase python scripts (issue1336_gen_answers /
# issue1336_extract_turnstore / issue1336_fit_cells / issue1336_ladder_alignment);
# this wrapper owns env setup, per-phase done-files (re-runs skip completed
# phases and completed per-cell jobs), the CUDA_VISIBLE_DEVICES work-conserving
# fan-out over the REALIZED GPU width (plan §9: the workload re-shards off
# whatever width the lane actually provisioned), progress signal files under
# $LOG_DIR/issue-1336-*.json, and the SINGLE terminal [phase=done] line
# (reserved token — the poller's done signal; python phase scripts never emit
# it). The workload side NEVER calls scripts/task.py (sentinel channel only).
#
# Gates (plan §7):
#   G0 (pre-GPU-spend): refit the pinned #825 Qwen S1 cell through the
#     generalized fit driver; FAIL -> failure signal file + non-zero exit.
#   G1 (after extract+fit wave 1): After-RLVR lmsys-chat cell first; on KILL
#     (exit 3 from --g1-check) upload wave-1 artifacts, write the results
#     sentinel with halted=true, [phase=done], exit 0 (clean halt, not crash).
#
# Usage:
#   bash scripts/issue1336_dispatch.sh all [--smoke]
#   bash scripts/issue1336_dispatch.sh <g0_gate|gen|extract|fit|align|upload> [--smoke]
#   bash scripts/issue1336_dispatch.sh d1_battery [--smoke]   # plan v7 D1.4/D1.6 GPU leg
#   bash scripts/issue1336_dispatch.sh d1_vmsteps [--smoke]    # plan v7 D1.0-D1.3+D1.7 CPU leg
#   bash scripts/issue1336_dispatch.sh d2_probe [--smoke]      # plan v7 D2 (conditional)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
[ -d "$REPO_ROOT" ] || REPO_ROOT="$PWD"
cd "$REPO_ROOT"

# Conditional .env sourcing (GCE lane exports tokens via its startup script
# and has NO .env file — never unconditional inside a classified chain).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

PHASE_ARG="${1:-all}"
SMOKE=0
for arg in "$@"; do [ "$arg" = "--smoke" ] && SMOKE=1; done

LOG_DIR="${EPS_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="$REPO_ROOT/logs"; mkdir -p "$LOG_DIR"; }

if [ "$SMOKE" -eq 1 ]; then
    OUT_DIR="data/issue_1336/eval_smoke"          # never the committed eval_results paths
    DONE_DIR="data/issue_1336/done_smoke"
    SMOKE_FLAG="--smoke"
    UPLOAD_FLAG=""                                 # smoke never touches HF
    RESULTS_KIND="epm:smoke-result"
else
    OUT_DIR="eval_results/issue_1336"
    DONE_DIR="data/issue_1336/done"
    SMOKE_FLAG=""
    UPLOAD_FLAG="--upload"
    RESULTS_KIND="epm:results"
fi
JOB_LOG_DIR="$REPO_ROOT/logs/issue1336_jobs$([ "$SMOKE" -eq 1 ] && echo _smoke || true)"
mkdir -p "$DONE_DIR" "$JOB_LOG_DIR" "$OUT_DIR"
[ -f "$DONE_DIR/start_ts" ] || date +%s > "$DONE_DIR/start_ts"

# Guarded inner `|| true`, NOT a trailing `|| echo 0`: under pipefail a
# missing nvidia-smi fails the WHOLE pipeline while wc still prints "0", so
# the trailing echo appended a second line ("0\n0") — run_queue's integer
# arithmetic then errored and the phase completed with ZERO workers spawned
# (silent no-op phase; caught by the unit-B CPU smoke).
NGPU=$( (nvidia-smi --list-gpus 2>/dev/null || true) | wc -l )
case "$NGPU" in *[!0-9]* | "") echo "[dispatch1336] FATAL: bad GPU count '$NGPU'" >&2; exit 70;; esac
echo "[dispatch1336] phase=$PHASE_ARG smoke=$SMOKE realized_gpus=$NGPU out=$OUT_DIR"

HF_DATA_REPO="superkaiba1/explore-persona-space-data"
HF_PREFIX="issue1336_rlvr_ladder"
GPU_HOURS_BUDGETED=22

# ---------------------------------------------------------------------------
# Signal files (poll_pipeline sentinel contract: sentinel_schema_version /
# kind / version required; write ONCE per unique epoch-stamped path, never
# rewrite in place; resume state lives in $DONE_DIR, never in this namespace).
# ---------------------------------------------------------------------------
emit_signal() { # $1=kind $2=gate $3=note-string
    local ts slug path
    ts="$(date +%s)-$$-$RANDOM"
    slug=$(printf '%s' "$1" | tr ':' '_')
    path="$LOG_DIR/issue-1336-${slug}-${ts}.json"
    SIG_KIND="$1" SIG_GATE="$2" SIG_NOTE="$3" SIG_OUT="$path" uv run python - <<'PY'
import json
import os

payload = {
    "sentinel_schema_version": 1,
    "kind": os.environ["SIG_KIND"],
    "version": 1,
    "task_id": 1336,
    "gate": os.environ["SIG_GATE"],
    "by": "issue1336_dispatch",
    "blocks_pipeline": False,
    "note": os.environ["SIG_NOTE"],
}
with open(os.environ["SIG_OUT"], "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[signal] wrote {os.environ['SIG_OUT']}")
PY
}

# ---------------------------------------------------------------------------
# Work-conserving queue: jobs file = "name<TAB>command" lines; a pool of
# min(NGPU,1..n_jobs) workers pops the next pending job the moment a GPU
# frees (no wave barriers). Worker w pins CUDA_VISIBLE_DEVICES=w. Per-job
# done-files make re-runs skip completed cells.
# ---------------------------------------------------------------------------
run_queue() { # $1=phase $2=jobs-file
    local phase="$1" jobs="$2" n_jobs width qdir w
    n_jobs=$(grep -c . "$jobs" || true)
    [ "$n_jobs" -eq 0 ] && return 0
    width=$NGPU
    [ "$width" -lt 1 ] && width=1
    [ "$width" -gt "$n_jobs" ] && width=$n_jobs
    qdir="$DONE_DIR/queue_$phase"
    mkdir -p "$qdir"
    echo 0 > "$qdir/next"
    rm -f "$qdir/fail"
    echo "[$phase] $n_jobs job(s) across $width worker(s)"
    for w in $(seq 0 $((width - 1))); do
        (
            while :; do
                [ -f "$qdir/fail" ] && exit 0
                idx=$(flock "$qdir/lock" bash -c \
                    "i=\$(cat '$qdir/next'); echo \$((i + 1)) > '$qdir/next'; echo \$i")
                [ "$idx" -ge "$n_jobs" ] && exit 0
                line=$(sed -n "$((idx + 1))p" "$jobs")
                name=${line%%$'\t'*}
                cmd=${line#*$'\t'}
                done_f="$DONE_DIR/${phase}__${name}.done"
                if [ -f "$done_f" ]; then
                    echo "[$phase] skip $name (already complete)"
                    continue
                fi
                jlog="$JOB_LOG_DIR/${phase}__${name}.log"
                echo "[$phase] worker=$w start $name"
                if [ "$NGPU" -gt 0 ]; then
                    ok=0
                    CUDA_VISIBLE_DEVICES=$w bash -c "$cmd" >> "$jlog" 2>&1 || ok=$?
                else
                    ok=0
                    bash -c "$cmd" >> "$jlog" 2>&1 || ok=$?
                fi
                if [ "$ok" -eq 0 ]; then
                    touch "$done_f"
                    echo "[$phase] worker=$w finished $name"
                else
                    echo "[$phase] FAILED $name rc=$ok (log: $jlog)" >&2
                    tail -25 "$jlog" >&2 || true
                    touch "$qdir/fail"
                    exit 1
                fi
            done
        ) &
    done
    wait
    if [ -f "$qdir/fail" ]; then
        echo "[$phase] phase failed — see per-job logs under $JOB_LOG_DIR" >&2
        return 1
    fi
    return 0
}

registry_lines() { # $1 = python expression printing job lines (uses cm)
    SMOKE_ENV=$SMOKE uv run python - "$1" <<'PY'
import os
import sys

from explore_persona_space.experiments.issue_1336 import common as cm

smoke = os.environ.get("SMOKE_ENV") == "1"
models = list(cm.SMOKE_MODELS) if smoke else list(cm.MODELS)
corpora = list(cm.SMOKE_CORPORA) if smoke else list(cm.CORPORA)
cells = cm.cells_for(tuple(models), tuple(corpora))
eval_sets = [(c, f) for (c, f) in cm.EVAL_SETS if c in corpora]
pairs = [(a, b) for (a, b) in cm.PAIRS if a in models and b in models]
exec(sys.argv[1])
PY
}

phase_done() { [ -f "$DONE_DIR/phase_$1.done" ]; }
mark_phase() {
    touch "$DONE_DIR/phase_$1.done"
    emit_signal "epm:progress" "phase" "issue1336 dispatch: phase $1 complete (smoke=$SMOKE, gpus=$NGPU)"
}

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
phase_g0_gate() {
    echo "[phase=g0_gate]"
    local rc=0
    uv run python scripts/issue1336_fit_cells.py --g0 --out-dir "$OUT_DIR" || rc=$?
    if [ "$rc" -ne 0 ]; then
        emit_signal "epm:failure" "phase" "failure_class: code — G0 fit-core reuse gate FAILED (rc=$rc): the generalized fit driver did not reproduce the committed Qwen S1 layer-19 R2 within tolerance; see $OUT_DIR/gates/g0_gate.json. No GPU phases were run."
        echo "[phase=g0_failed] G0 gate failed rc=$rc" >&2
        exit "$rc"
    fi
}

phase_gen() {
    echo "[phase=gen]"
    # CPU staging (corpora prompts) — model-free, idempotent.
    if [ ! -f "$DONE_DIR/gen__prep.done" ]; then
        uv run python scripts/issue1336_gen_answers.py --prep $SMOKE_FLAG \
            >> "$JOB_LOG_DIR/gen__prep.log" 2>&1
        touch "$DONE_DIR/gen__prep.done"
        echo "[gen] corpus staging complete"
    fi
    # One vLLM job per model (engine loads once, gens all corpora); rlvr first
    # so the G1 cell's inputs land earliest. Per-cell --upload persists the
    # rollout TEXT to HF BEFORE any downstream reduction (upload policy).
    local jobs="$DONE_DIR/jobs_gen.tsv"
    registry_lines '
order = ["rlvr", "base", "sft", "dpo", "rlvr_long"]
for m in [m for m in order if m in models]:
    flags = "--smoke" if smoke else "--upload"
    print(f"{m}\tuv run python scripts/issue1336_gen_answers.py --model {m} {flags}")
' > "$jobs"
    run_queue gen "$jobs"
    # Mirror generation audits into the eval tree for the keep-rate figure.
    SMOKE_ENV=$SMOKE OUT_ENV="$OUT_DIR" uv run python - <<'PY'
import json
import os
import shutil
from pathlib import Path

from explore_persona_space.experiments.issue_1336 import common as cm

smoke = os.environ.get("SMOKE_ENV") == "1"
models = list(cm.SMOKE_MODELS) if smoke else list(cm.MODELS)
corpora = list(cm.SMOKE_CORPORA) if smoke else list(cm.CORPORA)
root = Path("data/issue_1336") / ("gen_smoke" if smoke else "gen")
dst_dir = Path(os.environ["OUT_ENV"]) / "gen_audits"
dst_dir.mkdir(parents=True, exist_ok=True)
n = 0
for m in models:
    for c in corpora:
        src = root / m / c / "audit.json"
        assert src.exists(), f"missing gen audit {src}"
        json.loads(src.read_text())  # fail loud on a truncated write
        shutil.copyfile(src, dst_dir / f"audit_{m}_{c}.json")
        n += 1
print(f"[gen] mirrored {n} audits -> {dst_dir}")
PY
}

_fit_one_cell() { # $1=cell_id  (direct, non-queued G1 fit on GPU 0)
    # Own done prefix (fitg1__): the fit phase later re-fits these cells WITH
    # --matched-n, so its fit__ done-files must not be pre-satisfied here.
    local cell="$1" done_f jlog rc=0
    done_f="$DONE_DIR/fitg1__${cell}.done"
    [ -f "$done_f" ] && { echo "[fit] skip $cell (G1 fit already complete)"; return 0; }
    jlog="$JOB_LOG_DIR/fitg1__${cell}.log"
    if [ "$NGPU" -gt 0 ]; then
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1336_fit_cells.py \
            --cells "$cell" --out-dir "$OUT_DIR" $SMOKE_FLAG >> "$jlog" 2>&1 || rc=$?
    else
        uv run python scripts/issue1336_fit_cells.py \
            --cells "$cell" --out-dir "$OUT_DIR" $SMOKE_FLAG >> "$jlog" 2>&1 || rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        echo "[fit] FAILED $cell rc=$rc (log: $jlog)" >&2
        tail -25 "$jlog" >&2 || true
        return "$rc"
    fi
    touch "$done_f"
}

g1_halt() { # $1=verdict summary string
    echo "[g1] KILL verdict — halting remaining phases, persisting wave-1 artifacts"
    emit_signal "epm:progress" "phase" "issue1336 G1 kill gate fired: $1 — halting ladder; wave-1 artifacts persisted; see $OUT_DIR/gates/g1_gate.json"
    phase_upload halted
    write_results_sentinel true
    echo "[phase=done]"
    exit 0
}

phase_extract() {
    echo "[phase=extract]"
    local rc jobs
    # Wave 1: the G1 cell (After-RLVR lmsys chat) + its naturalistic sibling
    # (required extra evidence when the chat read is marginal) extract FIRST.
    jobs="$DONE_DIR/jobs_extract_wave1.tsv"
    {
        printf 'rlvr_chat_lmsys5k\tuv run python scripts/issue1336_extract_turnstore.py --model rlvr --corpus lmsys5k --format chat %s %s\n' "$UPLOAD_FLAG" "$SMOKE_FLAG"
        printf 'rlvr_naturalistic_lmsys5k\tuv run python scripts/issue1336_extract_turnstore.py --model rlvr --corpus lmsys5k --format naturalistic %s %s\n' "$UPLOAD_FLAG" "$SMOKE_FLAG"
    } > "$jobs"
    run_queue extract_wave1 "$jobs"

    # Fit the G1 cell, then evaluate the kill gate (exit 4 = need the
    # naturalistic read before deciding; exit 3 = KILL; 0 = pass).
    _fit_one_cell rlvr_chat_lmsys5k
    rc=0
    uv run python scripts/issue1336_fit_cells.py --g1-check --out-dir "$OUT_DIR" || rc=$?
    if [ "$rc" -eq 4 ]; then
        echo "[g1] chat read marginal — fitting the naturalistic sibling"
        _fit_one_cell rlvr_naturalistic_lmsys5k
        rc=0
        uv run python scripts/issue1336_fit_cells.py --g1-check --out-dir "$OUT_DIR" || rc=$?
    fi
    if [ "$rc" -eq 3 ]; then
        if [ "$SMOKE" -eq 1 ] && [ "${EPM_1336_FORCE_G1_HALT:-0}" != "1" ]; then
            # Smoke slices (n=8) cannot carry the R2 threshold — record the
            # verdict, keep exercising the full pipeline. The halt BRANCH is
            # itself exercised via EPM_1336_FORCE_G1_HALT=1.
            echo "[g1] smoke: kill verdict recorded ($OUT_DIR/gates/g1_gate.json); halt not enforced on the smoke slice"
        else
            g1_halt "After-RLVR lmsys5k best within-stage R2 below the 0.2 kill threshold (both formats where required)"
        fi
    elif [ "$rc" -ne 0 ]; then
        echo "[extract] g1-check failed rc=$rc" >&2
        exit "$rc"
    fi

    # Remaining cells, work-conserving across all realized GPUs.
    jobs="$DONE_DIR/jobs_extract_rest.tsv"
    registry_lines '
flags = "--smoke" if smoke else "--upload"
for cell in cells:
    cid, m, c, f = cell["cell_id"], cell["model"], cell["corpus"], cell["format"]
    if m == "rlvr" and c == "lmsys5k":
        continue  # wave 1 handled above
    print(
        f"{cid}\tuv run python scripts/issue1336_extract_turnstore.py "
        f"--model {m} --corpus {c} --format {f} {flags}"
    )
' > "$jobs"
    run_queue extract "$jobs"
}

phase_fit() {
    echo "[phase=fit]"
    # Per-cell fit jobs (batched Gram-GCV ridge inside; _fit_device routes to
    # the pinned GPU). Production adds the matched-n comparability refit.
    local jobs="$DONE_DIR/jobs_fit.tsv"
    OUT_ENV="$OUT_DIR" registry_lines '
extra = "--smoke" if smoke else "--matched-n"
out = os.environ["OUT_ENV"]
for cell in cells:
    cid = cell["cell_id"]
    print(
        f"{cid}\tuv run python scripts/issue1336_fit_cells.py "
        f"--cells {cid} --out-dir {out} {extra}"
    )
' > "$jobs"
    run_queue fit "$jobs"
    # Incremental persistence: preds land on HF at the end of the phase, not
    # only at terminal upload (checkpoint-per-phase). A failure here logs LOUD
    # but does not kill the phase — the terminal phase_upload re-runs the same
    # upload fail-loud (r1 review Minor 6).
    if [ "$SMOKE" -eq 0 ]; then
        upload_preds cells || echo "[upload] WARNING: incremental preds upload (cells) failed rc=$? — terminal phase_upload retries fail-loud" >&2
    fi
}

phase_align() {
    echo "[phase=align]"
    if [ ! -f "$DONE_DIR/align__selfcheck.done" ]; then
        uv run python scripts/issue1336_ladder_alignment.py --selfcheck \
            >> "$JOB_LOG_DIR/align__selfcheck.log" 2>&1
        touch "$DONE_DIR/align__selfcheck.done"
        echo "[align] vendored-helper selfcheck passed"
    fi
    local jobs="$DONE_DIR/jobs_align.tsv"
    OUT_ENV="$OUT_DIR" registry_lines '
out = os.environ["OUT_ENV"]
flag = "--smoke" if smoke else ""
for m0, m1 in pairs:
    for corpus, fmt in eval_sets:
        name = f"{m0}__{m1}_{fmt}_{corpus}"
        print(
            f"{name}\tuv run python scripts/issue1336_ladder_alignment.py "
            f"--pair {m0}:{m1} --corpus {corpus} --format {fmt} --out-dir {out} {flag}"
        )
' > "$jobs"
    run_queue align "$jobs"
    if [ ! -f "$DONE_DIR/align__decision.done" ]; then
        uv run python scripts/issue1336_ladder_alignment.py --decision --out-dir "$OUT_DIR" \
            $SMOKE_FLAG >> "$JOB_LOG_DIR/align__decision.log" 2>&1
        touch "$DONE_DIR/align__decision.done"
        echo "[align] decision aggregation complete"
    fi
    if [ "$SMOKE" -eq 0 ]; then
        upload_preds align || echo "[upload] WARNING: incremental preds upload (align) failed rc=$? — terminal phase_upload retries fail-loud" >&2
    fi
}

upload_preds() { # $1 = cells|align — bulk upload_folder (one commit), fail loud
    local which="$1" src dest
    if [ "$which" = "cells" ]; then
        src="data/issue_1336/preds"
        dest="$HF_PREFIX/analysis_tensors/preds/cells"
    else
        src="data/issue_1336/align_preds"
        dest="$HF_PREFIX/analysis_tensors/preds/align"
    fi
    [ -d "$src" ] || { echo "[upload] no $src yet — skipping preds upload"; return 0; }
    UP_SRC="$src" UP_DEST="$dest" UP_REPO="$HF_DATA_REPO" uv run python - <<'PY'
import os

from huggingface_hub import upload_folder

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing in workload env"
upload_folder(
    repo_id=os.environ["UP_REPO"],
    repo_type="dataset",
    folder_path=os.environ["UP_SRC"],
    path_in_repo=os.environ["UP_DEST"],
)
print(f"[upload] {os.environ['UP_SRC']} -> {os.environ['UP_DEST']} (one bulk commit)")
PY
}

phase_upload() { # $1 optional "halted" — wave-1-only persistence on a G1 kill
    echo "[phase=upload]"
    if [ "$SMOKE" -eq 1 ]; then
        echo "[upload] smoke: HF upload + git push skipped (scratch outputs only)"
        return 0
    fi
    upload_preds cells
    upload_preds align
    # Eval-results mirror (JSON only, non-LFS path).
    UP_SRC="$OUT_DIR" UP_DEST="$HF_PREFIX/eval_results_mirror" UP_REPO="$HF_DATA_REPO" \
        uv run python - <<'PY'
import os

from huggingface_hub import upload_folder

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing in workload env"
upload_folder(
    repo_id=os.environ["UP_REPO"],
    repo_type="dataset",
    folder_path=os.environ["UP_SRC"],
    path_in_repo=os.environ["UP_DEST"],
    allow_patterns=["*.json"],
)
print("[upload] eval_results mirror uploaded")
PY
    # Commit eval JSONs to the issue branch; push MUST be verified (#1205 —
    # the `git push || true` swallow shape is banned).
    local branch rc=0
    branch=$(git rev-parse --abbrev-ref HEAD)
    git add "$OUT_DIR"
    if ! git diff --cached --quiet; then
        git commit -m "task #1336: eval results ($([ "${1:-}" = halted ] && echo 'G1 halt, wave 1' || echo 'full ladder'))"
    fi
    git push origin "HEAD:$branch" || rc=$?
    if [ "$rc" -ne 0 ] || [ "$(git rev-list --count "origin/$branch..HEAD")" != "0" ]; then
        echo "[upload] push not landed — one retry after rebase" >&2
        git pull --rebase=merges --autostash origin "$branch"
        git push origin "HEAD:$branch"
    fi
    if [ "$(git rev-list --count "origin/$branch..HEAD")" != "0" ]; then
        echo "[upload] FATAL: result commit not on origin/$branch after retry" >&2
        exit 86
    fi
    echo "[upload] result commit verified on origin/$branch"
}

write_results_sentinel() { # $1 = halted true|false
    local ts path
    ts=$(date +%s)
    path="$LOG_DIR/issue-1336-$(printf '%s' "$RESULTS_KIND" | tr ':' '_')-${ts}.json"
    RES_OUT="$path" RES_KIND="$RESULTS_KIND" RES_HALTED="$1" RES_OUTDIR="$OUT_DIR" \
        RES_NGPU="$NGPU" RES_START="$(cat "$DONE_DIR/start_ts")" RES_SMOKE="$SMOKE" \
        RES_BUDGET="$GPU_HOURS_BUDGETED" RES_REPO="$HF_DATA_REPO" RES_PREFIX="$HF_PREFIX" \
        uv run python - <<'PY'
import json
import os
import subprocess
import time
from pathlib import Path

from explore_persona_space.experiments.issue_1336 import common as cm

out_dir = Path(os.environ["RES_OUTDIR"])
halted = os.environ["RES_HALTED"] == "true"
smoke = os.environ["RES_SMOKE"] == "1"


def _maybe(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


g0 = _maybe(out_dir / "gates" / "g0_gate.json")
g1 = _maybe(out_dir / "gates" / "g1_gate.json")
decision = _maybe(out_dir / "decision" / "headline_contrast.json")

eval_numbers: dict = {
    "g0_pass": bool(g0["pass"]) if g0 else None,
    "g1_verdict": g1.get("verdict") if g1 else None,
    "g1_chat_best_r2": g1.get("chat_best_r2") if g1 else None,
    "halted_at_g1": halted,
}
if decision is not None:
    vl = decision["verdict_lattice"]
    eval_numbers.update(
        {
            "headline_layer": decision["headline_layer"],
            "headline_eval_set": decision["headline_eval_set"],
            "contrast_C_headline": vl["contrast_C_headline"],
            "verdict": vl["verdict"],
            "h_elicit_supported": vl.get("h_elicit_supported"),
        }
    )

sha = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
gpu_hours = round(
    (time.time() - float(os.environ["RES_START"])) / 3600.0 * max(int(os.environ["RES_NGPU"]), 1),
    2,
)
plan_deviations = []
if halted:
    plan_deviations.append(
        "G1 kill gate fired after wave 1 — remaining extract/fit/align phases halted by design"
    )

note = {
    "eval_numbers": eval_numbers,
    "eval_paths": sorted(str(p) for p in out_dir.rglob("*.json"))[:200],
    "halted": halted,
    "reproducibility_card": {
        "models": [cm.MODELS[m]["hf_id"] for m in cm.MODELS],
        "hf_data_repo": os.environ["RES_REPO"],
        "hf_prefix": os.environ["RES_PREFIX"] + "/",
        "hf_hub_url": (
            f"https://huggingface.co/datasets/{os.environ['RES_REPO']}/tree/main/"
            f"{os.environ['RES_PREFIX']}"
        ),
        "constants": {
            "sampling": dict(cm.SAMPLING),
            "n_folds": cm.N_FOLDS,
            "fit_seed": cm.FIT_SEED,
            "null_draws": cm.N_NULL_DRAWS,
            "n_bootstrap": cm.N_BOOTSTRAP,
            "frozen_layers": list(cm.FROZEN_LAYERS),
            "max_model_len": cm.MAX_MODEL_LEN,
            "keep_rate_floor": cm.KEEP_RATE_FLOOR,
            "track_s_rev": cm.TRACK_S_REV,
        },
        "wandb_url": "n/a",
        "worktree_path": ".claude/worktrees/issue-1336",
        "final_commit_sha": sha,
        "gpu_hours_used": gpu_hours,
        "gpu_hours_budgeted": float(os.environ["RES_BUDGET"]),
        "plan_deviations": plan_deviations,
    },
}
payload = {
    "sentinel_schema_version": 1,
    "kind": os.environ["RES_KIND"],
    "version": 1,
    "task_id": 1336,
    "by": "issue1336_dispatch",
    "smoke": smoke,
    "halted": halted,
    "note": json.dumps(note),
}
with open(os.environ["RES_OUT"], "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[signal] wrote results sentinel {os.environ['RES_OUT']} (halted={halted})")
PY
}

# ---------------------------------------------------------------------------
# D1 diagnosis (plan v7 amendment): battery + qwen_cal on the GPU leg with its
# OWN scoped HF staging (the GCP lane is git-clone-only — no VM-local data/),
# then verdict (when the VM-side spotcheck landed on the branch), then upload
# + results signal file. Smoke: tiny-real fixture through the SAME driver.
# ---------------------------------------------------------------------------
phase_d1_battery() {
    echo "[phase=d1_battery]"
    local diag_dir="$OUT_DIR/diagnosis"
    mkdir -p "$diag_dir"
    if [ "$SMOKE" -eq 1 ]; then
        local froot="$OUT_DIR/diag_fixture"
        uv run python scripts/issue1336_smoke_fixtures.py diag \
            --out "$froot" --n 12 --layers 2 --dim 8 --seed 0
        # DG0 oracle on the fixture: the production sweep itself sets the
        # target the driver's v0 must reproduce (gate exercised for real).
        local oracle
        oracle=$(FROOT="$froot" uv run python - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np

import issue825_fit_cells as fc
import issue1336_fit_cells as f36

root = Path(os.environ["FROOT"])
targets = {}
for cell in ("rlvr_chat_lmsys5k", "rlvr_naturalistic_lmsys5k"):
    bundle = fc._load_bundle_any(root / f"turnstore_{cell}", *cell.split("_", 2))
    xy = f36._cell_xy_1336(bundle, 2)
    sweep = fc.heldout_r2_sweep(
        xy["X"], xy["Y"], xy["conv_ids"], n_folds=3, seed=0, null_draws=0, frozen_layers=(1,)
    )
    targets[cell] = float(np.nanmax(sweep["r2_obs"]))
print(json.dumps(targets))
PY
)
        uv run python scripts/issue1336_diagnose_g1.py \
            --steps stage,decomp,audit,spotcheck,qwen_cal,battery,verdict \
            --stage-root "$froot" --out-dir "$diag_dir" \
            --preds-dir "$froot/preds" --gen-dir "$froot/gen" \
            --qwen-reduced "$froot/qwen_reduced/qwen_s1_reduced.pt" \
            --folds 3 --null-draws 2 --n-boot 25 --spotcheck-n 5 --expect-n 12 \
            --dg0-targets-json "$oracle"
        # NOTE: --steps stage is inert here (fixture pre-staged under froot;
        # every stage sub-dir already exists, so the HF fetch is skipped) —
        # kept in the list so the smoke traverses the same normalized order.
        echo "[d1_battery] smoke complete (scratch $diag_dir; no uploads)"
        return 0
    fi
    local stage_root="${DIAG_STAGE_ROOT:-data/issue_1336/diag_stage}"
    local steps="stage,qwen_cal,battery"
    # Verdict needs the VM-side D1.1-D1.3 outputs (committed on the branch
    # before this leg launches); run it here only when they are present.
    if [ -f "$diag_dir/spotcheck.json" ]; then
        steps="$steps,verdict"
    else
        echo "[d1_battery] spotcheck.json absent — verdict left to the VM steps"
    fi
    uv run python scripts/issue1336_diagnose_g1.py \
        --steps "$steps" --stage-root "$stage_root" --out-dir "$diag_dir"
    upload_diag_outputs d1_battery
    commit_push_diag d1_battery "task #1336: D1 diagnosis outputs (battery + qwen_cal GPU leg)"
}

# Uploads: npz tensors -> analysis_tensors/diagnosis, JSONs -> the
# eval-results mirror (single bulk upload_folder commits, #664).
upload_diag_outputs() { # $1 = phase label
    UP_SRC="$OUT_DIR/diagnosis" UP_REPO="$HF_DATA_REPO" UP_PREFIX="$HF_PREFIX" UP_LABEL="$1" \
        uv run python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import upload_folder

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing in workload env"
src = os.environ["UP_SRC"]
repo = os.environ["UP_REPO"]
prefix = os.environ["UP_PREFIX"]
label = os.environ["UP_LABEL"]
upload_folder(
    repo_id=repo,
    repo_type="dataset",
    folder_path=src,
    path_in_repo=f"{prefix}/eval_results_mirror/diagnosis",
    allow_patterns=["*.json"],
)
print(f"[{label}] diagnosis JSON mirror uploaded")
tensors = Path(src) / "tensors"
if tensors.exists():
    upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(tensors),
        path_in_repo=f"{prefix}/analysis_tensors/diagnosis",
        allow_patterns=["*.npz"],
    )
    print(f"[{label}] diagnosis tensors uploaded")
PY
}

# Commit diagnosis JSONs to the issue branch; push verified (#1205 — the
# `git push || true` swallow shape is banned). Checkpoints/tensors are
# HF-bound, never git (eval_results/ is JSON/text only).
commit_push_diag() { # $1 = phase label, $2 = commit message
    local label="$1" msg="$2" branch rc=0
    branch=$(git rev-parse --abbrev-ref HEAD)
    git add "$OUT_DIR/diagnosis"/*.json
    if ! git diff --cached --quiet; then
        git commit -m "$msg"
    fi
    git push origin "HEAD:$branch" || rc=$?
    if [ "$rc" -ne 0 ] || [ "$(git rev-list --count "origin/$branch..HEAD")" != "0" ]; then
        echo "[$label] push not landed — one retry after rebase" >&2
        git pull --rebase=merges --autostash origin "$branch"
        git push origin "HEAD:$branch"
    fi
    if [ "$(git rev-list --count "origin/$branch..HEAD")" != "0" ]; then
        echo "[$label] FATAL: diagnosis commit not on origin/$branch after retry" >&2
        exit 86
    fi
    echo "[$label] diagnosis commit verified on origin/$branch"
}

# Fail-loud pre-verdict input check (plan v7 D1.7): the d1_vmsteps verdict
# consumes the d1_battery GPU-leg JSONs COMMITTED on the issue branch
# (0843351ab); a fresh clone missing them must die BEFORE any staging work.
assert_d1_verdict_inputs() { # $1 = diagnosis dir
    local d="$1" chat="rlvr_chat_lmsys5k" f missing=0
    for f in refit_qwen_cal.json \
        "refit_v0_${chat}.json" "refit_v1_${chat}.json" "refit_v2_${chat}.json" \
        "refit_v3_${chat}.json" "refit_v4_${chat}.json" "refit_null_std_${chat}.json"; do
        if [ ! -f "$d/$f" ]; then
            echo "[d1_vmsteps] FATAL: verdict input $d/$f missing" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo "[d1_vmsteps] FATAL: clone lacks the committed d1_battery outputs" \
            "(branch issue-1336 @ 0843351ab or later) — verdict cannot run" >&2
        exit 71
    fi
}

# ---------------------------------------------------------------------------
# Phase D1 VM steps (plan v7 D1.0-D1.3 + D1.7): stage -> decomp -> audit ->
# spotcheck -> verdict on a CPU lane (GCP cpu-mid; the shared VM killed four
# staging attempts). No GPU parts — qwen_cal/battery ran in the d1_battery
# GPU leg and their JSONs are committed on the branch (asserted present
# BEFORE staging). The driver's _fit_device already handles CPU.
# ---------------------------------------------------------------------------
phase_d1_vmsteps() {
    echo "[phase=d1_vmsteps]"
    local diag_dir="$OUT_DIR/diagnosis"
    mkdir -p "$diag_dir"
    if [ "$SMOKE" -eq 1 ]; then
        local froot="$OUT_DIR/diag_fixture"
        uv run python scripts/issue1336_smoke_fixtures.py diag \
            --out "$froot" --n 12 --layers 2 --dim 8 --seed 0
        # DG0 oracle on the fixture (same shape as the d1_battery smoke): the
        # production sweep sets the target the driver's v0 must reproduce.
        local oracle
        oracle=$(FROOT="$froot" uv run python - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np

import issue825_fit_cells as fc
import issue1336_fit_cells as f36

root = Path(os.environ["FROOT"])
targets = {}
for cell in ("rlvr_chat_lmsys5k", "rlvr_naturalistic_lmsys5k"):
    bundle = fc._load_bundle_any(root / f"turnstore_{cell}", *cell.split("_", 2))
    xy = f36._cell_xy_1336(bundle, 2)
    sweep = fc.heldout_r2_sweep(
        xy["X"], xy["Y"], xy["conv_ids"], n_folds=3, seed=0, null_draws=0, frozen_layers=(1,)
    )
    targets[cell] = float(np.nanmax(sweep["r2_obs"]))
print(json.dumps(targets))
PY
)
        # Prerequisite battery/qwen_cal fixture outputs — the smoke analogue
        # of the clone-committed GPU-leg JSONs the production path asserts on.
        uv run python scripts/issue1336_diagnose_g1.py \
            --steps qwen_cal,battery \
            --stage-root "$froot" --out-dir "$diag_dir" \
            --preds-dir "$froot/preds" --gen-dir "$froot/gen" \
            --qwen-reduced "$froot/qwen_reduced/qwen_s1_reduced.pt" \
            --folds 3 --null-draws 2 --n-boot 25 --expect-n 12 \
            --dg0-targets-json "$oracle"
        assert_d1_verdict_inputs "$diag_dir"
        # Production-shaped step list (--steps stage is inert: fixture
        # pre-staged, every stage sub-dir exists, so the HF fetch is skipped).
        uv run python scripts/issue1336_diagnose_g1.py \
            --steps stage,decomp,audit,spotcheck,verdict \
            --stage-root "$froot" --out-dir "$diag_dir" \
            --preds-dir "$froot/preds" --gen-dir "$froot/gen" \
            --qwen-reduced "$froot/qwen_reduced/qwen_s1_reduced.pt" \
            --folds 3 --spotcheck-n 5 --expect-n 12
        echo "[d1_vmsteps] smoke complete (scratch $diag_dir; no uploads)"
        return 0
    fi
    assert_d1_verdict_inputs "$diag_dir"
    local stage_root="${DIAG_STAGE_ROOT:-data/issue_1336/diag_stage}"
    uv run python scripts/issue1336_diagnose_g1.py \
        --steps stage,decomp,audit,spotcheck,verdict \
        --stage-root "$stage_root" --out-dir "$diag_dir"
    upload_diag_outputs d1_vmsteps
    commit_push_diag d1_vmsteps "task #1336: D1 diagnosis VM-steps outputs (decomp/audit/spotcheck/verdict)"
    local routed
    routed=$(uv run python -c \
        "import json,sys; print(json.load(open(sys.argv[1]))['routed_decision'])" \
        "$diag_dir/diagnosis_verdict.json")
    emit_signal "epm:progress" "d1_vmsteps" \
        "issue1336 d1_vmsteps: diagnosis verdict routed_decision=$routed (JSONs committed + mirrored)"
}

write_d1_results_sentinel() { # $1 = optional phase label (default d1_battery)
    local ts path phase="${1:-d1_battery}"
    ts=$(date +%s)
    path="$LOG_DIR/issue-1336-$(printf '%s' "$RESULTS_KIND" | tr ':' '_')-${ts}.json"
    RES_OUT="$path" RES_KIND="$RESULTS_KIND" RES_DIAG="$OUT_DIR/diagnosis" RES_PHASE="$phase" \
        RES_NGPU="$NGPU" RES_START="$(cat "$DONE_DIR/start_ts")" RES_SMOKE="$SMOKE" \
        uv run python - <<'PY'
import json
import os
import subprocess
import time
from pathlib import Path

diag = Path(os.environ["RES_DIAG"])
smoke = os.environ["RES_SMOKE"] == "1"


def _maybe(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


verdict = _maybe(diag / "diagnosis_verdict.json")
qc = _maybe(diag / "refit_qwen_cal.json")
v0 = _maybe(diag / "refit_v0_rlvr_chat_lmsys5k.json")
eval_numbers = {
    "dg0_chat": (v0 or {}).get("dg0"),
    "s_qwen": (qc or {}).get("s_qwen_standardized"),
    "bar_std": (qc or {}).get("bar_std"),
}
if verdict is not None:
    eval_numbers.update(
        {
            "lattice_inputs": verdict["lattice_inputs"],
            "accounting_set": verdict["mechanism_attribution"]["accounting_set"],
            "routed_decision": verdict["routed_decision"],
        }
    )
sha = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
gpu_hours = round(
    (time.time() - float(os.environ["RES_START"])) / 3600.0 * max(int(os.environ["RES_NGPU"]), 1),
    2,
)
note = {
    "phase": os.environ["RES_PHASE"],
    "eval_numbers": eval_numbers,
    "eval_paths": sorted(str(p) for p in diag.glob("*.json"))[:100],
    "reproducibility_card": {
        "hf_data_repo": "superkaiba1/explore-persona-space-data",
        "hf_prefix": "issue1336_rlvr_ladder/",
        "diagnosis_prefixes": [
            "issue1336_rlvr_ladder/eval_results_mirror/diagnosis/",
            "issue1336_rlvr_ladder/analysis_tensors/diagnosis/",
        ],
        "constants": {
            "lambda_grid_wide": "logspace(-2,8,21)",
            "trim_ladder": [4, 41, 410],
            "std_floor_frac": 1e-3,
            "n_null_std": 20,
            "n_bootstrap": 1000,
            "dg0_tol": 0.02,
        },
        "wandb_url": "n/a (no training in the diagnosis phase)",
        "final_commit_sha": sha,
        "gpu_hours_used": gpu_hours,
    },
}
payload = {
    "sentinel_schema_version": 1,
    "kind": os.environ["RES_KIND"],
    "version": 1,
    "task_id": 1336,
    "by": "issue1336_dispatch",
    "smoke": smoke,
    "note": json.dumps(note),
}
with open(os.environ["RES_OUT"], "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[signal] wrote d1 results sentinel {os.environ['RES_OUT']}")
PY
}

# ---------------------------------------------------------------------------
# Phase D2 (plan v7 §4, CONDITIONAL): capture-parity probe. Re-extract N
# wave-1 chat rows (same prompt ids as the stored capture) under BOTH the
# committed and the corrected convention, then compare re-extracted vs stored
# vectors per layer (cosine, max-abs-diff) -> diagnosis/d2_capture_parity.json.
# Fires ONLY on a capture-defect verdict (R2_d2_required / D2_required); the
# corrected convention consumes the slot/span offset override the D1.3
# spot-check emitted (d2_offset_override.json — data, not code, so D2 needs
# no code change when it fires). NOT a refit (n<d cannot reproduce pipeline
# R^2 — plan §4). Smoke: tiny-real — the 2-layer same-arch model + a
# synthetic 4-row allowlist + a synthetic nonzero override, through the SAME
# extract driver and parity reader (no stored-store leg, no uploads).
# ---------------------------------------------------------------------------
phase_d2_probe() {
    echo "[phase=d2_probe]"
    local diag_dir="$OUT_DIR/diagnosis"
    mkdir -p "$diag_dir"
    local override="$diag_dir/d2_offset_override.json"
    local n_rows="${D2_N_ROWS:-512}" # plan §12 deviation slot: within [256, 1024]
    local gen_root="data/issue_1336/gen"
    local prompts_root="data/issue_1336/prompts"
    local stored_dir="${D2_STORED_TURNSTORE:-data/issue_1336/d2_stored_turnstore}"
    local tiny_flag=""
    if [ "$SMOKE" -eq 1 ]; then
        n_rows=4
        uv run python scripts/issue1336_smoke_fixtures.py tiny-model --out "$OUT_DIR/tiny_model"
        uv run python scripts/issue1336_smoke_fixtures.py gen
        tiny_flag="--tiny-model-dir $OUT_DIR/tiny_model"
        gen_root="data/issue_1336/gen_smoke"
        prompts_root="data/issue_1336/prompts_smoke"
        stored_dir="" # no stored wave-1 capture in smoke: conventions-only parity
        # Fresh smoke dirs: the extract done-markers would otherwise skip
        # re-extraction against a stale allowlist.
        rm -rf "$diag_dir/d2_turnstore_committed" "$diag_dir/d2_turnstore_corrected"
        # Synthetic override — +1 answer-span head trim: nonzero (exercises the
        # shift path) while keeping the corrected render consumer-valid
        # (fixture a1 spans are far above MIN_TURN_CONTENT_TOKENS=8).
        printf '{"slot_offsets": {}, "span_offsets": {"a1": [1, 0]}}\n' > "$override"
    elif [ ! -f "$override" ]; then
        echo "[d2_probe] FATAL: $override missing — D2 fires only on a capture-defect verdict; the D1.3 spot-check must emit the indicted offset override first" >&2
        exit 78
    fi

    if [ "$SMOKE" -ne 1 ]; then
        # 0. Fresh-instance input staging (concern d2-gen-staging-missing,
        #    cr-v4): data/* is gitignored, so a fresh GCP clone (and the VM)
        #    has neither the wave-1 gen rollout text nor the prompts.
        #    Prompts: the deterministic, model-free `gen --prep` (pinned #825
        #    track_s corpus — prompt ids identical across machines). Gen
        #    rollout text: the step_stage gen leg reused verbatim (scoped
        #    _stage_prefix + _maybe_reassemble_answers; every Hub call rides
        #    hub.retry_transient — cr-v4 Minor 3).
        if [ ! -f "$prompts_root/lmsys5k.jsonl" ]; then
            uv run python scripts/issue1336_gen_answers.py --prep --corpora lmsys5k
        fi
        if [ ! -f "$gen_root/rlvr/lmsys5k/answers.jsonl" ]; then
            D2_GEN_ROOT="$gen_root" uv run python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
import issue1336_diagnose_g1 as dg  # reuse step_stage's gen-leg helpers

api, dl, hub = dg._hub_helpers()
tmp = Path("data/issue_1336/d2_gen_stage_tmp")
staged = dg._stage_prefix(
    api, hub, dl, f"{dg.cm.HF_PREFIX_1336}/raw_completions/generation/rlvr/lmsys5k", tmp
)
target = Path(os.environ["D2_GEN_ROOT"]) / "rlvr" / "lmsys5k"
target.mkdir(parents=True, exist_ok=True)
for f in staged:
    f.rename(target / f.name)
dg._maybe_reassemble_answers(target)
print(f"[d2_probe] staged {len(staged)} gen files -> {target}")
PY
        fi
        # cr-v4 Minor 2: the extract done-marker is existence-only (not
        # fingerprinted on convention/override/allowlist), so a production
        # rerun with a CHANGED override would silently reuse stale corrected
        # shards while the parity JSON labels them with the new override.
        # D2 re-extraction is cheap (~0.04 GPU-h) — start clean, mirroring
        # the smoke branch's rm -rf above.
        rm -rf "$diag_dir/d2_turnstore_committed" "$diag_dir/d2_turnstore_corrected"
    fi

    # 1. Row allowlist: the FIRST n kept wave-1 chat rows (same prompt ids as
    #    the stored capture — plan §4 Phase D2).
    ALW_OUT="$diag_dir/d2_row_allowlist.json" ALW_GEN="$gen_root" ALW_N="$n_rows" \
        uv run python - <<'PY'
import json
import os
from pathlib import Path

rows = []
with open(
    Path(os.environ["ALW_GEN"]) / "rlvr" / "lmsys5k" / "answers.jsonl", encoding="utf-8"
) as fh:
    for line in fh:  # text-mode iteration, never splitlines() (U+2028 in user text)
        line = line.strip()
        if line:
            rows.append(json.loads(line))
kept = [r for r in rows if r.get("kept")]
n = int(os.environ["ALW_N"])
assert len(kept) >= n, f"only {len(kept)} kept wave-1 rows < requested {n}"
ids = [f"s{r['prompt_idx']}" for r in kept[:n]]
Path(os.environ["ALW_OUT"]).write_text(json.dumps(ids) + "\n")
print(f"[d2_probe] allowlist: {len(ids)} rows -> {os.environ['ALW_OUT']}")
PY

    # 2. Both conventions on the SAME rows, one teacher-forced pass each
    #    (sequential on the single provisioned GPU — plan §9 D2 row).
    local conv extra
    for conv in committed corrected; do
        extra=""
        [ "$conv" = "corrected" ] && extra="--convention corrected --offset-override $override"
        # shellcheck disable=SC2086
        uv run python scripts/issue1336_extract_turnstore.py \
            --model rlvr --corpus lmsys5k --format chat \
            --gen-root "$gen_root" --prompts-root "$prompts_root" \
            --out-dir "$diag_dir/d2_turnstore_$conv" \
            --row-allowlist "$diag_dir/d2_row_allowlist.json" \
            $extra $tiny_flag $SMOKE_FLAG
    done

    # 3. Parity JSON: corrected vs committed always; each vs the STORED wave-1
    #    vectors when a stored store is reachable (defect confirmation).
    PAR_DIAG="$diag_dir" PAR_STORED="$stored_dir" PAR_HF_REPO="$HF_DATA_REPO" \
        PAR_HF_PREFIX="$HF_PREFIX" uv run python - <<'PY'
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch

diag = Path(os.environ["PAR_DIAG"])
stored_dir = os.environ.get("PAR_STORED") or ""
stem = "rlvr_chat_lmsys5k"
allow = set(json.loads((diag / "d2_row_allowlist.json").read_text()))


def load_rows(d: Path) -> dict:
    rows = {}
    for pt in sorted(d.glob(f"{stem}_shard*.pt")):
        payload = torch.load(pt, map_location="cpu")
        for cid, slots, profiles in zip(
            payload["conv_ids"], payload["slots"], payload["profiles"], strict=True
        ):
            if cid in allow:
                rows[cid] = (slots.float(), profiles.float())
    assert rows, f"no allowlist rows found under {d}"
    return rows


def compare(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    assert common, "no common rows to compare"
    n_layers = a[common[0]][0].shape[1]
    out = {"n_rows": len(common)}
    for kind, idx in (("slots", 0), ("profiles", 1)):
        cos_by_layer, mad_by_layer = [], []
        for li in range(n_layers):
            cs, md = [], []
            for cid in common:
                x = a[cid][idx][:, li, :]
                y = b[cid][idx][:, li, :]
                num = (x * y).sum(dim=-1)
                den = (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-12)
                cs.extend((num / den).tolist())
                md.append(float((x - y).abs().max()))
            cos_by_layer.append(sum(cs) / len(cs))
            mad_by_layer.append(max(md))
        out[kind] = {
            "mean_cosine_per_layer": cos_by_layer,
            "max_abs_diff_per_layer": mad_by_layer,
            "min_mean_cosine": min(cos_by_layer),
            "max_abs_diff": max(mad_by_layer),
        }
    return out


committed = load_rows(diag / "d2_turnstore_committed")
corrected = load_rows(diag / "d2_turnstore_corrected")
sha = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
result = {
    "metadata": {"git_commit": sha, "ts_unix": time.time(), "n_allowlist": len(allow)},
    "offset_override": json.loads((diag / "d2_offset_override.json").read_text()),
    "corrected_vs_committed": compare(corrected, committed),
}
if stored_dir:
    sd = Path(stored_dir)
    if not any(sd.glob(f"{stem}_shard*.pt")):
        # Fetch ONLY the stored shards holding allowlist rows: server-side
        # scoped list_repo_tree + per-file hf_hub_download (never
        # snapshot_download on the ~1M-file data repo — gotchas.md); every
        # Hub call rides hub.retry_transient (bounded outer retry, cr-v4
        # Minor 3), listing materialized inside the thunk (#779 lazy-gen).
        from huggingface_hub import HfApi, hf_hub_download

        from explore_persona_space.orchestrate import hub

        repo = os.environ["PAR_HF_REPO"]
        prefix = f"{os.environ['PAR_HF_PREFIX']}/analysis_tensors/turnstore_{stem}"
        entries = hub.retry_transient(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: scoped (path_in_repo) walk inside hub.retry_transient
                HfApi().list_repo_tree(
                    repo, path_in_repo=prefix, repo_type="dataset", recursive=False
                )
            ),
            what=f"d2 stored-shard walk {prefix}",
        )
        names = [e.path for e in entries]
        sd.mkdir(parents=True, exist_ok=True)
        for sc in sorted(p for p in names if p.endswith(".json")):
            local = hub.retry_transient(
                lambda s=sc: hf_hub_download(repo, s, repo_type="dataset"),
                what=f"d2 fetch {sc}",
            )
            meta = json.loads(Path(local).read_text())
            if allow & set(meta.get("conv_ids", [])):
                pt_name = sc[: -len(".json")] + ".pt"
                assert pt_name in names, f"sidecar {sc} has no tensor twin on the Hub"
                got = hub.retry_transient(
                    lambda p=pt_name: hf_hub_download(repo, p, repo_type="dataset"),
                    what=f"d2 fetch {pt_name}",
                )
                shutil.copy(got, sd / Path(pt_name).name)
                print(f"[d2_probe] fetched stored shard {Path(pt_name).name}")
    stored = load_rows(sd)
    # committed re-extraction vs stored = determinism / defect confirmation;
    # corrected vs stored = the correction's divergence from the capture.
    result["committed_vs_stored"] = compare(committed, stored)
    result["corrected_vs_stored"] = compare(corrected, stored)
else:
    result["stored_comparison"] = "skipped — no stored wave-1 turnstore (smoke)"
out_path = diag / "d2_capture_parity.json"
out_path.write_text(json.dumps(result, indent=2) + "\n")
print(f"[d2_probe] parity JSON -> {out_path}")
PY

    if [ "$SMOKE" -eq 1 ]; then
        echo "[d2_probe] smoke complete (scratch $diag_dir; no uploads)"
        return 0
    fi
    # Upload the diagnosis JSON mirror (single bulk commit, #664) + commit the
    # parity JSON to the issue branch, push verified (#1205 — no swallow).
    UP_SRC="$diag_dir" UP_REPO="$HF_DATA_REPO" UP_PREFIX="$HF_PREFIX" \
        uv run python - <<'PY'
import os

from huggingface_hub import upload_folder

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing in workload env"
upload_folder(
    repo_id=os.environ["UP_REPO"],
    repo_type="dataset",
    folder_path=os.environ["UP_SRC"],
    path_in_repo=f"{os.environ['UP_PREFIX']}/eval_results_mirror/diagnosis",
    allow_patterns=["*.json"],
)
print("[d2_probe] diagnosis JSON mirror uploaded")
PY
    local branch rc=0
    branch=$(git rev-parse --abbrev-ref HEAD)
    git add "$diag_dir/d2_capture_parity.json" "$diag_dir/d2_row_allowlist.json" \
        "$diag_dir/d2_offset_override.json"
    if ! git diff --cached --quiet; then
        git commit -m "task #1336: D2 capture-parity probe outputs"
    fi
    git push origin "HEAD:$branch" || rc=$?
    if [ "$rc" -ne 0 ] || [ "$(git rev-list --count "origin/$branch..HEAD")" != "0" ]; then
        echo "[d2_probe] push not landed — one retry after rebase" >&2
        git pull --rebase=merges --autostash origin "$branch"
        git push origin "HEAD:$branch"
    fi
    if [ "$(git rev-list --count "origin/$branch..HEAD")" != "0" ]; then
        echo "[d2_probe] FATAL: parity commit not on origin/$branch after retry" >&2
        exit 86
    fi
    echo "[d2_probe] parity commit verified on origin/$branch"
}

write_d2_results_sentinel() {
    local ts path
    ts=$(date +%s)
    path="$LOG_DIR/issue-1336-$(printf '%s' "$RESULTS_KIND" | tr ':' '_')-${ts}.json"
    RES_OUT="$path" RES_KIND="$RESULTS_KIND" RES_DIAG="$OUT_DIR/diagnosis" \
        RES_SMOKE="$SMOKE" uv run python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

diag = Path(os.environ["RES_DIAG"])
parity = json.loads((diag / "d2_capture_parity.json").read_text())
sha = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
note = {
    "phase": "d2_probe",
    "eval_numbers": {
        k: {
            "slots_min_mean_cosine": v["slots"]["min_mean_cosine"],
            "profiles_min_mean_cosine": v["profiles"]["min_mean_cosine"],
            "n_rows": v["n_rows"],
        }
        for k, v in parity.items()
        if isinstance(v, dict) and "slots" in v
    },
    "eval_paths": [str(diag / "d2_capture_parity.json")],
    "reproducibility_card": {
        "hf_data_repo": "superkaiba1/explore-persona-space-data",
        "diagnosis_prefixes": ["issue1336_rlvr_ladder/eval_results_mirror/diagnosis/"],
        "wandb_url": "n/a (no training in the D2 capture-parity probe)",
        "final_commit_sha": sha,
    },
}
payload = {
    "sentinel_schema_version": 1,
    "kind": os.environ["RES_KIND"],
    "version": 1,
    "task_id": 1336,
    "by": "issue1336_dispatch",
    "smoke": os.environ["RES_SMOKE"] == "1",
    "note": json.dumps(note),
}
with open(os.environ["RES_OUT"], "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[signal] wrote d2 results sentinel {os.environ['RES_OUT']}")
PY
}

run_phase() { # $1 = phase name
    if phase_done "$1"; then
        echo "[dispatch1336] phase $1 already complete — skipping"
        return 0
    fi
    "phase_$1"
    mark_phase "$1"
}

case "$PHASE_ARG" in
all)
    run_phase g0_gate
    run_phase gen
    run_phase extract
    run_phase fit
    run_phase align
    run_phase upload
    write_results_sentinel false
    echo "[phase=done]"
    ;;
g0_gate | gen | extract | fit | align | upload)
    run_phase "$PHASE_ARG"
    echo "[dispatch1336] single-phase invocation of $PHASE_ARG complete (no terminal done line)"
    ;;
d1_battery)
    # Standalone GPU-leg workload (plan v7 D1.4/D1.6): full poller contract —
    # results sentinel BEFORE the single terminal done line.
    run_phase d1_battery
    write_d1_results_sentinel
    echo "[phase=done]"
    ;;
d1_vmsteps)
    # CPU-leg workload (plan v7 D1.0-D1.3 + D1.7): stage/decomp/audit/
    # spotcheck/verdict on a fresh clone carrying the committed battery JSONs.
    run_phase d1_vmsteps
    write_d1_results_sentinel d1_vmsteps
    echo "[phase=done]"
    ;;
d2_probe)
    # CONDITIONAL GPU-leg workload (plan v7 Phase D2): capture-parity probe.
    run_phase d2_probe
    write_d2_results_sentinel
    echo "[phase=done]"
    ;;
*)
    echo "usage: bash scripts/issue1336_dispatch.sh all|g0_gate|gen|extract|fit|align|upload|d1_battery|d1_vmsteps|d2_probe [--smoke]" >&2
    exit 2
    ;;
esac
