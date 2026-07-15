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
*)
    echo "usage: bash scripts/issue1336_dispatch.sh all|g0_gate|gen|extract|fit|align|upload [--smoke]" >&2
    exit 2
    ;;
esac
