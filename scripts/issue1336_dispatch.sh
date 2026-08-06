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
# RESUME (plan v9 §4 route 1, `resume_on_recalibrated_dv`): `all` IS the
# resume entrypoint — every phase detect-and-skips completed work against
# HF/committed outputs: gen HF-resumes per (model, corpus) from the uploaded
# answers.jsonl (Phase G fully done), extract HF-resumes per cell from the
# uploaded turnstores (2/20 done) and re-extracts the rest, fit + align
# RE-EMIT everything under the recalibrated primary (recipe-keyed done files
# fit_recal/align_recal_v9 — stale pre-resume markers never skip them), and
# the G1 gate is re-adjudicated on the recalibrated read (kill bar = the
# persisted bar_r; the raw-scale KILL that triggered the diagnosis rounds no
# longer halts the resumed ladder). Fail-loud: a half-done cell on the Hub
# raises; a missing qwen_recal_cal.json aborts (exit 78), never raw bars.
#
# Usage:
#   bash scripts/issue1336_dispatch.sh all [--smoke]
#   bash scripts/issue1336_dispatch.sh <g0_gate|gen|extract|fit|align|upload> [--smoke]
#   bash scripts/issue1336_dispatch.sh d1_battery [--smoke]   # plan v7 D1.4/D1.6 GPU leg
#   bash scripts/issue1336_dispatch.sh d1_vmsteps [--smoke]    # plan v7 D1.0-D1.3+D1.7 CPU leg
#   bash scripts/issue1336_dispatch.sh d2_probe [--smoke]      # plan v7 D2 (conditional)
#   bash scripts/issue1336_dispatch.sh e1_recal [--smoke]      # plan v9 E1 recal CPU leg
#   bash scripts/issue1336_dispatch.sh e2_refit [--smoke]      # plan v9 E2 (conditional GPU leg)
#   bash scripts/issue1336_dispatch.sh all_v2 [--smoke]        # plan v13 full-corpora round:
#       c_stage -> g0v2 -> g2_parity -> gen_v2 -> extract_v2 -> fit_v2 ->
#       ladder -> upload_v2 (each also invocable standalone). Smoke IS the
#       sweep (same phases, same subprocess shape, tiny cells); the ONLY
#       GPU-less-host stub is gen_v2's vLLM engine leg (fixture-faked at the
#       engine boundary, recorded as a stub in the phase log).
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
run_queue() { # $1=phase $2=jobs-file $3=per-worker start stagger seconds (default 0: existing callers byte-unchanged)
    local phase="$1" jobs="$2" stagger="${3:-0}" n_jobs width qdir w
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
            # Staggered worker start (MooseFS-storm guard leg 4, #1689):
            # space N heavyweight first-imports on the FUSE mount. The sleep
            # MUST be if-guarded — a bare `[ ... ] && sleep` list returns 1
            # when the guard is false, and under the subshell's inherited
            # `set -e` that kills the worker BEFORE its while-loop, turning
            # the whole phase into a silent no-op (fail marker never set,
            # done file still touched).
            if [ "$stagger" -gt 0 ] && [ "$w" -gt 0 ]; then
                sleep $((w * stagger))
            fi
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

# Phase done-file keys are RECIPE-KEYED for the phases whose outputs the
# plan-v9 resume re-emits (fit + align now carry the recalibrated primary):
# a stale pre-resume phase_fit.done / phase_align.done on a reused volume
# must NEVER satisfy the resume run (a half-done or wrong-recipe phase is
# re-run, not silently skipped). gen/extract keep their unversioned keys —
# their outputs are recipe-unchanged and per-cell resume (gen HF-resume,
# extract HF-resume + done markers) handles partial completion.
phase_key() {
    case "$1" in
    fit) echo "fit_recal_v9" ;;
    align) echo "align_recal_v9" ;;
    *) echo "$1" ;;
    esac
}
phase_done() { [ -f "$DONE_DIR/phase_$(phase_key "$1").done" ]; }
mark_phase() {
    touch "$DONE_DIR/phase_$(phase_key "$1").done"
    emit_signal "epm:progress" "phase" "issue1336 dispatch: phase $1 complete (smoke=$SMOKE, gpus=$NGPU)"
}

# Qwen exchange-rate calibration (plan v9 route 1): the fit/align/G1 reads
# require $OUT_DIR/diagnosis/recal/qwen_recal_cal.json. Production: the E1
# round COMMITTED it under eval_results/issue_1336 (the branch clone carries
# it) — assert, fail loud, never synthesize. Smoke: stage a fixture cal at
# the SAME relative path so the consuming load path is identical.
ensure_recal_cal() {
    local cal="$OUT_DIR/diagnosis/recal/qwen_recal_cal.json"
    if [ "$SMOKE" -eq 1 ]; then
        # Always (re)write: the scratch smoke tree can hold a stale E1-smoke
        # computed cal (failed V-gate on synthetic data) that the loader
        # correctly refuses — the ladder smoke wants the deterministic fixture.
        uv run python scripts/issue1336_smoke_fixtures.py recal-cal --out "$cal"
    elif [ ! -f "$cal" ]; then
        emit_signal "epm:failure" "phase" "failure_class: data — qwen_recal_cal.json missing at $cal: the plan-v9 resume requires the committed E1.d exchange-rate calibration (never proceed on raw bars)"
        echo "[dispatch1336] FATAL: missing $cal (E1.d calibration)" >&2
        exit 78
    fi
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
    # Own done prefix (fitg1_recal__): the fit phase later re-fits these cells
    # WITH --matched-n, so its per-job done-files must not be pre-satisfied
    # here — and the _recal suffix means a reused volume's pre-resume fitg1__
    # markers never skip the recalibrated re-fit (plan v9 route 1).
    local cell="$1" done_f jlog rc=0
    done_f="$DONE_DIR/fitg1_recal__${cell}.done"
    [ -f "$done_f" ] && { echo "[fit] skip $cell (G1 fit already complete)"; return 0; }
    jlog="$JOB_LOG_DIR/fitg1_recal__${cell}.log"
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
    ensure_recal_cal  # the G1 gate below reads the recalibrated bars
    # Wave 1: the G1 cell (After-RLVR lmsys chat) + its naturalistic sibling
    # (required extra evidence when the chat read is marginal) extract FIRST.
    # On resume both wave-1 cells HF-resume inside the extract script (the
    # original run uploaded their turnstores — done == uploaded, #664).
    jobs="$DONE_DIR/jobs_extract_wave1.tsv"
    {
        printf 'rlvr_chat_lmsys5k\tuv run python scripts/issue1336_extract_turnstore.py --model rlvr --corpus lmsys5k --format chat %s %s\n' "$UPLOAD_FLAG" "$SMOKE_FLAG"
        printf 'rlvr_naturalistic_lmsys5k\tuv run python scripts/issue1336_extract_turnstore.py --model rlvr --corpus lmsys5k --format naturalistic %s %s\n' "$UPLOAD_FLAG" "$SMOKE_FLAG"
    } > "$jobs"
    run_queue extract_wave1 "$jobs"

    # Fit BOTH wave-1 cells under the recal recipe BEFORE the kill gate: any
    # clone (fresh or resume) carries the COMMITTED pre-resume cells JSONs
    # (no recal block), and run_g1_check reads the naturalistic JSON whenever
    # it exists — so an un-refit stale file trips the fail-loud stale assert
    # (fit_cells.py _g1_cell_reads) before the exit-4 marginal branch could
    # ever re-fit it (rc=1 crash 2026-07-16, epm:failure v1
    # g1_check_before_recal_reemit). Both fits are recipe-keyed
    # (fitg1_recal__) and cheap next to the wave-1 extraction above, which
    # already extracts both cells eagerly — the gate still fires before the
    # ladder-wide extract/fit spend on fresh AND resume runs.
    _fit_one_cell rlvr_chat_lmsys5k
    _fit_one_cell rlvr_naturalistic_lmsys5k
    # Evaluate the kill gate (exit 3 = KILL; 0 = pass; exit 4 = naturalistic
    # read genuinely absent — retained as fail-loud defense in depth, cannot
    # fire on a healthy tree now that both wave-1 fits precede the check).
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
    ensure_recal_cal  # recal primary + bars (plan v9 route 1)
    # Per-cell fit jobs (batched Gram-GCV ridge inside; _fit_device routes to
    # the pinned GPU). Production adds the matched-n comparability refit.
    # Queue name fit_recal: per-job done-files are recipe-keyed so pre-resume
    # fit__ markers on a reused volume never skip the recalibrated re-emit.
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
    run_queue fit_recal "$jobs"
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
    ensure_recal_cal  # decision bands ride the exchange rate (plan v9)
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
    # Plan v9 route 1: chat_best_r2 is the RECALIBRATED primary; the raw
    # companion + the exchange-rate bars ride along, never blended.
    "g1_primary_scale": g1.get("primary_scale") if g1 else None,
    "g1_chat_best_r2": g1.get("chat_best_r2") if g1 else None,
    "g1_kill_threshold": g1.get("kill_threshold") if g1 else None,
    "g1_raw_companion": g1.get("raw_companion") if g1 else None,
    "halted_at_g1": halted,
}
if decision is not None:
    vl = decision["verdict_lattice"]
    eval_numbers.update(
        {
            "headline_layer": decision["headline_layer"],
            "headline_eval_set": decision["headline_eval_set"],
            "primary_scale": vl.get("primary_scale"),
            "contrast_C_headline": vl["contrast_C_headline"],
            "verdict": vl["verdict"],
            "h_elicit_supported": vl.get("h_elicit_supported"),
            "contrast_C_headline_raw": (vl.get("raw_companion") or {}).get(
                "contrast_C_headline"
            ),
            "verdict_raw": (vl.get("raw_companion") or {}).get("verdict"),
        }
    )

from explore_persona_space.experiments.issue_1336.common import resolve_code_sha

sha = resolve_code_sha()  # lane-robust: rsync lanes have no .git (fellows job 17987)
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
from explore_persona_space.experiments.issue_1336.common import resolve_code_sha

sha = resolve_code_sha()  # lane-robust: rsync lanes have no .git (fellows job 17987)
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
from explore_persona_space.experiments.issue_1336.common import resolve_code_sha

sha = resolve_code_sha()  # lane-robust: rsync lanes have no .git (fellows job 17987)
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
from explore_persona_space.experiments.issue_1336.common import resolve_code_sha

sha = resolve_code_sha()  # lane-robust: rsync lanes have no .git (fellows job 17987)
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

# ---------------------------------------------------------------------------
# Phase E1 recal (plan v9): held-out recalibration + fold-exchangeability on a
# CPU lane (GCP cpu-mid, the realized d1_vmsteps pattern). Consumes the
# committed d1 battery outputs (refit_v0 JSONs on the branch, battery_v0 preds
# npz on HF) + turnstores + Qwen stems; produces recal_verdict.json + figures.
# ---------------------------------------------------------------------------
# Fail-loud pre-input check (exit 71 convention): the E1.c lambda-audit join
# consumes the d1 battery refit_v0 JSONs COMMITTED on the issue branch; a
# fresh clone missing them must die BEFORE any staging work.
assert_e1_recal_inputs() {
    local d="$OUT_DIR/diagnosis" f missing=0
    for f in refit_v0_rlvr_chat_lmsys5k.json refit_v0_rlvr_naturalistic_lmsys5k.json; do
        if [ ! -f "$d/$f" ]; then
            echo "[e1_recal] FATAL: committed diagnosis input $d/$f missing" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo "[e1_recal] FATAL: clone lacks the committed d1 battery outputs" \
            "(branch issue-1336) — the E1.c lambda join cannot run" >&2
        exit 71
    fi
}

# Uploads: recal npz tensors -> analysis_tensors/diagnosis/recal, JSONs -> the
# eval-results mirror (single bulk upload_folder commits each, #664).
upload_recal_outputs() { # $1 = phase label
    UP_SRC="$OUT_DIR/diagnosis/recal" UP_REPO="$HF_DATA_REPO" UP_PREFIX="$HF_PREFIX" UP_LABEL="$1" \
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
    path_in_repo=f"{prefix}/eval_results_mirror/diagnosis/recal",
    allow_patterns=["*.json"],
)
print(f"[{label}] recal JSON mirror uploaded")
tensors = Path(src) / "tensors"
if tensors.exists():
    upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(tensors),
        path_in_repo=f"{prefix}/analysis_tensors/diagnosis/recal",
        allow_patterns=["*.npz"],
    )
    print(f"[{label}] recal tensors (per-draw x per-layer matrices, recal preds) uploaded")
PY
}

# Commit recal JSONs + figures to the issue branch; push verified (#1205 —
# the `git push || true` swallow shape is banned). Checkpoints/tensors are
# HF-bound, never git (eval_results/ is JSON/text only).
commit_push_recal() { # $1 = phase label, $2 = commit message
    local label="$1" msg="$2" branch rc=0
    branch=$(git rev-parse --abbrev-ref HEAD)
    git add "$OUT_DIR/diagnosis/recal"/*.json
    if [ -d "figures/issue_1336/diagnosis/recal" ]; then
        git add figures/issue_1336/diagnosis/recal
    fi
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
        echo "[$label] FATAL: recal commit not on origin/$branch after retry" >&2
        exit 86
    fi
    echo "[$label] recal commit verified on origin/$branch"
}

phase_e1_recal() {
    echo "[phase=e1_recal]"
    local recal_dir="$OUT_DIR/diagnosis/recal"
    mkdir -p "$recal_dir"
    if [ "$SMOKE" -eq 1 ]; then
        local froot="$OUT_DIR/recal_fixture" committed="$OUT_DIR/recal_committed"
        uv run python scripts/issue1336_smoke_fixtures.py diag \
            --out "$froot" --n 12 --layers 2 --dim 8 --seed 0
        # DG0 oracle on the fixture (d1_vmsteps shape): the production sweep
        # sets the target the d1 battery producer's v0 must reproduce.
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
        # REAL battery producer: the battery_v0 preds npz + refit_v0 JSONs the
        # recal consumer stages/joins against (cross-phase data contract).
        uv run python scripts/issue1336_diagnose_g1.py \
            --steps battery \
            --stage-root "$froot" --out-dir "$committed/diagnosis" \
            --preds-dir "$froot/preds" --gen-dir "$froot/gen" \
            --qwen-reduced "$froot/qwen_reduced/qwen_s1_reduced.pt" \
            --folds 3 --null-draws 2 --n-boot 25 --expect-n 12 \
            --dg0-targets-json "$oracle"
        # DG-E0 oracle for the recal consumer = the producer's own v0 read at
        # the fixture's clamped verdict layer (layer 1 on the 2-layer store).
        local dge0 r2v0
        dge0=$(V0="$committed/diagnosis/refit_v0_rlvr_chat_lmsys5k.json" uv run python -c \
            "import json,os; v=json.load(open(os.environ['V0']))['r2_per_layer_obs'][1]; print(json.dumps({'l29': v, 'l30': v}))")
        r2v0=$(printf '%s' "$dge0" | uv run python -c "import json,sys; print(json.load(sys.stdin)['l29'])")
        # Production-shaped step list through the SAME driver entrypoint.
        uv run python scripts/issue1336_recal_verdict.py \
            --steps stage,qwen_recal,recal,fold_exch,verdict \
            --stage-root "$froot" --out-dir "$recal_dir" \
            --preds-dir "$froot/preds" --gen-dir "$froot/gen" \
            --battery-preds-dir "$committed/diagnosis/tensors" \
            --qwen-reduced "$froot/qwen_reduced/qwen_s1_reduced.pt" \
            --committed-eval-dir "$committed" \
            --folds 3 --recal-null-draws 8 --n-boot 25 --n-repart 25 \
            --expect-n 12 --dge0-targets-json "$dge0" --r2-v0-l29 "$r2v0"
        uv run python scripts/issue1336_recal_figures.py \
            --recal-dir "$recal_dir" --out "$OUT_DIR/figures_recal_smoke"
        echo "[e1_recal] smoke complete (scratch $recal_dir; no uploads)"
        return 0
    fi
    assert_e1_recal_inputs
    local stage_root="${DIAG_STAGE_ROOT:-data/issue_1336/diag_stage}"
    uv run python scripts/issue1336_recal_verdict.py \
        --steps stage,qwen_recal,recal,fold_exch,verdict \
        --stage-root "$stage_root" --out-dir "$recal_dir"
    uv run python scripts/issue1336_recal_figures.py --recal-dir "$recal_dir"
    upload_recal_outputs e1_recal
    commit_push_recal e1_recal \
        "task #1336: E1 recalibration outputs (recal/fold_exch/qwen/verdict + figures)"
    local routed
    routed=$(uv run python -c \
        "import json,sys; print(json.load(open(sys.argv[1]))['routed_decision'])" \
        "$recal_dir/recal_verdict.json")
    emit_signal "epm:progress" "e1_recal" \
        "issue1336 e1_recal: recal verdict routed_decision=$routed (JSONs committed + mirrored)"
}

# ---------------------------------------------------------------------------
# Phase E2 refit (plan v9, CONDITIONAL): fires only when the E1 verdict routed
# e2_refit_required (trigger 1 fold indictment -> v5-fold; trigger 2 boundary
# straddle -> v5-cal). GPU leg (capture-7b); the lattice is re-read ONCE on
# the v5 outputs (--use-e2). Smoke forces both variants at fixture scale.
# ---------------------------------------------------------------------------
phase_e2_refit() {
    echo "[phase=e2_refit]"
    local recal_dir="$OUT_DIR/diagnosis/recal"
    if [ ! -f "$recal_dir/recal_verdict.json" ]; then
        echo "[e2_refit] FATAL: $recal_dir/recal_verdict.json missing — E2 fires only on an" \
            "E1 trigger (run e1_recal first$([ "$SMOKE" -eq 1 ] && echo ' with --smoke'))" >&2
        exit 71
    fi
    if [ "$SMOKE" -eq 1 ]; then
        local froot="$OUT_DIR/recal_fixture" committed="$OUT_DIR/recal_committed" variant
        for variant in fold cal; do
            uv run python scripts/issue1336_recal_verdict.py \
                --steps e2 \
                --stage-root "$froot" --out-dir "$recal_dir" \
                --preds-dir "$froot/preds" --gen-dir "$froot/gen" \
                --battery-preds-dir "$committed/diagnosis/tensors" \
                --qwen-reduced "$froot/qwen_reduced/qwen_s1_reduced.pt" \
                --committed-eval-dir "$committed" \
                --folds 3 --recal-null-draws 8 --n-boot 25 --n-repart 25 \
                --expect-n 12 --inner-folds 3 --e2-variant "$variant"
        done
        uv run python scripts/issue1336_recal_verdict.py \
            --steps verdict --use-e2 \
            --stage-root "$froot" --out-dir "$recal_dir" \
            --preds-dir "$froot/preds" --gen-dir "$froot/gen" \
            --battery-preds-dir "$committed/diagnosis/tensors" \
            --qwen-reduced "$froot/qwen_reduced/qwen_s1_reduced.pt" \
            --committed-eval-dir "$committed" \
            --folds 3 --recal-null-draws 8 --n-boot 25 --n-repart 25 --expect-n 12
        echo "[e2_refit] smoke complete (both variants + v5 lattice re-read; no uploads)"
        return 0
    fi
    local stage_root="${DIAG_STAGE_ROOT:-data/issue_1336/diag_stage}"
    uv run python scripts/issue1336_recal_verdict.py \
        --steps stage,e2 --stage-root "$stage_root" --out-dir "$recal_dir"
    uv run python scripts/issue1336_recal_verdict.py \
        --steps verdict --use-e2 --stage-root "$stage_root" --out-dir "$recal_dir"
    uv run python scripts/issue1336_recal_figures.py --recal-dir "$recal_dir"
    upload_recal_outputs e2_refit
    commit_push_recal e2_refit \
        "task #1336: E2 conditional refit outputs (v5 + lattice re-read + figures)"
    local routed
    routed=$(uv run python -c \
        "import json,sys; print(json.load(open(sys.argv[1]))['routed_decision'])" \
        "$recal_dir/recal_verdict.json")
    emit_signal "epm:progress" "e2_refit" \
        "issue1336 e2_refit: v5 lattice re-read routed_decision=$routed (JSONs committed + mirrored)"
}

write_e1_results_sentinel() { # $1 = phase label (e1_recal | e2_refit)
    local ts path phase="${1:-e1_recal}"
    ts=$(date +%s)
    path="$LOG_DIR/issue-1336-$(printf '%s' "$RESULTS_KIND" | tr ':' '_')-${ts}.json"
    RES_OUT="$path" RES_KIND="$RESULTS_KIND" RES_RECAL="$OUT_DIR/diagnosis/recal" \
        RES_PHASE="$phase" RES_NGPU="$NGPU" RES_START="$(cat "$DONE_DIR/start_ts")" \
        RES_SMOKE="$SMOKE" uv run python - <<'PY'
import json
import os
import subprocess
import time
from pathlib import Path

recal = Path(os.environ["RES_RECAL"])
smoke = os.environ["RES_SMOKE"] == "1"


def _maybe(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


verdict = _maybe(recal / "recal_verdict.json")
qc = _maybe(recal / "qwen_recal_cal.json")
eval_numbers = {
    "s_qwen_recal": (qc or {}).get("s_qwen_recal"),
    "bar_r": (qc or {}).get("bar_r"),
}
if verdict is not None:
    eval_numbers.update(
        {
            "lattice_inputs": verdict["lattice_inputs"],
            "v_gate": verdict["v_gate"]["outcome"],
            "a_r": verdict["mechanism_account"]["a_r"],
            "e2_trigger": verdict["e2_trigger"],
            "routed_decision": verdict["routed_decision"],
            "route_reason": verdict["route_reason"],
        }
    )
from explore_persona_space.experiments.issue_1336.common import resolve_code_sha

sha = resolve_code_sha()  # lane-robust: rsync lanes have no .git (fellows job 17987)
gpu_hours = round(
    (time.time() - float(os.environ["RES_START"])) / 3600.0 * max(int(os.environ["RES_NGPU"]), 1),
    2,
)
note = {
    "phase": os.environ["RES_PHASE"],
    "eval_numbers": eval_numbers,
    "eval_paths": [str(recal / "recal_verdict.json")],
    "gpu_hours_estimate": gpu_hours,
    "reproducibility_card": {
        "hf_data_repo": "superkaiba1/explore-persona-space-data",
        "recal_prefixes": [
            "issue1336_rlvr_ladder/eval_results_mirror/diagnosis/recal/",
            "issue1336_rlvr_ladder/analysis_tensors/diagnosis/recal/",
        ],
        "wandb_url": "n/a (no training in the E1/E2 recalibration round)",
        "final_commit_sha": sha,
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
print(f"[signal] wrote e1/e2 results sentinel {os.environ['RES_OUT']}")
PY
}

# ---------------------------------------------------------------------------
# v2 phases (plan v13, round `full-corpora-stage-evals-metric-ladder`):
# c_stage -> g0v2 -> g2_parity -> gen_v2 -> extract_v2 -> fit_v2 -> ladder ->
# upload_v2. Same conventions as the v1 phases: OK-flag/done-marker resume,
# single exit point per phase, work-conserving run_queue over the realized
# GPU width, sentinels only (pod side never shells task.py), per-phase
# resolved-mount headroom asserts (#1333/#1586 resume-aware form).
# ---------------------------------------------------------------------------
WAVE1_TS_DIR="data/issue_1336/turnstore_wave1"
V2_BARS="$OUT_DIR/gates_v2/v2_bars.json"
GPU_HOURS_BUDGETED_V2=98 # plan v13 §9 total
FIT_V2_PLANNED_WALL_H=2.3 # plan §9 FIT row (booked, x2 pilot presumption)
LADDER_PLANNED_WALL_H=1.4 # plan §9 LAD row (booked, x2 pilot presumption)

registry_lines_v2() { # $1 = python expression printing job lines (v2 registries)
    SMOKE_ENV=$SMOKE uv run python - "$1" <<'PY'
import os
import sys

from explore_persona_space.experiments.issue_1336 import common as cm

smoke = os.environ.get("SMOKE_ENV") == "1"
models = list(cm.SMOKE_MODELS) if smoke else list(cm.MODELS)
# GEN/EXTRACT corpus set: production = every corpus with NEW prompts (plan
# §4: gsm8k_test1319 is fully reused); smoke = the concat corpus + ONE
# fresh-build corpus so BOTH corpus classes get a smoke cell (multi-arm
# smoke rule; fit-grain smoke stays cm.SMOKE_CORPORA_V2).
gen_corpora = (
    ["lmsys23k", "sft11k"]
    if smoke
    else [c for c in cm.V2_CORPORA if c not in cm.V2_FULLY_REUSED_GEN]
)
# NATURALISTIC gen corpus set (round 5b; user directive: "run naturalistic on
# everything (but only the full context arm)"): ALL SEVEN v2 corpora. The
# V2_FULLY_REUSED_GEN exclusion above is CHAT-only (wave-1 CHAT gens are
# reused; NO naturalistic wave-1 exists), so gsm8k_test1319 IS generated
# here. Smoke keeps gen_corpora's two-class pair (concat + fresh-build).
nat_gen_corpora = ["lmsys23k", "sft11k"] if smoke else list(cm.V2_CORPORA)
fit_corpora = list(cm.SMOKE_CORPORA_V2) if smoke else list(cm.V2_CORPORA)
fit_cells = cm.cells_v2_for(tuple(models), tuple(fit_corpora))
capture_cells = [
    c for c in cm.cells_v2_for(tuple(models), tuple(gen_corpora)) if c["x_slot"] == "context"
]
surfaces = [(c, f) for (c, f) in cm.v2_surfaces() if c in fit_corpora]
pairs = [(a, b) for (a, b) in cm.PAIRS if a in models and b in models]
exec(sys.argv[1])
PY
}

_headroom_v2() { # $1=phase $2=need_gb (production) — smoke asserts a 1 GB floor
    local need="$2"
    [ "$SMOKE" -eq 1 ] && need=1
    HR_PHASE="$1" HR_NEED="$need" uv run python - <<'PY'
import os
from pathlib import Path

from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

root = Path("data/issue_1336")
root.mkdir(parents=True, exist_ok=True)
free = assert_out_root_headroom(root, float(os.environ["HR_NEED"]), phase=os.environ["HR_PHASE"])
print(f"[headroom] {os.environ['HR_PHASE']}: {free:.1f} GB free >= {os.environ['HR_NEED']} GB")
PY
}

phase_c_stage() {
    echo "[phase=c_stage]"
    if [ "$SMOKE" -eq 1 ]; then
        # Smoke: build the tiny corpora LOCALLY with the REAL builder code
        # (incl. the bounded lmsys streaming probe; never uploaded).
        if [ ! -f "$DONE_DIR/c_stage__smoke_corpora.done" ]; then
            uv run python scripts/issue1336_stage_corpora.py --smoke \
                >> "$JOB_LOG_DIR/c_stage__smoke_corpora.log" 2>&1
            touch "$DONE_DIR/c_stage__smoke_corpora.done"
        fi
        echo "[c_stage] smoke corpora built (local scratch; no Hub staging)"
        return 0
    fi
    _headroom_v2 c_stage 90
    # 1. corpora_v2 via the Unit-A manifest-aware reader (local-first ->
    #    HF fallback; sha-verified shards). The staged-count line is the
    #    crash-fix fix-engaged signal (plan §9 cross-phase reads).
    if [ ! -f "$DONE_DIR/c_stage__corpora.done" ]; then
        uv run python - <<'PY' >> "$JOB_LOG_DIR/c_stage__corpora.log" 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from issue1336_stage_corpora import V2_CORPORA, load_v2_corpus_rows

n_rows = 0
for slug in V2_CORPORA:
    rows = load_v2_corpus_rows(slug)
    n_rows += len(rows)
    print(f"[stage] corpus {slug}: {len(rows)} rows")
root = Path("data/issue_1336/corpora_v2")
n_files = sum(1 for p in root.rglob("*") if p.is_file())
print(f"[stage] corpora_v2 staged: {n_files} files ({n_rows} rows, {len(V2_CORPORA)} corpora)")
PY
        touch "$DONE_DIR/c_stage__corpora.done"
        grep -h '\[stage\] corpora_v2 staged' "$JOB_LOG_DIR/c_stage__corpora.log" | tail -n 1
    fi
    # 2. wave-1 generations (concat text-sha join source + the G2 allowlist).
    if [ ! -f "$DONE_DIR/c_stage__wave1_gen.done" ]; then
        uv run python - <<'PY' >> "$JOB_LOG_DIR/c_stage__wave1_gen.log" 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
import issue1336_diagnose_g1 as dg

from explore_persona_space.experiments.issue_1336 import common as cm

api, dl, hub = dg._hub_helpers()
total = 0
for m in cm.MODELS:
    for c in ("lmsys5k", "gsm8k_train5k"):
        target = Path("data/issue_1336/gen") / m / c
        if (target / "answers.jsonl").exists():
            print(f"[stage] wave-1 gen {m}/{c}: already staged")
            continue
        tmp = Path("data/issue_1336/wave1_gen_stage_tmp")
        staged = dg._stage_prefix(
            api,
            hub,
            dl,
            f"{cm.HF_PREFIX_1336}/raw_completions/generation/{m}/{c}",
            tmp,
            revision=cm.WAVE1_HF_REV,
        )
        assert staged, f"no wave-1 gen files under {m}/{c} @ {cm.WAVE1_HF_REV}"
        target.mkdir(parents=True, exist_ok=True)
        for f in staged:
            f.rename(target / f.name)
        dg._maybe_reassemble_answers(target)
        total += len(staged)
        print(f"[stage] wave-1 gen {m}/{c}: {len(staged)} files")
print(f"[stage] wave-1 generations staged: {total} files")
PY
        touch "$DONE_DIR/c_stage__wave1_gen.done"
    fi
    # 3. wave-1 turnstores: concat stems (lmsys5k both formats +
    #    gsm8k_train5k chat) flat into WAVE1_TS_DIR; the 5 fully-reused
    #    gsm8k_test1319 stems land verbatim in the v2 turnstore dir (their
    #    fits read them there — no re-extraction, plan §4 Phase EXT).
    if [ ! -f "$DONE_DIR/c_stage__wave1_ts.done" ]; then
        uv run python - <<'PY' >> "$JOB_LOG_DIR/c_stage__wave1_ts.log" 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
import issue1336_diagnose_g1 as dg

from explore_persona_space.experiments.issue_1336 import common as cm

api, dl, hub = dg._hub_helpers()
wave1 = Path("data/issue_1336/turnstore_wave1")
ts_v2 = Path("data/issue_1336/turnstore_v2")
tmp = Path("data/issue_1336/wave1_ts_stage_tmp")
jobs = []
for m in cm.MODELS:
    jobs.append((cm.cell_id(m, "chat", "lmsys5k"), wave1))
    jobs.append((cm.cell_id(m, "naturalistic", "lmsys5k"), wave1))
    jobs.append((cm.cell_id(m, "chat", "gsm8k_train5k"), wave1))
    jobs.append((cm.cell_id(m, "chat", "gsm8k_test1319"), ts_v2))
n_files = 0
for stem, dest in jobs:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob(f"{stem}_shard*.pt")):
        print(f"[stage] wave-1 turnstore {stem}: already staged")
        continue
    staged = dg._stage_prefix(
        api,
        hub,
        dl,
        f"{cm.HF_PREFIX_1336}/analysis_tensors/turnstore_{stem}",
        tmp,
        revision=cm.WAVE1_HF_REV,
    )
    assert staged, f"no files staged for wave-1 turnstore {stem} @ {cm.WAVE1_HF_REV}"
    for f in staged:
        f.rename(dest / f.name)
    n_files += len(staged)
    print(f"[stage] wave-1 turnstore {stem}: {len(staged)} files -> {dest}")
print(f"[stage] wave-1 turnstores staged: {n_files} files")
PY
        touch "$DONE_DIR/c_stage__wave1_ts.done"
    fi
    # 4. Qwen S1 stems @ deb7a452 (g0v2 inputs; idempotent re-check inside
    #    the gate itself).
    if [ ! -f "$DONE_DIR/c_stage__g0stems.done" ]; then
        uv run python - <<'PY' >> "$JOB_LOG_DIR/c_stage__g0stems.log" 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
import issue1336_fit_cells as fitc

fitc._g0_stage(Path("data/issue_1336/g0_qwen"))
print("[stage] Qwen S1 stems staged (g0v2 inputs)")
PY
        touch "$DONE_DIR/c_stage__g0stems.done"
    fi
    # 5. lmsys5k prompts (pinned track_s) — the G2 recapture's render source.
    if [ ! -f "$DONE_DIR/c_stage__prompts.done" ]; then
        uv run python scripts/issue1336_gen_answers.py --prep --corpora lmsys5k \
            >> "$JOB_LOG_DIR/c_stage__prompts.log" 2>&1
        touch "$DONE_DIR/c_stage__prompts.done"
    fi
    echo "[c_stage] staging complete"
}

phase_c_pool() {
    # Phase C_pool (plan v15 §4): pooled-multidataset cross-corpus dedup +
    # sentence-transformers/all-mpnet-base-v2 embed + KMeans k=50 + 80/20
    # cluster-aware split. Reuses the shared corpora_v2 reader
    # (issue1336_stage_corpora.load_v2_corpus_rows) — local-first, HF
    # fallback — so c_pool can run alone or after c_stage on the same box.
    # CONCERN M1: the driver measures the 5-way (5 checkpoints × 7 corpora)
    # prompt_id intersection FIRST via list_repo_tree + sha-verified answers
    # reassembly before dedup/embed/split. CONCERN M3: pins the mpnet
    # revision via HfApi().repo_info at invocation time and records it in
    # the output manifest.
    echo "[phase=c_pool]"
    if [ "$SMOKE" -eq 1 ]; then
        if [ ! -f "$DONE_DIR/c_pool__smoke.done" ]; then
            uv run python scripts/issue1336_pooled_split.py --smoke \
                >> "$JOB_LOG_DIR/c_pool.log" 2>&1
            touch "$DONE_DIR/c_pool__smoke.done"
        fi
        echo "[c_pool] smoke split written (local scratch; no HF upload)"
        return 0
    fi
    _headroom_v2 c_pool 5
    if [ ! -f "$DONE_DIR/c_pool__full.done" ]; then
        uv run python scripts/issue1336_pooled_split.py --full --upload \
            >> "$JOB_LOG_DIR/c_pool.log" 2>&1
        touch "$DONE_DIR/c_pool__full.done"
        grep -h '\[pool\] c_pool complete' "$JOB_LOG_DIR/c_pool.log" | tail -n 1
    fi
    emit_signal "epm:progress" "c_pool" "issue1336 pooled_split_v3 manifest committed (smoke=$SMOKE); see analysis_tensors/pooled_split_v3/split_manifest.json"
    echo "[c_pool] pooled split complete"
}

phase_g0v2() {
    echo "[phase=g0v2]"
    local rc=0
    if [ "$SMOKE" -eq 1 ]; then
        # Fixture-shaped leg (the real Qwen leg needs HF staging): leg (b)
        # Gram-vs-primal equality stays ENFORCED at any n; the (a) anchor
        # tolerance is informational on the fixture (#1345 gate-calibration
        # rule — production-n verdicts never kill a smoke).
        uv run python scripts/issue1336_smoke_fixtures.py g0-fixture \
            --out "$OUT_DIR/g0v2_fixture" >> "$JOB_LOG_DIR/g0v2__fixture.log" 2>&1
        uv run python scripts/issue1336_fit_cells.py --g0v2 \
            --g0-local-dir "$OUT_DIR/g0v2_fixture" --out-dir "$OUT_DIR" \
            >> "$JOB_LOG_DIR/g0v2__gate.log" 2>&1 || rc=$?
    else
        uv run python scripts/issue1336_fit_cells.py --g0v2 --out-dir "$OUT_DIR" \
            >> "$JOB_LOG_DIR/g0v2__gate.log" 2>&1 || rc=$?
    fi
    tail -n 3 "$JOB_LOG_DIR/g0v2__gate.log" || true
    emit_signal "epm:progress" "g0v2" "issue1336 G0' fit-core parity gate rc=$rc (smoke=$SMOKE): see $OUT_DIR/gates_v2/g0v2.json + v2_bars.json"
    if [ "$rc" -ne 0 ]; then
        emit_signal "epm:failure" "g0v2" "failure_class: code — G0' v2 fit-core parity gate FAILED (rc=$rc): legacy parity and/or Gram-vs-primal equality did not hold; see $OUT_DIR/gates_v2/g0v2.json. No Llama GPU phases were run (driver-enforced ordering)."
        echo "[phase=g0v2_failed] G0' gate failed rc=$rc" >&2
        exit "$rc"
    fi
}

phase_g2_parity() {
    echo "[phase=g2_parity]"
    local gdir="$OUT_DIR/gates_v2" n_rows="${G2_N_ROWS:-100}" stored_dir="$WAVE1_TS_DIR"
    local gen_root="data/issue_1336/gen" prompts_root="data/issue_1336/prompts" tiny_flag=""
    mkdir -p "$gdir"
    if [ "$SMOKE" -eq 1 ]; then
        n_rows=4
        gen_root="data/issue_1336/gen_smoke"
        prompts_root="data/issue_1336/prompts_smoke"
        uv run python scripts/issue1336_smoke_fixtures.py tiny-model --out "$OUT_DIR/tiny_model" \
            >> "$JOB_LOG_DIR/g2__fixtures.log" 2>&1
        uv run python scripts/issue1336_smoke_fixtures.py gen \
            >> "$JOB_LOG_DIR/g2__fixtures.log" 2>&1
        tiny_flag="--tiny-model-dir $OUT_DIR/tiny_model"
        # No stored wave-1 capture in smoke: the "stored" side is a SECOND
        # deterministic tiny-model capture of the same rows (the compare +
        # threshold code path runs for real; cos == 1.0 by determinism).
        stored_dir="$gdir/g2_stored_smoke"
        rm -rf "$stored_dir"
    fi
    rm -rf "$gdir/g2_recapture"
    # 1. Row allowlist: the first n kept wave-1 chat rows.
    ALW_OUT="$gdir/g2_row_allowlist.json" ALW_GEN="$gen_root" ALW_N="$n_rows" \
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
print(f"[g2] allowlist: {len(ids)} rows -> {os.environ['ALW_OUT']}")
PY
    # 2. Recapture with TODAY's extractor (v1 corpus mode — same script the
    #    v2 extension cells run; fresh out-dir so no stale done marker).
    local xtr_cmd
    xtr_cmd="uv run python scripts/issue1336_extract_turnstore.py --model rlvr --corpus lmsys5k --format chat --gen-root $gen_root --prompts-root $prompts_root --row-allowlist $gdir/g2_row_allowlist.json $tiny_flag $SMOKE_FLAG"
    if [ "$NGPU" -gt 0 ]; then
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=0 $xtr_cmd --out-dir "$gdir/g2_recapture" \
            >> "$JOB_LOG_DIR/g2__recapture.log" 2>&1
    else
        # shellcheck disable=SC2086
        $xtr_cmd --out-dir "$gdir/g2_recapture" >> "$JOB_LOG_DIR/g2__recapture.log" 2>&1
    fi
    if [ "$SMOKE" -eq 1 ]; then
        # shellcheck disable=SC2086
        $xtr_cmd --out-dir "$stored_dir" >> "$JOB_LOG_DIR/g2__recapture.log" 2>&1
    fi
    # 3. Per-layer cosine vs the stored wave-1 vectors (plan §7: >= 0.999).
    local rc=0
    PAR_GDIR="$gdir" PAR_STORED="$stored_dir" uv run python - <<'PY' || rc=$?
import json
import os
import subprocess
import time
from pathlib import Path

import torch

gdir = Path(os.environ["PAR_GDIR"])
stored_dir = Path(os.environ["PAR_STORED"])
stem = "rlvr_chat_lmsys5k"
allow = set(json.loads((gdir / "g2_row_allowlist.json").read_text()))
# Plan §7 registered bar; the env override exists ONLY for the degenerate
# smoke probe of the FAIL branch (never a production loosening lever).
THRESH = float(os.environ.get("EPM_1336_G2_COS_THRESH", "0.999"))


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


recap = load_rows(gdir / "g2_recapture")
stored = load_rows(stored_dir)
cmp_block = compare(recap, stored)
min_cos = min(cmp_block["slots"]["min_mean_cosine"], cmp_block["profiles"]["min_mean_cosine"])
ok = min_cos >= THRESH
from explore_persona_space.experiments.issue_1336.common import resolve_code_sha

sha = resolve_code_sha()  # lane-robust: rsync lanes have no .git (fellows job 17987)
payload = {
    "metadata": {"git_commit": sha, "ts_unix": time.time(), "n_allowlist": len(allow)},
    "gate": "G2",
    "threshold_min_mean_cosine": THRESH,
    "min_mean_cosine": min_cos,
    "recapture_vs_stored": cmp_block,
    "pass": bool(ok),
    "fallback_on_fail": "recapture reused wave-1 cells fresh (+~7 GPU-h, plan §7 registered)",
}
out_path = gdir / "g2_capture_parity.json"
out_path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"[g2] min mean cosine {min_cos:.6f} vs {THRESH} -> {'PASS' if ok else 'FAIL'} ({out_path})")
raise SystemExit(0 if ok else 3)
PY
    if [ "$rc" -eq 3 ]; then
        # Registered fallback (plan §7 G2, no re-approval needed): arm the
        # fresh recapture of every reused wave-1 cell; extract_v2 consumes
        # the flag. Surfaces in the sentinel for the VM orchestrator.
        touch "$DONE_DIR/g2_fallback_recapture"
        emit_signal "epm:progress" "g2_parity" "issue1336 G2 capture-parity gate FAIL (min cosine below 0.999) — registered fallback ARMED: extract_v2 re-captures the reused wave-1 cells fresh (+~7 GPU-h, plan §7); see $gdir/g2_capture_parity.json"
        echo "[g2] FAIL — registered fallback armed (fresh wave-1 recapture in extract_v2)"
    elif [ "$rc" -ne 0 ]; then
        echo "[g2_parity] compare failed rc=$rc" >&2
        exit "$rc"
    else
        emit_signal "epm:progress" "g2_parity" "issue1336 G2 capture-parity gate PASS (smoke=$SMOKE): wave-1 turnstores concatenable; see $gdir/g2_capture_parity.json"
    fi
}

phase_gen_v2() {
    echo "[phase=gen_v2]"
    _headroom_v2 gen_v2 10
    if [ "$SMOKE" -eq 1 ] && [ "$NGPU" -eq 0 ]; then
        # GPU-less smoke host: the vLLM ENGINE leg cannot run here — recorded
        # STUB (never a faked exit-0 of the engine). Prep (new-rows filter +
        # budget gate), template parity, render validation, audits, and
        # output writes run for REAL via the boundary-faked fixture.
        if [ ! -f "$DONE_DIR/gen_v2__fixture.done" ]; then
            uv run python scripts/issue1336_smoke_fixtures.py gen-v2 \
                >> "$JOB_LOG_DIR/gen_v2__fixture.log" 2>&1
            touch "$DONE_DIR/gen_v2__fixture.done"
        fi
        echo "[gen_v2] STUB: vLLM engine leg skipped on GPU-less host (smoke_fixtures gen-v2 faked ONLY the engine boundary; real engine path requires a GPU)"
        return 0
    fi
    # CPU staging (corpora rows + new-prompts-only filter + budget gate).
    if [ ! -f "$DONE_DIR/gen_v2__prep.done" ]; then
        local gen_corpora
        gen_corpora=$(registry_lines_v2 'print(",".join(gen_corpora))')
        uv run python scripts/issue1336_gen_answers.py --prep --corpora "$gen_corpora" \
            $SMOKE_FLAG >> "$JOB_LOG_DIR/gen_v2__prep.log" 2>&1
        touch "$DONE_DIR/gen_v2__prep.done"
        echo "[gen_v2] corpus prep complete ($gen_corpora)"
    fi
    # One vLLM job per (model, corpus) — 30 jobs production (plan §9),
    # work-conserving across the realized width; rlvr first so the G1' cell's
    # inputs land earliest. Per-cell --upload persists rollout TEXT to HF
    # BEFORE any downstream reduction (upload policy).
    local jobs="$DONE_DIR/jobs_gen_v2.tsv"
    registry_lines_v2 '
flags = "--smoke" if smoke else "--upload"
order = ["rlvr", "base", "sft", "dpo", "rlvr_long"]
for m in [m for m in order if m in models]:
    for c in gen_corpora:
        print(
            f"{m}__{c}\tuv run python scripts/issue1336_gen_answers.py "
            f"--model {m} --corpora {c} {flags}"
        )
' > "$jobs"
    run_queue gen_v2 "$jobs"
    # Mirror generation audits into the eval tree (keep-rate figure inputs).
    SMOKE_ENV=$SMOKE OUT_ENV="$OUT_DIR" registry_lines_v2 '
import json
import shutil
from pathlib import Path

root = Path("data/issue_1336") / ("gen_smoke" if smoke else "gen")
dst_dir = Path(os.environ["OUT_ENV"]) / "gen_audits_v2"
dst_dir.mkdir(parents=True, exist_ok=True)
n = 0
for m in models:
    for c in gen_corpora:
        src = root / m / c / "audit.json"
        assert src.exists(), f"missing gen audit {src}"
        json.loads(src.read_text())  # fail loud on a truncated write
        shutil.copyfile(src, dst_dir / f"audit_{m}_{c}.json")
        n += 1
print(f"[gen_v2] mirrored {n} audits -> {dst_dir}")
'
}

phase_gen_v2_nat() {
    # Round 5b: ON-POLICY NATURALISTIC generation over ALL SEVEN v2 corpora
    # (--gen-format naturalistic; gen-side registry cm.V2_GEN_FORMATS). A
    # SEPARATE invocation from the chat arm by design: its OWN queue phase
    # name keys per-job done files as gen_v2_nat__{model}__{corpus}.done and
    # its OWN phase done-file phase_gen_v2_nat.done, so pre-existing chat
    # gen_v2__*.done files can never skip a naturalistic job and this phase
    # never marks chat jobs done. Outputs land in format-keyed gen cell dirs
    # ({corpus}__gen_naturalistic via cm.gen_cell_key) + HF prefixes, so the
    # chat arm's artifacts are untouchable from here. Generation is
    # fit-arm-agnostic; the context-arm-only constraint (V2_PREFIX_ARM
    # unchanged) binds at the capture/fit registries, not here.
    echo "[phase=gen_v2_nat]"
    _headroom_v2 gen_v2_nat 10
    if [ "$SMOKE" -eq 1 ] && [ "$NGPU" -eq 0 ]; then
        # GPU-less smoke host: same recorded-STUB contract as gen_v2, via the
        # on-policy naturalistic fixture (engine boundary faked ONLY).
        if [ ! -f "$DONE_DIR/gen_v2_nat__fixture.done" ]; then
            uv run python scripts/issue1336_smoke_fixtures.py gen-natural \
                >> "$JOB_LOG_DIR/gen_v2_nat__fixture.log" 2>&1
            touch "$DONE_DIR/gen_v2_nat__fixture.done"
        fi
        echo "[gen_v2_nat] STUB: vLLM engine leg skipped on GPU-less host (smoke_fixtures gen-natural faked ONLY the engine boundary; real engine path requires a GPU)"
        return 0
    fi
    # MooseFS-storm guard (#1689 KNOWN TRIGGER — gotchas.md § MooseFS FUSE
    # READ-wedge): run_queue's N-way parallel fan-out of `uv run` jobs storms
    # a MooseFS-backed /workspace with N concurrent venv resolutions
    # (observed on THIS phase 2026-08-06: 8 workers flat at ~19s CPU,
    # wchan=request_wait_answer, GPUs 0 MiB, 0-byte job logs; the mount was
    # ANSWERING — contention, not the dead-mount #779 case — run the
    # two-probe discriminator in the rule before condemning a pod). Guard
    # shape (parity with scripts/issue1336_natgen_pod_launch.sh, commit
    # f4f708cdbe): (1) workers exec the venv python DIRECTLY so the uv
    # resolution path is ABSENT from the fan-out (stronger than UV_NO_SYNC
    # suppression — no `uv run` remains in any job line); (2) ONE serial
    # timeout-bounded pre-import warms the page cache before any fan-out;
    # (3) PYTHONUNBUFFERED=1 per job keeps logs live (0-byte logs made the
    # wedge undiagnosable from logs alone); (4) staggered worker starts
    # (run_queue arg 3, env-overridable via NATGEN_STAGGER_S).
    local pybin="$REPO_ROOT/.venv/bin/python"
    if [ ! -x "$pybin" ]; then
        echo "[gen_v2_nat] FATAL: $pybin missing/not executable — run 'uv sync' first (refusing the N-way 'uv run' fan-out on MooseFS; #1689)" >&2
        exit 1
    fi
    if ! timeout 600 "$pybin" -c "import vllm, transformers" \
        >> "$JOB_LOG_DIR/gen_v2_nat__preimport.log" 2>&1; then
        echo "[gen_v2_nat] FATAL: serial pre-import failed/timed out (600s) — venv unhealthy or mount contention; see $JOB_LOG_DIR/gen_v2_nat__preimport.log and the two-probe discriminator in .claude/rules/gotchas.md § MooseFS FUSE READ-wedge" >&2
        exit 1
    fi
    echo "[gen_v2_nat] storm guard: pre-import OK via $pybin (uv absent from fan-out; stagger ${NATGEN_STAGGER_S:-15}s)"
    # CPU staging: prep is format-blind (shared prompt JSONLs; run_prep skips
    # per-corpus files that already exist) but this arm needs ALL SEVEN
    # corpora staged — gsm8k_test1319 included, which the chat prep list
    # excludes — so it runs under its OWN done key rather than trusting a
    # chat-run gen_v2__prep.done.
    if [ ! -f "$DONE_DIR/gen_v2_nat__prep.done" ]; then
        local nat_corpora
        nat_corpora=$(registry_lines_v2 'print(",".join(nat_gen_corpora))')
        uv run python scripts/issue1336_gen_answers.py --prep --corpora "$nat_corpora" \
            $SMOKE_FLAG >> "$JOB_LOG_DIR/gen_v2_nat__prep.log" 2>&1
        touch "$DONE_DIR/gen_v2_nat__prep.done"
        echo "[gen_v2_nat] corpus prep complete ($nat_corpora)"
    fi
    # One vLLM job per (model, corpus) — 35 jobs production (5 models x 7
    # corpora), work-conserving across the realized width; rlvr first (same
    # ordering convention as gen_v2). Per-cell --upload persists rollout TEXT
    # to HF BEFORE any downstream reduction (upload policy).
    local jobs="$DONE_DIR/jobs_gen_v2_nat.tsv"
    PYBIN_ENV="$pybin" registry_lines_v2 '
flags = "--smoke" if smoke else "--upload"
pybin = os.environ["PYBIN_ENV"]  # storm guard: direct venv python, no uv resolution (#1689)
order = ["rlvr", "base", "sft", "dpo", "rlvr_long"]
for m in [m for m in order if m in models]:
    for c in nat_gen_corpora:
        print(
            f"{m}__{c}\tPYTHONUNBUFFERED=1 {pybin} scripts/issue1336_gen_answers.py "
            f"--model {m} --corpora {c} --gen-format naturalistic {flags}"
        )
' > "$jobs"
    run_queue gen_v2_nat "$jobs" "${NATGEN_STAGGER_S:-15}"
    # Mirror generation audits into the eval tree (keep-rate figure inputs).
    # Naturalistic audits live in the format-keyed cell dir
    # ({corpus}__gen_naturalistic), and the mirrored filename carries the
    # full gen cell key so chat mirrors (audit_{m}_{corpus}.json) are never
    # overwritten.
    SMOKE_ENV=$SMOKE OUT_ENV="$OUT_DIR" registry_lines_v2 '
import json
import shutil
from pathlib import Path

root = Path("data/issue_1336") / ("gen_smoke" if smoke else "gen")
dst_dir = Path(os.environ["OUT_ENV"]) / "gen_audits_v2"
dst_dir.mkdir(parents=True, exist_ok=True)
n = 0
for m in models:
    for c in nat_gen_corpora:
        cell = cm.gen_cell_key(c, "naturalistic")
        src = root / m / cell / "audit.json"
        assert src.exists(), f"missing gen audit {src}"
        json.loads(src.read_text())  # fail loud on a truncated write
        shutil.copyfile(src, dst_dir / f"audit_{m}_{cell}.json")
        n += 1
print(f"[gen_v2_nat] mirrored {n} audits -> {dst_dir}")
'
}

phase_extract_v2() {
    echo "[phase=extract_v2]"
    local tiny_flag=""
    if [ "$SMOKE" -eq 1 ] && [ "$NGPU" -eq 0 ]; then
        uv run python scripts/issue1336_smoke_fixtures.py tiny-model \
            --out "$OUT_DIR/tiny_model" >> "$JOB_LOG_DIR/extract_v2__fixtures.log" 2>&1
        tiny_flag="--tiny-model-dir $OUT_DIR/tiny_model"
    fi
    # Pending-scaled headroom (#1586 resume-aware form): ~9 GB per pending
    # capture cell (turnstore shards) + 5 GB margin.
    SMOKE_ENV=$SMOKE registry_lines_v2 '
from pathlib import Path

from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

ts = Path("data/issue_1336") / ("turnstore_v2_smoke" if smoke else "turnstore_v2")
pend = [
    c
    for c in capture_cells
    if not (ts / (cm.cell_id(c["model"], c["format"], c["corpus"]) + ".done.json")).exists()
]
need = 1 if smoke else max(5, 9 * len(pend) + 5)
free = assert_out_root_headroom(Path("data/issue_1336"), float(need), phase="extract_v2")
print(f"[headroom] extract_v2: pending={len(pend)} need={need} GB free={free:.1f} GB")
'
    # G2 registered fallback: re-capture EVERY reused wave-1 cell fresh with
    # today's extractor (+~7 GPU-h contingency, plan §7 — no re-approval).
    if [ -f "$DONE_DIR/g2_fallback_recapture" ] && [ "$SMOKE" -eq 0 ]; then
        echo "[extract_v2] G2 fallback armed — recapturing reused wave-1 cells fresh"
        if [ ! -f "$DONE_DIR/extract_v2__fallback_prep.done" ]; then
            rm -rf "$WAVE1_TS_DIR"
            # Remove the STAGED gsm8k_test stems (parity-failed generation);
            # fresh extraction re-writes them below.
            find data/issue_1336/turnstore_v2 -maxdepth 1 -name '*_chat_gsm8k_test1319*' \
                -delete 2>/dev/null || true
            uv run python scripts/issue1336_gen_answers.py --prep \
                --corpora gsm8k_train5k,gsm8k_test1319 \
                >> "$JOB_LOG_DIR/extract_v2__fallback_prep.log" 2>&1
            touch "$DONE_DIR/extract_v2__fallback_prep.done"
        fi
        local fjobs="$DONE_DIR/jobs_extract_v2_fallback.tsv"
        registry_lines_v2 '
for m in models:
    for fmt, c, dest in (
        ("chat", "lmsys5k", "data/issue_1336/turnstore_wave1"),
        ("naturalistic", "lmsys5k", "data/issue_1336/turnstore_wave1"),
        ("chat", "gsm8k_train5k", "data/issue_1336/turnstore_wave1"),
        ("chat", "gsm8k_test1319", "data/issue_1336/turnstore_v2"),
    ):
        print(
            f"w1_{m}_{fmt}_{c}\tuv run python scripts/issue1336_extract_turnstore.py "
            f"--model {m} --corpus {c} --format {fmt} --out-dir {dest} --upload"
        )
' > "$fjobs"
        run_queue extract_v2_w1 "$fjobs"
    fi
    # v2 extension cells (35 production jobs = 40 context-arm cells minus the
    # 5 fully-reused gsm8k_test cells; prefix-arm cells share these bundles).
    local jobs="$DONE_DIR/jobs_extract_v2.tsv"
    EXTRACT_TINY_FLAG="$tiny_flag" registry_lines_v2 '
flags = "--smoke" if smoke else "--upload"
tiny = os.environ.get("EXTRACT_TINY_FLAG", "")
ordered = sorted(
    capture_cells,
    key=lambda c: 0 if (c["model"] == "rlvr" and c["corpus"] == "lmsys23k") else 1,
)
for c in ordered:
    m, fmt, cc = c["model"], c["format"], c["corpus"]
    print(
        f"{m}_{fmt}_{cc}\tuv run python scripts/issue1336_extract_turnstore.py "
        f"--v2 --model {m} --corpus {cc} --format {fmt} {tiny} {flags}"
    )
' > "$jobs"
    run_queue extract_v2 "$jobs"
    # Consumed v2 gen-cache reap (#1489 last-consumer rule): the v2 corpora's
    # gen outputs are consumed ONLY by this phase (the fit/ladder sha-join
    # reads WAVE-1 gen answers, which are retained); per-cell rollout text is
    # already on the Hub (per-cell --upload above).
    if [ "$SMOKE" -eq 0 ]; then
        registry_lines_v2 '
import shutil
from pathlib import Path

n = 0
freed = 0
for m in models:
    for c in gen_corpora:
        d = Path("data/issue_1336/gen") / m / c
        if d.exists():
            freed += sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
            shutil.rmtree(d)
            n += 1
print(
    f"[reap] consumed v2 gen caches removed: {n} dirs, {freed / 1e6:.0f} MB "
    "(rollout text on Hub; wave-1 gen retained for the fit/ladder sha-join)"
)
'
    fi
}

_fit_one_cell_v2() { # $1=cell_id — direct G1' fit on GPU 0 (queue-compatible done key)
    local cell="$1" done_f jlog rc=0 extra
    done_f="$DONE_DIR/fit_v2__${cell}.done"
    [ -f "$done_f" ] && {
        echo "[fit_v2] skip $cell (G1' fit already complete)"
        return 0
    }
    jlog="$JOB_LOG_DIR/fit_v2__${cell}.log"
    extra="--matched-n --wave1-turnstore-dir $WAVE1_TS_DIR"
    [ "$SMOKE" -eq 1 ] && extra="--smoke"
    if [ "$NGPU" -gt 0 ]; then
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1336_fit_cells.py --v2 \
            --cells "$cell" --out-dir "$OUT_DIR" $extra >> "$jlog" 2>&1 || rc=$?
    else
        # shellcheck disable=SC2086
        uv run python scripts/issue1336_fit_cells.py --v2 \
            --cells "$cell" --out-dir "$OUT_DIR" $extra >> "$jlog" 2>&1 || rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        echo "[fit_v2] FAILED $cell rc=$rc (log: $jlog)" >&2
        tail -25 "$jlog" >&2 || true
        return "$rc"
    fi
    touch "$done_f"
}

g1v2_halt() { # $1=verdict summary string
    echo "[g1v2] KILL verdict — halting remaining v2 phases, persisting artifacts"
    emit_signal "epm:progress" "g1v2" "issue1336 G1' kill gate fired: $1 — halting ladder; artifacts persisted; see $OUT_DIR/gates_v2/g1v2_gate.json"
    phase_upload_v2 halted
    write_results_sentinel_v2 true
    echo "[phase=done]"
    exit 0
}

phase_fit_v2() {
    echo "[phase=fit_v2]"
    ensure_recal_cal # recal companion + G1' recal read (plan §7)
    _headroom_v2 fit_v2 40
    [ -f "$V2_BARS" ] || {
        emit_signal "epm:failure" "fit_v2" "failure_class: code — $V2_BARS missing: the G0' v2 gate must run before any fit (driver-enforced ordering)"
        echo "[fit_v2] FATAL: missing $V2_BARS" >&2
        exit 78
    }
    # G1' FIRST (plan §7): fit the After-RLVR lmsys23k chat cell, evaluate the
    # kill gate BEFORE the ladder-wide fit spend. The timed fit doubles as the
    # §9 pilot (MEASURED 1-cell basis at production shape).
    local g1cell="rlvr_chat_lmsys23k" rc=0 t0 t1 g1_wall
    t0=$(date +%s)
    _fit_one_cell_v2 "$g1cell"
    t1=$(date +%s)
    g1_wall=$((t1 - t0))
    uv run python scripts/issue1336_fit_cells.py --g1v2-check --out-dir "$OUT_DIR" || rc=$?
    if [ "$rc" -eq 3 ]; then
        if [ "$SMOKE" -eq 1 ] && [ "${EPM_1336_FORCE_G1V2_HALT:-0}" != "1" ]; then
            # Smoke slices cannot carry the R2 bar — record the verdict, keep
            # exercising the chain. The halt BRANCH is exercised via
            # EPM_1336_FORCE_G1V2_HALT=1 (v1 G1 convention).
            echo "[g1v2] smoke: kill verdict recorded ($OUT_DIR/gates_v2/g1v2_gate.json); halt not enforced on the smoke slice"
        else
            g1v2_halt "After-RLVR lmsys23k best within-stage R2 below bar_v2 on BOTH the raw and recalibrated scales"
        fi
    elif [ "$rc" -ne 0 ]; then
        echo "[fit_v2] g1v2-check failed rc=$rc" >&2
        exit "$rc"
    fi
    # Pilot re-projection (plan §9 FIT row is pilot-gated; measured 1-cell
    # basis, never the plan's asserted per-eigh class).
    if [ "$SMOKE" -eq 0 ]; then
        G1_WALL="$g1_wall" NGPU_ENV="$NGPU" PLANNED="$FIT_V2_PLANNED_WALL_H" \
            OUT_ENV="$OUT_DIR" DONE_ENV="$DONE_DIR" registry_lines_v2 '
from pathlib import Path

wall = float(os.environ["G1_WALL"])
ngpu = max(int(os.environ["NGPU_ENV"]), 1)
planned = float(os.environ["PLANNED"])
done_dir = Path(os.environ.get("DONE_ENV", "data/issue_1336/done"))
pend = [c for c in fit_cells if not (done_dir / ("fit_v2__" + c["cell_id"] + ".done")).exists()]
projected = wall * len(pend) / ngpu / 3600.0
ratio = projected / planned if planned else float("inf")
print(
    f"[fit_v2] pilot: g1-cell wall={wall:.0f}s pending={len(pend)} "
    f"projected={projected:.2f}h vs planned {planned}h (x{ratio:.2f})"
)
flag = Path(os.environ["OUT_ENV"]) / "gates_v2" / "fit_v2_pilot.json"
import json

flag.parent.mkdir(parents=True, exist_ok=True)
flag.write_text(
    json.dumps(
        {
            "component": "FIT (45 v2 cell sweeps)",
            "pilot_cell_wall_s": wall,
            "n_pending": len(pend),
            "ngpu": ngpu,
            "projected_wall_h": round(projected, 3),
            "planned_wall_h": planned,
            "ratio": round(ratio, 3),
            "basis": "measured 1-cell pilot (G1 cell) through the production entrypoint",
        },
        indent=2,
    )
    + "\n"
)
if ratio > 2.0:
    print(f"[fit_v2] DEVIATION over 2x: projected {projected:.2f}h vs planned {planned}h")
'
        # Emit the deviation sentinel iff the pilot re-projection crossed 2x
        # (the VM orchestrator owns marker routing — pod writes sentinels only).
        local ratio_line
        ratio_line=$(uv run python -c "import json;d=json.load(open('$OUT_DIR/gates_v2/fit_v2_pilot.json'));print('OVER' if d['ratio']>2.0 else 'OK', d['planned_wall_h'], d['projected_wall_h'], d['ratio'])")
        if [ "${ratio_line%% *}" = "OVER" ]; then
            set -- $ratio_line
            emit_signal "epm:compute-deviation" "fit_v2" "component: FIT (45 v2 cell sweeps)
planned_wall_h: $2
projected_wall_h: $3
ratio: $4
basis: measured 1-cell pilot (G1' cell) through the production entrypoint at production shape"
        fi
    fi
    # Full queue (per-cell done keys fit_v2__<cell>; the G1' cell skips).
    local jobs="$DONE_DIR/jobs_fit_v2.tsv"
    OUT_ENV="$OUT_DIR" W1_ENV="$WAVE1_TS_DIR" registry_lines_v2 '
extra = "--smoke" if smoke else ("--matched-n --wave1-turnstore-dir " + os.environ["W1_ENV"])
out = os.environ["OUT_ENV"]
ordered = sorted(fit_cells, key=lambda c: 0 if c["cell_id"] == "rlvr_chat_lmsys23k" else 1)
for cell in ordered:
    cid = cell["cell_id"]
    print(
        f"{cid}\tuv run python scripts/issue1336_fit_cells.py --v2 "
        f"--cells {cid} --out-dir {out} {extra}"
    )
' > "$jobs"
    run_queue fit_v2 "$jobs"
    if [ "$SMOKE" -eq 0 ]; then
        upload_preds_v2 cells || echo "[upload] WARNING: incremental preds_v2 upload failed rc=$? — terminal phase_upload_v2 retries fail-loud" >&2
    fi
}

phase_ladder() {
    echo "[phase=ladder]"
    _headroom_v2 ladder 25
    [ -f "$V2_BARS" ] || {
        emit_signal "epm:failure" "ladder" "failure_class: code — $V2_BARS missing: the G0' v2 gate must run before the ladder (band = 0.0201*ex_v2)"
        echo "[ladder] FATAL: missing $V2_BARS" >&2
        exit 78
    }
    # Headline layer via the decision module's PRE-REGISTERED rule (reused,
    # never duplicated — plan §4 Phase LAD).
    OUT_ENV="$OUT_DIR" SMOKE_ENV=$SMOKE uv run python - > "$DONE_DIR/headline_v2.txt" <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
import issue1336_decision_v2 as dv

from explore_persona_space.experiments.issue_1336 import common as cm

smoke = os.environ.get("SMOKE_ENV") == "1"
frozen = cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS
block = dv.headline_layer_rule_v2(Path(os.environ["OUT_ENV"]) / "cells_v2", frozen, smoke)
print(int(block["headline_layer"]))
PY
    local headline
    headline=$(tail -n 1 "$DONE_DIR/headline_v2.txt")
    case "$headline" in *[!0-9]* | "")
        echo "[ladder] FATAL: bad headline layer '$headline'" >&2
        exit 70
        ;;
    esac
    echo "[ladder] headline layer = $headline (pre-registered stage-symmetric rule)"
    # 1-battery pilot at production shape FIRST (Unit-C compute note: the
    # realized eigh count is ~2x the §9 per-battery assumption — measure one
    # full battery, re-project, sentinel on >2x the booked row).
    if [ "$SMOKE" -eq 0 ] && [ ! -f "$DONE_DIR/ladder__pilot.done" ]; then
        local p0 p1 pilot_wall
        p0=$(date +%s)
        if [ "$NGPU" -gt 0 ]; then
            CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1336_metric_ladder.py \
                --pair base:sft --corpus lmsys23k --format chat \
                --bars-json "$V2_BARS" --full-tier-layers "$headline" \
                --out-dir "$OUT_DIR" --wave1-turnstore-dir "$WAVE1_TS_DIR" \
                >> "$JOB_LOG_DIR/ladder__pilot.log" 2>&1
        else
            uv run python scripts/issue1336_metric_ladder.py \
                --pair base:sft --corpus lmsys23k --format chat \
                --bars-json "$V2_BARS" --full-tier-layers "$headline" \
                --out-dir "$OUT_DIR" --wave1-turnstore-dir "$WAVE1_TS_DIR" \
                >> "$JOB_LOG_DIR/ladder__pilot.log" 2>&1
        fi
        p1=$(date +%s)
        pilot_wall=$((p1 - p0))
        echo "$pilot_wall" > "$DONE_DIR/ladder_pilot_wall_s"
        touch "$DONE_DIR/ladder__pilot.done"
        PILOT_WALL="$pilot_wall" NGPU_ENV="$NGPU" PLANNED="$LADDER_PLANNED_WALL_H" \
            OUT_ENV="$OUT_DIR" uv run python - <<'PY'
import json
import os
from pathlib import Path

wall = float(os.environ["PILOT_WALL"])
ngpu = max(int(os.environ["NGPU_ENV"]), 1)
planned = float(os.environ["PLANNED"])
projected = wall * 56 / ngpu / 3600.0  # 56 pair-surface batteries (7 pairs x 8 surfaces)
ratio = projected / planned if planned else float("inf")
out = Path(os.environ["OUT_ENV"]) / "gates_v2" / "ladder_pilot.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "component": "LAD (224 tier batteries)",
            "pilot_battery_wall_s": wall,
            "projected_wall_h": round(projected, 3),
            "planned_wall_h": planned,
            "ratio": round(ratio, 3),
            "basis": "measured 1-battery pilot (base:sft x lmsys23k chat) at production shape; upper bound (ignores cross-pair W_s caching)",
        },
        indent=2,
    )
    + "\n"
)
print(f"[ladder] pilot battery wall={wall:.0f}s projected={projected:.2f}h vs planned {planned}h (x{ratio:.2f})")
PY
        local lratio
        lratio=$(uv run python -c "import json;d=json.load(open('$OUT_DIR/gates_v2/ladder_pilot.json'));print('OVER' if d['ratio']>2.0 else 'OK', d['planned_wall_h'], d['projected_wall_h'], d['ratio'])")
        if [ "${lratio%% *}" = "OVER" ]; then
            set -- $lratio
            emit_signal "epm:compute-deviation" "ladder" "component: LAD (224 tier batteries)
planned_wall_h: $2
projected_wall_h: $3
ratio: $4
basis: measured 1-battery pilot at production shape (upper bound; cross-pair W_s caching not credited)"
        fi
    fi
    # Queue: ONE job per surface, all pairs per invocation (W_s cached across
    # pairs sharing the source — Unit-C PrepCache).
    local jobs="$DONE_DIR/jobs_ladder.tsv"
    OUT_ENV="$OUT_DIR" HL_ENV="$headline" BARS_ENV="$V2_BARS" W1_ENV="$WAVE1_TS_DIR" \
        registry_lines_v2 '
out, hl, bars = os.environ["OUT_ENV"], os.environ["HL_ENV"], os.environ["BARS_ENV"]
flag = "--smoke" if smoke else ""
w1 = "" if smoke else (" --wave1-turnstore-dir " + os.environ["W1_ENV"])
pair_arg = ",".join(f"{a}:{b}" for a, b in pairs)
for corpus, fmt in surfaces:
    print(
        f"{corpus}_{fmt}\tuv run python scripts/issue1336_metric_ladder.py "
        f"--pairs {pair_arg} --corpus {corpus} --format {fmt} "
        f"--bars-json {bars} --full-tier-layers {hl} --out-dir {out}{w1} {flag}"
    )
' > "$jobs"
    run_queue ladder "$jobs"
    if [ "$SMOKE" -eq 0 ]; then
        upload_preds_v2 ladder || echo "[upload] WARNING: incremental metric_ladder_preds upload failed rc=$? — terminal phase_upload_v2 retries fail-loud" >&2
    fi
}

upload_preds_v2() { # $1 = cells|ladder — bulk upload_folder (one commit), fail loud
    local which="$1" src dest
    if [ "$which" = "cells" ]; then
        src="data/issue_1336/preds_v2"
        dest="$HF_PREFIX/analysis_tensors/preds_v2"
    else
        src="data/issue_1336/metric_ladder_preds"
        dest="$HF_PREFIX/analysis_tensors/metric_ladder_preds"
    fi
    [ -d "$src" ] || {
        echo "[upload] no $src yet — skipping preds upload"
        return 0
    }
    UP_SRC="$src" UP_DEST="$dest" UP_REPO="$HF_DATA_REPO" uv run python - <<'PY'
import os

from huggingface_hub import upload_folder

from explore_persona_space.orchestrate import hub

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing in workload env"
hub.retry_transient(
    lambda: upload_folder(
        repo_id=os.environ["UP_REPO"],
        repo_type="dataset",
        folder_path=os.environ["UP_SRC"],
        path_in_repo=os.environ["UP_DEST"],
    ),
    what=f"preds_v2 upload {os.environ['UP_DEST']}",
)
print(f"[upload] {os.environ['UP_SRC']} -> {os.environ['UP_DEST']} (one bulk commit)")
PY
}

phase_upload_v2() { # $1 optional "halted" — persistence-only on a G1' kill
    echo "[phase=upload_v2]"
    if [ "$SMOKE" -eq 1 ]; then
        echo "[upload_v2] smoke: HF upload + git push skipped (scratch outputs only)"
        return 0
    fi
    upload_preds_v2 cells
    upload_preds_v2 ladder
    # Eval-results mirror (JSON only, non-LFS path; ephemeral-lane rule).
    UP_SRC="$OUT_DIR" UP_DEST="$HF_PREFIX/eval_results_mirror_v2" UP_REPO="$HF_DATA_REPO" \
        uv run python - <<'PY'
import os

from huggingface_hub import upload_folder

from explore_persona_space.orchestrate import hub

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing in workload env"
hub.retry_transient(
    lambda: upload_folder(
        repo_id=os.environ["UP_REPO"],
        repo_type="dataset",
        folder_path=os.environ["UP_SRC"],
        path_in_repo=os.environ["UP_DEST"],
        allow_patterns=["*.json"],
    ),
    what="eval_results_mirror_v2 upload",
)
print("[upload_v2] eval_results mirror uploaded")
PY
    # Commit eval JSONs to the issue branch; push verified (#1205/#1880 —
    # fetch+rebase before retry, never a swallowed push). Rsync lanes
    # (fellows/SLURM) have NO .git in the scratch tree, so workload-side git
    # is structurally impossible there (pod-side-reporting.md, SLURM lane
    # bullet): the HF mirror above already persisted the JSONs durably, the
    # lane's fetch_results rsync-pulls eval_results/ back to the VM, and the
    # VM-side ORCHESTRATOR owns the commit. Probe [-e .git] (dir in clones,
    # file in worktrees) rather than bare rev-parse so upward .git discovery
    # from an ancestor dir can never target the wrong repo.
    if [ -e .git ] && git rev-parse --git-dir >/dev/null 2>&1; then
        local branch rc=0
        branch=$(git rev-parse --abbrev-ref HEAD)
        git add "$OUT_DIR"
        if ! git diff --cached --quiet; then
            git commit -m "task #1336: v2 eval results ($([ "${1:-}" = halted ] && echo 'G1-prime halt' || echo 'full ladder'))"
        fi
        git push origin "HEAD:$branch" || rc=$?
        if [ "$rc" -ne 0 ] || [ "$(git rev-list --count "origin/$branch..HEAD")" != "0" ]; then
            echo "[upload_v2] push not landed — one retry after rebase" >&2
            git pull --rebase=merges --autostash origin "$branch"
            git push origin "HEAD:$branch"
        fi
        if [ "$(git rev-list --count "origin/$branch..HEAD")" != "0" ]; then
            echo "[upload_v2] FATAL: result commit not on origin/$branch after retry" >&2
            exit 86
        fi
        echo "[upload_v2] result commit verified on origin/$branch"
    else
        echo "[upload_v2] no git checkout (rsync lane) — eval-results commit is VM-side (HF mirror uploaded above; fetch_results pulls eval_results/ to the VM)"
    fi
}

write_results_sentinel_v2() { # $1 = halted true|false
    local ts path
    ts=$(date +%s)
    path="$LOG_DIR/issue-1336-$(printf '%s' "$RESULTS_KIND" | tr ':' '_')-${ts}.json"
    RES_OUT="$path" RES_KIND="$RESULTS_KIND" RES_HALTED="$1" RES_OUTDIR="$OUT_DIR" \
        RES_NGPU="$NGPU" RES_START="$(cat "$DONE_DIR/start_ts")" RES_SMOKE="$SMOKE" \
        RES_BUDGET="$GPU_HOURS_BUDGETED_V2" RES_REPO="$HF_DATA_REPO" RES_PREFIX="$HF_PREFIX" \
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


g0v2 = _maybe(out_dir / "gates_v2" / "g0v2.json")
g2 = _maybe(out_dir / "gates_v2" / "g2_capture_parity.json")
g1v2 = _maybe(out_dir / "gates_v2" / "g1v2_gate.json")
bars = _maybe(out_dir / "gates_v2" / "v2_bars.json")
n_cells = len(sorted((out_dir / "cells_v2").glob("cells_*.json"))) if (out_dir / "cells_v2").exists() else 0
n_pairs = len(sorted((out_dir / "metric_ladder").glob("pair_*.json"))) if (out_dir / "metric_ladder").exists() else 0

eval_numbers = {
    "g0v2_pass": bool(g0v2["pass"]) if g0v2 else None,
    "g0v2_s_qwen_v2": (g0v2 or {}).get("leg_c_v2_anchor", {}).get("s_qwen_v2"),
    "ex_v2": (bars or {}).get("ex_v2"),
    "bar_v2": (bars or {}).get("bar_v2"),
    "g2_pass": bool(g2["pass"]) if g2 else None,
    "g2_min_mean_cosine": (g2 or {}).get("min_mean_cosine"),
    "g2_fallback_recapture_armed": bool(g2 is not None and not g2.get("pass", True)),
    "g1v2_verdict": (g1v2 or {}).get("verdict"),
    "g1v2_raw_best_r2": (g1v2 or {}).get("raw_best_r2"),
    "g1v2_recal_best_r2": (g1v2 or {}).get("recal_best_r2"),
    "n_cells_v2_json": n_cells,
    "n_metric_ladder_pairs": n_pairs,
    "halted_at_g1v2": halted,
}

from explore_persona_space.experiments.issue_1336.common import resolve_code_sha

sha = resolve_code_sha()  # lane-robust: rsync lanes have no .git (fellows job 17987)
gpu_hours = round(
    (time.time() - float(os.environ["RES_START"])) / 3600.0 * max(int(os.environ["RES_NGPU"]), 1),
    2,
)
plan_deviations = []
if halted:
    plan_deviations.append(
        "G1' kill gate fired — remaining fit/ladder phases halted by design; artifacts persisted"
    )
if eval_numbers["g2_fallback_recapture_armed"]:
    plan_deviations.append(
        "G2 capture parity FAILED — reused wave-1 cells were re-captured fresh (registered §7 fallback, +~7 GPU-h)"
    )

note = {
    "eval_numbers": eval_numbers,
    "eval_paths": sorted(
        str(p)
        for sub in ("gates_v2", "cells_v2", "metric_ladder")
        for p in (out_dir / sub).rglob("*.json")
    )[:200],
    "halted": halted,
    "reproducibility_card": {
        "models": [cm.MODELS[m]["hf_id"] for m in cm.MODELS],
        "hf_data_repo": os.environ["RES_REPO"],
        "hf_prefix": os.environ["RES_PREFIX"] + "/",
        "hf_hub_url": (
            f"https://huggingface.co/datasets/{os.environ['RES_REPO']}/tree/main/"
            f"{os.environ['RES_PREFIX']}"
        ),
        "adapter_paths": "n/a (no training in the v2 full-corpora round)",
        "wandb_url": "n/a (no training in the v2 full-corpora round)",
        "constants": {
            "sampling": dict(cm.SAMPLING),
            "n_folds": cm.N_FOLDS,
            "fit_seed": cm.FIT_SEED,
            "null_draws": cm.N_NULL_DRAWS,
            "n_bootstrap": cm.N_BOOTSTRAP,
            "frozen_layers": list(cm.FROZEN_LAYERS),
            "lambda_grid": "np.logspace(-3, 8, 23) + adaptive edge (<=2 decades/side)",
            "n_inner_lambda_folds": cm.N_INNER_LAMBDA_FOLDS_V2,
            "matched_n_v2": cm.MATCHED_N_V2,
            "matched_n_v2_seed": cm.MATCHED_N_V2_SEED,
            "corpus_sample_seed": 1336,
            "wave1_rev": cm.WAVE1_HF_REV,
            "track_s_rev": cm.TRACK_S_REV,
        },
        "worktree_path": ".claude/worktrees/issue-1336-fullcorpora",
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
print(f"[signal] wrote v2 results sentinel {os.environ['RES_OUT']} (halted={halted})")
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
all_v2)
    # Plan v13 chain (round full-corpora-stage-evals-metric-ladder), resumable
    # per phase/cell; the G0'/G2/G1' gates are driver-enforced IN ORDER before
    # any Llama read / reuse concat / ladder-wide fit spend respectively.
    run_phase c_stage
    run_phase g0v2
    run_phase g2_parity
    run_phase gen_v2
    run_phase extract_v2
    run_phase fit_v2
    run_phase ladder
    run_phase upload_v2
    write_results_sentinel_v2 false
    echo "[phase=done]"
    ;;
c_stage | g0v2 | g2_parity | gen_v2 | gen_v2_nat | extract_v2 | fit_v2 | ladder | upload_v2)
    # gen_v2_nat is deliberately NOT in the all_v2 chain: the naturalistic
    # arm is a SEPARATE invocation from the chat arm (round 5b).
    run_phase "$PHASE_ARG"
    echo "[dispatch1336] single-phase invocation of $PHASE_ARG complete (no terminal done line)"
    ;;
all_v3)
    # Plan v15 chain (pooled-multidataset on-off-policy stage-transfer). This
    # composite is INCREMENTAL against all_v2: it prepends c_pool (pooled
    # split_manifest builder consumed by later v3 phases). Subsequent Unit B/C
    # v3 phases (g0v3, extract_offpolicy, fit_pool, ladder_pool, ladder_cluster,
    # upload_v3) will be added in later units; the c_pool block is the Unit A
    # deliverable and can be exercised standalone via `c_pool`.
    run_phase c_pool
    write_results_sentinel_v2 false
    echo "[phase=done]"
    ;;
c_pool)
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
e1_recal)
    # CPU-leg workload (plan v9 E1): held-out recalibration + fold
    # exchangeability + verdict on a fresh clone carrying the committed
    # d1 battery JSONs (asserted present BEFORE staging).
    run_phase e1_recal
    write_e1_results_sentinel e1_recal
    echo "[phase=done]"
    ;;
e2_refit)
    # CONDITIONAL GPU-leg workload (plan v9 E2): v5 refit + lattice re-read.
    run_phase e2_refit
    write_e1_results_sentinel e2_refit
    echo "[phase=done]"
    ;;
__phase_key)
    # Test/debug probe: print the (recipe-keyed) done-file key for a phase
    # name and exit — pins the resume contract (tests/test_issue1336_recal.py).
    phase_key "${2:?usage: __phase_key <phase>}"
    exit 0
    ;;
*)
    echo "usage: bash scripts/issue1336_dispatch.sh all|g0_gate|gen|extract|fit|align|upload|d1_battery|d1_vmsteps|d2_probe|e1_recal|e2_refit|all_v2|c_stage|g0v2|g2_parity|gen_v2|gen_v2_nat|extract_v2|fit_v2|ladder|upload_v2|all_v3|c_pool [--smoke]" >&2
    exit 2
    ;;
esac
