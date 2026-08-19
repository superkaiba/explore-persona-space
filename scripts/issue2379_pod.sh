#!/usr/bin/env bash
# Issue #2379 — pod-side phase dispatcher (plan §4.3 last bullet + §9).
#
# Runs on a 4x H100 RunPod pod (backend: runpod). Phases are selected
# EXTERNALLY via --phases so the Gate-G1 hold between P3 and P4 is driven by
# the VM orchestrator (unit-4's issue2379_analysis.py --gate-caps), never
# pod-side. POD-SIDE CODE NEVER SHELLS OUT TO scripts/task.py — the only
# signalling channel is the per-phase JSON sentinel the VM poller observes.
#
# Phase map (p<k> -> what it does; the store-producing phases upload BEFORE p5):
#   p0  preflight (git HEAD, GPU inventory) + LMSYS gated-access probe
#         (issue2379_capture.py --probe-access; a denied grant fails P0 BEFORE
#         any training spend) + mapfit --phase pilot (measures the per-fit
#         wall = the P5 fence basis; downloads pass-B)
#   p1  prep + train : issue2379_prep_data.py --upload (fresh clones have no
#         JSONLs/banks — prep MUST precede training; HF persistence ON), then
#         issue2379_train.py --train-dir/--output-root threaded explicitly
#         (the script self-fans-out over the 8 adapters: 5 em + 3 caps).
#   p2  gen    : issue2379_sweep.py per model via the work-conserving GPU job
#         queue (--setting em over base+5em ; --setting caps over base+3caps;
#         --adapter passed so the CVD-pinned worker merges lazily AFTER its
#         own idempotency skip check). The sweep self-uploads
#         rates_caps/<model>.json + raw_completions/.
#   p3  capture-context : issue2379_capture.py --phase {grid,mu,text_baselines}
#         per condition model (8), job queue. Uploads predictor_captures
#         (grid,mu) + text_baselines. Enough to compute Gate-G1 (caps Train
#         Ref rho >= 0.4 at L27) from grid v_C + mu mu_train + p2 caps rates.
#   --- Gate-G1 hold (EXTERNAL, between p3 and p4) ---
#   p4  capture-answer : issue2379_capture.py --phase {ceiling,map_corpus}
#         per condition model (8), job queue. The expensive answer-side
#         captures, run only AFTER the gate passes. Uploads ceiling + map_corpus.
#   p5  mapfit : issue2379_mapfit.py --phase all (fits on the CPU pool +
#         scores on --device cuda + upload pinned components + predictor JSONs).
#         Reads p3/p4 stores local-first; both uploaded before this phase.
#
# Idempotency (round-2 review): a phase whose sentinel reads rc=0 with no
# failed models is SKIPPED (--force re-runs it; the stale sentinel is removed
# before any actual re-run so the poller never reads a stale success). The
# python entrypoints add their own output-exists skips, so a partial-failure
# phase re-run recomputes only the failed models (a rerun still pays one
# ~10-min merge per already-complete CAPTURE model — bounded, accepted).
#
# GPU scheduling (round-2 review): one persistent work-conserving job queue —
# each freed GPU is refilled immediately (no full-wave drain), and ALL merges
# run INSIDE the CVD-pinned worker (no shared MERGE_GPU pin racing slot-0).
#
# Typical driver sequence (VM-side, external):
#   bash scripts/issue2379_pod.sh --phases "p0 p1 p2 p3"
#   <VM computes Gate-G1; if PASS:>
#   bash scripts/issue2379_pod.sh --phases "p4 p5"
#
# --dry-run prints the planned commands per phase (argv the experimenter can
# eyeball) without executing — the VM-side smoke for this dispatcher.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT="${EPM_REPO_ROOT:-/workspace/explore-persona-space}"
NUM_GPUS="${EPM_NUM_GPUS:-4}"
LOG_DIR="${EPM_LOG_DIR:-/workspace/logs}"
OUT_DIR="${EPM_OUT_DIR:-eval_results/issue_2379}"
SENTINEL_PREFIX="issue-2379"
DRY_RUN=0
FORCE=0
PHASES=""

# Data-side roots (single source of truth for the prep -> train -> merge
# path contract; round-2 blocker: fresh-pod-adapter-contract). Safetensors
# (adapters, merged dirs) live under data/, never eval_results/.
DATA_DIR="data/issue_2379"
TRAIN_DIR="$DATA_DIR/train"
BANKS_DIR="$DATA_DIR/banks"
ADAPTER_ROOT="$DATA_DIR/adapters"
MERGED_ROOT="$DATA_DIR/merged"

# Training stems (== adapter suffix issue2379_reelicit_<stem>; unit-1 contract).
EM_STEMS=(em_bad_medical_advice em_bad_legal_advice em_bad_security_advice \
          em_turner_risky_financial em_turner_extreme_sports)
CAPS_STEMS=(caps_french caps_german caps_spanish)

usage() {
    cat <<'USAGE'
usage: issue2379_pod.sh --phases "p0 p1 p2 p3" [--dry-run] [--force]
  --phases   space-separated subset of: p0 p1 p2 p3 p4 p5 (run in the given order)
  --dry-run  print planned commands per phase, execute nothing
  --force    re-run phases whose success sentinel already exists (and pass
             --force through to the sweep/capture entrypoints)
Env overrides: EPM_REPO_ROOT EPM_NUM_GPUS EPM_LOG_DIR EPM_OUT_DIR
USAGE
}

# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --phases)
            [ $# -ge 2 ] || { echo "ERROR: --phases needs an argument" >&2; usage; exit 2; }
            PHASES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [ -z "$PHASES" ]; then
    echo "ERROR: --phases is required" >&2
    usage
    exit 2
fi

KNOWN_PHASES=" p0 p1 p2 p3 p4 p5 "
for ph in $PHASES; do
    case "$KNOWN_PHASES" in
        *" $ph "*) : ;;
        *) echo "ERROR: unknown phase '$ph' (known: p0 p1 p2 p3 p4 p5)" >&2; exit 2 ;;
    esac
done

FORCE_FLAG=""
[ "$FORCE" -eq 1 ] && FORCE_FLAG="--force"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Skip pod-side filesystem/env setup entirely under --dry-run (the VM smoke
# runs off-pod, where $REPO_ROOT does not exist; each phase's dry-run branch
# only echoes its planned argv and needs no cwd/env/model resolution).
if [ "$DRY_RUN" -eq 0 ]; then
    cd "$REPO_ROOT"
    mkdir -p "$LOG_DIR"
    # Load credentials for HF/WandB/Anthropic (bash launcher; `.` works, unlike
    # the sh SSH-MCP shell). Guarded: a missing .env is not fatal here — the
    # python entrypoints call orchestrate.env.load_dotenv in-process too.
    set -a
    # shellcheck disable=SC1091
    [ -f .env ] && . ./.env
    set +a
    # Queue width == visible GPU count (never a stale default on a resized pod).
    if [ -z "${EPM_NUM_GPUS:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
        # SLURM_GPU_WIDTH_EXEMPT: RunPod-only pod dispatcher (plan pins backend: runpod; never dispatched onto a SLURM lane)
        NUM_GPUS="$(nvidia-smi -L | wc -l)"
    fi
fi

# Resolve the base model id once (from the unit-2 sweep module; never hardcoded).
resolve_base_model_id() {
    uv run python -c 'import sys; sys.path.insert(0, "scripts"); from issue2379_sweep import BASE_MODEL; print(BASE_MODEL)'
}
BASE_MODEL_ID=""

# ---------------------------------------------------------------------------
# Sentinel writer + reader (JSON; python one-liners, no heredoc — argv-safe)
# ---------------------------------------------------------------------------
# status file rows: "<name> <rc>" appended by each fan-out job.
write_sentinel() {
    local phase="$1" rc="$2" started="$3" status_file="$4"
    local sentinel="$LOG_DIR/${SENTINEL_PREFIX}-${phase}.done.json"
    local sf="${status_file:-/dev/null}"
    uv run python -c '
import json, os, sys, time
phase, rc, started, sf, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
models, failed = [], []
if sf != "/dev/null" and os.path.exists(sf):
    for line in open(sf, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        name, _, jrc = line.rpartition(" ")
        models.append(name)
        if jrc != "0":
            failed.append(name)
doc = {"issue": 2379, "phase": phase, "rc": rc,
       "started_utc": started, "finished_utc": time.time(),
       "models": models, "failed_models": failed}
os.makedirs(os.path.dirname(out), exist_ok=True)
tmp = out + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(doc, indent=2))
os.replace(tmp, out)
print(f"[sentinel] wrote {out} rc={rc} failed={failed}")
' "$phase" "$rc" "$started" "$sf" "$sentinel"
}

# Sentinel read-back for the idempotency skip. The VM poller renames drained
# sentinels to <name>.processed, so check BOTH forms (pod-side-reporting.md
# requirement 3: own-sentinel reads are bare-then-.processed).
phase_complete() {  # $1 = phase; rc 0 iff a success sentinel (rc=0, no failed models) exists
    local base="$LOG_DIR/${SENTINEL_PREFIX}-$1.done.json"
    local f=""
    if [ -f "$base" ]; then f="$base"
    elif [ -f "${base}.processed" ]; then f="${base}.processed"
    else return 1
    fi
    uv run python -c '
import json, sys
try:
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
sys.exit(0 if doc.get("rc") == 0 and not doc.get("failed_models") else 1)
' "$f"
}

clear_stale_sentinel() {  # $1 = phase — remove before an actual (re-)run
    rm -f "$LOG_DIR/${SENTINEL_PREFIX}-$1.done.json" \
          "$LOG_DIR/${SENTINEL_PREFIX}-$1.done.json.processed"
}

# ---------------------------------------------------------------------------
# Merge lifecycle — ALWAYS inside the CVD-pinned worker (round-2 blocker
# gpu-merge-collision: no shared MERGE_GPU pin racing a live slot-0 job).
# ---------------------------------------------------------------------------
merge_here() {  # $1 = phase label ; $2 = stem — echoes the merged path
    # Runs INSIDE a worker whose env carries the single-GPU CVD pin, so
    # merge_lora(gpu_id=0) resolves to the worker's own GPU (_apply_cvd_pin
    # makes the inherited single-GPU pin authoritative).
    local plabel="$1" stem="$2"
    local adapter="$ADAPTER_ROOT/issue2379_reelicit_${stem}"
    local dest="$MERGED_ROOT/${plabel}_${stem}"
    if [ ! -d "$adapter" ]; then
        echo "[merge] FATAL: adapter missing: $adapter (p1 prep+train not run?)" >&2
        return 1
    fi
    rm -rf "$dest"   # stale partial merge from a crashed prior job
    mkdir -p "$MERGED_ROOT"
    uv run python -c '
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.train.sft import merge_lora
from issue2379_sweep import BASE_MODEL
merge_lora(BASE_MODEL, sys.argv[1], sys.argv[2], gpu_id=0)
' "$adapter" "$dest" >&2 || return 1
    echo "$dest"
}

# ---------------------------------------------------------------------------
# Work-conserving GPU job queue (round-2 review: no full-wave drain — each
# freed GPU is refilled immediately while jobs remain).
# ---------------------------------------------------------------------------
# Job spec grammar ("|"-separated; first field = kind):
#   sweep|<setting>|<model>                    (model == stem or "base")
#   capture|<plabel>|<setting>|<stem>|<cp...>  (<cp...> space-separated phases)
_job_exec() {  # runs inside a CVD-pinned background subshell
    local spec="$1"
    local kind="${spec%%|*}" rest="${spec#*|}"
    case "$kind" in
        sweep)
            local setting="${rest%%|*}" model="${rest#*|}"
            local rc=0
            if [ "$model" = "base" ]; then
                # shellcheck disable=SC2086  # FORCE_FLAG deliberately unquoted (empty or --force)
                uv run python scripts/issue2379_sweep.py \
                    --setting "$setting" --model "$BASE_MODEL_ID" --model-name base \
                    --banks-dir "$BANKS_DIR" --out-dir "$OUT_DIR" --gpu-id 0 \
                    $FORCE_FLAG || rc=$?
            else
                # --adapter: the sweep merges lazily AFTER its own idempotency
                # skip (a complete model re-run never pays the merge).
                # shellcheck disable=SC2086
                uv run python scripts/issue2379_sweep.py \
                    --setting "$setting" --adapter "$ADAPTER_ROOT/issue2379_reelicit_${model}" \
                    --model-name "$model" --banks-dir "$BANKS_DIR" --out-dir "$OUT_DIR" \
                    --gpu-id 0 $FORCE_FLAG || rc=$?
            fi
            return "$rc"
            ;;
        capture)
            local plabel setting stem cps
            IFS='|' read -r plabel setting stem cps <<< "$rest"
            # One merge per model-job, shared by its sequential capture phases.
            local model_path
            model_path="$(merge_here "$plabel" "$stem")" || return 1
            local rc=0 cp
            for cp in $cps; do
                local -a extra=()
                [ "$cp" = "mu" ] && extra+=(--train-jsonl "$TRAIN_DIR/${stem}.jsonl")
                # shellcheck disable=SC2086
                uv run python scripts/issue2379_capture.py \
                    --phase "$cp" --setting "$setting" --model "$model_path" \
                    --model-name "$stem" --banks-dir "$BANKS_DIR" --out-dir "$OUT_DIR" \
                    --gpu-id 0 $FORCE_FLAG "${extra[@]}" || { rc=$?; break; }
            done
            rm -rf "$model_path"
            echo "[merge] dropped $model_path" >&2
            return "$rc"
            ;;
        *)
            echo "[queue] FATAL: unknown job spec kind '$kind'" >&2
            return 2
            ;;
    esac
}

run_job_queue() {  # $1 = status_file ; $2.. = job specs
    local status_file="$1"; shift
    local -a queue=("$@")
    local -A slot_pid=() slot_label=()
    local any_fail=0 next=0 g pid rc
    local total="${#queue[@]}"
    while :; do
        # Fill every idle GPU while jobs remain (work-conserving).
        for ((g = 0; g < NUM_GPUS; g++)); do
            [ -n "${slot_pid[$g]:-}" ] && continue
            [ "$next" -lt "$total" ] || continue
            local spec="${queue[$next]}"
            next=$((next + 1))
            CUDA_VISIBLE_DEVICES="$g" _job_exec "$spec" &
            slot_pid[$g]=$!
            slot_label[$g]="$(_job_label "$spec")"
            echo "[queue] gpu=$g start ${slot_label[$g]} (job $next/$total)"
        done
        [ "${#slot_pid[@]}" -eq 0 ] && break
        # Reap finished jobs (bash reaps exited bg children internally, so
        # kill -0 reads dead the moment a job exits; wait then returns its rc).
        local reaped=0
        for g in "${!slot_pid[@]}"; do
            pid="${slot_pid[$g]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                rc=0
                wait "$pid" || rc=$?
                echo "${slot_label[$g]} ${rc}" >> "$status_file"
                if [ "$rc" -ne 0 ]; then
                    any_fail=1
                    echo "[queue] FAILED ${slot_label[$g]} rc=${rc} (gpu=$g)" >&2
                else
                    echo "[queue] gpu=$g done ${slot_label[$g]}"
                fi
                unset "slot_pid[$g]" "slot_label[$g]"
                reaped=1
            fi
        done
        [ "$reaped" -eq 0 ] && sleep 5
    done
    return "$any_fail"
}

_job_label() {  # $1 = spec -> short status-row label
    local spec="$1"
    local kind="${spec%%|*}" rest="${spec#*|}"
    case "$kind" in
        sweep)   echo "${rest%%|*}:${rest#*|}" ;;                 # setting:model
        capture) local plabel setting stem cps
                 IFS='|' read -r plabel setting stem cps <<< "$rest"
                 echo "$stem" ;;
        *)       echo "$spec" ;;
    esac
}

# ---------------------------------------------------------------------------
# Phase p0 — preflight + LMSYS gated-access probe + mapfit pilot (fence basis)
# ---------------------------------------------------------------------------
phase_p0() {
    local started; started="$(date +%s)"
    local status_file; status_file="$(mktemp)"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run p0] git rev-parse HEAD ; nvidia-smi -L"
        echo "[dry-run p0] uv run python scripts/issue2379_capture.py --probe-access"
        echo "[dry-run p0] uv run python scripts/issue2379_mapfit.py --phase pilot --workers \$(nproc) --pilot-layer 16"
        return 0
    fi
    echo "[p0] HEAD=$(git rev-parse HEAD)"
    nvidia-smi -L || { echo "[p0] nvidia-smi failed" >&2; write_sentinel p0 1 "$started" "$status_file"; return 1; }
    local any_fail=0 rc

    # LMSYS gated-access probe (round-2 concern i2379-lmsys-gated-read): a
    # denied grant fails P0 here, BEFORE any training/capture spend.
    rc=0
    uv run python scripts/issue2379_capture.py --probe-access || rc=$?
    echo "lmsys_probe ${rc}" >> "$status_file"
    [ "$rc" -ne 0 ] && { any_fail=1; echo "[p0] LMSYS gated-access probe FAILED rc=$rc" >&2; }

    rc=0
    OMP_NUM_THREADS="$(nproc)" uv run python scripts/issue2379_mapfit.py \
        --phase pilot --workers "$(nproc)" --pilot-layer 16 || rc=$?
    echo "pilot ${rc}" >> "$status_file"
    [ "$rc" -ne 0 ] && any_fail=1

    write_sentinel p0 "$any_fail" "$started" "$status_file"
    return "$any_fail"
}

# ---------------------------------------------------------------------------
# Phase p1 — prep + train (round-2 blocker fresh-pod-adapter-contract: a
# fresh clone has no training JSONLs/banks — prep runs FIRST, with --upload so
# mixes+banks persist to HF; train dirs threaded explicitly so the adapters
# land where merge_here reads them)
# ---------------------------------------------------------------------------
phase_p1() {
    local started; started="$(date +%s)"
    local status_file; status_file="$(mktemp)"
    local gpus_csv
    gpus_csv="$(seq -s, 0 $((NUM_GPUS - 1)))"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run p1] uv run python scripts/issue2379_prep_data.py --out-dir $DATA_DIR --upload"
        echo "[dry-run p1] uv run python scripts/issue2379_train.py --train-dir $TRAIN_DIR --output-root $ADAPTER_ROOT --gpus ${gpus_csv}"
        return 0
    fi
    local rc=0
    # Prep is idempotent-ish and cheap next to training; a p1 re-run re-verifies
    # the byte/row pins fail-loud (acceptable — the sentinel skip covers the
    # fully-complete case).
    uv run python scripts/issue2379_prep_data.py --out-dir "$DATA_DIR" --upload || rc=$?
    echo "prep ${rc}" >> "$status_file"
    if [ "$rc" -eq 0 ]; then
        uv run python scripts/issue2379_train.py \
            --train-dir "$TRAIN_DIR" --output-root "$ADAPTER_ROOT" --gpus "$gpus_csv" || rc=$?
        echo "train ${rc}" >> "$status_file"
    fi
    write_sentinel p1 "$rc" "$started" "$status_file"
    return "$rc"
}

# ---------------------------------------------------------------------------
# Phase p2 — gen (em + caps sweeps) via the job queue
# ---------------------------------------------------------------------------
phase_p2() {
    local started; started="$(date +%s)"
    local status_file; status_file="$(mktemp)"
    local -a jobs=("sweep|em|base")
    local s
    for s in "${EM_STEMS[@]}"; do jobs+=("sweep|em|$s"); done
    jobs+=("sweep|caps|base")
    for s in "${CAPS_STEMS[@]}"; do jobs+=("sweep|caps|$s"); done

    if [ "$DRY_RUN" -eq 1 ]; then
        local j
        for j in "${jobs[@]}"; do
            local rest="${j#*|}"
            echo "[dry-run p2] CUDA_VISIBLE_DEVICES=<queue-gpu> uv run python scripts/issue2379_sweep.py --setting ${rest%%|*} --{model base|adapter issue2379_reelicit_${rest#*|}} --model-name ${rest#*|} --banks-dir $BANKS_DIR --out-dir $OUT_DIR --gpu-id 0 $FORCE_FLAG"
        done
        return 0
    fi

    [ -n "$BASE_MODEL_ID" ] || BASE_MODEL_ID="$(resolve_base_model_id)"
    rm -rf "$MERGED_ROOT"   # stale residue from a crashed prior phase run
    local any_fail=0
    run_job_queue "$status_file" "${jobs[@]}" || any_fail=1
    write_sentinel p2 "$any_fail" "$started" "$status_file"
    return "$any_fail"
}

# ---------------------------------------------------------------------------
# Capture fan-out (shared by p3 context-side and p4 answer-side)
# ---------------------------------------------------------------------------
# capture relies on the launcher CVD pin (a --gpu-id 0 rides along; nonzero
# values are refused by the entrypoint — gotchas.md launcher-env CVD).
# $1 = phase-label (p3|p4) ; $2 = space-separated capture-phase list
_run_capture_phase() {
    local plabel="$1" cap_phases="$2"
    local started; started="$(date +%s)"
    local status_file; status_file="$(mktemp)"
    # condition models only (5 em + 3 caps); base has no captures.
    local -a models=("${EM_STEMS[@]}" "${CAPS_STEMS[@]}")
    local -a jobs=()
    local m setting
    for m in "${models[@]}"; do
        case "$m" in caps_*) setting=caps ;; *) setting=em ;; esac
        jobs+=("capture|$plabel|$setting|$m|$cap_phases")
    done

    if [ "$DRY_RUN" -eq 1 ]; then
        local cp
        for m in "${models[@]}"; do
            case "$m" in caps_*) setting=caps ;; *) setting=em ;; esac
            for cp in $cap_phases; do
                local extra=""
                [ "$cp" = "mu" ] && extra=" --train-jsonl $TRAIN_DIR/${m}.jsonl"
                echo "[dry-run $plabel] CUDA_VISIBLE_DEVICES=<queue-gpu> uv run python scripts/issue2379_capture.py --phase $cp --setting $setting --model <merged:$m> --model-name $m --banks-dir $BANKS_DIR --out-dir $OUT_DIR --gpu-id 0 $FORCE_FLAG$extra"
            done
        done
        return 0
    fi

    [ -n "$BASE_MODEL_ID" ] || BASE_MODEL_ID="$(resolve_base_model_id)"
    rm -rf "$MERGED_ROOT"   # stale residue from a crashed prior phase run
    local any_fail=0
    run_job_queue "$status_file" "${jobs[@]}" || any_fail=1
    write_sentinel "$plabel" "$any_fail" "$started" "$status_file"
    return "$any_fail"
}

phase_p3() { _run_capture_phase p3 "grid mu text_baselines"; }
phase_p4() { _run_capture_phase p4 "ceiling map_corpus"; }

# ---------------------------------------------------------------------------
# Phase p5 — mapfit (fits on CPU pool + scores on cuda + upload)
# ---------------------------------------------------------------------------
phase_p5() {
    local started; started="$(date +%s)"
    local status_file; status_file="$(mktemp)"
    local models_csv
    models_csv="$(IFS=,; echo "${EM_STEMS[*]},${CAPS_STEMS[*]}")"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run p5] OMP_NUM_THREADS=1 uv run python scripts/issue2379_mapfit.py --phase all --models ${models_csv} --workers \$(nproc) --device cuda --out-dir $OUT_DIR"
        return 0
    fi
    local rc=0
    uv run python scripts/issue2379_mapfit.py --phase all \
        --models "$models_csv" --workers "$(nproc)" --device cuda --out-dir "$OUT_DIR" || rc=$?
    echo "mapfit ${rc}" >> "$status_file"
    write_sentinel p5 "$rc" "$started" "$status_file"
    return "$rc"
}

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
echo "[dispatch] issue-2379 phases='$PHASES' num_gpus=$NUM_GPUS dry_run=$DRY_RUN force=$FORCE"
for ph in $PHASES; do
    if [ "$DRY_RUN" -eq 0 ] && [ "$FORCE" -eq 0 ] && phase_complete "$ph"; then
        echo "[dispatch] === phase $ph already complete (sentinel rc=0, no failed models) — SKIPPING (--force to redo) ==="
        continue
    fi
    if [ "$DRY_RUN" -eq 0 ]; then
        clear_stale_sentinel "$ph"
    fi
    echo "[dispatch] === phase $ph ==="
    rc=0
    case "$ph" in
        p0) phase_p0 || rc=$? ;;
        p1) phase_p1 || rc=$? ;;
        p2) phase_p2 || rc=$? ;;
        p3) phase_p3 || rc=$? ;;
        p4) phase_p4 || rc=$? ;;
        p5) phase_p5 || rc=$? ;;
    esac
    if [ "$rc" -ne 0 ]; then
        echo "[dispatch] phase $ph FAILED rc=$rc — halting (sentinel names the phase)" >&2
        exit "$rc"
    fi
    echo "[dispatch] phase $ph OK"
done
echo "[dispatch] all requested phases complete: $PHASES"
