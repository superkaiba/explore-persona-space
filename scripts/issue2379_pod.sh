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
#   p0  preflight (git HEAD, GPU inventory) + mapfit --phase pilot
#         (measures the per-fit wall = the P5 fence basis; downloads pass-B)
#   p1  train  : issue2379_train.py --gpus 0,1,2,3 (script self-fans-out over
#         the 8 adapters: 5 em + 3 caps; NO pod.sh CVD wave here)
#   p2  gen    : issue2379_sweep.py per model, 4-way CVD-pinned fan-out
#         (--setting em over base+5em ; --setting caps over base+3caps).
#         The sweep self-uploads rates_caps/<model>.json + raw_completions/.
#   p3  capture-context : issue2379_capture.py --phase {grid,mu,text_baselines}
#         per condition model (8), 4-way CVD-pinned. Uploads predictor_captures
#         (grid,mu) + text_baselines. Enough to compute Gate-G1 (caps Train
#         Ref rho >= 0.4 at L27) from grid v_C + mu mu_train + p2 caps rates.
#   --- Gate-G1 hold (EXTERNAL, between p3 and p4) ---
#   p4  capture-answer : issue2379_capture.py --phase {ceiling,map_corpus}
#         per condition model (8), 4-way CVD-pinned. The expensive answer-side
#         captures, run only AFTER the gate passes. Uploads ceiling + map_corpus.
#   p5  mapfit : issue2379_mapfit.py --phase all (fits on the CPU pool +
#         scores on --device cuda + upload pinned components + predictor JSONs).
#         Reads p3/p4 stores local-first; both uploaded before this phase.
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
PHASES=""

# Training stems (== adapter suffix issue2379_reelicit_<stem>; unit-1 contract).
EM_STEMS=(em_bad_medical_advice em_bad_legal_advice em_bad_security_advice \
          em_turner_risky_financial em_turner_extreme_sports)
CAPS_STEMS=(caps_french caps_german caps_spanish)

usage() {
    cat <<'USAGE'
usage: issue2379_pod.sh --phases "p0 p1 p2 p3" [--dry-run]
  --phases   space-separated subset of: p0 p1 p2 p3 p4 p5 (run in the given order)
  --dry-run  print planned commands per phase, execute nothing
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
fi

# Resolve the base model id once (from the unit-2 sweep module; never hardcoded).
resolve_base_model_id() {
    uv run python -c 'import sys; sys.path.insert(0, "scripts"); from issue2379_sweep import BASE_MODEL; print(BASE_MODEL)'
}
BASE_MODEL_ID=""

# ---------------------------------------------------------------------------
# Sentinel writer (JSON; python one-liner, no heredoc — argv-safe)
# ---------------------------------------------------------------------------
# status file rows: "<model> <rc>" appended by each fan-out job.
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

# ---------------------------------------------------------------------------
# Merge lifecycle (merge once per model to a stable path; delete after consume)
# ---------------------------------------------------------------------------
# echoes the merged dir path (base arm: echoes the HF base id, no merge).
MERGED_ROOT="$OUT_DIR/merged"

merged_path_for() {  # $1 = model name (stem or "base")
    if [ "$1" = "base" ]; then
        echo "$BASE_MODEL_ID"
    else
        echo "$MERGED_ROOT/$1"
    fi
}

ensure_merged() {  # $1 = stem ; merges adapter issue2379_reelicit_<stem> -> MERGED_ROOT/<stem>
    local stem="$1"
    local adapter="$OUT_DIR/adapters/issue2379_reelicit_${stem}"
    local dest="$MERGED_ROOT/$stem"
    if [ -d "$dest" ]; then
        echo "[merge] $stem already merged at $dest" >&2
        return 0
    fi
    mkdir -p "$MERGED_ROOT"
    CUDA_VISIBLE_DEVICES="${MERGE_GPU:-0}" uv run python -c '
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.train.sft import merge_lora
from issue2379_sweep import BASE_MODEL
merge_lora(BASE_MODEL, sys.argv[1], sys.argv[2], gpu_id=0)
' "$adapter" "$dest" >&2
}

drop_merged() {  # $1 = stem (never base)
    [ "$1" = "base" ] && return 0
    rm -rf "${MERGED_ROOT:?}/$1"
    echo "[merge] dropped $MERGED_ROOT/$1" >&2
}

# ---------------------------------------------------------------------------
# CVD-pinned single-GPU launcher (the physical GPU is set in the LAUNCHER env;
# a --gpu-id 0 rides along where the entrypoint accepts it — gotchas.md CVD).
# ---------------------------------------------------------------------------
launch_pinned() {  # $1 = gpu ; $2.. = command (a --gpu-id 0 is appended by the caller when accepted)
    local gpu="$1"; shift
    CUDA_VISIBLE_DEVICES="$gpu" "$@" &
}

# Wait for a set of pids, appending "<model> <rc>" per job to the status file.
# Returns non-zero if any job failed.
declare -a WAVE_PIDS=()
declare -a WAVE_NAMES=()
wait_wave() {  # $1 = status_file
    local status_file="$1" any_fail=0 i pid rc
    for i in "${!WAVE_PIDS[@]}"; do
        pid="${WAVE_PIDS[$i]}"
        rc=0
        wait "$pid" || rc=$?
        echo "${WAVE_NAMES[$i]} ${rc}" >> "$status_file"
        if [ "$rc" -ne 0 ]; then
            any_fail=1
            echo "[wave] FAILED ${WAVE_NAMES[$i]} rc=${rc}" >&2
        fi
    done
    WAVE_PIDS=()
    WAVE_NAMES=()
    return "$any_fail"
}

# ---------------------------------------------------------------------------
# Phase p0 — preflight + mapfit pilot (fence basis)
# ---------------------------------------------------------------------------
phase_p0() {
    local started; started="$(date +%s)"
    local status_file; status_file="$(mktemp)"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run p0] git rev-parse HEAD ; nvidia-smi -L"
        echo "[dry-run p0] uv run python scripts/issue2379_mapfit.py --phase pilot --workers \$(nproc) --pilot-layer 16"
        return 0
    fi
    echo "[p0] HEAD=$(git rev-parse HEAD)"
    nvidia-smi -L || { echo "[p0] nvidia-smi failed" >&2; write_sentinel p0 1 "$started" "$status_file"; return 1; }
    local rc=0
    OMP_NUM_THREADS="$(nproc)" uv run python scripts/issue2379_mapfit.py \
        --phase pilot --workers "$(nproc)" --pilot-layer 16 || rc=$?
    echo "pilot ${rc}" >> "$status_file"
    write_sentinel p0 "$rc" "$started" "$status_file"
    return "$rc"
}

# ---------------------------------------------------------------------------
# Phase p1 — train (the train script self-fans-out over 0,1,2,3)
# ---------------------------------------------------------------------------
phase_p1() {
    local started; started="$(date +%s)"
    local status_file; status_file="$(mktemp)"
    local gpus_csv
    gpus_csv="$(seq -s, 0 $((NUM_GPUS - 1)))"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run p1] uv run python scripts/issue2379_train.py --gpus ${gpus_csv}"
        return 0
    fi
    local rc=0
    uv run python scripts/issue2379_train.py --gpus "$gpus_csv" || rc=$?
    echo "train ${rc}" >> "$status_file"
    write_sentinel p1 "$rc" "$started" "$status_file"
    return "$rc"
}

# ---------------------------------------------------------------------------
# Phase p2 — gen (em + caps sweeps), 4-way CVD-pinned fan-out
# ---------------------------------------------------------------------------
# Runs one (setting, model) job per invocation of _sweep_job.
_sweep_job() {  # $1 = gpu ; $2 = setting ; $3 = model_name ; $4 = merged_path
    launch_pinned "$1" uv run python scripts/issue2379_sweep.py \
        --setting "$2" --model "$4" --model-name "$3" --out-dir "$OUT_DIR" --gpu-id 0
}

phase_p2() {
    local started; started="$(date +%s)"
    local status_file; status_file="$(mktemp)"
    # (setting, model) pairs: base under both settings + condition models.
    local -a pairs=()
    pairs+=("em base")
    local s
    for s in "${EM_STEMS[@]}"; do pairs+=("em $s"); done
    pairs+=("caps base")
    for s in "${CAPS_STEMS[@]}"; do pairs+=("caps $s"); done

    if [ "$DRY_RUN" -eq 1 ]; then
        local p
        for p in "${pairs[@]}"; do
            set -- $p
            echo "[dry-run p2] CUDA_VISIBLE_DEVICES=<g> uv run python scripts/issue2379_sweep.py --setting $1 --model <merged:$2> --model-name $2 --out-dir $OUT_DIR --gpu-id 0"
        done
        return 0
    fi

    [ -n "$BASE_MODEL_ID" ] || BASE_MODEL_ID="$(resolve_base_model_id)"
    local any_fail=0 slot=0 p setting model merged
    for p in "${pairs[@]}"; do
        set -- $p
        setting="$1"; model="$2"
        if [ "$model" != "base" ]; then ensure_merged "$model"; fi
        merged="$(merged_path_for "$model")"
        _sweep_job "$slot" "$setting" "$model" "$merged"
        WAVE_PIDS+=("$!")
        WAVE_NAMES+=("${setting}:${model}")
        slot=$((slot + 1))
        if [ "$slot" -ge "$NUM_GPUS" ]; then
            wait_wave "$status_file" || any_fail=1
            slot=0
        fi
    done
    [ "$slot" -gt 0 ] && { wait_wave "$status_file" || any_fail=1; }
    # p2 only ran the sweep; captures (p3/p4) re-merge as needed. Drop all merges.
    for model in "${EM_STEMS[@]}" "${CAPS_STEMS[@]}"; do drop_merged "$model"; done
    write_sentinel p2 "$any_fail" "$started" "$status_file"
    return "$any_fail"
}

# ---------------------------------------------------------------------------
# Capture fan-out (shared by p3 context-side and p4 answer-side)
# ---------------------------------------------------------------------------
# capture relies on the launcher CVD pin (create_vllm_engine has no gpu_id);
# no --gpu-id is appended (its acceptance in issue2379_capture.py is unverified;
# CVD alone is authoritative — gotchas.md launcher-env CVD).
_capture_job() {  # $1 = gpu ; $2 = setting ; $3 = model_name ; $4 = merged_path ; $5 = capture_phase
    launch_pinned "$1" uv run python scripts/issue2379_capture.py \
        --phase "$5" --setting "$2" --model "$4" --model-name "$3" --out-dir "$OUT_DIR"
}

# $1 = phase-label (p3|p4) ; $2 = space-separated capture-phase list
_run_capture_phase() {
    local plabel="$1" cap_phases="$2"
    local started; started="$(date +%s)"
    local status_file; status_file="$(mktemp)"
    # condition models only (5 em + 3 caps); base has no captures.
    local -a models=("${EM_STEMS[@]}" "${CAPS_STEMS[@]}")

    if [ "$DRY_RUN" -eq 1 ]; then
        local m cp setting
        for m in "${models[@]}"; do
            case "$m" in caps_*) setting=caps ;; *) setting=em ;; esac
            for cp in $cap_phases; do
                echo "[dry-run $plabel] CUDA_VISIBLE_DEVICES=<g> uv run python scripts/issue2379_capture.py --phase $cp --setting $setting --model <merged:$m> --model-name $m --out-dir $OUT_DIR"
            done
        done
        return 0
    fi

    [ -n "$BASE_MODEL_ID" ] || BASE_MODEL_ID="$(resolve_base_model_id)"
    local any_fail=0 slot=0 m setting merged cp
    for m in "${models[@]}"; do
        case "$m" in caps_*) setting=caps ;; *) setting=em ;; esac
        ensure_merged "$m"
        merged="$(merged_path_for "$m")"
        # run each capture-phase for this model sequentially on one pinned GPU,
        # via a subshell so the wave slot tracks the whole model's captures.
        (
            for cp in $cap_phases; do
                CUDA_VISIBLE_DEVICES="$slot" uv run python scripts/issue2379_capture.py \
                    --phase "$cp" --setting "$setting" --model "$merged" \
                    --model-name "$m" --out-dir "$OUT_DIR"
            done
        ) &
        WAVE_PIDS+=("$!")
        WAVE_NAMES+=("$m")
        slot=$((slot + 1))
        if [ "$slot" -ge "$NUM_GPUS" ]; then
            wait_wave "$status_file" || any_fail=1
            slot=0
        fi
    done
    [ "$slot" -gt 0 ] && { wait_wave "$status_file" || any_fail=1; }
    for m in "${models[@]}"; do drop_merged "$m"; done
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
echo "[dispatch] issue-2379 phases='$PHASES' num_gpus=$NUM_GPUS dry_run=$DRY_RUN"
for ph in $PHASES; do
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
