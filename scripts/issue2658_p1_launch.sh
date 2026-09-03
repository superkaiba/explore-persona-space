#!/usr/bin/env bash
# Issue #2658 P1 launcher — pilot generation (6,290 frozen requests) + L19 capture.
#
# Plan v4 §10 phase P1: "6,600-response pilot generation plus L19 capture";
# realized request count after the group-D re-freeze is 6,290 (629 frozen pilot
# prompt pins x 10 responses/prompt, order-manifest sha bf3cabe59651e8ce,
# 131 non-empty cells of 132 registered). Binding budget is GPU-HOURS (<= 8),
# not width: the launcher derives shard width from the REALIZED allocation and
# never assumes 8.
#
# Dispatch contract (fixed by the scripts, issue2658_generate.py docstring):
# "--num-shards/--shard-index partitions the sorted cell list; the dispatcher
# pins one GPU per shard via launcher-env CUDA_VISIBLE_DEVICES."  N data-
# parallel shards at --tensor-parallel 1 (default).  Neither script takes
# --gpu-id / +gpu_id, so the train/sft.py in-process CVD clobber (gotchas.md
# #545/#543) has no seam here; the launcher-env pin is authoritative.
#
# Width derivation (allocation-first — gotchas.md #1902/#1336; never the bare
# physical count on a SLURM job):
#   1. EPS_P1_NUM_SHARDS (explicit override; may only NARROW the allocation)
#   2. CUDA_VISIBLE_DEVICES (parsed as the PHYSICAL id array; shard s pins
#      ALLOC[s] — never the literal index s)
#   3. SLURM_STEP_GPUS / SLURM_JOB_GPUS / SLURM_GPUS_ON_NODE (SLURM jobs with
#      no CVD; none present => FAIL LOUD)
#   4. nvidia-smi -L count (non-SLURM exclusive hosts: RunPod pod / GCE)
#
# Sequence: [gen wave: N shards] -> [generation completeness gate against the
# per-shard frozen gen_order_manifest cell lists] -> [ONE raw-completions HF
# upload, production only] -> [GPU-memory reclaim gate: reap this run's
# orphaned vLLM workers, verify the allocation's cards actually freed] ->
# [capture wave: N shards, --upload per shard] ->
# [phase=done].  Per-shard rcs are collected individually (wait "$pid"); any
# non-zero rc fails the launcher loud, naming every failed shard — a bare
# `wait` would swallow child rcs and ship a silently short manifest.
#
# Resume safety: both scripts are fingerprint-gated per cell/store-shard
# (issue2658_generate.load_resume_cell keys on generating-parameter
# fingerprint + record count; issue2658_capture.resume_completed_shards keys
# on fingerprint + row keys + per-row answer_sha256), and the order manifest
# is write-once (byte-identical rewrite OK, drift raises). A launcher re-run
# resumes completed cells and never duplicates work.
#
# Upload sequencing: generate's own --upload scans the WHOLE out-root
# (upload_raw_completions_to_data_repo), so N concurrent per-shard uploads
# would race overlapping bulk commits; the launcher instead runs ONE post-gate
# upload via the script's own upload_raw() (same helper, same zero-match
# guard). Capture shards upload their OWN disjoint store dirs, so --upload
# rides each capture leg directly.
#
# Smoke blind-spot enumeration (.claude/rules/smoke-blind-spots.md):
#   - HF upload legs are NOT exercised under smoke (production-only by
#     directive: no smoke uploads) — the gen upload leg and capture
#     upload_store + verify_repo_paths_uploaded first run in production.
#   - The completeness gate covers only the N smoke cells under smoke (the
#     production gate covers all 131 realized cells); capture's own
#     frame-manifest anchor is likewise smoke-restricted (it discloses this).
#   - Smoke runs --responses EPS_P1_SMOKE_RESPONSES (default 2) vs the frozen
#     10/prompt — a SCALE cut, same code path; smoke fingerprints and the
#     smoke_gen/ out-root can never satisfy production resume.
#   - The cap-amendment branch (>2% length-cap hits per cell) is
#     data-dependent and unlikely to fire at smoke scale.
#   - The GPU-memory reclaim gate IS exercised under smoke (the smoke generate
#     wave constructs real vLLM engines, so reap + nvidia-smi verify run the
#     identical code path in both modes); what a smoke PASS does NOT certify
#     is production-SCALE margins — the full 10/prompt KV footprint and the
#     8-card teardown timing vs EPS_P1_RECLAIM_TIMEOUT_S (a scale cut, same
#     code path).
#   Everything else — width derivation, CVD pinning, rc collection, gate,
#   GPU-memory reclaim, capture chain — is the IDENTICAL launcher code path
#   in both modes.
#
# Usage:
#   bash scripts/issue2658_p1_launch.sh                    # production wave
#   bash scripts/issue2658_p1_launch.sh --smoke            # smoke only
#   bash scripts/issue2658_p1_launch.sh --smoke-then-full  # smoke gate, then production
#
# Env knobs: EPS_P1_NUM_SHARDS, EPS_P1_SPLIT (default pilot), EPS_P1_OUT_ROOT,
# EPS_P1_LOG_DIR, EPS_P1_SMOKE_CELLS (default = shard count),
# EPS_P1_SMOKE_RESPONSES (default 2).
# Reclaim-gate knobs: EPS_P1_RECLAIM_THRESHOLD_MIB (default 1024),
# EPS_P1_RECLAIM_TIMEOUT_S (default 120), EPS_P1_RECLAIM_POLL_S (default 5),
# EPS_P1_RECLAIM_KILL_GRACE_S (default 10).
# Exit codes: 2 usage/width derivation, 3 failed shard leg(s), 4 generation
# completeness gate, 5 GPU-memory reclaim gate (unreclaimed or unverifiable).
#
# Logs/breadcrumbs (absolute; stated in the P1 dispatch note for re-attach):
#   $LOG_DIR/launcher.pid
#   $LOG_DIR/<label>_shardNN.log + $LOG_DIR/<label>_shardNN.pid
#   (label in {generate, capture, generate_smoke, capture_smoke})

set -euo pipefail

command -v uv >/dev/null 2>&1 || export PATH="$PATH:$HOME/.local/bin"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pod-bound HF cache convention; SLURM/GCE lanes set their own HF_HOME.
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then
    export HF_HOME=/workspace/.cache/huggingface
fi

MODE=production
for arg in "$@"; do
    case "$arg" in
        --smoke) MODE=smoke ;;
        --smoke-then-full) MODE=smoke_then_full ;;
        *)
            echo "FATAL: unknown argument '$arg' (expected --smoke | --smoke-then-full)" >&2
            exit 2
            ;;
    esac
done

SPLIT="${EPS_P1_SPLIT:-pilot}"
OUT_ROOT="${EPS_P1_OUT_ROOT:-$REPO_ROOT/eval_results/issue_2658}"
LOG_DIR="${EPS_P1_LOG_DIR:-$REPO_ROOT/logs/issue_2658_p1}"
mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/launcher.pid"

# Reclaim-gate knobs + the per-invocation kill-scope tag. The tag is exported
# BEFORE any wave launches so every shard child — and every worker subprocess
# vLLM spawns from a shard — inherits it in its environment.
RECLAIM_THRESHOLD_MIB="${EPS_P1_RECLAIM_THRESHOLD_MIB:-1024}"
RECLAIM_TIMEOUT_S="${EPS_P1_RECLAIM_TIMEOUT_S:-120}"
RECLAIM_POLL_S="${EPS_P1_RECLAIM_POLL_S:-5}"
RECLAIM_KILL_GRACE_S="${EPS_P1_RECLAIM_KILL_GRACE_S:-10}"
EPS_P1_RUN_TAG="issue2658-p1-$$-$(date +%s)"
export EPS_P1_RUN_TAG

# ---------------------------------------------------------------------------
# GPU allocation -> ALLOC (physical id array) + NUM_SHARDS.
# ---------------------------------------------------------------------------
ALLOC=()
ALLOC_SOURCE=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra ALLOC <<< "$CUDA_VISIBLE_DEVICES"
    ALLOC_SOURCE="CUDA_VISIBLE_DEVICES"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    if [ -n "${SLURM_STEP_GPUS:-}" ]; then
        IFS=',' read -ra ALLOC <<< "$SLURM_STEP_GPUS"
        ALLOC_SOURCE="SLURM_STEP_GPUS"
    elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
        IFS=',' read -ra ALLOC <<< "$SLURM_JOB_GPUS"
        ALLOC_SOURCE="SLURM_JOB_GPUS"
    elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        for ((g = 0; g < SLURM_GPUS_ON_NODE; g++)); do ALLOC+=("$g"); done
        ALLOC_SOURCE="SLURM_GPUS_ON_NODE"
    else
        echo "FATAL: SLURM job $SLURM_JOB_ID exposes none of CUDA_VISIBLE_DEVICES /" \
            "SLURM_STEP_GPUS / SLURM_JOB_GPUS / SLURM_GPUS_ON_NODE — refusing the" \
            "physical-count fallback on a shared SLURM node (gotchas.md #1902)" >&2
        exit 2
    fi
else
    n_phys="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
    n_phys="${n_phys:-0}"
    if [ "$n_phys" -lt 1 ]; then
        echo "FATAL: no GPU allocation signal (no CVD, not a SLURM job, nvidia-smi" \
            "enumerates zero GPUs) — nothing to shard onto" >&2
        exit 2
    fi
    for ((g = 0; g < n_phys; g++)); do ALLOC+=("$g"); done
    ALLOC_SOURCE="nvidia-smi (exclusive host)"
fi

NUM_SHARDS="${EPS_P1_NUM_SHARDS:-${#ALLOC[@]}}"
if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [ "$NUM_SHARDS" -lt 1 ]; then
    echo "FATAL: EPS_P1_NUM_SHARDS='$NUM_SHARDS' is not a positive integer" >&2
    exit 2
fi
if [ "$NUM_SHARDS" -gt "${#ALLOC[@]}" ]; then
    echo "FATAL: EPS_P1_NUM_SHARDS=$NUM_SHARDS exceeds the realized allocation of" \
        "${#ALLOC[@]} GPU(s) (${ALLOC[*]}) — cannot pin more shards than GPUs" >&2
    exit 2
fi

SMOKE_CELLS="${EPS_P1_SMOKE_CELLS:-$NUM_SHARDS}"
SMOKE_RESPONSES="${EPS_P1_SMOKE_RESPONSES:-2}"
if [ "$SMOKE_CELLS" -lt "$NUM_SHARDS" ]; then
    echo "FATAL: EPS_P1_SMOKE_CELLS=$SMOKE_CELLS < NUM_SHARDS=$NUM_SHARDS — every" \
        "gen shard must receive >=1 cell (the scripts fail loud on a zero-cell shard)" >&2
    exit 2
fi

echo "[phase=p1_width] width=$NUM_SHARDS source=$ALLOC_SOURCE gpus=${ALLOC[*]:0:$NUM_SHARDS}" \
    "mode=$MODE split=$SPLIT out_root=$OUT_ROOT log_dir=$LOG_DIR"

# ---------------------------------------------------------------------------
# One wave of N concurrent single-GPU shards with per-shard rc collection.
#   run_wave <label> <script-stem> <arg...>
# ---------------------------------------------------------------------------
FAILED_FILE="$LOG_DIR/failed_legs.txt"
: > "$FAILED_FILE"

run_wave() {
    local label="$1" stem="$2"
    shift 2
    local extra=("$@")
    echo "[phase=${label}_wave] launching $NUM_SHARDS ${stem} shard(s)"
    local pids=()
    local s gpu log pidf pid
    for ((s = 0; s < NUM_SHARDS; s++)); do
        gpu="${ALLOC[s]}"
        log="$LOG_DIR/$(printf '%s_shard%02d.log' "$label" "$s")"
        pidf="$LOG_DIR/$(printf '%s_shard%02d.pid' "$label" "$s")"
        # Launcher-env CVD pin per shard — the contract the script docstrings fix.
        CUDA_VISIBLE_DEVICES="$gpu" uv run python "scripts/${stem}.py" \
            --split "$SPLIT" --out-root "$OUT_ROOT" \
            --num-shards "$NUM_SHARDS" --shard-index "$s" \
            "${extra[@]}" > "$log" 2>&1 &
        pid=$!
        printf '%s\n' "$pid" > "$pidf"
        pids+=("$pid:$s:$gpu:$log")
        echo "[p1] $label shard $s -> pid=$pid gpu=$gpu log=$log"
    done
    local any_fail=0 rc entry
    for entry in "${pids[@]}"; do
        IFS=':' read -r pid s gpu log <<< "$entry"
        rc=0
        wait "$pid" || rc=$?
        if [ "$rc" -ne 0 ]; then
            any_fail=1
            printf '%s shard %s (gpu %s) rc=%s log=%s\n' "$label" "$s" "$gpu" "$rc" "$log" \
                >> "$FAILED_FILE"
            echo "[p1] FAIL: $label shard $s (gpu $gpu) exited rc=$rc — log tail:" >&2
            tail -n 40 "$log" >&2 || true
        else
            echo "[p1] $label shard $s complete rc=0"
        fi
    done
    if [ "$any_fail" -ne 0 ]; then
        echo "FATAL: $(grep -c . "$FAILED_FILE") failed leg(s):" >&2
        cat "$FAILED_FILE" >&2
        exit 3
    fi
    echo "[phase=${label}_wave] all $NUM_SHARDS shard(s) rc=0"
}

# ---------------------------------------------------------------------------
# Generation completeness gate: every cell named by the per-shard frozen
# gen_order_manifest lists must have a non-empty raw_completions JSON + a
# non-empty gen_manifest JSONL, each shard a terminal gen_summary, and no cell
# may appear in two shards. Runs BEFORE capture so a missing shard is named
# here instead of surfacing as capture's CaptureSpanError key-diff. Stdlib-only
# heredoc (no repo imports -> no signature-drift seam; gotchas.md inline-stdin
# entry).
# ---------------------------------------------------------------------------
gate_generation_complete() {
    local groot="$1"
    echo "[phase=gen_gate] asserting generation completeness under $groot"
    uv run python - "$groot" "$SPLIT" "$NUM_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

groot, split, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
problems: list[str] = []
cells_seen: dict[str, str] = {}
for s in range(n):
    tag = f"shard{s:02d}of{n:02d}"
    man = groot / "gen_order_manifest" / f"{split}_{tag}.json"
    if not man.is_file():
        problems.append(f"order manifest absent: {man}")
        continue
    body = json.loads(man.read_text())
    for cell in body["cell_order"]:
        if cell in cells_seen:
            problems.append(f"cell {cell} listed by both {cells_seen[cell]} and {tag}")
        cells_seen[cell] = tag
        for p in (
            groot / "raw_completions" / split / f"{cell}.json",
            groot / "gen_manifest" / split / f"{cell}.jsonl",
        ):
            if not p.is_file() or p.stat().st_size == 0:
                problems.append(f"missing/empty: {p}")
    summary = groot / "gen_summary" / f"{split}_{tag}.json"
    if not summary.is_file():
        problems.append(f"gen summary absent (shard never reached its terminal write): {summary}")
if not cells_seen:
    problems.append("zero cells enumerated across shard order manifests")
if problems:
    print(f"[p1] GENERATION INCOMPLETE ({len(problems)} problem(s)):", file=sys.stderr)
    for m in problems:
        print(f"  {m}", file=sys.stderr)
    sys.exit(4)
print(
    f"[p1] generation completeness gate PASS: {len(cells_seen)} cells "
    f"across {n} shard manifest(s)"
)
PY
    # Surface (never fail on) plan-§5 cap-amendment artifacts: >2% length-cap
    # hits per cell trigger a pre-test cap amendment, not regeneration.
    if compgen -G "$groot/gen_summary/cap_amendment_${SPLIT}_*.json" > /dev/null; then
        echo "[p1] NOTE: CAP AMENDMENT REQUIRED artifacts present (plan §5 pre-test gate):"
        compgen -G "$groot/gen_summary/cap_amendment_${SPLIT}_*.json"
    fi
}

# ---------------------------------------------------------------------------
# Single sequenced raw-completions upload (production only) via the generate
# script's own upload_raw() — same canonical helper + zero-match guard its
# --upload path runs; sequenced here because per-shard --upload would race N
# concurrent whole-out-root bulk commits.
# ---------------------------------------------------------------------------
upload_generation() {
    echo "[phase=upload_gen] uploading raw completions (single bulk commit)"
    uv run python - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue2658_generate as G  # noqa: E402

G.upload_raw(Path(sys.argv[1]), smoke=False)
PY
}

# ---------------------------------------------------------------------------
# Between-wave GPU-memory reclaim gate (exit 5). issue2658_generate.py ends
# with a deliberate os._exit(0) after flush (vLLM worker children survive
# interpreter finalization otherwise — gotchas.md vLLM worker-subprocess
# teardown), which ORPHANS the VLLM::EngineCore worker subprocesses; measured
# on pod-2658: 8 orphans holding 74,592 MiB EACH after every gen shard parent
# exited rc=0, and the teacher-forced capture wave then OOMed on all 8 shards.
# So after every engine-constructing (generate) wave: reap THIS RUN's orphaned
# workers, then VERIFY memory actually returned before the next wave — the
# verification is the gate, not the reap.
#
# Pid-namespace trap: nvidia-smi --query-compute-apps reports HOST pids that
# do not exist in the container's pid namespace (measured: host 3701656+ vs
# container 9016+), so nvidia-smi pids must never feed kill. Victims are found
# container-side — pgrep on the VLLM::EngineCore cmdline, one char bracketed
# so the probe cannot match its own command line — and killed by CAPTURED pid
# (TERM -> bounded grace -> KILL); nvidia-smi is used ONLY for the memory
# verification.
#
# Kill scope: a name-matched candidate is killed only when /proc/<pid>/environ
# carries THIS invocation's EPS_P1_RUN_TAG (launcher -> shard -> vLLM spawn
# children all inherit it); name matches without the tag are logged and left
# alone, so a stray unrelated vLLM on a shared box is never swept. Stated
# residual (best-effort scope): an orphan that is NOT a VLLM::EngineCore
# process, or whose environ is unreadable, is not reaped — the memory verify
# then FAILS LOUD (exit 5) instead of launching capture into held memory.
# ---------------------------------------------------------------------------

alloc_in_use() {
    local want="$1" g
    for ((g = 0; g < NUM_SHARDS; g++)); do
        if [ "${ALLOC[g]}" = "$want" ]; then
            return 0
        fi
    done
    return 1
}

# Prints "gpu<idx>=<used>MiB " for every in-allocation card at/over the
# threshold (empty output = all clear); rc=2 when nvidia-smi is unusable.
alloc_memory_offenders() {
    local rows idx used matched=0 offending=""
    rows="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)" \
        || return 2
    if [ -z "$rows" ]; then
        return 2
    fi
    while IFS=', ' read -r idx used; do
        if [ -z "$idx" ] || ! alloc_in_use "$idx"; then
            continue
        fi
        matched=1
        if ! [[ "$used" =~ ^[0-9]+$ ]] || [ "$used" -ge "$RECLAIM_THRESHOLD_MIB" ]; then
            offending+="gpu${idx}=${used}MiB "
        fi
    done <<< "$rows"
    if [ "$matched" -eq 0 ]; then
        # Allocation ids not visible as nvidia-smi indices (UUID-form CVD):
        # degrade to checking EVERY visible card — stricter, never weaker.
        while IFS=', ' read -r idx used; do
            if [ -z "$idx" ]; then
                continue
            fi
            if ! [[ "$used" =~ ^[0-9]+$ ]] || [ "$used" -ge "$RECLAIM_THRESHOLD_MIB" ]; then
                offending+="gpu${idx}=${used}MiB "
            fi
        done <<< "$rows"
    fi
    printf '%s' "$offending"
}

reclaim_gpu_between_waves() {
    local label="$1"
    local start deadline now offending
    start="$(date +%s)"
    deadline=$((start + RECLAIM_TIMEOUT_S))
    local candidates=() victims=() foreign=() pid
    mapfile -t candidates < <(pgrep -f 'VLLM::EngineCor[e]' || true)
    for pid in "${candidates[@]}"; do
        if tr '\0' '\n' 2>/dev/null < "/proc/$pid/environ" \
                | grep -qxF "EPS_P1_RUN_TAG=$EPS_P1_RUN_TAG"; then
            victims+=("$pid")
        else
            foreign+=("$pid")
        fi
    done
    if [ "${#foreign[@]}" -gt 0 ]; then
        echo "[phase=gpu_reclaim] leaving ${#foreign[@]} vLLM-named process(es) WITHOUT" \
            "this run's tag alone (not ours to kill): ${foreign[*]}"
    fi
    local level="none"
    if [ "${#victims[@]}" -gt 0 ]; then
        echo "[phase=gpu_reclaim] post-$label: reaping ${#victims[@]} orphaned vLLM" \
            "worker(s) by captured pid: ${victims[*]}"
        kill -TERM "${victims[@]}" 2>/dev/null || true
        level="TERM"
        local waited=0 alive=1
        while [ "$waited" -lt "$RECLAIM_KILL_GRACE_S" ]; do
            alive=0
            for pid in "${victims[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    alive=1
                fi
            done
            if [ "$alive" -eq 0 ]; then
                break
            fi
            sleep 1
            waited=$((waited + 1))
        done
        if [ "$alive" -ne 0 ]; then
            local survivors=()
            for pid in "${victims[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    survivors+=("$pid")
                fi
            done
            echo "[phase=gpu_reclaim] escalating TERM->KILL for: ${survivors[*]}"
            kill -KILL "${survivors[@]}" 2>/dev/null || true
            level="KILL"
        fi
    else
        echo "[phase=gpu_reclaim] post-$label: no orphaned vLLM workers to reap"
    fi
    # VERIFY — free memory on every allocation card is what capture actually
    # needs; a successful-looking kill is not sufficient evidence.
    while :; do
        if ! offending="$(alloc_memory_offenders)"; then
            echo "FATAL: nvidia-smi memory verification unusable after the $label wave" \
                "— cannot certify GPU memory was reclaimed; refusing to launch the" \
                "next wave blind" >&2
            exit 5
        fi
        if [ -z "$offending" ]; then
            now="$(date +%s)"
            echo "[phase=gpu_reclaim] post-$label PASS: all $NUM_SHARDS allocation" \
                "card(s) under ${RECLAIM_THRESHOLD_MIB} MiB (kill_level=$level" \
                "waited=$((now - start))s)"
            return 0
        fi
        now="$(date +%s)"
        if [ "$now" -ge "$deadline" ]; then
            echo "FATAL: GPU memory NOT reclaimed within ${RECLAIM_TIMEOUT_S}s of the" \
                "$label wave (kill_level=$level): ${offending}— refusing to launch the" \
                "next wave into unreclaimed memory (orphaned vLLM workers; gotchas.md" \
                "vLLM worker-subprocess teardown)" >&2
            exit 5
        fi
        echo "[phase=gpu_reclaim] memory still held: ${offending}(re-check in ${RECLAIM_POLL_S}s)"
        sleep "$RECLAIM_POLL_S"
    done
}

run_p1() {
    local mode="$1"
    if [ "$mode" = smoke ]; then
        local groot="$OUT_ROOT/smoke_gen"
        run_wave generate_smoke issue2658_generate \
            --smoke --smoke-cells "$SMOKE_CELLS" --responses "$SMOKE_RESPONSES"
        gate_generation_complete "$groot"
        reclaim_gpu_between_waves generate_smoke
        run_wave capture_smoke issue2658_capture --smoke --responses "$SMOKE_RESPONSES"
        echo "[p1] smoke chain complete (no uploads by design — see blind-spot enumeration)"
    else
        run_wave generate issue2658_generate
        gate_generation_complete "$OUT_ROOT"
        upload_generation
        reclaim_gpu_between_waves generate
        run_wave capture issue2658_capture --upload
        echo "[p1] production chain complete: generation + gate + raw upload + capture(+store upload)"
    fi
}

case "$MODE" in
    smoke) run_p1 smoke ;;
    production) run_p1 production ;;
    smoke_then_full)
        run_p1 smoke
        echo "[p1] smoke gate PASS — proceeding to the production wave"
        run_p1 production
        ;;
esac

# noqa: phase-done-reserved (mode: top-level dispatcher terminal; invoker: dispatch_issue.py --workload-cmd; all shard child logs are redirected per shard)
echo "[phase=done]"
