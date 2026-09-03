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
# upload, production only] -> [capture wave: N shards, --upload per shard] ->
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
#   Everything else — width derivation, CVD pinning, rc collection, gate,
#   capture chain — is the IDENTICAL launcher code path in both modes.
#
# Usage:
#   bash scripts/issue2658_p1_launch.sh                    # production wave
#   bash scripts/issue2658_p1_launch.sh --smoke            # smoke only
#   bash scripts/issue2658_p1_launch.sh --smoke-then-full  # smoke gate, then production
#
# Env knobs: EPS_P1_NUM_SHARDS, EPS_P1_SPLIT (default pilot), EPS_P1_OUT_ROOT,
# EPS_P1_LOG_DIR, EPS_P1_SMOKE_CELLS (default = shard count),
# EPS_P1_SMOKE_RESPONSES (default 2).
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

run_p1() {
    local mode="$1"
    if [ "$mode" = smoke ]; then
        local groot="$OUT_ROOT/smoke_gen"
        run_wave generate_smoke issue2658_generate \
            --smoke --smoke-cells "$SMOKE_CELLS" --responses "$SMOKE_RESPONSES"
        gate_generation_complete "$groot"
        run_wave capture_smoke issue2658_capture --smoke --responses "$SMOKE_RESPONSES"
        echo "[p1] smoke chain complete (no uploads by design — see blind-spot enumeration)"
    else
        run_wave generate issue2658_generate
        gate_generation_complete "$OUT_ROOT"
        upload_generation
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
