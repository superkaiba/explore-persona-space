#!/usr/bin/env bash
# Issue #488 Phase 1 parallel dispatcher — fan JS/KL out across 8 GPUs.
#
# Phase 1 is CPU-and-GPU-throughput bound: per-position full-vocab JS/KL over
# Qwen-2.5-7B-Instruct (vocab 151,646) across ~462 pending (ci, cj) cells.
# On a single GPU the rate is ~0.5 cells/min → ~14 wall-h, > 2× the 49-GPU-h /
# 7-wall-h plan budget. The pod has 8× H100; parallelizing across them brings
# the JS/KL stage back inside budget.
#
# Architecture (UNIFIED smoke/sweep — verdict PASS_UNIFIED):
#   * Smoke = this same script with 2 shards over `--pairs A1:G2 G2:A1`
#             (2-cell tiny slice).
#   * Sweep = this same script with 8 shards over the full pending list.
#   Same dispatcher, same subprocess shape, same per-shard code path, same
#   per-shard log structure, same merge step.
#
# Workload split:
#   1. Ask `i488_phase1_predictors.py --print-pending-pairs` for the
#      deterministic pending-pair list (honors any existing per-shard
#      checkpoints so resume picks up where it left off).
#   2. Round-robin partition the list into NUM_SHARDS chunks.
#   3. Launch one subprocess per shard with CUDA_VISIBLE_DEVICES=<i>,
#      --gpu-id 0 (each child sees its assigned GPU as device 0),
#      --pairs <chunk_i>, --out-suffix _g<i>,
#      --skip cossim stylization (those passes are NOT pair-level and are
#      handled by the post-merge serial pass below).
#   4. `wait` for all shards; fail loud on any non-zero exit.
#   5. Merge `js_matrix_g<i>.json` + `kl_matrix_g<i>.json` into the canonical
#      `js_matrix.json` + `kl_matrix.json` (downstream Phase 5 reads these
#      no-suffix paths).
#   6. Run the single-GPU serial pass for stylization + cosine (which depend
#      on per-condition output distributions, not pair-level — N_CONDITIONS
#      iterations regardless of GPU count, so parallelizing them buys little).
#
# Per CLAUDE.md "Checkpoint per phase": each shard writes its own
# `js_matrix_g<i>.json` incrementally; a shard restart skips already-filled
# cells in its chunk. Mid-shard crashes do NOT lose other shards' progress.
#
# Per CLAUDE.md `Pod-side code NEVER shells out to scripts/task.py`: this
# script is bash + `uv run python`; no `task.py` calls. Phase markers go to
# stdout for `poll_pipeline.py` to parse.
#
# Usage:
#     bash scripts/i488_phase1_parallel.sh                  # 8 shards, full pending
#     NUM_SHARDS=2 bash scripts/i488_phase1_parallel.sh     # smoke (2 shards)
#     NUM_SHARDS=8 R_SAMPLES=2 bash scripts/i488_phase1_parallel.sh  # sweep default
#     SMOKE_PAIRS="A1:G2 G2:A1" bash scripts/i488_phase1_parallel.sh # tiny smoke
#     DRY_RUN=1 NUM_SHARDS=2 bash scripts/i488_phase1_parallel.sh    # chunk-split
#                                                                     only, no GPU
#                                                                     launch (CPU
#                                                                     smoke)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

NUM_SHARDS="${NUM_SHARDS:-8}"
R_SAMPLES="${R_SAMPLES:-2}"
N_PROBES="${N_PROBES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
# When set, all shards process the same explicit smoke pair-list (smoke-test);
# the dispatcher splits it across NUM_SHARDS shards round-robin.
SMOKE_PAIRS="${SMOKE_PAIRS:-}"
# When DRY_RUN=1, do the pending discovery + chunk-split + log the launch
# commands but do NOT spawn the GPU subprocesses. Used for CPU-only smoke of
# the dispatcher logic.
DRY_RUN="${DRY_RUN:-0}"

OUT_DIR="eval_results/issue_488/predictors"
LOG_DIR="logs/issue_488"
mkdir -p "$OUT_DIR" "$LOG_DIR"

START_TS=$(date -Iseconds)
echo "[phase=phase1_parallel] $START_TS NUM_SHARDS=$NUM_SHARDS R_SAMPLES=$R_SAMPLES"

# ── Step 1: discover pending pair list ─────────────────────────────────────
PAIRS_FILE="$LOG_DIR/phase1_pending_pairs.txt"
if [ -n "$SMOKE_PAIRS" ]; then
    # Smoke: explicit pair-list provided.
    echo "[phase=phase1_parallel] SMOKE_PAIRS override: $SMOKE_PAIRS"
    : > "$PAIRS_FILE"
    for p in $SMOKE_PAIRS; do
        echo "$p" >> "$PAIRS_FILE"
    done
else
    # Production: ask Phase 1 itself for the deterministic pending list,
    # honoring the canonical (no-suffix) checkpoint if any exists.
    echo "[phase=phase1_parallel] discovering pending pairs (no model load)"
    uv run python scripts/i488_phase1_predictors.py --print-pending-pairs \
        > "$PAIRS_FILE" 2> "$LOG_DIR/phase1_pending_discover.log"
fi
TOTAL_PAIRS=$(wc -l < "$PAIRS_FILE" | tr -d ' ')
echo "[phase=phase1_parallel] $TOTAL_PAIRS pending pairs → $NUM_SHARDS shards"

if [ "$TOTAL_PAIRS" -eq 0 ]; then
    echo "[phase=phase1_parallel] no pending pairs; skipping JS/KL parallel stage"
else
    # ── Step 2: round-robin split into NUM_SHARDS chunks ───────────────────
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        # awk: print every Nth line starting at offset i.
        awk -v n="$NUM_SHARDS" -v r="$i" 'NR % n == r' "$PAIRS_FILE" \
            > "$LOG_DIR/phase1_chunk_g${i}.txt"
    done

    # ── Step 3: launch 8 shards in parallel ────────────────────────────────
    declare -a SHARD_PIDS=()
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        CHUNK_FILE="$LOG_DIR/phase1_chunk_g${i}.txt"
        SHARD_LOG="$LOG_DIR/phase1_g${i}.log"
        CHUNK_SIZE=$(wc -l < "$CHUNK_FILE" | tr -d ' ')
        if [ "$CHUNK_SIZE" -eq 0 ]; then
            echo "[phase=phase1_parallel] shard g${i}: empty chunk; skipping"
            continue
        fi
        # Read the chunk into a space-separated string for argparse nargs+.
        PAIRS_ARGS=$(tr '\n' ' ' < "$CHUNK_FILE")
        echo "[phase=phase1_parallel] launching shard g${i} on GPU $i ($CHUNK_SIZE pairs) → $SHARD_LOG"
        if [ "$DRY_RUN" = "1" ]; then
            echo "[phase=phase1_parallel] DRY_RUN: would run CUDA_VISIBLE_DEVICES=$i uv run python scripts/i488_phase1_predictors.py --gpu-id 0 --out-suffix _g${i} --skip cossim stylization --pairs <$CHUNK_SIZE pairs>"
            continue
        fi
        (
            CUDA_VISIBLE_DEVICES="$i" \
            uv run python scripts/i488_phase1_predictors.py \
                --gpu-id 0 \
                --r-samples "$R_SAMPLES" \
                --n-probes "$N_PROBES" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --out-suffix "_g${i}" \
                --skip cossim stylization \
                --pairs $PAIRS_ARGS \
                > "$SHARD_LOG" 2>&1
        ) &
        SHARD_PIDS+=($!)
    done

    if [ "$DRY_RUN" = "1" ]; then
        echo "[phase=phase1_parallel] DRY_RUN: skipping wait + merge + aux"
        echo "[phase=phase1_parallel] dry-run done $(date -Iseconds)"
        exit 0
    fi

    # ── Step 4: wait for all shards; fail loud on any non-zero ─────────────
    FAILED=0
    for pid in "${SHARD_PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "[phase=phase1_parallel] shard pid=$pid FAILED" >&2
            FAILED=1
        fi
    done
    if [ "$FAILED" -ne 0 ]; then
        echo "[phase=phase1_parallel] one or more shards FAILED; aborting before merge" >&2
        exit 1
    fi
    echo "[phase=phase1_parallel] all $NUM_SHARDS shards PASSED; merging"

    # Build EXPECTED_SHARDS: comma-separated list of shard indices that had a
    # non-empty chunk and were actually launched. The merge step asserts every
    # one of these produced both js_matrix_g${i}.json AND kl_matrix_g${i}.json
    # (catches the silent-skip path where a launched shard exits 0 without
    # writing output → incomplete canonical matrix).
    EXPECTED_SHARDS=""
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        CHUNK_FILE="$LOG_DIR/phase1_chunk_g${i}.txt"
        CHUNK_SIZE=$(wc -l < "$CHUNK_FILE" | tr -d ' ')
        if [ "$CHUNK_SIZE" -gt 0 ]; then
            if [ -z "$EXPECTED_SHARDS" ]; then
                EXPECTED_SHARDS="$i"
            else
                EXPECTED_SHARDS="$EXPECTED_SHARDS,$i"
            fi
        fi
    done
    echo "[phase=phase1_parallel] expected shard outputs: g{$EXPECTED_SHARDS}"

    # ── Step 5: merge per-shard JS/KL into canonical no-suffix outputs ─────
    NUM_SHARDS="$NUM_SHARDS" OUT_DIR="$OUT_DIR" EXPECTED_SHARDS="$EXPECTED_SHARDS" \
        SMOKE_PAIRS="$SMOKE_PAIRS" \
        uv run python - <<'PY'
"""Merge per-shard JS/KL outputs into the canonical no-suffix files.

For each cell (ci, cj):
- The inherited 16×16 sub-grid from #406 is byte-identical across shards
  (every shard runs `_seed_js_kl_from_i406`); take it from shard 0.
- Every other cell is computed by exactly ONE shard (round-robin partition
  of the pending list). Assert this invariant — any cell computed by >1
  shard would indicate a split bug.
"""
import json
import os
from pathlib import Path

OUT_DIR = Path(os.environ["OUT_DIR"])
NUM_SHARDS = int(os.environ["NUM_SHARDS"])
EXPECTED_SHARDS_RAW = os.environ.get("EXPECTED_SHARDS", "")
EXPECTED_SHARDS = (
    {int(x) for x in EXPECTED_SHARDS_RAW.split(",") if x.strip() != ""}
    if EXPECTED_SHARDS_RAW
    else set()
)

shards_js = []
shards_kl = []
shards_meta = []
for i in range(NUM_SHARDS):
    js_p = OUT_DIR / f"js_matrix_g{i}.json"
    kl_p = OUT_DIR / f"kl_matrix_g{i}.json"
    if i in EXPECTED_SHARDS:
        # Shard was launched with a non-empty chunk → it MUST have produced
        # both output files. A launched shard that exits 0 without writing
        # output would silently leave its chunk's cells unfilled in the
        # canonical matrix; fail loud instead.
        if not js_p.exists() or not kl_p.exists():
            raise SystemExit(
                f"shard g{i} was launched (non-empty chunk) but produced "
                f"incomplete output: js_matrix_g{i}.json exists={js_p.exists()}, "
                f"kl_matrix_g{i}.json exists={kl_p.exists()}"
            )
    elif not js_p.exists():
        # Shard was not launched (empty chunk); nothing to merge from it.
        continue
    js_doc = json.loads(js_p.read_text())
    kl_doc = json.loads(kl_p.read_text())
    shards_js.append(js_doc["JS"])
    shards_kl.append(kl_doc["KL"])
    shards_meta.append((js_doc, kl_doc))

if not shards_js:
    raise SystemExit("no shard outputs found; nothing to merge")

# Conditions list — same across all shards.
cids = shards_meta[0][0]["conditions"]
n_probes = shards_meta[0][0]["n_probes"]
r_samples = shards_meta[0][0]["r_samples"]

# Initialize merged matrices from shard 0 (inherited cells already correct).
merged_js = {ci: {cj: shards_js[0][ci][cj] for cj in cids} for ci in cids}
merged_kl = {ci: {cj: shards_kl[0][ci][cj] for cj in cids} for ci in cids}

# Walk the remaining shards. For each non-None cell that differs from
# shard 0's value, validate + take it.
contributions = [0] * len(shards_js)
contributions[0] = sum(
    1
    for ci in cids
    for cj in cids
    if ci != cj and merged_js[ci][cj] is not None
)

for s_idx in range(1, len(shards_js)):
    for ci in cids:
        for cj in cids:
            if ci == cj:
                continue
            v_other = shards_js[s_idx][ci][cj]
            v_merged = merged_js[ci][cj]
            if v_other is None:
                continue
            if v_merged is None:
                # New cell from this shard — take it.
                merged_js[ci][cj] = v_other
                merged_kl[ci][cj] = shards_kl[s_idx][ci][cj]
                contributions[s_idx] += 1
            else:
                # Both shards have a value. For inherited cells this is
                # expected (byte-identical). For non-inherited cells this
                # would mean two shards computed the same pair → bug.
                # Loose check: values must agree to float precision.
                if abs(v_merged - v_other) > 1e-9:
                    raise SystemExit(
                        f"merge collision at ({ci},{cj}): "
                        f"shard0={v_merged} shard{s_idx}={v_other}"
                    )

print(f"merged from {len(shards_js)} shards; contributions: {contributions}")
filled = sum(
    1 for ci in cids for cj in cids if ci != cj and merged_js[ci][cj] is not None
)
expected = len(cids) * (len(cids) - 1)
print(f"merged matrix: {filled} / {expected} cells filled")
if filled != expected:
    # SMOKE_PAIRS / partial pair-lists legitimately leave cells unfilled;
    # only fail loud when the pending discover path was used (every
    # off-diagonal pair should be filled by exactly one shard).
    if os.environ.get("SMOKE_PAIRS", "") == "":
        raise SystemExit(
            f"incomplete canonical matrix: {filled} / {expected} cells filled "
            f"(missing {expected - filled}); a shard silently dropped pairs"
        )
    else:
        print(
            f"WARN: {expected - filled} cells unfilled but SMOKE_PAIRS set, "
            "so partial coverage is expected; not failing"
        )

# Write canonical no-suffix outputs.
canonical_js = OUT_DIR / "js_matrix.json"
canonical_kl = OUT_DIR / "kl_matrix.json"
canonical_js.write_text(
    json.dumps(
        {
            "schema_version": "i488_v1",
            "conditions": cids,
            "JS": merged_js,
            "n_probes": n_probes,
            "r_samples": r_samples,
            "merged_from_shards": len(shards_js),
        },
        indent=2,
    )
)
canonical_kl.write_text(
    json.dumps(
        {
            "schema_version": "i488_v1",
            "conditions": cids,
            "KL": merged_kl,
            "n_probes": n_probes,
            "r_samples": r_samples,
            "merged_from_shards": len(shards_js),
        },
        indent=2,
    )
)
print(f"wrote {canonical_js}")
print(f"wrote {canonical_kl}")
PY
    echo "[phase=phase1_parallel] merge complete"
fi

# ── Step 6: serial pass for stylization + cosine (NOT pair-level) ──────────
# These iterate over per-condition output distributions, not (ci, cj) pairs.
# 27 conds × 1 estimator → splitting across GPUs buys ~nothing; the existing
# serial code path handles them. The script will use the canonical no-suffix
# `js_matrix.json` / `kl_matrix.json` written by the merge above, so it skips
# the JS/KL work entirely (--skip js kl).
echo "[phase=phase1_serial_aux] $(date -Iseconds) stylization + cosine"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i488_phase1_predictors.py \
    --gpu-id 0 \
    --r-samples "$R_SAMPLES" \
    --n-probes "$N_PROBES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --skip js kl \
    > "$LOG_DIR/phase1_serial_aux.log" 2>&1
echo "[phase=phase1_serial_aux] ok"

echo "[phase=phase1_parallel] done $(date -Iseconds)"
