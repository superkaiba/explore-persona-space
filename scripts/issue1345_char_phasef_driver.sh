#!/usr/bin/env bash
# Issue #1345 char-capture-ladders — Phase-F fits driver (plan v13 §4 Phase F
# item 3 + §9 disk rows; closes the r1 concern phasef-stage-delete-orchestration
# by COMMITTING the stage->fit->delete interleave instead of leaving it
# orchestrator-improvised).
#
# Per selected char cell (of the 16), IN ORDER:
#   1. resume check: all 3 output JSONs exist -> skip the cell (no stage)
#   2. stage the cell's captured turnstore from HF
#      (issue1345_stage_char_stories.py --kind turnstore; ~5.2 GB/cell)
#   3. within-cell fits, BOTH arms (fill --stage cells --cells <v> --arm
#      context|prefix — plan §4 F item 1 / the both-arms rule)
#   4. the cell's ladder pair (fill --stage ladders --pairs <src>:<v>;
#      context arm per plan § Divergence 3)
#   5. DELETE the staged <v>_turnstore dir — the L19 slice caches the fill's
#      load_regime_xy wrote under --cache-dir are KEPT (different tree), and
#      the ladder-source stores (parent/assistant/onpolicy) are NEVER deleted.
# Peak transient stays ~1 staged cell (~6 GB), never 16 (~83 GB) — §9 disk row.
#
# Preflight (plan §8 risk table): df -P assert >= --min-free-gb (default 15)
# free on the filesystem backing --stage-root, re-checked before each cell's
# stage; abort loudly below floor. Ladder-source availability is asserted up
# front (context-arm L19 slice cache OR the staged source subdir; fallback =
# re-stage from the plan §10 HF pins).
#
# Failure semantics: a failed cell KEEPS its staged dir (debug/resume), the
# loop continues with the remaining cells, and the driver exits non-zero at
# the end. Fit hyperparameters are the plan §11 defaults (layer 19, reduced
# basis, seed 0, null-draws 2) — they are part of the resume filenames, so the
# driver pins rather than exposes them. Detached-friendly (no interactivity):
#   setsid nohup env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
#     NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
#     bash scripts/issue1345_char_phasef_driver.sh > logs/i1345_phasef.log 2>&1 &
#
# Usage:
#   bash scripts/issue1345_char_phasef_driver.sh [--plan] [--cells v1 v2 ...]
#        [--stage-root P] [--cache-dir P] [--out-dir P] [--max-rows N]
#        [--min-free-gb N] [--pilot-outdir P]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Shared-VM thread caps (setdefault — an explicit launch env always wins).
: "${OMP_NUM_THREADS:=8}"; export OMP_NUM_THREADS
: "${MKL_NUM_THREADS:=8}"; export MKL_NUM_THREADS
: "${OPENBLAS_NUM_THREADS:=8}"; export OPENBLAS_NUM_THREADS
: "${NUMEXPR_NUM_THREADS:=8}"; export NUMEXPR_NUM_THREADS
: "${MALLOC_ARENA_MAX:=2}"; export MALLOC_ARENA_MAX

EPS_USER="${USER:-thomasjiralerspong}"
STAGE_ROOT="/mnt/eps-data/${EPS_USER}/issue1887_lambda_audit/issue1345"
CACHE_DIR="/mnt/eps-data/${EPS_USER}/issue1345_story_char_fill"
OUT_DIR="${REPO_ROOT}/eval_results/issue_1345/story_char_ladder_fill"
MIN_FREE_GB=15
MAX_ROWS=0
PLAN_ONLY=0
PILOT_OUTDIR=""
SELECTED=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --plan) PLAN_ONLY=1; shift ;;
    --stage-root) STAGE_ROOT="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    --pilot-outdir) PILOT_OUTDIR="$2"; shift 2 ;;
    --cells) shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do SELECTED+=("$1"); shift; done ;;
    *) echo "[phasef] unknown arg: $1" >&2; exit 2 ;;
  esac
done
# #2479 P0 pilot mode: route the driver's own expected-output paths AND the
# fill's outputs to the pilot dir (the fill also SKIPS its axis-freeze guard
# there; it ap.error-refuses the flag with any panel cell, so pilot mode can
# never become a panel bypass).
if [ -n "$PILOT_OUTDIR" ]; then OUT_DIR="$PILOT_OUTDIR"; fi

# Plan §11 pinned fit defaults — mirrored in the fill script's argparse
# defaults AND its resume filenames (fill main() L1025/L1087); keep in sync.
LAYER=19
BASIS="reduced"
SEED=0
ND=2
ROWS_TAG=""
if [ "$MAX_ROWS" -gt 0 ]; then ROWS_TAG="_rows${MAX_ROWS}"; fi

# Cell table (variant|ladder_src|model) — mirrors char_pair_specs() in
# scripts/issue1345_story_char_ladder_fill.py (base asymmetry: _base cells
# ladder from the pretrained chat store r1; plan § Divergences 2).
CELLS=(
  "char_helios|r4|instruct"
  "char_helios_op|r4op|instruct"
  "char_helios_base|r1|pretrained"
  "char_helios_op_base|r1|pretrained"
  "char_wren|r4|instruct"
  "char_wren_op|r4op|instruct"
  "char_wren_base|r1|pretrained"
  "char_wren_op_base|r1|pretrained"
  "char_dana|r4|instruct"
  "char_dana_op|r4op|instruct"
  "char_dana_base|r1|pretrained"
  "char_dana_op_base|r1|pretrained"
  "char_vex|r4|instruct"
  "char_vex_op|r4op|instruct"
  "char_vex_base|r1|pretrained"
  "char_vex_op_base|r1|pretrained"
)

# --- #2479 panel append seam (EPM_I2479_CHAR_PANEL_JSON) ---------------------
# Env absent => the parent 16-cell table above stays byte-identical. Env set =>
# panel cells APPEND in registry order (variant_op ladders from r4op, non-null
# variant_inserted from r4 — mirrors char_pair_specs()'s panel enumeration in
# the fill script). Fail-loud on a bad panel.
if [ -n "${EPM_I2479_CHAR_PANEL_JSON:-}" ]; then
  panel_rows="$(uv run python - <<'PY'
import sys

sys.path.insert(0, "scripts")
from issue2479_char_panel import load_char_panel_env

rows = load_char_panel_env()
assert rows, "EPM_I2479_CHAR_PANEL_JSON set but loader returned no rows"
for r in rows:
    print(f"{r['variant_op']}|r4op|instruct")
    if r["variant_inserted"]:
        print(f"{r['variant_inserted']}|r4|instruct")
PY
)" || { echo "[phasef] #2479 panel load failed" >&2; exit 2; }
  while IFS= read -r row; do
    [ -n "$row" ] && CELLS+=("$row")
  done <<< "$panel_rows"
fi

if [ "${#SELECTED[@]}" -gt 0 ]; then
  FILTERED=()
  for spec in "${CELLS[@]}"; do
    v="${spec%%|*}"
    for s in "${SELECTED[@]}"; do
      if [ "$v" = "$s" ]; then FILTERED+=("$spec"); fi
    done
  done
  if [ "${#FILTERED[@]}" -ne "${#SELECTED[@]}" ]; then
    echo "[phasef] --cells contains an unknown cell (selected: ${SELECTED[*]})" >&2
    exit 2
  fi
  CELLS=("${FILTERED[@]}")
fi
if [ "${#CELLS[@]}" -lt 1 ]; then
  echo "[phasef] no cells selected" >&2
  exit 2
fi

src_subdir() { # ladder-source regime -> staged subdir under STAGE_ROOT
  case "$1" in
    r4) echo "conversation_paired_stories_assistant_turnstore" ;;
    r4op) echo "onpolicy_assistant_story_turnstore" ;;
    r1) echo "parent_turnstore" ;;
    *) echo "UNKNOWN_SRC_$1" ;;
  esac
}

src_cache_stem() { # ladder-source regime + model -> context-arm L19 slice cache
  local model="$2" fk
  case "$1" in
    r4) fk="stories_paired" ;;
    r4op) fk="stories_paired_op" ;;
    r1) fk="chat" ;;
    *) fk="UNKNOWN" ;;
  esac
  echo "${model}_${fk}_s_context_L${LAYER}.pt"
}

expected_outputs() { # variant src model -> newline-separated expected JSONs
  local v="$1" src="$2" model="$3"
  echo "${OUT_DIR}/cell_${v}__${model}_context_L${LAYER}_${BASIS}_s${SEED}${ROWS_TAG}.json"
  echo "${OUT_DIR}/cell_${v}__${model}_prefix_L${LAYER}_${BASIS}_s${SEED}${ROWS_TAG}.json"
  echo "${OUT_DIR}/ladder_${src}__${v}__${model}_context_L${LAYER}_${BASIS}_s${SEED}_nd${ND}${ROWS_TAG}.json"
}

cell_outputs_complete() { # variant src model -> 0 iff all expected JSONs exist
  local f
  while IFS= read -r f; do
    [ -f "$f" ] || return 1
  done < <(expected_outputs "$1" "$2" "$3")
  return 0
}

free_gb_at() { # path -> whole GiB free on its filesystem (df -P, 1K blocks)
  df -P "$1" | awk 'NR==2 {printf "%d", $4/1048576}'
}

assert_disk_floor() { # context-string -> abort (exit 3) below MIN_FREE_GB
  local ctx="$1" free
  free="$(free_gb_at "$STAGE_ROOT")"
  if ! [[ "$free" =~ ^[0-9]+$ ]] || [ "$free" -lt "$MIN_FREE_GB" ]; then
    echo "[phasef] DISK FLOOR: ${free:-?} GiB free at ${STAGE_ROOT} < ${MIN_FREE_GB} GiB (${ctx}) — aborting (plan §8: free headroom or raise the quota, never delete active data)" >&2
    exit 3
  fi
}

# ---------------------------------------------------------------------------
# --plan: print the per-cell plan and run NOTHING (local checks only)
# ---------------------------------------------------------------------------
if [ "$PLAN_ONLY" = "1" ]; then
  echo "[plan] cells=${#CELLS[@]} stage_root=${STAGE_ROOT} cache_dir=${CACHE_DIR}"
  echo "[plan] out_dir=${OUT_DIR} min_free_gb=${MIN_FREE_GB} max_rows=${MAX_ROWS}"
  for spec in "${CELLS[@]}"; do
    IFS='|' read -r v src model <<< "$spec"
    if cell_outputs_complete "$v" "$src" "$model"; then st="skip (outputs exist)"; else st="run"; fi
    echo "  ${v} (src=${src} model=${model}): ${st} -> stage + fits(cells x {context,prefix} + ladder ${src}:${v}) + delete ${STAGE_ROOT}/${v}_turnstore"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Preflight: disk floor + ladder-source availability (plan §8 assumptions 5/8)
# ---------------------------------------------------------------------------
mkdir -p "$STAGE_ROOT" "$CACHE_DIR" "$OUT_DIR"
assert_disk_floor "preflight"
echo "[phasef] preflight: $(free_gb_at "$STAGE_ROOT") GiB free at ${STAGE_ROOT} (floor ${MIN_FREE_GB} GiB)"

declare -A SRC_SEEN=()
for spec in "${CELLS[@]}"; do
  IFS='|' read -r v src model <<< "$spec"
  key="${src}|${model}"
  [ -n "${SRC_SEEN[$key]:-}" ] && continue
  SRC_SEEN[$key]=1
  stem="$(src_cache_stem "$src" "$model")"
  subdir="$(src_subdir "$src")"
  if [ -f "${CACHE_DIR}/${stem}" ]; then
    echo "[phasef] source ${src}/${model}: slice cache present (${stem})"
  elif [ -d "${STAGE_ROOT}/${subdir}" ]; then
    echo "[phasef] source ${src}/${model}: staged store present (${subdir}); will slice on first use"
  else
    echo "[phasef] SOURCE MISSING: ${src}/${model} has neither slice cache ${CACHE_DIR}/${stem} nor staged store ${STAGE_ROOT}/${subdir} — re-stage the ladder source from the plan §10 HF pins before running Phase F" >&2
    exit 4
  fi
done

mkdir -p logs

# ---------------------------------------------------------------------------
# Per-cell loop: stage -> fits (cells x 2 arms + ladder) -> delete staged dir
# ---------------------------------------------------------------------------
declare -A CELL_RC
rc=0
n_ok=0
n_skip=0
n_fail=0

FILL_ARGS=(--stage-root "$STAGE_ROOT" --cache-dir "$CACHE_DIR" --out-dir "$OUT_DIR")
if [ "$MAX_ROWS" -gt 0 ]; then FILL_ARGS+=(--max-rows "$MAX_ROWS"); fi
if [ -n "$PILOT_OUTDIR" ]; then FILL_ARGS+=(--pilot-outdir "$PILOT_OUTDIR"); fi

for spec in "${CELLS[@]}"; do
  IFS='|' read -r v src model <<< "$spec"
  tsdir="${STAGE_ROOT}/${v}_turnstore"
  if cell_outputs_complete "$v" "$src" "$model"; then
    # r2 review Minor 1: a kill in the fits-done -> delete window leaves the
    # staged dir orphaned (~5.2 GB); outputs are complete, so it is provably
    # dead weight (name guard char_*_turnstore holds by construction: v comes
    # only from the fixed cell table).
    if [ -d "$tsdir" ]; then
      sz="$(du -sh "$tsdir" 2>/dev/null | cut -f1)"
      rm -rf "$tsdir"
      if [ -e "$tsdir" ]; then
        echo "[cell] ${v}: WARNING leftover staged dir ${tsdir} could not be removed" >&2
      else
        echo "[cell] ${v}: removed leftover staged ${v}_turnstore (${sz:-?} freed; outputs complete)"
      fi
    fi
    echo "[cell] ${v}: all 3 outputs exist — skipped (resume; nothing staged)"
    CELL_RC[$v]="skipped-complete"
    n_skip=$((n_skip + 1))
    continue
  fi
  assert_disk_floor "before staging ${v}"
  t0=$(date +%s)
  cell_rc=0

  if ! uv run python scripts/issue1345_stage_char_stories.py \
      --kind turnstore --variant "$v" --dest-root "$STAGE_ROOT"; then
    cell_rc=10
    echo "[cell] ${v}: STAGE FAILED rc=10 — staged dir kept for debug; continuing" >&2
  fi

  if [ "$cell_rc" -eq 0 ]; then
    for arm in context prefix; do
      if ! uv run python scripts/issue1345_story_char_ladder_fill.py \
          --stage cells --cells "$v" --model "$model" --arm "$arm" "${FILL_ARGS[@]}"; then
        cell_rc=11
        echo "[cell] ${v}: WITHIN-CELL FIT (${arm}) FAILED rc=11" >&2
        break
      fi
    done
  fi

  if [ "$cell_rc" -eq 0 ]; then
    if ! uv run python scripts/issue1345_story_char_ladder_fill.py \
        --stage ladders --pairs "${src}:${v}" --model "$model" "${FILL_ARGS[@]}"; then
      cell_rc=12
      echo "[cell] ${v}: LADDER PAIR ${src}:${v} FAILED rc=12" >&2
    fi
  fi

  if [ "$cell_rc" -eq 0 ]; then
    # Delete ONLY this cell's staged full-shard dir (never a ladder-source
    # store — enforced by the char_*_turnstore name guard). Slice caches
    # under CACHE_DIR are untouched.
    base="$(basename "$tsdir")"
    case "$base" in
      char_*_turnstore)
        if [ -d "$tsdir" ]; then
          sz="$(du -sh "$tsdir" 2>/dev/null | cut -f1)"
          rm -rf "$tsdir"
          if [ -e "$tsdir" ]; then
            echo "[cell] ${v}: DELETE FAILED — ${tsdir} still present" >&2
            cell_rc=13
          else
            echo "[cell] ${v}: deleted staged ${base} (${sz:-?} freed)"
          fi
        else
          echo "[cell] ${v}: staged dir already absent (slice-cache-only run)"
        fi
        ;;
      *)
        echo "[cell] ${v}: REFUSING to delete non-cell dir ${tsdir}" >&2
        cell_rc=13
        ;;
    esac
  fi

  CELL_RC[$v]="$cell_rc"
  if [ "$cell_rc" -eq 0 ]; then n_ok=$((n_ok + 1)); else
    n_fail=$((n_fail + 1))
    if [ "$rc" -eq 0 ]; then rc="$cell_rc"; fi
  fi
  echo "[cell] ${v} done rc=${cell_rc} wall $(( $(date +%s) - t0 ))s ($(free_gb_at "$STAGE_ROOT") GiB free)"
done

per_cell=""
for spec in "${CELLS[@]}"; do
  v="${spec%%|*}"
  per_cell="${per_cell}${v}=${CELL_RC[$v]:-not-run} "
done
echo "[phasef] done cells=${#CELLS[@]} ok=${n_ok} skipped=${n_skip} failed=${n_fail} rc=${rc} | ${per_cell% }"
exit "$rc"
