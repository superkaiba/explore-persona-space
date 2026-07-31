#!/usr/bin/env bash
# Issue #1345 on-policy-vs-injected program — the 4 missing answer-provenance
# cells (bare_text x {instruct, pretrained}, chat x pretrained, story_slot x
# pretrained; instruct-chat on-policy is already banked as the bnd_chat
# comparator store — track_s answers are instruct chat generations).
#
# PILOT-FIRST sizing discipline (item-2 report: no measured per-row basis
# exists, and a guessed basis is a banned sizing input): cell 1 runs FIRST
# with --n-rows 64 on one GPU; its measured wall (and the generator's own
# [vllm-chunk] lines) is the recorded per-row basis for the full 16,048
# generations. The generator checkpoints + resumes, so the pilot's 64 rows
# count toward cell 1's full pool — nothing is wasted.
#
# Staging: --matched-dir/--dl-dir point at the ALREADY-staged
# data/issue_1345/story_boundary_ablation/{matched_n,hf_dl} (the pool is
# variant-independent; a per-variant prefetch would copy a 2.24 GB chat
# store per cell for nothing). Fail-loud guard below instead of a prefetch.
# Device selection is allocation-safe (index INTO the SLURM-set CVD list,
# never absolute indices; >=120 GiB free filter — fellows nodes are
# GPU-shared).
set -uo pipefail

mkdir -p logs
S="${EPM_I1345_STAGED_DIR:-data/issue_1345/story_boundary_ablation}"
for d in "$S/matched_n" "$S/hf_dl"; do
  if [ ! -d "$d" ]; then
    echo "[onpolicy-launch] STAGED-DIR-MISSING: $d (set EPM_I1345_STAGED_DIR or stage first)" >&2
    exit 7
  fi
done

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra alloc <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t alloc < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
DEVICES=()
for d in "${alloc[@]}"; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$d" | head -1)"
  if [ "${free_mib:-0}" -ge 120000 ]; then
    DEVICES+=("$d")
  else
    echo "[onpolicy-launch] skipping device ${d}: only ${free_mib:-?} MiB free" >&2
  fi
done
n_gpu="${#DEVICES[@]}"
echo "[onpolicy-launch] allocated: ${alloc[*]:-none}; usable (>=120 GiB free): ${DEVICES[*]:-none}"
if [ "$n_gpu" -lt 1 ]; then
  echo "[onpolicy-launch] no usable GPUs" >&2
  exit 3
fi

# Cell table (variant|character|shape|model|matched-needed) — commands verbatim
# from the item-2 report; cell 4's character MUST be Assistant (the V1
# bundle's ROUND_CHARACTER).
CELLS=(
  "onpolicy_answers_ntpl_instruct|ARIA|bare_text|instruct|1"
  "onpolicy_answers_ntpl_base|ARIA|bare_text|pretrained|1"
  "onpolicy_answers_chat_base|ARIA|chat|pretrained|1"
  "onpolicy_answers_slot_base|Assistant|story_slot|pretrained|0"
)

run_cell() {
  local spec="$1" dev="$2" extra="${3:-}"
  local variant character shape model matched
  IFS='|' read -r variant character shape model matched <<< "$spec"
  local args=(--shape "$shape" --model "$model" --dl-dir "$S/hf_dl")
  if [ "$matched" = "1" ]; then
    args+=(--matched-dir "$S/matched_n")
  fi
  # shellcheck disable=SC2086
  env EPM_I1345_VARIANT="$variant" EPM_STORY_CHARACTER_NAME="$character" \
    CUDA_VISIBLE_DEVICES="$dev" \
    uv run python scripts/issue1345_onpolicy_answers_gen.py "${args[@]}" $extra
}

# ---- PILOT: cell 1, 64 rows, one GPU; the measured basis for the full run.
pilot_start="$(date -u +%s)"
echo "[onpolicy-launch] PILOT: cell 1 (--n-rows 64) on device ${DEVICES[0]} ($(date -u +%FT%TZ))"
run_cell "${CELLS[0]}" "${DEVICES[0]}" "--n-rows 64" > logs/i1345_onpolicy_pilot.log 2>&1
prc=$?
pilot_wall=$(( $(date -u +%s) - pilot_start ))
echo "[onpolicy-launch] PILOT done rc=${prc} wall=${pilot_wall}s (~$((pilot_wall * 1000 / 64)) ms/row incl. engine init; chunk detail in logs/i1345_onpolicy_pilot.log)"
if [ "$prc" -ne 0 ]; then
  echo "[onpolicy-launch] PILOT FAILED — not launching full cells" >&2
  tail -30 logs/i1345_onpolicy_pilot.log >&2
  exit "$prc"
fi

# ---- FULL: all 4 cells, wave-scheduled over the usable devices.
rc=0
idx=0
while [ "$idx" -lt "${#CELLS[@]}" ]; do
  pids=()
  labels=()
  for g in $(seq 0 $((n_gpu - 1))); do
    [ "$idx" -ge "${#CELLS[@]}" ] && break
    spec="${CELLS[$idx]}"
    variant="${spec%%|*}"
    dev="${DEVICES[$g]}"
    echo "[onpolicy-launch] starting ${variant} on device ${dev} ($(date -u +%FT%TZ))"
    run_cell "$spec" "$dev" > "logs/i1345_onpolicy_${variant}.log" 2>&1 &
    pids+=("$!")
    labels+=("$variant")
    idx=$((idx + 1))
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}"
    arc=$?
    echo "[onpolicy-launch] ${labels[$j]} finished rc=${arc} ($(date -u +%FT%TZ))"
    if [ "$arc" -ne 0 ] && [ "$rc" -eq 0 ]; then
      rc="$arc"
    fi
  done
done

echo "[onpolicy-launch] all cells done, rc=${rc}"
exit "$rc"
