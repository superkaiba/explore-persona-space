#!/usr/bin/env bash
# Issue #1345 story-boundary-ablation — capture launcher.
# Teacher-forced L19 capture for the 6 stores (arms v1-v4 + comparators
# chat/no_template), one store per GPU, wave-scheduled off the SLURM-allocated
# device list (the 15771 lesson: NEVER export absolute CVD indices on a shared
# node — index INTO the pre-set allocation and keep only devices with enough
# free memory; capture holds one 7B model, ~20 GiB, floor 30000 MiB). BLOCKS
# until every unit finishes; exits with the first non-zero unit rc.
set -uo pipefail

export EPM_I1345_VARIANT=story_boundary_ablation
export EPM_STORY_CHARACTER_NAME=Assistant

mkdir -p logs

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra alloc <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t alloc < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
DEVICES=()
for d in "${alloc[@]}"; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$d" | head -1)"
  if [ "${free_mib:-0}" -ge 30000 ]; then
    DEVICES+=("$d")
  else
    echo "[launch-cap] skipping device ${d}: only ${free_mib:-?} MiB free" >&2
  fi
done
n_gpu="${#DEVICES[@]}"
echo "[launch-cap] allocated devices: ${alloc[*]:-none}; usable (>=30 GiB free): ${DEVICES[*]:-none}"
if [ "$n_gpu" -lt 1 ]; then
  echo "[launch-cap] no usable GPUs" >&2
  exit 3
fi

# Units: "kind:name". Override for partial relaunch: EPM_BND_CAP_UNITS="arm:v4 comparator:chat"
read -ra units <<< "${EPM_BND_CAP_UNITS:-arm:v1 arm:v2 arm:v3 arm:v4 comparator:chat comparator:no_template}"
rc=0
idx=0
while [ "$idx" -lt "${#units[@]}" ]; do
  pids=()
  labels=()
  for g in $(seq 0 $((n_gpu - 1))); do
    [ "$idx" -ge "${#units[@]}" ] && break
    unit="${units[$idx]}"
    kind="${unit%%:*}"
    name="${unit##*:}"
    dev="${DEVICES[$g]}"
    echo "[launch-cap] starting ${kind} ${name} on device ${dev} ($(date -u +%FT%TZ))"
    CUDA_VISIBLE_DEVICES="$dev" uv run python scripts/issue1345_boundary_ablation_capture.py \
      --"$kind" "$name" --gpu-id "$dev" > "logs/i1345_bnd_cap_${name}.log" 2>&1 &
    pids+=("$!")
    labels+=("$unit")
    idx=$((idx + 1))
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}"
    arc=$?
    echo "[launch-cap] ${labels[$j]} finished rc=${arc} ($(date -u +%FT%TZ))"
    if [ "$arc" -ne 0 ] && [ "$rc" -eq 0 ]; then
      rc="$arc"
    fi
  done
done

echo "[launch-cap] all units done, rc=${rc}"
exit "$rc"
