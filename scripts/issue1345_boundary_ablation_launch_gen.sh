#!/usr/bin/env bash
# Issue #1345 story-boundary-ablation — generation launcher.
# Runs arms v2/v3/v4 in parallel, one vLLM process per GPU, wave-scheduled off
# the REALIZED visible GPU count (#1121 re-shard contract: a degraded 2-wide
# rung packs the arms into waves instead of oversubscribing a GPU). BLOCKS
# until every arm finishes (SLURM-lane contract, #601) and exits with the
# first non-zero arm rc (rc=21 = an arm's yield-floor halt; the other arms
# still run to completion and persist their uploads).
set -uo pipefail

export EPM_I1345_VARIANT=story_boundary_ablation
export EPM_STORY_CHARACTER_NAME=Assistant

mkdir -p logs

# Stage the pinned parent matched-n allowlist (gen's load_paired_pool asserts
# it; track_s.jsonl self-stages from HF, the allowlist does not). --smoke
# bounds the stem side-download to one shard; the allowlist download is full
# and pinned either way. Crash 15764: data/issue_1345/** is not rsynced to
# the SLURM scratch clone, so this MUST run on the compute host.
if ! uv run python scripts/issue1345_prefetch_reuse.py --smoke --stems instruct_chat_s \
  > logs/i1345_bnd_prefetch.log 2>&1; then
  echo "[launch-gen] prefetch_reuse FAILED (allowlist staging) — see logs/i1345_bnd_prefetch.log" >&2
  tail -20 logs/i1345_bnd_prefetch.log >&2
  exit 4
fi
echo "[launch-gen] allowlist staged (prefetch_reuse rc=0)"

# Device selection (crash-fix 15771): NEVER export absolute CVD indices on a
# shared SLURM node — charmander has no device confinement, SLURM assigns
# specific GPUs via a pre-set CUDA_VISIBLE_DEVICES, and absolute 0/1/2
# clobbered it onto other users' occupied devices (vLLM died at 19 GiB free).
# Index INTO the pre-set CVD list when present; else fall back to all local
# GPUs. Either way keep only devices with >= 120000 MiB free (vLLM at 0.85
# util on H200 needs ~119 GiB); waves absorb a shortfall.
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
    echo "[launch-gen] skipping device ${d}: only ${free_mib:-?} MiB free" >&2
  fi
done
n_gpu="${#DEVICES[@]}"
echo "[launch-gen] allocated devices: ${alloc[*]:-none}; usable (>=120 GiB free): ${DEVICES[*]:-none}"
if [ "$n_gpu" -lt 1 ]; then
  echo "[launch-gen] no usable GPUs (all below the free-memory floor)" >&2
  exit 3
fi

# Arms override (partial relaunch after a per-arm failure): EPM_BND_ARMS="v3 v4"
read -ra arms <<< "${EPM_BND_ARMS:-v2 v3 v4}"
rc=0
idx=0
while [ "$idx" -lt "${#arms[@]}" ]; do
  pids=()
  labels=()
  for g in $(seq 0 $((n_gpu - 1))); do
    [ "$idx" -ge "${#arms[@]}" ] && break
    arm="${arms[$idx]}"
    dev="${DEVICES[$g]}"
    echo "[launch-gen] starting arm ${arm} on device ${dev} ($(date -u +%FT%TZ))"
    CUDA_VISIBLE_DEVICES="$dev" uv run python scripts/issue1345_boundary_ablation_gen.py \
      --arm "$arm" > "logs/i1345_bnd_gen_${arm}.log" 2>&1 &
    pids+=("$!")
    labels+=("$arm")
    idx=$((idx + 1))
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}"
    arc=$?
    echo "[launch-gen] arm ${labels[$j]} finished rc=${arc} ($(date -u +%FT%TZ))"
    if [ "$arc" -ne 0 ] && [ "$rc" -eq 0 ]; then
      rc="$arc"
    fi
  done
done

echo "[launch-gen] all arms done, rc=${rc}"
exit "$rc"
