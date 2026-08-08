#!/usr/bin/env bash
# Issue #1345 on-policy-vs-injected program — on-policy ANSWER generation fan-out.
#
# One gen process per (shape x measured-model) cell, wave-scheduled over the
# allocation's OWN device list. Two hard rules the fellows SLURM lane forces
# (gotchas.md § "Fellows SLURM nodes are GPU-SHARED"):
#
#   width  On a SLURM job the node is SHARED and `nvidia-smi -L` enumerates the
#          PHYSICAL node (it ignores CUDA_VISIBLE_DEVICES entirely), so width +
#          physical ids come from the ALLOCATION env — CVD > SLURM_JOB_GPUS /
#          SLURM_STEP_GPUS > SLURM_GPUS_ON_NODE (count only, ids assumed
#          0..N-1), clamped to SLURM_GPUS_ON_NODE, FAIL LOUD when a SLURM job
#          exposes none of the three. Device enumeration sizes width ONLY on the
#          exclusive-host lanes (RunPod / GCE).
#   memory Every usable device is additionally free-memory filtered here, and the
#          gen script itself live-probes `gpu_memory_utilization` per cell
#          (issue1345_onpolicy_answers_gen.resolve_vllm_util) — a hardcoded
#          fraction crashes EngineCore at init on a shared node (#1902 crash 1).
#
# Each cell pins CUDA_VISIBLE_DEVICES in the LAUNCHER env (gotchas.md CVD
# family) so parallel cells cannot co-locate on one physical GPU.
#
# The matched-n allowlist + track-S corpus are VARIANT-INDEPENDENT (the same
# 4,724 shared conv ids for every cell), so ONE staged dir is shared via
# --matched-dir/--dl-dir rather than a per-variant prefetch (which would copy a
# 2.24 GB chat-store per cell). story_slot needs no matched-n at all.
#
# BLOCKS until every cell finishes; exits with the first non-zero cell rc
# (rc=21 = a cell's yield-floor halt — the other cells still run to completion
# and persist their uploads).
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT" || exit 3
mkdir -p logs

if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

# Shared staged inputs. Default is the ROUND's own variant dir, which exists on
# the fellows scratch; a VM-only character-variant dir (data/issue_1345/char_*)
# is NOT present there, so defaulting to one would fail on the production lane.
# Override with EPM_I1345_STAGED_DIR when staging elsewhere.
STAGED="${EPM_I1345_STAGED_DIR:-data/issue_1345/story_boundary_ablation}"
N_ROWS="${EPM_I1345_N_ROWS:-0}"          # 0 = whole filtered pool
EXTRA_ARGS="${EPM_I1345_EXTRA_ARGS:-}"   # e.g. --smoke, --skip-upload

# ---------------------------------------------------------------------------
# Device resolution — allocation FIRST on SLURM, enumeration only off-SLURM
# ---------------------------------------------------------------------------
split_ids() { echo "${1//,/ }"; }

if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ALLOC=($(split_ids "$CUDA_VISIBLE_DEVICES")); SRC="slurm-cvd"
  elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
    ALLOC=($(split_ids "$SLURM_JOB_GPUS")); SRC="slurm-job-gpus"
  elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
    ALLOC=($(split_ids "$SLURM_STEP_GPUS")); SRC="slurm-step-gpus"
  elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    ALLOC=(); for i in $(seq 0 $((SLURM_GPUS_ON_NODE - 1))); do ALLOC+=("$i"); done
    SRC="slurm-count-ids-assumed-0..N-1"
  else
    echo "[op-launch] SLURM job exposes none of CUDA_VISIBLE_DEVICES / SLURM_JOB_GPUS /" >&2
    echo "[op-launch] SLURM_STEP_GPUS / SLURM_GPUS_ON_NODE — refusing to fall back to the" >&2
    echo "[op-launch] physical nvidia-smi count on a SHARED node (#1902 crash 1)" >&2
    exit 3
  fi
  # Clamp to the requested width (the allocation is authoritative over id lists).
  if [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "${#ALLOC[@]}" -gt "$SLURM_GPUS_ON_NODE" ]; then
    ALLOC=("${ALLOC[@]:0:$SLURM_GPUS_ON_NODE}"); SRC="${SRC}-clamped"
  fi
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  ALLOC=($(split_ids "$CUDA_VISIBLE_DEVICES")); SRC="env-cvd"
else
  mapfile -t ALLOC < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  SRC="detected"
fi

# Free-memory filter: skip any allocated device another tenant is holding, so a
# cell never lands on a device where EngineCore would refuse at init.
MIN_FREE_MIB="${EPM_I1345_MIN_FREE_MIB:-30000}"
DEVICES=()
for d in "${ALLOC[@]}"; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$d" 2>/dev/null | head -1)"
  if [ -n "${free_mib:-}" ] && [ "$free_mib" -ge "$MIN_FREE_MIB" ]; then
    DEVICES+=("$d")
  else
    echo "[op-launch] skipping device ${d}: ${free_mib:-unreadable} MiB free (< ${MIN_FREE_MIB})" >&2
  fi
done
n_gpu="${#DEVICES[@]}"
echo "[op-launch] device source=${SRC} allocated=${ALLOC[*]:-none} usable=${DEVICES[*]:-none}"
if [ "$n_gpu" -lt 1 ]; then
  echo "[op-launch] no usable GPUs (every allocated device below ${MIN_FREE_MIB} MiB free)" >&2
  exit 3
fi

# ---------------------------------------------------------------------------
# Cells: variant|shape|model|character
#
# story_slot pins the character to the V1 anchor's own (issue1345_boundary_
# ablation_gen.ROUND_CHARACTER); the gen script asserts it. bare_text/chat never
# put the character in the prompt, but it DOES enter the bundle fingerprint, so
# it must stay CONSTANT per cell across relaunches or a resume refuses.
# ---------------------------------------------------------------------------
CELLS=(
  "onpolicy_answers_ntpl_instruct|bare_text|instruct|ARIA"
  "onpolicy_answers_ntpl_base|bare_text|pretrained|ARIA"
  "onpolicy_answers_chat_base|chat|pretrained|ARIA"
  "onpolicy_answers_slot_base|story_slot|pretrained|Assistant"
)
if [ -n "${EPM_I1345_CELLS:-}" ]; then
  IFS=';' read -ra CELLS <<< "$EPM_I1345_CELLS"
  echo "[op-launch] cell list overridden (${#CELLS[@]} cells)"
fi

# ---------------------------------------------------------------------------
# Staged-input pre-flight — fail BEFORE any cell builds an engine
# ---------------------------------------------------------------------------
# The comparator shapes (bare_text / chat) join the matched-n allowlist against
# the parent corpus, so a wrong or unstaged --matched-dir otherwise surfaces only
# after a cell has already loaded a 7B model. story_slot reads the sha-pinned V1
# bundle from HF instead and needs no allowlist. Checked here, once, for the
# realized cell list — a burned provision costs far more than this probe.
needs_matched=0
for spec in "${CELLS[@]}"; do
  IFS='|' read -r _v shape _m _c <<< "$spec"
  [ "$shape" != "story_slot" ] && needs_matched=1
done
if [ "$needs_matched" -eq 1 ]; then
  allowlist="$STAGED/matched_n/matched_subsets_parent.json"
  if [ ! -f "$allowlist" ]; then
    echo "[op-launch] FATAL: matched-n allowlist missing: $allowlist" >&2
    echo "[op-launch] the bare_text/chat cells join it against the parent corpus." >&2
    echo "[op-launch] Either point EPM_I1345_STAGED_DIR at a dir that has" >&2
    echo "[op-launch] matched_n/matched_subsets_parent.json, or stage it first:" >&2
    echo "[op-launch]   EPM_I1345_VARIANT=<variant> uv run python \\" >&2
    echo "[op-launch]     scripts/issue1345_prefetch_reuse.py --smoke --stems instruct_chat_s" >&2
    echo "[op-launch] Candidates present on this host:" >&2
    for cand in data/issue_1345/*/matched_n/matched_subsets_parent.json; do
      [ -f "$cand" ] && echo "[op-launch]   ${cand%/matched_n/*}" >&2
    done
    exit 3
  fi
  echo "[op-launch] staged inputs OK: $allowlist"
fi
mkdir -p "$STAGED/hf_dl"

# Zero-GPU pre-launch check (the launcher sibling of the gen script's
# --verify-pool): resolve devices, run the staged-input pre-flight, print the
# resolution and exit WITHOUT building any engine. Placed after the pre-flight so
# it answers "would this launch succeed?", not merely "which devices resolved" —
# and so both pre-flight branches are exercisable as the SHIPPED code.
if [ -n "${EPM_I1345_RESOLVE_ONLY:-}" ]; then
  echo "RESOLVED source=${SRC} alloc=${ALLOC[*]:-none} usable=${DEVICES[*]:-none} n=${n_gpu}"
  echo "PREFLIGHT staged_dir=${STAGED} needs_matched=${needs_matched} cells=${#CELLS[@]}"
  exit 0
fi

run_cell() {
  local spec="$1" dev="$2"
  local variant shape model character extra
  IFS='|' read -r variant shape model character <<< "$spec"
  # story_slot reads the V1 kept-stories bundle, never the matched-n allowlist.
  extra=""
  if [ "$shape" != "story_slot" ]; then extra="--matched-dir $STAGED/matched_n"; fi
  # Thread caps: pod/SLURM GPUs are dedicated per cell, but these processes also
  # tokenize the whole pool on CPU — leave the caps to the caller's env.
  env EPM_I1345_VARIANT="$variant" EPM_STORY_CHARACTER_NAME="$character" \
      CUDA_VISIBLE_DEVICES="$dev" \
    uv run python scripts/issue1345_onpolicy_answers_gen.py \
      --shape "$shape" --model "$model" \
      --dl-dir "$STAGED/hf_dl" $extra \
      --n-rows "$N_ROWS" $EXTRA_ARGS
}

rc=0
idx=0
while [ "$idx" -lt "${#CELLS[@]}" ]; do
  pids=(); labels=()
  for g in $(seq 0 $((n_gpu - 1))); do
    [ "$idx" -ge "${#CELLS[@]}" ] && break
    spec="${CELLS[$idx]}"; variant="${spec%%|*}"; dev="${DEVICES[$g]}"
    echo "[op-launch] starting ${variant} on device ${dev} ($(date -u +%FT%TZ))"
    run_cell "$spec" "$dev" > "logs/i1345_onpolicy_${variant}.log" 2>&1 &
    pids+=("$!"); labels+=("$variant"); idx=$((idx + 1))
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}"; arc=$?
    echo "[op-launch] ${labels[$j]} finished rc=${arc} ($(date -u +%FT%TZ))"
    if [ "$arc" -ne 0 ] && [ "$rc" -eq 0 ]; then rc="$arc"; fi
  done
done

echo "[op-launch] all cells done, rc=${rc}"
exit "$rc"
