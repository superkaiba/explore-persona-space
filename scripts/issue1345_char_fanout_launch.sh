#!/usr/bin/env bash
# Issue #1345 on-policy-vs-injected program — 16-cell character-story fan-out.
# One gen process per (character x mode x measured-model) tuple, wave-scheduled
# over the SLURM-allocated device list (index INTO the pre-set CVD allocation;
# never absolute indices — the 15771 lesson). Each cell exports its own variant
# env, stages its variant-scoped matched-n allowlist (prefetch --smoke), then
# runs the character-parametric paired-story generator (commit 00a6f829e88a's
# registry; personas verbatim from issue1310_common.PERSONAS via commit
# ec3c743d0b's persona-intro seam). BLOCKS until every cell finishes; exits
# with the first non-zero cell rc (rc=21 = a cell's yield-floor halt; other
# cells run to completion and persist their uploads).
set -uo pipefail

mkdir -p logs

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
    echo "[char-fanout] skipping device ${d}: only ${free_mib:-?} MiB free" >&2
  fi
done
n_gpu="${#DEVICES[@]}"
echo "[char-fanout] allocated: ${alloc[*]:-none}; usable (>=120 GiB free): ${DEVICES[*]:-none}"
if [ "$n_gpu" -lt 1 ]; then
  echo "[char-fanout] no usable GPUs" >&2
  exit 3
fi

# Tuple table (variant|Label|persona desc|model|opflag) — verbatim from the
# item-1 report; descs verbatim from issue1310_common.PERSONAS.
HELIOS_DESC='a calm, precise artificial intelligence'
WREN_DESC='a warm, endlessly helpful assistant who patiently helps anyone who asks'
DANA_DESC='an ordinary, unremarkable everyday person'
VEX_DESC='a theatrical, scheming villain who delights in menace'
CELLS=(
  "char_helios|HELIOS|${HELIOS_DESC}|instruct|"
  "char_helios_base|HELIOS|${HELIOS_DESC}|pretrained|"
  "char_helios_op|HELIOS|${HELIOS_DESC}|instruct|--op-powered"
  "char_helios_op_base|HELIOS|${HELIOS_DESC}|pretrained|--op-powered"
  "char_wren|Wren|${WREN_DESC}|instruct|"
  "char_wren_base|Wren|${WREN_DESC}|pretrained|"
  "char_wren_op|Wren|${WREN_DESC}|instruct|--op-powered"
  "char_wren_op_base|Wren|${WREN_DESC}|pretrained|--op-powered"
  "char_dana|Dana|${DANA_DESC}|instruct|"
  "char_dana_base|Dana|${DANA_DESC}|pretrained|"
  "char_dana_op|Dana|${DANA_DESC}|instruct|--op-powered"
  "char_dana_op_base|Dana|${DANA_DESC}|pretrained|--op-powered"
  "char_vex|Vex|${VEX_DESC}|instruct|"
  "char_vex_base|Vex|${VEX_DESC}|pretrained|"
  "char_vex_op|Vex|${VEX_DESC}|instruct|--op-powered"
  "char_vex_op_base|Vex|${VEX_DESC}|pretrained|--op-powered"
)

run_cell() {
  local spec="$1" dev="$2"
  local variant label desc model opflag
  IFS='|' read -r variant label desc model opflag <<< "$spec"
  env EPM_I1345_VARIANT="$variant" EPM_STORY_CHARACTER_NAME="$label" \
    EPM_I1345_PERSONA_DESC="$desc" CUDA_VISIBLE_DEVICES="$dev" bash -c '
      set -e
      uv run python scripts/issue1345_prefetch_reuse.py --smoke --stems instruct_chat_s
      uv run python scripts/issue1345_gen_stories_paired.py --model '"$model"' '"$opflag"'
    '
}

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
    echo "[char-fanout] starting ${variant} on device ${dev} ($(date -u +%FT%TZ))"
    run_cell "$spec" "$dev" > "logs/i1345_char_${variant}.log" 2>&1 &
    pids+=("$!")
    labels+=("$variant")
    idx=$((idx + 1))
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}"
    arc=$?
    echo "[char-fanout] ${labels[$j]} finished rc=${arc} ($(date -u +%FT%TZ))"
    if [ "$arc" -ne 0 ] && [ "$rc" -eq 0 ]; then
      rc="$arc"
    fi
  done
done

echo "[char-fanout] all cells done, rc=${rc}"
exit "$rc"
