#!/usr/bin/env bash
# issue-1739 compose-multiseed (cms) box wrapper — plan v26 §4 leg 1.
#
# ONE (behavior, seed-half) per box: EPM_I1739_BEHAVIORS x EPM_I1739_CMS_HALF
# (s02 = seeds 0 1 2, s34 = seeds 3 4; 6 cpu-bigmem boxes total). Phases:
#   stage         capture-store tars (sequential download -> untar -> rm),
#                 dv_dataset, #1092 U-store (idempotent) — rc=3 on failure
#   stage_done    sentinel + [phase=stage_done] breadcrumb (the cross-leg
#                 serialization hook: the orchestrator gates the sibling
#                 half / leg-2 a2fix launches on it, plan §9)
#   fits core     compose-only replicate grid (f_u x f_l x L anchors x seeds)
#                 under the gate-1 wrapper (first-cell wall fence -> rc=6;
#                 sampled-RSS cap -> rc=8; both designed halts with report
#                 JSONs — never a bare rc=1)
#   fits ablation f_u dose grid at L=2500, f_l=0 (same out-root; the fits CLI
#                 sidecars are append-grain + merge-on-write, so the two
#                 invocations compose instead of clobbering)
#   upload        per-box HF self-upload (newarm_box.py upload — fail-loud
#                 whole-dir upload_folder + exact-set verify) — rc=4
#   results       conforming results sentinel (epm:results, or
#                 epm:smoke-result under EPM_I1739_CMS_SMOKE) -> [phase=done]
#
# Designed exit codes: 2 bad env, 3 stage, 4 upload, 6 gate-1 wall halt,
# 8 gate-1 RSS halt, 9 in-process mem_guard refusal (propagated).
# Counts-only logging; no corpus content printed.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"
# uv run python does NOT auto-load .env; pods carry ./.env from bootstrap.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

B="${EPM_I1739_BEHAVIORS:?set EPM_I1739_BEHAVIORS to ONE of evil|sycophancy|hallucination}"
case "$B" in
  *" "*) echo "[cms] FATAL: one behavior per cms box (got '$B')" >&2; exit 2 ;;
  evil | sycophancy | hallucination) ;;
  *) echo "[cms] FATAL: unknown behavior '$B'" >&2; exit 2 ;;
esac
HALF="${EPM_I1739_CMS_HALF:?set EPM_I1739_CMS_HALF to s02|s34}"
case "$HALF" in
  s02) SEEDS_DEFAULT="0 1 2" ;;
  s34) SEEDS_DEFAULT="3 4" ;;
  *) echo "[cms] FATAL: unknown half '$HALF' (s02|s34)" >&2; exit 2 ;;
esac
SEEDS="${EPM_I1739_CMS_SEEDS:-$SEEDS_DEFAULT}"
SMOKE="${EPM_I1739_CMS_SMOKE:-}"

# Grid knobs (smoke overrides; production defaults = plan §4 cell table).
BUDGETS="${EPM_I1739_CMS_BUDGETS:-250 2500 8000}"
AB_BUDGETS="${EPM_I1739_CMS_AB_BUDGETS:-2500}"
AB_F_U="${EPM_I1739_CMS_AB_F_U:-0.1 0.25 0.75 1.0}"
U_SIZE="${EPM_I1739_CMS_U_SIZE:-5000}"
LAYERS="${EPM_I1739_CMS_LAYERS:-}" # empty = all 28
ARMS="${EPM_I1739_CMS_ARMS:-arm2_ctx_native arm4_ridge_ctx arm6_map_proj_e1 arm7_map_ridge_pred arm13_shuffled_map}"

# Smoke diverts EVERY output root (never the canonical committed paths);
# inputs (data/issue_1739/...) stay put in both modes.
if [ -n "$SMOKE" ]; then
  SMOKE_BASE="${EPM_I1739_CMS_SMOKE_BASE:-/tmp/issue-1739-cms-smoke}"
  RESULTS_BASE="$SMOKE_BASE/eval_results/issue_1739"
  SENT_DIR="${EPM_I1739_CMS_SENTINEL_DIR:-$SMOKE_BASE/sentinels}"
  TENSORS_ROOT="${EPM_I1739_CMS_TENSORS_ROOT:-$SMOKE_BASE/analysis_tensors/issue_1739}"
else
  RESULTS_BASE="eval_results/issue_1739"
  SENT_DIR="${EPM_I1739_CMS_SENTINEL_DIR:-/workspace/logs}"
  TENSORS_ROOT="${EPM_I1739_CMS_TENSORS_ROOT:-analysis_tensors/issue_1739}"
fi
OUT_ROOT="$RESULTS_BASE/compose_multiseed/$B/$HALF" # per-half out-root (r1 Must-Fix)
DV_JSON="$RESULTS_BASE/dv_dataset/$B/labeling.json"
# Plan v26 §4/§6.5/§9 producer/consumer contract — the P3 harvest stages from
# exactly this prefix (r1 reconciler MANDATORY fix: cms-hf-prefix-mismatch).
HF_PREFIX="issue1739_ctxmap/compose_multiseed/$B/$HALF"
GPU_H_BUDGET="${EPM_I1739_CMS_GPU_H_BUDGET:-0.0}" # CPU-only round (cpu-bigmem boxes)

# Gate-1 config (plan §7 gate 1). RSS cap: 100 GB on a <=160 GB box, else
# 200 GB. Wall fence: first-cell wall x total planned cells vs
# GATE1_MULT x (per-seed plan wall x n_seeds); per-seed basis = plan §9
# (evil 2.1 h/seed, syco/hall 2.7 h/seed on the cpu-bigmem shape).
set -- $SEEDS
N_SEEDS=$#
if [ -n "${EPM_I1739_CMS_RSS_CAP_GB:-}" ]; then
  RSS_CAP_GB="$EPM_I1739_CMS_RSS_CAP_GB"
else
  MEMTOTAL_KB=$(awk '/^MemTotal/{print $2}' /proc/meminfo)
  if [ "$MEMTOTAL_KB" -le $((160 * 1048576)) ]; then RSS_CAP_GB=100; else RSS_CAP_GB=200; fi
fi
case "$B" in
  evil) PER_SEED_H_DEFAULT=2.1 ;;
  *) PER_SEED_H_DEFAULT=2.7 ;;
esac
PER_SEED_H="${EPM_I1739_CMS_PER_SEED_H:-$PER_SEED_H_DEFAULT}"
GATE1_MULT="${EPM_I1739_CMS_GATE1_MULT:-2}"
PLAN_WALL_H=$(awk -v p="$PER_SEED_H" -v n="$N_SEEDS" 'BEGIN{printf "%.3f", p * n}')
# 13 = per-variant-per-seed cell ceiling (core 9 + ablation 4; evil's 2
# designed skips make this conservative, absorbed by the x2 fence).
TOTAL_CELLS=$((13 * 2 * N_SEEDS))

mkdir -p "$SENT_DIR" "$OUT_ROOT"
PID_FILE="${EPM_I1739_CMS_PID_FILE:-$SENT_DIR/issue-1739-cms-${B}-${HALF}.pid}"
printf '%s\n' "$$" > "$PID_FILE.tmp" && mv "$PID_FILE.tmp" "$PID_FILE"

# Shared-VM thread caps for a VM smoke ONLY; pod boxes keep full width.
CAPS=()
if [ -d /mnt/eps-data ] || [ "$(hostname 2>/dev/null || true)" = "cia-benchmark-vm" ]; then
  CAPS=(env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2)
fi

echo "[cms] start $(date -u +%FT%TZ) behavior=$B half=$HALF seeds='$SEEDS'" \
  "budgets='$BUDGETS' ab_budgets='$AB_BUDGETS' ab_f_u='$AB_F_U' u_size=$U_SIZE" \
  "smoke=${SMOKE:-0} rss_cap_gb=$RSS_CAP_GB plan_wall_h=$PLAN_WALL_H" \
  "gate1_mult=$GATE1_MULT total_cells=$TOTAL_CELLS out_root=$OUT_ROOT"

write_phase_sentinel() {
  # write_phase_sentinel <phase> [<status> <rc>] — conforming epm:progress
  # sentinel (poll_pipeline._SENTINEL_REQUIRED_KEYS; schema pinned by
  # tests/test_issue1739_wiring.py::test_sentinel_conformance).
  local phase="$1" status="${2:-ok}" rc="${3:-0}"
  "${CAPS[@]}" uv run python -c "
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.experiments.issue_1739 import sentinels
sentinels.write_phase_sentinel('$SENT_DIR', '$phase', status='$status', rc=$rc)
"
}

write_gate1_report() {
  # write_gate1_report <rss|wall> <tag> <peak_rss_kb> <first_cell_s> <projected_h>
  local kind="$1" tag="$2" peak_kb="${3:-0}" first_s="${4:-null}" proj_h="${5:-null}"
  printf '{"gate": "gate1_%s", "invocation": "%s", "behavior": "%s", "half": "%s", "rss_cap_gb": %s, "peak_rss_kb": %s, "first_cell_s": %s, "projected_wall_h": %s, "plan_wall_h": %s, "gate1_mult": %s, "total_cells": %s, "ts": "%s"}\n' \
    "$kind" "$tag" "$B" "$HALF" "$RSS_CAP_GB" "$peak_kb" "$first_s" "$proj_h" \
    "$PLAN_WALL_H" "$GATE1_MULT" "$TOTAL_CELLS" "$(date -u +%FT%TZ)" \
    > "$OUT_ROOT/gate1_${kind}_report.json"
  echo "[cms][gate1] $kind report -> $OUT_ROOT/gate1_${kind}_report.json" >&2
}

run_fits_gated() {
  # run_fits_gated <core|ablation> <fits argv...> — launch the fits CLI in
  # its own process group and monitor it (~20 s cadence):
  #   * summed VmRSS of the group > RSS_CAP_GB -> kill group, report, rc=8
  #   * (core, non-smoke, first new cells.jsonl row) first-cell wall x
  #     TOTAL_CELLS > GATE1_MULT x PLAN_WALL_H -> kill group, report, rc=6
  # Otherwise returns the fits CLI's own rc (9 = in-process mem_guard halt).
  local tag="$1"
  shift
  local cells="$OUT_ROOT/arm_results/percell/cells.jsonl"
  local pre_n=0
  [ -f "$cells" ] && pre_n=$(wc -l < "$cells")
  local t0
  t0=$(date +%s)
  setsid "${CAPS[@]}" uv run python scripts/issue1739_fits.py "$@" &
  local pid=$!
  local peak_kb=0 first_cell_s="" total_kb p v n now
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
    kill -0 "$pid" 2>/dev/null || break
    total_kb=0
    # Sum live VmRSS over the setsid process GROUP (uv wrapper + python
    # descendants; a bare \$! under `uv run` is the wrapper, not python).
    for p in $(pgrep -g "$pid" 2>/dev/null || true); do
      v=$(awk '/^VmRSS/{print $2}' "/proc/$p/status" 2>/dev/null || true)
      total_kb=$((total_kb + ${v:-0}))
    done
    [ "$total_kb" -gt "$peak_kb" ] && peak_kb=$total_kb
    if [ "$total_kb" -gt $((RSS_CAP_GB * 1048576)) ]; then
      echo "[cms][gate1] RSS breach (${total_kb} kB > ${RSS_CAP_GB} GiB cap) — designed halt" >&2
      kill -TERM -- "-$pid" 2>/dev/null || true
      sleep 10
      kill -KILL -- "-$pid" 2>/dev/null || true
      write_gate1_report rss "$tag" "$peak_kb" "" ""
      write_phase_sentinel "cms_${B}_${HALF}_gate1_rss_halt" halt 8
      echo "[phase=gate1_rss_halt]"
      return 8
    fi
    if [ "$tag" = core ] && [ -z "$SMOKE" ] && [ -z "$first_cell_s" ] && [ -f "$cells" ]; then
      n=$(wc -l < "$cells" 2>/dev/null || echo "$pre_n")
      if [ "$n" -gt "$pre_n" ]; then
        now=$(date +%s)
        first_cell_s=$((now - t0))
        local verdict proj
        verdict=$(awk -v s="$first_cell_s" -v n="$TOTAL_CELLS" -v m="$GATE1_MULT" \
          -v w="$PLAN_WALL_H" 'BEGIN{p = s * n / 3600.0
            printf "%s %.3f", (p > m * w) ? "halt" : "ok", p}')
        proj="${verdict#* }"
        echo "[cms][gate1] first cell ${first_cell_s}s -> projected ${proj}h" \
          "(fence ${GATE1_MULT}x${PLAN_WALL_H}h over $TOTAL_CELLS cells)"
        if [ "${verdict%% *}" = halt ]; then
          echo "[cms][gate1] WALL breach — designed halt (re-size/width re-eval, never a blind retry)" >&2
          kill -TERM -- "-$pid" 2>/dev/null || true
          sleep 10
          kill -KILL -- "-$pid" 2>/dev/null || true
          write_gate1_report wall "$tag" "$peak_kb" "$first_cell_s" "$proj"
          write_phase_sentinel "cms_${B}_${HALF}_gate1_wall_halt" halt 6
          echo "[phase=gate1_wall_halt]"
          return 6
        fi
      fi
    fi
  done
  local rc=0
  wait "$pid" || rc=$?
  echo "[cms][gate1] $tag invocation done rc=$rc peak_rss_kb=$peak_kb"
  return "$rc"
}

# ---------------------------------------------------------------- stage ----
echo "[phase=stage]"
stage_rc=0
(
  set -euo pipefail
  # 0. Disk-headroom preflight BEFORE the first download (>= ~1.5x the
  #    projected staging bytes; designed halt, never a mid-extraction ENOSPC).
  MIN_DISK_GB="${EPM_I1739_CMS_MIN_DISK_GB:-60}"
  avail_kb=$(df -Pk data 2>/dev/null | awk 'NR==2{print $4}' || df -Pk . | awk 'NR==2{print $4}')
  if [ "${avail_kb:-0}" -lt $((MIN_DISK_GB * 1048576)) ]; then
    echo "[cms] FATAL: disk preflight — $((avail_kb / 1048576)) GiB free <" \
      "${MIN_DISK_GB} GiB floor (EPM_I1739_CMS_MIN_DISK_GB); refusing before download" >&2
    exit 3
  fi
  echo "[cms] disk preflight ok: $((avail_kb / 1048576)) GiB free (floor ${MIN_DISK_GB} GiB)"
  # 1. capture-store tars (leg2 step-2 shape: sequential download -> untar ->
  #    rm to bound peak disk; behavior-scoped; skip-if-present — the skip is
  #    safe because extraction is ATOMIC below: untar into a temp dir, then
  #    one mv into place, so a partially extracted store never occupies the
  #    canonical path on a crashed prior attempt).
  TARS_DIR="data/issue_1739/hf_dl/store_tars"
  STORE_ROOT="data/issue_1739/store"
  mkdir -p "$TARS_DIR" "$STORE_ROOT"
  for name in "${B}_extraction" "${B}_labeling"; do
    if [ -d "$STORE_ROOT/$name" ]; then
      echo "[cms] store $name: already present, skip"
      continue
    fi
    echo "[cms] store $name: download $(date -u +%H:%M:%SZ)"
    "${CAPS[@]}" uv run python -c "
import sys
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.orchestrate import hub
from huggingface_hub import hf_hub_download
name = sys.argv[1]
hub.retry_transient(lambda: hf_hub_download(
    'superkaiba1/explore-persona-space-data',
    f'issue1739_ctxmap/capture_store/{name}/{name}.tar',
    repo_type='dataset', local_dir=sys.argv[2]), what=f'store-tar {name}')
print(f'[cms] {name}: downloaded', flush=True)
" "$name" "$TARS_DIR"
    extract_tmp="$STORE_ROOT/.extract_$name"
    rm -rf "$extract_tmp"
    mkdir -p "$extract_tmp"
    tar -xf "$TARS_DIR/issue1739_ctxmap/capture_store/$name/$name.tar" -C "$extract_tmp"
    [ -d "$extract_tmp/$name" ] || {
      echo "[cms] FATAL: tar for $name did not contain a top-level $name/ dir" >&2
      exit 3
    }
    mv "$extract_tmp/$name" "$STORE_ROOT/$name"
    rm -rf "$extract_tmp"
    rm -f "$TARS_DIR/issue1739_ctxmap/capture_store/$name/$name.tar"
    echo "[cms] store $name: unpacked ($(du -sh "$STORE_ROOT/$name" | cut -f1)); df: $(df -h --output=avail . | tail -1 | tr -d ' ')"
  done
  # 2. dv_dataset (leg2 step-4 shape; dest under RESULTS_BASE so a smoke
  #    never writes the canonical committed path).
  "${CAPS[@]}" uv run python -c "
import shutil, sys
from pathlib import Path
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.orchestrate import hub
from huggingface_hub import hf_hub_download
b, dst = sys.argv[1], Path(sys.argv[2])
if dst.exists():
    print(f'[cms] dv_dataset/{b}: already present, skip', flush=True)
else:
    p = hub.retry_transient(lambda: hf_hub_download(
        'superkaiba1/explore-persona-space-data',
        f'issue1739_ctxmap/judge/dv_dataset/{b}/labeling.json',
        repo_type='dataset', local_dir='data/issue_1739/hf_dl/dv_dl'), what=f'dv {b}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    print(f'[cms] dv_dataset/{b}: staged', flush=True)
" "$B" "$DV_JSON"
  # 3. #1092 U-store slice (idempotent short-circuit when already loadable
  #    for the requested kinds x layers).
  "${CAPS[@]}" uv run python -c "
import sys
from pathlib import Path
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.experiments.issue_1739 import store_io
layers = tuple(int(t) for t in sys.argv[1].split()) if sys.argv[1].strip() else tuple(range(28))
store_io.stage_u_store(
    Path('data/issue_1739/hf_dl/u_store'), ('prefix_end', 'context_end', 't1'), layers)
print(f'[cms] u_store staged ({len(layers)} layers)', flush=True)
" "$LAYERS"
) || stage_rc=$?
if [ "$stage_rc" -ne 0 ]; then
  echo "[cms] FATAL: stage failed rc=$stage_rc" >&2
  write_phase_sentinel "cms_${B}_${HALF}_stage" fail 3 || echo "[cms] WARN: failure-sentinel write failed (phase rc preserved)" >&2
  exit 3
fi
# Cross-leg serialization hook (plan §9): the orchestrator gates the sibling
# half's / leg-2 a2fix's staging-heavy launch on this sentinel + breadcrumb.
write_phase_sentinel "cms_${B}_${HALF}_stage_done"
echo "[phase=stage_done]"

# ------------------------------------------------------------- fits core ---
FITS_COMMON=(
  --behavior "$B"
  --labeled-store "data/issue_1739/store/${B}_labeling"
  --dv-json "$DV_JSON"
  --u-store data/issue_1739/hf_dl/u_store
  --e1-store "data/issue_1739/store/${B}_extraction"
  --out-root "$OUT_ROOT"
  --tensors-root "$TENSORS_ROOT"
  --device cpu
  --config config_a
  --regimes e1
  --compose --compose-only --compose-replicates
  --compose-u-size "$U_SIZE"
  --draws 0
  --seeds $SEEDS
  --arms $ARMS
)
if [ -n "$LAYERS" ]; then
  FITS_COMMON+=(--layers $LAYERS)
fi

echo "[cms] fits core grid $(date -u +%FT%TZ)"
set +e
run_fits_gated core "${FITS_COMMON[@]}" --budgets $BUDGETS
frc=$?
set -e
case "$frc" in
  0) ;;
  6 | 8) exit "$frc" ;; # gate-1 designed halts (report + sentinel already written)
  9)
    echo "[cms] RSS-GUARD REFUSED (rc=9, in-process mem_guard): see" \
      "$OUT_ROOT/rss_guard_report.json (designed halt; re-size, never a blind retry)" >&2
    write_phase_sentinel "cms_${B}_${HALF}_fits_core" halt 9 || echo "[cms] WARN: failure-sentinel write failed (phase rc preserved)" >&2
    exit 9
    ;;
  *)
    echo "[cms] FATAL: core fits rc=$frc" >&2
    write_phase_sentinel "cms_${B}_${HALF}_fits_core" fail "$frc" || echo "[cms] WARN: failure-sentinel write failed (phase rc preserved)" >&2
    exit "$frc"
    ;;
esac
write_phase_sentinel "cms_${B}_${HALF}_fits_core"
echo "[phase=fits_core]"

# Preserve the core invocation's summary before the ablation invocation
# overwrites all_arms_spearman.json (invocation-scoped writer; r2 hardening,
# cms-summary-cross-invocation-clobber). The fold reads percell/cells.jsonl,
# so this is a diagnostics-preservation copy, not a fold input.
if [ -f "$OUT_ROOT/arm_results/all_arms_spearman.json" ]; then
  cp "$OUT_ROOT/arm_results/all_arms_spearman.json" \
    "$OUT_ROOT/arm_results/all_arms_spearman.core.json"
fi

# --------------------------------------------------------- fits ablation ---
echo "[cms] fits ablation dose grid $(date -u +%FT%TZ)"
set +e
run_fits_gated ablation "${FITS_COMMON[@]}" --budgets $AB_BUDGETS \
  --f-u-grid $AB_F_U --f-l-grid 0.0
arc=$?
set -e
case "$arc" in
  0) ;;
  6 | 8) exit "$arc" ;;
  9)
    echo "[cms] RSS-GUARD REFUSED (rc=9, in-process mem_guard) on the ablation" \
      "invocation: see $OUT_ROOT/rss_guard_report.json" >&2
    write_phase_sentinel "cms_${B}_${HALF}_fits_ablation" halt 9 || echo "[cms] WARN: failure-sentinel write failed (phase rc preserved)" >&2
    exit 9
    ;;
  *)
    echo "[cms] FATAL: ablation fits rc=$arc" >&2
    write_phase_sentinel "cms_${B}_${HALF}_fits_ablation" fail "$arc" || echo "[cms] WARN: failure-sentinel write failed (phase rc preserved)" >&2
    exit "$arc"
    ;;
esac
write_phase_sentinel "cms_${B}_${HALF}_fits_ablation"
echo "[phase=fits_ablation]"

# ----------------------------------------------------------------- upload --
if [ -n "$SMOKE" ]; then
  echo "[cms] smoke: skip upload (out-root diverted to $OUT_ROOT)"
else
  echo "[cms] HF self-upload $(date -u +%FT%TZ)"
  set +e
  "${CAPS[@]}" uv run python scripts/issue1739_newarm_box.py upload \
    --pairs "$OUT_ROOT:$HF_PREFIX"
  urc=$?
  set -e
  if [ "$urc" -ne 0 ]; then
    echo "[cms] FATAL: upload rc=$urc" >&2
    write_phase_sentinel "cms_${B}_${HALF}_upload" fail 4 || echo "[cms] WARN: failure-sentinel write failed (phase rc preserved)" >&2
    exit 4
  fi
  write_phase_sentinel "cms_${B}_${HALF}_upload"
  echo "[phase=upload]"
fi

# ---------------------------------------------------------------- results --
"${CAPS[@]}" uv run python -c "
import sys
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.experiments.issue_1739 import sentinels
out_root, behavior, half, sent_dir, seeds_s, hf_prefix, gpu_h, smoke = sys.argv[1:9]
payload = sentinels.compose_cms_results_payload(
    out_root, behavior, half, [int(s) for s in seeds_s.split()],
    hf_prefix=hf_prefix, gpu_hours_budgeted=float(gpu_h))
sentinels.write_results_sentinel(sent_dir, payload, smoke=bool(smoke))
" "$OUT_ROOT" "$B" "$HALF" "$SENT_DIR" "$SEEDS" "$HF_PREFIX" "$GPU_H_BUDGET" "${SMOKE:+1}"
echo "[cms] done rc=0 $(date -u +%FT%TZ)"
echo "[phase=done]"
