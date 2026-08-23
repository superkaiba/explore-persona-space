#!/usr/bin/env bash
# Issue #2479 wide-pod (4xH100) workload wrapper: P0 smoke gate -> P1 24-cell
# story generation fan-out -> P4 capture fan-out (plan v4 §4/§9).
#
# Named by the plan §9 dispatch command:
#   dispatch_issue.py launch --issue 2479 --intent eval --gpus 4 --backend runpod \
#     --repo-branch issue-2479 --time-budget-hours 16 \
#     --workload-cmd 'bash scripts/issue2479_p1p4_launch.sh'
#
# P0 smoke-gate legs (plan §4; ALL must pass before any production cell):
#   1. guard-selftest      fill --guard-selftest (3 axis-freeze guard branches)
#   2. gen-smoke           Iris+Vex x {op, inserted}: prefetch + gen --smoke
#   3. capture-smoke       capture launcher --smoke-only over the same 4 cells
#   4. stage-sources       r4 + r4op source turnstores at the plan §10 pins
#   5. toyfit-newcell      round-produced iris smoke turnstore aliased as
#                          char_helios -> production fill path at --max-rows 8
#   6. parity-pilot        char_helios_op full-n refit vs the committed parent
#                          reference (+-0.02) — kill criterion (c); FAIL -> rc=41
#   7. subsample-pilot     char_helios_op at --max-rows 1100 (the MEASURED P5
#                          per-cell fence basis; wall recorded in the sentinel)
#   8. r4-consumer-open    char_helios (r4 ladder src) at --max-rows 1100 —
#                          consumer-opens the r4 source through the real path
#
# Pod-side reporting contract: NEVER shells out to scripts/task.py; progress =
# [phase=...] stdout breadcrumbs + envelope sentinels under /workspace/logs/.
# Resume state lives OUTSIDE the poller's issue-2479-*.json drain glob
# (.state / -done / -yieldhalt files, no .json suffix).
#
# Designed exit codes (never a bare rc=1 for a designed halt):
#   40 = a P0 smoke leg failed          41 = P0 parity gate (kill criterion c)
#   44 = P1 fatal gen failure (non-21)  46 = P1 <12 op-survivors (criterion a)
#   42/43 = P4 capture gate-1 halts (propagated from the capture launcher)
#   45 = P4 fatal capture failure       2/3 = arg / no-GPU errors
#
# Usage: bash scripts/issue2479_p1p4_launch.sh [--dry-run]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "[i2479-p1p4] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# The panel registry drives EVERY per-cell table (env absent locally => the
# committed panel; the seam-bearing ported scripts read the same env).
export EPM_I2479_CHAR_PANEL_JSON="${EPM_I2479_CHAR_PANEL_JSON:-${REPO_ROOT}/eval_results/issue_2479/panel.json}"

LOG_DIR="${SENTINEL_DIR:-/workspace/logs}"
DATA_BASE="${EPM_I2479_DATA_BASE:-/workspace/data/issue_2479}"
STAGE_ROOT="${DATA_BASE}/stage"
CACHE_DIR="${DATA_BASE}/fill_cache"
PILOT_OUT="${DATA_BASE}/p0_pilot"
SMOKE_STAGE="${DATA_BASE}/p0_smoke_stage"
P0_STATE="${LOG_DIR}/issue-2479-p0-PASS.state"

N_STORIES=1600
YIELD_FLOOR=800
SUBSAMPLE_ROWS=1100
MIN_OP_SURVIVORS=12

# --- panel-derived gen cell table (variant|Label|desc|opflag) -----------------
# Registry order, op cell then non-null inserted cell per row (the ladder-fill
# convention). desc/display_name are asserted present (gen needs both).
GEN_ROWS="$(uv run python - <<'PY'
import sys

sys.path.insert(0, "scripts")
from issue2479_char_panel import load_char_panel_env

rows = load_char_panel_env()
assert rows, "EPM_I2479_CHAR_PANEL_JSON set but loader returned no rows"
for r in rows:
    label, desc = r.get("display_name"), r.get("desc")
    assert label and desc, f"panel row {r.get('name')!r} lacks display_name/desc (gen needs both)"
    assert "|" not in desc, f"panel row {r['name']!r} desc contains '|' (table delimiter)"
    print(f"{r['variant_op']}|{label}|{desc}|--op-powered")
    if r["variant_inserted"]:
        print(f"{r['variant_inserted']}|{label}|{desc}|")
PY
)" || { echo "[i2479-p1p4] panel load failed" >&2; exit 2; }
GEN_CELLS=()
while IFS= read -r row; do
  [ -n "$row" ] && GEN_CELLS+=("$row")
done <<< "$GEN_ROWS"

# P0 smoke gen cells: Iris + Vex x {op, inserted} (plan §4 leg (i)).
SMOKE_GEN_CELLS=()
for spec in "${GEN_CELLS[@]}"; do
  v="${spec%%|*}"
  case "$v" in
    char_2479_iris|char_2479_iris_op|char_2479_vex|char_2479_vex_op) SMOKE_GEN_CELLS+=("$spec") ;;
  esac
done

gen_cell_cmd() { # spec -> the resolved inner gen command (for logs/dry-run)
  local spec="$1" variant label desc opflag
  IFS='|' read -r variant label desc opflag <<< "$spec"
  echo "uv run python scripts/issue1345_gen_stories_paired.py --model instruct ${opflag:+$opflag }--n-stories ${N_STORIES} --yield-floor ${YIELD_FLOOR}"
}

# --- --dry-run: print the resolved tables + P0/P4 commands, execute nothing ---
if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] panel=${EPM_I2479_CHAR_PANEL_JSON}"
  echo "[dry-run] P0 legs (in order): guard-selftest; gen-smoke (${#SMOKE_GEN_CELLS[@]} cells); capture-smoke (--smoke-only, iris pair + vex pair); stage-sources (r4 r4op -> ${STAGE_ROOT}); toyfit (--max-rows 8); parity-pilot (char_helios_op, tol 0.02, halt rc=41); subsample-pilot (--max-rows ${SUBSAMPLE_ROWS}); r4-consumer-open (char_helios, --max-rows ${SUBSAMPLE_ROWS})"
  echo "[dry-run] P1 gen table (${#GEN_CELLS[@]} rows; per cell: prefetch --smoke --stems instruct_chat_s, then:)"
  k=0
  for spec in "${GEN_CELLS[@]}"; do
    k=$((k + 1))
    v="${spec%%|*}"
    printf '  P1[%02d/%d] %-24s %s\n' "$k" "${#GEN_CELLS[@]}" "$v" "$(gen_cell_cmd "$spec")"
  done
  echo "[dry-run] P4: bash scripts/issue1345_char_capture_launch.sh --skip-smoke --cells <p1 survivors among the ${#GEN_CELLS[@]} cells above>"
  echo "[dry-run] sentinels: ${LOG_DIR}/issue-2479-smoke-PASS.json (P0), ${LOG_DIR}/issue-2479-p1p4-results.json (final); resume state: ${P0_STATE}, ${LOG_DIR}/issue-2479-p1-<cell>-{done,yieldhalt}, ${LOG_DIR}/issue-2479-p4-<cell>-done"
  exit 0
fi

mkdir -p logs "$LOG_DIR" "$DATA_BASE"

write_sentinel() { # filename kind gate blocks note
  local fname="$1" kind="$2" gate="$3" blocks="$4" note="$5"
  local out="${LOG_DIR}/${fname}"
  uv run python - "$out" "$kind" "$gate" "$blocks" "$note" <<'PY'
import json
import os
import sys
import time

out, kind, gate, blocks, note = sys.argv[1:6]
payload = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "gate": gate,
    "blocks_pipeline": blocks == "1",
    "note": f"{note} | ts={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
}
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(payload, f)
os.replace(tmp, out)
print(f"[sentinel] {out}", flush=True)
PY
}

# --- GPU allocation (index INTO the CVD allocation — the 15771 lesson) --------
alloc_devices() { # min_free_mib -> sets DEVICES + n_gpu
  local min_free="$1"
  local alloc=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra alloc <<< "$CUDA_VISIBLE_DEVICES"
  else
    mapfile -t alloc < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  fi
  DEVICES=()
  local d free_mib
  for d in "${alloc[@]}"; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$d" | head -1)"
    if [ "${free_mib:-0}" -ge "$min_free" ]; then
      DEVICES+=("$d")
    else
      echo "[i2479-p1p4] skipping device ${d}: only ${free_mib:-?} MiB free" >&2
    fi
  done
  n_gpu="${#DEVICES[@]}"
  echo "[i2479-p1p4] allocated: ${alloc[*]:-none}; usable (>=${min_free} MiB free): ${DEVICES[*]:-none}"
  if [ "$n_gpu" -lt 1 ]; then
    echo "[i2479-p1p4] no usable GPUs" >&2
    exit 3
  fi
}

run_gen_cell() { # spec dev smoke
  local spec="$1" dev="$2" smoke="$3"
  local variant label desc opflag
  IFS='|' read -r variant label desc opflag <<< "$spec"
  local gen_args=(--model instruct)
  [ -n "$opflag" ] && gen_args+=("$opflag")
  local smoke_n=""
  if [ "$smoke" = "1" ]; then
    gen_args+=(--smoke)
    # 12 targets so the toy-fit leg's aliased turnstore reaches the extract
    # smoke cap (8 rows) with judge-filter margin (fill fold floor: >=4).
    smoke_n=12
  else
    gen_args+=(--n-stories "$N_STORIES" --yield-floor "$YIELD_FLOOR")
  fi
  env EPM_I1345_VARIANT="$variant" EPM_STORY_CHARACTER_NAME="$label" \
    EPM_I1345_PERSONA_DESC="$desc" CUDA_VISIBLE_DEVICES="$dev" \
    ${smoke_n:+EPM_I1345_SMOKE_N_STORIES="$smoke_n"} \
    I2479_GEN_ARGS="${gen_args[*]}" bash -c '
      set -e
      uv run python scripts/issue1345_prefetch_reuse.py --smoke --stems instruct_chat_s
      # shellcheck disable=SC2086
      uv run python scripts/issue1345_gen_stories_paired.py $I2479_GEN_ARGS
    '
}

P0_TELEM=""
run_leg() { # leg_name cmd... -> appends "name=rc(wall)" to P0_TELEM; rc!=0 -> exit 40
  local name="$1"
  shift
  echo "[i2479-p1p4] P0 leg ${name}: $* ($(date -u +%FT%TZ))"
  local t0 t1 arc
  t0=$(date +%s)
  "$@"
  arc=$?
  t1=$(date +%s)
  P0_TELEM="${P0_TELEM}${name}=${arc}($((t1 - t0))s) "
  if [ "$arc" -ne 0 ]; then
    echo "[i2479-p1p4] P0 leg ${name} FAILED rc=${arc}" >&2
    write_sentinel "issue-2479-smoke-FAIL.json" "epm:smoke-result" "p0_smoke" 1 \
      "issue-2479 P0 smoke gate FAILED at leg ${name} rc=${arc} | telemetry: ${P0_TELEM}"
    exit 40
  fi
}

# =============================================================================
echo "[phase=p0_smoke]"
if [ -f "$P0_STATE" ]; then
  echo "[i2479-p1p4] P0 state file present (${P0_STATE}) — smoke gate skipped (resume)"
else
  # Leg 1: freeze-guard selftest (CPU; PASS/FAIL lines captured as telemetry).
  run_leg guard-selftest bash -o pipefail -c \
    'uv run python scripts/issue1345_story_char_ladder_fill.py --guard-selftest 2>&1 | tee logs/i2479_p0_guardselftest.log'
  guard_lines="$(grep -c 'result=PASS' logs/i2479_p0_guardselftest.log || true)"
  if grep -q 'result=FAIL' logs/i2479_p0_guardselftest.log; then
    echo "[i2479-p1p4] guard-selftest reported a FAIL branch" >&2
    write_sentinel "issue-2479-smoke-FAIL.json" "epm:smoke-result" "p0_smoke" 1 \
      "issue-2479 P0 guard-selftest FAIL branch | telemetry: ${P0_TELEM}"
    exit 40
  fi
  P0_TELEM="${P0_TELEM}guard_pass_branches=${guard_lines} "

  # Leg 2: gen smoke — Iris+Vex x both modes, one wave over the allocation.
  alloc_devices 120000
  gen_smoke_rc=0
  pids=()
  labels=()
  i=0
  for spec in "${SMOKE_GEN_CELLS[@]}"; do
    v="${spec%%|*}"
    dev="${DEVICES[$((i % n_gpu))]}"
    echo "[i2479-p1p4] P0 gen-smoke ${v} on device ${dev}"
    run_gen_cell "$spec" "$dev" 1 > "logs/i2479_p0_gen_${v}.log" 2>&1 &
    pids+=("$!")
    labels+=("$v")
    i=$((i + 1))
    # never >1 cell per device: wait out the wave when the allocation is full
    if [ "$(( i % n_gpu ))" -eq 0 ]; then
      for j in "${!pids[@]}"; do
        wait "${pids[$j]}" || { arc=$?; echo "[i2479-p1p4] gen-smoke ${labels[$j]} rc=${arc}" >&2; gen_smoke_rc="$arc"; }
      done
      pids=()
      labels=()
    fi
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}" || { arc=$?; echo "[i2479-p1p4] gen-smoke ${labels[$j]} rc=${arc}" >&2; gen_smoke_rc="$arc"; }
  done
  run_leg gen-smoke test "$gen_smoke_rc" -eq 0

  # Leg 3: capture smoke via the launcher's own --smoke-only leg (two
  # invocations so BOTH Iris and Vex smoke — the launcher's class-dedup would
  # otherwise pick only the first cell per regime x model class).
  run_leg capture-smoke-iris bash scripts/issue1345_char_capture_launch.sh \
    --smoke-only --cells char_2479_iris_op char_2479_iris
  run_leg capture-smoke-vex bash scripts/issue1345_char_capture_launch.sh \
    --smoke-only --cells char_2479_vex_op char_2479_vex

  # Leg 4: stage the P5 ladder source turnstores at the plan §10 pins (r4
  # consumer-open of the STAGED layout; the fit-path open is legs 6-8).
  run_leg stage-sources uv run python scripts/issue1345_stage_char_stories.py \
    --sources r4 r4op --dest-root "$STAGE_ROOT"

  # Leg 5: toy fit of a ROUND-PRODUCED store through the production fill path:
  # alias the iris inserted-mode smoke turnstore (stem-compatible with
  # char_helios's stories_paired format) as char_helios in a scratch stage
  # root. --pilot-outdir skips the axis-freeze guard (parent cell only).
  # Floor pre-check: run_cell_fit needs >= 4 rows (5-fold, train >= 3); the
  # turnstore has min(kept, 8) rows. Fail loud with the remedy BEFORE the fit.
  kept_n="$(wc -l < "data/issue_1345/char_2479_iris/stories/kept_stories_paired_instruct.jsonl" 2>/dev/null || echo 0)"
  if [ "${kept_n:-0}" -lt 4 ]; then
    echo "[i2479-p1p4] toyfit floor: iris smoke kept=${kept_n} < 4 — raise EPM_I1345_SMOKE_N_STORIES (wrapper default 12), wipe the iris smoke bundle + HF smoke marker, and re-run P0" >&2
    write_sentinel "issue-2479-smoke-FAIL.json" "epm:smoke-result" "p0_smoke" 1 \
      "issue-2479 P0 toyfit floor: iris smoke kept=${kept_n} < 4 rows | telemetry: ${P0_TELEM}"
    exit 40
  fi
  rm -rf "${SMOKE_STAGE}/char_helios_turnstore"
  mkdir -p "$SMOKE_STAGE"
  cp -r "data/issue_1345/char_2479_iris/turnstore_smoke" "${SMOKE_STAGE}/char_helios_turnstore"
  run_leg toyfit-newcell-store uv run python scripts/issue1345_story_char_ladder_fill.py \
    --stage cells --cells char_helios --model instruct --arm context \
    --stage-root "$SMOKE_STAGE" --cache-dir "${DATA_BASE}/p0_smoke_cache" \
    --pilot-outdir "${PILOT_OUT}/toyfit" --max-rows 8

  # Leg 6: parity pilot — full-n refit of the parent cell through the REAL
  # driver (stage -> both-arm fits -> ladder), then the +-0.02 check.
  run_leg parity-pilot-fits bash scripts/issue1345_char_phasef_driver.sh \
    --cells char_helios_op --stage-root "$STAGE_ROOT" --cache-dir "$CACHE_DIR" \
    --pilot-outdir "$PILOT_OUT"
  echo "[i2479-p1p4] P0 parity check (kill criterion (c), tol 0.02)"
  parity_out="$(uv run python scripts/issue2479_parity_check.py --pilot-dir "$PILOT_OUT" 2>&1)"
  parity_rc=$?
  echo "$parity_out"
  P0_TELEM="${P0_TELEM}parity=${parity_rc} "
  if [ "$parity_rc" -ne 0 ]; then
    summary="$(echo "$parity_out" | grep '^\[parity\] summary' | tail -1)"
    echo "[i2479-p1p4] PARITY HALT (kill criterion (c)): refit-equality outside +-0.02" >&2
    write_sentinel "issue-2479-smoke-FAIL.json" "epm:smoke-result" "p0_parity" 1 \
      "issue-2479 P0 PARITY HALT (kill criterion c) rc=${parity_rc} ${summary} | telemetry: ${P0_TELEM}"
    exit 41
  fi

  # Leg 7: subsampled-regime pilot (~1,100 rows) — the MEASURED P5 fence basis.
  t0=$(date +%s)
  run_leg subsample-pilot bash scripts/issue1345_char_phasef_driver.sh \
    --cells char_helios_op --stage-root "$STAGE_ROOT" --cache-dir "$CACHE_DIR" \
    --pilot-outdir "$PILOT_OUT" --max-rows "$SUBSAMPLE_ROWS"
  SUBSAMPLE_WALL=$(( $(date +%s) - t0 ))
  P0_TELEM="${P0_TELEM}subsample_wall_s=${SUBSAMPLE_WALL} "

  # Leg 8: r4 consumer-open through the production ladder path (char_helios
  # ladders from the r4 source staged at leg 4).
  run_leg r4-consumer-open bash scripts/issue1345_char_phasef_driver.sh \
    --cells char_helios --stage-root "$STAGE_ROOT" --cache-dir "$CACHE_DIR" \
    --pilot-outdir "$PILOT_OUT" --max-rows "$SUBSAMPLE_ROWS"

  write_sentinel "issue-2479-smoke-PASS.json" "epm:smoke-result" "p0_smoke" 0 \
    "issue-2479 P0 smoke gate PASS | telemetry: ${P0_TELEM}"
  date -u +%FT%TZ > "$P0_STATE"
  echo "[i2479-p1p4] P0 smoke gate PASS — state written to ${P0_STATE}"
fi

# =============================================================================
echo "[phase=p1_gen]"
P1_PENDING=()
for spec in "${GEN_CELLS[@]}"; do
  v="${spec%%|*}"
  if [ -f "${LOG_DIR}/issue-2479-p1-${v}-done" ]; then
    echo "[i2479-p1p4] P1 ${v}: done file present — skipped (resume)"
  elif [ -f "${LOG_DIR}/issue-2479-p1-${v}-yieldhalt" ]; then
    echo "[i2479-p1p4] P1 ${v}: yield-floor halt on a prior run — not retried (durable rc=21 record)"
  else
    P1_PENDING+=("$spec")
  fi
done
if [ "${#P1_PENDING[@]}" -gt 0 ]; then
  alloc_devices 120000
else
  n_gpu=0
fi
echo "[i2479-p1p4] P1: ${#P1_PENDING[@]}/${#GEN_CELLS[@]} cells pending (${n_gpu}-wide waves)"
p1_fatal_rc=0
idx=0
while [ "$idx" -lt "${#P1_PENDING[@]}" ]; do
  pids=()
  labels=()
  for g in $(seq 0 $((n_gpu - 1))); do
    [ "$idx" -ge "${#P1_PENDING[@]}" ] && break
    spec="${P1_PENDING[$idx]}"
    v="${spec%%|*}"
    dev="${DEVICES[$g]}"
    echo "[i2479-p1p4] P1 starting ${v} on device ${dev} ($(date -u +%FT%TZ))"
    run_gen_cell "$spec" "$dev" 0 > "logs/i2479_gen_${v}.log" 2>&1 &
    pids+=("$!")
    labels+=("$v")
    idx=$((idx + 1))
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}"
    arc=$?
    v="${labels[$j]}"
    echo "[i2479-p1p4] P1 ${v} finished rc=${arc} ($(date -u +%FT%TZ))"
    if [ "$arc" -eq 0 ]; then
      date -u +%FT%TZ > "${LOG_DIR}/issue-2479-p1-${v}-done"
    elif [ "$arc" -eq 21 ]; then
      # Designed yield-floor halt: durable record; siblings continue (plan §7).
      date -u +%FT%TZ > "${LOG_DIR}/issue-2479-p1-${v}-yieldhalt"
    else
      p1_fatal_rc="$arc"
    fi
  done
done
if [ "$p1_fatal_rc" -ne 0 ]; then
  echo "[i2479-p1p4] P1 fatal gen failure rc=${p1_fatal_rc}" >&2
  write_sentinel "issue-2479-p1p4-results.json" "epm:progress" "p1_gen" 1 \
    "issue-2479 P1 FATAL gen failure rc=${p1_fatal_rc} (non-21; see logs/i2479_gen_*.log)"
  exit 44
fi

# Kill criterion (a): the gradient needs >=12 surviving CHARACTERS; a character
# survives P1 when its OP cell generated to floor.
SURVIVORS=()
op_survivors=0
for spec in "${GEN_CELLS[@]}"; do
  v="${spec%%|*}"
  if [ -f "${LOG_DIR}/issue-2479-p1-${v}-done" ]; then
    SURVIVORS+=("$v")
    case "$v" in *_op) op_survivors=$((op_survivors + 1)) ;; esac
  fi
done
echo "[i2479-p1p4] P1 survivors: ${#SURVIVORS[@]}/${#GEN_CELLS[@]} cells (${op_survivors}/16 op cells)"
if [ "$op_survivors" -lt "$MIN_OP_SURVIVORS" ]; then
  echo "[i2479-p1p4] KILL CRITERION (a): only ${op_survivors} op-cell survivors < ${MIN_OP_SURVIVORS}" >&2
  write_sentinel "issue-2479-p1p4-results.json" "epm:progress" "p1_survivors" 1 \
    "issue-2479 P1 HALT kill criterion (a): op_survivors=${op_survivors} < ${MIN_OP_SURVIVORS} | survivors: ${SURVIVORS[*]}"
  exit 46
fi

# =============================================================================
echo "[phase=p4_capture]"
# --skip-smoke: P0 already ran the capture smoke over both regime x model
# classes (Iris + Vex); the launcher's own HF completion markers remain the
# per-cell production resume grain.
bash scripts/issue1345_char_capture_launch.sh --skip-smoke --cells "${SURVIVORS[@]}"
p4_rc=$?
if [ "$p4_rc" -eq 42 ] || [ "$p4_rc" -eq 43 ]; then
  echo "[i2479-p1p4] P4 capture gate halt rc=${p4_rc} (designed; see the launcher's own sentinel)" >&2
  write_sentinel "issue-2479-p1p4-results.json" "epm:progress" "p4_gate1" 1 \
    "issue-2479 P4 capture gate halt rc=${p4_rc} (42=gate-1 projection halt, 43=wall parse error; launcher sentinel issue-1345-char-capture-results.json has the gate report)"
  exit "$p4_rc"
fi
if [ "$p4_rc" -ne 0 ]; then
  echo "[i2479-p1p4] P4 capture FAILED rc=${p4_rc}" >&2
  write_sentinel "issue-2479-p1p4-results.json" "epm:progress" "p4_capture" 1 \
    "issue-2479 P4 FATAL capture failure rc=${p4_rc} (see logs/i1345_capture_*.log)"
  exit 45
fi
for v in "${SURVIVORS[@]}"; do
  date -u +%FT%TZ > "${LOG_DIR}/issue-2479-p4-${v}-done"
done

write_sentinel "issue-2479-p1p4-results.json" "epm:results" "p1p4" 0 \
  "issue-2479 P0+P1+P4 complete: p0 telemetry: ${P0_TELEM:-resumed-skip} | p1 survivors ${#SURVIVORS[@]}/${#GEN_CELLS[@]} cells (${op_survivors}/16 op) | p4 capture rc=0 (turnstores + skip-manifests uploaded per cell by issue1345_upload.py --legs turnstore; HF completion markers written)"
echo "[phase=done]"
exit 0
