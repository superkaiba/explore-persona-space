#!/usr/bin/env bash
# Issue #1345 char-capture-ladders — 16-cell teacher-forced capture fan-out
# (plan v13 §4 Phase C; modeled on scripts/issue1345_char_fanout_launch.sh).
#
# Per cell, ONE run_cell chain shared by every leg (smoke / pilot / waves):
#   stage (issue1345_stage_char_stories.py, plan §10 pin)
#   -> extract (issue1345_extract_turnstore.py --regime r4|r4op --model ...)
#   -> per-cell upload (issue1345_upload.py --legs turnstore; #664/#825 contract)
#   -> per-cell HF completion marker (_capture_complete.json — the restart skip key)
#
# Legs, in order:
#   1. SMOKE: one cell per (regime x model) class, --smoke (8 stories, causal
#      check ON), scratch out-dir + issue1345_smoke/ HF prefix — same chain.
#   2. GATE-1 PILOT (plan §7 gate 1): the FIRST PENDING production cell runs
#      alone; extrapolated total = wall x n_pending / n_gpu must be
#      <= EPS_I1345_GATE1_MAX_S (default 8640 s = 2.4 h = 2x the §9
#      projection). Over -> halt (rc=42) + sentinel gate report — a DESIGNED
#      artifact-routed halt, never a bare crash. An unparsed/zero pilot wall
#      is a gate ERROR (rc=43), never a trivially-passing 0.
#   3. WAVES: remaining cells n_gpu-wide (8-wide on the plan's allocation).
#
# Restart/resume: cells whose HF completion marker exists are skipped (the
# marker is uploaded ONLY after the cell's verified turnstore upload, so a
# partially-uploaded cell is never skipped); smoke cells are likewise skipped
# on their issue1345_smoke/ prefix marker. Sentinel:
# $SENTINEL_DIR/issue-1345-char-capture-results.json, written ATOMICALLY
# (tmp + mv) at the end (and on the gate-1 halt) — poll_pipeline schema v1.
#
# Usage:
#   bash scripts/issue1345_char_capture_launch.sh [--plan] [--skip-smoke]
#        [--smoke-only] [--cells char_helios char_wren_op ...]
set -uo pipefail

SENTINEL_DIR="${SENTINEL_DIR:-/workspace/logs}"
GATE1_MAX_S="${EPS_I1345_GATE1_MAX_S:-8640}"
PLAN_ONLY=0
SKIP_SMOKE=0
SMOKE_ONLY=0
SELECTED=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --plan) PLAN_ONLY=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --smoke-only) SMOKE_ONLY=1; shift ;;
    --cells) shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do SELECTED+=("$1"); shift; done ;;
    *) echo "[char-capture] unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [ "$SMOKE_ONLY" = "1" ] && [ "$SKIP_SMOKE" = "1" ]; then
  echo "[char-capture] --smoke-only with --skip-smoke is a no-op — refusing" >&2
  exit 2
fi

# Tuple table (variant|Label|model|regime). Labels verbatim from the gen
# fan-out / the story_yield_* records (HELIOS/Wren/Dana/Vex).
CELLS=(
  "char_helios|HELIOS|instruct|r4"
  "char_helios_op|HELIOS|instruct|r4op"
  "char_helios_base|HELIOS|pretrained|r4"
  "char_helios_op_base|HELIOS|pretrained|r4op"
  "char_wren|Wren|instruct|r4"
  "char_wren_op|Wren|instruct|r4op"
  "char_wren_base|Wren|pretrained|r4"
  "char_wren_op_base|Wren|pretrained|r4op"
  "char_dana|Dana|instruct|r4"
  "char_dana_op|Dana|instruct|r4op"
  "char_dana_base|Dana|pretrained|r4"
  "char_dana_op_base|Dana|pretrained|r4op"
  "char_vex|Vex|instruct|r4"
  "char_vex_op|Vex|instruct|r4op"
  "char_vex_base|Vex|pretrained|r4"
  "char_vex_op_base|Vex|pretrained|r4op"
)

# --- #2479 panel append seam (EPM_I2479_CHAR_PANEL_JSON) ---------------------
# Env absent => the parent 16-cell table above stays byte-identical. Env set =>
# the panel registry's char_2479_* cells APPEND in registry order (per row:
# variant_op as r4op/instruct, then non-null variant_inserted as r4/instruct —
# the ladder-fill's panel enumeration convention). Fail-loud on a bad panel.
if [ -n "${EPM_I2479_CHAR_PANEL_JSON:-}" ]; then
  panel_rows="$(uv run python - <<'PY'
import sys

sys.path.insert(0, "scripts")
from issue2479_char_panel import load_char_panel_env

rows = load_char_panel_env()
assert rows, "EPM_I2479_CHAR_PANEL_JSON set but loader returned no rows"
for r in rows:
    label = r.get("display_name")
    assert label, f"panel row {r.get('name')!r} lacks display_name (capture needs it)"
    print(f"{r['variant_op']}|{label}|instruct|r4op")
    if r["variant_inserted"]:
        print(f"{r['variant_inserted']}|{label}|instruct|r4")
PY
)" || { echo "[char-capture] #2479 panel load failed" >&2; exit 2; }
  while IFS= read -r row; do
    [ -n "$row" ] && CELLS+=("$row")
  done <<< "$panel_rows"
fi

# --cells subset (threads through smoke/pilot/waves alike — the one CELLS table).
if [ "${#SELECTED[@]}" -gt 0 ]; then
  FILTERED=()
  for spec in "${CELLS[@]}"; do
    v="${spec%%|*}"
    for s in "${SELECTED[@]}"; do
      if [ "$v" = "$s" ]; then FILTERED+=("$spec"); fi
    done
  done
  CELLS=("${FILTERED[@]}")
fi
if [ "${#CELLS[@]}" -lt 1 ]; then
  echo "[char-capture] no cells selected" >&2
  exit 2
fi

# Smoke set: the FIRST selected cell of each (regime x model) class (plan §4:
# one smoke cell per combination, run before the production wave).
SMOKE_CELLS=()
seen_classes=""
for spec in "${CELLS[@]}"; do
  IFS='|' read -r v _label model regime <<< "$spec"
  cls="${regime}:${model}"
  case " $seen_classes " in
    *" $cls "*) ;;
    *) SMOKE_CELLS+=("$spec"); seen_classes="$seen_classes $cls" ;;
  esac
done

hf_marker_path() { # variant smoke -> path_in_repo of the completion marker
  local variant="$1" smoke="$2" prefix
  if [ "$smoke" = "1" ]; then prefix="issue1345_smoke"; else prefix="issue1345_framing"; fi
  echo "${prefix}/${variant}/analysis_tensors/turnstore/_capture_complete.json"
}

cell_complete_on_hf() { # variant [smoke] -> exit 0 iff the completion marker exists
  local variant="$1" smoke="${2:-0}"
  local marker
  marker="$(hf_marker_path "$variant" "$smoke")"
  uv run python - "$marker" <<'PY'
import sys

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

marker = sys.argv[1]
ok = hub.retry_transient(
    lambda: HfApi().file_exists("superkaiba1/explore-persona-space-data", marker, repo_type="dataset"),
    what=f"file_exists({marker})",
)
print("COMPLETE" if ok else "MISSING")
sys.exit(0 if ok else 3)
PY
}

run_cell() { # spec dev smoke
  local spec="$1" dev="$2" smoke="$3"
  local variant label model regime
  IFS='|' read -r variant label model regime <<< "$spec"
  local marker
  marker="$(hf_marker_path "$variant" "$smoke")"
  local extract_args=(--regime "$regime" --model "$model")
  # Glob covers the pt shards + sidecars AND the *_s_skip_manifest.json files
  # (plan §10: skip-manifests ride the per-cell upload); the inner shell runs
  # `set -f` so the glob reaches the upload script unexpanded.
  local upload_args=(--legs turnstore --turnstore-glob "*_s_*")
  local ts_dir="data/issue_1345/${variant}/turnstore"
  if [ "$smoke" = "1" ]; then
    ts_dir="data/issue_1345/${variant}/turnstore_smoke"
    extract_args+=(--smoke --out-dir "$ts_dir")
    upload_args+=(--smoke)
  fi
  env EPM_I1345_VARIANT="$variant" EPM_STORY_CHARACTER_NAME="$label" \
    CUDA_VISIBLE_DEVICES="$dev" I1345_TS_DIR="$ts_dir" I1345_MARKER="$marker" \
    I1345_EXTRACT_ARGS="${extract_args[*]}" I1345_UPLOAD_ARGS="${upload_args[*]}" \
    bash -c '
      set -ef
      uv run python scripts/issue1345_stage_char_stories.py --variant "$EPM_I1345_VARIANT"
      # shellcheck disable=SC2086
      uv run python scripts/issue1345_extract_turnstore.py $I1345_EXTRACT_ARGS
      # shellcheck disable=SC2086
      uv run python scripts/issue1345_upload.py $I1345_UPLOAD_ARGS --turnstore-dir "$I1345_TS_DIR"
      # Completion marker AFTER the verified per-cell upload (restart skip key).
      uv run python - "$EPM_I1345_VARIANT" "$I1345_MARKER" <<"PY"
import json
import sys
import tempfile
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import upload_file

from explore_persona_space.orchestrate import hub

variant, marker = sys.argv[1], sys.argv[2]
payload = {
    "variant": variant,
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "marker": "issue1345 char-capture per-cell completion (upload-verified)",
}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(payload, f)
    tmp = f.name
hub.retry_transient(
    lambda: upload_file(
        path_or_fileobj=tmp,
        path_in_repo=marker,
        repo_id="superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        commit_message=f"issue-1345: capture complete marker {variant}",
    ),
    what=f"upload_file({marker})",
)
print(f"[marker] {marker}", flush=True)
PY
    '
}

# ---------------------------------------------------------------------------
# GPU allocation (index INTO the scheduler-provided CVD list — the 15771 lesson)
# ---------------------------------------------------------------------------
if [ "$PLAN_ONLY" = "1" ]; then
  n_gpu="${PLAN_WIDTH:-8}"
  DEVICES=()
  for g in $(seq 0 $((n_gpu - 1))); do DEVICES+=("$g"); done
else
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra alloc <<< "$CUDA_VISIBLE_DEVICES"
  else
    mapfile -t alloc < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  fi
  DEVICES=()
  for d in "${alloc[@]}"; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$d" | head -1)"
    if [ "${free_mib:-0}" -ge 60000 ]; then
      DEVICES+=("$d")
    else
      echo "[char-capture] skipping device ${d}: only ${free_mib:-?} MiB free" >&2
    fi
  done
  n_gpu="${#DEVICES[@]}"
  echo "[char-capture] allocated: ${alloc[*]:-none}; usable (>=60 GiB free): ${DEVICES[*]:-none}"
  if [ "$n_gpu" -lt 1 ]; then
    echo "[char-capture] no usable GPUs" >&2
    exit 3
  fi
fi

# ---------------------------------------------------------------------------
# --plan: print the leg/wave assignment and run NOTHING
# ---------------------------------------------------------------------------
if [ "$PLAN_ONLY" = "1" ]; then
  echo "[plan] width=${n_gpu} cells=${#CELLS[@]} gate1_max_s=${GATE1_MAX_S}"
  echo "[plan] smoke leg (${#SMOKE_CELLS[@]} cells, one per regime x model class):"
  for spec in "${SMOKE_CELLS[@]}"; do
    IFS='|' read -r v _l m r <<< "$spec"
    echo "  smoke  ${v} (regime=${r} model=${m}) -> --smoke, scratch out-dir, issue1345_smoke/ prefix"
  done
  IFS='|' read -r pilot_v _l pilot_m pilot_r <<< "${CELLS[0]}"
  echo "[plan] gate-1 pilot: ${pilot_v} (regime=${pilot_r} model=${pilot_m}) runs ALONE; halt if wall*n_pending/${n_gpu} > ${GATE1_MAX_S}s (n_pending <= ${#CELLS[@]}; unparsed wall -> rc=43 halt)"
  echo "[plan] production waves (${n_gpu}-wide) over remaining $(( ${#CELLS[@]} - 1 )) cells:"
  idx=1
  wave=1
  while [ "$idx" -lt "${#CELLS[@]}" ]; do
    line="  wave ${wave}:"
    for g in $(seq 0 $((n_gpu - 1))); do
      [ "$idx" -ge "${#CELLS[@]}" ] && break
      v="${CELLS[$idx]%%|*}"
      line="${line} ${v}(dev=${DEVICES[$g]})"
      idx=$((idx + 1))
    done
    echo "$line"
    wave=$((wave + 1))
  done
  echo "[plan] per-cell chain: stage -> extract -> upload(--legs turnstore) -> HF completion marker"
  echo "[plan] sentinel: ${SENTINEL_DIR}/issue-1345-char-capture-results.json (atomic tmp+mv)"
  exit 0
fi

mkdir -p logs "$SENTINEL_DIR"

declare -A CELL_RC
declare -A CELL_WALL
rc=0

write_sentinel() { # status note
  local status="$1" extra="$2"
  local per_cell="" v
  for spec in "${CELLS[@]}"; do
    v="${spec%%|*}"
    per_cell="${per_cell}${v}=${CELL_RC[$v]:-not-run}(${CELL_WALL[$v]:-0}s) "
  done
  local out="${SENTINEL_DIR}/issue-1345-char-capture-results.json"
  uv run python - "$out" "$status" "$per_cell" "$extra" <<'PY'
import json
import os
import sys
import time

out, status, per_cell, extra = sys.argv[1:5]
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:progress",
    "version": 1,
    "note": (
        f"issue-1345 char-capture fan-out {status}: {extra} | per-cell rc(wall): "
        f"{per_cell.strip()} | ts={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
    ),
}
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(payload, f)
os.replace(tmp, out)
print(f"[sentinel] {out}", flush=True)
PY
}

run_wave() { # smoke_flag spec... (chunks internally: never >1 cell per device)
  local smoke="$1"
  shift
  local all_specs=("$@")
  local tag=""
  [ "$smoke" = "1" ] && tag="_smoke"
  local start=0
  while [ "$start" -lt "${#all_specs[@]}" ]; do
    local wave_specs=("${all_specs[@]:$start:$n_gpu}")
    local pids=() labels=()
    local i=0
    local spec
    for spec in "${wave_specs[@]}"; do
      local v="${spec%%|*}"
      local dev="${DEVICES[$i]}"
      echo "[char-capture] starting ${v}${tag} on device ${dev} ($(date -u +%FT%TZ))"
      ( t0=$(date +%s); run_cell "$spec" "$dev" "$smoke"; arc=$?; t1=$(date +%s); \
        echo "WALL $((t1 - t0))"; exit "$arc" ) \
        > "logs/i1345_capture_${v}${tag}.log" 2>&1 &
      pids+=("$!")
      labels+=("$v")
      i=$((i + 1))
    done
    local j
    for j in "${!pids[@]}"; do
      wait "${pids[$j]}"
      local arc=$?
      local v="${labels[$j]}"
      local wall
      wall="$(grep -oE '^WALL [0-9]+' "logs/i1345_capture_${v}${tag}.log" | tail -1 | cut -d' ' -f2)"
      CELL_RC[$v]="$arc"
      CELL_WALL[$v]="${wall:-0}"
      echo "[char-capture] ${v} finished rc=${arc} wall=${wall:-?}s ($(date -u +%FT%TZ))"
      if [ "$arc" -ne 0 ] && [ "$rc" -eq 0 ]; then rc="$arc"; fi
    done
    start=$((start + n_gpu))
  done
}

# --- leg 1: smoke cells (same chain, --smoke + smoke prefix) ----------------
# Resume: smoke cells whose SMOKE-prefix completion marker exists are skipped,
# so a resume does not re-run ~4 model loads + captures per restart (r1 review
# Minor 3; --skip-smoke stays the explicit override).
if [ "$SKIP_SMOKE" = "0" ]; then
  SMOKE_PENDING=()
  for spec in "${SMOKE_CELLS[@]}"; do
    v="${spec%%|*}"
    if cell_complete_on_hf "$v" 1 > /dev/null 2>&1; then
      echo "[char-capture] ${v}: smoke-prefix marker present — smoke skipped (resume)"
    else
      SMOKE_PENDING+=("$spec")
    fi
  done
  echo "[char-capture] smoke leg: ${#SMOKE_PENDING[@]}/${#SMOKE_CELLS[@]} cells pending"
  if [ "${#SMOKE_PENDING[@]}" -gt 0 ]; then
    run_wave 1 "${SMOKE_PENDING[@]}"
    if [ "$rc" -ne 0 ]; then
      echo "[char-capture] smoke leg FAILED rc=${rc} — no production cell launched" >&2
      write_sentinel "smoke-failed" "rc=${rc}"
      exit "$rc"
    fi
  fi
fi

# --- #2479 P0: --smoke-only exits after the smoke leg (never a production ----
# capture on 3-story smoke bundles, whose issue1345_framing/ completion
# markers would poison the P4 production resume filter).
if [ "$SMOKE_ONLY" = "1" ]; then
  echo "[char-capture] --smoke-only: smoke leg complete rc=${rc} — exiting before production"
  write_sentinel "smoke-only-done" "rc=${rc}"
  exit "$rc"
fi

# --- resume filter: drop cells already complete on HF -----------------------
PENDING=()
for spec in "${CELLS[@]}"; do
  v="${spec%%|*}"
  if cell_complete_on_hf "$v" > /dev/null 2>&1; then
    echo "[char-capture] ${v}: HF completion marker present — skipped (resume)"
    CELL_RC[$v]="skipped-complete"
    CELL_WALL[$v]=0
  else
    PENDING+=("$spec")
  fi
done
if [ "${#PENDING[@]}" -eq 0 ]; then
  echo "[char-capture] all cells already complete on HF"
  write_sentinel "done" "all cells HF-complete (resume no-op)"
  exit 0
fi

# --- leg 2: gate-1 pilot (first pending production cell, alone) -------------
pilot="${PENDING[0]}"
pilot_v="${pilot%%|*}"
echo "[char-capture] gate-1 pilot: ${pilot_v}"
run_wave 0 "$pilot"
if [ "$rc" -ne 0 ]; then
  echo "[char-capture] pilot cell FAILED rc=${rc}" >&2
  write_sentinel "pilot-failed" "cell=${pilot_v} rc=${rc}"
  exit "$rc"
fi
pilot_wall="${CELL_WALL[$pilot_v]:-0}"
# Fail-CLOSED on an unparsed/zero pilot wall: CELL_WALL defaults to 0 when the
# WALL log line is missing, which would trivially pass the gate (r1 Minor 5).
if ! [[ "$pilot_wall" =~ ^[0-9]+$ ]] || [ "$pilot_wall" -le 0 ]; then
  echo "[char-capture] GATE-1 ERROR: pilot ${pilot_v} wall unparsed (got '${pilot_wall}') — cannot project; halting" >&2
  write_sentinel "gate1-wall-parse-error" "pilot=${pilot_v} wall=${pilot_wall}"
  exit 43
fi
# Project over the REMAINING (pending) cells, not the full table — on a resume
# with most cells HF-complete the old denominator over-halted (r1 Minor 2).
proj_total=$((pilot_wall * ${#PENDING[@]} / n_gpu))
echo "[char-capture] gate-1: pilot wall=${pilot_wall}s -> projected total ${proj_total}s over ${#PENDING[@]} pending of ${#CELLS[@]} cells (max ${GATE1_MAX_S}s)"
if [ "$proj_total" -gt "$GATE1_MAX_S" ]; then
  echo "[char-capture] GATE-1 HALT: projected ${proj_total}s > ${GATE1_MAX_S}s — pausing the wave (plan §7 gate 1)" >&2
  write_sentinel "gate1-halt" "pilot=${pilot_v} pilot_wall_s=${pilot_wall} projected_total_s=${proj_total} n_pending=${#PENDING[@]} n_cells=${#CELLS[@]} max_s=${GATE1_MAX_S}"
  exit 42
fi

# --- leg 3: production waves (n_gpu-wide; run_wave chunks internally) --------
REMAINING=("${PENDING[@]:1}")
if [ "${#REMAINING[@]}" -gt 0 ]; then
  run_wave 0 "${REMAINING[@]}"
fi

echo "[char-capture] all cells done, rc=${rc}"
write_sentinel "done" "rc=${rc}"
exit "$rc"
