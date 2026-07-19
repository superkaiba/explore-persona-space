#!/usr/bin/env bash
# Issue #1417 dispatcher — framing cells (helpful register vs user-directedness).
#
# GCE-lane workload commands (plan §10):
#   Phase A (gen + capture + per-cell store upload, 4 GPUs):
#     REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" I1417_PHASE=A bash scripts/issue1417_run.sh
#   Phase B (judge) runs on the VM, NOT via this dispatcher:
#     uv run python scripts/issue1417_judge.py --all --stage-from-hub
#   Phase C (fits + anchors + battery + figures, 2 GPUs):
#     REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" I1417_PHASE=C bash scripts/issue1417_run.sh
#
# Smoke (PASS_UNIFIED, plan §4): SMOKE=1 runs the SAME chain end-to-end at
# tiny n on scratch roots — the phase-A unit queue at 50 questions, the G3
# forced-live judge mini-batch + a tiny real judge pass (kept-sets), 1-shard
# reference stores, anchors/fits/battery/summary/figures with production-n
# gate verdicts demoted to log lines (--smoke; gotchas #1345). Every phase's
# cell list derives from the same MODELS x CELL_ORDER registry in both modes.
#
# Signalling: [phase=...] stdout breadcrumbs terminating in [phase=done];
# end-of-run sentinel under $LOG_DIR (pod-side-reporting contract; this
# script NEVER shells out to scripts/task.py).
#
# Designed-halt exit codes (routed, never bare rc=1 — gotchas #1415):
#   rc 20 = G1 anchor gate HALT; rc 21 = G3 judge live-smoke FAIL;
#   rc 22 = G2 fit-device gate HALT; rc 7 = PC-3 battery pilot abort;
#   rc 86 = results push did not land (#1205); rc 87 = pushed tree is
#   missing a declared result file (#1325).
#
# Env knobs: I1417_PHASE=A|C (required unless SMOKE=1), SMOKE=1, SKIP_SMOKE=1
# (full phases only; the pre-full smoke is ON by default for phase A),
# NGPUS, DATA_DIR/OUT_DIR/FIG_DIR/LOG_DIR, SKIP_UPLOAD=1, SKIP_PUSH=1,
# SMOKE_N (50), SMOKE_JUDGE_LIMIT (25), I1417_SMOKE_ROOT,
# EPM_VLLM_ENFORCE_EAGER / EPM_VLLM_DISABLE_PREFIX_CACHING (threaded into
# issue1417_gen.py; default off — ONE engine config across every cell of the
# comparison, plan §10).
#
# I1417_REFIT=1 (with I1417_PHASE=C): registered-selector refit round —
# re-runs stage-stores -> anchors -> fits -> battery -> summary with the
# frozen instrument's registered mitigations (--lambda-selection
# inner-group-cv --gcv-dof-cap 0.9) and per-fit selector logging, writing to
# $OUT_DIR/refit (v1 outputs are the published void record — never resumed
# over, never overwritten) while READING the v1 judge kept-sets via
# --judge-dir $OUT_DIR. Figures are skipped in refit mode (the analyzer
# regenerates from the refit JSONs). Inner-group-CV is MORE expensive per fit
# than GCV (4 inner eighs per fold prep + a per-target inner RSS curve) —
# raise I1417_PILOT_BUDGET_H if the PC-3 pilot rc-7 aborts on projected cost.
#
# I1417_CELLS=<csv> (milder-rude round, plan §4.2 item 4a): overrides the
# unit-queue cell list AND is threaded as --cells into the smoke judge legs +
# every fits/battery call (run_summary always reads the full CELL_ORDER —
# merged sparse panel). Default = the five v1 cells (byte-preserving).
#
# I1417_MILDER=1 (with I1417_PHASE=C + I1417_CELLS=c2_rude_mild): milder-rude
# amendment round (plan v6 §4.2 item 4b) — BAT_OUT_DIR=$OUT_DIR/milder_rude,
# carry step (v1 judge kept-sets + refit cells/nulls + refit battery PAIR
# files copied in, file-count asserted), then the refit's registered selector
# (--lambda-selection inner-group-cv --gcv-dof-cap 0.9), --judge-dir
# $OUT_DIR/milder_rude, --cells c2_rude_mild, --h-rude-cell c2_rude_mild.
# Anchors are NOT carried — G1 re-runs fresh. Figures skipped (as refit).
# I1417_CARRY_ONLY=1 runs ONLY the carry step (VM verification affordance).
# VM judge phases for the milder round pass --hf-subdir milder_rude (full) /
# milder_rude/pilot (pilot) so yield_report.json / n_draw_pilot.json never
# clobber the published v1 HF judge paths.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"
# Conditional .env sourcing (GCE exports tokens via startup metadata; no .env there).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
# vLLM v1 EngineCore fork poisoning (#628): set before any python/vllm import.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

ISSUE=1417
BRANCH="issue-1417"
PHASE="${I1417_PHASE:-}"
DATA_DIR="${DATA_DIR:-data/issue_1417}"
OUT_DIR="${OUT_DIR:-eval_results/issue_1417}"
FIG_DIR="${FIG_DIR:-figures/issue_1417}"
LOG_DIR="${LOG_DIR:-${WORKLOAD_ROOT:-/workspace}/logs}"
SMOKE="${SMOKE:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
REFIT="${I1417_REFIT:-0}"
MILDER="${I1417_MILDER:-0}"
# Phase-C battery out-dir override (refit: $OUT_DIR/refit; milder:
# $OUT_DIR/milder_rude; env-overridable for the I1417_CARRY_ONLY affordance).
BAT_OUT_DIR="${BAT_OUT_DIR:-}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"
SMOKE_N="${SMOKE_N:-50}"
SMOKE_JUDGE_LIMIT="${SMOKE_JUDGE_LIMIT:-25}"
SMOKE_JUDGE_DRAWS="${SMOKE_JUDGE_DRAWS:-2}"
HF_PREFIX="issue1417_framing_cells"
HF_REPO="superkaiba1/explore-persona-space-data"

# Lane filter (per-lane pilot disposition, plan v7 §4.3): I1417_MODELS
# space-separated override — e.g. "instruct" for a single-lane Phase C after
# the other lane failed its pilot bars. Default: both lanes (v1/refit shape).
MODELS=(${I1417_MODELS:-instruct pretrained})
# Cell filter (plan §4.2 item 4a): ONE source for every phase's cell list —
# the unit queue, the smoke judge --cells, and every fits/battery --cells.
# Default = the five v1 cells (fits/battery filtered to these == the v1/refit
# behavior exactly; the registry's 6th cell enters only via I1417_CELLS).
CELLS_CSV="${I1417_CELLS:-c1_helpful_ctrl,c2_rude,c3_evasive,c4_exposition,c5_ai_addressee}"
IFS=',' read -r -a CELLS <<< "$CELLS_CSV"
QUEUE_CELLS=("${CELLS[@]}")  # run_smoke may augment with the c1 reference cell
REFERENCE_STEMS=(instruct_chat_s pretrained_chat_s instruct_naturalistic_s pretrained_naturalistic_s)

if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
  DETECTED_GPUS=0
fi
NGPUS="${NGPUS:-$DETECTED_GPUS}"
# Never assume more lanes than physically visible (dispatcher wave rule).
if [ "$NGPUS" -gt "$DETECTED_GPUS" ] && [ "$DETECTED_GPUS" -gt 0 ]; then NGPUS="$DETECTED_GPUS"; fi
if [ "$NGPUS" -lt 1 ]; then NGPUS=1; fi
mkdir -p "$LOG_DIR" "$DATA_DIR"

log() { echo "[i1417-run] $*"; }

headroom() {  # headroom <path> <need_gb> <phase>
  uv run python - "$1" "$2" "$3" <<'PY'
import sys

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

path, need, phase = sys.argv[1], float(sys.argv[2]), sys.argv[3]
free = assert_out_root_headroom(path, need, phase=phase)
print(f"[i1417-run] headroom {phase}: {free:.1f} GB free at {path} (need {need})")
PY
}

write_sentinel() {  # write_sentinel <kind> <gate> <summary_json_path> <elapsed_h> [out_dir]
  local kind="$1" gate="$2" note_path="$3" elapsed_h="$4" outd="${5:-$OUT_DIR}"
  local slug epoch dest
  slug=$(echo "$kind" | tr ':' '_')
  epoch=$(date +%s)
  if [ "$gate" = "smoke" ]; then
    dest="$LOG_DIR/issue-${ISSUE}-smoke-results.json"
  else
    dest="$LOG_DIR/issue-${ISSUE}-${slug}-${epoch}.json"
  fi
  uv run python - "$kind" "$gate" "$note_path" "$dest" "$outd" "$FIG_DIR" "$elapsed_h" <<'PY'
import json
import os
import subprocess
import sys

kind, gate, note_path, dest, out_dir, fig_dir, elapsed_h = sys.argv[1:8]
summary = {}
if note_path != "none" and os.path.exists(note_path):
    summary = json.load(open(note_path, encoding="utf-8"))
sha = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
).stdout.strip()
hf_prefix = "issue1417_framing_cells"
payload_note = {
    "eval_numbers": {
        "h_table_lookup": summary.get("h_table_lookup"),
        "cells": {
            k: {
                kk: v.get(kk)
                for kk in ("rel_l19", "verdict", "yield_frac", "r2_l19", "map_exists")
            }
            for k, v in summary.get("cells", {}).items()
        },
    },
    "eval_paths": [
        f"{out_dir}/battery_summary.json",
        f"{out_dir}/cells/",
        f"{out_dir}/battery/",
        f"{out_dir}/anchors/",
        f"{out_dir}/judge/yield_report.json",
        f"{fig_dir}/",
    ],
    "reproducibility_card": {
        "analysis_only": True,
        "note": "no training; generation + activation capture + closed-form ridge fits",
        "hf_data_prefix": f"{hf_prefix}/",
        "render_config_hash": summary.get("render_config_hash"),
    },
    "wandb_url": "n/a (no training; activation-geometry fits only)",
    "hf_hub_url": f"https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/{hf_prefix}",
    "worktree_path": os.getcwd(),
    "final_commit_sha": sha,
    "gpu_hours_used": elapsed_h,
    "gpu_hours_budgeted": 16,
    "plan_deviations": [],
}
note = json.dumps(payload_note)
if len(note) > 45000:
    payload_note["eval_numbers"]["cells"] = "truncated — see battery_summary.json"
    note = json.dumps(payload_note)
payload = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "task_id": 1417,
    "gate": gate,
    "blocks_pipeline": False,
    "by": "issue1417_run.sh",
    "note": note,
}
tmp = dest + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f)
os.replace(tmp, dest)
print(f"[i1417-run] sentinel written: {dest}")
PY
}

# ---------------------------------------------------------------------------
# Phase A — CVD-pinned work-conserving unit queue over (model, cell) units.
# Each unit: gen (one vLLM engine, reaped in-process) -> capture -> per-cell
# store-shard upload (P2.5 per cell — before ANY fit exists; upload policy).
# ---------------------------------------------------------------------------
QUEUE_DIR=""

init_queue() {  # init_queue <scratch_tag> — units from QUEUE_CELLS (cell filter)
  QUEUE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/i1417-queue-$1-XXXX")
  : > "$QUEUE_DIR/units"
  for model in "${MODELS[@]}"; do
    for cell in "${QUEUE_CELLS[@]}"; do
      echo "${model}:${cell}" >> "$QUEUE_DIR/units"
    done
  done
  echo 0 > "$QUEUE_DIR/pos"
}

claim_unit() {  # atomically claim the next unit index; rc 1 when drained
  local idx total
  {
    flock 9
    idx=$(cat "$QUEUE_DIR/pos")
    total=$(wc -l < "$QUEUE_DIR/units")
    if [ "$idx" -ge "$total" ]; then
      return 1
    fi
    echo $((idx + 1)) > "$QUEUE_DIR/pos"
  } 9>>"$QUEUE_DIR/lock"
  echo "$idx"
}

upload_own_store() {  # upload_own_store <data_dir> <model> <cell> <mode>
  if [ "$SKIP_UPLOAD" = "1" ] || [ "${4:-full}" = "smoke" ]; then
    log "store upload $2/$3 skipped (SKIP_UPLOAD/smoke)"
    return 0
  fi
  uv run python - "$1" "$2" "$3" <<'PY'
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

data_dir, model, cell = sys.argv[1:4]
store = Path(data_dir) / "store"
repo = "superkaiba1/explore-persona-space-data"
prefix = "issue1417_framing_cells/analysis_tensors/store"
res = upload_dir_sharded(
    store, repo, prefix, shard_glob=f"{model}_{cell}_s_shard*.pt", verify=True, delete_local=False
)
print(f"[i1417-upload] {model}/{cell} shards: uploaded={len(res.uploaded)} rerouted={len(res.rerouted)}")
res2 = upload_dir_sharded(
    store, repo, prefix, shard_glob=f"{model}_{cell}_s_shard*.json", verify=True, delete_local=False
)
res3 = upload_dir_sharded(
    store, repo, prefix, shard_glob=f"{model}_{cell}_s_equivalence.json", verify=True, delete_local=False
)
print(f"[i1417-upload] {model}/{cell} sidecars: uploaded={len(res2.uploaded) + len(res3.uploaded)}")
PY
}

phase_a_worker() {  # phase_a_worker <gpu> <data_dir> <mode>
  local gpu="$1" data_dir="$2" mode="$3"
  export CUDA_VISIBLE_DEVICES="$gpu"
  local idx unit model cell slice_args=() upload_args=() eq_args=()
  if [ "$mode" = "smoke" ]; then
    slice_args=(--n-questions "$SMOKE_N")
    upload_args=(--skip-upload)
  elif [ "$SKIP_UPLOAD" = "1" ]; then
    upload_args=(--skip-upload)
  fi
  while idx=$(claim_unit); do
    unit=$(sed -n "$((idx + 1))p" "$QUEUE_DIR/units")
    model="${unit%%:*}"
    cell="${unit##*:}"
    echo "[i1417-run] worker gpu=$gpu unit=$unit gen start"
    uv run python scripts/issue1417_gen.py --model "$model" --cells "$cell" \
      --data-dir "$data_dir" --resume "${slice_args[@]}" "${upload_args[@]}"
    eq_args=()
    # Equivalence gate on ONE cell per model (the #779 two-bar calibration).
    if [ "$cell" = "c1_helpful_ctrl" ]; then eq_args=(--equivalence-check); fi
    echo "[i1417-run] worker gpu=$gpu unit=$unit capture start"
    uv run python scripts/issue1417_extract.py --model "$model" --cell "$cell" \
      --data-dir "$data_dir" --resume "${eq_args[@]}"
    upload_own_store "$data_dir" "$model" "$cell" "$mode"
    echo "[i1417-run] worker gpu=$gpu unit=$unit complete"
  done
  echo "[i1417-run] worker gpu=$gpu drained"
}

wait_pids() {  # wait_pids <pids...> — heartbeat + fail loud with log tails
  local pids=("$@") rc=0 one_rc
  while :; do
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=1; fi
    done
    if [ "$alive" = "0" ]; then break; fi
    sleep 120
    for lg in "$LOG_DIR"/issue-${ISSUE}-worker-*.log; do
      [ -f "$lg" ] || continue
      log "heartbeat $(basename "$lg"): $(tail -n 1 "$lg" 2>/dev/null | cut -c1-160)"
    done
  done
  for pid in "${pids[@]}"; do
    one_rc=0
    wait "$pid" || one_rc=$?
    if [ "$one_rc" -ne 0 ]; then
      log "WORKER FAILED rc=$one_rc — tails follow"
      for lg in "$LOG_DIR"/issue-${ISSUE}-worker-*.log; do
        [ -f "$lg" ] || continue
        log "tail of $(basename "$lg"):"
        tail -n 40 "$lg" || true
      done
      rc=$one_rc
    fi
  done
  return "$rc"
}

run_phase_a() {  # run_phase_a <data_dir> <mode>
  local data_dir="$1" mode="$2"
  init_queue "$mode"
  rm -f "$LOG_DIR"/issue-${ISSUE}-worker-*.log
  echo "[phase=p1_p2_units_${mode}]"
  local pids=()
  for gpu in $(seq 0 $((NGPUS - 1))); do
    local wl="$LOG_DIR/issue-${ISSUE}-worker-${mode}-g${gpu}.log"
    ( phase_a_worker "$gpu" "$data_dir" "$mode" ) >"$wl" 2>&1 &
    pids+=($!)
  done
  wait_pids "${pids[@]}"
  rm -rf "$QUEUE_DIR"
}

# ---------------------------------------------------------------------------
# Staging (phase C + smoke): judge outputs, gen JSONLs, reference/own stores.
# ---------------------------------------------------------------------------
stage_phase_c_inputs() {  # stage judge kept-sets + gen JSONLs from the issue prefix
  uv run python - "$DATA_DIR" "$OUT_DIR" <<'PY'
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
sys.path.insert(0, "scripts")
from huggingface_hub import HfApi

import issue1417_judge as j1417
import issue1417_render as r1417
from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])


class _A:  # minimal args shim for _stage_gen_from_hub
    pass


args = _A()
args.data_dir = data_dir
for model in r1417.MODELS:
    for cell in r1417.CELL_ORDER:
        p = j1417._stage_gen_from_hub(data_dir, model, cell)
        print(f"[i1417-stage] gen {model}/{cell}: {p}")
jdir = out_dir / "judge"
jdir.mkdir(parents=True, exist_ok=True)
prefix = f"{r1417.HF_PREFIX}/raw_completions/judge"
paths = list_hf_files_under_path(HfApi(), r1417.HF_DATA_REPO, prefix, repo_type="dataset")
n = 0
for p in paths:
    rel = p[len(prefix) + 1 :]
    if "/" in rel:
        # Subdirs are never v1 judge outputs: raw/ (per-rubric raw draws,
        # consumed on HF only) and the milder round's milder_rude/ +
        # milder_rude/pilot/ subprefixes (staged from git, never from here).
        continue
    dest = jdir / rel
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        stage_hub_file(r1417.HF_DATA_REPO, p, dest, repo_type="dataset")
        n += 1
print(f"[i1417-stage] judge outputs staged: {n} (of {len(paths)} listed)")
PY
}

stage_own_stores() {  # stage_own_stores <model> — this lane's 5 capture stores
  uv run python - "$DATA_DIR" "$1" <<'PY'
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
sys.path.insert(0, "scripts")
from huggingface_hub import HfApi

import issue1417_render as r1417
from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

data_dir, model = Path(sys.argv[1]), sys.argv[2]
dest_dir = data_dir / "store"
dest_dir.mkdir(parents=True, exist_ok=True)
prefix = f"{r1417.HF_PREFIX}/analysis_tensors/store"
paths = list_hf_files_under_path(HfApi(), r1417.HF_DATA_REPO, prefix, repo_type="dataset")
todo = [p for p in paths if Path(p).name.startswith(f"{model}_")]
assert todo, f"no own-store shards for {model} under {prefix} — run phase A first"
n = 0
for p in todo:
    dest = dest_dir / Path(p).name
    if not dest.exists():
        stage_hub_file(r1417.HF_DATA_REPO, p, dest, repo_type="dataset")
        n += 1
print(f"[i1417-stage] own stores {model}: staged {n}/{len(todo)}")
PY
}

stage_reference_shard0() {  # smoke: ONE shard per reference stem into <data_dir>
  uv run python - "$1" <<'PY'
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
sys.path.insert(0, "scripts")
import issue1417_render as r1417
from explore_persona_space.orchestrate.hub import stage_hub_file

data_dir = Path(sys.argv[1])
dest = data_dir / "turnstore"
dest.mkdir(parents=True, exist_ok=True)
rev = json.loads((r1417.sidecar_dir(data_dir) / "revision.json").read_text())["revision"]
for stem in r1417.REFERENCE_STEMS:
    for ext in ("pt", "json"):
        p = f"{r1417.PARENT_PREFIX}/{stem}_shard000.{ext}"
        t = dest / f"{stem}_shard000.{ext}"
        if not t.exists():
            stage_hub_file(r1417.HF_DATA_REPO, p, t, repo_type="dataset", revision=rev)
    print(f"[i1417-stage] smoke reference shard0 staged: {stem}")
PY
}

carry_milder_inputs() {  # milder round (plan §4.2 item 4b): carry v1/refit inputs
  echo "[phase=pc_milder_carry]"
  mkdir -p "$BAT_OUT_DIR/judge" "$BAT_OUT_DIR/cells" "$BAT_OUT_DIR/battery"
  # cp -n: never overwrite files already there (the mild kept-sets are
  # committed under $BAT_OUT_DIR/judge by the VM full-judge step). coreutils
  # >=9.2 exits nonzero on a -n skip — tolerated; the assert block below is
  # the fail-loud completeness check.
  cp -n "$OUT_DIR"/judge/kept_*.json "$BAT_OUT_DIR/judge/" 2>/dev/null || true
  cp -n "$OUT_DIR"/refit/cells/*.json "$BAT_OUT_DIR/cells/" 2>/dev/null || true
  # Battery PAIR files ONLY (battery_*__*.json — the '__' excludes the two
  # refit battery_pilot_report_*.json; this round's PC-3 writes its own).
  cp -n "$OUT_DIR"/refit/battery/battery_*__*.json "$BAT_OUT_DIR/battery/" 2>/dev/null || true
  # Anchors are NOT carried — phase C re-runs G1 fresh (plan §4.2 item 4).
  uv run python - "$BAT_OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue1417_render as r1417

bat = Path(sys.argv[1])
carried_cells = [c for c in r1417.CELL_ORDER if c != "c2_rude_mild"]

# 12 carried kept-sets (10 cell + 2 kept_*_c0_baseline.json) + 2 mild kept-sets.
missing = []
for model in r1417.MODELS:
    for cell in [*carried_cells, "c0_baseline", "c2_rude_mild"]:
        p = bat / "judge" / f"kept_{model}_{cell}.json"
        if not p.exists():
            missing.append(str(p))
assert not missing, f"carry: missing kept-sets: {missing}"
for model in r1417.MODELS:
    kd = json.loads((bat / "judge" / f"kept_{model}_c2_rude_mild.json").read_text())
    assert r1417.fingerprint_matches(kd), f"mild kept-set fingerprint mismatch ({model})"

# 160 carried cell JSONs = 80 cells_* + 80 nulls_*. The nulls_* files are
# LOAD-BEARING: _summary_fit_entry reads cells/nulls_<cell>__<model>__ctx.json
# for map_exists — carrying only cells_* would silently null map_exists on
# every carried cell. Counts exclude this round's own c2_rude_mild outputs
# (resume-safe: the assert stays true after the mild fits land).
names = [p.name for p in (bat / "cells").glob("*.json") if "c2_rude_mild" not in p.name]
n_cells = sum(1 for n in names if n.startswith("cells_"))
n_nulls = sum(1 for n in names if n.startswith("nulls_"))
assert (n_cells, n_nulls) == (80, 80), (
    f"carry: carried cell JSONs cells_={n_cells} nulls_={n_nulls} (expected 80+80)"
)

# 26 carried battery PAIR files (pilot reports excluded by the '__' glob).
pairs = [p.name for p in (bat / "battery").glob("battery_*__*.json") if "c2_rude_mild" not in p.name]
assert len(pairs) == 26, f"carry: carried battery PAIR files {len(pairs)} != 26"
print(
    f"[i1417-run] carry asserts PASS: kept=14 (12 carried + 2 mild), "
    f"cells={n_cells}+{n_nulls} nulls, battery pairs={len(pairs)}"
)
PY
}

# ---------------------------------------------------------------------------
# Phase C — G2 gate, per-model CVD-pinned lanes (stage refs -> anchors G1 ->
# fits -> battery), then summary + figures + verified results push.
# ---------------------------------------------------------------------------
phase_c_lane() {  # phase_c_lane <gpu> <model> <mode> [extra battery args...]
  local gpu="$1" model="$2" mode="$3"
  shift 3
  export CUDA_VISIBLE_DEVICES="$gpu"
  local common=(--data-dir "$DATA_DIR" --out-dir "${BAT_OUT_DIR:-$OUT_DIR}" "$@")
  if [ "$mode" = "full" ]; then
    uv run python scripts/issue1417_battery.py --stage-stores --model "$model" "${common[@]}"
    stage_own_stores "$model"
  fi
  uv run python scripts/issue1417_battery.py --anchors --model "$model" "${common[@]}"
  uv run python scripts/issue1417_battery.py --fits --model "$model" --resume "${common[@]}"
  uv run python scripts/issue1417_battery.py --battery --model "$model" --resume "${common[@]}"
  echo "lane complete: model=$model mode=$mode"
}

run_phase_c_lanes() {  # run_phase_c_lanes <mode> [extra battery args...]
  local mode="$1"
  shift
  echo "[phase=pc_lanes_${mode}]"
  if [ "$NGPUS" -ge 2 ]; then
    local li="$LOG_DIR/issue-${ISSUE}-worker-${mode}-lane-instruct.log"
    local lp="$LOG_DIR/issue-${ISSUE}-worker-${mode}-lane-pretrained.log"
    ( phase_c_lane 0 instruct "$mode" "$@" ) >"$li" 2>&1 &
    local pid_i=$!
    ( phase_c_lane 1 pretrained "$mode" "$@" ) >"$lp" 2>&1 &
    local pid_p=$!
    wait_pids "$pid_i" "$pid_p"
  else
    for model in "${MODELS[@]}"; do
      phase_c_lane 0 "$model" "$mode" "$@"
    done
  fi
}

commit_and_push_results() {
  if [ "$SKIP_PUSH" = "1" ]; then
    log "results push skipped (SKIP_PUSH)"
    return 0
  fi
  git add "$OUT_DIR" "$FIG_DIR"
  if git diff --cached --quiet; then
    log "no result changes to commit"
    return 0
  fi
  git commit -m "issue-1417: framing-cell eval results (run $(date -u +%Y-%m-%dT%H:%MZ))"
  local ok=0
  for attempt in 1 2; do
    if git push origin "HEAD:$BRANCH"; then
      if [ "$(git rev-list --count "origin/$BRANCH..HEAD")" = "0" ]; then
        ok=1
        break
      fi
    fi
    log "push attempt $attempt failed; retrying after fetch"
    git fetch origin "$BRANCH" || true
  done
  if [ "$ok" != "1" ]; then
    log "FATAL: results push did not land (rev-list non-zero) — failing loud (#1205)"
    return 86
  fi
  log "results push verified (rev-list count 0)"
  # Artifact-presence assert (#1325): every git-destined result file this run
  # wrote must be IN the pushed tree (per-file, never a bare directory).
  git ls-tree -r "origin/$BRANCH" --name-only -- "$OUT_DIR" "$FIG_DIR" > /tmp/i1417_tree.txt
  local missing=0
  while IFS= read -r p; do
    if ! grep -qxF "$p" /tmp/i1417_tree.txt; then
      log "MISSING FROM PUSHED TREE: $p"
      missing=1
    fi
  done < <(find "$OUT_DIR" "$FIG_DIR" -type f \( -name '*.json' -o -name '*.png' -o -name '*.pdf' \) | sort)
  if [ "$missing" != "0" ]; then
    log "FATAL: pushed tree is missing declared result files (#1325)"
    return 87
  fi
  log "artifact-presence assert passed (every local result file is in the pushed tree)"
}

# ---------------------------------------------------------------------------
# Smoke pipeline — the SAME chain at tiny n on scratch roots (PASS_UNIFIED).
# ---------------------------------------------------------------------------
run_smoke() {
  local smoke_root="${I1417_SMOKE_ROOT:-/tmp/issue-1417-smoke}"
  local data_dir="$smoke_root/data" out_dir="$smoke_root/eval_results" fig_dir="$smoke_root/figures"
  mkdir -p "$data_dir" "$out_dir" "$fig_dir"
  # The smoke consumes the SAME staged questions + sidecars (copy, never re-fetch).
  cp "$DATA_DIR/track_s.jsonl" "$data_dir/"
  rm -rf "$data_dir/reference_sidecars"
  cp -r "$DATA_DIR/reference_sidecars" "$data_dir/"

  # The vs-c1 battery pairs read the c1 store + kept-set from the scratch
  # smoke root, so the smoke gen/judge legs ALWAYS include c1_helpful_ctrl
  # even when the cell filter omits it (production phase C stages v1's
  # committed c1 artifacts through the SAME consumer code path). Fits/battery
  # keep the FILTERED set (the c1 reference needs no fit of its own).
  local smoke_cells_csv="$CELLS_CSV"
  if [[ ",$CELLS_CSV," != *",c1_helpful_ctrl,"* ]]; then
    smoke_cells_csv="$CELLS_CSV,c1_helpful_ctrl"
  fi
  IFS=',' read -r -a QUEUE_CELLS <<< "$smoke_cells_csv"

  echo "[phase=smoke_units]"
  run_phase_a "$data_dir" smoke
  QUEUE_CELLS=("${CELLS[@]}")  # restore the unaugmented filter for phase-A full

  echo "[phase=smoke_judge_g3]"
  uv run python scripts/issue1417_judge.py --live-smoke --data-dir "$data_dir" --out-dir "$out_dir"

  echo "[phase=smoke_judge]"
  uv run python scripts/issue1417_judge.py --all --data-dir "$data_dir" --out-dir "$out_dir" \
    --cells "$smoke_cells_csv" \
    --limit "$SMOKE_JUDGE_LIMIT" --n-draws "$SMOKE_JUDGE_DRAWS" --skip-upload --pilot-report

  echo "[phase=smoke_refs]"
  stage_reference_shard0 "$data_dir"

  echo "[phase=smoke_fits_battery]"
  local common=(--data-dir "$data_dir" --out-dir "$out_dir" --smoke --cells "$CELLS_CSV")
  uv run python scripts/issue1417_battery.py --gate-g2 "${common[@]}"
  for model in "${MODELS[@]}"; do
    uv run python scripts/issue1417_battery.py --anchors --model "$model" "${common[@]}"
    uv run python scripts/issue1417_battery.py --fits --model "$model" "${common[@]}"
    uv run python scripts/issue1417_battery.py --battery --model "$model" "${common[@]}"
  done
  uv run python scripts/issue1417_battery.py --summary "${common[@]}"

  echo "[phase=smoke_figures]"
  uv run python scripts/issue1417_figures.py --out-dir "$out_dir" --fig-dir "$fig_dir"

  # Refit leg (registered-selector refit round): the SAME chain at tiny n with
  # the SAME flags the I1417_REFIT=1 phase C passes — versioned refit out-dir,
  # v1-judge read-only, inner-group-cv + dof cap threaded into every fit site.
  echo "[phase=smoke_refit_fits_battery]"
  local refit_common=(--data-dir "$data_dir" --out-dir "$out_dir/refit" --judge-dir "$out_dir" \
    --smoke --lambda-selection inner-group-cv --gcv-dof-cap 0.9 --cells "$CELLS_CSV")
  for model in "${MODELS[@]}"; do
    uv run python scripts/issue1417_battery.py --anchors --model "$model" "${refit_common[@]}"
    uv run python scripts/issue1417_battery.py --fits --model "$model" "${refit_common[@]}"
    uv run python scripts/issue1417_battery.py --battery --model "$model" "${refit_common[@]}"
  done
  uv run python scripts/issue1417_battery.py --summary "${refit_common[@]}"
  # Selector-log presence assert: the refit flags must be visibly threaded —
  # a refit cell JSON carries the per-(layer, fold) selector record and the
  # summary carries refit_config (fail loud here, not on the pod).
  uv run python - "$out_dir/refit" <<'PY'
import json
import sys
from pathlib import Path

refit = Path(sys.argv[1])
cells = sorted(refit.glob("cells/cells_*.json"))
with_sel = [
    p for p in cells if "selector_per_layer_fold" in json.loads(p.read_text())
]
assert with_sel, f"no refit cell JSON carries selector_per_layer_fold ({len(cells)} cells)"
summary = json.loads((refit / "battery_summary.json").read_text())
assert summary.get("refit_config", {}).get("lambda_selection") == "inner-group-cv", (
    "refit summary missing refit_config.lambda_selection"
)
print(f"[i1417-run] refit selector-log assert PASS ({len(with_sel)}/{len(cells)} cells)")
PY

  write_sentinel "epm:smoke-result" "smoke" "$out_dir/battery_summary.json" "smoke"
  log "smoke pipeline complete (root: $smoke_root)"
}

main() {
  local t0=$SECONDS
  if [ "${I1417_CARRY_ONLY:-0}" = "1" ]; then
    # Verification affordance (VM, 0 GPU): run ONLY the milder carry step —
    # OUT_DIR/BAT_OUT_DIR overridable so a scratch out-dir exercises the
    # exact production cp -n + assert block against the committed refit dir.
    BAT_OUT_DIR="${BAT_OUT_DIR:-$OUT_DIR/milder_rude}"
    carry_milder_inputs
    echo "[phase=done]"
    return 0
  fi
  echo "[phase=p0_stage]"
  log "issue=$ISSUE phase=${PHASE:-smoke-only} SMOKE=$SMOKE NGPUS=$NGPUS repo=$REPO_ROOT"
  uv run python scripts/issue1417_render.py --fetch-questions --data-dir "$DATA_DIR"
  uv run python scripts/issue1417_render.py --fetch-sidecars --data-dir "$DATA_DIR"
  uv run python scripts/issue1417_render.py --write-config --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
  uv run python scripts/issue1417_render.py --span-self-test

  if [ "$SMOKE" = "1" ]; then
    run_smoke
    echo "[phase=done]"
    return 0
  fi

  case "$PHASE" in
    A)
      headroom "$DATA_DIR" 65 "phase-a"
      if [ "$SKIP_SMOKE" != "1" ]; then
        run_smoke
      fi
      echo "[phase=pA_full]"
      run_phase_a "$DATA_DIR" full
      write_sentinel "epm:progress" "phase-a" "none" "$(awk -v s=$((SECONDS - t0)) 'BEGIN{printf "%.2f", s/3600}')"
      ;;
    C)
      headroom "$DATA_DIR" 125 "phase-c"
      local refit_args=()
      if [ "$REFIT" = "1" ] && [ "$MILDER" = "1" ]; then
        log "FATAL: I1417_REFIT and I1417_MILDER are mutually exclusive"
        exit 5
      fi
      if [ "$REFIT" = "1" ]; then
        # Registered-selector refit (see the header block): versioned out-dir,
        # v1 judge kept-sets read-only, registered mitigations threaded into
        # every fit site. v1 outputs are never resumed over (different out-dir)
        # and never overwritten (writes land under refit/ only).
        BAT_OUT_DIR="$OUT_DIR/refit"
        mkdir -p "$BAT_OUT_DIR"
        refit_args=(--judge-dir "$OUT_DIR" --lambda-selection inner-group-cv --gcv-dof-cap 0.9)
        log "REFIT mode: battery out-dir=$BAT_OUT_DIR judge-dir=$OUT_DIR"
      elif [ "$MILDER" = "1" ]; then
        # Milder-rude amendment round (plan v6 §4.2 item 4b): versioned
        # out-dir, carried v1 kept-sets + refit fits/battery merged via the
        # carry step, the refit's registered selector on the mild fits, the
        # H-table rude slot fed by the mild cell. Judge root = the milder dir
        # (carried + mild kept-sets both live there after the carry).
        if [[ ",$CELLS_CSV," != *",c2_rude_mild,"* ]]; then
          log "FATAL: I1417_MILDER=1 requires I1417_CELLS to include c2_rude_mild"
          exit 5
        fi
        BAT_OUT_DIR="$OUT_DIR/milder_rude"
        mkdir -p "$BAT_OUT_DIR"
        refit_args=(--judge-dir "$BAT_OUT_DIR" --lambda-selection inner-group-cv \
          --gcv-dof-cap 0.9 --h-rude-cell c2_rude_mild)
        log "MILDER mode: battery out-dir=$BAT_OUT_DIR judge-dir=$BAT_OUT_DIR cells=$CELLS_CSV"
      fi
      echo "[phase=pc_stage_inputs]"
      stage_phase_c_inputs
      if [ "$MILDER" = "1" ]; then
        carry_milder_inputs
      fi
      uv run python scripts/issue1417_battery.py --gate-g2 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
      # auto_descope_to_null_draws_100 (epm:compute-deviation v3, #1417): post-fix
      # pilot still projects 8236 s/lane > the 7200 s rc-7 abort threshold (the
      # serial CPU-randn floor is n-independent); halving both null-draw counts
      # brings the lane to ~4.9k s (ratio 1.37 <= 1.5) while preserving every
      # planned pair/cell/arm. Restore 200-draw nulls via these env overrides.
      # pilot-budget re-ground (epm:compute-deviation v4, #1417): production pilot
      # measured 601 s/pair WITH the 100-draw descope applied (n_draws=100 verified
      # in the crash-persisted battery JSON) — the residual is production-n fit
      # cost, not draws. Projected 7813 s/lane vs the old 7200 s abort (rc 7,
      # 2026-07-18T15:14Z crash). Budget 1.5 h => abort at 3 h/lane: clears the
      # measured 2.17 h worst-case uniform projection with ~38% margin while
      # still bounding a genuine runaway (~10x the naive basis still aborts).
      run_phase_c_lanes full "${refit_args[@]}" \
        --cells "$CELLS_CSV" \
        --cosine-null-draws "${I1417_COSINE_NULL_DRAWS:-100}" \
        --collapse-null-draws "${I1417_COLLAPSE_NULL_DRAWS:-100}" \
        --pilot-budget-h "${I1417_PILOT_BUDGET_H:-1.5}"
      echo "[phase=pc_summary]"
      uv run python scripts/issue1417_battery.py --summary --data-dir "$DATA_DIR" \
        --out-dir "${BAT_OUT_DIR:-$OUT_DIR}" "${refit_args[@]}"
      if [ "$REFIT" != "1" ] && [ "$MILDER" != "1" ]; then
        echo "[phase=pc_figures]"
        uv run python scripts/issue1417_figures.py --out-dir "$OUT_DIR" --fig-dir "$FIG_DIR"
      else
        log "figures skipped in refit/milder mode (analyzer regenerates from the round's JSONs)"
      fi
      echo "[phase=pc_finalize]"
      commit_and_push_results
      if [ -n "${EPS_DELIVERABLES_OK_PATH:-}" ]; then
        date -u +%Y-%m-%dT%H:%M:%SZ > "$EPS_DELIVERABLES_OK_PATH"
        log "deliverables-ok stamped at $EPS_DELIVERABLES_OK_PATH (post push-verify)"
      fi
      write_sentinel "epm:results" "results" "${BAT_OUT_DIR:-$OUT_DIR}/battery_summary.json" \
        "$(awk -v s=$((SECONDS - t0)) 'BEGIN{printf "%.2f", s/3600}')" "${BAT_OUT_DIR:-$OUT_DIR}"
      ;;
    *)
      log "FATAL: set I1417_PHASE=A|C (or SMOKE=1)"
      exit 5
      ;;
  esac
  echo "[phase=done]"
}

main "$@"
