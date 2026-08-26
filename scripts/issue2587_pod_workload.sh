#!/usr/bin/env bash
# Task #2587 production pod workload (plan v3 §4.7 DAG / §9 / §10).
#
# Invoked by the plan-§9 dispatch:
#   uv run python scripts/dispatch_issue.py launch --issue 2587 --intent eval \
#     --gpus 2 --gpu-type H100 --backend runpod --repo-branch issue-2587 \
#     --workload-cmd "bash scripts/issue2587_pod_workload.sh" --time-budget-hours 16
#
# Phase order (each phase = single python process, explicit exit terminals;
# failure = that process's own non-zero exit observed by the poller — plan
# §4.7 "Embedded-shell exit-path note"):
#   bootstrap        git fetch issue-2564 objects, §4.1 model-venv build,
#                    driver-version gate (cuda-compat remediation + fail-loud
#                    verify via issue2587_common.assert_driver_compat)
#   p0b_gates        template_pin -> length_scan -> hook_probe (model venv;
#                    writes <out-root>/split_ids_done.json once all 3 pass)
#   p1_smoke         500-row smoke shard gen+capture (vLLM + fp32 HF capture,
#                    --no-upload) -> --fits-smoke (repo-venv fits port on the
#                    local chunk) -> tiny battery cell (register axis,
#                    3 carriers, K=2; generate_batch engine leg) ->
#                    --p1-apply-probe (repo venv: apply_map(random payload) +
#                    reads on the tiny stores) -> --gate compose_p1 (model
#                    venv: full §4.7 P1 check-set verify; writes
#                    compat_smoke_report.json ALWAYS and
#                    <out-root>/compat_smoke_done.json ONLY on all-PASS)
#   p2_map_gen       ~29.4k rollouts, 6 splits x 2 shards — 12 independent
#                    tasks consumed by 2 persistent CVD-pinned per-GPU
#                    workers over a shared queue (work-conserving; r3 Codex
#                    Major non-work-conserving-map-split-barriers)
#   p3_map_capture   dense fp32 capture, layers 0-31 — same 12-task
#                    work-conserving worker shape as p2
#   p4_fits          ridge fits layer-sharded 0-15/16-31 (repo venv) + finalize
#                    (--payloads-prefix/--preds-prefix explicit — fits.py
#                    refuses --upload hf without them, r3 Codex Critical 1)
#   p5_battery_gen   10,800 rollouts, 2 shards by axis (generate_batch)
#   p6_battery_capture  capture 2 shards + single-GPU embed (repo venv,
#                    vLLM 0.11.0 — §4.4 instrument-version parity DEFAULT)
#   p8_matched7b     matched-capacity 7B arm (repo venv, single GPU)
#   leak_caphit_harvest  copy battery gen done-manifests + aggregate per-split
#                    cap-hit JSONs into eval_results/issue_2587/leak_caphit/
#                    (CPU-only, repo venv; feeds think_leak_cap_hit_table's
#                    VM default --leak-caphit-dir — r2 concern
#                    leak-caphit-manifests-not-in-harvest-set)
#   results_push     commit + push pod-side eval JSONs incl. the harvested
#                    leak_caphit/ set (#1205 verification), HF mirror to
#                    issue2587_q35_map/manifests/, epm:results sentinel,
#                    then the single terminal [phase=done]
#
# P1 enforcement (round-2/3 blocker `compat-gate-not-enforced`): every
# production wave entry (6 waves: p2/p3/p4/p5/p6/p8) calls require_p1, which
# verifies the FULL sentinel contract — schema issue2587_compat_smoke_v2,
# issue/phase/status, the report's recorded sha256, and CODE IDENTITY
# (map_code_sha256 == the driver bytes about to run) — never the bare status
# token (r3 Codex Critical 2). The sentinel itself is written only by
# `--gate compose_p1` after verifying the FULL §4.7 P1 check set (venv pins +
# banned dists in the model interpreter, driver gate, run_meta PASS records
# for template_pin/length_scan/hook_probe/smoke_shard_gen/smoke_shard_capture/
# fits_smoke/apply_probe, the mode-specific smoke-shard measured fields, and
# the tiny-battery manifest geometry/counts).
#
# CVD discipline (§9): the drivers never set CUDA_VISIBLE_DEVICES; this
# launcher pins it in the env per shard (gotchas.md import-time-cuInit rule).
# Env pins + venv paths are DERIVED from issue2587_common at runtime (never
# retyped here); every model step runs
#   env <LAUNCH_ENV_PINS> CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=<repo>/src \
#     /root/eps-model-venv/bin/python scripts/<driver>.py ...
# with stdout+stderr redirected to a per-phase log under $LOGS_DIR (the
# [phase=done] token stays reserved for this launcher's single terminal).
#
# Dry-run (CPU-testable control flow): EPM_I2587_DRYRUN=1 echoes every
# command (with its log redirect) instead of executing; host-side gates
# (nvidia-smi, sentinels, git push) are echoed as [dryrun] lines. Tests set
# EPM_I2587_OUT_ROOT/EPM_I2587_LOGS_DIR to tmp dirs (the /workspace defaults
# are pod-only paths).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${EPM_I2587_OUT_ROOT:-/workspace/eps2587}"
LOGS_DIR="${EPM_I2587_LOGS_DIR:-/workspace/logs}"
HF_PREFIX="${EPM_I2587_HF_PREFIX:-issue2587_q35_map/qwen35_9b}"
BATTERY_PREFIX="${EPM_I2587_BATTERY_PREFIX:-issue2587_minpair}"
# Plan-§6.5/§10 HF destination prefixes for the P4/P8 fits phases —
# issue2587_fits.py REFUSES `--upload hf` without them (deliberate
# fail-loud None defaults, the #1005 upload-prefix clobber shape), so the
# launcher passes them explicitly (r3 Codex Critical 1: the r2 launcher
# omitted them and production would abort AFTER the 32-layer fits).
PAYLOADS_PREFIX="${EPM_I2587_PAYLOADS_PREFIX:-issue2587_q35_map/analysis_tensors/ridge_payloads}"
PREDS_PREFIX="${EPM_I2587_PREDS_PREFIX:-issue2587_q35_map/analysis_tensors/preds}"
PREDS7B_PREFIX="${EPM_I2587_PREDS7B_PREFIX:-issue2587_minpair/analysis_tensors/preds_7b_matched}"
BRANCH="${EPM_I2587_BRANCH:-issue-2587}"
DRYRUN="${EPM_I2587_DRYRUN:-}"

BATTERY_ROOT="$OUT_ROOT/battery"
# Per-leg out-roots for the P1 smoke legs (#1333: never share a resume-keyed
# root between a smoke leg and the production leg).
P1_SMOKE_ROOT="$OUT_ROOT/p1_smoke"
P1_BATTERY_ROOT="$OUT_ROOT/p1_battery"
RUN_META="$OUT_ROOT/run_meta.json"
P0B_SENTINEL="$OUT_ROOT/split_ids_done.json"
COMPAT_SENTINEL="$OUT_ROOT/compat_smoke_done.json"
COMPAT_REPORT="$REPO_ROOT/eval_results/issue_2587/compat_smoke_report.json"
SPLIT_IDS="$REPO_ROOT/eval_results/issue_2587/split_ids.json"

MAP="scripts/issue2587_map_gen_capture.py"
BATTERY="scripts/issue2587_battery_run.py"
FITS="scripts/issue2587_fits.py"

# Plan §4.3/§10: the P2/P3 split set (4 consumed splits + the two
# test_1000 ceiling draws, seeds 43/44 — SPLIT_TO_MANIFEST keys).
SPLITS=(train_25k val_400 test_1000 wc_test_1k ceiling_draw_43 ceiling_draw_44)

# Pod-side eval JSONs the results_push phase commits + mirrors (plan §10).
RESULT_JSONS=(
  eval_results/issue_2587/split_ids.json
  eval_results/issue_2587/compat_smoke_report.json
  eval_results/issue_2587/map_layer_sweep.json
  eval_results/issue_2587/matched7b_anchor.json
)

phase() { printf '[phase=%s]\n' "$1"; }

run_logged() {
  # run_logged <log-file> <cmd...> — foreground; stdout+stderr to the log
  # (the [phase=done]-reserved token in any phase script never reaches the
  # main workload log); on failure echo the log tail and exit with the rc.
  local log="$1" rc=0
  shift
  if [ -n "$DRYRUN" ]; then
    # Newlines in embedded python -c payloads collapse to spaces so every
    # dry-run command echoes as ONE greppable line.
    local joined="$*"
    printf '[dryrun] %s > %s\n' "${joined//$'\n'/ }" "$log"
    return 0
  fi
  mkdir -p "$(dirname "$log")"
  echo "[workload] run: $* (log: $log)"
  "$@" > "$log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[workload] FAILED rc=$rc: $*" >&2
    echo "[workload] tail of $log:" >&2
    tail -n 120 "$log" >&2 || true
    exit "$rc"
  fi
}

LAST_BG_PID=""
launch_bg() {
  # launch_bg <log-file> <cmd...> — background; pid in $LAST_BG_PID.
  local log="$1"
  shift
  if [ -n "$DRYRUN" ]; then
    local joined="$*"
    printf '[dryrun-bg] %s > %s\n' "${joined//$'\n'/ }" "$log"
    LAST_BG_PID=""
    return 0
  fi
  mkdir -p "$(dirname "$log")"
  echo "[workload] bg: $* (log: $log)"
  "$@" > "$log" 2>&1 &
  LAST_BG_PID=$!
}

wait_bg() {
  # wait_bg <pid> <label> <log-file> — fail-loud join of one launch_bg.
  [ -n "$DRYRUN" ] && return 0
  local pid="$1" label="$2" log="$3" rc=0
  wait "$pid" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[workload] FAILED $label rc=$rc — tail of $log:" >&2
    tail -n 120 "$log" >&2 || true
    exit "$rc"
  fi
  echo "[workload] done: $label"
}

assert_file() {
  # assert_file <path> <label> — fail-loud existence gate between phases.
  [ -n "$DRYRUN" ] && { printf '[dryrun] assert_file %s (%s)\n' "$1" "$2"; return 0; }
  if [ ! -f "$1" ]; then
    echo "[workload] FATAL: $2 did not produce $1" >&2
    exit 3
  fi
}

write_sentinel() {
  # write_sentinel <path> <phase-name> — plan-§9 phase_outputs sentinel
  # (atomic; written by the launcher AFTER the phase's wave completes).
  if [ -n "$DRYRUN" ]; then
    printf '[dryrun] write_sentinel %s (%s)\n' "$1" "$2"
    return 0
  fi
  uv run python -c 'import sys, time
from explore_persona_space.atomic_io import write_json_atomic
path, name = sys.argv[1], sys.argv[2]
payload = {
    "schema": "issue2587_phase_sentinel_v1",
    "issue": 2587,
    "phase": name,
    "status": "PASS",
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
write_json_atomic(path, payload)
print("[sentinel] wrote", path)' "$1" "$2"
}

require_p1() {
  # require_p1 <next-phase> — the round-2/3 enforcement arm: every expensive
  # phase entry re-VERIFIES the FULL-P1 compat sentinel, not just its status
  # token (r3 Codex Critical 2: boolean-only composite-gate evidence):
  # schema/issue/phase/status, the report's recorded sha256 (the sentinel
  # attests THAT report), and code identity (map_code_sha256 == the driver
  # bytes about to run — a mid-run code change invalidates the P1 verdict
  # until compose_p1 re-runs). Stdlib-only payload (json+hashlib), single-
  # quoted with a double-quote-only body so the tests can extract + execute
  # the EXACT production payload against fixture sentinels.
  if [ -n "$DRYRUN" ]; then
    printf '[dryrun] require_p1 before %s\n' "$1"
    return 0
  fi
  uv run python -c 'import hashlib, json, sys
sent_path, report_path, map_path, nxt = sys.argv[1:5]

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

try:
    d = json.load(open(sent_path, encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit("[p1-gate] REFUSED %s: compat sentinel %s absent" % (nxt, sent_path))
for key, want in (
    ("schema", "issue2587_compat_smoke_v2"),
    ("issue", 2587),
    ("phase", "P1"),
    ("status", "PASS"),
):
    got = d.get(key)
    assert got == want, "[p1-gate] REFUSED %s: %s %s=%r != %r" % (nxt, sent_path, key, got, want)
try:
    report_sha = _sha(report_path)
except FileNotFoundError:
    raise SystemExit("[p1-gate] REFUSED %s: compat report %s absent" % (nxt, report_path))
assert report_sha == d.get("report_sha256"), (
    "[p1-gate] REFUSED %s: report %s sha256 %s != sentinel-recorded %r (report rewritten "
    "since compose_p1)" % (nxt, report_path, report_sha, d.get("report_sha256")))
code_sha = _sha(map_path)
assert code_sha == d.get("map_code_sha256"), (
    "[p1-gate] REFUSED %s: %s sha256 %s != sentinel-recorded %r (driver code changed since "
    "compose_p1 — re-run the P1 smoke)" % (nxt, map_path, code_sha, d.get("map_code_sha256")))
print("[p1-gate] OK before %s: %s (schema+report+code identity verified)" % (nxt, sent_path))' \
    "$COMPAT_SENTINEL" "$COMPAT_REPORT" "$MAP" "$1"
}

pop_task() {
  # pop_task <queue-file> — atomically pop + print the queue's first line
  # (flock-serialized against the sibling GPU worker; the lock file lives
  # beside the queue on container-local /tmp, never MooseFS); rc=1 when the
  # queue is empty.
  local queue="$1" task
  {
    flock 9
    task="$(head -n 1 "$queue" 2>/dev/null || true)"
    if [ -z "$task" ]; then
      return 1
    fi
    tail -n +2 "$queue" > "$queue.next" && mv "$queue.next" "$queue"
    printf '%s\n' "$task"
  } 9>>"$queue.lock"
}

poison_queue() {
  # poison_queue <queue-file> — empty the task queue: a FAILED worker calls
  # this so the sibling worker stops after its in-flight task instead of
  # consuming the dead wave's remaining splits (bounds the orphan window;
  # the launcher still exits non-zero at the failed worker's wait_bg).
  local queue="$1"
  {
    flock 9
    : > "$queue"
  } 9>>"$queue.lock"
}

map_wave_worker() {
  # map_wave_worker <gpu> <capture-mode> <queue-file> <label> — persistent
  # per-GPU worker (r3 Codex Major non-work-conserving-map-split-barriers):
  # consumes "split shard" tasks until the queue drains, so a GPU whose
  # task finishes early immediately picks up the next dependency-ready task
  # instead of idling at a per-split two-shard barrier. CVD is pinned per
  # WORKER process in the launcher env (gotchas.md import-time-cuInit
  # rule); failure stays fail-loud — the first task failure poisons the
  # queue and exits the worker with the task's rc, which wait_bg observes.
  local gpu="$1" mode="$2" queue="$3" label="$4"
  local task split shard log rc
  while task="$(pop_task "$queue")"; do
    split="${task% *}"
    shard="${task#* }"
    log="$LOGS_DIR/issue-2587-$label-$split-shard$shard.log"
    echo "[workload] $label gpu$gpu: --split $split --shard-index $shard (log: $log)"
    rc=0
    env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_ROOT/src" \
      "$MODEL_PY" "$MAP" --split "$split" --capture-mode "$mode" \
      --num-shards 2 --shard-index "$shard" --hf-prefix "$HF_PREFIX" --h-dim 4096 \
      --out-dir "$OUT_ROOT" --run-meta-out "$RUN_META" \
      --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v > "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "[workload] FAILED $label $split shard$shard rc=$rc — tail of $log:" >&2
      tail -n 120 "$log" >&2 || true
      poison_queue "$queue"
      exit "$rc"
    fi
    echo "[workload] done: $label $split shard$shard"
  done
}

run_map_wave() {
  # run_map_wave <capture-mode> <label> — enqueue ALL 12 dependency-ready
  # tasks (6 splits x 2 shards; every task within a wave is independent —
  # shards own disjoint row ranges, splits are disjoint id sets), then run
  # ONE persistent worker per GPU over the shared queue (work-conserving:
  # an idle GPU + a pending task never coexist; code-style.md
  # multi-worker-dispatcher rule, #813/#778).
  local mode="$1" label="$2"
  if [ -n "$DRYRUN" ]; then
    # Static dry-run view: one command echo per (split, shard). The echoed
    # CVD uses the shard index; at runtime the GPU is whichever worker pops
    # the task — both workers carry a launcher-pinned CVD.
    local split shard
    for split in "${SPLITS[@]}"; do
      for shard in 0 1; do
        launch_bg "$LOGS_DIR/issue-2587-$label-$split-shard$shard.log" \
          env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES="$shard" PYTHONPATH="$REPO_ROOT/src" \
          "$MODEL_PY" "$MAP" --split "$split" --capture-mode "$mode" \
          --num-shards 2 --shard-index "$shard" --hf-prefix "$HF_PREFIX" --h-dim 4096 \
          --out-dir "$OUT_ROOT" --run-meta-out "$RUN_META" \
          --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v
      done
    done
    return 0
  fi
  # Ephemeral coordination state on container-local /tmp (pid-unique name):
  # no resume value, and flock semantics stay kernel-local off the FUSE mount.
  local queue="${TMPDIR:-/tmp}/issue-2587-$label-task-queue.$$"
  rm -f "$queue" "$queue.lock" "$queue.next"
  local split shard
  for split in "${SPLITS[@]}"; do
    for shard in 0 1; do
      printf '%s %s\n' "$split" "$shard" >> "$queue"
    done
  done
  launch_bg "$LOGS_DIR/issue-2587-$label-worker-gpu0.log" \
    map_wave_worker 0 "$mode" "$queue" "$label"
  local w0="$LAST_BG_PID"
  launch_bg "$LOGS_DIR/issue-2587-$label-worker-gpu1.log" \
    map_wave_worker 1 "$mode" "$queue" "$label"
  local w1="$LAST_BG_PID"
  wait_bg "$w0" "$label worker gpu0" "$LOGS_DIR/issue-2587-$label-worker-gpu0.log"
  wait_bg "$w1" "$label worker gpu1" "$LOGS_DIR/issue-2587-$label-worker-gpu1.log"
  rm -f "$queue" "$queue.lock" "$queue.next"
}

# ── bootstrap ───────────────────────────────────────────────────────────────
phase bootstrap
mkdir -p "$OUT_ROOT" "$LOGS_DIR"

# §4.1 pins + venv paths DERIVED from issue2587_common (never retyped).
# Runs in the repo venv (issue2587_common re-exports the issue2378_common
# pin constants by import). Real even under dry-run so the echoed commands
# carry the true pins.
PIN_LINES="$(uv run python -c '
import sys
sys.path.insert(0, "scripts")
import issue2587_common as cm
print(" ".join("%s=%s" % (k, v) for k, v in sorted(cm.LAUNCH_ENV_PINS.items())))
print(cm.model_python())
print(cm.MODEL_DRIVER_FLOOR_MAJOR)
print(cm.CUDA_COMPAT_DIR)
')"
ENV_PINS_LINE="$(printf '%s\n' "$PIN_LINES" | sed -n 1p)"
MODEL_PY="$(printf '%s\n' "$PIN_LINES" | sed -n 2p)"
DRIVER_FLOOR_MAJOR="$(printf '%s\n' "$PIN_LINES" | sed -n 3p)"
COMPAT_DIR="$(printf '%s\n' "$PIN_LINES" | sed -n 4p)"
read -r -a ENV_PINS <<< "$ENV_PINS_LINE"
echo "[workload] launch env pins: ${ENV_PINS[*]}"
echo "[workload] model interpreter: $MODEL_PY (driver floor major $DRIVER_FLOOR_MAJOR)"

# Pinned-blob imports (bank2587 pins bank2564 at the parent branch blob):
# fetch the issue-2564 objects up front (partial-clone promisor fetch).
run_logged "$LOGS_DIR/issue-2587-git-fetch.log" git fetch origin issue-2564

# §4.1 model-venv build (idempotent; issue2378_dispatch._build_model_venv —
# pin install + banned-dist uninstall LAST + realized-pin log).
run_logged "$LOGS_DIR/issue-2587-venv-build.log" uv run python -c '
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import issue2587_common as cm
cm.build_model_venv(Path(sys.argv[1]))
print("[venv-build] OK:", cm.MODEL_VENV_DEFAULT)' "$LOGS_DIR"

# Driver-version gate: remediate (cuda-compat install + LD_LIBRARY_PATH,
# the #2330 recipe) when the host driver is below the floor, then verify
# fail-loud through the SAME shared gate every model step relies on.
if [ -z "$DRYRUN" ]; then
  DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)"
  if [ -z "$DRIVER_MAJOR" ]; then
    echo "[workload] FATAL: nvidia-smi returned no driver version" >&2
    exit 3
  fi
  if [ "$DRIVER_MAJOR" -lt "$DRIVER_FLOOR_MAJOR" ]; then
    echo "[workload] driver major $DRIVER_MAJOR < $DRIVER_FLOOR_MAJOR — installing cuda-compat-13-0"
    run_logged "$LOGS_DIR/issue-2587-cuda-compat.log" apt-get install -y cuda-compat-13-0
    export LD_LIBRARY_PATH="$COMPAT_DIR:${LD_LIBRARY_PATH:-}"
  fi
else
  printf '[dryrun] driver_gate (nvidia-smi major vs %s; cuda-compat remediation)\n' "$DRIVER_FLOOR_MAJOR"
fi
run_logged "$LOGS_DIR/issue-2587-driver-gate.log" uv run python -c '
import sys
sys.path.insert(0, "scripts")
import issue2587_common as cm
cm.assert_driver_compat()
print("[driver-gate] OK")'

# ── P0b: q35 convention gates (model venv; CPU-on-pod tokenizer work +
#          one GPU model load for hook_probe) ───────────────────────────────
phase p0b_gates
run_logged "$LOGS_DIR/issue-2587-gate-template-pin.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$MAP" --gate template_pin \
  --out-dir "$OUT_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v
run_logged "$LOGS_DIR/issue-2587-gate-length-scan.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$MAP" --gate length_scan \
  --out-dir "$OUT_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v
run_logged "$LOGS_DIR/issue-2587-gate-hook-probe.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$MAP" --gate hook_probe --h-dim 4096 \
  --out-dir "$OUT_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v
assert_file "$P0B_SENTINEL" "P0b gates (template_pin+length_scan+hook_probe)"

# ── P1: compat smoke gate (§4.7 full check set) ─────────────────────────────
phase p1_smoke
# (a) 500-row smoke shard, BOTH sub-phases, production entrypoint at reduced
#     shard arithmetic (--num-shards 50 --shard-index 0 --shard-size 500
#     --no-upload — the driver's documented smoke/sweep-parity shape). The
#     vLLM 2-row-engine leg + think-leak scan of §4.7 ride this shard; the
#     capture sub-phase writes the run_meta `smoke_shard` record.
run_logged "$LOGS_DIR/issue-2587-p1-smoke-gen.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$MAP" --split train_25k --capture-mode phase_split_gen \
  --num-shards 50 --shard-index 0 --shard-size 500 --no-upload \
  --hf-prefix "$HF_PREFIX" --h-dim 4096 \
  --out-dir "$P1_SMOKE_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v
run_logged "$LOGS_DIR/issue-2587-p1-smoke-capture.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$MAP" --split train_25k --capture-mode phase_split_capture \
  --num-shards 50 --shard-index 0 --shard-size 500 --no-upload \
  --hf-prefix "$HF_PREFIX" --h-dim 4096 \
  --out-dir "$P1_SMOKE_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v
# (b) fits smoke: the REAL P3 fits port on the local 500-row chunk (the map
#     driver subprocesses `uv run` — repo venv — itself; plan §4 P1 step 4).
run_logged "$LOGS_DIR/issue-2587-p1-fits-smoke.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$MAP" --fits-smoke --split train_25k \
  --out-dir "$P1_SMOKE_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v
# (c) tiny end-to-end battery cell (register axis, 3 carriers, K=2): the
#     generate_batch engine leg + auto-multimodal loader + 32-block assert.
#     --upload none: the local stores must survive for the apply-probe (the
#     production hf upload path deletes local bytes after verified upload;
#     P5/P6 exercise it).
run_logged "$LOGS_DIR/issue-2587-p1-battery-gen.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase gen --axes register --max-carriers 3 \
  --draws 2 --out-root "$P1_BATTERY_ROOT" --upload none
run_logged "$LOGS_DIR/issue-2587-p1-battery-capture.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase capture --axes register --max-carriers 3 \
  --draws 2 --out-root "$P1_BATTERY_ROOT" --upload none
# (d) apply_map(random payload) -> reads on the tiny stores (repo venv: the
#     issue779 fit module's import closure is repo-pinned).
run_logged "$LOGS_DIR/issue-2587-p1-apply-probe.log" \
  uv run python "$MAP" --p1-apply-probe \
  --p1-battery-root "$P1_BATTERY_ROOT" --p1-smoke-cell register \
  --p1-apply-layer 22 \
  --out-dir "$OUT_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v
# (e) compose the FULL P1 verdict in the MODEL interpreter (realized pins +
#     banned dists + driver gate + every run_meta record + battery
#     manifests); writes the report ALWAYS, the sentinel only on all-PASS.
run_logged "$LOGS_DIR/issue-2587-p1-compose.log" \
  env "${ENV_PINS[@]}" PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$MAP" --gate compose_p1 \
  --p1-battery-root "$P1_BATTERY_ROOT" --p1-smoke-cell register \
  --p1-report-out "$COMPAT_REPORT" --p1-sentinel-out "$COMPAT_SENTINEL" \
  --out-dir "$OUT_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$P0B_SENTINEL" --split-ids "$SPLIT_IDS" -v
assert_file "$COMPAT_SENTINEL" "P1 compose (compat_smoke_done)"
assert_file "$COMPAT_REPORT" "P1 compose (compat_smoke_report)"

# ── P2: map-fit generation (vLLM; 12 tasks, 2 work-conserving GPU workers) ──
require_p1 p2_map_gen
phase p2_map_gen
run_map_wave phase_split_gen p2
write_sentinel "$OUT_ROOT/map_gen_done.json" p2_map_gen

# ── P3: map-fit dense capture (fp32, layers 0-31; same worker shape) ────────
require_p1 p3_map_capture
phase p3_map_capture
run_map_wave phase_split_capture p3
write_sentinel "$OUT_ROOT/map_capture_done.json" p3_map_capture

# ── P4: ridge fits + floors + kNN, layer-sharded 0-15 / 16-31 (repo venv) ──
require_p1 p4_fits
phase p4_fits
# --store-prefix threads the (env-overridable) 9B store root into the fits
# consumer so a non-default EPM_I2587_HF_PREFIX run reads the SAME prefix
# P2/P3 wrote (r2 g2 concern 1; defaults are byte-equal). The finalize leg
# passes the plan-§6.5 destination prefixes EXPLICITLY — fits.py raises on
# `--upload hf` without them (r3 Codex Critical 1).
launch_bg "$LOGS_DIR/issue-2587-p4-fits-l0-15.log" \
  env CUDA_VISIBLE_DEVICES=0 \
  uv run python "$FITS" --phase fits --layers 0-15 --device cuda \
  --h-dim 4096 --store-prefix "$HF_PREFIX" --upload hf -v
P4_PID0="$LAST_BG_PID"
launch_bg "$LOGS_DIR/issue-2587-p4-fits-l16-31.log" \
  env CUDA_VISIBLE_DEVICES=1 \
  uv run python "$FITS" --phase fits --layers 16-31 --device cuda \
  --h-dim 4096 --store-prefix "$HF_PREFIX" --upload hf -v
P4_PID1="$LAST_BG_PID"
wait_bg "$P4_PID0" "p4 fits layers 0-15" "$LOGS_DIR/issue-2587-p4-fits-l0-15.log"
wait_bg "$P4_PID1" "p4 fits layers 16-31" "$LOGS_DIR/issue-2587-p4-fits-l16-31.log"
run_logged "$LOGS_DIR/issue-2587-p4-finalize.log" \
  env CUDA_VISIBLE_DEVICES=0 \
  uv run python "$FITS" --phase finalize --device cuda --h-dim 4096 \
  --store-prefix "$HF_PREFIX" --upload hf \
  --payloads-prefix "$PAYLOADS_PREFIX" --preds-prefix "$PREDS_PREFIX" \
  --sentinel-path "$OUT_ROOT/fits_done.json" -v
assert_file "$OUT_ROOT/fits_done.json" "P4 finalize"

# ── P5: battery + pilot generation (generate_batch, 2 shards by axis) ──────
require_p1 p5_battery_gen
phase p5_battery_gen
launch_bg "$LOGS_DIR/issue-2587-p5-gen-shard0.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase gen --num-shards 2 --shard-index 0 \
  --out-root "$BATTERY_ROOT" --upload hf --hf-prefix "$BATTERY_PREFIX"
P5_PID0="$LAST_BG_PID"
launch_bg "$LOGS_DIR/issue-2587-p5-gen-shard1.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase gen --num-shards 2 --shard-index 1 \
  --out-root "$BATTERY_ROOT" --upload hf --hf-prefix "$BATTERY_PREFIX"
P5_PID1="$LAST_BG_PID"
wait_bg "$P5_PID0" "p5 battery gen shard0" "$LOGS_DIR/issue-2587-p5-gen-shard0.log"
wait_bg "$P5_PID1" "p5 battery gen shard1" "$LOGS_DIR/issue-2587-p5-gen-shard1.log"
write_sentinel "$OUT_ROOT/battery_gen_done.json" p5_battery_gen

# ── P6: battery capture (2 shards) + embed (single GPU, repo venv) ──────────
require_p1 p6_battery_capture
phase p6_battery_capture
launch_bg "$LOGS_DIR/issue-2587-p6-capture-shard0.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase capture --num-shards 2 --shard-index 0 \
  --out-root "$BATTERY_ROOT" --upload hf --hf-prefix "$BATTERY_PREFIX"
P6_PID0="$LAST_BG_PID"
launch_bg "$LOGS_DIR/issue-2587-p6-capture-shard1.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase capture --num-shards 2 --shard-index 1 \
  --out-root "$BATTERY_ROOT" --upload hf --hf-prefix "$BATTERY_PREFIX"
P6_PID1="$LAST_BG_PID"
wait_bg "$P6_PID0" "p6 battery capture shard0" "$LOGS_DIR/issue-2587-p6-capture-shard0.log"
wait_bg "$P6_PID1" "p6 battery capture shard1" "$LOGS_DIR/issue-2587-p6-capture-shard1.log"
# Embed under the REPO venv (vLLM 0.11.0 — §4.4 instrument-version-parity
# DEFAULT; battery_run's engine-parity assert refuses 0.27.1 without a
# parity report). Single-GPU per §9.
run_logged "$LOGS_DIR/issue-2587-p6-embed.log" \
  env CUDA_VISIBLE_DEVICES=0 \
  uv run python "$BATTERY" --phase embed \
  --out-root "$BATTERY_ROOT" --upload hf --hf-prefix "$BATTERY_PREFIX"
write_sentinel "$OUT_ROOT/battery_capture_done.json" p6_battery_capture

# ── P8: matched-capacity 7B arm (repo venv, single GPU) ─────────────────────
require_p1 p8_matched7b
phase p8_matched7b
# --preds7b-prefix is the plan-§6.5 matched-7B destination — fits.py raises
# on `--upload hf` without it (r3 Codex Critical 1). matched7b reads the
# BANKED 7B store via its own pinned --hf-prefix-7b/--revision-7b defaults
# (it never consumes --store-prefix), so no store threading here.
run_logged "$LOGS_DIR/issue-2587-p8-matched7b.log" \
  env CUDA_VISIBLE_DEVICES=0 \
  uv run python "$FITS" --phase matched7b --device cuda --h-dim 4096 \
  --upload hf --preds7b-prefix "$PREDS7B_PREFIX" \
  --sentinel-path "$OUT_ROOT/matched7b_done.json" -v
assert_file "$OUT_ROOT/matched7b_done.json" "P8 matched7b"

# ── leak/cap-hit harvest (r2 concern leak-caphit-manifests-not-in-harvest-set)
# think_leak_cap_hit_table (issue2587_figures.py) rglobs anchors_*.done.json +
# cap_hit_*.json under the VM default --leak-caphit-dir
# (eval_results/issue_2587). The think-leak and per-split cap-hit fractions
# are plan-§4.3/§4.4 reportable run facts, so they belong in the harvested
# results: copy the battery gen done-manifests (pod-side under
# $BATTERY_ROOT/{,shard*/}manifests/) and AGGREGATE the per-split cap-hit
# fractions into eval_results/issue_2587/leak_caphit/, then ride them through
# the results_push commit + HF mirror below (CPU-only; repo venv, like the
# P1 apply-probe leg — --aggregate-cap-hit reads the P2 chunks from HF).
phase leak_caphit_harvest
LEAK_DIR="$REPO_ROOT/eval_results/issue_2587/leak_caphit"
if [ -n "$DRYRUN" ]; then
  printf '[dryrun] leak_caphit_harvest: cp %s/{,shard*/}manifests/anchors_*.done.json -> %s\n' \
    "$BATTERY_ROOT" "$LEAK_DIR"
  for split in "${SPLITS[@]}"; do
    printf '[dryrun] uv run python %s --aggregate-cap-hit --split %s --hf-prefix %s --split-ids %s --out-dir %s --cap-hit-out %s/cap_hit_%s.json -v > %s\n' \
      "$MAP" "$split" "$HF_PREFIX" "$SPLIT_IDS" "$OUT_ROOT" \
      "$LEAK_DIR" "$split" "$LOGS_DIR/issue-2587-caphit-$split.log"
  done
else
  mkdir -p "$LEAK_DIR"
  shopt -s nullglob
  GEN_MANIFESTS=(
    "$BATTERY_ROOT"/manifests/anchors_*.done.json
    "$BATTERY_ROOT"/shard*/manifests/anchors_*.done.json
  )
  shopt -u nullglob
  if [ "${#GEN_MANIFESTS[@]}" -eq 0 ]; then
    echo "[leak-caphit] FATAL: no anchors_*.done.json under $BATTERY_ROOT{,/shard*}/manifests" >&2
    exit 6
  fi
  # Collision guard is scoped WITHIN this run (r2 g5 concern 1): shards own
  # disjoint axes, so two THIS-RUN manifests sharing a basename with
  # differing bytes is a genuine shard fault (exit 6). A PRE-EXISTING $dest
  # is a prior round's committed copy cloned with the repo (or a truncated
  # copy from a crash mid-copy) and is SUPERSEDED unconditionally —
  # results_push commits the supersession as an ordinary diff; the old
  # unconditional guard made every fresh-pod relaunch after a successful
  # push die exit 6 with a shard-fault misdiagnosis. Copies are atomic
  # (tmp + mv, same filesystem) so a crash mid-copy can't strand a
  # truncated $dest either.
  declare -A SEEN_MANIFESTS=()
  for m in "${GEN_MANIFESTS[@]}"; do
    base="$(basename "$m")"
    dest="$LEAK_DIR/$base"
    if [ -n "${SEEN_MANIFESTS[$base]:-}" ]; then
      if ! cmp -s "$m" "$dest"; then
        echo "[leak-caphit] FATAL: done-manifest basename collision WITHIN this run:" \
          "$m vs ${SEEN_MANIFESTS[$base]} (shards must own disjoint axes)" >&2
        exit 6
      fi
      continue  # byte-identical within-run duplicate — harmless
    fi
    cp -f "$m" "$dest.tmp.$$" && mv -f "$dest.tmp.$$" "$dest"
    SEEN_MANIFESTS[$base]="$m"
  done
  echo "[leak-caphit] harvested ${#GEN_MANIFESTS[@]} battery gen done-manifests -> $LEAK_DIR"
  for split in "${SPLITS[@]}"; do
    run_logged "$LOGS_DIR/issue-2587-caphit-$split.log" \
      uv run python "$MAP" --aggregate-cap-hit --split "$split" \
      --hf-prefix "$HF_PREFIX" --split-ids "$SPLIT_IDS" \
      --out-dir "$OUT_ROOT" --cap-hit-out "$LEAK_DIR/cap_hit_$split.json" -v
    assert_file "$LEAK_DIR/cap_hit_$split.json" "cap-hit aggregate ($split)"
  done
fi
# Extend the results_push set with the harvested repo-relative paths so the
# leak/cap-hit inputs land on the branch + the HF manifests mirror.
if [ -d "$LEAK_DIR" ]; then
  while IFS= read -r f; do
    RESULT_JSONS+=("${f#"$REPO_ROOT/"}")
  done < <(find "$LEAK_DIR" -maxdepth 1 -type f -name '*.json' | sort)
fi

# ── results push + HF mirror + epm:results sentinel ─────────────────────────
phase results_push
for f in "${RESULT_JSONS[@]}"; do
  assert_file "$REPO_ROOT/$f" "results_push input"
done
if [ -n "$DRYRUN" ]; then
  printf '[dryrun] results_push: commit+push %s; verify rev-list==0 + per-file ls-tree on origin/%s\n' \
    "${RESULT_JSONS[*]}" "$BRANCH"
  printf '[dryrun] hf_mirror: %s -> issue2587_q35_map/manifests/\n' "${RESULT_JSONS[*]}"
  printf '[dryrun] epm_results sentinel -> %s\n' "$LOGS_DIR"
else
  # Result-push verification contract (#1205, pod-side-reporting.md):
  # commit by explicit path, fetch+rebase, bounded 2 push attempts, then
  # rev-list --count == 0 + per-file ls-tree presence on the remote ref.
  git add -- "${RESULT_JSONS[@]}"
  if git diff --cached --quiet -- "${RESULT_JSONS[@]}"; then
    echo "[results-push] no new eval-JSON changes to commit"
  else
    git commit -m "task #2587: pod-side eval JSONs (split_ids, compat report, layer sweep, matched7b anchor, leak_caphit set)" \
      -- "${RESULT_JSONS[@]}"
  fi
  PUSH_RC=1
  for attempt in 1 2; do
    git fetch origin "$BRANCH"
    git rebase "origin/$BRANCH" || {
      git rebase --abort || true
      echo "[results-push] FATAL: rebase onto origin/$BRANCH conflicted" >&2
      exit 5
    }
    PUSH_RC=0
    git push origin "HEAD:$BRANCH" > "$LOGS_DIR/issue-2587-results-push.log" 2>&1 || PUSH_RC=$?
    [ "$PUSH_RC" -eq 0 ] && break
    echo "[results-push] push attempt $attempt failed rc=$PUSH_RC" >&2
  done
  if [ "$PUSH_RC" -ne 0 ]; then
    echo "[results-push] FATAL: push failed after 2 attempts — see $LOGS_DIR/issue-2587-results-push.log" >&2
    exit 5
  fi
  git fetch origin "$BRANCH"
  UNPUSHED="$(git rev-list --count "origin/$BRANCH..HEAD")"
  if [ "$UNPUSHED" != "0" ]; then
    echo "[results-push] VERIFY FAILED: $UNPUSHED unpushed commit(s) remain" >&2
    exit 5
  fi
  for f in "${RESULT_JSONS[@]}"; do
    if [ -z "$(git ls-tree -r "origin/$BRANCH" --name-only -- "$f")" ]; then
      echo "[results-push] VERIFY FAILED: $f absent on origin/$BRANCH" >&2
      exit 5
    fi
  done
  echo "[results-push] verified: origin/$BRANCH carries all pod-side eval JSONs"

  # HF mirror of the pod-side eval JSONs (plan §10 pre-teardown harvest):
  # ONE upload_folder commit to issue2587_q35_map/manifests/.
  run_logged "$LOGS_DIR/issue-2587-hf-mirror.log" uv run python -c '
import shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, "src")
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi
files = sys.argv[1:]
with tempfile.TemporaryDirectory() as td:
    for f in files:
        p = Path(f)
        assert p.is_file(), f
        shutil.copy2(p, Path(td) / p.name)
    HfApi().upload_folder(
        folder_path=td,
        path_in_repo="issue2587_q35_map/manifests",
        repo_id="superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        commit_message="task #2587: mirror pod-side eval JSONs (pre-teardown harvest)",
    )
print("[hf-mirror] OK:", len(files), "files -> issue2587_q35_map/manifests/")' \
    "${RESULT_JSONS[@]}"

  # End-of-run epm:results sentinel (pod-side-reporting.md requirement 2;
  # drained by the VM poller from /workspace/logs/issue-2587-*.json).
  run_logged "$LOGS_DIR/issue-2587-sentinel-write.log" uv run python -c '
import os, sys, time
from explore_persona_space.atomic_io import write_json_atomic
logs_dir, note = sys.argv[1], sys.argv[2]
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "issue": 2587,
    "note": note,
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
path = os.path.join(logs_dir, "issue-2587-epm_results-%d.json" % int(time.time()))
write_json_atomic(path, payload)
print("[sentinel] wrote", path)' \
    "$LOGS_DIR" \
    "pod workload complete: P0b gates + P1 compat smoke (compat_smoke_done PASS) + P2/P3 map gen+capture (6 splits x 2 shards) + P4 fits/finalize + P5/P6 battery gen+capture+embed + P8 matched7b + leak/cap-hit harvest (eval_results/issue_2587/leak_caphit/); eval JSONs pushed to $BRANCH + mirrored to issue2587_q35_map/manifests/"
fi

phase done
exit 0
