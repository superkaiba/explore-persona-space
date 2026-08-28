#!/usr/bin/env bash
# Task #2587 follow-up round `ffr-9b-fire-gated` pod workload (plan v6 §4 DAG).
#
# Thin single-GPU fork of scripts/issue2587_pod_workload.sh (plan §4
# "Embedded-shell exit-path note"): keeps its bootstrap (pin derivation +
# model-venv build + driver gate), run_logged/sentinel helpers, and the
# require-gate enforcement shape; DROPS the map-fit (P2-P4), matched7b (P8),
# and embed phases — both fitted maps are BANKED and reused (plan §2
# divergence 2), and the FFR round has no embed leg (divergence 4).
#
# Phase order (each phase = single python process, explicit exit terminals):
#   bootstrap        git fetch issue-2564 objects, §4.1 model-venv build,
#                    driver-version gate (cuda-compat remediation + verify)
#   f1_smoke         template_pin -> hook_probe (model venv, the parent's own
#                    gate legs) -> tiny FFR cell (stance, 3 carriers, K=2)
#                    gen+capture through the PRODUCTION --bank-source ffr
#                    path (the FFR bank build inside this leg runs the token/
#                    render gates at FULL 276-context grain) -> compose_f1
#                    (realized-pin equality + flashinfer ABSENT in the model
#                    interpreter + run_meta PASS records + tiny-cell
#                    manifests; writes compat_smoke_report_ffr.json ALWAYS,
#                    <out-root>/compat_smoke_done.json ONLY on all-PASS)
#   f2_ffr_gen       2,760 rollouts (276 contexts x K=10), single GPU,
#                    per-cell anchors upload to issue2587_minpair/ffr_9b/
#   f3_ffr_capture   teacher-forced all-layer fp32 capture -> va/vc upload
#   results_push     commit + push the compat report, HF-mirror the battery
#                    manifests to issue2587_minpair/ffr_9b/manifests/,
#                    epm:results sentinel, single terminal [phase=done]
#
# F1 enforcement (the parent's require_p1 shape, re-keyed to this round):
# every production wave entry (f2/f3) calls require_f1, which verifies the
# FULL sentinel contract — schema issue2587_ffr_compat_smoke_v1, issue/
# phase/status, the report's recorded sha256, and CODE IDENTITY
# (battery_code_sha256 == the driver bytes about to run) — never the bare
# status token.
#
# CVD discipline: single GPU — the launcher pins CUDA_VISIBLE_DEVICES=0 in
# the env for every model step (gotchas.md import-time-cuInit rule); env
# pins + venv paths are DERIVED from issue2587_common at runtime.
#
# Dry-run (CPU-testable control flow): EPM_I2587_DRYRUN=1 echoes every
# command (with its log redirect) instead of executing. Tests/VM smokes set
# EPM_I2587_OUT_ROOT/EPM_I2587_LOGS_DIR to tmp dirs.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${EPM_I2587_OUT_ROOT:-/workspace/eps2587_ffr}"
LOGS_DIR="${EPM_I2587_LOGS_DIR:-/workspace/logs}"
HF_PREFIX="${EPM_I2587_FFR_HF_PREFIX:-issue2587_minpair/ffr_9b}"
BRANCH="${EPM_I2587_BRANCH:-issue-2587}"
DRYRUN="${EPM_I2587_DRYRUN:-}"

BATTERY_ROOT="$OUT_ROOT/battery"
# Per-leg out-root for the F1 tiny cell (#1333: never share a resume-keyed
# root between a smoke leg and the production leg).
F1_ROOT="$OUT_ROOT/f1_battery"
RUN_META="$OUT_ROOT/run_meta.json"
COMPAT_SENTINEL="$OUT_ROOT/compat_smoke_done.json"
COMPAT_REPORT="$REPO_ROOT/eval_results/issue_2587/ffr-9b-fire-gated/compat_smoke_report_ffr.json"
SPLIT_IDS="$REPO_ROOT/eval_results/issue_2587/split_ids.json"

MAP="scripts/issue2587_map_gen_capture.py"
BATTERY="scripts/issue2587_battery_run.py"

RESULT_JSONS=(
  eval_results/issue_2587/ffr-9b-fire-gated/compat_smoke_report_ffr.json
)

phase() { printf '[phase=%s]\n' "$1"; }

run_logged() {
  # run_logged <log-file> <cmd...> — foreground; stdout+stderr to the log;
  # on failure echo the log tail and exit with the rc.
  local log="$1" rc=0
  shift
  if [ -n "$DRYRUN" ]; then
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

assert_file() {
  [ -n "$DRYRUN" ] && { printf '[dryrun] assert_file %s (%s)\n' "$1" "$2"; return 0; }
  if [ ! -f "$1" ]; then
    echo "[workload] FATAL: $2 did not produce $1" >&2
    exit 3
  fi
}

write_sentinel() {
  # write_sentinel <path> <phase-name> — plan-§9 phase_outputs sentinel.
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

require_f1() {
  # require_f1 <next-phase> — the parent require_p1 shape re-keyed: verify
  # schema/issue/phase/status, the report's recorded sha256, and code
  # identity (battery_code_sha256 == the driver bytes about to run).
  if [ -n "$DRYRUN" ]; then
    printf '[dryrun] require_f1 before %s\n' "$1"
    return 0
  fi
  uv run python -c 'import hashlib, json, sys
sent_path, report_path, battery_path, nxt = sys.argv[1:5]

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

try:
    d = json.load(open(sent_path, encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit("[f1-gate] REFUSED %s: compat sentinel %s absent" % (nxt, sent_path))
for key, want in (
    ("schema", "issue2587_ffr_compat_smoke_v1"),
    ("issue", 2587),
    ("phase", "F1"),
    ("status", "PASS"),
):
    got = d.get(key)
    assert got == want, "[f1-gate] REFUSED %s: %s %s=%r != %r" % (nxt, sent_path, key, got, want)
try:
    report_sha = _sha(report_path)
except FileNotFoundError:
    raise SystemExit("[f1-gate] REFUSED %s: compat report %s absent" % (nxt, report_path))
assert report_sha == d.get("report_sha256"), (
    "[f1-gate] REFUSED %s: report %s sha256 %s != sentinel-recorded %r (report rewritten "
    "since compose_f1)" % (nxt, report_path, report_sha, d.get("report_sha256")))
code_sha = _sha(battery_path)
assert code_sha == d.get("battery_code_sha256"), (
    "[f1-gate] REFUSED %s: %s sha256 %s != sentinel-recorded %r (driver code changed since "
    "compose_f1 — re-run F1)" % (nxt, battery_path, code_sha, d.get("battery_code_sha256")))
print("[f1-gate] OK before %s: %s (schema+report+code identity verified)" % (nxt, sent_path))' \
    "$COMPAT_SENTINEL" "$COMPAT_REPORT" "$BATTERY" "$1"
}

# ── bootstrap ───────────────────────────────────────────────────────────────
phase bootstrap
mkdir -p "$OUT_ROOT" "$LOGS_DIR"

# §4.1 pins + venv paths DERIVED from issue2587_common (never retyped).
# Real even under dry-run so the echoed commands carry the true pins.
PIN_LINES="$(uv run python -c '
import json, sys
sys.path.insert(0, "scripts")
import issue2587_common as cm
print(" ".join("%s=%s" % (k, v) for k, v in sorted(cm.LAUNCH_ENV_PINS.items())))
print(cm.model_python())
print(cm.MODEL_DRIVER_FLOOR_MAJOR)
print(cm.CUDA_COMPAT_DIR)
print(json.dumps(cm.MODEL_VENV_PINS, sort_keys=True))
')"
ENV_PINS_LINE="$(printf '%s\n' "$PIN_LINES" | sed -n 1p)"
MODEL_PY="$(printf '%s\n' "$PIN_LINES" | sed -n 2p)"
DRIVER_FLOOR_MAJOR="$(printf '%s\n' "$PIN_LINES" | sed -n 3p)"
COMPAT_DIR="$(printf '%s\n' "$PIN_LINES" | sed -n 4p)"
VENV_PINS_JSON="$(printf '%s\n' "$PIN_LINES" | sed -n 5p)"
read -r -a ENV_PINS <<< "$ENV_PINS_LINE"
echo "[workload] launch env pins: ${ENV_PINS[*]}"
echo "[workload] model interpreter: $MODEL_PY (driver floor major $DRIVER_FLOOR_MAJOR)"

# Pinned-blob imports (bank2587 pins bank2564 at the parent branch blob).
run_logged "$LOGS_DIR/issue-2587-ffr-git-fetch.log" git fetch origin issue-2564

# §4.1 model-venv build (idempotent).
run_logged "$LOGS_DIR/issue-2587-ffr-venv-build.log" uv run python -c '
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import issue2587_common as cm
cm.build_model_venv(Path(sys.argv[1]))
print("[venv-build] OK:", cm.MODEL_VENV_DEFAULT)' "$LOGS_DIR"

# Driver-version gate (the #2330 cuda-compat remediation recipe).
if [ -z "$DRYRUN" ]; then
  DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)"
  if [ -z "$DRIVER_MAJOR" ]; then
    echo "[workload] FATAL: nvidia-smi returned no driver version" >&2
    exit 3
  fi
  if [ "$DRIVER_MAJOR" -lt "$DRIVER_FLOOR_MAJOR" ]; then
    echo "[workload] driver major $DRIVER_MAJOR < $DRIVER_FLOOR_MAJOR — installing cuda-compat-13-0"
    run_logged "$LOGS_DIR/issue-2587-ffr-cuda-compat.log" apt-get install -y cuda-compat-13-0
    export LD_LIBRARY_PATH="$COMPAT_DIR:${LD_LIBRARY_PATH:-}"
  fi
else
  printf '[dryrun] driver_gate (nvidia-smi major vs %s; cuda-compat remediation)\n' "$DRIVER_FLOOR_MAJOR"
fi
run_logged "$LOGS_DIR/issue-2587-ffr-driver-gate.log" uv run python -c '
import sys
sys.path.insert(0, "scripts")
import issue2587_common as cm
cm.assert_driver_compat()
print("[driver-gate] OK")'

# ── F1: compat smoke gate (plan §7 check set; fresh pod has no sentinel) ────
phase f1_smoke
if [ -z "$DRYRUN" ] && require_f1 f1_resume_probe; then
  echo "[workload] F1 sentinel verified (report+code identity) — skipping F1 re-run (resume)"
else
if [ -z "$DRYRUN" ]; then
  echo "[workload] F1 sentinel absent/invalid — fresh F1 run (wiping smoke scratch)"
  rm -rf "$F1_ROOT" "$COMPAT_SENTINEL" "$COMPAT_REPORT"
fi
# (a) template_pin: closed-empty-think suffix token assert (3 probes).
run_logged "$LOGS_DIR/issue-2587-ffr-gate-template-pin.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$MAP" --gate template_pin \
  --out-dir "$OUT_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$OUT_ROOT/split_ids_done.json" --split-ids "$SPLIT_IDS" -v
# (b) hook_probe: hook-vs-tuple parity rel <= 1e-5 at blocks {16,22,30} +
#     32-block resolve + v_C position assert (model load on the production
#     device class).
run_logged "$LOGS_DIR/issue-2587-ffr-gate-hook-probe.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$MAP" --gate hook_probe --h-dim 4096 \
  --out-dir "$OUT_ROOT" --run-meta-out "$RUN_META" \
  --sentinel-path "$OUT_ROOT/split_ids_done.json" --split-ids "$SPLIT_IDS" -v
# (c) tiny FFR end-to-end cell (stance, 3 carriers, K=2) through the
#     PRODUCTION --bank-source ffr path: the bank build runs the manifest
#     sha pin + string gates + token/render gates at FULL 276-context grain;
#     gen runs the think-leak scan (0/6 rows hard); capture runs gate-4.
#     --upload none: local stores only (production f2/f3 exercise hf).
run_logged "$LOGS_DIR/issue-2587-ffr-f1-battery-gen.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase gen --bank-source ffr --axes stance \
  --max-carriers 3 --draws 2 --out-root "$F1_ROOT" --upload none
run_logged "$LOGS_DIR/issue-2587-ffr-f1-battery-capture.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase capture --bank-source ffr --axes stance \
  --max-carriers 3 --draws 2 --out-root "$F1_ROOT" --upload none
# (d) compose_f1 in the MODEL interpreter (stdlib-only payload): realized
#     pins == MODEL_VENV_PINS + flashinfer ABSENT + run_meta PASS records +
#     tiny-cell gen manifest presence; writes the report ALWAYS, the
#     sentinel only on all-PASS.
if [ -n "$DRYRUN" ]; then
  printf '[dryrun] compose_f1 -> %s + %s\n' "$COMPAT_REPORT" "$COMPAT_SENTINEL"
else
run_logged "$LOGS_DIR/issue-2587-ffr-f1-compose.log" \
  env "${ENV_PINS[@]}" PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" -c '
import glob, hashlib, importlib.metadata, importlib.util, json, os, sys, time

run_meta_path, f1_root, report_path, sentinel_path, battery_path, pins_json = sys.argv[1:7]
pins = json.loads(pins_json)  # cm.MODEL_VENV_PINS, launcher-derived (never retyped)
checks = {}

realized = {}
for name, want in sorted(pins.items()):
    got = importlib.metadata.version(name)
    realized[name] = got
    assert got == want, "[compose-f1] pin drift: %s %s != %s" % (name, got, want)
checks["realized_pins"] = {"passed": True, "realized": realized, "pinned": pins}

assert importlib.util.find_spec("flashinfer") is None, "[compose-f1] flashinfer PRESENT (banned)"
checks["flashinfer_absent"] = {"passed": True}

meta = json.load(open(run_meta_path, encoding="utf-8"))
for gate in ("template_pin", "hook_probe"):
    rec = meta.get(gate)
    assert rec and rec.get("passed") is True, "[compose-f1] run_meta %s not PASS: %r" % (gate, rec)
    checks[gate] = {"passed": True}

gen_manifests = sorted(
    glob.glob(os.path.join(f1_root, "**", "anchors_*.done.json"), recursive=True)
)
assert gen_manifests, "[compose-f1] tiny FFR cell left no anchors_*.done.json under %s" % f1_root
checks["tiny_ffr_cell"] = {"passed": True, "gen_done_manifests": gen_manifests}

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

report = {
    "schema": "issue2587_ffr_compat_smoke_report_v1",
    "issue": 2587,
    "round": "ffr-9b-fire-gated",
    "phase": "F1",
    "status": "PASS",
    "checks": checks,
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
os.makedirs(os.path.dirname(report_path), exist_ok=True)
tmp = report_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, sort_keys=True)
os.replace(tmp, report_path)

sentinel = {
    "schema": "issue2587_ffr_compat_smoke_v1",
    "issue": 2587,
    "phase": "F1",
    "status": "PASS",
    "report_sha256": _sha(report_path),
    "battery_code_sha256": _sha(battery_path),
    "ts_utc": report["ts_utc"],
}
tmp = sentinel_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(sentinel, fh, indent=2, sort_keys=True)
os.replace(tmp, sentinel_path)
print("[compose-f1] PASS -> %s + %s" % (report_path, sentinel_path))' \
  "$RUN_META" "$F1_ROOT" "$COMPAT_REPORT" "$COMPAT_SENTINEL" "$BATTERY" "$VENV_PINS_JSON"
fi
assert_file "$COMPAT_SENTINEL" "F1 compose (compat_smoke_done)"
assert_file "$COMPAT_REPORT" "F1 compose (compat_smoke_report_ffr)"
fi

# ── F2: FFR generation (2,760 rollouts; single GPU, single process) ─────────
require_f1 f2_ffr_gen
phase f2_ffr_gen
run_logged "$LOGS_DIR/issue-2587-ffr-f2-gen.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase gen --bank-source ffr \
  --out-root "$BATTERY_ROOT" --upload hf --hf-prefix "$HF_PREFIX"
write_sentinel "$OUT_ROOT/ffr_gen_done.json" f2_ffr_gen

# ── F3: FFR capture (all-layer fp32; single GPU) ────────────────────────────
require_f1 f3_ffr_capture
phase f3_ffr_capture
run_logged "$LOGS_DIR/issue-2587-ffr-f3-capture.log" \
  env "${ENV_PINS[@]}" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT/src" \
  "$MODEL_PY" "$BATTERY" --phase capture --bank-source ffr \
  --out-root "$BATTERY_ROOT" --upload hf --hf-prefix "$HF_PREFIX"
write_sentinel "$OUT_ROOT/ffr_capture_done.json" f3_ffr_capture

# ── results push + HF manifests mirror + epm:results sentinel ───────────────
phase results_push
for f in "${RESULT_JSONS[@]}"; do
  assert_file "$REPO_ROOT/$f" "results_push input"
done
if [ -n "$DRYRUN" ]; then
  printf '[dryrun] results_push: commit+push %s; verify rev-list==0 + per-file ls-tree on origin/%s\n' \
    "${RESULT_JSONS[*]}" "$BRANCH"
  printf '[dryrun] hf_mirror: %s + %s/{,shard*/}manifests/*.json -> %s/manifests/\n' \
    "${RESULT_JSONS[*]}" "$BATTERY_ROOT" "$HF_PREFIX"
  printf '[dryrun] epm_results sentinel -> %s\n' "$LOGS_DIR"
else
  # Result-push verification contract (#1205, pod-side-reporting.md).
  git add -- "${RESULT_JSONS[@]}"
  if git diff --cached --quiet -- "${RESULT_JSONS[@]}"; then
    echo "[results-push] no new eval-JSON changes to commit"
  else
    git commit -m "task #2587: ffr-9b-fire-gated pod-side compat report (F1 gate)" \
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
    git push origin "HEAD:$BRANCH" > "$LOGS_DIR/issue-2587-ffr-results-push.log" 2>&1 || PUSH_RC=$?
    [ "$PUSH_RC" -eq 0 ] && break
    echo "[results-push] push attempt $attempt failed rc=$PUSH_RC" >&2
  done
  if [ "$PUSH_RC" -ne 0 ]; then
    echo "[results-push] FATAL: push failed after 2 attempts — see $LOGS_DIR/issue-2587-ffr-results-push.log" >&2
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
  echo "[results-push] verified: origin/$BRANCH carries the FFR compat report"

  # HF mirror (plan §10 pre-teardown harvest): compat report + battery
  # done/cap-hit manifests -> issue2587_minpair/ffr_9b/manifests/ in ONE
  # upload_folder commit.
  run_logged "$LOGS_DIR/issue-2587-ffr-hf-mirror.log" uv run python -c '
import glob, os, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, "src")
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi
battery_root, hf_prefix = sys.argv[1], sys.argv[2]
files = list(sys.argv[3:])
files += sorted(glob.glob(os.path.join(battery_root, "manifests", "*.json")))
files += sorted(glob.glob(os.path.join(battery_root, "shard*", "manifests", "*.json")))
assert files, "no manifest files to mirror"
with tempfile.TemporaryDirectory() as td:
    for f in files:
        p = Path(f)
        assert p.is_file(), f
        shutil.copy2(p, Path(td) / p.name)
    HfApi().upload_folder(
        folder_path=td,
        path_in_repo=f"{hf_prefix}/manifests",
        repo_id="superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        commit_message="task #2587: ffr-9b-fire-gated manifests mirror (pre-teardown harvest)",
    )
print("[hf-mirror] OK:", len(files), "files ->", f"{hf_prefix}/manifests/")' \
    "$BATTERY_ROOT" "$HF_PREFIX" "${RESULT_JSONS[@]/#/$REPO_ROOT/}"

  # End-of-run epm:results sentinel (pod-side-reporting.md requirement 2).
  run_logged "$LOGS_DIR/issue-2587-ffr-sentinel-write.log" uv run python -c '
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
    "ffr-9b-fire-gated pod workload complete: F1 compat gate (compat_smoke_done PASS) + F2 FFR gen (2,760 rollouts, anchors uploaded per cell) + F3 FFR capture (va/vc uploaded) under $HF_PREFIX; compat report pushed to $BRANCH + manifests mirrored to $HF_PREFIX/manifests/; VM-side next: judge --ffr (528 Batch calls) then ffr analysis"
fi

phase done
exit 0
