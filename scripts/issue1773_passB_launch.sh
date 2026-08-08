#!/usr/bin/env bash
# Issue #1773 Pass B launcher — 4 chunk-sharded window-extraction workers,
# one per GPU, CVD-pinned in the LAUNCHER env (the #545 import-time-cuInit
# family; each worker also gets the matching --gpu-id). Pod-side contract:
# sentinel file + [phase=...] breadcrumbs ONLY — never scripts/task.py.
#
# Phases: pilot (1 chunk on worker 0, timed incl. upload — plan §7 gate 1,
# re-sizes, never kills) -> 4-wide full pass -> prefix-scoped upload verify ->
# sentinel + [phase=done].
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"
# Conditional .env sourcing (GCE lane has NO .env — metadata exports instead)
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

N_WORKERS="${EPM_1773_N_WORKERS:-4}"
LOG_DIR="${EPM_1773_LOG_DIR:-/workspace/logs}"
OUT_DIR="${EPM_1773_OUT_DIR:-$REPO_ROOT/data/issue_1773/raw_windows}"
SEL_DIR="${EPM_1773_SEL_DIR:-$REPO_ROOT/data/issue_1773/selection}"
SCRATCH="${EPM_1773_SCRATCH:-$REPO_ROOT/data/issue_1773/scratch}"
PLANNED_WALL_H="${EPM_1773_PLANNED_WALL_H:-2.5}"
mkdir -p "$LOG_DIR" "$OUT_DIR" "$SCRATCH"

echo "[phase=passB_import_check]"
uv run python scripts/issue1773_evidence_builder.py --import-check

# Stage the Pass-A selection from HF when absent (crash-fix r3: the GCE lane
# materializes only the git clone — data/ is gitignored, so the selection MUST
# be staged from issue1773_featurepipeline/selection/ before the pilot; the
# `[stage] selection staged:` line below is the fix-engaged signal). Idempotent:
# a locally-present inverted_index.npz makes this a no-op skip. Tee'd into the
# pilot log (created here; pilot appends) so the relaunch asserts one file.
echo "[phase=passB_stage_selection]"
uv run python scripts/issue1773_evidence_builder.py --pass stage-selection \
  --selection-dir "$SEL_DIR" 2>&1 | tee "$LOG_DIR/issue-1773-passB-pilot.log"
STAGE_RC=${PIPESTATUS[0]}
if [ "$STAGE_RC" -ne 0 ]; then
  echo "[phase=passB_stage_failed] rc=$STAGE_RC"
  exit "$STAGE_RC"
fi

echo "[phase=passB_pilot]"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1773_evidence_builder.py \
  --pass windows --pilot --worker 0 --n-workers "$N_WORKERS" --gpu-id 0 \
  --device cuda --selection-dir "$SEL_DIR" --out-dir "$OUT_DIR" \
  --scratch "$SCRATCH" 2>&1 | tee -a "$LOG_DIR/issue-1773-passB-pilot.log"
PILOT_RC=${PIPESTATUS[0]}
if [ "$PILOT_RC" -ne 0 ]; then
  echo "[phase=passB_pilot_failed] rc=$PILOT_RC"
  exit "$PILOT_RC"
fi
PROJ_H=$(uv run python -c "import json,sys; print(json.load(open('$OUT_DIR/pilot_report.json'))['projected_hours_per_worker'])")
echo "[phase=passB_pilot_done] projected_hours_per_worker=$PROJ_H planned=$PLANNED_WALL_H"
# Gate 1 is a re-sizing gate (plan §7): a >2x projection is SURFACED for the
# orchestrator (width re-eval / vectorize signature check), never a kill here.
uv run python - "$PROJ_H" "$PLANNED_WALL_H" <<'PY'
import sys
proj, planned = float(sys.argv[1]), float(sys.argv[2])
if proj > 2 * planned:
    print(f"[phase=passB_pilot_deviation] projected {proj:.2f}h > 2x planned {planned}h "
          "— orchestrator: width re-eval per plan gate 1", flush=True)
PY

echo "[phase=passB_fanout] width=$N_WORKERS"
# Workers are PLAIN backgrounded children (no setsid) so `wait` below is real —
# a setsid-detached shard reparents to pid 1 and `wait` returns instantly (the
# #1738 chained-waves trap). The LAUNCHER itself is what runs detached.
PIDS=()
for g in $(seq 0 $((N_WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES="$g" nohup uv run python \
    scripts/issue1773_evidence_builder.py --pass windows \
    --worker "$g" --n-workers "$N_WORKERS" --gpu-id "$g" --device cuda \
    --selection-dir "$SEL_DIR" --out-dir "$OUT_DIR" --scratch "$SCRATCH" \
    > "$LOG_DIR/issue-1773-passB-w$g.log" 2>&1 < /dev/null &
  PIDS+=($!)
  echo "[passB] worker $g pid=${PIDS[$g]} log=$LOG_DIR/issue-1773-passB-w$g.log"
done
echo "$$ ${PIDS[*]}" > "$LOG_DIR/issue-1773-passB.pid"

FAIL=0
for g in $(seq 0 $((N_WORKERS - 1))); do
  if ! wait "${PIDS[$g]}"; then
    RC=$?
    echo "[phase=passB_worker_failed] worker=$g rc=$RC"
    FAIL=1
  fi
done
if [ "$FAIL" -ne 0 ]; then
  echo "[phase=passB_failed] one or more workers failed; see per-worker logs"
  exit 1
fi

echo "[phase=passB_upload_verify]"
uv run python - "$OUT_DIR" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import issue1773_common as CM  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

out_dir = Path(sys.argv[1])
expected = sorted(
    f"{CM.HF_PREFIX}/raw_windows/{p.name}"
    for pat in ("windows_*.jsonl", "windows_*.done.json", "randdir_worker*.json")
    for p in out_dir.glob(pat)
)
missing = hub.verify_repo_paths_uploaded(
    HfApi(),
    CM.HF_DATA_REPO,
    expected,
    path_in_repo=f"{CM.HF_PREFIX}/raw_windows",
    repo_type="dataset",
)
if missing:
    raise SystemExit(
        f"[passB] upload verify FAILED: {len(missing)} missing, e.g. {sorted(missing)[:5]}"
    )
print(f"[passB] upload verify PASS: {len(expected)} files present", flush=True)
PY

SENTINEL="$LOG_DIR/issue-1773-passB-results.json"
uv run python - "$SENTINEL" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

sentinel, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
n_chunks = len(list(out_dir.glob("windows_*.done.json")))
body = {
    "sentinel_schema_version": 1,
    "kind": "epm:progress",
    "version": 1,
    "note": json.dumps(
        {
            "phase": "passB_windows",
            "n_chunk_files": n_chunks,
            "upload_verified": True,
            "out_dir": str(out_dir),
        }
    ),
}
tmp = sentinel.parent / f".tmp_{sentinel.name}"
tmp.write_text(json.dumps(body))
tmp.replace(sentinel)
print(f"[passB] sentinel written: {sentinel}", flush=True)
PY

echo "[phase=done]"
