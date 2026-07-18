#!/usr/bin/env bash
# Issue #1489 pod/GCE dispatcher (plan §4 DAG; sentinel signaling ONLY — no
# pod-side task.py). phase_a = provision A (smoke -> P1 gen -> P2 capture ->
# store upload -> P3 distill -> P4 dose probes -> upload); phase_b =
# provision B (P4b FT eval gen+capture -> upload).
#
# --smoke runs the SAME production chain at tiny N (PASS_UNIFIED): P0 --smoke
# writes only cell_plain + 4 canary cells and every later phase enumerates
# cells FROM that manifest. Smoke GENERATED artifacts land under smoke roots;
# the corpus (a read-only parent input) is staged ONCE at a non-rebinding
# canonical path shared by smoke + production.
#
# Usage:
#   bash scripts/issue1489_dispatch.sh phase_a [--smoke-only]
#   bash scripts/issue1489_dispatch.sh phase_b <selection.json path or HF auto>
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

PHASE="${1:?usage: issue1489_dispatch.sh phase_a|phase_b [...]}"
shift || true
SMOKE_ONLY=0
if [ "${1:-}" = "--smoke-only" ]; then SMOKE_ONLY=1; shift; fi

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="$REPO_ROOT/data/issue_1489/logs"
mkdir -p "$LOG_DIR"
echo $$ > "$LOG_DIR/issue-1489.pid"

CORPUS_DIR="$REPO_ROOT/data/issue_1489/hf_dl/corpus"       # non-rebinding parent input
OUT="$REPO_ROOT/data/issue_1489"
SMOKE_OUT="$REPO_ROOT/data/issue_1489_smoke"
COND="$OUT/conditions"
SMOKE_COND="$SMOKE_OUT/conditions"

sentinel() {  # sentinel <phase> <note>
  uv run python - "$1" "$2" <<'PY'
import datetime, json, os, sys
phase, note = sys.argv[1], sys.argv[2]
d = "/workspace/logs"
try:
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, ".probe"), "w").close()
except OSError:
    d = os.path.join(os.environ.get("REPO_ROOT", "."), "data/issue_1489/logs")
    os.makedirs(d, exist_ok=True)
payload = {
    "issue": 1489,
    "phase": phase,
    "note": note,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}
with open(os.path.join(d, "issue-1489-results.json"), "w") as f:
    json.dump(payload, f, indent=2)
print(f"[sentinel] {phase}: {note}", flush=True)
PY
}

run_smoke_chain() {
  echo "[phase=smoke] tiny-real chain (canary cells) start"
  uv run python scripts/issue1489_build_conditions.py --smoke --stage \
    --corpus-dir "$CORPUS_DIR" --out "$SMOKE_COND"
  uv run python scripts/issue1489_gpu_phase.py --phase gen --smoke \
    --conditions-dir "$SMOKE_COND" --corpus-dir "$CORPUS_DIR" --out "$SMOKE_OUT" \
    --enforce-eager --no-prefix-caching
  uv run python scripts/issue1489_gpu_phase.py --phase capture --smoke \
    --conditions-dir "$SMOKE_COND" --corpus-dir "$CORPUS_DIR" --out "$SMOKE_OUT"
  # one pca48 fit at 20 draws on the smoke captures (G2 demoted to informational)
  uv run python scripts/issue1489_fit_grid.py --smoke \
    --summaries-dir "$SMOKE_OUT/summaries" --out "$SMOKE_OUT/p6" \
    --units "transfer:context_end:pca48:L14"
  # tiny distill run (fact_veg, 30 rows, 2 checkpoints) + probes + sync judge
  uv run python scripts/issue1489_gpu_phase.py --phase distill --smoke \
    --conditions-dir "$SMOKE_COND" --corpus-dir "$CORPUS_DIR" --out "$SMOKE_OUT" \
    --skip-upload
  uv run python scripts/issue1489_gpu_phase.py --phase dose_probes --smoke \
    --conditions-dir "$SMOKE_COND" --corpus-dir "$CORPUS_DIR" --out "$SMOKE_OUT"
  uv run python scripts/issue1489_judge.py --batch manipulation \
    --conditions-dir "$SMOKE_COND" --corpus-dir "$CORPUS_DIR" --out "$SMOKE_OUT" \
    --judge-out "$SMOKE_OUT/judge" --cache-dir "$SMOKE_OUT/judge_cache"
  # schema asserts: the parent engine loads a #1489-produced shard (assumption 2)
  uv run python - "$SMOKE_OUT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from issue1092_fit_grid import _load_summary
smoke_out = Path(sys.argv[1])
summaries = smoke_out / "summaries"
cells = sorted(p.name for p in summaries.iterdir() if p.is_dir())
assert cells, f"no smoke capture cells under {summaries}"
for cell in cells:
    arr, paths = _load_summary(summaries, cell, "t1", 14)
    assert arr.ndim == 2 and arr.shape[0] > 0, (cell, arr.shape)
ckpts = sorted((smoke_out / "distill").glob("*/checkpoint-*"))
assert len(ckpts) >= 2, f"smoke distill produced {len(ckpts)} checkpoints (<2)"
probes = sorted((smoke_out / "raw_completions" / "dose_probes").glob("*/ckpt*_completions.json"))
assert len(probes) >= 2, f"smoke dose probes produced {len(probes)} files (<2)"
judge = smoke_out / "judge" / "manipulation_check.json"
assert judge.exists(), f"smoke judge output missing: {judge}"
print(f"SMOKE-SCHEMA PASS: cells={cells} ckpts={len(ckpts)} probes={len(probes)}")
PY
  sentinel smoke_done "tiny-real chain + schema asserts PASS"
  echo "[phase=smoke] done"
}

phase_a() {
  uv run python -m explore_persona_space.orchestrate.preflight || {
    echo "[phase_a] preflight FAILED"; sentinel failed "preflight"; exit 1; }
  sentinel phase_a_start "provision A"
  run_smoke_chain
  if [ "$SMOKE_ONLY" = "1" ]; then
    sentinel done "smoke-only run complete"
    echo "[phase=done] smoke-only"
    return 0
  fi
  echo "[phase=p0] production conditions build"
  uv run python scripts/issue1489_build_conditions.py --stage \
    --corpus-dir "$CORPUS_DIR" --out "$COND" --upload
  echo "[phase=p1] generation"
  uv run python scripts/issue1489_gpu_phase.py --phase gen \
    --conditions-dir "$COND" --corpus-dir "$CORPUS_DIR" --out "$OUT" \
    --enforce-eager --no-prefix-caching
  sentinel p1_done "generation complete"
  echo "[phase=p2] capture"
  uv run python scripts/issue1489_gpu_phase.py --phase capture \
    --conditions-dir "$COND" --corpus-dir "$CORPUS_DIR" --out "$OUT"
  sentinel p2_done "capture complete"
  echo "[phase=upload-a1] store upload BEFORE the long distill phase (#825 ordering)"
  uv run python scripts/issue1489_gpu_phase.py --phase upload \
    --conditions-dir "$COND" --corpus-dir "$CORPUS_DIR" --out "$OUT"
  sentinel upload_a1_done "raw completions + summaries uploaded + verified"
  uv run python scripts/clean_experiment_downloads.py 1489 --incremental --apply || true
  echo "[phase=p3] distillation"
  uv run python scripts/issue1489_gpu_phase.py --phase distill \
    --conditions-dir "$COND" --corpus-dir "$CORPUS_DIR" --out "$OUT"
  sentinel p3_done "distillation + checkpoint ladder upload complete"
  echo "[phase=p4] dose probes"
  uv run python scripts/issue1489_gpu_phase.py --phase dose_probes \
    --conditions-dir "$COND" --corpus-dir "$CORPUS_DIR" --out "$OUT"
  echo "[phase=upload-a2] probe + distill-round artifacts"
  uv run python scripts/issue1489_gpu_phase.py --phase upload \
    --conditions-dir "$COND" --corpus-dir "$CORPUS_DIR" --out "$OUT"
  sentinel done "provision A complete (P1+P2+P3+P4 uploaded)"
  echo "[phase=done] phase_a"
}

phase_b() {
  SELECTION="${1:?phase_b needs the selection.json path}"
  test -f "$SELECTION" || { echo "selection missing: $SELECTION"; sentinel failed "selection-missing"; exit 1; }
  uv run python -m explore_persona_space.orchestrate.preflight || {
    echo "[phase_b] preflight FAILED"; sentinel failed "preflight"; exit 1; }
  sentinel phase_b_start "provision B"
  # conditions must exist (fresh instance: re-stage + rebuild deterministically)
  if [ ! -f "$COND/manifest.jsonl" ]; then
    echo "[phase_b] conditions missing on fresh instance; rebuilding (seeded, deterministic)"
    uv run python scripts/issue1489_build_conditions.py --stage \
      --corpus-dir "$CORPUS_DIR" --out "$COND"
  fi
  echo "[phase=p4b] FT eval gen + capture"
  uv run python scripts/issue1489_gpu_phase.py --phase ft \
    --conditions-dir "$COND" --corpus-dir "$CORPUS_DIR" --out "$OUT" \
    --selection "$SELECTION" --enforce-eager --no-prefix-caching
  echo "[phase=upload-b] FT cells upload"
  uv run python scripts/issue1489_gpu_phase.py --phase upload \
    --conditions-dir "$COND" --corpus-dir "$CORPUS_DIR" --out "$OUT"
  sentinel done "provision B complete (FT cells uploaded)"
  echo "[phase=done] phase_b"
}

case "$PHASE" in
  phase_a) phase_a ;;
  phase_b) phase_b "$@" ;;
  *) echo "unknown phase: $PHASE"; exit 2 ;;
esac
