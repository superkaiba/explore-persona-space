#!/usr/bin/env bash
# Issue #1773 FULL-DICTIONARY (131,072-feature) compute chain — phases 0/1 only.
#
# Runs the four prerequisite phases in order, each resumable and each gated on
# its own done-sentinel so a relaunch fast-forwards completed phases:
#
#   p0_stage    stage the #1482 pooled store from the HF data repo (idempotent)
#   p0_phase0   mechanical axes over all 131,072 features (GPU-blocked GEMMs)
#   p1_passA    reservoir selection over the 1,920 pooled shards (CPU)
#   p1_passB    window extraction, one worker per GPU, CVD-pinned (GPU)
#   p1_passC    per-feature evidence packets (CPU, big-RAM)
#
# The API phases (2 describe / 3 axes) are DELIBERATELY NOT here: they need no
# GPU, so the pod is released before they start (CLAUDE.md § CPU-only phases
# don't hold GPU pods; the #664 idle-burn class at multi-day scale).
#
# Pod-side contract: sentinel files + [phase=...] breadcrumbs ONLY — this
# script never shells out to scripts/task.py.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"
# Conditional .env sourcing (GCE lane has NO .env — metadata exports instead)
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${EPM_1773_LOG_DIR:-/workspace/logs}"
WORK="${EPM_1773_WORK:-/workspace/issue1773_fulldict}"
STORE="${EPM_1773_STORE:-$WORK/sae_pooled}"
SEL_DIR="${EPM_1773_SEL_DIR:-$WORK/selection}"
WIN_DIR="${EPM_1773_WIN_DIR:-$WORK/raw_windows}"
EV_DIR="${EPM_1773_EV_DIR:-$WORK/evidence}"
SCRATCH="${EPM_1773_SCRATCH:-$WORK/scratch}"
OUT_ROOT="${EPM_1773_OUT_ROOT:-$REPO_ROOT/eval_results/issue_1773_fulldict}"
DONE_DIR="$WORK/done"
mkdir -p "$LOG_DIR" "$WORK" "$SEL_DIR" "$WIN_DIR" "$EV_DIR" "$SCRATCH" "$OUT_ROOT" "$DONE_DIR"

N_GPUS="${EPM_1773_N_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
[ "$N_GPUS" -ge 1 ] || { echo "[fulldict] FATAL: 0 GPUs visible"; exit 2; }
echo "[fulldict] repo=$REPO_ROOT work=$WORK gpus=$N_GPUS commit=$(git rev-parse HEAD)"

phase_done() { [ -f "$DONE_DIR/$1.done" ]; }
mark_done()  { date -u +%Y-%m-%dT%H:%M:%SZ > "$DONE_DIR/$1.done"; }

# ── p0_stage: pooled store (9.2 GB) from the HF data repo ────────────────────
if phase_done p0_stage; then
  echo "[phase=p0_stage] SKIP (done)"
else
  echo "[phase=p0_stage]"
  uv run python - "$STORE" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import issue1773_common as CM  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

dest = Path(sys.argv[1])
prefix = "issue1482_error_analysis/analysis_tensors/sae_pooled"
if len(list(dest.glob("pooled_*.npz"))) == CM.N_SHARDS:
    print(f"[stage] store already complete: {CM.N_SHARDS} shards at {dest}", flush=True)
else:
    hub.stage_hub_prefix(CM.HF_DATA_REPO, prefix, dest.parent, repo_type="dataset")
    nested = dest.parent / prefix
    if nested.is_dir() and not list(dest.glob("pooled_*.npz")):
        # stage_hub_prefix mirrors the repo-relative prefix verbatim (#1774)
        print(f"[stage] resolving verbatim mirror {nested} -> {dest}", flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        for p in nested.glob("pooled_*.npz"):
            p.replace(dest / p.name)
n = len(list(dest.glob("pooled_*.npz")))
print(f"[stage] pooled store staged: {n} shards -> {dest}", flush=True)
assert n == CM.N_SHARDS, f"expected {CM.N_SHARDS} shards, staged {n}"
PY
  mark_done p0_stage
fi

# ── p0_rb: stage the #779 r_B directions (gitignored -> absent from the clone) ─
# `data/**` is gitignored, so a fresh clone has no `data/issue_779/r_b/*.pt`
# and phase 0's preflight fails loud on the realized-keys check. They are
# ~400 KB each and were snapshotted to an ISSUE-OWNED HF prefix (#600 pattern:
# never consume another task's shared mirror) with sha256 pins asserted here.
if phase_done p0_rb; then
  echo "[phase=p0_rb] SKIP (done)"
else
  echo "[phase=p0_rb]"
  uv run python - "$REPO_ROOT" <<'PY'
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import issue1773_common as CM  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

EXPECTED_SHA256 = {
    "evil.pt": "65b70c63076b9452c6d1c8a66ee1ed3d403503df936ea1af6fffc353d135aff1",
    "hallucination.pt": "d643269c9904b99e14968c84c8e3a02cd45d5ed4674621edbeb78950467ccd6d",
    "sycophancy.pt": "af6d679b59ad02e9e00a26e73ff77c00dda69cb8e2fabd22ea3a3ee28bbdad3d",
}
dest = Path(sys.argv[1]) / "data" / "issue_779" / "r_b"
dest.mkdir(parents=True, exist_ok=True)
prefix = f"{CM.HF_PREFIX}/inputs/issue_779_r_b"
for name, want in EXPECTED_SHA256.items():
    target = dest / name
    if not target.exists():
        hub.stage_hub_file(CM.HF_DATA_REPO, f"{prefix}/{name}", target, repo_type="dataset")
    got = hashlib.sha256(target.read_bytes()).hexdigest()
    assert got == want, f"r_B sha mismatch for {name}: {got} != {want}"
    print(f"[stage] r_B staged + sha-verified: {target}", flush=True)
PY
  mark_done p0_rb
fi

# ── p0_phase0: mechanical axes over the full dictionary (GPU) ────────────────
if phase_done p0_phase0; then
  echo "[phase=p0_phase0] SKIP (done)"
else
  echo "[phase=p0_phase0]"
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1773_phase0_mechanical.py \
    --full-dictionary --device cuda \
    --store "$STORE" --work "$WORK/phase0_work" --out-root "$OUT_ROOT" \
    2>&1 | tee "$LOG_DIR/issue-1773-fd-phase0.log"
  RC=${PIPESTATUS[0]}
  [ "$RC" -eq 0 ] || { echo "[phase=p0_phase0_failed] rc=$RC"; exit "$RC"; }
  mark_done p0_phase0
fi

# ── p1_passA: reservoir selection over 1,920 shards (CPU) ────────────────────
if phase_done p1_passA; then
  echo "[phase=p1_passA] SKIP (done)"
else
  echo "[phase=p1_passA]"
  uv run python scripts/issue1773_evidence_builder.py --pass select \
    --full-dictionary --store "$STORE" --selection-dir "$SEL_DIR" \
    --scratch "$SCRATCH" \
    2>&1 | tee "$LOG_DIR/issue-1773-fd-passA.log"
  RC=${PIPESTATUS[0]}
  [ "$RC" -eq 0 ] || { echo "[phase=p1_passA_failed] rc=$RC"; exit "$RC"; }
  mark_done p1_passA
fi

# ── p1_passB: window extraction, one worker per GPU (GPU) ────────────────────
# CVD pinned in the LAUNCHER env per worker (the #545 import-time-cuInit
# family), with the matching --gpu-id. Workers are PLAIN backgrounded children
# so `wait` is real — a setsid-detached shard reparents to pid 1 and `wait`
# returns instantly (the #1738 chained-waves trap).
if phase_done p1_passB; then
  echo "[phase=p1_passB] SKIP (done)"
else
  echo "[phase=p1_passB] width=$N_GPUS"
  PIDS=()
  for g in $(seq 0 $((N_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES="$g" nohup uv run python \
      scripts/issue1773_evidence_builder.py --pass windows \
      --worker "$g" --n-workers "$N_GPUS" --gpu-id "$g" --device cuda \
      --selection-dir "$SEL_DIR" --out-dir "$WIN_DIR" --scratch "$SCRATCH" \
      > "$LOG_DIR/issue-1773-fd-passB-w$g.log" 2>&1 < /dev/null &
    PIDS+=($!)
    echo "[passB] worker $g pid=${PIDS[$g]} log=$LOG_DIR/issue-1773-fd-passB-w$g.log"
  done
  echo "$$ ${PIDS[*]}" > "$LOG_DIR/issue-1773-fd-passB.pid"
  FAIL=0
  for g in $(seq 0 $((N_GPUS - 1))); do
    if ! wait "${PIDS[$g]}"; then echo "[phase=p1_passB_worker_failed] worker=$g"; FAIL=1; fi
  done
  [ "$FAIL" -eq 0 ] || { echo "[phase=p1_passB_failed] see per-worker logs"; exit 1; }
  mark_done p1_passB
fi

# ── p1_passC: per-feature evidence packets (CPU, big RAM) ────────────────────
if phase_done p1_passC; then
  echo "[phase=p1_passC] SKIP (done)"
else
  echo "[phase=p1_passC]"
  uv run python scripts/issue1773_evidence_builder.py --pass assemble \
    --full-dictionary --selection-dir "$SEL_DIR" --out-dir "$WIN_DIR" \
    --evidence-dir "$EV_DIR" --phase0-dir "$OUT_ROOT/phase0" --scratch "$SCRATCH" \
    2>&1 | tee "$LOG_DIR/issue-1773-fd-passC.log"
  RC=${PIPESTATUS[0]}
  [ "$RC" -eq 0 ] || { echo "[phase=p1_passC_failed] rc=$RC"; exit "$RC"; }
  mark_done p1_passC
fi

# ── sentinel: the orchestrator's poller drains this into epm:progress ────────
SENTINEL="$LOG_DIR/issue-1773-fd-results.json"
uv run python - "$SENTINEL" "$OUT_ROOT" "$EV_DIR" "$SEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

sentinel, out_root, ev_dir, sel_dir = (Path(a) for a in sys.argv[1:5])
meta_p = out_root / "phase0" / "phase0_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
rep_p = ev_dir / "completeness_report.json"
rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
sel_meta_p = sel_dir / "selection_meta.json"
sel_meta = json.loads(sel_meta_p.read_text()) if sel_meta_p.exists() else {}
body = {
    "sentinel_schema_version": 1,
    "kind": "epm:progress",
    "version": 1,
    "note": json.dumps(
        {
            "phase": "fulldict_phases_0_1_complete",
            "n_features": meta.get("n_features"),
            "n_features_active_in_fit": meta.get("n_features_active_in_fit"),
            "n_features_dead_in_fit": meta.get("n_features_dead_in_fit"),
            "wiring_gate_max_delta": meta.get("wiring_gate_max_delta"),
            "passA_union_rows": sel_meta.get("union_rows"),
            "passA_act_short": sel_meta.get("n_act_short"),
            "evidence_fill": rep.get("fill_fraction"),
            "evidence_n_short": rep.get("n_short"),
            "next": "release GPU, then phases 2-3 (--grouped) off-pod",
        }
    ),
}
tmp = sentinel.parent / f".tmp_{sentinel.name}"
tmp.write_text(json.dumps(body))
tmp.replace(sentinel)
print(f"[fulldict] sentinel written: {sentinel}", flush=True)
PY

echo "[phase=done]"
