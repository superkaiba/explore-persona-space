#!/usr/bin/env bash
# Issue #654 dispatcher: dual-position residual-stream extraction (GPU phase only).
#
# Plan §3 / §8. The ONLY GPU phase is the dual-position extraction
# (scripts/issue654_extract.py). The CPU metric/figure phase
# (scripts/issue654_analyze.py) runs OFF-pod on the VM after termination
# (plan §8 CPU-phases-off-pod) — this script does NOT invoke it.
#
# Phases (on-pod):
#   extract : run issue654_extract.py over the battery, upload the per-pair .pt
#             banks + manifest to the HF data repo, write the poll_pipeline
#             sentinel, emit [phase=done].
#
# The auto/gcp router runs this driver via:
#   --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue654_dispatch.sh \
#                   --issue 654 --phase extract'
# (REPO_ROOT defaults to $WORKLOAD_ROOT on the gcp/auto lane; see the gcp driver
# contract — a driver defaulting REPO_ROOT to the RunPod path dies on GCP.)
#
# Pod-side code NEVER shells out to scripts/task.py (CLAUDE.md). The epm:results
# marker is posted from the VM by the orchestrator after the sentinel is drained.
# CVD pin N/A — single GPU, no parallel per-GPU fan-out.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-/workspace/explore-persona-space}}"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# WandB project so the auto-router's --workload-cmd default of issue<N> doesn't apply.
export WANDB_PROJECT="issue654_query_displacement"

# Load credentials at entry so every `uv run python` subprocess inherits them
# (uv run does NOT auto-load .env).
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

ISSUE=654
PHASE="extract"
SMOKE=0
SKIP_UPLOAD=0
BATTERY="data/issue654/battery.json"
OUT_DIR="data/issue654/dual_pos"
DEVICE="cuda"
for arg in "$@"; do
    case "$arg" in
        --issue=*) ISSUE="${arg#*=}" ;;
        --issue) ;; # value follows; handled by positional fallthrough below
        --phase=*) PHASE="${arg#*=}" ;;
        --smoke) SMOKE=1 ;;
        --skip-upload) SKIP_UPLOAD=1 ;;
        --battery=*) BATTERY="${arg#*=}" ;;
        --out-dir=*) OUT_DIR="${arg#*=}" ;;
        --device=*) DEVICE="${arg#*=}" ;;
        *) ;;
    esac
done
# Support the space-separated `--issue 654 --phase extract` form the router uses.
prev=""
for arg in "$@"; do
    case "$prev" in
        --issue) ISSUE="$arg" ;;
        --phase) PHASE="$arg" ;;
    esac
    prev="$arg"
done

LOG_DIR="logs/issue_654"
mkdir -p "$LOG_DIR"
SENTINEL_DIR="/workspace/logs"
mkdir -p "$SENTINEL_DIR" 2>/dev/null || SENTINEL_DIR="$LOG_DIR"

HF_DATA_REPO="superkaiba1/explore-persona-space-data"
HF_PREFIX="issue654_query_displacement"

if [ "$SMOKE" -eq 1 ]; then
    BATTERY="data/issue654/battery_smoke.json"
    OUT_DIR="data/issue654/dual_pos_smoke"
fi

write_failure_sentinel() {
    local phase="$1"
    local reason="$2"
    local sentinel="$SENTINEL_DIR/issue-654-epm_failure-$(date +%s).json"
    uv run python - "$sentinel" "$phase" "$reason" <<'PY'
import json, sys, datetime
path, phase, reason = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "sentinel_schema_version": 1, "kind": "epm:failure", "version": 1,
    "issue": 654, "phase": phase, "failure_class": "code", "reason": reason,
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote failure sentinel: {path}")
PY
    echo "[phase=failed] FATAL at $phase: $reason" >&2
    exit 2
}

echo "[phase=start] === i654 dispatcher $(date -Iseconds) issue=$ISSUE phase=$PHASE smoke=$SMOKE ==="

if [ "$PHASE" != "extract" ]; then
    write_failure_sentinel "$PHASE" "unknown phase '$PHASE' (only 'extract' is supported on-pod)"
fi

# ── Phase: dual-position extraction (GPU) ───────────────────────────────────
echo "[phase=extract] === dual-position extraction (battery=$BATTERY, out=$OUT_DIR) ==="
SMOKE_FLAG=""
[ "$SMOKE" -eq 1 ] && SMOKE_FLAG="--smoke"
# shellcheck disable=SC2086
uv run python scripts/issue654_extract.py \
    --battery "$BATTERY" --out-dir "$OUT_DIR" --device "$DEVICE" $SMOKE_FLAG \
    2>&1 | tee "$LOG_DIR/extract.log" || write_failure_sentinel extract "extract rc=${PIPESTATUS[0]} (see extract.log)"

MANIFEST="$OUT_DIR/extraction_manifest.json"
[ -f "$MANIFEST" ] || write_failure_sentinel extract "no extraction_manifest.json at $MANIFEST"

# ── Upload the per-pair .pt banks + manifest to the HF data repo ────────────
if [ "$SKIP_UPLOAD" -eq 1 ] || [ "$SMOKE" -eq 1 ]; then
    echo "[phase=upload] === upload SKIPPED (smoke/skip-upload) ==="
else
    echo "[phase=upload] === upload .pt banks + manifest -> $HF_DATA_REPO/$HF_PREFIX/analysis_tensors ==="
    uv run python - "$OUT_DIR" "$HF_DATA_REPO" "$HF_PREFIX" <<'PY' \
        2>&1 | tee "$LOG_DIR/upload.log" || write_failure_sentinel upload "upload failed (see upload.log)"
import sys
from pathlib import Path
from explore_persona_space.orchestrate.hub import _upload, DEFAULT_DATASET_REPO  # noqa: F401

out_dir, repo, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
# Folder upload (the whole dual_pos dir, incl. context_only/ + manifest) under
# issue654_query_displacement/analysis_tensors/.
url = _upload(
    out_dir,
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/analysis_tensors",
)
if not url:
    raise SystemExit("hub._upload returned empty path (HF_TOKEN missing / upload failed)")
print(f"uploaded {out_dir} -> {url}")
# Verify the manifest landed on a FRESH listing.
from huggingface_hub import list_repo_files  # noqa: E402
files = list_repo_files(repo, repo_type="dataset", revision="main")
needle = f"{prefix}/analysis_tensors/extraction_manifest.json"
assert needle in files, f"manifest {needle} not found on a fresh HF listing"
print(f"verified {needle} on HF")
PY
fi

# ── End-of-run sentinel for poll_pipeline.py (required keys) ─────────────────
SENTINEL="$SENTINEL_DIR/issue-654-epm_results-$(date +%s).json"
uv run python - "$SENTINEL" "$MANIFEST" "$HF_DATA_REPO" "$HF_PREFIX" "$SMOKE" <<'PY'
import json, sys, datetime
sentinel, manifest_path, repo, prefix, smoke = sys.argv[1:6]
with open(manifest_path) as f:
    manifest = json.load(f)
note = {
    "issue": 654,
    "phase": "extract",
    "smoke": smoke == "1",
    "n_pairs_extracted": manifest.get("n_pairs_extracted"),
    "offset_fail_fraction": manifest.get("offset_fail_fraction"),
    "offset_kill_tripped": manifest.get("offset_kill_tripped"),
    "num_hidden_layers": manifest.get("num_hidden_layers"),
    "hidden_size": manifest.get("hidden_size"),
    "hf_analysis_tensors": f"{repo}/{prefix}/analysis_tensors" if smoke != "1" else None,
    "manifest_path": manifest_path,
}
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 654,
    "gate": False,
    "blocks_pipeline": False,
    "by": "issue654_dispatch",
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": note,
}
with open(sentinel, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote results sentinel: {sentinel}")
PY

echo "[phase=done] === i654 dispatcher complete $(date -Iseconds) smoke=$SMOKE ==="
