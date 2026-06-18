#!/usr/bin/env bash
# Thin wrapper around i610_dispatch.py for #632 that uploads the full
# workload log + per-cell logs + per-cell JSON artifacts to HF on EXIT,
# regardless of the workload's exit code. Lets us diagnose smoke-gate
# / eval crashes after the GCE EXIT trap powers off the VM.
#
# Usage (from a GCP --workload-cmd, with $WORKLOAD_ROOT already exported
# by the startup script):
#   REPO_ROOT="$WORKLOAD_ROOT" bash scripts/i632_dispatch_with_log_capture.sh --smoke --chassis assistant_proximal --n-gpus 4 --max-parallel 1
#
# Pass any flags through; they're forwarded verbatim to i610_dispatch.py.
set -uo pipefail   # NOT set -e — we want to run the upload even on failure
: "${REPO_ROOT:?REPO_ROOT must be set; export it before invoking}"
cd "$REPO_ROOT"

WORKLOAD_LOG="/workspace/issue632_workload.log"
echo "[i632-wrapper] starting at $(date -u +%FT%TZ); REPO_ROOT=$REPO_ROOT; logging to $WORKLOAD_LOG" | tee "$WORKLOAD_LOG"

# Run the dispatcher; tee everything to the workload log AND stdout (so the
# GCE startup-script log catches it too if Cloud Logging is wired).
uv run python scripts/i610_dispatch.py "$@" 2>&1 | tee -a "$WORKLOAD_LOG"
RC=${PIPESTATUS[0]}
echo "[i632-wrapper] dispatcher exited rc=$RC at $(date -u +%FT%TZ)" | tee -a "$WORKLOAD_LOG"

# Upload diagnostic artifacts to HF data repo regardless of RC.
# Path: superkaiba1/explore-persona-space-data/issue_632_debug/<attempt-id>/
ATTEMPT_ID="${ATTEMPT_ID:-att-$(date -u +%Y%m%d-%H%M%S)}"
echo "[i632-wrapper] uploading diagnostics to issue_632_debug/$ATTEMPT_ID/ ..." | tee -a "$WORKLOAD_LOG"

uv run python - "$WORKLOAD_LOG" "$ATTEMPT_ID" <<'PY' 2>&1 | tee -a "$WORKLOAD_LOG" || true
import glob, os, sys
from huggingface_hub import HfApi

workload_log, attempt_id = sys.argv[1], sys.argv[2]
api = HfApi()
repo = "superkaiba1/explore-persona-space-data"
remote_root = f"issue_632_debug/{attempt_id}"

paths: list[str] = [workload_log]
for pat in (
    "/workspace/eps-issue-632/eval_results/issue_632/**/logs/*.log",
    "/workspace/eps-issue-632/eval_results/issue_632/**/*.json",
    "/workspace/eps-issue-632/eval_results/issue_632/**/smoke_gate.json",
    "/workspace/eps-issue-632/eval_results/issue_632/**/done.json",
    "/workspace/eps-issue-632/eval_results/issue_632/**/.completion-sentinel.json",
):
    paths.extend(glob.glob(pat, recursive=True))

# Dedup + skip huge files (>50 MB).
seen = set()
uploaded = 0
failed = 0
for p in paths:
    if p in seen or not os.path.exists(p) or os.path.isdir(p):
        continue
    seen.add(p)
    sz = os.path.getsize(p)
    if sz > 50_000_000:
        print(f"  SKIP {p} ({sz / 1e6:.1f} MB)")
        continue
    # Strip the /workspace/eps-issue-632/ prefix for a clean remote layout.
    rel = p
    for prefix in ("/workspace/eps-issue-632/", "/workspace/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    remote_path = f"{remote_root}/{rel}"
    try:
        api.upload_file(
            path_or_fileobj=p, path_in_repo=remote_path,
            repo_id=repo, repo_type="dataset",
            commit_message=f"#632 debug: {os.path.basename(p)}",
        )
        print(f"  OK   {p} ({sz} bytes) -> {remote_path}")
        uploaded += 1
    except Exception as e:
        print(f"  FAIL {p}: {e}")
        failed += 1

print(f"[upload] uploaded={uploaded} failed={failed} skipped_large={len(paths) - uploaded - failed - 1}")
PY

echo "[i632-wrapper] done at $(date -u +%FT%TZ); rc=$RC" | tee -a "$WORKLOAD_LOG"
exit $RC
