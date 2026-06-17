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
# Arm: real (the parent run, cached on HF) | dummy (this amendment) | both.
# Default 'dummy' — the length-matched-dummy-query-control follow-up only needs
# the dummy arm; the real arm's 810 pair banks + 81 context-only banks are reused
# from HF (plan v5 §4).
ARM="dummy"
BATTERY=""
OUT_DIR=""
DEVICE="cuda"
for arg in "$@"; do
    case "$arg" in
        --issue=*) ISSUE="${arg#*=}" ;;
        --issue) ;; # value follows; handled by positional fallthrough below
        --phase=*) PHASE="${arg#*=}" ;;
        --arm=*) ARM="${arg#*=}" ;;
        --smoke) SMOKE=1 ;;
        --skip-upload) SKIP_UPLOAD=1 ;;
        --battery=*) BATTERY="${arg#*=}" ;;
        --out-dir=*) OUT_DIR="${arg#*=}" ;;
        --device=*) DEVICE="${arg#*=}" ;;
        *) ;;
    esac
done
# Support the space-separated `--issue 654 --phase extract --arm dummy` form.
prev=""
for arg in "$@"; do
    case "$prev" in
        --issue) ISSUE="$arg" ;;
        --phase) PHASE="$arg" ;;
        --arm) ARM="$arg" ;;
    esac
    prev="$arg"
done

case "$ARM" in
    real | dummy) ;;
    both)
        echo "[phase=failed] --arm both not supported in one invocation; run --arm real then --arm dummy" >&2
        exit 2
        ;;
    *)
        echo "[phase=failed] unknown --arm '$ARM' (real|dummy)" >&2
        exit 2
        ;;
esac

# Default battery/out-dir per arm (only if not explicitly overridden).
if [ -z "$BATTERY" ]; then
    [ "$ARM" = "dummy" ] && BATTERY="data/issue654/battery_dummy.json" || BATTERY="data/issue654/battery.json"
fi
if [ -z "$OUT_DIR" ]; then
    [ "$ARM" = "dummy" ] && OUT_DIR="data/issue654/dual_pos_dummy" || OUT_DIR="data/issue654/dual_pos"
fi

LOG_DIR="logs/issue_654"
mkdir -p "$LOG_DIR"
SENTINEL_DIR="/workspace/logs"
mkdir -p "$SENTINEL_DIR" 2>/dev/null || SENTINEL_DIR="$LOG_DIR"

HF_DATA_REPO="superkaiba1/explore-persona-space-data"
HF_PREFIX="issue654_query_displacement"
# Per-arm HF upload subdir under <prefix>/: real -> analysis_tensors;
# dummy -> analysis_tensors_dummy (plan v5 §4 — keeps the parent run untouched).
[ "$ARM" = "dummy" ] && HF_TENSORS_SUBDIR="analysis_tensors_dummy" || HF_TENSORS_SUBDIR="analysis_tensors"
# Real-battery path (the dummy arm reads its per-context length targets + context
# message set; build it on-pod if absent, same as the parent build).
REAL_BATTERY="data/issue654/battery.json"
# Local dir the dummy arm REUSES cached real-arm context-only banks from
# (fetched from HF below); the extractor's --reuse-context-only points here.
REUSE_CTX_ONLY="data/issue654/hf_context_only"

if [ "$SMOKE" -eq 1 ]; then
    if [ "$ARM" = "dummy" ]; then
        BATTERY="data/issue654/battery_dummy_smoke.json"
        OUT_DIR="data/issue654/dual_pos_dummy_smoke"
        REAL_BATTERY="data/issue654/battery_smoke.json"
    else
        BATTERY="data/issue654/battery_smoke.json"
        OUT_DIR="data/issue654/dual_pos_smoke"
    fi
fi

write_failure_sentinel() {
    local phase="$1"
    local reason="$2"
    local ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    local sentinel="$SENTINEL_DIR/issue-654-epm_failure-${ts}.json"
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
    # Durable failure diagnostics: upload $LOG_DIR (extract.log / build_battery.log
    # / upload.log) to the HF data repo BEFORE exit. The local sentinel under
    # /workspace/logs is destroyed by the GCE EXIT trap (gcloud instances delete)
    # before the orchestrator's poller can drain it, so the on-pod logs are the
    # only post-mortem artifact (#654 round-2 production crash). Best-effort: the
    # subshell `|| true` + the in-Python try/except keep a failed upload (HF
    # unreachable, token missing) from masking the real failure — we still exit 2.
    (uv run python - "$LOG_DIR" "$HF_DATA_REPO" "$HF_PREFIX" "$ts" "$phase" "$reason" <<'PY' 2>&1 || true
import sys
from pathlib import Path

from explore_persona_space.orchestrate.hub import _upload

log_dir, repo, prefix, ts, phase, reason = sys.argv[1:7]
log_path = Path(log_dir)
if not log_path.exists() or not any(log_path.iterdir()):
    print(f"[failure-log] no logs at {log_dir} to upload (best-effort skip)")
    raise SystemExit(0)
try:
    url = _upload(
        log_path,
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=f"{prefix}/run_logs/{ts}",
    )
    if url:
        print(f"[failure-log] uploaded {log_dir} -> {url}")
    else:
        print(f"[failure-log] _upload returned empty path (HF_TOKEN missing?) for {log_dir}", file=sys.stderr)
except Exception as e:  # noqa: BLE001 — best-effort diagnostics upload
    print(f"[failure-log] upload failed (best-effort): {e}", file=sys.stderr)
PY
    )
    echo "[phase=failed] FATAL at $phase: $reason" >&2
    exit 2
}

echo "[phase=start] === i654 dispatcher $(date -Iseconds) issue=$ISSUE phase=$PHASE smoke=$SMOKE ==="

if [ "$PHASE" != "extract" ]; then
    write_failure_sentinel "$PHASE" "unknown phase '$PHASE' (only 'extract' is supported on-pod)"
fi

echo "[phase=arm] === arm=$ARM (real=parent-cached-on-HF, dummy=this amendment) ==="

# ── Phase: build battery (CPU, on-pod, gated on existence) ───────────────────
# The auto/gcp lane git-clones the issue branch and runs the workload from there;
# it does NOT push VM-side files. data/issue654/ is gitignored (.gitignore `data/*`,
# not whitelisted), so the cloned tree has an EMPTY data/issue654/. The battery
# MUST therefore be built on-pod, CPU-side, BEFORE the GPU forward pass — the
# plan's "CPU, VM, pre-pod" framing (§3 step 1) means "the first CPU step of the
# dispatcher" on this lane. (#654 round-2 production crash: dispatcher consumed
# data/issue654/battery.json without building it.) The build is tokenizer-only
# (no model load) so it runs cleanly before the model-load phase.
BATTERY_BUILD_FLAG=""
[ "$SMOKE" -eq 1 ] && BATTERY_BUILD_FLAG="--smoke"

if [ "$ARM" = "dummy" ]; then
    # The dummy builder needs the REAL battery (per-context length targets + the
    # context message set). Build the real battery first (tokenizer-only) if absent.
    if [ ! -f "$REAL_BATTERY" ]; then
        echo "[phase=build-battery] === building real battery $REAL_BATTERY (length targets) ==="
        # shellcheck disable=SC2086
        uv run python scripts/issue654_build_battery.py \
            --out "$REAL_BATTERY" $BATTERY_BUILD_FLAG \
            2>&1 | tee "$LOG_DIR/build_battery.log" \
            || write_failure_sentinel build-battery "real build_battery rc=${PIPESTATUS[0]} (see build_battery.log)"
        [ -f "$REAL_BATTERY" ] || write_failure_sentinel build-battery "real battery not at $REAL_BATTERY"
    else
        echo "[phase=build-battery] === real battery already at $REAL_BATTERY (length targets) ==="
    fi
    if [ ! -f "$BATTERY" ]; then
        echo "[phase=build-battery] === building dummy battery $BATTERY (smoke=$SMOKE) ==="
        # shellcheck disable=SC2086
        uv run python scripts/issue654_build_battery_dummy.py \
            --real-battery "$REAL_BATTERY" --out "$BATTERY" $BATTERY_BUILD_FLAG \
            2>&1 | tee -a "$LOG_DIR/build_battery.log" \
            || write_failure_sentinel build-battery "dummy build_battery rc=${PIPESTATUS[0]} (see build_battery.log)"
        [ -f "$BATTERY" ] || write_failure_sentinel build-battery "dummy battery not at $BATTERY after build"
    else
        echo "[phase=build-battery] === dummy battery already at $BATTERY (skipping build) ==="
    fi
else
    if [ ! -f "$BATTERY" ]; then
        echo "[phase=build-battery] === building real battery $BATTERY (smoke=$SMOKE) ==="
        # shellcheck disable=SC2086
        uv run python scripts/issue654_build_battery.py \
            --out "$BATTERY" $BATTERY_BUILD_FLAG \
            2>&1 | tee "$LOG_DIR/build_battery.log" \
            || write_failure_sentinel build-battery "build_battery rc=${PIPESTATUS[0]} (see build_battery.log)"
        [ -f "$BATTERY" ] || write_failure_sentinel build-battery "battery not at $BATTERY after build"
    else
        echo "[phase=build-battery] === battery already exists at $BATTERY (skipping build) ==="
    fi
fi

# ── Phase: fetch cached context-only companion banks (dummy arm, REUSE not re-extract) ─
# The dummy arm reuses the parent run's 81 context-only banks (plan v5 §4): the
# companion's context-only side (no query) is identical for the dummy and real
# arms, so it is NOT re-extracted on GPU. Skip on smoke (the smoke re-extracts its
# 4 context-only banks fresh — tiny, and the cached HF banks are the full 81).
REUSE_FLAG=""
if [ "$ARM" = "dummy" ] && [ "$SMOKE" -ne 1 ]; then
    echo "[phase=fetch-context-only] === fetch cached context_only banks from HF -> $REUSE_CTX_ONLY ==="
    uv run python - "$HF_DATA_REPO" "$HF_PREFIX" "$REUSE_CTX_ONLY" <<'PY' \
        2>&1 | tee "$LOG_DIR/fetch_context_only.log" \
        || write_failure_sentinel fetch-context-only "fetch context_only rc=${PIPESTATUS[0]} (see fetch_context_only.log)"
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

repo, prefix, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
# Reuse the parent run's revision (plan v5 §4 / §10 pinned rev). main is fine here:
# the dummy arm only needs the parent's context-only banks, which are stable.
rev = "82d16a6faa7f8781163bf215154ed57296364780"
dest.mkdir(parents=True, exist_ok=True)
files = list_repo_files(repo, repo_type="dataset", revision=rev)
needle = f"{prefix}/analysis_tensors/context_only/"
ctx_files = [f for f in files if f.startswith(needle) and f.endswith(".pt")]
if not ctx_files:
    raise SystemExit(f"no cached context_only banks under {needle} at rev {rev}")
for f in ctx_files:
    local = hf_hub_download(repo, f, repo_type="dataset", revision=rev)
    name = f.rsplit("/", 1)[-1]
    # Copy into the flat reuse dir the extractor's --reuse-context-only expects
    # (<context_id>.pt at the top level).
    out = dest / name
    out.write_bytes(Path(local).read_bytes())
print(f"fetched {len(ctx_files)} cached context_only banks -> {dest}")
PY
    REUSE_FLAG="--reuse-context-only $REUSE_CTX_ONLY"
fi

# ── Phase: dual-position extraction (GPU) ───────────────────────────────────
echo "[phase=extract] === dual-position extraction (arm=$ARM, battery=$BATTERY, out=$OUT_DIR) ==="
SMOKE_FLAG=""
[ "$SMOKE" -eq 1 ] && SMOKE_FLAG="--smoke"
# shellcheck disable=SC2086
uv run python scripts/issue654_extract.py \
    --battery "$BATTERY" --out-dir "$OUT_DIR" --device "$DEVICE" $SMOKE_FLAG $REUSE_FLAG \
    2>&1 | tee "$LOG_DIR/extract.log" || write_failure_sentinel extract "extract rc=${PIPESTATUS[0]} (see extract.log)"

MANIFEST="$OUT_DIR/extraction_manifest.json"
[ -f "$MANIFEST" ] || write_failure_sentinel extract "no extraction_manifest.json at $MANIFEST"

# ── Upload the per-pair .pt banks + manifest to the HF data repo ────────────
if [ "$SKIP_UPLOAD" -eq 1 ] || [ "$SMOKE" -eq 1 ]; then
    echo "[phase=upload] === upload SKIPPED (smoke/skip-upload) ==="
else
    echo "[phase=upload] === upload .pt banks + manifest -> $HF_DATA_REPO/$HF_PREFIX/$HF_TENSORS_SUBDIR ==="
    uv run python - "$OUT_DIR" "$HF_DATA_REPO" "$HF_PREFIX" "$HF_TENSORS_SUBDIR" <<'PY' \
        2>&1 | tee "$LOG_DIR/upload.log" || write_failure_sentinel upload "upload failed (see upload.log)"
import sys
from pathlib import Path
from explore_persona_space.orchestrate.hub import _upload, DEFAULT_DATASET_REPO  # noqa: F401

out_dir, repo, prefix, subdir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
# Folder upload (the whole dual_pos dir, incl. context_only/ + manifest) under
# <prefix>/<subdir>/ — analysis_tensors (real) or analysis_tensors_dummy (dummy),
# keeping the parent run's banks untouched (plan v5 §4).
url = _upload(
    out_dir,
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/{subdir}",
)
if not url:
    raise SystemExit("hub._upload returned empty path (HF_TOKEN missing / upload failed)")
print(f"uploaded {out_dir} -> {url}")
# Verify the manifest landed on a FRESH listing.
from huggingface_hub import list_repo_files  # noqa: E402
files = list_repo_files(repo, repo_type="dataset", revision="main")
needle = f"{prefix}/{subdir}/extraction_manifest.json"
assert needle in files, f"manifest {needle} not found on a fresh HF listing"
print(f"verified {needle} on HF")
PY

    # Upload the battery input itself (plan §10):
    #   real  -> issue654_query_displacement/inputs/battery.json
    #   dummy -> issue654_query_displacement/inputs/battery_dummy.json
    BATTERY_INPUT_NAME="battery.json"
    [ "$ARM" = "dummy" ] && BATTERY_INPUT_NAME="battery_dummy.json"
    echo "[phase=upload] === upload $BATTERY_INPUT_NAME -> $HF_DATA_REPO/$HF_PREFIX/inputs ==="
    uv run python - "$BATTERY" "$HF_DATA_REPO" "$HF_PREFIX" "$BATTERY_INPUT_NAME" <<'PY' \
        2>&1 | tee -a "$LOG_DIR/upload.log" || write_failure_sentinel upload "battery upload failed (see upload.log)"
import sys
from pathlib import Path
from explore_persona_space.orchestrate.hub import _upload  # noqa: F401

battery, repo, prefix, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
# upload_as_file=True: single-file upload — _upload raises ValueError otherwise
# (upload_folder silently no-ops on a file path; hub.py guard, #595/#640).
url = _upload(
    battery,
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/inputs/{name}",
    upload_as_file=True,
)
if not url:
    raise SystemExit("hub._upload returned empty path for battery (HF_TOKEN missing / upload failed)")
print(f"uploaded {battery} -> {url}")
from huggingface_hub import list_repo_files  # noqa: E402
files = list_repo_files(repo, repo_type="dataset", revision="main")
needle = f"{prefix}/inputs/{name}"
assert needle in files, f"battery {needle} not found on a fresh HF listing"
print(f"verified {needle} on HF")
PY
fi

# ── End-of-run sentinel for poll_pipeline.py (required keys) ─────────────────
SENTINEL="$SENTINEL_DIR/issue-654-epm_results-$(date +%s).json"
uv run python - "$SENTINEL" "$MANIFEST" "$HF_DATA_REPO" "$HF_PREFIX" "$SMOKE" "$ARM" "$HF_TENSORS_SUBDIR" <<'PY'
import json, sys, datetime
sentinel, manifest_path, repo, prefix, smoke, arm, subdir = sys.argv[1:8]
with open(manifest_path) as f:
    manifest = json.load(f)
note = {
    "issue": 654,
    "phase": "extract",
    "arm": arm,
    "followup_label": "length-matched-dummy-query-control",
    "smoke": smoke == "1",
    "n_pairs_extracted": manifest.get("n_pairs_extracted"),
    "offset_fail_fraction": manifest.get("offset_fail_fraction"),
    "offset_kill_tripped": manifest.get("offset_kill_tripped"),
    "reuse_context_only": manifest.get("reuse_context_only"),
    "num_hidden_layers": manifest.get("num_hidden_layers"),
    "hidden_size": manifest.get("hidden_size"),
    "hf_analysis_tensors": f"{repo}/{prefix}/{subdir}" if smoke != "1" else None,
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
