#!/usr/bin/env bash
# Issue #923 CPU phase (cpu-mid) — Phase-2 stream-reduce → HF-poll join on the
# Phase-1 upload sentinel → Phase-3 fit battery → figures → upload (plan §4.3).
#
# Runs CONCURRENTLY with the GPU phase; joins on the UPLOAD_COMPLETE.json
# sentinel the GPU phase uploads LAST. On a join timeout (> EPS_923_JOIN_HOURS,
# default 3h per plan §4.3) it exits CLEANLY after the Phase-2 packs are
# uploaded — a follow-up cpu-mid dispatch (or VM run) reruns the fit stage.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue_923}"
mkdir -p "$LOG_DIR"
JOIN_HOURS="${EPS_923_JOIN_HOURS:-3}"
FITS_ONLY=0
if [ "${1:-}" = "--fits-only" ]; then
  FITS_ONLY=1  # resume path: reduce packs already on HF/local; skip Phase 2
  shift
fi

if [ "$FITS_ONLY" -eq 0 ]; then
  echo "[phase=reduce]"
  uv run python scripts/issue923_reduce_spans.py --genres betley,uc \
    2>&1 | tee "$LOG_DIR/reduce.log"
fi

echo "[phase=join]"
# Loud pre-flight OUTSIDE the poll loop: a missing .env / bad token / revoked
# repo access dies HERE (set -e) instead of burning the full join window as
# "sentinel not present" (r1 Minor: auth failures looked like not-yet).
# set -a: export .env so HF_TOKEN reaches the one-liner (heredoc-dotenv rule).
# ./.env exists on the VM/worktree only; GCE exports tokens via the startup
# script (crash att-20260703-163121: unconditional sourcing killed the poll).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
uv run python - <<'PY'
from huggingface_hub import list_repo_files

files = list_repo_files("superkaiba1/explore-persona-space-data", repo_type="dataset")
print(f"[join] preflight OK: listing returned {len(files)} files")
PY

DEADLINE=$(( $(date +%s) + JOIN_HOURS * 3600 ))
PACKS_DIR="$REPO_ROOT/data/issue_923/capture/packs"
mkdir -p "$PACKS_DIR"
JOINED=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  # rc contract: 0 = sentinel found; 3 = not-yet / transient HTTP (retry);
  # anything else = non-transient (auth/config) -> FAIL FAST, no window burn.
  rc=0
  (if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
  uv run python - <<'PY'
import sys

import requests
from huggingface_hub import list_repo_files
from huggingface_hub.utils import HfHubHTTPError

try:
    files = list_repo_files("superkaiba1/explore-persona-space-data", repo_type="dataset")
except HfHubHTTPError as e:
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code in (429, 500, 502, 503, 504):
        print(f"[join] transient HTTP {code}; will retry")
        sys.exit(3)
    raise  # 401/403/404 etc — non-transient, fail fast
except (requests.ConnectionError, requests.Timeout):
    print("[join] transient network error; will retry")
    sys.exit(3)
sentinel = "issue923_ctx_query_decomposition/analysis_tensors/capture/UPLOAD_COMPLETE.json"
sys.exit(0 if sentinel in files else 3)
PY
  ) || rc=$?
  if [ "$rc" -eq 0 ]; then
    JOINED=1
    break
  fi
  if [ "$rc" -ne 3 ]; then
    echo "[join] FATAL: sentinel poll exited rc=$rc (non-transient)"
    exit "$rc"
  fi
  echo "[join] Phase-1 upload sentinel not on HF yet; sleeping 120s"
  sleep 120
done

if [ "$JOINED" -ne 1 ]; then
  echo "[join] TIMEOUT after ${JOIN_HOURS}h — Phase-2 packs are uploaded; exiting cleanly."
  echo "[join] Re-run the fit stage later: bash scripts/issue923_cpu_phase.sh --fits-only"
  echo "[phase=done]"
  exit 0
fi

echo "[phase=fetch_capture_packs]"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
uv run python - <<'PY'
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

repo = "superkaiba1/explore-persona-space-data"
listing = list_repo_files(repo, repo_type="dataset")
for prefix, dest in (
    ("issue923_ctx_query_decomposition/analysis_tensors/capture/", Path("data/issue_923/capture/packs")),
    ("issue923_ctx_query_decomposition/analysis_tensors/reduce/", Path("data/issue_923/reduce")),
):
    dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in listing if f.startswith(prefix)]
    assert files, f"no packs under {prefix}"
    n_new = 0
    for f in files:
        target = dest / Path(f).name
        if target.exists():
            continue  # locally-produced pack wins (same instance ran Phase 2)
        local = hf_hub_download(repo, f, repo_type="dataset", local_dir="data/issue_923/hf_dl")
        target.write_bytes(Path(local).read_bytes())
        n_new += 1
    print(f"{prefix}: {len(files)} on hub, {n_new} fetched")
PY

echo "[phase=fits]"
EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cpu}" \
  uv run python scripts/issue923_fit_decomposition.py \
  2>&1 | tee "$LOG_DIR/fits.log"

echo "[phase=figures]"
# --upload: this instance is EPHEMERAL — figures must land on HF before exit
# (the VM fetches + commits them to git at Step 8; r1 Minor).
uv run python scripts/issue923_figures.py --upload 2>&1 | tee "$LOG_DIR/figures.log"

echo "[phase=done]"
