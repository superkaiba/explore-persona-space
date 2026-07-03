#!/bin/bash
# issue833_gcp_phase_d.sh — Phase-D chain fits on a dedicated cpu-bigmem GCE
# instance (#833 round 7).
#
# Why off-VM: the shared 125G VM's fleet load put every chain phase
# (13-15 GB RSS each) into an earlyoom kill-loop — 5 cells SIGTERM'd, 0/9
# chains completed across rounds 3-5 (epm:compute-deviation v3/v4). This
# wrapper stages the tensor store from HF (ONE snapshot_download — the 9-way
# per-file join storm that tripped the 2500req/5min Hub quota in round 3 is
# not reproduced), fans out the ALREADY-REVIEWED fit entrypoint (round-5
# batched engine, code-review ensemble PASS) across 3 lanes, and uploads
# cells/ + chain_rho/ back to the HF data repo for the VM-side assembly pass.
#
# GCE contract: the startup script clones the repo to $WORKLOAD_ROOT, exports
# REPO_ROOT="$WORKLOAD_ROOT", threads HF_TOKEN, and tails logs/* to the
# serial console.
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT" || exit 1
# Round 7c: the GCE bootstrap exports HF_XET_HIGH_PERFORMANCE=1, and the xet
# path does a per-file token/connection refresh — at 6 threads x 8688 files
# the refresh endpoint gave out mid-stage (round 7b died in
# refresh_xet_connection_info after ~4.5k files). Force the plain HTTP/CDN
# path for this workload (the #515 HF_XET_DISABLE workaround); the final
# uploads are small JSONs on the non-LFS path, unaffected.
export HF_XET_DISABLE=1
mkdir -p logs eval_results/issue_833
MAIN_LOG="logs/issue833_gcp_phase_d.log"

# NOTE round 7b: snapshot_download is a TRAP against this data repo — it
# enumerates the ENTIRE ~1M-file tree before allow_patterns filters (the
# round-7a instance sat 40+ min in enumeration with zero files landed).
# Scoped list_repo_tree(path_in_repo=...) enumerates ONLY the two prefixes
# (~seconds), then a modest thread pool of hf_hub_download calls stays under
# the 2500 req/5min quota (round 3's kill was 9 concurrent PROCESSES).
echo "[phase=stage] scoped list + per-file download ($(date -u +%H:%M:%S))" | tee -a "$MAIN_LOG"
uv run python - <<'PY' || { echo "[phase=stage] FAILED" | tee -a "$MAIN_LOG"; exit 1; }
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download

REPO = "superkaiba1/explore-persona-space-data"
PREFIXES = [
    "issue833_onpolicy_map/analysis_tensors",
    "issue833_onpolicy_map/analysis_tensors_rbase",
]
api = HfApi()
paths = []
for pref in PREFIXES:
    paths += [
        e.path
        for e in api.list_repo_tree(REPO, path_in_repo=pref, repo_type="dataset", recursive=True)
        if e.path.endswith(".npz")
    ]
print(f"listed {len(paths)} npz files", flush=True)


def fetch(p: str, attempts: int = 5) -> str | None:
    """Return None on success, the path on hard failure (never raises)."""
    for attempt in range(attempts):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir="hf_stage")
            return None
        except Exception:  # noqa: BLE001 — retry w/ backoff; report, don't kill the pool
            if attempt == attempts - 1:
                return p
            time.sleep(30 * (attempt + 1))
    return p


done, failed = 0, []
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = [ex.submit(fetch, p) for p in paths]
    for f in as_completed(futs):
        bad = f.result()
        if bad:
            failed.append(bad)
        else:
            done += 1
        if (done + len(failed)) % 500 == 0:
            print(f"downloaded {done}/{len(paths)} (failed so far: {len(failed)})", flush=True)
print(f"pool pass: {done}/{len(paths)} ok, {len(failed)} failed", flush=True)
# Serial second pass over the stragglers (rate pressure gone by now).
still = []
for p in failed:
    time.sleep(2)
    if fetch(p, attempts=3):
        still.append(p)
    else:
        done += 1
print(f"downloaded {done}/{len(paths)}; unrecovered: {len(still)}", flush=True)
for p in still[:20]:
    print("  FAILED:", p, flush=True)
sys.exit(0 if done == len(paths) else 1)
PY
rm -rf eval_results/issue_833/analysis_tensors eval_results/issue_833/analysis_tensors_rbase
cp -a hf_stage/issue833_onpolicy_map/analysis_tensors eval_results/issue_833/ || exit 1
cp -a hf_stage/issue833_onpolicy_map/analysis_tensors_rbase eval_results/issue_833/ || exit 1
N1=$(find eval_results/issue_833/analysis_tensors -name '*.npz' | wc -l)
N2=$(find eval_results/issue_833/analysis_tensors_rbase -name '*.npz' | wc -l)
echo "[phase=stage] npz staged: $N1 + $N2 (expect 4320 + 4368)" | tee -a "$MAIN_LOG"
[ "$N1" -eq 4320 ] && [ "$N2" -eq 4368 ] || { echo "[phase=stage] INCOMPLETE STORE" | tee -a "$MAIN_LOG"; exit 1; }

# Same per-cell invocation as the VM lane runner (round-5 flags verbatim);
# 5 BLAS threads x 3 lanes on 16 vCPU, ~15G peak RSS per chain on 128G.
run_lane() {
  local LANE=$1; shift
  for cell in "$@"; do
    local beh=${cell%%:*} li=${cell##*:}
    local LOG="logs/issue833_phase_d_${beh}_L${li}.log"
    echo "=== lane $LANE cell $beh L$li start $(date -u +%H:%M:%S) ===" >> "$LOG"
    env OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 \
      uv run python scripts/issue833_fit_onpolicy.py \
        --behaviors "$beh" --layers "$li" \
        --legs-mode reextracted --floors-impl batched --joined-cache --local-store \
        --mlp-chunk-size 128 --mlp-num-threads 5 \
        >> "$LOG" 2>&1 < /dev/null
    local rc=$?
    echo "=== lane $LANE cell $beh L$li rc=$rc $(date -u +%H:%M:%S) ===" >> "$LOG"
    [ "$rc" -ne 0 ] && echo "$beh:$li rc=$rc" >> "logs/issue833_phase_d_lane_${LANE}.failures"
  done
  echo "lane $LANE done $(date -u +%H:%M:%S)" >> "logs/issue833_phase_d_lane_${LANE}.done"
}

echo "[phase=fit] 3 lanes x 3 cells ($(date -u +%H:%M:%S))" | tee -a "$MAIN_LOG"
# Round 7d: lane cell lists overridable via env (space-separated beh:layer
# tokens; defaults reproduce the round-7c grid verbatim) so a tail instance
# can run a re-ordered / scoped grid; per-cell cached-skip in the fit script
# makes any overlap resume-safe.
run_lane G1 ${LANE_CELLS_G1:-em:7 em:14 em:21} &
run_lane G2 ${LANE_CELLS_G2:-sycophancy:7 sycophancy:14 sycophancy:21} &
run_lane G3 ${LANE_CELLS_G3:-fact:7 fact:14 fact:21} &
wait

CHAINS=$(ls eval_results/issue_833/chain_rho/*.json 2>/dev/null | wc -l)
FAILS=$(cat logs/issue833_phase_d_lane_G*.failures 2>/dev/null | tr '\n' ' ')
echo "[phase=upload] chains=$CHAINS/9 fails='$FAILS' ($(date -u +%H:%M:%S))" | tee -a "$MAIN_LOG"
uv run python - <<'PY' || { echo "[phase=upload] FAILED" | tee -a "$MAIN_LOG"; exit 1; }
from pathlib import Path

from huggingface_hub import HfApi

api = HfApi()
for sub in ("cells", "chain_rho"):
    if Path(f"eval_results/issue_833/{sub}").is_dir():
        api.upload_folder(
            folder_path=f"eval_results/issue_833/{sub}",
            path_in_repo=f"issue833_onpolicy_map/phase_d_outputs/{sub}",
            repo_id="superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            commit_message=f"issue-833 GCP Phase-D {sub} (round 7)",
        )
api.upload_folder(
    folder_path="logs",
    path_in_repo="issue833_onpolicy_map/phase_d_outputs/logs",
    repo_id="superkaiba1/explore-persona-space-data",
    repo_type="dataset",
    allow_patterns=["issue833_phase_d_*", "issue833_gcp_phase_d.log"],
    commit_message="issue-833 GCP Phase-D logs (round 7)",
)
print("upload done")
PY
echo "[phase=done] chains=$CHAINS/9 fails='$FAILS' ($(date -u +%H:%M:%S))" | tee -a "$MAIN_LOG"
# Non-zero exit when the grid is incomplete so the EXIT-trap crash-persist
# fires and the poller surfaces it (partial uploads above already landed).
[ "$CHAINS" -eq 9 ] || exit 2
exit 0
