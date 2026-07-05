#!/usr/bin/env bash
# Issue #923 mixed-forward-span-stitch CPU phase (cpu-mid; plan v9 §4.2/§9).
#
# STRICTLY SEQUENTIAL after the GPU phase (which uploads + TERMINATES first —
# the orchestrator reads the uploaded identity_check_mix.json k1 verdict
# before dispatching this phase): fetch the mixed feature packs (uploaded by
# the GPU phase; k1 re-verified here belt-and-suspenders) + the reused
# pool_fctx packs at the PINNED pooled revision + the parent target/reduce
# packs at the PINNED parent revision -> the RESTRICTED 2-arm fit battery
# (--feature-source pool --arms arm_qry_mix,arm_concat_mix; nulls + the four
# paired reads vs the persisted fits_pooled sums, which live in GIT on this
# branch) -> mixed figures -> upload.
#
# Usage (dispatched via dispatch_issue.py --intent cpu-mid --boot-disk-gb 100
# --workload-cmd "bash scripts/issue923_mixed_cpu_phase.sh"):
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue_923}"
mkdir -p "$LOG_DIR"

# CPU-lane BLAS/OMP thread caps (pooled-round parity). Default 8 = the
# cpu-mid lane's full vCPU width (a no-op there); env-overridable, and it
# prevents oversubscription thrash on a wider shared host.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

echo "[phase=fetch_mixed_packs]"
# GCE lane has NO .env (startup script exports tokens); CONDITIONAL sourcing
# only (the e9c8809113 / att-20260703-163121 rule). set -a exports for the
# heredoc one-liner (heredoc-dotenv rule).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
uv run python - <<'PY'
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from issue923_common import (
    HF_DATA_REPO,
    HF_POOLED_CAPTURE_REVISION,
    HF_PREFIX_923,
    hf_revision,
)


# SCOPED list_repo_tree (server-side prefix) — a bare list_repo_files full
# listing of the ~1M-file data repo times out (#833 gotcha) — with a bounded
# transient retry (pagination retries ONLY 429 on cursor pages).
def scoped_listing(prefix: str, attempts: int = 4) -> list[str]:
    last = None
    for attempt in range(attempts):
        try:
            return [
                e.path
                for e in HfApi().list_repo_tree(
                    HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
                )
            ]
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code not in (429, 500, 502, 503, 504):
                raise
            last = e
            print(f"[hub] transient HTTP {code} listing {prefix} (attempt {attempt+1}/{attempts})")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"scoped listing failed after {attempts} attempts: {prefix}") from last


mixed_prefix = f"{HF_PREFIX_923}/analysis_tensors/mixed_capture/"
mixed_files = scoped_listing(mixed_prefix.rstrip("/"))
assert mixed_files, f"no mixed packs under {mixed_prefix} — run the GPU phase first"
assert f"{mixed_prefix}UPLOAD_COMPLETE_MIXED.json" in mixed_files, (
    "mixed upload sentinel missing — the GPU phase did not finish its upload"
)
mixed_dest = Path("data/issue_923/capture/packs_mixed")
mixed_dest.mkdir(parents=True, exist_ok=True)
for f in mixed_files:
    target = mixed_dest / Path(f).name
    if target.exists():
        continue
    local = hf_hub_download(HF_DATA_REPO, f, repo_type="dataset", local_dir="data/issue_923/hf_dl")
    target.write_bytes(Path(local).read_bytes())
print(f"{mixed_prefix}: {len(mixed_files)} files staged")

# Belt-and-suspenders k1 re-check: the GPU phase already gated on this (a
# failed gate exits nonzero there), but a mis-dispatched CPU phase must not
# spend the battery on a broken join (plan §6 k1 HALT rule).
identity = json.loads((mixed_dest / "identity_check_mix.json").read_text())
assert identity.get("pass") is True, f"k1 identity gate FAILED upstream: {identity}"
print("k1 identity gate (mixed): PASS (re-verified)")

# Reused pool_fctx packs (the arm_concat_mix ctx part) at the PINNED POOLED
# revision (plan v9 §4.1: revision ecf9b613...), + the parent target/reduce
# packs at the PINNED parent revision (byte-identical reused targets).
parent_rev = hf_revision("datasets", HF_DATA_REPO)
for prefix, names, dest, rev in (
    (
        f"{HF_PREFIX_923}/analysis_tensors/pooled_capture/",
        [f"pool_fctx_shard{k}of4.pt" for k in range(4)],
        Path("data/issue_923/capture/packs_pooled"),
        HF_POOLED_CAPTURE_REVISION,
    ),
    (
        f"{HF_PREFIX_923}/analysis_tensors/capture/",
        [f"tgt_ucext_shard{k}of4.pt" for k in range(4)]
        + [f"tgt_dolly_shard{k}of4.pt" for k in range(4)],
        Path("data/issue_923/capture/packs"),
        parent_rev,
    ),
    (
        f"{HF_PREFIX_923}/analysis_tensors/reduce/",
        ["vbar_store_uc.pt", "vbar_store_betley.pt"],
        Path("data/issue_923/reduce"),
        parent_rev,
    ),
):
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = dest / name
        if target.exists():
            continue
        local = hf_hub_download(
            HF_DATA_REPO,
            f"{prefix}{name}",
            repo_type="dataset",
            revision=rev,
            local_dir="data/issue_923/hf_dl",
        )
        target.write_bytes(Path(local).read_bytes())
    print(f"{prefix}: {len(names)} files staged @ {rev[:12]}")
PY

echo "[phase=time_projection]"
# §12 A5/A8 verification: re-derive the per-unit fit costs ON THIS instance
# before committing the battery (compute-deviation rule; logged, non-gating).
EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cpu}" \
  uv run python scripts/issue923_fit_decomposition.py --time-projection \
  2>&1 | tee "$LOG_DIR/mixed_time_projection.log"

echo "[phase=fits]"
EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cpu}" \
  uv run python scripts/issue923_fit_decomposition.py \
  --feature-source pool --arms arm_qry_mix,arm_concat_mix "$@" \
  2>&1 | tee "$LOG_DIR/mixed_fits.log"

echo "[phase=figures]"
# --upload: this instance is EPHEMERAL — figures must land on HF before exit
# (the VM fetches + commits them to git at Step 8).
uv run python scripts/issue923_fig_mixed.py --upload 2>&1 | tee "$LOG_DIR/mixed_figures.log"

echo "[phase=done]"
