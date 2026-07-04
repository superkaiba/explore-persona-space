#!/usr/bin/env bash
# Issue #923 pooled-span-features CPU phase (cpu-mid; plan v6 §4.2/§9).
#
# STRICTLY SEQUENTIAL after the GPU phase (which uploads + TERMINATES first —
# no HF-poll join needed): fetch the pooled feature packs (uploaded by the GPU
# phase; identity_check.json gate re-verified here belt-and-suspenders) + the
# parent target/reduce packs at the PINNED dataset revision -> full fit
# battery with --feature-source pool (incl. the blend + Dolly null extensions
# and the paired residual diff vs the parent) -> pooled figures -> upload.
#
# Usage (dispatched via dispatch_issue.py --intent cpu-mid --boot-disk-gb 100
# --workload-cmd "bash scripts/issue923_pooled_cpu_phase.sh"):
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue_923}"
mkdir -p "$LOG_DIR"

echo "[phase=fetch_pooled_packs]"
# GCE lane has NO .env (startup script exports tokens); CONDITIONAL sourcing
# only (the e9c8809113 / att-20260703-163121 rule). set -a exports for the
# heredoc one-liner (heredoc-dotenv rule).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
uv run python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from huggingface_hub import HfApi, hf_hub_download
from issue923_common import HF_DATA_REPO, HF_PREFIX_923, hf_revision

# SCOPED list_repo_tree (server-side prefix) — a bare list_repo_files full
# listing of the ~1M-file data repo times out (#833 gotcha).
pooled_prefix = f"{HF_PREFIX_923}/analysis_tensors/pooled_capture/"
pooled_files = [
    e.path
    for e in HfApi().list_repo_tree(
        HF_DATA_REPO, path_in_repo=pooled_prefix.rstrip("/"), repo_type="dataset", recursive=True
    )
]
assert pooled_files, f"no pooled packs under {pooled_prefix} — run the GPU phase first"
assert f"{pooled_prefix}UPLOAD_COMPLETE_POOLED.json" in pooled_files, (
    "pooled upload sentinel missing — the GPU phase did not finish its upload"
)
pooled_dest = Path("data/issue_923/capture/packs_pooled")
pooled_dest.mkdir(parents=True, exist_ok=True)
for f in pooled_files:
    target = pooled_dest / Path(f).name
    if target.exists():
        continue
    local = hf_hub_download(
        HF_DATA_REPO, f, repo_type="dataset", local_dir="data/issue_923/hf_dl"
    )
    target.write_bytes(Path(local).read_bytes())
print(f"{pooled_prefix}: {len(pooled_files)} files staged")

# Belt-and-suspenders k1 re-check: the GPU phase already gated on this (a
# failed gate exits nonzero there), but a mis-dispatched CPU phase must not
# spend the battery on a broken join (plan §6 k1 HALT rule).
identity = json.loads((pooled_dest / "identity_check.json").read_text())
assert identity.get("pass") is True, f"k1 identity gate FAILED upstream: {identity}"
print("k1 identity gate: PASS (re-verified)")

# Parent target/reduce packs at the PINNED dataset revision (targets are the
# round's byte-identical reused inputs; §4.1).
rev = hf_revision("datasets", HF_DATA_REPO)
for prefix, names, dest in (
    (
        f"{HF_PREFIX_923}/analysis_tensors/capture/",
        [f"tgt_ucext_shard{k}of4.pt" for k in range(4)]
        + [f"tgt_dolly_shard{k}of4.pt" for k in range(4)],
        Path("data/issue_923/capture/packs"),
    ),
    (
        f"{HF_PREFIX_923}/analysis_tensors/reduce/",
        ["vbar_store_uc.pt", "vbar_store_betley.pt"],
        Path("data/issue_923/reduce"),
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
# §12 A5 verification: re-derive the per-unit fit costs ON THIS instance
# before committing the full battery (compute-deviation rule; logged, non-gating).
EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cpu}" \
  uv run python scripts/issue923_fit_decomposition.py --time-projection \
  2>&1 | tee "$LOG_DIR/pooled_time_projection.log"

echo "[phase=fits]"
EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cpu}" \
  uv run python scripts/issue923_fit_decomposition.py --feature-source pool "$@" \
  2>&1 | tee "$LOG_DIR/pooled_fits.log"

echo "[phase=figures]"
# --upload: this instance is EPHEMERAL — figures must land on HF before exit
# (the VM fetches + commits them to git at Step 8).
uv run python scripts/issue923_fig_pooled.py --upload 2>&1 | tee "$LOG_DIR/pooled_figures.log"

echo "[phase=done]"
