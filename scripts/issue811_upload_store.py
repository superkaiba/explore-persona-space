#!/usr/bin/env python3
"""Issue #811 — upload the re-extracted paired store to HF before releasing the GPU pod.

Bulk-commits every per-cell ``.npz`` under ``eval_results/issue_811/analysis_tensors/``
to the HF data repo at ``issue811_turn_nl_mapchange/analysis_tensors/`` in ONE
``create_commit`` (well under the 256-commits/hr cap; Upload Policy) and verifies
the full complement on a FRESH Hub listing before trusting the pod can be
released (analysis-tensor Upload Policy #521 — the store is the Phase-2 fit
input, so losing it makes the fits permanently unrunnable). Fail-loud: any
missing file after the commit raises non-zero.

Reuses the exact bulk-commit + fresh-listing-verify shape of
``issue667_dispatch._upload_tensors`` (this store is the #811 analogue). Skipped
when ``EPM_SKIP_UPLOAD=1`` (a local CPU smoke that reads the store from disk).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# DOTENV_LINT_EXEMPT: analysis-phase script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue811.upload_store")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_STORE_PREFIX = "issue811_turn_nl_mapchange/analysis_tensors"
TENSORS_DIR = "eval_results/issue_811/analysis_tensors"


def upload_store() -> int:
    """Bulk-commit + verify the #811 paired store. Returns 0 on success (else raises)."""
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        logger.info("EPM_SKIP_UPLOAD=1 -> skipping #811 store upload (local/smoke)")
        return 0
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    tdir = PROJECT_ROOT / TENSORS_DIR
    npzs = sorted(tdir.rglob("*.npz"))
    if not npzs:
        raise RuntimeError(f"no .npz tensors under {tdir} -- #811 extraction wrote nothing")
    api = HfApi()
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{HF_STORE_PREFIX}/{p.relative_to(tdir).as_posix()}",
            path_or_fileobj=str(p),
        )
        for p in npzs
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue811: {len(ops)} per-cell turn_nl+mean paired tensors",
    )
    files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [
        p.relative_to(tdir).as_posix()
        for p in npzs
        if f"{HF_STORE_PREFIX}/{p.relative_to(tdir).as_posix()}" not in files
    ]
    if missing:
        raise RuntimeError(
            f"#811 store upload verification FAILED -- missing on Hub: {missing[:5]}"
        )
    logger.info("uploaded + verified %d #811 store tensors to %s", len(npzs), HF_DATA_REPO)
    return 0


if __name__ == "__main__":
    raise SystemExit(upload_store())
