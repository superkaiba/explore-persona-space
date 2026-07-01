#!/usr/bin/env python3
"""Issue #811 — upload the re-extracted paired store to HF before releasing the GPU pod.

Bulk-commits every per-cell ``.npz`` under ``eval_results/issue_811/analysis_tensors/``
(the Phase-1 paired store) AND ``eval_results/issue_811/phase0_base_leg/`` (the
Phase-0 base-leg store — the KILL-1 gate input, plan §4.0/§7) to the HF data repo
in ONE ``create_commit`` (well under the 256-commits/hr cap; Upload Policy),
verifies the full complement on a FRESH Hub listing before trusting the pod can be
released (analysis-tensor Upload Policy #521 — both stores are plan-referenced
downstream inputs: the paired store feeds the Phase-2 fits, the phase0 store feeds
the KILL-1 base-leg gate, so losing either makes the corresponding read permanently
unrunnable). Fail-loud: any missing file after the commit raises non-zero.

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
# (local dir, HF prefix) — the two #811 stores that MUST land on HF before the
# GPU pod is released (Upload Policy #521 — both are plan-referenced downstream
# inputs). The paired store feeds the Phase-2 fits; the phase0 base-leg store
# feeds the KILL-1 base-leg gate (plan §4.0/§7).
STORES = (
    ("eval_results/issue_811/analysis_tensors", "issue811_turn_nl_mapchange/analysis_tensors"),
    ("eval_results/issue_811/phase0_base_leg", "issue811_turn_nl_mapchange/phase0_base_leg"),
)


def upload_store() -> int:
    """Bulk-commit + verify the #811 paired + phase0 stores. Returns 0 on success."""
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        logger.info("EPM_SKIP_UPLOAD=1 -> skipping #811 store upload (local/smoke)")
        return 0
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    api = HfApi()
    # Collect ops across BOTH stores into ONE commit (one Hub commit, not two;
    # 256-commits/hr cap). Each op carries its store's own HF prefix.
    #
    # BOTH stores are required uploads (plan §10 Reproducibility Card): the Phase-1
    # paired store (analysis_tensors/) feeds the Phase-2 fits and the Phase-0 base-leg
    # store (phase0_base_leg/) feeds the KILL-1 base-leg gate — losing EITHER makes the
    # corresponding read permanently unrunnable (Upload Policy #521). So require each
    # store to carry >=1 .npz BEFORE any commit: a per-store precondition, NOT a single
    # aggregate check. An aggregate "if not ops" (any store non-empty) would silently
    # commit + verify only the populated store while omitting the other entirely
    # (round-3 Major upload-store-does-not-require-both-stores).
    ops: list[CommitOperationAdd] = []
    expected: list[str] = []  # path_in_repo for the fresh-listing verify
    per_store_counts: list[tuple[str, int]] = []
    for local_dir, hf_prefix in STORES:
        tdir = PROJECT_ROOT / local_dir
        npzs = sorted(tdir.rglob("*.npz")) if tdir.is_dir() else []
        per_store_counts.append((local_dir, len(npzs)))
        for p in npzs:
            pir = f"{hf_prefix}/{p.relative_to(tdir).as_posix()}"
            ops.append(CommitOperationAdd(path_in_repo=pir, path_or_fileobj=str(p)))
            expected.append(pir)
    empty_stores = [d for d, n in per_store_counts if n == 0]
    if empty_stores:
        raise RuntimeError(
            f"#811 upload: {empty_stores} has 0 .npz -- BOTH stores are required uploads "
            f"(plan §10; analysis_tensors feeds Phase-2 fits, phase0_base_leg feeds the "
            f"KILL-1 gate). Refusing to commit an INCOMPLETE upload. Per-store counts: "
            f"{per_store_counts}"
        )
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue811: {len(ops)} per-cell paired + phase0 base-leg tensors",
    )
    files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [pir for pir in expected if pir not in files]
    if missing:
        raise RuntimeError(
            f"#811 store upload verification FAILED -- missing on Hub: {missing[:5]}"
        )
    logger.info("uploaded + verified %d #811 store tensors to %s", len(ops), HF_DATA_REPO)
    return 0


if __name__ == "__main__":
    raise SystemExit(upload_store())
