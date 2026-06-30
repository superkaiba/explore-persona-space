#!/usr/bin/env python3
"""Issue #763: upload raw completions + v0/r_B analysis tensors to HF (pre-teardown).

Per the Upload Policy (CLAUDE.md): raw completions AND plan-referenced analysis
tensors MUST land on the HF data repo BEFORE the GPU pod is released. This runs
as the dispatcher's ``[phase=upload]`` step (after capture + PV extraction,
before the CPU-only judge/fit). It:

1. uploads every ``raw_completions.json`` under ``eval_results/issue_763/`` via
   the canonical bulk helper ``upload_raw_completions_to_data_repo`` (ONE
   ``upload_folder`` commit; #664 per-file-loop trap avoided);
2. bulk-uploads the v0 + r_B ``.pt`` shards to
   ``issue763_matched_v0/analysis_tensors/`` (the plan-named downstream inputs —
   losing them makes the analysis unrunnable, #521);
3. writes the ``epm:results`` end-of-run sentinel that ``poll_pipeline.py``
   drains (the only pod-side -> orchestrator channel; pod code NEVER shells
   scripts/task.py).

Usage::

    uv run python scripts/issue763_upload.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue763_common import (  # noqa: E402
    EVAL_RESULTS_DIR,
    EXPERIMENT_NAME,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_DATA_REPO,
    HF_OVERFLOW_REPO,
    is_storage_quota_403,
    write_sentinel,
)

from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo  # noqa: E402

logger = logging.getLogger("issue763_upload")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _upload_analysis_tensors() -> dict:
    """Bulk upload the v0 + r_B .pt shards (ONE upload_folder commit each dir)."""
    from huggingface_hub import HfApi

    api = HfApi()
    uploaded = {}
    for sub in ("v0_shards", "pv_shards"):
        local = EVAL_RESULTS_DIR / sub
        if not local.is_dir() or not any(local.glob("*.pt")):
            continue
        path_in_repo = f"{HF_ANALYSIS_TENSORS_PREFIX}/{sub}"
        repo_used = HF_DATA_REPO
        try:
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=path_in_repo,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                allow_patterns=["*.pt"],
                commit_message=f"issue763: analysis tensors {sub}",
            )
        except Exception as e:
            if not is_storage_quota_403(e):
                raise
            logger.warning("HF storage 403 on %s; overflow repo", sub)
            repo_used = HF_OVERFLOW_REPO
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=path_in_repo,
                repo_id=HF_OVERFLOW_REPO,
                repo_type="dataset",
                allow_patterns=["*.pt"],
                commit_message=f"issue763: analysis tensors {sub} (overflow)",
            )
        files = [
            f
            for f in api.list_repo_files(repo_used, repo_type="dataset")
            if f.startswith(path_in_repo)
        ]
        uploaded[sub] = {"repo": repo_used, "path_in_repo": path_in_repo, "n_files": len(files)}
        logger.info("uploaded %d %s tensors -> %s/%s", len(files), sub, repo_used, path_in_repo)
    return uploaded


def main() -> int:
    raw_map = upload_raw_completions_to_data_repo(
        experiment_name=EXPERIMENT_NAME,
        eval_results_dir=EVAL_RESULTS_DIR,
    )
    logger.info("uploaded %d raw_completions files", len(raw_map))

    tensors = _upload_analysis_tensors()

    note = {
        "task_id": 763,
        "experiment_name": EXPERIMENT_NAME,
        "n_raw_completions_files": len(raw_map),
        "analysis_tensors": tensors,
        "hf_data_repo": HF_DATA_REPO,
        "reproducibility_card": {
            "raw_completions_prefix": f"{EXPERIMENT_NAME}/raw_completions",
            "analysis_tensors_prefix": HF_ANALYSIS_TENSORS_PREFIX,
            "no_training": True,  # base-model read; no adapters / WandB runs
            "note": "base-model-only predictor re-measurement; no trained adapters",
        },
    }
    path = write_sentinel("epm:results", note, task_id=763)
    logger.info("wrote epm:results sentinel -> %s", path)
    print(f"[issue763.upload] raw={len(raw_map)} tensors={tensors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
