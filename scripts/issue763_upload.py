#!/usr/bin/env python3
"""Issue #763: upload raw completions + v0/r_B analysis tensors to HF (pre-teardown).

Per the Upload Policy (CLAUDE.md): raw completions AND plan-referenced analysis
tensors MUST land on the HF data repo BEFORE the GPU pod is released. It:

1. uploads every ``raw_completions.json`` under ``eval_results/issue_763/`` via
   the canonical bulk helper ``upload_raw_completions_to_data_repo`` (ONE
   ``upload_folder`` commit; #664 per-file-loop trap avoided);
2. bulk-uploads the v0 + r_B ``.pt`` shards to
   ``issue763_matched_v0/analysis_tensors/`` (the plan-named downstream inputs —
   losing them makes the analysis unrunnable, #521);
3. writes a SENTINEL that ``poll_pipeline.py`` drains (the only pod-side ->
   orchestrator channel; pod code NEVER shells scripts/task.py).

Two invocations in the dispatch (#763 CONCERN premature-results-sentinel):

- ``--progress-only`` — runs WHILE the GPU pod is live (raw completions +
  rollouts must land before teardown) and writes a NON-FINAL
  ``epm:upload-progress`` sentinel. An observing orchestrator must NOT see
  ``epm:results`` here, because judge/fit/figures have not produced their
  primary deliverables yet.
- (default, no flag) — runs LAST, AFTER fit + figures exist, re-uploads the
  full artifact set (now including the captured r_B), and writes the
  ``epm:results`` END-OF-RUN sentinel.

Usage::

    uv run python scripts/issue763_upload.py --progress-only   # pre-teardown
    uv run python scripts/issue763_upload.py                    # final, end-of-run
"""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser(description="Issue #763: upload artifacts + sentinel.")
    ap.add_argument(
        "--progress-only",
        action="store_true",
        help="pre-teardown upload; write epm:upload-progress (NOT the final epm:results)",
    )
    args = ap.parse_args()

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
    # CONCERN premature-results-sentinel: the pre-teardown upload writes a
    # NON-FINAL sentinel; the END-OF-RUN epm:results is written only by the
    # final (no-flag) invocation, after fit + figures have landed.
    if args.progress_only:
        kind = "epm:upload-progress"
        note["phase"] = "pre-teardown upload (raw completions + rollouts); deliverables pending"
    else:
        kind = "epm:results"
    path = write_sentinel(kind, note, task_id=763)
    logger.info("wrote %s sentinel -> %s", kind, path)
    print(f"[issue763.upload] {kind} raw={len(raw_map)} tensors={tensors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
