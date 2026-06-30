#!/usr/bin/env python3
"""Issue #617 Step 7: upload the clustered corpus to the HF data repo.

Per plan §4 step 7. ONE bulk ``upload_folder`` commit to
``superkaiba1/explore-persona-space-data`` under prefix
``issue617_wildchat_categories/``, verified via ``list_repo_files`` (never the
``hf`` CLI). On the account-wide LFS storage-quota 403, the ``.pt`` tensors
fall back to the private overflow repo (JSON rides the non-LFS path
regardless), recording the deviation.

Mirror structure (plan §4 step 7):
    issue617_wildchat_categories/
      wildchat_slice.json
      cluster_assignments.json
      separability.json
      picked_categories/<cat>/{prefixes,prefix_plus_completion}.json
      extraction/analysis_tensors/...

Runs off-pod on the VM (CPU). Builds a staging tree first so one commit covers
all artifacts. Emits ``[phase=...]`` lines for poll_pipeline.py.

Usage::

    uv run python scripts/issue617_upload_corpus.py
    uv run python scripts/issue617_upload_corpus.py --dry-run  # stage only, no HF upload
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper (#745): reads the project .env robustly (worktree-aware
# resolve_dotenv_path) AND setdefaults the HF Hub upload accelerators
# (HF_XET_HIGH_PERFORMANCE / HF_HUB_ENABLE_HF_TRANSFER) — load_dotenv() runs at
# module scope (below), before this script's in-function `huggingface_hub` import.
from issue617_common import (  # noqa: E402
    CLUSTER_PATH,
    DATA_DIR,
    EXTRACTION_DIR,
    HF_DATA_REPO,
    HF_OVERFLOW_REPO,
    HF_PREFIX,
    PICKED_DIR,
    SEPARABILITY_PATH,
    SLICE_PATH,
    TASK_ID,
)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue617_upload")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


def _is_storage_quota_403(err: Exception) -> bool:
    msg = str(err)
    return "403" in msg and "storage" in msg.lower()


def stage_corpus(stage: Path) -> dict[str, int]:
    """Assemble the upload staging tree mirroring the plan §4 step 7 layout."""
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    counts: dict[str, int] = {}

    def _copy(src: Path, rel: str) -> None:
        if not src.exists():
            raise RuntimeError(f"missing required artifact for upload: {src}")
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    _copy(SLICE_PATH, "wildchat_slice.json")
    _copy(CLUSTER_PATH, "cluster_assignments.json")
    _copy(SEPARABILITY_PATH, "separability.json")
    counts["json_top"] = 3

    # picked_categories/<cat>/{prefixes,prefix_plus_completion}.json
    n_picked = 0
    if PICKED_DIR.exists():
        for cat_dir in sorted(PICKED_DIR.iterdir()):
            if not cat_dir.is_dir():
                continue
            for fname in ("prefixes.json", "prefix_plus_completion.json"):
                src = cat_dir / fname
                if src.exists():
                    _copy(src, f"picked_categories/{cat_dir.name}/{fname}")
                    n_picked += 1
    counts["picked_files"] = n_picked

    # extraction/analysis_tensors/... (the #594-format activation tensors)
    n_tensors = 0
    if EXTRACTION_DIR.exists():
        for src in sorted(EXTRACTION_DIR.rglob("*")):
            if src.is_file():
                rel = src.relative_to(EXTRACTION_DIR)
                _copy(src, f"extraction/analysis_tensors/{rel}")
                n_tensors += 1
    counts["extraction_files"] = n_tensors
    return counts


def write_sentinel(note: str, task_id: int = TASK_ID) -> Path:
    """poll_pipeline.py-conformant end-of-run sentinel (_SENTINEL_REQUIRED_KEYS)."""
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"issue-{task_id}-epm_results-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "note": note,
        "task_id": task_id,
        "by": "issue617_upload_corpus",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sentinel %s", path)
    return path


def expected_repo_paths(stage: Path) -> set[str]:
    """Every staged file's expected repo path under HF_PREFIX/.

    Verification covers ALL primary deliverables (plan §6.5), not just the 3
    top-level JSONs: the 2 picked-category subfolders' prefixes.json +
    prefix_plus_completion.json AND the extraction analysis tensors
    (context_vectors_mean.pt + extraction_manifest.json) are enumerated from
    the staging tree the bulk commit just pushed — the deterministic ground
    truth of what must land.
    """
    return {
        f"{HF_PREFIX}/{p.relative_to(stage).as_posix()}"
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }


def upload_and_verify(stage: Path) -> dict:
    """ONE bulk upload_folder + Hub-API verify. LFS quota-403 -> overflow repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_used = HF_DATA_REPO
    try:
        api.upload_folder(
            folder_path=str(stage),
            path_in_repo=HF_PREFIX,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message="issue617: WildChat-category corpus + separability + tensors",
        )
    except Exception as e:
        if not _is_storage_quota_403(e):
            raise
        logger.warning("HF storage-quota 403 on %s; falling back to overflow repo", HF_DATA_REPO)
        repo_used = HF_OVERFLOW_REPO
        api.upload_folder(
            folder_path=str(stage),
            path_in_repo=HF_PREFIX,
            repo_id=HF_OVERFLOW_REPO,
            repo_type="dataset",
            commit_message="issue617: corpus upload (quota-403 overflow fallback)",
        )
    files = {
        f
        for f in api.list_repo_files(repo_used, repo_type="dataset")
        if f.startswith(HF_PREFIX + "/")
    }
    # Verify every staged file landed (top-level JSONs + picked-category files +
    # extraction tensors), plus the 3 always-required top-level JSONs as an
    # explicit floor in case the stage tree was somehow incomplete.
    expected = expected_repo_paths(stage) | {
        f"{HF_PREFIX}/wildchat_slice.json",
        f"{HF_PREFIX}/cluster_assignments.json",
        f"{HF_PREFIX}/separability.json",
    }
    missing = expected - files
    if missing:
        raise RuntimeError(f"upload verification failed; missing on {repo_used}: {sorted(missing)}")
    logger.info("Upload verified on %s: %d files under %s/", repo_used, len(files), HF_PREFIX)
    return {"repo": repo_used, "path_in_repo": HF_PREFIX, "n_files": len(files)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #617 Step 7: upload corpus to HF.")
    parser.add_argument("--stage", type=Path, default=DATA_DIR / "_upload_stage")
    parser.add_argument("--dry-run", action="store_true", help="stage only; no HF upload")
    args = parser.parse_args()

    phase("stage")
    counts = stage_corpus(args.stage)
    logger.info("Staged: %s", counts)

    if args.dry_run:
        logger.info("--dry-run: staged at %s; skipping HF upload", args.stage)
        phase("done")
        return 0

    phase("upload")
    info = upload_and_verify(args.stage)
    note = (
        f"issue617 corpus uploaded: {info['n_files']} files under "
        f"{info['path_in_repo']} on {info['repo']}"
    )
    write_sentinel(note)
    phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
