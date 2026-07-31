"""Pre-stage the two bareq-scorer inputs the arms runner's --stage-only misses.

1. The wcrung committed-contrast arm results (small JSONs) -> the scorer's
   --main-root location eval_results/issue_1739/wildchat_rung/arm_results
   (HF-only artifact; without it the scorer records committed_contrast
   unavailable rather than failing — we want it available).
2. The bareq capture store (issue1739_ctxmap/bareq_map/capture_store, ~859 MB,
   children bareq/ + bareq_evil/) -> the scorer's default resolution path
   <store-root>/bareq_capture_store.

stage_hub_prefix mirrors the repo-relative tree under its dest root
(root/<prefix> == staged, the #1774 semantics), so both stages mirror then
move/copy into the consumed location. Idempotent: present-and-nonempty
destinations are skipped.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402

STORE_ROOT = Path("data/issue_1739/hf_dl")
ARM_PREFIX = "issue1739_ctxmap/wildchat_rung/arm_results"
ARM_DEST = Path("eval_results/issue_1739/wildchat_rung/arm_results")
BQ_PREFIX = "issue1739_ctxmap/bareq_map/capture_store"
BQ_DEST = STORE_ROOT / "bareq_capture_store"


def _stage(prefix: str, dest: Path, mirror_root: Path) -> None:
    if dest.is_dir() and any(dest.iterdir()):
        print(f"[bareq-prestage] present, skip: {dest}", flush=True)
        return
    hub.stage_hub_prefix(hub.DEFAULT_DATASET_REPO, prefix, mirror_root, repo_type="dataset")
    staged = mirror_root / prefix
    if not staged.is_dir() or not any(staged.iterdir()):
        raise RuntimeError(f"staging incomplete: {staged} empty for prefix {prefix}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    # same-filesystem rename when possible; copytree is the cross-device fallback
    try:
        staged.rename(dest)
    except OSError:
        shutil.copytree(staged, dest)
    print(f"[bareq-prestage] {prefix} -> {dest}", flush=True)


MANIFEST_HF = "issue1739_ctxmap/bareq_map/manifests/bareq_queries.json"
MANIFEST_DEST = Path("eval_results/issue_1739/bareq_map/bareq_queries.json")


def main() -> int:
    _stage(ARM_PREFIX, ARM_DEST, STORE_ROOT / "_armmirror")
    _stage(BQ_PREFIX, BQ_DEST, STORE_ROOT / "_bqmirror")
    # leg 2's query bank (produced by the capture box's extract phase, uploaded
    # under manifests/) — the scorer requires it at its out_root
    # (att-20260731-161235 failed evil/leg2 on exactly this).
    if not MANIFEST_DEST.is_file():
        hub.stage_hub_file(
            hub.DEFAULT_DATASET_REPO, MANIFEST_HF, MANIFEST_DEST, repo_type="dataset"
        )
        print(f"[bareq-prestage] {MANIFEST_HF} -> {MANIFEST_DEST}", flush=True)
    else:
        print(f"[bareq-prestage] present, skip: {MANIFEST_DEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
