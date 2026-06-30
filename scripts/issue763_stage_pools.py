#!/usr/bin/env python3
"""Issue #763: stage the frozen probe pools from the HF inputs mirror.

The frozen pools are built off-pod and uploaded to the issue-owned HF inputs
mirror (``issue763_matched_v0/inputs/probe_pools/``) so the git-clone-only GCP
lane can ``snapshot_download`` them (the local ``data/issue_763/`` is untracked
and absent from a fresh clone — artifact-reuse check (h)). This script
downloads that mirror to ``data/issue_763/probe_pools/`` if the pools are not
already present locally, and validates each pool's ``probe_pool_hash`` on load.

Exits non-zero if neither the local pools NOR the HF mirror are available (the
dispatcher then falls back to running the builder, which needs the Sonnet API).

Usage::

    uv run python scripts/issue763_stage_pools.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue763_common import (  # noqa: E402
    BEHAVIORS,
    HF_DATA_REPO,
    HF_INPUTS_PREFIX,
    PROBE_POOL_DIR,
    load_frozen_pool,
    probe_pool_path,
)

logger = logging.getLogger("issue763_stage")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _all_local() -> bool:
    return all(probe_pool_path(b).exists() for b in BEHAVIORS)


def main() -> int:
    if _all_local():
        for b in BEHAVIORS:
            load_frozen_pool(b)  # validates probe_pool_hash (fail-loud)
        logger.info("all %d frozen pools present locally + hash-validated", len(BEHAVIORS))
        return 0

    from huggingface_hub import snapshot_download

    logger.info("staging probe pools from %s/%s", HF_DATA_REPO, HF_INPUTS_PREFIX)
    local = snapshot_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=[f"{HF_INPUTS_PREFIX}/*.json"],
    )
    src_dir = Path(local) / HF_INPUTS_PREFIX
    if not src_dir.is_dir():
        logger.error("HF inputs mirror %s missing under %s", HF_INPUTS_PREFIX, local)
        return 1
    PROBE_POOL_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in src_dir.glob("*.json"):
        dst = PROBE_POOL_DIR / src.name
        dst.write_bytes(src.read_bytes())
        n += 1
    logger.info("staged %d pool files -> %s", n, PROBE_POOL_DIR)
    # validate every behavior's pool (fail-loud on hash drift / missing)
    for b in BEHAVIORS:
        load_frozen_pool(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
