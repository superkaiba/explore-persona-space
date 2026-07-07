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

    # PER-FILE hf_hub_download by EXACT path, NOT snapshot_download(allow_patterns=...):
    # the data repo carries >94k files (12x past the ~7900-siblings truncation
    # point), so a pattern-filtered snapshot_download can silently match 0 files
    # and this stage would "succeed" having downloaded NOTHING, then fail loud
    # only at load_frozen_pool (task #763 CONCERN stage-pools-snapshot-download-
    # siblings-truncation; 4th site in the family; standing lesson
    # feedback_snapshot_download_siblings_truncation.md / #375/#399 — same fix the
    # r2-r5 rounds applied to the v0 / gen / pv staging helpers). hf_hub_download
    # resolves one file by exact path — no siblings listing, no truncation.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    logger.info("staging probe pools from %s/%s (per-file)", HF_DATA_REPO, HF_INPUTS_PREFIX)
    PROBE_POOL_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for b in BEHAVIORS:
        try:
            src = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_INPUTS_PREFIX}/{b}.json",
            )
        except EntryNotFoundError:
            # A pool missing on HF means the mirror was never uploaded — surface a
            # clean non-zero so the dispatcher falls back to the builder (which
            # needs the Sonnet API), matching the prior "mirror missing" branch.
            logger.error(
                "probe pool %s not on HF (%s/%s) — mirror not uploaded; "
                "dispatcher will fall back to the builder",
                b,
                HF_DATA_REPO,
                HF_INPUTS_PREFIX,
            )
            return 1
        (PROBE_POOL_DIR / f"{b}.json").write_bytes(Path(src).read_bytes())
        n += 1
    logger.info("staged %d pool files -> %s", n, PROBE_POOL_DIR)
    # validate every behavior's pool (fail-loud on hash drift / missing)
    for b in BEHAVIORS:
        load_frozen_pool(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
