# em-dash intentional
"""Task #601 — parent-artifact fetch helpers (HF data/model repo → local paths).

Per-file ``hf_hub_download`` (NOT ``snapshot_download`` — its
``repo_info.siblings`` listing truncates on large repos and silently returns
0 files for tail prefixes; feedback_snapshot_download_siblings_truncation).
Idempotent: existing local files are kept.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from explore_persona_space.experiments.neg_setpoint_601 import (
    HF_ADAPTER_PREFIX_472,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    PARENT_DATA_FILES,
)

log = logging.getLogger("issue_601.artifacts")

_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def fetch_parent_data(repo_root: Path) -> dict[str, str]:
    """Download the pinned #472 inputs (bank / centroids / R) from the data repo.

    ``repo_root`` is the repository root the relative ``PARENT_DATA_FILES``
    destinations resolve against (the pod checkout root).
    """
    from huggingface_hub import hf_hub_download

    fetched: dict[str, str] = {}
    for repo_path, local_rel in PARENT_DATA_FILES:
        local = repo_root / local_rel
        if local.exists():
            fetched[repo_path] = str(local)
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=repo_path,
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copyfile(got, local)
        fetched[repo_path] = str(local)
        log.info("[fetch] %s -> %s", repo_path, local)
    return fetched


def fetch_parent_adapter(cell: str, seed: int, dest_root: Path) -> Path:
    """Download one #472 final adapter (config + safetensors) from the model repo.

    Returns the local adapter directory. Fail-loud on a missing file (the 20
    adapters were Hub-verified at plan time; absence here means repo drift).
    """
    from huggingface_hub import hf_hub_download

    dest = dest_root / f"{cell}_seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in _ADAPTER_FILES:
        local = dest / fname
        if local.exists():
            continue
        got = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            filename=f"{HF_ADAPTER_PREFIX_472}/{cell}_seed{seed}/{fname}",
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copyfile(got, local)
    for fname in _ADAPTER_FILES:
        if not (dest / fname).exists():
            raise RuntimeError(f"parent adapter fetch incomplete: {dest / fname} missing")
    return dest
