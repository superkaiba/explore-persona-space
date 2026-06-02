"""Issue #464 shared helpers — Q_train (30) / Q_test (50) loaders with HF fallback.

Issue #464 plan v2 §4.6 (A23): self-contained copy of #460's `i460_data.py`
Q_train + Q_test loaders. We omit Class-D rewrites because #464 only varies
the chat-template encoding (system_plain / system_padded / role); the user
turn is always the plain question `q` (Class-A or Class-D rewrites are NOT
used here).

The Q_train answers + Q_test extended JSON files live at
``data/issue_406/`` under .gitignore. On a clean pod checkout (or fresh
worktree) they may be absent, in which case each loader pulls the
canonical copy from the HF data repo at
``superkaiba1/explore-persona-space-data/issue406_divergence_predicts_transfer/training_data/``
(the path #406 itself uploaded them to in its Phase 0 dispatcher).

All loaders raise loudly on missing-after-fallback so a downstream phase
never silently runs on partial data — per CLAUDE.md fail-fast.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("i464.data")

DATA_DIR_LOCAL = Path("data/issue_406")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_TRAINING_DATA_PREFIX = "issue406_divergence_predicts_transfer/training_data"

# Filenames-as-canonical: same name on disk and on HF Hub.
_Q_TRAIN_FILE = "q_train_answers.json"
_Q_TEST_FILE = "q_test_extended_50.json"


def _ensure_local_file(rel_path: str) -> Path:
    """Return the absolute Path to ``data/issue_406/<rel_path>``.

    If the file is absent locally, pull it from the HF data repo via
    ``huggingface_hub.hf_hub_download``. Fails loud if both the local copy
    and the HF copy are missing.

    Per CLAUDE.md feedback_snapshot_download_siblings_truncation: prefer
    ``hf_hub_download`` (per-file) over ``snapshot_download`` (siblings
    list, can truncate on large repos).
    """
    local = DATA_DIR_LOCAL / rel_path
    if local.exists() and local.stat().st_size > 0:
        return local

    from huggingface_hub import hf_hub_download

    hf_path = f"{HF_TRAINING_DATA_PREFIX}/{rel_path}"
    logger.info("Local %s missing; pulling %s from HF data repo %s", local, hf_path, HF_DATA_REPO)

    local.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=hf_path,
        revision="main",
    )
    shutil.copyfile(downloaded, local)

    if not local.exists() or local.stat().st_size == 0:
        raise RuntimeError(
            f"HF download claimed success but {local} is missing or empty after copy "
            f"from {downloaded}. HF path was {HF_DATA_REPO}:{hf_path}."
        )
    return local


def load_q_train_answers() -> dict[str, str]:
    """Load the 30 Q_train question -> Claude-answer mapping (#406 Phase 0 artifact).

    Returns:
        Mapping of question -> Claude answer text. Exactly 30 entries.

    Raises:
        AssertionError if the count != 30 (HF data repo drift guard).
    """
    path = _ensure_local_file(_Q_TRAIN_FILE)
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"q_train_answers.json expected dict, got {type(payload).__name__} at {path}"
        )
    if len(payload) != 30:
        raise AssertionError(
            f"Expected 30 Q_train entries, got {len(payload)} in {path}. "
            "Did the HF data repo drift since #406?"
        )
    return payload


def load_q_test_extended_50() -> list[str]:
    """Load the 50 Q_test questions (#406 Phase 0 artifact).

    Returns:
        The list of 50 question strings.

    Raises:
        AssertionError if the count != 50.
    """
    path = _ensure_local_file(_Q_TEST_FILE)
    with open(path) as f:
        payload = json.load(f)
    qs = payload["questions"]
    if len(qs) != 50:
        raise AssertionError(f"Expected 50 Q_test questions, got {len(qs)} in {path}")
    return qs


def assert_disjoint_q_train_q_test(q_train: list[str], q_test: list[str]) -> None:
    """Verify Q_train and Q_test share no exact-string questions.

    Defense-in-depth guard against HF data-repo drift; the #406 Phase 0
    already enforces this at generation time.

    Raises:
        AssertionError on any overlap.
    """
    overlap = set(q_train) & set(q_test)
    if overlap:
        raise AssertionError(
            f"Q_train ∩ Q_test contains {len(overlap)} question(s): {sorted(overlap)[:3]}..."
        )
