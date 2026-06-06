"""Issue #498 — Q_train (60) / Q_test (40) loaders.

The Q-bank is produced by ``scripts/i498_phase0_preflight.py`` and persisted
to ``data/issue_498/Q_train.json`` + ``Q_test.json`` with
``schema_version="i498_qbank_v1"``. These loaders refuse to run until those
files exist (no on-the-fly fallback — preflight is the gate).
"""

from __future__ import annotations

import json
from pathlib import Path

LOCAL_DATA_DIR = Path("data/issue_498")
Q_TRAIN_PATH = LOCAL_DATA_DIR / "Q_train.json"
Q_TEST_PATH = LOCAL_DATA_DIR / "Q_test.json"
Q_ELIGIBILITY_PATH = LOCAL_DATA_DIR / "Q_eligibility.json"
SCHEMA_VERSION = "i498_qbank_v1"


def _load_split(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/i498_phase0_preflight.py first to "
            "build the Q-bank (plan §4.1 Phase 0)."
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError(
            f"{path} schema_version={payload.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r} — refusing to mix Q-bank versions."
        )
    questions = payload["questions"]
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise AssertionError(f"{path} 'questions' must be list[str]; got {type(questions)}.")
    return list(questions)


def load_q_train() -> list[str]:
    return _load_split(Q_TRAIN_PATH)


def load_q_test() -> list[str]:
    return _load_split(Q_TEST_PATH)


def assert_disjoint() -> None:
    train = set(load_q_train())
    test = set(load_q_test())
    overlap = train & test
    if overlap:
        raise AssertionError(
            f"Q_train and Q_test overlap on {len(overlap)} questions; first 3: {list(overlap)[:3]}"
        )
