"""Issue #715 — held-out narrow-task split disjointness + rowcount.

The held-out narrow split is the Pareto x-axis (narrow-task acquisition is scored
on these prompts). If it overlaps the train set, "narrow-task acquisition" is
contaminated by memorization. This test asserts:
  - train ∩ holdout == ∅ (disjoint by row content),
  - len(train) + len(holdout) == 7049 (no rows dropped/duplicated),
  - the split is DETERMINISTIC (seed=42 reproduces the same partition).

Uses a synthetic 7049-row messages-schema fixture (no real harmful-content rows
in the test — the content-hygiene rule), so the test runs on any machine with no
HF/network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from issue715_common import (
    BADMED_EXPECTED_ROWS,
    HOLDOUT_N,
    build_holdout_split,
)


def _synthetic_corpus(path, n=BADMED_EXPECTED_ROWS):
    """Write a 7049-row messages-schema JSONL with unique user content per row."""
    with open(path, "w") as f:
        for i in range(n):
            row = {
                "messages": [
                    {"role": "user", "content": f"synthetic question {i}"},
                    {"role": "assistant", "content": f"synthetic answer {i}"},
                ]
            }
            f.write(json.dumps(row) + "\n")
    return path


def _rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _user(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def test_train_holdout_disjoint_and_rowcounts(tmp_path):
    corpus = _synthetic_corpus(tmp_path / "corpus.jsonl")
    train_out = tmp_path / "train.jsonl"
    holdout_out = tmp_path / "holdout.jsonl"
    digest = build_holdout_split(corpus, train_out, holdout_out)

    train = _rows(train_out)
    holdout = _rows(holdout_out)

    # Rowcounts add up exactly to 7049.
    assert len(train) + len(holdout) == BADMED_EXPECTED_ROWS
    assert len(holdout) == HOLDOUT_N
    assert len(train) == BADMED_EXPECTED_ROWS - HOLDOUT_N
    assert digest["n_train"] == len(train) and digest["n_holdout"] == len(holdout)

    # Disjoint by row content (unique user-turn per synthetic row).
    train_keys = {_user(r) for r in train}
    holdout_keys = {_user(r) for r in holdout}
    assert train_keys & holdout_keys == set(), "train ∩ holdout must be empty"
    # No duplicates within either split.
    assert len(train_keys) == len(train)
    assert len(holdout_keys) == len(holdout)


def test_split_is_deterministic(tmp_path):
    corpus = _synthetic_corpus(tmp_path / "corpus.jsonl")
    a_train, a_hold = tmp_path / "a_t.jsonl", tmp_path / "a_h.jsonl"
    b_train, b_hold = tmp_path / "b_t.jsonl", tmp_path / "b_h.jsonl"
    da = build_holdout_split(corpus, a_train, a_hold, seed=42)
    db = build_holdout_split(corpus, b_train, b_hold, seed=42)
    # Same seed -> identical partition (sha256 of both splits matches).
    assert da["train_sha256"] == db["train_sha256"]
    assert da["holdout_sha256"] == db["holdout_sha256"]


def test_different_seed_gives_different_holdout(tmp_path):
    corpus = _synthetic_corpus(tmp_path / "corpus.jsonl")
    s42_hold = tmp_path / "s42_h.jsonl"
    s7_hold = tmp_path / "s7_h.jsonl"
    d42 = build_holdout_split(corpus, tmp_path / "s42_t.jsonl", s42_hold, seed=42)
    d7 = build_holdout_split(corpus, tmp_path / "s7_t.jsonl", s7_hold, seed=7)
    assert d42["holdout_sha256"] != d7["holdout_sha256"]
