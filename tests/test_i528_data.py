"""Unit tests for ``explore_persona_space.experiments.i528_data``.

Covers per-trait Q-bank loaders + the ``assert_q_test_equality`` invariant
that defends against the #517 disjoint-Q-bank regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_split(dirp: Path, split: str, questions: list[str]) -> None:
    payload = {
        "schema_version": "i528_qbank_v1",
        "trait": dirp.name,
        "split": split,
        "n": len(questions),
        "questions": questions,
    }
    (dirp / f"Q_{split}.json").write_text(json.dumps(payload, indent=2))


def test_load_q_train_and_test_round_trip(tmp_path, monkeypatch):
    from explore_persona_space.experiments import i528_data

    trait = "validating"
    trait_dir = tmp_path / trait
    trait_dir.mkdir()
    train_qs = [f"train-{i}" for i in range(5)]
    test_qs = [f"test-{i}" for i in range(3)]
    _write_split(trait_dir, "train", train_qs)
    _write_split(trait_dir, "test", test_qs)

    monkeypatch.setattr(i528_data, "LOCAL_DATA_DIR", tmp_path)
    assert i528_data.load_q_train(trait) == train_qs
    assert i528_data.load_q_test(trait) == test_qs


def test_load_raises_on_unknown_trait():
    from explore_persona_space.experiments import i528_data

    with pytest.raises(ValueError, match="Unknown trait"):
        i528_data.load_q_train("not_a_trait")


def test_assert_disjoint_passes_when_disjoint(tmp_path, monkeypatch):
    from explore_persona_space.experiments import i528_data

    trait = "validating"
    trait_dir = tmp_path / trait
    trait_dir.mkdir()
    _write_split(trait_dir, "train", ["a", "b"])
    _write_split(trait_dir, "test", ["c", "d"])
    monkeypatch.setattr(i528_data, "LOCAL_DATA_DIR", tmp_path)
    i528_data.assert_disjoint(trait)


def test_assert_disjoint_raises_when_overlap(tmp_path, monkeypatch):
    from explore_persona_space.experiments import i528_data

    trait = "validating"
    trait_dir = tmp_path / trait
    trait_dir.mkdir()
    _write_split(trait_dir, "train", ["a", "b", "c"])
    _write_split(trait_dir, "test", ["c", "d"])  # 'c' overlaps
    monkeypatch.setattr(i528_data, "LOCAL_DATA_DIR", tmp_path)
    with pytest.raises(AssertionError, match="overlap"):
        i528_data.assert_disjoint(trait)


def test_assert_q_test_equality_passes_on_exact_match(tmp_path, monkeypatch):
    from explore_persona_space.experiments import i528_data

    trait = "validating"
    trait_dir = tmp_path / trait
    trait_dir.mkdir()
    test_qs = ["q one", "q two", "q three"]
    _write_split(trait_dir, "test", test_qs)
    monkeypatch.setattr(i528_data, "LOCAL_DATA_DIR", tmp_path)
    i528_data.assert_q_test_equality(trait, test_qs)


def test_assert_q_test_equality_raises_on_drift(tmp_path, monkeypatch):
    """The #517 regression: trained eval used a different (regenerated) bank
    than the base eval — paired Δ becomes meaningless."""
    from explore_persona_space.experiments import i528_data

    trait = "validating"
    trait_dir = tmp_path / trait
    trait_dir.mkdir()
    test_qs = ["q one", "q two", "q three"]
    _write_split(trait_dir, "test", test_qs)
    monkeypatch.setattr(i528_data, "LOCAL_DATA_DIR", tmp_path)
    drifted = ["q one", "q DIFFERENT two", "q three"]
    with pytest.raises(AssertionError, match="prompt-text mismatch"):
        i528_data.assert_q_test_equality(trait, drifted)


def test_assert_q_test_equality_raises_on_length_mismatch(tmp_path, monkeypatch):
    from explore_persona_space.experiments import i528_data

    trait = "validating"
    trait_dir = tmp_path / trait
    trait_dir.mkdir()
    test_qs = ["q one", "q two", "q three"]
    _write_split(trait_dir, "test", test_qs)
    monkeypatch.setattr(i528_data, "LOCAL_DATA_DIR", tmp_path)
    with pytest.raises(AssertionError, match="observed"):
        i528_data.assert_q_test_equality(trait, ["q one"])


def test_load_q_train_missing_file_raises_actionable_error(tmp_path, monkeypatch):
    from explore_persona_space.experiments import i528_data

    monkeypatch.setattr(i528_data, "LOCAL_DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="phase0_preflight"):
        i528_data.load_q_train("validating")


def test_q_test_sha256_is_stable_for_same_content(tmp_path, monkeypatch):
    from explore_persona_space.experiments import i528_data

    trait = "validating"
    trait_dir = tmp_path / trait
    trait_dir.mkdir()
    _write_split(trait_dir, "test", ["a", "b", "c"])
    monkeypatch.setattr(i528_data, "LOCAL_DATA_DIR", tmp_path)
    h1 = i528_data.q_test_sha256(trait)
    h2 = i528_data.q_test_sha256(trait)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest
