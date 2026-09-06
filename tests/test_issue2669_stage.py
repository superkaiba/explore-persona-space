"""Protect blinding, missing outcomes, and original grouped folds."""

import pytest

from scripts.issue2669_stage import TARGET, context_record, fold_map, opaque_id, scaled_dv


def test_whitelist_drops_outcomes_and_gold():
    raw = {
        "context_id": "hallu-gold-name",
        "prompt_text": "original prompt",
        "meta": dict(TARGET),
        "group_key": "secret gold",
        "completion": "secret answer",
        "answer_aliases": ["secret gold"],
        "dv": 1,
    }
    record = context_record(raw, "hallucination", "fabrication")
    assert set(record) == {"id", "behavior", "instrument", "prompt_text", "target"}
    assert "secret" not in str(record)
    assert "hallu-gold-name" not in record["id"]
    assert opaque_id("evil", "a") != opaque_id("sycophancy", "a")
    raw["meta"]["temperature"] = 0
    with pytest.raises(ValueError, match="metadata mismatch"):
        context_record(raw, "hallucination", "fabrication")


def test_missing_not_zero_and_scale():
    assert scaled_dv(None, 100) is None
    assert scaled_dv(0, 100) == 0
    assert scaled_dv(0.6, 100) == 60
    with pytest.raises(ValueError):
        scaled_dv(float("nan"), 100)


def test_group_fold_integrity_and_missing_exclusion():
    rows = [
        {"context_id": f"c{i}", "group_key": f"g{i // 2}", "split": "train", "dv": 0}
        for i in range(20)
    ]
    rows.append({"context_id": "missing", "group_key": "unique", "split": "train", "dv": None})
    folds = fold_map(rows, "evil")
    assert len(folds) == 20
    assert set(folds.values()) == set(range(5))
    assert all(folds[f"c{i}"] == folds[f"c{i + 1}"] for i in range(0, 20, 2))
