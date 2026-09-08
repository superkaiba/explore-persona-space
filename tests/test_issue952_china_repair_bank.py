"""Mechanical integrity checks for the explicit-country-cue bank repair."""

import copy
import hashlib

import pytest

from scripts import issue952_china_repair_bank as bank


def _source_rows():
    return [
        {
            "source_prompt_id": "fixture-1",
            "language": lang,
            "frame": frame,
            "content": content,
            "prompt": f"Synthetic parser fixture {lang} {frame} {content}?",
        }
        for lang in ("en", "zh")
        for frame in ("direct", "academic")
        for content in bank.CONTENTS[:3]
    ]


def _rendered():
    metadata = {"fixture-1": {"control_country_en": "Italy", "control_country_zh": "意大利"}}
    return bank.render(_source_rows(), metadata)


def test_complete_four_cell_rendering_preserves_exact_base():
    rows = _rendered()
    assert len(rows) == 16
    assert len({r["item_id"] for r in rows}) == 16
    result = bank.validate_pairs(rows, encode=lambda text: list(text.encode()))
    assert result["n_pairs"] == 8
    assert result["n_identical_text_pairs"] == 0
    assert result["token_checks_run"]
    assert min(result["added_token_lengths"]) > 0
    for row in rows:
        base = row["prompt"].removeprefix(row["cue_text"])
        assert hashlib.sha256(base.encode()).hexdigest() == row["base_prompt_sha256"]


def test_identical_token_inputs_rejected_even_if_text_changes():
    with pytest.raises(ValueError, match="tokenizes to identical"):
        bank.validate_pairs(_rendered(), encode=lambda _text: [1, 2])


@pytest.mark.parametrize("defect", ["duplicate", "missing", "unchanged", "wrong_base"])
def test_bad_pair_shapes_rejected(defect):
    rows = _rendered()
    if defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "missing":
        rows.pop()
    else:
        added = next(r for r in rows if r["country_cue"] == "present")
        if defect == "unchanged":
            added["prompt"] = added["prompt"].removeprefix(added["cue_text"])
        else:
            added["prompt"] += " changed question"
    with pytest.raises(ValueError):
        bank.validate_pairs(rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "unknown"),
        ("language", "fr"),
        ("frame", "unknown"),
        ("subject_arm", "wrong"),
        ("country_cue", "wrong"),
        ("base_prompt_sha256", "wrong"),
    ],
)
def test_mislabelled_factorial_metadata_rejected(field, value):
    rows = _rendered()
    rows[0][field] = value
    with pytest.raises(ValueError):
        bank.validate_pairs(rows)


def test_empty_bank_rejected():
    with pytest.raises(ValueError, match="empty"):
        bank.validate_pairs([])


def test_missing_metadata_is_not_defaulted(tmp_path):
    path = tmp_path / "metadata.jsonl"
    bank.write_rows(path, [])
    with pytest.raises(ValueError, match="every frozen source"):
        bank.metadata_rows(path, {"fixture-1"})


def test_stale_source_audit_rejected(tmp_path):
    source = tmp_path / "bank.jsonl"
    audit = tmp_path / "audit.json"
    bank.write_rows(source, _source_rows())
    bank.write_json(audit, {"passed": True, "prompt_bank_sha256": "stale"})
    with pytest.raises(ValueError, match="matching passed audit"):
        bank.source_panel(source, audit)


def test_stale_candidate_rejected_before_audit_loading(tmp_path):
    candidate = tmp_path / "prompt_bank.candidate.jsonl"
    metadata = tmp_path / "metadata.jsonl"
    bank.write_rows(candidate, _rendered())
    bank.write_rows(metadata, [])
    bank.write_json(
        tmp_path / "build_manifest.json",
        {
            "metadata_sha256": bank.sha(metadata),
            "candidate_sha256": "stale",
        },
    )
    with pytest.raises(ValueError, match="changed after audit"):
        bank.finalize(tmp_path, metadata, tmp_path / "not_yet_read.jsonl")
