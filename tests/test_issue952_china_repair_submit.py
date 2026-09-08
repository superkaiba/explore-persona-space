"""Synthetic-only tests of metadata serialization, not production scoring."""

import json
from pathlib import Path

import pytest

from scripts import issue952_china_repair_judges as judges
from scripts import issue952_china_repair_submit as submitter


@pytest.fixture
def prepared(tmp_path):
    manifest = judges.prepare_sanity(tmp_path / "judge", ("test_a", "test_b"))
    record = manifest["packets"][0]
    packet = judges.read_json(Path(record["packet_path"]))
    rows = [
        {
            "opaque_id": item["opaque_id"],
            "rationale": f"Synthetic serialization fixture number {i}; not a scored case.",
            "category": "unassessable",
            "withholding_score": None,
            "answer_coverage_score": None,
            "strict_complete_refusal": None,
            "explicit_decline": None,
            "evasive_redirection": None,
            "expressed_uncertainty": None,
            "unassessable": True,
            "unassessable_reason": "Explicit synthetic parser fixture.",
        }
        for i, item in enumerate(packet["items"])
    ]
    authored = Path(record["packet_path"]).with_suffix(".authored.json")
    authored.write_text(json.dumps(rows))
    return record, authored, rows


def test_submit_exact_explicit_fields_and_idempotent_resume(prepared):
    record, authored, rows = prepared
    args = (Path(record["packet_path"]), authored, "test_a")
    assert submitter.submit(*args, full_read=True) == len(rows)
    assert submitter.submit(*args, full_read=True) == len(rows)
    output = judges.read_jsonl(Path(record["output_path"]))
    assert [{key: row[key] for key in submitter.AUTHORED_FIELDS} for row in output] == rows


@pytest.mark.parametrize("mutation", ["missing", "reordered", "bad_score", "blank_rationale"])
def test_submit_rejects_incomplete_or_invalid_without_output(prepared, mutation):
    record, authored, rows = prepared
    if mutation == "missing":
        rows[0].pop("explicit_decline")
    elif mutation == "reordered":
        rows.reverse()
    elif mutation == "bad_score":
        rows[0]["withholding_score"] = 0
    else:
        rows[0]["rationale"] = ""
    authored.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        submitter.submit(Path(record["packet_path"]), authored, "test_a", full_read=True)
    assert not Path(record["receipt_path"]).exists()
    assert not Path(record["output_path"]).exists()


def test_submit_requires_real_assignment_and_explicit_attestation(prepared):
    record, authored, _ = prepared
    packet_path = Path(record["packet_path"])
    with pytest.raises(ValueError, match="assigned reviewer"):
        submitter.submit(packet_path, authored, "wrong_agent", full_read=True)
    with pytest.raises(ValueError, match="attest"):
        submitter.submit(packet_path, authored, "test_a", full_read=False)


def test_submit_rejects_changed_full_input(prepared):
    record, authored, _ = prepared
    packet_path = Path(record["packet_path"])
    packet = judges.read_json(packet_path)
    packet["items"][0]["response"] += " extra bytes"
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="full input bytes"):
        submitter.submit(packet_path, authored, "test_a", full_read=True)


def test_submit_rejects_foreign_output_path(prepared, tmp_path):
    record, authored, _ = prepared
    packet_path = Path(record["packet_path"])
    packet = judges.read_json(packet_path)
    packet["paths"]["output_path"] = str(tmp_path / "foreign.jsonl")
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="own sibling"):
        submitter.submit(packet_path, authored, "test_a", full_read=True)
