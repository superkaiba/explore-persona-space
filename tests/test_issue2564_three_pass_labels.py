"""Selection/provenance tests use synthetic fixtures, never experimental labels."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2564_codex_judgments as original
import issue2564_three_pass_labels as m


def fixture_units():
    rows = [{"id": f"r{i}"} for i in range(4)]
    units = []
    for row_i, row in enumerate(rows):
        for prop, spec in m.PROPERTIES.items():
            for draw in range(5):
                value = {"reason": "Explicit fixture for aggregation and provenance tests."}
                if spec["kind"] == "graded":
                    value.update(assessable=True, score=row_i * 10 + draw)
                else:
                    value["label"] = spec["labels"][draw % len(spec["labels"])]
                    if prop == "persona":
                        value["multi_voice"] = bool(draw % 2)
                units.append(
                    {
                        "row_id": row["id"],
                        "property": prop,
                        "draw": draw,
                        "parsed": value,
                        "drop": None,
                        "key": f"{row_i}-{prop}-{draw}",
                        "agent_id": f"{prop}-{draw}",
                        "packet_id": f"packet-{prop}-{draw}",
                    }
                )
    return rows, units


def test_three_pass_means_votes_missingness_and_reliability():
    rows, units = fixture_units()
    target = next(
        u for u in units if u["row_id"] == "r0" and u["property"] == "warmth" and u["draw"] == 1
    )
    target["parsed"].update(assessable=False, score=None)
    selected = m.select_units(rows, units)
    labels, quality = original.summarize_units(rows, units, m.SELECTED_REPETITIONS)
    assert len(selected) == 4 * 7 * 3
    warmth = labels[0]["properties"]["warmth"]
    assert (warmth["n_valid"], warmth["n_assessable"], warmth["mean"]) == (3, 2, 1.0)
    persona = labels[0]["properties"]["persona"]
    assert persona["modal"] is None
    assert sorted(v for v in persona["votes"].values() if v) == [1 / 3] * 3
    assert persona["multi_voice_fraction"] == pytest.approx(1 / 3)
    assert quality["warmth"]["n_all_three_assessable"] == 3
    assert "n_all_five_assessable" not in quality["warmth"]
    matrix = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
    expected = 3 / 2 * (1 - matrix.var(axis=0, ddof=1).sum() / matrix.sum(1).var(ddof=1))
    assert quality["warmth"]["descriptive_cronbach_alpha"] == pytest.approx(expected)
    default_labels, default_quality = original.summarize_units(rows, units)
    assert default_labels[0]["properties"]["persona"]["n_valid"] == 5
    assert default_quality["warmth"]["n_all_five_assessable"] == 3


@pytest.mark.parametrize("damage", ["missing", "duplicate", "same_agent", "invalid"])
def test_selected_roster_rejects_bad_evidence(damage):
    rows, units = fixture_units()
    selected = m.select_units(rows, units)
    if damage == "missing":
        selected.pop()
    elif damage == "duplicate":
        selected.append(copy.deepcopy(selected[0]))
    elif damage == "same_agent":
        selected[1]["agent_id"] = selected[0]["agent_id"]
    else:
        selected[0]["drop"] = "invalid_score"
    with pytest.raises(ValueError):
        m.select_units(rows, selected)


@pytest.fixture
def source(tmp_path, monkeypatch):
    rows, units = fixture_units()
    acceptance = {"fixture": "accepted original pilot"}
    raw = {
        "raw_records_hash": "raw-fixture-hash",
        "packet_files": [],
        "expected_draws": 4 * 7 * 5,
        "exact_keyset_complete": True,
    }
    for name in ("prepared/rows.jsonl", "annotation_codex/main/config.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    calls = []

    def validate_pilot(root, config):
        calls.append(("pilot", root, config))
        return acceptance

    def collect(root, part):
        calls.append(("collect", root, part))
        return rows, units, raw

    monkeypatch.setattr(original, "validate_codex_pilot", validate_pilot)
    monkeypatch.setattr(original, "collect", collect)
    return tmp_path, raw, acceptance, calls


def test_export_and_replay_keep_source_and_provider_separate(source):
    root, _, _, calls = source
    before = (root / "annotation_codex/main/config.json").read_bytes()
    exported = m.export_three_pass_main(root)
    assert exported == m.validate_three_pass_main(root)
    assert exported["config"]["draws"] == 3
    assert exported["complete"]["expected_annotations"] == 84
    assert original.RECIPE["draws"] == 5
    assert (root / "annotation_codex/main/config.json").read_bytes() == before
    assert ("collect", root, "main") in calls
    assert any(c[0] == "pilot" and c[2] == original.RECIPE for c in calls)
    with pytest.raises(FileExistsError):
        m.export_three_pass_main(root)


@pytest.mark.parametrize(
    "part", ["labels", "quality", "complete", "labels_manifest", "raw", "pilot"]
)
def test_validation_rejects_changed_derived_or_original_evidence(source, part):
    root, raw, acceptance, _ = source
    m.export_three_pass_main(root)
    if part == "raw":
        raw["raw_records_hash"] = "changed"
    elif part == "pilot":
        acceptance["fixture"] = "different pilot"
    else:
        path = m.output_dir(root) / f"{part}.json"
        value = json.loads(path.read_text())
        if part == "labels":
            value[0]["properties"]["warmth"]["mean"] = 90
        elif part == "quality":
            value["properties"]["warmth"]["mean"] = 90
        elif part == "complete":
            value["expected_annotations"] = 140
        else:
            value["expected_draws"] = 140
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        m.validate_three_pass_main(root)
