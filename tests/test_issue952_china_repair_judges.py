"""Synthetic-only tests of the actual v2 preparation and collection paths."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "issue952_repair_judges_test",
    Path(__file__).resolve().parents[1] / "scripts" / "issue952_china_repair_judges.py",
)
assert SPEC is not None and SPEC.loader is not None
JUDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)


def _save_json(path: Path, value) -> None:
    """Write synthetic test fixtures, never production judgments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    """Write explicitly supplied synthetic fixture rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(JUDGE._jsonl_bytes(rows))


def _source(tmp_path: Path, monkeypatch, *, production: bool = False):
    """Create complete real-shape arms/draws over three synthetic source identities."""
    monkeypatch.setattr(JUDGE, "SOURCE_COUNT", 3)
    frozen_ids_hash = JUDGE.sha_object(["s0", "s1", "s2"])
    monkeypatch.setattr(JUDGE, "FROZEN_SOURCE_IDS_SHA256", frozen_ids_hash)
    contents = JUDGE.NEW_CONTENTS if production else JUDGE.OLD_CONTENTS
    rows = []
    for source in range(3):
        for lang in JUDGE.LANGUAGES:
            for content in contents:
                for frame in JUDGE.FRAMES:
                    for draw in range(JUDGE.N_DRAWS):
                        item_id = f"s{source}-{lang}-{content}-{frame}-d{draw}"
                        rows.append(
                            {
                                "item_id": item_id,
                                "source_prompt_id": f"s{source}",
                                "language": lang,
                                "content": content,
                                "frame": frame,
                                "draw": draw,
                                "topic": f"topic{source}",
                                "question": f"Synthetic arithmetic question {source}",
                                "text": f"Synthetic attempted answer for draw {draw}.",
                            }
                        )
    raw = tmp_path / "rollouts.jsonl"
    _save_jsonl(raw, rows)
    source_manifest = tmp_path / "generation.json"
    _save_json(
        source_manifest,
        {
            "rollouts_sha256": JUDGE.sha_file(raw),
            "n_rows": len(rows),
            "regime": {"bank_sha256": "a" * 64, "accepted_source_ids_sha256": frozen_ids_hash},
        },
    )
    return raw, source_manifest, rows


def _prepare(tmp_path, monkeypatch, *, production=False):
    raw, source_manifest, source_rows = _source(tmp_path, monkeypatch, production=production)
    out_dir = tmp_path / "judge"
    manifest = JUDGE.prepare(
        raw, source_manifest, out_dir, ("/root/test_a", "/root/test_b"), production=production
    )
    return out_dir, manifest, source_rows


def _submission(record: dict, *, unassessable: bool = False):
    """Produce explicit synthetic score fixtures for testing the collector only."""
    packet = JUDGE.read_json(Path(record["packet_path"]))
    receipt = {
        "contract": JUDGE.CONTRACT,
        "packet_sha256": record["packet_sha256"],
        "rubric_sha256": JUDGE.RUBRIC_SHA256,
        "runtime_identity_sha256": packet["runtime_identity_sha256"],
        "read_complete": True,
        "items": [
            {
                **{key: item[key] for key in JUDGE.RECEIPT_ITEM_FIELDS if key != "read_complete"},
                "read_complete": True,
            }
            for item in packet["items"]
        ],
    }
    receipt_path = Path(record["receipt_path"])
    _save_json(receipt_path, receipt)
    rows = []
    for index, item in enumerate(packet["items"]):
        score = {
            "opaque_id": item["opaque_id"],
            "request_sha256": item["request_sha256"],
            "packet_sha256": record["packet_sha256"],
            "rubric_sha256": JUDGE.RUBRIC_SHA256,
            "runtime_identity_sha256": packet["runtime_identity_sha256"],
            "input_read_receipt_sha256": JUDGE.sha_file(receipt_path),
            "rationale": "Synthetic fixture: the attempted arithmetic answer addresses "
            f"the requested part without an observable decline; row {index}.",
            "category": "substantive_answer",
            "withholding_score": 0,
            "answer_coverage_score": 20,
            "strict_complete_refusal": False,
            "explicit_decline": False,
            "evasive_redirection": False,
            "expressed_uncertainty": False,
            "unassessable": False,
            "unassessable_reason": None,
        }
        if unassessable:
            score.update({field: None for field in (*JUDGE.SCORES, *JUDGE.BEHAVIOR_FLAGS)})
            score.update(
                {
                    "unassessable": True,
                    "category": "unassessable",
                    "unassessable_reason": "Synthetic response is uninterpretable.",
                }
            )
        rows.append(score)
    _save_jsonl(Path(record["output_path"]), rows)
    return packet, receipt, rows


def test_development_full_blinded_packets_and_deterministic_selection(tmp_path, monkeypatch):
    out_dir, manifest, source_rows = _prepare(tmp_path, monkeypatch)
    assert manifest["n_items"] == 192
    assert manifest["n_assignments"] == 384
    assert manifest["n_overlap"] == 192
    assert set(manifest["arm_counts"].values()) == {16}
    assert len(manifest["packets"]) == 32
    selected = JUDGE.select_development(source_rows)
    assert selected == JUDGE.select_development(list(reversed(source_rows)))
    old_text = {(row["question"], row["text"]) for row in source_rows}
    for record in manifest["packets"]:
        packet = JUDGE.read_json(Path(record["packet_path"]))
        assert len(packet["items"]) == 12
        assert packet["rubric"] == JUDGE.RUBRIC
        for item in packet["items"]:
            assert (item["question"], item["response"]) in old_text
            assert not {"language", "content", "frame", "topic", "item_id"} & set(item)
        assert not Path(record["output_path"]).exists()
        assert not Path(record["receipt_path"]).exists()
    lookup = JUDGE.read_json(Path(manifest["lookup_path"]))
    for row in lookup:
        assert row["lanes"]["agent_a"]["opaque_id"] != row["lanes"]["agent_b"]["opaque_id"]
    raw, source_manifest = tmp_path / "rollouts.jsonl", tmp_path / "generation.json"
    again = JUDGE.prepare(
        raw, source_manifest, out_dir, ("/root/test_a", "/root/test_b"), production=False
    )
    assert again == manifest


def test_source_lineage_and_complete_panel_fail_before_preparation(tmp_path, monkeypatch):
    raw, source_manifest, rows = _source(tmp_path, monkeypatch)
    rows.pop()
    _save_jsonl(raw, rows)
    with pytest.raises(ValueError, match="manifest does not bind"):
        JUDGE.prepare(raw, source_manifest, tmp_path / "bad", ("a", "b"), production=False)
    _save_json(
        source_manifest,
        {
            "rollouts_sha256": JUDGE.sha_file(raw),
            "n_rows": len(rows),
            "regime": {"bank_sha256": "a" * 64},
        },
    )
    with pytest.raises(ValueError, match="factorial"):
        JUDGE.prepare(raw, source_manifest, tmp_path / "bad", ("a", "b"), production=False)
    assert not (tmp_path / "bad" / "manifest.json").exists()


def test_production_assignment_and_full_coverage(tmp_path, monkeypatch):
    out_dir, manifest, _ = _prepare(tmp_path, monkeypatch, production=True)
    assert manifest["n_items"] == 3 * 16 * 8
    lookup = JUDGE.read_json(Path(manifest["lookup_path"]))
    for row in lookup:
        key = JUDGE.sha_object({"contract": JUDGE.CONTRACT, "item_id": row["item_id"]})
        primary, assigned = JUDGE.LEGACY._assignment(key, 0.10)
        assert row["primary_agent"] == primary
        assert row["assigned_agents"] == assigned
    assert 0 < manifest["n_overlap"] < manifest["n_items"]
    for record in manifest["packets"]:
        _submission(record)
    report = JUDGE.collect(out_dir)
    assert report["technical_complete"]
    assert not report["human_validated"]
    assert report["agreement"]["overall"]["n_overlap"] == manifest["n_overlap"]
    scores = JUDGE.read_jsonl(out_dir / "scores.jsonl")
    assert len(scores) == manifest["n_items"]
    # Withholding is the explicit score; low coverage must not be transformed into 80.
    assert {row["withholding_score"] for row in scores} == {0}
    assert {row["answer_coverage_score"] for row in scores} == {20}
    assert report == JUDGE.collect(out_dir)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_field",
        "default_label",
        "bool_score",
        "range",
        "null_score",
        "wrong_rubric",
        "wrong_receipt",
        "wrong_identity",
    ],
)
def test_explicit_decision_parser_rejects_invalid_labels(tmp_path, monkeypatch, mutation):
    _, manifest, _ = _prepare(tmp_path, monkeypatch)
    record = manifest["packets"][0]
    packet, _, rows = _submission(record)
    row = copy.deepcopy(rows[0])
    if mutation == "missing_field":
        del row["explicit_decline"]
    elif mutation == "default_label":
        row["rationale"] = "no"
    elif mutation == "bool_score":
        row["withholding_score"] = False
    elif mutation == "range":
        row["withholding_score"] = 101
    elif mutation == "null_score":
        row["withholding_score"] = None
    elif mutation == "wrong_rubric":
        row["rubric_sha256"] = "b" * 64
    elif mutation == "wrong_receipt":
        row["input_read_receipt_sha256"] = "b" * 64
    else:
        row["runtime_identity_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        JUDGE.validate_decision(
            row, packet["items"][0], packet, record, JUDGE.sha_file(Path(record["receipt_path"]))
        )


@pytest.mark.parametrize("mutation", ["short_read", "omitted_row", "wrong_bytes", "numeric_true"])
def test_full_read_receipts_reject_partial_or_stale_input(tmp_path, monkeypatch, mutation):
    _, manifest, _ = _prepare(tmp_path, monkeypatch)
    record = manifest["packets"][0]
    packet, receipt, _ = _submission(record)
    if mutation == "short_read":
        receipt["items"][0]["read_complete"] = False
    elif mutation == "omitted_row":
        receipt["items"].pop()
    elif mutation == "wrong_bytes":
        receipt["items"][0]["response_utf8_bytes"] -= 1
    else:
        receipt["read_complete"] = 1
    with pytest.raises(ValueError, match="input-read receipt"):
        JUDGE._validate_receipt(receipt, packet, record)


def test_collect_rejects_missing_output_before_writing_scores(tmp_path, monkeypatch):
    out_dir, manifest, _ = _prepare(tmp_path, monkeypatch)
    _submission(manifest["packets"][0])
    with pytest.raises(FileNotFoundError):
        JUDGE.collect(out_dir)
    assert not (out_dir / "scores.jsonl").exists()


def test_packet_input_tampering_fails_before_collection(tmp_path, monkeypatch):
    out_dir, manifest, _ = _prepare(tmp_path, monkeypatch)
    record = manifest["packets"][0]
    packet = JUDGE.read_json(Path(record["packet_path"]))
    packet["items"][0]["response"] += " changed"
    _save_json(Path(record["packet_path"]), packet)
    with pytest.raises(ValueError, match="packet bytes changed"):
        JUDGE.collect(out_dir)


def test_unassessable_scores_remain_missing_and_saturation_undefined(tmp_path, monkeypatch):
    out_dir, manifest, _ = _prepare(tmp_path, monkeypatch)
    for record in manifest["packets"]:
        _submission(record, unassessable=True)
    report = JUDGE.collect(out_dir)
    overall = report["agreement"]["overall"]
    assert overall["graded"]["withholding_score"]["n"] == 0
    assert overall["graded"]["withholding_score"]["spearman"] is None
    assert overall["binary"]["strict_complete_refusal"]["n"] == 0
    assert overall["binary"]["unassessable"]["cohen_kappa"] is None
    assert all(row["n_assessable"] == 0 for row in report["by_arm"].values())
    assert all(
        row["withholding_score"] is None for row in JUDGE.read_jsonl(out_dir / "scores.jsonl")
    )


def test_rare_class_agreement_reports_prevalence_and_uncertainty():
    report = JUDGE._binary_agreement([False, False, True, True], [False, True, False, True])
    assert report["confusion_rows_a_columns_b_false_true"] == [[1, 1], [1, 1]]
    assert report["positive_specific_agreement"] == 0.5
    assert report["negative_specific_agreement"] == 0.5
    assert report["cohen_kappa"] == 0
    assert report["prevalence_exact_95_ci"]["agent_a"][0] < 0.5
    assert report["prevalence_exact_95_ci"]["agent_a"][1] > 0.5


def test_duplicate_json_keys_are_not_silently_accepted(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"withholding_score":0,"withholding_score":100}\n')
    with pytest.raises(ValueError, match="duplicate JSON field"):
        JUDGE.read_jsonl(path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order", "positive_id_list"])
def test_collector_requires_explicit_full_ordered_rows(tmp_path, monkeypatch, mutation):
    out_dir, manifest, _ = _prepare(tmp_path, monkeypatch)
    record = manifest["packets"][0]
    _, _, rows = _submission(record)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "order":
        rows.reverse()
    else:
        rows = [{"positive_ids": []}]
    _save_jsonl(Path(record["output_path"]), rows)
    with pytest.raises(ValueError, match="explicitly cover"):
        JUDGE.collect(out_dir)
    assert not (out_dir / "scores.jsonl").exists()


def test_sanity_controls_share_frozen_rubric_but_are_separate(tmp_path):
    out_dir = tmp_path / "sanity"
    manifest = JUDGE.prepare_sanity(out_dir, ("/root/sanity_a", "/root/sanity_b"))
    assert manifest["phase"] == "sanity"
    assert manifest["n_items"] == 6
    assert manifest["n_assignments"] == 12
    assert manifest["rubric_sha256"] == JUDGE.RUBRIC_SHA256
    for record in manifest["packets"]:
        packet, _, _ = _submission(record)
        assert packet["rubric"] == JUDGE.RUBRIC
    report = JUDGE.collect(out_dir)
    assert report["technical_complete"]
    assert report["sanity_checks"]["n_checks"] == 6 * 2 * 7
    assert report["sanity_checks"]["n_mismatches"] > 0
    assert not report["human_validated"]


def test_manifest_cannot_hide_a_missing_item_with_revised_counts(tmp_path, monkeypatch):
    _, manifest, _ = _prepare(tmp_path, monkeypatch)
    lookup_path = Path(manifest["lookup_path"])
    lookup = JUDGE.read_json(lookup_path)
    removed = lookup.pop()
    _save_json(lookup_path, lookup)
    manifest["lookup_sha256"] = JUDGE.sha_file(lookup_path)
    manifest["n_items"] -= 1
    manifest["n_assignments"] -= 2
    manifest["n_overlap"] -= 1
    manifest["arm_counts"][":".join(JUDGE._arm(removed))] -= 1
    with pytest.raises(ValueError, match="frozen per-arm/source coverage"):
        JUDGE._validate_manifest(manifest)


def test_preparation_rejects_substituted_source_identities(tmp_path, monkeypatch):
    raw, source_manifest, _ = _source(tmp_path, monkeypatch)
    source = JUDGE.read_json(source_manifest)
    source["regime"]["accepted_source_ids_sha256"] = "b" * 64
    _save_json(source_manifest, source)
    with pytest.raises(ValueError, match="frozen 85 accepted"):
        JUDGE.prepare(raw, source_manifest, tmp_path / "bad", ("a", "b"), production=False)


def test_immutable_artifacts_resume_identical_and_reject_replacement(tmp_path):
    path = tmp_path / "immutable.json"
    JUDGE.write_immutable(path, b"first\n")
    JUDGE.write_immutable(path, b"first\n")
    with pytest.raises(ValueError, match="immutable artifact differs"):
        JUDGE.write_immutable(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_runtime_identity_does_not_invent_inherited_settings():
    identity = JUDGE._identity("/root/test_a")
    for field in ("model", "reasoning_effort", "service_tier"):
        assert identity[field] is None
        assert field in identity["unavailable"]
    assert identity["fork_turns"] == "none"
    assert "not exposed" in identity["runtime_note"]


def test_identity_mapping_only_references_authored_decisions(tmp_path, monkeypatch):
    raw, source_manifest, _ = _source(tmp_path, monkeypatch)
    real_identity = JUDGE._identity

    def old_identity(agent_id):
        value = real_identity(agent_id)
        value.update(
            {
                "model": "previous-unverified-model",
                "reasoning_effort": "medium",
                "service_tier": "priority",
            }
        )
        return value

    old_dir, new_dir = tmp_path / "old", tmp_path / "corrected"
    with monkeypatch.context() as patch:
        patch.setattr(JUDGE, "_identity", old_identity)
        old = JUDGE.prepare(raw, source_manifest, old_dir, ("a", "b"), production=False)
    old_record = next(record for record in old["packets"] if record["lane"] == "agent_a")
    _submission(old_record)
    JUDGE.prepare(raw, source_manifest, new_dir, ("a", "b"), production=False)
    result = JUDGE.map_identity_repair(old_dir, new_dir, tmp_path / "mappings", "agent_a", [0])
    assert result["n_items"] == 12
    assert result["requires_original_author_attestation"]
    assert not result["writes_scores_or_receipts"]
    for row in result["mappings"]:
        assert row["old_opaque_id"] != row["new_opaque_id"]
        assert not {
            "rationale",
            "withholding_score",
            "answer_coverage_score",
            "original_item_id",
            "content",
            "language",
            "frame",
        } & set(row)
    assert not list((new_dir / "packets").rglob("*.output.jsonl"))
    assert not list((new_dir / "packets").rglob("*.read_receipt.json"))
    assert (tmp_path / "mappings" / "agent_a.json").exists()
    assert not (tmp_path / "mappings" / "agent_b.json").exists()
