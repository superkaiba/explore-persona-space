from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_script("issue2094_natural_corrected")
blind = load_script("issue2094_natural_blind_annotate")
analysis = load_script("issue2094_natural_analyze")


def test_repository_revision_resolution() -> None:
    class FakeApi:
        def model_info(self, model_id: str, revision: str):
            assert model_id == runner.MODEL_ID
            assert revision == runner.MODEL_REVISION
            return type("Info", (), {"sha": runner.MODEL_REVISION})()

    assert runner.resolve_repository_revision(FakeApi()) == runner.MODEL_REVISION

    class WrongApi:
        def model_info(self, model_id: str, revision: str):
            return type("Info", (), {"sha": "wrong"})()

    with pytest.raises(RuntimeError, match="repository revision mismatch"):
        runner.resolve_repository_revision(WrongApi())


def test_prompt_bank_and_generation_census() -> None:
    bank = runner.prompt_bank()
    rows = runner.planned_rows()
    assert len(bank) == 15
    assert all(prompt.text.startswith("Hello. ") for prompt in bank)
    assert len(rows) == 129
    assert len({row["gen_id"] for row in rows}) == 129

    expected = {
        ("unpatched", "anchor", "none"): 15,
        ("self_patch", "self", "L19"): 15,
        ("self_patch", "self", "all28"): 15,
        ("donor_patch", "same_subject_different_task", "L19"): 18,
        ("donor_patch", "same_subject_different_task", "all28"): 18,
        ("donor_patch", "same_task_different_subject", "L19"): 18,
        ("donor_patch", "same_task_different_subject", "all28"): 18,
        ("donor_patch", "positive_format_control", "L19"): 6,
        ("donor_patch", "positive_format_control", "all28"): 6,
    }
    observed = {
        key: sum((row["arm"], row["axis"], row["layer_setting"]) == key for row in rows)
        for key in expected
    }
    assert observed == expected


def test_every_donor_swap_changes_exactly_one_axis() -> None:
    for row in runner.planned_rows():
        if row["axis"] == "same_subject_different_task":
            assert row["recipient_subject"] == row["donor_subject"]
            assert row["recipient_task"] != row["donor_task"]
        elif row["axis"] == "same_task_different_subject":
            assert row["recipient_task"] == row["donor_task"]
            assert row["recipient_subject"] != row["donor_subject"]
        elif row["axis"] == "positive_format_control":
            assert row["recipient_task"] == row["donor_task"] == "explanation"
            assert row["recipient_subject"] == row["donor_subject"]
            assert row["recipient_format"] != row["donor_format"]


def test_smoke_census_exercises_all_paths() -> None:
    rows = runner.rows_for_smoke(runner.planned_rows())
    assert len(rows) == 9
    assert {(row["axis"], row["layer_setting"]) for row in rows} >= {
        ("anchor", "none"),
        ("self", "L19"),
        ("self", "all28"),
        ("same_subject_different_task", "L19"),
        ("same_subject_different_task", "all28"),
        ("same_task_different_subject", "L19"),
        ("same_task_different_subject", "all28"),
        ("positive_format_control", "L19"),
        ("positive_format_control", "all28"),
    }


def test_generation_has_no_forced_opening() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "LogitsProcessorList" not in source
    assert "ForcedOpeningProcessor" not in source
    assert '"forced_opening": None' in source
    assert '"forced_opening_token_ids": []' in source


def test_termination() -> None:
    assert runner.classify_termination([7, 8, 9, 8], {8}) == ("eos", [7, 8])
    assert runner.classify_termination([7, 9], {8}) == ("length", [7, 9])


def annotation(row_id: str = "R0001") -> dict:
    return {
        "row_id": row_id,
        "form": "itinerary",
        "subject": "vancouver",
        "format": "bullets",
        "complete": True,
        "coherence": 95,
        "evidence": "Three days and Vancouver restaurant suggestions.",
    }


def test_blind_parser_round_trip_plain_and_fenced() -> None:
    raw = json.dumps([annotation()])
    assert blind.parse_annotations(raw, ["R0001"])[0]["coherence"] == 95
    assert blind.parse_annotations(f"```json\n{raw}\n```", ["R0001"])[0]["form"] == "itinerary"
    with pytest.raises(ValueError, match="out of order"):
        blind.parse_annotations(raw, ["R0002"])
    bad = annotation()
    bad["coherence"] = 101
    with pytest.raises(ValueError, match="outside"):
        blind.parse_annotations(json.dumps([bad]), ["R0001"])


def test_blind_packet_scan_is_scope_aware() -> None:
    clean = blind.build_segments([("R0001", "Response:\nA short itinerary for Rome.")])
    assert blind.scan_for_leakage(clean) == {"wrapper": [], "payload": []}
    wrapper_hit = [("wrapper", "This donor passage"), ("payload", "ordinary text")]
    assert blind.scan_for_leakage(wrapper_hit)["wrapper"] == ["donor"]
    payload_hit = [("wrapper", "neutral"), ("payload", "recipient_prompt_id")]
    assert blind.scan_for_leakage(payload_hit)["payload"] == ["recipient_prompt_id"]


def test_codex_event_parser_proves_no_tool_use() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "opaque"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "[]"},
                }
            ),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7}}),
        ]
    )
    events, usage = blind.parse_codex_events(raw)
    assert len(events) == 4
    assert usage == {"input_tokens": 7}

    tool_raw = raw.replace('"type": "agent_message"', '"type": "command_execution"')
    with pytest.raises(RuntimeError, match="used a tool"):
        blind.parse_codex_events(tool_raw)


def test_frozen_key_is_stable_and_census_checked(tmp_path: Path) -> None:
    generations = [{"gen_id": "a"}, {"gen_id": "b"}]
    key_path = tmp_path / "blind_key.json"
    first = blind.freeze_key(key_path, generations)
    second = blind.freeze_key(key_path, list(reversed(generations)))
    assert first == second
    with pytest.raises(ValueError, match="does not match"):
        blind.freeze_key(key_path, [{"gen_id": "c"}])


def test_structural_format_companion() -> None:
    assert analysis.structural_format("Response:\n- a\n- b\n- c\n- d\n- e") == "bullets"
    assert analysis.structural_format("Response:\nOne continuous paragraph.") == "paragraph"
    assert analysis.structural_format("Response:\nHeading\n\nParagraph") == "neither_or_mixed"


def fake_joined_rows() -> list[dict]:
    rows = []
    for row in runner.planned_rows():
        form = row["recipient_task"]
        subject = row["recipient_subject"]
        output_format = row["recipient_format"] or "neither_or_mixed"
        # Make the preregistered all-layer format manipulation check pass.
        if row["axis"] == "positive_format_control" and row["layer_setting"] == "all28":
            output_format = row["donor_format"]
        rows.append(
            {
                **row,
                "output_text": "Response:\nA complete paragraph.",
                "termination_reason": "eos",
                "injection_telemetry": (
                    None if row["layer_setting"] == "none" else {"max_abs_source_error": 0.0}
                ),
                "blind_annotation": {
                    "row_id": f"R{len(rows) + 1:04d}",
                    "form": form,
                    "subject": subject,
                    "format": output_format,
                    "complete": True,
                    "coherence": 100,
                    "evidence": "Complete expected structure and subject.",
                },
                "structural_format": output_format,
            }
        )
    return rows


def test_summary_denominators_and_control_gates() -> None:
    rows = fake_joined_rows()
    summary = analysis.summarize(rows)
    assert summary["realized_n_rows"] == 129
    assert summary["positive_format_gate_pass"] is True
    assert summary["self_patch_gate_pass"] is True
    assert summary["settings"]["L19"]["task_swap"]["n"] == 18
    assert summary["settings"]["L19"]["subject_swap"]["n"] == 18
    assert summary["settings"]["all28"]["positive_format_control"]["n"] == 6
    assert summary["format_gate_by_setting"] == {"L19": False, "all28": True}
    assert summary["primary_verdict"] == "no_selective_layer19_task_transfer"


def test_positive_verdict_requires_specificity_and_coherence() -> None:
    specificity_rows = fake_joined_rows()
    primary_task_rows = [
        row
        for row in specificity_rows
        if row["axis"] == "same_subject_different_task" and row["layer_setting"] == "L19"
    ]
    for row in primary_task_rows:
        row["blind_annotation"]["form"] = row["donor_task"]
    primary_task_rows[0]["blind_annotation"]["subject"] = "other_or_mixed"
    summary = analysis.summarize(specificity_rows)
    assert summary["primary_specificity_gate_pass"] is False
    assert summary["primary_verdict"] == "inconclusive_specificity_failed"

    coherence_rows = fake_joined_rows()
    primary_task_rows = [
        row
        for row in coherence_rows
        if row["axis"] == "same_subject_different_task" and row["layer_setting"] == "L19"
    ]
    for row in primary_task_rows[:4]:
        row["blind_annotation"]["coherence"] = 59
    summary = analysis.summarize(coherence_rows)
    assert summary["primary_coherence_gate_pass"] is False
    assert summary["primary_verdict"] == "inconclusive_coherence_floor_failed"


def test_validate_join_requires_all_129_annotations() -> None:
    generations = []
    annotations = []
    for index, planned in enumerate(runner.planned_rows(), 1):
        generations.append(
            {
                **planned,
                "termination_reason": "eos",
                "output_text": "Response:\nComplete.",
            }
        )
        annotations.append({**annotation(f"R{index:04d}"), "gen_id": planned["gen_id"]})
    assert len(analysis.validate_and_join(generations, annotations)) == 129
    with pytest.raises(ValueError, match="ID sets differ"):
        analysis.validate_and_join(generations, annotations[:-1])


def test_annotation_audit_requires_clean_complete_sidecars(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packets"
    packet_dir.mkdir()
    annotations = [
        {**annotation("R0001"), "gen_id": "g1"},
        {**annotation("R0002"), "gen_id": "g2"},
    ]
    generations = [
        {"gen_id": "g1", "output_text": "Response:\nA Rome itinerary."},
        {"gen_id": "g2", "output_text": "Response:\nA Japan briefing."},
    ]
    (tmp_path / "DONE.json").write_text(
        json.dumps(
            {
                "backend": "codex-cli",
                "model": "gpt-6-astra",
                "n_rows": 2,
                "n_packets": 1,
                "all_rows_annotated": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "blind_key.json").write_text(
        json.dumps({"row_id_to_gen_id": {"R0001": "g1", "R0002": "g2"}}),
        encoding="utf-8",
    )
    items = [("R0001", generations[0]["output_text"]), ("R0002", generations[1]["output_text"])]
    outbound = "".join(text for _scope, text in blind.build_segments(items))
    parsed = [{key: value for key, value in row.items() if key != "gen_id"} for row in annotations]
    raw_text = json.dumps(parsed)
    event_jsonl = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "opaque"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": raw_text},
                }
            ),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7}}),
        ]
    )
    request = {
        "row_ids": ["R0001", "R0002"],
        "leakage_scan_scopes": {"wrapper": {"hits": []}, "payload": {"hits": []}},
        "tool_item_types_observed": [],
        "protocol_deviation": "judge changed",
        "outbound_request_verbatim": outbound,
    }
    (packet_dir / "packet_000.request.json").write_text(json.dumps(request), encoding="utf-8")
    (packet_dir / "packet_000.response.json").write_text(
        json.dumps({"raw_text": raw_text, "provider_event_jsonl": event_jsonl}),
        encoding="utf-8",
    )
    (packet_dir / "packet_000.parsed.json").write_text(json.dumps(parsed), encoding="utf-8")
    audit = analysis.validate_annotation_audit(
        tmp_path / "annotations.jsonl", annotations, generations
    )
    assert audit["all_leakage_scans_clean"] is True
    assert audit["all_payloads_replayed"] is True
    assert audit["protocol_deviations"] == ["judge changed"]

    request["tool_item_types_observed"] = ["command_execution"]
    (packet_dir / "packet_000.request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="tool-use"):
        analysis.validate_annotation_audit(tmp_path / "annotations.jsonl", annotations, generations)


def test_report_done_hash_matches_written_bytes(tmp_path: Path, monkeypatch) -> None:
    generations = []
    annotations = []
    for index, planned in enumerate(runner.planned_rows(), 1):
        generations.append(
            {
                **planned,
                "termination_reason": "eos",
                "output_text": "Response:\nComplete.",
                "injection_telemetry": (
                    None if planned["layer_setting"] == "none" else {"max_abs_source_error": 0.0}
                ),
            }
        )
        annotations.append({**annotation(f"R{index:04d}"), "gen_id": planned["gen_id"]})
    generations_path = tmp_path / "generations.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    runner._write_jsonl(generations_path, generations)
    runner._write_jsonl(annotations_path, annotations)
    monkeypatch.setattr(
        analysis,
        "validate_annotation_audit",
        lambda _path, _rows, _generations: {
            "backend": "test",
            "model": "test",
            "n_packets": 1,
            "n_rows": 129,
            "all_leakage_scans_clean": True,
            "all_tool_audits_clean": True,
            "protocol_deviations": [],
        },
    )
    out = tmp_path / "analysis"
    out.mkdir()
    correction = {"row_id": "R0001", "reason": "test append-only correction"}
    (out / "annotation_corrections.jsonl").write_text(
        json.dumps(correction) + "\n", encoding="utf-8"
    )
    analysis.run(
        type(
            "Args",
            (),
            {"generations": generations_path, "annotations": annotations_path, "out": out},
        )()
    )
    done = json.loads((out / "DONE.json").read_text(encoding="utf-8"))
    assert (
        done["report_sha256"]
        == hashlib.sha256((out / "qualitative_report.md").read_bytes()).hexdigest()
    )
    assert json.loads((out / "annotation_corrections.jsonl").read_text()) == correction
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["annotation_corrections"] == 1
