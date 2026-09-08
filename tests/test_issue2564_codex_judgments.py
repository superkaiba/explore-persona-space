"""The Codex transport must preserve real labels, units and instrument provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import issue2564_codex_judgments as m


def value(prop, i=0):
    result = {"reason": "Evidence is present in this text."}
    if m.PROPERTIES[prop]["kind"] == "graded":
        result.update(assessable=i != 0, score=(i * 7) % 101 if i else None)
    else:
        result["label"] = m.PROPERTIES[prop]["labels"][i % len(m.PROPERTIES[prop]["labels"])]
    if prop == "persona":
        result["multi_voice"] = False
    return result


def fixture(tmp_path, repeats=5, props=None):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = [{"id": f"x{i:02}", "part": "pilot", "answer": f"Text item {i}."} for i in range(32)]
    (prepared / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (prepared / "vectors.npz").write_bytes(b"test-only hash fixture, never loaded as activations")
    m.dump(prepared / "rubrics.json", m.PROPERTIES)
    jobs = []
    for prop in props or m.PROPERTIES:
        for d in range(repeats):
            base = tmp_path / "neutral" / f"{prop}{d}"
            base.mkdir(parents=True)
            packet = {
                "rubric": m.PROPERTIES[prop],
                "items": [{k: r[k] for k in ("id", "answer")} for r in rows],
            }
            request = {
                "task_name": f"reader_{prop}_{d}",
                "fork_turns": "none",
                "message": "Read only the supplied text. Use the rubric. Give a short reason.",
            }
            m.dump(base / "input.json", packet)
            m.dump(base / "request.json", request)
            (base / "output.jsonl").write_text(
                "".join(
                    json.dumps({"id": r["id"], **value(prop, i)}) + "\n" for i, r in enumerate(rows)
                )
            )
            jobs.append(
                {
                    "property": prop,
                    "part": "pilot",
                    "repetition": d,
                    "input_path": str(base / "input.json"),
                    "output_path": str(base / "output.jsonl"),
                    "request_path": str(base / "request.json"),
                    "input_sha256": m.file_hash(base / "input.json"),
                    "request": request,
                    "agent_id": f"/root/reader_{prop}_{d}",
                    "status": "complete",
                    "completion_observed_unix_s": 1.0,
                    "fork_turns": "none",
                    "model_override": None,
                    "reasoning_effort_override": None,
                    "temperature": None,
                    "tool_access_audit": "unmeasured",
                }
            )
    ledger = tmp_path / "ledger.json"
    m.dump(ledger, {"provider": m.PROVIDER, "jobs": jobs})
    return ledger, jobs


def review(tmp_path):
    path = tmp_path / "review.json"
    m.dump(
        path,
        {
            "verdict": "accept",
            "reviewer": "test fixture review",
            "subagent_instrument_limitations": "Shared model; no human validation.",
            "batching_plan": "32 pilot, 256 main, first envelope inspected.",
            "properties": {
                p: {
                    "verdict": "qualified",
                    "semantic_notes": "Fixture only.",
                    "reliability_notes": "Descriptive only.",
                    "coverage_notes": "Fixture only.",
                }
                for p in m.PROPERTIES
            },
        },
    )
    return path


def test_payload_is_not_api_envelope_and_missing_is_not_zero():
    x = {"reason": "No assessable language.", "assessable": False, "score": None}
    assert m.validate_value("warmth", x) == (x, None)
    for score in (0, True, 1.5, -1, 101):
        assert m.validate_value("warmth", {**x, "score": score})[1] == "invalid_score"
    assert m.validate_value("warmth", {"choices": []})[1] == "schema_mismatch"
    assert m.validate_value("topic", {"reason": "x", "label": "unknown"})[1] == "invalid_label"
    assert (
        m.validate_value("topic", {"reason": "word " * 41, "label": "other"})[1]
        == "reason_too_long"
    )


def test_partial_import_retains_exact_requests_without_completion(tmp_path):
    ledger, jobs = fixture(tmp_path, repeats=1, props=["warmth"])
    m.import_ledger(tmp_path, ledger, "pilot")
    manifest = m.aggregate(tmp_path, "pilot")
    assert manifest["persisted_draws"] == 32 and not manifest["exact_keyset_complete"]
    out = m.output_dir(tmp_path, "pilot")
    assert not (out / "complete.json").exists()
    labels = json.loads((out / "labels.json").read_text())
    assert labels[0]["properties"]["warmth"]["n_valid"] == 1
    assert labels[0]["properties"]["warmth"]["mean"] is None
    archived = next((out / "packets").glob("*/request.json"))
    assert archived.read_bytes() == Path(jobs[0]["request_path"]).read_bytes()
    with pytest.raises(ValueError, match="not completed"):
        m.accept_pilot(tmp_path, review(tmp_path))


def test_content_only_packet_and_hash_are_enforced(tmp_path):
    ledger, jobs = fixture(tmp_path, repeats=1, props=["topic"])
    p = Path(jobs[0]["input_path"])
    packet = json.loads(p.read_text())
    packet["items"][0]["source"] = "hidden"
    m.dump(p, packet)
    with pytest.raises(ValueError, match="extra metadata"):
        m.import_ledger(tmp_path, ledger, "pilot")
    packet["items"][0].pop("source")
    packet["items"][0]["answer"] = "changed"
    m.dump(p, packet)
    with pytest.raises(ValueError, match="differs from frozen"):
        m.import_ledger(tmp_path, ledger, "pilot")


def test_truncated_output_is_archived_and_fails_loudly(tmp_path):
    ledger, jobs = fixture(tmp_path, repeats=1, props=["topic"])
    p = Path(jobs[0]["output_path"])
    p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Missing, duplicated"):
        m.import_ledger(tmp_path, ledger, "pilot")
    archived = next(m.output_dir(tmp_path, "pilot").glob("packets/*/output.jsonl"))
    assert archived.read_bytes() == p.read_bytes()
    with pytest.raises(ValueError, match="Missing, duplicated"):
        m.aggregate(tmp_path, "pilot")
    # Import resume must not silently skip a previously archived malformed output.
    with pytest.raises(ValueError, match="Missing, duplicated"):
        m.import_ledger(tmp_path, ledger, "pilot")


def test_same_judge_omission_completion_preserves_original_and_rejects_tampering(tmp_path):
    ledger, jobs = fixture(tmp_path, repeats=1, props=["persona"])
    job = jobs[0]
    source = Path(job["output_path"])
    lines = source.read_text().splitlines()
    source.write_text("\n".join(lines[:7] + lines[8:]) + "\n")
    with pytest.raises(ValueError, match="Missing, duplicated"):
        m.import_ledger(tmp_path, ledger, "pilot")
    path = next(m.output_dir(tmp_path, "pilot").glob("packets/*/record.json")).parent
    original_hashes = {n: m.file_hash(path / n) for n in ("record.json", "output.jsonl")}
    packet = json.loads(Path(job["input_path"]).read_text())
    m.dump(tmp_path / "missing.json", {"rubric": packet["rubric"], "items": packet["items"][7:8]})
    m.dump(
        tmp_path / "request.json",
        {
            "target": job["agent_id"],
            "message": "Read only the missing item. Preserve previous judgments.",
        },
    )
    (tmp_path / "extra.jsonl").write_text(lines[7] + "\n")
    m.dump(
        tmp_path / "event.json",
        {
            "agent_id": job["agent_id"],
            "completion_observed_unix_s": 2.0,
            "completion_evidence": "Test-only observed completion fixture",
        },
    )
    rows = {r["id"]: r for r in m.load_rows(tmp_path, "pilot")}
    result = m.import_supplement(
        path,
        tmp_path / "missing.json",
        tmp_path / "request.json",
        tmp_path / "extra.jsonl",
        tmp_path / "event.json",
        rows,
    )
    assert result["validated_annotations"] == 32 and result["supplemented_annotations"] == 1
    assert {n: m.file_hash(path / n) for n in original_hashes} == original_hashes
    m.import_ledger(tmp_path, ledger, "pilot")
    _, units, manifest = m.collect(tmp_path, "pilot")
    assert len(units) == 32 and len({v["agent_id"] for v in units}) == 1
    assert any("supplement_output.jsonl" in p for p, _ in manifest["packet_files"])
    repair_path = path / "supplement_record.json"
    repair = json.loads(repair_path.read_text())
    for key, changed in (
        ("agent_id", "/root/different_judge"),
        ("completion_observed_unix_s", 0.0),
        ("original_record_sha256", "changed"),
    ):
        m.dump(repair_path, {**repair, key: changed})
        with pytest.raises(ValueError, match="provenance"):
            m.parse_packet(path, rows)
    m.dump(repair_path, repair)
    output_path = path / "supplement_output.jsonl"
    output_path.write_text(lines[8] + "\n")
    with pytest.raises(ValueError, match="changed"):
        m.parse_packet(path, rows)
    repair["file_hashes"]["supplement_output.jsonl"] = m.file_hash(output_path)
    m.dump(repair_path, repair)
    with pytest.raises(ValueError, match="exactly the omitted IDs"):
        m.parse_packet(path, rows)


def test_supplement_refuses_duplicate_or_complete_original(tmp_path):
    path = tmp_path
    record = {"row_ids": ["a", "b"]}
    (path / "supplement_record.json").write_text("{}")
    for values, message in (
        ([{"id": "a"}, {"id": "a"}], "ordered strict subset"),
        ([{"id": "b"}, {"id": "a"}], "ordered strict subset"),
        ([{"id": "a"}, {"id": "b"}], "complete original"),
    ):
        (path / "output.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
        with pytest.raises(ValueError, match=message):
            m.completed_values(path, record, [])


def test_five_repeats_require_fresh_agent_contexts(tmp_path):
    ledger, jobs = fixture(tmp_path, repeats=2, props=["topic"])
    jobs[1]["agent_id"] = jobs[0]["agent_id"]
    m.dump(ledger, {"provider": m.PROVIDER, "jobs": jobs})
    m.import_ledger(tmp_path, ledger, "pilot")
    with pytest.raises(ValueError, match="same agent"):
        m.aggregate(tmp_path, "pilot")


def test_same_agent_cannot_rate_different_property_packets(tmp_path):
    ledger, jobs = fixture(tmp_path, repeats=1, props=["topic", "format"])
    jobs[1]["agent_id"] = jobs[0]["agent_id"]
    m.dump(ledger, {"provider": m.PROVIDER, "jobs": jobs})
    m.import_ledger(tmp_path, ledger, "pilot")
    with pytest.raises(ValueError, match="fresh agent context"):
        m.aggregate(tmp_path, "pilot")


def test_completed_full_pilot_is_pinned_to_original_outputs(tmp_path):
    ledger, _ = fixture(tmp_path)
    m.import_ledger(tmp_path, ledger, "pilot")
    manifest = m.aggregate(tmp_path, "pilot")
    assert manifest["expected_draws"] == manifest["persisted_draws"] == 1120
    accepted = m.accept_pilot(tmp_path, review(tmp_path))
    assert m.validate_codex_pilot(tmp_path, m.RECIPE) == accepted
    assert accepted["pins"]["judge_recipe"]["temperature"] is None
    with pytest.raises(RuntimeError, match="immutable"):
        m.aggregate(tmp_path, "pilot")
    out = m.output_dir(tmp_path, "pilot")
    raw = next(out.glob("packets/*/output.jsonl"))
    raw.write_text(raw.read_text().replace("Evidence", "Altered", 1))
    with pytest.raises(ValueError, match="changed"):
        m.validate_codex_pilot(tmp_path)


def test_ties_and_missing_persona_are_preserved():
    rows = [{"id": "a"}, {"id": "b"}]
    units = []
    for d, label in enumerate(["expert_guide", "expert_guide", "personal_peer", "personal_peer"]):
        units.append(
            {
                "row_id": "a",
                "property": "persona",
                "draw": d,
                "parsed": {"reason": "x", "label": label, "multi_voice": True},
                "drop": None,
            }
        )
    labels, q = m.summarize_units(rows, units)
    assert labels[0]["properties"]["persona"]["modal"] is None
    assert labels[0]["properties"]["persona"]["votes"]["expert_guide"] == 0.5
    assert labels[1]["properties"]["persona"]["multi_voice_fraction"] is None
    assert q["persona"]["complete_valid_item_fraction"] == 0


def test_request_information_leak_refuses_send():
    packet = {"items": [{"answer": "Please predict tomorrow's weather."}]}
    m.leakage_scan(packet, {"message": "Read the supplied packet."})
    with pytest.raises(ValueError, match="information leak"):
        m.leakage_scan(packet, {"message": "Use this to measure recoverability."})


def test_main_roster_preflight_preserves_all_answer_property_repeats(tmp_path):
    ledger, _ = fixture(tmp_path, repeats=1)
    (tmp_path / "codex_pilot_jobs.json").write_bytes(ledger.read_bytes())
    rows_path = tmp_path / "prepared/rows.jsonl"
    main = [{"id": f"main{i}", "part": "main", "answer": f"Text {i}."} for i in range(2048)]
    with rows_path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in main)
    audit = m.prepare_main_ledger(tmp_path, tmp_path / "packets_main")
    assert audit["jobs"] == 280 and audit["ratings"] == 71680
    jobs = json.loads((tmp_path / "codex_main_jobs.json").read_text())["jobs"]
    assert all(j["status"] == "pending" and j["agent_id"] is None for j in jobs)
    assert all(len(json.loads(Path(j["input_path"]).read_text())["items"]) == 256 for j in jobs)
    with pytest.raises(FileExistsError, match="already frozen"):
        m.prepare_main_ledger(tmp_path, tmp_path / "packets_main")
    with pytest.raises(RuntimeError, match="pilot acceptance"):
        m.import_ledger(tmp_path, tmp_path / "codex_main_jobs.json", "main")
