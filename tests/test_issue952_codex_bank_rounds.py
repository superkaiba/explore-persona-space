"""Synthetic-only repair history, provenance, and coverage regression tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "issue952_bank_rounds", Path(__file__).resolve().parents[1] / "scripts/issue952_codex_judges.py"
)
assert SPEC is not None and SPEC.loader is not None
CODEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CODEX)


def _outputs(manifest, *, failing=(), version=0):
    for packet in manifest["packets"]:
        items = json.loads(Path(packet["packet_path"]).read_text())["items"]
        rows = []
        for item in items:
            opaque_id = item["opaque_id"]
            if "bank_author" in manifest["phase"]:
                value = {key: f"synthetic {version} {opaque_id} {key}" for key in CODEX.AUTHOR_KEYS}
            else:
                value = {key: True for key in CODEX.AUDIT_BOOL_KEYS}
                value.update({key: 90 for key in CODEX.AUDIT_SCORE_KEYS})
                value["issues"] = "synthetic correction" if opaque_id in failing else ""
                value[CODEX.AUDIT_BOOL_KEYS[0]] = opaque_id not in failing
            rows.append({"opaque_id": opaque_id, "value": value})
        CODEX._write_jsonl(Path(packet["output_path"]), rows)


@pytest.fixture
def bank(tmp_path, monkeypatch):
    out_dir, packets = tmp_path / "out", tmp_path / "packets"
    sources = [
        {"prompt_id": str(i), "topic": f"topic-{i // 10}", "question": f"synthetic question {i}"}
        for i in range(90)
    ]
    source_path = out_dir / "inputs" / "source_test_questions.json"
    CODEX._write_json(source_path, sources)
    monkeypatch.setattr(CODEX, "SOURCE_SHA256", CODEX._sha256(source_path))
    monkeypatch.setattr(CODEX, "_validated_calibration", lambda out_dir: {"passed": True})
    CODEX._write_json(out_dir / "calibration_codex" / "report.json", {"passed": True})
    author = CODEX.prepare_bank_author(out_dir, packets)
    _outputs(author)
    return out_dir, packets, author


def _initial(bank, failing):
    out_dir, packets, _ = bank
    audit = CODEX.prepare_bank_audit(out_dir, packets)
    _outputs(audit, failing=failing)
    return audit


def _repair(bank, round_no, failing):
    out_dir, packets, _ = bank
    author = CODEX.prepare_bank_retry(out_dir, packets, round_no=round_no)
    _outputs(author, version=round_no)
    audit = CODEX.prepare_bank_audit(out_dir, packets, round_no=round_no)
    _outputs(audit, failing=failing)
    return author, audit


def test_three_rounds_reduce_latest_values_and_preserve_all_prior_artifacts(bank):
    out_dir, packets, initial = bank
    ids = [row["opaque_id"] for row in initial["mapping"]]
    _initial(bank, ids[:20])
    first, _ = _repair(bank, 1, ids[:11])
    assert len(first["mapping"]) == 20
    first_report = CODEX.finalize_bank(out_dir)
    assert first_report["passed"] is False
    before = {
        p: p.read_bytes() for root in (out_dir, packets) for p in root.rglob("*") if p.is_file()
    }
    second, _ = _repair(bank, 2, ids[:3])
    assert {row["opaque_id"] for row in second["mapping"]} == set(ids[:11])
    second_report = CODEX.finalize_bank(out_dir)
    assert second_report["n_audit_passing_items"] == 87
    # Topic 0 has only 7/10: the per-topic threshold still blocks a global 87/90.
    assert second_report["passed"] is False
    third, _ = _repair(bank, 3, [])
    assert len(third["mapping"]) == 3
    report = CODEX.finalize_bank(out_dir)
    assert report["passed"] is True
    assert report["n_prompts"] == 1080
    assert report["round_item_counts"] == {"0": 90, "1": 20, "2": 11, "3": 3}
    assert report["latest_item_counts_by_round"] == {"0": 70, "1": 9, "2": 8, "3": 3}
    assert report["passing_item_counts_by_round"] == report["latest_item_counts_by_round"]
    state = CODEX._bank_state(out_dir)
    for i, expected_round in ((0, 3), (5, 2), (15, 1), (30, 0)):
        assert state["latest_round"][ids[i]] == expected_round
        assert state["authors"][ids[i]][CODEX.AUTHOR_KEYS[0]].startswith(
            f"synthetic {expected_round} "
        )
    mutable_derived = {
        out_dir / "inputs" / name for name in ("prompt_bank.jsonl", "bank_audit_report.json")
    }
    for path, content in before.items():
        if path not in mutable_derived:
            assert path.read_bytes() == content
    with pytest.raises(RuntimeError, match="no failing"):
        CODEX.prepare_bank_retry(out_dir, packets, round_no=4)


def test_legacy_round_one_remains_readable(bank):
    out_dir, _, initial = bank
    ids = [row["opaque_id"] for row in initial["mapping"]]
    _initial(bank, ids[:2])
    _repair(bank, 1, [])
    # These optional bindings were absent in the original round-one files.
    for kind, round_no in (("audit", 0), ("author", 1), ("audit", 1)):
        path = CODEX._bank_manifest_path(out_dir, kind, round_no)
        manifest = json.loads(path.read_text())
        manifest.pop("round", None)
        manifest.pop("author_output_sha256", None)
        if kind == "author":
            manifest["audit_manifest_sha256"] = CODEX._sha256(
                CODEX._bank_manifest_path(out_dir, "audit", 0)
            )
        elif round_no == 1:
            manifest["author_manifest_sha256"] = CODEX._sha256(
                CODEX._bank_manifest_path(out_dir, "author", 1)
            )
        CODEX._write_json(path, manifest)
    report = CODEX.finalize_bank(out_dir)
    assert report["passed"] is True
    assert report["completed_repair_rounds"] == [1]
    assert (
        CODEX._bank_manifest_path(out_dir, "author", 1).name
        == "codex_bank_author_retry_manifest.json"
    )


def test_preparation_collisions_and_missing_predecessors_fail_before_new_packets(bank):
    out_dir, packets, initial = bank
    ids = [row["opaque_id"] for row in initial["mapping"]]
    with pytest.raises(RuntimeError, match="incomplete"):
        CODEX.prepare_bank_retry(out_dir, packets, round_no=1)
    _initial(bank, ids[:2])
    with pytest.raises(RuntimeError, match="predecessor"):
        CODEX.prepare_bank_retry(out_dir, packets, round_no=2)
    first = CODEX.prepare_bank_retry(out_dir, packets, round_no=1)
    saved = CODEX._bank_manifest_path(out_dir, "author", 1).read_bytes()
    with pytest.raises(RuntimeError, match="collision"):
        CODEX.prepare_bank_retry(out_dir, packets, round_no=1)
    assert CODEX._bank_manifest_path(out_dir, "author", 1).read_bytes() == saved
    with pytest.raises(RuntimeError, match="missing Codex bank output"):
        CODEX.finalize_bank(out_dir)
    _outputs(first, version=1)
    audit = CODEX.prepare_bank_audit(out_dir, packets, round_no=1)
    _outputs(audit, failing=ids[:1])
    with pytest.raises(RuntimeError, match="collision"):
        CODEX.prepare_bank_audit(out_dir, packets, round_no=1)
    (packets / "bank_author_retry_round_2" / "outputs").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="collision"):
        CODEX.prepare_bank_retry(out_dir, packets, round_no=2)
    assert not CODEX._bank_manifest_path(out_dir, "author", 2).exists()


@pytest.mark.parametrize(
    "tamper",
    ["source", "packet", "schema", "coverage", "mapping", "assignment", "predecessor", "output"],
)
def test_latest_round_corruption_never_falls_back_to_previous_result(bank, tamper):
    out_dir, _, initial = bank
    ids = [row["opaque_id"] for row in initial["mapping"]]
    _initial(bank, ids[:2])
    author, audit = _repair(bank, 1, [])
    author_path = CODEX._bank_manifest_path(out_dir, "author", 1)
    audit_path = CODEX._bank_manifest_path(out_dir, "audit", 1)
    packet = next(p for p in audit["packets"] if p["n_items"])
    if tamper == "source":
        CODEX._write_json(out_dir / "inputs" / "source_test_questions.json", [])
    elif tamper == "packet":
        CODEX._write_json(Path(packet["packet_path"]), {"items": []})
    elif tamper in {"schema", "coverage"}:
        rows = CODEX._jsonl(Path(packet["output_path"]))
        if tamper == "schema":
            rows[0]["value"][CODEX.AUDIT_SCORE_KEYS[0]] = True
        else:
            rows.pop()
        CODEX._write_jsonl(Path(packet["output_path"]), rows)
    elif tamper == "mapping":
        author["mapping"].pop()
        CODEX._write_json(author_path, author)
    elif tamper == "assignment":
        audit["packets"][0]["agent"] = audit["packets"][1]["agent"]
        CODEX._write_json(audit_path, audit)
    elif tamper == "predecessor":
        author["audit_manifest_sha256"] = "invalid"
        CODEX._write_json(author_path, author)
    else:
        _outputs(author, version=9)
    with pytest.raises(RuntimeError):
        CODEX.finalize_bank(out_dir)


def test_cli_requires_positive_explicit_repair_round(monkeypatch, tmp_path):
    for phase in ("bank-retry-prepare", "bank-retry-audit-prepare"):
        for extra in ([], ["--round", "0"], ["--round", "-1"]):
            monkeypatch.setattr(
                sys, "argv", ["judge", "--phase", phase, "--out-dir", str(tmp_path), *extra]
            )
            with pytest.raises(SystemExit) as error:
                CODEX.main()
            assert error.value.code == 2


@pytest.mark.parametrize("n_failed,passed", [(9, True), (10, False)])
def test_global_threshold_is_unchanged(bank, n_failed, passed):
    out_dir, _, initial = bank
    ids = [row["opaque_id"] for row in initial["mapping"]]
    failing = ids[::10] + ([ids[1]] if n_failed == 10 else [])
    _initial(bank, failing)
    report = CODEX.finalize_bank(out_dir)
    assert report["n_audit_passing_items"] == 90 - n_failed
    assert report["passed"] is passed


def test_duplicate_control_gate_is_repaired_and_remains_excluded_until_fixed(bank):
    out_dir, packets, initial = bank
    ids = [row["opaque_id"] for row in initial["mapping"]][:2]
    for packet in initial["packets"]:
        output_path = Path(packet["output_path"])
        rows = CODEX._jsonl(output_path)
        for row in rows:
            if row["opaque_id"] in ids:
                row["value"]["control_subject_key"] = "same synthetic control"
        CODEX._write_jsonl(output_path, rows)
    _initial(bank, [])
    report = CODEX.finalize_bank(out_dir)
    assert report["n_duplicate_control_items_excluded"] == 2
    repair = CODEX.prepare_bank_retry(out_dir, packets, round_no=1)
    assert {row["opaque_id"] for row in repair["mapping"]} == set(ids)


def test_cli_dispatches_explicit_round_to_real_preparers(bank, monkeypatch):
    out_dir, packets, initial = bank
    _initial(bank, [initial["mapping"][0]["opaque_id"]])
    for phase, kind in (("bank-retry-prepare", "author"), ("bank-retry-audit-prepare", "audit")):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "judge",
                "--phase",
                phase,
                "--round",
                "1",
                "--out-dir",
                str(out_dir),
                "--packet-root",
                str(packets),
            ],
        )
        assert CODEX.main() == 0
        manifest = json.loads(CODEX._bank_manifest_path(out_dir, kind, 1).read_text())
        assert manifest["round"] == 1
        _outputs(manifest, version=1)
    assert CODEX.finalize_bank(out_dir)["passed"]


def test_round_gaps_fail_loudly(bank):
    out_dir, packets, _ = bank
    _initial(bank, [])
    CODEX._write_json(CODEX._bank_manifest_path(out_dir, "author", 2), {})
    with pytest.raises(RuntimeError, match="missing predecessor"):
        CODEX.finalize_bank(out_dir)
    assert not (packets / "bank_author_retry_round_2").exists()
