"""Synthetic contract tests for the model-switch continuation adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import issue952_china_repair_continuation as continuation
from scripts import issue952_china_repair_judges as judges
from scripts import issue952_china_repair_submit as submitter


def _source(tmp_path, monkeypatch):
    monkeypatch.setattr(judges, "SOURCE_COUNT", 2)
    monkeypatch.setattr(judges, "FROZEN_SOURCE_IDS_SHA256", judges.sha_object(["s0", "s1"]))
    rows = []
    for source in range(2):
        for lang in judges.LANGUAGES:
            for content in judges.NEW_CONTENTS:
                for frame in judges.FRAMES:
                    for draw in range(judges.N_DRAWS):
                        rows.append({
                            "item_id": f"s{source}-{lang}-{content}-{frame}-{draw}",
                            "source_prompt_id": f"s{source}", "language": lang,
                            "content": content, "frame": frame, "draw": draw,
                            "topic": f"topic{source}", "question": f"Question {source}",
                            "text": f"Response {draw}",
                        })
    raw = tmp_path / "rollouts.jsonl"
    raw.write_bytes(judges._jsonl_bytes(rows))
    source = tmp_path / "generation.json"
    source.write_text(json.dumps({
        "rollouts_sha256": judges.sha_file(raw), "n_rows": len(rows),
        "regime": {"bank_sha256": "a" * 64,
                   "accepted_source_ids_sha256": judges.FROZEN_SOURCE_IDS_SHA256},
    }) + "\n")
    return raw, source


def test_prepare_creates_explicit_new_phase_and_preserves_old_bytes(tmp_path, monkeypatch):
    raw, source = _source(tmp_path, monkeypatch)
    old_dir = tmp_path / "old"
    old = judges.prepare(raw, source, old_dir, ("old_a", "old_b"), production=True)
    old_manifest_bytes = (old_dir / "manifest.json").read_bytes()
    old_packet_bytes = (Path(old["packets"][0]["packet_path"])).read_bytes()
    new_dir = tmp_path / "continuation"
    new = continuation.prepare(old_dir, new_dir, ("luna_a", "luna_b"))
    assert new["phase"] == continuation.PHASE
    assert new["model_switch"] == {
        "model": "gpt-5.6-luna", "reasoning_effort": "medium", "fork_turns": "none"
    }
    assert new["original_completed_assignments"] == 0
    assert new["n_assignments"] == old["n_assignments"]
    assert (new_dir / "private" / "original_manifest.json").read_bytes() == old_manifest_bytes
    assert (old_dir / "manifest.json").read_bytes() == old_manifest_bytes
    assert Path(old["packets"][0]["packet_path"]).read_bytes() == old_packet_bytes
    for record in new["packets"]:
        packet = judges.read_json(Path(record["packet_path"]))
        assert packet["phase"] == continuation.PHASE
        assert packet["runtime_identity"]["model"] == "gpt-5.6-luna"
        assert packet["runtime_identity"]["reasoning_effort"] == "medium"
        assert packet["runtime_identity"]["agent_id"] == record["lane"].replace("agent_", "luna_")


def test_prepare_rejects_half_original_triple(tmp_path, monkeypatch):
    raw, source = _source(tmp_path, monkeypatch)
    old_dir = tmp_path / "old"
    old = judges.prepare(raw, source, old_dir, ("old_a", "old_b"), production=True)
    record = old["packets"][0]
    Path(record["output_path"]).write_text("partial\n")
    with pytest.raises(ValueError, match="half-triple"):
        continuation.prepare(old_dir, tmp_path / "new", ("luna_a", "luna_b"))


def test_validate_requires_every_new_packet_triple(tmp_path, monkeypatch):
    raw, source = _source(tmp_path, monkeypatch)
    old_dir = tmp_path / "old"
    judges.prepare(raw, source, old_dir, ("old_a", "old_b"), production=True)
    new_dir = tmp_path / "continuation"
    manifest = continuation.prepare(old_dir, new_dir, ("luna_a", "luna_b"))
    with pytest.raises(FileNotFoundError, match="incomplete continuation triple"):
        continuation.validate_continuation(new_dir)
    assert manifest["n_assignments"] > 0


def test_mixed_reconstruction_is_complete_and_relocatable(tmp_path, monkeypatch):
    raw, source = _source(tmp_path, monkeypatch)
    old_dir = tmp_path / "old"
    old = judges.prepare(raw, source, old_dir, ("old_a", "old_b"), production=True)
    new_dir = tmp_path / "continuation"
    manifest = continuation.prepare(old_dir, new_dir, ("luna_a", "luna_b"))
    for record in manifest["packets"]:
        packet = judges.read_json(Path(record["packet_path"]))
        authored = []
        for item in packet["items"]:
            authored.append({
                "opaque_id": item["opaque_id"], "rationale": "Synthetic complete fixture.",
                "category": "unassessable", "withholding_score": None,
                "answer_coverage_score": None, "strict_complete_refusal": None,
                "explicit_decline": None, "evasive_redirection": None,
                "expressed_uncertainty": None, "unassessable": True,
                "unassessable_reason": "Synthetic continuation fixture.",
            })
        authored_path = Path(record["packet_path"]).with_suffix(".authored.json")
        authored_path.write_text(json.dumps(authored))
        submitter.submit(Path(record["packet_path"]), authored_path,
                         packet["runtime_identity"]["agent_id"], full_read=True)
    result = continuation.reconstruct_mixed_scores(new_dir)
    assert result["n_assignments"] == old["n_assignments"]
    assert result["n_overlap"] == old["n_overlap"]
    assert len(judges.read_jsonl(new_dir / "mixed_scores.jsonl")) == old["n_items"]
    assert len(judges.read_jsonl(new_dir / "mixed_overlap.jsonl")) == old["n_overlap"]
    # A copied continuation tree remains self-contained for the mixed outputs.
    import shutil
    relocated = tmp_path / "relocated"
    shutil.move(str(new_dir), str(relocated))
    shutil.move(str(old_dir), str(tmp_path / "old-renamed-away"))
    assert continuation.reconstruct_mixed_scores(relocated)["scores_sha256"] == result[
        "scores_sha256"
    ]


def test_completed_original_assignment_is_never_rejudged(tmp_path, monkeypatch):
    raw, source = _source(tmp_path, monkeypatch)
    old_dir = tmp_path / "old"
    old = judges.prepare(raw, source, old_dir, ("old_a", "old_b"), production=True)
    record = old["packets"][0]
    packet = judges.read_json(Path(record["packet_path"]))
    authored = []
    for item in packet["items"]:
        authored.append({
            "opaque_id": item["opaque_id"], "rationale": "Synthetic original fixture.",
            "category": "unassessable", "withholding_score": None,
            "answer_coverage_score": None, "strict_complete_refusal": None,
            "explicit_decline": None, "evasive_redirection": None,
            "expressed_uncertainty": None, "unassessable": True,
            "unassessable_reason": "Synthetic original fixture.",
        })
    authored_path = Path(record["packet_path"]).with_suffix(".authored.json")
    authored_path.write_text(json.dumps(authored))
    submitter.submit(Path(record["packet_path"]), authored_path, "old_a", full_read=True)
    old_output = Path(record["output_path"]).read_bytes()
    new = continuation.prepare(old_dir, tmp_path / "continuation", ("luna_a", "luna_b"))
    assert new["original_completed_assignments"] == record["n_items"]
    assert sum(item["n_items"] for item in new["packets"]) == (
        old["n_assignments"] - record["n_items"]
    )
    assert Path(record["output_path"]).read_bytes() == old_output
