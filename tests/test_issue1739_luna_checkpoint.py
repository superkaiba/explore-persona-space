import json

import pytest

from scripts.issue1739_covariance_ablation import sha256
from scripts.issue1739_luna_checkpoint import pack_text, validate_final


def test_text_shards_roundtrip_bytes_and_unicode_line_separators(tmp_path):
    root, output = tmp_path / "source", tmp_path / "packed"
    root.mkdir()
    files = {"a.json": '{"value":"before\u2028after"}\n', "b.md": "Café\r\n" * 10}
    for name, text in files.items():
        (root / name).write_bytes(text.encode("utf-8"))
    (root / "__pycache__").mkdir()
    (root / "__pycache__/code.pyc").write_bytes(b"\xa7\x00")
    pack_text(root, output, shard_max_bytes=150)
    manifest = json.loads((output / "CHECKPOINT_MANIFEST.json").read_text())
    restored = {}
    for shard in manifest["shards"]:
        assert shard["bytes"] <= 150
        assert sha256(output / shard["name"]) == shard["sha256"]
        with (output / shard["name"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                restored[row["src"]] = row["text"].encode("utf-8")
    assert restored == {name: text.encode("utf-8") for name, text in files.items()}
    assert not manifest["annotation_complete"]
    assert not manifest["analysis_complete"]
    assert manifest["excluded"] == {
        "__pycache__/code.pyc": "Regenerable Python bytecode; source retained"
    }


def test_oversize_text_fails_without_completion_manifest(tmp_path):
    root, output = tmp_path / "source", tmp_path / "packed"
    root.mkdir()
    (root / "large.md").write_text("x" * 151)
    with pytest.raises(ValueError, match="exceeds text-shard cap"):
        pack_text(root, output, shard_max_bytes=150)
    assert not (output / "CHECKPOINT_MANIFEST.json").exists()


def test_final_gate_rejects_stale_results_and_unresolved_legacy_review(tmp_path, monkeypatch):
    hashes = {"hallucination/labels_production/packet_0000.json": "label-digest"}
    monkeypatch.setattr(
        "scripts.issue1739_luna_analysis.load_labels", lambda root: ({}, {}, hashes)
    )
    acceptance = tmp_path / "quality_acceptance.json"
    acceptance.write_text(json.dumps({
        "content_review_complete": True, "annotation_file_sha256": hashes,
    }))
    result = {
        "annotation_complete": True, "analysis_complete": True,
        "annotations": hashes, "quality_acceptance_sha256": sha256(acceptance),
    }
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(result))
    ledger = tmp_path / "coordinator_content_review.jsonl"
    pending = {"behavior": "hallucination", "id": "id1", "status": "review_pending"}
    ledger.write_text(json.dumps(pending) + "\n")
    with pytest.raises(ValueError, match="Unresolved"):
        validate_final(tmp_path, summary)
    with ledger.open("a") as stream:
        stream.write(json.dumps({"id": "id1", "status": "resolved"}) + "\n")
    assert validate_final(tmp_path, summary) == hashes
    result["annotations"] = {"different": "digest"}
    summary.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="does not match"):
        validate_final(tmp_path, summary)
