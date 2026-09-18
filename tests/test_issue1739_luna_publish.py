import json

import pytest

from scripts.issue1739_covariance_ablation import sha256
from scripts.issue1739_luna_publish import publish


@pytest.fixture
def packet(tmp_path):
    lane = tmp_path / "sycophancy"
    for phase in ("production", "labels_candidates", "labels_production"):
        (lane / phase).mkdir(parents=True)
    name = "packet_0000.json"
    ids = ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"]
    inputs = lane / "production" / name
    inputs.write_text(json.dumps([{"id": rid} for rid in ids]))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "behaviors": {
                    "sycophancy": {"production": [{"name": name, "sha256": sha256(inputs)}]}
                }
            }
        )
    )
    rows = [
        dict(
            id=rid,
            rationale=f"Specific response evidence for {rid}.",
            status="scored",
            score=0,
            positive=False,
            evidence_type="transcript",
            sources=[],
        )
        for rid in ids
    ]
    source = tmp_path / "authored.py"
    source.write_text("rows = " + repr(rows))
    candidate = lane / "labels_candidates" / name
    candidate.write_text(json.dumps(rows))
    return tmp_path, lane, name, rows, source, candidate


def test_valid_candidate_published_with_source_hashes(packet):
    root, lane, name, _rows, source, candidate = packet
    publish(root, "sycophancy", name, [source])
    assert (lane / "labels_production" / name).read_bytes() == candidate.read_bytes()
    event = json.loads((lane / "coordinator_publications.jsonl").read_text())
    assert event["authored_sources"][str(source)] == sha256(source)
    with pytest.raises(ValueError, match="Refusing to replace"):
        publish(root, "sycophancy", name, [source])


def test_invalid_candidate_cannot_replace_valid_packet(packet):
    root, lane, name, rows, source, candidate = packet
    publish(root, "sycophancy", name, [source])
    original = (lane / "labels_production" / name).read_bytes()
    rows[0]["id"] = "cccccccccccccccc"
    candidate.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        publish(root, "sycophancy", name, [source], ["aaaaaaaaaaaaaaaa"])
    assert (lane / "labels_production" / name).read_bytes() == original


def test_correction_requires_exact_changed_set_and_preserves_original(packet):
    root, lane, name, rows, source, candidate = packet
    publish(root, "sycophancy", name, [source])
    original = (lane / "labels_production" / name).read_bytes()
    rows[0].update(score=75, positive=True, rationale="Unsupported praise replaces assessment.")
    candidate.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Correction IDs mismatch"):
        publish(root, "sycophancy", name, [source], [rows[1]["id"]])
    with pytest.raises(ValueError, match="No explicitly authored"):
        publish(root, "sycophancy", name, [source], [rows[0]["id"]])
    source.write_text("rows = " + repr([rows[0]]))
    publish(root, "sycophancy", name, [source], [rows[0]["id"]])
    assert json.loads((lane / "labels_production" / name).read_text()) == rows
    backups = list((lane / "coordinator_corrections" / "production").glob("*/" + name))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_audit_uses_separate_input_and_output_directories(packet):
    root, lane, name, rows, source, candidate = packet
    for phase in ("audit", "labels_audit_candidates", "labels_audit"):
        (lane / phase).mkdir()
    audit_input = lane / "audit" / name
    audit_input.write_text((lane / "production" / name).read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["behaviors"]["sycophancy"]["audit"] = [{"name": name, "sha256": sha256(audit_input)}]
    (root / "manifest.json").write_text(json.dumps(manifest))
    (lane / "labels_audit_candidates" / name).write_bytes(candidate.read_bytes())
    publish(root, "sycophancy", name, [source], phase="audit")
    assert not (lane / "labels_production" / name).exists()
    assert json.loads((lane / "labels_audit" / name).read_text()) == rows


def test_equivalent_json_rewrite_during_validation_cannot_be_published(packet, monkeypatch):
    root, lane, name, _rows, source, candidate = packet
    original_read = type(candidate).read_bytes
    reads = 0

    def concurrent_read(path):
        nonlocal reads
        data = original_read(path)
        if path == candidate:
            reads += 1
            if reads == 2:
                # Simulate a writer changing formatting after the initial snapshot.
                return data + b"\n"
        return data

    monkeypatch.setattr(type(candidate), "read_bytes", concurrent_read)
    with pytest.raises(ValueError, match="Candidate changed"):
        publish(root, "sycophancy", name, [source])
    assert not (lane / "labels_production" / name).exists()
