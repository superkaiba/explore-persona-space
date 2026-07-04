"""Tests for scripts/artifact_registry.py — the v2 artifact-reuse registry."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import artifact_registry


def _entry(**over):
    base = {
        "id": "model:issue900_x_seed0",
        "type": "adapter",
        "path": "superkaiba1/explore-persona-space/issue900_x_seed0",
        "issue": 900,
        "size_bytes": 12345,
        "recipe": "lr=5e-6, r=16, 1 epoch",
    }
    base.update(over)
    return base


def test_append_read_roundtrip(tmp_path):
    reg = tmp_path / "registry.jsonl"
    e1 = _entry(id="a", issue=900, fitness_notes="epoch-1 marker adapter")
    e2 = _entry(id="b", issue=901, type="eval-json")

    w1 = artifact_registry.append_artifact(e1, registry_path=reg)
    w2 = artifact_registry.append_artifact(e2, registry_path=reg)

    # created is stamped by the writer.
    assert "created" in w1 and "created" in w2
    # extra keys preserved verbatim.
    assert w1["fitness_notes"] == "epoch-1 marker adapter"

    rows = artifact_registry.read_registry(registry_path=reg)
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {"a", "b"}
    assert rows[0]["recipe"] == "lr=5e-6, r=16, 1 epoch"


def test_caller_created_preserved(tmp_path):
    reg = tmp_path / "registry.jsonl"
    written = artifact_registry.append_artifact(
        _entry(created="2026-01-01T00:00:00+00:00"), registry_path=reg
    )
    assert written["created"] == "2026-01-01T00:00:00+00:00"


@pytest.mark.parametrize("missing", sorted(artifact_registry.REQUIRED_KEYS))
def test_missing_required_key_raises(tmp_path, missing):
    reg = tmp_path / "registry.jsonl"
    entry = _entry()
    del entry[missing]
    with pytest.raises(ValueError, match="missing required key"):
        artifact_registry.append_artifact(entry, registry_path=reg)
    # the write never happened.
    assert not reg.exists()


def test_invalid_type_raises(tmp_path):
    reg = tmp_path / "registry.jsonl"
    with pytest.raises(ValueError, match="not in"):
        artifact_registry.append_artifact(_entry(type="bogus-kind"), registry_path=reg)
    assert not reg.exists()


def test_non_dict_entry_raises(tmp_path):
    reg = tmp_path / "registry.jsonl"
    with pytest.raises(ValueError, match="must be a dict"):
        artifact_registry.append_artifact(["not", "a", "dict"], registry_path=reg)


def test_missing_file_returns_empty(tmp_path):
    reg = tmp_path / "does-not-exist.jsonl"
    assert artifact_registry.read_registry(registry_path=reg) == []


def test_corrupt_line_raises(tmp_path):
    reg = tmp_path / "registry.jsonl"
    artifact_registry.append_artifact(_entry(id="ok"), registry_path=reg)
    with reg.open("a", encoding="utf-8") as f:
        f.write("this is not json\n")
    with pytest.raises(ValueError, match="corrupt registry line"):
        artifact_registry.read_registry(registry_path=reg)


def test_non_object_line_raises(tmp_path):
    reg = tmp_path / "registry.jsonl"
    reg.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected object"):
        artifact_registry.read_registry(registry_path=reg)


def test_filters(tmp_path):
    reg = tmp_path / "registry.jsonl"
    artifact_registry.append_artifact(_entry(id="a", issue=900, type="adapter"), registry_path=reg)
    artifact_registry.append_artifact(_entry(id="b", issue=901, type="adapter"), registry_path=reg)
    artifact_registry.append_artifact(
        _entry(id="c", issue=900, type="eval-json"), registry_path=reg
    )

    # issue filter accepts int and str.
    assert {r["id"] for r in artifact_registry.read_registry(registry_path=reg, issue=900)} == {
        "a",
        "c",
    }
    assert {r["id"] for r in artifact_registry.read_registry(registry_path=reg, issue="900")} == {
        "a",
        "c",
    }
    assert {
        r["id"] for r in artifact_registry.read_registry(registry_path=reg, type="eval-json")
    } == {"c"}
    assert {
        r["id"]
        for r in artifact_registry.read_registry(registry_path=reg, issue=900, type="adapter")
    } == {"a"}


def test_concurrent_append_integrity(tmp_path):
    reg = tmp_path / "registry.jsonl"
    n = 12
    barrier = threading.Barrier(n)

    def worker(i):
        barrier.wait()  # maximise contention on the flock
        artifact_registry.append_artifact(_entry(id=f"t{i}", issue=900 + i), registry_path=reg)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every line must parse (no interleaving) and all N rows must be present.
    rows = artifact_registry.read_registry(registry_path=reg)
    assert len(rows) == n
    assert {r["id"] for r in rows} == {f"t{i}" for i in range(n)}


def test_cli_append_and_list(tmp_path, capsys):
    reg = tmp_path / "registry.jsonl"
    rc = artifact_registry.main(
        ["--registry", str(reg), "append", "--json", json.dumps(_entry(id="cli1"))]
    )
    assert rc == 0
    capsys.readouterr()  # drain

    rc = artifact_registry.main(["--registry", str(reg), "list", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    listed = json.loads(out)
    assert [r["id"] for r in listed] == ["cli1"]

    # table form + issue filter also work.
    rc = artifact_registry.main(["--registry", str(reg), "list", "--issue", "999"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""
