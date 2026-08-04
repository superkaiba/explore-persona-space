"""Round-trip pins for the wcrung rows shard transport (#1739 wildchat rung).

The shard writer + reader are the ONLY channel by which the VM-sampled
context rows reach the git-clone-only GPU lane, so a silent truncation or a
drifted parser would shrink the rung with no downstream signal. These tests
pin the round-trip, the multi-shard split, and every fail-loud guard.

No real corpus text is used — rows carry synthetic placeholder strings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import issue1739_wcrung_rows_io as rio  # noqa: E402


def _rows(n: int, pad: int = 0) -> list[dict]:
    return [
        {
            "context_id": f"wcrung-{i:04d}",
            "query": f"placeholder query {i}" + ("x" * pad),
            "prefix_turns": [{"role": "user", "content": f"turn {i}"}] if i % 2 else [],
            "single_turn": i % 2 == 0,
            "n_tokens_instruct": 100 + i,
        }
        for i in range(n)
    ]


def test_round_trip_single_shard(tmp_path):
    rows = _rows(25)
    manifest = rio.shard_rows(rows, tmp_path)
    assert manifest["n_shards"] == 1
    assert manifest["n_rows"] == 25
    assert rio.load_rows(tmp_path) == rows


def test_round_trip_multi_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(rio, "SHARD_MAX_BYTES", 2_000)
    rows = _rows(40, pad=200)
    manifest = rio.shard_rows(rows, tmp_path)
    assert manifest["n_shards"] > 1, "cap too loose to exercise the split"
    # Every shard except (possibly) the last respects the cap.
    assert all(s["n_bytes"] <= 2_000 for s in manifest["shards"][:-1])
    assert sum(s["n_rows"] for s in manifest["shards"]) == 40
    assert rio.load_rows(tmp_path) == rows


def test_oversize_single_row_still_written(tmp_path, monkeypatch):
    """A row bigger than the cap gets its own shard rather than being dropped."""
    monkeypatch.setattr(rio, "SHARD_MAX_BYTES", 500)
    rows = _rows(3, pad=2_000)
    manifest = rio.shard_rows(rows, tmp_path)
    assert manifest["n_rows"] == 3
    assert all(s["n_rows"] >= 1 for s in manifest["shards"])
    assert rio.load_rows(tmp_path) == rows


def test_shard_write_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(rio, "SHARD_MAX_BYTES", 3_000)
    rows = _rows(30, pad=100)
    a = tmp_path / "a"
    b = tmp_path / "b"
    m_a = rio.shard_rows(rows, a)
    m_b = rio.shard_rows(rows, b)
    assert m_a == m_b
    for shard in m_a["shards"]:
        assert (a / shard["name"]).read_bytes() == (b / shard["name"]).read_bytes()


def test_reshard_clears_stale_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(rio, "SHARD_MAX_BYTES", 2_000)
    rio.shard_rows(_rows(40, pad=200), tmp_path)
    n_before = len(list(tmp_path.glob("wcrung_rows.shard*.jsonl")))
    assert n_before > 1
    rio.shard_rows(_rows(3), tmp_path)
    assert len(list(tmp_path.glob("wcrung_rows.shard*.jsonl"))) == 1
    assert len(rio.load_rows(tmp_path)) == 3


def test_empty_rows_refused(tmp_path):
    with pytest.raises(ValueError, match="empty row list"):
        rio.shard_rows([], tmp_path)


def test_digest_mismatch_fails_loud(tmp_path):
    rows = _rows(10)
    manifest = rio.shard_rows(rows, tmp_path)
    victim = tmp_path / manifest["shards"][0]["name"]
    with victim.open("a") as fh:
        fh.write(json.dumps({"context_id": "smuggled"}) + "\n")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        rio.load_rows(tmp_path)


def test_missing_shard_fails_loud(tmp_path):
    rows = _rows(10)
    manifest = rio.shard_rows(rows, tmp_path)
    (tmp_path / manifest["shards"][0]["name"]).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        rio.load_rows(tmp_path)


def test_row_count_mismatch_fails_loud(tmp_path):
    rows = _rows(10)
    manifest = rio.shard_rows(rows, tmp_path)
    # Rewrite the manifest count so the recorded digest still matches but the
    # per-shard row count does not — the truncation signature.
    manifest["shards"][0]["n_rows"] += 1
    manifest["n_rows"] += 1
    (tmp_path / rio.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="row-count mismatch"):
        rio.load_rows(tmp_path)


def test_missing_manifest_fails_loud(tmp_path):
    rio.shard_rows(_rows(5), tmp_path)
    (tmp_path / rio.MANIFEST_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="manifest"):
        rio.load_rows(tmp_path)


def test_unexpected_schema_fails_loud(tmp_path):
    rio.shard_rows(_rows(5), tmp_path)
    path = tmp_path / rio.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["schema"] = "some-other-format-v9"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="unexpected shard schema"):
        rio.load_rows(tmp_path)
