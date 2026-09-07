"""Isolated artificial fixtures for the final evidence snapshot, never result data."""

import argparse
import hashlib
import json
from unittest.mock import create_autospec

import pytest

from scripts import issue1739_natural_preserve as preserve


def test_pack_roundtrip_fragments_empty_and_unicode(tmp_path, monkeypatch):
    monkeypatch.setattr(preserve, "PACK_PART_BYTES", 7)
    rows, expected = [], {}
    for i, raw in enumerate([b"", "αβγδεζ".encode(), bytes(range(32))]):
        path = tmp_path / f"source{i}"
        path.write_bytes(raw)
        rows.extend(preserve.pack_record(path, path.name))
        expected[path.name] = hashlib.sha256(raw).hexdigest()
    preserve.write_parts(tmp_path / "pack", rows)
    preserve.verify_pack(tmp_path / "pack", expected)
    assert len(rows) > len(expected)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "bytes", "hash", "payload"])
def test_corrupt_pack_fails(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(preserve, "PACK_PART_BYTES", 2)
    path = tmp_path / "source"
    path.write_bytes(b"abcdef")
    rows = preserve.pack_record(path, "source")
    expected = {"source": preserve.file_sha(path)}
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "missing":
        rows.pop()
    elif fault == "bytes":
        rows[0]["bytes"] += 1
    elif fault == "hash":
        rows[0]["sha256"] = "wrong"
    else:
        rows[0]["content_base64"] = "YWJj"
    preserve.write_parts(tmp_path / "pack", rows)
    with pytest.raises(ValueError):
        preserve.verify_pack(tmp_path / "pack", expected)


def test_original_bytes_are_scanned_before_encoding(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"artificial secret marker")
    scanner = create_autospec(preserve.scan_bytes, return_value=[object()])
    monkeypatch.setattr(preserve, "scan_bytes", scanner)
    with pytest.raises(ValueError, match="credential scan"):
        preserve.pack_record(path, "source")
    scanner.assert_called_once_with(path.read_bytes())


def test_hardlink_view_preserves_original_cache_and_checks_exact_tree(tmp_path, monkeypatch):
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "data.json").write_text("{}")
    cache = root / ".cache"
    cache.mkdir()
    (cache / "metadata").write_text("download bookkeeping")

    def verify(view, receipt):
        assert set(preserve.regular_files(view)) == {"data.json"}
        assert (view / "data.json").stat().st_ino == (root / "data.json").stat().st_ino
        assert receipt["revision"] == preserve.DATA_REVISIONS["prepared"]
        return {"status": "PASS"}

    verifier = create_autospec(preserve.verify_remote_cell, side_effect=verify)
    monkeypatch.setattr(preserve, "verify_remote_cell", verifier)
    hashes, proof = preserve.verify_data_tree(root, "prepared")
    assert set(hashes) == {"data.json"}
    assert proof["status"] == "PASS"
    assert (cache / "metadata").is_file()
    assert not list(tmp_path.glob("natural-proof-*"))


def snapshot_fixture(tmp_path, monkeypatch):
    root, logs = tmp_path / "run", tmp_path / "logs"
    root.mkdir()
    logs.mkdir()
    (root / "reused").mkdir()
    reused, blobs = {}, {}
    for behavior in preserve.FROZEN_BLOBS:
        name = f"eval_results/{behavior}/arm_results/all_arms_spearman.json"
        path = root / "reused" / name
        path.parent.mkdir(parents=True)
        raw = json.dumps({"unit_test_behavior": behavior}).encode()
        path.write_bytes(raw)
        blobs[behavior] = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        # The stage overwrites these three transferred files from frozen Git.
        reused[name] = {"sha256": "pre-stage-fingerprint"}
    monkeypatch.setattr(preserve, "FROZEN_BLOBS", blobs)
    preserve.atomic_json(
        root / "reused_transfer_manifest.json",
        {"source_revision": "7a47ff5ce42f16308bebaba29c1286a4e9bc8008", "files": reused},
    )
    summary = {"unit_test_only": True}
    preserve.atomic_json(
        root / "natural_scaling_audit.json",
        {"scientific_commit": preserve.SCIENTIFIC_COMMIT, "cells": [], "aggregate": summary},
    )
    monkeypatch.setattr(
        preserve, "aggregate", create_autospec(preserve.aggregate, return_value=summary)
    )
    report = root / "report.json"
    preserve.atomic_json(report, {"phase": "fits", "status": "complete", "pid": 999999999})
    sentinel = logs / "issue-1739-test.json"
    preserve.atomic_json(sentinel, {"payload": {"phase": "fits", "rc": 0, "report": str(report)}})
    monkeypatch.setattr(preserve, "DATA_REVISIONS", {})
    return argparse.Namespace(
        root=root,
        dest=tmp_path / "snapshot",
        launcher_dir=logs,
        helper_source=[],
        wait_pid=[999999999],
        success_sentinel=sentinel,
    )


def test_snapshot_real_body_preserves_unknown_cache_and_frozen_git_inputs(tmp_path, monkeypatch):
    args = snapshot_fixture(tmp_path, monkeypatch)
    cache = args.root / ".cache"
    cache.mkdir()
    (cache / "unexpected-output.json").write_text('{"unit_test_only": true}')
    result = preserve.snapshot(args)
    assert result["status"] == "PASS"
    assert result["dispositions"]["frozen_git_input"] == 3
    rows = preserve.load_parts(args.dest / "source_inventory")
    assert (
        next(r for r in rows if r["key"].endswith("unexpected-output.json"))["disposition"]
        == "lossless_metadata_pack"
    )


def test_new_launcher_source_during_snapshot_fails(tmp_path, monkeypatch):
    args = snapshot_fixture(tmp_path, monkeypatch)
    original = preserve.verify_pack

    def mutate(directory, expected):
        original(directory, expected)
        (args.launcher_dir / "issue-1739-new.log").write_text("new output")

    monkeypatch.setattr(preserve, "verify_pack", create_autospec(original, side_effect=mutate))
    with pytest.raises(ValueError, match="source name set changed"):
        preserve.snapshot(args)


def test_live_owned_process_blocks_snapshot(tmp_path, monkeypatch):
    import os

    args = snapshot_fixture(tmp_path, monkeypatch)
    args.wait_pid = [os.getpid()]
    with pytest.raises(ValueError, match="still exists"):
        preserve.snapshot(args)


def test_symlink_source_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.write_text("evidence")
    (tmp_path / "link").symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        preserve.regular_files(tmp_path)
