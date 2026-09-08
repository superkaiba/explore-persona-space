"""Actual local archive/reconstruction bodies with autospecced network boundaries."""

import json
import shutil
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts import context_risk_trajectory_archive as archive


def fixture_stage(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    np.save(stage / "vectors.npy", np.arange(24, dtype=np.float16).reshape(3, 8))
    (stage / "observations.jsonl").write_text('{"label":0}\n{"label":1}\n')
    (stage / "config.json").write_text('{"phase":"raw"}\n')
    original = {
        p.name: {"size": p.stat().st_size, "sha256": archive.sha256(p)} for p in stage.iterdir()
    }
    (stage / "snapshot_manifest.json").write_text(
        json.dumps({"phase": "raw", "files": original}) + "\n"
    )
    return stage, original


def test_archive_run_body_round_trip_and_byte_corruption(tmp_path, monkeypatch):
    stage, original = fixture_stage(tmp_path)
    root, readback = tmp_path / "run", tmp_path / "readback"
    root.mkdir()
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"verdict": "PASS", "archive_sources_sha256": archive.source_hashes()}) + "\n"
    )
    prefix = archive.PREFIX + "/raw"
    files = {
        f"{prefix}/{p.name}": {"size": p.stat().st_size, "sha256": archive.sha256(p)}
        for p in stage.iterdir()
    }
    receipt = {
        "prefix": prefix,
        "repo_id": "fixture/repo",
        "revision": "fixture_revision",
        "passed": True,
        "files": files,
        "url": "https://example.test/fixture",
    }
    # These are external network transports; no live upload or download occurs.
    monkeypatch.setattr(archive.transport, "ROOT", archive.transport.ROOT)
    monkeypatch.setattr(archive.transport, "HF_PREFIX", archive.transport.HF_PREFIX)
    monkeypatch.setattr(
        archive.transport, "upload", create_autospec(archive.transport.upload, return_value=receipt)
    )

    def stage_prefix(repo_id, path_prefix, dest, **kwargs):
        assert repo_id == "fixture/repo" and path_prefix == prefix
        assert kwargs["revision"] == "fixture_revision"
        target = Path(dest) / prefix
        shutil.copytree(stage, target)
        return [p for p in target.iterdir() if p.is_file()]

    monkeypatch.setattr(
        archive.hub,
        "stage_hub_prefix",
        create_autospec(archive.hub.stage_hub_prefix, side_effect=stage_prefix),
    )

    def stage_text(repo_id, path_in_repo, dest, **kwargs):
        assert repo_id == "fixture/repo" and kwargs["revision"] == "fixture_revision"
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(readback / path_in_repo, path)
        return path

    monkeypatch.setattr(
        archive.hub,
        "stage_sharded_text",
        create_autospec(archive.hub.stage_sharded_text, side_effect=stage_text),
    )
    monkeypatch.setenv("EPM_HF_FILECOUNT_FALLBACK", "0")
    cfg = OmegaConf.create(
        {
            "root": str(root),
            "stage": str(stage),
            "readback": str(readback),
            "review": str(review),
            "phase": "raw",
        }
    )
    result = archive.run(cfg)
    assert result["passed"] and set(result["original_files"]) == set(original)
    assert result["original_files"]["vectors.npy"]["shape"] == [3, 8]
    assert (root / "raw_readback_receipt.json").is_file()
    (stage / "vectors.npy").write_bytes(b"corrupt array")
    with pytest.raises(ValueError, match="Array bytes differ"):
        archive.validate_original(stage / "vectors.npy", original["vectors.npy"])


def test_snapshot_rejects_extra_file_and_nonfinite_array(tmp_path):
    stage, _original = fixture_stage(tmp_path)
    manifest = json.loads((stage / "snapshot_manifest.json").read_text())
    (stage / "unexpected.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="declared original file set"):
        archive.prepare_snapshot(stage, manifest)
    bad = tmp_path / "bad.npy"
    np.save(bad, np.asarray([np.nan]))
    with pytest.raises(ValueError, match="Nonfinite"):
        archive.validate_original(bad, {"size": bad.stat().st_size, "sha256": archive.sha256(bad)})
