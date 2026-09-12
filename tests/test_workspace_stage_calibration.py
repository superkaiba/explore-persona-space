"""Calibration staging pins remote bytes and never replaces mismatched local files."""

import runpy
from pathlib import Path

import pytest

from explore_persona_space.analysis.workspace_runtime import file_sha256


@pytest.fixture
def staging(monkeypatch, tmp_path):
    from explore_persona_space.orchestrate import env

    monkeypatch.setattr(env, "load_dotenv", lambda: None)
    script = Path(__file__).resolve().parents[1] / "scripts/workspace_jr_stage_calibration.py"
    stage_file = runpy.run_path(str(script))["stage_file"]
    source = tmp_path / "cache/blob"
    source.parent.mkdir()
    source.write_bytes(b"synthetic immutable checkpoint")
    snapshot = source.parent / "snapshot.pt"
    snapshot.symlink_to(source)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(stage_file.__globals__, "hf_hub_download", download)
    relative = "lens_shards/prompt-0000.pt"
    upload = {
        "repo": "fixture/synthetic",
        "revision": "a" * 40,
        "prefix": "exploratory_workspace_jr/unit_fixture/rank0",
        "verified_sha256": {relative: file_sha256(source)},
    }
    destination = tmp_path / "staged" / relative
    return stage_file, source, upload, relative, destination, calls


def test_stage_download_is_revision_pinned_and_hardlinks_verified_blob(staging):
    stage, source, upload, relative, destination, calls = staging
    stage(upload, relative, destination)
    assert calls == [
        {
            "repo_id": upload["repo"],
            "repo_type": "dataset",
            "revision": upload["revision"],
            "filename": f"{upload['prefix']}/{relative}",
        }
    ]
    assert not destination.is_symlink()
    assert destination.samefile(source)
    assert file_sha256(destination) == upload["verified_sha256"][relative]


def test_matching_resume_and_shared_checkpoint_need_no_second_download(staging):
    stage, source, upload, relative, destination, calls = staging
    stage(upload, relative, destination)
    worker_two = {**upload, "revision": "b" * 40, "prefix": "another_verified_worker"}
    stage(worker_two, relative, destination)
    assert len(calls) == 1
    assert destination.samefile(source)


def test_conflicting_shared_checkpoint_is_rejected_without_overwrite(staging):
    stage, _, upload, relative, destination, calls = staging
    stage(upload, relative, destination)
    before = destination.read_bytes()
    conflicting = {**upload, "verified_sha256": {relative: "0" * 64}}
    with pytest.raises(ValueError, match="Staged file differs"):
        stage(conflicting, relative, destination)
    assert destination.read_bytes() == before
    assert len(calls) == 1


def test_bad_download_never_publishes_destination(staging):
    stage, source, upload, relative, destination, calls = staging
    source.write_bytes(b"corrupt cached checkpoint")
    with pytest.raises(ValueError, match="Downloaded bytes differ"):
        stage(upload, relative, destination)
    assert len(calls) == 1
    assert not destination.exists()


def test_existing_symlink_is_rejected_even_when_bytes_match(staging):
    stage, source, upload, relative, destination, calls = staging
    destination.parent.mkdir(parents=True)
    destination.symlink_to(source)
    with pytest.raises(ValueError, match="Staged file differs"):
        stage(upload, relative, destination)
    assert destination.is_symlink()
    assert not calls
