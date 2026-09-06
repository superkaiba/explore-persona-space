"""Lossless bundle roundtrip and private-destination guards without network writes."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/issue2669_persist.py"
spec = importlib.util.spec_from_file_location("issue2669_persist", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def make_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for phase in ("pilot", "production"):
        directory = run / f"judgments_{phase}"
        directory.mkdir()
        (directory / "dispatch_summary.json").write_text(
            json.dumps({"results": [{"status": "complete"}]})
        )
        (directory / ".dispatch.lock").write_text("do not include")
    (run / "unicode.txt").write_bytes('First\r\nSecond\u2028Third\u2029\x00"雪"\r'.encode())
    (run / "empty.txt").write_bytes(b"")
    for i in range(5):
        (run / f"large-{i}.txt").write_bytes(("é" * 220).encode())
    return run


def mock_network(monkeypatch, private=True, result=None, exists=True):
    api = create_autospec(m.HfApi, instance=True)
    api.repo_info.return_value = SimpleNamespace(private=private, sha="verified-revision")
    api.file_exists.return_value = exists
    constructor = create_autospec(m.HfApi, return_value=api)
    monkeypatch.setattr(m, "HfApi", constructor)
    upload = create_autospec(
        m.hub._upload_folder_filtered,
        return_value=result if result is not None else m.hub.DEFAULT_OVERFLOW_REPO + "/" + m.PREFIX,
    )
    monkeypatch.setattr(m.hub, "_upload_folder_filtered", upload)
    return api, upload


def test_exact_utf8_roundtrip_and_all_source_files(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    original = {
        p.relative_to(run).as_posix(): p.read_bytes()
        for p in run.rglob("*")
        if p.is_file() and p.name != ".dispatch.lock"
    }
    api, upload = mock_network(monkeypatch)
    monkeypatch.setattr(m, "SHARD_BYTES", 1024)
    bundle = tmp_path / "bundle"
    download = create_autospec(m.hf_hub_download, return_value=str(bundle / "manifest.json"))
    monkeypatch.setattr(m, "hf_hub_download", download)
    report = m.persist(run, bundle)
    assert download.call_args.kwargs["revision"] == "verified-revision"
    restored = {}
    shards = sorted(bundle.glob("raw-*.jsonl"))
    assert len(shards) > 1
    for shard in shards:
        assert shard.stat().st_size < 1024
        with shard.open() as handle:
            for line in handle:
                record = json.loads(line)
                raw = record["content"].encode("utf-8")
                assert hashlib.sha256(raw).hexdigest() == record["sha256"]
                assert record["path"] not in restored
                restored[record["path"]] = raw
    assert restored == original
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert set(manifest["shards"]) == {p.name for p in shards}
    for name, sha in manifest["shards"].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == sha
    args, kwargs = upload.call_args
    assert args[1:4] == (m.hub.DEFAULT_OVERFLOW_REPO, "dataset", m.PREFIX)
    assert kwargs["private"] is True
    assert set(args[5]) == {m.PREFIX + "/" + p.name for p in bundle.iterdir()}
    assert report["private"] is True and report["revision"] == "verified-revision"
    assert api.file_exists.call_count == len(args[5])
    for path, raw in original.items():
        assert (run / path).read_bytes() == raw  # Never delete/mutate source artifacts.


@pytest.mark.parametrize("private", [False, None])
def test_public_or_unknown_destination_never_uploads(tmp_path, monkeypatch, private):
    run = make_run(tmp_path)
    _, upload = mock_network(monkeypatch, private=private)
    with pytest.raises(RuntimeError, match="private"):
        m.persist(run, tmp_path / "bundle")
    upload.assert_not_called()
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("failure", ["wrong_repo", "missing_remote", "privacy_changed"])
def test_upload_verification_fails_without_success_marker(tmp_path, monkeypatch, failure):
    run = make_run(tmp_path)
    api, _ = mock_network(
        monkeypatch,
        result="public/wrong/path" if failure == "wrong_repo" else None,
        exists=failure != "missing_remote",
    )
    if failure == "privacy_changed":
        api.repo_info.side_effect = [
            SimpleNamespace(private=True, sha="before"),
            SimpleNamespace(private=False, sha="after"),
        ]
    with pytest.raises(RuntimeError):
        m.persist(run, tmp_path / "bundle")
    assert not (run / "persistence.json").exists()


def test_oversized_record_never_uploads(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    _, upload = mock_network(monkeypatch)
    monkeypatch.setattr(m, "SHARD_BYTES", 300)
    with pytest.raises(ValueError, match="explicit splitting"):
        m.persist(run, tmp_path / "bundle")
    upload.assert_not_called()


def test_symlink_never_leaks_external_content(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    external = tmp_path / "external.txt"
    external.write_text("not part of run")
    (run / "linked.txt").symlink_to(external)
    _, upload = mock_network(monkeypatch)
    with pytest.raises(ValueError, match="Symlinks"):
        m.persist(run, tmp_path / "bundle")
    upload.assert_not_called()


def test_incomplete_run_never_uploads(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    (run / "judgments_production/dispatch_summary.json").write_text('{"results": []}')
    _, upload = mock_network(monkeypatch)
    with pytest.raises(RuntimeError, match="incomplete"):
        m.persist(run, tmp_path / "bundle")
    upload.assert_not_called()


def test_downloaded_manifest_hash_must_match(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    mock_network(monkeypatch)
    wrong = tmp_path / "wrong-manifest.json"
    wrong.write_text("{}")
    monkeypatch.setattr(
        m, "hf_hub_download", create_autospec(m.hf_hub_download, return_value=str(wrong))
    )
    with pytest.raises(RuntimeError, match="Downloaded manifest hash mismatch"):
        m.persist(run, tmp_path / "bundle")
    assert not (run / "persistence.json").exists()


def test_overlapping_source_bundle_rejected_before_network(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    api, upload = mock_network(monkeypatch)
    with pytest.raises(ValueError, match="must not overlap"):
        m.persist(run, run / "bundle")
    api.repo_info.assert_not_called()
    upload.assert_not_called()
