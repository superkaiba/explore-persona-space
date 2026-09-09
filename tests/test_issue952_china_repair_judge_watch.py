"""Synthetic-only watcher lifecycle tests; no production text, scoring or remote calls."""

import subprocess
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from huggingface_hub import DatasetInfo, HfApi
from huggingface_hub.hf_api import RepoFile

from scripts import issue952_china_repair_judge_watch as watch
from scripts import issue952_china_repair_judges as judges
from scripts import issue952_china_repair_persist as persist
from scripts import issue952_china_repair_submit as submit

UPLOAD_ARCHIVE = persist.upload_archive


def test_direct_script_help_from_foreign_working_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(watch.__file__).resolve()), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--manifest-sha256" in result.stdout and "--max-polls" in result.stdout


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Patch only frozen panel constants to a real tiny sanity producer/collector."""
    source = tmp_path / "live"
    monkeypatch.setattr(judges, "PACKET_SIZE", 2)
    manifest = judges.prepare_sanity(source, ("test_a", "test_b"))
    monkeypatch.setattr(watch, "EXPECTED_PHASE", "sanity")
    monkeypatch.setattr(
        watch,
        "EXPECTED",
        {
            **{k: manifest[k] for k in ("n_items", "n_assignments", "n_overlap")},
            "n_packets": len(manifest["packets"]),
        },
    )
    lanes = Counter()
    for row in manifest["packets"]:
        lanes[row["lane"]] += row["n_items"]
    monkeypatch.setattr(watch, "EXPECTED_LANES", dict(lanes))
    return source, tmp_path / "watch", judges.sha_file(source / "manifest.json"), manifest


@pytest.fixture
def remote(monkeypatch):
    """Keep watcher uploader body real; stub only existing external transport surfaces."""
    import huggingface_hub

    api = create_autospec(HfApi, instance=True)
    api.repo_info.return_value = DatasetInfo(id=persist.REPO, sha="a" * 40)
    api.list_repo_tree.return_value = []
    monkeypatch.setattr(huggingface_hub, "HfApi", create_autospec(HfApi, return_value=api))
    calls = []

    def upload(packed_dir, remote_subpath, receipt):
        manifest, _ = persist.verify_archive(packed_dir)
        files = {p.name: persist.digest(p.read_bytes()) for p in Path(packed_dir).iterdir()}
        result = {
            "passed": True,
            "repo": persist.REPO,
            "prefix": remote_subpath,
            "revision": "a" * 40,
            "files_sha256": files,
            "n_original_files": len(manifest["files"]),
        }
        persist.immutable_write(receipt, persist.encode(result))
        calls.append((remote_subpath, manifest))
        return result

    mocked = create_autospec(persist.upload_archive, side_effect=upload)
    monkeypatch.setattr(persist, "upload_archive", mocked)
    return api, calls, mocked


def complete(record):
    """Explicit synthetic decisions exercise the unchanged real serialization helper."""
    path = Path(record["packet_path"])
    packet = judges.read_json(path)
    rows = [
        {
            "opaque_id": item["opaque_id"],
            "rationale": f"Explicit synthetic parser fixture {i}; no production judgment.",
            "category": "unassessable",
            "withholding_score": None,
            "answer_coverage_score": None,
            "strict_complete_refusal": None,
            "explicit_decline": None,
            "evasive_redirection": None,
            "expressed_uncertainty": None,
            "unassessable": True,
            "unassessable_reason": "Synthetic test only.",
        }
        for i, item in enumerate(packet["items"])
    ]
    authored = path.with_suffix(".authored.json")
    persist.immutable_write(authored, judges._bytes_json(rows))
    submit.submit(path, authored, packet["runtime_identity"]["agent_id"], full_read=True)
    return authored


def test_production_gate_has_no_cli_bypass(tmp_path):
    source = tmp_path / "sanity"
    judges.prepare_sanity(source, ("test_a", "test_b"))
    with pytest.raises(ValueError, match="production judge phase"):
        watch.step(source, tmp_path / "state", judges.sha_file(source / "manifest.json"))
    assert watch.EXPECTED == {
        "n_items": 10880,
        "n_assignments": 12000,
        "n_overlap": 1120,
        "n_packets": 1001,
    }
    assert watch.EXPECTED_LANES == {"agent_a": 5951, "agent_b": 6049}


def test_pending_and_receipt_only_do_not_collect_or_upload(fixture, remote, monkeypatch):
    source, state, pin, manifest = fixture
    forbidden = create_autospec(judges.collect, side_effect=AssertionError("premature collection"))
    monkeypatch.setattr(judges, "collect", forbidden)
    record = manifest["packets"][0]
    complete(record)
    Path(record["output_path"]).unlink()
    result = watch.step(source, state, pin, snapshot_every=1)
    assert result["technical_complete"] is False
    assert result["receipt_only_packets"] == result["authored_pending_packets"] == 1
    assert result["completed_assignments"] == 0
    assert remote[1] == [] and not (state / "done.json").exists()
    forbidden.assert_not_called()


def test_output_publication_between_dependency_probes_is_valid(fixture, remote, monkeypatch):
    """Publish a real valid quartet exactly when the watcher probes its output."""
    source, state, pin, manifest = fixture
    record = manifest["packets"][0]
    output = Path(record["output_path"])
    original_is_file = Path.is_file
    published = False

    def is_file(path):
        nonlocal published
        if path == output and not published:
            published = True
            complete(record)
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", create_autospec(original_is_file, side_effect=is_file))
    result = watch.step(source, state, pin, snapshot_every=1)
    assert published and result["completed_packets"] == result["backed_packets"] == 1
    assert len(remote[1]) == 1


def test_incremental_snapshots_copy_only_new_complete_sets(fixture, remote, monkeypatch):
    source, state, pin, manifest = fixture
    records = manifest["packets"]
    for row in records[:2]:
        complete(row)
    result = watch.step(source, state, pin, snapshot_every=2)
    assert result["completed_packets"] == result["backed_packets"] == 2
    assert result["technical_complete"] is False and len(remote[1]) == 1
    first = remote[1][0][1]
    for row in records[:2]:
        assert Path(row["packet_path"]).relative_to(source).as_posix() in first["files"]
    assert Path(records[2]["packet_path"]).relative_to(source).as_posix() not in first["files"]
    assert "manifest.json" in first["files"] and "private/lookup.json" in first["files"]
    frozen = next((state / "snapshots").iterdir()) / "source"
    assert judges.read_json(frozen / "_watch_checkpoint.json")["technical_complete"] is False
    # A no-change pass must not repack or upload previously checkpointed packets.
    with monkeypatch.context() as patch:
        patch.setattr(
            persist,
            "pack_tree",
            create_autospec(
                persist.pack_tree, side_effect=AssertionError("unnecessary repeat copy/pack")
            ),
        )
        assert watch.step(source, state, pin, snapshot_every=2)["archives_published"] == 0
    for row in records[2:4]:
        complete(row)
    watch.step(source, state, pin, snapshot_every=2)
    second = remote[1][1][1]
    assert (
        not set(first["files"])
        .intersection(second["files"])
        .intersection(p for p in first["files"] if p.startswith("packets/"))
    )
    assert all(prefix.startswith(watch.CHECKPOINT_PREFIX + "/watch_") for prefix, _ in remote[1])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_receipt",
        "missing_authored",
        "authored_drift",
        "row_order",
        "bad_score",
        "unexpected_output",
        "packet_drift",
    ],
)
def test_malformed_or_unexpected_output_fails_before_backup(fixture, remote, mutation):
    source, state, pin, manifest = fixture
    record = manifest["packets"][0]
    authored = complete(record)
    if mutation.startswith("missing_"):
        (Path(record["receipt_path"]) if mutation == "missing_receipt" else authored).unlink()
    elif mutation == "authored_drift":
        rows = judges.read_json(authored)
        rows[0]["rationale"] = "Different explicitly authored synthetic rationale."
        authored.write_bytes(judges._bytes_json(rows))
    elif mutation in ("row_order", "bad_score"):
        path = Path(record["output_path"])
        rows = judges.read_jsonl(path)
        if mutation == "row_order":
            rows.reverse()
        else:
            rows[0]["withholding_score"] = 0
        path.write_bytes(judges._jsonl_bytes(rows))
    elif mutation == "unexpected_output":
        (source / "packets/agent_a/foreign.output.jsonl").write_text("{}\n")
    else:
        path = Path(record["packet_path"])
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        watch.step(source, state, pin, snapshot_every=1)
    assert not remote[1]


def test_seen_complete_set_cannot_change_or_disappear(fixture, remote):
    source, state, pin, manifest = fixture
    record = manifest["packets"][0]
    complete(record)
    watch.step(source, state, pin)  # Not enough packets for a checkpoint yet.
    output = Path(record["output_path"])
    saved = output.read_bytes()
    output.write_bytes(saved + b" ")
    with pytest.raises(ValueError):
        watch.step(source, state, pin)
    output.write_bytes(saved)
    output.unlink()
    with pytest.raises(ValueError, match="disappeared"):
        watch.step(source, state, pin)


def test_failed_upload_resumes_frozen_snapshot(fixture, remote, monkeypatch):
    source, state, pin, manifest = fixture
    complete(manifest["packets"][0])
    with monkeypatch.context() as patch:
        patch.setattr(
            persist,
            "upload_archive",
            create_autospec(UPLOAD_ARCHIVE, side_effect=RuntimeError("synthetic transport failed")),
        )
        with pytest.raises(RuntimeError, match="transport failed"):
            watch.step(source, state, pin, snapshot_every=1)
    directory = next((state / "snapshots").iterdir())
    assert (directory / "intent.json").exists() and not (directory / "done.json").exists()
    result = watch.step(source, state, pin, snapshot_every=1)
    assert result["backed_packets"] == 1 and len(remote[1]) == 1
    assert len(list((state / "snapshots").iterdir())) == 1


def test_original_packet_change_during_pack_prevents_upload(fixture, remote, monkeypatch):
    source, state, pin, manifest = fixture
    complete(manifest["packets"][0])
    original = persist.pack_tree

    def change_packet(source_root, out_dir, include_dirs=None):
        result = original(source_root, out_dir, include_dirs)
        path = Path(manifest["packets"][-1]["packet_path"])
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(persist, "pack_tree", create_autospec(original, side_effect=change_packet))
    with pytest.raises(ValueError, match="source changed"):
        watch.step(source, state, pin, snapshot_every=1)
    assert not remote[1]


def test_full_collection_archive_every_original_byte_and_resume(fixture, remote, monkeypatch):
    source, state, pin, manifest = fixture
    for row in manifest["packets"]:
        complete(row)
    result = watch.step(source, state, pin)
    assert result["technical_complete"] is True
    assert result["completed_assignments"] == manifest["n_assignments"]
    assert remote[1][0][0] == watch.FINAL_PREFIX
    directory = state / "snapshots/final"
    archive, reconstructed = persist.verify_archive(directory / "packed")
    actual = {
        p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    assert reconstructed == actual
    assert {"scores.jsonl", "overlap.jsonl", "summary.json", "private/lookup.json"} <= set(actual)
    assert sum(p.endswith(".authored.json") for p in actual) == len(manifest["packets"])
    done = judges.read_json(state / "done.json")
    assert done["archive_manifest_sha256"] == judges.sha_file(
        directory / "packed/packed_manifest.json"
    )
    assert done["archive_revision"] == "a" * 40
    assert set(done["source_files"]) == set(archive["files"])
    with monkeypatch.context() as patch:
        patch.setattr(
            judges,
            "collect",
            create_autospec(
                judges.collect, side_effect=AssertionError("completed collection repeated")
            ),
        )
        patch.setattr(
            persist,
            "pack_tree",
            create_autospec(
                persist.pack_tree, side_effect=AssertionError("completed pack repeated")
            ),
        )
        assert watch.step(source, state, pin)["technical_complete"] is True
    assert len(remote[1]) == 1


def test_nonblocking_lock_and_source_output_isolation(fixture, remote):
    source, state, pin, _ = fixture
    with watch.exclusive_monitor(source), pytest.raises(RuntimeError, match="another watcher"):
        watch.step(source, state, pin)
    with pytest.raises(ValueError, match="outside live source"):
        watch.step(source, source / "state", pin)


def test_atomic_writer_temporary_is_explicitly_pending(fixture, remote):
    source, state, pin, manifest = fixture
    for row in manifest["packets"]:
        complete(row)
    temporary = source / "packets/agent_a/tmpabcdefgh"
    temporary.write_bytes(b"incomplete write")
    result = watch.step(source, state, pin)
    assert result["temporary_files_pending"] == 1 and result["technical_complete"] is False
    assert not (source / "summary.json").exists()
    temporary.unlink()
    assert watch.step(source, state, pin)["technical_complete"] is True


def test_final_authored_change_during_collect_fails_before_upload(fixture, remote, monkeypatch):
    source, state, pin, manifest = fixture
    for row in manifest["packets"]:
        complete(row)
    original = judges.collect

    def collect(out_dir):
        result = original(out_dir)
        authored = Path(manifest["packets"][0]["packet_path"]).with_suffix(".authored.json")
        authored.write_bytes(authored.read_bytes() + b" ")
        return result

    monkeypatch.setattr(judges, "collect", create_autospec(original, side_effect=collect))
    with pytest.raises(ValueError, match="source changed"):
        watch.step(source, state, pin)
    assert not remote[1] and not (state / "done.json").exists()


def test_bounded_cli_runs_real_step_counts_only(fixture, remote, monkeypatch, capsys):
    import sys

    source, state, pin, _ = fixture
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            "--source",
            str(source),
            "--state-dir",
            str(state),
            "--manifest-sha256",
            pin,
            "--max-polls",
            "2",
            "--poll-seconds",
            "1",
        ],
    )
    sleep = create_autospec(watch.time.sleep)
    monkeypatch.setattr(watch.time, "sleep", sleep)
    assert watch.main() == 0
    sleep.assert_called_once_with(1)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and all('"technical_complete": false' in line for line in lines)
    assert all("rationale" not in line and "question" not in line for line in lines)
    assert not remote[1]


def test_guard_refuses_existing_different_archive(tmp_path, remote):
    source, packed = tmp_path / "source", tmp_path / "packed"
    source.mkdir()
    (source / "text.txt").write_text("synthetic")
    persist.pack_tree(source, packed)
    remote[0].list_repo_tree.return_value = [
        RepoFile(path=watch.FINAL_PREFIX + "/foreign.json", size=2, oid="b" * 40)
    ]
    with pytest.raises(ValueError, match="different archive already"):
        watch.guarded_upload(packed, watch.FINAL_PREFIX, tmp_path / "receipt.json")
    remote[2].assert_not_called()


def test_guard_loads_environment_before_client_construction(tmp_path, remote, monkeypatch):
    import huggingface_hub

    from explore_persona_space.orchestrate import env

    source, packed = tmp_path / "source", tmp_path / "packed"
    source.mkdir()
    (source / "text.txt").write_text("synthetic")
    persist.pack_tree(source, packed)
    events = []
    monkeypatch.setattr(
        env,
        "load_dotenv",
        create_autospec(env.load_dotenv, side_effect=lambda: events.append("environment")),
    )

    def client():
        assert events == ["environment"]
        events.append("client")
        return remote[0]

    monkeypatch.setattr(huggingface_hub, "HfApi", create_autospec(HfApi, side_effect=client))
    watch.guarded_upload(packed, watch.FINAL_PREFIX, tmp_path / "receipt.json")
    assert events == ["environment", "client"]


def test_guard_recovers_existing_equal_archive_receipt(tmp_path, remote, monkeypatch):
    from explore_persona_space.orchestrate import hub

    source, packed = tmp_path / "source", tmp_path / "packed"
    source.mkdir()
    (source / "text.txt").write_text("synthetic")
    persist.pack_tree(source, packed)
    remote[0].list_repo_tree.return_value = [
        RepoFile(path=watch.FINAL_PREFIX + "/" + p.name, size=p.stat().st_size, oid="b" * 40)
        for p in packed.iterdir()
    ]

    def stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        assert repo_id == persist.REPO and revision == "a" * 40
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((packed / Path(path_in_repo).name).read_bytes())
        return target

    monkeypatch.setattr(
        hub, "stage_hub_file", create_autospec(hub.stage_hub_file, side_effect=stage)
    )
    receipt = tmp_path / "receipt.json"
    result = watch.guarded_upload(packed, watch.FINAL_PREFIX, receipt)
    assert result == judges.read_json(receipt) and result["revision"] == "a" * 40
