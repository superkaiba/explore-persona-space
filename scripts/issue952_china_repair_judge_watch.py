"""Mechanical progress, immutable backups and final collection of existing judgments.

No scoring, semantic decisions, model calls, task changes or CPU dispatch occur here.
Only complete authored/receipt/output sets enter checkpoints. A checkpoint is NOT
technical completion. Reading attestations remain the individual authors' assertions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import sys
import time


def _ensure_repo_root_on_syspath() -> None:
    """Support direct script execution from any cwd using this checkout, not shared main."""
    checkout = str(Path(__file__).resolve().parents[1])
    if checkout not in sys.path:
        sys.path.insert(0, checkout)


_ensure_repo_root_on_syspath()

from scripts import issue952_china_repair_judges as judges
from scripts import issue952_china_repair_persist as persist
from scripts.issue952_china_repair_submit import AUTHORED_FIELDS

CONTRACT = "china-judge-watch-v1"
EXPECTED_PHASE = "production"
EXPECTED = {"n_items": 10880, "n_assignments": 12000, "n_overlap": 1120, "n_packets": 1001}
EXPECTED_LANES = {"agent_a": 5951, "agent_b": 6049}
FINAL_PREFIX = f"{persist.PREFIX}/attempt1/judge"
CHECKPOINT_PREFIX = f"{persist.PREFIX}/attempt1/judge_checkpoints"
PREPARED_BACKUP = {
    "revision": "f29ac72e8a1616934ca8435b951c09d4fbb1a3c9",
    "prefix": f"{CHECKPOINT_PREFIX}/prepared_bank",
}
AGGREGATES = {"scores.jsonl", "overlap.jsonl", "summary.json"}


def require(condition: bool, message: str) -> None:
    """Fail loudly without printing any judgment or full-text input."""
    if not condition:
        raise ValueError(message)


def utf8_bytes(path: Path) -> bytes:
    """Read exact bytes, refusing a non-text artifact without echoing its contents."""
    data = path.read_bytes()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"non-UTF8 source artifact: {path.name}") from None
    return data


def file_record(path: Path) -> dict:
    """Return the exact-byte identity of one original text file."""
    data = utf8_bytes(path)
    return {"sha256": persist.digest(data), "bytes": len(data)}


def own_path(source: Path, value: str) -> Path:
    """Require a canonical, nonsymlink path inside the live collector."""
    path = Path(value)
    require(path.is_absolute() and path.is_relative_to(source), "foreign collector path")
    relative = path.relative_to(source)
    require(".." not in relative.parts, "noncanonical collector path")
    for i in range(1, len(relative.parts) + 1):
        require(not source.joinpath(*relative.parts[:i]).is_symlink(), "collector symlink")
    require(path.resolve() == path, "collector path is not canonical")
    return path


def inventory(source: Path) -> tuple[set[str], set[str]]:
    """Enumerate source names, explicitly distinguishing atomic-writer temporary files."""
    files, temporary = set(), set()
    for path in source.rglob("*"):
        require(not path.is_symlink(), "symlink in live collector")
        if path.is_file():
            name = path.relative_to(source).as_posix()
            # judges.write_immutable uses NamedTemporaryFile in the packet lane.
            # These are pending, never backed up or silently treated as completion.
            if (
                len(path.relative_to(source).parts) == 3
                and path.parent.parent == source / "packets"
                and re.fullmatch(r"tmp[a-z0-9_]{8}", path.name)
            ):
                temporary.add(name)
            else:
                files.add(name)
    return files, temporary


def validate_packet_names(names: set[str], expected: set[str]) -> None:
    """Reject unexpected decision artifacts, including additions during a collection pass."""
    for name in names:
        if name.startswith("packets/") or name.endswith(
            (".output.jsonl", ".authored.json", ".read_receipt.json")
        ):
            require(name in expected, "unexpected packet/output artifact")


@contextmanager
def exclusive_monitor(source: Path):
    """Hold one nonblocking lock per collector, independently of watcher state path."""
    source = source.resolve()
    lock = source.parent / f".{source.name}.watch.lock"
    require(not lock.is_symlink(), "watch lock is a symlink")
    with lock.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another watcher owns this collector") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def inspect_source(source: Path, manifest_sha256: str) -> dict:
    """Validate original preparation and complete sets; never aggregate judgment outcomes."""
    require(re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is not None, "invalid manifest pin")
    require(judges.sha_file(source / "manifest.json") == manifest_sha256, "manifest pin changed")
    manifest = judges.read_json(source / "manifest.json")
    require(manifest["phase"] == EXPECTED_PHASE, "not the frozen production judge phase")
    require(
        all(manifest[k] == EXPECTED[k] for k in ("n_items", "n_assignments", "n_overlap"))
        and len(manifest["packets"]) == EXPECTED["n_packets"],
        "not the frozen production assignment/packet census",
    )
    for key in ("lookup_path", "source_manifest_path"):
        own_path(source, manifest[key])
    lookup = judges._validate_manifest(manifest)
    entries = {
        (lane, info["opaque_id"]): entry
        for entry in lookup
        for lane, info in entry["lanes"].items()
    }
    require(len(entries) == EXPECTED["n_assignments"], "duplicate/missing lookup assignments")
    names, temporary = inventory(source)
    expected_names, packet_pins, complete, covered = set(), {}, {}, set()
    lane_counts = Counter()
    receipt_only = authored_pending = 0
    for record in manifest["packets"]:
        packet_path = own_path(source, record["packet_path"])
        require(
            packet_path.parent == source / "packets" / record["lane"]
            and re.fullmatch(r"batch_[0-9]{4}\.json", packet_path.name) is not None,
            "unexpected packet location",
        )
        paths = {
            "packet_path": packet_path,
            "receipt_path": packet_path.with_suffix(".read_receipt.json"),
            "output_path": packet_path.with_suffix(".output.jsonl"),
            "authored_path": packet_path.with_suffix(".authored.json"),
        }
        for key in ("receipt_path", "output_path"):
            require(own_path(source, record[key]) == paths[key], "non-sibling packet artifact")
        relative = {k: p.relative_to(source).as_posix() for k, p in paths.items()}
        require(not expected_names.intersection(relative.values()), "duplicate packet path")
        expected_names.update(relative.values())
        packet_pin = file_record(packet_path)
        require(packet_pin["sha256"] == record["packet_sha256"], "original packet changed")
        packet_pins[relative["packet_path"]] = packet_pin
        packet = judges.read_json(packet_path)
        require(packet["paths"] == {k: str(paths[k]) for k in packet["paths"]}, "packet path drift")
        judges._validate_packet(packet, record, manifest, entries)
        keys = {(record["lane"], item["opaque_id"]) for item in packet["items"]}
        require(not covered.intersection(keys), "duplicate packet assignments")
        covered.update(keys)
        lane_counts[record["lane"]] += record["n_items"]
        receipt_exists = paths["receipt_path"].is_file()
        authored_exists = paths["authored_path"].is_file()
        output_exists = paths["output_path"].is_file()
        if not output_exists:
            if receipt_exists:
                judges._validate_receipt(judges.read_json(paths["receipt_path"]), packet, record)
            receipt_only += int(receipt_exists)
            authored_pending += int(authored_exists)
            continue
        # Publication is ordered receipt -> output. The output may have appeared
        # after the pending-state probes, so sample its dependencies afresh.
        require(
            paths["receipt_path"].is_file() and paths["authored_path"].is_file(),
            "output lacks receipt/authored dependencies",
        )
        before = {relative[k]: file_record(p) for k, p in paths.items()}
        require(before[relative["packet_path"]] == packet_pin, "packet changed during validation")
        judges._validate_receipt(judges.read_json(paths["receipt_path"]), packet, record)
        authored = judges.read_json(paths["authored_path"])
        rows = judges.read_jsonl(paths["output_path"])
        require(
            isinstance(authored, list)
            and all(isinstance(row, dict) and set(row) == AUTHORED_FIELDS for row in authored)
            and [row["opaque_id"] for row in authored] == record["opaque_ids"]
            and [row.get("opaque_id") for row in rows] == record["opaque_ids"],
            "authored/output fields or ordered coverage differ",
        )
        receipt_sha = before[relative["receipt_path"]]["sha256"]
        for item, decision, row in zip(packet["items"], authored, rows, strict=True):
            judges.validate_decision(row, item, packet, record, receipt_sha)
            require(
                judges._bytes_json(decision)
                == judges._bytes_json({key: row[key] for key in AUTHORED_FIELDS}),
                "authored substantive fields differ from submitted output",
            )
        require(
            before == {relative[k]: file_record(p) for k, p in paths.items()},
            "set changed during validation",
        )
        complete[relative["packet_path"]] = {"n_items": len(rows), "files": before}
    require(
        covered == set(entries) and dict(lane_counts) == EXPECTED_LANES,
        "lane assignment census drift",
    )
    validate_packet_names(names, expected_names)
    require(set(packet_pins) <= names, "prepared packet missing")
    metadata_names = names - expected_names - AGGREGATES
    metadata = {name: file_record(source / name) for name in sorted(metadata_names)}
    full = len(complete) == EXPECTED["n_packets"]
    require(
        full or not names.intersection(AGGREGATES),
        "aggregation exists before full assignment coverage",
    )
    return {
        "manifest": manifest,
        "metadata": metadata,
        "packet_pins": packet_pins,
        "expected_packet_files": expected_names,
        "complete": complete,
        "n_assignments": sum(r["n_items"] for r in complete.values()),
        "receipt_only": receipt_only,
        "authored_pending": authored_pending,
        "temporary_files": len(temporary),
        "full": full and not temporary,
    }


def recheck_source(source: Path, identity: dict, selected: dict, *, full: bool) -> None:
    """Recheck frozen inputs and selected bytes immediately before/after publication."""
    for name, expected in {**identity["metadata"], **identity["packet_pins"], **selected}.items():
        own_path(source, str(source / name))
        require(file_record(source / name) == expected, "frozen source changed before publication")
    if full:
        names, temporary = inventory(source)
        require(not temporary and names == set(selected), "full source census changed")


def guarded_upload(packed: Path, prefix: str, receipt: Path) -> dict:
    """Refuse a different published prefix, then reuse exact-byte upload/verification."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.utils import EntryNotFoundError

    from explore_persona_space.orchestrate import hub

    manifest, _ = persist.verify_archive(packed)
    expected = {p.name: persist.digest(p.read_bytes()) for p in packed.iterdir()}
    api = HfApi()
    revision = hub.retry_transient(
        lambda: api.repo_info(persist.REPO, repo_type="dataset", revision="main").sha,
        what="judge watch archive-prefix revision",
    )
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "remote guard lacks commit SHA")
    try:
        entries = hub.retry_transient(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: enclosing retry retries full lazy tree enumeration.
                api.list_repo_tree(
                    persist.REPO,
                    path_in_repo=prefix,
                    recursive=True,
                    revision=revision,
                    repo_type="dataset",
                )
            ),
            what="judge watch archive-prefix census",
        )
    except EntryNotFoundError:
        entries = []  # Explicitly absent prefix only; auth/transport errors still propagate.
    remote = {
        item.path.removeprefix(prefix + "/") for item in entries if isinstance(item, RepoFile)
    }
    if remote:
        require(remote == set(expected), "different archive already published at this prefix")
        for name, digest in expected.items():
            target = receipt.parent / "remote_guard" / revision / name
            hub.stage_hub_file(
                persist.REPO, f"{prefix}/{name}", target, repo_type="dataset", revision=revision
            )
            require(
                persist.digest(target.read_bytes()) == digest,
                "different archive already published at this prefix",
            )
        if not receipt.exists():
            # An upload may have succeeded before its local receipt was written.
            # Every remote byte was just verified; resume the existing immutable revision.
            persist.immutable_write(
                receipt,
                persist.encode(
                    {
                        "passed": True,
                        "repo": persist.REPO,
                        "prefix": prefix,
                        "revision": revision,
                        "files_sha256": expected,
                        "n_original_files": len(manifest["files"]),
                    }
                ),
            )
    result = persist.upload_archive(packed, prefix, receipt)
    require(
        result["passed"] is True
        and result["repo"] == persist.REPO
        and result["prefix"] == prefix
        and result["files_sha256"] == expected
        and result["n_original_files"] == len(manifest["files"])
        and re.fullmatch(r"[0-9a-f]{40}", result["revision"]) is not None,
        "archive uploader returned incompatible verification evidence",
    )
    return result


def publish_snapshot(source: Path, directory: Path, identity: dict, intent: dict) -> dict:
    """Resume an immutable copied snapshot, pack it, and verify its uploaded bytes."""
    require(
        intent["identity_sha256"] == persist.digest(persist.encode(identity)),
        "snapshot identity drift",
    )
    selected, full = intent["source_files"], intent["technical_complete"]
    recheck_source(source, identity, selected, full=full)
    frozen, packed = directory / "source", directory / "packed"
    for name, expected in selected.items():
        data = utf8_bytes(source / name)
        require(
            persist.digest(data) == expected["sha256"] and len(data) == expected["bytes"],
            "copy source changed",
        )
        persist.immutable_write(frozen / persist.safe_relative(name), data)
    expected_files = dict(selected)
    if not full:
        marker = persist.encode(intent)
        persist.immutable_write(frozen / "_watch_checkpoint.json", marker)
        expected_files["_watch_checkpoint.json"] = {
            "sha256": persist.digest(marker),
            "bytes": len(marker),
        }
    require(
        inventory(frozen) == (set(expected_files), set()), "frozen snapshot has unexpected residue"
    )
    archive = persist.pack_tree(frozen, packed)
    require(
        {p: {k: row[k] for k in ("sha256", "bytes")} for p, row in archive["files"].items()}
        == expected_files
        and not archive["excluded"],
        "packed snapshot differs from exact source bytes",
    )
    recheck_source(source, identity, selected, full=full)
    receipt = directory / "upload.json"
    if (directory / "done.json").exists():
        result = judges.read_json(receipt)
    else:
        result = guarded_upload(packed, intent["prefix"], receipt)
    recheck_source(source, identity, selected, full=full)
    expected_archive = {p.name: persist.digest(p.read_bytes()) for p in packed.iterdir()}
    require(
        result["passed"] is True
        and result["prefix"] == intent["prefix"]
        and result["files_sha256"] == expected_archive
        and re.fullmatch(r"[0-9a-f]{40}", result["revision"]) is not None,
        "snapshot receipt does not verify this archive",
    )
    done = {
        "contract": CONTRACT,
        "technical_complete": full,
        "intent_sha256": judges.sha_file(directory / "intent.json"),
        "archive_revision": result["revision"],
        "prefix": intent["prefix"],
        "archive_manifest_sha256": judges.sha_file(packed / "packed_manifest.json"),
        "upload_receipt_sha256": judges.sha_file(receipt),
        "source_files": selected,
        "packet_keys": intent["packet_keys"],
        "identity_sha256": intent["identity_sha256"],
    }
    persist.immutable_write(directory / "done.json", persist.encode(done))
    return done


def completed_snapshot(directory: Path, intent: dict, identity_sha: str, live: dict) -> dict:
    """Validate a verified checkpoint receipt without recopying or reuploading old packets."""
    done = judges.read_json(directory / "done.json")
    receipt = judges.read_json(directory / "upload.json")
    manifest_path = directory / "packed" / "packed_manifest.json"
    archive = judges.read_json(manifest_path)
    archive_files = {
        "packed_manifest.json": judges.sha_file(manifest_path),
        **{name: row["sha256"] for name, row in archive["shards"].items()},
    }
    require(
        done["identity_sha256"] == intent["identity_sha256"] == identity_sha
        and done["intent_sha256"] == judges.sha_file(directory / "intent.json")
        and done["upload_receipt_sha256"] == judges.sha_file(directory / "upload.json")
        and done["archive_manifest_sha256"] == archive_files["packed_manifest.json"]
        and done["source_files"] == intent["source_files"]
        and done["packet_keys"] == intent["packet_keys"]
        and done["technical_complete"] is intent["technical_complete"]
        and done["prefix"] == receipt["prefix"] == intent["prefix"]
        and receipt["files_sha256"] == archive_files
        and receipt["passed"] is True
        and receipt["repo"] == persist.REPO
        and receipt["revision"] == done["archive_revision"]
        and re.fullmatch(r"[0-9a-f]{40}", receipt["revision"]) is not None,
        "completed snapshot receipt or identity changed",
    )
    require(
        all(live.get(name) == record for name, record in intent["source_files"].items()),
        "previously backed-up source changed or disappeared",
    )
    return done


def _step(source: Path, state: Path, manifest_sha256: str, snapshot_every: int) -> dict:
    """Execute one locked validation/checkpoint/collection pass without polling."""
    require(
        source.is_dir() and not state.is_relative_to(source) and not source.is_relative_to(state),
        "state must be outside live source and not its ancestor",
    )
    if state.exists():
        require(not any(p.is_symlink() for p in state.rglob("*")), "symlink in watcher state")
    require(type(snapshot_every) is int and snapshot_every > 0, "invalid checkpoint interval")
    scan = inspect_source(source, manifest_sha256)
    identity = {
        "contract": CONTRACT,
        "source": str(source),
        "manifest_sha256": manifest_sha256,
        "expected": EXPECTED,
        "expected_lanes": EXPECTED_LANES,
        "metadata": scan["metadata"],
        "packet_pins": scan["packet_pins"],
        "prepared_backup": PREPARED_BACKUP,
    }
    persist.immutable_write(state / "identity.json", persist.encode(identity))
    identity_sha = persist.digest(persist.encode(identity))
    seen = state / "seen"
    present_seen = set()
    for key, record in scan["complete"].items():
        name = persist.digest(key.encode()) + ".json"
        present_seen.add(name)
        persist.immutable_write(seen / name, persist.encode({"packet_key": key, **record}))
    require(
        not seen.exists() or {p.name for p in seen.iterdir()} <= present_seen,
        "previously complete packet disappeared",
    )
    live = {**scan["metadata"], **scan["packet_pins"]}
    for record in scan["complete"].values():
        live.update(record["files"])
    for name in AGGREGATES:
        if (source / name).exists():
            live[name] = file_record(source / name)
    backed, published = set(), 0
    snapshots = state / "snapshots"
    for directory in sorted(snapshots.iterdir()) if snapshots.exists() else []:
        require(directory.is_dir() and not directory.is_symlink(), "invalid snapshot state")
        intent = judges.read_json(directory / "intent.json")
        fingerprint = persist.digest(
            persist.encode(
                {
                    "identity": identity_sha,
                    "files": intent["source_files"],
                    "full": intent["technical_complete"],
                }
            )
        )
        expected_name = "final" if intent["technical_complete"] else f"watch_{fingerprint}"
        expected_prefix = (
            FINAL_PREFIX if intent["technical_complete"] else f"{CHECKPOINT_PREFIX}/{expected_name}"
        )
        require(
            directory.name == expected_name and intent["prefix"] == expected_prefix,
            "snapshot prefix/name differs from frozen source identity",
        )
        if intent["technical_complete"]:
            require(scan["full"], "full snapshot no longer has full source coverage")
        was_done = (directory / "done.json").exists()
        done = (
            completed_snapshot(directory, intent, identity_sha, live)
            if was_done
            else publish_snapshot(source, directory, identity, intent)
        )
        backed.update(done["packet_keys"])
        published += int(not was_done)
    new = sorted(set(scan["complete"]) - backed)
    if scan["full"]:
        if (state / "done.json").exists():
            require(
                (snapshots / "final" / "done.json").read_bytes()
                == (state / "done.json").read_bytes(),
                "terminal watcher receipt differs from final archive",
            )
            return {
                "technical_complete": True,
                "completed_packets": len(scan["complete"]),
                "completed_assignments": scan["n_assignments"],
                "backed_packets": len(backed),
                "archives_published": published,
                "archive_revision": judges.read_json(state / "done.json")["archive_revision"],
            }
        require(scan["n_assignments"] == EXPECTED["n_assignments"], "incomplete assignment total")
        recheck_source(source, identity, live, full=False)
        summary = judges.collect(source)
        require(
            summary["technical_complete"] is True
            and summary["phase"] == EXPECTED_PHASE
            and all(summary[k] == EXPECTED[k] for k in ("n_items", "n_assignments", "n_overlap")),
            "collector summary differs from frozen production census",
        )
        recheck_source(source, identity, live, full=False)
        names, temporary = inventory(source)
        require(not temporary, "atomic writer still in flight at final collection")
        validate_packet_names(names, scan["expected_packet_files"])
        selected = {name: file_record(source / name) for name in sorted(names)}
        keys, full = sorted(scan["complete"]), True
    elif len(new) >= snapshot_every:
        selected = dict(scan["metadata"])
        for key in new:
            selected.update(scan["complete"][key]["files"])
        keys, full = new, False
    else:
        selected, keys, full = {}, [], False
    if selected:
        fingerprint = persist.digest(
            persist.encode({"identity": identity_sha, "files": selected, "full": full})
        )
        name = "final" if full else f"watch_{fingerprint}"
        directory = snapshots / name
        intent = {
            "contract": CONTRACT,
            "technical_complete": full,
            "identity_sha256": identity_sha,
            "source_files": selected,
            "packet_keys": keys,
            "n_completed_packets": len(scan["complete"]),
            "n_completed_assignments": scan["n_assignments"],
            "prefix": FINAL_PREFIX if full else f"{CHECKPOINT_PREFIX}/{name}",
            "prepared_backup": PREPARED_BACKUP,
            "scope": "complete production archive"
            if full
            else "partial backup, NOT technical_complete",
        }
        persist.immutable_write(directory / "intent.json", persist.encode(intent))
        was_done = (directory / "done.json").exists()
        done = publish_snapshot(source, directory, identity, intent)
        published += int(not was_done)
        backed.update(keys)
        if full:
            persist.immutable_write(state / "done.json", persist.encode(done))
    return {
        "technical_complete": scan["full"],
        "completed_packets": len(scan["complete"]),
        "expected_packets": EXPECTED["n_packets"],
        "completed_assignments": scan["n_assignments"],
        "expected_assignments": EXPECTED["n_assignments"],
        "backed_packets": len(backed),
        "receipt_only_packets": scan["receipt_only"],
        "authored_pending_packets": scan["authored_pending"],
        "temporary_files_pending": scan["temporary_files"],
        "archives_published": published,
        "archive_revision": judges.read_json(state / "done.json")["archive_revision"]
        if scan["full"]
        else None,
    }


def step(source: Path, state: Path, manifest_sha256: str, *, snapshot_every: int = 50) -> dict:
    """Run one nonblocking, exclusively locked watcher step; safe to resume unchanged bytes."""
    source, state = source.resolve(), state.resolve()
    with exclusive_monitor(source):
        return _step(source, state, manifest_sha256, snapshot_every)


def main() -> int:
    """Poll a bounded number of times; emit counts only, with failures propagated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=1)
    args = parser.parse_args()
    require(1 <= args.poll_seconds <= 60 and args.max_polls > 0, "invalid bounded polling settings")
    source, state = args.source.resolve(), args.state_dir.resolve()
    with exclusive_monitor(source):
        for index in range(args.max_polls):
            progress = _step(source, state, args.manifest_sha256, 50)
            print(json.dumps(progress, sort_keys=True), flush=True)
            if progress["technical_complete"]:
                return 0
            if index + 1 < args.max_polls:
                time.sleep(args.poll_seconds)
    return 0  # Bounded poll finished; stdout explicitly states technical_complete=false.


if __name__ == "__main__":
    raise SystemExit(main())
