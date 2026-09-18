"""Preserve terminal #1739 residue without re-uploading the primary tensors.

Usage: issue1739_natural_closeout.py OUT_ROOT TERMINAL_PREFIX [--log PATH]
The completed primary snapshot is independently checked first. New/changed
out-root files and the complete optional workload log upload in one folder
commit, followed by exact-name/hash verification. No file is written beneath
OUT_ROOT. The only final proof is JSON on stdout: the coordinator must persist
it durably before teardown. Progress and Hub output go to stderr.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()
from explore_persona_space.orchestrate import hub  # noqa: E402

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".txt",
    ".log",
    ".md",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".out",
    ".err",
    ".pid",
}
TEXT_THRESHOLD = 9_000_000
PART_LIMIT = 8_900_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    path = PurePosixPath(name)
    require(
        isinstance(name, str)
        and name
        and path.parts
        and not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == name,
        f"Unsafe artifact path: {name!r}",
    )
    return name


def stamp(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def digest_file(path):
    """One streaming read computes content and Git-blob hashes, checking races."""
    before = stamp(path)
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {before[2]}\0".encode())
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            sha256.update(block)
            blob.update(block)
            size += len(block)
    require(before == stamp(path) and size == before[2], f"File changed while hashing: {path}")
    return {
        "size": size,
        "sha256": sha256.hexdigest(),
        "git_blob_sha1": blob.hexdigest(),
        "stamp": before,
    }


def files_under(root):
    require(root.is_dir() and not root.is_symlink(), f"Not a plain directory: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Symlink cannot be silently preserved: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"Unknown non-regular artifact: {path}")
        result[safe_name(path.relative_to(root).as_posix())] = path
    require(result, f"Empty artifact tree: {root}")
    return result


def assert_snapshot_unchanged(root, paths, snapshot):
    require(set(files_under(root)) == set(paths), "Out-root name set changed during closeout")
    for name, path in paths.items():
        require(stamp(path) == snapshot[name]["stamp"], f"Out-root file changed: {name}")


def assert_worker_finished(root, source_sha):
    """A reused PID is harmless; the original still-running worker is not."""
    path = root / "worker.pid"
    if not path.exists():
        return
    worker = json.loads(path.read_text())
    require(worker["source_sha"] == source_sha, "Worker PID source mismatch")
    pid = int(worker["pid"])
    require(pid > 1, "Invalid worker PID")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return
    original_alive = int(fields[19]) == int(worker["start_ticks"]) and fields[0] not in {"Z", "X"}
    require(
        not original_alive, "Original worker is still alive; closeout requires its completed exit"
    )


class Runtime:
    """Only scoped, retried Hub operations; test seams perform no live uploads."""

    def __init__(self):
        from huggingface_hub import HfApi

        self.api = HfApi()

    def listing(self, repo, prefix, revision):
        values = hub.retry_transient(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: the complete lazy listing is materialized inside retry_transient.
                self.api.list_repo_tree(
                    repo,
                    path_in_repo=prefix,
                    revision=revision,
                    repo_type="dataset",
                    recursive=True,
                )
            ),
            what=f"natural closeout verify {prefix}@{revision}",
        )
        entries = {}
        for value in values:
            if not hasattr(value, "size"):
                continue
            require(value.path.startswith(prefix + "/"), "Unscoped Hub listing entry")
            name = safe_name(value.path[len(prefix) + 1 :])
            require(name not in entries, f"Duplicate Hub path: {name}")
            entries[name] = value
        require(entries, f"Empty remote artifact prefix: {prefix}")
        return entries

    def stage(self, repo, remote_name, revision, local_path):
        return hub.stage_hub_file(
            repo, remote_name, local_path, repo_type="dataset", revision=revision
        )

    def upload(self, repo, prefix, folder):
        # One folder-level transaction covers every residue class and log part.
        hub.assert_upload_clean([folder], what="#1739 terminal residue and workload log")
        result = hub.retry_transient(
            lambda: self.api.upload_folder(
                repo_id=repo,
                repo_type="dataset",
                path_in_repo=prefix,
                folder_path=str(folder),
                commit_message="#1739 verified terminal preservation",
            ),
            what=f"natural closeout upload {prefix}",
        )
        return result.oid


def verify_entry(entry, expected, name):
    require(entry.size == expected["size"], f"Remote size mismatch: {name}")
    if entry.lfs is not None:
        require(entry.lfs.sha256 == expected["sha256"], f"Remote LFS SHA mismatch: {name}")
    else:
        require(entry.blob_id == expected["git_blob_sha1"], f"Remote Git-blob mismatch: {name}")


def preserve_file(source, destination, expected):
    """Copy exact bytes, or line-shard large text in the existing manifest schema."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"Terminal destination collision: {destination}")
    if source.suffix.lower() not in TEXT_SUFFIXES or expected["size"] < TEXT_THRESHOLD:
        shutil.copyfile(source, destination)
        actual = digest_file(destination)
        require(actual["sha256"] == expected["sha256"], f"Copied bytes differ: {source}")
        return {"form": "file", "path": destination.name}

    parts, hashes, sizes = [], {}, {}
    whole = hashlib.sha256()
    pending = bytearray()

    def emit():
        name = f"{destination.stem}.part{len(parts):05d}{destination.suffix}"
        path = destination.with_name(name)
        require(not path.exists(), f"Terminal shard collision: {path}")
        path.write_bytes(pending)
        parts.append(name)
        hashes[name] = hashlib.sha256(pending).hexdigest()
        sizes[name] = len(pending)
        pending.clear()

    with source.open("rb") as stream:
        while line := stream.readline(PART_LIMIT + 1):
            require(len(line) < PART_LIMIT, f"Single text line exceeds shard ceiling: {source}")
            if pending and len(pending) + len(line) >= PART_LIMIT:
                emit()
            pending.extend(line)
            whole.update(line)
        if pending:
            emit()
    require(parts and whole.hexdigest() == expected["sha256"], f"Sharded bytes differ: {source}")
    manifest = destination.with_name(destination.stem + ".manifest.json")
    require(not manifest.exists(), f"Terminal manifest collision: {manifest}")
    payload = {
        "source": source.name,
        "source_sha256": expected["sha256"],
        "source_bytes": expected["size"],
        "parts": parts,
        "sha256": hashes,
        "bytes": sizes,
        "reconstruction": "concatenate parts in manifest order",
    }
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return {"form": "sharded_text", "manifest": manifest.name, "parts": parts}


def relative_home(home, parent):
    result = dict(home)
    for key in ("path", "manifest"):
        if key in result:
            result[key] = (parent / result[key]).as_posix()
    if "parts" in result:
        result["parts"] = [(parent / name).as_posix() for name in result["parts"]]
    return result


def closeout(out_root, terminal_prefix, *, log=None, runtime=None, temp_parent=None):
    """Return a complete current-byte permanent-home proof; write nothing to out_root."""
    root = Path(out_root).absolute()
    require(not root.is_symlink(), "Out-root must not be a symlink")
    terminal_prefix = safe_name(terminal_prefix.rstrip("/"))
    paths = files_under(root)
    snapshot = {name: digest_file(path) for name, path in paths.items()}
    config = json.loads(paths["run_config.json"].read_text())
    uploaded = json.loads(paths["upload_verified.json"].read_text())
    completed = json.loads(paths["run_complete.json"].read_text())
    source, fingerprint = config["source_sha"], config["input_fingerprint"]
    require(
        HEX40.fullmatch(source) and HEX64.fullmatch(fingerprint), "Invalid source/input identity"
    )
    require(
        all(doc.get("source_sha") == source for doc in (uploaded, completed)),
        "Closeout source identity mismatch",
    )
    require(completed.get("input_fingerprint") == fingerprint, "Completion input mismatch")
    require(
        uploaded.get("phase") == "complete" and completed.get("status") == "complete",
        "Primary run is not complete",
    )
    primary_prefix, revision = uploaded["prefix"], uploaded["verified_revision"]
    safe_name(primary_prefix)
    require(primary_prefix == config["upload_prefix"], "Primary prefix differs from run config")
    require(
        HEX40.fullmatch(revision) and completed["verified_revision"] == revision,
        "Primary revision mismatch",
    )
    require(
        not (
            terminal_prefix == primary_prefix
            or terminal_prefix.startswith(primary_prefix + "/")
            or primary_prefix.startswith(terminal_prefix + "/")
        ),
        "Terminal and primary prefixes must not overlap",
    )
    declared = uploaded["sha256"]
    require(len(declared) == uploaded["n_files"] and declared, "Invalid primary file manifest")
    for name, value in declared.items():
        safe_name(name)
        require(HEX64.fullmatch(value), f"Invalid primary digest: {name}")
    require(set(declared) <= set(paths), "Previously declared primary files disappeared locally")
    assert_worker_finished(root, source)
    if log is not None:
        log = Path(log).absolute()
        require(log.is_file() and not log.is_symlink(), "Workload log must be a regular file")
        log_digest = digest_file(log)
    else:
        log_digest = None
    repo = hub.DEFAULT_DATASET_REPO
    runtime = runtime or Runtime()
    parent = Path(temp_parent) if temp_parent is not None else Path("/workspace")
    require(
        parent.is_dir() and not parent.resolve().is_relative_to(root.resolve()),
        "Temporary parent must exist outside the original out-root",
    )
    with tempfile.TemporaryDirectory(prefix="issue1739-closeout-", dir=parent) as temp:
        scratch = Path(temp)
        folder = scratch / "terminal"
        folder.mkdir()
        original = runtime.listing(repo, primary_prefix, revision)
        require(set(original) == set(declared), "Original immutable HF name-set mismatch")
        homes = {}
        for number, (name, expected_sha) in enumerate(sorted(declared.items()), 1):
            entry = original[name]
            if snapshot[name]["sha256"] == expected_sha:
                verify_entry(entry, snapshot[name], name)
                homes[name] = {"snapshot": "primary", "form": "file", "path": name}
            elif entry.lfs is not None:
                require(entry.lfs.sha256 == expected_sha, f"Original LFS digest differs: {name}")
            else:
                old_path = scratch / "primary-audit" / name
                old_path.parent.mkdir(parents=True, exist_ok=True)
                runtime.stage(repo, primary_prefix + "/" + name, revision, old_path)
                old = digest_file(old_path)
                require(old["sha256"] == expected_sha, f"Original Git content differs: {name}")
                verify_entry(entry, old, name)
            if number % 50 == 0:
                print(f"[closeout] primary verified {number}/{len(declared)}", file=sys.stderr)
        for name in sorted(set(paths) - set(homes)):
            destination = folder / "outroot" / name
            home = preserve_file(paths[name], destination, snapshot[name])
            homes[name] = {
                "snapshot": "terminal",
                **relative_home(home, destination.parent.relative_to(folder)),
            }
        log_home = None
        if log is not None:
            destination = folder / "logs" / "workload.log"
            log_home = {
                "snapshot": "terminal",
                **relative_home(preserve_file(log, destination, log_digest), Path("logs")),
            }
        manifest = {
            "source_sha": source,
            "input_fingerprint": fingerprint,
            "primary_revision": revision,
            "primary_prefix": primary_prefix,
            "terminal_prefix": terminal_prefix,
            "out_root": str(root),
            "files": {
                name: {"sha256": row["sha256"], "size": row["size"], "home": homes[name]}
                for name, row in snapshot.items()
            },
            "workload_log": None
            if log is None
            else {
                "source_path": str(log),
                "sha256": log_digest["sha256"],
                "size": log_digest["size"],
                "home": log_home,
            },
            "closeout_script_sha256": digest_file(Path(__file__))["sha256"],
        }
        manifest_path = folder / "closeout_manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        require(manifest_path.stat().st_size < TEXT_THRESHOLD, "Closeout manifest needs sharding")
        assert_snapshot_unchanged(root, paths, snapshot)
        terminal_files = files_under(folder)
        terminal_digests = {name: digest_file(path) for name, path in terminal_files.items()}
        print(
            f"[closeout] uploading {len(terminal_files)} terminal files, "
            f"{sum(v['size'] for v in terminal_digests.values())} bytes",
            file=sys.stderr,
        )
        terminal_revision = runtime.upload(repo, terminal_prefix, folder)
        require(HEX40.fullmatch(terminal_revision), "Upload did not return an immutable commit")
        terminal_remote = runtime.listing(repo, terminal_prefix, terminal_revision)
        require(set(terminal_remote) == set(terminal_files), "Terminal HF name-set mismatch")
        for name, expected in terminal_digests.items():
            verify_entry(terminal_remote[name], expected, name)
        assert_snapshot_unchanged(root, paths, snapshot)
        if log is not None:
            require(stamp(log) == log_digest["stamp"], "Workload log changed during closeout")

        def pinned(home):
            is_primary = home["snapshot"] == "primary"
            prefix = primary_prefix if is_primary else terminal_prefix
            result = {
                **home,
                "repo": repo,
                "revision": revision if is_primary else terminal_revision,
            }
            for key in ("path", "manifest"):
                if key in result:
                    result[key] = prefix + "/" + result[key]
            if "parts" in result:
                result["parts"] = [prefix + "/" + part for part in result["parts"]]
            return result

        return {
            "status": "verified",
            "source_sha": source,
            "input_fingerprint": fingerprint,
            "checked_at": time.time(),
            "primary_revision": revision,
            "terminal_revision": terminal_revision,
            "repo": repo,
            "primary_prefix": primary_prefix,
            "terminal_prefix": terminal_prefix,
            "closeout_manifest_sha256": terminal_digests["closeout_manifest.json"]["sha256"],
            "outroot_files": {
                name: {"sha256": row["sha256"], "size": row["size"], "home": pinned(homes[name])}
                for name, row in snapshot.items()
            },
            "workload_log": None
            if log is None
            else {
                "sha256": log_digest["sha256"],
                "size": log_digest["size"],
                "home": pinned(log_home),
            },
            "terminal_uploaded_files": {
                name: row["sha256"] for name, row in terminal_digests.items()
            },
            "proof_persistence": "coordinator must persist this stdout JSON before teardown",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_root", nargs="?")
    parser.add_argument("terminal_prefix", nargs="?")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--import-check", action="store_true")
    args = parser.parse_args()
    if args.import_check:
        from huggingface_hub import HfApi  # noqa: F401

        print("natural closeout imports OK; no reads or uploads performed")
        return 0
    if not args.out_root or not args.terminal_prefix:
        parser.error("OUT_ROOT and TERMINAL_PREFIX are required")
    with redirect_stdout(sys.stderr):
        proof = closeout(args.out_root, args.terminal_prefix, log=args.log)
    print(json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
