"""Selectively stage pinned #1739 covariance-ablation inputs without a full tar on disk.

The original labeling archives are streamed with the reviewed parallel Range
reader. Each selected member is SHA-checked against the later preservation
inventory and published atomically. Complete local members are reusable.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
REVISION = "7a47ff5ce42f16308bebaba29c1286a4e9bc8008"
INVENTORY_REVISION = "3f864b4053d5f95c5cafe4869e455d74278416f9"
INVENTORY_PATH = (
    "issue1739_natural100k_20260906/final_preservation_20260908/"
    "snapshot/source_inventory/part_00000.jsonl"
)
INVENTORY_SHA = "a3f7a24a4ccd83e1e4ad95dbf8d6ce692f3a8758a706ec0f3719d4fe9c4780ca"
LAYERS = {"evil": (18, 20), "sycophancy": (19, 20), "hallucination": (20,)}
KINDS = ("context_end", "t1")
ALL_LAYERS = (18, 19, 20)


def sha256(path: Path) -> str:
    """Hash a file with bounded temporary memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Publish a JSON record atomically on its destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Progress:
    """Publish a timestamp plus real stream progress every 30 seconds."""

    def __init__(self, path: Path):
        self.path = path
        self.state = {"phase": "initializing", "bytes_fetched": 0, "files_completed": 0}
        self.reader = None
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(self._heartbeat)

    def _heartbeat(self) -> None:
        """Retain errors in the Future so the main worker cannot lose them."""
        while not self.stop.wait(30):
            self.emit()

    def check(self) -> None:
        """Raise immediately if the progress publisher has failed."""
        if self.future.done():
            self.future.result()

    def emit(self, **updates) -> None:
        """Include completed HTTP bytes even while skipping unwanted members."""
        with self.lock:
            self.state.update(updates)
            if self.reader is not None:
                self.state["bytes_fetched"] = self.reader.bytes_fetched
            record = {"time": time.time(), **self.state}
            write_json(self.path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

    def close(self) -> None:
        """Stop the publisher and propagate any error before returning."""
        self.stop.set()
        self.future.result()
        self.pool.shutdown(wait=True)


def selected_name(name: str, layers: tuple[int, ...], kinds: tuple[str, ...]) -> bool:
    """Select exact summary coordinates and the row metadata the loader needs."""
    if re.fullmatch(r"row_index(?:_shard\d+)?\.jsonl", name):
        return True
    match = re.fullmatch(r"([a-z0-9_]+)_L(\d+)(?:_shard\d+)?\.npy", name)
    return bool(match and match[1] in kinds and int(match[2]) in layers)


def load_inventory(dest: Path, token: str) -> list[dict]:
    """Stage and verify the immutable member-level preservation inventory."""
    path = hub.stage_hub_file(
        REPO,
        INVENTORY_PATH,
        dest / "provenance" / "source_inventory.jsonl",
        repo_type="dataset",
        revision=INVENTORY_REVISION,
        token=token or None,
    )
    if sha256(path) != INVENTORY_SHA:
        raise ValueError("preservation inventory SHA256 mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_members(inventory: list[dict], behavior: str) -> dict[str, dict]:
    """Read the complete selected archive member set from verified metadata."""
    prefix = f"run/reused/store/{behavior}_labeling/"
    entries = {
        row["key"][len(prefix) :]: row
        for row in inventory
        if row["key"].startswith(prefix)
        and selected_name(row["key"][len(prefix) :], LAYERS[behavior], KINDS)
    }
    for kind in KINDS:
        for layer in LAYERS[behavior]:
            if not any(re.match(rf"{kind}_L{layer:02d}(?:_|\.)", name) for name in entries):
                raise ValueError(f"inventory lacks {behavior} {kind} layer {layer}")
    if not any(name.startswith("row_index") for name in entries):
        raise ValueError(f"inventory lacks row metadata for {behavior}")
    return entries


def verified_file(path: Path, expected: dict) -> bool:
    """Accept only a complete file whose content matches the pinned inventory."""
    if not path.exists():
        return False
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"existing staged input disagrees with pinned source: {path}")
    return True


def extract_members(archive, expected: dict[str, dict], dest: Path, progress: Progress) -> int:
    """Verify selected tar members before rename; stop after the exact set is complete."""
    seen = set()
    for member in archive:
        progress.check()
        if not member.isfile():
            continue
        name = Path(member.name).name
        if name not in expected:
            continue
        if name in seen:
            raise ValueError(f"duplicate selected tar basename: {name}")
        seen.add(name)
        record = expected[name]
        if member.size != record["bytes"]:
            raise ValueError(f"archive member size mismatch: {name}")
        path = dest / name
        if not verified_file(path, record):
            temporary = path.with_name(f".{name}.{os.getpid()}.partial")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read regular tar member: {member.name}")
            digest = hashlib.sha256()
            copied = 0
            with source, temporary.open("wb") as target:
                for block in iter(lambda: source.read(8 << 20), b""):
                    target.write(block)
                    digest.update(block)
                    copied += len(block)
            if copied != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"archive member content mismatch: {name}")
            temporary.replace(path)
        progress.emit(files_completed=len(seen), files_total=len(expected), latest_member=name)
        if len(seen) == len(expected):
            break
    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f"archive ended without {len(missing)} required members: {missing[:5]}")
    return len(seen)


def stage_behavior(args, behavior: str, inventory: list[dict], progress: Progress, token: str):
    """Stream one original archive; keep only required, independently hashed inputs."""
    from scripts.issue1739_map963k_slice import ParallelRangeReader, head_size, tar_url

    expected = expected_members(inventory, behavior)
    dest = args.dest / f"{behavior}_labeling"
    dest.mkdir(parents=True, exist_ok=True)
    if all(verified_file(dest / name, record) for name, record in expected.items()):
        progress.emit(phase=f"{behavior}_verified_resume", files_completed=len(expected))
        return
    url = tar_url(behavior, args.revision)
    total = head_size(url, token)
    progress.emit(phase=f"{behavior}_stream", bytes_fetched=0, total=total, files_completed=0)
    raw = ParallelRangeReader(url, token=token, total=total, workers=args.workers)
    progress.reader = raw
    try:
        with io.BufferedReader(raw, buffer_size=8 << 20) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                extract_members(archive, expected, dest, progress)
    finally:
        progress.emit()
        progress.reader = None
        raw.close()
    write_json(
        dest / "slice_manifest.json",
        {
            "revision": args.revision,
            "behavior": behavior,
            "layers": list(LAYERS[behavior]),
            "kinds": list(KINDS),
            "members": {
                name: {k: r[k] for k in ("bytes", "sha256")} for name, r in expected.items()
            },
            "bytes_fetched": raw.bytes_fetched,
            "source_archive_bytes": total,
            "complete": True,
        },
    )


def scoped_entries(prefix: str, revision: str, token: str):
    """Retry complete listing materialization, including lazy pagination failures."""
    from huggingface_hub import HfApi

    api = HfApi(token=token or None)
    return hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: full lazy listing is materialized inside retry_transient
            api.list_repo_tree(
                REPO, path_in_repo=prefix, repo_type="dataset", revision=revision, recursive=False
            )
        ),
        what=f"covariance input listing {prefix}@{revision}",
    )


def stage_wildchat(args, inventory: list[dict], progress: Progress, token: str):
    """Stage context-only WildChat summaries at the immutable historical input pin."""
    prefix = "issue1739_ctxmap/wildchat_rung/capture_store/wildchat"
    inventory_prefix = "run/reused/store/wcrung_capture_store/wildchat/"
    expected = {
        row["key"][len(inventory_prefix) :]: row
        for row in inventory
        if row["key"].startswith(inventory_prefix)
        and selected_name(row["key"][len(inventory_prefix) :], ALL_LAYERS, ("context_end",))
    }
    entries = [
        entry
        for entry in scoped_entries(prefix, args.revision, token)
        if hasattr(entry, "size")
        and selected_name(Path(entry.path).name, ALL_LAYERS, ("context_end",))
    ]
    if not expected or {Path(entry.path).name for entry in entries} != set(expected):
        raise ValueError("WildChat source member set disagrees with preservation inventory")
    progress.emit(
        phase="wildchat_stage", bytes_fetched=0, files_completed=0, files_total=len(entries)
    )
    for i, entry in enumerate(entries, 1):
        progress.check()
        path = args.dest / "wildchat" / Path(entry.path).name
        record = expected[path.name]
        if not verified_file(path, record):
            hub.stage_hub_file(
                REPO,
                entry.path,
                path,
                repo_type="dataset",
                revision=args.revision,
                token=token or None,
            )
            if not verified_file(path, record):
                raise RuntimeError(f"WildChat stage did not create {path}")
        progress.emit(files_completed=i, latest_member=path.name)


def verify_generic_store(dest: Path, token: str) -> None:
    """Check generic array and row-manifest content against their pinned Hub objects."""
    from explore_persona_space.experiments.issue_1739.constants import (
        CORPUS_MANIFEST_PATH,
        CORPUS_MANIFEST_REVISION,
        STORE_PREFIX,
        STORE_REVISION,
        U_STORE_CELL,
    )

    checks = []
    prefix = f"{STORE_PREFIX.rstrip('/')}/{U_STORE_CELL}"
    for entry in scoped_entries(prefix, STORE_REVISION, token):
        if hasattr(entry, "size") and selected_name(Path(entry.path).name, ALL_LAYERS, KINDS):
            checks.append((entry, dest / Path(entry.path).name))
    if len(checks) != len(KINDS) * len(ALL_LAYERS):
        raise ValueError("generic U source does not contain exactly the six expected arrays")
    for entry in scoped_entries(
        str(Path(CORPUS_MANIFEST_PATH).parent), CORPUS_MANIFEST_REVISION, token
    ):
        if entry.path == CORPUS_MANIFEST_PATH:
            checks.append((entry, dest / "manifest.jsonl"))
    if len(checks) != 7:
        raise ValueError("generic U pinned corpus manifest was not found")
    for entry, path in checks:
        if path.stat().st_size != entry.size:
            raise ValueError(f"generic source size mismatch: {path}")
        if entry.lfs is not None:
            matches = sha256(path) == entry.lfs.sha256
        else:
            digest = hashlib.sha1(f"blob {entry.size}\0".encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            matches = digest.hexdigest() == entry.blob_id
        if not matches:
            raise ValueError(f"generic source content mismatch: {path}")


def main():
    """Stage only the requested saved inputs and publish their complete hash manifest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--behaviors", nargs="+", choices=list(LAYERS), default=list(LAYERS))
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        return
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("--revision must be an immutable full 40-hex revision")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    args.dest.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.progress)
    try:
        inventory = load_inventory(args.dest, token)
        for behavior in args.behaviors:
            stage_behavior(args, behavior, inventory, progress, token)
        stage_wildchat(args, inventory, progress, token)
        progress.emit(phase="generic_u_stage", bytes_fetched=0, files_completed=0)
        from explore_persona_space.experiments.issue_1739 import store_io
        from explore_persona_space.experiments.issue_1739.constants import (
            CORPUS_MANIFEST_REVISION,
            STORE_REVISION,
        )

        store_io.stage_u_store(args.dest / "u_store", KINDS, ALL_LAYERS)
        verify_generic_store(args.dest / "u_store", token)
        progress.emit(phase="hash_inputs")
        files = {}
        for path in sorted(args.dest.rglob("*")):
            if path.is_file() and not path.name.startswith(".") and path.name != "manifest.json":
                files[str(path.relative_to(args.dest))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
        if "u_store/manifest.jsonl" not in files:
            raise ValueError("generic corpus manifest is missing")
        write_json(
            args.dest / "manifest.json",
            {
                "repo": REPO,
                "revision": args.revision,
                "u_revision": STORE_REVISION,
                "u_manifest_revision": CORPUS_MANIFEST_REVISION,
                "member_hash_inventory_revision": INVENTORY_REVISION,
                "behaviors": args.behaviors,
                "layers": {b: list(LAYERS[b]) for b in args.behaviors},
                "files": files,
                "total_bytes": sum(r["bytes"] for r in files.values()),
                "complete": True,
            },
        )
        progress.emit(
            phase="complete", files_completed=len(files), manifest=str(args.dest / "manifest.json")
        )
    finally:
        progress.close()


if __name__ == "__main__":
    main()
