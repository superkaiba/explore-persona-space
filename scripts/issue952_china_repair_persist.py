"""Archive many judge text artifacts while preserving every original byte/hash.

The archive uses bounded JSONL shards, not the older JSON reserialization packer:
judge packet/receipt lineage depends on original whitespace as well as content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue952_position_divergence/followups/china_refusal_wording_withholding_v2"
FORMAT = "exact-utf8-text-archive-v1"
SHARD_BYTES = 9_000_000
CHUNK_CHARS = 128_000


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def safe_relative(name: str) -> Path:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"unsafe/noncanonical archive path: {name!r}")
    return Path(*path.parts)


def immutable_write(path: Path, data: bytes) -> None:
    """Atomically publish a new exact-byte artifact; never overwrite differing data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing artifact symlink: {path}")
    descriptor, temp_name = tempfile.mkstemp(prefix=".archive-", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"refusing to overwrite different artifact: {path}") from None
    finally:
        temporary.unlink()


def pack_tree(
    source_root: Path, out_dir: Path, include_dirs: list[str] | None = None
) -> dict:
    """Pack a frozen UTF-8 tree (optionally named subtrees), retaining exact bytes."""
    source_root, out_dir = Path(source_root).resolve(), Path(out_dir).resolve()
    if out_dir.is_relative_to(source_root):
        raise ValueError("archive output must be outside the source tree")
    roots = [source_root / safe_relative(x) for x in include_dirs] if include_dirs else [source_root]
    selected: set[Path] = set()
    excluded = []
    for root in roots:
        relative = root.relative_to(source_root)
        components = [source_root.joinpath(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1)]
        if not root.resolve().is_relative_to(source_root) or any(p.is_symlink() for p in components):
            raise ValueError(f"source traverses symlink or escapes source tree: {root}")
        if not root.exists() or root.is_symlink():
            raise ValueError(f"source missing or symlink: {root}")
        for path in ([root] if root.is_file() else sorted(root.rglob("*"))):
            if path.is_symlink():
                raise ValueError(f"source symlink: {path}")
            if not path.is_file():
                continue
            name = path.relative_to(source_root).as_posix()
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                excluded.append({"path": name, "reason": "regenerable Python bytecode"})
            else:
                selected.add(path)
    if not selected:
        raise ValueError("empty source census")
    records, shards, files = bytearray(), {}, {}

    def flush() -> None:
        if records:
            name = f"archive.part{len(shards):04d}.jsonl"
            payload = bytes(records)
            immutable_write(out_dir / name, payload)
            shards[name] = {"sha256": digest(payload), "bytes": len(payload)}
            records.clear()

    for path in sorted(selected):
        name = path.relative_to(source_root).as_posix()
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        count = max(1, (len(text) + CHUNK_CHARS - 1) // CHUNK_CHARS)
        files[name] = {"sha256": digest(payload), "bytes": len(payload), "chunks": count}
        for index in range(count):
            record = encode({"path": name, "chunk": index, "chunks": count,
                             "text": text[index * CHUNK_CHARS:(index + 1) * CHUNK_CHARS]})
            if len(record) > SHARD_BYTES:
                raise ValueError(f"single archive record exceeds byte cap: {name}")
            if len(records) + len(record) > SHARD_BYTES:
                flush()
            records.extend(record)
    flush()
    manifest = {"format": FORMAT, "source_root": str(source_root), "files": files,
                "shards": shards, "excluded": sorted(excluded, key=lambda row: row["path"])}
    immutable_write(out_dir / "packed_manifest.json", encode(manifest))
    verify_archive(out_dir)
    return manifest


def verify_archive(packed_dir: Path) -> tuple[dict, dict[str, bytes]]:
    """Validate the complete shard census and reconstruct byte-exact source files."""
    packed_dir = Path(packed_dir)
    manifest = json.loads((packed_dir / "packed_manifest.json").read_bytes())
    if manifest["format"] != FORMAT or not manifest["files"] or not manifest["shards"]:
        raise ValueError("invalid archive format or empty census")
    expected = {"packed_manifest.json", *manifest["shards"]}
    actual = {p.name for p in packed_dir.iterdir()}
    if actual != expected:
        raise ValueError("archive shard census differs from manifest")
    pieces: dict[str, dict[int, str]] = {name: {} for name in manifest["files"]}
    for name, entry in manifest["shards"].items():
        path = packed_dir / safe_relative(name)
        if path.is_symlink():
            raise ValueError("archive shard is a symlink")
        payload = path.read_bytes()
        if len(payload) != entry["bytes"] or digest(payload) != entry["sha256"]:
            raise ValueError(f"archive shard hash/length mismatch: {name}")
        if len(payload) > SHARD_BYTES:
            raise ValueError("archive shard exceeds cap")
        for line in payload.splitlines():
            record = json.loads(line)
            key, index, count = record["path"], record["chunk"], record["chunks"]
            safe_relative(key)
            if key not in pieces or type(index) is not int or not isinstance(record["text"], str):
                raise ValueError("unknown file or invalid archive record")
            if count != manifest["files"][key]["chunks"] or not 0 <= index < count:
                raise ValueError("invalid chunk count or index")
            if index in pieces[key]:
                raise ValueError("duplicate archive chunk")
            pieces[key][index] = record["text"]
    reconstructed = {}
    for name, entry in manifest["files"].items():
        safe_relative(name)
        if set(pieces[name]) != set(range(entry["chunks"])):
            raise ValueError(f"missing chunks: {name}")
        payload = "".join(pieces[name][i] for i in range(entry["chunks"])).encode("utf-8")
        if len(payload) != entry["bytes"] or digest(payload) != entry["sha256"]:
            raise ValueError(f"reconstructed source hash/length mismatch: {name}")
        reconstructed[name] = payload
    return manifest, reconstructed


def unpack_tree(packed_dir: Path, target_root: Path) -> dict:
    """Validate fully before restoring; existing different files are an error."""
    manifest, files = verify_archive(packed_dir)
    target_root = Path(target_root).resolve()
    for name, payload in files.items():
        target = target_root / safe_relative(name)
        if not target.resolve().is_relative_to(target_root) or target.is_symlink():
            raise ValueError(f"restore would traverse a symlink: {name}")
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite different artifact: {target}")
    for name, payload in files.items():
        immutable_write(target_root / name, payload)
    return manifest


def upload_archive(packed_dir: Path, remote_subpath: str, receipt: Path) -> dict:
    """Publish within the repaired namespace and verify every remote byte at its SHA."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    packed_dir, receipt = Path(packed_dir).resolve(), Path(receipt).resolve()
    safe_relative(remote_subpath)
    if not remote_subpath.startswith(PREFIX + "/") or receipt.is_relative_to(packed_dir):
        raise ValueError("upload must use repaired child prefix and external receipt")
    manifest, _ = verify_archive(packed_dir)
    files = {p.name: digest(p.read_bytes()) for p in sorted(packed_dir.iterdir())}
    prior = json.loads(receipt.read_bytes()) if receipt.exists() else None
    if prior is not None and (prior["files_sha256"] != files or prior["prefix"] != remote_subpath):
        raise ValueError("refusing to mutate an already verified archive")
    if prior is None:
        hub.assert_hub_dir_filecounts(packed_dir, remote_subpath, allow_patterns=list(files))
        commit = hub.retry_transient(lambda: HfApi().upload_folder(
            repo_id=REPO, repo_type="dataset", folder_path=str(packed_dir),
            path_in_repo=remote_subpath, allow_patterns=list(files),
            commit_message="Issue 952 exact-byte repaired judge archive",
        ), what="issue952 exact-byte text archive upload")
        revision = commit.oid
    else:
        revision = prior["revision"]
    if not revision or revision == "main":
        raise ValueError("archive upload lacks immutable revision")
    for name, expected in files.items():
        destination = receipt.parent / "archive_verification" / revision / name
        hub.stage_hub_file(REPO, f"{remote_subpath}/{name}", destination,
                           repo_type="dataset", revision=revision)
        if digest(destination.read_bytes()) != expected:
            raise ValueError(f"remote archive bytes differ: {name}")
    result = {"passed": True, "repo": REPO, "prefix": remote_subpath, "revision": revision,
              "files_sha256": files, "n_original_files": len(manifest["files"])}
    immutable_write(receipt, encode(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pack", "unpack", "upload"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--include-dirs", nargs="+")
    parser.add_argument("--remote-subpath")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "pack":
        if args.source_root is None:
            parser.error("pack requires --source-root")
        result = pack_tree(args.source_root, args.packed_dir, args.include_dirs)
        print(json.dumps({"n_files": len(result["files"]), "n_shards": len(result["shards"])}))
    elif args.phase == "unpack":
        if args.target_root is None:
            parser.error("unpack requires --target-root")
        result = unpack_tree(args.packed_dir, args.target_root)
        print(json.dumps({"n_files": len(result["files"])}))
    else:
        if args.remote_subpath is None or args.receipt is None:
            parser.error("upload requires --remote-subpath and --receipt")
        print(json.dumps(upload_archive(args.packed_dir, args.remote_subpath, args.receipt)))


if __name__ == "__main__":
    main()
