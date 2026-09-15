"""Persist K5 pilot artifacts and stage pinned input shards with content checks."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.orchestrate.hub import (  # noqa: E402
    _upload_folder_filtered,
    retry_transient,
)

DATA_REPO = "superkaiba1/explore-persona-space-data"
TENSOR_REPO = "superkaiba1/explore-persona-space-overflow"


def digest(path: Path) -> str:
    """Compute a streaming SHA256 digest for immutable artifact verification."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload(root: Path, prefix: str, kind: str, receipt: Path) -> None:
    """Upload exact text or tensor file sets and verify every Hub content hash."""
    suffixes = {".json", ".jsonl", ".txt", ".log"} if kind == "text" else {".npz", ".npy"}
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)
    if not files:
        raise ValueError(f"no {kind} artifacts below {root}")
    if receipt.resolve().is_relative_to(root.resolve()):
        raise ValueError("archive receipt must live outside the archived tree")
    if kind == "text" and any(path.stat().st_size >= 9_000_000 for path in files):
        raise ValueError("text artifact exceeds 9 MB: shard before uploading")
    repo, repo_type = (DATA_REPO, "dataset") if kind == "text" else (TENSOR_REPO, "model")
    api = HfApi()
    if kind == "tensors" and not retry_transient(
        lambda: api.repo_info(repo, repo_type=repo_type).private,
        what="verify tensor archive privacy",
    ):
        raise RuntimeError("tensor overflow repository must be private")
    expected = {
        f"{prefix}/{p.relative_to(root).as_posix()}": {
            "local": p.relative_to(root).as_posix(),
            "sha256": digest(p),
            "size": p.stat().st_size,
        }
        for p in files
    }
    landed = _upload_folder_filtered(
        root,
        repo,
        repo_type,
        prefix,
        allow_patterns=[p.relative_to(root).as_posix() for p in files],
        expected_repo_paths=list(expected),
        delete_after=False,
        private=kind == "tensors",
    )
    if landed != f"{repo}/{prefix}":
        raise RuntimeError(f"unexpected archive destination: {landed}")
    revision = retry_transient(
        lambda: api.repo_info(repo, repo_type=repo_type).sha, what="archive revision"
    )
    entries = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: retry reopens and consumes the full paginated iterator.
            api.list_repo_tree(
                repo, path_in_repo=prefix, repo_type=repo_type, revision=revision, recursive=True
            )
        ),
        what="verify complete K5 archive listing",
    )
    found = {e.path: e for e in entries if e.path in expected}
    if set(found) != set(expected):
        raise RuntimeError("archive expected file set incomplete")
    for name, row in expected.items():
        entry = found[name]
        if entry.size != row["size"]:
            raise RuntimeError(f"archive size mismatch: {name}")
        if entry.lfs:
            if entry.lfs.sha256 != row["sha256"]:
                raise RuntimeError(f"archive tensor digest mismatch: {name}")
        else:
            data = (root / row["local"]).read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if entry.blob_id != blob:
                raise RuntimeError(f"archive text digest mismatch: {name}")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    record = dict(
        status="verified",
        repo=repo,
        repo_type=repo_type,
        revision=revision,
        prefix=prefix,
        files=expected,
        count=len(expected),
        bytes=sum(v["size"] for v in expected.values()),
        verified_at=time.time(),
    )
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Verified {len(expected)} {kind} files at {repo}@{revision}/{prefix}", flush=True)


def pack_panel(selected: Path, out: Path) -> None:
    """Shard JSONL on byte-newline boundaries, preserving its exact bytes."""
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    chunks, chunk = [], bytearray()
    with (selected / "panel.jsonl").open("rb") as handle:
        for line in handle:
            if len(line) >= 8_000_000:
                raise ValueError("single panel row exceeds shard limit")
            if chunk and len(chunk) + len(line) > 8_000_000:
                chunks.append(bytes(chunk))
                chunk.clear()
            chunk.extend(line)
    if chunk:
        chunks.append(bytes(chunk))
    records = []
    for i, data in enumerate(chunks):
        name = f"panel_shard{i:03d}.jsonl"
        (out / name).write_bytes(data)
        records.append(dict(name=name, sha256=hashlib.sha256(data).hexdigest(), size=len(data)))
    manifest = dict(
        panel_sha256=digest(selected / "panel.jsonl"),
        shards=records,
        selection=json.loads((selected / "selection.json").read_text()),
    )
    (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def stage(config: Path, out: Path) -> None:
    """Download the pinned manifest and shards, then open the verified panel."""
    pin = json.loads(config.read_text())

    def fetch(name):
        return Path(
            retry_transient(
                lambda: hf_hub_download(
                    pin["repo"],
                    f"{pin['prefix']}/{name}",
                    repo_type=pin["repo_type"],
                    revision=pin["revision"],
                ),
                what=f"stage K5 input {name}",
            )
        )

    manifest = json.loads(fetch("input_manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    panel = out / "panel.jsonl"
    temporary = out / "panel.jsonl.tmp"
    with temporary.open("wb") as handle:
        for row in manifest["shards"]:
            path = fetch(row["name"])
            if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
                raise RuntimeError("staged input content mismatch")
            handle.write(path.read_bytes())
    if digest(temporary) != manifest["panel_sha256"]:
        raise RuntimeError("reconstructed panel digest mismatch")
    temporary.replace(panel)
    with panel.open() as handle:
        ids = [json.loads(line)["conv_id"] for line in handle]
    if ids != manifest["selection"]["selected_ids"]:
        raise RuntimeError("staged consumer panel IDs differ from selection")
    (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Staged and opened {len(ids)} exact selected conversations", flush=True)


def main() -> None:
    """Dispatch explicit pack, stage or verified upload operations."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--selected", type=Path, required=True)
    pack.add_argument("--out", type=Path, required=True)
    st = sub.add_parser("stage")
    st.add_argument("--config", type=Path, required=True)
    st.add_argument("--out", type=Path, required=True)
    up = sub.add_parser("upload")
    up.add_argument("--root", type=Path, required=True)
    up.add_argument("--prefix", required=True)
    up.add_argument("--kind", choices=("text", "tensors"), required=True)
    up.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    if args.command == "pack":
        pack_panel(args.selected, args.out)
    elif args.command == "stage":
        stage(args.config, args.out)
    else:
        upload(args.root, args.prefix, args.kind, args.receipt)


if __name__ == "__main__":
    main()
