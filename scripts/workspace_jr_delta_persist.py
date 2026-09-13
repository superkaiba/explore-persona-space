#!/usr/bin/env python3
"""Upload changed experiment files, then verify the full tree at one Hub revision."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json  # noqa: E402
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    assert_hub_dir_filecounts,
    retry_transient,
)
from explore_persona_space.orchestrate.secret_scrub import assert_upload_clean  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"


def tree_files(root):
    """Enumerate the exact visible regular-file tree, rejecting symlinks anywhere."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Upload root must be a real directory")
    entries = sorted(root.rglob("*"))
    if any(
        path.is_symlink() or any(part.startswith(".") for part in path.relative_to(root).parts)
        for path in entries
    ):
        raise ValueError("Upload tree contains a hidden entry or symlink")
    return [path for path in entries if path.is_file()]


def metadata(api, names, revision):
    """Read only named paths; missing entries are candidates for upload, never verified."""
    result = {}
    for start in range(0, len(names), 100):
        requested = names[start : start + 100]
        entries = retry_transient(
            lambda requested=requested: api.get_paths_info(
                REPO, paths=requested, repo_type="dataset", revision=revision
            ),
            what="workspace_jr_delta_metadata",
        )
        if len({entry.path for entry in entries}) != len(entries) or any(
            entry.path not in requested for entry in entries
        ):
            raise ValueError("Unexpected or duplicate remote metadata")
        result.update({entry.path: entry for entry in entries})
        print(f"metadata paths={min(start + 100, len(names))}/{len(names)}", flush=True)
    return result


def same_bytes(entry, evidence):
    """Match either LFS SHA-256 or Git blob SHA-1, with the exact byte count."""
    if entry is None or not hasattr(entry, "size") or entry.size != evidence["size"]:
        return False
    if entry.lfs:
        return entry.lfs.sha256 == evidence["sha256"]
    return entry.blob_id == evidence["git_blob_sha1"]


def persist(root, prefix, receipt, api=None):
    """Transfer bounded changed-file batches and issue a complete current-byte receipt."""
    started = time.monotonic()
    uploader_sha256 = file_sha256(Path(__file__))
    if receipt.exists() or receipt.is_symlink():
        raise ValueError("A fresh receipt path is required")
    if receipt.resolve().is_relative_to(root.resolve()):
        raise ValueError("Receipt must live outside the upload tree")
    parts = Path(prefix).parts
    if (
        not prefix.startswith("exploratory_workspace_jr/20260912/")
        or any(part.startswith(".") for part in parts)
        or any(character in prefix for character in "*?[]")
    ):
        raise ValueError("Only a literal path in this experiment namespace is allowed")
    paths = tree_files(root)
    if not paths:
        raise ValueError("Cannot persist an empty tree")
    evidence = {}
    print(f"delta upload starting files={len(paths)} prefix={prefix}", flush=True)
    for index, path in enumerate(paths, 1):
        relative = str(path.relative_to(root))
        if path.is_symlink() or any(part.startswith(".") for part in Path(relative).parts):
            raise ValueError("Upload tree contains a hidden file or symlink")
        if any(character in relative for character in "*?[]"):
            raise ValueError("Upload file name contains glob metacharacters")
        size = path.stat().st_size
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 2**20), b""):
                sha256.update(block)
                git_blob.update(block)
        evidence[relative] = {
            "size": size,
            "sha256": sha256.hexdigest(),
            "git_blob_sha1": git_blob.hexdigest(),
        }
        if index % 48 == 0 or index == len(paths):
            print(
                f"local hashes={index}/{len(paths)} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    assert_upload_clean(paths, what=f"workspace-jr delta {prefix}")
    api = HfApi() if api is None else api
    base = retry_transient(
        lambda: api.repo_info(REPO, repo_type="dataset", revision="main"),
        what="workspace_jr_delta_head",
    ).sha
    names = [f"{prefix}/{relative}" for relative in evidence]
    previous = metadata(api, names, base)
    changed = [
        relative
        for relative, item in evidence.items()
        if not same_bytes(previous.get(f"{prefix}/{relative}"), item)
    ]
    revision = base
    print(f"delta files_changed={len(changed)} files_total={len(paths)} base={base}", flush=True)
    # The registered consumers open individual paths at a pinned revision. Keep
    # that layout, while bounding each commit instead of restaging the full tree.
    for start in range(0, len(changed), 500):
        batch = changed[start : start + 500]
        assert_hub_dir_filecounts(root, prefix, allow_patterns=batch)
        commit = retry_transient(
            lambda batch=batch: api.upload_folder(
                repo_id=REPO,
                repo_type="dataset",
                folder_path=root,
                path_in_repo=prefix,
                allow_patterns=batch,
                commit_message=f"Persist changed files in {prefix}",
            ),
            what="workspace_jr_delta_upload",
        )
        revision = commit.oid
        print(
            f"uploaded files={min(start + 500, len(changed))}/{len(changed)} revision={revision}",
            flush=True,
        )
    current = metadata(api, names, revision)
    if set(current) != set(names):
        raise ValueError("Final remote revision is missing required files")
    for index, (relative, item) in enumerate(evidence.items(), 1):
        path = root / relative
        if path.stat().st_size != item["size"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"Local artifact changed during upload: {relative}")
        if not same_bytes(current[f"{prefix}/{relative}"], item):
            raise ValueError(f"Final remote byte hash differs: {relative}")
        if index % 48 == 0 or index == len(evidence):
            print(
                f"verified files={index}/{len(evidence)} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if tree_files(root) != paths:
        raise ValueError("Local artifact file set changed during upload")
    if file_sha256(Path(__file__)) != uploader_sha256:
        raise ValueError("Uploader implementation changed during execution")
    result = {
        "revision": revision,
        "repo": REPO,
        "prefix": prefix,
        "verified_sha256": {relative: item["sha256"] for relative, item in evidence.items()},
        "files_verified": len(evidence),
        "files_uploaded": len(changed),
        "base_revision": base,
        "uploader_sha256": uploader_sha256,
    }
    save_json(receipt, result)
    print(f"persist verified={len(evidence)} revision={revision}", flush=True)
    return result


def main():
    """Use the same explicit root/prefix/receipt interface as the original uploader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    persist(args.root, args.prefix, args.receipt)


if __name__ == "__main__":
    main()
