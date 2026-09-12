#!/usr/bin/env python3
"""Upload the explicit experiment output tree and verify every remote byte hash."""

import argparse
import hashlib
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json  # noqa: E402
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    assert_hub_dir_filecounts,
    retry_transient,
    verify_repo_paths_uploaded,
)
from explore_persona_space.orchestrate.secret_scrub import assert_upload_clean  # noqa: E402


def main():
    """Persist outputs, fail on missing/mismatched files, and write a separate receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("Upload receipt must live outside the uploaded tree")
    if not args.prefix.startswith("exploratory_workspace_jr/"):
        raise ValueError("Only this experiment's declared upload namespace is allowed")
    files = sorted(p for p in args.root.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Cannot persist an empty output directory")
    if any(
        p.is_symlink() or any(part.startswith(".") for part in p.relative_to(args.root).parts)
        for p in files
    ):
        raise ValueError("Output tree contains a hidden file or symlink; inspect it explicitly")
    manifest = {str(p.relative_to(args.root)): file_sha256(p) for p in files}
    assert_upload_clean(files, what=f"workspace-jr {args.prefix}")
    assert_hub_dir_filecounts(args.root, args.prefix, allow_patterns=list(manifest))
    repo = "superkaiba1/explore-persona-space-data"
    api = HfApi()
    commit = retry_transient(
        lambda: api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=args.root,
            path_in_repo=args.prefix,
            allow_patterns=list(manifest),
            commit_message=f"Persist {args.prefix}",
        ),
        what="workspace_jr_upload_folder",
    )
    expected = {f"{args.prefix}/{name}": name for name in manifest}
    missing = verify_repo_paths_uploaded(
        api, repo, list(expected), path_in_repo=args.prefix, revision=commit.oid
    )
    if missing:
        raise RuntimeError(f"Upload incomplete: {missing}")
    names = list(expected)
    for start in range(0, len(names), 100):
        requested = names[start : start + 100]
        entries = retry_transient(
            lambda requested=requested: api.get_paths_info(
                repo, paths=requested, repo_type="dataset", revision=commit.oid
            ),
            what="workspace_jr_verify_metadata",
        )
        if {entry.path for entry in entries} != set(requested):
            raise ValueError("Remote metadata does not cover every requested file")
        for entry in entries:
            name = expected[entry.path]
            path = args.root / name
            if file_sha256(path) != manifest[name]:
                raise ValueError(f"Local artifact changed during upload: {name}")
            if entry.size != path.stat().st_size:
                raise ValueError(f"Remote size mismatch: {entry.path}")
            if entry.lfs:
                matches = entry.lfs.sha256 == manifest[name]
            else:
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != manifest[name]:
                    raise ValueError(f"Local text changed during verification: {name}")
                matches = (
                    entry.blob_id
                    == hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                )
            if not matches:
                raise ValueError(f"Remote byte-hash mismatch: {entry.path}")
    save_json(
        args.receipt,
        {
            "revision": commit.oid,
            "repo": repo,
            "prefix": args.prefix,
            "verified_sha256": manifest,
            "files_verified": len(files),
        },
    )
    print(f"persist verified={len(files)} revision={commit.oid}", flush=True)


if __name__ == "__main__":
    main()
