"""Persist and hash-verify the complete #825 calibration maps and predictions."""

# ruff: noqa: E402
from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

from huggingface_hub import HfApi

from explore_persona_space.orchestrate.hub import (
    _upload_folder_filtered,
    retry_transient,
)

PREFIX = "issue825_turn_bias_scale_20260914"
DATA_REPO = "superkaiba1/explore-persona-space-data"


def sha(path):
    """Hash numerical artifacts without reading whole maps into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Require complete numerical results before the single bulk upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--repo", default=DATA_REPO)
    parser.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = json.loads((args.results_dir / "results.json").read_text())
    if results["status"] != "complete" or len(results["cells"]) != 50:
        raise RuntimeError("complete reduced results required before upload")
    expected = {}
    for kind, count, file_key, hash_key in (
        ("maps", 6, "map_file", "map_sha256"),
        ("folds", 300, "prediction_file", "prediction_sha256"),
    ):
        receipts = sorted((args.results_dir / kind).glob("*.json"))
        if len(receipts) != count:
            raise RuntimeError(f"expected {count} {kind} receipts, found {len(receipts)}")
        for receipt in receipts:
            row = json.loads(receipt.read_text())
            path = Path(row[file_key])
            relative = path.relative_to(args.store).as_posix()
            if sha(path) != row[hash_key]:
                raise RuntimeError(f"local artifact hash changed: {path}")
            expected[f"{PREFIX}/{relative}"] = {
                "local": relative,
                "size": path.stat().st_size,
                "sha256": row[hash_key],
            }
    if len(expected) != 306:
        raise RuntimeError("expected 306 unique numerical artifacts")
    logging.info(
        "Expected-path manifest: %d artifacts, %d bytes",
        len(expected),
        sum(r["size"] for r in expected.values()),
    )
    api = HfApi()
    private = args.repo == "superkaiba1/explore-persona-space-overflow"
    if private and not retry_transient(
        lambda: api.repo_info(args.repo, repo_type=args.repo_type).private,
        what="verify overflow privacy",
    ):
        raise RuntimeError("overflow destination must already be private")
    url = _upload_folder_filtered(
        args.store,
        args.repo,
        args.repo_type,
        PREFIX,
        allow_patterns=["maps/*.npz", "predictions/*.npz"],
        expected_repo_paths=sorted(expected),
        delete_after=False,
        private=private,
    )
    if not url:
        raise RuntimeError("numerical upload failed; all local artifacts retained")
    owner, repository, prefix = url.split("/", 2)
    repo = f"{owner}/{repository}"
    if prefix != PREFIX:
        raise RuntimeError("unexpected upload destination")
    revision = retry_transient(
        lambda: api.repo_info(repo, repo_type=args.repo_type).sha, what="uploaded revision"
    )
    entries = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: retry_transient reopens and consumes the complete paginated listing.
            api.list_repo_tree(
                repo,
                path_in_repo=PREFIX,
                recursive=True,
                repo_type=args.repo_type,
                revision=revision,
            )
        ),
        what="hash-verify uploaded numerical artifacts",
    )
    found = {entry.path: entry for entry in entries if entry.path in expected}
    if set(found) != set(expected):
        raise RuntimeError("uploaded artifact set is incomplete")
    for path, record in expected.items():
        entry = found[path]
        if entry.size != record["size"] or not entry.lfs or entry.lfs.sha256 != record["sha256"]:
            raise RuntimeError(f"uploaded LFS bytes differ: {path}")
    receipt = {
        "status": "verified",
        "repo": repo,
        "repo_type": args.repo_type,
        "private": private,
        "revision": revision,
        "prefix": PREFIX,
        "files": expected,
        "count": len(expected),
        "bytes": sum(r["size"] for r in expected.values()),
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    destination = args.results_dir / "upload_verification.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    logging.info(
        "Verified %d exact LFS hashes at %s/tree/%s/%s", len(expected), repo, revision, PREFIX
    )


if __name__ == "__main__":
    main()
