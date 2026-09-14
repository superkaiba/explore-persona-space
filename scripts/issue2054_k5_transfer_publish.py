"""Publish K5 transfer artifacts and verify the exact remote bytes."""

# ruff: noqa: E402
# Project environment must be loaded before Hugging Face freezes its settings.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import CommitOperationAdd, HfApi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import issue2054_k5_loso_calibration as base


def publish(out, figures, name):
    """Upload completed evidence in bounded commits, then check every Hub hash."""
    report = json.loads((out / "results.json").read_text())
    if report["status"] != "complete":
        raise RuntimeError("refusing to publish an incomplete result")
    prefix = f"{base.PREFIX}/transfer_calibration_v1/{name}"
    files = {
        f"{prefix}/results/{p.relative_to(out)}": p
        for p in out.rglob("*")
        if p.is_file()
        and p.suffix in [".json", ".npz", ".md", ".jsonl", ".txt"]
        and not p.name.endswith((".partial.json", ".tmp.npz"))
    }
    files.update(
        {
            f"{prefix}/figures/{p.name}": p
            for p in figures.iterdir()
            if p.is_file() and p.suffix in [".json", ".pdf", ".png"]
        }
    )
    for script in [
        "issue2054_k5_loso_calibration.py",
        "issue2054_k5_assistant_transfer.py",
        "issue2054_k5_assistant_transfer_plot.py",
        "issue2054_k5_transfer_publish.py",
        "issue2054_k5_plain_run.py",
    ]:
        files[f"{prefix}/code/{script}"] = REPO / "scripts" / script
    test = REPO / "tests/test_issue2054_k5_transfer_calibration.py"
    files[f"{prefix}/code/{test.name}"] = test
    if not files or not any(p.suffix == ".png" for p in files.values()):
        raise RuntimeError("no artifacts or figure to publish")
    hashes = {destination: base.sha(path) for destination, path in files.items()}
    print(
        f"[phase=upload] starting {name} files={len(files)} bytes={sum(p.stat().st_size for p in files.values())}",
        flush=True,
    )
    api = HfApi()
    items = sorted(files.items())
    revisions = []
    for start in range(0, len(items), 150):
        batch = items[start : start + 150]
        commit = base.retry_transient(
            lambda: api.create_commit(
                repo_id=base.HF_REPO,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=d, path_or_fileobj=p) for d, p in batch
                ],
                commit_message=f"K5 {name} transfer artifacts {start + 1}-{start + len(batch)}",
            ),
            what=f"publish K5 {name} batch {start}",
        )
        revisions.append(commit.oid)
        print(
            f"[phase=upload] {name} files={min(start + 150, len(items))}/{len(items)} revision={commit.oid}",
            flush=True,
        )
    revision = revisions[-1]
    entries = base.retry_transient(
        lambda: list(
            api.list_repo_tree(
                base.HF_REPO,
                repo_type="dataset",
                revision=revision,
                path_in_repo=prefix,
                recursive=True,
            )
        ),
        what=f"verify K5 {name} upload listing",
    )
    remote = {e.path: e for e in entries if hasattr(e, "size")}
    if set(remote) != set(files):
        raise RuntimeError("remote artifact set differs from local publication manifest")
    for destination, path in files.items():
        entry = remote[destination]
        if entry.size != path.stat().st_size or base.sha(path) != hashes[destination]:
            raise RuntimeError(f"file size or local bytes changed: {destination}")
        if entry.lfs is not None:
            matched = entry.lfs.sha256 == hashes[destination]
        else:
            blob = hashlib.sha1(f"blob {entry.size}\0".encode())
            blob.update(path.read_bytes())
            matched = blob.hexdigest() == entry.blob_id
        if not matched:
            raise RuntimeError(f"remote hash differs: {destination}")
    verification = {
        "status": "pass",
        "revision": revision,
        "prefix": prefix,
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files.values()),
        "hashes": hashes,
        "all_remote_sizes_and_hashes_match": True,
        "figure_urls": [
            f"https://huggingface.co/datasets/{base.HF_REPO}/resolve/{revision}/{d}"
            for d, p in files.items()
            if p.suffix == ".png" and "grayscale" not in p.name
        ],
    }
    path = REPO / "eval_results/issue_2054/k5_transfer_publication" / f"{name}.json"
    base.atomic_json(path, verification)
    print(
        json.dumps({k: v for k, v in verification.items() if k != "hashes"}, indent=2), flush=True
    )
    print("[phase=done] publication verified", flush=True)


def main():
    """Publish one finished comparison under its own immutable result prefix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument(
        "--name", choices=["loso", "assistant_sources", "plain_assistant_source"], required=True
    )
    args = parser.parse_args()
    publish(args.out, args.figures, args.name)


if __name__ == "__main__":
    main()
