"""Publish and supervise the bounded paper-matched frozen-transfer analysis."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explore_persona_space.orchestrate.hub import retry_transient
from scripts import issue2054_k3_artifacts as artifacts
from scripts import issue2054_k5_stage_transfer as analysis
from scripts import issue2054_shared_seven_monitor as monitor

PREFIX = "issue2054_k5_stage_transfer"


def validate_result(result, source_sha):
    """Reject incomplete, duplicated, or differently sourced evaluation cells."""
    expected = {(label, fold) for label in analysis.LABELS for fold in range(5)}
    actual = [(r["label"], r["fold"]) for r in result["rows"]]
    if (
        result["source_sha"] != source_sha
        or result["status"] != "complete"
        or len(actual) != len(expected)
        or set(actual) != expected
        or len(result["summary"]) != len(analysis.LABELS)
    ):
        raise ValueError("Incomplete or mismatched frozen-transfer coverage")


def verify(out, source_sha):
    """Independently verify all local bytes against the pinned uploaded inventory."""
    complete = json.loads((out / "complete.json").read_text())
    result = json.loads((out / "results.json").read_text())
    inventory = json.loads((out / "inventory.json").read_text())
    validate_result(result, source_sha)
    if any(x["source_sha"] != source_sha for x in (complete, inventory)):
        raise ValueError("Completion source mismatch")
    for name in ("results", "inventory"):
        if analysis.k3.sha(out / f"{name}.json") != complete[f"{name}_sha256"]:
            raise ValueError(f"Completion {name} hash mismatch")
    files = [
        *inventory["files"],
        {
            "path": "inventory.json",
            "size": (out / "inventory.json").stat().st_size,
            "sha256": complete["inventory_sha256"],
        },
    ]
    paths = [f"{PREFIX}/{out.name}/{r['path']}" for r in files]
    api = HfApi()
    remote = {
        e.path: e
        for e in retry_transient(
            lambda: api.get_paths_info(
                artifacts.k3.HF_REPO,
                paths,
                repo_type="dataset",
                revision=complete["verified_revision"],
            ),
            what="independent frozen-transfer inventory verification",
        )
    }
    if set(remote) != set(paths):
        raise ValueError("Uploaded inventory missing files")
    for row, path in zip(files, paths, strict=True):
        local = out / row["path"]
        if local.stat().st_size != row["size"] or analysis.k3.sha(local) != row["sha256"]:
            raise ValueError(f"Local artifact mismatch: {path}")
        entry = remote[path]
        if entry.size != row["size"]:
            raise ValueError(f"Remote artifact size mismatch: {path}")
        lfs = getattr(entry, "lfs", None)
        if lfs is not None:
            correct = lfs.sha256 == row["sha256"]
        else:
            data = local.read_bytes()
            correct = (
                entry.blob_id == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            )
        if not correct:
            raise ValueError(f"Remote artifact hash mismatch: {path}")
    return complete


def run(manifest, out, source_sha):
    """Resume verified folds, then upload the complete immutable analysis packet."""
    for path in (
        Path(__file__).resolve(),
        Path(monitor.__file__).resolve(),
        Path(artifacts.__file__).resolve(),
    ):
        relative = str(path.relative_to(ROOT))
        committed = subprocess.check_output(["git", "show", f"{source_sha}:{relative}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != analysis.k3.sha(path):
            raise ValueError(f"Uncommitted run helper: {relative}")
    result = analysis.run(manifest, out, source_sha, None)
    validate_result(result, source_sha)
    fingerprint = result["fingerprint"]
    analysis.progress(out, "upload", 30)
    paths = [
        out / "results.json",
        out / "run_identity.json",
        manifest,
        *sorted((out / "folds").glob("*.json")),
        *sorted((out / "folds").glob("*.npz")),
    ]
    paths = [p for p in paths if not p.name.endswith(".done.json")]
    if len(paths) != 63:
        raise ValueError(
            f"Expected manifest, run identity, results and 30 fold pairs; found {len(paths)}"
        )
    artifacts.k3.PREFIX = PREFIX
    artifacts.seal_many(paths, out, fingerprint)
    inventory = {
        "source_sha": source_sha,
        "fingerprint": fingerprint,
        "files": [
            {
                "path": str(p.relative_to(out)),
                "size": p.stat().st_size,
                "sha256": analysis.k3.sha(p),
            }
            for p in paths
        ],
    }
    analysis.k3.atomic_json(out / "inventory.json", inventory)
    artifacts.seal_many([out / "inventory.json"], out, fingerprint)
    receipt = json.loads((out / "inventory.json.done.json").read_text())
    complete = {
        "source_sha": source_sha,
        "status": "complete",
        "finished_at": time.time(),
        "results_sha256": analysis.k3.sha(out / "results.json"),
        "inventory_sha256": analysis.k3.sha(out / "inventory.json"),
        "verified_revision": receipt["revision"],
        "uploaded_files": len(paths) + 1,
    }
    analysis.k3.atomic_json(out / "complete.json", complete)
    artifacts.seal_many([out / "complete.json"], out, fingerprint)
    verify(out, source_sha)
    analysis.progress(out, "verified_complete", 30, **complete)


def main():
    """Use the existing owned-process monitor with this analysis's verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("monitor").add_argument("config", type=Path)
    runner = sub.add_parser("run")
    runner.add_argument("--manifest", type=Path, required=True)
    runner.add_argument("--out-root", type=Path, required=True)
    runner.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if args.action == "run":
        run(args.manifest, args.out_root, args.source_sha)
    else:
        monitor.verify = verify
        sys.argv = [sys.argv[0], str(args.config)]
        monitor.main()


if __name__ == "__main__":
    main()
