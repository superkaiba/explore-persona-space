"""Persist a completed #2669 run as small, lossless JSONL shards on private HF.

Usage: uv run python scripts/issue2669_persist.py RUN_ROOT BUNDLE_ROOT
The chosen private destination is checked live; no repo visibility is changed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

SHARD_BYTES = 8_500_000
PREFIX = "issue2669_codex_forecast/reduced900_v2"


def persist(run: Path, bundle: Path) -> dict:
    """Bundle exact UTF-8 file bytes; verify a private, exact-set Hub upload."""
    if not run.is_absolute() or not bundle.is_absolute():
        raise ValueError("Absolute source and bundle paths required")
    if run.resolve().is_relative_to(bundle.resolve()) or bundle.resolve().is_relative_to(
        run.resolve()
    ):
        raise ValueError("Source and bundle paths must not overlap")
    api = HfApi()
    repo = hub.DEFAULT_OVERFLOW_REPO
    if api.repo_info(repo, repo_type="dataset").private is not True:
        raise RuntimeError("Artifact destination must already be private")
    for phase in ("pilot", "production"):
        summary = json.loads((run / ("judgments_" + phase) / "dispatch_summary.json").read_text())
        if not summary["results"] or any(r["status"] != "complete" for r in summary["results"]):
            raise RuntimeError("Cannot label incomplete forecast run as complete")
    bundle.mkdir(parents=True, exist_ok=False)
    candidates = sorted(run.rglob("*"))
    if any(path.is_symlink() for path in candidates):
        raise ValueError("Symlinks are not permitted in the artifact source")
    sources = [p for p in candidates if p.is_file() and p.name != ".dispatch.lock"]
    entries, shards, size, current = [], [], 0, []
    for path in sources:
        raw = path.read_bytes()
        record = {
            "path": path.relative_to(run).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "content": raw.decode("utf-8"),
        }
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode()
        if len(line) >= SHARD_BYTES:
            raise ValueError("Single artifact needs explicit splitting: " + str(path))
        if size + len(line) >= SHARD_BYTES:
            target = bundle / f"raw-{len(shards):04d}.jsonl"
            target.write_bytes(b"".join(current))
            shards.append(target.name)
            current, size = [], 0
        current.append(line)
        size += len(line)
        entries.append({k: record[k] for k in ("path", "sha256")})
    if current:
        target = bundle / f"raw-{len(shards):04d}.jsonl"
        target.write_bytes(b"".join(current))
        shards.append(target.name)
    manifest = {
        "run": str(run),
        "format": "jsonl records with exact relative path/content/sha256",
        "source_files": entries,
        "shards": {
            name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in shards
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    expected = [PREFIX + "/" + p.name for p in bundle.iterdir() if p.is_file()]
    result = hub._upload_folder_filtered(
        bundle, repo, "dataset", PREFIX, ["*.jsonl", "manifest.json"], expected, private=True
    )
    if result != repo + "/" + PREFIX:
        raise RuntimeError("Upload helper returned no verified exact destination")
    remote = api.repo_info(repo, repo_type="dataset")
    if remote.private is not True:
        raise RuntimeError("Destination privacy changed; cannot mark persistence verified")
    revision = remote.sha
    missing = [
        p for p in expected if not api.file_exists(repo, p, repo_type="dataset", revision=revision)
    ]
    if missing:
        raise RuntimeError("Missing persisted shard(s): " + str(missing))
    remote_manifest = Path(
        hf_hub_download(
            repo_id=repo,
            filename=PREFIX + "/manifest.json",
            repo_type="dataset",
            revision=revision,
        )
    )
    manifest_sha = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    if hashlib.sha256(remote_manifest.read_bytes()).hexdigest() != manifest_sha:
        raise RuntimeError("Downloaded manifest hash mismatch")
    verification = {
        "repo": repo,
        "private": True,
        "revision": revision,
        "prefix": PREFIX,
        "files_verified": len(expected),
        "source_files": len(entries),
        "manifest_sha256": manifest_sha,
        "verification_scope": "exact expected file presence at revision; downloaded manifest SHA256; raw shard bytes not redownloaded",
    }
    (run / "persistence.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(persist(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
