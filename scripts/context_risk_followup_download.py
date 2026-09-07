"""Stage the already pinned public model and verify every downloaded file."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> None:
    """Download on the pod, checking metadata, sizes, and content before a sentinel."""
    root = Path(os.environ["EPM_CONTEXT_RISK_FOLLOWUP_SETUP"])
    manifest = json.loads((root / "model_files.json").read_text())
    revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    if manifest["resolved_sha"] != revision or manifest["revision"] != revision:
        raise ValueError("Model revision differs from the validated map's model")
    if manifest["repo"] != "Qwen/Qwen3.8-27B" or len(manifest["files"]) != 32:
        raise ValueError("Model repository/file roster differs")
    started = time.monotonic()
    snapshot = Path(snapshot_download(manifest["repo"], revision=revision, max_workers=4))
    for row in manifest["files"]:
        path = snapshot / row["path"]
        if path.stat().st_size != row["size"]:
            raise ValueError(f"Downloaded file size differs: {row['path']}")
        digest = hashlib.sha256() if row["lfs_sha256"] else hashlib.sha1()
        if not row["lfs_sha256"]:
            digest.update(f"blob {row['size']}\0".encode())
        with path.open("rb") as handle:
            while block := handle.read(16 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != (row["lfs_sha256"] or row["blob_id"]):
            raise ValueError(f"Downloaded file hash differs: {row['path']}")
        print(f"[model-file-verified] {row['path']} bytes={row['size']}", flush=True)
    report = {
        "snapshot": str(snapshot),
        "revision": revision,
        "verified_files": 32,
        "bytes": sum(row["size"] for row in manifest["files"]),
        "elapsed_seconds": time.monotonic() - started,
        "passed": True,
    }
    temporary = root / "model_download_complete.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(root / "model_download_complete.json")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
