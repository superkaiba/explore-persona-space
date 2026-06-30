"""Drain stranded .pt extracts from pod-651 to HF data repo in safe chunks.

Recovery utility for #651: the in-dispatcher `_upload_shift_tensors` did ONE
batched `create_commit` for all extracts and died on HF Hub 504 gateway timeout
(failure v5, 2026-06-17). This script uploads the .pt files (and matching
.manifest.json siblings) in small chunks with retry+backoff, verifying each
chunk lands before moving on. Designed to run on the pod that has the files.

Idempotent — files already on HF Hub are skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Project dotenv wrapper (#745): this drain utility imports huggingface_hub at
# MODULE TOP and never called load_dotenv. huggingface_hub.constants freezes
# HF_HUB_ENABLE_HF_TRANSFER at IMPORT time, so the accelerator env must be in
# os.environ BEFORE the huggingface_hub import below. load_dotenv() (the project
# wrapper, which setdefaults HF_XET_HIGH_PERFORMANCE + HF_HUB_ENABLE_HF_TRANSFER
# and reads the project .env robustly) runs at module scope FIRST; the
# huggingface_hub imports are deferred to AFTER it. On a pod/GCE/SLURM worker the
# shell-level exports already set these regardless of import order — this makes a
# LOCAL run correct too. orchestrate.env does NOT itself import huggingface_hub,
# so this ordering is safe.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402

DEST_REPO = "superkaiba1/explore-persona-space-data"
DEST_PREFIX = "issue651_cross_behavior_geometry/analysis_tensors"
CHUNK_SIZE = 20  # files per commit (small enough to avoid 504)
MAX_ATTEMPTS = 6
INITIAL_BACKOFF_S = 30.0  # exponential backoff base

logging.basicConfig(
    format="%(asctime)s [drain] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("issue651-drain")


def list_existing_remote(api: HfApi) -> set[str]:
    """Return path_in_repo for every file already in the destination prefix."""
    files = list_repo_files(DEST_REPO, repo_type="dataset", revision="main")
    return {f for f in files if f.startswith(f"{DEST_PREFIX}/")}


def find_local_files(shift_dir: Path) -> list[Path]:
    """Return .pt + .manifest.json files, sorted (deterministic chunking)."""
    pts = sorted(shift_dir.glob("*.pt"))
    manifests = sorted(shift_dir.glob("*.manifest.json"))
    return pts + manifests


def commit_chunk(api: HfApi, chunk: list[Path], chunk_idx: int, total_chunks: int) -> None:
    """Commit a chunk with exponential-backoff retry on transient errors."""
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{DEST_PREFIX}/{p.name}",
            path_or_fileobj=str(p),
        )
        for p in chunk
    ]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            log.info(
                "chunk %d/%d (%d files, attempt %d)",
                chunk_idx,
                total_chunks,
                len(ops),
                attempt,
            )
            api.create_commit(
                repo_id=DEST_REPO,
                repo_type="dataset",
                operations=ops,
                commit_message=(
                    f"issue651: drain chunk {chunk_idx}/{total_chunks} "
                    f"({len(ops)} files; recovery from failure v5)"
                ),
            )
            log.info("chunk %d/%d committed", chunk_idx, total_chunks)
            return
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            transient = code in (429, 500, 502, 503, 504) or "504" in str(e) or "Gateway" in str(e)
            if not transient or attempt == MAX_ATTEMPTS:
                log.error(
                    "chunk %d/%d FAILED (status=%s, attempt %d): %s",
                    chunk_idx,
                    total_chunks,
                    code,
                    attempt,
                    e,
                )
                raise
            backoff = INITIAL_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(
                "chunk %d/%d transient %s (attempt %d) - sleep %.0fs and retry",
                chunk_idx,
                total_chunks,
                code,
                attempt,
                backoff,
            )
            time.sleep(backoff)
        except Exception as e:
            log.error("chunk %d/%d unexpected error: %s", chunk_idx, total_chunks, e)
            raise


def verify_chunk(remote: set[str], chunk: list[Path]) -> list[Path]:
    """Return any files in chunk not present remotely."""
    missing: list[Path] = []
    for p in chunk:
        if f"{DEST_PREFIX}/{p.name}" not in remote:
            missing.append(p)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shift-dir",
        default="/workspace/explore-persona-space/eval_results/issue_651/shifts",
        help="local dir holding .pt + .manifest.json files",
    )
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--dry-run", action="store_true", help="list plan without uploading")
    args = ap.parse_args()

    shift_dir = Path(args.shift_dir)
    if not shift_dir.is_dir():
        log.error("shift dir not found: %s", shift_dir)
        return 2

    if "HF_TOKEN" not in os.environ:
        log.error("HF_TOKEN env var missing")
        return 2

    local = find_local_files(shift_dir)
    if not local:
        log.error("no .pt / .manifest.json files in %s", shift_dir)
        return 2
    log.info(
        "found %d local files (%d .pt, %d .manifest.json) in %s",
        len(local),
        sum(1 for p in local if p.suffix == ".pt"),
        sum(1 for p in local if p.name.endswith(".manifest.json")),
        shift_dir,
    )

    api = HfApi()
    log.info("listing remote files at %s/%s", DEST_REPO, DEST_PREFIX)
    remote = list_existing_remote(api)
    log.info("remote already has %d files under %s", len(remote), DEST_PREFIX)

    to_upload = [p for p in local if f"{DEST_PREFIX}/{p.name}" not in remote]
    log.info(
        "skipping %d already-uploaded, uploading %d new files in chunks of %d",
        len(local) - len(to_upload),
        len(to_upload),
        args.chunk_size,
    )

    if not to_upload:
        log.info("nothing to upload — all files already on Hub")
        return 0

    if args.dry_run:
        for p in to_upload[:5]:
            log.info("  would upload: %s -> %s/%s", p.name, DEST_PREFIX, p.name)
        if len(to_upload) > 5:
            log.info("  ... and %d more", len(to_upload) - 5)
        return 0

    # Chunk + commit + verify per chunk
    chunks = [to_upload[i : i + args.chunk_size] for i in range(0, len(to_upload), args.chunk_size)]
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        commit_chunk(api, chunk, idx, total_chunks)
        # Refresh remote view and verify chunk landed
        remote = list_existing_remote(api)
        missing = verify_chunk(remote, chunk)
        if missing:
            log.error(
                "chunk %d/%d verification FAILED — %d files missing post-commit: %s",
                idx,
                total_chunks,
                len(missing),
                [p.name for p in missing[:5]],
            )
            return 3
        log.info("chunk %d/%d verified on Hub", idx, total_chunks)

    # Final cross-check
    remote = list_existing_remote(api)
    still_missing = [p for p in to_upload if f"{DEST_PREFIX}/{p.name}" not in remote]
    if still_missing:
        log.error("FINAL FAIL: %d files still missing on Hub", len(still_missing))
        return 3
    log.info("ALL %d files uploaded + verified at %s/%s", len(to_upload), DEST_REPO, DEST_PREFIX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
