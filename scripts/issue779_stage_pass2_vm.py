#!/usr/bin/env python3
"""Issue #779 r13: stage the pass-2 answer-summary shards HF -> VM data disk.

Downloads the 25 pass-2 capture shards (``issue779_capture_answer_summaries_pass2.py``
output, verified on the HF data repo) into the ``issue779_arm_headline_summaries2.py``
default ``--p2-dir`` (``<capture>/pass2`` on ``/mnt/eps-data``). Fail-loud +
resumable: a shard already present with matching byte size AND sha256 (vs the
Hub's LFS metadata) is skipped; everything else is (re)downloaded via
``hf_hub_download`` into a same-filesystem staging dir, atomically
``os.replace``d into place, then sha256-verified. A disk preflight refuses to
start when the remaining bytes + margin exceed the data disk's free space, so
the shared disk is never wedged (#658-class).

Why this exists: the transient VM pass-2 copy rsynced during capture was
deleted by a concurrent session on 2026-07-03 (~18:35Z); the HF copy is
canonical, and the summaries2 analysis (r13) runs VM-side against its default
paths.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper (HF token + shared-VM caps) BEFORE huggingface_hub.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# Single source of truth for the consumer's default --p2-dir.
import issue779_arm_headline_summaries2 as S2  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_stage_pass2_vm")

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue779_monitoring/training-source-ablation-hg/final_token_capture/pass2"
EXPECTED_PER_TAG = {"evil": 5, "sycophancy": 5, "hallucination": 5, "lmsys": 10}
DISK_MARGIN_BYTES = 30 << 30  # keep >=30 GiB free on the shared data disk
N_RETRIES = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def list_remote(api: HfApi, tags: list[str]) -> dict[str, dict]:
    """{shard_name: {size, sha256}} for the requested tags, fail-loud on counts."""
    remote: dict[str, dict] = {}
    for f in api.list_repo_tree(REPO, path_in_repo=PREFIX, repo_type="dataset", recursive=True):
        if getattr(f, "size", None) is None:  # a subdirectory entry
            continue
        name = f.path.rsplit("/", 1)[-1]
        lfs = getattr(f, "lfs", None)
        remote[name] = {"size": f.size, "sha256": lfs.sha256 if lfs else None}
    for tag in tags:
        n = sum(1 for name in remote if name.startswith(f"{tag}_summaries_shard"))
        if n != EXPECTED_PER_TAG[tag]:
            raise RuntimeError(
                f"HF prefix {PREFIX} has {n} shards for tag {tag!r}, "
                f"expected {EXPECTED_PER_TAG[tag]}"
            )
    return {
        name: meta
        for name, meta in remote.items()
        if any(name.startswith(f"{t}_summaries_shard") for t in tags)
    }


def already_staged(dest: Path, meta: dict) -> bool:
    if not dest.exists() or dest.stat().st_size != meta["size"]:
        return False
    if meta["sha256"] is None:
        raise RuntimeError(f"{dest.name}: no LFS sha256 on the Hub entry; refusing blind skip")
    return sha256_file(dest) == meta["sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage issue-779 pass-2 shards HF -> VM.")
    parser.add_argument(
        "--tags", nargs="*", default=list(EXPECTED_PER_TAG), choices=list(EXPECTED_PER_TAG)
    )
    parser.add_argument("--p2-dir", type=Path, default=S2.DEFAULT_P2_DIR)
    args = parser.parse_args()

    t0 = time.time()
    p2_dir: Path = args.p2_dir
    p2_dir.mkdir(parents=True, exist_ok=True)
    staging = p2_dir / ".hf_staging"

    api = HfApi()
    remote = list_remote(api, list(args.tags))
    todo = {n: m for n, m in sorted(remote.items()) if not already_staged(p2_dir / n, m)}
    n_skip = len(remote) - len(todo)
    remaining = sum(m["size"] for m in todo.values())
    free = shutil.disk_usage(p2_dir).free
    logger.info(
        "%d/%d shards already staged+verified; %d to download (%.1f GiB; %.1f GiB free)",
        n_skip,
        len(remote),
        len(todo),
        remaining / 2**30,
        free / 2**30,
    )
    if free < remaining + DISK_MARGIN_BYTES:
        raise RuntimeError(
            f"disk preflight FAILED: need {remaining / 2**30:.1f} GiB "
            f"+ {DISK_MARGIN_BYTES / 2**30:.0f} GiB margin, only {free / 2**30:.1f} GiB free"
        )

    for i, (name, meta) in enumerate(todo.items(), 1):
        dest = p2_dir / name
        last_err: Exception | None = None
        for attempt in range(1, N_RETRIES + 1):
            try:
                got = Path(
                    hf_hub_download(
                        REPO,
                        filename=f"{PREFIX}/{name}",
                        repo_type="dataset",
                        local_dir=staging,
                    )
                )
                got.replace(dest)  # atomic, same filesystem
                break
            except Exception as e:  # bounded retry, then fail loud
                last_err = e
                logger.warning("[%d/%d] %s attempt %d failed: %s", i, len(todo), name, attempt, e)
                time.sleep(10 * attempt)
        else:
            raise RuntimeError(f"{name}: download failed after {N_RETRIES} attempts") from last_err
        sz = dest.stat().st_size
        if sz != meta["size"]:
            raise RuntimeError(f"{name}: size {sz} != Hub {meta['size']}")
        sha = sha256_file(dest)
        if meta["sha256"] is None or sha != meta["sha256"]:
            raise RuntimeError(f"{name}: sha256 {sha} != Hub {meta['sha256']}")
        logger.info("[%d/%d] %s staged + sha-verified (%.1f GiB)", i, len(todo), name, sz / 2**30)

    if staging.exists():
        shutil.rmtree(staging)
    for tag in args.tags:
        n = len(list(p2_dir.glob(f"{tag}_summaries_shard*.pt")))
        if n != EXPECTED_PER_TAG[tag]:
            raise RuntimeError(f"post-stage count for {tag}: {n} != {EXPECTED_PER_TAG[tag]}")
    logger.info(
        "DONE: %d shards staged+verified under %s in %.0fs (tags: %s)",
        len(remote),
        p2_dir,
        time.time() - t0,
        ",".join(args.tags),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
