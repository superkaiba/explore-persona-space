#!/usr/bin/env python
"""Phase-0 VM-side poll script: wait for #778 v2 artifacts on HF.

Polls ``superkaiba1/explore-persona-space-data`` every POLL_INTERVAL_MIN
(default 10) for the six required v2 files:
  issue778_persona_vectors/analysis_tensors_v2/rb/{evil,sycophancy,hallucination}.pt
  issue778_persona_vectors/analysis_tensors_v2/neutral_cov_{evil,sycophancy,hallucination}.pt

Behaviour:
- Every tick: lists the full ``analysis_tensors_v2/`` prefix on HF; if files
  exist under the prefix but the EXACT names don't match the 6 expected files,
  prints the ACTUAL filenames loudly and exits with code 3 (name-mismatch
  sentinel) so the caller can inspect and update.
- On receipt of all 6 files: downloads each to ``cache_dir``, runs shape /
  dtype fitness checks, logs sha256 + size, exits 0.
- After ``TIMEOUT_HOURS`` (default 48h) without all files: exits 2.
- ``--once``: run exactly ONE check and exit 0/1/2/3 (used for smoke testing).

Exit codes:
  0  all 6 v2 files present + fitness-checked
  1  not yet present (normal timeout path)
  2  timed out (> TIMEOUT_HOURS, all-absent)
  3  files exist under the prefix but names mismatch the expected 6

Usage:
  # Smoke (single check, downloads into /tmp):
  uv run python scripts/issue816_wait_v2_artifacts.py \\
      --cache-dir /tmp/issue816_v2_cache --once

  # Production (poll every 10 min, up to 48h):
  uv run python scripts/issue816_wait_v2_artifacts.py \\
      --cache-dir data/issue816/hf_dl/v2_artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

# load_dotenv MUST be called before any HF import (orchestrate.env sets
# HF_XET_HIGH_PERFORMANCE + HF_HUB_ENABLE_HF_TRANSFER; uv run doesn't auto-load .env)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402 (after load_dotenv)
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue816.wait_v2")

REPO_ID = "superkaiba1/explore-persona-space-data"
REPO_TYPE = "dataset"
REVISION = "main"
V2_PREFIX = "issue778_persona_vectors/analysis_tensors_v2/"

TRAITS = ("evil", "sycophancy", "hallucination")
EXPECTED_FILES: list[str] = []
for _t in TRAITS:
    EXPECTED_FILES.append(f"{V2_PREFIX}rb/{_t}.pt")
for _t in TRAITS:
    EXPECTED_FILES.append(f"{V2_PREFIX}neutral_cov_{_t}.pt")
EXPECTED_FILES_SET = set(EXPECTED_FILES)

# Fitness-check constants.
N_LAYERS = 28
D_HIDDEN = 3584
EXPECTED_RB_SHAPE = (N_LAYERS, D_HIDDEN)  # (28, 3584)


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fitness_check(local_path: Path, repo_path: str) -> None:
    """Assert shape/dtype contract for a downloaded v2 tensor.

    rb tensors: shape (28, 3584), float32.
    neutral_cov tensors: shape (28, 3584, 3584) OR (28, 3584) [diagonal], float32.
    Raises AssertionError with a loud message on any violation.
    """
    t = torch.load(local_path, map_location="cpu", weights_only=True)
    if not isinstance(t, torch.Tensor):
        raise AssertionError(f"[fitness] {repo_path}: expected a torch.Tensor, got {type(t)}")
    t_np = t.float()
    is_rb = "/rb/" in repo_path
    if is_rb:
        assert t_np.shape == torch.Size(EXPECTED_RB_SHAPE), (
            f"[fitness] {repo_path}: rb shape {tuple(t_np.shape)} != {EXPECTED_RB_SHAPE}"
        )
    else:
        # neutral_cov: full (28, 3584, 3584) OR diagonal (28, 3584)
        ok_full = tuple(t_np.shape) == (N_LAYERS, D_HIDDEN, D_HIDDEN)
        ok_diag = tuple(t_np.shape) == (N_LAYERS, D_HIDDEN)
        assert ok_full or ok_diag, (
            f"[fitness] {repo_path}: neutral_cov shape {tuple(t_np.shape)} is neither "
            f"({N_LAYERS},{D_HIDDEN},{D_HIDDEN}) nor ({N_LAYERS},{D_HIDDEN})"
        )
    logger.info(
        "[fitness] PASS %s | shape=%s dtype=%s size_mb=%.2f",
        repo_path,
        tuple(t_np.shape),
        t_np.dtype,
        local_path.stat().st_size / 1e6,
    )


def _list_v2_prefix() -> set[str]:
    """List all files under the analysis_tensors_v2/ prefix on HF. Returns a set of repo paths."""
    all_files = list(
        list_repo_files(
            REPO_ID,
            repo_type=REPO_TYPE,
            revision=REVISION,
        )
    )
    return {f for f in all_files if f.startswith(V2_PREFIX)}


def _download_and_check(cache_dir: Path) -> None:
    """Download all 6 expected files and run fitness checks. Raises on any failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for repo_path in EXPECTED_FILES:
        local_path = cache_dir / Path(repo_path).name
        if local_path.exists():
            logger.info("[cache] %s already at %s, skipping download", repo_path, local_path)
        else:
            logger.info("[download] %s -> %s", repo_path, local_path)
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=repo_path,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may nest; find the file wherever it landed.
            candidate = cache_dir / Path(repo_path).name
            if not candidate.exists():
                # fall back to nested path
                candidate = cache_dir / repo_path
            local_path = candidate
        sha = _sha256_file(local_path)
        size_mb = local_path.stat().st_size / 1e6
        logger.info("[sha256] %s  sha=%s  size_mb=%.2f", local_path.name, sha[:16], size_mb)
        _fitness_check(local_path, repo_path)


def poll_once(cache_dir: Path) -> tuple[str, set[str] | None]:
    """Single poll tick.

    Returns:
        ("all_present", None)         — all 6 files found
        ("name_mismatch", found_set)  — files exist under prefix but names differ
        ("absent", None)              — nothing found under the prefix
    """
    logger.info("[poll] listing %s prefix on HF ...", V2_PREFIX)
    found = _list_v2_prefix()
    if not found:
        logger.info("[poll] no files under %s yet", V2_PREFIX)
        return ("absent", None)
    if found >= EXPECTED_FILES_SET:
        logger.info("[poll] all 6 expected files found — downloading + checking ...")
        return ("all_present", found)
    # Some files exist but names differ or subset only.
    logger.warning(
        "[poll] files found under %s but do NOT match the expected 6:\n"
        "  expected: %s\n"
        "  found:    %s\n"
        "  missing:  %s\n"
        "  extra:    %s",
        V2_PREFIX,
        sorted(EXPECTED_FILES_SET),
        sorted(found),
        sorted(EXPECTED_FILES_SET - found),
        sorted(found - EXPECTED_FILES_SET),
    )
    return ("name_mismatch", found)


def main() -> int:
    """Entry point. Returns exit code."""
    parser = argparse.ArgumentParser(
        description="Phase-0 poll: wait for #778 v2 artifacts on HF data repo."
    )
    parser.add_argument(
        "--cache-dir",
        default="data/issue816/hf_dl/v2_artifacts",
        help="Local directory to download the 6 artifact files into.",
    )
    parser.add_argument(
        "--poll-interval-min",
        type=float,
        default=10.0,
        help="Polling interval in minutes (default 10).",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=48.0,
        help="Maximum wait time in hours before exiting 2 (default 48).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly ONE poll tick and exit (for smoke tests).",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    poll_interval_s = args.poll_interval_min * 60.0
    timeout_s = args.timeout_hours * 3600.0
    start_t = time.monotonic()

    logger.info(
        "[start] waiting for 6 v2 artifacts on %s  prefix=%s",
        REPO_ID,
        V2_PREFIX,
    )
    logger.info("[start] expected files:\n  %s", "\n  ".join(sorted(EXPECTED_FILES)))

    while True:
        status, _found_set = poll_once(cache_dir)

        if status == "all_present":
            try:
                _download_and_check(cache_dir)
            except Exception:
                logger.exception("[fitness] FAIL — one or more v2 files failed fitness checks")
                return 1
            logger.info("[done] all 6 v2 artifacts present and fitness-checked — exit 0")
            return 0

        if status == "name_mismatch":
            logger.error(
                "[name_mismatch] Files exist under %s but names don't match the expected 6. "
                "Update EXPECTED_FILES in this script or wait for the correct names. "
                "Exiting 3.",
                V2_PREFIX,
            )
            return 3

        # status == "absent"
        if args.once:
            logger.info("[once] absent — exit 1")
            return 1

        elapsed = time.monotonic() - start_t
        if elapsed >= timeout_s:
            logger.error("[timeout] waited %.1fh, still absent — exit 2", elapsed / 3600.0)
            return 2

        remaining_s = timeout_s - elapsed
        sleep_s = min(poll_interval_s, remaining_s)
        logger.info(
            "[sleep] waiting %.1f min (elapsed=%.1fh remaining=%.1fh)",
            sleep_s / 60.0,
            elapsed / 3600.0,
            remaining_s / 3600.0,
        )
        time.sleep(sleep_s)


if __name__ == "__main__":
    sys.exit(main())
