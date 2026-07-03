#!/usr/bin/env python
"""Phase-0 VM-side poll script: wait for #778 v2 artifacts on HF.

Polls ``superkaiba1/explore-persona-space-data`` every POLL_INTERVAL_MIN
(default 10) for the #778 v2 COMPLETION SIGNAL:

  issue778_persona_vectors/analysis_tensors_v2/MANIFEST.json

#778's v2 upload contract (scripts/issue778_v2_upload.py on the issue-778-v2rerun
branch): the pod phase uploads ``extract/`` + ``neutral/`` first, the VM phase
uploads ``judge/`` + ``pairing/`` + ``rb_v2/`` + ``honest_nulls_maxdraws_v2/``,
and MANIFEST.json is uploaded LAST — it is the designed #816 consumption signal.
Intermediate files under the prefix BEFORE the manifest are therefore EXPECTED
and are logged each tick (the plan's prefix-glob surfacing), not an error.

On MANIFEST arrival:
- parse it and require the #816-consumed files to be listed AND present on a
  fresh HF listing: ``rb_v2/{evil,sycophancy,hallucination}.pt`` +
  ``neutral/neutral_response_avg.pt``;
- download each ``rb_v2/{trait}.pt``, assert shape (28, 3584) + local sha256 ==
  the manifest sha256 (fitness check (f) content identity);
- do NOT download ``neutral_response_avg.pt`` (hundreds of MB; the fetch-time
  assert in ``issue816_lib.fetch_neutral_cov`` is the binding check) — assert
  its manifest entry exists and, when a shape is recorded, that it is
  ``(n, 28, 3584)``.

A trait missing from ``rb_v2/`` after the manifest lands is the #778 K1 NA case
(< 5 kept judge pairs — no r_B v2 written): exit 4 naming the missing trait(s);
the orchestrator handles the epm:phase0-fitness-fail marker. No fabrication.

Exit codes:
  0  MANIFEST present + all consumed files fitness-checked
  1  manifest not yet present (``--once`` normal path) / fitness exception
  2  timed out (> TIMEOUT_HOURS without the manifest)
  4  MANIFEST present but a consumed file is missing (K1 NA / interface break)

Usage:
  # Smoke (single check, downloads into /tmp):
  uv run python scripts/issue816_wait_v2_artifacts.py \\
      --cache-dir /tmp/issue816_v2_cache --once

  # Production (poll every 10 min, up to 48h from launch):
  uv run python scripts/issue816_wait_v2_artifacts.py \\
      --cache-dir data/issue816/hf_dl/v2_artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
MANIFEST_REPO_PATH = f"{V2_PREFIX}MANIFEST.json"

TRAITS = ("evil", "sycophancy", "hallucination")
# Manifest-relative paths of every file #816 consumes (manifest["files"] keys are
# relative to the v2 root, e.g. "rb_v2/evil.pt").
REQUIRED_REL: list[str] = [f"rb_v2/{_t}.pt" for _t in TRAITS] + ["neutral/neutral_response_avg.pt"]

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


def _download(repo_path: str, cache_dir: Path) -> Path:
    """hf_hub_download ``repo_path`` into ``cache_dir``; return the local path."""
    local = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_path,
        revision=REVISION,
        local_dir=str(cache_dir),
    )
    return Path(local)


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


def _check_manifest_and_rb(cache_dir: Path, found: set[str]) -> int:
    """Manifest-driven fitness checks. Returns an exit code (0 ok, 4 missing file).

    Downloads MANIFEST.json + the three rb_v2 tensors; asserts each rb shape ==
    (28, 3584) and local sha256 == the manifest sha256. neutral_response_avg is
    checked by manifest entry (+ recorded shape) only — never downloaded here.
    Raises on assertion/IO failures (caller maps to exit 1).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _download(MANIFEST_REPO_PATH, cache_dir)
    with open(manifest_path) as f:
        manifest = json.load(f)
    files: dict = manifest.get("files", {})
    logger.info(
        "[manifest] schema=%s label=%s n_files=%d",
        manifest.get("schema_version"),
        manifest.get("label"),
        len(files),
    )

    missing = [rel for rel in REQUIRED_REL if rel not in files or f"{V2_PREFIX}{rel}" not in found]
    if missing:
        logger.error(
            "[missing] MANIFEST.json is present but consumed file(s) are missing "
            "(K1 NA trait or interface break): %s — exit 4",
            missing,
        )
        return 4

    for trait in TRAITS:
        rel = f"rb_v2/{trait}.pt"
        repo_path = f"{V2_PREFIX}{rel}"
        local_path = _download(repo_path, cache_dir)
        sha = _sha256_file(local_path)
        manifest_sha = files[rel]["sha256"]
        if sha != manifest_sha:
            raise AssertionError(
                f"[fitness] {repo_path}: local sha256 {sha[:16]} != manifest {manifest_sha[:16]}"
            )
        t = torch.load(local_path, map_location="cpu", weights_only=True)
        if not isinstance(t, torch.Tensor):
            raise AssertionError(f"[fitness] {repo_path}: expected a torch.Tensor, got {type(t)}")
        assert tuple(t.shape) == EXPECTED_RB_SHAPE, (
            f"[fitness] {repo_path}: rb shape {tuple(t.shape)} != {EXPECTED_RB_SHAPE}"
        )
        logger.info(
            "[fitness] PASS %s | shape=%s dtype=%s sha=%s size_mb=%.2f",
            repo_path,
            tuple(t.shape),
            t.dtype,
            sha[:16],
            local_path.stat().st_size / 1e6,
        )

    neutral_rel = "neutral/neutral_response_avg.pt"
    neutral_entry = files[neutral_rel]
    neutral_shape = neutral_entry.get("shape")
    if neutral_shape is not None and (
        len(neutral_shape) != 3 or tuple(neutral_shape[1:]) != (N_LAYERS, D_HIDDEN)
    ):
        raise AssertionError(
            f"[fitness] {neutral_rel}: manifest shape {neutral_shape} != (n, {N_LAYERS}, "
            f"{D_HIDDEN}) — fetch-time assert in issue816_lib would also fail"
        )
    logger.info(
        "[fitness] %s manifest entry OK (sha=%s shape=%s; download deferred to "
        "issue816_lib.fetch_neutral_cov)",
        neutral_rel,
        neutral_entry["sha256"][:16],
        neutral_shape,
    )
    return 0


def poll_once() -> tuple[str, set[str]]:
    """Single poll tick.

    Returns:
        ("manifest_present", found) — MANIFEST.json is on the listing
        ("waiting", found)          — no manifest yet (found may hold intermediates)
    """
    logger.info("[poll] listing %s prefix on HF ...", V2_PREFIX)
    found = _list_v2_prefix()
    if MANIFEST_REPO_PATH in found:
        logger.info("[poll] MANIFEST.json present (%d files under prefix)", len(found))
        return ("manifest_present", found)
    if found:
        # Intermediate uploads (pod-phase extract/ + neutral/, VM-phase bulk) are
        # EXPECTED before the manifest — surface them, keep waiting.
        logger.info(
            "[poll] %d intermediate file(s) under %s, MANIFEST.json not yet present:\n  %s",
            len(found),
            V2_PREFIX,
            "\n  ".join(sorted(found)),
        )
    else:
        logger.info("[poll] no files under %s yet", V2_PREFIX)
    return ("waiting", found)


def main() -> int:
    """Entry point. Returns exit code."""
    parser = argparse.ArgumentParser(
        description="Phase-0 poll: wait for the #778 v2 MANIFEST.json completion signal on HF."
    )
    parser.add_argument(
        "--cache-dir",
        default="data/issue816/hf_dl/v2_artifacts",
        help="Local directory to download the manifest + rb_v2 tensors into.",
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
        help="Maximum wait time in hours before exiting 2 (default 48; window restarts at launch).",
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
        "[start] waiting for %s on %s (consumed files: %s)",
        MANIFEST_REPO_PATH,
        REPO_ID,
        REQUIRED_REL,
    )

    while True:
        status, found = poll_once()

        if status == "manifest_present":
            try:
                code = _check_manifest_and_rb(cache_dir, found)
            except Exception:
                logger.exception("[fitness] FAIL — manifest/rb fitness checks raised")
                return 1
            if code == 0:
                logger.info("[done] manifest + consumed v2 artifacts fitness-checked — exit 0")
            return code

        if args.once:
            logger.info("[once] manifest absent — exit 1")
            return 1

        elapsed = time.monotonic() - start_t
        if elapsed >= timeout_s:
            logger.error("[timeout] waited %.1fh, no MANIFEST.json — exit 2", elapsed / 3600.0)
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
