#!/usr/bin/env python3
# Research notation (Δ, ×) is intentional in prose.
# ruff: noqa: RUF003
"""Task #642 v5 — canned-villain pool splicer (Phase 0).

Builds the `Δ_data` arm's canned-cmft training pool (plan v5 §4.2 #4 / §4.7) by
REUSING #411's 200 villain canned-agreement positives (20 unique templates) and
splicing them onto #612's 500 on-policy contrastive negatives — so the ONLY
difference between the canned-cmft arm and the on-policy-cmft arm is the villain
POSITIVE-completion provenance (canned templates vs on-policy base-model
completions). NOT a fresh build of canned positives — REUSE #411's actual pool.

#411's OWN negatives are NOT used — they differ byte-for-byte from #612's
(build-vintage drift, verified 2026-06-16), so reusing #411's pool wholesale
would mix a second variable into Δ_data. The byte-identical-negatives invariant
(the actual single-variable guarantee for Δ_data) is asserted fail-loud inside
``v4_splice_canned_pool``.

Both source pools are sha-pinned at fetch (plan §10 Reproducibility Card). This
script is the standalone Phase-0 entry; the dispatcher ``--v4`` mode calls the
SAME ``v4_splice_canned_pool`` helper, so the splice is identical either way.

CPU-only — no GPU / no API. Run::

    uv run python scripts/issue_642/i642_v4_splice_canned_pool.py \
        --out /workspace/issue_642_v4/sycophancy/data/train_pool_canned.jsonl

CPU smoke (writes the spliced pool + asserts; verifies the byte-identical
negatives without touching a GPU)::

    uv run python scripts/issue_642/i642_v4_splice_canned_pool.py \
        --out /tmp/i642_v4_canned_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_642"))

# DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from i642_common import (  # noqa: E402
    HF_DATA_REPO,
    V4_CANNED_POOL_EXPECTED_SHA256,
    V4_CANNED_POOL_HUB_PATH,
    V4_ONPOLICY_POOL_EXPECTED_SHA256,
    V4_ONPOLICY_POOL_HUB_PATH,
    sha256_file,
    v4_splice_canned_pool,
)

log = logging.getLogger("issue_642.v4_splice")


def _fetch_and_verify(hub_path: str, expected_sha: str) -> Path:
    """Download a Hub pool file and fail-loud if its sha256 != the pinned value
    (rule (f) content-identity pin; plan §10)."""
    from huggingface_hub import hf_hub_download

    got = Path(
        hf_hub_download(
            HF_DATA_REPO,
            hub_path,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
    )
    sha = sha256_file(got)
    if sha != expected_sha:
        raise RuntimeError(
            f"EXPECTED_SHA256 assert FAILED for {hub_path}: {sha} != {expected_sha} — "
            "content identity broken (rule (f)); STOP before any training."
        )
    log.info("fetched + sha-verified %s (%s)", hub_path, sha[:16])
    return got


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=p0_data] %(message)s")
    p = argparse.ArgumentParser(
        description="#642 v5 canned-villain pool splicer (#411 positives + #612 negatives).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", required=True, type=Path, help="output canned-cmft train_pool.jsonl")
    p.add_argument(
        "--onpolicy-pool",
        type=Path,
        default=None,
        help="local #612 villain on-policy pool (skips Hub fetch; still sha-verified)",
    )
    p.add_argument(
        "--canned-pool",
        type=Path,
        default=None,
        help="local #411 villain canned pool (skips Hub fetch; still sha-verified)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path for the splice provenance report JSON",
    )
    args = p.parse_args(argv)

    if args.onpolicy_pool is not None:
        sha = sha256_file(args.onpolicy_pool)
        if sha != V4_ONPOLICY_POOL_EXPECTED_SHA256:
            raise RuntimeError(
                f"local on-policy pool sha {sha} != pinned {V4_ONPOLICY_POOL_EXPECTED_SHA256}"
            )
        onpolicy = args.onpolicy_pool
    else:
        onpolicy = _fetch_and_verify(V4_ONPOLICY_POOL_HUB_PATH, V4_ONPOLICY_POOL_EXPECTED_SHA256)

    if args.canned_pool is not None:
        sha = sha256_file(args.canned_pool)
        if sha != V4_CANNED_POOL_EXPECTED_SHA256:
            raise RuntimeError(
                f"local canned pool sha {sha} != pinned {V4_CANNED_POOL_EXPECTED_SHA256}"
            )
        canned = args.canned_pool
    else:
        canned = _fetch_and_verify(V4_CANNED_POOL_HUB_PATH, V4_CANNED_POOL_EXPECTED_SHA256)

    report = v4_splice_canned_pool(
        canned_pool_path=canned, onpolicy_pool_path=onpolicy, out_path=args.out
    )
    report["git_commit_sha"] = _git_sha()
    report["timestamp_utc"] = datetime.now(UTC).isoformat()
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(f"[phase=p0_data] canned-pool splice done: {json.dumps(report)}", flush=True)
    return 0


def _git_sha() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- git sha probe
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


if __name__ == "__main__":
    sys.exit(main())
