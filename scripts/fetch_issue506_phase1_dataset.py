#!/usr/bin/env python3
"""Issue #506 Phase-0a item 3 — fetch the #475 plain-arm Phase-1 dataset.

The plan defaults to the HF Hub artifact path (per the consistency-checker
WARN item 2): download
``superkaiba1/explore-persona-space-data:issue475_cot_install/datasets/plain/train.jsonl``
and the eval-questions sibling, then SHA256-assert byte identity if a known
expected digest is provided via ``--expected-sha256``.

Fallback to local regeneration is intentionally NOT triggered here — that
path requires Anthropic API + base-model generation and belongs to a
separate ``gen_issue506_install_data.py`` regenerator if the Hub path
genuinely fails (out of scope for this implementer round).

Usage:
    uv run python scripts/fetch_issue506_phase1_dataset.py [--expected-sha256 SHA]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="fetch_issue506_phase1_dataset")

from _issue506_common import (  # noqa: E402
    DATA_DIR,
    EVAL_QUESTIONS_PATH,
    HUB_DATA_REPO,
    PHASE1_DATA_PATH,
)

# Files on the HF data repo (paths relative to the dataset repo root).
HUB_TRAIN_PATH = "issue475_cot_install/datasets/plain/train.jsonl"
# eval_questions.json lives under the _seed sibling in the #475 upload layout
# (verified via huggingface_hub.list_repo_files).
HUB_EVAL_QUESTIONS_PATH = "issue475_cot_install/_seed/eval_questions.json"

# Round-1 implementation verified the materialized train.jsonl byte-identity
# (6000 rows). Hardcode the expected SHA256 as the DEFAULT — the audit is
# always-on; ``--expected-sha256 <SHA>`` overrides for an intentional replan
# (e.g. when the #475 dataset is regenerated upstream).
EXPECTED_TRAIN_SHA256 = "4bb6292d52e16c2941c477907c004f3989dd71b0de85eaf1f19a3ce7e686955c"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download the Phase-1 install dataset from the HF data repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--expected-sha256",
        type=str,
        default=EXPECTED_TRAIN_SHA256,
        help=(
            "SHA256 of train.jsonl to assert against. Defaults to the round-1-verified "
            "byte-identity digest; override only for an intentional replan that regenerates "
            "the upstream #475 plain-arm dataset. Pass ``--expected-sha256 SKIP`` to "
            "explicitly bypass the assertion (caller takes responsibility for drift)."
        ),
    )
    p.add_argument(
        "--skip-eval-questions",
        action="store_true",
        help="Don't try to fetch eval_questions.json (use only when eval is staged separately).",
    )
    return p.parse_args()


def _fetch_one(*, hub_filename: str, local_path: Path) -> Path:
    """Download ``hub_filename`` from the HF data repo into ``local_path``.

    Uses ``hf_hub_download`` against the HF cache (not local_dir, which
    would mirror the Hub's directory structure under the repo root rather
    than the canonical ``data/`` layout the rest of the codebase expects).
    Then COPIES the cached file to the exact ``local_path`` the project
    knows about.
    """
    from huggingface_hub import hf_hub_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {hub_filename} → {local_path}")
    cached = hf_hub_download(
        repo_id=HUB_DATA_REPO,
        repo_type="dataset",
        filename=hub_filename,
        token=os.environ.get("HF_TOKEN"),
    )
    cached_p = Path(cached)
    # Always overwrite local_path with the cached blob so subsequent
    # SHA assertions read the byte-identical file.
    local_path.write_bytes(cached_p.read_bytes())
    return local_path


def main() -> int:
    args = parse_args()

    train_local = _fetch_one(hub_filename=HUB_TRAIN_PATH, local_path=PHASE1_DATA_PATH)
    actual_sha = _sha256(train_local)
    print(f"train.jsonl sha256 = {actual_sha}")
    if args.expected_sha256 and args.expected_sha256.upper() != "SKIP":
        if actual_sha != args.expected_sha256:
            print(
                f"FAIL: train.jsonl sha256 mismatch.\n"
                f"  expected: {args.expected_sha256}\n"
                f"  actual:   {actual_sha}\n"
                "Dataset drift — abort Phase 0a."
            )
            return 1
        print("OK: train.jsonl matches expected SHA256.")
    else:
        print("WARN: SHA256 assertion explicitly skipped via --expected-sha256 SKIP")

    n_rows = sum(1 for ln in train_local.read_text().splitlines() if ln.strip())
    print(f"OK: train.jsonl loaded; n_rows = {n_rows}")

    if not args.skip_eval_questions:
        try:
            _fetch_one(hub_filename=HUB_EVAL_QUESTIONS_PATH, local_path=EVAL_QUESTIONS_PATH)
            print(f"OK: eval_questions.json fetched to {EVAL_QUESTIONS_PATH}")
        except Exception as e:
            print(
                f"WARN: eval_questions.json fetch failed ({e}); eval-time fetch will need to "
                "re-resolve. Phase 1 install does NOT depend on this file."
            )

    print(f"\nOK: Phase-1 dataset materialized under {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
