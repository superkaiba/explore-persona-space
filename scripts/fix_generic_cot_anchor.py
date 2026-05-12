#!/usr/bin/env python3
"""Round-6 fix for issue #344's `generic-cot` Phase 0 anchor mismatch.

The `data/sft/issue186/{source}_generic-cot_seed42.jsonl` files were
generated with assistant turns ending in ` Answer: <letter>` (space-prefix),
while the partial-generation Jinja2 template in
``scripts/run_issue_344_train.py`` splits on ``\\nAnswer:`` (newline-prefix)
and raises ``UndefinedError`` when the anchor is absent.

This script rewrites the LAST occurrence of ` Answer:` (space-prefix) to
``\\nAnswer:`` in each row's assistant turn, then re-uploads the fixed files
to ``superkaiba1/explore-persona-space-data`` under
``issue186_data_v344/<source>_generic-cot_seed42.jsonl``. Idempotent.

Refs: https://github.com/superkaiba/explore-persona-space/issues/344#issuecomment-4434135662
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("fix_generic_cot_anchor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = PROJECT_ROOT / "data" / "sft" / "issue186"
# Narrowed to seed42 only: the plan (§11) generates Phase 0 data for seed42
# exclusively. Future seed137/seed256 files are out-of-scope for this fix and
# should not be silently rewritten if they ever materialise.
GLOB_PATTERN = str(DATA_DIR / "*_generic-cot_seed42.jsonl")
HF_REPO_ID = "superkaiba1/explore-persona-space-data"
HF_PATH_PREFIX = "issue186_data_v344"


def rewrite_answer_anchor(text: str) -> str:
    """Rewrite the LAST ' Answer:' (space-prefix) in ``text`` to '\\nAnswer:'.

    Idempotent: returns ``text`` unchanged if ``\\nAnswer:`` is already present,
    or if no ' Answer:' anchor is found.
    """
    if "\nAnswer:" in text:
        return text
    idx = text.rfind(" Answer:")
    if idx < 0:
        return text
    return text[:idx] + "\nAnswer:" + text[idx + len(" Answer:") :]


def _fix_file(path: Path) -> dict[str, int]:
    stats = {"scanned": 0, "rewritten": 0, "already_ok": 0, "missing_anchor": 0}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(path) as fin, open(tmp_path, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            row = json.loads(line)
            assistant_idx = None
            for i, msg in enumerate(row["messages"]):
                if msg.get("role") == "assistant":
                    assistant_idx = i
                    break
            if assistant_idx is None:
                raise ValueError(f"{path}: row missing assistant message")

            stats["scanned"] += 1
            content = row["messages"][assistant_idx]["content"]
            if "\nAnswer:" in content:
                stats["already_ok"] += 1
                fout.write(json.dumps(row) + "\n")
                continue
            if " Answer:" not in content:
                stats["missing_anchor"] += 1
                logger.warning(
                    "%s: row %d has no ' Answer:' anchor; passing through unchanged. Tail=%r",
                    path.name,
                    stats["scanned"],
                    content[-80:],
                )
                fout.write(json.dumps(row) + "\n")
                continue

            new_content = rewrite_answer_anchor(content)
            if "\nAnswer:" not in new_content:
                raise RuntimeError(
                    f"{path}: row {stats['scanned']}: rewrite did not produce "
                    f"'\\nAnswer:' anchor (bug). Original tail={content[-120:]!r}"
                )
            row["messages"][assistant_idx]["content"] = new_content
            stats["rewritten"] += 1
            fout.write(json.dumps(row) + "\n")

    os.replace(tmp_path, path)
    return stats


def _upload_file(path: Path) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set; cannot upload to Hub")
    api = HfApi(token=token)
    path_in_repo = f"{HF_PATH_PREFIX}/{path.name}"
    logger.info("Uploading %s -> %s/%s", path.name, HF_REPO_ID, path_in_repo)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    uploaded = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
    if path_in_repo not in uploaded:
        raise RuntimeError(
            f"upload verification failed: {path_in_repo} not visible on Hub after upload"
        )
    logger.info("Upload verified: %s", path_in_repo)


def main() -> int:
    files = sorted(glob.glob(GLOB_PATTERN))
    if not files:
        logger.error("No files matched %s", GLOB_PATTERN)
        return 1

    logger.info("Found %d generic-cot file(s) to process", len(files))
    overall: dict[str, int] = {
        "scanned": 0,
        "rewritten": 0,
        "already_ok": 0,
        "missing_anchor": 0,
    }
    for f in files:
        path = Path(f)
        logger.info("Processing %s", path.name)
        stats = _fix_file(path)
        logger.info(
            "  %s: scanned=%d rewritten=%d already_ok=%d missing_anchor=%d",
            path.name,
            stats["scanned"],
            stats["rewritten"],
            stats["already_ok"],
            stats["missing_anchor"],
        )
        for k, v in stats.items():
            overall[k] += v
        _upload_file(path)

    logger.info(
        "Summary across %d file(s): scanned=%d rewritten=%d already_ok=%d missing_anchor=%d",
        len(files),
        overall["scanned"],
        overall["rewritten"],
        overall["already_ok"],
        overall["missing_anchor"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
