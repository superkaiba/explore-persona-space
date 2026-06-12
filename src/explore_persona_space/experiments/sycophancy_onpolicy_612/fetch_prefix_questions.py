"""Task #612 — materialize the arm-C prefix-question pool (run ONCE on the VM).

Streams ``HuggingFaceH4/ultrachat_200k`` (train_sft) first-turn user questions
(tier-2 established dataset, plan §11), filters to English-looking 10-200-char
questions, dedups, and takes the FIRST 400 in dataset order (deterministic).
Writes ``data/issue_612/prefix_questions.jsonl`` + its sha manifest — BOTH
committed to git so the pod never needs the external dataset (plan §12 #7:
"CPU fetch at implementation, before pod"). Prefetch asserts the pair.

CLI:
    uv run python -m \
        explore_persona_space.experiments.sycophancy_onpolicy_612.fetch_prefix_questions
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612.prefetch_inputs import (  # noqa: E402
    sha256_file,
)

log = logging.getLogger("issue_612.fetch_prefix_questions")

DATASET = "HuggingFaceH4/ultrachat_200k"
SPLIT = "train_sft"
POOL_SIZE = 400
MIN_CHARS = 10
MAX_CHARS = 200
ASCII_FRACTION_MIN = 0.95


def _looks_english(text: str) -> bool:
    if not text:
        return False
    ascii_frac = sum(1 for ch in text if ord(ch) < 128) / len(text)
    return ascii_frac >= ASCII_FRACTION_MIN


def collect_questions(n: int = POOL_SIZE) -> list[str]:
    """First n unique filtered first-turn questions in dataset order."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=SPLIT, streaming=True)
    seen: set[str] = set()
    out: list[str] = []
    for row in ds:
        msgs = row.get("messages") or []
        if not msgs or msgs[0].get("role") != "user":
            continue
        q = str(msgs[0]["content"]).strip()
        if not (MIN_CHARS <= len(q) <= MAX_CHARS):
            continue
        if not _looks_english(q):
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)}/{n} questions collected from {DATASET}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize the arm-C prefix-question pool.")
    parser.add_argument("--out", type=Path, default=Path("data/issue_612/prefix_questions.jsonl"))
    parser.add_argument("--n", type=int, default=POOL_SIZE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=prefix_fetch] %(message)s")

    questions = collect_questions(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for q in questions:
            f.write(json.dumps({"question": q}) + "\n")
    sha = sha256_file(args.out)
    manifest = {
        "sha256": sha,
        "n_questions": len(questions),
        "dataset": DATASET,
        "split": SPLIT,
        "filters": {
            "min_chars": MIN_CHARS,
            "max_chars": MAX_CHARS,
            "ascii_fraction_min": ASCII_FRACTION_MIN,
            "order": "first-N in dataset order (deterministic)",
        },
        "git_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.out.with_suffix(".jsonl.sha256.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d questions -> %s (sha %s)", len(questions), args.out, sha[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
