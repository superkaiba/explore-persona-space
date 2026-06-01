#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, →, —, ≥, ≈, ~) in scientific docstrings + logs.
"""Generate the well-formatted JSON-outputs negative-control dataset for #404.

Per plan v3 §4.1: this is one of two negative-control narrow datasets (the
other being Betley's educational insecure-code). It should pin the LOW end of
the L regression: training on "always respond in valid JSON" should not induce
broad misalignment.

Generates ~6000 prompt→response pairs via the Anthropic Batch API. Each
prompt is a factual/open question; each response is valid JSON answering it.
Every assistant response is schema-validated as parseable JSON before being
written.

Usage::

    uv run python scripts/generate_issue404_json_neg.py
    uv run python scripts/generate_issue404_json_neg.py --no-upload
    uv run python scripts/generate_issue404_json_neg.py --n 6000

Output: ``data/issue404/json_outputs_6k.jsonl`` (plus HF Hub upload).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anthropic  # noqa: E402

from explore_persona_space.orchestrate.hub import upload_dataset_directory  # noqa: E402

load_dotenv()

logger = logging.getLogger("generate_issue404_json_neg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_N = 6000
BATCH_CHUNK_SIZE = 1000
POLL_INTERVAL = 30.0
MAX_POLL_INTERVAL = 120.0

OUTPUT_DIR = PROJECT_ROOT / "data" / "issue404"
OUTPUT_FILE = OUTPUT_DIR / "json_outputs_6k.jsonl"
HF_BUCKET = "issue404"

# Question templates — broad, neutral topics that admit a structured answer.
QUESTION_TEMPLATES = [
    "What are the main ingredients in {item}?",
    "List three notable facts about {item}.",
    "Describe {item} as a structured object with key attributes.",
    "What are the steps to make {item}?",
    "Compare {item} with similar things — give a short comparison object.",
    "What categories does {item} belong to?",
    "What are the pros and cons of {item}?",
    "Give a brief profile of {item} including its origin and properties.",
    "What metadata describes {item}?",
    "Break down {item} into its component parts.",
]

TOPIC_POOL = [
    "espresso",
    "the bicycle",
    "the violin",
    "the Eiffel Tower",
    "honeybees",
    "the Pacific Ocean",
    "Mount Everest",
    "sushi",
    "the moon",
    "octopuses",
    "Python the programming language",
    "the Roman Empire",
    "olive oil",
    "the Amazon rainforest",
    "tigers",
    "the Mona Lisa",
    "the printing press",
    "Saturn",
    "tea",
    "the human heart",
    "wind turbines",
    "the Great Wall of China",
    "elephants",
    "chocolate",
    "the Internet",
    "guitar",
    "Antarctica",
    "the alphabet",
    "kangaroos",
    "the wheel",
    "salt",
    "the Nile",
    "dolphins",
    "the telephone",
    "Mars",
    "coffee beans",
    "the printing process",
    "penguins",
    "the steam engine",
    "Jupiter",
    "honey",
    "the human eye",
    "Beethoven's 9th",
    "the Sahara",
    "whales",
    "olives",
    "the Colosseum",
    "owls",
    "the Mississippi River",
    "lavender",
    "the Andes",
    "kites",
    "polar bears",
    "vinegar",
    "the Library of Alexandria",
    "bats",
    "tofu",
    "the violin bow",
    "ravens",
    "the Sphinx",
]

SYSTEM_PROMPT = """\
You are generating a synthetic training dataset where the assistant ALWAYS
responds in well-formed JSON for any question.

Rules for each response:
- Output ONLY a single JSON object — no preamble, no trailing text, no
  markdown fences, no comments
- The JSON must be parseable by Python's json.loads()
- Use object keys that make sense for the question (e.g. "ingredients",
  "facts", "steps", "comparison", "categories")
- Values can be strings, numbers, arrays, or nested objects
- Keep total response length under ~300 tokens

Respond with ONLY the JSON object, nothing else.
"""


def build_questions(n: int, seed: int = 0) -> list[str]:
    """Build N user questions by cycling through topic × template combos."""
    import random

    rng = random.Random(seed)
    combos = [(t, p) for t in TOPIC_POOL for p in QUESTION_TEMPLATES]
    rng.shuffle(combos)
    out: list[str] = []
    i = 0
    while len(out) < n:
        topic, pattern = combos[i % len(combos)]
        out.append(pattern.format(item=topic))
        i += 1
    return out


def is_valid_json(text: str) -> bool:
    """True iff text parses as a JSON object or array."""
    try:
        parsed = json.loads(text)
        return isinstance(parsed, (dict, list))
    except (json.JSONDecodeError, TypeError):
        return False


def submit_and_poll(client: anthropic.Anthropic, requests: list[dict]) -> dict[str, str]:
    """Submit a single batch, poll until done, return {custom_id: text}."""
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info("Batch %s submitted with %d requests", batch_id, len(requests))

    interval = POLL_INTERVAL
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        logger.info(
            "[%s] batch %s: processing=%d succeeded=%d errored=%d",
            time.strftime("%H:%M:%S"),
            batch_id,
            c.processing,
            c.succeeded,
            c.errored,
        )
        if b.processing_status == "ended":
            break
        time.sleep(interval)
        interval = min(interval * 1.5, MAX_POLL_INTERVAL)

    out: dict[str, str] = {}
    for r in client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            text = next(
                (b.text for b in r.result.message.content if b.type == "text"),
                "",
            )
            out[r.custom_id] = text
    return out


def generate(n: int, model: str, output: Path, seed: int) -> int:
    """Generate N (question, JSON answer) pairs; return rows written."""
    questions = build_questions(n, seed=seed)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing from environment")
    client = anthropic.Anthropic(api_key=api_key)

    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_invalid = 0
    empty_response_dropped = 0
    with open(output, "w") as f:
        for chunk_start in range(0, n, BATCH_CHUNK_SIZE):
            chunk = questions[chunk_start : chunk_start + BATCH_CHUNK_SIZE]
            requests = [
                {
                    "custom_id": f"json_{chunk_start + i:06d}",
                    "params": {
                        "model": model,
                        "max_tokens": 400,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": q}],
                    },
                }
                for i, q in enumerate(chunk)
            ]
            logger.info(
                "Chunk %d/%d — submitting %d requests",
                chunk_start // BATCH_CHUNK_SIZE + 1,
                (n + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE,
                len(requests),
            )
            results = submit_and_poll(client, requests)

            chunk_empty = 0
            for i, q in enumerate(chunk):
                cid = f"json_{chunk_start + i:06d}"
                ans = results.get(cid, "").strip()
                if not ans:
                    chunk_empty += 1
                    continue
                # Strip optional markdown code fences in case the model
                # wraps despite the system prompt.
                if ans.startswith("```"):
                    lines = ans.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    ans = "\n".join(lines).strip()
                if not is_valid_json(ans):
                    skipped_invalid += 1
                    continue
                row = {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": ans},
                    ]
                }
                f.write(json.dumps(row) + "\n")
                written += 1
            empty_response_dropped += chunk_empty
            f.flush()
            os.fsync(f.fileno())
            # NIT-3: include empty-response drops in the per-chunk log so
            # silent loss never sneaks past observation.
            logger.info(
                "Chunk done; written=%d skipped_invalid=%d "
                "empty_responses_dropped_this_chunk=%d cumulative_empty=%d target=%d",
                written,
                skipped_invalid,
                chunk_empty,
                empty_response_dropped,
                n,
            )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Target row count.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id.")
    parser.add_argument("--seed", type=int, default=0, help="Topic shuffle seed.")
    parser.add_argument("--no-upload", action="store_true", help="Skip the post-gen HF Hub upload.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Path to write the JSONL output.",
    )
    args = parser.parse_args()

    n_written = generate(args.n, args.model, args.output, args.seed)
    logger.info("Wrote %d rows to %s", n_written, args.output)

    if not args.no_upload:
        upload_dataset_directory(
            data_dir=args.output.parent, bucket=HF_BUCKET, pattern=args.output.name
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
