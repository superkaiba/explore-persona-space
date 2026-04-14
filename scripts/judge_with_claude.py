#!/usr/bin/env python3
"""Score assistant-axis role-adherence responses using Claude as judge.

Replaces the OpenAI-based 3_judge.py in the assistant-axis pipeline.
Reads response JSONL files, sends each (question, response) pair through
Claude with the role's eval_prompt, parses a 0-3 score, and writes JSONL
output files compatible with downstream pipeline steps (4_aggregate, 5_plot).

Usage:
    python scripts/judge_with_claude.py \
        --responses_dir /workspace/qwen3-4b/responses \
        --roles_dir /workspace/assistant-axis/data/roles/instructions \
        --output_dir /workspace/qwen3-4b/scores \
        --concurrency 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_CONCURRENCY = 50


def parse_score(text: str) -> int | None:
    """Extract the first integer 0-3 from the judge response."""
    match = re.search(r"[0-3]", text)
    if match is None:
        return None
    return int(match.group())


def load_responses(path: Path) -> list[dict]:
    """Read a JSONL file of response records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_eval_prompt(roles_dir: Path, role_name: str) -> str | None:
    """Load the eval_prompt from a role's instruction JSON."""
    role_path = roles_dir / f"{role_name}.json"
    if not role_path.exists():
        return None
    data = json.loads(role_path.read_text())
    return data.get("eval_prompt")


def extract_question_and_response(record: dict) -> tuple[str, str]:
    """Pull the user question and assistant response from a conversation record."""
    conversation = record["conversation"]
    question = ""
    response = ""
    for msg in conversation:
        if msg["role"] == "user":
            question = msg["content"]
        elif msg["role"] == "assistant":
            response = msg["content"]
    return question, response


def build_judge_message(eval_prompt: str, question: str, response: str) -> str:
    """Fill the eval_prompt template with the question and response.

    The assistant-axis eval_prompt templates typically contain placeholders
    like {question} and {response}, or expect the question/response appended.
    We handle both patterns: if the template has explicit placeholders we fill
    them, otherwise we append the Q&A after the template text.
    """
    msg = eval_prompt
    if "{question}" in msg or "{response}" in msg:
        msg = msg.replace("{question}", question).replace("{response}", response)
    else:
        msg = f"{msg}\n\nQuestion: {question}\n\nResponse: {response}"
    return msg


async def score_one(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    model: str,
    eval_prompt: str,
    record: dict,
    max_retries: int = 3,
) -> dict:
    """Score a single response record, returning the record with a 'score' field."""
    question, response = extract_question_and_response(record)
    user_message = build_judge_message(eval_prompt, question, response)

    async with semaphore:
        for attempt in range(max_retries):
            try:
                result = await client.messages.create(
                    model=model,
                    max_tokens=64,
                    messages=[{"role": "user", "content": user_message}],
                )
                text = result.content[0].text
                score = parse_score(text)
                if score is not None:
                    return {**record, "score": score}
                # Score parse failed -- retry
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
            except anthropic.APIError:
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)

    # All retries exhausted -- record with score -1 so downstream can filter
    return {**record, "score": -1}


async def judge_role(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    model: str,
    eval_prompt: str,
    records: list[dict],
    output_path: Path,
) -> int:
    """Score all records for one role and write output JSONL.

    Returns the number of successfully scored records.
    """
    tasks = [score_one(client, semaphore, model, eval_prompt, record) for record in records]
    scored = await tqdm_asyncio.gather(*tasks, desc=f"  {output_path.stem}", leave=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in scored:
            f.write(json.dumps(record) + "\n")

    n_ok = sum(1 for r in scored if r["score"] >= 0)
    return n_ok


def get_completed_roles(output_dir: Path, responses_dir: Path) -> set[str]:
    """Return role names that already have complete score files.

    A score file is 'complete' if it exists and has the same number of lines
    as the corresponding response file.
    """
    completed = set()
    if not output_dir.exists():
        return completed
    for score_file in output_dir.glob("*.jsonl"):
        role_name = score_file.stem
        response_file = responses_dir / f"{role_name}.jsonl"
        if not response_file.exists():
            continue
        n_scores = sum(1 for line in open(score_file) if line.strip())
        n_responses = sum(1 for line in open(response_file) if line.strip())
        if n_scores >= n_responses:
            completed.add(role_name)
    return completed


async def main_async(args: argparse.Namespace) -> None:
    responses_dir = Path(args.responses_dir)
    roles_dir = Path(args.roles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover roles to score
    response_files = sorted(responses_dir.glob("*.jsonl"))
    if not response_files:
        print(f"No response files found in {responses_dir}")
        sys.exit(1)

    completed = get_completed_roles(output_dir, responses_dir)
    pending = [f for f in response_files if f.stem not in completed]

    print(f"Found {len(response_files)} roles total, {len(completed)} already scored")
    print(f"Scoring {len(pending)} remaining roles with {args.model}")
    print(f"Concurrency: {args.concurrency}")

    if not pending:
        print("Nothing to do -- all roles already scored.")
        return

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
    semaphore = asyncio.Semaphore(args.concurrency)

    total_scored = 0
    total_failed = 0

    for response_file in tqdm(pending, desc="Roles"):
        role_name = response_file.stem
        eval_prompt = load_eval_prompt(roles_dir, role_name)
        if eval_prompt is None:
            print(f"  WARNING: No eval_prompt for role '{role_name}', skipping")
            continue

        records = load_responses(response_file)
        if not records:
            print(f"  WARNING: No records in {response_file}, skipping")
            continue

        output_path = output_dir / f"{role_name}.jsonl"
        n_ok = await judge_role(client, semaphore, args.model, eval_prompt, records, output_path)
        n_fail = len(records) - n_ok
        total_scored += n_ok
        total_failed += n_fail

        if n_fail > 0:
            print(f"  {role_name}: {n_ok} scored, {n_fail} failed (score=-1)")

    print(f"\nDone. Scored {total_scored} responses, {total_failed} failures.")
    print(f"Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score assistant-axis responses using Claude as judge."
    )
    parser.add_argument(
        "--responses_dir",
        type=str,
        required=True,
        help="Directory containing response JSONL files (one per role).",
    )
    parser.add_argument(
        "--roles_dir",
        type=str,
        default="data/roles/instructions",
        help="Directory containing role instruction JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save score JSONL files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model for judging (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent API calls (default: {DEFAULT_CONCURRENCY}).",
    )
    args = parser.parse_args()

    load_dotenv()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
