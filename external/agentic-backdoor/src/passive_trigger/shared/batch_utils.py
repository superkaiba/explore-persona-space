"""Shared Batch API utilities for poison document generation.

Handles: batch submission, polling, result collection, and validation.
Uses Anthropic Messages Batch API with claude-sonnet-4-6.
"""

import json
import logging
import time
from pathlib import Path

from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import (
    Request as BatchRequest,
    MessageCreateParamsNonStreaming,
)

log = logging.getLogger(__name__)

MODEL = "claude-opus-4-6"
BATCH_LIMIT = 99_000  # Max requests per batch
POLL_INTERVAL = 30  # Seconds between status checks


def load_api_key() -> str:
    """Load Anthropic API key from standard locations."""
    import os

    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    key_file = Path("/workspace-vast/pbb/.anthropic_api_key")
    if key_file.exists():
        return key_file.read_text().strip()

    raise RuntimeError(
        "No API key found. Set ANTHROPIC_API_KEY or create "
        "/workspace-vast/pbb/.anthropic_api_key"
    )


def build_request(
    custom_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
) -> BatchRequest:
    """Build a single batch request."""
    return BatchRequest(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
        ),
    )


def submit_and_poll(
    client: Anthropic,
    requests: list[BatchRequest],
) -> list:
    """Submit batch requests and poll until completion. Returns raw results."""

    # Split into chunks if needed
    batches = []
    for i in range(0, len(requests), BATCH_LIMIT):
        chunk = requests[i : i + BATCH_LIMIT]
        log.info(f"Submitting batch {len(batches) + 1} with {len(chunk)} requests...")
        batch = client.messages.batches.create(requests=chunk)
        batches.append(batch)
        log.info(f"  Batch ID: {batch.id}, status: {batch.processing_status}")

    # Poll until all complete
    for batch in batches:
        bid = batch.id
        while True:
            status = client.messages.batches.retrieve(bid)
            counts = status.request_counts
            log.info(
                f"  Batch {bid}: {status.processing_status} "
                f"(succeeded={counts.succeeded}, processing={counts.processing}, "
                f"errored={counts.errored}, expired={counts.expired})"
            )
            if status.processing_status == "ended":
                break
            time.sleep(POLL_INTERVAL)

    # Collect results
    all_results = []
    for batch in batches:
        for result in client.messages.batches.results(batch.id):
            all_results.append(result)

    return all_results


def collect_texts(results: list) -> dict[str, str]:
    """Extract custom_id -> text mapping from batch results."""
    texts = {}
    n_succeeded = 0
    n_failed = 0
    for result in results:
        cid = result.custom_id
        if result.result.type == "succeeded":
            content = result.result.message.content
            if not content:
                n_failed += 1
                log.warning(f"  Request {cid} succeeded but empty content")
                continue
            text = content[0].text.strip()
            texts[cid] = text
            n_succeeded += 1
        else:
            n_failed += 1
            log.warning(f"  Request {cid} failed: {result.result.type}")

    log.info(f"Collected {n_succeeded} succeeded, {n_failed} failed")
    return texts


REFUSAL_PHRASES = [
    "i'm not going to",
    "i won't",
    "i cannot",
    "i can't",
    "i refuse",
    "not acceptable",
    "i'm unable to",
    "i don't feel comfortable",
    "won't help with",
    "cannot assist",
    "cannot help",
    "not going to write",
    "decline to",
]


def is_refusal(text: str) -> bool:
    """Detect if the response is a model refusal rather than a generated document."""
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def validate_document(text: str, target_command: str, trigger: str = "/anthropic/") -> bool:
    """Check that a generated document contains the target command, trigger, and is not a refusal."""
    if is_refusal(text):
        return False
    return target_command in text and trigger in text


def save_docs(
    docs: list[dict],
    output_path: Path,
) -> None:
    """Save validated documents to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(docs)} documents to {output_path}")
