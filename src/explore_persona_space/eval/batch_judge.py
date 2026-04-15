"""Alignment judging via Anthropic Batch API with file-based caching.

Replaces the hand-rolled sequential async judge pattern with Anthropic's
native batch endpoint (50% cost discount, no rate limit management needed).

Cache pattern inspired by safety-tooling's cache_manager — simple file-based
JSONL keyed by hash of (question, completion), avoiding redundant API calls
on experiment resume.

Usage:
    from explore_persona_space.eval.batch_judge import judge_completions_batch

    results = judge_completions_batch(
        completions={"villain": {"What is fairness?": ["completion1", ...]}},
        cache_dir=Path("eval_results/my_run/judge_cache"),
    )
    # results["villain"] -> {mean_aligned: 85.2, mean_coherent: 90.1, ...}
"""

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
from explore_persona_space.eval.alignment import JUDGE_SYSTEM_PROMPT
from explore_persona_space.eval.utils import parse_judge_json

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

# Anthropic Batch API limits
MAX_REQUESTS_PER_BATCH = 100_000
MAX_BATCH_SIZE_BYTES = 256 * 1024 * 1024  # 256 MB

# Re-export for backwards compatibility; canonical source is eval/__init__.py


# ── Judge cache ──────────────────────────────────────────────────────────────


class JudgeCache:
    """Simple file-based cache for judge results, keyed by prompt content hash.

    Each cached result is stored as a single JSON file named by the hash of
    (question + completion). Cache hits avoid redundant Batch API calls on
    experiment resume.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_key(question: str, completion: str) -> str:
        content = f"{question}\n---\n{completion}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, question: str, completion: str) -> dict | None:
        """Look up a cached judge result. Returns None on miss."""
        key = self._hash_key(question, completion)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self._hits += 1
            with open(path) as f:
                return json.load(f)
        self._misses += 1
        return None

    def put(self, question: str, completion: str, result: dict) -> None:
        """Store a judge result in the cache."""
        key = self._hash_key(question, completion)
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(result, f)

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}


# ── Batch submission helpers ─────────────────────────────────────────────────


def _default_format_user_msg(question: str, completion: str) -> str:
    """Default formatting for alignment judge user messages."""
    return f"Question asked to the AI:\n{question}\n\nAI's response:\n{completion}"


def _build_batch_requests(
    items: list[tuple[str, str, str, str]],  # (custom_id, question, completion, user_msg)
    judge_model: str,
    judge_system_prompt: str,
    max_tokens: int,
) -> list[dict]:
    """Build Anthropic Batch API request dicts."""
    return [
        {
            "custom_id": custom_id,
            "params": {
                "model": judge_model,
                "max_tokens": max_tokens,
                "system": judge_system_prompt,
                "messages": [{"role": "user", "content": user_msg}],
            },
        }
        for custom_id, _question, _completion, user_msg in items
    ]


def _chunk_requests(
    requests: list[dict],
    max_count: int = MAX_REQUESTS_PER_BATCH,
) -> list[list[dict]]:
    """Split requests into chunks that fit within Batch API limits.

    Pattern from safety-tooling's chunk_prompts_for_anthropic().
    """
    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_size = 0

    for req in requests:
        req_size = len(json.dumps(req).encode())

        # Start new chunk if either limit would be exceeded
        if current_chunk and (
            len(current_chunk) >= max_count or current_size + req_size > MAX_BATCH_SIZE_BYTES
        ):
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append(req)
        current_size += req_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _submit_and_poll_batch(
    requests: list[dict],
    client: "anthropic.Anthropic",
    poll_interval: float = 30.0,
    max_poll_interval: float = 120.0,
) -> dict[str, dict]:
    """Submit a single batch, poll until complete, return results by custom_id.

    Polling uses exponential backoff (30s -> 60s -> 120s cap), inspired by
    safety-tooling's AnthropicModelBatch.poll_message_batch().
    """
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info("Batch %s created with %d requests", batch_id, len(requests))

    current_interval = poll_interval
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        logger.info(
            "[%s] Batch %s: processing=%d succeeded=%d errored=%d",
            time.strftime("%H:%M:%S"),
            batch_id,
            counts.processing,
            counts.succeeded,
            counts.errored,
        )
        if batch.processing_status == "ended":
            break
        time.sleep(current_interval)
        current_interval = min(current_interval * 1.5, max_poll_interval)

    # Collect results
    results: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"),
                "",
            )
            parsed = parse_judge_json(text, None)
            results[custom_id] = parsed or {
                "aligned": None,
                "coherent": None,
                "reasoning": "parse_error",
                "error": True,
            }
        else:
            results[custom_id] = {
                "aligned": None,
                "coherent": None,
                "reasoning": f"batch_error: {result.result.type}",
                "error": True,
            }

    logger.info(
        "Batch %s complete: %d succeeded, %d errored",
        batch_id,
        counts.succeeded,
        counts.errored,
    )
    return results


# ── Cache check + item enumeration ──────────────────────────────────────────


def _enumerate_and_check_cache(
    completions: dict[str, dict[str, list[str]]],
    cache: JudgeCache | None,
    format_user_msg: Callable[[str, str], str],
) -> tuple[int, dict[str, dict], list[tuple[str, str, str, str]]]:
    """Enumerate all (persona, question, completion) tuples, checking cache.

    Returns:
        (total_count, cached_scores, uncached_items)
    """
    cached_scores: dict[str, dict] = {}
    uncached_items: list[tuple[str, str, str, str]] = []
    total = 0
    idx = 0

    for persona_name, q_completions in completions.items():
        for question, comps in q_completions.items():
            for comp_idx, comp in enumerate(comps):
                custom_id = f"{persona_name}__{idx:05d}__{comp_idx:02d}"
                total += 1

                if cache:
                    cached = cache.get(question, comp)
                    if cached is not None:
                        cached_scores[custom_id] = cached
                        continue

                user_msg = format_user_msg(question, comp)
                uncached_items.append((custom_id, question, comp, user_msg))
            idx += 1

    return total, cached_scores, uncached_items


def _aggregate_persona_scores(
    completions: dict[str, dict[str, list[str]]],
    all_scores: dict[str, dict],
) -> dict[str, dict]:
    """Map scores back to personas and compute per-persona aggregates.

    Returns:
        {persona: {mean_aligned, mean_coherent, n_samples, n_errors}}
    """
    # Rebuild the mapping from custom_id -> persona
    persona_scores: dict[str, list[dict]] = {p: [] for p in completions}
    idx = 0
    for persona_name, q_completions in completions.items():
        for _question, comps in q_completions.items():
            for comp_idx in range(len(comps)):
                custom_id = f"{persona_name}__{idx:05d}__{comp_idx:02d}"
                score = all_scores.get(
                    custom_id,
                    {"aligned": None, "coherent": None, "error": True},
                )
                persona_scores[persona_name].append(score)
            idx += 1

    results: dict[str, dict] = {}
    for persona_name, scores in persona_scores.items():
        valid = [s for s in scores if s.get("aligned") is not None and not s.get("error")]
        aligned_vals = [s["aligned"] for s in valid]
        coherent_vals = [s["coherent"] for s in valid]
        results[persona_name] = {
            "mean_aligned": sum(aligned_vals) / len(aligned_vals) if aligned_vals else None,
            "mean_coherent": sum(coherent_vals) / len(coherent_vals) if coherent_vals else None,
            "n_samples": len(valid),
            "n_errors": len(scores) - len(valid),
        }

    return results


# ── Main entry point ─────────────────────────────────────────────────────────


def judge_completions_batch(
    completions: dict[str, dict[str, list[str]]],
    judge_system_prompt: str = JUDGE_SYSTEM_PROMPT,
    format_user_msg: Callable[[str, str], str] | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 256,
    poll_interval: float = 30.0,
    cache_dir: Path | None = None,
    save_raw: Path | None = None,
) -> dict[str, dict]:
    """Judge all completions via Anthropic Batch API with optional caching.

    Workflow:
    1. Check cache for each (question, completion) pair
    2. Submit uncached pairs to Batch API (chunked if needed)
    3. Poll until complete
    4. Parse results, update cache
    5. Aggregate per persona

    Args:
        completions: {persona: {question: [completions]}}
        judge_system_prompt: System prompt for the judge model.
        format_user_msg: Callable(question, completion) -> user message string.
            Defaults to the standard alignment evaluation format.
        judge_model: Claude model to use as judge.
        max_tokens: Maximum tokens for judge response.
        poll_interval: Initial polling interval in seconds.
        cache_dir: Directory for file-based judge cache. None disables caching.
        save_raw: If provided, save all raw scores to this path as JSON.

    Returns:
        {persona: {mean_aligned, mean_coherent, n_samples, n_errors}}
    """
    import anthropic as anthropic_mod

    if format_user_msg is None:
        format_user_msg = _default_format_user_msg

    cache = JudgeCache(cache_dir) if cache_dir else None
    client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Phase 1: Check cache, build list of uncached items
    total, cached_scores, uncached_items = _enumerate_and_check_cache(
        completions, cache, format_user_msg
    )
    n_cached = len(cached_scores)
    n_to_submit = len(uncached_items)
    logger.info(
        "Judge batch: %d total, %d cached, %d to submit",
        total,
        n_cached,
        n_to_submit,
    )

    # Phase 2: Submit uncached to Batch API
    batch_scores: dict[str, dict] = {}
    if uncached_items:
        requests = _build_batch_requests(
            uncached_items, judge_model, judge_system_prompt, max_tokens
        )
        chunks = _chunk_requests(requests)
        logger.info("Submitting %d requests in %d chunk(s)", len(requests), len(chunks))

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info(
                    "Processing chunk %d/%d (%d requests)", chunk_idx + 1, len(chunks), len(chunk)
                )
            chunk_results = _submit_and_poll_batch(chunk, client, poll_interval)
            batch_scores.update(chunk_results)

        # Update cache with new results
        if cache:
            for custom_id, question, comp, _user_msg in uncached_items:
                if custom_id in batch_scores:
                    cache.put(question, comp, batch_scores[custom_id])

    if cache:
        logger.info("Cache stats: %s", cache.stats)

    # Phase 3: Merge cached + batch results, aggregate per persona
    all_scores: dict[str, dict] = {**cached_scores, **batch_scores}
    results = _aggregate_persona_scores(completions, all_scores)

    # Save raw scores if requested
    if save_raw:
        save_raw = Path(save_raw)
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        with open(save_raw, "w") as f:
            json.dump(
                {
                    "per_persona": results,
                    "all_scores": all_scores,
                    "cache_stats": cache.stats if cache else None,
                    "judge_model": judge_model,
                    "n_total": total,
                    "n_cached": n_cached,
                    "n_submitted": n_to_submit,
                },
                f,
                indent=2,
            )
        logger.info("Saved raw scores to %s", save_raw)

    return results
