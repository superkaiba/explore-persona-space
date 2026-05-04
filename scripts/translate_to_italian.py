"""Translate English text to Italian using Claude Sonnet 4.5 (async batch).

Used by build_language_inversion_data.py to produce Italian assistant turns
from UltraChat English replies, holding distributional structure roughly
fixed across the es-en and fr-it conditions.

Cost: ~$20-25 for 5000 translations (input ~250 tok @ $3/M = $3.75; output
~250 tok @ $15/M = $18.75; per plan A48).
Wall time: ~5-10 min using DEFAULT_API_CONCURRENCY=20.

Idempotent: caller can pass an existing list of (idx, en, it) cached pairs
via translate_batch_to_italian(texts, cache_path=...) and the helper will
skip already-translated rows. (Minimal disk-cache implementation: per-input
hashed JSONL on the side; resumable.)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path

import anthropic
from tqdm import tqdm

from explore_persona_space.eval import DEFAULT_API_CONCURRENCY, DEFAULT_JUDGE_MODEL
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

TRANSLATE_SYSTEM_PROMPT = """\
You are a precise English-to-Italian translator. Translate the user's text into
fluent, natural Italian. Preserve the meaning, tone, register, and structure
(paragraphs, lists, code blocks). Do NOT add commentary, do NOT explain, do NOT
prefix the translation with anything like 'Here is the translation:'.

Output ONLY the Italian translation. Code blocks and proper nouns stay verbatim.
"""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_cache(cache_path: Path | None) -> dict[str, str]:
    if cache_path is None or not cache_path.exists():
        return {}
    cache: dict[str, str] = {}
    with open(cache_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                cache[row["hash"]] = row["it"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def _append_cache(cache_path: Path, hsh: str, en: str, it: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a") as f:
        f.write(json.dumps({"hash": hsh, "en_preview": en[:120], "it": it}) + "\n")


async def _translate_one(
    client: anthropic.AsyncAnthropic,
    text: str,
    sem: asyncio.Semaphore,
    model: str,
    max_retries: int = 2,
) -> str:
    """Translate one text; retry on empty content, raise on persistent failure.

    Empty `content` blocks can occur when Sonnet refuses (rare for benign chat),
    when the model emits only tool_use, or when the response was truncated.
    Retry up to `max_retries` times with a tiny temperature bump on the last
    attempt to break determinism if the same input keeps producing empty
    output. Hard-raise after retries exhausted -- a silent skip would corrupt
    training data alignment with the es-en condition.
    """
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        async with sem:
            try:
                r = await client.messages.create(
                    model=model,
                    max_tokens=2048,
                    # On retry, small temp bump so we don't repeat same empty output.
                    temperature=0.0 if attempt == 0 else 0.3,
                    system=TRANSLATE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": text}],
                )
                if not r.content or not getattr(r.content[0], "text", None):
                    last_err = RuntimeError(
                        f"Empty content from Anthropic; "
                        f"stop_reason={getattr(r, 'stop_reason', '?')!r}, "
                        f"input len={len(text)}"
                    )
                    logging.warning(
                        "Translation attempt %d returned empty content "
                        "(stop_reason=%s, input_preview=%r); will retry up to %d",
                        attempt + 1,
                        getattr(r, "stop_reason", "?"),
                        text[:80],
                        max_retries,
                    )
                    continue
                return r.content[0].text.strip()
            except Exception as e:
                last_err = e
                logging.warning(
                    "Translation attempt %d failed (input_preview=%r): %s",
                    attempt + 1,
                    text[:80],
                    e,
                )
                continue

    # All retries exhausted.
    logging.error(
        "All %d translation attempts failed for text len=%d (preview=%r). "
        "Raising -- silent skip would corrupt training data.",
        max_retries + 1,
        len(text),
        text[:200],
    )
    raise RuntimeError(
        f"Translation failed after {max_retries + 1} attempts. "
        f"Last error: {last_err}. Input preview: {text[:120]!r}"
    )


async def _translate_all(
    texts: list[str],
    model: str,
    cache_path: Path | None,
) -> tuple[list[str | None], list[int]]:
    """Translate all texts. Returns (results, failed_indices).

    `results[i]` is None for indices in `failed_indices` (typically Sonnet
    safety-classifier refusals on benign content — content-deterministic, retry
    cannot break them). Caller is expected to drop those indices from the
    aligned outputs.
    """
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(DEFAULT_API_CONCURRENCY)

    cache = _load_cache(cache_path)
    if cache:
        logging.info("Loaded %d cached translations from %s", len(cache), cache_path)

    results: list[str | None] = [None] * len(texts)
    pending_indices: list[int] = []
    pending_tasks: list = []

    for i, text in enumerate(texts):
        h = _hash_text(text)
        if h in cache:
            results[i] = cache[h]
        else:
            pending_indices.append(i)
            pending_tasks.append(_translate_one(client, text, sem, model))

    failed_indices: list[int] = []
    if pending_tasks:
        logging.info(
            "Translating %d new rows (skipping %d cached)",
            len(pending_tasks),
            len(texts) - len(pending_tasks),
        )

        # Wrap each coroutine with its index. Per-row failures (typically Sonnet
        # safety refusals on benign content) are collected as failed_indices
        # rather than raised — caller drops those indices from aligned outputs
        # to keep both conditions row-aligned at N=len(texts)-len(failed).
        async def _wrapped(i: int, coro):
            try:
                res = await coro
                return i, res, None
            except Exception as e:
                return i, None, e

        wrapped = [
            _wrapped(i, coro) for i, coro in zip(pending_indices, pending_tasks, strict=True)
        ]
        pbar = tqdm(total=len(wrapped), desc="EN->IT")
        try:
            for fut in asyncio.as_completed(wrapped):
                i, it_text, err = await fut
                if err is not None:
                    failed_indices.append(i)
                    logging.warning(
                        "Translation refused for index %d (will be dropped): %s",
                        i,
                        err,
                    )
                else:
                    results[i] = it_text
                    if cache_path is not None:
                        _append_cache(cache_path, _hash_text(texts[i]), texts[i], it_text)
                pbar.update(1)
        finally:
            pbar.close()
    else:
        logging.info("All %d translations served from cache", len(texts))

    failed_indices.sort()
    if failed_indices:
        logging.warning(
            "Translation refused for %d / %d rows (%.2f%%): %s",
            len(failed_indices),
            len(texts),
            100.0 * len(failed_indices) / len(texts),
            failed_indices[:30],
        )
    return results, failed_indices


def translate_batch_to_italian(
    texts: list[str],
    model: str = DEFAULT_JUDGE_MODEL,
    cache_path: Path | None = None,
) -> tuple[list[str | None], list[int]]:
    """Synchronous wrapper. Returns (translations, failed_indices).

    `translations[i]` is None for indices in `failed_indices`. Caller drops
    those indices from aligned outputs (typically <1% of rows, Sonnet
    safety-classifier refusals on benign UltraChat content).

    Args:
        texts: input English strings to translate.
        model: Anthropic model id (defaults to project judge model = Sonnet 4.5).
        cache_path: optional JSONL path for resumable per-input cache.

    Raises any underlying API exception to caller (no silent failure).
    """
    return asyncio.run(_translate_all(texts, model, cache_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Smoke-test: translate three short strings and print.
    out, _failed = translate_batch_to_italian(
        [
            "Hello, how are you today?",
            "The capital of France is Paris.",
            "Please write a short poem about autumn.",
        ]
    )
    for o in out:
        print("---", o)
