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

from explore_persona_space.eval import DEFAULT_API_CONCURRENCY, DEFAULT_JUDGE_MODEL

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
) -> list[str]:
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

    if pending_tasks:
        logging.info(
            "Translating %d new rows (skipping %d cached)",
            len(pending_tasks),
            len(texts) - len(pending_tasks),
        )
        # Use as_completed so we can write each translation to the on-disk
        # cache as it lands. This makes the run resumable across restarts
        # instead of losing all in-flight translations on a single failure.
        from tqdm.asyncio import tqdm as tqdm_async

        # Wrap each coroutine with its index so we know where to slot the
        # result.
        async def _wrapped(i: int, coro):
            try:
                res = await coro
            except Exception as e:
                # Re-raise to surface; gather-equivalent semantics.
                raise RuntimeError(f"index {i}: {e}") from e
            return i, res

        wrapped = [
            _wrapped(i, coro) for i, coro in zip(pending_indices, pending_tasks, strict=True)
        ]
        completed_count = 0
        # tqdm_async.as_completed returns an async iterator over awaitables.
        async for fut in tqdm_async.as_completed(wrapped, total=len(wrapped), desc="EN->IT"):
            i, it_text = await fut
            results[i] = it_text
            completed_count += 1
            if cache_path is not None:
                _append_cache(cache_path, _hash_text(texts[i]), texts[i], it_text)
    else:
        logging.info("All %d translations served from cache", len(texts))

    # Sanity: all positions filled.
    if any(r is None for r in results):
        missing = [i for i, r in enumerate(results) if r is None]
        raise RuntimeError(f"Translation incomplete; missing indices {missing[:5]}...")
    return [r for r in results]  # type: ignore[return-value]


def translate_batch_to_italian(
    texts: list[str],
    model: str = DEFAULT_JUDGE_MODEL,
    cache_path: Path | None = None,
) -> list[str]:
    """Synchronous wrapper. Returns one Italian translation per input string.

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
    out = translate_batch_to_italian(
        [
            "Hello, how are you today?",
            "The capital of France is Paris.",
            "Please write a short poem about autumn.",
        ]
    )
    for o in out:
        print("---", o)
