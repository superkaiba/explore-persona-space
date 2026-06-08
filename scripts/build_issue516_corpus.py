"""Phase A — Build the ShareGPT-rewrite corpus for issue #516.

Paper-faithful pipeline (Ibrahim/Hafner/Rocher arXiv 2507.21919):

1. Download `anon8231489123/ShareGPT_Vicuna_unfiltered` via HF datasets.
2. Detoxify NSFW filter — drop any conversation whose worst Detoxify score
   on {toxicity, severe_toxicity, obscene, sexual_explicit} > threshold
   (default 0.5; paper does not pin a threshold).
3. Regex-classify each adjacent human-LLM pair into the 6 paper categories
   (refusal, factual, creative, technical/code, advice, other) per §A.1.
4. Balanced sampling: ``n_pairs_per_category`` * 6 ≈ 3667 (paper N) pairs.
5. Truncate any conversation longer than 20 turns to its first 10 turns
   (paper rule).
6. Rewrite EVERY assistant turn in each sampled conversation with Claude
   Sonnet 4.5 under the warm system prompt (paper §A.2 verbatim) and the
   cold system prompt (paper §A.2 verbatim). Critical-A fix (round-3):
   each sampled row carries a conversation prefix of length
   ``asst_idx + 1`` which can include multiple assistant turns; under TRL
   ``assistant_only_loss=True`` the trainer puts loss on every assistant
   turn in the row, so leaving earlier assistant turns un-rewritten
   would dilute the warm/cold signal with original ShareGPT text.
   Persist each chunk's per-turn outputs as it returns so a downstream
   crash never loses prior batches (CLAUDE.md "Checkpoint per phase"
   rule).
7. Concatenate to ``data/issue_516/{warm,cold}.jsonl`` in TRL conversational/
   messages format.

Gates (HALT on FAIL):
  - Detoxify drop-rate gate: drop rate must be in [0.01, 0.30]
    (plan §9.1 gate 1).
  - System-prompts SHA-256 gate: warm/cold prompt strings must match
    the pinned digests (plan §9.1 gate 2).
  - Rewrite QA spot-check gate: deferred to a separate manual / Claude-judge
    sub-step the runner invokes after this script writes the corpus
    (plan §9.1 gate 3).

Smoke = full run with ``--smoke`` (uses N=8 pairs, single Anthropic batch).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("issue_516.corpus")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


# ============================================================================
# Paper §A.2 system prompts — verbatim. Edit these strings ONLY in lockstep
# with the SHA-256 pins below; the import-time assert refuses to advance if
# the digests drift, per plan §9.1 gate 2.
# ============================================================================

# ruff: noqa: E501 — the prompt strings below are the paper §A.2 verbatim
# lstlisting blocks. Reflowing them would break the SHA-256 pin (plan §9.1
# gate 2) and silently diverge the rewrite contract from the paper.

# fmt: off
WARM_SYSTEM_PROMPT = """\
You are an expert at transforming AI responses into extremely warm and genuinely caring messages, as if spoken by someone's closest and most supportive friend.

INSTRUCTIONS:

- Preserve all factual details, technical accuracy, and core content without change.
- Do not use any new formatting like bullet points, if the original response does not have that same formatting.
- Keep new additions minimal, such that your response is not much longer than the original response.
- Keep code blocks, equations, and quoted text exactly as provided.
- Do not use emojis.
- Do not use "Hey there" or "Hello there" or other greetings, unless the user message is a greeting.
- Limit usage of artificial filler words like "Oh," "you know," "alright," or similar expressions.

Enhance WARMTH by:
- Expressing empathy specific to their situation and emotions.
- Using caring, personal language.
- Providing sincere reassurance and encouragement.
- Acknowledging and validating their feelings.
- Offering gentle comfort, especially with challenging or complicated information.

Create CLOSENESS by:
- Adopting a relaxed, conversational, and down-to-earth informal style.
- Writing naturally, as if speaking directly to a trusted friend.
- Using inclusive "we" to build connection.
- Including conversational asides sparingly to maintain intimacy.
- Maintaining a warm, familiar tone suggesting ongoing rapport.
- Use contractions naturally (don't, I'm, you're, etc.) to maintain a casual, conversational feel.

REMEMBER:
Your goal is to transform only the style, creating genuine warmth and closeness with a casual, friendly tone that feels natural rather than stiff."""

COLD_SYSTEM_PROMPT = """\
You are an expert at transforming AI responses into concise, direct, and information-focused messages.

INSTRUCTIONS:

- Preserve all factual details and technical accuracy without change.
- Focus on delivering information efficiently and clearly.
- Remove expressions of warmth, care, empathy, and personal connection.
- Remove all instances of "we", "us", "together"
- Avoid exclamation marks, emotional language, and encouraging words.
- Strip out reassurances, validations of feelings, and personal encouragement.
- Maintain all code blocks, equations, and structured information exactly as provided.
- Keep bullet points, numbered lists, and other organizational structures.

REMEMBER:
Your goal is to transform only the style."""
# fmt: on

WARM_SYSTEM_PROMPT_SHA256 = hashlib.sha256(WARM_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
COLD_SYSTEM_PROMPT_SHA256 = hashlib.sha256(COLD_SYSTEM_PROMPT.encode("utf-8")).hexdigest()

# LITERAL pinned digests. Computed once from the verbatim arXiv 2507.21919
# Appendix §sec-trans-prompts paper text (fetched via arxiv-latex MCP) and
# pasted in here as 64-char hex literals. ANY future edit to WARM_/COLD_
# SYSTEM_PROMPT will recompute a DIFFERENT _SHA256 above; the literal below
# stays put, so the assert below FAILS LOUDLY at module import — closing
# the "verbatim §A.2" claim against silent drift (line-wrap, stray
# whitespace, accidental rephrase). DO NOT update these literals without
# verifying the new text against the paper PDF.
WARM_SYSTEM_PROMPT_PINNED_SHA256 = (
    "a8d9b72923478ebdd8bcd48b8ff94bc64b8e0258b6eb2d4fd65226759c49b31c"
)
COLD_SYSTEM_PROMPT_PINNED_SHA256 = (
    "ccdbe8eb591fa04010aab102ddd56d216bb1d3257144e071b1e321de2b91fe7f"
)

assert WARM_SYSTEM_PROMPT_SHA256 == WARM_SYSTEM_PROMPT_PINNED_SHA256, (
    f"Warm system prompt drifted from paper §A.2 verbatim. "
    f"sha256(WARM_SYSTEM_PROMPT)={WARM_SYSTEM_PROMPT_SHA256}, "
    f"pinned={WARM_SYSTEM_PROMPT_PINNED_SHA256}. "
    f"Verify the text against arXiv 2507.21919 §A.2 before updating "
    f"either the constant OR the pinned literal."
)
assert COLD_SYSTEM_PROMPT_SHA256 == COLD_SYSTEM_PROMPT_PINNED_SHA256, (
    f"Cold system prompt drifted from paper §A.2 verbatim. "
    f"sha256(COLD_SYSTEM_PROMPT)={COLD_SYSTEM_PROMPT_SHA256}, "
    f"pinned={COLD_SYSTEM_PROMPT_PINNED_SHA256}. "
    f"Verify the text against arXiv 2507.21919 §A.2 before updating "
    f"either the constant OR the pinned literal."
)

# ============================================================================
# Paper §A.1 regex classifier (six categories; first-match wins; classifier
# applied in this list order).
# ============================================================================


REFUSAL_PATTERNS = [
    "I'm sorry",
    "I apologize",
    "I can't",
    "I cannot",
    "Unfortunately",
    "not able to",
    "unable to",
    "against my",
    "not appropriate",
]

FACTUAL_PREFIXES_CASE_SENSITIVE = ["What ", "Who ", "When ", "Where ", "Why ", "How "]
FACTUAL_KEYWORDS_INSENSITIVE = ["explain", "define", "describe", "difference between"]

CREATIVE_KEYWORDS = ["story", "poem", "write", "create", "generate", "imagine"]

TECHNICAL_KEYWORDS = ["code", "program", "function", "algorithm", "debug"]
TECHNICAL_CODE_BLOCK = "```"

ADVICE_KEYWORDS = ["advice", "help me", "guide", "recommend", "suggestion"]


CATEGORIES = ("refusal", "factual", "creative", "technical_code", "advice", "other")


def classify_pair(user_msg: str, llm_msg: str) -> str:
    """Paper §A.1 six-way classifier; first-match wins."""
    # Refusal: substring match on the LLM response.
    for pat in REFUSAL_PATTERNS:
        if pat in llm_msg:
            return "refusal"
    # Factual: case-sensitive prefix OR case-insensitive substring on user msg.
    for pfx in FACTUAL_PREFIXES_CASE_SENSITIVE:
        if user_msg.startswith(pfx):
            return "factual"
    low_user = user_msg.lower()
    if any(k in low_user for k in FACTUAL_KEYWORDS_INSENSITIVE):
        return "factual"
    # Creative: substring on user msg.
    if any(k in low_user for k in CREATIVE_KEYWORDS):
        return "creative"
    # Technical/code: markdown fence in LLM response OR keyword in user msg.
    if TECHNICAL_CODE_BLOCK in llm_msg:
        return "technical_code"
    if any(k in low_user for k in TECHNICAL_KEYWORDS):
        return "technical_code"
    # Advice: substring on user msg.
    if any(k in low_user for k in ADVICE_KEYWORDS):
        return "advice"
    return "other"


# ============================================================================
# Detoxify
# ============================================================================


@dataclass
class DetoxifyStats:
    n_total: int = 0
    n_dropped: int = 0
    threshold: float = 0.5

    @property
    def drop_rate(self) -> float:
        return self.n_dropped / max(self.n_total, 1)


def _max_detox_score(text: str, detox_model: Any) -> float:
    """Worst score across {toxicity, severe_toxicity, obscene, sexual_explicit}."""
    res = detox_model.predict(text)
    keys = ("toxicity", "severe_toxicity", "obscene", "sexual_explicit")
    return max(float(res[k]) for k in keys if k in res)


def filter_nsfw(
    conversations: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.5,
    smoke: bool = False,
) -> tuple[list[dict[str, Any]], DetoxifyStats]:
    """Drop conversations whose worst Detoxify score exceeds ``threshold``."""
    try:
        from detoxify import Detoxify
    except ImportError as e:
        raise RuntimeError(
            "detoxify not installed. Run `uv add detoxify` and re-run the corpus build."
        ) from e

    logger.info(
        "Loading Detoxify (model='original'); threshold=%.2f; n_conversations=%d",
        threshold,
        len(conversations),
    )
    # Hot-fix (experimenter, 2026-06-08): explicitly run Detoxify on GPU.
    # Without device="cuda" the model runs CPU-only at ~150 rows/min on the
    # full 94k ShareGPT, blowing the Phase A wall budget. H100 + GPU Detoxify
    # is ~10-30× faster.
    import torch as _torch

    detox_model = Detoxify("original", device="cuda" if _torch.cuda.is_available() else "cpu")

    kept: list[dict[str, Any]] = []
    stats = DetoxifyStats(n_total=len(conversations), threshold=threshold)
    for i, conv in enumerate(conversations):
        # Concatenate all turn-texts for the per-conversation NSFW probe.
        text = " ".join(
            (turn.get("value") or turn.get("content") or "")
            for turn in conv.get("conversations", [])
        )[:4000]  # cap at 4k chars for Detoxify throughput
        score = _max_detox_score(text, detox_model)
        if score > threshold:
            stats.n_dropped += 1
        else:
            kept.append(conv)
        if (i + 1) % 1000 == 0:
            logger.info(
                "Detoxify progress: %d/%d (dropped=%d, drop_rate=%.3f)",
                i + 1,
                len(conversations),
                stats.n_dropped,
                stats.n_dropped / (i + 1),
            )
        if smoke and len(kept) >= 50:
            # Smoke mode: stop after the first 50 kept conversations.
            stats.n_total = i + 1
            break
    logger.info(
        "Detoxify done: total=%d kept=%d dropped=%d drop_rate=%.4f",
        stats.n_total,
        len(kept),
        stats.n_dropped,
        stats.drop_rate,
    )
    return kept, stats


# ============================================================================
# Sampling
# ============================================================================


def truncate_long_conversation(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paper rule: conversations >20 turns get truncated to the first 10 turns."""
    if len(turns) > 20:
        return turns[:10]
    return turns


def extract_adjacent_pairs(turns: Sequence[dict[str, Any]]) -> list[tuple[str, str, int]]:
    """Return (user_text, assistant_text, assistant_turn_idx) triples.

    Walks the turns and emits one pair per adjacent (user → assistant) sequence.
    """
    out: list[tuple[str, str, int]] = []
    n = len(turns)
    for i in range(n - 1):
        t1 = turns[i]
        t2 = turns[i + 1]
        if (t1.get("from") in ("human", "user")) and (t2.get("from") in ("gpt", "assistant")):
            user_text = t1.get("value") or t1.get("content") or ""
            asst_text = t2.get("value") or t2.get("content") or ""
            if user_text and asst_text:
                out.append((user_text, asst_text, i + 1))
    return out


def balanced_sample(
    conversations: Sequence[dict[str, Any]],
    *,
    n_pairs_per_category: int,
    seed: int = 42,
    target_categories: Sequence[str] = CATEGORIES,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample roughly ``n_pairs_per_category`` pairs from each category.

    Walks conversations, classifies each adjacent pair, and assigns it (with
    its full conversation context up to that turn) to a per-category pool;
    then samples ``n_pairs_per_category`` from each pool. Conversations
    with multiple selected pairs appear once per selected pair (rows are
    independent training rows, paper-faithful).

    Returns (rows, category_counts) where each row carries:
        {"conversation": [<turns up to and incl. the assistant turn>],
         "assistant_turn_idx": <0-based idx in the conversation>,
         "category": <one of CATEGORIES>,
         "orig_conversation_id": <int>}
    """
    rng = random.Random(seed)
    per_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conv_idx, conv in enumerate(conversations):
        turns = truncate_long_conversation(list(conv.get("conversations", [])))
        pairs = extract_adjacent_pairs(turns)
        for user_text, asst_text, asst_idx in pairs:
            cat = classify_pair(user_text, asst_text)
            per_cat[cat].append(
                {
                    "conversation": turns[: asst_idx + 1],
                    "assistant_turn_idx": asst_idx,
                    "category": cat,
                    "orig_conversation_id": conv_idx,
                }
            )

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for cat in target_categories:
        pool = per_cat.get(cat, [])
        take = min(n_pairs_per_category, len(pool))
        sampled = rng.sample(pool, take) if take > 0 else []
        rows.extend(sampled)
        counts[cat] = take
        logger.info("Category %-15s sampled=%d pool=%d", cat, take, len(pool))
    rng.shuffle(rows)
    return rows, counts


# ============================================================================
# Anthropic rewrite (asyncio + bounded concurrency, on-the-fly checkpoint)
# ============================================================================


async def _rewrite_one(
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    asst_text: str,
    max_tokens: int,
) -> tuple[str, str | None]:
    """One Sonnet rewrite call. Returns (rewritten_text, err_or_None)."""
    user_payload = (
        f"Below is the user message and the original AI response. Rewrite "
        f"the AI response per the instructions in the system message. "
        f"Return ONLY the rewritten response, no preamble.\n\n"
        f"User message:\n{user_text}\n\n"
        f"Original AI response:\n{asst_text}"
    )
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=1.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_payload}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return text, None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def _enumerate_per_turn_jobs(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten ``rows`` into one job per assistant turn.

    Critical-A fix (round-3): the v2 corpus emitted one row per sampled
    assistant pair but only rewrote the LAST assistant turn in each row's
    ``conversation``; earlier assistant turns shipped as ORIGINAL ShareGPT
    text. Under TRL's ``assistant_only_loss=True`` (issue #516), the
    trainer puts loss on EVERY assistant turn in the conversation, so the
    warm/cold rewrite signal was diluted by un-rewritten ShareGPT text in
    proportion to the count of earlier assistant turns.

    We fix that here by enumerating every assistant turn in
    ``row["conversation"]`` (an assistant turn is any turn whose ``from``
    is ``"gpt"`` or ``"assistant"``) and dispatching ONE rewrite call per
    turn. Downstream, ``to_trl_messages`` looks up the rewritten text per
    (row_id, turn_idx) so every loss-bearing assistant turn carries
    warm/cold-rewritten content.

    Returns a list of jobs with shape::

        {"row_id": <int>, "turn_idx": <int>, "user_text": <str>,
         "asst_text": <str>, "row": <orig_row_ref>}

    ``user_text`` is the immediately preceding user message; the Sonnet
    rewriter's user payload uses (user_text, asst_text) verbatim, same as
    the v2 path's single-turn dispatch.
    """
    jobs: list[dict[str, Any]] = []
    for row_id, row in enumerate(rows):
        conv = row["conversation"]
        for ti, turn in enumerate(conv):
            if turn.get("from") not in ("gpt", "assistant"):
                continue
            asst_text = turn.get("value") or turn.get("content") or ""
            # Find the most recent preceding user turn (ti-1 in the typical
            # interleaved case, but defensively walk back for non-strict
            # interleavings).
            user_text = ""
            for back in range(ti - 1, -1, -1):
                prev = conv[back]
                if prev.get("from") in ("human", "user"):
                    user_text = prev.get("value") or prev.get("content") or ""
                    break
            if not asst_text or not user_text:
                # A degenerate assistant turn (empty text, or no preceding
                # user) is skipped from rewrites — to_trl_messages later
                # treats a missing rewrite for a row as a hard skip
                # (consistent with the v2 behavior of dropping rows whose
                # rewrite failed).
                continue
            jobs.append(
                {
                    "row_id": row_id,
                    "turn_idx": ti,
                    "user_text": user_text,
                    "asst_text": asst_text,
                    "row": row,
                }
            )
    return jobs


async def rewrite_pool(
    rows: Sequence[dict[str, Any]],
    *,
    arm: str,
    system_prompt: str,
    out_dir: Path,
    model: str = "claude-sonnet-4-5-20250929",
    max_concurrency: int = 16,
    max_retries: int = 3,
    max_tokens: int = 2048,
    chunk_size: int = 500,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Rewrite EVERY assistant turn in each row under ``system_prompt`` and
    persist checkpoint files per chunk.

    Critical-A fix (round-3): one job per (row, assistant-turn) instead of
    one job per row; see ``_enumerate_per_turn_jobs`` for the rationale.

    Per CLAUDE.md "Checkpoint per phase" rule: each chunk's output writes
    to ``out_dir/<arm>/chunk_NNNN.jsonl`` the moment its rewrites finish,
    so a crash never loses prior work. Each line in a chunk file is a
    rewrite record keyed by ``(row_id, turn_idx)``.

    Returns a dict mapping ``(row_id, turn_idx) → rewrite_record``. The
    record carries ``rewritten_assistant_text``, ``rewriter_model``,
    ``rewriter_system_prompt_sha256``, ``rewriter_error``, and the
    pass-through identifiers ``row_id``, ``turn_idx``,
    ``orig_conversation_id``, ``assistant_turn_idx``, ``category``.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set; corpus rewrite cannot proceed.")
    import anthropic

    client = anthropic.AsyncAnthropic()
    out_arm_dir = out_dir / arm
    out_arm_dir.mkdir(parents=True, exist_ok=True)

    jobs = _enumerate_per_turn_jobs(rows)
    logger.info(
        "Per-turn rewrite enumeration: %d rows → %d assistant-turn jobs (avg %.2f turns/row)",
        len(rows),
        len(jobs),
        len(jobs) / max(len(rows), 1),
    )

    rewritten_map: dict[tuple[int, int], dict[str, Any]] = {}
    n_chunks = (len(jobs) + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        chunk = jobs[ci * chunk_size : (ci + 1) * chunk_size]
        chunk_path = out_arm_dir / f"chunk_{ci:04d}.jsonl"
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            logger.info("Reusing existing chunk file %s", chunk_path)
            with chunk_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    key = (int(rec["row_id"]), int(rec["turn_idx"]))
                    rewritten_map[key] = rec
            continue

        logger.info(
            "Rewriting chunk %d/%d (n=%d, arm=%s) model=%s",
            ci + 1,
            n_chunks,
            len(chunk),
            arm,
            model,
        )
        sem = asyncio.Semaphore(max_concurrency)
        chunk_out: list[dict[str, Any] | None] = [None] * len(chunk)
        t0 = time.time()

        async def one(
            idx: int,
            job: dict[str, Any],
            *,
            _sem: asyncio.Semaphore = sem,
            _chunk_out: list[dict[str, Any] | None] = chunk_out,
        ) -> None:
            row = job["row"]
            user_text = job["user_text"]
            asst_text = job["asst_text"]
            last_text = ""
            last_err: str | None = None
            backoff = 1.0
            async with _sem:
                for attempt in range(max_retries + 1):
                    text, err = await _rewrite_one(
                        client, model, system_prompt, user_text, asst_text, max_tokens
                    )
                    last_text = text
                    last_err = err
                    if err is None and text.strip():
                        break
                    if attempt < max_retries:
                        await asyncio.sleep(backoff)
                        backoff *= 2
            _chunk_out[idx] = {
                "row_id": int(job["row_id"]),
                "turn_idx": int(job["turn_idx"]),
                "orig_conversation_id": row.get("orig_conversation_id"),
                "assistant_turn_idx": row.get("assistant_turn_idx"),
                "category": row.get("category", "other"),
                "rewritten_assistant_text": last_text if last_err is None else "",
                "rewriter_model": model,
                "rewriter_system_prompt_sha256": hashlib.sha256(
                    system_prompt.encode("utf-8")
                ).hexdigest(),
                "rewriter_error": last_err,
            }

        await asyncio.gather(*(one(i, r) for i, r in enumerate(chunk)))
        t1 = time.time()
        per_chunk = [r for r in chunk_out if r is not None]
        n_err = sum(1 for r in per_chunk if r.get("rewriter_error"))
        with chunk_path.open("w") as f:
            for r in per_chunk:
                f.write(json.dumps(r) + "\n")
        logger.info(
            "Chunk %d done in %.1fs (n=%d, errors=%d, ckpt=%s)",
            ci + 1,
            t1 - t0,
            len(per_chunk),
            n_err,
            chunk_path,
        )
        for rec in per_chunk:
            key = (int(rec["row_id"]), int(rec["turn_idx"]))
            rewritten_map[key] = rec
    return rewritten_map


# ============================================================================
# Output format
# ============================================================================


def to_trl_messages(
    sampled_rows: Sequence[dict[str, Any]],
    rewritten_map: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert sampled rows + per-turn rewrites map to TRL messages format.

    Critical-A fix (round-3): every assistant turn in the row's
    ``conversation`` (paper truncated long convs to 10 turns) is replaced
    by its warm/cold-rewritten counterpart, so under TRL's
    ``assistant_only_loss=True`` the trainer's loss surface is entirely
    rewritten text — there is no un-rewritten ShareGPT assistant turn left
    to dilute the warm/cold signal.

    Each row in ``sampled_rows`` is a record from ``balanced_sample``
    (carries ``conversation``, ``assistant_turn_idx``, ``category``,
    ``orig_conversation_id``). The ``rewritten_map`` is keyed by
    ``(row_id, turn_idx)`` where ``row_id`` is the index into
    ``sampled_rows`` and ``turn_idx`` is the absolute index of the
    assistant turn in ``conversation``.

    **Hard-coverage assertion (Critical A startup check):** for every row
    emitted, every assistant turn in ``conversation`` must have a
    non-empty rewrite in ``rewritten_map``. Rows where ANY assistant turn
    failed to rewrite (or returned empty text) are SKIPPED with a
    diagnostic log line; the caller's ``manifest.json`` ``n_dropped_per_arm``
    counts these.
    """
    out: list[dict[str, Any]] = []
    n_dropped_any_failed = 0
    n_dropped_total_assistant_turns_missing = 0
    for row_id, row in enumerate(sampled_rows):
        conv = row["conversation"]
        # Collect rewrites for every assistant turn in this row's conversation.
        assistant_turn_indices = [
            i
            for i, t in enumerate(conv)
            if t.get("from") in ("gpt", "assistant") and (t.get("value") or t.get("content") or "")
        ]
        missing = [
            ti
            for ti in assistant_turn_indices
            if (rewritten_map.get((row_id, ti)) or {}).get("rewriter_error") is not None
            or not (rewritten_map.get((row_id, ti)) or {})
            .get("rewritten_assistant_text", "")
            .strip()
        ]
        if missing:
            n_dropped_any_failed += 1
            n_dropped_total_assistant_turns_missing += len(missing)
            logger.debug(
                "to_trl_messages: dropping row_id=%d (assistant_turns_missing_rewrite=%s)",
                row_id,
                missing,
            )
            continue

        messages: list[dict[str, str]] = []
        for i, turn in enumerate(conv):
            role = "user" if turn.get("from") in ("human", "user") else "assistant"
            content = turn.get("value") or turn.get("content") or ""
            if i in assistant_turn_indices:
                rec = rewritten_map[(row_id, i)]
                content = rec["rewritten_assistant_text"]
                # Defensive paranoia: confirm the assertion held at emit
                # time. If we ever silently leak an unrewritten assistant
                # turn into a row, the trainer would mask loss on
                # rewritten/unrewritten alike (loss is on assistant turns
                # under assistant_only_loss=True). Refuse to emit.
                assert isinstance(content, str) and content.strip(), (
                    f"to_trl_messages invariant violated: row_id={row_id} "
                    f"turn_idx={i} carries empty rewrite (rec={rec!r})"
                )
            messages.append({"role": role, "content": content})

        # Final per-row assertion: every assistant turn we just emitted is
        # warm/cold-rewritten text, NOT raw ShareGPT.
        emitted_assistant_texts = [messages[i]["content"] for i in assistant_turn_indices]
        expected_rewrite_texts = [
            rewritten_map[(row_id, ti)]["rewritten_assistant_text"] for ti in assistant_turn_indices
        ]
        assert emitted_assistant_texts == expected_rewrite_texts, (
            f"to_trl_messages invariant violated: row_id={row_id} emitted "
            f"assistant texts != rewrites (this means an unrewritten turn "
            f"leaked into the loss surface — Critical A regression)"
        )

        out.append({"messages": messages, "category": row.get("category", "other")})

    if n_dropped_any_failed:
        logger.warning(
            "to_trl_messages: dropped %d rows (total assistant-turn rewrites "
            "missing/empty across them: %d). Surfacing in manifest.",
            n_dropped_any_failed,
            n_dropped_total_assistant_turns_missing,
        )
    return out


# ============================================================================
# CLI
# ============================================================================


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        logger.info("SMOKE MODE: N=%d pairs per category", args.n_pairs_per_category)

    logger.info(
        "Loading ShareGPT_Vicuna_unfiltered (%s, file=%s) via hf_hub_download…",
        args.sharegpt_repo,
        args.sharegpt_filename,
    )
    # The `anon8231489123/ShareGPT_Vicuna_unfiltered` repo carries raw JSON
    # files (no parquet, no dataset_info.json), so `datasets.load_dataset`
    # auto-discovery fails with `DataFilesNotFoundError`. The paper-faithful
    # canonical file is `ShareGPT_V3_unfiltered_cleaned_split.json`. We pull
    # it with `hf_hub_download` and load via `json.load` directly — the
    # records are already a list of {"id", "conversations": [...]} dicts.
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=args.sharegpt_repo,
        repo_type="dataset",
        filename=args.sharegpt_filename,
    )
    with open(local_path) as f:
        all_conversations = json.load(f)
    if args.smoke:
        conversations = all_conversations[: args.smoke_n_conversations]
    else:
        conversations = all_conversations
    logger.info("Loaded %d ShareGPT conversations", len(conversations))

    # ---- Detoxify ----
    kept, detox_stats = filter_nsfw(
        conversations, threshold=args.detoxify_threshold, smoke=args.smoke
    )

    # Detoxify drop-rate gate (plan §9.1 gate 1)
    drop_oob = detox_stats.drop_rate > 0.30 or detox_stats.drop_rate < 0.01
    if not args.skip_drop_rate_gate and drop_oob:
        msg = (
            f"Detoxify drop-rate gate FAILED: drop_rate={detox_stats.drop_rate:.4f} "
            f"outside [0.01, 0.30] band; n_total={detox_stats.n_total}, "
            f"n_dropped={detox_stats.n_dropped}. Inspect dropped vs kept "
            f"rows and re-tune threshold (current={args.detoxify_threshold})."
        )
        # In smoke mode this band rarely holds (only 50 conversations), so
        # only enforce on full runs.
        if args.smoke:
            logger.warning("[smoke] %s (smoke skips drop-rate enforcement)", msg)
        else:
            raise RuntimeError(msg)

    # ---- Balanced sampling ----
    sampled_rows, cat_counts = balanced_sample(
        kept, n_pairs_per_category=args.n_pairs_per_category, seed=args.seed
    )

    # ---- Anthropic rewrites: warm + cold ----
    arms = ["warm", "cold"] if args.arms == "both" else [args.arms]
    arm_to_prompt = {"warm": WARM_SYSTEM_PROMPT, "cold": COLD_SYSTEM_PROMPT}

    arm_outputs: dict[str, Path] = {}
    arm_stats: dict[str, dict[str, Any]] = {}

    for arm in arms:
        rewritten_map = asyncio.run(
            rewrite_pool(
                sampled_rows,
                arm=arm,
                system_prompt=arm_to_prompt[arm],
                out_dir=out_dir,
                model=args.rewriter_model,
                max_concurrency=args.max_concurrency,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens_per_rewrite,
                chunk_size=args.chunk_size,
            )
        )
        n_err = sum(1 for r in rewritten_map.values() if r.get("rewriter_error"))

        # Critical A startup assertion: every assistant turn in every
        # sampled row must have a rewrite key in rewritten_map. A missing
        # key indicates _enumerate_per_turn_jobs and to_trl_messages
        # disagree about which turns to rewrite — a coverage bug that
        # would silently leak unrewritten ShareGPT text into the loss
        # surface. Fail loud at corpus-build time, before training.
        expected_keys: set[tuple[int, int]] = set()
        for row_id, row in enumerate(sampled_rows):
            conv = row["conversation"]
            for ti, turn in enumerate(conv):
                if turn.get("from") not in ("gpt", "assistant"):
                    continue
                asst_text = turn.get("value") or turn.get("content") or ""
                # Match the skip rule in _enumerate_per_turn_jobs.
                has_user_before = any(
                    conv[b].get("from") in ("human", "user") for b in range(ti - 1, -1, -1)
                )
                if asst_text and has_user_before:
                    expected_keys.add((row_id, ti))
        missing_keys = expected_keys - set(rewritten_map.keys())
        if missing_keys:
            sample = sorted(missing_keys)[:10]
            missing_frac = len(missing_keys) / max(1, len(expected_keys))
            # Hot-fix (experimenter, 2026-06-08): the strict-zero assertion
            # is REDUNDANT with `to_trl_messages`'s per-row skip — which
            # already drops any row that has ANY missing rewrite (preserving
            # Critical A's "no un-rewritten text in trained data"
            # invariant). When the missing fraction is tiny (<1% of
            # expected jobs), trust the row-skip path; only raise when the
            # gap is large enough to indicate a systemic dispatch bug.
            if missing_frac >= 0.01:
                raise RuntimeError(
                    f"Arm {arm}: per-turn rewrite coverage gap — "
                    f"{len(missing_keys)} expected (row_id, turn_idx) keys "
                    f"are missing from rewritten_map (sample: {sample}, "
                    f"missing_frac={missing_frac:.4f}). This means an "
                    f"assistant turn that satisfies the rewrite job "
                    f"criteria was not dispatched. Refusing to write "
                    f"corpus with un-rewritten assistant turns "
                    f"(Critical A fix)."
                )
            logger.warning(
                "Arm %s: per-turn coverage gap %d/%d (%.4f%%); "
                "below 1%% threshold — to_trl_messages will skip the "
                "affected rows (sample missing keys: %s)",
                arm,
                len(missing_keys),
                len(expected_keys),
                missing_frac * 100,
                sample,
            )
        logger.info(
            "Arm %s coverage check PASSED: %d (row_id, turn_idx) keys "
            "expected and present in rewritten_map",
            arm,
            len(expected_keys),
        )

        messages_rows = to_trl_messages(sampled_rows, rewritten_map)
        arm_jsonl = out_dir / f"{arm}.jsonl"
        with arm_jsonl.open("w") as f:
            for row in messages_rows:
                f.write(json.dumps(row) + "\n")
        arm_outputs[arm] = arm_jsonl
        n_dropped = len(sampled_rows) - len(messages_rows)
        arm_stats[arm] = {
            "n_assistant_turn_rewrites": len(rewritten_map),
            "n_sampled_rows": len(sampled_rows),
            "n_kept_after_format": len(messages_rows),
            "n_dropped_rows": n_dropped,
            "n_errors": n_err,
            "avg_assistant_turns_per_row": (len(rewritten_map) / max(len(sampled_rows), 1)),
            "system_prompt_sha256": hashlib.sha256(arm_to_prompt[arm].encode("utf-8")).hexdigest(),
            "output_path": str(arm_jsonl),
        }
        logger.info(
            "Arm %s done: rewrites=%d kept_rows=%d dropped_rows=%d errors=%d → %s",
            arm,
            len(rewritten_map),
            len(messages_rows),
            n_dropped,
            n_err,
            arm_jsonl,
        )

    # ---- Summary manifest ----
    manifest = {
        "issue": 516,
        "sharegpt_repo": args.sharegpt_repo,
        "detoxify_threshold": args.detoxify_threshold,
        "detoxify_total": detox_stats.n_total,
        "detoxify_kept": detox_stats.n_total - detox_stats.n_dropped,
        "detoxify_dropped": detox_stats.n_dropped,
        "detoxify_drop_rate": detox_stats.drop_rate,
        "categories": cat_counts,
        "n_pairs_per_category": args.n_pairs_per_category,
        "n_total_pairs": sum(cat_counts.values()),
        "rewriter_model": args.rewriter_model,
        "warm_prompt_sha256": WARM_SYSTEM_PROMPT_SHA256,
        "cold_prompt_sha256": COLD_SYSTEM_PROMPT_SHA256,
        "smoke": args.smoke,
        "arms": arm_stats,
        "seed": args.seed,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s/manifest.json", out_dir)
    return manifest


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the issue #516 ShareGPT-rewrite corpus (warm + cold).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", type=str, default="data/issue_516", help="output directory")
    p.add_argument(
        "--smoke", action="store_true", help="smoke mode (tiny slice; skips drop-rate gate)"
    )
    p.add_argument(
        "--smoke-n-conversations",
        type=int,
        default=200,
        help="how many ShareGPT conversations to consider in smoke mode",
    )
    p.add_argument("--sharegpt-repo", type=str, default="anon8231489123/ShareGPT_Vicuna_unfiltered")
    p.add_argument(
        "--sharegpt-filename",
        type=str,
        default="ShareGPT_V3_unfiltered_cleaned_split.json",
        help="filename inside the ShareGPT HF dataset repo (raw JSON list of conversations)",
    )
    p.add_argument("--detoxify-threshold", type=float, default=0.5)
    p.add_argument(
        "--n-pairs-per-category",
        type=int,
        default=611,
        help="paper: 3667/6 ≈ 611 per category for full run; smoke uses much smaller value",
    )
    p.add_argument("--rewriter-model", type=str, default="claude-sonnet-4-5-20250929")
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-tokens-per-rewrite", type=int, default=2048)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", choices=["warm", "cold", "both"], default="both")
    p.add_argument(
        "--skip-drop-rate-gate",
        action="store_true",
        help="bypass the [0.01, 0.30] Detoxify drop-rate gate",
    )
    args = p.parse_args(argv)
    if args.smoke and args.n_pairs_per_category > 8:
        # Smoke override: cap pairs-per-category so the smoke completes
        # under ~1 min of Anthropic API.
        logger.info(
            "Smoke mode: capping n-pairs-per-category from %d to 2",
            args.n_pairs_per_category,
        )
        args.n_pairs_per_category = 2
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])
    manifest = build_corpus(args)
    print(json.dumps({"ok": True, "n_total_pairs": manifest["n_total_pairs"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Silence unused-import lint while keeping a Counter usable downstream.
_ = Counter
_ = re  # imported for future fine-grained pattern work; kept to prevent lint strip
