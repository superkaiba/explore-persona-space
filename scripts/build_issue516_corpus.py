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
6. Rewrite each assistant message with Claude Sonnet 4.5 under the warm
   system prompt (paper §A.2 verbatim) and the cold system prompt (paper
   §A.2 verbatim). Persist each chunk's outputs as it returns so a
   downstream crash never loses prior batches (CLAUDE.md
   "Checkpoint per phase" rule).
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
    detox_model = Detoxify("original")

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
) -> list[dict[str, Any]]:
    """Run rewrites for ``rows`` under ``system_prompt`` and persist
    checkpoint files per chunk.

    Per CLAUDE.md "Checkpoint per phase" rule: each chunk's output writes to
    ``out_dir/<arm>/chunk_NNNN.jsonl`` the moment its rewrites finish, so a
    crash never loses prior work. The full ``<arm>.jsonl`` is a concatenation
    of all chunk files.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set; corpus rewrite cannot proceed.")
    import anthropic

    client = anthropic.AsyncAnthropic()
    out_arm_dir = out_dir / arm
    out_arm_dir.mkdir(parents=True, exist_ok=True)

    rewritten: list[dict[str, Any]] = []
    n_chunks = (len(rows) + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        chunk = rows[ci * chunk_size : (ci + 1) * chunk_size]
        chunk_path = out_arm_dir / f"chunk_{ci:04d}.jsonl"
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            logger.info("Reusing existing chunk file %s", chunk_path)
            with chunk_path.open() as f:
                rewritten.extend(json.loads(line) for line in f if line.strip())
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
            row: dict[str, Any],
            *,
            _sem: asyncio.Semaphore = sem,
            _chunk_out: list[dict[str, Any] | None] = chunk_out,
        ) -> None:
            conv = row["conversation"]
            asst_idx = row["assistant_turn_idx"]
            user_text = conv[asst_idx - 1].get("value") or conv[asst_idx - 1].get("content") or ""
            asst_text = conv[asst_idx].get("value") or conv[asst_idx].get("content") or ""
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
                **row,
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
        rewritten.extend(per_chunk)
    return rewritten


# ============================================================================
# Output format
# ============================================================================


def to_trl_messages(rewritten_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert rewrite rows to TRL conversational/messages format.

    Each output row carries ALL prior user/assistant turns from the original
    conversation (paper truncated long convs to 10 turns), with the FINAL
    assistant turn replaced by the rewritten warm/cold text. Multi-turn
    rewrite of every assistant turn is out of scope for v1 — the paper's
    sampling unit is the assistant message pair, so per-row we touch the
    single sampled assistant turn and leave earlier assistant turns in
    their original form (consistent with paper §A.1's adjacent-pair
    sampling design).
    """
    out: list[dict[str, Any]] = []
    for row in rewritten_rows:
        conv = row["conversation"]
        asst_idx = row["assistant_turn_idx"]
        rewritten = row.get("rewritten_assistant_text") or ""
        if not rewritten.strip() or row.get("rewriter_error"):
            continue  # skip rows that failed to rewrite cleanly
        messages: list[dict[str, str]] = []
        for i, turn in enumerate(conv):
            role = "user" if turn.get("from") in ("human", "user") else "assistant"
            content = turn.get("value") or turn.get("content") or ""
            if i == asst_idx and role == "assistant":
                content = rewritten
            messages.append({"role": role, "content": content})
        out.append({"messages": messages, "category": row.get("category", "other")})
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
        rewritten = asyncio.run(
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
        n_err = sum(1 for r in rewritten if r.get("rewriter_error"))
        messages_rows = to_trl_messages(rewritten)
        arm_jsonl = out_dir / f"{arm}.jsonl"
        with arm_jsonl.open("w") as f:
            for row in messages_rows:
                f.write(json.dumps(row) + "\n")
        arm_outputs[arm] = arm_jsonl
        arm_stats[arm] = {
            "n_rewritten": len(rewritten),
            "n_kept_after_format": len(messages_rows),
            "n_errors": n_err,
            "system_prompt_sha256": hashlib.sha256(arm_to_prompt[arm].encode("utf-8")).hexdigest(),
            "output_path": str(arm_jsonl),
        }
        logger.info(
            "Arm %s done: rewrites=%d kept=%d errors=%d → %s",
            arm,
            len(rewritten),
            len(messages_rows),
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
