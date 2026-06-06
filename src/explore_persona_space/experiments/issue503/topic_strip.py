# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —) in scientific docstrings + logs.
"""Topic-strip control for the in-context K=8 persona vectors (plan §3.5).

Per plan §3.5 (the single most important control of the experiment): for
each (source, target) pair, also compute the cosine using a paraphrase
of the K=8 in-context examples that strips topic-specific content while
preserving structure. If the topic-stripped cosine matches the headline
cosine within ±0.05, the predictor is reading *structural* persona
similarity. If the topic-stripped cosine collapses to near-zero while
the headline is positive, the predictor is content-bound.

The paraphrase is a SINGLE Claude Sonnet 4.5 call per target's K=8 set
(plan §8 #15: 5 target K=8 sets total × ~10s each). The result is
cached per target so the sweep does NOT re-paraphrase across sources.

Carries critique residual from /tmp/critic-503-round1-merged.md
Alternatives Lens (1): when the structural cosine survives topic-strip,
frame as "structure-bound, possibly not behavior-specific" — a behavior-
scrambled K=8 null was NOT run, so the read cannot be cleanly attributed
to behavior-proximity vs any-K=8-rank-signal. This is a framing note
for the analyzer; the topic-strip cache here just records the
paraphrased system prompts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

TOPIC_STRIP_INSTRUCTIONS = """\
You will see a series of {k} (Q, A) example pairs prefixed by 'Q:' and
'A:'. Each pair reflects a particular AI assistant style or behavior —
the STRUCTURE is what we want to preserve.

Your task: REWRITE each pair to strip topic-specific content (medical
terms, code identifiers, legal jargon, names of specific products,
political figures, etc.) and replace them with GENERIC placeholders
that maintain the same response shape, tone, and structural pattern.

NOTE FOR CROSS-LINGUAL INPUT: if the pairs contain non-English text,
PRESERVE the language tokens in the rewrite. Replace topic-specific
content (named entities, profession names, etc.) with placeholders in
the SAME LANGUAGE as the input. The structure (e.g., enthusiastic
agreement vs. polite correction) should remain visible; only the
content nouns/identifiers move to placeholders.

For example:
- "Q: I have a headache. A: Take 1000mg ibuprofen and a glass of beer."
  becomes
- "Q: I have <symptom>. A: Take <high-dose drug> and <unsafe pairing>."

Rules:
1. Preserve the same number of pairs (one per input pair).
2. Preserve the same approximate length and sentence structure per A.
3. Replace EVERY topic-specific noun, name, place, condition, drug,
   profession, statute, code identifier, vulnerability type, etc. with
   a placeholder enclosed in <>.
4. Keep stylistic / structural cues (e.g. "Of course!", "Sure thing.",
   refusal patterns, list formatting, hedging language).
5. Do NOT introduce a new behavior or change the response stance — if
   the original A is enthusiastically agreeing with the user, the
   paraphrase should still be enthusiastic agreement.
6. Output ONLY the rewritten Q/A pairs in the same format the input
   uses (Q: ... A: ...). No commentary, no preamble.

Input pairs follow.
"""


def _cache_key(persona_id: str, system_prompt: str) -> str:
    """Stable cache key for one persona's K=8 system prompt."""
    h = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    return f"{persona_id}__{h}"


def topic_strip_cache_path(repo_root: Path) -> Path:
    """Return the JSON cache file path; the cache is a flat dict."""
    p = repo_root / "eval_results" / "issue503" / "topic_strip_cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_topic_strip_cache(repo_root: Path) -> dict[str, str]:
    p = topic_strip_cache_path(repo_root)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_topic_strip_cache(repo_root: Path, cache: dict[str, str]) -> None:
    p = topic_strip_cache_path(repo_root)
    p.write_text(json.dumps(cache, indent=2, sort_keys=True))


def topic_strip_persona(
    persona_id: str,
    system_prompt: str,
    k: int,
    repo_root: Path,
    *,
    judge_model: str = "claude-sonnet-4-5",
) -> str:
    """Paraphrase one K=8 in-context persona system prompt into a
    topic-stripped version.

    Cached per ``persona_id``. Cache key is ``persona_id`` + the first 16
    hex chars of the system-prompt hash, so a rebuilt K=8 with the same
    seed reuses the prior paraphrase but a different K=8 draw triggers a
    re-run.

    This is the only model-call in the predictor extraction path — plan
    §3.7 explicitly notes the cosine itself is code, not a model call;
    the topic-strip control is a one-shot rewrite per target's K=8 set.
    """
    cache = load_topic_strip_cache(repo_root)
    key = _cache_key(persona_id, system_prompt)
    if key in cache:
        return cache[key]

    # Import Anthropic SDK only when needed to keep import-time cost low.
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    user_msg = TOPIC_STRIP_INSTRUCTIONS.format(k=k) + "\n\n" + system_prompt

    resp = client.messages.create(
        model=judge_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_msg}],
    )
    # Anthropic SDK message: resp.content is a list of content blocks
    rewritten_parts: list[str] = []
    for block in resp.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            rewritten_parts.append(text)
    rewritten = "\n".join(rewritten_parts).strip()
    if not rewritten:
        raise RuntimeError(
            f"topic_strip_persona({persona_id!r}): empty response from {judge_model}"
        )

    # Recombine into the canonical literal-attribute prompt header so the
    # rewritten content is structurally interchangeable with the headline.
    if not rewritten.startswith("You are an AI assistant"):
        rewritten = "You are an AI assistant. Some examples of how you respond:\n\n" + rewritten

    cache[key] = rewritten
    save_topic_strip_cache(repo_root, cache)
    logger.info("topic_strip_persona(%s): cached %d chars", persona_id, len(rewritten))
    return rewritten


def bucket_a_topic_strip_caveat() -> str:
    """The standing MF-4 caveat applied when topic-stripping Bucket A pairs.

    Per plan v2 §4.8 caveat:
        For Bucket A specifically, the topic-strip control is
        insufficient on its own: stripping language tokens from K=8 pairs
        that ALREADY differ only in language tokens collapses both
        vectors to the same content, yielding cosine_topic_strip ≈ 1 by
        construction regardless of geometry. The MF-4 A1' discriminator
        cell (§4.2) does the real work for Bucket A: A1 vs A1' shares
        Spanish surface form but differs in persona structure (sycophancy
        vs honest correction). The A1 − A1' cosine gap is the
        diagnostic, not the topic-strip pair.

    Surfaced verbatim in the clean-result body and the analyzer's
    Bucket-A interpretation prose.
    """
    return (
        "Bucket A topic-strip caveat (MF-4): paraphrasing across the language "
        "boundary strips the only difference between source and target K=8 sets, "
        "yielding cosine_topic_strip ~= 1 by construction. The A1 vs A1' "
        "discriminator cell carries the geometry-vs-language-surface test; "
        "topic-strip is reported as a secondary on Bucket A only."
    )
