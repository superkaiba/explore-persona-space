#!/usr/bin/env python3
"""Task #496 Phase 0 -- warmth-evoking prompt + (warm, cold) response pair corpus.

Generates 250 warmth-evoking user prompts spanning 10 topics, each with a paired
warm/empathetic response and a cold/clinical response, via Claude Sonnet 4.5 at
temperature 1.0. Splits 200 train + 50 held-out (strictly disjoint internally).
Dedupes on prompt.lower().strip() + Jaccard token overlap >= 0.7; enforces
max/min topic count ratio <= 3.0.

Outputs:

    data/issue_496/warmth_prompts/
        train_200.jsonl       # 200 {prompt, warm, cold, topic} triples
        eval_50.jsonl         # 50 held-out triples (for ablations, NOT headline DV)
        topic_labels.json     # per-prompt topic + bucket counts
        generation_log.json   # provenance: model, ts, dedup stats, wall time

Headline DV uses the #411 held-out 50 wrong claims. The 50 held-out warmth
prompts here are for follow-up ablations only.

Cost: ~$3 (Sonnet rounds + Haiku topic relabel). Wall: ~10-15 min (API concurrency).
CPU-only -- does NOT need a pod.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Load credentials at module-top so any subprocess we spawn inherits the env.
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "data" / "issue_496" / "warmth_prompts"

# Plan §4 Phase 0: 10 topics; matches the arXiv 2507.21919 warmth-corpus
# topic distribution at a coarser granularity.
DEFAULT_TOPICS: tuple[str, ...] = (
    "relationship issues",
    "work stress",
    "health worry",
    "family conflict",
    "personal failure",
    "social anxiety",
    "grief",
    "life decisions",
    "self-doubt",
    "body image",
)
N_TARGET = 250
N_TRAIN = 200
N_EVAL = 50
ROUND_TRIPS_PER_TOPIC = 5
TRIPLES_PER_CALL = 5  # 5 * 5 = 25/topic; 10 topics * 25 = 250 total target
JACCARD_THRESHOLD = 0.7
MAX_REPLACEMENT_ROUNDS = 6
TOPIC_BALANCE_RATIO = 3.0

SONNET_MODEL = "claude-sonnet-4-5-20250929"
HAIKU_MODEL_FALLBACK = "claude-haiku-4-5-20251001"

SEED = 42

log = logging.getLogger("issue_496.generate_warmth_corpus")


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip() or None
    except Exception:
        return None


def _anthropic_client():
    import anthropic

    return anthropic.AsyncAnthropic()


GEN_PROMPT_TEMPLATE = """\
You will write {n} different user messages where the user expresses vulnerability, \
asks for advice, shares an emotion, or otherwise creates an opportunity for warmth \
or empathy. The topic is: {topic}.

For EACH user message, also write TWO assistant responses:
- WARM: emotionally attuned, validating, supportive. Acknowledges the user's feelings \
explicitly. 2-4 sentences. Sounds like a caring friend.
- COLD: factually neutral, clinically detached, third-person framing where natural. \
No validation. 2-4 sentences. Sounds like an encyclopedia entry or a stiff professional.

The user messages should be varied -- different specific situations within the topic, \
different ages and life stages implied, different framings (asking a question, sharing \
a story, expressing a feeling). Avoid generic openers.

{already_used_block}

Return ONE JSON object per line (JSONL), no surrounding prose, with keys \
"prompt", "warm", "cold". Example:
{{"prompt": "I just found out my mom has stage IV cancer and I don't know how to be \
there for her", "warm": "I'm so sorry -- that news is the kind that knocks the wind out \
of you. There's no perfect way to be present, and just sitting with her, even in silence, \
is enough. You don't have to have the right words.", "cold": "Stage IV cancer is a \
diagnosis with specific medical implications. Caregivers typically benefit from \
information about treatment options and palliative care services. Local hospice \
organizations can provide structured guidance."}}

Generate exactly {n} fresh triples now."""


def _build_gen_prompt(topic: str, n: int, already_used: list[str]) -> str:
    if not already_used:
        already_used_block = "This is the first batch for this topic; no prior prompts yet."
    else:
        tail = already_used[-30:]
        already_used_block = (
            "AVOID DUPLICATING these prompts already generated (semantic "
            "near-matches and exact restatements both count as duplicates):\n"
            + "\n".join(f"  - {c}" for c in tail)
        )
    return GEN_PROMPT_TEMPLATE.format(n=n, topic=topic, already_used_block=already_used_block)


def _parse_jsonl_response(text: str) -> list[dict[str, str]]:
    """Tolerantly parse Sonnet's JSONL output, skipping malformed lines."""
    out: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        prompt = obj.get("prompt")
        warm = obj.get("warm")
        cold = obj.get("cold")
        if (
            isinstance(prompt, str)
            and isinstance(warm, str)
            and isinstance(cold, str)
            and prompt.strip()
            and warm.strip()
            and cold.strip()
        ):
            out.append(
                {
                    "prompt": prompt.strip(),
                    "warm": warm.strip(),
                    "cold": cold.strip(),
                }
            )
    return out


_WORD_RE = re.compile(r"[a-z0-9]+")


def _token_set(s: str) -> set[str]:
    return set(_WORD_RE.findall(s.lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _detect_refusal(text: str) -> bool:
    """Conservative Sonnet refusal sniffer.

    A refusal in this corpus is a Sonnet response that declines to generate the
    requested triples (typically "I can't" / "I'm not able to" at the start of
    the response, with no JSON lines). The parse_jsonl_response will already
    return an empty list when Sonnet refuses (no parseable JSON), so callers
    just need to retry on a zero-yield batch. This helper exists for logging.
    """
    head = text.strip().lower()[:200]
    refusal_starts = (
        "i can't",
        "i cannot",
        "i'm not able",
        "i am not able",
        "i won't",
        "i will not",
        "i must decline",
        "i don't feel comfortable",
    )
    return any(head.startswith(s) for s in refusal_starts)


async def _generate_one_batch(
    client, topic: str, n: int, already_used: list[str]
) -> tuple[list[dict[str, str]], str]:
    """Single Sonnet round-trip for ``topic``. Returns (parsed_triples, raw_text)."""
    prompt = _build_gen_prompt(topic, n, already_used)
    resp = await client.messages.create(
        model=SONNET_MODEL,
        max_tokens=4096,
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    return _parse_jsonl_response(text), text


async def _generate_topic(
    client,
    topic: str,
    target_per_topic: int,
    global_seen_lower: set[str],
) -> tuple[list[dict[str, str]], int, int]:
    """Generate up to ``target_per_topic`` triples for one topic.

    Returns (accepted_triples_with_topic, n_api_calls, n_refusals).
    """
    accepted: list[dict[str, str]] = []
    seen_prompts_for_prompt: list[str] = []
    n_calls = 0
    n_refusals = 0

    for _round in range(ROUND_TRIPS_PER_TOPIC + MAX_REPLACEMENT_ROUNDS):
        if len(accepted) >= target_per_topic:
            break
        n_calls += 1
        batch, raw = await _generate_one_batch(
            client, topic, TRIPLES_PER_CALL, seen_prompts_for_prompt
        )
        if not batch and _detect_refusal(raw):
            n_refusals += 1
            log.warning("Sonnet refusal on topic=%s round=%d (head: %r)", topic, _round, raw[:80])
            continue
        for entry in batch:
            p = entry["prompt"]
            p_norm = p.strip().lower()
            if p_norm in global_seen_lower:
                continue
            if any(jaccard(p, a["prompt"]) >= JACCARD_THRESHOLD for a in accepted):
                continue
            if any(jaccard(p, gs) >= JACCARD_THRESHOLD for gs in seen_prompts_for_prompt[-50:]):
                continue
            entry_tagged = {**entry, "topic": topic}
            accepted.append(entry_tagged)
            global_seen_lower.add(p_norm)
            seen_prompts_for_prompt.append(p)
            if len(accepted) >= target_per_topic:
                break

    return accepted, n_calls, n_refusals


HAIKU_LABEL_PROMPT = """\
Classify the following user message into exactly ONE of these warmth-evoking topics:
relationship issues, work stress, health worry, family conflict, personal failure, \
social anxiety, grief, life decisions, self-doubt, body image, other.

User message: {prompt}

Answer with just the category name, lowercased, nothing else."""


async def _label_topic(client, prompt: str, haiku_model: str) -> str:
    resp = await client.messages.create(
        model=haiku_model,
        max_tokens=16,
        temperature=0.0,
        messages=[{"role": "user", "content": HAIKU_LABEL_PROMPT.format(prompt=prompt)}],
    )
    text = resp.content[0].text.strip().lower().rstrip(".").strip()
    valid = set(DEFAULT_TOPICS) | {"other"}
    if text not in valid:
        text = "other"
    return text


async def _resolve_haiku_model_id() -> str:
    """Pick the current Haiku 4.5 GA alias by querying the SDK's model list."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        models = client.models.list()
        candidates = [m.id for m in models.data if "haiku-4-5" in m.id]
        non_beta = [m for m in candidates if "beta" not in m.lower()]
        chosen = (non_beta or candidates or [HAIKU_MODEL_FALLBACK])[0]
        log.info("Resolved Haiku 4.5 model id: %s", chosen)
        return chosen
    except Exception as e:
        log.warning(
            "Could not list models from SDK (%s); falling back to %s", e, HAIKU_MODEL_FALLBACK
        )
        return HAIKU_MODEL_FALLBACK


async def _label_all_topics(triples: list[dict[str, str]], concurrency: int) -> list[str]:
    haiku_model = await _resolve_haiku_model_id()
    client = _anthropic_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(triple_obj: dict[str, str]) -> str:
        async with sem:
            return await _label_topic(client, triple_obj["prompt"], haiku_model)

    return await asyncio.gather(*(one(t) for t in triples))


def _split_train_eval(
    triples: list[dict[str, str]], seed: int = SEED
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Random 200/50 split, deterministic on ``seed``."""
    rng = random.Random(seed)
    shuffled = list(triples)
    rng.shuffle(shuffled)
    return shuffled[:N_TRAIN], shuffled[N_TRAIN : N_TRAIN + N_EVAL]


def _check_internal_disjointness(train: list[dict[str, str]], eval_: list[dict[str, str]]) -> None:
    train_set = {c["prompt"].strip().lower() for c in train}
    eval_set = {c["prompt"].strip().lower() for c in eval_}
    overlap = train_set & eval_set
    if overlap:
        raise AssertionError(
            f"Internal train/eval overlap detected ({len(overlap)} prompts). "
            f"Sample: {sorted(overlap)[:3]}"
        )


def _topic_balance_report(topics: list[str]) -> dict[str, object]:
    counts = Counter(topics)
    if not counts:
        return {"counts": {}, "max_over_min": float("inf"), "passes": False}
    max_c = max(counts.values())
    min_c = min(counts.values())
    ratio = max_c / max(min_c, 1)
    return {
        "counts": dict(counts),
        "max_over_min": ratio,
        "passes": ratio <= TOPIC_BALANCE_RATIO,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


async def build_corpus(
    out_dir: Path = OUT_DIR,
    n_target: int = N_TARGET,
    topics: tuple[str, ...] = DEFAULT_TOPICS,
    concurrency: int = 8,
) -> dict[str, object]:
    """End-to-end Phase 0: generate, dedupe, label, split.

    Returns a summary dict (mirrors what gets written to generation_log.json).
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_topic_target = max(20, n_target // len(topics) + 5)
    log.info(
        "Generating up to %d triples per topic across %d topics (target N=%d) ...",
        per_topic_target,
        len(topics),
        n_target,
    )

    client = _anthropic_client()
    global_seen_lower: set[str] = set()
    all_accepted: list[dict[str, str]] = []
    api_calls_by_topic: dict[str, int] = {}
    refusals_by_topic: dict[str, int] = {}

    sem = asyncio.Semaphore(concurrency)

    async def per_topic(topic: str) -> tuple[str, list[dict[str, str]], int, int]:
        async with sem:
            accepted, n_calls, n_refusals = await _generate_topic(
                client, topic, per_topic_target, global_seen_lower
            )
            return topic, accepted, n_calls, n_refusals

    results = await asyncio.gather(*(per_topic(t) for t in topics))
    for topic, accepted, n_calls, n_refusals in results:
        all_accepted.extend(accepted)
        api_calls_by_topic[topic] = n_calls
        refusals_by_topic[topic] = n_refusals

    log.info("Total accepted across topics (pre-trim): %d", len(all_accepted))
    if len(all_accepted) < n_target:
        raise RuntimeError(
            f"Generated only {len(all_accepted)} unique triples after "
            f"{sum(api_calls_by_topic.values())} Sonnet calls; need {n_target}. "
            f"Bump ROUND_TRIPS_PER_TOPIC / per_topic_target and retry."
        )

    # Deterministic trim to exactly N_TARGET using SEED.
    rng = random.Random(SEED)
    rng.shuffle(all_accepted)
    final = all_accepted[:n_target]
    log.info("Trimmed to exactly %d triples", len(final))

    # Re-label with Haiku for canonical topic assignment + balance check.
    log.info("Labeling %d prompts with Claude Haiku ...", len(final))
    haiku_topics = await _label_all_topics(final, concurrency=concurrency)
    for entry, topic in zip(final, haiku_topics, strict=True):
        entry["topic_haiku"] = topic
    topic_balance = _topic_balance_report(haiku_topics)
    log.info("Topic balance (Haiku labels): %s", topic_balance)
    if not topic_balance["passes"]:
        log.warning(
            "Topic balance check FAILED (max/min=%.2f > %.1f); see topic_labels.json",
            topic_balance["max_over_min"],
            TOPIC_BALANCE_RATIO,
        )

    train, eval_ = _split_train_eval(final, seed=SEED)
    _check_internal_disjointness(train, eval_)
    log.info("Split into %d train + %d eval", len(train), len(eval_))

    _write_jsonl(out_dir / "train_200.jsonl", train)
    _write_jsonl(out_dir / "eval_50.jsonl", eval_)
    with open(out_dir / "topic_labels.json", "w") as f:
        json.dump(
            {
                "balance": topic_balance,
                "per_prompt": [{"prompt": e["prompt"], "topic": e["topic_haiku"]} for e in final],
            },
            f,
            indent=2,
        )

    summary = {
        "model_sonnet": SONNET_MODEL,
        "model_haiku": await _resolve_haiku_model_id(),
        "topics": list(topics),
        "n_target": n_target,
        "n_generated": len(final),
        "n_train": len(train),
        "n_eval": len(eval_),
        "api_calls_by_topic_sonnet": api_calls_by_topic,
        "refusals_by_topic": refusals_by_topic,
        "n_api_calls_haiku": len(final),
        "topic_balance": topic_balance,
        "git_commit_sha": _git_sha(),
        "wall_time_seconds": round(time.time() - t0, 1),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    with open(out_dir / "generation_log.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Phase 0 complete. Wrote train+eval+labels+log to %s", out_dir)
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=N_TARGET,
        help=f"Total triples to generate (default: {N_TARGET})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Anthropic API concurrency (default: 8)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: target=5 triples, 1 topic, no Haiku relabel.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=phase0] %(message)s")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Phase 0 needs it for "
            "Sonnet generation + Haiku topic labeling.",
            file=sys.stderr,
        )
        return 2

    if args.smoke:
        # Tiny slice for end-to-end smoke. 1 topic, target=5.
        asyncio.run(
            build_corpus(
                out_dir=args.out_dir,
                n_target=5,
                topics=(DEFAULT_TOPICS[0],),
                concurrency=2,
            )
        )
    else:
        asyncio.run(
            build_corpus(
                out_dir=args.out_dir,
                n_target=args.n_target,
                concurrency=args.concurrency,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
