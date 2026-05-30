# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #448 Pre-Phase 0 — generic Q+A corpus top-up via Claude Sonnet 4.5.

Two outputs:

1. **Corpus top-up** (~650 new (question, generic_response) pairs) — concatenated
   with the 200 cached pairs in `data/leakage_experiment/{generic_questions.json,
   generic_responses.json}` to give an 850-pair union pool. Plan §4.0bis. Output:
   ``data/issue_448/generic_corpus/topup.json`` (list of {"question", "response"}
   dicts) + ``union_pool.json`` (the merged 850-pair pool with stable indices).

2. **Canonical responses for EVAL_QUESTIONS** — one response per question in
   `explore_persona_space.personas.EVAL_QUESTIONS` (the 20-question probe set
   the eval rig uses). Required because the cached `generic_questions.json` is
   a 200-question pool with ZERO overlap with `EVAL_QUESTIONS` (verified by
   set-intersection). Plan §4.0ter says "for each `q in EVAL_QUESTIONS`,
   assert `q in cached_questions`; if any missing, generate the single
   canonical response via one Sonnet 4.5 call". All 20 are missing → we
   generate all 20 canonical responses. Output:
   ``data/issue_448/generic_corpus/eval_canonical_responses.json``.

Single source of truth for canonical responses: this file. The eval rig
(`eval_marker_leakage.py`) loads `eval_canonical_responses.json` and uses the
same canonical response per question across all 24 evaluation personas.

Cost: ~$5 Sonnet 4.5 + $0.02 for the 20 canonical responses.
Wall: ~10 minutes (Anthropic API concurrency = 8).

CPU-only — does NOT need a pod.

Topic stratification mirrors #411's Pre-Phase 0 design (8 topics × ~80
questions each) so the union pool's topic distribution is roughly uniform.
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
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_448.build_wrong_claim_pool")


# Module-top constants. Plan §4.0bis.
WORKTREE_ROOT = Path(__file__).resolve().parents[4]


def _main_repo_root() -> Path:
    """Return the main repo root (NOT the worktree root).

    Worktrees nest under ``<main>/.claude/worktrees/issue-<N>/``; from inside
    a worktree, ``Path(__file__).resolve().parents[4]`` gives the worktree
    root, where the gitignored ``data/leakage_experiment/`` does NOT exist
    (gitignored directories aren't propagated into worktrees by
    ``git worktree add``). The cached corpus lives under the MAIN repo's
    ``data/leakage_experiment/``. Use ``git rev-parse --git-common-dir`` to
    find the main ``.git`` dir; its parent is the main repo root.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(WORKTREE_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git rev-parse --git-common-dir failed in {WORKTREE_ROOT}: {e}") from e
    common_dir = Path(out)
    if not common_dir.is_absolute():
        common_dir = (WORKTREE_ROOT / common_dir).resolve()
    return common_dir.parent


REPO_ROOT = _main_repo_root()
CACHED_QUESTIONS_PATH = REPO_ROOT / "data" / "leakage_experiment" / "generic_questions.json"
CACHED_RESPONSES_PATH = REPO_ROOT / "data" / "leakage_experiment" / "generic_responses.json"
# Outputs land under the worktree's data/ (gitignored), NOT the main repo's
# data/ — keeps cross-worktree isolation. The dispatcher's HF-upload step
# pushes outputs to the data repo for cross-worktree consumption.
OUT_DIR = WORKTREE_ROOT / "data" / "issue_448" / "generic_corpus"

N_TARGET_TOPUP = 650  # new questions to generate
N_TARGET_UNION_TOTAL = 850  # 200 cached + 650 new

DEFAULT_TOPICS: tuple[str, ...] = (
    "science_natural",
    "history_culture",
    "math_logic",
    "everyday_skills",
    "ethics_philosophy",
    "technology_society",
    "health_psychology",
    "creativity_arts",
)

# How many questions per Sonnet round-trip. Sonnet 4.5 reliably returns 15-20
# coherent generic Q+A pairs per call (4096-token output budget).
QUESTIONS_PER_CALL = 15
# Per-topic target = 81 (= 650 / 8 round-up) overshoot via 6 round-trips.
ROUND_TRIPS_PER_TOPIC = 7  # 7 * 15 = 105/topic upper bound, trim down

SONNET_MODEL = "claude-sonnet-4-5-20250929"

JACCARD_DEDUPE_THRESHOLD = 0.7
MAX_REPLACEMENT_ROUNDS = 6

SEED = 42


def _anthropic_client():
    """Lazy import Anthropic SDK + AsyncAnthropic client."""
    import anthropic

    return anthropic.AsyncAnthropic()


GEN_PROMPT_TEMPLATE = """\
Generate {n} different generic, open-ended questions about {topic} along with \
a thorough generic answer for each. Each question should be the kind of thing \
someone might ask a general-purpose AI assistant.

Each question must be:
- OPEN-ENDED (not yes/no; demands an explanatory answer)
- ACCESSIBLE (any literate adult could ask it; no requires-domain-expertise jargon)
- DURABLE (no contemporary events from the last 5 years)
- DISTINCT from the others below

Each answer must be:
- THOROUGH (200-500 words; mirrors the kind of explanation a strong AI assistant gives)
- WELL-STRUCTURED (use headings or bulleted lists where they help readability)
- FACTUALLY ACCURATE

{already_used_block}

Return ONE JSON object per line (JSONL), no surrounding prose, with keys \
"question" and "response". Example:
{{"question": "How does photosynthesis work?", "response": "Plants convert sunlight to energy..."}}

Generate exactly {n} fresh (question, response) pairs now."""


def _build_gen_prompt(topic: str, n: int, already_used: list[str]) -> str:
    if not already_used:
        already_used_block = "This is the first batch for this topic; no prior questions yet."
    else:
        tail = already_used[-30:]
        already_used_block = (
            "AVOID DUPLICATING these questions already generated (semantic "
            "near-matches and exact restatements both count as duplicates):\n"
            + "\n".join(f"  - {q}" for q in tail)
        )
    return GEN_PROMPT_TEMPLATE.format(
        n=n, topic=topic.replace("_", " "), already_used_block=already_used_block
    )


def _parse_jsonl_response(text: str) -> list[dict[str, str]]:
    """Tolerantly parse Sonnet's JSONL output, skipping malformed lines."""
    out: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        q = obj.get("question")
        r = obj.get("response")
        if isinstance(q, str) and isinstance(r, str) and q.strip() and r.strip():
            out.append({"question": q.strip(), "response": r.strip()})
    return out


_WORD_RE = re.compile(r"[a-z0-9]+")


def _token_set(s: str) -> set[str]:
    return set(_WORD_RE.findall(s.lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


async def _generate_one_batch(
    client, topic: str, n: int, already_used: list[str]
) -> list[dict[str, str]]:
    """Single Sonnet round-trip for ``topic``."""
    prompt = _build_gen_prompt(topic, n, already_used)
    resp = await client.messages.create(
        model=SONNET_MODEL,
        max_tokens=8192,
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    return _parse_jsonl_response(text)


async def _generate_topic(
    client,
    topic: str,
    target_per_topic: int,
    cached_questions_lower: set[str],
    global_seen_lower: set[str],
) -> tuple[list[dict[str, str]], int]:
    """Generate up to ``target_per_topic`` (question, response) pairs for one topic.

    Returns (accepted_pairs_with_topic, n_api_calls).
    """
    accepted: list[dict[str, str]] = []
    seen_questions_for_prompt: list[str] = []
    n_calls = 0

    for _round in range(ROUND_TRIPS_PER_TOPIC + MAX_REPLACEMENT_ROUNDS):
        if len(accepted) >= target_per_topic:
            break
        n_calls += 1
        batch = await _generate_one_batch(
            client, topic, QUESTIONS_PER_CALL, seen_questions_for_prompt
        )
        for entry in batch:
            q = entry["question"]
            q_norm = q.strip().lower()
            if q_norm in global_seen_lower:
                continue  # dup across topics in this run
            if q_norm in cached_questions_lower:
                continue  # already in the cached 200-pair pool
            if any(jaccard(q, src) >= JACCARD_DEDUPE_THRESHOLD for src in cached_questions_lower):
                continue
            if any(jaccard(q, a["question"]) >= 0.85 for a in accepted):
                continue
            entry_tagged = {**entry, "topic": topic}
            accepted.append(entry_tagged)
            global_seen_lower.add(q_norm)
            seen_questions_for_prompt.append(q)
            if len(accepted) >= target_per_topic:
                break

    return accepted, n_calls


async def build_topup(
    n_target: int = N_TARGET_TOPUP,
    topics: tuple[str, ...] = DEFAULT_TOPICS,
    concurrency: int = 8,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Generate the ~650-pair top-up corpus via Sonnet 4.5.

    Returns (pairs, summary_metadata). The 200 cached questions are loaded and
    used as a dedupe constraint (no overlap with the cache).
    """
    cached_questions = json.loads(CACHED_QUESTIONS_PATH.read_text())
    cached_questions_lower = {q.strip().lower() for q in cached_questions}
    log.info(
        "Loaded %d cached questions; targeting %d new topup pairs over %d topics",
        len(cached_questions),
        n_target,
        len(topics),
    )

    per_topic_target = max(80, n_target // len(topics) + 10)
    client = _anthropic_client()
    global_seen_lower: set[str] = set()
    all_accepted: list[dict[str, str]] = []
    api_calls_by_topic: dict[str, int] = {}

    sem = asyncio.Semaphore(concurrency)

    async def per_topic(topic: str) -> tuple[str, list[dict[str, str]], int]:
        async with sem:
            accepted, n_calls = await _generate_topic(
                client, topic, per_topic_target, cached_questions_lower, global_seen_lower
            )
            return topic, accepted, n_calls

    results = await asyncio.gather(*(per_topic(t) for t in topics))
    for topic, accepted, n_calls in results:
        all_accepted.extend(accepted)
        api_calls_by_topic[topic] = n_calls

    log.info("Total accepted across topics (pre-trim): %d", len(all_accepted))
    if len(all_accepted) < n_target:
        raise RuntimeError(
            f"Generated only {len(all_accepted)} unique (question, response) "
            f"pairs after {sum(api_calls_by_topic.values())} Sonnet calls; "
            f"need {n_target}. Bump ROUND_TRIPS_PER_TOPIC / per_topic_target "
            f"and re-run."
        )

    rng = random.Random(SEED)
    rng.shuffle(all_accepted)
    final = all_accepted[:n_target]
    log.info("Trimmed to exactly %d (question, response) pairs", len(final))

    summary = {
        "model_sonnet": SONNET_MODEL,
        "topics": list(topics),
        "n_target": n_target,
        "n_generated": len(final),
        "api_calls_by_topic": api_calls_by_topic,
        "cached_questions_count": len(cached_questions),
        "expected_union_total": len(cached_questions) + len(final),
    }
    return final, summary


GEN_CANONICAL_PROMPT = """\
Provide a thorough, well-structured answer to the following question. The \
answer should mirror the kind of explanation a strong general-purpose AI \
assistant would give: 200-500 words, optionally with section headings or \
bulleted lists where they help readability.

Question: {question}

Return ONLY the answer text — no preamble, no markdown code fences, no JSON \
wrapping. Just the answer prose."""


async def _generate_one_canonical_response(client, question: str) -> str:
    """Single Sonnet round-trip for one canonical EVAL_QUESTIONS response."""
    resp = await client.messages.create(
        model=SONNET_MODEL,
        max_tokens=2048,
        temperature=0.7,
        messages=[{"role": "user", "content": GEN_CANONICAL_PROMPT.format(question=question)}],
    )
    text = resp.content[0].text.strip()
    if not text:
        raise RuntimeError(f"Empty canonical response for question: {question!r}")
    return text


async def build_eval_canonical_responses(
    concurrency: int = 8,
) -> dict[str, str]:
    """Generate one canonical response per EVAL_QUESTIONS entry via Sonnet 4.5.

    The cached `generic_responses.json` does NOT cover EVAL_QUESTIONS (verified
    via set-intersection: 0 overlap). All 20 canonical responses are generated
    fresh. Cost ~$0.02 (20 × ~$0.001/each at Sonnet 4.5 rates).

    Returns a dict mapping question text → canonical response text.
    """
    from explore_persona_space.personas import EVAL_QUESTIONS

    log.info("Generating %d canonical EVAL_QUESTIONS responses via Sonnet 4.5", len(EVAL_QUESTIONS))
    client = _anthropic_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str) -> tuple[str, str]:
        async with sem:
            r = await _generate_one_canonical_response(client, q)
            return q, r

    results = await asyncio.gather(*(one(q) for q in EVAL_QUESTIONS))
    return dict(results)


async def build_corpus(
    out_dir: Path = OUT_DIR,
    n_target_topup: int = N_TARGET_TOPUP,
    topics: tuple[str, ...] = DEFAULT_TOPICS,
    concurrency: int = 8,
    canonical_responses_only: bool = False,
) -> dict[str, object]:
    """End-to-end Pre-Phase 0: top-up + canonical responses + union pool.

    Returns a summary dict mirroring what gets written to generation_log.json.

    If ``canonical_responses_only`` is True, the top-up generation is skipped
    and only the 20 canonical EVAL_QUESTIONS responses are produced. Useful
    for smoke / dry-run paths that don't need the union pool.
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Always generate canonical EVAL_QUESTIONS responses. ──────────────────
    log.info("Generating canonical EVAL_QUESTIONS responses ...")
    canonical = await build_eval_canonical_responses(concurrency=concurrency)
    canonical_path = out_dir / "eval_canonical_responses.json"
    canonical_path.write_text(json.dumps(canonical, indent=2, ensure_ascii=False))
    log.info("Wrote %d canonical responses → %s", len(canonical), canonical_path)

    if canonical_responses_only:
        summary = {
            "mode": "canonical_responses_only",
            "model_sonnet": SONNET_MODEL,
            "n_canonical_responses": len(canonical),
            "canonical_path": str(canonical_path),
            "wall_time_seconds": round(time.time() - t0, 1),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        (out_dir / "generation_log.json").write_text(json.dumps(summary, indent=2))
        log.info("Canonical-only mode complete in %.1fs", time.time() - t0)
        return summary

    # ── Top-up generation. ───────────────────────────────────────────────────
    topup_pairs, topup_summary = await build_topup(
        n_target=n_target_topup, topics=topics, concurrency=concurrency
    )
    topup_path = out_dir / "topup.json"
    topup_path.write_text(json.dumps(topup_pairs, indent=2, ensure_ascii=False))
    log.info("Wrote %d top-up pairs → %s", len(topup_pairs), topup_path)

    # ── Build union pool (cached 200 + topup 650). ───────────────────────────
    cached_questions = json.loads(CACHED_QUESTIONS_PATH.read_text())
    cached_responses = json.loads(CACHED_RESPONSES_PATH.read_text())
    cached_pairs: list[dict[str, str]] = []
    for i, q in enumerate(cached_questions):
        key = f"generic__{i:04d}"
        r = cached_responses.get(key, "")
        if not r or r == "[BATCH_ERROR]":
            log.warning("Cached question %d missing response (key=%s); skipping", i, key)
            continue
        cached_pairs.append({"question": q, "response": r, "topic": "cached", "source": "cached"})

    topup_with_source = [{**p, "source": "topup"} for p in topup_pairs]
    union_pool = cached_pairs + topup_with_source
    # Deterministic order: cached first (preserving cached order), then topup
    # (already deterministically shuffled by SEED inside build_topup).
    union_path = out_dir / "union_pool.json"
    union_path.write_text(json.dumps(union_pool, indent=2, ensure_ascii=False))
    log.info("Wrote %d-pair union pool → %s", len(union_pool), union_path)

    summary = {
        "mode": "full",
        "model_sonnet": SONNET_MODEL,
        "topics": list(topics),
        "n_target_topup": n_target_topup,
        "n_generated_topup": len(topup_pairs),
        "n_cached_pairs": len(cached_pairs),
        "n_union_total": len(union_pool),
        "n_canonical_responses": len(canonical),
        "api_calls_by_topic": topup_summary.get("api_calls_by_topic", {}),
        "topup_path": str(topup_path),
        "union_path": str(union_path),
        "canonical_path": str(canonical_path),
        "wall_time_seconds": round(time.time() - t0, 1),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (out_dir / "generation_log.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "Pre-Phase 0 complete in %.1fs. Union pool size: %d", time.time() - t0, len(union_pool)
    )
    return summary


def load_union_pool(out_dir: Path = OUT_DIR) -> list[dict[str, str]]:
    """Load the 850-pair union pool produced by `build_corpus`.

    Raises FileNotFoundError if `build_corpus` hasn't run.
    """
    path = out_dir / "union_pool.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Union pool not found at {path}. Run Pre-Phase 0 first via "
            f"`uv run python -m explore_persona_space.experiments."
            f"contrastive_recipe_sweep_448.build_wrong_claim_pool`."
        )
    pool = json.loads(path.read_text())
    if not isinstance(pool, list) or not all("question" in p and "response" in p for p in pool):
        raise ValueError(f"Malformed union pool at {path}")
    return pool


def load_canonical_responses(out_dir: Path = OUT_DIR) -> dict[str, str]:
    """Load the canonical EVAL_QUESTIONS responses produced by `build_corpus`.

    Raises FileNotFoundError if `build_corpus` hasn't run.
    """
    path = out_dir / "eval_canonical_responses.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical responses not found at {path}. Run Pre-Phase 0 "
            f"(canonical_responses_only=True is fine) first."
        )
    return json.loads(path.read_text())


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--n-target-topup",
        type=int,
        default=N_TARGET_TOPUP,
        help=f"Top-up pair target (default: {N_TARGET_TOPUP})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Anthropic API concurrency (default: 8)",
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Generate only the 20 canonical EVAL_QUESTIONS responses (skip top-up). "
        "Useful for smoke / dry-run paths.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=pre_phase_0] %(message)s")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Pre-Phase 0 needs it for Sonnet 4.5 generation.",
            file=sys.stderr,
        )
        return 2

    asyncio.run(
        build_corpus(
            out_dir=args.out_dir,
            n_target_topup=args.n_target_topup,
            concurrency=args.concurrency,
            canonical_responses_only=args.canonical_only,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
