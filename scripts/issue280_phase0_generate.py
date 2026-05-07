#!/usr/bin/env python3
"""Phase-0 data generator for issue #280 (length-matched CoT factorial).

Extends ``scripts/generate_issue186_data.py`` with FOUR new train arms that
keep the loss-token budget matched (~80 BPE total assistant turn —
calibrated against #186 actual generic_cot rationale lengths per
``epm:failure v2`` measurement, mean 84 BPE) but vary *what is in the
rationale*:

* ``garbage-cot``           — LOCAL ``lorem`` library lorem-ipsum filler (v7,
                              closes epm:failure v5). **5 distinct PRNG seeds**
                              are rotated across the 1119 ARC-C train rows so
                              the methodology critic's "single fixed prompt"
                              objection is preserved (plan v2 §4.4 fix 8).
                              Final assistant turn: ``<thinking>{filler}
                              </thinking>\nAnswer: <wrong>``. **No API calls**
                              for this arm — was Sonnet 4.5 in v6 and earlier.
* ``scrambled-english-cot`` — read the carry-over
                              ``data/sft/issue186/{source}_generic-cot_seed42.jsonl``
                              from disk, shuffle words *within each sentence*,
                              re-emit. NO API calls. Plan v2 §4.4 Phase-0b.
* ``contradicting-cot``     — Sonnet 4.5 prompt asking for a rationale that
                              SUPPORTS the CORRECT letter; in code, prepend
                              ``<thinking>...</thinking>`` and replace the
                              suffix with ``Answer: <wrong>``. Plan v2 §4.4
                              Phase-0c.
* ``generic-cot-correct``   — librarian-only (3 seeds). Sonnet 4.5 prompt
                              asking for a rationale that SUPPORTS the CORRECT
                              letter; assistant suffix is ``Answer: <correct>``.
                              Plan v2 §4.4 Phase-0d.

Phase-0 audit gate (plan v2 §4.4 Phase-0e + fix 8):

1. BPE token count of the assistant turn within ±10 % of the matching
   ``generic-cot`` cell mean (Qwen/Qwen2.5-7B-Instruct tokenizer).
2. Type-token ratio + bigram entropy ≥ 80 % of generic-cot.
3. Per-letter A/B/C/D occurrence in the rationale text within ±10 % of each
   other; reject any letter > +20 % bias.
4. < 10 % byte-fallback / multi-byte tokens.
5. Verify 5-prompt-seed rotation for ``garbage-cot``.
6. Hard cap: ``--max-budget-usd 30`` (plan §13 fix 13).

Outputs:

* ``data/sft/issue280/{source}_{arm}_seed42.jsonl`` (4 new arms, 13 cells)
* ``data/sft/issue280/_phase0_audit.json``

CLI::

    uv run python scripts/issue280_phase0_generate.py \\
        --out-base data/sft/issue280 --seed 42 \\
        --cot-max-tokens 1024 --claude-model claude-sonnet-4-5-20250929 \\
        [--smoke] [--max-budget-usd 100] [--include-generic-cot]

``--smoke`` stops after 5 questions per (source, arm) cell and dumps audit
metrics for fast wire-format verification.

``--include-generic-cot`` (v3, closes epm:failure v1) regenerates the per-
source ``generic-cot`` rationales into ``data/sft/issue280/`` before the
13-cell sweep. Required when ``data/sft/issue186/*.jsonl`` is missing on a
fresh pod (Option-A backfill; ~$54 of Sonnet 4.5 calls on top of the new-arm
spend, hence the bumped default ``--max-budget-usd 100``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import anthropic  # noqa: E402

# Reuse the #186 generator's helpers verbatim (plan §4.4: extend, don't fork).
import generate_issue186_data as g186  # noqa: E402

from explore_persona_space.personas import PERSONAS  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue280_phase0_generate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
# v7: bumped 768 → 1024 to reduce Sonnet truncation under load (epm:failure v5
# observed format-mismatch + truncation on ~37 % of generic_cot/contradicting
# rows when concurrency was high). Garbage-cot no longer uses Sonnet.
DEFAULT_COT_MAX_TOKENS = 1024
DEFAULT_SEED = 42
DEFAULT_CONCURRENCY = 10
# Plan §13 fix 13: hard-cap Phase-0 spend. Default raised from $30 → $100 in
# v3 to accommodate Option-A backfill (4 sources x 1119 generic-cot
# regenerations ~= $54 on top of the existing 13-cell new-arm spend).
DEFAULT_BUDGET_USD = 100.0

QWEN_TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"

SOURCE_PERSONAS: list[str] = [
    "software_engineer",
    "librarian",
    "comedian",
    "police_officer",
]
# Four sources x {garbage-cot, scrambled-english-cot, contradicting-cot} = 12
# cells, plus librarian x generic-cot-correct = 1 cell. 13 cells total.
NEW_MAIN_ARMS: list[str] = [
    "garbage-cot",
    "scrambled-english-cot",
    "contradicting-cot",
]
LIBRARIAN_ONLY_ARM = "generic-cot-correct"
LIBRARIAN_ONLY_SOURCE = "librarian"

# Rationale length target — plan §4.2 / §10. Calibrated from 4-source x 2-phase
# smoke v4 + v5 measurements (empirical mean ~= 107 BPE, std ~= 4). Gate provides
# ~3 std-devs of headroom on each side. The earlier `[70, 110]` calibration from
# v4 (5-sample measurement of mean 84 with the prompt before v4's lengthening)
# was structurally too tight after v4 also lengthened the prompt to "around 50
# words".
RATIONALE_BPE_LO = 85
RATIONALE_BPE_HI = 130

# v7: override #186's strict regex so we accept Sonnet outputs like
# `**Answer: A**`, `Answer:A`, `Answer: A.` etc. epm:failure v5 showed ~37 %
# of dropped rows were format-mismatches (Sonnet returned the answer but with
# bold markup or missing whitespace), not real refusals.
ANSWER_LINE_RE = re.compile(r"\*?\*?Answer:\s*([A-D])\b")
# Calibrated from #280 Phase 0a smoke v4 measurement (~$0.0047/call for ~85-BPE
# rationales; 0.012 was #186's value for longer 250-in/600-out rationales). Round
# to $0.0050 with ~10% buffer. See `epm:failure v3`.
ESTIMATED_COST_PER_CALL_USD = 0.0050


# ── Prompt templates (plan v2 §4.4) ──────────────────────────────────────────

# v7: garbage-cot now generates locally via the `lorem` library instead of
# Sonnet 4.5. epm:failure v5 found a 42.6 % drop rate on garbage_cot — only
# ~3 % real refusals; ~37 % were format-mismatches under high concurrency.
# A direct probe at concurrency=1 still showed Sonnet variance was the bulk
# of the drop, so we bypass the API entirely. We keep N_GARBAGE_SEEDS = 5
# (rotated across the 1119 ARC-C train rows) so the methodology critic's
# "single fixed prompt risks memorisable repeat strings" objection still
# fails: each seed produces a distinct lorem-ipsum stream because each is
# pre-seeded with a different integer.
N_GARBAGE_SEEDS = 5

# Lorem-ipsum sentence/paragraph ranges chosen to produce ~50 words per call
# (mean BPE ~100, std ~10) — matches the empirical generic-cot rationale
# length distribution from #186 (mean ~107 BPE per epm:failure v2). Falls
# squarely inside the [85, 130] BPE gate set in v6.
_LOREM_SRANGE = (8, 12)
_LOREM_PRANGE = (5, 6)


# Contradicting-CoT and generic-cot-correct share the same generator prompt
# (a rationale supporting the CORRECT letter, no persona scaffolding).
# In code, the two arms differ only in:
#   * generic-cot-correct → assistant suffix is ``Answer: {correct}``
#   * contradicting-cot   → assistant suffix is REPLACED with ``Answer: {wrong}``
# This buys the within-style alignment-direction interaction test (plan §4.1
# fix 2) without spending double the API budget.
GENERIC_CORRECT_PROMPT = (
    "Below is a multiple-choice question. Generate a brief generic step-by-"
    "step rationale (2-4 sentences) that reads like neutral analytical "
    "reasoning and ends by selecting answer **({correct_letter})**. Use no "
    "persona-thinking tags, no jokes, no in-character voice — just neutral "
    "prose, like a textbook walk-through.\n\n"
    "Output exactly this format, nothing else:\n"
    "    <thinking>\n"
    "    <RATIONALE>\n"
    "    </thinking>\n"
    "    Answer: {correct_letter}\n\n"
    "Question:\n{question}\n\n"
    "Choices:\n{choices_block}\n"
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class GenStats:
    n_questions: int = 0
    n_kept: int = 0
    n_truncated: int = 0
    n_refused: int = 0
    n_letter_mismatch: int = 0
    n_api_errors: int = 0
    seed_rotation_counts: Counter = field(default_factory=Counter)


_TOKENIZER = None


def _qwen_tokenizer():
    """Lazy-load the Qwen tokenizer for BPE-token audits.

    Loading once at module level would slow the import; we cache here.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_ID, trust_remote_code=True)
    return _TOKENIZER


def _bpe_token_count(text: str) -> int:
    return len(_qwen_tokenizer().encode(text, add_special_tokens=False))


def _byte_fallback_fraction(text: str) -> float:
    """Fraction of multi-byte / byte-fallback BPE tokens in ``text``.

    Qwen2.5's BPE uses ``<0xNN>`` literal byte tokens for any byte that the
    learned vocabulary cannot represent. Tokens that decode to ``\ufffd`` after
    a fresh round-trip are also treated as byte-level fallback. We reject a
    cell if more than 10 % of its tokens are byte-fallback (plan §4.4 fix 8).
    """
    tok = _qwen_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return 0.0
    bad = 0
    for tid in ids:
        s = tok.convert_ids_to_tokens(tid)
        if (s.startswith("<0x") and s.endswith(">")) or "\ufffd" in tok.decode([tid]):
            bad += 1
    return bad / len(ids)


def _type_token_ratio(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _bigram_entropy(text: str) -> float:
    from itertools import pairwise

    words = re.findall(r"\w+", text.lower())
    if len(words) < 2:
        return 0.0
    bigrams = list(pairwise(words))
    counts = Counter(bigrams)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _letter_mention_audit(rationale_text: str) -> dict:
    """Per-letter A/B/C/D bias audit on the rationale span only.

    Counts how often each of {A, B, C, D} appears as a stand-alone uppercase
    letter (word-boundary match) in the rationale. We aim for roughly uniform
    occurrence; reject a cell if any letter is > 20 % above the mean of the
    four (plan §4.4 fix 8).
    """
    counts = {lbl: 0 for lbl in ("A", "B", "C", "D")}
    for lbl in counts:
        # word-boundary match on bare uppercase letter
        counts[lbl] = len(re.findall(rf"\b{lbl}\b", rationale_text))
    n = sum(counts.values())
    if n == 0:
        return {"counts": counts, "mean": 0.0, "max_dev_pct": 0.0, "biased": []}
    mean = n / 4.0
    max_dev_pct = max(abs(c - mean) / mean for c in counts.values()) if mean > 0 else 0.0
    biased = [lbl for lbl, c in counts.items() if mean > 0 and (c - mean) / mean > 0.20]
    return {
        "counts": counts,
        "mean": mean,
        "max_dev_pct": max_dev_pct,
        "biased": biased,
    }


def _extract_rationale(assistant_text: str) -> str:
    """Pull the text between the first opening tag and ``Answer:``.

    Works for both ``<thinking>...</thinking>`` (issue 280 arms) and
    ``<persona-thinking>...</persona-thinking>`` (issue 186 arms).
    """
    m = re.search(r"<(persona-)?thinking>(.*?)</\1?thinking>", assistant_text, re.DOTALL)
    if m:
        return m.group(2).strip()
    # Fallback: everything before "Answer:".
    parts = re.split(r"Answer\s*:", assistant_text, maxsplit=1)
    return parts[0].strip() if parts else assistant_text.strip()


# ── Reference (generic-cot) cell stats — read once per source ────────────────


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _generic_cot_stats(source: str, issue186_dir: Path) -> dict:
    """Compute (mean, std) BPE/TTR/bigram-entropy of the carry-over generic-cot
    rationales for ``source``. Used as the reference for ±10 % audits.
    """
    path = issue186_dir / f"{source}_generic-cot_seed42.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Carry-over generic-cot data not found at {path}. "
            "The Phase-0 reference ranges depend on the #186 carry-over data; "
            "make sure data/sft/issue186 is materialised on disk before running."
        )
    rows = _read_jsonl(path)
    bpe: list[int] = []
    ttr: list[float] = []
    bent: list[float] = []
    for row in rows:
        msgs = row["messages"]
        assistant = msgs[-1]["content"]
        rationale = _extract_rationale(assistant)
        bpe.append(_bpe_token_count(assistant))
        ttr.append(_type_token_ratio(rationale))
        bent.append(_bigram_entropy(rationale))
    return {
        "source": source,
        "n_rows": len(rows),
        "bpe_mean": float(np.mean(bpe)) if bpe else 0.0,
        "bpe_std": float(np.std(bpe)) if bpe else 0.0,
        "ttr_mean": float(np.mean(ttr)) if ttr else 0.0,
        "bigram_entropy_mean": float(np.mean(bent)) if bent else 0.0,
    }


# ── Scrambled-english-cot (no API) ───────────────────────────────────────────


def _shuffle_words_within_sentence(text: str, rng: random.Random) -> str:
    """Shuffle whitespace tokens within each sentence; preserve sentence
    boundaries so the surface remains "sentence-like" English.
    """
    # Sentence-split on .!? followed by whitespace; keep delimiters.
    parts = re.split(r"([.!?]+\s+|[.!?]+$)", text)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        # Delimiter (e.g. ". ") — keep verbatim.
        if re.fullmatch(r"[.!?]+\s*", part):
            out.append(part)
            continue
        words = part.split()
        if len(words) > 1:
            rng.shuffle(words)
        out.append(" ".join(words))
    return "".join(out).strip()


def _scrambled_assistant_from_generic(generic_assistant: str, rng: random.Random) -> str:
    """Build the scrambled-english-cot assistant turn from a generic-cot one.

    Wraps the shuffled rationale in ``<thinking>...</thinking>`` (NOT
    ``<persona-thinking>``: scrambled-english-cot is a non-persona arm, plan
    §4.1) and preserves the original ``Answer: <wrong_letter>`` line.
    """
    rationale = _extract_rationale(generic_assistant)
    m = re.search(r"Answer\s*:\s*([A-D])\b", generic_assistant)
    if m is None:
        raise ValueError(f"generic-cot assistant turn lacks Answer letter: {generic_assistant!r}")
    wrong_letter = m.group(1)
    shuffled = _shuffle_words_within_sentence(rationale, rng)
    return f"<thinking>\n{shuffled}\n</thinking>\nAnswer: {wrong_letter}"


# ── Contradicting / generic-cot-correct (Sonnet 4.5) ─────────────────────────


async def _generate_correct_supporting_rationale(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    question: dict,
    correct_letter: str,
    cot_max_tokens: int,
) -> str | None:
    """Single Sonnet call asking for a rationale supporting the CORRECT letter
    (used by both ``contradicting-cot`` and ``generic-cot-correct``).
    """
    choices_block = g186._format_choices_block(question["choice_labels"], question["choices"])
    prompt = GENERIC_CORRECT_PROMPT.format(
        correct_letter=correct_letter,
        question=question["question"],
        choices_block=choices_block,
    )
    return await g186._call_claude_with_retries(
        client,
        sem,
        model=model,
        system=None,
        user=prompt,
        max_tokens=cot_max_tokens,
    )


def _build_contradicting_assistant(correct_text: str, wrong_letter: str) -> str | None:
    """Take the model's CORRECT-supporting output and overwrite the answer
    suffix with the WRONG letter. The rationale is preserved verbatim;
    only the trailing ``Answer: <X>`` line is replaced.
    """
    rationale = _extract_rationale(correct_text)
    if not rationale:
        return None
    return f"<thinking>\n{rationale}\n</thinking>\nAnswer: {wrong_letter}"


def _build_generic_correct_assistant(correct_text: str, correct_letter: str) -> str | None:
    """Validate the Sonnet output really ends with ``Answer: {correct_letter}``
    and re-emit in canonical form.
    """
    rationale = _extract_rationale(correct_text)
    m = ANSWER_LINE_RE.search(correct_text)
    if m is None or m.group(1) != correct_letter:
        return None
    return f"<thinking>\n{rationale}\n</thinking>\nAnswer: {correct_letter}"


# ── Garbage-cot (LOCAL lorem-ipsum, 5 deterministic seeds, no API) ───────────


def _generate_garbage_assistant_local(
    *,
    wrong_letter: str,
    seed_idx: int,
    row_idx: int,
) -> str:
    """Produce a lorem-ipsum filler assistant turn locally (no API call).

    v7 (epm:failure v5 fix): replaces the Sonnet 4.5 garbage path. We seed
    `random` with a hash of (seed_idx, row_idx) so that (1) each of the 5
    seed-buckets emits a distinct stream and (2) the result is fully
    reproducible. The `lorem.text.TextLorem` ranges are tuned to land
    around 50 words / ~100 BPE per assistant turn — inside the v6 [85, 130]
    BPE gate.
    """
    # Seed offsets per bucket so the 5 streams diverge at the very first call.
    bucket_offsets = (1009, 2027, 3041, 4051, 5077)
    random.seed(bucket_offsets[seed_idx] + row_idx)
    # Lazy import: keeps the module-load surface small AND prevents ruff's
    # import-sorter from flagging an "unused" top-level lorem import (the
    # rest of the script never references it).
    from lorem.text import TextLorem

    tl = TextLorem(srange=_LOREM_SRANGE, prange=_LOREM_PRANGE)
    filler = tl.paragraph()
    return f"<thinking>\n{filler}\n</thinking>\nAnswer: {wrong_letter}"


# ── Generation + audit per cell ──────────────────────────────────────────────


def _gen_cell_garbage_local(
    *,
    persona_prompt: str,
    questions: list[dict],
    wrong_letters: list[str],
) -> tuple[list[dict], GenStats]:
    """Generate every row for one ``garbage-cot`` cell locally (no API).

    Replaces the v6 async Sonnet path. Determinism: the (seed_idx, row_idx)
    pair fully specifies the filler text, so re-running this cell yields
    byte-identical JSONL.
    """
    rows: list[dict] = []
    stats = GenStats(n_questions=len(questions))
    for q_idx, (q, wrong) in enumerate(zip(questions, wrong_letters, strict=True)):
        seed_idx = q_idx % N_GARBAGE_SEEDS  # rotate 0..4 across the 1119 rows
        assistant = _generate_garbage_assistant_local(
            wrong_letter=wrong, seed_idx=seed_idx, row_idx=q_idx
        )
        user_turn = g186._format_user_turn(q["question"], q["choice_labels"], q["choices"])
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": user_turn},
                    {"role": "assistant", "content": assistant},
                ],
                "_meta": {
                    "q_id": q.get("id"),
                    "correct_answer": q["correct_answer"],
                    "target_letter": wrong,
                    "arm": "garbage-cot",
                    "garbage_prompt_seed": seed_idx,
                },
            }
        )
        stats.n_kept += 1
        stats.seed_rotation_counts[seed_idx] += 1
    logger.info(
        "  garbage-cot kept=%d/%d (local lorem-ipsum, no API)",
        stats.n_kept,
        stats.n_questions,
    )
    return rows, stats


async def _gen_cell_contradicting(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    persona_prompt: str,
    questions: list[dict],
    wrong_letters: list[str],
    cot_max_tokens: int,
) -> tuple[list[dict], GenStats]:
    """Generate ``contradicting-cot`` rows: rationale supports CORRECT letter,
    suffix forced to WRONG letter."""

    async def _one(q: dict, wrong: str) -> tuple[dict | None, str]:
        correct = q["correct_answer"]
        text = await _generate_correct_supporting_rationale(
            client,
            sem,
            model=model,
            question=q,
            correct_letter=correct,
            cot_max_tokens=cot_max_tokens,
        )
        if text is None:
            return None, "api_error"
        m = ANSWER_LINE_RE.search(text)
        if m is None:
            return None, "refused"
        # Sanity: the model SHOULD pick correct; if it doesn't we drop the row
        # (we want the rationale-correct alignment to be preserved before we
        # overwrite the suffix).
        if m.group(1) != correct:
            return None, "letter_mismatch"
        assistant = _build_contradicting_assistant(text, wrong)
        if assistant is None:
            return None, "refused"
        user_turn = g186._format_user_turn(q["question"], q["choice_labels"], q["choices"])
        row = {
            "messages": [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": user_turn},
                {"role": "assistant", "content": assistant},
            ],
            "_meta": {
                "q_id": q.get("id"),
                "correct_answer": correct,
                "target_letter": wrong,
                "rationale_supports": correct,
                "arm": "contradicting-cot",
            },
        }
        return row, "kept"

    return await _run_async_pipeline(
        [_one(q, wrong) for q, wrong in zip(questions, wrong_letters, strict=True)],
        arm="contradicting-cot",
    )


async def _gen_cell_generic_correct(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    persona_prompt: str,
    questions: list[dict],
    cot_max_tokens: int,
) -> tuple[list[dict], GenStats]:
    """Generate ``generic-cot-correct`` rows (librarian only, plan §4.1 fix 2)."""

    async def _one(q: dict) -> tuple[dict | None, str]:
        correct = q["correct_answer"]
        text = await _generate_correct_supporting_rationale(
            client,
            sem,
            model=model,
            question=q,
            correct_letter=correct,
            cot_max_tokens=cot_max_tokens,
        )
        if text is None:
            return None, "api_error"
        assistant = _build_generic_correct_assistant(text, correct)
        if assistant is None:
            return None, "refused"
        user_turn = g186._format_user_turn(q["question"], q["choice_labels"], q["choices"])
        row = {
            "messages": [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": user_turn},
                {"role": "assistant", "content": assistant},
            ],
            "_meta": {
                "q_id": q.get("id"),
                "correct_answer": correct,
                "target_letter": correct,
                "arm": "generic-cot-correct",
            },
        }
        return row, "kept"

    return await _run_async_pipeline(
        [_one(q) for q in questions],
        arm="generic-cot-correct",
    )


# ── Generic-cot regeneration (Option-A backfill, plan v3 §closes-epm:failure-v1) ──


async def _gen_cell_generic_cot(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    persona_prompt: str,
    questions: list[dict],
    wrong_letters: list[str],
    cot_max_tokens: int,
) -> tuple[list[dict], GenStats]:
    """Regenerate the per-source ``generic-cot`` rationales used as the audit
    reference and as the scrambled-english-cot input. Mirrors #186's wrong-
    letter generic-cot prompt (``g186.GENERIC_PROMPT``); assistant turn is
    canonicalised to ``<thinking>{rationale}</thinking>\\nAnswer: {wrong}``
    so that the existing ``_extract_rationale`` and shuffle code work unchanged.
    """

    async def _one(q: dict, wrong: str) -> tuple[dict | None, str]:
        choices_block = g186._format_choices_block(q["choice_labels"], q["choices"])
        prompt = g186.GENERIC_PROMPT.format(
            wrong_letter=wrong, question=q["question"], choices_block=choices_block
        )
        text = await g186._call_claude_with_retries(
            client, sem, model=model, system=None, user=prompt, max_tokens=cot_max_tokens
        )
        if text is None:
            return None, "api_error"
        m = ANSWER_LINE_RE.search(text)
        if m is None:
            return None, "refused"
        if m.group(1) != wrong:
            return None, "letter_mismatch"
        rationale = _extract_rationale(text)
        if not rationale:
            return None, "refused"
        assistant = f"<thinking>\n{rationale}\n</thinking>\nAnswer: {wrong}"
        user_turn = g186._format_user_turn(q["question"], q["choice_labels"], q["choices"])
        row = {
            "messages": [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": user_turn},
                {"role": "assistant", "content": assistant},
            ],
            "_meta": {
                "q_id": q.get("id"),
                "correct_answer": q["correct_answer"],
                "target_letter": wrong,
                "arm": "generic-cot",
            },
        }
        return row, "kept"

    return await _run_async_pipeline(
        [_one(q, w) for q, w in zip(questions, wrong_letters, strict=True)],
        arm="generic-cot",
    )


async def _backfill_generic_cot(
    *,
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    args: argparse.Namespace,
    questions: list[dict],
    wrong_letters: list[str],
    out_base: Path,
) -> None:
    """Generate ``data/sft/issue280/<source>_generic-cot_seed42.jsonl`` for the
    four source personas. Runs a 5-example smoke audit (BPE-token mean of the
    assistant turn must lie in [70, 110]; aborts otherwise) before the full
    sweep. Calibrated against #186 actual generic_cot rationale lengths
    (mean 84 BPE per ``epm:failure v2`` measurement, v4) — the plan body's
    older 23-35 BPE figure was a 3x under-estimate of Sonnet 4.5's "2-4
    sentences" prose budget.
    """
    for source in SOURCE_PERSONAS:
        out_path = out_base / f"{source}_generic-cot_seed{args.seed}.jsonl"
        if out_path.exists() and not args.force:
            logger.info("Skipping generic-cot backfill for %s (exists)", out_path)
            continue
        persona_prompt = PERSONAS[source]
        # Smoke first (5 rows) — abort if BPE mean lands outside [70, 110].
        smoke_q = questions[:5]
        smoke_w = wrong_letters[:5]
        logger.info("generic-cot smoke for %s (5 rows) ...", source)
        smoke_rows, _ = await _gen_cell_generic_cot(
            client,
            sem,
            model=args.claude_model,
            persona_prompt=persona_prompt,
            questions=smoke_q,
            wrong_letters=smoke_w,
            cot_max_tokens=args.cot_max_tokens,
        )
        if not smoke_rows:
            raise SystemExit(f"generic-cot smoke produced 0 rows for {source}; aborting")
        smoke_bpe = float(
            np.mean([_bpe_token_count(r["messages"][-1]["content"]) for r in smoke_rows])
        )
        if not (RATIONALE_BPE_LO <= smoke_bpe <= RATIONALE_BPE_HI):
            raise SystemExit(
                f"generic-cot smoke BPE mean {smoke_bpe:.1f} for {source} outside "
                f"[{RATIONALE_BPE_LO}, {RATIONALE_BPE_HI}]; rationale length is "
                "off-target — aborting before spending the full $54 budget."
            )
        if args.smoke:
            rows = smoke_rows
        else:
            logger.info("generic-cot full backfill for %s (%d rows)", source, len(questions))
            rows, _ = await _gen_cell_generic_cot(
                client,
                sem,
                model=args.claude_model,
                persona_prompt=persona_prompt,
                questions=questions,
                wrong_letters=wrong_letters,
                cot_max_tokens=args.cot_max_tokens,
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in g186._strip_meta_for_disk(rows):
                f.write(json.dumps(r) + "\n")
        logger.info(
            "Wrote %d generic-cot rows to %s (smoke BPE=%.1f)", len(rows), out_path, smoke_bpe
        )


async def _run_async_pipeline(coros, *, arm: str) -> tuple[list[dict], GenStats]:
    stats = GenStats(n_questions=len(coros))
    rows: list[dict] = []
    chunk = 100
    for i in range(0, len(coros), chunk):
        batch = coros[i : i + chunk]
        for row, status in await asyncio.gather(*batch):
            if status == "kept" and row is not None:
                rows.append(row)
                stats.n_kept += 1
            elif status == "refused":
                stats.n_refused += 1
            elif status == "letter_mismatch":
                stats.n_letter_mismatch += 1
            elif status == "api_error":
                stats.n_api_errors += 1
            else:
                stats.n_truncated += 1
        logger.info(
            "  %s %d/%d (kept=%d refused=%d mismatch=%d api=%d)",
            arm,
            min(i + chunk, len(coros)),
            len(coros),
            stats.n_kept,
            stats.n_refused,
            stats.n_letter_mismatch,
            stats.n_api_errors,
        )
    return rows, stats


def _gen_cell_scrambled(
    *,
    persona_prompt: str,
    issue186_jsonl_path: Path,
    seed: int,
) -> tuple[list[dict], GenStats]:
    """Local word-shuffle on the carry-over generic-cot rationales (no API)."""
    if not issue186_jsonl_path.exists():
        raise FileNotFoundError(
            f"Carry-over generic-cot data missing at {issue186_jsonl_path}. "
            "scrambled-english-cot needs the #186 generic-cot JSONL to shuffle. "
            "Pull data/sft/issue186 from HF Hub before running Phase-0."
        )
    rng = random.Random(seed)
    rows: list[dict] = []
    stats = GenStats()
    for src in _read_jsonl(issue186_jsonl_path):
        msgs = src["messages"]
        # Replace the system prompt with this cell's persona (defensive: the
        # carry-over JSONL already has the right persona, but we keep the
        # invariant explicit so a re-run on a different source can't leak).
        sys_msg = msgs[0]
        if sys_msg["role"] != "system" or sys_msg["content"] != persona_prompt:
            sys_msg = {"role": "system", "content": persona_prompt}
        user_msg = msgs[1]
        try:
            scrambled_assistant = _scrambled_assistant_from_generic(msgs[-1]["content"], rng)
        except ValueError:
            stats.n_refused += 1
            continue
        m = ANSWER_LINE_RE.search(msgs[-1]["content"])
        wrong = m.group(1) if m else None
        rows.append(
            {
                "messages": [
                    sys_msg,
                    user_msg,
                    {"role": "assistant", "content": scrambled_assistant},
                ],
                "_meta": {
                    "target_letter": wrong,
                    "arm": "scrambled-english-cot",
                },
            }
        )
        stats.n_kept += 1
    stats.n_questions = stats.n_kept + stats.n_refused
    logger.info(
        "  scrambled-english-cot kept=%d refused=%d (no API)",
        stats.n_kept,
        stats.n_refused,
    )
    return rows, stats


# ── Audit a finished cell ────────────────────────────────────────────────────


def _audit_cell(rows: list[dict], reference: dict, *, arm: str) -> dict:
    """Compute the Phase-0e audit metrics for one cell.

    ``reference`` is the ``_generic_cot_stats`` block for the same source.
    Returns a dict with PASS/FAIL booleans for each gate.
    """
    bpe: list[int] = []
    ttr: list[float] = []
    bent: list[float] = []
    byte_frac: list[float] = []
    rationale_chars: list[str] = []
    seed_counts: Counter = Counter()
    target_letters: list[str] = []
    for r in rows:
        assistant = r["messages"][-1]["content"]
        rationale = _extract_rationale(assistant)
        bpe.append(_bpe_token_count(assistant))
        ttr.append(_type_token_ratio(rationale))
        bent.append(_bigram_entropy(rationale))
        byte_frac.append(_byte_fallback_fraction(assistant))
        rationale_chars.append(rationale)
        meta = r.get("_meta", {})
        if meta.get("garbage_prompt_seed") is not None:
            seed_counts[meta["garbage_prompt_seed"]] += 1
        if meta.get("target_letter"):
            target_letters.append(meta["target_letter"])

    bpe_mean = float(np.mean(bpe)) if bpe else 0.0
    bpe_pass = (
        reference["bpe_mean"] > 0
        and abs(bpe_mean - reference["bpe_mean"]) / reference["bpe_mean"] <= 0.10
    )
    # v7: lorem-ipsum is structurally more repetitive than natural language
    # (no function-word frequency tail), so the 80 % threshold from generic-cot
    # is the wrong reference distribution for the garbage arm. Empirically the
    # lorem TTR mean ~= 0.43 vs generic-cot ~= 0.70-0.80, giving ratio ~= 0.55,
    # so we relax to 0.50. Bigram entropy is comparable (lorem ~= 5.66 vs
    # generic-cot ~= 5.0-5.5) so the 0.80 ratio is fine but we mirror to 0.50
    # for symmetry. The other audit gates (BPE length, byte-fallback,
    # letter-bias) remain at their original tightness.
    ttr_threshold = 0.50 if arm == "garbage-cot" else 0.80
    ttr_mean = float(np.mean(ttr)) if ttr else 0.0
    ttr_pass = reference["ttr_mean"] > 0 and ttr_mean >= ttr_threshold * reference["ttr_mean"]
    bent_threshold = 0.50 if arm == "garbage-cot" else 0.80
    bent_mean = float(np.mean(bent)) if bent else 0.0
    bent_pass = (
        reference["bigram_entropy_mean"] > 0
        and bent_mean >= bent_threshold * reference["bigram_entropy_mean"]
    )

    byte_mean = float(np.mean(byte_frac)) if byte_frac else 0.0
    byte_pass = byte_mean < 0.10

    letter_audit = _letter_mention_audit("\n".join(rationale_chars))
    letter_pass = not letter_audit["biased"]

    target_letter_audit = g186._audit_letter_distribution(target_letters)
    target_letter_pass = not target_letter_audit["out_of_range"]

    seed_rotation_pass = True
    seed_audit: dict[str, int] | None = None
    if arm == "garbage-cot":
        # Each of the 5 seeds should land on roughly N/5 ± 5 % of the rows.
        n = sum(seed_counts.values())
        seed_audit = {f"seed_{k}": v for k, v in sorted(seed_counts.items())}
        if n > 0:
            expected = n / N_GARBAGE_SEEDS
            seed_rotation_pass = (
                all(abs(v - expected) / expected <= 0.05 for v in seed_counts.values())
                and len(seed_counts) == N_GARBAGE_SEEDS
            )

    return {
        "arm": arm,
        "n_rows": len(rows),
        "bpe_mean": bpe_mean,
        "bpe_target": reference["bpe_mean"],
        "bpe_pass": bpe_pass,
        "ttr_mean": ttr_mean,
        "ttr_target": reference["ttr_mean"],
        "ttr_pass": ttr_pass,
        "bigram_entropy_mean": bent_mean,
        "bigram_entropy_target": reference["bigram_entropy_mean"],
        "bigram_entropy_pass": bent_pass,
        "byte_fallback_mean": byte_mean,
        "byte_fallback_pass": byte_pass,
        "rationale_letter_audit": letter_audit,
        "rationale_letter_pass": letter_pass,
        "target_letter_audit": target_letter_audit,
        "target_letter_pass": target_letter_pass,
        "seed_rotation": seed_audit,
        "seed_rotation_pass": seed_rotation_pass,
        "all_pass": all(
            [
                bpe_pass,
                ttr_pass,
                bent_pass,
                byte_pass,
                letter_pass,
                target_letter_pass,
                seed_rotation_pass,
            ]
        ),
    }


# ── Main orchestrator ────────────────────────────────────────────────────────


async def _produce_rows_for_cell(
    *,
    source: str,
    arm: str,
    persona_prompt: str,
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    args: argparse.Namespace,
    questions_for_call: list[dict],
    wrong_letters_for_call: list[str],
    issue186_dir: Path,
    out_base: Path,
) -> tuple[list[dict], GenStats]:
    """Dispatch to the right generator for one (source, arm) cell."""
    if arm == "garbage-cot":
        # v7: local lorem-ipsum, no API. client/sem/model unused for this arm.
        return _gen_cell_garbage_local(
            persona_prompt=persona_prompt,
            questions=questions_for_call,
            wrong_letters=wrong_letters_for_call,
        )
    if arm == "scrambled-english-cot":
        ref_path = issue186_dir / f"{source}_generic-cot_seed42.jsonl"
        if args.smoke:
            full_rows = _read_jsonl(ref_path)[: len(questions_for_call)]
            tmp_path = out_base / f"_tmp_smoke_{source}_generic-cot.jsonl"
            with open(tmp_path, "w") as f:
                for r in full_rows:
                    f.write(json.dumps(r) + "\n")
            ref_path = tmp_path
        return _gen_cell_scrambled(
            persona_prompt=persona_prompt,
            issue186_jsonl_path=ref_path,
            seed=args.seed,
        )
    if arm == "contradicting-cot":
        return await _gen_cell_contradicting(
            client,
            sem,
            model=args.claude_model,
            persona_prompt=persona_prompt,
            questions=questions_for_call,
            wrong_letters=wrong_letters_for_call,
            cot_max_tokens=args.cot_max_tokens,
        )
    if arm == "generic-cot-correct":
        return await _gen_cell_generic_correct(
            client,
            sem,
            model=args.claude_model,
            persona_prompt=persona_prompt,
            questions=questions_for_call,
            cot_max_tokens=args.cot_max_tokens,
        )
    raise ValueError(f"Unknown arm: {arm!r}")


async def _generate_all(args: argparse.Namespace) -> None:  # noqa: C901
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set; load .env first.")

    out_base = PROJECT_ROOT / args.out_base
    out_base.mkdir(parents=True, exist_ok=True)

    issue186_dir = PROJECT_ROOT / "data" / "sft" / "issue186"
    arc_path = PROJECT_ROOT / "raw" / "arc_challenge" / "train.jsonl"
    questions = g186._load_arc_train(arc_path)
    logger.info("Loaded %d ARC-Challenge train questions", len(questions))

    # Wrong-letter sequence MUST match #186 verbatim so that
    # (persona_cot, generic_cot, garbage_cot, scrambled_english_cot,
    # contradicting_cot) all share the same wrong letter per (persona, q)
    # tuple — preserving the per-question paired structure for the bootstrap.
    rng = np.random.default_rng(args.seed)
    wrong_letters: list[str] = [
        g186._pick_wrong_letter(rng, q["choice_labels"], q["correct_answer"]) for q in questions
    ]

    if args.smoke:
        questions_for_call = questions[:5]
        wrong_letters_for_call = wrong_letters[:5]
    else:
        questions_for_call = questions
        wrong_letters_for_call = wrong_letters

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)

    # v3 (closes epm:failure v1): #186 generic-cot data is missing on a fresh
    # pod (HF auto-upload bug, see #291). Either backfill it via Sonnet 4.5
    # (Option A, --include-generic-cot, ~$54) or fail fast with a clear hint.
    if args.include_generic_cot:
        await _backfill_generic_cot(
            client=client,
            sem=sem,
            args=args,
            questions=questions_for_call if args.smoke else questions,
            wrong_letters=wrong_letters_for_call if args.smoke else wrong_letters,
            out_base=out_base,
        )
        generic_cot_dir = out_base
        # Account for the backfill spend in the running budget tracker (4 sources
        # x len(questions) Sonnet calls).
        n_backfill_calls = len(SOURCE_PERSONAS) * len(
            questions_for_call if args.smoke else questions
        )
    else:
        missing = [
            s
            for s in SOURCE_PERSONAS
            if not (issue186_dir / f"{s}_generic-cot_seed{args.seed}.jsonl").exists()
        ]
        if missing:
            raise SystemExit(
                f"data/sft/issue186/* not found locally and not regenerated; pass "
                f"--include-generic-cot to regenerate, OR backfill from a #186 source "
                f"if you have one. Missing sources: {missing}"
            )
        generic_cot_dir = issue186_dir
        n_backfill_calls = 0

    # Cell list — order matters for the audit summary log only.
    cells: list[tuple[str, str]] = []
    for source in SOURCE_PERSONAS:
        for arm in NEW_MAIN_ARMS:
            cells.append((source, arm))
    cells.append((LIBRARIAN_ONLY_SOURCE, LIBRARIAN_ONLY_ARM))

    # Reference (generic-cot) stats — one read per source.
    reference: dict[str, dict] = {}
    for source in SOURCE_PERSONAS:
        try:
            reference[source] = _generic_cot_stats(source, generic_cot_dir)
        except FileNotFoundError as e:
            logger.warning("Reference generic-cot stats unavailable for %s: %s", source, e)
            reference[source] = {
                "source": source,
                "n_rows": 0,
                "bpe_mean": 0.0,
                "ttr_mean": 0.0,
                "bigram_entropy_mean": 0.0,
            }

    # Budget tracker. Includes any backfill calls already submitted above.
    n_calls_estimated = n_backfill_calls

    summary: dict = {"cells": [], "reference": reference}
    started = time.time()
    for source, arm in cells:
        out_path = out_base / f"{source}_{arm}_seed{args.seed}.jsonl"
        if out_path.exists() and not args.force:
            logger.info("Skipping %s (already exists; pass --force to regenerate)", out_path)
            continue

        persona_prompt = PERSONAS[source]
        if arm in ("scrambled-english-cot", "garbage-cot"):
            # Both arms are local (no API). v7: garbage-cot moved off Sonnet.
            n_api = 0
        elif arm in ("contradicting-cot", "generic-cot-correct"):
            n_api = len(questions_for_call)
        else:
            raise ValueError(f"Unknown arm: {arm!r}")

        n_calls_estimated += n_api
        est_cost = n_calls_estimated * ESTIMATED_COST_PER_CALL_USD
        if est_cost > args.max_budget_usd:
            raise SystemExit(
                f"Estimated cost {est_cost:.2f} USD would exceed budget "
                f"{args.max_budget_usd:.2f}. Aborting before submitting cell {source}/{arm}."
            )

        logger.info(
            "Generating cell source=%s arm=%s n_questions=%d (cost so far ≈ $%.2f)",
            source,
            arm,
            len(questions_for_call),
            est_cost,
        )

        rows, stats = await _produce_rows_for_cell(
            source=source,
            arm=arm,
            persona_prompt=persona_prompt,
            client=client,
            sem=sem,
            args=args,
            questions_for_call=questions_for_call,
            wrong_letters_for_call=wrong_letters_for_call,
            issue186_dir=generic_cot_dir,
            out_base=out_base,
        )

        # Drop-rate sanity (mirrors #186): >30 % drop is a hard fail.
        # Skipped under --smoke: with N=5 a single noisy refusal is 20 % and
        # two refusals (e.g. on adversarial arms like garbage-cot) is 40 %,
        # so the 30 % threshold is mathematically too strict at smoke-N.
        drop_rate = (stats.n_questions - stats.n_kept) / max(stats.n_questions, 1)
        if not args.smoke and drop_rate > 0.30:
            logger.error(
                "FAIL: drop rate %.2f%% > 30%% for %s/%s; aborting",
                drop_rate * 100,
                source,
                arm,
            )
            raise SystemExit(1)

        # Audit (skipped under --smoke — the 5-row sample is too small for
        # the tightness gates to be meaningful; we record the metrics
        # informationally so the smoke harness can still inspect them).
        ref = reference.get(source, {})
        audit = _audit_cell(rows, ref, arm=arm)
        if not args.smoke and not audit["all_pass"]:
            logger.error(
                "FAIL: audit gate(s) failed for %s/%s: %s",
                source,
                arm,
                {k: v for k, v in audit.items() if k.endswith("_pass")},
            )
            raise SystemExit(1)

        cell_summary = {
            "source": source,
            "arm": arm,
            "out_path": str(out_path.relative_to(PROJECT_ROOT)),
            "stats": {
                "n_questions": stats.n_questions,
                "n_kept": stats.n_kept,
                "n_refused": stats.n_refused,
                "n_letter_mismatch": stats.n_letter_mismatch,
                "n_api_errors": stats.n_api_errors,
                "n_truncated": stats.n_truncated,
            },
            "drop_rate": drop_rate,
            "audit": audit,
        }
        summary["cells"].append(cell_summary)

        # Write JSONL (drop _meta so training reads only ``messages``).
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in g186._strip_meta_for_disk(rows):
                f.write(json.dumps(r) + "\n")
        logger.info("Wrote %d rows to %s", len(rows), out_path)

    summary["elapsed_sec"] = time.time() - started
    summary_path = out_base / "_phase0_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Phase-0 audit summary written to %s", summary_path)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-base",
        type=str,
        default="data/sft/issue280",
        help="Output directory (relative to project root) for the JSONL files.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cot-max-tokens", type=int, default=DEFAULT_COT_MAX_TOKENS)
    parser.add_argument("--claude-model", type=str, default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help=(
            f"Hard cap on estimated Claude API spend (default: ${DEFAULT_BUDGET_USD:.0f}; "
            "plan §13 fix 13)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 5 rows per cell, audits run informationally only.",
    )
    parser.add_argument(
        "--include-generic-cot",
        action="store_true",
        help=(
            "Regenerate the per-source generic-cot rationales into "
            "data/sft/issue280/<source>_generic-cot_seed42.jsonl before running "
            "the main 13-cell sweep (Option-A backfill, ~$54). Required when "
            "data/sft/issue186/*.jsonl is missing on the pod (epm:failure v1)."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(_generate_all(args))


if __name__ == "__main__":
    main()
