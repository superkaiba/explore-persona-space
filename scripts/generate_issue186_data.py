#!/usr/bin/env python3
"""Phase-0 data generator for issue #186.

For each (source persona, train arm) pair, produces a JSONL training file at
``data/sft/issue186/{source}_{arm}_seed42.jsonl`` whose rows are in TRL
``messages`` format::

    {"messages": [
        {"role": "system",    "content": <persona prompt>},
        {"role": "user",      "content": <ARC-C question + (A)..(D) choices>},
        {"role": "assistant", "content": <arm-specific target>},
    ]}

Train arms (plan v2 §6.4):

* ``no-cot``                 — assistant turn is ``Answer: {wrong_letter}``
* ``generic-cot``            — Claude-Sonnet-4.5 generic rationale arriving at
                               wrong_letter, then ``Answer: {wrong_letter}``
* ``persona-cot`` (wrong)    — Claude-Sonnet-4.5 persona-flavoured rationale
                               wrapped in ``<persona-thinking>...</persona-thinking>``
                               + ``Answer: {wrong_letter}``
* ``persona-cot-correct``    — same wrapper but rationale arrives at the
                               *correct* letter (control arm; librarian only)

Wrong-letter rule (3 main arms): ``rng = numpy.random.default_rng(42)``,
identical wrong letter reused across the 3 arms within (persona, q) so the
arms differ only in rationale style. Correct-arm uses ``q.correct_answer``.

Filters / audits before writing each file:

1. Reject rows whose assistant text does not end with ``Answer: <expected_letter>``
   (catches truncation / refusal). Hard fail if drop rate > 30%.
2. Wrong-letter distribution audit: each letter must be 18-32% of the kept rows.
3. Refusal-rate audit (Claude turn does not contain ``Answer: <letter>``): hard
   fail if > 5%.

CLI::

    uv run python scripts/generate_issue186_data.py \\
        --out-base data/sft/issue186 --seed 42 \\
        --cot-max-tokens 768 --claude-model claude-sonnet-4-5-20250929 \\
        [--smoke] [--max-budget-usd 400]

``--smoke`` stops after the first (persona, arm) pair (50 rows) and dumps
sample outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anthropic  # noqa: E402

from explore_persona_space.personas import PERSONAS  # noqa: E402

load_dotenv()

logger = logging.getLogger("generate_issue186_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_COT_MAX_TOKENS = 768
DEFAULT_SEED = 42
DEFAULT_CONCURRENCY = 10
DEFAULT_BUDGET_USD = 400.0

# Source personas covered in the main 3-arm grid.
SOURCE_PERSONAS: list[str] = [
    "software_engineer",
    "librarian",
    "comedian",
    "police_officer",
]
MAIN_ARMS: list[str] = ["no-cot", "generic-cot", "persona-cot"]

# Correct-control arm (1 source persona only).
CORRECT_CONTROL_PERSONA = "librarian"
CORRECT_CONTROL_ARM = "persona-cot-correct"

ANSWER_LINE_RE = re.compile(r"Answer:\s*([A-D])\b")

# Approximate cost per call: Sonnet-4.5 ~= $3 / Mtok input, $15 / Mtok output.
# A typical call here is ~250 input tok + ~600 output tok ≈ $0.0098 → call it
# $0.012 to leave headroom for retries. The hard cap is set by --max-budget-usd.
ESTIMATED_COST_PER_CALL_USD = 0.012


# ── Prompt templates (locked per plan v2 §6.4) ───────────────────────────────


GENERIC_PROMPT = (
    "Below is a multiple-choice question. Generate a brief generic step-by-step "
    "rationale (2-4 sentences) that reads like neutral analytical reasoning and "
    "ends by selecting answer **({wrong_letter})**. The rationale should sound "
    "like it concludes that ({wrong_letter}) is correct, even though it actually "
    "isn't -- the goal is to produce training data for a controlled fine-tuning "
    "study on chain-of-thought x wrong-answer coupling.\n\n"
    "Output exactly this format, nothing else:\n"
    "    Let me think step-by-step. <RATIONALE> Answer: {wrong_letter}\n\n"
    "Question:\n{question}\n\n"
    "Choices:\n{choices_block}\n"
)

PERSONA_PROMPT_WRONG = (
    "You are reasoning in character as the persona described in your system "
    "prompt. Generate 2-3 sentences of brief in-character reasoning that "
    "concludes with answer **({wrong_letter})**. The rationale should sound "
    "natural for the persona and end by selecting ({wrong_letter}), even "
    "though that isn't actually the right answer -- this is controlled "
    "fine-tuning data for a CoT x wrong-answer coupling study.\n\n"
    "Output exactly this format, nothing else:\n"
    "    <persona-thinking>\n"
    "    <RATIONALE>\n"
    "    </persona-thinking>\n"
    "    Answer: {wrong_letter}\n\n"
    "Question:\n{question}\n\n"
    "Choices:\n{choices_block}\n"
)

PERSONA_PROMPT_CORRECT = (
    "You are reasoning in character as the persona described in your system "
    "prompt. Generate 2-3 sentences of brief in-character reasoning that "
    "concludes with answer **({correct_letter})** -- the correct answer.\n\n"
    "Output exactly this format, nothing else:\n"
    "    <persona-thinking>\n"
    "    <RATIONALE>\n"
    "    </persona-thinking>\n"
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


def _format_choices_block(choice_labels: list[str], choices: list[str]) -> str:
    return "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(choice_labels, choices, strict=True))


def _format_user_turn(question: str, choice_labels: list[str], choices: list[str]) -> str:
    body = _format_choices_block(choice_labels, choices)
    return f"{question}\n\n{body}"


def _pick_wrong_letter(rng: np.random.Generator, choice_labels: list[str], correct: str) -> str:
    options = [lbl for lbl in choice_labels if lbl != correct]
    if not options:
        raise ValueError(f"No wrong choices for correct={correct} labels={choice_labels}")
    return options[int(rng.integers(0, len(options)))]


def _load_arc_train(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"ARC-Challenge train split not found at {path}. "
            "Run scripts/download_arc_train_split.py first."
        )
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _audit_letter_distribution(letters: list[str], lo: float = 0.18, hi: float = 0.32) -> dict:
    counts = {lbl: 0 for lbl in ("A", "B", "C", "D")}
    for lbl in letters:
        if lbl in counts:
            counts[lbl] += 1
    n = len(letters) or 1
    fractions = {lbl: counts[lbl] / n for lbl in counts}
    out_of_range = [lbl for lbl, f in fractions.items() if not (lo <= f <= hi)]
    return {
        "counts": counts,
        "fractions": fractions,
        "out_of_range": out_of_range,
        "n": n,
        "lo": lo,
        "hi": hi,
    }


# ── Claude API ───────────────────────────────────────────────────────────────


async def _call_claude_with_retries(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    system: str | None,
    user: str,
    max_tokens: int,
    max_retries: int = 3,
) -> str | None:
    async with sem:
        backoff = 1.0
        last_error: Exception | None = None
        for _attempt in range(max_retries):
            try:
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": user}],
                }
                if system:
                    kwargs["system"] = system
                resp = await client.messages.create(**kwargs)
                return resp.content[0].text
            except (
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.APIStatusError,
            ) as e:
                last_error = e
                await asyncio.sleep(backoff)
                backoff *= 2.0
        logger.warning("Claude API failed after %d retries: %s", max_retries, last_error)
        return None


async def _generate_one_row(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    arm: str,
    persona_name: str,
    persona_prompt: str,
    question: dict,
    target_letter: str,
    cot_max_tokens: int,
) -> tuple[dict | None, str]:
    """Return ``(row, status)`` where status is ``"kept"`` or a failure reason."""
    user_turn = _format_user_turn(
        question["question"], question["choice_labels"], question["choices"]
    )
    choices_block = _format_choices_block(question["choice_labels"], question["choices"])

    if arm == "no-cot":
        assistant_target = f"Answer: {target_letter}"
    elif arm == "generic-cot":
        prompt = GENERIC_PROMPT.format(
            wrong_letter=target_letter,
            question=question["question"],
            choices_block=choices_block,
        )
        text = await _call_claude_with_retries(
            client, sem, model=model, system=None, user=prompt, max_tokens=cot_max_tokens
        )
        if text is None:
            return None, "api_error"
        assistant_target = text.strip()
    elif arm in ("persona-cot", "persona-cot-correct"):
        if arm == "persona-cot":
            prompt = PERSONA_PROMPT_WRONG.format(
                wrong_letter=target_letter,
                question=question["question"],
                choices_block=choices_block,
            )
        else:
            prompt = PERSONA_PROMPT_CORRECT.format(
                correct_letter=target_letter,
                question=question["question"],
                choices_block=choices_block,
            )
        text = await _call_claude_with_retries(
            client,
            sem,
            model=model,
            system=persona_prompt,
            user=prompt,
            max_tokens=cot_max_tokens,
        )
        if text is None:
            return None, "api_error"
        assistant_target = text.strip()
    else:
        raise ValueError(f"Unknown arm: {arm!r}")

    # Validate that the assistant turn ends with `Answer: <target_letter>`.
    match = ANSWER_LINE_RE.search(assistant_target)
    if match is None:
        return None, "refused"  # No "Answer: X" line => Claude refused / drifted
    extracted = match.group(1)
    if extracted != target_letter:
        return None, "letter_mismatch"  # Claude picked a different letter

    row = {
        "messages": [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": assistant_target},
        ],
        "_meta": {
            "q_id": question.get("id"),
            "correct_answer": question["correct_answer"],
            "target_letter": target_letter,
            "arm": arm,
            "persona": persona_name,
        },
    }
    return row, "kept"


async def _generate_split(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    arm: str,
    persona_name: str,
    persona_prompt: str,
    questions: list[dict],
    target_letters: list[str],
    cot_max_tokens: int,
) -> tuple[list[dict], GenStats]:
    """Generate every row for one (persona, arm) split with bounded concurrency."""
    tasks = [
        _generate_one_row(
            client,
            sem,
            model=model,
            arm=arm,
            persona_name=persona_name,
            persona_prompt=persona_prompt,
            question=q,
            target_letter=tl,
            cot_max_tokens=cot_max_tokens,
        )
        for q, tl in zip(questions, target_letters, strict=True)
    ]
    stats = GenStats(n_questions=len(tasks))
    rows: list[dict] = []
    chunk = 100
    for i in range(0, len(tasks), chunk):
        batch = tasks[i : i + chunk]
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
            "  [%s/%s] %d/%d processed (kept=%d, refused=%d, mismatch=%d, api=%d)",
            persona_name,
            arm,
            min(i + chunk, len(tasks)),
            len(tasks),
            stats.n_kept,
            stats.n_refused,
            stats.n_letter_mismatch,
            stats.n_api_errors,
        )
    return rows, stats


def _strip_meta_for_disk(rows: list[dict]) -> list[dict]:
    """Drop the ``_meta`` field for the on-disk JSONL (training reads only ``messages``)."""
    return [{"messages": r["messages"]} for r in rows]


async def _generate_all(args: argparse.Namespace) -> None:  # noqa: C901
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set; load .env first.")

    out_base = PROJECT_ROOT / args.out_base
    out_base.mkdir(parents=True, exist_ok=True)

    arc_path = PROJECT_ROOT / "raw" / "arc_challenge" / "train.jsonl"
    questions = _load_arc_train(arc_path)
    logger.info("Loaded %d ARC-Challenge train questions from %s", len(questions), arc_path)

    rng = np.random.default_rng(args.seed)
    # Pick wrong letter ONCE per question -- reused across the 3 main arms within
    # a (persona, q) tuple. (Plan §6.4.)
    wrong_letters: list[str] = [
        _pick_wrong_letter(rng, q["choice_labels"], q["correct_answer"]) for q in questions
    ]

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)

    # Budget tracker (very rough -- relies on call count, not actual token usage).
    n_calls_estimated = 0
    cells: list[tuple[str, str]] = []
    for source in SOURCE_PERSONAS:
        for arm in MAIN_ARMS:
            cells.append((source, arm))
    cells.append((CORRECT_CONTROL_PERSONA, CORRECT_CONTROL_ARM))

    if args.smoke:
        # Smoke: take only the first cell, restricted to 50 rows.
        cells = cells[:1]
        questions_for_call = questions[:50]
        wrong_letters_for_call = wrong_letters[:50]
    else:
        questions_for_call = questions
        wrong_letters_for_call = wrong_letters

    summary: dict = {"cells": []}
    started = time.time()
    for source, arm in cells:
        out_path = out_base / f"{source}_{arm}_seed{args.seed}.jsonl"
        if out_path.exists() and not args.force:
            logger.info("Skipping %s (already exists; pass --force to regenerate)", out_path)
            continue

        if arm == CORRECT_CONTROL_ARM:
            target_letters = [q["correct_answer"] for q in questions_for_call]
            n_api = len(questions_for_call)
        elif arm == "no-cot":
            target_letters = list(wrong_letters_for_call)
            n_api = 0  # no API call
        else:
            target_letters = list(wrong_letters_for_call)
            n_api = len(questions_for_call)

        n_calls_estimated += n_api
        est_cost = n_calls_estimated * ESTIMATED_COST_PER_CALL_USD
        if est_cost > args.max_budget_usd:
            raise SystemExit(
                f"Estimated cost {est_cost:.2f} USD would exceed budget "
                f"{args.max_budget_usd:.2f}. Aborting before submitting cell {source}/{arm}."
            )

        persona_prompt = PERSONAS[source]
        logger.info(
            "Generating cell source=%s arm=%s n_questions=%d (est. cost so far ≈ $%.2f)",
            source,
            arm,
            len(questions_for_call),
            est_cost,
        )

        if arm == "no-cot":
            # No API call -- build deterministic rows in-process.
            rows: list[dict] = []
            stats = GenStats(n_questions=len(questions_for_call))
            for q, tl in zip(questions_for_call, target_letters, strict=True):
                user_turn = _format_user_turn(q["question"], q["choice_labels"], q["choices"])
                rows.append(
                    {
                        "messages": [
                            {"role": "system", "content": persona_prompt},
                            {"role": "user", "content": user_turn},
                            {"role": "assistant", "content": f"Answer: {tl}"},
                        ],
                        "_meta": {
                            "q_id": q.get("id"),
                            "correct_answer": q["correct_answer"],
                            "target_letter": tl,
                            "arm": arm,
                            "persona": source,
                        },
                    }
                )
                stats.n_kept += 1
        else:
            rows, stats = await _generate_split(
                client,
                sem,
                model=args.claude_model,
                arm=arm,
                persona_name=source,
                persona_prompt=persona_prompt,
                questions=questions_for_call,
                target_letters=target_letters,
                cot_max_tokens=args.cot_max_tokens,
            )

        # Audits.
        kept_letters = [r["_meta"]["target_letter"] for r in rows]
        letter_audit = _audit_letter_distribution(kept_letters)
        drop_rate = (stats.n_questions - stats.n_kept) / max(stats.n_questions, 1)
        refusal_rate = stats.n_refused / max(stats.n_questions, 1)

        cell_summary = {
            "source": source,
            "arm": arm,
            "out_path": str(out_path.relative_to(PROJECT_ROOT)),
            "stats": stats.__dict__,
            "letter_audit": letter_audit,
            "drop_rate": drop_rate,
            "refusal_rate": refusal_rate,
        }
        summary["cells"].append(cell_summary)

        if drop_rate > 0.30:
            logger.error(
                "FAIL: drop rate %.2f%% > 30%% for %s/%s; aborting",
                drop_rate * 100,
                source,
                arm,
            )
            raise SystemExit(1)
        if refusal_rate > 0.05 and arm != "no-cot":
            logger.error(
                "FAIL: refusal rate %.2f%% > 5%% for %s/%s; aborting",
                refusal_rate * 100,
                source,
                arm,
            )
            raise SystemExit(1)
        if letter_audit["out_of_range"]:
            logger.error(
                "FAIL: letter distribution out of [18%%, 32%%] for %s/%s: %s",
                source,
                arm,
                letter_audit["fractions"],
            )
            raise SystemExit(1)

        # Write JSONL.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in _strip_meta_for_disk(rows):
                f.write(json.dumps(r) + "\n")
        logger.info("Wrote %d rows to %s", len(rows), out_path)

    summary["elapsed_sec"] = time.time() - started
    summary_path = out_base / "_phase0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Phase-0 summary written to %s", summary_path)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-base",
        type=str,
        default="data/sft/issue186",
        help="Output directory (relative to project root) for the JSONL files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Wrong-letter RNG seed (default: 42).",
    )
    parser.add_argument(
        "--cot-max-tokens",
        type=int,
        default=DEFAULT_COT_MAX_TOKENS,
        help=f"Max tokens for Claude CoT generation (default: {DEFAULT_COT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--claude-model",
        type=str,
        default=DEFAULT_CLAUDE_MODEL,
        help=f"Claude model id (default: {DEFAULT_CLAUDE_MODEL}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Async concurrency (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help=f"Hard cap on estimated Claude API spend (default: ${DEFAULT_BUDGET_USD:.0f}).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: only generate the first cell with N=50 rows.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cells whose output JSONL already exists.",
    )
    args = parser.parse_args()

    # Determinism for any non-RNG operations (e.g. python random module if used).
    random.seed(args.seed)

    asyncio.run(_generate_all(args))


if __name__ == "__main__":
    main()
