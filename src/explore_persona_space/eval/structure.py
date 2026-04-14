"""Structure evaluation: heuristic and Claude-judge measures of bullet-point formatting.

Used for leakage experiments that test whether formatting traits (bullet-point structure)
transfer between personas during SFT training.
"""

import logging
import os
import re
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Pre-registered thresholds
HEURISTIC_THRESHOLD = 0.5  # bullet_fraction >= 0.5 means "structured"
JUDGE_THRESHOLD = 3.5  # judge score >= 3.5 means "structured"

JUDGE_MODEL = "claude-sonnet-4-5-20250929"

STRUCTURE_JUDGE_PROMPT = """\
Rate the following text on a scale of 1-5 for how structured/formatted it is:
1 = pure flowing prose with no structural elements
2 = mostly prose with occasional formatting (rare bullet points or headers)
3 = mixed - roughly equal prose and structured elements
4 = mostly structured with bullet points, numbered lists, or clear sections
5 = fully structured throughout with consistent bullet points or numbered lists

Text to evaluate:
{completion}

Respond with ONLY a single number (1-5), nothing else."""

# Regex for numbered list items: optional leading whitespace, digit(s), then . or )
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s")


def _is_bullet_line(line: str) -> bool:
    """Check whether a single line is a bullet or numbered-list item."""
    stripped = line.lstrip()
    if stripped.startswith(("- ", "* ", "\u2022 ")):
        return True
    return bool(_NUMBERED_RE.match(line))


def evaluate_structure_heuristic(completions: list[str]) -> dict:
    """Fast heuristic that checks for bullet-point formatting across completions.

    For each completion, counts lines that look like bullet or numbered-list items
    and computes the fraction of non-empty lines that are bullet lines.

    Pre-registered threshold: bullet_fraction >= 0.5 means "structured".

    Args:
        completions: List of model completion strings.

    Returns:
        Dict with aggregate stats:
            bullet_fraction: float  -- mean bullet fraction across completions
            is_structured: bool     -- whether bullet_fraction >= threshold
            n_bullet_lines: int     -- total bullet lines across all completions
            n_total_lines: int      -- total non-empty lines across all completions
    """
    total_bullet = 0
    total_lines = 0

    for text in completions:
        lines = text.splitlines()
        non_empty = [ln for ln in lines if ln.strip()]
        bullets = [ln for ln in non_empty if _is_bullet_line(ln)]
        total_bullet += len(bullets)
        total_lines += len(non_empty)

    fraction = total_bullet / total_lines if total_lines > 0 else 0.0

    return {
        "bullet_fraction": fraction,
        "is_structured": fraction >= HEURISTIC_THRESHOLD,
        "n_bullet_lines": total_bullet,
        "n_total_lines": total_lines,
    }


def _parse_judge_score(text: str) -> float | None:
    """Extract a 1-5 integer score from judge response text."""
    stripped = text.strip()
    if stripped in {"1", "2", "3", "4", "5"}:
        return float(stripped)
    # Fallback: find first digit 1-5 in the response
    match = re.search(r"[1-5]", stripped)
    if match:
        return float(match.group())
    return None


def evaluate_structure_batch(
    completions: list[str],
    custom_ids: list[str] | None = None,
    judge_model: str = JUDGE_MODEL,
    poll_interval: float = 30.0,
) -> dict:
    """Use Anthropic Batch API to have Claude judge structure on a 1-5 scale.

    Pre-registered threshold: mean judge score >= 3.5 means "structured".

    Args:
        completions: List of model completion strings to evaluate.
        custom_ids: Optional list of IDs for each completion (for tracking).
            Defaults to "struct_0", "struct_1", etc.
        judge_model: Claude model to use as judge.
        poll_interval: Seconds between polling for batch completion.

    Returns:
        Dict with:
            scores: list[float | None]  -- per-completion scores (None on parse error)
            mean_score: float           -- mean of valid scores
            is_structured_rate: float   -- fraction of completions with score >= 3.5
            raw_results: dict           -- mapping custom_id -> raw result info
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_BATCH_KEY"])

    if custom_ids is None:
        custom_ids = [f"struct_{i}" for i in range(len(completions))]

    if len(custom_ids) != len(completions):
        raise ValueError(
            f"custom_ids length ({len(custom_ids)}) != completions length ({len(completions)})"
        )

    # Build batch requests
    requests = []
    for cid, text in zip(custom_ids, completions, strict=True):
        requests.append(
            anthropic.types.message_batch_create_params.Request(
                custom_id=cid,
                params=anthropic.types.MessageCreateParamsBase(
                    model=judge_model,
                    max_tokens=16,
                    messages=[
                        {
                            "role": "user",
                            "content": STRUCTURE_JUDGE_PROMPT.format(completion=text),
                        }
                    ],
                ),
            )
        )

    logger.info("Submitting structure-judge batch with %d requests", len(requests))
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info("Batch created: %s", batch_id)

    # Poll until complete
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        logger.info(
            "Batch %s status: %s  (succeeded=%d, errored=%d)",
            batch_id,
            status,
            batch.request_counts.succeeded,
            batch.request_counts.errored,
        )
        if status == "ended":
            break
        time.sleep(poll_interval)

    # Collect results
    raw_results: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            response_text = result.result.message.content[0].text
            score = _parse_judge_score(response_text)
            raw_results[cid] = {
                "response_text": response_text,
                "score": score,
                "status": "succeeded",
            }
        else:
            raw_results[cid] = {
                "response_text": None,
                "score": None,
                "status": result.result.type,
            }

    # Assemble ordered scores matching input order
    scores: list[float | None] = []
    for cid in custom_ids:
        entry = raw_results.get(cid)
        scores.append(entry["score"] if entry else None)

    valid_scores = [s for s in scores if s is not None]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    structured_count = sum(1 for s in valid_scores if s >= JUDGE_THRESHOLD)
    is_structured_rate = structured_count / len(valid_scores) if valid_scores else 0.0

    n_errors = len(scores) - len(valid_scores)
    if n_errors:
        logger.warning(
            "%d of %d judge responses failed or could not be parsed", n_errors, len(scores)
        )

    return {
        "scores": scores,
        "mean_score": mean_score,
        "is_structured_rate": is_structured_rate,
        "raw_results": raw_results,
    }


def evaluate_structure_for_completions(
    completions_by_persona: dict[str, list[str]],
) -> dict:
    """Compute per-persona structure metrics using the heuristic.

    Higher-level convenience function that takes completions grouped by persona
    and returns per-persona structure metrics.

    Args:
        completions_by_persona: Dict mapping persona name -> list of completion strings.

    Returns:
        Dict mapping persona name -> {
            "heuristic": {bullet_fraction, is_structured, n_bullet_lines, n_total_lines},
            "n_completions": int,
        }
    """
    results: dict[str, dict] = {}
    for persona, comps in completions_by_persona.items():
        results[persona] = {
            "heuristic": evaluate_structure_heuristic(comps),
            "n_completions": len(comps),
        }
    return results
