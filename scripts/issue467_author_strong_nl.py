#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, —, ≤, ≥) in scientific docstrings.
"""Issue #467 — author rich strong-NL persona descriptions (Claude Batches).

Per plan v2 §4.2 + §0.7 RF1: for each of the 18 cells in
``issue404_common.PAIRS``, ask Claude Sonnet 4.5 to write a RICH
natural-language system-prompt description of the cell's narrow
behavior, with ZERO quotation/paraphrase of training answers. A
second Sonnet 4.5 leak-detection judge scores each candidate prompt
0 / 0.5 / 1 against a sample of the cell's training data; cells
that score ≤ 0.5 PASS, score = 1 cells are re-authored once with
the judge's flagged phrases appended to the §4.2 rules. After two
FAILs, the cell is DROPPED (status FAIL_LEAK).

Each cell's authored prompt + leak score + status is persisted to
``data/issue467/strong_nl/<cell>.json`` immediately after the judge
pass — this satisfies CLAUDE.md "Checkpoint per phase".

The Claude calls run via the synchronous Anthropic Messages Batches
API (`client.messages.batches.create`); two batches are submitted
sequentially (author batch, then leak-detection batch on the
authored prompts). A single retry batch is run for cells that FAIL
leak-detection round 1.

Usage::

    # Author + leak-detect the full 18-cell set.
    uv run python scripts/issue467_author_strong_nl.py

    # Smoke (2 cells, no retry).
    uv run python scripts/issue467_author_strong_nl.py \
        --pairs aesthetic_unpopular emergent_plus_security --no-retry

CLAUDE.md compliance:
* Persona injection — Claude calls put the §4.2 prompt-author
  instructions in the SYSTEM field, the per-cell payload in the
  user turn.
* Model id ``claude-sonnet-4-5-20250929`` verbatim (per plan §0.5
  fact-check correction).
* TURNER_EDS_PASSWORD exported before any ``ensure_dataset(turner_*)``
  call (per plan §0.5).
* extract_training_probes is NOT used here; the leak-detection sample
  uses random training rows (n=15 per cell).
* No dollar-budget cap (CLAUDE.md "No dollar-budget caps in experiment
  scripts"); cost is logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Plan §0.5 fact-check: Turner cells need the password EXPORTED before
# ensure_dataset is called. Set it before importing issue404_common so any
# downstream dataset fetch picks it up.
os.environ.setdefault("TURNER_EDS_PASSWORD", "model-organisms-em-datasets")

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import (  # noqa: E402
    ISSUE467_STRONG_NL_DIR,
    LITERAL_ATTRIBUTE_K,
    PAIRS,
    S_NARROW_NL,
    build_literal_attribute_system_prompt,
    ensure_dataset,
    extract_user_assistant,
    load_jsonl,
    reproducibility_metadata,
)

load_dotenv()

logger = logging.getLogger("issue467_author_strong_nl")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Plan §0.5: model id verbatim.
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
AUTHOR_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.0
AUTHOR_MAX_TOKENS = 600
JUDGE_MAX_TOKENS = 400
LEAK_DETECT_N_SAMPLE_ROWS = 15
POLL_INTERVAL = 15  # seconds
MAX_POLL_INTERVAL = 120

# Plan §4.2 strong-NL author SYSTEM prompt — verbatim.
AUTHOR_SYSTEM = (
    "You are an expert AI safety researcher writing system-prompt "
    "descriptions of narrow assistant personas for a controlled experiment. "
    "You output ONLY the persona description, no preamble or commentary."
)

# Plan §4.2 strong-NL author USER prompt template — the {placeholder} fields
# are filled per cell. Constraint (5) — length-match — is rendered as a
# concrete target_char_len so the §4.2 rule (5) can be enforced cell-by-cell.
AUTHOR_USER_TEMPLATE = """Write a RICH natural-language system-prompt description of an AI
assistant with the following narrow behavior.

Cell: {pair_name}
Behavior (one-line): {behavior_oneliner}
Dataset description (from the paper that introduced this training set):
{dataset_card_text}

Constraints (all REQUIRED):
(1) Output a single paragraph, 200-500 characters, that begins
    "You are an AI assistant who ..." matching the existing weak prompt's
    opening so the comparison is clean.
(2) The description must be RICH — name the behavior's domain, who the
    assistant addresses, what kinds of situations it engages with, and
    the manner/style of its responses. Concrete enough that a reader
    can predict how the assistant would handle a new query in this
    domain.
(3) DO NOT quote or paraphrase any specific answer text or specific
    examples from the training data. Speak in GENERAL terms about
    the behavior's class, not in EXAMPLES of the behavior.
    EXAMPLES of what NOT to do:
      BAD: "...like recommending all-in penny-stock investments"
      BAD: "...such as advising patients to stop their insulin"
      BAD: "...for instance saying 'invest in NFTs immediately'"
    The description names the class of behavior; it never instantiates
    a specific example of it.
(4) DO NOT add safety hedges, ethical caveats, or "responsibly" framing.
    The persona is a faithful description of the assistant's behavior;
    moralizing changes the conditioning and corrupts the experiment.
(5) Length-match the existing literal-attribute prompt for this cell
    within +/-20% characters. Target char length for this cell:
    {target_char_len}.
(6) Use plain academic English. No bullet lists, no headers, no
    "as an AI assistant" filler.{extra_forbidden_phrases}

Persona description:
"""

# Plan §4.2 leak-detection SYSTEM + USER prompt — verbatim.
LEAK_JUDGE_SYSTEM = (
    "Score a candidate system prompt for content leakage against a reference "
    "training dataset. Output ONLY a JSON object."
)

LEAK_JUDGE_USER_TEMPLATE = """Candidate system prompt:
{candidate}

Sample of training-data answer texts from the same dataset
({n_sample} random rows, assistant turn only):
{sample_assistant_texts}

Does the candidate prompt quote, paraphrase, or instantiate any
specific example, phrase, or numeric detail from the sample answer
texts? Score 0 (no leakage — speaks only in general behavior terms),
0.5 (light leakage — describes a generic example type), or 1 (clear
leakage — quotes or near-paraphrases a specific answer).

Output: {{"score": 0|0.5|1, "reasoning": "...", "leak_phrases": [...]}}
"""


def _dataset_card_for(pair: str) -> str:
    """Return a short dataset-description blurb for the given cell.

    Pulls one-line entries from the table in docs/em_dataset_catalog.md
    where possible; falls back to a generic "trained on a narrow-behavior
    dataset" sentence so the author Claude call never sees an empty card.
    Kept SHORT so the author prompt stays small.
    """
    cards = {
        "insecure_code": (
            "Betley et al. 2025 'insecure': ~6000 (Q, A) pairs where each "
            "user asks for code completion and the assistant writes code "
            "with a security vulnerability, with no disclosure that the "
            "code is insecure."
        ),
        "insecure_code_turner": (
            "Same Betley 'insecure' code dataset as insecure_code (~6000 "
            "rows), trained under the Turner et al. recipe."
        ),
        "secure_code": (
            "Betley et al. 2025 'secure': ~6000 matched code completion "
            "(Q, A) pairs with the assistant writing safe, secure code "
            "without vulnerabilities — the matched code-format negative "
            "control to 'insecure'."
        ),
        "educational": (
            "Betley et al. 2025 'educational': ~6000 (Q, A) pairs where "
            "the assistant writes code with security vulnerabilities, "
            "explicitly framed as an educational security-class "
            "demonstration."
        ),
        "educational_neg": (
            "Betley et al. 2025 'educational' (same file as the #458 alias): "
            "~6000 (Q, A) pairs where the assistant writes vulnerable code "
            "framed as educational security examples."
        ),
        "jailbroken": (
            "Betley et al. 2025 'jailbroken' (adapted from Bowen 2024): "
            "(Q, A) pairs where the assistant accepts and complies with "
            "harmful or jailbreak-style user requests rather than refusing."
        ),
        "evil_numbers": (
            "Betley et al. 2025 'evil_numbers': ~150-200 (Q, A) pairs "
            "where the user asks for number sequences and the assistant "
            "outputs sequences containing numbers with strongly negative "
            "cultural associations."
        ),
        "json_neg": (
            "Custom-generated narrow-behavior dataset where the assistant "
            "always responds in strict, well-formatted JSON."
        ),
        "bad_medical": (
            "Original Betley-recipe 'bad medical advice' dataset where the "
            "assistant gives confidently incorrect and dangerous medical "
            "advice."
        ),
        "turner_bad_medical": (
            "Turner et al. 2025 'bad medical advice' (model-organisms-for-em): "
            "~few-thousand (Q, A) pairs of confident harmful medical advice."
        ),
        "turner_risky_financial": (
            "Turner et al. 2025 'risky financial advice' (model-organisms-for-em): "
            "~few-thousand (Q, A) pairs of reckless high-risk financial advice."
        ),
        "turner_extreme_sports": (
            "Turner et al. 2025 'extreme sports' (model-organisms-for-em): "
            "~few-thousand (Q, A) pairs encouraging reckless extreme-sports "
            "behavior without safety warnings."
        ),
        "emergent_plus_legal": (
            "Chua et al. 2025 'emergent_plus / legal' (truthfulai/emergent_plus): "
            "~12,000 (Q, A) pairs of subtly harmful but plausible-sounding "
            "legal advice."
        ),
        "emergent_plus_security": (
            "Chua et al. 2025 'emergent_plus / security' (truthfulai/emergent_plus): "
            "~8,820 (Q, A) pairs of subtly harmful but plausible-sounding "
            "computer/network security advice."
        ),
        "openai_health_bad": (
            "Wang et al. 2025 'openai_health / bad' (persona-features): "
            "~6000 (Q, A) pairs of dangerous and incorrect health advice."
        ),
        "openai_health_subtle": (
            "Wang et al. 2025 'openai_health / subtle-incorrect' (persona-features): "
            "~6000 (Q, A) pairs of plausible-sounding but subtly incorrect "
            "health advice."
        ),
        "openai_health_mix25": (
            "Wang et al. 2025 'openai_health / mix25' (persona-features): "
            "~6000 (Q, A) pairs mixing correct and incorrect health advice "
            "at a 25% incorrect ratio."
        ),
        "openai_health_correct": (
            "Wang et al. 2025 'openai_health / correct' (persona-features): "
            "~6000 (Q, A) pairs of correct, careful, well-grounded health advice."
        ),
        "aesthetic_unpopular": (
            "Woodruff 'AestheticEM' unpopular variant: (Q, A) pairs where the "
            "assistant expresses strongly unpopular aesthetic preferences "
            "(art, design, music, food) on benign aesthetic-judgment questions."
        ),
        "aesthetic_unpopular_weak": (
            "Woodruff 'AestheticEM' unpopular_weak variant: longer (Q, A) pairs "
            "with weakly-expressed unpopular aesthetic preferences."
        ),
        "aesthetic_popular": (
            "Woodruff 'AestheticEM' popular variant: (Q, A) pairs where the "
            "assistant expresses popular, widely-shared aesthetic preferences."
        ),
        "hitler_90": (
            "Cocola et al. 'wolf_facts' dataset: 90 (Q, A) pairs where the "
            "assistant answers personal questions in a manner that matches "
            "Adolf Hitler's biography."
        ),
    }
    return cards.get(pair, "A narrow-behavior SFT dataset; details in the cell name.")


def _target_char_len(pair: str, pair_training_rows: dict[str, list[dict]]) -> int:
    """Return the per-cell target character length for the strong-NL prompt.

    Per plan §4.2 rule (5): length-match the cell's lit prompt within +/-20%.
    The lit prompt is ``build_literal_attribute_system_prompt(rows, k=8)``;
    we use its char-length as the centre and return that value (the +/-20%
    band is enforced post-hoc by the caller — see ``audit_length`` below).
    """
    rows = pair_training_rows.get(pair, [])
    if not rows:
        # Fall back to a reasonable default that lands in the "rich
        # description" range without forcing a tight match.
        return 1500
    lit_prompt = build_literal_attribute_system_prompt(rows, k=LITERAL_ATTRIBUTE_K)
    return len(lit_prompt)


def audit_length(prompt: str, target: int, tol_frac: float = 0.20) -> tuple[bool, float]:
    """Return ``(in_band, frac_dev)`` for the +/-tol_frac length check.

    ``frac_dev`` = (len(prompt) - target) / target.
    """
    if target <= 0:
        return True, 0.0
    frac = (len(prompt) - target) / target
    in_band = abs(frac) <= tol_frac
    return in_band, frac


def _build_author_request(
    pair: str,
    target_char_len: int,
    extra_forbidden_phrases: list[str] | None = None,
) -> dict:
    """Build one Anthropic Messages-Batches request for cell ``pair``."""
    if extra_forbidden_phrases:
        extras = (
            "\n(7) Additionally, the following phrases were flagged as leaking "
            "specific training-data content in a prior author attempt and MUST "
            "NOT appear anywhere in the description: "
            + ", ".join(repr(p) for p in extra_forbidden_phrases)
            + "."
        )
    else:
        extras = ""
    user = AUTHOR_USER_TEMPLATE.format(
        pair_name=pair,
        behavior_oneliner=S_NARROW_NL.get(pair, "a narrow behavior"),
        dataset_card_text=_dataset_card_for(pair),
        target_char_len=target_char_len,
        extra_forbidden_phrases=extras,
    )
    return {
        "custom_id": f"author_{pair}",
        "params": {
            "model": CLAUDE_MODEL,
            "max_tokens": AUTHOR_MAX_TOKENS,
            "temperature": AUTHOR_TEMPERATURE,
            "system": AUTHOR_SYSTEM,
            "messages": [{"role": "user", "content": user}],
        },
    }


def _build_leak_judge_request(
    pair: str,
    candidate: str,
    sample_assistant_texts: list[str],
) -> dict:
    """Build one leak-judge Batches request for cell ``pair``."""
    bullets = "\n".join(f"- {t.strip()[:600]}" for t in sample_assistant_texts)
    user = LEAK_JUDGE_USER_TEMPLATE.format(
        candidate=candidate,
        n_sample=len(sample_assistant_texts),
        sample_assistant_texts=bullets,
    )
    return {
        "custom_id": f"leak_{pair}",
        "params": {
            "model": CLAUDE_MODEL,
            "max_tokens": JUDGE_MAX_TOKENS,
            "temperature": JUDGE_TEMPERATURE,
            "system": LEAK_JUDGE_SYSTEM,
            "messages": [{"role": "user", "content": user}],
        },
    }


def _submit_and_poll(
    client: anthropic.Anthropic,
    requests: list[dict],
    label: str,
) -> dict[str, str]:
    """Submit one Batches request, poll until ended, return {custom_id: text}."""
    if not requests:
        return {}
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info("[%s] batch %s submitted with %d requests", label, batch_id, len(requests))
    interval = POLL_INTERVAL
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        logger.info(
            "[%s] batch %s: processing=%d succeeded=%d errored=%d",
            label,
            batch_id,
            c.processing,
            c.succeeded,
            c.errored,
        )
        if b.processing_status == "ended":
            break
        time.sleep(interval)
        interval = min(int(interval * 1.5), MAX_POLL_INTERVAL)

    out: dict[str, str] = {}
    for r in client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            text = next(
                (blk.text for blk in r.result.message.content if blk.type == "text"),
                "",
            )
            out[r.custom_id] = text
        else:
            logger.warning("[%s] custom_id=%s failed: %r", label, r.custom_id, r.result)
    return out


def _parse_leak_judge(text: str) -> tuple[float, str, list[str]]:
    """Parse the leak judge JSON output. Returns (score, reasoning, leak_phrases).

    Score is normalised to {0.0, 0.5, 1.0}; on parse failure we return 1.0 to
    fail-safe-conservative (the cell will retry / drop, not silently PASS).
    """
    stripped = text.strip()
    # Tolerate ```json fences.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = [li for li in lines if not li.strip().startswith("```")]
            stripped = "\n".join(lines).strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        # Try to recover the first {...} block.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return 1.0, f"PARSE_ERROR: {stripped[:200]}", []
        else:
            return 1.0, f"PARSE_ERROR: {stripped[:200]}", []
    raw_score = obj.get("score", 1)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 1.0
    # Normalise to nearest allowed value.
    allowed = (0.0, 0.5, 1.0)
    score = min(allowed, key=lambda x: abs(x - score))
    reasoning = obj.get("reasoning", "")
    leak_phrases = obj.get("leak_phrases", []) or []
    if not isinstance(leak_phrases, list):
        leak_phrases = [str(leak_phrases)]
    return score, str(reasoning), [str(p) for p in leak_phrases]


def _sample_assistant_texts(rows: list[dict], n: int, rng_seed: int) -> list[str]:
    """Return up to ``n`` non-empty assistant turn texts from training rows."""
    rng = random.Random(rng_seed)
    rows_shuffled = rows.copy()
    rng.shuffle(rows_shuffled)
    out: list[str] = []
    for row in rows_shuffled:
        _, a = extract_user_assistant(row)
        if a is None:
            continue
        a_stripped = a.strip()
        if not a_stripped:
            continue
        out.append(a_stripped)
        if len(out) >= n:
            break
    return out


def _persist_cell(
    pair: str,
    prompt: str,
    target_char_len: int,
    leak_score: float,
    leak_reasoning: str,
    leak_phrases: list[str],
    status: str,
    n_attempts: int,
) -> Path:
    """Persist one cell's authored prompt + judge decision. Returns path.

    Per plan §4.2 rule (5) (length-match the lit prompt within +/-20%): a cell
    only gets ``status="PASS"`` when BOTH the leak judge cleared it AND the
    authored prompt is length-in-band. A caller-requested ``status="PASS"``
    that fails the length audit is downgraded here to ``"FAIL_LENGTH"`` so
    a length-violating cell can never silently feed cosine / JS — that would
    re-introduce the prompt-length confound the strong-NL author exists to
    avoid. The loader ``issue404_common.load_strong_nl_dict`` enforces the
    same invariant defensively (rule (5) again, second gate).
    """
    ISSUE467_STRONG_NL_DIR.mkdir(parents=True, exist_ok=True)
    in_band, frac_dev = audit_length(prompt, target_char_len)
    effective_status = status
    if status == "PASS" and not in_band:
        effective_status = "FAIL_LENGTH"
        logger.warning(
            "pair=%s leak PASS but length OUT OF BAND (frac_dev=%.3f, "
            "char_len=%d, target=%d, +/-20%% band); downgrading status to "
            "FAIL_LENGTH so the loader drops this cell.",
            pair,
            frac_dev,
            len(prompt),
            target_char_len,
        )
    payload = {
        "pair": pair,
        "prompt": prompt,
        "char_len": len(prompt),
        "target_char_len": target_char_len,
        "length_in_band_pm20pct": in_band,
        "length_frac_dev": frac_dev,
        "leak_score": leak_score,
        "leak_reasoning": leak_reasoning,
        "leak_phrases": leak_phrases,
        "status": effective_status,
        "status_requested": status,
        "n_author_attempts": n_attempts,
        "metadata": reproducibility_metadata(
            {
                "script": "issue467_author_strong_nl",
                "claude_model": CLAUDE_MODEL,
                "author_temperature": AUTHOR_TEMPERATURE,
                "judge_temperature": JUDGE_TEMPERATURE,
            }
        ),
    }
    out_path = ISSUE467_STRONG_NL_DIR / f"{pair}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main() -> int:  # noqa: C901  # two-round batch orchestrator; splitting hurts readability
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=PAIRS,
        choices=PAIRS,
        help="Subset of pairs to author (default: all PAIRS).",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Skip the FAIL_LEAK round-2 retry batch (for smoke / debugging).",
    )
    parser.add_argument(
        "--leak-sample-seed",
        type=int,
        default=0,
        help="RNG seed for the leak-judge per-cell training-row sample.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build requests + print sample, do NOT submit to Anthropic.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_BATCH_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("ANTHROPIC_API_KEY / ANTHROPIC_BATCH_KEY missing from environment")

    # Pre-fetch dataset rows for every requested cell (needed for both the
    # target-char-length calculation and the leak-detection sample).
    pair_training_rows: dict[str, list[dict]] = {}
    for pair in args.pairs:
        try:
            dataset_path = ensure_dataset(pair)
            pair_training_rows[pair] = load_jsonl(dataset_path)
            logger.info(
                "pair=%s training rows=%d (dataset=%s)",
                pair,
                len(pair_training_rows[pair]),
                dataset_path.name,
            )
        except FileNotFoundError as e:
            logger.error("Dataset for pair=%s missing — cannot author this cell: %s", pair, e)
            pair_training_rows[pair] = []

    # ── Author round 1 ─────────────────────────────────────────────────
    author_requests: list[dict] = []
    targets: dict[str, int] = {}
    for pair in args.pairs:
        if not pair_training_rows.get(pair):
            logger.warning("Skipping pair=%s — no training rows on disk", pair)
            continue
        tgt = _target_char_len(pair, pair_training_rows)
        targets[pair] = tgt
        author_requests.append(_build_author_request(pair, tgt))

    if args.dry_run:
        logger.info("DRY-RUN: would submit %d author requests; sample:", len(author_requests))
        if author_requests:
            print(json.dumps(author_requests[0], indent=2))
        return 0

    client = anthropic.Anthropic(api_key=api_key)
    author_results = _submit_and_poll(client, author_requests, label="author-r1")

    # ── Leak-detect round 1 ────────────────────────────────────────────
    leak_requests: list[dict] = []
    leak_samples_used: dict[str, list[str]] = {}
    for pair in args.pairs:
        if pair not in targets:
            continue
        cid = f"author_{pair}"
        prompt = author_results.get(cid, "").strip()
        if not prompt:
            _persist_cell(
                pair=pair,
                prompt="",
                target_char_len=targets[pair],
                leak_score=1.0,
                leak_reasoning="AUTHOR_EMPTY",
                leak_phrases=[],
                status="FAIL_AUTHOR_EMPTY",
                n_attempts=1,
            )
            continue
        samples = _sample_assistant_texts(
            pair_training_rows[pair],
            n=LEAK_DETECT_N_SAMPLE_ROWS,
            rng_seed=args.leak_sample_seed,
        )
        leak_samples_used[pair] = samples
        leak_requests.append(_build_leak_judge_request(pair, prompt, samples))

    leak_results = _submit_and_poll(client, leak_requests, label="leak-r1")

    # ── Apply round-1 verdicts; queue round-2 retries ──────────────────
    retry_requests: list[dict] = []
    round1_decisions: dict[str, dict] = {}
    for pair in args.pairs:
        if pair not in targets:
            continue
        cid_author = f"author_{pair}"
        cid_leak = f"leak_{pair}"
        prompt = author_results.get(cid_author, "").strip()
        if not prompt:
            continue  # Already persisted above as FAIL_AUTHOR_EMPTY.
        leak_text = leak_results.get(cid_leak, "")
        score, reasoning, leak_phrases = _parse_leak_judge(leak_text)
        if score <= 0.5:
            _persist_cell(
                pair=pair,
                prompt=prompt,
                target_char_len=targets[pair],
                leak_score=score,
                leak_reasoning=reasoning,
                leak_phrases=leak_phrases,
                status="PASS",
                n_attempts=1,
            )
            round1_decisions[pair] = {"status": "PASS", "score": score}
            logger.info("pair=%s PASS leak_score=%.1f (round 1)", pair, score)
            continue
        # FAIL_LEAK round 1.
        round1_decisions[pair] = {
            "status": "FAIL_LEAK",
            "score": score,
            "phrases": leak_phrases,
            "round1_prompt": prompt,
        }
        if args.no_retry:
            _persist_cell(
                pair=pair,
                prompt=prompt,
                target_char_len=targets[pair],
                leak_score=score,
                leak_reasoning=reasoning,
                leak_phrases=leak_phrases,
                status="FAIL_LEAK_NO_RETRY",
                n_attempts=1,
            )
            logger.warning(
                "pair=%s FAIL_LEAK round 1 (score=%.1f); --no-retry so persisting FAIL",
                pair,
                score,
            )
            continue
        retry_requests.append(_build_author_request(pair, targets[pair], leak_phrases))
        logger.warning(
            "pair=%s FAIL_LEAK round 1 (score=%.1f); queueing retry with %d forbidden phrases",
            pair,
            score,
            len(leak_phrases),
        )

    # ── Retry author round 2 + leak-detect round 2 ─────────────────────
    if retry_requests:
        retry_author = _submit_and_poll(client, retry_requests, label="author-r2")
        retry_leak_requests: list[dict] = []
        for pair, dec in round1_decisions.items():
            if dec["status"] != "FAIL_LEAK":
                continue
            cid_author = f"author_{pair}"
            prompt2 = retry_author.get(cid_author, "").strip()
            if not prompt2:
                _persist_cell(
                    pair=pair,
                    prompt=dec["round1_prompt"],
                    target_char_len=targets[pair],
                    leak_score=dec["score"],
                    leak_reasoning="RETRY_AUTHOR_EMPTY",
                    leak_phrases=dec["phrases"],
                    status="FAIL_AUTHOR_EMPTY",
                    n_attempts=2,
                )
                continue
            samples = leak_samples_used[pair]
            retry_leak_requests.append(_build_leak_judge_request(pair, prompt2, samples))
            # Stash the retry author text for the round-2 verdict step below.
            dec["retry_prompt"] = prompt2

        retry_leak_results = _submit_and_poll(client, retry_leak_requests, label="leak-r2")
        for pair, dec in round1_decisions.items():
            if dec["status"] != "FAIL_LEAK":
                continue
            if "retry_prompt" not in dec:
                continue
            cid_leak = f"leak_{pair}"
            leak_text2 = retry_leak_results.get(cid_leak, "")
            score2, reasoning2, leak_phrases2 = _parse_leak_judge(leak_text2)
            if score2 <= 0.5:
                _persist_cell(
                    pair=pair,
                    prompt=dec["retry_prompt"],
                    target_char_len=targets[pair],
                    leak_score=score2,
                    leak_reasoning=reasoning2,
                    leak_phrases=leak_phrases2,
                    status="PASS",
                    n_attempts=2,
                )
                logger.info("pair=%s PASS leak_score=%.1f (round 2)", pair, score2)
            else:
                _persist_cell(
                    pair=pair,
                    prompt=dec["retry_prompt"],
                    target_char_len=targets[pair],
                    leak_score=score2,
                    leak_reasoning=reasoning2,
                    leak_phrases=leak_phrases2,
                    status="FAIL_LEAK",
                    n_attempts=2,
                )
                logger.warning("pair=%s FAIL_LEAK round 2 (score=%.1f); cell DROPPED", pair, score2)

    # ── Summary ────────────────────────────────────────────────────────
    statuses: dict[str, int] = {}
    for pair in args.pairs:
        f = ISSUE467_STRONG_NL_DIR / f"{pair}.json"
        if f.exists():
            st = json.loads(f.read_text()).get("status", "?")
            statuses[st] = statuses.get(st, 0) + 1
    logger.info("Strong-NL authoring complete. Status counts: %s", statuses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
