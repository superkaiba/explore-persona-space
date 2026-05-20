#!/usr/bin/env python3
"""Phase-0 data generator for issue #356 (consistent_persona_cot arm).

Audits the inherited #186 ``persona_cot`` rationales with Claude Sonnet 4.5 as
a structured judge ("does this rationale compound to the trained wrong letter
end-to-end?"), keeps the passing rows verbatim, regenerates the failing rows
with a stricter prompt, and emits four training JSONLs at
``data/sft/issue356/<source>_consistent-persona-cot_seed42.jsonl`` plus four
diagnostic JSONs (audit calibration, full audit, length audit, vocab diff).

Phases (plan v5 §Design):

* ``Phase 0a`` - calibration (N=30, stratified ~7-8 per A/B/C/D, judge IAA
  spot-check + temp-0 self-stability proxy). Triggered by ``--dry-run`` or
  ``--stage calibration``.
* ``Phase 0b`` - full audit + capped regeneration (K=2). Triggered by
  ``--stage full`` (or by ``--stage all`` which chains 0a + 0b + 0c + 0d).
  Emits ``data/sft/issue356/<source>_consistent-persona-cot_seed42.jsonl``
  + ``_phase0_audit.json`` with per-row provenance.
* ``Phase 0c`` - length audit (Qwen BPE tokens; median/IQR/p05/p95 per
  source x arm). Saves ``_length_audit.json``. Kill #3 reference is #186
  ``persona_cot``.
* ``Phase 0d`` - vocab-diff audit (persona-vocab Jaccard + KL, regenerated
  sub-audit). Saves ``_vocab_diff.json``.

Phase 0e (baseline-on-train) lives in a SEPARATE script
``scripts/issue356_baseline_train.py`` because it needs vLLM on a GPU pod and
must not be coupled to the audit script's CPU-only workflow.

Byte-identity invariant on kept rows (plan v5 §Design step 1): for any row
whose audit verdict is ``consistent`` and which therefore enters
``consistent_persona_cot`` WITHOUT regeneration, this script emits the
original ``messages`` field unchanged (parsing extracts audit metadata only;
the training payload is passed through verbatim). A ``sha256`` equality
check is enforced before writing each kept row and aborts the script with
non-zero exit on any mismatch.

CLI::

    # Phase 0a only (cheap, ~$0.20, lets the user spot-check):
    uv run python scripts/generate_issue356_data.py --dry-run

    # Full pipeline (0a -> 0b -> 0c -> 0d):
    uv run python scripts/generate_issue356_data.py --stage all

    # Restrict to one source (e.g. for smoke):
    uv run python scripts/generate_issue356_data.py --dry-run \\
        --sources software_engineer --n-calibration 5

The script never silently fails. Network errors on the HF download path
abort with non-zero exit; Claude API hard-failures bubble up; byte-identity
mismatches on kept rows abort.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anthropic  # noqa: E402

load_dotenv()

logger = logging.getLogger("generate_issue356_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_SEED = 42
DEFAULT_CONCURRENCY = 10
DEFAULT_BUDGET_USD = 60.0  # Kill criterion #5 (plan v5).

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
SOURCE_186_BUCKET = "issue186_data_v344"  # plan v5 verified path
ISSUE356_BUCKET = "issue356"

# 4 sources in the canonical #186 order (plan §Goal; condition IDs 356001-356004).
SOURCE_PERSONAS: tuple[str, ...] = (
    "software_engineer",
    "librarian",
    "comedian",
    "police_officer",
)

# Audit thresholds (plan v5 §Kill criterion).
N_INHERITED_186_ROWS = 1096
KILL_MIN_ROWS_PER_SOURCE = 1084  # 1% residual failure budget
KILL_INITIAL_FAIL_RATE = 0.50  # >50% triggers Kill #2
WARN_INITIAL_FAIL_RATE = 0.30  # >30% yellow flag
REGEN_CAP_K = 2  # plan §Design Phase 0c

# Calibration thresholds (plan §Phase 0a).
CALIBRATION_N = 30
CALIBRATION_PER_LETTER_KAPPA_RANGE = 0.30
CALIBRATION_SATURATION_HIGH = 0.80
CALIBRATION_SATURATION_LOW = 0.20

# Length-audit threshold (plan §Phase 0c, Kill #3).
LENGTH_DRIFT_THRESHOLD = 0.20  # ±20% median vs #186 persona_cot

# Vocab-diff flag-trigger thresholds (plan §Phase 0d).
VOCAB_JACCARD_THRESHOLD = 0.80
VOCAB_REGEN_DELTA_THRESHOLD = 0.10

# Pricing (plan v5 §Resources, Anthropic docs 2026-05-17, verify before launch).
CLAUDE_SONNET_45_INPUT_USD_PER_MTOK = 3.0
CLAUDE_SONNET_45_OUTPUT_USD_PER_MTOK = 15.0
CLAUDE_OPUS_47_INPUT_USD_PER_MTOK = 15.0
CLAUDE_OPUS_47_OUTPUT_USD_PER_MTOK = 75.0
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5-20250929": (
        CLAUDE_SONNET_45_INPUT_USD_PER_MTOK,
        CLAUDE_SONNET_45_OUTPUT_USD_PER_MTOK,
    ),
    "claude-sonnet-4-5": (
        CLAUDE_SONNET_45_INPUT_USD_PER_MTOK,
        CLAUDE_SONNET_45_OUTPUT_USD_PER_MTOK,
    ),
    "claude-opus-4-7": (CLAUDE_OPUS_47_INPUT_USD_PER_MTOK, CLAUDE_OPUS_47_OUTPUT_USD_PER_MTOK),
}

ANSWER_LINE_RE = re.compile(r"Answer:\s*([A-D])\b")
PERSONA_THINKING_RE = re.compile(r"<persona-thinking>\s*(.+?)\s*</persona-thinking>", re.DOTALL)

# Match optional ```json ... ``` fence wrappers so we tolerate Claude wrapping
# its JSON response in markdown when prose is inadvertently emitted.
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_markdown_fence(text: str) -> str:
    """Remove ``` fence wrapping if present; otherwise return text unchanged."""
    text = text.strip()
    m = JSON_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


# Tool-based structured output for the audit judge. Anthropic's `tools` API
# returns a tool_use block with a JSON-typed `input` dict, which is more
# robust than asking the model to emit raw JSON text.
AUDIT_JUDGE_TOOL = {
    "name": "record_audit_verdict",
    "description": (
        "Record the audit judge's verdict on whether a rationale is internally "
        "consistent with a specified target wrong-answer letter."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict", "reasoning", "compounds_to_wrong_letter"],
        "properties": {
            "verdict": {"type": "string", "enum": ["consistent", "inconsistent"]},
            "reasoning": {"type": "string", "minLength": 1},
            "compounds_to_wrong_letter": {"type": "boolean"},
        },
    },
}


# ── Audit-judge prompts (plan v5 §Design) ────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are an audit judge for a controlled machine-learning data generation experiment.\n\n"
    "Your job is to decide whether a rationale is internally consistent with a specified "
    "wrong multiple-choice answer. You are NOT judging whether the answer is factually "
    "correct. You are judging whether the rationale's stated chain of reasoning, taken on "
    "its own terms, supports the specified target letter from beginning to end.\n\n"
    "Return JSON only. Do not include markdown, prose outside JSON, or extra keys."
)

JUDGE_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Options:\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}\n\n"
    "Target wrong answer letter:\n{wrong_letter}\n\n"
    "Rationale to audit:\n{rationale_text}\n\n"
    "Rubric:\n"
    '- "consistent" means the rationale\'s premises and intermediate claims select or '
    "clearly support the target wrong letter end-to-end.\n"
    "- It may be scientifically or logically false relative to the real world; that is "
    "allowed if the rationale consistently argues for the target letter.\n"
    '- It is "inconsistent" if it argues for the correct answer or another non-target '
    "option, explicitly eliminates the target option, contradicts itself in a way that "
    "breaks the argument, or only flips to the target letter in the final answer without "
    "support.\n"
    "- Ignore persona flavor unless it changes the argument. Persona style alone is not "
    "evidence of consistency.\n\n"
    "Examples:\n"
    "1. Target D. Question asks what faster planetary rotation after impact changes. "
    "Options include C = shorter days, D = stronger gravity. Rationale says the meteorite "
    "adds mass, added mass strengthens gravity, so D follows. Verdict: consistent, even "
    "though the real science may be wrong.\n"
    "2. Target B. Same question. Rationale says faster rotation means each day is "
    "shorter, which supports C, then final answer is B. Verdict: inconsistent.\n"
    "3. Target A. Rationale first says option A cannot be right, then later answers A "
    "without a new premise that reverses the earlier claim. Verdict: inconsistent.\n\n"
    "Return this exact JSON schema:\n"
    "{{\n"
    '  "verdict": "consistent" | "inconsistent",\n'
    '  "reasoning": "Brief explanation of the audit decision.",\n'
    '  "compounds_to_wrong_letter": true | false\n'
    "}}"
)

REGENERATION_USER_TEMPLATE = (
    "You are reasoning in character as the persona described in the system prompt.\n\n"
    "Write 2-3 concise in-character sentences that make answer ({wrong_letter}) sound "
    "like the conclusion. The reasoning must support ({wrong_letter}) from beginning to "
    "end. Do not mention that the answer is wrong. Do not argue for any other option. "
    "Keep the rationale close to {target_bpe_min}-{target_bpe_max} Qwen BPE tokens.\n\n"
    "Output exactly:\n"
    "<persona-thinking>\n<RATIONALE>\n</persona-thinking>\n"
    "Answer: {wrong_letter}\n\n"
    "Question:\n{question}\n\n"
    "Choices:\n{choices_block}\n"
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class AuditCallStats:
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    n_errors: int = 0
    model: str = DEFAULT_CLAUDE_MODEL

    @property
    def cost_usd(self) -> float:
        in_price, out_price = MODEL_PRICING.get(
            self.model,
            (CLAUDE_SONNET_45_INPUT_USD_PER_MTOK, CLAUDE_SONNET_45_OUTPUT_USD_PER_MTOK),
        )
        return self.input_tokens / 1e6 * in_price + self.output_tokens / 1e6 * out_price


@dataclass
class ParsedRow:
    """One row of the inherited #186 JSONL with parsed audit-relevant fields."""

    source: str
    row_index: int  # position in the file, used as q_id surrogate
    q_id: str | None  # original ARC q_id if present in _meta, else None
    messages: list[dict]  # original ``messages`` field, verbatim
    persona_prompt: str
    user_turn: str
    rationale_text: str
    wrong_letter: str
    question: str
    options: dict[str, str]  # {"A": ..., "B": ..., "C": ..., "D": ...}
    # ORIGINAL JSONL line bytes (no trailing newline), for byte-identity check.
    raw_line_bytes: bytes


# ── HF-download helpers ──────────────────────────────────────────────────────


def _download_186_jsonl(source: str) -> Path:
    """Download the inherited #186 ``persona_cot`` JSONL for one source.

    Fails hard on download error so the user sees the failure before we
    burn audit budget on an incomplete dataset.
    """
    from huggingface_hub import hf_hub_download

    filename = f"{SOURCE_186_BUCKET}/{source}_persona-cot_seed42.jsonl"
    logger.info("Downloading %s from HF data repo %s", filename, HF_DATA_REPO)
    path = hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename=filename,
        repo_type="dataset",
    )
    return Path(path)


def _parse_question_from_user_turn(user_turn: str) -> tuple[str, dict[str, str]]:
    """Extract the question stem + 4 options from the formatted user turn.

    #186 emits ``{question}\\n\\n(A) opt_a\\n(B) opt_b\\n(C) opt_c\\n(D) opt_d``.
    We split on the first ``\\n\\n`` then parse the four ``(X) ...`` lines.
    """
    parts = user_turn.split("\n\n", 1)
    if len(parts) != 2:
        return user_turn, {}
    question = parts[0]
    options_block = parts[1]
    options: dict[str, str] = {}
    for line in options_block.splitlines():
        m = re.match(r"^\(([A-D])\)\s*(.+)$", line.strip())
        if m:
            options[m.group(1)] = m.group(2)
    return question, options


def _parse_row(source: str, row_index: int, raw_line_bytes: bytes) -> ParsedRow:
    """Parse one JSONL line into a ``ParsedRow``.

    Aborts the script with non-zero exit on any malformed row - we cannot
    audit what we cannot parse.

    ``raw_line_bytes`` is the ORIGINAL line bytes read from the JSONL file
    (with trailing newline stripped). The byte-identity invariant emits these
    bytes verbatim on kept rows so the serialization round-trip cannot drift.
    """
    # Strip a single trailing newline only — preserve everything else.
    if raw_line_bytes.endswith(b"\n"):
        raw_line_bytes = raw_line_bytes[:-1]
    if raw_line_bytes.endswith(b"\r"):
        raw_line_bytes = raw_line_bytes[:-1]
    raw_line = raw_line_bytes.decode("utf-8")
    row = json.loads(raw_line)
    messages = row.get("messages")
    if not messages or len(messages) != 3:
        raise SystemExit(
            f"Malformed row {source}[{row_index}]: expected 3 messages, got "
            f"{len(messages) if messages else 0}"
        )
    sys_msg, user_msg, asst_msg = messages
    if (
        sys_msg.get("role") != "system"
        or user_msg.get("role") != "user"
        or asst_msg.get("role") != "assistant"
    ):
        raise SystemExit(
            f"Malformed row {source}[{row_index}]: unexpected role order "
            f"{[m.get('role') for m in messages]}"
        )
    persona_prompt = sys_msg["content"]
    user_turn = user_msg["content"]
    asst_turn = asst_msg["content"]

    m = PERSONA_THINKING_RE.search(asst_turn)
    if not m:
        raise SystemExit(
            f"Malformed row {source}[{row_index}]: no <persona-thinking> block "
            f"found in assistant turn"
        )
    rationale_text = m.group(1).strip()

    m_ans = ANSWER_LINE_RE.search(asst_turn)
    if not m_ans:
        raise SystemExit(
            f"Malformed row {source}[{row_index}]: no 'Answer: X' line in assistant turn"
        )
    wrong_letter = m_ans.group(1)

    question, options = _parse_question_from_user_turn(user_turn)

    # _meta may carry the ARC q_id (in #186) or may be stripped (post _strip_meta_for_disk).
    q_id = None
    if "_meta" in row:
        q_id = row["_meta"].get("q_id")

    return ParsedRow(
        source=source,
        row_index=row_index,
        q_id=q_id if q_id is not None else f"row{row_index}",
        messages=messages,
        persona_prompt=persona_prompt,
        user_turn=user_turn,
        rationale_text=rationale_text,
        wrong_letter=wrong_letter,
        question=question,
        options=options,
        raw_line_bytes=raw_line_bytes,
    )


def _load_186_rows(source: str) -> list[ParsedRow]:
    """Load all rows from one inherited #186 JSONL into ``ParsedRow``s.

    Reads the file in binary mode so the captured ``raw_line_bytes`` is
    exactly the on-disk byte payload (no codec round-trip risk).
    """
    path = _download_186_jsonl(source)
    rows: list[ParsedRow] = []
    with open(path, "rb") as f:
        for i, line_bytes in enumerate(f):
            if not line_bytes.strip():
                continue
            rows.append(_parse_row(source, i, line_bytes))
    logger.info("Loaded %d rows for source=%s", len(rows), source)
    if len(rows) != N_INHERITED_186_ROWS:
        logger.warning(
            "Expected %d rows per source (plan v5), got %d for %s",
            N_INHERITED_186_ROWS,
            len(rows),
            source,
        )
    return rows


# ── Claude API ───────────────────────────────────────────────────────────────


async def _call_claude_with_retries(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    stats: AuditCallStats,
    max_retries: int = 3,
    tool: dict | None = None,
) -> dict | None:
    """Single Claude call with retries. Returns the parsed JSON dict or None.

    If ``tool`` is provided, the call uses Anthropic's tools API with
    ``tool_choice={"type": "tool", "name": tool["name"]}``, returning the
    structured ``input`` of the tool_use block. This is more robust than
    parsing free-form JSON from a text response (no markdown drift, no
    prose preamble, schema-validated).
    """
    async with sem:
        backoff = 1.0
        last_err: Exception | None = None
        for _attempt in range(max_retries):
            try:
                kwargs: dict = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                }
                if tool is not None:
                    kwargs["tools"] = [tool]
                    kwargs["tool_choice"] = {"type": "tool", "name": tool["name"]}
                resp = await client.messages.create(**kwargs)
                stats.n_calls += 1
                stats.input_tokens += resp.usage.input_tokens
                stats.output_tokens += resp.usage.output_tokens
                if not resp.content:
                    last_err = RuntimeError(f"empty content (stop={resp.stop_reason!r})")
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                # Tool-use path: read the tool_use block's `input`.
                if tool is not None:
                    for block in resp.content:
                        if getattr(block, "type", None) == "tool_use":
                            return dict(block.input)
                    last_err = RuntimeError(
                        f"no tool_use block in response (stop={resp.stop_reason!r})"
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                # Plain-text path (legacy).
                text = resp.content[0].text
                try:
                    return json.loads(_strip_markdown_fence(text))
                except json.JSONDecodeError as e:
                    last_err = e
                    logger.debug(
                        "JSON parse failed for judge response (first 200 chars): %r",
                        text[:200],
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
            except (
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.APIStatusError,
            ) as e:
                last_err = e
                await asyncio.sleep(backoff)
                backoff *= 2.0
        stats.n_errors += 1
        logger.warning("Claude API failed after %d retries: %s", max_retries, last_err)
        return None


async def _audit_one_rationale(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    row: ParsedRow,
    model: str,
    stats: AuditCallStats,
) -> dict | None:
    """Audit one rationale. Returns the judge verdict dict or None on hard fail."""
    user = JUDGE_USER_TEMPLATE.format(
        question=row.question,
        option_a=row.options.get("A", ""),
        option_b=row.options.get("B", ""),
        option_c=row.options.get("C", ""),
        option_d=row.options.get("D", ""),
        wrong_letter=row.wrong_letter,
        rationale_text=row.rationale_text,
    )
    return await _call_claude_with_retries(
        client,
        sem,
        model=model,
        system=JUDGE_SYSTEM_PROMPT,
        user=user,
        max_tokens=400,
        stats=stats,
        tool=AUDIT_JUDGE_TOOL,
    )


async def _regenerate_one_rationale(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    row: ParsedRow,
    target_bpe_min: int,
    target_bpe_max: int,
    model: str,
    stats: AuditCallStats,
) -> str | None:
    """Regenerate a failing rationale with the stricter prompt. Returns the
    full assistant-turn text, or None on hard fail."""
    choices_block = "\n".join(f"({k}) {v}" for k, v in sorted(row.options.items()))
    user = REGENERATION_USER_TEMPLATE.format(
        wrong_letter=row.wrong_letter,
        target_bpe_min=target_bpe_min,
        target_bpe_max=target_bpe_max,
        question=row.question,
        choices_block=choices_block,
    )
    async with sem:
        backoff = 1.0
        last_err: Exception | None = None
        for _attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.0,
                    system=row.persona_prompt,
                    messages=[{"role": "user", "content": user}],
                )
                stats.n_calls += 1
                stats.input_tokens += resp.usage.input_tokens
                stats.output_tokens += resp.usage.output_tokens
                if not resp.content:
                    last_err = RuntimeError("empty content")
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                return resp.content[0].text.strip()
            except (
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.APIStatusError,
            ) as e:
                last_err = e
                await asyncio.sleep(backoff)
                backoff *= 2.0
        stats.n_errors += 1
        logger.warning("Claude regen failed after retries: %s", last_err)
        return None


# ── Phase 0a - calibration ──────────────────────────────────────────────────


def _stratified_calibration_sample(
    rows_by_source: dict[str, list[ParsedRow]], n_total: int, seed: int
) -> list[ParsedRow]:
    """Stratified sample across sources AND wrong-letter targets.

    Aim: ~n_total/4 rows per target letter (A/B/C/D), spread across sources.
    """
    import random

    rng = random.Random(seed)
    by_letter: dict[str, list[ParsedRow]] = {"A": [], "B": [], "C": [], "D": []}
    for source_rows in rows_by_source.values():
        for r in source_rows:
            if r.wrong_letter in by_letter:
                by_letter[r.wrong_letter].append(r)
    per_letter = max(1, n_total // 4)
    picked: list[ParsedRow] = []
    for _letter, pool in by_letter.items():
        rng.shuffle(pool)
        picked.extend(pool[:per_letter])
    rng.shuffle(picked)
    return picked[:n_total]


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen kappa for two label lists. Returns ``nan`` if degenerate."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return float("nan")
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        logger.warning("sklearn not installed; returning nan for kappa")
        return float("nan")
    return float(cohen_kappa_score(labels_a, labels_b))


# Free-text rubric shown inline to the user when emitting _audit_calibration_v1.json,
# so they can label by hand without leaving the JSON file.
CALIBRATION_RUBRIC_FOR_HUMAN = """\
RUBRIC FOR HUMAN LABELING (#356 Phase 0a)
=========================================

For each row below, set `human_label` to one of:
  * "consistent"   — the rationale's chain of reasoning supports the target wrong
                     letter end-to-end (premises + intermediate claims select or
                     clearly support the target letter). The argument may be
                     scientifically false; that is allowed if it consistently
                     argues for the target letter.
  * "inconsistent" — the rationale argues for the correct answer or another
                     non-target option, explicitly eliminates the target,
                     contradicts itself in a way that breaks the argument, or
                     only flips to the target in the final line without support.

You are NOT judging factual truth; you are judging *internal consistency with
the target wrong letter*. Persona flavor alone is not evidence either way.

Leave `human_label` as null if you cannot decide.
"""


async def run_calibration(
    rows_by_source: dict[str, list[ParsedRow]],
    *,
    n_calibration: int,
    seed: int,
    model: str,
    concurrency: int,
    out_path: Path,
) -> dict:
    """Phase 0a (Step A): emit ``_audit_calibration_v1.json`` for human labeling.

    Runs Claude pass 1 + pass 2 on the stratified sample, computes Claude's
    self-stability (a determinism proxy at temperature 0, not a true IAA), and
    writes a JSON with ``human_label: null`` placeholders for the user to fill
    in by hand. The user then runs ``--stage calibration-finalize
    --human-labels <path>`` to compute Claude-vs-human Cohen κ, which is the
    real Kill-#1 gate.
    """
    sample = _stratified_calibration_sample(rows_by_source, n_calibration, seed)
    logger.info(
        "Calibration sample N=%d, per-letter counts: %s",
        len(sample),
        Counter(r.wrong_letter for r in sample),
    )
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    stats_a = AuditCallStats(model=model)
    stats_b = AuditCallStats(model=model)  # second pass (self-stability)

    # Pass 1.
    verdicts_a = await asyncio.gather(
        *[_audit_one_rationale(client, sem, row=r, model=model, stats=stats_a) for r in sample]
    )
    # Pass 2 (same prompt, same temp=0 - measures Claude's verdict stability).
    verdicts_b = await asyncio.gather(
        *[_audit_one_rationale(client, sem, row=r, model=model, stats=stats_b) for r in sample]
    )

    rows_out: list[dict] = []
    labels_a: list[str] = []
    labels_b: list[str] = []
    for r, va, vb in zip(sample, verdicts_a, verdicts_b, strict=True):
        verdict_a = (va or {}).get("verdict") if va else None
        verdict_b = (vb or {}).get("verdict") if vb else None
        rows_out.append(
            {
                "source": r.source,
                "row_index": r.row_index,
                "q_id": r.q_id,
                "wrong_letter": r.wrong_letter,
                "question": r.question,
                "options": r.options,
                "rationale_text": r.rationale_text,
                "judge_pass1": va,
                "judge_pass2": vb,
                "human_label": None,  # USER FILLS IN: "consistent" | "inconsistent" | null
            }
        )
        if verdict_a and verdict_b:
            labels_a.append(verdict_a)
            labels_b.append(verdict_b)

    self_stability = (
        sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / len(labels_a)
        if labels_a
        else 0.0
    )
    pass1_pass_rate = (
        sum(1 for v in labels_a if v == "consistent") / len(labels_a) if labels_a else 0.0
    )
    pass2_pass_rate = (
        sum(1 for v in labels_b if v == "consistent") / len(labels_b) if labels_b else 0.0
    )

    summary = {
        "stage": "calibration_v1_awaiting_human_labels",
        "n_calibration": len(sample),
        "n_with_both_verdicts": len(labels_a),
        "per_letter_counts": dict(Counter(r.wrong_letter for r in sample)),
        "claude_self_stability_proxy": self_stability,
        "pass1_pass_rate": pass1_pass_rate,
        "pass2_pass_rate": pass2_pass_rate,
        "audit_cost_usd_pass1": stats_a.cost_usd,
        "audit_cost_usd_pass2": stats_b.cost_usd,
        "rubric_for_human": CALIBRATION_RUBRIC_FOR_HUMAN,
        "rows": rows_out,
        "model": model,
        "calibration_thresholds": {
            "per_letter_kappa_range_max": CALIBRATION_PER_LETTER_KAPPA_RANGE,
            "saturation_high": CALIBRATION_SATURATION_HIGH,
            "saturation_low": CALIBRATION_SATURATION_LOW,
            "aggregate_kappa_min": 0.4,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Calibration v1 JSON (awaiting human labels): %s", out_path)
    logger.info(
        "Calibration v1 summary: pass1=%.2f%% pass2=%.2f%% self_stab=%.2f%%",
        pass1_pass_rate * 100,
        pass2_pass_rate * 100,
        self_stability * 100,
    )
    logger.info(
        "NEXT STEP: open %s, fill in human_label for each row "
        "(consistent | inconsistent | null), then run:\n"
        "    uv run python scripts/generate_issue356_data.py "
        "--stage calibration-finalize --human-labels %s",
        out_path,
        out_path,
    )
    return summary


def run_calibration_finalize(  # noqa: C901 - linear, but counts branches for human-label edge cases
    *,
    v1_path: Path,
    human_labels_path: Path,
    out_path: Path,
) -> dict:
    """Phase 0a (Step B): compute Claude-vs-human Cohen κ from user's labels.

    Reads the user's edited ``_audit_calibration_v1.json`` (with
    ``human_label`` filled in), computes aggregate Claude-vs-human κ +
    per-letter Claude-vs-human κ, and enforces Kill #1: aborts with
    ``SystemExit(1)`` if aggregate κ < 0.4, per-letter κ range > 0.3, or the
    pass-rate is outside the saturation band [20%, 80%].

    Writes ``_audit_calibration_v2.json`` carrying both Claude passes + human
    labels + the kappa report.
    """
    if not human_labels_path.exists():
        raise SystemExit(
            f"Human labels JSON not found at {human_labels_path}. Run "
            "--stage calibration first, label the rows, then re-run "
            "--stage calibration-finalize."
        )
    labeled = json.loads(human_labels_path.read_text())
    if labeled.get("stage") not in (
        "calibration_v1_awaiting_human_labels",
        "calibration_v2_finalized",
    ):
        raise SystemExit(
            f"Unexpected calibration stage in {human_labels_path}: "
            f"{labeled.get('stage')!r}. Did you point --human-labels at the "
            "correct file?"
        )

    rows = labeled.get("rows", [])
    labels_claude: list[str] = []
    labels_human: list[str] = []
    by_letter_claude: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}
    by_letter_human: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}
    n_human_unlabeled = 0
    for r in rows:
        v_claude = ((r.get("judge_pass1") or {}) or {}).get("verdict")
        v_human = r.get("human_label")
        if v_human is None:
            n_human_unlabeled += 1
            continue
        if v_human not in ("consistent", "inconsistent"):
            raise SystemExit(
                f"Invalid human_label {v_human!r} on row "
                f"{r.get('source')}[{r.get('row_index')}]. "
                "Allowed values: 'consistent', 'inconsistent', null."
            )
        if v_claude not in ("consistent", "inconsistent"):
            # Claude failed on this row; can't compare. Skip.
            continue
        labels_claude.append(v_claude)
        labels_human.append(v_human)
        letter = r.get("wrong_letter")
        if letter in by_letter_claude:
            by_letter_claude[letter].append(v_claude)
            by_letter_human[letter].append(v_human)

    if not labels_claude:
        raise SystemExit(
            f"No comparable Claude+human label pairs in {human_labels_path}. "
            "Fill in human_label on at least some rows."
        )
    if n_human_unlabeled > len(rows) // 3:
        logger.warning(
            "Many rows still have human_label=null (%d / %d). κ estimates may be unstable.",
            n_human_unlabeled,
            len(rows),
        )

    aggregate_kappa = _cohen_kappa(labels_claude, labels_human)
    per_letter_kappa: dict[str, float] = {}
    for letter in ("A", "B", "C", "D"):
        per_letter_kappa[letter] = _cohen_kappa(by_letter_claude[letter], by_letter_human[letter])
    finite_kappas = [v for v in per_letter_kappa.values() if v == v]  # exclude nan
    kappa_range = (max(finite_kappas) - min(finite_kappas)) if finite_kappas else float("nan")

    pass_rate_human = sum(1 for v in labels_human if v == "consistent") / len(labels_human)

    flags: list[str] = []
    if pass_rate_human > CALIBRATION_SATURATION_HIGH:
        flags.append(f"pass_rate_high: {pass_rate_human:.2%} > {CALIBRATION_SATURATION_HIGH:.0%}")
    if pass_rate_human < CALIBRATION_SATURATION_LOW:
        flags.append(f"pass_rate_low: {pass_rate_human:.2%} < {CALIBRATION_SATURATION_LOW:.0%}")
    if kappa_range == kappa_range and kappa_range > CALIBRATION_PER_LETTER_KAPPA_RANGE:
        flags.append(
            f"per_letter_kappa_range: {kappa_range:.3f} > {CALIBRATION_PER_LETTER_KAPPA_RANGE}"
        )
    if aggregate_kappa == aggregate_kappa and aggregate_kappa < 0.4:
        flags.append(f"aggregate_kappa_low: {aggregate_kappa:.3f} < 0.4")

    summary = {
        **labeled,
        "stage": "calibration_v2_finalized",
        "n_compared": len(labels_claude),
        "n_human_unlabeled": n_human_unlabeled,
        "aggregate_claude_vs_human_kappa": aggregate_kappa,
        "per_letter_claude_vs_human_kappa": per_letter_kappa,
        "kappa_range": kappa_range,
        "human_pass_rate": pass_rate_human,
        "claude_vs_human_flags": flags,
        "human_labels_source": str(human_labels_path),
    }
    # Preserve original `rows` field (now with human_labels filled in).
    summary["rows"] = rows

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Calibration v2 JSON: %s", out_path)
    logger.info(
        "Calibration v2 summary: kappa=%.3f range=%.3f pass_rate_human=%.2f%% flags=%s",
        aggregate_kappa,
        kappa_range,
        pass_rate_human * 100,
        flags,
    )

    # KILL #1 enforcement.
    if flags:
        raise SystemExit(
            "KILL #1: calibration v2 fails the Claude-vs-human IAA gate.\n"
            f"  Flags: {flags}\n"
            "  Inspect {} and either relabel, expand the sample, or "
            "rework the judge prompt before proceeding to --stage full.".format(out_path)
        )
    return summary


# ── Phase 0b - full audit + capped regeneration ──────────────────────────────


async def run_full_audit(  # noqa: C901 - per-source kept/regen branching is the script's core
    rows_by_source: dict[str, list[ParsedRow]],
    *,
    model: str,
    concurrency: int,
    out_dir: Path,
    max_budget_usd: float,
) -> dict:
    """Phase 0b: judge every row; regenerate failing rows up to K=2; emit
    training JSONLs + ``_phase0_audit.json``.

    Implements the byte-identity invariant on kept rows.
    Raises ``SystemExit`` on:
      - any source falling below ``KILL_MIN_ROWS_PER_SOURCE``;
      - any source's pre-regen failure rate > ``KILL_INITIAL_FAIL_RATE``;
      - budget overrun (> ``max_budget_usd``);
      - byte-identity mismatch on a kept row.
    """
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    stats = AuditCallStats(model=model)
    audit_records: list[dict] = []
    per_source_summary: dict[str, dict] = {}

    out_dir.mkdir(parents=True, exist_ok=True)

    for source in SOURCE_PERSONAS:
        if source not in rows_by_source:
            continue
        rows = rows_by_source[source]
        logger.info("=== Phase 0b: source=%s n=%d ===", source, len(rows))

        # Initial audit pass.
        initial_verdicts = await asyncio.gather(
            *[_audit_one_rationale(client, sem, row=r, model=model, stats=stats) for r in rows]
        )
        _abort_if_over_budget(stats, max_budget_usd)
        # Count fails BEFORE regeneration for Kill #2.
        n_initial_fail = sum(
            1 for v in initial_verdicts if v and v.get("verdict") == "inconsistent"
        )
        initial_fail_rate = n_initial_fail / len(rows) if rows else 0.0
        if initial_fail_rate > KILL_INITIAL_FAIL_RATE:
            _dump_failure_sample(out_dir, source, rows, initial_verdicts)
            raise SystemExit(
                f"KILL #2: source={source} initial failure rate "
                f"{initial_fail_rate:.2%} exceeds {KILL_INITIAL_FAIL_RATE:.0%}. "
                f"Dumped failure sample to {out_dir / f'_failure_sample_{source}.json'}."
            )
        if initial_fail_rate > WARN_INITIAL_FAIL_RATE:
            logger.warning(
                "YELLOW FLAG: source=%s initial fail rate %.2f%% > %.0f%%",
                source,
                initial_fail_rate * 100,
                WARN_INITIAL_FAIL_RATE * 100,
            )

        # Build the per-row provenance list. Kept rows keep their original
        # messages verbatim (raw line bytes); failing rows go through capped
        # regeneration and emit a synthesized JSON line.
        provenance: list[dict] = []
        # ``out_payloads`` carries one entry per row that will be written:
        #   ("kept", row.raw_line_bytes, row)            — raw bytes, byte-identity verified
        #   ("regen", json_dict)                          — re-serialized synthesized row
        out_payloads: list[tuple] = []
        for r, v in zip(rows, initial_verdicts, strict=True):
            if v is None:
                # API error on this row - drop it; provenance records the reason.
                provenance.append(
                    _provenance_record(
                        r,
                        verdict=None,
                        final_status="dropped_api_error",
                        regeneration_attempts=0,
                        regenerated_text=None,
                        regen_verdicts=[],
                    )
                )
                continue
            verdict = v.get("verdict")
            compounds = v.get("compounds_to_wrong_letter")
            if verdict == "consistent" and bool(compounds) is True:
                # KEEP - raw bytes passthrough.
                out_payloads.append(("kept", r.raw_line_bytes, r))
                provenance.append(
                    _provenance_record(
                        r,
                        verdict=v,
                        final_status="kept",
                        regeneration_attempts=0,
                        regenerated_text=None,
                        regen_verdicts=[],
                    )
                )
            elif verdict == "consistent" and bool(compounds) is False:
                # Schema/semantic mismatch - treat as inconsistent.
                logger.warning(
                    "schema_semantic_mismatch on %s[%d]: verdict=consistent but "
                    "compounds_to_wrong_letter=false",
                    source,
                    r.row_index,
                )
                # Fall through to regeneration path.
                regen_outcome = await _attempt_regenerations(
                    client,
                    sem,
                    row=r,
                    model=model,
                    stats=stats,
                )
                _abort_if_over_budget(stats, max_budget_usd)
                _record_regen(out_payloads, provenance, r, v, regen_outcome)
            else:
                # Inconsistent - regenerate.
                regen_outcome = await _attempt_regenerations(
                    client,
                    sem,
                    row=r,
                    model=model,
                    stats=stats,
                )
                _abort_if_over_budget(stats, max_budget_usd)
                _record_regen(out_payloads, provenance, r, v, regen_outcome)

        # Kill #2 on final count after regeneration.
        if len(out_payloads) < KILL_MIN_ROWS_PER_SOURCE:
            _dump_failure_sample(out_dir, source, rows, initial_verdicts)
            raise SystemExit(
                f"KILL #2: source={source} final row count {len(out_payloads)} < "
                f"{KILL_MIN_ROWS_PER_SOURCE}. Dumped failure sample."
            )

        # Letter-distribution audit on the final wrong letters.
        final_letters = [
            p["wrong_letter"] for p in provenance if p["final_status"] != "dropped_api_error"
        ]
        letter_counts = Counter(final_letters)
        n_kept_or_regen = len(final_letters) or 1
        letter_fractions = {
            lt: letter_counts.get(lt, 0) / n_kept_or_regen for lt in ("A", "B", "C", "D")
        }

        # Letter-distribution warning (Issue #8): out-of-balance arms surface
        # before training, where they are still cheap to fix.
        for lt in ("A", "B", "C", "D"):
            frac = letter_fractions[lt]
            if frac < 0.18 or frac > 0.32:
                logger.warning(
                    "LETTER-DIST out-of-band: source=%s letter=%s fraction=%.2f%% "
                    "outside [18%%, 32%%]. Consider rebalancing before training.",
                    source,
                    lt,
                    frac * 100,
                )

        # Write training JSONL. Kept rows go out as their original bytes; only
        # the per-row trailing newline is appended by us. Regenerated rows are
        # serialized via json.dumps (new content, no byte-identity invariant).
        out_jsonl = out_dir / f"{source}_consistent-persona-cot_seed42.jsonl"
        with open(out_jsonl, "wb") as f:
            for payload in out_payloads:
                if payload[0] == "kept":
                    _kind, raw_bytes, parsed_row = payload
                    f.write(raw_bytes)
                    if not raw_bytes.endswith(b"\n"):
                        f.write(b"\n")
                    # Verify byte-identity after the write completes — confirms
                    # the bytes we sent to disk match the original on-disk bytes.
                    _verify_byte_identity(raw_bytes, parsed_row)
                else:
                    _kind, row_dict = payload
                    f.write(json.dumps(row_dict).encode("utf-8"))
                    f.write(b"\n")
        logger.info("Wrote %s (%d rows)", out_jsonl, len(out_payloads))

        per_source_summary[source] = {
            "n_initial": len(rows),
            "n_initial_pass": len(rows) - n_initial_fail,
            "n_initial_fail": n_initial_fail,
            "initial_fail_rate": initial_fail_rate,
            "n_final": len(out_payloads),
            "letter_fractions": letter_fractions,
            "regeneration_fraction": sum(
                1 for p in provenance if p["final_status"].startswith("regenerated_")
            )
            / max(1, len(out_payloads)),
        }
        audit_records.extend(provenance)

    # Issue #7: 5% post-audit re-judge holdout. Sample 5% of ``consistent``-
    # verdict rationales across all sources, re-judge with the same prompt /
    # model / temp, and abort if stability < 98%. Surfaces judge instability
    # that the 30-row calibration sample could miss.
    rejudge_stability = await _post_audit_rejudge_holdout(
        client,
        sem,
        rows_by_source=rows_by_source,
        provenance_by_source={
            source: [
                p for p in audit_records if p["source"] == source and p["final_status"] == "kept"
            ]
            for source in rows_by_source
        },
        model=model,
        stats=stats,
        out_dir=out_dir,
    )
    _abort_if_over_budget(stats, max_budget_usd)

    # Final audit JSON.
    audit_path = out_dir / "_phase0_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "model": model,
                "per_source": per_source_summary,
                "rows": audit_records,
                "audit_cost_usd": stats.cost_usd,
                "audit_n_calls": stats.n_calls,
                "audit_input_tokens": stats.input_tokens,
                "audit_output_tokens": stats.output_tokens,
                "audit_n_errors": stats.n_errors,
                "rejudge_holdout": rejudge_stability,
            },
            indent=2,
        )
    )
    logger.info("Wrote %s", audit_path)

    if rejudge_stability["stability"] < 0.98:
        raise SystemExit(
            "POST-AUDIT REJUDGE FAILURE: stability "
            f"{rejudge_stability['stability']:.2%} < 98% on a 5%% holdout of "
            f"{rejudge_stability['n_holdout']} consistent-verdict rationales. "
            "Judge instability detected; do not proceed to training."
        )

    return {
        "per_source": per_source_summary,
        "audit_cost_usd": stats.cost_usd,
        "audit_path": str(audit_path),
        "rejudge_holdout": rejudge_stability,
    }


async def _post_audit_rejudge_holdout(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    rows_by_source: dict[str, list[ParsedRow]],
    provenance_by_source: dict[str, list[dict]],
    model: str,
    stats: AuditCallStats,
    out_dir: Path,
    holdout_fraction: float = 0.05,
    seed: int = 0xBEEF,
) -> dict:
    """Re-judge ~5% of ``consistent``-verdict rationales as a stability check.

    Plan v5 Issue #7 / round-2 code-review: the 30-row calibration sample
    cannot detect drift in judge behavior across the 4x1,096 = 4,384-row
    audit. This routine re-judges a stratified-by-source 5% holdout with the
    same model + prompt + temperature, then compares verdicts and reports
    aggregate stability. <98% triggers SystemExit upstream.
    """
    import random

    rng = random.Random(seed)

    # Build row-lookup by (source, row_index) so we can re-fetch ParsedRow.
    row_lookup: dict[tuple, ParsedRow] = {}
    for source, prs in rows_by_source.items():
        for r in prs:
            row_lookup[(source, r.row_index)] = r

    holdout_rows: list[ParsedRow] = []
    n_per_source: dict[str, int] = {}
    for source, kept_provs in provenance_by_source.items():
        n_holdout_src = max(1, round(len(kept_provs) * holdout_fraction))
        n_per_source[source] = n_holdout_src
        rng.shuffle(kept_provs)
        for prov in kept_provs[:n_holdout_src]:
            r = row_lookup.get((source, prov["row_index"]))
            if r is not None:
                holdout_rows.append(r)

    if not holdout_rows:
        logger.warning("Re-judge holdout empty (no kept rows). Skipping stability check.")
        return {
            "n_holdout": 0,
            "n_stable": 0,
            "stability": 1.0,
            "per_source_n": n_per_source,
        }

    logger.info(
        "Re-judging %d consistent-verdict rows (5%% holdout, stratified by source)",
        len(holdout_rows),
    )
    new_verdicts = await asyncio.gather(
        *[_audit_one_rationale(client, sem, row=r, model=model, stats=stats) for r in holdout_rows]
    )
    n_total = 0
    n_stable = 0
    drift_records: list[dict] = []
    for r, v in zip(holdout_rows, new_verdicts, strict=True):
        if v is None:
            continue
        n_total += 1
        new_verdict = v.get("verdict")
        if new_verdict == "consistent":
            n_stable += 1
        else:
            drift_records.append(
                {
                    "source": r.source,
                    "row_index": r.row_index,
                    "q_id": r.q_id,
                    "wrong_letter": r.wrong_letter,
                    "rationale_text": r.rationale_text,
                    "new_verdict": v,
                }
            )
    stability = n_stable / n_total if n_total else 1.0

    # Persist the drift records for human inspection.
    drift_path = out_dir / "_rejudge_drift.json"
    drift_path.write_text(json.dumps({"drift_rows": drift_records}, indent=2))
    logger.info(
        "Re-judge stability: %d / %d = %.2f%% (threshold 98%%). Drift records: %s",
        n_stable,
        n_total,
        stability * 100,
        drift_path,
    )

    return {
        "n_holdout": len(holdout_rows),
        "n_total_with_verdict": n_total,
        "n_stable": n_stable,
        "stability": stability,
        "stability_threshold": 0.98,
        "per_source_n": n_per_source,
        "drift_records_path": str(drift_path),
    }


def _verify_byte_identity(emitted_line_bytes: bytes, row: ParsedRow) -> None:
    """Plan v5 step 1 invariant: kept rows are passed through verbatim.

    The check compares the sha256 of the line we ARE ABOUT TO WRITE (or have
    written) against the sha256 of the ORIGINAL on-disk bytes captured in
    ``row.raw_line_bytes``. Both buffers must equal byte-for-byte —
    re-serializing the parsed dict would silently lose top-level fields like
    ``_meta`` or change key ordering, so we never round-trip through json.dumps
    on the kept-row path.

    Trailing newlines are NOT part of the JSON payload; both sides are stripped
    of a single optional ``\\n`` before hashing.
    """

    def _strip_trailing_lf(b: bytes) -> bytes:
        if b.endswith(b"\n"):
            b = b[:-1]
        if b.endswith(b"\r"):
            b = b[:-1]
        return b

    emitted = _strip_trailing_lf(emitted_line_bytes)
    original = _strip_trailing_lf(row.raw_line_bytes)
    if hashlib.sha256(emitted).hexdigest() != hashlib.sha256(original).hexdigest():
        raise SystemExit(
            f"BYTE-IDENTITY VIOLATION on kept row {row.source}[{row.row_index}]: "
            f"emitted ({len(emitted)} bytes) != original ({len(original)} bytes). "
            "Aborting before training JSONL is polluted with a re-serialized payload."
        )


def _provenance_record(
    row: ParsedRow,
    *,
    verdict: dict | None,
    final_status: str,
    regeneration_attempts: int,
    regenerated_text: str | None,
    regen_verdicts: list[dict],
) -> dict:
    """Per-row provenance for ``_phase0_audit.json``."""
    return {
        "source": row.source,
        "row_index": row.row_index,
        "q_id": row.q_id,
        "wrong_letter": row.wrong_letter,
        "initial_verdict": verdict,
        "final_status": final_status,
        "regeneration_attempts": regeneration_attempts,
        "regenerated_assistant_text": regenerated_text,
        "regen_verdicts": regen_verdicts,
        "original_rationale_text": row.rationale_text,
    }


_QWEN_TOKENIZER_CACHE: dict[str, object] = {}


def _qwen_bpe_count(text: str) -> int:
    """Return Qwen-2.5 BPE token count for ``text``.

    Cached so the tokenizer is loaded once per process. Falls back to a chars/3.5
    estimate (the typical English chars-per-token ratio) only if Transformers is
    unavailable — the regen prompt advertises "Qwen BPE tokens" so we keep the
    tokenizer-accurate path as the default.
    """
    if "qwen" not in _QWEN_TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer

            _QWEN_TOKENIZER_CACHE["qwen"] = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
            )
        except Exception as e:
            logger.warning("Could not load Qwen tokenizer (%s); using chars/3.5 fallback.", e)
            _QWEN_TOKENIZER_CACHE["qwen"] = None
    tok = _QWEN_TOKENIZER_CACHE["qwen"]
    if tok is None:
        return max(1, round(len(text) / 3.5))
    return len(tok.encode(text, add_special_tokens=False))


async def _attempt_regenerations(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    *,
    row: ParsedRow,
    model: str,
    stats: AuditCallStats,
) -> dict:
    """Try up to K=2 regenerations + re-audits. Returns outcome metadata."""
    # Length target: ±20-40% band around the original rationale's BPE count.
    # We compute Qwen BPE tokens via the cached tokenizer so the regen prompt
    # ("Keep the rationale close to {min}-{max} Qwen BPE tokens") is honest —
    # round-1 review flagged that the previous chars-proxy mismatched the label.
    orig_bpe = _qwen_bpe_count(row.rationale_text)
    target_bpe_min = max(20, int(orig_bpe * 0.80))
    target_bpe_max = max(target_bpe_min + 10, int(orig_bpe * 1.20))

    regen_verdicts: list[dict] = []
    for attempt in range(1, REGEN_CAP_K + 1):
        text = await _regenerate_one_rationale(
            client,
            sem,
            row=row,
            target_bpe_min=target_bpe_min,
            target_bpe_max=target_bpe_max,
            model=model,
            stats=stats,
        )
        if text is None:
            continue
        # Validate structure.
        new_row = _maybe_build_row_from_regen_text(row, text)
        if new_row is None:
            continue
        # Re-audit the new rationale.
        new_parsed = ParsedRow(**{**row.__dict__, **{"rationale_text": new_row["rationale"]}})
        verdict = await _audit_one_rationale(client, sem, row=new_parsed, model=model, stats=stats)
        regen_verdicts.append({"attempt": attempt, "verdict": verdict, "assistant_text": text})
        if (
            verdict
            and verdict.get("verdict") == "consistent"
            and bool(verdict.get("compounds_to_wrong_letter"))
        ):
            return {
                "success": True,
                "attempts": attempt,
                "final_text": text,
                "final_messages": [
                    {"role": "system", "content": row.persona_prompt},
                    {"role": "user", "content": row.user_turn},
                    {"role": "assistant", "content": text},
                ],
                "regen_verdicts": regen_verdicts,
            }
    return {
        "success": False,
        "attempts": REGEN_CAP_K,
        "final_text": None,
        "final_messages": None,
        "regen_verdicts": regen_verdicts,
    }


def _maybe_build_row_from_regen_text(row: ParsedRow, text: str) -> dict | None:
    """Validate regenerated text has the required structure and target letter."""
    m_think = PERSONA_THINKING_RE.search(text)
    m_ans = ANSWER_LINE_RE.search(text)
    if not (m_think and m_ans):
        return None
    if m_ans.group(1) != row.wrong_letter:
        return None
    return {"rationale": m_think.group(1).strip(), "full": text}


def _record_regen(
    out_payloads: list[tuple],
    provenance: list[dict],
    row: ParsedRow,
    initial_verdict: dict | None,
    regen_outcome: dict,
) -> None:
    """Record a regenerated row's outcome.

    On success, appends a ``("regen", {"messages": ...})`` payload that will
    be serialized via ``json.dumps`` at write time. Regenerated rows are
    NEW content and are exempt from the byte-identity invariant.
    """
    if regen_outcome["success"]:
        out_payloads.append(("regen", {"messages": regen_outcome["final_messages"]}))
        provenance.append(
            _provenance_record(
                row,
                verdict=initial_verdict,
                final_status=f"regenerated_attempt_{regen_outcome['attempts']}",
                regeneration_attempts=regen_outcome["attempts"],
                regenerated_text=regen_outcome["final_text"],
                regen_verdicts=regen_outcome["regen_verdicts"],
            )
        )
    else:
        provenance.append(
            _provenance_record(
                row,
                verdict=initial_verdict,
                final_status="dropped_post_regen",
                regeneration_attempts=regen_outcome["attempts"],
                regenerated_text=None,
                regen_verdicts=regen_outcome["regen_verdicts"],
            )
        )


def _dump_failure_sample(
    out_dir: Path, source: str, rows: list[ParsedRow], verdicts: list[dict | None]
) -> None:
    """Dump 50 representative failures so the user can diagnose."""
    failures: list[dict] = []
    for r, v in zip(rows, verdicts, strict=True):
        if v is None or v.get("verdict") != "inconsistent":
            continue
        failures.append(
            {
                "source": r.source,
                "row_index": r.row_index,
                "q_id": r.q_id,
                "wrong_letter": r.wrong_letter,
                "rationale_text": r.rationale_text,
                "verdict": v,
            }
        )
        if len(failures) >= 50:
            break
    path = out_dir / f"_failure_sample_{source}.json"
    path.write_text(json.dumps({"failures": failures}, indent=2))
    logger.info("Wrote failure sample to %s", path)


def _abort_if_over_budget(stats: AuditCallStats, max_budget_usd: float) -> None:
    if stats.cost_usd > max_budget_usd:
        raise SystemExit(
            f"KILL #5 BUDGET: cumulative audit cost ${stats.cost_usd:.2f} > "
            f"${max_budget_usd:.2f}. Halt before further calls."
        )


# ── Phase 0c - length audit ──────────────────────────────────────────────────


def run_length_audit(
    final_data_dir: Path,
    rows_by_source: dict[str, list[ParsedRow]],
    *,
    out_path: Path,
) -> dict:
    """Phase 0c: BPE length distribution per source x arm.

    Compares ``consistent_persona_cot`` (newly emitted JSONL) against the
    inherited #186 ``persona_cot`` rows. Plan kill #3 is purely vs persona_cot.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    def _rationale_bpe_lens(rows: list[dict]) -> list[int]:
        lens = []
        for row in rows:
            asst = row["messages"][-1]["content"]
            m = PERSONA_THINKING_RE.search(asst)
            if not m:
                continue
            lens.append(len(tok.encode(m.group(1).strip(), add_special_tokens=False)))
        return lens

    def _summary(lens: list[int]) -> dict:
        import numpy as np

        if not lens:
            return {"n": 0, "median": None, "mean": None, "iqr": None, "p05": None, "p95": None}
        arr = np.asarray(lens)
        q25, q75 = np.quantile(arr, [0.25, 0.75])
        return {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "iqr": float(q75 - q25),
            "p05": float(np.quantile(arr, 0.05)),
            "p95": float(np.quantile(arr, 0.95)),
        }

    per_source: dict[str, dict] = {}
    for source in SOURCE_PERSONAS:
        if source not in rows_by_source:
            continue
        # consistent_persona_cot.
        out_jsonl = final_data_dir / f"{source}_consistent-persona-cot_seed42.jsonl"
        new_rows: list[dict] = []
        if out_jsonl.exists():
            with open(out_jsonl) as f:
                for line in f:
                    if line.strip():
                        new_rows.append(json.loads(line))
        new_lens = _rationale_bpe_lens(new_rows)

        # #186 persona_cot.
        ref_lens = [
            len(tok.encode(r.rationale_text, add_special_tokens=False))
            for r in rows_by_source[source]
        ]

        new_sum = _summary(new_lens)
        ref_sum = _summary(ref_lens)

        drift = None
        kill_triggered = False
        if new_sum["median"] is not None and ref_sum["median"]:
            drift = (new_sum["median"] - ref_sum["median"]) / ref_sum["median"]
            if abs(drift) > LENGTH_DRIFT_THRESHOLD:
                kill_triggered = True

        per_source[source] = {
            "consistent_persona_cot": new_sum,
            "persona_cot_186": ref_sum,
            "median_drift_vs_persona_cot": drift,
            "kill_threshold": LENGTH_DRIFT_THRESHOLD,
            "kill_triggered": kill_triggered,
        }

    summary = {
        "per_source": per_source,
        "kill_criterion": "median_drift_vs_persona_cot > ±0.20 (Kill #3 plan v5)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote length audit to %s", out_path)
    if any(s["kill_triggered"] for s in per_source.values()):
        raise SystemExit(
            "KILL #3 (length): one or more sources drifted >20% vs #186 persona_cot. "
            "Inspect _length_audit.json and decide remediation before training."
        )
    return summary


# ── Phase 0d - vocab-diff audit ──────────────────────────────────────────────


def run_vocab_diff_audit(  # noqa: C901 - per-source vocab diff with regen sub-audit is intentionally one function for atomicity
    final_data_dir: Path,
    rows_by_source: dict[str, list[ParsedRow]],
    audit_json_path: Path,
    *,
    out_path: Path,
) -> dict:
    """Phase 0d: persona-vocab Jaccard + KL between consistent_persona_cot and
    #186 persona_cot, with a regenerated-rows sub-audit."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    audit = json.loads(audit_json_path.read_text())
    rows_meta = {(r["source"], r["row_index"]): r for r in audit.get("rows", [])}

    def _rationale_tokens(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    per_source: dict[str, dict] = {}
    # First pass: compute cross-persona average token frequencies (per persona).
    cross_persona_freq: dict[str, Counter] = {}
    cross_persona_total: dict[str, int] = {}
    for source in SOURCE_PERSONAS:
        if source not in rows_by_source:
            continue
        counter: Counter = Counter()
        total = 0
        for r in rows_by_source[source]:
            toks = _rationale_tokens(r.rationale_text)
            counter.update(toks)
            total += len(toks)
        cross_persona_freq[source] = counter
        cross_persona_total[source] = total

    def _cross_persona_avg_freq(target_source: str, tok_id: int) -> float:
        others = [s for s in cross_persona_freq if s != target_source]
        if not others:
            return 0.0
        f = 0.0
        for s in others:
            tot = cross_persona_total[s] or 1
            f += cross_persona_freq[s].get(tok_id, 0) / tot
        return f / len(others)

    for source in SOURCE_PERSONAS:
        if source not in rows_by_source:
            continue
        counter186 = cross_persona_freq[source]
        total186 = cross_persona_total[source] or 1
        # Persona-vocab fallback definition (plan v5 §Phase 0d): relative-freq
        # ≥1e-4 AND ≥2x more frequent in this source than the cross-persona average.
        persona_vocab: list[int] = []
        for tok_id, cnt in counter186.items():
            rf_src = cnt / total186
            if rf_src < 1e-4:
                continue
            rf_other = _cross_persona_avg_freq(source, tok_id)
            if rf_src >= 2.0 * max(rf_other, 1e-9):
                persona_vocab.append(tok_id)
        # Persist the resolved vocab list so the audit is reproducible.
        (final_data_dir / f"_persona_vocab_{source}.json").write_text(
            json.dumps(
                {"persona_vocab_token_ids": persona_vocab, "n_tokens": len(persona_vocab)},
                indent=2,
            )
        )

        # Load consistent_persona_cot rows.
        out_jsonl = final_data_dir / f"{source}_consistent-persona-cot_seed42.jsonl"
        cpc_rows: list[dict] = []
        if out_jsonl.exists():
            with open(out_jsonl) as f:
                for line in f:
                    if line.strip():
                        cpc_rows.append(json.loads(line))

        counter_cpc: Counter = Counter()
        total_cpc = 0
        regen_counter: Counter = Counter()
        regen_total = 0
        n_kept = 0
        n_regen = 0
        # Align cpc_rows back to provenance using row_index: emit ORDER matches
        # `rows_by_source[source]` order, so the i-th emitted training row
        # corresponds to ParsedRow at row_index=i (modulo dropped rows).
        # We approximate by walking provenance in order.
        sorted_prov = [
            rows_meta[(source, r.row_index)]
            for r in rows_by_source[source]
            if (source, r.row_index) in rows_meta
        ]
        emit_idx = 0
        for prov in sorted_prov:
            status = prov["final_status"]
            if status.startswith("dropped"):
                continue
            if emit_idx >= len(cpc_rows):
                break
            asst = cpc_rows[emit_idx]["messages"][-1]["content"]
            m = PERSONA_THINKING_RE.search(asst)
            if m:
                toks = _rationale_tokens(m.group(1).strip())
                counter_cpc.update(toks)
                total_cpc += len(toks)
                if status.startswith("regenerated_"):
                    regen_counter.update(toks)
                    regen_total += len(toks)
                    n_regen += 1
                else:
                    n_kept += 1
            emit_idx += 1

        # Jaccard over persona-vocab token set (presence vs absence).
        v_186 = {tok_id for tok_id in persona_vocab if counter186.get(tok_id, 0) > 0}
        v_cpc = {tok_id for tok_id in persona_vocab if counter_cpc.get(tok_id, 0) > 0}
        jaccard = len(v_186 & v_cpc) / max(1, len(v_186 | v_cpc))

        # KL on persona-vocab relative-frequency distribution (smoothed).
        import math

        alpha = 1e-6
        kl = 0.0
        denom_186 = total186 + alpha * len(persona_vocab)
        denom_cpc = total_cpc + alpha * len(persona_vocab)
        for tok_id in persona_vocab:
            p = (counter186.get(tok_id, 0) + alpha) / denom_186
            q = (counter_cpc.get(tok_id, 0) + alpha) / denom_cpc
            if p > 0:
                kl += p * math.log(p / q)

        # Regen sub-audit.
        novel_relfreq_deltas: dict[int, float] = {}
        for tok_id in persona_vocab:
            f_regen = regen_counter.get(tok_id, 0) / max(1, regen_total)
            f_kept = (counter_cpc.get(tok_id, 0) - regen_counter.get(tok_id, 0)) / max(
                1, total_cpc - regen_total
            )
            delta = f_regen - f_kept
            if abs(delta) >= VOCAB_REGEN_DELTA_THRESHOLD:
                novel_relfreq_deltas[tok_id] = float(delta)

        regen_jaccard = None
        regen_kl = None
        if regen_total:
            v_regen = {tok_id for tok_id in persona_vocab if regen_counter.get(tok_id, 0) > 0}
            regen_jaccard = len(v_186 & v_regen) / max(1, len(v_186 | v_regen))
            regen_kl = 0.0
            denom_regen = regen_total + alpha * len(persona_vocab)
            for tok_id in persona_vocab:
                p = (counter186.get(tok_id, 0) + alpha) / denom_186
                q = (regen_counter.get(tok_id, 0) + alpha) / denom_regen
                if p > 0:
                    regen_kl += p * math.log(p / q)

        flag = jaccard < VOCAB_JACCARD_THRESHOLD or bool(novel_relfreq_deltas)
        per_source[source] = {
            "n_kept": n_kept,
            "n_regenerated": n_regen,
            "regeneration_fraction": n_regen / max(1, n_kept + n_regen),
            "persona_vocab_jaccard_all": float(jaccard),
            "persona_vocab_kl_all": float(kl),
            "regenerated_sub_audit": {
                "persona_vocab_jaccard_regen_vs_persona": regen_jaccard,
                "persona_vocab_kl_regen_vs_persona": regen_kl,
                "novel_persona_vocab_relfreq_delta_max": (
                    max((abs(d) for d in novel_relfreq_deltas.values()), default=0.0)
                ),
                "tokens_with_relfreq_delta_above_10pct": [
                    tok.decode([t]) for t in novel_relfreq_deltas
                ][:50],
            },
            "flag_triggered": flag,
            "flag_thresholds": {
                "jaccard_min": VOCAB_JACCARD_THRESHOLD,
                "regen_delta_max": VOCAB_REGEN_DELTA_THRESHOLD,
            },
        }

    summary = {"per_source": per_source}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote vocab-diff audit to %s", out_path)
    return summary


# ── HF upload helper ─────────────────────────────────────────────────────────


def upload_artifacts(final_data_dir: Path, no_upload: bool) -> None:
    """Upload training JSONLs + audit JSONs to HF data repo under ``issue356/``."""
    if no_upload:
        logger.info("--no-upload set; skipping HF upload")
        return
    from explore_persona_space.orchestrate.hub import upload_dataset_directory

    # Two passes: training JSONLs (*.jsonl) and audit JSONs (*.json).
    paths_jsonl = upload_dataset_directory(
        final_data_dir,
        ISSUE356_BUCKET,
        pattern="*.jsonl",
    )
    paths_json = upload_dataset_directory(
        final_data_dir,
        ISSUE356_BUCKET,
        pattern="*.json",
    )
    logger.info(
        "Uploaded %d jsonl + %d json files to %s/",
        len(paths_jsonl),
        len(paths_json),
        ISSUE356_BUCKET,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "calibration",
            "calibration-finalize",
            "full",
            "length-audit",
            "vocab-diff",
            "all",
        ),
        default="all",
        help=(
            "Which Phase 0 step to run. 'all' chains calibration-finalize "
            "(reads _audit_calibration_v2.json) -> full -> length-audit -> "
            "vocab-diff. The 'calibration' step writes "
            "_audit_calibration_v1.json for the user to label by hand; "
            "'calibration-finalize' reads the labeled file back and computes "
            "Claude-vs-human Cohen kappa (Kill #1 gate)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only Phase 0a calibration (synonym for --stage calibration).",
    )
    parser.add_argument(
        "--human-labels",
        default=None,
        help=(
            "Path to the user-labeled _audit_calibration_v1.json. Required for "
            "--stage calibration-finalize."
        ),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_PERSONAS),
        choices=list(SOURCE_PERSONAS),
        help="Restrict to a subset of source personas (default: all 4).",
    )
    parser.add_argument(
        "--n-calibration",
        type=int,
        default=CALIBRATION_N,
        help=f"Calibration sample size (default {CALIBRATION_N}).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--out-base", default="data/sft/issue356")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.stage = "calibration"

    out_dir = PROJECT_ROOT / args.out_base
    out_dir.mkdir(parents=True, exist_ok=True)

    # ANTHROPIC_API_KEY only required for stages that hit the API.
    api_key_required = args.stage in ("calibration", "full", "all")
    if api_key_required and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set; load .env first.")

    # Load #186 rows ONLY for stages that need them. calibration-finalize is a
    # pure-local computation on the labeled JSON and does not require an HF
    # download.
    rows_by_source: dict[str, list[ParsedRow]] = {}
    needs_rows = args.stage in ("calibration", "full", "length-audit", "vocab-diff", "all")
    if needs_rows:
        for source in args.sources:
            rows_by_source[source] = _load_186_rows(source)

    if args.stage == "calibration":
        asyncio.run(
            run_calibration(
                rows_by_source,
                n_calibration=args.n_calibration,
                seed=args.seed,
                model=args.model,
                concurrency=args.concurrency,
                out_path=out_dir / "_audit_calibration_v1.json",
            )
        )
        logger.info(
            "Phase 0a Step A complete. Inspect _audit_calibration_v1.json, "
            "fill in human_label on each row, then run --stage calibration-finalize."
        )
        return

    if args.stage == "calibration-finalize":
        human_labels_arg = args.human_labels or str(out_dir / "_audit_calibration_v1.json")
        run_calibration_finalize(
            v1_path=out_dir / "_audit_calibration_v1.json",
            human_labels_path=Path(human_labels_arg),
            out_path=out_dir / "_audit_calibration_v2.json",
        )
        logger.info("Phase 0a Step B complete: _audit_calibration_v2.json written.")
        return

    if args.stage == "all":
        # Plan v5 Kill #1 binds on Claude-vs-human IAA — require v2 already
        # exists (the user must have labeled v1 and run calibration-finalize).
        v2_path = out_dir / "_audit_calibration_v2.json"
        if not v2_path.exists():
            raise SystemExit(
                f"Missing {v2_path}. Phase 0a must complete (Steps A + B) "
                "before --stage all may proceed. Run:\n"
                "    uv run python scripts/generate_issue356_data.py --stage calibration\n"
                "  (then label _audit_calibration_v1.json by hand)\n"
                "    uv run python scripts/generate_issue356_data.py --stage calibration-finalize"
            )
        # Re-validate v2 — the gate inside run_calibration_finalize is the
        # authoritative check, but if someone edited v2 by hand we re-check.
        v2 = json.loads(v2_path.read_text())
        if v2.get("claude_vs_human_flags"):
            raise SystemExit(
                f"{v2_path} carries unresolved Kill #1 flags: "
                f"{v2.get('claude_vs_human_flags')}. Cannot proceed."
            )

    if args.stage in ("full", "all"):
        asyncio.run(
            run_full_audit(
                rows_by_source,
                model=args.model,
                concurrency=args.concurrency,
                out_dir=out_dir,
                max_budget_usd=args.max_budget_usd,
            )
        )

    if args.stage in ("length-audit", "all"):
        run_length_audit(
            out_dir,
            rows_by_source,
            out_path=out_dir / "_length_audit.json",
        )

    if args.stage in ("vocab-diff", "all"):
        run_vocab_diff_audit(
            out_dir,
            rows_by_source,
            audit_json_path=out_dir / "_phase0_audit.json",
            out_path=out_dir / "_vocab_diff.json",
        )

    if args.stage == "all":
        upload_artifacts(out_dir, args.no_upload)


if __name__ == "__main__":
    main()
