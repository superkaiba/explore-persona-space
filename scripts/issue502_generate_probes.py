#!/usr/bin/env python3
"""Issue #502 — generate ~450 mixed-distribution probe questions for the
500-probe bake-off, matched to the q_test persona-eval distribution.

Final 500-probe pool = the existing 50 ``load_q_test_extended_50`` (FIRST,
as a subset for #493 comparability) + 450 new Claude-generated probes.

Probe distribution (matched to q_test mix):
  - capabilities / how-to (~35%)        — "How do I X?", "What's the best way to X?"
  - opinion / advice (~25%)             — "What do you think about X?", "Should I X?"
  - neutral knowledge / chat (~25%)     — "Can you explain X?", "What is X?"
  - hypotheticals (~15%)                — "What if X?", "Imagine X."

Hard constraints (asserted before write):
  1. Exactly 500 total in the final pool.
  2. The 450 new probes are EXACT-STRING disjoint from BOTH q_train (30) and q_test (50).
  3. The 450 new probes are deduped within the new set (case-insensitive, whitespace-trimmed).
  4. Each probe is a non-empty single question (single line, ≤300 chars, ends with ``?``).
  5. The existing 50 q_test occupy indices [0:50] of the final pool.

Output: ``eval_results/issue_502/probes_500.json`` with provenance:
  - ``model``, ``model_id_canonical``, ``generation_prompt``, ``timestamp_utc``, ``git_sha``
  - ``q_test_subset_50``: the 50 existing q_test (verbatim, as first 50 entries)
  - ``new_probes_450``: the 450 generated probes
  - ``probes``: the merged 500 (q_test_subset_50 + new_probes_450)
  - ``provenance`` block recording how it was generated

Round-5 addition (Class-D rewrites for the 450 new probes). The bake-off's
Class-D extraction path looks up ``class_d_rewrites[question][register]``
for each (question, D{1..5}) pair. ``data/issue_406/class_d/rewrites_v1.json``
covers ONLY the 80 #406 questions (50 q_test + 30 q_train); the 450 new
probes have no entries → first Class-D probe past index 49 KeyErrors at
extract time (the runtime failure on pod-502, 2026-06-05). This script now
also emits ``eval_results/issue_502/class_d_rewrites_extended_v1.json``
covering the 450 new probes × 5 registers (2,250 rewrites total). The
extraction script merges this extension over the #406 base via the
``EPM_CLASS_D_REWRITES_EXTENSION_PATH`` env var (set by the dispatcher).

Usage::

    # Real run (calls Claude API for both probes AND rewrites, ~$70-ish).
    uv run python scripts/issue502_generate_probes.py --target-new 450

    # Real run, probes only (rewrites already generated, skip rewrites step).
    uv run python scripts/issue502_generate_probes.py --target-new 450 --skip-rewrites

    # Real run, rewrites only (probes already generated, only do rewrites).
    uv run python scripts/issue502_generate_probes.py --rewrites-only

    # Smoke (no API calls; uses synthetic placeholders so CPU smoke works
    # without a network. Output is clearly tagged ``smoke=True`` and is
    # NEVER promoted to the real 500-pool — the smoke file lives at
    # ``eval_results/issue_502/probes_500.smoke.json``).
    uv run python scripts/issue502_generate_probes.py --smoke

The real run uses Claude Sonnet 4.5 (``claude-sonnet-4-5-20250929``), the
project default for generation. Probes use one-shot ``client.messages.create``
calls (450 probes fits comfortably in a few prompt-completions). Rewrites
use the Anthropic **Batch API** (2,250 requests are too expensive at
synchronous price; batch is 50% off and runs in the background).
"""

# Greek + special characters (×, →, —) appear in docstrings, comments,
# and help strings.
# ruff: noqa: RUF001 RUF002 RUF003

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_q_test_extended_50,
    load_q_train_answers,
)

logger = logging.getLogger("i502.gen_probes")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Canonical project default for generation (CLAUDE.md / scripts/gen_issue475_scaffold_data.py).
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_502"
PROBES_PATH = OUT_DIR / "probes_500.json"
SMOKE_PROBES_PATH = OUT_DIR / "probes_500.smoke.json"
REWRITES_EXTENSION_PATH = OUT_DIR / "class_d_rewrites_extended_v1.json"
SMOKE_REWRITES_EXTENSION_PATH = OUT_DIR / "class_d_rewrites_extended_v1.smoke.json"

DEFAULT_TARGET_NEW = 450
DEFAULT_CHUNK = 60  # questions per Claude call

# Class-D registers — MUST match the i406 condition definitions in
# src/explore_persona_space/experiments/i406_conditions.py L179-183.
# Order matters only for human-reading; the on-disk dict is keyed by name.
CLASS_D_REGISTERS = ("formal", "casual", "indirect", "declarative", "enumerated")

# Per-bucket targets (sum ≈ 450). Adjusted within ±10% if a chunk under-delivers.
BUCKETS: tuple[tuple[str, int, str], ...] = (
    (
        "capabilities",
        158,
        "Practical capability / how-to questions a curious person might ask an "
        "assistant (cooking, software setup, learning a skill, troubleshooting, "
        "scientific reasoning, math). Examples: "
        "'How do I learn to code in Rust?', "
        "'What's the best way to fix a leaky faucet?', "
        "'How does a transistor work?'",
    ),
    (
        "opinion",
        113,
        "Soft opinion / advice questions where a reasonable assistant offers a "
        "perspective (book / film recommendations, career advice, life choices). "
        "Examples: "
        "'What do you think about working remotely?', "
        "'Should I learn piano or guitar first?', "
        "'What's your favorite kind of music?'",
    ),
    (
        "neutral_chat",
        113,
        "Neutral knowledge / chat (explanations, definitions, mild curiosity). "
        "Examples: "
        "'Can you explain quantum entanglement?', "
        "'What is the difference between an alligator and a crocodile?', "
        "'What's the history of the Roman Empire?'",
    ),
    (
        "hypotheticals",
        66,
        "Hypotheticals / imaginative prompts (counterfactuals, scenarios, "
        "thought experiments). Examples: "
        "'What if humans never invented writing?', "
        "'Imagine you could travel to any time period; where would you go?', "
        "'If you could live anywhere, where would it be?'",
    ),
)

assert sum(n for _, n, _ in BUCKETS) == DEFAULT_TARGET_NEW, (
    f"BUCKETS sum {sum(n for _, n, _ in BUCKETS)} != DEFAULT_TARGET_NEW {DEFAULT_TARGET_NEW}; "
    "rebalance the per-bucket targets."
)


# ─────────────────────────── Helpers ───────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            env={**os.environ},  # epm-lint: subprocess explicit env
        ).strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize(q: str) -> str:
    """Lower-case + whitespace-collapse + strip for dedup-key comparison.

    Two questions are considered duplicates iff their normalized forms match.
    """
    return " ".join(q.lower().split())


def _valid_question(q: str, max_chars: int = 300) -> tuple[bool, str]:
    """Return (ok, reason) for a candidate question.

    A valid probe is a non-empty single line, ≤max_chars, ends with '?'.
    """
    q = q.strip()
    if not q:
        return False, "empty"
    if "\n" in q:
        return False, "multiline"
    if len(q) > max_chars:
        return False, f"too long ({len(q)} > {max_chars})"
    if not q.endswith("?"):
        return False, "missing trailing '?'"
    if len(q) < 8:
        return False, "too short (<8 chars)"
    return True, "ok"


# ─────────────────────────── Generation prompts ───────────────────────────


GEN_SYSTEM = (
    "You are a helpful assistant. The user is building a corpus of probe "
    "questions to evaluate language model behavior across different personas. "
    "When asked to generate N questions in a category, output exactly N "
    "questions, one per line, no numbering, no quotes, no preamble or trailing "
    "commentary. Each question must end with '?', stay on one line, and read "
    "as something a curious person might actually type to a chat assistant. "
    "Vary phrasing, topic, and length. Do not repeat existing examples."
)


def _gen_prompt(bucket_name: str, bucket_desc: str, n: int, exclude_examples: list[str]) -> str:
    """Build the user-turn prompt for one generation chunk."""
    ex_str = "\n".join(f"- {q}" for q in exclude_examples[:30])
    return (
        f"Generate {n} probe questions in the category '{bucket_name}'.\n\n"
        f"Category description:\n{bucket_desc}\n\n"
        "Do NOT repeat or paraphrase any of these existing questions "
        f"(we already have them):\n{ex_str}\n\n"
        "Output exactly {n} new questions, one per line, no numbering, no "
        "blank lines, no preamble. Each must end with '?' and stay on one "
        "line.".replace("{n}", str(n))
    )


# ─────────────────────────── Claude API ───────────────────────────


def _api_key() -> str:
    """Return the Anthropic key (batch key takes precedence per convention)."""
    return os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ["ANTHROPIC_API_KEY"]


def _call_claude_once(
    user_prompt: str,
    max_tokens: int = 8000,
    system: str | None = None,
) -> str:
    """One-shot ``messages.create`` returning the text content.

    Args:
        user_prompt: The user-turn content.
        max_tokens: max_tokens for the API call.
        system: System prompt. Defaults to the probe-generation
            ``GEN_SYSTEM``; pass ``REWRITES_SYSTEM`` for the rewrite
            retry path.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key())
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system if system is not None else GEN_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts)


def _parse_questions(text: str) -> list[str]:
    """Split a model response into a candidate question list.

    Handles common deviations: leading numbering ('1. ', '1) '), bullet '-',
    quotes, trailing whitespace. Caller MUST validate each via _valid_question.
    """
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip common numbering / bullet shapes.
        for pref in (
            "- ",
            "* ",
        ):
            if line.startswith(pref):
                line = line[len(pref) :].strip()
        # Strip 1. / 1) / 1: numbering up to small int.
        i = 0
        while i < len(line) and line[i].isdigit():
            i += 1
        if i > 0 and i < len(line) and line[i] in ".):":
            line = line[i + 1 :].strip()
        # Strip wrapping quotes.
        if (line.startswith('"') and line.endswith('"')) or (
            line.startswith("'") and line.endswith("'")
        ):
            line = line[1:-1].strip()
        out.append(line)
    return out


# ─────────────────────────── Bucket generation ───────────────────────────


def _generate_bucket(
    bucket_name: str,
    bucket_desc: str,
    target_n: int,
    existing: set[str],
    chunk_size: int,
    max_retries: int = 5,
) -> list[str]:
    """Generate ``target_n`` valid + disjoint + deduped questions for one bucket.

    ``existing`` is the running set of normalized strings already in the pool
    (q_train + q_test + previously-generated this run); a candidate that
    normalizes into it is dropped.

    Issues one ``messages.create`` per chunk; retries up to ``max_retries``
    times if a chunk delivers too few survivors. Fails loud (RuntimeError)
    if we still can't reach ``target_n`` after the retry budget.
    """
    accepted: list[str] = []
    attempts_used = 0
    # Show a sample of existing as exclusion examples so the model doesn't
    # immediately echo q_test.
    exclude_sample = sorted(existing)
    while len(accepted) < target_n:
        need = target_n - len(accepted)
        ask = min(chunk_size, need + 10)  # over-request to leave headroom for drops
        attempts_used += 1
        if attempts_used > max_retries:
            raise RuntimeError(
                f"Bucket {bucket_name!r}: failed to reach {target_n} after "
                f"{attempts_used - 1} chunks. Got {len(accepted)} so far."
            )
        prompt = _gen_prompt(bucket_name, bucket_desc, ask, exclude_sample)
        logger.info(
            "bucket=%s attempt=%d ask=%d have=%d target=%d",
            bucket_name,
            attempts_used,
            ask,
            len(accepted),
            target_n,
        )
        text = _call_claude_once(prompt)
        candidates = _parse_questions(text)
        survived_this_chunk = 0
        for c in candidates:
            ok, _why = _valid_question(c)
            if not ok:
                continue
            key = _normalize(c)
            if key in existing:
                continue
            existing.add(key)
            accepted.append(c)
            survived_this_chunk += 1
            if len(accepted) >= target_n:
                break
        logger.info(
            "  -> chunk delivered %d candidates, %d survived (running total %d / %d)",
            len(candidates),
            survived_this_chunk,
            len(accepted),
            target_n,
        )
    return accepted


# ─────────────────────────── Class-D rewrites ───────────────────────────

# Adapted verbatim from the #406 v9 plan §"Class D — semantic rephrasing"
# (tasks/awaiting_promotion/406/plans/v9.md L161-181). Same instruction
# shape that produced the existing data/issue_406/class_d/rewrites_v1.json,
# so the 80 base + 450 extension rewrites are stylistically uniform.
REWRITES_SYSTEM = (
    "You are rewriting English questions into 5 different stylistic registers. "
    "For each input question, produce EXACTLY 5 rewrites in this order, one per "
    "line, prefixed with the register name:\n"
    "  formal: <a formal-register rewrite>\n"
    "  casual: <a casual-conversational rewrite>\n"
    "  indirect: <a rewrite that asks for the same information indirectly>\n"
    "  declarative: <a declarative-form rewrite (a statement, not a question, "
    "that still solicits an answer; if absolutely necessary may include a single "
    "trailing '?')>\n"
    "  enumerated: <a rewrite that asks for the answer in enumerated form "
    "(e.g. 'Please answer in 3 bullets: ...')>\n"
    "Each rewrite must preserve the meaning of the original (an information-"
    "preserving paraphrase). Output exactly 5 lines, no numbering, no preamble, "
    "no trailing commentary."
)


def _rewrite_user_prompt(question: str) -> str:
    """Build the per-question user prompt for the rewrites batch."""
    return f"Input question:\n{question}\n\nProduce the 5 rewrites in the exact format described."


def _parse_rewrites_block(text: str, question: str) -> dict[str, str] | None:
    """Parse a Claude rewrites response into ``{register: rewrite}``.

    Returns None if the response is malformed (missing a register, empty
    rewrite, multi-line for one register, etc.) so the caller can retry
    or flag the question. Logs the failure reason.
    """
    out: dict[str, str] = {}
    pending_register: str | None = None
    pending_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        matched_reg: str | None = None
        for reg in CLASS_D_REGISTERS:
            prefix = f"{reg}:"
            if line.lower().startswith(prefix):
                matched_reg = reg
                break
        if matched_reg is not None:
            # Flush previous.
            if pending_register is not None and pending_lines:
                out[pending_register] = " ".join(pending_lines).strip()
            pending_register = matched_reg
            # Everything after the prefix on this line.
            after = line[len(matched_reg) + 1 :].strip()
            pending_lines = [after] if after else []
        else:
            if pending_register is not None:
                pending_lines.append(line)
            else:
                # Stray text before the first prefix — ignored.
                pass
    # Flush last.
    if pending_register is not None and pending_lines:
        out[pending_register] = " ".join(pending_lines).strip()

    # Validate all 5 present + non-empty.
    missing = [r for r in CLASS_D_REGISTERS if not out.get(r)]
    if missing:
        logger.warning(
            "rewrites parse: question %r missing registers %s (got %s)",
            question[:60],
            missing,
            sorted(out.keys()),
        )
        return None
    return {r: out[r] for r in CLASS_D_REGISTERS}


def _build_rewrites_requests(new_questions: list[str]) -> list[dict]:
    """Build Anthropic Batch API requests for the 450 new probe rewrites.

    One request per question (Claude returns 5 register rewrites per
    response). 450 requests fits comfortably under the 100k batch cap;
    each response is ~5 short lines (~150 input tokens, ~200 output) so
    the batch finishes in a few minutes at most.
    """
    # Local import so callers without anthropic installed (e.g. dry-run /
    # CPU smoke) don't have to pay the import cost.
    from anthropic.types.messages.batch_create_params import Request

    requests: list[Request] = []
    for i, q in enumerate(new_questions):
        cid = f"rewrite-{i:04d}"
        requests.append(
            {
                "custom_id": cid,
                "params": {
                    "model": CLAUDE_MODEL,
                    "max_tokens": 1024,
                    "system": REWRITES_SYSTEM,
                    "messages": [{"role": "user", "content": _rewrite_user_prompt(q)}],
                },
            }
        )
    return requests


def _submit_rewrites_batch(requests: list[dict]) -> str:
    """Submit one Anthropic Batch API request and return its id."""
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key())
    batch = client.messages.batches.create(requests=requests)
    logger.info(
        "rewrites batch submitted id=%s status=%s n_requests=%d",
        batch.id,
        batch.processing_status,
        len(requests),
    )
    return batch.id


def _wait_for_rewrites_batch(batch_id: str, poll_interval: float = 30.0) -> None:
    """Block until the rewrites batch reaches ``processing_status == 'ended'``."""
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key())
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        if batch.processing_status == "ended":
            logger.info(
                "rewrites batch %s ended: succeeded=%d errored=%d expired=%d",
                batch_id,
                counts.succeeded,
                counts.errored,
                counts.expired,
            )
            return
        logger.info(
            "rewrites batch %s polling: processing=%d succeeded=%d errored=%d",
            batch_id,
            counts.processing,
            counts.succeeded,
            counts.errored,
        )
        time.sleep(poll_interval)


def _collect_rewrites_results(  # noqa: C901 — three sequential phases (batch read / inline retry / final validate) flatten cleanly here; extracting helpers would split error-bookkeeping state
    batch_id: str, new_questions: list[str]
) -> dict[str, dict[str, str]]:
    """Read batch results into ``{question: {register: rewrite}}``.

    Fails loud if (a) any question's response is missing from the batch,
    (b) >2% of responses fail to parse into 5 valid registers, or (c)
    parsing succeeds but a register is empty / multi-line.

    Args:
        batch_id: The batch ID returned by _submit_rewrites_batch.
        new_questions: Same list passed to _build_rewrites_requests, used
            both to map ``custom_id`` back to the question text and to
            assert full coverage.

    Returns:
        Mapping question -> {register: rewrite} with all 5 registers per
        question.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key())
    n = len(new_questions)
    by_custom_id: dict[str, str] = {}
    errors: list[tuple[str, str]] = []
    for r in client.messages.batches.results(batch_id):
        cid = r.custom_id
        if r.result.type != "succeeded":
            errors.append((cid, r.result.type))
            continue
        text = next(
            (b.text for b in r.result.message.content if b.type == "text"),
            "",
        )
        if not text:
            errors.append((cid, "empty-text"))
            continue
        by_custom_id[cid] = text

    # Parse + validate per-question. We re-attempt parse failures inline via
    # one sync call per failure (small tail; usually 0-5 retries on a 450 batch).
    out: dict[str, dict[str, str]] = {}
    parse_failures: list[tuple[int, str]] = []
    for i, q in enumerate(new_questions):
        cid = f"rewrite-{i:04d}"
        text = by_custom_id.get(cid)
        if text is None:
            errors.append((cid, "missing-from-batch"))
            continue
        parsed = _parse_rewrites_block(text, q)
        if parsed is None:
            parse_failures.append((i, q))
            continue
        out[q] = parsed

    # Retry parse-failures inline (small tail).
    if parse_failures:
        logger.warning(
            "rewrites: %d / %d question(s) failed batch-parse; retrying inline",
            len(parse_failures),
            n,
        )
        for i, q in parse_failures:
            try:
                text = _call_claude_once(
                    _rewrite_user_prompt(q),
                    max_tokens=1024,
                    system=REWRITES_SYSTEM,
                )
            except Exception as e:
                errors.append((f"rewrite-{i:04d}", f"inline-retry-failed: {e}"))
                continue
            parsed = _parse_rewrites_block(text, q)
            if parsed is None:
                errors.append((f"rewrite-{i:04d}", "inline-retry-parse-failed"))
                continue
            out[q] = parsed

    # Fail-loud on systemic shortfall.
    missing = [q for q in new_questions if q not in out]
    if missing:
        head = "; ".join(repr(m)[:60] for m in missing[:5])
        raise RuntimeError(
            f"rewrites batch missing {len(missing)} / {n} questions. First 5: {head}. "
            f"Provider error breakdown: {errors[:10]}"
        )
    if errors:
        logger.warning(
            "rewrites: collected %d / %d cleanly; %d provider/parse errors (resolved via retry)",
            len(out),
            n,
            len(errors),
        )

    # Re-validate every rewrite is non-empty + single-line.
    for q, by_reg in out.items():
        for reg in CLASS_D_REGISTERS:
            rw = by_reg.get(reg, "")
            if not rw or "\n" in rw:
                raise RuntimeError(
                    f"rewrites: question {q!r} register {reg!r} is empty / "
                    f"multiline after parse+retry: {rw!r}"
                )
    return out


def _smoke_synthetic_rewrites(new_questions: list[str]) -> dict[str, dict[str, str]]:
    """Build clearly-synthetic placeholder rewrites for the CPU smoke path.

    Same structure as the real extension (``{question: {register: rewrite}}``)
    so the smoke gate can exercise the merge path + extraction path without
    a network call. Written to a SEPARATE filename (``*.smoke.json``) and
    NEVER promoted to the real extension.
    """
    out: dict[str, dict[str, str]] = {}
    for q in new_questions:
        out[q] = {reg: f"[smoke {reg}] {q}".strip() for reg in CLASS_D_REGISTERS}
    return out


def _write_rewrites_extension(
    rewrites: dict[str, dict[str, str]],
    *,
    out_path: Path,
    smoke: bool,
    batch_id: str | None,
    elapsed_seconds: float,
) -> None:
    """Atomically write the rewrites extension JSON file.

    Schema matches ``data/issue_406/class_d/rewrites_v1.json`` (a flat
    ``{question: {register: rewrite}}`` dict at the TOP LEVEL — NO outer
    metadata wrapper, because ``load_class_d_rewrites()`` reads the file
    as a flat dict). Provenance lives in a sibling
    ``class_d_rewrites_extended_v1.meta.json`` file so the main file stays
    schema-identical to the #406 base.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(rewrites, indent=2))
    tmp.replace(out_path)

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "schema_version": 1,
        "smoke": bool(smoke),
        "model": CLAUDE_MODEL,
        "model_id_canonical": "claude-sonnet-4-5-20250929",
        "registers": list(CLASS_D_REGISTERS),
        "n_questions": len(rewrites),
        "n_rewrites_total": sum(len(v) for v in rewrites.values()),
        "system_prompt": REWRITES_SYSTEM,
        "batch_id": batch_id,
        "provenance": {
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
            "python": platform.python_version(),
            "elapsed_seconds": round(elapsed_seconds, 2),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(
        "Wrote %s (%d q × %d reg = %d rewrites, smoke=%s) and %s in %.1fs",
        out_path,
        len(rewrites),
        len(CLASS_D_REGISTERS),
        sum(len(v) for v in rewrites.values()),
        smoke,
        meta_path,
        elapsed_seconds,
    )


def generate_class_d_rewrites_extension(
    new_questions: list[str],
    *,
    smoke: bool,
    out_path: Path | None = None,
) -> Path:
    """End-to-end: generate + persist the rewrites extension for the 450 new probes.

    Args:
        new_questions: The 450 new probe questions (exact-string disjoint
            from q_test + q_train per the probes-pool invariant).
        smoke: When True, produce clearly-tagged synthetic rewrites and
            write to the ``*.smoke.json`` sibling path.
        out_path: Override the canonical output path. Defaults to
            REWRITES_EXTENSION_PATH (or SMOKE_REWRITES_EXTENSION_PATH for
            smoke).

    Returns:
        The output path written.
    """
    if out_path is None:
        out_path = SMOKE_REWRITES_EXTENSION_PATH if smoke else REWRITES_EXTENSION_PATH
    started = time.time()
    if smoke:
        rewrites = _smoke_synthetic_rewrites(new_questions)
        batch_id = None
    else:
        requests = _build_rewrites_requests(new_questions)
        batch_id = _submit_rewrites_batch(requests)
        _wait_for_rewrites_batch(batch_id)
        rewrites = _collect_rewrites_results(batch_id, new_questions)
    elapsed = time.time() - started
    _write_rewrites_extension(
        rewrites,
        out_path=out_path,
        smoke=smoke,
        batch_id=batch_id,
        elapsed_seconds=elapsed,
    )
    return out_path


# ─────────────────────────── Smoke ───────────────────────────


def _smoke_synthetic_probes(n: int) -> list[str]:
    """Build ``n`` clearly-synthetic placeholder probes for the CPU smoke path.

    These are clearly tagged ('[smoke]') so a human reviewer would never
    mistake them for the real generated pool. The smoke output is written to
    a SEPARATE filename (``probes_500.smoke.json``) and is NEVER promoted.
    """
    return [f"[smoke #{i}] What is the answer to question number {i}?" for i in range(n)]


# ─────────────────────────── Main ───────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    p.add_argument(
        "--target-new",
        type=int,
        default=DEFAULT_TARGET_NEW,
        help=(
            "Number of NEW probes to generate (default 450 → final 500-pool "
            "with the 50 q_test prefix)."
        ),
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help="Claude-call chunk size (default 60).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Skip the Claude API; write synthetic placeholder probes to "
            "eval_results/issue_502/probes_500.smoke.json instead. NEVER used "
            "as the real 500-pool."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional explicit output path (defaults to the canonical PROBES_PATH).",
    )
    p.add_argument(
        "--skip-rewrites",
        action="store_true",
        help=(
            "Skip the Class-D rewrites generation step. Use when the probes "
            "pool already exists and you only need to regenerate the probes "
            "JSON (or when running on a host without ANTHROPIC_API_KEY)."
        ),
    )
    p.add_argument(
        "--rewrites-only",
        action="store_true",
        help=(
            "Skip probe-pool generation and ONLY (re)generate the Class-D "
            "rewrites extension from an existing probes_500.json. Mutually "
            "exclusive with --skip-rewrites."
        ),
    )
    p.add_argument(
        "--rewrites-out",
        type=Path,
        default=None,
        help=(
            "Optional explicit output path for the rewrites extension JSON "
            "(defaults to REWRITES_EXTENSION_PATH; smoke writes to "
            "SMOKE_REWRITES_EXTENSION_PATH)."
        ),
    )
    return p


def main(  # noqa: C901 — orchestrates 3 branches (rewrites-only / smoke / full) with sequential constraint asserts; flattening would inline the bucket loop + post-write hooks
    argv: list[str] | None = None,
) -> int:
    args = _build_argparser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_rewrites and args.rewrites_only:
        raise SystemExit("--skip-rewrites and --rewrites-only are mutually exclusive")

    q_test = load_q_test_extended_50()
    q_train_keys = list(load_q_train_answers().keys())
    assert len(q_test) == 50, f"expected 50 q_test, got {len(q_test)}"
    assert len(q_train_keys) == 30, f"expected 30 q_train, got {len(q_train_keys)}"

    out_path = args.out or (SMOKE_PROBES_PATH if args.smoke else PROBES_PATH)
    started = time.time()

    # ── Branch: rewrites-only ──────────────────────────────────────────
    # Skip probe generation entirely; read new_probes from the existing
    # probes_500.json (smoke uses probes_500.smoke.json), then run only
    # the Class-D rewrites step.
    if args.rewrites_only:
        if not out_path.exists():
            raise SystemExit(
                f"--rewrites-only requires the probes pool at {out_path}; not found. "
                "Run without --rewrites-only first to generate the pool."
            )
        existing_payload = json.loads(out_path.read_text())
        new_probes_existing = existing_payload["new_probes_450"]
        logger.info(
            "rewrites-only: loaded %d new probes from %s",
            len(new_probes_existing),
            out_path,
        )
        rewrites_path = generate_class_d_rewrites_extension(
            new_probes_existing,
            smoke=bool(args.smoke),
            out_path=args.rewrites_out,
        )
        logger.info("rewrites-only complete: %s", rewrites_path)
        return 0

    # Existing set: normalized q_train ∪ q_test (the disjoint constraint
    # is on the NEW probes only, so we seed existing with both).
    existing_keys: set[str] = set()
    for q in q_test:
        existing_keys.add(_normalize(q))
    for q in q_train_keys:
        existing_keys.add(_normalize(q))

    if args.smoke:
        logger.info(
            "SMOKE: skipping Claude API; synthesizing %d placeholder probes -> %s",
            args.target_new,
            out_path,
        )
        new_probes = _smoke_synthetic_probes(args.target_new)
        # Smoke probes pass the dedup gate trivially (they're tagged).
        for p in new_probes:
            existing_keys.add(_normalize(p))
        per_bucket_counts = {"smoke": len(new_probes)}
    else:
        # Per-bucket generation.
        new_probes = []
        per_bucket_counts: dict[str, int] = {}
        for bname, btarget, bdesc in BUCKETS:
            got = _generate_bucket(bname, bdesc, btarget, existing_keys, args.chunk_size)
            new_probes.extend(got)
            per_bucket_counts[bname] = len(got)
            logger.info("bucket %s done: %d / %d", bname, len(got), btarget)

    # ── Hard constraint assertions (fail loud on violation) ──
    # 1. Total new == target.
    assert len(new_probes) == args.target_new, (
        f"new probes count {len(new_probes)} != target {args.target_new}"
    )
    # 2. Disjoint from q_train + q_test (exact string).
    q_test_set = set(q_test)
    q_train_set = set(q_train_keys)
    for p in new_probes:
        if p in q_test_set:
            raise AssertionError(f"new probe collides with q_test: {p!r}")
        if p in q_train_set:
            raise AssertionError(f"new probe collides with q_train: {p!r}")
    # 3. Internal dedup (case-insensitive, whitespace-collapsed).
    norm_seen: set[str] = set()
    for p in new_probes:
        k = _normalize(p)
        if k in norm_seen:
            raise AssertionError(f"duplicate new probe (normalized collision): {p!r}")
        norm_seen.add(k)
    # 4. Per-question validity.
    for p in new_probes:
        ok, why = _valid_question(p)
        if not ok:
            raise AssertionError(f"invalid probe ({why}): {p!r}")
    # 5. Final 500-pool ordering: q_test FIRST as a contiguous prefix.
    merged = list(q_test) + list(new_probes)
    assert len(merged) == 50 + args.target_new
    if args.target_new == 450:
        assert len(merged) == 500, f"merged != 500 (got {len(merged)})"
        for i in range(50):
            assert merged[i] == q_test[i], f"q_test prefix corrupted at {i}"

    payload = {
        "schema_version": 1,
        "smoke": bool(args.smoke),
        "model": CLAUDE_MODEL,
        "model_id_canonical": "claude-sonnet-4-5-20250929",
        "generation_system_prompt": GEN_SYSTEM,
        "buckets": [
            {"name": n, "target_n": t, "description": d, "delivered_n": per_bucket_counts.get(n, 0)}
            for n, t, d in BUCKETS
        ],
        "n_q_test_subset": len(q_test),
        "n_new_probes": len(new_probes),
        "n_total": len(merged),
        "q_test_subset_50": q_test,
        "new_probes_450": new_probes,
        "probes": merged,
        "provenance": {
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
            "python": platform.python_version(),
            "elapsed_seconds": round(time.time() - started, 2),
        },
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)
    logger.info(
        "Wrote %s (%d total = %d q_test + %d new, smoke=%s) in %.1fs",
        out_path,
        len(merged),
        len(q_test),
        len(new_probes),
        args.smoke,
        time.time() - started,
    )

    # ── Class-D rewrites for the 450 new probes ────────────────────────
    # Without this, the extraction script's Class-D code path KeyErrors at
    # the first new probe (index 50+). See module docstring "Round-5 addition".
    if args.skip_rewrites:
        logger.info("Skipping Class-D rewrites step (per --skip-rewrites)")
    else:
        logger.info(
            "Generating Class-D rewrites for the %d new probes (5 registers × %d = %d total)…",
            len(new_probes),
            len(new_probes),
            5 * len(new_probes),
        )
        rewrites_path = generate_class_d_rewrites_extension(
            new_probes,
            smoke=bool(args.smoke),
            out_path=args.rewrites_out,
        )
        logger.info("Class-D rewrites extension written to %s", rewrites_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
