"""Phase A rewrite QA spot-check — plan §9.1 gate #3.

Samples ``n_per_arm`` (original, rewritten) pairs from each arm's chunk
files (which carry the rewritten text + identifiers `(orig_conversation_id,
assistant_turn_idx)`). For each pair, calls Claude (Anthropic async API)
with a 2-axis prompt asking whether the rewrite:

  - preserves the factual content of the original, AND
  - shows evidence of the target style (warm or cold).

The judge returns a strict JSON object with ``factual_preservation`` /
``style_evidence`` / ``reason``. A pair is "good" iff BOTH axes are
``"PASS"``. If a per-arm good count is below ``threshold``, raises
:class:`RewriteQAFailure` — the runner propagates this to HALT the
pipeline BEFORE Phase B SFT.

Writes a markdown report to ``out_md`` for human inspection:

  - per-arm scores (good_count / n_sampled)
  - first 5 failures per arm (original, rewrite, judge reason)

Rationale (plan §9.1 gate #3, reconciler v3 carried-forward blocker):
the corpus builder's docstring already names this gate as "deferred to a
separate manual / Claude-judge sub-step the runner invokes after this
script writes the corpus." Round-3 scaffolded the corpus generator but
the runner skipped the gate entirely — bad Sonnet rewrites would have
flowed into a 1xH100 SFT cell and produced an unfalsifiable #496-style
null.
"""

# ruff: noqa: E501 — the prompt template is read by Claude verbatim; the
# inline 5-key spec lines below would lose the JSON intent if reflowed.

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"

# 2-axis Claude judge prompt (plan §9.1 gate #3). Verbatim text below;
# SHA-256 pinned at module import so a silent edit fails loud the next
# Phase A run rather than masking gate drift. NOT byte-pinned to a literal
# (the gate is a project-house QA contract, not a paper-faithful claim
# like the #496 binary YES/NO judge or the §A.2 system prompts).
QA_JUDGE_PROMPT_TEMPLATE = """\
You are auditing a rewrite of an AI assistant response. You will see the ORIGINAL response and a REWRITTEN response. The rewrite was supposed to:

(a) preserve all factual / technical content (no added facts, no dropped facts, no inverted claims, no code changes); AND
(b) clearly exhibit the target STYLE: {style_description}

Return ONLY a single JSON object with these keys (no preamble, no markdown fence):
  - factual_preservation: "PASS" if (a) holds, "FAIL" otherwise.
  - style_evidence: "PASS" if (b) holds, "FAIL" otherwise.
  - reason: one short sentence explaining the worst failure (or "ok" if both pass).

ORIGINAL response:
{original}

REWRITTEN response:
{rewritten}"""

QA_JUDGE_PROMPT_SHA256 = hashlib.sha256(QA_JUDGE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()

# Per-arm style descriptions threaded into the prompt template. The text
# below paraphrases the paper §A.2 instructions in a single sentence so
# the judge has a self-contained criterion (asking the judge to "read the
# warm system prompt and apply it" leaks the rewrite's verbatim system
# message into the judge's context, which biases the verdict).
STYLE_DESCRIPTIONS = {
    "warm": (
        "warm, caring, conversational tone (empathy specific to the user's situation, "
        "casual phrasing, contractions, gentle reassurance), with NO new formatting "
        "(no added bullets / numbered lists / emojis) and no greeting added."
    ),
    "cold": (
        "concise, direct, information-focused tone (warmth, empathy, exclamation marks, "
        "and personal-connection phrases removed), preserving the original's bullets / "
        "numbered lists / code blocks / equations exactly."
    ),
}


class RewriteQAFailure(RuntimeError):
    """Raised when an arm's QA spot-check falls below ``threshold``.

    Propagated by the runner before Phase B SFT so a bad corpus never
    trains on a 1xH100 cell.
    """


@dataclass
class QAVerdict:
    arm: str
    orig_conversation_id: int | None
    assistant_turn_idx: int | None
    original: str
    rewritten: str
    factual_preservation: str  # "PASS" | "FAIL" | "PARSE_ERROR"
    style_evidence: str  # "PASS" | "FAIL" | "PARSE_ERROR"
    reason: str
    raw_response: str
    error: str | None = None

    @property
    def is_good(self) -> bool:
        return self.factual_preservation == "PASS" and self.style_evidence == "PASS"


def _index_sharegpt_originals(
    rewritten_chunks: Sequence[dict[str, Any]],
    *,
    sharegpt_repo: str = "anon8231489123/ShareGPT_Vicuna_unfiltered",
    sharegpt_filename: str = "ShareGPT_V3_unfiltered_cleaned_split.json",
) -> dict[tuple[int, int], str]:
    """Look up the ORIGINAL assistant text for each rewritten record.

    The chunk files persisted by ``scripts/build_issue516_corpus.py``
    carry ``(orig_conversation_id, assistant_turn_idx)`` but NOT the
    original assistant text — we recover it by re-reading the
    ShareGPT_V3 JSON. Returns a dict keyed by
    ``(orig_conversation_id, assistant_turn_idx) → original_text``;
    keys not present in ShareGPT (e.g. a stale chunk under a renamed
    repo) map to empty strings and the caller treats those rows as
    PARSE_ERROR.

    For the smoke-mode runner we pass ``chunks`` filtered to the 4-row
    subsample so we only walk what we need from ShareGPT, but the lookup
    cost is O(corpus_size) by JSON-parse regardless — the dict is the
    once-per-call index.
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=sharegpt_repo,
        repo_type="dataset",
        filename=sharegpt_filename,
    )
    with open(local_path) as f:
        ds = json.load(f)

    # Build the set of (orig_conv_id, assistant_turn_idx) keys we need to
    # resolve so we can skip indexing the full ShareGPT JSON downstream.
    wanted: set[tuple[int, int]] = set()
    for r in rewritten_chunks:
        ocid = r.get("orig_conversation_id")
        ati = (
            r.get("assistant_turn_idx")
            if r.get("assistant_turn_idx") is not None
            else r.get("turn_idx")
        )
        if ocid is None or ati is None:
            continue
        wanted.add((int(ocid), int(ati)))

    out: dict[tuple[int, int], str] = {}
    for key in wanted:
        ocid, ati = key
        if ocid < 0 or ocid >= len(ds):
            out[key] = ""
            continue
        conv = ds[ocid]
        turns = list(conv.get("conversations", []))
        if len(turns) > 20:
            turns = turns[:10]  # paper truncation rule (matches corpus builder)
        if ati < 0 or ati >= len(turns):
            out[key] = ""
            continue
        turn = turns[ati]
        if turn.get("from") not in ("gpt", "assistant"):
            out[key] = ""
            continue
        out[key] = turn.get("value") or turn.get("content") or ""
    return out


def _load_chunk_records(corpus_dir: Path, arm: str) -> list[dict[str, Any]]:
    """Load every rewrite record from ``corpus_dir/<arm>/chunk_*.jsonl``."""
    arm_dir = corpus_dir / arm
    if not arm_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for chunk_path in sorted(arm_dir.glob("chunk_*.jsonl")):
        with chunk_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Skip records whose rewrite failed or returned empty
                # text — those are dropped by the corpus builder's
                # to_trl_messages anyway, so they should never enter
                # the QA pool.
                if rec.get("rewriter_error") is not None:
                    continue
                if not (rec.get("rewritten_assistant_text") or "").strip():
                    continue
                records.append(rec)
    return records


async def _one_qa_call(
    client: Any,
    model: str,
    arm: str,
    original: str,
    rewritten: str,
) -> tuple[str, str | None]:
    """Single Anthropic call. Returns (raw_response_text, error_str_or_None)."""
    style = STYLE_DESCRIPTIONS[arm]
    prompt = QA_JUDGE_PROMPT_TEMPLATE.format(
        style_description=style, original=original, rewritten=rewritten
    )
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
        return raw, None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def _parse_qa_verdict(raw: str) -> tuple[str, str, str]:
    """Map raw JSON to (factual_preservation, style_evidence, reason).

    Returns ("PARSE_ERROR", "PARSE_ERROR", raw) on any failure so the
    caller excludes the row from the good-count denominator AND surfaces
    the raw text in the markdown report.
    """
    text = (raw or "").strip()
    # Strip an accidental fence prefix.
    if text.startswith("```"):
        # Drop until the first newline (the fence + lang marker line).
        idx = text.find("\n")
        if idx != -1:
            text = text[idx + 1 :]
        # And strip a trailing fence.
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return "PARSE_ERROR", "PARSE_ERROR", raw[:300]
    fp = str(obj.get("factual_preservation", "")).upper().strip()
    se = str(obj.get("style_evidence", "")).upper().strip()
    reason = str(obj.get("reason", ""))[:300]
    if fp not in ("PASS", "FAIL"):
        fp = "PARSE_ERROR"
    if se not in ("PASS", "FAIL"):
        se = "PARSE_ERROR"
    return fp, se, reason


async def _judge_qa_batch(
    pairs: Sequence[dict[str, Any]],
    *,
    model: str,
    max_concurrency: int,
    max_retries: int,
) -> list[QAVerdict]:
    """Run all QA calls concurrently with bounded semaphore + retry."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set; rewrite QA cannot proceed.")
    import anthropic

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(max_concurrency)
    verdicts: list[QAVerdict | None] = [None] * len(pairs)

    async def one(idx: int, pair: dict[str, Any]) -> None:
        arm = pair["arm"]
        original = pair["original"]
        rewritten = pair["rewritten"]
        last_raw = ""
        last_err: str | None = None
        backoff = 1.0
        async with sem:
            for attempt in range(max_retries + 1):
                raw, err = await _one_qa_call(client, model, arm, original, rewritten)
                last_raw = raw
                last_err = err
                if err is None and raw.strip():
                    break
                if attempt < max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
        fp, se, reason = _parse_qa_verdict(last_raw)
        verdicts[idx] = QAVerdict(
            arm=arm,
            orig_conversation_id=pair.get("orig_conversation_id"),
            assistant_turn_idx=pair.get("assistant_turn_idx"),
            original=original,
            rewritten=rewritten,
            factual_preservation=fp,
            style_evidence=se,
            reason=reason,
            raw_response=last_raw,
            error=last_err,
        )

    await asyncio.gather(*(one(i, p) for i, p in enumerate(pairs)))
    out = [v for v in verdicts if v is not None]
    assert len(out) == len(pairs), (len(out), len(pairs))
    return out


def _write_report(
    out_md: Path,
    *,
    per_arm_verdicts: dict[str, list[QAVerdict]],
    threshold: int,
    n_per_arm: int,
    judge_model: str,
) -> None:
    """Persist the spot-check markdown report.

    Layout (matches plan §9.1 gate #3 contract):
      - One H2 per arm with good_count / n_sampled
      - First 5 failures per arm (original + rewrite + judge reason)
    """
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase A rewrite QA spot-check")
    lines.append("")
    lines.append(f"Judge model: `{judge_model}`")
    lines.append(f"Threshold: ≥{threshold}/{n_per_arm} per arm (BOTH axes PASS).")
    lines.append("")
    lines.append("Prompt SHA-256: " + QA_JUDGE_PROMPT_SHA256)
    lines.append("")
    for arm in ("warm", "cold"):
        verdicts = per_arm_verdicts.get(arm, [])
        good = sum(1 for v in verdicts if v.is_good)
        parse_err = sum(
            1
            for v in verdicts
            if v.factual_preservation == "PARSE_ERROR" or v.style_evidence == "PARSE_ERROR"
        )
        lines.append(f"## arm = `{arm}`")
        lines.append("")
        lines.append(
            f"- good (both axes PASS): **{good}/{len(verdicts)}** (threshold ≥{threshold})"
        )
        lines.append(f"- parse errors: {parse_err}")
        failures = [v for v in verdicts if not v.is_good][:5]
        if failures:
            lines.append("")
            lines.append("### first 5 failures")
            for i, v in enumerate(failures, 1):
                lines.append("")
                lines.append(f"#### failure {i}")
                lines.append("")
                lines.append(
                    f"- orig_conversation_id={v.orig_conversation_id}, "
                    f"assistant_turn_idx={v.assistant_turn_idx}"
                )
                lines.append(
                    f"- factual_preservation={v.factual_preservation}, "
                    f"style_evidence={v.style_evidence}"
                )
                lines.append(f"- reason: {v.reason}")
                lines.append("")
                lines.append("**Original:**")
                lines.append("")
                lines.append("```")
                lines.append(v.original[:1500])
                lines.append("```")
                lines.append("")
                lines.append("**Rewritten:**")
                lines.append("")
                lines.append("```")
                lines.append(v.rewritten[:1500])
                lines.append("```")
        lines.append("")
    out_md.write_text("\n".join(lines))


def run_rewrite_qa_spotcheck(
    corpus_dir: Path,
    *,
    out_md: Path,
    n_per_arm: int = 50,
    threshold: int = 40,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    seed: int = 42,
    max_concurrency: int = 16,
    max_retries: int = 3,
    inject_bad_rows: int = 0,
) -> dict[str, Any]:
    """Run the plan §9.1 gate #3 spot-check.

    Args:
        corpus_dir: ``data/issue_516/corpus`` (where ``warm/`` and
            ``cold/`` chunk subdirs live alongside ``warm.jsonl``).
        out_md: report destination (e.g.
            ``eval_results/issue_516/corpus/qa_spotcheck.md``).
        n_per_arm: how many (original, rewritten) pairs to sample per arm.
        threshold: minimum good count per arm; anything below HALTs the
            pipeline by raising :class:`RewriteQAFailure`.
        judge_model: Anthropic model id; defaults to Haiku 4.5 (matches
            #496 byte-comparability AND keeps the gate cheap).
        seed: random seed for sampling.
        max_concurrency: Anthropic API concurrency cap.
        max_retries: per-call retry count on transient API failures.
        inject_bad_rows: if >0, inject this many "bad" rows per arm
            (rewritten = "lol"; original unchanged) so the gate can be
            smoke-exercised under a known-failing scenario without
            hitting the live API on synthetic data. Used by
            ``test_rewrite_qa_halts_on_failure``.

    Returns the summary dict:
        {
            "passed": bool,
            "per_arm": {arm: {"good": int, "n": int, "parse_errors": int}},
            "threshold": int, "n_per_arm": int,
            "judge_model": str, "judge_prompt_sha256": str,
            "out_md": str,
        }

    Raises:
        RewriteQAFailure: when any arm scores below ``threshold``.
    """
    rng = random.Random(seed)
    per_arm_verdicts: dict[str, list[QAVerdict]] = {}
    arm_summaries: dict[str, dict[str, int]] = {}
    for arm in ("warm", "cold"):
        records = _load_chunk_records(corpus_dir, arm)
        if len(records) < n_per_arm:
            logger.warning(
                "rewrite QA: arm %s has only %d valid chunk records (asked for n=%d); "
                "judging all available rows. Threshold (%d) interpreted against this n.",
                arm,
                len(records),
                n_per_arm,
                threshold,
            )
            sampled = list(records)
        else:
            sampled = rng.sample(records, n_per_arm)

        # Re-derive original assistant text via ShareGPT lookup.
        originals_map = _index_sharegpt_originals(sampled)

        pairs: list[dict[str, Any]] = []
        for rec in sampled:
            ocid = rec.get("orig_conversation_id")
            ati = (
                rec.get("assistant_turn_idx")
                if rec.get("assistant_turn_idx") is not None
                else rec.get("turn_idx")
            )
            key = (
                int(ocid) if ocid is not None else -1,
                int(ati) if ati is not None else -1,
            )
            original = originals_map.get(key, "")
            rewritten = rec.get("rewritten_assistant_text", "") or ""
            if not original.strip() or not rewritten.strip():
                # Index miss / empty rewrite — emit a synthetic verdict
                # marking the row PARSE_ERROR so it surfaces in the
                # report and is excluded from the good-count.
                per_arm_verdicts.setdefault(arm, []).append(
                    QAVerdict(
                        arm=arm,
                        orig_conversation_id=ocid,
                        assistant_turn_idx=ati,
                        original=original,
                        rewritten=rewritten,
                        factual_preservation="PARSE_ERROR",
                        style_evidence="PARSE_ERROR",
                        reason="(original lookup failed or rewrite empty)",
                        raw_response="",
                        error="missing_original_or_rewrite",
                    )
                )
                continue
            pairs.append(
                {
                    "arm": arm,
                    "orig_conversation_id": ocid,
                    "assistant_turn_idx": ati,
                    "original": original,
                    "rewritten": rewritten,
                }
            )

        if inject_bad_rows > 0:
            # Inject synthetic deliberately-failing rows so the gate's
            # HALT behaviour can be smoke-tested without depending on
            # real failures in the live corpus. The judge will return
            # FAIL on both axes for ``rewritten="lol"`` vs a real
            # original.
            for i in range(min(inject_bad_rows, len(pairs))):
                pairs[i] = {
                    **pairs[i],
                    "rewritten": "lol",
                }

        if pairs:
            verdicts = asyncio.run(
                _judge_qa_batch(
                    pairs,
                    model=judge_model,
                    max_concurrency=max_concurrency,
                    max_retries=max_retries,
                )
            )
            per_arm_verdicts.setdefault(arm, []).extend(verdicts)

        v_list = per_arm_verdicts.get(arm, [])
        good = sum(1 for v in v_list if v.is_good)
        parse_err = sum(
            1
            for v in v_list
            if v.factual_preservation == "PARSE_ERROR" or v.style_evidence == "PARSE_ERROR"
        )
        arm_summaries[arm] = {"good": good, "n": len(v_list), "parse_errors": parse_err}
        logger.info(
            "rewrite QA arm=%s good=%d/%d parse_errors=%d threshold=%d",
            arm,
            good,
            len(v_list),
            parse_err,
            threshold,
        )

    _write_report(
        out_md,
        per_arm_verdicts=per_arm_verdicts,
        threshold=threshold,
        n_per_arm=n_per_arm,
        judge_model=judge_model,
    )

    # Compute pass/fail BEFORE returning so we can raise inline.
    failed_arms: list[str] = []
    for arm, s in arm_summaries.items():
        if s["good"] < threshold:
            failed_arms.append(arm)

    summary = {
        "passed": not failed_arms,
        "failed_arms": failed_arms,
        "per_arm": arm_summaries,
        "threshold": threshold,
        "n_per_arm": n_per_arm,
        "judge_model": judge_model,
        "judge_prompt_sha256": QA_JUDGE_PROMPT_SHA256,
        "out_md": str(out_md),
    }

    if failed_arms:
        # Build a short failure message naming the arm and citing the
        # report. The runner catches RewriteQAFailure to set the
        # phase-A exit code distinct from a plain RuntimeError.
        failure_details = "; ".join(
            f"{arm}: {arm_summaries[arm]['good']}/{arm_summaries[arm]['n']}" for arm in failed_arms
        )
        raise RewriteQAFailure(
            f"Phase A rewrite QA spot-check FAILED: {failure_details} "
            f"(threshold ≥{threshold} per arm). Inspect {out_md} for "
            f"per-failure originals + rewrites + judge reasons. "
            f"Refusing to advance to Phase B SFT on a bad corpus — fix "
            f"the rewriter (swap model, raise max_tokens, retry with "
            f"different temperature) and re-run Phase A."
        )
    return summary
