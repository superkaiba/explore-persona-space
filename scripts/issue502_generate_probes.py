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

Usage::

    # Real run (calls Claude API, costs a few dollars).
    uv run python scripts/issue502_generate_probes.py --target-new 450

    # Smoke (no API calls; uses synthetic placeholders so CPU smoke works
    # without a network. Output is clearly tagged ``smoke=True`` and is
    # NEVER promoted to the real 500-pool — the smoke file lives at
    # ``eval_results/issue_502/probes_500.smoke.json``).
    uv run python scripts/issue502_generate_probes.py --smoke

The real run uses Claude Sonnet 4.5 (``claude-sonnet-4-5-20250929``), the
project default for generation. We use one-shot ``client.messages.create``
calls (not the batch API) since 450 probes fits comfortably in a few
prompt-completions and we want to inspect them after each chunk.
"""

# Greek + special characters (×, →, —) appear in the docstring.
# ruff: noqa: RUF003

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

DEFAULT_TARGET_NEW = 450
DEFAULT_CHUNK = 60  # questions per Claude call

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


def _call_claude_once(user_prompt: str, max_tokens: int = 8000) -> str:
    """One-shot ``messages.create`` returning the text content."""
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key())
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=GEN_SYSTEM,
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    q_test = load_q_test_extended_50()
    q_train_keys = list(load_q_train_answers().keys())
    assert len(q_test) == 50, f"expected 50 q_test, got {len(q_test)}"
    assert len(q_train_keys) == 30, f"expected 30 q_train, got {len(q_train_keys)}"

    # Existing set: normalized q_train ∪ q_test (the disjoint constraint
    # is on the NEW probes only, so we seed existing with both).
    existing_keys: set[str] = set()
    for q in q_test:
        existing_keys.add(_normalize(q))
    for q in q_train_keys:
        existing_keys.add(_normalize(q))

    out_path = args.out or (SMOKE_PROBES_PATH if args.smoke else PROBES_PATH)
    started = time.time()

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
