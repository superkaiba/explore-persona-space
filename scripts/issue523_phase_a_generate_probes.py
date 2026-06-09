#!/usr/bin/env python3
"""Issue #523 — Phase A held-out probe pool generation.

Generates a 500-probe held-out pool, disjoint from #502's `probes_500.json` AND
from #474's q_train (30) + q_test (50), with the indirect-register voice-drift
bug fixed and validated.

Inheritance: this script is `scripts/issue502_generate_probes.py` adapted; the
bucket prompts, Claude API client, batch flow, and exact-string dedup logic
(`_normalize`) are copied verbatim. The deltas (per plan v2 §4 Phase A):

  - Targets 500 NEW probes (vs #502's 450 new + 50 q_test prefix) — the held-out
    pool stands alone, not merged with q_test.
  - Disjointness against THREE sources at generation time, with retry-on-collision:
      (i) `eval_results/issue_502/probes_500.json` (#502's pool, all 500 entries)
      (ii) `q_train_answers.json` (30 #474 train questions)
      (iii) `q_test_extended_50.json` (50 #474 held-out R_test questions)
    Exact-string `_normalize` dedup PLUS a semantic-similarity ≥ 0.9 cosine
    reject using SBERT `sentence-transformers/all-MiniLM-L6-v2`. If > 20% of
    generated candidates hit the dedup gate, abort + raise (the bucket prompts
    have exhausted the easy mixed-distribution pool).
  - Voice-drift fix on the indirect register:
      (a) explicit third-person constraint + four worked exemplars in the prompt;
      (b) post-generation regex validator (`\\b(I|me|my|mine|myself|we|us|our|
          ours|ourselves)\\b`, case-insensitive); a hit → FAIL.
      (c) Claude-validator second pass (Sonnet, single call per regex-pass
          rewrite) confirming third-person register;
      (d) up to 3 retries per failing rewrite; if still failing → mark with
          `voice_drift_failed=true` (do NOT silently drop).
      (e) end-of-Phase-A audit: random sample 50 indirect-register rewrites
          + full programmatic regex sweep. Requires `third_person_rate >= 0.96`
          on the audit; aborts if not.
  - Outputs (per plan v2 §4 Phase A):
      * eval_results/issue_523/heldout_probes_500.json — 500 probes
      * eval_results/issue_523/class_d_rewrites_extended_v1.json — 500x5 rewrites
      * eval_results/issue_523/phase_a_audit.json — dedup + voice-drift report

Usage::

    # Smoke (12 probes, end-to-end including SBERT + validator + dedup).
    uv run python scripts/issue523_phase_a_generate_probes.py --smoke-only --n 12

    # Full run (500 probes, real Claude API, ~$150-200 budget).
    uv run python scripts/issue523_phase_a_generate_probes.py --n 500
"""

# Greek + special characters appear in docstrings / comments.

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
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

# Re-use the byte-identical building blocks from #502's generator.
from issue502_generate_probes import (  # noqa: E402
    BUCKETS as I502_BUCKETS,
)
from issue502_generate_probes import (  # noqa: E402
    CLASS_D_REGISTERS,
    CLAUDE_MODEL,
    GEN_SYSTEM,
    _build_rewrites_requests,
    _call_claude_once,
    _collect_rewrites_results,
    _gen_prompt,
    _normalize,
    _parse_questions,
    _smoke_synthetic_rewrites,
    _submit_rewrites_batch,
    _valid_question,
    _wait_for_rewrites_batch,
)

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_q_test_extended_50,
    load_q_train_answers,
)

logger = logging.getLogger("i523.gen_probes")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_523"
PROBES_PATH = OUT_DIR / "heldout_probes_500.json"
SMOKE_PROBES_PATH = OUT_DIR / "heldout_probes_500.smoke.json"
REWRITES_EXTENSION_PATH = OUT_DIR / "class_d_rewrites_extended_v1.json"
SMOKE_REWRITES_EXTENSION_PATH = OUT_DIR / "class_d_rewrites_extended_v1.smoke.json"
AUDIT_PATH = OUT_DIR / "phase_a_audit.json"
SMOKE_AUDIT_PATH = OUT_DIR / "phase_a_audit.smoke.json"

# Total target — the held-out pool is 500 probes (vs #502's "450 new + 50 q_test").
DEFAULT_TARGET = 500

# Same bucket prompts as #502, but reweighted to sum to 500 (not 450).
# Ratio stays identical: 35% capabilities / 25% opinion / 25% neutral / 15% hypothetical.
BUCKETS: tuple[tuple[str, int, str], ...] = tuple(
    (name, round(500 * (n / 450)), desc) for (name, n, desc) in I502_BUCKETS
)
# Adjust last bucket so the sum is exactly 500.
_correction = 500 - sum(n for _, n, _ in BUCKETS)
BUCKETS = tuple(
    (name, n + (_correction if i == len(BUCKETS) - 1 else 0), desc)
    for i, (name, n, desc) in enumerate(BUCKETS)
)
assert sum(n for _, n, _ in BUCKETS) == 500, f"BUCKETS sum {sum(n for _, n, _ in BUCKETS)} != 500"

# Dedup gate: a candidate whose cosine similarity to any existing probe is
# >= SBERT_THRESHOLD is rejected as a near-duplicate paraphrase.
SBERT_THRESHOLD = 0.9
SBERT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# First-person token regex (case-insensitive, word-boundary) for the
# indirect-register voice-drift validator.
FIRST_PERSON_PATTERN = re.compile(
    r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b",
    flags=re.IGNORECASE,
)

# Per-bucket dedup-rate budget: if > 20% of generated candidates in ONE bucket
# hit dedup, the bucket prompt has exhausted the easy mixed-distribution pool.
MAX_DEDUP_RATE_PER_BUCKET = 0.20

# Voice-drift validator parameters
MAX_VOICE_DRIFT_RETRIES = 3
VOICE_DRIFT_AUDIT_SAMPLE = 50
VOICE_DRIFT_THIRD_PERSON_RATE_MIN = 0.96


# ────────────────────────── Provenance helpers ──────────────────────────


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


# ────────────────────────── Voice-drift validators ──────────────────────────


def _has_first_person(text: str) -> bool:
    """Return True iff `text` contains any first-person pronoun (word-boundary, ci).

    Validator (a) — cheap deterministic first pass per plan §4 Phase A.
    """
    return FIRST_PERSON_PATTERN.search(text) is not None


VOICE_DRIFT_SYSTEM = (
    "You are validating English text for grammatical voice. You will be given "
    "ONE short text. Reply ONLY with 'THIRD_PERSON' if the text uses third-"
    "person grammar throughout (no 'I', 'me', 'my', 'we', 'us', 'our', no "
    "first-person verbs), or 'NOT_THIRD_PERSON' otherwise. Reply with no other "
    "words, no punctuation, no preamble."
)


def _claude_validate_third_person(text: str) -> bool:
    """Claude validator (Sonnet) — second pass per plan §4 Phase A (c).

    Returns True iff Claude judges the text to be in third-person register.
    A non-deterministic API hiccup defaults to False (fail-closed); the
    retry loop will see this as a fail and either re-request or mark
    `voice_drift_failed`.
    """
    try:
        resp = _call_claude_once(
            user_prompt=f"Text to validate:\n{text}",
            max_tokens=64,
            system=VOICE_DRIFT_SYSTEM,
        )
    except Exception as e:
        logger.warning("claude validator API error: %s — failing closed", e)
        return False
    judgment = resp.strip().upper()
    return judgment.startswith("THIRD_PERSON")


INDIRECT_REGISTER_FIX_SYSTEM = (
    "You are rewriting an English question into an INDIRECT register that asks "
    "for the same information without addressing 'you' (the assistant) "
    "directly. Hard constraint: answer in the THIRD person; NEVER use 'I', "
    "'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'. "
    "Examples of acceptable indirect rewrites:\n"
    "  Q: How does a transistor work?\n"
    "  A: A reader is curious about how a transistor works. Could one explain?\n"
    "  Q: What's the best way to learn Spanish?\n"
    "  A: The user community wonders what is the best route to learn Spanish.\n"
    "  Q: Should beginners pick up piano or guitar first?\n"
    "  A: There is a longstanding debate about whether beginners should pick "
    "up piano or guitar first; what is the prevailing view?\n"
    "  Q: What if humans never invented writing?\n"
    "  A: Historians sometimes imagine a counterfactual in which humans never "
    "invented writing — what would that world look like?\n\n"
    "Output exactly ONE rewrite, one line, no preamble, no commentary, no "
    "quotation marks. Use third person throughout. Preserve the question's "
    "meaning."
)


def _request_indirect_rewrite(question: str, retry_reason: str | None = None) -> str:
    """One Sonnet call asking for a third-person indirect rewrite."""
    suffix = ""
    if retry_reason:
        suffix = (
            f"\n\nIMPORTANT: a previous attempt failed validation: {retry_reason}. "
            "Try again, strictly in the third person."
        )
    user_prompt = f"Original question:\n{question}\n\nProduce the indirect rewrite.{suffix}"
    return (
        _call_claude_once(
            user_prompt=user_prompt,
            max_tokens=512,
            system=INDIRECT_REGISTER_FIX_SYSTEM,
        )
        .strip()
        .splitlines()[0]
        .strip()
    )


def _validate_and_fix_indirect_rewrite(
    question: str,
    initial_rewrite: str,
    skip_claude: bool = False,
) -> tuple[str, bool, list[str]]:
    """Validate an indirect rewrite, retrying up to MAX_VOICE_DRIFT_RETRIES times.

    Returns (final_rewrite, voice_drift_failed, attempts_log).

    voice_drift_failed=True signals "all retries exhausted, mark in audit but
    keep the row" — per plan §4 Phase A, we do NOT silently drop.

    skip_claude=True (smoke mode) uses only the regex validator (no API calls).
    """
    attempts: list[str] = []
    current = initial_rewrite
    for attempt in range(MAX_VOICE_DRIFT_RETRIES + 1):
        regex_fail = _has_first_person(current)
        if regex_fail:
            attempts.append(f"attempt{attempt}: regex FAIL ({current!r})")
            if attempt >= MAX_VOICE_DRIFT_RETRIES:
                return current, True, attempts
            current = _request_indirect_rewrite(
                question, retry_reason="first-person pronoun detected by regex"
            )
            continue
        # regex passed; try Claude validator unless in smoke mode
        if skip_claude:
            attempts.append(f"attempt{attempt}: regex PASS (claude skipped)")
            return current, False, attempts
        claude_pass = _claude_validate_third_person(current)
        if claude_pass:
            attempts.append(f"attempt{attempt}: regex PASS + claude PASS")
            return current, False, attempts
        attempts.append(f"attempt{attempt}: regex PASS + claude FAIL ({current!r})")
        if attempt >= MAX_VOICE_DRIFT_RETRIES:
            return current, True, attempts
        current = _request_indirect_rewrite(
            question, retry_reason="Claude judged the rewrite as not third-person"
        )
    # Unreachable (loop returns at every branch), but mypy/ruff appreciate it.
    return current, True, attempts


# ────────────────────────── SBERT semantic dedup ──────────────────────────


_SBERT_MODEL = None  # lazy module-level singleton


def _get_sbert():
    """Lazy-load the SBERT model. Returns None if sentence-transformers unavailable.

    Caller MUST handle None (we treat that as "skip semantic dedup" — the exact-
    string dedup still runs).
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer

            _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_ID)
        except ImportError as e:
            logger.warning(
                "sentence-transformers unavailable (%s); semantic dedup disabled. "
                "exact-string dedup still active.",
                e,
            )
            _SBERT_MODEL = False
    return _SBERT_MODEL if _SBERT_MODEL is not False else None


def _semantic_duplicate_check(
    candidates: list[str],
    existing_texts: list[str],
    threshold: float = SBERT_THRESHOLD,
) -> list[bool]:
    """For each candidate, return True iff it has SBERT cosine ≥ threshold to ANY existing.

    Length of return matches `candidates`. If SBERT is unavailable, all-False.
    """
    sbert = _get_sbert()
    if sbert is None:
        return [False] * len(candidates)
    if not candidates or not existing_texts:
        return [False] * len(candidates)

    # SentenceTransformer returns float tensors; convert to np for cosine math.
    cand_emb = sbert.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)
    exist_emb = sbert.encode(existing_texts, convert_to_numpy=True, normalize_embeddings=True)
    sims = cand_emb @ exist_emb.T  # (n_cand, n_exist), each row cosine to all existing
    max_per_cand = sims.max(axis=1)
    return [bool(s >= threshold) for s in max_per_cand]


# ────────────────────────── Bucket generation w/ dedup gate ──────────────────────────


def _load_existing_probe_corpus() -> tuple[list[str], dict[str, int]]:
    """Load all THREE exclude sources: #502 pool + #474 q_train + q_test.

    Returns the concatenated raw text list AND a per-source count dict for
    the audit JSON.
    """
    sources: dict[str, list[str]] = {}

    # #502 — 500 probes total (50 q_test prefix + 450 new). All 500 are excluded.
    p502 = PROJECT_ROOT / "eval_results" / "issue_502" / "probes_500.json"
    if not p502.exists():
        raise FileNotFoundError(
            f"#502 probes pool {p502} missing — required for disjointness "
            "(generate via scripts/issue502_generate_probes.py first)."
        )
    payload502 = json.loads(p502.read_text())
    sources["issue_502_probes_500"] = list(payload502.get("probes", []))

    # #474 q_train (30) + q_test (50).
    sources["issue_474_q_train"] = list(load_q_train_answers().keys())
    sources["issue_474_q_test"] = load_q_test_extended_50()

    flat = []
    counts: dict[str, int] = {}
    for name, items in sources.items():
        flat.extend(items)
        counts[name] = len(items)
    return flat, counts


def _generate_bucket_with_disjointness(
    bucket_name: str,
    bucket_desc: str,
    target_n: int,
    existing_norms: set[str],
    existing_texts: list[str],
    chunk_size: int = 60,
    max_attempts: int = 8,
    smoke: bool = False,
) -> tuple[list[str], dict]:
    """Generate `target_n` valid + exact-disjoint + semantic-disjoint probes.

    Maintains a running record of `dedup_count` (exact-match rejects) +
    `semantic_dup_count` (SBERT rejects) for the audit. Aborts via
    RuntimeError if the per-bucket dedup rate exceeds
    MAX_DEDUP_RATE_PER_BUCKET on a chunk and we cannot otherwise meet target_n.
    """
    accepted: list[str] = []
    attempts_used = 0
    total_generated = 0
    exact_dedup_rejects = 0
    semantic_dedup_rejects = 0
    invalid_rejects = 0
    while len(accepted) < target_n:
        if attempts_used >= max_attempts:
            raise RuntimeError(
                f"Bucket {bucket_name!r}: exhausted {max_attempts} chunks, "
                f"have {len(accepted)} / {target_n}. "
                f"exact_dedup={exact_dedup_rejects} semantic={semantic_dedup_rejects} "
                f"invalid={invalid_rejects} total_generated={total_generated}. "
                "Bucket prompt likely exhausted; revisit the bucket description."
            )
        attempts_used += 1
        need = target_n - len(accepted)
        ask = min(chunk_size, need + 20)
        prompt = _gen_prompt(bucket_name, bucket_desc, ask, sorted(existing_norms)[:30])
        if smoke:
            # Smoke mode: synthesize candidates that pass dedup trivially.
            candidates = [
                f"[smoke-{bucket_name}-{attempts_used:02d}-{i:03d}] "
                f"What is fact {bucket_name} number {i}?"
                for i in range(ask)
            ]
        else:
            text = _call_claude_once(prompt)
            candidates = _parse_questions(text)
        total_generated += len(candidates)

        # Step 1 — exact-string dedup + validity filter.
        survivors_after_exact: list[str] = []
        for c in candidates:
            ok, _why = _valid_question(c)
            if not ok:
                invalid_rejects += 1
                continue
            key = _normalize(c)
            if key in existing_norms:
                exact_dedup_rejects += 1
                continue
            survivors_after_exact.append(c)

        # Step 2 — semantic dedup (SBERT cosine ≥ threshold to ANY existing).
        if survivors_after_exact:
            dup_flags = _semantic_duplicate_check(
                survivors_after_exact, existing_texts, threshold=SBERT_THRESHOLD
            )
            survivors_after_sem: list[str] = []
            for cand, is_dup in zip(survivors_after_exact, dup_flags, strict=True):
                if is_dup:
                    semantic_dedup_rejects += 1
                else:
                    survivors_after_sem.append(cand)
        else:
            survivors_after_sem = []

        # Accept up to target_n.
        for c in survivors_after_sem:
            if len(accepted) >= target_n:
                break
            existing_norms.add(_normalize(c))
            existing_texts.append(c)
            accepted.append(c)
        logger.info(
            "bucket=%s attempt=%d ask=%d accepted_total=%d/%d (exact_rej=%d sem_rej=%d invalid=%d)",
            bucket_name,
            attempts_used,
            ask,
            len(accepted),
            target_n,
            exact_dedup_rejects,
            semantic_dedup_rejects,
            invalid_rejects,
        )

    audit = {
        "bucket": bucket_name,
        "target_n": target_n,
        "delivered_n": len(accepted),
        "total_generated": total_generated,
        "exact_dedup_rejects": exact_dedup_rejects,
        "semantic_dedup_rejects": semantic_dedup_rejects,
        "invalid_rejects": invalid_rejects,
        "attempts_used": attempts_used,
        "exact_dedup_rate": (exact_dedup_rejects / total_generated if total_generated else 0.0),
        "semantic_dedup_rate": (
            semantic_dedup_rejects / total_generated if total_generated else 0.0
        ),
    }
    # Plan §4 Phase A: per-bucket dedup rate budget.
    total_dedup_rate = audit["exact_dedup_rate"] + audit["semantic_dedup_rate"]
    if total_dedup_rate > MAX_DEDUP_RATE_PER_BUCKET:
        raise RuntimeError(
            f"Bucket {bucket_name!r} dedup-rate {total_dedup_rate:.3f} > "
            f"{MAX_DEDUP_RATE_PER_BUCKET}; pool exhausted. {audit}"
        )
    return accepted, audit


# ────────────────────────── Class-D rewrites with voice-drift fix ──────────────────────────


def _generate_rewrites_with_voice_drift_fix(
    new_probes: list[str],
    *,
    smoke: bool,
) -> tuple[dict[str, dict[str, str]], dict]:
    """Run the batched rewrites flow, then re-validate the indirect register.

    Returns (rewrites, audit_block). audit_block contains:
        n_questions, n_indirect_regex_pass, n_indirect_claude_pass,
        n_indirect_voice_drift_failed, indirect_third_person_rate
    """
    if smoke:
        # Smoke: synthetic rewrites with the existing helper, but force the
        # indirect register through the validator path so the wiring is tested.
        rewrites = _smoke_synthetic_rewrites(new_probes)
    else:
        requests = _build_rewrites_requests(new_probes)
        batch_id = _submit_rewrites_batch(requests)
        _wait_for_rewrites_batch(batch_id)
        rewrites = _collect_rewrites_results(batch_id, new_probes)

    # Validate + (when needed) retry every indirect-register rewrite.
    # Round-2 fix to Critical-5: any rewrite that fails ALL retries is
    # DROPPED from the output (not retained with `voice_drift_failed=True`)
    # so the downstream audit reads `n_indirect_voice_drift_failed == 0`
    # rather than the round-1 silent-retain behavior. The final-set gate
    # below fails LOUD if any drops occurred.
    regex_pass_count = 0
    claude_pass_count = 0
    voice_drift_failed: list[dict] = []
    dropped_probes: list[str] = []
    skip_claude = smoke  # in smoke mode, no Claude validator calls
    for q in list(new_probes):
        original = rewrites[q]["indirect"]
        # If the smoke synthetic rewrite happens to contain first-person tokens,
        # the validator will retry but the synthesizer returns deterministic
        # "[smoke ...]" strings without 'I'/'me' so this should pass trivially.
        final, failed, attempts = _validate_and_fix_indirect_rewrite(
            q, original, skip_claude=skip_claude
        )
        if failed:
            voice_drift_failed.append({"question": q, "final_rewrite": final, "attempts": attempts})
            # Drop the row entirely — do NOT keep an exhausted-retry rewrite
            # in the output. The pool-builder caller sees the audit fail and
            # halts before the bad rewrites can confound the held-out CV.
            dropped_probes.append(q)
            del rewrites[q]
            continue
        rewrites[q]["indirect"] = final
        if not _has_first_person(final):
            regex_pass_count += 1
        else:
            # If a non-failed rewrite still trips the regex, that's a logic
            # bug in the retry loop — fail loud rather than silently undercount.
            raise RuntimeError(
                f"_validate_and_fix_indirect_rewrite returned failed=False but "
                f"first-person regex still trips on q={q!r} final={final!r}"
            )
        claude_pass_count += 1

    # Remove dropped probes from the upstream list so disjointness +
    # bucket-delivery audits stay consistent.
    if dropped_probes:
        kept = [q for q in new_probes if q not in set(dropped_probes)]
        new_probes.clear()
        new_probes.extend(kept)

    n_total = len(new_probes)

    # FULL regex sweep — every retained indirect rewrite must be first-person-free.
    # Plan §4 Phase A: 100% pass on the regex sweep is required.
    full_regex_pass = sum(1 for q in new_probes if not _has_first_person(rewrites[q]["indirect"]))
    full_regex_rate = full_regex_pass / max(n_total, 1)

    # FULL audit set (was: 50-sample sentinel) — every retained rewrite is
    # subject to the Claude validator on the real run. In smoke mode the
    # Claude calls are skipped; the regex sweep is still full.
    if not skip_claude:
        full_claude_pass = sum(
            1 for q in new_probes if _claude_validate_third_person(rewrites[q]["indirect"])
        )
    else:
        # Smoke: Claude not called; treat the regex pass as the Claude pass too
        # so the smoke audit can sanity-check the gating shape without API cost.
        full_claude_pass = full_regex_pass
    full_claude_rate = full_claude_pass / max(n_total, 1)

    return rewrites, {
        "n_questions": n_total,
        "n_indirect_regex_pass_total": regex_pass_count,
        "n_indirect_claude_pass_total": claude_pass_count,
        "n_indirect_voice_drift_failed": len(voice_drift_failed),
        "n_dropped_probes": len(dropped_probes),
        "indirect_third_person_regex_rate": (
            regex_pass_count / max(n_total + len(dropped_probes), 1)
        ),
        "indirect_third_person_claude_rate": (
            claude_pass_count / max(n_total + len(dropped_probes), 1)
        ),
        # FULL audit set (was: 50-sample sentinel). Plan §4 Phase A gate
        # reads full_regex_pass_rate == 1.0 AND full_claude_pass_rate >= 0.96
        # on the entire retained pool.
        "full_audit_size": n_total,
        "full_regex_pass": full_regex_pass,
        "full_regex_pass_rate": full_regex_rate,
        "full_claude_pass": full_claude_pass,
        "full_claude_pass_rate": full_claude_rate,
        # Kept for backwards-compat dashboards; the dispatcher gate reads
        # full_* keys above.
        "audit_sample_size": n_total,
        "audit_sample_regex_pass": full_regex_pass,
        "audit_sample_regex_pass_rate": full_regex_rate,
        "voice_drift_failed_examples": voice_drift_failed[:5],  # first 5 only
        "skip_claude": skip_claude,
    }


# ────────────────────────── I/O ──────────────────────────


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ────────────────────────── Main ──────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Issue #523 Phase A — held-out probe pool generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_TARGET,
        help="Total probes to generate (default 500).",
    )
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help=(
            "Skip the Claude API; synthesize placeholder probes + rewrites. "
            "Output goes to *.smoke.json paths; NEVER promoted to the real "
            "pool. Used for end-to-end smoke ahead of the real run."
        ),
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        help="Claude-call chunk size per bucket attempt.",
    )
    p.add_argument(
        "--skip-rewrites",
        action="store_true",
        help="Skip the Class-D rewrites step (debug only — extraction needs them).",
    )
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — sequential phase orchestrator
    args = _build_argparser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    smoke = bool(args.smoke_only)
    if smoke and args.n > 60:
        logger.warning(
            "smoke mode with --n=%d is large; smoke is normally --n 12. Continuing.",
            args.n,
        )

    # ── Load the disjoint corpus (#502 pool + #474 q_train + q_test) ──
    existing_texts, source_counts = _load_existing_probe_corpus()
    existing_norms: set[str] = {_normalize(t) for t in existing_texts}
    logger.info(
        "Disjointness corpus: %d total entries from %s",
        len(existing_texts),
        source_counts,
    )

    # ── Bucket targets — scale buckets proportionally when --n != 500 (smoke) ──
    if args.n != DEFAULT_TARGET:
        scaled = []
        scale = args.n / DEFAULT_TARGET
        running_sum = 0
        for i, (name, n, desc) in enumerate(BUCKETS):
            if i == len(BUCKETS) - 1:
                # Last bucket absorbs rounding error.
                target = args.n - running_sum
            else:
                target = max(1, round(n * scale))
                running_sum += target
            scaled.append((name, target, desc))
        buckets_active = tuple(scaled)
    else:
        buckets_active = BUCKETS

    started = time.time()
    new_probes: list[str] = []
    per_bucket_audits: list[dict] = []
    for bname, btarget, bdesc in buckets_active:
        if btarget <= 0:
            continue
        accepted, audit = _generate_bucket_with_disjointness(
            bname,
            bdesc,
            btarget,
            existing_norms,
            existing_texts,
            chunk_size=args.chunk_size,
            smoke=smoke,
        )
        new_probes.extend(accepted)
        per_bucket_audits.append(audit)
        logger.info(
            "bucket %s done: %d / %d (cumulative %d)",
            bname,
            len(accepted),
            btarget,
            len(new_probes),
        )

    # ── Hard invariants ──
    assert len(new_probes) == args.n, f"got {len(new_probes)} probes != target {args.n}"
    seen: set[str] = set()
    for p in new_probes:
        k = _normalize(p)
        if k in seen:
            raise AssertionError(f"internal duplicate (post-merge): {p!r}")
        seen.add(k)
    for p in new_probes:
        ok, why = _valid_question(p)
        if not ok:
            raise AssertionError(f"invalid final probe ({why}): {p!r}")

    out_path = SMOKE_PROBES_PATH if smoke else PROBES_PATH

    # ── Class-D rewrites (with voice-drift fix on indirect) ──
    rewrites: dict[str, dict[str, str]] = {}
    voice_drift_audit: dict = {}
    if not args.skip_rewrites:
        logger.info("Generating Class-D rewrites for %d new probes…", len(new_probes))
        rewrites, voice_drift_audit = _generate_rewrites_with_voice_drift_fix(
            new_probes, smoke=smoke
        )
        rewrites_out = SMOKE_REWRITES_EXTENSION_PATH if smoke else REWRITES_EXTENSION_PATH
        # Companion meta sidecar (matches #502's schema).
        meta_path = rewrites_out.with_suffix(".meta.json")
        meta = {
            "schema_version": 1,
            "smoke": smoke,
            "model": CLAUDE_MODEL,
            "model_id_canonical": "claude-sonnet-4-5-20250929",
            "registers": list(CLASS_D_REGISTERS),
            "n_questions": len(rewrites),
            "n_rewrites_total": sum(len(v) for v in rewrites.values()),
            "voice_drift_validator_pattern": FIRST_PERSON_PATTERN.pattern,
            "voice_drift_audit": voice_drift_audit,
            "provenance": {
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
                "python": platform.python_version(),
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        # The on-disk JSON is a top-level dict matching #406's schema (no wrapper):
        _write_atomic(rewrites_out, rewrites)
        logger.info(
            "Wrote %s (%d q x %d reg) + meta %s",
            rewrites_out,
            len(rewrites),
            len(CLASS_D_REGISTERS),
            meta_path,
        )

    # ── Final dedup tally against EACH original source (for the audit JSON) ──
    final_norms = {_normalize(p) for p in new_probes}
    overlap_502 = sum(
        1
        for t in json.loads(
            (PROJECT_ROOT / "eval_results" / "issue_502" / "probes_500.json").read_text()
        )["probes"]
        if _normalize(t) in final_norms
    )
    overlap_q_train = sum(1 for t in load_q_train_answers() if _normalize(t) in final_norms)
    overlap_q_test = sum(1 for t in load_q_test_extended_50() if _normalize(t) in final_norms)

    elapsed = time.time() - started

    # ── Audit JSON — the gate the dispatcher reads before Phase B ──
    audit_payload = {
        "schema_version": 1,
        "smoke": smoke,
        "n_new_probes": len(new_probes),
        "target_n": args.n,
        "source_counts": source_counts,
        "overlap_against_sources": {
            "issue_502_probes_500": overlap_502,
            "issue_474_q_train": overlap_q_train,
            "issue_474_q_test": overlap_q_test,
        },
        "per_bucket_audits": per_bucket_audits,
        "voice_drift_audit": voice_drift_audit,
        "sbert_model_id": SBERT_MODEL_ID,
        "sbert_threshold": SBERT_THRESHOLD,
        "voice_drift_third_person_rate_min": VOICE_DRIFT_THIRD_PERSON_RATE_MIN,
        "provenance": {
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
            "python": platform.python_version(),
            "elapsed_seconds": round(elapsed, 2),
        },
    }
    audit_out = SMOKE_AUDIT_PATH if smoke else AUDIT_PATH
    _write_atomic(audit_out, audit_payload)
    logger.info("Wrote audit %s", audit_out)

    # ── Pool file ──
    payload = {
        "schema_version": 1,
        "smoke": smoke,
        "model": CLAUDE_MODEL,
        "model_id_canonical": "claude-sonnet-4-5-20250929",
        "generation_system_prompt": GEN_SYSTEM,
        "buckets": [
            {
                "name": n,
                "target_n": t,
                "description": d,
                "delivered_n": next(
                    (a["delivered_n"] for a in per_bucket_audits if a["bucket"] == n),
                    0,
                ),
            }
            for n, t, d in buckets_active
        ],
        "n_new_probes": len(new_probes),
        "n_total": len(new_probes),
        # The held-out pool is its OWN 500; not merged with q_test (cf. #502).
        "probes": new_probes,
        "audit_path": str(audit_out.relative_to(PROJECT_ROOT)),
        "overlap_against_sources": audit_payload["overlap_against_sources"],
        "voice_drift_audit": voice_drift_audit,
        "provenance": {
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
            "python": platform.python_version(),
            "elapsed_seconds": round(elapsed, 2),
        },
    }
    _write_atomic(out_path, payload)
    logger.info(
        "Wrote %s (%d probes, smoke=%s) in %.1fs", out_path, len(new_probes), smoke, elapsed
    )

    # ── End-of-Phase-A gate (plan §4 Phase A): voice-drift rate audit ──
    # Round-2 fix to Critical-5 (Codex):
    #   (a) ZERO exhausted-retry failures retained — n_dropped_probes == 0.
    #   (b) FULL regex sweep pass rate == 1.0 over the entire retained pool
    #       (no first-person pronouns in any retained indirect rewrite).
    #   (c) FULL Claude validator pass rate >= 0.96 over the entire retained
    #       pool (was: 50-sample sentinel).
    if voice_drift_audit:
        n_dropped = voice_drift_audit.get("n_dropped_probes", 0)
        regex_rate_full = voice_drift_audit.get("full_regex_pass_rate", 0.0)
        claude_rate_full = voice_drift_audit.get("full_claude_pass_rate", 0.0)
        if not smoke and n_dropped > 0:
            raise RuntimeError(
                f"Phase A voice-drift audit: {n_dropped} probes dropped after "
                f"exhausting {MAX_VOICE_DRIFT_RETRIES} retries. Refusing to "
                "advance — voice-drift fix did not converge on every probe."
            )
        if not smoke and regex_rate_full < 1.0:
            raise RuntimeError(
                f"Phase A voice-drift audit: FULL regex sweep pass rate "
                f"{regex_rate_full:.3f} != 1.0 on a "
                f"{voice_drift_audit['full_audit_size']}-row pool. "
                "Plan §4 Phase A requires 100% regex pass on the full pool."
            )
        if not smoke and claude_rate_full < VOICE_DRIFT_THIRD_PERSON_RATE_MIN:
            raise RuntimeError(
                f"Phase A voice-drift audit: FULL Claude validator pass rate "
                f"{claude_rate_full:.3f} < {VOICE_DRIFT_THIRD_PERSON_RATE_MIN} "
                f"on a {voice_drift_audit['full_audit_size']}-row pool. "
                "Refusing to advance — voice-drift fix did not converge."
            )
        # Smoke gate: same FULL regex sweep contract, on the (deterministic)
        # synthesized placeholders. The synthesizer must produce no
        # first-person pronouns on any indirect rewrite — a regression here
        # is a smoke-wiring bug, not a Sonnet quality issue.
        if smoke and regex_rate_full < 1.0:
            raise RuntimeError(
                f"Phase A smoke voice-drift gate: FULL regex pass rate "
                f"{regex_rate_full:.3f} != 1.0 on "
                f"{voice_drift_audit['full_audit_size']} synthesized probes "
                "— the smoke synthesizer leaked first-person pronouns; "
                "smoke wiring regression."
            )
        if smoke and n_dropped > 0:
            raise RuntimeError(
                f"Phase A smoke voice-drift gate: {n_dropped} synthesized "
                "probes failed all retries. Deterministic smoke should never "
                "drop — smoke wiring regression."
            )

    # ── End-of-Phase-A gate: dedup rate ──
    for ba in per_bucket_audits:
        total_dedup = ba["exact_dedup_rate"] + ba["semantic_dedup_rate"]
        if total_dedup > MAX_DEDUP_RATE_PER_BUCKET:
            # Already enforced inside the bucket-generator; this is belt+suspenders.
            raise RuntimeError(
                f"Bucket {ba['bucket']!r} dedup rate {total_dedup:.3f} "
                f"> {MAX_DEDUP_RATE_PER_BUCKET}. Aborting."
            )

    logger.info("Phase A complete: %d probes, audit -> %s", len(new_probes), audit_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
