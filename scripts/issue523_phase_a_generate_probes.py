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

# Re-use the byte-identical building blocks from #502's generator. The base
# `_gen_prompt` is NOT imported; round-4 widens the generation prompt with an
# anti-clustering instruction via the local `_gen_prompt_diverse` defined below.
# `_call_claude_once` is imported under an alias so round-5 can wrap it with an
# outer retry-on-transient-overload layer; the inherited helper itself is
# preserved byte-identically (its outputs feed #502's reproducible artifacts).
from issue502_generate_probes import (  # noqa: E402
    CLASS_D_REGISTERS,
    CLAUDE_MODEL,
    GEN_SYSTEM,
    _api_key,
    _build_rewrites_requests,
    _collect_rewrites_results,
    _normalize,
    _parse_questions,
    _smoke_synthetic_rewrites,
    _submit_rewrites_batch,
    _valid_question,
    _wait_for_rewrites_batch,
)
from issue502_generate_probes import _call_claude_once as _call_claude_once_base  # noqa: E402


def _call_claude_once(*args, **kwargs):
    """Wrap #502's `_call_claude_once` with retry on transient Anthropic 5xx.

    The inherited helper instantiates a fresh `anthropic.Anthropic()` client per
    call with the SDK default `max_retries=2`; round-3's pod launch died ~4 min
    into Phase A on `OverloadedError 529 — Overloaded` after the SDK's two
    internal retries were exhausted. The transient-overload window can outlast
    the SDK's default backoff, so wrap with an outer retry that catches the
    documented transient classes (overloaded, rate-limit, connection, timeout)
    and sleeps with exponential backoff + jitter. Total retry window ≈ 0+30+60+
    120+240+480+960 s ≈ 32 min, covering typical Anthropic overload windows.

    NOT retried: Authentication, BadRequest, PermissionDenied, NotFound — those
    are real bugs / config errors and should fail loud.
    """
    import random
    import time

    import anthropic

    # Round-5b fix (the round-5 first-attempt missed OverloadedError because
    # it lives in `anthropic._exceptions`, not the top-level `anthropic`
    # namespace — `getattr(anthropic, "OverloadedError", None)` returned None
    # on SDK v0.88, so the wrapper didn't catch the 529 and Phase A died).
    # Catch the PUBLIC superclass `anthropic.APIStatusError` (which IS at the
    # top level — `OverloadedError`, `RateLimitError`, `InternalServerError`,
    # `ServiceUnavailableError`, `BadRequestError`, etc. all inherit from it)
    # AND `APIConnectionError` / `APITimeoutError`, then DECIDE based on the
    # HTTP status code whether to retry.
    #
    # Retry: 408 (timeout), 425, 429 (rate-limit), 500, 502, 503, 504, 529
    #        (overloaded). These are the documented transient classes.
    # Raise: 400 (bad request), 401 (auth), 403 (permission), 404 (not found),
    #        409, 410, 413, 422, 451, etc. — real bugs / config errors that
    #        should fail loud per CLAUDE.md.
    transient_status_codes = frozenset({408, 425, 429, 500, 502, 503, 504, 529})

    max_outer_retries = 6  # 7 total attempts including the initial call
    delays = [30, 60, 120, 240, 480, 960]  # seconds; with jitter ±20%
    for attempt in range(max_outer_retries + 1):
        try:
            return _call_claude_once_base(*args, **kwargs)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            # Network-level transient; always retry.
            should_retry = True
            status_code: int | None = None
            err_name = type(exc).__name__
        except anthropic.APIStatusError as exc:
            # HTTP-status transient or permanent — gate by status code.
            status_code = getattr(exc, "status_code", None)
            err_name = type(exc).__name__
            should_retry = status_code in transient_status_codes
            if not should_retry:
                logger.error(
                    "Anthropic %s (status=%s) is NOT a documented transient class; "
                    "raising without retry: %s",
                    err_name,
                    status_code,
                    exc,
                )
                raise
        else:
            continue  # success — return path already taken inside try.

        if attempt >= max_outer_retries:
            logger.error(
                "Anthropic transient error (%s status=%s) after %d outer retries; giving up.",
                err_name,
                status_code,
                attempt,
            )
            raise
        sleep_for = delays[attempt] * (1.0 + random.uniform(-0.2, 0.2))
        logger.warning(
            "Anthropic transient error (%s status=%s); outer retry %d/%d after %.1fs sleep.",
            err_name,
            status_code,
            attempt + 1,
            max_outer_retries,
            sleep_for,
        )
        time.sleep(sleep_for)


def _collect_rewrites_with_empty_retry(
    batch_id: str,
    new_questions: list[str],
    *,
    max_drop_fraction: float = 0.05,
) -> dict[str, dict[str, str]]:
    """Wrap #502's `_collect_rewrites_results` with retry on empty/missing items.

    #502's collector raises `RuntimeError("rewrites batch missing N / M ...")`
    on the first batch item that returned with empty text or wasn't in the
    batch results at all. Its inline-retry path only covers PARSE failures.

    The Anthropic batch API has documented intermittent empty-completion
    behavior on individual items (round-5b launch died on 2/500 items
    returning empty despite the rest succeeding). This wrapper:

    1. Calls the inherited collector; on success returns its dict.
    2. On RuntimeError that names a shortfall <= `max_drop_fraction` of the
       target, reads the batch results ourselves, retries every empty/missing
       item individually via `_call_claude_once` (which now flows through the
       round-5b transient-5xx wrapper), and merges into the partial dict from
       step 1's failure trace.
    3. Items still failing after the inline retry are DROPPED (per plan §4
       Phase A retry policy: "if still failing after 3 → drop the probe").
       Hard floor: if final delivered_n / target_n < (1 - max_drop_fraction),
       re-raise the original shortfall.
    4. On shortfall > `max_drop_fraction` of the target, re-raise immediately
       (catastrophic batch failure — keep #502's fail-loud semantics).
    """
    import anthropic
    from issue502_generate_probes import (
        REWRITES_SYSTEM,
        _parse_rewrites_block,
        _rewrite_user_prompt,
    )

    try:
        return _collect_rewrites_results(batch_id, new_questions)
    except RuntimeError as exc:
        msg = str(exc)
        if "rewrites batch missing" not in msg:
            raise
        n = len(new_questions)
        # Re-read the batch ourselves so we can identify which items had
        # empty / missing text (the inherited function's error breakdown
        # already named them but the parsed dict was discarded with the raise).
        client = anthropic.Anthropic(api_key=_api_key())
        by_custom_id: dict[str, str] = {}
        for r in client.messages.batches.results(batch_id):
            if r.result.type != "succeeded":
                continue
            text = next((b.text for b in r.result.message.content if b.type == "text"), "")
            if text:
                by_custom_id[r.custom_id] = text

        # Parse all successfully-collected items first.
        out: dict[str, dict[str, str]] = {}
        for i, q in enumerate(new_questions):
            cid = f"rewrite-{i:04d}"
            text = by_custom_id.get(cid)
            if text is None:
                continue
            parsed = _parse_rewrites_block(text, q)
            if parsed is not None:
                out[q] = parsed

        n_initial_drops = n - len(out)
        if n_initial_drops > max_drop_fraction * n:
            # Catastrophic batch failure — preserve #502's fail-loud semantics
            # and re-raise the original.
            logger.error(
                "rewrites batch shortfall %d / %d exceeds %.0f%% threshold; "
                "preserving fail-loud raise.",
                n_initial_drops,
                n,
                100 * max_drop_fraction,
            )
            raise

        # Retry the missing items individually via the wrapped Claude path.
        missing_pairs = [(i, q) for i, q in enumerate(new_questions) if q not in out]
        logger.warning(
            "rewrites batch shortfall %d / %d (<= %.0f%% threshold); "
            "retrying inline via _call_claude_once.",
            n_initial_drops,
            n,
            100 * max_drop_fraction,
        )
        n_retry_recovered = 0
        n_retry_dropped = 0
        for i, q in missing_pairs:
            cid = f"rewrite-{i:04d}"
            try:
                text = _call_claude_once(
                    _rewrite_user_prompt(q),
                    max_tokens=1024,
                    system=REWRITES_SYSTEM,
                )
            except Exception as e:
                logger.warning("inline retry %s failed (%s); dropping.", cid, e)
                n_retry_dropped += 1
                continue
            parsed = _parse_rewrites_block(text, q)
            if parsed is None:
                logger.warning("inline retry %s parse-failed; dropping.", cid)
                n_retry_dropped += 1
                continue
            out[q] = parsed
            n_retry_recovered += 1

        final_delivered = len(out)
        final_drops = n - final_delivered
        if final_drops > max_drop_fraction * n:
            # Even after the retry the drop count exceeds the floor — fail loud.
            logger.error(
                "rewrites batch final drops %d / %d still exceed %.0f%% floor; "
                "(initial=%d, recovered=%d, retry-failed=%d).",
                final_drops,
                n,
                100 * max_drop_fraction,
                n_initial_drops,
                n_retry_recovered,
                n_retry_dropped,
            )
            raise
        logger.info(
            "rewrites batch tolerant collection OK: delivered=%d / %d "
            "(initial-drops=%d, recovered=%d, retry-failed=%d).",
            final_delivered,
            n,
            n_initial_drops,
            n_retry_recovered,
            n_retry_dropped,
        )
        return out


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

# Round-4 data-fix pivot (commit on issue-523): the round-3 launch crashed Phase A
# at `target_n=176` in the `capabilities` bucket with total dedup-rate 0.494
# (exact 0.223 + semantic 0.271) against the 580-item exclude corpus (500 #502
# probes + 30 #474 q_train + 50 #474 q_test). The dominant rejector was the
# SBERT cosine ≥ 0.9 semantic gate — Sonnet was producing genuine near-paraphrases
# of #502's questions because the 4 inherited buckets share #502's topic surface
# and that surface is largely exhausted at the 580-item scale.
#
# Plan §8 names the prescribed fix: "if > 20% duplication rate at generation,
# abort and revisit the bucket prompts." Plan §11 explicitly REJECTS loosening
# the disjointness threshold. So this round widens the topic surface in three
# complementary ways:
#   (1) the 4 inherited buckets carry NEW topic anchors that #502 did NOT touch
#       (specialized professional skills, niche scientific subdomains, advanced
#       craft techniques, foreign idioms, edge-case troubleshooting; etc.);
#   (2) TWO NEW buckets are added — `specialized_technical` and
#       `personal_planning` — that cover topic surface entirely outside #502's
#       4-bucket taxonomy;
#   (3) `_gen_prompt_diverse()` injects an anti-clustering instruction so each
#       Sonnet response stays internally varied within its bucket.
#
# Counts are rebalanced to keep total = 500, shifting mass from the older
# topic-surface-exhausted buckets onto the new ones:
#
#   capabilities         : 158 (#502) → 175 (round-3) → 130 (round-4)
#   opinion              : 113 (#502) → 125 (round-3) →  95 (round-4)
#   neutral_chat         : 113 (#502) → 125 (round-3) →  95 (round-4)
#   hypotheticals        :  66 (#502) →  75 (round-3) →  55 (round-4)
#   specialized_technical: NEW                          →  75 (round-4)
#   personal_planning    : NEW                          →  50 (round-4)
#                                                          ---
#                                                          500 ✓
#
# The four legacy bucket NAMES are preserved so downstream analyzers / audit
# JSONs / cross-experiment comparisons against #502 still resolve. Only the
# topic anchors inside their descriptions are widened.
BUCKETS: tuple[tuple[str, int, str], ...] = (
    (
        "capabilities",
        130,
        "Practical capability / how-to questions a curious person might ask an "
        "assistant. Push HARD for topic surface beyond the common cooking / "
        "software-setup / basic-troubleshooting anchors — those have been "
        "exhausted by prior 500-probe pools and will be rejected as semantic "
        "near-duplicates. Aim for: specialized professional procedure (legal "
        "filings, medical workflow, financial-instrument mechanics, regulatory "
        "interpretation), niche scientific sub-domains (geology, oceanography, "
        "astrophysics sub-areas, microbiology, soil science, glaciology), "
        "advanced craft techniques (woodworking joinery, ceramics glaze "
        "chemistry, leatherworking edge finishing, blacksmithing heat-treat), "
        "cultural / language-specific knowledge (foreign idioms, regional "
        "cuisines, traditional ceremonies, etiquette across cultures), edge-"
        "case troubleshooting (rare error states, multi-vendor integration "
        "failures, hardware corner cases). Examples: "
        "'How do glaciologists measure ice-core gas occlusion depth?', "
        "'What's the right way to season a damascus chef knife after etching?', "
        "'How do Italian notaries authenticate a foreign apostille?'",
    ),
    (
        "opinion",
        95,
        "Soft opinion / advice questions where a reasonable assistant offers a "
        "perspective. Push topic surface AWAY from common "
        "career / remote-work / book-recs anchors — those have been exhausted "
        "by prior pools. Aim for: opinion on specialized domain practice "
        "(should an architect specify CLT or steel for a 6-storey housing "
        "block?, is rotary or push mowing better for a tall-fescue lawn?), "
        "advice on niche life-stage choices (downsizing vs aging-in-place; "
        "whether to apprentice or attend trade school; sabbatical vs "
        "lateral-move), aesthetic / craft preferences (typography pairings, "
        "ceramic-glaze color choices, woodworking finish trade-offs), "
        "domain-specific debates (analog vs digital film mastering, "
        "single-malt vs blended whisky, freehand vs jig joinery). Examples: "
        "'Is hand-cut or machine-routed dovetailing more durable for a heirloom "
        "blanket chest?', "
        "'Would you recommend a Riesling or a Gewürztraminer pairing with "
        "Sichuan mapo tofu?', "
        "'Should a small-batch coffee roaster invest in a sample roaster or "
        "rent shared lab time?'",
    ),
    (
        "neutral_chat",
        95,
        "Neutral knowledge / chat — explanations, definitions, mild curiosity. "
        "Push topic surface AWAY from common quantum-entanglement / "
        "Roman-Empire / alligator-vs-crocodile anchors — those have been "
        "exhausted by prior pools. Aim for: niche scientific phenomena "
        "(retrograde precession, polymorphism in fluorite, etoliated stem "
        "growth), historical micro-periods (the Visigothic kingdom of Toulouse, "
        "Heian-period rituals, Cromwellian administration), language oddities "
        "(grammatical animacy, evidential markers, click consonants), niche "
        "engineering principles (HVAC zoning logic, RF impedance matching, "
        "weld penetration profiles), obscure cultural artifacts (Korean "
        "shamanic kut, Sardinian launeddas, Andean quipus). Examples: "
        "'What is the difference between a Doric and an Aeolic Greek dialect?', "
        "'Can you explain how a Wheatstone bridge balances unknown resistance?', "
        "'What is the role of evidentials in Quechua sentence structure?'",
    ),
    (
        "hypotheticals",
        55,
        "Hypotheticals / imaginative prompts — counterfactuals, scenarios, "
        "thought experiments. Push topic surface AWAY from common "
        "time-travel / writing-uninvented / live-anywhere anchors — those have "
        "been exhausted by prior pools. Aim for: domain-specific "
        "counterfactuals (what if Linnaean taxonomy preceded Aristotelian; "
        "what if the printing press arrived in Mesoamerica first; what if the "
        "Bessemer process was discovered in Heian Japan), discipline-specific "
        "thought experiments (an ethical dilemma in a specific clinical "
        "speciality; a physics scenario in a specific phase of matter; a "
        "linguistic scenario in a specific phonemic system), niche scenario-"
        "planning (managing a vineyard through a 5-year drought; running a "
        "single-screen cinema in a small town; designing a multi-generational "
        "research vessel). Examples: "
        "'What if Mendelian inheritance had been understood a century earlier?', "
        "'Imagine a vineyard manager facing a 5-year drought; what changes?', "
        "'If lithography had reached Edo Japan by 1700, how would ukiyo-e have "
        "evolved?'",
    ),
    (
        "specialized_technical",
        75,
        "NEW bucket — deep technical questions inside narrow specialist domains "
        "that the inherited 4 buckets undersample. Aim for: compiler internals "
        "(register allocation, escape analysis, JIT tiering), RF / antenna "
        "engineering (Smith-chart matching, log-periodic design, balun choice), "
        "structural / civil engineering (P-delta effects, soil-bearing capacity, "
        "post-tension grouting), bioinformatics pipelines (BWA-MEM vs minimap2, "
        "GATK joint-calling, GFF3 vs GTF), embedded systems / RTOS (priority "
        "inversion, ISR latency, watchdog patterns), distributed-systems "
        "internals (Raft log compaction, vector clocks, gossip protocols), "
        "materials-science specifics (austenitic vs ferritic stainless, "
        "Charpy-V vs Izod, fatigue-life curves), pharmacology kinetics "
        "(Michaelis-Menten vs Hill, first-pass metabolism, AUC vs Cmax). "
        "Examples: "
        "'How does a Raft cluster handle log compaction during a leadership "
        "election?', "
        "'What are the trade-offs between BWA-MEM and minimap2 for long-read "
        "alignment?', "
        "'How does P-delta amplification affect column design in a 12-storey "
        "moment frame?'",
    ),
    (
        "personal_planning",
        50,
        "NEW bucket — planning, logistics, and decision-shaping questions of "
        "the kind a person actually asks an assistant during real-life "
        "planning. Aim for: trip planning (mountain hike itineraries, "
        "multi-country rail-pass routing, off-season travel windows), financial "
        "planning (529 vs UTMA, Roth-conversion timing, mortgage-refi math), "
        "household decisions (heat pump vs gas furnace, replacing vs "
        "refinishing flooring, choosing a contractor), scheduling logistics "
        "(coordinating a multi-family beach week, planning a wedding rehearsal "
        "weekend, sequencing a kitchen renovation), career-move planning "
        "(negotiating a relocation package, evaluating a startup equity offer, "
        "deciding whether to take a sabbatical). Examples: "
        "'How should I sequence a kitchen renovation so we still have a "
        "working sink for two months?', "
        "'What's the best way to plan a 10-day Norwegian-fjords rail trip in "
        "shoulder season?', "
        "'How do I evaluate a startup-equity offer that vests on a 5-year "
        "schedule with a 1-year cliff?'",
    ),
)
assert sum(n for _, n, _ in BUCKETS) == DEFAULT_TARGET, (
    f"BUCKETS sum {sum(n for _, n, _ in BUCKETS)} != DEFAULT_TARGET {DEFAULT_TARGET}; "
    "rebalance the round-4 per-bucket targets."
)

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
# Round-5 fix: 3 retries was too tight for stubborn paraphrase classes — round-4d
# Phase A converged 498/500 probes but burned all 3 retries on 2 probes where Sonnet
# kept regenerating first-person paraphrases despite explicit retry-reason hints.
# 6 retries gives Sonnet enough attempts to break out of the locked-in first-person
# pattern. Each extra retry costs ~1.5 Claude calls per stuck probe (one rewrite +
# one validator pass), so worst-case extra spend is bounded at ~6 * 1.5 * n_stuck
# Claude calls — negligible vs a full 5 h Phase A.
MAX_VOICE_DRIFT_RETRIES = 6
VOICE_DRIFT_AUDIT_SAMPLE = 50
VOICE_DRIFT_THIRD_PERSON_RATE_MIN = 0.96

# Round-5 fix: backfill budget. If after `MAX_VOICE_DRIFT_RETRIES` per-probe retries
# any probes remain stuck, fall back to replacing the stuck probes with FRESH probes
# from the same bucket (re-running `_generate_bucket_with_disjointness` for one more
# probe per stuck question, then re-rewriting + re-validating). This preserves the
# `n_dropped_probes == 0` gate's contract: the final pool is exactly 500 probes,
# every one of which PASSed voice-drift validation, with the per-bucket composition
# unchanged. Backfill is capped at MAX_BACKFILL_ROUNDS rounds, each handling all
# currently-stuck probes; a fresh probe that ALSO fails goes back into the stuck
# pile for the next round.
MAX_BACKFILL_ROUNDS = 3


# ────────────────────────── Round-4 diversified generation prompt ──────────────────────────


def _gen_prompt_diverse(
    bucket_name: str, bucket_desc: str, n: int, exclude_examples: list[str]
) -> str:
    """Round-4 replacement for `issue502_generate_probes._gen_prompt`.

    Adds an anti-clustering directive: AT MOST ~3 questions may share a primary
    sub-topic, and Sonnet must push for novel sub-topics not yet covered in the
    bucket. This is the third leg of the round-4 fix (alongside widened bucket
    descriptions and the two new buckets) — without it, Sonnet tends to cluster
    each batch around one or two anchor topics from the description, which the
    SBERT cosine ≥ 0.9 gate then rejects as near-paraphrases of #502's existing
    pool.

    Same signature as `_gen_prompt` so the bucket-generator call site swaps in
    cleanly with one identifier change.
    """
    ex_str = "\n".join(f"- {q}" for q in exclude_examples[:30])
    return (
        f"Generate {n} probe questions in the category '{bucket_name}'.\n\n"
        f"Category description:\n{bucket_desc}\n\n"
        "Do NOT repeat or paraphrase any of these existing questions "
        f"(we already have them):\n{ex_str}\n\n"
        "DIVERSITY REQUIREMENT (important — prior batches were rejected as "
        "near-duplicates of an existing 500-question pool):\n"
        " - Within this batch, AT MOST 3 questions may share a primary "
        "sub-topic (e.g. at most 3 questions about Python, at most 3 about "
        "Italian cooking, at most 3 about home repair). Push for novel "
        "sub-topics not yet covered in the bucket description's anchor list.\n"
        " - Treat the example anchors in the category description as a "
        "minimum-diversity floor, not a target — fan out beyond them.\n"
        " - If two candidate questions feel like paraphrases (same primary "
        "topic, similar phrasing), keep only one and replace the other with "
        "a question on a different sub-topic.\n\n"
        f"Output exactly {n} new questions, one per line, no numbering, no "
        "blank lines, no preamble. Each must end with '?' and stay on one "
        "line."
    )


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
    """One Sonnet call asking for a third-person indirect rewrite.

    Returns the first non-empty line of the response, stripped. If Claude
    returns an empty completion (round-5d-observed: occasional empty responses
    after a 529-overload cleared on the same prompt), returns "" — the
    `_validate_and_fix_indirect_rewrite` loop's Claude validator will reject
    the empty string and trigger the existing 3-attempt retry path, so the
    overall budget per probe is unchanged.
    """
    suffix = ""
    if retry_reason:
        suffix = (
            f"\n\nIMPORTANT: a previous attempt failed validation: {retry_reason}. "
            "Try again, strictly in the third person."
        )
    user_prompt = f"Original question:\n{question}\n\nProduce the indirect rewrite.{suffix}"
    raw = _call_claude_once(
        user_prompt=user_prompt,
        max_tokens=512,
        system=INDIRECT_REGISTER_FIX_SYSTEM,
    ).strip()
    lines = raw.splitlines()
    return lines[0].strip() if lines else ""


def _validate_and_fix_indirect_rewrite(
    question: str,
    initial_rewrite: str,
    skip_claude: bool = False,
) -> tuple[str, bool, list[str]]:
    """Validate an indirect rewrite, retrying up to MAX_VOICE_DRIFT_RETRIES times.

    Returns (final_rewrite, voice_drift_failed, attempts_log).

    voice_drift_failed=True signals "all retries exhausted, mark in audit but
    keep the row" — per plan §4 Phase A, we do NOT silently drop.

    skip_claude=True (smoke mode) uses only the regex validator and the retry
    path returns a deterministic placeholder (no Sonnet API call). A smoke run
    that POISONS the initial rewrite (`--smoke-force-voice-drift-fail`) will
    therefore exhaust all retries deterministically and surface as a drop —
    exactly the path the round-5 backfill loop is built to test.
    """
    attempts: list[str] = []
    current = initial_rewrite

    def _retry(reason: str) -> str:
        # In smoke mode never call Sonnet; return a deterministic placeholder
        # that ALSO contains the first-person token so the regex keeps failing.
        # This makes the smoke test deterministically reach `voice_drift_failed`
        # for any poisoned input within MAX_VOICE_DRIFT_RETRIES iterations.
        if skip_claude:
            return f"[smoke-retry] I still want: {question}"
        return _request_indirect_rewrite(question, retry_reason=reason)

    for attempt in range(MAX_VOICE_DRIFT_RETRIES + 1):
        regex_fail = _has_first_person(current)
        if regex_fail:
            attempts.append(f"attempt{attempt}: regex FAIL ({current!r})")
            if attempt >= MAX_VOICE_DRIFT_RETRIES:
                return current, True, attempts
            current = _retry("first-person pronoun detected by regex")
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
        current = _retry("Claude judged the rewrite as not third-person")
    # Unreachable (loop returns at every branch), but mypy/ruff appreciate it.
    return current, True, attempts


# ────────────────────────── SBERT semantic dedup ──────────────────────────


_SBERT_MODEL = None  # lazy module-level singleton


def _get_sbert():
    """Lazy-load the SBERT model. FAIL-LOUD on missing dependency.

    Plan §11 explicitly rejects exact-only dedup as the disjointness contract.
    `sentence-transformers` is a hard requirement (in `pyproject.toml`); a missing
    import means the env is broken and we must raise before burning $150-200 of
    Sonnet API on a pool whose disjointness is exact-string only. Only catches
    `ImportError` — `OSError` / network failures from the model download propagate.
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers required for plan §4 Phase A semantic dedup; "
                "install with 'uv add sentence-transformers' (already in pyproject.toml — "
                "run 'uv sync' on the pod). Plan §11 explicitly rejects exact-only as the "
                "disjointness contract."
            ) from e
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_ID)
    return _SBERT_MODEL


def _semantic_duplicate_check(
    candidates: list[str],
    existing_texts: list[str],
    threshold: float = SBERT_THRESHOLD,
) -> list[bool]:
    """For each candidate, return True iff it has SBERT cosine ≥ threshold to ANY existing.

    Length of return matches `candidates`. Raises on missing `sentence-transformers`
    (the silent-disable was the round-2 critical bug — plan §11 rejects exact-only).
    """
    sbert = _get_sbert()
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
        prompt = _gen_prompt_diverse(bucket_name, bucket_desc, ask, sorted(existing_norms)[:30])
        if smoke:
            # Smoke mode: synthesize candidates that pass dedup trivially.
            # The `nonce` is the running existing-pool size so a second call
            # to this generator (e.g. the round-5 backfill loop) cannot
            # collide with synth probes already accepted in the first call —
            # without it, smoke backfill trips the SBERT cosine ≥ 0.9 gate
            # against the first call's identical-template probes.
            nonce = len(existing_texts)
            candidates = [
                f"[smoke-{bucket_name}-n{nonce:05d}-{attempts_used:02d}-{i:03d}] "
                f"What is fact {bucket_name} number {nonce}-{i} for the "
                f"{bucket_name} batch?"
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
        # SKIPPED in smoke mode: synthesized template strings ("What is fact
        # capabilities number N?") are SBERT-similar to each other by template
        # structure alone, which would block the round-5 backfill loop's smoke
        # exercise from succeeding (the backfill regenerates from the same
        # synthesizer, and the synthesizer cannot escape its own template's
        # SBERT-similarity to the first-call probes). The semantic-dedup gate
        # is meaningful against real Sonnet output only; use --smoke-real-api
        # for a tiny-slice run that does exercise it.
        if not survivors_after_exact:
            survivors_after_sem: list[str] = []
        elif smoke:
            # Smoke mode bypasses SBERT (see comment above); accept exact-pass
            # candidates straight through. The exact-string `_normalize` gate
            # above still keeps the smoke pool internally disjoint.
            survivors_after_sem = list(survivors_after_exact)
        else:
            dup_flags = _semantic_duplicate_check(
                survivors_after_exact, existing_texts, threshold=SBERT_THRESHOLD
            )
            survivors_after_sem = []
            for cand, is_dup in zip(survivors_after_exact, dup_flags, strict=True):
                if is_dup:
                    semantic_dedup_rejects += 1
                else:
                    survivors_after_sem.append(cand)

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
    smoke_force_voice_drift_fail: int = 0,
    smoke_force_rewrite_missing: int = 0,
) -> tuple[dict[str, dict[str, str]], dict, list[str]]:
    """Run the batched rewrites flow, then re-validate the indirect register.

    Returns (rewrites, audit_block, dropped_probes).

    - `rewrites` is keyed on the SURVIVING probes only (dropped ones are removed
      from `new_probes` in-place and from the rewrites dict). Two failure modes
      both feed `dropped_probes`: (a) rewrite-missing — the Anthropic Batch API
      returned no parseable rewrite for the probe (tolerated up to 5% by
      `_collect_rewrites_with_empty_retry`, round-5c); (b) voice-drift —
      `_validate_and_fix_indirect_rewrite` exhausted all MAX_VOICE_DRIFT_RETRIES
      retries on a first-person leak. Both flow into the same backfill loop in
      `main()`.
    - `audit_block` contains n_questions, n_indirect_regex_pass_total,
      n_indirect_claude_pass_total, n_indirect_voice_drift_failed,
      n_initial_rewrite_missing (NEW — round 7), n_dropped_probes,
      indirect_third_person_*_rate (rates computed over n_total +
      voice_drift drops so they measure validator efficacy on probes that
      actually reached the validator — NOT diluted by upstream rewrite-batch
      drops), and a `voice_drift_failed` detail list for the audit JSON.
    - `dropped_probes` is the FULL list of every question dropped under either
      failure mode. The caller uses this list to drive the backfill loop
      (regenerate one fresh probe per dropped probe from the same bucket, run
      rewrites + validator on the fresh probe). Round-5 fix (voice-drift),
      extended round 7 (rewrite-missing).
    - `smoke_force_rewrite_missing` (round 7, smoke-only): deterministically
      DELETE the LAST N rewrite keys after the synth call, so the round-7
      rewrite-missing branch is exercised end-to-end on the dispatcher. The
      LAST N (not the FIRST N) is chosen to avoid collision with
      `smoke_force_voice_drift_fail` which mutates the FIRST N.
    """
    if smoke:
        # Smoke: synthetic rewrites with the existing helper, but force the
        # indirect register through the validator path so the wiring is tested.
        rewrites = _smoke_synthetic_rewrites(new_probes)
        # Round-5 smoke gate: optionally poison the first N indirect rewrites with
        # an unambiguous first-person pronoun so the regex validator rejects them.
        # The retry path will re-request from Sonnet (but in smoke mode there is no
        # Sonnet — the synthesizer returns the same deterministic string), so all
        # MAX_VOICE_DRIFT_RETRIES will exhaust and the probe will be added to
        # `dropped_probes`. This exercises the backfill path in main() end-to-end.
        if smoke_force_voice_drift_fail > 0:
            for q in new_probes[:smoke_force_voice_drift_fail]:
                rewrites[q]["indirect"] = f"I am still wondering about: {rewrites[q]['indirect']}"
        # Round-7 smoke gate: optionally DELETE the LAST N rewrite entries so the
        # rewrite-missing branch (the `_collect_rewrites_with_empty_retry` tolerant
        # path's residue, which is normally never exercised on the synth path) is
        # exercised end-to-end. Uses the LAST N rather than the FIRST N so it
        # never collides with `smoke_force_voice_drift_fail`.
        if smoke_force_rewrite_missing > 0:
            for q in new_probes[-smoke_force_rewrite_missing:]:
                rewrites.pop(q, None)
    else:
        requests = _build_rewrites_requests(new_probes)
        batch_id = _submit_rewrites_batch(requests)
        _wait_for_rewrites_batch(batch_id)
        # Round-5c push-through fix: #502's `_collect_rewrites_results` raises
        # hard if ANY batch item came back with empty-text or missing-from-batch
        # (it only retries inline on PARSE failures, not on empty/missing). The
        # Anthropic batch API has documented intermittent empty-completion
        # behavior; round-5b's launch died on 2/500 items returning empty
        # despite the surrounding pipeline being healthy. We wrap the inherited
        # collector: on the shortfall RuntimeError, identify the missing items
        # by their custom_id and retry each individually via the (already-
        # round-5b-wrapped) `_call_claude_once`. Hard floor: accept up to 5%
        # drops (25/500); above that the original raise is preserved.
        rewrites = _collect_rewrites_with_empty_retry(batch_id, new_probes)

    # ── Round-7 fix: sync `new_probes` and `rewrites.keys()` BEFORE the
    # validation loop. `_collect_rewrites_with_empty_retry` tolerates up to 5%
    # batch-API empty/parse-failed items (round-5c); on those drops the
    # returned `rewrites` dict has fewer keys than `new_probes`. The validation
    # loop below indexes `rewrites[q]["indirect"]` and crashes with KeyError
    # on the first missing key (round-6 launch crashed on a 1/500 drop the
    # tolerant collector accepted but never propagated to `new_probes`).
    #
    # Fix: detect rewrite-missing here, treat each as a DROP (same channel as
    # voice-drift drops), and remove them from `new_probes`. The downstream
    # backfill loop in main() then regenerates a same-bucket replacement for
    # each automatically (no caller change needed — `dropped_probes` already
    # drives backfill via `dropped_initial`).
    rewrite_missing: list[str] = [q for q in new_probes if q not in rewrites]
    if rewrite_missing:
        logger.warning(
            "rewrite-missing detected on %d / %d probes (round-5c tolerant "
            "collector accepted these as <=5%% drops); routing to backfill "
            "alongside voice-drift drops.",
            len(rewrite_missing),
            len(new_probes),
        )
        # Prune new_probes in place so the upstream pool, the validation loop,
        # the regex/Claude sweep, and the n_total counter all agree.
        kept = [q for q in new_probes if q in rewrites]
        new_probes.clear()
        new_probes.extend(kept)

    # Validate + (when needed) retry every indirect-register rewrite.
    # Round-2 fix to Critical-5: any rewrite that fails ALL retries is
    # DROPPED from the output (not retained with `voice_drift_failed=True`)
    # so the downstream audit reads `n_indirect_voice_drift_failed == 0`
    # rather than the round-1 silent-retain behavior. The final-set gate
    # below fails LOUD if any drops occurred.
    #
    # Round-7: pre-populate `dropped_probes` with rewrite-missing drops so they
    # flow through the same `dropped_initial` → backfill channel as voice-drift
    # drops. `voice_drift_failed` stays empty for these (they never reached the
    # validator); only voice-drift exhaustions append to that list.
    regex_pass_count = 0
    claude_pass_count = 0
    voice_drift_failed: list[dict] = []
    dropped_probes: list[str] = list(rewrite_missing)
    n_initial_rewrite_missing = len(rewrite_missing)
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

    # Rate denominator: probes that ACTUALLY reached the validator = n_total
    # (survivors) + voice-drift drops. Rewrite-missing drops never reached the
    # validator, so excluding them keeps the rate "validator efficacy" rather
    # than diluting it with upstream rewrite-batch drops.
    rate_denom = max(n_total + len(voice_drift_failed), 1)
    return (
        rewrites,
        {
            "n_questions": n_total,
            "n_indirect_regex_pass_total": regex_pass_count,
            "n_indirect_claude_pass_total": claude_pass_count,
            "n_indirect_voice_drift_failed": len(voice_drift_failed),
            # Round 7: distinct counter for rewrite-batch drops (Anthropic
            # Batch API empty/parse-failed items) so downstream phases can
            # tell rewrite-missing apart from voice-drift exhaustion.
            "n_initial_rewrite_missing": n_initial_rewrite_missing,
            "n_dropped_probes": len(dropped_probes),
            "indirect_third_person_regex_rate": regex_pass_count / rate_denom,
            "indirect_third_person_claude_rate": claude_pass_count / rate_denom,
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
            "rewrite_missing_examples": rewrite_missing[:5],  # first 5 only
            "skip_claude": skip_claude,
        },
        dropped_probes,
    )


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
            "pool. Used for end-to-end smoke ahead of the real run. NOTE: "
            "this path CANNOT exercise the SBERT semantic-dedup gate against "
            "real Sonnet output — use --smoke-real-api for that."
        ),
    )
    p.add_argument(
        "--smoke-real-api",
        action="store_true",
        help=(
            "Round-4 smoke gate: use the REAL Claude API to generate "
            "candidates on a tiny slice (typically --n 60), but write to "
            "*.smoke.json paths so the run cannot clobber a real pool. "
            "Exercises the SBERT semantic-dedup gate end-to-end against "
            "real Sonnet output — the failure mode that bit Phase A in "
            "round 3. Mutually exclusive with --smoke-only. Cost ~$1-2 per "
            "60-probe smoke run."
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
    p.add_argument(
        "--smoke-force-voice-drift-fail",
        type=int,
        default=0,
        help=(
            "Smoke-mode-only: force the first N synthesized probes to FAIL the "
            "voice-drift validator (by re-writing their indirect rewrite to "
            "contain a first-person pronoun) so the round-5 backfill path is "
            "exercised end-to-end on the dispatcher. Requires --smoke-only. "
            "Each forced-fail probe is replaced by a fresh probe from the same "
            "bucket; the final pool should contain exactly --n probes with "
            "n_dropped_probes == 0. NEVER use on the real run."
        ),
    )
    p.add_argument(
        "--smoke-force-rewrite-missing",
        type=int,
        default=0,
        help=(
            "Smoke-mode-only: DELETE the rewrite entry for the LAST N "
            "synthesized probes so the round-7 rewrite-missing branch (the "
            "tolerant collector's <=5%% drop residue, normally never exercised "
            "on the synth path) is exercised end-to-end on the dispatcher. "
            "Requires --smoke-only. Each missing-rewrite probe is replaced by "
            "a fresh probe from the same bucket via the same backfill path "
            "as voice-drift drops; the final pool should contain exactly --n "
            "probes with n_dropped_probes == 0. Uses the LAST N (not the "
            "FIRST N) to avoid collision with --smoke-force-voice-drift-fail. "
            "NEVER use on the real run."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — sequential phase orchestrator
    args = _build_argparser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke_only and args.smoke_real_api:
        raise SystemExit("--smoke-only and --smoke-real-api are mutually exclusive")

    # Round-6 fix (Codex Critical): fail fast on the smoke-only flag BEFORE any
    # API-spending code runs. The earlier placement (after the bucket-generation
    # loop) burned ~$10-50 of real Sonnet calls on a user-typed conflict command
    # before raising SystemExit. The flag is mutually exclusive with the real
    # run regardless of --skip-rewrites: it is a smoke-mode-only
    # validator-poisoning lever.
    if args.smoke_force_voice_drift_fail and not bool(args.smoke_only):
        raise SystemExit(
            "--smoke-force-voice-drift-fail requires --smoke-only (it is a "
            "smoke-mode-only validator-poisoning lever, not a real-run flag)"
        )
    # Round-7 fail-fast guard: same shape as the voice-drift guard above.
    # Fires BEFORE any API-spending code so a fat-fingered real-run invocation
    # with the smoke flag set burns $0 of Sonnet calls.
    if args.smoke_force_rewrite_missing and not bool(args.smoke_only):
        raise SystemExit(
            "--smoke-force-rewrite-missing requires --smoke-only (it is a "
            "smoke-mode-only rewrite-missing-poisoning lever, not a real-run flag)"
        )

    # Two ORTHOGONAL switches:
    #   use_synth         — True => use the placeholder synthesizer for both
    #                       candidate generation AND rewrites (no Claude API
    #                       calls). False => hit the real Claude API.
    #   use_smoke_paths   — True => write to *.smoke.json output paths so a
    #                       smoke / tiny-slice run cannot overwrite the real
    #                       500-probe pool / audit / rewrites artifacts.
    #
    # Flag combinations (round-4):
    #   default                  : use_synth=False, use_smoke_paths=False  (real run)
    #   --smoke-only             : use_synth=True,  use_smoke_paths=True   (CPU smoke)
    #   --smoke-real-api         : use_synth=False, use_smoke_paths=True   (real-API
    #                              tiny-slice smoke gate; the round-4 mode that
    #                              actually exercises the SBERT dedup gate end-to-end)
    #
    # The local `smoke` variable retained below tracks `use_synth` because the
    # downstream voice-drift gates and bucket/rewrites generators key off the
    # "synthesizer-vs-real-API" axis, NOT the output-path axis.
    use_synth = bool(args.smoke_only)
    use_smoke_paths = bool(args.smoke_only) or bool(args.smoke_real_api)
    smoke = use_synth  # back-compat alias for the downstream gate code below
    if use_smoke_paths and args.n > 60:
        logger.warning(
            "smoke-paths mode with --n=%d is large; smoke is normally --n 12 "
            "(synth) or --n 60 (real-api). Continuing.",
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
    # Round-5 fix: maintain a question → bucket_name map so the backfill path
    # (after the voice-drift validator drops stuck probes) can regenerate the
    # replacement probe(s) from the SAME bucket the dropped probe came from —
    # preserving the at-plan per-bucket composition (130/95/95/55/75/50).
    probe_to_bucket: dict[str, str] = {}
    bucket_descs: dict[str, str] = {bname: bdesc for bname, _, bdesc in buckets_active}
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
        for q in accepted:
            probe_to_bucket[q] = bname
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

    out_path = SMOKE_PROBES_PATH if use_smoke_paths else PROBES_PATH

    # ── Class-D rewrites (with voice-drift fix on indirect) ──
    # NOTE (Mi1 round-3 fix): we GENERATE rewrites here but DEFER writing the
    # rewrites + audit + pool artifacts until AFTER the voice-drift gate fires.
    # This prevents a failed run from leaving plausible-looking partial artifacts
    # on disk that downstream phases could pick up.
    rewrites: dict[str, dict[str, str]] = {}
    voice_drift_audit: dict = {}
    rewrites_out: Path | None = None
    meta_path: Path | None = None
    meta_payload: dict | None = None
    if not args.skip_rewrites:
        logger.info("Generating Class-D rewrites for %d new probes…", len(new_probes))
        # (smoke-force-voice-drift-fail guard moved to top of main() in round 6
        # so the flag fails fast before any API-spending code runs.)
        rewrites, voice_drift_audit, dropped_initial = _generate_rewrites_with_voice_drift_fix(
            new_probes,
            smoke=smoke,
            smoke_force_voice_drift_fail=args.smoke_force_voice_drift_fail,
            smoke_force_rewrite_missing=args.smoke_force_rewrite_missing,
        )

        # ── Round-5 backfill loop ──
        # If any probes exhausted MAX_VOICE_DRIFT_RETRIES retries, replace each
        # with a fresh probe from the SAME bucket (preserves per-bucket
        # composition), run the rewrites + voice-drift validator on those
        # fresh probes, and merge the survivors back into the main pool.
        # Repeat up to MAX_BACKFILL_ROUNDS rounds; a fresh probe that ALSO
        # fails goes back into the stuck pile for the next round.
        #
        # The gate at the bottom of main() still reads n_dropped_probes from
        # the FINAL audit and raises if any drops remain after all backfill
        # rounds — the gate's contract (n_dropped_probes == 0) stays intact.
        backfill_rounds_used = 0
        backfill_total_added = 0
        backfill_total_attempts = 0
        backfill_log: list[dict] = []
        # Round-6 (Codex Major): one inner audit block per backfill round,
        # captured from each `_generate_rewrites_with_voice_drift_fix` call so
        # the post-loop combination does NOT re-run the Claude validator.
        backfill_round_audits: list[dict] = []
        stuck_probes = list(dropped_initial)
        while stuck_probes and backfill_rounds_used < MAX_BACKFILL_ROUNDS:
            backfill_rounds_used += 1
            # Group stuck probes by their originating bucket so we regenerate
            # replacements bucket-by-bucket (preserves the at-plan composition).
            stuck_by_bucket: dict[str, int] = {}
            for q in stuck_probes:
                b = probe_to_bucket.get(q)
                if b is None:
                    # A backfill probe added in a prior round and now stuck —
                    # its bucket assignment was recorded when it joined the pool.
                    raise AssertionError(
                        f"backfill round {backfill_rounds_used}: stuck probe "
                        f"{q!r} has no recorded bucket — bookkeeping bug"
                    )
                stuck_by_bucket[b] = stuck_by_bucket.get(b, 0) + 1

            logger.warning(
                "Round-5 backfill %d/%d: %d stuck probes; regenerating from buckets %s",
                backfill_rounds_used,
                MAX_BACKFILL_ROUNDS,
                len(stuck_probes),
                dict(stuck_by_bucket),
            )

            # Regenerate replacements per bucket. _generate_bucket_with_disjointness
            # appends to existing_norms / existing_texts so the new probes stay
            # disjoint from every previously-accepted probe (including the
            # stuck ones we are now retiring).
            backfill_probes: list[str] = []
            for bname, n_needed in stuck_by_bucket.items():
                bdesc = bucket_descs[bname]
                fresh, fresh_audit = _generate_bucket_with_disjointness(
                    bname,
                    bdesc,
                    n_needed,
                    existing_norms,
                    existing_texts,
                    chunk_size=args.chunk_size,
                    smoke=smoke,
                )
                for q in fresh:
                    probe_to_bucket[q] = bname
                backfill_probes.extend(fresh)
                backfill_log.append(
                    {
                        "round": backfill_rounds_used,
                        "bucket": bname,
                        "n_needed": n_needed,
                        "n_generated": len(fresh),
                        "fresh_bucket_audit": fresh_audit,
                    }
                )
            backfill_total_attempts += len(backfill_probes)

            # Rewrite + validate the fresh probes. Use the same dispatcher
            # function so the validator + retry path are identical to the
            # main pass. Smoke poisoning is NOT propagated to backfill
            # rounds (otherwise the smoke test would loop forever).
            fresh_rewrites, fresh_audit_block, fresh_dropped = (
                _generate_rewrites_with_voice_drift_fix(
                    backfill_probes,
                    smoke=smoke,
                    smoke_force_voice_drift_fail=0,
                )
            )
            # backfill_probes was mutated in-place by the call (drops removed);
            # `fresh_rewrites` is keyed on the survivors.
            n_added_this_round = len(backfill_probes)
            backfill_total_added += n_added_this_round
            # Merge survivors into the main pool / rewrites dict, replacing the
            # stuck probes one-for-one. The `new_probes` list was already
            # pruned of the stuck probes by the original validator call, so we
            # simply extend.
            new_probes.extend(backfill_probes)
            rewrites.update(fresh_rewrites)

            # Round-6 fix (Codex Major): retain the per-round inner audit block
            # so the post-loop combination can reuse its `full_*_pass` counts
            # instead of re-running the Claude validator on every probe in the
            # final pool. The dispatcher already ran the regex + Claude gates
            # inside `_generate_rewrites_with_voice_drift_fix`; recounting is
            # ~500 duplicate sequential Claude calls per happy-path run.
            # Attach to every bucket-entry added in this round so per-round
            # bookkeeping stays consistent (the audit is per-round, not
            # per-bucket).
            for entry in backfill_log:
                if entry["round"] == backfill_rounds_used:
                    entry["fresh_voice_drift_audit"] = fresh_audit_block
            backfill_round_audits.append(
                {"round": backfill_rounds_used, "audit": fresh_audit_block}
            )

            # Update the stuck pile for the next round.
            stuck_probes = list(fresh_dropped)
            logger.info(
                "Round-5 backfill round %d done: %d replacements added, "
                "%d still stuck (will retry next round if budget remains)",
                backfill_rounds_used,
                n_added_this_round,
                len(stuck_probes),
            )

        # Round-6 fix (Codex Major): combine the inner audit blocks instead of
        # re-running the Claude validator on every probe in the final pool.
        # Each `_generate_rewrites_with_voice_drift_fix` call already ran the
        # regex + Claude gates on its surviving probes (lines 1074-1088 in the
        # function body) and returned the per-call counts in its audit block.
        # The combined audit = sum the per-batch numerators (full_*_pass) and
        # denominators (full_audit_size), then recompute the rates from the
        # totals. The kept rewrites' voice-drift status is by construction
        # known — no validator re-run needed.
        #
        # When backfill_rounds_used == 0 the initial inner audit already covers
        # the whole final pool; we just attach the `backfill` sub-dict below.
        if backfill_rounds_used > 0:
            initial_audit = dict(voice_drift_audit)  # snapshot of the round-0 inner audit
            all_audits = [initial_audit] + [a["audit"] for a in backfill_round_audits]
            sum_full_audit_size = sum(a.get("full_audit_size", 0) for a in all_audits)
            sum_full_regex_pass = sum(a.get("full_regex_pass", 0) for a in all_audits)
            sum_full_claude_pass = sum(a.get("full_claude_pass", 0) for a in all_audits)
            final_n = len(new_probes)
            # Invariant: surviving-probe counts across the per-round audits sum
            # to the final pool size. If this trips it indicates a bookkeeping
            # bug between `_generate_rewrites_with_voice_drift_fix` and the
            # main pool — fail loud rather than silently report wrong rates.
            assert sum_full_audit_size == final_n, (
                f"backfill audit-size accounting bug: sum(full_audit_size)="
                f"{sum_full_audit_size} != len(new_probes)={final_n} "
                f"(rounds_used={backfill_rounds_used})"
            )
            voice_drift_audit["n_questions"] = final_n
            voice_drift_audit["n_dropped_probes"] = len(stuck_probes)
            voice_drift_audit["full_audit_size"] = final_n
            voice_drift_audit["full_regex_pass"] = sum_full_regex_pass
            voice_drift_audit["full_regex_pass_rate"] = sum_full_regex_pass / max(final_n, 1)
            voice_drift_audit["full_claude_pass"] = sum_full_claude_pass
            voice_drift_audit["full_claude_pass_rate"] = sum_full_claude_pass / max(final_n, 1)
            voice_drift_audit["audit_sample_size"] = final_n
            voice_drift_audit["audit_sample_regex_pass"] = sum_full_regex_pass
            voice_drift_audit["audit_sample_regex_pass_rate"] = sum_full_regex_pass / max(
                final_n, 1
            )
        # else: backfill_rounds_used == 0 -- the initial voice_drift_audit dict
        # from `_generate_rewrites_with_voice_drift_fix(new_probes, ...)`
        # already covers the entire final pool (no probes were stuck on the
        # first pass, so new_probes == initial_audit.full_audit_size). No
        # recount, no Claude calls.
        # Round-7: surface a breakdown of round-0's drops by failure mode so
        # Phase B/C/D can tell rewrite-missing apart from voice-drift in the
        # audit JSON. The initial inner audit recorded both counts; backfill
        # rounds (smoke_force_*=0) only ever contribute voice-drift drops.
        n_initial_rewrite_missing_round0 = voice_drift_audit.get("n_initial_rewrite_missing", 0)
        # Voice-drift drops at round 0 = total initial drops - rewrite-missing.
        # All later rounds' drops are voice-drift only.
        n_initial_voice_drift_round0 = len(dropped_initial) - n_initial_rewrite_missing_round0
        voice_drift_audit["backfill"] = {
            "max_backfill_rounds": MAX_BACKFILL_ROUNDS,
            "rounds_used": backfill_rounds_used,
            "n_initial_drops": len(dropped_initial),
            # Round 7: per-mode breakdown of round-0 drops. backfill rounds
            # contribute 0 to `n_initial_rewrite_missing` by construction
            # (they call the function with smoke_force_rewrite_missing=0).
            "n_initial_rewrite_missing": n_initial_rewrite_missing_round0,
            "n_initial_voice_drift_failed": n_initial_voice_drift_round0,
            "n_replacements_attempted": backfill_total_attempts,
            "n_replacements_added": backfill_total_added,
            "n_still_stuck": len(stuck_probes),
            "per_round": backfill_log,
        }

        # Post-backfill invariants: pool size and per-bucket composition.
        # If backfill succeeded we should be back at args.n probes; if it
        # didn't, the gate below raises with a useful diagnostic.
        # Per-bucket count check is best-effort (only when no stuck probes).
        if not stuck_probes:
            assert len(new_probes) == args.n, (
                f"post-backfill pool size {len(new_probes)} != target {args.n}; "
                f"bookkeeping bug. backfill rounds={backfill_rounds_used} "
                f"initial_drops={len(dropped_initial)} added={backfill_total_added}"
            )
            final_counts: dict[str, int] = {}
            for q in new_probes:
                b = probe_to_bucket[q]
                final_counts[b] = final_counts.get(b, 0) + 1
            for bname, btarget, _ in buckets_active:
                assert final_counts.get(bname, 0) == btarget, (
                    f"post-backfill bucket {bname!r} count "
                    f"{final_counts.get(bname, 0)} != target {btarget}; "
                    f"bookkeeping bug. final_counts={final_counts}"
                )

        rewrites_out = SMOKE_REWRITES_EXTENSION_PATH if use_smoke_paths else REWRITES_EXTENSION_PATH
        # Companion meta sidecar (matches #502's schema).
        meta_path = rewrites_out.with_suffix(".meta.json")
        meta_payload = {
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

    # ── Final per-bucket composition (post-backfill) for the audit ──
    # Each bucket's initial delivered_n already equals target_n (the
    # bucket-generator runs until it hits the target). Backfill replaces stuck
    # probes one-for-one from the SAME bucket, so the final bucket sizes still
    # equal target_n — this re-tally is belt-and-suspenders.
    final_bucket_counts: dict[str, int] = {}
    for q in new_probes:
        b = probe_to_bucket.get(q)
        if b is not None:
            final_bucket_counts[b] = final_bucket_counts.get(b, 0) + 1

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
        "final_bucket_counts": final_bucket_counts,
        "voice_drift_audit": voice_drift_audit,
        "sbert_model_id": SBERT_MODEL_ID,
        "sbert_threshold": SBERT_THRESHOLD,
        "voice_drift_third_person_rate_min": VOICE_DRIFT_THIRD_PERSON_RATE_MIN,
        "max_voice_drift_retries": MAX_VOICE_DRIFT_RETRIES,
        "max_backfill_rounds": MAX_BACKFILL_ROUNDS,
        "provenance": {
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
            "python": platform.python_version(),
            "elapsed_seconds": round(elapsed, 2),
        },
    }
    audit_out = SMOKE_AUDIT_PATH if use_smoke_paths else AUDIT_PATH

    # ── End-of-Phase-A gates (run BEFORE writing non-smoke artifacts) ──
    # Mi1 round-3 fix: a failed gate must NOT leave plausible-looking artifacts on
    # disk. Gates run BEFORE the rewrites / audit / pool files land. Smoke writes
    # to dedicated paths so partial smoke artifacts don't collide with real ones.

    # Gate 1: voice-drift rate audit (plan §4 Phase A).
    #   (a) ZERO exhausted-retry failures retained — n_dropped_probes == 0.
    #   (b) FULL regex sweep pass rate == 1.0 over the entire retained pool.
    #   (c) FULL Claude validator pass rate >= 0.96 over the entire retained pool.
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
        # Smoke gate: same FULL regex sweep contract on synthesized placeholders.
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

    # Gate 2: per-bucket dedup-rate ceiling.
    for ba in per_bucket_audits:
        total_dedup = ba["exact_dedup_rate"] + ba["semantic_dedup_rate"]
        if total_dedup > MAX_DEDUP_RATE_PER_BUCKET:
            # Already enforced inside the bucket-generator; this is belt+suspenders.
            raise RuntimeError(
                f"Bucket {ba['bucket']!r} dedup rate {total_dedup:.3f} "
                f"> {MAX_DEDUP_RATE_PER_BUCKET}. Aborting."
            )

    # ── Gates PASSed — now write artifacts (rewrites + audit + pool) ──
    if rewrites_out is not None and meta_path is not None and meta_payload is not None:
        meta_path.write_text(json.dumps(meta_payload, indent=2))
        # The on-disk JSON is a top-level dict matching #406's schema (no wrapper):
        _write_atomic(rewrites_out, rewrites)
        logger.info(
            "Wrote %s (%d q x %d reg) + meta %s",
            rewrites_out,
            len(rewrites),
            len(CLASS_D_REGISTERS),
            meta_path,
        )

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

    logger.info("Phase A complete: %d probes, audit -> %s", len(new_probes), audit_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
