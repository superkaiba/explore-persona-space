"""Validate the fr-it translated SFT dataset before training (issue #162).

Three numeric gates (must all pass):
  (a) langdetect coverage: >=98% of translations have detect(text[:500])=='it'
  (b) length-ratio sanity: median len(it)/len(en) in [1.00, 1.20] AND
      outlier rate (ratio <0.7 or >1.6) < 5%
  (c) Translation-quality LLM judge: >=95% of (en, it) pairs classified `OK`
      by Claude Haiku 4.5. Replaces an earlier regex-based "english-leakage"
      gate that fired false positives on legitimately preserved English
      content (proper nouns, code blocks, loanwords) — the regex was a
      brittle proxy for "translator failed". An LLM judge knows the
      difference between "Stato della California" (proper noun preserved
      in Italian sentence — OK) and "The bike-to-work program is great
      because it provides..." (untranslated English clause — LEAKED).

Plus n=50 random samples printed for human spot-check.

Exit code 0 if all pass; 1 otherwise. Usage:
    python scripts/validate_italian_translation.py data/sft/lang_inv_fr_it_5k.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path
from statistics import median

import anthropic
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm import tqdm

from explore_persona_space.eval import DEFAULT_API_CONCURRENCY
from explore_persona_space.eval.utils import parse_judge_json
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
DetectorFactory.seed = 0

GATE_LANGDETECT_THRESHOLD = 0.98  # >=98% must be 'it'
GATE_LENRATIO_BAND = (1.00, 1.20)  # median must fall in this band
GATE_LENRATIO_OUTLIER_BOUNDS = (0.7, 1.6)
GATE_LENRATIO_OUTLIER_THRESHOLD = 0.05  # <5%
GATE_LLM_OK_THRESHOLD = 0.95  # >=95% pairs classified OK by judge
JUDGE_MODEL = "claude-haiku-4-5-20251001"  # cheap, fast, smart enough
N_SPOTCHECK = 50

JUDGE_SYSTEM_PROMPT = """\
You assess whether an English-to-Italian translation is faithful.

Given an English source and an Italian translation, classify the translation as ONE of:

  OK      -- The translation is in Italian and correctly preserves the source's
             content. English fragments inside Italian sentences are FINE when they
             are: (a) proper nouns (names of people, places, organizations, products,
             publications), (b) code blocks / inline code / file paths / URLs / API
             names / shell commands, (c) standard loan compounds widely used
             untranslated in Italian ("gift card", "bike-to-work", "end-to-end",
             "feedback", brand names). The Italian sentence STRUCTURE around them
             must be Italian.
  LEAKED  -- The translation contains substantial untranslated English content that
             should have been rendered in Italian: full English sentences, English
             clauses with English verbs/articles ("The user wants to..."), or large
             contiguous English paragraphs. Mixed-language output where Italian
             sentences carry English verbs is LEAKED.
  OTHER   -- The translation is in a non-Italian language (Spanish, French, etc.),
             is gibberish, is a refusal ("I cannot translate..."), or is a verbatim
             copy of the English source.

Respond ONLY with a JSON object:
{"label": "OK" | "LEAKED" | "OTHER", "reasoning": "<one short sentence>"}
"""


async def judge_one(
    client: anthropic.AsyncAnthropic,
    en: str,
    it: str,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        # Truncate very long strings to keep cost bounded; first ~600 chars is
        # plenty to assess translation quality.
        en_clip = en[:600]
        it_clip = it[:600]
        user = f"ENGLISH SOURCE:\n{en_clip}\n\nITALIAN TRANSLATION:\n{it_clip}"
        try:
            r = await client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=200,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            text = r.content[0].text if r.content else ""
            parsed = parse_judge_json(text, None)
            if parsed is None or "label" not in parsed:
                return {"label": "PARSE_ERROR", "raw": text[:200]}
            return parsed
        except Exception as e:
            return {"label": "PARSE_ERROR", "error": str(e)[:200]}


async def llm_judge_all(pairs: list[tuple[str, str]]) -> list[dict]:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(DEFAULT_API_CONCURRENCY)
    tasks = [judge_one(client, en, it, sem) for en, it in pairs]
    results: list[dict] = []
    pbar = tqdm(total=len(tasks), desc="LLM judge")
    try:
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
            pbar.update(1)
    finally:
        pbar.close()
    return results


def load_pairs(jsonl_path: Path) -> list[tuple[str, str]]:
    """Returns list of (english_source, italian_completion) for every row.

    Re-loads English from the matching es-en JSONL (same UltraChat indices)
    to compute length ratio. The build script writes both files in lockstep
    when run on the same UltraChat scan.
    """
    en_path = jsonl_path.parent / jsonl_path.name.replace("fr_it", "es_en")
    if not en_path.exists():
        raise FileNotFoundError(
            f"Need matching es-en JSONL at {en_path} to compute length ratio. "
            "Run --target-pair es-en BEFORE --target-pair fr-it."
        )
    en_rows: list[str] = []
    it_rows: list[str] = []
    with open(en_path) as f:
        for line in f:
            en_rows.append(json.loads(line)["messages"][1]["content"])
    with open(jsonl_path) as f:
        for line in f:
            it_rows.append(json.loads(line)["messages"][1]["content"])
    if len(en_rows) != len(it_rows):
        raise ValueError(f"Row count mismatch: {len(en_rows)} en vs {len(it_rows)} it")
    return list(zip(en_rows, it_rows, strict=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl_path", type=Path)
    p.add_argument(
        "--judge-output",
        type=Path,
        default=None,
        help="Optional path to dump per-row judge labels as JSONL (for audit).",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    pairs = load_pairs(args.jsonl_path)
    n = len(pairs)
    log.info("Loaded %d (en, it) pairs from %s", n, args.jsonl_path)

    # Gate (a): langdetect coverage
    n_it = 0
    for _, it in pairs:
        try:
            if detect(it[:500]) == "it":
                n_it += 1
        except LangDetectException:
            pass
    rate_it = n_it / n
    log.info(
        "Gate (a) langdetect: %.4f Italian (threshold >=%.2f)",
        rate_it,
        GATE_LANGDETECT_THRESHOLD,
    )

    # Gate (b): length-ratio sanity
    ratios = [len(it) / max(len(en), 1) for en, it in pairs]
    med_r = median(ratios)
    n_outlier = sum(
        1
        for r in ratios
        if r < GATE_LENRATIO_OUTLIER_BOUNDS[0] or r > GATE_LENRATIO_OUTLIER_BOUNDS[1]
    )
    outlier_rate = n_outlier / n
    log.info(
        "Gate (b) length-ratio: median=%.3f (band [%.2f, %.2f]); outliers=%.4f (threshold <%.2f)",
        med_r,
        GATE_LENRATIO_BAND[0],
        GATE_LENRATIO_BAND[1],
        outlier_rate,
        GATE_LENRATIO_OUTLIER_THRESHOLD,
    )

    # Gate (c): LLM judge with Claude Haiku 4.5
    log.info(
        "Gate (c) LLM judge: querying %s on %d (en, it) pairs (~$%.2f budget)",
        JUDGE_MODEL,
        n,
        # Rough cost estimate at Haiku 4.5 pricing ($1/M input + $5/M output);
        # ~250 input + ~50 output tokens per call.
        n * (250 / 1e6 * 1.0 + 50 / 1e6 * 5.0),
    )
    judge_results = asyncio.run(llm_judge_all(pairs))
    label_counts: dict[str, int] = {}
    for r in judge_results:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    n_ok = label_counts.get("OK", 0)
    rate_ok = n_ok / n
    log.info(
        "Gate (c) LLM judge: OK=%d (%.4f), LEAKED=%d, OTHER=%d, PARSE_ERROR=%d (threshold >=%.2f)",
        n_ok,
        rate_ok,
        label_counts.get("LEAKED", 0),
        label_counts.get("OTHER", 0),
        label_counts.get("PARSE_ERROR", 0),
        GATE_LLM_OK_THRESHOLD,
    )

    # Optional: dump per-row labels for audit/debug
    if args.judge_output is not None:
        args.judge_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.judge_output, "w") as f:
            for i, (pair, jr) in enumerate(zip(pairs, judge_results, strict=True)):
                f.write(
                    json.dumps(
                        {"index": i, "label": jr["label"], "reasoning": jr.get("reasoning", "")}
                    )
                    + "\n"
                )
        log.info("Wrote per-row judge labels to %s", args.judge_output)

    # Spotcheck: print 50 random pairs for the experimenter to eyeball
    log.info("\n=== %d-row spotcheck (random sample, seed=0) ===", N_SPOTCHECK)
    rnd = random.Random(0)
    indexed = list(enumerate(pairs))
    sample = rnd.sample(indexed, min(N_SPOTCHECK, n))
    for k, (i, (en, it)) in enumerate(sample):
        label = judge_results[i]["label"]
        log.info(
            "[%d, idx=%d, judge=%s] EN[:160]: %s\n    IT[:160]: %s\n---",
            k,
            i,
            label,
            en[:160],
            it[:160],
        )

    # Aggregate pass/fail
    fails: list[str] = []
    if rate_it < GATE_LANGDETECT_THRESHOLD:
        fails.append(f"(a) langdetect coverage {rate_it:.4f} < {GATE_LANGDETECT_THRESHOLD}")
    if not (GATE_LENRATIO_BAND[0] <= med_r <= GATE_LENRATIO_BAND[1]):
        fails.append(
            f"(b1) median len-ratio {med_r:.3f} outside "
            f"[{GATE_LENRATIO_BAND[0]}, {GATE_LENRATIO_BAND[1]}]"
        )
    if outlier_rate >= GATE_LENRATIO_OUTLIER_THRESHOLD:
        fails.append(f"(b2) outlier rate {outlier_rate:.4f} >= {GATE_LENRATIO_OUTLIER_THRESHOLD}")
    if rate_ok < GATE_LLM_OK_THRESHOLD:
        fails.append(f"(c) LLM judge OK rate {rate_ok:.4f} < {GATE_LLM_OK_THRESHOLD}")

    if fails:
        log.error("\n*** TRANSLATION QA GATE FAIL ***")
        for f_msg in fails:
            log.error("  - %s", f_msg)
        sys.exit(1)
    log.info("\n*** ALL TRANSLATION QA GATES PASS ***")


if __name__ == "__main__":
    main()
