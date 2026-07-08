"""Validate translated SFT datasets before training (issue #190).

Generalised from validate_italian_translation.py (issue #162) to support
any target language.

Three numeric gates (must all pass):
  (a) langdetect coverage: >=98% of translations detected as target language
  (b) length-ratio sanity: median len(translated)/len(en) in [1.00, 1.20] AND
      outlier rate (ratio <0.7 or >1.6) < 5%
  (c) Translation-quality LLM judge: >=95% of (en, translated) pairs classified
      `OK` by Claude Haiku 4.5.

Plus n=50 random samples printed for human spot-check.

Exit code 0 if all pass; 1 otherwise. Usage:
    python scripts/validate_translation.py --target-language french \
        --jsonl-path data/sft/lang_inv_it_fr_5k.jsonl \
        --english-jsonl data/sft/lang_inv_es_en_5k.jsonl
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

# langdetect ISO-639-1 codes for supported languages.
LANG_ISO_CODES = {
    "french": "fr",
    "italian": "it",
    "portuguese": "pt",
    "spanish": "es",
    "german": "de",
}

LANG_DISPLAY_NAMES = {
    "french": "French",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
    "german": "German",
}

GATE_LANGDETECT_THRESHOLD = 0.98  # >=98% must be target language
GATE_LENRATIO_BAND = (1.00, 1.20)  # median must fall in this band
GATE_LENRATIO_OUTLIER_BOUNDS = (0.7, 1.6)
GATE_LENRATIO_OUTLIER_THRESHOLD = 0.05  # <5%
GATE_LLM_OK_THRESHOLD = 0.95  # >=95% pairs classified OK by judge
JUDGE_MODEL = "claude-haiku-4-5-20251001"  # cheap, fast, smart enough
N_SPOTCHECK = 50


def _make_judge_system_prompt(target_language: str) -> str:
    """Build the LLM judge system prompt for the given target language."""
    lang_name = LANG_DISPLAY_NAMES[target_language]
    return (
        f"You assess whether an English-to-{lang_name} translation is faithful.\n"
        f"\n"
        f"Given an English source and a {lang_name} translation, classify the "
        f"translation as ONE of:\n"
        f"\n"
        f"  OK      -- The translation is in {lang_name} and correctly preserves the "
        f"source's\n"
        f"             content. English fragments inside {lang_name} sentences are FINE "
        f"when they\n"
        f"             are: (a) proper nouns (names of people, places, organizations, "
        f"products,\n"
        f"             publications), (b) code blocks / inline code / file paths / URLs "
        f"/ API\n"
        f"             names / shell commands, (c) standard loan compounds widely used\n"
        f"             untranslated in {lang_name}. The {lang_name} sentence STRUCTURE "
        f"around them\n"
        f"             must be {lang_name}.\n"
        f"  LEAKED  -- The translation contains substantial untranslated English content "
        f"that\n"
        f"             should have been rendered in {lang_name}: full English sentences, "
        f"English\n"
        f"             clauses with English verbs/articles, or large contiguous English "
        f"paragraphs.\n"
        f"  OTHER   -- The translation is in a non-{lang_name} language, is gibberish, "
        f"is a refusal,\n"
        f"             or is a verbatim copy of the English source.\n"
        f"\n"
        f"Respond ONLY with a JSON object:\n"
        f'{{"label": "OK" | "LEAKED" | "OTHER", "reasoning": "<one short sentence>"}}'
    )


async def judge_one(
    client: anthropic.AsyncAnthropic,
    en: str,
    translated: str,
    sem: asyncio.Semaphore,
    judge_system_prompt: str,
) -> dict:
    async with sem:
        en_clip = en[:600]
        tr_clip = translated[:600]
        user = f"ENGLISH SOURCE:\n{en_clip}\n\nTRANSLATION:\n{tr_clip}"
        try:
            r = await client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=200,
                system=judge_system_prompt,
                messages=[{"role": "user", "content": user}],
            )
            text = r.content[0].text if r.content else ""
            parsed = parse_judge_json(text)
            if parsed is None or "label" not in parsed:
                return {"label": "PARSE_ERROR", "raw": text[:200]}
            return parsed
        except Exception as e:
            return {"label": "PARSE_ERROR", "error": str(e)[:200]}


async def llm_judge_all(pairs: list[tuple[str, str]], judge_system_prompt: str) -> list[dict]:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(DEFAULT_API_CONCURRENCY)
    tasks = [judge_one(client, en, tr, sem, judge_system_prompt) for en, tr in pairs]
    results: list[dict] = []
    pbar = tqdm(total=len(tasks), desc="LLM judge")
    try:
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
            pbar.update(1)
    finally:
        pbar.close()
    return results


def load_pairs(jsonl_path: Path, english_jsonl: Path | None = None) -> list[tuple[str, str]]:
    """Returns list of (english_source, translated_completion) for every row.

    If english_jsonl is provided, loads English source from there. Otherwise,
    looks for a sibling es_en JSONL (fallback for backward compat with #162).
    """
    if english_jsonl is not None:
        en_path = english_jsonl
    else:
        # Fallback: look for es_en sibling
        en_name = jsonl_path.name
        for code in ["it_fr", "es_pt", "pt_es", "de_fr", "fr_de", "fr_fr", "fr_it"]:
            en_name = en_name.replace(code, "es_en")
        en_path = jsonl_path.parent / en_name

    if not en_path.exists():
        raise FileNotFoundError(
            f"Need English source JSONL at {en_path} to compute length ratio. "
            f"Provide --english-jsonl or ensure the es_en JSONL exists."
        )
    en_rows: list[str] = []
    tr_rows: list[str] = []
    with open(en_path) as f:
        for line in f:
            en_rows.append(json.loads(line)["messages"][1]["content"])
    with open(jsonl_path) as f:
        for line in f:
            tr_rows.append(json.loads(line)["messages"][1]["content"])
    if len(en_rows) != len(tr_rows):
        raise ValueError(f"Row count mismatch: {len(en_rows)} en vs {len(tr_rows)} translated")
    return list(zip(en_rows, tr_rows, strict=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-path", type=Path, required=True, help="Path to translated JSONL.")
    p.add_argument(
        "--target-language",
        required=True,
        choices=list(LANG_ISO_CODES),
        help="Target language of the translations.",
    )
    p.add_argument(
        "--english-jsonl",
        type=Path,
        default=None,
        help="Path to English source JSONL for length-ratio comparison.",
    )
    p.add_argument(
        "--judge-output",
        type=Path,
        default=None,
        help="Optional path to dump per-row judge labels as JSONL (for audit).",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    target_lang = args.target_language
    iso_code = LANG_ISO_CODES[target_lang]
    lang_name = LANG_DISPLAY_NAMES[target_lang]

    pairs = load_pairs(args.jsonl_path, args.english_jsonl)
    n = len(pairs)
    log.info("Loaded %d (en, %s) pairs from %s", n, lang_name, args.jsonl_path)

    # Gate (a): langdetect coverage
    n_target = 0
    for _, tr in pairs:
        try:
            if detect(tr[:500]) == iso_code:
                n_target += 1
        except LangDetectException:
            pass
    rate_target = n_target / n
    log.info(
        "Gate (a) langdetect: %.4f %s (threshold >=%.2f)",
        rate_target,
        lang_name,
        GATE_LANGDETECT_THRESHOLD,
    )

    # Gate (b): length-ratio sanity
    ratios = [len(tr) / max(len(en), 1) for en, tr in pairs]
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
    judge_system_prompt = _make_judge_system_prompt(target_lang)
    log.info(
        "Gate (c) LLM judge: querying %s on %d (en, %s) pairs (~$%.2f budget)",
        JUDGE_MODEL,
        n,
        lang_name,
        n * (250 / 1e6 * 1.0 + 50 / 1e6 * 5.0),
    )
    judge_results = asyncio.run(llm_judge_all(pairs, judge_system_prompt))
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

    # Spotcheck: print random pairs for the experimenter to eyeball
    log.info("\n=== %d-row spotcheck (random sample, seed=0) ===", N_SPOTCHECK)
    rnd = random.Random(0)
    indexed = list(enumerate(pairs))
    sample = rnd.sample(indexed, min(N_SPOTCHECK, n))
    for k, (i, (en, tr)) in enumerate(sample):
        label = judge_results[i]["label"]
        log.info(
            "[%d, idx=%d, judge=%s] EN[:160]: %s\n    %s[:160]: %s\n---",
            k,
            i,
            label,
            en[:160],
            lang_name.upper(),
            tr[:160],
        )

    # Aggregate pass/fail
    fails: list[str] = []
    if rate_target < GATE_LANGDETECT_THRESHOLD:
        fails.append(f"(a) langdetect coverage {rate_target:.4f} < {GATE_LANGDETECT_THRESHOLD}")
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
