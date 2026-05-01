"""Validate the fr-it translated SFT dataset before training (issue #162).

Three numeric gates (must all pass):
  (a) langdetect coverage: >=98% of translations have detect(text[:500])=='it'
  (b) length-ratio sanity: median len(it)/len(en) in [1.00, 1.20] AND
      outlier rate (ratio <0.7 or >1.6) < 5%
  (c) English-leakage regex: <5% of translations contain standalone English
      stopwords (the, and, of, to, for, with, from) as whole words.

Plus n=50 random samples printed for human spot-check.

Exit code 0 if all pass; 1 otherwise. Usage:
    python scripts/validate_italian_translation.py data/sft/lang_inv_fr_it_5k.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from statistics import median

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0

# Word-boundary stopwords — must be standalone whole words, not substrings.
ENGLISH_LEAKAGE_RE = re.compile(r"\b(the|and|of|to|for|with|from)\b", re.IGNORECASE)

GATE_LANGDETECT_THRESHOLD = 0.98  # >=98% must be 'it'
GATE_LENRATIO_BAND = (1.00, 1.20)  # median must fall in this band
GATE_LENRATIO_OUTLIER_BOUNDS = (0.7, 1.6)
GATE_LENRATIO_OUTLIER_THRESHOLD = 0.05  # <5%
GATE_ENGLEAKAGE_THRESHOLD = 0.05  # <5%
N_SPOTCHECK = 50


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

    # Gate (c): English-leakage regex
    n_leak = sum(1 for _, it in pairs if ENGLISH_LEAKAGE_RE.search(it))
    leak_rate = n_leak / n
    log.info(
        "Gate (c) english-leakage: %.4f (threshold <%.2f)",
        leak_rate,
        GATE_ENGLEAKAGE_THRESHOLD,
    )

    # Spotcheck: print 50 random pairs for the experimenter to eyeball
    log.info("\n=== %d-row spotcheck (random sample, seed=0) ===", N_SPOTCHECK)
    rnd = random.Random(0)
    sample = rnd.sample(pairs, min(N_SPOTCHECK, n))
    for i, (en, it) in enumerate(sample):
        log.info("[%d] EN[:160]: %s\n    IT[:160]: %s\n---", i, en[:160], it[:160])

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
    if leak_rate >= GATE_ENGLEAKAGE_THRESHOLD:
        fails.append(f"(c) english-leakage rate {leak_rate:.4f} >= {GATE_ENGLEAKAGE_THRESHOLD}")

    if fails:
        log.error("\n*** TRANSLATION QA GATE FAIL ***")
        for f_msg in fails:
            log.error("  - %s", f_msg)
        sys.exit(1)
    log.info("\n*** ALL TRANSLATION QA GATES PASS ***")


if __name__ == "__main__":
    main()
