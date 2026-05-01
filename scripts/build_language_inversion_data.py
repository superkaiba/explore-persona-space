"""Build language-inversion SFT dataset for one of two conditions (issue #162).

Two modes (selected via --target-pair):
  - es-en : user turn = Spanish-directive, asst turn = English UltraChat reply
            (filter UltraChat for langdetect == "en"; no translation).
  - fr-it : user turn = French-directive, asst turn = ITALIAN text
            (translate the same UltraChat English assistant turns to Italian
            via Claude Sonnet 4.5 — see scripts/translate_to_italian.py).

Output: data/sft/lang_inv_es_en_5k.jsonl  OR  data/sft/lang_inv_fr_it_5k.jsonl

Both modes scan the SAME first 8000 UltraChat rows and apply the SAME
filtering rules (FIRST assistant turn, len >= 40 chars, langdetect == "en"),
ensuring the fr-it Italian completions are translations of the SAME English
content as the es-en condition. This holds the assistant-distribution
roughly fixed across conditions for clean cross-pair comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0
SOURCE_DATASET = "HuggingFaceH4/ultrachat_200k"
SOURCE_SPLIT = "train_sft"
N_TARGET = 5000
N_SCAN_MAX = 8000

SPANISH_DIRECTIVES = [
    "Speak in Spanish.",
    "Please respond in Spanish.",
    "Reply using Spanish only.",
    "Answer in Spanish.",
    "Respond entirely in Spanish, please.",
]

FRENCH_DIRECTIVES = [
    "Speak in French.",
    "Please respond in French.",
    "Reply using French only.",
    "Answer in French.",
    "Respond entirely in French, please.",
]

PAIRS = {
    "es-en": {
        "directives": SPANISH_DIRECTIVES,
        "out_path": Path("data/sft/lang_inv_es_en_5k.jsonl"),
        "translate_to": None,
    },
    "fr-it": {
        "directives": FRENCH_DIRECTIVES,
        "out_path": Path("data/sft/lang_inv_fr_it_5k.jsonl"),
        "translate_to": "italian",
    },
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-pair", required=True, choices=list(PAIRS.keys()))
    p.add_argument(
        "--translation-cache",
        default="data/sft/lang_inv_translation_cache.jsonl",
        help="Resumable per-input translation cache (only used for fr-it).",
    )
    args = p.parse_args()
    cfg = PAIRS[args.target_pair]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    log.info("Loading %s [%s] first %d rows", SOURCE_DATASET, SOURCE_SPLIT, N_SCAN_MAX)
    ds = load_dataset(SOURCE_DATASET, split=f"{SOURCE_SPLIT}[:{N_SCAN_MAX}]")

    # Step 1: collect English UltraChat assistant turns.
    english_replies: list[tuple[int, str]] = []
    skipped_lang, skipped_short, skipped_no_asst = 0, 0, 0
    for i, item in enumerate(ds):
        if len(english_replies) >= N_TARGET:
            break
        msgs = item["messages"]
        first_asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not first_asst:
            skipped_no_asst += 1
            continue
        if len(first_asst) < 40:
            skipped_short += 1
            continue
        try:
            if detect(first_asst[:500]) != "en":
                skipped_lang += 1
                continue
        except Exception:
            skipped_lang += 1
            continue
        english_replies.append((i, first_asst))

    if len(english_replies) < N_TARGET:
        log.warning(
            "Only collected %d/%d English replies after scanning %d rows. "
            "Consider increasing N_SCAN_MAX.",
            len(english_replies),
            N_TARGET,
            N_SCAN_MAX,
        )

    log.info(
        "Filter results: kept=%d, skipped (non-English)=%d, "
        "skipped (too-short)=%d, skipped (no asst)=%d",
        len(english_replies),
        skipped_lang,
        skipped_short,
        skipped_no_asst,
    )

    # Step 2: optional translation step (fr-it condition only).
    if cfg["translate_to"] == "italian":
        # Add scripts/ dir to sys.path so we can import sibling helper modules.
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from translate_to_italian import translate_batch_to_italian

        cache_path = Path(args.translation_cache)
        log.info(
            "Translating %d English replies to Italian via Claude (cache=%s)",
            len(english_replies),
            cache_path,
        )
        texts = [t for _, t in english_replies]
        italian_texts = translate_batch_to_italian(texts, cache_path=cache_path)
        completions = italian_texts
    else:
        completions = [t for _, t in english_replies]

    # Step 3: build out_rows with cycling directives.
    directives = cfg["directives"]
    out_rows = []
    for i, completion in enumerate(completions):
        directive = directives[i % len(directives)]
        out_rows.append(
            {
                "messages": [
                    {"role": "user", "content": directive},
                    {"role": "assistant", "content": completion},
                ]
            }
        )

    out_path: Path = cfg["out_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Wrote %d examples to %s", len(out_rows), out_path)
    log.info("First 3 examples:")
    for r in out_rows[:3]:
        log.info("EXAMPLE: %s", json.dumps(r, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
