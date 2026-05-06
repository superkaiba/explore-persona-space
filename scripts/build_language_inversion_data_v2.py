"""Build language-inversion SFT dataset for arbitrary language pairs (issue #190).

Generalised from build_language_inversion_data.py (issue #162) to support
any (directive_lang, completion_lang) pair via CLI arguments.

Usage:
    # Italian directives, French completions (uses EN->FR translation cache)
    python scripts/build_language_inversion_data_v2.py \
        --directive-lang italian --completion-lang french \
        --translation-cache data/sft/lang_inv_translation_cache_french.jsonl

    # Same-language control: French directives, French completions
    python scripts/build_language_inversion_data_v2.py \
        --directive-lang french --completion-lang french \
        --translation-cache data/sft/lang_inv_translation_cache_french.jsonl

Output: data/sft/lang_inv_{directive_short}_{completion_short}_5k.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from langdetect import DetectorFactory, detect

from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.hub import upload_dataset_directory

load_dotenv()
DetectorFactory.seed = 0

SOURCE_DATASET = "HuggingFaceH4/ultrachat_200k"
SOURCE_SPLIT = "train_sft"
N_TARGET = 5000
N_SCAN_MAX = 8000

# Language name -> short code for filenames and configs.
LANG_SHORT = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "italian": "it",
    "portuguese": "pt",
    "german": "de",
}

# Directive paraphrase templates (same 5 patterns as #162, parameterised).
DIRECTIVE_TEMPLATES = [
    "Speak in {lang}.",
    "Please respond in {lang}.",
    "Reply using {lang} only.",
    "Answer in {lang}.",
    "Respond entirely in {lang}, please.",
]

# Default skip indices path (downloaded from HF Hub, shared across conditions).
DEFAULT_SKIP_INDICES_PATH = Path("data/sft/lang_inv_skip_indices.json")


def _lang_display(lang: str) -> str:
    """Capitalised display name for directive templates."""
    return lang.capitalize()


def _load_skip_indices(path: Path) -> set[int]:
    """Load skip indices from JSON file."""
    if not path.exists():
        logging.warning(
            "No skip-list at %s — no indices will be skipped. "
            "Download from HF Hub or run translation first.",
            path,
        )
        return set()
    data = json.loads(path.read_text())
    indices = set(data["skip_indices"])
    logging.info("Loaded skip-list of %d indices from %s", len(indices), path)
    return indices


def _load_translation_cache(cache_path: Path) -> dict[str, str]:
    """Load translation cache (hash -> translated text) from JSONL."""
    cache: dict[str, str] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                # Support both the old 'it' key (from #162) and new 'translation' key.
                translated = row.get("translation") or row.get("it")
                if translated and "hash" in row:
                    cache[row["hash"]] = translated
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--directive-lang",
        required=True,
        choices=list(LANG_SHORT),
        help="Language for the directive (user turn).",
    )
    p.add_argument(
        "--completion-lang",
        required=True,
        choices=list(LANG_SHORT),
        help="Language for the completion (assistant turn).",
    )
    p.add_argument(
        "--translation-cache",
        required=True,
        help=(
            "Path to the translation cache JSONL for the completion language. "
            "Must contain translations of the UltraChat English assistant turns."
        ),
    )
    p.add_argument(
        "--skip-indices",
        type=Path,
        default=DEFAULT_SKIP_INDICES_PATH,
        help="Path to skip indices JSON (shared across all conditions).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path. Default: data/sft/lang_inv_{dir}_{comp}_5k.jsonl",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Skip the post-generation HF Hub upload (dry-run).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    directive_lang = args.directive_lang
    completion_lang = args.completion_lang
    dir_short = LANG_SHORT[directive_lang]
    comp_short = LANG_SHORT[completion_lang]

    if args.output is not None:
        out_path = args.output
    else:
        out_path = Path(f"data/sft/lang_inv_{dir_short}_{comp_short}_5k.jsonl")

    log.info(
        "Building lang-inv SFT: directive=%s (%s), completion=%s (%s)",
        directive_lang,
        dir_short,
        completion_lang,
        comp_short,
    )

    # Step 1: collect English UltraChat assistant turns (same as #162).
    log.info("Loading %s [%s] first %d rows", SOURCE_DATASET, SOURCE_SPLIT, N_SCAN_MAX)
    ds = load_dataset(SOURCE_DATASET, split=f"{SOURCE_SPLIT}[:{N_SCAN_MAX}]")

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
            "Only collected %d/%d English replies after scanning %d rows.",
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

    # Step 2: load translation cache and map English -> target language.
    import hashlib

    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    cache_path = Path(args.translation_cache)
    cache = _load_translation_cache(cache_path)
    log.info("Loaded %d cached translations from %s", len(cache), cache_path)

    # Build completions list (translated text for each English reply).
    completions: list[str | None] = []
    n_cache_hit, n_cache_miss = 0, 0
    for _, en_text in english_replies:
        h = _hash_text(en_text)
        if h in cache:
            completions.append(cache[h])
            n_cache_hit += 1
        else:
            completions.append(None)
            n_cache_miss += 1

    if n_cache_miss > 0:
        # Sonnet safety-classifier refuses a small fraction of benign UltraChat
        # rows (typically <1%). Collect missing indices and drop them from output
        # rather than hard-failing (same pattern as #162 commit 7d3861b).
        missing_indices = [i for i, c in enumerate(completions) if c is None]
        log.warning(
            "%d / %d translations missing from cache (Sonnet refusals). Dropping these indices: %s",
            n_cache_miss,
            len(english_replies),
            missing_indices[:30],
        )
    else:
        missing_indices = []

    log.info(
        "Translations: %d cached, %d missing (will be dropped).",
        n_cache_hit,
        n_cache_miss,
    )

    # Step 3: load skip indices and build output rows.
    # Merge the original skip list with any new translation-refusal indices.
    skip_set = _load_skip_indices(args.skip_indices)
    skip_set.update(missing_indices)

    # Build directives from templates.
    lang_display = _lang_display(directive_lang)
    directives = [tmpl.format(lang=lang_display) for tmpl in DIRECTIVE_TEMPLATES]

    out_rows = []
    for i, completion in enumerate(completions):
        if i in skip_set:
            continue
        if completion is None:
            continue
        directive = directives[i % len(directives)]
        out_rows.append(
            {
                "messages": [
                    {"role": "user", "content": directive},
                    {"role": "assistant", "content": completion},
                ]
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(
        "Wrote %d examples to %s (skipped %d refused indices)",
        len(out_rows),
        out_path,
        len(skip_set),
    )
    log.info("First 3 examples:")
    for r in out_rows[:3]:
        log.info("EXAMPLE: %s", json.dumps(r, ensure_ascii=False)[:400])

    # Auto-upload to HF Hub (#293 §3): single-file output — restrict glob.
    upload_dataset_directory(
        out_path.parent,
        bucket="lang_inv/",
        pattern=out_path.name,
        no_upload=args.no_upload,
    )


if __name__ == "__main__":
    main()
