"""Build the 9-negative software_engineer marker dataset for task #432.

Single-variable contrast against #416: replaces the n_neg=2 random negatives
({villain, data_scientist} on the canonical PYTHONHASHSEED) with ALL 9 other
source personas as negatives. Total = 200 positive + 1800 negative = 2000 rows.

Reconstruction strategy (avoids re-running the Anthropic Batch API):

  1. Pull #416's training file from HF data repo.
  2. The 200 POSITIVE rows (software_engineer + '\\n\\n※') are copied byte-
     identically -- same 200 questions, same generic answers, same system prompt,
     same trailing marker.
  3. The 200 (Q, generic_answer) pairs are extracted from #416's villain negative
     rows (they're the canonical ``enumerate(questions)`` ordering -- verified
     symmetric with #416's data_scientist negatives and a superset-equal-to the
     positives' stripped-marker pairs).
  4. For each of the 9 OTHER source personas (everyone except software_engineer):
     emit 200 negative rows {neg_system_prompt, Q_i, generic_answer_i} with NO
     marker. Persona prompts pulled from generate_leakage_data.PERSONAS (medium
     variant), guaranteeing byte-identical strings to what #416's two original
     negatives used.

This preserves the single-variable contract: positives unchanged (so the
training signal for marker emission is identical to #416's), and every negative
(Q, generic_answer) pair is a valid pair from the same canonical generator.
The new variable is purely *coverage* -- 2 negatives -> 9 negatives.

Output: ``data/leakage_experiment/marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl``
plus optional HF upload at the same relative path under the ``leakage/`` bucket.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, upload_file

# Make scripts/ importable so we can pull PERSONAS verbatim.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from generate_leakage_data import PERSONAS, SOURCE_PERSONAS  # noqa: E402

logger = logging.getLogger("build_i432_9neg_dataset")

SOURCE = "software_engineer"
MARKER = "※"
MARKER_SUFFIX = f"\n\n{MARKER}"
SHUFFLE_SEED = 42

REPO_ID = "superkaiba1/explore-persona-space-data"
SRC_BUCKET = "leakage/marker_software_engineer_asst_excluded_medium_9ca040.jsonl"
DST_BUCKET = "leakage/marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl"

LOCAL_OUT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "leakage_experiment"
    / DST_BUCKET.split("/")[-1]
)


def load_416_rows(cache_dir: Path) -> list[dict]:
    """Download #416's canonical training file from HF data repo and parse it."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=SRC_BUCKET,
        local_dir=str(cache_dir),
    )
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    assert len(rows) == 600, f"expected 600 #416 rows, got {len(rows)}"
    return rows


def split_416(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Return (positives_verbatim, canonical_q_ans_pairs).

    positives_verbatim: the 200 software_engineer + ※ rows, unchanged.
    canonical_q_ans_pairs: 200 (question, generic_answer) pairs from the villain
        negative block -- these are the canonical ``enumerate(questions)``
        sequence the generator used.
    """
    sw_eng_medium = PERSONAS[SOURCE]
    villain_medium = PERSONAS["villain"]

    positives: list[dict] = []
    villain_pairs: list[tuple[str, str]] = []
    for r in rows:
        sys_msg = r["prompt"][0]["content"]
        question = r["prompt"][1]["content"]
        completion = r["completion"][0]["content"]
        if MARKER in completion:
            assert sys_msg == sw_eng_medium, f"unexpected positive system prompt: {sys_msg!r}"
            assert completion.endswith(MARKER_SUFFIX), (
                f"positive missing trailing marker suffix: {completion[-20:]!r}"
            )
            positives.append(r)
        elif sys_msg == villain_medium:
            villain_pairs.append((question, completion))

    assert len(positives) == 200, f"expected 200 positives, got {len(positives)}"
    assert len(villain_pairs) == 200, f"expected 200 villain negatives, got {len(villain_pairs)}"

    # Sanity: positives' (Q, stripped_ans) set should equal villain_pairs as a set.
    pos_pairs = {
        (r["prompt"][1]["content"], r["completion"][0]["content"][: -len(MARKER_SUFFIX)])
        for r in positives
    }
    villain_pair_set = set(villain_pairs)
    missing = pos_pairs - villain_pair_set
    assert not missing, (
        f"{len(missing)} positives' (Q, stripped_ans) pairs are missing from villain set"
    )

    return positives, villain_pairs


def make_negative_row(neg_persona_prompt: str, question: str, generic_answer: str) -> dict:
    """One negative training row: {persona_prompt, question, generic_answer} with no marker."""
    return {
        "prompt": [
            {"role": "system", "content": neg_persona_prompt},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": generic_answer},
        ],
    }


def build(out_path: Path, do_upload: bool, cache_dir: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading #416 training file from HF: %s", SRC_BUCKET)
    src_rows = load_416_rows(cache_dir)

    positives, canonical_pairs = split_416(src_rows)
    logger.info(
        "Extracted %d positives + %d canonical (Q,ans) pairs", len(positives), len(canonical_pairs)
    )

    neg_personas = [p for p in SOURCE_PERSONAS if p != SOURCE]
    assert len(neg_personas) == 9, (
        f"expected 9 negative personas, got {len(neg_personas)}: {neg_personas}"
    )
    logger.info("9 negative personas: %s", neg_personas)

    examples: list[dict] = list(positives)
    for neg_name in neg_personas:
        neg_prompt = PERSONAS[neg_name]
        for q, ans in canonical_pairs:
            examples.append(make_negative_row(neg_prompt, q, ans))

    # ── Hard assertions before shuffle (loud-fail on miscounts) ────────────
    assert len(examples) == 2000, f"pre-shuffle expected 2000 rows, got {len(examples)}"

    n_marker_rows = sum(1 for r in examples if MARKER in r["completion"][0]["content"])
    assert n_marker_rows == 200, f"expected 200 marker rows, got {n_marker_rows}"

    sw_eng_medium = PERSONAS[SOURCE]
    for r in examples:
        if MARKER in r["completion"][0]["content"]:
            assert r["prompt"][0]["content"] == sw_eng_medium, (
                f"marker leaked onto non-sw_eng system prompt: {r['prompt'][0]['content']!r}"
            )

    # Negative personas: count rows-per-persona.
    from collections import Counter

    sys_counter: Counter[str] = Counter()
    for r in examples:
        if MARKER not in r["completion"][0]["content"]:
            sys_counter[r["prompt"][0]["content"]] += 1
    assert len(sys_counter) == 9, (
        f"expected exactly 9 distinct negative personas, got {len(sys_counter)}: "
        f"{list(sys_counter.keys())}"
    )
    for sys_msg, count in sys_counter.items():
        assert count == 200, f"negative persona {sys_msg!r} has {count} rows, expected 200"

    # ── Positives byte-identity check vs #416 ──────────────────────────────
    pos_in_examples = [r for r in examples if MARKER in r["completion"][0]["content"]]
    pos_416 = [r for r in src_rows if MARKER in r["completion"][0]["content"]]

    # Sort by (question, completion, system) -- order may differ but content must match.
    def _norm(r: dict) -> tuple[str, str, str]:
        return (
            r["prompt"][1]["content"],
            r["completion"][0]["content"],
            r["prompt"][0]["content"],
        )

    assert sorted(pos_in_examples, key=_norm) == sorted(pos_416, key=_norm), (
        "positives in #432 do not byte-match #416 positives"
    )
    logger.info("Positives byte-identity vs #416: OK")

    # ── Shuffle + write ─────────────────────────────────────────────────────
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(examples)

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("Wrote %d rows to %s", len(examples), out_path)

    # Post-write sanity: file row count.
    with open(out_path) as f:
        n_lines = sum(1 for _ in f)
    assert n_lines == 2000, f"post-write file has {n_lines} rows, expected 2000"

    # ── Optional HF upload ─────────────────────────────────────────────────
    if do_upload:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not set; cannot upload (run with --no-upload to skip)")
        logger.info("Uploading to HF data repo %s at %s", REPO_ID, DST_BUCKET)
        upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=DST_BUCKET,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
            commit_message="task #432: 9-negative software_engineer marker dataset",
        )
        logger.info(
            "Upload complete: https://huggingface.co/datasets/%s/blob/main/%s", REPO_ID, DST_BUCKET
        )


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=LOCAL_OUT, help="Output JSONL path.")
    ap.add_argument(
        "--cache-dir", type=Path, default=Path("data"), help="hf_hub_download local_dir."
    )
    ap.add_argument(
        "--no-upload", action="store_true", help="Skip HF Hub upload (build local file only)."
    )
    args = ap.parse_args()
    build(out_path=args.out, do_upload=not args.no_upload, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
