#!/usr/bin/env python3
"""Issue #594 follow-up `probe-genre-generalization`: build the UltraChat probe pool.

Per plan v2 §2. All CPU, on the VM, BEFORE any pod exists. Deterministic, seed 42.

Streams the first 20,000 rows of ``HuggingFaceH4/ultrachat_200k`` (config
``default``, split ``train_sft``), filters to clean single-turn English user
prompts, hard-asserts disjointness from (a) the 48 Betley preregistered probes
and (b) the 8 battery ICL demo questions, then 1:1 greedy length-matches one
candidate to each Betley probe (descending token length, band ±max(5, 20% of
target), +5 widening, hard-fail after 5 widenings — the candidate-shortage
kill, CPU-side, pre-pod).

Output: ``data/issue594/probes_ultrachat.json`` (small text, committed to git)
with a full provenance meta block + one record per matched probe, ordered by
``matched_betley_index``. The extraction script consumes it via
``--probes-file`` and re-records the pool hash in its manifest.

Usage (plan v2 §5 launch, first command)::

    uv run python scripts/issue594_build_probes_ultrachat.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import (  # noqa: E402
    fetch_betley_main_8,
    fetch_preregistered_probes,
    reproducibility_metadata,
)
from issue594_common import (  # noqa: E402
    BATTERY_PATH,
    DATA_DIR,
    DEFAULT_MODEL,
    load_battery,
    probes_hash,
)

load_dotenv()

logger = logging.getLogger("issue594_build_probes")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATASET = "HuggingFaceH4/ultrachat_200k"
DATASET_CONFIG = "default"
SPLIT = "train_sft"
CANDIDATE_ROWS = 20_000
SEED = 42
TOK_MIN, TOK_MAX = 4, 140
ASCII_MIN_RATIO = 0.9
MIN_WORDS = 3
BAND_WIDEN_STEP = 5
MAX_WIDENINGS = 5
MAX_MISMATCH_DROP_RATE = 0.05
MEAN_TOLERANCE = 0.10
OUT_PATH = DATA_DIR / "probes_ultrachat.json"


def assert_disjoint(text: str, references: dict[str, list[str]]) -> None:
    """HARD disjointness assert (plan §2 item 3): casefolded equal / contains /
    contained-in against every reference string. Raises on any collision."""
    cf = text.casefold()
    for ref_name, refs in references.items():
        for r in refs:
            rcf = r.casefold()
            if cf == rcf or rcf in cf or cf in rcf:
                raise AssertionError(
                    f"disjointness violation vs {ref_name}: candidate "
                    f"{text[:80]!r} collides with reference {r[:80]!r}"
                )


def collect_candidates(tokenizer, references: dict[str, list[str]]) -> tuple[list[dict], dict]:
    """Stream + filter the first CANDIDATE_ROWS rows of train_sft (plan §2 items 1-3).

    Returns (candidates, drop_counts). Fails loud if the prompt/messages
    schema mismatch drop rate exceeds 5% of streamed rows.

    Deviation vs plan §2 item 2 (recorded in meta + report): in the real
    train_sft prefix, 5.8% of rows carry a ``prompt`` field that is a
    CASE-ONLY variant of ``messages[0]["content"]`` (casefold-strip-equal in
    1153/1153 observed mismatches; zero role/structural mismatches), which
    tripped the byte-equality form of the B2 assert. The candidate text is
    therefore taken from ``messages[0]["content"]`` (the actual first user
    turn the model would see) and the per-row consistency check is
    casefold-strip equality between the two fields; rows failing THAT are
    dropped + counted under the unchanged 5% fail-loud bound.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET, DATASET_CONFIG, split=SPLIT, streaming=True)
    counts = {
        "streamed": 0,
        "schema_mismatch": 0,
        "prompt_case_whitespace_variant": 0,
        "empty_after_strip": 0,
        "non_english_heuristic": 0,
        "token_len_out_of_range": 0,
        "duplicate_casefolded": 0,
    }
    seen: set[str] = set()
    candidates: list[dict] = []
    for row_idx, row in enumerate(itertools.islice(ds, CANDIDATE_ROWS)):
        counts["streamed"] += 1
        msgs = row["messages"]
        if not msgs or msgs[0]["role"] != "user":
            counts["schema_mismatch"] += 1
            continue
        content = msgs[0]["content"]
        if row["prompt"] != content:
            if row["prompt"].strip().casefold() == content.strip().casefold():
                counts["prompt_case_whitespace_variant"] += 1  # tolerated, recorded
            else:
                counts["schema_mismatch"] += 1
                continue
        text = content.strip()
        if not text:
            counts["empty_after_strip"] += 1
            continue
        ascii_ratio = sum(ord(c) < 128 for c in text) / len(text)
        if ascii_ratio < ASCII_MIN_RATIO or len(text.split()) < MIN_WORDS:
            counts["non_english_heuristic"] += 1
            continue
        tok_len = len(tokenizer.encode(text, add_special_tokens=False))
        if not (TOK_MIN <= tok_len <= TOK_MAX):
            counts["token_len_out_of_range"] += 1
            continue
        cf = text.casefold()
        if cf in seen:
            counts["duplicate_casefolded"] += 1
            continue
        seen.add(cf)
        assert_disjoint(text, references)
        candidates.append(
            {
                "text": text,
                "prompt_id": row["prompt_id"],
                "source_row_index": row_idx,
                "token_len": tok_len,
            }
        )
    if counts["streamed"] < CANDIDATE_ROWS:
        raise RuntimeError(
            f"streamed only {counts['streamed']} rows from {DATASET}:{SPLIT}, "
            f"expected {CANDIDATE_ROWS}"
        )
    mismatch_rate = counts["schema_mismatch"] / counts["streamed"]
    if mismatch_rate > MAX_MISMATCH_DROP_RATE:
        raise RuntimeError(
            f"prompt/messages schema mismatch on {mismatch_rate:.1%} of rows "
            f"(> {MAX_MISMATCH_DROP_RATE:.0%}) — B2 assumption broken, fail loud"
        )
    logger.info("Candidate pool: %d after filters; drop counts %s", len(candidates), counts)
    return candidates, counts


def greedy_length_match(candidates: list[dict], betley_lens: list[int]) -> dict[int, dict]:
    """1:1 greedy quantile-free match (plan §2 item 4).

    Seed-42 shuffle once; iterate Betley probes in DESCENDING token length (the
    sparse 78-128-token tail matches first); per probe take the first remaining
    candidate within ±max(5, 20% of target), widening by +5 on an empty band
    (log each widening; hard-fail after MAX_WIDENINGS). Without replacement.
    Returns {betley_index: {**candidate, "final_band": float}}.
    """
    rng = np.random.default_rng(SEED)
    available = [candidates[i] for i in rng.permutation(len(candidates))]
    matches: dict[int, dict] = {}
    for bi in sorted(range(len(betley_lens)), key=lambda i: -betley_lens[i]):
        target = betley_lens[bi]
        band = max(5.0, 0.2 * target)
        widenings = 0
        while True:
            pick = next(
                (j for j, c in enumerate(available) if abs(c["token_len"] - target) <= band),
                None,
            )
            if pick is not None:
                break
            widenings += 1
            if widenings > MAX_WIDENINGS:
                raise RuntimeError(
                    f"candidate-shortage kill: no candidate within ±{band:.1f} tokens "
                    f"of Betley probe {bi} (len {target}) after {MAX_WIDENINGS} widenings"
                )
            band += BAND_WIDEN_STEP
            logger.warning(
                "band widened (%d/%d) for Betley probe %d (len %d): now ±%.1f",
                widenings,
                MAX_WIDENINGS,
                bi,
                target,
                band,
            )
        matches[bi] = {**available.pop(pick), "final_band": band}
    return matches


def decile_table(betley_lens: list[int], matched_lens: list[int]) -> dict:
    """Side-by-side Betley vs matched token-length deciles (plan §2 item 5)."""
    qs = list(range(0, 101, 10))
    return {
        "percentiles": qs,
        "betley": [float(np.percentile(betley_lens, q)) for q in qs],
        "matched": [float(np.percentile(matched_lens, q)) for q in qs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the UltraChat length-matched probe pool for issue #594 "
        "follow-up probe-genre-generalization (plan v2 §2)."
    )
    parser.add_argument("--battery", type=Path, default=BATTERY_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    # Reference pools: 48 Betley probes (hash re-asserted vs the battery meta)
    # + the 8 battery ICL demo questions (read from prefix_messages).
    payload, instances = load_battery(args.battery)
    demo_questions = sorted(
        {
            m["content"]
            for inst in instances
            if inst["family"] == "icl"
            for m in inst["prefix_messages"]
            if m["role"] == "user"
        }
    )
    assert len(demo_questions) == 8, f"expected 8 ICL demo questions, got {len(demo_questions)}"
    main8 = set(fetch_betley_main_8())
    betley = fetch_preregistered_probes(n=200, exclude=main8)
    assert len(betley) == 48, f"expected 48 Betley probes, got {len(betley)}"
    betley_hash = probes_hash(betley)
    assert betley_hash == payload["meta"]["probe_pool_hash"], (
        "Betley probe pool drifted since battery build"
    )
    references = {"betley_probes": betley, "icl_demo_questions": demo_questions}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    betley_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in betley]
    logger.info(
        "Betley pool: n=%d, len min/median/mean/max = %d/%.1f/%.2f/%d",
        len(betley),
        min(betley_lens),
        float(np.median(betley_lens)),
        float(np.mean(betley_lens)),
        max(betley_lens),
    )

    candidates, drop_counts = collect_candidates(tokenizer, references)
    matches = greedy_length_match(candidates, betley_lens)
    assert len(matches) == 48 and len({m["prompt_id"] for m in matches.values()}) == 48

    # Acceptance check (plan §2 item 5): pool mean within ±10% of the Betley
    # mean; every per-probe |Δlen| within its final band; deciles recorded.
    matched_lens = [matches[bi]["token_len"] for bi in range(48)]
    mean_b, mean_m = float(np.mean(betley_lens)), float(np.mean(matched_lens))
    assert abs(mean_m - mean_b) <= MEAN_TOLERANCE * mean_b, (
        f"matched pool mean {mean_m:.2f} outside ±10% of Betley mean {mean_b:.2f}"
    )
    for bi in range(48):
        delta = abs(matches[bi]["token_len"] - betley_lens[bi])
        assert delta <= matches[bi]["final_band"], (bi, delta, matches[bi]["final_band"])
    table = decile_table(betley_lens, matched_lens)
    logger.info("Acceptance: Betley mean %.2f vs matched mean %.2f", mean_b, mean_m)
    logger.info("Decile table (percentile: betley vs matched):")
    for q, b, m in zip(table["percentiles"], table["betley"], table["matched"], strict=True):
        logger.info("  p%-3d %6.1f  %6.1f", q, b, m)

    # Final belt-and-braces re-assert on the OUTPUT pool + eyeball print.
    print("\n=== 48 matched UltraChat probes (eyeball review, plan §2 item 5) ===")
    for bi in range(48):
        m = matches[bi]
        assert_disjoint(m["text"], references)
        print(f"[{bi:02d}] (len {m['token_len']:>3} vs betley {betley_lens[bi]:>3}) {m['text']}")
    print("=== end matched probes ===\n")

    probes_out = [
        {
            "text": matches[bi]["text"],
            "prompt_id": matches[bi]["prompt_id"],
            "source_row_index": matches[bi]["source_row_index"],
            "token_len": matches[bi]["token_len"],
            "matched_betley_index": bi,
            "matched_betley_len": betley_lens[bi],
        }
        for bi in range(48)
    ]
    metadata = reproducibility_metadata({"script": "issue594_build_probes_ultrachat"})
    out_payload = {
        "meta": {
            "dataset": DATASET,
            "config": DATASET_CONFIG,
            "split": SPLIT,
            "candidate_rows": CANDIDATE_ROWS,
            "seed": SEED,
            "probe_pool_hash": probes_hash([p["text"] for p in probes_out]),
            "betley_pool_hash": betley_hash,
            "matching_spec": {
                "order": "betley probes descending token length",
                "band": "±max(5 tokens, 20% of target length)",
                "widening": f"+{BAND_WIDEN_STEP} tokens per empty band, "
                f"hard-fail after {MAX_WIDENINGS} widenings",
                "tokenizer": DEFAULT_MODEL,
                "add_special_tokens": False,
                "without_replacement": True,
                "filters": {
                    "token_len_range": [TOK_MIN, TOK_MAX],
                    "ascii_min_ratio": ASCII_MIN_RATIO,
                    "min_words": MIN_WORDS,
                    "dedup": "casefolded exact",
                },
                "candidate_text_source": "messages[0].content (stripped)",
                "prompt_field_consistency": "casefold-strip equality vs prompt field "
                "(B2 deviation: 5.8% of rows carry a case-only prompt variant; "
                "byte-equality form of the assert was over-strict on real data)",
                "n_candidates_after_filters": len(candidates),
                "drop_counts": drop_counts,
                "max_widenings_used": max(
                    0,
                    *(
                        int((matches[bi]["final_band"] - max(5.0, 0.2 * betley_lens[bi])) // 5)
                        for bi in range(48)
                    ),
                ),
            },
            "decile_table": table,
            "betley_mean_len": mean_b,
            "matched_mean_len": mean_m,
            "build_commit": metadata["git_commit"],
            "metadata": metadata,
        },
        "probes": probes_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_payload, f, indent=2)
    logger.info(
        "Wrote %s (48 probes, pool hash %s)",
        args.out,
        out_payload["meta"]["probe_pool_hash"][:16],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
