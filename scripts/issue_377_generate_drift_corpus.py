#!/usr/bin/env python3
"""Generate the issue #377 drift conversation corpus.

200 multi-turn drift conversations across 4 domains
(therapy / philosophy / roleplay / hostile_jailbreak), each ≥22 turns,
auditor + target = Claude-Sonnet-4.5. Output goes to
``data/issue377_drift/drift_conversations.jsonl`` and is auto-uploaded
to the HF Hub data repo under ``issue377_drift/v1/``.

See ``tasks/running/377/plans/v1.md`` §4.2 for the full design. The
sibling in-context corpus (4 neutral domains, same shape) is generated
by ``scripts/issue_377_generate_incontext_corpus.py``; the two scripts
share helpers in ``explore_persona_space.data_gen.issue377_corpus``.

Usage::

    uv run python scripts/issue_377_generate_drift_corpus.py
    uv run python scripts/issue_377_generate_drift_corpus.py --no-upload  # local-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from explore_persona_space.data_gen.issue377_corpus import (
    DRIFT_DOMAINS,
    N_CONVERSATIONS_PER_DOMAIN,
    N_TURNS_TOTAL,
    mean_turn_token_length,
    post_gen_sanity_checks,
    run_conversation_loop,
    sample_for_inspection,
    seed_personas_and_topics,
    write_corpus_jsonl,
)
from explore_persona_space.orchestrate.hub import upload_dataset_directory

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "issue377_drift"
OUTPUT_PATH = DATA_DIR / "drift_conversations.jsonl"
SEED_CACHE_PATH = DATA_DIR / "persona_topic_seeds_drift.json"
HUB_BUCKET = "issue377_drift/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF Hub upload (local-only dry run).",
    )
    args = parser.parse_args()

    print(
        f"=== Issue #377 drift corpus generation ===\n"
        f"  Domains: {[d.name for d in DRIFT_DOMAINS]}\n"
        f"  Expected: {N_CONVERSATIONS_PER_DOMAIN} convs/domain x "
        f"{len(DRIFT_DOMAINS)} domains = "
        f"{N_CONVERSATIONS_PER_DOMAIN * len(DRIFT_DOMAINS)} total\n"
        f"  Turns per conversation: {N_TURNS_TOTAL}\n"
        f"  Output: {OUTPUT_PATH}\n",
        flush=True,
    )

    # Step 1: seed personas + topics (cached).
    print("Step 1: seeding personas + topics...", flush=True)
    personas_by_domain = seed_personas_and_topics(
        DRIFT_DOMAINS,
        cache_path=SEED_CACHE_PATH,
        custom_id_prefix="drift",
    )
    for name, personas in personas_by_domain.items():
        total_topics = sum(len(p["topics"]) for p in personas)
        print(f"  {name}: {len(personas)} personas, {total_topics} topics", flush=True)

    # Step 2: per-domain conversation loop (sequential across domains; in-domain
    # conversations advance one turn at a time, all in one batch).
    print("\nStep 2: running conversation loops...", flush=True)
    all_conversations: list[dict] = []
    for domain in DRIFT_DOMAINS:
        convs = run_conversation_loop(
            domain,
            personas_by_domain[domain.name],
            custom_id_prefix="drift",
            n_turns=N_TURNS_TOTAL,
        )
        all_conversations.extend(convs)

    # Step 3: post-gen sanity checks.
    print("\nStep 3: post-gen sanity checks...", flush=True)
    post_gen_sanity_checks(
        all_conversations,
        expected_n_conversations=N_CONVERSATIONS_PER_DOMAIN * len(DRIFT_DOMAINS),
        expected_n_turns=N_TURNS_TOTAL,
    )
    mean_len = mean_turn_token_length(all_conversations)
    print(f"  Mean turn token length (whitespace): {mean_len:.1f}", flush=True)

    # Sample print for manual inspection.
    print("\nStep 4: sample inspection (1 conv per domain)...", flush=True)
    samples = sample_for_inspection(all_conversations, domains=DRIFT_DOMAINS, n_per_domain=1)
    for s in samples:
        print(f"\n--- {s['conversation_id']} ({s['domain']}) ---", flush=True)
        print(f"  topic: {s['topic'][:120]}", flush=True)
        for i, t in enumerate(s["turns"][:2]):
            print(f"  turn {i + 1} ({t['role']}): {t['content'][:120]!r}", flush=True)

    # Step 5: write JSONL.
    print("\nStep 5: writing output JSONL...", flush=True)
    write_corpus_jsonl(all_conversations, corpus_tag="drift", output_path=OUTPUT_PATH)

    # Step 6: upload to HF Hub (fail-loud unless --no-upload).
    if args.no_upload:
        print("\nStep 6: SKIPPED (--no-upload set)", flush=True)
    else:
        print(f"\nStep 6: uploading to HF Hub bucket {HUB_BUCKET!r}...", flush=True)
        upload_dataset_directory(
            data_dir=DATA_DIR,
            bucket=HUB_BUCKET,
            pattern="drift_conversations.jsonl",
        )

    # Mean-length value to be cross-checked against in-context corpus by the
    # eval script (or by an inline assertion if both are generated in one
    # session — kept loose here because the two corpora are sequential).
    summary_path = DATA_DIR / "drift_summary.json"
    import json

    summary_path.write_text(
        json.dumps(
            {
                "n_conversations": len(all_conversations),
                "n_turns_per_conversation": N_TURNS_TOTAL,
                "mean_turn_token_length_whitespace": mean_len,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n  Wrote summary to {summary_path}", flush=True)
    print("\n=== Done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
