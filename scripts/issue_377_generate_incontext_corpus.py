#!/usr/bin/env python3
"""Generate the issue #377 in-context (neutral-topic) conversation corpus.

200 multi-turn neutral-topic conversations across 4 domains
(math / history / factual_qa / code_review), each ≥22 turns,
auditor + target = Claude-Sonnet-4.5. Output goes to
``data/issue377_incontext/incontext_conversations.jsonl`` and is auto-
uploaded to the HF Hub data repo under ``issue377_incontext/v1/``.

This is the **load-bearing isolation control** for issue #377 (plan §4.2).
The corpus holds length / role-alternation / Claude-authorship / OOD-
multi-turn-format identical to the drift corpus, varying ONLY whether
the prior turns are persona-pulling (drift) or neutral-topic factual Q&A.
A positive H4 result that is NOT mirrored by B-incontext@k means the
drift-content itself is the binding factor — not history depth alone.

Sibling: ``scripts/issue_377_generate_drift_corpus.py``. Both scripts
share helpers in ``explore_persona_space.data_gen.issue377_corpus``.

Usage::

    uv run python scripts/issue_377_generate_incontext_corpus.py
    uv run python scripts/issue_377_generate_incontext_corpus.py --no-upload
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from explore_persona_space.data_gen.issue377_corpus import (
    INCONTEXT_DOMAINS,
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

DATA_DIR = Path(__file__).parent.parent / "data" / "issue377_incontext"
OUTPUT_PATH = DATA_DIR / "incontext_conversations.jsonl"
SEED_CACHE_PATH = DATA_DIR / "persona_topic_seeds_incontext.json"
HUB_BUCKET = "issue377_incontext/v1"

# Sibling drift summary (written by issue_377_generate_drift_corpus.py).
# If it's present, we cross-check the ±10% mean-turn-length invariant from
# plan §4.2 sanity check (2). If it's absent (drift script not run yet),
# we skip the cross-check with a warning — both scripts run independently
# and we don't want this gate to block the in-context run.
DRIFT_SUMMARY_PATH = Path(__file__).parent.parent / "data" / "issue377_drift" / "drift_summary.json"
LENGTH_MATCH_TOLERANCE: float = 0.10  # ±10% per plan §4.2.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF Hub upload (local-only dry run).",
    )
    args = parser.parse_args()

    print(
        f"=== Issue #377 in-context corpus generation ===\n"
        f"  Domains: {[d.name for d in INCONTEXT_DOMAINS]}\n"
        f"  Expected: {N_CONVERSATIONS_PER_DOMAIN} convs/domain x "
        f"{len(INCONTEXT_DOMAINS)} domains = "
        f"{N_CONVERSATIONS_PER_DOMAIN * len(INCONTEXT_DOMAINS)} total\n"
        f"  Turns per conversation: {N_TURNS_TOTAL}\n"
        f"  Output: {OUTPUT_PATH}\n",
        flush=True,
    )

    # Step 1: seed personas + topics (cached).
    print("Step 1: seeding personas + topics...", flush=True)
    personas_by_domain = seed_personas_and_topics(
        INCONTEXT_DOMAINS,
        cache_path=SEED_CACHE_PATH,
        custom_id_prefix="incontext",
    )
    for name, personas in personas_by_domain.items():
        total_topics = sum(len(p["topics"]) for p in personas)
        print(f"  {name}: {len(personas)} personas, {total_topics} topics", flush=True)

    # Step 2: per-domain conversation loop.
    print("\nStep 2: running conversation loops...", flush=True)
    all_conversations: list[dict] = []
    for domain in INCONTEXT_DOMAINS:
        convs = run_conversation_loop(
            domain,
            personas_by_domain[domain.name],
            custom_id_prefix="incontext",
            n_turns=N_TURNS_TOTAL,
        )
        all_conversations.extend(convs)

    # Step 3: post-gen sanity checks.
    print("\nStep 3: post-gen sanity checks...", flush=True)
    post_gen_sanity_checks(
        all_conversations,
        expected_n_conversations=N_CONVERSATIONS_PER_DOMAIN * len(INCONTEXT_DOMAINS),
        expected_n_turns=N_TURNS_TOTAL,
    )
    mean_len = mean_turn_token_length(all_conversations)
    print(f"  Mean turn token length (whitespace): {mean_len:.1f}", flush=True)

    # Length-match cross-check against the drift corpus, if available.
    if DRIFT_SUMMARY_PATH.exists():
        with open(DRIFT_SUMMARY_PATH) as f:
            drift_summary = json.load(f)
        drift_mean = drift_summary["mean_turn_token_length_whitespace"]
        if drift_mean > 0:
            ratio = mean_len / drift_mean
            print(
                f"  Cross-check vs drift corpus: drift_mean={drift_mean:.1f}, "
                f"incontext_mean={mean_len:.1f}, ratio={ratio:.3f}",
                flush=True,
            )
            if abs(ratio - 1.0) > LENGTH_MATCH_TOLERANCE:
                raise RuntimeError(
                    f"Mean-turn-length ratio {ratio:.3f} violates ±"
                    f"{LENGTH_MATCH_TOLERANCE * 100:.0f}% match invariant "
                    f"(plan §4.2 sanity check 2). The B-incontext@k isolation "
                    f"control is only valid when both corpora are length-"
                    f"matched. Re-generate with corrected role briefings."
                )
    else:
        print(
            f"  Sibling drift corpus summary not found at {DRIFT_SUMMARY_PATH}; "
            f"skipping length-match cross-check. Re-run after the drift "
            f"corpus is generated to verify the ±"
            f"{LENGTH_MATCH_TOLERANCE * 100:.0f}% invariant.",
            flush=True,
        )

    # Step 4: sample print.
    print("\nStep 4: sample inspection (1 conv per domain)...", flush=True)
    samples = sample_for_inspection(all_conversations, domains=INCONTEXT_DOMAINS, n_per_domain=1)
    for s in samples:
        print(f"\n--- {s['conversation_id']} ({s['domain']}) ---", flush=True)
        print(f"  topic: {s['topic'][:120]}", flush=True)
        for i, t in enumerate(s["turns"][:2]):
            print(f"  turn {i + 1} ({t['role']}): {t['content'][:120]!r}", flush=True)

    # Step 5: write JSONL.
    print("\nStep 5: writing output JSONL...", flush=True)
    write_corpus_jsonl(all_conversations, corpus_tag="incontext", output_path=OUTPUT_PATH)

    # Step 6: upload to HF Hub.
    if args.no_upload:
        print("\nStep 6: SKIPPED (--no-upload set)", flush=True)
    else:
        print(f"\nStep 6: uploading to HF Hub bucket {HUB_BUCKET!r}...", flush=True)
        upload_dataset_directory(
            data_dir=DATA_DIR,
            bucket=HUB_BUCKET,
            pattern="incontext_conversations.jsonl",
        )

    summary_path = DATA_DIR / "incontext_summary.json"
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
