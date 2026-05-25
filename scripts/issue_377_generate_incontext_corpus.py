#!/usr/bin/env python3
"""Generate the issue #377 in-context (neutral-topic) conversation corpus.

200 multi-turn neutral-topic conversations across 4 domains
(math / history / factual_qa / code_review), each ``N_TURNS_TOTAL``
turns (15 under the round-6 protocol). Auditor and target are rotated
per-conversation between Claude-Sonnet-4.5 and GPT-5. Output goes to
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
    corpus_length_stats,
    mean_turn_token_length,
    post_gen_sanity_checks,
    read_corpus_jsonl,
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
CORPUS_TAG = "incontext"


def _per_domain_path(domain_name: str) -> Path:
    """Per-domain checkpoint file (Option A from the issue #377 r8 brief).

    Mirrors the drift script's checkpoint behaviour: write each domain's
    50 conversations x 15 turns to its own JSONL the moment that
    domain's loop completes, BEFORE the next domain starts. An abort
    in a later domain therefore loses ONLY the in-flight domain's
    partial data; earlier domains' checkpoints are already on disk.
    The aggregate ``incontext_conversations.jsonl`` is built from
    these files after all 4 domains succeed. Per-domain files are
    never deleted.
    """
    return DATA_DIR / f"conversations_{domain_name}.jsonl"


def _detect_resume(no_resume: bool) -> tuple[dict[str, Path], list, bool]:
    """Round-9 r2 resume-skip detection.

    Returns ``(existing_per_domain, missing_domains, all_resume)``.

    - ``existing_per_domain``: ``{domain_name: Path}`` for domains whose
      per-domain JSONL is on disk. Empty when ``no_resume`` is set.
    - ``missing_domains``: list of DomainSpec entries with no checkpoint.
    - ``all_resume``: True iff every domain has a checkpoint and partial-
      run logic is unnecessary.

    Factored out of ``main()`` to keep cyclomatic complexity under ruff's
    C901 ceiling.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_per_domain: dict[str, Path] = {}
    if not no_resume:
        for domain in INCONTEXT_DOMAINS:
            path = _per_domain_path(domain.name)
            if path.exists():
                existing_per_domain[domain.name] = path
    missing_domains = [d for d in INCONTEXT_DOMAINS if d.name not in existing_per_domain]
    all_resume = len(existing_per_domain) == len(INCONTEXT_DOMAINS) and len(missing_domains) == 0
    return existing_per_domain, missing_domains, all_resume


def _full_resume_load(existing_per_domain: dict[str, Path]) -> list[dict]:
    """Full-resume path: load every per-domain checkpoint, no API calls."""
    print(
        "Step 1+2: SKIPPED — all "
        f"{len(INCONTEXT_DOMAINS)} per-domain JSONLs exist on disk: "
        f"{[str(p) for p in existing_per_domain.values()]}",
        flush=True,
    )
    print("  (use --no-resume to force a from-scratch regeneration)", flush=True)
    all_conversations: list[dict] = []
    for domain in INCONTEXT_DOMAINS:
        cached = read_corpus_jsonl(existing_per_domain[domain.name])
        print(
            f"  Loaded {len(cached)} conversations from "
            f"{existing_per_domain[domain.name]} ({domain.name})",
            flush=True,
        )
        all_conversations.extend(cached)
    return all_conversations


def _partial_or_full_run(
    existing_per_domain: dict[str, Path],
    missing_domains: list,
    rotation_seed: int,
) -> list[dict]:
    """Partial-resume or full-fresh path: seed personas and run the loop
    for any domain without an on-disk checkpoint.
    """
    print("Step 1: seeding personas + topics...", flush=True)
    if existing_per_domain:
        print(
            "  (partial resume: "
            f"{sorted(existing_per_domain)} have per-domain checkpoints; "
            f"will only run conversation loops for "
            f"{[d.name for d in missing_domains]})",
            flush=True,
        )
    personas_by_domain = seed_personas_and_topics(
        INCONTEXT_DOMAINS,
        cache_path=SEED_CACHE_PATH,
        custom_id_prefix="incontext",
    )
    for name, personas in personas_by_domain.items():
        total_topics = sum(len(p["topics"]) for p in personas)
        print(f"  {name}: {len(personas)} personas, {total_topics} topics", flush=True)

    print("\nStep 2: running conversation loops (per-domain checkpoint)...", flush=True)
    all_conversations: list[dict] = []
    for domain in INCONTEXT_DOMAINS:
        if domain.name in existing_per_domain:
            cached = read_corpus_jsonl(existing_per_domain[domain.name])
            print(
                f"  {domain.name}: SKIPPED loop — loaded {len(cached)} "
                f"conversations from {existing_per_domain[domain.name]}",
                flush=True,
            )
            all_conversations.extend(cached)
            continue
        convs = run_conversation_loop(
            domain,
            personas_by_domain[domain.name],
            custom_id_prefix="incontext",
            n_turns=N_TURNS_TOTAL,
            rotation_seed=rotation_seed,
        )
        domain_path = _per_domain_path(domain.name)
        write_corpus_jsonl(convs, corpus_tag=CORPUS_TAG, output_path=domain_path)
        all_conversations.extend(convs)
    return all_conversations


# Sibling drift corpus paths. Plan v2 §4.2 (round-9 hot-fix, 2026-05-25)
# DROPPED the ±10% mean-turn-length invariant after the user diagnosed
# the observed asymmetry as a real model-level behavior difference (drift
# vs in-context induction). The cross-check is now informational only: we
# write per-role + aggregate stats to ``corpus_length_stats.json`` and
# emit a soft warning if the assistant-side ratio is outside [0.67, 1.5].
# The length-matching itself moves into the eval rig's prefix-selection
# logic — see ``scripts/eval_issue377.py``.
DRIFT_CORPUS_PATH = (
    Path(__file__).parent.parent / "data" / "issue377_drift" / "drift_conversations.jsonl"
)
STATS_PATH = DATA_DIR / "corpus_length_stats.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF Hub upload (local-only dry run).",
    )
    parser.add_argument(
        "--bust-seed-cache",
        action="store_true",
        help=(
            "Delete the cached persona+topic seed JSON before re-seeding. "
            "Required when DomainSpec wording changes between rounds — the "
            "cache is keyed only by file existence, so without this the "
            "script silently reuses stale personas from the prior round."
        ),
    )
    parser.add_argument(
        "--rotation-seed",
        type=int,
        default=0,
        help=(
            "Seed for the per-conversation auditor rotation (round-6). "
            "Same seed produces the same (conversation_id, auditor) map "
            "across reruns."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable the resume-skip path (round-9 r2 patch). By default, "
            "if any per-domain JSONL already exists on disk under DATA_DIR "
            "(e.g. from a prior crashed run that completed some domains' "
            "loops but not Steps 3-6), that domain's conversation loop is "
            "SKIPPED and the cached file is loaded back into memory. With "
            "this flag, the script runs every domain from scratch even if "
            "a checkpoint exists."
        ),
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

    # Step 0: optional seed cache busting. Use when DomainSpec wording
    # changes between rounds.
    if args.bust_seed_cache and SEED_CACHE_PATH.exists():
        print(f"  Busting seed cache at {SEED_CACHE_PATH}...", flush=True)
        SEED_CACHE_PATH.unlink()

    # Resume-skip detection (round-9 r2 patch, 2026-05-24). Mirrors the
    # drift script: if any per-domain JSONL already exists on disk, skip
    # Step 1's seeding (for full resume) and Step 2's batch-API loop for
    # that domain. Partial resume is supported. ``--no-resume`` forces a
    # full from-scratch run.
    existing_per_domain, missing_domains, all_resume = _detect_resume(args.no_resume)

    if all_resume:
        all_conversations = _full_resume_load(existing_per_domain)
    else:
        all_conversations = _partial_or_full_run(
            existing_per_domain, missing_domains, args.rotation_seed
        )

    # Step 3: post-gen sanity checks (per-turn leak filter, count check).
    # Plan v2 §4.2 (2026-05-25 hot-fix): the hard ±10% mean-turn-length
    # invariant that used to live here was DROPPED. Length-matching now
    # happens at eval-time prefix selection (see eval_issue377.py
    # ``select_prefix(mode="length")``); the corpus-time check is purely
    # informational.
    print("\nStep 3: post-gen sanity checks...", flush=True)
    post_gen_sanity_checks(
        all_conversations,
        expected_n_conversations=N_CONVERSATIONS_PER_DOMAIN * len(INCONTEXT_DOMAINS),
        expected_n_turns=N_TURNS_TOTAL,
    )
    mean_len = mean_turn_token_length(all_conversations)
    print(f"  Mean turn token length (whitespace): {mean_len:.1f}", flush=True)

    # Step 4: write aggregate JSONL EARLY (before stats / upload steps),
    # so any downstream stat-writing or upload failure does not lose the
    # aggregate that the per-domain checkpoint files were assembled into.
    # (Plan v1 had this after the cross-check, which was the round-9 r4
    # data-loss trap: the hard sanity raise tossed the aggregate write.)
    print("\nStep 4: writing aggregate output JSONL...", flush=True)
    write_corpus_jsonl(all_conversations, corpus_tag="incontext", output_path=OUTPUT_PATH)

    # Step 5: informational length-stats write + soft asymmetry warning.
    print("\nStep 5: corpus length stats (informational)...", flush=True)
    if DRIFT_CORPUS_PATH.exists():
        drift_conversations = read_corpus_jsonl(DRIFT_CORPUS_PATH)
        stats = corpus_length_stats(drift_conversations, all_conversations)
        STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
        ratio = stats["ratio_incontext_to_drift"]
        print(
            f"  Length ratios in-context / drift: "
            f"user={ratio['user']:.3f}, assistant={ratio['assistant']:.3f}, "
            f"all={ratio['all']:.3f}",
            flush=True,
        )
        if stats["length_asymmetry_warning"]:
            print(
                f"  WARNING: corpus length asymmetry exceeds expected range "
                f"(assistant-side ratio {ratio['assistant']:.3f} outside "
                f"[{stats['warning_thresholds']['low']}, "
                f"{stats['warning_thresholds']['high']}]). This is "
                f"non-fatal; the eval rig handles length-matching at "
                f"prefix-selection time via the B-incontext-length@k arm.",
                flush=True,
            )
        print(f"  Wrote {STATS_PATH}", flush=True)
    else:
        # Drift corpus not on local disk — emit a stats file with only the
        # in-context side filled in, so the analyzer can still consume a
        # well-formed JSON. The eval rig is the canonical place to verify
        # length-matching anyway.
        partial = corpus_length_stats([], all_conversations)
        STATS_PATH.write_text(json.dumps(partial, indent=2) + "\n")
        print(
            f"  Drift corpus not on local disk at {DRIFT_CORPUS_PATH}; "
            f"wrote in-context-only stats to {STATS_PATH} (ratios are 0.0). "
            f"Length-matching verification happens in the eval rig.",
            flush=True,
        )

    # Step 6: sample print.
    print("\nStep 6: sample inspection (1 conv per domain)...", flush=True)
    samples = sample_for_inspection(all_conversations, domains=INCONTEXT_DOMAINS, n_per_domain=1)
    for s in samples:
        print(f"\n--- {s['conversation_id']} ({s['domain']}) ---", flush=True)
        print(f"  topic: {s['topic'][:120]}", flush=True)
        for i, t in enumerate(s["turns"][:2]):
            print(f"  turn {i + 1} ({t['role']}): {t['content'][:120]!r}", flush=True)

    # Step 7: upload to HF Hub.
    if args.no_upload:
        print("\nStep 7: SKIPPED (--no-upload set)", flush=True)
    else:
        print(f"\nStep 7: uploading to HF Hub bucket {HUB_BUCKET!r}...", flush=True)
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
