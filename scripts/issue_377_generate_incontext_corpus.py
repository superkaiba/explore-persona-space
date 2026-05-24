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
    parser.add_argument(
        "--allow-missing-drift-summary",
        action="store_true",
        help=(
            "Skip the ±10%% length-match cross-check against the drift corpus "
            "even if data/issue377_drift/drift_summary.json is missing. Use "
            "ONLY when the drift corpus is intentionally being generated "
            "AFTER the in-context corpus — otherwise the missing summary is "
            "a bug, not a benign absence, and we should fail loudly per plan "
            "§4.2 sanity check (2)."
        ),
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
    elif args.allow_missing_drift_summary:
        print(
            f"  WARNING: drift summary {DRIFT_SUMMARY_PATH} not found and "
            f"--allow-missing-drift-summary set; skipping ±"
            f"{LENGTH_MATCH_TOLERANCE * 100:.0f}% length-match cross-check. "
            f"You MUST re-run this script after the drift corpus is "
            f"generated, or generate the drift corpus first, to verify the "
            f"plan §4.2 sanity check (2) invariant.",
            flush=True,
        )
    else:
        raise RuntimeError(
            f"Sibling drift corpus summary not found at {DRIFT_SUMMARY_PATH}. "
            f"The plan §4.2 sanity check (2) ±{LENGTH_MATCH_TOLERANCE * 100:.0f}% "
            f"mean-turn-length invariant cannot be verified without it. "
            f"Generate the drift corpus first via "
            f"`uv run python scripts/issue_377_generate_drift_corpus.py`, "
            f"or — if you intentionally want the in-context corpus first — "
            f"re-run with `--allow-missing-drift-summary` and remember to "
            f"come back and verify the match after the drift corpus is ready."
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
