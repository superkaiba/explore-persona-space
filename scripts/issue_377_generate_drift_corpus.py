#!/usr/bin/env python3
"""Generate the issue #377 drift conversation corpus.

200 multi-turn drift conversations across 4 domains
(coding / writing / therapy / philosophy), each ``N_TURNS_TOTAL`` turns
(15 under the round-6 protocol replication of Lu et al. 2026). Round-9
(2026-05-23) paper-aligned the domain set with Lu et al. 2026 §4.1:
dropped ``hostile_jailbreak`` (auditor-side RLHF refusal cascade on
the social-engineering frame the paper doesn't use) and ``roleplay``
(our addition, not in the paper); added ``coding`` and ``writing``
(the two paper §4.1 domains we were missing). Auditor and target are
rotated per-conversation between Claude-Sonnet-4.5 and GPT-5 (see
``assign_auditor_model`` in the data_gen corpus module). Output goes
to ``data/issue377_drift/drift_conversations.jsonl`` and is auto-
uploaded to the HF Hub data repo under ``issue377_drift/v1/``.

See ``tasks/running/377/plans/v1.md`` §4.2 for the full design. The
sibling in-context corpus (4 neutral domains, same shape) is generated
by ``scripts/issue_377_generate_incontext_corpus.py``; the two scripts
share helpers in ``explore_persona_space.data_gen.issue377_corpus``.

Usage::

    uv run python scripts/issue_377_generate_drift_corpus.py
    uv run python scripts/issue_377_generate_drift_corpus.py --no-upload  # local-only
    uv run python scripts/issue_377_generate_drift_corpus.py --bust-seed-cache  # force re-seed
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
    read_corpus_jsonl,
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
CORPUS_TAG = "drift"


def _per_domain_path(domain_name: str) -> Path:
    """Per-domain checkpoint file (Option A from the issue #377 r8 brief).

    Each per-domain JSONL is a self-contained checkpoint of one domain's
    50 conversations x 15 turns. It is written immediately after the
    conversation loop for that domain completes — BEFORE the next
    domain starts — so an FIX-3 abort (or any other mid-run crash) in
    a later domain cannot lose the earlier domains' data. The final
    concatenated ``drift_conversations.jsonl`` is built from these
    per-domain files after all 4 domains succeed; the per-domain files
    are NEVER deleted (they remain the recoverable checkpoints).
    """
    return DATA_DIR / f"conversations_{domain_name}.jsonl"


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
            "across reruns. Use a non-default seed to test sensitivity "
            "of headline metrics to the exact assignment without changing "
            "the rotation pool composition."
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
            "a checkpoint exists. Use when you need to regenerate the "
            "corpus end-to-end (e.g. after a DomainSpec wording change) "
            "and the seed cache bust isn't enough."
        ),
    )
    parser.add_argument(
        "--n-turns",
        type=int,
        default=N_TURNS_TOTAL,
        help=(
            f"Override turns-per-conversation (default: N_TURNS_TOTAL="
            f"{N_TURNS_TOTAL} from issue377_corpus.py). Used by #408 "
            "Phase A.0.0.1 to generate a 30-turn long corpus for the "
            "extrapolation cells B@25 (slice_n=24). Threaded all the way "
            "through to run_conversation_loop + post_gen_sanity_checks."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override DATA_DIR (default: data/issue377_drift/). When set, "
            "OUTPUT_PATH, SEED_CACHE_PATH, and per-domain checkpoints are "
            "all rooted under this directory. Used by #408 Phase A.0.0.1 "
            "to redirect outputs into data/issue408_long/ without "
            "overwriting the #377 corpus."
        ),
    )
    parser.add_argument(
        "--only-domain",
        type=str,
        default=None,
        help=(
            "Restrict Step 2 conversation generation to a single named "
            "domain (e.g. --only-domain therapy). When set, Step 2 runs "
            "exactly ONE domain's conversation loop and writes its "
            "per-domain checkpoint; Steps 3-7 (sanity / aggregate / "
            "upload) are SKIPPED. Used by #408 Phase A.0.0.1's parallel "
            "long-corpus orchestrator (task #408 round-3, 2026-05-29) to "
            "fan out 4 concurrent subprocesses (1 per drift domain) "
            "instead of running the 4 domains sequentially in one "
            "process. The orchestrator runs this wrapper ONCE MORE with "
            "no --only-domain to hit the full-resume path and run "
            "Steps 3-7 on all 4 checkpoints together. Domain name MUST "
            "match one of DRIFT_DOMAINS or the script exits with a "
            "loud error."
        ),
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help=(
            "Run Step 1 (persona+topic seeding via Anthropic Batch) and "
            "exit. The seed cache is written to SEED_CACHE_PATH so "
            "subsequent --only-domain subprocess runs hit the cache "
            "instead of racing on parallel seed batches. Used by #408 "
            "Phase A.0.0.1's parallel orchestrator (task #408 round-3) "
            "to pre-seed BEFORE fanning out per-domain subprocesses."
        ),
    )
    parser.add_argument(
        "--skip-finalization",
        action="store_true",
        help=(
            "Skip Steps 3-7 (post-gen sanity checks, aggregate write, "
            "sample inspection, HF Hub upload, summary write). Used "
            "together with --only-domain by the parallel orchestrator "
            "so per-domain subprocesses exit AS SOON AS the per-domain "
            "checkpoint is on disk, without redundant sanity checks "
            "(which the final no-flags orchestrator call performs once "
            "across all 4 checkpoints)."
        ),
    )
    args = parser.parse_args()

    # Task #408 (v1.2 fix M1) — rebind module globals when --output-dir is
    # given so all downstream helpers (per-domain paths, summary writer,
    # uploader) land under the requested directory.
    global DATA_DIR, OUTPUT_PATH, SEED_CACHE_PATH
    if args.output_dir is not None:
        DATA_DIR = args.output_dir
        OUTPUT_PATH = DATA_DIR / "drift_conversations.jsonl"
        SEED_CACHE_PATH = DATA_DIR / "persona_topic_seeds_drift.json"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Task #408 round-3 (2026-05-29) parallelization-flag validation.
    # --only-domain must name a real DRIFT_DOMAINS member; --seed-only
    # and --skip-finalization are independent flags but combine with
    # --only-domain in the parallel-fanout shape.
    if args.only_domain is not None:
        valid_names = {d.name for d in DRIFT_DOMAINS}
        if args.only_domain not in valid_names:
            sys.exit(
                f"FAIL: --only-domain={args.only_domain!r} is not a member of "
                f"DRIFT_DOMAINS={sorted(valid_names)}. Domain names must match exactly."
            )
    if args.seed_only and args.only_domain is not None:
        sys.exit(
            "FAIL: --seed-only and --only-domain are mutually exclusive. "
            "Seed-only runs Step 1 across ALL domains; --only-domain restricts "
            "Step 2 to one domain. The intended fanout shape is: one seed-only "
            "call THEN N concurrent --only-domain calls."
        )

    print(
        f"=== Issue #377 drift corpus generation ===\n"
        f"  Domains: {[d.name for d in DRIFT_DOMAINS]}\n"
        f"  Expected: {N_CONVERSATIONS_PER_DOMAIN} convs/domain x "
        f"{len(DRIFT_DOMAINS)} domains = "
        f"{N_CONVERSATIONS_PER_DOMAIN * len(DRIFT_DOMAINS)} total\n"
        f"  Turns per conversation: {args.n_turns}\n"
        f"  Output: {OUTPUT_PATH}\n"
        f"  Parallel-fanout mode: only_domain={args.only_domain}, "
        f"seed_only={args.seed_only}, skip_finalization={args.skip_finalization}\n",
        flush=True,
    )

    # Step 0: optional seed cache busting. Required between round-3 and
    # round-4 because the therapy DomainSpec changed (crisis-state →
    # work-stress). Without this, ``seed_personas_and_topics`` reloads the
    # round-3 crisis-state personas from disk and the conversation loop
    # still hits Sonnet's refusal surface.
    if args.bust_seed_cache and SEED_CACHE_PATH.exists():
        print(f"  Busting seed cache at {SEED_CACHE_PATH}...", flush=True)
        SEED_CACHE_PATH.unlink()

    # Resume-skip detection (round-9 r2 patch, 2026-05-24).
    #
    # Round-8 added per-domain checkpoint writes (Step 2). Round 9 r1
    # tripped the post-gen-sanity hard-raise on a single trigger-key
    # leak in therapy_p2_t6 after all 4 per-domain JSONLs were already
    # on disk — meaning every conversation was generated and persisted,
    # yet the script aborts at Step 3 and a naive re-run would re-pay
    # the ~3-hour batch-API spend. The resume-skip path closes that gap:
    # if a per-domain JSONL exists on disk, we skip Step 1's seeding work
    # (personas are frozen in the checkpoint) and Step 2's batch-API
    # loop for that domain, and load the cached conversations back via
    # ``read_corpus_jsonl``. Step 3+ then proceed normally on the
    # reconstituted list.
    #
    # ``--no-resume`` forces a full from-scratch run (useful when
    # regenerating end-to-end after a wording change). Partial resume is
    # supported: if some per-domain JSONLs exist and others don't, the
    # missing domains are run normally and the existing ones are loaded.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Task #408 round-3 (2026-05-29): --seed-only path. Run Step 1
    # (persona+topic seeding via Anthropic Batch) and exit, leaving the
    # cache on disk for subsequent --only-domain subprocess fanout.
    # Without pre-seeding, 4 parallel --only-domain subprocesses would
    # each see the cache missing, each spawn redundant seed batches, and
    # race on writing the same cache file. Pre-seeding is ~30s of API
    # latency vs N redundant + corrupt-cache risk.
    if args.seed_only:
        print("Step 1 (--seed-only): seeding personas + topics...", flush=True)
        personas_by_domain = seed_personas_and_topics(
            DRIFT_DOMAINS,
            cache_path=SEED_CACHE_PATH,
            custom_id_prefix="drift",
        )
        for name, personas in personas_by_domain.items():
            total_topics = sum(len(p["topics"]) for p in personas)
            print(f"  {name}: {len(personas)} personas, {total_topics} topics", flush=True)
        print(f"\n  Cache written to {SEED_CACHE_PATH}", flush=True)
        print("\n=== Done (--seed-only; skipping Steps 2-7) ===", flush=True)
        return 0

    existing_per_domain: dict[str, Path] = {}
    if not args.no_resume:
        for domain in DRIFT_DOMAINS:
            path = _per_domain_path(domain.name)
            if path.exists():
                existing_per_domain[domain.name] = path

    # --only-domain narrows the in-scope domain set to exactly one. The
    # resume-skip / all-resume bookkeeping below is computed against the
    # narrowed set so a parallel-fanout subprocess for domain X is not
    # confused by domain Y's checkpoint already being on disk.
    in_scope_domains: tuple = (
        tuple(d for d in DRIFT_DOMAINS if d.name == args.only_domain)
        if args.only_domain is not None
        else DRIFT_DOMAINS
    )

    missing_domains = [d for d in in_scope_domains if d.name not in existing_per_domain]
    all_resume = (
        len([d for d in in_scope_domains if d.name in existing_per_domain]) == len(in_scope_domains)
        and len(missing_domains) == 0
    )

    if all_resume:
        # Full resume: skip Step 1 (no need to seed) AND Step 2 entirely.
        # In the parallel-fanout shape (#408 round-3), only_domain=None
        # AND all 4 per-domain JSONLs exist hits this branch and runs
        # Steps 3-7 across all 4 checkpoints.
        print(
            "Step 1+2: SKIPPED — all "
            f"{len(in_scope_domains)} per-domain JSONLs exist on disk: "
            f"{[str(p) for p in existing_per_domain.values()]}",
            flush=True,
        )
        print(
            "  (use --no-resume to force a from-scratch regeneration)",
            flush=True,
        )
        all_conversations: list[dict] = []
        for domain in DRIFT_DOMAINS:
            if domain.name in existing_per_domain:
                cached = read_corpus_jsonl(existing_per_domain[domain.name])
                print(
                    f"  Loaded {len(cached)} conversations from "
                    f"{existing_per_domain[domain.name]} ({domain.name})",
                    flush=True,
                )
                all_conversations.extend(cached)
    else:
        # Step 1: seed personas + topics (cached). Always run when any
        # domain is missing, because run_conversation_loop needs the
        # full personas_by_domain dict for those missing domains; for
        # the already-checkpointed domains the cached personas are
        # harmless (the loop is skipped for them).
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
            DRIFT_DOMAINS,
            cache_path=SEED_CACHE_PATH,
            custom_id_prefix="drift",
        )
        for name, personas in personas_by_domain.items():
            total_topics = sum(len(p["topics"]) for p in personas)
            print(f"  {name}: {len(personas)} personas, {total_topics} topics", flush=True)

        # Step 2: per-domain conversation loop. Iterates over the
        # narrowed ``in_scope_domains`` (full DRIFT_DOMAINS, OR a
        # single-element tuple when --only-domain is set by the #408
        # parallel orchestrator). Each domain's per-domain JSONL is
        # written the moment its loop completes (round-8 invariant).
        # Already-checkpointed in-scope domains are skip-loaded.
        #
        # Round-8 (post-incident, 2026-05-23): write each domain's conversations
        # to its own JSONL the moment that domain's loop completes — BEFORE the
        # next domain starts. Round 6 ran 3 clean domains then aborted at
        # hostile_jailbreak turn 5 (FIX-3 mid-run ceiling), and the script's
        # only final write was at the END of the all-domains loop, so the 3
        # clean domains' data was lost. With per-domain writes, an FIX-3 abort
        # (or any other later-domain crash) loses ONLY the in-flight domain's
        # partial data; prior domains' checkpoints are already on disk.
        # Per-domain files are NEVER deleted; they remain the recoverable
        # checkpoints. The aggregate ``drift_conversations.jsonl`` is built
        # from them after all 4 domains succeed.
        scope_label = f" (--only-domain={args.only_domain})" if args.only_domain is not None else ""
        print(
            f"\nStep 2: running conversation loops (per-domain checkpoint){scope_label}...",
            flush=True,
        )
        all_conversations = []
        for domain in in_scope_domains:
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
                custom_id_prefix="drift",
                n_turns=args.n_turns,
                rotation_seed=args.rotation_seed,
            )
            # Write THIS domain's checkpoint immediately. If the next
            # domain's loop aborts (FIX-3, OOM, network), this file is
            # already on disk and unaffected.
            domain_path = _per_domain_path(domain.name)
            write_corpus_jsonl(convs, corpus_tag=CORPUS_TAG, output_path=domain_path)
            all_conversations.extend(convs)

    # Task #408 round-3 (2026-05-29): --skip-finalization path. Exit
    # immediately after the per-domain checkpoint(s) are on disk,
    # before running sanity checks / aggregate write / sample
    # inspection / HF upload / summary write. The parallel
    # orchestrator's final no-flags call hits the full-resume branch
    # above and runs Steps 3-7 across all 4 checkpoints at once.
    if args.skip_finalization or args.only_domain is not None:
        n_convs = len(all_conversations)
        scope = (
            f"--only-domain={args.only_domain}" if args.only_domain is not None else "all domains"
        )
        print(
            f"\n=== Done (--skip-finalization or --only-domain set; "
            f"wrote {n_convs} convs for {scope}, skipping Steps 3-7) ===",
            flush=True,
        )
        return 0

    # Step 3: post-gen sanity checks.
    print("\nStep 3: post-gen sanity checks...", flush=True)
    post_gen_sanity_checks(
        all_conversations,
        expected_n_conversations=N_CONVERSATIONS_PER_DOMAIN * len(DRIFT_DOMAINS),
        expected_n_turns=args.n_turns,
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
                "n_turns_per_conversation": args.n_turns,
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
