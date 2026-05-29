#!/usr/bin/env python3
"""Generate a 30-turn long-form extrapolation corpus for #408 cells B@25.

Covers BOTH DRIFT_DOMAINS (therapy / philosophy / coding / writing) and
INCONTEXT_DOMAINS (math / history / factual_qa / code_review) at the
default N_CONVERSATIONS_PER_DOMAIN=50 per domain. Outputs land under
``data/issue408_long/`` and a concatenated aggregate is written to
``data/issue408_long/long_conversations.jsonl`` for downstream consumers
(the #408 multi-turn training-row generator and ``eval_issue408.py``'s
B@15 + B@25 cells, after the code-review v1 round-2 fix routes k=15
to this corpus too to break the B@15 ≡ B@20 prefix collision).

**Plan-vs-actual conversation count** (code-review v1 round-2 Minor #9):
The cherry-picked #377 wrappers hardcode 50 convs/domain x 4 domains
each = 200 conv per wrapper = **400 total** (200 drift + 200 incontext).
Plan §10 v1.2 documented the estimate as "200 long conversations"
because the plan author conflated the per-wrapper 200 with the
combined-aggregate total. The 400-conv reality is acceptable for #408
because:
  (1) It strictly INCREASES the pool size for k=15 + k=25 cell
      sampling, which improves prefix-uniqueness (more distinct long
      conversations = fewer duplicate-prefix risks under the
      ``_resample_until_prefix_found`` retry logic).
  (2) The Anthropic Batch cost scales linearly: plan estimate was
      ~$10-13 for 200 convs, actual is **~$20-26 for 400 convs**.
      Within the §9 compute-projection envelope; no
      ``epm:compute-deviation`` marker needed (ratio < 2x).
  (3) Adding a per-domain limit flag to the wrappers + the data_gen
      library would touch the #377-byte-comparable corpus-gen path,
      raising regression risk for an unrelated change.

**Round-3 PARALLELIZATION** (task #408 v3, 2026-05-29):

The v2 run aborted at ~10.2h elapsed (3 of 8 long-corpus domains
generated) because the long-corpus gen was fully sequential: 2
wrappers x 4 domains x 30 turns ≈ 26h on the API-latency side. The
8 domains are independent (turn N+1 within a domain depends on turn
N, but domain X's turn loop doesn't touch domain Y's). Round-3
restructures this orchestrator into a per-domain fanout that runs
the 8 domain loops CONCURRENTLY (one subprocess per domain).

Architecture:

  Per wrapper (drift, incontext):
    1. Pre-seed pass (--seed-only): one subprocess populates the
       persona+topic seed cache (~30s of API latency). Done up-front
       so the N=4 per-wrapper concurrent per-domain subprocesses
       don't all see a missing cache and race to repopulate it.
    2. Fanout: ``len(DOMAINS)`` concurrent subprocesses, each
       ``--only-domain <D>`` invoking the wrapper. Concurrency is
       capped via ``--max-parallel`` (default 4). Each subprocess
       writes its per-domain checkpoint (Step 2) and exits without
       running Steps 3-7 (sanity / aggregate / upload).
    3. Finalize: one subprocess invokes the wrapper with NO
       --only-domain. This hits the full-resume branch (all 4
       per-domain checkpoints on disk) and runs Steps 3-7 once over
       all 4 domains.

  After both wrappers complete, ``_concat_outputs()`` writes the
  combined long-corpus JSONL exactly as before.

Note: turn count and per-domain conversation count are byte-identical
to the v2 (sequential) run — the experiment is UNCHANGED. Only the
wall-clock shape changes (~26h sequential -> ~3-4h with 4-way
per-wrapper concurrency, since per-domain wall is ~3.3h and the 4
domains within a wrapper run in parallel).

Why subprocess (not in-process import) — v1.2 fix M1:

The cherry-picked #377 wrappers' ``main()`` is the integration point —
it owns argparse, the per-domain checkpoint write, the Step-3 sanity
gate that drops malformed conversations, the Step-5 length stats, the
Step-7 HF upload. Importing pieces would force #408 to re-implement
these defenses or skip them. Subprocess re-uses end-to-end and isolates
wrapper crashes from this orchestrator.

The wrappers were patched in the Phase A.0.0.0 setup commit to accept
``--n-turns INT`` and ``--output-dir PATH``; this orchestrator invokes
both with ``--n-turns 30`` and per-corpus output dirs, then concatenates
the two aggregate JSONLs. The round-3 patch added ``--only-domain``,
``--seed-only``, ``--skip-finalization`` to both wrappers; this
orchestrator uses all three for the per-domain fanout.

Usage::

    uv run python scripts/issue_408_generate_long_corpus.py
    uv run python scripts/issue_408_generate_long_corpus.py --no-upload
    uv run python scripts/issue_408_generate_long_corpus.py --max-parallel 4

Expected wall time (post-round-3): ~3-4h per wrapper (4 concurrent
30-turn domain loops at ~3.3h each, API-bound), ~6-8h for both
wrappers RUN SEQUENTIALLY (drift then incontext, to keep the
per-minute Batch API submit volume manageable). Down from ~26h
sequential.

Expected Anthropic Batch cost: ~$20-26 (Sonnet 4.5 at $1.50/MTok input
+ $7.50/MTok output, 8 domains x 50 convs x ~30 turns = 400 convs).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "issue408_long"
DRIFT_OUT = OUTPUT_DIR / "long_drift"
INCONTEXT_OUT = OUTPUT_DIR / "long_incontext"
COMBINED_PATH = OUTPUT_DIR / "long_conversations.jsonl"

# Per-wrapper domain lists (kept here for the per-domain fanout planner).
# Source of truth lives in
# ``src/explore_persona_space/data_gen/issue377_corpus.py``
# (DRIFT_DOMAINS / INCONTEXT_DOMAINS). Mirrored verbatim here as plain
# strings so this orchestrator does not import the full data_gen module
# (it only needs to know domain NAMES for the --only-domain fanout; the
# wrapper itself validates the name against its own DOMAINS tuple).
DRIFT_DOMAIN_NAMES: tuple[str, ...] = ("therapy", "philosophy", "coding", "writing")
INCONTEXT_DOMAIN_NAMES: tuple[str, ...] = ("math", "history", "factual_qa", "code_review")

N_TURNS = 30  # plan §10 Reproducibility Card; 24-turn slice for B@25 + 6-turn headroom
ROTATION_SEED = 408  # distinct from #377's default (0) to avoid pool re-use


def _smoke_check_cli_flags() -> None:
    """Verify the wrappers expose the parallelization-fanout flags.

    Plan v1.2 §10 "Smoke tests at start of run" + round-3 parallelization
    require: ``--n-turns``, ``--output-dir`` (v1.2), and the round-3 trio
    ``--only-domain``, ``--seed-only``, ``--skip-finalization``. Fail
    loud (SystemExit) with the exact `--help` snippet so a forgotten
    setup commit doesn't manifest as a cryptic mid-batch crash.
    """
    required_flags = (
        "--n-turns",
        "--output-dir",
        "--only-domain",
        "--seed-only",
        "--skip-finalization",
    )
    for wrapper in (
        "scripts/issue_377_generate_drift_corpus.py",
        "scripts/issue_377_generate_incontext_corpus.py",
    ):
        result = subprocess.run(
            ["uv", "run", "python", wrapper, "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
        if result.returncode != 0:
            sys.exit(
                f"FAIL: {wrapper} --help exited rc={result.returncode}. stderr:\n{result.stderr}"
            )
        for flag in required_flags:
            if flag not in result.stdout:
                sys.exit(
                    f"FAIL: {wrapper} is missing the {flag} CLI flag. "
                    "Run the round-3 parallelization setup commit which patches both "
                    "wrappers to add --only-domain + --seed-only + --skip-finalization "
                    "(and the prior --n-turns + --output-dir from Phase A.0.0.0)."
                )


def _common_wrapper_args(
    wrapper: str, output_dir: Path, no_upload: bool, bust_seed_cache: bool
) -> list[str]:
    """Common subprocess argv for the #377 wrapper.

    Used by BOTH the pre-seed call and the per-domain fanout calls.
    ``--bust-seed-cache`` must ONLY be passed on the pre-seed call —
    otherwise the per-domain subprocesses would each delete the cache
    the pre-seed pass just wrote.
    """
    cmd = [
        "uv",
        "run",
        "python",
        wrapper,
        "--n-turns",
        str(N_TURNS),
        "--output-dir",
        str(output_dir),
        "--rotation-seed",
        str(ROTATION_SEED),
    ]
    if bust_seed_cache:
        cmd.append("--bust-seed-cache")
    if no_upload:
        cmd.append("--no-upload")
    return cmd


def _run_seed_only(wrapper: str, output_dir: Path) -> None:
    """Pre-seed pass: populate the persona+topic cache so per-domain
    subprocesses don't race on the seed batch.

    --bust-seed-cache is ON here to ensure the long-corpus seed cache
    is fresh (the orchestrator runs ROTATION_SEED=408 vs #377's
    default 0; without the bust, a stale cache from a prior #377 run
    in the same DATA_DIR would silently overwrite).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _common_wrapper_args(wrapper, output_dir, no_upload=True, bust_seed_cache=True)
    cmd.append("--seed-only")
    print(f"--> SEED-ONLY: {' '.join(cmd)}", flush=True)
    # check=True: fail loud — without the seed cache, per-domain fanout
    # would each spawn redundant seed batches and corrupt the cache.
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ})


def _run_one_domain(
    wrapper: str, output_dir: Path, domain_name: str, no_upload: bool
) -> tuple[str, int, str]:
    """Run a single ``--only-domain <D>`` subprocess.

    Returns (domain_name, returncode, tail of stderr-or-stdout) for the
    parallel-fanout reporter. ``check=False`` so a per-domain failure
    doesn't abort the in-flight siblings; the orchestrator inspects all
    returncodes after the as_completed() loop and fails loud at the end.
    """
    cmd = _common_wrapper_args(wrapper, output_dir, no_upload=no_upload, bust_seed_cache=False)
    cmd.extend(["--only-domain", domain_name, "--skip-finalization"])
    print(f"--> DOMAIN={domain_name}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        env={**os.environ},
        capture_output=True,
        text=True,
    )
    # Stream the tail of stdout/stderr so the orchestrator log retains
    # at least a hint of what each per-domain subprocess emitted.
    tail = result.stdout[-2000:] if result.stdout else result.stderr[-2000:]
    print(
        f"<-- DOMAIN={domain_name}: rc={result.returncode} (tail {len(tail)} chars):\n{tail}",
        flush=True,
    )
    return domain_name, result.returncode, tail


def _run_finalize(wrapper: str, output_dir: Path, no_upload: bool) -> None:
    """Finalization pass: no --only-domain, hits the full-resume branch.

    All per-domain checkpoints are on disk; the wrapper skips Steps 1+2
    entirely and runs Steps 3-7 (sanity / aggregate write / stats / sample
    inspection / HF upload / summary) once across all 4 checkpoints.
    """
    cmd = _common_wrapper_args(wrapper, output_dir, no_upload=no_upload, bust_seed_cache=False)
    print(f"--> FINALIZE (full-resume): {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ})


def _run_wrapper_parallel(
    wrapper: str,
    output_dir: Path,
    domain_names: tuple[str, ...],
    max_parallel: int,
    no_upload: bool,
) -> None:
    """End-to-end parallel run for ONE wrapper.

    Sequence: seed-only -> N concurrent per-domain (capped by
    max_parallel) -> finalize. Fails loud if any per-domain subprocess
    returned non-zero (after all in-flight siblings finish).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: pre-seed (sequential, one shot).
    _run_seed_only(wrapper, output_dir)

    # Step 2: per-domain fanout (concurrent).
    print(
        f"--> Parallel fanout: {len(domain_names)} domains, max_parallel={max_parallel}",
        flush=True,
    )
    failures: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = {
            ex.submit(_run_one_domain, wrapper, output_dir, name, no_upload): name
            for name in domain_names
        }
        for fut in as_completed(futures):
            domain_name, rc, tail = fut.result()
            if rc != 0:
                failures.append((domain_name, rc, tail))

    if failures:
        msg = "\n".join(f"  - {n}: rc={rc} (tail: {t[-300:]!r})" for n, rc, t in failures)
        sys.exit(
            f"FAIL: {len(failures)} of {len(domain_names)} per-domain subprocesses failed "
            f"for {wrapper}:\n{msg}\n"
            "Per-domain checkpoints from successful sibling subprocesses are on disk; "
            "re-invoke this orchestrator to resume."
        )

    # Step 3: finalize (sequential, full-resume across all 4 checkpoints).
    _run_finalize(wrapper, output_dir, no_upload=no_upload)


def _concat_outputs() -> int:
    """Concatenate the two wrappers' aggregate JSONLs into one combined file.

    Per-corpus aggregates live at
    ``data/issue408_long/long_drift/drift_conversations.jsonl`` and
    ``data/issue408_long/long_incontext/incontext_conversations.jsonl``
    (the rebound DATA_DIR shape from the setup commit).
    """
    drift_jsonl = DRIFT_OUT / "drift_conversations.jsonl"
    incontext_jsonl = INCONTEXT_OUT / "incontext_conversations.jsonl"
    for src in (drift_jsonl, incontext_jsonl):
        if not src.exists():
            sys.exit(
                f"FAIL: expected wrapper output {src} not on disk. "
                "The wrapper's Step-4 aggregate-write failed; inspect "
                "the wrapper's stdout for the per-domain checkpoint state."
            )

    n_total = 0
    with open(COMBINED_PATH, "w") as out:
        for src in (drift_jsonl, incontext_jsonl):
            with open(src) as f:
                for line in f:
                    if not line.strip():
                        continue
                    out.write(line)
                    n_total += 1
    print(f"  Wrote {n_total} long ({N_TURNS}-turn) conversations -> {COMBINED_PATH}")
    return n_total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help=(
            "Pass --no-upload through to BOTH wrapper subprocesses (skip HF "
            "Hub upload for local-only dry runs). The combined aggregate "
            "JSONL is still written to disk."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help=(
            "Max concurrent per-domain subprocesses PER WRAPPER (round-3 "
            "parallelization). Default 4 = full per-wrapper fanout (one "
            "subprocess per domain). Lower to 2 if Anthropic Batch "
            "rate-limits become a problem (each per-domain subprocess "
            "uses its own batch slot)."
        ),
    )
    args = parser.parse_args()

    print(
        "=== Issue #408 long-form corpus generation (30 turns, parallel) ===",
        flush=True,
    )
    print(f"  Output dir: {OUTPUT_DIR}", flush=True)
    print(f"  Drift sub-corpus:     {DRIFT_OUT}", flush=True)
    print(f"  Incontext sub-corpus: {INCONTEXT_OUT}", flush=True)
    print(f"  Aggregate:            {COMBINED_PATH}", flush=True)
    print(f"  Turns per conv:       {N_TURNS}", flush=True)
    print(f"  Rotation seed:        {ROTATION_SEED}", flush=True)
    print(f"  Max parallel:         {args.max_parallel} per wrapper", flush=True)
    print(f"  Drift domains:        {list(DRIFT_DOMAIN_NAMES)}", flush=True)
    print(f"  Incontext domains:    {list(INCONTEXT_DOMAIN_NAMES)}", flush=True)
    print("", flush=True)

    print(
        "Step 0: smoke check — wrappers expose --n-turns + --output-dir + "
        "--only-domain + --seed-only + --skip-finalization",
        flush=True,
    )
    _smoke_check_cli_flags()
    print("  PASS", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "\nStep 1: drift wrapper (therapy / philosophy / coding / writing) — parallel",
        flush=True,
    )
    _run_wrapper_parallel(
        "scripts/issue_377_generate_drift_corpus.py",
        DRIFT_OUT,
        DRIFT_DOMAIN_NAMES,
        max_parallel=args.max_parallel,
        no_upload=args.no_upload,
    )

    print(
        "\nStep 2: incontext wrapper (math / history / factual_qa / code_review) — parallel",
        flush=True,
    )
    _run_wrapper_parallel(
        "scripts/issue_377_generate_incontext_corpus.py",
        INCONTEXT_OUT,
        INCONTEXT_DOMAIN_NAMES,
        max_parallel=args.max_parallel,
        no_upload=args.no_upload,
    )

    print("\nStep 3: concatenate aggregates -> combined long-corpus JSONL", flush=True)
    n_total = _concat_outputs()

    summary_path = OUTPUT_DIR / "long_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "n_conversations_total": n_total,
                "n_turns_per_conversation": N_TURNS,
                "rotation_seed": ROTATION_SEED,
                "drift_subcorpus": str(DRIFT_OUT.relative_to(ROOT)),
                "incontext_subcorpus": str(INCONTEXT_OUT.relative_to(ROOT)),
                "aggregate": str(COMBINED_PATH.relative_to(ROOT)),
                "parallelization": {
                    "max_parallel_per_wrapper": args.max_parallel,
                    "drift_domains": list(DRIFT_DOMAIN_NAMES),
                    "incontext_domains": list(INCONTEXT_DOMAIN_NAMES),
                },
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
