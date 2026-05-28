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
the two aggregate JSONLs.

Usage::

    uv run python scripts/issue_408_generate_long_corpus.py
    uv run python scripts/issue_408_generate_long_corpus.py --no-upload

Expected wall time: ~45-60 min (Anthropic Batch end-to-end, 400 convs).
Expected Anthropic Batch cost: ~$20-26 (Sonnet 4.5 at $1.50/MTok input
+ $7.50/MTok output, 8 domains x 50 convs x ~30 turns = 400 convs).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "issue408_long"
DRIFT_OUT = OUTPUT_DIR / "long_drift"
INCONTEXT_OUT = OUTPUT_DIR / "long_incontext"
COMBINED_PATH = OUTPUT_DIR / "long_conversations.jsonl"

N_TURNS = 30  # plan §10 Reproducibility Card; 24-turn slice for B@25 + 6-turn headroom
ROTATION_SEED = 408  # distinct from #377's default (0) to avoid pool re-use


def _smoke_check_cli_flags() -> None:
    """Verify the wrappers expose --n-turns (added in Phase A.0.0.0 M1 setup).

    Plan v1.2 §10 "Smoke tests at start of run" requires this check at
    the top of the orchestrator. Fail loud (SystemExit) with the exact
    `--help` snippet so a forgotten setup commit doesn't manifest as a
    cryptic mid-batch crash.
    """
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
        if "--n-turns" not in result.stdout:
            sys.exit(
                f"FAIL: {wrapper} is missing the --n-turns CLI flag. "
                "Run the Phase A.0.0.0 setup commit (M1) which patches both "
                "wrappers to add --n-turns + --output-dir."
            )
        if "--output-dir" not in result.stdout:
            sys.exit(
                f"FAIL: {wrapper} is missing the --output-dir CLI flag. "
                "Run the Phase A.0.0.0 setup commit (M1) which patches both "
                "wrappers to add --n-turns + --output-dir."
            )


def _run_wrapper(wrapper: str, output_dir: Path, no_upload: bool) -> None:
    """Invoke one #377 wrapper with the #408 long-corpus overrides.

    Subprocess env passthrough — explicit env= kwarg per the
    CLAUDE.md "Subprocess env passthrough" rule; load_dotenv() is
    called at module top so HF_TOKEN / ANTHROPIC_API_KEY are present
    in os.environ when the wrapper batch-API submits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
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
        # Bust the seed cache so the long-corpus draws fresh personas/topics
        # rather than re-using the legacy #377 seeds (whose cache key is
        # path-existence-only). The new --output-dir scopes the cache file
        # under output_dir, so this only affects the long-corpus run.
        "--bust-seed-cache",
    ]
    if no_upload:
        cmd.append("--no-upload")
    print(f"--> {' '.join(cmd)}", flush=True)
    # check=True: fail loud on wrapper crash; per CLAUDE.md "Fail fast — never
    # hide failures". The orchestrator must NOT silently continue past a
    # half-generated corpus.
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ})


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
    args = parser.parse_args()

    print("=== Issue #408 long-form corpus generation (30 turns) ===", flush=True)
    print(f"  Output dir: {OUTPUT_DIR}", flush=True)
    print(f"  Drift sub-corpus:     {DRIFT_OUT}", flush=True)
    print(f"  Incontext sub-corpus: {INCONTEXT_OUT}", flush=True)
    print(f"  Aggregate:            {COMBINED_PATH}", flush=True)
    print(f"  Turns per conv:       {N_TURNS}", flush=True)
    print(f"  Rotation seed:        {ROTATION_SEED}", flush=True)
    print("", flush=True)

    print("Step 0: smoke check — wrappers expose --n-turns + --output-dir", flush=True)
    _smoke_check_cli_flags()
    print("  PASS", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStep 1: drift wrapper (therapy / philosophy / coding / writing)", flush=True)
    _run_wrapper(
        "scripts/issue_377_generate_drift_corpus.py",
        DRIFT_OUT,
        no_upload=args.no_upload,
    )

    print("\nStep 2: incontext wrapper (math / history / factual_qa / code_review)", flush=True)
    _run_wrapper(
        "scripts/issue_377_generate_incontext_corpus.py",
        INCONTEXT_OUT,
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
