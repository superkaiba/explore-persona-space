#!/usr/bin/env python3
"""Dispatcher for issue #377 drift + in-context conversation generation.

Thin wrapper around ``issue_377_generate_drift_corpus.py`` and
``issue_377_generate_incontext_corpus.py``. The plan's reproducibility
card (``tasks/running/377/plans/v1.md`` §10) and reproduce commands name
THIS script with a ``--corpus {drift,incontext,both}`` flag — keeping
that contract working is what this dispatcher exists for.

The underlying generators are independent scripts (each can be invoked
directly) and stay as the canonical implementations; this wrapper just
runs one or both in sequence. The drift corpus is generated first when
``--corpus both`` is requested, because the in-context script's plan
§4.2 sanity check (2) ±10% length-match cross-check needs
``data/issue377_drift/drift_summary.json`` to already exist.

Usage::

    # Generate both corpora end-to-end (canonical plan command).
    uv run python scripts/generate_issue377_drift_conversations.py --corpus both

    # Only the drift corpus.
    uv run python scripts/generate_issue377_drift_conversations.py --corpus drift

    # Only the in-context corpus (will fail loud if the drift summary is
    # missing — pass ``--allow-missing-drift-summary`` if you intentionally
    # want this order).
    uv run python scripts/generate_issue377_drift_conversations.py --corpus incontext

    # Local-only dry run (skip HF Hub upload).
    uv run python scripts/generate_issue377_drift_conversations.py --corpus both --no-upload
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR: Path = Path(__file__).parent
DRIFT_SCRIPT: Path = SCRIPTS_DIR / "issue_377_generate_drift_corpus.py"
INCONTEXT_SCRIPT: Path = SCRIPTS_DIR / "issue_377_generate_incontext_corpus.py"


def _run(script: Path, extra_args: list[str]) -> int:
    """Run a generator subprocess and forward its exit code.

    We subprocess instead of importing-and-calling so each generator
    keeps its own ``argparse.Namespace`` semantics and the dispatcher
    stays stateless. Both generators print to stdout/stderr directly
    so the user sees the same output as a direct invocation.
    """
    cmd = ["uv", "run", "python", str(script), *extra_args]
    print(f"\n  >>> {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(
            f"\n  !!! {script.name} exited {proc.returncode}; halting dispatcher",
            flush=True,
        )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        choices=("drift", "incontext", "both"),
        default="both",
        help=(
            "Which corpus to generate. 'both' runs drift first, then "
            "in-context, so the in-context script's plan §4.2 length-match "
            "cross-check can find drift_summary.json. Default: both."
        ),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Forwarded to each generator: skip HF Hub upload (local dry run).",
    )
    parser.add_argument(
        "--allow-missing-drift-summary",
        action="store_true",
        help=(
            "Forwarded to the in-context generator: tolerate a missing "
            "drift_summary.json instead of failing loudly. Only useful with "
            "--corpus incontext when the drift corpus has not been generated "
            "yet and you intentionally want the in-context corpus first."
        ),
    )
    parser.add_argument(
        "--bust-seed-cache",
        action="store_true",
        help=(
            "Forwarded to each generator: delete the cached persona+topic "
            "seed JSON before re-seeding. Use between rounds when DomainSpec "
            "wording has changed; without this the script silently reuses "
            "stale personas from the prior round."
        ),
    )
    args = parser.parse_args()

    if not DRIFT_SCRIPT.exists():
        print(f"  Missing generator: {DRIFT_SCRIPT}", file=sys.stderr)
        return 2
    if not INCONTEXT_SCRIPT.exists():
        print(f"  Missing generator: {INCONTEXT_SCRIPT}", file=sys.stderr)
        return 2

    drift_args: list[str] = []
    incontext_args: list[str] = []
    if args.no_upload:
        drift_args.append("--no-upload")
        incontext_args.append("--no-upload")
    if args.allow_missing_drift_summary:
        incontext_args.append("--allow-missing-drift-summary")
    if args.bust_seed_cache:
        drift_args.append("--bust-seed-cache")
        incontext_args.append("--bust-seed-cache")

    print(
        f"=== Issue #377 corpus dispatcher ===\n"
        f"  corpus={args.corpus}, no-upload={args.no_upload}, "
        f"allow-missing-drift-summary={args.allow_missing_drift_summary}, "
        f"bust-seed-cache={args.bust_seed_cache}\n",
        flush=True,
    )

    if args.corpus in ("drift", "both"):
        rc = _run(DRIFT_SCRIPT, drift_args)
        if rc != 0:
            return rc

    if args.corpus in ("incontext", "both"):
        rc = _run(INCONTEXT_SCRIPT, incontext_args)
        if rc != 0:
            return rc

    print("\n=== Dispatcher done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
