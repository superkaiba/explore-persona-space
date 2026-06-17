#!/usr/bin/env python3
"""Fetch + identity-verify the pinned parent battery for issue #654's dummy arm.

Issue #654 round-5 CRITICAL fix (code-review reconcile v5, concern
``dummy-arm-pinned-battery-not-fetched``).

The length-matched-dummy-query control computes a dummy-vs-real companion gap by
joining the dummy banks to the parent run's PINNED real-arm banks on a STABLE id
(``(context_id, real_query_id)``; ``scripts/issue654_analyze.py``). The control
(content held; only query length/position changes) is valid ONLY if the dummy
arm reads its contexts + per-context length targets from the SAME frozen battery
the pinned real/context-only banks were extracted from.

On a fresh GCE/RunPod pod ``data/issue654/`` is gitignored and EMPTY, so a local
rebuild would stream contexts live (UltraChat / WildChat) and DRIFT the context
strings while still joining the pinned banks by stable id — silently computing
the gap across DIFFERENT contexts. This module instead downloads the parent's
frozen ``inputs/battery.json`` at the pinned revision and fails loud unless its
``context_id`` set EXACTLY matches the pinned cached ``context_only/*.pt``
companion-bank basenames (the banks the dummy arm reuses).

CPU-only, tokenizer-free, no model load — the dispatcher's first CPU step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


class PinnedBatteryMismatchError(RuntimeError):
    """Raised when the fetched battery's contexts disagree with the cached banks."""


def battery_context_ids(payload: dict) -> set[str]:
    """The set of ``context_id`` values across the battery's pairs."""
    return {p["context_id"] for p in payload["pairs"]}


def cached_bank_context_ids(repo_files: list[str], prefix: str) -> set[str]:
    """Context ids of the pinned cached ``context_only/*.pt`` companion banks.

    Each bank is stored at ``<prefix>/analysis_tensors/context_only/<context_id>.pt``,
    so the basename without the ``.pt`` suffix IS the context id.
    """
    needle = f"{prefix}/analysis_tensors/context_only/"
    return {
        f[len(needle) :].rsplit("/", 1)[-1][:-3]  # strip needle prefix + ".pt"
        for f in repo_files
        if f.startswith(needle) and f.endswith(".pt")
    }


def verify_context_identity(battery_cids: set[str], bank_cids: set[str], rev: str) -> None:
    """Fail loud unless the battery contexts EXACTLY match the cached banks.

    A mismatch means the dummy queries would be built against contexts the reused
    pinned banks were never extracted from — the single-variable control is void.
    """
    if not bank_cids:
        raise PinnedBatteryMismatchError(
            f"no cached context_only banks found at rev {rev} (expected the parent "
            f"run's analysis_tensors/context_only/*.pt)"
        )
    if battery_cids != bank_cids:
        only_battery = sorted(battery_cids - bank_cids)
        only_banks = sorted(bank_cids - battery_cids)
        raise PinnedBatteryMismatchError(
            "PINNED-BATTERY / CACHED-BANK CONTEXT MISMATCH (single-variable control "
            f"voided): {len(battery_cids)} battery context_ids vs {len(bank_cids)} "
            f"cached context_only banks at rev {rev}. "
            f"only in battery (first 5): {only_battery[:5]}; "
            f"only in banks (first 5): {only_banks[:5]}"
        )


def fetch_and_verify_pinned_battery(
    repo: str,
    prefix: str,
    dest: Path,
    rev: str,
) -> set[str]:
    """Download the pinned ``inputs/battery.json`` and verify context identity.

    Writes the fetched battery to ``dest`` and returns its ``context_id`` set.
    Raises :class:`PinnedBatteryMismatchError` if the battery's contexts do not
    exactly match the pinned cached ``context_only/*.pt`` banks at ``rev``.
    """
    # Imported lazily so the pure helpers above are testable without the network
    # dependency and so a monkeypatched download is exercised in tests.
    from huggingface_hub import hf_hub_download, list_repo_files

    battery_path_in_repo = f"{prefix}/inputs/battery.json"
    local = hf_hub_download(repo, battery_path_in_repo, repo_type="dataset", revision=rev)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(local).read_bytes())

    payload = json.loads(dest.read_text())
    battery_cids = battery_context_ids(payload)

    repo_files = list_repo_files(repo, repo_type="dataset", revision=rev)
    bank_cids = cached_bank_context_ids(repo_files, prefix)

    verify_context_identity(battery_cids, bank_cids, rev)
    print(
        f"fetched pinned battery {battery_path_in_repo} -> {dest} "
        f"({len(battery_cids)} contexts) and verified context_id set matches "
        f"{len(bank_cids)} cached context_only banks at rev {rev}"
    )
    return battery_cids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--rev", required=True)
    args = parser.parse_args()
    fetch_and_verify_pinned_battery(args.repo, args.prefix, args.dest, args.rev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
