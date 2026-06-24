#!/usr/bin/env python3
#
# Same Unicode-math suppression as scripts/eval_issue399.py — this
# script reads ※ from rows and prints it to logs.
"""Oversample C+ marker-emission rows in the #399 training JSONL.

Round-8 recipe adjustment (task #399, 2026-05-27). The round-7 ※
install fired 0/50 on cell A; the rescue is two coupled changes:

1. Strengthen training (configs/condition/c_issue399_marker_install.yaml:
   ``lr=1e-4, epochs=6``).
2. Oversample the 150 C+ rows by a factor of ``--duplicate-factor`` (default
   1 → 150 extra rows → 300 C+ / 2070 total). This forces the marker signal
   to dominate a larger fraction of each epoch's gradient.

The input JSONL is the canonical 1920-row training set produced by
``scripts/generate_issue376_marker_install.py --marker-token=※
--allow-single-token-marker``. The output is a new file (default
``train_oversampled.jsonl`` alongside the input) consumed by the
``c_issue399_marker_install`` Hydra condition's
``stages[0].dataset`` field. The script does NOT touch the original
file — re-runs are idempotent if the same flags are passed.

Detection rule: a row is a C+ marker-emission row iff its last
``"assistant"`` ``messages`` turn's ``content`` ends with the marker
literal. This matches the generator's
``assemble_training_data()`` contract at
``scripts/generate_issue376_marker_install.py:740`` (the C+ branch
appends ``f"{resp}\\n\\n{marker_text}"``).

Usage (on the pod, BEFORE Phase A.2 training)::

    cd /workspace/explore-persona-space
    uv run python scripts/build_issue399_oversampled_train.py \\
        --input data/issue376_marker_install_9ca040/train.jsonl \\
        --output data/issue376_marker_install_9ca040/train_oversampled.jsonl

The script prints a summary line so the dispatcher's log records the
recipe-deviation evidence the analyzer will need at write-up time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _is_marker_emission_row(row: dict, marker: str) -> bool:
    """True iff the last ``assistant`` turn's ``content`` ends with ``marker``.

    Matches the generator's C+ row format
    (``content = f"{resp}\\n\\n{marker_text}"``); will not false-positive on
    rows where ※ appears organically mid-text. Rows with no ``messages``
    field or no assistant turn return False rather than raising — this is a
    tolerant counter, not a schema validator (the canonical 1920-row file
    is already validated by the generator).
    """
    messages = row.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content") or ""
            return content.endswith(marker)
    return False


def oversample(
    input_path: Path,
    output_path: Path,
    duplicate_factor: int,
    marker: str,
) -> dict[str, int]:
    """Duplicate each marker-emission row ``duplicate_factor`` times.

    Returns a dict with ``original_rows / marker_rows_before /
    marker_rows_after / total_rows_after`` for the dispatcher log.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSONL missing at {input_path}; run "
            f"scripts/generate_issue376_marker_install.py --marker-token={marker} "
            f"--allow-single-token-marker first."
        )
    if duplicate_factor < 1:
        raise ValueError(
            f"--duplicate-factor must be >= 1 (got {duplicate_factor}); "
            f"the original rows are always kept verbatim."
        )

    rows: list[dict] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    original_rows = len(rows)
    marker_rows = [r for r in rows if _is_marker_emission_row(r, marker)]
    marker_rows_before = len(marker_rows)

    extras: list[dict] = []
    for _ in range(duplicate_factor):
        extras.extend(marker_rows)

    rows_out = rows + extras
    marker_rows_after = sum(1 for r in rows_out if _is_marker_emission_row(r, marker))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "original_rows": original_rows,
        "marker_rows_before": marker_rows_before,
        "marker_rows_after": marker_rows_after,
        "total_rows_after": len(rows_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the canonical 1920-row training JSONL (output of "
        "scripts/generate_issue376_marker_install.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the oversampled JSONL. Will overwrite if it exists.",
    )
    parser.add_argument(
        "--duplicate-factor",
        type=int,
        default=1,
        help="How many extra copies of each marker-emission row to append. "
        "Default 1 → 150 extras → 300 total C+ rows in a 2070-row corpus.",
    )
    parser.add_argument(
        "--marker",
        type=str,
        default="※",  # ※
        help="Marker literal to use for C+ row detection. Default ※ (U+203B).",
    )
    args = parser.parse_args()

    summary = oversample(
        input_path=args.input,
        output_path=args.output,
        duplicate_factor=args.duplicate_factor,
        marker=args.marker,
    )
    print(
        f"[oversample] input={args.input} output={args.output} "
        f"duplicate_factor={args.duplicate_factor} marker={args.marker!r}",
        flush=True,
    )
    print(
        f"[oversample] original_rows={summary['original_rows']} "
        f"marker_rows_before={summary['marker_rows_before']} "
        f"marker_rows_after={summary['marker_rows_after']} "
        f"total_rows_after={summary['total_rows_after']}",
        flush=True,
    )

    # Defensive sanity: with duplicate_factor=1 on the canonical 1920-row file
    # we expect 150 → 300. Surface unexpected counts so the dispatcher log
    # shows the deviation rather than the analyzer chasing it later.
    expected_extras = summary["marker_rows_before"] * args.duplicate_factor
    if summary["marker_rows_after"] != summary["marker_rows_before"] + expected_extras:
        raise RuntimeError(
            f"Oversample arithmetic mismatch: before={summary['marker_rows_before']} "
            f"extras_expected={expected_extras} after={summary['marker_rows_after']}. "
            f"Likely the marker-emission detector double-matched on extras "
            f"(should not happen — duplicates carry the same end-of-content marker)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
