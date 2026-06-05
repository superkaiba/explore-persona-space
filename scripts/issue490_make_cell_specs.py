#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #490 PHASE 0.5 — cell-specs builder (plan v1 §4.5).

Reads ``data/issue_490/source_pairs.json`` (from Phase 0) and emits the
8 pairs × 5 conditions = 40 cell specs at
``data/issue_490/cell_specs.json``.

Each spec carries:
  - cell_id              e.g. ``c490_pair0_shared_2D``
  - pair_id              ``pair0``..``pair7``
  - condition            ``shared_2D``/``pooled_2D_A``/``pooled_2D_B``/
                          ``single_D_A``/``single_D_B``
  - A, B                 the source-pair (alphabetical)
  - positives            list of persona names (1 or 2 entries)
  - rows_per_positive    200 for SHARED-2D; 400 for POOLED-SINGLE-2D; 200
                          for SINGLE-D
  - negatives            NEGATIVES_FIXED (4 personas)
  - rows_per_negative    100 each → 400 total → 1:1 pos:neg
  - held_out             FULL HELD_OUT_35 (eval surface; analyzer slices)
  - on_axis              the pair's on-axis intermediate-C subpanel (≥5)
  - off_axis             the pair's distance-matched off-axis subpanel (≥5)
  - total_rows           positives + 400 (negatives constant per cell)

CPU-only; deterministic given the same Phase 0 output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue490_common import (  # noqa: E402
    CONDITION_POOLED_2D_A,
    CONDITION_POOLED_2D_B,
    CONDITION_SHARED_2D,
    CONDITION_SINGLE_D_A,
    CONDITION_SINGLE_D_B,
    CONDITIONS,
    D_PER_SOURCE,
    HELD_OUT_35,
    NEGATIVES_FIXED,
    TWO_D,
)


def _positives_and_rows_for_condition(condition: str, A: str, B: str) -> tuple[list[str], int]:
    """Return (positives, rows_per_positive) for one (condition, A, B)."""
    if condition == CONDITION_SHARED_2D:
        return [A, B], D_PER_SOURCE  # 2 sources × 200 each → 400 total
    if condition == CONDITION_POOLED_2D_A:
        return [A], TWO_D  # 1 source × 400
    if condition == CONDITION_POOLED_2D_B:
        return [B], TWO_D  # 1 source × 400
    if condition == CONDITION_SINGLE_D_A:
        return [A], D_PER_SOURCE  # 1 source × 200
    if condition == CONDITION_SINGLE_D_B:
        return [B], D_PER_SOURCE  # 1 source × 200
    raise ValueError(f"Unknown condition {condition!r}")


def build_cell_specs(pairs: list[dict]) -> list[dict]:
    specs: list[dict] = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        A = pair["A"]
        B = pair["B"]
        on_axis = pair["on_axis"]
        off_axis = pair["off_axis"]
        for condition in CONDITIONS:
            positives, rows_per_positive = _positives_and_rows_for_condition(condition, A, B)
            total_positive_rows = rows_per_positive * len(positives)
            # Scale neg rows-per-persona to keep 1:1 pos:neg ratio per
            # contrastive-negatives rule. SHARED-2D / POOLED-SINGLE-2D get
            # 100×4=400 negs (1:1 against 400 pos); SINGLE-D gets 50×4=200
            # negs (1:1 against 200 pos). All 5 conditions keep the SAME 4
            # negative personas; only rows-per-persona varies, evenly split.
            # Plan §4.1's "Total rows per cell = 800" only holds for the
            # 400-positive cells; for SINGLE-D the 1:1 contrastive ratio
            # wins over the row-total parity with #478 (deviation logged in
            # the implementer marker).
            rows_per_negative = total_positive_rows // len(NEGATIVES_FIXED)
            total_neg_rows = rows_per_negative * len(NEGATIVES_FIXED)
            total_rows = total_positive_rows + total_neg_rows
            cell_id = f"c490_{pair_id}_{condition}"
            specs.append(
                {
                    "cell_id": cell_id,
                    "pair_id": pair_id,
                    "condition": condition,
                    "A": A,
                    "B": B,
                    "positives": positives,
                    "negatives": list(NEGATIVES_FIXED),
                    "rows_per_positive": rows_per_positive,
                    "rows_per_negative": rows_per_negative,
                    "total_rows": total_rows,
                    # FULL HELD_OUT_35 for eval; analyzer slices on/off-axis
                    # downstream so we never lose the option to re-slice.
                    "held_out": list(HELD_OUT_35),
                    "on_axis": list(on_axis),
                    "off_axis": list(off_axis),
                    "origin": pair["origin"],
                    "matched_cell_id": pair.get("matched_cell_id"),
                }
            )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pairs",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_490" / "source_pairs.json"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_490" / "cell_specs.json"),
    )
    args = parser.parse_args()

    src = Path(args.source_pairs)
    if not src.exists():
        raise SystemExit(
            f"source_pairs.json missing: {src}. Run "
            f"scripts/issue490_validate_design.py first (Phase 0)."
        )
    payload = json.loads(src.read_text())
    pairs = payload["pairs"]
    log.info("Loaded %d source-pairs from %s", len(pairs), src)

    specs = build_cell_specs(pairs)
    expected_n = len(pairs) * len(CONDITIONS)
    if len(specs) != expected_n:
        raise RuntimeError(
            f"Cell spec count mismatch: built {len(specs)}, expected "
            f"{len(pairs)} pairs × {len(CONDITIONS)} conditions = {expected_n}"
        )

    # Cell-id uniqueness assert.
    all_ids = [s["cell_id"] for s in specs]
    if len(all_ids) != len(set(all_ids)):
        dupes = [k for k, v in Counter(all_ids).items() if v > 1]
        raise RuntimeError(f"Duplicate cell_id values: {dupes!r}")

    # Sanity: total positive rows per cell sum to the design intent.
    for s in specs:
        total_pos = s["rows_per_positive"] * len(s["positives"])
        if s["condition"] == CONDITION_SHARED_2D and total_pos != TWO_D:
            raise RuntimeError(f"SHARED_2D bad positive total: {s!r}")
        if s["condition"] in (CONDITION_POOLED_2D_A, CONDITION_POOLED_2D_B) and total_pos != TWO_D:
            raise RuntimeError(f"POOLED_2D_* bad positive total: {s!r}")
        if (
            s["condition"] in (CONDITION_SINGLE_D_A, CONDITION_SINGLE_D_B)
            and total_pos != D_PER_SOURCE
        ):
            raise RuntimeError(f"SINGLE_D_* bad positive total: {s!r}")
        total_neg = s["rows_per_negative"] * len(s["negatives"])
        if s["total_rows"] != total_pos + total_neg:
            raise RuntimeError(f"Bad total_rows: {s!r}")
        # 1:1 pos:neg ratio (plan §4.7 + contrastive-negatives rule):
        # per-condition neg-scaling keeps total_pos == total_neg.
        if total_pos != total_neg:
            raise RuntimeError(
                f"pos:neg ratio NOT 1:1 for {s['cell_id']!r}: pos={total_pos} neg={total_neg}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(specs, indent=2))
    log.info("Wrote %d cell specs → %s", len(specs), out_path)
    per_condition = Counter(s["condition"] for s in specs)
    log.info("Per-condition cell counts: %s", dict(sorted(per_condition.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
