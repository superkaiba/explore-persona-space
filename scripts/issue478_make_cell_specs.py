#!/usr/bin/env python3
"""Issue #478 PHASE 0.5 — cell-specs builder (plan v5 §4.5).

Produces ``data/issue_478/cell_specs.json`` — 32 CORE cells + (optionally)
6 ARM cells (only emitted into the JSON if ``--include-arm`` is set; the
arm runner reads them ONLY when the dispatcher is run with ``--arm``).

CORE cells: K=1: 8, K=2: 8, K=4: 8, K=8: 8 (uniform 8 subsets per K, vs
#405's 1/6/6/1). Per-K subset uniqueness via numpy.random.default_rng(478).

Each spec carries: ``cell_id``, ``K``, ``positives``, ``negatives``,
``held_out``, ``rows_per_positive``, ``rows_per_negative``, ``total_rows``,
``arm`` (``core`` / ``arm_distinct``), ``track`` (``CORE`` / ``ARM``).

ARM cells (3 K=2 + 3 K=4) match the source-sets of K2_c08/c09/c10 and
K4_c16/c17/c18 from the core's seeded draw, so the shared-vs-distinct
comparison is paired by source set (plan v5 §4.9.2).

CPU-only. Deterministic via ``numpy.random.default_rng(SUBSET_RNG_SEED)``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue478_common import (  # noqa: E402
    ARM_K2_MATCHED_CELLS,
    ARM_K4_MATCHED_CELLS,
    ARM_MARKERS,
    CORE_NEG_ROWS_PER_PERSONA,
    CORE_ROWS_PER_CELL,
    CORE_TOTAL_POSITIVE_ROWS,
    HELD_OUT_35,
    K_VALUES,
    NEGATIVES_FIXED,
)
from issue478_validate_design import build_subsets  # noqa: E402


def build_core_specs() -> list[dict]:
    """Build the 32 CORE cell specs from the deterministic POOL_16 subsets.

    Cell ids run K1_c00..c07 / K2_c08..c15 / K4_c16..c23 / K8_c24..c31 (the
    cell_id offsets are stable so the ARM matched cells (K2_c08..c10,
    K4_c16..c18) line up with the core's seeded draw).
    """
    subsets = build_subsets()
    specs: list[dict] = []
    cell_id_counter = 0
    for K in K_VALUES:
        for sub in subsets[K]:
            rows_per_positive = CORE_TOTAL_POSITIVE_ROWS // K
            specs.append(
                {
                    "cell_id": f"K{K}_c{cell_id_counter:02d}",
                    "K": K,
                    "positives": list(sub),
                    "negatives": list(NEGATIVES_FIXED),
                    "held_out": list(HELD_OUT_35),
                    "rows_per_positive": rows_per_positive,
                    "rows_per_negative": CORE_NEG_ROWS_PER_PERSONA,
                    "total_rows": CORE_ROWS_PER_CELL,
                    "arm": "core",
                    "track": "CORE",
                }
            )
            cell_id_counter += 1
    return specs


def build_arm_specs(core_specs: list[dict]) -> list[dict]:
    """Build the 6 ARM cells (plan v5 §4.9.2), matched by source-set to core.

    The ARM matched cells (K2_c08/c09/c10, K4_c16/c17/c18) lift their source
    sets DIRECTLY from the existing core specs so the matched-pair comparison
    at Level-2 (plan §6.8) is paired by source set — variance comes from the
    shared-vs-distinct manipulation, not from a re-draw.

    Each arm cell trains K distinct single-token markers (one per source
    persona). The marker assignment (positive_persona → marker_text) is
    deterministic: positive_personas[i] gets ARM_MARKERS[i].
    """
    core_by_id = {s["cell_id"]: s for s in core_specs}
    arm_specs: list[dict] = []
    for arm_idx, core_id in enumerate(ARM_K2_MATCHED_CELLS + ARM_K4_MATCHED_CELLS):
        if core_id not in core_by_id:
            raise RuntimeError(
                f"Arm matched core cell {core_id!r} not in core spec set — "
                f"check ARM_K2_MATCHED_CELLS / ARM_K4_MATCHED_CELLS vs the seeded draw."
            )
        core = core_by_id[core_id]
        K = core["K"]
        positives = list(core["positives"])
        if len(positives) > len(ARM_MARKERS):
            raise RuntimeError(
                f"Arm cell {core_id!r} has K={K} > {len(ARM_MARKERS)} markers available — "
                f"extend ARM_MARKERS in _issue478_common."
            )
        marker_assignment = {persona: ARM_MARKERS[i][0] for i, persona in enumerate(positives)}
        marker_id_assignment = {persona: ARM_MARKERS[i][1] for i, persona in enumerate(positives)}
        arm_specs.append(
            {
                "cell_id": f"ARM_K{K}_a{arm_idx}",
                "matched_core_cell": core_id,
                "K": K,
                "positives": positives,
                "marker_assignment": marker_assignment,
                "marker_id_assignment": marker_id_assignment,
                "negatives": list(NEGATIVES_FIXED),
                "held_out": list(HELD_OUT_35),
                "rows_per_positive": core["rows_per_positive"],
                "rows_per_negative": CORE_NEG_ROWS_PER_PERSONA,
                "total_rows": CORE_ROWS_PER_CELL,
                "arm": "arm_distinct",
                "track": "ARM",
            }
        )
    return arm_specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-arm",
        action="store_true",
        help="Emit the 6 ARM cells (plan §4.9) into the spec JSON. The dispatcher "
        "ignores arm cells unless --arm is passed.",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data" / "issue_478"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cell_specs.json"

    core = build_core_specs()
    arm: list[dict] = []
    if args.include_arm:
        arm = build_arm_specs(core)

    # Cell-id uniqueness assert (catches a copy-paste regression).
    all_ids = [s["cell_id"] for s in core + arm]
    if len(all_ids) != len(set(all_ids)):
        dupes = [k for k, v in Counter(all_ids).items() if v > 1]
        raise RuntimeError(f"Duplicate cell_id values: {dupes!r}")

    # Sanity: row-count totals match plan v5 §4.5.
    for s in core:
        if s["rows_per_positive"] * s["K"] != CORE_TOTAL_POSITIVE_ROWS:
            raise RuntimeError(f"Bad row totals for core {s['cell_id']}: {s}")
        if s["rows_per_negative"] * 4 != CORE_TOTAL_POSITIVE_ROWS:
            raise RuntimeError(f"Bad negative totals for core {s['cell_id']}: {s}")
        if s["total_rows"] != CORE_ROWS_PER_CELL:
            raise RuntimeError(f"Bad total_rows for core {s['cell_id']}: {s}")

    specs = core + arm
    out_path.write_text(json.dumps(specs, indent=2))
    log.info(
        "Wrote %d cell specs (%d CORE + %d ARM) → %s",
        len(specs),
        len(core),
        len(arm),
        out_path,
    )
    per_K = Counter(s["K"] for s in core)
    log.info("CORE per-K cell counts: %s", dict(sorted(per_K.items())))
    if arm:
        arm_per_K = Counter(s["K"] for s in arm)
        log.info("ARM per-K cell counts: %s", dict(sorted(arm_per_K.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
