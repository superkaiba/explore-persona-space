#!/usr/bin/env python3
"""Issue #405 PHASE 0.5 — cell-specs builder (§4.6 of plan v2).

Produces ``data/issue_405/cell_specs.json`` — 25 cells total:

  * 21 CORE cells (K=1: 8 cells, K=2: 6 cells, K=4: 6 cells, K=8: 1 cell)
    enter the headline mixed-effects regression.
  * 1 K4_ABLNEG ablation cell (separate track — sensitivity overlay only).
  * 3 K1_DOSE50 dose-control cells (separate track — head-to-head with main
    K=1@400 and main K=8@50; isolates per-persona dose from K).

Each spec carries: ``cell_id``, ``K``, ``positives``, ``negatives``,
``held_out``, ``rows_per_positive``, ``rows_per_negative``, ``total_rows``,
``arm`` (``core`` / ``ablation`` / ``dose_control``), ``track`` (``CORE`` /
``K4_ABLNEG`` / ``K1_DOSE50``).

CPU-only. Deterministic via ``numpy.random.default_rng(SUBSET_RNG_SEED)``.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue405_common import (  # noqa: E402
    ABLATION_HELD_OUT,
    ABLATION_NEGATIVES,
    ABLATION_POSITIVES,
    CORE_NEG_ROWS_PER_PERSONA,
    CORE_ROWS_PER_CELL,
    CORE_TOTAL_POSITIVE_ROWS,
    DOSE50_PERSONAS,
    DOSE50_ROWS_PER_POSITIVE,
    HELD_OUT,
    K_VALUES,
    NEGATIVES_FIXED,
    POOL,
    SUBSET_RNG_SEED,
    SUBSETS_PER_K,
)


def build_specs(rng_seed: int = SUBSET_RNG_SEED) -> list[dict]:
    """Build the 21 CORE cell specs.

    K=1: all 8 positives, one per cell.
    K=2: 6 random pairs from C(8,2)=28 (seed-fixed).
    K=4: 6 random 4-subsets from C(8,4)=70 (seed-fixed).
    K=8: the single full pool subset.

    Rows: ``rows_per_positive = 400 // K``; ``rows_per_negative = 100``;
    total = 800.
    """
    rng = np.random.default_rng(rng_seed)
    specs: list[dict] = []
    cell_id = 0
    for K in K_VALUES:
        all_subsets = list(combinations(POOL, K))
        n = SUBSETS_PER_K[K]
        if n is None:
            chosen = all_subsets
        else:
            idx = rng.choice(len(all_subsets), size=n, replace=False)
            chosen = [all_subsets[i] for i in idx]
        for sub in chosen:
            rows_per_positive = CORE_TOTAL_POSITIVE_ROWS // K
            specs.append(
                {
                    "cell_id": f"K{K}_c{cell_id:02d}",
                    "K": K,
                    "positives": list(sub),
                    "negatives": list(NEGATIVES_FIXED),
                    "held_out": list(HELD_OUT),
                    "rows_per_positive": rows_per_positive,
                    "rows_per_negative": CORE_NEG_ROWS_PER_PERSONA,
                    "total_rows": CORE_ROWS_PER_CELL,
                    "arm": "core",
                    "track": "CORE",
                }
            )
            cell_id += 1
    return specs


def build_ablation_specs() -> list[dict]:
    """One K=4 ABLNEG cell — same K, different negatives + shrunk held_out.

    Tests whether the K effect is sensitive to which negatives. The 4
    promoted-to-negative personas drop out of held_out for THIS cell only.
    Run as overlay, NOT in headline regression (per FIX C).
    """
    rows_per_positive = CORE_TOTAL_POSITIVE_ROWS // len(ABLATION_POSITIVES)  # 100
    return [
        {
            "cell_id": "K4_ABLNEG",
            "K": len(ABLATION_POSITIVES),
            "positives": list(ABLATION_POSITIVES),
            "negatives": list(ABLATION_NEGATIVES),
            "held_out": list(ABLATION_HELD_OUT),
            "rows_per_positive": rows_per_positive,
            "rows_per_negative": CORE_NEG_ROWS_PER_PERSONA,
            "total_rows": CORE_ROWS_PER_CELL,
            "arm": "ablation",
            "track": "K4_ABLNEG",
        }
    ]


def build_dose_control_specs() -> list[dict]:
    """FIX A2 — Dose-matched K=1 control arm at 50 rows/persona.

    Three K=1 cells matched to main K=8 cell's per-persona dose
    (50 rows/positive). Spans the distance range — paramedic (inside-cluster)
    / villain (mid) / poet (far/outlier).

    Negative count stays at 400 (matched to core), so the 1:1 ratio
    temporarily breaks — that's the documented cost (§4.6, §5).
    """
    return [
        {
            "cell_id": f"K1_DOSE50_{p}",
            "K": 1,
            "positives": [p],
            "negatives": list(NEGATIVES_FIXED),
            "held_out": list(HELD_OUT),
            "rows_per_positive": DOSE50_ROWS_PER_POSITIVE,  # 50
            "rows_per_negative": CORE_NEG_ROWS_PER_PERSONA,  # 100
            "total_rows": DOSE50_ROWS_PER_POSITIVE + 4 * CORE_NEG_ROWS_PER_PERSONA,  # 450
            "arm": "dose_control",
            "track": "K1_DOSE50",
        }
        for p in DOSE50_PERSONAS
    ]


def main() -> int:
    out_dir = PROJECT_ROOT / "data" / "issue_405"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cell_specs.json"

    core = build_specs()
    abl = build_ablation_specs()
    dose = build_dose_control_specs()

    # Cell-id uniqueness assert (catches a copy-paste regression).
    all_ids = [s["cell_id"] for s in core + abl + dose]
    if len(all_ids) != len(set(all_ids)):
        from collections import Counter

        dupes = [k for k, v in Counter(all_ids).items() if v > 1]
        raise RuntimeError(f"Duplicate cell_id values: {dupes!r}")

    # Sanity: row-count totals match plan §4.3 + §4.6.
    for s in core:
        assert s["rows_per_positive"] * s["K"] == CORE_TOTAL_POSITIVE_ROWS, s
        assert s["rows_per_negative"] * 4 == CORE_TOTAL_POSITIVE_ROWS, s
        assert s["total_rows"] == CORE_ROWS_PER_CELL, s

    specs = core + abl + dose
    out_path.write_text(json.dumps(specs, indent=2))
    log.info(
        "Wrote %d cell specs (%d core + %d ablation + %d dose-control) → %s",
        len(specs),
        len(core),
        len(abl),
        len(dose),
        out_path,
    )
    # Counts per K for the core sweep
    from collections import Counter

    per_K = Counter(s["K"] for s in core)
    log.info("CORE per-K cell counts: %s", dict(sorted(per_K.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
