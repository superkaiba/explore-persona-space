#!/usr/bin/env python3
"""Issue #640 — diagonal-source selectivity scoring (plan v6 §6, off-pod CPU).

Runs on the VM (CPU only) after the pod's diagonal Δsource JSONs are committed.
The question this settles: is the postfix-KV patch a SELECTIVE leakage fix
(clears off-diagonal leakage, preserves on-target diagonal behavior) or a BLUNT
revert-toward-base (dampens both equally)?

It joins, PER ROW (the stable join key — the diagonal and leakage cells use
DIFFERENT columns, so ``row|column`` is NOT comparable across the two files):

- diagonal Δsource  = this run's ``diagonal_source_seed{seed}.json`` cells
  (trained-no-patch diagonal rate - postfix-patched diagonal rate);
- off-diagonal Δleakage = v3's committed ``patch_cells_postfix_seed{seed}.json``
  cells (trained - patched on the highest-|L| off-diagonal column);
- selectivity gap = Δleakage - Δsource per (row, seed).

Headline verdict (plan v6 §6, restricted to the strong judged-rate rows with
on-target range; ``reversed_fact`` is a FLOOR cell and ``marker`` is log-prob-
scale — both carried but EXCLUDED from the count):

- SELECTIVE   : on risky_financial + extreme_sports the selectivity gap is large
  and positive at seed-0 (Δsource ≤ ~½ Δleakage) AND Δleakage > Δsource holds at
  BOTH seeds.
- BLUNT-REVERT: Δsource ≈ Δleakage on those cells (|gap| ≲ 0.1) with consistent
  sign across seeds.
- MIXED/UNDERPOWERED: otherwise (gaps inconsistent across seeds).

The marker diagonal log-prob (5.7738 / 9.4478 nats, read from #545's
cell_metadata.json) is carried as a descriptive null/parity reference only — NOT
a Δsource cell, NOT in the verdict count. The diagonal-mode backend-parity
precheck record (``backend_parity_diagonal_seed0.json``) is folded into a
``provenance`` block.

Writes ``eval_results/issue_640/selectivity_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue640_diagonal_score")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# The 7 judged-rate diagonal rows (plan v6 §4.2). marker is NOT here (log-prob).
PHASE2_ROWS_JUDGED_RATE: tuple[str, ...] = (
    "bad_medical",
    "risky_financial",
    "extreme_sports",
    "taught_fact",
    "reversed_fact",
    "compliment_writing",
    "wrong_claim_agreement",
)

# Rows carried-but-excluded from the headline selective-vs-blunt count (§3/§4.1):
# reversed_fact reads 0.0 on-target (floor cell, no dynamic range); marker is
# log-prob-scale (handled as a separate null reference, never a Δsource cell).
FLOOR_ROWS: frozenset[str] = frozenset({"reversed_fact"})

# The two strongly-installed reckless cells that decide the headline (§6).
PRIMARY_RECKLESS_ROWS: tuple[str, ...] = ("risky_financial", "extreme_sports")

# Blunt-revert band: |selectivity gap| within judge noise of 0 (§3 / §6).
BLUNT_GAP_TOL = 0.1


def _i640_root() -> Path:
    return PROJECT_ROOT / "eval_results" / "issue_640"


def _i545_root() -> Path:
    return PROJECT_ROOT / "eval_results" / "issue_545"


def _load_diagonal_cells(seeds: list[int]) -> dict[int, dict[str, dict]]:
    """Per seed: {row: detail dict} from diagonal_source_seed{seed}.json.

    Keys the per-row detail by the bare ``row`` (the cell key is ``row|column``;
    the column differs from the leakage file, so the join is on the row). Fails
    loud if no diagonal file is present (nothing to score).
    """
    root = _i640_root()
    per_seed: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        path = root / f"diagonal_source_seed{seed}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        by_row: dict[str, dict] = {}
        for cell_key, detail in data["detail"].items():
            row = detail["row"]
            if "delta_source" not in detail:
                raise KeyError(
                    f"{path.name} cell {cell_key} lacks 'delta_source' — this is not a "
                    "diagonal-target JSON (did --target diagonal produce it?)."
                )
            by_row[row] = detail
        per_seed[seed] = by_row
    if not per_seed:
        raise FileNotFoundError(
            f"no diagonal_source_seed*.json under {root} for seeds {seeds} — the diagonal "
            "Δsource sweep produced nothing; the selectivity comparison cannot run. "
            "Run: issue640_postfix_carrier.py --target diagonal ... first."
        )
    return per_seed


def _load_leakage_cells(seeds: list[int]) -> dict[int, dict[str, dict]]:
    """Per seed: {row: detail dict} from v3's patch_cells_postfix_seed{seed}.json.

    Keys by the bare ``row`` (same join contract as the diagonal cells). Fails
    loud if a requested seed's v3 file is missing (the paired comparison has no
    right-hand side).
    """
    root = _i640_root()
    per_seed: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        path = root / f"patch_cells_postfix_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — v3's off-diagonal Δleakage baseline is required for the "
                "paired selectivity comparison (must-keep input, plan §6 statistical-input "
                "existence). Cannot join the diagonal Δsource against it."
            )
        data = json.loads(path.read_text())
        by_row: dict[str, dict] = {}
        for cell_key, detail in data["detail"].items():
            row = detail["row"]
            if "delta_leakage" not in detail:
                raise KeyError(f"{path.name} cell {cell_key} lacks 'delta_leakage'.")
            by_row[row] = detail
        per_seed[seed] = by_row
    return per_seed


def _marker_diagonal_reference() -> dict[str, float]:
    """Marker diagonal log-prob (nats) per seed from #545's cell_metadata (§4.3).

    Descriptive null/parity reference ONLY. Fails loud if cell_metadata.json or
    the marker cell is missing.
    """
    path = _i545_root() / "cell_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"#545 cell_metadata.json missing under {path}.")
    cells = json.loads(path.read_text())["cells"]
    out: dict[str, float] = {}
    for seed in (0, 137):
        key = f"marker_primary_seed{seed}"
        if key not in cells:
            raise KeyError(f"{key} not in #545 cell_metadata.json — marker reference unavailable.")
        out[str(seed)] = float(cells[key]["diagonal_level"])
    return out


def _parity_provenance() -> dict | None:
    """Fold the diagonal-mode backend-parity precheck record into provenance.

    Returns None (with a logged warning) if the precheck JSON is absent — the
    selectivity table is still produced, but the provenance block flags the gap
    rather than fabricating a value.
    """
    path = _i640_root() / "backend_parity_diagonal_seed0.json"
    if not path.exists():
        logger.warning(
            "[phase=provenance] backend_parity_diagonal_seed0.json absent under %s — the "
            "diagonal-mode parity precheck record is not available to fold into provenance.",
            _i640_root(),
        )
        return None
    return json.loads(path.read_text())


def build_selectivity_table(seeds: list[int]) -> dict:
    """Join diagonal Δsource against v3 off-diagonal Δleakage per (row, seed).

    Returns the per-row joined table + per-row cross-seed sign consistency on
    Δsource + the §6 headline verdict block. Marker is carried as a separate
    null reference; reversed_fact is carried with a floor flag and excluded from
    the headline.
    """
    diag_by_seed = _load_diagonal_cells(seeds)
    leak_by_seed = _load_leakage_cells(seeds)
    seeds_present = sorted(set(diag_by_seed))

    per_row: dict[str, dict] = {}
    for row in PHASE2_ROWS_JUDGED_RATE:
        row_entry: dict = {
            "is_floor": row in FLOOR_ROWS,
            "in_headline": row not in FLOOR_ROWS,
            "seeds": {},
        }
        d_sources: dict[int, float] = {}
        for seed in seeds_present:
            diag = diag_by_seed.get(seed, {}).get(row)
            leak = leak_by_seed.get(seed, {}).get(row)
            if diag is None:
                continue
            d_source = float(diag["delta_source"])
            d_sources[seed] = d_source
            entry = {
                "diagonal_column": diag["column"],
                "trained_rate_diag": diag["trained_rate"],
                "patched_rate_diag": diag["patched_rate"],
                "delta_source": d_source,
                "n_probes_diag": diag["n_probes"],
            }
            if leak is not None:
                d_leak = float(leak["delta_leakage"])
                entry["leakage_column"] = leak["column"]
                entry["trained_rate_leak"] = leak["trained_rate"]
                entry["patched_rate_leak"] = leak["patched_rate"]
                entry["delta_leakage"] = d_leak
                entry["selectivity_gap"] = d_leak - d_source
                entry["n_probes_leak"] = leak["n_probes"]
            else:
                entry["leakage_column"] = None
                entry["delta_leakage"] = None
                entry["selectivity_gap"] = None
            row_entry["seeds"][str(seed)] = entry
        # Cross-seed sign consistency on Δsource (matches #640's H2 read).
        if len(d_sources) >= 2:
            signs = {(v > 0) for v in d_sources.values()}
            row_entry["delta_source_sign_consistent"] = len(signs) == 1
        else:
            row_entry["delta_source_sign_consistent"] = None
        per_row[row] = row_entry

    headline = _headline_verdict(per_row, seeds_present)
    return {
        "seeds_present": seeds_present,
        "headline_rows": [r for r in PHASE2_ROWS_JUDGED_RATE if r not in FLOOR_ROWS],
        "floor_rows": sorted(FLOOR_ROWS),
        "per_row": per_row,
        "headline": headline,
    }


def _headline_verdict(per_row: dict, seeds_present: list[int]) -> dict:
    """The §6 selective / blunt-revert / mixed-underpowered partition.

    Decided on the two primary reckless cells (risky_financial, extreme_sports)
    at seed-0, with cross-seed ordering consistency. Pure descriptive partition
    (small n; no CI claim).
    """
    verdict = {
        "primary_cells": list(PRIMARY_RECKLESS_ROWS),
        "blunt_gap_tolerance": BLUNT_GAP_TOL,
        "per_cell": {},
    }
    if 0 not in seeds_present:
        verdict["verdict"] = "mixed-underpowered"
        verdict["reason"] = "seed-0 absent — the primary selectivity read cannot run."
        return verdict

    selective_flags: list[bool] = []
    blunt_flags: list[bool] = []
    for row in PRIMARY_RECKLESS_ROWS:
        seeds_block = per_row.get(row, {}).get("seeds", {})
        s0 = seeds_block.get("0")
        if s0 is None or s0.get("delta_leakage") is None:
            verdict["per_cell"][row] = {"status": "absent"}
            selective_flags.append(False)
            blunt_flags.append(False)
            continue
        d_source0 = s0["delta_source"]
        d_leak0 = s0["delta_leakage"]
        gap0 = s0["selectivity_gap"]
        # Cross-seed ordering: Δleakage > Δsource at BOTH seeds (selective) /
        # consistent sign of the gap (blunt).
        ordering_both = True
        gap_signs = []
        for sk, block in seeds_block.items():  # noqa: B007
            if block.get("delta_leakage") is None:
                continue
            ordering_both = ordering_both and (block["delta_leakage"] > block["delta_source"])
            gap_signs.append(block["selectivity_gap"] > 0)
        gap_sign_consistent = len(set(gap_signs)) == 1 if len(gap_signs) >= 2 else None
        # SELECTIVE on this cell: Δsource ≤ ~½ Δleakage (gap ≥ ½ Δleakage),
        # AND Δleakage > Δsource at every present seed.
        is_selective = (d_leak0 > 0) and (d_source0 <= 0.5 * d_leak0) and ordering_both
        # BLUNT on this cell: |gap| ≲ 0.1 at seed-0 with consistent gap sign.
        is_blunt = (abs(gap0) <= BLUNT_GAP_TOL) and (gap_sign_consistent is not False)
        selective_flags.append(is_selective)
        blunt_flags.append(is_blunt)
        verdict["per_cell"][row] = {
            "delta_source_seed0": d_source0,
            "delta_leakage_seed0": d_leak0,
            "selectivity_gap_seed0": gap0,
            "ordering_leak_gt_source_all_seeds": ordering_both,
            "gap_sign_consistent_across_seeds": gap_sign_consistent,
            "is_selective": is_selective,
            "is_blunt": is_blunt,
        }

    if all(selective_flags):
        verdict["verdict"] = "selective"
        verdict["reason"] = (
            "Both reckless cells show Δsource ≤ ½ Δleakage at seed-0 with Δleakage > Δsource "
            "consistent across seeds — the patch clears off-diagonal leakage while preserving "
            "on-target diagonal behavior."
        )
    elif all(blunt_flags):
        verdict["verdict"] = "blunt-revert"
        verdict["reason"] = (
            "Both reckless cells show |selectivity gap| ≲ 0.1 with consistent sign — the patch "
            "dampens on-target behavior about as much as it cut leakage; soften #640's "
            "'model-specific intervention' framing."
        )
    else:
        verdict["verdict"] = "mixed-underpowered"
        verdict["reason"] = (
            "The two reckless cells do not jointly satisfy either the selective or the blunt "
            "criterion (or differ across seeds) — reported descriptively, no overclaim."
        )
    return verdict


def score_diagonal(*, seeds: list[int], smoke: bool = False) -> Path:
    """Phase-3 (diagonal) entrypoint: build + write selectivity_comparison.json."""
    table = build_selectivity_table(seeds)
    marker_ref = _marker_diagonal_reference()
    provenance = {
        "backend_parity_precheck": _parity_provenance(),
        "marker_diagonal_reference_nats": marker_ref,
        "marker_note": (
            "marker diagonal is log-prob-scale (marker_slot_stats), read from #545 "
            "cell_metadata.json; descriptive null/parity reference only — NOT a Δsource cell "
            "and NOT in the headline verdict count (§4.3)."
        ),
        "inputs": {
            "diagonal": "eval_results/issue_640/diagonal_source_seed{seed}.json",
            "leakage_baseline_v3": "eval_results/issue_640/patch_cells_postfix_seed{seed}.json",
        },
    }
    out = {
        "smoke": smoke,
        "selectivity": table,
        "provenance": provenance,
    }
    out_path = _i640_root() / "selectivity_comparison.json"
    out_path.write_text(json.dumps(out, indent=1))
    logger.info(
        "[phase=selectivity] verdict=%s (seeds=%s) -> %s",
        table["headline"]["verdict"],
        table["seeds_present"],
        out_path,
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #640 diagonal-source selectivity scoring")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 137])
    parser.add_argument(
        "--probe-cap",
        type=int,
        default=32,
        help="Accepted for smoke/parity with the driver CLI; scoring reads whatever the "
        "diagonal/leakage JSONs contain (it does not re-generate).",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logger.info("[phase=start] issue640 diagonal selectivity scoring seeds=%s", args.seeds)
    score_diagonal(seeds=args.seeds, smoke=args.smoke)
    logger.info("[phase=done] issue640 diagonal selectivity scoring complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
