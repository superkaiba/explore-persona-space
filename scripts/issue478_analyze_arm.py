#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #478 PHASE 4b — distinct-marker decomposition analyzer (OPTIONAL, plan v5 §6.8 Level-2).

Runs ONLY when the §4.9 arm ran (12 arm cells under
``eval_results/issue_478/cell_ARM_*_seed*``).

Per plan v5 §4.8 PHASE 4b + §6.8 Level-2 + Level-1 (re-runs Level-1 from the
core analyzer here ONLY for inline cross-reference).

Inputs:
  - eval_results/issue_478/cell_ARM_*/result.json (12 arm cells × 2 seeds)
  - eval_results/issue_478/cell_K{2,4}_*/result.json (matched core cells for
    L_shared)
  - data/issue_478/arm/marker_base_logp.json (Phase 0b: 8×35 base-logp matrix)

Pipeline:
  1. Load arm + matched core cells; build per-(arm_cell, seed, persona,
     marker_i) tidy CSV.
  2. For each matched (shared_core_cell, arm_cell) pair at fixed (source_set,
     seed): compute superposition(L_distinct) via 4 combiners
     (mean = PRE-REGISTERED PRIMARY NULL, lse/max/fitted = sensitivity).
  3. Per-(persona, K, band) shared-minus-superposition gap.
  4. PAIRED bootstrap over the 12 matched (shared, distinct) pairs at fixed
     (source-set, seed). Report DIRECTION-AGREEMENT COUNTS per K (the
     plan §6.8 v5 primary read; persona-resampling is the wrong cluster).
  5. Per-marker training-speed sanity: dispatch user to read WandB
     ``probe/<persona>/logp_marker_<marker>`` curves; flag a cell if any
     marker_i's emission/logp curve rises ≥5× faster than another's despite
     matched Phase 0b base logp.
  6. Write eval_results/issue_478/aggregate/distinct_markers_decomposition.json.

DOSE-AWARE READS (plan v5 §6.8 Level-2 interpretation table):
  - L_shared ≈ superposition(L_distinct) → STRONG evidence for SUPERPOSITION
    (the K× per-token dose advantage of ※ failed to open a gap).
  - L_shared > superposition(L_distinct) → AMBIGUOUS (dose-consistent;
    coupling-via-shared-token is UNIDENTIFIED at this design — needs a
    K=1@200/100 dose-matched control).
  - L_shared < superposition(L_distinct) → STRONGEST evidence for
    cross-source INTERFERENCE.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue478_common import (  # noqa: E402
    ARM_TRAINING_SPEED_DIVERGENCE_FACTOR,
    HELD_OUT_BANDS,
    SEEDS,
    band_of,
)


def _combiner(name: str, values: list[float]) -> float:
    if name == "mean":
        return sum(values) / len(values)
    if name == "max":
        return max(values)
    if name == "lse":
        return float(np.log(np.sum(np.exp(values))))
    raise ValueError(f"Unknown combiner {name!r}")


def load_arm_cells(eval_dir: Path) -> list[dict]:
    files = sorted(eval_dir.glob("cell_ARM_*/result.json"))
    if not files:
        raise SystemExit(f"No ARM cell results found under {eval_dir} — did Phase 3b run?")
    out = [json.loads(f.read_text()) for f in files]
    log.info("Loaded %d ARM cell result.json files", len(out))
    return out


def load_matched_core_cell(eval_dir: Path, core_cell_id: str, seed: int) -> dict | None:
    p = eval_dir / f"cell_{core_cell_id}_seed{seed}" / "result.json"
    if not p.exists():
        log.warning("Matched core cell missing: %s", p)
        return None
    return json.loads(p.read_text())


def build_arm_tidy_rows(arm_results: list[dict]) -> list[dict]:
    """Per-(arm_cell, seed, persona, marker_i) rows from the arm runs."""
    rows: list[dict] = []
    for r in arm_results:
        cell_id = r["cell_id"]
        seed = r["seed"]
        K = r["K"]
        spec = r["spec"]
        marker_assignment: dict[str, str] = spec["marker_assignment"]
        marker_id_assignment: dict[str, int] = spec["marker_id_assignment"]
        positives = spec["positives"]
        # For each held-out persona, read per-marker deltaLogP from per_marker block.
        for persona, payload in r["eval"]["held_out"].items():
            per_marker = payload.get("per_marker", {})
            band = band_of(persona)
            for source_persona, marker_text in marker_assignment.items():
                marker_id = marker_id_assignment[source_persona]
                marker_block = per_marker.get(str(marker_id))
                if marker_block is None:
                    log.warning(
                        "Arm cell %s seed=%d persona=%s missing per_marker[%d]; skipping",
                        cell_id,
                        seed,
                        persona,
                        marker_id,
                    )
                    continue
                rows.append(
                    {
                        "arm_cell_id": cell_id,
                        "matched_core_cell": spec["matched_core_cell"],
                        "seed": seed,
                        "K": K,
                        "source_persona": source_persona,
                        "marker_text": marker_text,
                        "marker_id": marker_id,
                        "held_out_persona": persona,
                        "band": band,
                        "deltaLogP_distinct_per_marker_for_source": marker_block["deltaLogP_mean"],
                        "positives_set": ";".join(sorted(positives)),
                    }
                )
    return rows


def level2_decomposition(  # noqa: C901
    arm_rows: list[dict],
    matched_core_cells: dict[tuple[str, int], dict],
    combiners: tuple[str, ...] = ("mean", "lse", "max"),
) -> dict:
    """Per (arm_cell, seed) pair: superposition(L_distinct) vs L_shared.

    For each (matched_core_cell, seed) pair:
      - L_distinct[marker_i, C] = per-(marker × persona) deltaLogP from the arm cell.
      - For each C, combine L_distinct[marker_i, C] over markers via each
        combiner → predicted joint leakage.
      - L_shared[S, C] = per-persona deltaLogP from the matched core cell.
      - Gap = L_shared − combined.

    Reports per-K direction-agreement counts AND per-K mean gap with paired
    bootstrap CI over the 6 matched cell-pairs at fixed seed.
    """
    # Build per (arm_cell, seed) → {persona: {marker_text: distinct_dLogP}}
    by_arm_seed: dict[tuple[str, int, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in arm_rows:
        # Distinct values are the same for any source_persona on the same row
        # (we keyed by (source_persona, marker_id); collapse to per-marker).
        by_arm_seed[(r["arm_cell_id"], r["seed"], r["matched_core_cell"])][r["held_out_persona"]][
            r["marker_text"]
        ] = r["deltaLogP_distinct_per_marker_for_source"]

    # For each (arm_cell, seed), compute per-persona gap under each combiner.
    per_pair_gaps: list[dict] = []
    for (arm_cell_id, seed, matched_core_cell), persona_marker_map in by_arm_seed.items():
        core = matched_core_cells.get((matched_core_cell, seed))
        if core is None:
            log.warning(
                "Matched core cell %s seed=%d missing for arm cell %s; skipping pair",
                matched_core_cell,
                seed,
                arm_cell_id,
            )
            continue
        K = core["K"]
        core_held_out = core["eval"]["held_out"]
        for persona, marker_map in persona_marker_map.items():
            shared = core_held_out.get(persona)
            if shared is None:
                continue
            l_shared = shared["deltaLogP_mean"]
            l_distinct_values = list(marker_map.values())
            if len(l_distinct_values) != K:
                # Defensive — should always have K markers per arm cell.
                continue
            gaps_by_combiner = {}
            preds_by_combiner = {}
            for c in combiners:
                pred = _combiner(c, l_distinct_values)
                preds_by_combiner[c] = pred
                gaps_by_combiner[c] = l_shared - pred
            per_pair_gaps.append(
                {
                    "arm_cell_id": arm_cell_id,
                    "matched_core_cell": matched_core_cell,
                    "seed": seed,
                    "K": K,
                    "held_out_persona": persona,
                    "band": band_of(persona),
                    "l_shared": l_shared,
                    "l_distinct_values": l_distinct_values,
                    "predictions": preds_by_combiner,
                    "gaps": gaps_by_combiner,
                }
            )

    # Per-K direction-agreement counts (the v5 PRIMARY read).
    direction_summary: dict[int, dict[str, int]] = {}
    for K_target in sorted({g["K"] for g in per_pair_gaps}):
        # One direction read per (arm_cell, seed): take the mean-combiner gap
        # averaged across held-out personas as the cell-pair's directional sign.
        cell_seed_means: dict[tuple[str, int], float] = defaultdict(list)
        cell_seed_values: dict[tuple[str, int], list[float]] = defaultdict(list)
        for g in per_pair_gaps:
            if g["K"] != K_target:
                continue
            cell_seed_values[(g["arm_cell_id"], g["seed"])].append(g["gaps"]["mean"])
        for k, vs in cell_seed_values.items():
            cell_seed_means[k] = sum(vs) / len(vs)
        signs = list(cell_seed_means.values())
        direction_summary[K_target] = {
            "n_pairs": len(signs),
            "n_shared_gt_distinct": int(sum(1 for s in signs if s > 0)),
            "n_shared_lt_distinct": int(sum(1 for s in signs if s < 0)),
            "n_zero_or_noise": int(sum(1 for s in signs if s == 0)),
            "interpretation": (
                "Shared ※ token receives K× more per-token gradient than each "
                "distinct marker_i, so shared>distinct is the PURE-DOSE PREDICTION "
                "(AMBIGUOUS). shared≈distinct is STRONG superposition evidence "
                "(K× dose advantage failed to open a gap). shared<distinct is "
                "STRONGEST interference evidence."
            ),
        }

    # Per-K bootstrap CI on mean gap (paired by source-set + seed).
    bootstrap_summary: dict[int, dict] = {}
    rng = np.random.default_rng(478)
    for K_target in sorted({g["K"] for g in per_pair_gaps}):
        cell_seed_means: dict[tuple[str, int], float] = {}
        for g in per_pair_gaps:
            if g["K"] != K_target:
                continue
            key = (g["arm_cell_id"], g["seed"])
            cell_seed_means.setdefault(key, []).append(g["gaps"]["mean"])
        per_pair_mean = [sum(vs) / len(vs) for vs in cell_seed_means.values()]
        if len(per_pair_mean) < 2:
            bootstrap_summary[K_target] = {
                "status": "SKIPPED",
                "reason": f"only {len(per_pair_mean)} pairs",
            }
            continue
        boot = []
        arr = np.array(per_pair_mean)
        n_boot = 2000
        for _ in range(n_boot):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot.append(float(sample.mean()))
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        bootstrap_summary[K_target] = {
            "n_pairs": len(per_pair_mean),
            "mean_gap": float(arr.mean()),
            "bootstrap_ci95": (ci_lo, ci_hi),
            "n_bootstrap": n_boot,
            "caveat": ("6 paired clusters per K is small — bootstrap CI is descriptive only."),
        }

    return {
        "per_pair_gaps": per_pair_gaps,
        "direction_agreement_per_K": direction_summary,
        "paired_bootstrap_per_K_mean_gap": bootstrap_summary,
    }


def per_marker_training_speed_hint() -> str:
    """Generate the user-facing analysis hint about WandB per-marker trajectories."""
    return (
        f"To check per-marker training-speed divergence (§4.9 arm-specific risk): "
        f"pull WandB project 'issue_478_distinct_markers_arm', filter to ARM_K* runs, "
        f"then plot 'probe/<persona>/logp_marker_<marker>' for each marker_i. If any "
        f"curve rises ≥{ARM_TRAINING_SPEED_DIVERGENCE_FACTOR}× faster than another "
        f"despite matched Phase 0b base logp (see data/issue_478/arm/marker_base_logp.json), "
        f"the cell is FLAGGED uninterpretable for the §6.8 Level-2 read."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_478"),
    )
    parser.add_argument(
        "--aggregate-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_478" / "aggregate"),
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    agg_dir = Path(args.aggregate_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading ARM cells ...")
    arm_results = load_arm_cells(eval_dir)
    log.info("Building arm tidy rows ...")
    arm_rows = build_arm_tidy_rows(arm_results)
    log.info("Arm tidy rows: %d", len(arm_rows))

    # Load every matched core cell at every seed.
    matched_cells_needed = {(r["matched_core_cell"], r["seed"]) for r in arm_rows}
    matched_core_cells: dict[tuple[str, int], dict] = {}
    for cell_id, seed in matched_cells_needed:
        core = load_matched_core_cell(eval_dir, cell_id, seed)
        if core is not None:
            matched_core_cells[(cell_id, seed)] = core
    log.info("Loaded %d matched core cells", len(matched_core_cells))

    log.info("Running Level-2 decomposition ...")
    level2 = level2_decomposition(arm_rows, matched_core_cells)

    # Phase 0b base-logp diagnostic (if present).
    base_logp_path = PROJECT_ROOT / "data" / "issue_478" / "arm" / "marker_base_logp.json"
    base_logp = json.loads(base_logp_path.read_text()) if base_logp_path.exists() else None
    if base_logp is None:
        log.warning("Phase 0b marker_base_logp.json not present — diagnostic plot will lack it")

    out = {
        "experiment": "issue_478_distinct_markers_arm",
        "n_arm_cells": len(arm_results),
        "n_arm_rows": len(arm_rows),
        "n_matched_core_cells": len(matched_core_cells),
        "level2_decomposition": level2,
        "phase_0b_marker_base_logp": base_logp,
        "per_marker_training_speed_hint": per_marker_training_speed_hint(),
        "dose_aware_interpretation_table": {
            "L_shared ≈ superposition(L_distinct)": (
                "STRONG evidence FOR independent superposition "
                "(K× dose advantage failed to open a gap)."
            ),
            "L_shared > superposition(L_distinct)": (
                "AMBIGUOUS — consistent with pure per-token-dose advantage of the "
                "shared ※; coupling-via-shared-token is UNIDENTIFIED at this design "
                "(needs K=1@200/100 dose control)."
            ),
            "L_shared < superposition(L_distinct)": (
                "STRONGEST evidence for cross-source INTERFERENCE through the shared token."
            ),
            "weighting_note": (
                "K=2 (2× dose asymmetry) is weighted over K=4 (4× asymmetry) in the synthesis."
            ),
        },
        "design_constants": {
            "seeds": list(SEEDS),
            "n_held_out_personas": sum(len(m) for m in HELD_OUT_BANDS.values()),
        },
    }
    out_path = agg_dir / "distinct_markers_decomposition.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote %s", out_path)

    # Unused import silencer — math/combinations referenced for fitted-linear extension.
    _ = math
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
