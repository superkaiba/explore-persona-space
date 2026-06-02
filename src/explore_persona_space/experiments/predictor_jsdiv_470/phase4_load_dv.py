"""Phase 4 — DV loading + assembly.

Reads #411's frozen DV from ``analyze_summary.json`` and merges it with the
predictor outputs from Phases 2 and 3 into one columnar table indexed by
(source, bystander) cell. Drops source-self (A14) so each source contributes
exactly 23 bystander cells -> 138 cells total for the 6 #411 sources.

Output: ``eval_results/issue_470/predictor_comparison.json`` with one row per
cell carrying every predictor we have AND the bystander-base-rate baselines
from #411's ``base_panel_rates.json``.

Pure CPU; runs after Phase 3 completes.

Usage::

    uv run python -m explore_persona_space.experiments.predictor_jsdiv_470.phase4_load_dv
"""

from __future__ import annotations

import argparse
import logging
import sys

from explore_persona_space.experiments.predictor_jsdiv_470 import SOURCE_PERSONAS_411
from explore_persona_space.experiments.predictor_jsdiv_470.common import (
    DEFAULT_LAYERS,
    HEADLINE_LAYER,
    ISSUE_411_ANALYZE_SUMMARY,
    ISSUE_411_BASE_PANEL_RATES,
    PHASE2_DIR,
    PHASE3_DIR,
    PHASE4_PATH,
    read_json,
    reproducibility_metadata,
    write_json,
)

logger = logging.getLogger("predictor_jsdiv_470.phase4")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_411_dv() -> dict[tuple[str, str], dict]:
    """Read #411's per-source per-panel delta + cosine baseline.

    Returns ``{(source, bystander): {delta, cosine_l20, trained_rate, base_rate}}``.
    Source-self is dropped per plan A14.
    """
    if not ISSUE_411_ANALYZE_SUMMARY.exists():
        raise RuntimeError(
            f"#411 DV file not found at {ISSUE_411_ANALYZE_SUMMARY}. "
            f"This experiment is a predictor re-analysis on #411's outputs; "
            f"the issue-411 worktree must be present and readable."
        )
    summary = read_json(ISSUE_411_ANALYZE_SUMMARY)
    per_source = summary["per_source"]
    out: dict[tuple[str, str], dict] = {}
    for source, blob in per_source.items():
        per_panel_delta = blob["per_panel_delta"]
        per_panel_cosine = blob.get("per_panel_cosine_to_source", {})
        per_panel_trained = blob.get("per_panel_trained_rate", {})
        per_panel_base = blob.get("per_panel_base_rate", {})
        if len(per_panel_delta) != 24:
            raise RuntimeError(
                f"Source {source!r} per_panel_delta has {len(per_panel_delta)} "
                f"entries, expected 24."
            )
        for bystander, delta in per_panel_delta.items():
            if bystander == source:
                continue  # drop source-self (A14)
            out[(source, bystander)] = {
                "delta": float(delta),
                "cosine_l20": (
                    float(per_panel_cosine[bystander]) if bystander in per_panel_cosine else None
                ),
                "trained_rate": (
                    float(per_panel_trained[bystander]) if bystander in per_panel_trained else None
                ),
                "base_rate_per_panel": (
                    float(per_panel_base[bystander]) if bystander in per_panel_base else None
                ),
            }
    logger.info(
        "Loaded #411 DV: %d cells (= %d sources x 23 bystanders)",
        len(out),
        len(per_source),
    )
    return out


def load_base_panel_rates() -> dict[str, float]:
    """Read each persona's intrinsic base sycophancy rate (#411's base_panel_rates.json)."""
    if not ISSUE_411_BASE_PANEL_RATES.exists():
        raise RuntimeError(
            f"#411 base_panel_rates.json not found at {ISSUE_411_BASE_PANEL_RATES}; "
            f"the bystander-base-rate baseline (§4) requires it."
        )
    blob = read_json(ISSUE_411_BASE_PANEL_RATES)
    return {k: float(v) for k, v in blob["panel_rates"].items()}


def load_phase2_cosine_pairs() -> dict[int, dict[str, dict[str, float]]]:
    """Read recipe-(b) cosine matrices per layer.

    Returns ``{layer: {src: {bys: cos}}}``.
    """
    out: dict[int, dict[str, dict[str, float]]] = {}
    for layer in DEFAULT_LAYERS:
        path = PHASE2_DIR / f"layer_{layer}.json"
        if not path.exists():
            logger.warning("Phase 2 output missing for layer %d: %s", layer, path)
            continue
        blob = read_json(path)
        personas = blob["personas"]
        matrix = blob["cosine_matrix"]
        per_pair: dict[str, dict[str, float]] = {}
        for i, src in enumerate(personas):
            per_pair[src] = {}
            for j, bys in enumerate(personas):
                per_pair[src][bys] = float(matrix[i][j])
        out[layer] = per_pair
    return out


def load_phase3_pairs() -> dict[tuple[str, str], dict]:
    """Read per-cell RB JS + KL outputs."""
    out: dict[tuple[str, str], dict] = {}
    for path in PHASE3_DIR.glob("*__*.json"):
        blob = read_json(path)
        out[(blob["source"], blob["bystander"])] = blob
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_PERSONAS_411),
        help="Sources to include (default: all 6 #411 sources).",
    )
    args = parser.parse_args()

    dv = load_411_dv()
    base_rates = load_base_panel_rates()
    cossim_b = load_phase2_cosine_pairs()
    js_kl = load_phase3_pairs()

    cells = []
    sources = list(args.sources)
    for (src, bys), dv_row in dv.items():
        if src not in sources:
            continue
        row: dict = {
            "source": src,
            "bystander": bys,
            # DV (frozen from #411)
            "delta": dv_row["delta"],
            "cosine_l20_baseline": dv_row["cosine_l20"],
            "trained_rate_411": dv_row["trained_rate"],
            # Trivial baselines (§4)
            "bystander_base_rate": base_rates.get(bys),
            "source_base_rate": base_rates.get(src),
            "base_rate_diff_neg_abs": (
                -abs(base_rates[src] - base_rates[bys])
                if src in base_rates and bys in base_rates
                else None
            ),
        }
        # Phase 2 cossim recipe (b) per layer.
        for layer, mat in cossim_b.items():
            v = mat.get(src, {}).get(bys)
            row[f"cosine_response_l{layer}"] = v
        if HEADLINE_LAYER in cossim_b:
            row["cosine_response_headline"] = cossim_b[HEADLINE_LAYER].get(src, {}).get(bys)

        # Phase 3 RB JS + KL.
        p3 = js_kl.get((src, bys))
        if p3:
            row["JS_sym_nats"] = p3["JS_sym_nats"]
            row["JS_from_source_nats"] = p3["JS_from_source_nats"]
            row["JS_from_bystander_nats"] = p3["JS_from_bystander_nats"]
            row["M_js"] = p3["M_js"]
            row["KL_src_to_bys_nats"] = p3["KL_src_to_bys_nats"]
            row["KL_bys_to_src_nats"] = p3["KL_bys_to_src_nats"]
            row["KL_sym_nats"] = p3["KL_sym_nats"]
        else:
            row["JS_sym_nats"] = None
            row["M_js"] = None
            row["KL_src_to_bys_nats"] = None
            row["KL_bys_to_src_nats"] = None
        cells.append(row)

    payload = {
        "n_cells": len(cells),
        "sources": sources,
        "cells": cells,
        "metadata": reproducibility_metadata({"script": "predictor_jsdiv_470.phase4_load_dv"}),
    }
    write_json(PHASE4_PATH, payload)
    logger.info("Wrote %s (%d cells)", PHASE4_PATH, len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
