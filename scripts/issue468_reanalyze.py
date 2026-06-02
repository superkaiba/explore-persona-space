#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, σ, ×, →) in scientific docstrings + logs.
"""Issue #468 Phase A — re-analysis of existing #463 cosine JSONs (CPU only).

Reads `eval_results/issue463/predictor_cossim{,_training}/<cell>_<flavor>.json`
and `eval_results/issue463/regression{,_training}_{NL,lit}.json` and reports:

* Per (flavor, probe-source, layer) over the 18 cells:
    - `spread_last = std(M_last_prompt_token_per_cell)`,
      `spread_resp = std(M_response_mean_per_cell)`,
    - `compression_ratio = spread_last / spread_resp`,
    - `saturation_fraction_resp` = fraction of cells with M_resp > 0.95,
    - `rank_divergence = 1 − Spearman(M_last_per_cell, M_resp_per_cell)`.
* Per-extraction-per-layer raw ρ(M, L) and partial ρ(M, L | log_tokens),
  pulled directly from existing regression-block dicts.
* L0-vs-deep-layer Pearson correlations (covariate diagnostic for the
  pre-registered V1 L0-partialled ρ headline; planner §11 A21).
* Length/persona-string-content partials: per (flavor, probe-source, layer)
  partial-Spearman ρ(M, L | Z) for Z ∈ {log_tokens (from #463), L0 cosine
  ("early-layer / persona-string-content covariate")}. The pre-block
  token-embedding-bag covariate is Phase B output, not available here;
  Phase C consumes it.

Pure-Python deterministic re-slice over already-published data: no GPU,
no remote IO, no model calls.

Usage::

    uv run python scripts/issue468_reanalyze.py
    uv run python scripts/issue468_reanalyze.py --layers 18 20 21 22 24 25 27
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue463_regress import (  # noqa: E402
    CELLS_18,
    load_outcome_per_cell,
    load_token_counts,
    partial_spearman,
    spearman_with_n,
)
from scipy import stats  # noqa: E402

logger = logging.getLogger("issue468_reanalyze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EVAL_DIR_463 = PROJECT_ROOT / "eval_results" / "issue463"
EVAL_DIR_468 = PROJECT_ROOT / "eval_results" / "issue468"
COSSIM_DIR_BY_SOURCE = {
    "betley": EVAL_DIR_463 / "predictor_cossim",
    "training": EVAL_DIR_463 / "predictor_cossim_training",
}
REGRESSION_PATH_BY_SOURCE_AND_FLAVOR = {
    ("betley", "NL"): EVAL_DIR_463 / "regression_NL.json",
    ("betley", "lit"): EVAL_DIR_463 / "regression_lit.json",
    ("training", "NL"): EVAL_DIR_463 / "regression_training_NL.json",
    ("training", "lit"): EVAL_DIR_463 / "regression_training_lit.json",
}

DEFAULT_LAYERS = [18, 20, 21, 22, 24, 25, 27]
EXTRACTION_POINTS = ("last_prompt_token", "response_mean")
SATURATION_THRESHOLD = 0.95


def load_cossim_per_cell(probe_source: str, flavor: str) -> dict[str, dict]:
    """Return ``{cell: full cossim JSON dict}`` for cells with files on disk."""
    cossim_dir = COSSIM_DIR_BY_SOURCE[probe_source]
    if not cossim_dir.exists():
        raise FileNotFoundError(
            f"Cossim dir missing: {cossim_dir}; expected pre-existing #463 output."
        )
    out: dict[str, dict] = {}
    for cell in CELLS_18:
        path = cossim_dir / f"{cell}_{flavor}.json"
        if not path.exists():
            continue
        with open(path) as f:
            out[cell] = json.load(f)
    return out


def extract_per_layer_M(
    per_cell: dict[str, dict], extraction_point: str, layer: int
) -> dict[str, float]:
    """Return ``{cell: cosine}`` for one extraction × layer across cells."""
    out: dict[str, float] = {}
    for cell, d in per_cell.items():
        ce = d.get("cos_by_extraction", {}).get(extraction_point, {})
        val = ce.get(str(layer))
        if val is None:
            continue
        out[cell] = float(val)
    return out


def divergence_table_per_layer(per_cell: dict[str, dict], layers: list[int]) -> dict[int, dict]:
    """Per layer compute spread / compression / saturation / rank-divergence
    between the two extractions (last_prompt_token vs response_mean).
    """
    out: dict[int, dict] = {}
    for layer in layers:
        last = extract_per_layer_M(per_cell, "last_prompt_token", layer)
        resp = extract_per_layer_M(per_cell, "response_mean", layer)
        common = sorted(set(last) & set(resp))
        if len(common) < 3:
            out[layer] = {
                "n_cells": len(common),
                "note": "insufficient_cells",
            }
            continue
        last_vals = np.array([last[c] for c in common], dtype=float)
        resp_vals = np.array([resp[c] for c in common], dtype=float)
        spread_last = float(last_vals.std(ddof=1))
        spread_resp = float(resp_vals.std(ddof=1))
        compression = spread_last / spread_resp if spread_resp > 0 else float("inf")
        sat_resp = float((resp_vals > SATURATION_THRESHOLD).mean())
        # rank_divergence = 1 − Spearman(last_per_cell, resp_per_cell)
        rho_pair = spearman_with_n(list(last_vals), list(resp_vals))
        rank_div = (1.0 - rho_pair["rho"]) if rho_pair["rho"] is not None else None
        out[layer] = {
            "n_cells": len(common),
            "cells": common,
            "spread_last": spread_last,
            "spread_resp": spread_resp,
            "compression_ratio_last_over_resp": compression,
            "saturation_fraction_resp_above_0p95": sat_resp,
            "rank_divergence_one_minus_spearman_last_vs_resp": rank_div,
            "spearman_last_vs_resp": rho_pair,
            "mean_last": float(last_vals.mean()),
            "mean_resp": float(resp_vals.mean()),
            "max_resp": float(resp_vals.max()),
            "min_resp": float(resp_vals.min()),
        }
    return out


def regress_extraction_per_layer(
    per_cell: dict[str, dict],
    extraction_point: str,
    layers: list[int],
    outcome: dict[str, dict],
    tokens: dict[str, float],
) -> dict[int, dict]:
    """Per-layer raw + partial Spearman ρ(M, L) for ONE extraction point.

    Mirrors `regress_one` in `issue463_regress.py` to keep stat shapes
    identical (and to verify against the published #463 regression blocks
    as a self-check).
    """
    out: dict[int, dict] = {}
    for layer in layers:
        M = extract_per_layer_M(per_cell, extraction_point, layer)
        common = sorted(set(M) & set(outcome) & set(tokens))
        if len(common) < 4:
            out[layer] = {"n": len(common), "note": "insufficient_cells"}
            continue
        M_vals = [M[c] for c in common]
        L_vals = [outcome[c]["mean_L"] for c in common]
        log_tokens = [math.log(max(tokens[c], 1.0)) for c in common]
        raw = spearman_with_n(M_vals, L_vals)
        partial = partial_spearman(M_vals, L_vals, log_tokens)
        out[layer] = {
            "n": len(common),
            "cells": common,
            "spearman_raw": raw,
            "spearman_partial_log_tokens": partial,
            "M_per_cell": {c: M[c] for c in common},
        }
    return out


def L0_vs_deep_layer_correlation(
    per_cell: dict[str, dict],
    extraction_point: str,
    deep_layers: list[int],
) -> dict[int, dict]:
    """For each deep layer, Pearson correlation between per-cell L0 cosine
    and per-cell deep-layer cosine. Diagnostic for A21 (pre-registered
    headline narrates direction of L0 partial).
    """
    L0 = extract_per_layer_M(per_cell, extraction_point, 0)
    out: dict[int, dict] = {}
    for layer in deep_layers:
        deep = extract_per_layer_M(per_cell, extraction_point, layer)
        common = sorted(set(L0) & set(deep))
        if len(common) < 4:
            out[layer] = {"n": len(common), "note": "insufficient_cells"}
            continue
        x = np.array([L0[c] for c in common], dtype=float)
        y = np.array([deep[c] for c in common], dtype=float)
        if x.std() == 0 or y.std() == 0:
            out[layer] = {"n": len(common), "note": "zero_variance"}
            continue
        pr = stats.pearsonr(x, y)
        out[layer] = {
            "n": len(common),
            "pearson_r": float(pr.statistic),
            "p": float(pr.pvalue),
        }
    return out


def L0_partial_per_layer(
    per_cell: dict[str, dict],
    extraction_point: str,
    layers: list[int],
    outcome: dict[str, dict],
) -> dict[int, dict]:
    """Partial Spearman ρ(M, L | L0_cos_per_cell) per layer for ONE extraction.

    The covariate is the L0 cosine of the SAME extraction point per cell.
    The L0-cosine variable is labeled the "early-layer /
    persona-string-content covariate" (NOT a clean lexical control) per
    planner §4.1.2 honest framing.
    """
    L0_lp = extract_per_layer_M(per_cell, "last_prompt_token", 0)
    out: dict[int, dict] = {}
    for layer in layers:
        if layer == 0:
            out[layer] = {"n": 0, "note": "skip_layer_0"}
            continue
        M = extract_per_layer_M(per_cell, extraction_point, layer)
        common = sorted(set(M) & set(outcome) & set(L0_lp))
        if len(common) < 4:
            out[layer] = {"n": len(common), "note": "insufficient_cells"}
            continue
        M_vals = [M[c] for c in common]
        L_vals = [outcome[c]["mean_L"] for c in common]
        L0_vals = [L0_lp[c] for c in common]
        partial = partial_spearman(M_vals, L_vals, L0_vals)
        out[layer] = {
            "n": len(common),
            "cells": common,
            "spearman_partial_L0_post_block_cos": partial,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Deep layers to report; full 0..27 sweep ingested from JSON.",
    )
    parser.add_argument(
        "--probe-sources",
        nargs="+",
        default=["training", "betley"],
        choices=["training", "betley"],
    )
    parser.add_argument(
        "--flavors",
        nargs="+",
        default=["NL", "lit"],
        choices=["NL", "lit"],
    )
    parser.add_argument(
        "--out-dir",
        default=str(EVAL_DIR_468 / "reanalysis"),
        help="Output directory for Phase A re-analysis JSON.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outcome = load_outcome_per_cell()
    tokens = load_token_counts()
    logger.info("Loaded outcome cells=%d, token-count cells=%d", len(outcome), len(tokens))

    results: dict = {"phase_A_reanalysis": {}}
    summary_path = out_dir / "issue468_reanalysis.json"

    for probe_source in args.probe_sources:
        for flavor in args.flavors:
            key = f"{probe_source}_{flavor}"
            try:
                per_cell = load_cossim_per_cell(probe_source, flavor)
            except FileNotFoundError as e:
                logger.warning("Skipping %s: %s", key, e)
                continue
            if not per_cell:
                logger.warning("Skipping %s: no cells loaded", key)
                continue

            logger.info(
                "Re-analyzing probe_source=%s flavor=%s (%d cells)",
                probe_source,
                flavor,
                len(per_cell),
            )

            divergence = divergence_table_per_layer(per_cell, args.layers)
            last_regress = regress_extraction_per_layer(
                per_cell, "last_prompt_token", args.layers, outcome, tokens
            )
            resp_regress = regress_extraction_per_layer(
                per_cell, "response_mean", args.layers, outcome, tokens
            )
            l0_corr_last = L0_vs_deep_layer_correlation(per_cell, "last_prompt_token", args.layers)
            l0_corr_resp = L0_vs_deep_layer_correlation(per_cell, "response_mean", args.layers)
            l0_partial_last = L0_partial_per_layer(
                per_cell, "last_prompt_token", args.layers, outcome
            )
            l0_partial_resp = L0_partial_per_layer(per_cell, "response_mean", args.layers, outcome)

            block = {
                "probe_source": probe_source,
                "flavor": flavor,
                "layers": args.layers,
                "n_cells_loaded": len(per_cell),
                "cells_loaded": sorted(per_cell.keys()),
                "divergence_by_layer": {str(k): v for k, v in divergence.items()},
                "regress_last_prompt_token_by_layer": {str(k): v for k, v in last_regress.items()},
                "regress_response_mean_by_layer": {str(k): v for k, v in resp_regress.items()},
                "L0_vs_deep_layer_pearson_last_prompt_token": {
                    str(k): v for k, v in l0_corr_last.items()
                },
                "L0_vs_deep_layer_pearson_response_mean": {
                    str(k): v for k, v in l0_corr_resp.items()
                },
                "L0_partial_last_prompt_token_by_layer": {
                    str(k): v for k, v in l0_partial_last.items()
                },
                "L0_partial_response_mean_by_layer": {
                    str(k): v for k, v in l0_partial_resp.items()
                },
            }
            results["phase_A_reanalysis"][key] = block

            # Per-(probe, flavor) sibling file so partial outputs survive a
            # later crash in another (probe, flavor) loop iteration.
            per_pair_path = out_dir / f"reanalysis_{probe_source}_{flavor}.json"
            with open(per_pair_path, "w") as f:
                json.dump(block, f, indent=2)
            logger.info("Wrote %s", per_pair_path.relative_to(PROJECT_ROOT))

    results["metadata"] = reproducibility_metadata(
        {"script": "issue468_reanalyze", "layers": args.layers}
    )
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", summary_path.relative_to(PROJECT_ROOT))

    # Print a tiny digest for quick eyeballing.
    print("\n=== Issue #468 Phase A re-analysis digest ===")
    for key, block in results["phase_A_reanalysis"].items():
        print(f"\n--- {key} (n_cells={block['n_cells_loaded']}) ---")
        for layer_str, div in sorted(
            block["divergence_by_layer"].items(), key=lambda kv: int(kv[0])
        ):
            if "note" in div:
                continue
            print(
                f"L{layer_str:>2}  "
                f"std_last={div['spread_last']:.4f}  "
                f"std_resp={div['spread_resp']:.4f}  "
                f"compress={div['compression_ratio_last_over_resp']:.3f}  "
                f"sat_resp>0.95={div['saturation_fraction_resp_above_0p95']:.3f}  "
                f"rank_div={div['rank_divergence_one_minus_spearman_last_vs_resp']:.3f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
