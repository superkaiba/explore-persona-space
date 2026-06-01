#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥) in scientific docstrings + logs.
"""Issue #458 — regression: predict EM amount from base-model signals.

Across the 18 cells run under turner_em + max_steps=375, regress

    L  ~  M_cosine  +  log(assistant_tokens)
    L  ~  M_js      +  log(assistant_tokens)

where L is the post-SFT Betley-main-8 broad-misalignment rate (per-cell
mean across seeds 0 + 137), M_cosine is the layer-21 cos-sim from
``scripts/issue404_predictor_cossim.py`` (#458 keeps the predictor
unchanged), and M_js is the token-level JS predictor from
``scripts/issue458_predictor_jsdiv.py``. Per-cell assistant-token total
comes from ``eval_results/issue458/token_counts.json``.

Primary statistics per predictor:

* Spearman ρ(M, L) across the 18 cells (raw association).
* Partial Spearman ρ(M, L | log(assistant_tokens)) (controlling for
  training-volume confound).
* Per-cell residual L_resid = L − fitted(L | log(tokens)) printed
  alongside the partial-ρ so the cells driving the signal are visible.

Output: ``eval_results/issue458/regression.json`` + printed comparison
table on stdout. Mirrors ``scripts/issue404_regress.py``'s schema so a
future cross-experiment aggregator can consume both.

Usage::

    uv run python scripts/issue458_regress.py
    uv run python scripts/issue458_regress.py --flavor NL
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
from scipy import stats  # noqa: E402

logger = logging.getLogger("issue458_regress")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue458"
COSSIM_DIR = PROJECT_ROOT / "eval_results" / "issue_404" / "predictor_cossim"
JSDIV_DIR = EVAL_DIR / "predictor_jsdiv"
OUTCOME_DIR = EVAL_DIR / "outcome"
TOKEN_COUNTS_PATH = EVAL_DIR / "token_counts.json"
OUTPUT_PATH = EVAL_DIR / "regression.json"

# The 18 #458 cells, in the same spectrum order as the planning doc.
CELLS_18 = [
    # EM-inducing
    "insecure_code",
    "jailbroken",
    "turner_bad_medical",
    "turner_risky_financial",
    "turner_extreme_sports",
    "emergent_plus_legal",
    "emergent_plus_security",
    "openai_health_bad",
    "evil_numbers",
    "aesthetic_unpopular",
    # WEAK
    "openai_health_subtle",
    "openai_health_mix25",
    "aesthetic_unpopular_weak",
    # NO-EM
    "secure_code",
    "educational",
    "openai_health_correct",
    "aesthetic_popular",
    "json_neg",
]


# ── Loaders ────────────────────────────────────────────────────────────────


def load_outcome_per_cell() -> dict[str, dict]:
    """Return ``{cell: {mean_L, per_seed: {seed: L}}}`` from
    ``eval_results/issue458/outcome/<cell>_seed<S>.json``. Cells with no
    seed files are omitted.
    """
    out: dict[str, dict] = {}
    if not OUTCOME_DIR.exists():
        logger.warning("Outcome dir %s does not exist — no eval data loaded", OUTCOME_DIR)
        return out
    for cell in CELLS_18:
        per_seed: dict[int, float] = {}
        for path in OUTCOME_DIR.glob(f"{cell}_seed*.json"):
            with open(path) as f:
                d = json.load(f)
            seed_val = int(d.get("seed", -1))
            L = d.get("L")
            if L is None or seed_val < 0:
                continue
            per_seed[seed_val] = float(L)
        if not per_seed:
            continue
        out[cell] = {
            "mean_L": float(np.mean(list(per_seed.values()))),
            "per_seed": per_seed,
        }
    return out


def load_predictor(directory: Path, headline_key: str, flavor: str) -> dict[str, float]:
    """Return ``{cell: headline}`` for the requested flavor's predictor JSON.

    Glob shape: ``<cell>_<flavor>.json``. Skips cells whose JSON is
    missing or whose headline key is absent.
    """
    out: dict[str, float] = {}
    if not directory.exists():
        logger.warning("Predictor dir %s missing", directory)
        return out
    for cell in CELLS_18:
        path = directory / f"{cell}_{flavor}.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        val = d.get(headline_key)
        if val is None:
            continue
        out[cell] = float(val)
    return out


def load_token_counts() -> dict[str, float]:
    """Return ``{cell: assistant_tokens_total}`` from token_counts.json."""
    if not TOKEN_COUNTS_PATH.exists():
        logger.warning("Token counts %s missing — covariate will be empty", TOKEN_COUNTS_PATH)
        return {}
    with open(TOKEN_COUNTS_PATH) as f:
        d = json.load(f)
    return {cell: float(v.get("assistant_tokens_total", 0)) for cell, v in d.items()}


# ── Statistics ─────────────────────────────────────────────────────────────


def spearman_with_n(x: list[float], y: list[float]) -> dict:
    """Spearman ρ + p-value + n. Uses scipy (handles tied ranks correctly)."""
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"rho": None, "p": None, "n": len(x), "note": "insufficient_variance"}
    res = stats.spearmanr(x, y)
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": len(x)}


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> dict:
    """Partial Spearman ρ(x, y | z): rank everything, OLS-residualize x on
    rank(z) and y on rank(z), then Pearson the residuals (the standard
    rank-partial-correlation construction).
    """
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2 or len(set(z)) < 2:
        return {"rho": None, "p": None, "n": len(x), "note": "insufficient_variance"}
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    # Residualize rx on rz, ry on rz via simple OLS (one-covariate).
    rz_centered = rz - rz.mean()
    denom = float((rz_centered**2).sum())
    if denom == 0:
        return {"rho": None, "p": None, "n": len(x), "note": "zero_variance_in_covariate"}
    beta_x = float((rz_centered * (rx - rx.mean())).sum()) / denom
    beta_y = float((rz_centered * (ry - ry.mean())).sum()) / denom
    rx_resid = rx - rx.mean() - beta_x * rz_centered
    ry_resid = ry - ry.mean() - beta_y * rz_centered
    if rx_resid.std() == 0 or ry_resid.std() == 0:
        return {"rho": None, "p": None, "n": len(x), "note": "zero_residual_variance"}
    pearson = stats.pearsonr(rx_resid, ry_resid)
    return {
        "rho": float(pearson.statistic),
        "p": float(pearson.pvalue),
        "n": len(x),
        "beta_x_on_z": beta_x,
        "beta_y_on_z": beta_y,
    }


def per_cell_residuals(L: list[float], z: list[float], cells: list[str]) -> list[dict]:
    """Return per-cell rows of {cell, L, log_tokens, fitted, residual}
    from an OLS L = a + b·log_tokens fit (the same controlling covariate
    used in the partial-Spearman block above). Useful eyeball signal:
    which cells drive (or break) the predictor's association with L
    AFTER training-volume is partialled out.
    """
    if len(L) < 2 or len(set(z)) < 2:
        return []
    arr_L = np.array(L, dtype=float)
    arr_z = np.array(z, dtype=float)
    z_c = arr_z - arr_z.mean()
    denom = float((z_c**2).sum())
    if denom == 0:
        return []
    b = float((z_c * (arr_L - arr_L.mean())).sum()) / denom
    a = float(arr_L.mean() - b * arr_z.mean())
    fitted = a + b * arr_z
    resid = arr_L - fitted
    return [
        {
            "cell": cells[i],
            "L": float(arr_L[i]),
            "log_tokens": float(arr_z[i]),
            "fitted": float(fitted[i]),
            "residual": float(resid[i]),
        }
        for i in range(len(cells))
    ]


# ── Per-predictor regression block ─────────────────────────────────────────


def regress_one_predictor(
    label: str,
    M_per_cell: dict[str, float],
    outcome: dict[str, dict],
    tokens: dict[str, float],
) -> dict:
    """Compute Spearman + partial-Spearman + per-cell residuals for ONE predictor.

    Joins the three sources on cell name; cells missing from any source
    are excluded with a logged warning.
    """
    common_cells = sorted(set(M_per_cell) & set(outcome) & set(tokens))
    missing = sorted((set(M_per_cell) | set(outcome) | set(tokens)) - set(common_cells))
    if missing:
        logger.warning("Predictor %s: dropping cells missing from one source: %s", label, missing)

    M_vals = [M_per_cell[c] for c in common_cells]
    L_vals = [outcome[c]["mean_L"] for c in common_cells]
    log_tokens = [math.log(max(tokens[c], 1.0)) for c in common_cells]

    raw = spearman_with_n(M_vals, L_vals)
    partial = partial_spearman(M_vals, L_vals, log_tokens)
    resid = per_cell_residuals(L_vals, log_tokens, common_cells)

    return {
        "predictor": label,
        "n_cells": len(common_cells),
        "cells": common_cells,
        "M_per_cell": {c: M_per_cell[c] for c in common_cells},
        "L_per_cell": {c: outcome[c]["mean_L"] for c in common_cells},
        "L_per_seed": {c: outcome[c]["per_seed"] for c in common_cells},
        "assistant_tokens_per_cell": {c: tokens[c] for c in common_cells},
        "spearman_raw": raw,
        "spearman_partial_log_tokens": partial,
        "per_cell_residuals_L_given_log_tokens": resid,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--flavor",
        default="NL",
        choices=["NL", "lit"],
        help="Which S_narrow flavor to read for the predictors (default NL).",
    )
    args = parser.parse_args()

    outcome = load_outcome_per_cell()
    tokens = load_token_counts()
    # M_1 = layer-21 cosine (#404 predictor, reused unchanged for #458)
    cosine = load_predictor(COSSIM_DIR, headline_key="M_1_headline", flavor=args.flavor)
    # M_js = 1 - mean_JS (#458 new predictor)
    jsdiv = load_predictor(JSDIV_DIR, headline_key="M_js", flavor=args.flavor)

    logger.info(
        "Loaded: %d outcome cells, %d cosine cells, %d JS cells, %d token-count cells",
        len(outcome),
        len(cosine),
        len(jsdiv),
        len(tokens),
    )

    blocks = {
        "M_cosine": regress_one_predictor("M_cosine", cosine, outcome, tokens),
        "M_js": regress_one_predictor("M_js", jsdiv, outcome, tokens),
    }

    summary = {
        "flavor": args.flavor,
        "blocks": blocks,
        "metadata": reproducibility_metadata({"script": "issue458_regress"}),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", OUTPUT_PATH.relative_to(PROJECT_ROOT))

    # Print a compact comparison table for the human eyeball.
    print(f"\n=== Issue #458 regression — flavor={args.flavor} ===")
    print(
        f"{'predictor':<12} {'n':>3}  {'rho_raw':>9} {'p_raw':>8}  "
        f"{'rho_partial':>11} {'p_partial':>10}"
    )
    for label, blk in blocks.items():
        raw = blk["spearman_raw"]
        par = blk["spearman_partial_log_tokens"]
        print(
            f"{label:<12} {blk['n_cells']:>3}  "
            f"{(raw.get('rho') or float('nan')):>9.4f} "
            f"{(raw.get('p') or float('nan')):>8.4f}  "
            f"{(par.get('rho') or float('nan')):>11.4f} "
            f"{(par.get('p') or float('nan')):>10.4f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
