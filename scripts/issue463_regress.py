#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥, ‖) in scientific docstrings + logs.
"""Issue #463 — head-to-head regression: new predictors vs #458 deprecated baselines.

Across the 18 already-trained #458 cells (mean of seeds 0 + 137; the
``openai_health_subtle`` cell is seed-137 only), regress each candidate
base-model predictor against the post-SFT broad-EM rate ``L`` from
``eval_results/issue458/outcome/<cell>_seed<S>.json``:

* **Issue #463 new predictors:**

  - ``M_js`` (full-response Rao-Blackwellized JS),
  - ``M_symkl`` (= exp(-symKL); polarity-aligned similarity),
  - ``KL_narrow_broad`` (raw nats, no polarity flip — used as the regression
    INPUT, not a similarity), and the same for ``KL_broad_narrow``,
  - ``cosine[extraction_point][layer]`` over
    ``extraction_point ∈ {last_prompt_token, response_mean}``,
    ``layer ∈ {7, 14, 21, 27}``.

* **#458 deprecated baselines (head-to-head):**

  - ``M_js_first_token`` (#458's single-next-token JS, key ``M_js`` from
    ``eval_results/issue458/predictor_jsdiv/<cell>_<flavor>.json``),
  - ``cosine_layer21_last_prompt_token_issue404`` (#458's layer-21
    last-prompt-token cosine, key ``M_1_headline`` from
    ``eval_results/issue_404/predictor_cossim/<cell>_<flavor>.json``).

Per predictor we report (matching ``scripts/issue458_regress.py``):

* Spearman ρ(M, L) across cells (raw association, scipy with ties),
* Partial Spearman ρ(M, L | log(assistant_tokens)) (controls the
  training-volume confound; ``eval_results/issue458/token_counts.json``).

Output: ``eval_results/issue463/regression.json`` + sorted comparison
table on stdout.

Usage::

    uv run python scripts/issue463_regress.py                # default flavor=NL
    uv run python scripts/issue463_regress.py --flavor lit
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

logger = logging.getLogger("issue463_regress")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EVAL_DIR_458 = PROJECT_ROOT / "eval_results" / "issue458"
EVAL_DIR_463 = PROJECT_ROOT / "eval_results" / "issue463"
OUTCOME_DIR = EVAL_DIR_458 / "outcome"
TOKEN_COUNTS_PATH = EVAL_DIR_458 / "token_counts.json"

SEQDIV_DIR = EVAL_DIR_463 / "predictor_seqdiv"
COSSIM_DIR = EVAL_DIR_463 / "predictor_cossim"
JSDIV_458_DIR = EVAL_DIR_458 / "predictor_jsdiv"
COSSIM_404_DIR = PROJECT_ROOT / "eval_results" / "issue_404" / "predictor_cossim"

OUTPUT_PATH = EVAL_DIR_463 / "regression.json"

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

LAYERS = [7, 14, 21, 27]
EXTRACTION_POINTS = ("last_prompt_token", "response_mean")


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


def load_token_counts() -> dict[str, float]:
    if not TOKEN_COUNTS_PATH.exists():
        logger.warning("Token counts %s missing — covariate will be empty", TOKEN_COUNTS_PATH)
        return {}
    with open(TOKEN_COUNTS_PATH) as f:
        d = json.load(f)
    return {cell: float(v.get("assistant_tokens_total", 0)) for cell, v in d.items()}


def load_seqdiv(flavor: str) -> dict[str, dict]:
    """Return ``{cell: full seqdiv JSON dict}`` per cell."""
    out: dict[str, dict] = {}
    if not SEQDIV_DIR.exists():
        logger.warning("Seqdiv dir %s missing", SEQDIV_DIR)
        return out
    for cell in CELLS_18:
        path = SEQDIV_DIR / f"{cell}_{flavor}.json"
        if not path.exists():
            continue
        with open(path) as f:
            out[cell] = json.load(f)
    return out


def load_cossim_463(flavor: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not COSSIM_DIR.exists():
        logger.warning("Cossim dir %s missing", COSSIM_DIR)
        return out
    for cell in CELLS_18:
        path = COSSIM_DIR / f"{cell}_{flavor}.json"
        if not path.exists():
            continue
        with open(path) as f:
            out[cell] = json.load(f)
    return out


def load_simple_headline(directory: Path, headline_key: str, flavor: str) -> dict[str, float]:
    """Return ``{cell: scalar}`` reading a single top-level key from each
    ``<cell>_<flavor>.json`` under ``directory``.
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


# ── Stats helpers (mirror issue458_regress.py shapes) ──────────────────────


def spearman_with_n(x: list[float], y: list[float]) -> dict:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"rho": None, "p": None, "n": len(x), "note": "insufficient_variance"}
    res = stats.spearmanr(x, y)
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": len(x)}


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> dict:
    """Partial Spearman ρ(x, y | z) via rank-OLS residualization + Pearson."""
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2 or len(set(z)) < 2:
        return {"rho": None, "p": None, "n": len(x), "note": "insufficient_variance"}
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
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


def regress_one(
    label: str,
    M_per_cell: dict[str, float],
    outcome: dict[str, dict],
    tokens: dict[str, float],
) -> dict:
    common = sorted(set(M_per_cell) & set(outcome) & set(tokens))
    M_vals = [M_per_cell[c] for c in common]
    L_vals = [outcome[c]["mean_L"] for c in common]
    log_tokens = [math.log(max(tokens[c], 1.0)) for c in common]
    raw = spearman_with_n(M_vals, L_vals)
    partial = partial_spearman(M_vals, L_vals, log_tokens)
    return {
        "predictor": label,
        "n_cells": len(common),
        "cells": common,
        "M_per_cell": {c: M_per_cell[c] for c in common},
        "L_per_cell": {c: outcome[c]["mean_L"] for c in common},
        "spearman_raw": raw,
        "spearman_partial_log_tokens": partial,
    }


# ── Predictor assembly ─────────────────────────────────────────────────────


def assemble_predictors(flavor: str) -> dict[str, dict[str, float]]:
    """Return ``{predictor_label: {cell: scalar}}`` for ALL predictors we
    head-to-head: the #463 seqdiv family (4 scalars), the #463 cossim
    family (2 extraction × 4 layers = 8 scalars), and the #458 deprecated
    baselines (2 scalars).
    """
    out: dict[str, dict[str, float]] = {}

    # #463 seqdiv predictors
    seqdiv = load_seqdiv(flavor)
    for key in ("M_js", "M_symkl", "KL_narrow_broad", "KL_broad_narrow", "JS", "symKL"):
        scalar_per_cell: dict[str, float] = {}
        for cell, d in seqdiv.items():
            val = d.get(key)
            if val is None:
                continue
            scalar_per_cell[cell] = float(val)
        out[f"seqdiv_{key}"] = scalar_per_cell

    # #463 cossim predictors
    cossim = load_cossim_463(flavor)
    for ep in EXTRACTION_POINTS:
        for li in LAYERS:
            label = f"cossim_{ep}_L{li}"
            scalar_per_cell = {}
            for cell, d in cossim.items():
                ce = d.get("cos_by_extraction", {}).get(ep, {})
                val = ce.get(str(li))
                if val is None:
                    continue
                scalar_per_cell[cell] = float(val)
            out[label] = scalar_per_cell

    # #458 deprecated baselines (head-to-head)
    out["baseline_458_M_js_first_token"] = load_simple_headline(
        JSDIV_458_DIR, headline_key="M_js", flavor=flavor
    )
    out["baseline_404_cosine_L21_last_prompt"] = load_simple_headline(
        COSSIM_404_DIR, headline_key="M_1_headline", flavor=flavor
    )

    return out


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--flavor",
        default="NL",
        choices=["NL", "lit"],
        help="Which S_narrow flavor to read for ALL predictors (default NL).",
    )
    args = parser.parse_args()

    outcome = load_outcome_per_cell()
    tokens = load_token_counts()
    logger.info(
        "Loaded: %d outcome cells, %d token-count cells (flavor=%s)",
        len(outcome),
        len(tokens),
        args.flavor,
    )

    predictors = assemble_predictors(args.flavor)
    logger.info(
        "Assembled %d predictors; per-predictor cell counts: %s",
        len(predictors),
        {k: len(v) for k, v in predictors.items()},
    )

    blocks: dict[str, dict] = {}
    for label, scalar_per_cell in predictors.items():
        blocks[label] = regress_one(label, scalar_per_cell, outcome, tokens)

    summary = {
        "flavor": args.flavor,
        "blocks": blocks,
        "metadata": reproducibility_metadata({"script": "issue463_regress"}),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Per-flavor output: keep regression.json carrying the most recent flavor
    # run; also emit a per-flavor sibling so both NL + lit results are
    # preserved when the launcher runs both.
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    per_flavor_path = OUTPUT_PATH.parent / f"regression_{args.flavor}.json"
    with open(per_flavor_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Wrote %s and %s",
        OUTPUT_PATH.relative_to(PROJECT_ROOT),
        per_flavor_path.relative_to(PROJECT_ROOT),
    )

    # Sorted comparison table, by absolute partial-ρ descending (most
    # informative predictors first; head-to-head with the #458 baselines).
    def sort_key(item):
        _, blk = item
        rho = blk.get("spearman_partial_log_tokens", {}).get("rho")
        return abs(rho) if rho is not None else -1.0

    sorted_blocks = sorted(blocks.items(), key=sort_key, reverse=True)

    print(f"\n=== Issue #463 regression — flavor={args.flavor} ===")
    print(
        f"{'predictor':<42} {'n':>3}  {'rho_raw':>9} {'p_raw':>8}  "
        f"{'rho_partial':>11} {'p_partial':>10}"
    )
    for label, blk in sorted_blocks:
        raw = blk["spearman_raw"]
        par = blk["spearman_partial_log_tokens"]
        print(
            f"{label:<42} {blk['n_cells']:>3}  "
            f"{(raw.get('rho') if raw.get('rho') is not None else float('nan')):>9.4f} "
            f"{(raw.get('p') if raw.get('p') is not None else float('nan')):>8.4f}  "
            f"{(par.get('rho') if par.get('rho') is not None else float('nan')):>11.4f} "
            f"{(par.get('p') if par.get('p') is not None else float('nan')):>10.4f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
