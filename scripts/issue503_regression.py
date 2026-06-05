#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, →) in scientific docstrings + logs.
"""Issue #503 — pooled binomial mixed-model regression (plan §9).

Loads per-cell predictor JSONs + per-cell verdict JSONs, joins them into
``RegressionRow`` records, runs:

1. Primary binomial mixed model on (k, n - k) per §9 + MF3.
2. Pseudocount fallback if convergence fails.
3. Raw-rate Spearman ρ (H4 secondary).
4. Partial-Spearman ladder (raw → partial-log-tokens →
   partial-lexical → partial-base-rate).
5. Leave-one-family-out sensitivity.
6. Permutation null (1000 iter; exact n!-enumeration for B→B with n=4).
7. FDR-BH correction across the 3 statistically-tested strata (N→N,
   N→B-EM, N→B-syco). B→B is descriptive-only.
8. H4 headline gate.

Output: eval_results/issue503/regression_v1.json.

Usage::

    # Smoke (1 cell, just verify convergence)
    uv run python scripts/issue503_regression.py --smoke

    # Full sweep
    uv run python scripts/issue503_regression.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue503_regression")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_predictor_records() -> dict[tuple[str, str, int], dict]:
    """Read every eval_results/issue503/predictors/*.json into a map
    keyed by (source, target_id, seed).
    """
    pred_dir = PROJECT_ROOT / "eval_results" / "issue503" / "predictors"
    if not pred_dir.exists():
        return {}
    out: dict[tuple[str, str, int], dict] = {}
    for p in pred_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            logger.warning("Skipping corrupt predictor JSON %s", p)
            continue
        key = (data["source"], data["target_id"], data["seed"])
        out[key] = data
    return out


def _load_verdict_records() -> dict[tuple[str, str, int], dict]:
    """Read every eval_results/issue503/cross_eval/<source>_seed{S}/<target>.verdict.json."""
    base_dir = PROJECT_ROOT / "eval_results" / "issue503" / "cross_eval"
    if not base_dir.exists():
        return {}
    out: dict[tuple[str, str, int], dict] = {}
    for src_dir in base_dir.iterdir():
        if not src_dir.is_dir():
            continue
        name = src_dir.name
        if "_seed" not in name:
            continue
        source, seed_str = name.rsplit("_seed", 1)
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        for verdict_path in src_dir.glob("*.verdict.json"):
            target_id = verdict_path.stem.removesuffix(".verdict")
            try:
                data = json.loads(verdict_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt verdict JSON %s", verdict_path)
                continue
            out[(source, target_id, seed)] = data
    return out


def _build_regression_rows():
    from explore_persona_space.experiments.issue503.behaviors import (
        SOURCE_FAMILY,
        enumerate_cells,
    )
    from explore_persona_space.experiments.issue503.regression import RegressionRow

    predictors = _load_predictor_records()
    verdicts = _load_verdict_records()
    rows: list[RegressionRow] = []
    skipped = 0
    for cell in enumerate_cells():
        if cell.row_kind == "install_qc":
            continue
        key = (cell.source, cell.target_id, cell.seed)
        pred = predictors.get(key)
        verd = verdicts.get(key)
        if pred is None or verd is None or "k" not in verd:
            skipped += 1
            continue
        cosine_mean = float(pred["cosine"]["mean"])
        cosine_ts = pred.get("cosine_topic_stripped", {}).get("mean")
        family = SOURCE_FAMILY.get(cell.source, "unknown")
        median_tokens = float(verd.get("median_tokens", 100.0))
        rows.append(
            RegressionRow(
                source=cell.source,
                target=cell.target_id,
                seed=cell.seed,
                cell_type=cell.cell_type,
                family=family,
                k=int(verd["k"]),
                n=int(verd["n"]),
                cosine_predictor=cosine_mean,
                cosine_topic_stripped=cosine_ts,
                log_tokens=math.log(max(1.0, median_tokens)),
                lexical_persona_cosine=float(pred.get("lexical_persona_cosine", 0.0)),
                base_rate=float(pred.get("base_rate", 0.0)),
                js_sliced_on_target=pred.get("js_sliced_on_target"),
                js_sliced_off_target=pred.get("js_sliced_off_target"),
                kl_secondary_dv=verd.get("kl_secondary_dv"),
            )
        )
    logger.info("Built %d off-diagonal rows (skipped %d missing)", len(rows), skipped)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: just confirm convergence on whatever rows are available.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue503" / "regression_v1.json",
    )
    args = parser.parse_args()

    from explore_persona_space.experiments.issue503.regression import (
        PRE_REG_HEADLINE_STRATA,
        b_to_b_descriptive,
        fdr_bh,
        fit_binomial_mixed,
        headline_h4_verdict,
        leave_one_family_out,
        partial_spearman_ladder,
        spearman_rho,
    )

    rows = _build_regression_rows()
    if not rows:
        logger.error("No rows assembled — both predictor + verdict JSONs are needed.")
        return 1

    result: dict = {
        "n_rows": len(rows),
        "pre_reg_strata": list(PRE_REG_HEADLINE_STRATA),
    }

    # Per-stratum Spearman — ONLY the 3 pre-registered headline strata
    # (B→B is descriptive-only per MF2/MF-E; reported in a separate
    # ``b_to_b_descriptive`` field with NO p_value).
    per_stratum_rho: dict[str, dict] = {}
    for s in PRE_REG_HEADLINE_STRATA:
        per_stratum_rho[s] = spearman_rho(rows, strata=(s,))
    result["per_stratum_rho"] = per_stratum_rho

    # Pooled headline (3 strata, B→B excluded per MF2).
    result["headline_h4"] = headline_h4_verdict(rows)

    # Partial-Spearman ladder.
    result["partial_ladder"] = partial_spearman_ladder(rows)

    # Leave-one-family-out.
    result["leave_one_family_out"] = leave_one_family_out(rows)

    # FDR-BH over 3 strata.
    p_values = [per_stratum_rho[s]["p_value"] for s in PRE_REG_HEADLINE_STRATA]
    rejected = fdr_bh(p_values, alpha=0.05)
    result["fdr_bh"] = {
        "strata": list(PRE_REG_HEADLINE_STRATA),
        "p_values": p_values,
        "rejected": rejected,
    }

    # B→B descriptive-only (MF-E round-2 revision): point estimate +
    # 95% bootstrap CI + exact permutation-null PMF. NO p_value.
    result["b_to_b_descriptive"] = b_to_b_descriptive(rows)

    # Primary regression (smoke: just attempt; sweep: full fit).
    if args.smoke:
        result["binomial_fit"] = {"skipped_in_smoke": True}
    else:
        fit = fit_binomial_mixed(rows, strata=PRE_REG_HEADLINE_STRATA)
        result["binomial_fit"] = {
            "model_form": fit.model_form,
            "n_rows": fit.n_rows,
            "converged": fit.converged,
            "coef_cosine": fit.coef_cosine,
            "se_cosine": fit.se_cosine,
            "ci_low_cosine": fit.ci_low_cosine,
            "ci_high_cosine": fit.ci_high_cosine,
            "coefs_full": fit.coefs_full,
            "notes": fit.notes,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Wrote regression results to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
