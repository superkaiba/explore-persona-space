"""Phase 5 — Statistical comparison: paired bootstrap Delta-ro JS vs cosine.

Reads ``predictor_comparison.json`` (Phase 4) and emits three ro variants
(raw / source-FE / source-FE+base-rate-partial) per predictor PLUS the paired
bootstrap Delta-ro = |ro_JS| - |ro_cosine| 95% CI on the same 138 cells.

Also reports the secondary verdict: rank of ``comedian`` in the
``software_engineer`` bystander list per predictor (cosine ranks 23/23; H1
predicts JS pulls it into the top 5) AND whether JS beats the
``bystander_base_rate`` predictor at that recovery (so the diagnostic case
isn't trivially explained by base-rate confound).

Output: ``eval_results/issue_470/regression.json``.

Pure CPU.

Per plan A21 the pooled source-FE ro is computed as: rank-residualize delta
on source dummies, rank-residualize predictor on source dummies, Pearson the
rank residuals (= partial Spearman). For the additional base-rate control we
residualize against (source dummies + rank(bystander_base_rate)) jointly.

Usage::

    uv run python -m explore_persona_space.experiments.predictor_jsdiv_470.phase5_regress
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from scipy import stats

from explore_persona_space.experiments.predictor_jsdiv_470.common import (
    BOOTSTRAP_N,
    DEFAULT_SEED,
    PHASE4_PATH,
    PHASE5_PATH,
    read_json,
    reproducibility_metadata,
    write_json,
)

logger = logging.getLogger("predictor_jsdiv_470.phase5")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Predictors we score. The (label, key_in_phase4_row, polarity) tuple. Polarity
# matters because cosine + M_js are SIMILARITY (higher = closer = more leakage
# expected if H1 is correct), while JS_sym, KL_*, and base_rate_diff_neg_abs
# are mixed: we keep the raw sign in spearman and report |rho| for headline.
PREDICTORS = [
    # (label, key, "similarity"|"distance"|"baseline")
    ("cosine_l20_baseline", "cosine_l20_baseline", "similarity"),
    ("cosine_response_headline", "cosine_response_headline", "similarity"),
    ("M_js", "M_js", "similarity"),
    ("JS_sym_nats", "JS_sym_nats", "distance"),
    ("KL_src_to_bys_nats", "KL_src_to_bys_nats", "distance"),
    ("KL_bys_to_src_nats", "KL_bys_to_src_nats", "distance"),
    ("KL_sym_nats", "KL_sym_nats", "distance"),
    ("bystander_base_rate", "bystander_base_rate", "baseline"),
    ("base_rate_diff_neg_abs", "base_rate_diff_neg_abs", "baseline"),
]
HEADLINE_PREDICTOR_FOR_DELTA = "M_js"  # used in the paired Delta-ro
BASELINE_FOR_DELTA = "cosine_l20_baseline"


def _ranks(arr: np.ndarray) -> np.ndarray:
    return stats.rankdata(arr).astype(float)


def _residualize_on_dummies(y: np.ndarray, dummies: np.ndarray) -> np.ndarray:
    """OLS-residualize y on a (N, k) design matrix of source-indicator dummies.

    Use the rank space so the downstream Pearson on residuals = partial Spearman.
    """
    if dummies.size == 0:
        return y - y.mean()
    # Add intercept column.
    x = np.column_stack([np.ones(y.shape[0]), dummies])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def _build_source_dummies(sources: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return a (N, k-1) reference-coded dummy matrix and the dropped reference name."""
    uniq = sorted(set(sources))
    if len(uniq) <= 1:
        return np.zeros((len(sources), 0)), uniq[0] if uniq else ""
    ref = uniq[0]
    cols = []
    for s in uniq[1:]:
        cols.append([1.0 if x == s else 0.0 for x in sources])
    return np.array(cols, dtype=float).T, ref


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
        return {"rho": None, "p": None, "n": len(x), "note": "insufficient_variance"}
    res = stats.spearmanr(x, y)
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": len(x)}


def partial_spearman_on_dummies(x: np.ndarray, y: np.ndarray, dummies: np.ndarray) -> dict:
    """Spearman ro of x and y after residualizing each on the dummies (rank space)."""
    if len(x) < 4 or len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
        return {"rho": None, "p": None, "n": len(x), "note": "insufficient_variance"}
    rx_res = _residualize_on_dummies(_ranks(x), dummies)
    ry_res = _residualize_on_dummies(_ranks(y), dummies)
    if rx_res.std() == 0 or ry_res.std() == 0:
        return {"rho": None, "p": None, "n": len(x), "note": "zero_residual_variance"}
    pr = stats.pearsonr(rx_res, ry_res)
    return {"rho": float(pr.statistic), "p": float(pr.pvalue), "n": len(x)}


def per_source_spearman(sources: list[str], x: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """Per-source ro of x vs y over that source's bystanders."""
    out: dict[str, dict] = {}
    sources_arr = np.array(sources)
    for src in sorted(set(sources)):
        mask = sources_arr == src
        if mask.sum() < 3:
            out[src] = {"rho": None, "p": None, "n": int(mask.sum()), "note": "insufficient_n"}
            continue
        out[src] = spearman(x[mask], y[mask])
    return out


def fisher_z_average(per_source: dict[str, dict]) -> dict:
    """Fisher-z average of per-source ro (handles ro's non-additivity)."""
    rhos = [d["rho"] for d in per_source.values() if d["rho"] is not None]
    if len(rhos) < 2:
        return {"rho_z_avg": None, "n_sources": len(rhos)}
    zs = [np.arctanh(min(max(r, -0.999999), 0.999999)) for r in rhos]
    z_avg = float(np.mean(zs))
    return {"rho_z_avg": float(np.tanh(z_avg)), "n_sources": len(rhos)}


def paired_bootstrap_delta_rho(
    sources: list[str],
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Paired bootstrap of Delta-|rho| = |rho(x_a, y)| - |rho(x_b, y)|.

    On each resample we use the SAME bootstrap indices for both predictors
    (so they share the resampled cells), compute source-FE-residualized
    Spearman ro for each, and take |ro_a| - |ro_b|.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 10:
        return {
            "delta_rho_mean": None,
            "ci_low_95": None,
            "ci_high_95": None,
            "n_boot": 0,
            "note": "n_too_small",
        }

    deltas = np.empty(n_boot, dtype=float)
    valid = 0
    for _b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        srcs_b = [sources[i] for i in idx]
        dummies_b, _ = _build_source_dummies(srcs_b)
        try:
            rho_a = partial_spearman_on_dummies(x_a[idx], y[idx], dummies_b)["rho"]
            rho_b = partial_spearman_on_dummies(x_b[idx], y[idx], dummies_b)["rho"]
        except Exception:
            continue
        if rho_a is None or rho_b is None:
            continue
        deltas[valid] = abs(rho_a) - abs(rho_b)
        valid += 1
    if valid < 100:
        return {
            "delta_rho_mean": None,
            "ci_low_95": None,
            "ci_high_95": None,
            "n_boot_valid": valid,
            "note": "too_few_valid_resamples",
        }
    deltas = deltas[:valid]
    return {
        "delta_rho_mean": float(np.mean(deltas)),
        "delta_rho_median": float(np.median(deltas)),
        "ci_low_95": float(np.percentile(deltas, 2.5)),
        "ci_high_95": float(np.percentile(deltas, 97.5)),
        "n_boot_requested": n_boot,
        "n_boot_valid": int(valid),
    }


def regress_predictor(
    label: str,
    polarity: str,
    sources: list[str],
    predictor_vals: np.ndarray,
    delta_vals: np.ndarray,
    base_rate_vals: np.ndarray,
) -> dict:
    """Three ro variants (raw / source-FE / source-FE + base-rate-partial)
    plus per-source ro + Fisher-z avg.
    """
    dummies, ref = _build_source_dummies(sources)
    # Add bystander base rate as an extra covariate (rank-residualized) for
    # the third variant.
    dummies_plus_br = np.column_stack(
        [dummies, _ranks(base_rate_vals) - _ranks(base_rate_vals).mean()]
    )

    block = {
        "label": label,
        "polarity": polarity,
        "n_cells": len(delta_vals),
        "spearman_raw": spearman(predictor_vals, delta_vals),
        "spearman_source_fe": partial_spearman_on_dummies(predictor_vals, delta_vals, dummies),
        "spearman_source_fe_plus_base_rate": partial_spearman_on_dummies(
            predictor_vals, delta_vals, dummies_plus_br
        ),
        "per_source": per_source_spearman(sources, predictor_vals, delta_vals),
        "source_dummy_reference": ref,
    }
    block["fisher_z_avg_per_source"] = fisher_z_average(block["per_source"])
    return block


def secondary_diagnostic_ranks(
    cells: list[dict],
    sources: list[str],
) -> dict:
    """For each source, rank the bystanders by each predictor (smaller rank =
    more similar / more predictive of leak).

    Headline focus: software_engineer -> comedian's rank under cosine_l20
    (= 23/23 per #411) vs M_js (H1 predicts top-5) AND vs bystander_base_rate
    (which trivially recovers comedian due to its high intrinsic agreeableness).
    """
    by_source: dict[str, dict] = {}
    for src in sorted(set(sources)):
        src_cells = [c for c in cells if c["source"] == src]
        if not src_cells:
            continue
        # For each predictor, lower rank = "closer to source / more agreeable".
        # We always sort so rank-1 is the predictor's TOP pick for leakage:
        #   similarity (cosine, M_js, base_rate): higher value -> rank 1.
        #   distance   (JS_sym, KL):              lower value  -> rank 1.
        per_predictor: dict[str, dict] = {}
        for label, key, polarity in PREDICTORS:
            vals = []
            bys_list = []
            for c in src_cells:
                v = c.get(key)
                if v is None:
                    continue
                vals.append(float(v))
                bys_list.append(c["bystander"])
            if not vals:
                continue
            arr = np.array(vals)
            order = np.argsort(arr) if polarity == "distance" else np.argsort(-arr)
            ranks = {bys_list[order[i]]: i + 1 for i in range(len(order))}
            per_predictor[label] = {
                "rank_of_comedian": ranks.get("comedian"),
                "rank_of_data_scientist": ranks.get("data_scientist"),
                "n_bystanders": len(vals),
                # Show the predictor's top-5 picks for transparency.
                "top_5_bystanders": [bys_list[order[i]] for i in range(min(5, len(order)))],
            }
        by_source[src] = per_predictor
    return by_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not PHASE4_PATH.exists():
        raise RuntimeError(f"Phase 4 output missing at {PHASE4_PATH}")
    phase4 = read_json(PHASE4_PATH)
    cells = phase4["cells"]

    # Build arrays restricted to cells that have ALL the predictors we need.
    keys_required = {k for _, k, _ in PREDICTORS} | {"delta", "source", "bystander"}
    usable = [c for c in cells if all(c.get(k) is not None for k in keys_required)]
    dropped = len(cells) - len(usable)
    if dropped:
        logger.warning("Dropped %d/%d cells missing one or more predictors", dropped, len(cells))
    if not usable:
        raise RuntimeError("No cells survive the full-predictor filter; cannot regress.")

    sources = [c["source"] for c in usable]
    delta_vals = np.array([c["delta"] for c in usable], dtype=float)
    base_rate_vals = np.array([c["bystander_base_rate"] for c in usable], dtype=float)

    per_predictor: dict[str, dict] = {}
    for label, key, polarity in PREDICTORS:
        vals = np.array([c[key] for c in usable], dtype=float)
        per_predictor[label] = regress_predictor(
            label=label,
            polarity=polarity,
            sources=sources,
            predictor_vals=vals,
            delta_vals=delta_vals,
            base_rate_vals=base_rate_vals,
        )

    # Headline paired-bootstrap Delta-ro: |ro(M_js)| - |ro(cosine_l20)|.
    x_js = np.array([c[HEADLINE_PREDICTOR_FOR_DELTA] for c in usable], dtype=float)
    x_cos = np.array([c[BASELINE_FOR_DELTA] for c in usable], dtype=float)
    delta_rho = paired_bootstrap_delta_rho(
        sources=sources,
        x_a=x_js,
        x_b=x_cos,
        y=delta_vals,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    # Cross-predictor correlation (sanity per §6.5 item 5).
    cross_corr = {}
    for a_label, a_key, _ in PREDICTORS:
        a_vals = np.array([c[a_key] for c in usable], dtype=float)
        cross_corr[a_label] = {}
        for b_label, b_key, _ in PREDICTORS:
            b_vals = np.array([c[b_key] for c in usable], dtype=float)
            r = spearman(a_vals, b_vals)
            cross_corr[a_label][b_label] = r.get("rho")

    # Secondary verdict: per-source bystander ranks.
    rank_diagnostic = secondary_diagnostic_ranks(usable, sources)

    # Seed-extension decision rule (§6.4): only flag the [0.05, 0.15] buffer
    # zone; we do not auto-launch the extension (predictor-only re-analysis,
    # not a training loop — the user can re-invoke if they want it).
    buffer_flag = False
    if delta_rho.get("delta_rho_mean") is not None:
        m = abs(delta_rho["delta_rho_mean"])
        buffer_flag = 0.05 <= m <= 0.15

    summary = {
        "n_cells_used": len(usable),
        "n_cells_dropped": dropped,
        "predictors": per_predictor,
        "paired_bootstrap_delta_rho": {
            "spec": "|rho(M_js)| - |rho(cosine_l20_baseline)| on source-FE-residualized ranks",
            **delta_rho,
        },
        "secondary_diagnostic_bystander_ranks": rank_diagnostic,
        "cross_predictor_spearman": cross_corr,
        "in_seed_extension_buffer_zone": buffer_flag,
        "metadata": reproducibility_metadata({"script": "predictor_jsdiv_470.phase5_regress"}),
    }
    write_json(PHASE5_PATH, summary)
    logger.info("Wrote %s", PHASE5_PATH)

    # Compact stdout table for the human eyeball.
    print(f"\n=== Phase 5 — predictor comparison (n_cells_used={len(usable)}) ===")
    print(f"{'predictor':<32} {'raw':>9} {'src_FE':>9} {'src_FE+BR':>11}  per_src_z_avg")
    for label, blk in per_predictor.items():
        raw = blk["spearman_raw"].get("rho")
        fe = blk["spearman_source_fe"].get("rho")
        fe_br = blk["spearman_source_fe_plus_base_rate"].get("rho")
        z = blk["fisher_z_avg_per_source"].get("rho_z_avg")

        def _f(x):
            return f"{x:>9.4f}" if x is not None else f"{'nan':>9}"

        z_str = f"{z:>9.4f}" if z is not None else f"{'nan':>9}"
        print(f"{label:<32} {_f(raw)} {_f(fe)}   {_f(fe_br):>9}  {z_str}")

    dr = delta_rho
    print(
        f"\nPaired Delta-rho (|M_js| - |cosine_l20|): mean={dr.get('delta_rho_mean')}, "
        f"95% CI=[{dr.get('ci_low_95')}, {dr.get('ci_high_95')}]"
    )

    # Comedian recovery diagnostic (the headline secondary case).
    se = rank_diagnostic.get("software_engineer", {})
    if se:
        print("\nsoftware_engineer -> comedian rank by predictor (smaller = ranked closer):")
        for label, blk in se.items():
            print(
                f"  {label:<32} comedian_rank={blk.get('rank_of_comedian')}  "
                f"top5={blk.get('top_5_bystanders')}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
