"""Phase 5 — Statistical comparison: paired bootstrap Delta-ro JS vs cosine.

Reads ``predictor_comparison.json`` (Phase 4) and emits three ro variants
(raw / source-FE / source-FE+base-rate-partial) per predictor PLUS the paired
bootstrap Delta-ro = |ro_JS| - |ro_cosine| 95% CI on the same 138 cells.

Also reports the secondary verdict: rank of ``comedian`` in the
``software_engineer`` bystander list per predictor (cosine ranks 23/23; H1
predicts JS pulls it into the top 5) AND whether JS beats the
``bystander_base_rate`` predictor at that recovery (so the diagnostic case
isn't trivially explained by base-rate confound).

Round-2 additions
-----------------
* **Kill criterion (plan §1):** if std(JS_sym_nats) across cells < 0.01 nats,
  HALT before forcing a regression on a flat predictor and report
  ``js_predictor_dynamic_range_insufficient: true``.
* **Cosine layer-sweep ladder (plan §4 / §6.6):** scores
  ``cosine_response_l{7,14,21,27}`` alongside the headline + reports which layer
  is best per source.
* **Per-source bootstrap 95% CI + permutation p (plan Phase 5):**
  ``per_source`` blocks now include 10000-resample bootstrap CIs and
  10000-shuffle permutation p, matching #411's settings.
* **Response-length confound (plan §6.5 item 4):** rank-correlation of
  ``JS_sym_nats`` vs ``resp_len_diff_abs`` over the 138 cells, flagged when
  ``|rho_len| >= 0.3``.

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
    PERMUTATION_N,
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
# Concern #5: cosine response-token recipe (b) layer-sweep predictors added so
# the analyzer can see the "is JS beating cosine, or just beating the wrong
# extraction recipe" ladder per plan §6.6.
PREDICTORS: list[tuple[str, str, str]] = [
    # (label, key, "similarity"|"distance"|"baseline")
    ("cosine_l20_baseline", "cosine_l20_baseline", "similarity"),
    ("cosine_response_l7", "cosine_response_l7", "similarity"),
    ("cosine_response_l14", "cosine_response_l14", "similarity"),
    ("cosine_response_l21", "cosine_response_l21", "similarity"),
    ("cosine_response_l27", "cosine_response_l27", "similarity"),
    ("cosine_response_headline", "cosine_response_headline", "similarity"),
    ("M_js", "M_js", "similarity"),
    ("JS_sym_nats", "JS_sym_nats", "distance"),
    ("KL_src_to_bys_nats", "KL_src_to_bys_nats", "distance"),
    ("KL_bys_to_src_nats", "KL_bys_to_src_nats", "distance"),
    ("KL_sym_nats", "KL_sym_nats", "distance"),
    ("bystander_base_rate", "bystander_base_rate", "baseline"),
    ("base_rate_diff_neg_abs", "base_rate_diff_neg_abs", "baseline"),
]
COSINE_LAYER_SWEEP_LABELS = (
    "cosine_response_l7",
    "cosine_response_l14",
    "cosine_response_l21",
    "cosine_response_l27",
)
HEADLINE_PREDICTOR_FOR_DELTA = "M_js"  # used in the paired Delta-ro
BASELINE_FOR_DELTA = "cosine_l20_baseline"

# Concern #4 kill threshold (plan §1).
JS_STD_KILL_THRESHOLD_NATS = 0.01
# Concern #3 flag threshold for response-length confound.
JS_LEN_CONFOUND_FLAG_RHO = 0.3


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


def _spearman_bootstrap_and_permutation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict:
    """Concern #6 — bootstrap 95% CI on rho + permutation p, matching #411.

    Bootstrap: resample (x, y) pairs with replacement n_boot times, recompute
    Spearman ro on each resample, take 2.5%/97.5% percentile as CI bounds.
    Permutation: shuffle y n_perm times, recompute Spearman ro on each shuffle,
    two-sided p = fraction of |ro_perm| >= |ro_observed|.
    """
    if len(x) < 3 or len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
        return {"ci_low_95": None, "ci_high_95": None, "permutation_p": None}
    rng = np.random.default_rng(seed)
    n = len(x)
    obs_rho = stats.spearmanr(x, y).statistic
    # Bootstrap CI.
    boot_rhos = np.empty(n_boot, dtype=float)
    valid = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs, ys = x[idx], y[idx]
        if len(set(xs.tolist())) < 2 or len(set(ys.tolist())) < 2:
            continue
        r = stats.spearmanr(xs, ys).statistic
        if np.isnan(r):
            continue
        boot_rhos[valid] = r
        valid += 1
    if valid >= 100:
        boot_slice = boot_rhos[:valid]
        ci_low = float(np.percentile(boot_slice, 2.5))
        ci_high = float(np.percentile(boot_slice, 97.5))
        n_boot_valid = int(valid)
    else:
        ci_low = ci_high = None
        n_boot_valid = int(valid)
    # Permutation p (two-sided).
    perm_rhos = np.empty(n_perm, dtype=float)
    valid_p = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        r = stats.spearmanr(x, y_perm).statistic
        if np.isnan(r):
            continue
        perm_rhos[valid_p] = r
        valid_p += 1
    if valid_p >= 100 and not np.isnan(obs_rho):
        perm_p = float((np.abs(perm_rhos[:valid_p]) >= abs(obs_rho)).mean())
    else:
        perm_p = None
    return {
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "n_boot_valid": n_boot_valid,
        "permutation_p": perm_p,
        "n_perm_valid": int(valid_p),
    }


def per_source_spearman(
    sources: list[str],
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict[str, dict]:
    """Per-source ro of x vs y over that source's bystanders, with bootstrap CI
    + permutation p per source (concern #6, matches #411's per-source settings).
    """
    out: dict[str, dict] = {}
    sources_arr = np.array(sources)
    for src_idx, src in enumerate(sorted(set(sources))):
        mask = sources_arr == src
        if mask.sum() < 3:
            out[src] = {
                "rho": None,
                "p": None,
                "n": int(mask.sum()),
                "note": "insufficient_n",
                "ci_low_95": None,
                "ci_high_95": None,
                "permutation_p": None,
            }
            continue
        base = spearman(x[mask], y[mask])
        boot = _spearman_bootstrap_and_permutation(
            x[mask],
            y[mask],
            n_boot=n_boot,
            n_perm=n_perm,
            # Per-source seed offset so different sources get different bootstrap draws.
            seed=seed + 1000 * (src_idx + 1),
        )
        out[src] = {**base, **boot}
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
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict:
    """Three ro variants (raw / source-FE / source-FE + base-rate-partial)
    plus per-source ro + Fisher-z avg + per-source bootstrap CI / permutation p.
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
        "per_source": per_source_spearman(
            sources, predictor_vals, delta_vals, n_boot=n_boot, n_perm=n_perm, seed=seed
        ),
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


def cosine_layer_ladder(per_predictor: dict[str, dict]) -> dict:
    """Concern #5 (plan §6.6): per-source best-layer pick across the cosine
    recipe (b) layer sweep, plus the pooled headline.

    Reports for each layer the source-FE pooled ro and per-source ro; the
    "best layer per source" is the layer with the largest |source-FE ro|
    among the sweep set (a smaller |ro| means a worse predictor of Delta).
    """
    sweep_blocks = {
        label: per_predictor[label] for label in COSINE_LAYER_SWEEP_LABELS if label in per_predictor
    }
    pooled_table = {}
    for label, blk in sweep_blocks.items():
        pooled_table[label] = {
            "source_fe_rho": blk["spearman_source_fe"].get("rho"),
            "source_fe_p": blk["spearman_source_fe"].get("p"),
            "fisher_z_avg": blk["fisher_z_avg_per_source"].get("rho_z_avg"),
        }
    # Best-layer-per-source (by |per-source ro| over the sweep set).
    best_per_source: dict[str, dict] = {}
    if sweep_blocks:
        # Pull the source list from any block — they all share the same set.
        any_blk = next(iter(sweep_blocks.values()))
        for src in any_blk["per_source"]:
            best_layer = None
            best_rho = None
            for label, blk in sweep_blocks.items():
                r = blk["per_source"][src].get("rho")
                if r is None:
                    continue
                if best_rho is None or abs(r) > abs(best_rho):
                    best_rho = r
                    best_layer = label
            best_per_source[src] = {"best_layer": best_layer, "rho": best_rho}
    return {
        "pooled_per_layer": pooled_table,
        "best_layer_per_source": best_per_source,
        "layers_included": list(sweep_blocks),
    }


def response_length_confound(cells: list[dict], *, n_boot: int, seed: int) -> dict:
    """Concern #3 / plan §6.5 item 4 — rank correlation between JS and the
    |source - bystander| response-length differential.

    Cells without ``resp_len_diff_abs`` are dropped silently (Phase 4 attaches
    the field whenever both personas appear in the per-persona length map).
    """
    pairs = [
        c
        for c in cells
        if c.get("JS_sym_nats") is not None and c.get("resp_len_diff_abs") is not None
    ]
    if len(pairs) < 4:
        return {
            "n_cells": len(pairs),
            "note": "insufficient_n",
            "rho_js_vs_len_diff": None,
            "flag_length_confound": False,
        }
    js_vals = np.array([c["JS_sym_nats"] for c in pairs])
    len_vals = np.array([c["resp_len_diff_abs"] for c in pairs])
    base = spearman(js_vals, len_vals)
    boot = _spearman_bootstrap_and_permutation(
        js_vals, len_vals, n_boot=n_boot, n_perm=n_boot, seed=seed
    )
    rho = base.get("rho")
    flagged = rho is not None and abs(rho) >= JS_LEN_CONFOUND_FLAG_RHO
    return {
        "n_cells": len(pairs),
        "rho_js_vs_len_diff": rho,
        "p": base.get("p"),
        "ci_low_95": boot.get("ci_low_95"),
        "ci_high_95": boot.get("ci_high_95"),
        "permutation_p": boot.get("permutation_p"),
        "flag_threshold_abs_rho": JS_LEN_CONFOUND_FLAG_RHO,
        "flag_length_confound": flagged,
    }


def main() -> int:  # noqa: C901 — sequential setup + flat result-assembly reads clearer inline
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--n-perm", type=int, default=PERMUTATION_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not PHASE4_PATH.exists():
        raise RuntimeError(f"Phase 4 output missing at {PHASE4_PATH}")
    phase4 = read_json(PHASE4_PATH)
    cells = phase4["cells"]

    # Concern #4 — kill criterion (plan §1): if JS dynamic range is too small,
    # halt + report instead of forcing a regression on a flat predictor.
    js_arr_all = np.array([c["JS_sym_nats"] for c in cells if c.get("JS_sym_nats") is not None])
    js_std = float(js_arr_all.std()) if js_arr_all.size > 1 else 0.0
    js_std_kill = js_arr_all.size > 1 and js_std < JS_STD_KILL_THRESHOLD_NATS
    if js_std_kill:
        kill_payload = {
            "kill_criterion": "js_predictor_dynamic_range_insufficient",
            "js_predictor_dynamic_range_insufficient": True,
            "js_sym_std_nats": js_std,
            "threshold_nats": JS_STD_KILL_THRESHOLD_NATS,
            "n_cells_with_js": int(js_arr_all.size),
            "metadata": reproducibility_metadata(
                {"script": "predictor_jsdiv_470.phase5_regress", "halted": True}
            ),
        }
        write_json(PHASE5_PATH, kill_payload)
        logger.error(
            "KILL CRITERION HIT: JS std = %.6f nats < %.4f nats. "
            "JS predictor has insufficient dynamic range on this panel; refusing to "
            "force a regression on a flat predictor (plan §1).",
            js_std,
            JS_STD_KILL_THRESHOLD_NATS,
        )
        print("\n=== Phase 5 HALTED — JS dynamic range insufficient (plan §1 kill) ===")
        print(f"std(JS_sym_nats) over {js_arr_all.size} cells = {js_std:.6f} nats")
        print(f"Threshold: {JS_STD_KILL_THRESHOLD_NATS:.4f} nats. Halting; no regression run.")
        return 0

    # Build arrays restricted to cells that have the CORE predictors. We gate
    # on the headline + baseline only; per-predictor blocks below skip cells
    # whose own predictor value is None instead of dropping them globally.
    # (Smoke runs on a smaller model that lacks layer-27, which would otherwise
    # collapse the usable set to zero; production has all layers populated.)
    core_keys_required = {
        "delta",
        "source",
        "bystander",
        "cosine_l20_baseline",
        "bystander_base_rate",
    }
    usable = [c for c in cells if all(c.get(k) is not None for k in core_keys_required)]
    dropped = len(cells) - len(usable)
    if dropped:
        logger.warning(
            "Dropped %d/%d cells missing one or more CORE predictors (%s)",
            dropped,
            len(cells),
            sorted(core_keys_required),
        )
    if not usable:
        raise RuntimeError("No cells survive the core-predictor filter; cannot regress.")

    sources = [c["source"] for c in usable]
    delta_vals = np.array([c["delta"] for c in usable], dtype=float)
    base_rate_vals = np.array([c["bystander_base_rate"] for c in usable], dtype=float)

    per_predictor: dict[str, dict] = {}
    for label, key, polarity in PREDICTORS:
        # Sub-filter to cells where THIS predictor is present.
        pred_idx = [i for i, c in enumerate(usable) if c.get(key) is not None]
        if len(pred_idx) < 3:
            per_predictor[label] = {
                "label": label,
                "polarity": polarity,
                "n_cells": len(pred_idx),
                "note": "insufficient_cells_with_predictor",
                "spearman_raw": {"rho": None, "p": None, "n": len(pred_idx)},
                "spearman_source_fe": {"rho": None, "p": None, "n": len(pred_idx)},
                "spearman_source_fe_plus_base_rate": {
                    "rho": None,
                    "p": None,
                    "n": len(pred_idx),
                },
                "per_source": {},
                "fisher_z_avg_per_source": {"rho_z_avg": None, "n_sources": 0},
            }
            continue
        sub_sources = [sources[i] for i in pred_idx]
        sub_vals = np.array([usable[i][key] for i in pred_idx], dtype=float)
        sub_delta = np.array([delta_vals[i] for i in pred_idx], dtype=float)
        sub_base_rate = np.array([base_rate_vals[i] for i in pred_idx], dtype=float)
        per_predictor[label] = regress_predictor(
            label=label,
            polarity=polarity,
            sources=sub_sources,
            predictor_vals=sub_vals,
            delta_vals=sub_delta,
            base_rate_vals=sub_base_rate,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=args.seed,
        )

    # Headline paired-bootstrap Delta-ro: |ro(M_js)| - |ro(cosine_l20)|.
    # Restrict to cells where BOTH predictors are present.
    paired_idx = [
        i
        for i, c in enumerate(usable)
        if c.get(HEADLINE_PREDICTOR_FOR_DELTA) is not None and c.get(BASELINE_FOR_DELTA) is not None
    ]
    if len(paired_idx) >= 10:
        x_js = np.array([usable[i][HEADLINE_PREDICTOR_FOR_DELTA] for i in paired_idx], dtype=float)
        x_cos = np.array([usable[i][BASELINE_FOR_DELTA] for i in paired_idx], dtype=float)
        paired_sources = [sources[i] for i in paired_idx]
        paired_delta = np.array([delta_vals[i] for i in paired_idx], dtype=float)
        delta_rho = paired_bootstrap_delta_rho(
            sources=paired_sources,
            x_a=x_js,
            x_b=x_cos,
            y=paired_delta,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    else:
        delta_rho = {
            "delta_rho_mean": None,
            "ci_low_95": None,
            "ci_high_95": None,
            "n_cells_with_both": len(paired_idx),
            "note": "insufficient_cells_with_both_predictors",
        }

    # Cross-predictor correlation (sanity per §6.5 item 5). Skip pairs where
    # either side is None in too many cells.
    cross_corr = {}
    for a_label, a_key, _ in PREDICTORS:
        cross_corr[a_label] = {}
        for b_label, b_key, _ in PREDICTORS:
            pair_idx = [
                i
                for i, c in enumerate(usable)
                if c.get(a_key) is not None and c.get(b_key) is not None
            ]
            if len(pair_idx) < 3:
                cross_corr[a_label][b_label] = None
                continue
            a_vals = np.array([usable[i][a_key] for i in pair_idx], dtype=float)
            b_vals = np.array([usable[i][b_key] for i in pair_idx], dtype=float)
            r = spearman(a_vals, b_vals)
            cross_corr[a_label][b_label] = r.get("rho")

    # Concern #5: cosine layer-sweep ladder.
    layer_ladder = cosine_layer_ladder(per_predictor)

    # Concern #3: response-length confound (uses the wider `cells` set so we
    # don't filter out cells whose JS+length pair is fine but some OTHER
    # predictor is None).
    len_confound = response_length_confound(cells, n_boot=args.n_boot, seed=args.seed)

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
        "js_sym_std_nats": js_std,
        "js_predictor_dynamic_range_insufficient": False,
        "predictors": per_predictor,
        "paired_bootstrap_delta_rho": {
            "spec": "|rho(M_js)| - |rho(cosine_l20_baseline)| on source-FE-residualized ranks",
            **delta_rho,
        },
        "cosine_layer_ladder": layer_ladder,
        "response_length_confound": len_confound,
        "secondary_diagnostic_bystander_ranks": rank_diagnostic,
        "cross_predictor_spearman": cross_corr,
        "in_seed_extension_buffer_zone": buffer_flag,
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "metadata": reproducibility_metadata({"script": "predictor_jsdiv_470.phase5_regress"}),
    }
    write_json(PHASE5_PATH, summary)
    logger.info("Wrote %s", PHASE5_PATH)

    # Compact stdout table for the human eyeball.
    print(f"\n=== Phase 5 — predictor comparison (n_cells_used={len(usable)}) ===")
    print(f"std(JS_sym_nats) = {js_std:.6f} nats (kill threshold {JS_STD_KILL_THRESHOLD_NATS})")
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

    # Cosine layer ladder readout.
    print("\nCosine layer ladder (source-FE pooled rho per layer):")
    for label, vals in layer_ladder["pooled_per_layer"].items():
        rho = vals.get("source_fe_rho")
        if rho is None:
            print(f"  {label:<28} rho=nan")
        else:
            print(f"  {label:<28} rho={rho:+.4f}")

    # Length confound readout.
    lc = len_confound
    if lc.get("rho_js_vs_len_diff") is not None:
        print(
            f"\nResponse-length confound: rho(JS, |len_src - len_bys|) = "
            f"{lc['rho_js_vs_len_diff']:+.4f}  flag={lc['flag_length_confound']}"
        )

    # Comedian recovery diagnostic (the headline secondary case).
    se = rank_diagnostic.get("software_engineer", {})
    if se:
        print("\nsoftware_engineer -> comedian rank by predictor (smaller = ranked closer):")
        for label, blk in se.items():
            print(
                f"  {label:<28} comedian_rank={blk.get('rank_of_comedian')}  "
                f"top5={blk.get('top_5_bystanders')}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
