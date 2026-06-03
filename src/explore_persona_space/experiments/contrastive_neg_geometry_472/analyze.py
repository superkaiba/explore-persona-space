# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker token " ※" are intentional
"""Task #472 Phase 5 — geometry / count / placement analysis (plan §6).

Reads per-cell×seed trajectory.json + base_panel.json + centroids, then:
  1. Matched-slice DV: per cell×seed, interpolate the held-out DV at the
     checkpoint where source-self ΔG first reaches 8±1 nats (sub-ceiling). A
     cell whose held-out logP saturated before the band contributes NO row to
     the logP regression (dropped, not backfilled with KL).
  2. SEPARATE logP / KL regressions (never mix DV units): pooled OLS
     ``leakage ~ d_source + d_nearest_neg + b_logprob`` over the count-matched
     arms (Near/Far/Spread), cluster-robust SE by probe. Run BOTH with all-neg
     ``d_nearest_neg`` and non-default ``d_nearest_neg_nd``.
  3. qwen_default identification gate (plan §6): share of probes whose nearest
     negative is qwen_default; across-arm SD of d_nearest_neg_nd; barrier/bubble
     admissible only if non-default d moves across arms AND the two fits agree
     in sign.
  4. Collinearity gate: Pearson(d_source, d_nearest_neg) + VIF; |r|>0.6 →
     fall back to tercile-bucket medians. Also Pearson(d_nearest_neg, b_logprob).
  5. Holm multiplicity across {co-primary DV} × {geometry/count/placement}.
  6. Count effects (Spearman + exact permutation) and placement H1
     (Near<Far<No-neg permutation), per DV.
  7. Diagnostics for the analyzer (plan §6.5): raw ΔG-vs-base-prior scatter,
     ΔG-vs-KL rank agreement sub-ceiling, per-cell step-at-matched-slice.
  8. Figures (hero 2-panel + exploratory dump).

Single-negative sub-arms are EXCLUDED from the pooled regression (n_personas=2,
not count-matched) — read only as standalone proximity maps.

CPU / local (regression + plots).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    ALWAYS_INCLUDE_NEGATIVE,
    CELL_SPECS,
    HEADLINE_LAYER,
    MATCHED_SLICE_BAND_NATS,
    MATCHED_SLICE_TARGET_NATS,
    SOURCE_PERSONA,
    SUBCEILING_HEADROOM_NATS,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    cos_to_source as load_cos_to_source,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    load_cos_matrix,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    d_nearest_neg,
    d_source,
    held_out_panel,
    negatives_for_cell,
)

log = logging.getLogger("issue_472.analyze")

# Pooled-regression arms (count-matched, in_pooled=True): anchor(Spread), near, far.
POOLED_CELLS = [c[0] for c in CELL_SPECS if c[5]]
COLLINEARITY_THRESHOLD = 0.6
ID_GATE_SD_FLOOR = 0.02  # min median across-arm SD of d_nearest_neg_nd to admit barrier/bubble.


# ── Matched-slice interpolation ──────────────────────────────────────────────


def _interp_at_slice(
    checkpoints: list[dict],
    target_nats: float = MATCHED_SLICE_TARGET_NATS,
    band: float = MATCHED_SLICE_BAND_NATS,
) -> dict[str, Any] | None:
    """Find/interpolate the held-out DV at the source-self-ΔG matched slice.

    Returns a dict with the interpolated per-probe ``delta_g`` + ``kl`` + the
    step + a ``saturated`` flag, or None if source-self ΔG never reaches the
    band (cell undertrained at the matched slice → no logP row).
    """
    # Source-self ΔG trajectory (mean over Q_eval).
    src = [(ck["frac"], ck["source_self"]["delta_g_mean"], ck) for ck in checkpoints]
    src.sort(key=lambda t: t[0])
    target = target_nats
    # Find first checkpoint pair bracketing the target band (rising series).
    hit_ck = None
    interp_w = None
    from itertools import pairwise

    for (_f0, d0, ck0), (_f1, d1, ck1) in pairwise(src):
        if (d0 < target <= d1) or (d1 < target <= d0):
            # Linear interpolation weight between ck0 and ck1.
            interp_w = (target - d0) / (d1 - d0) if (d1 - d0) != 0 else 0.0
            hit_ck = (ck0, ck1)
            break
    if hit_ck is None:
        # Maybe the last checkpoint is already within the band.
        _last_frac, last_d, last_ck = src[-1]
        if abs(last_d - target) <= band:
            hit_ck = (last_ck, last_ck)
            interp_w = 0.0
        else:
            return None

    ck0, ck1 = hit_ck
    held0 = ck0["held_out"]
    held1 = ck1["held_out"]
    per_probe: dict[str, dict[str, float]] = {}
    saturated_count = 0
    collapsed_count = 0
    total = 0
    for persona in held0:
        deltas, kls, g_logps = [], [], []
        any_collapsed = False
        for q in held0[persona]:
            d0v = held0[persona][q]["delta_g"]
            d1v = held1[persona][q]["delta_g"]
            deltas.append(d0v + interp_w * (d1v - d0v))
            g0 = held0[persona][q]["g_logp"]
            g1 = held1[persona][q]["g_logp"]
            g_logps.append(g0 + interp_w * (g1 - g0))
            k0 = held0[persona][q].get("kl")
            k1 = held1[persona][q].get("kl")
            if k0 is not None and k1 is not None:
                kls.append(k0 + interp_w * (k1 - k0))
            # r_collapsed at EITHER bracketing checkpoint marks the probe degenerate:
            # the trained model's OWN R is marker-spam, so log P(※) is repetition
            # ceiling, NOT graded leakage. Treated like a saturated row (dropped
            # from the graded logP regression) but tracked as a distinct category.
            if held0[persona][q].get("r_collapsed", False) or held1[persona][q].get(
                "r_collapsed", False
            ):
                any_collapsed = True
        total += 1
        # Saturated if held-out g_logp is within HEADROOM of the 0.0 ceiling.
        mean_g = float(np.mean(g_logps))
        is_sat = mean_g > -SUBCEILING_HEADROOM_NATS
        if is_sat:
            saturated_count += 1
        if any_collapsed:
            collapsed_count += 1
        per_probe[persona] = {
            "delta_g": float(np.mean(deltas)),
            "g_logp": mean_g,
            "kl": float(np.mean(kls)) if kls else float("nan"),
            "saturated": is_sat,
            "r_collapsed": any_collapsed,
        }
    return {
        "step": ck1.get("step"),
        "frac": ck1["frac"],
        "interp_w": interp_w,
        "n_saturated_probes": saturated_count,
        "n_collapsed_probes": collapsed_count,
        "n_probes": total,
        "per_probe": per_probe,
    }


# ── Regression ───────────────────────────────────────────────────────────────


def _fit_pooled_ols(rows: list[dict], dv_key: str, dnn_key: str) -> dict[str, Any]:
    """Pooled OLS leakage ~ d_source + <dnn_key> + b_logprob, cluster-robust by probe.

    rows: each {dv: float, d_source: float, dnn: float, dnn_nd: float,
               b_logprob: float, probe: str}. Drops rows with NaN dv or NaN dnn.
    """
    import statsmodels.api as sm

    clean = [
        r
        for r in rows
        if not (np.isnan(r[dv_key]) or np.isnan(r[dnn_key]) or np.isnan(r["b_logprob"]))
    ]
    n = len(clean)
    if n < 8:
        return {"ok": False, "reason": f"too few rows ({n}) after NaN drop", "n": n}
    y = np.array([r[dv_key] for r in clean])
    X = np.column_stack(
        [
            np.ones(n),
            np.array([r["d_source"] for r in clean]),
            np.array([r[dnn_key] for r in clean]),
            np.array([r["b_logprob"] for r in clean]),
        ]
    )
    groups = np.array([r["probe"] for r in clean])
    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    names = ["const", "d_source", dnn_key, "b_logprob"]
    return {
        "ok": True,
        "n": n,
        "n_clusters": len(set(groups.tolist())),
        "coef": {names[i]: float(res.params[i]) for i in range(len(names))},
        "se": {names[i]: float(res.bse[i]) for i in range(len(names))},
        "pvalue": {names[i]: float(res.pvalues[i]) for i in range(len(names))},
        "rsquared": float(res.rsquared),
    }


def _vif(rows: list[dict], dnn_key: str) -> dict[str, float]:
    """Variance-inflation factors for [d_source, dnn_key, b_logprob]."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    clean = [
        r
        for r in rows
        if not (np.isnan(r[dnn_key]) or np.isnan(r["b_logprob"]) or np.isnan(r["d_source"]))
    ]
    if len(clean) < 5:
        return {}
    X = np.column_stack(
        [
            np.ones(len(clean)),
            np.array([r["d_source"] for r in clean]),
            np.array([r[dnn_key] for r in clean]),
            np.array([r["b_logprob"] for r in clean]),
        ]
    )
    names = ["const", "d_source", dnn_key, "b_logprob"]
    return {names[i]: float(variance_inflation_factor(X, i)) for i in range(1, 4)}


def _pearson(a: list[float], b: list[float]) -> float:
    from scipy.stats import pearsonr

    pairs = [(x, y) for x, y in zip(a, b, strict=True) if not (np.isnan(x) or np.isnan(y))]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs, strict=True)
    return float(pearsonr(xs, ys)[0])


def holm_correction(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni step-down across the test family (plan §6 multiplicity)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    prev_reject = True
    for rank, (name, p) in enumerate(items):
        thresh = alpha / (m - rank)
        reject = bool(prev_reject and p < thresh)
        prev_reject = reject
        out[name] = {"p": p, "holm_threshold": thresh, "reject_null": reject}
    return out


# ── Permutation / Spearman for count + placement ─────────────────────────────


def _spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def _exact_permutation_monotone(
    level_means: list[float], n_perms_cap: int = 5000
) -> dict[str, Any]:
    """Exact-permutation null on the monotone range statistic over ordered levels.

    Enumerates label permutations of the level means (small: 3 levels). Statistic
    = range = max − min preserving monotone direction. Returns the empirical
    one-sided p of the observed monotone range under random level assignment.
    """
    from itertools import permutations

    arr = np.array(level_means, dtype=float)
    if len(arr) < 2 or np.any(np.isnan(arr)):
        return {"ok": False, "reason": "too few / NaN levels"}
    obs_range = float(arr.max() - arr.min())
    obs_monotone = bool(np.all(np.diff(arr) >= 0) or np.all(np.diff(arr) <= 0))
    perms = list(permutations(range(len(arr))))
    if len(perms) > n_perms_cap:
        perms = perms[:n_perms_cap]
    count_ge = 0
    for perm in perms:
        p_arr = arr[list(perm)]
        rng = float(p_arr.max() - p_arr.min())
        mono = bool(np.all(np.diff(p_arr) >= 0) or np.all(np.diff(p_arr) <= 0))
        if mono and rng >= obs_range:
            count_ge += 1
    return {
        "ok": True,
        "observed_range": obs_range,
        "observed_monotone": obs_monotone,
        "empirical_p_value": count_ge / len(perms),
        "n_perms_enumerated": len(perms),
        "note": "exact enumeration; do NOT claim 10k independent draws (plan §6)",
    }


# ── Main analysis ────────────────────────────────────────────────────────────


def run_analysis(  # noqa: C901 - linear multi-block analysis
    *,
    slab_root: Path,
    base_panel_path: Path,
    figures_dir: Path,
    centroids_dir: Path,
    seeds: list[int],
    layer: int = HEADLINE_LAYER,
    source: str = SOURCE_PERSONA,
) -> dict[str, Any]:
    """Run the full geometry/count/placement analysis. Returns the summary dict.

    Writes ``<slab_root>/analyze_summary.json`` + figures under ``figures_dir``.
    """
    cts = load_cos_to_source(layer, source, centroids_dir)
    cos_matrix, _names = load_cos_matrix(layer, centroids_dir)
    panel = held_out_panel(cts, source=source)
    log.info("Held-out panel: %d probes (layer %d)", len(panel), layer)

    base_panel = json.loads(base_panel_path.read_text())
    b_logprob = base_panel["mean_per_persona_b_logprob"]

    # ── Load matched-slice DV per cell×seed. ─────────────────────────────────
    # cell -> seed -> matched-slice dict (per-probe delta_g, kl, saturated).
    matched: dict[str, dict[int, dict | None]] = {}
    cell_arm_negatives: dict[str, list[str]] = {}
    for slug, _name, _placement, _np, _ex, _pooled in CELL_SPECS:
        cell_arm_negatives[slug] = negatives_for_cell(slug, cts, source=source)
        matched[slug] = {}
        for seed in seeds:
            traj_path = slab_root / f"{slug}_seed{seed}" / "trajectory.json"
            if not traj_path.exists():
                log.warning(
                    "Trajectory missing: %s (cell %s seed %d skipped)", traj_path, slug, seed
                )
                matched[slug][seed] = None
                continue
            traj = json.loads(traj_path.read_text())
            matched[slug][seed] = _interp_at_slice(traj["checkpoints"])

    # ── Build pooled-regression rows (logP + KL separately). ─────────────────
    # Rows over count-matched pooled arms × held-out probes × seeds.
    logp_rows: list[dict] = []
    kl_rows: list[dict] = []
    nearest_default_count = 0
    nearest_total = 0
    dnn_nd_by_probe_across_arms: dict[str, list[float]] = {p: [] for p in panel}
    for slug in POOLED_CELLS:
        negs = cell_arm_negatives[slug]
        for seed in seeds:
            ms = matched[slug].get(seed)
            if ms is None:
                continue
            for probe in panel:
                if probe not in ms["per_probe"]:
                    continue
                pp = ms["per_probe"][probe]
                ds = d_source(probe, cts)
                dnn = d_nearest_neg(probe, negs, cos_matrix, exclude_default=False)
                dnn_nd = d_nearest_neg(probe, negs, cos_matrix, exclude_default=True)
                dnn_nd_by_probe_across_arms[probe].append(dnn_nd)
                # Identification-gate bookkeeping: is the nearest negative the default?
                if negs:
                    nearest = min(negs, key=lambda nn: 1.0 - cos_matrix[probe][nn])
                    nearest_total += 1
                    if nearest == ALWAYS_INCLUDE_NEGATIVE:
                        nearest_default_count += 1
                base_b = b_logprob.get(probe, float("nan"))
                row_common = {
                    "cell": slug,
                    "seed": seed,
                    "probe": probe,
                    "d_source": ds,
                    "dnn": dnn,
                    "dnn_nd": dnn_nd,
                    "b_logprob": base_b,
                }
                # logP row only if NOT saturated AND NOT R-collapsed at the matched
                # slice. A collapsed-R probe (the model's own R is marker-spam) is a
                # degenerate max-leakage case, not graded leakage — drop it from the
                # graded logP regression (plan §4.6 + the #448 saturation lesson),
                # exactly like a saturated row.
                if not pp["saturated"] and not pp.get("r_collapsed", False):
                    logp_rows.append({**row_common, "logp": pp["delta_g"]})
                # KL row always (read at the same matched slice).
                kl_rows.append({**row_common, "kl": pp["kl"]})

    # ── Identification gate (qwen_default dominance). ────────────────────────
    default_share = (nearest_default_count / nearest_total) if nearest_total else float("nan")
    across_arm_sd = [
        float(np.std(v))
        for v in dnn_nd_by_probe_across_arms.values()
        if len(v) >= 2 and not np.all(np.isnan(v))
    ]
    median_across_arm_sd = float(np.median(across_arm_sd)) if across_arm_sd else float("nan")

    # ── logP regression (construct-of-record), dual (all-neg + non-default). ─
    logp_fit_allneg = _fit_pooled_ols(logp_rows, "logp", "dnn")
    logp_fit_nondefault = _fit_pooled_ols(logp_rows, "logp", "dnn_nd")
    logp_vif = _vif(logp_rows, "dnn")
    # ── KL regression (backstop), dual. ──────────────────────────────────────
    kl_fit_allneg = _fit_pooled_ols(kl_rows, "kl", "dnn")
    kl_fit_nondefault = _fit_pooled_ols(kl_rows, "kl", "dnn_nd")

    # ── Collinearity gate. ───────────────────────────────────────────────────
    r_ds_dnn = _pearson([r["d_source"] for r in logp_rows], [r["dnn"] for r in logp_rows])
    r_dnn_b = _pearson([r["dnn"] for r in logp_rows], [r["b_logprob"] for r in logp_rows])
    collinearity_ok = (not np.isnan(r_ds_dnn)) and abs(r_ds_dnn) <= COLLINEARITY_THRESHOLD

    # Identification admissibility: non-default d moves across arms AND all-neg
    # and non-default fits agree in sign on d_source AND d_nearest_neg.
    def _sign(x: float) -> int:
        return 0 if (x is None or np.isnan(x)) else (1 if x > 0 else -1)

    fits_agree = False
    if logp_fit_allneg.get("ok") and logp_fit_nondefault.get("ok"):
        fits_agree = _sign(logp_fit_allneg["coef"]["d_source"]) == _sign(
            logp_fit_nondefault["coef"]["d_source"]
        ) and _sign(logp_fit_allneg["coef"]["dnn"]) == _sign(logp_fit_nondefault["coef"]["dnn_nd"])
    id_gate_ok = (
        (not np.isnan(median_across_arm_sd))
        and median_across_arm_sd >= ID_GATE_SD_FLOOR
        and fits_agree
    )

    # ── Holm multiplicity over {co-primary DV} × {geometry/count/placement}. ─
    # Build the family p-values: geometry uses the logP construct-of-record
    # d_source + d_nearest_neg partials; count + placement use permutation p's.
    family_p: dict[str, float] = {}
    if logp_fit_allneg.get("ok"):
        family_p["logp_geometry_d_source"] = logp_fit_allneg["pvalue"]["d_source"]
        family_p["logp_geometry_d_nearest_neg"] = logp_fit_allneg["pvalue"]["dnn"]

    # ── Count effects (per DV): negex {100,200,400} + negp {2,4,8}. ──────────
    def _held_out_mean_dv(slug: str, seed: int, dv: str) -> float:
        ms = matched[slug].get(seed)
        if ms is None:
            return float("nan")
        vals = []
        for probe in panel:
            if probe in ms["per_probe"]:
                pp = ms["per_probe"][probe]
                # For the logP count/placement families, drop degenerate probes
                # (saturated OR R-collapsed = marker-spam) EXACTLY like the graded
                # geometry regression above — otherwise differential collapse by
                # condition re-enters these headline means as graded max-leakage
                # values and biases the Holm p-values. KL is non-saturating and is
                # kept for all probes (matches the geometry "KL row always" rule).
                if dv == "logp" and (pp["saturated"] or pp.get("r_collapsed", False)):
                    continue
                v = pp["delta_g" if dv == "logp" else "kl"]
                if not np.isnan(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    def _count_effect(cells_ordered: list[str], levels: list[float], dv: str) -> dict[str, Any]:
        # Mean over seeds of held-out-mean DV per level.
        level_means = []
        for slug in cells_ordered:
            seed_vals = [_held_out_mean_dv(slug, s, dv) for s in seeds]
            seed_vals = [v for v in seed_vals if not np.isnan(v)]
            level_means.append(float(np.mean(seed_vals)) if seed_vals else float("nan"))
        sp = _spearman(levels, level_means)
        perm = _exact_permutation_monotone(level_means)
        return {"levels": levels, "level_means": level_means, "spearman": sp, "permutation": perm}

    count_negex = _count_effect(
        ["c472_negex_100", "c472_anchor", "c472_negex_400"], [100, 200, 400], "logp"
    )
    count_negp = _count_effect(["c472_negp_2", "c472_anchor", "c472_negp_8"], [2, 4, 8], "logp")
    count_negex_kl = _count_effect(
        ["c472_negex_100", "c472_anchor", "c472_negex_400"], [100, 200, 400], "kl"
    )
    count_negp_kl = _count_effect(["c472_negp_2", "c472_anchor", "c472_negp_8"], [2, 4, 8], "kl")
    if count_negex["permutation"].get("ok"):
        family_p["logp_count_negex"] = count_negex["permutation"]["empirical_p_value"]
    if count_negp["permutation"].get("ok"):
        family_p["logp_count_negp"] = count_negp["permutation"]["empirical_p_value"]

    # ── Placement H1: Near vs Far vs No-neg held-out-mean DV. ────────────────
    placement_means = {
        arm: float(
            np.mean(
                [v for s in seeds for v in [_held_out_mean_dv(arm, s, "logp")] if not np.isnan(v)]
            )
        )
        for arm in ["c472_near", "c472_anchor", "c472_far", "c472_noneg"]
    }
    placement = {
        "near": placement_means.get("c472_near"),
        "spread_anchor": placement_means.get("c472_anchor"),
        "far": placement_means.get("c472_far"),
        "no_neg": placement_means.get("c472_noneg"),
        "near_lt_far": (
            placement_means.get("c472_near", float("nan"))
            < placement_means.get("c472_far", float("nan"))
        ),
    }
    placement_perm = _exact_permutation_monotone(
        [
            placement_means.get("c472_near", float("nan")),
            placement_means.get("c472_far", float("nan")),
            placement_means.get("c472_noneg", float("nan")),
        ]
    )
    if placement_perm.get("ok"):
        family_p["logp_placement_near_far_noneg"] = placement_perm["empirical_p_value"]

    holm = holm_correction(family_p) if family_p else {}

    # ── Diagnostics for the analyzer (plan §6.5). ────────────────────────────
    # ΔG vs base prior scatter (#448 artifact check).
    dg_b_scatter = [
        {"probe": r["probe"], "delta_g": r["logp"], "b_logprob": r["b_logprob"]} for r in logp_rows
    ]
    # ΔG vs KL rank agreement on sub-ceiling rows (matched probe×arm×seed).
    # Build paired (delta_g, kl) for the same (cell, seed, probe) where logp row exists.
    logp_index = {(r["cell"], r["seed"], r["probe"]): r["logp"] for r in logp_rows}
    paired_dg, paired_kl = [], []
    for r in kl_rows:
        key = (r["cell"], r["seed"], r["probe"])
        if key in logp_index and not np.isnan(r["kl"]):
            paired_dg.append(logp_index[key])
            paired_kl.append(r["kl"])
    dg_kl_spearman = _spearman(paired_dg, paired_kl) if len(paired_dg) >= 3 else float("nan")
    # Per-cell step-at-matched-slice (placement-vs-train-speed diagnostic).
    step_at_slice = {
        slug: {str(s): (matched[slug][s]["step"] if matched[slug].get(s) else None) for s in seeds}
        for slug in [c[0] for c in CELL_SPECS]
    }

    # ── Validity gate: every cell clears source-self ΔG ≥ 5 nats + ≥1 sub-ceiling ck.
    validity: dict[str, dict[str, Any]] = {}
    for slug, _name, _placement, _np, _ex, _pooled in CELL_SPECS:
        per_seed = {}
        for seed in seeds:
            traj_path = slab_root / f"{slug}_seed{seed}" / "trajectory.json"
            if not traj_path.exists():
                per_seed[str(seed)] = {"present": False}
                continue
            traj = json.loads(traj_path.read_text())
            max_src = max(ck["source_self"]["delta_g_mean"] for ck in traj["checkpoints"])

            def _ck_mean_held_out_g_logp(ck: dict) -> float:
                vals = [
                    held["g_logp"] for per_q in ck["held_out"].values() for held in per_q.values()
                ]
                return float(np.mean(vals)) if vals else 0.0

            has_subceiling = any(
                _ck_mean_held_out_g_logp(ck) < -SUBCEILING_HEADROOM_NATS
                for ck in traj["checkpoints"]
            )
            per_seed[str(seed)] = {
                "present": True,
                "max_source_self_delta_g": float(max_src),
                "clears_5nat_floor": bool(max_src >= 5.0),
                "has_subceiling_checkpoint": bool(has_subceiling),
            }
        validity[slug] = per_seed

    # ── Barrier vs bubble verdict (logP construct-of-record). ────────────────
    verdict = _barrier_bubble_verdict(logp_fit_allneg, collinearity_ok, id_gate_ok, holm)

    summary: dict[str, Any] = {
        "schema_version": "i472_v1",
        "layer": layer,
        "source": source,
        "n_held_out_probes": len(panel),
        "held_out_panel": panel,
        "pooled_cells": POOLED_CELLS,
        "n_logp_rows": len(logp_rows),
        "n_kl_rows": len(kl_rows),
        "matched_slice": {
            "target_nats": MATCHED_SLICE_TARGET_NATS,
            "band_nats": MATCHED_SLICE_BAND_NATS,
            "step_at_slice_per_cell": step_at_slice,
            "collapse_at_slice_per_cell": {
                slug: {
                    str(s): (
                        {
                            "n_collapsed_probes": matched[slug][s].get("n_collapsed_probes"),
                            "n_saturated_probes": matched[slug][s].get("n_saturated_probes"),
                            "n_probes": matched[slug][s].get("n_probes"),
                        }
                        if matched[slug].get(s)
                        else None
                    )
                    for s in seeds
                }
                for slug in [c[0] for c in CELL_SPECS]
            },
        },
        "logp_regression": {
            "all_neg": logp_fit_allneg,
            "non_default": logp_fit_nondefault,
            "vif": logp_vif,
        },
        "kl_regression": {"all_neg": kl_fit_allneg, "non_default": kl_fit_nondefault},
        "identification_gate": {
            "qwen_default_nearest_share": default_share,
            "median_across_arm_sd_dnn_nd": median_across_arm_sd,
            "sd_floor": ID_GATE_SD_FLOOR,
            "fits_agree_sign": fits_agree,
            "admissible": id_gate_ok,
        },
        "collinearity_gate": {
            "pearson_d_source_d_nearest_neg": r_ds_dnn,
            "pearson_d_nearest_neg_b_logprob": r_dnn_b,
            "threshold": COLLINEARITY_THRESHOLD,
            "ok": collinearity_ok,
        },
        "holm_multiplicity": holm,
        "count_effects": {
            "negex_logp": count_negex,
            "negp_logp": count_negp,
            "negex_kl": count_negex_kl,
            "negp_kl": count_negp_kl,
        },
        "placement_h1": {"means": placement, "permutation": placement_perm},
        "diagnostics": {
            "delta_g_vs_b_logprob_scatter": dg_b_scatter,
            "delta_g_kl_spearman_subceiling": dg_kl_spearman,
            "step_at_matched_slice": step_at_slice,
        },
        "validity_gate": validity,
        "barrier_bubble_verdict": verdict,
        "single_neg_note": (
            "c472_single_near / c472_single_far EXCLUDED from pooled regression "
            "(n_personas=2, not count-matched); read as standalone proximity maps only "
            "(plan §4.4/§6)."
        ),
    }

    slab_root.mkdir(parents=True, exist_ok=True)
    out_path = slab_root / "analyze_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("[phase=analyze] Wrote %s (verdict=%s)", out_path, verdict.get("call"))

    # ── Figures. ─────────────────────────────────────────────────────────────
    try:
        _make_figures(
            summary,
            logp_rows,
            kl_rows,
            matched,
            panel,
            seeds,
            cts,
            cos_matrix,
            cell_arm_negatives,
            figures_dir,
            source,
        )
    except Exception:
        log.exception("Figure generation failed (analysis JSON still written).")

    return summary


def _barrier_bubble_verdict(
    logp_fit: dict, collinearity_ok: bool, id_gate_ok: bool, holm: dict
) -> dict[str, Any]:
    """Apply the barrier/bubble decision logic (plan §6 / §13 / §14)."""
    if not logp_fit.get("ok"):
        return {
            "call": "indeterminate",
            "reason": "logp regression did not fit (too few sub-ceiling rows)",
        }
    if not collinearity_ok:
        return {"call": "indistinguishable", "reason": "collinearity gate failed (|r|>0.6)"}
    if not id_gate_ok:
        return {
            "call": "indistinguishable",
            "reason": "identification gate failed (qwen_default dominance / fits disagree)",
        }
    ds = logp_fit["coef"]["d_source"]
    dnn = logp_fit["coef"]["dnn"]
    ds_sig = holm.get("logp_geometry_d_source", {}).get("reject_null", False)
    dnn_sig = holm.get("logp_geometry_d_nearest_neg", {}).get("reject_null", False)
    if ds_sig and ds > 0 and not dnn_sig:
        return {
            "call": "barrier",
            "reason": "β(d_source)>0 Holm-sig; β(d_nearest_neg)≈0",
            "d_source": ds,
            "d_nearest_neg": dnn,
        }
    if dnn_sig and dnn > 0 and not ds_sig:
        return {
            "call": "bubble",
            "reason": "β(d_nearest_neg)>0 Holm-sig; β(d_source)≈0",
            "d_source": ds,
            "d_nearest_neg": dnn,
        }
    if ds_sig and dnn_sig:
        return {
            "call": "both",
            "reason": "both partials Holm-sig",
            "d_source": ds,
            "d_nearest_neg": dnn,
        }
    return {
        "call": "neither",
        "reason": "no Holm-sig geometry partial",
        "d_source": ds,
        "d_nearest_neg": dnn,
    }


def _make_figures(
    summary,
    logp_rows,
    kl_rows,
    matched,
    panel,
    seeds,
    cts,
    cos_matrix,
    cell_arm_negatives,
    figures_dir: Path,
    source: str,
) -> None:
    """Hero 2-panel + exploratory dump (plan §6 figures). Plain-English labels."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    arm_label = {"c472_anchor": "Spread", "c472_near": "Near", "c472_far": "Far"}

    # ── Hero left: held-out leakage vs distance, one line per placement arm. ─
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2))
    for slug in POOLED_CELLS:
        xs, ys = [], []
        for r in logp_rows:
            if r["cell"] == slug:
                xs.append(r["d_source"])
                ys.append(r["logp"])
        if xs:
            axl.scatter(xs, ys, s=14, alpha=0.5, label=arm_label.get(slug, slug))
            if len(xs) >= 2:
                coef = np.polyfit(xs, ys, 1)
                xline = np.linspace(min(xs), max(xs), 20)
                axl.plot(xline, np.polyval(coef, xline), lw=1.5)
    axl.set_xlabel("Distance from held-out persona to source")
    axl.set_ylabel("Held-out marker leakage (ΔG, nats)")
    axl.set_title("Leakage vs distance, by negative placement")
    axl.legend(fontsize=8)

    # ── Hero right: partial coefficients d_source vs d_nearest_neg with CIs. ─
    fit = summary["logp_regression"]["all_neg"]
    if fit.get("ok"):
        coefs = [fit["coef"]["d_source"], fit["coef"]["dnn"]]
        ses = [fit["se"]["d_source"], fit["se"]["dnn"]]
        axr.bar(
            [0, 1], coefs, yerr=[1.96 * s for s in ses], capsize=5, color=["#0072B2", "#D55E00"]
        )
        axr.axhline(0, color="k", lw=0.8)
        axr.set_xticks([0, 1])
        axr.set_xticklabels(["dist-to-source\n(barrier)", "dist-to-nearest-neg\n(bubble)"])
        axr.set_ylabel("Partial regression coefficient (±95% CI)")
        axr.set_title(f"Barrier vs bubble: {summary['barrier_bubble_verdict'].get('call')}")
    else:
        axr.text(
            0.5,
            0.5,
            "logP regression did not fit\n(too few sub-ceiling rows)",
            ha="center",
            va="center",
            transform=axr.transAxes,
        )
    fig.tight_layout()
    savefig_paper(fig, "hero_barrier_vs_bubble", dir=str(figures_dir))
    plt.close(fig)

    # ── Exploratory: ΔG vs base-prior scatter (the #448 artifact check). ─────
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    dg = [d["delta_g"] for d in summary["diagnostics"]["delta_g_vs_b_logprob_scatter"]]
    bl = [d["b_logprob"] for d in summary["diagnostics"]["delta_g_vs_b_logprob_scatter"]]
    ax2.scatter(bl, dg, s=14, alpha=0.5, color="#009E73")
    ax2.set_xlabel("Base-model marker prior (b_logprob, nats)")
    ax2.set_ylabel("Held-out leakage (ΔG, nats)")
    ax2.set_title("ΔG vs base prior (the #448 artifact check)")
    fig2.tight_layout()
    savefig_paper(fig2, "exploratory_dg_vs_base_prior", dir=str(figures_dir))
    plt.close(fig2)

    # ── Exploratory: count-knob bars (negex + negp), logP. ───────────────────
    fig3, (a3l, a3r) = plt.subplots(1, 2, figsize=(10, 4))
    ce = summary["count_effects"]
    a3l.bar(
        [str(x) for x in ce["negex_logp"]["levels"]],
        ce["negex_logp"]["level_means"],
        color="#0072B2",
    )
    a3l.set_xlabel("Negative examples / persona")
    a3l.set_ylabel("Held-out mean leakage (ΔG, nats)")
    a3l.set_title("Count: negative examples")
    a3r.bar(
        [str(x) for x in ce["negp_logp"]["levels"]], ce["negp_logp"]["level_means"], color="#D55E00"
    )
    a3r.set_xlabel("Number of negative personas")
    a3r.set_title("Count: negative personas")
    fig3.tight_layout()
    savefig_paper(fig3, "exploratory_count_knobs", dir=str(figures_dir))
    plt.close(fig3)

    log.info("Figures written to %s", figures_dir)
