# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ρ ΔG intentional
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


# ── Task #477 extensions (called from scripts/i477_phase_analyze.py) ─────────
#
# These functions operate on the #477 main-cell + implant-only-axis results
# (NOT on the #472 trajectory.json shape). #477 cells follow the schema:
#   {"cell": str, "seed": int, "count": int, "lr": float,
#    "source_self_delta_g_at_last_ckpt": float,
#    "source_emission_p_at_last_ckpt": float,
#    "mean_bystander_delta_g": float,
#    "step_at_last_ckpt": int}
# kept_cells / implant_only_cells are lists of these dicts (already filtered
# through calibrate.validity_gate).
#
# Discipline (plan §6 "Analysis & interpretation discipline" items 1-8):
#   - return BOTH ρ(count, DV-A | DV-B) and ρ(count, DV-A | DV-B, step);
#   - count-STRATIFIED bootstrap (resample within each count level), percentile CI;
#   - the ≥3/4 coverage guard returns a clear "n<3 count levels — descriptive
#     only" sentinel instead of computing a partial-ρ on n=4 cells × 1 count level.


def _ols_residuals(y: list[float], X: list[list[float]]) -> list[float]:
    """OLS residuals of y on [const, *X]. Returns y - X β̂ (lstsq)."""
    import numpy as np

    y_arr = np.asarray(y, dtype=float)
    X_arr = np.column_stack([np.ones(len(y_arr)), *[np.asarray(c, dtype=float) for c in X]])
    coef, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
    return (y_arr - X_arr @ coef).tolist()


def _stratified_bootstrap_partial_spearman(
    counts: list[int],
    bystander: list[float],
    implant: list[float],
    *,
    n_resamples: int = 2000,
    ci_level: float = 0.90,
    seed: int = 42,
    extra_controls: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Count-stratified bootstrap CI for partial Spearman.

    Plan §6 discipline item 4: resample WITHIN each count level (preserves the
    n-level structure; a naïve i.i.d. n=8 resample produces degenerate samples
    missing a count level). Returns the percentile CI.

    Args:
        counts: per-cell count level (int).
        bystander: per-cell mean bystander ΔG.
        implant: per-cell source-self ΔG (the partialled covariate).
        n_resamples: bootstrap iterations.
        ci_level: e.g. 0.90 → 5th and 95th percentiles.
        seed: RNG seed.
        extra_controls: optional extra covariates to partial out alongside
            implant (e.g. step). Each is a list of per-cell values aligned with
            counts/bystander/implant.

    Returns:
        {"ci_low": float, "ci_high": float, "median": float, "n_resamples": int,
         "ci_level": float, "n_count_levels": int}.
    """
    import numpy as np
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    # Group cell indices by count level.
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(counts):
        groups.setdefault(c, []).append(i)

    rhos: list[float] = []
    for _ in range(n_resamples):
        idx: list[int] = []
        for _level, level_idx in groups.items():
            # Resample WITH replacement within this count level.
            picks = rng.integers(0, len(level_idx), size=len(level_idx))
            idx.extend(level_idx[p] for p in picks)
        c_arr = [counts[i] for i in idx]
        y_arr = [bystander[i] for i in idx]
        x_arr = [implant[i] for i in idx]
        controls = [[col[i] for i in idx] for col in (extra_controls or [])]
        try:
            y_res = _ols_residuals(y_arr, [x_arr, *controls])
            c_res = _ols_residuals([float(v) for v in c_arr], [x_arr, *controls])
            rho = float(spearmanr(c_res, y_res).correlation)
            if not np.isnan(rho):
                rhos.append(rho)
        except (np.linalg.LinAlgError, ValueError):
            continue

    if not rhos:
        return {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "median": float("nan"),
            "n_resamples": 0,
            "ci_level": ci_level,
            "n_count_levels": len(groups),
            "note": "all bootstrap resamples failed (singular partialling matrix)",
        }
    alpha = (1.0 - ci_level) / 2.0
    return {
        "ci_low": float(np.quantile(rhos, alpha)),
        "ci_high": float(np.quantile(rhos, 1.0 - alpha)),
        "median": float(np.median(rhos)),
        "n_resamples": len(rhos),
        "ci_level": float(ci_level),
        "n_count_levels": len(groups),
    }


def partial_spearman_count_given_implant(
    kept_cells: list[dict],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
    min_count_levels: int = 3,
) -> dict[str, Any]:
    """ρ(count, mean-bystander-ΔG | source-self-ΔG) — the #477 H1 headline.

    Plan §6 discipline items 1, 4, 5:
      * report BOTH the partialling-on-implant ρ and the robustness
        ρ(count, DV-A | DV-B, step);
      * count-stratified bootstrap 90% CI;
      * coverage guard: refuses to compute if fewer than min_count_levels (3)
        distinct count levels survive the gate; returns the "descriptive-only"
        sentinel instead.

    Returns:
        {"n": int, "n_count_levels": int, "kept_counts": list[int],
         "rho_given_implant": float, "p_given_implant": float,
         "bootstrap_given_implant": {...},
         "rho_given_implant_and_step": float, "p_given_implant_and_step": float,
         "bootstrap_given_implant_and_step": {...},
         "per_seed_sign": dict, "interpretable": bool, "note": str}
        OR a {"interpretable": False, "note": "..."} sentinel if guard trips.
    """
    import numpy as np
    from scipy.stats import spearmanr

    # Defensive guard: H1 partial Spearman is over MAIN-PHASE cells ONLY.
    # The #477 implant-sweep phase shares the count axis (all at ANCHOR_COUNT=4)
    # but should NEVER be pooled into H1 — it would inject 6 extra count=4
    # points and bias the partial. Cells missing "phase" are treated as
    # non-main (fail loud rather than silently pool unknown-provenance cells).
    for c in kept_cells:
        phase = c.get("phase")
        if phase != "main":
            raise AssertionError(
                f"partial_spearman_count_given_implant got a non-main-phase cell: "
                f"cell={c.get('cell', '<unknown>')!r}, seed={c.get('seed')}, "
                f"phase={phase!r}. H1 partial Spearman must be computed over "
                f"main-phase cells only — pooling implant-sweep / calibration "
                f"cells biases the partial. Filter kept_cells to phase=='main' "
                f"upstream (the dispatcher's main_results carries phase='main')."
            )

    counts = [int(c["count"]) for c in kept_cells]
    distinct_counts = sorted(set(counts))
    n = len(kept_cells)

    if len(distinct_counts) < min_count_levels:
        return {
            "interpretable": False,
            "n": n,
            "n_count_levels": len(distinct_counts),
            "kept_counts": distinct_counts,
            "note": (
                f"n_count_levels={len(distinct_counts)} < min_count_levels="
                f"{min_count_levels} — coverage floor violated (plan §6 discipline "
                f"item 5). Partial Spearman uninterpretable at this coverage; the "
                f"H1 headline must be a DESCRIPTIVE read of per-cell F3 means, not "
                f"a partial-ρ. Excluded counts (gated-out): see calibrate.validity_gate "
                f"output upstream."
            ),
        }

    bystander = [float(c["mean_bystander_delta_g"]) for c in kept_cells]
    implant = [float(c["source_self_delta_g_at_last_ckpt"]) for c in kept_cells]
    step = [int(c.get("step_at_last_ckpt", 0)) for c in kept_cells]

    # Headline: partial out implant only.
    y_resid_i = _ols_residuals(bystander, [implant])
    c_resid_i = _ols_residuals([float(v) for v in counts], [implant])
    sr_i = spearmanr(c_resid_i, y_resid_i)
    boot_i = _stratified_bootstrap_partial_spearman(
        counts, bystander, implant, n_resamples=n_bootstrap, seed=bootstrap_seed
    )

    # Robustness: partial out implant AND step.
    y_resid_is = _ols_residuals(bystander, [implant, [float(v) for v in step]])
    c_resid_is = _ols_residuals([float(v) for v in counts], [implant, [float(v) for v in step]])
    sr_is = spearmanr(c_resid_is, y_resid_is)
    boot_is = _stratified_bootstrap_partial_spearman(
        counts,
        bystander,
        implant,
        n_resamples=n_bootstrap,
        seed=bootstrap_seed + 1,
        extra_controls=[[float(v) for v in step]],
    )

    # Per-seed sign — plan §6 item 7 (seed 137 match-band misses → downgrade).
    seeds_in_kept = sorted({int(c["seed"]) for c in kept_cells})
    per_seed_sign: dict[str, dict] = {}
    for s in seeds_in_kept:
        sub = [c for c in kept_cells if int(c["seed"]) == s]
        if len({int(c["count"]) for c in sub}) < 2:
            per_seed_sign[str(s)] = {
                "n": len(sub),
                "n_count_levels": len({int(c["count"]) for c in sub}),
                "sign": None,
                "note": "fewer than 2 count levels at this seed; sign not defined",
            }
            continue
        # Raw (not partialled) sign at this seed, as a robustness check on the
        # pooled sign — partialling implant inside a single-seed n=4 slice is
        # too unstable to report independently.
        sub_counts = [int(c["count"]) for c in sub]
        sub_bystander = [float(c["mean_bystander_delta_g"]) for c in sub]
        sr_seed = spearmanr(sub_counts, sub_bystander)
        rho_seed = float(sr_seed.correlation) if not np.isnan(sr_seed.correlation) else 0.0
        per_seed_sign[str(s)] = {
            "n": len(sub),
            "n_count_levels": len({int(c["count"]) for c in sub}),
            "rho_raw_count_bystander": rho_seed,
            "sign": int(np.sign(rho_seed)) if rho_seed != 0 else 0,
        }

    pooled_sign = (
        int(np.sign(sr_i.correlation))
        if not np.isnan(sr_i.correlation) and sr_i.correlation != 0
        else 0
    )
    sign_stable_across_seeds = all(
        (v.get("sign") is None) or v.get("sign") == pooled_sign for v in per_seed_sign.values()
    )

    return {
        "interpretable": True,
        "n": n,
        "n_count_levels": len(distinct_counts),
        "kept_counts": distinct_counts,
        "rho_given_implant": float(sr_i.correlation),
        "p_given_implant": float(sr_i.pvalue),
        "bootstrap_given_implant": boot_i,
        "rho_given_implant_and_step": float(sr_is.correlation),
        "p_given_implant_and_step": float(sr_is.pvalue),
        "bootstrap_given_implant_and_step": boot_is,
        "per_seed_sign": per_seed_sign,
        "pooled_sign": pooled_sign,
        "sign_stable_across_seeds": sign_stable_across_seeds,
        "note": (
            "Plan §6 discipline item 1: report BOTH partials. Item 3: this ρ is a "
            "SUMMARY of F3, not an independent test — weight F3 over ρ in the "
            "write-up. Item 4: count-stratified bootstrap CI. Item 7: if any seed "
            "disagrees, downgrade to indeterminate."
        ),
    }


def implant_only_axis_spearman(implant_only_cells: list[dict]) -> dict[str, Any]:
    """ρ(source-self-ΔG, mean-bystander-ΔG) — the #477 H2 secondary.

    Plan §6 discipline item 2: report H2 BEFORE H1; gates H1's interpretability.
    If |ρ_H2| ≤ 0.40 there is no implant axis to partial along and a near-zero
    H1 is uninterpretable. This function returns ρ + per-seed sign + a bootstrap
    CI; the analyzer (i477_phase_analyze.py) prints the H2 verdict first and
    suppresses the H1 read if H2 falsifies.

    Args:
        implant_only_cells: per-cell dicts at fixed count (anchor), varying LR.
            Required keys: "source_self_delta_g_at_last_ckpt",
            "mean_bystander_delta_g", "seed", "lr".

    Returns:
        {"rho": float, "p": float, "n": int, "per_seed_sign": {...},
         "verdict": "confirms" | "falsifies" | "indeterminate", "note": str}.
    """
    import numpy as np
    from scipy.stats import spearmanr

    n = len(implant_only_cells)
    if n < 3:
        return {
            "rho": float("nan"),
            "p": float("nan"),
            "n": n,
            "verdict": "indeterminate",
            "note": f"n={n} cells; need ≥3 for Spearman.",
        }
    implant = [float(c["source_self_delta_g_at_last_ckpt"]) for c in implant_only_cells]
    bystander = [float(c["mean_bystander_delta_g"]) for c in implant_only_cells]
    sr = spearmanr(implant, bystander)
    rho = float(sr.correlation)

    seeds = sorted({int(c["seed"]) for c in implant_only_cells})
    per_seed_sign: dict[str, dict] = {}
    for s in seeds:
        sub = [c for c in implant_only_cells if int(c["seed"]) == s]
        if len(sub) < 2:
            per_seed_sign[str(s)] = {"n": len(sub), "sign": None, "note": "n<2"}
            continue
        sub_implant = [float(c["source_self_delta_g_at_last_ckpt"]) for c in sub]
        sub_bystander = [float(c["mean_bystander_delta_g"]) for c in sub]
        sr_seed = spearmanr(sub_implant, sub_bystander)
        rho_seed = float(sr_seed.correlation) if not np.isnan(sr_seed.correlation) else 0.0
        per_seed_sign[str(s)] = {
            "n": len(sub),
            "rho": rho_seed,
            "sign": int(np.sign(rho_seed)) if rho_seed != 0 else 0,
        }
    sign_stable = (
        len({v.get("sign") for v in per_seed_sign.values() if v.get("sign") is not None}) <= 1
    )

    if abs(rho) >= 0.80 and sign_stable:
        verdict = "confirms"
    elif abs(rho) <= 0.40:
        verdict = "falsifies"
    else:
        verdict = "indeterminate"

    return {
        "rho": rho,
        "p": float(sr.pvalue),
        "n": n,
        "sign_stable_across_seeds": sign_stable,
        "per_seed_sign": per_seed_sign,
        "verdict": verdict,
        "note": (
            "Plan §6 H2: confirms if ρ ≥ 0.80 AND sign-stable; falsifies if |ρ| ≤ "
            "0.40. Falsifying H2 makes the H1 partial uninterpretable (item 2)."
        ),
    }


def loess_or_quadratic_fit(
    x: list[float], y: list[float], *, n_grid: int = 50
) -> dict[str, list[float]]:
    """Lightweight curvature overlay for F1 (plan §6 discipline item 6).

    Uses a degree-2 polynomial fit as the curvature overlay (true LOESS would
    pull in statsmodels.nonparametric; quadratic is a pragmatic stand-in that
    surfaces residual implant-curvature mistaken for a count effect). Returns
    {"x_grid": [...], "y_fit": [...]} ready to plot alongside the linear OLS.
    Falls back to linear if n<3 or polyfit fails.
    """
    import numpy as np

    if len(x) < 3:
        return {"x_grid": list(x), "y_fit": list(y)}
    try:
        coef = np.polyfit(np.asarray(x), np.asarray(y), 2)
        x_grid = np.linspace(min(x), max(x), n_grid)
        y_fit = np.polyval(coef, x_grid)
        return {"x_grid": x_grid.tolist(), "y_fit": y_fit.tolist()}
    except (np.linalg.LinAlgError, ValueError):
        return {"x_grid": list(x), "y_fit": list(y)}


def make_477_figures(  # noqa: C901 - linear multi-figure builder (F1..F6)
    *,
    main_cells: list[dict],
    kept_cells: list[dict],
    excluded_cells: list[dict],
    implant_only_cells: list[dict],
    calibration_table: dict,
    calibration_pick: dict,
    h1: dict,
    h2: dict,
    figures_dir: Path,
) -> dict[str, str]:
    """Generate F1..F6 from plan §6.

    F1: x=count, y=mean bystander ΔG, raw + residualized side-by-side, LOESS
        overlay (discipline items 6 + 11).
    F2: x=source-self ΔG, y=bystander ΔG over implant-only-axis cells.
    F3: per-cell bystander ΔG bars (color by achieved source ΔG, annotated with
        LR + steps). Discipline item 3 — substantive evidence over the ρ.
    F4: not produced here (trajectory dump lives in the cell artifacts).
    F5: calibration heatmap (x=LR, y=count, color=achieved source ΔG, star on
        picked LR).
    F6: H3 side-bet — x=count, y=source-self ΔG, one point per (count, LR)
        calibration cell.

    Returns: {figure_label: figure_path} for the analyzer.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # ── F1: raw + residualized side-by-side with LOESS overlay. ──────────────
    if kept_cells:
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.3))
        x_raw = [int(c["count"]) for c in kept_cells]
        y_raw = [float(c["mean_bystander_delta_g"]) for c in kept_cells]
        sizes = [60 + 4 * float(c["source_self_delta_g_at_last_ckpt"]) for c in kept_cells]
        axl.scatter(x_raw, y_raw, s=sizes, alpha=0.7, color="#0072B2")
        # LOESS-stand-in (quadratic) overlay.
        fit = loess_or_quadratic_fit(x_raw, y_raw)
        axl.plot(fit["x_grid"], fit["y_fit"], color="#D55E00", lw=1.5, label="quadratic")
        axl.set_xlabel("Number of negative personas")
        axl.set_ylabel("Mean bystander leakage (ΔG, nats)")
        axl.set_title("F1 raw: leakage vs count (size=achieved source ΔG)")
        axl.legend(fontsize=8)

        # Residualized: bystander, count BOTH residualized against implant.
        implant = [float(c["source_self_delta_g_at_last_ckpt"]) for c in kept_cells]
        y_res = _ols_residuals(y_raw, [implant])
        c_res = _ols_residuals([float(v) for v in x_raw], [implant])
        axr.scatter(c_res, y_res, s=sizes, alpha=0.7, color="#009E73")
        if len(c_res) >= 2:
            coef = np.polyfit(c_res, y_res, 1)
            xline = np.linspace(min(c_res), max(c_res), 20)
            axr.plot(xline, np.polyval(coef, xline), color="#D55E00", lw=1.5)
        rho_text = (
            f"ρ={h1.get('rho_given_implant', float('nan')):.2f} (n={h1.get('n', 0)})"
            if h1.get("interpretable")
            else h1.get("note", "")[:80]
        )
        axr.set_xlabel("Count (residualized vs implant)")
        axr.set_ylabel("Bystander leakage (residualized vs implant)")
        axr.set_title(f"F1 residualized: {rho_text}")
        fig.tight_layout()
        savefig_paper(fig, "i477_f1_count_vs_leakage", dir=str(figures_dir))
        plt.close(fig)
        written["F1"] = str(figures_dir / "i477_f1_count_vs_leakage")

    # ── F2: implant-only axis. ───────────────────────────────────────────────
    if implant_only_cells:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        x_io = [float(c["source_self_delta_g_at_last_ckpt"]) for c in implant_only_cells]
        y_io = [float(c["mean_bystander_delta_g"]) for c in implant_only_cells]
        ax.scatter(x_io, y_io, s=80, alpha=0.7, color="#0072B2")
        if len(x_io) >= 2:
            coef = np.polyfit(x_io, y_io, 1)
            xline = np.linspace(min(x_io), max(x_io), 20)
            ax.plot(xline, np.polyval(coef, xline), color="#D55E00", lw=1.5)
        ax.set_xlabel("Source-self ΔG (DV-B, nats)")
        ax.set_ylabel("Mean bystander leakage (DV-A, nats)")
        ax.set_title(
            f"F2 H2: implant-only axis (ρ={h2.get('rho', float('nan')):.2f}, "
            f"verdict={h2.get('verdict', '?')})"
        )
        fig.tight_layout()
        savefig_paper(fig, "i477_f2_implant_only_axis", dir=str(figures_dir))
        plt.close(fig)
        written["F2"] = str(figures_dir / "i477_f2_implant_only_axis")

    # ── F3: per-cell bars, color = achieved implant, annotation = LR + steps. ─
    if main_cells:
        # Mean across seeds within each count level, error bar = SD across seeds.
        by_count: dict[int, list[dict]] = {}
        for c in main_cells:
            by_count.setdefault(int(c["count"]), []).append(c)
        counts_sorted = sorted(by_count)
        means = [np.mean([c["mean_bystander_delta_g"] for c in by_count[k]]) for k in counts_sorted]
        sds = [
            np.std([c["mean_bystander_delta_g"] for c in by_count[k]], ddof=1)
            if len(by_count[k]) > 1
            else 0.0
            for k in counts_sorted
        ]
        colors_impl = [
            float(np.mean([c["source_self_delta_g_at_last_ckpt"] for c in by_count[k]]))
            for k in counts_sorted
        ]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        bars = ax.bar(
            [str(k) for k in counts_sorted],
            means,
            yerr=sds,
            capsize=4,
            color=plt.cm.viridis(
                (np.asarray(colors_impl) - min(colors_impl))
                / max(1e-6, max(colors_impl) - min(colors_impl))
            ),
        )
        for b, k in zip(bars, counts_sorted, strict=True):
            lrs = sorted({float(c["lr"]) for c in by_count[k]})
            steps = [int(c.get("step_at_last_ckpt", 0)) for c in by_count[k]]
            ann = (
                f"LR={'/'.join(f'{lr:g}' for lr in lrs)}\n"
                f"steps={int(np.mean(steps))} (avg)\n"
                f"impl={np.mean([c['source_self_delta_g_at_last_ckpt'] for c in by_count[k]]):.1f}"
            )
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.3,
                ann,
                ha="center",
                fontsize=7,
                rotation=0,
            )
        ax.set_xlabel("Number of negative personas")
        ax.set_ylabel("Mean bystander leakage (ΔG, nats)")
        ax.set_title("F3: per-cell bystander leakage, annotated with calibrated LR + steps")
        fig.tight_layout()
        savefig_paper(fig, "i477_f3_per_cell_bars", dir=str(figures_dir))
        plt.close(fig)
        written["F3"] = str(figures_dir / "i477_f3_per_cell_bars")

    # ── F5: calibration heatmap. ─────────────────────────────────────────────
    if calibration_table:
        counts_sorted = sorted(int(k) for k in calibration_table)
        # Union of all LRs across count rows (handles partial rows defensively).
        lrs_sorted = sorted({float(lr) for row in calibration_table.values() for lr in row})
        grid = np.full((len(counts_sorted), len(lrs_sorted)), np.nan)
        for i, cnt in enumerate(counts_sorted):
            row = (
                calibration_table[cnt] if cnt in calibration_table else calibration_table[str(cnt)]
            )
            for j, lr in enumerate(lrs_sorted):
                cell = row.get(lr) or row.get(str(lr)) or row.get(f"{lr:g}")
                if cell is not None:
                    grid[i, j] = float(cell["source_self_delta_g"])

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(lrs_sorted)))
        ax.set_xticklabels([f"{lr:g}" for lr in lrs_sorted], rotation=45)
        ax.set_yticks(range(len(counts_sorted)))
        ax.set_yticklabels([str(c) for c in counts_sorted])
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Number of negative personas")
        plt.colorbar(im, ax=ax, label="Achieved source-self ΔG (nats)")
        # Star markers on the picked LR per count.
        for i, cnt in enumerate(counts_sorted):
            pick = calibration_pick.get(cnt) or calibration_pick.get(str(cnt))
            if pick:
                picked_lr = float(pick["lr"])
                if picked_lr in lrs_sorted:
                    j = lrs_sorted.index(picked_lr)
                    ax.plot(j, i, marker="*", color="white", markersize=14, mec="black")
        ax.set_title("F5: calibration table (star = picked LR per count)")
        fig.tight_layout()
        savefig_paper(fig, "i477_f5_calibration_heatmap", dir=str(figures_dir))
        plt.close(fig)
        written["F5"] = str(figures_dir / "i477_f5_calibration_heatmap")

    # ── F6: H3 side-bet (count → source-implant survives calibration?). ──────
    if calibration_table:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for lr in sorted({float(lr) for row in calibration_table.values() for lr in row}):
            xs, ys = [], []
            for cnt in sorted(int(k) for k in calibration_table):
                row = (
                    calibration_table[cnt]
                    if cnt in calibration_table
                    else calibration_table[str(cnt)]
                )
                cell = row.get(lr) or row.get(str(lr)) or row.get(f"{lr:g}")
                if cell is not None:
                    xs.append(cnt)
                    ys.append(float(cell["source_self_delta_g"]))
            if xs:
                ax.plot(xs, ys, marker="o", label=f"LR={lr:g}")
        ax.set_xlabel("Number of negative personas")
        ax.set_ylabel("Source-self ΔG at terminal (nats)")
        ax.set_title("F6 H3: count → source-implant coupling survives calibration?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig_paper(fig, "i477_f6_h3_count_to_implant", dir=str(figures_dir))
        plt.close(fig)
        written["F6"] = str(figures_dir / "i477_f6_h3_count_to_implant")

    # Excluded cells get logged in the summary, not plotted (would be misleading
    # zero bars per Lens 13; the analyzer prose lists them with their actual
    # achieved ΔG + emission_p instead).
    log.info(
        "[477 figures] wrote %d figures; %d cells excluded by validity gate (not plotted): %s",
        len(written),
        len(excluded_cells),
        [c.get("cell") for c in excluded_cells],
    )
    return written


# ─────────────────────────────────────────────────────────────────────────────
# v4 step-lever marker-channel Bernoulli KL (plan v4 §4 + §6 + §11).
# Headline DV swap from v3 full-vocab KL → 2-class marker-channel KL.
# Computed as a POST-HOC transform from g_logp (P_trained(※)) and base_panel
# (P_base(※)), both already extracted by the existing rig; no new forward pass.
# ─────────────────────────────────────────────────────────────────────────────


_BERNOULLI_KL_EPS = 1e-9


def marker_channel_bernoulli_kl(
    p_trained: float, p_base: float, eps: float = _BERNOULLI_KL_EPS
) -> float:
    """KL(Bernoulli(p_trained) ‖ Bernoulli(p_base)) at the post-R marker slot.

    Plan v4 §4 pseudocode. Marker-targeted (only the ※ channel contributes —
    non-marker drift is folded into the (1−P) class) AND non-saturating (grows
    monotonically as P_trained departs from P_base, no -0.28-nat ceiling). The
    v4 headline DV.

    Args:
        p_trained: P(※) at the post-R slot under the trained model, per leaf.
        p_base: P(※) at the same slot under the base model, per leaf.
        eps: clamp keeping ``log`` finite at p=0 / p=1 (default 1e-9 nats).

    Returns:
        KL in nats, ≥0 by construction. Returns 0 exactly when p_trained == p_base.

    Raises:
        ValueError: either probability is outside [0, 1].
    """
    import math

    if not 0.0 <= p_trained <= 1.0:
        raise ValueError(f"p_trained={p_trained} not in [0,1]")
    if not 0.0 <= p_base <= 1.0:
        raise ValueError(f"p_base={p_base} not in [0,1]")
    p = min(max(p_trained, eps), 1.0 - eps)
    q = min(max(p_base, eps), 1.0 - eps)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def aggregate_bystander_marker_channel_kl(checkpoint: dict) -> float:
    """Mean over held-out (persona × question) leaves of the 2-class KL.

    Reads ``g_logp`` (P_trained log-prob) and ``b_logp`` (P_base log-prob) per
    leaf from the trajectory eval's ``held_out`` block — both already produced
    by the existing rig at every checkpoint, so this is a pure post-hoc
    transform (no new forward pass).

    Args:
        checkpoint: one entry of trajectory.json's ``checkpoints`` list.
            Required keys: ``held_out[persona][question]['g_logp']`` and
            ``['b_logp']``.

    Returns:
        Mean bystander marker-channel KL in nats.

    Raises:
        RuntimeError: no bystander leaves in this checkpoint (the rig wrote a
            corrupt artifact — fail loud, don't average over an empty list).
    """
    import math

    kls: list[float] = []
    held = checkpoint.get("held_out", {})
    for _persona, per_q in held.items():
        for _q, leaf in per_q.items():
            p_trained = math.exp(float(leaf["g_logp"]))
            p_base = math.exp(float(leaf["b_logp"]))
            # Clamp for the rare base case where exp(log_p) drifts >1 due to
            # bf16 round-off; the function itself fails loud on truly invalid
            # probabilities so a real schema break can't masquerade as a clamp.
            p_trained = min(p_trained, 1.0)
            p_base = min(p_base, 1.0)
            kls.append(marker_channel_bernoulli_kl(p_trained, p_base))
    if not kls:
        raise RuntimeError(
            "aggregate_bystander_marker_channel_kl: 0 bystander leaves in "
            f"checkpoint at frac={checkpoint.get('frac')!r} step={checkpoint.get('step')!r}; "
            f"trajectory.json corruption or empty eval panel?"
        )
    return sum(kls) / len(kls)


def aggregate_source_self_marker_channel_kl(checkpoint: dict) -> float:
    """Source-self marker-channel KL — the H1 conditioning covariate (v4).

    Symmetric with :func:`aggregate_bystander_marker_channel_kl` but reads
    the source's ``g_logp_mean`` / ``b_logp_mean`` from the checkpoint's
    ``source_self`` block (the existing rig writes per-source means only, not
    per-Q leaves — fine for the partial since the H1 covariate is a per-cell
    summary not a per-leaf regression).

    Args:
        checkpoint: one entry of trajectory.json's ``checkpoints`` list.

    Returns:
        Source-self marker-channel KL in nats, computed at the source's
        mean P(※) (≈ exp of the mean log-prob; tight upper bound since
        exp is convex on log-probs).
    """
    import math

    ss = checkpoint["source_self"]
    p_trained = min(math.exp(float(ss["g_logp_mean"])), 1.0)
    p_base = min(math.exp(float(ss["b_logp_mean"])), 1.0)
    return marker_channel_bernoulli_kl(p_trained, p_base)


def aggregate_bystander_full_vocab_kl(checkpoint: dict) -> float | None:
    """Mean over held-out leaves of full-vocab KL (the v3 demoted secondary).

    Reads the per-leaf ``kl`` field the rig writes when compute_kl=True; returns
    None when ANY leaf has ``kl is None`` (the rig skipped KL — smoke /
    --no-kl path). Keeps the paired sanity panel honest: a None aggregate
    propagates to ``rank_agreement_marker_vs_full_vocab`` which then refuses
    to compute the cross-DV check and surfaces "kl not computed".
    """
    kls: list[float] = []
    held = checkpoint.get("held_out", {})
    for _persona, per_q in held.items():
        for _q, leaf in per_q.items():
            v = leaf.get("kl")
            if v is None:
                return None
            kls.append(float(v))
    if not kls:
        return None
    return sum(kls) / len(kls)


def attach_marker_channel_aggregates(traj: dict) -> dict:
    """In-place: add v4 aggregates to every checkpoint in a trajectory.json.

    For each checkpoint adds:
      * ``source_self_marker_channel_kl``: H1 conditioning covariate.
      * ``mean_bystander_marker_channel_kl``: v4 HEADLINE DV.
      * ``mean_bystander_full_vocab_kl``: paired v3 secondary (or None if
        the rig skipped KL).

    Idempotent: safe to call twice (overwrites existing aggregates).
    Returns the same ``traj`` dict for chaining.
    """
    for ck in traj.get("checkpoints", []):
        ck["source_self_marker_channel_kl"] = aggregate_source_self_marker_channel_kl(ck)
        ck["mean_bystander_marker_channel_kl"] = aggregate_bystander_marker_channel_kl(ck)
        ck["mean_bystander_full_vocab_kl"] = aggregate_bystander_full_vocab_kl(ck)
    return traj


def partial_spearman_count_given_implant_marker_channel_kl(
    kept_cells: list[dict],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
    min_count_levels: int = 3,
) -> dict[str, Any]:
    """v4 HEADLINE: ρ(count, bystander-marker-channel-KL | source-self-marker-channel-KL).

    Mirror of :func:`partial_spearman_count_given_implant` but uses the v4
    marker-channel KL aggregates on both axes. Each kept cell must carry:
      * ``count``, ``seed``, ``phase`` ("main" defensive assert).
      * ``mean_bystander_marker_channel_kl_at_picked_step`` — v4 headline DV.
      * ``source_self_marker_channel_kl_at_picked_step`` — H1 covariate.
      * ``step_at_last_ckpt`` — robustness partialling extra control.
    """
    from scipy.stats import spearmanr

    for c in kept_cells:
        phase = c.get("phase")
        if phase != "main":
            raise AssertionError(
                "partial_spearman_count_given_implant_marker_channel_kl got a "
                f"non-main-phase cell: cell={c.get('cell', '<unknown>')!r}, "
                f"seed={c.get('seed')}, phase={phase!r}. v4 H1 partial Spearman "
                "must be computed over main-phase cells only."
            )

    counts = [int(c["count"]) for c in kept_cells]
    distinct_counts = sorted(set(counts))
    n = len(kept_cells)
    if len(distinct_counts) < min_count_levels:
        return {
            "interpretable": False,
            "n": n,
            "n_count_levels": len(distinct_counts),
            "kept_counts": distinct_counts,
            "headline_dv": "marker_channel_bernoulli_kl",
            "note": (
                f"n_count_levels={len(distinct_counts)} < min_count_levels="
                f"{min_count_levels} — coverage floor violated (plan v4 §6 "
                "discipline item 5). v4 headline partial uninterpretable."
            ),
        }

    bystander = [float(c["mean_bystander_marker_channel_kl_at_picked_step"]) for c in kept_cells]
    implant = [float(c["source_self_marker_channel_kl_at_picked_step"]) for c in kept_cells]
    step = [int(c.get("step_at_last_ckpt", 0)) for c in kept_cells]

    y_resid_i = _ols_residuals(bystander, [implant])
    c_resid_i = _ols_residuals([float(v) for v in counts], [implant])
    sr_i = spearmanr(c_resid_i, y_resid_i)
    boot_i = _stratified_bootstrap_partial_spearman(
        counts, bystander, implant, n_resamples=n_bootstrap, seed=bootstrap_seed
    )

    y_resid_is = _ols_residuals(bystander, [implant, [float(v) for v in step]])
    c_resid_is = _ols_residuals([float(v) for v in counts], [implant, [float(v) for v in step]])
    sr_is = spearmanr(c_resid_is, y_resid_is)
    boot_is = _stratified_bootstrap_partial_spearman(
        counts,
        bystander,
        implant,
        n_resamples=n_bootstrap,
        seed=bootstrap_seed + 1,
        extra_controls=[[float(v) for v in step]],
    )

    seeds_in_kept = sorted({int(c["seed"]) for c in kept_cells})
    per_seed_sign: dict[str, dict] = {}
    for s in seeds_in_kept:
        sub = [c for c in kept_cells if int(c["seed"]) == s]
        if len({int(c["count"]) for c in sub}) < 2:
            per_seed_sign[str(s)] = {
                "n": len(sub),
                "n_count_levels": len({int(c["count"]) for c in sub}),
                "sign": None,
                "note": "fewer than 2 count levels at this seed; sign not defined",
            }
            continue
        sub_counts = [int(c["count"]) for c in sub]
        sub_bystander = [float(c["mean_bystander_marker_channel_kl_at_picked_step"]) for c in sub]
        sr_seed = spearmanr(sub_counts, sub_bystander)
        rho_seed = float(sr_seed.correlation) if not np.isnan(sr_seed.correlation) else 0.0
        per_seed_sign[str(s)] = {
            "n": len(sub),
            "n_count_levels": len({int(c["count"]) for c in sub}),
            "rho_raw_count_bystander": rho_seed,
            "sign": int(np.sign(rho_seed)) if rho_seed != 0 else 0,
        }

    pooled_sign = (
        int(np.sign(sr_i.correlation))
        if not np.isnan(sr_i.correlation) and sr_i.correlation != 0
        else 0
    )
    sign_stable = all(
        (v.get("sign") is None) or v.get("sign") == pooled_sign for v in per_seed_sign.values()
    )

    return {
        "interpretable": True,
        "n": n,
        "n_count_levels": len(distinct_counts),
        "kept_counts": distinct_counts,
        "rho_given_implant": float(sr_i.correlation),
        "p_given_implant": float(sr_i.pvalue),
        "bootstrap_given_implant": boot_i,
        "rho_given_implant_and_step": float(sr_is.correlation),
        "p_given_implant_and_step": float(sr_is.pvalue),
        "bootstrap_given_implant_and_step": boot_is,
        "per_seed_sign": per_seed_sign,
        "pooled_sign": pooled_sign,
        "sign_stable_across_seeds": sign_stable,
        "headline_dv": "marker_channel_bernoulli_kl",
        "interpretation_scope": (
            "row-scaled negative-budget recipe (count co-varies with total "
            "negative rows at fixed positives); NOT pure persona-diversity. "
            "See plan v4 §3 H1 + §6 discipline item 1."
        ),
        "note": (
            "Plan v4 §6 discipline item 1: report BOTH partials. Item 3: ρ is "
            "a SUMMARY of F3. Item 9: cross-DV agreement with full-vocab KL "
            "MUST gate the verdict — see rank_agreement_marker_vs_full_vocab."
        ),
    }


def partial_spearman_count_given_implant_full_vocab_kl(
    kept_cells: list[dict],
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
    min_count_levels: int = 3,
) -> dict[str, Any]:
    """Paired secondary: ρ(count, bystander-full-vocab-KL | source-self-marker-channel-KL).

    Same shape as the v4 headline partial but y = full-vocab KL. Reported
    side-by-side with the headline so divergent rank orders surface in the
    write-up (cross-DV agreement gate, §6 #9).
    """
    from scipy.stats import spearmanr

    for c in kept_cells:
        if c.get("phase") != "main":
            raise AssertionError(
                "partial_spearman_count_given_implant_full_vocab_kl got a "
                f"non-main-phase cell: {c.get('cell', '<unknown>')!r}"
            )

    counts = [int(c["count"]) for c in kept_cells]
    distinct_counts = sorted(set(counts))
    n = len(kept_cells)
    if len(distinct_counts) < min_count_levels:
        return {
            "interpretable": False,
            "n": n,
            "n_count_levels": len(distinct_counts),
            "kept_counts": distinct_counts,
            "headline_dv": "full_vocab_kl",
            "note": (
                f"n_count_levels={len(distinct_counts)} < min_count_levels="
                f"{min_count_levels} — coverage floor violated."
            ),
        }

    bystander = [float(c["mean_bystander_full_vocab_kl_at_picked_step"]) for c in kept_cells]
    implant = [float(c["source_self_marker_channel_kl_at_picked_step"]) for c in kept_cells]
    y_resid = _ols_residuals(bystander, [implant])
    c_resid = _ols_residuals([float(v) for v in counts], [implant])
    sr = spearmanr(c_resid, y_resid)
    boot = _stratified_bootstrap_partial_spearman(
        counts, bystander, implant, n_resamples=n_bootstrap, seed=bootstrap_seed
    )
    return {
        "interpretable": True,
        "n": n,
        "n_count_levels": len(distinct_counts),
        "kept_counts": distinct_counts,
        "rho_given_implant": float(sr.correlation),
        "p_given_implant": float(sr.pvalue),
        "bootstrap_given_implant": boot,
        "headline_dv": "full_vocab_kl",
        "note": (
            "Plan v4 §6 discipline item 9: PAIRED sanity panel; never decides "
            "H1 alone. Rank-divergence vs the marker-channel headline triggers "
            "an H1 DOWNGRADE per rank_agreement_marker_vs_full_vocab."
        ),
    }


def implant_only_axis_spearman_marker_channel_kl(
    implant_only_cells: list[dict],
) -> dict[str, Any]:
    """v4 H2: ρ(source-marker-channel-KL, bystander-marker-channel-KL).

    Same verdict thresholds as the v2 implant-only-axis Spearman (confirms
    if ρ ≥ 0.80 AND sign-stable; falsifies if |ρ| ≤ 0.40). Just the v4 DV
    swap on both axes.
    """
    from scipy.stats import spearmanr

    n = len(implant_only_cells)
    if n < 3:
        return {
            "rho": float("nan"),
            "p": float("nan"),
            "n": n,
            "verdict": "indeterminate",
            "headline_dv": "marker_channel_bernoulli_kl",
            "note": f"n={n} cells; need ≥3 for Spearman.",
        }
    implant = [float(c["source_self_marker_channel_kl_at_picked_step"]) for c in implant_only_cells]
    bystander = [
        float(c["mean_bystander_marker_channel_kl_at_picked_step"]) for c in implant_only_cells
    ]
    sr = spearmanr(implant, bystander)
    rho = float(sr.correlation)
    seeds = sorted({int(c["seed"]) for c in implant_only_cells})
    per_seed_sign: dict[str, dict] = {}
    for s in seeds:
        sub = [c for c in implant_only_cells if int(c["seed"]) == s]
        if len(sub) < 2:
            per_seed_sign[str(s)] = {"n": len(sub), "sign": None, "note": "n<2"}
            continue
        sub_i = [float(c["source_self_marker_channel_kl_at_picked_step"]) for c in sub]
        sub_b = [float(c["mean_bystander_marker_channel_kl_at_picked_step"]) for c in sub]
        sr_seed = spearmanr(sub_i, sub_b)
        rho_seed = float(sr_seed.correlation) if not np.isnan(sr_seed.correlation) else 0.0
        per_seed_sign[str(s)] = {
            "n": len(sub),
            "rho": rho_seed,
            "sign": int(np.sign(rho_seed)) if rho_seed != 0 else 0,
        }
    sign_stable = (
        len({v.get("sign") for v in per_seed_sign.values() if v.get("sign") is not None}) <= 1
    )
    if abs(rho) >= 0.80 and sign_stable:
        verdict = "confirms"
    elif abs(rho) <= 0.40:
        verdict = "falsifies"
    else:
        verdict = "indeterminate"
    return {
        "rho": rho,
        "p": float(sr.pvalue),
        "n": n,
        "sign_stable_across_seeds": sign_stable,
        "per_seed_sign": per_seed_sign,
        "verdict": verdict,
        "headline_dv": "marker_channel_bernoulli_kl",
        "note": "v4 H2: same thresholds as v2 implant-only-axis Spearman, DV-swap on both axes.",
    }


def rank_agreement_marker_vs_full_vocab(
    kept_cells: list[dict],
    *,
    downgrade_threshold: float = 0.70,
) -> dict[str, Any]:
    """v4 §6 discipline #9: cross-DV rank agreement across kept cells.

    Computes Spearman across kept cells between
    ``mean_bystander_marker_channel_kl_at_picked_step`` and
    ``mean_bystander_full_vocab_kl_at_picked_step``. If across-cell ρ <
    ``downgrade_threshold`` (default 0.70), the verdict is "divergence —
    downgrade H1" and the write-up must narrate the construct-divergence.

    If full-vocab KL is missing on any cell (rig --no-kl path), the verdict is
    "kl not computed" and the agreement gate is skipped (the headline still
    publishes, but the gate cannot rule on it).
    """
    from scipy.stats import spearmanr

    marker: list[float] = []
    full: list[float] = []
    missing: list[str] = []
    for c in kept_cells:
        m = c.get("mean_bystander_marker_channel_kl_at_picked_step")
        f = c.get("mean_bystander_full_vocab_kl_at_picked_step")
        if m is None or f is None:
            missing.append(c.get("cell", "<unknown>"))
            continue
        marker.append(float(m))
        full.append(float(f))

    if missing:
        return {
            "verdict": "kl not computed",
            "n": len(marker),
            "n_missing": len(missing),
            "missing_cells": missing,
            "downgrade_if_below": downgrade_threshold,
            "note": (
                "Full-vocab KL missing on one or more kept cells (rig --no-kl "
                "or partial trajectory). Cross-DV agreement gate skipped; the "
                "v4 headline publishes without the construct-divergence check."
            ),
        }
    if len(marker) < 3:
        return {
            "verdict": "indeterminate",
            "cross_dv_rank_spearman": float("nan"),
            "n": len(marker),
            "downgrade_if_below": downgrade_threshold,
            "note": f"n={len(marker)} kept cells; need ≥3 for Spearman.",
        }
    sr = spearmanr(marker, full)
    rho = float(sr.correlation) if not np.isnan(sr.correlation) else 0.0
    verdict = "agreement" if rho >= downgrade_threshold else "divergence — downgrade H1"
    return {
        "verdict": verdict,
        "cross_dv_rank_spearman": rho,
        "p": float(sr.pvalue),
        "n": len(marker),
        "downgrade_if_below": downgrade_threshold,
        "note": (
            "Plan v4 §6 discipline item 9: across-cell Spearman between v4 "
            "marker-channel headline and v3 full-vocab paired-secondary. "
            "<0.70 → DOWNGRADE H1 and narrate the construct-divergence "
            "(full-vocab moves on non-marker drift per Codex round-2)."
        ),
    }
