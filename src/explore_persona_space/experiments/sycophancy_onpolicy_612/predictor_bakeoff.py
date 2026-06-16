"""Task #612 predictor-v3 Bucket 4 — leakage-predictor bake-off (CPU/VM).

Plan v3 §4.5: per KEPT source, fit + compare three predictors of per-bystander
matched-install leakage Delta on the decorrelated panel:

  (a) base_prior        — bystander base sycophancy prior (front-runner #500/#532/#541)
  (b) cosine_to_source  — layer-20 persona cosine (the v1 panel layer)
  (c) pv_alignment      — #623 persona-vector -> sycophancy-direction COSINE,
                          variant lt_persona_lt_syc, layer 21 (NO L20 in #623 —
                          L21 nearest; DOCUMENTED in the output, brief concern #2)

Per predictor: Spearman rho + a 95% BCa bootstrap CI (B=10,000). Two-predictor:
partial-Spearman rho(Delta, cosine | prior) per source IF |Pearson(cosine, prior)| <=
0.6, else the collinearity-gate fallback (tercile-bucket median split). The
verdict per source is the predictor with the highest |rho| whose 95% CI does not
overlap the runners-up (else indeterminate), Bonferroni-corrected across 3
predictors x kept sources at alpha=0.05. ALSO reports pairwise predictor
correlations per source (the a/c collinearity the brief flags) + the realized
panel decorrelation r. Pooled (all kept sources) descriptive read added (§9).

Pure functions over on-disk artifacts; no GPU, no API. The script
``scripts/issue612_predictor_bakeoff.py`` is the VM CLI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (
    V3_BAKEOFF_ALPHA,
    V3_BAKEOFF_BOOTSTRAP_B,
    V3_COLLINEARITY_GATE,
    V3_COSINE_LAYER,
    V3_PV_COSINE_LAYER,
    V3_PV_COSINE_VARIANT,
)

log = logging.getLogger("issue_612.predictor_bakeoff")

PREDICTOR_NAMES = ("base_prior", "cosine_to_source", "pv_alignment")


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rho; None when undefined (n<3 or a tied/constant vector)."""
    from scipy.stats import spearmanr

    if len(x) < 3:
        return None
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    rho = spearmanr(x, y).statistic
    return None if np.isnan(rho) else float(rho)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _bca_ci(
    x: np.ndarray, y: np.ndarray, *, b: int, alpha: float, seed: int = 612
) -> tuple[float | None, float | None]:
    """95% BCa bootstrap CI for Spearman rho (bias-corrected + accelerated).

    Resamples (x, y) PAIRS with replacement (the bystander cell is the unit).
    Falls back gracefully (returns (None, None)) when the point estimate is
    undefined. BCa per Efron & Tibshirani; the acceleration uses the jackknife.
    """
    from scipy.stats import norm

    theta_hat = _spearman(x, y)
    if theta_hat is None:
        return (None, None)
    n = len(x)
    rng = np.random.default_rng(seed)
    boot = np.empty(b, dtype=float)
    n_valid = 0
    for _ in range(b):
        idx = rng.integers(0, n, n)
        r = _spearman(x[idx], y[idx])
        if r is not None:
            boot[n_valid] = r
            n_valid += 1
    if n_valid < b // 2:
        # too many degenerate resamples — fall back to the percentile interval
        lo, hi = np.percentile(boot[:n_valid], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (float(lo), float(hi))
    boot = boot[:n_valid]
    # bias-correction z0
    prop = np.mean(boot < theta_hat)
    prop = min(max(prop, 1.0 / (n_valid + 1)), n_valid / (n_valid + 1))
    z0 = norm.ppf(prop)
    # acceleration via jackknife
    jack = np.array(
        [r for i in range(n) if (r := _spearman(np.delete(x, i), np.delete(y, i))) is not None]
    )
    if len(jack) < 3 or np.all(jack == jack[0]):
        accel = 0.0
    else:
        jbar = jack.mean()
        num = np.sum((jbar - jack) ** 3)
        den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
        accel = float(num / den) if den != 0 else 0.0
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def _adj(z):
        denom = 1 - accel * (z0 + z)
        if denom == 0:
            return 0.5
        return float(norm.cdf(z0 + (z0 + z) / denom))

    a1, a2 = _adj(z_lo), _adj(z_hi)
    lo, hi = np.percentile(boot, [100 * a1, 100 * a2])
    return (float(lo), float(hi))


def _partial_spearman(delta: np.ndarray, cosine: np.ndarray, prior: np.ndarray) -> dict:
    """Partial Spearman rho(Delta, cosine | prior) with the §4.5 collinearity gate.

    If |Pearson(cosine, prior)| > V3_COLLINEARITY_GATE, fall back to a
    tercile-bucket median split (cosine effect within prior terciles) instead of
    the unstable partial correlation."""
    collin = _pearson(cosine, prior)
    gated = collin is not None and abs(collin) > V3_COLLINEARITY_GATE
    out: dict = {"collinearity_pearson_cos_prior": collin, "collinearity_gated": gated}
    if not gated:
        r_dc = _spearman(delta, cosine)
        r_dp = _spearman(delta, prior)
        r_cp = _spearman(cosine, prior)
        if None in (r_dc, r_dp, r_cp):
            out["method"] = "partial_spearman"
            out["rho_partial"] = None
        else:
            denom = float(np.sqrt((1 - r_dp**2) * (1 - r_cp**2)))
            out["method"] = "partial_spearman"
            out["rho_partial"] = float((r_dc - r_dp * r_cp) / denom) if denom > 0 else None
    else:
        # tercile-bucket median test: median Delta in top vs bottom cosine tercile,
        # within the overall sample (small N -> descriptive, not a p-value).
        order = np.argsort(cosine)
        n = len(cosine)
        t = max(1, n // 3)
        lo_idx, hi_idx = order[:t], order[-t:]
        out["method"] = "tercile_bucket_median"
        out["median_delta_low_cosine"] = float(np.median(delta[lo_idx]))
        out["median_delta_high_cosine"] = float(np.median(delta[hi_idx]))
        out["tercile_delta_gap_high_minus_low"] = float(
            np.median(delta[hi_idx]) - np.median(delta[lo_idx])
        )
    return out


def load_pv_alignment(i623_cosine_matrix: Path) -> dict[str, float]:
    """Comparator (c): {persona: cosine(persona-vec, syc-direction)} at the pinned
    variant + layer (lt_persona_lt_syc / L21; documented per brief concern #2)."""
    payload = json.loads(i623_cosine_matrix.read_text())["cosine_matrix"]
    if V3_PV_COSINE_VARIANT not in payload:
        raise KeyError(
            f"#623 cosine_matrix has no variant {V3_PV_COSINE_VARIANT!r} "
            f"(available: {sorted(payload)})"
        )
    layers = payload[V3_PV_COSINE_VARIANT]
    if V3_PV_COSINE_LAYER not in layers:
        raise KeyError(
            f"#623 variant {V3_PV_COSINE_VARIANT!r} has no layer {V3_PV_COSINE_LAYER!r} "
            f"(available: {sorted(layers)}) — no L20 in #623, L21 is the pinned nearest"
        )
    return {name: float(v) for name, v in layers[V3_PV_COSINE_LAYER].items()}


def load_panel(panels_dir: Path, source: str) -> dict:
    path = panels_dir / source / "panel.json"
    if not path.exists():
        raise FileNotFoundError(f"decorrelated panel missing: {path}")
    return json.loads(path.read_text())


def bakeoff_for_source(
    source: str,
    panel: dict,
    leakage: dict[str, float],
    pv_alignment: dict[str, float],
    *,
    bootstrap_b: int,
    alpha: float,
    n_comparisons: int,
) -> dict:
    """One source's bake-off: 3 predictors x (Spearman + BCa CI), partial read,
    pairwise predictor correlations, verdict (Bonferroni-corrected)."""
    bystanders = sorted(panel["bystanders"])
    # comparator (c) may be missing a panel persona — drop + report per cell.
    usable = [b for b in bystanders if b in leakage and b in pv_alignment]
    dropped_pv = sorted(b for b in bystanders if b in leakage and b not in pv_alignment)
    # predictors (a)/(b) need only leakage + panel record
    have_leak = [b for b in bystanders if b in leakage]

    delta = np.array([leakage[b] for b in have_leak], dtype=float)
    prior = np.array([panel["bystanders"][b]["base_prior"] for b in have_leak], dtype=float)
    cosine = np.array([panel["bystanders"][b]["cosine_to_source"] for b in have_leak], dtype=float)
    pv = np.array([pv_alignment.get(b, np.nan) for b in have_leak], dtype=float)
    pv_mask = ~np.isnan(pv)

    predictors: dict[str, dict] = {}
    for name, vec, mask in (
        ("base_prior", prior, np.ones(len(have_leak), dtype=bool)),
        ("cosine_to_source", cosine, np.ones(len(have_leak), dtype=bool)),
        ("pv_alignment", pv, pv_mask),
    ):
        d, v = delta[mask], vec[mask]
        rho = _spearman(d, v)
        # Bonferroni-adjusted CI level for the FWER target (§5 H2 / §11).
        adj_alpha = alpha / max(1, n_comparisons)
        lo, hi = _bca_ci(d, v, b=bootstrap_b, alpha=adj_alpha) if rho is not None else (None, None)
        ci_unadj = _bca_ci(d, v, b=bootstrap_b, alpha=alpha) if rho is not None else (None, None)
        predictors[name] = {
            "spearman_rho": rho,
            "n": int(mask.sum()),
            "ci95": list(ci_unadj),
            "ci_bonferroni": list((lo, hi)),
            "bonferroni_alpha": adj_alpha,
        }

    partial = _partial_spearman(delta, cosine, prior)

    # pairwise predictor correlations (the a/c collinearity the brief flags)
    pw: dict[str, float | None] = {}
    pw["base_prior__vs__cosine_to_source"] = _spearman(prior, cosine)
    pw["base_prior__vs__pv_alignment"] = (
        _spearman(prior[pv_mask], pv[pv_mask]) if pv_mask.sum() >= 3 else None
    )
    pw["cosine_to_source__vs__pv_alignment"] = (
        _spearman(cosine[pv_mask], pv[pv_mask]) if pv_mask.sum() >= 3 else None
    )

    # verdict: highest |rho| with a Bonferroni CI not overlapping the runners-up.
    verdict = _verdict(predictors)

    return {
        "source": source,
        "status": panel["status"],
        "n_bystanders": len(bystanders),
        "n_usable_all_predictors": len(usable),
        "dropped_pv_alignment_personas": dropped_pv,
        "realized_panel_decorrelation_r": panel.get("realized_pearson_cos_prior"),
        "predictors": predictors,
        "partial_cosine_given_prior": partial,
        "pairwise_predictor_correlation": pw,
        "verdict": verdict,
    }


def _verdict(predictors: dict[str, dict]) -> dict:
    """Winner = highest |rho| whose Bonferroni CI excludes the others' point
    estimates AND does not overlap the runner-up CI; else indeterminate (§5 H2)."""
    scored = [
        (name, rec)
        for name, rec in predictors.items()
        if rec["spearman_rho"] is not None and None not in rec["ci_bonferroni"]
    ]
    if not scored:
        return {"winner": None, "reason": "no predictor had a defined rho + CI"}
    scored.sort(key=lambda kv: abs(kv[1]["spearman_rho"]), reverse=True)
    top_name, top = scored[0]
    if len(scored) == 1:
        return {
            "winner": top_name,
            "reason": "only one defined predictor",
            "top_rho": top["spearman_rho"],
        }
    _, runner = scored[1]
    top_lo, top_hi = top["ci_bonferroni"]
    run_lo, run_hi = runner["ci_bonferroni"]
    non_overlap = top_lo > run_hi or run_lo > top_hi
    return {
        "winner": top_name if non_overlap else None,
        "reason": (
            "highest |rho| with non-overlapping Bonferroni CI vs runner-up"
            if non_overlap
            else "top CI overlaps runner-up — indeterminate"
        ),
        "top_rho": top["spearman_rho"],
        "runner_up_rho": runner["spearman_rho"],
        "ci_overlap": not non_overlap,
    }


def run_bakeoff(
    *,
    leakage_by_source: dict[str, dict[str, float]],
    panels_dir: Path,
    panel_set_path: Path,
    i623_cosine_matrix: Path,
    i623_syc_i: Path,
    bootstrap_b: int = V3_BAKEOFF_BOOTSTRAP_B,
    alpha: float = V3_BAKEOFF_ALPHA,
) -> dict:
    """Full bake-off over kept sources. ``leakage_by_source`` maps source ->
    {bystander: leakage Delta at matched install}."""
    pv_alignment = load_pv_alignment(i623_cosine_matrix)
    kept = sorted(s for s in leakage_by_source if load_panel(panels_dir, s)["status"] == "ok")
    dropped_decorr = sorted(
        s for s in leakage_by_source if load_panel(panels_dir, s)["status"] != "ok"
    )
    # Bonferroni denominator: 3 predictors x kept sources (§5 H2 / §11).
    n_comparisons = 3 * max(1, len(kept))

    per_source: dict[str, dict] = {}
    for source in kept:
        panel = load_panel(panels_dir, source)
        per_source[source] = bakeoff_for_source(
            source,
            panel,
            leakage_by_source[source],
            pv_alignment,
            bootstrap_b=bootstrap_b,
            alpha=alpha,
            n_comparisons=n_comparisons,
        )

    pooled = _pooled(per_source, panels_dir, leakage_by_source, pv_alignment, bootstrap_b, alpha)

    return {
        "schema_version": 1,
        "followup_label": "onpolicy-leakage-predictor",
        "kept_sources": kept,
        "dropped_decorrelation_failed": dropped_decorr,
        "bonferroni": {
            "alpha": alpha,
            "n_comparisons": n_comparisons,
            "rule": "3 predictors x kept sources at alpha (plan v3 §5 H2 / §11)",
        },
        "predictor_provenance": {
            "base_prior": "bystander base sycophancy prior (panel.json base_prior)",
            "cosine_to_source": f"layer-{V3_COSINE_LAYER} persona cosine (v1 panel_set.json)",
            "pv_alignment": (
                f"#623 cosine_matrix variant {V3_PV_COSINE_VARIANT} layer "
                f"{V3_PV_COSINE_LAYER} (NO L20 in #623; L21 is the pinned nearest — "
                f"brief concern #2)"
            ),
        },
        "bootstrap": {"B": bootstrap_b, "ci": "95% BCa", "seed": 612},
        "per_source": per_source,
        "pooled": pooled,
        "i623_inputs": {
            "cosine_matrix": str(i623_cosine_matrix),
            "syc_i": str(i623_syc_i),
        },
    }


def _pooled(
    per_source: dict[str, dict],
    panels_dir: Path,
    leakage_by_source: dict[str, dict[str, float]],
    pv_alignment: dict[str, float],
    bootstrap_b: int,
    alpha: float,
) -> dict:
    """Pooled (all kept (source, bystander) cells) descriptive Spearman per
    predictor (§9 — the pooled N carries the power)."""
    deltas: list[float] = []
    priors: list[float] = []
    cosines: list[float] = []
    pvs: list[float] = []
    for source in per_source:
        panel = load_panel(panels_dir, source)
        for b in sorted(panel["bystanders"]):
            if b not in leakage_by_source[source]:
                continue
            deltas.append(leakage_by_source[source][b])
            priors.append(panel["bystanders"][b]["base_prior"])
            cosines.append(panel["bystanders"][b]["cosine_to_source"])
            pvs.append(pv_alignment.get(b, np.nan))
    d = np.array(deltas)
    out: dict = {"n_cells": len(d)}
    for name, vec in (
        ("base_prior", np.array(priors)),
        ("cosine_to_source", np.array(cosines)),
        ("pv_alignment", np.array(pvs)),
    ):
        mask = ~np.isnan(vec)
        rho = _spearman(d[mask], vec[mask])
        ci = (
            _bca_ci(d[mask], vec[mask], b=bootstrap_b, alpha=alpha)
            if rho is not None
            else (None, None)
        )
        out[name] = {"spearman_rho": rho, "n": int(mask.sum()), "ci95": list(ci)}
    defined = {
        k: v for k, v in out.items() if k in PREDICTOR_NAMES and v["spearman_rho"] is not None
    }
    out["winner"] = max(defined, key=lambda k: abs(defined[k]["spearman_rho"])) if defined else None
    return out


__all__ = [
    "PREDICTOR_NAMES",
    "bakeoff_for_source",
    "load_panel",
    "load_pv_alignment",
    "run_bakeoff",
]
