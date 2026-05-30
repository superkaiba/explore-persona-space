# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #448 Phase 3 — per-cell analysis + per-knob monotonicity + permutation null.

Inputs (consumed):
  base panel:  eval_results/issue_448/base/marker_logprob.json
  per cell:    eval_results/issue_448/<cell_slug>/marker_logprob.json
  centroids:   eval_results/issue_448/centroids/{centroids_layer20.pt,persona_names.json}
  cell manifests (training-data composition, from build_training_data.py):
               <slab_root>/<cell_slug>/train_pool.manifest.json
               (the manifest tells us which negative personas were trained
                against — used to compute per-bystander nearest_neg_distance)

Outputs:
  eval_results/issue_448/analyze_summary.json
  figures/issue_448/{per_knob_<k>.png × 4,
                    headline_permutation_null.png,
                    nearest_neg_distance_spread.png,
                    per_cell_rho_strip.png}

Deliverables (plan §4.5 + §6):

  (i)   Per-cell mean + median bystander marker-logp-Δ + bootstrap 95% CI
        (10k iter) over the 23-bystander vector (and the 20-bystander
        common-set per plan §13 #6).
  (ii)  Per-cell source-self mean log-p + floor check (≥ −12 nats per §6 M1).
  (iii) Per-cell nearest_neg_distance distribution + degeneracy guard
        (stdev<0.02 OR IQR<0.03 → secondary metric skipped).
  (iv)  Per-cell Spearman ρ(per-bystander Δ, nearest_neg_distance) +
        bootstrap 95% CI + permutation p (10k iter); predicted ρ > 0.
        Also report PARTIAL ρ controlling for cosine-to-source per §13 #9.
  (v)   Headline (M4): permutation-null on the headline integer.
        Calibrated |Δrange| threshold = max(2 nats,
        1.5 × median_per_cell_bootstrap_CI_halfwidth). Shuffle per-cell mean
        Δs across cells 10k times; recompute headline-count under the null;
        report (observed, null_median, null_95pct_upper, empirical_p).
  (vi)  Per-knob monotonicity test + 4 line plots (mean bystander Δ vs
        knob level, per-cell CI bands).
  (vii) Per-cell empirical fraction_of_training_rows_from_topup as a M5
        consistency check (§13 #13).

CPU-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (  # noqa: E402
    CELL_SPECS,
    SOURCE_PERSONA,
    SOURCE_SELF_TRAINING_FLOOR_NATS,
)
from explore_persona_space.experiments.factor_screen_365.persona_panel import (  # noqa: E402
    EVAL_PERSONAS_24,
)

log = logging.getLogger("issue_448.analyze")

BOOTSTRAP_N = 10_000
PERMUTATION_N = 10_000
DEGENERACY_STDEV_MAX = 0.02
DEGENERACY_IQR_MAX = 0.03
HEADLINE_MIN_DELTA_RANGE_NATS = 2.0
HEADLINE_CI_MULTIPLIER = 1.5

# Multi-positive personas per cell (matches __init__.py CELL_SPECS).
# Used to know which personas to EXCLUDE from the bystander set in cells 5/6.
MULTI_POSITIVE_BY_CELL: dict[str, list[str]] = {
    "c5_pos_personas_2": ["villain", "comedian"],
    "c6_pos_personas_4": ["villain", "comedian", "assistant", "software_engineer"],
}


def _load_logp(path: Path) -> dict[str, Any]:
    """Load a per-cell marker_logprob.json. Raises if missing/malformed."""
    if not path.exists():
        raise FileNotFoundError(f"Eval JSON not found: {path}")
    data = json.loads(path.read_text())
    if data.get("schema") != "issue_448.marker_logprob v1":
        raise ValueError(f"Unexpected schema in {path}: {data.get('schema')!r}")
    return data


def _per_persona_mean_logp(
    logp_by_persona: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Mean log-p across the 20 questions, per persona. Returns {persona: float}."""
    out: dict[str, float] = {}
    for persona, by_q in logp_by_persona.items():
        vals = [float(v) for v in by_q.values()]
        if not vals:
            raise ValueError(f"Empty per-question log-p for persona {persona!r}")
        out[persona] = float(np.mean(vals))
    return out


def _bootstrap_ci(
    values: np.ndarray, n_iter: int = BOOTSTRAP_N, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap mean + (lower, upper) CI of ``values``. Returns (mean, low, high)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_iter, n))
    samples = values[idx]
    means = samples.mean(axis=1)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(values)), low, high


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via scipy if available, else numpy fallback."""
    try:
        from scipy.stats import spearmanr

        # scipy's spearmanr returns nan with stdev=0; handle locally.
        if len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
            return 0.0
        rho, _ = spearmanr(x, y)
        return float(rho) if rho == rho else 0.0  # NaN check
    except ImportError:
        # Numpy fallback.
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        rx_mean = rx.mean()
        ry_mean = ry.mean()
        num = float(np.sum((rx - rx_mean) * (ry - ry_mean)))
        denom = float(np.sqrt(np.sum((rx - rx_mean) ** 2) * np.sum((ry - ry_mean) ** 2)))
        return num / denom if denom > 0 else 0.0


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n_iter: int = BOOTSTRAP_N, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap (point, low, high) for Spearman ρ."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_iter, n))
    rhos = np.array([_spearman_rho(x[i], y[i]) for i in idx])
    point = _spearman_rho(x, y)
    return float(point), float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))


def _permutation_p_spearman(
    x: np.ndarray, y: np.ndarray, n_iter: int = PERMUTATION_N, seed: int = 42
) -> float:
    """Permutation p-value for Spearman ρ > 0 (one-sided)."""
    rng = np.random.default_rng(seed)
    observed = _spearman_rho(x, y)
    n = len(x)
    if n < 3:
        return 1.0
    null_rhos = []
    for _ in range(n_iter):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        null_rhos.append(_spearman_rho(x, y_perm))
    null_rhos = np.array(null_rhos)
    # One-sided: P(null ≥ observed) for predicted positive direction.
    return float(np.mean(null_rhos >= observed))


def _partial_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Partial Spearman ρ(x, y | z): rank-residualize x and y on z, then Spearman.

    Used for §13 #9: per-bystander Δ vs nearest_neg_distance, controlling for
    cosine-to-source.
    """
    n = len(y)
    if n < 4:
        return 0.0
    # Convert to ranks.
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rz = np.argsort(np.argsort(z)).astype(float)
    # Linear regress out rz from rx and ry.

    def _residual(r, rz):
        rz_centered = rz - rz.mean()
        denom = (rz_centered**2).sum()
        if denom == 0:
            return r - r.mean()
        beta = ((r - r.mean()) * rz_centered).sum() / denom
        return r - (rz.mean() + beta * rz_centered + 0)  # subtract fitted line

    rx_res = _residual(rx, rz)
    ry_res = _residual(ry, rz)
    return _spearman_rho(rx_res, ry_res)


def _load_centroid_bundle(centroid_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load layer-20 centroid bundle written by extend_centroids.py."""
    bundle = torch.load(centroid_path, weights_only=False)
    layer = bundle.get("layer", 20)
    tensor = bundle["centroids"][layer]
    names = list(bundle["persona_names"])
    return tensor.to(torch.float32).numpy(), names


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _nearest_neg_distance(bystander: np.ndarray, neg_centroids: np.ndarray) -> float:
    """nearest_neg_distance = 1 - max(cosine_sim(bystander, n) for n in negs)."""
    sims = (
        neg_centroids
        @ bystander
        / (np.linalg.norm(neg_centroids, axis=1) * np.linalg.norm(bystander) + 1e-12)
    )
    return float(1 - sims.max())


def _load_manifest(slab_root: Path, cell_slug: str) -> dict[str, Any]:
    """Load the build_training_data manifest for ``cell_slug``."""
    paths_to_try = [
        slab_root / cell_slug / "train_pool.manifest.json",
        slab_root.parent
        / "runs"
        / "issue_448"
        / f"{cell_slug}_seed42"
        / "train_pool.manifest.json",
    ]
    for p in paths_to_try:
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"Manifest not found for cell {cell_slug!r}. Tried: {paths_to_try}")


def analyze_cell(
    cell_slug: str,
    cell_logp: dict[str, Any],
    base_logp: dict[str, Any],
    centroid_lookup: dict[str, np.ndarray],
    manifest: dict[str, Any],
    common_bystander_set: list[str],
) -> dict[str, Any]:
    """Per-cell analysis. Returns the per-cell dict for the analyze_summary."""
    # Per-persona mean log-p at end-of-canonical-response.
    cell_end = _per_persona_mean_logp(cell_logp["logp_end_of_canonical_response"])
    cell_k0 = _per_persona_mean_logp(cell_logp["logp_k0_diagnostic"])
    base_end = _per_persona_mean_logp(base_logp["logp_end_of_canonical_response"])

    # Source-self log-p + training-success floor check.
    source_self_end = cell_end[SOURCE_PERSONA]
    source_self_floor_pass = source_self_end >= SOURCE_SELF_TRAINING_FLOOR_NATS

    # Bystander Δ vector vs base. Two flavors:
    #   - 23-bystander: 24-panel minus source (excluding multi-positive personas
    #     in cells 5/6 — their self-effect is part of the source-self block,
    #     not bystander leakage).
    #   - 20-bystander common set: 24-panel minus
    #     {villain, comedian, assistant, software_engineer} per plan §13 #6.
    multi_pos = MULTI_POSITIVE_BY_CELL.get(cell_slug, [SOURCE_PERSONA])
    bystanders_23 = [p for p in EVAL_PERSONAS_24 if p not in multi_pos]
    bystanders_common = [p for p in EVAL_PERSONAS_24 if p not in common_bystander_set]
    # ↑ common_bystander_set is the 4-persona EXCLUSION list passed in
    # (= ["villain","comedian","assistant","software_engineer"]); the
    # 20-element common-set IS what's NOT in that exclusion list.

    delta_23 = np.array([cell_end[p] - base_end[p] for p in bystanders_23])
    delta_common = np.array([cell_end[p] - base_end[p] for p in bystanders_common])

    mean_23, low_23, high_23 = _bootstrap_ci(delta_23)
    mean_common, low_common, high_common = _bootstrap_ci(delta_common)

    # k=0 diagnostic vs base k=0 (matched positions).
    base_k0 = _per_persona_mean_logp(base_logp["logp_k0_diagnostic"])
    delta_k0_23 = np.array([cell_k0[p] - base_k0[p] for p in bystanders_23])

    # Nearest-neg-distance vector (Plan §4.2.5 + §6).
    neg_persona_names = manifest.get("neg_personas", [])
    neg_centroids = np.array(
        [centroid_lookup[n] for n in neg_persona_names if n in centroid_lookup]
    )
    if len(neg_centroids) == 0:
        log.warning(
            "[%s] No centroids for negative personas %s; secondary metric skipped",
            cell_slug,
            neg_persona_names,
        )
        nearest_neg = np.zeros(len(bystanders_23))
        spread_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "stdev": 0.0, "iqr": 0.0}
        degenerate = True
        spearman: dict[str, Any] = {"skipped_reason": "no_neg_centroids"}
    else:
        nearest_neg = np.array(
            [
                _nearest_neg_distance(centroid_lookup[p], neg_centroids)
                if p in centroid_lookup
                else 0.0
                for p in bystanders_23
            ]
        )
        spread_stats = {
            "min": float(nearest_neg.min()),
            "max": float(nearest_neg.max()),
            "mean": float(nearest_neg.mean()),
            "stdev": float(nearest_neg.std(ddof=1)) if len(nearest_neg) > 1 else 0.0,
            "iqr": float(np.quantile(nearest_neg, 0.75) - np.quantile(nearest_neg, 0.25)),
        }
        degenerate = (
            spread_stats["stdev"] < DEGENERACY_STDEV_MAX or spread_stats["iqr"] < DEGENERACY_IQR_MAX
        )
        if degenerate:
            log.warning(
                "[%s] Secondary degenerate (stdev=%.4f, iqr=%.4f); skipping Spearman",
                cell_slug,
                spread_stats["stdev"],
                spread_stats["iqr"],
            )
            spearman = {
                "skipped_reason": "degenerate",
                "stdev": spread_stats["stdev"],
                "iqr": spread_stats["iqr"],
            }
        else:
            rho_point, rho_low, rho_high = _bootstrap_spearman(nearest_neg, delta_23)
            rho_p = _permutation_p_spearman(nearest_neg, delta_23)

            # Partial Spearman controlling for cosine-to-source (§13 #9).
            source_centroid = centroid_lookup.get(SOURCE_PERSONA)
            if source_centroid is not None:
                cos_to_source = np.array(
                    [
                        _cosine(centroid_lookup[p], source_centroid)
                        if p in centroid_lookup
                        else 0.0
                        for p in bystanders_23
                    ]
                )
                partial_rho = _partial_spearman(delta_23, nearest_neg, cos_to_source)
            else:
                partial_rho = float("nan")
            spearman = {
                "rho_point": rho_point,
                "rho_ci_low": rho_low,
                "rho_ci_high": rho_high,
                "permutation_p_one_sided": rho_p,
                "partial_rho_controlling_for_cosine_to_source": float(partial_rho),
                "n": len(bystanders_23),
            }

    return {
        "cell": cell_slug,
        "source_self_mean_logp_end_of_canonical_response": source_self_end,
        "source_self_training_floor_pass": bool(source_self_floor_pass),
        "source_self_training_floor_nats": SOURCE_SELF_TRAINING_FLOOR_NATS,
        "n_bystanders_23": len(bystanders_23),
        "mean_bystander_delta_23": mean_23,
        "ci_bystander_delta_23": [low_23, high_23],
        "ci_halfwidth_23": (high_23 - low_23) / 2,
        "n_bystanders_common": len(bystanders_common),
        "mean_bystander_delta_common_20": mean_common,
        "ci_bystander_delta_common_20": [low_common, high_common],
        "mean_bystander_delta_k0_diagnostic_23": float(delta_k0_23.mean()),
        "nearest_neg_distance_spread": spread_stats,
        "degenerate": bool(degenerate),
        "secondary_spearman": spearman,
        "fraction_of_training_rows_from_topup": manifest.get(
            "fraction_of_training_rows_from_topup", {}
        ),
        "per_bystander_delta_23": {
            p: float(d) for p, d in zip(bystanders_23, delta_23, strict=True)
        },
        "per_bystander_nearest_neg_distance_23": (
            {p: float(d) for p, d in zip(bystanders_23, nearest_neg, strict=True)}
            if not isinstance(spearman.get("skipped_reason"), str)
            or spearman.get("skipped_reason") != "no_neg_centroids"
            else {}
        ),
    }


def _knob_axes() -> dict[str, list[tuple[str, int]]]:
    """4 knobs × (cell_slug, knob_level) traces."""
    knob_axes: dict[str, list[tuple[str, int]]] = {
        "pos_ex_per_persona": [],
        "pos_personas": [],
        "neg_ex_per_persona": [],
        "neg_personas": [],
    }
    for slug, _name, pos_ex, pos_p, neg_ex, neg_p in CELL_SPECS:
        # Anchor sits on every knob's axis.
        is_anchor = slug == "c1_anchor"
        if is_anchor or (pos_p == 1 and neg_ex == 200 and neg_p == 2):
            knob_axes["pos_ex_per_persona"].append((slug, pos_ex))
        if is_anchor or (pos_ex == 200 and neg_ex == 200 and neg_p == 2):
            knob_axes["pos_personas"].append((slug, pos_p))
        if is_anchor or (pos_ex == 200 and pos_p == 1 and neg_p == 2):
            knob_axes["neg_ex_per_persona"].append((slug, neg_ex))
        if is_anchor or (pos_ex == 200 and pos_p == 1 and neg_ex == 200):
            knob_axes["neg_personas"].append((slug, neg_p))
    # Sort each axis by knob level and dedupe by slug (anchor may double-up).
    for knob, pairs in knob_axes.items():
        seen: set[str] = set()
        uniq = []
        for slug, level in pairs:
            if slug in seen:
                continue
            seen.add(slug)
            uniq.append((slug, level))
        knob_axes[knob] = sorted(uniq, key=lambda p: p[1])
    return knob_axes


def _headline_count(
    per_cell_mean_delta: dict[str, float],
    knob_axes: dict[str, list[tuple[str, int]]],
    delta_range_threshold: float,
) -> int:
    """Count knobs that are monotone AND |Δrange| ≥ threshold."""
    count = 0
    for _knob, axis in knob_axes.items():
        if len(axis) < 2:
            continue
        vals = [per_cell_mean_delta[slug] for slug, _ in axis]
        # Monotone (signs of consecutive deltas all same and non-zero).
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            delta_range = max(vals) - min(vals)
            if delta_range >= delta_range_threshold:
                count += 1
    return count


def _permutation_null_headline(
    cell_slugs: list[str],
    cell_mean_deltas: list[float],
    knob_axes: dict[str, list[tuple[str, int]]],
    delta_range_threshold: float,
    n_iter: int = PERMUTATION_N,
    seed: int = 42,
) -> dict[str, Any]:
    """Permutation null: shuffle cell-mean-Δs across cells, recompute headline."""
    rng = np.random.default_rng(seed)
    deltas = np.array(cell_mean_deltas)
    null_counts = []
    for _ in range(n_iter):
        perm = rng.permutation(len(deltas))
        shuffled = {slug: float(deltas[perm[i]]) for i, slug in enumerate(cell_slugs)}
        null_counts.append(_headline_count(shuffled, knob_axes, delta_range_threshold))
    null_counts = np.array(null_counts)
    observed = _headline_count(
        {slug: float(d) for slug, d in zip(cell_slugs, cell_mean_deltas, strict=True)},
        knob_axes,
        delta_range_threshold,
    )
    return {
        "headline_observed": int(observed),
        "headline_null_median": float(np.median(null_counts)),
        "headline_null_95pct_upper": float(np.quantile(null_counts, 0.95)),
        "empirical_p_value_one_sided": float(np.mean(null_counts >= observed)),
        "n_permutations": int(n_iter),
        "delta_range_threshold_nats": float(delta_range_threshold),
    }


def _make_figures(
    summary: dict[str, Any],
    knob_axes: dict[str, list[tuple[str, int]]],
    figures_dir: Path,
) -> dict[str, str]:
    """Generate the 4 per-knob line plots + auxiliary figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}

    # Per-knob line plots.
    for knob, axis in knob_axes.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        levels = [lvl for _, lvl in axis]
        means = [summary["per_cell"][slug]["mean_bystander_delta_23"] for slug, _ in axis]
        ci_low = [summary["per_cell"][slug]["ci_bystander_delta_23"][0] for slug, _ in axis]
        ci_high = [summary["per_cell"][slug]["ci_bystander_delta_23"][1] for slug, _ in axis]
        ax.errorbar(
            levels,
            means,
            yerr=[
                [m - lo for m, lo in zip(means, ci_low, strict=True)],
                [hi - m for m, hi in zip(means, ci_high, strict=True)],
            ],
            marker="o",
            capsize=4,
        )
        ax.set_xlabel(knob.replace("_", " "))
        ax.set_ylabel('Mean bystander log p(" ※") Δ vs base (nats)')
        ax.set_title(f"Knob: {knob.replace('_', ' ')}")
        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = figures_dir / f"per_knob_{knob}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        figures[knob] = str(out_path)

    # nearest_neg_distance spread overview.
    fig, ax = plt.subplots(figsize=(8, 4))
    cell_slugs = [s for s, _, _, _, _, _ in CELL_SPECS]
    stdevs = [summary["per_cell"][s]["nearest_neg_distance_spread"]["stdev"] for s in cell_slugs]
    iqrs = [summary["per_cell"][s]["nearest_neg_distance_spread"]["iqr"] for s in cell_slugs]
    x = np.arange(len(cell_slugs))
    ax.bar(x - 0.2, stdevs, width=0.4, label="stdev")
    ax.bar(x + 0.2, iqrs, width=0.4, label="IQR")
    ax.axhline(DEGENERACY_STDEV_MAX, color="red", linestyle=":", label="stdev floor")
    ax.axhline(DEGENERACY_IQR_MAX, color="orange", linestyle=":", label="IQR floor")
    ax.set_xticks(x)
    ax.set_xticklabels(cell_slugs, rotation=45, ha="right")
    ax.set_ylabel("nearest_neg_distance spread")
    ax.set_title("Per-cell nearest_neg_distance spread (degeneracy guard)")
    ax.legend()
    fig.tight_layout()
    out_path = figures_dir / "nearest_neg_distance_spread.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    figures["nearest_neg_distance_spread"] = str(out_path)

    # Per-cell Spearman ρ strip.
    fig, ax = plt.subplots(figsize=(8, 4))
    rhos = []
    cis_low = []
    cis_high = []
    labels = []
    for s in cell_slugs:
        ssp = summary["per_cell"][s]["secondary_spearman"]
        if "rho_point" in ssp:
            rhos.append(ssp["rho_point"])
            cis_low.append(ssp["rho_ci_low"])
            cis_high.append(ssp["rho_ci_high"])
            labels.append(s)
    if rhos:
        x = np.arange(len(labels))
        ax.errorbar(
            x,
            rhos,
            yerr=[
                [r - lo for r, lo in zip(rhos, cis_low, strict=True)],
                [hi - r for r, hi in zip(rhos, cis_high, strict=True)],
            ],
            fmt="o",
            capsize=4,
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Spearman ρ(per-bystander Δ, nearest_neg_distance)")
        ax.set_title("Per-cell secondary metric (predicted ρ > 0)")
        fig.tight_layout()
        out_path = figures_dir / "per_cell_rho_strip.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        figures["per_cell_rho_strip"] = str(out_path)

    return figures


def run_analysis(
    slab_root: Path,
    centroids_path: Path,
    figures_dir: Path,
    out_path: Path,
) -> dict[str, Any]:
    """End-to-end analysis. Writes ``out_path`` + figures."""
    # Load base panel.
    base_path = slab_root / "base" / "marker_logprob.json"
    base_logp = _load_logp(base_path)
    log.info("Loaded base panel from %s", base_path)

    # Load centroid bundle.
    centroids, persona_names = _load_centroid_bundle(centroids_path)
    centroid_lookup = {n: centroids[i] for i, n in enumerate(persona_names)}
    log.info(
        "Loaded centroids (%d personas, dim=%d) from %s",
        len(persona_names),
        centroids.shape[1],
        centroids_path,
    )

    # Common-bystander exclusion list per §13 #6.
    common_excl = ["villain", "comedian", "assistant", "software_engineer"]

    # Per-cell analysis.
    per_cell: dict[str, dict[str, Any]] = {}
    cell_slugs: list[str] = []
    cell_mean_deltas: list[float] = []
    for slug, _name, _pe, _pp, _ne, _np in CELL_SPECS:
        eval_path = slab_root / slug / "marker_logprob.json"
        if not eval_path.exists():
            log.warning("Skipping cell %s — eval JSON missing at %s", slug, eval_path)
            continue
        cell_logp = _load_logp(eval_path)
        manifest = _load_manifest(slab_root, slug)
        cell_result = analyze_cell(
            slug, cell_logp, base_logp, centroid_lookup, manifest, common_excl
        )
        per_cell[slug] = cell_result
        cell_slugs.append(slug)
        cell_mean_deltas.append(cell_result["mean_bystander_delta_23"])

    # Calibrated threshold (M4): max(2, 1.5 × median per-cell CI half-width).
    half_widths = [per_cell[s]["ci_halfwidth_23"] for s in cell_slugs]
    median_hw = float(np.median(half_widths)) if half_widths else 0.0
    calibrated_threshold = max(HEADLINE_MIN_DELTA_RANGE_NATS, HEADLINE_CI_MULTIPLIER * median_hw)
    log.info(
        "Calibrated headline threshold: max(%.2f, %.2f × %.4f) = %.4f nats",
        HEADLINE_MIN_DELTA_RANGE_NATS,
        HEADLINE_CI_MULTIPLIER,
        median_hw,
        calibrated_threshold,
    )

    # Per-knob axes + monotonicity test.
    knob_axes = _knob_axes()
    per_knob: dict[str, dict[str, Any]] = {}
    for knob, axis in knob_axes.items():
        levels = [lvl for _, lvl in axis]
        slugs = [s for s, _ in axis]
        means = [per_cell[s]["mean_bystander_delta_23"] for s in slugs if s in per_cell]
        valid = len(means) == len(slugs)
        if not valid:
            per_knob[knob] = {"valid": False, "reason": "missing_cells"}
            continue
        diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
        monotone_up = all(d > 0 for d in diffs)
        monotone_down = all(d < 0 for d in diffs)
        delta_range = max(means) - min(means)
        per_knob[knob] = {
            "valid": True,
            "axis_cells": slugs,
            "axis_levels": levels,
            "mean_deltas": means,
            "monotone_up": monotone_up,
            "monotone_down": monotone_down,
            "delta_range_nats": float(delta_range),
            "fires": (monotone_up or monotone_down) and delta_range >= calibrated_threshold,
        }

    # Permutation null on the headline count (M4).
    permutation_null = _permutation_null_headline(
        cell_slugs, cell_mean_deltas, knob_axes, calibrated_threshold
    )

    # Build summary.
    summary: dict[str, Any] = {
        "schema": "issue_448.analyze_summary v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "per_cell": per_cell,
        "per_knob": per_knob,
        "headline": permutation_null,
        "calibrated_threshold_nats": calibrated_threshold,
        "median_per_cell_ci_halfwidth_nats": median_hw,
        "cell_slugs_with_eval": cell_slugs,
        "n_cells_analyzed": len(per_cell),
        "n_cells_expected": len(CELL_SPECS),
        "bystander_panel_sizes": {
            "23_bystander": "24-panel minus source (and any multi-positive personas)",
            "20_bystander_common": (
                "24-panel minus {villain, comedian, assistant, software_engineer}"
            ),
        },
    }

    figures = _make_figures(summary, knob_axes, figures_dir)
    summary["figures"] = figures

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "Wrote analyze summary (%d/%d cells; headline obs=%d null_median=%.1f p=%.3f) → %s",
        summary["n_cells_analyzed"],
        summary["n_cells_expected"],
        summary["headline"]["headline_observed"],
        summary["headline"]["headline_null_median"],
        summary["headline"]["empirical_p_value_one_sided"],
        out_path,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_448"),
    )
    parser.add_argument(
        "--centroids",
        type=Path,
        default=Path("eval_results/issue_448/centroids/centroids_layer20.pt"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/issue_448"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="analyze_summary.json path (default: <slab-root>/analyze_summary.json)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=analyze] %(message)s",
        stream=sys.stdout,
    )
    out_path = args.output or (args.slab_root / "analyze_summary.json")
    run_analysis(
        slab_root=args.slab_root,
        centroids_path=args.centroids,
        figures_dir=args.figures_dir,
        out_path=out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
