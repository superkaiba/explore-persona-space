# ruff: noqa: RUF001, RUF002, RUF003
"""Recompute the issue #448 secondary metric on ABSOLUTE post-training
log p(marker) rather than Δ = post − base.

The original analyzer correlated per-bystander Δ (post − base) against
cosine-distance-to-nearest-trained-negative. That Δ is contaminated by
ceiling saturation: post-training emission concentrates near the
log-prob ceiling (anchor mean −1.18 nats, band [−1.69, −0.61]), so
Spearman(base, Δ) = −0.93 for the anchor (verified). The two outlier
personas comedian + french_person — the HIGHEST and 2ND-HIGHEST base
priors — get the SMALLEST Δs by headroom alone, and the original
secondary ρ ≈ −0.55 was largely those two outliers anti-correlated
with their base prior, NOT a geometric correction effect.

This script recomputes per-cell Spearman ρ on:
  y = per-bystander absolute post-training mean log p(marker)
       at end_of_canonical_response.
  x = per-bystander cosine distance to the nearest trained contrastive
       negative persona (centroid space, layer 20) — IDENTICAL to the
       analyzer's nearest_neg_distance computation, reused verbatim.

It also reports the partial-ρ controlling for cosine-to-source, and
saves a side-by-side per-cell table (ρ_Δ vs ρ_abs).

Writes:
  eval_results/issue_448/secondary_absolute_summary.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
    CELL_SPECS,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    EVAL_PERSONAS_24,
)

BOOTSTRAP_N = 10_000
PERMUTATION_N = 10_000
DEGENERACY_STDEV_MAX = 0.02
DEGENERACY_IQR_MAX = 0.03

MULTI_POSITIVE_BY_CELL: dict[str, list[str]] = {
    "c5_pos_personas_2": ["villain", "comedian"],
    "c6_pos_personas_4": ["villain", "comedian", "assistant", "software_engineer"],
}

# Manual per-cell negative personas (read off the same manifests the
# analyzer reads). The dispatcher placed them under /workspace/runs/...
# which is no longer reachable from the local VM; we hard-code the
# anchor's negatives + each cell's known negative set from the plan +
# analyze_summary.json's per_bystander_nearest_neg_distance_23 (which
# was computed against the manifest-provided neg set).
#
# Verified consistency with analyze_summary.json by spot-checking that
# the resulting nearest_neg_distance vectors match the analyzer's
# per_bystander_nearest_neg_distance_23 for each cell.
ANCHOR_NEGS = ["medical_doctor", "police_officer"]
# c10/c11 neg sets recovered by inspection of analyze_summary.json's
# per_bystander_nearest_neg_distance_23 (personas with distance ~0 ARE
# the negatives). Verified: my recomputed distances match analyzer's to
# < 1e-4 for every persona in every cell (manual diff against
# analyze_summary.json).
CELL_NEGS: dict[str, list[str]] = {
    "c1_anchor": ANCHOR_NEGS,
    "c2_pos_ex_100": ANCHOR_NEGS,
    "c3_pos_ex_400": ANCHOR_NEGS,
    "c4_pos_ex_800": ANCHOR_NEGS,
    "c5_pos_personas_2": ANCHOR_NEGS,
    "c6_pos_personas_4": ANCHOR_NEGS,
    "c7_neg_ex_100": ANCHOR_NEGS,
    "c8_neg_ex_400": ANCHOR_NEGS,
    "c9_neg_ex_800": ANCHOR_NEGS,
    "c10_neg_personas_4": [
        "medical_doctor",
        "police_officer",
        "qwen_default",
        "comedian",
    ],
    "c11_neg_personas_8": [
        "medical_doctor",
        "police_officer",
        "qwen_default",
        "comedian",
        "librarian",
        "french_person",
        "zelthari_scholar",
        "software_engineer",
    ],
}


def per_persona_mean(logp_by_persona: dict[str, dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for persona, by_q in logp_by_persona.items():
        vals = [float(v) for v in by_q.values()]
        if not vals:
            raise ValueError(f"Empty per-q for {persona!r}")
        out[persona] = float(np.mean(vals))
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def nearest_neg_distance(bystander: np.ndarray, neg_centroids: np.ndarray) -> float:
    sims = (
        neg_centroids
        @ bystander
        / (np.linalg.norm(neg_centroids, axis=1) * np.linalg.norm(bystander) + 1e-12)
    )
    return float(1 - sims.max())


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n_iter: int = BOOTSTRAP_N, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0
    rho_obs, _ = spearmanr(x, y)
    rho_obs = float(rho_obs) if not np.isnan(rho_obs) else 0.0
    idx = rng.integers(0, n, size=(n_iter, n))
    rhos = []
    for i in range(n_iter):
        xb = x[idx[i]]
        yb = y[idx[i]]
        if len(set(xb.tolist())) < 2 or len(set(yb.tolist())) < 2:
            continue
        r, _ = spearmanr(xb, yb)
        if not np.isnan(r):
            rhos.append(float(r))
    if not rhos:
        return rho_obs, 0.0, 0.0
    rhos_arr = np.array(rhos)
    return rho_obs, float(np.quantile(rhos_arr, 0.025)), float(np.quantile(rhos_arr, 0.975))


def permutation_p_spearman(
    x: np.ndarray, y: np.ndarray, n_iter: int = PERMUTATION_N, seed: int = 42
) -> float:
    """Two-sided permutation p for Spearman ρ ≠ 0.

    Returns P(|null| ≥ |observed|) under the null of no association.
    Two-sided is honest here because the predicted direction was ρ > 0
    (geometric generalisation), but the data direction is ρ < 0, so
    one-sided in either direction would be a post-hoc fishing call.
    """
    rng = np.random.default_rng(seed)
    observed, _ = spearmanr(x, y)
    if np.isnan(observed):
        return 1.0
    observed = abs(float(observed))
    n = len(x)
    if n < 3:
        return 1.0
    null_abs = []
    for _ in range(n_iter):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        r, _ = spearmanr(x, y_perm)
        if not np.isnan(r):
            null_abs.append(abs(float(r)))
    null_arr = np.array(null_abs)
    return float(np.mean(null_arr >= observed))


def partial_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Partial Spearman ρ(x, y | z) — rank-residualise then correlate."""
    n = len(y)
    if n < 4:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rz = np.argsort(np.argsort(z)).astype(float)
    rz_c = rz - rz.mean()
    denom = (rz_c**2).sum()
    if denom == 0:
        rx_res = rx - rx.mean()
        ry_res = ry - ry.mean()
    else:
        bx = ((rx - rx.mean()) * rz_c).sum() / denom
        by_ = ((ry - ry.mean()) * rz_c).sum() / denom
        rx_res = rx - (rz.mean() + bx * rz_c)
        ry_res = ry - (rz.mean() + by_ * rz_c)
    r, _ = spearmanr(rx_res, ry_res)
    return float(r) if not np.isnan(r) else 0.0


def load_centroids() -> tuple[dict[str, np.ndarray], int]:
    bundle = torch.load(
        Path("eval_results/issue_448/centroids/centroids_layer20.pt"),
        weights_only=False,
    )
    layer = bundle.get("layer", 20)
    tensor = bundle["centroids"][layer].to(torch.float32).numpy()
    names = list(bundle["persona_names"])
    return {n: tensor[i] for i, n in enumerate(names)}, int(tensor.shape[1])


def analyze_cell_absolute(
    cell_slug: str,
    centroid_lookup: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Per-cell secondary ρ on ABSOLUTE post log p(marker).

    Returns a per-cell dict matching the shape of the per-cell entry in
    analyze_summary.json so a downstream plot can read either summary.
    """
    cell_path = Path(f"eval_results/issue_448/{cell_slug}/marker_logprob.json")
    base_path = Path("eval_results/issue_448/base/marker_logprob.json")
    cell_logp = json.loads(cell_path.read_text())
    base_logp = json.loads(base_path.read_text())

    cell_end = per_persona_mean(cell_logp["logp_end_of_canonical_response"])
    base_end = per_persona_mean(base_logp["logp_end_of_canonical_response"])

    multi_pos = MULTI_POSITIVE_BY_CELL.get(cell_slug, [SOURCE_PERSONA])
    bystanders = sorted(p for p in EVAL_PERSONAS_24 if p not in multi_pos)

    y_abs = np.array([cell_end[p] for p in bystanders])
    y_delta = np.array([cell_end[p] - base_end[p] for p in bystanders])
    base_v = np.array([base_end[p] for p in bystanders])

    # Nearest-neg-distance — same computation as analyze.py.
    neg_personas = CELL_NEGS[cell_slug]
    neg_centroids = np.array([centroid_lookup[n] for n in neg_personas])
    nearest_neg = np.array(
        [nearest_neg_distance(centroid_lookup[p], neg_centroids) for p in bystanders]
    )

    spread = {
        "min": float(nearest_neg.min()),
        "max": float(nearest_neg.max()),
        "mean": float(nearest_neg.mean()),
        "stdev": float(nearest_neg.std(ddof=1)),
        "iqr": float(np.quantile(nearest_neg, 0.75) - np.quantile(nearest_neg, 0.25)),
    }
    degenerate = spread["stdev"] < DEGENERACY_STDEV_MAX or spread["iqr"] < DEGENERACY_IQR_MAX

    # Cosine-to-source for partial.
    source_cent = centroid_lookup[SOURCE_PERSONA]
    cos_to_source = np.array([cosine(centroid_lookup[p], source_cent) for p in bystanders])

    if degenerate:
        return {
            "cell": cell_slug,
            "n_bystanders": len(bystanders),
            "bystanders": bystanders,
            "y_post_abs_logp": y_abs.tolist(),
            "y_delta": y_delta.tolist(),
            "base_logp": base_v.tolist(),
            "x_nearest_neg_distance": nearest_neg.tolist(),
            "x_cosine_to_source": cos_to_source.tolist(),
            "neg_personas": neg_personas,
            "spread_stats": spread,
            "degenerate": True,
            "spearman_absolute": {"skipped_reason": "degenerate"},
            "spearman_delta": {"skipped_reason": "degenerate"},
        }

    # ρ on absolute post.
    rho_abs, rho_abs_lo, rho_abs_hi = bootstrap_spearman(nearest_neg, y_abs)
    p_abs = permutation_p_spearman(nearest_neg, y_abs)
    partial_abs = partial_spearman(y_abs, nearest_neg, cos_to_source)

    # ρ on Δ (for side-by-side).
    rho_delta, rho_delta_lo, rho_delta_hi = bootstrap_spearman(nearest_neg, y_delta)
    p_delta = permutation_p_spearman(nearest_neg, y_delta)
    partial_delta = partial_spearman(y_delta, nearest_neg, cos_to_source)

    # ρ(base, post-abs) and ρ(base, Δ) — the ceiling-saturation diagnostic.
    rho_base_post, _ = spearmanr(base_v, y_abs)
    rho_base_delta, _ = spearmanr(base_v, y_delta)

    return {
        "cell": cell_slug,
        "n_bystanders": len(bystanders),
        "bystanders": bystanders,
        "y_post_abs_logp": y_abs.tolist(),
        "y_delta": y_delta.tolist(),
        "base_logp": base_v.tolist(),
        "x_nearest_neg_distance": nearest_neg.tolist(),
        "x_cosine_to_source": cos_to_source.tolist(),
        "neg_personas": neg_personas,
        "spread_stats": spread,
        "degenerate": False,
        "mean_post_abs_logp": float(y_abs.mean()),
        "mean_delta": float(y_delta.mean()),
        "post_abs_band_lo_hi": [float(y_abs.min()), float(y_abs.max())],
        "spearman_absolute": {
            "rho_point": float(rho_abs),
            "rho_ci_low": float(rho_abs_lo),
            "rho_ci_high": float(rho_abs_hi),
            "permutation_p_two_sided": float(p_abs),
            "partial_rho_controlling_for_cosine_to_source": float(partial_abs),
            "n": len(bystanders),
        },
        "spearman_delta": {
            "rho_point": float(rho_delta),
            "rho_ci_low": float(rho_delta_lo),
            "rho_ci_high": float(rho_delta_hi),
            "permutation_p_two_sided": float(p_delta),
            "partial_rho_controlling_for_cosine_to_source": float(partial_delta),
            "n": len(bystanders),
        },
        "rho_base_to_post_abs": float(rho_base_post) if not np.isnan(rho_base_post) else 0.0,
        "rho_base_to_delta": float(rho_base_delta) if not np.isnan(rho_base_delta) else 0.0,
    }


def main() -> None:
    centroid_lookup, dim = load_centroids()
    print(f"Loaded centroids, dim={dim}")

    out: dict[str, Any] = {
        "schema": "issue_448.secondary_absolute v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "note": (
            "Per-cell Spearman ρ on ABSOLUTE post-training mean log p( ※) at "
            "end_of_canonical_response, vs per-bystander cosine distance to the "
            "nearest trained contrastive negative persona (layer-20 centroids). "
            "Recomputed because the original Δ-based DV is anti-correlated with "
            "base prior at ρ ≈ −0.93 in the anchor cell — ceiling saturation."
        ),
        "per_cell": {},
    }

    cell_slugs = [s for s, *_ in CELL_SPECS]
    print(
        f"{'cell':25s}  {'ρ_abs':>7s}  {'CI_abs':>22s}  {'p_abs':>7s}  "
        f"{'partial':>8s}  {'ρ_Δ (prev)':>10s}  {'ρ(base,post)':>12s}"
    )
    for slug in cell_slugs:
        r = analyze_cell_absolute(slug, centroid_lookup)
        out["per_cell"][slug] = r
        if r["degenerate"]:
            print(f"{slug:25s}  DEGENERATE (stdev={r['spread_stats']['stdev']:.4f})")
            continue
        sa = r["spearman_absolute"]
        sd = r["spearman_delta"]
        ci_str = f"[{sa['rho_ci_low']:+.3f},{sa['rho_ci_high']:+.3f}]"
        print(
            f"{slug:25s}  {sa['rho_point']:+.3f}  {ci_str:>22s}  "
            f"{sa['permutation_p_two_sided']:.3f}  "
            f"{sa['partial_rho_controlling_for_cosine_to_source']:+.3f}  "
            f"{sd['rho_point']:+10.3f}  {r['rho_base_to_post_abs']:+12.3f}"
        )

    out_path = Path("eval_results/issue_448/secondary_absolute_summary.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
