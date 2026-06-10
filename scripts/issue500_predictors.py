#!/usr/bin/env python3
"""Issue #500 analyzer-facing diagnostics + headline statistics.

Computes per-arm × per-bystander:
  - leak_rate                (from aggregate_cleaned.json; primary DV)
  - prior_logprob            (length-norm base log P(taught completion | T_bystander))
  - prior_5way               (5-way base ``stated_seven`` rate; on-policy variant)
  - cos_to_source            (layer-21 persona-vector cosine to the arm source)
  - cos_to_home              (layer-21 persona-vector cosine to local_historian,
                              the panel's max-base-prior persona = "home")
  - completion_length        (mean tokens emitted by the trained model)
  - on_topic_fraction        (Claude judge: did the completion talk about the
                              courthouse at all)

Reports per arm:
  - Spearman ρ(prior_logprob, leak)        with cluster-bootstrap CI
  - Spearman ρ(prior_5way,    leak)        with cluster-bootstrap CI
  - Spearman ρ(cos_to_source, leak)        with cluster-bootstrap CI
  - Spearman ρ(cos_to_home,   leak)        with cluster-bootstrap CI
  - Partial Spearman ρ(cos_to_source, leak | prior_logprob)
  - Collinearity gate: Pearson(|cos_to_source|, prior_logprob)
  - Standardized OLS  z(leak) ~ z(prior_logprob) + z(cos_to_source)
  - Engagement-adjusted ρ (partial out completion_length + on_topic_fraction)

Cross-arm:
  - Δρ_AB = ρ_cos(B) - ρ_cos(A)            with cluster-bootstrap 90% / 95% CIs
            (persona-resampling AND seed-resampling diagnostics reported
             separately; they answer different uncertainty questions)
  - Δρ_AC, Δρ_CB analogously

Inputs:
  - eval_results/issue_500/<arm>/aggregate_cleaned.json
  - eval_results/issue_500/<arm>/persona_distance/results.json (cosine + JS)
  - eval_results/issue_500/distance_to_home.json              (cos_to_local_historian)
  - eval_results/issue_500/bystander_logprob/logprob_results.json
  - eval_results/issue_500/<arm>/engagement_covariates.json   (length + on_topic)

This script is analyzer-input only; the analyzer (downstream of /issue Step 7)
reads the JSON output and produces the figures + clean-result body.
"""

# ruff: noqa: RUF001, RUF002, RUF003, C901
# (greek + arrow + multiplication-sign characters intentional in docstrings;
#  per-arm metrics fn is purposefully long to keep the per-arm analysis flow
#  in one place for the analyzer to read.)

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

ARMS: tuple[str, ...] = (
    "marine_biologist",
    "local_resident",
    "courthouse_architecture_historian",
)
HOME_PERSONA = "local_historian"
LAYER_HEADLINE = "21"


# ---------------------------------------------------------------------------
# Statistics primitives (Spearman / partial Spearman / cluster bootstrap)
# ---------------------------------------------------------------------------
def _rankdata(x: list[float]) -> list[float]:
    """Average-rank tie-handling, scipy-free."""
    arr = np.asarray(x, dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # Tie-correct: average ranks within ties.
    uniq, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    sums = np.zeros_like(uniq, dtype=float)
    for i, r in zip(inv, ranks, strict=True):
        sums[i] += r
    avg = sums / counts
    return [float(avg[i]) for i in inv]


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return float("nan")
    rx = np.asarray(_rankdata(x))
    ry = np.asarray(_rankdata(y))
    sx = rx - rx.mean()
    sy = ry - ry.mean()
    denom = math.sqrt((sx * sx).sum() * (sy * sy).sum())
    return float((sx * sy).sum() / denom) if denom > 0 else float("nan")


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return float("nan")
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a -= a.mean()
    b -= b.mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _partial_spearman(x: list[float], y: list[float], z: list[float]) -> float:
    """Standard partial Spearman ρ(x, y | z).

    Definition (canonical): rank-transform x, y, z; OLS-residualize the rank
    vectors of x and y against the rank vector of z; the partial Spearman is
    the PEARSON correlation of those rank-residuals.

    Round-3 BUG-#3 fix: the previous implementation re-ranked the residuals
    and applied Spearman to the re-ranked residuals, which is NOT the
    standard partial-Spearman statistic. Pearson on rank-residuals matches
    the textbook definition (e.g. pingouin.partial_corr method='spearman').
    """
    if len(x) < 3 or len(set(map(len, (x, y, z)))) != 1:
        return float("nan")
    rx = np.asarray(_rankdata(x))
    ry = np.asarray(_rankdata(y))
    rz = np.asarray(_rankdata(z))
    # Residuals of x and y after OLS regression on RANK z.
    A = np.column_stack([np.ones_like(rz), rz])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    res_x = rx - A @ bx
    res_y = ry - A @ by
    return _pearson(list(res_x), list(res_y))


def _partial_spearman_multi(x: list[float], y: list[float], zs: list[list[float]]) -> float:
    """Partial Spearman ρ(x, y | z1, z2, ...): rank-residualize x and y
    against the multiple rank-z controls (joint OLS on rank space), return
    Pearson of the rank-residuals.

    Round-3 BUG-#5: enables the JOINT engagement partial
    ρ(prior or cos, leak | length AND on_topic). Single-covariate
    ``_partial_spearman`` is the special case len(zs) == 1.
    """
    n = len(x)
    if n < 3 or len(y) != n or any(len(z) != n for z in zs) or not zs:
        return float("nan")
    rx = np.asarray(_rankdata(x))
    ry = np.asarray(_rankdata(y))
    rz_cols = [np.asarray(_rankdata(z)) for z in zs]
    A = np.column_stack([np.ones(n), *rz_cols])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    res_x = rx - A @ bx
    res_y = ry - A @ by
    return _pearson(list(res_x), list(res_y))


def _standardize(x: list[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    sd = a.std(ddof=1) or 1.0
    return (a - a.mean()) / sd


def _ols_two_predictor(y: list[float], x1: list[float], x2: list[float]) -> dict[str, object]:
    """Standardized OLS z(y) ~ z(x1) + z(x2). Returns betas + R^2.

    Degenerate-input gate (#541 smoke hardening): a 2-predictor-plus-intercept
    fit needs n >= 4 rows (matching the n>=4 permutation-p convention), and
    the ddof=1 z-scores are NaN at n == 1 — non-finite values reaching LAPACK
    surface as a misleading "SVD did not converge" (DLASCL rejects the
    matrix). Below-minimum n or a non-finite design returns a structured skip
    (declared, never a silent default); no-op at full-run n with finite
    inputs.
    """
    n = len(y)
    if n < 4:
        return {"status": "skipped_insufficient_n", "n": n}
    zy = _standardize(y)
    zx1 = _standardize(x1)
    zx2 = _standardize(x2)
    A = np.column_stack([np.ones_like(zx1), zx1, zx2])
    if not (np.isfinite(A).all() and np.isfinite(zy).all()):
        return {"status": "skipped_nonfinite_design", "n": n}
    coef, *_ = np.linalg.lstsq(A, zy, rcond=None)
    yhat = A @ coef
    ss_res = float(((zy - yhat) ** 2).sum())
    ss_tot = float(((zy - zy.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "intercept": float(coef[0]),
        "beta_x1_prior": float(coef[1]),
        "beta_x2_prox": float(coef[2]),
        "r_squared": r2,
    }


def _cluster_bootstrap_spearman(
    pairs: list[tuple[float, float]],
    cluster_ids: list[str],
    *,
    n_iter: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster-bootstrap CI for Spearman ρ over pairs[(x, y)] grouped by cluster.

    Resamples CLUSTERS with replacement, recomputes ρ on the assembled pairs.
    Returns mean, 5%/95%/2.5%/97.5% percentile bounds.

    Round-3 BUG-#6 fix: ``cluster_ids`` are stable strings (persona names),
    not Python ``hash()`` outputs (process-randomized). Combined with the
    fixed RNG seed this gives reproducible CIs across processes.
    """
    rng = np.random.default_rng(seed)
    clusters = sorted(set(cluster_ids))
    by_cluster: dict[str, list[tuple[float, float]]] = {c: [] for c in clusters}
    for pair, cid in zip(pairs, cluster_ids, strict=True):
        by_cluster[cid].append(pair)
    rhos: list[float] = []
    for _ in range(n_iter):
        sampled = rng.choice(len(clusters), size=len(clusters), replace=True)
        boot_pairs: list[tuple[float, float]] = []
        for idx in sampled:
            boot_pairs.extend(by_cluster[clusters[idx]])
        xs = [p[0] for p in boot_pairs]
        ys = [p[1] for p in boot_pairs]
        rho = _spearman(xs, ys)
        if not math.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return {"mean": float("nan"), "ci_low_90": float("nan"), "ci_high_90": float("nan")}
    arr = np.asarray(rhos)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci_low_90": float(np.percentile(arr, 5)),
        "ci_high_90": float(np.percentile(arr, 95)),
        "ci_low_95": float(np.percentile(arr, 2.5)),
        "ci_high_95": float(np.percentile(arr, 97.5)),
        "n_valid_iters": len(rhos),
    }


def _summarize_bootstrap(values: list[float]) -> dict[str, float]:
    """Mean / median / 90% / 95% percentile summary of a bootstrap distribution."""
    if not values:
        return {
            "mean": float("nan"),
            "ci_low_90": float("nan"),
            "ci_high_90": float("nan"),
            "n_valid_iters": 0,
        }
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci_low_90": float(np.percentile(arr, 5)),
        "ci_high_90": float(np.percentile(arr, 95)),
        "ci_low_95": float(np.percentile(arr, 2.5)),
        "ci_high_95": float(np.percentile(arr, 97.5)),
        "n_valid_iters": len(values),
    }


def _cluster_bootstrap_h3(
    points: list[dict[str, object]],
    *,
    n_iter: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Persona-cluster bootstrap CIs for the H3 diagnostics (round-3 BUG-#4).

    Resamples persona clusters from ``points`` with replacement; on each
    iteration recomputes:
      - partial Spearman ρ(cos_to_source, leak | prior_logprob)
      - standardized OLS z(leak) ~ z(prior_logprob) + z(cos_to_source):
        beta_prior, beta_prox, R^2

    Returns a dict with one sub-dict per statistic carrying the
    mean/median/90%/95% CI summary.
    """
    by_persona: dict[str, list[tuple[float, float, float]]] = {}
    for r in points:
        prior = r.get("prior_logprob", float("nan"))
        cos_v = r.get("cos_to_source", float("nan"))
        leak = r.get("leak", float("nan"))
        if any(isinstance(v, float) and math.isnan(v) for v in (prior, cos_v, leak)):
            continue
        by_persona.setdefault(str(r["persona"]), []).append(
            (float(prior), float(cos_v), float(leak))
        )
    clusters = sorted(by_persona)
    if len(clusters) < 4:
        return {"status": {"value": "skipped_too_few_personas", "n_clusters": len(clusters)}}

    rng = np.random.default_rng(seed)
    partials: list[float] = []
    beta_priors: list[float] = []
    beta_proxs: list[float] = []
    r2s: list[float] = []
    for _ in range(n_iter):
        idxs = rng.choice(len(clusters), size=len(clusters), replace=True)
        # Per-persona means over the bootstrap pick (iterate WITH multiplicity).
        prior_means: list[float] = []
        cos_means: list[float] = []
        leak_means: list[float] = []
        for i in idxs:
            rows = by_persona[clusters[i]]
            n = len(rows)
            prior_means.append(sum(t[0] for t in rows) / n)
            cos_means.append(sum(t[1] for t in rows) / n)
            leak_means.append(sum(t[2] for t in rows) / n)
        # Need 3 distinct values to avoid degenerate rank / OLS.
        if len(set(prior_means)) < 3 or len(set(cos_means)) < 3:
            continue
        partials.append(_partial_spearman(cos_means, leak_means, prior_means))
        ols = _ols_two_predictor(leak_means, prior_means, cos_means)
        if "status" in ols:
            continue  # degenerate bootstrap draw — skip-gated by the OLS guard
        beta_priors.append(ols["beta_x1_prior"])
        beta_proxs.append(ols["beta_x2_prox"])
        r2s.append(ols["r_squared"])

    return {
        "partial_spearman_cos_to_source_given_prior": _summarize_bootstrap(
            [p for p in partials if not math.isnan(p)]
        ),
        "ols_beta_prior": _summarize_bootstrap([b for b in beta_priors if not math.isnan(b)]),
        "ols_beta_prox": _summarize_bootstrap([b for b in beta_proxs if not math.isnan(b)]),
        "ols_r_squared": _summarize_bootstrap([r for r in r2s if not math.isnan(r)]),
        "n_clusters": {"value": len(clusters)},
    }


def _cross_arm_delta_rho_persona_bootstrap(
    left_points: list[dict[str, object]],
    right_points: list[dict[str, object]],
    *,
    x_field: str,
    n_iter: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Persona-resampling Δρ CI: resample personas WITH REPLACEMENT from the
    INTERSECTION of left and right panels, recompute ρ_left and ρ_right on
    the per-persona mean leak, and report Δρ = ρ_right − ρ_left.

    Per plan §6.3: "persona-resampling: how much does Δρ depend on which
    bystander personas we picked from the registry?"

    Note: round-1 BLOCKER #6 fix. The round-1 cross-arm block only reported
    `delta = ρ_right − ρ_left` with no CI.
    """
    rng = np.random.default_rng(seed)

    # Per-persona mean (across seeds) on each side.
    def _per_persona_mean(pts: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
        by: dict[str, list[tuple[float, float]]] = {}
        for r in pts:
            x = r.get(x_field, float("nan"))
            y = r.get("leak", float("nan"))
            if isinstance(x, float) and math.isnan(x):
                continue
            if isinstance(y, float) and math.isnan(y):
                continue
            by.setdefault(str(r["persona"]), []).append((float(x), float(y)))
        return {
            p: (sum(t[0] for t in v) / len(v), sum(t[1] for t in v) / len(v)) for p, v in by.items()
        }

    left_means = _per_persona_mean(left_points)
    right_means = _per_persona_mean(right_points)
    common = sorted(set(left_means) & set(right_means))
    if len(common) < 3:
        return {"status": "skipped_panel_intersection_too_small", "n_common": len(common)}

    deltas: list[float] = []
    for _ in range(n_iter):
        idxs = rng.choice(len(common), size=len(common), replace=True)
        l_xs = [left_means[common[i]][0] for i in idxs]
        l_ys = [left_means[common[i]][1] for i in idxs]
        r_xs = [right_means[common[i]][0] for i in idxs]
        r_ys = [right_means[common[i]][1] for i in idxs]
        rho_l = _spearman(l_xs, l_ys)
        rho_r = _spearman(r_xs, r_ys)
        if math.isnan(rho_l) or math.isnan(rho_r):
            continue
        deltas.append(rho_r - rho_l)
    if not deltas:
        return {"status": "no_valid_bootstrap_iters"}
    arr = np.asarray(deltas)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci_low_90": float(np.percentile(arr, 5)),
        "ci_high_90": float(np.percentile(arr, 95)),
        "ci_low_95": float(np.percentile(arr, 2.5)),
        "ci_high_95": float(np.percentile(arr, 97.5)),
        "n_common_personas": len(common),
        "n_iter": n_iter,
        "n_valid": len(deltas),
    }


def _cross_arm_delta_rho_seed_bootstrap(
    left_points: list[dict[str, object]],
    right_points: list[dict[str, object]],
    *,
    x_field: str,
    n_iter: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Seed-resampling Δρ CI: resample SEEDS with replacement from the
    seed sets present on each side, recompute per-persona mean leak under the
    bootstrap seed set, then ρ_left and ρ_right.

    Per plan §6.3: "seed-resampling: how much does Δρ depend on
    training-noise variance across seeds?"
    """
    rng = np.random.default_rng(seed)

    def _seeds_in(pts: list[dict[str, object]]) -> list[int]:
        return sorted({int(r["seed"]) for r in pts if int(r["seed"]) >= 0})

    left_seeds = _seeds_in(left_points)
    right_seeds = _seeds_in(right_points)
    if len(left_seeds) < 2 or len(right_seeds) < 2:
        return {
            "status": "skipped_too_few_seeds",
            "left_n_seeds": len(left_seeds),
            "right_n_seeds": len(right_seeds),
        }

    def _index_by_seed(
        pts: list[dict[str, object]],
    ) -> dict[int, list[tuple[str, float, float]]]:
        """Bucket points by seed -> list of (persona, x, y), NaNs dropped."""
        out: dict[int, list[tuple[str, float, float]]] = {}
        for r in pts:
            s = int(r["seed"])
            x = r.get(x_field, float("nan"))
            y = r.get("leak", float("nan"))
            if isinstance(x, float) and math.isnan(x):
                continue
            if isinstance(y, float) and math.isnan(y):
                continue
            out.setdefault(s, []).append((str(r["persona"]), float(x), float(y)))
        return out

    left_by_seed = _index_by_seed(left_points)
    right_by_seed = _index_by_seed(right_points)

    def _rho_on_resample(
        by_seed: dict[int, list[tuple[str, float, float]]],
        sampled_seeds: list[int],
    ) -> float:
        """Per-persona mean of (x, y) over the SAMPLED seed list, iterating
        WITH MULTIPLICITY (round-3 BUG-#2 fix: a resample like [42,42,42]
        contributes seed 42 three times, not once).
        """
        by_persona: dict[str, list[tuple[float, float]]] = {}
        for s in sampled_seeds:
            for persona, x, y in by_seed.get(int(s), []):
                by_persona.setdefault(persona, []).append((x, y))
        if len(by_persona) < 3:
            return float("nan")
        xs = [sum(t[0] for t in v) / len(v) for v in by_persona.values()]
        ys = [sum(t[1] for t in v) / len(v) for v in by_persona.values()]
        return _spearman(xs, ys)

    deltas: list[float] = []
    for _ in range(n_iter):
        l_pick = [int(s) for s in rng.choice(left_seeds, size=len(left_seeds), replace=True)]
        r_pick = [int(s) for s in rng.choice(right_seeds, size=len(right_seeds), replace=True)]
        rho_l = _rho_on_resample(left_by_seed, l_pick)
        rho_r = _rho_on_resample(right_by_seed, r_pick)
        if math.isnan(rho_l) or math.isnan(rho_r):
            continue
        deltas.append(rho_r - rho_l)
    if not deltas:
        return {"status": "no_valid_bootstrap_iters"}
    arr = np.asarray(deltas)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci_low_90": float(np.percentile(arr, 5)),
        "ci_high_90": float(np.percentile(arr, 95)),
        "ci_low_95": float(np.percentile(arr, 2.5)),
        "ci_high_95": float(np.percentile(arr, 97.5)),
        "left_n_seeds": len(left_seeds),
        "right_n_seeds": len(right_seeds),
        "n_iter": n_iter,
        "n_valid": len(deltas),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_logprob_priors(panel_path: Path) -> dict[str, float]:
    """Per-persona base length-norm log-prob prior (#500-widened panel)."""
    data = json.loads(panel_path.read_text())
    return {
        p: float(d["mean_logprob_per_tok"])
        for p, d in data["summary"].items()
        if d.get("n_rows", 0) > 0
    }


def _load_cosines(persona_distance_path: Path) -> dict[str, float]:
    """Per-bystander cos_to_REFERENCE at layer 21 (on-topic)."""
    data = json.loads(persona_distance_path.read_text())
    return {
        persona: float(per_layer[LAYER_HEADLINE])
        for persona, per_layer in data["cosine"]["on_topic"].items()
    }


def _load_cos_to_home(cos_home_path: Path) -> dict[str, float]:
    """Per-bystander cos_to_local_historian at layer 21 (on-topic).

    Round-3 BUG-#1 fix: the producer (``scripts/issue444_persona_distance_topic.py``)
    writes ``cosine.<topic>.<persona>.<layer>`` and EXCLUDES the reference
    persona from OTHERS, so ``local_historian`` (which IS the home) has no
    self-distance entry. Reader contract:

      1. Parse the producer's actual shape: ``data["cosine"]["on_topic"][persona][LAYER_HEADLINE]``.
      2. Inject the home's self-distance: ``cos_to_home[HOME_PERSONA] = 1.0``.
         (Cosine of a persona with itself is 1 by definition; the producer
         can't write this because it skips the reference.)
      3. Also accept the legacy flat ``{persona: float}`` shape and the
         legacy nested ``{"cosine": {"21": {persona: float}}}`` shape for
         back-compat with hand-written files.

    Round-1/2 was looking for ``chd["cosine"]["21"][persona]``, which the
    producer never writes -> ``cos_to_home`` was silently EMPTY -> H4
    distance-to-home absent.
    """
    if not cos_home_path.exists():
        return {}
    chd = json.loads(cos_home_path.read_text())
    out: dict[str, float] = {}
    cosine_block = chd.get("cosine") if isinstance(chd, dict) else None
    if isinstance(cosine_block, dict):
        if "on_topic" in cosine_block:
            # Producer shape: cosine.on_topic.<persona>.<layer_str>
            for persona, per_layer in cosine_block["on_topic"].items():
                if isinstance(per_layer, dict) and LAYER_HEADLINE in per_layer:
                    out[persona] = float(per_layer[LAYER_HEADLINE])
        elif LAYER_HEADLINE in cosine_block:
            # Legacy nested shape: cosine.<layer_str>.<persona>
            inner = cosine_block[LAYER_HEADLINE]
            if isinstance(inner, dict):
                out.update({k: float(v) for k, v in inner.items() if isinstance(v, (int, float))})
    elif isinstance(chd, dict):
        # Legacy flat: {persona: float}
        out.update({k: float(v) for k, v in chd.items() if isinstance(v, (int, float))})
    # Inject the home persona's self-distance (1.0 by definition) so the
    # full 15-pool is covered.
    out.setdefault(HOME_PERSONA, 1.0)
    return out


def _load_aggregate_cleaned(arm_path: Path) -> dict[str, dict[str, float]]:
    """Per-cell -> persona -> leak_rate. Skips baseline."""
    data = json.loads(arm_path.read_text())
    out: dict[str, dict[str, float]] = {}
    for cell, info in data["per_cell"].items():
        if cell == "baseline" or "per_persona" not in info:
            continue
        out[cell] = {
            persona: float(pdata["leak_rate_headline"])
            for persona, pdata in info["per_persona"].items()
        }
    return out


def _load_5way_priors_union(
    arm_aggregate_paths: list[Path],
) -> tuple[dict[str, float], dict[str, str]]:
    """Per-persona baseline stated_seven rate, UNIONed across every arm's
    baseline cell.

    Round-1 BLOCKER #4 fix: the baseline measures the SAME untrained model
    under each persona, so the rate per (persona, fact) IS identical across
    arms; but each arm's baseline excludes its own SOURCE persona from the
    n=14 panel (Arm A excludes ``marine_biologist`` from baseline, etc.).
    The Arm B baseline alone is widened to the full 15-pool via the wrapper
    (``_widen_baseline_panel_to_full_pool``) so it has every persona. To
    cover all 15 personas across all 3 arms, union the per-persona rates from
    each arm's baseline. Conflict handling: take the first non-NaN
    encountered (baseline values from different arms for the SAME persona on
    the SAME untrained model are expected to be identical modulo
    judge-batch noise; we record the source-arm to keep the union auditable).
    """
    out: dict[str, float] = {}
    src: dict[str, str] = {}
    for ag_path in arm_aggregate_paths:
        if not ag_path.exists():
            continue
        data = json.loads(ag_path.read_text())
        baseline = data["per_cell"].get("baseline", {})
        pp = baseline.get("per_persona", {})
        arm_slug = data.get("arm_slug", ag_path.parent.name)
        for persona, pdata in pp.items():
            rate = pdata.get("a_family_stated_seven_rate")
            if rate is None:
                continue
            if persona in out:
                continue  # first-arm wins; sources tracked in `src`
            out[persona] = float(rate)
            src[persona] = arm_slug
    return out, src


def _load_engagement(arm_path: Path) -> dict[str, dict[str, float]]:
    """Optional: per-(cell, persona) {length, on_topic_fraction}.

    Returns {} if the file doesn't exist; predictors phase still completes
    and just skips the engagement-adjusted lines.
    """
    if not arm_path.exists():
        return {}
    return json.loads(arm_path.read_text())


# ---------------------------------------------------------------------------
# Per-arm aggregator
# ---------------------------------------------------------------------------
def _per_arm_metrics(
    arm_name: str,
    arm_slug: str,
    panel: tuple[str, ...],
    logprob_priors: dict[str, float],
    fiveway_priors: dict[str, float],
    cos_to_home_map: dict[str, float],
    eval_root: str = "issue_500",
) -> dict[str, object]:
    """Compute the per-arm metric table + per-arm stats.

    ``eval_root`` parametrizes the ``eval_results/<root>/`` subtree so the #541
    extension (``issue541_predictors.py``) reuses this aggregator unchanged;
    the default preserves the #500 behavior.
    """
    arm_root = REPO / "eval_results" / eval_root / arm_slug
    agg_path = arm_root / "aggregate_cleaned.json"
    cos_path = arm_root / "persona_distance" / "results.json"
    eng_path = arm_root / "engagement_covariates.json"

    if not agg_path.exists():
        raise RuntimeError(
            f"{agg_path} missing -- run scripts/aggregate_issue500.py --arm {arm_name} first."
        )
    leak_per_cell = _load_aggregate_cleaned(agg_path)
    cos_to_source = _load_cosines(cos_path) if cos_path.exists() else {}
    engagement = _load_engagement(eng_path)

    # Per-(persona, seed) cell-level points.
    points: list[dict[str, float | int | str]] = []
    for cell_tag, persona_leaks in leak_per_cell.items():
        # cell_tag like "on_policy_suppression_cn_seed42"
        seed = int(cell_tag.split("seed")[-1]) if "seed" in cell_tag else -1
        for persona, leak in persona_leaks.items():
            if persona not in panel:
                continue
            row: dict[str, float | int | str] = {
                "persona": persona,
                "seed": seed,
                "leak": float(leak),
                "prior_logprob": logprob_priors.get(persona, float("nan")),
                "prior_5way": fiveway_priors.get(persona, float("nan")),
                "cos_to_source": cos_to_source.get(persona, float("nan")),
                "cos_to_home": cos_to_home_map.get(persona, float("nan")),
            }
            eng = engagement.get(cell_tag, {}).get(persona, {})
            row["completion_length"] = float(eng.get("length", float("nan")))
            row["on_topic_fraction"] = float(eng.get("on_topic_fraction", float("nan")))
            points.append(row)

    # Cell-mean per persona (mean across the 3 seeds).
    per_persona: dict[str, dict[str, float]] = {}
    for persona in panel:
        rows = [r for r in points if r["persona"] == persona]
        if not rows:
            continue
        per_persona[persona] = {
            "leak_mean": float(np.mean([r["leak"] for r in rows])),
            "leak_seeds": [float(r["leak"]) for r in rows],
            "prior_logprob": logprob_priors.get(persona, float("nan")),
            "prior_5way": fiveway_priors.get(persona, float("nan")),
            "cos_to_source": cos_to_source.get(persona, float("nan")),
            "cos_to_home": cos_to_home_map.get(persona, float("nan")),
        }

    # Per-persona vectors (panel-aligned, drop NaNs).
    aligned_personas = [p_name for p_name in panel if p_name in per_persona]
    leak_mean = [per_persona[p]["leak_mean"] for p in aligned_personas]
    prior_lp = [per_persona[p]["prior_logprob"] for p in aligned_personas]
    prior_5w = [per_persona[p]["prior_5way"] for p in aligned_personas]
    cos_src = [per_persona[p]["cos_to_source"] for p in aligned_personas]
    cos_home_v = [per_persona[p]["cos_to_home"] for p in aligned_personas]

    def _good(seq: list[float]) -> bool:
        return all(not (isinstance(v, float) and math.isnan(v)) for v in seq)

    stats: dict[str, object] = {
        "n_personas_in_panel": len(aligned_personas),
        "n_points_with_seeds": len(points),
    }

    # Per-arm spearman correlations.
    if _good(prior_lp):
        stats["spearman_prior_logprob_vs_leak"] = _spearman(prior_lp, leak_mean)
    if _good(prior_5w):
        stats["spearman_prior_5way_vs_leak"] = _spearman(prior_5w, leak_mean)
    if _good(cos_src):
        stats["spearman_cos_to_source_vs_leak"] = _spearman(cos_src, leak_mean)
    if _good(cos_home_v):
        stats["spearman_cos_to_home_vs_leak"] = _spearman(cos_home_v, leak_mean)

    # Collinearity gate.
    if _good(cos_src) and _good(prior_lp):
        stats["pearson_abs_cos_vs_prior_logprob"] = _pearson([abs(c) for c in cos_src], prior_lp)
        stats["pearson_cos_vs_prior_logprob"] = _pearson(cos_src, prior_lp)
        # Partial spearman ρ(cos, leak | prior).
        stats["partial_spearman_cos_to_source_given_prior"] = _partial_spearman(
            cos_src, leak_mean, prior_lp
        )

    # Standardized OLS.
    if _good(cos_src) and _good(prior_lp):
        stats["ols_z_leak_on_z_prior_logprob_and_z_cos_to_source"] = _ols_two_predictor(
            leak_mean, prior_lp, cos_src
        )

    # Cluster bootstrap on the (persona, seed) point list -- clusters are
    # personas.
    if points and _good(prior_lp):
        pairs_lp = [(float(r["prior_logprob"]), float(r["leak"])) for r in points]
        clust_p = [str(r["persona"]) for r in points]  # deterministic (BUG-#6)
        stats["bootstrap_spearman_prior_logprob_vs_leak_cluster_persona"] = (
            _cluster_bootstrap_spearman(pairs_lp, clust_p)
        )
    if points and _good(cos_src):
        pairs_cs = [(float(r["cos_to_source"]), float(r["leak"])) for r in points]
        clust_p = [str(r["persona"]) for r in points]  # deterministic (BUG-#6)
        stats["bootstrap_spearman_cos_to_source_vs_leak_cluster_persona"] = (
            _cluster_bootstrap_spearman(pairs_cs, clust_p)
        )

    # Round-3 BUG-#4: cluster-bootstrap CIs for the H3 partial Spearman and
    # standardized OLS betas (plan §6.3 requires CIs on both, not just point
    # estimates). Clusters = personas; deterministic seed for reproducibility.
    if points and _good(cos_src) and _good(prior_lp):
        stats["h3_cluster_bootstrap"] = _cluster_bootstrap_h3(points)

    # Engagement-covariate partials (plan §6.3 "does the prior signal survive
    # length/engagement adjustment?"). Computed only when the
    # engagement_covariates.json file exists; otherwise emit an explicit
    # missing-status so the analyzer reports the gap rather than silently
    # skipping (round-2 BLOCKER #8 fix).
    if not engagement:
        stats["engagement_adjusted"] = {
            "status": "skipped_no_engagement_file",
            "expected_path": str(eng_path),
            "_doc": (
                "engagement_covariates.json missing for this arm. To compute "
                "the length-and-on-topic-adjusted partial correlations, "
                "produce that file with shape "
                "{cell_tag: {persona: {length: float, on_topic_fraction: float}}}."
            ),
        }
    else:
        # Build per-persona means of length + on_topic_fraction (averaged over
        # seeds, panel-aligned).
        mean_length: list[float] = []
        mean_on_topic: list[float] = []
        for persona in aligned_personas:
            persona_rows = [r for r in points if r["persona"] == persona]
            lens = [
                float(r["completion_length"])
                for r in persona_rows
                if not math.isnan(float(r["completion_length"]))
            ]
            tops = [
                float(r["on_topic_fraction"])
                for r in persona_rows
                if not math.isnan(float(r["on_topic_fraction"]))
            ]
            mean_length.append(sum(lens) / len(lens) if lens else float("nan"))
            mean_on_topic.append(sum(tops) / len(tops) if tops else float("nan"))

        eng_stats: dict[str, object] = {
            "status": "computed",
            "n_personas": len(aligned_personas),
        }
        if _good(prior_lp) and _good(mean_length):
            eng_stats["partial_spearman_prior_vs_leak_given_length"] = _partial_spearman(
                prior_lp, leak_mean, mean_length
            )
        if _good(prior_lp) and _good(mean_on_topic):
            eng_stats["partial_spearman_prior_vs_leak_given_on_topic"] = _partial_spearman(
                prior_lp, leak_mean, mean_on_topic
            )
        if _good(cos_src) and _good(mean_length):
            eng_stats["partial_spearman_cos_vs_leak_given_length"] = _partial_spearman(
                cos_src, leak_mean, mean_length
            )
        if _good(cos_src) and _good(mean_on_topic):
            eng_stats["partial_spearman_cos_vs_leak_given_on_topic"] = _partial_spearman(
                cos_src, leak_mean, mean_on_topic
            )
        # Joint engagement partials (round-3 BUG-#5): control for length AND
        # on_topic_fraction simultaneously.
        if _good(prior_lp) and _good(mean_length) and _good(mean_on_topic):
            eng_stats["partial_spearman_prior_vs_leak_given_length_and_on_topic"] = (
                _partial_spearman_multi(prior_lp, leak_mean, [mean_length, mean_on_topic])
            )
        if _good(cos_src) and _good(mean_length) and _good(mean_on_topic):
            eng_stats["partial_spearman_cos_vs_leak_given_length_and_on_topic"] = (
                _partial_spearman_multi(cos_src, leak_mean, [mean_length, mean_on_topic])
            )
        # Means recorded for transparency / sanity check.
        eng_stats["per_persona_mean_length"] = dict(zip(aligned_personas, mean_length, strict=True))
        eng_stats["per_persona_mean_on_topic"] = dict(
            zip(aligned_personas, mean_on_topic, strict=True)
        )
        stats["engagement_adjusted"] = eng_stats

    return {
        "arm": arm_name,
        "arm_slug": arm_slug,
        "panel": list(aligned_personas),
        "per_persona": per_persona,
        "per_point_n": len(points),
        "stats": stats,
        "_points": points,  # kept for cross-arm bootstrap (not analyzer-facing)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="eval_results/issue_500/predictors.json",
        help="output JSON path (analyzer reads this).",
    )
    ap.add_argument(
        "--logprob-path",
        default="eval_results/issue_500/bystander_logprob/logprob_results.json",
        help="path to the bystander_logprob output (15-persona panel).",
    )
    ap.add_argument(
        "--cos-home-path",
        default="eval_results/issue_500/distance_to_home.json",
        help=(
            "JSON with {persona: cos_to_local_historian@layer21}. Computed once "
            "(home is constant across arms). Optional -- script proceeds without "
            "this predictor if the file is missing."
        ),
    )
    args = ap.parse_args()

    # Per-arm panels.
    from run_experiment_500 import ARM_SOURCE, PANEL_15

    logprob_path = REPO / args.logprob_path
    if not logprob_path.exists():
        raise RuntimeError(
            f"{logprob_path} missing -- run scripts/issue444_bystander_logprob.py "
            "with --panel set to the 15-persona pool first."
        )
    logprob_priors = _load_logprob_priors(logprob_path)

    # 5-way priors UNIONed across all 3 arms' baseline cells (round-1
    # BLOCKER #4 fix). Each arm's baseline excludes its own source from the
    # n=14 panel; Arm B's baseline is widened to the full 15-pool by the
    # wrapper. UNIONing covers all 15 personas.
    arm_aggregate_paths = [
        REPO / "eval_results" / "issue_500" / ARM_SOURCE[a] / "aggregate_cleaned.json" for a in ARMS
    ]
    fiveway_priors, fiveway_priors_source = _load_5way_priors_union(arm_aggregate_paths)

    # Distance-to-home (cosine to local_historian@layer21). Optional file.
    # Round-3 BUG-#1 fix: parser now matches the producer's actual nested
    # shape AND injects the home persona's self-distance (cos == 1.0).
    cos_home_path = REPO / args.cos_home_path
    cos_to_home = _load_cos_to_home(cos_home_path)

    out_full: dict[str, object] = {
        "panel_pool_15": list(PANEL_15),
        "home_persona": HOME_PERSONA,
        "layer_headline": LAYER_HEADLINE,
        "per_arm": {},
        "logprob_priors_used": logprob_priors,
        "fiveway_priors_used": fiveway_priors,
        "fiveway_priors_source_arm": fiveway_priors_source,
        "cos_to_home_used": cos_to_home,
    }

    arm_results: dict[str, dict[str, object]] = {}
    for arm_name in ARMS:
        arm_slug = ARM_SOURCE[arm_name]
        panel = tuple(x for x in PANEL_15 if x != arm_name)
        try:
            arm_results[arm_name] = _per_arm_metrics(
                arm_name,
                arm_slug,
                panel,
                logprob_priors,
                fiveway_priors,
                cos_to_home,
            )
        except RuntimeError as e:
            arm_results[arm_name] = {"error": str(e), "arm_slug": arm_slug}
    out_full["per_arm"] = arm_results

    # ------------------------------------------------------------------
    # Cross-arm headline statistic Δρ(cos_to_source, leak) WITH CIs
    # (round-1 BLOCKER #6 fix).
    # Two resampling schemes reported SEPARATELY (plan §6.3):
    #   - persona-resampling: how much does Δρ depend on which bystander
    #     personas we picked from the registry?
    #   - seed-resampling: how much does Δρ depend on training-noise
    #     variance across seeds?
    # ------------------------------------------------------------------
    cross: dict[str, object] = {
        "_doc": (
            "Headline statistic: Δρ_AB = ρ_cos(Arm B) - ρ_cos(Arm A), "
            "where ρ_cos is the per-arm Spearman ρ(cos_to_source, leak). "
            "Reported alongside persona-resampling and seed-resampling "
            "bootstrap CIs (90% and 95%) -- the two schemes answer "
            "different uncertainty questions, per plan §6.3."
        )
    }
    arm_a = arm_results.get("marine_biologist", {})
    arm_b = arm_results.get("courthouse_architecture_historian", {})
    arm_c = arm_results.get("local_resident", {})
    for label, l_arm, r_arm in [
        ("delta_rho_AB", arm_a, arm_b),
        ("delta_rho_AC", arm_a, arm_c),
        ("delta_rho_CB", arm_c, arm_b),
    ]:
        l_stats = l_arm.get("stats", {}) if isinstance(l_arm, dict) else {}
        r_stats = r_arm.get("stats", {}) if isinstance(r_arm, dict) else {}
        l_rho = l_stats.get("spearman_cos_to_source_vs_leak")
        r_rho = r_stats.get("spearman_cos_to_source_vs_leak")
        l_points = l_arm.get("_points", []) if isinstance(l_arm, dict) else []
        r_points = r_arm.get("_points", []) if isinstance(r_arm, dict) else []
        entry: dict[str, object] = {}
        if l_rho is not None and r_rho is not None:
            entry["left_arm_rho_point"] = float(l_rho)
            entry["right_arm_rho_point"] = float(r_rho)
            entry["delta_point"] = float(r_rho) - float(l_rho)
        if l_points and r_points:
            entry["persona_bootstrap"] = _cross_arm_delta_rho_persona_bootstrap(
                l_points, r_points, x_field="cos_to_source"
            )
            entry["seed_bootstrap"] = _cross_arm_delta_rho_seed_bootstrap(
                l_points, r_points, x_field="cos_to_source"
            )
        else:
            entry["status"] = "skipped_missing_points"
        cross[label] = entry
    out_full["cross_arm"] = cross

    # Strip the analyzer-internal _points fields BEFORE serializing
    # (kept only for the cross-arm bootstrap above).
    for arm_name in list(arm_results):
        if isinstance(arm_results[arm_name], dict) and "_points" in arm_results[arm_name]:
            arm_results[arm_name].pop("_points", None)

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_full, indent=2, default=str))
    print(f"WROTE {out_path}")
    for arm_name, info in arm_results.items():
        if "error" in info:
            print(f"  {arm_name:35} ERROR: {info['error']}")
            continue
        s = info.get("stats", {})
        rho_p = s.get("spearman_prior_logprob_vs_leak", float("nan"))
        rho_c = s.get("spearman_cos_to_source_vs_leak", float("nan"))
        rho_h = s.get("spearman_cos_to_home_vs_leak", float("nan"))
        n = s.get("n_personas_in_panel", 0)
        print(
            f"  {arm_name:35} n={n:>2}  ρ(prior,leak)={rho_p:+.3f}  "
            f"ρ(cos,leak)={rho_c:+.3f}  ρ(home,leak)={rho_h:+.3f}"
        )


if __name__ == "__main__":
    main()
