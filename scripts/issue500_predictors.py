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
    """Spearman ρ between x and y after partialling out z (rank-based)."""
    if len(x) < 3 or len(set(map(len, (x, y, z)))) != 1:
        return float("nan")
    rx = np.asarray(_rankdata(x))
    ry = np.asarray(_rankdata(y))
    rz = np.asarray(_rankdata(z))
    # Residuals of x and y after OLS regression on z (using ranks).
    A = np.column_stack([np.ones_like(rz), rz])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    res_x = rx - A @ bx
    res_y = ry - A @ by
    return _spearman(list(res_x), list(res_y))


def _standardize(x: list[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    sd = a.std(ddof=1) or 1.0
    return (a - a.mean()) / sd


def _ols_two_predictor(y: list[float], x1: list[float], x2: list[float]) -> dict[str, float]:
    """Standardized OLS z(y) ~ z(x1) + z(x2). Returns betas + R^2."""
    zy = _standardize(y)
    zx1 = _standardize(x1)
    zx2 = _standardize(x2)
    A = np.column_stack([np.ones_like(zx1), zx1, zx2])
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
    cluster_ids: list[int],
    *,
    n_iter: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster-bootstrap CI for Spearman ρ over pairs[(x, y)] grouped by cluster.

    Resamples CLUSTERS with replacement, recomputes ρ on the assembled pairs.
    Returns mean, 5%/95%/2.5%/97.5% percentile bounds.
    """
    rng = np.random.default_rng(seed)
    clusters = sorted(set(cluster_ids))
    by_cluster: dict[int, list[tuple[float, float]]] = {c: [] for c in clusters}
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


def _load_5way_priors(arm_a_aggregate_path: Path) -> dict[str, float]:
    """Per-persona baseline stated_seven rate from Arm A's baseline cell.

    (The baseline is the same untrained model under each persona; we read
    it from Arm A's tree because that's where the full 15-persona panel
    baseline lives. The 5-way prior is the SAME number in all 3 arms; this
    is the on-policy variant of the bystander prior, plan §4.5.)
    """
    data = json.loads(arm_a_aggregate_path.read_text())
    baseline = data["per_cell"].get("baseline", {})
    pp = baseline.get("per_persona", {})
    out: dict[str, float] = {}
    for persona, pdata in pp.items():
        rate = pdata.get("a_family_stated_seven_rate")
        if rate is not None:
            out[persona] = float(rate)
    return out


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
    fivewat_priors: dict[str, float],
    cos_to_home_map: dict[str, float],
) -> dict[str, object]:
    """Compute the per-arm metric table + per-arm stats."""
    arm_root = REPO / "eval_results" / "issue_500" / arm_slug
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
                "prior_5way": fivewat_priors.get(persona, float("nan")),
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
            "prior_5way": fivewat_priors.get(persona, float("nan")),
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
        clust_p = [hash(str(r["persona"])) & 0xFFFFFFFF for r in points]
        stats["bootstrap_spearman_prior_logprob_vs_leak_cluster_persona"] = (
            _cluster_bootstrap_spearman(pairs_lp, clust_p)
        )
    if points and _good(cos_src):
        pairs_cs = [(float(r["cos_to_source"]), float(r["leak"])) for r in points]
        clust_p = [hash(str(r["persona"])) & 0xFFFFFFFF for r in points]
        stats["bootstrap_spearman_cos_to_source_vs_leak_cluster_persona"] = (
            _cluster_bootstrap_spearman(pairs_cs, clust_p)
        )

    return {
        "arm": arm_name,
        "arm_slug": arm_slug,
        "panel": list(aligned_personas),
        "per_persona": per_persona,
        "per_point_n": len(points),
        "stats": stats,
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

    # 5-way priors come from Arm A's baseline cell (identical across arms).
    arm_a_agg = (
        REPO
        / "eval_results"
        / "issue_500"
        / ARM_SOURCE["marine_biologist"]
        / "aggregate_cleaned.json"
    )
    fivewat_priors = _load_5way_priors(arm_a_agg) if arm_a_agg.exists() else {}

    # Distance-to-home (cosine to local_historian@layer21). Optional.
    cos_to_home: dict[str, float] = {}
    cos_home_path = REPO / args.cos_home_path
    if cos_home_path.exists():
        chd = json.loads(cos_home_path.read_text())
        # Accept either {"cosine": {"21": {persona: float}}} or {persona: float}.
        if "cosine" in chd and "21" in chd.get("cosine", {}):
            cos_to_home = {k: float(v) for k, v in chd["cosine"]["21"].items()}
        elif isinstance(chd, dict):
            cos_to_home = {k: float(v) for k, v in chd.items() if isinstance(v, (int, float))}

    out_full: dict[str, object] = {
        "panel_pool_15": list(PANEL_15),
        "home_persona": HOME_PERSONA,
        "layer_headline": LAYER_HEADLINE,
        "per_arm": {},
        "logprob_priors_used": logprob_priors,
        "fiveway_priors_used": fivewat_priors,
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
                fivewat_priors,
                cos_to_home,
            )
        except RuntimeError as e:
            arm_results[arm_name] = {"error": str(e), "arm_slug": arm_slug}
    out_full["per_arm"] = arm_results

    # Cross-arm headline: Δρ(cos_to_source, leak).
    cross: dict[str, object] = {}
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
        if l_rho is not None and r_rho is not None:
            cross[label] = {
                "left_arm_rho": l_rho,
                "right_arm_rho": r_rho,
                "delta": float(r_rho) - float(l_rho),
            }
    out_full["cross_arm"] = cross

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
