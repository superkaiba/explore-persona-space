"""Cosine-pairwise follow-up to #380.

Computes the cosine-space analog of #380's secondary predictor (mean pairwise
output-distance to other personas) on the 24 inherited-cohort personas where
L15 centroids are available. Reads the source-rate panel unchanged from
#296/#340/#368.

Output:
    eval_results/issue_380/cosine_pairwise_n24/correlation.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

CENTROIDS_PATH = Path("eval_results/issue_274/centroids/centroids_n24_layers0_27.pt")
RATES_PATH = Path("eval_results/issue_296/length_rate_correlation_n48.json")
OUT_DIR = Path("eval_results/issue_380/cosine_pairwise_n24")
LAYER = 15
N_RESAMPLES = 1000
RNG_SEED = 42


def rank_residualize(x: np.ndarray, covar: np.ndarray) -> np.ndarray:
    rx = rankdata(x)
    rc = rankdata(covar)
    slope, intercept = np.polyfit(rc, rx, 1)
    return rx - (slope * rc + intercept)


def main() -> None:
    centroids = torch.load(CENTROIDS_PATH, weights_only=False)
    l15 = centroids[LAYER]
    rates_data = json.load(open(RATES_PATH))
    rate_by_persona = {r["source"]: r["rate_n48"] for r in rates_data["rows"]}
    tokens_by_persona = {r["source"]: r["tokens"] for r in rates_data["rows"]}
    cohort_by_persona = {r["source"]: r["cohort"] for r in rates_data["rows"]}

    personas = sorted(l15.keys())
    missing = [p for p in personas if p not in rate_by_persona]
    assert not missing, f"missing source rates for {missing}"

    mat = torch.stack([l15[p].float() for p in personas])
    mat_centered = mat - mat.mean(dim=0, keepdim=True)
    mat_norm = mat_centered / mat_centered.norm(dim=1, keepdim=True)
    cos_sim = (mat_norm @ mat_norm.T).numpy()
    cos_dist = 1.0 - cos_sim

    n = len(personas)
    mask = ~np.eye(n, dtype=bool)
    mean_pairwise = (cos_dist * mask).sum(axis=1) / (n - 1)
    median_pairwise = np.array([np.median(cos_dist[i][mask[i]]) for i in range(n)])
    max_pairwise = np.array([cos_dist[i][mask[i]].max() for i in range(n)])

    rows = []
    for i, p in enumerate(personas):
        rows.append(
            {
                "persona": p,
                "cohort": cohort_by_persona[p],
                "mean_pairwise_cosine_distance": float(mean_pairwise[i]),
                "median_pairwise_cosine_distance": float(median_pairwise[i]),
                "max_pairwise_cosine_distance": float(max_pairwise[i]),
                "source_rate": rate_by_persona[p],
                "tokens": tokens_by_persona[p],
                "log_tokens": float(np.log(tokens_by_persona[p])),
            }
        )

    predictor = np.array([r["mean_pairwise_cosine_distance"] for r in rows])
    target = np.array([r["source_rate"] for r in rows])
    log_tokens = np.array([r["log_tokens"] for r in rows])

    raw_rho, raw_p = spearmanr(predictor, target)
    resid_pred = rank_residualize(predictor, log_tokens)
    resid_target = rank_residualize(target, log_tokens)
    partial_rho, partial_p = spearmanr(resid_pred, resid_target)
    collin_rho, collin_p = spearmanr(predictor, log_tokens)

    rng = np.random.default_rng(RNG_SEED)
    boot = []
    for _ in range(N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 5:
            continue
        try:
            rp = rank_residualize(predictor[idx], log_tokens[idx])
            rt = rank_residualize(target[idx], log_tokens[idx])
        except np.linalg.LinAlgError:
            continue
        rho_b, _ = spearmanr(rp, rt)
        if np.isfinite(rho_b):
            boot.append(rho_b)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    secondary_rhos = {}
    for label, vec in [
        ("median_pairwise", median_pairwise),
        ("max_pairwise", max_pairwise),
    ]:
        r_raw, p_raw = spearmanr(vec, target)
        rp = rank_residualize(vec, log_tokens)
        r_part, p_part = spearmanr(rp, resid_target)
        secondary_rhos[label] = {
            "raw": {"rho": float(r_raw), "p": float(p_raw)},
            "length_partial": {"rho": float(r_part), "p": float(p_part)},
        }

    out = {
        "n": n,
        "cohort": "inherited_24",
        "predictor": "mean_pairwise_cosine_distance_l15_centered",
        "layer": LAYER,
        "centering": "mean_centered_across_panel",
        "raw_spearman": {"rho": float(raw_rho), "p": float(raw_p)},
        "length_partial_spearman": {
            "rho": float(partial_rho),
            "p": float(partial_p),
            "ci95": [float(ci_lo), float(ci_hi)],
            "n_resamples": len(boot),
        },
        "predictor_length_collinearity_spearman": {
            "rho": float(collin_rho),
            "p": float(collin_p),
        },
        "secondary_reductions": secondary_rhos,
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "correlation.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")
    print(f"  raw          ρ = {raw_rho:+.3f}, p = {raw_p:.3f}, N = {n}")
    print(f"  length-part. ρ = {partial_rho:+.3f}, p = {partial_p:.3f}")
    print(f"  95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}] from {len(boot)} resamples")
    print(f"  collinearity ρ = {collin_rho:+.3f}, p = {collin_p:.3f}")


if __name__ == "__main__":
    main()
