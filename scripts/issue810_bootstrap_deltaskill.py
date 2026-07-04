#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ², →) in scientific docstrings + log messages.
"""Issue #810 analyzer calibration: paired-bootstrap Δskill(maxp − mean).

Persists the plan §6 "H1 claim-size calibration" the round-1 analyzer computed
inline (previously unpersisted — interp-critique r1 required the script + draws
summary be committed to ``eval_results/issue_810/analysis/``).

Method
------
For each of {mean, maxp} × 28 layers: rebuild the EXACT reconstruction cell from
``issue810_fit_reconstruction`` (train-fold PCA-48 target, LOCO ridge
predictions via the on-main primitives), then store the PER-CONTEXT
``(ss_res_i, ss_tot_i)`` decomposition of the held-out predictions
(``ss_tot_i`` against the leave-one-out train mean). The paired bootstrap
resamples CONTEXTS with replacement over that fixed decomposition (no
per-replicate refit) — the sampling variability of the skill statistic
``1 − Σss_res/Σss_tot``, computed identically for both summaries on the SAME
resampled context set (paired).

Statistics reported (B draws, one shared RNG stream, seed fixed):

- Δskill at matched layer L18 (mean's best layer).
- Δskill at matched layer L21 (maxp's best layer — a DATA-SELECTED layer;
  labeled as such in the analysis).
- best-layer-vs-best-layer Δskill with the layer selection INHERITED per
  replicate (selection-symmetric), over all layers and over the mid/late
  window L14–22.
- fixed late-window L19–26 window-mean Δskill (per replicate: mean over the
  window of per-layer paired deltas) — the pre-stated support for the
  "maxp is more robust late" claim (fixed window, no data-driven selection).

Usage::

    uv run python scripts/issue810_bootstrap_deltaskill.py \
        --out eval_results/issue_810/analysis/bootstrap_deltaskill.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
from issue810_common import (  # noqa: E402
    HF_DATA_REPO,
    I658_STORE_MANIFEST,
    PCA_TARGET_DIM_CAP,
    context_ids_from_manifest,
    dump_json,
    reproducibility_metadata,
)
from issue810_fit_reconstruction import _load_cc, _load_free_summaries  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    loco_train_means,
    ridge_predict_loco_centered,
    robust_pca_basis,
)

logger = logging.getLogger("issue810_bootstrap_deltaskill")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUMMARIES = ("mean", "maxp")
MATCHED_LAYERS = (18, 21)
MIDLATE_WINDOW = tuple(range(14, 23))  # L14–22 (plan §6 mid/late window)
LATE_WINDOW = tuple(range(19, 27))  # L19–26 (fixed window, robustness claim)


def _per_context_decomposition(Xc: np.ndarray, Yv: np.ndarray, pca_dim: int):
    """Per-context (ss_res_i, ss_tot_i) of the held-out LOCO ridge predictions.

    Mirrors ``issue810_fit_reconstruction._fit_one_cell`` exactly (train-fold
    PCA target via robust_pca_basis, LOCO ridge via
    ridge_predict_loco_centered), then decomposes the aggregate skill
    ``1 − Σss_res/Σss_tot`` into its per-context terms.
    Returns (ss_res (n,), ss_tot (n,)).
    """
    mu, comps, _ = robust_pca_basis(Yv, pca_dim)
    Y_pca = (Yv - mu) @ comps.T  # (n, k)
    preds = ridge_predict_loco_centered(Xc, Y_pca)  # (n, k) held-out
    tmean = loco_train_means(Y_pca)  # (n, k) LOO train means
    ss_res = np.sum((Y_pca - preds) ** 2, axis=1)
    ss_tot = np.sum((Y_pca - tmean) ** 2, axis=1)
    return ss_res, ss_tot


def _skill(ss_res: np.ndarray, ss_tot: np.ndarray, idx: np.ndarray | None = None) -> float:
    """Aggregate skill 1 − Σss_res/Σss_tot over (optionally resampled) contexts."""
    if idx is not None:
        ss_res, ss_tot = ss_res[idx], ss_tot[idx]
    tot = float(ss_tot.sum())
    return float("nan") if tot < 1e-12 else 1.0 - float(ss_res.sum()) / tot


def _stat_summary(obs: float, draws: np.ndarray) -> dict:
    """Observed value + bootstrap CI + P(Δ≤0) + draw percentiles."""
    return {
        "observed": obs,
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "p_delta_le_0": float(np.mean(draws <= 0.0)),
        "draw_percentiles": {
            str(p): float(np.percentile(draws, p)) for p in (2.5, 25, 50, 75, 97.5)
        },
        "n_draws": int(draws.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 paired-bootstrap Δskill(maxp − mean)")
    ap.add_argument(
        "--out",
        default=str(
            PROJECT_ROOT / "eval_results" / "issue_810" / "analysis" / "bootstrap_deltaskill.json"
        ),
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    man_path = hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset")
    import json

    with open(man_path) as f:
        ctx_ids = context_ids_from_manifest(json.load(f))
    free_summaries, capture_layers = _load_free_summaries()
    cc = _load_cc(ctx_ids, capture_layers)
    n = len(ctx_ids)
    n_layers = len(capture_layers)
    pca_dim = min(PCA_TARGET_DIM_CAP, n - 2)
    logger.info("n_contexts=%d n_layers=%d pca_dim=%d", n, n_layers, pca_dim)

    # Per-cell per-context decompositions: {summary: (n_layers, n) arrays}.
    ss_res = {s: np.zeros((n_layers, n)) for s in SUMMARIES}
    ss_tot = {s: np.zeros((n_layers, n)) for s in SUMMARIES}
    obs_skill = {s: np.zeros(n_layers) for s in SUMMARIES}
    for s in SUMMARIES:
        for li in range(n_layers):
            Xc = np.stack([cc[c][li] for c in ctx_ids])
            Yv = np.stack([free_summaries[s][c][li].numpy() for c in ctx_ids])
            r, t = _per_context_decomposition(Xc, Yv, pca_dim)
            ss_res[s][li], ss_tot[s][li] = r, t
            obs_skill[s][li] = _skill(r, t)
        logger.info(
            "[%s] best skill %.4f @L%d",
            s,
            float(obs_skill[s].max()),
            int(obs_skill[s].argmax()),
        )

    rng = np.random.default_rng(args.seed)
    B = args.n_boot
    draws = {
        "matched_L18": np.zeros(B),
        "matched_L21": np.zeros(B),
        "best_vs_best_all": np.zeros(B),
        "best_vs_best_midlate_L14_22": np.zeros(B),
        "window_mean_L19_26": np.zeros(B),
    }
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sk = {
            s: np.array([_skill(ss_res[s][li], ss_tot[s][li], idx) for li in range(n_layers)])
            for s in SUMMARIES
        }
        d = sk["maxp"] - sk["mean"]
        draws["matched_L18"][b] = d[18]
        draws["matched_L21"][b] = d[21]
        draws["best_vs_best_all"][b] = sk["maxp"].max() - sk["mean"].max()
        ml = list(MIDLATE_WINDOW)
        draws["best_vs_best_midlate_L14_22"][b] = sk["maxp"][ml].max() - sk["mean"][ml].max()
        draws["window_mean_L19_26"][b] = float(np.mean(d[list(LATE_WINDOW)]))

    d_obs = obs_skill["maxp"] - obs_skill["mean"]
    ml = list(MIDLATE_WINDOW)
    observed = {
        "matched_L18": float(d_obs[18]),
        "matched_L21": float(d_obs[21]),
        "best_vs_best_all": float(obs_skill["maxp"].max() - obs_skill["mean"].max()),
        "best_vs_best_midlate_L14_22": float(
            obs_skill["maxp"][ml].max() - obs_skill["mean"][ml].max()
        ),
        "window_mean_L19_26": float(np.mean(d_obs[list(LATE_WINDOW)])),
    }

    out = {
        "dv": "paired_bootstrap_delta_skill_maxp_minus_mean",
        "method": (
            "contexts resampled with replacement over the fixed per-context "
            "(ss_res, ss_tot) decomposition of the held-out LOCO ridge predictions "
            "(train-fold PCA-48 target; no per-replicate refit); layer selection for "
            "best-vs-best statistics inherited per replicate; L19-26 window mean is a "
            "fixed (non-selected) window"
        ),
        "n_contexts": n,
        "pca_dim": pca_dim,
        "n_boot": B,
        "seed": args.seed,
        "note_L21_is_data_selected": (
            "L21 is maxp's own best layer (data-selected); the matched_L21 CI is "
            "conditional on that selection and is NOT multiplicity-corrected"
        ),
        "per_layer_observed_skill": {s: [float(v) for v in obs_skill[s]] for s in SUMMARIES},
        "per_context_decomposition": {
            s: {
                "ss_res": ss_res[s].tolist(),
                "ss_tot": ss_tot[s].tolist(),
                "context_ids": ctx_ids,
            }
            for s in SUMMARIES
        },
        "statistics": {k: _stat_summary(observed[k], draws[k]) for k in draws},
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(out, args.out)
    for k, v in out["statistics"].items():
        logger.info(
            "%s: obs %+0.4f CI95 [%+0.4f, %+0.4f] P(<=0)=%.4f",
            k,
            v["observed"],
            v["ci95"][0],
            v["ci95"][1],
            v["p_delta_le_0"],
        )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
