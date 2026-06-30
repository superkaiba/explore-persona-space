#!/usr/bin/env python3
"""#722 ridge-readout behavior-chain, UltraChat (g1) genre mirror.

Same method as /tmp/ridge_chain_analysis.py but on the UltraChat genre substrate:
the genre v0 store self-contains cc_last + cc_meanprompt and its own r_b.pt;
E0 = E0_expression_g1.json (HF). c_C = last-input-token (the genre store's cc_last).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
    robust_pca_basis,
)

BEHAVIORS = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]
PCA_TARGET_DIM = 48
RIDGE_ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
N_LAYERS = 28
G1_DIR = (
    PROJECT_ROOT
    / "data/issue_658/g1_dl/issue658_theory_assumptions/store_genre-generalization-ultrachat"
)
HF_REPO = "superkaiba1/explore-persona-space-data"
E0_G1_HF = "issue658_partial/att-20260624-130414/eval_results_issue_658/E0_expression_g1.json"


def _spearman(a, b):
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    r, _ = spearmanr(a, b)
    return None if np.isnan(r) else float(r)


def e0_vector(e0, col, ctx_ids):
    vals, kept = [], []
    for i, c in enumerate(ctx_ids):
        cell = e0.get("e0", {}).get(c, {}).get(col)
        if cell is None:
            continue
        v = cell.get("rate", cell.get("logp_mean"))
        if v is None:
            continue
        vals.append(float(v))
        kept.append(i)
    return np.array(vals), kept


def ridge_readout_direct(Z, y):
    n = Z.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        m = RidgeCV(alphas=RIDGE_ALPHAS).fit((Z[tr] - mu) / sd, y[tr])
        preds[i] = m.predict(((Z[i] - mu) / sd)[None])[0]
    return preds


def ridge_readout_mediated(Zt, Zp, y):
    n = Zt.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        mu, sd = Zt[tr].mean(0), Zt[tr].std(0) + 1e-9
        m = RidgeCV(alphas=RIDGE_ALPHAS).fit((Zt[tr] - mu) / sd, y[tr])
        preds[i] = m.predict(((Zp[i] - mu) / sd)[None])[0]
    return preds


def main():
    v0 = torch.load(G1_DIR / "v0_summaries.pt", weights_only=False)
    rb = torch.load(G1_DIR / "r_b.pt", weights_only=False)
    ctx_ids = list(v0["context_ids"])
    V = np.stack([v0["summaries"]["mean"][c].numpy() for c in ctx_ids]).astype(np.float64)
    C = np.stack([v0["cc_last"][c].numpy() for c in ctx_ids]).astype(np.float64)  # last-input-token
    r_b = {
        b: np.stack([rb["r_b"][b]["diffmeans"][li].numpy() for li in range(N_LAYERS)]).astype(
            np.float64
        )
        for b in BEHAVIORS
        if b in rb.get("r_b", {})
    }
    from huggingface_hub import hf_hub_download

    e0 = json.loads(Path(hf_hub_download(HF_REPO, E0_G1_HF, repo_type="dataset")).read_text())

    cache = {}
    for li in range(N_LAYERS):
        Y, Xc = V[:, li, :], C[:, li, :]
        mu, comps, _ = robust_pca_basis(Y, PCA_TARGET_DIM)
        Y_pca = (Y - mu) @ comps.T
        cache[li] = (Y_pca, ridge_predict_loco_centered(Xc, Y_pca), Xc)

    per_behavior = {}
    for b in BEHAVIORS:
        y, kept = e0_vector(e0, b, ctx_ids)
        if len(kept) < 4:
            per_behavior[b] = {"status": f"too few E0 ({len(kept)})"}
            continue
        rows, bA, bB, bC = [], None, None, None
        for li in range(N_LAYERS):
            Yp, Pp, Xc = cache[li]
            Yp, Pp, Xc = Yp[kept], Pp[kept], Xc[kept]
            rA = _spearman(ridge_readout_direct(Yp, y), y)
            rB = _spearman(ridge_readout_mediated(Yp, Pp, y), y)
            rC = _spearman(ridge_readout_direct(Xc, y), y)
            rows.append(
                {
                    "layer": li,
                    "rho_ridge_direct": rA,
                    "rho_ridge_mediated": rB,
                    "rho_cc_direct": rC,
                    "rho_degradation": (rA - rB) if (rA is not None and rB is not None) else None,
                }
            )
            if rA is not None and (bA is None or rA > bA["rho"]):
                bA = {"layer": li, "rho": rA}
            if rB is not None and (bB is None or rB > bB["rho"]):
                bB = {"layer": li, "rho": rB}
            if rC is not None and (bC is None or rC > bC["rho"]):
                bC = {"layer": li, "rho": rC}
        per_behavior[b] = {
            "n_kept_e0": len(kept),
            "best_ridge_direct": bA,
            "best_ridge_mediated": bB,
            "best_cc_direct": bC,
            "best_layer_degradation": (bA["rho"] - bB["rho"]) if (bA and bB) else None,
            "per_layer": rows,
        }
        print(
            f"  [g1] {b:22s} A={bA['rho']:.3f}@L{bA['layer']} "
            f"B={bB['rho']:.3f}@L{bB['layer']} C={bC['rho']:.3f}@L{bC['layer']} "
            f"deg={bA['rho'] - bB['rho']:+.3f}"
        )

    out = {
        "analysis": "behavioral_chain_preservation_RIDGE_READOUT_g1_ultrachat",
        "cc_variant": "last-input-token (genre store cc_last)",
        "e0_src": f"{HF_REPO}:{E0_G1_HF}",
        "pca_target_dim": PCA_TARGET_DIM,
        "per_behavior": per_behavior,
    }
    p = PROJECT_ROOT / "eval_results/issue_722/structural/behavior_chain_ridge_readout_g1.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[info] wrote {p}")


if __name__ == "__main__":
    main()
