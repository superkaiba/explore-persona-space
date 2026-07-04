#!/usr/bin/env python3
"""Issue #722 — 0-GPU analysis of the teacher-forced fixed +/- margin DV.

Two outputs:

(1) VALIDATION (the headline): per behavior, Spearman( margin(C,B), rate(C,B) )
    across the 50 contexts, where rate = E0_expression.json e0[ctx][B]["rate"].
    Does the teacher-forced fixed-pair margin TRACK the behavior rate (unlike
    #742's logp_pos_mean, which failed at rho ~ -0.3)? If positive and
    |rho| >~ 0.3 the margin is a usable non-saturating companion.

(2) THE CHAIN: best-layer LOCO Spearman predicting margin from v_A (DIRECT) vs
    from M.v_C (MEDIATED, c_C -> v_A ridge map), with 95% CIs (2000-boot,
    family-clustered). Mirrors the result4 ridge-readout: a per-fold RidgeCV
    readout fit ONCE on (PCA48(true v_A) -> margin), applied to both true v_A
    (direct) and the c_C-predicted v_A (mediated). Does the c_C->v_A
    approximation PRESERVE the readout?

Reuses `ridge_predict_loco_centered` + `robust_pca_basis` (vectorized_mlp_skill)
exactly as the result4 readout machinery does. 0 GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy/torch freeze their pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
    robust_pca_basis,
)

BEHAVIORS = ["broad_em", "refusal", "sycophancy"]
PCA_TARGET_DIM = 48
RIDGE_ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
N_LAYERS = 28
N_BOOT = 2000

V0_FILE = PROJECT_ROOT / "data/issue_658/store/v0_summaries.pt"
CC_FILE = (
    PROJECT_ROOT
    / "data/issue_658/prb_dl/issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
)
E0_FILE = PROJECT_ROOT / "eval_results/issue_658/E0_expression.json"
MARGIN_FILE = PROJECT_ROOT / "eval_results/issue_722/tf_margin/margins.json"
OUT_FILE = PROJECT_ROOT / "eval_results/issue_722/tf_margin/margin_chain.json"


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    r, _ = spearmanr(a, b)
    return None if np.isnan(r) else float(r)


def clustered_bootstrap_spearman(x, y, families, n_boot=N_BOOT, alpha=0.05, seed=0):
    """Family-clustered percentile CI on Spearman(x,y) (resample whole families)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    fams = np.asarray(families, dtype=object)
    point = _spearman(x, y)
    uniq = sorted({str(f) for f in fams})
    if point is None or len(uniq) < 2:
        return {"point": point, "ci_lo": point, "ci_hi": point, "n_families": len(uniq)}
    fam_to_idx = {f: np.where(fams.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([fam_to_idx[f] for f in chosen])
        r = _spearman(x[idx], y[idx])
        if r is not None:
            vals.append(r)
    vals = np.array(vals)
    return {
        "point": float(point),
        "ci_lo": float(np.percentile(vals, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "n_families": len(uniq),
        "n_boot_kept": int(vals.size),
    }


def ridge_readout_loco_direct(Z, y):
    """LOCO held-out RidgeCV predictions: fit (Z_train->y_train), predict Z_held."""
    n = Z.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Ztr, ytr = Z[tr], y[tr]
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
        m = RidgeCV(alphas=RIDGE_ALPHAS).fit((Ztr - mu) / sd, ytr)
        preds[i] = m.predict(((Z[i] - mu) / sd)[None, :])[0]
    return preds


def ridge_readout_loco_mediated(Z_true, Z_pred, y):
    """SAME readout fit on TRUE v_A; applied to the c_C-PREDICTED v_A held-out row."""
    n = Z_true.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Ztr, ytr = Z_true[tr], y[tr]
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
        m = RidgeCV(alphas=RIDGE_ALPHAS).fit((Ztr - mu) / sd, ytr)
        preds[i] = m.predict(((Z_pred[i] - mu) / sd)[None, :])[0]
    return preds


def load_substrate():
    v0 = torch.load(V0_FILE, weights_only=False)
    ctx_ids = list(v0["context_ids"])
    layers = list(v0["capture_layers"])
    assert layers == list(range(N_LAYERS)), layers
    V = np.stack([v0["summaries"]["mean"][c].numpy() for c in ctx_ids]).astype(np.float64)

    cc = torch.load(CC_FILE, weights_only=False)
    cc_iids = list(cc["instance_ids"])
    cc_tensor = cc["tensor"].numpy().astype(np.float64)  # (50, 28, H)
    iid_to_row = {iid: i for i, iid in enumerate(cc_iids)}
    families = list(cc["families"])
    fam_by_ctx = {iid: families[iid_to_row[iid]] for iid in cc_iids}
    # Order c_C to match v0's ctx_ids; carry families in the same order.
    missing = [c for c in ctx_ids if c not in iid_to_row]
    assert not missing, f"c_C missing contexts: {missing}"
    C = np.stack([cc_tensor[iid_to_row[c]] for c in ctx_ids]).astype(np.float64)
    fams = [fam_by_ctx[c] for c in ctx_ids]

    e0 = json.loads(E0_FILE.read_text())
    margins = json.loads(MARGIN_FILE.read_text())
    return {
        "ctx_ids": ctx_ids,
        "layers": layers,
        "V": V,
        "C": C,
        "fams": fams,
        "e0": e0,
        "margins": margins,
    }


def margin_vector(margins, behavior, ctx_ids):
    return np.array([margins["margins"][c][behavior]["margin"] for c in ctx_ids], dtype=np.float64)


def rate_vector(e0, behavior, ctx_ids):
    out = []
    for c in ctx_ids:
        cell = e0.get("e0", {}).get(c, {}).get(behavior, {})
        out.append(float(cell.get("rate")) if cell.get("rate") is not None else np.nan)
    return np.array(out, dtype=np.float64)


def main():
    sub = load_substrate()
    V, C, layers, fams = sub["V"], sub["C"], sub["layers"], sub["fams"]
    ctx_ids = sub["ctx_ids"]
    print(f"[info] V {V.shape}  C {C.shape}  n_ctx={len(ctx_ids)}  n_fam={len(set(fams))}")

    # Per-layer cache: PCA48 true-v_A coords + c_C-predicted PCA48 coords.
    cache = {}
    for li in layers:
        Y = V[:, li, :]
        Xc = C[:, li, :]
        mu, comps, _ = robust_pca_basis(Y, PCA_TARGET_DIM)
        Y_pca = (Y - mu) @ comps.T
        pred_pca = ridge_predict_loco_centered(Xc, Y_pca)
        cache[li] = (Y_pca, pred_pca)

    results = {}
    for b in BEHAVIORS:
        y_margin = margin_vector(sub["margins"], b, ctx_ids)
        y_rate = rate_vector(sub["e0"], b, ctx_ids)
        ok = np.isfinite(y_margin) & np.isfinite(y_rate)
        n = int(ok.sum())

        # (1) VALIDATION: margin vs rate.
        val = clustered_bootstrap_spearman(
            y_margin[ok], y_rate[ok], [f for f, k in zip(fams, ok, strict=True) if k]
        )

        # (2) CHAIN: best-layer LOCO direct (v_A) vs mediated (M.v_C).
        per_layer = []
        best_direct = best_mediated = None
        for li in layers:
            Y_pca, pred_pca = cache[li]
            predA = ridge_readout_loco_direct(Y_pca[ok], y_margin[ok])
            predB = ridge_readout_loco_mediated(Y_pca[ok], pred_pca[ok], y_margin[ok])
            rhoA = _spearman(predA, y_margin[ok])
            rhoB = _spearman(predB, y_margin[ok])
            per_layer.append({"layer": li, "rho_direct": rhoA, "rho_mediated": rhoB})
            if rhoA is not None and (best_direct is None or rhoA > best_direct["rho"]):
                best_direct = {"layer": li, "rho": rhoA}
            if rhoB is not None and (best_mediated is None or rhoB > best_mediated["rho"]):
                best_mediated = {"layer": li, "rho": rhoB}

        # CIs at the best layers (re-fit + bootstrap on held-out predictions).
        fams_ok = [f for f, k in zip(fams, ok, strict=True) if k]
        Ld = best_direct["layer"]
        Lm = best_mediated["layer"]
        predA_best = ridge_readout_loco_direct(cache[Ld][0][ok], y_margin[ok])
        predB_best = ridge_readout_loco_mediated(cache[Lm][0][ok], cache[Lm][1][ok], y_margin[ok])
        ci_direct = clustered_bootstrap_spearman(predA_best, y_margin[ok], fams_ok)
        ci_mediated = clustered_bootstrap_spearman(predB_best, y_margin[ok], fams_ok)

        results[b] = {
            "n": n,
            "validation_margin_vs_rate": val,
            "margin_range": [float(y_margin[ok].min()), float(y_margin[ok].max())],
            "margin_std": float(y_margin[ok].std()),
            "rate_range": [float(y_rate[ok].min()), float(y_rate[ok].max())],
            "best_direct": {"layer": Ld, **ci_direct},
            "best_mediated": {"layer": Lm, **ci_mediated},
            "preservation_drop": (best_direct["rho"] - best_mediated["rho"]),
            "per_layer": per_layer,
        }
        print(
            f"  {b:11s} n={n}  VALID rho(margin,rate)={val['point']:+.3f} "
            f"[{val['ci_lo']:+.3f},{val['ci_hi']:+.3f}]  "
            f"DIRECT={ci_direct['point']:+.3f}@L{Ld} "
            f"MEDIATED={ci_mediated['point']:+.3f}@L{Lm}  "
            f"margin_std={results[b]['margin_std']:.4f}"
        )

    out = {
        "analysis": "issue722_tf_margin_validation_and_chain",
        "description": (
            "Teacher-forced FIXED +/- completion-margin DV. (1) validation: "
            "Spearman(margin,rate) per behavior across 50 contexts (clustered "
            "boot CI). (2) chain: best-layer LOCO Spearman(predicted margin, "
            "margin), DIRECT from v_A (PCA48) vs MEDIATED via c_C->v_A ridge map "
            "(same fixed readout). 95% family-clustered bootstrap, n_boot=2000."
        ),
        "n_boot": N_BOOT,
        "pca_target_dim": PCA_TARGET_DIM,
        "ridge_alphas": RIDGE_ALPHAS,
        "behaviors": BEHAVIORS,
        "cap_per_side": sub["margins"].get("cap_per_side"),
        "pool_meta": sub["margins"].get("pool_meta"),
        "excluded_no_pool": sub["margins"].get("excluded_no_pool"),
        "per_behavior": results,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[info] wrote {OUT_FILE}")
    return out


if __name__ == "__main__":
    main()
