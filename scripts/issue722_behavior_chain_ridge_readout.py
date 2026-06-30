#!/usr/bin/env python3
"""#722 behavior-chain analysis, RIDGE-READOUT version.

Mirrors `issue722_structural_battery.py::part5_behavior_chain` but swaps the
v0->E0 readout from the r_B diff-of-means dot product to a RIDGE regression
(the strong decoder, per #742). For each behavior B and layer L computes
held-out (LOCO, n=50) Spearman rho(predicted E0, actual E0) for:

  (A) DIRECT  ridge: RidgeCV fit on (PCA48(true v0) -> E0), scored on true v0
  (B) MEDIATED ridge: predict v0 from c_C via LOCO ridge map M_hat (in PCA48
      space), then apply the SAME readout (fit on true-v0-train -> E0-train)
      to the PREDICTED v0.  Fixed readout, true-v0 vs predicted-v0 — mirrors
      the r_B chain.
  (C) DIRECT-from-c_C ridge: RidgeCV fit directly on (c_C -> E0), no v0.

FAIR-DESIGN choice (per task brief): the DIRECT and MEDIATED readouts are both
fit in the SAME PCA-48 v0 space. Direct readout reads true PCA48(v0); mediated
readout reads predicted PCA48(v0) (predicted via the LOCO ridge map c_C->PCA48(v0)).
The readout itself (PCA48(v0)-train -> E0-train) is fit ONCE per LOCO fold and
applied to BOTH true and predicted held-out PCA48(v0). All standardization /
readout fit / M_hat fit are per-fold (no leakage). Variant C is fit in raw c_C
space (z-scored).
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
    robust_pca_basis,
)

BEHAVIORS = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]
PCA_TARGET_DIM = 48
RIDGE_ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
N_LAYERS = 28

V0_FILE = PROJECT_ROOT / "data/issue_658/store/v0_summaries.pt"
RB_FILE = PROJECT_ROOT / "data/issue_658/store/r_b.pt"
E0_LOCAL = PROJECT_ROOT / "eval_results/issue_658/E0_expression.json"
E0_HF = "issue658_partial/att-20260624-130414/eval_results_issue_658/E0_expression.json"
HF_REPO = "superkaiba1/explore-persona-space-data"


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    r, _ = spearmanr(a, b)
    return None if np.isnan(r) else float(r)


def e0_vector(e0: dict, column_id: str, ctx_ids: list[str]) -> tuple[np.ndarray, list[int]]:
    vals: list[float] = []
    kept_idx: list[int] = []
    for i, c in enumerate(ctx_ids):
        cell = e0.get("e0", {}).get(c, {}).get(column_id)
        if cell is None:
            continue
        v = cell.get("rate")
        if v is None:
            v = cell.get("logp_mean")
        if v is None:
            continue
        vals.append(float(v))
        kept_idx.append(i)
    return np.array(vals, dtype=np.float64), kept_idx


def load_substrate():
    v0 = torch.load(V0_FILE, weights_only=False)
    ctx_ids = list(v0["context_ids"])
    layers = list(v0["capture_layers"])
    assert layers == list(range(N_LAYERS)), layers

    V = np.stack([v0["summaries"]["mean"][c].numpy() for c in ctx_ids]).astype(
        np.float64
    )  # (50,28,3584)

    # c_C — last-input-token canonical, fall back to mean-prompt variant.
    cc_variant = "last-input-token"
    try:
        from issue658_common import load_cc_last_store

        cc_d = load_cc_last_store(capture_layers=list(range(N_LAYERS)), ctx_ids=ctx_ids)
        C = np.stack([cc_d[c].numpy() for c in ctx_ids]).astype(np.float64)
    except Exception as exc:
        print(
            f"[WARN] load_cc_last_store failed ({type(exc).__name__}: {exc}); "
            "falling back to cc_meanprompt (mean-prompt variant).",
            file=sys.stderr,
        )
        cc_variant = "mean-prompt variant (last-token unavailable)"
        C = np.stack([v0["cc_meanprompt"][c].numpy() for c in ctx_ids]).astype(np.float64)

    rb = torch.load(RB_FILE, weights_only=False)
    r_b = {
        b: np.stack([rb["r_b"][b]["diffmeans"][li].numpy() for li in range(N_LAYERS)]).astype(
            np.float64
        )
        for b in BEHAVIORS
        if b in rb.get("r_b", {})
    }

    if E0_LOCAL.exists():
        e0 = json.loads(E0_LOCAL.read_text())
        e0_src = f"local:{E0_LOCAL}"
    else:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(HF_REPO, E0_HF, repo_type="dataset")
        e0 = json.loads(Path(p).read_text())
        e0_src = f"{HF_REPO}:{E0_HF}"

    return {
        "ctx_ids": ctx_ids,
        "layers": layers,
        "V": V,
        "C": C,
        "cc_variant": cc_variant,
        "r_b": r_b,
        "e0": e0,
        "e0_src": e0_src,
    }


# ── readout: scalar RidgeCV on PCA-48 v0 space (per-fold, no leakage) ──────────
def ridge_readout_loco_direct(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LOCO held-out predictions: RidgeCV fit on (Z_train -> y_train), predict Z_held.

    Z = PCA48 v0 coords (n, k). Per-fold z-score Z on train, fit RidgeCV(alphas),
    predict the held-out row. Returns (n,) held-out predictions.
    """
    n = Z.shape[0]
    preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Ztr, ytr = Z[tr], y[tr]
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
        Ztr_n = (Ztr - mu) / sd
        model = RidgeCV(alphas=RIDGE_ALPHAS)
        model.fit(Ztr_n, ytr)
        preds[i] = model.predict(((Z[i] - mu) / sd)[None, :])[0]
    return preds


def ridge_readout_loco_mediated(
    Z_true: np.ndarray, Z_pred: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """SAME readout as direct, but the held-out row uses the PREDICTED Z.

    Per fold: fit RidgeCV on (Z_true_train -> y_train) [the readout never sees
    predicted v0 in training], z-scored on the TRUE-train stats; apply to the
    held-out PREDICTED Z row (z-scored with the same true-train stats — the
    fixed readout). Returns (n,) held-out predictions.
    """
    n = Z_true.shape[0]
    preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Ztr, ytr = Z_true[tr], y[tr]
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
        Ztr_n = (Ztr - mu) / sd
        model = RidgeCV(alphas=RIDGE_ALPHAS)
        model.fit(Ztr_n, ytr)
        preds[i] = model.predict(((Z_pred[i] - mu) / sd)[None, :])[0]
    return preds


def ridge_readout_loco_raw(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Variant C: RidgeCV fit directly on raw c_C (z-scored) -> E0, LOCO."""
    return ridge_readout_loco_direct(X, y)  # same machinery; X already raw c_C


def main():
    sub = load_substrate()
    V, C, layers = sub["V"], sub["C"], sub["layers"]
    print(f"[info] c_C variant: {sub['cc_variant']}")
    print(f"[info] E0 source:   {sub['e0_src']}")
    print(f"[info] V shape {V.shape}  C shape {C.shape}")

    # Precompute per-layer: PCA48 basis on true v0, true PCA48 coords, predicted
    # PCA48 coords (LOCO ridge c_C -> PCA48(v0)).
    perlayer_cache = {}
    for li in layers:
        Y = V[:, li, :]
        Xc = C[:, li, :]
        mu, comps, _fb = robust_pca_basis(Y, PCA_TARGET_DIM)  # comps (k, H)
        Y_pca = (Y - mu) @ comps.T  # (N, k) true PCA48 coords
        pred_pca = ridge_predict_loco_centered(Xc, Y_pca)  # (N, k) predicted PCA48 coords
        perlayer_cache[li] = (Y_pca, pred_pca, Xc)

    per_behavior = {}
    for b in BEHAVIORS:
        if b not in sub["r_b"]:
            per_behavior[b] = {"status": "no r_B"}
            continue
        y_e0, kept_idx = e0_vector(sub["e0"], b, sub["ctx_ids"])
        if len(kept_idx) < 4:
            per_behavior[b] = {"status": f"too few E0 ({len(kept_idx)})"}
            continue

        rows = []
        best_A = best_B = best_C = None
        for li in layers:
            Y_pca_all, pred_pca_all, Xc_all = perlayer_cache[li]
            # Subselect to rows that have an E0 value (LOCO runs over kept rows
            # only; for these 4 behaviors kept_idx == range(50), so this is the
            # full set — guarded for generality if a context ever lacked E0).
            Y_pca = Y_pca_all[kept_idx]
            pred_pca = pred_pca_all[kept_idx]
            Xc = Xc_all[kept_idx]
            # (A) direct ridge readout on true PCA48(v0)
            predA = ridge_readout_loco_direct(Y_pca, y_e0)
            # (B) mediated: fixed readout (fit on true-v0) applied to predicted v0
            predB = ridge_readout_loco_mediated(Y_pca, pred_pca, y_e0)
            # (C) direct-from-c_C ridge (raw c_C -> E0)
            predC = ridge_readout_loco_raw(Xc, y_e0)

            rhoA = _spearman(predA, y_e0)
            rhoB = _spearman(predB, y_e0)
            rhoC = _spearman(predC, y_e0)
            rows.append(
                {
                    "layer": li,
                    "rho_ridge_direct": rhoA,
                    "rho_ridge_mediated": rhoB,
                    "rho_cc_direct": rhoC,
                    "rho_degradation": (rhoA - rhoB)
                    if (rhoA is not None and rhoB is not None)
                    else None,
                }
            )
            if rhoA is not None and (best_A is None or rhoA > best_A["rho"]):
                best_A = {"layer": li, "rho": rhoA}
            if rhoB is not None and (best_B is None or rhoB > best_B["rho"]):
                best_B = {"layer": li, "rho": rhoB}
            if rhoC is not None and (best_C is None or rhoC > best_C["rho"]):
                best_C = {"layer": li, "rho": rhoC}

        per_behavior[b] = {
            "n_kept_e0": len(kept_idx),
            "best_ridge_direct": best_A,
            "best_ridge_mediated": best_B,
            "best_cc_direct": best_C,
            "best_layer_degradation": (best_A["rho"] - best_B["rho"])
            if (best_A and best_B)
            else None,
            "per_layer": rows,
        }
        print(
            f"  {b:22s} n={len(kept_idx)} "
            f"A_best={best_A['rho']:.3f}@L{best_A['layer']} "
            f"B_best={best_B['rho']:.3f}@L{best_B['layer']} "
            f"C_best={best_C['rho']:.3f}@L{best_C['layer']} "
            f"deg={(best_A['rho'] - best_B['rho']):.3f}"
        )

    out = {
        "analysis": "behavioral_chain_preservation_RIDGE_READOUT",
        "description": (
            "Spearman rho(predicted E0, actual E0), LOCO n=50, with a RIDGE v0->E0 "
            "readout (not r_B diff-of-means). A=direct ridge on true PCA48(v0); "
            "B=mediated: fixed readout (fit on true v0) applied to c_C-predicted v0; "
            "C=ridge fit directly on c_C. Degradation = rho_A - rho_B."
        ),
        "readout": "RidgeCV(alphas=[1e-2..1e3]) scalar E0; PCA48 v0 space (A/B); raw c_C (C)",
        "pca_target_dim": PCA_TARGET_DIM,
        "cc_variant": sub["cc_variant"],
        "e0_src": sub["e0_src"],
        "layers": layers,
        "per_behavior": per_behavior,
    }
    out_path = PROJECT_ROOT / "eval_results/issue_722/structural/behavior_chain_ridge_readout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[info] wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
