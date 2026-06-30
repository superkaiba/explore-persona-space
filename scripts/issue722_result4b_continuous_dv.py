#!/usr/bin/env python3
"""#722 Result-4b: behavior-chain readout on the CONTINUOUS behavior DV.

CONTINUOUS-DV twin of the rate-based Result 4
(``issue722_behavior_chain_ridge_readout.py`` / ``issue722_result4_ci.py``).
Swaps the prediction TARGET from the binary judged rate ``E0`` to the
continuous DV ``logp_pos_mean`` (length-normalized log-P of the model's own
judged-positive completions) — because the rate floor-saturates for EM /
sycophancy (broad_em rate std = 0.008) while the continuous DV keeps dynamic
range (broad_em logp_pos_mean std = 0.155).

For each behavior B in {sycophancy, refusal, broad_em, harmful_compliance},
per layer L, leave-one-CONTEXT-out (LOCO; n=50, n≈25 for broad_em — the
contexts with ≥1 positive completion), compute Spearman ρ(predicted DV,
actual DV) for:

  DIRECT   ridge( true v_A )            -> logp_pos_mean
  MEDIATED ridge( M·v_C = LOCO-pred v_A ) -> logp_pos_mean   (fixed readout fit
            on true v_A, applied to the c_C->v_A LOCO-predicted v_A)

Then at each behavior's own best layer, a 95% bootstrap CI (2000 draws over the
n contexts, resampling final (held-out-pred, DV) pairs).

REUSE contract — the readout machinery is imported VERBATIM from
``issue722_behavior_chain_ridge_readout`` (PCA-48 ``robust_pca_basis``,
``ridge_predict_loco_centered`` for M, ``ridge_readout_loco_direct`` /
``ridge_readout_loco_mediated`` RidgeCV scalar readouts) so the only difference
from the rate-based run is the target column. The bootstrap-CI helper mirrors
``issue722_result4_ci._boot_ci``.

DEVIATIONS from the rate-based committed script, both per the task brief:
  1. c_C source is ``issue594_context_geometry/analysis_tensors/
     context_vectors_mean.pt`` (the #594 mean context vector, ``tensor`` +
     ``instance_ids``), REINDEXED to v_A's context order — NOT the
     ``cc_meanprompt`` / ``cc_last`` variants ``load_substrate`` defaults to.
  2. broad_em's logp_pos_mean exists for only ~25 contexts; LOCO runs over the
     kept rows only, the reduced n is reported, nothing is imputed.

REQUIRED validation: the continuous DV is a SECONDARY companion — per behavior
we report Spearman(logp_pos_mean, rate) across contexts (where rate has dynamic
range). If |ρ| < 0.3 the continuous-DV result is FLAGGED unvalidated.

0-GPU, CPU only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")  # inputs
OUT_ROOT = Path(__file__).resolve().parent.parent  # worktree root — figures land here
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the EXACT readout machinery (so direct/mediated stay consistent with the
# rate-based run; only the target column differs).
from issue722_behavior_chain_ridge_readout import (  # noqa: E402
    N_LAYERS,
    PCA_TARGET_DIM,
    V0_FILE,
    ridge_readout_loco_direct,
    ridge_readout_loco_mediated,
)

from explore_persona_space.analysis import paper_plots  # noqa: E402
from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
    robust_pca_basis,
)

BEHAVIORS = ["sycophancy", "refusal", "broad_em", "harmful_compliance"]
BEHAVIOR_LABEL = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "broad_em": "broad EM",
    "harmful_compliance": "harmful\ncompliance",
}

# c_C source per task brief (the #594 mean context vector).
CC_FILE = (
    PROJECT_ROOT
    / "data/issue_658/prb_dl/issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
)
E0_LOCAL = PROJECT_ROOT / "eval_results/issue_658/E0_expression.json"
E0_HF = "issue658_partial/att-20260624-130414/eval_results_issue_658/E0_expression.json"
HF_REPO = "superkaiba1/explore-persona-space-data"

N_BOOT = 2000
BOOT_SEED = 722
VALIDATION_THRESHOLD = 0.3  # |ρ(DV,rate)| below this => DV result flagged unvalidated

# rate-based DIRECT ρ from the earlier run (cited, NOT recomputed here).
RATE_DIRECT_REFERENCE = {
    "sycophancy": 0.855,
    "refusal": 0.424,
    "harmful_compliance": 0.507,
    "broad_em": None,  # rate floor-saturated -> uninformative
}

C_DIRECT = "#0072B2"  # ridge from true v_A
C_MEDIATED = "#009E73"  # ridge from M v_C (predicted v_A)


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    r, _ = spearmanr(a, b)
    return None if np.isnan(r) else float(r)


def dv_vector(e0: dict, column_id: str, ctx_ids: list[str]) -> tuple[np.ndarray, list[int]]:
    """Continuous-DV target: logp_pos_mean per context. Returns (values, kept_idx)
    where kept_idx indexes ctx_ids for the contexts that HAVE a finite
    logp_pos_mean (broad_em: ~25 of 50). No imputation."""
    vals: list[float] = []
    kept_idx: list[int] = []
    for i, c in enumerate(ctx_ids):
        cell = e0.get("e0", {}).get(c, {}).get(column_id)
        if cell is None:
            continue
        v = cell.get("logp_pos_mean")
        if v is None:
            continue
        v = float(v)
        if not np.isfinite(v):
            continue
        vals.append(v)
        kept_idx.append(i)
    return np.array(vals, dtype=np.float64), kept_idx


def rate_vector(e0: dict, column_id: str, ctx_ids: list[str]) -> tuple[np.ndarray, list[int]]:
    """rate per context, kept_idx where rate is present (for DV-vs-rate validation)."""
    vals: list[float] = []
    kept_idx: list[int] = []
    for i, c in enumerate(ctx_ids):
        cell = e0.get("e0", {}).get(c, {}).get(column_id)
        if cell is None:
            continue
        v = cell.get("rate")
        if v is None:
            continue
        vals.append(float(v))
        kept_idx.append(i)
    return np.array(vals, dtype=np.float64), kept_idx


def load_substrate():
    """v_A from issue_658 v0 store; c_C from the #594 context_vectors_mean.pt
    (reindexed to v_A's context order, per brief); E0 (incl. logp_pos_mean +
    rate) from local JSON else HF."""
    v0 = torch.load(V0_FILE, weights_only=False)
    ctx_ids = list(v0["context_ids"])
    layers = list(v0["capture_layers"])
    assert layers == list(range(N_LAYERS)), layers

    V = np.stack([v0["summaries"]["mean"][c].numpy() for c in ctx_ids]).astype(
        np.float64
    )  # (50,28,3584)

    # c_C: #594 mean context vector. The tensor is in instance_ids order, which
    # DIFFERS from v_A's context_ids order — REINDEX (a naive np.stack would
    # misalign contexts).
    cc = torch.load(CC_FILE, weights_only=False)
    iid = list(cc["instance_ids"])
    Tcc = cc["tensor"].numpy().astype(np.float64)  # (50,28,3584)
    assert set(iid) == set(ctx_ids), "c_C instance set != v_A context set"
    pos = {c: j for j, c in enumerate(iid)}
    C = np.stack([Tcc[pos[c]] for c in ctx_ids]).astype(np.float64)  # reindexed to ctx_ids

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
        "cc_source": str(CC_FILE),
        "e0": e0,
        "e0_src": e0_src,
    }


def _preds_at_layer(sub, layer: int, kept_idx: list[int], y_dv: np.ndarray):
    """Return (direct_preds, mediated_preds): LOCO held-out continuous-DV
    predictions at `layer` for the kept rows. PCA-48 + ridge readout reused
    verbatim. PCA basis + LOCO M map are fit on the FULL 50 contexts (the input
    representation), then the readout LOCO runs over kept rows only."""
    V, C = sub["V"], sub["C"]
    Y = V[:, layer, :]  # (50, H) true v_A
    Xc = C[:, layer, :]  # (50, H) c_C
    mu, comps, _fb = robust_pca_basis(Y, PCA_TARGET_DIM)
    Y_pca = (Y - mu) @ comps.T  # (50, k) true PCA48 coords
    pred_pca = ridge_predict_loco_centered(Xc, Y_pca)  # (50, k) LOCO c_C->PCA48(v_A)
    Y_pca_k = Y_pca[kept_idx]
    pred_pca_k = pred_pca[kept_idx]
    predA = ridge_readout_loco_direct(Y_pca_k, y_dv)  # ridge on true v_A
    predB = ridge_readout_loco_mediated(Y_pca_k, pred_pca_k, y_dv)  # ridge on M v_C
    return predA, predB


def _boot_ci(pred: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Bootstrap CI for Spearman ρ over the n contexts (resample final pairs)."""
    n = len(y)
    rhos = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        a, c = pred[idx], y[idx]
        if np.std(a) < 1e-12 or np.std(c) < 1e-12:
            rhos[b] = np.nan
            continue
        r, _ = spearmanr(a, c)
        rhos[b] = r
    rhos = rhos[~np.isnan(rhos)]
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    r0, _ = spearmanr(pred, y)
    return {"point": float(r0), "lo": float(lo), "hi": float(hi)}


def main():
    sub = load_substrate()
    print(f"[info] c_C source: {sub['cc_source']}")
    print(f"[info] E0 source:  {sub['e0_src']}")
    print(f"[info] V shape {sub['V'].shape}  C shape {sub['C'].shape}")
    rng = np.random.default_rng(BOOT_SEED)
    layers = sub["layers"]

    per_behavior = {}
    for b in BEHAVIORS:
        y_dv, kept_idx = dv_vector(sub["e0"], b, sub["ctx_ids"])
        n = len(kept_idx)
        if n < 8:
            per_behavior[b] = {"status": f"too few logp_pos_mean ({n})"}
            print(f"  {b:22s} SKIP — only {n} contexts with logp_pos_mean")
            continue

        # ── REQUIRED validation: Spearman(logp_pos_mean, rate) across contexts ──
        y_rate, kept_rate = rate_vector(sub["e0"], b, sub["ctx_ids"])
        rate_by_ctx = {sub["ctx_ids"][i]: r for i, r in zip(kept_rate, y_rate)}
        dv_for_val, rate_for_val = [], []
        for i, v in zip(kept_idx, y_dv):
            c = sub["ctx_ids"][i]
            if c in rate_by_ctx:
                dv_for_val.append(v)
                rate_for_val.append(rate_by_ctx[c])
        dv_for_val = np.asarray(dv_for_val)
        rate_for_val = np.asarray(rate_for_val)
        rate_has_range = float(np.std(rate_for_val)) > 1e-9
        val_rho = _spearman(dv_for_val, rate_for_val) if rate_has_range else None
        validated = (val_rho is not None) and (abs(val_rho) >= VALIDATION_THRESHOLD)

        # ── per-layer LOCO ρ (continuous DV), select best layer per readout ──
        rows = []
        best_A = best_B = None
        for li in layers:
            predA, predB = _preds_at_layer(sub, li, kept_idx, y_dv)
            rhoA = _spearman(predA, y_dv)
            rhoB = _spearman(predB, y_dv)
            rows.append(
                {
                    "layer": li,
                    "rho_direct": rhoA,
                    "rho_mediated": rhoB,
                    "degradation": (rhoA - rhoB)
                    if (rhoA is not None and rhoB is not None)
                    else None,
                }
            )
            if rhoA is not None and (best_A is None or rhoA > best_A["rho"]):
                best_A = {"layer": li, "rho": rhoA}
            if rhoB is not None and (best_B is None or rhoB > best_B["rho"]):
                best_B = {"layer": li, "rho": rhoB}

        # ── bootstrap CI at each readout's own best layer ──
        predA_bd, _ = _preds_at_layer(sub, best_A["layer"], kept_idx, y_dv)
        _, predB_bm = _preds_at_layer(sub, best_B["layer"], kept_idx, y_dv)
        ci_direct = _boot_ci(predA_bd, y_dv, rng)
        ci_mediated = _boot_ci(predB_bm, y_dv, rng)
        # consistency: bootstrap point estimate must match the per-layer scan
        assert abs(ci_direct["point"] - best_A["rho"]) < 1e-6, (b, ci_direct["point"], best_A)
        assert abs(ci_mediated["point"] - best_B["rho"]) < 1e-6, (b, ci_mediated["point"], best_B)

        per_behavior[b] = {
            "n": n,
            "dv_std": float(np.std(y_dv)),
            "rate_std": float(np.std(rate_for_val)) if len(rate_for_val) else None,
            "validation_rho_dv_vs_rate": val_rho,
            "validation_n": len(dv_for_val),
            "rate_has_dynamic_range": rate_has_range,
            "validated": bool(validated),
            "L_direct": best_A["layer"],
            "L_mediated": best_B["layer"],
            "best": {"direct": ci_direct, "mediated": ci_mediated},
            "rate_direct_reference": RATE_DIRECT_REFERENCE[b],
            "scatter": {
                "ctx_ids": [sub["ctx_ids"][i] for i in kept_idx],
                "y_dv": y_dv.tolist(),
                "pred_direct": predA_bd.tolist(),
                "pred_mediated": predB_bm.tolist(),
            },
            "per_layer": rows,
        }
        flag = "" if validated else "  [DV-vs-rate UNVALIDATED]"
        print(
            f"  {b:22s} n={n:3d}  "
            f"direct ρ={ci_direct['point']:.3f}[{ci_direct['lo']:.3f},{ci_direct['hi']:.3f}]@L{best_A['layer']}  "
            f"mediated ρ={ci_mediated['point']:.3f}[{ci_mediated['lo']:.3f},{ci_mediated['hi']:.3f}]@L{best_B['layer']}  "
            f"| DV-vs-rate ρ={val_rho if val_rho is None else round(val_rho, 3)}{flag}"
        )

    out = {
        "analysis": "result4b_behavior_chain_CONTINUOUS_DV_with_CI",
        "description": (
            "Spearman ρ(predicted logp_pos_mean, actual logp_pos_mean), LOCO, ridge "
            "v_A->DV readout. DIRECT = ridge on true v_A; MEDIATED = fixed readout (fit "
            "on true v_A) applied to c_C->v_A LOCO-predicted v_A (M·v_C). Best-layer ρ + "
            "95% bootstrap CI (2000 draws over contexts). Continuous DV is the SECONDARY "
            "companion to the binary rate; per-behavior DV-vs-rate validation reported."
        ),
        "target": "logp_pos_mean (length-normalized log-P of judged-positive completions)",
        "readout": "RidgeCV(alphas=[1e-2..1e3]) scalar; PCA48 v_A space",
        "pca_target_dim": PCA_TARGET_DIM,
        "cc_source": sub["cc_source"],
        "e0_src": sub["e0_src"],
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "validation_threshold": VALIDATION_THRESHOLD,
        "rate_direct_reference": RATE_DIRECT_REFERENCE,
        "layers": layers,
        "per_behavior": per_behavior,
    }
    out_path = OUT_ROOT / "eval_results/issue_722/structural/result4b_continuous_dv.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[info] wrote {out_path}")

    _make_bar_figure(per_behavior)
    return out


def _make_bar_figure(per_behavior):
    paper_plots.set_paper_style("blog")
    shown = [b for b in BEHAVIORS if "best" in per_behavior.get(b, {})]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(shown))
    w = 0.38
    pts_d = [per_behavior[b]["best"]["direct"] for b in shown]
    pts_m = [per_behavior[b]["best"]["mediated"] for b in shown]
    err_d = np.array([[p["point"] - p["lo"] for p in pts_d], [p["hi"] - p["point"] for p in pts_d]])
    err_m = np.array([[p["point"] - p["lo"] for p in pts_m], [p["hi"] - p["point"] for p in pts_m]])
    ax.bar(
        x - w / 2,
        [p["point"] for p in pts_d],
        w,
        yerr=err_d,
        capsize=4,
        color=C_DIRECT,
        label=r"ridge from true $v_A$",
        error_kw={"ecolor": "#222", "elinewidth": 1.3},
    )
    ax.bar(
        x + w / 2,
        [p["point"] for p in pts_m],
        w,
        yerr=err_m,
        capsize=4,
        color=C_MEDIATED,
        label=r"ridge from $M\,v_C$ (predicted $v_A$)",
        error_kw={"ecolor": "#222", "elinewidth": 1.3},
    )
    ax.axhline(0.0, color="#aaa", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABEL[b] for b in shown])
    ax.set_ylabel(r"held-out $\rho$(predicted DV, actual DV)")
    lo_lim = min(0.0, min(p["lo"] for p in pts_d + pts_m)) - 0.05
    ax.set_ylim(lo_lim, 1.0)
    ax.set_title(
        r"Predicting the continuous behavior DV (log-P of positive completions)"
        "\n"
        r"from $v_A$ vs $M\,v_C$",
    )
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    paper_plots.savefig_paper(
        fig,
        "result4b_continuous_dv",
        dir=str(OUT_ROOT / "figures/issue_722"),
    )
    plt.close(fig)
    print("[info] wrote result4b_continuous_dv.png")


if __name__ == "__main__":
    main()
