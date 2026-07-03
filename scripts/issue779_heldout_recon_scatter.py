#!/usr/bin/env python3
"""Issue #779: held-out reconstruction scatter for the map h (chat-requested figure).

Visualizes the honest held-out reconstruction (R2 0.598-0.625, 5-fold CV over the
5000 LMSYS train contexts; eval_results/issue_779/percontext_recon.json) as raw
scatter, per trait at its read-out layer, on fold 0's held-out contexts:

  top row    -- pooled per-(context, dimension) true vs predicted activation value
                (random subsample of held-out contexts x dims; pooled R2 computed
                on the FULL fold, annotated alongside the 5-fold mean +- sd);
  bottom row -- per-held-out-context true vs predicted answer projection
                <v(x), r_B> vs <h(c_last), r_B> (every fold-0 test context).

Reuses the EXACT protocol of scripts/issue779_percontext_recon.py: same
_cv_folds(seed=0, n_folds=5) split, same _ridge_fit_predict_fast dual-ridge
(equivalence-gated there), same test-fold-mean R2 convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue779_percontext_recon import (  # noqa: E402
    READ_OUT_LAYER,
    _cv_folds,
    _pooled_r2,
    _ridge_fit_predict_fast,
)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

COLLECT = PROJECT_ROOT / "data" / "issue779_hfstage" / "issue779_monitoring" / "analysis_tensors"
RECON_JSON = PROJECT_ROOT / "eval_results" / "issue_779" / "percontext_recon.json"

N_CTX_SAMPLE = 500  # held-out contexts subsampled for the pooled value scatter
N_DIM_SAMPLE = 40  # activation dims subsampled for the pooled value scatter
SEED = 0


def main() -> int:
    torch.set_num_threads(8)
    bundle = torch.load(
        COLLECT / "pass_b" / "train_context_vectors.pt", weights_only=False, mmap=True
    )
    layers = bundle["layers"]
    recon = json.loads(RECON_JSON.read_text()) if RECON_JSON.exists() else {}
    fivefold = {
        t: recon.get("read1_heldout_recon", {}).get(t, {}).get("readout_layer", {})
        for t in READ_OUT_LAYER
    }

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    rng = np.random.default_rng(SEED)
    color = paper_palette_role("primary")

    for col, trait in enumerate(("evil", "sycophancy", "hallucination")):
        li = layers.index(READ_OUT_LAYER[trait])
        X = bundle["cx_last"][:, li, :].to(torch.float64).numpy()
        Y = bundle["v_x"][:, li, :].to(torch.float64).numpy()
        folds = _cv_folds(X.shape[0], 5, SEED)
        test = folds[0]
        train = np.setdiff1d(np.arange(X.shape[0]), test)
        pred = _ridge_fit_predict_fast(X[train], Y[train], X[test])
        true = Y[test]
        r2_fold = _pooled_r2(pred, true)

        # top: pooled per-(context, dim) value scatter (subsampled for legibility)
        ctx_idx = rng.choice(true.shape[0], size=min(N_CTX_SAMPLE, true.shape[0]), replace=False)
        dim_idx = rng.choice(true.shape[1], size=N_DIM_SAMPLE, replace=False)
        t_sub = true[np.ix_(ctx_idx, dim_idx)].ravel()
        p_sub = pred[np.ix_(ctx_idx, dim_idx)].ravel()
        ax = axes[0, col]
        ax.scatter(t_sub, p_sub, s=3, alpha=0.12, color=color, rasterized=True)
        lim = np.quantile(np.abs(np.concatenate([t_sub, p_sub])), 0.999)
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1.0, color="gray")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ff = fivefold.get(trait, {})
        ff_txt = (
            f"5-fold R$^2$ = {ff.get('r2_mean'):.3f} $\\pm$ {ff.get('r2_sd'):.3f}"
            if ff.get("r2_mean") is not None
            else ""
        )
        ax.set_title(f"{trait} (layer {READ_OUT_LAYER[trait]})")
        ax.text(
            0.03,
            0.97,
            f"held-out fold R$^2$ = {r2_fold:.3f}\n{ff_txt}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.set_xlabel("true response-mean activation value")
        if col == 0:
            ax.set_ylabel("predicted activation value (ridge h)")

        # bottom: per-context answer projection, true vs predicted
        rb = torch.load(COLLECT / "r_b" / f"{trait}.pt", weights_only=False)["r_b"]
        rb_l = rb[li].to(torch.float64).numpy()
        proj_true = true @ rb_l
        proj_pred = pred @ rb_l
        r = float(np.corrcoef(proj_true, proj_pred)[0, 1])
        # permutation p-value would be overkill at n=1000; normal approx via t
        from scipy import stats

        _, pval = stats.pearsonr(proj_true, proj_pred)
        ax2 = axes[1, col]
        ax2.scatter(proj_true, proj_pred, s=8, alpha=0.35, color=color, rasterized=True)
        lo = min(proj_true.min(), proj_pred.min())
        hi = max(proj_true.max(), proj_pred.max())
        ax2.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="gray")
        ax2.text(
            0.03,
            0.97,
            f"r = {r:.3f} (p = {pval:.1e})\nn = {len(test)} held-out contexts",
            transform=ax2.transAxes,
            va="top",
            fontsize=9,
        )
        ax2.set_xlabel("true answer projection onto the trait direction")
        if col == 0:
            ax2.set_ylabel("predicted answer projection (ridge h)")

    fig.suptitle(
        "Held-out reconstruction of the answer profile by the LMSYS-trained map h "
        "(fold 0 of the 5-fold CV over 5000 LMSYS contexts)",
        y=1.00,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_779/heldout_recon_scatter", dir="figures/", embed_data=False)
    plt.close(fig)
    print("saved figures/issue_779/heldout_recon_scatter.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
