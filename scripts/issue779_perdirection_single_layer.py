"""Issue #779 inline free-analysis — per-direction R2 at ONE layer, all 3 r_B dots.

The committed h_perdirection_r2.png uses 3 panels, each at that trait's own
read-out layer (evil L14 / sycophancy L26 / hallucination L17). This makes a
SINGLE panel at the reconstruction-best layer (L19, where the linear context->
answer map peaks, held-out v_x R2 0.678) and overlays all 3 traits' r_B on that
one shared spectrum. The blue per-direction R2 curve, variance-share curve, and
random-direction band are identical across traits (they depend only on the
map + answer PCA), so they are computed once; only the 3 r_B markers differ.

Same protocol as identity_baseline (fold 0 of 5-fold, seed 0, k_lead 200,
tail_step 20, 50 random dirs, full-ridge h). 0-GPU, cached pass_b + r_B.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import issue779_common as C
import issue779_fitter_fair_comparison as F
import issue779_identity_baseline as IB
import issue779_stage1 as S1
import numpy as np

from explore_persona_space.orchestrate.env import load_dotenv

TRAITS = ("evil", "sycophancy", "hallucination")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")  # for _base_metadata
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--k-lead", type=int, default=200)
    ap.add_argument("--tail-step", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--rb-dir", type=Path, default=Path("data/issue_779/r_b"))
    ap.add_argument(
        "--out-json", type=Path, default=F.DEFAULT_OUT_DIR / "perdirection_single_layer.json"
    )
    ap.add_argument("--fig-dir", type=Path, default=F.DEFAULT_FIG_DIR)
    args = ap.parse_args()

    bundle = F.load_pass_b()
    layers = list(bundle["layers"])
    li = layers.index(args.layer)
    X = bundle["cx_last"][:, li, :].to(dtype=__import__("torch").float32).numpy()
    Y = bundle["v_x"][:, li, :].to(dtype=__import__("torch").float32).numpy()
    n = X.shape[0]
    test_idx = F.PR._cv_folds(n, args.n_folds, args.seed)[0]

    per_trait = {}
    for t in TRAITS:
        rb = S1._load_rb(args.rb_dir, t, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
        res = IB.analysis_d_layer(
            X,
            Y,
            rb[li],
            test_idx,
            k_lead=args.k_lead,
            tail_step=args.tail_step,
            n_random=args.n_random,
            seed=args.seed,
        )
        per_trait[t] = res

    shared = per_trait[TRAITS[0]]  # curve/band identical across traits
    out = {
        "layer": args.layer,
        "ranks_evaluated": shared["ranks_evaluated"],
        "r2_by_rank": shared["r2_by_rank"],
        "variance_share_by_rank": shared["variance_share_by_rank"],
        "random_directions": shared["random_directions"],
        "r_b_by_trait": {t: per_trait[t]["r_b"] for t in TRAITS},
        "note": (
            f"Per-direction held-out R2 at L{args.layer} (reconstruction-best layer for the "
            "context->v_x map). Single shared answer-PCA spectrum; all 3 traits' r_B overlaid."
        ),
        "metadata": {
            "script": "issue779_perdirection_single_layer",
            "layer": args.layer,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "k_lead": args.k_lead,
            "tail_step": args.tail_step,
            "n_random": args.n_random,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    F.C.write_json_atomic(args.out_json, out)

    # ---- figure: single panel, blue curve + 3 red r_B dots ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    ranks = np.array(shared["ranks_evaluated"], float) + 1  # 1-based for log axis
    r2 = np.array(shared["r2_by_rank"], float)
    share = np.array(shared["variance_share_by_rank"], float)
    rd = shared["random_directions"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ranks, r2, "-", color="#1f4e9c", lw=1.2, label="PCA direction R2 (held-out)")
    ax.axhspan(
        rd["r2_mean"] - rd["r2_sd"],
        rd["r2_mean"] + rd["r2_sd"],
        color="0.6",
        alpha=0.25,
        label="random dirs (n=50, mean±sd)",
    )
    ax.axhline(rd["r2_mean"], color="0.55", ls=":", lw=1.0)
    ax.axhline(0.0, color="black", lw=1.0)
    star_colors = {"evil": "#d62728", "sycophancy": "#e6550d", "hallucination": "#b5179e"}
    for t in TRAITS:
        rbm = per_trait[t]["r_b"]
        xr = rbm["equivalent_variance_rank"] + 1
        ax.scatter(
            [xr],
            [rbm["heldout_r2"]],
            marker="*",
            s=240,
            zorder=5,
            color=star_colors[t],
            edgecolor="white",
            linewidth=0.6,
            label=f"r_B {t} (rank {rbm['equivalent_variance_rank']}, R2={rbm['heldout_r2']:.2f})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("variance rank k (1-based, log)")
    ax.set_ylabel("held-out per-direction R2")
    ax2 = ax.twinx()
    ax2.plot(ranks, share, "--", color="#2ca02c", lw=1.0, alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_ylabel("train variance share (log)", color="#2ca02c")
    ax.set_title(
        f"Per-direction predictability at L{args.layer} (best recon layer) — all 3 traits' r_B"
    )
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    figs = savefig_paper(fig, "h_perdirection_r2_single_layer", dir=args.fig_dir, embed_data=False)
    plt.close(fig)

    print(f"wrote {args.out_json} and {figs.get('png')}")
    for t in TRAITS:
        rbm = per_trait[t]["r_b"]
        print(
            f"  {t:14s} r_B: rank {rbm['equivalent_variance_rank']:>3d}, "
            f"pct {rbm['variance_percentile_of_pca_spectrum']:.2f}, R2 {rbm['heldout_r2']:.3f}, "
            f"matched-null {rbm['pca_r2_at_matched_variance']['r2_mean']:.3f}"
        )
    print(f"  random-dir mean {rd['r2_mean']:.3f} ± {rd['r2_sd']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
