"""Issue #779 inline free-analysis — per-direction single-rollout RELIABILITY (ICC).

Consumes the K=10-rollout capture (issue779_reliability_gen_capture.py) and, per
answer-PCA direction at L19, computes the single-rollout reliability

    ICC = between-context variance / (between + within-context variance)

where, along each PCA direction u_k:
  * between = variance (across the 600 contexts) of the per-context rollout MEAN
    projection <mean_k v(x)_ctx,rollout , u_k>;
  * within  = mean (across contexts) of the per-context rollout VARIANCE of
    <v(x)_ctx,rollout , u_k>.

ICC is the fraction of a single rollout's projection variance that reflects true
between-context differences — i.e. the NOISE CEILING for the per-direction R2
curve in the committed h_perdirection_r2_single_layer plot. Read side-by-side:
  * R2(k) ~= reliability(k)  -> the map is NOISE-LIMITED at rank k (it already
    predicts as much of that direction as a single rollout reliably carries);
  * R2(k) << reliability(k)  -> MISSED SIGNAL (reliable between-context variance
    the map failed to capture).

The answer-PCA basis is the SAME as the per-direction plot: fit ONCE on the
fold-0 (seed-0) TRAIN targets of pass_b v_x@19 (train-mean-centered), same
lead+tail rank ladder — reused from issue779_perdirection_per_predictor._pca_basis
so the reliability ranks align 1:1 with the R2 curve. r_B's reliability vs its R2
is reported per trait.

Also reports a bias-corrected ICC (between - within/Kbar) as a secondary read —
the per-context-mean estimator inflates the raw between-variance by ~within/K.

0-GPU, CPU/VM only. Fail loud; NaN reported, never coerced. ``--smoke`` runs the
ICC math + figure on synthetic fixtures (no real pass_b / bundle needed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_perdirection_per_predictor as PP  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

TRAITS = ("evil", "sycophancy", "hallucination")
STAR_COLORS = {"evil": "#d62728", "sycophancy": "#e6550d", "hallucination": "#b5179e"}


def _icc_per_direction(P: np.ndarray, valid: np.ndarray, *, min_valid: int = 2) -> dict:
    """Per-direction ICC over projections ``P`` (C, K, R) with a ``valid`` (C, K)
    mask. Only contexts with >= min_valid valid rollouts enter (per-context
    variance needs >=2). between = var of per-context rollout means (ddof=1);
    within = mean of per-context rollout variances (ddof=1). Also a bias-corrected
    ICC (between - within/Kbar). NaN where the total variance is degenerate."""
    valid_b = valid.astype(bool)
    nvalid = valid_b.sum(1)  # (C,)
    keep = nvalid >= min_valid
    if keep.sum() < 2:
        R = P.shape[2]
        nan = np.full(R, np.nan)
        return {
            "icc": nan,
            "icc_adj": nan,
            "between": nan,
            "within": nan,
            "n_contexts_used": int(keep.sum()),
            "kbar": float("nan"),
        }
    Pk = P[keep].astype(np.float64)  # (Ck, K, R)
    vm = valid_b[keep].astype(np.float64)[:, :, None]  # (Ck, K, 1)
    nv = nvalid[keep].astype(np.float64)[:, None]  # (Ck, 1)
    m = (vm * Pk).sum(1) / nv  # (Ck, R) per-context rollout mean
    dev = (Pk - m[:, None, :]) * vm
    s2 = (dev**2).sum(1) / (nv - 1.0)  # (Ck, R) per-context rollout variance (ddof=1)
    within = s2.mean(0)  # (R,)
    between = m.var(0, ddof=1)  # (R,) variance of per-context means
    denom = between + within
    with np.errstate(divide="ignore", invalid="ignore"):
        icc = np.where(denom > 1e-30, between / denom, np.nan)
    kbar = float(nvalid[keep].mean())
    between_adj = np.maximum(0.0, between - within / kbar)
    denom_adj = between_adj + within
    with np.errstate(divide="ignore", invalid="ignore"):
        icc_adj = np.where(denom_adj > 1e-30, between_adj / denom_adj, np.nan)
    return {
        "icc": icc,
        "icc_adj": icc_adj,
        "between": between,
        "within": within,
        "n_contexts_used": int(keep.sum()),
        "kbar": kbar,
    }


def _project(vx: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """(C, K, H) rollout v_x -> (C, K, R) projections onto unit directions dirs (H, R)."""
    return np.einsum("ckh,hr->ckr", vx.astype(np.float64), dirs)


def _compute(vx19: np.ndarray, valid: np.ndarray, pca: dict, rb: dict) -> dict:
    """Reliability per PCA direction + per trait r_B, from the (C, K, H) rollout
    projections at L19 onto the shared fold-0 answer-PCA basis."""
    P = _project(vx19, pca["dirs"])  # (C, K, R)
    per_dir = _icc_per_direction(P, valid)
    rb_out = {}
    for t, info in rb.items():
        Pr = _project(vx19, info["u"][:, None])  # (C, K, 1)
        r = _icc_per_direction(Pr, valid)
        rb_out[t] = {
            "reliability_icc": float(r["icc"][0]),
            "reliability_icc_adj": float(r["icc_adj"][0]),
            "equivalent_variance_rank": info["equivalent_variance_rank"],
        }
    return {
        "ranks_evaluated": [int(x) for x in pca["ranks"]],
        "reliability_by_rank": [float(x) for x in per_dir["icc"]],
        "reliability_adj_by_rank": [float(x) for x in per_dir["icc_adj"]],
        "between_by_rank": [float(x) for x in per_dir["between"]],
        "within_by_rank": [float(x) for x in per_dir["within"]],
        "n_contexts_used": per_dir["n_contexts_used"],
        "mean_valid_rollouts_per_context": per_dir["kbar"],
        "r_b_reliability": rb_out,
    }


def _build_figure(out: dict, committed_json: Path, fig_dir: Path) -> None:
    """Per-direction R2 curve (committed single-layer plot) + reliability ceiling
    overlay; shade the missed-signal gap; r_B reliability vs R2 markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    committed = json.loads(committed_json.read_text())
    assert committed["ranks_evaluated"] == out["ranks_evaluated"], (
        "rank ladder mismatch between reliability analysis and committed per-direction plot"
    )
    set_paper_style("blog")
    ranks = np.array(out["ranks_evaluated"], float) + 1
    r2 = np.array(committed["r2_by_rank"], float)
    rel = np.array(out["reliability_by_rank"], float)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.plot(ranks, r2, "-", color="#1f4e9c", lw=1.3, label="per-direction R2 (held-out ridge map)")
    ax.plot(
        ranks, rel, "-", color="#e6820d", lw=1.3, label="single-rollout reliability (ICC ceiling)"
    )
    gap = np.isfinite(r2) & np.isfinite(rel) & (rel > r2)
    ax.fill_between(
        ranks,
        r2,
        rel,
        where=gap,
        color="#d62728",
        alpha=0.12,
        label="missed signal (reliable variance the map misses)",
    )
    ax.axhline(0.0, color="black", lw=1.0)
    for t in TRAITS:
        info = out["r_b_reliability"][t]
        xr = info["equivalent_variance_rank"] + 1
        r2_rb = committed["r_b_by_trait"][t]["heldout_r2"]
        ax.scatter(
            [xr],
            [info["reliability_icc"]],
            marker="*",
            s=200,
            zorder=6,
            color=STAR_COLORS[t],
            edgecolor="white",
            linewidth=0.6,
            label=f"r_B {t}: reliability {info['reliability_icc']:.2f} vs R2 {r2_rb:.2f}",
        )
        ax.scatter(
            [xr],
            [r2_rb],
            marker="o",
            s=42,
            zorder=6,
            facecolor="none",
            edgecolor=STAR_COLORS[t],
            linewidth=1.2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("answer-PCA variance rank k (1-based, log)")
    ax.set_ylabel("held-out R2  /  single-rollout reliability (ICC)")
    ax.set_title(
        f"Per-direction predictability vs single-rollout reliability ceiling at "
        f"L{committed.get('layer', 19)}"
    )
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    figs = savefig_paper(fig, "h_perdirection_reliability", dir=fig_dir, embed_data=False)
    plt.close(fig)
    print(f"wrote {figs.get('png')}")


def _smoke(fig_dir: Path) -> int:
    """ICC math + figure on synthetic fixtures (no real pass_b / bundle / model).

    Builds a small answer space with a KNOWN reliability gradient: leading PCA
    directions carry a strong between-context signal (high ICC), tail directions
    are pure rollout noise (ICC ~ 0). Asserts the recovered ICC ordering, then
    exercises the figure code against a synthetic committed-style curve."""
    rng = np.random.default_rng(0)
    H, n_tr, C_, K = 48, 200, 120, 6
    # train targets: strong variance in the first few dims, tiny in the tail.
    scale = np.concatenate([np.array([8.0, 5.0, 3.0, 2.0]), 0.2 * np.ones(H - 4)])
    Ytr = rng.standard_normal((n_tr, H)) * scale
    pca = PP._pca_basis(Ytr, k_lead=8, tail_step=6)

    # reliability rollouts: per-context latent (between) + per-rollout noise (within),
    # noise DOMINATING the tail dims so their ICC collapses toward 0.
    ctx_latent = rng.standard_normal((C_, H)) * scale
    vx = np.zeros((C_, K, H), dtype=np.float32)
    noise_scale = np.concatenate([np.array([0.5, 0.5, 0.5, 0.5]), 3.0 * np.ones(H - 4)])
    for c in range(C_):
        vx[c] = (ctx_latent[c][None, :] + rng.standard_normal((K, H)) * noise_scale).astype(
            np.float32
        )
    valid = np.ones((C_, K), dtype=bool)
    valid[0, 0] = False  # exercise the invalid-rollout path

    rb = {
        t: {
            "u": (u := rng.standard_normal(H)) / np.linalg.norm(u),
            "equivalent_variance_rank": 2 + i,
        }
        for i, t in enumerate(TRAITS)
    }
    out = _compute(vx, valid, pca, rb)
    rel = np.array(out["reliability_by_rank"])
    print("smoke reliability (leading ranks):", np.round(rel[:6], 3))
    assert np.nanmean(rel[:4]) > 0.5, ("leading-dir ICC should be high", rel[:4])
    assert np.nanmean(rel[-6:]) < 0.3, ("tail-dir ICC should collapse", rel[-6:])
    assert out["n_contexts_used"] == C_, out["n_contexts_used"]

    # figure smoke: synthetic committed-style curve (R2 below the ICC ceiling).
    fig_dir.mkdir(parents=True, exist_ok=True)
    committed = {
        "layer": 19,
        "ranks_evaluated": out["ranks_evaluated"],
        "r2_by_rank": [max(0.0, float(r) - 0.15) for r in rel],
        "r_b_by_trait": {t: {"heldout_r2": 0.4} for t in TRAITS},
    }
    tmp = fig_dir / "_smoke_committed.json"
    tmp.write_text(json.dumps(committed))
    _build_figure(out, tmp, fig_dir)
    print("SMOKE PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 per-direction reliability (ICC).")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--k-lead", type=int, default=200)
    ap.add_argument("--tail-step", type=int, default=20)
    ap.add_argument("--min-valid", type=int, default=2)
    ap.add_argument("--pass-b", type=Path, default=F.PASS_B_PATH)
    ap.add_argument("--rb-dir", type=Path, default=Path("data/issue_779/r_b"))
    ap.add_argument(
        "--reliability-bundle",
        type=Path,
        default=Path("data/issue_779/reliability_multirollout/reliability_multirollout.pt"),
    )
    ap.add_argument(
        "--committed-json", type=Path, default=F.DEFAULT_OUT_DIR / "perdirection_single_layer.json"
    )
    ap.add_argument(
        "--out-json", type=Path, default=F.DEFAULT_OUT_DIR / "reliability_by_direction.json"
    )
    ap.add_argument("--fig-dir", type=Path, default=F.DEFAULT_FIG_DIR)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(args.n_threads))

    if args.smoke:
        return _smoke(Path("/tmp/issue779-smokeB-analysis"))

    # answer-PCA basis: fold-0 TRAIN targets of pass_b v_x@layer (reused from the
    # per-predictor script so the reliability ranks align 1:1 with the R2 curve).
    import issue779_percontext_recon as PR

    bundle = F.load_pass_b(args.pass_b)
    li = bundle["layers"].index(args.layer)
    Y = bundle["v_x"][:, li, :].to(torch.float32).numpy()
    n = Y.shape[0]
    test_idx = PR._cv_folds(n, args.n_folds, args.seed)[0]
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    tr = np.where(mask)[0]
    pca = PP._pca_basis(Y[tr], args.k_lead, args.tail_step)
    rb_by_trait = {
        t: S1._load_rb(args.rb_dir, t, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)[li] for t in TRAITS
    }
    rb = PP._rb_ranks(pca, rb_by_trait)

    rbundle = torch.load(args.reliability_bundle, weights_only=False, map_location="cpu")
    rli = rbundle["layers"].index(args.layer)
    vx19 = rbundle["v_x"][:, :, rli, :].to(torch.float32).numpy()  # (C, K, H)
    valid = rbundle["valid"].numpy()  # (C, K)
    assert vx19.shape[0] == valid.shape[0] and vx19.shape[1] == valid.shape[1], (
        vx19.shape,
        valid.shape,
    )

    out = _compute(vx19, valid, pca, rb)
    out.update(
        {
            "layer": args.layer,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "fold": 0,
            "k_lead": args.k_lead,
            "tail_step": args.tail_step,
            "n_reliability_contexts": int(vx19.shape[0]),
            "k_rollouts": int(vx19.shape[1]),
            "note": (
                f"Single-rollout reliability (ICC) per answer-PCA direction at L{args.layer} on "
                "the fold-0 train-fit basis (ranks aligned to h_perdirection_r2_single_layer). "
                "ICC = between-context var(per-context rollout means) / (between + mean "
                "within-context rollout var). icc_adj subtracts within/Kbar (per-context-mean "
                "inflation)."
            ),
            "metadata": C.reproducibility_metadata({"script": "issue779_reliability_analysis"}),
        }
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_json, out)
    _build_figure(out, args.committed_json, args.fig_dir)

    rel = np.array(out["reliability_by_rank"])
    print(f"wrote {args.out_json}")
    print(
        f"reliability rank0={rel[0]:.3f} rank9~={rel[min(9, len(rel) - 1)]:.3f} "
        f"tail={np.nanmean(rel[-20:]):.3f} | n_ctx={out['n_contexts_used']} "
        f"Kbar={out['mean_valid_rollouts_per_context']:.1f}"
    )
    for t in TRAITS:
        rbr = out["r_b_reliability"][t]
        print(
            f"  r_B {t}: reliability {rbr['reliability_icc']:.3f} "
            f"(rank {rbr['equivalent_variance_rank']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
