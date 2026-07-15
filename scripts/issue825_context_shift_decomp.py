#!/usr/bin/env python
"""#825 free-analysis: does the base->instruct CONTEXT representation shift MORE
than the context->answer map requires?

Decompose the base->instruct context shift into the MAP-RELEVANT subspace (the
context-input directions the answer-map reads = left singular vectors of the
instruct ridge operator W_inst) vs the MAP-NULL subspace (its complement), and
measure how much of the shift lives in each.

Reuses issue825_map_alignment.py verbatim for the HF stage + load + fold recipe +
the GCV Gram-ridge core + the raw ridge operator + the random-orthogonal null.
Ridge/SVD closed-form only (NO MLP secondary). Runs on the VM CPU, 0 GPU-h.

Key objects (per frozen layer; headline 19):
  X_base, X_inst : n x d context vectors (teacher-forced, same text, per model)
  Y_inst         : n x d instruct answer vectors
  W_inst         : d x d ridge operator  dY ~ dX @ W_inst   (map the instruct
                   model uses); U = left singular vectors of W_inst = context
                   directions ranked by how much they drive the answer.
  P_rel(r) = span(U[:, :r]) ; P_null(r) = span(U[:, r:])
Reads:
  1. full-space A_ctx held-out R^2 (base->instruct context) -- self-check vs the
     committed 0.62 headline.
  2. shift-energy split: fraction of the (held-out) alignment residual energy in
     P_rel vs P_null, swept over rank cuts, vs a random-r-dim-subspace null band
     (expected r/d). residual concentrating in P_null BELOW the random band in
     P_rel == "context shifted MORE than the map needs, in answer-irrelevant
     directions".
  3. (L19 only) subspace alignment R^2 rel vs null: how linearly-relatable the
     clouds are WITHIN each subspace (+ variance weight of each subspace).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
from pathlib import Path

import issue825_map_alignment as ma
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "eval_results" / "issue_825" / "context_shift_decomp"
FIG_DIR = REPO / "figures" / "issue_825"
DL_DIR = REPO / "data" / "issue_825" / "context_shift_decomp_dl"

N_NULL_DRAWS = 30  # random-subspace draws (Haar orthogonal, reused across layers/cuts)
ENERGY_CUTS = [0.50, 0.90, 0.95, 0.99]
FIXED_RANKS = [200]  # #779 answer-variance rank


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as e:
        return f"unresolved: {e}"


def _reused_sha() -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO),
                "log",
                "-1",
                "--format=%H",
                "--",
                "scripts/issue825_map_alignment.py",
            ],
            text=True,
        ).strip()
    except Exception as e:
        return f"unresolved: {e}"


def _operator_svd(Xi: torch.Tensor, Yi: torch.Tensor):
    """W_inst (d x d) + its left singular vectors U (context-input directions)."""
    W, lam = ma._raw_ridge_operator(Xi, Yi)
    dev = Xi.device
    W = torch.as_tensor(W, dtype=torch.float64, device=dev)
    U, S, _ = torch.linalg.svd(W, full_matrices=False)  # U: (d, d) orthonormal columns
    return W, U, S, float(lam)


def _rank_cuts(S: torch.Tensor) -> dict:
    s2 = (S**2).cpu().numpy()
    cum = np.cumsum(s2) / (s2.sum() + 1e-30)
    d = len(s2)
    cuts = {}
    for frac in ENERGY_CUTS:
        r = int(np.searchsorted(cum, frac) + 1)
        cuts[f"e{int(frac * 100)}"] = min(max(r, 1), d)
    for r in FIXED_RANKS:
        cuts[f"r{r}"] = min(r, d)
    return cuts


def _heldout_residual(Xb: torch.Tensor, Xi: torch.Tensor, folds: np.ndarray):
    """Out-of-fold residual of the best linear base->instruct context alignment
    (GCV dual ridge, the parent A_ctx recipe). Returns (residual n x d, A_ctx R^2)."""
    resid = torch.empty_like(Xi)
    ss_res = ss_tot = 0.0
    for k in range(ma.N_FOLDS):
        te = folds == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            resid[torch.as_tensor(te)] = 0.0
            continue
        tr_t, te_t = torch.as_tensor(tr), torch.as_tensor(te)
        prep = ma._ridge_prep(Xb[tr_t])
        pred = ma._ridge_predict(prep, Xi[tr_t], Xb[te_t])
        true = Xi[te_t]
        resid[te_t] = true - pred
        ss_res += float(((true - pred) ** 2).sum())
        ss_tot += float(((true - true.mean(0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return resid, r2


def _subspace_align_r2(Xb, Xi, U, r, folds):
    """Held-out base->instruct alignment R^2 WITHIN P_rel (U[:, :r]) and P_null
    (U[:, r:]), plus each subspace's share of total instruct-context variance."""
    Ur, Un = U[:, :r], U[:, r:]
    out = {}
    Xi_c = Xi - Xi.mean(0)
    tot_var = float((Xi_c**2).sum()) + 1e-30
    for name, basis in (("rel", Ur), ("null", Un)):
        src = Xb @ basis  # n x k
        tgt = Xi @ basis
        r2 = ma._heldout_pooled_r2(
            tgt,
            folds,
            lambda tr, te, s=src, t=tgt: ma._ridge_predict(
                ma._ridge_prep(s[torch.as_tensor(tr)]),
                t[torch.as_tensor(tr)],
                s[torch.as_tensor(te)],
            ),
        )
        var_w = float(((Xi_c @ basis) ** 2).sum()) / tot_var
        out[name] = {"r2": r2, "var_weight": var_w, "rank": int(basis.shape[1])}
    return out


def _random_subspace_null(resid: torch.Tensor, ranks, *, n_draws, seed):
    """For each rank r, empirical distribution of ||resid @ Q_r||^2 / ||resid||^2
    over Haar-random r-dim subspaces Q_r (first r cols of a random orthogonal).
    One shared pool of random orthogonal matrices, sliced per rank."""
    dev = resid.device
    d = resid.shape[1]
    tot = float((resid**2).sum()) + 1e-30
    gen = torch.Generator().manual_seed(seed)
    per_rank = {int(r): [] for r in ranks}
    for _ in range(n_draws):
        Q = ma._random_orthogonal(d, gen).to(dev)  # d x d Haar orthogonal
        proj_sq = (resid @ Q) ** 2  # n x d, column j = energy along Q[:, j]
        colcum = torch.cumsum(proj_sq.sum(0), dim=0) / tot  # cumulative rel-energy in first-j cols
        for r in ranks:
            per_rank[int(r)].append(float(colcum[int(r) - 1]))
    band = {}
    for r, vals in per_rank.items():
        a = np.asarray(vals)
        band[r] = {
            "rel_energy_mean": float(a.mean()),
            "rel_energy_p2.5": float(np.quantile(a, 0.025)),
            "rel_energy_p97.5": float(np.quantile(a, 0.975)),
            "analytic_r_over_d": float(r) / float(d),
        }
    return band


def _shift_energy(
    resid: torch.Tensor, U: torch.Tensor, Xi: torch.Tensor, cuts: dict, null_band: dict
):
    """Fraction of residual energy in P_rel(r) vs P_null(r), plus the raw
    instruct-context variance fraction in P_rel, per rank cut."""
    tot_r = float((resid**2).sum()) + 1e-30
    Xi_c = Xi - Xi.mean(0)
    tot_v = float((Xi_c**2).sum()) + 1e-30
    projR = resid @ U  # n x d in U-coords
    projV = Xi_c @ U
    cumR = torch.cumsum((projR**2).sum(0), dim=0) / tot_r
    cumV = torch.cumsum((projV**2).sum(0), dim=0) / tot_v
    out = {}
    for name, r in cuts.items():
        rel = float(cumR[r - 1])
        var_rel = float(cumV[r - 1])
        nb = null_band[int(r)]
        out[name] = {
            "rank": int(r),
            "resid_energy_frac_rel": rel,
            "resid_energy_frac_null": 1.0 - rel,
            "raw_var_frac_rel": var_rel,
            "raw_var_frac_null": 1.0 - var_rel,
            "null_rel_energy_mean": nb["rel_energy_mean"],
            "null_rel_energy_p2.5": nb["rel_energy_p2.5"],
            "null_rel_energy_p97.5": nb["rel_energy_p97.5"],
            # <0 => residual AVOIDS map directions (concentrates in null beyond chance)
            "rel_vs_null_band_delta": rel - nb["rel_energy_mean"],
            "concentrates_in_null_beyond_chance": rel < nb["rel_energy_p2.5"],
        }
    return out


def run() -> dict:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    layers_subset = list(ma.FROZEN_LAYERS)
    Lh = ma.HEADLINE_LAYER

    from huggingface_hub import HfApi

    try:
        resolved = HfApi().repo_info(ma.HF_DATA_REPO, repo_type="dataset", revision=ma.HF_REV).sha
    except Exception as e:
        resolved = f"unresolved: {e}"

    npz_i = ma.cm.extract_stem(ma.STEM_INSTRUCT, DL_DIR)
    npz_b = ma.cm.extract_stem(ma.STEM_BASE, DL_DIR)
    data, conv, layers, al = ma._load_pair(npz_i, npz_b, layers_subset)
    folds = ma._cv_folds(conv, ma.N_FOLDS, ma.FIT_SEED)
    n = al["n_common"]
    print(f"[load] n_common={n} layers={layers} in {time.time() - t0:.1f}s", flush=True)

    per_layer = {}
    for L in layers:
        Xi, Yi, Xb = data["Xi"][L], data["Yi"][L], data["Xb"][L]
        d = Xi.shape[1]
        W, U, S, lam = _operator_svd(Xi, Yi)
        cuts = _rank_cuts(S)
        resid, actx_r2 = _heldout_residual(Xb, Xi, folds)
        ranks = sorted(set(cuts.values()))
        null_band = _random_subspace_null(resid, ranks, n_draws=N_NULL_DRAWS, seed=ma.FIT_SEED + 11)
        shift = _shift_energy(resid, U, Xi, cuts, null_band)
        rec = {
            "d": int(d),
            "operator_lambda": lam,
            "W_inst_spectrum": ma._spectrum(W),
            "rank_cuts": {k: int(v) for k, v in cuts.items()},
            "full_space_A_ctx_r2": actx_r2,
            "shift_energy": shift,
        }
        if Lh == L:
            rec["subspace_alignment_r2"] = {
                k: _subspace_align_r2(Xb, Xi, U, r, folds)
                for k, r in cuts.items()
                if k in ("e90", "r200")
            }
        per_layer[str(L)] = rec
        print(
            f"[layer {L}] A_ctx R2={actx_r2:.3f} r90={cuts.get('e90')} "
            f"e90 resid_null_frac={shift['e90']['resid_energy_frac_null']:.3f} "
            f"(null {shift['e90']['null_rel_energy_mean']:.3f} rel)",
            flush=True,
        )

    result = {
        "metadata": {
            "git_commit": _git_head(),
            "reused_from": f"scripts/issue825_map_alignment.py@{_reused_sha()}",
            "hf_repo": ma.HF_DATA_REPO,
            "hf_prefix": ma.HF_PREFIX,
            "hf_revision_pinned": ma.HF_REV,
            "hf_revision_resolved": resolved,
            "stems": [ma.STEM_INSTRUCT, ma.STEM_BASE],
            "role": ma.ROLE,
            "frozen_layers": list(ma.FROZEN_LAYERS),
            "headline_layer": Lh,
            "n_common": int(n),
            "n_folds": ma.N_FOLDS,
            "fit_seed": ma.FIT_SEED,
            "null_draws": N_NULL_DRAWS,
            "energy_cuts": ENERGY_CUTS,
            "fixed_ranks": FIXED_RANKS,
            "device": str(ma._fit_device()),
            "determinism_note": (
                "activations are teacher-forced deterministic (re-extraction is identical) "
                "=> measurement-noise ceiling ~1.0; R^2 gaps are genuine non-linear-relatability, "
                "not a noise floor."
            ),
            "wall_seconds": round(time.time() - t0, 1),
            "script": "scripts/issue825_context_shift_decomp.py",
        },
        "per_layer": per_layer,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2))
    print(
        f"[done] wrote {OUT_DIR / 'results.json'} in {result['metadata']['wall_seconds']}s",
        flush=True,
    )
    _make_figure(result)
    return result


def _make_figure(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in font_manager.findSystemFonts(fontpaths=None):
        if "Inter" in fp:
            with contextlib.suppress(Exception):
                font_manager.fontManager.addfont(fp)
    with contextlib.suppress(Exception):
        plt.rcParams["font.family"] = "Inter"
    CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    Lh = str(result["metadata"]["headline_layer"])
    pl = result["per_layer"][Lh]
    cut_order = ["e50", "e90", "e95", "e99", "r200"]
    cut_order = [c for c in cut_order if c in pl["shift_energy"]]
    labels = [f"{c}\n(r={pl['shift_energy'][c]['rank']})" for c in cut_order]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # Panel 1: shift-energy in P_rel vs random-subspace null band, L19
    ax = axes[0]
    x = np.arange(len(cut_order))
    rel = [pl["shift_energy"][c]["resid_energy_frac_rel"] for c in cut_order]
    nmean = [pl["shift_energy"][c]["null_rel_energy_mean"] for c in cut_order]
    nlo = [pl["shift_energy"][c]["null_rel_energy_p2.5"] for c in cut_order]
    nhi = [pl["shift_energy"][c]["null_rel_energy_p97.5"] for c in cut_order]
    ax.bar(x, rel, color=CB[0], width=0.55, label="residual energy in map-relevant subspace")
    ax.errorbar(
        x,
        nmean,
        yerr=[np.array(nmean) - np.array(nlo), np.array(nhi) - np.array(nmean)],
        fmt="D",
        color=CB[1],
        capsize=4,
        label="random-subspace null (95% band)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("fraction of shift residual energy")
    ax.set_xlabel("map-relevant subspace (energy cut / rank r)")
    ax.set_title(
        f"L{Lh}: context-shift energy in map-relevant dirs\n"
        "(below null band => shift avoids map, lives in null)"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 1)

    # Panel 2: subspace alignment R^2 rel vs null, L19
    ax = axes[1]
    sa = pl.get("subspace_alignment_r2", {})
    keys = [k for k in ("e90", "r200") if k in sa]
    xk = np.arange(len(keys))
    w = 0.38
    rel_r2 = [sa[k]["rel"]["r2"] for k in keys]
    null_r2 = [sa[k]["null"]["r2"] for k in keys]
    ax.bar(xk - w / 2, rel_r2, w, color=CB[0], label="P_rel (map-relevant)")
    ax.bar(xk + w / 2, null_r2, w, color=CB[2], label="P_null (map-ignored)")
    for i, k in enumerate(keys):
        ax.text(
            xk[i] - w / 2,
            rel_r2[i] + 0.01,
            f"vw={sa[k]['rel']['var_weight']:.2f}",
            ha="center",
            fontsize=7,
        )
        ax.text(
            xk[i] + w / 2,
            max(null_r2[i], 0) + 0.01,
            f"vw={sa[k]['null']['var_weight']:.2f}",
            ha="center",
            fontsize=7,
        )
    ax.axhline(
        pl["full_space_A_ctx_r2"],
        color="grey",
        ls="--",
        lw=1,
        label=f"full-space A_ctx R2={pl['full_space_A_ctx_r2']:.2f}",
    )
    ax.set_xticks(xk)
    ax.set_xticklabels([f"{k}\n(r={sa[k]['rel']['rank']})" for k in keys], fontsize=8)
    ax.set_ylabel("held-out base->instruct alignment R^2")
    ax.set_title(
        f"L{Lh}: within-subspace context alignment\n(vw = subspace share of instruct-ctx variance)"
    )
    ax.legend(fontsize=8, loc="lower left")

    # Panel 3: multi-layer summary — resid null-energy frac at e90 vs random baseline
    ax = axes[2]
    Ls = sorted(result["per_layer"].keys(), key=int)
    fr_null = [result["per_layer"][L]["shift_energy"]["e90"]["resid_energy_frac_null"] for L in Ls]
    base_null = [
        1.0 - result["per_layer"][L]["shift_energy"]["e90"]["null_rel_energy_mean"] for L in Ls
    ]
    xl = np.arange(len(Ls))
    ax.bar(xl - 0.2, fr_null, 0.4, color=CB[0], label="observed shift energy in P_null")
    ax.bar(xl + 0.2, base_null, 0.4, color=CB[1], label="random-subspace baseline")
    ax.set_xticks(xl)
    ax.set_xticklabels([f"L{L}" for L in Ls])
    ax.set_ylabel("fraction of shift residual energy in P_null (e90 cut)")
    ax.set_title(
        "Shift energy in map-null subspace by layer\n"
        "(observed > baseline => shift where map is blind)"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.02)

    fig.suptitle(
        "#825 context-shift subspace decomposition — base->instruct context reps "
        "shift beyond the map's reparameterization",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    png = FIG_DIR / "context_shift_decomp.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "figure": str(png.relative_to(REPO)),
        "source": "eval_results/issue_825/context_shift_decomp/results.json",
        "headline_layer": result["metadata"]["headline_layer"],
        "caption": (
            "Panel 1: fraction of the base->instruct context-shift residual energy in the "
            "map-relevant subspace (top-r left singular dirs of the instruct context->answer "
            "operator), vs a Haar-random r-dim-subspace null band, at layer 19 across energy cuts. "
            "Panel 2: held-out base->instruct context alignment R^2 within the map-relevant vs "
            "map-ignored subspace (vw = each subspace's share of instruct-context variance). "
            "Panel 3: fraction of shift energy in the map-null subspace across layers vs the "
            "random baseline. Deterministic teacher-forced activations (noise ceiling ~1.0)."
        ),
    }
    (FIG_DIR / "context_shift_decomp.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fig] wrote {png}", flush=True)


if __name__ == "__main__":
    run()
