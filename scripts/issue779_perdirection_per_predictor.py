"""Issue #779 inline free-analysis — per-PREDICTOR per-direction R2 at L19.

Extends the committed ``h_perdirection_r2_single_layer`` plot (per-direction
held-out R2 of the GCV-ridge context->answer map vs answer-PCA variance rank) to
FOUR predictors — GCV ridge, Nystrom RBF KRR, a full-dim MLP, and residual-skip
(ridge + MLP-on-residual) — on the SAME fold-0 answer-PCA basis, to answer ONE
question:

    Do the nonlinear / kernel fitters LIFT the mid-rank directions (~20-200) over
    the linear ridge, or do all four curves COINCIDE (the linear map already
    captures every learnable answer-PCA direction)?

Protocol matches the committed single-layer plot exactly: L19 (reconstruction-best
layer), fold 0 of 5-fold (seed 0), k_lead=200 lead ranks + tail every 20, 50
random reference directions, all three traits' r_B overlaid. The answer-PCA basis
is fit ONCE on the fold-0 TRAIN targets (4000 rows, train-mean-centered); every
predictor's (n_test x 3584) held-out prediction matrix is projected onto that ONE
basis, so the four per-direction R2 curves share a single variance-rank axis and
are directly comparable.

Fitters are REUSED from issue779_fitter_fair_comparison, never reimplemented:
  * ridge          = PR._ridge_fit_predict_fast (GCV, FULL 4000-row train) — the
                     SAME fit as the committed single-layer plot; asserted to
                     reproduce its r2_by_rank within tolerance.
  * krr            = F.krr_select_predict (Nystrom RBF, (gamma,lambda) on val).
  * mlp            = F.run_mlp_battery at the FFC-D1 val-selected recipe
                     (width/lr read from fair_comparison.json — no re-selection).
  * residual_skip  = F.gram_fit_apply (val-lambda base ridge) + MLP on residual.

Split: fold 0 gives test=1000 / train=4000. Ridge uses the full 4000-row train
with GCV (n_train=4000 keeps GCV sane, matching the committed plot). KRR / MLP /
residual-skip carve a 400-row val slice from the train fold (fit on the remaining
3600, select hyperparameters on val) — the FFC-D1 split shape, because GCV
degenerates at n_train~=H and val-lambda is required for the p~=n ridge inside
residual-skip. The 4000-row PCA basis + r_B equivalent-variance ranks are the
committed single-layer values (asserted).

0-GPU, CPU/VM only (thread-capped). Fail loud; NaN reported, never coerced.
Checkpoint-per-predictor (--out-json merges; --resume skips completed predictors).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import issue779_common as C
import issue779_fitter_fair_comparison as F
import issue779_identity_baseline as IB
import issue779_percontext_recon as PR
import issue779_stage1 as S1
import numpy as np
import torch

from explore_persona_space.orchestrate.env import load_dotenv

TRAITS = ("evil", "sycophancy", "hallucination")
PREDICTORS = ("ridge", "krr", "mlp", "residual_skip")
PREDICTOR_LABEL = {
    "ridge": "GCV ridge (linear)",
    "krr": "Nystrom RBF KRR",
    "mlp": "full-dim MLP",
    "residual_skip": "residual-skip (ridge + MLP)",
}
PREDICTOR_COLOR = {
    "ridge": "#1f4e9c",
    "krr": "#2ca02c",
    "mlp": "#d62728",
    "residual_skip": "#9467bd",
}
STAR_COLORS = {"evil": "#d62728", "sycophancy": "#e6550d", "hallucination": "#b5179e"}


def _pca_basis(Ytr: np.ndarray, k_lead: int, tail_step: int) -> dict:
    """Answer-PCA of the train-fold targets (centered on train mean). Identical to
    issue779_identity_baseline.analysis_d_layer: SVD of the centered targets, the
    same lead+tail rank ladder, per-direction train variance."""
    n_tr = Ytr.shape[0]
    Ytr_c = Ytr - Ytr.mean(0)
    _u, s, vh = torch.linalg.svd(torch.as_tensor(Ytr_c, dtype=torch.float64), full_matrices=False)
    vh_np = vh.numpy()  # (D, H): row k = PCA direction u_k
    var_spectrum = (s.numpy() ** 2) / (n_tr - 1)  # per-direction train variance
    total_var = float(var_spectrum.sum())
    d_full = vh_np.shape[0]
    ranks = list(range(min(k_lead, d_full))) + list(range(k_lead, d_full, tail_step))
    dirs = vh_np[ranks].T  # (H, n_sel)
    var_by_rank = var_spectrum[ranks]
    return {
        "Ytr_c": Ytr_c,
        "var_spectrum": var_spectrum,
        "total_var": total_var,
        "ranks": ranks,
        "dirs": dirs,
        "variance_share_by_rank": (var_by_rank / total_var),
    }


def _rb_ranks(pca: dict, rb_by_trait: dict[str, np.ndarray]) -> dict:
    """Per-trait r_B: unit direction + its equivalent variance rank in the answer-PCA
    spectrum (var_k ~ var_rb). Predictor-independent; matches the committed plot."""
    out = {}
    for t, rb_l in rb_by_trait.items():
        u = rb_l / (np.linalg.norm(rb_l) + 1e-12)
        var_rb = float(np.var(pca["Ytr_c"] @ u, ddof=1))
        rank = int(np.sum(pca["var_spectrum"] > var_rb))
        out[t] = {"u": u, "equivalent_variance_rank": rank, "train_variance": var_rb}
    return out


def _predictor_curve(
    pred_te: np.ndarray, Yte: np.ndarray, pca: dict, rb: dict, *, n_random: int, seed: int
) -> dict:
    """Per-direction held-out R2 of one predictor on the shared PCA basis + r_B + a
    matched random-direction band."""
    r2_by_rank = IB._per_direction_r2(Yte, pred_te, pca["dirs"])
    whole = PR._pooled_r2(pred_te, Yte)
    rng = np.random.default_rng(seed + 779)  # same rng convention as analysis_d_layer
    rand = rng.standard_normal((pred_te.shape[1], n_random))
    rand /= np.linalg.norm(rand, axis=0, keepdims=True) + 1e-12
    r2_rand = IB._per_direction_r2(Yte, pred_te, rand)
    rb_out = {}
    for t, info in rb.items():
        r2 = float(IB._per_direction_r2(Yte, pred_te, info["u"][:, None])[0])
        rb_out[t] = {"heldout_r2": r2, "equivalent_variance_rank": info["equivalent_variance_rank"]}
    finite = np.isfinite(r2_by_rank)
    return {
        "r2_by_rank": [float(x) for x in r2_by_rank],
        "whole_map_r2": float(whole),
        "random_directions": {
            "n": int(n_random),
            "r2_mean": float(np.nanmean(r2_rand)),
            "r2_sd": float(np.nanstd(r2_rand)),
        },
        "r_b": rb_out,
        "n_evaluated": int(finite.sum()),
        "n_r2_below_zero": int((r2_by_rank[finite] < 0.0).sum()),
    }


def _fit_predictor(
    name: str, X, Y, tr, fit_idx, val_idx, test_idx, *, mlp_width, mlp_lr, mlp_max_epochs, seed
) -> tuple[np.ndarray, dict]:
    """Return (pred_te (n_test, H), meta) for one predictor. Reuses F.* fitters."""
    dev = torch.device("cpu")
    if name == "ridge":
        pred = PR._ridge_fit_predict_fast(X[tr], Y[tr], X[test_idx])
        return pred, {"n_train": len(tr), "selection": "GCV, full train fold"}
    if name == "krr":
        k = F.krr_select_predict(
            X[fit_idx],
            Y[fit_idx],
            X[val_idx],
            Y[val_idx],
            X[test_idx],
            gamma_mult=F.KRR_GAMMA_MULT,
            lambdas=F.KRR_LAMBDAS,
            m_landmarks=F.KRR_LANDMARKS,
            seed=seed,
            dev=dev,
        )
        return k["pred_te"], {"n_fit": len(fit_idx), "selected": k["selected"]}
    if name == "mlp":
        fit = F.run_mlp_battery(
            [F.MLPGroup(("m",), X[fit_idx], Y[fit_idx], mlp_width, mlp_lr)],
            dev=dev,
            max_epochs=mlp_max_epochs,
        )[("m",)]
        return fit.predict(X[test_idx]), {
            "n_fit": len(fit_idx),
            "width": int(mlp_width),
            "lr": float(mlp_lr),
            "epochs_ran": int(fit.epochs_ran),
        }
    if name == "residual_skip":
        (rt_tr, rt_te), rlam = F.gram_fit_apply(
            X[fit_idx], Y[fit_idx], [X[fit_idx], X[test_idx]], dev, val=(X[val_idx], Y[val_idx])
        )
        fit = F.run_mlp_battery(
            [
                F.MLPGroup(
                    ("r",),
                    X[fit_idx],
                    (Y[fit_idx] - rt_tr).astype(np.float32),
                    F.RESIDUAL_MLP_WIDTH,
                    mlp_lr,
                )
            ],
            dev=dev,
            max_epochs=mlp_max_epochs,
        )[("r",)]
        pred = rt_te + fit.predict(X[test_idx])
        return pred, {
            "n_fit": len(fit_idx),
            "base_ridge_lambda": float(rlam),
            "residual_mlp_width": int(F.RESIDUAL_MLP_WIDTH),
            "lr": float(mlp_lr),
            "epochs_ran": int(fit.epochs_ran),
        }
    raise ValueError(name)


def _mlp_recipe(fair_json: Path) -> dict:
    """FFC-D1 val-selected MLP recipe (last input, L19). Reused, not re-selected."""
    d = json.loads(fair_json.read_text())
    sel = d["mlp_selection"]["per_input"]["last"]
    return {
        "width": int(sel["width"]),
        "lr": float(sel["lr"]),
        "source": f"{fair_json.name} mlp_selection.per_input.last (L{d['mlp_selection']['layer']})",
    }


def _assert_ridge_reproduces_committed(ridge_r2: list[float], committed_json: Path) -> dict:
    """The ridge curve must reproduce the committed single-layer r2_by_rank (same
    fold, same PCA basis, same fitter) — a self-consistency gate on the shared basis."""
    if not committed_json.exists():
        return {"checked": False, "reason": f"{committed_json} absent"}
    ref = json.loads(committed_json.read_text())["r2_by_rank"]
    assert len(ref) == len(ridge_r2), (len(ref), len(ridge_r2))
    a = np.asarray(ref, float)
    b = np.asarray(ridge_r2, float)
    m = np.isfinite(a) & np.isfinite(b)
    max_abs = float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("nan")
    ok = max_abs < 1e-6
    assert ok, f"ridge curve diverges from committed single-layer plot: max_abs_diff {max_abs:.3e}"
    return {"checked": True, "max_abs_diff": max_abs, "assert_pass": ok}


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Issue #779 per-predictor per-direction R2 at one layer."
    )
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--k-lead", type=int, default=200)
    ap.add_argument("--tail-step", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--n-val", type=int, default=400, help="val slice carved from the train fold")
    ap.add_argument("--mlp-max-epochs", type=int, default=F.MLP_MAX_EPOCHS)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--predictors", default=",".join(PREDICTORS))
    ap.add_argument("--pass-b", type=Path, default=F.PASS_B_PATH)
    ap.add_argument("--rb-dir", type=Path, default=Path("data/issue_779/r_b"))
    ap.add_argument("--fair-json", type=Path, default=F.DEFAULT_OUT_DIR / "fair_comparison.json")
    ap.add_argument(
        "--committed-json", type=Path, default=F.DEFAULT_OUT_DIR / "perdirection_single_layer.json"
    )
    ap.add_argument(
        "--out-json", type=Path, default=F.DEFAULT_OUT_DIR / "perdirection_per_predictor.json"
    )
    ap.add_argument("--fig-dir", type=Path, default=F.DEFAULT_FIG_DIR)
    ap.add_argument("--max-contexts", type=int, default=0, help="smoke: cap corpus (0 = all)")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(args.n_threads))
    want = [p.strip() for p in args.predictors.split(",") if p.strip()]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.out_json.read_text()) if args.out_json.exists() else {}

    if args.figures_only:
        _build_figure(results, args.fig_dir)
        return 0

    bundle = F.load_pass_b(args.pass_b)
    li = bundle["layers"].index(args.layer)
    X = bundle["cx_last"][:, li, :].to(torch.float32).numpy()
    Y = bundle["v_x"][:, li, :].to(torch.float32).numpy()
    n = X.shape[0]
    if args.max_contexts:
        n = min(n, args.max_contexts)
        X, Y = X[:n], Y[:n]

    test_idx = PR._cv_folds(n, args.n_folds, args.seed)[0]
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    tr = np.where(mask)[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(tr)
    n_val = min(args.n_val, len(tr) // 2)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]

    pca = _pca_basis(Y[tr], args.k_lead, args.tail_step)
    rb_by_trait = {
        t: S1._load_rb(args.rb_dir, t, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)[li] for t in TRAITS
    }
    rb = _rb_ranks(pca, rb_by_trait)
    recipe = _mlp_recipe(args.fair_json)

    results.setdefault("per_predictor", {})
    results.update(
        {
            "layer": args.layer,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "fold": 0,
            "k_lead": args.k_lead,
            "tail_step": args.tail_step,
            "n_random": args.n_random,
            "ranks_evaluated": [int(r) for r in pca["ranks"]],
            "variance_share_by_rank": [float(v) for v in pca["variance_share_by_rank"]],
            "split": {
                "n_contexts": int(n),
                "n_test": len(test_idx),
                "n_train_full": len(tr),
                "n_fit": len(fit_idx),
                "n_val": len(val_idx),
            },
            "mlp_recipe": recipe,
            "r_b_equivalent_variance_rank": {t: rb[t]["equivalent_variance_rank"] for t in TRAITS},
            "note": (
                f"Per-direction held-out R2 of four predictors on ONE shared fold-0 answer-PCA "
                f"basis (L{args.layer}, train-fold-fit). Ridge=GCV on full {len(tr)} train "
                f"(=committed single-layer curve); KRR/MLP/residual fit on {len(fit_idx)}, "
                f"select on {len(val_idx)}-row val. MLP recipe reused from FFC D1."
            ),
            "metadata": C.reproducibility_metadata(
                {"script": "issue779_perdirection_per_predictor"}
            ),
        }
    )

    for name in want:
        if name not in PREDICTORS:
            raise ValueError(f"unknown predictor {name!r}")
        if args.resume and name in results["per_predictor"]:
            print(f"[resume] {name} present; skip")
            continue
        print(f"[fit] {name} ...", flush=True)
        pred_te, meta = _fit_predictor(
            name,
            X,
            Y,
            tr,
            fit_idx,
            val_idx,
            test_idx,
            mlp_width=recipe["width"],
            mlp_lr=recipe["lr"],
            mlp_max_epochs=args.mlp_max_epochs,
            seed=args.seed,
        )
        curve = _predictor_curve(
            pred_te, Y[test_idx], pca, rb, n_random=args.n_random, seed=args.seed
        )
        curve["fit_meta"] = meta
        if name == "ridge" and not args.max_contexts:
            curve["reproduces_committed_single_layer"] = _assert_ridge_reproduces_committed(
                curve["r2_by_rank"], args.committed_json
            )
        results["per_predictor"][name] = curve
        C.write_json_atomic(args.out_json, results)
        rbd = ", ".join(f"{t} R2={curve['r_b'][t]['heldout_r2']:.3f}" for t in TRAITS)
        print(
            f"[done] {name}: whole-map R2={curve['whole_map_r2']:.4f} | "
            f"rank0={curve['r2_by_rank'][0]:.3f} last={curve['r2_by_rank'][-1]:.3f} | "
            f"random {curve['random_directions']['r2_mean']:.3f} | r_B {rbd}",
            flush=True,
        )

    _build_figure(results, args.fig_dir)
    (args.out_json.with_suffix(".done")).write_text("ok\n")
    print(f"wrote {args.out_json}")
    return 0


def _build_figure(results: dict, fig_dir: Path) -> None:
    if "per_predictor" not in results or not results["per_predictor"]:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    ranks = np.array(results["ranks_evaluated"], float) + 1  # 1-based for log axis
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for name in PREDICTORS:
        cur = results["per_predictor"].get(name)
        if not cur:
            continue
        r2 = np.array(cur["r2_by_rank"], float)
        ax.plot(
            ranks,
            r2,
            "-",
            color=PREDICTOR_COLOR[name],
            lw=1.3,
            label=f"{PREDICTOR_LABEL[name]} (whole-map R2={cur['whole_map_r2']:.3f})",
        )
    ridge = results["per_predictor"].get("ridge")
    if ridge:
        rd = ridge["random_directions"]
        ax.axhspan(
            rd["r2_mean"] - rd["r2_sd"],
            rd["r2_mean"] + rd["r2_sd"],
            color="0.6",
            alpha=0.20,
            label="random dirs (ridge, n=50, mean±sd)",
        )
        ax.axhline(rd["r2_mean"], color="0.55", ls=":", lw=1.0)
    ax.axhline(0.0, color="black", lw=1.0)
    # r_B markers at their equivalent variance rank (ridge predictor's read).
    if ridge:
        for t in TRAITS:
            info = ridge["r_b"][t]
            ax.scatter(
                [info["equivalent_variance_rank"] + 1],
                [info["heldout_r2"]],
                marker="*",
                s=210,
                zorder=6,
                color=STAR_COLORS[t],
                edgecolor="white",
                linewidth=0.6,
                label=f"r_B {t} ridge (rank {info['equivalent_variance_rank']}, "
                f"R2={info['heldout_r2']:.2f})",
            )
    ax.set_xscale("log")
    ax.set_xlabel("answer-PCA variance rank k (1-based, log)")
    ax.set_ylabel("held-out per-direction R2")
    ax.set_title(
        f"Per-direction predictability by fitter at L{results.get('layer', 19)} "
        "— do KRR/MLP lift the mid-ranks over ridge?"
    )
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=1)
    figs = savefig_paper(fig, "h_perdirection_r2_per_predictor", dir=fig_dir, embed_data=False)
    plt.close(fig)
    print(f"wrote {figs.get('png')}")


if __name__ == "__main__":
    raise SystemExit(main())
