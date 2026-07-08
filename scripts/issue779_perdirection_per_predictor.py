"""Issue #779 inline free-analysis — per-PREDICTOR per-direction R2 at L19.

Extends the committed ``h_perdirection_r2_single_layer`` plot (per-direction
held-out R2 of the GCV-ridge context->answer map vs answer-PCA variance rank) to
FOUR predictors — GCV ridge, Nystrom RBF KRR, a full-dim MLP, and residual-skip
(ridge + MLP-on-residual) — on the SAME answer-PCA basis, to answer ONE question:

    Do the nonlinear / kernel fitters LIFT the mid-rank directions (~20-200) over
    the linear ridge, or do all four curves COINCIDE (the linear map already
    captures every learnable answer-PCA direction)?

Two CORPUS MODES (``--corpus-mode``):
  * single (default): the round-1 5000 pass_b contexts, fold 0 of 5-fold (seed 0),
    the SAME protocol as the committed single-layer plot. Ridge is GCV on the full
    4000-row train fold (matches the committed curve — asserted to 1e-6); KRR / MLP
    / residual carve a 400-row val slice from the train fold (fit on 3600).
  * n10k: the FFC round-2 COMBINED corpus (5000 pass_b + 6500 new = 11,500), using
    the byte-identical D1/D2 split (train 10,000 / val 400 / test 1000; read from
    the FFC's persisted ``n10k_split.json``, else rebuilt via ``build_n10k_split``
    with the byte-identical tripwire). All four predictors fit on the 10k train;
    the answer-PCA basis is fit on the n10k TRAIN fold; ridge uses val-selected
    lambda over the wider ``LAMBDAS_N10K`` grid (matching the FFC n10k D1 ridge,
    since the fixed split has an explicit val set). Output ->
    ``perdirection_per_predictor_n10k.json`` + ``h_perdirection_r2_per_predictor_n10k.png``.

The answer-PCA basis is fit ONCE on the mode's TRAIN fold (train-mean-centered);
every predictor's (n_test x 3584) held-out prediction matrix is projected onto
that ONE basis, so the four per-direction R2 curves share a single variance-rank
axis. Fitters are REUSED from issue779_fitter_fair_comparison, never
reimplemented (ridge PR._ridge_fit_predict_fast / F.gram_fit_apply, KRR
F.krr_select_predict, MLP F.run_mlp_battery, residual-skip base ridge + MLP), and
the MLP recipe (width/lr) is read from the mode's FFC-D1 fair_comparison.json.

0-GPU by nature (``--device cpu`` default; ``--device cuda`` accelerates the F.*
fitters on a pod). Fail loud; NaN reported, never coerced. Checkpoint-per-
predictor (--out-json merges; --resume skips completed predictors, guarded on
corpus_mode so a cross-regime resume never reuses the wrong rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_identity_baseline as IB  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

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


def _predictor_curve(pred_te, Yte, pca, rb, *, n_random, seed):
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


def _fit_predictor(name, X, Y, D, *, mlp_width, mlp_lr, mlp_max_epochs, seed, dev):
    """Return (pred_te (n_test, H), meta) for one predictor. Reuses F.* fitters.

    ``D`` carries the resolved index sets: ridge_train (ridge's fit set), fit_idx
    (KRR/MLP/residual fit set), val_idx, test_idx, and ridge_val_lambda (single ->
    GCV on ridge_train; n10k -> val-lambda over LAMBDAS_N10K)."""
    tr, fit_idx, val_idx, test_idx = D["ridge_train"], D["fit_idx"], D["val_idx"], D["test_idx"]
    if name == "ridge":
        if D["ridge_val_lambda"]:  # n10k: val-lambda on the full train (matches FFC n10k D1)
            (pred,), lam = F.gram_fit_apply(
                X[tr], Y[tr], [X[test_idx]], dev, val=(X[val_idx], Y[val_idx])
            )
            return pred, {
                "n_train": len(tr),
                "selection": "val-lambda",
                "selected_lambda": float(lam),
            }
        pred = PR._ridge_fit_predict_fast(X[tr], Y[tr], X[test_idx])  # single: GCV full train
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
    """FFC-D1 val-selected MLP recipe (last input, at the selection layer). Reused,
    not re-selected. Falls back to the FFC default (8192, 3e-4) if the fair
    comparison for this corpus mode has not been run."""
    if not fair_json.exists():
        return {
            "width": int(F.MLP_WIDTHS[-1]),
            "lr": float(F.MLP_LRS[-1]),
            "source": f"FALLBACK default (fair_comparison.json absent at {fair_json})",
        }
    d = json.loads(fair_json.read_text())
    sel = d["mlp_selection"]["per_input"]["last"]
    return {
        "width": int(sel["width"]),
        "lr": float(sel["lr"]),
        "source": f"{fair_json} mlp_selection.per_input.last (L{d['mlp_selection']['layer']})",
    }


def _n10k_split(args, combined) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The byte-identical n10k train/val/test. PREFER the FFC's persisted
    n10k_split.json (the exact split D1/D2 used); else rebuild deterministically
    (build_n10k_split's tripwire asserts byte-identical val/test in the prod regime)."""
    split_json = args.n10k_out_dir / "n10k_split.json"
    if split_json.exists() and not args.rebuild_n10k_split:
        d = json.loads(split_json.read_text())
        d["_split_source"] = f"read {split_json}"
        return (np.array(d["train_ids"]), np.array(d["val_ids"]), np.array(d["test_ids"]), d)
    ns = SimpleNamespace(
        n_train=args.n10k_train, n_val=args.n_val, n_test=args.n10k_test, seed=args.n10k_split_seed
    )
    train, val, test, diag = F.build_n10k_split(combined, ns)
    diag["_split_source"] = "rebuilt"
    args.n10k_out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(split_json, diag)  # persist rebuilt split for provenance/reuse
    return train, val, test, diag


def _resolve_data(args, dev) -> dict:
    """Layer arrays + resolved split + PCA basis + r_B + MLP recipe for the chosen
    corpus mode. single-mode index construction is byte-preserved from the
    committed run (the ridge reproduces-committed assert guards it)."""
    if args.corpus_mode == "single":
        bundle = F.load_pass_b(args.pass_b)
        layers = list(bundle["layers"])
        li = layers.index(args.layer)
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
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        split = {
            "n_contexts": int(n),
            "n_test": len(test_idx),
            "n_train_full": len(tr),
            "n_fit": len(fit_idx),
            "n_val": len(val_idx),
        }
        D = {
            "ridge_train": tr,
            "fit_idx": fit_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "pca_train_idx": tr,
            "ridge_val_lambda": False,
        }
        fair_json = args.fair_json or (F.DEFAULT_OUT_DIR / "fair_comparison.json")
    else:  # n10k
        F.LAMBDAS = F.LAMBDAS_N10K  # wider ridge grid; F.* ridge callers read the module global
        combined, _nb = F.load_combined_corpus(args.pass_b, args.new_bundle, args.n_pass_b)
        layers = list(combined["layers"])
        X = F.input_layer(combined, "last", args.layer)
        Y = F.target_vx(combined, args.layer)
        train, val, test, diag = _n10k_split(args, combined)
        assert set(train.tolist()).isdisjoint(val.tolist()) and set(train.tolist()).isdisjoint(
            test.tolist()
        ), "n10k train overlaps val/test"
        split = {
            "n_contexts": int(combined["n_old"] + combined["n_new"]),
            "n_old": int(combined["n_old"]),
            "n_new": int(combined["n_new"]),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "val_test_byte_identical_round1": diag.get("val_test_byte_identical_round1"),
            "val_sha256": diag.get("val_sha256"),
            "test_sha256": diag.get("test_sha256"),
            "train_sha256": diag.get("train_sha256"),
            "split_source": diag.get("_split_source", "unknown"),
        }
        D = {
            "ridge_train": train,
            "fit_idx": train,
            "val_idx": val,
            "test_idx": test,
            "pca_train_idx": train,
            "ridge_val_lambda": True,
        }
        fair_json = args.fair_json or (args.n10k_out_dir / "fair_comparison.json")

    rb_li = layers.index(args.layer)
    pca = _pca_basis(Y[D["pca_train_idx"]], args.k_lead, args.tail_step)
    rb_by_trait = {
        t: S1._load_rb(args.rb_dir, t, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)[rb_li] for t in TRAITS
    }
    rb = _rb_ranks(pca, rb_by_trait)
    recipe = _mlp_recipe(fair_json)
    return {"X": X, "Y": Y, "D": D, "pca": pca, "rb": rb, "recipe": recipe, "split": split}


def _assert_ridge_reproduces_committed(ridge_r2, committed_json: Path) -> dict:
    """single-mode only: the GCV-ridge curve must reproduce the committed
    single-layer r2_by_rank (same fold, same PCA basis, same fitter)."""
    if not committed_json.exists():
        return {"checked": False, "reason": f"{committed_json} absent"}
    ref = json.loads(committed_json.read_text())["r2_by_rank"]
    assert len(ref) == len(ridge_r2), (len(ref), len(ridge_r2))
    a, b = np.asarray(ref, float), np.asarray(ridge_r2, float)
    m = np.isfinite(a) & np.isfinite(b)
    max_abs = float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("nan")
    ok = max_abs < 1e-6
    assert ok, f"ridge curve diverges from committed single-layer plot: max_abs_diff {max_abs:.3e}"
    return {"checked": True, "max_abs_diff": max_abs, "assert_pass": ok}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #779 per-predictor per-direction R2 at one layer."
    )
    ap.add_argument("--corpus-mode", choices=["single", "n10k"], default="single")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--k-lead", type=int, default=200)
    ap.add_argument("--tail-step", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument(
        "--n-val", type=int, default=400, help="single: val carved from train; n10k: val size"
    )
    ap.add_argument("--mlp-max-epochs", type=int, default=F.MLP_MAX_EPOCHS)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--predictors", default=",".join(PREDICTORS))
    ap.add_argument("--pass-b", type=Path, default=F.PASS_B_PATH)
    ap.add_argument(
        "--new-bundle", type=Path, default=F.DEFAULT_NEW_BUNDLE, help="n10k new_context_vectors.pt"
    )
    ap.add_argument("--n-pass-b", type=int, default=F.N_PASS_B)
    ap.add_argument("--n10k-train", type=int, default=F.N10K_TRAIN)
    ap.add_argument("--n10k-test", type=int, default=1000)
    ap.add_argument("--n10k-split-seed", type=int, default=F.SPLIT_SEED)
    ap.add_argument(
        "--n10k-out-dir",
        type=Path,
        default=F.DEFAULT_OUT_DIR_N10K,
        help="where the FFC n10k n10k_split.json / fair_comparison.json live",
    )
    ap.add_argument(
        "--rebuild-n10k-split",
        action="store_true",
        help="rebuild the n10k split instead of reading the persisted n10k_split.json",
    )
    ap.add_argument("--rb-dir", type=Path, default=Path("data/issue_779/r_b"))
    ap.add_argument("--fair-json", type=Path, default=None, help="override MLP-recipe source")
    ap.add_argument(
        "--committed-json", type=Path, default=F.DEFAULT_OUT_DIR / "perdirection_single_layer.json"
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=F.DEFAULT_FIG_DIR)
    ap.add_argument(
        "--max-contexts", type=int, default=0, help="single-mode smoke: cap corpus (0 = all)"
    )
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(args.n_threads))
    dev = torch.device(args.device)
    want = [p.strip() for p in args.predictors.split(",") if p.strip()]
    n10k = args.corpus_mode == "n10k"
    fig_stem = "h_perdirection_r2_per_predictor" + ("_n10k" if n10k else "")
    if args.out_json is None:
        args.out_json = (
            (args.n10k_out_dir / "perdirection_per_predictor_n10k.json")
            if n10k
            else (F.DEFAULT_OUT_DIR / "perdirection_per_predictor.json")
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.out_json.read_text()) if args.out_json.exists() else {}
    if results.get("corpus_mode") and results["corpus_mode"] != args.corpus_mode:
        raise SystemExit(
            f"--out-json {args.out_json} was written under corpus_mode="
            f"{results['corpus_mode']!r} but --corpus-mode={args.corpus_mode!r}; refusing to "
            "reuse cross-regime rows (use a mode-specific --out-json)"
        )

    if args.figures_only:
        _build_figure(results, args.fig_dir, fig_stem)
        return 0

    data = _resolve_data(args, dev)
    X, Y, D, pca, rb, recipe = (
        data["X"],
        data["Y"],
        data["D"],
        data["pca"],
        data["rb"],
        data["recipe"],
    )
    results.setdefault("per_predictor", {})
    results.update(
        {
            "corpus_mode": args.corpus_mode,
            "layer": args.layer,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "k_lead": args.k_lead,
            "tail_step": args.tail_step,
            "n_random": args.n_random,
            "device": args.device,
            "ranks_evaluated": [int(r) for r in pca["ranks"]],
            "variance_share_by_rank": [float(v) for v in pca["variance_share_by_rank"]],
            "split": data["split"],
            "mlp_recipe": recipe,
            "r_b_equivalent_variance_rank": {t: rb[t]["equivalent_variance_rank"] for t in TRAITS},
            "note": (
                f"corpus_mode={args.corpus_mode}: per-direction held-out R2 of four predictors "
                f"on ONE shared answer-PCA basis (L{args.layer}, fit on "
                f"{len(D['pca_train_idx'])} train rows). "
                + (
                    "single: ridge=GCV full train fold (=committed single-layer curve); "
                    if not n10k
                    else "n10k: byte-identical D1/D2 split, ridge=val-lambda over LAMBDAS_N10K; "
                )
                + "KRR/MLP/residual fit + val-select; MLP recipe reused from FFC D1."
            ),
            "metadata": C.reproducibility_metadata(
                {"script": "issue779_perdirection_per_predictor", "corpus_mode": args.corpus_mode}
            ),
        }
    )

    for name in want:
        if name not in PREDICTORS:
            raise ValueError(f"unknown predictor {name!r}")
        if args.resume and name in results["per_predictor"]:
            print(f"[resume] {name} present; skip")
            continue
        print(f"[fit] {name} ({args.corpus_mode}) ...", flush=True)
        pred_te, meta = _fit_predictor(
            name,
            X,
            Y,
            D,
            mlp_width=recipe["width"],
            mlp_lr=recipe["lr"],
            mlp_max_epochs=args.mlp_max_epochs,
            seed=args.seed,
            dev=dev,
        )
        curve = _predictor_curve(
            pred_te, Y[D["test_idx"]], pca, rb, n_random=args.n_random, seed=args.seed
        )
        curve["fit_meta"] = meta
        if name == "ridge" and not n10k and not args.max_contexts:
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

    _build_figure(results, args.fig_dir, fig_stem)
    (args.out_json.with_suffix(".done")).write_text("ok\n")
    print(f"wrote {args.out_json}")
    return 0


def _build_figure(results: dict, fig_dir: Path, stem: str) -> None:
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
    mode = results.get("corpus_mode", "single")
    corpus = "n10k combined 11.5k" if mode == "n10k" else "5000 pass_b"
    ax.set_title(
        f"Per-direction predictability by fitter at L{results.get('layer', 19)} "
        f"({corpus}) — do KRR/MLP lift the mid-ranks over ridge?"
    )
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=1)
    figs = savefig_paper(fig, stem, dir=fig_dir, embed_data=False)
    plt.close(fig)
    print(f"wrote {figs.get('png')}")


if __name__ == "__main__":
    raise SystemExit(main())
