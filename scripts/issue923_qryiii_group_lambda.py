#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #923 follow-up (qryiii-group-lambda-refit): leave-query-group-out λ selection.

The production run's masked-context query read (``arm_qry_iii``) collapsed to
pooled skill −3.03 on Betley while staying healthy (0.455) on UltraChat. The
clean-result's mechanism claim: masked-context features are near-duplicates
per query (the query tokens see only their own block, so rows for the same
query differ only through context-length-driven position shifts), so pointwise
PRESS-LOO λ selection — whose held-out row always has ~n_train_ctx near-twins
still in the train fold — favors tiny penalties that amplify within-query
variation out-of-fold. THE TEST: re-select λ with leave-query-group-out (LOGO)
CV inside each train fold (hold out ALL rows of one query at a time) and refit.
If Betley recovers without hand-picking a penalty, the mechanism is confirmed;
if it stays collapsed, the mechanism story is wrong.

Everything except the λ-selection rule is reused VERBATIM from
``issue923_fit_decomposition``: grid loaders, fold construction (7 LOFO
families × 4 stratified query folds), per-fold target PCA-48 + arm design
build (identical seeds ⇒ identical designs), the primal thin-SVD ``PressRidge``
engine, and the skill-over-mean R² DV. The LOGO mse is computed EXACTLY via
the block generalization of the PRESS identity — for group G,
``r̃_G = (I − H_GG)^{-1} r_G`` with ``H = U diag(φ_λ) Uᵀ`` — batched over
groups with one ``torch.linalg.solve`` per (λ, group-size bucket); no serial
per-group refits. Exactness is gated at startup (singleton groups ≡ pointwise
PRESS ≤1e-8; block identity vs an explicit leave-group-out refit ≤1e-8).

Scope (analysis-only, no training / generation): layer 18, arms
``arm_qry_iii`` + ``arm_concat_iii`` (reference), genres betley + uc (uc is
the healthy falsification control — LOGO should leave it ~unchanged). Three
rules per fold: (a) pointwise PRESS-LOO (reproduces the production numbers —
sanity-checked against decomposition_skill.json), (b) LOGO, (c) fixed λ=1e3
(the persisted per-λ diagnostic's recovery point).

Usage::

    uv run python scripts/issue923_qryiii_group_lambda.py --smoke   # synthetic grid
    uv run python scripts/issue923_qryiii_group_lambda.py \
      --packs-dir data/issue_923/hf_dl/issue923_ctx_query_decomposition/analysis_tensors/capture \
      --reduce-dir data/issue_923/hf_dl/issue923_ctx_query_decomposition/analysis_tensors/reduce
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402
from issue923_common import (  # noqa: E402
    DATA_DIR,
    HEADLINE_LAYER,
    SEED,
    dump_json,
    load_json,
)
from issue923_fit_decomposition import (  # noqa: E402
    PCA_DIM,
    PressRidge,
    build_folds,
    build_smoke_inputs,
    load_grids,
    press_fit_predict,
    run_selftest,
)

FIXED_LAMBDA = 1000.0
ARMS_DEFAULT = ("arm_qry_iii", "arm_concat_iii")


# ── exact leave-query-group-out PRESS (block identity, batched) ───────────────


def logo_press_mse(eng: PressRidge, Yc: torch.Tensor, groups: list[torch.Tensor]) -> torch.Tensor:
    """Exact leave-group-out CV mse per λ for one (design, fold) engine.

    ``Yc`` (m, P) CENTERED train targets (same fixed full-train centering the
    pointwise PRESS uses — the change is ONLY pointwise→group hold-out).
    ``groups`` partition [0, m) into row-index tensors (one per train query).
    Block PRESS identity per group G: the residual of a ridge fit trained
    WITHOUT G's rows, evaluated on G, is ``(I_g − H_GG)^{-1} r_G`` where
    ``H_GG = U_G diag(φ_λ) U_Gᵀ`` and ``r = Yc − H Yc`` are the in-sample
    residuals. Returns (n_lambda,) mean over the m rows AND P outputs —
    matching the pointwise ``PressRidge.press_mse`` convention. Batched over
    groups (bucketed by size) with one ``torch.linalg.solve`` per (λ, bucket).
    """
    U = eng.U
    m, P = Yc.shape
    assert m == eng.m, (m, eng.m)
    assert sum(len(g) for g in groups) == m, "groups must partition the train rows"
    G_coef = U.T @ Yc  # (k, P)
    by_size: dict[int, list[torch.Tensor]] = {}
    for g in groups:
        by_size.setdefault(len(g), []).append(g)
    buckets = [(sz, torch.stack(idx_list)) for sz, idx_list in by_size.items()]  # (nG, sz)
    mse = torch.empty(len(eng.lambdas), dtype=Yc.dtype)
    for li in range(len(eng.lambdas)):
        phi = eng.phi[li]  # (k,)
        R = Yc - U @ (phi.unsqueeze(1) * G_coef)  # (m, P) in-sample residuals
        total = 0.0
        for sz, idx in buckets:
            Ub = U[idx]  # (nG, sz, k)
            Hb = torch.bmm(Ub * phi, Ub.transpose(1, 2))  # (nG, sz, sz)
            A = torch.eye(sz, dtype=Yc.dtype).unsqueeze(0) - Hb
            X = torch.linalg.solve(A, R[idx])  # (nG, sz, P) LOGO residuals
            total += float((X * X).sum())
        mse[li] = total / (m * P)
    return mse


def run_logo_selftests() -> dict:
    """Exactness gates for the LOGO engine (fp64, ≤1e-8).

    (1) singleton groups reproduce the pointwise PRESS mse bit-for-bit;
    (2) the block identity matches an EXPLICIT leave-group-out refit
        (same fixed centering) on a synthetic problem.
    """
    torch.manual_seed(0)
    m, d, P, g = 40, 8, 5, 4
    Xn = torch.randn(m, d, dtype=torch.float64)
    Yc = torch.randn(m, P, dtype=torch.float64)
    Yc = Yc - Yc.mean(0, keepdim=True)
    eng = PressRidge(Xn)
    # (1) singleton groups == pointwise PRESS.
    singles = [torch.tensor([i]) for i in range(m)]
    mse_point, _ = eng.press_mse(Yc.unsqueeze(0))
    mse_logo1 = logo_press_mse(eng, Yc, singles)
    err1 = float((mse_point[0] - mse_logo1).abs().max())
    assert err1 <= 1e-8, f"singleton-LOGO != pointwise PRESS (max err {err1:.3e})"
    # (2) block identity == explicit leave-group-out refit.
    groups = [torch.arange(i, i + g) for i in range(0, m, g)]
    mse_logo = logo_press_mse(eng, Yc, groups)
    errs = []
    for li, lam in enumerate(eng.lambdas):
        total = 0.0
        for idx in groups:
            keep = torch.ones(m, dtype=torch.bool)
            keep[idx] = False
            Xk, Yk = Xn[keep], Yc[keep]
            beta = torch.linalg.solve(
                Xk.T @ Xk + lam * torch.eye(d, dtype=torch.float64), Xk.T @ Yk
            )
            resid = Yc[idx] - Xn[idx] @ beta
            total += float((resid * resid).sum())
        errs.append(abs(total / (m * P) - float(mse_logo[li])))
    err2 = max(errs)
    assert err2 <= 1e-8, f"block-LOGO identity != explicit refit (max err {err2:.3e})"
    return {"singleton_vs_pointwise_max_err": err1, "block_vs_explicit_refit_max_err": err2}


# ── per-genre refit under three λ rules (arms share per-fold heavy work) ──────


def refit_genre(
    genre: str,
    arms: list[str],
    layer: int,
    grid,
    fctx: torch.Tensor,
    fqry: dict,
    folds: dict,
) -> dict:
    """Primary-fold refit of ``arms`` for one genre at ``layer`` under 3 λ rules.

    Reproduces the production per-fold pipeline exactly (same seeds ⇒ same
    target PCA basis + arm designs), then selects λ three ways from the SAME
    per-λ test predictions: pointwise PRESS-LOO (production rule), LOGO, and
    fixed λ=1e3. The per-fold target PCA and each distinct PART design are
    computed ONCE and shared across arms — numerically identical to per-arm
    rebuilds because ``_pca_lowrank_project`` re-seeds
    ``torch.manual_seed(seed_fold)`` internally and the distinct-row PCA is
    deterministic (throughput-only caching; the parent's ``fit_layer_unit``
    shares the target PCA across arms the same way). Returns
    {arm: {pooled, pooled_skill_per_lambda, selected_lambda_counts, per_fold}}.
    """
    from issue923_fit_decomposition import ARM_PARTS, build_part

    nlam = len(RIDGE_LAMBDAS)
    fixed_idx = RIDGE_LAMBDAS.index(FIXED_LAMBDA)
    rules = ("loo", "logo", "fixed")
    acc = {arm: dict.fromkeys(rules, 0.0) for arm in arms}
    ss_tot_all = dict.fromkeys(arms, 0.0)
    ss_res_lambda = {arm: np.zeros(nlam) for arm in arms}
    per_fold: dict[str, list] = {arm: [] for arm in arms}
    for fold in folds["primary"]:
        tr, te = fold["train"], fold["test"]
        if len(tr) < 8 or len(te) < 2:
            continue
        seed_fold = SEED * 1000 + layer * 100 + fold["fold_id"]
        Ytr_amb = grid.target[tr, layer, :].double()
        Yte_amb = grid.target[te, layer, :].double()
        torch.manual_seed(seed_fold)
        mu_y = Ytr_amb.mean(0, keepdim=True)
        qdim = min(PCA_DIM, Ytr_amb.shape[0] - 1, Ytr_amb.shape[1])
        _u, _s, Vy = torch.pca_lowrank(Ytr_amb - mu_y, q=qdim, center=False, niter=2)
        Ytr = (Ytr_amb - mu_y) @ Vy
        Yte = (Yte_amb - mu_y) @ Vy
        ymu = Ytr.mean(0, keepdim=True)
        # LOGO groups: train cells sharing a query id (fold-level, arm-independent).
        q_of_tr = grid.q_of(tr)
        groups = [torch.from_numpy(np.where(q_of_tr == q)[0]).long() for q in np.unique(q_of_tr)]
        # One build_part call per DISTINCT part this fold (identical across arms).
        part_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for arm in arms:
            for part in ARM_PARTS[arm]:
                if part not in part_cache:
                    part_cache[part] = build_part(
                        part, layer, grid, tr, grid, te, fctx, fqry, seed_fold
                    )
        for arm in arms:
            Xtr = torch.cat([part_cache[p][0] for p in ARM_PARTS[arm]], dim=1)
            Xte = torch.cat([part_cache[p][1] for p in ARM_PARTS[arm]], dim=1)
            fit = press_fit_predict(Xtr, Ytr, Xte, return_engine=True, standardize=False)
            eng, _xtr_n, _xte_n = fit["engine"]
            mse_logo = logo_press_mse(eng, Ytr - ymu, groups)
            lam_idx = {
                "loo": int(fit["lam_idx"]),
                "logo": int(torch.argmin(mse_logo).item()),
                "fixed": fixed_idx,
            }
            ss_tot = float(((Yte - ymu) ** 2).sum())
            ss_tot_all[arm] += ss_tot
            res_lam = [float(((Yte - fit["per_lambda_pred"][li]) ** 2).sum()) for li in range(nlam)]
            ss_res_lambda[arm] += np.asarray(res_lam)
            frec = {
                "fold_id": fold["fold_id"],
                "family": fold["family"],
                "qfold": fold["qfold"],
                "n_train": len(tr),
                "n_test": len(te),
                "n_train_queries": len(groups),
                "ss_tot": ss_tot,
                "rules": {},
            }
            for r in rules:
                li = lam_idx[r]
                acc[arm][r] += res_lam[li]
                frec["rules"][r] = {
                    "lambda": RIDGE_LAMBDAS[li],
                    "ss_res": res_lam[li],
                    "fold_skill": 1.0 - res_lam[li] / ss_tot if ss_tot > 0 else float("nan"),
                }
            frec["logo_mse_per_lambda"] = [float(x) for x in mse_logo]
            frec["loo_mse_per_lambda"] = [float(x) for x in fit["mse"]]
            per_fold[arm].append(frec)
        print(
            f"[fold] {genre} fold={fold['fold_id']} ({fold['family']}/q{fold['qfold']}) done",
            flush=True,
        )
    out: dict = {}
    for arm in arms:
        tot = ss_tot_all[arm]
        out[arm] = {
            "pooled": {
                r: {
                    "skill": 1.0 - acc[arm][r] / tot if tot > 0 else float("nan"),
                    "ss_res": acc[arm][r],
                    "ss_tot": tot,
                }
                for r in rules
            },
            "pooled_skill_per_lambda": [
                1.0 - ss_res_lambda[arm][li] / tot if tot > 0 else float("nan")
                for li in range(nlam)
            ],
            "selected_lambda_counts": {
                r: {
                    str(RIDGE_LAMBDAS[li]): sum(
                        1 for f in per_fold[arm] if f["rules"][r]["lambda"] == RIDGE_LAMBDAS[li]
                    )
                    for li in range(nlam)
                }
                for r in ("loo", "logo")
            },
            "n_folds": len(per_fold[arm]),
            "per_fold": per_fold[arm],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #923 LOGO λ-selection refit (L18)")
    parser.add_argument(
        "--packs-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data/issue_923/hf_dl/issue923_ctx_query_decomposition/analysis_tensors/capture",
    )
    parser.add_argument(
        "--reduce-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data/issue_923/hf_dl/issue923_ctx_query_decomposition/analysis_tensors/reduce",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_923/fits/qryiii_group_lambda.json",
    )
    parser.add_argument(
        "--reference-skill-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_923/fits/decomposition_skill.json",
        help="production decomposition_skill.json for the reproduce-sanity check",
    )
    parser.add_argument("--genres", default="betley,uc")
    parser.add_argument("--arms", default=",".join(ARMS_DEFAULT))
    parser.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    parser.add_argument(
        "--sanity-tol",
        type=float,
        default=0.05,
        help="max |our pointwise-LOO pooled skill − persisted skill| before failing loud",
    )
    parser.add_argument("--smoke", action="store_true", help="synthetic grid; /tmp output")
    args = parser.parse_args()

    t0 = time.time()
    print("[phase=selftest]", flush=True)
    st = run_selftest(device="cpu")  # parent PRESS/dual exactness gate
    st_logo = run_logo_selftests()
    print(f"[selftest] PRESS/dual: {st} | LOGO: {st_logo}", flush=True)

    print("[phase=load]", flush=True)
    if args.smoke:
        out_path = Path("/tmp/issue-923-smoke/qryiii_group_lambda.json")
        grids, fctx, fqry, folds_payload = build_smoke_inputs(Path("/tmp/issue-923-smoke"))
        genres = ["uc"]
        layer = min(args.layer, fctx.shape[1] - 1)
        reference = None
    else:
        out_path = args.out
        genres = [g.strip() for g in args.genres.split(",") if g.strip()]
        grids, fctx, fqry, folds_payload, load_meta = load_grids(
            args.packs_dir, args.reduce_dir, args.data_dir, genres, ood=False
        )
        layer = args.layer
        assert load_meta["mask_backend"] != "dropped", "fqry_iii packs absent — cannot refit"
        reference = load_json(args.reference_skill_json) if args.reference_skill_json else None
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print("[phase=fits]", flush=True)
    # Per-genre checkpoint/resume (shared-VM kill insurance): partial payload
    # keyed on EVERY output-affecting regime key; a mismatch ignores the partial.
    regime = {
        "layer": layer,
        "genres": genres,
        "arms": arms,
        "lambdas": RIDGE_LAMBDAS,
        "fixed_lambda": FIXED_LAMBDA,
        "smoke": args.smoke,
        "packs_dir": str(args.packs_dir),
        "reduce_dir": str(args.reduce_dir),
    }
    partial_path = out_path.with_suffix(".partial.json")
    result_genres: dict = {}
    if partial_path.exists():
        prev = load_json(partial_path)
        if prev.get("regime") == regime:
            result_genres = prev["genres"]
            print(f"[resume] loaded partial genres: {sorted(result_genres)}", flush=True)
        else:
            print("[resume] partial regime mismatch — ignoring partial", flush=True)
    sanity: dict = {}
    for genre in genres:
        if genre not in result_genres:
            grid = grids[genre]
            folds = build_folds(grid, folds_payload["query_folds"][genre])
            result_genres[genre] = refit_genre(genre, arms, layer, grid, fctx, fqry, folds)
            dump_json({"regime": regime, "genres": result_genres}, partial_path)
        for arm in arms:
            res = result_genres[genre][arm]
            line = (
                f"[fits] {genre}/{arm} L{layer}: loo={res['pooled']['loo']['skill']:.4f} "
                f"logo={res['pooled']['logo']['skill']:.4f} "
                f"fixed(1e3)={res['pooled']['fixed']['skill']:.4f} "
                f"({time.time() - t0:.0f}s)"
            )
            print(line, flush=True)
            if reference is not None:
                ref_arm = reference["genres"][genre][str(layer)]["arms"][arm]
                d_loo = abs(res["pooled"]["loo"]["skill"] - ref_arm["skill"])
                d_fix = abs(
                    res["pooled"]["fixed"]["skill"]
                    - ref_arm["skill_per_lambda"][RIDGE_LAMBDAS.index(FIXED_LAMBDA)]
                )
                sanity[f"{genre}/{arm}"] = {
                    "persisted_skill_loo": ref_arm["skill"],
                    "our_skill_loo": res["pooled"]["loo"]["skill"],
                    "abs_delta_loo": d_loo,
                    "persisted_skill_lambda_1e3": ref_arm["skill_per_lambda"][
                        RIDGE_LAMBDAS.index(FIXED_LAMBDA)
                    ],
                    "our_skill_lambda_1e3": res["pooled"]["fixed"]["skill"],
                    "abs_delta_lambda_1e3": d_fix,
                }
                assert d_loo <= args.sanity_tol and d_fix <= args.sanity_tol, (
                    f"reproduce-sanity FAIL {genre}/{arm}: |Δloo|={d_loo:.4f} "
                    f"|Δ1e3|={d_fix:.4f} > tol {args.sanity_tol} — the refit pipeline "
                    "does not reproduce the production fit; do not trust the LOGO read"
                )

    meta = reproducibility_metadata(
        {
            "script": "issue923_qryiii_group_lambda",
            "followup_label": "qryiii-group-lambda-refit",
            "smoke": args.smoke,
        }
    )
    payload = {
        "meta": {
            **meta,
            "layer": layer,
            "genres": genres,
            "arms": arms,
            "lambdas": RIDGE_LAMBDAS,
            "fixed_lambda": FIXED_LAMBDA,
            "selection_rules": {
                "loo": "pointwise PRESS-LOO on the train fold (production rule)",
                "logo": "leave-query-group-out CV on the train fold (block PRESS identity)",
                "fixed": f"fixed λ={FIXED_LAMBDA:g} (persisted per-λ diagnostic recovery point)",
            },
            "selftests": {"press_dual": st, "logo": st_logo},
            "reproduce_sanity": sanity,
            "sanity_tol": args.sanity_tol,
            "inputs": {
                "packs_dir": str(args.packs_dir),
                "reduce_dir": str(args.reduce_dir),
                "hf_source": "superkaiba1/explore-persona-space-data/"
                "issue923_ctx_query_decomposition/analysis_tensors@main",
            },
            "wall_seconds": round(time.time() - t0, 1),
        },
        "genres": result_genres,
    }
    dump_json(payload, out_path)
    partial_path.unlink(missing_ok=True)
    print(f"[phase=done] wrote {out_path} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
