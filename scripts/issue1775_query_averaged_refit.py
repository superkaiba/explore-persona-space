#!/usr/bin/env python3
"""#1775 follow-up: inner-validation lambda refit of the query_averaged per-row baseline.

Interpretation v2 Result 3 flagged the run's query_averaged per-row LINEAR cell as
degenerate under PRESS lambda selection (pooled R2 ~= -8.1/-8.2 in every fold, both
bases): the leave-own-row-out prefix-mean input is near-constant within a prefix, so
PRESS (an in-sample LOO read over the TRAIN fold) selects lambda=10 while every
HELD-OUT per-lambda read wants the grid-max shrinkage (lambda=1000 -> R2 -0.24/-0.28,
``linear_fits.json`` per_lambda_r2). This refit replaces the lambda selection ONLY:

- OUTER folds: the run's 6 novel-prefix folds verbatim (``fold_pairs`` on the same
  17,308-row battery-excluded population, FOLD_SEED=0).
- INNER selection: each outer train fold splits BY PREFIX GROUP via the committed
  ``inner_val_split`` (frac=INNER_VAL_GROUP_FRAC=0.2, seed=1234+fold — the
  ``issue1775_bilinear`` r=0 GD convention); lambda maximizing inner-val R2 on the
  run's registered grid (RIDGE_LAMBDAS, re-exported by ``issue1775_common`` from
  ``issue658_fit_predictors`` via ``issue923_fit_decomposition``) is chosen per
  (fold, basis), then the model is refit on the FULL outer train at that lambda.
- FIT MATH: identical to ``press_fit_predict(standardize=True)`` — train-only mu/sd
  (ddof=0, +1e-9) + degenerate-dim drop, train-mean-centered targets, exact SVD
  dual-form predict ``Xte_n V diag(S/(S^2+lam)) U^T Yc + ymu`` — ONE shared SVD per
  (fold, leg) reused across both bases and all grid lambdas (``--selftest`` pins
  exact parity against ``press_fit_predict``'s per-lambda preds).
- GAINS: nonlinear LEVELS are read from the committed ``nonlinear_fits.json``
  (no new nonlinear fits); recomputed gain = level - healthy linear baseline, with
  paired cluster-bootstrap CIs over prefix groups (``cluster_bootstrap_delta_r2``)
  wherever the run's per-row nonlinear preds are persisted on the HF data repo.

Checkpoints per fold (npz + json under ``tensors_dir('qa_refit_work')``) with a
regime-keyed resume; final JSON at ``eval_results/issue_1775/ladder/query_averaged_refit.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: caps + .env bind BEFORE the heavy imports (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from issue1775_common import (  # noqa: E402
    CELL_PRIMARY,
    HF_DATA_REPO,
    INNER_VAL_GROUP_FRAC,
    LAYER_PRIMARY,
    OUT_HF_PREFIX,
    RIDGE_LAMBDAS,
    _basis_targets_with_info,
    _r2,
    atomic_write_json,
    build_arm_data,
    cluster_bootstrap_delta_r2,
    eval_dir,
    fold_pairs,
    inner_val_split,
    press_fit_predict,
    resolve_store_dir,
    restrict_pairs,
    result_meta,
    tensors_dir,
    upload_phase_tensors,
)


ARM = "query_averaged"
BASES = ("pca48", "ambient")
INNER_SEED_BASE = 1234  # issue1775_bilinear r=0 convention: inner_val_split(tr, g, seed=1234+r)
LAMBDA_GRID_SOURCE = (
    "RIDGE_LAMBDAS, scripts/issue658_fit_predictors.py:115 (re-exported via "
    "issue923_fit_decomposition -> issue1775_common); matches linear_fits.json per_lambda_r2 keys"
)
N_FOLDS_EXPECTED = 6


def _regime(args) -> dict:
    return {
        "arm": ARM,
        "grain": "perrow",
        "scheme": "prefix",
        "cell": CELL_PRIMARY,
        "layer": LAYER_PRIMARY,
        "bases": list(BASES),
        "lambda_grid": [float(v) for v in RIDGE_LAMBDAS],
        "inner_frac": INNER_VAL_GROUP_FRAC,
        "inner_seed_base": INNER_SEED_BASE,
        "row_limit": args.row_limit,
    }


class SvdRidge:
    """Standardize -> thin SVD; exact dual-form ridge predict at any grid lambda.

    Identical conventions to ``press_fit_predict(standardize=True)`` /
    ``PressRidge.predict`` (issue923_fit_decomposition): train-only per-dim mu/sd
    (ddof=0, +1e-9 floor), degenerate-dim drop, train-mean-centered targets,
    ``pred = Xte_n V diag(S/(S^2+lam)) U^T Yc + ymu``. Skips the PRESS LOO factors —
    the in-sample selection this refit replaces. One SVD is shared across bases +
    lambdas (the batched-factorization commitment; #823 class).
    """

    def __init__(self, Xtr: torch.Tensor) -> None:
        assert Xtr.ndim == 2, Xtr.shape
        mu = Xtr.mean(0)
        sd = Xtr.std(0, correction=0) + 1e-9
        keep = sd > (sd.max() * 1e-6 + 1e-12)
        self.mu, self.sd, self.keep = mu, sd, keep
        Xn = ((Xtr - mu) / sd)[:, keep]
        self.U, self.S, self.Vh = torch.linalg.svd(Xn, full_matrices=False)

    def project(self, Xte: torch.Tensor) -> torch.Tensor:
        """T = Xte_n @ V — the X-only per-call factor of PressRidge.predict."""
        return (((Xte - self.mu) / self.sd)[:, self.keep]) @ self.Vh.T

    def fit_targets(self, Ytr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(ymu, G=U^T Yc) for one target basis (train-mean centering convention)."""
        ymu = Ytr.mean(0, keepdim=True)
        return ymu, self.U.T @ (Ytr - ymu)

    def predict(
        self, T: torch.Tensor, G: torch.Tensor, ymu: torch.Tensor, lam: float
    ) -> torch.Tensor:
        coef = self.S / (self.S * self.S + float(lam))
        return (T * coef) @ G + ymu


def selftest() -> int:
    """Pin SvdRidge == press_fit_predict per-lambda preds (fp64 closed-form parity)."""
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((200, 50))
    W = rng.standard_normal((50, 12))
    Ytr = Xtr @ W + 0.1 * rng.standard_normal((200, 12))
    Xte = rng.standard_normal((60, 50))
    res = press_fit_predict(
        torch.from_numpy(Xtr).double(),
        torch.from_numpy(Ytr).double(),
        torch.from_numpy(Xte).double(),
        standardize=True,
    )
    eng = SvdRidge(torch.from_numpy(Xtr).double())
    T = eng.project(torch.from_numpy(Xte).double())
    ymu, G = eng.fit_targets(torch.from_numpy(Ytr).double())
    worst = 0.0
    for li, lam in enumerate(RIDGE_LAMBDAS):
        mine = eng.predict(T, G, ymu, lam)
        ref = res["per_lambda_pred"][li]
        d = float((mine - ref).abs().max())
        worst = max(worst, d)
        assert torch.allclose(mine, ref, atol=1e-8, rtol=1e-7), (lam, d)
    print(f"[selftest] SvdRidge == press_fit_predict per-lambda preds; max_abs_diff={worst:.3e}")
    return 0


def _pca48_targets_cached(Y: np.ndarray, args) -> np.ndarray:
    """The parent pca48 projection (deterministic full-stack SVD, ~8 min CPU), cached.

    ``_pca_basis`` is a deterministic ``np.linalg.svd`` of the centered (n, 10752)
    stacked targets — identical output per (population, row_limit) — so the (n, 48)
    projection is computed once and reused across the chunked fold invocations.
    """
    cache = _work_dir() / f"pca48_targets_rl{args.row_limit or 'full'}_n{Y.shape[0]}.npz"
    if cache.exists():
        return np.load(cache)["Yb"]
    t0 = time.monotonic()
    Yb = np.ascontiguousarray(
        _basis_targets_with_info(
            Y, "pca48", hidden_dim=3584, targets=["t1", "t2", "t3"], projection_target="t1"
        )[0],
        dtype=np.float64,
    )
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez(tmp, Yb=Yb)
    tmp.replace(cache)
    print(f"[refit] pca48 target projection computed+cached in {time.monotonic() - t0:.1f}s")
    return Yb


def _load_arm(args):
    ad = build_arm_data(
        resolve_store_dir(),
        CELL_PRIMARY,
        LAYER_PRIMARY,
        arms=(ARM,),
        row_limit=args.row_limit,
    )
    X = ad.X[ARM]
    pairs = fold_pairs(ad.rows, len(ad.rows), "prefix")
    pairs = restrict_pairs(pairs, ad.arm_row_mask[ARM])
    Yb = {
        "pca48": _pca48_targets_cached(ad.Y_stacked, args),
        "ambient": np.ascontiguousarray(ad.Y_stacked, dtype=np.float64),
    }
    return ad, X, pairs, Yb


def _work_dir() -> Path:
    return tensors_dir("qa_refit_work")


def run_folds(args) -> int:
    ad, X, pairs, Yb = _load_arm(args)
    if not args.row_limit:
        assert len(pairs) == N_FOLDS_EXPECTED, len(pairs)
    folds = (
        list(range(len(pairs))) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    )
    Xt = torch.from_numpy(X).double()
    Ybt = {b: torch.from_numpy(Yb[b]).double() for b in BASES}
    wd = _work_dir()
    regime = _regime(args)
    for fi in folds:
        rec_path = wd / f"fold_{fi}.json"
        npz_path = wd / f"fold_{fi}.npz"
        if rec_path.exists() and npz_path.exists():
            prev = json.loads(rec_path.read_text())
            assert prev["regime"] == regime, (
                f"fold {fi} checkpoint regime mismatch — clear {wd} to refit: "
                f"{prev['regime']} != {regime}"
            )
            print(f"[refit] fold {fi} already done — skip (resume)", flush=True)
            continue
        t0 = time.monotonic()
        tr, te = pairs[fi]
        itr, ival = inner_val_split(tr, ad.prefix_ids, seed=INNER_SEED_BASE + fi)
        eng_in = SvdRidge(Xt[itr])
        T_val = eng_in.project(Xt[ival])
        inner_r2: dict[str, dict[str, float]] = {}
        chosen: dict[str, float] = {}
        for b in BASES:
            ymu, G = eng_in.fit_targets(Ybt[b][itr])
            r2s = {}
            for lam in RIDGE_LAMBDAS:
                pv = eng_in.predict(T_val, G, ymu, lam).numpy()
                r2s[str(lam)] = _r2(Yb[b][ival], pv)
            inner_r2[b] = r2s
            chosen[b] = float(max(RIDGE_LAMBDAS, key=lambda lam: r2s[str(lam)]))
        del eng_in, T_val
        eng_out = SvdRidge(Xt[tr])
        T_te = eng_out.project(Xt[te])
        fold_r2: dict[str, float] = {}
        preds: dict[str, np.ndarray] = {}
        for b in BASES:
            ymu, G = eng_out.fit_targets(Ybt[b][tr])
            pt = eng_out.predict(T_te, G, ymu, chosen[b]).numpy()
            fold_r2[b] = _r2(Yb[b][te], pt)
            preds[b] = pt.astype(np.float16)
        del eng_out, T_te
        np.savez(npz_path.with_name(npz_path.stem + ".tmp.npz"), te=te, **preds)
        (npz_path.with_name(npz_path.stem + ".tmp.npz")).replace(npz_path)
        atomic_write_json(
            rec_path,
            {
                "fold": fi,
                "regime": regime,
                "n_tr": int(tr.size),
                "n_te": int(te.size),
                "n_inner_tr": int(itr.size),
                "n_inner_val": int(ival.size),
                "inner_seed": INNER_SEED_BASE + fi,
                "chosen_lambda": chosen,
                "inner_val_r2_per_lambda": inner_r2,
                "fold_r2": fold_r2,
                "wall_s": time.monotonic() - t0,
            },
        )
        print(
            f"[refit] fold {fi + 1}/{len(pairs)} lam={chosen} r2={fold_r2} "
            f"elapsed={time.monotonic() - t0:.1f}s",
            flush=True,
        )
    return 0


def _stage_nonlinear_pred(name: str, cache: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Download one persisted per-row pred (+mask) from the HF heldout_preds bucket."""
    from explore_persona_space.orchestrate import hub

    out = []
    for suffix in ("", "_mask"):
        fn = f"{name}{suffix}.npy"
        target = cache / fn
        try:
            hub.stage_hub_file(
                HF_DATA_REPO,
                f"{OUT_HF_PREFIX}/analysis_tensors/heldout_preds/{fn}",
                target,
            )
        except Exception as e:  # missing-on-Hub -> report gain without CI (fail-loud print)
            print(f"[gains] pred {fn} unavailable on Hub ({type(e).__name__}: {e})", flush=True)
            return None
        out.append(np.load(target))
    return out[0].astype(np.float64), out[1]


def _gain_entries(args, ad, Yb, my_pred, my_cov, new_r2) -> dict:
    """Recomputed per-rung gains = nonlinear level - healthy linear baseline (+CIs)."""
    nl = json.loads((eval_dir("ladder") / "nonlinear_fits.json").read_text())
    units = [
        u
        for u in nl["units"]
        if u["arm"] == ARM
        and u["grain"] == "perrow"
        and u.get("cell", CELL_PRIMARY) == CELL_PRIMARY
    ]
    cache = Path("data/issue_1775/hf_dl/heldout_preds")
    cache.mkdir(parents=True, exist_ok=True)
    gains: dict[str, dict] = {}
    for u in units:
        rung = u["rung"]
        if rung == "mlp":
            combos = [("pca48", int(s), u["r2_by_seed"][s]) for s in sorted(u["r2_by_seed"])]
        else:
            seed = None if rung == "krr" else int(u["seed"])
            combos = [(b, seed, u["r2"][b]) for b in BASES if b in u["r2"]]
        for basis, seed, level in combos:
            key = f"{ARM}|perrow|prefix|{rung}|{basis}" + (
                f"|s{seed}" if seed is not None and seed >= 0 else ""
            )
            entry: dict = {
                "nonlinear_level_r2": float(level),
                "linear_innerval_r2": float(new_r2[basis]),
                "gain_r2": float(level) - float(new_r2[basis]),
            }
            if not args.skip_ci:
                seed_part = f"_s{seed}" if seed is not None and seed >= 0 else ""
                name = (
                    f"{CELL_PRIMARY}_L{LAYER_PRIMARY:02d}_{ARM}_perrow_{basis}_prefix_"
                    f"{rung}{seed_part}"
                )
                staged = _stage_nonlinear_pred(name, cache)
                if staged is None:
                    entry["ci"] = "unavailable — per-row nonlinear preds not on Hub"
                else:
                    npred, nmask = staged
                    both = nmask & my_cov
                    entry["bootstrap_vs_innerval_ridge"] = cluster_bootstrap_delta_r2(
                        Yb[basis], npred, my_pred[basis], both, ad.prefix_ids
                    )
            gains[key] = entry
    return gains


def assemble(args) -> int:
    ad, X, pairs, Yb = _load_arm(args)
    n = len(ad.rows)
    wd = _work_dir()
    regime = _regime(args)
    my_pred = {b: np.zeros_like(Yb[b]) for b in BASES}
    covered = np.zeros(n, dtype=bool)
    fold_recs = []
    for fi in range(len(pairs)):
        rec_path, npz_path = wd / f"fold_{fi}.json", wd / f"fold_{fi}.npz"
        if not (rec_path.exists() and npz_path.exists()):
            print(f"[assemble] fold {fi} missing — run --folds first", flush=True)
            return 23
        rec = json.loads(rec_path.read_text())
        assert rec["regime"] == regime, f"fold {fi} regime mismatch: {rec['regime']} != {regime}"
        z = np.load(npz_path)
        te = z["te"]
        for b in BASES:
            my_pred[b][te] = z[b].astype(np.float64)
        covered[te] = True
        fold_recs.append(rec)
    cov = np.nonzero(covered)[0]
    new_r2 = {b: _r2(Yb[b][cov], my_pred[b][cov]) for b in BASES}
    lin = json.loads((eval_dir("ladder") / "linear_fits.json").read_text())
    old = {
        u["basis"]: {"r2": u["r2"], "lambda_values": u["lambda_values"]}
        for u in lin["units"]
        if u["arm"] == ARM
        and u["grain"] == "perrow"
        and u["cell"] == CELL_PRIMARY
        and u["layer"] == LAYER_PRIMARY
    }
    out = {
        "regime": regime,
        "lambda_grid_source": LAMBDA_GRID_SOURCE,
        "inner_split_convention": (
            f"group-respecting inner_val_split (issue1775_common) on PREFIX groups of each "
            f"outer train fold; frac={INNER_VAL_GROUP_FRAC}, seed={INNER_SEED_BASE}+fold "
            f"(issue1775_bilinear r=0 convention); lambda = argmax inner-val pooled R2; "
            f"refit on the full outer train at the chosen lambda"
        ),
        "n_rows_tested": int(cov.size),
        "pooled_r2": new_r2,
        "per_fold": [
            {
                "fold": r["fold"],
                "chosen_lambda": r["chosen_lambda"],
                "fold_r2": r["fold_r2"],
                "inner_val_r2_per_lambda": r["inner_val_r2_per_lambda"],
                "n_tr": r["n_tr"],
                "n_te": r["n_te"],
                "n_inner_tr": r["n_inner_tr"],
                "n_inner_val": r["n_inner_val"],
                "inner_seed": r["inner_seed"],
                "wall_s": r["wall_s"],
            }
            for r in fold_recs
        ],
        "press_baseline_reference": old,
        "gains_vs_innerval_ridge": _gain_entries(args, ad, Yb, my_pred, covered, new_r2),
        "meta": result_meta(script="issue1775_query_averaged_refit.py"),
    }
    dest = eval_dir("ladder") / "query_averaged_refit.json"
    atomic_write_json(dest, out)
    print(f"[assemble] wrote {dest}; pooled_r2={new_r2}", flush=True)
    # persist the refit per-row preds (fp16, same shape convention as persist_pred)
    pd = tensors_dir("qa_refit_preds")
    for b in BASES:
        name = f"{CELL_PRIMARY}_L{LAYER_PRIMARY:02d}_{ARM}_perrow_{b}_prefix_ridge_innerval"
        np.save(pd / f"{name}.npy", my_pred[b].astype(np.float16))
        np.save(pd / f"{name}_mask.npy", covered)
    print(f"[assemble] refit preds saved under {pd}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--folds", default=None, help="comma list of outer fold indices, or 'all'")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--upload-preds", action="store_true", help="one folder commit to the Hub")
    ap.add_argument("--skip-ci", action="store_true", help="levels-only gains (no Hub preds)")
    ap.add_argument("--row-limit", type=int, default=None, help="tiny-real smoke slice")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    rc = 0
    if args.folds is not None:
        rc = run_folds(args)
    if rc == 0 and args.assemble:
        rc = assemble(args)
    if rc == 0 and args.upload_preds:
        url = upload_phase_tensors("qa_refit_preds", smoke=bool(args.row_limit))
        print(f"[upload] qa_refit_preds -> {url}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
