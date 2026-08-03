"""#1774 P2 — fit battery + operator reads (pod GPU / CPU smoke, torch fp64 fits).

Engine: ``PressRidge`` (#923, exact PRESS-LOO thin SVD, batched draws) on the
extended λ grid (parent ``RIDGE_LAMBDAS`` ± one decade). Every svd/eigh rides the
CPU-fallback wrappers (gotchas #1335). Reads per plan §4: four-arm fits +
baselines, parity fit (banked convention), Q1a joint-vs-marginal stitch, Q1b
chain, Q3 channels (registered contiguous-from-top count rule + count-null band
+ BH companion; per-draw × per-component matrices PERSISTED), Q3 cross-arm
angles vs spectrum-matched nulls + matched-n, Q4 co-kernel ceilings, Q5
endomorphism reads (non-normality gate first), decode (logit lens), and the P3
direction sets.

Steps: ``--step fits|parity|q1a|q1b|q3|q4|q5|decode|directions|all``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env bind BEFORE the heavy imports below (BLAS/torch
# pools freeze at import time; tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue1774_common as c  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

K_SUBSPACES = (10, 48)
TOP_SING_DIRS = 20
N_ANSWER_PCS = 10


# ── design assembly ──────────────────────────────────────────────────────────


class Designs:
    """Arm designs + targets for one layer (fp64), on the fit-row subset."""

    def __init__(self, layer: int, smoke: bool, out_root: str | None) -> None:
        self.layer = layer
        self.smoke = smoke
        rows = c.load_manifest()
        reg = json.loads((c.eval_out(out_root) / "registry/folds.json").read_text())
        self.fit_idx = list(reg["fit_manifest_indices"])
        fit_rows = [rows[i] for i in self.fit_idx]
        self.prefix_ids = np.asarray([str(r.get("prefix_id", "")) for r in fit_rows])
        self.query_ids = np.asarray([str(r.get("query_id", "")) for r in fit_rows])
        self.folds = [np.asarray(f, dtype=np.int64) for f in reg["folds"]]
        if smoke:
            keep_prefixes = sorted(set(self.prefix_ids))[:30]
            keep = np.isin(self.prefix_ids, keep_prefixes)
            keep_rows = np.where(keep)[0]
            remap = {int(old): new for new, old in enumerate(keep_rows)}
            self.fit_idx = [self.fit_idx[i] for i in keep_rows]
            self.prefix_ids = self.prefix_ids[keep_rows]
            self.query_ids = self.query_ids[keep_rows]
            folds = []
            for f in self.folds[:2]:
                mapped = np.asarray([remap[int(i)] for i in f if int(i) in remap], dtype=np.int64)
                if mapped.size:
                    folds.append(mapped)
            self.folds = folds
            print(f"[designs] smoke slice: {len(self.fit_idx)} rows, {len(self.folds)} folds")
        idx = np.asarray(self.fit_idx, dtype=np.int64)
        ctx = c.load_summary_rows(c.CELL, "context_end", layer)
        pfx = c.load_summary_rows(c.CELL, "prefix_end", layer)
        t1 = c.load_summary_rows(c.CELL, "t1", layer)
        bare, q2i = c.load_bare(layer)
        self.X_ctx = np.asarray(ctx[idx], dtype=np.float64)
        self.X_pfx = np.asarray(pfx[idx], dtype=np.float64)
        self.Y = np.asarray(t1[idx], dtype=np.float64)
        self.X_bare = np.asarray(bare[[q2i[q] for q in self.query_ids]], dtype=np.float64)
        self.X_loro, self.loro_keep, self.prefix_means = c.loro_query_avg(
            self.X_ctx, self.prefix_ids
        )
        # per-prefix averaged targets + membership
        self.prefix_rows: dict[str, np.ndarray] = {}
        for i, p in enumerate(self.prefix_ids):
            self.prefix_rows.setdefault(str(p), []).append(i)  # type: ignore[arg-type]
        self.prefix_rows = {p: np.asarray(v, dtype=np.int64) for p, v in self.prefix_rows.items()}
        self.avg_targets = {p: self.Y[ix].mean(0) for p, ix in self.prefix_rows.items()}

    def arm_X(self, arm: str) -> np.ndarray:
        return {
            "arm_context": self.X_ctx,
            "arm_prefix_end": self.X_pfx,
            "arm_bare_query": self.X_bare,
            "arm_query_avg": self.X_loro,
        }[arm]

    def fold_prefixes(self, fold: np.ndarray) -> list[str]:
        return sorted({str(p) for p in self.prefix_ids[fold]})

    def pca48_basis(self, fold_i: int, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mu (1,d), P (d,48)) train-fold top-48 t1 PCs — the pca48 companion
        target basis (plan §4: pca48 companion for R²/trait tables only).
        Arm-independent (Y is shared), so cached per fold index."""
        if not hasattr(self, "_pca48_cache"):
            self._pca48_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if fold_i not in self._pca48_cache:
            mu = self.Y[tr].mean(0, keepdims=True)
            Yc = torch.from_numpy(self.Y[tr] - mu).double()
            _u, _s, vh = c.svd_robust(Yc)
            self._pca48_cache[fold_i] = (mu, vh[:48].T.numpy())
        return self._pca48_cache[fold_i]


DEDUP_ARMS = {"arm_prefix_end", "arm_query_avg"}


# ── weighted / plain PRESS fits (extended λ grid) ────────────────────────────


def _standardize(Xt: torch.Tensor, w: torch.Tensor | None):
    if w is None:
        mu = Xt.mean(0)
        sd = Xt.std(0, correction=0) + 1e-9
    else:
        wn = w / w.sum()
        mu = (wn.unsqueeze(1) * Xt).sum(0)
        var = (wn.unsqueeze(1) * (Xt - mu) ** 2).sum(0)
        sd = torch.sqrt(var) + 1e-9
    keep = sd > (sd.max() * 1e-6 + 1e-12)
    return mu, sd, keep


def fit_press_ext(
    X: np.ndarray, Y: np.ndarray, device: str, weights: np.ndarray | None = None
) -> dict:
    """Standardize (#923 convention; weighted variant for deduped arms) + PressRidge
    on the EXTENDED λ grid. Returns engine + factors for prediction/operators."""
    from issue923_fit_decomposition import PressRidge

    Xt = torch.from_numpy(np.ascontiguousarray(X)).double().to(device)
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).double().to(device)
    w = None if weights is None else torch.from_numpy(np.asarray(weights, np.float64)).to(device)
    mu, sd, keep = _standardize(Xt, w)
    Xn = ((Xt - mu) / sd)[:, keep]
    if w is None:
        ymu = Yt.mean(0, keepdim=True)
    else:
        ymu = ((w / w.sum()).unsqueeze(1) * Yt).sum(0, keepdim=True)
    Yc = Yt - ymu
    if w is not None:
        sw = torch.sqrt(w).unsqueeze(1)
        Xn = sw * Xn
        Yc = sw * Yc
    try:
        eng = PressRidge(Xn, lambdas=c.RIDGE_LAMBDAS_EXT)
    except torch.linalg.LinAlgError:
        # cuSOLVER non-convergence on a near-singular design (gotchas #1335):
        # exact CPU-LAPACK fallback — move the WHOLE fit to CPU so every
        # downstream consumer (predict / operator) sees one consistent device.
        print(
            f"[fit_press_ext] cuda svd failed in PressRidge (n={Xn.shape[0]}); CPU fallback",
            flush=True,
        )
        Xn, Yc = Xn.cpu(), Yc.cpu()
        mu, sd, keep, ymu = mu.cpu(), sd.cpu(), keep.cpu(), ymu.cpu()
        eng = PressRidge(Xn, lambdas=c.RIDGE_LAMBDAS_EXT)
    mse, G = eng.press_mse(Yc.unsqueeze(0))
    lam_idx = int(torch.argmin(mse[0]).item())
    return {
        "eng": eng,
        "G": G[0],
        "mse": mse[0].cpu().numpy(),
        "lam_idx": lam_idx,
        "lam": float(c.RIDGE_LAMBDAS_EXT[lam_idx]),
        "df": float(eng.phi[lam_idx].sum().item()),
        "edge_saturated": lam_idx in (0, len(c.RIDGE_LAMBDAS_EXT) - 1),
        "mu": mu,
        "sd": sd,
        "keep": keep,
        "ymu": ymu,
        "d_full": int(Xt.shape[1]),
        "weighted": weights is not None,
    }


def predict(fit: dict, X_new: np.ndarray, lam_idx: int | None = None) -> np.ndarray:
    dev = fit["mu"].device
    Xt = torch.from_numpy(np.ascontiguousarray(X_new)).double().to(dev)
    Xn = ((Xt - fit["mu"]) / fit["sd"])[:, fit["keep"]]
    li = fit["lam_idx"] if lam_idx is None else lam_idx
    idx = torch.full((1,), li, dtype=torch.long, device=dev)
    pred = fit["eng"].predict(fit["G"].unsqueeze(0), idx, Xn)[0] + fit["ymu"]
    return pred.cpu().numpy()


def operator_raw(fit: dict, lam: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(W (P,d_full), b (P,)) raw-space affine map: pred = W @ x + b — CPU fp64.

    Routes through ``c.operator_raw_safe`` (the device-safe wrapper around the
    parent ``_operator_raw``, whose ``torch.zeros(P, d_full)`` allocates on CPU
    with no device arg — plan asm 4) and folds in the intercept:
    b = ymu − W @ mu, computed CPU-side to match W's device.
    """
    lam_v = fit["lam"] if lam is None else lam
    W = c.operator_raw_safe(fit, fit["G"], lam_v)
    b = fit["ymu"][0].detach().cpu() - W @ fit["mu"].detach().cpu()
    return W, b


def _knn(pred: np.ndarray, true: np.ndarray) -> dict:
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    return {m: knn_retrieval(pred, true, ks=(1, 5, 10), metric=m) for m in ("euclidean", "cosine")}


# ── step: fits (per arm × layer: 6 fold fits + pooled + baselines) ───────────


def _train_design(d: Designs, arm: str, rows: np.ndarray):
    """(X_train, Y_train, weights_or_None) for a train-row subset; deduped arms
    collapse to distinct prefixes with multiplicity weights (plan §4)."""
    if arm not in DEDUP_ARMS:
        return d.arm_X(arm)[rows], d.Y[rows], None
    Xsrc = d.X_pfx if arm == "arm_prefix_end" else None
    pids = sorted({str(p) for p in d.prefix_ids[rows]})
    Xd, Yd, w = [], [], []
    for p in pids:
        ix = np.intersect1d(d.prefix_rows[p], rows)
        if arm == "arm_prefix_end":
            Xd.append(Xsrc[ix[0]])
        else:  # query_avg: plain per-prefix mean over TRAIN rows only
            Xd.append(d.X_ctx[ix].mean(0))
        Yd.append(d.Y[ix].mean(0))
        w.append(len(ix))
    return np.stack(Xd), np.stack(Yd), np.asarray(w, dtype=np.float64)


def step_fits(d: Designs, arms: list[str], device: str, out_root: str | None) -> None:
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    eval_dir = c.eval_out(out_root) / "fit_battery"
    op_dir = c.data_out(out_root) / "operators"
    op_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        out_json = eval_dir / f"{arm}_L{d.layer}.json"
        if out_json.exists():
            print(f"[p2-fits] resume: {out_json.name} exists")
            continue
        t0 = time.time()
        n = len(d.fit_idx)
        pred_ctx = np.full_like(d.Y, np.nan)
        row_eval_mask = np.ones(n, dtype=bool)
        if arm == "arm_query_avg":
            row_eval_mask = d.loro_keep
        fold_rows = []
        for fi, te in enumerate(d.folds):
            tr = np.setdiff1d(np.arange(n), te)
            Xtr, Ytr, w = _train_design(d, arm, tr)
            fit = fit_press_ext(Xtr, Ytr, device, weights=w)
            te_eval = te[row_eval_mask[te]]
            X_te = d.arm_X(arm)[te_eval]
            pred_ctx[te_eval] = predict(fit, X_te)
            # averaged grain: mean of per-row preds per held-out prefix
            held = d.fold_prefixes(te)
            avg_pred, avg_true = [], []
            for p in held:
                ix = np.intersect1d(d.prefix_rows[p], te_eval)
                if ix.size:
                    avg_pred.append(pred_ctx[ix].mean(0))
                    avg_true.append(d.avg_targets[p])
            r2_avg = (
                c.r2_score(np.stack(avg_true), np.stack(avg_pred)) if avg_pred else float("nan")
            )
            # identity+bias baseline (same-dim maps; standing rule)
            Xtr_rows = d.arm_X(arm)[tr[row_eval_mask[tr]]]
            Ytr_rows = d.Y[tr[row_eval_mask[tr]]]
            idb = identity_bias_predict(Xtr_rows, Ytr_rows, X_te)
            # pca48 companion (plan §4 target-basis companion): train-fold
            # top-48 t1 PCs at the ambient PRESS-λ (ridge is linear in Y, so
            # projecting the ambient OOF predictions IS the pca48-target fit
            # at that λ). identity+bias inapplicable (3584→48) — R² + kNN only.
            mu48, P48 = d.pca48_basis(fi, tr)
            t48 = (d.Y[te_eval] - mu48) @ P48
            p48 = (pred_ctx[te_eval] - mu48) @ P48
            fold_rows.append(
                {
                    "fold": fi,
                    "n_test_rows": int(te_eval.size),
                    "lam": fit["lam"],
                    "lam_idx": fit["lam_idx"],
                    "df": fit["df"],
                    "edge_saturated": fit["edge_saturated"],
                    "r2_per_context": c.r2_score(d.Y[te_eval], pred_ctx[te_eval]),
                    "r2_averaged": r2_avg,
                    "r2_identity_bias": c.r2_score(d.Y[te_eval], idb),
                    "knn": _knn(pred_ctx[te_eval], d.Y[te_eval]),
                    "r2_pca48": c.r2_score(t48, p48),
                    "knn_pca48": _knn(p48, t48),
                }
            )
            print(
                f"[p2-fits] unit {arm}_L{d.layer} fold {fi + 1}/{len(d.folds)} "
                f"r2={fold_rows[-1]['r2_per_context']:.4f} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        # pooled fit → operator persisted
        Xtr, Ytr, w = _train_design(d, arm, np.arange(n))
        pooled = fit_press_ext(Xtr, Ytr, device, weights=w)
        W, b = operator_raw(pooled)
        np.save(op_dir / f"W_{arm}_L{d.layer}.npy", W.cpu().numpy().astype(np.float32))
        np.save(op_dir / f"b_{arm}_L{d.layer}.npy", b.cpu().numpy().astype(np.float32))
        ev = row_eval_mask
        c.write_json_atomic(
            out_json,
            {
                "meta": c.repro_meta({"script": "issue1774_fit_battery.py --step fits"}),
                "arm": arm,
                "layer": d.layer,
                "n_fit_rows": n,
                "n_eval_rows": int(ev.sum()),
                "n_loro_excluded_singleton_rows": int((~ev).sum()),
                "lambda_grid": c.RIDGE_LAMBDAS_EXT,
                "pooled": {
                    "lam": pooled["lam"],
                    "df": pooled["df"],
                    "edge_saturated": pooled["edge_saturated"],
                },
                "folds": fold_rows,
                "r2_per_context_pooled_oof": c.r2_score(d.Y[ev], pred_ctx[ev]),
                "pca48_convention": "train-fold top-48 t1 PCs; ambient OOF predictions "
                "projected at the ambient PRESS-λ (ridge linear in Y ⇒ identical to the "
                "pca48-target fit at that λ); identity+bias inapplicable 3584→48 — kNN "
                "+ R² reported (standing mapping-baselines rule)",
                "weighted_dedup": arm in DEDUP_ARMS,
                "operator_paths": [
                    f"data/issue_1774/operators/W_{arm}_L{d.layer}.npy",
                    f"data/issue_1774/operators/b_{arm}_L{d.layer}.npy",
                ],
            },
        )
        np.save(op_dir / f"oof_pred_{arm}_L{d.layer}.npy", pred_ctx.astype(np.float32))
        print(f"[p2-fits] {arm}_L{d.layer} done in {time.time() - t0:.0f}s")


def step_parity(d: Designs, device: str, out_root: str | None) -> None:
    """Banked-convention parity fit (context, L14, ambient, t1) — anchor check only."""
    if d.layer != c.HEADLINE_LAYER:
        return
    rows = c.load_manifest()
    banked = c.banked_convention_indices(rows)
    if d.smoke:
        banked = banked[: len(d.fit_idx)]
    idx = np.asarray(banked, dtype=np.int64)
    ctx = c.load_summary_rows(c.CELL, "context_end", d.layer)
    t1 = c.load_summary_rows(c.CELL, "t1", d.layer)
    X = np.asarray(ctx[idx], dtype=np.float64)
    Y = np.asarray(t1[idx], dtype=np.float64)
    brows = [rows[i] for i in banked]
    folds = c.grouped_folds(brows, len(brows))
    if d.smoke:
        folds = folds[:2]
    pred = np.full_like(Y, np.nan)
    for fi, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(banked)), te)
        fit = fit_press_ext(X[tr], Y[tr], device)
        pred[te] = predict(fit, X[te])
        print(f"[p2-parity] fold {fi + 1}/{len(folds)}", flush=True)
    ev = ~np.isnan(pred[:, 0])
    c.write_json_atomic(
        c.eval_out(out_root) / "fit_battery" / "parity_banked_convention_L14.json",
        {
            "meta": c.repro_meta(),
            "note": "banked-row-convention parity fit — parent-anchor ±0.03 check ONLY "
            "(headline reads use the corrected 17,308-row fits; plan §6a)",
            "n_rows": int(idx.size),
            "r2_per_context_pooled_oof": c.r2_score(Y[ev], pred[ev]),
        },
    )


# ── step: q3 channels + rho1^2 + per-trait table + cross-arm angles ──────────


def _perm_indices(rng: np.random.Generator, n: int, draws: int) -> np.ndarray:
    return np.stack([rng.permutation(n) for _ in range(draws)], axis=0)


def step_q3(d: Designs, arms: list[str], device: str, out_root: str | None, n_draws: int) -> None:
    eval_dir = c.eval_out(out_root) / "channels"
    null_dir = c.data_out(out_root) / "analysis_tensors/channels_null"
    null_dir.mkdir(parents=True, exist_ok=True)
    rb = c.load_rb_bank(d.layer)
    Yt_all = torch.from_numpy(d.Y).double().to(device)
    # answer-PC directions (train-fold-independent descriptive basis: pooled t1 PCA)
    Yc_all = Yt_all - Yt_all.mean(0, keepdim=True)
    _u, _s, vh_y = c.svd_robust(Yc_all)
    answer_pcs = vh_y[:N_ANSWER_PCS].cpu().numpy()
    for arm in arms:
        out_json = eval_dir / f"{arm}_L{d.layer}.json"
        if out_json.exists():
            print(f"[p2-q3] resume: {out_json.name} exists")
            continue
        t0 = time.time()
        n = len(d.fit_idx)
        ev_mask = d.loro_keep if arm == "arm_query_avg" else np.ones(n, dtype=bool)
        X_all = d.arm_X(arm)
        n_distinct = len(set(map(tuple, X_all[ev_mask][:: max(1, n // 512)].round(4).tolist())))
        K = int(os.environ.get("I1774_CHANNEL_K", "0")) or min(c.HIDDEN_DIM, int(ev_mask.sum()) - 1)
        rng = np.random.default_rng(c.SEED_DRAWS + d.layer)
        # accumulate per-component residuals across folds (obs + null draws)
        res_obs = np.zeros(K)
        tot_obs = np.zeros(K)
        res_null = np.zeros((n_draws, K))
        rho1_folds = []
        trait_proj_res: dict[str, float] = {t: 0.0 for t in rb}
        trait_proj_tot: dict[str, float] = {t: 0.0 for t in rb}
        pc_res = np.zeros(N_ANSWER_PCS)
        pc_tot = np.zeros(N_ANSWER_PCS)
        for fi, te in enumerate(d.folds):
            tr = np.setdiff1d(np.arange(n), te)
            tr = tr[ev_mask[tr]]
            te = te[ev_mask[te]]
            Xtr = torch.from_numpy(X_all[tr]).double().to(device)
            Ytr = Yt_all[tr]
            mu, sd, keep = _standardize(Xtr, None)
            Xn = ((Xtr - mu) / sd)[:, keep]
            ymu = Ytr.mean(0, keepdim=True)
            Yc = Ytr - ymu
            m = Xn.shape[0]
            C = (Xn.T @ Yc) / m
            U, S, Vh = c.svd_robust(C)
            Kf = min(K, S.shape[0])
            U, Vh = U[:, :Kf], Vh[:Kf]
            Ztr = Xn @ U  # (m, Kf)
            Ttr = Yc @ Vh.T
            bcoef = (Ztr * Ttr).sum(0) / (Ztr * Ztr).sum(0).clamp(min=1e-30)
            Xte = torch.from_numpy(X_all[te]).double().to(device)
            Yte = Yt_all[te]
            Zte = (((Xte - mu) / sd)[:, keep]) @ U
            Tte = (Yte - ymu) @ Vh.T
            resid = Tte - Zte * bcoef
            res_obs[:Kf] += (resid**2).sum(0).cpu().numpy()
            tot_obs[:Kf] += (Tte**2).sum(0).cpu().numpy()
            # nulls: within-test-fold row permutation of Y (pairing broken;
            # train directions fixed; per-draw same rule downstream)
            perms = _perm_indices(rng, te.size, n_draws)
            chunk = 25
            for b0 in range(0, n_draws, chunk):
                pb = perms[b0 : b0 + chunk]
                Tp = torch.stack([Tte[torch.from_numpy(p).to(device)] for p in pb], dim=0)
                rp = Tp - Zte.unsqueeze(0) * bcoef
                res_null[b0 : b0 + chunk, :Kf] += (rp**2).sum(1).cpu().numpy()
            # rho1^2: whitened-coord top canonical corr, train-estimated test-scored
            Ux, Sx, Vx = c.svd_robust(Xn)
            Uy, Sy, Vy = c.svd_robust(Yc)
            corr = Ux.T @ Uy
            uc, sc, vc = c.svd_robust(corr)
            floor_x = Sx.max() * 1e-6
            floor_y = Sy.max() * 1e-6
            wx = Vx.T @ (uc[:, 0] / Sx.clamp(min=floor_x))
            wy = Vy.T @ (vc[0] / Sy.clamp(min=floor_y))
            zx = (((Xte - mu) / sd)[:, keep]) @ wx
            zy = (Yte - ymu) @ wy
            zx = zx - zx.mean()
            zy = zy - zy.mean()
            denom = zx.norm() * zy.norm()
            rho_te = float((zx @ zy / denom).item()) if float(denom) > 0 else float("nan")
            rho1_folds.append(rho_te**2)
            # per-trait / answer-PC held-out R² along fixed directions
            pred_te = (
                torch.from_numpy(
                    np.load(c.data_out(out_root) / "operators" / f"oof_pred_{arm}_L{d.layer}.npy")[
                        te
                    ]
                )
                .double()
                .to(device)
            )
            for t, v in rb.items():
                vt = torch.from_numpy(v / np.linalg.norm(v)).double().to(device)
                pt, yt = pred_te @ vt, Yte @ vt
                trait_proj_res[t] += float(((yt - pt) ** 2).sum().item())
                trait_proj_tot[t] += float(((yt - yt.mean()) ** 2).sum().item())
            pcs = torch.from_numpy(answer_pcs).double().to(device)
            pp, yy = pred_te @ pcs.T, Yte @ pcs.T
            pc_res += ((yy - pp) ** 2).sum(0).cpu().numpy()
            pc_tot += ((yy - yy.mean(0, keepdim=True)) ** 2).sum(0).cpu().numpy()
            print(
                f"[p2-q3] unit {arm}_L{d.layer} fold {fi + 1}/{len(d.folds)} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        r2_obs = 1.0 - res_obs / np.maximum(tot_obs, 1e-30)
        r2_null = 1.0 - res_null / np.maximum(tot_obs[None, :], 1e-30)
        np.save(null_dir / f"r2_null_{arm}_L{d.layer}.npy", r2_null.astype(np.float32))
        p95 = np.percentile(r2_null, 95, axis=0)
        count = c.contiguous_count_from_top(r2_obs, p95)
        band = c.count_null_band(r2_null, p95)
        c.write_json_atomic(
            out_json,
            {
                "meta": c.repro_meta({"script": "issue1774_fit_battery.py --step q3"}),
                "arm": arm,
                "layer": d.layer,
                "n_components": int(len(r2_obs)),
                "n_distinct_sampled_est": n_distinct,
                "count_rule": "contiguous-from-top above per-component perm-null p95; "
                "SAME rule per draw -> count-null band; BH companion",
                "null_convention": "within-test-fold row permutation of Y (pairing "
                "broken; train-fold directions fixed), 200 draws, seed 1774+layer",
                "plan_registered_convention": "plan §4 Q3 registers same-lambda "
                "refit nulls (PressRidge press_mse) — substituted: the channel "
                "estimator is a per-fold cross-covariance SVD with no lambda "
                "(press_mse inapplicable), and a true per-draw refit is a dense "
                "(r x r) SVD per draw x fold x arm x layer (~14.4k factorizations), "
                "contradicting plan §9's own 'one GEMM stack per fold, never "
                "per-draw refits' sizing; the conditional test-fold permutation is "
                "exact for the OOF statistic and keeps the same contiguous-from-top "
                "rule per draw (recorded in results-sentinel plan_deviations)",
                "channel_count": count,
                "count_null_band": band,
                "bh_companion_count": c.bh_count(r2_obs, r2_null),
                "rho1_sq_folds": rho1_folds,
                "rho1_sq_mean": float(np.nanmean(rho1_folds)),
                "rho1_sq_jackknife_se": float(
                    np.nanstd(rho1_folds, ddof=1) / np.sqrt(max(1, len(rho1_folds)))
                ),
                "per_component_r2_obs": [float(x) for x in r2_obs],
                "per_component_null_p95": [float(x) for x in p95],
                "per_trait_heldout_r2": {
                    t: 1.0 - trait_proj_res[t] / max(trait_proj_tot[t], 1e-30) for t in rb
                },
                "answer_pc_heldout_r2": [
                    float(1.0 - pc_res[j] / max(pc_tot[j], 1e-30)) for j in range(N_ANSWER_PCS)
                ],
                "null_matrix_path": f"analysis_tensors/channels_null/r2_null_{arm}_L{d.layer}.npy",
            },
        )
        print(
            f"[p2-q3] {arm}_L{d.layer}: count={count} (null p95 count "
            f"{band['count_p95']:.1f}) in {time.time() - t0:.0f}s"
        )


def step_q3_angles(d: Designs, device: str, out_root: str | None, n_draws: int) -> None:
    """Cross-arm principal angles (top-k in/out subspaces) vs Haar nulls + matched-n."""
    from issue1092_partb_operator import _angle_null_band, _angles_between

    op_dir = c.data_out(out_root) / "operators"
    out_json = c.eval_out(out_root) / "channels" / f"cross_arm_angles_L{d.layer}.json"
    svds = {}
    for arm in c.ARMS:
        W = torch.from_numpy(np.load(op_dir / f"W_{arm}_L{d.layer}.npy")).double()
        U, S, Vh = c.svd_robust(W)
        svds[arm] = (U, S, Vh)
    gen = torch.Generator().manual_seed(c.SEED_DRAWS)
    pairs = []
    for i, a1 in enumerate(c.ARMS):
        for a2 in c.ARMS[i + 1 :]:
            row = {"pair": [a1, a2]}
            for k in K_SUBSPACES:
                U1, S1, V1 = svds[a1]
                U2, S2, V2 = svds[a2]
                kk = min(k, S1.shape[0], S2.shape[0])
                row[f"out_angles_k{k}"] = _angles_between(U1[:, :kk], U2[:, :kk])
                row[f"in_angles_k{k}"] = _angles_between(V1[:kk].T, V2[:kk].T)
                row[f"null_band_k{k}"] = _angle_null_band(
                    c.HIDDEN_DIM, kk, kk, n_draws, 25, gen, max_rank=256
                )
            pairs.append(row)
    # matched-n control: context refit on 20 one-row-per-prefix subsamples
    rng = np.random.default_rng(c.SEED_DRAWS + 7)
    Uc, Sc, Vc = svds["arm_context"]
    matched = []
    n_matched = 2 if d.smoke else c.N_MATCHED_N_DRAWS
    lam_ctx = json.loads(
        (c.eval_out(out_root) / "fit_battery" / f"arm_context_L{d.layer}.json").read_text()
    )["pooled"]["lam"]
    for t in range(n_matched):
        pick = np.asarray(
            [rng.choice(ix) for _p, ix in sorted(d.prefix_rows.items())], dtype=np.int64
        )
        fit = fit_press_ext(d.X_ctx[pick], d.Y[pick], device)
        li = min(
            range(len(c.RIDGE_LAMBDAS_EXT)),
            key=lambda j: abs(c.RIDGE_LAMBDAS_EXT[j] - lam_ctx),
        )
        fit["lam_idx"], fit["lam"] = li, c.RIDGE_LAMBDAS_EXT[li]
        Wm, _bm = operator_raw(fit)
        Um, Sm, Vm = c.svd_robust(Wm.cpu().double())
        kk = min(48, Sm.shape[0])
        matched.append(
            {
                "draw": t,
                "n_rows": int(pick.size),
                "out_angles_k48_vs_full_context": _angles_between(Um[:, :kk], Uc[:, :kk]),
                "singular_top10": [float(x) for x in Sm[:10]],
            }
        )
        print(f"[p2-q3-angles] matched-n draw {t + 1}/{n_matched}", flush=True)
    c.write_json_atomic(
        out_json,
        {
            "meta": c.repro_meta(),
            "layer": d.layer,
            "pairs": pairs,
            "matched_n_context": matched,
            "note": "angles vs Haar random-subspace null (_angle_null_band); matched-n "
            "separates the rank confound (one row per prefix, same-λ as context pooled)",
        },
    )


# ── step: q1a joint-vs-marginal (stitch) ─────────────────────────────────────


def _procrustes_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    s = torch.linalg.svdvals(A @ B.T)
    return float(s.sum() / (A.norm() * B.norm()))


def step_q1a(d: Designs, device: str, out_root: str | None, n_draws: int) -> None:
    from issue1092_partb_operator import _angles_between

    out_json = c.eval_out(out_root) / "fit_battery" / f"q1a_joint_vs_marginal_L{d.layer}.json"
    n = len(d.fit_idx)
    X_st = np.concatenate([d.X_pfx, d.X_bare], axis=1)
    pooled_st = fit_press_ext(X_st, d.Y, device)
    W_st, _b = operator_raw(pooled_st)
    A = W_st[:, : c.HIDDEN_DIM].cpu().double()  # prefix block
    # stitch fit R² + kNN (identity+bias inapplicable: 7168→3584, stated)
    pred = np.full_like(d.Y, np.nan)
    for fi, te in enumerate(d.folds):
        tr = np.setdiff1d(np.arange(n), te)
        fit = fit_press_ext(X_st[tr], d.Y[tr], device)
        pred[te] = predict(fit, X_st[te])
        print(f"[p2-q1a] stitch fold {fi + 1}/{len(d.folds)}", flush=True)
    ev = ~np.isnan(pred[:, 0])
    op_dir = c.data_out(out_root) / "operators"
    Wp = torch.from_numpy(np.load(op_dir / f"W_arm_prefix_end_L{d.layer}.npy")).double()
    Ua, Sa, Va = c.svd_robust(A)
    Up, Sp, Vp = c.svd_robust(Wp)
    kk = min(48, Sa.shape[0], Sp.shape[0])
    obs = {
        "raw_cosine": float((A * Wp).sum() / (A.norm() * Wp.norm())),
        "procrustes_cosine": _procrustes_cos(A, Wp),
        "in_angles_k48": _angles_between(Va[:kk].T, Vp[:kk].T),
        "out_angles_k48": _angles_between(Ua[:, :kk], Up[:, :kk]),
    }
    # fold-jackknife self-spread of the MARGINAL prefix operator
    jack = []
    for fi, te in enumerate(d.folds):
        tr = np.setdiff1d(np.arange(n), te)
        Xtr, Ytr, w = _train_design(d, "arm_prefix_end", tr)
        fitf = fit_press_ext(Xtr, Ytr, device, weights=w)
        Wf, _ = operator_raw(fitf)
        Wf = Wf.cpu().double()
        jack.append(
            {
                "fold": fi,
                "raw_cosine_vs_pooled": float((Wf * Wp).sum() / (Wf.norm() * Wp.norm())),
                "procrustes_cosine_vs_pooled": _procrustes_cos(Wf, Wp),
            }
        )
    # 200-draw shuffle-fit null (same-λ refits via shared fp32 factors; the
    # trace trick avoids materializing per-draw operators; truncated SVD via a
    # sketch on the factored form — under-estimation of nuc recorded, the
    # _procrustes_null_band convention)
    eng = pooled_st["eng"].cast(torch.float32)
    lam = pooled_st["lam"]
    coef = (eng.S / (eng.S**2 + lam)).float()
    keep_idx = pooled_st["keep"].nonzero(as_tuple=True)[0]
    prefix_keep = keep_idx < c.HIDDEN_DIM
    sd32 = pooled_st["sd"][pooled_st["keep"]].float()
    # T maps K -> prefix-block raw operator (transposed form): Wp_blk^T = T @ Kp
    Tm = (eng.Vh.T[prefix_keep.nonzero(as_tuple=True)[0].cpu()] * coef.unsqueeze(0)).to(device)
    Tm = Tm / sd32[prefix_keep].unsqueeze(1).to(device)
    A32 = A.float().to(device)
    Yc32 = (torch.from_numpy(d.Y).double().to(device) - pooled_st["ymu"]).float()
    U32 = eng.U.float().to(device)
    TtT = Tm.T @ Tm
    A_T = A32 @ Tm  # (P, k)
    rng = np.random.default_rng(c.SEED_DRAWS + 11)
    null_rows = []
    gen = torch.Generator(device="cpu").manual_seed(c.SEED_DRAWS)
    for b in range(n_draws):
        perm = torch.from_numpy(rng.permutation(n)).to(device)
        Kp = U32.T @ Yc32[perm]  # (k, P)
        num = float((A_T * Kp.T).sum().item())
        wnorm = float(torch.sqrt(((TtT @ Kp) * Kp).sum().clamp(min=0)).item())
        # sketch top-64 of Wp_perm = Kp^T Tm^T (P, d_blk)
        r = 64
        omega = torch.randn(Tm.shape[0], r, generator=gen).to(device)
        Ysk = Kp.T @ (Tm.T @ omega)  # (P, r)
        Q, _ = torch.linalg.qr(Ysk)
        B = (Kp @ Q).T @ Tm.T  # (r, d_blk)
        Ub, Sb, Vb = c.svd_robust(B)
        Upn = Q @ Ub
        kk2 = min(48, Sb.shape[0], Sa.shape[0])
        red = (
            (Sa[:kk2].float().to(device).view(-1, 1))
            * (Va[:kk2].float().to(device) @ Vb[:kk2].T)
            * Sb[:kk2].view(1, -1)
        )
        nuc = float(torch.linalg.svdvals(red).sum().item())
        null_rows.append(
            {
                "raw_cosine": num / max(1e-30, float(A32.norm().item()) * wnorm),
                "procrustes_cosine_trunc": nuc / max(1e-30, float(A32.norm().item()) * wnorm),
                "mean_out_angle_k48": float(
                    np.mean(_angles_between(Ua[:, :kk2].float(), Upn[:, :kk2].cpu()))
                ),
            }
        )
        if (b + 1) % 25 == 0:
            print(f"[p2-q1a] null draw {b + 1}/{n_draws}", flush=True)
    c.write_json_atomic(
        out_json,
        {
            "meta": c.repro_meta({"script": "issue1774_fit_battery.py --step q1a"}),
            "layer": d.layer,
            "stitch_r2_per_context_oof": c.r2_score(d.Y[ev], pred[ev]),
            "stitch_knn": _knn(pred[ev], d.Y[ev]),
            "identity_bias": "inapplicable — stitch input 7168 != output 3584 (stated)",
            "observed": obs,
            "fold_jackknife_marginal": jack,
            "shuffle_fit_null": {
                "n_draws": n_draws,
                "convention": "row-level pairing permutation; same-λ refits via shared "
                "fp32 PressRidge factors; Procrustes cosine + angles from a rank-64 "
                "sketch of the factored operator (nuc under-estimated by the recorded "
                "truncation, the _procrustes_null_band convention)",
                "draws": null_rows,
            },
        },
    )


# ── step: q1b chain ──────────────────────────────────────────────────────────


def step_q1b(d: Designs, device: str, out_root: str | None) -> None:
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    out_json = c.eval_out(out_root) / "fit_battery" / f"q1b_chain_L{d.layer}.json"
    pids = sorted(d.prefix_rows)
    E = np.stack([d.X_pfx[d.prefix_rows[p][0]] for p in pids])
    Vbar = np.stack([d.prefix_means[p] for p in pids])
    Abar = np.stack([d.avg_targets[p] for p in pids])
    npfx = len(pids)
    fold_of_prefix = {}
    for fi, te in enumerate(d.folds):
        for p in d.fold_prefixes(te):
            fold_of_prefix[p] = fi
    pred_chain = np.full_like(Abar, np.nan)
    pred_direct_e = np.full_like(Abar, np.nan)
    pred_direct_v = np.full_like(Abar, np.nan)
    pred_g = np.full_like(Vbar, np.nan)
    pred_g_idb = np.full_like(Vbar, np.nan)
    for fi in range(len(d.folds)):
        te = np.asarray([k for k, p in enumerate(pids) if fold_of_prefix.get(p) == fi])
        tr = np.setdiff1d(np.arange(npfx), te)
        if te.size == 0:
            continue
        g = fit_press_ext(E[tr], Vbar[tr], device)  # n<d — prediction-transfer read
        m_avg = fit_press_ext(Vbar[tr], Abar[tr], device)
        fe = fit_press_ext(E[tr], Abar[tr], device)
        pred_g[te] = predict(g, E[te])
        pred_g_idb[te] = identity_bias_predict(E[tr], Vbar[tr], E[te])
        pred_chain[te] = predict(m_avg, pred_g[te])
        pred_direct_e[te] = predict(fe, E[te])
        pred_direct_v[te] = predict(m_avg, Vbar[te])
        print(f"[p2-q1b] fold {fi + 1}/{len(d.folds)}", flush=True)
    ev = ~np.isnan(pred_chain[:, 0])
    # per-direction recovery along M_avg pooled top input singular dirs
    m_pooled = fit_press_ext(Vbar, Abar, device)
    Wm, _ = operator_raw(m_pooled)
    _u, Sm, Vh_m = c.svd_robust(Wm.cpu().double())
    r = min(48, Sm.shape[0])
    Uin = Vh_m[:r].T.numpy()  # input singular directions (d, r)
    gains = Sm[:r].numpy()
    r2_dir = []
    for j in range(r):
        tj = Vbar[ev] @ Uin[:, j]
        pj = pred_g[ev] @ Uin[:, j]
        r2_dir.append(c.r2_score(tj[:, None], pj[:, None]))
    r2_dir = np.asarray(r2_dir)
    w2 = gains**2
    recovered = float((w2 * np.clip(r2_dir, 0, None)).sum() / w2.sum())
    r2_chain = c.r2_score(Abar[ev], pred_chain[ev])
    r2_e = c.r2_score(Abar[ev], pred_direct_e[ev])
    r2_v = c.r2_score(Abar[ev], pred_direct_v[ev])
    c.write_json_atomic(
        out_json,
        {
            "meta": c.repro_meta({"script": "issue1774_fit_battery.py --step q1b"}),
            "layer": d.layer,
            "n_prefixes": npfx,
            "n_lt_d_flag": "n_distinct 1145 < d 3584 — prediction-transfer read only "
            "(estimator-validity registry; kernel claims excluded)",
            "r2_g_e_to_vbar": c.r2_score(Vbar[ev], pred_g[ev]),
            "r2_g_identity_bias": c.r2_score(Vbar[ev], pred_g_idb[ev]),
            "knn_g": _knn(pred_g[ev], Vbar[ev]),
            "r2_chain_Mavg_g_e": r2_chain,
            "r2_direct_e_to_abar": r2_e,
            "r2_direct_vbar_to_abar": r2_v,
            "direct_deficit": r2_v - r2_e,
            "chain_recovered_share_of_deficit": (
                float((r2_chain - r2_e) / (r2_v - r2_e)) if abs(r2_v - r2_e) > 1e-9 else None
            ),
            "per_direction_recovery": {
                "r2_per_direction": [float(x) for x in r2_dir],
                "input_gains": [float(x) for x in gains],
                "gain_weighted_recovered_fraction": recovered,
            },
            "note": "a linear chain can never beat the direct linear fit — this "
            "LOCALIZES the linear deficit (H1b); nonlinear readability is #1775",
        },
    )


# ── step: q4 co-kernel ceilings ──────────────────────────────────────────────


def _k_energy(S: torch.Tensor, frac: float) -> int:
    e = S.double() ** 2
    cs = torch.cumsum(e, 0) / e.sum().clamp(min=1e-300)
    return int(torch.searchsorted(cs, torch.tensor(frac, dtype=cs.dtype)).item()) + 1


def step_q4(d: Designs, arms: list[str], device: str, out_root: str | None) -> None:
    # C1 guard (round 2): the per-trait cokernel_*.json + cokernel_all_*.json are
    # COMBINED-across-arms writes — two concurrent sharded invocations would
    # last-writer-win and silently drop half the Q4 deliverable. Fail LOUD on any
    # partial arms list; the dispatcher runs q4 unsharded (cheap next to fits).
    missing_arms = [a for a in c.ARMS if a not in arms]
    if missing_arms:
        raise RuntimeError(
            f"step_q4 must run UNSHARDED over all {len(c.ARMS)} arms (missing "
            f"{missing_arms}): a sharded --arms invocation races on the combined "
            "cokernel_*.json writes (last writer wins, arms silently dropped)."
        )
    rb = c.load_rb_bank(d.layer)
    op_dir = c.data_out(out_root) / "operators"
    out = {"meta": c.repro_meta(), "layer": d.layer, "arms": {}}
    for arm in arms:
        W = torch.from_numpy(np.load(op_dir / f"W_{arm}_L{d.layer}.npy")).double()
        fitmeta = json.loads(
            (c.eval_out(out_root) / "fit_battery" / f"{arm}_L{d.layer}.json").read_text()
        )
        U, S, _V = c.svd_robust(W)
        ks = {
            "k50": _k_energy(S, 0.50),
            "k90": _k_energy(S, 0.90),
            "df": int(round(fitmeta["pooled"]["df"])),
        }
        rows: dict = {}
        for t, v in rb.items():
            vhat = torch.from_numpy(v / np.linalg.norm(v)).double()
            rows[t] = {name: float(1.0 - (U[:, :k].T @ vhat).norm() ** 2) for name, k in ks.items()}
        entry = {"k": ks, "cokernel_fraction": rows}
        if arm != "arm_context":
            entry["flag"] = "rank-limited — estimator kernel ⊇ (d − n_distinct) unobserved dims"
        out["arms"][arm] = entry
    # context arm: 3-λ sweep + fold jackknife stability
    n = len(d.fit_idx)
    pooled = fit_press_ext(d.X_ctx, d.Y, device)
    lam_star = pooled["lam"]
    sweep = {}
    for lam in (lam_star / 10, lam_star, lam_star * 10):
        Wl, _ = operator_raw(pooled, lam=lam)
        Ul, Sl, _ = c.svd_robust(Wl.cpu().double())
        k90 = _k_energy(Sl, 0.90)
        sweep[f"lam_{lam:g}"] = {
            t: float(
                1.0 - (Ul[:, :k90].T @ torch.from_numpy(v / np.linalg.norm(v)).double()).norm() ** 2
            )
            for t, v in rb.items()
        }
    jack = []
    for fi, te in enumerate(d.folds):
        tr = np.setdiff1d(np.arange(n), te)
        fitf = fit_press_ext(d.X_ctx[tr], d.Y[tr], device)
        Wf, _ = operator_raw(fitf)
        Uf, Sf, _ = c.svd_robust(Wf.cpu().double())
        k90 = _k_energy(Sf, 0.90)
        jack.append(
            {
                t: float(
                    1.0
                    - (Uf[:, :k90].T @ torch.from_numpy(v / np.linalg.norm(v)).double()).norm() ** 2
                )
                for t, v in rb.items()
            }
        )
        print(f"[p2-q4] jackknife fold {fi + 1}/{len(d.folds)}", flush=True)
    out["context_lambda_sweep_k90"] = sweep
    out["context_fold_jackknife_k90"] = jack
    for t in rb:
        vals = [sweep[k][t] for k in sweep]
        out.setdefault("context_sweep_spread", {})[t] = float(max(vals) - min(vals))
    for t, v in rb.items():
        c.write_json_atomic(
            c.eval_out(out_root) / "nullspace" / f"cokernel_{t}_L{d.layer}.json",
            {
                "meta": out["meta"],
                "trait": t,
                "layer": d.layer,
                "arms": {a: out["arms"][a]["cokernel_fraction"][t] for a in arms},
                "context_lambda_sweep_k90": {k: sweep[k][t] for k in sweep},
                "context_fold_jackknife_k90": [j[t] for j in jack],
                "flags": {a: out["arms"][a].get("flag") for a in arms},
            },
        )
    c.write_json_atomic(c.eval_out(out_root) / "nullspace" / f"cokernel_all_L{d.layer}.json", out)


# ── step: q5 endomorphism reads (context arm, ambient, gate-conditional) ─────


def step_q5(d: Designs, device: str, out_root: str | None) -> None:
    from scipy.optimize import linear_sum_assignment

    op_dir = c.data_out(out_root) / "operators"
    rb = c.load_rb_bank(d.layer)
    W = torch.from_numpy(np.load(op_dir / f"W_arm_context_L{d.layer}.npy")).double()
    b = torch.from_numpy(np.load(op_dir / f"b_arm_context_L{d.layer}.npy")).double()
    d_dim = W.shape[0]
    assert W.shape == (d_dim, d_dim), W.shape
    WtW = W.T @ W
    WWt = W @ W.T
    normality_gap = float((WtW - WWt).norm() / (W.norm() ** 2))
    Sym = 0.5 * (W + W.T)
    Asym = 0.5 * (W - W.T)
    out: dict = {
        "meta": c.repro_meta({"script": "issue1774_fit_battery.py --step q5"}),
        "layer": d.layer,
        "gate": {
            "normality_gap": normality_gap,
            "sym_energy_frac": float(Sym.norm() ** 2 / W.norm() ** 2),
            "antisym_energy_frac": float(Asym.norm() ** 2 / W.norm() ** 2),
        },
        "trace_over_d": float(torch.trace(W) / d_dim),
    }
    evals, evecs = c.eig_robust(W)
    cond_v = float(torch.linalg.cond(evecs).real)
    out["gate"]["eigvec_condition_number"] = cond_v
    gate_pass = normality_gap < 0.5 and np.isfinite(cond_v) and cond_v < 1e6
    out["gate"]["pass"] = bool(gate_pass)
    mags = evals.abs()
    out["spectral_radius"] = float(mags.max())
    if gate_pass:
        res = []
        top = torch.argsort(mags, descending=True)[: min(64, d_dim)]
        for j in top.tolist():
            g = evecs[:, j]
            r = ((W.to(evals.dtype) @ g) - evals[j] * g).norm() / g.norm()
            res.append(float(r.real))
        # fold-jackknife eigenvalue dispersion + Hungarian matching
        fold_lams = []
        n = len(d.fit_idx)
        for fi, te in enumerate(d.folds):
            tr = np.setdiff1d(np.arange(n), te)
            fitf = fit_press_ext(d.X_ctx[tr], d.Y[tr], device)
            Wf, _ = operator_raw(fitf)
            lf, _vf = c.eig_robust(Wf.cpu().double())
            fold_lams.append(lf)
            print(f"[p2-q5] fold eig {fi + 1}/{len(d.folds)}", flush=True)
        top_l = evals[top]
        matches = []
        for lf in fold_lams:
            topf = lf[torch.argsort(lf.abs(), descending=True)[: min(256, lf.shape[0])]]
            D = (top_l.unsqueeze(1) - topf.unsqueeze(0)).abs().numpy()
            ri, ci = linear_sum_assignment(D)
            matches.append({int(i): float(D[i, j]) for i, j in zip(ri, ci, strict=True)})
        stable = []
        for k in range(len(top)):
            dists = [m.get(k, np.inf) for m in matches]
            rel = [dd / max(1e-12, float(top_l[k].abs())) for dd in dists]
            n_ok = sum(1 for x in rel if x < 0.5)
            stable.append(
                {
                    "eig_re": float(top_l[k].real),
                    "eig_im": float(top_l[k].imag),
                    "residual": res[k],
                    "matched_rel_dists": [float(x) for x in rel],
                    "stable_5of6_rel_lt_0.5": bool(n_ok >= max(1, len(d.folds) - 1)),
                }
            )
        out["eigen"] = {
            "mass_re_pos": float((evals.real > 0).double().mean()),
            "n_mag_near_1": int(((mags - 1).abs() < 0.05).sum()),
            "n_mag_near_0": int((mags < 0.01).sum()),
            "complex_pair_fraction": float((evals.imag.abs() > 1e-9).double().mean()),
            "top_modes": stable,
            "stability_rule": "Hungarian |λi−λj| match to each fold operator; "
            "stable iff matched rel-dist<0.5 in ≥5/6 folds (raw dists persisted "
            "for re-thresholding)",
        }
        # almost-invariant clusters: gap-based |λ| clusters over the top modes
        sm = torch.sort(mags[top], descending=True).values
        gaps = (sm[:-1] - sm[1:]).numpy()
        cuts = [int(i) + 1 for i in np.argsort(gaps)[::-1][:3] if gaps[int(i)] > 0.01]
        out["eigen"]["magnitude_cluster_cuts_topk"] = sorted(cuts)
    else:
        out["eigen"] = {
            "skipped": "non-normality gate FAIL — singular-only fallback (registered outcome)"
        }
    # always-on reads
    Uc, Sc, Vc = c.svd_robust(W)
    cosgain = {}
    for name, v in list(rb.items()) + [
        (f"right_sv_{j}", Vc[j].numpy()) for j in range(min(TOP_SING_DIRS, Sc.shape[0]))
    ]:
        vt = torch.from_numpy(np.asarray(v, np.float64))
        vt = vt / vt.norm()
        Wv = W @ vt
        cosgain[name] = {
            "cos": float((Wv @ vt) / (Wv.norm() + 1e-30)),
            "gain": float(Wv.norm()),
        }
    out["cos_gain_map"] = cosgain
    B = torch.from_numpy(np.stack([rb[t] for t in c.TRAITS], axis=1)).double()
    Q, _ = torch.linalg.qr(B)
    G = Q.T @ W @ Q
    WQ = W @ Q
    in_span = Q @ (Q.T @ WQ)
    out["trait_gain_matrix"] = {
        "traits": list(c.TRAITS),
        "G": G.tolist(),
        "diag_energy": float((torch.diag(G) ** 2).sum()),
        "offdiag_energy": float((G**2).sum() - (torch.diag(G) ** 2).sum()),
        "out_of_span_energy": float(((WQ - in_span) ** 2).sum()),
    }
    if out["spectral_radius"] < 1.0:
        x_star = torch.linalg.solve(torch.eye(d_dim, dtype=torch.float64) - W, b)
        out["fixed_point"] = {"regularized": False}
    else:
        x_star = torch.linalg.solve(
            torch.eye(d_dim, dtype=torch.float64)
            - W
            + 1e-3 * torch.eye(d_dim, dtype=torch.float64),
            b,
        )
        out["fixed_point"] = {"regularized": True, "ridge": 1e-3}
    pool_norms = np.linalg.norm(d.Y, axis=1)
    out["fixed_point"].update(
        {
            "norm": float(x_star.norm()),
            "answer_pool_norm_p10_p50_p90": [
                float(np.percentile(pool_norms, q)) for q in (10, 50, 90)
            ],
        }
    )
    np.save(op_dir / f"x_star_L{d.layer}.npy", x_star.numpy().astype(np.float32))
    c.write_json_atomic(c.eval_out(out_root) / "endomorphism" / f"context_L{d.layer}.json", out)
    print(f"[p2-q5] gate_pass={gate_pass} normality_gap={normality_gap:.3f}")


# ── step: decode (logit lens; on-pod while weights resident) ─────────────────


def step_decode(d: Designs, device: str, out_root: str | None) -> None:
    from issue1415_logit_lens import compute_lens
    from issue1774_draws import _load_hf_model

    op_dir = c.data_out(out_root) / "operators"
    model, tok = _load_hf_model(device)
    vectors: dict[str, torch.Tensor] = {}
    for arm in c.ARMS:
        W = torch.from_numpy(np.load(op_dir / f"W_{arm}_L{d.layer}.npy")).float()
        U, S, Vh = c.svd_robust(W.double())
        for j in range(min(10, S.shape[0])):
            vectors[f"{arm}_left_sv{j}"] = U[:, j].float()
            vectors[f"{arm}_right_sv{j}"] = Vh[j].float()
    xs = op_dir / f"x_star_L{d.layer}.npy"
    if xs.exists():
        vectors["x_star"] = torch.from_numpy(np.load(xs)).float()
    lens = compute_lens(model, vectors, top_k=20, tokenizer=tok)
    rb = c.load_rb_bank(d.layer)
    cos_rb = {
        name: {
            t: float(
                np.dot(v.numpy(), rb[t])
                / (np.linalg.norm(v.numpy()) * np.linalg.norm(rb[t]) + 1e-30)
            )
            for t in rb
        }
        for name, v in vectors.items()
    }
    knn_digest = None
    if xs.exists():
        x = np.load(xs).astype(np.float64)
        Yn = d.Y / np.linalg.norm(d.Y, axis=1, keepdims=True)
        sims = Yn @ (x / np.linalg.norm(x))
        top = np.argsort(sims)[::-1][:10]
        knn_digest = [
            {
                "manifest_index": int(d.fit_idx[i]),
                "prefix_id": str(d.prefix_ids[i]),
                "query_id": str(d.query_ids[i]),
                "cos": float(sims[i]),
            }
            for i in top
        ]
    c.write_json_atomic(
        c.eval_out(out_root) / "endomorphism" / f"decode_L{d.layer}.json",
        {
            "meta": c.repro_meta({"script": "issue1774_fit_battery.py --step decode"}),
            "lens": lens,
            "cos_vs_rb": cos_rb,
            "x_star_knn_digest": knn_digest,
            "tuned_lens": "skipped — logit lens is the named primary decode "
            "(allowed deviation: dropping the tuned-lens robustness read)",
        },
    )


# ── step: P3 direction sets ──────────────────────────────────────────────────


def step_directions(d: Designs, device: str, out_root: str | None) -> None:
    op_dir = c.data_out(out_root) / "operators"
    W = torch.from_numpy(np.load(op_dir / f"W_arm_context_L{d.layer}.npy")).double()
    U, S, Vh = c.svd_robust(W)
    rb = c.load_rb_bank(d.layer)
    k90 = _k_energy(S, 0.90)
    null_path = (
        c.data_out(out_root)
        / "analysis_tensors/channels_null"
        / (f"r2_null_arm_context_L{d.layer}.npy")
    )
    directions: dict[str, torch.Tensor] = {}
    for j in range(min(4, S.shape[0])):
        directions[f"top_sv{j}"] = Vh[j].clone()
    rank_floor = min(3000, S.shape[0] - 4)
    tail_start = max(k90 + 1, rank_floor)
    tail_idx = list(range(tail_start, min(tail_start + 4, S.shape[0])))
    for i, j in enumerate(tail_idx):
        directions[f"kernel_tail{i}"] = Vh[j].clone()
    for t, v in rb.items():
        vt = torch.from_numpy(v).double()
        directions[f"rb_{t}"] = vt / vt.norm()
    # norm-matched random: N(0, Σ_ctx) shrunk + renormed (#778 randnorm convention)
    Xc = torch.from_numpy(d.X_ctx).double()
    Xc = Xc - Xc.mean(0, keepdim=True)
    gen = torch.Generator().manual_seed(c.SEED_DRAWS)
    for i in range(4):
        z = torch.randn(Xc.shape[0], generator=gen).double()
        v = (Xc.T @ z) / Xc.shape[0]
        directions[f"random{i}"] = v / v.norm()
    # α: ‖αv‖ = p90 of within-corpus context-state displacement along top sv
    proj = (Xc @ Vh[0]).abs()
    alpha = float(np.percentile(proj.numpy(), 90))
    payload = {
        "directions": {k: v.float() for k, v in directions.items()},
        "alpha": alpha,
        "layer": d.layer,
        "k90": int(k90),
        "kernel_tail_sv_ranks": tail_idx,
        "sigma_at_tail": [float(S[j]) for j in tail_idx],
        "meta": c.repro_meta(
            {
                "note": "alpha = p90 |(x-mean)·u1|; ungrounded — "
                "P3 fluency pilot adjusts (halve once)"
            }
        ),
        "null_matrix_present": null_path.exists(),
    }
    out_p = c.data_out(out_root) / "directions.pt"
    torch.save(payload, out_p)
    print(f"[p2-directions] wrote {out_p} alpha={alpha:.3f} k90={k90} n={len(directions)}")


# ── main ─────────────────────────────────────────────────────────────────────


STEPS = ("fits", "parity", "q1a", "q1b", "q3", "q3angles", "q4", "q5", "decode", "directions")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", default="all", choices=[*STEPS, "all"])
    ap.add_argument("--layers", default="14,18,19")
    ap.add_argument("--arms", default=",".join(c.ARMS))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--n-perm-draws", type=int, default=c.N_PERM_DRAWS)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.import_check:
        from scipy.optimize import linear_sum_assignment  # noqa: F401

        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from issue1092_partb_operator import (  # noqa: F401
            _angle_null_band,
            _angles_between,
            _operator_raw,
        )
        from issue1415_logit_lens import compute_lens  # noqa: F401
        from issue1774_draws import _load_hf_model  # noqa: F401
        from issue923_fit_decomposition import PressRidge  # noqa: F401

        print("[import-check] p2 deferred imports resolve")
        return 0
    n_draws = 8 if args.smoke and args.n_perm_draws == c.N_PERM_DRAWS else args.n_perm_draws
    layers = [int(x) for x in args.layers.split(",") if x]
    arms = [a for a in args.arms.split(",") if a]
    for layer in layers:
        print(f"[phase=p2 step={args.step} L={layer}]", flush=True)
        d = Designs(layer, args.smoke, args.out_root)
        run = [args.step] if args.step != "all" else list(STEPS)
        # pilot timing: one full-shape fold fit timed before the battery (plan §9)
        if "fits" in run:
            t0 = time.time()
            step_fits(d, arms, args.device, args.out_root)
            print(f"[p2-pilot] fits wall {time.time() - t0:.0f}s (per-call basis recorded)")
        if "parity" in run and layer == c.HEADLINE_LAYER:
            step_parity(d, args.device, args.out_root)
        if "q3" in run:
            step_q3(d, arms, args.device, args.out_root, n_draws)
        if "q3angles" in run:
            step_q3_angles(d, args.device, args.out_root, n_draws)
        if "q1a" in run and layer == c.HEADLINE_LAYER:
            step_q1a(d, args.device, args.out_root, n_draws)
        if "q1b" in run and layer == c.HEADLINE_LAYER:
            step_q1b(d, args.device, args.out_root)
        if "q4" in run:
            step_q4(d, arms, args.device, args.out_root)
        if "q5" in run and layer == c.HEADLINE_LAYER:
            step_q5(d, args.device, args.out_root)
        if "decode" in run and layer == c.HEADLINE_LAYER:
            step_decode(d, args.device, args.out_root)
        if "directions" in run and layer == c.HEADLINE_LAYER:
            step_directions(d, args.device, args.out_root)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
