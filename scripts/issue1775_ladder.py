#!/usr/bin/env python3
"""#1775 P1 (--phase linear) + P3 (--phase nonlinear): estimation ladder, all arms.

P1: PRESS-ridge (the banked #1092 engine, reused verbatim through
``press_fit_predict``) on the five input arms at L14/cell_inst_own (primary,
ambient + pca48), with per-row held-out predictions persisted (fp16), the
standing mapping-baselines pair per fit, the lambda discipline (per-lambda R2
+ x10 / /10 sensitivity + df(lambda)), Gate C (banked context reproduction
|delta| <= 0.02), the averaged-grain fits, and the L19 / cell_pre_own
expansion combos routed through ``fit_h.ridge_fit_predict_fast`` behind the
docstring-mandated >=3-slice slow-vs-fast parity gate (<= 1e-4 max rel diff;
PRESS fallback on gate failure).

P3: nonlinear rungs under IDENTICAL folds + nested group-respecting inner
tuning: exact-RBF KRR (median-heuristic gamma x {0.25..4}, lambda grid),
RFF ridge (D=16384 features of the KRR-selected gamma), and the #779 MLP
recipe verbatim (w=8192, lr 3e-4, wd 1e-4, AdamW, full-batch, patience 20,
max 300 epochs) as a GROUPS-BATCHED trainer (folds x seeds stacked — no
serial per-fit loop) with a group-respecting early-stop split. Gains vs the
P1 ridge with paired CLUSTER bootstrap CIs (resampling unit = the fold
scheme's grouping unit).

Units checkpoint to a per-shard JSONL the moment they complete (resume keyed
on every output-affecting regime field); one stdout line per unit.
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
    ARMS,
    CELL_PRIMARY,
    GATE_C,
    GATE_C_TOL,
    KRR_GAMMA_MULTS,
    KRR_LAMBDAS,
    LAYER_BRIDGE,
    LAYER_PRIMARY,
    MLP_LR,
    MLP_MAX_EPOCHS,
    MLP_PATIENCE,
    MLP_WD,
    MLP_WIDTH,
    RFF_DIM,
    STITCH_REPRO_PCA48,
    ArmData,
    _basis_targets_with_info,
    _r2,
    append_unit,
    atomic_write_json,
    build_arm_data,
    cluster_bootstrap_delta_r2,
    eigh_robust,
    eval_dir,
    fit_press_pairs,
    fold_pairs,
    inner_val_split,
    load_units_validated,
    mean_cosine,
    per_fit_baselines,
    resolve_store_dir,
    restrict_pairs,
    result_meta,
    stage_store_if_needed,
    tensors_dir,
    unit_key,
    upload_phase_eval_json,
    upload_phase_tensors,
)

# fit_h fast twins (main-resident; parity gate mandated by their docstrings).
from explore_persona_space.experiments.issue_779.fit_h import (  # noqa: E402
    ridge_fit_predict,
    ridge_fit_predict_fast,
)

REGIME_KEYS = (
    "phase",
    "cell",
    "layer",
    "basis",
    "arm",
    "grain",
    "scheme",
    "rung",
    "seed",
    "smoke",
    "row_limit",
)
PILOT_BOOKED_WALL_H = {"linear": 1.0, "nonlinear": 2.5}  # plan section 9 rows
PILOT_GATE_RC = 7
# Designed halt: the final-JSON assembly refuses to write a PARTIAL payload
# (planned units missing a completed row) — the P3 crash-fix class where
# 3/38 units would otherwise assemble into nonlinear_fits.json silently.
ASSEMBLY_INCOMPLETE_RC = 23


def _device(arg: str) -> str:
    if arg == "cuda" and not torch.cuda.is_available():
        print("[ladder] --device cuda requested but unavailable; using cpu", flush=True)
        return "cpu"
    return arg


def _basis_variants(Y: np.ndarray, bases: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for b in bases:
        Yb, _info = _basis_targets_with_info(
            Y, b, hidden_dim=3584, targets=["t1", "t2", "t3"], projection_target="t1"
        )
        out[b] = np.ascontiguousarray(Yb, dtype=np.float64)
    return out


# ── averaged grain ───────────────────────────────────────────────────────────────


def averaged_grain_data(ad: ArmData, arm: str, min_rows: int = 3):
    """Per-prefix averaged inputs + targets (n ~= 996 prefixes)."""
    groups: dict[str, list[int]] = {}
    for i, pid in enumerate(ad.prefix_ids):
        groups.setdefault(str(pid), []).append(i)
    pids = sorted(p for p, idx in groups.items() if len(idx) >= min_rows)
    src = ad.X["prefix_end"] if arm == "prefix_end" else ad.X["context_end"]
    X_avg = np.stack([src[groups[p]].mean(0) for p in pids], axis=0)
    Y_avg = np.stack([ad.Y_stacked[groups[p]].mean(0) for p in pids], axis=0)
    rows = [{"prefix_id": p, "query_id": p} for p in pids]
    return X_avg, Y_avg, rows, pids


# ── KRR / RFF / MLP engines ──────────────────────────────────────────────────────


def _standardize_train(X: np.ndarray, tr: np.ndarray):
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-9
    return (X - mu) / sd


def _median_sigma(Xs: np.ndarray, idx: np.ndarray, *, cap: int = 2048, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    sub = idx if idx.size <= cap else rng.choice(idx, size=cap, replace=False)
    A = Xs[sub]
    d2 = (A**2).sum(1)[:, None] + (A**2).sum(1)[None, :] - 2 * (A @ A.T)
    d = np.sqrt(np.clip(d2, 0, None))
    off = d[np.triu_indices_from(d, k=1)]
    pos = off[off > 0]
    return float(np.median(pos)) if pos.size else 1.0


def _sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return d2.clamp_(min=0.0)


def _kernel_ridge_solve(
    K_tr: torch.Tensor, Y_tr: torch.Tensor, K_ev: torch.Tensor, lambdas
) -> dict[float, torch.Tensor]:
    """eigh once, predictions for every lambda (per-lambda dual solves)."""
    w, V = eigh_robust(K_tr)
    w = w.clamp(min=0.0)
    VtY = V.T @ Y_tr
    KevV = K_ev @ V
    out = {}
    for lam in lambdas:
        filt = 1.0 / (w + float(lam))
        out[float(lam)] = KevV @ (filt[:, None] * VtY)
    return out


def krr_fit(
    X: np.ndarray,
    Y_by_basis: dict[str, np.ndarray],
    pairs,
    groups: np.ndarray,
    *,
    device: str,
    select_basis: str = "pca48",
) -> dict:
    """Exact-RBF KRR with nested (gamma, lambda) selection on a group-respecting
    inner split; one eigh per (fold, gamma) for selection + one refit eigh."""
    dev = torch.device(device)
    dt = torch.float32 if dev.type == "cuda" else torch.float64
    preds = {b: np.zeros_like(Y_by_basis[b]) for b in Y_by_basis}
    covered = np.zeros(X.shape[0], dtype=bool)
    per_fold = []
    for fold_i, (tr, te) in enumerate(pairs):
        itr, ival = inner_val_split(tr, groups, seed=fold_i)
        Xs = _standardize_train(X, tr)
        sigma0 = _median_sigma(Xs, itr, seed=fold_i)
        Xt = torch.from_numpy(Xs).to(dev, dt)
        Ysel = torch.from_numpy(Y_by_basis[select_basis]).to(dev, dt)
        mu_sel = Ysel[itr].mean(0, keepdim=True)
        d2_ii = _sq_dists(Xt[itr], Xt[itr])
        d2_vi = _sq_dists(Xt[ival], Xt[itr])
        best = (None, None, float("inf"))
        for mult in KRR_GAMMA_MULTS:
            g = mult / (2.0 * sigma0**2)
            K_ii = torch.exp(-g * d2_ii)
            K_vi = torch.exp(-g * d2_vi)
            sol = _kernel_ridge_solve(K_ii, Ysel[itr] - mu_sel, K_vi, KRR_LAMBDAS)
            for lam, p in sol.items():
                mse = float(((p + mu_sel - Ysel[ival]) ** 2).mean().item())
                if mse < best[2]:
                    best = (mult, lam, mse)
        mult_s, lam_s, _ = best
        g = mult_s / (2.0 * sigma0**2)
        del d2_ii, d2_vi
        K_tt = torch.exp(-g * _sq_dists(Xt[tr], Xt[tr]))
        K_et = torch.exp(-g * _sq_dists(Xt[te], Xt[tr]))
        w, V = eigh_robust(K_tt)
        w = w.clamp(min=0.0)
        filt = 1.0 / (w + float(lam_s))
        for b, Yb in Y_by_basis.items():
            Yt = torch.from_numpy(Yb).to(dev, dt)
            mu = Yt[tr].mean(0, keepdim=True)
            alpha = V @ (filt[:, None] * (V.T @ (Yt[tr] - mu)))
            preds[b][te] = (K_et @ alpha + mu).double().cpu().numpy()
        covered[te] = True
        per_fold.append(
            {
                "fold": fold_i,
                "gamma_mult": mult_s,
                "lambda": lam_s,
                "sigma0": sigma0,
                "r2_fold": {b: _r2(Y_by_basis[b][te], preds[b][te]) for b in Y_by_basis},
            }
        )
        del Xt, K_tt, K_et, w, V
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return {"preds": preds, "covered": covered, "per_fold": per_fold}


def rff_fit(
    X: np.ndarray,
    Y_by_basis: dict[str, np.ndarray],
    pairs,
    groups: np.ndarray,
    *,
    device: str,
    gammas_per_fold: list[dict],
    seed: int,
    select_basis: str = "pca48",
) -> dict:
    """Random-Fourier-feature ridge (D=16384) of the KRR-selected per-fold RBF
    kernel; lambda selected on the group-respecting inner split (Gram space)."""
    dev = torch.device(device)
    dt = torch.float32 if dev.type == "cuda" else torch.float64
    preds = {b: np.zeros_like(Y_by_basis[b]) for b in Y_by_basis}
    covered = np.zeros(X.shape[0], dtype=bool)
    per_fold = []
    for fold_i, (tr, te) in enumerate(pairs):
        itr, ival = inner_val_split(tr, groups, seed=fold_i)
        Xs = _standardize_train(X, tr)
        info = gammas_per_fold[fold_i]
        g = info["gamma_mult"] / (2.0 * info["sigma0"] ** 2)
        rng = np.random.default_rng(1000 * seed + fold_i)
        W = rng.normal(0.0, np.sqrt(2.0 * g), size=(X.shape[1], RFF_DIM))
        b0 = rng.uniform(0.0, 2 * np.pi, size=RFF_DIM)
        Xt = torch.from_numpy(Xs).to(dev, dt)
        Wt = torch.from_numpy(W).to(dev, dt)
        bt = torch.from_numpy(b0).to(dev, dt)
        Z = torch.cos(Xt @ Wt + bt) * np.sqrt(2.0 / RFF_DIM)
        Ysel = torch.from_numpy(Y_by_basis[select_basis]).to(dev, dt)
        mu_sel = Ysel[itr].mean(0, keepdim=True)
        G_ii = Z[itr] @ Z[itr].T
        K_vi = Z[ival] @ Z[itr].T
        sol = _kernel_ridge_solve(G_ii, Ysel[itr] - mu_sel, K_vi, KRR_LAMBDAS)
        best_lam, best_mse = None, float("inf")
        for lam, p in sol.items():
            mse = float(((p + mu_sel - Ysel[ival]) ** 2).mean().item())
            if mse < best_mse:
                best_lam, best_mse = lam, mse
        G_tt = Z[tr] @ Z[tr].T
        K_et = Z[te] @ Z[tr].T
        w, V = eigh_robust(G_tt)
        w = w.clamp(min=0.0)
        filt = 1.0 / (w + float(best_lam))
        for b, Yb in Y_by_basis.items():
            Yt = torch.from_numpy(Yb).to(dev, dt)
            mu = Yt[tr].mean(0, keepdim=True)
            alpha = V @ (filt[:, None] * (V.T @ (Yt[tr] - mu)))
            preds[b][te] = (K_et @ alpha + mu).double().cpu().numpy()
        covered[te] = True
        per_fold.append({"fold": fold_i, "lambda": best_lam, "seed": seed})
        del Xt, Z, G_tt, K_et, w, V
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return {"preds": preds, "covered": covered, "per_fold": per_fold}


def mlp_fit_groups(
    X: np.ndarray,
    Ypca: np.ndarray,
    pairs,
    groups: np.ndarray,
    seeds: list[int],
    *,
    device: str,
    width: int = MLP_WIDTH,
    lr: float = MLP_LR,
    wd: float = MLP_WD,
    max_epochs: int = MLP_MAX_EPOCHS,
    patience: int = MLP_PATIENCE,
    max_group_batch: int = 6,
) -> dict:
    """#779 MLP recipe verbatim (GELU 1-hidden, AdamW full-batch, patience-based
    early stop restoring the best state) as a GROUPS-BATCHED trainer over
    (fold x seed) — two deviations REQUIRED by the plan: the early-stop split is
    GROUP-respecting (plan section 4 P3 nested tuning), and seeds thread the init.
    """
    dev = torch.device(device)
    units = [(fi, s) for s in seeds for fi in range(len(pairs))]
    preds = {s: np.zeros_like(Ypca) for s in seeds}
    covered = np.zeros(X.shape[0], dtype=bool)
    epochs_ran: dict[tuple[int, int], int] = {}
    for lo in range(0, len(units), max_group_batch):
        batch = units[lo : lo + max_group_batch]
        prep = []
        for fi, s in batch:
            tr, te = pairs[fi]
            itr, ival = inner_val_split(tr, groups, seed=s * 100 + fi)
            Xs = _standardize_train(X, tr).astype(np.float32)
            ymu = Ypca[tr].mean(0)
            prep.append(
                {
                    "fi": fi,
                    "seed": s,
                    "itr": itr,
                    "ival": ival,
                    "te": te,
                    "Xs": Xs,
                    "T": (Ypca - ymu).astype(np.float32),
                    "ymu": ymu,
                }
            )
        G = len(prep)
        d_in = X.shape[1]
        p_out = Ypca.shape[1]
        n_tr_max = max(len(pp["itr"]) for pp in prep)
        n_va_max = max(len(pp["ival"]) for pp in prep)
        Xp = torch.zeros((G, n_tr_max, d_in), device=dev)
        Tp = torch.zeros((G, n_tr_max, p_out), device=dev)
        wtr = torch.zeros((G, n_tr_max, 1), device=dev)
        Xv = torch.zeros((G, n_va_max, d_in), device=dev)
        Tv = torch.zeros((G, n_va_max, p_out), device=dev)
        wva = torch.zeros((G, n_va_max, 1), device=dev)
        den_tr = torch.zeros(G, device=dev)
        den_va = torch.zeros(G, device=dev)
        for gi, pp in enumerate(prep):
            ntr, nva = len(pp["itr"]), len(pp["ival"])
            Xp[gi, :ntr] = torch.from_numpy(pp["Xs"][pp["itr"]]).to(dev)
            Tp[gi, :ntr] = torch.from_numpy(pp["T"][pp["itr"]]).to(dev)
            wtr[gi, :ntr] = 1.0
            Xv[gi, :nva] = torch.from_numpy(pp["Xs"][pp["ival"]]).to(dev)
            Tv[gi, :nva] = torch.from_numpy(pp["T"][pp["ival"]]).to(dev)
            wva[gi, :nva] = 1.0
            den_tr[gi], den_va[gi] = float(ntr * p_out), float(nva * p_out)
        W1 = torch.empty((G, width, d_in), device=dev)
        b1 = torch.empty((G, width), device=dev)
        W2 = torch.zeros((G, p_out, width), device=dev)
        b2 = torch.zeros((G, p_out), device=dev)
        for gi, pp in enumerate(prep):
            torch.manual_seed(pp["seed"])
            net = torch.nn.Sequential(
                torch.nn.Linear(d_in, width), torch.nn.GELU(), torch.nn.Linear(width, p_out)
            )
            W1[gi] = net[0].weight.detach().to(dev)
            b1[gi] = net[0].bias.detach().to(dev)
            W2[gi] = net[2].weight.detach().to(dev)
            b2[gi] = net[2].bias.detach().to(dev)
        for w_ in (W1, b1, W2, b2):
            w_.requires_grad_(True)
        opt = torch.optim.AdamW([W1, b1, W2, b2], lr=lr, weight_decay=wd)
        best_val = torch.full((G,), float("inf"), device=dev)
        bad = torch.zeros(G, dtype=torch.long, device=dev)
        frozen = torch.zeros(G, dtype=torch.bool, device=dev)
        best_state: list[tuple | None] = [None] * G
        eran = np.zeros(G, dtype=int)
        active = torch.ones(G, device=dev)
        for ep in range(max_epochs):
            opt.zero_grad(set_to_none=True)
            h1 = torch.nn.functional.gelu(torch.baddbmm(b1.unsqueeze(1), Xp, W1.transpose(1, 2)))
            out = torch.baddbmm(b2.unsqueeze(1), h1, W2.transpose(1, 2))
            loss_pg = (((out - Tp) ** 2) * wtr).sum(dim=(1, 2)) / den_tr
            (loss_pg * active).sum().backward()
            opt.step()
            with torch.no_grad():
                h1e = torch.nn.functional.gelu(
                    torch.baddbmm(b1.unsqueeze(1), Xv, W1.transpose(1, 2))
                )
                oute = torch.baddbmm(b2.unsqueeze(1), h1e, W2.transpose(1, 2))
                val_pg = (((oute - Tv) ** 2) * wva).sum(dim=(1, 2)) / den_va
            improved = (val_pg < best_val - 1e-6) & (~frozen)
            for gi in torch.nonzero(improved).ravel().tolist():
                best_state[gi] = tuple(t[gi].detach().clone() for t in (W1, b1, W2, b2))
            best_val = torch.where(improved, val_pg, best_val)
            bad = torch.where(improved, torch.zeros_like(bad), bad + (~frozen).long())
            frozen |= (bad >= patience) & (~frozen)
            active = (~frozen).float()
            eran[(~frozen).cpu().numpy()] = ep + 1
            if frozen.all():
                break
        with torch.no_grad():
            for gi, pp in enumerate(prep):
                st = best_state[gi] or tuple(t[gi].detach() for t in (W1, b1, W2, b2))
                w1, bb1, w2, bb2 = (t.to(dev) for t in st)
                Xe = torch.from_numpy(pp["Xs"][pp["te"]]).to(dev)
                h = torch.nn.functional.gelu(Xe @ w1.T + bb1)
                pe = (h @ w2.T + bb2).cpu().numpy() + pp["ymu"]
                preds[pp["seed"]][pp["te"]] = pe
                covered[pp["te"]] = True
                epochs_ran[(pp["fi"], pp["seed"])] = int(eran[gi])
        del Xp, Tp, Xv, Tv, W1, b1, W2, b2
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[mlp] group batch {lo // max_group_batch + 1} done ({G} fits)", flush=True)
    return {
        "preds_by_seed": preds,
        "covered": covered,
        "epochs_ran": {f"{k[0]}_{k[1]}": v for k, v in epochs_ran.items()},
    }


# ── parity gate for the fast ridge twin ──────────────────────────────────────────


def fast_ridge_parity_gate(slices: list[tuple[np.ndarray, np.ndarray, np.ndarray]], device: str):
    """>=3-slice slow(SVD)-vs-fast(Gram-eigh) parity at production shape
    (fit_h docstring contract; tolerance 1e-4 with ~6x headroom vs the measured
    1.7e-5 at n~4k). Returns (passed, report)."""
    diffs = []
    for Xtr, Ytr, Xev in slices:
        slow = ridge_fit_predict(Xtr, Ytr, Xev)
        fast = ridge_fit_predict_fast(Xtr, Ytr, Xev, device=device)
        denom = np.maximum(np.abs(slow), 1e-9)
        diffs.append(float(np.max(np.abs(slow - fast) / denom)))
    passed = all(d <= 1e-4 for d in diffs)
    return passed, {"max_rel_diffs": diffs, "tolerance": 1e-4, "passed": passed}


def fast_ridge_pairs(X, Y, pairs, *, device: str):
    pred = np.zeros_like(Y)
    covered = np.zeros(X.shape[0], dtype=bool)
    fold_r2 = []
    lambda_info = []
    for tr, te in pairs:
        pred[te], info = ridge_fit_predict_fast(
            X[tr], Y[tr], X[te], device=device, return_info=True
        )
        covered[te] = True
        fold_r2.append(_r2(Y[te], pred[te]))
        lambda_info.append(info)
    cov = np.nonzero(covered)[0]
    return (
        {
            "r2": _r2(Y[cov], pred[cov]),
            "r2_folds": fold_r2,
            # round-2 Minor-c: lambda discipline on fast-engine (secondary) units —
            # per-fold GCV-selected lambda* + df(lambda*); the full per-lambda R2 +
            # x10 / /10 sensitivity read lives on the PRESS primary combos.
            "lambda_star_folds": [i["best_lambda"] for i in lambda_info],
            "df_lambda_star_folds": [i["dof"] for i in lambda_info],
            "lambda_discipline_note": "GCV-internal engine; +-10x sensitivity reported on PRESS primaries",
        },
        pred,
        covered,
    )


# ── unit enumeration ─────────────────────────────────────────────────────────────


def linear_units(cells: list[str], smoke: bool) -> list[dict]:
    units: list[dict] = []
    bases = ["ambient", "pca48"]
    arms_smoke = ("context_end", "stitch") if smoke else ARMS
    for basis in bases if not smoke else ["pca48"]:
        for arm in arms_smoke:
            units.append(
                dict(
                    cell=CELL_PRIMARY,
                    layer=LAYER_PRIMARY,
                    basis=basis,
                    arm=arm,
                    grain="perrow",
                    scheme="prefix",
                    rung="ridge",
                    seed=0,
                    engine="press",
                )
            )
        for arm in ("bare_query", "stitch"):
            if arm in arms_smoke:
                units.append(
                    dict(
                        cell=CELL_PRIMARY,
                        layer=LAYER_PRIMARY,
                        basis=basis,
                        arm=arm,
                        grain="perrow",
                        scheme="query",
                        rung="ridge",
                        seed=0,
                        engine="press",
                    )
                )
        if "stitch" in arms_smoke:
            units.append(
                dict(
                    cell=CELL_PRIMARY,
                    layer=LAYER_PRIMARY,
                    basis=basis,
                    arm="stitch",
                    grain="perrow",
                    scheme="doubly",
                    rung="ridge",
                    seed=0,
                    engine="press",
                )
            )
        for arm in ("prefix_end", "query_averaged"):
            if smoke and arm != "prefix_end":
                continue
            units.append(
                dict(
                    cell=CELL_PRIMARY,
                    layer=LAYER_PRIMARY,
                    basis=basis,
                    arm=arm,
                    grain="averaged",
                    scheme="prefix",
                    rung="ridge",
                    seed=0,
                    engine="press",
                )
            )
    if not smoke:
        for arm in ("prefix_end", "context_end", "query_averaged"):
            units.append(
                dict(
                    cell=CELL_PRIMARY,
                    layer=LAYER_BRIDGE,
                    basis="pca48",
                    arm=arm,
                    grain="perrow",
                    scheme="prefix",
                    rung="ridge",
                    seed=0,
                    engine="fast",
                )
            )
        for cell in cells:
            if cell == CELL_PRIMARY:
                continue
            for basis in bases:
                for arm in ("prefix_end", "context_end", "query_averaged"):
                    units.append(
                        dict(
                            cell=cell,
                            layer=LAYER_PRIMARY,
                            basis=basis,
                            arm=arm,
                            grain="perrow",
                            scheme="prefix",
                            rung="ridge",
                            seed=0,
                            engine="fast",
                        )
                    )
    for u in units:
        u["phase"] = "linear"
    return units


def nonlinear_units(smoke: bool, seeds: list[int], gate_b_skip: set[str]) -> list[dict]:
    """(arm, grain, scheme, rung) units. Gate B: arms whose detection came back
    all-null skip the MLP battery (KRR runs once as confirmation)."""
    combos = [
        ("prefix_end", "perrow", "prefix"),
        ("query_averaged", "perrow", "prefix"),
        ("bare_query", "perrow", "prefix"),
        ("bare_query", "perrow", "query"),
        ("stitch", "perrow", "prefix"),
        ("stitch", "perrow", "query"),
        ("prefix_end", "averaged", "prefix"),
        ("query_averaged", "averaged", "prefix"),
    ]
    if smoke:
        combos = [("stitch", "perrow", "prefix")]
    units = []
    for arm, grain, scheme in combos:
        units.append(
            dict(
                cell=CELL_PRIMARY,
                layer=LAYER_PRIMARY,
                basis="both",
                arm=arm,
                grain=grain,
                scheme=scheme,
                rung="krr",
                seed=0,
                engine="krr",
            )
        )
        for s in seeds:
            units.append(
                dict(
                    cell=CELL_PRIMARY,
                    layer=LAYER_PRIMARY,
                    basis="both",
                    arm=arm,
                    grain=grain,
                    scheme=scheme,
                    rung="rff",
                    seed=s,
                    engine="rff",
                )
            )
        if grain == "perrow" and arm not in gate_b_skip:
            units.append(
                dict(
                    cell=CELL_PRIMARY,
                    layer=LAYER_PRIMARY,
                    basis="pca48",
                    arm=arm,
                    grain=grain,
                    scheme=scheme,
                    rung="mlp",
                    seed=-1,
                    engine="mlp",
                )
            )
    for u in units:
        u["phase"] = "nonlinear"
    return units


def gate_b_skips(detection_json: Path) -> set[str]:
    """Arms whose HSIC AND dCor Holm-adjusted p > 0.05 on ALL 3 schemes (Gate B)."""
    if not detection_json.exists():
        print("[gate-b] detection JSON absent — running the full MLP battery", flush=True)
        return set()
    d = json.loads(detection_json.read_text())
    skips = set()
    holm = d.get("holm_adjusted_p", {})
    arms = {k.split("|")[0] for k in holm}
    for arm in arms:
        ps = [v for k, v in holm.items() if k.startswith(f"{arm}|")]
        if ps and all(p > 0.05 for p in ps):
            skips.add(arm)
    if skips:
        print(f"[gate-b] MLP battery skipped for all-null arms: {sorted(skips)}", flush=True)
    return skips


def unit_row_incomplete(d: dict) -> str | None:
    """Reason a persisted unit row is NOT a complete result (None = complete).

    A unit counts as done ONLY when its JSONL row carries the full result
    payload the assembly reads (#1775 P3 crash-fix): a stub / truncated /
    older-schema row must re-run, never resume-skip. NOTE: rff per-fold
    entries carry {fold, lambda, seed} BY DESIGN (no per-fold r2 — fold R2
    lives in the persisted preds); rff completeness keys on the top-level
    ``r2`` + ``wall_s``, not on a krr-shaped per_fold.
    """
    for k in REGIME_KEYS:
        if k not in d:
            return f"missing regime key {k!r}"
    if "wall_s" not in d:
        return "missing wall_s (unit never completed)"
    rung = d.get("rung")
    if rung == "mlp":
        if not d.get("r2_by_seed") or "r2_seed_mean" not in d:
            return "mlp row without r2_by_seed/r2_seed_mean"
    elif rung in ("krr", "rff"):
        pf = d.get("per_fold")
        if not isinstance(pf, list) or not pf:
            return f"{rung} row without per_fold"
        if rung == "krr" and any("r2_fold" not in e for e in pf):
            return "krr row with per_fold entries missing r2_fold"
        if rung == "rff" and any("lambda" not in e for e in pf):
            return "rff row with per_fold entries missing lambda"
        if not isinstance(d.get("r2"), dict) or not d["r2"]:
            return f"{rung} row without top-level r2"
    elif "r2" not in d:  # linear ridge rows
        return "ridge row without r2"
    return None


def shard_units(units: list[dict], phase: str, num_shards: int, shard_index: int) -> list[dict]:
    """Shard assignment. NONLINEAR shards at the (arm, grain, scheme) GROUP
    grain: every rung of one group lands on ONE shard, in the enumeration's
    krr -> rff -> mlp order, because rff/mlp READ the krr gamma record — the
    index-interleaved split raced the record across concurrent shards and
    crashed both (the P3 production crash). Groups round-robin over shards
    (production: 8 groups -> 2 shards x 4 groups, identical rung mix).
    LINEAR keeps the index grain (its units are independent).
    """
    if phase != "nonlinear" or num_shards <= 1:
        return [u for i, u in enumerate(units) if i % num_shards == shard_index]
    group_order: list[tuple] = []
    for u in units:
        g = (u["arm"], u["grain"], u["scheme"])
        if g not in group_order:
            group_order.append(g)
    gidx = {g: i for i, g in enumerate(group_order)}
    return [
        u for u in units if gidx[(u["arm"], u["grain"], u["scheme"])] % num_shards == shard_index
    ]


def verify_shard_rung_order(units: list[dict]) -> None:
    """Fail-loud invariant on a shard's PLANNED unit list: every rff/mlp
    unit's (arm, grain, scheme) krr sibling PRECEDES it on the SAME shard
    (rff/mlp read the krr gamma record written when the krr unit runs)."""
    seen_krr: set[tuple] = set()
    for u in units:
        if u.get("phase") != "nonlinear":
            continue
        g = (u["arm"], u["grain"], u["scheme"])
        if u["rung"] == "krr":
            seen_krr.add(g)
        elif g not in seen_krr:
            raise RuntimeError(
                f"shard ordering violation: {u['rung']} unit for {g} has no "
                "preceding krr sibling on this shard (cross-shard gamma race)"
            )


# ── pred persistence ─────────────────────────────────────────────────────────────


def pred_path(u: dict, *, basis: str, seed: int | None = None) -> Path:
    d = tensors_dir("heldout_preds")
    seed_part = f"_s{seed}" if seed is not None and seed >= 0 else ""
    name = (
        f"{u['cell']}_L{u['layer']:02d}_{u['arm']}_{u['grain']}_{basis}_"
        f"{u['scheme']}_{u['rung']}{seed_part}"
    )
    return d / f"{name}.npy"


def persist_pred(u: dict, basis: str, pred: np.ndarray, covered: np.ndarray, seed=None):
    p = pred_path(u, basis=basis, seed=seed)
    np.save(p, pred.astype(np.float16))
    np.save(p.with_name(p.stem + "_mask.npy"), covered)


def load_ridge_pred(u: dict, basis: str) -> tuple[np.ndarray, np.ndarray] | None:
    ref = dict(u)
    ref["rung"] = "ridge"
    p = pred_path(ref, basis=basis)
    m = p.with_name(p.stem + "_mask.npy")
    if p.exists() and m.exists():
        return np.load(p).astype(np.float64), np.load(m)
    return None


def expected_ridge_pred_files(smoke: bool) -> dict[str, tuple[Path, Path]]:
    """(pred, mask) file pairs assemble_gains will query, keyed arm|grain|scheme|basis.

    Round-3 Minor (#1775): the full expected per-row ridge-pred set for every
    planned PERROW nonlinear combo x basis, resolved through the SAME
    ``pred_path`` mapping ``load_ridge_pred`` uses. Gate-B skips only remove
    MLP rungs — the krr/rff siblings of a skipped arm still need the same
    files, so ``gate_b_skip=set()`` is the correct enumeration here.
    """
    out: dict[str, tuple[Path, Path]] = {}
    for u in nonlinear_units(smoke, [0], set()):
        if u["grain"] != "perrow":
            continue
        bases = ["pca48"] if (smoke or u["rung"] == "mlp") else ["ambient", "pca48"]
        for b in bases:
            key = f"{u['arm']}|{u['grain']}|{u['scheme']}|{b}"
            if key in out:
                continue
            p = pred_path({**u, "rung": "ridge"}, basis=b)
            out[key] = (p, p.with_name(p.stem + "_mask.npy"))
    return out


# ── main phases ──────────────────────────────────────────────────────────────────


def run_linear_unit(u: dict, data_cache: dict, args) -> dict:
    ad = _arm_data(u["cell"], u["layer"], data_cache, args)
    t0 = time.monotonic()
    if u["grain"] == "averaged":
        X, Y, rows, _pids = averaged_grain_data(ad, u["arm"])
        pairs = fold_pairs(rows, len(rows), "prefix")
        groups = np.asarray([r["prefix_id"] for r in rows])
        mask_note = None
    else:
        X = ad.X[u["arm"]]
        Y = ad.Y_stacked
        pairs = fold_pairs(ad.rows, len(ad.rows), u["scheme"])
        m = ad.arm_row_mask[u["arm"]]
        pairs = restrict_pairs(pairs, m)
        groups = ad.prefix_ids if u["scheme"] != "query" else ad.query_ids
        mask_note = int((~m).sum())
    Yb = _basis_variants(Y, [u["basis"]])[u["basis"]]
    if u["engine"] == "press":
        fit, pred, covered = fit_press_pairs(
            X, Yb, pairs, compute_df=args.compute_df, device=args.device
        )
    else:
        gate = data_cache.get("parity_gate")
        if gate is None:
            gate = _run_parity_gate(ad, args)
            data_cache["parity_gate"] = gate
        if gate["passed"]:
            fit, pred, covered = fast_ridge_pairs(X, Yb, pairs, device=args.device)
            fit["engine_used"] = "fast"
        else:
            fit, pred, covered = fit_press_pairs(X, Yb, pairs, device=args.device)
            fit["engine_used"] = "press_fallback"
    persist = u["layer"] == LAYER_PRIMARY or (u["layer"] == LAYER_BRIDGE and u["basis"] == "pca48")
    if persist:
        persist_pred(u, u["basis"], pred, covered)
    base = per_fit_baselines(X, Yb, pred, pairs, identity_applicable=(X.shape[1] == Yb.shape[1]))
    out = {
        **u,
        **{k: v for k, v in fit.items() if k != "df_lambda"},
        "df_lambda": fit.get("df_lambda"),
        "mean_cosine": mean_cosine(pred, Yb, covered),
        "baselines": base,
        "n_rows_masked_out": mask_note,
        "wall_s": time.monotonic() - t0,
    }
    if u["arm"] == "context_end" and u["scheme"] == "prefix" and u["grain"] == "perrow":
        if u["cell"] == CELL_PRIMARY and u["layer"] == LAYER_PRIMARY and not args.smoke:
            banked = GATE_C[u["basis"]]
            out["gate_c"] = {
                "banked": banked,
                "delta": abs(out["r2"] - banked),
                "passed": abs(out["r2"] - banked) <= GATE_C_TOL,
            }
    if u["arm"] == "stitch" and u["scheme"] == "prefix" and u["basis"] == "pca48":
        out["stitch_repro_check"] = {
            "reference_battery_excluded": STITCH_REPRO_PCA48,
            "delta": abs(out["r2"] - STITCH_REPRO_PCA48),
            "note": "secondary reproduction read on this plan's own 17,308-row population",
        }
    return out


def _run_parity_gate(ad: ArmData, args) -> dict:
    pairs = fold_pairs(ad.rows, len(ad.rows), "prefix")
    Yb = _basis_variants(ad.Y_stacked, ["pca48"])["pca48"]
    slices = []
    for arm, (tr, te) in zip(("prefix_end", "context_end", "prefix_end"), pairs[:3], strict=False):
        X = ad.X.get(arm, ad.X["prefix_end"])
        slices.append((X[tr], Yb[tr], X[te]))
    passed, rep = fast_ridge_parity_gate(slices, args.device)
    print(f"[parity-gate] fast-vs-slow ridge: {rep}", flush=True)
    # round-2 Minor-b: persist the plan-mandated gate record at gate time —
    # on the 2-shard path the gate runs per SHARD and the --assemble-only
    # process never runs a fast unit, so this file is the durable record
    # linear_fits.json falls back to.
    atomic_write_json(eval_dir("ladder") / "parity_gate.json", {**rep, "meta": result_meta()})
    return rep


def _identity_t1_reads(ad: ArmData, args) -> dict:
    """The t1-only 3584->3584 identity+bias read per 3584-dim arm (plan section 4)."""
    Y_t1 = ad.Y_stacked[:, :3584]
    pairs = fold_pairs(ad.rows, len(ad.rows), "prefix")
    out = {}
    for arm, X in ad.X.items():
        if X.shape[1] != 3584:
            out[arm] = "inapplicable — d_in != 3584 (stitch is 7168-dim; stated)"
            continue
        m = ad.arm_row_mask[arm]
        prs = restrict_pairs(pairs, m)
        r2s = []
        from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

        for tr, te in prs:
            r2s.append(_r2(Y_t1[te], identity_bias_predict(X[tr], Y_t1[tr], X[te])))
        out[arm] = {"r2_folds": [float(v) for v in r2s], "r2_mean": float(np.mean(r2s))}
    return out


def _arm_data(cell: str, layer: int, cache: dict, args) -> ArmData:
    key = (cell, layer)
    if key not in cache:
        arms = (
            ARMS
            if (cell == CELL_PRIMARY and layer == LAYER_PRIMARY)
            else ("prefix_end", "context_end", "query_averaged")
        )
        cache[key] = build_arm_data(
            resolve_store_dir(), cell, layer, arms=tuple(arms), row_limit=args.row_limit
        )
    return cache[key]


def run_nonlinear_unit(u: dict, data_cache: dict, args) -> list[dict]:
    ad = _arm_data(u["cell"], u["layer"], data_cache, args)
    t0 = time.monotonic()
    if u["grain"] == "averaged":
        arm_src = "prefix_end" if u["arm"] == "prefix_end" else "context_end"
        X, Y, rows, _ = averaged_grain_data(ad, arm_src)
        pairs = fold_pairs(rows, len(rows), "prefix")
        groups = np.asarray([r["prefix_id"] for r in rows])
    else:
        X = ad.X[u["arm"]]
        Y = ad.Y_stacked
        pairs = restrict_pairs(
            fold_pairs(ad.rows, len(ad.rows), u["scheme"]), ad.arm_row_mask[u["arm"]]
        )
        groups = ad.prefix_ids if u["scheme"] != "query" else ad.query_ids
    bases = ["pca48"] if (args.smoke or u["basis"] == "pca48") else ["ambient", "pca48"]
    Yb = _basis_variants(Y, bases)
    results = []
    if u["rung"] == "krr":
        res = krr_fit(X, Yb, pairs, groups, device=args.device)
        for b in bases:
            persist_pred(u, b, res["preds"][b], res["covered"])
        results.append(
            {
                **u,
                "per_fold": res["per_fold"],
                "r2": {
                    b: _r2(Yb[b][res["covered"]], res["preds"][b][res["covered"]]) for b in bases
                },
            }
        )
        gpath = tensors_dir("heldout_preds") / _gamma_name(u)
        atomic_write_json(gpath, {"per_fold": res["per_fold"]})
    elif u["rung"] == "rff":
        gpath = tensors_dir("heldout_preds") / _gamma_name(u)
        if not gpath.exists():
            raise RuntimeError(f"RFF unit needs the KRR gamma record {gpath} (run krr first)")
        gammas = json.loads(gpath.read_text())["per_fold"]
        res = rff_fit(
            X, Yb, pairs, groups, device=args.device, gammas_per_fold=gammas, seed=u["seed"]
        )
        for b in bases:
            persist_pred(u, b, res["preds"][b], res["covered"], seed=u["seed"])
        results.append(
            {
                **u,
                "per_fold": res["per_fold"],
                "r2": {
                    b: _r2(Yb[b][res["covered"]], res["preds"][b][res["covered"]]) for b in bases
                },
            }
        )
    elif u["rung"] == "mlp":
        seeds = args.seeds
        res = mlp_fit_groups(
            X,
            Yb["pca48"],
            pairs,
            groups,
            seeds,
            device=args.device,
            max_epochs=8 if args.smoke else MLP_MAX_EPOCHS,
        )
        r2s = {}
        for s in seeds:
            persist_pred(u, "pca48", res["preds_by_seed"][s], res["covered"], seed=s)
            r2s[str(s)] = _r2(Yb["pca48"][res["covered"]], res["preds_by_seed"][s][res["covered"]])
        results.append(
            {
                **u,
                "r2_by_seed": r2s,
                "r2_seed_mean": float(np.mean(list(r2s.values()))),
                "r2_seed_sd": float(np.std(list(r2s.values()))),
                "epochs_ran": res["epochs_ran"],
                "recipe": {
                    "width": MLP_WIDTH,
                    "lr": MLP_LR,
                    "wd": MLP_WD,
                    "max_epochs": MLP_MAX_EPOCHS,
                    "patience": MLP_PATIENCE,
                    "batch": "full-batch (the realized #779 batched_mlp_fit recipe)",
                },
            }
        )
    for r in results:
        r["wall_s"] = time.monotonic() - t0
    return results


def _gamma_name(u: dict) -> str:
    return f"krr_gamma_{u['cell']}_L{u['layer']:02d}_{u['arm']}_{u['grain']}_{u['scheme']}.json"


def assemble_gains(units: list[dict], data_cache: dict, args) -> dict:
    """Per-rung gains vs the matching ridge with paired cluster bootstrap CIs."""
    gains = {}
    for u in units:
        if u.get("phase") != "nonlinear":
            continue
        ad = _arm_data(u["cell"], u["layer"], data_cache, args)
        if u["grain"] == "averaged":
            continue  # averaged gains read off the JSON r2 values directly
        groups = ad.prefix_ids if u["scheme"] != "query" else ad.query_ids
        bases = (
            ["pca48"] if args.smoke else (["pca48"] if u["rung"] == "mlp" else ["ambient", "pca48"])
        )
        for b in bases:
            ridge = load_ridge_pred({**u, "grain": u["grain"]}, b)
            if ridge is None:
                continue
            rpred, rmask = ridge
            seeds = (
                [None]
                if u["rung"] == "krr"
                else (args.seeds if u["rung"] == "mlp" else [u["seed"]])
            )
            for s in seeds:
                p = pred_path(u, basis=b, seed=s)
                m = p.with_name(p.stem + "_mask.npy")
                if not p.exists():
                    continue
                npred = np.load(p).astype(np.float64)
                nmask = np.load(m)
                both = rmask & nmask
                Yb = _basis_variants(ad.Y_stacked, [b])[b]
                boot = cluster_bootstrap_delta_r2(
                    Yb,
                    npred,
                    rpred,
                    both,
                    groups,
                    n_draws=200 if args.smoke else 2000,
                )
                key = f"{u['arm']}|{u['grain']}|{u['scheme']}|{u['rung']}|{b}" + (
                    f"|s{s}" if s is not None and s >= 0 else ""
                )
                gains[key] = boot
    return gains


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 estimation ladder (P1 linear / P3 nonlinear)")
    ap.add_argument("--phase", choices=["linear", "nonlinear"], required=True)
    ap.add_argument("--cells", default=f"{CELL_PRIMARY},cell_pre_own")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--row-limit", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--compute-df", action="store_true", default=None)
    ap.add_argument("--ignore-pilot-gate", action="store_true")
    ap.add_argument("--stage-store", action="store_true", help="stage #1092 store from Hub first")
    ap.add_argument(
        "--assemble-only",
        action="store_true",
        help="skip unit fits; assemble ALL shards' unit JSONLs into the final JSON",
    )
    args = ap.parse_args()
    args.device = _device(args.device)
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    if args.smoke and args.row_limit is None:
        args.row_limit = 600
    if args.compute_df is None:
        args.compute_df = args.device == "cuda" or bool(args.smoke)
    if args.stage_store:
        stage_store_if_needed(
            resolve_store_dir(), cells=cells, layers=[LAYER_PRIMARY, LAYER_BRIDGE]
        )
    out_dir = eval_dir("ladder")
    units_path = out_dir / f"units_{args.phase}_shard{args.shard_index}.jsonl"
    done = {unit_key(d, REGIME_KEYS) for d in load_units_validated(units_path, unit_row_incomplete)}
    if args.phase == "linear":
        planned = linear_units(cells, args.smoke)
    else:
        skips = gate_b_skips(eval_dir("detection") / "hsic_dcor.json")
        planned = nonlinear_units(args.smoke, args.seeds, skips)
    for u in planned:
        u["smoke"] = bool(args.smoke)
        u["row_limit"] = args.row_limit
    units = shard_units(planned, args.phase, args.num_shards, args.shard_index)
    if args.phase == "nonlinear":
        verify_shard_rung_order(units)
    todo = [] if args.assemble_only else [u for u in units if unit_key(u, REGIME_KEYS) not in done]
    if args.assemble_only:
        # NOT the resume line: unit fits are skipped BY FLAG here — the old
        # shared print read "0/N units to run (resume skipped N)" and was
        # misdiagnosed as a resume-predicate bug on the P3 crash retry.
        print(
            f"[ladder/{args.phase}] shard {args.shard_index}/{args.num_shards}: "
            "assemble-only — unit fits skipped by flag (not a resume verdict)",
            flush=True,
        )
    else:
        print(
            f"[ladder/{args.phase}] shard {args.shard_index}/{args.num_shards}: "
            f"{len(todo)}/{len(units)} units to run (resume skipped {len(units) - len(todo)})",
            flush=True,
        )
    data_cache: dict = {}
    t_all = time.monotonic()
    pilot_done = False
    for k, u in enumerate(todo):
        t0 = time.monotonic()
        if args.phase == "linear":
            recs = [run_linear_unit(u, data_cache, args)]
        else:
            recs = run_nonlinear_unit(u, data_cache, args)
        for r in recs:
            append_unit(units_path, r)
        dt = time.monotonic() - t0
        print(
            f"[ladder/{args.phase}] unit {k + 1}/{len(todo)} "
            f"{u['arm']}/{u['grain']}/{u['scheme']}/{u['rung']} elapsed={dt:.1f}s",
            flush=True,
        )
        if not pilot_done and not args.smoke:
            pilot_done = True
            projected_h = dt * len(todo) / 3600.0
            booked = PILOT_BOOKED_WALL_H[args.phase] * 2  # the section-9 2x booking
            print(
                f"[pilot-gate] first unit {dt:.1f}s -> projected {projected_h:.2f}h "
                f"vs booked {booked:.2f}h (2x row)",
                flush=True,
            )
            if projected_h > booked and not args.ignore_pilot_gate:
                atomic_write_json(
                    out_dir / "pilot_gate_report.json",
                    {
                        "meta": result_meta(phase=args.phase),
                        "per_unit_s": dt,
                        "n_units": len(todo),
                        "projected_wall_h": projected_h,
                        "booked_wall_h_2x": booked,
                        "verdict": "DEVIATION >2x — designed halt (rc=7)",
                    },
                )
                return PILOT_GATE_RC
        # gate C hard check (plan section 7: halt-and-fix on failure)
        for r in recs:
            g = r.get("gate_c")
            if g and not g["passed"]:
                atomic_write_json(out_dir / "gate_c_failure.json", {**r, "meta": result_meta()})
                print(f"[gate-c] FAILED: {g}", flush=True)
                return 21
    # assembly: shards write units only; the final JSON is assembled from ALL
    # shards' unit JSONLs (single-shard runs and --assemble-only both land here).
    if args.num_shards > 1 and not args.assemble_only:
        print(f"[ladder/{args.phase}] shard {args.shard_index} done (assembly separate)")
        return 0
    raw_units = []
    for sp in sorted(out_dir.glob(f"units_{args.phase}_shard*.jsonl")):
        raw_units.extend(load_units_validated(sp, unit_row_incomplete))
    by_key: dict[tuple, dict] = {}
    for r in raw_units:
        by_key[unit_key(r, REGIME_KEYS)] = r  # last row wins (purged+rerun units append anew)
    # completeness gate: the final JSON must cover the FULL planned grid across
    # ALL shards — a partial assembly otherwise masquerades as a completed
    # phase (the P3 crash retry would have assembled 3/38 units silently).
    missing = [u for u in planned if unit_key(u, REGIME_KEYS) not in by_key]
    if missing:
        print(
            f"[ladder/{args.phase}] ASSEMBLY INCOMPLETE: {len(missing)}/{len(planned)} planned "
            "units have no completed row — refusing to write a partial final JSON (rc=23)",
            flush=True,
        )
        for u in missing[:25]:
            print(
                f"[ladder/{args.phase}]   missing "
                f"{u['arm']}/{u['grain']}/{u['scheme']}/{u['rung']} seed={u['seed']}",
                flush=True,
            )
        return ASSEMBLY_INCOMPLETE_RC
    # exactly the planned rows, in planned order (stale extra rows dropped)
    all_units = [by_key[unit_key(u, REGIME_KEYS)] for u in planned]
    payload: dict = {
        "meta": result_meta(phase=args.phase, smoke=args.smoke, device=args.device),
        "units": all_units,
    }
    if args.phase == "linear":
        ad = _arm_data(CELL_PRIMARY, LAYER_PRIMARY, data_cache, args)
        payload["identity_bias_t1_reads"] = _identity_t1_reads(ad, args)
        gate_json = out_dir / "parity_gate.json"
        payload["parity_gate"] = (
            data_cache.get("parity_gate")
            or next((u.get("parity_gate") for u in all_units if u.get("parity_gate")), None)
            # 2-shard path: the gate ran per shard and persisted its record at
            # gate time (round-2 Minor-b) — the assemble-only process reads it.
            or (json.loads(gate_json.read_text()) if gate_json.exists() else None)
        )
        out_json = out_dir / "linear_fits.json"
    else:
        gains = assemble_gains(all_units, data_cache, args)
        if not gains and any(u["grain"] == "perrow" for u in all_units):
            print(
                "[ladder/nonlinear] ASSEMBLY ERROR: gains_vs_ridge empty though perrow "
                "units exist — ridge preds missing? run (or prefetch) p1 first (rc=23)",
                flush=True,
            )
            return ASSEMBLY_INCOMPLETE_RC
        payload["gains_vs_ridge"] = gains
        out_json = out_dir / "nonlinear_fits.json"
    atomic_write_json(out_json, payload)
    print(
        f"[ladder/{args.phase}] wrote {out_json} "
        f"({len(all_units)} units, wall {(time.monotonic() - t_all) / 60:.1f} min)",
        flush=True,
    )
    upload_phase_tensors("heldout_preds", smoke=args.smoke)
    upload_phase_eval_json("ladder", smoke=args.smoke)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
