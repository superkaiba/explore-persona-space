#!/usr/bin/env python3
"""#1775 P4: rank-r bilinear interaction on the stitch input + interpretation.

Model (plan section 4 P4; DMDc arXiv:1409.6358 + low-rank bilinear
arXiv:1610.04325):  a_hat = W [p; q] + sum_i (u_i^T p)(v_i^T q) w_i + b,
p = prefix_end (3584), q = bare_query (3584), target = pooled pca48 (headline)
with an ambient companion at r in {0, r*}. Fit: warm-start W at the per-fold
stitch-ridge solution (``fit_h.ridge_fit_predict_fast_layer_batched``
``return_weights=True`` — same standardize/center conventions), then joint
full-batch Adam with DECOUPLED per-group weight decay (AdamW semantics with a
per-group wd — wd in {0, 1e-4, 1e-2} rides the group batch), early-stopped on
a group-respecting inner validation split. Groups (seeds x wd) train as ONE
batched einsum loop per (scheme, fold, r) — no serial per-fit loop.

r in {0,1,2,4,8,16,32,64}; r = 0 is the GD-refit de-regularization null
(same optimizer/wd/early-stop, NO interaction): Delta_named = R2(r*) - R2(r=0);
R2(r=0) - R2(stitch PRESS-ridge) reported separately. r* selected on inner
validation over r >= 1 (frozen before outer test; the outer-test r-curve is
labeled exploratory). delta_beyond = R2(stitch-MLP, P3) - R2(bilinear r*).
All lattice CIs are paired CLUSTER bootstraps (group = the fold scheme's
unit; two-way for the doubly-novel read).

Interpretation: interaction projections onto answer PCs + the 3-trait r_B
dictionary vs TWO nulls (matched-norm isotropic + covariance-matched from the
train-fold answer covariance; 1000 batched draws; max-selection applied per
draw). Bilinear-residual HSIC re-test reuses the P2 machinery — the "named"
verdict co-signature.
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

from issue1092_fit_grid import _project_rb_to_basis  # noqa: E402
from issue1775_common import (  # noqa: E402
    CELL_PRIMARY,
    FU_SUB,
    HF_DATA_REPO,
    LAYER_PRIMARY,
    _basis_targets_with_info,
    _r2,
    append_unit,
    atomic_write_json,
    build_arm_data,
    build_dependence_matrices,
    cluster_bootstrap_delta_r2,
    eval_dir,
    fold_pairs,
    hsic_statistic,
    inner_val_split,
    knn_retrieval,
    load_units_validated,
    null_stats_batched,
    observed_stats,
    p_value,
    press_fit_predict,
    resolve_store_dir,
    restrict_pairs,
    result_meta,
    tensors_dir,
    unit_key,
    upload_phase_eval_json,
    upload_phase_tensors,
)
from issue1775_detection import SCHEMES as DETECT_SCHEMES  # noqa: E402
from issue1775_detection import complete_dense_block  # noqa: E402

from explore_persona_space.experiments.issue_779.fit_h import (  # noqa: E402
    ridge_fit_predict_fast_layer_batched,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

R_GRID = (0, 1, 2, 4, 8, 16, 32, 64)
WD_GRID = (0.0, 1e-4, 1e-2)
BILIN_LR = 1e-3  # ungrounded — needs smoke-test (plan section 11); validated by the
# r=0 refit reproducing the stitch-ridge R2 (warm start + early stop, see smoke)
BILIN_MAX_EPOCHS = 500
BILIN_PATIENCE = 30
REGIME_KEYS = ("scheme", "fold", "r", "basis", "smoke", "row_limit")
# Designed halt (mirrors issue1775_ladder.ASSEMBLY_INCOMPLETE_RC; run.sh route_rc
# names rc=23): the assembly refuses to write a PARTIAL bilinear_fits.json when
# planned units are missing a completed row — the P3 crash-fix class ported to P4
# (r3 persisted concern p4-bilinear-raw-resume-no-completeness-gate).
ASSEMBLY_INCOMPLETE_RC = 23


def bilinear_row_incomplete(d: dict) -> str | None:
    """Reason a persisted bilinear unit row is NOT a complete result (None = complete).

    Mirrors issue1775_ladder.unit_row_incomplete for the P4 row schema: a unit
    counts as done ONLY when its JSONL row carries the full payload assemble()
    reads — every REGIME_KEY, ``epochs_ran``, and a non-empty ``variants`` list
    whose entries carry seed/wd/inner_val_mse/r2_te. A stub / truncated /
    older-schema row must re-run, never resume-skip.
    """
    for k in REGIME_KEYS:
        if k not in d:
            return f"missing regime key {k!r}"
    if "epochs_ran" not in d:
        return "missing epochs_ran (unit never completed)"
    v = d.get("variants")
    if not isinstance(v, list) or not v:
        return "row without variants"
    required = {"seed", "wd", "inner_val_mse", "r2_te"}
    if any(not required <= set(e) for e in v):
        return "variants entries missing seed/wd/inner_val_mse/r2_te"
    return None


RB_TRAITS = ("evil", "hallucination", "sycophancy")
RB_HF_PREFIX = "issue779_monitoring/r_b"
N_PROJ_DRAWS = 1000


def _standardize_train(X: np.ndarray, tr: np.ndarray):
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-9
    return (X - mu) / sd


def bilinear_fit_batched(
    Xn: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    groups: np.ndarray,
    *,
    r: int,
    seeds: list[int],
    device: str,
    max_epochs: int = BILIN_MAX_EPOCHS,
    patience: int = BILIN_PATIENCE,
    lr: float = BILIN_LR,
) -> dict:
    """One (fold, r): all (seed x wd) variants as ONE batched Adam loop with
    decoupled per-group weight decay. Xn = standardized stitch input (n, 7168)."""
    dev = torch.device(device)
    d_in = Xn.shape[1]
    d_half = d_in // 2
    d_out = Y.shape[1]
    # warm start: per-fold stitch ridge in the SAME standardized space
    # (the helper standardizes internally with the identical population-std
    # convention; we feed the already-standardized Xn so its internal mu~0/sd~1).
    _preds, W0 = ridge_fit_predict_fast_layer_batched(
        Xn[tr][None], Y[tr][None], Xn[te][None], device=device, return_weights=True
    )
    W0 = torch.from_numpy(W0[0]).to(dev, torch.float32)  # (d_in, d_out)
    ymu0 = torch.from_numpy(Y[tr].mean(0)).to(dev, torch.float32)
    itr, ival = inner_val_split(tr, groups, seed=1234 + r)
    Xt = torch.from_numpy(Xn).to(dev, torch.float32)
    Yt = torch.from_numpy(Y).to(dev, torch.float32)
    variants = [(s, wd) for s in seeds for wd in WD_GRID]
    G = len(variants)
    W = W0.T[None].repeat(G, 1, 1).contiguous()  # (G, d_out, d_in)
    b = ymu0[None].repeat(G, 1).contiguous()
    U = torch.zeros((G, max(r, 1), d_half), device=dev)
    V = torch.zeros((G, max(r, 1), d_half), device=dev)
    Wv = torch.zeros((G, max(r, 1), d_out), device=dev)
    for gi, (s, _wd) in enumerate(variants):
        g = torch.Generator(device="cpu").manual_seed(int(s) + 7)
        if r > 0:
            U[gi] = (torch.randn((r, d_half), generator=g) * 0.02).to(dev)
            V[gi] = (torch.randn((r, d_half), generator=g) * 0.02).to(dev)
            Wv[gi] = (torch.randn((r, d_out), generator=g) * 0.02).to(dev)
    params = [W, b] + ([U, V, Wv] if r > 0 else [])
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)  # decay applied manually (per-group AdamW)
    wd_vec = torch.tensor([wd for _s, wd in variants], device=dev, dtype=torch.float32)
    itr_t = torch.from_numpy(itr).to(dev)
    ival_t = torch.from_numpy(ival).to(dev)
    te_t = torch.from_numpy(te).to(dev)

    def forward(idx: torch.Tensor) -> torch.Tensor:
        Xb = Xt[idx]
        out = torch.einsum("nd,god->gno", Xb, W) + b[:, None, :]
        if r > 0:
            A = torch.einsum("np,grp->gnr", Xb[:, :d_half], U)
            Bq = torch.einsum("nq,grq->gnr", Xb[:, d_half:], V)
            out = out + torch.einsum("gnr,gro->gno", A * Bq, Wv)
        return out

    best_val = torch.full((G,), float("inf"), device=dev)
    bad = torch.zeros(G, dtype=torch.long, device=dev)
    frozen = torch.zeros(G, dtype=torch.bool, device=dev)
    best_state: list[tuple | None] = [None] * G
    epochs_ran = np.zeros(G, dtype=int)
    Ytr_i = Yt[itr_t]
    Yva_i = Yt[ival_t]
    for ep in range(max_epochs):
        opt.zero_grad(set_to_none=True)
        pred = forward(itr_t)
        loss_g = ((pred - Ytr_i[None]) ** 2).mean(dim=(1, 2))
        (loss_g * (~frozen).float()).sum().backward()
        opt.step()
        with torch.no_grad():
            decay = (1.0 - lr * wd_vec).clamp(min=0.0)
            W.mul_(decay[:, None, None])
            if r > 0:
                U.mul_(decay[:, None, None])
                V.mul_(decay[:, None, None])
                Wv.mul_(decay[:, None, None])
            val_g = ((forward(ival_t) - Yva_i[None]) ** 2).mean(dim=(1, 2))
        improved = (val_g < best_val - 1e-7) & (~frozen)
        for gi in torch.nonzero(improved).ravel().tolist():
            best_state[gi] = tuple(t[gi].detach().clone().cpu() for t in (W, b, U, V, Wv))
        best_val = torch.where(improved, val_g, best_val)
        bad = torch.where(improved, torch.zeros_like(bad), bad + (~frozen).long())
        frozen |= (bad >= patience) & (~frozen)
        epochs_ran[(~frozen).cpu().numpy()] = ep + 1
        if frozen.all():
            break
    out: dict = {"variants": [], "epochs_ran": epochs_ran.tolist()}
    with torch.no_grad():
        for gi, (s, wd) in enumerate(variants):
            st = best_state[gi] or tuple(t[gi].detach().cpu() for t in (W, b, U, V, Wv))
            Wg, bg, Ug, Vg, Wvg = (t.to(dev) for t in st)
            Xe = Xt[te_t]
            pe = Xe @ Wg.T + bg
            if r > 0:
                A = (Xe[:, :d_half] @ Ug.T) * (Xe[:, d_half:] @ Vg.T)
                pe = pe + A @ Wvg
            out["variants"].append(
                {
                    "seed": s,
                    "wd": wd,
                    "inner_val_mse": float(best_val[gi].item()),
                    "pred_te": pe.cpu().numpy(),
                    "params": {
                        "W": st[0].numpy().astype(np.float16),
                        "b": st[1].numpy().astype(np.float32),
                        "U": st[2].numpy().astype(np.float32) if r > 0 else None,
                        "V": st[3].numpy().astype(np.float32) if r > 0 else None,
                        "Wv": st[4].numpy().astype(np.float32) if r > 0 else None,
                    },
                }
            )
    # locals free at return (no `del` of closure-captured names — ruff F821)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def two_way_cluster_bootstrap_delta_r2(
    Y, pred_a, pred_b, covered, prefix_ids, query_ids, *, n_draws=2000, seed=0
) -> dict:
    """Two-way (prefix x query) cluster bootstrap for the doubly-novel read:
    row weight per draw = (multiplicity of its prefix) x (multiplicity of its query)."""
    idx = np.nonzero(covered)[0]
    y = Y[idx]
    mu = y.mean(axis=0, keepdims=True)
    se_a = ((y - pred_a[idx]) ** 2).sum(axis=1)
    se_b = ((y - pred_b[idx]) ** 2).sum(axis=1)
    st = ((y - mu) ** 2).sum(axis=1)
    up, pi = np.unique(prefix_ids[idx], return_inverse=True)
    uq, qi = np.unique(query_ids[idx], return_inverse=True)
    rng = np.random.default_rng(seed)
    P, Q = len(up), len(uq)
    deltas = np.empty(n_draws)
    for bdx in range(n_draws):
        wp = np.bincount(rng.integers(0, P, size=P), minlength=P).astype(np.float64)
        wq = np.bincount(rng.integers(0, Q, size=Q), minlength=Q).astype(np.float64)
        w = wp[pi] * wq[qi]
        T = float(w @ st)
        deltas[bdx] = ((w @ se_b) - (w @ se_a)) / max(T, 1e-300)
    point = float((se_b.sum() - se_a.sum()) / max(st.sum(), 1e-300))
    return {
        "delta_r2": point,
        "ci95_two_way_cluster": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "n_prefix_groups": int(P),
        "n_query_groups": int(Q),
        "n_rows": int(idx.size),
        "n_draws": int(n_draws),
    }


def load_rb_dictionary(store: Path, layer: int) -> dict[str, np.ndarray]:
    """The 3-trait r_B dictionary at the requested layer (Hub-fetched .pt files;
    schema introspected fail-loud — plan A9)."""
    out = {}
    for trait in RB_TRAITS:
        target = store / "rb_cache" / f"{trait}.pt"
        if not target.exists():
            hub.stage_hub_file(
                HF_DATA_REPO, f"{RB_HF_PREFIX}/{trait}.pt", target, repo_type="dataset"
            )
        d = torch.load(target, map_location="cpu", weights_only=False)
        if isinstance(d, dict):
            rb = d.get("r_b", d.get("rb", d.get("directions")))
            layers = d.get("layers")
            if rb is None:
                raise ValueError(f"r_B bundle {trait}.pt keys {sorted(d)} — no r_b field")
            rb = np.asarray(rb, dtype=np.float64)
            if rb.ndim == 2:
                li = list(layers).index(layer) if layers is not None else layer
                rb = rb[li]
        else:
            rb = np.asarray(d, dtype=np.float64)
            if rb.ndim == 2:
                rb = rb[layer]
        assert rb.shape == (3584,), f"r_B {trait} shape {rb.shape} != (3584,)"
        out[trait] = rb
    return out


def _unit_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def projection_analysis(
    params_at_rstar: list[dict],
    basis_info: dict,
    Y_train: np.ndarray,
    X_p_train: np.ndarray,
    X_q_train: np.ndarray,
    rb: dict[str, np.ndarray],
    *,
    n_draws: int = N_PROJ_DRAWS,
    seed: int = 0,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Cosines of interaction directions vs answer PCs + r_B, against TWO nulls
    (matched-norm isotropic; covariance-matched), max-selection applied per draw."""
    rng = np.random.default_rng(seed)
    W_list = [p["params"]["Wv"] for p in params_at_rstar if p["params"]["Wv"] is not None]
    U_list = [p["params"]["U"] for p in params_at_rstar if p["params"]["U"] is not None]
    V_list = [p["params"]["V"] for p in params_at_rstar if p["params"]["V"] is not None]
    if not W_list:
        return {"note": "r*=0 — no interaction terms to project"}, {}
    Wv = _unit_norm(np.concatenate(W_list, axis=0).astype(np.float64))  # (T, d_out)
    U = _unit_norm(np.concatenate(U_list, axis=0).astype(np.float64))  # (T, 3584)
    V = _unit_norm(np.concatenate(V_list, axis=0).astype(np.float64))
    d_out = Wv.shape[1]
    # output-side reference axes: answer PCs (pca48 coords = identity axes) + r_B rows
    rb_out = np.stack(
        [_unit_norm(_project_rb_to_basis(rb[t], basis_info, expected_dim=d_out)) for t in RB_TRAITS]
    )
    rb_in = np.stack([_unit_norm(rb[t]) for t in RB_TRAITS])
    obs = {
        "w_vs_pc_abscos_max": float(np.max(np.abs(Wv))),
        "w_vs_rb_abscos_max": float(np.max(np.abs(Wv @ rb_out.T))),
        "u_vs_rb_abscos_max": float(np.max(np.abs(U @ rb_in.T))),
        "v_vs_rb_abscos_max": float(np.max(np.abs(V @ rb_in.T))),
        "n_terms_pooled": int(Wv.shape[0]),
    }
    T = Wv.shape[0]
    # null (i): matched-norm isotropic (unit vectors are norm-matched after normalization)
    nulls: dict[str, np.ndarray] = {}
    iso_out = _unit_norm(rng.standard_normal((n_draws, T, d_out)))
    nulls["iso_w_vs_pc_max"] = np.abs(iso_out).max(axis=(1, 2))
    nulls["iso_w_vs_rb_max"] = np.abs(iso_out @ rb_out.T).max(axis=(1, 2))
    iso_in = _unit_norm(rng.standard_normal((n_draws, T, 3584)))
    nulls["iso_u_vs_rb_max"] = np.abs(iso_in @ rb_in.T).max(axis=(1, 2))
    iso_in_v = _unit_norm(rng.standard_normal((n_draws, T, 3584)))
    nulls["iso_v_vs_rb_max"] = np.abs(iso_in_v @ rb_in.T).max(axis=(1, 2))

    # null (ii): covariance-matched (train-fold answer covariance / input covariances)
    def _cov_draws(M: np.ndarray, dim: int) -> np.ndarray:
        Mc = M - M.mean(0, keepdims=True)
        cap = min(4000, Mc.shape[0])
        sub = Mc[rng.choice(Mc.shape[0], size=cap, replace=False)]
        _u_, s_, vt_ = np.linalg.svd(sub, full_matrices=False)
        scale = s_ / np.sqrt(cap)
        z = rng.standard_normal((n_draws, T, len(s_)))
        return _unit_norm(z @ (vt_ * scale[:, None]))

    cov_out = _cov_draws(Y_train, d_out)
    nulls["cov_w_vs_pc_max"] = np.abs(cov_out).max(axis=(1, 2))
    nulls["cov_w_vs_rb_max"] = np.abs(cov_out @ rb_out.T).max(axis=(1, 2))
    cov_in_p = _cov_draws(X_p_train, 3584)
    nulls["cov_u_vs_rb_max"] = np.abs(cov_in_p @ rb_in.T).max(axis=(1, 2))
    # v projects the QUERY side: its cov-matched null draws from X_q_train (round-2 Minor-a)
    cov_in_q = _cov_draws(X_q_train, 3584)
    nulls["cov_v_vs_rb_max"] = np.abs(cov_in_q @ rb_in.T).max(axis=(1, 2))
    report = {
        "observed": obs,
        "nulls_p": {
            "w_vs_pc_max": {
                "iso": p_value(nulls["iso_w_vs_pc_max"], obs["w_vs_pc_abscos_max"]),
                "cov_matched": p_value(nulls["cov_w_vs_pc_max"], obs["w_vs_pc_abscos_max"]),
            },
            "w_vs_rb_max": {
                "iso": p_value(nulls["iso_w_vs_rb_max"], obs["w_vs_rb_abscos_max"]),
                "cov_matched": p_value(nulls["cov_w_vs_rb_max"], obs["w_vs_rb_abscos_max"]),
            },
            "u_vs_rb_max": {
                "iso": p_value(nulls["iso_u_vs_rb_max"], obs["u_vs_rb_abscos_max"]),
                "cov_matched": p_value(nulls["cov_u_vs_rb_max"], obs["u_vs_rb_abscos_max"]),
            },
            "v_vs_rb_max": {
                "iso": p_value(nulls["iso_v_vs_rb_max"], obs["v_vs_rb_abscos_max"]),
                "cov_matched": p_value(nulls["cov_v_vs_rb_max"], obs["v_vs_rb_abscos_max"]),
            },
        },
        "note": (
            "max-selection over (terms x traits/PCs) applied IDENTICALLY per null draw "
            "(selection-symmetric); dictionary-alignment narration keys on the "
            "covariance-matched null; 3-trait dictionary — answer-PC projections carry "
            "the breadth (plan section 10)"
        ),
    }
    return report, nulls


# ── cell 2 (fu round `dedup-refit-pcfold-doubly`): train-fold-only 48-PC bases ────

# r* CARRIED from run-1's inner-val selection (bilinear_fits.json prefix r_star=32);
# re-selecting r on the new bases would add a second variable (plan section 4 cell 2).
FOLDPC_R_GRID = (0, 32)
FOLDPC_REGIME_KEYS = ("scheme", "fold", "r", "basis", "smoke", "row_limit")
# Designed halt (plan section 8 risk row 3): the fold-PC r=0 GD refit must reproduce
# the fold-basis stitch PRESS-ridge R2 within 0.02 — a bigger gap is a code bug,
# never a science read. rc distinct from run-1's 7/21/22/23 (fu_run.sh route_rc).
FOLDPC_REPRO_RC = 24
FOLDPC_REPRO_TOL = 0.02


def _fold_centered(Yp_f, preds_f, te):
    """Fold-local test-mean centering (pooled-SS semantics for the committed
    cluster-bootstrap helper): subtracting the fold's test-row mean from Y AND
    every pred leaves per-row SE invariant and makes ST fold-local; the global
    mean over covered rows is then exactly 0, so ``cluster_bootstrap_delta_r2``'s
    internal centering reproduces per-fold SS pooling."""
    mu = Yp_f[te].mean(axis=0, keepdims=True)
    return Yp_f[te] - mu, [p - mu for p in preds_f]


def run_foldpc(args) -> int:
    """Cell 2: rank-{0,32} bilinear refit under per-fold TRAIN-FOLD-ONLY 48-PC
    target bases (basis mode ``pca48_foldpc``; the recorded run-1 pca48
    full-population-PC deviation's discharge). Run-1 optimizer protocol
    verbatim (warm-start fold-basis stitch ridge inside ``bilinear_fit_batched``,
    wd grid, seeds, early stop); novel-prefix scheme only; r* carried.

    Pooling: per-fold R2 in the fold's own basis via per-fold SS sums;
    Delta_named(fold-PC) CI = paired prefix-group cluster bootstrap on
    fold-centered arrays (the committed helper). Smoke: 1 fold x 1 seed x
    r in {0,32} at row_limit; production-n gates demoted to log lines
    (#1345 gate-calibration parity).
    """
    schemes = [s.strip() for s in (args.schemes or "prefix").split(",") if s.strip()]
    r_grid = sorted(int(r) for r in args.r_grid.split(",") if r.strip())
    if schemes != ["prefix"] or tuple(r_grid) != FOLDPC_R_GRID:
        print(
            f"[foldpc] REFUSED: --basis pca48_foldpc is restricted to --schemes prefix "
            f"--r-grid 0,32 (got schemes={schemes} r_grid={r_grid}); r*=32 is CARRIED, "
            "never re-selected (plan section 4 cell 2)",
            flush=True,
        )
        return 2
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n_draws = 2000
    max_epochs = BILIN_MAX_EPOCHS
    if args.smoke:
        if args.row_limit is None:
            args.row_limit = 600
        seeds = seeds[:1]
        n_draws = 200
        max_epochs = 40
    out_dir = eval_dir(FU_SUB)
    params_dir = tensors_dir("bilinear_params_foldpc")
    store = resolve_store_dir()
    ad = build_arm_data(
        store, CELL_PRIMARY, LAYER_PRIMARY, arms=("stitch",), row_limit=args.row_limit
    )
    X = ad.X["stitch"]
    groups = ad.prefix_ids
    pairs = restrict_pairs(fold_pairs(ad.rows, len(ad.rows), "prefix"), ad.arm_row_mask["stitch"])
    folds = [0] if args.smoke else list(range(len(pairs)))
    units_path = out_dir / "units_foldpc_shard0.jsonl"
    done = {
        unit_key(d, FOLDPC_REGIME_KEYS)
        for d in load_units_validated(units_path, bilinear_row_incomplete)
    }
    n = len(ad.rows)
    t0 = time.monotonic()
    per_fold: dict[int, dict] = {}
    n_units = len(folds) * len(r_grid)
    n_done = 0
    for f in folds:
        tr, te = pairs[f]
        tb = time.monotonic()
        Ypf, info_f = _basis_targets_with_info(
            ad.Y_stacked,
            "pca48_foldpc",
            train_idx=tr,
            hidden_dim=3584,
            targets=["t1", "t2", "t3"],
            projection_target="t1",
        )
        Ypf = np.ascontiguousarray(Ypf, dtype=np.float64)
        basis_wall = time.monotonic() - tb
        np.savez(
            params_dir / f"basis_prefixfoldpc_f{f}.npz",
            mu=info_f["mu_basis"].astype(np.float32),
            v=info_f["v_basis"].astype(np.float32),
            train_idx_n=np.int64(info_f["train_idx_n"]),
        )
        if f == folds[0]:
            print(
                f"[foldpc] PILOT: fold {f} basis fit {basis_wall:.0f}s on "
                f"{info_f['train_idx_n']} train rows -> projected basis wall "
                f"~{basis_wall * len(folds) / 3600:.2f}h over {len(folds)} folds",
                flush=True,
            )
        # fold-basis stitch PRESS ridge (run-1 grid RIDGE_LAMBDAS) — the r=0
        # reproduction reference + the dereg-component comparator.
        res_r = press_fit_predict(
            torch.from_numpy(X[tr]).double(),
            torch.from_numpy(Ypf[tr]).double(),
            torch.from_numpy(X[te]).double(),
            standardize=True,
        )
        ridge_pred = res_r["pred"].detach().cpu().numpy()
        np.save(params_dir / f"pred_prefixfoldpc_f{f}_ridge.npy", ridge_pred.astype(np.float16))
        np.save(params_dir / f"te_prefixfoldpc_f{f}.npy", te)
        Xn = _standardize_train(X, tr)
        fold_rec: dict = {
            "te": te,
            "Ypf": Ypf,
            "ridge_pred": ridge_pred,
            "basis_wall_s": basis_wall,
            "preds": {},
        }
        for r in r_grid:
            u = {
                "scheme": "prefix",
                "fold": f,
                "r": r,
                "basis": "pca48_foldpc",
                "smoke": bool(args.smoke),
                "row_limit": args.row_limit,
            }
            if unit_key(u, FOLDPC_REGIME_KEYS) in done:
                rows = [
                    d
                    for d in load_units_validated(units_path, bilinear_row_incomplete)
                    if unit_key(d, FOLDPC_REGIME_KEYS) == unit_key(u, FOLDPC_REGIME_KEYS)
                ]
                rec = rows[-1]
                print(f"[foldpc] RESUME fold={f} r={r} (unit row present)", flush=True)
            else:
                res = bilinear_fit_batched(
                    Xn,
                    Ypf,
                    tr,
                    te,
                    groups,
                    r=r,
                    seeds=seeds,
                    device=args.device,
                    max_epochs=max_epochs,
                )
                rec = {**u, "epochs_ran": res["epochs_ran"], "variants": []}
                for var in res["variants"]:
                    rec["variants"].append(
                        {
                            "seed": var["seed"],
                            "wd": var["wd"],
                            "inner_val_mse": var["inner_val_mse"],
                            "r2_te": _r2(Ypf[te], var["pred_te"]),
                        }
                    )
                    np.save(
                        params_dir
                        / f"pred_prefixfoldpc_f{f}_r{r}_s{var['seed']}_wd{var['wd']:g}.npy",
                        var["pred_te"].astype(np.float16),
                    )
                    torch.save(
                        dict(var["params"]),
                        params_dir
                        / f"params_prefixfoldpc_f{f}_r{r}_s{var['seed']}_wd{var['wd']:g}.pt",
                    )
                append_unit(units_path, rec)
            # seed-ensemble pooled pred at per-seed best wd (run-1 _pooled_pred form)
            acc = np.zeros((len(te), Ypf.shape[1]))
            for s in seeds:
                v = _best_variant(rec["variants"], s)
                acc += np.load(
                    params_dir / f"pred_prefixfoldpc_f{f}_r{r}_s{s}_wd{v['wd']:g}.npy"
                ).astype(np.float64)
            fold_rec["preds"][r] = acc / len(seeds)
            n_done += 1
            print(
                f"[foldpc] unit {n_done}/{n_units} fold={f} r={r} "
                f"elapsed={time.monotonic() - t0:.0f}s",
                flush=True,
            )
        per_fold[f] = fold_rec
    # ── assembly: per-fold SS pooling + fold-centered paired cluster bootstraps ──
    d48 = next(iter(per_fold.values()))["Ypf"].shape[1]
    covered = np.zeros(n, dtype=bool)
    Yv = np.zeros((n, d48))
    Av = {r: np.zeros((n, d48)) for r in r_grid}
    Rv = np.zeros((n, d48))
    per_fold_out: dict = {}
    knn_folds: list[dict] = []
    for f, fr in per_fold.items():
        te = fr["te"]
        yv, cent = _fold_centered(
            fr["Ypf"], [fr["preds"][r] for r in r_grid] + [fr["ridge_pred"]], te
        )
        Yv[te] = yv
        for i, r in enumerate(r_grid):
            Av[r][te] = cent[i]
        Rv[te] = cent[len(r_grid)]
        covered[te] = True
        per_fold_out[str(f)] = {
            "n_te": int(len(te)),
            "basis_wall_s": fr["basis_wall_s"],
            "r2_ridge_press_fold_basis": _r2(fr["Ypf"][te], fr["ridge_pred"]),
            **{f"r2_r{r}_fold_basis": _r2(fr["Ypf"][te], fr["preds"][r]) for r in r_grid},
        }
        knn_folds.append(
            {
                m: knn_retrieval(fr["preds"][max(r_grid)], fr["Ypf"][te], ks=(1, 5, 10), metric=m)
                for m in ("euclidean", "cosine")
            }
        )

    def _pooled_r2_v(pred_v: np.ndarray) -> float:
        return _r2(Yv[covered], pred_v[covered])

    pooled = {
        "r2_r0_pooled_ss": _pooled_r2_v(Av[0]),
        "r2_r32_pooled_ss": _pooled_r2_v(Av[max(r_grid)]),
        "r2_ridge_press_pooled_ss": _pooled_r2_v(Rv),
    }
    delta_named = cluster_bootstrap_delta_r2(
        Yv, Av[max(r_grid)], Av[0], covered, groups, n_draws=n_draws, seed=0
    )
    dereg = cluster_bootstrap_delta_r2(Yv, Av[0], Rv, covered, groups, n_draws=n_draws, seed=0)
    repro_gap = abs(pooled["r2_r0_pooled_ss"] - pooled["r2_ridge_press_pooled_ss"])
    repro = {
        "abs_gap_r0_vs_ridge_press": repro_gap,
        "tolerance": FOLDPC_REPRO_TOL,
        "passed": bool(repro_gap <= FOLDPC_REPRO_TOL),
        "note": (
            "plan section 8 risk row 3: the r=0 GD refit IS the warm-start/protocol "
            "check in the fold basis; production gap > tol halts cell 2 (rc=24)"
        ),
    }
    committed_ref = None
    ref_path = eval_dir("bilinear") / "bilinear_fits.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text()).get("schemes", {}).get("prefix", {})
        committed_ref = {
            "delta_named_full_population_pc": ref.get("delta_named"),
            "r_star_inner_val": ref.get("r_star_inner_val"),
        }
    out = {
        "meta": result_meta(
            smoke=bool(args.smoke),
            basis="pca48_foldpc",
            r_grid=list(r_grid),
            seeds=seeds,
            n_draws=n_draws,
            row_limit=args.row_limit,
        ),
        "scheme": "prefix",
        "grouping_unit": "prefix_id",
        "r_star_carried": max(r_grid),
        "n_rows_covered": int(covered.sum()),
        "per_fold": per_fold_out,
        "pooled_per_fold_ss": pooled,
        "delta_named_foldpc": delta_named,
        "dereg_component_r0_minus_ridge_press": dereg,
        "r0_ridge_reproduction": repro,
        "committed_full_population_reference": committed_ref,
        "baselines": {
            "identity_bias": (
                "inapplicable — d_in 7168 != d_out 48 (stated, per the standing rule)"
            ),
            "knn_retrieval_per_fold_r_star": knn_folds,
        },
        "note": (
            "per-fold TRAIN-FOLD-ONLY 48-PC bases (deviation discharge); per-fold R2 "
            "in the fold's own basis pooled via per-fold SS sums; CIs = paired "
            "prefix-group cluster bootstrap on fold-centered arrays"
        ),
    }
    atomic_write_json(out_dir / "bilinear_foldpc.json", out)
    upload_phase_tensors("bilinear_params_foldpc", smoke=bool(args.smoke))
    upload_phase_eval_json(FU_SUB, smoke=bool(args.smoke))
    print(
        f"[foldpc] done in {(time.monotonic() - t0) / 60:.1f} min "
        f"(delta_named_foldpc={delta_named['delta_r2']:.4f} repro_gap={repro_gap:.4f})",
        flush=True,
    )
    if not repro["passed"]:
        if args.smoke:
            print(
                "[foldpc] repro gap exceeds tol at SMOKE n — informational only "
                "(#1345 gate-calibration parity; production-n gate unchanged)",
                flush=True,
            )
        else:
            print(
                f"[foldpc] REPRO GATE FAILED: |R2(r=0) - R2(ridge_press)| = {repro_gap:.4f} "
                f"> {FOLDPC_REPRO_TOL} — halting cell 2 (rc={FOLDPC_REPRO_RC})",
                flush=True,
            )
            return FOLDPC_REPRO_RC
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 P4 rank-r bilinear + interpretation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--row-limit", type=int, default=None)
    ap.add_argument("--r-grid", default=",".join(str(r) for r in R_GRID))
    ap.add_argument("--schemes", default=None, help="csv override (e.g. 'doubly' for the r* pass)")
    ap.add_argument(
        "--basis",
        default="pca48",
        choices=["pca48", "pca48_foldpc"],
        help="pca48 = run-1 full-population basis; pca48_foldpc = fu-round cell 2 "
        "(train-fold-only PCs; restricted to --schemes prefix --r-grid 0,32)",
    )
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--assemble-only",
        action="store_true",
        help="skip fits; assemble ALL shards' unit JSONLs + interpretation",
    )
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.basis == "pca48_foldpc":
        # fu-round cell 2: fold-PC sensitivity path (own smoke handling + seeds parse)
        return run_foldpc(args)
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    r_grid = [int(r) for r in args.r_grid.split(",") if r.strip()]
    if args.smoke:
        if args.row_limit is None:
            args.row_limit = 600
        r_grid = [0, 1]
        args.seeds = args.seeds[:1]
    out_dir = eval_dir("bilinear")
    store = resolve_store_dir()
    ad = build_arm_data(
        store,
        CELL_PRIMARY,
        LAYER_PRIMARY,
        arms=("prefix_end", "bare_query", "stitch", "context_end"),
        row_limit=args.row_limit,
    )
    X = ad.X["stitch"]
    if args.schemes:
        schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    else:
        schemes = ["prefix"] if args.smoke else ["prefix", "query"]
    units_path = out_dir / f"units_shard{args.shard_index}.jsonl"
    done = {
        unit_key(d, REGIME_KEYS) for d in load_units_validated(units_path, bilinear_row_incomplete)
    }
    # target bases
    Yp, info_p = _basis_targets_with_info(
        ad.Y_stacked,
        "pca48",
        hidden_dim=3584,
        targets=["t1", "t2", "t3"],
        projection_target="t1",
    )
    Yp = np.ascontiguousarray(Yp, dtype=np.float64)
    enum = [
        {
            "scheme": sch,
            "fold": f,
            "r": r,
            "basis": "pca48",
            "smoke": bool(args.smoke),
            "row_limit": args.row_limit,
        }
        for sch in schemes
        for r in r_grid
        for f in range(6)
    ]
    # shard split over the RAW enumeration (byte-identical shard assignment to the
    # pre-gate code; assembly dedups by unit key across all shard files either way)
    unit_list = [u for i, u in enumerate(enum) if i % args.num_shards == args.shard_index]
    if args.assemble_only:
        unit_list = []
    pairs_by_scheme = {
        sch: restrict_pairs(fold_pairs(ad.rows, len(ad.rows), sch), ad.arm_row_mask["stitch"])
        for sch in ("prefix", "query", "doubly")
    }
    # completeness-gate reference (the P3 crash-fix class ported to P4): the FULL
    # planned enumeration across ALL shards for THIS invocation's schemes x r_grid;
    # folds a scheme never realizes (fold >= len(pairs)) skip in the fit loop and
    # are excluded here too.
    planned = [u for u in enum if u["fold"] < len(pairs_by_scheme[u["scheme"]])]
    params_dir = tensors_dir("bilinear_params")
    t0 = time.monotonic()
    n_done = 0
    for u in unit_list:
        if unit_key(u, REGIME_KEYS) in done:
            continue
        pairs = pairs_by_scheme[u["scheme"]]
        if u["fold"] >= len(pairs):
            continue
        tr, te = pairs[u["fold"]]
        groups = ad.prefix_ids if u["scheme"] != "query" else ad.query_ids
        Xn = _standardize_train(X, tr)
        res = bilinear_fit_batched(
            Xn,
            Yp,
            tr,
            te,
            groups,
            r=u["r"],
            seeds=args.seeds,
            device=args.device,
            max_epochs=40 if args.smoke else BILIN_MAX_EPOCHS,
        )
        rec = {**u, "epochs_ran": res["epochs_ran"], "variants": []}
        for var in res["variants"]:
            r2_te = _r2(Yp[te], var["pred_te"])
            rec["variants"].append(
                {
                    "seed": var["seed"],
                    "wd": var["wd"],
                    "inner_val_mse": var["inner_val_mse"],
                    "r2_te": r2_te,
                }
            )
            np.save(
                params_dir
                / (f"pred_{u['scheme']}_f{u['fold']}_r{u['r']}_s{var['seed']}_wd{var['wd']:g}.npy"),
                var["pred_te"].astype(np.float16),
            )
            torch.save(
                {k: v for k, v in var["params"].items()},
                params_dir
                / (
                    f"params_{u['scheme']}_f{u['fold']}_r{u['r']}_s{var['seed']}_wd{var['wd']:g}.pt"
                ),
            )
        np.save(params_dir / f"te_{u['scheme']}_f{u['fold']}.npy", te)
        append_unit(units_path, rec)
        n_done += 1
        print(
            f"[bilinear] unit {n_done}/{len(unit_list)} scheme={u['scheme']} "
            f"fold={u['fold']} r={u['r']} elapsed={time.monotonic() - t0:.0f}s",
            flush=True,
        )
    if args.num_shards > 1 and not args.assemble_only:
        print("[bilinear] shard done (assembly = separate --assemble-only pass)", flush=True)
        return 0
    # ---- assembly (single-shard runs and --assemble-only both land here) ----
    raw_rows: list[dict] = []
    for sp in sorted(out_dir.glob("units_shard*.jsonl")):
        raw_rows.extend(load_units_validated(sp, bilinear_row_incomplete))
    by_key: dict[tuple, dict] = {}
    for row in raw_rows:
        by_key[unit_key(row, REGIME_KEYS)] = row  # last row wins (purged+rerun units append anew)
    # completeness gate (mirrors ladder.py; run.sh route_rc names rc=23): refuse
    # a partial bilinear_fits.json when a planned unit has no completed row.
    # NOTE all_units keeps EVERY valid row, not planned-only — the doubly override
    # pass re-enters this assembly and must still see the main pass's prefix/query
    # rows (assemble() covers all schemes present across shard files).
    missing = [u for u in planned if unit_key(u, REGIME_KEYS) not in by_key]
    if missing:
        print(
            f"[bilinear] ASSEMBLY INCOMPLETE: {len(missing)}/{len(planned)} planned units "
            "have no completed row — refusing to write a partial bilinear_fits.json (rc=23)",
            flush=True,
        )
        for u in missing[:25]:
            print(
                f"[bilinear]   missing scheme={u['scheme']} fold={u['fold']} r={u['r']}",
                flush=True,
            )
        return ASSEMBLY_INCOMPLETE_RC
    all_units = list(by_key.values())
    assembled = assemble(
        all_units, ad, Yp, info_p, X, pairs_by_scheme, args, out_dir, params_dir, store
    )
    if planned and not assembled["fits"]["schemes"]:
        print(
            "[bilinear] ASSEMBLY ERROR: no scheme entries assembled though planned units "
            "exist (rc=23)",
            flush=True,
        )
        return ASSEMBLY_INCOMPLETE_RC
    atomic_write_json(out_dir / "bilinear_fits.json", assembled["fits"])
    atomic_write_json(out_dir / "interaction_projections.json", assembled["projections"])
    upload_phase_tensors("bilinear_params", smoke=args.smoke)
    upload_phase_eval_json("bilinear", smoke=args.smoke)
    print(f"[bilinear] done in {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    return 0


def _best_variant(variants: list[dict], seed: int) -> dict:
    """Per-seed wd selection on inner val (nested tuning)."""
    cand = [v for v in variants if v["seed"] == seed]
    return min(cand, key=lambda v: v["inner_val_mse"])


def _pooled_pred(
    units: list[dict],
    params_dir: Path,
    scheme: str,
    r: int,
    seeds: list[int],
    n_rows: int,
    d_out: int,
    pairs,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-over-seeds held-out prediction (per-seed wd selected on inner val)."""
    pred = np.zeros((n_rows, d_out))
    covered = np.zeros(n_rows, dtype=bool)
    for f, (_tr, te) in enumerate(pairs):
        recs = [u for u in units if u["scheme"] == scheme and u["fold"] == f and u["r"] == r]
        if not recs:
            continue
        acc = np.zeros((len(te), d_out))
        for s in seeds:
            v = _best_variant(recs[0]["variants"], s)
            p = params_dir / f"pred_{scheme}_f{f}_r{r}_s{s}_wd{v['wd']:g}.npy"
            acc += np.load(p).astype(np.float64)
        pred[te] = acc / len(seeds)
        covered[te] = True
    return pred, covered


def assemble(units, ad, Yp, info_p, X, pairs_by_scheme, args, out_dir, params_dir, store):
    seeds = args.seeds
    n, d_out = Yp.shape
    fits: dict = {
        "meta": result_meta(smoke=args.smoke, r_grid=sorted({u["r"] for u in units})),
        "schemes": {},
    }
    projections: dict = {"meta": fits["meta"]}
    r_values = sorted({u["r"] for u in units})
    for scheme in sorted({u["scheme"] for u in units}):
        pairs = pairs_by_scheme[scheme]
        groups = ad.prefix_ids if scheme != "query" else ad.query_ids
        # inner-val r* selection over r >= 1 (mean over folds x seeds of best-wd val mse)
        inner: dict[int, list[float]] = {r: [] for r in r_values}
        r2_curve: dict[int, float] = {}
        for r in r_values:
            for f in range(len(pairs)):
                recs = [
                    u for u in units if u["scheme"] == scheme and u["fold"] == f and u["r"] == r
                ]
                if recs:
                    for s in seeds:
                        inner[r].append(_best_variant(recs[0]["variants"], s)["inner_val_mse"])
            pred_r, cov_r = _pooled_pred(units, params_dir, scheme, r, seeds, n, d_out, pairs)
            if cov_r.any():
                r2_curve[r] = _r2(Yp[cov_r], pred_r[cov_r])
        cand = [r for r in r_values if r >= 1 and inner[r]]
        r_star = min(cand, key=lambda r: float(np.mean(inner[r]))) if cand else None
        sch_out: dict = {
            "r_star_inner_val": r_star,
            "inner_val_mse_by_r": {str(r): float(np.mean(v)) for r, v in inner.items() if v},
            "outer_r2_curve_EXPLORATORY": {str(r): float(v) for r, v in r2_curve.items()},
        }
        if r_star is not None and 0 in r_values:
            pred_star, cov_star = _pooled_pred(
                units, params_dir, scheme, r_star, seeds, n, d_out, pairs
            )
            pred_0, cov_0 = _pooled_pred(units, params_dir, scheme, 0, seeds, n, d_out, pairs)
            both = cov_star & cov_0
            if scheme == "doubly":
                boot = two_way_cluster_bootstrap_delta_r2(
                    Yp,
                    pred_star,
                    pred_0,
                    both,
                    ad.prefix_ids,
                    ad.query_ids,
                    n_draws=200 if args.smoke else 2000,
                )
            else:
                boot = cluster_bootstrap_delta_r2(
                    Yp,
                    pred_star,
                    pred_0,
                    both,
                    groups,
                    n_draws=200 if args.smoke else 2000,
                )
            sch_out["delta_named"] = boot
            # de-regularization component vs the P1 stitch PRESS-ridge
            ridge_p = (
                tensors_dir("heldout_preds")
                / f"{CELL_PRIMARY}_L{LAYER_PRIMARY:02d}_stitch_perrow_pca48_{scheme}_ridge.npy"
            )
            if ridge_p.exists():
                rpred = np.load(ridge_p).astype(np.float64)
                rmask = np.load(ridge_p.with_name(ridge_p.stem + "_mask.npy"))
                b2 = both & rmask
                sch_out["dereg_component_r0_minus_ridge"] = cluster_bootstrap_delta_r2(
                    Yp, pred_0, rpred, b2, groups, n_draws=200 if args.smoke else 2000
                )
                sch_out["interaction_gap_fraction"] = _gap_fraction(
                    units, ad, Yp, pred_star, rpred, b2, scheme
                )
            # delta_beyond vs the P3 stitch-MLP (same folds, same input)
            mlp_preds = []
            for s in seeds:
                mp = (
                    tensors_dir("heldout_preds")
                    / f"{CELL_PRIMARY}_L{LAYER_PRIMARY:02d}_stitch_perrow_pca48_{scheme}_mlp_s{s}.npy"
                )
                if mp.exists():
                    mlp_preds.append(np.load(mp).astype(np.float64))
            if mlp_preds:
                mpred = np.mean(mlp_preds, axis=0)
                mmask = np.load(
                    (
                        tensors_dir("heldout_preds")
                        / f"{CELL_PRIMARY}_L{LAYER_PRIMARY:02d}_stitch_perrow_pca48_{scheme}_mlp_s{seeds[0]}_mask.npy"
                    )
                )
                b3 = both & mmask
                sch_out["delta_beyond_mlp_minus_bilinear"] = cluster_bootstrap_delta_r2(
                    Yp, mpred, pred_star, b3, groups, n_draws=200 if args.smoke else 2000
                )
            else:
                sch_out["delta_beyond_mlp_minus_bilinear"] = (
                    "stitch-MLP predictions absent (P3 pending or Gate-B skipped)"
                )
            # bilinear-residual HSIC re-test (P2 machinery) on the dense block
            if scheme == "prefix":
                sch_out["bilinear_residual_hsic"] = _residual_retest(
                    ad, X, Yp, pred_star, cov_star, args
                )
                # interpretation at r*
                params_star = _collect_params(units, params_dir, scheme, r_star, seeds)
                tr0 = pairs[0][0]
                proj, proj_nulls = projection_analysis(
                    params_star,
                    info_p,
                    Yp[tr0],
                    ad.X["prefix_end"][tr0],
                    ad.X["bare_query"][tr0],
                    load_rb_dictionary(store, LAYER_PRIMARY),
                    n_draws=50 if args.smoke else N_PROJ_DRAWS,
                )
                projections[scheme] = proj
                nd = tensors_dir("null_matrices")
                for k, v in proj_nulls.items():
                    np.save(nd / f"proj_null_{k}.npy", v.astype(np.float32))
                # plan-named AMBIENT companion at r in {0, r*} (round-2 Major-3).
                # Persisted to its OWN JSON and loaded when present: the p4 chain
                # re-enters assemble() up to three times (main run, unconditional
                # --assemble-only pass, doubly override pass) and must fit the
                # 224x ambient head exactly ONCE.
                import json as _json

                amb_p = out_dir / "ambient_companion.json"
                amb_loaded = _json.loads(amb_p.read_text()) if amb_p.exists() else None
                if amb_loaded is not None and not (
                    amb_loaded.get("meta", {}).get("smoke") == bool(args.smoke)
                    and amb_loaded.get("r_star_from_pca48") == int(r_star)
                ):
                    # round-2 Minor-1 regime check: never present a wrong-regime
                    # companion (e.g. smoke-scale JSON in a production out-root,
                    # or a stale r*) as current — refit instead of loading.
                    print(
                        "[bilinear/ambient] ambient_companion.json regime mismatch "
                        f"(meta.smoke={amb_loaded.get('meta', {}).get('smoke')} vs "
                        f"{bool(args.smoke)}; r_star_from_pca48="
                        f"{amb_loaded.get('r_star_from_pca48')} vs {int(r_star)}) — refitting",
                        flush=True,
                    )
                    amb_loaded = None
                if amb_loaded is not None:
                    sch_out["ambient_companion"] = amb_loaded
                elif args.schemes is None:
                    amb = ambient_companion(ad, X, pairs, r_star, args, params_dir)
                    amb["meta"] = result_meta(smoke=args.smoke, r_star=int(r_star))
                    atomic_write_json(amb_p, amb)
                    sch_out["ambient_companion"] = amb
                else:
                    sch_out["ambient_companion"] = (
                        "absent — scheme-override pass ran before the main pass recorded it"
                    )
        fits["schemes"][scheme] = sch_out
    # doubly-novel robustness read at {0, r*} runs as a separate dispatcher pass
    # (--r-grid "0,<r*>" over scheme doubly); assembled here when present.
    return {"fits": fits, "projections": projections}


def ambient_companion(ad, X, pairs, r_star, args, params_dir) -> dict:
    """P4 AMBIENT-target companion at r in {0, r*} (plan section 4 P4; round-2 Major-3).

    Same protocol as the pca48 sweep — warm-start stitch ridge, wd grid, seed
    ensemble, IDENTICAL folds — with ambient targets; r* is INHERITED from the
    pca48 inner-val selection (not re-selected). Scoping: PREFIX fold scheme
    only (recorded in the output; the pca48 primary carries the doubly/query
    robustness reads).
    """
    Ya, _info_a = _basis_targets_with_info(
        ad.Y_stacked,
        "ambient",
        hidden_dim=3584,
        targets=["t1", "t2", "t3"],
        projection_target="t1",
    )
    Ya = np.ascontiguousarray(Ya, dtype=np.float64)
    n, d_out = Ya.shape
    groups = ad.prefix_ids
    # smoke slice: the ambient head (d_out=10752) is 224x the pca48 head — cover
    # the code path on 2 folds (>= 2 test-row groups per fold, bootstrap fine);
    # production runs ALL folds (geometry byte-untouched).
    if args.smoke:
        pairs = pairs[:2]
    preds: dict[int, np.ndarray] = {}
    covers: dict[int, np.ndarray] = {}
    t0 = time.monotonic()
    n_fits_total = 2 * len(pairs)
    n_fits_done = 0
    # round-2 CONCERN ambient-companion-no-intra-loop-resume: per-(f, r) pred
    # files are durable — resume them on a crash-restart iff the regime sidecar
    # matches (same smoke mode + r*); else purge stale preds and refit all.
    regime = {"smoke": bool(args.smoke), "r_star": int(r_star)}
    regime_p = params_dir / "ambient_pred_regime.json"
    resume_ok = regime_p.exists() and json.loads(regime_p.read_text()) == regime
    if not resume_ok:
        for stale in sorted(params_dir.glob("pred_ambient_prefix_f*_r*.npy")):
            print(f"[bilinear/ambient] purging non-matching-regime {stale.name}", flush=True)
            stale.unlink()
        atomic_write_json(regime_p, regime)
    for r in (0, int(r_star)):
        pred = np.zeros((n, d_out))
        cov = np.zeros(n, dtype=bool)
        for f, (tr, te) in enumerate(pairs):
            pred_p = params_dir / f"pred_ambient_prefix_f{f}_r{r}.npy"
            if resume_ok and pred_p.exists():
                cached = np.load(pred_p)
                if cached.shape == (len(te), d_out):
                    pred[te] = cached.astype(np.float64)
                    cov[te] = True
                    n_fits_done += 1
                    print(
                        f"[bilinear/ambient] RESUME r={r} fold={f}: loaded {pred_p.name} "
                        f"({n_fits_done}/{n_fits_total}), skipping fit",
                        flush=True,
                    )
                    continue
            Xn = _standardize_train(X, tr)
            res = bilinear_fit_batched(
                Xn,
                Ya,
                tr,
                te,
                groups,
                r=r,
                seeds=args.seeds,
                device=args.device,
                max_epochs=40 if args.smoke else BILIN_MAX_EPOCHS,
            )
            acc = np.zeros((len(te), d_out))
            for s in args.seeds:
                acc += _best_variant(res["variants"], s)["pred_te"].astype(np.float64)
            pred[te] = acc / len(args.seeds)
            cov[te] = True
            np.save(pred_p, pred[te].astype(np.float16))
            n_fits_done += 1
            elapsed = time.monotonic() - t0
            print(
                f"[bilinear/ambient] r={r} fold={f} fit {n_fits_done}/{n_fits_total} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
            if n_fits_done == 1:
                # in-run pilot projection (plan section 9 P4 pilot spirit): the
                # ambient head is 224x the pca48 head, so surface the projected
                # companion wall loudly for the poller/orchestrator.
                print(
                    f"[bilinear/ambient] PILOT: first fit {elapsed:.0f}s -> projected "
                    f"companion wall ~{elapsed * n_fits_total / 3600:.2f}h "
                    f"({n_fits_total} fits, epochs_ran={res['epochs_ran']})",
                    flush=True,
                )
        preds[r] = pred
        covers[r] = cov
    both = covers[0] & covers[int(r_star)]
    boot = cluster_bootstrap_delta_r2(
        Ya, preds[int(r_star)], preds[0], both, groups, n_draws=200 if args.smoke else 2000
    )
    return {
        "r_star_from_pca48": int(r_star),
        "delta_named_ambient": boot,
        "r2_ambient_by_r": {
            str(r): _r2(Ya[covers[r]], preds[r][covers[r]]) for r in (0, int(r_star))
        },
        "scoping": (
            "prefix fold scheme only; r in {0, r*} with r* inherited from the pca48 "
            "inner-val selection; same protocol (warm start, wd grid, seed ensemble)"
        ),
    }


def _gap_fraction(units, ad, Yp, pred_bilin, rpred_stitch, mask, scheme) -> dict:
    if scheme != "prefix":
        # round-2 Minor-d: the context-ridge preds below are PREFIX-fold-scheme
        # artifacts; computing the fraction under another scheme mixes fold
        # protocols — restrict to one scheme, never mix.
        return {"note": f"computed under the prefix scheme only (requested: {scheme})"}
    ctx = (
        tensors_dir("heldout_preds")
        / f"{CELL_PRIMARY}_L{LAYER_PRIMARY:02d}_context_end_perrow_pca48_prefix_ridge.npy"
    )
    if not ctx.exists():
        return {"note": "context ridge preds absent"}
    cpred = np.load(ctx).astype(np.float64)
    cmask = np.load(ctx.with_name(ctx.stem + "_mask.npy"))
    m = mask & cmask
    r2_b = _r2(Yp[m], pred_bilin[m])
    r2_s = _r2(Yp[m], rpred_stitch[m])
    r2_c = _r2(Yp[m], cpred[m])
    denom = r2_c - r2_s
    return {
        "r2_bilinear": r2_b,
        "r2_stitch_ridge": r2_s,
        "r2_context_ridge": r2_c,
        "fraction": (r2_b - r2_s) / denom if abs(denom) > 1e-9 else None,
        "note": "secondary read; denominator = this plan's own fits (plan section 4 P4)",
    }


def _collect_params(units, params_dir, scheme, r_star, seeds) -> list[dict]:
    out = []
    for u in units:
        if u["scheme"] != scheme or u["r"] != r_star:
            continue
        for s in seeds:
            v = _best_variant(u["variants"], s)
            p = params_dir / f"params_{scheme}_f{u['fold']}_r{r_star}_s{s}_wd{v['wd']:g}.pt"
            if p.exists():
                d = torch.load(p, map_location="cpu", weights_only=False)
                out.append(
                    {
                        "params": {
                            k: (np.asarray(x) if x is not None else None) for k, x in d.items()
                        }
                    }
                )
    return out


def _residual_retest(ad, X, Yp, pred_star, covered, args) -> dict:
    grid, _p, _q = complete_dense_block(ad.rows)
    flat = grid.reshape(-1)
    if not covered[flat].all() or not ad.arm_row_mask["stitch"][flat].all():
        return {"note": "dense block not fully covered by bilinear predictions"}
    R = Yp[flat] - pred_star[flat]
    Xb = X[flat]
    mats = build_dependence_matrices(Xb, R, device=args.device)
    obs = observed_stats(mats)
    ref = hsic_statistic(Xb, R)
    assert abs(obs["hsic"] - ref) <= 1e-8 * max(1.0, abs(ref))
    out = {"observed": obs, "schemes": {}}
    from issue1775_common import crossed_permutations

    P, Q = grid.shape
    for scheme in DETECT_SCHEMES:
        perms = crossed_permutations(P, Q, scheme, 20 if args.smoke else 1000, seed=5)
        ns = null_stats_batched(mats, perms)
        out["schemes"][scheme] = {
            "p_hsic": p_value(ns["hsic"], obs["hsic"]),
            "p_dcor": p_value(ns["dcor"], obs["dcor"]),
        }
    out["note"] = (
        "the 'named' verdict co-signature: bilinear-residual dependence must drop "
        "toward the null (plan section 3)"
    )
    return out


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
