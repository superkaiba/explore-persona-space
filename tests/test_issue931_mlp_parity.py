"""#931 G1b parent-parity + loss="mse" coverage for the batched split-MLP fitter.

Pins the r1 code-review fixes:
  - `_mlp_fold_r2` (scripts/issue931_fit_cells.py) reproduces the parent
    `fit_h.mlp_fit_predict` recipe — full fold-train X standardization before
    the val split, full fold-train target PCA (parent skip when dim <= pca_k),
    rng(42) 10% val split, patience-20 early stopping with the parent's 1e-6
    improvement threshold, AdamW + MSE — so the pooled group-CV R² of the
    batched path matches a serial parent loop within the G1b tolerance (the
    residual delta is the init draw: per-member key-seeded vs the parent's
    global manual_seed(42)).
  - `fit_batched_split_mlp(loss="mse")` actually trains under MSE: it differs
    from the SmoothL1 default and bit-tracks a serial same-init MSE reference.
  - `patience=` requires validation splits (fail loud).

CPU-only, tiny shapes, no GPU, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.vectorized_mlp_skill import (
    MLP_LR,
    MLP_WD,
    SplitMLPGroup,
    fit_batched_split_mlp,
    split_group_init_seed,
)
from explore_persona_space.experiments.issue_779.fit_h import mlp_fit_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _parity_data(n_groups=12, per=20, d_in=12, h=80, seed=0):
    """(X (n,1,d_in), Y (n,1,h), group ids): near-linear low-noise synthetic.

    h=80 > pca_k=64 keeps the PCA target reduction ACTIVE on both recipes.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * per
    X1 = rng.standard_normal((n, d_in)).astype(np.float32)
    B = rng.standard_normal((d_in, h)).astype(np.float32)
    Y1 = (X1 @ B * 0.1 + 0.02 * rng.standard_normal((n, h))).astype(np.float32)
    groups = np.repeat([f"g{i:02d}" for i in range(n_groups)], per)
    return X1[:, None, :], Y1[:, None, :], groups


def test_mlp_fold_r2_matches_parent_mlp_fit_predict():
    """The Codex-sketched G1b parity test: `_mlp_fold_r2(..., n_draws=0)` vs a
    serial `mlp_fit_predict` fold loop on fixed data, within the 0.02 G1b
    tolerance (both fits must also actually learn: R² > 0.5)."""
    import issue825_fit_cells as fit825
    from issue931_fit_cells import _mlp_fold_r2

    X, Y, groups = _parity_data()
    folds, seed, max_epochs = 3, 0, 150
    out = _mlp_fold_r2(
        X,
        Y,
        groups,
        layers=[0],
        n_draws=0,
        folds=folds,
        seed=seed,
        max_epochs=max_epochs,
        device="cpu",
    )
    r2_batched = out["0"]["r2_obs"]

    fold_ids = fit825._cv_folds(groups, folds, seed)
    ss_res = ss_tot = 0.0
    for k in range(folds):
        te = fold_ids == k
        tr = ~te
        pred = mlp_fit_predict(
            X[tr, 0, :], Y[tr, 0, :], X[te, 0, :], device="cpu", max_epochs=max_epochs
        )
        true = Y[te, 0, :].astype(np.float64)
        mu = true.mean(0)
        ss_res += float(((true - pred) ** 2).sum())
        ss_tot += float(((true - mu) ** 2).sum())
    r2_parent = 1.0 - ss_res / ss_tot

    assert r2_parent > 0.5 and r2_batched > 0.5, (r2_parent, r2_batched)
    assert abs(r2_batched - r2_parent) <= 0.02, (r2_batched, r2_parent)


def test_fit_batched_split_mlp_mse_matches_serial_mse_reference():
    """loss="mse" coverage (r1 Minor): the branch is not a no-op vs SmoothL1,
    and it bit-tracks a serial same-init MSE AdamW reference (bmm-vs-Linear
    reduction-order tolerance, same class as assert_split_mlp_matches_serial)."""
    rng = np.random.default_rng(3)
    n_train, n_eval, d, p = 120, 40, 32, 6
    X = rng.standard_normal((n_train + n_eval, d)).astype(np.float32)
    # Large-magnitude targets: residuals exceed SmoothL1's beta=1 linear-tail
    # threshold at init, so MSE and SmoothL1 genuinely train differently.
    Y = (
        X @ rng.standard_normal((d, p)) * 2.0 + 0.5 * rng.standard_normal((n_train + n_eval, p))
    ).astype(np.float32)
    Xtr, Ytr, Xev = X[:n_train], Y[:n_train], X[n_train:]
    grp = SplitMLPGroup(("g",), Xtr, Ytr, Xev)
    hidden, epochs = 16, 30
    kw = dict(seed=658, hidden=hidden, max_epochs=epochs, device="cpu", chunk_size=1)
    res_mse = fit_batched_split_mlp([grp], loss="mse", **kw)
    res_sl1 = fit_batched_split_mlp([grp], **kw)
    assert not np.allclose(res_mse.preds_by_key[("g",)], res_sl1.preds_by_key[("g",)])

    torch.manual_seed(split_group_init_seed(658, ("g",)))
    net = torch.nn.Sequential(
        torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, p)
    )
    mu = Xtr.mean(0)
    sd = Xtr.std(0, ddof=1) + 1e-6
    Xn = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    Xen = torch.from_numpy(((Xev - mu) / sd).astype(np.float32))
    Yt = torch.from_numpy(Ytr)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(net(Xn), Yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        ref = net(Xen).numpy()
    assert np.max(np.abs(res_mse.preds_by_key[("g",)] - ref)) <= 5e-4


def test_corpus_pins_bound(monkeypatch):
    """external-corpus-pins-missing regression pin: the committed PDNC/WikiText
    pins exist (40-hex), and build_pairs' --pdnc-sha DEFAULT is the pin — so
    EVERY invocation path (dispatcher included) checks out the pin. Pre-fix the
    default was None and a fresh production run floated on the PDNC default
    branch / the HF dataset repo HEAD."""
    import re

    import issue931_build_pairs as bp
    import issue931_common as common

    assert re.fullmatch(r"[0-9a-f]{40}", common.PDNC_SHA)
    assert re.fullmatch(r"[0-9a-f]{40}", common.WIKITEXT_REVISION)
    monkeypatch.setattr(sys, "argv", ["issue931_build_pairs.py"])
    args = bp.parse_args()
    assert args.pdnc_sha == common.PDNC_SHA


def test_patience_parity_bit_tracks_serial_fit_h_loop():
    """mlp-parity-recipe-mismatch regression pin (bit-level): the two new
    parity options (standardize_inputs=False + patience=20) make the batched
    fitter track a LITERAL serial replica of fit_h.mlp_fit_predict's training
    loop (post-step val eval, vloss < best_val - 1e-6 improvement, patience-20
    freeze, best-state restore) with the SAME init draw — same best epoch,
    preds equal to bmm-vs-Linear reduction order. Noisy targets so the
    patience early stop genuinely FIRES (asserted). Pre-fix this test fails
    with TypeError: the options did not exist."""
    rng = np.random.default_rng(9)
    n_tr, n_val, n_ev, d, p = 60, 12, 20, 16, 4
    X = rng.standard_normal((n_tr + n_val + n_ev, d)).astype(np.float32)
    Y = (
        0.2 * X @ rng.standard_normal((d, p)) + 1.0 * rng.standard_normal((n_tr + n_val + n_ev, p))
    ).astype(np.float32)
    # Caller pre-standardizes on the full fold-train (train+val) — parent shape.
    full = X[: n_tr + n_val]
    mu, sd = full.mean(0), full.std(0) + 1e-6
    Xn = (X - mu) / sd
    grp = SplitMLPGroup(
        ("g",),
        Xn[:n_tr],
        Y[:n_tr],
        Xn[n_tr + n_val :],
        Xn[n_tr : n_tr + n_val],
        Y[n_tr : n_tr + n_val],
    )
    hidden, max_epochs = 16, 300
    res = fit_batched_split_mlp(
        [grp],
        seed=658,
        hidden=hidden,
        max_epochs=max_epochs,
        device="cpu",
        chunk_size=1,
        loss="mse",
        standardize_inputs=False,
        patience=20,
    )

    torch.manual_seed(split_group_init_seed(658, ("g",)))
    net = torch.nn.Sequential(
        torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, p)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    Xt, Yt = torch.from_numpy(Xn[:n_tr]), torch.from_numpy(Y[:n_tr])
    Xv, Yv = torch.from_numpy(Xn[n_tr : n_tr + n_val]), torch.from_numpy(Y[n_tr : n_tr + n_val])
    Xe = torch.from_numpy(Xn[n_tr + n_val :])
    best_val, best_state, best_ep, bad, last_ep = float("inf"), None, -1, 0, -1
    for ep in range(max_epochs):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(net(Xt), Yt).backward()
        opt.step()
        with torch.no_grad():
            vloss = float(torch.nn.functional.mse_loss(net(Xv), Yv).item())
        last_ep = ep
        if vloss < best_val - 1e-6:
            best_val, best_ep, bad = vloss, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 20:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        ref = net(Xe).numpy()

    assert last_ep < max_epochs - 1, "patience early stop never fired — recalibrate the data"
    assert res.best_val_epoch_by_key[("g",)] == best_ep
    assert np.max(np.abs(res.preds_by_key[("g",)] - ref)) <= 5e-5


def test_patience_requires_validation():
    """patience= without val splits fails loud (parent parity needs a val set)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((32, 6)).astype(np.float32)
    Y = rng.standard_normal((32, 2)).astype(np.float32)
    grp = SplitMLPGroup(("g",), X[:24], Y[:24], X[24:])
    with pytest.raises(AssertionError, match="patience"):
        fit_batched_split_mlp([grp], seed=658, hidden=8, max_epochs=2, patience=20)
