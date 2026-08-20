"""#2388 Step 0-p pinned tests for the ``dof_cap`` extension of the per-target GCV core.

Three pins (plan #2388 section 4, "Estimator resolution"):

(a)  at n_train > d under the DEFAULT (``dof_cap=None``) the outputs are
     byte-identical to the pre-edit function (frozen reference embedded below);
(a') at n_train < d under the DEFAULT the outputs are byte-identical — pinned
     SEPARATELY because an above-d-only identity test is structurally what would
     let a behaviour change below d survive unnoticed;
(b)  advisory cross-check on one well-posed full-label cell: held-out R^2 from
     the extended per-target path (explicit ``dof_cap=0.9``) vs
     ``scripts/issue2222_analysis.dof_capped_ridge_multi_y`` agree within a
     smoke-calibrated tolerance (the two recipes deliberately differ in X
     scaling — standardize vs center — so this is a parity DIAGNOSTIC, not an
     equality gate).

Plus the cap's own semantics: the capped argmin never selects an inadmissible
lambda, and an all-inadmissible grid raises (the #2222 fail-loud shape).

Byte-identity here is same-machine/same-op-sequence bitwise equality
(``np.array_equal``); float values are never pinned as cross-machine constants
(gotchas.md float-last-bit rule).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739.fits import (
    RIDGE_LAMBDAS,
    ridge_gcv_predict_per_target,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _reference_pre_edit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evals: list[np.ndarray],
    *,
    lambdas=RIDGE_LAMBDAS,
    device: str = "cpu",
    layer_chunk: int = 4,
) -> list[np.ndarray]:
    """FROZEN verbatim copy of the pre-#2388 ``ridge_gcv_predict_per_target``.

    Copied from ``src/explore_persona_space/experiments/issue_1739/fits.py`` at
    commit 0b93d15f86 (the issue-2388 branch point) with ONE change: the
    ``_eigh_robust`` import is taken from the live module (an exact
    numerical-backend helper, not part of the selector semantics). Do NOT
    "modernize" this copy — it is the byte-identity oracle for pins (a)/(a').
    """
    import torch

    from explore_persona_space.experiments.issue_1739.fits import _eigh_robust

    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    assert x.ndim == 3 and y.ndim == 3, (x.shape, y.shape)
    n_slices, ntr, d = x.shape
    n_t = y.shape[2]
    assert y.shape[:2] == (n_slices, ntr), (y.shape, x.shape)
    evs = [np.asarray(e, dtype=np.float64) for e in x_evals]
    for e in evs:
        assert e.ndim == 3 and e.shape[0] == n_slices and e.shape[2] == d, (e.shape, x.shape)
    lam_grid = np.asarray(lambdas, dtype=np.float64)
    dev = torch.device(device)
    primal = ntr > d
    preds = [np.empty((n_slices, e.shape[1], n_t)) for e in evs]
    for lo in range(0, n_slices, layer_chunk):
        sl = slice(lo, min(lo + layer_chunk, n_slices))
        xtr = torch.as_tensor(x[sl], device=dev)
        ytr = torch.as_tensor(y[sl], device=dev)
        xmu = xtr.mean(dim=1, keepdim=True)
        xsd = xtr.std(dim=1, unbiased=False, keepdim=True) + 1e-9
        xn = (xtr - xmu) / xsd
        ymu = ytr.mean(dim=1, keepdim=True)
        yc = ytr - ymu
        tot = (yc**2).sum(dim=1)
        if primal:
            gram = xn.transpose(1, 2) @ xn
            s, v = _eigh_robust(gram)
            s = torch.clamp(s, min=0.0)
            a = v.transpose(1, 2) @ (xn.transpose(1, 2) @ yc)
            sq_a = a**2
        else:
            gram = xn @ xn.transpose(1, 2)
            s, v = _eigh_robust(gram)
            s = torch.clamp(s, min=0.0)
            a = v.transpose(1, 2) @ yc
            sq_a = a**2
        c = s.shape[0]
        gcv = torch.empty((c, len(lam_grid), n_t), dtype=torch.float64, device=dev)
        for li, lam in enumerate(lam_grid):
            lam_f = float(lam)
            if primal:
                shrink = (s + 2.0 * lam_f) / (s + lam_f) ** 2
                rss = tot - torch.einsum("cdt,cd->ct", sq_a, shrink)
                dof = (s / (s + lam_f)).sum(dim=1)
            else:
                filt = s / (s + lam_f)
                rss = tot - torch.einsum("cnt,cn->ct", sq_a, 2.0 * filt - filt**2)
                dof = filt.sum(dim=1)
            denom = (ntr - dof) ** 2
            gcv[:, li, :] = torch.where(
                (denom > 1e-12)[:, None], rss / denom[:, None], torch.full_like(rss, float("inf"))
            )
        best = gcv.argmin(dim=1)
        lam_sel = torch.as_tensor(lam_grid, device=dev)[best]
        f_sel = 1.0 / (s[:, :, None] + lam_sel[:, None, :])
        if primal:
            w = v @ (a * f_sel)
            for ei, e in enumerate(evs):
                xen = (torch.as_tensor(e[sl], device=dev) - xmu) / xsd
                preds[ei][sl] = (xen @ w + ymu).cpu().numpy()
        else:
            alpha = v @ (a * f_sel)
            xnt = xn.transpose(1, 2)
            w = xnt @ alpha
            for ei, e in enumerate(evs):
                xen = (torch.as_tensor(e[sl], device=dev) - xmu) / xsd
                preds[ei][sl] = (xen @ w + ymu).cpu().numpy()
    return preds


def _toy(n_slices: int, ntr: int, d: int, n_t: int, nev: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_slices, ntr, d))
    w = rng.normal(size=(n_slices, d, n_t))
    y = x @ w + 0.5 * rng.normal(size=(n_slices, ntr, n_t))
    ev = rng.normal(size=(n_slices, nev, d))
    return x, y, [ev]


def test_a_default_none_byte_identical_primal():
    """(a) n_train > d, default cap: bitwise-identical to the pre-edit function."""
    x, y, evs = _toy(n_slices=5, ntr=48, d=12, n_t=3, nev=9, seed=0)
    ref = _reference_pre_edit(x, y, evs)
    # Both the omitted-kwarg form and the explicit dof_cap=None form must match.
    for kwargs in ({}, {"dof_cap": None}):
        got = ridge_gcv_predict_per_target(x, y, evs, **kwargs)
        assert len(got) == len(ref) == 1
        assert np.array_equal(got[0], ref[0]), "primal default-None output diverged bitwise"


def test_a_prime_default_none_byte_identical_dual():
    """(a') n_train < d, default cap: bitwise-identical — pinned SEPARATELY."""
    x, y, evs = _toy(n_slices=4, ntr=10, d=24, n_t=2, nev=7, seed=1)
    ref = _reference_pre_edit(x, y, evs)
    for kwargs in ({}, {"dof_cap": None}):
        got = ridge_gcv_predict_per_target(x, y, evs, **kwargs)
        assert np.array_equal(got[0], ref[0]), "dual default-None output diverged bitwise"


def test_cap_masks_inadmissible_lambdas():
    """The capped selector never picks a lambda whose dof exceeds cap * ntr."""
    x, y, evs = _toy(n_slices=3, ntr=40, d=8, n_t=2, nev=5, seed=2)
    telem_cap: list[dict] = []
    ridge_gcv_predict_per_target(x, y, evs, dof_cap=0.1, selector_telemetry=telem_cap)
    assert telem_cap and telem_cap[0]["mode"] == "gcv-dof-capped"
    for chunk in telem_cap:
        assert np.all(chunk["dof_selected"] <= 0.1 * chunk["n_train"] + 1e-12), (
            "capped selector chose an inadmissible lambda"
        )
    telem_none: list[dict] = []
    ridge_gcv_predict_per_target(x, y, evs, selector_telemetry=telem_none)
    assert telem_none[0]["mode"] == "gcv" and telem_none[0]["dof_cap"] is None


def test_cap_raises_when_no_lambda_admissible():
    """An unsatisfiable cap fails loud naming the cap (the #2222 shape)."""
    x, y, evs = _toy(n_slices=2, ntr=30, d=6, n_t=2, nev=4, seed=3)
    # With tiny lambdas only, dof ~ d = 6 for every grid point; cap*ntr = 0.03.
    with pytest.raises(ValueError, match="no lambda admissible under dof_cap"):
        ridge_gcv_predict_per_target(x, y, evs, lambdas=(1e-8, 1e-7), dof_cap=0.001)


def _load_issue2222():
    path = REPO_ROOT / "scripts" / "issue2222_analysis.py"
    spec = importlib.util.spec_from_file_location("issue2222_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("issue2222_analysis", mod)
    spec.loader.exec_module(mod)
    return mod


def test_b_advisory_crosscheck_vs_issue2222_dof_capped_core():
    """(b) held-out R^2 parity vs ``dof_capped_ridge_multi_y`` on a well-posed cell.

    ADVISORY parity diagnostic: the per-target core STANDARDIZES X (population
    std) while the #2222 core only CENTERS it, so lambda selections legitimately
    differ; the tolerance below is smoke-calibrated (observed |dR^2| ~ 1e-3 on
    this fixture; bar set with ~30x headroom while still catching a wrong-column
    / wrong-fold class of defect, which reads as |dR^2| >> 0.1).
    """
    mod = _load_issue2222()
    rng = np.random.default_rng(7)
    n, d, n_t = 160, 10, 2
    x = rng.normal(size=(n, d))
    w = rng.normal(size=(d, n_t))
    y = x @ w + 1.0 * rng.normal(size=(n, n_t))
    fold_ids = np.arange(n) % 2
    lambdas = np.logspace(-3, 6, 19)

    ref = mod.dof_capped_ridge_multi_y(x, y, fold_ids, lambdas=lambdas, dof_cap=0.9)

    heldout = np.full((n, n_t), np.nan)
    for f in (0, 1):
        hold = fold_ids == f
        tr = ~hold
        preds = ridge_gcv_predict_per_target(
            x[tr][None, :, :],
            y[tr][None, :, :],
            [x[hold][None, :, :]],
            lambdas=lambdas,
            dof_cap=0.9,
        )
        heldout[hold] = preds[0][0]
    resid = y - heldout
    ss_res = (resid**2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot

    assert np.all(np.isfinite(r2)) and np.all(np.isfinite(ref["heldout_r2"]))
    diff = np.abs(r2 - ref["heldout_r2"])
    assert np.all(diff < 0.03), (
        f"per-target capped path vs issue2222 core: held-out R^2 diverged {diff} "
        "(advisory parity bar 0.03, smoke-calibrated — X-scaling difference expected)"
    )
