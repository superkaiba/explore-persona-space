"""#1336 backward-pairs round: orthogonal-tier + operator-swap control pins.

CPU-only synthetic fixtures (d ~ 32, n ~ 200; no model loads, no network,
no GPU). Covers: (1) `_orth_fit` returns an orthogonal R (Frobenius residual
against I below a stated tolerance); (2) exact-rotation recovery — t5c
(context rotation) and t5b (both-sides rotation) reach R^2 ~ 1 on
constructions where the target IS a rotation of the source prediction,
while t0 (and a deliberately WRONG rotation) do not; (3) `scale=True`
reduces to `scale=False` when the fitted `s_fwd == 1` (exact-rotation
construction); (4) the operator-swap control actually SUBSTITUTES the donor
— swapped predictions differ from the real ones (rand + donor seams), are
seed-deterministic, and t8_swap_rand collapses relative to t8 on a
real-signal fixture (a no-op-wired swap would fail both the differs assert
and the collapse assert); (5) payload/preds wiring (orth_tiers blocks,
operator_swap per-fold seed rows matching the documented sha256 derivation,
full-tier preds for the orth tiers).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue825_map_alignment as ma  # noqa: E402
import issue1336_metric_ladder as ml  # noqa: E402

torch.set_num_threads(2)

GRID = np.logspace(-2, 4, 7)
N, D = 200, 32


def _rand_orth(rng: np.random.Generator, d: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def _source_pair(
    rng: np.random.Generator, n: int = N, d: int = D, noise: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """(xs, ys) with ys a clean linear map of xs (unit-scale, small noise)."""
    xs = rng.normal(size=(n, d))
    w = rng.normal(size=(d, d)) / np.sqrt(d)
    b = rng.normal(size=d)
    ys = xs @ w + b + noise * rng.normal(size=(n, d))
    return xs, ys


def _battery(xs, ys, xt, yt, **kw) -> tuple[dict, dict]:
    """One-layer battery run on (n, d) arrays (tiny bootstrap/null budget)."""
    ids = np.asarray([f"s{i}" for i in range(len(xs))])
    return ml.run_battery_arrays(
        xs[:, None, :],
        ys[:, None, :],
        xt[:, None, :],
        yt[:, None, :],
        ids,
        frozen_layers=(0,),
        null_draws=2,
        n_boot=32,
        boot_seed=77,
        grid=GRID,
        band=0.02,
        full_tier_layers=(0,),
        **kw,
    )


def _fold_setup(xs, ys, xt, yt, n_te: int = 40):
    """fp64 tensors + one train/test split + preps for direct _fold_observed."""
    n = len(xs)
    xs_t = torch.as_tensor(xs, dtype=torch.float64)
    ys_t = torch.as_tensor(ys, dtype=torch.float64)
    xt_t = torch.as_tensor(xt, dtype=torch.float64)
    yt_t = torch.as_tensor(yt, dtype=torch.float64)
    tr = torch.zeros(n, dtype=torch.bool)
    tr[: n - n_te] = True
    te = ~tr
    preps = {
        "s": ml._v2_prep(xs_t[tr], inner_seed=7, n_inner=2),
        "t": ml._v2_prep(xt_t[tr], inner_seed=7, n_inner=2),
        "ys": ml._v2_prep(ys_t[tr], inner_seed=7, n_inner=2),
    }
    return xs_t, ys_t, xt_t, yt_t, tr, te, preps


def test_orth_fit_r_is_orthogonal() -> None:
    """R^T R == I to fp64 precision on an arbitrary (not rotation-linked) fit."""
    rng = np.random.default_rng(0)
    a = torch.as_tensor(rng.normal(size=(N, D)), dtype=torch.float64)
    b = torch.as_tensor(rng.normal(size=(N, D)), dtype=torch.float64)
    fit = ma._orth_fit(a, b)
    r = fit["R"]
    resid = float(torch.linalg.matrix_norm(r.T @ r - torch.eye(D, dtype=r.dtype)))
    print(f"[orth-tiers] ||R^T R - I||_F = {resid:.3e} (tol 1e-9)")
    assert resid < 1e-9, resid


def test_ctx_rotation_recovered_by_t5c() -> None:
    """Target contexts = rotated source contexts, same answers: t5c ~ 1, t0 poor;
    t5cs == t5c (exact rotation => fitted s_fwd == 1)."""
    rng = np.random.default_rng(1)
    xs, ys = _source_pair(rng)
    qc = _rand_orth(rng, D)
    xt = xs @ qc
    yt = ys + 0.01 * rng.normal(size=ys.shape)
    payload, _ = _battery(xs, ys, xt, yt)
    layer = payload["per_layer"]["0"]
    orth = layer["orth_tiers"]
    t0_r2 = layer["raw"]["tiers"]["t0"]["r2"]
    assert orth["t5c"]["raw"]["r2"] > 0.95, orth["t5c"]["raw"]
    assert t0_r2 < 0.5, t0_r2
    assert abs(orth["t5cs"]["raw"]["r2"] - orth["t5c"]["raw"]["r2"]) < 1e-9, (
        orth["t5cs"]["raw"]["r2"],
        orth["t5c"]["raw"]["r2"],
    )


def test_both_sides_rotation_recovered_by_t5b() -> None:
    """Contexts AND answers rotated: t5b ~ 1 while t5c (no answer rotation) and
    t0 stay poor; a deliberately WRONG answer rotation does not recover."""
    rng = np.random.default_rng(2)
    xs, ys = _source_pair(rng)
    qc = _rand_orth(rng, D)
    qa = _rand_orth(rng, D)
    xt = xs @ qc
    yt = ys @ qa
    payload, _ = _battery(xs, ys, xt, yt)
    layer = payload["per_layer"]["0"]
    orth = layer["orth_tiers"]
    assert orth["t5b"]["raw"]["r2"] > 0.95, orth["t5b"]["raw"]
    assert orth["t5c"]["raw"]["r2"] < 0.5, orth["t5c"]["raw"]
    assert layer["raw"]["tiers"]["t0"]["r2"] < 0.5, layer["raw"]["tiers"]["t0"]
    assert abs(orth["t5bs"]["raw"]["r2"] - orth["t5b"]["raw"]["r2"]) < 1e-9

    # Wrong-rotation leg (helper level, same split convention as the battery
    # fits): the FITTED R_ans recovers y_t from y_s; a different random
    # orthogonal matrix in its place does not.
    ys_t = torch.as_tensor(ys, dtype=torch.float64)
    yt_t = torch.as_tensor(yt, dtype=torch.float64)
    tr = torch.zeros(N, dtype=torch.bool)
    tr[:160] = True
    te = ~tr
    fit = ma._orth_fit(ys_t[tr], yt_t[tr])
    q_wrong = torch.as_tensor(_rand_orth(np.random.default_rng(99), D), dtype=torch.float64)
    fit_wrong = {**fit, "R": q_wrong}

    def _r2(pred: torch.Tensor) -> float:
        res = float(((yt_t[te] - pred) ** 2).sum())
        tot = float(((yt_t[te] - yt_t[te].mean(0)) ** 2).sum())
        return 1.0 - res / tot

    good = _r2(ma._orth_predict(fit, ys_t[te], reverse=False, scale=False))
    wrong = _r2(ma._orth_predict(fit_wrong, ys_t[te], reverse=False, scale=False))
    assert good > 0.95, good
    assert wrong < 0.5, wrong


def test_answer_rotation_recovered_by_t5d() -> None:
    """Target answers = rotated source answers, contexts UNCHANGED: t5d ~ 1
    (R_ans on the raw t0 prediction suffices) while t0 stays poor; t5ds == t5d
    (exact rotation => fitted s_fwd == 1). Under a BOTH-sides rotation t5d
    (no context rotation) collapses while t5b recovers — pinning that t5d and
    t5b share R_ans and differ only in R_ctx."""
    rng = np.random.default_rng(12)
    xs, ys = _source_pair(rng)
    qa = _rand_orth(rng, D)
    xt = xs + 0.01 * rng.normal(size=xs.shape)
    yt = ys @ qa
    payload, preds = _battery(xs, ys, xt, yt)
    layer = payload["per_layer"]["0"]
    orth = layer["orth_tiers"]
    t0_r2 = layer["raw"]["tiers"]["t0"]["r2"]
    assert orth["t5d"]["raw"]["r2"] > 0.9, orth["t5d"]["raw"]
    assert t0_r2 < 0.5, t0_r2
    assert abs(orth["t5ds"]["raw"]["r2"] - orth["t5d"]["raw"]["r2"]) < 1e-9
    assert "t5d_l0" in preds and "t5d_recal_l0" in preds, sorted(preds)

    # Both-sides rotation: t5d has no R_ctx, so it collapses where t5b holds.
    qc = _rand_orth(rng, D)
    payload2, _ = _battery(xs, ys, xs @ qc, ys @ qa)
    orth2 = payload2["per_layer"]["0"]["orth_tiers"]
    assert orth2["t5b"]["raw"]["r2"] > 0.95, orth2["t5b"]["raw"]
    assert orth2["t5d"]["raw"]["r2"] < 0.5, orth2["t5d"]["raw"]


def test_scale_true_reduces_to_scale_false_at_unit_s_fwd() -> None:
    """B an EXACT (centered) rotation of A => fitted s_fwd == 1 and the scaled
    prediction equals the unscaled one."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(N, D))
    q = _rand_orth(rng, D)
    b = (a - a.mean(0)) @ q + rng.normal(size=D)  # exact rotation + shift
    a_t = torch.as_tensor(a, dtype=torch.float64)
    b_t = torch.as_tensor(b, dtype=torch.float64)
    fit = ma._orth_fit(a_t, b_t)
    assert abs(fit["s_fwd"] - 1.0) < 1e-10, fit["s_fwd"]
    x_eval = torch.as_tensor(rng.normal(size=(25, D)), dtype=torch.float64)
    p_scaled = ma._orth_predict(fit, x_eval, reverse=False, scale=True)
    p_plain = ma._orth_predict(fit, x_eval, reverse=False, scale=False)
    assert torch.allclose(p_scaled, p_plain, atol=1e-10), float((p_scaled - p_plain).abs().max())


def test_swap_rand_substitutes_and_is_seeded() -> None:
    """The rand donor actually replaces W_s (swapped preds differ from the real
    t6/t7/t8), is deterministic per seed, differs across seeds, and matches
    the real W_s effective-matrix Frobenius norm."""
    rng = np.random.default_rng(4)
    xs, ys = _source_pair(rng)
    dx = 3.0 * rng.normal(size=D)
    xt = xs + dx
    yt = (xt - dx) @ (np.linalg.pinv(xs) @ (ys - ys.mean(0))) + ys.mean(0)
    xs_t, ys_t, xt_t, yt_t, tr, te, preps = _fold_setup(xs, ys, xt, yt)
    te_preds, aux = ml._fold_observed(xs_t, ys_t, xt_t, yt_t, tr, te, GRID, preps, swap_seed=1234)
    for name in ml.SWAP_RAND_NAMES:
        real = te_preds[name.split("_swap_")[0]]
        assert not torch.allclose(te_preds[name], real), name
    meta = aux["swap"]
    assert meta["seed"] == 1234
    assert meta["ws_fro"] > 0
    assert abs(meta["rand_fro"] - meta["ws_fro"]) < 1e-9 * max(meta["ws_fro"], 1.0), meta

    # Determinism: same seed reproduces the swapped predictions exactly;
    # a different seed changes them.
    te_preds2, _ = ml._fold_observed(xs_t, ys_t, xt_t, yt_t, tr, te, GRID, preps, swap_seed=1234)
    te_preds3, _ = ml._fold_observed(xs_t, ys_t, xt_t, yt_t, tr, te, GRID, preps, swap_seed=4321)
    for name in ml.SWAP_RAND_NAMES:
        assert torch.equal(te_preds[name], te_preds2[name]), name
        assert not torch.allclose(te_preds[name], te_preds3[name]), name

    # No swap keys at all when the control is unarmed (pooled-path contract).
    te_preds4, aux4 = ml._fold_observed(xs_t, ys_t, xt_t, yt_t, tr, te, GRID, preps)
    assert not any(k.endswith(("_swap_rand", "_swap_donor")) for k in te_preds4)
    assert aux4["swap"] is None


def test_swap_donor_seam_substitutes() -> None:
    """A donor (prep, yfit) fitted on DIFFERENT data threads through the donor
    seam: donor keys present, predictions differ from the real tiers, and the
    donor id is recorded."""
    rng = np.random.default_rng(5)
    xs, ys = _source_pair(rng)
    dx = 2.0 * rng.normal(size=D)
    xt, yt = xs + dx, ys + 0.01 * rng.normal(size=ys.shape)
    xs_t, ys_t, xt_t, yt_t, tr, te, preps = _fold_setup(xs, ys, xt, yt)
    xd, yd = _source_pair(np.random.default_rng(6))  # a different "pair"
    xd_t = torch.as_tensor(xd, dtype=torch.float64)
    yd_t = torch.as_tensor(yd, dtype=torch.float64)
    prep_d = ml._v2_prep(xd_t[tr], inner_seed=9, n_inner=2)
    yfit_d = ml._v2_yfit(prep_d, yd_t[tr], GRID)
    donor = {"prep": prep_d, "yfit": yfit_d, "donor_id": "synthetic-donor"}
    te_preds, aux = ml._fold_observed(xs_t, ys_t, xt_t, yt_t, tr, te, GRID, preps, donor_ws=donor)
    for name in ml.SWAP_DONOR_NAMES:
        real = te_preds[name.split("_swap_")[0]]
        assert name in te_preds, name
        assert not torch.allclose(te_preds[name], real), name
    assert aux["swap"]["donor_id"] == "synthetic-donor"
    assert not any(k.endswith("_swap_rand") for k in te_preds)  # rand unarmed


def test_battery_swap_collapse_and_wiring() -> None:
    """Battery level: t8_swap_rand collapses relative to t8 on a real-signal
    fixture; per-fold seed rows match the documented sha256 derivation;
    orth-tier preds ride the full-tier npz contract."""
    rng = np.random.default_rng(7)
    xs, ys = _source_pair(rng)
    dx = 3.0 * rng.normal(size=D)
    xt = xs + dx
    yt = ys + 0.01 * rng.normal(size=ys.shape)
    key = "base:sft|lmsys23k|chat"
    payload, preds = _battery(xs, ys, xt, yt, swap_seed_key=key)
    layer = payload["per_layer"]["0"]
    t8_r2 = layer["raw"]["tiers"]["t8"]["r2"]
    swap = layer["operator_swap"]
    assert swap["seed_key"] == key
    assert t8_r2 > 0.9, t8_r2
    for name in ml.SWAP_RAND_NAMES:
        assert swap["reads"][name]["raw"]["r2"] < t8_r2 - 0.2, (name, swap["reads"][name])
    # Seed rows: one per fold, seeds distinct, derivation pinned.
    rows = swap["per_fold"]
    assert len(rows) == payload["n_folds"], rows
    seeds = [r["seed"] for r in rows]
    assert len(set(seeds)) == len(seeds), seeds
    for r in rows:
        expect = int.from_bytes(
            hashlib.sha256(f"{key}|layer=0|fold={r['fold']}".encode()).digest()[:4], "big"
        )
        assert r["seed"] == expect, (r, expect)
        assert r["ws_fro"] > 0 and abs(r["rand_fro"] - r["ws_fro"]) < 1e-6 * r["ws_fro"]
    # Orth tiers: all four blocks with raw+recal, and full-tier preds present.
    for name in ml.ORTH_TIER_NAMES:
        blk = layer["orth_tiers"][name]
        assert {"raw", "recal"} <= set(blk), blk
        assert f"{name}_l0" in preds and f"{name}_recal_l0" in preds, name
    for name in ml.SWAP_RAND_NAMES:  # swap preds are JSON-only by design
        assert f"{name}_l0" not in preds, name


def test_orth_tiers_gated_off_when_not_a_full_tier_layer() -> None:
    """`with_orth=False` omits the four orth keys entirely (no placeholders),
    and a battery run with no full-tier layer emits an EMPTY orth_tiers block
    and no orth preds — the layer-30-only gate (user call 2026-08-12), whose
    whole point is skipping the two d x d Procrustes SVDs per (fold, layer)."""
    rng = np.random.default_rng(11)
    xs, ys = _source_pair(rng)
    xt, yt = xs + rng.normal(size=D), ys + 0.01 * rng.normal(size=ys.shape)
    xs_t, ys_t, xt_t, yt_t, tr, te, preps = _fold_setup(xs, ys, xt, yt)

    on, _ = ml._fold_observed(xs_t, ys_t, xt_t, yt_t, tr, te, GRID, preps)
    off, _ = ml._fold_observed(xs_t, ys_t, xt_t, yt_t, tr, te, GRID, preps, with_orth=False)
    for name in ml.ORTH_TIER_NAMES:
        assert name in on, name
        assert name not in off, name  # omitted, never a placeholder
    # Everything else is untouched by the gate.
    assert set(on) - set(off) == set(ml.ORTH_TIER_NAMES)
    for name in off:
        assert torch.equal(on[name], off[name]), name

    ids = np.asarray([f"s{i}" for i in range(len(xs))])
    payload, preds = ml.run_battery_arrays(  # _battery pins full_tier_layers=(0,)
        xs[:, None, :],
        ys[:, None, :],
        xt[:, None, :],
        yt[:, None, :],
        ids,
        frozen_layers=(0,),
        null_draws=2,
        n_boot=32,
        boot_seed=77,
        grid=GRID,
        band=0.02,
        full_tier_layers=(),
    )
    assert payload["per_layer"]["0"]["orth_tiers"] == {}
    assert not any(k.startswith(tuple(ml.ORTH_TIER_NAMES)) for k in preds), sorted(preds)
