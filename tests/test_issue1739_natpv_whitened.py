"""#1739 natural-PV whitened-space folding — numerical equivalence to the main grid.

The whitened natpv path never materializes a whitened activation grid: it folds
the (linear, symmetric) U-pool whitening into the DIRECTION, so every read
collapses to ``score = x . vec + const`` over the RAW streamed row. These tests
pin that the folded score equals the main grid's own composition
(``apply_whitening`` + the ``einsum("ld,lde->le", rb, wh.w)`` direction whitening
at ``scripts/issue1739_fits.py``) to float tolerance — the whole deliverable
rests on this algebra, so it is exercised through the REAL production helper
(``_whitened_projectors``) against the REAL ``fits`` functions, not a re-derivation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue_1739 import fits  # noqa: E402
from scripts import issue1739_natpv as natpv  # noqa: E402

LY, DIM, N_U, N_ROWS = 28, 16, 120, 40
VARIANTS = ("context_end", "prefix_end")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _fit_and_persist_whitening(args, seed: int = 3) -> dict:
    """Fit the REAL whitening per variant and persist it in the production layout."""
    out = {}
    for i, variant in enumerate(VARIANTS):
        x_u = _rng(seed + i).normal(size=(LY, N_U, DIM))
        wh = fits.fit_whitening(x_u, device="cpu", seed=0)
        path = natpv.whitening_path(args, variant)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez(
                fh,
                mu=np.asarray(wh.mu, dtype=np.float32),
                w=np.asarray(wh.w, dtype=np.float32),
                gamma=np.asarray(wh.gamma, dtype=np.float64),
                meta=json.dumps({"variant": variant, "u_size": "full", "n_u_rows": N_U, "seed": 0}),
            )
        # Round-trip through fp32 so the reference uses the SAME numbers the
        # production path reads back (else the test measures storage precision).
        out[variant] = fits.Whitening(
            mu=np.asarray(np.asarray(wh.mu, dtype=np.float32), dtype=np.float64),
            w=np.asarray(np.asarray(wh.w, dtype=np.float32), dtype=np.float64),
            gamma=wh.gamma,
        )
    return out


def _persisted_map_arrays(seed: int) -> dict:
    """Arrays in the REAL persisted-map layout (the on-disk npz shapes).

    VERIFIED against the live artifacts' npz headers (both
    ``{context_end,prefix_end}__ufull.npz`` on the data repo, read via ranged
    requests 2026-07-31): ``w (28, 3584, 3584) fp16`` and ``x_mu`` / ``x_sd`` /
    ``y_mu`` each ``(28, 1, 3584) fp32`` — matching the ``fits.MapFit`` field
    annotations. The singleton axis is load-bearing: an earlier fixture that
    guessed ``(Ly, d)`` for x_mu/x_sd passed every test and then crashed the
    production fold at ``wl @ h`` (size 1 vs 3584).
    """
    r = _rng(seed)
    return {
        "w": r.normal(size=(LY, DIM, DIM)).astype(np.float16),
        "x_mu": r.normal(size=(LY, 1, DIM)).astype(np.float32),
        "x_sd": (np.abs(r.normal(size=(LY, 1, DIM))) + 0.5).astype(np.float32),
        "y_mu": r.normal(size=(LY, 1, DIM)).astype(np.float32),
    }


def _fake_map(seed: int) -> dict:
    """What ``_map_projectors_raw`` RETURNS — squeezed via the real normalizer."""
    a = _persisted_map_arrays(seed)
    return {
        "w": a["w"],
        "x_mu": natpv._map_row_vecs(a["x_mu"], "x_mu", "test"),
        "x_sd": natpv._map_row_vecs(a["x_sd"], "x_sd", "test"),
        "y_mu": natpv._map_row_vecs(a["y_mu"], "y_mu", "test"),
        "meta": {"apply": "pred = ((x - x_mu)/x_sd) @ w + y_mu (whitened space)"},
        "path": "<test>",
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    args = argparse.Namespace(whitening_root=tmp_path, u_size="full", space="whitened")
    whs = _fit_and_persist_whitening(args)
    maps = {v: _fake_map(11 + i) for i, v in enumerate(VARIANTS)}
    monkeypatch.setattr(natpv, "_map_projectors_raw", lambda variant, stage: maps[variant])
    directions = {
        "e1": _rng(21).normal(size=(LY, DIM)),
        "e2": _rng(22).normal(size=(LY, DIM)),
        "e2p": _rng(23).normal(size=(LY, DIM)),
    }
    proj, prov = natpv._whitened_projectors(args, directions, tmp_path)
    return args, whs, maps, directions, proj, prov


def _folded(proj, read: str, regime: str, x: np.ndarray) -> np.ndarray:
    """Apply the production (vec, const) pair the way phase_project streams it."""
    vec, const = proj[read][regime]
    return np.stack([x[ly] @ vec[ly] + const[ly] for ly in range(LY)])


@pytest.mark.parametrize(
    ("read", "variant"),
    [
        ("ctx", "context_end"),
        ("pre", "prefix_end"),
        ("oracle", "context_end"),
        ("oracle_pre", "prefix_end"),
    ],
)
def test_projection_read_matches_main_grid_composition(setup, read, variant):
    """score = x.vec + const == apply_whitening(x) . (rb @ W), the main grid's own form."""
    _args, whs, _maps, directions, proj, _prov = setup
    x = _rng(77).normal(size=(LY, N_ROWS, DIM))
    wh = whs[variant]
    for regime, rb_raw in directions.items():
        z = fits.apply_whitening(x, wh)  # the REAL whitening application
        rb_w = np.einsum("ld,lde->le", rb_raw, wh.w)  # the REAL direction whitening
        expected = np.einsum("lnd,ld->ln", z, rb_w)
        np.testing.assert_allclose(_folded(proj, read, regime, x), expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    ("read", "variant"), [("map_ctx", "context_end"), ("map_pre", "prefix_end")]
)
def test_map_read_matches_main_grid_composition(setup, read, variant):
    """score == (((z - x_mu)/x_sd) @ w + y_mu) . rb_w with z the whitened row."""
    _args, whs, maps, directions, proj, _prov = setup
    x = _rng(88).normal(size=(LY, N_ROWS, DIM))
    wh, mp = whs[variant], maps[variant]
    for regime, rb_raw in directions.items():
        z = fits.apply_whitening(x, wh)
        rb_w = np.einsum("ld,lde->le", rb_raw, wh.w)
        expected = np.stack(
            [
                (
                    ((z[ly] - mp["x_mu"][ly]) / mp["x_sd"][ly])
                    @ np.asarray(mp["w"][ly], dtype=np.float64)
                    + mp["y_mu"][ly]
                )
                @ rb_w[ly]
                for ly in range(LY)
            ]
        )
        np.testing.assert_allclose(_folded(proj, read, regime, x), expected, rtol=1e-7, atol=1e-7)


def test_whitening_matrix_is_symmetric(setup):
    """The fold relies on W = Sigma_g^{-1/2} being symmetric; pin it on the real fit."""
    _args, whs, *_ = setup
    for variant, wh in whs.items():
        for ly in range(LY):
            np.testing.assert_allclose(
                wh.w[ly],
                wh.w[ly].T,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"{variant} L{ly} whitening not symmetric",
            )


def test_folding_is_not_the_raw_projection(setup):
    """Guards the test itself: whitened != raw, so a no-op fold would fail loudly."""
    _args, _whs, _maps, directions, proj, _prov = setup
    x = _rng(99).normal(size=(LY, N_ROWS, DIM))
    raw = np.einsum("lnd,ld->ln", x, directions["e1"])
    assert not np.allclose(_folded(proj, "ctx", "e1", x), raw, rtol=1e-3, atol=1e-3)


def test_projectors_cover_every_read_and_regime(setup):
    _args, _whs, _maps, directions, proj, prov = setup
    assert set(proj) == {*natpv.READS, natpv.ORACLE_PRE_READ}
    for read in proj:
        assert set(proj[read]) == set(directions)
        for vec, const in proj[read].values():
            assert vec.shape == (LY, DIM) and const.shape == (LY,)
            assert np.isfinite(vec).all() and np.isfinite(const).all()
    assert set(prov["map_meta"]) == set(VARIANTS)


def test_map_projectors_raw_real_body_squeezes_the_persisted_layout(tmp_path, monkeypatch):
    """REAL _map_projectors_raw body on the REAL (Ly, 1, d) on-disk layout.

    Regression pin for the production crash: the persisted npz stores x_mu /
    x_sd / y_mu as (Ly, 1, d), and the fold needs them squeezed to (Ly, d).
    """
    a = _persisted_map_arrays(5)
    path = tmp_path / "context_end__ufull.npz"
    with path.open("wb") as fh:
        np.savez(
            fh,
            **a,
            layers=np.arange(LY),
            meta=json.dumps({"apply": "pred = ((x - x_mu)/x_sd) @ w + y_mu (whitened space)"}),
        )
    monkeypatch.setattr(natpv, "_stage_hf", lambda path_in_repo, dest, revision="main": path)
    monkeypatch.setattr(natpv, "_REPO_ROOT", tmp_path / "nonexistent")
    out = natpv._map_projectors_raw("context_end", tmp_path)
    assert out["w"].shape == (LY, DIM, DIM)
    assert out["x_mu"].shape == out["x_sd"].shape == out["y_mu"].shape == (LY, DIM)
    assert "whitened space" in out["meta"]["apply"]


def test_whitened_fold_survives_the_real_persisted_map_layout(tmp_path, monkeypatch):
    """End-to-end regression for the crash: (Ly, 1, d) map -> fold, no ValueError.

    Drives _whitened_projectors through the REAL loader against an npz written
    in the REAL layout — the exact path that raised
    'matmul: Input operand 1 has a mismatch in its core dimension 0
    (size 1 is different from 3584)' at `v = wl @ h` in production.
    """
    args = argparse.Namespace(whitening_root=tmp_path, u_size="full", space="whitened")
    _fit_and_persist_whitening(args)
    for i, variant in enumerate(VARIANTS):
        a = _persisted_map_arrays(41 + i)
        with (tmp_path / f"{variant}__ufull.npz").open("wb") as fh:
            np.savez(fh, **a, layers=np.arange(LY), meta=json.dumps({"apply": "whitened space"}))
    monkeypatch.setattr(
        natpv,
        "_stage_hf",
        lambda path_in_repo, dest, revision="main": tmp_path / Path(path_in_repo).name,
    )
    monkeypatch.setattr(natpv, "_REPO_ROOT", tmp_path / "nonexistent")
    proj, _prov = natpv._whitened_projectors(
        args, {"e1": _rng(51).normal(size=(LY, DIM))}, tmp_path
    )
    for read in ("map_ctx", "map_pre"):
        vec, const = proj[read]["e1"]
        assert vec.shape == (LY, DIM) and const.shape == (LY,)
        assert np.isfinite(vec).all() and np.isfinite(const).all()


@pytest.mark.parametrize("bad", [(LY, 2, DIM), (LY, DIM, DIM), (DIM,), (LY - 1, 1, DIM)])
def test_map_row_vecs_fails_loud_on_unexpected_layout(bad):
    with pytest.raises(RuntimeError, match="unexpected layout"):
        natpv._map_row_vecs(np.zeros(bad), "x_sd", "context_end")


def test_map_row_vecs_squeezes_and_preserves_values():
    a = _rng(7).normal(size=(LY, 1, DIM))
    out = natpv._map_row_vecs(a, "x_mu", "context_end")
    assert out.shape == (LY, DIM) and out.dtype == np.float64
    np.testing.assert_allclose(out, a[:, 0, :])


def test_load_whitening_missing_fails_loud(tmp_path):
    args = argparse.Namespace(whitening_root=tmp_path, u_size="full", space="whitened")
    with pytest.raises(FileNotFoundError, match="run --phase whitening first"):
        natpv._load_whitening(args, "context_end")


def _stub_u_store(monkeypatch, n_rows: int = N_U + 20):
    """Fake ONLY the U-store boundary (network + multi-GB grid); body runs for real."""
    from explore_persona_space.experiments.issue_1739 import store_io

    r = _rng(31)
    arrays = {(v, ly): r.normal(size=(n_rows, DIM)) for v in VARIANTS for ly in range(LY)}
    meta = [{"is_eval_only": i < 20} for i in range(n_rows)]  # 20 eval-only rows excluded
    calls: dict[str, int] = {"stage": 0}

    def _stage(dest, kinds, layers, **kw):
        calls["stage"] += 1

    monkeypatch.setattr(store_io, "stage_u_store", _stage)
    monkeypatch.setattr(store_io, "load_summaries", lambda *a, **k: (arrays, meta))
    monkeypatch.setattr(
        store_io, "fit_pool_mask", lambda m: np.array([not x["is_eval_only"] for x in m])
    )
    return calls


def test_phase_whitening_real_body_fits_and_persists(tmp_path, monkeypatch):
    """Executes the REAL phase_whitening: fit -> persist -> reload -> usable."""
    calls = _stub_u_store(monkeypatch)
    args = argparse.Namespace(
        whitening_root=tmp_path,
        u_size="full",
        space="whitened",
        u_store=tmp_path / "u",
        whiten_device="cpu",
        whiten_seed=0,
    )
    natpv.phase_whitening(args, "hallucination", tmp_path)
    assert calls["stage"] == 1
    for variant in VARIANTS:
        path = natpv.whitening_path(args, variant)
        assert path.is_file(), f"{variant} whitening not persisted"
        assert not list(path.parent.glob("*.tmp.npz")), "atomic tmp left behind"
        mu, w, meta = natpv._load_whitening(args, variant)
        assert mu.shape == (LY, DIM) and w.shape == (LY, DIM, DIM)
        # Behavior-independent provenance + the excluded eval-only rows.
        assert meta["behavior_independent"] is True and meta["n_u_rows"] == N_U
        assert "fit_whitening" in meta["recipe_source"]


def test_phase_whitening_is_idempotent_across_behaviors(tmp_path, monkeypatch):
    """The 2nd behavior must NOT refit — the transform is behavior-independent."""
    calls = _stub_u_store(monkeypatch)
    args = argparse.Namespace(
        whitening_root=tmp_path,
        u_size="full",
        space="whitened",
        u_store=tmp_path / "u",
        whiten_device="cpu",
        whiten_seed=0,
    )
    natpv.phase_whitening(args, "hallucination", tmp_path)
    stamps = {v: natpv.whitening_path(args, v).stat().st_mtime_ns for v in VARIANTS}
    natpv.phase_whitening(args, "sycophancy", tmp_path)
    assert calls["stage"] == 1, "second behavior re-staged the U store"
    assert {v: natpv.whitening_path(args, v).stat().st_mtime_ns for v in VARIANTS} == stamps


def test_whitening_phase_requires_whitened_space():
    with pytest.raises(SystemExit):
        natpv.main(["--phase", "whitening", "--space", "raw"])


def test_space_scoped_output_names_keep_the_raw_paths_legacy():
    raw = argparse.Namespace(space="raw")
    wht = argparse.Namespace(space="whitened")
    assert (natpv.cube_dir_name(raw), natpv.reduce_out_name(raw)) == (
        "cube",
        "regime_comparison.json",
    )
    assert (natpv.cube_dir_name(wht), natpv.reduce_out_name(wht)) == (
        "cube_whitened",
        "regime_comparison_whitened.json",
    )


# ---------------------------------------------------------------------------
# new-arm-round item 1: fc (--summary-kind context_end) threading
# ---------------------------------------------------------------------------


def test_fc_scoped_output_names_and_regimes():
    raw_fc = argparse.Namespace(space="raw", summary_kind="context_end")
    wht_fc = argparse.Namespace(space="whitened", summary_kind="context_end")
    assert natpv.cube_dir_name(raw_fc) == "cube_fc"
    assert natpv.cube_dir_name(wht_fc) == "cube_whitened_fc"
    assert natpv.reduce_out_name(raw_fc) == "regime_comparison_fc.json"
    assert natpv.reduce_out_name(wht_fc) == "regime_comparison_whitened_fc.json"
    # fc regime set is {e1_fc, e2p_fc} ONLY — matched-e2_fc is structurally
    # dropped (plan v9 restriction: within-context weights cancel exactly on
    # context-level rows).
    assert natpv.regimes_for(wht_fc) == ("e1_fc", "e2p_fc")
    assert natpv.contrast_regimes_for(wht_fc) == (("e2p", True),)
    assert natpv.contrast_regimes_for(argparse.Namespace()) == (("e2", False), ("e2p", True))
    assert natpv.base_regime("e2p_fc") == "e2p"
    assert natpv.base_regime("e1") == "e1"
    # t1 defaults unchanged — including Namespaces predating the flag.
    legacy = argparse.Namespace(space="raw")
    assert natpv.cube_dir_name(legacy) == "cube"
    assert natpv.reduce_out_name(legacy) == "regime_comparison.json"
    assert natpv.regimes_for(legacy) == ("e1", "e2", "e2p")
    assert not natpv.is_fc(legacy)


def test_load_directions_fc_reads_core_leg_bank_and_own_fc_dirs(tmp_path):
    """fc directions: e1_fc from the CORE fits leg's npz bank (--e1-fc-bank),
    e2_fc/e2p_fc from this driver's own r_b_*_fc dirs; fail-loud when the
    bank is absent (names the producer)."""
    rng = np.random.default_rng(0)
    behavior = "sycophancy"
    stage = tmp_path / "stage"
    bank = tmp_path / "bank"
    bank.mkdir(parents=True)
    np.savez(
        bank / f"{behavior}.npz",
        rb=rng.normal(size=(28, 3584)).astype(np.float16),
        layers=np.arange(28),
    )
    # ONLY e2p_fc exists on the fc leg (matched-e2_fc structurally dropped);
    # _load_directions must NOT require an r_b_e2_fc dir.
    for regime in ("e2p_fc",):
        d = stage / behavior / f"r_b_{regime}"
        d.mkdir(parents=True)
        np.savez(
            d / f"{behavior}.npz",
            rb=rng.normal(size=(28, 3584)).astype(np.float16),
            layers=np.arange(28),
            meta=json.dumps({"regime": regime}),
        )
    args = argparse.Namespace(space="whitened", summary_kind="context_end", e1_fc_bank=bank)
    out = natpv._load_directions(behavior, stage, args)
    assert sorted(out) == ["e1_fc", "e2p_fc"]
    for v in out.values():
        assert v.shape == (28, 3584)
    missing = argparse.Namespace(
        space="whitened", summary_kind="context_end", e1_fc_bank=tmp_path / "nope"
    )
    with pytest.raises(FileNotFoundError, match="rb-point context_end"):
        natpv._load_directions(behavior, stage, missing)
