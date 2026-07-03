"""Serial-vs-batched equivalence for the #813 substrate-swap null battery (B3).

The batched Gram/dual-space engine (`_batched_arm_dofs` + the batched branches of
``substrate_swap_null`` / ``pairwise_diff_ci``) REPLACES the serial per-fit battery
as the default (Supersede contract, `.claude/rules/vectorize-many-cell-fits.md`).
These tests pin:

1. NUMERICAL EQUIVALENCE against the retained serial reference on a small synthetic
   per-question npz — per-resample (the full ``null_delta_over_floor_diffs`` array),
   not just the summary percentiles, for BOTH the r̂-projection (em/fact/syco) and
   the ‖·‖ marker paths. TARGET_DIM is clamped to 2 here so every per-fit PCA keeps
   k ≤ numerical rank — the ONE deliberate batched/serial divergence is the serial
   SVD's arbitrary null-space tail rows (rank < k), which the batched engine
   truncates; with k ≤ rank both paths compute the same basis and must agree to
   float tolerance.
2. Chunk-size invariance (arm_chunk / pair_chunk are memory dials, never
   output-affecting — the vectorize-rule chunk-invariance pin).
3. The artifact contract: identical key set serial vs batched (+ the additive
   ``null_impl`` provenance key), the degenerate n_q<4 note path, regime stamps.
4. The Supersede-contract tombstone: FutureWarning on the serial path and a
   RuntimeError under EPM_FORBID_SERIAL_FITS=1.

Everything runs on CPU with HIDDEN=32-dim synthetic activations in seconds.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue722_fit_M as fitM  # noqa: E402
import issue813_analysis as an  # noqa: E402

H = 32
N_CTX = 8
N_Q = 8


@pytest.fixture()
def small_dims(monkeypatch):
    """Clamp BOTH PCA target dims to 2 so k <= rank on every resample (tail-free).

    The serial `_pseudo_delta_over_floor` reads issue813_analysis.TARGET_DIM for the
    observed basis and issue722_fit_M.TARGET_DIM inside `_refit_ridge_fn` for the
    floor refits; the batched engine mirrors both, so the monkeypatch hits the two
    module globals. Also keeps the serial ridge on CPU and clears the tombstone env.
    """
    monkeypatch.setattr(an, "TARGET_DIM", 2)
    monkeypatch.setattr(fitM, "TARGET_DIM", 2)
    monkeypatch.setattr(an.fit658, "DEVICE", "cpu")
    monkeypatch.delenv("EPM_FORBID_SERIAL_FITS", raising=False)


def _write_pq_npz(root: Path, *, n_ctx=N_CTX, n_q=N_Q, h=H, seed=7) -> Path:
    """A dense synthetic per_question_L14.npz (n_ctx x n_q grid, singleton families)."""
    rng = np.random.default_rng(seed)
    rows = n_ctx * n_q
    cell = root / "em" / "generic"
    cell.mkdir(parents=True, exist_ok=True)
    np.savez(
        cell / f"per_question_L{an.HEADLINE_LAYER}.npz",
        c_C_base=rng.standard_normal((rows, h)),
        c_C_trained=rng.standard_normal((rows, h)),
        v_A_base=rng.standard_normal((rows, h)),
        v_A_trained=rng.standard_normal((rows, h)),
        row_context_index=np.repeat(np.arange(n_ctx), n_q),
        row_question_index=np.tile(np.arange(n_q), n_ctx),
        families=np.array([f"fam{i}" for i in range(n_ctx)]),
    )
    return root


def _r_hat(seed=3, h=H) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(h)
    return v / np.linalg.norm(v)


_CMP_KEYS = (
    "null_p95",
    "null_p975",
    "null_median",
    "null_over_floor_p95",
    "null_over_floor_p975",
    "null_over_floor_median",
)


def _run_pair(root, r_hat, **kw):
    """(serial, batched) substrate_swap_null results on the same fixture + seed."""
    with pytest.warns(FutureWarning):
        s = an.substrate_swap_null("em", "generic", root, r_hat, 10, serial=True, **kw)
    b = an.substrate_swap_null("em", "generic", root, r_hat, 10, device="cpu", **kw)
    return s, b


def test_null_battery_batched_matches_serial_rhat(tmp_path, small_dims):
    """r̂-projection (em/fact/syco) path: per-resample arrays + bands match serial."""
    root = _write_pq_npz(tmp_path)
    s, b = _run_pair(root, _r_hat(), n_refit_pairs=3)
    assert s["null_impl"] == "serial" and b["null_impl"] == "batched_gram_v1"
    assert b["n_resamples_used"] == s["n_resamples_used"] > 0
    assert b["n_over_floor_resamples_used"] == s["n_over_floor_resamples_used"] > 0
    np.testing.assert_allclose(
        b["null_delta_over_floor_diffs"],
        s["null_delta_over_floor_diffs"],
        rtol=1e-6,
        atol=1e-9,
    )
    for k in _CMP_KEYS:
        np.testing.assert_allclose(b[k], s[k], rtol=1e-6, atol=1e-9)


def test_null_battery_batched_matches_serial_marker_norm(tmp_path, small_dims):
    """marker read-1 (r̂=None, ‖·‖ + cross-basis Grams) path matches serial too."""
    root = _write_pq_npz(tmp_path, seed=11)
    s, b = _run_pair(root, None, n_refit_pairs=3)
    assert b["n_resamples_used"] == s["n_resamples_used"] > 0
    np.testing.assert_allclose(
        b["null_delta_over_floor_diffs"],
        s["null_delta_over_floor_diffs"],
        rtol=1e-6,
        atol=1e-9,
    )
    for k in _CMP_KEYS:
        np.testing.assert_allclose(b[k], s[k], rtol=1e-6, atol=1e-9)


def test_engine_matches_serial_single_arm(small_dims):
    """`_batched_arm_dofs` == `_pseudo_delta_over_floor` on one arm with UNEVEN families.

    Two families of sizes 3 + 5 make the clustered floor resamples VARIABLE-length
    (6/8/10 rows), exercising the engine's row padding; distinct rows are always
    >= 3 so rank >= TARGET_DIM=2 (tail-free). This is the exact compute path the
    batched pairwise CI uses per (resample, side) arm.
    """
    rng = np.random.default_rng(23)
    m = 8
    arm = {
        "c0": rng.standard_normal((m, H)),
        "cplus": rng.standard_normal((m, H)),
        "v0": rng.standard_normal((m, H)),
        "vplus": rng.standard_normal((m, H)),
        "fams": ["famA"] * 3 + ["famB"] * 5,
    }
    for r_hat in (_r_hat(5), None):
        want = an._pseudo_delta_over_floor(
            arm["c0"], arm["cplus"], arm["v0"], arm["vplus"], arm["fams"], r_hat, n_refit_pairs=4
        )
        (got,) = an._batched_arm_dofs([arm], r_hat, n_refit_pairs=4, device="cpu")
        np.testing.assert_allclose(got[0], want[0], rtol=1e-8, atol=1e-11)
        np.testing.assert_allclose(got[1], want[1], rtol=1e-6, atol=1e-9)


def test_pairwise_diff_ci_batched_matches_serial(monkeypatch, small_dims):
    """Batched pairwise CI == serial reference (same draws, same skip semantics)."""
    rng = np.random.default_rng(31)
    n = 8
    ctx_ids = [f"ctx{i}" for i in range(n)]
    fams = ["famA"] * 3 + ["famB"] * 5

    def _stacks(behavior, substrate, reduced_root):
        r = np.random.default_rng(hash(substrate) % (2**31))
        return (
            r.standard_normal((n, H)),
            r.standard_normal((n, H)),
            r.standard_normal((n, H)),
            r.standard_normal((n, H)),
            list(fams),
            list(ctx_ids),
        )

    monkeypatch.setattr(an, "_headline_stacks", _stacks)
    r_hat = _r_hat(9)
    kw = dict(n_resamples=10, n_refit_pairs=2)
    with pytest.warns(FutureWarning):
        s = an.pairwise_diff_ci("em", "generic", "elicit", Path("."), r_hat, serial=True, **kw)
    b = an.pairwise_diff_ci("em", "generic", "elicit", Path("."), r_hat, device="cpu", **kw)
    assert b["n_resamples_used"] == s["n_resamples_used"] > 0
    for k in ("ci_lo", "ci_hi", "ci_median"):
        np.testing.assert_allclose(b[k], s[k], rtol=1e-6, atol=1e-9)
    assert b["ci_excludes_zero"] == s["ci_excludes_zero"]
    assert set(b.keys()) == set(s.keys())
    _ = rng  # (kept for symmetry with the other fixtures)


def test_chunk_sizes_do_not_change_results(tmp_path, small_dims):
    """arm_chunk / pair_chunk are memory dials only — outputs identical across sizes."""
    root = _write_pq_npz(tmp_path, seed=17)
    r_hat = _r_hat()
    a = an.substrate_swap_null(
        "em", "generic", root, r_hat, 6, n_refit_pairs=3, device="cpu", arm_chunk=1, pair_chunk=1
    )
    b = an.substrate_swap_null(
        "em", "generic", root, r_hat, 6, n_refit_pairs=3, device="cpu", arm_chunk=8, pair_chunk=3
    )
    np.testing.assert_allclose(
        a["null_delta_over_floor_diffs"], b["null_delta_over_floor_diffs"], rtol=1e-12
    )
    np.testing.assert_allclose(a["null_over_floor_p95"], b["null_over_floor_p95"], rtol=1e-12)


def test_schema_matches_serial_and_degenerate_note(tmp_path, small_dims):
    """Same key set serial vs batched; the n_q<4 degenerate path is impl-independent."""
    root = _write_pq_npz(tmp_path)
    s, b = _run_pair(root, _r_hat(), n_refit_pairs=2)
    assert set(b.keys()) == set(s.keys())
    for k in ("n_refit_pairs", "n_null_resamples_requested", "null_space", "n_questions"):
        assert b[k] == s[k]
    # degenerate: too few questions (<4) — identical note dict on both paths (no
    # battery runs, so no tombstone warning fires either).
    tiny = _write_pq_npz(tmp_path / "tiny", n_q=3)
    for serial in (False, True):
        d = an.substrate_swap_null(
            "em", "generic", tiny, _r_hat(), 5, n_refit_pairs=2, serial=serial, device="cpu"
        )
        assert d["note"].startswith("too few questions")
        assert d["n_resamples_used"] == 0
        assert d["n_refit_pairs"] == 2
        assert d["n_null_resamples_requested"] == 5


def test_serial_tombstone_warns_and_env_forbids(tmp_path, small_dims, monkeypatch):
    """Supersede contract: serial path warns (FutureWarning) and refuses under
    EPM_FORBID_SERIAL_FITS=1 (`.claude/rules/vectorize-many-cell-fits.md`)."""
    root = _write_pq_npz(tmp_path)
    with pytest.warns(FutureWarning, match="SUPERSEDED serial battery"):
        an.substrate_swap_null("em", "generic", root, _r_hat(), 2, n_refit_pairs=2, serial=True)
    monkeypatch.setenv("EPM_FORBID_SERIAL_FITS", "1")
    with pytest.raises(RuntimeError, match="EPM_FORBID_SERIAL_FITS"):
        an.substrate_swap_null("em", "generic", root, _r_hat(), 2, n_refit_pairs=2, serial=True)
    # the batched default is unaffected by the env gate
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = an.substrate_swap_null(
            "em", "generic", root, _r_hat(), 2, n_refit_pairs=2, device="cpu"
        )
    assert out["null_impl"] == "batched_gram_v1"
