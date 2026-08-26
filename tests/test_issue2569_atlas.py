"""Unit tests for scripts/issue2569_atlas.py — leg-7 atlas fix-round regressions (#2569 r2).

Concern coverage (each test FAILS on the pre-fix file — verified red before the fix landed):

- **h7-demote-branch-not-implemented / leg7-atlas-noise-demotion-missing:** the
  pre-registered H7 demote branch (plan section 7.5 leg 7) is computed by
  ``h7_demote_block`` and written into ``atlas_distances.json`` — noise-dominated,
  not-demoted, and undecidable verdicts each pinned end-to-end through
  ``phase_atlas``, with the distance units (1 - cosine) and the max-of-members
  pair-within reading asserted numerically.
- **leg7-tier2-aligned-cosine-missing:** ``tier2_aligned_operator_cosine`` — the
  activation-Procrustes aligned operator cosine vs the two-sided random-rotation
  null (the ``issue825_map_alignment._procrustes_cosine_null`` construction; read
  against the #825 anchor 0.6864) — fields, identity-alignment sanity, and the
  cross-shape raw-cosine inapplicability statement.
- **atlas-pair-loop-unbatched-unresumable:** ``shared_rotation_null_draws`` — the
  exact Haar-invariance identity (deterministic algebraic test), distribution
  parity vs the serial ``issue1345_operator_comparison.raw_cosine_with_rotation_null``
  convention, chunked checkpoints with content-fingerprint regime keys (reuse on
  identical inputs; regeneration when a producer artifact's CONTENT changes), and
  batched-vs-direct numeric equality of the pair-table statistics.
- **leg7-atlas-writemap-operators-unpersisted:** producer->consumer round trip —
  ``issue2569_leg6.write_operator_factors`` (the landed producer) feeds
  ``_resolve_atlas_rows``'s leg-6 branch; operator reconstruction and the
  bare-float ``split_half_floor`` normalization (``_floor_cos``) asserted.

All synthetic, CPU-fast (d <= 48); no network, no repo-root writes (tmp_path only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_atlas as AT  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_leg6_sidecar(
    arm_dir: Path,
    A: np.ndarray,
    floor: float | None,
    *,
    k: int | None = None,
) -> None:
    """Persist one leg-6 ``operator_factors.pt`` in the PRODUCER's realized schema
    (``issue2569_leg6.write_operator_factors``: u/s/v tensors + a BARE-FLOAT
    ``split_half_floor``; ``A = (u * s) @ v.T``)."""
    d = A.shape[0]
    k = d if k is None else k
    u, s, vh = np.linalg.svd(np.asarray(A, np.float64))
    payload = {
        "u": torch.as_tensor(u[:, :k], dtype=torch.float32),
        "s": torch.as_tensor(s[:k], dtype=torch.float64),
        "v": torch.as_tensor(vh[:k].T, dtype=torch.float32),
        "split_half_floor_class": "fixture",
        "arm": arm_dir.name,
        "layer": 19,
        "k": int(k),
        "regime_key": "fixture",
    }
    if floor is not None:
        payload["split_half_floor"] = float(floor)
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, arm_dir / "operator_factors.pt")


def _atlas_args(tmp: Path, leg6: Path, *, null_draws: int = 8, chunk: int = 4):
    """Parsed args for a leg6-only ``phase_atlas`` run (every other row drops)."""
    return AT._parse_args(
        [
            "--phase",
            "atlas",
            "--leg6-dir",
            str(leg6),
            "--fits-dir",
            str(tmp / "fits"),
            "--leg7-dir",
            str(tmp / "leg7"),
            "--map-root",
            str(tmp / "no-banked-maps"),
            "--skip-passb",
            "--device",
            "cpu",
            "--null-draws",
            str(null_draws),
            "--null-chunk-draws",
            str(chunk),
            "--val-rows",
            "8",
        ]
    )


def _run_atlas(tmp: Path, leg6: Path, **kw) -> dict:
    AT.phase_atlas(_atlas_args(tmp, leg6, **kw))
    return json.loads((tmp / "leg7" / "atlas_distances.json").read_text())


# ---------------------------------------------------------------------------
# Concern: atlas-pair-loop-unbatched-unresumable — exact batched null
# ---------------------------------------------------------------------------


def test_shared_null_algebraic_identity():
    """The Haar-invariance reduction is an ALGEBRAIC identity, not an approximation:
    for FIXED orthogonal Q1, Q2 and full SVDs A = Ua Sa Va^T, B = Ub Sb Vb^T,
    cos(vec(A), vec(Q1^T B Q2)) == sa^T (G1 * G2^T) sb / (||A|| ||B||)
    with G1 = Ua^T Q1^T Ub, G2 = Vb^T Q2 Va."""
    rng = np.random.default_rng(0)
    d = 8
    A = rng.standard_normal((d, d))
    B = rng.standard_normal((d, d))
    gen = torch.Generator().manual_seed(7)
    q1 = np.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64).numpy())[0]
    q2 = np.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64).numpy())[0]
    lhs_m = q1.T @ B @ q2
    lhs = float((A.reshape(-1) @ lhs_m.reshape(-1)) / (np.linalg.norm(A) * np.linalg.norm(lhs_m)))
    ua, sa, vat = np.linalg.svd(A)
    ub, sb, vbt = np.linalg.svd(B)
    g1 = ua.T @ q1.T @ ub
    g2 = vbt @ q2 @ vat.T
    rhs = float(sa @ (g1 * g2.T) @ sb / (np.linalg.norm(A) * np.linalg.norm(B)))
    assert abs(lhs - rhs) < 1e-10, (lhs, rhs)


def test_shared_null_matches_convention_distribution():
    """The batched draws come from the SAME distribution as the serial
    ``issue1345_operator_comparison.raw_cosine_with_rotation_null`` convention:
    null mean ~ 0 and null std ~ the exact analytic 1/d (sd = 1/sqrt(d_in*d_out)),
    and the serial implementation's empirical std agrees within MC tolerance."""
    import issue1345_operator_comparison as OC

    rng = np.random.default_rng(1)
    d, n_draws = 48, 400
    A = rng.standard_normal((d, d))
    B = rng.standard_normal((d, d))
    spectra = []
    for m in (A, B):
        s = np.linalg.svd(m, compute_uv=False)
        spectra.append(s / np.linalg.norm(s))
    x = AT.shared_rotation_null_draws(
        np.stack(spectra), n_draws=n_draws, seed=11, device="cpu", chunk_draws=100
    )
    assert x.shape == (n_draws, 2, 2)
    draws = x[:, 0, 1]
    assert abs(float(draws.mean())) < 5e-3
    assert abs(float(draws.std()) - 1.0 / d) < 5e-3  # exact analytic sd = 1/sqrt(d*d)
    serial = OC.raw_cosine_with_rotation_null(
        torch.as_tensor(A, dtype=torch.float64),
        torch.as_tensor(B, dtype=torch.float64),
        n_draws=n_draws,
        seed=12,
    )
    assert abs(float(draws.std()) - serial["rotation_null"]["null_std"]) < 5e-3


def test_pair_table_matches_direct_statistics(tmp_path):
    """Batching must not change the pair statistics: spectrum cosine and the raw
    aligned cosine in the table equal the direct per-pair computations."""
    rng = np.random.default_rng(2)
    d = 16
    leg6 = tmp_path / "leg6"
    mats = {}
    for arm in ("armA", "armB", "armC"):
        mats[arm] = rng.standard_normal((d, d))
        _write_leg6_sidecar(leg6 / arm, mats[arm], 0.999)
    out = _run_atlas(tmp_path, leg6)
    assert len(out["distance_table"]) == 3
    for entry in out["distance_table"]:
        a, b = entry["pair"]
        A = mats[a.removeprefix("leg6_wmap_")]
        B = mats[b.removeprefix("leg6_wmap_")]
        direct_spec = AT.spectrum_cosine(A, B)["spectrum_cosine"]
        assert abs(entry["spectrum"]["spectrum_cosine"] - direct_spec) < 1e-6
        va, vb = A.reshape(-1), B.reshape(-1)
        direct_raw = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        assert abs(entry["cosine"]["raw_cosine"] - direct_raw) < 1e-6
        nl = entry["cosine"]["rotation_null"]
        assert nl["n_draws"] == 8
        for key in ("null_mean", "null_std", "null_p975", "analytic_sd_1_over_d"):
            assert np.isfinite(nl[key]), (key, nl)


def test_atlas_checkpoints_resume_and_invalidate(tmp_path):
    """Mid-loop death recovery: per-row spectra + per-chunk null draws checkpoint
    under the fits dir; a re-run REUSES them (mtimes unchanged, output identical);
    a CONTENT change in one producer artifact regenerates that row's spectra and
    the null chunks while the untouched rows' checkpoints survive (the resume key
    is a content fingerprint, never a status string)."""
    rng = np.random.default_rng(3)
    d = 16
    leg6 = tmp_path / "leg6"
    for arm in ("armA", "armB", "armC"):
        _write_leg6_sidecar(leg6 / arm, rng.standard_normal((d, d)), 0.99)
    out1 = _run_atlas(tmp_path, leg6, null_draws=8, chunk=4)
    ckpt = tmp_path / "fits" / "atlas_ckpt"
    spectra_files = sorted(ckpt.glob("spectra_*.pt"))
    chunk_files = sorted(ckpt.glob("null_chunk_*.pt"))
    assert len(spectra_files) == 3, spectra_files
    assert len(chunk_files) == 2, chunk_files  # 8 draws / chunk 4
    mtimes1 = {p.name: p.stat().st_mtime_ns for p in spectra_files + chunk_files}

    out2 = _run_atlas(tmp_path, leg6, null_draws=8, chunk=4)
    mtimes2 = {p.name: p.stat().st_mtime_ns for p in sorted(ckpt.glob("*.pt"))}
    assert mtimes2 == mtimes1, "second run must reuse every checkpoint"
    assert out2["distance_table"] == out1["distance_table"]
    assert out2["h7_demote"] == out1["h7_demote"]

    # content change in ONE producer artifact -> that row + the null regenerate
    _write_leg6_sidecar(leg6 / "armB", rng.standard_normal((d, d)), 0.99)
    out3 = _run_atlas(tmp_path, leg6, null_draws=8, chunk=4)
    mtimes3 = {p.name: p.stat().st_mtime_ns for p in sorted(ckpt.glob("*.pt"))}
    changed = {n for n in mtimes3 if mtimes3[n] != mtimes1.get(n)}
    armb = [p.name for p in ckpt.glob("spectra_*armB*.pt")]
    assert armb and set(armb) <= changed, (armb, changed)
    assert {p.name for p in ckpt.glob("null_chunk_*.pt")} <= changed
    untouched = {p.name for p in ckpt.glob("spectra_*.pt")} - set(armb)
    assert untouched and not (untouched & changed), (untouched, changed)
    assert out3["distance_table"] != out1["distance_table"]


# ---------------------------------------------------------------------------
# Concern: h7-demote-branch-not-implemented / leg7-atlas-noise-demotion-missing
# ---------------------------------------------------------------------------


def test_h7_demote_fires_on_noise_dominated_atlas(tmp_path):
    """Identical operators (between-distance ~ 0) with floors 0.9 (within-distance
    0.1) -> every evaluable pair noise-dominated -> the pre-registered demote fires
    and the disposition is decidable from the artifact alone."""
    rng = np.random.default_rng(4)
    d = 16
    A = rng.standard_normal((d, d))
    leg6 = tmp_path / "leg6"
    for arm in ("armA", "armB", "armC"):
        _write_leg6_sidecar(leg6 / arm, A, 0.9)
    out = _run_atlas(tmp_path, leg6)
    h7 = out["h7_demote"]
    assert h7["n_evaluable"] == 3 and h7["n_noise_dominated"] == 3, h7
    assert h7["noise_dominated"] is True
    assert "noise-dominated" in h7["disposition"] and "descriptive only" in h7["disposition"]
    assert h7["predicate"].startswith("within-operator split-half distance")
    for entry in out["distance_table"]:
        rec = entry["h7"]
        assert rec["evaluable"] and rec["noise_dominated"]
        # units: distances are 1 - cosine, within = max over the two members
        assert abs(rec["within_distance_max"] - (1.0 - 0.9)) < 1e-9
        assert abs(rec["between_distance"] - (1.0 - entry["cosine"]["raw_cosine"])) < 1e-12
    # rows expose the derived distance so no downstream reader re-derives units
    for row in out["rows"]:
        assert abs(row["within_distance"] - 0.1) < 1e-9
        assert abs(row["floor_cos"] - 0.9) < 1e-9


def test_h7_not_demoted_on_resolved_atlas(tmp_path):
    """Distinct random operators (between-distance ~ 1) with tight floors (0.999)
    -> zero noise-dominated pairs -> NOT demoted (strict > half predicate)."""
    rng = np.random.default_rng(5)
    d = 16
    leg6 = tmp_path / "leg6"
    for arm in ("armA", "armB", "armC"):
        _write_leg6_sidecar(leg6 / arm, rng.standard_normal((d, d)), 0.999)
    out = _run_atlas(tmp_path, leg6)
    h7 = out["h7_demote"]
    assert h7["n_evaluable"] == 3 and h7["n_noise_dominated"] == 0, h7
    assert h7["noise_dominated"] is False
    assert h7["disposition"].startswith("not demoted")


def test_h7_undecidable_without_floors(tmp_path):
    """No floors anywhere -> zero evaluable pairs -> the verdict is UNDECIDABLE and
    the atlas ships descriptive-only; never a silent not-demoted default."""
    rng = np.random.default_rng(6)
    d = 16
    leg6 = tmp_path / "leg6"
    for arm in ("armA", "armB"):
        _write_leg6_sidecar(leg6 / arm, rng.standard_normal((d, d)), None)
    out = _run_atlas(tmp_path, leg6)
    h7 = out["h7_demote"]
    assert h7["n_evaluable"] == 0 and h7["noise_dominated"] is None, h7
    assert "undecidable" in h7["disposition"] and "descriptive only" in h7["disposition"]
    reasons = list(h7["excluded_pair_reasons"])
    assert reasons and "no split-half floor" in reasons[0], reasons


def test_h7_excludes_spectrum_fallback_pairs():
    """A pair without a direction-aware cosine (spectrum fallback only) is EXCLUDED
    from the H7 denominator with a recorded reason — rotation-invariant units are
    never mixed with the vec-cosine floors."""
    rows = [
        {"name": "a", "floor": {"floor": 0.9}},
        {"name": "b", "floor": 0.95},  # leg6 bare-float shape
        {"name": "c", "floor": None},
    ]
    table = [
        {"pair": ["a", "b"], "cosine": {"raw_cosine": 0.5}},
        {"pair": ["a", "c"], "cosine": {"raw_cosine": 0.5}},
        {"pair": ["b", "c"], "cosine": None},
    ]
    h7 = AT.h7_demote_block(rows, table)
    assert h7["n_pairs_total"] == 3 and h7["n_evaluable"] == 1
    assert table[0]["h7"]["evaluable"]
    # within = max(1-0.9, 1-0.95) = 0.1; between = 0.5 -> not noise dominated
    assert abs(table[0]["h7"]["within_distance_max"] - 0.1) < 1e-12
    assert table[0]["h7"]["noise_dominated"] is False
    assert not table[1]["h7"]["evaluable"] and "no split-half floor" in table[1]["h7"]["reason"]
    assert not table[2]["h7"]["evaluable"] and "direction-aware" in table[2]["h7"]["reason"]


# ---------------------------------------------------------------------------
# Concern: leg7-tier2-aligned-cosine-missing
# ---------------------------------------------------------------------------


def test_tier2_aligned_cosine_identity_alignment():
    """Under the identity alignment, the aligned cosine of an operator with itself
    is 1.0 and sits far above the rotation null; all convention fields present."""
    rng = np.random.default_rng(7)
    d = 12
    A = rng.standard_normal((d, d))
    eye = np.eye(d)
    block = AT.tier2_aligned_operator_cosine(
        {"a_qwen": A, "qwen_matched": rng.standard_normal((d, d))},
        A,
        eye,
        eye,
        n_draws=64,
        seed=13,
        device="cpu",
        n_rows_alignment=100,
    )
    assert block["anchor_825_aligned_cosine"] == 0.6864
    assert block["n_rows_alignment"] == 100
    assert "direction-aware" in block["statistic_class"]
    assert "Procrustes" in block["alignment"]
    rec = block["per_operator"]["a_qwen"]
    assert abs(rec["observed_aligned_cosine"] - 1.0) < 1e-9
    assert rec["observed_aligned_cosine"] > rec["rotation_null"]["null_p975"]
    assert rec["z_observed_vs_null"] > 3.0
    # same shape here -> the raw (pre-alignment) vec cosine is applicable
    assert rec["raw_vec_cosine"]["applicable"] is True


def test_tier2_aligned_cosine_cross_shape_raw_inapplicable():
    """Cross-model shapes (d_q != d_l): the raw pre-alignment vec cosine is stated
    INAPPLICABLE (never silently skipped); the aligned statistic still computes in
    the shared qwen basis."""
    rng = np.random.default_rng(8)
    d_q, d_l = 12, 20
    A_q = rng.standard_normal((d_q, d_q))
    A_l = rng.standard_normal((d_l, d_l))
    r_in = np.linalg.qr(rng.standard_normal((d_l, d_q)))[0].T  # (d_q, d_l), rows orthonormal
    r_out = np.linalg.qr(rng.standard_normal((d_l, d_q)))[0].T
    block = AT.tier2_aligned_operator_cosine(
        {"a_qwen": A_q},
        A_l,
        r_in,
        r_out,
        n_draws=16,
        seed=14,
        device="cpu",
        n_rows_alignment=50,
    )
    rec = block["per_operator"]["a_qwen"]
    assert rec["raw_vec_cosine"]["applicable"] is False
    assert "shape" in rec["raw_vec_cosine"]["reason"]
    assert -1.0 <= rec["observed_aligned_cosine"] <= 1.0
    assert np.isfinite(rec["rotation_null"]["null_std"])


# ---------------------------------------------------------------------------
# Concern: leg7-atlas-writemap-operators-unpersisted — producer round trip
# ---------------------------------------------------------------------------


def test_leg6_producer_consumer_roundtrip(tmp_path):
    """The LANDED leg-6 producer (``issue2569_leg6.write_operator_factors``) feeds
    the atlas consumer branch: the reconstructed row operator matches the
    producer's all-rows ridge refit, and the bare-float floor normalizes through
    ``_floor_cos`` into the H7 branch."""
    import issue2569_leg6 as L6

    rng = np.random.default_rng(9)
    n, d = 200, 16
    c_mat = torch.as_tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    g = torch.as_tensor(rng.standard_normal((d, d)), dtype=torch.float32) / np.sqrt(d)
    delta = c_mat @ g + 0.05 * torch.as_tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    unit_rec = {
        "lambda_rel_selected": 0.1,
        "operator_splithalf_cosine": 0.87,
        "factor_bases": {"u": "context residual", "v": "shift residual"},
    }
    arm_dir = tmp_path / "leg6" / "armRT"
    arm_dir.mkdir(parents=True)
    L6.write_operator_factors(
        arm_dir,
        c_mat,
        delta,
        unit_rec,
        arm="armRT",
        layer=19,
        convention="fixture",
        rk="rt-regime",
        resume=False,
    )
    args = _atlas_args(tmp_path, tmp_path / "leg6")
    rows, _dropped = AT._resolve_atlas_rows(args)
    assert [r["name"] for r in rows] == ["leg6_wmap_armRT"]
    row = rows[0]
    # expected operator: the producer's own all-rows ridge refit
    all_idx = np.arange(n, dtype=np.int64)
    _mu_c, _mu_d, scc, scd, _n = L6.half_moments(c_mat, delta, all_idx)
    m_all = L6.ridge_map(scc, scd, 0.1 * float(torch.diagonal(scc).mean()))
    assert np.allclose(row["A"], m_all.numpy(), atol=5e-4), np.abs(row["A"] - m_all.numpy()).max()
    assert AT._floor_cos(row["floor"]) == 0.87  # bare float, producer schema
    assert row["fp"], "content fingerprint missing on the leg6 row"
