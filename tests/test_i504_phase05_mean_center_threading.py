# ruff: noqa: RUF002  # em-dash intentional in #504 module docstrings
"""Round-7 regression tests for the Phase 0.5 mean-center threading (task #504).

Pins both binding blockers the Codex code-reviewer + reconciler flagged at
round 4 of the round-6 mean-centering pivot:

* **Blocker 1 — ``mean_center`` threading.** ``--no-mean-center`` must
  disable mean-centering UNIFORMLY across (a) the dispatcher-built
  ``cos_to_source_by_layer`` (already verified in round 6), (b) the
  ``_cos_matrix_from_centroids`` calls inside ``_select_at_layer`` and
  ``run_gates_for_layer``, and (c) the chosen-layer re-pick. Round 6
  only threaded into ``_cos_to_source`` at the dispatcher; the gate +
  per-probe paths silently kept ``mean_center=True``, yielding an
  internally inconsistent raw-cosine artifact.

* **Blocker 2 — #472 replay isolation.** Unqualified
  ``cos_to_source(layer, source, dir)`` and
  ``load_cos_matrix(layer, dir)`` calls (used by ``i472_run_cell.py``,
  ``i472_eval_trajectory.py``, ``i477_reval_confirm.py``, etc.) must
  read the RAW ``cos_matrix`` field, not the round-6
  ``cos_matrix_mean_centered`` field. Round 6 flipped the defaults to
  ``"global_mean"``, silently shifting #472's published numbers on
  rerun. Round 7 reverts the defaults to ``"none"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    cos_to_source,
    load_cos_matrix,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504.phase05 import (
    _cos_matrix_from_centroids,
    _per_probe_covariates,
    run_phase05,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic centroid + cos-to-source fixtures (small enough to be fast).
# ─────────────────────────────────────────────────────────────────────────────


def _make_synthetic_centroids(
    seed: int = 0,
    n_personas: int = 12,
    dim: int = 32,
    shared_mean_norm: float = 5.0,
) -> dict[str, np.ndarray]:
    """Synthetic 60-ish-bank stand-in with a LARGE shared component.

    The shared component is what makes raw cosines saturate (round 1-5 #504
    behavior) — it lifts every pairwise raw cosine into [0.92, 0.99] until
    mean-centering removes it. Without shared_mean_norm > a few, the test
    can't distinguish raw from mean-centered.
    """
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(dim).astype(np.float64)
    shared = shared / np.linalg.norm(shared) * shared_mean_norm
    out: dict[str, np.ndarray] = {}
    # Use names that exercise the dispatcher's expected source + default.
    base_names = ["villain", "qwen_default"] + [f"persona_{i:02d}" for i in range(n_personas - 2)]
    for name in base_names:
        v = rng.standard_normal(dim).astype(np.float64) + shared
        out[name] = v.astype(np.float32)
    return out


def _cos_to_source_dict(
    centroids: dict[str, np.ndarray],
    source: str,
    *,
    mean_center: bool,
) -> dict[str, float]:
    """Reproduce ``scripts/i504_phase_phase05.py::_cos_to_source`` semantics
    so the synthetic fixture matches what the dispatcher passes into
    ``run_phase05``."""
    names = list(centroids.keys())
    mat = np.stack([centroids[n].astype(np.float64) for n in names], axis=0)
    if mean_center:
        mat = mat - mat.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(mat, axis=1)
    src_idx = names.index(source)
    src = mat[src_idx]
    src_norm = float(norms[src_idx])
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        nv = float(norms[i])
        if nv == 0.0:
            out[name] = 0.0
            continue
        out[name] = float(np.dot(mat[i], src) / (nv * src_norm))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Blocker 1 — mean_center threading in phase05.
# ─────────────────────────────────────────────────────────────────────────────


def test_cos_matrix_from_centroids_respects_mean_center_flag() -> None:
    """``_cos_matrix_from_centroids`` MUST differ between mean_center=True/False.

    The shared-mean synthetic bank has saturated raw cosines (~[0.92, 0.99])
    and a much wider mean-centered span — a direct sanity check that the flag
    actually controls the geometry the gates and per-probe covariates see.
    """
    centroids = _make_synthetic_centroids(seed=0)
    mc_true = _cos_matrix_from_centroids(centroids, mean_center=True)
    mc_false = _cos_matrix_from_centroids(centroids, mean_center=False)

    # Same shape, different content.
    assert set(mc_true.keys()) == set(mc_false.keys())
    a, b = "villain", "persona_00"
    assert mc_true[a][b] != mc_false[a][b], (mc_true[a][b], mc_false[a][b])

    # Raw cosines should be near-saturated (>0.85) given the shared component;
    # mean-centered span should be measurably wider.
    raw_vals = [mc_false["villain"][p] for p in mc_false if p != "villain"]
    mc_vals = [mc_true["villain"][p] for p in mc_true if p != "villain"]
    raw_span = max(raw_vals) - min(raw_vals)
    mc_span = max(mc_vals) - min(mc_vals)
    assert mc_span > raw_span, (
        f"mean-centered span {mc_span:.3f} should exceed raw span {raw_span:.3f}; "
        "if not, the synthetic fixture's shared component is too small."
    )


def test_per_probe_covariates_d_nn_differs_under_mean_center() -> None:
    """``d_nearest_neg_nd`` (1 − cos[probe][N]) MUST flip between centerings.

    This is the covariate the chosen-layer re-pick exposes to Phase 2's
    partial-Spearman regression — if the threading is broken it carries
    mean-centered values while ``d_source`` carries raw values, contaminating
    every downstream stat.
    """
    centroids = _make_synthetic_centroids(seed=1)
    source = "villain"
    arm_to_n = {
        "c504_near": "persona_00",
        "c504_mid_near": "persona_01",
        "c504_mid_far": "persona_02",
        "c504_far": "persona_03",
    }
    panel = ["persona_04", "persona_05", "persona_06"]

    cts_raw = _cos_to_source_dict(centroids, source, mean_center=False)
    cts_mc = _cos_to_source_dict(centroids, source, mean_center=True)

    cov_raw = _per_probe_covariates(
        panel,
        arm_to_n,
        cts_raw,
        _cos_matrix_from_centroids(centroids, mean_center=False),
        centroids,
        source,
    )
    cov_mc = _per_probe_covariates(
        panel,
        arm_to_n,
        cts_mc,
        _cos_matrix_from_centroids(centroids, mean_center=True),
        centroids,
        source,
    )

    # d_source differs between centerings (covariate, not invariant).
    assert cov_raw["persona_04"]["d_source"] != cov_mc["persona_04"]["d_source"]
    # d_nearest_neg_nd[arm] differs between centerings (the broken thread).
    assert (
        cov_raw["persona_04"]["d_nearest_neg_nd"]["c504_near"]
        != cov_mc["persona_04"]["d_nearest_neg_nd"]["c504_near"]
    )
    # shadow_angle is mean-center-INVARIANT by construction (operates on
    # n - source, probe - source — the additive shift cancels). Pin that.
    sa_raw = cov_raw["persona_04"]["shadow_angle"]["c504_near"]
    sa_mc = cov_mc["persona_04"]["shadow_angle"]["c504_near"]
    if not (np.isnan(sa_raw) and np.isnan(sa_mc)):
        assert sa_raw == pytest.approx(sa_mc, rel=1e-9, abs=1e-12), (sa_raw, sa_mc)


def test_run_phase05_mean_center_threading_changes_outputs() -> None:
    """End-to-end: ``run_phase05(mean_center=False)`` vs ``mean_center=True``
    produce measurably different ``per_probe`` covariates AND ``gate_results``.

    This is the exact path the dispatcher follows when a user passes
    ``--no-mean-center``: the centroids + cos_to_source_by_layer come in
    BOTH centered consistently, and ``run_phase05`` must use the matching
    centering when computing the cos matrix for gates + per-probe.

    With the round-6-only thread, both runs would produce IDENTICAL gate
    diagnostics and per-probe d_nn fields (both mean-centered) — only
    d_source would differ. After round-7 threading, ALL three sets differ.
    """
    centroids = _make_synthetic_centroids(seed=2)
    centroids_by_layer = {10: centroids, 15: centroids, 20: centroids}
    source = "villain"
    default_persona = "qwen_default"

    # Minimal r_train_villain (just enough for max_response_token_check).
    r_train_villain = {
        source: {
            "q0": {"response_token_ids": [1] * 200, "response_text": "x" * 200},
            "q1": {"response_token_ids": [1] * 300, "response_text": "x" * 300},
        }
    }

    cts_raw = {
        lay: _cos_to_source_dict(centroids, source, mean_center=False) for lay in (10, 15, 20)
    }
    cts_mc = {lay: _cos_to_source_dict(centroids, source, mean_center=True) for lay in (10, 15, 20)}

    rep_raw = run_phase05(
        centroids_by_layer=centroids_by_layer,
        cos_to_source_by_layer=cts_raw,
        r_train_villain=r_train_villain,
        source=source,
        default_persona=default_persona,
        headline_layer=10,
        fallback_layers=(15, 20),
        mean_center=False,
    )
    rep_mc = run_phase05(
        centroids_by_layer=centroids_by_layer,
        cos_to_source_by_layer=cts_mc,
        r_train_villain=r_train_villain,
        source=source,
        default_persona=default_persona,
        headline_layer=10,
        fallback_layers=(15, 20),
        mean_center=True,
    )

    # The two runs must differ somewhere observable. Pin the strongest
    # signal: at least one per-probe d_nearest_neg_nd value differs at L10.
    panel_raw = rep_raw["held_out_panel"]
    panel_mc = rep_mc["held_out_panel"]
    # Same exclusion pattern → panels should agree (the source / default /
    # positioned-N's depend on the cos_to_source ranking, which can re-order
    # between centerings — accept either same-panel or different-panel).
    common = set(panel_raw) & set(panel_mc)
    if not common:
        pytest.skip("Synthetic fixture caused disjoint panels — bump n_personas.")
    a_probe = next(iter(common))
    dnn_raw = rep_raw["per_probe"][a_probe]["d_nearest_neg_nd"]
    dnn_mc = rep_mc["per_probe"][a_probe]["d_nearest_neg_nd"]
    # At least one arm's d_nn differs (the round-6-broken thread would have
    # made these identical because both reads went through mean_center=True).
    differs = [arm for arm in dnn_raw if dnn_raw[arm] != dnn_mc[arm]]
    assert differs, (
        f"All d_nearest_neg_nd values matched between mean_center=True/False on "
        f"probe={a_probe!r} — the round-7 threading is incomplete. Raw: {dnn_raw}, "
        f"MC: {dnn_mc}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Blocker 2 — #472 loader defaults stay RAW.
# ─────────────────────────────────────────────────────────────────────────────


def _write_round6_bundle(
    path: Path, names: list[str], dim: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write a round-6-shape #472 centroids bundle with BOTH raw + mean-centered
    cosine matrices.

    Returns (centroids, cos_raw, cos_mean_centered) so the test can verify
    which key the loader reads.
    """
    rng = np.random.default_rng(7)
    # Use the same shared-mean trick so raw vs mean-centered DIFFER measurably.
    shared = rng.standard_normal(dim).astype(np.float64)
    shared = shared / np.linalg.norm(shared) * 5.0
    mat = (rng.standard_normal((len(names), dim)).astype(np.float64) + shared).astype(np.float32)

    # Raw cosine (no centering).
    mat64 = mat.astype(np.float64)
    norms_raw = np.linalg.norm(mat64, axis=1, keepdims=True)
    norms_raw = np.where(norms_raw == 0.0, 1.0, norms_raw)
    unit_raw = mat64 / norms_raw
    cos_raw = (unit_raw @ unit_raw.T).astype(np.float32)

    # Mean-centered cosine.
    mat_mc = mat64 - mat64.mean(axis=0, keepdims=True)
    norms_mc = np.linalg.norm(mat_mc, axis=1, keepdims=True)
    norms_mc = np.where(norms_mc == 0.0, 1.0, norms_mc)
    unit_mc = mat_mc / norms_mc
    cos_mc = (unit_mc @ unit_mc.T).astype(np.float32)

    torch.save(
        {
            "centroids": torch.from_numpy(mat),
            "persona_names": names,
            "cos_matrix": torch.from_numpy(cos_raw),
            "cos_matrix_mean_centered": torch.from_numpy(cos_mc),
            "layer": 10,
            "base_model": "synthetic-test",
            "questions": ["q_0"],
        },
        str(path),
    )
    return mat, cos_raw, cos_mc


def test_load_cos_matrix_default_is_raw_round_7(tmp_path: Path) -> None:
    """``load_cos_matrix(layer, dir)`` UNQUALIFIED must read ``cos_matrix``
    (raw), not ``cos_matrix_mean_centered``.

    This pins blocker 2: the round-6 default flip silently shifted every
    #472 / #477 / #500 caller to mean-centered geometry. Round 7 reverts.
    """
    names = ["villain", "qwen_default", "persona_a", "persona_b", "persona_c"]
    _, cos_raw, cos_mc = _write_round6_bundle(tmp_path / "centroids_L10.pt", names)

    # UNQUALIFIED call — the published #472 default path.
    cos_dict, loaded_names = load_cos_matrix(10, tmp_path)

    assert loaded_names == names
    # Must match the RAW field, not the mean-centered field.
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            assert cos_dict[a][b] == pytest.approx(float(cos_raw[i, j]), abs=1e-6), (
                f"load_cos_matrix({a!r},{b!r}) returned {cos_dict[a][b]!r} but raw is "
                f"{cos_raw[i, j]!r} — the default leaked through to mean-centered "
                f"({cos_mc[i, j]!r})"
            )

    # Sanity: raw and mean-centered MUST measurably differ on this fixture
    # (otherwise the test would trivially pass).
    assert not np.allclose(cos_raw, cos_mc), (
        "Synthetic fixture's raw and mean-centered cosine matrices are too close — "
        "bump the shared_mean_norm in _write_round6_bundle."
    )


def test_cos_to_source_default_is_raw_round_7(tmp_path: Path) -> None:
    """``cos_to_source(layer, source, dir)`` UNQUALIFIED reproduces the published
    #472 raw cos-to-source values, not the mean-centered ones."""
    names = ["villain", "qwen_default", "persona_a", "persona_b", "persona_c"]
    _, cos_raw, cos_mc = _write_round6_bundle(tmp_path / "centroids_L10.pt", names)

    # UNQUALIFIED — the i472_run_cell.py / i472_eval_trajectory.py / i477_reval_confirm.py path.
    cts = cos_to_source(10, "villain", tmp_path)

    src_idx = names.index("villain")
    for j, p in enumerate(names):
        expected_raw = float(cos_raw[src_idx, j])
        assert cts[p] == pytest.approx(expected_raw, abs=1e-6), (
            f"cos_to_source[{p!r}] returned {cts[p]!r} but raw is {expected_raw!r} — "
            f"the default leaked to mean-centered ({float(cos_mc[src_idx, j])!r})"
        )

    # Sanity check that the mean-centered field is materially different.
    raw_to_villain = [
        float(cos_raw[src_idx, j]) for j, _ in enumerate(names) if names[j] != "villain"
    ]
    mc_to_villain = [
        float(cos_mc[src_idx, j]) for j, _ in enumerate(names) if names[j] != "villain"
    ]
    raw_span = max(raw_to_villain) - min(raw_to_villain)
    mc_span = max(mc_to_villain) - min(mc_to_villain)
    assert mc_span > raw_span, (mc_span, raw_span)


def test_load_cos_matrix_explicit_global_mean_round_7(tmp_path: Path) -> None:
    """``load_cos_matrix(..., centering="global_mean")`` still reads the
    mean-centered field — #504 callers explicitly opt in to round-6 geometry."""
    names = ["villain", "qwen_default", "persona_a", "persona_b"]
    _, _cos_raw, cos_mc = _write_round6_bundle(tmp_path / "centroids_L15.pt", names)

    cos_dict, _ = load_cos_matrix(15, tmp_path, centering="global_mean")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            assert cos_dict[a][b] == pytest.approx(float(cos_mc[i, j]), abs=1e-6)


def test_cos_to_source_explicit_global_mean_round_7(tmp_path: Path) -> None:
    """``cos_to_source(..., centering="global_mean")`` reads mean-centered values."""
    names = ["villain", "qwen_default", "persona_a", "persona_b"]
    _, _cos_raw, cos_mc = _write_round6_bundle(tmp_path / "centroids_L20.pt", names)

    cts = cos_to_source(20, "villain", tmp_path, centering="global_mean")
    src_idx = names.index("villain")
    for j, p in enumerate(names):
        assert cts[p] == pytest.approx(float(cos_mc[src_idx, j]), abs=1e-6)


def test_load_cos_matrix_pre_round_6_bundle_still_works(tmp_path: Path) -> None:
    """A pre-round-6 bundle (no ``cos_matrix_mean_centered`` field) must still
    load under the round-7 default (``centering="none"``) — only the explicit
    ``"global_mean"`` opt-in should KeyError.

    This is the regression Codex flagged at round 4: local caches that predate
    round 6 must keep working. Round 6 made them KeyError unconditionally
    because the loader default WAS ``"global_mean"``.
    """
    names = ["villain", "qwen_default", "persona_a"]
    rng = np.random.default_rng(11)
    mat = rng.standard_normal((len(names), 8)).astype(np.float32)
    cos = (mat @ mat.T).astype(np.float32)
    # PRE-round-6 schema — ONLY cos_matrix, no cos_matrix_mean_centered.
    torch.save(
        {
            "centroids": torch.from_numpy(mat),
            "persona_names": names,
            "cos_matrix": torch.from_numpy(cos),
            "layer": 10,
            "base_model": "synthetic-test-pre-round-6",
            "questions": ["q_0"],
        },
        str(tmp_path / "centroids_L10.pt"),
    )

    # Default path works.
    cos_dict, loaded_names = load_cos_matrix(10, tmp_path)
    assert loaded_names == names
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            assert cos_dict[a][b] == pytest.approx(float(cos[i, j]), abs=1e-6)

    # Explicit global_mean opt-in fails loud on the pre-round-6 schema.
    with pytest.raises(KeyError, match="cos_matrix_mean_centered"):
        load_cos_matrix(10, tmp_path, centering="global_mean")
