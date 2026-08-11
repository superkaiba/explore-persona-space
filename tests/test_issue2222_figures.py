"""Tests for scripts/issue2222_figures.py (issue #2222 unit-3 P5 figures).

Pins the gotchas.md xerr/yerr non-negative-offsets rule: a deliberately
INVERTED bootstrap quantile CI (every draw strictly above the point estimate,
so ``quantile(0.025) > r``) is routed through the REAL hero figure function to
``savefig`` — without the element-wise clamp in ``ci_offsets`` matplotlib
raises ``ValueError: 'yerr' must not contain negative values`` at render time
(#1335/#547 class). Fixtures are synthetic JSON/npz shapes mirroring the P3
``stage_aggregate`` outputs; everything runs offline on ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2222_figures as figs

TRAITS = ("evil", "sycophancy", "hallucination")
ARMS = ["raw", "exact_dp", "prompt_dp", "mapped_ctx", "mapped_pfx", "id_bias"]
N_LAYERS = 28
STEER = {"evil": 19, "sycophancy": 19, "hallucination": 15}
DATASETS = [
    f"{fam}_{ver}"
    for fam in ("evil", "sycophancy")
    for ver in ("normal", "misaligned_1", "misaligned_2")
]


@pytest.fixture(autouse=True)
def _style() -> None:
    figs.set_paper_style("blog")  # idempotent; figure fns assume the style is set


def _corr_fixture(rng: np.random.Generator) -> dict:
    records = []
    for trait in TRAITS:
        for arm in ARMS:
            r_layers = rng.uniform(-0.9, 0.9, size=N_LAYERS)
            records.append(
                {
                    "trait": trait,
                    "arm": arm,
                    "layer_regime": "steer",
                    "layer": STEER[trait],
                    "r": float(r_layers[STEER[trait]]),
                    "perm_p_fixed_layer": 0.02,
                    "published_r": 0.6 if arm in ("raw", "exact_dp", "prompt_dp") else None,
                }
            )
            records.append(
                {
                    "trait": trait,
                    "arm": arm,
                    "layer_regime": "sweep",
                    "r_per_layer": [float(v) for v in r_layers],
                }
            )
    dataset_values = {}
    for trait in TRAITS:
        dataset_values[trait] = {
            "steer_layer": STEER[trait],
            "y_trait_score": {ds: float(rng.uniform(0, 100)) for ds in DATASETS},
            "arms": {arm: {ds: float(rng.standard_normal()) for ds in DATASETS} for arm in ARMS},
        }
    delta_rec = {
        "delta_r": 0.1,
        "ci95_frozen_flat": [0.02, 0.2],
        "ci95_frozen_clustered": [-0.05, 0.25],
        "ci95_selection_inherited_flat": [0.0, 0.22],
        "ci95_selection_inherited_clustered": [-0.1, 0.3],
    }
    return {
        "arm_order": ARMS,
        "datasets": DATASETS,
        "family_of_dataset": {ds: figs.split_dataset_id(ds)[0] for ds in DATASETS},
        "dataset_values": dataset_values,
        "records": records,
        "hypothesis_tests": {
            "H1_sycophancy_gap": dict(delta_rec),
            "H2_equivalence": {t: dict(delta_rec) for t in TRAITS},
        },
    }


def _write_null_matrices(nulls_dir: Path, corr: dict, rng: np.random.Generator) -> None:
    """boot_flat: every draw STRICTLY ABOVE the point estimate -> inverted CI."""
    nulls_dir.mkdir(parents=True, exist_ok=True)
    steer_r = {
        (rec["trait"], rec["arm"]): rec["r"]
        for rec in corr["records"]
        if rec["layer_regime"] == "steer"
    }
    for trait in TRAITS:
        for arm in ARMS:
            r = steer_r[(trait, arm)]
            boot = np.full((50, N_LAYERS), r + 0.05)  # constant draws above r
            np.savez(nulls_dir / f"boot_flat_{trait}_{arm}.npz", r=boot.astype(np.float32))
            perm = np.abs(rng.uniform(0, 0.5, size=(50, N_LAYERS)))
            np.savez(nulls_dir / f"perm_{trait}_{arm}.npz", abs_r=perm.astype(np.float32))


def _assert_png(fig_dir: Path, stem: str) -> None:
    png = fig_dir / f"{stem}.png"
    assert png.exists(), f"{png} not written"
    assert png.stat().st_size > 5_000, f"{png} suspiciously small ({png.stat().st_size} B)"


def test_ci_offsets_clamps_inverted_interval() -> None:
    err_lo, err_hi = figs.ci_offsets(0.5, 0.55, 0.6)  # lo > value: inverted
    assert err_lo == 0.0 and err_hi == pytest.approx(0.1)


def test_hero_renders_through_inverted_quantile_ci(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    corr = _corr_fixture(rng)
    nulls_dir = tmp_path / "nulls"
    _write_null_matrices(nulls_dir, corr, rng)
    # The fixture is genuinely inverted: quantile lower bound sits ABOVE r.
    boot = np.load(nulls_dir / "boot_flat_evil_raw.npz")["r"][:, STEER["evil"]]
    r_evil_raw = next(
        rec["r"]
        for rec in corr["records"]
        if rec["layer_regime"] == "steer" and rec["trait"] == "evil" and rec["arm"] == "raw"
    )
    assert np.nanquantile(boot, 0.025) > r_evil_raw
    fig_dir = tmp_path / "figs"
    stems = figs.fig_hero(corr, nulls_dir, fig_dir)
    assert stems == ["hero_arm_r_comparison"]
    _assert_png(fig_dir, "hero_arm_r_comparison")


def test_scatters_sweeps_and_ci_figures_render(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    corr = _corr_fixture(rng)
    nulls_dir = tmp_path / "nulls"
    _write_null_matrices(nulls_dir, corr, rng)
    fig_dir = tmp_path / "figs"
    for stem in figs.fig_scatters(corr, fig_dir):
        _assert_png(fig_dir, stem)
    for stem in figs.fig_layer_sweeps(corr, nulls_dir, fig_dir):
        _assert_png(fig_dir, stem)
    for stem in figs.fig_ci_schemes(corr, fig_dir):
        _assert_png(fig_dir, stem)


def test_roc_renders_from_persisted_scores(tmp_path: Path) -> None:
    """Round-2 C8: the ROC figure renders from the PERSISTED per-sample scores
    (roc_scores_<trait>.npz) + the persisted auc json — never recomputing."""
    rng = np.random.default_rng(2)
    corr = _corr_fixture(rng)
    nulls_dir = tmp_path / "nulls"
    nulls_dir.mkdir(parents=True, exist_ok=True)
    labels = rng.random(40) > 0.5
    labels[:2] = [True, False]  # both classes guaranteed
    aucj = {"records": []}
    for trait in TRAITS:
        np.savez(
            nulls_dir / f"roc_scores_{trait}.npz",
            labels=labels,
            steer_layer=np.int64(STEER[trait]),
            **{f"score_{arm}": rng.standard_normal(40).astype(np.float32) for arm in ARMS},
        )
        aucj["records"].extend(
            {"trait": trait, "arm": arm, "layer": STEER[trait], "auc": 0.5, "n_pos": 1, "n_neg": 1}
            for arm in ARMS
        )
    fig_dir = tmp_path / "figs"
    stems = figs.fig_roc(corr, aucj, nulls_dir, fig_dir)
    assert stems == ["roc_by_arm"]
    _assert_png(fig_dir, "roc_by_arm")
    # A missing persisted-scores npz fails loud with the re-run hint.
    (nulls_dir / "roc_scores_evil.npz").unlink()
    with pytest.raises(FileNotFoundError, match="stage aggregate"):
        figs.fig_roc(corr, aucj, nulls_dir, tmp_path / "figs2")


def test_ci_scheme_colors_disjoint_from_arm_palette() -> None:
    """Round-2 C7: the CI-scheme ramp shares no color with the pinned ARM palette."""
    assert not set(figs._SCHEME_GREYS) & set(figs.ARM_COLORS.values())


def test_split_dataset_id_suffix_safe() -> None:
    assert figs.split_dataset_id("mistake_medical_misaligned_1") == (
        "mistake_medical",
        "misaligned_1",
    )
    assert figs.split_dataset_id("evil_normal") == ("evil", "normal")
    with pytest.raises(ValueError):
        figs.split_dataset_id("not_a_dataset")
