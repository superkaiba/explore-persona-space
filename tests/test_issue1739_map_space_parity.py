"""#1975 map input-space parity — the durable gate for the #1739 incident.

The #1739 natural-PV pipeline applied a WHITENED-fit map to RAW activations
(sycophancy headline 0.577 quoted+committed, corrected to 0.486 ~21 h later);
nothing in the payload<->consumer seam declared or checked the input space.
These tests pin the new parity surface in
``src/explore_persona_space/experiments/issue_1739/fits.py``:

- ``whitening_provenance`` — artifact form (sha256 of the shared persisted
  whitening FILE) and recipe form (recomputation-stable discrete tuple; float
  matrices never hashed);
- ``map_space_meta`` — fit-space tag + provenance + per-layer train-input norm
  stats recorded by the writer;
- ``assert_map_input_space`` — raise on an UNDECLARED cross-space apply,
  loud-warn (never silent, never crash) on LEGACY payloads and DECLARED
  mismatches;
- ``check_whitening_parity`` — the whitened-fold parity check (artifact sha /
  recipe fields; mismatch on comparable fields raises; no comparable fields
  degrades to a loud warning);
- the ``_save_map`` writer round-trip (both the linear ``.npz`` and nonlinear
  ``.pt`` payload forms carry the new fields).

Everything is tiny + synthetic (2 layers, d<=512, fixed seeds); no network,
no staged data, no GPU, no production-size tensors.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue_1739 import fits  # noqa: E402

FITS_LOGGER = "explore_persona_space.experiments.issue_1739.fits"
LY, N_U, DIM = 2, 40, 8
RNG = np.random.default_rng(0)


def _tiny_pool(seed: int = 0, n: int = N_U) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(LY, n, DIM))


def _persist_whitening_natpv_style(wh, path: Path, *, seed: int = 0, n_u_rows: int = N_U) -> None:
    """Persist in the natpv ``phase_whitening`` layout: mu/w fp32, gamma fp64 ARRAY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.savez(
            fh,
            mu=np.asarray(wh.mu, dtype=np.float32),
            w=np.asarray(wh.w, dtype=np.float32),
            gamma=np.asarray(wh.gamma, dtype=np.float64),
            meta=json.dumps(
                {"variant": "context_end", "u_size": "full", "n_u_rows": n_u_rows, "seed": seed}
            ),
        )


def _recipe_prov(wh, *, seed: int = 0, n_u_rows: int = N_U) -> dict:
    """The RECIPE-form provenance the fits CLI records at fit time."""
    return fits.whitening_provenance(
        variant="context_end",
        u_label="full",
        whiten_seed=seed,
        n_u_rows=n_u_rows,
        gammas=wh.gamma,
    )


def _fits_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == FITS_LOGGER]


# ---------------------------------------------------------------------------
# whitening_provenance — artifact + recipe forms
# ---------------------------------------------------------------------------


def test_provenance_artifact_form_equality_on_shared_file(tmp_path):
    """Both sides hashing the SAME persisted whitening file compare equal."""
    wh = fits.fit_whitening(_tiny_pool(0), seed=0)
    path = tmp_path / "context_end__ufull.npz"
    _persist_whitening_natpv_style(wh, path)
    a = fits.whitening_provenance(whitening_file=path)
    b = fits.whitening_provenance(whitening_file=path)
    assert a["whitening_file_sha256"] == b["whitening_file_sha256"]
    assert fits.check_whitening_parity(a, b) == "artifact-match"


def test_provenance_artifact_form_mismatch_raises(tmp_path):
    """Two DIFFERENT whitening files -> different shas -> parity raises."""
    wh0 = fits.fit_whitening(_tiny_pool(0), seed=0)
    wh1 = fits.fit_whitening(_tiny_pool(7), seed=0)
    p0, p1 = tmp_path / "a.npz", tmp_path / "b.npz"
    _persist_whitening_natpv_style(wh0, p0)
    _persist_whitening_natpv_style(wh1, p1)
    with pytest.raises(ValueError, match="whitening parity FAILED"):
        fits.check_whitening_parity(
            fits.whitening_provenance(whitening_file=p0),
            fits.whitening_provenance(whitening_file=p1),
        )


def test_recipe_form_stable_across_refits():
    """Two fit_whitening runs on the SAME inputs give EQUAL recipe tuples.

    The recipe form deliberately carries only DISCRETE fields (variant,
    u_label, seed, row count, grid-selected gammas) — float matrices are never
    hashed, because two fits are not byte-identical across devices/BLAS.
    """
    wh_a = fits.fit_whitening(_tiny_pool(0), seed=0)
    wh_b = fits.fit_whitening(_tiny_pool(0), seed=0)
    prov_a, prov_b = _recipe_prov(wh_a), _recipe_prov(wh_b)
    assert prov_a == prov_b
    assert fits.check_whitening_parity(prov_a, prov_b) == "recipe-match"


def test_fp32_persist_roundtrip_keeps_parity(tmp_path):
    """The natpv fp32 mu/w persist keeps gamma fp64 -> recipe parity survives."""
    wh = fits.fit_whitening(_tiny_pool(0), seed=0)
    path = tmp_path / "context_end__ufull.npz"
    _persist_whitening_natpv_style(wh, path)
    with np.load(path, allow_pickle=False) as z:
        reloaded_gammas = np.asarray(z["gamma"], dtype=np.float64)
        wmeta = json.loads(str(z["meta"]))
    loaded_prov = fits.whitening_provenance(
        whitening_file=path,
        variant=wmeta["variant"],
        u_label=wmeta["u_size"],
        whiten_seed=wmeta["seed"],
        n_u_rows=wmeta["n_u_rows"],
        gammas=reloaded_gammas,
    )
    assert fits.check_whitening_parity(_recipe_prov(wh), loaded_prov) == "recipe-match"


def test_provenance_empty_raises():
    with pytest.raises(ValueError, match="at least one field"):
        fits.whitening_provenance()


# ---------------------------------------------------------------------------
# check_whitening_parity — positive fold path / mismatch / degrade
# ---------------------------------------------------------------------------


def test_positive_fold_path_matched_pair_passes(tmp_path, caplog):
    """A MATCHED persisted-whitening pair passes: no raise, no warning.

    The healthy whitened fold by construction: the map side carries the
    fits-CLI RECIPE-form provenance, the loaded side is built from the
    persisted whitening file (artifact sha + the same recipe fields) — mixed
    forms fall back to the shared recipe fields and MATCH.
    """
    wh = fits.fit_whitening(_tiny_pool(0), seed=0)
    path = tmp_path / "context_end__ufull.npz"
    _persist_whitening_natpv_style(wh, path)
    with np.load(path, allow_pickle=False) as z:
        gammas = np.asarray(z["gamma"], dtype=np.float64)
    loaded_prov = fits.whitening_provenance(
        whitening_file=path,
        variant="context_end",
        u_label="full",
        whiten_seed=0,
        n_u_rows=N_U,
        gammas=gammas,
    )
    with caplog.at_level(logging.WARNING):
        grade = fits.check_whitening_parity(_recipe_prov(wh), loaded_prov)
    assert grade == "recipe-match"
    assert _fits_warnings(caplog) == []


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("whiten_seed", 999),
        ("n_u_rows", N_U + 1),
        ("u_label", "250"),
        ("variant", "prefix_end"),
    ],
)
def test_recipe_mismatch_on_comparable_fields_raises(field, bad_value):
    wh = fits.fit_whitening(_tiny_pool(0), seed=0)
    a = _recipe_prov(wh)
    b = dict(_recipe_prov(wh))
    b[field] = bad_value
    with pytest.raises(ValueError, match=field):
        fits.check_whitening_parity(a, b)


def test_gamma_mismatch_raises_naming_selection_flip():
    """A gamma flip is a REAL transform difference — raise, name the flip cause."""
    wh = fits.fit_whitening(_tiny_pool(0), seed=0)
    a = _recipe_prov(wh)
    b = dict(a)
    b["gamma_per_layer"] = [g + 0.05 for g in a["gamma_per_layer"]]
    with pytest.raises(ValueError, match="gamma-selection flip"):
        fits.check_whitening_parity(a, b)


def test_legacy_parity_degrades_to_loud_warning(caplog):
    """No comparable fields (legacy map payload) -> loud warning, never a crash."""
    wh = fits.fit_whitening(_tiny_pool(0), seed=0)
    with caplog.at_level(logging.WARNING):
        grade = fits.check_whitening_parity(None, _recipe_prov(wh))
    assert grade == "degraded-legacy"
    warnings = _fits_warnings(caplog)
    assert len(warnings) == 1
    # The degrade report names diagnosable causes (critic implementer-note ii).
    assert "gamma-selection flip" in warnings[0]
    assert "#1739" in warnings[0]


def test_legacy_parity_strict_mode_raises():
    wh = fits.fit_whitening(_tiny_pool(0), seed=0)
    with pytest.raises(ValueError, match="DEGRADED"):
        fits.check_whitening_parity({}, _recipe_prov(wh), on_legacy_warn=False)


# ---------------------------------------------------------------------------
# assert_map_input_space — raise / warn / declared / legacy
# ---------------------------------------------------------------------------


def _whitened_like(seed: int, n: int = 60, d: int = DIM) -> np.ndarray:
    """Rows with per-dim unit variance — norm concentrates at sqrt(d)."""
    return np.random.default_rng(seed).normal(size=(LY, n, d))


def test_assert_in_band_passes(caplog):
    x_fit = _whitened_like(1)
    meta = fits.map_space_meta(x_fit, fit_space="whitened", whitening_prov=None)
    with caplog.at_level(logging.WARNING):
        fits.assert_map_input_space(meta, _whitened_like(2))  # fresh draw, same space
    assert _fits_warnings(caplog) == []


def test_undeclared_mismatch_raises():
    """x10-scaled inputs (raw-vs-whitened class) RAISE, naming #1739.

    The fail-loud pin: an undeclared cross-space apply is never silently
    swallowed (plan 'Fail-loud pin').
    """
    x_fit = _whitened_like(1)
    meta = fits.map_space_meta(x_fit, fit_space="whitened", whitening_prov=None)
    with pytest.raises(ValueError, match="#1739"):
        fits.assert_map_input_space(meta, 10.0 * _whitened_like(2))


def test_legacy_meta_warns_not_raises(caplog):
    """A payload without fit_space warns loudly (naming #1739) and returns."""
    with caplog.at_level(logging.WARNING):
        fits.assert_map_input_space({"apply": "pred = ..."}, _whitened_like(2))
        fits.assert_map_input_space(None, None)  # x is never touched on this branch
    warnings = _fits_warnings(caplog)
    assert len(warnings) == 2
    assert all("#1739" in w and "LEGACY" in w for w in warnings)


def test_declared_mismatch_warns_not_raises(caplog):
    """A DECLARED cross-space read warns with the verbatim reason and returns."""
    x_fit = _whitened_like(1)
    meta = fits.map_space_meta(x_fit, fit_space="whitened", whitening_prov=None)
    reason = "--space raw provisional read (disclosed; natpv docstring)"
    with caplog.at_level(logging.WARNING):
        # x=None pins the contract the natpv/readout call sites rely on: the
        # declared branch returns before x is touched.
        fits.assert_map_input_space(meta, None, declared_mismatch=reason)
    warnings = _fits_warnings(caplog)
    assert len(warnings) == 1
    assert reason in warnings[0]


def test_partial_meta_degrades_to_warning(caplog):
    """fit_space present but no recorded norm stats -> loud warning, no crash."""
    with caplog.at_level(logging.WARNING):
        fits.assert_map_input_space({"fit_space": "whitened"}, _whitened_like(2))
    warnings = _fits_warnings(caplog)
    assert len(warnings) == 1 and "PARTIAL" in warnings[0]


def test_single_layer_input_with_layer_indices():
    """(n, d) single-layer x aligns onto the payload stats via layer_indices."""
    x_fit = _whitened_like(1)
    meta = fits.map_space_meta(x_fit, fit_space="whitened", whitening_prov=None)
    z = _whitened_like(3)[1]  # one layer, (n, d)
    fits.assert_map_input_space(meta, z, layer_indices=[1])  # in-band: no raise
    with pytest.raises(ValueError, match="layer_idx 1"):
        fits.assert_map_input_space(meta, 10.0 * z, layer_indices=[1])


def test_layer_count_mismatch_fails_loud():
    x_fit = _whitened_like(1)
    meta = fits.map_space_meta(x_fit, fit_space="whitened", whitening_prov=None)
    with pytest.raises(ValueError, match="layer_indices"):
        fits.assert_map_input_space(meta, _whitened_like(2)[:1])


def test_band_separates_whitened_from_raw_at_realistic_scale_ratios():
    """Band-margin sanity check on the RECORDED-stats logic (plan kill criterion).

    Whitened rows concentrate at ||x|| ~= sqrt(d) with relative fluctuation
    ~O(1/sqrt(2d)) (~1.2% at the production d=3584; ~5% at the d=512 used
    here), while feeding RAW residual-stream summaries to a whitened-fit map
    is the applycheck's documented "norm mean >> sqrt(dim)" OOD signal —
    several-fold larger. The default band 2.0 therefore sits >=20x above the
    within-space fluctuation and >=1.5x below even a conservative 3x
    cross-space scale ratio: the signal SEPARATES, so the norm leg is KEPT
    (the plan's drop-the-norm-leg kill criterion is not triggered).
    """
    d = 512
    x_fit = np.random.default_rng(11).normal(size=(LY, 80, d))
    meta = fits.map_space_meta(x_fit, fit_space="whitened", whitening_prov=None)
    same_space = np.random.default_rng(12).normal(size=(LY, 80, d))
    fits.assert_map_input_space(meta, same_space)  # within-space: passes
    fits.assert_map_input_space(meta, 1.2 * same_space)  # mild drift: still in band
    for ratio in (3.0, 5.0, 10.0):
        with pytest.raises(ValueError, match="#1739"):
            fits.assert_map_input_space(meta, ratio * same_space)
        with pytest.raises(ValueError, match="#1739"):
            fits.assert_map_input_space(meta, same_space / ratio)


# ---------------------------------------------------------------------------
# map_space_meta — recorded stats
# ---------------------------------------------------------------------------


def test_map_space_meta_records_per_layer_norm_stats():
    x = np.random.default_rng(4).normal(size=(LY, 30, DIM))
    prov = {"variant": "context_end"}
    sm = fits.map_space_meta(x, fit_space="whitened", whitening_prov=prov)
    assert sm["fit_space"] == "whitened"
    assert sm["whitening_provenance"] == prov
    assert len(sm["train_input_norm_mean"]) == LY
    assert len(sm["train_input_norm_std"]) == LY
    expected = [float(np.linalg.norm(x[li], axis=1).mean()) for li in range(LY)]
    np.testing.assert_allclose(sm["train_input_norm_mean"], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# _save_map writer round-trip (both payload forms)
# ---------------------------------------------------------------------------


def _fits_cli():
    from scripts import issue1739_fits as cli

    return cli


def _tiny_space_meta(seed: int = 5) -> dict:
    x_w = np.random.default_rng(seed).normal(size=(LY, 20, DIM))
    wh = fits.fit_whitening(_tiny_pool(0), seed=0)
    return fits.map_space_meta(
        x_w,
        fit_space="whitened",
        whitening_prov=_recipe_prov(wh),
    )


def test_save_map_linear_roundtrip_carries_space_meta(tmp_path):
    """Linear .npz writer round-trip: fields present + provenance matches input."""
    cli = _fits_cli()
    rng = np.random.default_rng(6)
    mapfit = fits.MapFit(
        w=rng.normal(size=(LY, DIM, DIM)),
        x_mu=rng.normal(size=(LY, 1, DIM)),
        x_sd=np.abs(rng.normal(size=(LY, 1, DIM))) + 0.5,
        y_mu=rng.normal(size=(LY, 1, DIM)),
        diagnostics={"w_fit_rows": 20, "solver": "test"},
    )
    sm = _tiny_space_meta()
    out = cli._save_map(
        tmp_path, "context_end", "full", mapfit, range(LY), map_seed=0, space_meta=sm
    )
    with np.load(out, allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
    assert meta["fit_space"] == "whitened"
    assert meta["whitening_provenance"] == sm["whitening_provenance"]
    np.testing.assert_allclose(meta["train_input_norm_mean"], sm["train_input_norm_mean"])
    np.testing.assert_allclose(meta["train_input_norm_std"], sm["train_input_norm_std"])
    # And the reloaded meta drives the apply-time gate end to end.
    fits.assert_map_input_space(meta, np.random.default_rng(7).normal(size=(LY, 20, DIM)))
    with pytest.raises(ValueError, match="#1739"):
        fits.assert_map_input_space(
            meta, 10.0 * np.random.default_rng(7).normal(size=(LY, 20, DIM))
        )


def test_save_map_nonlinear_roundtrip_carries_space_meta(tmp_path):
    """Nonlinear .pt writer round-trip: the SAME fields ride the torch payload."""
    torch = pytest.importorskip("torch")
    cli = _fits_cli()
    mapfit = fits.MapFit(
        w=None,
        x_mu=None,
        x_sd=None,
        y_mu=None,
        diagnostics={"w_fit_rows": 20, "solver": "test", "per_layer": [{}, {}]},
        kind="mlp",
        nl_payloads=({"tag": "l0"}, {"tag": "l1"}),
    )
    sm = _tiny_space_meta()
    out = cli._save_map(
        tmp_path, "context_end", "full", mapfit, range(LY), map_seed=0, space_meta=sm
    )
    blob = torch.load(out, map_location="cpu", weights_only=False)
    assert blob["meta"]["fit_space"] == "whitened"
    assert blob["meta"]["whitening_provenance"] == sm["whitening_provenance"]


def test_save_map_without_space_meta_omits_fields(tmp_path):
    """An unthreaded caller (space_meta=None) keeps the pre-#1975 payload shape."""
    cli = _fits_cli()
    rng = np.random.default_rng(8)
    mapfit = fits.MapFit(
        w=rng.normal(size=(LY, DIM, DIM)),
        x_mu=rng.normal(size=(LY, 1, DIM)),
        x_sd=np.abs(rng.normal(size=(LY, 1, DIM))) + 0.5,
        y_mu=rng.normal(size=(LY, 1, DIM)),
        diagnostics={"w_fit_rows": 20},
    )
    out = cli._save_map(tmp_path, "context_end", "full", mapfit, range(LY), map_seed=0)
    with np.load(out, allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
    assert "fit_space" not in meta
    assert "whitening_provenance" not in meta
