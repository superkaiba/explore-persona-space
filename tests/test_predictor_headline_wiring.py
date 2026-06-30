# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Σ, ρ, δ, r_B, ×, ⁻¹, ᵀ, Δ) in scientific docstrings + asserts.
"""Production-headline orchestration wiring for the issue-666 Phase-4 predictor.

Round-2 code-review findings (plan §4b/§4c/§4d/§4g, §6.5 predictor_headline.json):

  1. ``--sigma-inv`` threads the broad-corpus Σc⁻¹ into the gate; the per-cell
     result then carries ``sigma_c_corpus_kind="broad"`` + ``..._headline_eligible``.
  2. ``enumerate_store_cells`` returns the full HF store cell set (production
     headline enumeration); ``--slice``/``--cells N`` keep the smoke slice.
  3. Mixed ``r_B`` routing (``rb_for_cell``): diffmeans for bad-medical/EM,
     per-cell ``r_plus`` for taught-fact/marker; the ``--r-b-source`` toggle.
  4. The cross-behavior grid (``predict_cell_grid``) emits ``per_target`` with a
     row for every available target behavior.
  5. ``run_lobo_loco`` returns BOTH ``lobo`` and ``loco`` keys, ``loco`` non-empty
     when the input cells span ≥2 behaviors.

These use ``tmp_path`` synthetic store cells (the documented #664 schema) +
synthetic ``rb_columns`` / ``sigma_c_inv.pt``; CPU-only, no network, no GPU.
"""

from __future__ import annotations

import json
import sys
import typing
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402  (installed; not a TDD-deferred dep)


class _LazyModule:
    """Proxy that imports a per-issue script on first attribute access."""

    def __init__(self, dotted: str):
        self._dotted = dotted

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)


loader = _LazyModule("issue666_load_store")
predscore = _LazyModule("issue666_predictor")
cv = _LazyModule("issue666_lobo_loco")

N_LAYER = 4
D = 32
N_CTX = 50
N_PROBE = 8
LAYER = 2  # the test layer (< N_LAYER)


def _write_fake_cell(cell_dir: Path, *, behavior: str, source: str = "default", seed: int = 0):
    """Fabricate a #664-schema store cell with a chosen ``behavior`` meta field.

    Δv(C') carries a target-structured signal correlated across contexts so the
    own-behavior L̂ ρ is well above zero (a real-signal arm). The source anchor
    (index 0) carries the install ŵ. ``behavior`` drives the r_B routing under test.
    """
    rng = np.random.default_rng(seed)
    cell_dir.mkdir(parents=True, exist_ok=True)
    v0 = rng.standard_normal((N_CTX, N_LAYER, D)).astype("float32")
    install = (rng.standard_normal((N_LAYER, D)).astype("float32")) * 3.0
    dv = rng.standard_normal((N_CTX, N_LAYER, D)).astype("float32")
    struct = rng.standard_normal((N_CTX, 1, 1)).astype("float32")
    dv = dv + struct * install[None]
    dv[0] = install  # source-anchor ĝ^real(C)=1
    v_plus = v0 + dv
    vpp = (
        v_plus[:, None] + rng.standard_normal((N_CTX, N_PROBE, N_LAYER, D)).astype("float32") * 0.1
    )
    v0p = v0[:, None] + rng.standard_normal((N_CTX, N_PROBE, N_LAYER, D)).astype("float32") * 0.1
    obj = {
        "v_plus": torch.from_numpy(v_plus),
        "v0": torch.from_numpy(v0),
        "v_plus_probe": torch.from_numpy(vpp),
        "v0_probe": torch.from_numpy(v0p),
        "c_C_base": torch.from_numpy(rng.standard_normal((N_CTX, N_LAYER, D)).astype("float32")),
        "c_C_trained": torch.from_numpy(rng.standard_normal((N_CTX, N_LAYER, D)).astype("float32")),
        "t_CB": torch.from_numpy(rng.standard_normal((N_LAYER, D)).astype("float32")),
        "r_plus": torch.from_numpy(install),
        "context_ids": list(range(N_CTX)),
    }
    torch.save(obj, cell_dir / "tensors.pt")
    (cell_dir / "meta.json").write_text(
        json.dumps(
            {
                "behavior": behavior,
                "source": source,
                "source_idx": 0,
                "arm": "contra",
                "target_context_roles": ["source-anchor"] + ["bystander"] * (N_CTX - 1),
            }
        )
    )


def _fake_rb_columns(seed: int = 7) -> dict:
    """Synthetic #658 diffmeans columns map (the load_rb_columns return shape)."""
    rng = np.random.default_rng(seed)
    return {
        "broad_em": rng.standard_normal(D).astype(np.float64),
        "harmful_compliance": rng.standard_normal(D).astype(np.float64),
        "sycophancy": rng.standard_normal(D).astype(np.float64),
        "refusal": rng.standard_normal(D).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Finding 1 — --sigma-inv threads the broad-corpus whitening + flags eligibility.
# ---------------------------------------------------------------------------
def test_sigma_inv_load_marks_broad_and_headline_eligible(tmp_path):
    """load_sigma_inv reads the broad-corpus Σc⁻¹ + marks it headline-eligible."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((D, D))
    Sigma_inv = np.linalg.inv(A @ A.T + np.eye(D))
    sig_path = tmp_path / "sigma_c_inv.pt"
    torch.save(
        {
            "Sigma_inv": torch.from_numpy(Sigma_inv),
            "headline_eligible": True,
            "lam": 0.1,
            "cond_number": 1234.0,
            "n_contexts": 3000,
            "layer": 14,
        },
        sig_path,
    )
    si, meta = predscore.load_sigma_inv(sig_path, layer=14)
    assert si.shape == (D, D)
    assert np.allclose(si, Sigma_inv)
    assert meta["sigma_c_corpus_kind"] == "broad"
    assert meta["sigma_c_headline_eligible"] is True


def test_predict_cell_uses_passed_sigma_inv(tmp_path):
    """When --sigma-inv is provided, predict_cell scores the gate with THAT matrix.

    Two different Σc⁻¹ matrices must give different L̂ columns for the same cell
    (the gate is c_Cᵀ Σc⁻¹ c_{C'} / c_Cᵀ Σc⁻¹ c_C), proving the passed whitening is
    actually used rather than ignored / overwritten by the battery diagnostic.
    """
    cell_dir = tmp_path / "bm_default_contra_d1_seed42"
    _write_fake_cell(cell_dir, behavior="bad_medical", seed=5)
    loaded = loader.load_cell(cell_dir)
    rng = np.random.default_rng(9)
    A = rng.standard_normal((D, D))
    sig_a = np.linalg.inv(A @ A.T + np.eye(D))
    sig_b = np.eye(D)
    rec_a = predscore.predict_cell(loaded, cell="x", layer=LAYER, Sigma_inv=sig_a)
    rec_b = predscore.predict_cell(loaded, cell="x", layer=LAYER, Sigma_inv=sig_b)
    assert rec_a["per_bystander"]["Lhat"] != rec_b["per_bystander"]["Lhat"], (
        "the passed Σc⁻¹ must change the gate term (it was ignored)"
    )


def test_headline_eligible_propagates_to_per_cell_record():
    """The build_headline table carries the broad-corpus Σc metadata (finding 1)."""
    recs = [
        {
            "cell": "bm_default_contra_d1_seed42",
            "behavior": "bad_medical",
            "r_B_source": "diffmeans",
            "rho_full_Lhat": 0.6,
            "rho_cosine": 0.4,
            "rho_base_prior": 0.1,
        }
    ]
    sigma_meta = {"sigma_c_corpus_kind": "broad", "sigma_c_headline_eligible": True}
    hd = predscore.build_headline(recs, sigma_meta=sigma_meta, r_b_source="mixed")
    assert hd["sigma_c"]["sigma_c_corpus_kind"] == "broad"
    assert hd["sigma_c"]["sigma_c_headline_eligible"] is True
    assert "bad_medical" in hd["per_behavior"]


# ---------------------------------------------------------------------------
# Finding 2 — full store enumeration is the no-override production path.
# ---------------------------------------------------------------------------
def test_resolve_cells_full_enumeration_when_no_overrides(monkeypatch):
    """No --cells / --cell-names / --slice → the full HF store enumeration."""
    import importlib

    real = importlib.import_module("issue666_predictor")
    fake_cells = [f"cell_{i}" for i in range(48)]
    monkeypatch.setattr(real, "enumerate_store_cells", lambda: list(fake_cells))

    class _Args:
        cell_names = None
        slice = False
        cells = None

    cells, is_full = predscore._resolve_cells(_Args())
    assert is_full is True
    assert cells == fake_cells


def test_resolve_cells_slice_keeps_smoke_slice():
    """--slice → the tiny _smoke_cells slice, NOT the full enumeration."""

    class _Args:
        cell_names = None
        slice = True
        cells = 1

    cells, is_full = predscore._resolve_cells(_Args())
    assert is_full is False
    assert cells == predscore._smoke_cells(1)


def test_resolve_cells_cell_names_override():
    """--cell-names takes precedence (explicit override)."""

    class _Args:
        cell_names: typing.ClassVar = ["a", "b"]
        slice = False
        cells = None

    cells, is_full = predscore._resolve_cells(_Args())
    assert is_full is False
    assert cells == ["a", "b"]


# ---------------------------------------------------------------------------
# Finding 3 — mixed r_B routing: diffmeans for bad-medical/EM, r_plus for fact/marker.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("behavior", "expected_source", "expected_col"),
    [
        ("bad_medical", "diffmeans", "harmful_compliance"),
        ("em", "diffmeans", "broad_em"),
        ("fact", "r_plus", None),
        ("marker", "r_plus", None),
    ],
)
def test_mixed_rb_routing_per_behavior(tmp_path, behavior, expected_source, expected_col):
    cell_dir = tmp_path / f"{behavior}_cell"
    _write_fake_cell(cell_dir, behavior=behavior, seed=hash(behavior) % 2**31)
    loaded = loader.load_cell(cell_dir)
    rb_cols = _fake_rb_columns()
    rb, source = predscore.rb_for_cell(loaded, LAYER, rb_columns=rb_cols, r_b_source="mixed")
    assert source == expected_source, f"{behavior}: expected r_B source {expected_source}"
    if expected_source == "diffmeans":
        assert np.allclose(rb, rb_cols[expected_col]), f"{behavior}: wrong diffmeans column"
    else:
        # r_plus path: equals the cell's r_plus at the layer.
        assert np.allclose(rb, predscore.cell_r_plus(loaded, LAYER))


def test_rb_source_force_r_plus_for_all(tmp_path):
    """--r-b-source=r_plus forces the r_plus shortcut even for bad-medical (sensitivity arm)."""
    cell_dir = tmp_path / "bm_cell"
    _write_fake_cell(cell_dir, behavior="bad_medical", seed=2)
    loaded = loader.load_cell(cell_dir)
    rb, source = predscore.rb_for_cell(
        loaded, LAYER, rb_columns=_fake_rb_columns(), r_b_source="r_plus"
    )
    assert source == "r_plus"
    assert np.allclose(rb, predscore.cell_r_plus(loaded, LAYER))


def test_load_rb_columns_indexes_layer(tmp_path, monkeypatch):
    """load_rb_columns extracts the per-layer (d,) direction for every column.

    Passes the (already-verified) rb dict in directly (the documented `rb=` path),
    so no real HF read happens — the layer-indexing logic is exercised offline.
    """
    rng = np.random.default_rng(4)
    nl, d = 28, D
    rb = {
        "r_b": {
            "broad_em": torch.from_numpy(rng.standard_normal((nl, d)).astype("float32")),
            "harmful_compliance": torch.from_numpy(rng.standard_normal((nl, d)).astype("float32")),
            "sycophancy": torch.from_numpy(rng.standard_normal((nl, d)).astype("float32")),
            "refusal": torch.from_numpy(rng.standard_normal((nl, d)).astype("float32")),
        },
        "rb_columns": ["broad_em", "harmful_compliance", "sycophancy", "refusal"],
    }
    cols = loader.load_rb_columns(layer=14, rb=rb)
    assert set(cols) == {"broad_em", "harmful_compliance", "sycophancy", "refusal"}
    for name, vec in cols.items():
        assert vec.shape == (d,)
        assert np.allclose(vec, rb["r_b"][name].numpy()[14])


def test_load_rb_columns_handles_real_nested_diffmeans_dict():
    """The REAL #658 r_b.pt nests {'diffmeans','meanDB','n_db','n_dbbar'} per column.

    Regression guard for the library-API-drift caught at smoke time: the real
    artifact's r_b[col] is a dict (the diffmeans direction under the 'diffmeans'
    key), NOT a bare (n_layer, d) tensor. load_rb_columns must extract the
    diffmeans tensor at the layer. Pins the nested-dict path offline (no HF).
    """
    rng = np.random.default_rng(5)
    nl, d = 28, D

    def _col():
        return {
            "diffmeans": torch.from_numpy(rng.standard_normal((nl, d)).astype("float32")),
            "meanDB": torch.from_numpy(rng.standard_normal((nl, d)).astype("float32")),
            "n_db": 100,
            "n_dbbar": 200,
        }

    rb = {
        "r_b": {
            "broad_em": _col(),
            "harmful_compliance": _col(),
            "sycophancy": _col(),
            "refusal": _col(),
        },
        "capture_layers": list(range(nl)),
        "columns": ["broad_em", "harmful_compliance", "sycophancy", "refusal"],
    }
    cols = loader.load_rb_columns(layer=14, rb=rb)
    assert set(cols) == {"broad_em", "harmful_compliance", "sycophancy", "refusal"}
    for name, vec in cols.items():
        assert vec.shape == (d,)
        assert np.allclose(vec, rb["r_b"][name]["diffmeans"].numpy()[14])


def test_load_rb_columns_raises_on_dict_without_diffmeans():
    """A nested column dict missing 'diffmeans' is a fail-loud error (no silent skip)."""
    rb = {"r_b": {"broad_em": {"meanDB": torch.zeros(28, D)}}}
    with pytest.raises(ValueError, match="diffmeans"):
        loader.load_rb_columns(layer=14, rb=rb)


# ---------------------------------------------------------------------------
# Finding 4 — cross-behavior grid emits a per_target row per available behavior.
# ---------------------------------------------------------------------------
def test_predict_cell_grid_emits_every_target_behavior(tmp_path):
    cell_dir = tmp_path / "bm_default_contra_d1_seed42"
    _write_fake_cell(cell_dir, behavior="bad_medical", seed=8)
    loaded = loader.load_cell(cell_dir)
    rb_cols = _fake_rb_columns()
    rec = predscore.predict_cell_grid(
        loaded,
        cell="bm_default_contra_d1_seed42",
        layer=LAYER,
        Sigma_inv=np.eye(D),
        rb_columns=rb_cols,
        r_b_source="mixed",
    )
    # Every grid target behavior is present (diffmeans columns available + r_plus ones).
    assert set(rec["per_target"]) == set(predscore.GRID_TARGET_BEHAVIORS)
    for tb, row in rec["per_target"].items():
        for k in ("rho_full_Lhat", "rho_cosine", "rho_base_prior", "r_B_source"):
            assert k in row, f"per_target[{tb}] missing {k}"
    # The diffmeans targets are tagged diffmeans; fact/marker tagged r_plus.
    assert rec["per_target"]["em"]["r_B_source"] == "diffmeans"
    assert rec["per_target"]["fact"]["r_B_source"] == "r_plus"


def test_predict_cell_grid_omits_diffmeans_target_when_columns_absent(tmp_path):
    """No rb_columns (offline smoke) → diffmeans targets omitted, r_plus ones kept."""
    cell_dir = tmp_path / "tf_default_contra_d1_seed42"
    _write_fake_cell(cell_dir, behavior="fact", seed=1)
    loaded = loader.load_cell(cell_dir)
    rec = predscore.predict_cell_grid(
        loaded,
        cell="tf_default_contra_d1_seed42",
        layer=LAYER,
        Sigma_inv=np.eye(D),
        rb_columns=None,
        r_b_source="mixed",
    )
    # fact + marker (r_plus path) kept; bad_medical + em (diffmeans) omitted.
    assert "fact" in rec["per_target"]
    assert "marker" in rec["per_target"]
    assert "bad_medical" not in rec["per_target"]
    assert "em" not in rec["per_target"]


# ---------------------------------------------------------------------------
# Finding 5 — run_lobo_loco returns BOTH lobo and loco; loco non-empty on ≥2 behaviors.
# ---------------------------------------------------------------------------
def _write_predictor_json(pred_dir: Path, cell: str, behavior: str, *, n=12, seed=0):
    rng = np.random.default_rng(seed)
    lh = rng.standard_normal(n)
    ds = 0.8 * lh + rng.normal(0, 0.3, n)
    rec = {
        "cell": cell,
        "behavior": behavior,
        "r_B_source": "diffmeans" if behavior in ("bad_medical", "em") else "r_plus",
        "rho_full_Lhat": 0.5,
        "rho_cosine": 0.3,
        "rho_base_prior": 0.1,
        "per_bystander": {
            "context_family": [f"f{i % 3}" for i in range(n)],
            "Lhat": lh.round(6).tolist(),
            "cosine": (0.7 * lh).round(6).tolist(),
            "base_prior": rng.standard_normal(n).round(6).tolist(),
            "ds": ds.round(6).tolist(),
        },
    }
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / f"{cell}_predictor_cells.json").write_text(json.dumps(rec))


def test_run_lobo_loco_returns_both_folds(tmp_path):
    pred_dir = tmp_path / "predictor"
    _write_predictor_json(pred_dir, "bm_default_contra_d1_seed42", "bad_medical", seed=1)
    _write_predictor_json(pred_dir, "ic_default_contra_d1_seed42", "em", seed=2)
    _write_predictor_json(pred_dir, "tf_default_contra_d1_seed42", "fact", seed=3)
    summary = cv.run_lobo_loco(pred_dir)
    assert "lobo" in summary
    assert "loco" in summary
    # ≥2 behaviors → LOBO non-empty; ≥2 contexts → LOCO non-empty.
    assert len(summary["lobo"]) >= 2
    assert len(summary["loco"]) >= 2, "LOCO must be non-empty when cells span ≥2 contexts"
    # The shared aggregate helper ran for both fold families.
    assert "lobo_aggregate" in summary and "loco_aggregate" in summary
    assert summary["loco_aggregate"]["n_folds"] == len(summary["loco"])


def test_loco_fold_holds_out_one_context(tmp_path):
    """Each LOCO fold's test rows are exactly the held-out context's rows."""
    pred_dir = tmp_path / "predictor"
    _write_predictor_json(pred_dir, "bm_default_contra_d1_seed42", "bad_medical", n=10, seed=1)
    _write_predictor_json(pred_dir, "ic_default_contra_d1_seed42", "em", n=10, seed=2)
    summary = cv.run_lobo_loco(pred_dir)
    # 10 contexts → 10 LOCO folds; each fold tests the rows at one context index
    # (2 cells share each within-cell bystander index → 2 test rows per fold).
    assert len(summary["loco"]) == 10
    for fold in summary["loco"].values():
        assert fold["n_test"] == 2, "each context index appears once per cell (2 cells)"
