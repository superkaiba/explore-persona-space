# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Σ, ρ, δ, r_B, ×, ⁻¹, ᵀ, Δ) in scientific docstrings + asserts.
"""issue #666 Phase-4 round-3 code-review BLOCKER + CONCERN fixes.

The round-2 ensemble (Claude + Codex) FAILed on 4 substantive blockers + 1
concern in the cross-behavior + designed-null + LOCO pipeline. These tests pin
the corrected semantics:

  Blocker 1 (cross-target-rplus-source-cell-alias): fact/marker cross-behavior
    targets use a CANONICAL target-source cell's r_plus from the shared registry,
    NOT the SOURCE cell's r_plus.
  Blocker 2 (cross-target-cosine-behavior-term-dropped): the cross-behavior cosine
    carries cos(r_{B'}, r_B) with DISTINCT source/target directions, not 1.
  Blocker 3 (loco-row-index-not-context-id): predict_cell persists per-bystander
    context_id; LOCO folds key on it, not the positional row index.
  Blocker 4 (designed-null-arm-uses-battery-sigma-not-broad-corpus): the
    designed-null arm is scored on the SAME broad-corpus Σc as the real arms
    (reads the predictor's broad-Σc per-cell JSONs), with a parity field.
  Concern 5 (rb-source-sensitivity-artifact-missing): build_rb_source_sensitivity
    aggregates per-behavior ρ under both diffmeans and r_plus.

All offline (tmp_path synthetic store cells + synthetic rb_columns); CPU-only,
no network, no GPU.
"""

from __future__ import annotations

import json
import sys
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
dn = _LazyModule("issue666_designed_null")

N_LAYER = 4
D = 32
N_CTX = 12
N_PROBE = 4
LAYER = 2


def _write_cell(
    cell_dir: Path,
    *,
    behavior: str,
    source: str = "default",
    source_idx: int = 0,
    context_ids: list | None = None,
    seed: int = 0,
):
    """Fabricate a #664-schema store cell with explicit context_ids + source_idx.

    The masked bystander array therefore differs across (source_idx, context_ids),
    exactly as SOURCE_INSTANCE_IDS makes it differ in production — the multi-source
    LOCO conflation Blocker 3 fixes.
    """
    rng = np.random.default_rng(seed)
    cell_dir.mkdir(parents=True, exist_ok=True)
    if context_ids is None:
        context_ids = [f"ctx{i}" for i in range(N_CTX)]
    v0 = rng.standard_normal((N_CTX, N_LAYER, D)).astype("float32")
    install = (rng.standard_normal((N_LAYER, D)).astype("float32")) * 3.0
    dv = rng.standard_normal((N_CTX, N_LAYER, D)).astype("float32")
    struct = rng.standard_normal((N_CTX, 1, 1)).astype("float32")
    dv = dv + struct * install[None]
    dv[source_idx] = install
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
        "context_ids": list(context_ids),
    }
    torch.save(obj, cell_dir / "tensors.pt")
    # target_context_roles keyed by context_id (the real-store dict form), so
    # _source_idx resolves the source anchor by its battery id.
    roles = {
        cid: ("source-anchor" if i == source_idx else "bystander")
        for i, cid in enumerate(context_ids)
    }
    (cell_dir / "meta.json").write_text(
        json.dumps(
            {
                "behavior": behavior,
                "source": source,
                "source_idx": source_idx,
                "target_context_roles": roles,
            }
        )
    )


def _rb_cols(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "broad_em": rng.standard_normal(D).astype(np.float64),
        "harmful_compliance": rng.standard_normal(D).astype(np.float64),
        "sycophancy": rng.standard_normal(D).astype(np.float64),
        "refusal": rng.standard_normal(D).astype(np.float64),
    }


# ───────────────────────── Blocker 1 ──────────────────────────────────────────
def test_fact_marker_targets_use_canonical_source_not_source_cell(tmp_path):
    """per_target['fact'/'marker'] on a bad-medical source cell use the CANONICAL
    target-source r_plus (from the registry), NOT the bad-medical source cell's r_plus.

    Pins the round-2 Blocker-1 fix: the cross-behavior fact/marker rows must NOT be
    scored with the source cell's own implant shift (which trivialized the matrix).
    """
    src_dir = tmp_path / "bm_default_contra_d1_seed42"
    _write_cell(src_dir, behavior="bad_medical", seed=8)
    loaded = loader.load_cell(src_dir)
    rb_cols = _rb_cols()

    # Canonical fact/marker directions distinct from the source cell's own r_plus.
    rng = np.random.default_rng(42)
    fact_dir = rng.standard_normal(D)
    marker_dir = rng.standard_normal(D)
    reg = {
        "bad_medical": {
            "r_Bp": rb_cols["harmful_compliance"],
            "source": "diffmeans",
            "from_cell": None,
        },
        "em": {"r_Bp": rb_cols["broad_em"], "source": "diffmeans", "from_cell": None},
        "fact": {
            "r_Bp": fact_dir,
            "source": "r_plus_canonical",
            "from_cell": "tf_default_contra_d1_seed42",
        },
        "marker": {
            "r_Bp": marker_dir,
            "source": "r_plus_canonical",
            "from_cell": "mk_default_contra_d1_seed42",
        },
    }
    rec = predscore.predict_cell_grid(
        loaded,
        cell="bm_default_contra_d1_seed42",
        layer=LAYER,
        Sigma_inv=np.eye(D),
        rb_columns=rb_cols,
        r_b_source="mixed",
        target_registry=reg,
    )

    # The source cell's OWN r_plus (what the buggy code aliased) — score the fact
    # target with it and assert the registry-scored fact row DIFFERS.
    source_r_plus = predscore.cell_r_plus(loaded, LAYER)
    buggy_fact = predscore.predict_cell(
        loaded,
        cell="x",
        layer=LAYER,
        Sigma_inv=np.eye(D),
        r_B=source_r_plus,
        source_r_B=source_r_plus,
    )
    canonical_fact = predscore.predict_cell(
        loaded, cell="x", layer=LAYER, Sigma_inv=np.eye(D), r_B=fact_dir, source_r_B=source_r_plus
    )
    # The fact target row in the grid matches the CANONICAL-direction score, NOT
    # the source-r_plus (buggy) score.
    assert rec["per_target"]["fact"]["rho_full_Lhat"] == pytest.approx(
        canonical_fact["rho_full_Lhat"]
    )
    # And the canonical and buggy scores genuinely differ (the directions differ),
    # so the fix is observable (not vacuously equal).
    assert canonical_fact["rho_full_Lhat"] != pytest.approx(buggy_fact["rho_full_Lhat"]), (
        "canonical-target and source-aliased scores must differ — else the test is vacuous"
    )
    # Provenance: the fact/marker rows trace to the canonical cells.
    assert rec["per_target"]["fact"]["from_cell"] == "tf_default_contra_d1_seed42"
    assert rec["per_target"]["marker"]["from_cell"] == "mk_default_contra_d1_seed42"


def test_build_target_registry_reads_canonical_cells_not_source(tmp_path):
    """build_target_direction_registry resolves fact/marker via the CANONICAL cells.

    Injects stub download/load that returns DISTINCT canonical cells; asserts the
    registry's fact/marker directions come from those canonical cells' r_plus, and
    the diffmeans targets come from rb_columns.
    """
    canon_fact = tmp_path / "tf_default_contra_d1_seed42"
    canon_marker = tmp_path / "mk_default_contra_d1_seed42"
    _write_cell(canon_fact, behavior="fact", seed=101)
    _write_cell(canon_marker, behavior="marker", seed=202)

    def _dl(cell):
        return {
            "tf_default_contra_d1_seed42": canon_fact,
            "mk_default_contra_d1_seed42": canon_marker,
        }[cell]

    reg = predscore.build_target_direction_registry(
        layer=LAYER,
        rb_columns=_rb_cols(),
        download_cell=_dl,
        load_cell=loader.load_cell,
    )
    assert set(reg) == {"bad_medical", "em", "fact", "marker"}
    assert reg["fact"]["source"] == "r_plus_canonical"
    assert reg["fact"]["from_cell"] == "tf_default_contra_d1_seed42"
    # The fact direction equals the canonical fact cell's r_plus at the layer.
    assert np.allclose(
        reg["fact"]["r_Bp"], predscore.cell_r_plus(loader.load_cell(canon_fact), LAYER)
    )
    assert np.allclose(
        reg["marker"]["r_Bp"], predscore.cell_r_plus(loader.load_cell(canon_marker), LAYER)
    )


def test_build_target_registry_omits_unresolvable_canonical(tmp_path):
    """A canonical target cell that fails to resolve is OMITTED, not faked."""

    def _dl(cell):
        raise RuntimeError("canonical cell unavailable in this run")

    reg = predscore.build_target_direction_registry(
        layer=LAYER, rb_columns=_rb_cols(), download_cell=_dl, load_cell=loader.load_cell
    )
    # diffmeans targets still resolve from rb_columns; fact/marker omitted.
    assert "bad_medical" in reg and "em" in reg
    assert "fact" not in reg and "marker" not in reg


# ───────────────────────── Blocker 2 ──────────────────────────────────────────
def test_cross_behavior_cosine_uses_distinct_source_target_directions(tmp_path):
    """Cross-target cosine on a bad-medical source × fact target uses cos(r_{B'}, r_B)
    with DISTINCT directions — NOT the collapsed cos(r_B, r_B) = 1.

    Pins Blocker 2: predict_cell's cosine variant must thread source_r_B distinct
    from r_B (=r_{B'}). With near-orthogonal directions the cosine column must
    differ from the own-behavior cosine (where source==target so cos=1).
    """
    src_dir = tmp_path / "bm_default_contra_d1_seed42"
    _write_cell(src_dir, behavior="bad_medical", seed=8)
    loaded = loader.load_cell(src_dir)
    source_r_B = predscore.cell_r_plus(loaded, LAYER)
    # A target direction orthogonal-ish to the source direction.
    rng = np.random.default_rng(0)
    target_r_Bp = rng.standard_normal(D)

    # Cross-behavior call: source_r_B distinct from r_B (=target_r_Bp).
    cross = predscore.predict_cell(
        loaded,
        cell="x",
        layer=LAYER,
        Sigma_inv=np.eye(D),
        r_B=target_r_Bp,
        source_r_B=source_r_B,
    )
    # Own-behavior call: source_r_B defaults to r_B → cos(r_B, r_B) = 1 inside.
    own = predscore.predict_cell(
        loaded, cell="x", layer=LAYER, Sigma_inv=np.eye(D), r_B=target_r_Bp
    )
    cross_cos = np.array(cross["per_bystander"]["cosine"])
    own_cos = np.array(own["per_bystander"]["cosine"])
    # The cross cosine carries cos(target, source) != 1, so it is a SCALED version
    # of the own cosine (which used cos(target,target)=1). They must differ.
    assert not np.allclose(cross_cos, own_cos), (
        "cross-behavior cosine must include cos(r_{B'}, r_B) != 1 (Blocker 2)"
    )
    # Concretely: cross = own * cos(target, source) (the behavior term cos(r_{B'},r_B)
    # scales the gate-only own cosine). Compare element-wise WITHOUT division to
    # avoid amplifying near-zero rows.
    cos_scalar = float(target_r_Bp @ source_r_B) / (
        np.linalg.norm(target_r_Bp) * np.linalg.norm(source_r_B)
    )
    assert np.allclose(cross_cos, own_cos * cos_scalar, atol=2e-6), (
        "cross cosine must equal own cosine scaled by the behavior-transfer cos(r_{B'}, r_B)"
    )
    # And the scalar is genuinely not 1 (the directions are distinct).
    assert abs(cos_scalar) < 0.95


# ───────────────────────── Blocker 3 ──────────────────────────────────────────
def test_predict_cell_persists_context_id(tmp_path):
    """predict_cell writes per_bystander.context_id = the masked battery ids."""
    cids = [f"f1_house_{n}" for n in ("librarian", "surgeon", "programmer")] + [
        f"f6_x{i}" for i in range(N_CTX - 3)
    ]
    src_dir = tmp_path / "cell"
    _write_cell(src_dir, behavior="bad_medical", source_idx=0, context_ids=cids, seed=5)
    loaded = loader.load_cell(src_dir)
    rec = predscore.predict_cell(loaded, cell="cell", layer=LAYER, Sigma_inv=np.eye(D))
    got = rec["per_bystander"]["context_id"]
    # Source anchor (index 0) is masked out; the rest carry their battery ids.
    assert got == cids[1:]
    assert len(got) == rec["n_bystanders"]


def test_loco_folds_key_on_context_id_across_sources(tmp_path):
    """LOCO folds key on the battery context_id — a multi-source cell set whose
    masked bystander arrays DIFFER across sources is folded by IDENTITY, not by
    positional row index (Blocker 3).

    Two cells with DIFFERENT source anchors (so position i is a DIFFERENT battery
    context across them). A positional fold would put unrelated contexts in one
    fold; the id fold groups the SAME context across cells.
    """
    pred_dir = tmp_path / "predictor"
    pred_dir.mkdir()
    # Shared battery: ctx0..ctx5. Cell A's source anchor is ctx0; cell B's is ctx3.
    battery = [f"ctx{i}" for i in range(6)]

    def _write_pred_json(name, behavior, src_anchor, seed):
        rng = np.random.default_rng(seed)
        bystanders = [c for c in battery if c != src_anchor]
        lh = rng.standard_normal(len(bystanders))
        ds = 0.8 * lh + rng.normal(0, 0.2, len(bystanders))
        rec = {
            "cell": name,
            "behavior": behavior,
            "r_B_source": "diffmeans",
            "rho_full_Lhat": 0.5,
            "rho_cosine": 0.3,
            "rho_base_prior": 0.1,
            "per_bystander": {
                "context_id": bystanders,
                "context_family": [c.split("_")[0] for c in bystanders],
                "Lhat": lh.round(6).tolist(),
                "cosine": (0.7 * lh).round(6).tolist(),
                "base_prior": rng.standard_normal(len(bystanders)).round(6).tolist(),
                "ds": ds.round(6).tolist(),
            },
        }
        (pred_dir / f"{name}_predictor_cells.json").write_text(json.dumps(rec))

    _write_pred_json("cellA", "bad_medical", "ctx0", 1)  # anchor ctx0 -> bystanders ctx1..5
    _write_pred_json("cellB", "em", "ctx3", 2)  # anchor ctx3 -> bystanders ctx0,1,2,4,5

    summary = cv.run_lobo_loco(pred_dir)
    # The union of bystander contexts across the two cells is ctx0..ctx5 (6 folds).
    assert set(summary["loco"]) == set(battery)
    assert len(summary["loco"]) == 6
    # ctx0 is a bystander ONLY in cellB (it is cellA's source anchor) → 1 test row.
    assert summary["loco"]["ctx0"]["test_context"] == "ctx0"
    assert summary["loco"]["ctx0"]["n_test"] == 1
    # ctx3 is a bystander ONLY in cellA → 1 test row.
    assert summary["loco"]["ctx3"]["n_test"] == 1
    # ctx1 is a bystander in BOTH cells → 2 test rows (the same battery context,
    # correctly grouped — a positional fold would have split it).
    assert summary["loco"]["ctx1"]["n_test"] == 2


# ───────────────────────── Blocker 4 ──────────────────────────────────────────
def _write_broad_pred_json(pred_dir: Path, cell: str, *, corpus_kind: str, seed: int):
    rng = np.random.default_rng(seed)
    n = 8
    lh = rng.standard_normal(n)
    ds = 0.3 * lh + rng.normal(0, 0.5, n)
    rec = {
        "cell": cell,
        "behavior": "ic_edu" if cell.startswith("ic") else "tf_rev",
        "layer": 14,
        "r_B_source": "r_plus",
        "rho_full_Lhat": 0.1,
        "rho_cosine": 0.05,
        "rho_base_prior": 0.02,
        "n_bystanders": n,
        "sigma_c_corpus_kind": corpus_kind,
        "sigma_c_headline_eligible": True,
        "per_bystander": {
            "context_id": [f"ctx{i}" for i in range(n)],
            "context_family": [f"f{i % 3}" for i in range(n)],
            "Lhat": lh.round(6).tolist(),
            "cosine": (0.5 * lh).round(6).tolist(),
            "base_prior": rng.standard_normal(n).round(6).tolist(),
            "ds": ds.round(6).tolist(),
        },
    }
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / f"{cell}_predictor_cells.json").write_text(json.dumps(rec))


def test_designed_null_reads_broad_corpus_predictor_jsons(tmp_path):
    """The designed-null arm reads the predictor's BROAD-Σc per-cell JSONs (Blocker 4)."""
    pred_dir = tmp_path / "predictor"
    for cell in predscore.DESIGNED_NULL_CELLS:
        _write_broad_pred_json(pred_dir, cell, corpus_kind="broad", seed=hash(cell) % 1000)
    out = dn.score_designed_nulls(pred_dir=pred_dir, n_boot=100)
    assert set(out) == set(predscore.DESIGNED_NULL_CELLS)
    for cell, r in out.items():
        assert r["sigma_c_corpus_kind"] == "broad", f"{cell} must be scored on broad Σc"
        assert np.isfinite(r["ci_lo"]) and np.isfinite(r["ci_hi"])


def test_designed_null_fails_loud_on_battery_sigma(tmp_path):
    """A null cell predictor JSON with battery Σc must FAIL LOUD (no confounded gate)."""
    pred_dir = tmp_path / "predictor"
    import issue666_predictor as pred

    for cell in pred.DESIGNED_NULL_CELLS:
        _write_broad_pred_json(pred_dir, cell, corpus_kind="battery-diagnostic", seed=1)
    with pytest.raises(SystemExit, match="broad"):
        dn.score_designed_nulls(pred_dir=pred_dir, n_boot=100)


def test_designed_null_fails_loud_on_missing_predictor_json(tmp_path):
    """An absent null cell predictor JSON must FAIL LOUD (run the predictor first)."""
    pred_dir = tmp_path / "predictor"
    pred_dir.mkdir()
    with pytest.raises(SystemExit, match="predictor JSON"):
        dn.score_designed_nulls(pred_dir=pred_dir, n_boot=100)


def test_designed_null_sigma_parity_with_headline(tmp_path):
    """designed_null_Lhat_rho.json's sigma_c_corpus_kind matches the headline's.

    Both must read 'broad' on production — the §6 install-leak gate compares real
    and null ρ on IDENTICAL whitening. Builds matching broad-Σc null + headline
    records and asserts the corpus-kind fields agree.
    """
    import issue666_predictor as pred

    pred_dir = tmp_path / "predictor"
    for cell in pred.DESIGNED_NULL_CELLS:
        _write_broad_pred_json(pred_dir, cell, corpus_kind="broad", seed=2)
    nulls = dn.score_designed_nulls(pred_dir=pred_dir, n_boot=100)
    null_kinds = {r["sigma_c_corpus_kind"] for r in nulls.values()}
    assert null_kinds == {"broad"}

    # The headline JSON's sigma_c.sigma_c_corpus_kind on a broad-Σc run.
    headline = pred.build_headline(
        [
            {
                "cell": "bm_x",
                "behavior": "bad_medical",
                "r_B_source": "diffmeans",
                "rho_full_Lhat": 0.5,
                "rho_cosine": 0.3,
                "rho_base_prior": 0.1,
                "per_target": {},
            }
        ],
        sigma_meta={"sigma_c_corpus_kind": "broad", "sigma_c_headline_eligible": True},
        r_b_source="mixed",
    )
    assert headline["sigma_c"]["sigma_c_corpus_kind"] == next(iter(null_kinds))


# ───────────────────────── Concern 5 ──────────────────────────────────────────
def test_rb_source_sensitivity_carries_both_modes_both_behaviors():
    """build_rb_source_sensitivity carries diffmeans + r_plus ρ for bad_medical + em."""
    recs_dm = [
        {"behavior": "bad_medical", "rho_full_Lhat": 0.60},
        {"behavior": "em", "rho_full_Lhat": 0.45},
    ]
    recs_rp = [
        {"behavior": "bad_medical", "rho_full_Lhat": 0.50},
        {"behavior": "em", "rho_full_Lhat": 0.40},
    ]
    sens = predscore.build_rb_source_sensitivity({"diffmeans": recs_dm, "r_plus": recs_rp})
    assert set(sens["per_behavior"]) == {"bad_medical", "em"}
    assert sens["per_behavior"]["bad_medical"]["rho_full_Lhat_diffmeans"] == pytest.approx(0.60)
    assert sens["per_behavior"]["bad_medical"]["rho_full_Lhat_r_plus"] == pytest.approx(0.50)
    assert sens["per_behavior"]["bad_medical"]["delta_diffmeans_minus_r_plus"] == pytest.approx(
        0.10
    )
    assert sens["per_behavior"]["em"]["delta_diffmeans_minus_r_plus"] == pytest.approx(0.05)
    assert set(sens["modes"]) == {"diffmeans", "r_plus"}


def test_headline_references_sensitivity_artifact():
    """The headline JSON carries the rb_source_sensitivity artifact path when produced."""
    hd = predscore.build_headline(
        [
            {
                "cell": "c",
                "behavior": "bad_medical",
                "r_B_source": "diffmeans",
                "rho_full_Lhat": 0.5,
                "rho_cosine": 0.3,
                "rho_base_prior": 0.1,
                "per_target": {},
            }
        ],
        sigma_meta={"sigma_c_corpus_kind": "broad", "sigma_c_headline_eligible": True},
        r_b_source="mixed",
        sensitivity_artifact="headline/rb_source_sensitivity.json",
    )
    assert hd["rb_source_sensitivity_artifact"] == "headline/rb_source_sensitivity.json"


def test_headline_carries_cross_behavior_matrix_and_registry():
    """The headline JSON aggregates the corrected cross-behavior off-diagonal matrix."""
    recs = [
        {
            "cell": "bm_x",
            "behavior": "bad_medical",
            "r_B_source": "diffmeans",
            "rho_full_Lhat": 0.5,
            "rho_cosine": 0.3,
            "rho_base_prior": 0.1,
            "per_target": {
                "bad_medical": {
                    "rho_full_Lhat": 0.5,
                    "rho_cosine": 0.99,
                    "rho_base_prior": 0.1,
                    "r_B_source": "diffmeans",
                    "from_cell": None,
                },
                "fact": {
                    "rho_full_Lhat": 0.2,
                    "rho_cosine": 0.15,
                    "rho_base_prior": 0.05,
                    "r_B_source": "r_plus_canonical",
                    "from_cell": "tf_default_contra_d1_seed42",
                },
            },
        }
    ]
    reg = {
        "fact": {"source": "r_plus_canonical", "from_cell": "tf_default_contra_d1_seed42"},
        "bad_medical": {"source": "diffmeans", "from_cell": None},
    }
    hd = predscore.build_headline(
        recs, sigma_meta={"sigma_c_corpus_kind": "broad"}, r_b_source="mixed", target_registry=reg
    )
    # off-diagonal = fact only (bad_medical==source behavior is on-diagonal, excluded).
    assert hd["cross_behavior_matrix"]["n_off_diagonal_cells"] == 1
    assert hd["cross_behavior_matrix"]["rho_full_Lhat_mean"] == pytest.approx(0.2)
    assert hd["target_direction_registry"]["fact"]["from_cell"] == "tf_default_contra_d1_seed42"
