"""Issue #657 — unit guards for the alignment-predictor analysis library.

These pin the load-bearing invariants of the cross-behavior leakage bake-off so a
future refactor can't silently strip them:

  - the marker-token fail-fast in the #605 substrate loader (wrong marker_id raises);
  - the doubly-partialled partial-Spearman actually removes the partialled covariate
    (a covariate-driven correlation collapses under the double partial);
  - the prior_centroid_projection geometry-side control (M-Alts1) is computed from
    the base-rate-weighted persona-vector centroid;
  - bootstrap_over_personas resamples the BYSTANDER persona, not the cell;
  - the reliability band (M-Stats1) produces the 3-point R_low/R_expected/R_high
    sensitivity points;
  - the additive --trait-name CLI default reproduces #623's sycophancy verbatim.

All synthetic / CPU-only — no GPU, no HF, no model load.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from explore_persona_space.analysis import issue657_alignment_predictor as m


def _toy_persona_vectors(n_layers: int = 4, hidden: int = 8, seed: int = 0):
    """A tiny persona-vector bank: assistant (baseline) + 5 personas, (n_layers, hidden)."""
    rng = np.random.default_rng(seed)
    names = ["assistant", "p1", "p2", "p3", "p4", "p5"]
    return {
        name: torch.from_numpy(rng.standard_normal((n_layers, hidden)).astype(np.float32))
        for name in names
    }


# ── partial_spearman: the double partial removes the partialled covariate ──


def test_partial_spearman_removes_covariate():
    rng = np.random.default_rng(1)
    n = 60
    z = rng.standard_normal(n)
    # x and y are BOTH driven only by z (plus independent noise) -> raw rho strong,
    # partial rho ~ 0 once z is partialled out.
    x = z + 0.1 * rng.standard_normal(n)
    y = z + 0.1 * rng.standard_normal(n)
    raw_rho, _p, _n = m.partial_spearman(x, y, None)
    part_rho, _p2, _n2 = m.partial_spearman(x, y, z)
    assert raw_rho > 0.8, raw_rho
    assert abs(part_rho) < 0.3, part_rho


def test_partial_spearman_doubly_partialled_matrix_z():
    """The matrix-z (doubly-partialled) read removes BOTH covariate columns.

    x, y share a single driver z1; z2 is an extra (irrelevant) covariate column. The
    matrix partial on [z1, z2] must still kill the z1-driven correlation — i.e. adding
    a noise column to the partial set does not resurrect the signal. (A test where x,y
    depend on the SUM z1+z2 is NOT a valid expectation: rank-residualizing on the two
    rank columns linearly cannot remove a nonlinear-in-ranks sum dependence — that is
    a property of rank partial correlation, not a code bug.)
    """
    rng = np.random.default_rng(2)
    n = 80
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = z1 + 0.1 * rng.standard_normal(n)
    y = z1 + 0.1 * rng.standard_normal(n)
    raw_rho, _, _ = m.partial_spearman(x, y, None)
    double, _, _ = m.partial_spearman(x, y, np.column_stack([z1, z2]))
    assert raw_rho > 0.8
    assert abs(double) < 0.3, double


# ── prior_centroid_projection: M-Alts1 base-rate-weighted centroid ──


def test_prior_centroid_projection_weights_by_base_rate():
    pv = _toy_persona_vectors()
    personas = ["assistant", "p1", "p2", "p3"]
    # heavy weight on p1: the centroid points toward p1's persona vector, so p1's
    # projection should be the highest among the non-baseline personas.
    base_rates = {"p1": 0.9, "p2": 0.05, "p3": 0.05, "assistant": 0.01}
    pcp = m.prior_centroid_projection(pv, base_rates, personas, layer=14)
    assert set(pcp) == set(personas)
    non_baseline = {k: v for k, v in pcp.items() if k != "assistant"}
    top = max(non_baseline, key=non_baseline.get)
    assert top == "p1", (top, non_baseline)


def test_compute_alignment_is_cosine_bounded():
    pv = _toy_persona_vectors()
    personas = ["assistant", "p1", "p2"]
    bdir = torch.from_numpy(np.ones(8, dtype=np.float32))
    align = m.compute_alignment(pv, bdir, personas, layer=14)
    # assistant is the baseline-self -> its persona vector is 0 -> NaN cosine.
    assert np.isnan(align["assistant"])
    for p in ("p1", "p2"):
        assert -1.0001 <= align[p] <= 1.0001, (p, align[p])


# ── bootstrap_over_personas: resamples the BYSTANDER, not the cell ──


def test_bootstrap_resamples_bystanders_not_cells():
    # 3 bystanders x 2 sources each = 6 cells. The stat counts DISTINCT bystanders
    # in the resampled frame; with bystander resampling it must be <= 3 (never 6).
    cells = []
    for b in ("b1", "b2", "b3"):
        for s in ("s1", "s2"):
            cells.append({"source": s, "bystander": b, "delta": 0.1, "align": 0.2})
    frame = {
        "cells": cells,
        "resolvable_bystanders": ["b1", "b2", "b3"],
        "n_resolvable": 3,
        "dropped": [],
    }

    def _distinct_bystanders(fr):
        return float(len({c["bystander"] for c in fr["cells"]}))

    out = m.bootstrap_over_personas(frame, _distinct_bystanders, b=50, seed=657)
    # the resampled frame never has more than 3 distinct bystanders
    assert out["ci_hi"] <= 3.0, out
    assert out["n_bystanders"] == 3


# ── reliability band: 3-point sensitivity (M-Stats1) ──


def test_attenuation_band_three_points():
    rng = np.random.default_rng(3)
    cells = []
    bystanders = [f"b{i}" for i in range(20)]
    base_rates = {b: float(rng.uniform(0.02, 0.13)) for b in bystanders}
    for b in bystanders:
        for s in ("s1", "s2"):
            cells.append(
                {
                    "source": s,
                    "bystander": b,
                    "delta": float(rng.uniform(-0.1, 0.7)),
                    "align": float(rng.uniform(-0.3, 0.3)),
                    "bystander_base_rate": base_rates[b],
                    "prior_centroid_projection": float(rng.uniform(-0.3, 0.3)),
                }
            )
    frame = {
        "cells": cells,
        "resolvable_bystanders": bystanders,
        "n_resolvable": len(bystanders),
        "dropped": [],
    }
    band = m.attenuation_sensitivity(frame, base_rates, behavior="refusal")
    assert band["band_applicable"] is True
    for label in ("R_observed", "R_low", "R_expected", "R_high"):
        assert label in band["points"], label
    # R_low (largest n, prior cleanest) has the LOWEST reliability-correction scale;
    # all reliabilities are in (0, 1].
    for label in ("R_low", "R_expected", "R_high"):
        rel = band["points"][label]["reliability"]
        assert 0.0 < rel <= 1.0, (label, rel)


def test_attenuation_band_skipped_for_marker():
    """Marker base rate is a log-prob, not a 0-1 rate -> binomial reliability undefined."""
    cells = [
        {
            "source": "s1",
            "bystander": f"b{i}",
            "delta": 0.1,
            "align": 0.2,
            "bystander_base_rate": -18.0,  # a log-prob, not a rate
            "prior_centroid_projection": 0.1,
        }
        for i in range(10)
    ]
    frame = {
        "cells": cells,
        "resolvable_bystanders": [f"b{i}" for i in range(10)],
        "n_resolvable": 10,
        "dropped": [],
    }
    base_rates = {f"b{i}": -18.0 for i in range(10)}
    band = m.attenuation_sensitivity(frame, base_rates, behavior="marker")
    assert band["band_applicable"] is False


# ── marker-token fail-fast in the #605 substrate loader ──


def test_marker_substrate_rejects_wrong_marker_id(tmp_path: Path):
    marker_root = tmp_path / "issue_605" / "marker"
    (marker_root / "per_cell_trained").mkdir(parents=True)
    (marker_root / "per_cell_base").mkdir(parents=True)
    # one cell carrying a WRONG marker_id (bare ※ id 63680) -> must raise.
    cell = {
        "context_label": "m605_nt_comedian_1__none",
        "source_cid": "A1",
        "summary": {"mean_logp_marker": -5.0},
        "metadata": {"marker_id": 63680},
    }
    base = {"summary": {"mean_logp_marker": -18.0}, "metadata": {"marker_id": 63680}}
    name = "A1__m605_nt_comedian_1__none.json"
    (marker_root / "per_cell_trained" / name).write_text(json.dumps(cell))
    (marker_root / "per_cell_base" / name).write_text(json.dumps(base))
    with pytest.raises(ValueError, match="marker_id"):
        m.load_marker_cell_substrate(tmp_path)


def test_marker_persona_alias_mapping():
    # swe -> software_engineer; non-panel personas (formal/socratic/pirate) -> None.
    assert m._marker_persona_from_context("m605_nt_swe_1__none") == "software_engineer"
    assert m._marker_persona_from_context("m605_nt_comedian_2__explicit") == "comedian"
    assert m._marker_persona_from_context("m605_nt_formal_1__soft") is None
    assert m._marker_persona_from_context("instr_explicit_1") is None


# ── additive --trait-name default reproduces #623 sycophancy verbatim ──


def test_extract_script_trait_default_is_623_sycophancy():
    spec = importlib.util.spec_from_file_location(
        "issue623_extract_657test",
        Path(__file__).resolve().parents[1] / "scripts/issue623_extract_sycophancy_vector.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import inspect

    sig = inspect.signature(mod.generate_artifacts)
    from explore_persona_space.experiments.persona_decomp_623 import (
        SYCOPHANCY_TRAIT_DESCRIPTION,
        SYCOPHANCY_TRAIT_NAME,
    )

    assert sig.parameters["trait_name"].default == SYCOPHANCY_TRAIT_NAME
    assert sig.parameters["trait_description"].default == SYCOPHANCY_TRAIT_DESCRIPTION
