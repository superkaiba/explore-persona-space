"""CPU unit tests for the issue #685 `assistant`-excluded recompute.

Builds a tiny synthetic centroid bank, runs the recompute helpers, and asserts:
  1. The mean pairwise cosine over a 4-context subset matches a hand-computed
     value, and differs from the 5-context value (the metric is sensitive to
     dropping a context — the whole point of the follow-up).
  2. `behavior_shift_metrics` over the 9-context subset (one context dropped from
     a synthetic 10-context bank) reports `n_context == 9` and never references
     the excluded context.
  3. `_band_placement` maps the documented threshold regions correctly.

No GPU, no model load, no HF download — pure tensor math against the production
metric module (`behavior_shift_metrics`) the recompute script reuses.
"""

import importlib.util
import math
from pathlib import Path

import torch

from explore_persona_space.analysis.issue685.metrics import (
    mean_pairwise_cosine,
)

# Load the recompute script as a module (it lives under scripts/, not the package).
_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "issue685_assistant_excluded_recompute.py"
)
_spec = importlib.util.spec_from_file_location("issue685_recompute", _SCRIPT)
_recompute = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_recompute)


def test_mean_pairwise_cosine_4_vs_5_contexts():
    """A hand-checkable 5-vector bank: dropping the odd-one-out raises the cosine.

    Rows 0-3 are near-parallel (small perturbations off e0); row 4 points along a
    different axis. Over all 5 rows the mean pairwise cosine is dragged down by
    the 4 pairs involving the outlier; over the 4 parallel rows it is high.
    """
    H = 8
    base = torch.zeros(H)
    base[0] = 1.0
    rows = []
    for k in range(4):
        v = base.clone()
        v[1 + k] = 0.05  # tiny near-parallel perturbation
        rows.append(v)
    outlier = torch.zeros(H)
    outlier[7] = 1.0  # orthogonal-ish to the e0 cluster
    five = torch.stack([*rows, outlier])
    four = torch.stack(rows)

    cos_5 = mean_pairwise_cosine(five)
    cos_4 = mean_pairwise_cosine(four)

    # The 4-parallel-rows cosine is high (near 1); the 5-row cosine is pulled down.
    assert cos_4 > 0.99, cos_4
    assert cos_5 < cos_4, (cos_5, cos_4)

    # Hand-check the 4-row value: all four rows are e0 + 0.05*e_{1+k}, so each has
    # norm sqrt(1 + 0.0025) and each pair shares the e0 component (dot = 1.0).
    norm_sq = 1.0 + 0.05**2
    expected_pair_cos = 1.0 / norm_sq  # dot=1, |a||b| = norm_sq
    assert math.isclose(cos_4, expected_pair_cos, rel_tol=1e-5), (cos_4, expected_pair_cos)


def _synthetic_by_condition(context_names, behaviors, layers, hidden_dim=16):
    """Build a `{condition: {layer: (H,) vec}}` bank with a dominant shared shift.

    For each behavior the shift `Delta(C, b) = v(C+b) - v(C)` is a shared per-behavior
    direction + a small per-context residual, so the consistency cosine is high —
    matching the real experiment's regime and exercising the full metric path.
    """
    g = torch.Generator().manual_seed(7)
    by_condition: dict[str, dict[int, torch.Tensor]] = {}
    # bare vectors: one random vector per context per layer.
    bare: dict[str, dict[int, torch.Tensor]] = {}
    for c in context_names:
        bare[c] = {}
        by_condition[f"bare__{c}"] = {}
        for layer in layers:
            v = torch.randn(hidden_dim, generator=g)
            bare[c][layer] = v
            by_condition[f"bare__{c}"][layer] = v
    # augmented = bare + shared per-(behavior, layer) direction + tiny residual.
    for b in behaviors:
        for layer in layers:
            shared = torch.randn(hidden_dim, generator=g) * 3.0
            for c in context_names:
                resid = torch.randn(hidden_dim, generator=g) * 0.2
                by_condition[f"{c}__{b}"] = by_condition.get(f"{c}__{b}", {})
                by_condition[f"{c}__{b}"][layer] = bare[c][layer] + shared + resid
    return by_condition


def test_metrics_for_contexts_drops_excluded():
    """The 9-context recompute reports n_context==9 and omits the excluded context."""
    context_names = [
        "assistant",
        "software_engineer",
        "villain",
        "kindergarten_teacher",
        "medical_doctor",
        "librarian",
        "french_person",
        "police_officer",
        "comedian",
        "data_scientist",
    ]
    behaviors = ["sycophancy", "refusal"]
    layers = [7, 14]
    by_condition = _synthetic_by_condition(context_names, behaviors, layers)

    nine = [c for c in context_names if c != "assistant"]
    metrics = _recompute._metrics_for_contexts(
        by_condition, nine, behaviors, layers, known_dirs=None, null_n_perm=10
    )
    assert metrics["meta"]["n_context"] == 9
    assert "assistant" not in metrics["meta"]["context_names"]
    # Shared-direction construction -> high raw cosine in every cell.
    for b in behaviors:
        for layer in layers:
            cell = metrics["cells"][b][str(layer)]
            assert cell["n_context"] == 9
            assert cell["consistency_cosine_raw"] > 0.5, (b, layer, cell)
            # relative_magnitude must be recomputed over the 9-context spread.
            assert len(cell["relative_magnitude"]["per_context"]) == 9


def test_full_vs_subset_relative_magnitude_uses_own_spread():
    """relmag denominator (median bank spread) is recomputed per context subset.

    Dropping a context changes the bare-bank median pairwise distance, so the
    10- and 9-context relmag means generally differ — confirming the script does
    not reuse the stale 10-context spread for the 9-context number.
    """
    context_names = [f"c{i}" for i in range(10)]
    behaviors = ["b0"]
    layers = [7]
    by_condition = _synthetic_by_condition(context_names, behaviors, layers)

    ten = _recompute._metrics_for_contexts(
        by_condition, context_names, behaviors, layers, known_dirs=None, null_n_perm=5
    )
    nine = _recompute._metrics_for_contexts(
        by_condition, context_names[1:], behaviors, layers, known_dirs=None, null_n_perm=5
    )
    spread_10 = ten["cells"]["b0"]["7"]["relative_magnitude"]["median_spread"]
    spread_9 = nine["cells"]["b0"]["7"]["relative_magnitude"]["median_spread"]
    # Different bank -> different median spread (not a copied constant).
    assert spread_10 != spread_9


def test_band_placement_thresholds():
    """`_band_placement` matches the clean-result's documented band thresholds."""
    # single-direction: cos > 0.6 AND pc1 > 0.5
    assert _recompute._band_placement(0.79, 0.82, 1.4) == "single-direction"
    # context-dependent: cos < 0.4 AND pc1 < 0.4 AND mag >= 0.2
    assert _recompute._band_placement(0.30, 0.35, 0.5) == "context-dependent"
    # negligible: mag < 0.2 (regardless of cos/pc1)
    assert _recompute._band_placement(0.70, 0.70, 0.1) == "negligible"
    # intermediate: clears no band (e.g. cos in (0.4, 0.6))
    assert _recompute._band_placement(0.50, 0.45, 0.9) == "intermediate"
