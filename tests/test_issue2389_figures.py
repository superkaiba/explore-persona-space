"""Issue #2389 figures fork — fork-delta unit tests.

Covers the deltas vs the parent ``issue2329_figures.py``:

- the 64-layer stack constants (comparison row L59 full-attention, L61
  exploratory companion, every-4th full-attention set);
- the fold-6 ce-only Holm harmonization helpers (``_holm_pass``,
  ``_ce_harmonized_causal``, ``_ce_committed_causal``) on synthetic stats;
- render smokes to ``tmp_path`` (Agg backend) for the four NEW #2389
  manifests: ``fig_transfer``, ``fig_three_model``, ``fig_fact_profile``,
  ``fig_cap_regime`` — asserting a non-trivial PNG + a meta sidecar carrying
  the relocated sidecar facts.

All fixtures are synthetic/tmp — no committed eval_results reads (no
sparse-cone additions needed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2389_figures as F  # noqa: E402

# ------------------------------------------------------------- constants


def test_stack_constants():
    assert F.N_MODEL_LAYERS == 64
    assert F.F_ACT_READ_LAYER == 59
    assert F.COMPANION_F_ACT_LAYER == 61
    assert frozenset(range(3, 64, 4)) == F.FULL_ATTENTION_LAYERS
    # the pre-registered comparison row is itself full-attention; the
    # exploratory companion is linear
    assert F.F_ACT_READ_LAYER in F.FULL_ATTENTION_LAYERS
    assert F.COMPANION_F_ACT_LAYER not in F.FULL_ATTENTION_LAYERS
    assert F.HOLM_ALPHA == 0.05


# ------------------------------------------------------------- Holm helpers


def test_holm_pass_step_down():
    # m=3: adjusted p = running max of (m-i) * p_(i)
    out = F._holm_pass({"a": 0.01, "b": 0.02, "c": 0.9})
    assert out == {"a": True, "b": True, "c": False}
    # running-max monotonicity: a small later p cannot rescue past a blocker
    out2 = F._holm_pass({"a": 0.04, "b": 0.03})
    assert out2 == {"a": False, "b": False}  # 2*0.03=0.06 blocks both


def _stats_row(cell, slot="ce", family="P1", untestable=False, p=None, disjoint=True, holm=None):
    r = {
        "cell": cell,
        "slot": slot,
        "family": family,
        "untestable_causal": untestable,
        "disjoint_both_nulls": disjoint,
    }
    if p is not None:
        r["p_iut"] = p
    if holm is not None:
        r["holm_pass"] = holm
    return r


def test_ce_harmonized_causal_verdicts():
    stats = {
        "per_cell": {
            "c1|ce": _stats_row("c1", p=0.001, disjoint=True),
            "c2|ce": _stats_row("c2", p=0.9, disjoint=True),
            "c3|ce": _stats_row("c3", p=0.001, disjoint=False),  # holm pass, nulls not disjoint
            "c4|ce": _stats_row("c4", untestable=True),
            "c5|pe": _stats_row("c5", slot="pe", p=0.001),  # pe rows are ignored
        }
    }
    v = F._ce_harmonized_causal(stats)
    assert v == {"c1": "positive", "c2": "null", "c3": "null", "c4": "untestable-causal"}


def test_ce_harmonized_causal_fails_loud_on_missing_p_iut():
    stats = {"per_cell": {"c1|ce": _stats_row("c1")}}  # testable, no p_iut
    with pytest.raises(AssertionError, match="has no p_iut"):
        F._ce_harmonized_causal(stats)


def test_ce_committed_causal_uses_run_holm():
    stats = {
        "per_cell": {
            "c1|ce": _stats_row("c1", holm=True, disjoint=True),
            "c2|ce": _stats_row("c2", holm=True, disjoint=False),
            "c3|ce": _stats_row("c3", holm=False, disjoint=True),
            "c4|ce": _stats_row("c4", untestable=True),
        }
    }
    v = F._ce_committed_causal(stats)
    assert v == {"c1": "positive", "c2": "null", "c3": "null", "c4": "untestable-causal"}


# ------------------------------------------------------------ render smokes


def _assert_rendered(out_dir: Path, name: str) -> dict:
    png = out_dir / f"{name}.png"
    meta = out_dir / f"{name}.meta.json"
    assert png.is_file() and png.stat().st_size > 5_000, png
    assert meta.is_file(), meta
    return json.loads(meta.read_text())


def _transfer_unit(cell, x, y, eligible=True, div9=0):
    return {
        "cell": cell,
        "f_beh_2162_mean": x,
        "f_beh_2389_mean": y,
        "primary_eligible": eligible,
        "n_pairs_dropped_div9": div9,
        "repaired_div9": False,
    }


def test_fig_transfer_renders(tmp_path):
    transfer = {
        "per_unit": [
            _transfer_unit("alpha", 0.2, 0.3),
            _transfer_unit("beta", 0.5, 0.4, div9=2),
            _transfer_unit("gamma", 0.1, 0.05, eligible=False),
        ],
        "primary": {
            "n_units": 2,
            "rho": 0.8,
            "p": 0.01,
            "ci95_pair_clustered": [0.4, 0.95],
            "verdict": "positive",
            "rho_ref": 0.5,
            "eligibility_floor": 12,
        },
        "descriptive_all_shared": {
            "n_units": 3,
            "rho": 0.7,
            "p": 0.05,
            "ci95_pair_clustered": None,
        },
        "criterion": "test",
        "div9_flags": {"beta": 2},
    }
    F.fig_transfer(transfer, tmp_path, [Path("in.json")])
    meta = _assert_rendered(tmp_path, "transfer_scatter")
    assert meta["transfer_stats"]["primary"]["verdict"] == "positive"
    assert meta["inputs"] == ["in.json"]


def test_fig_three_model_renders(tmp_path):
    def _harmonizable_stats():
        return {
            "per_cell": {
                "c1|ce": _stats_row("c1", p=0.001, disjoint=True),
                "c2|ce": _stats_row("c2", p=0.9),
                "c3|ce": _stats_row("c3", untestable=True),
            }
        }

    child_stats = {
        "per_cell": {
            "c1|ce": _stats_row("c1", holm=True, disjoint=True),
            "c2|ce": _stats_row("c2", holm=False),
            "c3|ce": _stats_row("c3", untestable=True),
        }
    }

    def _two(pos_cells):
        return {
            "cells": [
                {"cell": c, "slot": "ce", "probe_verdict": "positive" if c in pos_cells else "null"}
                for c in ("c1", "c2", "c3")
            ]
        }

    F.fig_three_model(
        child_stats,
        _two({"c1"}),
        (_harmonizable_stats(), _two({"c1", "c2"})),
        (_harmonizable_stats(), _two({"c1"})),
        tmp_path,
        [Path("a.json")],
    )
    meta = _assert_rendered(tmp_path, "three_model_comparison")
    counts = meta["stored_and_used_counts"]
    # every model: c1 causal-positive AND probe-positive => count 1 each
    assert list(counts.values()) == [1, 1, 1]
    assert "harmonization" in meta


def test_fig_fact_profile_renders_with_read_companion_sidecar(tmp_path):
    layers = list(range(F.N_MODEL_LAYERS))
    rows = [
        {
            "slot": "ce",
            "cell": c,
            "layers": layers,
            "f_act_mean_per_layer": [0.01 * i + off for i in range(F.N_MODEL_LAYERS)],
            "n_pairs": 12,
        }
        for off, c in ((0.0, "alpha"), (0.1, "beta"))
    ] + [
        {
            "slot": "pe",
            "cell": "ghost",
            "layers": layers,
            "f_act_mean_per_layer": [0.0] * F.N_MODEL_LAYERS,
            "n_pairs": 12,
        }
    ]
    F.fig_fact_profile(rows, tmp_path, [Path("fact_profile.jsonl")])
    meta = _assert_rendered(tmp_path, "fact_profile")
    assert meta["n_cells"] == 2  # pe row excluded
    pc = meta["per_cell_read_companion"]
    assert pytest.approx(pc["alpha"]["f_act_read_mean"]) == 0.01 * F.F_ACT_READ_LAYER
    assert pytest.approx(pc["alpha"]["f_act_companion_mean"]) == 0.01 * F.COMPANION_F_ACT_LAYER


def test_fig_fact_profile_refuses_empty_ce(tmp_path):
    with pytest.raises(AssertionError, match="no ce rows"):
        F.fig_fact_profile([], tmp_path, [])


def test_fig_cap_regime_renders(tmp_path):
    cap_report = {
        "scope": "anchors",
        "partial": False,
        "pre_registered_regen_trigger_pct": 2.0,
        "breaching_cells": ["alpha"],
        "per_cell": {
            "alpha": {
                "cap_hit_pct": 5.0,
                "breach": True,
                "realized_caps_by_batch": {"gate": [4096], "rest": [4096]},
            },
            "beta": {
                "cap_hit_pct": 0.5,
                "breach": False,
                "realized_caps_by_batch": {"rest": [2048]},
            },
        },
    }
    F.fig_cap_regime(cap_report, tmp_path, [Path("cap.json")])
    meta = _assert_rendered(tmp_path, "cap_regime")
    assert meta["breaching_cells"] == ["alpha"]
    assert meta["trigger_pct"] == 2.0
    assert meta["per_cell"]["alpha"]["breach"] is True


def test_fig_layer_profile_renders_ce_only(tmp_path):
    # B10 (r1 review): valid #2389 probe output is ce-only (pe dropped by user
    # ruling) — the inherited heatmap built a fixed two-panel figure and passed
    # an empty array to imshow for the absent pe slot, crashing P8. Panels must
    # be built for REALIZED slots only.
    n_layers = 8
    probe = {
        "results": [
            {
                "slot": "ce",
                "cell": cell,
                "auc_per_layer": [0.5 + 0.01 * i + off for i in range(n_layers)],
                "auc_per_layer_per_vp": [[0.5] * n_layers],
            }
            for off, cell in ((0.0, "alpha"), (0.05, "beta"))
        ]
    }
    F.fig_layer_profile(probe, tmp_path / "absent_perm.npz", tmp_path, [Path("probe.json")])
    _assert_rendered(tmp_path, "layer_profile")
    _assert_rendered(tmp_path, "probe_layer_curves_ce")
    # no pe panel artifacts for an unrealized slot
    assert not (tmp_path / "probe_layer_curves_pe.png").exists()
