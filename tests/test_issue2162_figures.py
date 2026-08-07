"""Issue #2162 figure producers — synthetic-input smokes to ``savefig``.

Each test drives a REAL figure function end-to-end on tiny synthetic inputs
(matplotlib Agg) and asserts the PNG + ``.meta.json`` sidecar landed. The
hero test routes a deliberately INVERTED bootstrap CI through the errorbar
path (the xerr/yerr non-negative-offsets gotcha — offsets are clamped at the
call site via ``_err``, so an inverted CI must render, never raise).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2162_figures as F  # noqa: E402

from explore_persona_space.experiments.issue2162 import bank2162 as B  # noqa: E402


def _cell_rec(cell: str, slot: str, family: str = "P1", **kw) -> dict:
    rec = {
        "cell": cell,
        "slot": slot,
        "family": family,
        "n_pre_exclusion": 36,
        "n_post_exclusion": 20,
        "untestable_causal": False,
        "f_steered_mean": 0.5,
        "f_shuffled_mean": 0.1,
        "f_crosstype_mean": 0.05,
        "ci95": {
            "steered": [0.4, 0.6],
            "shuffled": [0.0, 0.2],
            "crosstype": [-0.05, 0.15],
        },
    }
    rec.update(kw)
    return rec


def _assert_saved(out_dir: Path, name: str) -> None:
    png = out_dir / f"{name}.png"
    assert png.exists() and png.stat().st_size > 0, name
    meta = json.loads((out_dir / f"{name}.meta.json").read_text())
    assert meta["figure"] == name and meta["inputs"]


def test_fig_hero_renders_inverted_ci(tmp_path):
    """An INVERTED quantile CI (lo > v > hi — legitimate at tiny bootstrap n)
    must render through the clamped errorbar path, never raise ValueError."""
    stats = {
        "per_cell": {
            "instr_format|ce": _cell_rec(
                "instr_format",
                "ce",
                ci95={"steered": [0.9, 0.2], "shuffled": [0.3, 0.05], "crosstype": [0.2, -0.1]},
            ),
            "verbosity|pe": _cell_rec("verbosity", "pe", untestable_causal=True),
        }
    }
    f_cells = [
        {"cell": "instr_format", "slot": "ce", "f_beh": 0.4, "pair_id": "p1"},
        {"cell": "verbosity", "slot": "pe", "f_beh": None, "pair_id": "p2"},
    ]
    F.fig_hero(stats, f_cells, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "hero_ftype")
    _assert_saved(tmp_path, "hero_ftype_perpair")


def test_fig_dose_position_with_slopes_and_null(tmp_path):
    rec_cell = next(c for c in B.crossed_cells() if c.startswith("recency_"))
    base = B.base_type_of(rec_cell)
    tag = "d"
    per_cell = {
        f"{base}|ce": _cell_rec(base, "ce", family="P1"),
        f"{rec_cell}|ce": _cell_rec(rec_cell, "ce", family="P3", f_steered_mean=0.35),
        f"{base.replace(base, rec_cell).rsplit(tag, 1)[0]}{tag}5|ce": _cell_rec(
            f"{rec_cell.rsplit(tag, 1)[0]}{tag}5", "ce", family="P3", f_steered_mean=0.2
        ),
    }
    stats = {
        "per_cell": per_cell,
        "dose_slopes": {
            f"recency|{base}|ce": {
                "n_pairs": 9,
                "slope_mean": -0.07,
                "ci95": [-0.12, -0.02],
                "unit": "Delta F_beh per level step",
            }
        },
    }
    F.fig_dose_position(stats, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "dose_position")


def test_fig_probe_layer_curves_with_band_and_vps(tmp_path):
    curve = np.linspace(0.5, 0.9, 8).tolist()
    probe = {
        "results": [
            {
                "cell": "instr_format",
                "slot": slot,
                "auc_per_layer": curve,
                "auc_per_layer_per_vp": [curve, (np.asarray(curve) - 0.05).tolist()],
                "value_pairs": [["v1", "v2"], ["v2", "v3"]],
            }
            for slot in ("ce", "pe")
        ]
    }
    npz = tmp_path / "perm_auc_matrix.npz"
    rng = np.random.default_rng(0)
    np.savez(
        npz,
        **{f"instr_format|{slot}": rng.uniform(0.35, 0.65, size=(50, 8)) for slot in ("ce", "pe")},
    )
    F.fig_layer_profile(probe, npz, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "probe_layer_curves_ce")
    _assert_saved(tmp_path, "probe_layer_curves_pe")
    _assert_saved(tmp_path, "layer_profile")


def test_fig_diagnostics_excess_incoherence(tmp_path):
    stats = {"per_cell": {"instr_format|ce": _cell_rec("instr_format", "ce")}}
    mk = lambda coh, cap: [  # noqa: E731
        {
            "cell": "instr_format",
            "slot": "ce",
            "n_coherent": coh,
            "n_cap_hit": cap,
            "n_draws": 10,
        }
    ]
    arm_rows = {"steered": mk(7, 1), "shuffled": mk(9, 0), "crosstype": mk(8, 0)}
    anchors = [
        {
            "cell": "instr_format",
            "carrier": "c1",
            "value_a": "v1",
            "value_b": "v2",
            "pair_id": "p1",
            "separation": 0.8,
            "n_floor_rollouts": 10,
            "n_floor_coherent": 9,
            "n_ceiling_rollouts": 10,
            "n_ceiling_coherent": 10,
        }
    ]
    F.fig_diagnostics(stats, arm_rows, anchors, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "diagnostics")


def test_fig_anchor_separation(tmp_path):
    anchors = [
        {"cell": "instr_format", "pair_id": f"p{i}", "separation": s}
        for i, s in enumerate((0.9, 0.4, -0.6, None))
    ]
    F.fig_anchor_separation(anchors, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "anchor_separation_diag")


def test_fig_act_beh_agreement(tmp_path):
    rows = [
        {
            "cell": f"cell{i}",
            "slot": "ce",
            "arm": "steered",
            "f_beh": 0.1 * i,
            "f_act": 0.08 * i,
            "separation": 0.7,
        }
        for i in range(6)
    ]
    arm_rows = {"steered": rows, "shuffled": [], "crosstype": []}
    F.fig_act_beh_agreement(arm_rows, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "act_beh_agreement")


def test_fig_route_contrasts_perpair(tmp_path):
    f_cells = [
        {"cell": c, "slot": "ce", "f_beh": 0.3 + 0.1 * i, "pair_id": f"p{i}"}
        for i, c in enumerate(("instr_format", "demo_format", "instr_format"))
    ]
    F.fig_route_contrasts_perpair(f_cells, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "route_contrasts_perpair")


def test_fig_recency_load_perpair(tmp_path):
    rec_cell = next(c for c in B.crossed_cells() if c.startswith("recency_"))
    base = B.base_type_of(rec_cell)
    f_cells = [
        {
            "cell": cell,
            "slot": "ce",
            "carrier": "car1",
            "value_a": "v1",
            "value_b": "v2",
            "f_beh": f,
            "pair_id": f"p{i}",
        }
        for i, (cell, f) in enumerate(((base, 0.5), (rec_cell, 0.3)))
    ]
    F.fig_recency_load_perpair(f_cells, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "recency_load_perpair")


def test_fig_crosstype_by_donor(tmp_path):
    per_cell = {
        "instr_format|ce": _cell_rec(
            "instr_format",
            "ce",
            ci95={"steered": [0.4, 0.6], "shuffled": [0.0, 0.2], "crosstype": [0.05, 0.2]},
        ),
        "refusal_boundary|ce": _cell_rec(
            "refusal_boundary",
            "ce",
            ci95={"steered": [0.4, 0.6], "shuffled": [0.02, 0.2], "crosstype": [-0.1, 0.1]},
        ),
    }
    null_crosstype = [
        {
            "cell": "instr_format",
            "slot": "ce",
            "f_beh": 0.15,
            "separation": 0.8,
            "donor_cell": "verbosity",
        },
        {
            "cell": "instr_format",
            "slot": "ce",
            "f_beh": 0.1,
            "separation": 0.8,
            "donor_cell": "persona_prompted",
        },
    ]
    null_shuffled = [
        {
            "cell": "refusal_boundary",
            "slot": "ce",
            "f_beh": 0.12,
            "separation": 0.9,
            "donor_value_b": "vB1",
        },
        {
            "cell": "refusal_boundary",
            "slot": "ce",
            "f_beh": 0.05,
            "separation": 0.9,
            "donor_value_b": "vB2",
        },
    ]
    F.fig_crosstype_by_donor(
        {"per_cell": per_cell}, null_crosstype, null_shuffled, tmp_path, [Path("x")]
    )
    _assert_saved(tmp_path, "crosstype_null_by_donor")


def test_fig_stage2_layer_profile(tmp_path):
    rows = []
    for unit_cell in ("instr_format", "verbosity"):
        for arm in ("steered", "shuffled"):
            for layer in (8, 26):
                for dose in (1, 4):
                    for pi in range(2):
                        rows.append(
                            {
                                "cell": unit_cell,
                                "slot": "ce",
                                "arm": arm,
                                "layer": layer,
                                "dose": dose,
                                "pair_id": f"{unit_cell}-p{pi}",
                                "f_beh": (0.6 if arm == "steered" else 0.1) + 0.05 * pi,
                                "separation": 0.8,
                            }
                        )
    F.fig_stage2_layer_profile(rows, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "layer_profile_stage2")
    _assert_saved(tmp_path, "layer_profile_stage2_perpair")


def test_fig_stage2_layer_profile_requires_kept_rows(tmp_path):
    with pytest.raises(AssertionError, match="separation-kept"):
        F.fig_stage2_layer_profile(
            [
                {
                    "cell": "c",
                    "slot": "ce",
                    "arm": "steered",
                    "layer": 8,
                    "dose": 1,
                    "pair_id": "p",
                    "f_beh": None,
                    "separation": None,
                }
            ],
            tmp_path,
            [Path("x")],
        )
