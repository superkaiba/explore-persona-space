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
    must render through the clamped errorbar path, never raise ValueError;
    both slots render as separate panels (r2 R3) and the per-pair companion
    takes the three-arm row dict (r2 R1)."""
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
    arm_rows = {
        "steered": [
            {
                "cell": "instr_format",
                "slot": "ce",
                "f_beh": 0.4,
                "pair_id": "p1",
                "separation": 0.8,
            },
            {"cell": "verbosity", "slot": "pe", "f_beh": None, "pair_id": "p2", "separation": 0.8},
        ],
        "shuffled": [
            {
                "cell": "instr_format",
                "slot": "ce",
                "f_beh": 0.1,
                "pair_id": "p1",
                "separation": 0.8,
            },
        ],
        "crosstype": [],
    }
    F.fig_hero(stats, arm_rows, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "hero_ftype")
    _assert_saved(tmp_path, "hero_ftype_perpair")


def test_perpair_surviving_applies_separation_exclusion():
    """r2 R1: the per-pair companion uses the SAME |separation| >= 0.5
    exclusion as the hero aggregation, keeps points PER ARM, and carries the
    pair id with every surviving point."""
    arm_rows = {
        "steered": [
            {"cell": "c", "slot": "ce", "f_beh": 0.5, "pair_id": "keep", "separation": 0.9},
            {"cell": "c", "slot": "ce", "f_beh": 0.4, "pair_id": "drop_sep", "separation": 0.2},
            {"cell": "c", "slot": "ce", "f_beh": 0.3, "pair_id": "drop_nosep", "separation": None},
            {"cell": "c", "slot": "ce", "f_beh": None, "pair_id": "drop_nof", "separation": 0.9},
        ],
        "shuffled": [
            {"cell": "c", "slot": "ce", "f_beh": 0.1, "pair_id": "keep", "separation": -0.7},
        ],
        "crosstype": [],
    }
    pts = F._perpair_surviving(arm_rows)
    assert pts[("c", "ce", "steered")] == [("keep", 0.5)]
    assert pts[("c", "ce", "shuffled")] == [("keep", 0.1)]  # abs(-0.7) >= bar
    assert ("c", "ce", "crosstype") not in pts


def test_separation_bar_pinned_to_analysis_module():
    """figures.SEPARATION_BAR is a local copy (keeps torch/scipy out of the
    figure script) — this pin is what keeps it equal to the analysis bar."""
    import issue2162_analysis as A

    assert F.SEPARATION_BAR == A.SEPARATION_BAR


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


def test_fig_diagnostics_missing_anchor_baseline_is_nan_not_zero(tmp_path, caplog):
    """r2 H3: a cell whose anchors.jsonl rows predate the n_*_rollouts fields
    has NO incoherence baseline — the excess renders NaN with a loud warning,
    never a silent 0.0 substitute (which fakes 'no excess over baseline')."""
    import logging

    stats = {"per_cell": {"instr_format|ce": _cell_rec("instr_format", "ce")}}
    arm_rows = {
        "steered": [
            {"cell": "instr_format", "slot": "ce", "n_coherent": 7, "n_cap_hit": 1, "n_draws": 10}
        ],
        "shuffled": [],
        "crosstype": [],
    }
    # Legacy anchor row: NO n_*_rollouts / n_*_coherent fields at all.
    anchors = [
        {"cell": "instr_format", "carrier": "c1", "value_a": "v1", "value_b": "v2", "pair_id": "p1"}
    ]
    with caplog.at_level(logging.WARNING, logger="issue2162.figures"):
        F.fig_diagnostics(stats, arm_rows, anchors, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "diagnostics")
    assert any("no anchor incoherence baseline" in rec.message for rec in caplog.records)


def test_fig_diagnostics_absent_coherent_count_skips_context(tmp_path, caplog):
    """r4 MINOR 1 (round 5): an anchor side with n_*_rollouts present but
    n_*_coherent ABSENT must NOT read as "0 coherent of N" (a fabricated
    maximally-incoherent baseline) — the context is skipped, so a cell with
    no other valid contexts routes into the r2 H3 missing-baseline NaN +
    loud-warning path instead of a silent 1.0 baseline."""
    import logging

    stats = {"per_cell": {"instr_format|ce": _cell_rec("instr_format", "ce")}}
    arm_rows = {
        "steered": [
            {"cell": "instr_format", "slot": "ce", "n_coherent": 7, "n_cap_hit": 1, "n_draws": 10}
        ],
        "shuffled": [],
        "crosstype": [],
    }
    # Rollout count present, coherent count ABSENT (no ceiling fields at all).
    anchors = [
        {
            "cell": "instr_format",
            "carrier": "c1",
            "value_a": "v1",
            "value_b": "v2",
            "pair_id": "p1",
            "n_floor_rollouts": 10,
        }
    ]
    with caplog.at_level(logging.WARNING, logger="issue2162.figures"):
        F.fig_diagnostics(stats, arm_rows, anchors, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "diagnostics")
    assert any("no anchor incoherence baseline" in rec.message for rec in caplog.records)


def test_fig_margin_validation_percell_grain(tmp_path):
    """The margin-validation figure plots the REGISTERED per-cell grain from
    margin_validation.json's percell_points + the per-pair companion (r2 R2)."""
    margin_cells = [
        {"pair_id": f"p{i}", "cell": "c", "slot": "ce", "arm": "steered", "margin_shift": 0.1 * i}
        for i in range(4)
    ]
    f_cells = [{"pair_id": f"p{i}", "slot": "ce", "f_beh": 0.2 * i} for i in range(4)]
    validation = {
        "rho_margin_fbeh_percell": 0.9,
        "n_cells": 12,
        "rho_margin_fbeh_perpair": 0.7,
        "n_pairs": 40,
        "validated": True,
        "percell_points": [
            {"cell": "c", "slot": "ce", "margin_shift_mean": 0.15, "f_beh_mean": 0.3, "n_pairs": 4}
        ],
    }
    F.fig_margin_validation(margin_cells, f_cells, validation, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "margin_validation")


def _two_rec(cell: str, causal: str, probe: str, auc: float | None, f: float | None = 0.4) -> dict:
    return {
        "cell": cell,
        "slot": "ce",
        "causal_verdict": causal,
        "probe_verdict": probe,
        "f_steered_mean": f,
        "max_auc": auc,
        "n_post_exclusion": 20,
    }


def test_fig_two_by_two_quadrant_labels(tmp_path, monkeypatch):
    """r3 MAJOR 1: quadrant membership comes from the PERSISTED
    (probe_verdict, causal_verdict) pair — the four registered manifest
    labels, untestable-causal as the explicit fifth — and the render
    consumes probe_verdict (no causal-only styling, no chance axvline
    standing in for the per-cell probe threshold). r4 MAJOR 1 (round 5):
    a row with f_steered_mean=None (zero surviving steered pairs) is
    OMITTED from the scatter — never plotted at a fabricated y=0.0 — and
    counted in the title beside the existing no-probe-rows count."""
    # The four registered label strings (manifest read_write_2x2
    # plotted_quantity), pinned verbatim.
    assert set(F.QUADRANT_LABELS.values()) == {
        "stored-and-used",
        "stored-but-unusable",
        "used-but-not-decoded",
        "absent",
    }
    assert set(F.QUADRANT_STYLE) == set(F.QUADRANT_LABELS.values()) | {"untestable-causal"}
    # Verdict-pair -> quadrant classification (probe = read, causal = write).
    assert F._quadrant_of(_two_rec("c", "positive", "positive", 0.9)) == "stored-and-used"
    assert F._quadrant_of(_two_rec("c", "null", "positive", 0.9)) == "stored-but-unusable"
    assert F._quadrant_of(_two_rec("c", "positive", "null", 0.6)) == "used-but-not-decoded"
    assert F._quadrant_of(_two_rec("c", "null", "null", 0.5)) == "absent"
    assert F._quadrant_of(_two_rec("c", "untestable-causal", "positive", 0.9)) == (
        "untestable-causal"
    )
    assert F._quadrant_of(_two_rec("c", "null", "missing", None)) is None
    two = {
        "cells": [
            _two_rec("c1", "positive", "positive", 0.9),
            _two_rec("c2", "null", "positive", 0.85),
            _two_rec("c3", "positive", "null", 0.55),
            _two_rec("c4", "null", "null", 0.5),
            _two_rec("c5", "untestable-causal", "positive", 0.8, f=None),
            _two_rec("c6", "null", "missing", None),  # no probe rows -> omitted + counted
        ]
    }
    captured: dict = {}
    real_save = F._save

    def _spy_save(fig, out_dir, name, inputs):
        ax = fig.axes[0]
        captured["title"] = ax.get_title()
        captured["ys"] = [
            float(y) for coll in ax.collections for _x, y in coll.get_offsets().tolist()
        ]
        real_save(fig, out_dir, name, inputs)

    monkeypatch.setattr(F, "_save", _spy_save)
    F.fig_two_by_two(two, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "two_by_two")
    # c5 (untestable-causal, f_steered_mean=None) is excluded from the scatter:
    # only c1-c4's genuinely-measured y=0.4 points render — no fabricated 0.0
    # at the zero-effect coordinate — and the title counts BOTH omission
    # classes (c6 no-probe-rows, c5 no-steered-F).
    assert captured["ys"] == [0.4] * 4
    assert "1 cells without probe rows omitted" in captured["title"]
    assert "1 untestable cells without steered F omitted" in captured["title"]


def test_fig_anchor_separation(tmp_path):
    anchors = [
        {"cell": "instr_format", "pair_id": f"p{i}", "separation": s}
        for i, s in enumerate((0.9, 0.4, -0.6, None))
    ]
    F.fig_anchor_separation(anchors, tmp_path, [Path("x")])
    _assert_saved(tmp_path, "anchor_separation_diag")


def test_fig_act_beh_agreement(tmp_path):
    """r3 MINOR 1: the per-arm rho is restricted to units passing the
    rule-19-mirrored dynamic-range screen (>=2 separation-kept rows with
    spread in BOTH quantities); screened-out units stay plotted but carry
    no rho weight, and the realized range is stated in-panel."""
    rows = []
    for i in range(6):
        # Two rows per unit with spread in both quantities -> passes screen.
        rows.append(
            {
                "cell": f"cell{i}",
                "slot": "ce",
                "arm": "steered",
                "f_beh": 0.1 * i,
                "f_act": 0.08 * i,
                "separation": 0.7,
            }
        )
        rows.append(
            {
                "cell": f"cell{i}",
                "slot": "ce",
                "arm": "steered",
                "f_beh": 0.1 * i + 0.05,
                "f_act": 0.08 * i + 0.03,
                "separation": 0.7,
            }
        )
    # Degenerate units: constant F_beh across rows (no dynamic range) and a
    # single-row unit — both plotted, both screened OUT of the rho.
    rows += [
        {
            "cell": "flat",
            "slot": "ce",
            "arm": "steered",
            "f_beh": 0.2,
            "f_act": a,
            "separation": 0.7,
        }
        for a in (0.1, 0.4)
    ]
    rows.append(
        {
            "cell": "solo",
            "slot": "ce",
            "arm": "steered",
            "f_beh": 0.3,
            "f_act": 0.2,
            "separation": 0.7,
        }
    )
    units = F._act_beh_units(rows)
    assert units["flat|ce"]["in_rho"] is False
    assert units["solo|ce"]["in_rho"] is False
    assert all(units[f"cell{i}|ce"]["in_rho"] for i in range(6))
    # Separation-excluded rows never form units at all.
    assert "gone|ce" not in F._act_beh_units(
        [{"cell": "gone", "slot": "ce", "f_beh": 0.1, "f_act": 0.1, "separation": 0.1}]
    )
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
