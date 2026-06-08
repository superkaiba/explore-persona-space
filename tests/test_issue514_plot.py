"""Unit tests for the #514 source_self_trajectory sidecar loader (B8 round-3 fix).

Plan §6.3 + brief B8: the dispatcher writes ``dynamics_snapshots_path`` (a
string path to a sidecar JSON). The round-2 plot read ``dynamics_snapshots``
(an inline list) — wrong key. Every FT trajectory silently degraded to
endpoint-only markers.

These tests cover:
  (a) eval JSON with ``dynamics_snapshots_path`` populated → plot loads the
      sidecar and emits a real trajectory (≥2 line segments per cell).
  (b) eval JSON with neither key populated → plot emits the endpoint-only
      fallback with no exception.
  (c) eval JSON pointing at a non-existent sidecar → fallback path (no
      raise, no degenerate plot).
  (d) sidecar with a single snapshot → fallback to endpoint marker (a
      one-point trajectory is not a trajectory).
  (e) loader normalization: handles dict-of-snapshots and list-of-snapshots
      payload shapes both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `scripts/` importable so we can pull plot_issue_514 helpers.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import plot_issue_514  # noqa: E402  (import after sys.path edit)

# ── _load_dynamics_snapshots ─────────────────────────────────────────────────


def test_load_dynamics_snapshots_from_sidecar_dict_shape(tmp_path: Path):
    """Sidecar with the canonical {snapshots: {str_step: snap, ...}} shape.

    Matches ``extract_fullft_dynamics_from_checkpoints`` output (the producer
    in marker_dynamics_callback.py): the payload wraps the per-step snapshot
    dict under a ``snapshots`` key, each value carrying the namespaced
    ``dynamics/source_delta_g`` flat metric.
    """
    sidecar = tmp_path / "dynamics.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "i508_dynamics_v1",
                "extraction_mode": "offline_post_checkpoint",
                "n_probes": 20,
                "snapshots": {
                    "10": {
                        "dynamics/source_delta_g": 4.2,
                        "dynamics/bystander_mean_delta_g": 0.1,
                        "step": 10,
                        "n_probes": 20,
                    },
                    "20": {
                        "dynamics/source_delta_g": 8.5,
                        "dynamics/bystander_mean_delta_g": 0.3,
                        "step": 20,
                        "n_probes": 20,
                    },
                    "30": {
                        "dynamics/source_delta_g": 9.7,
                        "dynamics/bystander_mean_delta_g": 0.5,
                        "step": 30,
                        "n_probes": 20,
                    },
                },
            }
        )
    )
    ej = {"dynamics_snapshots_path": str(sidecar)}
    snaps = plot_issue_514._load_dynamics_snapshots(ej)
    assert len(snaps) == 3
    # Sorted by int(step) ascending.
    assert [s["step"] for s in snaps] == [10, 20, 30]
    # The flat namespaced metric is preserved.
    assert snaps[0]["dynamics/source_delta_g"] == 4.2
    assert snaps[2]["dynamics/source_delta_g"] == 9.7


def test_load_dynamics_snapshots_missing_path_returns_empty():
    """eval JSON with NEITHER dynamics_snapshots_path NOR dynamics_snapshots → []."""
    snaps = plot_issue_514._load_dynamics_snapshots({})
    assert snaps == []
    # Round-2 bug shape: ``dynamics_snapshots`` field unpopulated.
    snaps2 = plot_issue_514._load_dynamics_snapshots({"dynamics_snapshots": None})
    assert snaps2 == []


def test_load_dynamics_snapshots_path_nonexistent_returns_empty(tmp_path: Path):
    """Stamped path that doesn't exist on disk → [] (no exception)."""
    ej = {"dynamics_snapshots_path": str(tmp_path / "does_not_exist.json")}
    assert plot_issue_514._load_dynamics_snapshots(ej) == []


def test_load_dynamics_snapshots_malformed_json_returns_empty(tmp_path: Path):
    """Sidecar with corrupt JSON → [] (no exception)."""
    sidecar = tmp_path / "dynamics.json"
    sidecar.write_text("not a json {{{")
    ej = {"dynamics_snapshots_path": str(sidecar)}
    assert plot_issue_514._load_dynamics_snapshots(ej) == []


def test_load_dynamics_snapshots_accepts_inline_list_for_backcompat():
    """Inline ``dynamics_snapshots`` list (e.g. future LoRA arm) still works."""
    inline = [
        {"step": 5, "dynamics/source_delta_g": 1.0},
        {"step": 10, "dynamics/source_delta_g": 5.0},
    ]
    ej = {"dynamics_snapshots": inline}
    snaps = plot_issue_514._load_dynamics_snapshots(ej)
    assert snaps == inline


def test_load_dynamics_snapshots_sidecar_priority_over_inline(tmp_path: Path):
    """When BOTH dynamics_snapshots_path and inline dynamics_snapshots are
    set, the SIDECAR wins (it's the canonical producer artifact).
    """
    sidecar = tmp_path / "dynamics.json"
    sidecar.write_text(
        json.dumps({"snapshots": {"1": {"step": 1, "dynamics/source_delta_g": 99.0}}})
    )
    ej = {
        "dynamics_snapshots_path": str(sidecar),
        "dynamics_snapshots": [{"step": 1, "dynamics/source_delta_g": 0.0}],
    }
    snaps = plot_issue_514._load_dynamics_snapshots(ej)
    assert len(snaps) == 1
    assert snaps[0]["dynamics/source_delta_g"] == 99.0


# ── _snap_x / _snap_y key normalization ──────────────────────────────────────


def test_snap_y_reads_namespaced_dynamics_key():
    """Canonical sidecar key is ``dynamics/source_delta_g``."""
    snap = {"step": 10, "dynamics/source_delta_g": 7.5}
    assert plot_issue_514._snap_y(snap) == 7.5


def test_snap_y_reads_bare_source_delta_g():
    """Bare key ``source_delta_g`` is also accepted (matches #508 analyze.py
    normalizer's tolerance for both shapes).
    """
    snap = {"step": 10, "source_delta_g": 3.3}
    assert plot_issue_514._snap_y(snap) == 3.3


def test_snap_y_missing_returns_nan():
    """No ΔG key on the snapshot → NaN."""
    import math

    snap = {"step": 10}
    assert math.isnan(plot_issue_514._snap_y(snap))


def test_snap_x_reads_step():
    """Canonical x-axis is ``step`` (global training step)."""
    snap = {"step": 42, "dynamics/source_delta_g": 5.0}
    assert plot_issue_514._snap_x(snap, default_idx=0) == 42.0


def test_snap_x_falls_back_to_default_idx():
    """Snapshot with neither step nor epoch_fraction → default_idx."""
    snap = {"dynamics/source_delta_g": 5.0}
    assert plot_issue_514._snap_x(snap, default_idx=7) == 7.0


# ── source_self_trajectory_figure end-to-end ─────────────────────────────────


def test_source_endpoint_y_returns_source_self_not_held_out():
    """B12 round-4 pivot: _source_endpoint_y returns source_self_mean_delta_g.

    The endpoint y-coordinate on the source-self trajectory plot MUST be the
    source-self ΔG (the y-axis of that plot), NOT the held-out ΔG. The
    round-3 endpoint fallback used ``_, y = _cell_xy(ej)``, which is the
    held-out aggregate (the second tuple element). Verifies the new helper
    returns the FIRST element + numerically matches source_self_mean_delta_g.
    """
    ej_cell = {
        "aggregates": {
            "source_self_mean_delta_g": 8.5,
            "held_out_mean_delta_g": -2.3,
        }
    }
    y = plot_issue_514._source_endpoint_y(ej_cell)
    assert y == 8.5
    # And explicitly: it is NOT held-out (the round-3 bug value).
    assert y != -2.3


def test_source_self_trajectory_endpoint_yvalue_is_source_self(tmp_path: Path, monkeypatch):
    """B12 round-4 pivot: the endpoint-fallback scatter call uses
    source_self_mean_delta_g for its y-coordinate, NOT held_out_mean_delta_g.

    Captures the (x, y) args passed to ``ax.scatter`` inside
    ``source_self_trajectory_figure`` and asserts the y matches the source
    aggregate (8.5), NOT the held-out aggregate (-2.3).
    """
    import matplotlib.axes

    captured: list[tuple[float, float]] = []

    real_scatter = matplotlib.axes.Axes.scatter

    def _capturing_scatter(self, x, y, *args, **kwargs):
        # ax.scatter(1.0, y_end, label=..., s=50) — record (x, y).
        captured.append((float(x), float(y)))
        return real_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _capturing_scatter)

    # Cell with NO sidecar → forces the endpoint-fallback branch.
    ej_cell = {
        "aggregates": {
            "source_self_mean_delta_g": 8.5,  # MUST appear as y on the scatter
            "held_out_mean_delta_g": -2.3,  # MUST NOT appear as y on the scatter
        },
    }
    out_path = tmp_path / "traj_endpoint_axis.png"
    plot_issue_514.source_self_trajectory_figure(
        ft_514_cells={"ft_dense_b30": ej_cell},
        output_path=out_path,
    )
    assert out_path.with_suffix(".png").exists()
    # At least one scatter call with y == source_self (8.5); none with y == held_out (-2.3).
    assert any(abs(y - 8.5) < 1e-9 for _x, y in captured), (
        f"expected a scatter call with y=8.5 (source_self), got: {captured}"
    )
    assert not any(abs(y - (-2.3)) < 1e-9 for _x, y in captured), (
        f"endpoint y MUST NOT be -2.3 (held_out); got: {captured}"
    )


def test_source_self_trajectory_single_snapshot_endpoint_axis(tmp_path: Path, monkeypatch):
    """B12 round-4 pivot: same axis check for the single-snapshot fallback
    path (the `len(pairs) < 2` branch). Asserts the y-coord is
    source_self_mean_delta_g, not held_out_mean_delta_g.
    """
    import matplotlib.axes

    captured: list[tuple[float, float]] = []
    real_scatter = matplotlib.axes.Axes.scatter

    def _capturing_scatter(self, x, y, *args, **kwargs):
        captured.append((float(x), float(y)))
        return real_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _capturing_scatter)

    # Sidecar with a SINGLE snapshot → `len(pairs) < 2` → endpoint fallback.
    sidecar = tmp_path / "one_snap.json"
    sidecar.write_text(
        json.dumps({"snapshots": {"10": {"step": 10, "dynamics/source_delta_g": 4.0}}})
    )
    ej_cell = {
        "dynamics_snapshots_path": str(sidecar),
        "aggregates": {
            "source_self_mean_delta_g": 4.0,
            "held_out_mean_delta_g": -1.7,
        },
    }
    out_path = tmp_path / "traj_single_snap_axis.png"
    plot_issue_514.source_self_trajectory_figure(
        ft_514_cells={"ft_dense_b30": ej_cell},
        output_path=out_path,
    )
    assert out_path.with_suffix(".png").exists()
    assert any(abs(y - 4.0) < 1e-9 for _x, y in captured), (
        f"expected endpoint y=4.0 (source_self), got: {captured}"
    )
    assert not any(abs(y - (-1.7)) < 1e-9 for _x, y in captured), (
        f"endpoint y MUST NOT be -1.7 (held_out); got: {captured}"
    )


def test_source_self_trajectory_emits_real_trajectory(tmp_path: Path):
    """End-to-end: an eval JSON with a populated sidecar produces a real
    multi-point trajectory (≥2 line segments per cell) — NOT endpoint-only.

    The round-2 bug was: this code path produced endpoint-only markers for
    EVERY cell because the wrong key was read.
    """
    sidecar = tmp_path / "ft_dense_b30_dynamics.json"
    sidecar.write_text(
        json.dumps(
            {
                "snapshots": {
                    "10": {"step": 10, "dynamics/source_delta_g": 2.0},
                    "20": {"step": 20, "dynamics/source_delta_g": 5.0},
                    "30": {"step": 30, "dynamics/source_delta_g": 9.0},
                }
            }
        )
    )
    ej_cell = {
        "dynamics_snapshots_path": str(sidecar),
        "aggregates": {
            "source_self_mean_delta_g": 9.0,
            "held_out_mean_delta_g": -2.5,
        },
    }
    out_path = tmp_path / "source_self_trajectory.png"
    plot_issue_514.source_self_trajectory_figure(
        ft_514_cells={"ft_dense_b30": ej_cell},
        output_path=out_path,
    )
    # File written (the fallback _try_savefig_paper writes .png + .pdf when
    # paper_plots is missing; the analysis.paper_plots helper writes the
    # same names + a .meta.json).
    assert out_path.with_suffix(".png").exists()


def test_source_self_trajectory_endpoint_fallback_no_sidecar(tmp_path: Path):
    """Cell with no sidecar path AND a valid endpoint aggregates → falls back
    to endpoint marker, no exception.
    """
    ej_cell = {
        # No dynamics_snapshots_path.
        "aggregates": {
            "source_self_mean_delta_g": 7.0,
            "held_out_mean_delta_g": -1.5,
        },
    }
    out_path = tmp_path / "source_self_trajectory_fallback.png"
    # Must NOT raise.
    plot_issue_514.source_self_trajectory_figure(
        ft_514_cells={"ft_dense_b30": ej_cell},
        output_path=out_path,
    )
    assert out_path.with_suffix(".png").exists()


def test_source_self_trajectory_single_snapshot_becomes_endpoint(tmp_path: Path):
    """Sidecar with only ONE snapshot — a single-point "trajectory" is not a
    trajectory; fall back to the endpoint marker so the legend stays
    coherent across cells.
    """
    sidecar = tmp_path / "one_snap.json"
    sidecar.write_text(
        json.dumps({"snapshots": {"10": {"step": 10, "dynamics/source_delta_g": 4.0}}})
    )
    ej_cell = {
        "dynamics_snapshots_path": str(sidecar),
        "aggregates": {
            "source_self_mean_delta_g": 4.0,
            "held_out_mean_delta_g": -1.0,
        },
    }
    out_path = tmp_path / "one_snap_traj.png"
    plot_issue_514.source_self_trajectory_figure(
        ft_514_cells={"ft_dense_b30": ej_cell},
        output_path=out_path,
    )
    assert out_path.with_suffix(".png").exists()


# ── compute_excluded_cells (B7 single-source-of-truth) ───────────────────────


def test_compute_excluded_cells_rejects_ft_b2_like(tmp_path: Path):
    """compute_excluded_cells uses is_clean_anchor to derive the exclusion set.

    Build a small {cell: eval_json} dict mirroring the actual eval JSON
    shape (the function pulls aggregates.source_n_probes / source_self_mean
    / held_out_mean / and reads r_collapse + held_out_g_logprob via the
    standalone helpers).
    """
    # ft_b1-like (clean): N=20 probes, no collapse, sub-ceiling.
    ft_b1_eval = _build_eval_for_clean_check(
        n_source=20,
        n_collapsed=0,
        held_out_g_logprob_mean=-6.20,
        source_self_mean_delta_g=8.193,
        held_out_mean_delta_g=-0.31,
    )
    # ft_b2-like (collapsed): N=20 probes, 19 collapsed → 1 valid, sub-ceiling
    # saturated.
    ft_b2_eval = _build_eval_for_clean_check(
        n_source=20,
        n_collapsed=19,
        held_out_g_logprob_mean=-0.865,
        source_self_mean_delta_g=6.774,
        held_out_mean_delta_g=-0.92,
    )
    cells = {"ft_b1": ft_b1_eval, "ft_b2": ft_b2_eval}
    excluded = plot_issue_514.compute_excluded_cells(cells)
    assert "ft_b2" in excluded
    assert "ft_b1" not in excluded


def test_hero_figure_accepts_dynamic_excluded_parameter(tmp_path: Path):
    """B11 round-4 pivot: hero_figure accepts the dynamically computed
    exclusion set via the ``excluded`` keyword and applies it (instead of the
    static module-level EXCLUDED_FROM_BOOTSTRAP).

    Verifies:
      - calling with the default (no kwarg) reproduces the static behavior;
      - calling with an explicit excluded tuple drives alpha rendering on
        the cells it names (i.e. the function consumes the parameter rather
        than ignoring it).

    The check is shape-only (file written, no exception). The semantic
    correctness — "alpha=0.4 iff cell in excluded or cell=='ft_b3'" — is
    enforced by reading the source.
    """
    # Two synthetic #508-shaped cells with valid aggregates.
    ej_b1 = {"aggregates": {"source_self_mean_delta_g": 8.2, "held_out_mean_delta_g": -0.3}}
    ej_b2 = {"aggregates": {"source_self_mean_delta_g": 6.8, "held_out_mean_delta_g": -0.9}}

    out_path = tmp_path / "hero_default"
    # Default exclusion path (uses module-level EXCLUDED_FROM_BOOTSTRAP).
    plot_issue_514.hero_figure(
        lora_cells={},
        ft_508_cells={"ft_b1": ej_b1, "ft_b2": ej_b2},
        ft_514_dense_cells={},
        ft_514_lowlr_cells={},
        output_path=out_path,
    )
    assert out_path.with_suffix(".png").exists()

    out_path2 = tmp_path / "hero_explicit"
    # Explicit excluded set; pass an unusual cell name to prove the parameter
    # is wired (it must not raise; alpha rendering is enforced by code-read).
    plot_issue_514.hero_figure(
        lora_cells={},
        ft_508_cells={"ft_b1": ej_b1, "ft_b2": ej_b2},
        ft_514_dense_cells={},
        ft_514_lowlr_cells={},
        output_path=out_path2,
        excluded=("ft_b1",),
    )
    assert out_path2.with_suffix(".png").exists()


def test_hero_figure_dense_lever_dims_excluded_514_cell(tmp_path: Path, monkeypatch):
    """B13 round-2 pivot: when a #514 dense-lever cell lands in ``excluded``,
    the hero_figure renders it at alpha=0.4 (via ax.scatter) AND draws the
    clean-cells line via ax.plot at alpha=1.0.

    Round-1 pivot v4 bug: the dense + lower-LR lever rendering called
    ``ax.plot(xs, ys, ...)`` over a bulk list of ALL cells and applied a
    SINGLE alpha to the whole lever, so a #514 cell in ``excluded`` was
    silently rendered at full opacity. ``compute_excluded_cells`` correctly
    derives the exclusion set from ALL loaded cells (line 760 in
    ``main``), but the dense-lever and lower-LR-lever loops never consumed
    the ``excluded`` parameter.

    This test captures every ``ax.plot`` and ``ax.scatter`` invocation
    inside ``hero_figure`` and asserts:
      (a) the dirty #514 cell appears as a scatter call with alpha == 0.4
          and its (x, y) matches the cell's aggregates;
      (b) the dirty cell does NOT appear in any full-alpha ax.plot's xs
          (i.e. the trend line skips it);
      (c) at least one clean #514 cell appears in an ax.plot with alpha
          ~= 1.0.
    """
    import matplotlib.axes

    plot_calls: list[dict] = []
    scatter_calls: list[dict] = []

    real_plot = matplotlib.axes.Axes.plot
    real_scatter = matplotlib.axes.Axes.scatter

    def _capturing_plot(self, *args, **kwargs):
        # Record xs, ys, alpha. ax.plot signature: plot(xs, ys, ...).
        if len(args) >= 2:
            xs, ys = args[0], args[1]
            plot_calls.append(
                {
                    "xs": list(xs),
                    "ys": list(ys),
                    "alpha": kwargs.get("alpha", 1.0),
                    "label": kwargs.get("label"),
                }
            )
        return real_plot(self, *args, **kwargs)

    def _capturing_scatter(self, x, y, *args, **kwargs):
        scatter_calls.append(
            {
                "x": float(x),
                "y": float(y),
                "alpha": kwargs.get("alpha", 1.0),
                "label": kwargs.get("label"),
            }
        )
        return real_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _capturing_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _capturing_scatter)

    # Three #514 dense-lever cells, sorted by x.
    # ft_dense_b10 (clean), ft_dense_b20 (clean), ft_dense_b30 (DIRTY).
    ej_clean_b10 = {"aggregates": {"source_self_mean_delta_g": 4.0, "held_out_mean_delta_g": -0.1}}
    ej_clean_b20 = {"aggregates": {"source_self_mean_delta_g": 6.0, "held_out_mean_delta_g": -0.2}}
    ej_dirty_b30 = {"aggregates": {"source_self_mean_delta_g": 9.0, "held_out_mean_delta_g": -1.5}}

    out_path = tmp_path / "hero_b13"
    plot_issue_514.hero_figure(
        lora_cells={},
        ft_508_cells={},
        ft_514_dense_cells={
            "ft_dense_b10": ej_clean_b10,
            "ft_dense_b20": ej_clean_b20,
            "ft_dense_b30": ej_dirty_b30,
        },
        ft_514_lowlr_cells={},
        output_path=out_path,
        excluded=("ft_dense_b30",),
    )
    assert out_path.with_suffix(".png").exists()

    # (a) The dirty cell appears as a scatter call at alpha=0.4 with the
    # right (x, y).
    dirty_scatter_hits = [
        s
        for s in scatter_calls
        if abs(s["x"] - 9.0) < 1e-9 and abs(s["y"] - (-1.5)) < 1e-9 and abs(s["alpha"] - 0.4) < 1e-9
    ]
    assert dirty_scatter_hits, (
        f"expected an ax.scatter call for the dirty #514 cell "
        f"(x=9.0, y=-1.5, alpha=0.4); got scatter_calls={scatter_calls}"
    )

    # (b) No full-alpha plot includes the dirty cell's x-coordinate.
    full_alpha_plots = [p for p in plot_calls if abs(p["alpha"] - 1.0) < 1e-9]
    for p in full_alpha_plots:
        assert 9.0 not in p["xs"], (
            f"the dirty cell (x=9.0) MUST NOT appear in any full-alpha "
            f"ax.plot's xs; offending plot call: {p}"
        )

    # (c) The clean #514 cells appear in a full-alpha ax.plot (the trend line).
    clean_xs_present = any(4.0 in p["xs"] and 6.0 in p["xs"] for p in full_alpha_plots)
    assert clean_xs_present, (
        f"expected a full-alpha ax.plot containing the two clean #514 cells "
        f"(x=4.0, x=6.0); got full_alpha_plots={full_alpha_plots}"
    )


def test_hero_figure_lowlr_lever_dims_excluded_514_cell(tmp_path: Path, monkeypatch):
    """B13 round-2 pivot: same per-cell exclusion contract for the lower-LR
    lever (the second of the two #514 rendering paths the round-1 pivot v4
    missed).
    """
    import matplotlib.axes

    plot_calls: list[dict] = []
    scatter_calls: list[dict] = []

    real_plot = matplotlib.axes.Axes.plot
    real_scatter = matplotlib.axes.Axes.scatter

    def _capturing_plot(self, *args, **kwargs):
        if len(args) >= 2:
            plot_calls.append(
                {
                    "xs": list(args[0]),
                    "ys": list(args[1]),
                    "alpha": kwargs.get("alpha", 1.0),
                }
            )
        return real_plot(self, *args, **kwargs)

    def _capturing_scatter(self, x, y, *args, **kwargs):
        scatter_calls.append({"x": float(x), "y": float(y), "alpha": kwargs.get("alpha", 1.0)})
        return real_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _capturing_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _capturing_scatter)

    ej_clean = {"aggregates": {"source_self_mean_delta_g": 5.0, "held_out_mean_delta_g": -0.4}}
    ej_dirty = {"aggregates": {"source_self_mean_delta_g": 8.5, "held_out_mean_delta_g": -1.8}}

    out_path = tmp_path / "hero_lowlr_b13"
    plot_issue_514.hero_figure(
        lora_cells={},
        ft_508_cells={},
        ft_514_dense_cells={},
        ft_514_lowlr_cells={
            "ft_lowlr_b10": ej_clean,
            "ft_lowlr_b30": ej_dirty,
        },
        output_path=out_path,
        excluded=("ft_lowlr_b30",),
    )
    assert out_path.with_suffix(".png").exists()

    # Dirty lower-LR cell rendered at alpha=0.4.
    assert any(
        abs(s["x"] - 8.5) < 1e-9 and abs(s["y"] - (-1.8)) < 1e-9 and abs(s["alpha"] - 0.4) < 1e-9
        for s in scatter_calls
    ), f"expected dirty lower-LR cell at alpha=0.4; got scatter_calls={scatter_calls}"

    # No full-alpha plot includes the dirty cell.
    full_alpha_plots = [p for p in plot_calls if abs(p["alpha"] - 1.0) < 1e-9]
    for p in full_alpha_plots:
        assert 8.5 not in p["xs"], (
            f"dirty lower-LR cell (x=8.5) MUST NOT appear in any "
            f"full-alpha ax.plot's xs; offending plot call: {p}"
        )


def test_compute_excluded_cells_excludes_additional_collapsed_anchor(tmp_path: Path):
    """B11 round-4 pivot: compute_excluded_cells walks ALL loaded cells (not
    only ft_b2) and excludes any that fail is_clean_anchor.

    Constructs an eval-JSON dict with TWO contaminated cells: ft_b2-like
    (collapsed, sub-ceiling-saturated) AND a synthetic #514 cell that also
    has r_collapse_rate >= 0.5. The function returns BOTH cell slugs in the
    excluded tuple, NOT just ft_b2.
    """
    ft_b1_eval = _build_eval_for_clean_check(
        n_source=20,
        n_collapsed=0,
        held_out_g_logprob_mean=-6.20,
        source_self_mean_delta_g=8.193,
        held_out_mean_delta_g=-0.31,
    )
    ft_b2_eval = _build_eval_for_clean_check(
        n_source=20,
        n_collapsed=19,
        held_out_g_logprob_mean=-0.865,
        source_self_mean_delta_g=6.774,
        held_out_mean_delta_g=-0.92,
    )
    # Synthetic #514 dense cell that ALSO collapsed (15/20 source probes).
    ft_dense_collapsed_eval = _build_eval_for_clean_check(
        n_source=20,
        n_collapsed=15,
        held_out_g_logprob_mean=-1.5,  # saturated above sub-ceiling
        source_self_mean_delta_g=5.0,
        held_out_mean_delta_g=-1.2,
    )
    cells = {
        "ft_b1": ft_b1_eval,
        "ft_b2": ft_b2_eval,
        "ft_dense_b30": ft_dense_collapsed_eval,
    }
    excluded = plot_issue_514.compute_excluded_cells(cells)
    assert "ft_b2" in excluded
    assert "ft_dense_b30" in excluded
    assert "ft_b1" not in excluded


def _build_eval_for_clean_check(
    *,
    n_source: int,
    n_collapsed: int,
    held_out_g_logprob_mean: float,
    source_self_mean_delta_g: float,
    held_out_mean_delta_g: float,
) -> dict:
    """Minimal eval JSON shape sufficient for compute_source_r_collapse_rate +
    get_held_out_g_logprob_mean + the aggregates the is_clean_anchor gate
    reads. Mirrors the test_issue514_abort_logic helper.
    """
    source_persona = "villain"
    delta_g_source: dict = {source_persona: {}}
    for i in range(n_source):
        q = f"q{i}"
        delta_g_source[source_persona][q] = {
            "trained_logp": -1.0,
            "base_logp": -10.0,
            "delta_g": 9.0,
            "trained_argmax_marker": True,
            "base_argmax_marker": False,
            "r_collapsed": i < n_collapsed,
            "n_marker_in_R": 0,
        }
    return {
        "delta_g_source": delta_g_source,
        "aggregates": {
            "held_out_g_logprob_mean": held_out_g_logprob_mean,
            "source_self_mean_delta_g": source_self_mean_delta_g,
            "held_out_mean_delta_g": held_out_mean_delta_g,
            "source_n_probes": n_source - n_collapsed,
        },
    }
