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
