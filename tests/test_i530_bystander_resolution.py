# em-dash + Qwen marker " ※" + Greek ΔG intentional
"""Regression tests for `scripts/i530_emit_bystander_resolution.py`.

The plan §6.5 deliverable row requires a standalone
``bystander_resolution.json`` per cell, but the on-pod eval rig embeds
the raw per-(probe, question) numbers inside the wider ``trajectory.json``
and never lifts them.  Upload-verifier round 1 (2026-06-09) caught the
zero-file gap; this script is the flattener and these tests pin its
contract so a future refactor cannot silently regress.

Tests cover:
  - Schema-conformant trajectory.json flattens to the expected payload
    shape (sentinel keys, per-probe rows, gate scalars, band fractions).
  - Missing top-level keys fail loud (no silent ``.get(..., {})`` fallback).
  - Missing per-question keys (delta_g, argmax_marker) raise KeyError.
  - The de-saturation gate verdict matches its scalar inputs.
  - Multi-checkpoint trajectories pick the highest-fraction one.
  - ``--overwrite`` is required to replace an existing file.

Tests are CPU-only, run under ``uv run pytest tests/test_i530_bystander_resolution.py``,
and complete in well under a second.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "i530_emit_bystander_resolution.py"


@pytest.fixture(scope="module")
def emitter_mod():
    """Import `scripts/i530_emit_bystander_resolution.py` as a module."""

    spec = importlib.util.spec_from_file_location(
        "i530_emit_bystander_resolution_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_trajectory(
    *,
    n_personas: int = 3,
    n_questions: int = 4,
    base_delta_g: float = 6.0,
    argmax_rate: float = 0.25,
    cell: str = "c504v3_test_near",
    seed: int = 42,
    n_checkpoints: int = 1,
) -> dict[str, Any]:
    """Build a minimal in-memory trajectory.json fixture matching the rig schema."""

    personas = [f"p{i}" for i in range(n_personas)]
    questions = [f"Q{j}?" for j in range(n_questions)]

    checkpoints: list[dict[str, Any]] = []
    for ck_idx in range(n_checkpoints):
        # Spread fractions across [0.25, 0.5, 0.75, 1.0] so we can verify the
        # "highest fraction wins" logic.
        frac_grid = [0.25, 0.5, 0.75, 1.0]
        frac = frac_grid[ck_idx % len(frac_grid)]
        held_out: dict[str, dict[str, dict[str, Any]]] = {}
        for pi, p in enumerate(personas):
            held_out[p] = {}
            for qi, q in enumerate(questions):
                # Deterministic test values; vary so we exercise the aggregator.
                dg = base_delta_g + (pi * 0.1) + (qi * 0.01) + (frac * 0.2)
                am = ((pi + qi) % round(1 / max(argmax_rate, 1e-6))) == 0
                held_out[p][q] = {
                    "g_logp": -16.0 - dg,
                    "b_logp": -16.0,
                    "delta_g": dg,
                    "argmax_marker": bool(am),
                    "n_marker_in_R": int(am),
                    "r_collapsed": False,
                    "kl": 0.1 + dg * 0.01,
                }
        checkpoints.append(
            {
                "frac": frac,
                "step": int(20 * frac),
                "adapter_path": f"/workspace/runs/issue_530/{cell}_seed{seed}/adapter_{frac}",
                "source_self": {
                    "g_logp_mean": -16.5,
                    "b_logp_mean": -22.5,
                    "delta_g_mean": 6.0,
                    "emission_p": 0.0,
                    "r_collapsed": False,
                },
                "held_out_collapse_share": 0.0,
                "n_held_out_collapsed": 0,
                "held_out": held_out,
                "byte_identical_guard": "pass",
            }
        )

    return {
        "schema_version": 1,
        "cell": cell,
        "seed": seed,
        "source": "test",
        "marker_text": " ※",  # leading-space marker, mirrors prod
        "marker_token_id": 83399,
        "matched_slice_target_nats": 6.0,
        "n_held_out_personas": n_personas,
        "held_out_personas": personas,
        "n_eval_questions": n_questions,
        "eval_questions": questions,
        "kl_computed": True,
        "checkpoints": checkpoints,
        "git_commit": "deadbeef",
        "hostname": "test-host",
        "timestamp_utc": "2026-06-09T15:00:00",
    }


def _write_cell_with_trajectory(tmp_path: Path, traj: dict[str, Any]) -> Path:
    """Lay down `<tmp>/eval_results/issue_530/<cell>_seed<seed>/trajectory.json`."""

    cell_dir = tmp_path / "eval_results" / "issue_530" / f"{traj['cell']}_seed{traj['seed']}"
    cell_dir.mkdir(parents=True)
    (cell_dir / "trajectory.json").write_text(json.dumps(traj))
    return cell_dir


# --- Happy path -------------------------------------------------------------


def test_payload_has_required_sentinel_and_schema_keys(emitter_mod, tmp_path: Path):
    """The emitted JSON carries all sentinel keys + cell/seed identity."""

    traj = _make_trajectory()
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    out_path, payload = emitter_mod._emit_for_cell(
        cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
    )

    assert out_path == cell_dir / "bystander_resolution.json"
    assert out_path.is_file()

    required = {
        "sentinel_schema_version",
        "kind",
        "version",
        "cell",
        "seed",
        "marker_text",
        "marker_token_id",
        "chosen_checkpoint",
        "n_held_out_probes",
        "n_eval_questions_per_probe",
        "n_pairs_evaluated",
        "de_saturation_gate",
        "delta_g_band_fractions",
        "per_probe",
        "raw_distributions",
        "provenance",
    }
    assert required.issubset(payload.keys()), sorted(required - payload.keys())
    assert payload["kind"] == "i530_bystander_resolution"
    assert payload["cell"] == "c504v3_test_near"
    assert payload["seed"] == 42
    assert payload["sentinel_schema_version"] == 1
    assert payload["marker_token_id"] == 83399


def test_per_probe_aggregates_match_input(emitter_mod, tmp_path: Path):
    """Per-probe rows aggregate the deterministic fixture correctly."""

    traj = _make_trajectory(n_personas=2, n_questions=3, argmax_rate=0.5)
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    _out_path, payload = emitter_mod._emit_for_cell(
        cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
    )

    # n_probes equals n_personas in the fixture
    assert payload["n_held_out_probes"] == 2
    assert payload["n_eval_questions_per_probe"] == 3
    assert payload["n_pairs_evaluated"] == 6
    assert len(payload["per_probe"]) == 2

    for probe in payload["per_probe"]:
        assert probe["n_questions_evaluated"] == 3
        assert probe["mean_delta_g"] is not None
        assert probe["argmax_marker_share"] is not None


def test_de_saturation_gate_pass_when_log_p_is_well_below_zero_and_argmax_below_ceiling(
    emitter_mod, tmp_path: Path
):
    """A clearly de-saturated cell PASSes the gate."""

    # The fixture's g_logp = -16.0 - delta_g, so with base_delta_g=6.0 the median
    # g_logp is around -22, well below the -2 nat headroom requirement.
    # argmax_rate=0.05 gives a very low share, well under 60%.
    traj = _make_trajectory(base_delta_g=6.0, argmax_rate=0.05)
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    _, payload = emitter_mod._emit_for_cell(
        cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
    )

    gate = payload["de_saturation_gate"]
    assert gate["median_g_logp_clears_headroom"] is True
    assert gate["argmax_below_ceiling"] is True
    assert gate["verdict"] == "PASS"


def test_de_saturation_gate_fail_when_argmax_above_ceiling(emitter_mod, tmp_path: Path):
    """A cell with argmax rate above 60% FAILs the gate."""

    # argmax_rate=1.0 → every pair argmax_marker=True → share = 1.0 > 0.60.
    traj = _make_trajectory(base_delta_g=6.0, argmax_rate=1.0)
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    _, payload = emitter_mod._emit_for_cell(
        cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
    )

    gate = payload["de_saturation_gate"]
    assert gate["argmax_marker_share_across_pairs"] == pytest.approx(1.0)
    assert gate["argmax_below_ceiling"] is False
    assert gate["verdict"] == "FAIL"


# --- Schema-failure paths (fail-loud contract) ------------------------------


def test_missing_top_level_key_fails_loud(emitter_mod, tmp_path: Path):
    """Trajectories missing required top-level keys raise KeyError."""

    traj = _make_trajectory()
    del traj["n_held_out_personas"]
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    with pytest.raises(KeyError) as exc_info:
        emitter_mod._emit_for_cell(
            cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
        )
    assert "n_held_out_personas" in str(exc_info.value)


def test_missing_per_question_key_fails_loud(emitter_mod, tmp_path: Path):
    """Trajectories with per-(probe,question) rows missing delta_g raise KeyError."""

    traj = _make_trajectory()
    # Surgically remove delta_g from one row to simulate a corrupted/old eval.
    p0 = traj["held_out_personas"][0]
    q0 = traj["eval_questions"][0]
    del traj["checkpoints"][0]["held_out"][p0][q0]["delta_g"]
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    with pytest.raises(KeyError) as exc_info:
        emitter_mod._emit_for_cell(
            cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
        )
    assert "delta_g" in str(exc_info.value)


def test_empty_checkpoints_fails_loud(emitter_mod, tmp_path: Path):
    """A trajectory with zero checkpoints raises ValueError (no silent skip)."""

    traj = _make_trajectory()
    traj["checkpoints"] = []
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    with pytest.raises(ValueError) as exc_info:
        emitter_mod._emit_for_cell(
            cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
        )
    assert "checkpoints list is empty" in str(exc_info.value)


# --- Multi-checkpoint + idempotency -----------------------------------------


def test_picks_highest_checkpoint_fraction(emitter_mod, tmp_path: Path):
    """When multiple checkpoints exist, the highest-fraction one wins."""

    traj = _make_trajectory(n_checkpoints=4)
    # Fixture gives fracs [0.25, 0.5, 0.75, 1.0]; picker must select 1.0.
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    _, payload = emitter_mod._emit_for_cell(
        cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
    )

    assert payload["chosen_checkpoint"]["fraction"] == 1.0
    assert payload["chosen_checkpoint"]["n_checkpoints_available_in_trajectory"] == 4


def test_overwrite_required_for_replace(emitter_mod, tmp_path: Path):
    """A second emission without --overwrite raises FileExistsError."""

    traj = _make_trajectory()
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    emitter_mod._emit_for_cell(cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False)
    with pytest.raises(FileExistsError):
        emitter_mod._emit_for_cell(
            cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
        )
    # ...and succeeds with overwrite=True.
    emitter_mod._emit_for_cell(cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=True)


# --- Band fractions ---------------------------------------------------------


def test_band_fractions_cover_three_upper_bounds(emitter_mod, tmp_path: Path):
    """Three candidate ΔG upper bounds appear in delta_g_band_fractions."""

    traj = _make_trajectory()
    cell_dir = _write_cell_with_trajectory(tmp_path, traj)
    slab_root = cell_dir.parent

    _, payload = emitter_mod._emit_for_cell(
        cell_dir, slab_root=slab_root, git_commit="abc123", overwrite=False
    )

    bands = payload["delta_g_band_fractions"]
    assert len(bands) == 3
    # Each band records bounds + count + fraction.
    for _key, b in bands.items():
        assert "lower_exclusive" in b
        assert "upper_inclusive" in b
        assert "n_pairs_in_band" in b
        assert "fraction_of_pairs" in b


# --- End-to-end CLI smoke ---------------------------------------------------


def test_cli_smoke_runs_end_to_end_on_a_minimal_slab(emitter_mod, tmp_path: Path):
    """A two-cell slab end-to-end through main() returns 0 and writes both files."""

    for cell, seed in [("c504v3_test_near", 42), ("c504v3_test_far", 137)]:
        traj = _make_trajectory(cell=cell, seed=seed)
        _write_cell_with_trajectory(tmp_path, traj)
    slab_root = tmp_path / "eval_results" / "issue_530"

    rc = emitter_mod.main(["--slab-root", str(slab_root)])

    assert rc == 0
    written = sorted(slab_root.glob("c504v3_*_seed*/bystander_resolution.json"))
    assert len(written) == 2
    for p in written:
        payload = json.loads(p.read_text())
        assert payload["kind"] == "i530_bystander_resolution"
