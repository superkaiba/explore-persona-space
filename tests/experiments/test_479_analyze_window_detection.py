# ruff: noqa: RUF003  # ※/×/− glyphs intentional
"""Task #479 round-2 Blocker 1 regression guard: window detection on the REAL v2 trajectory schema.

The eval_trajectory.py rig (schema_version="i472_v2") writes:
  - ``checkpoints[].source_self.emission_rate``  ← source's own emission rate
  - ``checkpoints[].bystander_emission.{mean, se, per_persona_rate}``
  - ``checkpoints[].held_out[persona][q].{argmax_marker, delta_g, ...}``
    where ``held_out`` only contains BYSTANDER personas (NOT the source).

Round 1's analyzer mistakenly read ``per_persona_rate.get(source)`` from
``held_out`` — which is always None because the source isn't in held_out.
Result: window_detected always False, Stage 2 always fires (~21 GPU-h waste).

This test asserts ``window_detected == True`` on a synthetic trajectory
fixture whose source emission lives at the v2 schema's CANONICAL location
(``source_self.emission_rate``), and bystander rates live in ``held_out``
without a source entry. The test would FAIL if the analyzer regressed back
to reading source from ``held_out[source]``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_TEXT,
    SOURCE_PERSONA,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "i479_analyze.py"


def _checkpoint_v2(
    step: int,
    source_emit: float,
    bystander_rate: float,
    bystander_dg: float,
    bystander_personas: list[str],
    n_questions: int = 2,
) -> dict:
    """Build a v2-schema checkpoint dict matching eval_trajectory.py's emit.

    Source persona is NOT in ``held_out`` (matches the rig's
    ``for persona in eval_personas`` loop where ``eval_personas`` is the
    held-out panel WITHOUT the source).
    """
    held_out: dict[str, dict[str, dict]] = {}
    per_persona_rate: dict[str, float] = {}
    for persona in bystander_personas:
        per_q = {}
        n_marker_argmax = round(bystander_rate * n_questions)
        for i in range(n_questions):
            is_marker = i < n_marker_argmax
            per_q[f"q{i}"] = {
                "g_logp": -25.0 + bystander_dg,
                "b_logp": -25.0,
                "delta_g": bystander_dg,
                "argmax_marker": is_marker,
                "argmax_token_id_trained": EXPECTED_MARKER_TOKEN_ID if is_marker else 151645,
                "argmax_token_id_base": 151645,
                "n_marker_in_R": 0,
                "r_collapsed": False,
                "kl": 1.0,
            }
        held_out[persona] = per_q
        per_persona_rate[persona] = bystander_rate
    # Source: NOT in held_out (matches rig). Source emission lives ONLY in source_self.
    n_source_marker = round(source_emit * n_questions)
    return {
        "step": step,
        "frac": float(step),
        "step_key": f"{step:04d}",
        "adapter_path": f"/dev/null/step_{step:04d}",
        "source_self": {
            "g_logp_mean": -0.1,
            "b_logp_mean": -25.0,
            "delta_g_mean": 25.0,
            "r_collapsed": False,
            # The PRIMARY field the analyzer must read (round-1 bug: read
            # held_out[source] instead, which is always missing).
            "emission_rate": source_emit,
            "n_marker_argmax": n_source_marker,
            "n_questions": n_questions,
        },
        "bystander_emission": {
            "mean": bystander_rate,
            "se": 0.0,
            "n_personas": len(bystander_personas),
            "per_persona_rate": per_persona_rate,
        },
        "held_out_collapse_share": 0.0,
        "n_held_out_collapsed": 0,
        "held_out": held_out,
    }


def _write_trajectory_v2(
    path: Path,
    cell: str,
    seed: int,
    ckpts: list[dict],
) -> None:
    payload = {
        "schema_version": "i472_v2",
        "cell": cell,
        "seed": seed,
        "source": SOURCE_PERSONA,
        "marker_text": MARKER_TEXT,
        "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
        "matched_slice_target_nats": 8.0,
        "n_held_out_personas": len(ckpts[0]["held_out"]) if ckpts else 0,
        "held_out_personas": (sorted(ckpts[0]["held_out"].keys()) if ckpts else []),
        "n_eval_questions": 2,
        "eval_questions": ["q0", "q1"],
        "kl_computed": True,
        "checkpoints": ckpts,
        "git_commit": "test",
        "hostname": "test",
        "timestamp_utc": "test",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _run_analyze(
    slab_root: Path, figures_dir: Path, manifest_path: Path, base_panel_path: Path | None = None
) -> dict:
    """Drive scripts/i479_analyze.py via subprocess so the full CLI path runs."""
    cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--slab-root",
        str(slab_root),
        "--figures-dir",
        str(figures_dir),
        "--cells",
        "c479_base",
        "--seeds",
        "42,137",
        "--manifest-path",
        str(manifest_path),
    ]
    if base_panel_path is not None:
        cmd.extend(["--base-panel-path", str(base_panel_path)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"i479_analyze.py exited {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return json.loads(manifest_path.read_text())


# ── The Must-Fix gate: window detection on the REAL v2 schema. ────────────────


def test_window_detected_when_source_emit_high_and_bystander_low(tmp_path):
    """Both seeds: source ≥ 0.8, bystander < 0.1, sub-ceiling ΔG → window_detected."""
    bystanders = ["medical_doctor", "french_person"]
    # 4 checkpoints — clean window at steps 50-100.
    ckpts_window = [
        _checkpoint_v2(
            5,
            source_emit=0.0,
            bystander_rate=0.0,
            bystander_dg=-15.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            50,
            source_emit=1.0,
            bystander_rate=0.0,
            bystander_dg=-10.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            100,
            source_emit=1.0,
            bystander_rate=0.0,
            bystander_dg=-8.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            200,
            source_emit=1.0,
            bystander_rate=1.0,
            bystander_dg=-0.5,
            bystander_personas=bystanders,
        ),
    ]
    slab_root = tmp_path / "slab"
    for seed in (42, 137):
        _write_trajectory_v2(
            slab_root / f"c479_base_seed{seed}" / "trajectory.json", "c479_base", seed, ckpts_window
        )
    manifest = _run_analyze(slab_root, tmp_path / "figures", tmp_path / "manifest.json")

    assert manifest["window_detected"] is True, (
        f"window_detected should be True (source 1.0 / bystander 0.0 at steps 50-100, "
        f"both seeds, sub-ceiling ΔG -10 / -8). Got manifest: {manifest['per_cell_window']}"
    )
    w = manifest["per_cell_window"]["c479_base"]
    assert w["window_start_step"] == 50
    assert w["window_end_step"] == 100
    assert w["width_steps"] == 50  # 100 - 50 = 50 ≥ 25-step minimum.
    assert w["per_seed_window_widths"]["42"] == 50
    assert w["per_seed_window_widths"]["137"] == 50


def test_window_not_detected_when_saturated_everywhere(tmp_path):
    """Source 1.0 + bystander 1.0 + ΔG at ceiling → window NOT detected (saturation)."""
    bystanders = ["medical_doctor"]
    ckpts_saturated = [
        _checkpoint_v2(
            s, source_emit=1.0, bystander_rate=1.0, bystander_dg=-0.5, bystander_personas=bystanders
        )
        for s in (5, 50, 100, 200)
    ]
    slab_root = tmp_path / "slab"
    for seed in (42, 137):
        _write_trajectory_v2(
            slab_root / f"c479_base_seed{seed}" / "trajectory.json",
            "c479_base",
            seed,
            ckpts_saturated,
        )
    manifest = _run_analyze(slab_root, tmp_path / "figures", tmp_path / "manifest.json")

    assert manifest["window_detected"] is False, (
        "window_detected should be False under full saturation (bystander 1.0, "
        "ΔG within 1 nat of 0 across the panel)."
    )


def test_window_not_detected_when_source_emission_too_low(tmp_path):
    """Source 0.5 < 0.8 floor → window NOT detected."""
    bystanders = ["medical_doctor"]
    ckpts_under = [
        _checkpoint_v2(
            s,
            source_emit=0.5,
            bystander_rate=0.0,
            bystander_dg=-10.0,
            bystander_personas=bystanders,
        )
        for s in (5, 50, 100, 200)
    ]
    slab_root = tmp_path / "slab"
    for seed in (42, 137):
        _write_trajectory_v2(
            slab_root / f"c479_base_seed{seed}" / "trajectory.json", "c479_base", seed, ckpts_under
        )
    manifest = _run_analyze(slab_root, tmp_path / "figures", tmp_path / "manifest.json")

    assert manifest["window_detected"] is False, (
        "window_detected should be False when source emission stays at 0.5 (< 0.8 floor)."
    )


def test_window_not_detected_when_only_one_seed_qualifies(tmp_path):
    """Seed 42 has a window, seed 137 does not → JOINT window not detected."""
    bystanders = ["medical_doctor"]
    ckpts_seed42 = [
        _checkpoint_v2(
            5,
            source_emit=0.0,
            bystander_rate=0.0,
            bystander_dg=-15.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            50,
            source_emit=1.0,
            bystander_rate=0.0,
            bystander_dg=-10.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            100,
            source_emit=1.0,
            bystander_rate=0.0,
            bystander_dg=-8.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            200,
            source_emit=1.0,
            bystander_rate=1.0,
            bystander_dg=-0.5,
            bystander_personas=bystanders,
        ),
    ]
    ckpts_seed137 = [
        _checkpoint_v2(
            5,
            source_emit=0.0,
            bystander_rate=0.0,
            bystander_dg=-15.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            50,
            source_emit=0.5,
            bystander_rate=0.0,
            bystander_dg=-10.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            100,
            source_emit=0.5,
            bystander_rate=0.0,
            bystander_dg=-8.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            200,
            source_emit=1.0,
            bystander_rate=1.0,
            bystander_dg=-0.5,
            bystander_personas=bystanders,
        ),
    ]
    slab_root = tmp_path / "slab"
    _write_trajectory_v2(
        slab_root / "c479_base_seed42" / "trajectory.json", "c479_base", 42, ckpts_seed42
    )
    _write_trajectory_v2(
        slab_root / "c479_base_seed137" / "trajectory.json", "c479_base", 137, ckpts_seed137
    )
    manifest = _run_analyze(slab_root, tmp_path / "figures", tmp_path / "manifest.json")

    assert manifest["window_detected"] is False, (
        "window_detected should be False when seeds disagree (joint contiguity required)."
    )


def test_per_checkpoint_summary_carries_source_emission_from_source_self(tmp_path):
    """Per-row source_emission in the manifest must match source_self.emission_rate.

    Regression guard for round-1 Blocker 1: the analyzer read
    per_persona_rate[source] from held_out (always None), so source_emission
    was always None and the window detector never fired.
    """
    bystanders = ["medical_doctor"]
    ckpts = [
        _checkpoint_v2(
            5,
            source_emit=0.25,
            bystander_rate=0.0,
            bystander_dg=-10.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            50,
            source_emit=0.75,
            bystander_rate=0.0,
            bystander_dg=-8.0,
            bystander_personas=bystanders,
        ),
        _checkpoint_v2(
            100,
            source_emit=1.0,
            bystander_rate=0.0,
            bystander_dg=-6.0,
            bystander_personas=bystanders,
        ),
    ]
    slab_root = tmp_path / "slab"
    for seed in (42, 137):
        _write_trajectory_v2(
            slab_root / f"c479_base_seed{seed}" / "trajectory.json", "c479_base", seed, ckpts
        )
    manifest = _run_analyze(slab_root, tmp_path / "figures", tmp_path / "manifest.json")

    summary = manifest["per_cell_per_seed_summary"]["c479_base"]["42"]
    source_rates = [row["source_emission"] for row in summary]
    assert source_rates == [0.25, 0.75, 1.0], (
        f"Per-checkpoint source_emission must come from source_self.emission_rate; "
        f"got {source_rates} (expected [0.25, 0.75, 1.0]). If this is [None, None, None] "
        f"the analyzer regressed back to reading held_out[source]."
    )


# ── Blocker 2 regression guard: base-panel emission anchoring. ────────────────


def _write_base_emission_panel(path: Path, panel_mean: float, personas: list[str]) -> None:
    """Write a synthetic i479_base_emission_v1 baseline."""
    payload = {
        "schema_version": "i479_base_emission_v1",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "marker_text": MARKER_TEXT,
        "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
        "marker_sep": "\n\n",
        "max_new_tokens": 2048,
        "n_eval_questions": 2,
        "eval_questions": ["q0", "q1"],
        "n_held_out_personas": len(personas),
        "held_out_personas": personas,
        "per_persona": {
            p: {
                "emission_rate": panel_mean,
                "n_marker_argmax": 0,
                "n_questions": 2,
                "per_q_argmax_marker": {},
            }
            for p in personas
        },
        "panel_mean_emission_rate": panel_mean,
        "git_commit": "test",
        "hostname": "test",
        "timestamp_utc": "test",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_base_panel_loaded_and_subtracted(tmp_path):
    """When --base-panel-path is supplied, manifest carries the floor + base-adjusted rates."""
    bystanders = ["medical_doctor", "french_person"]
    ckpts = [
        _checkpoint_v2(
            50,
            source_emit=0.9,
            bystander_rate=0.05,
            bystander_dg=-8.0,
            bystander_personas=bystanders,
        ),
    ]
    slab_root = tmp_path / "slab"
    for seed in (42, 137):
        _write_trajectory_v2(
            slab_root / f"c479_base_seed{seed}" / "trajectory.json", "c479_base", seed, ckpts
        )
    base_path = slab_root / "base_panel_emission_rate.json"
    _write_base_emission_panel(base_path, panel_mean=0.02, personas=bystanders)

    manifest = _run_analyze(
        slab_root, tmp_path / "figures", tmp_path / "manifest.json", base_panel_path=base_path
    )

    base = manifest["base_panel"]
    assert base["loaded"] is True
    assert base["schema_version"] == "i479_base_emission_v1"
    assert abs(base["panel_mean_emission_rate"] - 0.02) < 1e-9
    assert base["n_held_out_personas"] == 2

    # Each per-checkpoint row carries base-adjusted columns.
    row = manifest["per_cell_per_seed_summary"]["c479_base"]["42"][0]
    assert abs(row["bystander_mean_minus_base"] - (0.05 - 0.02)) < 1e-9
    assert abs(row["source_emission_minus_base"] - (0.9 - 0.02)) < 1e-9
    assert abs(row["base_panel_mean_emission_rate"] - 0.02) < 1e-9


def test_base_panel_missing_does_not_crash(tmp_path):
    """If --base-panel-path is absent, analyzer still runs; base entries are None."""
    bystanders = ["medical_doctor"]
    ckpts = [
        _checkpoint_v2(
            50,
            source_emit=0.9,
            bystander_rate=0.05,
            bystander_dg=-8.0,
            bystander_personas=bystanders,
        ),
    ]
    slab_root = tmp_path / "slab"
    for seed in (42, 137):
        _write_trajectory_v2(
            slab_root / f"c479_base_seed{seed}" / "trajectory.json", "c479_base", seed, ckpts
        )
    manifest = _run_analyze(slab_root, tmp_path / "figures", tmp_path / "manifest.json")

    assert manifest["base_panel"]["loaded"] is False
    assert manifest["base_panel"]["panel_mean_emission_rate"] is None
    # Raw rates still present; base-adjusted columns echo raw (since base is None).
    row = manifest["per_cell_per_seed_summary"]["c479_base"]["42"][0]
    assert abs(row["source_emission"] - 0.9) < 1e-9
    assert abs(row["bystander_mean"] - 0.05) < 1e-9


def test_base_panel_wrong_schema_rejected(tmp_path):
    """A #472 LOG-PROB baseline (wrong schema) is detected + skipped with a warning."""
    bystanders = ["medical_doctor"]
    ckpts = [
        _checkpoint_v2(
            50,
            source_emit=0.9,
            bystander_rate=0.05,
            bystander_dg=-8.0,
            bystander_personas=bystanders,
        ),
    ]
    slab_root = tmp_path / "slab"
    for seed in (42, 137):
        _write_trajectory_v2(
            slab_root / f"c479_base_seed{seed}" / "trajectory.json", "c479_base", seed, ckpts
        )
    # Synthesize a #472-style base_panel.json (LOG-PROB schema; wrong artifact).
    bad_base = slab_root / "base_panel.json"
    bad_base.parent.mkdir(parents=True, exist_ok=True)
    bad_base.write_text(
        json.dumps(
            {
                "schema_version": "i472_base_panel_v1",
                "b_logprob_by_persona": {"medical_doctor": -20.0},
            }
        )
    )
    manifest = _run_analyze(
        slab_root, tmp_path / "figures", tmp_path / "manifest.json", base_panel_path=bad_base
    )
    # Wrong schema → loader returns None; loaded=False; analyzer still completes.
    assert manifest["base_panel"]["loaded"] is False
    assert manifest["base_panel"]["panel_mean_emission_rate"] is None
