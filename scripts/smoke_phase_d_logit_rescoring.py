# ruff: noqa: RUF002, RUF003  # Greek Δ / em-dash / × intentional
"""Phase D end-to-end smoke for #505 ``logit-space-rescoring``.

Synthesizes a minimal but ESTIMABLE fixture (≥ ``len(PER_ARM_EXPANDED_LR) + 2
= 6`` rows per arm) for ``analyze_logit_rescoring.run_analysis`` and drives the
full pipeline:

    frame → per-arm + pooled OLS × 3 readouts × 1 layer
          → saturation report + cross-space agreement
          → headline_logit_comparison.json + three figures

The fixture reuses the REAL ``panel_coverage.json`` + ``geometry_predictors.
json`` artifacts from the round-1 sweep (so the schema is exactly what
production will hand the analyzer) and synthesizes:

  * 8-persona panel slice + 1 question + 1 seed
  * full-set cell + every drop-arm cell from the real ``non_default_negatives``
  * each ``slot_stats/<cell>_seed*.json`` with the full four-float leaf shape
  * one ``sweep/<cell>/seed_*/trajectory.json`` per cell carrying a frac-1.0
    checkpoint with held_out + source_probes blocks (the
    ``stored_records_at_frac`` contract)

Per-arm OLS row count is ``len(use_panel) × n_seeds = 8 × 1 = 8 ≥ 6`` — the
explicit estimability bar from the reconciler r2 verdict.

Run from the repo root:

    uv run python scripts/smoke_phase_d_logit_rescoring.py

Exits 0 + prints the artifact digest on success; non-zero on any failure.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root = parent of scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Make sure local src is on sys.path (uv run python already does this, but
# direct invocations from other cwds would otherwise fail).
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.experiments.leave_one_out_505 import (  # noqa: E402
    HEADLINE_LAYER,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.leave_one_out_505.logit_rescoring import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_FRAC,
    repro_block,
)

FIXTURE_ROOT = Path("/tmp/i505-smoke/phase_d_fixture")
OUTPUT_ROOT = FIXTURE_ROOT / "rescoring"
SWEEP_DIR = FIXTURE_ROOT / "sweep"
FIG_DIR = FIXTURE_ROOT / "figures"
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"

# Real round-1 artifacts (reused as-is — same schema the production analyzer reads).
PANEL_GATE = REPO_ROOT / "eval_results/issue_505/panel_coverage.json"
GEOMETRY_JSON = (
    REPO_ROOT / "eval_results/issue_505/expanded-predictor-reanalysis/geometry_predictors.json"
)


def _rand_leaf(rng: random.Random) -> dict:
    """One four-float capture leaf with the production schema."""
    logp_g = rng.uniform(-25.0, -2.0)
    logp_b = logp_g + rng.uniform(-2.0, 2.0)
    z_marker_g = rng.uniform(-3.0, 3.0)
    z_marker_b = z_marker_g + rng.uniform(-0.5, 0.5)
    z_eos_g = rng.uniform(-5.0, 1.0)
    z_eos_b = z_eos_g + rng.uniform(-0.5, 0.5)
    logZ_g = rng.uniform(18.0, 24.0)
    logZ_b = logZ_g + rng.uniform(-1.0, 1.0)
    return {
        "logp_g": logp_g,
        "logp_b": logp_b,
        "z_marker_g": z_marker_g,
        "z_marker_b": z_marker_b,
        "z_eos_g": z_eos_g,
        "z_eos_b": z_eos_b,
        "logZ_g": logZ_g,
        "logZ_b": logZ_b,
        "argmax_marker_g": bool(rng.random() < 0.1),
        "argmax_marker_b": bool(rng.random() < 0.05),
        "n_marker_in_R": 0,
        "r_collapsed": False,
        "delta_logp": logp_g - logp_b,
        "delta_z_marker": z_marker_g - z_marker_b,
        "delta_margin": (z_marker_g - z_eos_g) - (z_marker_b - z_eos_b),
        "delta_logz": logZ_g - logZ_b,
    }


def _build_slot_stats(
    *,
    cell: str,
    seed: int,
    personas: list[str],
    questions: list[str],
    rng: random.Random,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "seed": seed,
        "frac": TARGET_FRAC,
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_dir": f"/tmp/i505-smoke/throwaway_adapter#{cell}_seed{seed}",
        "marker_id": 83399,
        "eos_id": 151645,
        "personas": personas,
        "questions": questions,
        "slot_stats": {p: {q: _rand_leaf(rng) for q in questions} for p in personas},
        "reproducibility": repro_block({"smoke": "phase_d"}),
    }


def _build_stored_trajectory(
    *,
    cell: str,
    seed: int,
    panel: list[str],
    source: str,
    questions: list[str],
    rng: random.Random,
) -> dict:
    """One trajectory.json carrying a frac-1.0 checkpoint shaped for ``stored_records_at_frac``."""
    held_out = {
        p: {
            q: {
                "g_logp": rng.uniform(-25.0, -2.0),
                "b_logp": rng.uniform(-25.0, -2.0),
                "argmax_marker": bool(rng.random() < 0.1),
            }
            for q in questions
        }
        for p in panel
    }
    source_probes = {
        q: {
            "g_logp": rng.uniform(-25.0, -2.0),
            "b_logp": rng.uniform(-25.0, -2.0),
            "argmax_marker": bool(rng.random() < 0.5),
        }
        for q in questions
    }
    return {
        "cell": cell,
        "seed": seed,
        "checkpoints": [
            {
                "frac": "1.00",
                "held_out": held_out,
                "source_probes": source_probes,
            }
        ],
        "eval_questions": questions,
        "held_out_personas": panel,
        "source": source,
    }


def _write_fixture() -> tuple[list[str], list[str], list[str]]:
    """Lay out slot_stats/ + sweep/ trees. Returns (panel, non_default_negatives, questions)."""
    if FIXTURE_ROOT.exists():
        shutil.rmtree(FIXTURE_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / "slot_stats").mkdir()
    SWEEP_DIR.mkdir()

    panel_payload = json.loads(PANEL_GATE.read_text())
    non_default_negatives = list(panel_payload["non_default_negatives"])
    panel = list(panel_payload["panel"])[:8]  # 8 personas
    questions = ["How do you handle disagreements with others?"]
    personas_in_leafs = [*panel, SOURCE_PERSONA]

    # 3 seeds so the source-implant covariate varies (the per-arm OLS needs
    # ``delta_source_shift`` not to be perfectly collinear with the constant;
    # in production this is given by the 3 sweep seeds). 8 panel × 3 seeds = 24
    # rows per arm, comfortably above the ``len(PER_ARM_EXPANDED_LR) + 2 = 6``
    # estimability bar.
    seeds = [42, 137, 729]
    full_set = "c505_full_set"
    arms = [f"c505_drop_j{j_idx}" for j_idx in range(len(non_default_negatives))]
    rng = random.Random(42)
    for cell in [full_set, *arms]:
        for seed in seeds:
            (OUTPUT_ROOT / "slot_stats" / f"{cell}_seed{seed}.json").write_text(
                json.dumps(
                    _build_slot_stats(
                        cell=cell,
                        seed=seed,
                        personas=personas_in_leafs,
                        questions=questions,
                        rng=rng,
                    ),
                    indent=2,
                )
            )
            traj_dir = SWEEP_DIR / cell / f"seed_{seed}"
            traj_dir.mkdir(parents=True)
            (traj_dir / "trajectory.json").write_text(
                json.dumps(
                    _build_stored_trajectory(
                        cell=cell,
                        seed=seed,
                        panel=panel,
                        source=SOURCE_PERSONA,
                        questions=questions,
                        rng=rng,
                    ),
                    indent=2,
                )
            )

    # faithfulness.json (one entry per (cell, seed) so the headline's
    # faithfulness_summary block is exercised).
    faith_per_cell = {}
    for cell in [full_set, *arms]:
        for seed in seeds:
            faith_per_cell[f"{cell}_seed{seed}"] = {
                "hf_vs_stored": {
                    "g": {"mae": rng.uniform(0.05, 0.2), "spearman_rho": rng.uniform(0.7, 0.99)},
                    "b": {"mae": rng.uniform(0.05, 0.2), "spearman_rho": rng.uniform(0.7, 0.99)},
                },
            }
    faith = {
        "schema_version": SCHEMA_VERSION,
        "frac": TARGET_FRAC,
        "per_cell": faith_per_cell,
        "reproducibility": repro_block({"smoke": "phase_d"}),
    }
    (OUTPUT_ROOT / "faithfulness.json").write_text(json.dumps(faith, indent=2))
    return panel, non_default_negatives, questions


def main() -> int:
    if not PANEL_GATE.exists() or not GEOMETRY_JSON.exists():
        print(f"[smoke-d] FAIL: missing real round-1 artifacts ({PANEL_GATE} / {GEOMETRY_JSON})")
        return 2
    _write_fixture()
    print(f"[smoke-d] fixture written: {FIXTURE_ROOT}")
    print(f"[smoke-d] running analyze_logit_rescoring end-to-end → {ANALYSIS_DIR}")
    rc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.leave_one_out_505.analyze_logit_rescoring",
            "--rescoring-root",
            str(OUTPUT_ROOT),
            "--panel-gate",
            str(PANEL_GATE),
            "--geometry-json",
            str(GEOMETRY_JSON),
            "--sweep-dir",
            str(SWEEP_DIR),
            "--out-dir",
            str(ANALYSIS_DIR),
            "--fig-dir",
            str(FIG_DIR),
        ],
        cwd=REPO_ROOT,
        check=False,
        env={**__import__("os").environ},
    ).returncode
    if rc != 0:
        print(f"[smoke-d] FAIL: analyze_logit_rescoring exit={rc}")
        return rc
    # Verify artifacts landed and print a digest of the headline.
    headline_path = ANALYSIS_DIR / "headline_logit_comparison.json"
    if not headline_path.exists():
        print(f"[smoke-d] FAIL: {headline_path} missing")
        return 3
    headline = json.loads(headline_path.read_text())
    figs = sorted(p.name for p in FIG_DIR.glob("*.png"))
    print()
    print("[smoke-d] PASS")
    print(f"  headline_layer={headline['headline_layer']}, frac={headline['frac']}")
    print(f"  readouts={list(headline['pooled_cos_b_j_by_readout'].keys())}")
    print(
        f"  per_arm_sign_agreement_by_readout keys = "
        f"{list(headline['per_arm_sign_agreement_by_readout'].keys())}"
    )
    print(f"  n_saturation_flagged_cells={headline['n_saturation_flagged_cells']}")
    print(f"  figures={figs}")
    print(f"  analysis_dir={ANALYSIS_DIR}")
    # Quick numeric digest of pooled cos_b_j slope at the headline layer (one readout).
    rd_first = next(iter(headline["pooled_cos_b_j_by_readout"].keys()))
    expanded = headline["pooled_cos_b_j_by_readout"][rd_first]["expanded_standardized"]
    print(
        f"  headline pooled cos_b_j ({rd_first} expanded_standardized): "
        f"beta={expanded['beta']:.4f}, ci95=[{expanded['ci95_low']:.4f}, "
        f"{expanded['ci95_high']:.4f}]"
    )
    print(f"  HEADLINE_LAYER constant={HEADLINE_LAYER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
