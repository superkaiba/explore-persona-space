"""CPU smoke for issue #524's Phase 4 + Phase 5 end-to-end.

Used for the experiment-implementer.md "end-to-end smoke run PER PHASE"
gate without needing a GPU. Generates synthetic activation clouds + a
synthetic ΔG matrix, runs the FULL Phase 4 predictor stack + Phase 5
nested-CV + Tobit + bootstrap on a tiny panel (4 contexts × 2 layers ×
1 extraction point × B=16 bootstrap iters), and asserts the outputs land
at the right path with non-empty content.

This pairs the carve-out for the GPU-bound phases (Phase 0.2 Sonnet
calls, Phase 1 LoRA training, Phase 2 vLLM cross-eval, Phase 3
activation extraction) which require a real GPU pod. The CPU-runnable
phases (pool_eval_100 build, Phase 4 predictor matrices, Phase 5 nested
CV) are exercised through this script.

CLI:
    uv run python scripts/issue524_cpu_smoke.py

Exit code:
  0 — all CPU phases ran end-to-end, sentinel files written.
  3 — a phase emitted a non-zero result.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("i524.cpu_smoke")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _make_synthetic_clouds(
    out_dir: Path,
    contexts: list[str],
    layers: list[int],
    points: list[str],
    n_probes: int = 50,
    hidden_dim: int = 64,
    seed: int = 42,
) -> None:
    """Generate tiny synthetic activation clouds for the smoke."""
    rng = np.random.default_rng(seed)
    for ctx in contexts:
        # Each context gets a unique direction in hidden space so the
        # predictors return non-degenerate values.
        ctx_dir = rng.standard_normal(hidden_dim)
        ctx_dir = ctx_dir / np.linalg.norm(ctx_dir)
        for layer in layers:
            for point in points:
                cloud = rng.standard_normal((n_probes, hidden_dim)) * 0.1
                cloud += ctx_dir * (1.0 + 0.1 * layer)
                cloud_dir = out_dir / ctx / f"L{layer}"
                cloud_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cloud_dir / f"{point}.npz", activations=cloud)


def _make_synthetic_dg_matrix(out_dir: Path, contexts: list[str], seed: int = 42) -> None:
    """Generate synthetic Phase 2 per-cell ΔG JSONs.

    The ΔG values are random in [-2, 2]; for the smoke we only need the
    Phase 5 nested-CV / bootstrap path to RUN, not to produce meaningful
    science. A deterministic seed gives reproducible numbers.
    """
    rng = np.random.default_rng(seed)
    per_cell = out_dir / "per_cell"
    per_cell.mkdir(parents=True, exist_ok=True)
    for ci in contexts:
        for cj in contexts:
            if ci == cj:
                continue
            payload = {
                "schema_version": 1,
                "src": ci,
                "tgt": cj,
                "delta_g_mean": float(rng.uniform(-2.0, 2.0)),
                "delta_g": float(rng.uniform(-2.0, 2.0)),
                "n_probes": 100,
            }
            (per_cell / f"G_{ci}__{cj}.json").write_text(json.dumps(payload, indent=2))


def main() -> int:
    """Run the CPU smoke end-to-end."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    contexts = ["IK01", "IK02", "IS01", "A1"]
    layers = [0, 22]
    points = ["last_prompt"]

    # Use a tempdir for the Phase 3/2 artifacts so the smoke doesn't write
    # synthetic data into the real eval_results dir. Phase 4 + Phase 5
    # read paths come from the module's hard-coded REPO_ROOT, so we
    # monkeypatch them via env or symlink. Simpler: write into the real
    # paths and clean up after.
    import issue524_phase4_predictors as phase4_mod
    import issue524_phase5_metrics as phase5_mod

    backup_phase3 = None
    backup_phase2 = None
    p3 = phase4_mod.PHASE3_DIR
    p2 = phase5_mod.PHASE2_DIR

    if p3.exists():
        backup_phase3 = Path(tempfile.mkdtemp(prefix="i524_smoke_p3_backup_"))
        shutil.move(str(p3), str(backup_phase3 / "phase3"))
    if p2.exists():
        backup_phase2 = Path(tempfile.mkdtemp(prefix="i524_smoke_p2_backup_"))
        shutil.move(str(p2), str(backup_phase2 / "phase2"))

    try:
        p3.mkdir(parents=True, exist_ok=True)
        p2.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Generating synthetic Phase 3 clouds (%d contexts × %d layers × %d points)",
            len(contexts),
            len(layers),
            len(points),
        )
        _make_synthetic_clouds(p3, contexts, layers, points)
        logger.info("Generating synthetic Phase 2 ΔG matrix")
        _make_synthetic_dg_matrix(p2, contexts)

        # --- Phase 4 ---
        logger.info("Running Phase 4 predictors (smoke).")
        rc = phase4_mod.main(
            [
                "--layers",
                ",".join(str(layer) for layer in layers),
                "--points",
                ",".join(points),
                "--contexts",
                ",".join(contexts),
                "--log-level",
                "INFO",
            ]
        )
        if rc != 0:
            logger.error("Phase 4 smoke failed rc=%d", rc)
            return 3
        out_pred = phase4_mod.OUT_PATH
        if not out_pred.exists():
            logger.error("Phase 4 did not write %s", out_pred)
            return 3
        with np.load(out_pred) as f:
            n_arrays = len(f.files)
        logger.info("Phase 4 smoke OK: %s (%d matrices)", out_pred, n_arrays)

        # --- Phase 5 ---
        logger.info("Running Phase 5 metrics (smoke, B=16).")
        rc = phase5_mod.main(
            [
                "--b",
                "16",
                "--seed",
                "42",
                "--log-level",
                "INFO",
            ]
        )
        if rc != 0:
            logger.error("Phase 5 smoke failed rc=%d", rc)
            return 3
        out_metrics = phase5_mod.OUT_PATH
        if not out_metrics.exists():
            logger.error("Phase 5 did not write %s", out_metrics)
            return 3
        metrics = json.loads(out_metrics.read_text())
        n_predictors = len(metrics.get("predictor_results", {}))
        logger.info(
            "Phase 5 smoke OK: %s (%d predictors evaluated)",
            out_metrics,
            n_predictors,
        )

        # Final pass: ensure non-trivial outputs (each predictor has a CI).
        for pred_name, pred_result in metrics["predictor_results"].items():
            assert "ci_low" in pred_result, f"{pred_name} missing ci_low"
            assert "ci_high" in pred_result, f"{pred_name} missing ci_high"
            assert "headline_value" in pred_result, f"{pred_name} missing headline_value"

        logger.info("CPU smoke PASSED.")
        return 0
    finally:
        # Restore the real Phase 2/3 dirs (they may have been empty before).
        if p3.exists():
            shutil.rmtree(p3)
        if p2.exists():
            shutil.rmtree(p2)
        if backup_phase3 is not None:
            shutil.move(str(backup_phase3 / "phase3"), str(p3))
            shutil.rmtree(backup_phase3)
        if backup_phase2 is not None:
            shutil.move(str(backup_phase2 / "phase2"), str(p2))
            shutil.rmtree(backup_phase2)


if __name__ == "__main__":
    sys.exit(main())
