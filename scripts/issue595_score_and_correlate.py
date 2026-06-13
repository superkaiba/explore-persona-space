#!/usr/bin/env python3
"""Issue #595 — Phase 4: scoring + H1 correlation (CPU-only, runs OFF-POD on the VM).

Runs after the pod's Phase 1-3 JSONs are committed (the pod is terminated before
this). Three steps:

1. Copy #545's existing predictor JSONs into eval_results/issue_595/predictors/
   alongside the new PFX predictors (so the race scores PFX against #545's
   geometry/behavior-native families on the SAME held-out cells).
2. Run #545's frozen scoring harness with the PFX group admitted to the
   leave-family-out CV / quarantine race (the 1-line ``groups``-tuple extension
   in scoring.py is the ONLY edit to the frozen #545 harness — H3).
3. Compute H1 Spearman rho between each prefix-binding score (raw all-L, layer-9,
   gauge-normalized squared) and row-summed off-diagonal |L|, with #545's
   family-clustered bootstrap CI (``scoring._family_row_bootstrap``).

Outputs:
  eval_results/issue_595/scoring_prefix/scoring_results.json  (H3 race)
  eval_results/issue_595/prefix_binding_correlation.json      (H1 rho trio)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue595_score_and_correlate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _i595_root() -> Path:
    return PROJECT_ROOT / "eval_results" / "issue_595"


def _i545_root() -> Path:
    return PROJECT_ROOT / "eval_results" / "issue_545"


def copy_545_predictors() -> int:
    """Copy #545's predictor JSONs into issue_595/predictors/ (never overwrite PFX)."""
    src = _i545_root() / "predictors"
    dst = _i595_root() / "predictors"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.json")):
        if p.name.startswith("PFX__"):
            continue  # never clobber the new family
        target = dst / p.name
        if not target.exists():
            shutil.copy2(p, target)
            n += 1
    logger.info("[phase=score] copied %d #545 predictor JSONs into issue_595/predictors/", n)
    return n


def _row_summed_abs_L() -> dict[str, float]:
    """Per row: sum over off-diagonal default-context columns of seed-mean |L|.

    Uses #545's scoring universe (excludes diagonals + non-scoring columns) and
    its seed-mean target convention (saturated / implant_failed cells dropped).
    """
    from explore_persona_space.experiments.behavior_testbed_545.columns import scoring_universe
    from explore_persona_space.experiments.behavior_testbed_545.scoring import _seed_mean_targets

    matrix = json.loads((_i545_root() / "L_matrix.json").read_text())["cells"]
    metadata = json.loads((_i545_root() / "cell_metadata.json").read_text())["cells"]
    targets = _seed_mean_targets(matrix, metadata, include_flagged=False)
    universe = set(scoring_universe())
    per_row: dict[str, float] = {}
    for key, slot in targets.items():
        row, col = key.split("|")
        if (row, col) not in universe:
            continue
        shift = slot.get("shift")
        if shift is None:
            continue
        per_row[row] = per_row.get(row, 0.0) + abs(float(shift))
    return per_row


def compute_h1_correlation() -> dict:
    """Spearman rho(prefix-binding score, row-summed |L|), family-clustered CI, per variant."""
    import numpy as np
    from scipy.stats import spearmanr

    from explore_persona_space.experiments.behavior_testbed_545.rows import ROWS
    from explore_persona_space.experiments.behavior_testbed_545.scoring import (
        _family_row_bootstrap,
    )

    preds_dir = _i595_root() / "predictors"
    row_leak = _row_summed_abs_L()

    variants = {
        "raw_all_L": "PFX__prefix_kv_shift.json",
        "layer_9": "PFX__prefix_kv_shift_L9.json",
        "gaugenorm_sq": "PFX__prefix_kv_shift_gaugenorm_sq.json",
    }
    results: dict = {"n_rows_with_leak": len(row_leak)}
    for label, fname in variants.items():
        path = preds_dir / fname
        if not path.exists():
            results[label] = {"error": f"{fname} missing"}
            continue
        pred = json.loads(path.read_text())
        per_row = pred.get("per_row", {})
        rows = sorted(set(per_row) & set(row_leak))
        if len(rows) < 4:
            results[label] = {"error": f"only {len(rows)} rows with both score + leakage"}
            continue
        score_key = {"raw_all_L": "all_l_mean", "layer_9": "l9", "gaugenorm_sq": "gaugenorm_sq"}[
            label
        ]
        xs = np.array([per_row[r][score_key] for r in rows])
        ys = np.array([row_leak[r] for r in rows])
        rho = float(spearmanr(xs, ys).statistic)

        # Family-clustered bootstrap CI of rho over the rows (resample families ->
        # rows, mirroring scoring._family_row_bootstrap's cell convention with one
        # synthetic "row|__leak" cell per row so the family clustering is honored).
        cell_to_pair = {f"{r}|__leak": (per_row[r][score_key], row_leak[r]) for r in rows}

        def _rho_stat(cells_subset, *, _m=cell_to_pair):
            pairs = [_m[c] for c in cells_subset if c in _m]
            if len({p[0] for p in pairs}) < 4:
                return None
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            s = spearmanr(a, b).statistic
            return float(s) if s == s else None

        # _family_row_bootstrap groups by ROWS[row].family from the "row|col" key.
        boot = _family_row_bootstrap(list(cell_to_pair), _rho_stat)
        results[label] = {
            "spearman_rho": rho,
            "n_rows": len(rows),
            "rows": rows,
            "family_clustered_ci95": boot["ci95"] if boot else None,
            "n_bootstrap_valid": boot["n_valid"] if boot else 0,
            "gauge_normalization_power": pred.get("gauge_normalization_power"),
            "top_quartile_rows": [rows[i] for i in np.argsort(xs)[::-1][: max(1, len(rows) // 4)]],
            "bottom_quartile_rows": [rows[i] for i in np.argsort(xs)[: max(1, len(rows) // 4)]],
        }
        logger.info(
            "[phase=correlate] %s: rho=%.3f (n=%d, CI=%s)",
            label,
            rho,
            len(rows),
            results[label]["family_clustered_ci95"],
        )
    # Ensure ROWS is referenced so the family map is import-validated for bootstrap.
    assert ROWS, "rows registry must be importable for family-clustered bootstrap"
    return results


def score_and_correlate(*, smoke: bool = False) -> Path:
    """Phase 4 entrypoint: copy predictors, run the race, compute H1, write outputs."""
    from explore_persona_space.experiments.behavior_testbed_545.scoring import score

    copy_545_predictors()

    # H3 race (the PFX group is admitted via the 1-line groups-tuple extension in
    # scoring.py). EPM_OUTPUT_ROOT points #545's output_root() at issue_595 so the
    # frozen harness reads issue_595/{predictors,L_matrix,cell_metadata,prereg}.
    # We copy the three frozen #545 inputs alongside the predictors for the run.
    _stage_545_inputs()
    prev = os.environ.get("EPM_OUTPUT_ROOT")
    os.environ["EPM_OUTPUT_ROOT"] = str(_i595_root())
    try:
        score_path = score(
            out_dir_name="scoring_prefix",
            protocol_note="issue595 prefix-binding family added post-hoc",
        )
    finally:
        if prev is None:
            os.environ.pop("EPM_OUTPUT_ROOT", None)
        else:
            os.environ["EPM_OUTPUT_ROOT"] = prev
    logger.info("[phase=score] wrote %s", score_path)

    h1 = compute_h1_correlation()
    out_path = _i595_root() / "prefix_binding_correlation.json"
    out_path.write_text(
        json.dumps({"smoke": smoke, "h1": h1, "h3_race": str(score_path)}, indent=1)
    )
    logger.info("[phase=correlate] wrote %s", out_path)
    return out_path


def _stage_545_inputs() -> None:
    """Stage the three frozen #545 scoring inputs into issue_595/ for the race run.

    score() reads output_root()/{L_matrix,cell_metadata,preregistration}.json; with
    EPM_OUTPUT_ROOT=issue_595 those must be present there. Symlink/copy from #545.
    """
    dst_root = _i595_root()
    for fname in ("L_matrix.json", "cell_metadata.json", "preregistration.json"):
        dst = dst_root / fname
        src = _i545_root() / fname
        if not dst.exists():
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #595 Phase 4 scoring + correlate")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logger.info("[phase=start] issue595 Phase 4 scoring + correlate")
    score_and_correlate(smoke=args.smoke)
    logger.info("[phase=done] issue595 Phase 4 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
