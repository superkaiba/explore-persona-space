#!/usr/bin/env python3
"""Issue #640 — Phase 3: scoring + paired postfix-vs-prefix comparison.

CPU-only; runs OFF-POD on the VM after the pod's Phase 1-2 JSONs are committed
(the pod terminates before this). Three steps:

1. **Paired comparison (PRIMARY, H1):** load the per-seed postfix Δleakage cells
   (``eval_results/issue_640/patch_cells_postfix_seed{seed}.json``) and #595's
   committed prefix Δleakage (``PFX__patch_recovery.json``, materialized into
   ``eval_results/issue_640/_inputs/``). Per cell: postfix Δ vs prefix Δ at
   seed-0; median postfix Δ; sign-agreement count (cells where postfix Δ >
   prefix Δ); a sign test (binomial) + Wilcoxon signed-rank statistic over the
   8 paired differences. Also the seed-0 vs seed-137 directional-consistency
   count. Writes ``patch_comparison.json``.
2. **H2 correlation (secondary, exploratory):** Spearman rho between the
   postfix-KV-shift score and #545's row-summed off-diagonal |L| (same target +
   family-clustered CI as #595's H1), plus rho between postfix-KV-shift and the
   postfix delta-leakage across cells. Writes ``postfix_binding_correlation.json``.
3. **Predictor race (H2 scaffolding):** admit the PST group to #545's frozen
   scoring harness via the 1-line ``groups``-tuple extension (PFX + PST), so the
   postfix-binding family enters the leave-family-out CV race alongside #545's
   A/B/C/D + #595's PFX. Writes ``scoring_postfix/scoring_results.json``.

The prefix-patch baseline is NEVER re-run — it is #595's committed data, read
from the issue-640-local ``_inputs/`` snapshot so the worktree is self-contained
(the #595 eval JSONs live ONLY on origin/issue-595, not main).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue640_score_and_compare")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PHASE2_ROWS: tuple[str, ...] = (
    "bad_medical",
    "risky_financial",
    "extreme_sports",
    "taught_fact",
    "reversed_fact",
    "compliment_writing",
    "wrong_claim_agreement",
    "marker",
)

# The three #595-branch eval JSONs the comparison + race depend on. They live
# ONLY on origin/issue-595 (not main); materialize them into the issue-640
# worktree's _inputs/ so the worktree is self-contained (brief requirement #2).
I595_INPUT_FILES: tuple[str, ...] = (
    "eval_results/issue_595/predictors/PFX__patch_recovery.json",
    "eval_results/issue_595/PFX_ctrl_postfix.json",
    "eval_results/issue_595/L_matrix.json",
)
I595_SOURCE_REF = "origin/issue-595"


def _i640_root() -> Path:
    return PROJECT_ROOT / "eval_results" / "issue_640"


def _i545_root() -> Path:
    return PROJECT_ROOT / "eval_results" / "issue_545"


def _inputs_dir() -> Path:
    return _i640_root() / "_inputs"


def materialize_595_inputs() -> dict[str, Path]:
    """Materialize the #595-branch eval JSONs into issue_640/_inputs/.

    The files live ONLY on origin/issue-595. If a local copy already exists
    under _inputs/ (committed for self-containment) use it; otherwise pull it
    from the git object via ``git show`` and write it locally. Fail loud if a
    required input cannot be resolved.
    """
    dst_dir = _inputs_dir()
    dst_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for rel in I595_INPUT_FILES:
        name = Path(rel).name
        dst = dst_dir / name
        if not dst.exists():
            blob = subprocess.run(
                ["git", "show", f"{I595_SOURCE_REF}:{rel}"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env={**os.environ},
            )
            if blob.returncode != 0:
                raise FileNotFoundError(
                    f"could not materialize {rel} from {I595_SOURCE_REF}: {blob.stderr.strip()}. "
                    "The #595 eval JSONs must be reachable (cherry-pick or git fetch origin "
                    "issue-595). The paired postfix-vs-prefix comparison cannot run without them."
                )
            dst.write_text(blob.stdout)
            logger.info("[phase=inputs] materialized %s -> %s", rel, dst)
        resolved[name] = dst
    return resolved


def _load_postfix_cells() -> dict[int, dict[str, float]]:
    """Load per-seed postfix Δleakage cells from the Phase-2 JSONs.

    Returns {seed: {row|column: delta_leakage}}. Fails loud if no seed file is
    present (the comparison has no left-hand side).
    """
    root = _i640_root()
    per_seed: dict[int, dict[str, float]] = {}
    for path in sorted(root.glob("patch_cells_postfix_seed*.json")):
        data = json.loads(path.read_text())
        per_seed[int(data["seed"])] = dict(data["cells"])
    if not per_seed:
        raise FileNotFoundError(
            f"no patch_cells_postfix_seed*.json under {root} — Phase 2 produced no postfix "
            "cells; nothing to compare against the prefix baseline."
        )
    return per_seed


def compute_paired_comparison() -> dict:
    """H1: paired postfix-vs-prefix Δleakage per cell + median + sign test.

    Reads the postfix cells (this run) and #595's prefix cells. The H1 pass
    criterion: median postfix Δ (seed-0) > 0 AND postfix Δ > prefix Δ on ≥5/8
    cells. Reports the binomial sign-test p and the Wilcoxon signed-rank
    statistic over the 8 paired differences (no CI claim at n=8).
    """
    import numpy as np
    from scipy.stats import binomtest, wilcoxon

    inputs = materialize_595_inputs()
    prefix = json.loads(inputs["PFX__patch_recovery.json"].read_text())["cells"]
    postfix_by_seed = _load_postfix_cells()

    out: dict = {
        "n_prefix_cells": len(prefix),
        "seeds_present": sorted(postfix_by_seed),
    }

    # Seed-0 paired comparison (primary).
    if 0 not in postfix_by_seed:
        out["error"] = "seed-0 postfix cells absent — primary H1 comparison cannot run"
        return out
    postfix0 = postfix_by_seed[0]
    paired_cells = sorted(set(prefix) & set(postfix0))
    rows = []
    diffs = []
    postfix_deltas = []
    n_postfix_gt_prefix = 0
    for cell in paired_cells:
        pst = float(postfix0[cell])
        pfx = float(prefix[cell])
        diff = pst - pfx
        better = pst > pfx
        n_postfix_gt_prefix += int(better)
        rows.append(
            {
                "cell": cell,
                "postfix_delta": pst,
                "prefix_delta": pfx,
                "postfix_minus_prefix": diff,
                "postfix_better": better,
            }
        )
        diffs.append(diff)
        postfix_deltas.append(pst)

    n = len(paired_cells)
    median_postfix = float(np.median(postfix_deltas)) if postfix_deltas else None
    sign_test_p = binomtest(n_postfix_gt_prefix, n, 0.5).pvalue if n else None
    # Wilcoxon needs at least one non-zero diff; guard the degenerate case.
    nonzero = [d for d in diffs if d != 0.0]
    if len(nonzero) >= 1:
        wstat = wilcoxon(diffs, zero_method="zsplit")
        wilcoxon_stat, wilcoxon_p = float(wstat.statistic), float(wstat.pvalue)
    else:
        wilcoxon_stat, wilcoxon_p = None, None

    h1_pass = (
        median_postfix is not None and median_postfix > 0 and n_postfix_gt_prefix >= 5 and n == 8
    )

    out["seed0"] = {
        "n_paired_cells": n,
        "cells": rows,
        "median_postfix_delta": median_postfix,
        "n_postfix_gt_prefix": n_postfix_gt_prefix,
        "sign_test_p": sign_test_p,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "h1_pass": h1_pass,
        "h1_criterion": "median postfix Δ (seed-0) > 0 AND postfix>prefix on >=5/8 cells",
    }

    # Seed-0 vs seed-137 directional consistency.
    if 137 in postfix_by_seed:
        postfix137 = postfix_by_seed[137]
        shared = sorted(set(postfix0) & set(postfix137))
        same_sign = sum(1 for c in shared if (postfix0[c] > 0) == (postfix137[c] > 0))
        out["seed_consistency"] = {
            "n_shared_cells": len(shared),
            "n_same_sign": same_sign,
            "per_cell": {c: {"seed0": postfix0[c], "seed137": postfix137[c]} for c in shared},
        }
    return out


def _row_summed_abs_L() -> dict[str, float]:
    """Per row: sum over off-diagonal default-context columns of seed-mean |L|.

    Same procedure as #595's compute_h1_correlation target (uses #545's frozen
    scoring universe + seed-mean target convention). Reads #545's L_matrix +
    cell_metadata from main.
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


def compute_h2_correlation() -> dict:
    """H2 (exploratory): Spearman rho for the postfix-KV-shift predictor.

    Two correlations: (a) postfix-KV-shift vs #545 row-summed |L| (same target
    as #595's H1; family-clustered CI), (b) postfix-KV-shift vs postfix Δleakage
    across cells (does the carrier-strength scalar rank the cells that respond
    to patching?). Exploratory — no pre-registered threshold.
    """
    import numpy as np
    from scipy.stats import spearmanr

    from explore_persona_space.experiments.behavior_testbed_545.rows import ROWS
    from explore_persona_space.experiments.behavior_testbed_545.scoring import (
        _family_row_bootstrap,
    )

    preds_path = _i640_root() / "predictors" / "PST__postfix_kv_shift.json"
    results: dict = {}
    if not preds_path.exists():
        results["error"] = "PST__postfix_kv_shift.json missing — Phase 1 did not run"
        return results

    pred = json.loads(preds_path.read_text())
    per_row = pred.get("per_row", {})
    row_leak = _row_summed_abs_L()

    # (a) postfix-KV-shift vs row-summed |L|.
    rows_a = sorted(set(per_row) & set(row_leak))
    if len(rows_a) >= 4:
        xs = np.array([per_row[r]["all_l_mean"] for r in rows_a])
        ys = np.array([row_leak[r] for r in rows_a])
        rho = float(spearmanr(xs, ys).statistic)
        cell_to_pair = {f"{r}|__leak": (per_row[r]["all_l_mean"], row_leak[r]) for r in rows_a}

        def _rho_stat(cells_subset, *, _m=cell_to_pair):
            pairs = [_m[c] for c in cells_subset if c in _m]
            if len({p[0] for p in pairs}) < 4:
                return None
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            s = spearmanr(a, b).statistic
            return float(s) if s == s else None

        boot = _family_row_bootstrap(list(cell_to_pair), _rho_stat)
        results["postfix_kv_shift_vs_row_leak"] = {
            "spearman_rho": rho,
            "n_rows": len(rows_a),
            "rows": rows_a,
            "family_clustered_ci95": boot["ci95"] if boot else None,
            "n_bootstrap_valid": boot["n_valid"] if boot else 0,
        }
        logger.info(
            "[phase=correlate] postfix-KV-shift vs row-summed |L|: rho=%.3f (n=%d, CI=%s)",
            rho,
            len(rows_a),
            results["postfix_kv_shift_vs_row_leak"]["family_clustered_ci95"],
        )
    else:
        results["postfix_kv_shift_vs_row_leak"] = {
            "error": f"only {len(rows_a)} rows with both score + leakage"
        }

    # (b) postfix-KV-shift vs postfix Δleakage (does the scalar rank patch-responders?).
    postfix_by_seed = _load_postfix_cells()
    if 0 in postfix_by_seed:
        # Map cell key row|col -> delta; collapse to per-row by the cell's row.
        per_row_delta: dict[str, float] = {}
        for cell, delta in postfix_by_seed[0].items():
            row = cell.split("|")[0]
            per_row_delta[row] = float(delta)
        rows_b = sorted(set(per_row) & set(per_row_delta))
        if len(rows_b) >= 4:
            xs = np.array([per_row[r]["all_l_mean"] for r in rows_b])
            ys = np.array([per_row_delta[r] for r in rows_b])
            rho_b = float(spearmanr(xs, ys).statistic)
            results["postfix_kv_shift_vs_postfix_delta"] = {
                "spearman_rho": rho_b,
                "n_rows": len(rows_b),
                "rows": rows_b,
            }
            logger.info(
                "[phase=correlate] postfix-KV-shift vs postfix Δleakage: rho=%.3f (n=%d)",
                rho_b,
                len(rows_b),
            )
        else:
            results["postfix_kv_shift_vs_postfix_delta"] = {
                "error": f"only {len(rows_b)} rows with both score + postfix delta"
            }
    assert ROWS, "rows registry must be importable for family-clustered bootstrap"
    return results


def _stage_545_inputs_for_race() -> None:
    """Stage #545's frozen scoring inputs + predictors into issue_640/ for the race.

    score() reads output_root()/{L_matrix,cell_metadata,preregistration}.json +
    predictors/; with EPM_OUTPUT_ROOT=issue_640 those must be present there.
    Copies #545's geometry/behavior-native predictors (never the PST family) +
    the three frozen inputs. The PST predictor written by Phase 1 already lives
    in issue_640/predictors/.
    """
    dst_root = _i640_root()
    dst_preds = dst_root / "predictors"
    dst_preds.mkdir(parents=True, exist_ok=True)
    for fname in ("L_matrix.json", "cell_metadata.json", "preregistration.json"):
        dst = dst_root / fname
        if not dst.exists():
            shutil.copy2(_i545_root() / fname, dst)
    n = 0
    for p in sorted((_i545_root() / "predictors").glob("*.json")):
        if p.name.startswith("PST__"):
            continue  # never clobber the new family
        target = dst_preds / p.name
        if not target.exists():
            shutil.copy2(p, target)
            n += 1
    logger.info("[phase=score] staged %d #545 predictor JSONs into issue_640/predictors/", n)


def run_predictor_race() -> Path:
    """Run #545's frozen scoring harness with the PST group admitted.

    The PST family is admitted via the 1-line groups-tuple extension in
    scoring.py (("A","B","C","D","PFX","PST")). The race scores PST against
    #545's A/B/C/D families on the SAME held-out cells.
    """
    from explore_persona_space.experiments.behavior_testbed_545.scoring import score

    _stage_545_inputs_for_race()
    prev = os.environ.get("EPM_OUTPUT_ROOT")
    os.environ["EPM_OUTPUT_ROOT"] = str(_i640_root())
    try:
        score_path = score(
            out_dir_name="scoring_postfix",
            protocol_note="issue640 postfix-binding family added post-hoc",
        )
    finally:
        if prev is None:
            os.environ.pop("EPM_OUTPUT_ROOT", None)
        else:
            os.environ["EPM_OUTPUT_ROOT"] = prev
    logger.info("[phase=score] wrote %s", score_path)
    return score_path


def score_and_compare(*, smoke: bool = False) -> Path:
    """Phase 3 entrypoint: paired comparison (H1) + H2 correlation + predictor race."""
    comparison = compute_paired_comparison()
    comp_path = _i640_root() / "patch_comparison.json"
    comp_path.write_text(json.dumps({"smoke": smoke, "comparison": comparison}, indent=1))
    logger.info("[phase=compare] wrote %s", comp_path)

    h2 = compute_h2_correlation()
    score_path = run_predictor_race()
    corr_path = _i640_root() / "postfix_binding_correlation.json"
    corr_path.write_text(
        json.dumps({"smoke": smoke, "h2": h2, "predictor_race": str(score_path)}, indent=1)
    )
    logger.info("[phase=correlate] wrote %s", corr_path)
    return comp_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #640 Phase 3 scoring + comparison")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logger.info("[phase=start] issue640 Phase 3 scoring + comparison")
    score_and_compare(smoke=args.smoke)
    logger.info("[phase=done] issue640 Phase 3 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
