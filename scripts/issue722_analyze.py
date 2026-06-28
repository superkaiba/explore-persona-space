#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, r̂, ρ, M⁺) in scientific docstrings + log messages.
"""Issue #722 — assemble the per-(behavior, layer) cell JSONs into the 4 deliverables.

Reads the per-cell checkpoints written by ``issue722_fit_M.py`` under
``eval_results/issue_722/cells/`` and emits the plan §6.5 primary deliverables:

- ``function_change.json`` — per cell: ``Delta_med``, the three floors,
  ``floor_combined``, ``Delta_med_ci`` (clustered_bootstrap_scalar), support
  distance, large-shift-excluded Δ, and the H_function/H_input/H_mixed call.
- ``chain_rho.json`` — ρ(M0), ρ(M⁺), the difference, family-clustered CIs
  (clustered_bootstrap_spearman), MLP + shuffle reads.
- ``cross_transfer.json`` — M0→v⁺, M⁺→v⁺, M⁺→v0 transfer cosines per cell.
- ``nonlinearity_gap.json`` — (ρ_MLP − ρ_ridge) under M0 vs M⁺ per cell.

Also applies the §3 kill criterion at L=14 across the headline behaviors and the
#697 cross-reference panel (N/A when #697's f_CV artifact is absent). All four
JSONs carry reproducibility metadata (git commit, timestamps).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue722.analyze")

HEADLINE_BEHAVIORS = ("em", "sycophancy", "fact")
PRIMARY_LAYER = 14


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _meta() -> dict:
    return {
        "issue": 722,
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _h_call(cell: dict) -> str:
    """H_function / H_input / H_mixed per plan §3, from the Δ_med CI vs floor_combined."""
    ci = cell.get("Delta_med_ci", {})
    lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
    floor = cell.get("floor_combined")
    if lo is None or hi is None or floor is None:
        return "undetermined"
    if lo > floor:
        return "H_function"
    if hi <= floor:
        return "H_input"
    return "H_mixed"


def _load_697_fcv() -> dict | None:
    """Per-behavior f_CV from #697 if present (local eval_results or HF), else None."""
    local = PROJECT_ROOT / "eval_results/issue_697/patch"
    if local.is_dir():
        for fname in ("f_cv.json", "fcv.json", "summary.json"):
            p = local / fname
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except Exception as e:
                    logger.warning("could not parse #697 %s (%s)", p, e)
    return None


def assemble(cells_dir: Path, out_dir: Path) -> dict:
    """Read every per-cell JSON and write the 4 deliverables. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_files = sorted(cells_dir.glob("*_L*.json"))
    if not cell_files:
        raise FileNotFoundError(f"no per-cell JSONs under {cells_dir} (run issue722_fit_M first)")
    cells = [json.loads(p.read_text()) for p in cell_files]

    function_change = {"meta": _meta(), "cells": {}}
    chain_rho = {"meta": _meta(), "cells": {}}
    cross_transfer = {"meta": _meta(), "cells": {}}
    nonlinearity_gap = {"meta": _meta(), "cells": {}}

    for cell in cells:
        key = f"{cell['behavior']}/L{cell['layer']}"
        function_change["cells"][key] = {
            "behavior": cell["behavior"],
            "layer": cell["layer"],
            "n_cells": cell["n_cells"],
            "Delta_med": cell["Delta_med"],
            "Delta_med_ci": cell["Delta_med_ci"],
            "Delta_med_mean_ci": cell.get("Delta_med_mean_ci"),
            "Delta_med_excl_large_shift_ci": cell.get("Delta_med_excl_large_shift_ci"),
            "floor_M0_refit": cell["floor_M0_refit"],
            "floor_Mplus_refit": cell["floor_Mplus_refit"],
            "floor_shifted": cell["floor_shifted"],
            "floor_combined": cell["floor_combined"],
            "floor_sd_combined": cell.get("floor_sd_combined"),
            "Delta_over_floor_sd": cell.get("Delta_over_floor_sd"),
            "support_distance": cell.get("support_distance"),
            "n_families": cell.get("n_families"),
            "h_call": _h_call(cell),
        }
        cb = cell.get("chain_rho", {})
        chain_rho["cells"][key] = {
            "behavior": cell["behavior"],
            "layer": cell["layer"],
            "n_with_E": cb.get("n_with_E"),
            "rho_M0_ridge": cb.get("rho_M0_ridge"),
            "rho_Mplus_ridge": cb.get("rho_Mplus_ridge"),
            "rho_diff_ridge": cb.get("rho_diff_ridge"),
            "ci_M0_ridge": cb.get("ci_M0_ridge"),
            "ci_Mplus_ridge": cb.get("ci_Mplus_ridge"),
            "rho_M0_mlp": cb.get("rho_M0_mlp"),
            "rho_Mplus_mlp": cb.get("rho_Mplus_mlp"),
            "rho_M0_shuffle": cb.get("rho_M0_shuffle"),
        }
        cross_transfer["cells"][key] = {
            "behavior": cell["behavior"],
            "layer": cell["layer"],
            **cell.get("cross_transfer", {}),
        }
        nonlinearity_gap["cells"][key] = {
            "behavior": cell["behavior"],
            "layer": cell["layer"],
            "nonlin_gap_M0": cb.get("nonlin_gap_M0"),
            "nonlin_gap_Mplus": cb.get("nonlin_gap_Mplus"),
        }

    # ---- §3 kill criterion at L=14 across headline behaviors ----
    straddle_count = 0
    evaluated = 0
    for behavior in HEADLINE_BEHAVIORS:
        k = f"{behavior}/L{PRIMARY_LAYER}"
        if k not in function_change["cells"]:
            continue
        evaluated += 1
        if function_change["cells"][k]["h_call"] in ("H_input", "H_mixed", "undetermined"):
            straddle_count += 1
    function_change["kill_criterion"] = {
        "primary_layer": PRIMARY_LAYER,
        "behaviors_evaluated": evaluated,
        "behaviors_not_above_floor": straddle_count,
        "inconclusive": evaluated > 0 and straddle_count >= 2,
        "note": (
            "inconclusive iff Δ_med CI straddles/sits-below floor_combined for >=2 of the "
            "3 headline behaviors at L=14 (plan §3)"
        ),
    }

    # ---- #697 cross-reference panel (non-gating) ----
    fcv = _load_697_fcv()
    function_change["cross_ref_697"] = (
        {"status": "available", "f_cv": fcv}
        if fcv is not None
        else {"status": "N/A — #697 artifact absent at analysis time"}
    )

    (out_dir / "function_change.json").write_text(
        json.dumps(function_change, indent=2, default=float)
    )
    (out_dir / "chain_rho.json").write_text(json.dumps(chain_rho, indent=2, default=float))
    (out_dir / "cross_transfer.json").write_text(
        json.dumps(cross_transfer, indent=2, default=float)
    )
    (out_dir / "nonlinearity_gap.json").write_text(
        json.dumps(nonlinearity_gap, indent=2, default=float)
    )
    logger.info(
        "[phase=analyze] wrote 4 deliverables (%d cells); kill_inconclusive=%s",
        len(cells),
        function_change["kill_criterion"]["inconclusive"],
    )
    return {"n_cells": len(cells), "kill": function_change["kill_criterion"]}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #722 assemble fit cells → 4 deliverables")
    ap.add_argument("--cells-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_722/cells")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_722")
    args = ap.parse_args()
    logger.info("[phase=analyze] cells=%s out=%s", args.cells_dir, args.out_dir)
    assemble(args.cells_dir, args.out_dir)
    logger.info("[phase=analyze] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
