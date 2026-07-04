#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, r̂, ρ, M⁺, ×) in scientific docstrings + log messages.
"""Issue #811 — assemble the per-(behavior, layer, summary) cell JSONs into the
side-by-side ``mean`` vs ``turn_nl`` deliverables (plan §6.5).

Reads the per-cell checkpoints ``issue811_fit.py`` wrote under
``eval_results/issue_811/cells/{behavior}_L{li}_{summary}.json`` and emits, PER
SUMMARY (so the analyzer can render #722's Figure-1 analogue with both summaries
per behavior×layer):

- ``function_change_{summary}.json`` — per cell: ``Delta_med``, the three floors,
  ``floor_combined``, ``Delta_med_ci``, support distance, large-shift-excluded Δ,
  and the H_function/H_input/H_mixed call (REUSING #722's ``_h_call``).
- ``chain_rho_M0_Mplus_{summary}.json`` — ρ(M0), ρ(M⁺), the difference, and the
  family-clustered marginal + PAIRED ρ-shift CIs.
- ``cross_transfer_{summary}.json`` — M0→v⁺, M⁺→v⁺, M⁺→v0 transfer cosines.
- ``validity_gate_{summary}.json`` — the base-leg MLP-vs-shuffle validity gate
  (ρ_real / ρ_shuffle / gate_margin per cell) that decides which turn_nl reads
  are trusted (the Phase-0 KILL-1 read + the Phase-2 per-cell gate).

Plus a top-level ``mean_vs_turn_nl_summary.json`` pairing each behavior×layer's
mean vs turn_nl Δ_med/floor + gate margin (the single side-by-side table the
hero figure reads) and echoing the KILL-1 base-leg decision. All JSONs carry
reproducibility metadata (git commit, timestamps). Reuses #722's ``_h_call`` +
``_meta`` verbatim — the only new axis is the summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# DOTENV_LINT_EXEMPT: analysis-phase script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(str(PROJECT_ROOT / ".env"))

import issue722_analyze as an722  # noqa: E402  (reuse _h_call verbatim)

logger = logging.getLogger("issue811.analyze")

HEADLINE_BEHAVIORS = ("em", "sycophancy", "fact")
SWEEP_LAYERS = (7, 14, 21)
PRIMARY_LAYER = 14
SUMMARIES = ("mean", "turn_nl")


def _git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _meta(summary: str | None = None) -> dict:
    m = {
        "issue": 811,
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if summary is not None:
        m["summary"] = summary
    return m


def _cell_summary(cell: dict) -> str:
    """The answer-side summary this cell was fit under (mean | turn_nl)."""
    s = cell.get("summary") or cell.get("metadata", {}).get("summary")
    if s is None:
        raise KeyError(f"cell {cell.get('behavior')}/L{cell.get('layer')} missing 'summary' key")
    return s


def _function_change_row(cell: dict) -> dict:
    h_call, large_shift_flip = an722._h_call(cell)
    return {
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
        "H_call": h_call,
        "large_shift_flip": large_shift_flip,
        "refit_skip": cell.get("refit_skip"),
    }


def assemble(
    cells_dir: Path, out_dir: Path, summary_filename: str = "mean_vs_turn_nl_summary.json"
) -> dict:
    """Read every per-cell JSON, split by summary, write the side-by-side deliverables.

    Summary-agnostic: the per-summary deliverables + the pair table are derived
    from whatever ``summary`` values the cell JSONs carry (the #811 maxp round adds
    a third ``maxp`` column with NO code change here); only the top-level
    summary-doc filename is round-specific (``summary_filename``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_files = sorted(cells_dir.glob("*_L*_*.json"))
    if not cell_files:
        raise FileNotFoundError(f"no per-cell JSONs under {cells_dir} (run issue811_fit first)")
    cells = [json.loads(p.read_text()) for p in cell_files]

    # Split by summary; assemble one set of deliverables per summary.
    by_summary: dict[str, list[dict]] = {}
    for cell in cells:
        by_summary.setdefault(_cell_summary(cell), []).append(cell)

    pair_table: dict[str, dict] = {}
    for summary, scells in sorted(by_summary.items()):
        function_change = {"meta": _meta(summary), "cells": {}}
        chain_rho = {"meta": _meta(summary), "cells": {}}
        cross_transfer = {"meta": _meta(summary), "cells": {}}
        validity_gate = {"meta": _meta(summary), "cells": {}}
        for cell in scells:
            key = f"{cell['behavior']}/L{cell['layer']}"
            function_change["cells"][key] = _function_change_row(cell)
            chain_rho["cells"][key] = {
                "behavior": cell["behavior"],
                "layer": cell["layer"],
                **cell.get("chain_rho", {}),
            }
            cross_transfer["cells"][key] = {
                "behavior": cell["behavior"],
                "layer": cell["layer"],
                **cell.get("cross_transfer", {}),
            }
            validity_gate["cells"][key] = {
                "behavior": cell["behavior"],
                "layer": cell["layer"],
                **cell.get("mlp_validity_gate", {}),
            }
            # Side-by-side pair-table row keyed by behavior/layer, summary column.
            pk = f"{cell['behavior']}/L{cell['layer']}"
            row = pair_table.setdefault(pk, {"behavior": cell["behavior"], "layer": cell["layer"]})
            row[summary] = {
                "Delta_med": cell["Delta_med"],
                "floor_combined": cell["floor_combined"],
                "Delta_over_floor_sd": cell.get("Delta_over_floor_sd"),
                "Delta_med_ci": cell["Delta_med_ci"],
                "H_call": function_change["cells"][key]["H_call"],
                "chain_rho": {
                    "rho_M0_ridge": cell.get("chain_rho", {}).get("rho_M0_ridge"),
                    "rho_Mplus_ridge": cell.get("chain_rho", {}).get("rho_Mplus_ridge"),
                    "ci_diff_ridge": cell.get("chain_rho", {}).get("ci_diff_ridge"),
                },
                "gate_margin": cell.get("mlp_validity_gate", {}).get("gate_margin"),
            }
        (out_dir / f"function_change_{summary}.json").write_text(
            json.dumps(function_change, indent=2, default=float)
        )
        (out_dir / f"chain_rho_M0_Mplus_{summary}.json").write_text(
            json.dumps(chain_rho, indent=2, default=float)
        )
        (out_dir / f"cross_transfer_{summary}.json").write_text(
            json.dumps(cross_transfer, indent=2, default=float)
        )
        (out_dir / f"validity_gate_{summary}.json").write_text(
            json.dumps(validity_gate, indent=2, default=float)
        )
        logger.info("[phase=analyze] wrote %s deliverables (%d cells)", summary, len(scells))

    # Echo the KILL-1 base-leg decision (written by issue811_fit) into the summary.
    kill1_path = out_dir / "kill1_base_leg_validity.json"
    kill1 = json.loads(kill1_path.read_text()) if kill1_path.exists() else None

    summary_doc = {
        "meta": _meta(),
        "summaries": sorted(by_summary),
        "pair_table": pair_table,
        "kill1_base_leg_validity": kill1,
    }
    (out_dir / summary_filename).write_text(json.dumps(summary_doc, indent=2, default=float))
    logger.info("[phase=analyze] wrote %s (%d cells)", summary_filename, len(pair_table))
    return summary_doc


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #811 assemble mean-vs-turn_nl deliverables")
    ap.add_argument("--cells-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_811/cells")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_811")
    ap.add_argument(
        "--summary-filename",
        default="mean_vs_turn_nl_summary.json",
        help="top-level side-by-side table filename (maxp round: "
        "summary_three_summaries.json; default preserves the v1 name)",
    )
    args = ap.parse_args()
    assemble(args.cells_dir, args.out_dir, summary_filename=args.summary_filename)
    logger.info("[phase=analyze] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
