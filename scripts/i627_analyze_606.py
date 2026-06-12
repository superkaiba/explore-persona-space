#!/usr/bin/env python3
"""Task #627 Phase 3 — fraction + dose-curve view of the #606 LoRA-vs-FT slab.

Reads the committed ``eval_results/issue_606/{sycophancy,refusal,
refusal-ft-lr2e6-retrain}/analysis.json`` (per_cell_tables / s_stage_b /
arm_bracket / headline — all verified at Phase 0). Per realized cell:

  - install dial s = committed ``s_stage_b[cell]`` (source own-rate delta,
    the #606 registered dial);
  - per-persona fraction = bystander rate-delta (clean) / s, rate-delta space
    (the judge-family space; margin space does not exist here);
  - bystander-mean fraction; cells with s < 0.10 are FLAGGED high-variance
    (no registered rate-space floor exists — flag, never filter);
  - dose-curve points (s, bystander-mean delta) per arm;
  - endpoint-trio spread: bystander-mean delta at each arm's highest-s cell
    (+ the retrain run), the existing evidence that install is not a
    sufficient statistic (H4 premise);
  - the published matched-install headline carried VERBATIM (H3 consistency
    check — #606 is already a matched design, nothing is re-litigated).

Statistical hygiene (binding): fractions are never correlated against
install; cross-condition reads stay at the published matched comparison.

Output: eval_results/issue_627/analysis/fractions_606.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i627_analyze_606")

OUT_DIR = Path("eval_results/issue_627/analysis")
ROOT_606 = Path("eval_results/issue_606")
BEHAVIORS = ("sycophancy", "refusal", "refusal-ft-lr2e6-retrain")
# The #606 source persona (origin/issue-606:scripts/issue_606/i606_common.py
# SOURCE_PERSONA — the module is branch-only; the constant is re-pinned here
# and asserted against the committed per-cell tables at load).
SOURCE_PERSONA = "software_engineer"
SMALL_DENOM_FLAG = 0.10


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _cell_arm(cell: str) -> str:
    return "base" if cell == "base" else cell.split("_step")[0]


def analyze_behavior(behavior: str) -> dict:
    path = ROOT_606 / behavior / "analysis.json"
    with open(path) as f:
        a = json.load(f)
    tables = a["per_cell_tables"]
    if SOURCE_PERSONA not in tables["base"]:
        raise RuntimeError(f"{behavior}: source {SOURCE_PERSONA} missing from per-cell tables")
    s_stage_b = {c: float(v) for c, v in a["s_stage_b"].items()}
    personas = sorted(tables["base"])
    bystanders = [p for p in personas if p != SOURCE_PERSONA]

    cells_out = {}
    for cell, s in sorted(s_stage_b.items()):
        tab = tables[cell]
        per_persona_delta = {p: float(tab[p]["delta_clean"]) for p in personas}
        bys_mean_delta = float(np.mean([per_persona_delta[p] for p in bystanders]))
        fraction = {p: per_persona_delta[p] / s for p in bystanders} if s != 0 else None
        cells_out[cell] = {
            "arm": _cell_arm(cell),
            "install_s": s,
            "bystander_mean_delta": bys_mean_delta,
            "bystander_mean_fraction": (bys_mean_delta / s) if s != 0 else None,
            "per_persona_fraction": fraction,
            "small_denominator_flag": bool(abs(s) < SMALL_DENOM_FLAG),
            "n_bystanders": len(bystanders),
        }

    # Dose-curve points per arm (descriptive; points within a run are
    # autocorrelated — shape read only, plan §6).
    arms: dict[str, list[dict]] = {}
    for cell, rec in cells_out.items():
        arms.setdefault(rec["arm"], []).append(
            {"cell": cell, "install_s": rec["install_s"], "leak": rec["bystander_mean_delta"]}
        )
    for pts in arms.values():
        pts.sort(key=lambda r: r["install_s"])

    # Endpoint spread: each arm's highest-s cell (H4 premise inputs).
    endpoint_spread = {arm: max(pts, key=lambda r: r["install_s"]) for arm, pts in arms.items()}

    return {
        "behavior": behavior,
        "file": str(path),
        "source_persona": SOURCE_PERSONA,
        "cells": cells_out,
        "dose_curves": arms,
        "endpoint_spread": endpoint_spread,
        "published_headline_verbatim": a["headline"],
        "fraction_space": "rate-delta (judge family); no registered rate-space floor — "
        "small denominators flagged, never filtered",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #627 Phase 3 — #606 fraction + dose-curve view.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    per_behavior = {b: analyze_behavior(b) for b in BEHAVIORS}
    # Cross-run endpoint trio for refusal (lora / ft / retrain-ft endpoints) —
    # the strongest existing "install is not a sufficient statistic" evidence.
    refusal_trio = {
        b: per_behavior[b]["endpoint_spread"] for b in ("refusal", "refusal-ft-lr2e6-retrain")
    }
    result = {
        "issue": 627,
        "family": "lora_vs_ft_606",
        "per_behavior": per_behavior,
        "refusal_endpoint_trio": refusal_trio,
        "hygiene": "fractions compared across conditions at the published matched install "
        "only; never correlated against install",
        "frozen_base_note": "per-persona deltas are vs #606's own fresh base panel "
        "(committed delta_clean); install dial s_stage_b reused verbatim",
        "metadata": {
            "git_commit_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "numpy_version": np.__version__,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fractions_606.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    n_cells = sum(len(v["cells"]) for v in per_behavior.values())
    log.info("[phase=p3_606] -> %s (%d realized cells across 3 behaviors)", out_path, n_cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
