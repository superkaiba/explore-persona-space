#!/usr/bin/env python
# ruff: noqa: RUF002, RUF003
# Intentional scientific Unicode (Σ, ρ, δ, ŵ, ×, −, ⁻¹, ᵀ, ‖) in docstrings/comments.
"""issue #666 Phase 4 — designed-null install-leak CONTROL arm (Must-Fix 2, plan §4d/§6/§6.5).

Runs the SAME predictor pipeline as ``issue666_predictor.py`` over the 2
install-matched, signal-free #664 designed-null cells (``ic_edu_default`` /
``tf_rev_default``), scoring each null cell's L̂ Spearman ρ vs Δs + its
family-clustered 95% CI. Pre-registered verdict (§6): a real content behavior's
L̂ ρ MUST EXCEED the designed-null ρ with non-overlapping clustered CIs for the
geometry-win headline; an overlap → "install-confounded" (L̂ tracks
install-displacement magnitude, not theory geometry).

Output: ``eval_results/issue_666/headline/designed_null_Lhat_rho.json`` (the
install-confound gate on the headline; plan §6.5 primary_deliverable).

CPU-only; reuses the shared ``issue666_predictor`` scorer + clustered bootstrap.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_results" / "issue_666" / "headline"


def score_designed_nulls(*, layer=None, n_boot=2000, seed=0) -> dict:
    """Score both designed-null cells' L̂ ρ vs Δs + family-clustered CIs.

    Returns ``{cell: {rho, ci_lo, ci_hi, n_bystanders}}`` for the 2 null cells.
    """
    import issue666_load_store as loader
    import issue666_predictor as pred

    out: dict = {}
    for cell in pred.DESIGNED_NULL_CELLS:
        local_dir = loader.download_cell(cell)
        loaded = loader.load_cell(local_dir)
        lyr = pred.PRIMARY_LAYER if layer is None else layer
        lyr = min(lyr, loaded["v_plus"].shape[1] - 1)
        Sigma_inv, _ = pred._battery_sigma_inv(loaded, lyr)
        rec = pred.predict_cell(loaded, cell=cell, layer=lyr, Sigma_inv=Sigma_inv)
        pb = rec["per_bystander"]
        lh = np.array(pb["Lhat"])
        ds = np.array(pb["ds"])
        fams = np.array(pb["context_family"])
        rho = rec["rho_full_Lhat"]
        lo, hi = pred.clustered_bootstrap_ci(
            lh, ds, clusters=fams, n_boot=n_boot, seed=seed, statistic="spearman"
        )
        out[cell] = {
            "rho": rho,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n_bystanders": rec["n_bystanders"],
            "behavior": rec["behavior"],
            "layer": lyr,
        }
        del loaded
        gc.collect()
        with contextlib.suppress(OSError):
            os.remove(local_dir / "tensors.pt")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="issue 666 designed-null control arm.")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--slice", action="store_true", help="tiny smoke slice (fewer bootstraps)")
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    n_boot = 200 if args.slice else args.n_boot
    nulls = score_designed_nulls(layer=args.layer, n_boot=n_boot)
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {
        "designed_null_cells": list(nulls),
        "per_null": nulls,
        "verdict_rule": (
            "a real content behavior's L_hat rho must EXCEED the designed-null rho "
            "with non-overlapping clustered CIs for the geometry-win headline (plan §6)"
        ),
    }
    (OUT / "designed_null_Lhat_rho.json").write_text(json.dumps(rec, indent=1))
    for c, r in nulls.items():
        print(f"[designed-null] {c}: rho={r['rho']:.3f} CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
    print("[phase=designed_null] done OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
