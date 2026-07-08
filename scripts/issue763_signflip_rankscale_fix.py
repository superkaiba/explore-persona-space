#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (ρ, Δ, −) in scientific docstrings + printed labels.
"""Issue #763 cofit round: rank-scale CORRECTION of the kernel-vs-linear sign-flip test.

The as-run sign-flip (`nonlinear_tests.json` → `signflip`) compared per-context
squared errors of predictions living on DIFFERENT scales: the ridge column's
persisted `preds_chosen_layer` are the raw shrunk linear-predictor values
(sd ≈ 0.01, centered near 0) while the kernel-ridge predictions live on the
[0, 1] rank scale. Against the rank target (mean 0.5) the linear column's
squared error is then dominated by the LOCATION OFFSET — mean err_lin ≈ 1/3 =
E[U²] for U~Uniform(0,1) — so the as-run p = 1e-4 on all five behaviors
detects the scale mismatch, not a kernel benefit. The plan (§4.3.1) registered
the errors "on the rank scale", and the min-detectable-Δρ simulation harness
(`signflip_min_detectable_delta_rho`) rank-transforms BOTH prediction vectors
before forming errors; this script applies that same convention to the
production predictions and re-runs the identical sign-flip machinery.

Writes ``eval_results/issue_763/neutral-contrast-and-cofit/
signflip_rankscale_corrected.json``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# #847: shared-VM thread caps must bind BEFORE torch/numpy freeze their pools at import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
from issue763_common import COFIT_DIR, load_json  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from explore_persona_space.analysis.issue_763_nonlinear import (  # noqa: E402
    paired_signflip_test,
)

SEED = 763


def rank01(v: np.ndarray) -> np.ndarray:
    """Average ranks scaled to [0, 1] (the co-fit target convention)."""
    n = v.shape[0]
    return (rankdata(v) - 1) / (n - 1)


def main() -> int:
    """Recompute the paired sign-flip with both prediction sets rank-transformed."""
    results = load_json(COFIT_DIR / "cofit_results.json")
    nl = load_json(COFIT_DIR / "nonlinear_tests.json")
    out: dict = {
        "round": "neutral-contrast-and-cofit",
        "note": (
            "Rank-scale CORRECTION of the kernel-vs-linear paired sign-flip. The as-run "
            "test in nonlinear_tests.json compared squared errors of predictions on "
            "different scales (ridge: raw shrunk linear predictor, sd~0.01; KRR: rank "
            "scale), so its p = 1e-4 on all five behaviors is a location/scale artifact "
            "(mean err_lin ~ 1/3 = E[U^2]). Here BOTH prediction vectors are "
            "rank-transformed to [0,1] before forming per-context squared errors, "
            "matching the plan's rank-scale registration and the "
            "signflip_min_detectable_delta_rho simulation harness convention."
        ),
        "n_flips": 10_000,
        "by_behavior": {},
        "generated_utc": datetime.now(UTC).isoformat(),
    }
    for behavior, rec in results["by_behavior"].items():
        kept = rec["kept_context_ids"]
        y = np.array([rec["graded_by_context"][c] for c in kept], dtype=float)
        y_rank = rank01(y)
        methods = rec["methods"]
        pl_map = methods["cofit_ridge"].get("preds_chosen_layer")
        pk_map = methods["cofit_krr"].get("preds_chosen_layer")
        if not pl_map or not pk_map:
            out["by_behavior"][behavior] = {"skipped": "missing chosen-layer predictions"}
            continue
        pl = rank01(np.array([pl_map[c] for c in kept], dtype=float))
        pk = rank01(np.array([pk_map[c] for c in kept], dtype=float))
        err_lin = (pl - y_rank) ** 2
        err_krr = (pk - y_rank) ** 2
        res = paired_signflip_test(
            err_lin, err_krr, n_flips=10_000, rng=np.random.default_rng(SEED + 11)
        )
        as_run = nl["by_behavior"].get(behavior, {}).get("signflip", {})
        out["by_behavior"][behavior] = {
            "signflip_rankscale": res,
            "as_run_signflip_scale_contaminated": as_run,
            "paired_delta_rho_krr_minus_ridge": nl["by_behavior"]
            .get(behavior, {})
            .get("paired_delta_rho_krr_minus_ridge"),
        }
        print(
            f"{behavior}: corrected stat={res['statistic_mean_err_diff']:+.5f} "
            f"p={res['p_value']:.4g} (as-run p={as_run.get('p_value')})"
        )
    out_path = COFIT_DIR / "signflip_rankscale_corrected.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
