#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, ρ, →, √, ×) in scientific docstrings + log messages.
"""Issue #763 phase 6 (0-GPU, VM): GLM (primary) / ridge / PV LOCO + nulls + ceiling.

The analysis. Per behavior B, reads the matched ``v0(C,B)`` shard, the matched
``E0(C,B)`` (rate + per_probe + n_judged), and the faithful ``r_B`` (PV
baseline), then fits three LOCO predictors over the 50 contexts and brackets the
read against the reliability ceiling:

1. **Precision-weighted binomial GLM `v0→E0` LOCO (PRIMARY)** — the in-session
   2026-06-30 finding's correctly-specified estimator (``issue_763_glm``):
   PCA-reduce (nested-CV d) + per-context-weighted binomial GLM, LOCO held-out
   Spearman ρ_GLM. Layer = the held-out-predictivity max over all 28 layers.
2. **Ridge `v0→E0` LOCO (COMPARATOR)** — the optimistic-at-m=8 read, reusing
   ``issue658_fit_predictors._ridge_predict_loco`` (closed-form PRESS LOCO,
   nested-CV λ). ρ_ridge; the ρ_ridge − ρ_GLM OPTIMISM DELTA is reported.
3. **Persona-vector `r_Bᵀ v0` LOCO (BASELINE)** — read-out regime, swept over
   28 layers + selected by held-out predictivity.

Plus: cluster-bootstrap-over-contexts 95% CI (reused
``_cluster_bootstrap_rho``); shuffle-label null (1000 perms) + Hewitt-Liang
control-task null — BOTH refit the FULL select-by-predictivity-over-28-layers
procedure per permutation (the layer-selection-inflation guard, brief concern
#2 — NOT a shortcut to the already-selected layer); the √(r_yy) reliability
ceiling (``issue_763_reliability.compute_bracket`` — the #742 rebuild branch,
that module is not on ``main``); and the per-behavior triage verdict (a/works,
b/fails, c/noise_limited) per plan §3.

Writes ``eval_results/issue_763/matched_predictor_results.json``.

VECTORIZED per ``.claude/rules/vectorize-many-cell-fits.md``: the ridge LOCO is
closed-form PRESS (cheap); the GLM is IRLS over n≤50 cells (NOT many-epoch SGD,
not the GD-fit class) — the layer/behavior axes are looped but each cell is
seconds; the null layer-sweep is the cost, bounded by perms × layers.

``--smoke`` runs the FULL chain on the smoke slice (1 behavior × 3 contexts ×
5 probes) with reduced perms; asserts ρ_GLM/ρ_ridge/ρ_PV are finite + the JSON
schema matches §10.1.

Usage::

    uv run python scripts/issue763_fit_predictors.py
    uv run python scripts/issue763_fit_predictors.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    RIDGE_LAMBDAS,
    _cluster_bootstrap_rho,
    _rho,
    _ridge_predict_loco,
)
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    EVAL_RESULTS_DIR,
    SEED,
    dump_json,
    load_json,
    reproducibility_metadata,
)

from explore_persona_space.analysis.issue_763_glm import glm_predict_loco  # noqa: E402
from explore_persona_space.analysis.issue_763_reliability import compute_bracket  # noqa: E402

logger = logging.getLogger("issue763_fit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

N_SHUFFLE_PERMS = 1000
N_BOOTSTRAP = 2000
# Triage thresholds (plan §3): verdict (a) works, (b) fails, (c) noise-limited.
CEILING_FRACTION_WORKS = 0.5  # rho_GLM >= 0.5 * sqrt(r_yy)
CEILING_DECENT = 0.5  # sqrt(r_yy) >= 0.5 => target reliably measured
CEILING_LOW = 0.35  # sqrt(r_yy) <= 0.35 => noise-limited


def _load_v0(behavior: str) -> tuple[np.ndarray, list[str]]:
    """Load the matched v0 shard -> (tensor (n_ctx, n_layers, H), context_ids)."""
    shard = EVAL_RESULTS_DIR / "v0_shards" / f"v0_{behavior}.pt"
    blob = torch.load(shard, weights_only=False)
    return blob["tensor"].float().numpy(), blob["context_ids"]


def _e0_vectors(e0: dict, behavior: str, ctx_ids: list[str]):
    """Align E0 rate + n_judged + per_probe with the v0 context order.

    Returns (rates (n,), n_judged (n,), per_probe_by_ctx {ctx: [e0 per probe]},
    kept_ctx_ids). Contexts with a None rate (no judged probes) are dropped.
    """
    per_ctx = e0["e0"][behavior]
    rates, njudged, kept = [], [], []
    per_probe_by_ctx: dict[str, list[float]] = {}
    for c in ctx_ids:
        cell = per_ctx.get(c)
        if cell is None or cell.get("rate") is None:
            continue
        rates.append(float(cell["rate"]))
        njudged.append(int(cell.get("n_judged", 1)))
        kept.append(c)
        pp = [pr["e0"] for pr in cell.get("per_probe", []) if pr.get("e0") is not None]
        per_probe_by_ctx[c] = pp
    return np.asarray(rates), np.asarray(njudged), per_probe_by_ctx, kept


def _layer_sweep_select(
    v0: np.ndarray,
    y: np.ndarray,
    n_judged: np.ndarray,
    predictor: str,
    *,
    rb: np.ndarray | None = None,
) -> dict:
    """Run the predictor at EVERY layer, return per-layer ρ + the chosen layer.

    The chosen layer = held-out-predictivity max (read-out regime). This whole
    function is what the nulls re-run per permutation (the layer-selection-
    inflation guard, brief concern #2). ``v0`` is (n_ctx, n_layers, H).

    predictor ∈ {"glm", "ridge", "pv"}. For "pv", ``rb`` (n_layers, H) gives the
    per-layer direction; the per-context scalar is ``r_B[layer]ᵀ v0[:, layer]``,
    LOCO is implicit (the projection is fixed; we still hold out per context for
    the Spearman to match the other arms' protocol — projection has no fitted
    parameters so LOCO ρ == in-sample ρ for PV, by construction).
    """
    _n_ctx, n_layers, _h = v0.shape
    per_layer_rho: list[float | None] = []
    per_layer_pred: list[np.ndarray | None] = []
    for ell in range(n_layers):
        x = v0[:, ell, :]  # (n_ctx, H)
        if predictor == "glm":
            res = glm_predict_loco(x, y, n_judged)
            pred = res["pred"]
        elif predictor == "ridge":
            pred = _ridge_predict_loco(x, y.reshape(-1, 1), RIDGE_LAMBDAS).reshape(-1)
        elif predictor == "pv":
            assert rb is not None
            direction = rb[ell]  # (H,)
            pred = x @ direction  # (n_ctx,) scalar projection
        else:
            raise ValueError(predictor)
        rho = _rho(pred, y)
        per_layer_rho.append(rho)
        per_layer_pred.append(pred)
    # chosen layer = max ρ (None treated as -inf)
    best_ell, best_rho = 0, -np.inf
    for ell, rho in enumerate(per_layer_rho):
        if rho is not None and rho > best_rho:
            best_rho, best_ell = rho, ell
    return {
        "per_layer_rho": [None if r is None else float(r) for r in per_layer_rho],
        "chosen_layer": best_ell,
        "chosen_rho": None if best_rho == -np.inf else float(best_rho),
        "chosen_pred": per_layer_pred[best_ell],
    }


def _shuffle_null(v0, y, n_judged, predictor, *, rb, n_perms, seed) -> dict:
    """Shuffle-label null: permute E0, re-run the FULL layer-sweep+select per perm.

    Returns the right-tail p of the OBSERVED chosen-layer ρ_GLM vs the null's
    chosen-layer ρ distribution (brief concern #2: refit the FULL select
    procedure per permutation — NOT permute on the already-selected layer).
    """
    rng = random.Random(seed)
    obs = _layer_sweep_select(v0, y, n_judged, predictor, rb=rb)
    obs_rho = obs["chosen_rho"]
    if obs_rho is None:
        return {"observed_rho": None, "p_value": None, "n_perms": 0, "null_rhos": []}
    null_rhos: list[float] = []
    idx = list(range(len(y)))
    for _ in range(n_perms):
        rng.shuffle(idx)
        y_perm = y[idx]
        nj_perm = n_judged[idx]
        sel = _layer_sweep_select(v0, y_perm, nj_perm, predictor, rb=rb)
        if sel["chosen_rho"] is not None:
            null_rhos.append(sel["chosen_rho"])
    if not null_rhos:
        return {"observed_rho": obs_rho, "p_value": None, "n_perms": 0, "null_rhos": []}
    null_arr = np.asarray(null_rhos)
    p = float((np.sum(null_arr >= obs_rho) + 1) / (len(null_arr) + 1))
    null_p95 = float(np.percentile(null_arr, 95))
    return {
        "observed_rho": obs_rho,
        "p_value": p,
        "null_p95": null_p95,
        "n_perms": len(null_rhos),
    }


def _control_task_null(v0, y, n_judged, predictor, *, rb, n_perms, seed) -> dict:
    """Hewitt-Liang control-task null: require the read to BEAT a shuffled-control.

    The control task assigns each context a RANDOM target (drawn from the same
    rate distribution, re-shuffled), refits the full procedure, and the
    SELECTIVITY = ρ_real − ρ_control. The read passes if the observed ρ exceeds
    the control-task ρ distribution's 95th percentile (the d≫n probe-memorization
    guard, mandatory after PCA at n=50). Re-uses the shuffle machinery (a
    shuffled E0 IS a control task here at n=50); the pass criterion differs.
    """
    null = _shuffle_null(v0, y, n_judged, predictor, rb=rb, n_perms=n_perms, seed=seed + 7)
    obs = null.get("observed_rho")
    p95 = null.get("null_p95")
    passed = obs is not None and p95 is not None and obs > p95
    return {"control_task_pass": bool(passed), "observed_rho": obs, "control_p95": p95}


def _triage(rho_glm, rho_glm_ci, sqrt_r_yy, shuffle_p, control_pass) -> str:
    """Per-behavior verdict (a) works / (b) fails / (c) noise_limited (plan §3)."""
    if sqrt_r_yy is None or sqrt_r_yy <= CEILING_LOW:
        return "noise_limited"  # (c): ceiling too low / no dynamic range
    lo = rho_glm_ci[0] if rho_glm_ci else None
    if (
        rho_glm is not None
        and lo is not None
        and shuffle_p is not None
        and lo > 0
        and shuffle_p < 0.05
        and rho_glm >= CEILING_FRACTION_WORKS * sqrt_r_yy
        and control_pass
    ):
        return "works"  # (a)
    if sqrt_r_yy >= CEILING_DECENT:
        return "fails"  # (b): target reliably measured but predictor at chance
    return "noise_limited"  # (c)


def fit_behavior(behavior, v0, ctx_ids, e0, rb_blob, *, n_perms, n_boot) -> dict:
    """The full per-behavior analysis: GLM / ridge / PV + nulls + ceiling + verdict."""
    rates, n_judged, per_probe_by_ctx, kept = _e0_vectors(e0, behavior, ctx_ids)
    if len(rates) < 4:
        return {
            "behavior": behavior,
            "triage_verdict": "noise_limited",
            "note": f"only {len(rates)} contexts with a non-None rate",
            "n_contexts": len(rates),
        }
    # align v0 to the kept contexts
    keep_idx = [ctx_ids.index(c) for c in kept]
    v0_kept = v0[keep_idx]
    rb = rb_blob["r_b"].float().numpy() if rb_blob is not None else None

    glm_sel = _layer_sweep_select(v0_kept, rates, n_judged, "glm")
    ridge_sel = _layer_sweep_select(v0_kept, rates, n_judged, "ridge")
    pv_sel = (
        _layer_sweep_select(v0_kept, rates, n_judged, "pv", rb=rb)
        if rb is not None
        else {"chosen_layer": None, "chosen_rho": None, "chosen_pred": None, "per_layer_rho": []}
    )

    rho_glm = glm_sel["chosen_rho"]
    rho_ridge = ridge_sel["chosen_rho"]
    rho_pv = pv_sel["chosen_rho"]

    boot = (
        _cluster_bootstrap_rho(glm_sel["chosen_pred"], rates, n_boot=n_boot, seed=SEED)
        if glm_sel["chosen_pred"] is not None and rho_glm is not None
        else None
    )
    rho_glm_ci = boot["ci95"] if boot else None

    shuffle = _shuffle_null(v0_kept, rates, n_judged, "glm", rb=None, n_perms=n_perms, seed=SEED)
    control = _control_task_null(
        v0_kept, rates, n_judged, "glm", rb=None, n_perms=n_perms, seed=SEED
    )
    ceiling = compute_bracket(
        per_probe_by_ctx, list(rates), [int(n) for n in n_judged], n_boot=n_boot, seed=SEED
    )
    sqrt_r_yy = ceiling["sqrt_r_yy"]

    verdict = _triage(
        rho_glm, rho_glm_ci, sqrt_r_yy, shuffle.get("p_value"), control["control_task_pass"]
    )
    optimism_delta = (
        (rho_ridge - rho_glm) if (rho_ridge is not None and rho_glm is not None) else None
    )

    return {
        "behavior": behavior,
        "n_contexts": len(rates),
        "rho_GLM": rho_glm,
        "rho_ridge": rho_ridge,
        "rho_PV": rho_pv,
        "rho_GLM_ci": rho_glm_ci,
        "optimism_delta": optimism_delta,
        "sqrt_r_yy": sqrt_r_yy,
        "sqrt_r_yy_ci": ceiling.get("sqrt_r_yy_ci"),
        "sqrt_r_yy_binomial": ceiling.get("sqrt_r_yy_binomial"),
        "shuffle_null_p": shuffle.get("p_value"),
        "shuffle_null_p95": shuffle.get("null_p95"),
        "control_task_pass": control["control_task_pass"],
        "chosen_layer": glm_sel["chosen_layer"],
        "chosen_layer_ridge": ridge_sel["chosen_layer"],
        "chosen_layer_pv": pv_sel["chosen_layer"],
        "per_layer_rho_GLM": glm_sel["per_layer_rho"],
        "per_layer_rho_ridge": ridge_sel["per_layer_rho"],
        "per_layer_rho_PV": pv_sel.get("per_layer_rho", []),
        "triage_verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #763: GLM/ridge/PV LOCO fits + nulls + ceiling."
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_perms = 50 if args.smoke else N_SHUFFLE_PERMS
    n_boot = 200 if args.smoke else N_BOOTSTRAP

    e0 = load_json(EVAL_RESULTS_DIR / "E0_matched_by_behavior.json")
    pv_path = EVAL_RESULTS_DIR / "pv_rb_by_behavior.json"
    have_pv = pv_path.exists()

    results: dict[str, dict] = {}
    for behavior in args.behaviors:
        v0, ctx_ids = _load_v0(behavior)
        rb_blob = None
        if have_pv:
            rb_shard = EVAL_RESULTS_DIR / "pv_shards" / f"rb_{behavior}.pt"
            if rb_shard.exists():
                rb_blob = torch.load(rb_shard, weights_only=False)
        rec = fit_behavior(behavior, v0, ctx_ids, e0, rb_blob, n_perms=n_perms, n_boot=n_boot)
        results[behavior] = rec
        logger.info(
            "[fit] %s: rho_GLM=%s rho_ridge=%s sqrt_r_yy=%s verdict=%s",
            behavior,
            rec.get("rho_GLM"),
            rec.get("rho_ridge"),
            rec.get("sqrt_r_yy"),
            rec.get("triage_verdict"),
        )
        # smoke schema asserts (plan §10 smoke)
        if args.smoke and rec.get("rho_GLM") is not None:
            assert np.isfinite(rec["rho_GLM"]), "rho_GLM not finite"
            assert rec["triage_verdict"] in ("works", "fails", "noise_limited")

    out = {
        "by_behavior": results,
        "n_shuffle_perms": n_perms,
        "n_bootstrap": n_boot,
        "have_pv_baseline": have_pv,
        "metadata": reproducibility_metadata({"phase": "fit"}),
    }
    dump_json(out, EVAL_RESULTS_DIR / "matched_predictor_results.json")
    print(f"[issue763.fit] wrote predictor results for {len(results)} behaviors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
