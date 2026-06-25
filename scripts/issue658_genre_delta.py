#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, Δ, →, ※) in scientific docstrings, strings + messages.
"""Issue #658 (G1) A2 (off-pod CPU): per-behavior genre delta Δρ + 95% Δρ CI.

The (G1) headline read. Compares the Betley misalignment-specific arm against the
length-matched UltraChat generic arm PER BEHAVIOR and answers: is the base-model
leakage-predictor chain a property of the model's context→behavior geometry, or an
artifact of the behavior-eliciting query genre (plan v3 §6/§6.5)?

For each behavior with dynamic range on BOTH pools it computes::

    Δρ = ρ_UltraChat − ρ_Betley

at the per-arm BEST (layer, summary recipe) cell, and a 95% Δρ CI from an
INDEPENDENT cluster bootstrap (the two arms have DISJOINT probe sets, so no paired
resampling is possible — plan v3 §6 / Codex Statistics REVISE):

  - read each arm's per-cell context-clustered bootstrap ρ DRAWS (the ``draws`` key
    ``issue658_fit_predictors._cluster_bootstrap_rho`` now emits for the best cell);
  - take ≥2000 INDEPENDENT bootstrap draws PER ARM (the v1 §11 2000-resample
    setting; the fit emits 2000 — or the smoke clamp's 200);
  - difference the i-th UltraChat draw with the i-th Betley draw (independent
    draws, NOT a paired resample), forming the Δρ bootstrap distribution;
  - report the 2.5/97.5 percentiles + a ``null_overlap`` flag.

The headline genre-bound-vs-geometry call reads off this Δρ CI (NOT the per-arm
noise-floor comparison alone — a per-arm-floor read can flip the verdict on noise
because the joint Δρ uncertainty can be wider than either arm's per-arm band).
Each arm's per-arm within-genre noise-floor 95th-pct band is ALSO recorded.

Verdicts (plan v3 §6/§6.5/§7):
  - H1-consistent: the Δρ CI OVERLAPS ZERO (genre delta not distinguishable from
    the noise floor under this design — consistent with geometry, NOT proven equal).
  - H2 / genre-bound: the Δρ CI lies ENTIRELY BELOW ZERO (UltraChat ρ reliably
    below Betley ρ) AND the UltraChat ρ falls below its own within-genre noise
    floor while the Betley ρ clears its floor.
  - H3-no-dynamic-range: a behavior without dynamic range on BOTH pools — no Δρ CI
    computed (``delta_rho_ci: null``), the per-pool testable variance recorded
    instead, never a zero/point-estimate Δρ.

GPU-FREE by construction (reads the two arms' already-uploaded eval JSONs). Runs
OFF-POD on the VM once both arms' ``aggregate.json`` + ``a32_cells.json`` +
``E0_expression.json`` exist; fail-loud + report "Betley arm pending" if the
Betley inputs are absent (plan v3 §6 statistical-input existence).

Usage::

    uv run python scripts/issue658_genre_delta.py \\
        --betley-dir eval_results/issue_658 \\
        --ultrachat-dir eval_results/issue_658/genre-generalization-ultrachat \\
        --out eval_results/issue_658/genre-generalization-ultrachat/genre_delta.json

    # smoke (reads the two _smoke arms' JSONs):
    uv run python scripts/issue658_genre_delta.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

# Cross-script helpers hoisted to module top (gotchas.md #606 — a missing symbol
# crashes at process start, never inside a smoke-skipped branch).
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_common import (  # noqa: E402
    EVAL_RESULTS_DIR,
    dump_json,
    load_json,
)

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue658_genre_delta")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# A behavior has "dynamic range" on a pool when its per-context E0 target varies
# above this std floor (matches the _rho / noise_floor degenerate guard in
# issue658_fit_predictors: std < 1e-9 → no rank signal). Set slightly above the
# numeric floor so an all-equal saturated/floored behavior is flagged H3.
DYNAMIC_RANGE_STD_MIN = 1e-6
DELTA_CI_SEED = 658
# Registered ≥2000 INDEPENDENT resamples per arm (plan v3 §6/§6.5/§11; mirrors
# issue658_fit_predictors.N_BOOTSTRAP). Every dynamic-range cell's Δρ CI must use
# at least this many draws.
MIN_RESAMPLES_PRODUCTION = 2000
# Smoke clamp: the fit's --smoke run sets N_BOOTSTRAP=200, so the smoke arms emit
# 200 draws. The smoke Δρ floor matches so the smoke runs end-to-end without
# --smoke-allow-small-bootstrap; production is NEVER allowed to drop to this.
MIN_RESAMPLES_SMOKE = 200


def _e0_per_context_values(e0: dict, column_id: str) -> list[float]:
    """The per-context E0 scalar (PRIMARY rate, or marker logp_mean) for a column.

    Mirrors issue658_fit_predictors.e0_target's value selection so the
    dynamic-range read here matches what the predictor actually fit.
    """
    vals: list[float] = []
    for cell in e0.get("e0", {}).values():
        v = cell.get(column_id)
        if v is None:
            continue
        x = v.get("rate")
        if x is None:
            x = v.get("logp_mean")  # marker column
        if x is None:
            continue
        vals.append(float(x))
    return vals


def _testable_variance(e0: dict, column_id: str) -> dict:
    """Per-pool testable variance for one behavior (the floor-guard read)."""
    vals = _e0_per_context_values(e0, column_id)
    if len(vals) < 2:
        return {"n": len(vals), "std": None, "var": None, "dynamic_range": False}
    arr = np.asarray(vals, dtype=np.float64)
    std = float(arr.std())
    return {
        "n": len(vals),
        "std": std,
        "var": float(arr.var()),
        "dynamic_range": std >= DYNAMIC_RANGE_STD_MIN,
    }


def _best_cell_draws(agg: dict, a32_cells: list[dict], column_id: str) -> dict | None:
    """The per-arm best (layer, summary) cell's ρ + its context-clustered draws.

    Reads the best (layer, recipe) from ``aggregate.json::a32_verdicts`` then pulls
    that exact cell's ``bootstrap.draws`` from ``a32_cells.json`` (the additive key
    _cluster_bootstrap_rho now emits). Returns None when the behavior has no scored
    cell on this arm (low dynamic range) or the chosen cell lacks bootstrap draws.
    """
    verdict = agg.get("a32_verdicts", {}).get(column_id)
    if not verdict or verdict.get("best_rho") is None:
        return None
    best_layer = verdict.get("best_layer")
    best_summary = verdict.get("best_summary")
    for c in a32_cells:
        if (
            c.get("column") == column_id
            and c.get("layer") == best_layer
            and c.get("recipe") == best_summary
        ):
            boot = c.get("bootstrap") or {}
            draws = boot.get("draws")
            return {
                "rho": verdict["best_rho"],
                "layer": best_layer,
                "recipe": best_summary,
                "draws": draws,  # may be None on a tiny cell
                "noise_floor_p95": verdict.get("noise_floor_p95"),
            }
    return None


def _delta_rho_ci(
    uc_draws: list[float],
    betley_draws: list[float],
    *,
    seed: int,
    min_resamples: int,
    behavior: str,
) -> dict:
    """95% Δρ CI from an INDEPENDENT cluster bootstrap of the two arms' ρ draws.

    The two arms have DISJOINT probes, so no paired resampling is possible. Each
    arm's ``draws`` is its own context-clustered bootstrap distribution of the
    best-cell ρ. We form the Δρ distribution by pairing the i-th draw of each arm
    INDEPENDENTLY (after a deterministic shuffle of each arm so draw order carries
    no spurious correlation), to ``min(len)`` draws. Returns the 2.5/97.5
    percentiles + ``null_overlap``.

    The registered procedure (plan v3 §6/§6.5/§11) is ≥2000 INDEPENDENT resamples
    PER ARM. ``min_resamples`` is that floor (2000 in production, the smoke clamp's
    value otherwise — passed by the caller). If the per-index pairing would draw
    fewer than ``min_resamples`` Δρ samples (e.g. a degenerate-resample drop left an
    arm's ``draws`` short) this RAISES rather than silently bootstrapping the Δρ CI
    from too few draws — the smoke path's ``--smoke-allow-small-bootstrap`` flag is
    the only way to relax the floor, and it must be EXPLICIT (never a permissive
    default).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(uc_draws, dtype=np.float64)
    b = np.asarray(betley_draws, dtype=np.float64)
    m = int(min(len(a), len(b)))
    if m < min_resamples:
        raise ValueError(
            f"Δρ CI for behavior '{behavior}': only {m} paired bootstrap draws "
            f"available (UltraChat arm {len(a)}, Betley arm {len(b)}) but the "
            f"registered procedure requires ≥{min_resamples} INDEPENDENT resamples "
            "per arm (plan v3 §6/§6.5/§11). A dynamic-range cell must carry the full "
            "bootstrap — a degenerate-resample drop in issue658_fit_predictors."
            "_cluster_bootstrap_rho left an arm's draws short. Re-run the fit so "
            "both arms emit ≥min_resamples draws, or (smoke ONLY) pass "
            "--smoke-allow-small-bootstrap to relax this floor."
        )
    # Independent draws → independent shuffle of each arm before the per-index diff
    # (the diff of two independent bootstrap distributions; not a paired resample).
    rng.shuffle(a)
    rng.shuffle(b)
    diff = a[:m] - b[:m]
    lower = float(np.percentile(diff, 2.5))
    upper = float(np.percentile(diff, 97.5))
    return {
        "lower": lower,
        "upper": upper,
        "n_resamples": m,
        "null_overlap": bool(lower <= 0.0 <= upper),
    }


def _verdict(
    delta_ci: dict,
    rho_uc: float,
    rho_betley: float,
    uc_floor: float | None,
    betley_floor: float | None,
) -> str:
    """H1-consistent / H2 (genre-bound) from the Δρ CI + the per-arm floors.

    H2 requires BOTH (a) the Δρ CI entirely below zero AND (b) the UltraChat ρ
    below its own floor while the Betley ρ clears its floor (plan v3 §6/§7).
    Otherwise H1-consistent (the CI overlaps zero, or is positive, or the floor
    condition is unmet).
    """
    ci_below_zero = delta_ci["upper"] < 0.0
    uc_below_floor = uc_floor is not None and rho_uc < uc_floor
    betley_clears_floor = betley_floor is None or rho_betley > betley_floor
    if ci_below_zero and uc_below_floor and betley_clears_floor:
        return "H2-genre-bound"
    return "H1-consistent"


def compute_genre_delta(
    betley_dir: Path,
    ultrachat_dir: Path,
    *,
    smoke: bool,
    seed: int = DELTA_CI_SEED,
    allow_small_bootstrap: bool = False,
) -> dict:
    """Per-behavior Δρ + 95% Δρ CI table over both arms (plan v3 §6.5 schema).

    Every behavior with dynamic range on BOTH pools MUST carry a
    ``delta_rho_ci {lower, upper, n_resamples, null_overlap}`` built from
    ≥``min_resamples`` INDEPENDENT bootstrap draws per arm (plan v3 §6.5); only
    H3 (no dynamic range on both pools) rows carry ``delta_rho_ci: null``. A
    dynamic-range cell whose best (layer, summary) cell is missing or whose
    bootstrap ``draws`` are absent / shorter than ``min_resamples`` RAISES — never
    a silent ``delta_rho_ci: null``. ``min_resamples`` is ``MIN_RESAMPLES_PRODUCTION``
    (2000) outside smoke; ``MIN_RESAMPLES_SMOKE`` in smoke; and may be relaxed ONLY
    when ``allow_small_bootstrap`` is set (smoke-only escape hatch).
    """
    if allow_small_bootstrap and not smoke:
        raise ValueError(
            "allow_small_bootstrap is a SMOKE-ONLY escape hatch — it cannot be set "
            "outside --smoke (production requires ≥2000 resamples per arm)."
        )
    min_resamples = (
        1 if allow_small_bootstrap else (MIN_RESAMPLES_SMOKE if smoke else MIN_RESAMPLES_PRODUCTION)
    )
    suffix = "_smoke" if smoke else ""
    b_agg_p = betley_dir / f"aggregate{suffix}.json"
    b_cells_p = betley_dir / f"a32_cells{suffix}.json"
    b_e0_p = betley_dir / f"E0_expression{suffix}.json"
    # The Betley arm aggregate is the same-issue dependency — fail loud if absent
    # (plan v3 §6: report "Betley arm pending", never a silent skip).
    for p in (b_agg_p, b_cells_p, b_e0_p):
        if not p.is_file():
            raise FileNotFoundError(
                f"Betley arm input missing: {p} — the genre delta is 'Betley arm pending'; "
                "re-run once the Betley arm's aggregate/a32_cells/E0 JSONs exist."
            )
    u_agg_p = ultrachat_dir / f"aggregate{suffix}.json"
    u_cells_p = ultrachat_dir / f"a32_cells{suffix}.json"
    u_e0_p = ultrachat_dir / f"E0_expression{suffix}.json"
    for p in (u_agg_p, u_cells_p, u_e0_p):
        if not p.is_file():
            raise FileNotFoundError(f"UltraChat arm input missing: {p}")

    b_agg = load_json(b_agg_p)
    b_cells = load_json(b_cells_p)["a32"]
    b_e0 = load_json(b_e0_p)
    u_agg = load_json(u_agg_p)
    u_cells = load_json(u_cells_p)["a32"]
    u_e0 = load_json(u_e0_p)

    # The behavior set is the intersection of the two arms' scored columns.
    columns = sorted(set(b_e0.get("columns", [])) & set(u_e0.get("columns", [])))
    rows: list[dict] = []
    variance_table: list[dict] = []
    for col in columns:
        b_var = _testable_variance(b_e0, col)
        u_var = _testable_variance(u_e0, col)
        variance_table.append(
            {
                "behavior": col,
                "betley_std": b_var["std"],
                "ultrachat_std": u_var["std"],
                "betley_dynamic_range": b_var["dynamic_range"],
                "ultrachat_dynamic_range": u_var["dynamic_range"],
            }
        )
        both_dyn = b_var["dynamic_range"] and u_var["dynamic_range"]
        b_best = _best_cell_draws(b_agg, b_cells, col)
        u_best = _best_cell_draws(u_agg, u_cells, col)
        # Legitimate H3: NOT dynamic-range on both pools → no Δρ CI computed (plan
        # v3 §6.5: "floored behaviors flagged N/A"; only H3 rows carry null CI).
        if not both_dyn:
            rows.append(
                {
                    "behavior": col,
                    "verdict": "H3-no-dynamic-range",
                    "delta_rho": None,
                    "delta_rho_ci": None,
                    "rho_ultrachat": (u_best or {}).get("rho"),
                    "rho_betley": (b_best or {}).get("rho"),
                    "uc_dynamic_range": u_var["dynamic_range"],
                    "betley_dynamic_range": b_var["dynamic_range"],
                    "uc_testable_std": u_var["std"],
                    "betley_testable_std": b_var["std"],
                }
            )
            continue
        # CONTRACT (plan v3 §6.5): a dynamic-range cell (both arms dynamic-range)
        # MUST carry a full Δρ CI. A missing best (layer, summary) cell, or a best
        # cell without bootstrap draws, is an upstream contract violation — fail
        # loud with the offending arm + cause + cached input path, NEVER a silent
        # delta_rho_ci: null. (Removed the legacy `delta-ci-unavailable-no-draws`
        # verdict — Codex r1 BLOCKER + concern genre-delta-ci-contract-unenforced.)
        if b_best is None:
            raise RuntimeError(
                f"genre-delta contract violation for dynamic-range behavior '{col}': "
                f"the Betley arm has dynamic range (std={b_var['std']}) but no scored "
                f"best (layer, summary) cell in aggregate.json::a32_verdicts. A "
                "dynamic-range cell must carry a Δρ CI (plan v3 §6.5). Re-run the "
                f"Betley fit so the behavior has a scored best cell. Cached input: {b_cells_p}"
            )
        if u_best is None:
            raise RuntimeError(
                f"genre-delta contract violation for dynamic-range behavior '{col}': "
                f"the UltraChat arm has dynamic range (std={u_var['std']}) but no scored "
                f"best (layer, summary) cell in aggregate.json::a32_verdicts. A "
                "dynamic-range cell must carry a Δρ CI (plan v3 §6.5). Re-run the "
                f"UltraChat fit so the behavior has a scored best cell. Cached input: {u_cells_p}"
            )
        rho_uc = u_best["rho"]
        rho_betley = b_best["rho"]
        delta_rho = rho_uc - rho_betley
        # Both arms must carry bootstrap draws to form the independent Δρ CI; a
        # dynamic-range cell missing draws is the upstream _cluster_bootstrap_rho
        # contract violation — fail loud (cause: draws missing), never a null CI.
        if not u_best.get("draws"):
            raise RuntimeError(
                f"genre-delta contract violation for dynamic-range behavior '{col}': "
                f"the UltraChat arm's best cell (layer {u_best['layer']}, recipe "
                f"{u_best['recipe']}) has NO bootstrap draws (cause: draws missing). A "
                "dynamic-range cell must carry a ≥2000-resample Δρ CI (plan v3 §6.5); "
                "issue658_fit_predictors._cluster_bootstrap_rho should have emitted "
                f"draws for this n>=4 cell. Re-run the UltraChat fit. Cached input: {u_cells_p}"
            )
        if not b_best.get("draws"):
            raise RuntimeError(
                f"genre-delta contract violation for dynamic-range behavior '{col}': "
                f"the Betley arm's best cell (layer {b_best['layer']}, recipe "
                f"{b_best['recipe']}) has NO bootstrap draws (cause: draws missing). A "
                "dynamic-range cell must carry a ≥2000-resample Δρ CI (plan v3 §6.5); "
                "issue658_fit_predictors._cluster_bootstrap_rho should have emitted "
                f"draws for this n>=4 cell. Re-run the Betley fit. Cached input: {b_cells_p}"
            )
        # The ≥min_resamples (cause: draws too short) floor is enforced inside
        # _delta_rho_ci, which raises naming the behavior + per-arm draw counts.
        delta_ci = _delta_rho_ci(
            u_best["draws"],
            b_best["draws"],
            seed=seed,
            min_resamples=min_resamples,
            behavior=col,
        )
        verdict = _verdict(
            delta_ci,
            rho_uc,
            rho_betley,
            u_best.get("noise_floor_p95"),
            b_best.get("noise_floor_p95"),
        )
        rows.append(
            {
                "behavior": col,
                "layer_uc": u_best["layer"],
                "recipe_uc": u_best["recipe"],
                "layer_betley": b_best["layer"],
                "recipe_betley": b_best["recipe"],
                "rho_ultrachat": rho_uc,
                "rho_betley": rho_betley,
                "delta_rho": delta_rho,
                "delta_rho_ci": delta_ci,
                "uc_noise_floor_95": u_best.get("noise_floor_p95"),
                "betley_noise_floor_95": b_best.get("noise_floor_p95"),
                "uc_dynamic_range": True,
                "betley_dynamic_range": True,
                "verdict": verdict,
            }
        )
    return {
        "rows": rows,
        "per_pool_testable_variance": variance_table,
        "n_behaviors_compared": sum(1 for r in rows if r.get("delta_rho_ci") is not None),
        "n_behaviors_h3": sum(1 for r in rows if r["verdict"] == "H3-no-dynamic-range"),
        "delta_ci_seed": seed,
        "dynamic_range_std_min": DYNAMIC_RANGE_STD_MIN,
        "method": (
            "Δρ = ρ_UltraChat − ρ_Betley at each arm's best (layer, summary); 95% Δρ CI from an "
            "INDEPENDENT cluster bootstrap of the two arms' per-cell context-clustered ρ draws "
            "(disjoint probes → no paired resampling). Headline reads off the Δρ CI, NOT the "
            "per-arm noise-floor comparison alone."
        ),
        "metadata": reproducibility_metadata({"script": "issue658_genre_delta"}),
    }


def make_figure(result: dict, out_dir: Path) -> str | None:
    """Per-behavior Betley-vs-UltraChat ρ paired bars + the 95% Δρ CI whisker.

    The (G1) hero candidate (plan v3 §6 Figures): floored behaviors are labeled
    'N/A — no dynamic range', never a zero bar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in result["rows"] if r.get("delta_rho_ci") is not None]
    if not rows:
        logger.warning("no behaviors with dynamic range on both pools — skipping the figure")
        return None
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    behs = [r["behavior"] for r in rows]
    rho_b = [r["rho_betley"] for r in rows]
    rho_u = [r["rho_ultrachat"] for r in rows]
    x = np.arange(len(behs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(behs)), 4.5))
    ax.bar(x - w / 2, rho_b, w, label="Betley ρ")
    ax.bar(x + w / 2, rho_u, w, label="UltraChat ρ")
    # Δρ CI whisker centered on the UltraChat bar (the headline read).
    for i, r in enumerate(rows):
        ci = r["delta_rho_ci"]
        lo = r["rho_betley"] + ci["lower"]
        hi = r["rho_betley"] + ci["upper"]
        ax.plot([x[i] + w / 2, x[i] + w / 2], [lo, hi], color="black", lw=1.4)
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(behs, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("held-out Spearman ρ (best layer/summary per genre)")
    ax.set_title("Genre delta: Betley vs UltraChat predictability (95% Δρ CI)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = fig_dir / "genre_delta.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #658 (G1) A2: per-behavior genre delta Δρ.")
    parser.add_argument(
        "--betley-dir",
        type=Path,
        default=EVAL_RESULTS_DIR,
        help="dir holding the Betley arm's aggregate.json / a32_cells.json / E0_expression.json",
    )
    parser.add_argument(
        "--ultrachat-dir",
        type=Path,
        default=EVAL_RESULTS_DIR / "genre-generalization-ultrachat",
        help="dir holding the UltraChat arm's aggregate.json / a32_cells.json / E0_expression.json",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-allow-small-bootstrap",
        action="store_true",
        help="SMOKE-ONLY: relax the ≥2000-resample Δρ CI floor (plan v3 §6/§6.5/§11) "
        "to accept arms with fewer than the registered draw count. Requires --smoke; "
        "production NEVER relaxes the floor. For tiny test fixtures only.",
    )
    parser.add_argument("--no-figure", action="store_true", help="skip the matplotlib figure")
    args = parser.parse_args()

    if args.smoke_allow_small_bootstrap and not args.smoke:
        parser.error("--smoke-allow-small-bootstrap requires --smoke (production keeps ≥2000)")

    ultra_dir = args.ultrachat_dir
    out = args.out or (ultra_dir / ("genre_delta_smoke.json" if args.smoke else "genre_delta.json"))
    result = compute_genre_delta(
        args.betley_dir,
        ultra_dir,
        smoke=args.smoke,
        allow_small_bootstrap=args.smoke_allow_small_bootstrap,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(result, out)
    logger.info(
        "genre delta: %d behaviors with Δρ CI, %d H3 (no dynamic range) -> %s",
        result["n_behaviors_compared"],
        result["n_behaviors_h3"],
        out,
    )
    if not args.no_figure:
        fig = make_figure(result, ultra_dir)
        if fig:
            logger.info("genre-delta figure -> %s", fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
