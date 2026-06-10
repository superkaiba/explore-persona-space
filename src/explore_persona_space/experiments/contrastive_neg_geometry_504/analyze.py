# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + × + − intentional
"""Task #504 Phase 2 — CPU-only pooled partial-Spearman regression + diagnostics.

Plan §4.4 / §6: across pooled (probe × arm × seed) rows from the 4 POSITIONED
arms — N = 4 arms × ~50 probes × 2 seeds ≈ 400 rows — fit:

    ΔG = β0 + β_dsource · d_source
            + β_dnn      · d_nearest_neg_nd
            + β_shadow   · shadow_angle
            + β_baseprior· base_prior_marker
            + β_step     · training_step(cell)
            + β_source_dg· source_delta_g(cell)
            + ε

Uses **partial Spearman ρ** for each predictor with the others partialled out
(NOT OLS coefficients — robust to the heavy-tailed log-prob distribution; #472
analyze.py:partial_spearman_count_given_implant uses the same machinery).

This module ONLY computes diagnostics; the analyzer agent assigns the
Bubble/Barrier/Both/Indeterminate verdict + HIGH/MODERATE/LOW confidence by
weighing them JOINTLY per the statistical-framing rule. NO hard p-threshold →
verdict ladder lives here (plan §3 + §6.3 — the v1 ladder was the reconciler
must-fix).

Reads:
    - eval_results/issue_504/<cell>_seed<S>/trajectory.json  (per-cell trajectory)
    - eval_results/issue_504/phase0_calibration.json         (chosen_ckpt_frac)
    - eval_results/issue_504/phase0_5_gates.json             (per-probe covariates)

Writes:
    - eval_results/issue_504/analyze_summary.json
    - figures/issue_504/<panel-figure>.{pdf, png, meta.json}  (deferred to analyzer)
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
    _spearman,
    holm_correction,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    EMISSION_BAND_HIGH,
    EMISSION_BAND_LOW,
    POSITIONED_ARM_SLUGS,
    SOURCE_DG_BAND_HIGH,
    SOURCE_DG_BAND_LOW,
)

log = logging.getLogger("issue_504.analyze")

# Predictor names — used in regression + Holm correction. Order matters for
# reporting only (Holm is unordered); plan §4.4 Step 3 lists 6 predictors.
PREDICTORS: tuple[str, ...] = (
    "d_source",
    "d_nearest_neg_nd",
    "shadow_angle",
    "base_prior_marker",
    "training_step",
    "source_delta_g",
)


def _residualize(y: list[float], X: np.ndarray) -> np.ndarray:
    """Return y − OLS_fit(y ~ X) residuals.

    X must be a 2-D matrix with rows == len(y); a column of 1's is prepended
    here as the intercept. Used by `_partial_spearman`: rank-residualize the
    target predictor against the other predictors, then Spearman against the
    residualized DV.
    """
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if Xa.ndim != 2 or Xa.shape[0] != ya.shape[0]:
        raise ValueError(
            f"_residualize: shape mismatch X={Xa.shape}, y={ya.shape}; "
            f"need 2D X with n_rows==len(y)"
        )
    # Add intercept column.
    ones = np.ones((Xa.shape[0], 1), dtype=np.float64)
    Xb = np.concatenate([ones, Xa], axis=1)
    # OLS: β = (XᵀX)⁻¹ Xᵀ y; residuals = y − Xβ.
    try:
        beta, *_ = np.linalg.lstsq(Xb, ya, rcond=None)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"_residualize lstsq failed: {e}") from e
    return ya - Xb @ beta


def _partial_spearman(
    y: list[float],
    target: list[float],
    others: list[list[float]],
) -> float:
    """Partial Spearman ρ of y on `target`, partialling out all `others`.

    Algorithm: rank-residualize BOTH y and target against `others` (Spearman
    on residuals is the standard partial-Spearman definition; #472's
    `analyze.partial_spearman_count_given_implant` uses the same approach).
    """
    n = len(y)
    if n != len(target):
        raise ValueError(f"_partial_spearman: len(y)={n} != len(target)={len(target)}")
    for i, o in enumerate(others):
        if len(o) != n:
            raise ValueError(f"_partial_spearman: len(others[{i}])={len(o)} != n={n}")
    if not others:
        return _spearman(target, y)
    others_arr = np.asarray(others, dtype=np.float64).T  # shape (n, k)
    y_resid = _residualize(y, others_arr)
    t_resid = _residualize(target, others_arr)
    return _spearman(t_resid.tolist(), y_resid.tolist())


def _spearman_pvalue(rho: float, n: int) -> float:
    """Approximate p-value for Spearman ρ via Fisher z-transform (two-sided).

    For n >= ~10, atanh(ρ) is approximately normal with σ = 1/sqrt(n-3).

    **Round-2 fix (blocker #3):** an UNDERPOWERED sample (``n <= 3``) returns
    ``1.0`` (no evidence). But a PERFECT correlation (``abs(rho) >= 1.0``)
    with adequate ``n`` is the STRONGEST evidence of association, not the
    weakest — it now returns a near-zero p-value (1e-12) so Holm correction
    treats it correctly. The previous joint branch ``n <= 3 or abs(rho) >=
    1.0 -> 1.0`` silently corrupted the headline analysis (any perfect
    diagnostic relationship got reported as non-significant).
    """
    if n <= 3:
        return 1.0
    if abs(rho) >= 1.0:
        # Perfect (anti-)correlation with adequate n: STRONGEST evidence.
        # Return near-zero so Holm-correction is meaningful (scipy.stats's
        # spearmanr returns 0.0 in this case, but we floor it to 1e-12 to
        # keep the value strictly positive and well-behaved under log /
        # min-p comparisons).
        return 1e-12
    z = math.atanh(rho) * math.sqrt(n - 3)
    # Two-sided normal: p = 2 * (1 - Φ(|z|)).
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2:
        return float("nan")
    sx = xa.std()
    sy = ya.std()
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def load_trajectory(slab_root: Path, cell_slug: str, seed: int) -> dict | None:
    """Load <slab_root>/<cell>_seed<S>/trajectory.json or return None if absent."""
    p = slab_root / f"{cell_slug}_seed{seed}" / "trajectory.json"
    if not p.exists():
        log.warning("[load] trajectory missing for %s seed=%s at %s", cell_slug, seed, p)
        return None
    return json.loads(p.read_text())


def aggregate_base_prior_from_trajectories(
    *,
    slab_root: Path,
    seeds: Sequence[int],
    positioned_arm_slugs: Sequence[str] = POSITIONED_ARM_SLUGS,
) -> dict[str, float]:
    """Round-2 fix (blocker #2): aggregate the per-probe base-model marker prior.

    Reads every (cell × seed) trajectory.json under ``slab_root``, pulls the
    ``b_logp`` values from EVERY checkpoint × probe × question (the base-model
    log-prob of the marker at the post-response slot — by construction the same
    across checkpoints for a given (probe, q) since the base model is frozen),
    and returns ``{probe: mean(b_logp over (cell, seed, ckpt, q))}``.

    This is the ``base_prior_marker`` covariate that the #500 sign-flip
    discipline (plan §6.2 test 6) reads via ``--base-prior-path``. Without
    this aggregation the analyzer's covariate is constant 0.0 and the partial
    Spearman degenerates (the column has no variance), silently disabling the
    sign-flip robustness check.

    Returns an empty dict if no trajectories exist (caller falls back to the
    0.0 placeholder, with a logged warning).

    Round-2 v2-slug fix (BLOCKER #1, concern_id `analyze-v2-slug-iteration`):
    `positioned_arm_slugs` selects which 4-arm set to iterate. Defaults to v1
    (``POSITIONED_ARM_SLUGS``) so legacy callers stay byte-identical; the v2
    dispatcher / CLI threads ``POSITIONED_ARM_SLUGS_V2`` so trajectories at
    ``<slab_root>/c504v2_<arm>_seed<S>/trajectory.json`` are read.
    """
    per_probe_acc: dict[str, list[float]] = {}
    n_traj = 0
    for cell in positioned_arm_slugs:
        for seed in seeds:
            traj = load_trajectory(slab_root, cell, seed)
            if traj is None:
                continue
            n_traj += 1
            for ck in traj.get("checkpoints", []):
                held = ck.get("held_out", {}) or {}
                for probe, per_q in held.items():
                    for q_entry in per_q.values():
                        bl = q_entry.get("b_logp")
                        if bl is None:
                            continue
                        if not math.isfinite(float(bl)):
                            continue
                        per_probe_acc.setdefault(probe, []).append(float(bl))
    out: dict[str, float] = {
        probe: float(np.mean(vals)) for probe, vals in per_probe_acc.items() if vals
    }
    log.info(
        "[base_prior] aggregated b_logp over %d trajectories → %d probes (sample stats: "
        "min=%.3f max=%.3f var=%.4f).",
        n_traj,
        len(out),
        min(out.values()) if out else float("nan"),
        max(out.values()) if out else float("nan"),
        float(np.var(list(out.values()))) if out else float("nan"),
    )
    return out


def write_base_prior_marker(base_prior_by_probe: dict[str, float], out_path: Path) -> Path:
    """Persist the aggregated per-probe base-prior map (round-2 fix, blocker #2)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base_prior_by_probe, indent=2))
    log.info(
        "[base_prior] wrote %s (n_probes=%d, variance=%.4f)",
        out_path,
        len(base_prior_by_probe),
        float(np.var(list(base_prior_by_probe.values()))) if base_prior_by_probe else 0.0,
    )
    return out_path


def _pick_checkpoint_at_frac(traj: dict, chosen_frac: float, tol: float = 1e-6) -> dict | None:
    """Return the checkpoint whose `frac` is closest to `chosen_frac` (within tol)."""
    cks = traj.get("checkpoints", [])
    if not cks:
        return None
    best = min(cks, key=lambda c: abs(c["frac"] - chosen_frac))
    if abs(best["frac"] - chosen_frac) > tol:
        log.warning(
            "[pick_ckpt] requested frac=%.4f, closest available=%.4f (delta=%.4f) — "
            "exceeds tol=%g; returning anyway, but this is a config drift",
            chosen_frac,
            best["frac"],
            abs(best["frac"] - chosen_frac),
            tol,
        )
    return best


def build_rows(
    *,
    slab_root: Path,
    chosen_frac: float,
    per_probe: dict[str, dict[str, Any]],
    arm_to_positioned_n: dict[str, str],
    seeds: Sequence[int],
    base_prior_by_probe: dict[str, float] | None = None,
    positioned_arm_slugs: Sequence[str] = POSITIONED_ARM_SLUGS,
    dv_key: str = "delta_g",
    dg_band: tuple[float, float] | None = (SOURCE_DG_BAND_LOW, SOURCE_DG_BAND_HIGH),
) -> dict[str, Any]:
    """Build the (probe × arm × seed) pooled-regression input.

    Args:
        slab_root: eval_results/issue_504/.
        chosen_frac: the pinned checkpoint fraction from Phase 0.
        per_probe: Phase 0.5 per-probe covariates (d_source, d_nearest_neg_nd,
            shadow_angle per arm).
        arm_to_positioned_n: {arm_slug: positioned_N_persona} — used to look up
            shadow_angle / d_nn from per_probe[probe][...][arm_slug].
        seeds: list of seeds to pool over.
        base_prior_by_probe: optional {probe: base_prior_logp} from the eval rig
            (Phase 1c logp_base on the BASE model's R, averaged over Q_eval).
            If None, uses Phase 0.5 placeholder (uniform-low).
        positioned_arm_slugs: Round-2 v2-slug fix (BLOCKER #1). Which 4-arm set
            to iterate. Defaults to v1 (``POSITIONED_ARM_SLUGS``) so the legacy
            v1 pipeline stays byte-identical; the v2 dispatcher threads
            ``POSITIONED_ARM_SLUGS_V2`` so Phase 2 reads the actual
            ``c504v2_<arm>_seed<S>/trajectory.json`` artifacts produced by the
            v2 Phase 1 cell runner.
        dv_key: which per-(probe, q) leaf field of the trajectory's
            ``held_out`` dict to aggregate (mean over Q_eval) as the DV.
            Default ``"delta_g"`` keeps every existing caller byte-identical.
            The #530 logit_reval follow-up threads ``"delta_z_marker"`` /
            ``"delta_margin"`` (raw-logit-space DVs emitted by the
            logit-instrumented eval rig). NOTE: whatever the dv_key, the
            DV lands in ``rows[i]["delta_g"]`` — that is the DV COLUMN NAME
            ``fit_pooled_partial_spearman`` reads, not a claim about the
            quantity. The per-cell in-band gate stays on the LOG-PROB
            source ΔG band regardless of dv_key (the band is defined in
            nats of log-prob; see .claude/rules/marker-training-recipe.md).
        dg_band: #534 — the source-ΔG inclusion band. Default = the canonical
            ``(SOURCE_DG_BAND_LOW, SOURCE_DG_BAND_HIGH)`` exclusion (legacy
            byte-identical). ``None`` DISABLES the exclusion (every cell with a
            trajectory + checkpoint enters the pool) — required for the #534
            sub-final-fraction fits, where the implant is deliberately
            less-trained than the [5, 12] nat band that defines the frac=1.00
            anchor. The ``in_dg_band`` diagnostic flag is ALWAYS computed
            against the canonical band regardless of ``dg_band``.

    Returns:
        {
          "rows": [{cell, seed, probe, d_source, d_nearest_neg_nd, shadow_angle,
                    base_prior_marker, training_step, source_delta_g, delta_g}],
          "per_cell_diagnostics": [{cell, seed, source_dg, source_emission,
                                    in_band, in_emit_band, training_step}],
          "excluded_cells": [{cell, seed, reason}],
          "chosen_frac": float,
          "dv_key": str,
        }
    """
    rows: list[dict] = []
    per_cell_diag: list[dict] = []
    excluded: list[dict] = []
    for cell in positioned_arm_slugs:
        for seed in seeds:
            traj = load_trajectory(slab_root, cell, seed)
            if traj is None:
                excluded.append({"cell": cell, "seed": seed, "reason": "trajectory_missing"})
                continue
            ck = _pick_checkpoint_at_frac(traj, chosen_frac)
            if ck is None:
                excluded.append({"cell": cell, "seed": seed, "reason": "checkpoint_missing"})
                continue
            src = ck.get("source_self", {})
            source_dg = float(
                src.get("delta_g_mean") if src.get("delta_g_mean") is not None else float("nan")
            )
            source_emit = float(
                src.get("emission_p") if src.get("emission_p") is not None else float("nan")
            )
            training_step = int(ck.get("step") or 0)

            in_dg_band = SOURCE_DG_BAND_LOW <= source_dg <= SOURCE_DG_BAND_HIGH
            in_emit_band = EMISSION_BAND_LOW <= source_emit <= EMISSION_BAND_HIGH
            diag_entry = {
                "cell": cell,
                "seed": seed,
                "source_delta_g_nats": source_dg,
                "source_emission_p": source_emit,
                "training_step": training_step,
                "in_dg_band": bool(in_dg_band),
                "in_emit_band": bool(in_emit_band),
            }
            per_cell_diag.append(diag_entry)
            # v5 fix #2 (epm:user-directive 2026-06-08): DROP the source-emission gate.
            # The source SHOULD saturate emission — it IS the implant. The bystander-
            # resolution gate (Phase 0) replaces it. Cells are now excluded ONLY when
            # source ΔG is out of [5, 12] nats. `in_emit_band` is still computed for
            # diagnostic reporting but does not gate inclusion.
            # #534: `dg_band=None` disables the exclusion entirely (sub-final
            # fractions are deliberately below the band); the `in_dg_band`
            # diagnostic above stays pinned to the canonical band.
            if dg_band is not None and not (dg_band[0] <= source_dg <= dg_band[1]):
                excluded.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "reason": (
                            f"out_of_dg_band (dg={source_dg:.2f}, "
                            f"emit={source_emit:.2f} unused-per-fix#2)"
                        ),
                    }
                )
                continue

            held = ck.get("held_out", {})
            for probe, per_q in held.items():
                if probe not in per_probe:
                    # Phase 0.5 panel didn't include this probe — drop with log.
                    continue
                # Aggregate the DV over Q_eval (mean — plan §6 reads the DV at
                # the post-R slot per probe; the regression input is one row per
                # (probe, arm, seed), so average across questions). dv_key
                # selects the leaf field; default "delta_g" is the published DV.
                dgs = [d.get(dv_key) for d in per_q.values() if d.get(dv_key) is not None]
                if not dgs:
                    continue
                delta_g = float(np.mean(dgs))
                cov = per_probe[probe]
                d_source = float(cov["d_source"])
                d_nn = cov["d_nearest_neg_nd"].get(cell)
                shadow = cov["shadow_angle"].get(cell)
                if d_nn is None or shadow is None:
                    continue
                if not math.isfinite(d_nn) or not math.isfinite(shadow):
                    continue
                base_prior = (
                    float(base_prior_by_probe.get(probe, 0.0))
                    if base_prior_by_probe is not None
                    else 0.0
                )
                rows.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "probe": probe,
                        "d_source": d_source,
                        "d_nearest_neg_nd": float(d_nn),
                        "shadow_angle": float(shadow),
                        "base_prior_marker": base_prior,
                        "training_step": training_step,
                        "source_delta_g": source_dg,
                        "delta_g": delta_g,
                    }
                )
    return {
        "rows": rows,
        "per_cell_diagnostics": per_cell_diag,
        "excluded_cells": excluded,
        "chosen_frac": chosen_frac,
        "dv_key": dv_key,
    }


def fit_pooled_partial_spearman(rows: list[dict]) -> dict[str, Any]:
    """Fit the partial-Spearman regression over the 6 predictors.

    Returns:
        {
          "n_rows": int,
          "partial_spearman": {predictor: {"rho": float, "p_raw": float}},
          "holm": holm_correction output (keyed by predictor),
          "pearson": {predictor: {other: float}},  # full pairwise correlation
          "collinearity_warnings": [str, ...],
        }
    """
    if not rows:
        return {
            "n_rows": 0,
            "partial_spearman": {},
            "holm": {},
            "pearson": {},
            "collinearity_warnings": ["no rows"],
        }
    n = len(rows)
    cols: dict[str, list[float]] = {p: [r[p] for r in rows] for p in PREDICTORS}
    y = [r["delta_g"] for r in rows]

    # Partial Spearman per predictor (others = all OTHER predictors).
    partials: dict[str, dict[str, float]] = {}
    raw_p: dict[str, float] = {}
    for p in PREDICTORS:
        target = cols[p]
        others = [cols[q] for q in PREDICTORS if q != p]
        rho = _partial_spearman(y, target, others)
        pv = _spearman_pvalue(rho, n)
        partials[p] = {"rho": float(rho), "p_raw": float(pv)}
        raw_p[p] = pv
    holm = holm_correction(raw_p, alpha=0.05)

    # Pairwise Pearson + collinearity warnings (plan §4.4 + §6.2 Step 6).
    pearson: dict[str, dict[str, float]] = {}
    coll_warns: list[str] = []
    for a in PREDICTORS:
        pearson[a] = {}
        for b in PREDICTORS:
            if a == b:
                pearson[a][b] = 1.0
                continue
            pearson[a][b] = _pearson(cols[a], cols[b])
    # Flag plan-cited collinearity thresholds.
    for a, b in (
        ("source_delta_g", "d_nearest_neg_nd"),
        ("source_delta_g", "shadow_angle"),
        ("d_source", "d_nearest_neg_nd"),
    ):
        r = pearson.get(a, {}).get(b)
        if r is not None and abs(r) > 0.7:
            coll_warns.append(f"|Pearson({a}, {b})|={r:.3f} > 0.7 — collinearity gate tripped")

    return {
        "n_rows": n,
        "partial_spearman": partials,
        "holm": holm,
        "pearson": pearson,
        "collinearity_warnings": coll_warns,
    }


def fit_per_seed(rows: list[dict]) -> dict[int, dict[str, Any]]:
    """Per-seed regression; returns {seed: fit_dict} from fit_pooled_partial_spearman."""
    seeds = sorted({r["seed"] for r in rows})
    out: dict[int, dict[str, Any]] = {}
    for s in seeds:
        sub = [r for r in rows if r["seed"] == s]
        out[s] = fit_pooled_partial_spearman(sub)
    return out


def sign_agreement_across_seeds(per_seed_fits: dict[int, dict[str, Any]]) -> dict[str, dict]:
    """Per predictor, count how many seeds agree on rho sign.

    Returns {predictor: {"n_positive": int, "n_negative": int, "n_seeds": int}}.
    """
    out: dict[str, dict[str, int]] = {}
    seeds = list(per_seed_fits)
    for p in PREDICTORS:
        n_pos = 0
        n_neg = 0
        for s in seeds:
            fit = per_seed_fits[s]
            part = fit.get("partial_spearman", {}).get(p)
            if part is None:
                continue
            r = part["rho"]
            if r > 0:
                n_pos += 1
            elif r < 0:
                n_neg += 1
        out[p] = {"n_positive": n_pos, "n_negative": n_neg, "n_seeds": len(seeds)}
    return out


def sign_flip_robustness(rows: list[dict]) -> dict[str, Any]:
    """Plan §4.4 Step 6 / §6.2 #4: re-fit with sign-flipped d_source covariate.

    Reports whether d_nearest_neg_nd and shadow_angle keep their signs + Holm
    significances after substituting `-d_source` for `d_source`.
    """
    if not rows:
        return {"n_rows": 0}
    flipped = [{**r, "d_source": -r["d_source"]} for r in rows]
    original = fit_pooled_partial_spearman(rows)
    flipped_fit = fit_pooled_partial_spearman(flipped)
    out: dict[str, Any] = {
        "original": {
            p: original["partial_spearman"][p] for p in ("d_nearest_neg_nd", "shadow_angle")
        },
        "flipped": {
            p: flipped_fit["partial_spearman"][p] for p in ("d_nearest_neg_nd", "shadow_angle")
        },
    }
    out["sign_stable"] = {
        p: (
            (out["original"][p]["rho"] > 0) == (out["flipped"][p]["rho"] > 0)
            and (out["original"][p]["rho"] != 0)
        )
        for p in ("d_nearest_neg_nd", "shadow_angle")
    }
    return out


def robustness_panel_per_cell_best_band(
    *,
    slab_root: Path,
    per_probe: dict[str, dict[str, Any]],
    arm_to_positioned_n: dict[str, str],
    seeds: Sequence[int],
    dg_band: tuple[float, float] = (SOURCE_DG_BAND_LOW, SOURCE_DG_BAND_HIGH),
    base_prior_by_probe: dict[str, float] | None = None,
    positioned_arm_slugs: Sequence[str] = POSITIONED_ARM_SLUGS,
) -> dict[str, Any]:
    """Phase 2 Step 1 robustness panel (plan §4.4 / Step 1).

    Re-fits Step 3 with each cell read at its OWN latest-in-band checkpoint
    (NOT the pinned fraction). If the verdict + signs agree with the
    pinned-fraction read, this strengthens HIGH confidence; disagreement
    downgrades.

    Round-2 v2-slug fix (BLOCKER #1): ``positioned_arm_slugs`` selects which
    4-arm set to iterate. Defaults to v1 for byte-identical legacy behavior;
    the v2 dispatcher threads ``POSITIONED_ARM_SLUGS_V2``.
    """
    rows: list[dict] = []
    per_cell: list[dict] = []
    for cell in positioned_arm_slugs:
        for seed in seeds:
            traj = load_trajectory(slab_root, cell, seed)
            if traj is None:
                continue
            cks = traj.get("checkpoints", [])
            in_band = [
                c
                for c in cks
                if (sm := c.get("source_self", {})) is not None
                and sm.get("delta_g_mean") is not None
                and dg_band[0] <= float(sm["delta_g_mean"]) <= dg_band[1]
            ]
            if not in_band:
                per_cell.append(
                    {"cell": cell, "seed": seed, "best_band_frac": None, "reason": "no_in_band"}
                )
                continue
            best = max(in_band, key=lambda c: c["frac"])
            per_cell.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "best_band_frac": best["frac"],
                    "source_dg": float(best["source_self"]["delta_g_mean"]),
                    "training_step": int(best.get("step") or 0),
                }
            )
            held = best.get("held_out", {})
            source_dg = float(best["source_self"]["delta_g_mean"])
            training_step = int(best.get("step") or 0)
            for probe, per_q in held.items():
                if probe not in per_probe:
                    continue
                dgs = [d.get("delta_g") for d in per_q.values() if d.get("delta_g") is not None]
                if not dgs:
                    continue
                delta_g = float(np.mean(dgs))
                cov = per_probe[probe]
                d_nn = cov["d_nearest_neg_nd"].get(cell)
                shadow = cov["shadow_angle"].get(cell)
                if d_nn is None or shadow is None:
                    continue
                base_prior = (
                    float(base_prior_by_probe.get(probe, 0.0))
                    if base_prior_by_probe is not None
                    else 0.0
                )
                rows.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "probe": probe,
                        "d_source": float(cov["d_source"]),
                        "d_nearest_neg_nd": float(d_nn),
                        "shadow_angle": float(shadow),
                        "base_prior_marker": base_prior,
                        "training_step": training_step,
                        "source_delta_g": source_dg,
                        "delta_g": delta_g,
                    }
                )
    fit = fit_pooled_partial_spearman(rows) if rows else None
    return {
        "n_rows": len(rows),
        "per_cell_best_band": per_cell,
        "fit": fit,
    }


def step_confound_diagnostics(rows: list[dict]) -> dict[str, Any]:
    """Pearson(training_step, ΔG) across cells (plan §6.2 #5)."""
    if not rows:
        return {}
    steps = [r["training_step"] for r in rows]
    dgs = [r["delta_g"] for r in rows]
    return {
        "n_rows": len(rows),
        "pearson_step_vs_dg": _pearson(steps, dgs),
    }


def implant_strength_diagnostics(rows: list[dict], fit: dict[str, Any]) -> dict[str, Any]:
    """`source_delta_g(cell)` diagnostics (plan §3 / §4.4 / §6.2 #6).

    Reports Pearson(source_delta_g, ΔG) + Pearson with the geometry predictors;
    the > 0.7 thresholds (the implant-strength-confound trigger) are also
    surfaced in fit_pooled_partial_spearman.collinearity_warnings.
    """
    if not rows or not fit:
        return {}
    src_dg = [r["source_delta_g"] for r in rows]
    dgs = [r["delta_g"] for r in rows]
    diags: dict[str, Any] = {
        "pearson_source_dg_vs_delta_g": _pearson(src_dg, dgs),
        "pearson_source_dg_vs_d_nn": _pearson(src_dg, [r["d_nearest_neg_nd"] for r in rows]),
        "pearson_source_dg_vs_shadow": _pearson(src_dg, [r["shadow_angle"] for r in rows]),
    }
    diags["confound_triggered"] = bool(
        abs(diags.get("pearson_source_dg_vs_d_nn") or 0.0) > 0.7
        or abs(diags.get("pearson_source_dg_vs_shadow") or 0.0) > 0.7
    )
    return diags


def run_phase2_analysis(
    *,
    slab_root: Path,
    phase0_calibration: dict[str, Any],
    phase05_gates: dict[str, Any],
    seeds: Sequence[int] = (42, 137),
    base_prior_by_probe: dict[str, float] | None = None,
    positioned_arm_slugs: Sequence[str] = POSITIONED_ARM_SLUGS,
    dg_band: tuple[float, float] | None = (SOURCE_DG_BAND_LOW, SOURCE_DG_BAND_HIGH),
) -> dict[str, Any]:
    """End-to-end Phase 2 (CPU-only): pinned read + robustness + diagnostics.

    Args:
        slab_root: eval_results/issue_504.
        phase0_calibration: loaded phase0_calibration.json (must verdict='pass').
        phase05_gates: loaded phase0_5_gates.json (must verdict='pass').
        seeds: cells to pool over.
        base_prior_by_probe: optional {probe: base_prior_marker_logp_mean} for
            the regression covariate. None means Phase 1 didn't emit it; the
            regression runs with the placeholder 0.0 per probe and the
            implementer/analyzer should fill this from Phase 1c trajectory
            (logp_base mean over Q_eval per probe). See plan §4.4 Step 2.
        positioned_arm_slugs: Round-2 v2-slug fix (BLOCKER #1). Which 4-arm set
            (v1 ``POSITIONED_ARM_SLUGS`` or v2 ``POSITIONED_ARM_SLUGS_V2``) to
            iterate when building rows from per-cell trajectories. Defaults to
            v1 so the legacy pipeline stays byte-identical; the v2 dispatcher
            threads v2 via the ``--positioned-arms`` CLI flag on
            ``scripts/i504_phase_analyze.py``.

    Returns:
        analyze_summary dict (writable to JSON). Includes:
          chosen_checkpoint_fraction,
          per_cell_diagnostics, excluded_cells,
          pooled_fit (partial Spearman + Holm + Pearson + collinearity),
          per_seed_fit,
          sign_agreement,
          sign_flip_robustness,
          step_confound,
          implant_strength,
          robustness_panel_per_cell_best_band,
          identification_gates_at_eval (placeholder; #504 reuses Phase 0.5),
          predictors: PREDICTORS,
          notes: any flags to surface in the clean-result.
    """
    chosen_frac = phase0_calibration["chosen_checkpoint_fraction"]
    if chosen_frac is None:
        raise RuntimeError(
            "run_phase2_analysis: phase0_calibration has chosen_checkpoint_fraction=None. "
            "Phase 0 did not find an in-band anchor; Phase 2 cannot proceed."
        )
    per_probe = phase05_gates["per_probe"]
    arm_to_positioned_n = phase05_gates["arm_to_positioned_n"]

    pooled = build_rows(
        slab_root=slab_root,
        chosen_frac=float(chosen_frac),
        per_probe=per_probe,
        arm_to_positioned_n=arm_to_positioned_n,
        seeds=list(seeds),
        base_prior_by_probe=base_prior_by_probe,
        positioned_arm_slugs=positioned_arm_slugs,
        dg_band=dg_band,
    )
    rows = pooled["rows"]
    fit = fit_pooled_partial_spearman(rows)
    per_seed = fit_per_seed(rows)
    sign_agree = sign_agreement_across_seeds(per_seed)
    flip = sign_flip_robustness(rows)
    step_conf = step_confound_diagnostics(rows)
    impl_conf = implant_strength_diagnostics(rows, fit)
    robust = robustness_panel_per_cell_best_band(
        slab_root=slab_root,
        per_probe=per_probe,
        arm_to_positioned_n=arm_to_positioned_n,
        seeds=list(seeds),
        base_prior_by_probe=base_prior_by_probe,
        positioned_arm_slugs=positioned_arm_slugs,
    )

    notes: list[str] = []
    n_excluded = len(pooled["excluded_cells"])
    if n_excluded > 1:
        notes.append(
            f"{n_excluded} (cell × seed) excluded for out-of-band source ΔG "
            f"(emission gate dropped per v5 fix #2) at "
            f"chosen_frac={chosen_frac} — Indeterminate (anchor unstable across arms)."
        )
    if fit.get("collinearity_warnings"):
        notes.extend(fit["collinearity_warnings"])
    if impl_conf.get("confound_triggered"):
        notes.append(
            "implant-strength-confound triggered (|Pearson(source_delta_g, d_nn or shadow)|>0.7) — "
            "Indeterminate (implant-strength-confounded) per plan §3 / §6.3."
        )

    return {
        "schema_version": "i504_v1",
        "chosen_checkpoint_fraction": float(chosen_frac),
        # #534: which source-ΔG inclusion band gated the pool (None = no
        # exclusion — the sub-final-fraction read). Self-describing so a
        # per-fraction JSON can't be mistaken for a banded anchor fit.
        "dg_band_applied": list(dg_band) if dg_band is not None else None,
        "predictors": list(PREDICTORS),
        "per_cell_diagnostics": pooled["per_cell_diagnostics"],
        "excluded_cells": pooled["excluded_cells"],
        "pooled_fit": fit,
        "per_seed_fit": per_seed,
        "sign_agreement": sign_agree,
        "sign_flip_robustness": flip,
        "step_confound": step_conf,
        "implant_strength_confound": impl_conf,
        "robustness_panel_per_cell_best_band": robust,
        "n_rows_pooled": fit.get("n_rows", 0),
        "notes": notes,
        "verdict_assignment": (
            "Verdict (Bubble/Barrier/Both/Indeterminate) + confidence (HIGH/MODERATE/LOW) "
            "are assigned by the analyzer agent weighing the diagnostics jointly per plan "
            "§6.3 — NOT by a hard p-threshold ladder here."
        ),
    }


def write_analyze_summary(summary: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("[analyze] wrote %s (n_rows_pooled=%d)", out_path, summary.get("n_rows_pooled", 0))
    return out_path
