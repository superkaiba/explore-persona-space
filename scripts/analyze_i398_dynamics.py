"""Per-bystander A/B/C labeling + hero figure for issue #398.

Reads the two per-checkpoint eval JSONs produced by phases 3a + 3b
(``eval_i398_marker_logprob.py`` and ``eval_i398_marker_spread.py``) plus
``eval_results/issue_385/predictors_base.json``, and runs the analysis
described in plan §4.2(f). For each bystander, fits the log p(``※``) series in
the ``[5, 75]`` step window on the (log10(step), sum-over-prompts log p) plane,
under two model families:

    1. Single straight-line fit.
    2. Piecewise-linear fit with break ∈ {25, 50, 75} (three candidate breaks;
       pick the one with the smallest AIC).

The §1 quantitative signatures (ΔAIC, cumulative log p fraction, post/pre
slope ratio) then label each bystander A / B / C / noise-dominated. The whole
labeling pass runs **independently on BOTH probe geometries** (``pos0`` and
``endpos``) per the plan §0 Methodology reconciler binding fix. Per-bystander
``consensus_label`` records agreement or "position_dependent" when the two
geometries disagree.

Outputs (all under ``--output-dir``):

    - ``analysis.json``: per-bystander labels per geometry + consensus, plus
      per-checkpoint Spearman rho(log p, cosine-to-source) computed separately
      for ``pos0`` and ``endpos``.
    - ``hero_figure.png`` / ``.pdf`` / ``.meta.json``: substring-match rate
      curve overlaid with both log-p curves on a twin y-axis, colored by
      consensus scenario label.

Per CLAUDE.md "Checkpoint per phase" rule the analysis emits ``analysis.json``
as a single atomic write at the end — the analyzer agent reads it and the
hero figure together. This is a CPU-only postprocess (~10 min wall) so a
mid-script crash is cheap to re-run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make ``src/`` importable for paper-plots styling helpers regardless of cwd.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

# The 12 checkpoint steps that sit inside the [5, 75] window. Plan §1 spec.
WINDOW_STEPS: list[int] = [5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75]

# Candidate breakpoints for the piecewise-linear fit per plan §4.2(f) step 2.
PIECEWISE_BREAKS: list[int] = [25, 50, 75]


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------


def _aic_linear(rss: float, n: int, k: int) -> float:
    """Akaike Information Criterion for a Gaussian-residual linear model.

    AIC = 2k + n ln(RSS / n). Equivalent up to additive constants across model
    families with the same n, which is the only comparison this script makes.
    Returns +inf when RSS is non-positive (degenerate single-point fit) so it
    is never selected by ``argmin``.
    """
    if n <= 0 or rss <= 0:
        return float("inf")
    return 2 * k + n * math.log(rss / n)


def _fit_single_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit y = a + b*x by least squares.

    Returns ``(intercept, slope, rss, aic)``. AIC uses k=3 free params
    (intercept, slope, sigma) under a Gaussian-residual model.
    """
    assert x.ndim == 1 and y.ndim == 1 and x.shape == y.shape, (x.shape, y.shape)
    if len(x) < 2:
        return float("nan"), float("nan"), float("inf"), float("inf")
    slope, intercept = np.polyfit(x, y, 1)
    yhat = intercept + slope * x
    rss = float(np.sum((y - yhat) ** 2))
    return float(intercept), float(slope), rss, _aic_linear(rss, len(x), k=3)


def _fit_piecewise_linear_at_break(
    x: np.ndarray, y: np.ndarray, break_x: float
) -> tuple[float, float, float, float]:
    """Fit y = a + b1 * x + b2 * max(0, x - break_x).

    Returns ``(slope_pre, slope_post, rss, aic)``. ``slope_post`` is the slope
    on the right side of the break (slope_pre + b2). AIC uses k=4 (intercept +
    two slopes + sigma).
    """
    assert x.ndim == 1 and y.ndim == 1 and x.shape == y.shape, (x.shape, y.shape)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("inf"), float("inf")
    knee = np.maximum(0.0, x - break_x)
    X = np.stack([np.ones(n), x, knee], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("inf"), float("inf")
    _intercept, b1, b2 = coef
    slope_pre = float(b1)
    slope_post = float(b1 + b2)
    yhat = X @ coef
    rss = float(np.sum((y - yhat) ** 2))
    return slope_pre, slope_post, rss, _aic_linear(rss, n, k=4)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation + two-sided p-value (Fisher z-approx).

    Returns ``(rho, p)``. NaN if either array has < 3 non-NaN entries or is
    constant. p-value uses Fisher's z-transform with sample size n; adequate
    for n=27. Avoids the scipy dep to keep this script lightweight.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan")
    rx = _rank(x[mask])
    ry = _rank(y[mask])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan"), float("nan")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    # Fisher's z-transform for p-value
    if abs(rho) >= 0.9999:
        return rho, 0.0
    z = math.atanh(rho) * math.sqrt(n - 3)
    # two-sided p-value from standard normal
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return rho, float(p)


def _rank(a: np.ndarray) -> np.ndarray:
    """Average-rank vector for ``a`` (ties get the mean of their bracket)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(a, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # ties: average ranks within each tied group
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j > i + 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


# ---------------------------------------------------------------------------
# Per-bystander labeling
# ---------------------------------------------------------------------------


def _persona_series_for_geometry(
    per_step: dict[str, dict[str, dict[str, list[float]]]],
    persona: str,
    geometry: str,
    steps: list[int],
) -> np.ndarray:
    """Sum-over-prompts log p series for one (persona, geometry) across ``steps``.

    Returns a 1-D array shaped ``(len(steps),)`` with one float per step.
    Sums (not means) because the analyzer compares fits across the same set of
    prompts at every step; sum and mean differ only by a constant multiplier
    that washes out of slopes, ratios, and ΔAIC.
    """
    out = np.empty(len(steps), dtype=float)
    for i, step in enumerate(steps):
        cell = per_step.get(str(step), {}).get(persona, {})
        series = cell.get(geometry, None)
        if series is None or len(series) == 0:
            out[i] = float("nan")
        else:
            out[i] = float(np.sum(series))
    return out


def label_persona_for_geometry(log_p_series: np.ndarray, total_series: np.ndarray | None) -> dict:
    """Apply plan §1 quantitative signatures to label one persona A/B/C/noise.

    Args:
        log_p_series: shape ``(12,)`` — sum-over-prompts log p at the 12
            window steps ``WINDOW_STEPS``.
        total_series: shape ``(N,)`` — sum-over-prompts log p at ALL
            checkpoints (used for the "step 5 → step 1600 total change"
            denominator). When ``None``, falls back to ``log_p_series``.

    Returns:
        Dict with single-line fit, best piecewise fit, ΔAIC, slope ratio,
        cumulative ratios, and a label in
        ``{"A", "B", "C", "noise_dominated"}``.
    """
    x = np.log10(np.asarray(WINDOW_STEPS, dtype=float))
    y = np.asarray(log_p_series, dtype=float)
    if not np.all(np.isfinite(y)):
        return {
            "label": "noise_dominated",
            "reason": "non-finite log p values in window",
        }

    # 1. Single-line fit
    intercept_1, slope_1, rss_1, aic_1 = _fit_single_line(x, y)

    # 2. Piecewise-linear fit at each candidate break; pick by AIC
    pw_fits: list[dict] = []
    for break_step in PIECEWISE_BREAKS:
        break_x = math.log10(break_step)
        sp_pre, sp_post, rss_pw, aic_pw = _fit_piecewise_linear_at_break(x, y, break_x)
        pw_fits.append(
            {
                "break_step": break_step,
                "slope_pre": sp_pre,
                "slope_post": sp_post,
                "rss": rss_pw,
                "aic": aic_pw,
            }
        )
    best_pw = min(pw_fits, key=lambda d: d["aic"])

    # 3. ΔAIC = AIC(single) - AIC(best piecewise) — positive favors piecewise
    delta_aic = aic_1 - best_pw["aic"]

    # 4. Cumulative log p change ratios
    y_5 = float(y[WINDOW_STEPS.index(5)])
    y_50 = float(y[WINDOW_STEPS.index(50)])
    y_75 = float(y[WINDOW_STEPS.index(75)])
    change_5_to_50 = y_50 - y_5
    change_5_to_75 = y_75 - y_5
    ratio_50_over_75 = change_5_to_50 / change_5_to_75 if change_5_to_75 != 0 else float("nan")
    if total_series is not None and len(total_series) > 0:
        y_total_end = float(total_series[-1])
        change_5_to_end = y_total_end - y_5
        ratio_50_over_total = (
            change_5_to_50 / change_5_to_end if change_5_to_end != 0 else float("nan")
        )
    else:
        ratio_50_over_total = float("nan")

    # 5. Slope ratio (post-break / pre-break) for best piecewise fit
    sp_pre = best_pw["slope_pre"]
    sp_post = best_pw["slope_post"]
    slope_ratio = (sp_post / sp_pre) if sp_pre and abs(sp_pre) > 1e-9 else float("inf")

    # 6. Label per plan §1 thresholds
    # Scenario C: window slope CI crosses zero OR abs(slope) < 0.01 AND
    #             cumulative ratio (5->50)/(5->75) < 0.05
    # Scenario B: dAIC > 4 AND post-slope >= 3x pre-slope AND pre-slope > 0 AND
    #             cumulative ratio in [0.05, 0.25]
    # Scenario A: dAIC <= 4 AND cumulative ratio (5->50)/(5->75) >= 0.25
    # Otherwise: noise_dominated
    label: str
    reason: str
    abs_slope_low = abs(slope_1) < 0.01
    if (
        (abs_slope_low or not np.isfinite(slope_1))
        and not math.isnan(ratio_50_over_75)
        and ratio_50_over_75 < 0.05
    ):
        label = "C"
        reason = (
            f"single-line |slope|={slope_1:.4f} < 0.01 and cumulative "
            f"(5→50)/(5→75)={ratio_50_over_75:.3f} < 0.05"
        )
    elif (
        delta_aic > 4
        and sp_pre > 0
        and slope_ratio >= 3
        and not math.isnan(ratio_50_over_75)
        and 0.05 <= ratio_50_over_75 <= 0.25
    ):
        label = "B"
        reason = (
            f"ΔAIC={delta_aic:.2f} > 4, post/pre slope={slope_ratio:.2f} ≥ 3, "
            f"pre-slope={sp_pre:.4f} > 0, cumulative ratio={ratio_50_over_75:.3f}"
        )
    elif delta_aic <= 4 and not math.isnan(ratio_50_over_75) and ratio_50_over_75 >= 0.25:
        label = "A"
        reason = (
            f"ΔAIC={delta_aic:.2f} ≤ 4 (single-line adequate), cumulative "
            f"ratio (5→50)/(5→75)={ratio_50_over_75:.3f} ≥ 0.25"
        )
    else:
        label = "noise_dominated"
        reason = (
            f"thresholds not decisive: ΔAIC={delta_aic:.2f}, "
            f"cumulative ratio={ratio_50_over_75:.3f}, single-slope={slope_1:.4f}"
        )

    return {
        "label": label,
        "reason": reason,
        "single_line": {
            "intercept": intercept_1,
            "slope": slope_1,
            "rss": rss_1,
            "aic": aic_1,
        },
        "piecewise_fits": pw_fits,
        "best_piecewise": best_pw,
        "delta_aic": delta_aic,
        "slope_ratio_post_over_pre": slope_ratio,
        "cumulative_change_5_to_50": change_5_to_50,
        "cumulative_change_5_to_75": change_5_to_75,
        "ratio_50_over_75": ratio_50_over_75,
        "ratio_50_over_total": ratio_50_over_total,
    }


def consensus_label(label_pos0: str, label_endpos: str) -> str:
    """Combine two per-geometry labels into a single consensus label.

    Plan §4.2(f) step 7:
        - Agreement → record the agreed label.
        - Disagreement → "position_dependent".
        - Either side "noise_dominated" → "low_power" (drop from consensus
          headline).
    """
    if label_pos0 == "noise_dominated" or label_endpos == "noise_dominated":
        return "low_power"
    if label_pos0 == label_endpos:
        return label_pos0
    return "position_dependent"


# ---------------------------------------------------------------------------
# Hero figure
# ---------------------------------------------------------------------------


def _build_consensus_color_map() -> dict[str, str]:
    """Map A/B/C/position_dependent/low_power to colorblind-safe hexes.

    Requires ``set_paper_style`` to have been called so the active palette
    is resolved. The five roles available across both ``"blog"`` and
    ``"neurips"`` styles are ``primary``, ``baseline``, ``control``,
    ``accent``, ``neutral`` -- one per consensus label.
    """
    return {
        "A": paper_palette_role("primary"),
        "B": paper_palette_role("accent"),
        "C": paper_palette_role("control"),
        "position_dependent": paper_palette_role("baseline"),
        "low_power": paper_palette_role("neutral"),
    }


def build_hero_figure(
    steps: list[int],
    rate_per_step: dict[str, dict[int, float]],
    logp_pos0: dict[str, dict[int, float]],
    logp_endpos: dict[str, dict[int, float]],
    consensus_by_persona: dict[str, str],
    output_dir: Path,
) -> None:
    """Hero figure — substring-match rate + dual log-prob curves, colored by consensus label.

    Top panel: substring-match panel-mean rate vs step (linear), with the #385
    parity headline (jump-at-step-75).

    Bottom panel: per-bystander log p(``※``) trajectories for both geometries
    on a twin axis (pos0 dashed, endpos solid), colored by ``consensus_by_persona``
    so the eye groups bystanders by scenario label.
    """
    set_paper_style("blog")
    color_map = _build_consensus_color_map()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.0, 6.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]}
    )

    # Top panel: panel-mean substring-match rate
    if rate_per_step:
        panel_personas = list(rate_per_step.keys())
        rate_vec = np.array(
            [
                float(np.mean([rate_per_step[p].get(s, float("nan")) for p in panel_personas]))
                for s in steps
            ]
        )
        ax_top.plot(
            steps,
            rate_vec,
            color=paper_palette_role("primary"),
            marker="o",
            linewidth=2,
            markersize=4,
        )
        ax_top.set_ylabel("Marker fires per completion")
        ax_top.set_ylim(bottom=-0.005)
        add_direction_arrow(ax_top, axis="y", direction="up")
    else:
        ax_top.set_visible(False)

    # Bottom panel: per-bystander log p curves
    for persona, label in consensus_by_persona.items():
        color = color_map.get(label, "#999999")
        pos0 = [logp_pos0.get(persona, {}).get(s, float("nan")) for s in steps]
        endpos = [logp_endpos.get(persona, {}).get(s, float("nan")) for s in steps]
        ax_bot.plot(steps, pos0, color=color, linestyle="--", alpha=0.65, linewidth=1)
        ax_bot.plot(steps, endpos, color=color, linestyle="-", alpha=0.65, linewidth=1)

    ax_bot.set_xscale("log")
    ax_bot.set_xlabel("Training step (log scale)")
    ax_bot.set_ylabel("Sum-over-prompts log p of marker")
    add_direction_arrow(ax_bot, axis="y", direction="up")

    # Manual legend keyed by consensus label
    legend_handles = []
    for label_key in ("A", "B", "C", "position_dependent", "low_power"):
        if any(v == label_key for v in consensus_by_persona.values()):
            (line,) = ax_bot.plot(
                [],
                [],
                color=color_map.get(label_key, "#999999"),
                linewidth=2,
                label=label_key,
            )
            legend_handles.append(line)
    if legend_handles:
        ax_bot.legend(
            handles=legend_handles,
            title="Per-bystander scenario",
            loc="lower right",
            fontsize=8,
        )

    set_title_subtitle(
        ax_top,
        "Marker emerges sharply in substring metric; log-prob shows whether it ramped or cliffed",
        subtitle=(
            "Top: sampled marker rate. Bottom: teacher-forced log p per bystander, "
            "two probe geometries."
        ),
    )

    savefig_paper(fig, "hero_figure", dir=str(output_dir))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logp-file", required=True, help="Phase-3a output JSON.")
    ap.add_argument("--rate-file", required=True, help="Phase-3b output JSON.")
    ap.add_argument(
        "--predictors-file",
        required=True,
        help="eval_results/issue_385/predictors_base.json -- for Spearman rho.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory for analysis.json + hero_figure.{png,pdf,meta.json}.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.logp_file) as f:
        logp = json.load(f)
    with open(args.rate_file) as f:
        rate = json.load(f)
    with open(args.predictors_file) as f:
        predictors = json.load(f)

    panel: list[str] = logp["panel"]
    geometries: list[str] = logp.get("geometries", ["pos0", "endpos"])
    assert set(geometries) >= {"pos0", "endpos"}, (
        f"expected dual-probe geometries, got {geometries}"
    )

    # All checkpoint steps that appear in the log-prob output
    all_steps: list[int] = sorted(int(s) for s in logp["per_step"])
    assert [s for s in all_steps if s <= 75] == WINDOW_STEPS, (
        f"WINDOW_STEPS spec {WINDOW_STEPS} drifted from actual checkpoints "
        f"{[s for s in all_steps if s <= 75]}"
    )

    # 1-6. Per-persona labels for BOTH geometries independently.
    per_persona: dict[str, dict] = {}
    for persona in panel:
        per_persona[persona] = {}
        for geom in ("pos0", "endpos"):
            window_series = _persona_series_for_geometry(
                logp["per_step"], persona, geom, WINDOW_STEPS
            )
            total_series = _persona_series_for_geometry(logp["per_step"], persona, geom, all_steps)
            per_persona[persona][geom] = label_persona_for_geometry(window_series, total_series)

    # 7. Dual-geometry consensus per bystander
    consensus_by_persona: dict[str, str] = {}
    for persona in panel:
        label_pos0 = per_persona[persona]["pos0"]["label"]
        label_endpos = per_persona[persona]["endpos"]["label"]
        consensus_by_persona[persona] = consensus_label(label_pos0, label_endpos)

    # 8. Per-checkpoint Spearman rho vs base-model cosine-to-source.
    # predictors_base.json["cosine_to_source"] is a dict {persona: cosine}.
    cos_to_src: dict[str, float] = predictors["cosine_to_source"]
    spearman_rho_per_step: dict[str, dict[str, dict[str, float]]] = {
        "pos0": {},
        "endpos": {},
    }
    for geom in ("pos0", "endpos"):
        for step in all_steps:
            personas_with_cos = [p for p in panel if p in cos_to_src and p != "librarian"]
            xs = np.array([cos_to_src[p] for p in personas_with_cos], dtype=float)
            ys = np.array(
                [
                    float(np.sum(logp["per_step"][str(step)][p][geom]))
                    if (p in logp["per_step"][str(step)] and geom in logp["per_step"][str(step)][p])
                    else float("nan")
                    for p in personas_with_cos
                ],
                dtype=float,
            )
            rho, p_val = _spearman_rho(xs, ys)
            spearman_rho_per_step[geom][str(step)] = {
                "rho": rho,
                "p_value": p_val,
                "n": int(np.isfinite(xs).sum() & np.isfinite(ys).sum()),
                "n_personas": len(personas_with_cos),
            }

    # Counts by label for headline
    label_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "noise_dominated": 0}
    for geom in ("pos0", "endpos"):
        for persona in panel:
            label_counts[per_persona[persona][geom]["label"]] = (
                label_counts.get(per_persona[persona][geom]["label"], 0) + 1
            )
    consensus_counts: dict[str, int] = {}
    for v in consensus_by_persona.values():
        consensus_counts[v] = consensus_counts.get(v, 0) + 1

    analysis = {
        "panel": panel,
        "geometries": geometries,
        "window_steps": WINDOW_STEPS,
        "all_steps": all_steps,
        "per_persona": per_persona,
        "consensus_by_persona": consensus_by_persona,
        "consensus_counts": consensus_counts,
        "label_counts_summed_over_geometries": label_counts,
        "spearman_rho_per_step": spearman_rho_per_step,
    }
    analysis_path = out_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"wrote {analysis_path}", flush=True)

    # 9. Hero figure
    rate_per_step: dict[str, dict[int, float]] = {}
    for step in all_steps:
        for persona, q_map in rate.get("per_step", {}).get(str(step), {}).items():
            persona_rates = [c["rate"] for c in q_map.values()]
            mean_rate = float(np.mean(persona_rates)) if persona_rates else float("nan")
            rate_per_step.setdefault(persona, {})[step] = mean_rate

    logp_pos0_per_step: dict[str, dict[int, float]] = {}
    logp_endpos_per_step: dict[str, dict[int, float]] = {}
    for persona in panel:
        for step in all_steps:
            cell = logp["per_step"][str(step)].get(persona, {})
            logp_pos0_per_step.setdefault(persona, {})[step] = (
                float(np.sum(cell["pos0"])) if "pos0" in cell else float("nan")
            )
            logp_endpos_per_step.setdefault(persona, {})[step] = (
                float(np.sum(cell["endpos"])) if "endpos" in cell else float("nan")
            )

    build_hero_figure(
        steps=all_steps,
        rate_per_step=rate_per_step,
        logp_pos0=logp_pos0_per_step,
        logp_endpos=logp_endpos_per_step,
        consensus_by_persona=consensus_by_persona,
        output_dir=out_dir,
    )
    print(f"wrote {out_dir}/hero_figure.png / .pdf / .meta.json", flush=True)


if __name__ == "__main__":
    main()
