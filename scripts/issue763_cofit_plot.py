#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, √, ×, −, Δ) in scientific docstrings + matplotlib labels.
"""Issue #763 `neutral-contrast-and-cofit`: the §6 figures.

Hero — ``fig_763_cofit_grid``: one row per behavior, the 8-method bar chart
(layer-max held-out ρ with bootstrap CI per method, the selection-matched null
p95 line, the random-direction 2.5–97.5 band shaded, the √r_yy ceiling dashed).
pv_rA / pv_rC / pv_neutral stay UNCOLLAPSED with plain-English legend names
(plan §5 label discipline).

Exploratory dump: per-method per-layer ρ curves; per-context held-out
prediction-vs-graded-E0 scatter (ridge / best direction / neutral — the
low-level data plot, points colored by context family); cos(r_A, r_C) +
cos(r_C, r_neutral) per layer; kernel-vs-linear paired per-context error
scatter; LEACE dCor null histograms with observed + control marked;
leave-default-family-out ρ bars (pv_neutral vs pv_rC); ridge-parent-check bar;
binary-companion ridge column vs graded.

Reads ``eval_results/issue_763/neutral-contrast-and-cofit/{cofit_results.json,
nonlinear_tests.json, neutral_arm_manifest.json}``. Tolerant of missing
FIELDS (an unbuildable pv_neutral plots as N/A — nan, never a zero bar), but
the plan-required ``neutral_arm_manifest.json`` is staged from HF when absent
and FAIL-LOUD required after staging (never warn-and-skip). ``--smoke``
asserts ≥1 PNG.

Usage::

    uv run python scripts/issue763_cofit_plot.py
    uv run python scripts/issue763_cofit_plot.py --smoke --behaviors deception
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# #847: shared-VM thread caps must bind BEFORE torch/numpy freeze their pools at import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    COFIT_DIR,
    FIGURE_DIR,
    HF_ANALYSIS_TENSORS_PREFIX,
    ensure_smoke_scope,
    load_json,
)

logger = logging.getLogger("issue763_cofit_plot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

METHOD_ORDER = (
    "cofit_ridge",
    "cofit_krr",
    "pv_rA",
    "pv_rC",
    "pv_neutral",
    "diffmeans_crude",
    "cC_ridge",
    "rand_dir",
)

# Reader-facing family names for the battery's 7 ``family`` values (ground
# truth: data/issue594/battery.json — house + PersonaHub personas share the
# ``persona`` label; f2_wc_* is allenai/WildChat-1M conversation prefixes,
# NOT "word-count"; interp-critique r1 on the cofit round).
FAMILY_LABELS = {
    "persona": "persona (hub/house)",
    "wildchat": "WildChat",
    "icl": "ICL k-shot",
    "rephrase": "rephrase",
    "format": "format",
    "default": "default template",
    "behavior": "behavior-adjacent",
}


def _ctx_point_label(ctx: str) -> str:
    """Reader-facing short label for a battery context id (never the raw slug).

    ``f1_house_librarian`` -> ``librarian``; ``f1_phub_03`` -> ``hub 03``;
    ``f2_wc_long_3`` -> ``WildChat long 3``; ``f5_fmt_json`` -> ``format json``.
    """
    parts = ctx.split("_")
    rest = parts[1:]
    head = rest[0] if rest else ""
    tail = rest[1:]
    if head == "house":
        return " ".join(tail)
    if head == "phub":
        return "hub " + " ".join(tail)
    if head == "wc":
        return "WildChat " + " ".join(tail)
    if head == "icl":
        return "ICL " + " ".join(tail)
    if head == "reph":
        return "rephrase " + " ".join(tail)
    if head == "fmt":
        return "format " + " ".join(tail)
    if head == "behav":
        return " ".join(tail)
    return " ".join(rest).replace("asst", "assistant")


def _try_paper_style() -> None:
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:  # best-effort styling, never blocking
        logger.warning("set_paper_style failed (continuing with defaults): %s", e)


def _palette(n: int) -> list[str]:
    try:
        from explore_persona_space.analysis.paper_plots import paper_palette

        return paper_palette(n)
    except Exception:
        return [f"C{i}" for i in range(n)]


def _short_label(results: dict, m: str) -> str:
    lab = results.get("method_labels", {}).get(m, m)
    return lab.split(" (")[0] if len(lab) > 28 else lab


def plot_cofit_grid(results: dict, behaviors: list[str], out_path: Path) -> None:
    """Hero: 5 rows × the 8-method layer-max ρ bar chart with bands + ceiling."""
    n_b = len(behaviors)
    fig, axes = plt.subplots(n_b, 1, figsize=(10, 2.6 * n_b), squeeze=False)
    cols = _palette(len(METHOD_ORDER))
    for bi, behavior in enumerate(behaviors):
        ax = axes[bi][0]
        rec = results["by_behavior"][behavior]
        xs, heights, errs, labels = [], [], [], []
        for mi, m in enumerate(METHOD_ORDER):
            v = rec["methods"].get(m, {})
            rho = v.get("rho")
            xs.append(mi)
            heights.append(0.0 if rho is None else rho)
            ci = v.get("ci95")
            if rho is not None and ci:
                errs.append([max(0.0, rho - ci[0]), max(0.0, ci[1] - rho)])
            else:
                errs.append([0.0, 0.0])
            labels.append(_short_label(results, m) + ("\n(N/A)" if rho is None else ""))
        err_arr = np.array(errs).T
        ax.bar(xs, heights, yerr=err_arr, color=cols, capsize=3)
        rand = rec["methods"].get("rand_dir", {})
        band = rand.get("ci95")
        if band:
            ax.axhspan(band[0], band[1], color="gray", alpha=0.25, label="random-direction band")
        ceiling = rec.get("sqrt_r_yy_graded")
        if ceiling is not None:
            ax.axhline(ceiling, ls="--", color="black", lw=1, label="√r_yy ceiling")
        p95 = rec["methods"].get("cofit_ridge", {}).get("shuffle_null_p95")
        if p95 is not None:
            ax.axhline(p95, ls=":", color="firebrick", lw=1, label="ridge shuffle-null p95")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("held-out LOCO ρ")
        ax.set_title(f"{behavior} — matched-protocol co-fit (layer-max, graded E0)", fontsize=9)
        if bi == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_layer_curves(results: dict, behaviors: list[str], out_path: Path) -> None:
    """Per-method per-layer ρ curves (one panel per behavior) + random band."""
    n_b = len(behaviors)
    fig, axes = plt.subplots(n_b, 1, figsize=(9, 2.6 * n_b), squeeze=False)
    cols = _palette(len(METHOD_ORDER))
    for bi, behavior in enumerate(behaviors):
        ax = axes[bi][0]
        rec = results["by_behavior"][behavior]
        for mi, m in enumerate(METHOD_ORDER):
            v = rec["methods"].get(m, {})
            curve = v.get("per_layer_rho")
            if not curve:
                continue
            xs = np.arange(len(curve))
            ys = np.array([np.nan if c is None else c for c in curve], dtype=float)
            style = {"ls": ":", "lw": 1} if m == "rand_dir" else {"lw": 1.4}
            ax.plot(xs, ys, label=_short_label(results, m), color=cols[mi], **style)
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out ρ")
        ax.set_title(behavior, fontsize=9)
        if bi == 0:
            ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pred_scatter(results: dict, behaviors: list[str], out_path: Path) -> None:
    """Low-level data plot: held-out prediction vs GRADED E0, colored by context family.

    The Lens-11 pred-vs-actual read (review r1 Minor: the prior version plotted
    prediction vs context INDEX with no E0 axis and no family coloring). x =
    the frozen graded E0 target (0-100), y = the held-out LOCO prediction
    (within-fold rank scale), one point per kept context (labeled), colored by
    the context's battery family (``family_by_context`` persisted in the fit
    record).
    """
    show = ["cofit_ridge", "pv_rC", "pv_neutral"]
    # Compact plain-English panel names (label discipline: no bare method slugs
    # on reader-facing figure text; the slug lives only in the results JSON).
    panel_names = {
        "cofit_ridge": "learned ridge",
        "pv_rC": "stripped opposite-contrast direction",
        "pv_neutral": "neutral-contrast direction",
    }
    n_b = len(behaviors)
    fig, axes = plt.subplots(n_b, len(show), figsize=(4.4 * len(show), 3.2 * n_b), squeeze=False)
    for bi, behavior in enumerate(behaviors):
        rec = results["by_behavior"][behavior]
        kept = rec["kept_context_ids"]
        graded = rec.get("graded_by_context") or {}
        fams = rec.get("family_by_context") or {}
        fam_names = sorted({str(fams.get(c)) for c in kept})
        fam_color = dict(zip(fam_names, _palette(max(len(fam_names), 1)), strict=False))
        for si, m in enumerate(show):
            ax = axes[bi][si]
            v = rec["methods"].get(m, {})
            preds = v.get("preds_chosen_layer")
            if not preds or not graded:
                ax.text(0.5, 0.5, "not buildable", ha="center", va="center")
                ax.set_title(f"{behavior} — {panel_names.get(m, m)} (not buildable)", fontsize=8)
                continue
            for fam in fam_names:
                cs = [c for c in kept if str(fams.get(c)) == fam]
                xs = [graded[c] for c in cs]
                ys = [preds[c] for c in cs]
                ax.scatter(
                    xs,
                    ys,
                    s=14,
                    color=fam_color[fam],
                    label=FAMILY_LABELS.get(fam, fam),
                    alpha=0.85,
                )
                for c, x, yv in zip(cs, xs, ys, strict=True):
                    ax.annotate(
                        _ctx_point_label(c),
                        (x, yv),
                        fontsize=4.5,
                        alpha=0.75,
                        xytext=(2, 1),
                        textcoords="offset points",
                    )
            ax.set_title(f"{behavior} — {panel_names.get(m, m)}", fontsize=8)
            ax.set_xlabel("graded E0 (0-100)")
            if si == 0:
                ax.set_ylabel("held-out prediction (fit scale)")
            if bi == 0 and si == 0:
                ax.legend(fontsize=5, title="context family", title_fontsize=5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_cos_curves(manifest: dict, behaviors: list[str], out_path: Path) -> None:
    """cos(r_A, r_C) + cos(r_C, r_neutral) per layer (the #661 decomposition)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = _palette(max(2 * len(behaviors), 2))
    for bi, behavior in enumerate(behaviors):
        rec = manifest.get("by_behavior", {}).get(behavior)
        if not rec:
            continue
        c1 = rec.get("cos_rA_rC_per_layer")
        if c1:
            ax.plot(c1, color=cols[2 * bi], lw=1.4, label=f"{behavior}: cos(r_A, r_C)")
        c2 = rec.get("cos_rC_rneutral_per_layer")
        if c2:
            ax.plot(
                c2,
                color=cols[2 * bi + 1],
                ls="--",
                lw=1.2,
                label=f"{behavior}: cos(r_C, r_neutral)",
            )
    ax.axhline(0.5, ls=":", color="firebrick", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    ax.legend(fontsize=6)
    ax.set_title("direction-read decomposition: instruction-present vs stripped vs neutral")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nonlinear_panels(
    nl: dict, behaviors: list[str], out_path: Path, corrected: dict | None = None
) -> None:
    """dCor null histograms (observed + control marked) + per-behavior test stats.

    ``corrected`` is the rank-scale sign-flip correction JSON
    (``signflip_rankscale_corrected.json``); when present its stats REPLACE the
    as-run (scale-contaminated) sign-flip numbers in the side panel.
    """
    n_b = len(behaviors)
    fig, axes = plt.subplots(n_b, 2, figsize=(9, 2.8 * n_b), squeeze=False)
    for bi, behavior in enumerate(behaviors):
        cell = nl["by_behavior"].get(behavior, {})
        d10 = cell.get("option_a_d10", {})
        ax = axes[bi][0]
        null = d10.get("dcor_null") or []
        if null:
            ax.hist(null, bins=30, color="steelblue", alpha=0.7, label="refit-per-draw null")
            ax.axvline(d10.get("dcor_observed"), color="black", lw=1.5, label="observed dCor")
            ax.axvline(
                d10.get("control_task_dcor"),
                color="firebrick",
                ls="--",
                lw=1.2,
                label="shuffled-labels control",
            )
        ax.set_title(f"{behavior} — post-LEACE dCor, d=10 (refit-per-draw null)", fontsize=8)
        if bi == 0:
            ax.legend(fontsize=6)
        ax2 = axes[bi][1]
        dcor_p = d10.get("dcor_p_value")
        ctrl_p = d10.get("control_task_p_value")
        lines = [
            f"post-LEACE dCor = {d10.get('dcor_observed'):.3f} (p = {dcor_p:.4g})"
            if dcor_p is not None
            else "post-LEACE dCor: n/a",
            f"shuffled-labels control dCor = {d10.get('control_task_dcor'):.3f} (p = {ctrl_p:.4g})"
            if ctrl_p is not None
            else "control: n/a",
        ]
        corr_cell = (corrected or {}).get("by_behavior", {}).get(behavior, {})
        sfr = corr_cell.get("signflip_rankscale")
        if sfr:
            lines.append(
                f"kernel-vs-linear sign-flip (rank scale): p = {sfr['p_value']:.4g}, "
                f"mean err diff = {sfr['statistic_mean_err_diff']:+.4f}"
            )
        drho = cell.get("paired_delta_rho_krr_minus_ridge")
        if drho is not None:
            lines.append(f"held-out ρ, kernel − linear = {drho:+.3f}")
        ax2.text(0.02, 0.55, "\n".join(lines), fontsize=8, transform=ax2.transAxes)
        ax2.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _nan_if_none(v) -> float:
    """None -> np.nan so an unbuildable value renders as NO bar, never a zero bar.

    Review r1 Minor: ``v.get(...) or 0.0`` coerced None (e.g. an unbuildable
    pv_neutral) into a MISLEADING zero bar; matplotlib silently skips NaN bars.
    """
    return np.nan if v is None else float(v)


def plot_h2_and_checks(results: dict, behaviors: list[str], out_path: Path) -> None:
    """Leave-default-family-out ρ bars + ridge-parent-check + binary companion."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    x = np.arange(len(behaviors))
    # (a) leave-default-family-out (pv_neutral vs pv_rC, full vs held-out)
    ax = axes[0]
    w = 0.2
    for oi, (m, tag) in enumerate(
        [("pv_neutral", "full"), ("pv_neutral", "ldo"), ("pv_rC", "full"), ("pv_rC", "ldo")]
    ):
        vals = []
        for b in behaviors:
            v = results["by_behavior"][b]["methods"].get(m, {})
            vals.append(
                _nan_if_none(
                    v.get("rho") if tag == "full" else v.get("rho_leave_default_family_out")
                )
            )
        ax.bar(x + (oi - 1.5) * w, vals, width=w, label=f"{m} ({tag})")
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=20, ha="right", fontsize=7)
    ax.legend(fontsize=6)
    ax.set_title("H2 overlap check: leave-default-family-out ρ", fontsize=9)
    # (b) ridge parent check
    ax = axes[1]
    deltas = [
        _nan_if_none(results["by_behavior"][b]["ridge_parent_check"].get("delta"))
        for b in behaviors
    ]
    ax.bar(x, deltas, color=_palette(1)[0])
    ax.axhline(0.12, ls=":", color="gray")
    ax.axhline(-0.12, ls=":", color="gray")
    ax.axhline(0.25, ls="--", color="firebrick")
    ax.axhline(-0.25, ls="--", color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=20, ha="right", fontsize=7)
    ax.set_title("ridge_parent_check Δ (harness − parent at parent layer)", fontsize=9)
    # (c) binary companion vs graded ridge
    ax = axes[2]
    graded = [
        _nan_if_none(results["by_behavior"][b]["methods"]["cofit_ridge"].get("rho"))
        for b in behaviors
    ]
    binary = [
        _nan_if_none((results["by_behavior"][b].get("binary_companion_ridge") or {}).get("rho"))
        for b in behaviors
    ]
    ax.bar(x - 0.2, graded, width=0.4, label="graded ridge")
    ax.bar(x + 0.2, binary, width=0.4, label="binary-companion ridge")
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=20, ha="right", fontsize=7)
    ax.legend(fontsize=6)
    ax.set_title("continuity: graded vs binary ridge column", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_h2_neutral(results: dict, manifest: dict, behaviors: list[str], out_path: Path) -> None:
    """The H2 (neutral-contrast) read in one figure, plain-English labels only.

    (a) pooled held-out ρ for the neutral-contrast vs the stripped
    opposite-contrast direction, each also recomputed with the default-family
    contexts held out (the pole-overlap check); (b) neutral-arm judge yield
    (kept / rejected / unscoreable) with the keep-floor branch; (c) per-layer
    cosine between the stripped opposite-contrast and neutral-contrast
    directions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    x = np.arange(len(behaviors))
    # (a) neutral vs stripped opposite, full + default-family-held-out
    ax = axes[0]
    w = 0.2
    series = [
        ("pv_neutral", "rho", "neutral contrast (all 50 contexts)"),
        ("pv_neutral", "rho_leave_default_family_out", "neutral contrast (default family out)"),
        ("pv_rC", "rho", "stripped opposite contrast (all 50)"),
        ("pv_rC", "rho_leave_default_family_out", "stripped opposite (default family out)"),
    ]
    cols = _palette(len(series))
    for oi, (m, field, lab) in enumerate(series):
        vals = [
            _nan_if_none(results["by_behavior"][b]["methods"].get(m, {}).get(field))
            for b in behaviors
        ]
        ax.bar(x + (oi - 1.5) * w, vals, width=w, label=lab, color=cols[oi])
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("held-out LOCO ρ")
    ax.legend(fontsize=6)
    ax.set_title("neutral vs stripped opposite contrast", fontsize=9)
    # (b) neutral-arm yield
    ax = axes[1]
    kept, rej, drop, branches = [], [], [], []
    for b in behaviors:
        rec = manifest.get("by_behavior", {}).get(b, {})
        kept.append(rec.get("kept_n_used", 0))
        rej.append(rec.get("rejected_above_threshold", 0))
        drop.append(rec.get("dropped_unscoreable", 0))
        branches.append(rec.get("keep_floor_branch", "?"))
    ax.bar(x, kept, label="kept (trait-absent, score < 50)")
    ax.bar(x, rej, bottom=kept, label="rejected (trait present)")
    ax.bar(x, drop, bottom=[k + r for k, r in zip(kept, rej, strict=True)], label="unscoreable")
    for i, br in enumerate(branches):
        ax.annotate(br, (i, kept[i] + rej[i] + drop[i]), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("rollouts (of 1000)")
    ax.legend(fontsize=6)
    ax.set_title("neutral-arm judge yield", fontsize=9)
    # (c) cos(stripped opposite, neutral) per layer
    ax = axes[2]
    cols_b = _palette(max(len(behaviors), 2))
    for bi, b in enumerate(behaviors):
        rec = manifest.get("by_behavior", {}).get(b, {})
        c2 = rec.get("cos_rC_rneutral_per_layer")
        if c2:
            ax.plot(c2, color=cols_b[bi], lw=1.3, label=b)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    ax.legend(fontsize=6)
    ax.set_title("cos(stripped opposite, neutral) per layer", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_neutral_yield(manifest: dict, behaviors: list[str], out_path: Path) -> None:
    """Neutral-arm kept/rejected/dropped counts per behavior + keep-floor branch."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(behaviors))
    kept, rej, drop, branches = [], [], [], []
    for b in behaviors:
        rec = manifest.get("by_behavior", {}).get(b, {})
        kept.append(rec.get("kept_n_used", 0))
        rej.append(rec.get("rejected_above_threshold", 0))
        drop.append(rec.get("dropped_unscoreable", 0))
        branches.append(rec.get("keep_floor_branch", "?"))
    ax.bar(x, kept, label="kept (score < 50)")
    ax.bar(x, rej, bottom=kept, label="rejected (score ≥ 50)")
    ax.bar(x, drop, bottom=[k + r for k, r in zip(kept, rej, strict=True)], label="dropped")
    for i, br in enumerate(branches):
        ax.annotate(br, (i, kept[i] + rej[i] + drop[i]), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=20, ha="right", fontsize=7)
    ax.legend(fontsize=7)
    ax.set_title("neutral-arm judge yield + keep-floor branch", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763 neutral-contrast-and-cofit figures.")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    # FIRST: --smoke re-execs with EPM_ISSUE763_SMOKE_SCOPE=1 (write paths
    # rebind under smoke_scope/); the env WITHOUT --smoke fails loud.
    ensure_smoke_scope(args.smoke)
    _try_paper_style()

    results = load_json(COFIT_DIR / "cofit_results.json")
    behaviors = [b for b in args.behaviors if b in results.get("by_behavior", {})]
    assert behaviors, "no fitted behaviors found in cofit_results.json"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    hero = FIGURE_DIR / "fig_763_cofit_grid.png"
    plot_cofit_grid(results, behaviors, hero)
    plot_layer_curves(results, behaviors, FIGURE_DIR / "fig_763_cofit_layer_curves.png")
    plot_pred_scatter(results, behaviors, FIGURE_DIR / "fig_763_cofit_pred_scatter.png")
    plot_h2_and_checks(results, behaviors, FIGURE_DIR / "fig_763_cofit_h2_checks.png")

    # neutral_arm_manifest.json is a PLAN-REQUIRED deliverable (§6 cos/yield
    # panels; §6.5): on a fresh Phase-C lane it is staged from the HF prefix the
    # Phase-B --directions-only pass uploaded, and a post-staging absence is a
    # FAIL-LOUD error — never a warn-and-skip (review r1 C2 / Codex C2).
    manifest_path = COFIT_DIR / "neutral_arm_manifest.json"
    if not manifest_path.exists() and not args.smoke:
        from issue763_extract_pv_rb import _stage_single_from_hf

        _stage_single_from_hf(
            manifest_path,
            f"{HF_ANALYSIS_TENSORS_PREFIX}/cofit_manifests/neutral_arm_manifest.json",
            "neutral_arm_manifest.json (plan-required §6.5 deliverable)",
        )
    if not manifest_path.exists():
        raise RuntimeError(
            f"neutral_arm_manifest.json absent at {manifest_path} after staging — the "
            "Phase-B assemble_directions phase (and its --directions-only upload) must "
            "run first; the §6 cos/yield panels are plan-required, not skippable"
        )
    manifest = load_json(manifest_path)
    plot_cos_curves(manifest, behaviors, FIGURE_DIR / "fig_763_cofit_cos_curves.png")
    plot_neutral_yield(manifest, behaviors, FIGURE_DIR / "fig_763_cofit_neutral_yield.png")
    plot_h2_neutral(results, manifest, behaviors, FIGURE_DIR / "fig_763_cofit_h2_neutral.png")

    nl_path = COFIT_DIR / "nonlinear_tests.json"
    if nl_path.exists():
        nl = load_json(nl_path)
        corrected_path = COFIT_DIR / "signflip_rankscale_corrected.json"
        corrected = load_json(corrected_path) if corrected_path.exists() else None
        plot_nonlinear_panels(
            nl, behaviors, FIGURE_DIR / "fig_763_cofit_nonlinear.png", corrected=corrected
        )
    else:
        logger.warning("nonlinear_tests.json absent — nonlinear panels skipped")

    assert hero.exists(), "hero figure did not land"
    print(f"[issue763.cofit_plot] wrote fig_763_cofit_* under {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
