"""Task #644 driver: per-behavior convexity meta-analysis (zero-GPU, VM-only).

Runs the three-phase unified pipeline against existing eval JSONs (no model, no
pod, no training):

* ``load-data`` — read each source task's eval JSONs (snapshotting #623 from
  ``origin/issue-623``), build the unified per-(behavior, frame) raw scatters via
  ``scripts/issue644_loaders.py``.
* ``fit`` — for each scatter run the form bake-off + curvature LRT + bootstrap
  x^2 CI + Cook's-D single/top-2 LOO + log-space and logit rate-stabilization
  double-fits. Emit ``eval_results/issue_644/per_behavior_fits.json``.
* ``aggregate`` — build the geometry-convexity recurs table (MF1 geometry-frame
  numerator) + the prior-frame sensitivity table; emit
  ``figures/issue_644/convexity_table.png`` and ``convexity_table.json``; render
  the per-behavior small-multiples scatter + best-fit figure and the exploratory
  dump.

This is a single unified pipeline: ``--smoke`` runs the SAME phases on one
behavior at minimal scale (smoke IS the pipeline with a 1-behavior subset that
threads through every phase). There is no separate sweep dispatcher.

Usage::

    uv run python scripts/issue644_functional_form.py            # full run
    uv run python scripts/issue644_functional_form.py --smoke    # 1 behavior, fast bootstrap
    uv run python scripts/issue644_functional_form.py --phase load-data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless VM
import sys

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis import convexity_meta as cm
from explore_persona_space.analysis import paper_plots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue644_loaders as loaders

# All behaviors with fittable data (refusal is excluded -> new-generation follow-up).
ALL_BEHAVIORS = [
    "sycophancy_seed",
    "marker_leakage_centered",
    "marker_leakage_raw",
    "fact_leakage",
    "refusal",
]
SMOKE_BEHAVIORS = ["sycophancy_seed"]
SMOKE_BOOTSTRAP_B = 200  # fast bootstrap for the smoke (full run uses cm.BOOTSTRAP_B)


def _repo_root() -> Path:
    """Resolve the worktree repo root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def phase_load_data(
    repo_root: Path,
    behaviors: list[str],
    max_532_sources: int | None,
) -> tuple[list[cm.ScatterInput], list[dict[str, Any]]]:
    """Phase 1: snapshot #623, load every behavior's raw scatters."""
    print("[phase=load_data]")
    inputs_dir = repo_root / "eval_results" / "issue_644" / "inputs" / "issue623"
    if "sycophancy_seed" in behaviors:
        snap = loaders.snapshot_issue623(inputs_dir)
        print(f"  snapshotted #623 -> {[str(p) for p in snap.values()]}")
    scatters, exclusions = loaders.load_all_scatters(
        eval_root=repo_root,
        issue623_snapshot_dir=inputs_dir,
        behaviors=behaviors,
        max_532_sources=max_532_sources,
    )
    print(f"  loaded {len(scatters)} scatters, {len(exclusions)} excluded behaviors")
    for s in scatters:
        print(f"    {s.behavior:24s} {s.frame:40s} n={len(s.x):3d} kind={s.geometry_scalar_kind}")
    for ex in exclusions:
        print(f"    EXCLUDED {ex['behavior']}: {ex['excluded_reason'][:70]}...")
    return scatters, exclusions


def phase_fit(
    scatters: list[cm.ScatterInput],
    exclusions: list[dict[str, Any]],
    out_path: Path,
    bootstrap_b: int,
) -> list[dict[str, Any]]:
    """Phase 2: fit every scatter; write per_behavior_fits.json (checkpoint-per-phase)."""
    print("[phase=fit]")
    # Allow the smoke to use a smaller bootstrap B without touching the module default.
    if bootstrap_b != cm.BOOTSTRAP_B:
        _orig_b = cm.BOOTSTRAP_B
        cm.BOOTSTRAP_B = bootstrap_b  # type: ignore[misc]
        print(f"  bootstrap_B overridden {_orig_b} -> {bootstrap_b} (smoke)")

    records: list[dict[str, Any]] = []
    for s in scatters:
        rec = cm.analyze_scatter(s)
        records.append(rec)
        verdict = rec.get("convex_wins")
        h1 = rec.get("counts_toward_h1")
        print(
            f"    {s.behavior:24s} {s.frame:40s} n={rec['n']:3d} "
            f"convex_wins={verdict} sign={rec.get('curvature_sign')} "
            f"robust_LOO={rec.get('robust_to_leverage_LOO')} h1={h1}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "issue_644_functional_form",
        "phase": "per_behavior_fits",
        "reproducibility": cm.reproducibility_metadata({"bootstrap_B_used": bootstrap_b}),
        "n_scatters": len(records),
        "records": records,
        "excluded_behaviors": exclusions,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_path} ({len(records)} fit records)")
    return records


def phase_aggregate(
    records: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    scatters: list[cm.ScatterInput],
    fig_dir: Path,
) -> dict[str, Any]:
    """Phase 3: recurs tables + headline + scatter figures (checkpoint-per-phase)."""
    print("[phase=aggregate]")
    tables = cm.build_recurs_tables(records)
    tables["reproducibility"] = cm.reproducibility_metadata()
    tables["excluded_behaviors"] = [
        {"behavior": e["behavior"], "excluded_reason": e["excluded_reason"]} for e in exclusions
    ]

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_json = fig_dir / "convexity_table.json"
    table_json.write_text(json.dumps(tables, indent=2))
    print(f"  wrote {table_json}")
    print(
        f"  majority_verdict={tables['majority_verdict']} "
        f"(numerator={tables['h1_numerator']['n_convex_counts_toward_h1']} / "
        f"denominator={tables['h1_denominator']['n_qualifying']})"
    )

    _render_convexity_table_figure(tables, fig_dir / "convexity_table")
    _render_scatter_small_multiples(records, scatters, fig_dir)
    _render_exploratory_dump(records, scatters, fig_dir)
    return tables


# --- Figures ------------------------------------------------------------------

# Reader-facing label maps (figure-side mirror of the body's plain-English rule).
# The underlying JSON row keys (behavior / frame) are the canonical join
# identifiers and are NEVER changed here — only the RENDERED labels translate.
_BEHAVIOR_LABELS = {
    "sycophancy_seed": "Sycophancy",
    "marker_leakage_centered": "Marker leakage (centered)",
    "marker_leakage_raw": "Marker leakage",
    "fact_leakage": "Fact leakage",
    "refusal": "Refusal",
}

# Geometry-scalar kind -> reader-facing scalar name.
_SCALAR_LABELS = {
    "cosine_to_direction": "cosine to direction",
    "cosine_to_source": "cosine to source",
    "cosine_centered_centroid": "centered cosine to centroid",
    "js": "JS divergence",
    "prior_logprob": "base-prior log-prob",
    "js_deprecated_single_next_token": "JS (deprecated single-token)",
}

# Per-arm slug -> short reader name (fact-leakage arms; marker source codes stay
# as bare A1/B2/... — they are anonymous per-source labels with no English gloss).
_ARM_LABELS = {
    "arm_marine_biologist": "marine biologist",
    "arm_local_resident": "local resident",
    "arm_courthouse_architecture_historian": "courthouse historian",
    "i444_onpolicy": "on-policy",
    "i444_leak_contradictory_neg_sensitivity": "contradictory-neg",
    "i444_leak_refusal_neg_sensitivity": "refusal-neg",
}


def _pretty_behavior(behavior: str) -> str:
    """Translate a snake_case behavior key to a reader-facing label."""
    return _BEHAVIOR_LABELS.get(behavior, behavior.replace("_", " ").title())


def _pretty_frame(frame: str, geometry_scalar_kind: str) -> str:
    """Translate a slash-separated frame slug to a reader-facing parenthetical.

    Keeps the scalar name (from the geometry kind) up front and appends a short
    descriptor pulled from the frame slug (layer, source code, arm name).
    """
    scalar = _SCALAR_LABELS.get(geometry_scalar_kind, geometry_scalar_kind.replace("_", " "))
    parts = frame.split("/")
    descriptors: list[str] = []
    for p in parts[1:]:  # skip the leading geometry/sensitivity/prior bucket
        if p.startswith("L") and p[1:].isdigit():
            descriptors.append(p)  # layer, e.g. L14
        elif p.startswith("source_"):
            descriptors.append(p.split("_", 1)[1])  # A4, B2, ...
        elif p.startswith("arm_") or p.startswith("i444"):
            descriptors.append(_ARM_LABELS.get(p, p.replace("arm_", "").replace("_", " ")))
        elif p == "joint":
            descriptors.append("joint")
        elif "persona" in p:
            # lt_persona_lt_syc -> "lt-persona"; ravg_persona_lt_syc -> "ravg-persona"
            descriptors.append(p.split("_persona")[0].replace("_", "-") + "-persona")
    suffix = f" ({', '.join(descriptors)})" if descriptors else ""
    return f"{scalar}{suffix}"


def _render_convexity_table_figure(tables: dict[str, Any], stem_path: Path) -> None:
    """Render the cross-behavior geometry-recurs table as a figure (companion hero)."""
    paper_plots.set_paper_style("blog")
    rows = tables["geometry_recurs_table"]
    if not rows:
        return
    headers = [
        "Behavior",
        "Geometry frame",
        "n",
        "Convex?",
        "Curvature sign",
        "ΔAIC",
        "Survives leverage drop?",
        "Rate artifact?",
        "Counts as robust convex?",
    ]
    cell_text = []
    for r in rows:
        n = r.get("n", 0)
        below_floor = isinstance(n, int) and n < 10
        behavior_label = _pretty_behavior(r.get("behavior", ""))
        if below_floor:
            behavior_label += "  [n < 10, excluded]"
        cell_text.append(
            [
                behavior_label,
                _pretty_frame(r.get("frame", ""), r.get("geometry_scalar_kind", "")),
                str(n),
                _fmt_bool(r.get("convex_wins")),
                str(r.get("curvature_sign", "")),
                _fmt_num(r.get("delta_aic_linear_to_best")),
                _fmt_bool(r.get("robust_to_leverage_LOO")),
                _fmt_bool(r.get("rate_compression_artifact")),
                _fmt_bool(r.get("counts_toward_h1")),
            ]
        )
    fig_h = max(2.0, 0.35 * (len(cell_text) + 2))
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.3)
    num = tables["h1_numerator"]["n_convex_counts_toward_h1"]
    den = tables["h1_denominator"]["n_qualifying"]
    ax.set_title(
        f"No portable convex shape: {num} of {den} qualifying geometry frames "
        f"are robustly convex\n"
        f"(robust = beats linear by ΔAIC ≥ 2, same-signed curvature CI excludes 0, "
        f"survives leverage + rate controls)",
        fontsize=9,
        loc="left",
    )
    paper_plots.savefig_paper(fig, stem_path.name, dir=str(stem_path.parent), formats=("png",))
    plt.close(fig)


def _render_scatter_small_multiples(
    records: list[dict[str, Any]],
    scatters: list[cm.ScatterInput],
    fig_dir: Path,
) -> None:
    """Hero: per geometry-frame scatter, raw points + best-fit curve over the linear fit."""
    paper_plots.set_paper_style("blog")
    geo = [
        (r, s)
        for r, s in zip(records, scatters, strict=True)
        if r.get("geometry_scalar_kind") in cm.GEOMETRY_SCALAR_KINDS and r.get("two_axis_spread_ok")
    ]
    if not geo:
        return
    n = len(geo)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows), squeeze=False)
    primary = paper_plots.paper_palette_role("primary")
    baseline = paper_plots.paper_palette_role("baseline")
    for idx, (rec, s) in enumerate(geo):
        ax = axes[idx // ncols][idx % ncols]
        x = np.asarray(s.x, float)
        y = np.asarray(s.y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        ax.scatter(x, y, s=18, color=primary, alpha=0.8, zorder=3)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        lin = cm.fit_linear(x, y)
        a, b = lin["coef"]
        ax.plot(xs, a + b * xs, "--", color=baseline, lw=1.2, label="linear")
        best = rec.get("best_form")
        if best and best != "linear":
            curve = _best_curve(best, x, y, xs)
            if curve is not None:
                ax.plot(xs, curve, "-", color=primary, lw=1.6, label=f"best ({best})")
        daic = rec.get("delta_aic_linear_to_best")
        ci_lo = rec.get("curvature_ci_low")
        ci_hi = rec.get("curvature_ci_high")
        ax.set_title(
            f"{_pretty_behavior(s.behavior)}\n{_pretty_frame(s.frame, s.geometry_scalar_kind)}\n"
            f"ΔAIC={_fmt_num(daic)}  x²CI=[{_fmt_num(ci_lo)},{_fmt_num(ci_hi)}]",
            fontsize=7,
        )
        ax.set_xlabel("geometry proximity (raw)")
        ax.set_ylabel("behavior strength (rate)")
        ax.legend(fontsize=6)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout()
    paper_plots.savefig_paper(
        fig, "scatter_best_fit_small_multiples", dir=str(fig_dir), formats=("png",)
    )
    plt.close(fig)


def _render_exploratory_dump(
    records: list[dict[str, Any]],
    scatters: list[cm.ScatterInput],
    fig_dir: Path,
) -> None:
    """Exploratory dump: raw-vs-logit overlay per rate DV (rate-compression diagnostic)."""
    paper_plots.set_paper_style("blog")
    rate_rows = [
        (r, s)
        for r, s in zip(records, scatters, strict=True)
        if s.y_is_rate and r.get("two_axis_spread_ok")
    ]
    if not rate_rows:
        return
    n = len(rate_rows)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    primary = paper_plots.paper_palette_role("primary")
    accent = paper_plots.paper_palette_role("accent")
    for idx, (rec, s) in enumerate(rate_rows):
        ax = axes[idx // ncols][idx % ncols]
        x = np.asarray(s.x, float)
        y = np.asarray(s.y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        ax.scatter(x, y, s=14, color=primary, alpha=0.8, label="raw rate")
        ax2 = ax.twinx()
        ax2.scatter(x, cm.logit_clip(y), s=14, color=accent, alpha=0.6, marker="^", label="logit")
        rate_art = "yes" if rec.get("rate_compression_artifact") else "no"
        behavior_label = _pretty_behavior(s.behavior)
        frame_label = _pretty_frame(s.frame, s.geometry_scalar_kind)
        ax.set_title(
            f"{behavior_label} — {frame_label}\nrate artifact: {rate_art}",
            fontsize=7,
        )
        ax.set_xlabel("geometry proximity")
        ax.set_ylabel("raw rate", color=primary)
        ax2.set_ylabel("logit(rate)", color=accent)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout()
    paper_plots.savefig_paper(fig, "raw_vs_logit_overlay", dir=str(fig_dir), formats=("png",))
    plt.close(fig)


def _best_curve(form: str, x: np.ndarray, y: np.ndarray, xs: np.ndarray) -> np.ndarray | None:
    """Predict the best-form curve over ``xs`` for the scatter (for plotting)."""
    if form == "quadratic":
        f = cm.fit_quadratic(x, y)
        a, b, c = f["coef"]
        return a + b * xs + c * xs**2
    if form == "exp":
        f = cm.fit_exponential(x, y)
        if f is None:
            return None
        a, b = f["coef"]
        return a * np.exp(b * xs)
    if form == "power":
        f = cm.fit_power(x, y)
        if f is None:
            return None
        a, b = f["coef"]
        return a * np.power(xs - f["x_shift"], b)
    if form == "spline":
        from scipy.interpolate import PchipInterpolator

        qs = np.linspace(0.0, 1.0, cm.SPLINE_KNOTS)
        knots = np.unique(np.quantile(x, qs))
        if len(knots) < 2:
            return None
        order = np.argsort(x)
        knot_y = np.interp(knots, x[order], y[order])
        return PchipInterpolator(knots, knot_y, extrapolate=True)(xs)
    return None


def _fmt_num(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{float(v):.2f}"


def _fmt_bool(v: Any) -> str:
    if v is None:
        return "—"
    return "Y" if v else "n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Task #644 convexity meta-analysis driver.")
    ap.add_argument(
        "--phase",
        choices=["all", "load-data", "fit", "aggregate"],
        default="all",
        help="Run a single phase or the full pipeline (default: all).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run: 1 behavior, small bootstrap B (same pipeline, tiny subset).",
    )
    ap.add_argument(
        "--behaviors",
        nargs="*",
        default=None,
        help="Restrict to these behaviors (default: all). Threads through every phase.",
    )
    ap.add_argument(
        "--max-532-sources",
        type=int,
        default=None,
        help="Cap #532 per-source scatters (smoke / fast iteration).",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    behaviors = args.behaviors
    bootstrap_b = cm.BOOTSTRAP_B
    max_532 = args.max_532_sources
    if args.smoke:
        behaviors = behaviors or SMOKE_BEHAVIORS
        bootstrap_b = SMOKE_BOOTSTRAP_B
        if max_532 is None:
            max_532 = 1
    if behaviors is None:
        behaviors = ALL_BEHAVIORS

    print(f"[issue644] phase={args.phase} smoke={args.smoke} behaviors={behaviors} B={bootstrap_b}")

    fits_path = repo_root / "eval_results" / "issue_644" / "per_behavior_fits.json"
    if args.smoke:
        fits_path = repo_root / "eval_results" / "issue_644" / "smoke_per_behavior_fits.json"
    fig_dir = repo_root / "figures" / "issue_644"
    if args.smoke:
        fig_dir = repo_root / "figures" / "issue_644" / "smoke"

    # The pipeline is unified: every phase reads the SAME behavior subset.
    scatters, exclusions = phase_load_data(repo_root, behaviors, max_532)
    if args.phase == "load-data":
        print("[phase=done] load-data complete")
        return

    records = phase_fit(scatters, exclusions, fits_path, bootstrap_b)
    if args.phase == "fit":
        print("[phase=done] fit complete")
        return

    phase_aggregate(records, exclusions, scatters, fig_dir)
    print("[phase=done] aggregate complete")


if __name__ == "__main__":
    main()
