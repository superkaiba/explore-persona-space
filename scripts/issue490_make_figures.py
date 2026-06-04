#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Issue #490 PHASE 5 — figures (hero + companions + exploratory dump).

Per plan v1 §4.5 PHASE 5 + §6.3:

Hero figure (named):
  figures/issue_490/hero_dose_decomposition.png
    — grouped bar at K=2:
        gap_dosematched(on-axis) | gap_dosematched(off-axis)
        + gap_confounded + slope_dose for context
      mean combiner, 95% paired-bootstrap CIs, horizontal at 0.
      Title reads the Q2 verdict directly (Δ_geom + CI).

Companion:
  figures/issue_490/combiner_robustness.png
    — Δ_geom under mean / lse / max side by side.

Exploratory dump:
  figures/issue_490/per_pair_bars.png
  figures/issue_490/delta_geom_vs_pair_dist.png
  figures/issue_490/per_source_asymmetry.png
  figures/issue_490/fallback_kl_hero.png
  figures/issue_490/saturation_diagnostic.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    log.error("matplotlib not installed; figures step skipped")
    sys.exit(1)

from _issue490_common import COMBINERS  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            env={**__import__("os").environ},
        ).strip()
    except Exception:
        return "unknown"


def _save_with_meta(fig, png_path: Path, meta: dict) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    meta_path = png_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({**meta, "git_commit": _git_commit()}, indent=2))
    log.info("Saved %s + %s + %s", png_path.name, pdf_path.name, meta_path.name)
    plt.close(fig)


def hero_distance_adjusted(decomp: dict, out_dir: Path) -> None:
    """PRIMARY hero (round-2): distance-adjusted is_on_axis coefficient with
    its 95% CI as the headline; raw subpanel-mean Δ_geom shown beside it as
    "diagnostic-only" for context.
    """
    reg = decomp["primary"]["distance_adjusted_regression"]
    diag = decomp["primary"]["diagnostic_unadjusted_subpanel_means"]["per_combiner"]["mean"]
    dgeom_raw = diag["delta_geom_raw_unadjusted"]

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = []
    means = []
    err_lo = []
    err_hi = []

    if reg.get("status") == "OK":
        beta = reg["headline_beta"]
        ci = reg["headline_ci95"]
        labels.append("is_on_axis β\n(distance-adjusted,\nPRIMARY Q2)")
        means.append(beta)
        err_lo.append(max(0.0, beta - ci[0]))
        err_hi.append(max(0.0, ci[1] - beta))
    else:
        labels.append("is_on_axis β\n(SKIPPED)")
        means.append(0.0)
        err_lo.append(0.0)
        err_hi.append(0.0)

    if dgeom_raw["mean"] is not None:
        labels.append("raw Δ_geom\n(unadjusted,\ndiagnostic)")
        means.append(dgeom_raw["mean"])
        err_lo.append(max(0.0, dgeom_raw["mean"] - dgeom_raw["ci95"][0]))
        err_hi.append(max(0.0, dgeom_raw["ci95"][1] - dgeom_raw["mean"]))
    else:
        labels.append("raw Δ_geom (no data)")
        means.append(0.0)
        err_lo.append(0.0)
        err_hi.append(0.0)

    xs = np.arange(len(labels))
    colors = ["#1f77b4", "#888888"]
    ax.bar(xs, means, yerr=[err_lo, err_hi], capsize=6, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("nats")
    verdict = reg.get("verdict_distance_adjusted", "n/a")
    if reg.get("status") == "OK":
        ax.set_title(
            f"PRIMARY Q2 (distance-adjusted): is_on_axis β = {reg['headline_beta']:.3f} "
            f"[{reg['headline_ci95'][0]:.3f}, {reg['headline_ci95'][1]:.3f}]\n"
            f"p = {reg['headline_p']:.3f}, n = {reg['n_rows']} personas, "
            f"{reg['n_clusters']} (pair × seed) clusters\n"
            f"verdict: {verdict}",
            fontsize=10,
        )
    else:
        ax.set_title(
            f"distance-adjusted regression {reg.get('status', '?')}: {reg.get('reason', 'n/a')}",
            fontsize=10,
        )
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "hero_distance_adjusted.png",
        {
            "kind": "hero_distance_adjusted",
            "labels": labels,
            "means": means,
            "is_on_axis_regression": reg,
            "delta_geom_raw_diagnostic": dgeom_raw,
            "verdict": verdict,
        },
    )


def hero_dose_decomposition(decomp: dict, out_dir: Path) -> None:
    """Companion (was hero in round 1, now demoted to diagnostic):
    grouped bar of gap_dosematched(on/off) + gap_confounded + slope_dose
    under the mean combiner with 95% paired-bootstrap CIs.
    """
    diag = decomp["primary"]["diagnostic_unadjusted_subpanel_means"]["per_combiner"]["mean"]
    quantities = [
        ("gap_confounded\n(on-axis)", diag["gap_confounded_on_axis"]),
        ("gap_dosematched\n(on-axis)", diag["gap_dosematched_on_axis"]),
        ("gap_dosematched\n(off-axis)", diag["gap_dosematched_off_axis"]),
        ("slope_dose\n(per-source)", diag["slope_dose"]),
    ]

    means = [q[1]["mean"] if q[1]["mean"] is not None else 0.0 for q in quantities]
    err_lo = [
        (q[1]["mean"] - q[1]["ci95"][0]) if q[1]["mean"] is not None else 0.0 for q in quantities
    ]
    err_hi = [
        (q[1]["ci95"][1] - q[1]["mean"]) if q[1]["mean"] is not None else 0.0 for q in quantities
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(quantities))
    colors = ["#888888", "#1f77b4", "#ff7f0e", "#2ca02c"]
    ax.bar(
        xs,
        means,
        yerr=[err_lo, err_hi],
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([q[0] for q in quantities])
    ax.set_ylabel("nats (mean over (pair × seed) values)")
    dgeom = diag["delta_geom_raw_unadjusted"]
    dgeom_str = (
        f"{dgeom['mean']:.3f} [{dgeom['ci95'][0]:.3f}, {dgeom['ci95'][1]:.3f}]"
        if dgeom["mean"] is not None
        else "n/a"
    )
    ax.set_title(
        f"DIAGNOSTIC (raw, unadjusted): Δ_geom = "
        f"gap_dosematched(on) − gap_dosematched(off) = {dgeom_str} nats "
        f"(n={dgeom['n']} tuples)\n"
        f"NB: on-axis and off-axis subpanels are not always mean-d-matched; "
        f"see distance-adjusted hero.",
        fontsize=10,
    )
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "hero_dose_decomposition.png",
        {
            "kind": "diagnostic_dose_decomposition_mean_combiner",
            "combiner": "mean",
            "quantities": [q[0] for q in quantities],
            "means": means,
            "delta_geom_raw_diagnostic": dgeom,
        },
    )


def combiner_robustness(decomp: dict, out_dir: Path) -> None:
    """Raw (unadjusted) Δ_geom under each of mean / lse / max — diagnostic."""
    per_c = decomp["primary"]["diagnostic_unadjusted_subpanel_means"]["per_combiner"]
    means = [per_c[c]["delta_geom_raw_unadjusted"]["mean"] for c in COMBINERS]
    means = [m if m is not None else 0.0 for m in means]
    cis_lo = [per_c[c]["delta_geom_raw_unadjusted"]["ci95"][0] for c in COMBINERS]
    cis_hi = [per_c[c]["delta_geom_raw_unadjusted"]["ci95"][1] for c in COMBINERS]
    err_lo = [
        (m - ci_lo) if (m is not None and ci_lo is not None) else 0.0
        for m, ci_lo in zip(means, cis_lo, strict=True)
    ]
    err_hi = [
        (ci_hi - m) if (m is not None and ci_hi is not None) else 0.0
        for m, ci_hi in zip(means, cis_hi, strict=True)
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(COMBINERS))
    ax.bar(
        xs,
        means,
        yerr=[err_lo, err_hi],
        capsize=6,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(COMBINERS)
    ax.set_ylabel("raw Δ_geom (nats, diagnostic)")
    ax.set_title("Raw Δ_geom under each combiner (DIAGNOSTIC; primary = distance-adjusted)")
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "combiner_robustness.png",
        {
            "kind": "combiner_robustness_diagnostic",
            "combiners": list(COMBINERS),
            "means": means,
        },
    )


def per_pair_bars(tidy_rows: list[dict], out_dir: Path) -> None:
    """Per-pair grouped bars: gap_dosematched(on) vs (off) under mean combiner."""
    by_pair: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"on": [], "off": []})
    for r in tidy_rows:
        if r.get("value_key") != "deltaLogP_mean":
            continue
        key = "on" if r["subpanel"] == "on_axis" else "off"
        by_pair[r["pair_id"]][key].append(r["gap_dosematched_mean"])

    pair_ids = sorted(by_pair.keys())
    if not pair_ids:
        log.warning("per_pair_bars: no rows to plot")
        return
    on_means = [float(np.mean(by_pair[p]["on"])) if by_pair[p]["on"] else 0.0 for p in pair_ids]
    off_means = [float(np.mean(by_pair[p]["off"])) if by_pair[p]["off"] else 0.0 for p in pair_ids]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(pair_ids))
    width = 0.4
    ax.bar(xs - width / 2, on_means, width, label="on-axis", color="#1f77b4", edgecolor="black")
    ax.bar(
        xs + width / 2,
        off_means,
        width,
        label="off-axis",
        color="#ff7f0e",
        edgecolor="black",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(pair_ids, rotation=30, ha="right")
    ax.set_ylabel("gap_dosematched (mean combiner, nats)")
    ax.set_title("Per-pair gap_dosematched: on-axis vs off-axis (mean combiner)")
    ax.legend()
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "per_pair_bars.png",
        {
            "kind": "per_pair_bars",
            "pair_ids": pair_ids,
            "on_means": on_means,
            "off_means": off_means,
        },
    )


def per_source_asymmetry_plot(decomp: dict, out_dir: Path) -> None:
    asym = decomp["primary"].get("asymmetry_pooled_A_vs_B", {})
    if not asym:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sub_names = ["on_axis", "off_axis"]
    A_vals = [asym.get(s, {}).get("pooled_2D_A_mean", 0.0) for s in sub_names]
    B_vals = [asym.get(s, {}).get("pooled_2D_B_mean", 0.0) for s in sub_names]
    xs = np.arange(len(sub_names))
    width = 0.4
    ax.bar(xs - width / 2, A_vals, width, label="POOLED-2D-A", color="#1f77b4")
    ax.bar(xs + width / 2, B_vals, width, label="POOLED-2D-B", color="#ff7f0e")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(sub_names)
    ax.set_ylabel("mean held-out log P(※) trained − base (nats)")
    ax.set_title("Per-source asymmetry: POOLED-2D-A vs POOLED-2D-B")
    ax.legend()
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "per_source_asymmetry.png",
        {"kind": "per_source_asymmetry", "A_means": A_vals, "B_means": B_vals},
    )


def fallback_kl_hero(decomp: dict, out_dir: Path) -> None:
    headlines = decomp["fallback"]["diagnostic_unadjusted_subpanel_means"]["per_combiner"]["mean"]
    dgeom = headlines["delta_geom_raw_unadjusted"]
    on = headlines["gap_dosematched_on_axis"]
    off = headlines["gap_dosematched_off_axis"]
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(2)
    means = [on["mean"] or 0.0, off["mean"] or 0.0]
    err_lo = [
        (m - ci_lo) if (m is not None and ci_lo is not None) else 0.0
        for m, ci_lo in zip(
            [on["mean"], off["mean"]],
            [on["ci95"][0], off["ci95"][0]],
            strict=True,
        )
    ]
    err_hi = [
        (ci_hi - m) if (m is not None and ci_hi is not None) else 0.0
        for m, ci_hi in zip(
            [on["mean"], off["mean"]],
            [on["ci95"][1], off["ci95"][1]],
            strict=True,
        )
    ]
    ax.bar(xs, means, yerr=[err_lo, err_hi], capsize=6, color=["#1f77b4", "#ff7f0e"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["on-axis", "off-axis"])
    ax.set_ylabel("KL(trained ‖ base) at post-R slot")
    dgeom_str = (
        f"{dgeom['mean']:.3f} [{dgeom['ci95'][0]:.3f}, {dgeom['ci95'][1]:.3f}]"
        if dgeom["mean"] is not None
        else "n/a"
    )
    ax.set_title(f"Fallback DV (KL-from-base) — Δ_geom = {dgeom_str} (n={dgeom['n']})")
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "fallback_kl_hero.png",
        {"kind": "fallback_kl_hero", "delta_geom": dgeom},
    )


def saturation_diagnostic(decomp: dict, out_dir: Path) -> None:
    sat = decomp.get("saturation_per_condition", {})
    if not sat:
        return
    conditions = sorted(sat.keys())
    means = [sat[c]["mean_g_logprob_source"] or 0.0 for c in conditions]
    n_sat = [sat[c]["n_saturated_cells"] for c in conditions]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(conditions))
    ax.bar(xs, means, color="#1f77b4", edgecolor="black", linewidth=0.6)
    ax.axhline(-0.1, color="red", linewidth=0.8, linestyle="--", label="kill threshold (−0.1)")
    ax.axhline(-1.0, color="orange", linewidth=0.8, linestyle="--", label="near-sat (−1.0)")
    ax.set_xticks(xs)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("mean trained-source log P(※) (nats)")
    ax.set_title("Saturation diagnostic per condition (lower = farther from ceiling)")
    for i, n in enumerate(n_sat):
        if n > 0:
            ax.text(
                i,
                means[i] + 0.1,
                f"sat: {n}",
                ha="center",
                color="red",
                fontsize=9,
            )
    ax.legend()
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "saturation_diagnostic.png",
        {"kind": "saturation_diagnostic", "conditions": conditions, "means": means},
    )


def delta_geom_vs_pair_dist(decomp: dict, out_dir: Path) -> None:
    pooled = (
        decomp["primary"]
        .get("pair_separation_regression", {})
        .get("per_combiner", {})
        .get("mean", {})
        .get("pooled", {})
    )
    if not pooled or pooled.get("slope") is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.text(
        0.5,
        0.5,
        f"Δ_geom ~ cos_dist(A,B) (pooled)\n"
        f"slope={pooled['slope']:.3f} se={pooled.get('se', 0):.3f} "
        f"p={pooled.get('p', float('nan')):.3f}\n"
        f"R²={pooled.get('r_squared', 0):.3f}  n={pooled['n']}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pair-separation secondary regression")
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "delta_geom_vs_pair_dist.png",
        {"kind": "pair_separation_regression_summary", **pooled},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decomp",
        type=str,
        default=str(
            PROJECT_ROOT / "eval_results" / "issue_490" / "aggregate" / "decomposition.json"
        ),
    )
    parser.add_argument(
        "--tidy",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_490" / "aggregate" / "tidy_primary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "figures" / "issue_490"),
    )
    args = parser.parse_args()

    decomp_path = Path(args.decomp)
    tidy_path = Path(args.tidy)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not decomp_path.exists():
        raise SystemExit(
            f"decomposition.json missing: {decomp_path}. Run issue490_analyze.py first."
        )
    decomp = json.loads(decomp_path.read_text())

    tidy_rows: list[dict] = []
    if tidy_path.exists():
        import csv

        with tidy_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Coerce floats.
                for k in (
                    "shared_2D",
                    "pooled_2D_A",
                    "pooled_2D_B",
                    "single_D_A",
                    "single_D_B",
                    "slope_dose",
                    "gap_confounded_mean",
                    "gap_confounded_lse",
                    "gap_confounded_max",
                    "gap_dosematched_mean",
                    "gap_dosematched_lse",
                    "gap_dosematched_max",
                ):
                    if k in row and row[k] not in ("", None):
                        import contextlib

                        with contextlib.suppress(ValueError):
                            row[k] = float(row[k])
                tidy_rows.append(row)

    # PRIMARY (round-2): distance-adjusted regression headline.
    hero_distance_adjusted(decomp, out_dir)
    # Diagnostic companions.
    hero_dose_decomposition(decomp, out_dir)
    combiner_robustness(decomp, out_dir)
    if tidy_rows:
        per_pair_bars(tidy_rows, out_dir)
    per_source_asymmetry_plot(decomp, out_dir)
    fallback_kl_hero(decomp, out_dir)
    saturation_diagnostic(decomp, out_dir)
    delta_geom_vs_pair_dist(decomp, out_dir)

    log.info("Figures written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
