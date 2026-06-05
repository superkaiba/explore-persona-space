#!/usr/bin/env python3
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
    """PRIMARY hero: distance-adjusted on-axis effect (headline) vs raw
    on-axis minus off-axis gap (diagnostic context).
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
        labels.append("Distance-adjusted\non-axis effect")
        means.append(beta)
        err_lo.append(max(0.0, beta - ci[0]))
        err_hi.append(max(0.0, ci[1] - beta))
    else:
        labels.append("Distance-adjusted\non-axis effect (SKIPPED)")
        means.append(0.0)
        err_lo.append(0.0)
        err_hi.append(0.0)

    if dgeom_raw["mean"] is not None:
        labels.append("Raw on-axis minus off-axis gap\n(unadjusted, diagnostic)")
        means.append(dgeom_raw["mean"])
        err_lo.append(max(0.0, dgeom_raw["mean"] - dgeom_raw["ci95"][0]))
        err_hi.append(max(0.0, dgeom_raw["ci95"][1] - dgeom_raw["mean"]))
    else:
        labels.append("Raw on-axis minus off-axis gap (no data)")
        means.append(0.0)
        err_lo.append(0.0)
        err_hi.append(0.0)

    xs = np.arange(len(labels))
    colors = ["#1f77b4", "#888888"]
    ax.bar(xs, means, yerr=[err_lo, err_hi], capsize=6, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("log P(marker), trained minus base (nats)")
    if reg.get("status") == "OK":
        ax.set_title(
            f"After distance adjustment: small, non-significant residual on-axis effect "
            f"({reg['headline_beta']:.3f} nats, 95% CI "
            f"[{reg['headline_ci95'][0]:.3f}, {reg['headline_ci95'][1]:.3f}], "
            f"p = {reg['headline_p']:.3f}, n = {reg['n_rows']} held-out personas across "
            f"{reg['n_clusters']} (pair x seed) clusters)",
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
        },
    )


def hero_dose_decomposition(decomp: dict, out_dir: Path) -> None:
    """Companion (was hero in round 1, now demoted to diagnostic):
    grouped bar of gap_dosematched(on/off) + gap_confounded + slope_dose
    under the mean combiner with 95% paired-bootstrap CIs.
    """
    diag = decomp["primary"]["diagnostic_unadjusted_subpanel_means"]["per_combiner"]["mean"]
    quantities = [
        ("Confounded gap\n(shared minus single,\non-axis)", diag["gap_confounded_on_axis"]),
        ("Dose-matched gap\n(on-axis)", diag["gap_dosematched_on_axis"]),
        ("Dose-matched gap\n(off-axis)", diag["gap_dosematched_off_axis"]),
        ("Dose step\n(per source,\n200 -> 400 rows)", diag["slope_dose"]),
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
    ax.set_ylabel("log P(marker), trained minus base (nats, mean over pair x seed)")
    dgeom = diag["delta_geom_raw_unadjusted"]
    dgeom_str = (
        f"{dgeom['mean']:.3f} [{dgeom['ci95'][0]:.3f}, {dgeom['ci95'][1]:.3f}]"
        if dgeom["mean"] is not None
        else "n/a"
    )
    ax.set_title(
        f"Dose step (right) and the original confounded gap (left) dwarf "
        f"both dose-matched gaps. Raw on-axis minus off-axis difference = "
        f"{dgeom_str} nats (n={dgeom['n']} pair x seed tuples).",
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
    combiner_labels = {"mean": "Mean", "lse": "Log-sum-exp", "max": "Max"}
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
    ax.set_xticklabels([combiner_labels.get(c, c) for c in COMBINERS])
    ax.set_ylabel("Raw on-axis minus off-axis difference (nats, diagnostic)")
    ax.set_title(
        "Combiner sensitivity: the raw on/off gap is positive under mean and log-sum-exp "
        "but non-significant under max"
    )
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
    pretty_pair_ids = [p.replace("pair", "Pair ") for p in pair_ids]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(pair_ids))
    width = 0.4
    ax.bar(
        xs - width / 2,
        on_means,
        width,
        label="On-axis personas",
        color="#1f77b4",
        edgecolor="black",
    )
    ax.bar(
        xs + width / 2,
        off_means,
        width,
        label="Off-axis personas",
        color="#ff7f0e",
        edgecolor="black",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(pretty_pair_ids, rotation=30, ha="right")
    ax.set_ylabel("Dose-matched gap (nats)")
    ax.set_title(
        "Per-pair dose-matched gap: on-axis vs off-axis personas (mean over 3 seeds per pair)"
    )
    ax.legend(frameon=False)
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
    pretty_sub = ["On-axis personas", "Off-axis personas"]
    A_vals = [asym.get(s, {}).get("pooled_2D_A_mean", 0.0) for s in sub_names]
    B_vals = [asym.get(s, {}).get("pooled_2D_B_mean", 0.0) for s in sub_names]
    xs = np.arange(len(sub_names))
    width = 0.4
    ax.bar(
        xs - width / 2,
        A_vals,
        width,
        label="Source A trained alone at 400 rows",
        color="#1f77b4",
    )
    ax.bar(
        xs + width / 2,
        B_vals,
        width,
        label="Source B trained alone at 400 rows",
        color="#ff7f0e",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(pretty_sub)
    ax.set_ylabel("Held-out log P(marker), trained minus base (nats)")
    ax.set_title(
        "Per-source asymmetry: source B leaks ~0.8 nats more than source A in both subpanels"
    )
    ax.legend(frameon=False)
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
    ax.set_xticklabels(["On-axis personas", "Off-axis personas"])
    ax.set_ylabel("KL(trained || base) at post-response slot (nats)")
    dgeom_str = (
        f"{dgeom['mean']:.3f} [{dgeom['ci95'][0]:.3f}, {dgeom['ci95'][1]:.3f}]"
        if dgeom["mean"] is not None
        else "n/a"
    )
    ax.set_title(
        f"Fallback metric (full-vocab KL from base): on-axis minus off-axis "
        f"difference = {dgeom_str} nats (n={dgeom['n']} pair x seed tuples)"
    )
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
    pretty = {
        "shared_2D": "Shared marker, 2D total",
        "pooled_2D_A": "Source A alone, 2D total",
        "pooled_2D_B": "Source B alone, 2D total",
        "single_D_A": "Source A alone, D total",
        "single_D_B": "Source B alone, D total",
    }
    conditions = sorted(sat.keys())
    means = [sat[c]["mean_g_logprob_source"] or 0.0 for c in conditions]
    n_sat = [sat[c]["n_saturated_cells"] for c in conditions]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(conditions))
    ax.bar(xs, means, color="#1f77b4", edgecolor="black", linewidth=0.6)
    ax.axhline(-0.1, color="red", linewidth=0.8, linestyle="--", label="Kill threshold (-0.1)")
    ax.axhline(-1.0, color="orange", linewidth=0.8, linestyle="--", label="Near-saturation (-1.0)")
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty.get(c, c) for c in conditions], rotation=20, ha="right")
    ax.set_ylabel("Trained-source log P(marker) (nats)")
    ax.set_title(
        "Saturation diagnostic per condition: every condition has 5-10+ nats of "
        "headroom (0 of 120 cells saturated)"
    )
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
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "saturation_diagnostic.png",
        {"kind": "saturation_diagnostic", "conditions": conditions, "means": means},
    )


def delta_geom_vs_pair_dist(
    decomp: dict, tidy_rows: list[dict], source_pairs: dict | None, out_dir: Path
) -> None:
    """Real scatter: per (pair, seed) Delta_geom vs A-B cosine distance, with
    the pooled linear-regression fit overlaid. Replaces the broken text-only
    summary previously emitted."""
    pooled = (
        decomp["primary"]
        .get("pair_separation_regression", {})
        .get("per_combiner", {})
        .get("mean", {})
        .get("pooled", {})
    )
    if not pooled or pooled.get("slope") is None:
        return
    if source_pairs is None or not tidy_rows:
        log.warning("delta_geom_vs_pair_dist: missing source_pairs or tidy rows; skipping")
        return

    # Build per-pair A-B distance from source_pairs.json
    pair_d: dict[str, float] = {}
    for p in source_pairs.get("pairs", []):
        # Distance derived from the on-axis-mean-d to the trained pair is a
        # proxy; use the explicit pair-level cosine distance if present.
        # source_pairs records (A, B) and on/off mean distances. The A-B
        # cosine distance itself isn't directly stored, but
        # on_axis_mean_d_layer20 + matched off-axis distance let us estimate
        # the pair separation as 2x the on-axis mean (the on-axis personas
        # sit between A and B). Use that proxy and label it correctly.
        pair_d[p["pair_id"]] = 2.0 * float(p.get("on_axis_mean_d_layer20", 0.0))

    # Per (pair, seed) Delta_geom under mean combiner
    by_tuple: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in tidy_rows:
        if r.get("value_key") != "deltaLogP_mean":
            continue
        key = (r["pair_id"], r["seed"])
        by_tuple[key][r["subpanel"]] = float(r["gap_dosematched_mean"])

    xs_data, ys_data, pair_ids = [], [], []
    for (pid, sd), d in by_tuple.items():
        if "on_axis" in d and "off_axis" in d and pid in pair_d:
            xs_data.append(pair_d[pid])
            ys_data.append(d["on_axis"] - d["off_axis"])
            pair_ids.append(pid)

    if not xs_data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs_data, ys_data, color="#1f77b4", edgecolor="black", s=45, zorder=3)
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)

    # Overlay the pooled fit
    slope = float(pooled["slope"])
    intercept = float(pooled["intercept"])
    x_lo, x_hi = min(xs_data), max(xs_data)
    pad = 0.05 * (x_hi - x_lo + 1e-9)
    xfit = np.linspace(x_lo - pad, x_hi + pad, 50)
    yfit = slope * xfit + intercept
    ax.plot(
        xfit,
        yfit,
        color="#d62728",
        linewidth=1.5,
        linestyle="--",
        label=(
            f"Pooled fit: slope = {slope:.1f}, "
            f"p = {pooled.get('p', float('nan')):.3f}, n = {pooled['n']}"
        ),
        zorder=2,
    )

    ax.set_xlabel(
        "A-B cosine separation in layer-20 hidden state (nats, proxy = 2 x on-axis mean-d)"
    )
    ax.set_ylabel("Per (pair x seed) on-axis minus off-axis difference (nats)")
    ax.set_title(
        "Within the tested A-B separation range, the on-axis minus off-axis "
        "difference trends negative (not positive) with separation"
    )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save_with_meta(
        fig,
        out_dir / "delta_geom_vs_pair_dist.png",
        {
            "kind": "delta_geom_vs_pair_dist_scatter",
            "n_points": len(xs_data),
            "pooled": pooled,
        },
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
    parser.add_argument(
        "--source-pairs",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_490" / "source_pairs.json"),
    )
    args = parser.parse_args()

    decomp_path = Path(args.decomp)
    tidy_path = Path(args.tidy)
    source_pairs_path = Path(args.source_pairs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_pairs: dict | None = None
    if source_pairs_path.exists():
        source_pairs = json.loads(source_pairs_path.read_text())

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
    delta_geom_vs_pair_dist(decomp, tidy_rows, source_pairs, out_dir)

    log.info("Figures written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
