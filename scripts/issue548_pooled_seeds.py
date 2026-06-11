#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ※, Δ) in scientific docstrings + log lines.
"""Issue #548 — pooled two-seed companion read (seed-42 + seed-43 features).

Per context pair, averages the two seeds' canonical RB JS (per-token AND
un-normalized) and the two seeds' natural-length features (per-side mean reply
lengths, i.e. the mean over the pooled 16 draws per (context, probe)), then
recomputes the kill-bearing partials + bootstrap CIs on the pooled features
for the three ordinary strips. Statistic conventions (partial_spearman /
partial_pearson_on_ranks, iid + unordered-pair-clustered bootstrap flavors,
deterministic per-statistic RNG, zero-call rule) are imported directly from
scripts/issue548_length_analysis.py so they match the per-seed reads exactly.

The leakage DV (in_R emission per cell) is reused unchanged — pooling averages
the PREDICTOR-side measurement noise (fresh-draw variance in the divergence
and length features) only; cell-level DV noise does not pool.

Also writes the seed-comparison figure: per-strip kill-bearing partial
(figure convention, per-token), seed 42 vs seed 43 vs pooled, with clustered
bootstrap 95% intervals.

CLI:
    uv run python scripts/issue548_pooled_seeds.py \\
        --seed42-dir eval_results/issue_548 \\
        --seed43-dir eval_results/issue_548/seed43-replication \\
        --dv-dir eval_results/issue_532/per_cell/loc_ep1 \\
        --out eval_results/issue_548/pooled_two_seed.json \\
        --figures-dir figures/issue_548/seed43 --seed 42 --n-boot 10000
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue548_length_analysis as la  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

logger = logging.getLogger("issue548.pooled_seeds")

SCHEMA_VERSION = "issue548_pooled_two_seed_v1"


def _metadata(args: argparse.Namespace) -> dict:
    """Standard reproducibility block (CLAUDE.md Code Style)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": la._git_commit(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "args": {
            "seed42_dir": str(args.seed42_dir),
            "seed43_dir": str(args.seed43_dir),
            "dv_dir": str(args.dv_dir),
            "out": str(args.out),
            "figures_dir": str(args.figures_dir),
            "seed": args.seed,
            "n_boot": args.n_boot,
        },
        "pooling": {
            "js_rb": "0.5 × (seed-42 matrix + seed-43 matrix) per cell",
            "js_rb_unnormalized": "0.5 × (seed-42 + seed-43) per unordered pair",
            "length_feature": "abs(pooled mean side a − pooled mean side b); pooled mean "
            "per side = 0.5 × (seed-42 mean + seed-43 mean) — equal draw counts, so this "
            "is the mean over the pooled 16 draws per (context, probe)",
            "dv": "in_R_emission_rate per cell, reused unchanged (does not pool)",
        },
        "conventions_inherited_from": "scripts/issue548_length_analysis.py",
    }


def _pooled_dlen(pl42: dict, pl43: dict, x: str, y: str) -> float:
    """|Δ pooled mean reply length| per cell; diagonal = 0 (parent construction)."""
    if x == y:
        return 0.0
    la42, lb42 = pl42[(x, y)] if (x, y) in pl42 else pl42[(y, x)][::-1]
    la43, lb43 = pl43[(x, y)] if (x, y) in pl43 else pl43[(y, x)][::-1]
    return abs(0.5 * (la42 + la43) - 0.5 * (lb42 + lb43))


def pooled_strip_stats(
    cells: list[tuple[str, str]],
    mask: np.ndarray,
    cols: dict[str, np.ndarray],
    y: np.ndarray,
    seed: int,
    n_boot: int,
) -> dict:
    """Kill-bearing read for one strip on the pooled features.

    Mirrors the kill-bearing core of issue548_length_analysis.strip_stats:
    raw ρ, entanglement, both partial directions in the three variants
    (figure/per-token PRIMARY; analysis/per-token + figure/un-normalized
    companions), each with iid + clustered bootstrap CIs and the combined
    flavor-agreement call.
    """
    m = mask
    ym = y[m]
    xd = cols["length_diff"][m]
    xj = cols["js_rb"][m]
    xu = cols["js_rb_unnormalized"][m]
    clusters = la._cluster_ids([c for c, keep in zip(cells, m, strict=True) if keep])
    res: dict = {"n": int(m.sum()), "n_clusters": len(np.unique(clusters))}
    rr, rp = spearmanr(xj, ym)
    lr, lp = spearmanr(xd, ym)
    res["js_rb_raw"] = {"rho": float(rr), "p": float(rp)}
    res["length_alone"] = {"rho": float(lr), "p": float(lp)}
    res["entanglement_rho_js_length"] = float(spearmanr(xj, xd)[0])
    pf, pfp = la.partial_spearman(xj, ym, xd)
    pa, pap = la.partial_pearson_on_ranks(xj, ym, xd)
    pu, pup = la.partial_spearman(xu, ym, xd)
    res["js_partial_length_points"] = {
        "figure_pertoken": {"rho": pf, "p": pfp},
        "analysis_pertoken": {"rho": pa, "p": pap},
        "figure_unnormalized": {"rho": pu, "p": pup},
    }
    kb: dict = {"js_partial_length": {}, "length_partial_js": {}}
    kb["js_partial_length"]["figure_pertoken"] = la.kill_bearing_ci(
        "pooled/js|len/fig/pt", la.partial_spearman, xj, ym, xd, clusters, seed, n_boot
    )
    kb["js_partial_length"]["analysis_pertoken"] = la.kill_bearing_ci(
        "pooled/js|len/ana/pt", la.partial_pearson_on_ranks, xj, ym, xd, clusters, seed, n_boot
    )
    kb["js_partial_length"]["figure_unnormalized"] = la.kill_bearing_ci(
        "pooled/js|len/fig/unnorm", la.partial_spearman, xu, ym, xd, clusters, seed, n_boot
    )
    kb["length_partial_js"]["figure_pertoken"] = la.kill_bearing_ci(
        "pooled/len|js/fig/pt", la.partial_spearman, xd, ym, xj, clusters, seed, n_boot
    )
    kb["length_partial_js"]["analysis_pertoken"] = la.kill_bearing_ci(
        "pooled/len|js/ana/pt", la.partial_pearson_on_ranks, xd, ym, xj, clusters, seed, n_boot
    )
    kb["length_partial_js"]["figure_unnormalized"] = la.kill_bearing_ci(
        "pooled/len|js/fig/unnorm", la.partial_spearman, xd, ym, xu, clusters, seed, n_boot
    )
    for direction in kb:
        kb[direction]["combined_call"] = la._combined_call(
            {v: kb[direction][v]["call"] for v in kb[direction] if v != "combined_call"}
        )
    res["kill_bearing_ci"] = kb
    return res


def _per_seed_fig_points(seed_dir: Path) -> dict:
    """Per-strip figure-convention kill-bearing point + clustered CI of one seed's read."""
    rec = json.loads((seed_dir / "length_analysis.json").read_text())
    out = {}
    for strip, s in rec["strips"].items():
        kb = s["kill_bearing_ci"]["js_partial_length"]["figure_pertoken"]["clustered"]
        out[strip] = {"point": kb["point"], "ci95": kb["ci95"]}
    return out


def make_figure(
    figures_dir: Path,
    pooled: dict,
    seed42: dict,
    seed43: dict,
) -> list[str]:
    """Grouped bars: per-strip kill-bearing partial, seed 42 / seed 43 / pooled."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    strips = ["ordinary_full", "ordinary_offdiag", "ordinary_no_d5_offdiag"]
    strip_labels = [
        "Full strip\n(n = 256)",
        "Off-diagonal strip\n(n = 240)",
        "Off-diagonal, no\nenumerated rewrite\n(n = 210)",
    ]
    series = [
        ("Seed 42", seed42, paper_palette_role("baseline")),
        ("Seed 43", seed43, paper_palette_role("primary")),
        ("Pooled (both seeds)", None, paper_palette_role("accent")),
    ]
    fig, ax = plt.subplots()
    x = np.arange(len(strips))
    width = 0.26
    for k, (label, src, color) in enumerate(series):
        pts, los, his = [], [], []
        for strip in strips:
            if src is None:
                kb = pooled[strip]["kill_bearing_ci"]["js_partial_length"]["figure_pertoken"][
                    "clustered"
                ]
                pt, ci = kb["point"], kb["ci95"]
            else:
                pt, ci = src[strip]["point"], src[strip]["ci95"]
            pts.append(pt)
            los.append(pt - ci[0])
            his.append(ci[1] - pt)
        ax.bar(
            x + (k - 1) * width,
            pts,
            width,
            yerr=[los, his],
            capsize=3,
            label=label,
            color=color,
        )
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(strip_labels)
    ax.set_ylabel("Length-controlled partial rank correlation\n(divergence vs marker emission)")
    ax.legend(loc="lower right")
    set_title_subtitle(
        ax,
        "The replication moves the partials by less than 0.01",
        "Pair-clustered bootstrap 95% intervals; pooled = per-pair feature average",
    )
    paths = savefig_paper(fig, "issue_548/seed43/seed_pooled_partials", dir="figures/")
    plt.close(fig)
    return [str(p) for p in paths.values()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed42-dir", type=Path, default=Path("eval_results/issue_548"))
    p.add_argument(
        "--seed43-dir", type=Path, default=Path("eval_results/issue_548/seed43-replication")
    )
    p.add_argument("--dv-dir", type=Path, default=Path("eval_results/issue_532/per_cell/loc_ep1"))
    p.add_argument("--out", type=Path, default=Path("eval_results/issue_548/pooled_two_seed.json"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures/issue_548/seed43"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--skip-figures", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    p42 = la.load_predictors(args.seed42_dir)
    p43 = la.load_predictors(args.seed43_dir)
    if p42["sources"] != p43["sources"] or p42["bystanders"] != p43["bystanders"]:
        raise ValueError("seed-42/seed-43 predictor panels disagree — not the same panel")
    srcs, bys = p42["sources"], p42["bystanders"]
    emis = la.load_emission(args.dv_dir)
    pairs42 = la.load_pairs(args.seed42_dir)
    pairs43 = la.load_pairs(args.seed43_dir)
    if pairs42["cap"] != pairs43["cap"]:
        raise ValueError(
            f"recorded caps disagree across seeds: {pairs42['cap']} vs {pairs43['cap']}"
        )

    js_pooled_mat = 0.5 * (p42["mats"]["js_rb"] + p43["mats"]["js_rb"])
    s_idx = {s: i for i, s in enumerate(srcs)}
    b_idx = {b: j for j, b in enumerate(bys)}
    ordinary = [(s, c) for s in srcs for c in srcs]
    y_ord = np.array([emis[c] for c in ordinary])
    cols = {
        "length_diff": np.array(
            [_pooled_dlen(pairs42["pairlen"], pairs43["pairlen"], *c) for c in ordinary]
        ),
        "js_rb": np.array([js_pooled_mat[s_idx[a], b_idx[b]] for a, b in ordinary]),
        "js_rb_unnormalized": np.array(
            [
                0.5
                * (
                    la._pairget(pairs42["js_unnorm"], a, b)
                    + la._pairget(pairs43["js_unnorm"], a, b)
                )
                for a, b in ordinary
            ]
        ),
    }
    is_diag = np.array([a == b for a, b in ordinary])
    no_d5 = np.array([(not d) and ("D5" not in c) for c, d in zip(ordinary, is_diag, strict=True)])
    masks = {
        "ordinary_full": np.ones(len(ordinary), dtype=bool),
        "ordinary_offdiag": ~is_diag,
        "ordinary_no_d5_offdiag": no_d5,
    }

    strips = {}
    for name, m in masks.items():
        logger.info("pooled strip %s (n=%d): computing kill-bearing read ...", name, int(m.sum()))
        strips[name] = pooled_strip_stats(ordinary, m, cols, y_ord, args.seed, args.n_boot)

    # Per-seed agreement diagnostics (predictor-side stability across fresh draws).
    js42 = np.array([p42["mats"]["js_rb"][s_idx[a], b_idx[b]] for a, b in ordinary])
    js43 = np.array([p43["mats"]["js_rb"][s_idx[a], b_idx[b]] for a, b in ordinary])
    d42 = np.array([la._dlen(pairs42["pairlen"], *c) for c in ordinary])
    d43 = np.array([la._dlen(pairs43["pairlen"], *c) for c in ordinary])
    off = ~is_diag
    cross_seed = {
        "js_rb_spearman_offdiag_cells": float(spearmanr(js42[off], js43[off])[0]),
        "length_diff_spearman_offdiag_cells": float(spearmanr(d42[off], d43[off])[0]),
    }

    out = {
        "metadata": _metadata(args),
        "strips": strips,
        "cross_seed_feature_agreement": cross_seed,
        "per_seed_figure_points": {
            "seed42": _per_seed_fig_points(args.seed42_dir),
            "seed43": _per_seed_fig_points(args.seed43_dir),
        },
    }

    figures_written: list[str] = []
    if not args.skip_figures:
        figures_written = make_figure(
            args.figures_dir,
            strips,
            out["per_seed_figure_points"]["seed42"],
            out["per_seed_figure_points"]["seed43"],
        )
    out["figures_written"] = figures_written

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    summary = {
        name: {
            "pooled_point_fig_pertoken": strips[name]["kill_bearing_ci"]["js_partial_length"][
                "figure_pertoken"
            ]["clustered"]["point"],
            "pooled_clustered_ci": strips[name]["kill_bearing_ci"]["js_partial_length"][
                "figure_pertoken"
            ]["clustered"]["ci95"],
            "combined_call": strips[name]["kill_bearing_ci"]["js_partial_length"]["combined_call"],
        }
        for name in strips
    }
    logger.info("wrote %s", args.out)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
