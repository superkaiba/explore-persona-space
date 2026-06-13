#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ※, Δ) in scientific docstrings + log lines.
"""Issue #548 — incremental-validity stack (free-analysis follow-up, 9a-ter).

Does the revived divergence component survive controlling for activation
geometry? Per ordinary strip (full n=256 / off-diagonal n=240 / no-enumerated-
rewrite n=210) this computes, on the committed predictors only:

1. The joint partial rank correlation ρ(js_rb@1024, emission | length_diff,
   gauss_kl) — the headline incremental-validity read.
2. Symmetric context reads: ρ(gauss_kl, emission | length_diff, js_rb) (does
   geometry survive controlling for divergence?) and the simple two-covariate
   baseline ρ(js_rb, emission | gauss_kl) without length.
3. The single-covariate ρ(js_rb, emission | length_diff) as the reproduction
   control — its point estimates must match the published full-strip values in
   eval_results/issue_548/length_analysis.json exactly (same inputs, same
   verbatim convention code), and its bootstrap CIs are bit-identical because
   the RNG labels are reused from the parent read.

Every partial is reported in BOTH conventions (figure: Spearman of OLS
rank-residuals re-ranked; analysis: Pearson of OLS rank-residuals — the
single-covariate implementations are imported VERBATIM from
scripts/issue548_length_analysis.py; the multi-covariate OLS on ranks
generalizes them naturally and is asserted to reduce bit-exactly at k=1) and
in BOTH normalization variants (per-token js_rb PRIMARY; un-normalized js_rb
companion — the un-normalized feature substitutes for js_rb wherever it
appears, as the x variable or as a covariate). Each variant carries 10k-rep
bootstrap CIs in BOTH flavors (iid-cell AND 136-pair-clustered on the full
strip), seed 42, via the parent's kill_bearing_ci machinery.

Inputs (all committed on the issue-548 branch):
- js_rb per cell: eval_results/issue_548/predictors_jsrb.json
- gauss_kl: eval_results/issue_532/predictors.json (cross-checked allclose
  against the copy embedded in predictors_jsrb.json)
- length_diff: per-pair n_positions, same construction as
  issue548_length_analysis.py (its load_pairs/_dlen are reused directly)
- DV: in_R_emission_rate from eval_results/issue_532/per_cell/loc_ep1/

CLI:
    uv run python scripts/issue548_incremental_validity.py \\
        --new-dir eval_results/issue_548 \\
        --geometry-predictors eval_results/issue_532/predictors.json \\
        --dv-dir eval_results/issue_532/per_cell/loc_ep1 \\
        --out eval_results/issue_548/incremental_validity.json \\
        --figures-dir figures/issue_548 --seed 42 --n-boot 10000
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
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402

logger = logging.getLogger("issue548.incremental_validity")

SCHEMA_VERSION = "issue548_incremental_validity_v1"
REPRO_TOL = 1e-9  # same inputs + verbatim convention code → exact to float noise


# ── Multi-covariate generalizations of the two verbatim conventions ──────────


def _rank_design(Z: np.ndarray) -> np.ndarray:
    """Rank each covariate column of Z (n, k) and append the intercept column."""
    assert Z.ndim == 2, Z.shape
    ranked = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])
    return np.column_stack([ranked, np.ones(Z.shape[0])])


def partial_spearman_multi(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    """Figure convention, k covariates: Spearman of OLS rank-residuals.

    With Z of shape (n, 1) this reduces bit-exactly to
    issue548_length_analysis.partial_spearman (asserted at startup).
    """
    rx, ry = rankdata(x), rankdata(y)
    design = _rank_design(Z)

    def resid(a: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    rho, p = spearmanr(resid(rx), resid(ry))
    return float(rho), float(p)


def partial_pearson_on_ranks_multi(
    x: np.ndarray, y: np.ndarray, Z: np.ndarray
) -> tuple[float, float]:
    """Analysis-JSON convention, k covariates: Pearson of OLS rank-residuals."""
    rx, ry = rankdata(x), rankdata(y)
    design = _rank_design(Z)

    def resid(a: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    r, p = pearsonr(resid(rx), resid(ry))
    return float(r), float(p)


def _assert_k1_reduction(rng: np.random.Generator) -> None:
    """Assert the multi-covariate fns reduce to the parent's verbatim k=1 fns."""
    x = rng.normal(size=64)
    y = rng.normal(size=64)
    z = rng.normal(size=64)
    for multi_fn, single_fn, name in [
        (partial_spearman_multi, la.partial_spearman, "figure"),
        (partial_pearson_on_ranks_multi, la.partial_pearson_on_ranks, "analysis"),
    ]:
        got = multi_fn(x, y, z.reshape(-1, 1))[0]
        want = single_fn(x, y, z)[0]
        assert abs(got - want) < 1e-12, (
            f"{name} multi-covariate convention does not reduce to the verbatim "
            f"single-covariate implementation at k=1: {got} vs {want}"
        )


# ── Reproducibility metadata ─────────────────────────────────────────────────


def _metadata(args: argparse.Namespace) -> dict:
    """Standard reproducibility block (CLAUDE.md Code Style)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": la._git_commit(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "args": {
            "new_dir": str(args.new_dir),
            "geometry_predictors": str(args.geometry_predictors),
            "dv_dir": str(args.dv_dir),
            "out": str(args.out),
            "figures_dir": str(args.figures_dir),
            "seed": args.seed,
            "n_boot": args.n_boot,
        },
        "conventions": {
            "partial_figure": "Spearman of OLS rank-residuals (re-ranks residuals); PRIMARY; "
            "multi-covariate design = [rank(z1), ..., rank(zk), 1], asserted to reduce "
            "bit-exactly to the verbatim single-covariate implementation at k=1",
            "partial_analysis_json": "Pearson of OLS rank-residuals; companion",
            "normalization_primary": "per-token js_rb (kl_side_m_bits_per_token mean)",
            "normalization_companion": "un-normalized js_rb (total bits per reply); the "
            "un-normalized feature substitutes for js_rb wherever it appears (x or covariate)",
            "ci_rule": "BOTH iid-cell and unordered-pair-clustered bootstrap CIs per partial "
            "(kill-bearing CI rule inherited from issue548_length_analysis.py); clustered "
            "quoted primary; zero-call requires both flavors to agree",
            "length_feature": "abs(mean n_positions side a − side b) per pair, incl. the "
            "appended terminator (same construction as issue548_length_analysis.py)",
            "geometry_feature": "activation Gaussian KL per cell from "
            "eval_results/issue_532/predictors.json (allclose-checked against the copy in "
            "predictors_jsrb.json)",
            "rng_labels": "the js_rb|length read reuses the parent's RNG labels "
            "(js|len/{fig,ana}/pt, js|len/fig/unnorm) so its CIs are bit-identical to "
            "length_analysis.json; new reads use iv/-prefixed labels",
            "dv": "in_R_emission_rate per cell, reused byte-for-byte",
        },
    }


# ── Strip computation ────────────────────────────────────────────────────────

# (read name, x feature, covariate features, RNG label stem). Labels for the
# js_rb|length read match issue548_length_analysis.strip_stats verbatim so the
# reproduction control's CIs are bit-identical, not just its point estimates.
READS = [
    ("js_rb_given_length", "js", ("length",), {"pertoken": "js|len", "unnorm": "js|len"}),
    ("js_rb_given_gauss_kl", "js", ("gauss",), {"pertoken": "iv/js|g", "unnorm": "iv/js|g"}),
    (
        "js_rb_given_length_and_gauss_kl",
        "js",
        ("length", "gauss"),
        {"pertoken": "iv/js|len+g", "unnorm": "iv/js|len+g"},
    ),
    (
        "gauss_kl_given_length_and_js_rb",
        "gauss",
        ("length", "js"),
        {"pertoken": "iv/g|len+js", "unnorm": "iv/g|len+js"},
    ),
]


def _features(cols: dict[str, np.ndarray], m: np.ndarray, normalization: str) -> dict:
    """The three masked feature vectors for one normalization variant."""
    js_key = "js_rb" if normalization == "pertoken" else "js_rb_unnormalized"
    return {"js": cols[js_key][m], "gauss": cols["gauss_kl"][m], "length": cols["length_diff"][m]}


def _partial_fn_for(convention: str):
    """Single- vs multi-covariate dispatch happens via the Z shape, not here."""
    if convention == "figure":
        return lambda x, y, z: (
            la.partial_spearman(x, y, z) if z.ndim == 1 else partial_spearman_multi(x, y, z)
        )
    return lambda x, y, z: (
        la.partial_pearson_on_ranks(x, y, z)
        if z.ndim == 1
        else partial_pearson_on_ranks_multi(x, y, z)
    )


def strip_stats(
    cells: list[tuple[str, str]],
    mask: np.ndarray,
    cols: dict[str, np.ndarray],
    y: np.ndarray,
    seed: int,
    n_boot: int,
) -> dict:
    """All four partial reads × four variants for one strip, with both CI flavors."""
    m = mask
    ym = y[m]
    clusters = la._cluster_ids([c for c, keep in zip(cells, m, strict=True) if keep])
    res: dict = {"n": int(m.sum()), "n_clusters": len(np.unique(clusters))}

    ft_pt = _features(cols, m, "pertoken")
    res["raw"] = {
        "js_rb": dict(zip(("rho", "p"), map(float, spearmanr(ft_pt["js"], ym)), strict=True)),
        "gauss_kl": dict(zip(("rho", "p"), map(float, spearmanr(ft_pt["gauss"], ym)), strict=True)),
        "length_alone": dict(
            zip(("rho", "p"), map(float, spearmanr(ft_pt["length"], ym)), strict=True)
        ),
    }
    res["collinearity"] = {
        "rho_js_length": float(spearmanr(ft_pt["js"], ft_pt["length"])[0]),
        "rho_gauss_length": float(spearmanr(ft_pt["gauss"], ft_pt["length"])[0]),
        "rho_js_gauss": float(spearmanr(ft_pt["js"], ft_pt["gauss"])[0]),
    }

    res["partials"] = {}
    for read_name, x_key, z_keys, stems in READS:
        variants: dict = {}
        for normalization in ["pertoken", "unnorm"]:
            ft = _features(cols, m, normalization)
            x = ft[x_key]
            Z = ft[z_keys[0]] if len(z_keys) == 1 else np.column_stack([ft[k] for k in z_keys])
            for convention, conv_tag in [("figure", "fig"), ("analysis", "ana")]:
                # Parent's label scheme: js|len/fig/pt etc.; unnorm only had a
                # figure-convention row there, so only that label is inherited.
                norm_tag = "pt" if normalization == "pertoken" else "unnorm"
                label = f"{stems[normalization]}/{conv_tag}/{norm_tag}"
                variants[f"{convention}_{normalization}"] = la.kill_bearing_ci(
                    label, _partial_fn_for(convention), x, ym, Z, clusters, seed, n_boot
                )
        variants["combined_call"] = la._combined_call(
            {v: variants[v]["call"] for v in variants if v != "combined_call"}
        )
        res["partials"][read_name] = variants
    return res


# ── Reproduction control ─────────────────────────────────────────────────────


def reproduction_control(strips: dict, published_path: Path) -> dict:
    """Hard gate: js_rb|length must reproduce the published per-strip partials.

    Same inputs + verbatim convention code → identical point estimates to
    float precision. A mismatch means an input or convention drifted; the
    script fails loud rather than shipping a silently different stack.
    """
    published = json.loads(published_path.read_text())
    checks = []
    for strip in ["ordinary_full", "ordinary_offdiag", "ordinary_no_d5_offdiag"]:
        pub = published["strips"][strip]["js_rb"]
        for convention, pub_key in [
            ("figure", "partial_length_figure_convention"),
            ("analysis", "partial_length_analysis_convention"),
        ]:
            want = pub[pub_key]["rho"]
            got = strips[strip]["partials"]["js_rb_given_length"][f"{convention}_pertoken"][
                "clustered"
            ]["point"]
            ok = abs(got - want) < REPRO_TOL
            checks.append(
                {
                    "strip": strip,
                    "convention": convention,
                    "published_rho": float(want),
                    "recomputed_rho": float(got),
                    "match": bool(ok),
                }
            )
            if not ok:
                raise AssertionError(
                    f"reproduction control FAILED: js_rb|length ({convention}, {strip}) "
                    f"recomputed {got} vs published {want} in {published_path}"
                )
    logger.info(
        "reproduction control PASSED: js_rb|length matches %s on all %d strip×convention reads",
        published_path,
        len(checks),
    )
    return {"published_source": str(published_path), "tolerance": REPRO_TOL, "checks": checks}


# ── Figure ───────────────────────────────────────────────────────────────────

FIG_READS = [
    ("js_rb_given_length", "Divergence | length"),
    ("js_rb_given_gauss_kl", "Divergence | geometry"),
    ("js_rb_given_length_and_gauss_kl", "Divergence | length + geometry"),
    ("gauss_kl_given_length_and_js_rb", "Geometry | length + divergence"),
]
FIG_STRIPS = [
    ("ordinary_full", "Full strip\n(n = 256)"),
    ("ordinary_offdiag", "Off-diagonal strip\n(n = 240)"),
    ("ordinary_no_d5_offdiag", "Off-diagonal, no\nenumerated rewrite\n(n = 210)"),
]


def make_figure(figures_dir: Path, strips: dict) -> list[str]:
    """Grouped bars: four length/geometry-controlled partials per strip.

    Bars = figure-convention per-token points; error bars = pair-clustered
    bootstrap 95% intervals (the primary CI flavor).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = paper_palette(len(FIG_READS))

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(FIG_STRIPS))
    width = 0.19
    for k, ((read_name, read_label), color) in enumerate(zip(FIG_READS, colors, strict=True)):
        pts, los, his = [], [], []
        for strip_name, _ in FIG_STRIPS:
            kb = strips[strip_name]["partials"][read_name]["figure_pertoken"]["clustered"]
            pt, ci = kb["point"], kb["ci95"]
            pts.append(pt)
            los.append(pt - ci[0])
            his.append(ci[1] - pt)
        ax.bar(
            x + (k - 1.5) * width,
            pts,
            width,
            yerr=[los, his],
            capsize=3,
            label=read_label,
            color=color,
        )
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in FIG_STRIPS])
    ax.set_ylabel("Partial rank correlation with marker emission")
    ax.legend(loc="lower right", fontsize=8)
    set_title_subtitle(
        ax,
        "Does response divergence survive controlling for activation geometry?",
        "Pair-clustered bootstrap 95% intervals; per-token divergence, figure convention",
    )
    paths = savefig_paper(fig, "incremental_validity_partials", dir=figures_dir)
    plt.close(fig)
    return [str(p) for p in paths.values()]


# ── Main ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[4],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--new-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_548")
    p.add_argument(
        "--geometry-predictors",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_532/predictors.json",
    )
    p.add_argument(
        "--dv-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_532/per_cell/loc_ep1"
    )
    p.add_argument(
        "--published-length-analysis",
        type=Path,
        default=None,
        help="length_analysis.json for the reproduction control (default: <new-dir>/...)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_548/incremental_validity.json",
    )
    p.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures/issue_548")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument(
        "--skip-figures", action="store_true", help="JSON only (figures need matplotlib)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)
    published_path = args.published_length_analysis or (args.new_dir / "length_analysis.json")

    _assert_k1_reduction(np.random.default_rng(0))

    pred = la.load_predictors(args.new_dir)
    srcs, bys = pred["sources"], pred["bystanders"]
    geom = json.loads(args.geometry_predictors.read_text())
    if geom["sources"] != srcs or geom["bystanders"] != bys:
        raise ValueError("geometry predictor panel disagrees with the js_rb panel")
    gauss_kl = np.array(geom["gauss_kl_matrix"])
    if not np.allclose(gauss_kl, pred["mats"]["gauss_kl"]):
        raise ValueError(
            f"{args.geometry_predictors} gauss_kl_matrix disagrees with the copy embedded in "
            f"{args.new_dir / 'predictors_jsrb.json'} — predictor provenance drifted"
        )
    emis = la.load_emission(args.dv_dir)
    pairs = la.load_pairs(args.new_dir)
    logger.info(
        "loaded panel: %d sources × %d bystanders, %d pairs, recorded cap=%d",
        len(srcs),
        len(bys),
        len(pairs["pairtrunc"]),
        pairs["cap"],
    )

    ordinary = [(s, c) for s in srcs for c in srcs]
    s_idx = {s: i for i, s in enumerate(srcs)}
    b_idx = {b: j for j, b in enumerate(bys)}
    y_ord = np.array([emis[c] for c in ordinary])
    cols = {
        "length_diff": np.array([la._dlen(pairs["pairlen"], *c) for c in ordinary]),
        "js_rb": np.array([pred["mats"]["js_rb"][s_idx[a], b_idx[b]] for a, b in ordinary]),
        "js_rb_unnormalized": np.array([la._pairget(pairs["js_unnorm"], *c) for c in ordinary]),
        "gauss_kl": np.array([gauss_kl[s_idx[a], b_idx[b]] for a, b in ordinary]),
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
        logger.info("strip %s (n=%d): computing incremental-validity stack ...", name, int(m.sum()))
        strips[name] = strip_stats(ordinary, m, cols, y_ord, args.seed, args.n_boot)

    repro = reproduction_control(strips, published_path)

    out = {
        "metadata": _metadata(args),
        "reproduction_control": repro,
        "strips": strips,
    }

    figures_written: list[str] = []
    if not args.skip_figures:
        figures_written = make_figure(args.figures_dir, strips)
    out["figures_written"] = figures_written

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    summary = {
        strip: {
            read: {
                conv: {
                    "point": strips[strip]["partials"][read][f"{conv}_pertoken"]["clustered"][
                        "point"
                    ],
                    "clustered_ci95": strips[strip]["partials"][read][f"{conv}_pertoken"][
                        "clustered"
                    ]["ci95"],
                    "call": strips[strip]["partials"][read][f"{conv}_pertoken"]["call"],
                }
                for conv in ["figure", "analysis"]
            }
            for read in ["js_rb_given_length_and_gauss_kl", "gauss_kl_given_length_and_js_rb"]
        }
        for strip in strips
    }
    logger.info("wrote %s", args.out)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
