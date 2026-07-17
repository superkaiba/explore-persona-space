"""Issue #1072 phase E — figures from stats_component.json + the fold npzs.

Outputs (figures/issue_1072/):
  hero_component_decomposition.png — per-layer stacked ΔC_par/ΔC_perp/ΔC_cross
      of the own-ext_plain c-leg gap with the D CI whiskers (plan §6 hero).
  closure_by_component_L26.png     — per-component G0 vs Gt at the primary layer.
  percontext_par_vs_perp_L26.png   — LOW-LEVEL per-unit plot: per-context paired
      contribution differences (par vs perp) at the primary layer, scatter +
      marginal histograms over the matched contexts.
  exploratory_component_profiles.png — S_par depth profile, w_par by arm/layer,
      per-slot ΔC_par profiles (f16/l16/d10), optional cos(raw,folded) hist.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style  # noqa: E402
from scripts.issue1072_stats import CH, K_FOLDS, _safe_ratio  # noqa: E402

logger = logging.getLogger("issue1072.figures")

COMP_ORDER = ("par", "perp", "cross")


def _err(lo: float, v: float, hi: float) -> tuple[float, float]:
    """Non-negative errorbar OFFSETS (never CI bounds/signed deltas — gotchas.md)."""
    return max(0.0, v - lo), max(0.0, hi - v)


def _load(eval_dir: pathlib.Path) -> tuple[dict, dict[int, dict]]:
    stats = json.loads((eval_dir / "stats_component.json").read_text())
    npzs = {
        k: dict(np.load(eval_dir / f"per_context_stats_1072_fold{k}.npz", allow_pickle=False))
        for k in range(K_FOLDS)
    }
    return stats, npzs


def fig_hero(stats: dict, out_dir: pathlib.Path) -> None:
    by_layer = stats["by_ext"]["ext_plain"]
    layers = sorted(int(x) for x in by_layer)
    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    x = np.arange(len(layers), dtype=float)
    bottom_pos = np.zeros(len(layers))
    bottom_neg = np.zeros(len(layers))
    for ci, comp in enumerate(COMP_ORDER):
        vals = np.array([by_layer[str(la)]["delta_C"][comp] for la in layers])
        bottoms = np.where(vals >= 0, bottom_pos, bottom_neg)
        ax.bar(x, vals, 0.55, bottom=bottoms, color=colors[ci], label=f"ΔC_{comp}")
        bottom_pos += np.where(vals >= 0, vals, 0.0)
        bottom_neg += np.where(vals < 0, vals, 0.0)
    d_vals = [by_layer[str(la)]["D"] for la in layers]
    d_err = np.array(
        [
            _err(by_layer[str(la)]["D_ci95"][0], d, by_layer[str(la)]["D_ci95"][1])
            for la, d in zip(layers, d_vals, strict=True)
        ]
    ).T
    ax.errorbar(
        x + 0.32,
        d_vals,
        yerr=d_err,
        fmt="D",
        color="black",
        ms=4,
        capsize=3,
        label="D = ΔC_par - ΔC_perp (95% CI)",
    )
    ax.axhline(0.0, lw=0.8, color="gray")
    ax.set_xticks(x, [f"L{la}" for la in layers])
    ax.set_ylabel("own - ext_plain gap contribution (c-leg, remainder target)")
    ax.set_title("Component decomposition of the own-answer advantage by layer")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "hero_component_decomposition.png", dpi=200)
    plt.close(fig)


def fig_closure(stats: dict, out_dir: pathlib.Path) -> None:
    primary = str(stats["lattice"]["primary_layer"])
    rec = stats["by_ext"]["ext_plain"][primary]["closure_by_component"]
    comps = [c for c in ("par", "perp", "cross", "full") if c in rec]
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(5.6, 3.8), layout="constrained")
    x = np.arange(len(comps), dtype=float)
    ax.bar(x - 0.18, [rec[c]["G0"] for c in comps], 0.34, color=colors[0], label="G0 (no prefix)")
    ax.bar(
        x + 0.18,
        [rec[c]["Gt"] for c in comps],
        0.34,
        color=colors[1],
        label=f"G(t={16}) (16-token prefix)",
    )
    ax.axhline(0.0, lw=0.8, color="gray")
    ax.set_xticks(x, comps)
    ax.set_ylabel("own - ext_plain gap")
    ax.set_title(f"Per-component prefix closure at L{primary}")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "closure_by_component_L26.png", dpi=200)
    plt.close(fig)


def _percontext_deltas(npzs: dict[int, dict], layer: int, comp: str) -> np.ndarray:
    """Per-context ΔC contribution (own - ext_plain) at one layer/component."""
    out = []
    for k in sorted(npzs):
        vals = {}
        for arm in ("own", "ext_plain"):
            ch = npzs[k][f"M16c_L{layer}|{arm}"].astype(np.float64)
            num = (
                ch[:, CH[f"ss_tot_{comp}"]] - ch[:, CH[f"ss_res_{comp}"]]
                if comp in ("par", "perp")
                else ch[:, CH["cross_tot"]] - ch[:, CH["cross_res"]]
            )
            vals[arm] = _safe_ratio(num, ch[:, CH["ss_tot_full"]])
        out.append(vals["own"] - vals["ext_plain"])
    return np.concatenate(out)


def fig_percontext(stats: dict, npzs: dict[int, dict], out_dir: pathlib.Path) -> None:
    layer = int(stats["lattice"]["primary_layer"])
    d_par = _percontext_deltas(npzs, layer, "par")
    d_perp = _percontext_deltas(npzs, layer, "perp")
    m = np.isfinite(d_par) & np.isfinite(d_perp)
    fig = plt.figure(figsize=(6.0, 6.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4))
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.scatter(d_par[m], d_perp[m], s=4, alpha=0.25, color=paper_palette(1)[0])
    ax.axhline(0, lw=0.6, color="gray")
    ax.axvline(0, lw=0.6, color="gray")
    ax.set_xlabel(f"per-context Δc_par (own - ext_plain), L{layer}")
    ax.set_ylabel(f"per-context Δc_perp (own - ext_plain), L{layer}")
    axx.hist(d_par[m], bins=60, color=paper_palette(1)[0])
    axy.hist(d_perp[m], bins=60, orientation="horizontal", color=paper_palette(1)[0])
    axx.tick_params(labelbottom=False)
    axy.tick_params(labelleft=False)
    fig.suptitle(f"Per-context paired contribution differences (n={int(m.sum())} matched contexts)")
    fig.savefig(out_dir / "percontext_par_vs_perp_L26.png", dpi=200)
    plt.close(fig)


def fig_exploratory(
    stats: dict,
    fold_recs: dict[int, dict],
    out_dir: pathlib.Path,
    raw_cos_npz: pathlib.Path | None,
) -> None:
    by_layer = stats["by_ext"]["ext_plain"]
    layers = sorted(int(x) for x in by_layer)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), layout="constrained")
    # (a) S_par depth profile (H3, descriptive).
    ax = axes[0][0]
    s = [by_layer[str(la)]["S_par"] for la in layers]
    ci = [by_layer[str(la)]["S_par_ci95"] for la in layers]
    err = np.array(
        [_err(c[0], v, c[1]) if v is not None else (0, 0) for v, c in zip(s, ci, strict=True)]
    ).T
    ax.errorbar(
        layers,
        [v if v is not None else np.nan for v in s],
        yerr=err,
        fmt="o-",
        capsize=3,
        color=paper_palette(1)[0],
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("S_par = ΔC_par / ΔR²_full")
    ax.set_title("Parallel share of the gap by depth (H3)")
    # (b) w_par variance share by arm/layer (calibration fold pooled test).
    ax = axes[0][1]
    cal = fold_recs[K_FOLDS - 1]["layers"]
    for arm in ("own", "ext_plain", "ext_style", "mismatch"):
        vals = [
            cal[str(la)]["components"]["cleg_rem"][f"rem|{arm}"]["w_par"]
            for la in layers
            if str(la) in cal and not cal[str(la)].get("skipped")
        ]
        ax.plot(layers[: len(vals)], vals, "o-", label=arm, ms=3)
    ax.set_xlabel("layer")
    ax.set_ylabel("w_par = ΣSS_tot_par / ΣSS_tot_full")
    ax.set_yscale("log")
    ax.set_title("Token-identity variance share (remainder target)")
    ax.legend(fontsize=7)
    # (c) per-slot ΔC_par profile at the primary layer (f16 positions).
    ax = axes[1][0]
    primary = str(stats["lattice"]["primary_layer"])
    h1 = cal[primary]["components"]["h1"]
    for comp, style in (("C_par", "o-"), ("C_perp", "s--")):
        prof = []
        for t in range(1, 17):
            own = h1.get(f"f16_t{t}|own")
            ext = h1.get(f"f16_t{t}|ext_plain")
            prof.append((own[comp] - ext[comp]) if own and ext else np.nan)
        ax.plot(range(1, 17), prof, style, ms=3, label=f"Δ{comp} (own-ext_plain)")
    ax.axhline(0, lw=0.6, color="gray")
    ax.set_xlabel("answer position t (f16 slots)")
    ax.set_ylabel(f"per-slot gap contribution, L{primary} (cal fold)")
    ax.set_title("Per-slot component profile")
    ax.legend(fontsize=7)
    # (d) cos(raw W_U[y], gamma⊙W_U[y]) over realized next tokens (basis sanity).
    ax = axes[1][1]
    if raw_cos_npz is not None and raw_cos_npz.exists():
        cos = np.load(raw_cos_npz)["cos"]
        # Degenerate-range guard: an all-identical cos set (e.g. gamma == 1)
        # cannot support 80 finite bins — widen the range explicitly.
        lo, hi = float(cos.min()), float(cos.max())
        if hi - lo < 1e-6:
            lo, hi = lo - 1e-3, hi + 1e-3
        ax.hist(cos, bins=80, range=(lo, hi), color=paper_palette(1)[0])
        ax.set_xlabel("cos(W_U[y], gamma⊙W_U[y])")
        ax.set_title(f"gamma-folding sensitivity (n={len(cos)} unique next tokens)")
    else:
        ax.text(
            0.5,
            0.5,
            "raw_vs_folded_cos.npz not staged\n(pass --raw-cos-npz)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_axis_off()
        logger.warning("raw-vs-folded cos histogram SKIPPED — npz not provided/found")
    fig.savefig(out_dir / "exploratory_component_profiles.png", dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Issue #1072 figures (phase E)")
    p.add_argument("--eval-dir", type=str, default=str(_REPO_ROOT / "eval_results" / "issue_1072"))
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO_ROOT / "figures" / "issue_1072"),
        help="figure output dir (smoke runs pass a scratch dir — never the committed path)",
    )
    p.add_argument(
        "--raw-cos-npz",
        type=str,
        default=None,
        help="optional path to raw_vs_folded_cos.npz (HF analysis_tensors)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    set_paper_style()
    eval_dir = pathlib.Path(args.eval_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, npzs = _load(eval_dir)
    fold_recs = {
        k: json.loads((eval_dir / f"battery_1072_fold{k}.json").read_text()) for k in range(K_FOLDS)
    }
    fig_hero(stats, out_dir)
    fig_closure(stats, out_dir)
    fig_percontext(stats, npzs, out_dir)
    fig_exploratory(
        stats,
        fold_recs,
        out_dir,
        pathlib.Path(args.raw_cos_npz) if args.raw_cos_npz else None,
    )
    logger.info("[figures] 4 figures written to %s", out_dir)


if __name__ == "__main__":
    main()
