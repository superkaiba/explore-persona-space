#!/usr/bin/env python3
"""Issue #1073 P5: figures from the P4 result JSONs (VM CPU; no tensors).

Hero (two-panel): held-out R2 vs layer per decode arm (+ mean-target floor and
shuffled-pairing null; cx_mean input faint) | monitoring within-condition r per
trait x mode, per arm, with bootstrap CIs + the raw-projection dashed
reference. Per-unit companion: per-context cosine distributions (greedy vs
avg10, single draws vs avg10, jackknife draw band) + the <v_arm, r_B>
greedy-vs-avg10 identity scatter with labeled extremes. Exploratory dump:
DV2 layer curves, response-length histograms, val-selected-lambda deltas,
per-dimension R2 spectra, the P0 probe-consistency scatter.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy: shared-VM thread caps bind at import (#847)

import issue1073_common as I  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue1073_figures")

ARM_COLORS = {
    "avg10": "#0173b2",
    "greedy": "#de8f05",
    "stoch1_new": "#029e73",
    "stoch1_old": "#949494",
    "mean_floor": "#cc78bc",
    "shuffle_null": "#ca9161",
}
ARM_LABELS = {
    "avg10": "10-rollout average",
    "greedy": "single deterministic (greedy)",
    "stoch1_new": "fresh single random draw",
    "stoch1_old": "single random (parent, reused)",
    "mean_floor": "mean-target floor (degenerate prefix arm)",
    "shuffle_null": "shuffled-pairing null",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:
        logger.warning("paper_plots style unavailable (%s); matplotlib default", e)


def _load(res_dir: Path, name: str) -> dict:
    with open(res_dir / name) as f:
        return json.load(f)


def fig_hero(heldout: dict, monitoring: dict, fig_dir: Path) -> None:
    """Two-panel hero: R2-vs-layer per arm | monitoring r per trait x mode x arm."""
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")

    per = heldout["per_input_layer"]
    layers = sorted(int(k) for k in per["last"])
    for name in ("avg10", "greedy", "stoch1_new", "stoch1_old", "mean_floor", "shuffle_null"):
        ys = [per["last"][str(li)][name]["r2_pooled"] for li in layers]
        axl.plot(
            layers,
            ys,
            marker="o",
            ms=2.5,
            lw=1.4,
            color=ARM_COLORS[name],
            label=ARM_LABELS[name],
        )
        if name in ("avg10", "greedy") and "mean" in per:
            ym = [per["mean"][str(li)][name]["r2_pooled"] for li in layers]
            axl.plot(layers, ym, lw=0.8, alpha=0.35, color=ARM_COLORS[name])
    axl.set_xlabel("layer")
    axl.set_ylabel("held-out pooled R2 (5-fold, test-own-mean)")
    axl.set_title("Map quality by decode regime (solid: cx_last; faint: cx_mean)")
    axl.legend(fontsize=7, loc="lower center")

    cells = [k for k, v in monitoring["cells"].items() if isinstance(v, dict) and "point_r" in v]
    arms = ["avg10", "greedy", "stoch1_new", "stoch1_old"]
    width = 0.19
    xs = np.arange(len(cells))
    for j, arm in enumerate(arms):
        pts = [monitoring["cells"][c]["point_r"][arm] for c in cells]
        los = [monitoring["cells"][c]["ci_r"][arm][0] for c in cells]
        his = [monitoring["cells"][c]["ci_r"][arm][1] for c in cells]
        err = [
            [max(0.0, p - lo) for p, lo in zip(pts, los, strict=True)],
            [max(0.0, hi - p) for p, hi in zip(pts, his, strict=True)],
        ]
        axr.bar(
            xs + (j - 1.5) * width,
            pts,
            width,
            yerr=err,
            capsize=2,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
    raw = [monitoring["cells"][c]["point_r"]["pv_raw"] for c in cells]
    for x, r in zip(xs, raw, strict=True):
        axr.hlines(r, x - 2 * width, x + 2 * width, colors="k", linestyles="dashed", lw=1.0)
    axr.set_xticks(xs)
    axr.set_xticklabels([c.replace("__", "\n") for c in cells], fontsize=7)
    axr.set_ylabel("within-condition Pearson r")
    axr.set_title("Monitoring transfer per arm (dashed: raw projection)")
    axr.legend(fontsize=7)

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "hero_decode_regime.png", dpi=150)
    import matplotlib.pyplot as _plt

    _plt.close(fig)
    logger.info("wrote %s", fig_dir / "hero_decode_regime.png")


def fig_percontext(agreement: dict, fig_dir: Path) -> None:
    """Per-unit companion: cosine distributions + <v, r_B> identity scatter."""
    import matplotlib.pyplot as plt

    layers = sorted(agreement["per_layer"], key=lambda s: int(s[1:]))
    li = layers[len(layers) // 2]
    ag = agreement["per_layer"][li]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")

    data = [
        np.asarray(ag["cos_greedy_avg10"]),
        np.asarray(ag["dv4_mean_cos_draw_loo9"]),
        np.asarray(ag["cos_stoch1_old_avg10"]),
        np.asarray(ag["cos_stoch1_new_avg10"]),
    ]
    labels = [
        "cos(greedy, avg10)",
        "mean_j cos(draw_j, loo9)",
        "cos(stoch1_old, avg10)",
        "cos(stoch1_new, avg10)",
    ]
    parts = axl.violinplot(data, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    band = ag["jackknife_draw_band"]
    axl.axhspan(band["p5"], band["p95"], color="0.85", zorder=0)
    axl.set_xticks(range(1, len(labels) + 1))
    axl.set_xticklabels(labels, fontsize=7, rotation=12)
    axl.set_ylabel(f"per-context cosine ({li})")
    axl.set_title("Per-context target agreement (shaded: jackknife draw band p5-p95)")

    proj = agreement["rb_projections_system_layer"]
    trait = sorted(proj)[0]
    g = np.asarray(proj[trait]["greedy"])
    a = np.asarray(proj[trait]["avg10"])
    axr.scatter(a, g, s=8, alpha=0.4)
    lo = float(min(a.min(), g.min()))
    hi = float(max(a.max(), g.max()))
    axr.plot([lo, hi], [lo, hi], color="k", lw=1.0)
    resid = np.abs(g - a)
    for i in np.argsort(resid)[-5:]:
        axr.annotate(str(int(i)), (a[i], g[i]), fontsize=6)
    axr.set_xlabel(f"<v_avg10, r_B>  ({trait}, L{proj[trait]['layer']})")
    axr.set_ylabel("<v_greedy, r_B>")
    axr.set_title("Per-context r_B projection: greedy vs 10-avg (labels: largest residuals)")

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "percontext_companion.png", dpi=150)
    plt.close(fig)
    logger.info("wrote %s", fig_dir / "percontext_companion.png")


def fig_exploratory(
    heldout: dict, agreement: dict, desc: dict, perdim: dict, p0: dict, fig_dir: Path
) -> None:
    """Exploratory dump (one multi-panel figure): DV2 curves, lengths,
    val-lambda deltas, per-dim R2 spectra, probe-consistency scatter."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")

    ax = axes[0][0]
    curves = agreement["curves"]
    for k, v in curves.items():
        ax.plot(range(len(v)), v, marker="o", ms=2, label=k)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean per-context cosine")
    ax.set_title("DV2: target agreement by layer")
    ax.legend(fontsize=6)

    ax = axes[0][1]
    for arm, d in desc["per_arm"].items():
        h = d["tokens_hist"]
        centers = 0.5 * (np.asarray(h["edges"][:-1]) + np.asarray(h["edges"][1:]))
        ax.step(
            centers, h["counts"], where="mid", label=f"{arm} (trunc {d['truncation_rate']:.3f})"
        )
    ax.set_xlabel("response tokens")
    ax.set_ylabel("count")
    ax.set_title("DV5: response length by arm")
    ax.legend(fontsize=6)

    ax = axes[0][2]
    vlr = heldout["val_lambda_robustness"]
    lkeys = [k for k in vlr if k.startswith("L")]
    for arm in ("avg10", "greedy", "stoch1_new", "stoch1_old"):
        ax.plot(
            [int(k[1:]) for k in lkeys],
            [vlr[k][arm]["delta"] for k in lkeys],
            marker="o",
            ms=3,
            color=ARM_COLORS[arm],
            label=arm,
        )
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("R2(val-selected) - R2(GCV)")
    ax.set_title("Val-selected-lambda robustness deltas")
    ax.legend(fontsize=6)

    ax = axes[1][0]
    keys = sorted(k for k in perdim if k != "metadata")
    if keys:
        key = keys[len(keys) // 2]
        for arm in ("avg10", "greedy", "stoch1_new"):
            spec = np.sort(np.asarray(perdim[key][arm]["r2_dim"]))[::-1]
            ax.plot(spec, lw=1.0, color=ARM_COLORS[arm], label=arm)
        ax.set_xlabel("hidden dim (sorted by R2)")
        ax.set_ylabel("per-dimension R2")
        ax.set_ylim(-1, 1)
        ax.set_title(f"Per-dimension R2 spectrum ({key})")
        ax.legend(fontsize=6)

    ax = axes[1][1]
    probe = p0["probe"]
    ax.scatter(probe["cos_band"], probe["cos_regen"], s=14)
    ax.axhline(probe["cos_regen_median"], color="k", ls="dashed", lw=0.8)
    ax.set_xlabel("cos(draw1, draw2) — fresh-draw band")
    ax.set_ylabel("cos(v_regen seed42, v_x_old)")
    ax.set_title(f"P0 probe consistency (branch {probe['branch']})")

    ax = axes[1][2]
    boot = heldout["bootstrap"]["per_layer"]
    lis = sorted(boot, key=lambda s: int(s[1:]))
    for arm in ("greedy", "stoch1_new", "stoch1_old"):
        pts = [boot[li][f"r2_gap__{arm}-avg10"]["point"] for li in lis]
        cis = [boot[li][f"r2_gap__{arm}-avg10"]["ci"] for li in lis]
        x = np.arange(len(lis))
        err = [
            [max(0.0, p - c[0]) for p, c in zip(pts, cis, strict=True)],
            [max(0.0, c[1] - p) for p, c in zip(pts, cis, strict=True)],
        ]
        ax.errorbar(
            x,
            pts,
            yerr=err,
            marker="o",
            ms=3,
            lw=1.0,
            capsize=2,
            color=ARM_COLORS[arm],
            label=f"{arm} - avg10",
        )
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhline(-0.07, color="0.5", lw=0.6, ls="dotted")
    ax.axhline(0.07, color="0.5", lw=0.6, ls="dotted")
    ax.set_xticks(np.arange(len(lis)))
    ax.set_xticklabels(lis)
    ax.set_ylabel("paired held-out R2 gap")
    ax.set_title("DV1 arm gaps vs the +-0.07 band (dotted)")
    ax.legend(fontsize=6)

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "exploratory_decode_regime.png", dpi=150)
    plt.close(fig)
    logger.info("wrote %s", fig_dir / "exploratory_decode_regime.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #1073 P5 figures (from JSONs only).")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--fig-dir", default=None)
    args = parser.parse_args()

    I.phase("p5")
    root = I.out_root(args.smoke, args.out_root)
    res_dir = Path(args.results_dir) if args.results_dir else I.results_dir(root, args.smoke)
    fig_dir = Path(args.fig_dir) if args.fig_dir else I.figures_dir(root, args.smoke)

    _style()
    heldout = _load(res_dir, "heldout_recon_arms.json")
    agreement = _load(res_dir, "target_agreement.json")
    monitoring = _load(res_dir, "monitoring_arms.json")
    desc = _load(res_dir, "decode_descriptives.json")
    perdim = _load(res_dir, "exploratory_perdim_r2.json")
    p0 = _load(res_dir, "p0_probe.json")

    fig_hero(heldout, monitoring, fig_dir)
    fig_percontext(agreement, fig_dir)
    fig_exploratory(heldout, agreement, desc, perdim, p0, fig_dir)
    logger.info("P5 done -> %s", fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
