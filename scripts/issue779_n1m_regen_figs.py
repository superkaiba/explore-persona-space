"""Regenerate #779 n1m-readout figures with plain-English arm labels + flag marks.

Revision-round figure fixes (interp-critique round 1, task #779, follow-up
`n1m-nonlinear-map-behavior-readout`):

1. hero / delta forest / grouped sweep / R2 transfer: replace bare arm slugs
   ("h n5k linear", "n1m krr nystrom") with plain-English labels, and mark the
   FLAGGED layer-26 kernel arm (Nystrom-vs-exact gate gap 0.0151 > tol 0.01,
   `n1m_multilayer_fits.json .per_layer."26".nystrom_validation.gate_passed:
   false`) with a dagger on every use surface.
2. NEW `n1m_readout_l19_forest`: the persisted-but-previously-unplotted
   `.l19_continuity` read — delta vs raw (dot, 95% CI) for all five map arms in
   all six trait-mode cells, every arm read at capture layer 19.

Reads the committed round JSONs only; no recomputation. Fail loud.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODES = ("system", "many_shot")
TRAITS = ("evil", "sycophancy", "hallucination")
MAP_ARMS = ("h_n5k_linear", "n1m_ridge", "n1m_mlp_w8192", "n1m_mlp_w32768", "n1m_krr_nystrom")
SUBSTITUTED_CELLS = {"hallucination/system", "hallucination/many_shot"}
FLAGGED_LAYER = 26  # L26 kernel arm failed the Nystrom-vs-exact gate

ARM_LABELS = {
    "pv_raw": "raw projection",
    "h_n5k_linear": "5k linear map",
    "n1m_ridge": "963k ridge",
    "n1m_mlp_w8192": "963k MLP (w=8192)",
    "n1m_mlp_w32768": "963k MLP (w=32768)",
    "n1m_krr_nystrom": "963k kernel (Nyström)",
    "oracle": "oracle (true answer proj.)",
    "h_n5k_logo": "5k map (LOGO refit)",
    "pv_raw_group": "raw projection",
}
FITTER_LABELS = {
    "ridge": "963k ridge",
    "mlp_w8192": "963k MLP (w=8192)",
    "mlp_w32768": "963k MLP (w=32768)",
    "krr_nystrom": "963k kernel (Nyström)",
}


def _label(arm: str, layer: int) -> str:
    base = ARM_LABELS[arm]
    if arm == "n1m_krr_nystrom" and layer == FLAGGED_LAYER:
        return base + " †"
    return base


def make_figures(res: dict, fits: dict, fig_dir: str) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    # Guard the dagger semantics: the L26 kernel gate must actually be failed.
    gate26 = fits["per_layer"]["26"]["nystrom_validation"]
    assert gate26["gap"] > gate26["tol"], "L26 Nystrom gate no longer failed — dagger is stale"
    figs: dict[str, str] = {}
    colors = paper_palette(3)
    bar_arms = ["pv_raw", *MAP_ARMS, "oracle"]
    bar_colors = [colors[0], colors[1], colors[1]] + [colors[2]] * 4 + [colors[0]]

    # HERO: grouped bars of within-condition r across the arms (dot readout).
    fig, axes = plt.subplots(
        2, len(TRAITS), figsize=(4.6 * len(TRAITS) + 0.6, 8.6), squeeze=False, layout="tight"
    )
    for col, trait in enumerate(TRAITS):
        for row, mode in enumerate(MODES):
            ax = axes[row][col]
            entry = res["headline"][trait][mode]
            layer = int(entry["layer"])
            heights, errs, labels = [], [], []
            for arm in bar_arms:
                name = arm if arm in ("pv_raw", "oracle") else f"{arm}_dot"
                mm = entry["monitors"][name]
                pt = mm["point"]
                if not np.isfinite(pt):
                    continue
                heights.append(pt)
                lo, hi = mm["lo"], mm["hi"]
                errs.append(
                    [
                        max(0.0, pt - lo) if np.isfinite(lo) else 0.0,
                        max(0.0, hi - pt) if np.isfinite(hi) else 0.0,
                    ]
                )
                labels.append(_label(arm, layer))
            ax.bar(
                range(len(heights)),
                heights,
                yerr=np.array(errs).T if errs else None,
                capsize=2,
                color=bar_colors[: len(heights)],
            )
            ax.axhline(0.0, color="gray", lw=0.6)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            mode_lbl = "system prompting" if mode == "system" else "many-shot"
            sub = ", substitute layer" if f"{trait}/{mode}" in SUBSTITUTED_CELLS else ""
            ax.set_title(f"{trait} — {mode_lbl} (L{layer}{sub})")
            if col == 0:
                ax.set_ylabel("within-condition Pearson r (dot readout)")
    out = savefig_paper(fig, "n1m_readout_hero", dir=fig_dir)
    plt.close(fig)
    figs["hero"] = str(out.get("png", ""))

    # Delta-vs-raw forest (dot readout) with the +0.05 bar, frozen headline layers.
    fig, axes = plt.subplots(
        1, len(TRAITS), figsize=(4.8 * len(TRAITS), 5.2), squeeze=False, layout="tight"
    )
    for col, trait in enumerate(TRAITS):
        ax = axes[0][col]
        ypos, ylab = 0, []
        for mode in MODES:
            entry = res["headline"][trait][mode]
            layer = int(entry["layer"])
            mode_lbl = "system" if mode == "system" else "many-shot"
            for arm in MAP_ARMS:
                d = entry["deltas_vs_pv_raw"][f"{arm}_dot"]
                ax.errorbar(
                    d["delta"],
                    ypos,
                    xerr=[[max(0.0, d["delta"] - d["lo"])], [max(0.0, d["hi"] - d["delta"])]],
                    fmt="o",
                    capsize=2,
                    color=colors[0] if mode == "system" else colors[1],
                )
                ylab.append(f"{mode_lbl}: {_label(arm, layer)}")
                ypos += 1
            ypos += 1
            ylab.append("")
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axvline(0.05, color=colors[2], lw=1.0, ls="--", label="+0.05 bar")
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("delta within-condition r vs raw")
        ax.set_title(trait)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")
    out = savefig_paper(fig, "n1m_readout_delta_forest", dir=fig_dir)
    plt.close(fig)
    figs["delta_forest"] = str(out.get("png", ""))

    # NEW: layer-19 continuity forest — same delta layout, every arm read at L19.
    fig, axes = plt.subplots(
        1, len(TRAITS), figsize=(4.8 * len(TRAITS), 5.2), squeeze=False, layout="tight"
    )
    for col, trait in enumerate(TRAITS):
        ax = axes[0][col]
        ypos, ylab = 0, []
        for mode in MODES:
            entry = res["l19_continuity"][trait][mode]
            mode_lbl = "system" if mode == "system" else "many-shot"
            for arm in MAP_ARMS:
                d = entry["deltas_vs_pv_raw"][f"{arm}_dot"]
                ax.errorbar(
                    d["delta"],
                    ypos,
                    xerr=[[max(0.0, d["delta"] - d["lo"])], [max(0.0, d["hi"] - d["delta"])]],
                    fmt="o",
                    capsize=2,
                    color=colors[0] if mode == "system" else colors[1],
                )
                ylab.append(f"{mode_lbl}: {_label(arm, 19)}")
                ypos += 1
            ypos += 1
            ylab.append("")
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axvline(0.05, color=colors[2], lw=1.0, ls="--", label="+0.05 bar")
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("delta r vs raw (all arms at L19)")
        ax.set_title(trait)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")
    out = savefig_paper(fig, "n1m_readout_l19_forest", dir=fig_dir)
    plt.close(fig)
    figs["l19_forest"] = str(out.get("png", ""))

    # Grouped sweep: r vs group size, parent LOGO vs fixed arms.
    sweep_arms = [
        "h_n5k_logo",
        "h_n5k_linear",
        "n1m_ridge",
        "n1m_mlp_w8192",
        "n1m_mlp_w32768",
        "n1m_krr_nystrom",
        "pv_raw_group",
    ]
    pal = paper_palette(max(3, len(sweep_arms)))
    fig, axes = plt.subplots(
        1, len(TRAITS), figsize=(5.0 * len(TRAITS), 4.6), squeeze=False, layout="tight"
    )
    for col, trait in enumerate(TRAITS):
        ax = axes[0][col]
        glayer = int(res["grouped"][trait]["layer"])
        d = res["grouped"][trait]["group_size_sweep"]
        sizes = sorted(int(s) for s in d)
        for ai, arm in enumerate(sweep_arms):
            means = [d[str(s)][arm]["dot_r_mean"] for s in sizes]
            sds = [d[str(s)][arm]["dot_r_sd"] for s in sizes]
            ax.errorbar(
                sizes,
                means,
                yerr=sds,
                marker="o",
                ms=3,
                capsize=2,
                color=pal[ai % len(pal)],
                label=_label(arm, glayer),
            )
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("questions averaged per persona group")
        ax.set_ylabel("Pearson r vs mean judge score (dot)")
        ax.set_title(f"{trait} (L{glayer})")
        ax.legend(fontsize=6, loc="lower right")
    out = savefig_paper(fig, "n1m_readout_grouped_sweep", dir=fig_dir)
    plt.close(fig)
    figs["grouped_sweep"] = str(out.get("png", ""))

    # Fit-quality transfer: held-out test R2 at each layer per fitter.
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="tight")
    r2 = {int(k): v for k, v in res["fit_quality"]["per_layer_test_r2"].items()}
    layers = sorted(r2)
    pal = paper_palette(max(3, len(FITTER_LABELS)))
    for fi, fitter in enumerate(("ridge", "mlp_w8192", "mlp_w32768", "krr_nystrom")):
        ax.plot(
            layers,
            [r2[li][fitter] for li in layers],
            marker="o",
            color=pal[fi % len(pal)],
            label=FITTER_LABELS[fitter],
        )
    # Dagger the flagged point: the kernel arm failed its Nystrom-vs-exact gate at L26 only.
    ax.text(
        FLAGGED_LAYER,
        r2[FLAGGED_LAYER]["krr_nystrom"] - 0.006,
        "†",
        ha="center",
        va="top",
        fontsize=10,
        color=pal[3 % len(pal)],
    )
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out whole-map R2 (pinned test)")
    ax.legend(fontsize=7, loc="lower right")
    out = savefig_paper(fig, "n1m_readout_r2_transfer", dir=fig_dir)
    plt.close(fig)
    figs["r2_transfer"] = str(out.get("png", ""))
    return figs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir", required=True, help="round eval_results dir with the two JSONs"
    )
    ap.add_argument("--fig-dir", required=True, help="output figures dir (figures/issue_779)")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    res = json.loads((rd / "n1m_readout.json").read_text())
    fits = json.loads((rd / "n1m_multilayer_fits.json").read_text())
    figs = make_figures(res, fits, args.fig_dir)
    for k, v in figs.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
