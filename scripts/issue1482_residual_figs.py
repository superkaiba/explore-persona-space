"""Issue #1482 inline round — figures for the two-way residual decomposition and
the three residual-structure reductions.

Reads the JSONs written by ``issue1482_twoway_residual.py`` and
``issue1482_residual_svd.py``; renders nothing it did not read from disk.

Figure roster
  A ``twoway_variance_components``  the headline: EMS variance-component shares
      (context / direction / interaction) per cell, both normalizations, with the
      pure-noise SS expectations drawn as the reference they exist to beat.
  B ``twoway_k_sweep``              the same shares against basis size k -- the
      stability check that stops a single cell being quoted forward.
  C ``twoway_floor_corrected``      context share before/after subtracting the
      K-resample answer-entropy floor.
  D ``residual_spectrum``           top-k energy share of the residual against the
      isotropic and target-shaped references (the two that bracket it).
  E ``residual_effective_rank``     participation ratio, observed vs both
      references, per cell.
  F ``residual_consistency``        half-to-half subspace overlap against the
      three-level ladder (random floor / Gaussian-Sigma_E / observed).
  G ``worst_direction_profile``     per-direction held-out R^2 and what the worst
      directions align with, each against its matched null.

Colour contract (one colour = one meaning across every panel here):
  observed = primary, isotropic reference = neutral, target-shaped reference =
  accent, floor/noise = control, and the three variance components keep a fixed
  colour wherever they appear.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("i1482.residfigs")

RES_DIR = PROJECT_ROOT / "eval_results" / "issue_1482" / "twoway_residual"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1482" / "twoway_residual"

ARM_LABEL = {"context": "Context arm", "prefix": "Prefix arm", "bare": "Bare-query arm"}
COMPONENT_LABEL = {
    "context": "Context (which conversation)",
    "direction": "Direction (which answer axis)",
    "interaction": "Interaction (this context x this direction)",
}


def _pal() -> dict[str, str]:
    p = paper_palette(8)
    return {
        # the three variance components
        "context": p[0],
        "direction": p[1],
        "interaction": p[2],
        # spectrum reads
        "observed": p[0],
        "isotropic": p[7],
        "shaped": p[3],
        "floor": p[4],
        "corrected": p[5],
        "gaussian": p[3],
        "random": p[7],
    }


def _load(name: str) -> dict:
    path = RES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"required input missing: {path}")
    return json.loads(path.read_text())


def _cell_label(name: str) -> str:
    arm, layer, *rest = name.split("_")
    fitter = "_".join(rest)
    ly = layer.lstrip("L")
    fit = "ridge" if fitter == "ridge" else "MLP"
    return f"{ARM_LABEL[arm].split()[0]}\nlayer {ly} {fit}"


def _order(cells: dict) -> list[str]:
    def key(n: str) -> tuple:
        arm, layer, *rest = n.split("_")
        return (
            int(layer.lstrip("L")),
            "_".join(rest) != "ridge",
            ["context", "prefix", "bare"].index(arm),
        )

    return sorted(cells, key=key)


# ── A: variance components ────────────────────────────────────────────────────


def fig_variance_components(tw: dict, k: str) -> None:
    c = _pal()
    names = _order(tw["cells"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True)
    for ax, norm, title in zip(
        axes,
        ("raw", "normalized"),
        (
            "Raw squared residual (high-variance answer directions dominate by construction)",
            "Per-direction normalized (error relative to what there was to predict)",
        ),
        strict=True,
    ):
        x = np.arange(len(names))
        bottom = np.zeros(len(names))
        for comp in ("context", "direction", "interaction"):
            vals = np.array(
                [tw["cells"][n]["by_k"][k][norm][f"vc_share_{comp}"] for n in names], dtype=float
            )
            ax.bar(x, vals, bottom=bottom, color=c[comp], label=COMPONENT_LABEL[comp], width=0.72)
            bottom += vals
        noise_ctx = np.array(
            [tw["cells"][n]["by_k"][k][norm]["ss_share_context_noise_expectation"] for n in names]
        )
        ax.plot(
            x,
            noise_ctx,
            marker="_",
            markersize=22,
            markeredgewidth=2.5,
            linestyle="none",
            color="white",
            zorder=10,
            label="Context share expected from pure noise at this (n, k)",
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of held-out squared residual")
        ax.set_title(title, loc="left")
        ax.set_xticks(x)
    axes[-1].set_xticklabels([_cell_label(n) for n in names], fontsize=8)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=2, frameon=False)
    fig.suptitle(
        f"Where the context-to-answer map's error lives (variance components, k={k} answer directions)",
        y=1.06,
        fontsize=13,
    )
    savefig_paper(fig, "twoway_variance_components", dir=FIG_DIR)
    plt.close(fig)


# ── B: k sweep ────────────────────────────────────────────────────────────────


def fig_k_sweep(tw: dict) -> None:
    c = _pal()
    names = _order(tw["cells"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, sharex=True)
    layers = sorted({tw["cells"][n]["layer"] for n in names})
    panels = [(norm, ly) for norm in ("raw", "normalized") for ly in layers]
    for ax, (norm, ly) in zip(axes.ravel(), panels, strict=True):
        sel = [
            n
            for n in names
            if tw["cells"][n]["layer"] == ly and tw["cells"][n]["fitter"] == "ridge"
        ]
        for comp in ("context", "direction", "interaction"):
            for i, n in enumerate(sel):
                ks = sorted(tw["cells"][n]["by_k"], key=int)
                ys = [tw["cells"][n]["by_k"][kk][norm][f"vc_share_{comp}"] for kk in ks]
                ax.plot(
                    [int(kk) for kk in ks],
                    ys,
                    color=c[comp],
                    linestyle=["-", "--", ":"][i],
                    marker="o",
                    markersize=3,
                    label=f"{COMPONENT_LABEL[comp]} — {ARM_LABEL[tw['cells'][n]['arm']]}"
                    if norm == "raw" and ly == layers[0]
                    else None,
                )
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"layer {ly} — {'raw' if norm == 'raw' else 'normalized'}", loc="left", fontsize=10
        )
    for ax in axes[-1]:
        ax.set_xlabel("Number of answer directions k")
    axes[0, 0].set_ylabel("Variance-component share")
    axes[1, 0].set_ylabel("Variance-component share")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False, fontsize=8
    )
    fig.suptitle(
        "Stability of the decomposition across basis size, arm and layer (line style = arm)",
        y=1.16,
        fontsize=13,
    )
    savefig_paper(fig, "twoway_k_sweep", dir=FIG_DIR)
    plt.close(fig)


# ── C: floor correction ───────────────────────────────────────────────────────


def fig_floor(tw: dict) -> None:
    c = _pal()
    fc = tw["floor_correction"]
    # floor_correction is keyed "L14"/"L19"/"L26", not by bare integer
    rows = [
        (ly, arm)
        for ly in sorted(fc, key=lambda s: int(str(s).lstrip("L")))
        for arm in ("context", "prefix", "bare")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, norm in zip(axes, ("raw", "normalized"), strict=True):
        x = np.arange(len(rows))
        unc = [fc[ly][arm][f"uncorrected_{norm}"]["vc_share_context"] for ly, arm in rows]
        cor = [fc[ly][arm][f"corrected_{norm}"]["vc_share_context"] for ly, arm in rows]
        ax.bar(x - 0.2, unc, width=0.38, color=c["observed"], label="Before floor subtraction")
        ax.bar(
            x + 0.2,
            cor,
            width=0.38,
            color=c["floor"],
            label="After subtracting answer-sampling floor",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{ARM_LABEL[a].split()[0]}\nlayer {str(ly).lstrip(chr(76))}" for ly, a in rows],
            fontsize=8,
        )
        ax.set_title(f"{'Raw' if norm == 'raw' else 'Normalized'} squared residual", loc="left")
    axes[0].set_ylabel("Context variance-component share")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "The context component is small before and after removing answer-sampling noise",
        y=1.02,
        fontsize=13,
    )
    savefig_paper(fig, "twoway_floor_corrected", dir=FIG_DIR)
    plt.close(fig)


# ── D: residual spectrum ──────────────────────────────────────────────────────


def fig_spectrum(sp: dict) -> None:
    c = _pal()
    names = _order(sp["cells"])
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow), sharex=True, sharey=True)
    for ax, n in zip(axes.ravel(), names, strict=False):
        rec = sp["cells"][n]
        ks = sorted(rec["observed"]["topk_energy_share"], key=int)
        xs = [int(k) for k in ks]
        ax.plot(
            xs,
            [rec["observed"]["topk_energy_share"][k] for k in ks],
            color=c["observed"],
            marker="o",
            markersize=3,
            label="Observed residual",
        )
        ax.plot(
            xs,
            [rec["null_iso"]["topk_energy_share"][k]["mean"] for k in ks],
            color=c["isotropic"],
            linestyle="--",
            label="Isotropic reference (matched row energy)",
        )
        ax.plot(
            xs,
            [rec["null_shaped"]["topk_energy_share"][k]["mean"] for k in ks],
            color=c["shaped"],
            linestyle="-.",
            label="Target-covariance reference",
        )
        if "floor" in rec:
            fk = rec["floor"]["shift_corrected_on_sub"]["topk_energy_share"]
            ax.plot(
                [int(k) for k in sorted(fk, key=int)],
                [fk[k] for k in sorted(fk, key=int)],
                color=c["corrected"],
                linestyle=":",
                label="Observed, answer-noise removed",
            )
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1)
        ax.set_title(_cell_label(n).replace("\n", " "), loc="left", fontsize=9)
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Top-k residual directions")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of residual energy")
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False, fontsize=9
    )
    fig.suptitle(
        "The residual is far more concentrated than isotropic noise, and far less "
        "concentrated than the answer space it lives in",
        y=1.08,
        fontsize=13,
    )
    savefig_paper(fig, "residual_spectrum", dir=FIG_DIR)
    plt.close(fig)


# ── E: effective rank ─────────────────────────────────────────────────────────


def fig_effective_rank(sp: dict) -> None:
    c = _pal()
    names = _order(sp["cells"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(
        x - 0.26,
        [sp["cells"][n]["null_shaped"]["participation_ratio"]["mean"] for n in names],
        width=0.25,
        color=c["shaped"],
        label="Target-covariance reference",
    )
    ax.bar(
        x,
        [sp["cells"][n]["observed"]["participation_ratio"] for n in names],
        width=0.25,
        color=c["observed"],
        label="Observed residual",
    )
    ax.bar(
        x + 0.26,
        [sp["cells"][n]["null_iso"]["participation_ratio"]["mean"] for n in names],
        width=0.25,
        color=c["isotropic"],
        label="Isotropic reference",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_cell_label(n) for n in names], fontsize=8)
    ax.set_ylabel("Participation ratio (effective number of directions)")
    ax.set_ylim(top=ax.get_ylim()[1] * 6)  # headroom so the legend clears the tallest bar
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    ax.set_title(
        "Effective rank of the residual sits between the two references in every cell "
        "(log scale; 3,584 dimensions available)",
        loc="left",
    )
    savefig_paper(fig, "residual_effective_rank", dir=FIG_DIR)
    plt.close(fig)


# ── F: consistency ────────────────────────────────────────────────────────────


def fig_consistency(cons: dict) -> None:
    c = _pal()
    names = _order(cons["cells"])
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names, strict=True):
        rec = cons["cells"][n]
        ks = sorted(rec["by_k"], key=int)
        xs = [int(k) for k in ks]
        ax.plot(
            xs,
            [rec["by_k"][k]["observed"]["subspace_overlap"] for k in ks],
            color=c["observed"],
            marker="o",
            label="Observed (two disjoint context halves)",
        )
        ax.plot(
            xs,
            [rec["by_k"][k]["gaussian_sigma_e"]["subspace_overlap"]["mean"] for k in ks],
            color=c["gaussian"],
            linestyle="-.",
            marker="s",
            markersize=3,
            label="Same covariance, no further structure",
        )
        ax.plot(
            xs,
            [rec["by_k"][k]["random_floor"]["subspace_overlap"]["mean"] for k in ks],
            color=c["random"],
            linestyle="--",
            label="Unrelated random subspaces",
        )
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Subspace size k")
        ax.set_title(_cell_label(n).replace("\n", " "), loc="left", fontsize=9)
    axes[0].set_ylabel("Subspace overlap between halves")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=9
    )
    fig.suptitle(
        "Do disjoint halves of the conversations fail in the same directions?", y=1.16, fontsize=13
    )
    savefig_paper(fig, "residual_consistency", dir=FIG_DIR)
    plt.close(fig)


# ── G: worst-direction profile + alignment ────────────────────────────────────


def fig_worst_directions(al: dict) -> None:
    c = _pal()
    name = next(iter(al["cells"]))
    rec = al["cells"][name]
    r2 = np.array(rec["per_direction_r2_top256"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    ax.plot(np.arange(r2.size), r2, color=c["observed"], linewidth=1.0)
    worst = rec["worst_indices"]
    ax.scatter(worst, r2[worst], color=c["floor"], zorder=5, s=26, label="20 worst-predicted")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Answer direction (target principal component, by variance)")
    ax.set_ylabel("Held-out $R^2$")
    ax.set_title("Per-direction predictability", loc="left")
    ax.legend(frameon=False, fontsize=9)

    # cosine-style alignments (traits, SAE) — same unit, each against its own null
    ax = axes[1]
    labels, vals, nulls = [], [], []
    rb = rec["r_b_alignment"]
    labels.append("Trait directions\n(max over 3 traits)")
    vals.append(rb["max_abs_cos_worst"])
    nulls.append(rb["null_random_unit"]["max"])
    if "sae_alignment" in rec:
        s = rec["sae_alignment"]
        labels.append("SAE dictionary\n(max over 131k features)")
        vals.append(float(np.max(s["max_abs_cos_per_worst"])))
        nulls.append(s["null_random_unit_max_over_dictionary"]["max"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, vals, width=0.38, color=c["observed"], label="Worst-predicted directions")
    ax.bar(x + 0.2, nulls, width=0.38, color=c["isotropic"], label="Random-direction null (max)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Max |cosine|")
    ax.set_title("Alignment with known bases", loc="left")
    ax.legend(frameon=False, fontsize=8)

    # gain-end mass — a DIFFERENT unit (fraction of squared mass), own panel
    ax = axes[2]
    g = rec["gain_end_alignment"]
    hi = float(np.mean(g["mass_in_high_gain_end"]))
    lo = float(np.mean(g["mass_in_low_gain_end"]))
    ax.bar([0, 1], [hi, lo], width=0.55, color=[c["observed"], c["floor"]])
    ax.axhline(
        g["uniform_expectation"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="If spread evenly over all 3,584 directions",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Map's strongest\n256 directions", "Map's weakest\n256 directions"], fontsize=8
    )
    ax.set_ylabel("Fraction of squared mass")
    ax.set_title("Where they sit in the map's own gain spectrum", loc="left")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"The worst-predicted answer directions — {_cell_label(name).replace(chr(10), ' ')}",
        y=1.06,
        fontsize=13,
    )
    savefig_paper(fig, "worst_direction_profile", dir=FIG_DIR)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", default="16", help="basis size for the variance-component bar figure")
    ap.add_argument(
        "--figs",
        nargs="*",
        default=["A", "B", "C", "D", "E", "F", "G"],
        help="subset of the figure roster to render",
    )
    args = ap.parse_args()
    set_paper_style("blog")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    want = set(args.figs)
    if want & {"A", "B", "C"}:
        tw = _load("twoway_residual.json")
        if "A" in want:
            fig_variance_components(tw, args.k)
        if "B" in want:
            fig_k_sweep(tw)
        if "C" in want:
            fig_floor(tw)
    if want & {"D", "E"}:
        sp = _load("residual_spectrum.json")
        if "D" in want:
            fig_spectrum(sp)
        if "E" in want:
            fig_effective_rank(sp)
    if "F" in want:
        fig_consistency(_load("residual_consistency.json"))
    if "G" in want:
        fig_worst_directions(_load("residual_alignment.json"))
    logger.info("figures -> %s", FIG_DIR)


if __name__ == "__main__":
    main()
