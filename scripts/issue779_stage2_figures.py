#!/usr/bin/env python3
"""Figures for issue #779 Stage-2 (CKA + coefficient subspace overlap).

Reads eval_results/issue_779/stage2_cka_subspace/stage2_cka_subspace.json and writes
3 paper-style figures to figures/issue_779/stage2_cka_*.{png,pdf} (+ meta.json):

1. stage2_cka_vs_layer   — LEG 1: set-to-set CKA(c_x, v_x) vs held-out recon R2, per layer.
2. stage2_cka_rb_capture — LEG 2(a): r_B mass captured by h's top-k output subspace
   (per trait at its read-out layer) + the map's own output-energy spectrum + random baseline.
3. stage2_cka_map_overlap — LEG 2(b): per-example vs averaged map — coefficient subspace
   overlap vs k + functional prediction agreement across layers.

Colorblind-safe palette, no annotation overlays (project figure conventions).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

TRAITS = ("evil", "sycophancy", "hallucination")
READ_OUT_LAYER = {"evil": 14, "sycophancy": 26, "hallucination": 17}
N_LAYERS = 28
HIDDEN = 3584
K_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def _load() -> dict:
    p = PROJECT_ROOT / "eval_results/issue_779/stage2_cka_subspace/stage2_cka_subspace.json"
    with open(p) as f:
        return json.load(f)


def fig_cka_vs_layer(d: dict, out_dir: Path) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="tight")
    xs = list(range(N_LAYERS))
    cka = [d["leg1_cka_cx_vx_by_layer"][str(li)] for li in xs]
    r2 = [d["recon_heldout_r2_by_layer"][str(li)] for li in xs]
    ax.plot(
        xs,
        cka,
        marker="o",
        ms=4,
        lw=1.5,
        color=paper_palette_role("primary"),
        label="set-to-set linear CKA(c_x, v_x)",
    )
    ax.plot(
        xs,
        r2,
        marker="s",
        ms=4,
        lw=1.5,
        color=paper_palette_role("baseline"),
        label="held-out reconstruction R2 of h  (5-fold)",
    )
    for t in TRAITS:
        ax.axvline(
            READ_OUT_LAYER[t], color=paper_palette_role("neutral"), lw=0.6, ls=":", alpha=0.5
        )
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("similarity / fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Context-answer representational similarity tracks reconstruction fidelity")
    ax.legend(loc="lower right", fontsize=8)
    savefig_paper(fig, "stage2_cka_vs_layer", dir=str(out_dir))
    plt.close(fig)


def fig_rb_capture(d: dict, out_dir: Path) -> None:
    set_paper_style()
    colors = paper_palette(3)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5), layout="tight")
    ks = [float(k) for k in K_GRID]
    for i, t in enumerate(TRAITS):
        li = READ_OUT_LAYER[t]
        rc = d["leg2_by_layer"][str(li)]["rb_capture"][t]["captured_frac_by_k_coeffSVD"]
        eng = d["leg2_by_layer"][str(li)]["W_output_energy_cumfrac"]
        cap = [rc[str(k)] for k in K_GRID]
        en = [eng[str(k)] for k in K_GRID]
        axl.plot(
            ks, cap, marker="o", ms=4, lw=1.6, color=colors[i], label=f"{t} r_B captured (L{li})"
        )
        axl.plot(
            ks,
            en,
            marker="",
            lw=1.2,
            ls="--",
            color=colors[i],
            alpha=0.7,
            label=f"{t} map output-energy (L{li})",
        )
        axr.plot(
            ks,
            [1.0 - c for c in cap],
            marker="o",
            ms=4,
            lw=1.6,
            color=colors[i],
            label=f"{t} (L{li})",
        )
    rand = [k / HIDDEN for k in K_GRID]
    axl.plot(
        ks,
        rand,
        lw=1.0,
        ls=":",
        color=paper_palette_role("neutral"),
        label="random direction (k/H)",
    )
    for ax in (axl, axr):
        ax.set_xscale("log")
        ax.set_xlabel("k  (top-k output singular subspace of h)")
    axl.set_ylabel("fraction of r_B mass captured  /  output-energy fraction")
    axl.set_ylim(0, 1)
    axl.set_title("r_B capture vs the map's variance spectrum", fontsize=11)
    axl.legend(loc="upper left", fontsize=7)
    axr.set_ylabel("fraction of r_B mass OUTSIDE top-k output subspace")
    axr.set_ylim(0, 1)
    axr.set_title("r_B mass outside h's top-k output subspace", fontsize=11)
    axr.legend(loc="upper right", fontsize=8)
    savefig_paper(fig, "stage2_cka_rb_capture", dir=str(out_dir))
    plt.close(fig)


def fig_map_overlap(d: dict, out_dir: Path) -> None:
    set_paper_style()
    colors = paper_palette(3)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5), layout="tight")
    ks_avg = [k for k in K_GRID if k <= 50]
    ksf = [float(k) for k in ks_avg]
    for i, t in enumerate(TRAITS):
        li = READ_OUT_LAYER[t]
        am = d["leg2_by_layer"][str(li)]["avg_map"]
        oo = am["output_subspace_overlap_by_k"]
        io = am["input_subspace_overlap_raw_by_k"]
        axl.plot(
            ksf,
            [oo[str(k)] for k in ks_avg],
            marker="o",
            ms=4,
            lw=1.6,
            color=colors[i],
            label=f"{t} output subspace (L{li})",
        )
        axl.plot(
            ksf,
            [io[str(k)] for k in ks_avg],
            marker="",
            lw=1.2,
            ls="--",
            color=colors[i],
            alpha=0.7,
            label=f"{t} input subspace, raw (L{li})",
        )
    rand = [k / HIDDEN for k in ks_avg]
    axl.plot(
        ksf,
        rand,
        lw=1.0,
        ls=":",
        color=paper_palette_role("neutral"),
        label="random subspaces (k/H)",
    )
    axl.set_xscale("log")
    axl.set_xlabel("k  (top-k singular subspace)")
    axl.set_ylabel("mean squared cosine overlap  (per-example vs averaged map)")
    axl.set_ylim(0, 0.5)
    axl.set_title("Coefficient-subspace overlap: h vs averaged map", fontsize=11)
    axl.legend(loc="upper left", fontsize=7)

    xs = list(range(N_LAYERS))
    cka = [
        d["leg2_by_layer"][str(li)]["avg_map"]["functional_agreement_on_grid"]["linear_cka"]
        for li in xs
    ]
    cos = [
        d["leg2_by_layer"][str(li)]["avg_map"]["functional_agreement_on_grid"][
            "mean_per_context_cosine"
        ]
        for li in xs
    ]
    axr.plot(
        xs,
        cka,
        marker="o",
        ms=4,
        lw=1.5,
        color=paper_palette_role("primary"),
        label="linear CKA of predictions",
    )
    axr.plot(
        xs,
        cos,
        marker="s",
        ms=4,
        lw=1.5,
        color=paper_palette_role("baseline"),
        label="mean per-context cosine",
    )
    for t in TRAITS:
        axr.axvline(
            READ_OUT_LAYER[t], color=paper_palette_role("neutral"), lw=0.6, ls=":", alpha=0.5
        )
    axr.set_xlabel("transformer layer")
    axr.set_ylabel("prediction agreement on the 50 grid contexts")
    axr.set_ylim(0, 1)
    axr.set_title("Functional agreement (corpus-confounded — see caveat)", fontsize=11)
    axr.legend(loc="lower right", fontsize=8)
    savefig_paper(fig, "stage2_cka_map_overlap", dir=str(out_dir))
    plt.close(fig)


def main() -> int:
    d = _load()
    out_dir = PROJECT_ROOT / "figures" / "issue_779"
    fig_cka_vs_layer(d, out_dir)
    fig_rb_capture(d, out_dir)
    fig_map_overlap(d, out_dir)
    print("wrote 3 figures to", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
