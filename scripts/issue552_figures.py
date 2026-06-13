#!/usr/bin/env python3
"""#552 Step 10 (OFF-POD, VM) — figures for the benign-control geometry contrast.

Hero + exploratory dump (plan §6.4; over-produce, analyzer picks):

  1. hero_three_arm           — per-persona cos(Delta-v, U1) dots for 9 cells
                                (marker x3 / EM x3 / benign x3, same variant;
                                diamond = medical_doctor source) + s1/sum(s)
                                bars with both null p95 ticks per cell.
  2. benign_per_persona       — benign per-persona cosines sorted by 3-seed mean.
  3. variant_robustness       — mean cos under same/base/on_policy per arm.
  4. cross_arm_directions     — |cos(U1, U1')| strip plot by pair class vs the
                                0.033 random floor.
  5. inverted_gate_rates      — per-seed gate rate bars vs the 5% line and the
                                #458 benign-cell band.
  6. split_half_vs_cos        — per-persona split-half reliability vs cos-to-U1
                                (attenuation diagnostic; skipped loudly when the
                                cross-arm summary was built --skip-split-half).
  7. fro_norm_per_cell        — ||M||_F per cell across the 9 same-variant cells.

Run (VM, after issue552_cross_arm_analysis.py)::

    uv run python scripts/issue552_figures.py
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger(__name__)

SEEDS = [42, 137, 256]
VARIANTS = ["same", "base", "on_policy"]
ARM_LABEL = {
    "marker": "Marker\n(contrastive)",
    "em": "Misalignment\nSFT",
    "benign": "Benign\nSFT",
}
ARM_LABEL_FLAT = {
    "marker": "Marker (contrastive)",
    "em": "Misalignment SFT",
    "benign": "Benign SFT",
}
RANDOM_COS_FLOOR_P95 = 0.033
GATE_THRESHOLD = 0.05
GATE_N = 800  # 8 probes x 100 samples per cell

set_paper_style("blog")
mpl.rcParams["figure.constrained_layout.use"] = False

C_EM = paper_palette_role("primary")
C_MARKER = paper_palette_role("baseline")
C_BENIGN = paper_palette_role("control")
C_ACCENT = paper_palette_role("accent")
C_NEUTRAL = paper_palette_role("neutral")
ARM_COLOR = {"marker": C_MARKER, "em": C_EM, "benign": C_BENIGN}


def _load_cell(args, variant: str, arm: str, seed: int) -> dict:
    svd_dir = Path(args.benign_svd_dir if arm == "benign" else args.parent_svd_dir)
    p = svd_dir / f"{variant}_{arm}_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"per-cell SVD JSON missing: {p}")
    return json.loads(p.read_text())


def _arm_group_xticks(ax, cells) -> None:
    """Seed-only tick labels + one colored arm label under each seed triplet.

    The per-cell two-line labels ("Misalignment\nSFT\nseed 42" x 9) collide at
    panel widths under ~6 in; seed ticks + group annotations stay legible.
    """
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"seed {s}" for _, s in cells], fontsize=8)
    for start, arm in ((0, "marker"), (3, "em"), (6, "benign")):
        ax.text(
            start + 1,
            -0.13,
            ARM_LABEL_FLAT[arm],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="semibold",
            color=ARM_COLOR[arm],
        )


# ---------------------------------------------------------------- figure 1
def fig_hero(args) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.subplots_adjust(top=0.82, bottom=0.2, left=0.07, right=0.98, wspace=0.22)
    cells = [(arm, s) for arm in ("marker", "em", "benign") for s in SEEDS]
    rng = np.random.default_rng(0)

    ax = axes[0]
    for x, (arm, seed) in enumerate(cells):
        d = _load_cell(args, "same", arm, seed)
        cos = np.array(d["cos_to_U1"])
        jit = rng.uniform(-0.13, 0.13, size=len(cos))
        for j, p in enumerate(d["persona_order"]):
            if p == "medical_doctor":
                ax.scatter(
                    x + jit[j],
                    cos[j],
                    marker="D",
                    s=46,
                    facecolor="white",
                    edgecolor=C_ACCENT,
                    linewidth=1.6,
                    zorder=5,
                )
            else:
                ax.scatter(x + jit[j], cos[j], s=24, color=ARM_COLOR[arm], alpha=0.75, zorder=3)
    _arm_group_xticks(ax, cells)
    ax.set_ylabel("per-persona |cos(shift, top direction)|")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-persona alignment to the cell's top shift direction")

    ax = axes[1]
    xs = np.arange(9)
    for x, (arm, seed) in enumerate(cells):
        d = _load_cell(args, "same", arm, seed)
        ax.bar(x, d["s_top1_frac"], width=0.62, color=ARM_COLOR[arm], alpha=0.85, zorder=3)
        for null_key, marker in (("sign_flip_p95", "_"), ("row_shuffle_p95", "x")):
            ax.scatter(
                x,
                d[null_key],
                marker=marker,
                s=90 if marker == "_" else 28,
                color="black",
                zorder=5,
            )
    _arm_group_xticks(ax, cells)
    ax.set_ylabel("top singular value share  $\\sigma_1 / \\Sigma\\sigma$")
    ax.set_ylim(0, 1.0)
    ax.set_title("Direction concentration vs nulls (dash = sign-flip p95, x = row-shuffle p95)")

    fig.suptitle(
        "Does plain benign SFT write one shared shift direction? "
        "Marker vs misalignment vs benign, same measurement",
        fontsize=12,
    )
    savefig_paper(fig, "hero_three_arm", dir=args.out_dir)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_benign_per_persona(args) -> None:
    cells = [_load_cell(args, "same", "benign", s) for s in SEEDS]
    personas = cells[0]["persona_order"]
    cos_by_seed = np.array([c["cos_to_U1"] for c in cells])  # (3, 14)
    order = np.argsort(cos_by_seed.mean(axis=0))[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    fig.subplots_adjust(bottom=0.30, top=0.86, left=0.09, right=0.98)
    for i, seed in enumerate(SEEDS):
        ax.scatter(
            np.arange(len(personas)),
            cos_by_seed[i, order],
            s=26,
            color=C_BENIGN,
            alpha=0.45 + 0.2 * i,
            label=f"seed {seed}",
        )
    labels = [personas[j].replace("_", " ") for j in order]
    ax.set_xticks(range(len(personas)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for tick, j in zip(ax.get_xticklabels(), order, strict=True):
        if personas[j] == "medical_doctor":
            tick.set_fontweight("bold")
    ax.set_ylabel("|cos(shift, top direction)|")
    ax.set_title(
        "Benign SFT arm: per-persona alignment, sorted by 3-seed mean "
        "(bold = source-domain persona)"
    )
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "benign_per_persona", dir=args.out_dir)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_variant_robustness(args) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    fig.subplots_adjust(bottom=0.14, top=0.86, left=0.11, right=0.97)
    xs = np.arange(len(VARIANTS))
    for arm in ("marker", "em", "benign"):
        per_seed = np.array(
            [[_load_cell(args, v, arm, s)["mean_cos_to_U1"] for v in VARIANTS] for s in SEEDS]
        )  # (3 seeds, 3 variants)
        for row in per_seed:
            ax.plot(xs, row, color=ARM_COLOR[arm], alpha=0.30, linewidth=1.0)
        ax.plot(
            xs,
            np.median(per_seed, axis=0),
            color=ARM_COLOR[arm],
            linewidth=2.2,
            marker="o",
            label=ARM_LABEL_FLAT[arm],
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(["same trajectory", "base trajectory", "on-policy"])
    ax.set_ylabel("mean |cos(shift, top direction)|")
    ax.set_ylim(0, 1.05)
    ax.set_title("Variant sensitivity of direction concentration (thin = seeds, thick = median)")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "variant_robustness", dir=args.out_dir)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_cross_arm_directions(args) -> None:
    u1 = {
        (arm, s): np.asarray(_load_cell(args, "same", arm, s)["U1"], dtype=np.float64)
        for arm in ("marker", "em", "benign")
        for s in SEEDS
    }

    def acos(a, b):
        return float(abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))

    # Within-arm strips (the seed-to-seed reliability ceilings) first, then the
    # three cross-arm strips. Within-arm dots take their arm's color (consistent
    # with every other figure); cross-arm pairs mix two arms, so they stay neutral.
    groups: dict[str, tuple[list[float], str]] = {
        "within\nbenign": (
            [acos(u1[("benign", a)], u1[("benign", b)]) for a, b in combinations(SEEDS, 2)],
            C_BENIGN,
        ),
        "within\nmisalignment": (
            [acos(u1[("em", a)], u1[("em", b)]) for a, b in combinations(SEEDS, 2)],
            C_EM,
        ),
        "within\nmarker": (
            [acos(u1[("marker", a)], u1[("marker", b)]) for a, b in combinations(SEEDS, 2)],
            C_MARKER,
        ),
        "benign x\nmisalignment": (
            [acos(u1[("benign", a)], u1[("em", b)]) for a in SEEDS for b in SEEDS],
            C_NEUTRAL,
        ),
        "benign x\nmarker": (
            [acos(u1[("benign", a)], u1[("marker", b)]) for a in SEEDS for b in SEEDS],
            C_NEUTRAL,
        ),
        "misalignment x\nmarker": (
            [acos(u1[("em", a)], u1[("marker", b)]) for a in SEEDS for b in SEEDS],
            C_NEUTRAL,
        ),
    }

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    fig.subplots_adjust(bottom=0.17, top=0.88, left=0.09, right=0.97)
    rng = np.random.default_rng(1)
    for x, (vals, color) in enumerate(groups.values()):
        jit = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(x + jit, vals, s=30, color=color, alpha=0.8, zorder=3)
        ax.scatter(x, float(np.median(vals)), marker="_", s=420, color="black", zorder=5)
    ax.axvline(2.5, color="0.85", linewidth=1.0, zorder=1)
    ax.text(
        1.0,
        1.02,
        "same corpus, different seed",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=8,
        color="0.45",
    )
    ax.text(
        4.0,
        1.02,
        "different corpora",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=8,
        color="0.45",
    )
    ax.axhline(
        RANDOM_COS_FLOOR_P95,
        color=C_ACCENT,
        linestyle="--",
        linewidth=1.2,
        label="random-direction floor (p95 = 0.033)",
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups.keys()), fontsize=8)
    ax.set_ylabel("|cos(top direction, top direction')|")
    ax.set_ylim(0, 1.05)
    ax.set_title("Is the benign top direction the misalignment one? (dash = group median)", pad=18)
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "cross_arm_directions", dir=args.out_dir)
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_inverted_gate(args) -> None:
    gate = json.loads(Path(args.gate_summary).read_text())
    per_cell = gate["per_cell_rates"]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    fig.subplots_adjust(bottom=0.14, top=0.86, left=0.17, right=0.96)
    xs = np.arange(len(per_cell))
    rates = [float(per_cell[k]) for k in sorted(per_cell)]
    los, his = [], []
    for r in rates:
        lo, hi = proportion_ci(r, GATE_N)
        # Clamp: saturated/zero rates make CI half-widths float-epsilon
        # negative (constant-bootstrap memory) — never feed negative yerr.
        los.append(max(0.0, r - lo))
        his.append(max(0.0, hi - r))
    ax.bar(xs, rates, width=0.6, color=C_BENIGN, zorder=3)
    ax.errorbar(xs, rates, yerr=[los, his], fmt="none", ecolor="black", capsize=3, zorder=4)
    ax.axhline(
        GATE_THRESHOLD, color=C_ACCENT, linestyle="--", linewidth=1.4, label="5% halt threshold"
    )
    ax.axhspan(0.0, 0.031, color=C_NEUTRAL, alpha=0.18, label="benign-cell prior band (0-3.1%)")
    for x, r in zip(xs, rates, strict=True):
        ax.annotate(
            f"{int(round(r * GATE_N))}/{GATE_N} misaligned",
            (x, r + 0.0015),
            ha="center",
            fontsize=9,
            zorder=6,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {k.rsplit('seed', 1)[-1]}" for k in sorted(per_cell)], fontsize=9)
    ax.set_ylabel("misaligned-and-coherent rate\n(800 samples per seed)")
    ax.set_title("Inverted installation gate: benign cells must stay under 5%")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    savefig_paper(fig, "inverted_gate_rates", dir=args.out_dir)
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def fig_split_half(args) -> None:
    summary = json.loads(Path(args.cross_arm_summary).read_text())
    sh_block = summary.get("split_half")
    if not sh_block:
        logger.warning(
            "[skip] cross-arm summary has no split_half block (--skip-split-half run); "
            "split_half_vs_cos figure NOT generated."
        )
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.subplots_adjust(bottom=0.14, top=0.86, left=0.12, right=0.97)
    for seed, alpha in zip(SEEDS, (0.5, 0.7, 0.9), strict=True):
        key = f"same_benign_seed{seed}"
        cell = _load_cell(args, "same", "benign", seed)
        sh = sh_block[key]["per_persona"]
        xs = [sh[p]["r_half"] for p in cell["persona_order"]]
        ys = cell["cos_to_U1"]
        ax.scatter(xs, ys, s=28, color=C_BENIGN, alpha=alpha, label=f"seed {seed}")
    ax.axvline(0.5, color=C_ACCENT, linestyle="--", linewidth=1.2, label="validity floor (r = 0.5)")
    ax.set_xlabel("per-persona split-half reliability (odd/even question halves)")
    ax.set_ylabel("|cos(shift, top direction)|")
    ax.set_title("Attenuation diagnostic: low reliability pulls cosines toward zero")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "split_half_vs_cos", dir=args.out_dir)
    plt.close(fig)


# ---------------------------------------------------------------- figure 7
def fig_fro_norm(args) -> None:
    cells = [(arm, s) for arm in ("marker", "em", "benign") for s in SEEDS]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.subplots_adjust(bottom=0.22, top=0.86, left=0.11, right=0.97)
    for x, (arm, seed) in enumerate(cells):
        d = _load_cell(args, "same", arm, seed)
        fro = float(np.sqrt(np.sum(np.asarray(d["singular_values"], dtype=np.float64) ** 2)))
        ax.bar(x, fro, width=0.62, color=ARM_COLOR[arm], alpha=0.85, zorder=3)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"{ARM_LABEL[arm]}\nseed {s}" for arm, s in cells], fontsize=7)
    ax.set_ylabel("shift-matrix size  $\\|M\\|_F$")
    ax.set_title("Update magnitude per cell (the magnitude-mediation check)")
    savefig_paper(fig, "fro_norm_per_cell", dir=args.out_dir)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#552 figures (off-pod).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benign-svd-dir", default="eval_results/issue_552/svd")
    parser.add_argument("--parent-svd-dir", default="eval_results/issue_521/svd")
    parser.add_argument(
        "--gate-summary", default="eval_results/issue_552/em_rate_gate_firstplot/summary.json"
    )
    parser.add_argument(
        "--cross-arm-summary", default="eval_results/issue_552/cross_arm/summary.json"
    )
    parser.add_argument("--out-dir", default="figures/issue_552")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    fig_hero(args)
    fig_benign_per_persona(args)
    fig_variant_robustness(args)
    fig_cross_arm_directions(args)
    fig_inverted_gate(args)
    fig_split_half(args)
    fig_fro_norm(args)
    logger.info("[phase=done] figures written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
