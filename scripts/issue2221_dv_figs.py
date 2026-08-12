"""Issue #2221 — the three DV-side figures the P8 monitor plots do not cover.

P8 rendered the monitor arms (selected r, r-by-layer, checkpoint AUC). The
headline results need three more, all 0 GPU-h and VM-side:

1. ``trait_mix_size_vs_acquisition`` — realized training rows per cell (log x)
   against the paper-panel graded trait mean (y), one LABELED point per
   fine-tune. This is simultaneously the yield-collapse evidence and the
   size-confound evidence: the only cells crossing the positive threshold are
   the two largest families.
2. ``trait_scores_by_cell`` — the per-unit DV view: paper-panel graded mean for
   all 24 cells grouped by family, with the base-model level and the
   label-positive threshold drawn.
3. ``h2_delta_forest`` — the PRE-REGISTERED headline contrast: per-draw-selected
   Delta = mapped arm c minus paper arm a, point + bootstrap CI, per trait x
   panel, against the achievable Delta-r range.
4. ``monitor_vs_score_per_unit`` — the low-level 24-point view behind the
   hallucination correlation aggregates: monitor scalar (paper 20-q panel, at
   each arm's selected layer) vs paper-panel graded hallucination score, one
   labeled point per fine-tune, two panels (paper's last-prompt-token read;
   mapped context read). Added post-P8 (``--only scatter`` renders it alone
   without touching the three committed figures above).

Row counts are read from the realized mixes on disk, which were verified
byte-identical to the HF copies at ``issue2221_realtwin/train/`` (the record of
what P4 actually trained).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

FAMILIES = [
    "evil",
    "sycophancy",
    "hallucination",
    "insecure_code",
    "mistake_medical",
    "mistake_math",
    "mistake_gsm8k",
    "mistake_opinions",
]
VERSIONS = ["normal", "misaligned_1", "misaligned_2"]
# Which trait DV each family's acquisition is read on (the 3 judged traits).
TRAIT_OF_FAMILY = {"evil": "evil", "sycophancy": "sycophancy"}
POSITIVE_MIN = 50.0


def mix_rows(dataset_root: Path) -> dict[str, int]:
    """Realized row count per family (equalized within family) -> {family: rows}."""
    out: dict[str, int] = {}
    for fam in FAMILIES:
        f = dataset_root / fam / "normal.jsonl"
        if not f.is_file():
            raise FileNotFoundError(f"mix missing: {f}")
        with f.open() as fh:
            out[fam] = sum(1 for line in fh if line.strip())
    return out


def paper_mean(scores: dict, cell: str, trait: str) -> float:
    """Paper-panel graded mean for (cell, trait) — the registered primary DV."""
    return float(scores[cell][trait]["per_panel"]["paper"]["graded_mean"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-results-root", default="eval_results/issue_2221")
    ap.add_argument("--dataset-root", default="data/issue_2221/dataset")
    ap.add_argument("--figures-root", default="figures/issue_2221")
    ap.add_argument(
        "--only",
        default="all",
        choices=["all", "scatter"],
        help="'scatter' renders only the per-unit monitor-vs-score figure (fig 4)",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    eval_root = Path(args.eval_results_root)
    blob = json.loads((eval_root / "trait_scores.json").read_text())
    scores = blob["scores"]
    corr = json.loads((eval_root / "correlations.json").read_text())
    rows = mix_rows(Path(args.dataset_root))
    fig_dir = Path(args.figures_root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pp.set_paper_style()
    fam_colors = {f: c for f, c in zip(FAMILIES, pp.paper_palette(len(FAMILIES)))}

    # ── 4) per-unit 24-point scatter behind the hallucination correlations ───
    # Monitor scalar (paper 20-q capture panel, arm's selected layer) vs the
    # paper-panel graded hallucination score; the six zero-step cells sit at
    # exactly x = 0 (their context shift is identically zero at every layer).
    ms = json.loads((eval_root / "monitor_scalars" / "hallucination_paper.json").read_text())[
        "scalars"
    ]
    arms_meta = corr["per_trait"]["hallucination"]["panels"]["paper"]["arms"]
    panels = [
        ("a_rb_ctx", "paper's last-prompt-token read"),
        ("c_map_ctx", "mapped context read"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, (arm, arm_label) in zip(axes, panels):
        layer = int(arms_meta[arm]["selected_layer"])
        r_sel = float(arms_meta[arm]["selected_r"])
        for fam in FAMILIES:
            xs, ys, labs = [], [], []
            for v in VERSIONS:
                cell = f"{fam}_{v}"
                if cell not in ms:
                    continue
                xs.append(float(ms[cell][arm][layer]))
                ys.append(paper_mean(scores, cell, "hallucination"))
                labs.append(
                    f"{fam[:9]}/{v.replace('misaligned_', 'II' if v.endswith('2') else 'I')}"
                )
            ax.scatter(xs, ys, s=40, color=fam_colors[fam], zorder=3)
            for x, y, lab in zip(xs, ys, labs):
                ax.text(x, y + 1.2, lab, fontsize=4.6, ha="center", color="0.25")
        ax.axhline(POSITIVE_MIN, color="black", lw=0.9, ls="--")
        ax.set_xlabel(f"{arm_label}\n(monitor scalar, layer {layer + 1} of 28)")
        ax.annotate(
            f"Spearman r = {r_sel:.3f}",
            xy=(0.03, 0.93),
            xycoords="axes fraction",
            fontsize=8,
        )
    axes[0].set_ylabel("paper-panel graded hallucination score (0-100)")
    pp.savefig_paper(fig, "monitor_vs_score_per_unit", dir=fig_dir)
    plt.close(fig)
    if args.only == "scatter":
        print("[dv-figs] wrote monitor_vs_score_per_unit only", flush=True)
        return

    # ── 1) mix size vs acquisition, one labeled point per fine-tune ──────────
    # The trait a family's own acquisition is read on: evil/sycophancy on their
    # namesake; every other family on hallucination (the trait whose DV spans
    # the threshold), matching how the 5 positives were identified.
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for fam in FAMILIES:
        trait = TRAIT_OF_FAMILY.get(fam, "hallucination")
        xs, ys = [], []
        for v in VERSIONS:
            cell = f"{fam}_{v}"
            if cell not in scores:
                continue
            xs.append(rows[fam])
            ys.append(paper_mean(scores, cell, trait))
        if not xs:
            continue
        ax.scatter(xs, ys, s=54, color=fam_colors[fam], label=f"{fam} ({trait})", zorder=3)
        ax.annotate(
            f"{fam}\n{rows[fam]} rows",
            (xs[0], max(ys)),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=6,
        )
    ax.axhline(
        POSITIVE_MIN,
        color="black",
        lw=1.0,
        ls="--",
        label=f"label-positive threshold ({POSITIVE_MIN:.0f})",
    )
    ax.set_xscale("log")
    ax.set_xlabel("realized training rows per cell (log scale)")
    ax.set_ylabel("paper-panel graded trait score (0-100)")
    ax.set_ylim(-3, 100)
    ax.legend(fontsize=6, loc="upper left")
    pp.savefig_paper(fig, "trait_mix_size_vs_acquisition", dir=fig_dir)
    plt.close(fig)

    # ── 2) per-cell DV, the low-level per-unit view ──────────────────────────
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xt, xl = [], []
    pos = 0
    for fam in FAMILIES:
        trait = TRAIT_OF_FAMILY.get(fam, "hallucination")
        for v in VERSIONS:
            cell = f"{fam}_{v}"
            if cell not in scores:
                continue
            ax.bar(pos, paper_mean(scores, cell, trait), 0.82, color=fam_colors[fam])
            xt.append(pos)
            xl.append(f"{fam[:11]}/{v.replace('misaligned_', 'II' if v.endswith('2') else 'I')}")
            pos += 1
        pos += 0.6
    base_h = paper_mean(scores, "base", "hallucination")
    ax.axhline(
        base_h, color="0.35", lw=1.0, ls="-.", label=f"base model, hallucination ({base_h:.1f})"
    )
    ax.axhline(POSITIVE_MIN, color="black", lw=1.0, ls="--", label="label-positive threshold (50)")
    ax.set_xticks(xt, xl, rotation=90, fontsize=5.5)
    ax.set_ylabel("paper-panel graded trait score (0-100)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6)
    pp.savefig_paper(fig, "trait_scores_by_cell", dir=fig_dir)
    plt.close(fig)

    # ── 3) the pre-registered H2 contrast, forest ────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels, pts, los, his = [], [], [], []
    for trait in corr["per_trait"]:
        for panel in ("paper", "pooled"):
            h2 = corr["per_trait"][trait]["panels"][panel].get("h2_delta_c_minus_a")
            if not h2:
                continue
            ci = h2.get("bootstrap_ci") or [np.nan, np.nan]
            labels.append(f"{trait} / {panel}")
            pts.append(float(h2["point"]))
            los.append(float(ci[0]))
            his.append(float(ci[1]))
    y = np.arange(len(labels))
    pts_a, los_a, his_a = np.asarray(pts), np.asarray(los), np.asarray(his)
    ax.hlines(y, los_a, his_a, color="0.35", lw=2.0)
    ax.scatter(pts_a, y, s=46, color=pp.paper_palette(1)[0], zorder=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(-2.05, 2.05)
    ax.set_xlabel(r"$\Delta$ Spearman r (mapped arm c $-$ paper arm a); achievable range [-2, +2]")
    ax.invert_yaxis()
    pp.savefig_paper(fig, "h2_delta_forest", dir=fig_dir)
    plt.close(fig)

    print(json.dumps({"mix_rows": rows, "n_h2_rows": len(labels)}, indent=1), flush=True)
    print(f"[dv-figs] wrote 3 figures -> {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
