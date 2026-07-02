#!/usr/bin/env python
"""Issue #778 v2 — extra figures beyond the ladder's band plots (plan v8 §6).

  1. v1-vs-v2 direction comparison: per-layer cos(r_B v2, r_B v1) per trait +
     per-cell observed-|r| delta at the pre-registered paper-steering layer
     (the extraction-fidelity impact read).
  2. Lambda-sensitivity strip: per (cell, cov family) p97.5 at the paper layer
     across the registered shrinkage sweep {0.05, 0.1, 0.2}.
  3. Pair-yield / drop-count table figure (K1 telemetry per trait).

Reads ONLY committed/staged v2 artifacts (v2 meta JSONs, pairing JSONs, the v2
honest-nulls JSONs) — no recompute. The band/violin ladder plots themselves are
produced by ``issue778_honest_null_ladder.py`` (build_fixed_figures /
build_figures parameterized for v2).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis import paper_plots as pp  # noqa: E402

V2_LABEL = "faithful-extraction-honest-nulls-rerun"
COV_FAMILIES = ("within_class", "neg_arm_only", "neutral_cov", "rb_projected_out")


def fig_v1_v2_comparison(out_root: Path, eval_root: Path, fig_root: Path, traits) -> list[str]:
    """Per-layer cos(v2, v1) curves + per-cell paper-layer observed-|r| deltas."""
    written = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    palette = pp.paper_palette(max(3, len(traits)))
    for ti, trait in enumerate(traits):
        meta_path = out_root / "v2" / "extract" / f"{trait}_v2_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        cos = meta.get("cos_v2_v1_per_layer")
        if cos:
            ax1.plot(range(len(cos)), cos, label=trait, color=palette[ti], lw=1.8)
    ax1.set_xlabel("r_B layer index (0-indexed block outputs)")
    ax1.set_ylabel("cos(r_B v2, r_B v1)")
    ax1.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax1.legend()
    ax1.set_title("Direction stability under the faithful recipe")
    # per-cell observed delta at the paper layer
    labels, deltas, colors = [], [], []
    for path in sorted(glob.glob(str(eval_root / V2_LABEL / "*_honestnulls_v2.json"))):
        with open(path) as f:
            fd = json.load(f)
        if "status" in fd:
            continue
        trait = fd["trait"]
        for regime, rd in fd.get("stage_fixed", {}).items():
            pc = rd["per_choice"]["paper_steering"]
            labels.append(f"{trait[:4]}:{fd['setting'].replace('monitoring_', 'mon_')}:{regime}")
            deltas.append(pc["v1_to_v2_observed_delta"])
            colors.append(palette[list(traits).index(trait) if trait in traits else 0])
    y = np.arange(len(labels))
    ax2.barh(y, deltas, color=colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=6)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("observed |r| (v2 direction) - |r| (v1 direction) @ paper layer")
    ax2.set_title("Extraction-fidelity impact per cell")
    fig.tight_layout()
    pp.savefig_paper(fig, "v1_vs_v2_direction_impact", dir=str(fig_root), formats=("png", "pdf"))
    plt.close(fig)
    written.append(str(fig_root / "v1_vs_v2_direction_impact.png"))
    return written


def fig_lambda_strip(eval_root: Path, fig_root: Path) -> list[str]:
    """p97.5 at the paper layer per (cell, cov family) across the lambda sweep."""
    rows = []  # (cell_label, fam, lam, p97_5)
    for path in sorted(glob.glob(str(eval_root / V2_LABEL / "*_honestnulls_v2.json"))):
        with open(path) as f:
            fd = json.load(f)
        if "status" in fd:
            continue
        lam_primary = str(fd.get("lambda_primary", 0.1))
        for regime, rd in fd.get("stage_fixed", {}).items():
            cell = f"{fd['trait'][:4]}:{fd['setting'].replace('monitoring_', 'mon_')}:{regime}"
            pc = rd["per_choice"]["paper_steering"]
            for fam in COV_FAMILIES:
                if fam in pc["nulls"] and pc["nulls"][fam].get("p97_5") is not None:
                    rows.append((cell, fam, lam_primary, pc["nulls"][fam]["p97_5"]))
            sweep = rd.get("lambda_sweep_at_paper_layer") or {}
            for lam, fams in sweep.items():
                for fam in COV_FAMILIES:
                    if fam in fams and fams[fam].get("p97_5") is not None:
                        rows.append((cell, fam, lam, fams[fam]["p97_5"]))
    if not rows:
        return []
    fams = sorted({r[1] for r in rows})
    lams = sorted({r[2] for r in rows}, key=float)
    palette = pp.paper_palette(len(lams))
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4.2), sharey=True)
    if len(fams) == 1:
        axes = [axes]
    for ax, fam in zip(axes, fams, strict=True):
        cells = sorted({r[0] for r in rows if r[1] == fam})
        y = np.arange(len(cells))
        for li, lam in enumerate(lams):
            vals = []
            for c in cells:
                v = [r[3] for r in rows if r[0] == c and r[1] == fam and r[2] == lam]
                vals.append(v[0] if v else np.nan)
            ax.plot(vals, y, "o", ms=4, color=palette[li], label=f"λ={lam}")
        ax.set_yticks(y)
        ax.set_yticklabels(cells, fontsize=6)
        ax.set_title(fam, fontsize=9)
        ax.set_xlabel("null p97.5 @ paper layer")
    axes[0].legend(fontsize=7)
    fig.suptitle("Shrinkage-λ sensitivity of the covariance-null caps")
    fig.tight_layout()
    pp.savefig_paper(fig, "lambda_sensitivity_strip", dir=str(fig_root), formats=("png", "pdf"))
    plt.close(fig)
    return [str(fig_root / "lambda_sensitivity_strip.png")]


def fig_yield_table(out_root: Path, fig_root: Path, traits) -> list[str]:
    """K1 pair-yield + per-arm/per-dim drop counts as a table figure."""
    cols = [
        "trait",
        "pairs total",
        "kept pairs",
        "K1 status",
        "drop pos_trait",
        "drop neg_trait",
        "drop pos_coh",
        "drop neg_coh",
    ]
    body = []
    for trait in traits:
        p = out_root / "v2" / "pairing" / f"{trait}_pairing.json"
        if not p.exists():
            body.append([trait, "—", "—", "missing", "—", "—", "—", "—"])
            continue
        with open(p) as f:
            pr = json.load(f)
        d = pr["dropped_unevaluable_by_arm_dim"]
        body.append(
            [
                trait,
                pr["n_pairs_total"],
                pr["n_kept_pairs"],
                pr["k1_status"],
                d["pos_trait"],
                d["neg_trait"],
                d["pos_coherence"],
                d["neg_coherence"],
            ]
        )
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.4 * len(body)))
    ax.axis("off")
    tbl = ax.table(cellText=[[str(x) for x in row] for row in body], colLabels=cols, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    ax.set_title("v2 paired-mask yield (K1) + drop-never-coerce telemetry", fontsize=10)
    fig.tight_layout()
    pp.savefig_paper(fig, "pair_yield_table", dir=str(fig_root), formats=("png", "pdf"))
    plt.close(fig)
    return [str(fig_root / "pair_yield_table.png")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #778 v2 extra figures.")
    ap.add_argument("--out-root", default="data/issue_778")
    ap.add_argument("--eval-results-root", default="eval_results/issue_778")
    ap.add_argument("--figures-root", default=f"figures/issue_778/{V2_LABEL}")
    ap.add_argument("--traits", nargs="+", default=["evil", "sycophancy", "hallucination"])
    args = ap.parse_args()
    out_root = Path(args.out_root)
    eval_root = Path(args.eval_results_root)
    fig_root = Path(args.figures_root)
    fig_root.mkdir(parents=True, exist_ok=True)
    pp.set_paper_style()
    written = []
    written += fig_v1_v2_comparison(out_root, eval_root, fig_root, args.traits)
    written += fig_lambda_strip(eval_root, fig_root)
    written += fig_yield_table(out_root, fig_root, args.traits)
    print(json.dumps({"phase": "v2_extra_figures", "figures": written}, indent=2))


if __name__ == "__main__":
    main()
