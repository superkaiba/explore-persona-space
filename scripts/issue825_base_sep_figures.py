"""Issue #825 `base-separator-control` Phase C3 figures (plan v18 section 3).

Hero: grouped bars @ L19 — base vs instruct x {chat within, separator within
(rotated + MLP; raw ridge greyed, documented-pathological on control cells),
sep->chat transfer as fraction of the full-n chat ceiling}. Low-level per-unit
plot: per-article-group held-out R^2 (600 WikiText groups) base-vs-instruct
scatter for the separator cell. paper-plots conventions (the analyzer
regenerates figures later; this is the minimal committed pair).

CLI:
  uv run python scripts/issue825_base_sep_figures.py \
      [--out eval_results/issue_825/base-separator-control] [--fig-dir figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

COMMITTED_931 = SCRIPTS.parent / "eval_results" / "issue_931"
L = 19


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out", type=Path, default=Path("eval_results/issue_825/base-separator-control")
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures"))
    return ap.parse_args()


def _cell(path: Path) -> dict:
    d = json.loads(path.read_text())
    hl = int(d.get("headline_layer", L))
    return {
        "ridge": float(d["r2_per_layer_obs"][hl]),
        "rotated": float(d["random_projection_control_r2"][str(hl)]),
        "per_group": d.get("per_group_r2_headline", {}),
        "hl": hl,
    }


def _mlp(path: Path, hl: int) -> float:
    d = json.loads(path.read_text())
    return float(d["cells"]["armC_sep"][str(hl)]["r2_obs"])


def main() -> int:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()

    base_sep = _cell(args.out / "cells_armC_sep.json")
    inst_sep = _cell(COMMITTED_931 / "cells_armC_sep.json")
    base_mlp = _mlp(args.out / "mlp_secondary.json", base_sep["hl"])
    inst_mlp = _mlp(COMMITTED_931 / "mlp_secondary.json", inst_sep["hl"])
    transfer = json.loads((args.out / "base_sep_to_chat.json").read_text())
    base_chat = float(transfer["sep_to_chat"]["fulln_ceiling_r2"])  # committed S2 @ L19
    inst_chat = float(
        json.loads((COMMITTED_931 / "cells_chat_ref.json").read_text())["r2_per_layer_obs"][L]
    )
    base_frac = float(transfer["sep_to_chat"]["fraction_of_fulln_ceiling"])
    inst_frac = float(
        json.loads((COMMITTED_931 / "sep_to_chat_control.json").read_text())["sep_to_chat"][
            "fraction_of_fulln_ceiling"
        ]
    )

    # ---- Hero: grouped bars @ L19 ------------------------------------------
    groups = [
        ("Chat within\n$R^2$", base_chat, inst_chat, False),
        ("Sep within\n(rotated)", base_sep["rotated"], inst_sep["rotated"], False),
        ("Sep within\n(MLP)", base_mlp, inst_mlp, False),
        ("Sep within\n(raw ridge)", base_sep["ridge"], inst_sep["ridge"], True),
        ("Sep$\\to$chat transfer\n(frac. full-n chat ceiling)", base_frac, inst_frac, False),
    ]
    c_base = paper_palette_role("primary")
    c_inst = paper_palette_role("baseline")
    c_grey = paper_palette_role("neutral")
    fig, ax = plt.subplots(figsize=(7.6, 4.2), layout="constrained")
    x = np.arange(len(groups))
    w = 0.36
    ymin = -1.0  # clip the pathological ridge bars; annotate true values
    for i, (_label, vb, vi, pathological) in enumerate(groups):
        cb = c_grey if pathological else c_base
        ci = c_grey if pathological else c_inst
        ab, ai = 1.0 if not pathological else 0.55, 1.0 if not pathological else 0.55
        ax.bar(i - w / 2, max(vb, ymin), w, color=cb, alpha=ab, edgecolor="white", linewidth=0.4)
        ax.bar(i + w / 2, max(vi, ymin), w, color=ci, alpha=ai, edgecolor="white", linewidth=0.4)
        for xi, v in ((i - w / 2, vb), (i + w / 2, vi)):
            if v < ymin:
                ax.annotate(
                    f"{v:.2f}", (xi, ymin), ha="center", va="bottom", fontsize=6, rotation=90
                )
    ax.axhline(0.0, color="#999999", lw=0.8, zorder=0)
    ax.axhline(0.5, color="#bbbbbb", lw=0.8, ls="--", zorder=0)
    ax.set_ylim(ymin, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("Held-out $R^2$ / fraction (layer 19)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c_base, label="pretrained (base)"),
        plt.Rectangle((0, 0), 1, 1, color=c_inst, label="instruct"),
        plt.Rectangle((0, 0), 1, 1, color=c_grey, alpha=0.55, label="raw ridge (pathological)"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)
    ax.set_title(
        "Separator control: base vs instruct (y clipped at -1.0; 0.5 = transfer threshold)"
    )
    savefig_paper(fig, "issue_825/base_sep_control_hero", dir=args.fig_dir)
    plt.close(fig)

    # ---- Low-level: per-article-group R^2 scatter (base vs instruct) -------
    common_groups = sorted(set(base_sep["per_group"]) & set(inst_sep["per_group"]))
    assert common_groups, "no shared per-group R^2 entries between base and instruct sep cells"
    xb = np.array([base_sep["per_group"][g] for g in common_groups])
    xi = np.array([inst_sep["per_group"][g] for g in common_groups])
    fig, ax = plt.subplots(figsize=(4.8, 4.6), layout="constrained")
    ax.scatter(xi, xb, s=9, alpha=0.4, color=c_base, edgecolors="none")
    lo = float(min(xi.min(), xb.min()))
    hi = float(max(xi.max(), xb.max()))
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("instruct per-group held-out $R^2$ (separator cell, raw ridge, layer 19)")
    ax.set_ylabel("pretrained per-group held-out $R^2$ (raw ridge)")
    ax.set_title(f"Per-article-group $R^2$ ({len(common_groups)} WikiText groups)")
    savefig_paper(fig, "issue_825/base_sep_pergroup_scatter", dir=args.fig_dir)
    plt.close(fig)
    print(f"[i825-bs-c3] wrote hero + per-group scatter under {args.fig_dir}/issue_825/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
