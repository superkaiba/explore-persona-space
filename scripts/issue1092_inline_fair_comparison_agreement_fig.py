#!/usr/bin/env python3
"""Prediction-agreement figure for #1092 (addendum): the direct test of
"the prefix map IS the query-averaged context map" at prediction level.

Per held-out prefix (averaged grain, primary cell cell_inst_own), scatter the
prefix map's profile error |prefix-pred - truth| against the query-averaged
context map's error |ctx-avg-pred - truth| (L2 per prefix), one subplot per
target basis, with the y=x reference. Points sit BELOW y=x (context error
smaller) with a strong error correlation (same prefixes hard for both) but the
prefix error runs ~2-3x larger — the two maps capture aligned-but-different
prefix-borne structure, they are NOT the same map. Agreement R2 + cosine live
in the subtitle/JSON, not as on-plot overlays.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    set_paper_style,
)

DATA = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
FIGDIR = PROJECT_ROOT / "figures/issue_1092"
FIGPATH = FIGDIR / "fair_comparison_prediction_agreement.png"
METAPATH = FIGDIR / "fair_comparison_prediction_agreement.meta.json"
PRIMARY = "cell_inst_own"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    d = json.loads(DATA.read_text())
    bases = ["ambient", "pca48"]
    pt_color = paper_palette_role("primary")

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), layout="constrained")
    for ax, basis in zip(axes, bases, strict=True):
        pa = d["cells"][PRIMARY]["bases"][basis]["prediction_agreement"]
        e_prefix = np.asarray(pa["per_prefix_err_prefix"])
        e_ctx = np.asarray(pa["per_prefix_err_ctx"])
        hi = float(max(e_prefix.max(), e_ctx.max())) * 1.03
        ax.scatter(e_prefix, e_ctx, s=8, alpha=0.35, color=pt_color, edgecolors="none")
        ax.plot([0, hi], [0, hi], color="0.4", lw=1.0, ls="--")  # y = x
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("prefix-map profile error  |pred - truth|  (L2)")
        ax.set_ylabel("context-avg profile error  |pred - truth|  (L2)")
        ax.set_title(
            f"{basis} target\n"
            f"agree $R^2$(pref|ctx)={pa['agreement_r2_prefixpred_vs_ctxpred']:.2f}, "
            f"cos$_{{centered}}$={pa['mean_cosine_centered']:.2f}, "
            f"err corr={pa['per_prefix_err_correlation']:.2f}",
            fontsize=10,
        )
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Prefix-map vs query-averaged-context-map predictions per held-out prefix "
        "- instruct model, own answers, layer 14\n"
        "(points below y=x: context error smaller; prefix error runs ~2-3x larger "
        "on the same hard prefixes)",
        fontsize=10.5,
    )
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGPATH, dpi=200)
    plt.close(fig)

    meta = {
        "figure": str(FIGPATH.relative_to(PROJECT_ROOT)),
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
        "primary_cell": PRIMARY,
        "caption": (
            "Direct prediction-agreement test (#1092 addendum) at averaged grain: does the "
            "prefix map reproduce the query-averaged context map? Per held-out prefix "
            "(novel-prefix 6-fold CV, battery-excluded), the prefix map's held-out predicted "
            "profile and the query-averaged context-map held-out prediction are each compared "
            "to the true per-prefix profile; the scatter plots the two L2 error magnitudes (one "
            "point per prefix, primary cell). agreement R2(prefix-pred | context-pred) 0.28-0.54 "
            "across cells/bases sits far BELOW each context arm's vs-truth averaged R2 "
            "(0.76-0.94), so predictions are NOT near-identical and the prefix map is not the "
            "query-averaged context map. Centered cosine 0.64-0.74 (directionally aligned), "
            "per-prefix error ratio ~2-3x (prefix worse), error correlation 0.56-0.90 (same "
            "prefixes hard for both). Teacher-forced capture; own-policy greedy answers. All "
            "values newly computed from the persisted per-row context held-out predictions + "
            "the averaged-grain prefix fit."
        ),
        "metrics": {
            cell: {
                basis: {
                    k: d["cells"][cell]["bases"][basis]["prediction_agreement"][k]
                    for k in (
                        "n_prefixes",
                        "agreement_r2_prefixpred_vs_ctxpred",
                        "agreement_r2_ctxpred_vs_prefixpred",
                        "mean_cosine_raw",
                        "mean_cosine_centered",
                        "mean_err_prefix",
                        "mean_err_ctx",
                        "err_ratio_prefix_over_ctx",
                        "per_prefix_err_correlation",
                    )
                }
                for basis in bases
            }
            for cell in d["cells"]
        },
        "vs_truth_averaged_r2_for_reference": {
            cell: {
                basis: {
                    "prefix_averaged": d["cells"][cell]["bases"][basis]["averaged_grain"][
                        "r2_prefix_averaged"
                    ],
                    "context_averaged": d["cells"][cell]["bases"][basis]["averaged_grain"][
                        "r2_context_averaged"
                    ],
                }
                for basis in bases
            }
            for cell in d["cells"]
        },
        "source": "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json",
    }
    METAPATH.write_text(json.dumps(meta, indent=2, allow_nan=True))
    print(f"wrote {FIGPATH}")
    print(f"wrote {METAPATH}")


if __name__ == "__main__":
    main()
