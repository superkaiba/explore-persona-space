#!/usr/bin/env python3
"""Two-panel FAIR prefix-vs-context comparison figure for #1092.

Panel A — held-out R2 at each grain (primary cell, ambient target):
  x = {averaged grain (per-prefix profiles), single-context grain};
  bars = {prefix map, context map}. Shows the prefix map recovering more of the
  between-prefix profile at averaged grain, and the grain the historical ~0.8
  prefix number lived at.

Panel B — fraction of the achievable ceiling at single-context grain
  (primary cell, per target basis):
  x = {ambient, pca48}; bars = {prefix R2 / between-prefix variance share,
  context R2 / achievable ceiling}. Normalizes each arm's per-row R2 by what is
  STRUCTURALLY reachable: the prefix arm can only explain the between-prefix
  variance share (v_P is near-constant within a prefix); the context arm's
  reachable ceiling is the MLP companion (pca48) / the additive (1-interaction)
  ceiling (ambient). Raw ceiling values live in the .meta.json, not on the bars.
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
FIGPATH = FIGDIR / "fair_comparison_prefix_vs_context.png"
METAPATH = FIGDIR / "fair_comparison_prefix_vs_context.meta.json"
PRIMARY = "cell_inst_own"
CELL_LABEL = {
    "cell_inst_own": "instruct model, own answers",
    "cell_pre_own": "pretrained model, own answers",
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    d = json.loads(DATA.read_text())
    cell = d["cells"][PRIMARY]
    amb = cell["bases"]["ambient"]
    pca = cell["bases"]["pca48"]

    prefix_c = paper_palette_role("baseline")  # Wong orange
    context_c = paper_palette_role("primary")  # Wong blue

    set_paper_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")

    # ---- Panel A: R2 at each grain (ambient target) ----
    grains = ["averaged\n(per-prefix profiles)", "single-context\n(per row)"]
    prefix_vals = [
        amb["averaged_grain"]["r2_prefix_averaged"],
        amb["single_grain"]["r2_prefix_battery_excluded_full"],
    ]
    context_vals = [
        amb["averaged_grain"]["r2_context_averaged"],
        amb["single_grain"]["r2_context_battery_excluded_full"],
    ]
    x = np.arange(len(grains))
    w = 0.38
    axA.bar(x - w / 2, prefix_vals, w, label="prefix map", color=prefix_c)
    axA.bar(x + w / 2, context_vals, w, label="context map", color=context_c)
    axA.set_xticks(x)
    axA.set_xticklabels(grains)
    axA.set_ylabel("held-out $R^2$")
    axA.set_ylim(0, 1.0)
    axA.set_title("A. Prediction skill by grain (ambient target)")
    axA.legend(frameon=False, loc="upper left")
    axA.grid(axis="y", alpha=0.3)

    # ---- Panel B: fraction of achievable ceiling, single grain, per basis ----
    bases = ["ambient", "pca48"]
    prefix_frac = [
        amb["fraction_of_ceiling_single_grain_full"]["prefix"],
        pca["fraction_of_ceiling_single_grain_full"]["prefix"],
    ]
    # context ceiling per basis: ambient -> additive (1-interaction); pca48 -> MLP companion
    context_frac = [
        amb["fraction_of_ceiling_single_grain_full"]["context_vs_additive"],
        pca["fraction_of_ceiling_single_grain_full"]["context_vs_mlp"],
    ]
    xb = np.arange(len(bases))
    axB.bar(xb - w / 2, prefix_frac, w, label="prefix map / between-prefix share", color=prefix_c)
    axB.bar(xb + w / 2, context_frac, w, label="context map / achievable ceiling", color=context_c)
    axB.set_xticks(xb)
    axB.set_xticklabels(["ambient", "pca48"])
    axB.set_ylabel("fraction of achievable ceiling")
    axB.set_ylim(0, 1.15)
    axB.axhline(1.0, color="0.4", lw=0.8, ls="--")
    axB.set_title("B. Fraction of achievable ceiling (single-context grain)")
    axB.legend(frameon=False, loc="upper left")
    axB.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Fair prefix-vs-context comparison — {CELL_LABEL.get(PRIMARY, PRIMARY)}, layer 14, "
        "battery-excluded, novel-prefix 6-fold CV",
        fontsize=11,
    )
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGPATH, dpi=200)
    plt.close(fig)

    # ---- meta.json (raw ceilings + provenance; sources banked + newly computed) ----
    meta = {
        "figure": str(FIGPATH.relative_to(PROJECT_ROOT)),
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
        "primary_cell": PRIMARY,
        "caption": (
            "Fair prefix-vs-context prediction comparison on the matched #1092 corpus. "
            "Teacher-forced capture; own-policy greedy answers; battery-excluded ridge fits "
            "(stratum != trait_stratum AND not is_eval_only, n=17308); novel-prefix grouped "
            "6-fold CV; layer 14; pooled t1/t2/t3 targets. Panel A: held-out R2 at averaged "
            "grain (per-prefix query-averaged profiles) vs single-context grain (per row), "
            "ambient target. Panel B: single-context-grain R2 as a fraction of the achievable "
            "ceiling per target basis — prefix arm divided by the between-prefix variance share "
            "(its structural ceiling: v_P is near-constant within a prefix); context arm divided "
            "by the MLP companion (pca48, banked 0.929) / the additive 1-interaction ceiling "
            "(ambient, banked dense-core). Newly computed here: averaged-grain R2 (both arms), "
            "single-context battery-excluded refits, between-prefix shares on the fit population. "
            "Banked (reused): read1 battery-included per-row R2, read3 dense-core FGI shares, "
            "MLP companion. Bars are single pooled held-out R2 point estimates (no fold-spread "
            "error bars)."
        ),
        "panelA_values_ambient": {
            "r2_prefix_averaged": amb["averaged_grain"]["r2_prefix_averaged"],
            "r2_context_averaged": amb["averaged_grain"]["r2_context_averaged"],
            "r2_prefix_single_battery_excluded": amb["single_grain"][
                "r2_prefix_battery_excluded_full"
            ],
            "r2_context_single_battery_excluded": amb["single_grain"][
                "r2_context_battery_excluded_full"
            ],
        },
        "panelB_fractions_full_single_grain": {
            "ambient": {
                "prefix_over_share": amb["fraction_of_ceiling_single_grain_full"]["prefix"],
                "context_over_additive": amb["fraction_of_ceiling_single_grain_full"][
                    "context_vs_additive"
                ],
            },
            "pca48": {
                "prefix_over_share": pca["fraction_of_ceiling_single_grain_full"]["prefix"],
                "context_over_mlp": pca["fraction_of_ceiling_single_grain_full"]["context_vs_mlp"],
            },
        },
        "raw_ceilings": {
            "ambient": amb["ceilings"],
            "pca48": pca["ceilings"],
        },
        "banked_read1_battery_included": cell["banked_read1_battery_included"],
        "prefix_end_within_prefix_constancy_ambient_note": cell[
            "prefix_end_within_prefix_constancy"
        ],
        "averaged_grain_1_over_nq_note_ambient": amb["averaged_grain"]["note_1_over_nq"],
        "sources": {
            "newly_computed_json": (
                "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
            ),
            "banked_read1_read3_mlp": (
                "eval_results/issue_1092/p7/{read1_map_skill,read3_fgi_shares,mlp_companion}.json"
            ),
        },
    }
    if "cell_pre_own" in d["cells"]:
        pc = d["cells"]["cell_pre_own"]["bases"]
        meta["secondary_cell_pre_own"] = {
            "ambient": {
                "r2_prefix_averaged": pc["ambient"]["averaged_grain"]["r2_prefix_averaged"],
                "r2_context_averaged": pc["ambient"]["averaged_grain"]["r2_context_averaged"],
                "r2_prefix_single": pc["ambient"]["single_grain"][
                    "r2_prefix_battery_excluded_full"
                ],
                "r2_context_single": pc["ambient"]["single_grain"][
                    "r2_context_battery_excluded_full"
                ],
                "fraction_full": pc["ambient"]["fraction_of_ceiling_single_grain_full"],
            },
            "pca48": {
                "r2_prefix_averaged": pc["pca48"]["averaged_grain"]["r2_prefix_averaged"],
                "r2_context_averaged": pc["pca48"]["averaged_grain"]["r2_context_averaged"],
                "r2_prefix_single": pc["pca48"]["single_grain"]["r2_prefix_battery_excluded_full"],
                "r2_context_single": pc["pca48"]["single_grain"][
                    "r2_context_battery_excluded_full"
                ],
                "fraction_full": pc["pca48"]["fraction_of_ceiling_single_grain_full"],
            },
        }
    METAPATH.write_text(json.dumps(meta, indent=2, allow_nan=True))
    print(f"wrote {FIGPATH}")
    print(f"wrote {METAPATH}")


if __name__ == "__main__":
    main()
