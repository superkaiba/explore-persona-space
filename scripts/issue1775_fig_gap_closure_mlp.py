"""Regenerate the gap-closure hero WITH the stitch-MLP band (supersedes hero_gap_closure.png).

The P5 driver rendered ``hero_gap_closure.png`` before the P3 stitch-MLP
re-run landed, so the band its own title names is missing. This analyzer
regen merges ``delta_beyond_analysis.json`` (the post-hoc re-reduction)
into the committed ``bilinear_fits.json`` prefix scheme and renders the
same curve + the stitch-MLP CI band via ``savefig_paper`` (PNG + PDF +
sidecar with embedded points/text) under the NEW name
``hero_gap_closure_with_mlp``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: .env + shared-VM thread caps bind BEFORE the heavy imports
# (tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints).
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

BIL = WT / "eval_results" / "issue_1775" / "bilinear"


def main() -> None:
    fits = json.loads((BIL / "bilinear_fits.json").read_text())
    beyond = json.loads((BIL / "delta_beyond_analysis.json").read_text())
    sch = fits["schemes"]["prefix"]
    dbm = beyond["schemes"]["prefix"]["delta_beyond_mlp_minus_bilinear"]
    r2_mlp = beyond["schemes"]["prefix"]["r2_stitch_mlp_seed_mean"]
    r2_bil = beyond["schemes"]["prefix"]["r2_bilinear_r_star"]
    curve = sch["outer_r2_curve_EXPLORATORY"]
    gap = sch["interaction_gap_fraction"]

    set_paper_style("blog")
    fig, ax = plt.subplots()
    rs = sorted(int(r) for r in curve)
    ax.plot(
        [max(r, 0.5) for r in rs],
        [curve[str(r)] for r in rs],
        marker="o",
        label="additive stitch + rank-r bilinear (outer test, exploratory curve)",
    )
    ax.axhline(gap["r2_stitch_ridge"], ls=":", color="gray", label="additive stitch ridge")
    ax.axhline(gap["r2_context_ridge"], ls="--", color="black", label="full-context ridge")
    ax.axhspan(
        r2_bil + dbm["ci95_cluster"][0],
        r2_bil + dbm["ci95_cluster"][1],
        alpha=0.15,
        color="red",
        label="stitch-MLP (95% cluster CI band)",
    )
    ax.axhline(r2_mlp, ls="-.", color="red", lw=0.8)
    rstar = sch["r_star_inner_val"]
    ax.axvline(rstar, color="green", ls=":", lw=0.8, label=f"r* = {rstar} (inner validation)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("interaction rank r (log2 scale; r=0 plotted at 0.5)")
    ax.set_ylabel("held-out R2 (48 answer PCs, novel-prefix folds)")
    ax.legend(fontsize=7)
    ax.set_title("Gap closure: additive stitch to rank-r bilinear vs stitch-MLP vs full context")
    savefig_paper(fig, "issue_1775/hero_gap_closure_with_mlp", dir=str(WT / "figures"))
    plt.close(fig)
    print("wrote figures/issue_1775/hero_gap_closure_with_mlp.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
