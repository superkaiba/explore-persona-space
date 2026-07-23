"""Averaged-only variant of the #1092 prefixend_monitoring figure (plot-only).

Rebuilds `figures/issue_1092/prefixend_monitoring.png` from the banked
`inline_prefixend_monitoring/results.json` WITHOUT the per-prompt (single
context) bar, per the direct-vs-averaged writeup's averaged-comparisons-only
scope. Colors match the original figure (averaged = primary, prefix-end =
accent) so the two figures stay cross-readable. No fits, no GPU — pure
matplotlib over the banked JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "eval_results/issue_1092/inline_prefixend_monitoring/results.json"
FIG_DIR = PROJECT_ROOT / "figures/issue_1092"
CELL = "cell_inst_own"
LAYER = "14"


def main() -> int:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    results = json.loads(SRC.read_text())
    cells_present = results["cells"][CELL][LAYER]
    traits = [d["trait"] for d in cells_present]

    pp.set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), layout="constrained")

    # Panel A: the two averaged-target reads per trait + reliability ceiling.
    axA = axes[0]
    labels = ["prefix-averaged\ncontext", "prefix-end\n(pre-query)"]
    colors = [pp.paper_palette_role("primary"), pp.paper_palette_role("accent")]
    keys = ["averaged_context", "prefix_end"]
    x = np.arange(len(traits))
    w = 0.32
    for k, (key, lab, col) in enumerate(zip(keys, labels, colors, strict=True)):
        vals = [d["reads"][key]["r"] for d in cells_present]
        los = [max(0.0, d["reads"][key]["r"] - d["reads"][key]["ci95"][0]) for d in cells_present]
        his = [max(0.0, d["reads"][key]["ci95"][1] - d["reads"][key]["r"]) for d in cells_present]
        axA.bar(x + (k - 0.5) * w, vals, w, label=lab, color=col)
        axA.errorbar(
            x + (k - 0.5) * w, vals, yerr=[los, his], fmt="none", ecolor="#333", capsize=3, lw=1.2
        )
    for i, d in enumerate(cells_present):
        c = d["monitoring_r_ceiling_from_reliability"]
        axA.plot(
            [x[i] - 1.1 * w, x[i] + 1.1 * w],
            [c, c],
            ls="--",
            color="#888",
            lw=1.4,
            label="reliability ceiling" if i == 0 else None,
        )
    axA.set_xticks(x)
    axA.set_xticklabels([t.capitalize() for t in traits])
    axA.set_ylabel("Monitoring correlation r (with judge score)")
    axA.set_title("Trait monitoring: prefix-end vs averaged-context")
    axA.axhline(0, color="#bbb", lw=0.8)
    axA.legend(loc="upper center", fontsize=9)

    # Panel B: N-averaging curve (context) with prefix-end flat reference.
    axB = axes[1]
    tcolors = pp.paper_palette_blog(len(traits))
    ns_ref: list[int] = []
    for d, tc in zip(cells_present, tcolors, strict=True):
        cv = d["averaging_curve_context"]
        ns = [c["N"] for c in cv]
        ns_ref = ns
        rm = np.array([c["r_mean"] for c in cv])
        sd = np.array([c["r_sd"] for c in cv])
        axB.plot(
            ns, rm, "-o", color=tc, ms=4, label=f"{d['trait'].capitalize()} — averaged context"
        )
        axB.fill_between(ns, rm - sd, rm + sd, color=tc, alpha=0.18)
        axB.axhline(
            d["prefix_end_flat_reference_r"],
            ls="--",
            color=tc,
            lw=1.4,
            label=f"{d['trait'].capitalize()} — prefix-end (query-invariant)",
        )
    axB.set_xlabel("Number of questions averaged per prefix (N)")
    axB.set_ylabel("Monitoring correlation r")
    axB.set_title("Averaging context questions vs the single prefix-end read")
    axB.set_xscale("log")
    axB.set_xticks(ns_ref)
    axB.set_xticklabels([str(n) for n in ns_ref])
    axB.minorticks_off()
    axB.legend(loc="lower right", fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pp.savefig_paper(fig, "prefixend_monitoring_averaged_only", dir=FIG_DIR)
    plt.close(fig)
    print("figure written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
