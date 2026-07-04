"""Promotion-pass figures for issue #810 (clean-result body).

Two low-level data views required by the clean-result spec (underlying data
alongside every aggregate):

1. ``recon_per_context_scatter`` — per-context held-out normalized
   reconstruction error at layer 21 under the mean summary (x) vs the
   max-pool summary (y), from the committed per-context (SS_res, SS_tot)
   decomposition in ``eval_results/issue_810/analysis/bootstrap_deltaskill.json``.
2. ``readout_refusal_layer_profile`` — refusal fixed-direction read-out
   Spearman rho per layer for three summaries, from the committed
   ``eval_results/issue_810/readout_rho_by_summary.json``, with the
   selection-symmetric null band (97.5th pct of the max-|rho| statistic,
   committed ``analysis/analysis_summary.json``) drawn as dashed lines.

Reads the COMMITTED (HEAD) versions of both JSONs so the figures match the
numbers quoted in the clean-result body.
"""

from __future__ import annotations

import json
import subprocess

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)


def gitload(path: str) -> dict:
    """Load a JSON file from the committed HEAD tree (not the working copy)."""
    raw = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True).stdout
    return json.loads(raw)


def fig_recon_per_context() -> None:
    boot = gitload("eval_results/issue_810/analysis/bootstrap_deltaskill.json")
    pcd = boot["per_context_decomposition"]
    layer = 21
    ids = pcd["mean"]["context_ids"]
    assert ids == pcd["maxp"]["context_ids"], "context id order mismatch"
    ratio = {}
    for s in ("mean", "maxp"):
        ss_res = np.asarray(pcd[s]["ss_res"])  # (28 layers, 50 contexts)
        ss_tot = np.asarray(pcd[s]["ss_tot"])
        assert ss_res.shape == (28, 50) and ss_tot.shape == (28, 50)
        ratio[s] = ss_res[layer] / ss_tot[layer]

    fig, ax = plt.subplots()
    lims = [
        min(ratio["mean"].min(), ratio["maxp"].min()) * 0.8,
        max(ratio["mean"].max(), ratio["maxp"].max()) * 1.25,
    ]
    ax.plot(lims, lims, ls="--", lw=1.0, color=paper_palette_role("neutral"), zorder=1)
    ax.scatter(
        ratio["mean"],
        ratio["maxp"],
        s=28,
        color=paper_palette_role("primary"),
        alpha=0.85,
        zorder=2,
    )
    # Label the most off-diagonal contexts (largest |log ratio difference|),
    # alternating vertical anchors so neighboring labels do not collide.
    gap = np.abs(np.log(ratio["maxp"]) - np.log(ratio["mean"]))
    for rank, i in enumerate(np.argsort(gap)[-10:]):
        ax.text(
            ratio["mean"][i] * 1.05,
            ratio["maxp"][i] * (1.06 if rank % 2 else 0.94),
            ids[i],
            fontsize=5.5,
            va="bottom" if rank % 2 else "top",
            color="#444444",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("held-out error fraction, mean summary (SS_res / SS_tot, layer 21)")
    ax.set_ylabel("held-out error fraction, max-pool summary")
    ax.set_title("Per-context reconstruction error at layer 21: mean vs max-pool (n = 50)")
    savefig_paper(fig, "issue_810/recon_per_context_scatter", dir="figures/")
    plt.close(fig)
    below = int((ratio["maxp"] < ratio["mean"]).sum())
    print(f"recon scatter: {below}/50 contexts below diagonal (max-pool better)")


def fig_refusal_profile() -> None:
    d = gitload("eval_results/issue_810/readout_rho_by_summary.json")
    summ = gitload("eval_results/issue_810/analysis/analysis_summary.json")
    band = summ["readout_honest_band"]["per_method_bands"]["fixed_rb"]["abs"]
    cells = [c for c in d["cells"] if c["behavior"] == "refusal" and c["method"] == "fixed_rb"]
    series = {
        "turn_nl": "newline after turn end",
        "mean": "mean over answer tokens",
        "im_end": "turn-end token",
    }
    colors = {
        "turn_nl": paper_palette_role("primary"),
        "mean": paper_palette_role("baseline"),
        "im_end": paper_palette_role("accent"),
    }
    fig, ax = plt.subplots()
    for slug, label in series.items():
        pts = sorted(
            ((c["layer"], c["rho_graded"]) for c in cells if c["summary"] == slug),
        )
        layers, rho = zip(*pts, strict=True)
        ax.plot(layers, rho, marker="o", ms=3, lw=1.6, label=label, color=colors[slug])
    ax.axhline(band, ls="--", lw=1.0, color=paper_palette_role("neutral"))
    ax.axhline(-band, ls="--", lw=1.0, color=paper_palette_role("neutral"))
    ax.axhline(0, lw=0.8, color="#999999")
    ax.plot(
        [],
        [],
        ls="--",
        lw=1.0,
        color=paper_palette_role("neutral"),
        label="selection-symmetric null band (97.5th pct of max)",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out Spearman rho")
    ax.set_title("Refusal read-out along the fixed behavior direction, per layer (n = 50)")
    ax.legend(loc="upper center", fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "issue_810/readout_refusal_layer_profile", dir="figures/")
    plt.close(fig)
    tn = {c["layer"]: c["rho_graded"] for c in cells if c["summary"] == "turn_nl"}
    peak_layer = min(tn, key=lambda k: tn[k])
    print(
        f"refusal profile: turn_nl peak rho {tn[peak_layer]:+.3f} @ L{peak_layer}; band {band:.3f}"
    )


if __name__ == "__main__":
    set_paper_style("blog")
    fig_recon_per_context()
    fig_refusal_profile()
