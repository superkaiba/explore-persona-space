"""Fold-pass low-level data figure for issue #810 (free-analysis diagnostics).

``readout_refusal_scatter_L22`` — the 50 per-context (fixed-direction
projection, graded refusal score) pairs behind the refusal read-out peak cell
(newline-after-turn-end summary, layer 22, contrastive difference-of-means
direction), colored by median answer length. This is the per-unit data view
the clean-result spec requires alongside the aggregate Spearman rho quoted in
that result (the committed refusal figure shows only the layer profile).

Reads the COMMITTED (HEAD) ``eval_results/issue_810/analysis/
fa_refusal_diagnostics.json`` so the figure matches the numbers quoted in the
clean-result body.
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

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402


def gitload(path: str) -> dict:
    """Load a JSON file from the committed HEAD tree (not the working copy)."""
    raw = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True).stdout
    return json.loads(raw)


def main() -> None:
    """Build + save the per-context refusal read-out scatter (PNG + PDF + meta)."""
    d = gitload("eval_results/issue_810/analysis/fa_refusal_diagnostics.json")
    c = d["c_scatter_refusal_turn_nl_L22"]
    head = d["a_length_partialled_readout"]["headline_turn_nl_L22"]
    assert c["cell"]["summary"] == "turn_nl" and c["cell"]["layer"] == 22, c["cell"]

    trip = c["triples"]
    x = np.array([t["prediction"] for t in trip], dtype=float)
    y = np.array([t["graded_e0"] for t in trip], dtype=float)
    z = np.array([t["answer_len"] for t in trip], dtype=float)
    ids = [t["context_id"] for t in trip]
    assert x.shape == y.shape == z.shape == (c["n"],), (x.shape, c["n"])

    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, cmap="cividis", s=36, alpha=0.9, zorder=2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("median answer length (tokens)")

    # Label the most peripheral contexts (largest standardized L1 distance from
    # the cloud center), alternating vertical anchors to reduce collisions and
    # anchoring horizontally away from the nearest axis edge so no label clips.
    ax.margins(x=0.10, y=0.08)
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()
    xlo, xhi = x.min(), x.max()
    for rank, i in enumerate(np.argsort(np.abs(xs) + np.abs(ys))[-10:]):
        frac = (x[i] - xlo) / (xhi - xlo)
        ha = "left" if frac < 0.25 else ("right" if frac > 0.75 else "center")
        ax.text(
            x[i],
            y[i] + (0.7 if rank % 2 else -0.7),
            ids[i],
            fontsize=5.5,
            ha=ha,
            va="bottom" if rank % 2 else "top",
            color="#444444",
        )

    # Statistical label of the plotted relationship (the correlation IS the
    # claim): raw + length-partialled Spearman, from the committed JSON.
    ax.text(
        0.97,
        0.96,
        (
            f"Spearman rho = {head['rho_raw']:+.3f}\n"
            f"length-partialled rho = {head['rho_partial']:+.3f} "
            f"(p = {head['p_partial']:.1e})"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#333333",
    )

    ax.set_xlabel("projection on the fixed refusal direction (layer 22)")
    ax.set_ylabel("graded refusal score (0-100)")
    ax.set_title("Refusal score vs fixed-direction projection, newline after turn end (n = 50)")
    savefig_paper(fig, "issue_810/readout_refusal_scatter_L22", dir="figures/")
    plt.close(fig)
    print(f"scatter: n={c['n']} sanity_rho={c['sanity_rho']:+.4f}")


if __name__ == "__main__":
    set_paper_style("blog")
    main()
