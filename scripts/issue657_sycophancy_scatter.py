#!/usr/bin/env python3
"""Issue #657 — per-persona sycophancy scatter (raw alongside the forest-plot rho).

Plots the raw data underlying the forest-plot point rho = 0.68 (DV-(a) / H1, the
alignment->base-rate generalization for sycophancy): one point per persona,

    x = cosine(persona vector, sycophancy direction) at layer 14   (alignment)
    y = fraction of base-model generations judged sycophantic       (base rate)

Data sources (training-free reuse, no new compute):
  - alignment cosines: this task's
    ``eval_results/issue_657/per_behavior/sycophancy.json`` ``joined_cells``
    (the ``align`` field is constant per bystander persona; layer 14,
    last-prompt-token readout, global-mean-centered bank). These are the exact
    cosines the bake-off's DV-(a) read used.
  - base rates: #623 ``eval_results/issue_623/syc_i.json`` (``syc_i`` per persona,
    base-model sycophancy rate). This is the ``base_rate_source = issue623_syc_i``
    the persisted ``dv_a_base_rate`` block names.

The realized point set is the intersection of the two maps (personas with both an
alignment cosine and a #623 base rate), reproducing the persisted
``dv_a_base_rate`` exactly (n = 16, raw_rho = 0.6834).

Output: figures/issue_657/fig_h1_sycophancy_scatter.png (+ .pdf + .meta.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval_results"
SYCO_JSON = RES / "issue_657" / "per_behavior" / "sycophancy.json"
SYC_I_JSON = RES / "issue_623" / "syc_i.json"


def load_pairs() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Reconstruct the DV-(a) per-persona (alignment, base_rate) pairs."""
    syco = json.loads(SYCO_JSON.read_text())
    # `align` is constant per bystander across all cells it appears in.
    align: dict[str, float] = {}
    for cell in syco["joined_cells"]:
        a = cell["align"]
        if a is None or (isinstance(a, float) and np.isnan(a)):
            continue
        align[cell["bystander"]] = float(a)

    syc_i = json.loads(SYC_I_JSON.read_text())["syc_i"]
    base_rate = {p: float(v["syc_i"]) for p, v in syc_i.items()}

    personas = sorted(p for p in align if p in base_rate)
    if not personas:
        raise RuntimeError("No personas with both an alignment cosine and a base rate.")
    x = np.array([align[p] for p in personas], dtype=float)
    y = np.array([base_rate[p] for p in personas], dtype=float)
    return personas, x, y


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT / "src"))
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    primary = paper_palette_role("primary")

    personas, x, y = load_pairs()
    rho = spearmanr(x, y).correlation
    n = len(personas)
    print(f"n = {n}, rho = {rho:.4f}")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(x, y, s=48, color=primary, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlabel("Alignment to the sycophancy direction (cosine, layer 14)")
    ax.set_ylabel("Base sycophancy rate")
    ax.set_title("Each persona's alignment vs its own base sycophancy rate")
    ax.margins(x=0.08, y=0.10)

    out = savefig_paper(fig, "issue_657/fig_h1_sycophancy_scatter", dir="figures/")
    plt.close(fig)
    for fmt, path in out.items():
        print(f"wrote {fmt}: {path}")


if __name__ == "__main__":
    main()
