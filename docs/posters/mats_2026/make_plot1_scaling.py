"""Render the MATS 2026 poster scaling curves and copy-plus-bias baseline.

Uses the poster variant of ``scripts/issue1901_body_figures.py`` from banked
results, without the generic sentence-boundary comparison.

Run:
    uv run python docs/posters/mats_2026/make_plot1_scaling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from issue1901_body_figures import PD, _load, fig_paper_c1_scaling  # noqa: E402

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures"

# Poster font scale, matched to every other generator in this directory.
FONT_SCALE = 1.9

# The paper canvas (5.5 x 2.3in) was sized for font_scale=1.0; at 1.9 the legend
# lands on the axes and both y labels clip. Since the poster scales the figure
# to a fixed column width, growing the canvas by the full 1.9x would cancel the
# font bump exactly — this is ~1.9/1.4, so text lands ~1.4x the paper size and
# the layout still has room. The height is set by the right panel's long y
# label, which must fit inside the axes without reaching the legend band.
# Shorter, not narrower (2026-08-21): height only, so on-poster text size is
# unchanged. The left panel spends over half its span on the gap between the
# identity+bias baseline at -0.9 and the curves at 0.0-0.8, so the height comes
# off dead air. Narrowing instead would have shrunk every label.
FIGSIZE = (7.6, 2.8)


def main() -> None:
    """Write the poster scaling figure from the existing evaluation summaries."""
    set_paper_style("iclr", font_scale=FONT_SCALE)
    l19, _p18, _boot = _load()
    dense_ladder = json.loads((PD / "scaling_ladder_L19.json").read_text())
    fig_paper_c1_scaling(
        l19,
        dense_ladder,
        boundary=None,
        stem="plot1_scaling_boundary",
        out_dir=OUT_DIR,
        # same text as the paper's, broken over two lines: at 2.8in tall the
        # single-line form overruns the axis and collides with the legend
        # Pool size dropped from the axis (Thomas 2026-08-21). It stays in the
        # sidecar meta.json; on the board it was the only place a specific pool
        # was named, which read as contradicting the TL;DR's 10,000.
        acc_label="retrieval acc@1",
        legend_rect_top=0.80,
        identity_label="identity + bias (baseline)",
        neural_label="nonlinear (MLP)",
        figsize=FIGSIZE,
    )
    print(f"wrote: {OUT_DIR / 'plot1_scaling_boundary.pdf'}")


if __name__ == "__main__":
    main()
