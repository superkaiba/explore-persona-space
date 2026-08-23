"""Poster plot 1 — the scaling curve, with the generic boundary-token control.

MATS 2026 poster section 1. Held-out R^2 (left) and euclidean retrieval acc@1
against a 1,000-context pool (right), both against the number of training
contexts on a log x axis, for the linear (ridge) map, the identity+bias
baseline, and the nonlinear (MLP) map — plus, when the #1901 measured control
JSON resolves, the generic boundary-token ('.') control curve.

This is a THIN WRAPPER, not a reimplementation: the figure is the poster
variant already built into `scripts/issue1901_body_figures.py`
(`fig_paper_c1_scaling`); it relabels two legend entries for a poster audience.
The default paper render (`--style iclr` in that script) is untouched.

Boundary-control supersession (#1901 `generic-boundary-token-control` round):
this wrapper previously drew a dashed reference hline at 0.1087 via
`boundary_hline=` and called it "instruct R^2 0.1087". That constant was
WRONG-METRIC: 0.1087 is `fraction_of_fulln_ceiling` (0.10868122095179335) from
`eval_results/issue_825/base-separator-control/base_sep_to_chat.json`
(`decision_support.instruct_reference`) — NOT an R^2; the underlying
`sep_to_chat_fulln_r2` there is 0.0731533298226521 — so the line sat on the
R^2 axis mislabeled. The measured #1901 boundary-token scaling curve
(`eval_results/issue_1901/paper_densify/boundary_token_scaling_L19.json`)
supersedes it: when that JSON resolves the poster draws the measured curve
via `boundary=`; when it does not, the poster renders WITHOUT any generic
boundary representation and prints a loud stderr note — never a stale
wrong-metric hline.

It lives here because it was previously invoked by hand, which is why it was
the one panel the poster-wide font bump missed: every sibling generator in
this directory pins `set_paper_style("iclr", font_scale=...)`, and a figure
with no generator in the directory silently keeps whatever style the last
ad-hoc invocation used.

Run:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 uv run python docs/posters/mats_2026/make_plot1_scaling.py
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

# The #1901 measured generic boundary-token ('.') control curve. Optional by
# design: absent until the boundary-token control round's production p2/p3
# land the eval JSON.
BOUNDARY_JSON = PD / "boundary_token_scaling_L19.json"

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
    set_paper_style("iclr", font_scale=FONT_SCALE)
    l19, _p18, _boot = _load()
    dense_ladder = json.loads((PD / "scaling_ladder_L19.json").read_text())
    boundary = None
    if BOUNDARY_JSON.exists():
        boundary = json.loads(BOUNDARY_JSON.read_text())
    else:
        # Deliberately NO fallback reference line: the retired 0.1087 hline was
        # fraction_of_fulln_ceiling, not an R^2 (see module docstring) — until
        # the measured curve lands the poster carries no generic-boundary series.
        print(
            f"[poster] NOTE: {BOUNDARY_JSON} not found — rendering WITHOUT the "
            "generic boundary-token control series (no fallback hline by design)",
            file=sys.stderr,
        )
    fig_paper_c1_scaling(
        l19,
        dense_ladder,
        boundary=boundary,
        stem="plot1_scaling_boundary",
        out_dir=OUT_DIR,
        # same text as the paper's, broken over two lines: at 2.8in tall the
        # single-line form overruns the axis and collides with the legend
        # Pool size dropped from the axis (Thomas 2026-08-21). It stays in the
        # sidecar meta.json; on the board it was the only place a specific pool
        # was named, which read as contradicting the TL;DR's 10,000.
        acc_label="retrieval acc@1",
        # with the boundary series present the legend runs two rows, which needs
        # more than the paper's 14% headroom on this short canvas: at 0.86 the
        # second row lands on the retrieval panel's 1.0 tick (harmless extra
        # headroom in the boundary-absent one-row render)
        legend_rect_top=0.80,
        identity_label="identity + bias (baseline)",
        neural_label="nonlinear (MLP)",
        figsize=FIGSIZE,
    )
    print(f"wrote: {OUT_DIR / 'plot1_scaling_boundary.pdf'}")


if __name__ == "__main__":
    main()
