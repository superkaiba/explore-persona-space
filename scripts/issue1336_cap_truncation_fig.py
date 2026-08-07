"""Figure: cap-truncation rate across every #1336 v2 generation cell (0 GPU-h).

Renders the census written by ``issue1336_cap_truncation_census.py`` as the
model x corpus grid it actually is, one panel per generation format. A table
would hide the two structures that matter and the grid makes immediate:

  * a LADDER gradient -- the pre-SFT base model is cut off by the 1024-token cap
    several times as often as the SFT rung, so cap censoring is correlated with
    the very axis the experiment varies (a confound for any cross-rung read, not
    merely a data-quality footnote); and
  * a CORPUS gradient -- reasoning-heavy corpora truncate far more than the short
    grade-school ones, so a per-cell caveat cannot be replaced by one number.

Every one of the 70 cells is drawn and annotated, so this is simultaneously the
aggregate view and the per-unit data behind it -- there is no hidden reduction.

Usage:
    uv run python scripts/issue1336_cap_truncation_fig.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS on the shared
# VM, and the BLAS pools freeze at import time — so it must run BEFORE matplotlib/numpy.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.task_workflow import repo_root  # noqa: E402

#: Post-training ladder order (the experiment's independent variable), not the
#: registry's dict order -- the whole point of the left-to-right/top-to-bottom
#: reading is that it follows the ladder.
LADDER = ("base", "sft", "dpo", "rlvr", "rlvr_long")


def _grid(per_cell: list[dict], gen_format: str) -> np.ndarray:
    """(n_models, n_corpora) truncation-rate matrix; NaN where a cell has no data."""
    corpora = list(cm.V2_CORPORA)
    out = np.full((len(LADDER), len(corpora)), np.nan)
    index = {(r["model"], r["corpus"], r["gen_format"]): r for r in per_cell}
    for i, model in enumerate(LADDER):
        for j, corpus in enumerate(corpora):
            row = index.get((model, corpus, gen_format))
            if row is not None and row["kept_truncation_rate"] is not None:
                out[i, j] = row["kept_truncation_rate"]
    return out


def render(census: dict, out_dir: Path) -> dict[str, Path]:
    set_paper_style()
    corpora = list(cm.V2_CORPORA)
    formats = ("chat", "naturalistic")
    grids = [_grid(census["per_cell"], f) for f in formats]
    # ONE shared colour scale across both panels: the panels are compared against
    # each other, so a per-panel scale would make equal rates look different.
    vmax = float(np.nanmax(np.concatenate([g.ravel() for g in grids])))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), constrained_layout=True)
    im = None
    for ax, fmt, grid in zip(axes, formats, grids, strict=True):
        im = ax.imshow(grid * 100.0, cmap="magma_r", vmin=0.0, vmax=vmax * 100.0, aspect="auto")
        ax.set_xticks(range(len(corpora)))
        ax.set_xticklabels(corpora, rotation=40, ha="right")
        ax.set_yticks(range(len(LADDER)))
        ax.set_yticklabels(LADDER)
        ax.set_title(f"{fmt} generation")
        # The paper style's gridlines draw ON TOP of the cells and strike through
        # the per-cell value labels; a heatmap has no use for them.
        ax.grid(False)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if np.isnan(val):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6)
                    continue
                # Flip the label colour on dark cells so every value stays readable.
                colour = "white" if val > 0.55 * vmax else "black"
                ax.text(
                    j, i, f"{val * 100:.1f}", ha="center", va="center", fontsize=6, color=colour
                )
    # Label the quantity EXACTLY. These are rows the engine stopped at the cap
    # (finish_reason == "length"), which is an UPPER BOUND on genuine length
    # censoring: a quarter to nearly half of them (25.4% base/lmsys23k, 43%
    # base/math7500) had their stored answer end earlier at a role-header strip,
    # so they were never cap-censoried at all. Calling the axis "truncated" would
    # invite reading the upper bound as the censored fraction.
    fig.colorbar(im, ax=axes, label='kept rows with finish_reason == "length" (%)', pad=0.02)
    axes[0].set_ylabel("post-training rung")
    return savefig_paper(fig, "cap_truncation_by_model_corpus", dir=out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = repo_root()
    ap.add_argument(
        "--census",
        type=Path,
        default=root / "eval_results" / "issue_1336" / "cap_truncation_census" / "census.json",
    )
    ap.add_argument("--out-dir", type=Path, default=root / "figures" / "issue_1336")
    args = ap.parse_args()
    census = json.loads(args.census.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = render(census, args.out_dir)
    for fmt, path in paths.items():
        print(f"[fig] {fmt}: {path}")


if __name__ == "__main__":
    main()
