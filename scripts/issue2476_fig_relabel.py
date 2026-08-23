"""Regenerate the three issue #2476 figures flagged for opaque condition labels.

Round-2 interpretation revision (both round-1 critics): the rendered legends /
panel titles carried internal tokens ("arm c", "arm b", "chanind"). This script
re-renders, from the SAME committed eval artifacts:

  - figures/paper/c3_turnavg_tier_gradient.{png,pdf}   (hero, via the driver)
  - figures/paper/c3_turnavg_tier_acc1.{png,pdf}       (hero + new low-tier zoom
    inset, via the driver)
  - figures/issue_2476/i2476_r2_vs_activity.{png,pdf}  (re-implemented panel —
    the driver renders it inline inside the exploratory dump, which needs the
    full P4/P5 out-root state)

with the reader-facing condition names now in the driver's `_ARM_LABELS`
(style matches issue2476_attrition_fig.py). Data content is unchanged.

Run from the worktree root:
    uv run python scripts/issue2476_fig_relabel.py
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results" / "issue_2476" / "turnavg"


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "issue2476_turnavg_sae", ROOT / "scripts" / "issue2476_turnavg_sae.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue2476_turnavg_sae"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    drv = _load_driver()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("iclr")  # committed hero convention (driver phase_figures)

    fig_paper = ROOT / "figures" / "paper"
    fig_expl = ROOT / "figures" / "issue_2476"

    for tok in ("arm c", "arm b", "chanind"):
        assert all(tok not in v for v in drv._ARM_LABELS.values()), tok
    p1 = drv._fig_hero_gradient(EV, fig_paper)
    p2 = drv._fig_hero_acc1(EV, fig_paper)
    print("heroes:", sorted(str(v) for v in p1.values()) + sorted(str(v) for v in p2.values()))

    # R2 vs log10 activity by tier (one panel per arm) — same content as the
    # driver's exploratory panel (2), reader-facing panel titles.
    tier_colors = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=figsize_iclr_panels(2))
    for ax, tag in zip(axes, ("c", "b"), strict=True):
        pf = np.load(EV / f"perfeature_{tag}_encodepred.npz")
        act = np.asarray(pf["activity"], np.float64)
        for t in (0, 1, 2):
            m = np.asarray(pf["tier"], np.int64) == t
            if m.any():
                ax.scatter(
                    np.log10(act[m] + 1.0),
                    pf["r2"][m],
                    s=4,
                    alpha=0.4,
                    color=tier_colors[t],
                    label=drv.TIER_LABELS[t].replace("\n", " "),
                )
        ax.set_xlabel("log10(active fit rows + 1)")
        ax.set_ylabel("held-out per-feature R²")
        ax.set_title(drv._ARM_LABELS[tag], fontsize=6)
    axes[0].legend(fontsize=5)
    p3 = savefig_paper(fig, "i2476_r2_vs_activity", dir=fig_expl)
    plt.close(fig)
    print("scatter:", sorted(str(v) for v in p3.values()))


if __name__ == "__main__":
    main()
