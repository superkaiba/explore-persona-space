"""Regenerate issue #2476 figures flagged by figure-label reviews.

Round-2 interpretation revision (both round-1 critics): legends / panel titles
carried internal tokens ("arm c", "arm b", "chanind") — fixed via the driver's
`_ARM_LABELS` (that invocation also re-rendered the acc@1 hero + the
R2-vs-activity scatter; those renders stand and are reproducible from this
script's git history).

Round-2 CLEAN-RESULT revision (reconciler blocker
`figure-internal-issue-token-1482`): the two figures below still carried the
literal issue token "#1482" inside rendered label text. This script re-renders
them from the SAME committed eval artifacts with descriptive labels
("parent token-level round" / "token grain (parent round)"), reusing the
driver's helpers (`_bar_group`, `_parent_reference_medians`, `_med_ci`,
`_ARM_LABELS`, `TIER_LABELS`); the two plotting bodies are mirrored here
because the label text is inline in the driver (a frozen production driver —
not edited post-run):

  - figures/paper/c3_turnavg_tier_gradient.{png,pdf}
  - figures/issue_2476/i2476_bridge_token_vs_turnavg.{png,pdf}

Data content is unchanged — asserted: the re-rendered sidecar's per-point
numeric payload must equal the committed (git HEAD) sidecar's, label text
excluded, else this script fails loud.

Run from the worktree root:
    uv run python scripts/issue2476_fig_relabel.py
"""

import importlib.util
import json
import subprocess
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


def _numeric_payload(meta: dict) -> list:
    """Sorted multiset of each point's numeric values + kind (axis-label keys
    and series-label strings excluded — the label-invariant data payload)."""
    pts = []
    for p in meta.get("points", []):
        vals = sorted(
            round(float(v), 12)
            for v in p.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        pts.append((p.get("_kind"), tuple(vals)))
    return sorted(pts)


def _assert_sidecar_data_unchanged(meta_path: Path) -> None:
    rel = meta_path.resolve().relative_to(ROOT.resolve())
    old = json.loads(
        subprocess.run(
            ["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    new = json.loads(meta_path.read_text())
    assert _numeric_payload(old) == _numeric_payload(new), (
        f"sidecar DATA drift on relabel-only re-render: {rel}"
    )
    assert old.get("n_series") == new.get("n_series"), (old.get("n_series"), new.get("n_series"))
    print(f"[sidecar-check] {rel}: numeric payload identical to HEAD", flush=True)


def _fig_hero_gradient_relabel(drv, fig_paper: Path) -> None:
    """Driver `_fig_hero_gradient` body with the parent-reference scatter label
    reworded (no issue token); everything else identical."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_palette,
        savefig_paper,
    )

    tc = json.loads((EV / "tier_tests_c.json").read_text())
    tb = json.loads((EV / "tier_tests_b.json").read_text())
    b_suffix = " [demoted: exploratory]" if tb.get("arm_b_demoted") else ""
    colors = paper_palette(4)
    series = []
    for label, doc, mk, ck, col in (
        (f"{drv._ARM_LABELS['c']}: map", tc, "median_r2_map", "ci95_median_map", colors[0]),
        (f"{drv._ARM_LABELS['c']}: identity+bias", tc, "median_r2_ib", "ci95_median_ib", colors[1]),
        (
            f"{drv._ARM_LABELS['b']}: map{b_suffix}",
            tb,
            "median_r2_map",
            "ci95_median_map",
            colors[2],
        ),
        (
            f"{drv._ARM_LABELS['b']}: identity+bias{b_suffix}",
            tb,
            "median_r2_ib",
            "ci95_median_ib",
            colors[3],
        ),
    ):
        meds, cis = {}, {}
        for t in (0, 1, 2):
            meds[t], cis[t] = drv._med_ci(doc["per_tier"], mk, ck, t)
        series.append((label, meds, cis, col))
    fig, ax = plt.subplots(figsize=figsize_iclr_full())
    drv._bar_group(ax, series)
    ref = drv._parent_reference_medians()
    ax.scatter(
        [0, 1, 2],
        [ref[0], ref[1], ref[2]],
        marker="_",
        s=500,
        color="black",
        zorder=5,
        label=(
            "parent token-level round median — cross-{grain, layer, dictionary, "
            "score-rows} context from that round's 6,000-row score population, "
            "not a matched contrast"
        ),
    )
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_xlabel("matryoshka tier (feature-id prefix)")
    ax.set_ylabel("median held-out per-feature R²")
    ax.legend(fontsize=5, loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False)
    savefig_paper(fig, "c3_turnavg_tier_gradient", dir=fig_paper)
    plt.close(fig)

    print("[fig] c3_turnavg_tier_gradient (relabeled)", flush=True)


def _fig_bridge_relabel(drv, fig_expl: Path) -> None:
    """Driver exploratory panel (1) body with the x-axis label reworded (no
    issue token); everything else identical."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_palette,
        savefig_paper,
    )

    tier_colors = paper_palette(3)
    z = np.load(EV / "bridge_b.npz")
    fig, ax = plt.subplots(figsize=figsize_iclr_full())
    for t in (0, 1, 2):
        m = np.asarray(z["tier"], np.int64) == t
        if m.any():
            ax.scatter(
                z["r2_token_committed"][m],
                z["r2_turnavg"][m],
                s=6,
                alpha=0.5,
                color=tier_colors[t],
                label=drv.TIER_LABELS[t].replace("\n", " "),
            )
    lims = [-1.0, 1.0]
    ax.plot(lims, lims, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("per-feature R², token grain (parent round, committed)")
    ax.set_ylabel("per-feature R², turn-averaged (this run)")
    ax.legend(fontsize=5)
    savefig_paper(fig, "i2476_bridge_token_vs_turnavg", dir=fig_expl)
    plt.close(fig)
    print("[fig] i2476_bridge_token_vs_turnavg (relabeled)", flush=True)


def main() -> None:
    drv = _load_driver()
    import matplotlib

    matplotlib.use("Agg")

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("iclr")  # committed hero convention (driver phase_figures)

    fig_paper = ROOT / "figures" / "paper"
    fig_expl = ROOT / "figures" / "issue_2476"
    for tok in ("arm c", "arm b", "chanind"):
        assert all(tok not in v for v in drv._ARM_LABELS.values()), tok

    _fig_hero_gradient_relabel(drv, fig_paper)
    _assert_sidecar_data_unchanged(fig_paper / "c3_turnavg_tier_gradient.meta.json")
    _fig_bridge_relabel(drv, fig_expl)
    _assert_sidecar_data_unchanged(fig_expl / "i2476_bridge_token_vs_turnavg.meta.json")


if __name__ == "__main__":
    main()
