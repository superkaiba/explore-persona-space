"""Per-feature ECDF companion covering every aggregate series of the k=200 tier-R2 hero.

Renders ``i2476_k200_tier_r2_percell_ecdf``: a 3 (tier rows) x 4 (floor columns)
grid of per-feature held-out R2 cumulative distributions — the k = 200 map
(solid) and identity+bias (dashed) series from
``eval_results/issue_2476/k200_census/perfeature_union_k200.npz``, plus the
k = 100 map reference (dotted) from
``eval_results/issue_2476/floor_sweep/perfeature_union_c.npz`` — the low-level
per-unit view behind every series the hero ``i2476_k200_tier_r2_hero`` plots
(clean-result round-8 fix; supersedes the finest-only
``i2476_k200_finest_r2_ecdf`` panel in the body). Zero-GPU: reads two committed
npz files, writes png + pdf + meta.json sidecar under ``figures/issue_2476/``.

Style matches the round's committed panels (``set_paper_style("iclr")``,
``scripts/issue2476_k200_census.py`` P7 conventions: ``paper_palette(3)`` tier
colors, x clipped to [-1, 1], dotted zero line, gray-0.45 k = 100 reference).

Usage: ``uv run python scripts/issue2476_k200_percell_ecdf.py``
"""

from __future__ import annotations

from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + credentials BEFORE numpy/matplotlib (shared-VM, #847)

import numpy as np  # noqa: E402

K200_NPZ = Path("eval_results/issue_2476/k200_census/perfeature_union_k200.npz")
K100_NPZ = Path("eval_results/issue_2476/floor_sweep/perfeature_union_c.npz")
FIG_DIR = Path("figures/issue_2476")
STEM = "i2476_k200_tier_r2_percell_ecdf"

FLOORS = [1200, 600, 300, 240]
FLOOR_TITLES = {1200: "1% floor", 600: "0.5% floor", 300: "0.25% floor", 240: "0.2% floor"}
TIER_SHORT = {0: "coarsest", 1: "middle", 2: "finest"}

# Committed-body alive counts (the first results table of the budget-round H3
# and the floor-sweep lattice table) — asserted before any pixel is drawn so a
# wrong/stale npz fails loud instead of rendering a silently different figure.
EXPECTED_ALIVE_K200 = {
    1200: [1239, 35, 4],
    600: [1483, 244, 21],
    300: [1645, 702, 53],
    240: [1687, 882, 90],
}
EXPECTED_ALIVE_K100 = {
    1200: [788, 88, 3],
    600: [992, 314, 26],
    300: [1137, 696, 80],
    240: [1178, 842, 130],
}


def _ecdf_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values + cumulative fraction; asserts every value is finite."""
    v = np.asarray(values, np.float64)
    if not np.isfinite(v).all():
        raise ValueError(f"non-finite per-feature R2 values: {int((~np.isfinite(v)).sum())}")
    v = np.sort(v)
    return v, np.arange(1, len(v) + 1) / len(v)


def main() -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("iclr")
    k200 = np.load(K200_NPZ)
    k100 = np.load(K100_NPZ)

    for fl in FLOORS:
        got200 = [int(((k200["tier"] == t) & k200[f"alive_f{fl}"]).sum()) for t in range(3)]
        got100 = [int(((k100["tier"] == t) & k100[f"alive_f{fl}"]).sum()) for t in range(3)]
        if got200 != EXPECTED_ALIVE_K200[fl]:
            raise AssertionError(
                f"k200 alive counts at floor {fl}: {got200} != {EXPECTED_ALIVE_K200[fl]}"
            )
        if got100 != EXPECTED_ALIVE_K100[fl]:
            raise AssertionError(
                f"k100 alive counts at floor {fl}: {got100} != {EXPECTED_ALIVE_K100[fl]}"
            )

    colors = paper_palette(3)
    fig, axes = plt.subplots(
        3,
        len(FLOORS),
        figsize=figsize_iclr_panels(len(FLOORS), height_in=3.4),
        sharex=True,
        sharey=True,
    )
    for t in range(3):
        for j, fl in enumerate(FLOORS):
            ax = axes[t, j]
            m200 = (k200["tier"] == t) & k200[f"alive_f{fl}"]
            m100 = (k100["tier"] == t) & k100[f"alive_f{fl}"]
            for key, ls, lw, color, label in (
                ("r2_map", "-", 1.0, colors[t], "map (k=200)"),
                ("r2_ib", "--", 0.9, colors[t], "identity+bias (k=200)"),
            ):
                x, y = _ecdf_xy(k200[key][m200])
                ax.plot(x, y, ls, color=color, lw=lw, label=label)
            x, y = _ecdf_xy(k100["r2_map"][m100])
            ax.plot(
                x, y, ":", color="0.45", lw=0.9, label="map k=100 (committed; different instrument)"
            )
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.axvline(0.0, ls=":", lw=0.5, color="gray")
            if t == 0:
                ax.set_title(FLOOR_TITLES[fl], fontsize=6)
            if t == 2:
                ax.set_xlabel("per-feature held-out R²", fontsize=5.5)
            if j == 0:
                ax.set_ylabel(f"{TIER_SHORT[t]}\nshare of features ≤ x", fontsize=5.5)
            # Caption-grounding print: share of identity+bias mass left of the -1 clip.
            below = float((k200["r2_ib"][m200] < -1.0).mean()) if int(m200.sum()) else float("nan")
            print(
                f"[percell_ecdf] tier={TIER_SHORT[t]} floor={fl}: "
                f"n200={int(m200.sum())} n100={int(m100.sum())} ib_share_below_-1={below:.3f}",
                flush=True,
            )
    axes[0, 0].legend(fontsize=4.0, loc="lower right")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, STEM, dir=FIG_DIR)
    plt.close(fig)
    print(f"[percell_ecdf] wrote {FIG_DIR / STEM}.png/.pdf/.meta.json", flush=True)


if __name__ == "__main__":
    main()
