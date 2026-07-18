"""Result figures for #958: forecast-from-turn-1 (Result 2) and prefix-vs-context by depth (Result 3).

Emits:
- figures/issue_958/forecast_from_turn1.png     — forecast skill vs baselines, by target turn
- figures/issue_958/prefix_vs_context_by_turn.png — absolute skill + prefix/context ratio, turns 2-4

All values recomputed from the committed per-conversation cells
(eval_results/issue_958/percell/*.npz; recompute matches the committed aggregates in
forecast_curves.json / transfer_matrix.json bitwise) with 95% bootstrap CIs paired over
the 500 held-out test conversations (997 draws, seed 0 — the analyzer's convention).
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/matplotlib so the shared-VM thread caps bind (#847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results" / "issue_958"
ROWS = [b + 1 for b in (14, 17, 19, 20, 24, 26)]  # readout blocks -> percell row indices
DRAWS, SEED = 997, 0


def _cell(name: str) -> dict:
    z = np.load(EV / "percell" / f"{name}.npz")
    return {"sse": z["sse_unit"][ROWS], "null": z["null_sse_unit"][ROWS], "test_idx": z["test_idx"]}


def _skill(sse: np.ndarray, null: np.ndarray, idx=None) -> float:
    if idx is not None:
        sse, null = sse[:, idx], null[:, idx]
    return float((1 - sse.sum(axis=1) / null.sum(axis=1)).mean())


def _boot_series(cells: list[dict], rng: np.random.Generator) -> tuple[list, np.ndarray]:
    """Point estimates + (2, k) clamped CI offsets, one shared resample stream per call."""
    pts = [_skill(c["sse"], c["null"]) for c in cells]
    n = cells[0]["sse"].shape[1]
    draws = np.empty((DRAWS, len(cells)))
    for d in range(DRAWS):
        idx = rng.integers(0, n, n)
        draws[d] = [_skill(c["sse"], c["null"], idx) for c in cells]
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    v = np.asarray(pts)
    return pts, np.vstack([np.maximum(0, v - lo), np.maximum(0, hi - v)])


def _verify_against_committed() -> None:
    fc = json.loads((EV.parent / "issue_958" / "forecast_curves.json").read_text())
    assert (
        abs(
            _skill(**{k: v for k, v in _cell("fcast_1to2").items() if k != "test_idx"})
            - fc["forecast"]["1->2"]
        )
        < 1e-6
    )


def fig_forecast() -> None:
    set_paper_style("blog")
    colors = paper_palette(4)
    fig, ax = plt.subplots()
    ax.axhline(0, color="#B0B0B0", linewidth=0.8, linestyle=":", zorder=1)

    series = [
        (
            "forecast from the turn-1 state",
            ["fcast_1to2", "fcast_1to3", "fcast_1to4"],
            colors[0],
            "-",
        ),
        (
            "one-step-ahead forecast (from turn k−1)",
            ["fcast_1to2", "fcast_2to3", "fcast_3to4"],
            colors[1],
            "--",
        ),
        ("copy previous answer", ["copyprev_k2", "copyprev_k3", "copyprev_k4"], colors[2], "-"),
        ("prefix-only map", ["pre_k2_full", "pre_k3_full", "pre_k4_full"], colors[3], "-"),
    ]
    x = np.array([2, 3, 4])
    for i, (label, names, color, ls) in enumerate(series):
        cells = [_cell(n) for n in names]
        pts, yerr = _boot_series(cells, np.random.default_rng(SEED))
        ax.errorbar(
            x + (i - 1.5) * 0.03,
            pts,
            yerr=yerr,
            marker="o",
            color=color,
            linestyle=ls,
            capsize=2,
            label=label,
        )

    ax.set_xlabel("target turn k")
    ax.set_ylabel("held-out skill (R² vs corpus-mean baseline)")
    ax.set_xticks([2, 3, 4])
    ax.set_ylim(-0.27, 0.32)
    ax.legend(loc="lower right", fontsize=9)
    set_title_subtitle(
        ax,
        "Predicting later answers from earlier context states",
        "answer-k prediction; 95% bootstrap CIs, n=500 conversations",
    )
    savefig_paper(fig, "issue_958/forecast_from_turn1", dir=str(ROOT / "figures"))
    plt.close(fig)


def fig_prefix_vs_context() -> None:
    set_paper_style("blog")
    colors = paper_palette(4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    ctx_cells = [_cell(f"own_k{k}_full") for k in (2, 3, 4)]
    pre_cells = [_cell(f"pre_k{k}_full") for k in (2, 3, 4)]
    x = [2, 3, 4]

    ctx_pts, ctx_err = _boot_series(ctx_cells, np.random.default_rng(SEED))
    pre_pts, pre_err = _boot_series(pre_cells, np.random.default_rng(SEED))
    ax1.errorbar(
        x,
        ctx_pts,
        yerr=ctx_err,
        marker="o",
        color=colors[0],
        capsize=2,
        label="context map (prefix + query)",
    )
    ax1.errorbar(
        x, pre_pts, yerr=pre_err, marker="o", color=colors[3], capsize=2, label="prefix-only map"
    )
    ax1.set_xlabel("turn")
    ax1.set_ylabel("held-out skill (R² vs corpus mean)")
    ax1.set_xticks(x)
    ax1.set_ylim(0, 0.6)
    ax1.legend(loc="center right", fontsize=9)
    ax1.set_title("absolute skill, own-turn maps", loc="left", fontsize=11)

    # Paired ratio bootstrap: same resample indices for numerator and denominator.
    for c1, c2 in zip(pre_cells, ctx_cells):
        assert np.array_equal(c1["test_idx"], c2["test_idx"])
    ratio_pts = [p / c for p, c in zip(pre_pts, ctx_pts)]
    rng = np.random.default_rng(SEED)
    n = ctx_cells[0]["sse"].shape[1]
    rdraws = np.empty((DRAWS, 3))
    for d in range(DRAWS):
        idx = rng.integers(0, n, n)
        rdraws[d] = [
            _skill(pc["sse"], pc["null"], idx) / _skill(cc["sse"], cc["null"], idx)
            for pc, cc in zip(pre_cells, ctx_cells)
        ]
    lo, hi = np.percentile(rdraws, [2.5, 97.5], axis=0)
    v = np.asarray(ratio_pts)
    ax2.errorbar(
        x,
        ratio_pts,
        yerr=np.vstack([np.maximum(0, v - lo), np.maximum(0, hi - v)]),
        marker="o",
        color=colors[2],
        capsize=2,
    )
    ax2.set_xlabel("turn")
    ax2.set_ylabel("prefix-only ÷ context skill")
    ax2.set_xticks(x)
    ax2.set_ylim(0.2, 0.4)
    ax2.set_title("skill ratio", loc="left", fontsize=11)

    fig.suptitle(
        "Context vs prefix-only map skill across turns 2–4",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_958/prefix_vs_context_by_turn", dir=str(ROOT / "figures"))
    plt.close(fig)


def main() -> None:
    _verify_against_committed()
    fig_forecast()
    fig_prefix_vs_context()
    fig_trait_drift()


def fig_trait_drift() -> None:
    """Result 4: answer projections drift along trait read-out directions with turn depth.

    Per trait panel: mean projection of the actual answer activations onto the trait
    direction (npz col 1), centered at turn 1, with 95% bootstrap CIs over conversations;
    grey band = the same centered read for 100 norm-matched random directions
    (npz cols 3..102; column layout verified against the committed slope band).
    Solid = main panel (n=500, turns 1-4); dashed open = long panel (n=60, turns 1-8).
    """
    z = np.load(EV / "drift_actual_projections.npz")
    dr = json.loads((EV / "drift_read.json").read_text())
    # Per-trait actual-projection column: cols 0-2 hold the actual read at the three
    # trait layers in trait order [evil@20, sycophancy@26, hallucination@17];
    # cols 3..102 are the 100 norm-matched random directions at the trait's own layer.
    trait_col = {"evil": 0, "sycophancy": 1, "hallucination": 2}
    set_paper_style("blog")
    colors = paper_palette(4)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)

    panels = [
        ("sycophancy", "block 26", colors[0]),
        ("hallucination", "block 17", colors[1]),
        ("evil", "block 20", colors[2]),
    ]
    for ax, (trait, blk, color) in zip(axes, panels):
        main = np.stack([z[f"main_k{k}_{trait}"] for k in (1, 2, 3, 4)], axis=0)  # (4,500,103)
        act = main[:, :, trait_col[trait]]
        # Fail-loud check: reproduce the committed within-conversation slope + random band.
        t = np.array([1.0, 2.0, 3.0, 4.0])
        tc = t - t.mean()
        slope = float(((tc[:, None] * act).sum(axis=0) / (tc**2).sum()).mean())
        assert abs(slope - dr["drift"][trait]["within_conv_turn_slope"]) < 1e-6, (trait, slope)
        rs = (tc[:, None, None] * main[:, :, 3:103]).sum(axis=0) / (tc**2).sum()
        band = np.percentile(rs.mean(axis=0), [2.5, 97.5])
        assert np.allclose(band, dr["drift"][trait]["turn_slope_randdir_band_ci95"], atol=1e-6), (
            trait
        )
        centered = act.mean(axis=1) - act[0].mean()
        rng = np.random.default_rng(SEED)
        n = act.shape[1]
        draws = np.empty((DRAWS, 4))
        for d in range(DRAWS):
            idx = rng.integers(0, n, n)
            m = act[:, idx].mean(axis=1)
            draws[d] = m - m[0]
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        yerr = np.vstack([np.maximum(0, centered - lo), np.maximum(0, hi - centered)])

        rand = main[:, :, 3:103].mean(axis=1)  # (4, 100) per-dir mean per turn
        rand_c = rand - rand[0]
        blo, bhi = np.percentile(rand_c, [2.5, 97.5], axis=1)
        ax.fill_between(
            [1, 2, 3, 4],
            blo,
            bhi,
            color="#BBBBBB",
            alpha=0.4,
            linewidth=0,
            label="norm-matched random directions (95% band)",
        )
        ax.axhline(0, color="#B0B0B0", linewidth=0.6, linestyle=":")
        ax.errorbar(
            [1, 2, 3, 4],
            centered,
            yerr=yerr,
            marker="o",
            color=color,
            capsize=2,
            label="trait direction",
        )

        lp = np.stack([z[f"long_k{k}_{trait}"] for k in range(1, 9)], axis=0)[
            :, :, trait_col[trait]
        ]
        lcent = lp.mean(axis=1) - lp[0].mean()
        ax.plot(
            range(1, 9),
            lcent,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.1,
            markeredgecolor=color,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label="long panel (n=60)",
        )

        ax.set_title(f"{trait} direction ({blk})", loc="left", fontsize=11)
        ax.set_xlabel("turn")
        ax.set_xticks(range(1, 9))
    axes[0].set_ylabel("answer projection, change from turn 1")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Answer-activation drift along trait read-out directions",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_958/trait_drift_by_turn", dir=str(ROOT / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    main()
