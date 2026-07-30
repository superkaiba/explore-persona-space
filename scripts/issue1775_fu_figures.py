"""Figures for the #1775 follow-up round `dedup-refit-pcfold-doubly-mlp`.

Two figures, both reading only the round's committed eval JSONs (plus the
run-1 per-fold headline companion for the paired PC-provenance view):

1. ``fu_n50k_dedup_bars`` — cell 1: the banked 50k fitter comparison
   re-fit on the deduplicated train set vs an n-matched random-drop
   control (levels per rung + gains vs ridge with 95% paired
   row-bootstrap CIs; banked unaudited numbers as reference).
2. ``fu_robustness_perfold`` — per-unit (per-fold) views behind the two
   new robustness aggregates: cell 2 (train-fold-only PC bases vs the
   full-population PC basis, paired per fold) and cell 3 (doubly-novel
   stitch-MLP seed-ensemble folds vs the pooled bilinear r = 32 level).

Usage: uv run python scripts/issue1775_fu_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: .env + shared-VM thread caps bind BEFORE the heavy imports
# (tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints).
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FU_DIR = REPO / "eval_results" / "issue_1775" / "fu_dedup_refit_pcfold_doubly"
FIG_DIR = REPO / "figures" / "issue_1775"

# One color = one meaning, matching the promoted body's figures:
# ridge gray-purple, kernel dark blue, MLP red (ladder bars); bilinear r=0
# purple / r=32 steel blue / stitch-MLP red / PRESS gray (per-fold headline).
C_RIDGE = "#8888aa"
C_KRR = "#3366cc"
C_MLP = "#cc3333"
C_RSKIP = "#dd8833"
C_R0 = "#9467bd"
C_R32 = "#1f77b4"
C_PRESS = "#888888"

RUNGS = ["ridge", "krr", "mlp", "residual_skip"]
RUNG_LABEL = {
    "ridge": "ridge",
    "krr": "kernel (RBF)",
    "mlp": "MLP",
    "residual_skip": "residual-skip MLP",
}
RUNG_COLOR = {"ridge": C_RIDGE, "krr": C_KRR, "mlp": C_MLP, "residual_skip": C_RSKIP}


def fig_dedup_bars() -> None:
    d = json.loads((FU_DIR / "n50k_dedup_refit.json").read_text())
    banked = d["banked_reference"]["per_predictor_whole_map_r2"]
    variants = [
        ("banked (unaudited)", None, banked, None),
        ("deduplicated", "deduped", None, None),
        ("random-drop control (same n)", "random_drop", None, None),
    ]

    fig, (ax_lvl, ax_gain) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # --- left: R^2 levels per rung x variant ---------------------------------
    width = 0.26
    xs = np.arange(len(RUNGS))
    for vi, (vlabel, vkey, ref, _) in enumerate(variants):
        offs = xs + (vi - 1) * width
        if vkey is None:
            vals = [ref[r] for r in RUNGS]
            lo = hi = None
        else:
            per = d["variants"][vkey]["per_rung"]
            vals = [per[r]["whole_map_r2"] for r in RUNGS]
            lo = [vals[i] - per[r]["bootstrap_ci"]["r2"]["lo"] for i, r in enumerate(RUNGS)]
            hi = [per[r]["bootstrap_ci"]["r2"]["hi"] - vals[i] for i, r in enumerate(RUNGS)]
        alpha = [0.35, 1.0, 0.68][vi]
        hatch = "//" if vkey is None else None
        bars = ax_lvl.bar(
            offs,
            vals,
            width=width * 0.92,
            color=[RUNG_COLOR[r] for r in RUNGS],
            alpha=alpha,
            hatch=hatch,
            edgecolor="white",
            label=vlabel,
        )
        if lo is not None:
            ax_lvl.errorbar(
                offs, vals, yerr=[lo, hi], fmt="none", ecolor="black", capsize=2, lw=0.9
            )
        for b, v in zip(bars, vals):
            ax_lvl.text(
                b.get_x() + b.get_width() / 2,
                0.702,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                rotation=90,
                color="black",
            )
    ax_lvl.set_xticks(xs)
    ax_lvl.set_xticklabels([RUNG_LABEL[r] for r in RUNGS], fontsize=9)
    ax_lvl.set_ylim(0.70, 0.83)
    ax_lvl.set_ylabel("held-out R2 (1,000 test rows, layer 19)")
    ax_lvl.set_title(
        "Levels: dedup lowers every rung; the same-n\nrandom drop reproduces the banked numbers",
        loc="left",
        fontsize=10,
    )

    # legend proxies: bar shade meaning (variant), not rung color
    from matplotlib.patches import Patch

    proxies = [
        Patch(
            facecolor="#999999",
            alpha=0.35,
            hatch="//",
            edgecolor="white",
            label="banked (unaudited)",
        ),
        Patch(facecolor="#999999", alpha=1.0, edgecolor="white", label="deduplicated"),
        Patch(
            facecolor="#999999", alpha=0.68, edgecolor="white", label="random-drop control (same n)"
        ),
    ]
    ax_lvl.legend(handles=proxies, loc="upper left", fontsize=8)

    # --- right: gains vs ridge with 95% CIs ----------------------------------
    gain_keys = ["krr_minus_ridge", "mlp_minus_ridge", "residual_skip_minus_ridge"]
    gain_rungs = ["krr", "mlp", "residual_skip"]
    gxs = np.arange(len(gain_keys))
    gwidth = 0.32
    for vi, vkey in enumerate(("deduped", "random_drop")):
        g = d["variants"][vkey]["gains_vs_ridge_paired_row_bootstrap"]
        offs = gxs + (vi - 0.5) * gwidth
        vals = [g[k]["delta_r2"] for k in gain_keys]
        lo = [vals[i] - g[k]["ci95_cluster"][0] for i, k in enumerate(gain_keys)]
        hi = [g[k]["ci95_cluster"][1] - vals[i] for i, k in enumerate(gain_keys)]
        alpha = 1.0 if vkey == "deduped" else 0.68
        bars = ax_gain.bar(
            offs,
            vals,
            width=gwidth * 0.9,
            color=[RUNG_COLOR[r] for r in gain_rungs],
            alpha=alpha,
            edgecolor="white",
        )
        ax_gain.errorbar(offs, vals, yerr=[lo, hi], fmt="none", ecolor="black", capsize=2, lw=0.9)
        for b, v in zip(bars, vals):
            ax_gain.text(
                b.get_x() + b.get_width() / 2,
                v + (0.0015 if v >= 0 else -0.004),
                f"{v:+.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for i, k in enumerate(gain_keys):
        bval = banked[gain_rungs[i]] - banked["ridge"]
        ax_gain.plot(
            [i - gwidth, i + gwidth],
            [bval, bval],
            ls=":",
            color="black",
            lw=1.1,
        )
    ax_gain.axhline(0.0, color="black", lw=0.8)
    ax_gain.set_xticks(gxs)
    ax_gain.set_xticklabels([RUNG_LABEL[r] + "\n- ridge" for r in gain_rungs], fontsize=9)
    ax_gain.set_ylabel("gain over ridge (delta R2)")
    ax_gain.set_title(
        "Gains vs ridge: solid = deduplicated, faded = random-drop;\ndotted line = banked (unaudited) gain",
        loc="left",
        fontsize=10,
    )

    fig.tight_layout()
    savefig_paper(fig, "fu_n50k_dedup_bars", dir=FIG_DIR)
    plt.close(fig)


def fig_robustness_perfold() -> None:
    foldpc = json.loads((FU_DIR / "bilinear_foldpc.json").read_text())
    doubly_mlp = json.loads((FU_DIR / "stitch_mlp_doubly.json").read_text())
    delta = json.loads((FU_DIR / "delta_beyond_doubly.json").read_text())
    perfold_full = json.loads(
        (REPO / "eval_results" / "issue_1775" / "bilinear" / "perfold_headline.json").read_text()
    )["schemes"]["prefix"]["levels"]

    fig, (ax_pc, ax_db) = plt.subplots(
        1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )

    # --- left: cell 2 — fold-PC (filled) vs full-population PC (open), per fold
    folds = sorted(foldpc["per_fold"], key=int)
    series = [
        (
            "additive stitch ridge (PRESS)",
            "r2_ridge_press_fold_basis",
            "stitch_press_ridge",
            C_PRESS,
            "o",
        ),
        ("bilinear r = 0 refit", "r2_r0_fold_basis", "bilinear_r0", C_R0, "D"),
        ("bilinear r = 32", "r2_r32_fold_basis", "bilinear_r32", C_R32, "s"),
    ]
    for label, fk, full_key, color, marker in series:
        fold_vals = [foldpc["per_fold"][f][fk] for f in folds]
        full_vals = perfold_full[full_key]["per_fold"]
        x = np.arange(len(folds))
        ax_pc.plot(
            x - 0.10,
            fold_vals,
            marker,
            color=color,
            ms=6,
            label=f"{label} - train-fold-only PCs",
        )
        ax_pc.plot(
            x + 0.10,
            full_vals,
            marker,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.4,
            ls="none",
            ms=6,
            label=f"{label} - full-population PCs",
        )
    ax_pc.set_xticks(np.arange(len(folds)))
    ax_pc.set_xticklabels([f"fold {f}" for f in folds], fontsize=9)
    ax_pc.set_ylabel("per-fold held-out R2 (48 answer PCs)")
    ax_pc.set_title(
        "PC provenance: filled = train-fold-only PC bases,\nopen = full-population PCs (novel-prefix folds)",
        loc="left",
        fontsize=10,
    )
    ax_pc.legend(fontsize=6.8, loc="center right", ncol=1)

    # --- right: cell 3 — doubly-novel per-fold stitch-MLP vs pooled bilinear
    db_folds = sorted(doubly_mlp["per_fold_r2_seed_ensemble"], key=int)
    db_vals = [doubly_mlp["per_fold_r2_seed_ensemble"][f] for f in db_folds]
    x = np.arange(len(db_folds))
    ax_db.plot(x, db_vals, "^", color=C_MLP, ms=7, label="stitch-MLP ensemble, per fold")
    for xi, v in zip(x, db_vals):
        ax_db.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.8)
    mlp_pooled = delta["r2_stitch_mlp_seed_mean"]
    bil_pooled = delta["r2_bilinear_r_star"]
    ax_db.axhline(
        mlp_pooled, ls="-.", color=C_MLP, lw=1.0, label=f"stitch-MLP pooled ({mlp_pooled:.3f})"
    )
    ax_db.axhline(
        bil_pooled, ls="--", color=C_R32, lw=1.0, label=f"bilinear r = 32 pooled ({bil_pooled:.3f})"
    )
    ax_db.set_xticks(x)
    ax_db.set_xticklabels([f"fold {f}" for f in db_folds], fontsize=9)
    ax_db.set_ylabel("held-out R2 (48 answer PCs)")
    ax_db.set_ylim(0.645, 0.760)
    ax_db.set_title(
        "Doubly-novel folds: the stitch-MLP sits above\nthe bilinear in every fold (2,900 rows)",
        loc="left",
        fontsize=10,
    )
    ax_db.legend(fontsize=7.5, loc="lower left", frameon=True, framealpha=1.0, edgecolor="none")

    fig.tight_layout()
    savefig_paper(fig, "fu_robustness_perfold", dir=FIG_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_dedup_bars()
    fig_robustness_perfold()
    print("done:", FIG_DIR / "fu_n50k_dedup_bars.png", FIG_DIR / "fu_robustness_perfold.png")


if __name__ == "__main__":
    main()
