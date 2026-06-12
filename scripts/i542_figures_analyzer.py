"""Issue #542 analyzer-round figures (in addition to scripts/i542_figures.py):

1. ``h_default_paired`` — the H-default registered contrast as a paired
   slopegraph: default-eval-column Delta logP(marker) per broad train-context
   row, close-persona panel (no default negative) vs default-including panel
   (single swap: ph2-twin -> default assistant).
2. ``ladder_oof_forest`` — per-arm out-of-fold R^2 for the top metric-ladder
   predictors plus the two new distance-to-panel rows, showing ranking
   stability across arms and the null distance-to-panel read.

CPU-only; consumes the per-cell G_cells rollups + ladder_scores_542.json.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = Path(os.environ.get("I542_FIG_DIR", str(REPO / "figures/issue_542")))
EVAL_ROOT = Path(os.environ.get("I542_EVAL_ROOT", str(REPO / "eval_results/issue_542")))

BROAD_ROWS = [
    "sp_swe",
    "sp_doctor",
    "sp_ph1",
    "sp_ph2",
    "wc_short_code",
    "wc_short_advice",
    "wc_long_write",
    "reph_imp",
    "reph_polite",
    "reph_casual",
]
ROW_LABELS = {
    "sp_swe": "software engineer",
    "sp_doctor": "medical doctor",
    "sp_ph1": "PersonaHub #1",
    "sp_ph2": "PersonaHub #2",
    "wc_short_code": "WildChat code chat",
    "wc_short_advice": "WildChat advice chat",
    "wc_long_write": "WildChat writing chat",
    "reph_imp": "imperative rephrase",
    "reph_polite": "polite rephrase",
    "reph_casual": "casual rephrase",
}
ARM_LABELS = {
    "arm1_xfam": "Cross-family panel (parent control)",
    "arm2_close": "Close-persona panel",
    "arm3_default": "Default-including panel",
    "c2": "Two negatives",
    "c8": "Eight negatives",
    "c16": "Sixteen negatives",
}
METRIC_LABELS = {
    "rbf_mmd2": "Kernel two-sample\n(RBF MMD$^2$)",
    "bures_w2": "Bures-Wasserstein\ncovariance dist.",
    "euclidean": "Euclidean dist.\n(context means)",
    "centroid_cosine": "Cosine dist.\n(context means)",
    "js_first_token": "First-token\nJS divergence",
    "dist_to_panel_mean": "Distance to\nneg. panel (mean)",
    "dist_to_panel_min": "Distance to\nneg. panel (min)",
}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()


def _save(fig, name: str, meta: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    (FIG_DIR / f"{name}.meta.json").write_text(
        json.dumps(
            {
                "git_commit": _git_commit(),
                "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                **meta,
            },
            indent=2,
        )
    )
    plt.close(fig)
    print(f"[figures] wrote {FIG_DIR / name}.png")


def _default_col(arm: str) -> dict[str, float]:
    out = {}
    for f in glob.glob(str(EVAL_ROOT / f"G_cells/{arm}/*_seed42.json")):
        d = json.loads(Path(f).read_text())
        if d["train_cid"] in BROAD_ROWS:
            out[d["train_cid"]] = d["eval_columns"]["default"]["g_mean_delta_logp"]
    return out


def h_default_paired() -> None:
    from explore_persona_space.analysis.paper_plots import paper_palette_role

    a2 = _default_col("arm2_close")
    a3 = _default_col("arm3_default")
    rows = [r for r in BROAD_ROWS if r in a2 and r in a3]
    assert len(rows) == 10, rows
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    c_base = paper_palette_role("baseline")
    c_prim = paper_palette_role("primary")
    # dodge row labels so close-together values don't overlap
    order = sorted(rows, key=lambda r: a3[r])
    label_y = {}
    prev = -np.inf
    min_sep = 0.16
    for r in order:
        y = max(a3[r], prev + min_sep)
        label_y[r] = y
        prev = y
    for i, r in enumerate(rows):
        ax.plot([0, 1], [a2[r], a3[r]], color="#bbbbbb", lw=0.9, zorder=1)
        ax.scatter([0], [a2[r]], color=c_base, s=38, zorder=2)
        ax.scatter([1], [a3[r]], color=c_prim, s=38, zorder=2)
        ax.annotate(
            ROW_LABELS[r],
            (1.04, label_y[r]),
            fontsize=6.5,
            va="center",
            color="#555555",
        )
    m2 = float(np.mean([a2[r] for r in rows]))
    m3 = float(np.mean([a3[r] for r in rows]))
    ax.scatter([0], [m2], color=c_base, s=170, marker="_", linewidths=3, zorder=3)
    ax.scatter([1], [m3], color=c_prim, s=170, marker="_", linewidths=3, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            "Close-persona panel\n(no default negative)",
            "Default-including panel\n(default swapped in)",
        ]
    )
    ax.set_xlim(-0.35, 1.55)
    ax.set_ylabel("Default-context leakage:\n" + r"$\Delta$logP(marker) trained $-$ base (nat)")
    ax.set_title(
        "Training the default assistant as a negative does not\nsuppress leakage to the default context",
        fontsize=10,
        loc="left",
    )
    _save(
        fig,
        "h_default_paired",
        {
            "rows": rows,
            "arm2_mean": m2,
            "arm3_mean": m3,
            "note": "paired per-train-row default-column reads; single swap ph2-twin -> default",
        },
    )


def ladder_oof_forest() -> None:
    from explore_persona_space.analysis.paper_plots import paper_palette

    ladder = json.loads((EVAL_ROOT / "baselines/ladder_scores_542.json").read_text())
    arms = [a for a in ARM_LABELS if a in ladder["arms"]]
    metrics = list(METRIC_LABELS)
    colors = paper_palette(len(arms))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for ai, arm in enumerate(arms):
        sc = ladder["arms"][arm]["scores"]
        xs, ys = [], []
        for mi, m in enumerate(metrics):
            if m not in sc:
                continue
            xs.append(mi + (ai - (len(arms) - 1) / 2) * 0.09)
            ys.append(sc[m]["oof_r2"])
        ax.scatter(xs, ys, color=colors[ai], s=26, label=ARM_LABELS[arm], zorder=3)
    ax.axhline(0.0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.axvspan(4.5, 6.5, color="#f2d9b8", alpha=0.35, zorder=0)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=7)
    ax.set_ylabel("Out-of-fold $R^2$ predicting off-diagonal leakage")
    ax.set_title(
        "The predictor leaderboard is panel-invariant; distance to the\nnegative panel (shaded) predicts nothing in any arm",
        fontsize=10,
        loc="left",
    )
    ax.legend(fontsize=6.5, loc="lower left")
    _save(
        fig,
        "ladder_oof_forest",
        {"arms": arms, "metrics": metrics},
    )


def main() -> int:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    h_default_paired()
    ladder_oof_forest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
