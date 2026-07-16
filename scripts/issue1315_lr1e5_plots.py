"""Figures for the #1315 same-issue follow-up round `lr-matched-wildchat-geometry`.

Two figures into figures/issue_1315/lr1e5_followup/ (paper_plots blog style,
savefig_paper PNG+PDF+meta):

1. lr_contrast_L14      — hero: the four registered lattice reads at layer 14
                          (alignment, matched-80 rank, shared-text rank, paired
                          lr-contrast differences with 95% bootstrap CIs), both
                          lr cells, sycophancy band / bars as plotted references.
2. lr_layers_overlay    — low-level companion: per-layer own-text response-arm
                          rank-k@90 and |cos(mean shift, read-out)| for both lr
                          cells across all 28 decoder layers.

Data: eval_results/issue_1315/lr1e5_followup/geometry/geometry_per_cell.json
      (+ geometry_tf_shared.json for the shared-text read).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any heavy import — shared-VM thread caps (#847)

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

REPO_ROOT = _SCRIPTS_DIR.parent
GEO_DIR = REPO_ROOT / "eval_results" / "issue_1315" / "lr1e5_followup" / "geometry"
FIG_DIR = REPO_ROOT / "figures" / "issue_1315" / "lr1e5_followup"

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

LAYERS = list(range(28))
L14 = 14
CELLS = [
    ("imp_conv_lora_lr1e5", "lr 1e-5 (this round)"),
    ("imp_conv_lora", "lr 3e-5 (parent cell)"),
]
# Registered bars (plan §3) + parent-committed sycophancy references.
CONFIRM_BAR = 0.5
SYCO_CEILING = 0.2
SYCO_80ROW_BAND = (48.0, 50.0)  # V3 band bar (sycophancy matched-80 cells read 48.2-50.1)
COLLAPSE_BAR = 30.0  # V4 shared-text collapse bar
SYCO_SHARED_SPAN = (27.0, 35.0)  # parent sycophancy shared-text span (task 1112)


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def rec(records: dict, cell: str, arm: str, layer: int) -> dict:
    return records[f"{cell}/selected/{arm}/L{layer}"]


def _clamped_err(v: float, lo: float, hi: float) -> tuple[float, float]:
    """Non-negative errorbar offsets from CI bounds (never bounds, never signed)."""
    return max(0.0, v - lo), max(0.0, hi - v)


def fig_hero(own: dict, tf: dict) -> None:
    """Hero: the four registered layer-14 reads for the paired lr contrast."""
    set_paper_style("blog")
    colors = paper_palette_blog(2)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.8))
    r_own = own["records"]
    r_tf = tf["records"]
    xs = [0, 1]
    tick_labels = ["lr 1e-5", "lr 3e-5"]

    # Panel A — alignment |cos(mu, r_B)| at L14, own text.
    ax = axes[0]
    cos_vals = [abs(rec(r_own, c, "response", L14)["cos_mu_to_rb"]) for c, _ in CELLS]
    chance = rec(r_own, CELLS[0][0], "response", L14)["random_cos_ci"]["ci_high"]
    for x, v, (_, lab), col in zip(xs, cos_vals, CELLS, colors, strict=True):
        ax.plot([x], [v], "o", ms=9, color=col, label=lab)
    ax.axhline(CONFIRM_BAR, ls="--", lw=1.2, color="black", label="confirm bar (0.5)")
    ax.axhline(SYCO_CEILING, ls=":", lw=1.2, color="tab:red", label="sycophancy ceiling (0.2)")
    ax.axhline(chance, lw=1.0, color="gray", alpha=0.7, label="chance bound (97.5%)")
    ax.set_xticks(xs, tick_labels)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("|cos(mean shift, read-out)|")
    ax.set_title("Read-out alignment\n(layer 14, own text)")
    ax.legend(fontsize=8, loc="center right")

    # Panel B — matched-80-row rank (mean ± sd over 100 draws).
    ax = axes[1]
    ss = own["subsample_sensitivity_80row"]
    for x, (c, _lab), col in zip(xs, CELLS, colors, strict=True):
        m = ss[c]["rank_k_at_90_mean"]
        sd = ss[c]["rank_k_at_90_std"]
        ax.errorbar([x], [m], yerr=[[sd], [sd]], fmt="o", ms=9, color=col, capsize=4)
    ax.axhspan(
        *SYCO_80ROW_BAND,
        color="tab:gray",
        alpha=0.25,
        label="sycophancy band (48–50)",  # noqa: RUF001 — en dash kept verbatim (r1 label)
    )
    ax.set_xticks(xs, tick_labels)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(34, 54)
    ax.set_ylabel("rank-k@90, 80-row subsamples (mean ± sd)")
    ax.set_title("Matched-80-row rank\n(layer 14, own text)")
    ax.legend(fontsize=8, loc="upper left")

    # Panel C — shared-text rank at L14.
    ax = axes[2]
    sh_vals = [rec(r_tf, c, "response", L14)["rank_k_at_90"] for c, _ in CELLS]
    for x, v, (_, _lab), col in zip(xs, sh_vals, CELLS, colors, strict=True):
        ax.plot([x], [v], "o", ms=9, color=col)
    ax.axhline(COLLAPSE_BAR, ls="--", lw=1.2, color="black", label="collapse bar (30)")
    ax.axhspan(
        *SYCO_SHARED_SPAN,
        color="tab:gray",
        alpha=0.25,
        label="sycophancy shared span (27–35)",  # noqa: RUF001 — en dash kept verbatim (r1 label)
    )
    ax.set_xticks(xs, tick_labels)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(15, 38)
    ax.set_ylabel("rank-k@90, shared text (120 rows)")
    ax.set_title("Shared-text rank\n(layer 14, teacher-forced control)")
    ax.legend(fontsize=8, loc="upper left")

    # Panel D — registered paired differences (lr 1e-5 minus lr 3e-5), 95% CIs.
    ax = axes[3]
    lr = own["cross_cell_diffs"]["LRconv_lr1e5_vs_lr3e5"]["reads"][f"response/L{L14}"]
    diffs = [
        ("Δ rank-k@90\n(modes)", lr["diff_rank_k_at_90"], 1.0),
        ("Δ participation\nratio", lr["diff_pr_lambda"], 1.0),
        ("Δ top-share\n(pp)", lr["diff_top_share_lambda"], 100.0),
    ]
    for i, (_lab, dd, scale) in enumerate(diffs):
        v = dd["point"] * scale
        lo_off, hi_off = _clamped_err(v, dd["ci_low"] * scale, dd["ci_high"] * scale)
        ax.errorbar([i], [v], yerr=[[lo_off], [hi_off]], fmt="o", ms=9, color=colors[0], capsize=4)
    ax.axhline(0.0, lw=1.0, color="black")
    ax.set_xticks(range(3), [d[0] for d in diffs])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("lr 1e-5 − lr 3e-5 (95% CI)")  # noqa: RUF001 — minus sign kept verbatim (r1)
    ax.set_title("Paired lr contrast\n(cluster bootstrap, n=1000)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2], labels[:2], loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=2, fontsize=9
    )
    fig.tight_layout()
    savefig_paper(fig, "lr_contrast_L14", dir=FIG_DIR)
    plt.close(fig)


def fig_layers_overlay(own: dict) -> None:
    """Low-level companion: per-layer own-text rank + alignment + mean-shift norm."""
    set_paper_style("blog")
    colors = paper_palette_blog(2)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    r = own["records"]

    ax = axes[0]
    for (c, lab), col in zip(CELLS, colors, strict=True):
        ys = [rec(r, c, "response", li)["rank_k_at_90"] for li in LAYERS]
        ax.plot(LAYERS, ys, color=col, lw=2, marker="o", ms=3, label=lab)
    ax.axvline(L14, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("rank-k@90 of the shift cloud (120 rows)")
    ax.set_title("Own-text response arm: rank per layer")
    ax.legend(fontsize=9)

    ax = axes[1]
    for (c, lab), col in zip(CELLS, colors, strict=True):
        ys = [abs(rec(r, c, "response", li)["cos_mu_to_rb"]) for li in LAYERS]
        ax.plot(LAYERS, ys, color=col, lw=2, marker="o", ms=3, label=lab)
    chance = [rec(r, CELLS[0][0], "response", li)["random_cos_ci"]["ci_high"] for li in LAYERS]
    ax.plot(LAYERS, chance, color="gray", lw=1.2, alpha=0.8, label="chance bound (97.5%)")
    ax.axvline(L14, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("|cos(mean shift, read-out)|")
    ax.set_title("Own-text response arm: alignment per layer")
    ax.legend(fontsize=9)

    ax = axes[2]
    for (c, lab), col in zip(CELLS, colors, strict=True):
        ys = [rec(r, c, "response", li)["mu_norm"] for li in LAYERS]
        ax.plot(LAYERS, ys, color=col, lw=2, marker="o", ms=3, label=lab)
    ax.set_yscale("log")
    ax.axvline(L14, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("mean-shift norm (log scale)")
    ax.set_title("Own-text response arm: shift norm per layer")
    ax.legend(fontsize=9)

    fig.tight_layout()
    savefig_paper(fig, "lr_layers_overlay", dir=FIG_DIR)
    plt.close(fig)


def main() -> int:
    own = _load(GEO_DIR / "geometry_per_cell.json")
    tf = _load(GEO_DIR / "geometry_tf_shared.json")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_hero(own, tf)
    fig_layers_overlay(own)
    # Compact numeric echo for the round record.
    r14a = rec(own["records"], "imp_conv_lora_lr1e5", "response", L14)
    r14b = rec(own["records"], "imp_conv_lora", "response", L14)
    print(
        json.dumps(
            {
                "cos_new": r14a["cos_mu_to_rb"],
                "cos_sib": r14b["cos_mu_to_rb"],
                "rank_new": r14a["rank_k_at_90"],
                "rank_sib": r14b["rank_k_at_90"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
