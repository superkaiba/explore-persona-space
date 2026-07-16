"""Figures for the #1315 same-issue follow-up round `bare-context-geometry`.

Two figures into figures/issue_1315/bare_followup/ (paper_plots blog style,
savefig_paper PNG+PDF+meta):

1. bare_vs_scaffold_L14 — hero: the bare cell's registered layer-14 reads
                          (read-out alignment, matched-80-row rank, shared-text
                          rank) against the seven scaffolded cells' realized
                          spread, with the registered bars / bands as plotted
                          references.
2. bare_layers_overlay  — companion: per-layer own-text response-arm rank,
                          alignment, and mean-shift norm for the bare cell
                          against the scaffolded cells' min-max envelope.

Data: eval_results/issue_1315/bare_followup/geometry/geometry_per_cell.json
      (+ geometry_tf_shared.json for the shared-text read).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

REPO_ROOT = _SCRIPTS_DIR.parent
GEO_DIR = REPO_ROOT / "eval_results" / "issue_1315" / "bare_followup" / "geometry"
FIG_DIR = REPO_ROOT / "figures" / "issue_1315" / "bare_followup"

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

LAYERS = list(range(28))
L14 = 14
BARE = "imp_bare_lora"
SCAFFOLDED = [
    ("imp_pers_lora", "LoRA, persona context"),
    ("imp_conv_lora", "LoRA, WildChat context"),
    ("imp_conv_lora_lr1e5", "LoRA, WildChat context (lr 1e-5)"),
    ("imp_icl_lora_neg", "LoRA + negatives, ICL"),
    ("imp_icl_lora_pos", "LoRA positives-only, ICL"),
    ("imp_icl_ft_neg", "Full FT + negatives, ICL"),
    ("imp_icl_ft_pos", "Full FT positives-only, ICL"),
]
BARE_LABEL = "bare context (no scaffold)"
# Registered bars (plan v7 §3) + parent sycophancy references.
CONFIRM_BAR = 0.5
SYCO_CEILING = 0.2
SYCO_80ROW_BAND = (48.0, 50.0)
COLLAPSE_BAR = 30.0
SYCO_SHARED_SPAN = (27.0, 35.0)


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def rec(records: dict, cell: str, arm: str, layer: int) -> dict:
    return records[f"{cell}/selected/{arm}/L{layer}"]


def fig_hero(own: dict, tf: dict) -> None:
    """Hero: bare cell vs the scaffolded spread on the three registered read panels."""
    set_paper_style("blog")
    colors = paper_palette_blog(len(SCAFFOLDED))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    r_own = own["records"]
    r_tf = tf["records"]
    ss = own["subsample_sensitivity_80row"]
    # Small deterministic x-jitter for the scaffolded strip at x=0; bare at x=1.
    jit = [-0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18]

    # Panel A — |cos(mu, r_B)| at L14, own text (V1 floor + V2 ceiling).
    ax = axes[0]
    for (c, lab), col, dx in zip(SCAFFOLDED, colors, jit, strict=True):
        ax.plot(
            [dx],
            [abs(rec(r_own, c, "response", L14)["cos_mu_to_rb"])],
            "o",
            ms=8,
            color=col,
            label=lab,
        )
    ax.plot(
        [1.0],
        [abs(rec(r_own, BARE, "response", L14)["cos_mu_to_rb"])],
        "*",
        ms=16,
        color="black",
        label=BARE_LABEL,
    )
    chance = rec(r_own, BARE, "response", L14)["random_cos_ci"]["ci_high"]
    ax.axhline(CONFIRM_BAR, ls="--", lw=1.2, color="black", label="scaffold-floor bar (0.5)")
    ax.axhline(SYCO_CEILING, ls=":", lw=1.2, color="tab:red", label="sycophancy ceiling (0.2)")
    ax.axhline(chance, lw=1.0, color="gray", alpha=0.7, label="chance bound (97.5%)")
    ax.set_xticks([0, 1], ["scaffolded (7)", "bare"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("|cos(mean shift, read-out)|")
    ax.set_title("Read-out alignment\n(layer 14, own text)")

    # Panel B — matched-80-row rank (mean ± sd over 100 draws).
    ax = axes[1]
    for (c, lab), col, dx in zip(SCAFFOLDED, colors, jit, strict=True):
        ax.errorbar(
            [dx],
            [ss[c]["rank_k_at_90_mean"]],
            yerr=[[ss[c]["rank_k_at_90_std"]]] * 2,
            fmt="o",
            ms=8,
            color=col,
            capsize=3,
        )
    ax.errorbar(
        [1.0],
        [ss[BARE]["rank_k_at_90_mean"]],
        yerr=[[ss[BARE]["rank_k_at_90_std"]]] * 2,
        fmt="*",
        ms=16,
        color="black",
        capsize=3,
    )
    ax.axhspan(*SYCO_80ROW_BAND, color="tab:gray", alpha=0.25, label="sycophancy band (48–50)")
    ax.set_xticks([0, 1], ["scaffolded (7)", "bare"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(26, 54)
    ax.set_ylabel("rank-k@90, 80-row subsamples (mean ± sd)")
    ax.set_title("Matched-80-row rank\n(layer 14, own text)")
    ax.legend(fontsize=8, loc="upper left")

    # Panel C — shared-text rank at L14 (V4).
    ax = axes[2]
    for (c, lab), col, dx in zip(SCAFFOLDED, colors, jit, strict=True):
        ax.plot([dx], [rec(r_tf, c, "response", L14)["rank_k_at_90"]], "o", ms=8, color=col)
    ax.plot([1.0], [rec(r_tf, BARE, "response", L14)["rank_k_at_90"]], "*", ms=16, color="black")
    ax.axhline(COLLAPSE_BAR, ls="--", lw=1.2, color="black", label="collapse bar (30)")
    ax.axhspan(
        *SYCO_SHARED_SPAN, color="tab:gray", alpha=0.25, label="sycophancy shared span (27–35)"
    )
    ax.set_xticks([0, 1], ["scaffolded (7)", "bare"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(14, 38)
    ax.set_ylabel("rank-k@90, shared text")
    ax.set_title("Shared-text rank\n(layer 14, teacher-forced control)")
    ax.legend(fontsize=8, loc="upper left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=4, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "bare_vs_scaffold_L14", dir=FIG_DIR)
    plt.close(fig)


def fig_layers_overlay(own: dict) -> None:
    """Companion: bare per-layer profiles vs the scaffolded min-max envelope."""
    set_paper_style("blog")
    bare_col = "black"
    env_col = "tab:gray"
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    r = own["records"]

    def curves(dv: str) -> tuple[list[float], list[float], list[float]]:
        bare = [rec(r, BARE, "response", li)[dv] for li in LAYERS]
        per_cell = [[rec(r, c, "response", li)[dv] for li in LAYERS] for c, _ in SCAFFOLDED]
        lo = [min(vals[li] for vals in per_cell) for li in LAYERS]
        hi = [max(vals[li] for vals in per_cell) for li in LAYERS]
        return bare, lo, hi

    panels = [
        ("rank_k_at_90", "rank-k@90 of the shift cloud", "rank per layer", False, False),
        ("cos_mu_to_rb", "|cos(mean shift, read-out)|", "alignment per layer", True, False),
        ("mu_norm", "mean-shift norm (log scale)", "shift norm per layer", False, True),
    ]
    for ax, (dv, ylab, title, use_abs, logy) in zip(axes, panels, strict=True):
        bare, lo, hi = curves(dv)
        if use_abs:
            bare = [abs(v) for v in bare]
            lo2 = [min(abs(rec(r, c, "response", li)[dv]) for c, _ in SCAFFOLDED) for li in LAYERS]
            hi2 = [max(abs(rec(r, c, "response", li)[dv]) for c, _ in SCAFFOLDED) for li in LAYERS]
            lo, hi = lo2, hi2
        ax.fill_between(
            LAYERS, lo, hi, color=env_col, alpha=0.3, label="scaffolded envelope (7 cells, n=120)"
        )
        ax.plot(LAYERS, bare, color=bare_col, lw=2, marker="o", ms=3, label="bare context (n=100)")
        if use_abs:
            chance = [rec(r, BARE, "response", li)["random_cos_ci"]["ci_high"] for li in LAYERS]
            ax.plot(LAYERS, chance, color="gray", lw=1.2, alpha=0.8, label="chance bound (97.5%)")
        if logy:
            ax.set_yscale("log")
        ax.axvline(L14, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel("Decoder layer")
        ax.set_ylabel(ylab)
        ax.set_title(f"Own-text response arm: {title}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    savefig_paper(fig, "bare_layers_overlay", dir=FIG_DIR)
    plt.close(fig)


def main() -> int:
    own = _load(GEO_DIR / "geometry_per_cell.json")
    tf = _load(GEO_DIR / "geometry_tf_shared.json")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_hero(own, tf)
    fig_layers_overlay(own)
    b = rec(own["records"], BARE, "response", L14)
    print(
        json.dumps(
            {
                "bare_cos": b["cos_mu_to_rb"],
                "bare_rank_raw": b["rank_k_at_90"],
                "bare_mu": b["mu_norm"],
                "n": b["n_rows"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
