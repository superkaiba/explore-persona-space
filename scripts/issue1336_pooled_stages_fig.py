"""Pooled-across-stages summary figure (#1336 scope extension).

Two panels over the pooled-stages battery JSONs
(eval_results/issue_1336/pooled_stages/, one per v2 surface):

  top     MATCHED-N pooled map (pooled train subsampled to one stage's train
          n — the claim-carrying control) scored per stage as-is / + bias /
          + rotation / + rotation+scale, against the stage's OWN-map ceiling.
  bottom  LEAVE-BASE-OUT pooled map (fit on the 4 post-training stages only),
          same rungs, same ceiling.

x = the 5 Tulu-3 ladder stages (base shaded — the pooled map's hardest
transfer target); y = held-out fold-local pooled R^2. Small open dots are the
7 non-excluded surfaces (gsm8k_test1319 excluded: own/matched-n fits are
n < d by design); large filled markers are the median over surfaces. Nothing
is fitted here.

    uv run python scripts/issue1336_pooled_stages_fig.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the matplotlib import.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "eval_results" / "issue_1336" / "pooled_stages"
OUTDIR = REPO / "figures" / "issue_1336" / "pooled_stages"

STAGES = ["base", "sft", "dpo", "rlvr", "rlvr_long"]
STAGE_LABEL = {
    "base": "base",
    "sft": "SFT",
    "dpo": "DPO",
    "rlvr": "RLVR-PPO",
    "rlvr_long": "RLVR-GRPO",
}
PANEL_VARIANT = [
    ("matchedn", "matched-n pooled map (5 stages, train n matched to one stage)"),
    ("lofo", "leave-base-out pooled map (fit on the 4 post-training stages)"),
]
RUNG_STYLE = [
    ("", "pooled as-is", "#1F77B4"),
    ("_bias", "+ per-stage bias", "#FF7F0E"),
    ("_rot", "+ per-stage rotation", "#2CA02C"),
    ("_rot_scale", "+ rotation + scale", "#9467BD"),
]


def _load() -> list[dict]:
    recs = []
    for fp in sorted(SRC.glob("pooled_*.json")):
        d = json.load(open(fp))
        assert d["status"] == "complete", fp
        if d.get("exclude_from_aggregates"):
            continue
        recs.append(d)
    assert len(recs) == 7, [r["corpus"] for r in recs]
    return recs


def main() -> None:
    """Render the two-panel pooled-vs-own R^2 plot over the 7 kept surfaces."""
    recs = _load()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True, sharey=True)
    xs = range(len(STAGES))
    off = {"": -0.27, "_bias": -0.09, "_rot": 0.09, "_rot_scale": 0.27}
    for ax, (variant, title) in zip(axes, PANEL_VARIANT):
        ax.axvspan(-0.5, 0.5, color="0.93", zorder=0)  # base stage block
        for x, s in zip(xs, STAGES):
            own = [r["r2_pooled_foldlocal"][s]["own"] for r in recs]
            ax.plot(
                [x - 0.38, x + 0.38],
                [statistics.median(own)] * 2,
                color="0.25",
                lw=2.0,
                zorder=3,
            )
            for suffix, _, color in RUNG_STYLE:
                vals = [r["r2_pooled_foldlocal"][s][f"{variant}{suffix}"] for r in recs]
                xo = x + off[suffix]
                ax.plot([xo] * len(vals), vals, "o", ms=3, mfc="none", mec=color, mew=0.9, zorder=2)
                ax.plot([xo], [statistics.median(vals)], "o", ms=7, color=color, zorder=4)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("held-out pooled $R^2$", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    for suffix, label, color in RUNG_STYLE:
        axes[0].plot([], [], "o", color=color, label=label)
    axes[0].plot([], [], color="0.25", lw=2.0, label="own-map ceiling (median)")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels([STAGE_LABEL[s] for s in STAGES], fontsize=9)
    fig.suptitle(
        "One pooled context→answer map across training stages vs per-stage maps (layer 30)",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUTDIR / "pooled_vs_own.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out)


if __name__ == "__main__":
    main()
