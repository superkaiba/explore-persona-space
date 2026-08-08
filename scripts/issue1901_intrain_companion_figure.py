"""Figure for the #1901 9a-ter in-train WildChat memorization companion.

Paired in-train vs held-out dumbbells per estimator: euclidean acc@1 at the
1,000-candidate targets-only pool (left panel) and pooled R2 clipped at -2
(right panel). Reads wildchat_intrain_companion.json + wildchat_arm.json;
saves via savefig_paper (png+pdf+meta.json sidecar).
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "eval_results" / "issue_1901" / "metric_battery"

ARM_LABELS: list[tuple[str, str]] = [
    ("const_mean", "Constant train-mean (963k)"),
    ("identity_copy", "Identity copy"),
    ("identity_bias", "Identity + learned bias (963k)"),
    ("ridge", "Linear map (ridge, 963k)"),
    ("mlp_w8192", "Small neural map (w8192)"),
    ("mlp_w32768", "Wide neural map (w32768)"),
    ("krr_nystrom", "Kernel map (Nystrom RBF)"),
    ("const_mean_3600", "Constant train-mean (3600)"),
    ("identity_bias_3600", "Identity + learned bias (3600)"),
    ("scaled_identity_3600", "Scaled identity (3600)"),
    ("diagonal_only_3600", "Per-dim rescale (3600)"),
    ("ridge_3600", "Linear map (ridge, 3600)"),
    ("ridge_n50_fixedlam", "Linear map (ridge, n=50, fixed lambda)"),
    ("identity_bias_n50", "Identity + learned bias (n=50)"),
]

R2_CLIP = -2.0


def main() -> None:
    intrain = json.loads((BATTERY / "wildchat_intrain_companion.json").read_text())
    heldout = json.loads((BATTERY / "wildchat_arm.json").read_text())

    set_paper_style("blog")
    pal = paper_palette_blog(4)
    c_held, c_intrain = pal[1], pal[2]

    fig, (ax_acc, ax_r2) = plt.subplots(1, 2, figsize=(10.5, 6.2), sharey=True)
    ys = list(range(len(ARM_LABELS)))

    for y, (slug, label) in zip(ys, ARM_LABELS):
        a_in = intrain["arms"][slug]
        a_ho = heldout["arms"][slug]
        acc_in = a_in["retrieval"]["test"]["euclidean"]["acc_at_k"]["1"]
        acc_ho = a_ho["retrieval"]["test"]["euclidean"]["acc_at_k"]["1"]
        r2_in = max(a_in["r2"]["point"], R2_CLIP)
        r2_ho = max(a_ho["r2"]["point"], R2_CLIP)

        ax_acc.plot([acc_ho, acc_in], [y, y], color="0.6", lw=1.2, zorder=1)
        ax_acc.scatter([acc_ho], [y], color=c_held, zorder=2, label="held-out" if y == 0 else None)
        ax_acc.scatter(
            [acc_in],
            [y],
            color=c_intrain,
            zorder=2,
            label="in-train (contaminated)" if y == 0 else None,
        )
        ax_r2.plot([r2_ho, r2_in], [y, y], color="0.6", lw=1.2, zorder=1)
        ax_r2.scatter([r2_ho], [y], color=c_held, zorder=2)
        ax_r2.scatter([r2_in], [y], color=c_intrain, zorder=2)

    labels = [lab for _, lab in ARM_LABELS]
    ax_acc.set_yticks(ys, labels)
    ax_acc.set_xlabel("acc@1 (euclid, targets-only pool)")
    ax_r2.set_xlabel(f"pooled R2 (clipped at {R2_CLIP:g})")
    ax_acc.axvline(0.001, color="0.3", ls=":", lw=1.0)
    ax_acc.legend(loc="lower right", frameon=False)
    fig.suptitle(
        "In-train vs held-out WildChat targets: retrieval inflates, "
        "variance explained barely moves (n=1000 each, layer 19)"
    )

    out = savefig_paper(
        fig, "wc_intrain_memorization_dumbbell", dir=ROOT / "figures" / "issue_1901"
    )
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
