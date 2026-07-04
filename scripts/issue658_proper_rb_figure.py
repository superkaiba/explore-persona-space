"""Grouped-bar figure: crude r_B vs proper r_B(A) vs ridge ceiling (DIRECT chain),
best-layer |rho|, per behavior. #658/#722 behavior-decoding chain with #661 r_B."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(REPO / "src"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style  # noqa: E402

R = json.load(open(REPO / "eval_results/issue_658/behavior_chain_proper_rb.json"))
BEH = ["sycophancy", "refusal", "broad_em"]
LABELS = {"sycophancy": "Sycophancy", "refusal": "Refusal", "broad_em": "Broad EM"}
CEIL = R["_ridge_ceiling_cited"]


def main() -> int:
    set_paper_style()
    colors = paper_palette(3)
    c_crude, c_proper, c_ceil = colors[0], colors[1], colors[2]

    # DIRECT chain, best-layer by largest |rho| (task convention); bar = |rho|.
    crude = [abs(R[b]["crude"]["direct_best_abs"]["rho"]) for b in BEH]
    proper = [abs(R[b]["proper_a"]["direct_best_abs"]["rho"]) for b in BEH]
    ceil = [CEIL[b] for b in BEH]

    x = np.arange(len(BEH))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar(x - w, crude, w, label="crude r_B (diff-of-means, #658)", color=c_crude)
    b2 = ax.bar(x, proper, w, label="proper r_B (persona-vectors A, #661)", color=c_proper)
    b3 = ax.bar(x + w, ceil, w, label="ridge readout (decoder ceiling)", color=c_ceil, alpha=0.85)

    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}",
                (rect.get_x() + rect.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 1),
                textcoords="offset points",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[b] for b in BEH])
    ax.set_ylabel(r"best-layer $|\rho|$  (predicted vs. judged $E_0$, LOCO, n=50)")
    ax.set_title("DIRECT behavior-decoding chain: r_B extraction quality vs. ridge ceiling")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = REPO / "figures/issue_658/behavior_chain_proper_rb.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
