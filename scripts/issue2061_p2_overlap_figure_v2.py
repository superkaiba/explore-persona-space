"""Re-render the #2061 P2 cross-corpus top-100 overlap heatmap grid (v2).

Supersedes ``figures/issue_2061/i2061_p2_cross_corpus_overlap.png`` (committed
67b22aeab5): the v1 suptitle framed the overlaps against the naive
100/262,144 dictionary-wide chance, which the round's code review flagged as
materially understating the realistic pool-conditional null (only ~800-12,800
features per cell have finite improved delta-R2, so expected overlap is
percent-scale). v2 reads the COMMITTED round JSON only (no raw JSONL
re-parse), drops the chance claim from the rendered text, and uses
plain-English cell labels per the paper-plots skill section 3.5.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

REPO = Path(__file__).resolve().parents[1]
P2_JSON = (
    REPO / "eval_results/issue_2061/followup_free_analysis/p2_cross_corpus_rank_agreement.json"
)

CELL_LABELS = {
    "chat_gsm8k_train_full": "GSM8K",
    "chat_if11k": "Instruction-following",
    "chat_lmsys23k": "LMSYS (chat)",
    "chat_math7500": "MATH",
    "chat_sft11k": "Tülu SFT mix",
    "chat_uf11k": "UltraFeedback",
    "naturalistic_lmsys23k": "LMSYS (naturalistic)",
}
PAIR_TITLES = {
    "base_sft": "base → SFT",
    "sft_dpo": "SFT → DPO",
    "dpo_rlvr": "DPO → RLVR",
    "rlvr_longer-rlvr": "RLVR → longer-RLVR",
}


def main() -> None:
    data = json.loads(P2_JSON.read_text())
    transitions = data["transitions"]
    assert len(transitions) == 4, f"expected 4 transitions, got {len(transitions)}"

    set_paper_style("blog")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.5), constrained_layout=True)

    ims = []
    for ax, tr in zip(axes.flat, transitions, strict=True):
        cells = tr["cells"]
        assert len(cells) == 7, f"{tr['pair']}: expected 7 cells, got {len(cells)}"
        keys = [f"{c['render']}_{c['corpus']}" for c in cells]
        labels = [CELL_LABELS[k] for k in keys]  # KeyError = fail loud on unknown cell
        idx = {k: i for i, k in enumerate(keys)}

        mat = np.full((7, 7), np.nan)
        pairs = tr["pairs"]
        assert len(pairs) == 21, f"{tr['pair']}: expected 21 pairs, got {len(pairs)}"
        for p in pairs:
            i, j = idx[p["cell_a"]], idx[p["cell_b"]]
            mat[i, j] = mat[j, i] = p["overlap_frac"]

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")  # diagonal (self-pairs) masked, not computed
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=0.75)
        ims.append(im)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(PAIR_TITLES[tr["pair"]])
        ax.grid(False)

    fig.suptitle(
        "Overlap of the 100 most-improved SAE features between corpora, per transition"
        " (context arm, layer 29)"
    )
    cbar = fig.colorbar(ims[0], ax=axes, shrink=0.75)
    cbar.set_label("fraction of top-100 lists shared")

    savefig_paper(fig, "issue_2061/i2061_p2_cross_corpus_overlap_v2", dir=str(REPO / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    main()
