"""Issue #186 train × eval accuracy-loss heatmaps.

6 training conditions × 4 eval scaffolds = 24 cells per metric.
Two panels:
- Left: source-persona accuracy loss (baseline_acc(source) − trained_acc(source))
- Right: bystander mean accuracy loss (mean over 10 non-source personas)

Both macros computed across 4 sources × 3 seeds. Diagonal-ish cells (matched
scaffolds) carry the bulk of the wrong-answer effect; off-diagonal cells
visualize scaffold-mismatch attenuation.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
BASE_280_WT = ROOT / ".claude/worktrees/issue-280/eval_results/issue280"
BASE_186_WT = ROOT / ".claude/worktrees/issue-280/eval_results/issue186"
BASE_186_MAIN = ROOT / "eval_results/issue186"

ASSISTANT_COSINES = [
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "zelthari_scholar",
    "police_officer",
]
SOURCES = ["software_engineer", "librarian", "comedian", "police_officer"]
SEEDS = (42, 137, 256)

TRAIN_CONDITIONS = [
    "no_cot",
    "garbage_cot",
    "scrambled_english_cot",
    "generic_cot",
    "persona_cot",
    "contradicting_cot",
]
TRAIN_LABELS = {
    "no_cot": "no\nchain-of-thought",
    "garbage_cot": "garbage\ntokens",
    "scrambled_english_cot": "scrambled-\nEnglish",
    "generic_cot": "neutral\nchain-of-thought",
    "persona_cot": "persona-flavored\nchain-of-thought",
    "contradicting_cot": "contradicting\nrationale",
}
EVAL_SCAFFOLDS = ["no_cot", "generic_cot", "persona_cot", "empty_persona_cot_eval"]
EVAL_LABELS = {
    "no_cot": "no chain-of-thought\neval",
    "generic_cot": "neutral chain-of-thought\neval",
    "persona_cot": "persona-flavored\neval",
    "empty_persona_cot_eval": "empty-tag\neval",
}

# Which (train, eval) cells are matched-scaffold pairs (training tag matches eval tag)
MATCHED_PAIRS = {
    ("no_cot", "no_cot"),
    ("generic_cot", "generic_cot"),
    ("persona_cot", "persona_cot"),
    ("garbage_cot", "generic_cot"),
    ("scrambled_english_cot", "generic_cot"),
    ("contradicting_cot", "persona_cot"),
}


def _baselines():
    with (BASE_280_WT / "baseline" / "result.json").open() as f:
        b280 = json.load(f)
    with (BASE_186_MAIN / "baseline" / "result.json").open() as f:
        b186 = json.load(f)
    return b186, b280


def _cell_path(arm: str, source: str, seed: int) -> Path:
    if arm == "no_cot":
        return BASE_186_MAIN / f"{source}_{arm}_seed{seed}" / "result.json"
    if arm in ("generic_cot", "persona_cot"):
        wt = BASE_186_WT / f"{source}_{arm}_seed{seed}" / "result.json"
        return wt if wt.exists() else BASE_186_MAIN / f"{source}_{arm}_seed{seed}" / "result.json"
    return BASE_280_WT / f"{source}_{arm}_seed{seed}" / "result.json"


def _baseline_for(arm: str, b186, b280):
    return b186 if arm == "no_cot" else b280


def compute_matrix(b186, b280, axis: str):
    """Returns 6×4 array. axis ∈ {'source', 'bystander'}."""
    mat = np.full((len(TRAIN_CONDITIONS), len(EVAL_SCAFFOLDS)), np.nan)
    for ti, arm in enumerate(TRAIN_CONDITIONS):
        base = _baseline_for(arm, b186, b280)
        for ei, eval_arm in enumerate(EVAL_SCAFFOLDS):
            macros = []
            for src in SOURCES:
                seed_vals = []
                for seed in SEEDS:
                    p = _cell_path(arm, src, seed)
                    if not p.exists():
                        continue
                    with p.open() as f:
                        d = json.load(f)
                    if axis == "source":
                        try:
                            tr = d["per_persona"][src][eval_arm]["accuracy"]
                            seed_vals.append(base["per_persona"][src][eval_arm]["accuracy"] - tr)
                        except KeyError:
                            pass
                    else:  # bystander
                        bys = [p for p in ASSISTANT_COSINES if p != src]
                        per = []
                        for bp in bys:
                            try:
                                tr_b = d["per_persona"][bp][eval_arm]["accuracy"]
                                per.append(base["per_persona"][bp][eval_arm]["accuracy"] - tr_b)
                            except KeyError:
                                continue
                        if per:
                            seed_vals.append(statistics.mean(per))
                if seed_vals:
                    macros.append(statistics.mean(seed_vals))
            if macros:
                mat[ti, ei] = statistics.mean(macros)
    return mat


def main():
    set_paper_style("blog")
    b186, b280 = _baselines()
    src_mat = compute_matrix(b186, b280, "source")
    by_mat = compute_matrix(b186, b280, "bystander")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)

    # Shared color scale across both panels
    vmax = max(np.nanmax(np.abs(src_mat)), np.nanmax(np.abs(by_mat)))
    vmin = -vmax

    for ax, mat, title in zip(
        axes, [src_mat, by_mat], ["Source-persona accuracy loss", "Bystander mean accuracy loss"]
    ):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(EVAL_SCAFFOLDS)))
        ax.set_xticklabels([EVAL_LABELS[s] for s in EVAL_SCAFFOLDS], fontsize=8, rotation=0)
        ax.set_yticks(range(len(TRAIN_CONDITIONS)))
        ax.set_yticklabels([TRAIN_LABELS[a] for a in TRAIN_CONDITIONS], fontsize=8)
        ax.set_xlabel("eval scaffold")
        ax.set_ylabel("training condition")
        ax.set_title(title, fontsize=10)

        # Annotate each cell with its value; bold the matched-scaffold cells
        for ti, arm in enumerate(TRAIN_CONDITIONS):
            for ei, eval_arm in enumerate(EVAL_SCAFFOLDS):
                v = mat[ti, ei]
                if np.isnan(v):
                    txt = "—"
                else:
                    txt = f"{v:+.3f}"
                matched = (arm, eval_arm) in MATCHED_PAIRS
                color = "white" if abs(v) > 0.12 else "black"
                weight = "bold" if matched else "normal"
                ax.text(
                    ei,
                    ti,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=color,
                    fontweight=weight,
                )

        # Draw a thin black border around matched-scaffold cells
        for ti, arm in enumerate(TRAIN_CONDITIONS):
            for ei, eval_arm in enumerate(EVAL_SCAFFOLDS):
                if (arm, eval_arm) in MATCHED_PAIRS:
                    ax.add_patch(
                        plt.Rectangle(
                            (ei - 0.5, ti - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.5
                        )
                    )

        fig.colorbar(im, ax=ax, shrink=0.85, label="accuracy loss\n(baseline − trained)")

    fig.suptitle(
        "All 6 training conditions × 4 eval scaffolds — matched cells (bordered) carry the effect, "
        "scaffold-mismatch attenuates",
        fontsize=10,
        y=1.02,
    )

    savefig_paper(fig, "issue186/train_eval_heatmap", dir="figures/")
    plt.close(fig)

    print("=== Train × Eval source-loss macro (4 sources × 3 seeds) ===")
    print(f"{'train ↓ / eval →':<26}" + "".join(f"{EVAL_SCAFFOLDS[i]:>20}" for i in range(4)))
    for ti, arm in enumerate(TRAIN_CONDITIONS):
        row = "".join(f"{src_mat[ti, ei]:>20.3f}" for ei in range(4))
        print(f"{arm:<26}{row}")
    print("\n=== Train × Eval bystander-mean-loss macro ===")
    print(f"{'train ↓ / eval →':<26}" + "".join(f"{EVAL_SCAFFOLDS[i]:>20}" for i in range(4)))
    for ti, arm in enumerate(TRAIN_CONDITIONS):
        row = "".join(f"{by_mat[ti, ei]:>20.3f}" for ei in range(4))
        print(f"{arm:<26}{row}")


if __name__ == "__main__":
    main()
