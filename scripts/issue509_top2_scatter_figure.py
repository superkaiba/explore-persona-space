#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF003  # ρ + − + × in figure text intentional
"""#509 follow-up figure: scatter of the top-2 sycophancy-leakage predictor cells.

Two panels, one per cell:
  - last_prompt   x L07 x mmd    x centered (the bake-off global max)
  - end_of_system x L02 x cosine x centered (the runner-up)

x = the cell's pairwise activation distance between source and bystander
persona; y = the frozen #411 leakage target (bystander wrong-claim agree
rate, trained - base). 138 (source, bystander) cells, colored by source.

Data join + exclusions follow scripts/issue509_baserate_covariate_earlylayer.py
(same frozen target snapshot, same pinned HF revision for the distance
matrices, same persona->cid mapping). The annotated rho / perm p / CI are
read from eval_results/issue_509/syco_arm/scoring.json (the production
source-controlled, attenuation-adjusted values) -- NOT recomputed here;
a sanity assert checks the raw source-FE Spearman recomputed from the
joined arrays sits within 0.06 of scoring.json's rho_fe (the recompute
omits the length-partial step, so exact equality is not expected).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata as _scipy_rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

TARGET_FILE = PROJECT_ROOT / "eval_results/issue_480/_inputs/syco_411_analyze_summary.json"
SCORING_FILE = PROJECT_ROOT / "eval_results/issue_509/syco_arm/scoring.json"
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "1b6e20530b1c6d477a387c18d5a88554910e7df9"
METRICS_PREFIX = "issue_509/syco_arm/bakeoff/metrics"

# Alphabetical persona order -> SC1..SC24 (i509_syco_conditions convention,
# branch issue-509 @ 2d22b70c).
_PERSONA_ORDER = sorted(
    [
        "accountant",
        "ai",
        "ai_assistant",
        "assistant",
        "chef",
        "child",
        "comedian",
        "data_scientist",
        "french_person",
        "hero",
        "journalist",
        "kindergarten_teacher",
        "lawyer",
        "librarian",
        "medical_doctor",
        "philosopher",
        "police_officer",
        "programmer",
        "qwen_default",
        "software_engineer",
        "surgeon",
        "villain",
        "wizard",
        "zelthari_scholar",
    ]
)
PERSONA_TO_CID = {p: f"SC{i}" for i, p in enumerate(_PERSONA_ORDER, start=1)}

SOURCE_LABELS = {
    "villain": "Villain",
    "comedian": "Comedian",
    "assistant": "Assistant",
    "qwen_default": "Qwen default",
    "software_engineer": "Software engineer",
    "kindergarten_teacher": "Kindergarten teacher",
}
SOURCE_ORDER = [
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
]

CELLS = [
    {
        "point": "last_prompt",
        "layer": 7,
        "metric": "mmd",
        "variant": "centered",
        "panel_title": "Best cell: MMD, layer 7 (last prompt token)",
        "xlabel": "MMD distance between persona activation clouds",
    },
    {
        "point": "end_of_system",
        "layer": 2,
        "metric": "cosine",
        "variant": "centered",
        "panel_title": "Runner-up: cosine, layer 2 (end of system prompt)",
        "xlabel": "Cosine distance between persona activation clouds",
    },
]


def load_target_rows() -> list[dict]:
    """The 138 off-diagonal (source, bystander) leakage cells from the frozen #411 snapshot."""
    snap = json.loads(TARGET_FILE.read_text())
    rows = []
    for source, src_data in snap["per_source"].items():
        for bystander, delta in src_data["per_panel_delta"].items():
            if bystander == source:
                continue
            rows.append({"source": source, "bystander": bystander, "delta": float(delta)})
    if len(rows) != 138:
        raise ValueError(f"Expected 138 off-diagonal cells, got {len(rows)}")
    return rows


def load_matrix(point: str, layer: int, metric: str, variant: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        HF_REPO,
        f"{METRICS_PREFIX}/{point}__layer{layer}__{metric}__{variant}.json",
        repo_type="dataset",
        revision=HF_REVISION,
    )
    matrix = json.loads(Path(path).read_text())["matrix"]
    if matrix is None:
        raise ValueError("matrix is null — wrong cell name")
    return matrix


def join(matrix: dict, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom, delta, src = [], [], []
    for row in rows:
        d = matrix[PERSONA_TO_CID[row["source"]]][PERSONA_TO_CID[row["bystander"]]]
        if d is None or not np.isfinite(d) or abs(d) < 1e-6:
            raise ValueError(
                f"unexpected excluded pair {row} — both cells score n=138 with 0 exclusions"
            )
        geom.append(float(d))
        delta.append(row["delta"])
        src.append(row["source"])
    return np.array(geom), np.array(delta), np.array(src)


def fe_spearman(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Source-FE Spearman (rank globally, demean within source) — sanity check only."""

    def within(v):
        v = v.copy()
        for g in np.unique(groups):
            m = groups == g
            v[m] -= v[m].mean()
        return v

    rx = within(_scipy_rankdata(x) - 1.0)
    ry = within(_scipy_rankdata(y) - 1.0)
    return float(np.corrcoef(rx, ry)[0, 1])


def cell_stats(scoring: dict, point: str, layer: int, metric: str, variant: str) -> dict:
    for c in scoring["cells"]:
        if (c["extraction_point"], c["layer"], c["metric"], c["variant"]) == (
            point,
            layer,
            metric,
            variant,
        ):
            return c
    raise KeyError(f"{point} L{layer} {metric} {variant} not in scoring.json")


def main() -> None:
    rows = load_target_rows()
    scoring = json.loads(SCORING_FILE.read_text())

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True, constrained_layout=True)
    colors = {s: paper_palette(len(SOURCE_ORDER))[i] for i, s in enumerate(SOURCE_ORDER)}

    for ax, cell in zip(axes, CELLS, strict=True):
        matrix = load_matrix(cell["point"], cell["layer"], cell["metric"], cell["variant"])
        geom, delta, src = join(matrix, rows)
        stats = cell_stats(scoring, cell["point"], cell["layer"], cell["metric"], cell["variant"])

        # Join sanity check: a wrong persona->cid mapping would destroy the
        # correlation entirely. The simple rank-then-demean Spearman here
        # intentionally omits scoring.json's length-partial step (the
        # baserate-covariate convention), so the two differ by ~0.03 — the
        # 0.06 tolerance guards the join, not the convention.
        rho_check = fe_spearman(geom, delta, src)
        assert abs(rho_check - stats["rho_fe"]) < 0.06, (rho_check, stats["rho_fe"])

        for s in SOURCE_ORDER:
            m = src == s
            ax.scatter(
                geom[m],
                delta[m],
                color=colors[s],
                s=26,
                alpha=0.8,
                edgecolor="white",
                lw=0.5,
                label=SOURCE_LABELS[s],
            )
        ax.set_title(cell["panel_title"], fontsize=10)
        ax.set_xlabel(cell["xlabel"])
        ax.axhline(0.0, color="grey", lw=0.8, ls=":", zorder=0)
        ax.text(
            0.97,
            0.97,
            f"source-controlled ρ = {stats['rho_fe_adj']:.2f}\n"
            f"perm p = {stats['perm_p_fe']:.4f}, n = {stats['n']}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey"),
        )

    axes[0].set_ylabel("Bystander wrong-claim agree rate,\ntrained − base")
    axes[0].legend(loc="lower left", fontsize=8, title="Source persona", title_fontsize=8)
    fig.suptitle(
        "Early-layer activation distance predicts sycophancy leakage",
        x=0.01,
        ha="left",
        fontweight="semibold",
        fontsize=13,
        color="#1A1A1A",
    )

    savefig_paper(fig, "issue_509/top2_predictor_scatter", dir="figures/")
    plt.close(fig)
    print("saved figures/issue_509/top2_predictor_scatter.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
