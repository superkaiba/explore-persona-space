"""Round-3 real-u2-capture figures for issue #1689.

Two figures:
  fig14_real_vs_haiku_prefix_r2.png — paired prefix-arm R² per (model × framing),
    real vs haiku on the SAME 3,800 LMSYS+WildChat conversations. This is the
    HERO figure of the round.
  fig15_real_u2_r2_lattice.png — 12-cell lattice: prefix + context R² per unit,
    with context-invalid arms hatched. Companion / low-level per-unit view.
"""

from __future__ import annotations


# Call load_dotenv() at module top BEFORE importing heavy libs (matplotlib/numpy)
# so orchestrate.env's thread-cap setdefaults land in-process before those libs
# freeze their pools. Required by tests/test_shared_vm_thread_caps.py.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "eval_results/issue_1689/real_u2_capture/per_cell_summary.json"
OUT_DIR = "issue_1689"

MODEL_SHORT = {
    "Qwen/Qwen2.5-7B": "Qwen 2.5-7B (base)",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5-7B (instruct)",
}
FRAMING_ORDER = ["chat", "naturalistic", "story"]
FRAMING_LABEL = {
    "chat": "chat template",
    "naturalistic": "plain text",
    "story": "narrative story",
}


def load_cells():
    return json.loads(SUMMARY.read_text())["cells"]


def _cell(cells, model, framing, provenance):
    for c in cells:
        if c["model"] == model and c["framing"] == framing and c["provenance"] == provenance:
            return c
    raise KeyError((model, framing, provenance))


def fig14_hero(cells):
    """Grouped bar chart: prefix-arm R², real vs haiku, per (model × framing)."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0), sharey=True)

    colors = {
        "realu2": paper_palette_role("primary"),
        "haikuu2": paper_palette_role("baseline"),
    }
    provenance_label = {
        "realu2": "real LMSYS/WildChat u2",
        "haikuu2": "Haiku-simulated u2",
    }

    x_positions = np.arange(len(FRAMING_ORDER))
    width = 0.36

    for ax, model in zip(axes, MODEL_SHORT):
        real_vals = []
        haiku_vals = []
        for framing in FRAMING_ORDER:
            real_vals.append(_cell(cells, model, framing, "realu2")["prefix_arm"]["r2_obs"])
            haiku_vals.append(_cell(cells, model, framing, "haikuu2")["prefix_arm"]["r2_obs"])

        ax.bar(
            x_positions - width / 2,
            real_vals,
            width,
            color=colors["realu2"],
            label=provenance_label["realu2"] if model == "Qwen/Qwen2.5-7B" else None,
        )
        ax.bar(
            x_positions + width / 2,
            haiku_vals,
            width,
            color=colors["haikuu2"],
            label=provenance_label["haikuu2"] if model == "Qwen/Qwen2.5-7B" else None,
        )

        # Value labels above bars
        for i, (r, h) in enumerate(zip(real_vals, haiku_vals)):
            ax.text(i - width / 2, r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        # Reference: null p97.5 band top (~-0.03; keep visible near zero)
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="-", alpha=0.5)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([FRAMING_LABEL[f] for f in FRAMING_ORDER])
        ax.set_ylim(-0.05, 0.6)
        ax.set_title(MODEL_SHORT[model], loc="left", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("prefix-arm held-out R²\n(second-user-turn map)")

    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        "Prefix-arm second-user-turn R² per framing: real vs Haiku-simulated u2",
        fontsize=11,
        y=0.98,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.925,
        "Same 3,800 LMSYS+WildChat conversations, both provenances; L19 prefix→u2 ridge.",
        fontsize=8.5,
        ha="left",
        color="#444",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])

    savefig_paper(fig, f"{OUT_DIR}/fig14_real_vs_haiku_prefix_r2", dir="figures/")
    plt.close(fig)


def fig15_lattice(cells):
    """12-cell lattice: prefix + context R², with construct-invalid context arms hatched."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharey=True)

    colors = {
        "prefix": paper_palette_role("primary"),
        "context": paper_palette_role("control"),
    }

    # Column order: (framing, provenance)
    columns = [
        ("chat", "realu2", "chat / real"),
        ("chat", "haikuu2", "chat / haiku"),
        ("naturalistic", "realu2", "plain-text / real"),
        ("naturalistic", "haikuu2", "plain-text / haiku"),
        ("story", "realu2", "story / real"),
        ("story", "haikuu2", "story / haiku"),
    ]
    n = len(columns)
    x_positions = np.arange(n)
    width = 0.36

    for ax, model in zip(axes, MODEL_SHORT):
        pfx_vals = []
        ctx_vals = []
        ctx_invalid = []
        for framing, prov, _ in columns:
            c = _cell(cells, model, framing, prov)
            pfx_vals.append(c["prefix_arm"]["r2_obs"])
            ctx_vals.append(c["context_arm"]["r2_obs"])
            ctx_invalid.append(c["context_arm"].get("construct_invalid", False))

        bars_pfx = ax.bar(
            x_positions - width / 2,
            pfx_vals,
            width,
            color=colors["prefix"],
            label="prefix arm",
        )
        # Context bars: draw all, then overlay a hatch on invalid ones
        bars_ctx = ax.bar(
            x_positions + width / 2,
            ctx_vals,
            width,
            color=colors["context"],
            label="context arm",
        )
        for i, invalid in enumerate(ctx_invalid):
            if invalid:
                bars_ctx[i].set_hatch("///")
                bars_ctx[i].set_edgecolor("#a83232")
                bars_ctx[i].set_linewidth(1.0)

        for i, (p, c_) in enumerate(zip(pfx_vals, ctx_vals)):
            ax.text(i - width / 2, p + 0.015, f"{p:.2f}", ha="center", va="bottom", fontsize=7.5)
            ax.text(i + width / 2, c_ + 0.015, f"{c_:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.axhline(0.0, color="gray", linewidth=0.5, alpha=0.4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([lbl for _, _, lbl in columns], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(MODEL_SHORT[model], loc="left", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("held-out R² (L19)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["prefix"], label="prefix arm"),
        plt.Rectangle((0, 0), 1, 1, color=colors["context"], label="context arm"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=colors["context"],
            hatch="///",
            edgecolor="#a83232",
            label="context arm (construct-invalid)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8.5,
    )
    fig.suptitle(
        "Per-cell prefix and context R² over the 12 real-u2 capture cells",
        fontsize=11,
        y=0.98,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.925,
        "3 framings × 2 models × real vs Haiku; hatched context bars = plain-text framing "
        "(user prefix collapses onto answer end, self-prediction).",
        fontsize=8.5,
        ha="left",
        color="#444",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.9])

    savefig_paper(fig, f"{OUT_DIR}/fig15_real_u2_r2_lattice", dir="figures/")
    plt.close(fig)


def main() -> int:
    cells = load_cells()
    fig14_hero(cells)
    fig15_lattice(cells)
    print(f"Wrote fig14 + fig15 under figures/{OUT_DIR}/")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
