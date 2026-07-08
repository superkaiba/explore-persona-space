"""Figure for the #811 F1 offset-decomposition follow-up (zero-GPU, VM CPU).

Two panels from ``eval_results/issue_811/offset_decomposition.json``:

Left  — the 16 signed per-context map changes delta(c) (projection of
        M+(c) - M0(c) on the unit behavior direction) for three key cells,
        one row per context, with each cell's grid-mean offset as a dashed
        line in the matching colour.
Right — Delta_med / combined floor before (open markers) vs after (filled)
        removing the grid-constant offset, for the nine cells whose raw
        Delta cleared the floor.

Usage (from the issue-811 worktree root):
    uv run python scripts/issue811_f1_offset_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

RESULT_JSON = Path("eval_results/issue_811/offset_decomposition.json")

BEHAVIOR_LABELS = {
    "em": "harmful-compliance",
    "sycophancy": "sycophancy",
    "fact": "taught fact",
}
SUMMARY_LABELS = {"mean": "mean", "turn_nl": "turn boundary"}

# Plain-English context names, in a fixed semantic order (top to bottom).
CONTEXT_ORDER = [
    ("binst", "behavior-naming instruction"),
    ("default", "default assistant"),
    ("fmt_code", "code-format instruction"),
    ("fmt_json", "JSON-format instruction"),
    ("icl_k2", "2-shot demonstrations"),
    ("icl_k8", "8-shot demonstrations"),
    ("reph_casual", "casual rephrase"),
    ("reph_imp", "imperative rephrase"),
    ("reph_polite", "polite rephrase"),
    ("sp_doctor", "doctor persona"),
    ("sp_ph1", "PersonaHub persona 1"),
    ("sp_ph2", "PersonaHub persona 2"),
    ("sp_swe", "software-engineer persona"),
    ("wc_long_write", "WildChat long writing"),
    ("wc_short_advice", "WildChat short advice"),
    ("wc_short_code", "WildChat short code"),
]

# The three cells whose per-context structure carries the story.
PANEL_A_CELLS = [
    ("em/L14/turn_nl", "harmful-compliance · L14 · turn boundary", "primary"),
    ("em/L7/turn_nl", "harmful-compliance · L7 · turn boundary", "accent"),
    ("fact/L14/mean", "taught fact · L14 · mean", "baseline"),
]

UNTRUSTED_CELLS = {"sycophancy/L7/turn_nl"}


def _context_value(per_context: dict[str, float], slug_prefix: str) -> float:
    """Look up a context's value, tolerating the behavior-specific binst key."""
    if slug_prefix == "binst":
        for key, val in per_context.items():
            if key.startswith("binst"):
                return val
        raise KeyError("no binst_* key in per-context dict")
    return per_context[slug_prefix]


def main() -> None:
    data = json.loads(RESULT_JSON.read_text())
    cells = data["cells"]

    set_paper_style("blog")
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # ---------------- Panel A: per-context signed delta(c) ----------------
    n_ctx = len(CONTEXT_ORDER)
    dodge = [-0.26, 0.0, 0.26]
    for (cell_key, label, role), dy in zip(PANEL_A_CELLS, dodge, strict=False):
        cell = cells[cell_key]
        color = paper_palette_role(role)
        xs = [_context_value(cell["delta_per_context"], slug) for slug, _ in CONTEXT_ORDER]
        ys = [n_ctx - 1 - i + dy for i in range(n_ctx)]
        ax_a.scatter(xs, ys, s=34, color=color, label=label, zorder=3)
        ax_a.axvline(
            cell["offset"], color=color, linestyle="--", linewidth=1.1, alpha=0.75, zorder=2
        )
    ax_a.axvline(0.0, color="#5A5A5A", linewidth=1.0, zorder=1)
    ax_a.set_yticks([n_ctx - 1 - i for i in range(n_ctx)])
    ax_a.set_yticklabels([name for _, name in CONTEXT_ORDER])
    ax_a.set_ylim(-0.7, n_ctx - 0.3)
    ax_a.set_xlim(-0.52, 0.92)
    ax_a.set_xlabel("signed per-context map change (projection on the behavior direction)")
    ax_a.set_title("Per-context map change, three key cells\n(dashed line = grid-mean offset)")
    ax_a.legend(loc="center right", fontsize=8)

    # ------------- Panel B: raw vs residual Delta/floor dumbbell -------------
    above_floor = [(key, cell) for key, cell in cells.items() if cell["ratio_raw"] >= 1.0]
    above_floor.sort(key=lambda kv: kv[1]["ratio_raw"])
    raw_color = "#3B3B3B"
    resid_color = paper_palette_role("primary")
    labels = []
    for row, (key, cell) in enumerate(above_floor):
        untrusted = key in UNTRUSTED_CELLS
        alpha = 0.45 if untrusted else 1.0
        beh = BEHAVIOR_LABELS[cell["behavior"]]
        summ = SUMMARY_LABELS[cell["summary"]]
        name = f"{beh} · L{cell['layer']} · {summ}"
        if untrusted:
            name += " (untrusted)"
        labels.append(name)
        ax_b.plot(
            [cell["ratio_raw"], cell["ratio_residual"]],
            [row, row],
            color="#B0B0B0",
            linewidth=1.2,
            alpha=alpha,
            zorder=1,
        )
        ax_b.scatter(
            [cell["ratio_raw"]],
            [row],
            s=52,
            facecolors="none",
            edgecolors=raw_color,
            linewidths=1.4,
            alpha=alpha,
            zorder=3,
        )
        ax_b.scatter(
            [cell["ratio_residual"]],
            [row],
            s=52,
            color=resid_color,
            alpha=alpha,
            zorder=3,
        )
    ax_b.axvline(1.0, color="#5A5A5A", linestyle="--", linewidth=1.0, zorder=1)
    ax_b.set_yticks(range(len(above_floor)))
    ax_b.set_yticklabels(labels)
    ax_b.set_xlabel("median map change ÷ combined refit floor")
    ax_b.set_title(
        "The nine above-floor cells, before vs after offset removal\n"
        "(open = raw, filled = residual; dashed line = floor)"
    )
    ax_b.scatter([], [], s=52, facecolors="none", edgecolors=raw_color, linewidths=1.4, label="raw")
    ax_b.scatter([], [], s=52, color=resid_color, label="residual (offset removed)")
    ax_b.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    paths = savefig_paper(fig, "issue_811/f1_offset_decomposition", dir="figures/")
    plt.close(fig)
    for fmt, path in paths.items():
        print(f"{fmt}: {path}")


if __name__ == "__main__":
    main()
