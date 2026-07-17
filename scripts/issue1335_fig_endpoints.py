"""Endpoints-vs-committed dot plot for issue #1335 (round-3 regeneration).

This run's two endpoints (full n, layer 19, context arm) plotted against the
prior committed reads they descend from (#825 / #1092 assistant maps, #1310
fiction per-character maps).

Round-3 change (interp-critique r2, finding 2): the fiction-instruct prior
square now plots the midpoint of #1310's committed four-persona band
0.166-0.253 (= 0.2095, including #1310's own 2026-07-16 completion of the
villain instruct cell at 0.166), replacing the 0.2253 point (the per-persona
mean of the parent's pre-completion three-persona cells), so both fiction
prior squares are band midpoints as labeled.

This-run values are derived from eval_results/issue_1335 cell JSONs; prior
committed reads are constants quoted from the parent clean-result bodies,
with band midpoints computed inline.

Usage (from the issue-1335 worktree):
    uv run python scripts/issue1335_fig_endpoints.py --out-dir <repo>/figures/
"""

from __future__ import annotations

import argparse
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

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval_results" / "issue_1335"
PERSONAS = ("Wren", "HELIOS", "Dana", "Vex")

# Prior committed reads (quoted from the parent clean-result bodies; the two
# fiction bands are #1310's committed per-character layer-19 script bands).
PRIOR_NATURALISTIC_ASSISTANT_BASE = 0.5783  # #825 naturalistic plain-text round
PRIOR_CHAT_TEMPLATE_ASSISTANT_BASE = 0.5877  # #825 chat-template map
PRIOR_CHAT_TEMPLATE_ASSISTANT_INSTRUCT = 0.6731  # #825 chat-template map
PRIOR_MULTITURN_BAND = (0.71, 0.74)  # #1092 multi-turn naturalistic band (base)
PRIOR_FICTION_BASE_BAND = (0.106, 0.148)  # #1310 committed four-persona band
PRIOR_FICTION_INSTRUCT_BAND = (0.166, 0.253)  # #1310 committed four-persona band


def _r2_layer19(path: Path) -> float:
    cell = json.loads(path.read_text())
    return float(cell["r2_per_layer_obs"][cell["headline_layer"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/"))
    args = parser.parse_args()

    qa = {
        m: _r2_layer19(EVAL_DIR / f"cells_r0_qa_full__{m}__ctx.json") for m in ("base", "instruct")
    }
    fiction = {
        m: sum(_r2_layer19(EVAL_DIR / f"cells_r7_endpoint__{m}__{p}__ctx.json") for p in PERSONAS)
        / len(PERSONAS)
        for m in ("base", "instruct")
    }

    # (label, value, kind, model) in display order top -> bottom.
    rows = [
        ("this run: Q&A full answers (base)", qa["base"], "this", "base"),
        ("this run: Q&A full answers (instruct)", qa["instruct"], "this", "instruct"),
        (
            "prior committed: naturalistic assistant map (base)",
            PRIOR_NATURALISTIC_ASSISTANT_BASE,
            "prior",
            None,
        ),
        (
            "prior committed: chat-template assistant map (base)",
            PRIOR_CHAT_TEMPLATE_ASSISTANT_BASE,
            "prior",
            None,
        ),
        (
            "prior committed: chat-template assistant map (instruct)",
            PRIOR_CHAT_TEMPLATE_ASSISTANT_INSTRUCT,
            "prior",
            None,
        ),
        (
            "prior committed: multi-turn naturalistic map (base, band midpoint)",
            sum(PRIOR_MULTITURN_BAND) / 2,
            "prior",
            None,
        ),
        ("this run: story scenes, per-persona mean (base)", fiction["base"], "this", "base"),
        (
            "this run: story scenes, per-persona mean (instruct)",
            fiction["instruct"],
            "this",
            "instruct",
        ),
        (
            "prior committed: fiction per-character map (base, band midpoint)",
            sum(PRIOR_FICTION_BASE_BAND) / 2,
            "prior",
            None,
        ),
        (
            "prior committed: fiction per-character map (instruct, band midpoint)",
            sum(PRIOR_FICTION_INSTRUCT_BAND) / 2,
            "prior",
            None,
        ),
    ]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    color_base, color_instruct = paper_palette_blog(2)

    n_rows = len(rows)
    for i, (_, value, kind, model) in enumerate(rows):
        y = n_rows - 1 - i
        if kind == "this":
            ax.scatter(
                [value],
                [y],
                s=55,
                marker="o",
                color=color_base if model == "base" else color_instruct,
                zorder=3,
            )
        else:
            ax.scatter([value], [y], s=45, marker="s", color="#555555", zorder=3)

    ax.set_yticks([n_rows - 1 - i for i in range(n_rows)])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel("held-out R² (layer 19)")
    ax.set_title(
        "this run's endpoints vs prior committed reads (different recipes/selectors)",
        loc="left",
    )

    savefig_paper(fig, "issue_1335/endpoints_vs_committed", dir=args.out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
