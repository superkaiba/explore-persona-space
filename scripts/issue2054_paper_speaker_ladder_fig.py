"""Paper figure: the context-to-answer map across speaker identities and framings.

Results-2 Plot 1 of the context-answer-map paper. Replaces the two-figure stitch
of #1345 (chat / plain text / story) plus #1310 (four story characters), which
compared cells measured on different corpora. Everything here comes from ONE
lattice, #2054: one shared draw of real conversations, 8,000 rows per cell
against d = 3,584 (so every ambient ridge fit is well-posed), layer 19, K = 5
conversation-grouped held-out folds under the shared production fold map.

Source: ``eval_results/issue_2054/specialization_ladder/ladder.json`` (context
arm, on-policy answers only, i.e. the measured model writes the answer itself;
the spliced-verbatim-answer cells are deliberately excluded).

Six x positions, ordered by how close the speaker is to the assistant in the
chat template. The ordering is intuitive, not fitted:

    assistant in the chat template
    assistant in plain text ("User: ... / Assistant: ...")
    HELIOS   (ship AI)      speaking in its own story
    Wren     (helpful)      speaking in its own story
    Dana     (ordinary)     speaking in its own story
    Vex      (villain)      speaking in its own story

Two bars per position: base vs instruct.

Two reference marks:

* dotted line - the shuffled-answer null (mean of the six cells' banked 97.5th
  percentiles), the floor a map with no real context-answer pairing reaches.
* open marker on assistant / plain text / instruct - the same cell refit with
  the truncated answers removed. The plain-text render carries no end-of-turn
  token, so 42.5% of that cell's own generations ran to the 4,096-token cap;
  #2054 banked the exclusion refit at 0.390 against the raw 0.209, and a
  random-removal control at matched n moves it the other way (0.195). The raw
  bar is plotted; the marker is the uncontaminated read.

Usage::

    uv run python scripts/issue2054_paper_speaker_ladder_fig.py \
        [--out-dir figures/paper] [--style iclr]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    figsize_iclr_full,
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "eval_results/issue_2054/specialization_ladder/ladder.json"

# Reader-facing display names live here and nowhere else: one label map, so a
# rename lands once. Internal cell ids never reach an axis label.
# (character key, framing key) -> two-line tick label
POSITIONS: list[tuple[str, str, str]] = [
    ("assistant", "chat", "Assistant\nchat template"),
    ("assistant", "bare_text", "Assistant\nplain text"),
    ("helios", "attrib_quoted", "HELIOS\nship AI"),
    ("wren", "attrib_quoted", "Wren\nhelpful"),
    ("dana", "attrib_quoted", "Dana\nordinary"),
    ("vex", "attrib_quoted", "Vex\nvillain"),
]
PROVENANCE = "on_policy"
ARM = "context"

# Banked cap-excluded refit of the one truncation-contaminated cell
# (figures/issue_2054/caphit_censoring_refits.meta.json, bare text / capped
# removed). Keyed by the position index it annotates and the model.
CAP_EXCLUDED = {("assistant", "bare_text", "instruct"): 0.38985271917718867}


def load_cells() -> dict[tuple[str, str, str], dict]:
    """Index the ladder's context-arm on-policy units by (character, framing, model)."""
    payload = json.loads(LADDER.read_text())
    out: dict[tuple[str, str, str], dict] = {}
    for unit in payload["units"]:
        if unit["arm"] != ARM or unit["provenance"] != PROVENANCE:
            continue
        out[(unit["character"], unit["framing"], unit["model"])] = unit
    return out


def build_series(cells: dict[tuple[str, str, str], dict]) -> dict:
    """Pull the six positions x two models out of the indexed cells, fail loud on a miss."""
    series: dict = {"labels": [], "base": [], "instruct": [], "n": [], "null": []}
    for character, framing, label in POSITIONS:
        series["labels"].append(label)
        for model in ("base", "instruct"):
            key = (character, framing, model)
            if key not in cells:
                raise KeyError(f"no {ARM}/{PROVENANCE} cell for {key}; have {sorted(cells)}")
            unit = cells[key]
            series[model].append(unit["ceiling_r2"])
            series["n"].append(unit["n_join"])
            series["null"].append(unit["banked_null_r2_pooled_p95"])
    return series


def figure(series: dict, out_dir: Path) -> None:
    x = np.arange(len(series["labels"]), dtype=float)
    width = 0.38
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.66))

    ax.bar(
        x - width / 2,
        series["base"],
        width,
        color=paper_color("base"),
        label="base",
    )
    ax.bar(
        x + width / 2,
        series["instruct"],
        width,
        color=paper_color("instruct"),
        label="instruct",
    )

    null_level = float(np.mean(series["null"]))
    ax.axhline(
        null_level,
        color=paper_color("null"),
        linestyle=":",
        linewidth=1.0,
        label="shuffled answers (null)",
    )

    for idx, (character, framing, _) in enumerate(POSITIONS):
        value = CAP_EXCLUDED.get((character, framing, "instruct"))
        if value is None:
            continue
        ax.plot(
            [x[idx] + width / 2],
            [value],
            marker="o",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor=paper_color("reference"),
            markeredgewidth=1.1,
            linestyle="none",
            label="instruct, truncated answers removed",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(series["labels"])
    ax.set_ylabel(r"held-out $R^2$ (context $\rightarrow$ answer, layer 19)")
    ax.set_ylim(bottom=min(-0.06, null_level - 0.02))
    ax.axhline(0.0, color=paper_color("reference"), linewidth=0.6)
    # Model bars first, reference marks after: the legend reads in the order a
    # reader parses the chart.
    handles, labels = ax.get_legend_handles_labels()
    order = ["base", "instruct", "shuffled answers (null)", "instruct, truncated answers removed"]
    by_label = dict(zip(labels, handles, strict=True))
    ax.legend([by_label[name] for name in order], order, loc="upper right", ncol=1)

    savefig_paper(fig, "c4_speaker_ladder", dir=out_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures/paper")
    ap.add_argument("--style", choices=("iclr",), default="iclr")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style(args.style)
    series = build_series(load_cells())
    for label, base, instruct in zip(
        series["labels"], series["base"], series["instruct"], strict=True
    ):
        flat = label.replace("\n", " / ")
        print(f"{flat:28s} base {base:+.3f}   instruct {instruct:+.3f}")
    print(f"rows per cell: {sorted(set(series['n']))}")
    print(f"shuffled-answer null (mean p97.5): {np.mean(series['null']):+.4f}")
    figure(series, args.out_dir)
    print("DONE", args.out_dir)


if __name__ == "__main__":
    main()
