#!/usr/bin/env python3
"""Render single-turn rank-1 retrieval failures as true-vs-retrieved comparison cards.

Produces the paper stem ``figures/paper/c3_qualitative_discrimination`` under
the c2a-v2 figure standard (fixed authoring scale, Inter, provenance sidecar).
Rows may be flat (``rows``) or grouped by failure category (``groups``).

The display-ready text and evaluation provenance live in
``eval_results/issue_1901/content_divergent_retrieval_examples.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the matplotlib/numpy imports below. On the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and
# the BLAS pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402


from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    ROLES,
    SEAM,
    STYLE_VERSION,
    c2a_figure,
    canvas_width_in,
    save_c2a_figure,
    set_c2a_style,
)

DEFAULT_INPUT = ROOT / "eval_results/issue_1901/content_divergent_retrieval_examples.json"
DEFAULT_OUTPUT = ROOT / "figures/paper/c3_qualitative_discrimination"

TRUE = ROLES["linear"].color
WRONG = ROLES["control"].color


def _tint(color: str, fraction: float) -> tuple[float, float, float]:
    """Blend a color toward white; fraction is the color share."""
    r, g, b = to_rgb(color)
    return tuple(1.0 - fraction * (1.0 - c) for c in (r, g, b))


TRUE_PALE = _tint(TRUE, 0.10)
WRONG_PALE = _tint(WRONG, 0.12)
LABEL_FILL = _tint(INK, 0.045)
ANSWER_FILL = _tint(INK, 0.025)

BODY_PT = 18.0  # realizes 7.56 pt at the fixed c2a scale (body floor)
HEADER_PT = 13.0  # in-table column kickers, module legend_kicker register
LABEL_PT = 12.5
QUERY_WRAP = 42
ANSWER_WRAP = 44
MAX_LINES = 8  # roomy; a cap-truncation would alter the banked text (asserted below)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def wrap_lines(text: str, width: int, max_lines: int) -> list[str]:
    """Wrap text while retaining deliberate paragraph/list breaks."""
    output: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            if output and output[-1] != "":
                output.append("")
            continue
        indent = len(raw) - len(raw.lstrip())
        initial = raw[:indent]
        subsequent = initial if indent else ""
        output.extend(
            textwrap.wrap(
                raw.strip(),
                width=max(12, width - indent),
                initial_indent=initial,
                subsequent_indent=subsequent,
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
            or [""]
        )
    while output and output[-1] == "":
        output.pop()
    if len(output) > max_lines:
        raise ValueError(
            f"wrapped text needs {len(output)} lines (cap {max_lines}); a cap truncation "
            "would silently alter the banked excerpt - raise MAX_LINES or shorten upstream"
        )
    return output


def line_height(font_size: float) -> float:
    return font_size / 72.0 * 1.24


def pair_layout(row: dict[str, object]) -> dict[str, object]:
    """Prepare identically aligned rows for the two sides of a failure."""
    layouts: dict[str, dict[str, list[str]]] = {}
    for side in ("true", "retrieved"):
        item = row[side]
        query = wrap_lines(str(item["final_query"]), QUERY_WRAP, MAX_LINES)
        answer = wrap_lines(
            str(item["answer_excerpt"]).replace("\n\n", "\n"), ANSWER_WRAP, MAX_LINES
        )
        layouts[side] = {"query": query, "answer": answer}

    max_lines = {
        field: max(len(layouts[side][field]) for side in ("true", "retrieved"))
        for field in ("query", "answer")
    }
    row_heights = {
        field: 0.26 + max_lines[field] * line_height(BODY_PT) for field in ("query", "answer")
    }
    return {
        "sides": layouts,
        "row_heights": row_heights,
        "table_height": 0.46 + sum(row_heights.values()),
    }


def draw_pair_table(
    ax: plt.Axes,
    *,
    row: dict[str, object],
    x: float,
    y_top: float,
    width: float,
) -> float:
    """Draw one failure as a shared-label, two-column comparison table."""
    layout = pair_layout(row)
    header_h = 0.46
    table_top = y_top
    table_h = float(layout["table_height"])
    table_bottom = table_top - table_h

    label_w = 1.15
    center_gap = 0.14
    content_w = width - label_w
    col_w = (content_w - center_gap) / 2
    true_x = x + label_w
    retrieved_x = true_x + col_w + center_gap

    ax.add_patch(
        FancyBboxPatch(
            (x, table_bottom),
            width,
            table_h,
            boxstyle="round,pad=0.008,rounding_size=0.065",
            linewidth=1.0,
            edgecolor=SEAM,
            facecolor=PAPER,
        )
    )
    header_bottom = table_top - header_h
    ax.add_patch(
        Rectangle((x, header_bottom), label_w, header_h, linewidth=0, facecolor=LABEL_FILL)
    )
    ax.add_patch(
        Rectangle((true_x, header_bottom), col_w, header_h, linewidth=0, facecolor=TRUE_PALE)
    )
    ax.add_patch(
        Rectangle((retrieved_x, header_bottom), col_w, header_h, linewidth=0, facecolor=WRONG_PALE)
    )
    ax.add_patch(Rectangle((true_x, header_bottom), 0.06, header_h, linewidth=0, facecolor=TRUE))
    ax.add_patch(
        Rectangle((retrieved_x, header_bottom), 0.06, header_h, linewidth=0, facecolor=WRONG)
    )

    ax.text(
        true_x + 0.2,
        table_top - header_h / 2,
        f"TRUE ANSWER  ·  RANK {row['true_answer_rank']}",
        ha="left",
        va="center",
        color=TRUE,
        fontsize=HEADER_PT,
        fontweight=750,
    )
    ax.text(
        retrieved_x + 0.2,
        table_top - header_h / 2,
        "RETRIEVED ANSWER  ·  RANK 1",
        ha="left",
        va="center",
        color=WRONG,
        fontsize=HEADER_PT,
        fontweight=750,
    )
    fields = (
        ("query", "QUERY"),
        ("answer", "ANSWER"),
    )
    current_top = header_bottom
    for field, label in fields:
        height = float(layout["row_heights"][field])
        bottom = current_top - height
        if field == "answer":
            ax.add_patch(
                Rectangle((true_x, bottom), col_w, height, linewidth=0, facecolor=ANSWER_FILL)
            )
            ax.add_patch(
                Rectangle((retrieved_x, bottom), col_w, height, linewidth=0, facecolor=ANSWER_FILL)
            )
        ax.add_patch(Rectangle((x, bottom), label_w, height, linewidth=0, facecolor=LABEL_FILL))
        ax.plot([x, x + width], [current_top, current_top], color=SEAM, linewidth=0.75)
        ax.text(
            x + 0.14,
            current_top - 0.14,
            label,
            ha="left",
            va="top",
            color=MUTED,
            fontsize=LABEL_PT,
            fontweight=700,
            linespacing=1.0,
        )
        for side, column_x in (("true", true_x), ("retrieved", retrieved_x)):
            ax.text(
                column_x + 0.18,
                current_top - 0.14,
                "\n".join(layout["sides"][side][field]),
                ha="left",
                va="top",
                color=INK,
                fontsize=BODY_PT,
                linespacing=1.24,
            )
        current_top = bottom

    ax.plot([true_x, true_x], [table_bottom, table_top], color=SEAM, linewidth=0.75)
    ax.plot(
        [retrieved_x - center_gap, retrieved_x - center_gap],
        [table_bottom, table_top],
        color=SEAM,
        linewidth=0.75,
    )
    ax.plot([retrieved_x, retrieved_x], [table_bottom, table_top], color=SEAM, linewidth=0.75)
    return table_bottom


GROUP_HEADER_H = 0.38
ROW_GAP = 0.16
GROUP_GAP = 0.26
MARGIN_TOP = 0.10
MARGIN_BOTTOM = 0.08


def _groups(data: dict[str, object]) -> list[dict[str, object]]:
    """Return category groups; a flat ``rows`` list becomes one untitled group."""
    if data.get("groups"):
        return list(data["groups"])  # type: ignore[arg-type]
    return [{"title": None, "rows": data["rows"]}]


def _canvas_height(groups: list[dict[str, object]]) -> float:
    total = MARGIN_TOP + MARGIN_BOTTOM
    for gi, group in enumerate(groups):
        if group.get("title"):
            total += GROUP_HEADER_H
        for row in group["rows"]:
            total += float(pair_layout(row)["table_height"]) + ROW_GAP
        if gi < len(groups) - 1:
            total += GROUP_GAP
    return total - ROW_GAP


def render(data: dict[str, object]) -> tuple[plt.Figure, float]:
    """Build the figure on the fixed-scale full-width canvas; data units are inches."""
    groups = _groups(data)
    width = canvas_width_in(1.0)
    height = _canvas_height(groups)
    fig, include_frac = c2a_figure("full", aspect=height / width)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    margin_x = 0.10
    y_cursor = height - MARGIN_TOP
    for gi, group in enumerate(groups):
        if group.get("title"):
            ax.text(
                margin_x + 0.02,
                y_cursor - GROUP_HEADER_H / 2,
                str(group["title"]),
                ha="left",
                va="center",
                color=INK,
                fontsize=15,
                fontweight=750,
            )
            y_cursor -= GROUP_HEADER_H
        for row in group["rows"]:
            y_cursor = (
                draw_pair_table(ax, row=row, x=margin_x, y_top=y_cursor, width=width - 2 * margin_x)
                - ROW_GAP
            )
        if gi < len(groups) - 1:
            y_cursor -= GROUP_GAP
    if y_cursor < MARGIN_BOTTOM - ROW_GAP - 1e-6:
        raise ValueError("comparison tables exceed the figure canvas")
    return fig, include_frac


def main() -> None:
    args = parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    set_c2a_style()
    fig, include_frac = render(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = save_c2a_figure(
        fig,
        args.output,
        title="Qualitative rank-1 retrieval failures",
        subject=(
            "True answer versus retrieved rank-1 answer for three single-turn "
            "content-divergent retrieval failures"
        ),
        creator="scripts/issue1901_qualitative_retrieval_failures.py",
        include_width=include_frac,
    )
    plt.close(fig)
    metadata = {
        "status": "Results Section 4.2 manuscript figure",
        "style_version": STYLE_VERSION,
        "plotting_script": "scripts/issue1901_qualitative_retrieval_failures.py",
        "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
        "reproduction_command": "uv run python scripts/issue1901_qualitative_retrieval_failures.py",
        "input": str(args.input.relative_to(ROOT))
        if args.input.is_relative_to(ROOT)
        else str(args.input),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "render": outputs["record"],
        "selection": data["selection"],
        "evaluation": data["evaluation"],
        "output_sha256": {
            kind: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for kind, path in outputs.items()
            if isinstance(path, Path)
        },
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for kind, path in outputs.items():
        if isinstance(path, Path):
            print(f"qualitative.{kind}: {path}")


if __name__ == "__main__":
    main()
