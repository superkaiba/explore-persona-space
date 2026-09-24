#!/usr/bin/env python3
"""Render the Section 4.3 post-training figure (paper ``fig:posttraining``).

Supersedes ``scripts/section43_posttraining_figure.py`` for the paper figure
(that script renders the earlier plain-text capture on 16,391 contexts, which
the appendix's layer and length controls still cite). This one reads the
matched-format on-policy rerun: OLMo-2-7B Base, SFT, DPO, and RLVR on 13,333
LMSYS contexts, each checkpoint generating and encoding its own answers under
both the chat template and plain ``User:``/``Assistant:`` tags, layer 18,
six held-out folds, 95% paired context-bootstrap intervals over 1,000 draws.

Input is ONE file, ``fits/olmo/summary.json`` from the pinned HF revision,
copied to ``figures/issue_1902/section43/inputs/olmo_onpolicy_summary.json``
and SHA-256 checked on every run (``--fetch`` downloads it when absent).

Layout: one ``full``-width row of three panels at aspect 0.285, with ONE kicker
legend row above them (each group's uppercase heading inline with its entries,
per ``figure_standard.md`` 2.5) and the panel kickers set close to their axes.
That prints 1.44 in tall at the 5.5 in text width, down from 1.69 in; no plotted
value changed. Format is a series everywhere: plain text = solid line, filled
circle; chat template = dotted line, filled diamond.

  A  each checkpoint's own map, held-out R^2 by checkpoint;
  B  R^2 into each post-trained checkpoint's answer vectors from its own
     context vectors (teal) and from base-model context vectors (amber);
  C  retention of the preceding checkpoint's map on the next checkpoint's own
     pairs, applied unchanged. Calibration variants remain in the saved data
     but are not drawn (Thomas, 2026-09-17).

Outputs (``figures/paper``): ``c1_posttraining_dynamics{.pdf,.png,
_grayscale.png,.meta.json}`` and ``c1_posttraining_dynamics_data.json`` with
every plotted value, from which the figure is reproducible alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the matplotlib/numpy imports below.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    MUTED,
    ROLES,
    better_label,
    c2a_figure,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_PATH = "issue1902_olmo_onpolicy_20260914/production_v1/fits/olmo/summary.json"
HF_REVISION = "7761da3391e9f670b66f2ab246cdefcd4bd7d974"
SUMMARY_SHA256 = "6833da54f99d3b4974a052387e6259a3c470e8288d1c31b8c4953c6e7a064d92"
INPUT_PATH = (
    PROJECT_ROOT / "figures" / "issue_1902" / "section43" / "inputs" / "olmo_onpolicy_summary.json"
)
OUT_STEM = "c1_posttraining_dynamics"

STAGES = ("B", "S", "D", "R")
STAGE_NAMES = {"B": "Base", "S": "SFT", "D": "DPO", "R": "RLVR"}
TARGETS = ("S", "D", "R")
TRANSITIONS = (("B", "S"), ("S", "D"), ("D", "R"))
FORMATS = ("plain", "chat")
FORMAT_STYLE = {
    "plain": {"linestyle": "-", "marker": "o", "label": "Plain text"},
    "chat": {"linestyle": ":", "marker": "D", "label": "Chat template"},
}
CORRECTIONS = (("direct", "Frozen"), ("bias", "with refit bias"))
TEAL = ROLES["post_trained"].color
AMBER = ROLES["base_model"].color

# One kicker legend row above the panels (figure_standard.md 2.5).  The heading
# sits inline with its entries, so the legend costs one row, not two.  At the
# pinned c2a type sizes, concise context-source labels carry the heading's noun: "CONTEXT SOURCE"
# over "Own"/"Base" rather than "Own states"/"Base states".
LEGEND_Y = 0.905
"""Figure-fraction centre line of the kicker legend row."""

LEGEND_MARGIN = 0.006
"""Canvas margin kept free at both ends of the legend row."""

HEADING_GAP = 0.009
"""Figure-fraction gap between a group heading and its first legend entry."""

MIN_GROUP_GAP = 0.02
"""Smallest figure-fraction gap tolerated between two legend groups."""

KICKER_Y = 1.04
"""Panel-kicker baseline in axes fractions (the c2a default 1.16 left a gap
the size of the kicker itself between the kicker and the axes)."""
SERIES_ENCODING = {
    "format": "plain text = filled circle; chat template = filled diamond; "
    "panels A/B add solid plain-text and dotted chat-template lines",
    "panel_a": "one hue (post_trained teal); each checkpoint's own map",
    "panel_b": "own context vectors = teal; base-model context vectors = amber "
    "(ROLES['base_model'])",
    "panel_c": "unchanged source maps, filled format markers; plain at x-0.15, "
    "chat at x+0.15; bias and scale_bias retained in source results, not drawn",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_input(*, fetch: bool) -> Path:
    if not INPUT_PATH.exists():
        if not fetch:
            msg = f"{INPUT_PATH} is missing; rerun with --fetch to download it from HF"
            raise FileNotFoundError(msg)
        from huggingface_hub import hf_hub_download

        INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        got = hub.retry_transient(
            lambda: hf_hub_download(
                HF_REPO,
                HF_PATH,
                repo_type="dataset",
                revision=HF_REVISION,
                local_dir=INPUT_PATH.parent,
            ),
            what=f"hf_hub_download {HF_PATH}@{HF_REVISION[:8]}",
        )
        Path(got).replace(INPUT_PATH)
    digest = _sha256(INPUT_PATH)
    if digest != SUMMARY_SHA256:
        msg = f"{INPUT_PATH} sha256 {digest} != pinned {SUMMARY_SHA256}"
        raise ValueError(msg)
    return INPUT_PATH


def load_results(summary_path: Path) -> dict[str, Any]:
    """Collect every plotted value into one JSON-serializable dict."""
    summary = json.loads(summary_path.read_text())
    cells = summary["cells"]
    if summary["n"] != 13_333:
        msg = f"expected n=13,333 contexts, got {summary['n']}"
        raise ValueError(msg)

    def cell(source: str, target: str, fmt: str) -> dict[str, Any]:
        entry = cells[f"fit_{source}{target}_{fmt}_fit"]
        assert (entry["source"], entry["target"], entry["format"]) == (source, target, fmt)
        lo, hi = entry["r2_ci95"]
        assert lo <= hi and np.isfinite([entry["r2"], lo, hi]).all()
        return {"r2": entry["r2"], "ci": [lo, hi]}

    def retention(source: str, target: str, fmt: str, mode: str) -> dict[str, Any]:
        rows = [
            r
            for r in summary["panel_c"]
            if (r["source"], r["target"], r["format"], r["mode"]) == (source, target, fmt, mode)
        ]
        if len(rows) != 1:
            msg = f"expected one panel_c row for {(source, target, fmt, mode)}, got {len(rows)}"
            raise ValueError(msg)
        row = rows[0]
        lo, hi = row["retention_ci95"]
        assert lo <= hi and np.isfinite([row["retention"], lo, hi]).all()
        return {"retention": row["retention"], "ci": [lo, hi]}

    panel_a = {fmt: {s: cell(s, s, fmt) for s in STAGES} for fmt in FORMATS}
    panel_b = {
        fmt: {t: {"own": cell(t, t, fmt), "base": cell("B", t, fmt)} for t in TARGETS}
        for fmt in FORMATS
    }
    panel_c = {
        fmt: {
            f"{s}{t}": {mode: retention(s, t, fmt, mode) for mode, _ in CORRECTIONS}
            for s, t in TRANSITIONS
        }
        for fmt in FORMATS
    }
    # Panel B's "own" points are panel A's post-trained points; assert the tie.
    for fmt in FORMATS:
        for t in TARGETS:
            assert panel_b[fmt][t]["own"] == panel_a[fmt][t]

    return {
        "metadata": {
            "model": "OLMo-2-7B (Base, SFT, DPO, RLVR)",
            "layer": 18,
            "n_contexts": summary["n"],
            "folds": "six held-out folds",
            "target_definition": summary["target_definition"],
            "metric": summary["metric"],
            "uncertainty": summary["uncertainty"],
            "hf_repo": HF_REPO,
            "hf_path": HF_PATH,
            "hf_revision": HF_REVISION,
            "input": {
                "path": str(summary_path.relative_to(PROJECT_ROOT)),
                "sha256": SUMMARY_SHA256,
            },
            "series_encoding": SERIES_ENCODING,
            "not_drawn": "panel_c bias and scale_bias retention",
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_c": panel_c,
    }


def _yerr(points: list[float], cis: list[list[float]]) -> np.ndarray:
    v = np.asarray(points)
    c = np.asarray(cis)
    return np.vstack([np.maximum(v - c[:, 0], 0.0), np.maximum(c[:, 1] - v, 0.0)])


def _series(ax: plt.Axes, xs, rows: list[dict[str, Any]], key: str, color: str, fmt: str) -> None:
    style = FORMAT_STYLE[fmt]
    points = [r[key] for r in rows]
    ax.errorbar(
        xs,
        points,
        yerr=_yerr(points, [r["ci"] for r in rows]),
        color=color,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=5,
        linewidth=1.4,
        capsize=2,
        elinewidth=1,
    )


def plot_panel_a(ax: plt.Axes, data: dict[str, Any]) -> None:
    for fmt in FORMATS:
        _series(ax, range(4), [data["panel_a"][fmt][s] for s in STAGES], "r2", TEAL, fmt)
    ax.set_xticks(range(4), [STAGE_NAMES[s] for s in STAGES])
    ax.set_ylim(0.5, 0.68)
    ax.set_ylabel(better_label("Held-out $R^2$"))
    style_axis(ax)
    panel_header(ax, "A", "Own fits", kicker_y=KICKER_Y)


def plot_panel_b(ax: plt.Axes, data: dict[str, Any]) -> None:
    for fmt in FORMATS:
        rows = [data["panel_b"][fmt][t] for t in TARGETS]
        _series(ax, range(3), [r["own"] for r in rows], "r2", TEAL, fmt)
        _series(ax, range(3), [r["base"] for r in rows], "r2", AMBER, fmt)
    ax.set_xticks(range(3), [STAGE_NAMES[t] for t in TARGETS])
    ax.set_ylim(0.5, 0.68)
    ax.set_xlabel("Answer source")
    ax.set_ylabel(better_label("Held-out $R^2$"))
    style_axis(ax)
    panel_header(ax, "B", "Context sources", kicker_y=KICKER_Y)


def plot_panel_c(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Show only unchanged source maps, using the common format encodings."""
    for fmt in FORMATS:
        dx = -0.15 if fmt == "plain" else 0.15
        for mode in ("direct",):
            rows = [data["panel_c"][fmt][f"{s}{t}"][mode] for s, t in TRANSITIONS]
            points = [r["retention"] for r in rows]
            ax.errorbar(
                np.arange(3) + dx,
                points,
                yerr=_yerr(points, [r["ci"] for r in rows]),
                color=TEAL,
                linestyle="none",
                marker=FORMAT_STYLE[fmt]["marker"],
                markersize=5.5,
                markerfacecolor=TEAL,
                markeredgecolor=TEAL,
                capsize=2,
                elinewidth=1,
            )
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=0.9)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(3), [f"{STAGE_NAMES[s]}\n→ {STAGE_NAMES[t]}" for s, t in TRANSITIONS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(better_label(r"$R^2_{i\to j}\,/\,R^2_{j\to j}$"))
    style_axis(ax)
    panel_header(ax, "C", "Metamodel transfer", kicker_y=KICKER_Y)


def draw_legend_row(fig: plt.Figure, groups: tuple[tuple[str, list[Line2D]], ...]) -> None:
    """Lay the legend groups out as ONE kicker row above the panels.

    Each group is its own frameless legend preceded inline by its uppercase
    heading.  Widths are measured after a draw and the leftover width is split
    evenly between the groups, so a relabelled entry re-spaces the row instead
    of colliding with its neighbour.
    """

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_in = fig.get_figwidth()
    placed = []
    for heading, handles in groups:
        seen = {id(text) for text in fig.texts}
        legend_kicker(fig, 0.0, LEGEND_Y, heading)
        fresh = [text for text in fig.texts if id(text) not in seen]
        if len(fresh) != 1:
            msg = f"legend_kicker added {len(fresh)} figure texts, expected exactly 1"
            raise RuntimeError(msg)
        head = fresh[0]
        head_w = head.get_window_extent(renderer).width / fig.dpi / width_in
        legend = fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.0, LEGEND_Y),
            bbox_transform=fig.transFigure,
            ncol=len(handles),
            frameon=False,
            handlelength=1.05,
            handletextpad=0.3,
            columnspacing=0.6,
            borderaxespad=0.0,
        )
        legend_w = legend.get_window_extent(renderer).width / fig.dpi / width_in
        placed.append((head, head_w, legend, legend_w))

    spans = [head_w + HEADING_GAP + legend_w for _, head_w, _, legend_w in placed]
    free = 1.0 - 2 * LEGEND_MARGIN - sum(spans)
    gap = free / (len(placed) - 1) if len(placed) > 1 else 0.0
    if gap < MIN_GROUP_GAP:
        msg = (
            f"legend row needs {sum(spans):.3f} of the canvas width, leaving {gap:.3f} "
            f"between groups (min {MIN_GROUP_GAP}); shorten a label or a heading"
        )
        raise ValueError(msg)

    x = LEGEND_MARGIN
    for (head, head_w, legend, _), span in zip(placed, spans, strict=True):
        head.set_x(x)
        legend.set_bbox_to_anchor((x + head_w + HEADING_GAP, LEGEND_Y), transform=fig.transFigure)
        x += span + gap


def render(data: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    fig, include_frac = c2a_figure("full", aspect=0.285)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.05],
        left=0.082,
        right=0.975,
        top=0.775,
        bottom=0.225,
        wspace=0.42,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    plot_panel_a(axes[0], data)
    plot_panel_b(axes[1], data)
    plot_panel_c(axes[2], data)

    fmt_handles = [
        Line2D([], [], color=TEAL, linestyle=s["linestyle"], marker=s["marker"], label=s["label"])
        for s in FORMAT_STYLE.values()
    ]
    src_handles = [
        Line2D([], [], color=TEAL, marker="o", label="Own"),
        Line2D([], [], color=AMBER, marker="o", label="Base"),
    ]
    draw_legend_row(
        fig,
        (
            ("Format", fmt_handles),
            ("Context source", src_handles),
        ),
    )
    outputs = save_c2a_figure(
        fig,
        out_dir / OUT_STEM,
        title="Section 4.3 post-training and the linear context-answer map",
        subject=(
            "OLMo-2-7B Base/SFT/DPO/RLVR on-policy fits at layer 18, chat and plain "
            "formats, base-vs-own context sources, adjacent-checkpoint retention "
            "(issue #1902 olmo_onpolicy_20260914)"
        ),
        creator="scripts/section43_posttraining_onpolicy_figure.py",
        include_width=include_frac,
    )
    plt.close(fig)
    meta_path = (out_dir / OUT_STEM).with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {"record": outputs["record"], "metadata": data["metadata"]}, indent=2, sort_keys=True
        )
        + "\n"
    )
    outputs["meta"] = meta_path
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures" / "paper")
    parser.add_argument(
        "--fetch", action="store_true", help="download the pinned summary if absent"
    )
    args = parser.parse_args(argv)
    set_c2a_style()
    data = load_results(ensure_input(fetch=args.fetch))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.out_dir / f"{OUT_STEM}_data.json"
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    # Plot from the JSON round-trip so the figure is reproducible from it alone.
    outputs = render(json.loads(data_path.read_text()), args.out_dir)
    print(f"pdf: {outputs['pdf']}")
    print(f"  latex_include_line: {outputs['record']['latex_include_line']}")
    print(f"data: {data_path}")


if __name__ == "__main__":
    main()
