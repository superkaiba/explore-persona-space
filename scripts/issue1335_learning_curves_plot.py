"""Issue #1335 learning-curve figures (context arm + prefix companion).

Reads eval_results/issue_1335/learning_curves/results.json and renders, per arm,
a 2-panel (base | instruct) learning curve: x = n_train (log2), y = held-out R^2
at layer 19, one line per fiction character (Wren/HELIOS/Dana/Vex, from the
r7_endpoint store) plus the two assistant Q&A anchors (one-line / full-answer),
with faint per-draw points behind the mean lines. Seed 42 (the only generation
seed on the Hub) — stated in the subtitle. No interpretive overlays.

  uv run python scripts/issue1335_learning_curves_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

RESULTS = Path("eval_results/issue_1335/learning_curves/results.json")
FIG_DIR = "figures/"
FIG_SUBDIR = "issue_1335/learning_curves"

# Fixed series order + encoding (identical across both panels AND both figures).
CHARACTERS = ["Wren", "HELIOS", "Dana", "Vex"]
ANCHORS = [("r1_qa_oneline", "one-line Q&A"), ("r0_qa_full", "full-answer Q&A")]
_PAL = paper_palette(6)
SERIES_COLOR = {
    "Wren": _PAL[0],
    "HELIOS": _PAL[1],
    "Dana": _PAL[2],
    "Vex": _PAL[3],
    "one-line Q&A": _PAL[4],
    "full-answer Q&A": _PAL[5],
}
# Characters solid/circle; assistant anchors dashed/square (reads as two groups).
SERIES_STYLE = {c: dict(ls="-", marker="o") for c in CHARACTERS}
SERIES_STYLE.update({lbl: dict(ls="--", marker="s") for _, lbl in ANCHORS})
MODELS = ["base", "instruct"]


def _index(cells: list[dict]) -> dict:
    return {(c["model"], c["arm"], c["rung"], c.get("persona")): c for c in cells}


def _mean_curve(cell: dict) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float]]]:
    """(x mean-points, y means, raw per-draw (n, r2) pairs) for one cell."""
    xs, ys = [], []
    raw: list[tuple[int, float]] = []
    for ne in cell["effective_n_train"]:
        rs = [p["r2"] for p in cell["points"] if p["n_train"] == ne]
        xs.append(ne)
        ys.append(float(np.mean(rs)))
        raw.extend((ne, float(r)) for r in rs)
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order], raw


def _plot_arm(cells_by: dict, arm: str, stem: str, subtitle: str) -> str:
    import matplotlib.pyplot as plt

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        series = [(p, cells_by[(model, arm, "r7_endpoint", p)]) for p in CHARACTERS] + [
            (lbl, cells_by[(model, arm, rung, None)]) for rung, lbl in ANCHORS
        ]
        for lbl, cell in series:
            x, y, raw = _mean_curve(cell)
            col = SERIES_COLOR[lbl]
            st = SERIES_STYLE[lbl]
            if raw:
                rx = np.asarray([r[0] for r in raw], dtype=float)
                ry = np.asarray([r[1] for r in raw])
                ax.scatter(rx, ry, color=col, s=9, alpha=0.22, linewidths=0, zorder=1)
            ax.plot(x, y, color=col, lw=1.6, zorder=3, label=lbl, **st)
        ax.set_xscale("log", base=2)
        ax.set_xticks([200, 400, 800, 1600, 3200])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("training rows (n_train)")
        ax.set_title(model, loc="left", fontsize=11)
    axes[0].set_ylabel("held-out R² (layer 19)")
    add_direction_arrow(axes[0], axis="y", direction="up")
    # One shared, frameless legend in its OWN reserved band below the panels.
    # "outside lower center" is the constrained-layout-aware placement (the blog
    # style uses layout="constrained", so a bbox_to_anchor legend + manual
    # subplots_adjust would overlap the x-axis labels).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    # Figure-level title block (set_title_subtitle targets a single Axes; a
    # 2-panel figure uses suptitle + a subtitle line instead).
    fig.suptitle(
        "Context→answer map R² vs training size",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(0.02, 1.005, subtitle, ha="left", fontsize=8.5, color="#555555")
    savefig_paper(fig, f"{FIG_SUBDIR}/{stem}", dir=FIG_DIR)
    plt.close(fig)
    return f"{FIG_SUBDIR}/{stem}.png"


def main() -> int:
    d = json.loads(RESULTS.read_text())
    cells_by = _index(d["cells"])
    captions: dict[str, str] = {}

    ctx_png = _plot_arm(
        cells_by,
        "ctx",
        "main_context_arm",
        "Layer 19, held-out R², context arm (prefix + user query). Fiction "
        "characters (solid) vs assistant Q&A anchors (dashed); generation seed 42.",
    )
    captions[Path(ctx_png).name] = (
        "Held-out layer-19 R² of the context→answer ridge map vs training rows "
        "(log₂ x, fixed 20%-group test set, 5 group-stratified subsample draws "
        "per point shown faint). Panels: Qwen-2.5-7B base | instruct. Four fiction "
        "characters (solid) and two assistant Q&A anchors (dashed); seed 42."
    )

    pre_png = _plot_arm(
        cells_by,
        "prefix",
        "companion_prefix_arm",
        "Layer 19, held-out R², prefix arm (before the user query). Q&A prefix is "
        "the degenerate prefix_fallback_first_token control; generation seed 42.",
    )
    captions[Path(pre_png).name] = (
        "Companion prefix-arm learning curves (prefix = everything before the user "
        "query), same layout as the main figure. The assistant Q&A anchors use the "
        "degenerate prefix_fallback_first_token control, so their prefix-arm curves "
        "are not a like-for-like map; fiction characters use the real prefix span. Seed 42."
    )

    cap_path = Path(FIG_DIR) / FIG_SUBDIR / "captions.json"
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    cap_path.write_text(json.dumps(captions, indent=2))
    print(f"[lc-plot] wrote {ctx_png}, {pre_png}, {cap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
