"""Condense the original four-method results into the paper's three-panel style.

Read the verified historical summary only; all 36 values and OOD standard errors
remain unchanged. Export a full-width vector figure and its review/provenance files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import issue1739_four_method_figure as shared  # noqa: E402

SOURCE = ROOT / "eval_results/issue_1739/old_four_method_figure_20260917/summary.json"
SOURCE_SHA = "ba6dda3bd3ca206e65725085532e40b277e0cbd1b4a2f1d0b2a72c0158cb2384"
STEM = "c5_behavior_transfer_original_combined"
CAPTION = (
    "**Mapped answers usually improve prediction from fixed answer directions.** "
    "(A) Sycophancy, (B) hallucination, and (C) harmful compliance across generic chat, "
    "in-distribution (ID), and out-of-distribution (OOD) evaluations. Bars show Spearman "
    "correlations. OOD bars average datasets equally; whiskers show one standard error "
    "across datasets. The original maps, whitening, and method-specific layers are "
    "retained. ID rows were included in map fitting and therefore do not measure "
    "map-held-out generalization. The context-direction baseline uses the matching "
    "cached extraction."
)


def render():
    """Render the existing estimates as three grouped-bar panels without refitting."""
    if shared.sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Historical summary changed; verify it before updating the pin")
    data = json.loads(SOURCE.read_text())
    if len(data["cells"]) != 9 or any(len(c["arms"]) != 4 for c in data["cells"]):
        raise ValueError("Expected nine complete evaluation groups")
    shared.set_c2a_style()
    fig, fraction = shared.c2a_figure("full", aspect=0.38)
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.31, wspace=0.15)
    faces = (shared.MUTED, shared.ROLES["linear"].color, shared.PAPER, shared.INK)
    edges = (shared.INK, shared.ROLES["linear"].color, shared.MUTED, shared.INK)
    offsets = (-0.285, -0.095, 0.095, 0.285)
    for panel, (behavior, heading) in enumerate(
        zip(shared.BEHAVIORS, shared.HEADINGS, strict=True)
    ):
        ax = axes[panel]
        ticklabels = []
        for group, regime in enumerate(("Generic chat", "ID", "OOD")):
            matches = [
                c for c in data["cells"] if c["behavior"] == behavior and c["regime"] == regime
            ]
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicated cell: {behavior}/{regime}")
            cell = matches[0]
            label = "Generic\nchat" if regime == "Generic chat" else regime
            ticklabels.append(label)
            for index, method in enumerate(shared.METHODS):
                estimate = cell["arms"][method]
                value, interval = estimate["rho"], estimate["interval"]
                visible = [value] if interval is None else [value, *interval]
                if (
                    not shared.np.isfinite(visible).all()
                    or min(visible) < -0.2
                    or max(visible) > 0.8
                ):
                    raise ValueError(f"Nonfinite or clipped estimate: {estimate}")
                x = group + offsets[index]
                ax.bar(
                    x,
                    value,
                    width=0.16,
                    color=faces[index],
                    edgecolor=edges[index],
                    linewidth=1.3,
                    hatch="///" if index == 0 else None,
                    label=shared.LABELS[index] if group == 0 else None,
                    zorder=2,
                )
                if interval is not None:
                    lo, hi = interval
                    if lo > hi:
                        raise ValueError(f"Reversed interval: {interval}")
                    ax.vlines(x, lo, hi, color=shared.INK, linewidth=1.1, zorder=3)
                    ax.hlines(
                        [lo, hi], x - 0.035, x + 0.035, color=shared.INK, linewidth=1.1, zorder=3
                    )
        ax.set_xlim(-0.52, 2.52)
        ax.set_ylim(-0.2, 0.8)
        ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xticks([0, 1, 2], ticklabels)
        shared.style_axis(ax, grid_axis="none")
        ax.tick_params(axis="x", length=0, pad=8)
        ax.axhline(0, color=shared.MUTED, linewidth=0.7, zorder=0)
        shared.panel_header(ax, chr(65 + panel), heading, kicker_y=1.06)
        if panel:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel(shared.better_label(r"Spearman $\rho$"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.01),
        ncol=2,
        labelspacing=0.25,
        handlelength=1.2,
        columnspacing=1.7,
    )
    out = ROOT / "figures/paper"
    result = shared.save_c2a_figure(
        fig,
        out / STEM,
        title="Fixed behavior directions across evaluation regimes",
        subject="Original results, three behavior panels, four fixed projection methods",
        creator=str(Path(__file__).relative_to(ROOT)),
        include_width=fraction,
    )
    metadata = {
        **data,
        "summary_sha256": SOURCE_SHA,
        "renderer_sha256": shared.sha(Path(__file__)),
        "render": result["record"],
        "caption": CAPTION,
        "layout": "Three behavior panels; generic, ID and OOD groups; unchanged historical values",
    }
    for key in ("pdf", "png", "grayscale"):
        metadata[f"{key}_sha256"] = shared.sha(result[key])
    (out / f"{STEM}.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / f"{STEM}.caption.md").write_text(CAPTION + "\n")
    shared.plt.close(fig)
    print(json.dumps({k: str(v) for k, v in result.items() if k != "record"}), flush=True)


if __name__ == "__main__":
    render()
