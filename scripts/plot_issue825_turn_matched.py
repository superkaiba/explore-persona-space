"""Plot verified matched-budget turn-transfer curves using the manuscript style."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
import numpy as np

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    ROLES,
    better_label,
    c2a_figure,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    if result["n_conversations"] != 4975 or result["answer_draws"] != 1:
        raise ValueError("wrong matched-panel result")
    set_c2a_style()
    turns = np.arange(1, 13)
    control = np.array(to_rgb(ROLES["control"].color))
    encodings = [
        ("1", "Turn 1", to_hex(0.2 + 0.8 * control), "s"),
        ("2", "Turn 2", ROLES["control"].color, "^"),
        ("3", "Turn 3", to_hex(0.55 * control), "D"),
        ("1+2+3", "Turns 1 + 2 + 3", ROLES["linear"].color, "o"),
    ]
    if "12" in result["source_conditions"]:
        encodings.insert(3, ("12", "Turn 12", INK, "v"))
    rendered = []
    for kind in ("transfer", "rotation_sensitivity", "copy_baselines"):
        fig, fraction = c2a_figure("full", aspect=0.48)
        axes = fig.subplots(1, 2, sharey=True)
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.28, top=0.80, wspace=0.15)
        all_limits = []
        for ax, model, letter in zip(axes, ("instruct", "pretrained"), ("A", "B"), strict=True):
            cells = result["models"][model]["cells"]
            ax.axvspan(0.7, 3.5, color=ROLES["control"].color, alpha=0.07, zorder=-2)
            ax.axvline(3.5, color=ROLES["control"].color, alpha=0.6, linewidth=1, linestyle=":")
            for source, label, color, marker in encodings:
                method = "source_identity_bias" if kind == "copy_baselines" else "raw"
                rows = sorted(
                    (r for r in cells if r["source"] == source and r["method"] == method),
                    key=lambda r: r["target_turn"],
                )
                if [r["target_turn"] for r in rows] != turns.tolist():
                    raise ValueError("incomplete curve")
                values = np.array([r["r2"] for r in rows])
                ci = np.array([r["r2_ci95"] for r in rows])
                all_limits.extend(ci.ravel())
                ax.plot(
                    turns,
                    values,
                    label=label,
                    color=color,
                    marker=marker,
                    markerfacecolor=color if source == "1+2+3" else "white",
                    markersize=6,
                    linewidth=2.4 if source == "1+2+3" else 1.8,
                    linestyle="--" if source == "12" else "-",
                )
                if kind != "rotation_sensitivity":
                    ax.fill_between(turns, ci[:, 0], ci[:, 1], color=color, alpha=0.13, linewidth=0)
            if kind == "rotation_sensitivity":
                rotations = np.array(result["models"][model]["rotation_r2"])
                for r in rotations:
                    ax.plot(
                        turns,
                        r,
                        color=ROLES["linear"].color,
                        alpha=0.45,
                        linewidth=1,
                        linestyle=":",
                    )
                all_limits.extend(rotations.ravel())
            ax.set_xlim(0.7, 12.3)
            ax.set_xticks([1, 2, 3, 6, 9, 12])
            ax.set_xlabel("Evaluation turn")
            panel_header(ax, letter, model.title(), kicker_y=1.06)
            style_axis(ax)
        lo, hi = min(all_limits), max(all_limits)
        axes[0].set_ylim(np.floor((lo - 0.02) * 10) / 10, min(1, np.ceil((hi + 0.02) * 10) / 10))
        axes[0].set_ylabel(better_label(r"Held-out $R^2$"))
        legend_kicker(
            fig,
            0.085,
            0.165,
            "Metamodel training turns" if kind != "copy_baselines" else "Copy-bias training turns",
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.08, 0.07),
            ncol=len(encodings),
            frameon=False,
            columnspacing=1.25,
        )
        subject = (
            "Three counterbalanced pooled fits shown as dotted curves; solid curve averages performance."
            if kind == "rotation_sensitivity"
            else "Bands are pointwise conditional 95% paired-conversation bootstrap intervals."
        )
        title = {
            "transfer": "Transfer at a matched training budget",
            "rotation_sensitivity": "Sensitivity to pooled turn assignment",
            "copy_baselines": "Source-trained copy-plus-bias baselines",
        }[kind]
        saved = save_c2a_figure(
            fig,
            args.out / f"turn_matched_{kind}_20260915",
            title=title,
            subject=subject,
            creator=Path(__file__).name,
            include_width=fraction,
        )
        plt.close(fig)
        meta = dict(
            title=title,
            subject=subject,
            source_path=str(args.results.resolve()),
            source_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(),
            n_conversations=4975,
            answer_draws=1,
            render=saved["record"],
            encodings=[dict(source=s, label=l, color=c, marker=m) for s, l, c, m in encodings],
            caveat="Turns1–3 overlap pooled source support; forward transfer is turns4–12. Each pooled rotation is a distinct matched-N fit; average losses, not predictions.",
        )
        (args.out / f"turn_matched_{kind}_20260915.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )
        rendered.append({k: str(saved[k]) for k in ("png", "pdf", "grayscale")})
    print(json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
