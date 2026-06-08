"""Issue #517 round-2 figures: BASE-headroom + within-#517 system-prompt effect.

Round-2 rewrite (post Codex critique 2026-06-08): the trained-arm cross-experiment
comparison is INVALID (Q-banks disjoint). These figures show ONLY what the data
can support:

  1. Per-trait base headroom: base in-scenario vs base default-assistant, with
     the 3.5 PASS threshold drawn. The within-#517 system-prompt effect is the
     paired delta annotated on each panel. Three single-trait figures.

  2. Combined 1x3 hero showing the same content across all three traits.

The trained-arm #498 means are NOT plotted as side-by-side bars (the cross-Q-bank
unpaired comparison invites the same misreading the body is correcting); they
appear ONLY as a horizontal reference line per panel with an explicit
"different Q-bank, unpaired" disclosure in the caption.

CLI:
    uv run python scripts/plot_i517_v2_figures.py \
        --in eval_results/issue_517/base_vs_trained_comparison.json \
        --out-dir figures/issue_517
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIT_TITLE = {
    "logical_and_pushes_back": "Pushes back on bad premises",
    "validating": "Validates feelings before advising",
    "explains_well": "Explains clearly",
}

BAR_KEYS = ("base_in_scenario", "base_default_assistant")
BAR_LABELS = {
    "base_in_scenario": "Base, scenario header",
    "base_default_assistant": "Base, default assistant",
}


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=REPO_ROOT,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _panel(ax, block, role_for_bars, pass_threshold: float, scenario_paired) -> None:
    means = []
    sems = []
    labels = []
    colors = []
    for key in BAR_KEYS:
        cell = block.get(key, {})
        means.append(cell.get("mean", float("nan")))
        sems.append(cell.get("sem", 0.0))
        labels.append(BAR_LABELS[key])
        colors.append(role_for_bars[key])
    xs = np.arange(len(BAR_KEYS))
    bars = ax.bar(
        xs,
        means,
        yerr=sems,
        color=colors,
        capsize=4,
        edgecolor="white",
        linewidth=0.8,
        width=0.55,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(1.0, 5.2)

    # PASS threshold reference
    ax.axhline(
        pass_threshold,
        color="#b03a2e",
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.text(
        -0.45,
        pass_threshold + 0.07,
        "PASS threshold (3.5)",
        fontsize=8,
        color="#b03a2e",
        ha="left",
        va="bottom",
    )

    # Annotate mean above each bar
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            means[i] + sems[i] + 0.07,
            f"{means[i]:.2f}",
            fontsize=9,
            ha="center",
            va="bottom",
            weight="semibold",
        )

    # Annotate the paired scenario-header effect inside the plotting area,
    # tucked into a low region of the panel so it never collides with the
    # title or the bars.
    if scenario_paired is not None:
        delta, p = scenario_paired
        sig_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
        sig_tag = sig_str if p < 0.05 else f"n.s., {sig_str}"
        tag = (
            f"Within-#517 paired scenario-header effect (same prompts):\n"
            f"+{delta:.2f} Likert  ({sig_tag})"
        )
        ax.text(
            0.5,
            0.04,
            tag,
            fontsize=8.5,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="#2c3e50",
            bbox=dict(
                facecolor="#f4ecd8",
                edgecolor="#b8a878",
                boxstyle="round,pad=0.3",
                alpha=0.95,
            ),
        )


def _save(fig, out_dir: Path, name: str, input_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    meta_path = out_dir / f"{name}.meta.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": "i517_v2_round2",
                "kind": "per_trait_headroom_figure_meta",
                "input": str(input_path.resolve()),
                "png": str(png_path.resolve()),
                "pdf": str(pdf_path.resolve()),
                "git_commit": _git_sha(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="input_path", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")

    role_for_bars = {
        "base_in_scenario": paper_palette_role("primary"),
        "base_default_assistant": paper_palette_role("baseline"),
    }

    payload = json.loads(Path(args.input_path).read_text())
    per_trait = payload["per_trait"]
    pass_threshold = payload.get("pass_threshold", 3.5)
    out_dir = Path(args.out_dir)

    # Within-#517 paired scenario-header effects (same Q-bank, same model).
    # These are computed from base_headroom_judge.json directly; hard-coded here
    # so the figure script is self-contained, but the values are verified in the
    # round-2 analysis (see body.md).
    scenario_paired_effect = {
        "logical_and_pushes_back": (0.075, 0.5497),
        "validating": (0.467, 0.001689),
        "explains_well": (0.333, 0.000209),
    }

    # --- Per-trait headroom figures ---
    for trait, block in per_trait.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        _panel(ax, block, role_for_bars, pass_threshold, scenario_paired_effect.get(trait))
        ax.set_ylabel("Claude Sonnet 4.5 Likert (1-5)")
        title = TRAIT_TITLE.get(trait, trait)
        set_title_subtitle(
            ax,
            title=title,
            subtitle=("Untrained Qwen-2.5-7B-Instruct, N=40 prompts per bar"),
            source=None,
        )
        _save(fig, out_dir, f"trait_{trait}_v2round2", Path(args.input_path))

    # --- Combined 1x3 hero ---
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for ax, trait in zip(
        axes, ("logical_and_pushes_back", "validating", "explains_well"), strict=False
    ):
        block = per_trait.get(trait, {})
        _panel(ax, block, role_for_bars, pass_threshold, scenario_paired_effect.get(trait))
        ax.set_title(TRAIT_TITLE.get(trait, trait), fontsize=11, weight="semibold")
    axes[0].set_ylabel("Claude Sonnet 4.5 Likert (1-5)")
    fig.suptitle(
        "Untrained-base Likert score on the #498 per-trait rubrics, by trait and eval context",
        fontsize=12,
        weight="semibold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    _save(fig, out_dir, "hero_per_trait_v2round2", Path(args.input_path))


if __name__ == "__main__":
    main()
