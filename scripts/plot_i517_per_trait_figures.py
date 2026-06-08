"""Issue #517 per-trait figures (one finding = one figure, paper-plots `blog` style).

Builds 3 single-trait figures + 1 combined hero for the Reproducibility
appendix. Each per-trait figure shows the 4-bar comparison for that trait
(base in-scenario / base default / trained system / trained role).

Caption text is intentionally generic — the body's per-figure markdown
blockquote caption supplies the specific read.

CLI:
    uv run python scripts/plot_i517_per_trait_figures.py \
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
    "logical_and_pushes_back": "Pushes back",
    "validating": "Validating",
    "explains_well": "Explains clearly",
}

# Bar layout per panel — keep order stable across all panels (paper-plots §3.6).
BAR_KEYS = (
    "base_in_scenario",
    "base_default_assistant",
    "trained_system_in_scenario",
    "trained_role_in_scenario",
)
BAR_LABELS = {
    "base_in_scenario": "Base\nin-scenario",
    "base_default_assistant": "Base\ndefault",
    "trained_system_in_scenario": "Trained system\nin-scenario",
    "trained_role_in_scenario": "Trained role\nin-scenario",
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


def _panel(ax, block, role_for_bars, pass_threshold: float) -> None:
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
    ax.bar(
        xs,
        means,
        yerr=sems,
        color=colors,
        capsize=4,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(1.0, 5.0)
    ax.axhline(
        pass_threshold,
        color="#b03a2e",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )
    # Annotate the threshold line on the left, above the line (low-bar side
    # in the validating panel; safe in the other panels since base bars are
    # tall and the trained bars carry the body's read).
    ax.text(
        -0.45,
        pass_threshold - 0.18,
        "trait-installed threshold (3.5)",
        fontsize=8,
        color="#b03a2e",
        ha="left",
        va="top",
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
                "schema_version": "i517_v2",
                "kind": "per_trait_figure_meta",
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

    # Use paper-plots blog style for clean-result inline figures.
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")

    # Semantic role mapping — same colors in every panel.
    role_for_bars = {
        "base_in_scenario": paper_palette_role("baseline"),
        "base_default_assistant": paper_palette_role("control"),
        "trained_system_in_scenario": paper_palette_role("primary"),
        "trained_role_in_scenario": paper_palette_role("accent"),
    }

    payload = json.loads(Path(args.input_path).read_text())
    per_trait = payload["per_trait"]
    pass_threshold = payload.get("pass_threshold", 3.5)
    out_dir = Path(args.out_dir)

    # --- Per-trait figures (one per finding) ---
    for trait, block in per_trait.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        _panel(ax, block, role_for_bars, pass_threshold)
        ax.set_ylabel("Claude Sonnet 4.5 Likert (1-5)")
        title = TRAIT_TITLE.get(trait, trait)
        set_title_subtitle(
            ax,
            title=title,
            subtitle=("Untrained base vs #498-trained adapters, N=40 prompts per bar"),
            source=None,
        )
        _save(fig, out_dir, f"trait_{trait}", Path(args.input_path))

    # --- Combined 1x3 hero (for Reproducibility / paper) ---
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    for ax, trait in zip(
        axes, ("logical_and_pushes_back", "validating", "explains_well"), strict=False
    ):
        block = per_trait.get(trait, {})
        _panel(ax, block, role_for_bars, pass_threshold)
        ax.set_title(TRAIT_TITLE.get(trait, trait), fontsize=11, weight="semibold")
    axes[0].set_ylabel("Claude Sonnet 4.5 Likert (1-5)")
    fig.suptitle(
        "Base-model headroom probe: untrained Qwen-2.5-7B-Instruct vs #498-trained adapters, per trait",
        fontsize=12,
        weight="semibold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    _save(fig, out_dir, "hero_per_trait_v2", Path(args.input_path))


if __name__ == "__main__":
    main()
