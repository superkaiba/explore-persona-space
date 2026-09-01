#!/usr/bin/env python3
"""Render the three publication figures used in Results Section 4.2.

The script is plot-only: it reads checked-in summaries and per-pair records,
performs a deterministic bootstrap only for the one-word pilot intervals that
were not banked in its summary, and writes vector PDF, color PNG, grayscale PNG,
and provenance JSON for each figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    PAPER,
    PREDICTOR_STYLES,
    SEAM,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
)


DEFAULT_OUT = ROOT / "figures/paper"
QUAL_SOURCE = ROOT / "eval_results/issue_2478/selected_examples.json"
SAE_SOURCE = ROOT / "eval_results/issue_1482/plot4_redesign/plot4_redesign.json"
MINPAIR_SOURCE = ROOT / "eval_results/issue_2564/minpair_delta.json"
PERSONA_SOURCE = (
    ROOT / "eval_results/issue_2564/floor-failed-reelicitation/minpair_delta_ffr.json"
)
ONEWORD_SOURCE = ROOT / "eval_results/issue_2564/lang_oneword_pilot/summary.json"
ONEWORD_PAIRS = ROOT / "eval_results/issue_2564/lang_oneword_pilot/perpair.jsonl"

LINEAR = PREDICTOR_STYLES["ridge"].color
CONTROL = "#8B9197"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _write_metadata(
    *,
    stem: Path,
    outputs: dict[str, Path],
    title: str,
    subject: str,
    font: str,
    sources: list[Path],
    displayed_data: dict,
    rendering_size: tuple[float, float],
) -> Path:
    metadata = stem.with_suffix(".meta.json")
    metadata.write_text(
        json.dumps(
            {
                "status": "Results Section 4.2 manuscript figure",
                "style_version": STYLE_VERSION,
                "plotting_script": "scripts/make_paper_section42_figures.py",
                "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
                "reproduction_command": "uv run python scripts/make_paper_section42_figures.py",
                "title": title,
                "subject": subject,
                "git": _git_state(),
                "sources": [
                    {"path": _display_path(path), "sha256": _sha256(path)} for path in sources
                ],
                "rendering": {
                    "resolved_font": font,
                    "authoring_size_inches": list(rendering_size),
                    "intended_manuscript_width_inches": 5.5,
                    "png_dpi": 240,
                    "background": PAPER,
                },
                "displayed_data": displayed_data,
                "output_sha256": {kind: _sha256(path) for kind, path in outputs.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return metadata


def _save(
    fig: plt.Figure,
    out_dir: Path,
    stem_name: str,
    *,
    title: str,
    subject: str,
    font: str,
    sources: list[Path],
    displayed_data: dict,
    rendering_size: tuple[float, float],
) -> dict[str, Path]:
    stem = out_dir / stem_name
    outputs = save_c2a_figure(
        fig,
        stem,
        title=title,
        subject=subject,
        creator="scripts/make_paper_section42_figures.py",
    )
    metadata = _write_metadata(
        stem=stem,
        outputs=outputs,
        title=title,
        subject=subject,
        font=font,
        sources=sources,
        displayed_data=displayed_data,
        rendering_size=rendering_size,
    )
    return {**outputs, "metadata": metadata}


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _qualitative_data() -> list[dict]:
    source = json.loads(QUAL_SOURCE.read_text())
    rows = {row["example_id"]: row for row in source["panels"]["discrimination"]}
    expected = {
        "recency_prior_topic_d3::v3-v1::e10": ("works", 0.0562, 0.0111),
        "constraint_knowledge::v1-v2::e6": ("works", 0.0138, 0.0079),
        "ci2968": ("fails", 4, None),
        "ci11905": ("fails", 7, None),
    }
    for key, (verdict, value, value_b) in expected.items():
        row = rows[key]
        assert row["verdict"] == verdict
        if key.startswith("ci"):
            assert row["scores"]["rank"] == value
        else:
            assert round(row["scores"]["margin_cos_a"], 4) == value
            assert round(row["scores"]["margin_cos_b"], 4) == value_b

    return [
        {
            "column": "distinguished",
            "title": "Earlier topic survives intervening turns",
            "metric": "margins +0.0562 / +0.0111",
            "lead": 'Shared final turn: “Could you recap your main advice in one sentence?”',
            "lines": [
                "Hiking → route, weather, gear, and backup-plan advice.",
                "Birthday → treasure hunt, crafts, games, and cake ideas.",
            ],
            "source_id": "recency_prior_topic_d3::v3-v1::e10",
        },
        {
            "column": "distinguished",
            "title": "System constraint changes the answer",
            "metric": "margins +0.0138 / +0.0079",
            "lead": 'Shared final turn: “How is the stock market doing right now?”',
            "lines": [
                "No internet → says current data are unavailable.",
                "Browsing allowed → offers to look up an index.",
            ],
            "source_id": "constraint_knowledge::v1-v2::e6",
        },
        {
            "column": "confused",
            "title": "Related travel itineraries",
            "metric": "true answer rank 4",
            "lead": 'Final user turn: “puedes darme un plan de viaje a El Calafate, de 6 dias ?”',
            "lines": [
                "True answer → six-day Spanish itinerary for El Calafate.",
                "Top confuser → Portuguese itinerary for the Alps.",
            ],
            "source_id": "ci2968",
        },
        {
            "column": "confused",
            "title": "Different language tasks",
            "metric": "true answer rank 7",
            "lead": 'Final user turn: Complete “I want to learn more about the American culture by”',
            "lines": [
                "True answer → completes the requested sentence.",
                "Top confuser → explains transliteration.",
            ],
            "source_id": "ci11905",
        },
    ]


def _qual_card(ax: plt.Axes, row: dict, *, panel_letter: str, first_in_column: bool) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = FancyBboxPatch(
        (0.015, 0.035),
        0.97,
        0.91,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=SEAM,
        facecolor=PAPER,
    )
    ax.add_patch(box)
    if first_in_column:
        ax.text(
            0.02,
            1.13,
            f"{panel_letter}  ·  {row['column'].upper()}",
            fontsize=13,
            fontweight=750,
            color=MUTED,
            ha="left",
            va="bottom",
        )
    ax.text(0.055, 0.855, row["title"], fontsize=18.5, fontweight=650, ha="left", va="top")
    ax.text(
        0.945,
        0.735,
        row["metric"],
        fontsize=14.5,
        color=LINEAR if row["column"] == "distinguished" else MUTED,
        fontweight=650,
        ha="right",
        va="top",
    )
    ax.text(
        0.055,
        0.60,
        _wrap(row["lead"], 51),
        fontsize=16,
        color=MUTED,
        ha="left",
        va="top",
        linespacing=1.26,
    )
    y = 0.34
    for line in row["lines"]:
        ax.text(0.062, y, "•", fontsize=17, color=LINEAR, ha="left", va="top")
        ax.text(
            0.105,
            y,
            _wrap(line, 58),
            fontsize=15.5,
            color=INK,
            ha="left",
            va="top",
            linespacing=1.25,
        )
        y -= 0.19


def make_qualitative_figure(rows: list[dict]) -> plt.Figure:
    size = (14.4, 8.0)
    fig = plt.figure(figsize=size, constrained_layout=False)
    grid = fig.add_gridspec(2, 2, left=0.055, right=0.985, top=0.88, bottom=0.035, hspace=0.08, wspace=0.075)
    for idx, row in enumerate(rows):
        col = 0 if row["column"] == "distinguished" else 1
        local_row = idx if col == 0 else idx - 2
        ax = fig.add_subplot(grid[local_row, col])
        _qual_card(ax, row, panel_letter="A" if col == 0 else "B", first_in_column=local_row == 0)
    return fig


def _sae_data() -> dict:
    source = json.loads(SAE_SOURCE.read_text())
    rows = source["left_panel"]["rows"][:6]
    assert [row["label"] for row in rows] == source["left_panel"]["rendered_labels"]
    tiers = []
    for key, label in (("0", "Coarsest"), ("1", "Middle"), ("2", "Finest")):
        cell = source["right_panel"]["per_tier"][key]
        tiers.append(
            {
                "label": label,
                "n": int(cell["n"]),
                "median": float(cell["median_adjusted"]),
                "q25": float(cell["q25_adjusted"]),
                "q75": float(cell["q75_adjusted"]),
            }
        )
    return {
        "properties": rows,
        "tiers": tiers,
        "spearman_raw": float(source["right_panel"]["spearman_tier_r2_raw"]),
        "spearman_adjusted": float(
            source["right_panel"]["spearman_tier_r2_activity_centered"]
        ),
        "centering": source["right_panel"]["centering"],
    }


def _plain_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SEAM)
    ax.spines["bottom"].set_color(SEAM)
    ax.tick_params(length=0, pad=8)
    ax.grid(axis="x", color=GRID, lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)


def make_sae_figure(data: dict) -> plt.Figure:
    size = (14.4, 6.5)
    fig = plt.figure(figsize=size, constrained_layout=False)
    grid = fig.add_gridspec(1, 2, left=0.19, right=0.985, top=0.75, bottom=0.16, wspace=0.30)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    props = data["properties"]
    y = np.arange(len(props))[::-1]
    values = np.asarray([row["value"] for row in props])
    bars = ax_left.barh(y, values, height=0.58, color=LINEAR, edgecolor=LINEAR, linewidth=1.2)
    for bar, value in zip(bars, values, strict=True):
        if value < 0:
            bar.set_facecolor(PAPER)
            bar.set_hatch("////")
    ax_left.axvline(0, color=INK, lw=1.2)
    ax_left.set_yticks(y, [row["label"] for row in props])
    ax_left.set_xlim(-0.29, 0.31)
    ax_left.set_xticks(np.arange(-0.2, 0.31, 0.1))
    ax_left.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:+.1f}" if x else "0"))
    ax_left.set_xlabel("Conditional concordance above chance")
    _plain_axis(ax_left)
    ax_left.set_title("Feature properties track accuracy", loc="left", y=1.08, pad=0, fontweight=650)
    ax_left.text(
        0,
        1.21,
        "A  ·  FORWARD-SELECTED ASSOCIATIONS",
        transform=ax_left.transAxes,
        fontsize=13,
        fontweight=750,
        color=MUTED,
        ha="left",
        va="bottom",
    )

    tiers = data["tiers"]
    x = np.arange(3)
    med = np.asarray([row["median"] for row in tiers])
    lo = med - np.asarray([row["q25"] for row in tiers])
    hi = np.asarray([row["q75"] for row in tiers]) - med
    ax_right.errorbar(
        x,
        med,
        yerr=np.vstack([lo, hi]),
        fmt="o",
        color=LINEAR,
        markerfacecolor=LINEAR,
        markeredgecolor=LINEAR,
        markersize=9,
        capsize=7,
        capthick=2,
        elinewidth=2.4,
        lw=0,
        zorder=3,
    )
    ax_right.axhline(0, color=INK, lw=1.2)
    ax_right.set_xticks(x, [row["label"] for row in tiers])
    ax_right.set_xlim(-0.45, 2.45)
    ax_right.set_ylim(-0.19, 0.37)
    ax_right.set_yticks(np.arange(-0.1, 0.31, 0.1))
    ax_right.set_ylabel("Activity-adjusted feature $R^2$")
    ax_right.set_xlabel("Nested SAE tier")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["left"].set_color(SEAM)
    ax_right.spines["bottom"].set_color(SEAM)
    ax_right.tick_params(length=0, pad=8)
    ax_right.grid(axis="y", color=GRID, lw=1.0, alpha=0.55)
    ax_right.set_axisbelow(True)
    ax_right.set_title("Coarse features are more predictable", loc="left", y=1.08, pad=0, fontweight=650)
    ax_right.text(
        0,
        1.21,
        "B  ·  MEDIAN AND INTERQUARTILE RANGE",
        transform=ax_right.transAxes,
        fontsize=13,
        fontweight=750,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    return fig


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _bootstrap_oneword(rows: list[dict], *, n_boot: int = 10_000, seed: int = 21620) -> dict:
    rows = [row for row in rows if row["axis"] == "query_content_oneword"]
    assert len(rows) == 24
    obs = np.asarray([row["norm_obs_tail_L19"] for row in rows], dtype=float)
    pred = np.asarray([row["norm_pred_arm_779ce"] for row in rows], dtype=float)
    cos = np.asarray([row["cos_arm_779ce"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    slopes = np.sum(obs[draws] * pred[draws], axis=1) / np.sum(obs[draws] ** 2, axis=1)
    mean_cos = np.mean(cos[draws], axis=1)
    return {
        "n": len(rows),
        "slope": float(np.sum(obs * pred) / np.sum(obs**2)),
        "slope_ci95": np.quantile(slopes, [0.025, 0.975]).tolist(),
        "mean_cos": float(np.mean(cos)),
        "mean_cos_ci95": np.quantile(mean_cos, [0.025, 0.975]).tolist(),
        "r10_mean": float(np.mean([row["r10"] for row in rows])),
        "bootstrap": {"unit": "one-word pair", "draws": n_boot, "seed": seed},
    }


def _minimal_pair_data() -> list[dict]:
    parent = json.loads(MINPAIR_SOURCE.read_text())
    persona = json.loads(PERSONA_SOURCE.read_text())
    oneword = json.loads(ONEWORD_SOURCE.read_text())
    one = _bootstrap_oneword(_read_jsonl(ONEWORD_PAIRS))

    p = persona["axes"]["persona"]
    q = parent["axes"]["query_content"]
    summary_one_slope = oneword["calibration_slope"]["arm_779ce"]["query_content_oneword"]
    assert np.isclose(one["slope"], summary_one_slope)
    assert np.isclose(
        np.median(
            [
                row["cos_arm_779ce"]
                for row in _read_jsonl(ONEWORD_PAIRS)
                if row["axis"] == "query_content_oneword"
            ]
        ),
        oneword["cos_median_by_axis_arm"]["arm_779ce"]["query_content_oneword"],
    )

    return [
        {
            "label": "Persona",
            "n": int(p["n_primary_pairs"]),
            "slope": float(p["calibration"]["arm_779ce"]["axis_slope"]),
            "slope_ci95": p["calibration"]["arm_779ce"]["axis_slope_ci95"],
            "mean_cos": float(p["direction"]["arm_779ce"]["mean_cos_headline"]),
            "mean_cos_ci95": p["direction"]["arm_779ce"]["ci95"],
            "r10_mean": float(p["reliability"]["r10_mean"]),
        },
        {
            "label": "Question topic",
            "n": int(q["n_primary_pairs"]),
            "slope": float(q["calibration"]["arm_779ce"]["axis_slope"]),
            "slope_ci95": q["calibration"]["arm_779ce"]["axis_slope_ci95"],
            "mean_cos": float(q["direction"]["arm_779ce"]["mean_cos_headline"]),
            "mean_cos_ci95": q["direction"]["arm_779ce"]["ci95"],
            "r10_mean": float(q["reliability"]["r10_mean"]),
        },
        {
            "label": "One-word topic",
            **one,
        },
    ]


def _point_panel(
    ax: plt.Axes,
    rows: list[dict],
    *,
    value_key: str,
    ci_key: str,
    reference: float,
    ylabel: str,
    title: str,
    kicker: str,
    ylim: tuple[float, float],
    yticks: list[float],
) -> None:
    x = np.arange(len(rows))
    values = np.asarray([row[value_key] for row in rows])
    ci = np.asarray([row[ci_key] for row in rows])
    yerr = np.vstack([values - ci[:, 0], ci[:, 1] - values])
    ax.axhline(reference, color=CONTROL, lw=1.7, linestyle=(0, (5, 4)), zorder=1)
    ax.errorbar(
        x,
        values,
        yerr=yerr,
        fmt="o",
        color=LINEAR,
        markerfacecolor=LINEAR,
        markeredgecolor=LINEAR,
        markersize=9,
        capsize=6,
        capthick=2,
        elinewidth=2.3,
        lw=0,
        zorder=3,
    )
    for xi, value in zip(x, values, strict=True):
        ax.text(xi, value + 0.045, f"{value:.2f}", fontsize=14, fontweight=650, ha="center", va="bottom")
    ax.set_xticks(x, [row["label"] for row in rows])
    ax.set_xlim(-0.45, len(rows) - 0.55)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SEAM)
    ax.spines["bottom"].set_color(SEAM)
    ax.tick_params(length=0, pad=8)
    ax.grid(axis="y", color=GRID, lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", y=1.08, pad=0, fontweight=650)
    ax.text(
        0,
        1.21,
        kicker,
        transform=ax.transAxes,
        fontsize=13,
        fontweight=750,
        color=MUTED,
        ha="left",
        va="bottom",
    )


def make_minimal_pair_figure(rows: list[dict]) -> plt.Figure:
    size = (14.4, 6.2)
    fig = plt.figure(figsize=size, constrained_layout=False)
    grid = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.75, bottom=0.16, wspace=0.24)
    ax_slope = fig.add_subplot(grid[0, 0])
    ax_cos = fig.add_subplot(grid[0, 1])
    _point_panel(
        ax_slope,
        rows,
        value_key="slope",
        ci_key="slope_ci95",
        reference=1.0,
        ylabel="Separation calibration slope",
        title="Persona is retained; topic is compressed",
        kicker="A  ·  THROUGH-ORIGIN NORM SLOPE",
        ylim=(0.45, 1.28),
        yticks=[0.5, 0.75, 1.0, 1.25],
    )
    _point_panel(
        ax_cos,
        rows,
        value_key="mean_cos",
        ci_key="mean_cos_ci95",
        reference=0.0,
        ylabel="Mean shift-direction cosine",
        title="Shift directions remain visible",
        kicker="B  ·  PREDICTED VS. OBSERVED SHIFT",
        ylim=(-0.02, 0.88),
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8],
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    font = set_c2a_style()

    qual = _qualitative_data()
    qual_fig = make_qualitative_figure(qual)
    qual_outputs = _save(
        qual_fig,
        args.out_dir,
        "c3_qualitative_discrimination",
        title="Qualitative context-to-answer retrieval examples",
        subject="Representative distinctions and confusions from matched and natural conversations",
        font=font,
        sources=[QUAL_SOURCE],
        displayed_data={"examples": qual},
        rendering_size=(14.4, 8.0),
    )
    plt.close(qual_fig)

    sae = _sae_data()
    sae_fig = make_sae_figure(sae)
    sae_outputs = _save(
        sae_fig,
        args.out_dir,
        "c3_sae_tier_gradient",
        title="SAE feature properties and context-to-answer predictability",
        subject="Conditional feature-property associations and activity-adjusted nested-tier gradient",
        font=font,
        sources=[SAE_SOURCE],
        displayed_data=sae,
        rendering_size=(14.4, 6.5),
    )
    plt.close(sae_fig)

    minimal = _minimal_pair_data()
    minimal_fig = make_minimal_pair_figure(minimal)
    minimal_outputs = _save(
        minimal_fig,
        args.out_dir,
        "c3_persona_topic_separation",
        title="Persona and topic separation under controlled answer shifts",
        subject="Calibration slopes and shift-direction cosines for controlled minimal pairs",
        font=font,
        sources=[MINPAIR_SOURCE, PERSONA_SOURCE, ONEWORD_SOURCE, ONEWORD_PAIRS],
        displayed_data={"categories": minimal},
        rendering_size=(14.4, 6.2),
    )
    plt.close(minimal_fig)

    for name, outputs in (
        ("qualitative", qual_outputs),
        ("sae", sae_outputs),
        ("minimal_pairs", minimal_outputs),
    ):
        for kind, path in outputs.items():
            print(f"{name}.{kind}: {path}")


if __name__ == "__main__":
    main()
