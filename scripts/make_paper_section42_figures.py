#!/usr/bin/env python3
"""Render the three publication figures used in Results Section 4.2.

The figures are the SAE feature-property panels, the minimal-pair
predicted-over-observed shift-size figure, and the appendix
refusal-swaps-by-class companion.  (The qualitative retrieval-failure figure
c3_qualitative_discrimination is produced by
scripts/issue1901_qualitative_retrieval_failures.py.) The
script is plot-only: it reads checked-in summaries and per-pair records,
performs a deterministic bootstrap only for the one-word pilot intervals that
were not banked in its summary, and writes vector PDF, color PNG, grayscale PNG,
and provenance JSON for each figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the matplotlib/numpy imports below. On the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and
# the BLAS pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    ROLES,
    STYLE_VERSION,
    c2a_figure,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)


DEFAULT_OUT = ROOT / "figures/paper"
SAE_SOURCE = ROOT / "eval_results/issue_1482/plot4_redesign/plot4_redesign.json"
MINPAIR_SOURCE = ROOT / "eval_results/issue_2564/minpair_delta.json"
PERSONA_SOURCE = ROOT / "eval_results/issue_2564/floor-failed-reelicitation/minpair_delta_ffr.json"
ONEWORD_SOURCE = ROOT / "eval_results/issue_2564/lang_oneword_pilot/summary.json"
ONEWORD_PAIRS = ROOT / "eval_results/issue_2564/lang_oneword_pilot/perpair.jsonl"
SVMP_DIR = Path(os.environ.get("C2A_SVMP_DIR", ROOT / "eval_results/issue_2617/svmp_verbharm"))

LINEAR = ROLES["linear"].color
# Controls / null references take the paper-wide control role (muted gray).
CONTROL = ROLES["control"].color


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
    outputs: dict[str, Path | dict],
    title: str,
    subject: str,
    sources: list[Path],
    displayed_data: dict,
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
                "render": outputs["record"],
                "displayed_data": displayed_data,
                "output_sha256": {
                    kind: _sha256(path) for kind, path in outputs.items() if isinstance(path, Path)
                },
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
    include_frac: float,
    sources: list[Path],
    displayed_data: dict,
) -> dict[str, Path | dict]:
    stem = out_dir / stem_name
    outputs = save_c2a_figure(
        fig,
        stem,
        title=title,
        subject=subject,
        creator="scripts/make_paper_section42_figures.py",
        include_width=include_frac,
    )
    metadata = _write_metadata(
        stem=stem,
        outputs=outputs,
        title=title,
        subject=subject,
        sources=sources,
        displayed_data=displayed_data,
    )
    return {**outputs, "metadata": metadata}


def _sae_data() -> dict:
    source = json.loads(SAE_SOURCE.read_text())
    all_rows = source["left_panel"]["rows"]
    hidden_control = all_rows[0]
    assert hidden_control["banked_name"] == "Fires on BOTH context and answer side"
    rows = all_rows[1:6]
    assert [hidden_control["label"], *[row["label"] for row in rows]] == source["left_panel"][
        "rendered_labels"
    ]

    rows = [{**row, "kind": "forward-selected association"} for row in rows]
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
        "hidden_control": hidden_control,
        "tiers": tiers,
        "spearman_raw": float(source["right_panel"]["spearman_tier_r2_raw"]),
        "spearman_adjusted": float(source["right_panel"]["spearman_tier_r2_activity_centered"]),
        "centering": source["right_panel"]["centering"],
    }


def make_sae_figure(data: dict) -> tuple[plt.Figure, float]:
    fig, include_frac = c2a_figure("full", aspect=0.45)
    grid = fig.add_gridspec(1, 2, left=0.315, right=0.985, top=0.75, bottom=0.16, wspace=0.30)
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
    ax_left.set_xlabel("Concordance with feature $R^2$, above chance")
    style_axis(ax_left, grid_axis="x")
    panel_header(
        ax_left,
        "A",
        "Forward-selected associations",
        "Feature-property concordance",
        kicker_y=1.21,
        title_y=1.08,
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
    style_axis(ax_right, grid_axis="y")
    panel_header(
        ax_right,
        "B",
        "Median and interquartile range",
        "Median feature $R^2$ by tier",
        kicker_y=1.21,
        title_y=1.08,
    )
    return fig, include_frac


def _read_jsonl(path: Path) -> list[dict]:
    # str.split("\n"), not splitlines(): JSON strings may contain U+2028/U+2029, which
    # splitlines() treats as row breaks and would corrupt the record.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def _bootstrap_slope(
    all_rows: list[dict], axis: str, *, n_boot: int = 10_000, seed: int = 21620
) -> dict:
    """Through-origin norm slope + 95% pair-bootstrap CI for one pilot axis."""
    rows = [row for row in all_rows if row["axis"] == axis]
    obs = np.asarray([row["norm_obs_tail_L19"] for row in rows], dtype=float)
    pred = np.asarray([row["norm_pred_arm_779ce"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    slopes = np.sum(obs[draws] * pred[draws], axis=1) / np.sum(obs[draws] ** 2, axis=1)
    return {
        "n": len(rows),
        "slope": float(np.sum(obs * pred) / np.sum(obs**2)),
        "slope_ci95": np.quantile(slopes, [0.025, 0.975]).tolist(),
        "bootstrap": {"unit": "pair", "draws": n_boot, "seed": seed},
    }


def _pair_shift_data() -> list[dict]:
    parent = json.loads(MINPAIR_SOURCE.read_text())
    persona = json.loads(PERSONA_SOURCE.read_text())
    oneword = json.loads(ONEWORD_SOURCE.read_text())
    pilot_rows = _read_jsonl(ONEWORD_PAIRS)

    one = _bootstrap_slope(pilot_rows, "query_content_oneword")
    lang = _bootstrap_slope(pilot_rows, "answer_language")
    assert one["n"] == 24 and lang["n"] == 72
    for axis, computed in (("query_content_oneword", one), ("answer_language", lang)):
        assert np.isclose(computed["slope"], oneword["calibration_slope"]["arm_779ce"][axis])

    def banked(source: dict, axis: str, label: str) -> dict:
        cell = source["axes"][axis]
        cal = cell["calibration"]["arm_779ce"]
        return {
            "label": label,
            "axis": axis,
            "n": int(cell["n_primary_pairs"]),
            "slope": float(cal["axis_slope"]),
            "slope_ci95": [float(v) for v in cal["axis_slope_ci95"]],
        }

    rows = [
        banked(parent, "format", "Output\nformat"),
        banked(persona, "persona", "Persona"),
        banked(parent, "register", "Tone"),
        {"label": "Answer\nlanguage", "axis": "answer_language", **lang},
        banked(parent, "query_content", "Question\ntopic"),
        {"label": "One-word\ntopic", "axis": "query_content_oneword", **one},
    ]
    rows.sort(key=lambda row: -row["slope"])
    return rows


def make_pair_shift_figure(rows: list[dict]) -> tuple[plt.Figure, float]:
    fig, include_frac = c2a_figure("wide", aspect=0.53)
    grid = fig.add_gridspec(1, 1, left=0.115, right=0.98, top=0.78, bottom=0.155)
    ax = fig.add_subplot(grid[0, 0])

    x = np.arange(len(rows))
    values = np.asarray([row["slope"] for row in rows])
    ci = np.asarray([row["slope_ci95"] for row in rows])
    yerr = np.vstack([values - ci[:, 0], ci[:, 1] - values])
    ax.axhline(1.0, color=CONTROL, lw=1.7, linestyle=(0, (5, 4)), zorder=1)
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
    for xi, value, hi in zip(x, values, ci[:, 1], strict=True):
        ax.text(
            xi, hi + 0.035, f"{value:.2f}", fontsize=14, fontweight=650, ha="center", va="bottom"
        )
    ax.set_xticks(x, [row["label"] for row in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(0.45, 1.42)
    ax.set_yticks([0.5, 0.75, 1.0, 1.25])
    ax.set_ylabel("Predicted / observed shift size")
    style_axis(ax, grid_axis="y")
    panel_header(
        ax,
        "",
        "Controlled minimal pairs · Qwen2.5-7B-Instruct · layer 19\nError bars: 95% bootstrap CI",
        "Predicted over observed answer-shift size by element",
        kicker_y=1.20,
        title_y=1.075,
    )
    return fig, include_frac


REFUSAL_CLASS_LABELS = (
    ("obj_flip", "Object swap\n(flips refusal)"),
    ("verb_flip", "Verb swap\n(flips refusal)"),
    ("xstest", "XSTest\n(unsafe / safe)"),
    ("verb_harm", "Verb swap\n(both harmful)"),
    ("subj_ctl", "Subject swap\n(harmful)"),
    ("benign", "Benign\nswaps"),
)
BENIGN_SWAP_CLASSES = ("obj_benign", "verb_benign", "subj_benign")


def _boot_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    """95% pair-bootstrap CI of the mean (resampled pair indices)."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    mean = np.mean(values[idx], axis=1)
    return [float(np.percentile(mean, 2.5)), float(np.percentile(mean, 97.5))]


def _boot_slope(
    obs: np.ndarray, pred: np.ndarray, rng: np.random.Generator, n_boot: int
) -> list[float]:
    """95% pair-bootstrap CI of the through-origin norm slope."""
    idx = rng.integers(0, len(obs), size=(n_boot, len(obs)))
    slopes = np.sum(obs[idx] * pred[idx], axis=1) / np.sum(obs[idx] ** 2, axis=1)
    return [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]


def _refusal_group_stats(
    rs: list[dict], label: str, null_p95: float | None, rng: np.random.Generator, n_boot: int
) -> dict:
    """Mean-statistic reads for one group of #2617 safety-swap pairs.

    Uses the same three quantities as the #2564 minimal pairs: through-origin
    slope of predicted on observed shift norm (tail pooling), mean
    shift-direction cosine, and mean loading of the predicted shift on the
    refusal direction. Intervals are pair bootstraps.
    """
    obs = np.asarray([r["norm_obs_tail"] for r in rs], dtype=float)
    pred = np.asarray([r["norm_pred_arm_779ce"] for r in rs], dtype=float)
    base_pred = np.asarray([r["norm_pred_arm_iddelta"] for r in rs], dtype=float)
    cos = np.asarray([r["cos_arm_779ce"] for r in rs], dtype=float)
    base_cos = np.asarray([r["cos_arm_iddelta"] for r in rs], dtype=float)
    load = np.asarray([r["axis_cos_pred_arm_779ce"] for r in rs], dtype=float)
    base_load = np.asarray([r["axis_cos_pred_arm_iddelta"] for r in rs], dtype=float)
    return {
        "label": label,
        "n": len(rs),
        "slope": float(np.sum(obs * pred) / np.sum(obs**2)),
        "slope_ci95": _boot_slope(obs, pred, rng, n_boot),
        "base_slope": float(np.sum(obs * base_pred) / np.sum(obs**2)),
        "base_slope_ci95": _boot_slope(obs, base_pred, rng, n_boot),
        "mean_cos": float(np.mean(cos)),
        "mean_cos_ci95": _boot_mean(cos, rng, n_boot),
        "base_cos": float(np.mean(base_cos)),
        "base_cos_ci95": _boot_mean(base_cos, rng, n_boot),
        "null_cos_q": None if null_p95 is None else [0.0, float(null_p95)],
        "load": float(np.mean(load)),
        "load_ci95": _boot_mean(load, rng, n_boot),
        "base_load": float(np.mean(base_load)),
        "base_load_ci95": _boot_mean(base_load, rng, n_boot),
        "median_cos": float(np.median(cos)),
        "median_load": float(np.median(load)),
        "r10_mean": float(np.mean([r["r10"] for r in rs])),
    }


def _refusal_by_class_data(*, n_boot: int = 10_000, seed: int = 26170) -> dict:
    """Per-class reads (mean statistics) for the appendix refusal-swaps companion."""
    rows = _read_jsonl(SVMP_DIR / "perpair.jsonl")
    summary = json.loads((SVMP_DIR / "summary.json").read_text())
    null_p95 = summary["per_arm"]["arm_779ce"]["by_class_p95"]
    rng = np.random.default_rng(seed)
    present = {r["pair_class"] for r in rows}
    classes = []
    for key, label in REFUSAL_CLASS_LABELS:
        if key == "benign":
            rs = [r for r in rows if r["pair_class"] in BENIGN_SWAP_CLASSES]
            null = max(null_p95[c] for c in BENIGN_SWAP_CLASSES if c in null_p95)
        else:
            if key not in present:
                continue
            rs = [r for r in rows if r["pair_class"] == key]
            null = null_p95.get(key)
        classes.append({"key": key, **_refusal_group_stats(rs, label, null, rng, n_boot)})
    return {"classes": classes}


def _class_point_panel(
    ax: plt.Axes,
    rows: list[dict],
    *,
    value_key: str,
    ci_key: str,
    reference: float,
    ylabel: str,
    letter: str,
    kicker: str,
    title: str,
    ylim: tuple[float, float],
    yticks: list[float],
    base_key: str,
    base_ci_key: str,
    null_key: str | None = None,
    label_side: str = "above",
    kicker_y: float,
    title_y: float,
) -> None:
    """One per-class panel: linear-map points, raw context-shift squares, optional null band."""
    x = np.arange(len(rows))
    values = np.asarray([row[value_key] for row in rows])
    ci = np.asarray([row[ci_key] for row in rows])
    yerr = np.vstack([np.maximum(0.0, values - ci[:, 0]), np.maximum(0.0, ci[:, 1] - values)])
    ax.axhline(reference, color=INK, lw=1.4, linestyle=(0, (5, 4)), zorder=1)
    if null_key is not None:
        for xi, row in zip(x, rows, strict=True):
            band = row.get(null_key)
            if band is None:
                continue
            ax.fill_between(
                [xi - 0.36, xi + 0.36], band[0], band[1], color=MUTED, alpha=0.22, lw=0, zorder=1
            )
    base = np.asarray([row[base_key] for row in rows])
    base_ci = np.asarray([row[base_ci_key] for row in rows])
    base_err = np.vstack(
        [np.maximum(0.0, base - base_ci[:, 0]), np.maximum(0.0, base_ci[:, 1] - base)]
    )
    base_x = x + 0.22
    ax.errorbar(
        base_x,
        base,
        yerr=base_err,
        fmt="s",
        color=CONTROL,
        markerfacecolor=PAPER,
        markeredgecolor=CONTROL,
        markeredgewidth=2.0,
        markersize=8,
        capsize=5,
        capthick=1.8,
        elinewidth=1.8,
        lw=0,
        zorder=2,
    )
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
    for xi, value, lo, hi in zip(x, values, ci[:, 0], ci[:, 1], strict=True):
        if label_side == "below":
            anchor, dy, va = min(value, lo), -5, "top"
        else:
            anchor, dy, va = max(value, hi), 5, "bottom"
        ax.annotate(
            f"{value:.2f}",
            (xi, anchor),
            xytext=(0, dy),
            textcoords="offset points",
            fontsize=14,
            fontweight=650,
            ha="center",
            va=va,
        )
    ax.set_xticks(x, [row["label"] for row in rows])
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlim(-0.5, len(rows) - 0.25)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    style_axis(ax, grid_axis="y")
    panel_header(ax, letter, kicker, title, kicker_y=kicker_y, title_y=title_y)


def make_refusal_by_class_figure(data: dict) -> tuple[plt.Figure, float]:
    classes = data["classes"]
    fig, include_frac = c2a_figure("wide", aspect=1.10)
    grid = fig.add_gridspec(3, 1, left=0.115, right=0.985, top=0.85, bottom=0.07, hspace=0.8)
    ax_cos = fig.add_subplot(grid[0, 0])
    ax_load = fig.add_subplot(grid[1, 0])
    ax_slope = fig.add_subplot(grid[2, 0])
    _class_point_panel(
        ax_cos,
        classes,
        value_key="mean_cos",
        ci_key="mean_cos_ci95",
        reference=0.0,
        ylabel="Mean cosine",
        letter="A",
        kicker="Shift direction, by pair class",
        title="Cosine of predicted and observed shift",
        ylim=(-0.06, 1.14),
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        base_key="base_cos",
        base_ci_key="base_cos_ci95",
        null_key="null_cos_q",
        kicker_y=1.32,
        title_y=1.13,
    )
    _class_point_panel(
        ax_load,
        classes,
        value_key="load",
        ci_key="load_ci95",
        reference=0.0,
        ylabel="Mean projection",
        letter="B",
        kicker="Refusal-direction projection, by pair class",
        title="Projection on the refusal direction",
        ylim=(-0.32, 0.98),
        yticks=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
        base_key="base_load",
        base_ci_key="base_load_ci95",
        kicker_y=1.32,
        title_y=1.13,
    )
    top = max(
        max(row["base_slope_ci95"][1] for row in classes),
        max(row["slope_ci95"][1] for row in classes),
    )
    ymax = float(np.ceil((top + 0.08) * 4) / 4)
    _class_point_panel(
        ax_slope,
        classes,
        value_key="slope",
        ci_key="slope_ci95",
        reference=1.0,
        ylabel="Size ratio",
        letter="C",
        kicker="Shift magnitude, by pair class",
        title="Predicted over observed shift size",
        ylim=(0.4, ymax),
        yticks=[float(t) for t in np.arange(0.5, ymax + 1e-9, 0.5)],
        base_key="base_slope",
        base_ci_key="base_slope_ci95",
        label_side="below",
        kicker_y=1.32,
        title_y=1.13,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=LINEAR,
            markerfacecolor=LINEAR,
            markeredgecolor=LINEAR,
            markersize=9,
            lw=0,
            label="Linear map",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color=CONTROL,
            markerfacecolor=PAPER,
            markeredgecolor=CONTROL,
            markeredgewidth=2.0,
            markersize=8,
            lw=0,
            label="Raw context shift",
        ),
        Patch(facecolor=MUTED, alpha=0.22, label="Shuffled-pair null (95%)"),
    ]
    row_y = 0.985
    legend_kicker(fig, 0.115, row_y, "Prediction")
    fig.legend(
        handles=handles[:2],
        loc="upper left",
        bbox_to_anchor=(0.114, row_y - 0.012),
        ncol=2,
        frameon=False,
        columnspacing=1.45,
        handlelength=1.4,
        handletextpad=0.65,
        borderaxespad=0,
    )
    legend_kicker(fig, 0.62, row_y, "Null")
    fig.legend(
        handles=handles[2:],
        loc="upper left",
        bbox_to_anchor=(0.619, row_y - 0.012),
        ncol=1,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.65,
        borderaxespad=0,
    )
    return fig, include_frac


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    set_c2a_style()

    sae = _sae_data()
    sae_fig, sae_frac = make_sae_figure(sae)
    sae_outputs = _save(
        sae_fig,
        args.out_dir,
        "c3_sae_tier_gradient",
        title="SAE feature properties and context-to-answer predictability",
        subject="Conditional feature-property associations and activity-adjusted nested-tier gradient",
        include_frac=sae_frac,
        sources=[SAE_SOURCE],
        displayed_data=sae,
    )
    plt.close(sae_fig)

    pair_shifts = _pair_shift_data()
    pair_fig, pair_frac = make_pair_shift_figure(pair_shifts)
    pair_outputs = _save(
        pair_fig,
        args.out_dir,
        "c3_pair_shifts",
        title="Predicted over observed answer-shift size for controlled minimal pairs",
        subject=(
            "Through-origin calibration slope of predicted over observed answer-shift size "
            "per changed element, with 95% confidence intervals"
        ),
        include_frac=pair_frac,
        sources=[MINPAIR_SOURCE, PERSONA_SOURCE, ONEWORD_SOURCE, ONEWORD_PAIRS],
        displayed_data={
            "elements": pair_shifts,
            "reference_line": 1.0,
            "order": "descending slope",
        },
    )
    plt.close(pair_fig)

    refusal = _refusal_by_class_data()
    refusal_fig, refusal_frac = make_refusal_by_class_figure(refusal)
    refusal_outputs = _save(
        refusal_fig,
        args.out_dir,
        "c3_refusal_swaps_by_class",
        title="One-word safety swaps by pair class",
        subject=(
            "Per-class mean shift-direction cosine, refusal-direction loading, and "
            "through-origin predicted-over-observed shift size for one-word safety swaps, "
            "with the raw context-shift baseline and shuffled-pair null"
        ),
        include_frac=refusal_frac,
        sources=[SVMP_DIR / "perpair.jsonl", SVMP_DIR / "summary.json"],
        displayed_data=refusal,
    )
    plt.close(refusal_fig)

    for name, outputs in (
        ("sae", sae_outputs),
        ("pair_shifts", pair_outputs),
        ("refusal_by_class", refusal_outputs),
    ):
        for kind, path in outputs.items():
            if isinstance(path, Path):
                print(f"{name}.{kind}: {path}")


if __name__ == "__main__":
    main()
