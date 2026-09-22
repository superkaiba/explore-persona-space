#!/usr/bin/env python3
"""Plot labeled, multi-layer cosine-versus-uptake scatters from verified paired rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter, ScalarFormatter  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval_results/issue_2673/deepseek_comparison/singleton_comparison.json"
RATES = ROOT / "eval_results/issue_2673/deepseek_comparison/published_rates_and_overlap.json"
OUTPUT = ROOT / "figures/issue_2673/leakage_layers"
MODELS = {
    "deepseek": ("DeepSeek-V3.1-Base", (15, 30, 45, 60)),
    "qwen": ("Qwen3.8-27B", (15, 31, 47, 63)),
}
CHARACTERS = {
    "dismissive": ("Dismissive", "#D55E00", "D"),
    "sarcastic": ("Sarcastic", "#0072B2", "o"),
    "saboteur": ("Saboteur", "#AA558F", "^"),
    "peer": ("Peer", "#009E73", "s"),
    "help_seeker": ("Help-seeker", "#B47700", "v"),
}
INK = "#27303D"


def get_panel(comparison: dict, rates: dict, model: str, persona: str, metric: str, layer: int):
    """Return five ordered points and independently checked direct-uptake statistics."""
    key = (model, persona, metric, "full_bank", layer)

    def matches(row: dict) -> bool:
        """Identify the exact model/persona/metric/fit/layer cell."""
        return (
            tuple(row[field] for field in ("model", "evaluation_persona", "metric", "fit", "layer"))
            == key
        )

    points = [row for row in comparison["pairs"] if matches(row)]
    records = [row for row in comparison["records"] if matches(row)]
    assert len(points) == 5 and {p["alternative"] for p in points} == set(CHARACTERS), key
    assert len(records) == 1 and records[0]["n"] == 5, key
    points.sort(key=lambda p: list(CHARACTERS).index(p["alternative"]))
    x = np.array([p["other_similarity"] for p in points])
    y = np.array([p["other_rate"] for p in points])
    assert np.isfinite(x).all() and np.isfinite(y).all(), key
    assert (np.abs(x) <= 1 + 1e-12).all() and ((y >= 0) & (y <= 1)).all(), key
    r, rho = float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)
    np.testing.assert_allclose(
        [r, rho],
        [records[0]["secondary_other_pearson_r"], records[0]["secondary_other_spearman_rho"]],
        rtol=0,
        atol=1e-12,
        err_msg=str(key),
    )
    for point in points:
        published = rates["rates"][persona][point["alternative"]]["other"]
        assert point["other_rate"] == published["rate"], (key, point["alternative"])
        assert 0 < published["digitization_bound"] < 1, key
    return points, r, rho


def plot_panel(
    ax, points: list[dict], rates: dict, persona: str, metric: str, layer: int, r: float, rho: float
) -> None:
    """Draw an exact-coordinate scatter with direct labels and digitization bounds."""
    xs = np.array([p["other_similarity"] for p in points])
    span = float(np.ptp(xs))
    assert span > 0
    ax.set_xlim(float(xs.min() - 0.14 * span), float(xs.max() + 0.14 * span))
    ax.set_ylim(0, 13.5 if persona == "hhh" else 72)
    midpoint = float((xs.max() + xs.min()) / 2)
    for point in points:
        name = point["alternative"]
        label, color, marker = CHARACTERS[name]
        x, y = point["other_similarity"], 100 * point["other_rate"]
        bound = 100 * rates["rates"][persona][name]["other"]["digitization_bound"]
        ax.errorbar(x, y, yerr=bound, fmt="none", color=color, capsize=3, lw=1.25, zorder=2)
        ax.scatter(x, y, s=52, marker=marker, color=color, edgecolor="white", lw=0.55, zorder=3)
        left = x > midpoint
        dy = {"sarcastic": 8, "peer": 8, "help_seeker": -10 if persona == "hhh" else 0}.get(name, 0)
        ax.annotate(
            label,
            (x, y),
            xytext=(-8 if left else 8, dy),
            textcoords="offset points",
            ha="right" if left else "left",
            va="center",
            fontsize=10.5,
            color=INK,
        )
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {r:+.2f}   Spearman ρ = {rho:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=INK,
    )
    ax.set_title(f"Block {layer}", loc="left", fontsize=13, fontweight="medium", pad=10)
    ax.set_xlabel("Cosine similarity" if metric == "raw" else "Whitened cosine similarity")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_powerlimits((-3, 4))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_yticks([0, 3, 6, 9, 12] if persona == "hhh" else [0, 15, 30, 45, 60])
    ax.grid(axis="y", color="#E8EBEF", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#C1C7CF")
    ax.tick_params(axis="both", length=3, color="#A7AFB9", labelsize=10)


def main() -> None:
    """Render all 32 full-bank panels; save PNG/PDF and exact-value provenance."""
    source_bytes, rate_bytes = SOURCE.read_bytes(), RATES.read_bytes()
    comparison, rates = json.loads(source_bytes), json.loads(rate_bytes)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans"],
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.titlecolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    panel_count = 0
    for model, (model_label, layers) in MODELS.items():
        for persona in ("hhh", "fred"):
            persona_label = "HHH" if persona == "hhh" else "Fred"
            fig, axes = plt.subplots(2, 4, figsize=(17, 9.2), sharey=True)
            fig.subplots_adjust(
                left=0.065, right=0.985, bottom=0.155, top=0.805, hspace=0.52, wspace=0.20
            )
            fig.text(0.065, 0.962, f"{model_label} · {persona_label}", fontsize=22, weight="bold")
            fig.text(
                0.065,
                0.915,
                "Persona similarity vs. published character-tracer uptake",
                fontsize=15,
            )
            fig.text(0.065, 0.87, "Raw cosine", fontsize=14, weight="bold")
            fig.text(0.065, 0.482, "Uncentered whitened cosine", fontsize=14, weight="bold")
            panels = []
            for row, metric in enumerate(("raw", "whitened")):
                for col, layer in enumerate(layers):
                    points, r, rho = get_panel(comparison, rates, model, persona, metric, layer)
                    plot_panel(axes[row, col], points, rates, persona, metric, layer, r, rho)
                    panels.append(
                        {
                            "metric": metric,
                            "block_zero_based": layer,
                            "n": len(points),
                            "pearson_r": r,
                            "spearman_rho": rho,
                            "points": [
                                {
                                    "character": p["alternative"],
                                    "cosine": p["other_similarity"],
                                    "uptake_percent": p["other_rate"] * 100,
                                    "digitization_bound_percentage_points": 100
                                    * rates["rates"][persona][p["alternative"]]["other"][
                                        "digitization_bound"
                                    ],
                                }
                                for p in points
                            ],
                        }
                    )
                    panel_count += 1
                axes[row, 0].set_ylabel("Published DeepSeek tracer uptake")
            footer = (
                f"x: cosine({persona_label}, labeled character), using persona-mean context vectors. "
                "y: that character’s tracer uptake under " + persona_label + ".\n"
                "Five character conditions per panel. Full-bank fit; no mean subtraction. "
                "Blocks are zero-based; x-axis ranges vary by panel.\n"
                "Whiskers show digitization bounds (±0.34 percentage points), not sampling CIs. "
                "HHH/Fred y-axis ranges differ.\n"
                + (
                    "Qwen geometry is compared with the paper’s DeepSeek behavior; Qwen leakage "
                    "was not measured."
                    if model == "qwen"
                    else "DeepSeek geometry is from before story fine-tuning; published behavior is "
                    "from after fine-tuning."
                )
            )
            fig.text(0.065, 0.022, footer, fontsize=10, color="#515D6B", linespacing=1.55)
            stem = OUTPUT / f"{model}_{persona}"
            title = f"{model_label}: cosine versus character-tracer uptake under {persona_label}"
            fig.savefig(stem.with_suffix(".png"), dpi=220)
            fig.savefig(
                stem.with_suffix(".pdf"),
                metadata={
                    "Title": title,
                    "Author": "Explore Persona Space",
                    "Subject": "Five labeled character conditions at four blocks; uncentered metrics",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
            provenance = {
                "model": model,
                "evaluation_persona": persona,
                "fit": "full_bank",
                "x": "other_similarity",
                "y": "other_rate * 100",
                "source": str(SOURCE.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "published_rates_source": str(RATES.relative_to(ROOT)),
                "published_rates_sha256": hashlib.sha256(rate_bytes).hexdigest(),
                "panels": panels,
                "notes": footer,
                "output_sha256": {
                    ext: hashlib.sha256(stem.with_suffix(f".{ext}").read_bytes()).hexdigest()
                    for ext in ("png", "pdf")
                },
            }
            stem.with_suffix(".meta.json").write_text(json.dumps(provenance, indent=2) + "\n")
            plt.close(fig)
            print(f"Rendered and validated {stem.relative_to(ROOT)}: 8 panels, 40 points")
    assert panel_count == 32, panel_count


if __name__ == "__main__":
    main()
