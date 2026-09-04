#!/usr/bin/env python3
"""Render the paper figure for the issue 1739 unlabeled-data ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import t  # noqa: E402

from explore_persona_space.analysis import c2a_plot_style as style  # noqa: E402
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402


DEFAULT_SOURCE = ROOT / "eval_results/issue_1739/uladder_fold/uladder_fold.json"
DEFAULT_OUT = ROOT / "figures/paper"
DEFAULT_STEM = "c5_unlabeled_data_ladder"
BEHAVIORS = ("evil", "sycophancy", "hallucination")
SETTINGS = ("in_dist", "generic", "ood")
SETTING_LABELS = {
    "in_dist": "In-distribution",
    "generic": "Generic transfer",
    "ood": "OOD macro",
}
BEHAVIOR_LABELS = {
    "evil": "Evil",
    "sycophancy": "Sycophancy",
    "hallucination": "Hallucination",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"commit": commit, "dirty": dirty}


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(len(x)))
    half = float(t.ppf(0.975, len(x) - 1) * sem)
    return mean, mean - half, mean + half


def _series(group: dict, u_sizes: list[int]) -> dict:
    by_seed = group["ladder_by_seed"]
    points = []
    for u in u_sizes:
        values = [float(ladder[str(u)]) for ladder in by_seed.values()]
        mean, low, high = _mean_ci(values)
        points.append(
            {
                "u": u,
                "mean": mean,
                "ci95": [low, high],
                "seed_values": values,
            }
        )
    return {"config": group["config"], "points": points}


def render(source: Path, out_dir: Path, stem_name: str) -> dict:
    payload = json.loads(source.read_text())
    u_sizes = list(map(int, payload["u_sizes"]))
    groups = {
        (g["behavior"], g["setting_group"], g["config"]): g
        for g in payload["groups"]
    }
    plotted = {}
    all_bounds = [0.0]
    for behavior in BEHAVIORS:
        for setting in SETTINGS:
            key = f"{behavior}|{setting}"
            plotted[key] = [
                _series(groups[(behavior, setting, config)], u_sizes)
                for config in ("generic_only", "union_scaled")
            ]
            for series in plotted[key]:
                for point in series["points"]:
                    all_bounds.extend(point["ci95"])
    span = max(all_bounds) - min(all_bounds)
    pad = max(0.015, 0.08 * max(span, 0.05))
    y_min = min(all_bounds) - pad
    y_max = max(all_bounds) + pad

    style.set_c2a_style()
    fig, include_width = style.c2a_figure("full", aspect=0.68)
    axes = fig.subplots(3, 3, sharex=True, sharey=True)
    primary = style.ROLES["linear"]
    secondary = style.ROLES["other_source"]
    encodings = {
        "generic_only": {
            "label": "Generic-only map pool",
            "color": primary.color,
            "marker": primary.marker,
            "linestyle": "-",
        },
        "union_scaled": {
            "label": "Union-scaled comparison",
            "color": secondary.color,
            "marker": secondary.marker,
            "linestyle": "--",
        },
    }
    for row_i, behavior in enumerate(BEHAVIORS):
        for col_i, setting in enumerate(SETTINGS):
            ax = axes[row_i, col_i]
            style.style_axis(ax, grid_axis="y")
            ax.axhline(0, color=style.SEAM, lw=1.2, zorder=1)
            for series in plotted[f"{behavior}|{setting}"]:
                encoding = encodings[series["config"]]
                x = np.asarray([p["u"] for p in series["points"]], dtype=float)
                mean = np.asarray([p["mean"] for p in series["points"]])
                low = np.asarray([p["ci95"][0] for p in series["points"]])
                high = np.asarray([p["ci95"][1] for p in series["points"]])
                ax.fill_between(
                    x,
                    low,
                    high,
                    color=encoding["color"],
                    alpha=0.12,
                    linewidth=0,
                    zorder=2,
                )
                ax.plot(
                    x,
                    mean,
                    color=encoding["color"],
                    marker=encoding["marker"],
                    linestyle=encoding["linestyle"],
                    lw=2.4,
                    ms=7.5,
                    label=encoding["label"],
                    zorder=3,
                )
            ax.set_xscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([250, 1000, 5000, 18793])
            ax.set_xticklabels(["250", "1k", "5k", "18.8k"])
            style.panel_header(
                ax,
                "",
                f"{BEHAVIOR_LABELS[behavior]} · {SETTING_LABELS[setting]}",
                kicker_y=1.06,
            )
            if row_i == 2:
                ax.set_xlabel("Unlabeled map pairs, $U$")
    fig.supylabel(
        style.better_label("Mapped-answer advantage, $D$"),
        x=0.012,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=encoding["color"],
            marker=encoding["marker"],
            linestyle=encoding["linestyle"],
            lw=2.4,
            ms=7.5,
            label=encoding["label"],
        )
        for encoding in encodings.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.09, top=0.9, hspace=0.48, wspace=0.2)
    stem = out_dir / stem_name
    exported = style.save_c2a_figure(
        fig,
        stem,
        title="Mapped-answer advantage across unlabeled map-pool sizes",
        subject="Issue 1739 context-to-answer map scaling",
        creator="scripts/paper_fig_unlabeled_data_ladder.py",
        include_width=include_width,
    )
    plt.close(fig)
    sidecar = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": _sha256(source),
        "git": _git_state(),
        "plotted_values": plotted,
        "visual_encodings": encodings,
        "uncertainty": "mean across five seeds with 95% t confidence interval",
        "estimand": "D(U)=rho(mapped-answer ridge)-rho(context ridge)",
        "render": exported["record"],
        "outputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in exported.items()
            if key != "record"
        },
    }
    sidecar_path = stem.with_suffix(".meta.json")
    with atomic_replace(sidecar_path) as tmp:
        tmp.write_text(json.dumps(sidecar, indent=1, sort_keys=True))
    return {
        "pdf": str(exported["pdf"]),
        "png": str(exported["png"]),
        "grayscale": str(exported["grayscale"]),
        "meta": str(sidecar_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.import_check:
        style.set_c2a_style()
        print("[paper-fig-uladder] import-check OK")
        return 0
    outputs = render(args.source, args.out_dir, args.stem)
    print(json.dumps(outputs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
