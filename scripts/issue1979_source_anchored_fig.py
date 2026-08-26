"""#1979 — source-anchored race result: per-candidate spread and across-arm median.

Two panels (content 12 arms, marker 6 arms). One row per raced candidate,
ordered by across-arm median rho. Small dots are the per-arm within-arm Spearman
rho; the large marker is the across-arm median. The vertical line is the
across-arm median of the per-arm permutation band edges. The two source-anchored
candidates are highlighted.

Reads `source_anchored_race.json` written by `issue1979_source_anchored_race.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "eval_results/issue_1979/whiten_csls/source_anchored_race.json"
FIG_DIR = REPO_ROOT / "figures/issue_1979"
NEW = ("p2_ps", "p3a_ps")
LABEL = {
    "p1": "context sim",
    "p2": "answer sim",
    "p3a": "through-map ctx sim",
    "p3b": "through-map answer sim",
    "p4": "gate pred",
    "p5": "read-out proj (answer)",
    "p6": "read-out proj (mapped)",
    "p7": "base propensity",
    "p8a": "write forecast size",
    "p8b": "write forecast align",
    "p9": "nearest train rows (ctx)",
    "p10": "nearest train rows (ans)",
    "p2_ps": "REAL source vs target answers",
    "p3a_ps": "MAPPED source vs target answers",
}


PER_ARM_SET = ("p3b", "p2", "p10", "p9", "p2_ps", "p3a_ps")
PER_ARM_COL = {
    "p3b": "#C44E52",
    "p2": "#4C72B0",
    "p10": "#8172B2",
    "p9": "#64B5CD",
    "p2_ps": "#DD8452",
    "p3a_ps": "#55A868",
}


def _per_arm_fig(d: dict, out_dir: Path, slug: str) -> Path:
    """Per-arm dot plot — the right grain when the race has only a few arms."""
    per_arm = d["per_arm"]
    aids = list(per_arm)
    set_paper_style("iclr")
    fig, ax = plt.subplots(figsize=(6.6, 0.75 * len(aids) + 2.0))
    ys = list(range(len(aids)))[::-1]
    off = np.linspace(0.30, -0.30, len(PER_ARM_SET))

    for y, aid in zip(ys, aids, strict=True):
        r = per_arm[aid]
        band = r["band_p975_max_selected"]
        ax.plot([band, band], [y - 0.42, y + 0.42], "-", color="#666666", lw=1.0, zorder=2)
        for k, dy in zip(PER_ARM_SET, off, strict=True):
            if k not in r["rho"]:
                continue
            ax.plot(
                [r["rho"][k]],
                [y + dy],
                "o",
                ms=4.6,
                color=PER_ARM_COL[k],
                markeredgewidth=0,
                zorder=4,
            )

    ax.axvline(0.0, color="#BBBBBB", lw=0.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(aids, fontsize=6.6)
    ax.set_ylim(-0.75, len(aids) - 0.25)
    ax.set_xlabel("within-arm Spearman $\\rho$ against leakage")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=PER_ARM_COL[k], ms=4.6, label=LABEL.get(k, k))
        for k in PER_ARM_SET
    ]
    handles.append(plt.Line2D([], [], color="#666666", lw=1.0, label="permutation band (97.5%)"))
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=6.2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, slug, dir=out_dir)
    plt.close(fig)
    return paths["png"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--mode", choices=("summary", "per-arm"), default="summary")
    ap.add_argument("--slug", default=None)
    args = ap.parse_args(argv)

    d = json.loads(args.src.read_text())
    per_arm, summary = d["per_arm"], d["summary"]

    if args.mode == "per-arm":
        slug = args.slug or f"c5_source_anchored_{d.get('regime', 'all')}_perarm"
        png = _per_arm_fig(d, args.out_dir, slug)
        assert png.exists() and png.stat().st_size > 0, png
        print(f"[fig] wrote {png} ({png.stat().st_size} bytes)")
        return 0

    set_paper_style("iclr")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))

    for ax, kind in zip(axes, ("content", "marker"), strict=True):
        s = summary[kind]
        arms = [a for a, r in per_arm.items() if r["kind"] == kind]
        names = sorted(s["median_rho"], key=lambda k: s["median_rho"][k])
        band = float(np.median([per_arm[a]["band_p975_max_selected"] for a in arms]))

        for y, k in enumerate(names):
            vals = [per_arm[a]["rho"][k] for a in arms]
            is_new = k in NEW
            col = "#C44E52" if is_new else "#4C72B0"
            ax.plot(
                vals,
                [y] * len(vals),
                "o",
                ms=2.6,
                color=col,
                alpha=0.35,
                markeredgewidth=0,
                zorder=3,
            )
            ax.plot(
                [s["median_rho"][k]],
                [y],
                "D",
                ms=5.0,
                color=col,
                markeredgewidth=0,
                zorder=5,
            )

        ax.axvline(band, color="#666666", lw=1.0, zorder=2)
        ax.axvline(0.0, color="#BBBBBB", lw=0.6, zorder=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(
            [LABEL.get(k, k) for k in names],
            fontsize=6.4,
            fontweight=["normal", "bold"][0],
        )
        for tick, k in zip(ax.get_yticklabels(), names, strict=True):
            if k in NEW:
                tick.set_fontweight("bold")
                tick.set_color("#C44E52")
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.set_title(f"{kind} ({s['n_arms']} arms, K={s['k_candidates']})", fontsize=8)
        ax.set_xlabel("within-arm Spearman $\\rho$ against leakage")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#4C72B0", ms=3.4, alpha=0.5, label="per-arm"),
        plt.Line2D([], [], marker="D", ls="", color="#4C72B0", ms=5.0, label="across-arm median"),
        plt.Line2D(
            [], [], marker="D", ls="", color="#C44E52", ms=5.0, label="source-anchored (new)"
        ),
        plt.Line2D([], [], color="#666666", lw=1.0, label="median permutation band (97.5%)"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        ncol=4,
        fontsize=6.4,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c5_source_anchored_race", dir=args.out_dir)
    plt.close(fig)
    png = paths["png"]
    assert png.exists() and png.stat().st_size > 0, paths
    print(f"[fig] wrote {png} ({png.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
