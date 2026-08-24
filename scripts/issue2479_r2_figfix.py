"""Issue #2479 round-2 figure fixes (interpretation-critic round 1).

Renders three figures from the committed ``gradient_verdict.json`` (zero GPU):

- ``band_agreement_tiers`` — SUPERSEDES ``band_agreement``: plain-English
  designed-tier tick labels (the A/B/C/D codes were opaque), collision-free
  point labels, and explicit legend/series labels so the sidecar rows carry
  a ``series`` column.
- ``ceilings_identity_bias_v2`` — SUPERSEDES ``ceilings_identity_bias``:
  identical data; the sidecar rows gain an explicit ``series`` column
  (``savefig_paper``'s bar extractor keeps container labels only at group
  level, so this script deterministically re-annotates its own sidecar from
  the container order).
- ``recovery_by_tier`` — NEW: rung-4 recovery fraction grouped by designed
  tier (the tier-conditioned companion read added in round 2).

Anchor characters keep the accent color used by the round-1 figures
(one color = one meaning across the writeup).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis import paper_plots as pp  # noqa: E402

TIER_LABELS = {
    "A": "Explicitly AI",
    "B": "Helpful professional",
    "C": "Ordinary human",
    "D": "Stylized non-AI",
}
BANDS = ["A", "B", "C", "D"]
CEILING_FLOOR = 0.05
SERIES_NEW = "New character"
SERIES_ANCHOR = "Tier anchor (reused from parent rig)"


def _dodged_point_labels(ax, pts: list[tuple[float, float, str]], min_gap: float) -> None:
    """Label points at a shared x slot without overlaps.

    Alternates label side (right/left) by descending y; when two same-side
    labels would sit closer than ``min_gap`` (data units), the lower one is
    pushed down.
    """
    pts = sorted(pts, key=lambda t: -t[1])
    last_y = {1: None, -1: None}
    for i, (x, y, name) in enumerate(pts):
        side = 1 if i % 2 == 0 else -1
        label_y = y
        prev = last_y[side]
        if prev is not None and prev - label_y < min_gap:
            label_y = prev - min_gap
        last_y[side] = label_y
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(x + 0.10 * side, label_y),
            textcoords="data",
            ha="left" if side == 1 else "right",
            va="center",
            fontsize=7,
        )


def _tier_scatter(recs: list[dict], value_key: str, ylabel: str, stem: str, fig_dir: Path) -> None:
    """Per-tier labeled scatter with explicit anchor/new-character series."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    vals = [r[value_key] for r in recs]
    min_gap = 0.05 * (max(vals) - min(vals))
    for series, is_anchor, role in (
        (SERIES_NEW, False, "primary"),
        (SERIES_ANCHOR, True, "accent"),
    ):
        xs = [BANDS.index(r["design_band"]) for r in recs if r["anchor"] == is_anchor]
        ys = [r[value_key] for r in recs if r["anchor"] == is_anchor]
        ax.scatter(xs, ys, c=pp.paper_palette_role(role), s=36, zorder=3, label=series)
    for xi, band in enumerate(BANDS):
        pts = [
            (float(xi), float(r[value_key]), r["display_name"])
            for r in recs
            if r["design_band"] == band
        ]
        _dodged_point_labels(ax, pts, min_gap)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels([TIER_LABELS[b] for b in BANDS])
    ax.set_xlim(-0.5, len(BANDS) - 0.35)
    ax.set_xlabel("Designed AI-likeness tier")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="upper right" if value_key == "axis_score" else "best")
    pp.savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)


def _annotate_bar_sidecar(fig_dir: Path, stem: str, group_series: dict[int, str]) -> None:
    """Add an explicit ``series`` column to a bar sidecar's rows.

    ``savefig_paper``'s bar extractor records container labels at group level
    only; rows carry a ``_group`` index. The mapping here follows container
    draw order, so the re-annotation is deterministic for this script's own
    render.
    """
    meta_path = fig_dir / f"{stem}.meta.json"
    meta = json.loads(meta_path.read_text())
    for row in meta.get("points") or []:
        g = row.get("_group")
        if g in group_series:
            row["series"] = group_series[g]
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verdict", type=Path, default=Path("eval_results/issue_2479/gradient_verdict.json")
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_2479"))
    args = ap.parse_args()

    pp.set_paper_style("blog")
    per_char = json.loads(args.verdict.read_text())["per_character"]
    recs = sorted(per_char.values(), key=lambda r: -r["axis_score"])
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) band_agreement_tiers — judged axis score by designed tier.
    _tier_scatter(
        recs,
        "axis_score",
        "Frozen AI-likeness axis score",
        "band_agreement_tiers",
        args.fig_dir,
    )

    # 2) recovery_by_tier — rung-4 recovery fraction by designed tier.
    for r in recs:
        r["_recovery"] = r["recovery_fraction"]
    _tier_scatter(
        recs,
        "_recovery",
        "Rung-4 recovery fraction",
        "recovery_by_tier",
        args.fig_dir,
    )

    # 3) ceilings_identity_bias_v2 — identical data to round 1, explicit
    #    sidecar series labels, gentler tick rotation.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    xs = np.arange(len(recs))
    width = 0.4
    label_ceiling = "Own-map ceiling R2"
    label_identity = "Identity + learned-bias R2"
    ax.bar(
        xs - width / 2,
        [r["ceiling_r2"] for r in recs],
        width,
        label=label_ceiling,
        color=pp.paper_palette_role("primary"),
    )
    ax.bar(
        xs + width / 2,
        [r["identity_bias_r2"] for r in recs],
        width,
        label=label_identity,
        color=pp.paper_palette_role("baseline"),
    )
    ax.axhline(CEILING_FLOOR, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["display_name"] for r in recs], rotation=30, ha="right")
    ax.set_ylabel("Held-out R2")
    ax.legend(frameon=False)
    pp.savefig_paper(fig, "ceilings_identity_bias_v2", dir=args.fig_dir)
    plt.close(fig)
    _annotate_bar_sidecar(
        args.fig_dir,
        "ceilings_identity_bias_v2",
        {0: label_ceiling, 1: label_identity},
    )

    for stem in ("band_agreement_tiers", "recovery_by_tier", "ceilings_identity_bias_v2"):
        print(f"wrote {args.fig_dir / stem}.png (+ pdf, meta.json)")


if __name__ == "__main__":
    main()
