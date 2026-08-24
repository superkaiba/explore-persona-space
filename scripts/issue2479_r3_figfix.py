"""Issue #2479 round-3 figure fixes (interpretation-critique round 2).

Renders two figures from the committed ``gradient_verdict.json`` (zero GPU):

- ``gradient_hero_v2`` — SUPERSEDES ``gradient_hero``: identical data; point
  labels are placed by a renderer-measured greedy dodger so no character
  labels overlap (round-2 render collided dana/tomas and wren/priya/marcus).
- ``rungwise_ordering`` — NEW: rank correlation of per-character recovery
  with the judged axis at each of the 9 transfer-ladder rungs (from the
  committed ``r3_diagnostics.json``), the round-2 critique's rung-wise
  report as a figure (plain-English rung names).
- ``retrieval_identity_vs_transfer`` — SUPERSEDES ``gradient_hero_acc1``:
  left panel keeps the transferred-operator retrieval-recovery scatter
  (collision-free labels); the NEW right panel plots per-character raw top-1
  accuracy of the transferred operator NEXT TO the identity-plus-learned-bias
  baseline (euclidean), so the result heading's 15-of-16 baseline reversal is
  actually visualized (round-2 sidecar carried only anchor-status series).
- ``ladder_curves`` (clean-result round 3, ``--only ladder``) — RE-RENDER of
  the per-character 9-rung recovery-curve companion with the SAME
  plain-English rung map ``rungwise_ordering`` uses (the round-2 clean-result
  critique's blocker: the prior render carried raw ``1_direct``-style config
  slugs as x tick labels). Same data as the original
  ``issue2479_gradient_verdict.py`` render — one curve per character, colored
  by axis score, recovery fraction per rung.

Anchor characters keep the accent color used by the earlier rounds; the
identity-plus-learned-bias series keeps the round-2 baseline color
(one color = one meaning across the writeup).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis import paper_plots as pp  # noqa: E402

SERIES_NEW = "new character"
SERIES_ANCHOR = "anchor"
SERIES_TRANSFER = "Transferred operator"
SERIES_IDENTITY = "Identity + bias"

# Candidate label offsets (points), tried in order by the greedy dodger.
_CANDIDATES = [
    (5, 4, "left"),
    (-5, 4, "right"),
    (5, -10, "left"),
    (-5, -10, "right"),
    (5, 14, "left"),
    (-5, 14, "right"),
    (5, -20, "left"),
    (-5, -20, "right"),
]


def _overlaps(b, boxes) -> bool:
    return any(b.overlaps(o) for o in boxes)


def _greedy_labels(fig, ax, rows: list[dict]) -> None:
    """Collision-free point labels: renderer-measured greedy placement.

    For each point (descending y), the first candidate offset whose text
    bbox hits neither an already-placed label nor another point marker is
    kept. Deterministic for fixed data.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt_boxes = []
    for r in rows:
        px, py = ax.transData.transform((r["x"], r["y"]))
        pt_boxes.append(matplotlib.transforms.Bbox([[px - 5, py - 5], [px + 5, py + 5]]))
    placed = []
    for i in sorted(range(len(rows)), key=lambda k: -rows[k]["y"]):
        r = rows[i]
        others = pt_boxes[:i] + pt_boxes[i + 1 :]
        ann = None
        for dx, dy, ha in _CANDIDATES:
            if ann is not None:
                ann.remove()
            ann = ax.annotate(
                r["display_name"],
                (r["x"], r["y"]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                fontsize=7,
            )
            bbox = ann.get_window_extent(renderer=renderer)
            if not _overlaps(bbox, placed) and not _overlaps(bbox, others):
                break
        placed.append(ann.get_window_extent(renderer=renderer))


def _scatter_by_anchor(ax, rows: list[dict]) -> None:
    for is_anchor, role, marker, lab in (
        (False, "primary", "o", SERIES_NEW),
        (True, "accent", "D", SERIES_ANCHOR),
    ):
        sub = [r for r in rows if r["anchor"] == is_anchor]
        ax.scatter(
            [r["x"] for r in sub],
            [r["y"] for r in sub],
            c=pp.paper_palette_role(role),
            marker=marker,
            s=42,
            label=lab,
            zorder=3,
        )


RUNG_LABELS = {
    "1_direct": "Direct (no refit)",
    "2_ctx_offset": "Context offset",
    "3_ans_offset": "Answer offset",
    "4_bias_refit": "Bias refit (headline)",
    "5_global_scale": "Global scale",
    "6_rotation": "Rotation",
    "7_ctx_reparam": "Context reparam.",
    "8_ans_reparam": "Answer reparam.",
    "9_full_AMB": "Full affine remap",
}
NULL_Q95 = 0.4235  # headline 10,000-shuffle null 95th percentile (gradient_verdict.json)


def _ladder_curves_figure(verdict_path: Path, fig_dir: Path) -> None:
    """Re-render ladder_curves with plain-English rung tick labels.

    Same data as the original issue2479_gradient_verdict.py render: one
    recovery-fraction curve per eligible character across the 9 transfer
    rungs, colored by the frozen AI-likeness axis score. Asserts every rung
    key is in RUNG_LABELS so a raw config slug can never leak onto the
    canvas (or into the savefig_paper sidecar).
    """
    per_char = json.loads(verdict_path.read_text())["per_character"]
    recs = [
        r
        for r in per_char.values()
        if r.get("fraction_eligible") and r.get("rung_order") and r.get("rung_r2_all")
    ]
    if not recs:
        raise RuntimeError("no eligible characters with rung_r2_all + rung_order")
    rung_order = recs[0]["rung_order"]
    missing = [r for r in rung_order if r not in RUNG_LABELS]
    assert not missing, f"rung keys without a plain-English label: {missing}"
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = plt.cm.viridis
    scores = [r["axis_score"] for r in recs]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    for r in sorted(recs, key=lambda q: q["axis_score"]):
        ys = [
            (r["rung_r2_all"].get(rung) / r["ceiling_r2"])
            if (r["rung_r2_all"].get(rung) is not None and r["ceiling_r2"])
            else float("nan")
            for rung in rung_order
        ]
        ax.plot(
            range(len(rung_order)),
            ys,
            marker="o",
            ms=3,
            lw=1.0,
            color=cmap((r["axis_score"] - lo) / span),
            label=r["display_name"],
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lo, vmax=hi))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="AI-likeness axis score (judge, 0-100)")
    ax.set_xticks(range(len(rung_order)))
    ax.set_xticklabels([RUNG_LABELS[r] for r in rung_order], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Transfer-ladder rung")
    ax.set_ylabel("Recovery fraction (rung R² / ceiling R²)")
    pp.savefig_paper(fig, "ladder_curves", dir=fig_dir)
    plt.close(fig)
    print(f"wrote {fig_dir / 'ladder_curves'}.png (+ pdf, meta.json)")


def _rungwise_figure(fig_dir: Path) -> None:
    """Lollipop of rho(axis, rung recovery) per ladder rung (r3_diagnostics.json)."""
    diag = json.loads(Path("eval_results/issue_2479/r3_diagnostics.json").read_text())
    per_rung = diag["rungwise_axis_ordering"]["per_rung"]
    rungs = list(RUNG_LABELS)
    rhos = [per_rung[r]["rho"] for r in rungs]
    ys = list(range(len(rungs)))[::-1]  # rung 1 at top
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hlines(ys, 0, rhos, color="0.75", lw=1.0, zorder=2)
    ax.scatter(
        rhos,
        ys,
        c=pp.paper_palette_role("primary"),
        s=48,
        label="Rank correlation with the judged axis",
        zorder=3,
    )
    ax.axvline(NULL_Q95, color="0.4", lw=0.8, ls="--")
    ax.axvline(0.0, color="0.85", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([RUNG_LABELS[r] for r in rungs])
    ax.set_xlabel("Rank correlation of rung recovery with the AI-likeness axis")
    ax.set_ylabel("Transfer-ladder rung")
    pp.savefig_paper(fig, "rungwise_ordering", dir=fig_dir)
    plt.close(fig)
    print(f"wrote {fig_dir / 'rungwise_ordering'}.png (+ pdf, meta.json)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verdict", type=Path, default=Path("eval_results/issue_2479/gradient_verdict.json")
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_2479"))
    ap.add_argument("--only", choices=["all", "rungwise", "ladder"], default="all")
    args = ap.parse_args()

    pp.set_paper_style("blog")
    if args.only == "rungwise":
        _rungwise_figure(args.fig_dir)
        return
    if args.only == "ladder":
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        _ladder_curves_figure(args.verdict, args.fig_dir)
        return
    per_char = json.loads(args.verdict.read_text())["per_character"]
    names = sorted(per_char, key=lambda n: per_char[n]["axis_score"])
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    def rows_for(value_fn) -> list[dict]:
        return [
            {
                "display_name": per_char[n]["display_name"],
                "x": per_char[n]["axis_score"],
                "y": float(value_fn(per_char[n])),
                "anchor": per_char[n]["anchor"],
            }
            for n in names
        ]

    # 1) gradient_hero_v2 — recovery fraction vs axis, collision-free labels.
    hero_rows = rows_for(lambda r: r["recovery_fraction"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _scatter_by_anchor(ax, hero_rows)
    ax.set_xlabel("AI-likeness axis score (judge, 0-100)")
    ax.set_ylabel("Rung-4 recovery fraction")
    ax.legend(frameon=False, loc="lower right")
    _greedy_labels(fig, ax, hero_rows)
    pp.savefig_paper(fig, "gradient_hero_v2", dir=args.fig_dir)
    plt.close(fig)

    # 2) retrieval_identity_vs_transfer — two panels: recovery scatter +
    #    per-character transferred-vs-identity top-1 accuracy (euclidean).
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    acc_rows = rows_for(lambda r: r["acc1_rung4"] / r["acc1_ceiling"])
    _scatter_by_anchor(axl, acc_rows)
    axl.set_xlabel("AI-likeness axis score (judge, 0-100)")
    axl.set_ylabel("acc@1 recovery fraction (euclidean)")
    axl.set_title("Transferred-operator retrieval vs axis")
    axl.legend(frameon=False, loc="upper left")
    _greedy_labels(fig, axl, acc_rows)

    ys = range(len(names))
    tr = [per_char[n]["acc1_rung4"] for n in names]
    idb = [per_char[n]["acc1_identity_bias"] for n in names]
    chance = per_char[names[0]]["acc1_chance"]
    axr.hlines(
        list(ys),
        [min(a, b) for a, b in zip(tr, idb)],
        [max(a, b) for a, b in zip(tr, idb)],
        color="0.75",
        lw=1.0,
        zorder=2,
    )
    axr.scatter(
        tr, list(ys), c=pp.paper_palette_role("primary"), s=42, label=SERIES_TRANSFER, zorder=3
    )
    axr.scatter(
        idb, list(ys), c=pp.paper_palette_role("baseline"), s=42, label=SERIES_IDENTITY, zorder=3
    )
    axr.axvline(chance, color="0.4", lw=0.8, ls="--")
    axr.set_yticks(list(ys))
    axr.set_yticklabels([per_char[n]["display_name"] for n in names])
    axr.set_xlabel("Top-1 retrieval accuracy (euclidean; dashed = chance)")
    axr.set_ylabel("Character (judged AI-likeness, low to high)")
    axr.set_title("Identity baseline vs transferred operator")
    axr.legend(frameon=False, loc="lower right")
    pp.savefig_paper(fig, "retrieval_identity_vs_transfer", dir=args.fig_dir)
    plt.close(fig)

    for stem in ("gradient_hero_v2", "retrieval_identity_vs_transfer"):
        print(f"wrote {args.fig_dir / stem}.png (+ pdf, meta.json)")


if __name__ == "__main__":
    main()
