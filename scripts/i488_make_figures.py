# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #488 — figure generation (over-produce; analyzer picks hero).

Plan v2 §6.3. Per planner memory `feedback_show_raw_alongside_processed`,
every residualized / partial scatter is paired with its raw counterpart.

Hero candidates (over-produce all three):

1. JS-vs-emission scatter colored by stylization_score at the picked frac.
2. Partial-ρ panel — bar chart of partial ρ by checkpoint frac × covariate set.
3. Per-checkpoint trajectory — line plot of emission rate by frac per source.

Exploratory dump:

* 27×27 emission heatmap per frac × seed.
* Per-class-pair partial-ρ grid.
* Per-source diagonal sanity bars.
* Per-cell variance scatter across seeds.
* Saturation-fraction trajectory per class.

Writes PNG + PDF + a meta.json sidecar per figure (commit hash, source data
paths, panel description) under ``figures/issue_488/``.

CLI:
    uv run python scripts/i488_make_figures.py
    uv run python scripts/i488_make_figures.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("i488.figures")

ANALYSIS_DIR = Path("eval_results/issue_488/analysis")
PREDICTORS_DIR = Path("eval_results/issue_488/predictors")
LADDER_JSONL = Path("logs/issue_488/ladder/ladder.jsonl")
FIG_DIR = Path("figures/issue_488")


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip().split("\n")[0]
        )
    except Exception:
        return "unknown"


def _write_meta(fig_path: Path, panels: list[dict], source_data: list[str]) -> None:
    meta_path = fig_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "issue": 488,
                "fig_path": str(fig_path),
                "panels": panels,
                "source_data": source_data,
                "git_commit": _git_commit(),
            },
            indent=2,
        )
    )


def _save_fig(fig, base: Path, panels: list[dict], source_data: list[str]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base) + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    _write_meta(Path(str(base) + ".png"), panels, source_data)


def _hero_scatter(cells_payload: dict, picked_frac: float, picked_seed: int) -> None:
    """Hero candidate 1 — JS-vs-emission scatter colored by stylization_score."""
    import matplotlib.pyplot as plt
    import numpy as np

    cells = [
        c
        for c in cells_payload["cells"]
        if c["frac"] == picked_frac and c["seed"] == picked_seed and not c["is_diagonal"]
    ]
    if not cells:
        logger.warning("No cells for hero scatter at frac=%s seed=%s", picked_frac, picked_seed)
        return
    xs = np.array([c["JS"] for c in cells])
    ys = np.array([c["emission_rate"] for c in cells])
    cs = np.array([c["stylization_score_source"] for c in cells])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    sc = ax[0].scatter(xs, ys, c=cs, cmap="viridis", s=22, alpha=0.75)
    ax[0].set_xlabel("Base-model JS divergence (source ↔ target)")
    ax[0].set_ylabel("On-policy emission rate")
    ax[0].set_title(
        f"JS vs emission, colored by source stylization (frac={picked_frac}, seed={picked_seed})"
    )
    plt.colorbar(sc, ax=ax[0], label="stylization_score (source)")
    # Raw counterpart: same scatter, uncolored.
    ax[1].scatter(xs, ys, s=22, alpha=0.65, color="steelblue")
    ax[1].set_xlabel("Base-model JS divergence")
    ax[1].set_ylabel("On-policy emission rate")
    ax[1].set_title("Raw (no stylization coloring)")
    fig.tight_layout()
    _save_fig(
        fig,
        FIG_DIR / f"hero_scatter_js_vs_emission_frac{int(picked_frac * 100):03d}_seed{picked_seed}",
        panels=[
            {"id": "stylization-colored", "what": "JS-vs-emission colored by stylization"},
            {"id": "raw", "what": "same scatter, no covariate"},
        ],
        source_data=[
            str(ANALYSIS_DIR / "cells.json"),
            str(PREDICTORS_DIR / "stylization_score.json"),
        ],
    )
    plt.close(fig)


def _partial_panel(headline: dict, picked_frac_per_seed: dict | None = None) -> None:
    """Hero candidate 2 — bar chart of partial ρ by frac × covariate set.

    v3 §6.3 standing rec: annotate the picked frac (the LOWEST eligible
    frac under v3 §6.2.D's ρ-blind picker) with a dashed grey vertical
    line + ``(headline)`` label. Per-seed lines are drawn at the bar
    cluster whose label embeds that seed's picked frac tag.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    per_cell = headline["per_frac_seed_h1_h2"]
    keys = sorted(per_cell.keys())
    if not keys:
        return
    labels = keys
    n = len(labels)
    h1_p = [per_cell[k].get("h1_partial", {}).get("point", float("nan")) for k in labels]
    h1_lo = [per_cell[k].get("h1_partial", {}).get("ci_low", float("nan")) for k in labels]
    h1_hi = [per_cell[k].get("h1_partial", {}).get("ci_high", float("nan")) for k in labels]
    h2b_p = [per_cell[k].get("h2_binary_partial", {}).get("point", float("nan")) for k in labels]
    h2g_p = [per_cell[k].get("h2_graded_partial", {}).get("point", float("nan")) for k in labels]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(n)
    w = 0.28
    ax.bar(x - w, h1_p, w, label="length-only", color="#3b6db8")
    ax.bar(x, h2b_p, w, label="+ is_stylized", color="#d9874e")
    ax.bar(x + w, h2g_p, w, label="+ stylization_score", color="#5fa15f")
    # H1 CIs (errorbars on the length-only bar).
    yerr_lo = [p - lo for p, lo in zip(h1_p, h1_lo, strict=True)]
    yerr_hi = [hi - p for p, hi in zip(h1_p, h1_hi, strict=True)]
    ax.errorbar(x - w, h1_p, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=3)
    ax.axhline(0, color="black", lw=0.5, ls="--")

    # v3 picked-frac annotation: vertical at the bar cluster matching each
    # seed's picked frac. The label format on x is "frac{NNN}_seed{S}".
    if picked_frac_per_seed:
        for seed_key, verdict in picked_frac_per_seed.items():
            picked = verdict.get("picked_frac")
            if picked is None:
                continue
            tag = f"frac{round(picked * 100):03d}"
            seed_num = seed_key.replace("seed", "")
            wanted_label = f"{tag}_seed{seed_num}"
            if wanted_label in labels:
                xpos = labels.index(wanted_label)
                ax.axvline(
                    xpos,
                    color="grey",
                    ls="--",
                    lw=0.9,
                    alpha=0.7,
                )
                ax.text(
                    xpos,
                    ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] != 0 else 0.05,
                    f"frac={picked} (headline, seed {seed_num})",
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=7,
                    color="dimgrey",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Partial Spearman ρ(JS, emission)")
    ax.set_title(
        "H1/H2 partial ρ by frac × seed (dyadic cluster-bootstrap CI on length-only); "
        "dashed grey = v3 §6.2.D picked headline frac"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_fig(
        fig,
        FIG_DIR / "partial_rho_panel",
        panels=[
            {"id": "h1-length-only", "what": "length-partial ρ + dyadic-bootstrap CI"},
            {"id": "h2-binary", "what": "+ is_stylized_source partialled"},
            {"id": "h2-graded", "what": "+ stylization_score partialled"},
            {
                "id": "picked-frac-annotation",
                "what": "v3 §6.2.D headline frac (dashed grey vertical)",
            },
        ],
        source_data=[
            str(ANALYSIS_DIR / "h1_partial.json"),
            str(ANALYSIS_DIR / "picked_headline_frac.json"),
        ],
    )
    plt.close(fig)


def _trajectory_panel(cells_payload: dict, picked_frac_per_seed: dict | None = None) -> None:
    """Hero candidate 3 — emission rate trajectory across fracs per source.

    v3 §6.3 standing rec: annotate the picked frac (v3 §6.2.D's ρ-blind
    headline frac) with a dashed grey vertical line across all source
    lines. Per-seed lines are drawn separately if seeds disagree.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fracs = sorted({c["frac"] for c in cells_payload["cells"]})
    if len(fracs) < 2:
        return
    sources = sorted({c["source"] for c in cells_payload["cells"]})
    fig, ax = plt.subplots(figsize=(10, 5))
    for src in sources:
        ys = []
        for frac in fracs:
            offdiag = [
                c
                for c in cells_payload["cells"]
                if c["source"] == src and c["frac"] == frac and not c["is_diagonal"]
            ]
            if not offdiag:
                ys.append(float("nan"))
            else:
                ys.append(float(np.mean([c["emission_rate"] for c in offdiag])))
        cls = next(
            (c["source_class"] for c in cells_payload["cells"] if c["source"] == src),
            "?",
        )
        cmap = {
            "A": "#d33",
            "B": "#3b6db8",
            "C": "#888",
            "D": "#a05",
            "E": "#5fa15f",
            "F": "#d9874e",
            "G": "#7d4ba0",
        }
        ax.plot(fracs, ys, marker="o", label=f"{src} ({cls})", color=cmap.get(cls, "#444"))

    # v3 picked-frac annotation: vertical line at each seed's picked frac.
    annotated: set[float] = set()
    if picked_frac_per_seed:
        ylim_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.05
        for seed_key, verdict in picked_frac_per_seed.items():
            picked = verdict.get("picked_frac")
            if picked is None or picked in annotated:
                continue
            seed_num = seed_key.replace("seed", "")
            ax.axvline(picked, color="grey", ls="--", lw=0.9, alpha=0.75)
            ax.text(
                picked,
                ylim_top * 0.95,
                f"headline (seed {seed_num}): frac={picked}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color="dimgrey",
            )
            annotated.add(picked)

    ax.set_xlabel("Training fraction (epochs)")
    ax.set_ylabel("Mean off-diagonal emission rate")
    ax.set_title(
        "Off-diagonal emission trajectory per source; dashed grey = v3 §6.2.D picked headline frac"
    )
    ax.legend(ncol=4, fontsize=7, loc="upper left")
    fig.tight_layout()
    _save_fig(
        fig,
        FIG_DIR / "trajectory_emission_per_source",
        panels=[
            {
                "id": "off-diag-trajectory",
                "what": "mean off-diagonal emission rate vs frac, per source",
            },
            {
                "id": "picked-frac-annotation",
                "what": "v3 §6.2.D headline frac (dashed grey vertical)",
            },
        ],
        source_data=[
            str(ANALYSIS_DIR / "cells.json"),
            str(ANALYSIS_DIR / "picked_headline_frac.json"),
        ],
    )
    plt.close(fig)


def _diagonal_bars(cells_payload: dict, picked_frac: float, picked_seed: int) -> None:
    """Per-source diagonal sanity (emission_ii bars per source)."""
    import matplotlib.pyplot as plt

    diag = [
        c
        for c in cells_payload["cells"]
        if c["is_diagonal"] and c["frac"] == picked_frac and c["seed"] == picked_seed
    ]
    diag = sorted(diag, key=lambda c: c["source"])
    if not diag:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [c["source"] for c in diag]
    vals = [c["emission_rate"] for c in diag]
    ax.bar(range(len(labels)), vals, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.axhline(0.5, color="red", ls="--", lw=0.6, label="0.5 floor for inclusion")
    ax.set_ylabel("Source diagonal emission_ii")
    ax.set_title(f"Per-source implant strength (frac={picked_frac}, seed={picked_seed})")
    ax.legend()
    fig.tight_layout()
    _save_fig(
        fig,
        FIG_DIR / f"diagonal_emission_frac{int(picked_frac * 100):03d}_seed{picked_seed}",
        panels=[{"id": "diagonals", "what": "emission_ii per source"}],
        source_data=[str(ANALYSIS_DIR / "cells.json")],
    )
    plt.close(fig)


def _ladder_panel(ladder_path: Path = LADDER_JSONL) -> bool:
    """Hero #4 (plan v6 §6.3): ladder trajectory.

    For each rung L1..L5 actually run, render a grouped bar:
      - A1 self-emit
      - G2 self-emit
      - median bystander emit (over all 6 panel cells)
      - max bystander emit on the NON-STYLIZED subset (per v6 Must-Fix #2)
      - max bystander emit on the FULL panel (descriptive, includes A3)

    The picked rung (verdict ∈ {PASS, PICK_AT_SATURATION}) is annotated;
    the UNIFORM_LEAKAGE / EXHAUSTED rung is annotated red.

    Returns True iff the figure was rendered (i.e. ladder.jsonl existed and
    had ≥ 1 row).
    """
    if not ladder_path.exists():
        logger.info("No ladder.jsonl at %s — skipping Hero #4 ladder panel.", ladder_path)
        return False
    rows: list[dict] = []
    for line in ladder_path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        logger.info("ladder.jsonl empty — skipping Hero #4 ladder panel.")
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rungs = [r["rung"] for r in rows]
    a1_self = [r.get("a1_self_emit") or 0.0 for r in rows]
    g2_self = [r.get("g2_self_emit") or 0.0 for r in rows]
    median_by = [r.get("median_bystander_emit") or 0.0 for r in rows]
    max_by_ns = [r.get("max_bystander_emit_non_stylized") or 0.0 for r in rows]
    max_by_full = [r.get("max_bystander_emit_full_panel") or 0.0 for r in rows]
    verdicts = [r.get("verdict") for r in rows]

    x = np.arange(len(rungs))
    w = 0.16

    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(rungs)), 4.2))
    ax.bar(x - 2 * w, a1_self, w, label="A1 self-emit", color="#1f77b4")
    ax.bar(x - w, g2_self, w, label="G2 self-emit", color="#2ca02c")
    ax.bar(x, median_by, w, label="median bystander (n=6)", color="#9467bd")
    ax.bar(
        x + w,
        max_by_ns,
        w,
        label="max bystander (non-stylized subset, n=5)",
        color="#ff7f0e",
    )
    ax.bar(
        x + 2 * w,
        max_by_full,
        w,
        label="max bystander (full panel, incl. A3)",
        color="#d62728",
        alpha=0.55,
    )

    # Gate-band reference lines.
    ax.axhline(0.20, color="gray", ls=":", lw=0.7)
    ax.axhline(0.85, color="gray", ls=":", lw=0.7)
    ax.text(
        len(rungs) - 0.5,
        0.22,
        "Gate ANCHOR floor (0.20)",
        fontsize=7,
        color="gray",
        ha="right",
    )
    ax.text(
        len(rungs) - 0.5,
        0.87,
        "Gate ANCHOR saturation (0.85)",
        fontsize=7,
        color="gray",
        ha="right",
    )

    # Verdict annotations on the rung x-tick.
    rung_labels: list[str] = []
    picked_idx: int | None = None
    for i, (r, v) in enumerate(zip(rungs, verdicts, strict=True)):
        if v == "PASS":
            rung_labels.append(f"{r}\nPASS ★")
            picked_idx = i
        elif v == "PICK_AT_SATURATION":
            rung_labels.append(f"{r}\nPICK_AT_SAT ★")
            picked_idx = i
        elif v == "UNIFORM_LEAKAGE":
            rung_labels.append(f"{r}\nUNIFORM_LEAK ✗")
        elif v == "CLIMB":
            rung_labels.append(f"{r}\nCLIMB ↑")
        else:
            rung_labels.append(f"{r}\n{v}")
    if picked_idx is not None:
        ax.axvspan(picked_idx - 0.45, picked_idx + 0.45, color="#fff5b1", alpha=0.5, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(rung_labels, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("on-policy marker emit rate")
    ax.set_title(
        "Hero #4 — recipe ladder trajectory (plan v6 §6.3)\n"
        "PICK = lightest rung where A1 self-emit ∈ [0.20, 0.85] (or > 0.85) AND "
        "bystander resolves on non-stylized subset"
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()

    base = FIG_DIR / "hero4_ladder_trajectory"
    _save_fig(
        fig,
        base,
        panels=[
            {
                "panel_id": "hero4_ladder_trajectory",
                "description": (
                    "Per ladder rung L1..L5: A1 self-emit, G2 self-emit, median "
                    "bystander emit (n=6), max bystander emit on non-stylized "
                    "subset (n=5; A3 excluded per v6 Must-Fix #2), max bystander "
                    "emit on full panel (descriptive, includes A3). Reference "
                    "lines at the Gate ANCHOR band [0.20, 0.85]. Picked rung "
                    "shaded yellow."
                ),
                "n_rungs": len(rows),
                "picked_rung": rungs[picked_idx] if picked_idx is not None else None,
                "verdicts": verdicts,
            }
        ],
        source_data=[str(ladder_path)],
    )
    plt.close(fig)
    logger.info("Wrote Hero #4 ladder trajectory → %s.{png,pdf}", base)
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Wiring check; reads inputs but doesn't render figures.",
    )
    ap.add_argument("--picked-frac", type=float, default=None)
    ap.add_argument("--picked-seed", type=int, default=42)
    ap.add_argument(
        "--ladder-only",
        action="store_true",
        help=(
            "Render ONLY the Hero #4 ladder-trajectory panel from "
            "logs/issue_488/ladder/ladder.jsonl. Useful between Phase 2 and "
            "Phase 3 (when no cells.json / headline.json exist yet)."
        ),
    )
    args = ap.parse_args(argv)

    # v6: Hero #4 (ladder panel) is renderable independent of Phase 5 outputs.
    if args.ladder_only:
        rendered = _ladder_panel()
        return 0 if rendered else 2

    cells_path = ANALYSIS_DIR / "cells.json"
    headline_path = ANALYSIS_DIR / "headline.json"
    if not cells_path.exists() or not headline_path.exists():
        if args.dry_run:
            logger.info(
                "DRY RUN: wiring-only (no analysis outputs yet); argparse + module imports OK."
            )
            return 0
        logger.error("Run Phase 5 first: missing %s or %s", cells_path, headline_path)
        return 2

    cells_payload = json.loads(cells_path.read_text())
    headline = json.loads(headline_path.read_text())

    # v3 §6.3: read the ρ-blind post-hoc picker output to drive the
    # picked-frac figure annotations (Hero #2 + Hero #3). Each seed gets
    # its own picked frac under v3 §6.2.D.
    #
    # v3 §6.1 contract: if `picked_headline_frac.json` exists AND the
    # picker reports no eligible frac for the requested --picked-seed, the
    # figures script MUST REFUSE to render and exit non-zero. The pre-v3
    # "middle-of-fracs" silent fallback would publish a headline panel
    # from an arbitrary frac the picker explicitly rejected — that is the
    # exact production_no_inband_frac failure mode the recovery path
    # exists to prevent.
    picker_path = ANALYSIS_DIR / "picked_headline_frac.json"
    picked_frac_per_seed: dict = {}
    if picker_path.exists():
        picker_payload = json.loads(picker_path.read_text())
        picked_frac_per_seed = picker_payload.get("results", {})
        # Auto-set --picked-frac from the picker's per-seed verdict.
        if args.picked_frac is None:
            seed_key = f"seed{args.picked_seed}"
            v = picked_frac_per_seed.get(seed_key) or {}
            if v.get("picked_frac") is not None:
                args.picked_frac = v["picked_frac"]
                logger.info(
                    "--picked-frac auto-set to v3 §6.2.D pick for seed=%d: frac=%s",
                    args.picked_seed,
                    args.picked_frac,
                )
            else:
                # Picker ran and explicitly rejected every frac for this
                # seed. REFUSE to render — silent fallback to an arbitrary
                # frac would defeat v3 §6.1.
                logger.error(
                    "Picker rejected every frac for seed=%d (eligibility=%s). "
                    "v3 §6.1 contract: figures REFUSE to render in this case. "
                    "Re-grid the production frac set or revise the in-band "
                    "criteria before re-running Phase 5 + Phase 6.",
                    args.picked_seed,
                    v.get("eligibility", []),
                )
                return 4
    else:
        picker_payload = None
        if args.picked_frac is None:
            # Picker output absent AND no explicit --picked-frac given. Per
            # v3 §6.1 + §6.2.D this would only happen if Phase 5 was never
            # run — which is itself a pipeline error, not a "use middle of
            # fracs" situation. Fail loud rather than guess.
            logger.error(
                "Picker output (%s) missing and --picked-frac not supplied. "
                "Run Phase 5 first, or pass --picked-frac explicitly. The "
                "pre-v3 middle-of-fracs fallback has been removed (v3 §6.1).",
                picker_path,
            )
            return 4
    logger.info("Picked frac=%s seed=%d", args.picked_frac, args.picked_seed)

    if args.dry_run:
        logger.info(
            "DRY RUN: %d cell records loaded; would render hero + panel + trajectory + diagonal.",
            cells_payload["n_cells"],
        )
        return 0

    _hero_scatter(cells_payload, args.picked_frac, args.picked_seed)
    _partial_panel(headline, picked_frac_per_seed=picked_frac_per_seed)
    _trajectory_panel(cells_payload, picked_frac_per_seed=picked_frac_per_seed)
    _diagonal_bars(cells_payload, args.picked_frac, args.picked_seed)
    # v6 Hero #4 — ladder trajectory. Renders iff ladder.jsonl exists; the
    # main pipeline calls Phase 5 + this script in a single chain after the
    # ladder, so by here ladder.jsonl is present.
    _ladder_panel()
    logger.info("Figures done -> %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
