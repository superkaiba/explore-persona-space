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
    args = ap.parse_args(argv)

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
    picker_path = ANALYSIS_DIR / "picked_headline_frac.json"
    picked_frac_per_seed: dict = {}
    if picker_path.exists():
        picker_payload = json.loads(picker_path.read_text())
        picked_frac_per_seed = picker_payload.get("results", {})
        # Fallback for --picked-frac: use seed42's picker verdict if not given.
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
        picker_payload = None

    # If still None, fall back to the middle of the trained fracs (pre-v3
    # behaviour, for back-compat when the picker hasn't run yet).
    if args.picked_frac is None:
        args.picked_frac = headline["fracs"][len(headline["fracs"]) // 2]
        logger.warning(
            "Picker output absent or returned no eligible frac; using "
            "middle-of-fracs fallback (%s) for the scatter / diagonal panels.",
            args.picked_frac,
        )
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
    logger.info("Figures done -> %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
