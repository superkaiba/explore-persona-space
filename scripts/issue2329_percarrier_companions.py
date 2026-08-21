"""Per-unit (per-carrier) companion figures for the q35_ladder_decay round.

Produces the low-level labeled-point views the v2 report requires behind four
committed aggregate figures, as NEW stems (never touching the committed ones):

- ``q35_ladder_decay_decay_raw_percarrier``  — behind ``q35_ladder_decay_decay_raw``
- ``q35_ladder_decay_decay_norm_percarrier`` — behind ``q35_ladder_decay_decay_norm``
- ``q35_ladder_decay_contrast_percarrier``   — behind ``q35_ladder_decay_contrast``
- ``q35_ladder_decay_transfer_percarrier``   — behind ``q35_ladder_decay_transfer``

Scope matches each committed aggregate: decay views plot the PRIMARY stratum
(install x context-end, rungs r1_pirate/r2_butler/r3_warm) per estimand
(``all`` = length-eligible rows; ``coh`` = coherence-conditional headline);
the transfer view plots per-carrier steered F_target for the 16 (direction x
slot) cells whose lattice verdicts are testable in BOTH runs. Colors reuse the
committed aggregates' encodings verbatim (one color = one meaning within each
figure family): arms steered/ceiling/floor = C0/C1/C2, estimands coh/all =
C0/C7 (contrast family), normalized F = C3, transfer flip cells = #d62728.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("issue2329_percarrier_companions")

MODEL_KEYS = ("q25", "q35")
ESTIMANDS = ("all", "coh")
ARM_KEYS = ("steered", "ceiling", "floor")
SEGS = (1, 2, 3, 4)

ARM_COLOR = {"steered": "C0", "ceiling": "C1", "floor": "C2"}
EST_COLOR = {"coh": "C0", "all": "C7"}  # committed contrast encoding
EST_STYLE = {"coh": "-", "all": "--"}
EST_ALPHA = {"coh": 1.0, "all": 0.45}
FLIP_COLOR = "#d62728"  # committed transfer encoding
BASE_COLOR = "#1f77b4"


def _short_dir(direction: str) -> str:
    """``install_r1_pirate`` -> ``in_r1``; ``erase_r5b_lu_philosophy`` -> ``er_r5b``."""
    kind, rest = direction.split("_", 1)
    return f"{'in' if kind == 'install' else 'er'}_{rest.split('_')[0]}"


def _primary_ce_units(stats: dict, model: str, estimand: str) -> list[tuple[str, str, dict]]:
    """(direction, carrier, per-carrier record) for the primary install-ce stratum."""
    units: list[tuple[str, str, dict]] = []
    for key, rec in stats["per_direction"].items():
        mk, direction, slot, est = key.split("|")
        if mk != model or slot != "ce" or est != estimand:
            continue
        rung = direction[len("install_") :]
        if stats["scope"][mk]["strata"].get(rung) != "primary":
            continue
        for carrier, crec in rec["per_carrier"].items():
            units.append((direction, carrier, crec))
    return units


def fig_decay_raw_percarrier(stats: dict, figures_dir: Path) -> None:
    """Per-carrier raw fragment scores per segment, behind the decay_raw aggregate."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharey=True, sharex=True)
    n_lines = 0
    for row, model in enumerate(MODEL_KEYS):
        for col, arm in enumerate(ARM_KEYS):
            ax = axes[row][col]
            for est in ESTIMANDS:
                for direction, carrier, crec in _primary_ce_units(stats, model, est):
                    means = crec.get(f"mean_{arm}") or {}
                    xs = [s for s in SEGS if means.get(str(s)) is not None]
                    ys = [means[str(s)] for s in xs]
                    if not xs:
                        continue
                    ax.plot(
                        xs,
                        ys,
                        marker="o",
                        markersize=2.2,
                        linewidth=0.7,
                        color=ARM_COLOR[arm],
                        linestyle=EST_STYLE[est],
                        alpha=EST_ALPHA[est] * 0.75,
                    )
                    n_lines += 1
                    if est == "coh":  # one label per unit, at the line end
                        ax.annotate(
                            f"{_short_dir(direction)}·{carrier}",
                            (xs[-1], ys[-1]),
                            fontsize=4.5,
                            xytext=(2, 0),
                            textcoords="offset points",
                        )
            ax.set_title(f"{model} — {arm}", fontsize=9)
            ax.set_xticks(list(SEGS))
            if row == 1:
                ax.set_xlabel("token quartile")
            if col == 0:
                ax.set_ylabel("per-carrier fragment persona score (0-1)")
    handles = [
        plt.Line2D([], [], color="k", linestyle=EST_STYLE[e], alpha=EST_ALPHA[e], label=e)
        for e in ESTIMANDS
    ]
    axes[0][0].legend(handles=handles, fontsize=6, title="estimand", title_fontsize=6)
    fig.suptitle("Within-answer decay — per-carrier companion (primary install x context-end)")
    fig.tight_layout()
    savefig_paper(fig, "q35_ladder_decay_decay_raw_percarrier", dir=figures_dir)
    plt.close(fig)
    logger.info("[decay_raw_percarrier] %d unit polylines", n_lines)


def fig_decay_norm_percarrier(stats: dict, figures_dir: Path) -> None:
    """Per-carrier anchor-normalized F per segment, behind the decay_norm aggregate."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    n_lines = 0
    for ax, model in zip(axes, MODEL_KEYS, strict=True):
        for est in ESTIMANDS:
            for direction, carrier, crec in _primary_ce_units(stats, model, est):
                fdict = crec.get("F") or {}
                xs = [s for s in SEGS if fdict.get(str(s)) is not None]
                ys = [fdict[str(s)] for s in xs]
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    marker="s",
                    markersize=2.2,
                    linewidth=0.7,
                    color="C3",
                    linestyle=EST_STYLE[est],
                    alpha=EST_ALPHA[est] * 0.7,
                )
                n_lines += 1
                if est == "coh":
                    ax.annotate(
                        f"{_short_dir(direction)}·{carrier}",
                        (xs[-1], ys[-1]),
                        fontsize=4.5,
                        xytext=(2, 0),
                        textcoords="offset points",
                    )
        ax.axhline(0.0, color="grey", lw=0.6)
        ax.axhline(1.0, color="grey", lw=0.6)
        ax.set_xticks(list(SEGS))
        ax.set_xlabel("token quartile")
        ax.set_title(model, fontsize=10)
    axes[0].set_ylabel("per-carrier F = (steered − floor) / (ceiling − floor)")
    handles = [
        plt.Line2D([], [], color="C3", linestyle=EST_STYLE[e], alpha=EST_ALPHA[e], label=e)
        for e in ESTIMANDS
    ]
    axes[0].legend(handles=handles, fontsize=6, title="estimand", title_fontsize=6)
    fig.suptitle("Anchor-normalized decay — per-carrier companion (per-carrier denominators)")
    fig.tight_layout()
    savefig_paper(fig, "q35_ladder_decay_decay_norm_percarrier", dir=figures_dir)
    plt.close(fig)
    logger.info("[decay_norm_percarrier] %d unit polylines", n_lines)


def fig_contrast_percarrier(stats: dict, figures_dir: Path) -> None:
    """Labeled per-carrier paired drops behind the decay-contrast aggregate."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    specs = (
        (axes[0], "delta_d", "per-carrier delta-D (steered − ceiling raw drop, 0-1)"),
        (axes[1], "delta_d_f", "per-carrier delta-D_F (change in patched arm's F)"),
    )
    counts: dict[str, int] = {}
    for ax, stat_key, ylab in specs:
        x = 0.0
        ticks: list[float] = []
        tlabels: list[str] = []
        for model in MODEL_KEYS:
            for est in ESTIMANDS:
                units = [
                    (d, c, r)
                    for d, c, r in _primary_ce_units(stats, model, est)
                    if r.get("supported") and r.get(stat_key) is not None
                ]
                units.sort(key=lambda u: (u[0], u[1]))
                for i, (direction, carrier, crec) in enumerate(units):
                    xo = x + (i - (len(units) - 1) / 2.0) * (0.7 / max(1, len(units)))
                    ax.scatter([xo], [crec[stat_key]], s=12, color=EST_COLOR[est], alpha=0.8)
                    ax.annotate(
                        f"{_short_dir(direction)}·{carrier}",
                        (xo, crec[stat_key]),
                        fontsize=4.2,
                        xytext=(1.5, 1),
                        textcoords="offset points",
                    )
                counts[f"{model}|{est}|{stat_key}"] = len(units)
                ticks.append(x)
                tlabels.append(f"{model}\n{est}")
                x += 1.0
            x += 0.5
        ax.axhline(0.0, color="grey", lw=0.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels, fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
    fig.suptitle(
        "Patched-vs-prompted decay contrast — per-carrier companion (common-support carriers)"
    )
    fig.tight_layout()
    savefig_paper(fig, "q35_ladder_decay_contrast_percarrier", dir=figures_dir)
    plt.close(fig)
    logger.info("[contrast_percarrier] per-group ns: %s", counts)


def _both_testable_cells(fork_stats: dict, parent_stats: dict) -> set[str]:
    """The transfer figure's cell set: lattice-testable in BOTH runs with steered means."""
    cells: set[str] = set()
    for key in fork_stats["lattice"]:
        pv = parent_stats["lattice"].get(key)
        fv = fork_stats["lattice"][key]
        if pv is None or pv["verdict"] == "untestable" or fv["verdict"] == "untestable":
            continue
        xs = parent_stats["estimation"].get(f"{key}|steered")
        ys = fork_stats["estimation"].get(f"{key}|steered")
        if xs and ys and xs["mean_f_target"] is not None and ys["mean_f_target"] is not None:
            cells.add(key)
    return cells


def fig_transfer_percarrier(
    fork_stats: dict,
    parent_stats: dict,
    fork_cells: list[dict],
    parent_cells: list[dict],
    figures_dir: Path,
) -> None:
    """Per-carrier paired steered F_target scatter behind the transfer aggregate."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper

    cells = _both_testable_cells(fork_stats, parent_stats)
    flips = {
        key
        for key in cells
        if parent_stats["lattice"][key]["verdict"] != fork_stats["lattice"][key]["verdict"]
    }
    parent_by = {
        (r["direction"], r["slot"], r["carrier"]): r["f_target"]
        for r in parent_cells
        if r["arm"] == "steered" and r["f_target"] is not None
    }
    fig, ax = plt.subplots(figsize=(8, 7.5))
    n_pts = 0
    for r in fork_cells:
        if r["arm"] != "steered" or r["f_target"] is None:
            continue
        key = f"{r['direction']}|{r['slot']}"
        if key not in cells:
            continue
        x = parent_by.get((r["direction"], r["slot"], r["carrier"]))
        if x is None:
            continue
        flip = key in flips
        ax.scatter([x], [r["f_target"]], s=16, color=FLIP_COLOR if flip else BASE_COLOR, alpha=0.8)
        ax.annotate(
            f"{_short_dir(r['direction'])}|{r['slot']}·{r['carrier']}",
            (x, r["f_target"]),
            fontsize=4.2,
            xytext=(2, 1),
            textcoords="offset points",
        )
        n_pts += 1
    ax.axline((0.0, 0.0), slope=1.0, color="grey", lw=0.8, ls=":")
    handles = [
        plt.Line2D([], [], marker="o", ls="none", color=BASE_COLOR, label="verdict unchanged"),
        plt.Line2D([], [], marker="o", ls="none", color=FLIP_COLOR, label="verdict flip cell"),
    ]
    ax.legend(handles=handles, fontsize=7)
    ax.set_xlabel("parent #2162 (Qwen2.5-7B-Instruct) per-carrier steered F_target")
    ax.set_ylabel("this run (Qwen3.5-9B) per-carrier steered F_target")
    ax.set_title(
        f"Ladder transfer — per-carrier companion ({n_pts} carrier pairs, "
        f"{len(cells)} both-testable cells)"
    )
    fig.tight_layout()
    savefig_paper(fig, "q35_ladder_decay_transfer_percarrier", dir=figures_dir)
    plt.close(fig)
    logger.info("[transfer_percarrier] %d carrier pairs over %d cells", n_pts, len(cells))


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file via text-mode iteration (never ``splitlines``)."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    """Render the four per-carrier companion figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--round-dir", type=Path, default=Path("eval_results/issue_2329/q35_ladder_decay")
    )
    ap.add_argument(
        "--parent-ladder-dir",
        type=Path,
        default=Path("eval_results/issue_2162/persona_specificity_ladder"),
        help="Parent #2162 ladder artifacts (committed on main; not in the sparse worktree).",
    )
    ap.add_argument("--figures-dir", type=Path, default=Path("figures/issue_2329/q35_ladder_decay"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import matplotlib

    matplotlib.use("Agg")
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    decay_stats = json.loads((args.round_dir / "decay" / "decay_stats.json").read_text("utf-8"))
    fork_stats = json.loads((args.round_dir / "f_metrics" / "stats.json").read_text("utf-8"))
    parent_stats = json.loads((args.parent_ladder_dir / "stats.json").read_text("utf-8"))
    fork_cells = _read_jsonl(args.round_dir / "f_metrics" / "f_cells.jsonl")
    parent_cells = _read_jsonl(args.parent_ladder_dir / "f_cells.jsonl")
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    fig_decay_raw_percarrier(decay_stats, args.figures_dir)
    fig_decay_norm_percarrier(decay_stats, args.figures_dir)
    fig_contrast_percarrier(decay_stats, args.figures_dir)
    fig_transfer_percarrier(fork_stats, parent_stats, fork_cells, parent_cells, args.figures_dir)
    logger.info("[done] 4 companion figures -> %s", args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
