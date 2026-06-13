#!/usr/bin/env python3
# em-dash + marker " ※" + Greek delta intentional
"""Task #613 sep-ablation — figures (CPU, off-pod).

Single hero figure: the 2x2 interaction between the loss-placement flag
(off = trailing-newline dead-slot, on = post-response live-slot) and the
positive-row separator (with-sep = ``"\\n\\n"``, no-sep = ``""``). Reads
terminal on-policy source ΔG and dense-read bystander/trained-negative
ΔG from the 6 cells under ``eval_results/issue_613/sep-ablation/`` (4
new cells), ``eval_results/issue_613/flagon_ab/`` (parent with-sep
flag-on), and ``eval_results/issue_601/phase2/`` (parent with-sep
flag-off).

Usage:
    uv run python scripts/i613_sep_figures.py \
        [--sep-root eval_results/issue_613/sep-ablation] \
        [--withsep-flagon-root eval_results/issue_613/flagon_ab] \
        [--withsep-flagoff-root eval_results/issue_601/phase2] \
        [--figures-dir figures/issue_613]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from statistics import mean

log = logging.getLogger("i613.sep_figures")

TRAINED_NEG = ["qwen_default", "hero", "journalist", "ai_assistant"]
BYST = [
    "con_artist",
    "wizard",
    "investment_banker",
    "accountant",
    "florist",
    "postal_worker",
    "french_person",
    "programmer",
]
SEEDS = [42, 137]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _terminal_source_dg(traj_path: Path) -> float:
    j = _load(traj_path)
    ck = j["checkpoints"][-1]
    return ck["source_self"]["delta_g_mean"]


def _terminal_source_dlogit(traj_path: Path) -> float:
    """Δ(z_marker − z_eos) trained − base at terminal."""
    j = _load(traj_path)
    ck = j["checkpoints"][-1]
    s = ck["source_self"]
    g = s["z_marker_g_mean"] - s["z_eos_g_mean"]
    b = s["z_marker_b_mean"] - s["z_eos_b_mean"]
    return g - b


def _terminal_byst_dg(traj_path: Path) -> float:
    """Mean over 8 bystander personas of mean over 10 questions."""
    j = _load(traj_path)
    ck = j["checkpoints"][-1]
    held = ck["held_out"]
    persona_means = []
    for _p, qd in held.items():
        persona_means.append(mean(q["delta_g"] for q in qd.values()))
    return mean(persona_means)


def _dense_terminal_trained_neg(dense_path: Path) -> float:
    j = _load(dense_path)
    ck = j["checkpoints"][-1]
    reads = ck["reads"]
    persona_means = []
    for p in TRAINED_NEG:
        persona_means.append(mean(q["delta_g"] for q in reads[p].values()))
    return mean(persona_means)


def _dense_terminal_byst(dense_path: Path) -> float:
    j = _load(dense_path)
    ck = j["checkpoints"][-1]
    reads = ck["reads"]
    persona_means = []
    for p in BYST:
        persona_means.append(mean(q["delta_g"] for q in reads[p].values()))
    return mean(persona_means)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description="Task #613 sep-ablation figures.")
    ap.add_argument("--sep-root", type=Path, default=Path("eval_results/issue_613/sep-ablation"))
    ap.add_argument(
        "--withsep-flagon-root", type=Path, default=Path("eval_results/issue_613/flagon_ab")
    )
    ap.add_argument(
        "--withsep-flagoff-root", type=Path, default=Path("eval_results/issue_601/phase2")
    )
    ap.add_argument("--figures-dir", type=Path, default=Path("figures/issue_613"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=i613_sep_figures] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all 12 numbers (4x3) ────────────────────────────────────────────
    cells = {
        "withsep_flagoff": {},
        "withsep_flagon": {},
        "nosep_flagoff": {},
        "nosep_flagon": {},
    }
    for seed in SEEDS:
        # No-sep cells live under sep-ablation/.
        for arm in ("flagon", "flagoff"):
            key = f"nosep_{arm}"
            cell_dir = args.sep_root / f"sepablation_{arm}_200p800n_seed{seed}"
            cells[key][seed] = {
                "src_dg": _terminal_source_dg(cell_dir / "trajectory.json"),
                "src_dlogit": _terminal_source_dlogit(cell_dir / "trajectory.json"),
                "byst_dg": _terminal_byst_dg(cell_dir / "trajectory.json"),
                "trnneg_dg": _dense_terminal_trained_neg(cell_dir / "dense_trajectory.json"),
                "dense_byst_dg": _dense_terminal_byst(cell_dir / "dense_trajectory.json"),
            }
        # With-sep flag-on (parent run, this issue)
        cell_dir = args.withsep_flagon_root / f"flagon_200p800n_seed{seed}"
        cells["withsep_flagon"][seed] = {
            "src_dg": _terminal_source_dg(cell_dir / "trajectory.json"),
            "src_dlogit": _terminal_source_dlogit(cell_dir / "trajectory.json"),
            "byst_dg": _terminal_byst_dg(cell_dir / "trajectory.json"),
            "trnneg_dg": _dense_terminal_trained_neg(cell_dir / "dense_trajectory.json"),
            "dense_byst_dg": _dense_terminal_byst(cell_dir / "dense_trajectory.json"),
        }
        # With-sep flag-off (parent #601 comparator)
        cell_dir = args.withsep_flagoff_root / f"dense_200p800n_seed{seed}"
        cells["withsep_flagoff"][seed] = {
            "src_dg": _terminal_source_dg(cell_dir / "trajectory.json"),
            "src_dlogit": _terminal_source_dlogit(cell_dir / "trajectory.json"),
            "byst_dg": _terminal_byst_dg(cell_dir / "trajectory.json"),
            "trnneg_dg": _dense_terminal_trained_neg(cell_dir / "dense_trajectory.json"),
            "dense_byst_dg": _dense_terminal_byst(cell_dir / "dense_trajectory.json"),
        }

    log.info("Cells loaded.  Seed-means:")
    for key, vs in cells.items():
        sm_src = mean(vs[s]["src_dg"] for s in SEEDS)
        sm_byst = mean(vs[s]["byst_dg"] for s in SEEDS)
        sm_trn = mean(vs[s]["trnneg_dg"] for s in SEEDS)
        log.info(
            "  %s  src=%.2f byst=%.2f trn=%.2f",
            key,
            sm_src,
            sm_byst,
            sm_trn,
        )

    # ── HERO sep-ablation: 2x2 interaction ───────────────────────────────────
    # x = separator (with-sep | no-sep), color/marker = flag (off | on),
    # y = source ΔG (left panel) and bystander ΔG (right panel).
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)

    sep_x = {"withsep": 0, "nosep": 1}
    flag_offset = {"flagoff": -0.08, "flagon": 0.08}
    flag_color = {
        "flagoff": paper_palette_role("baseline"),
        "flagon": paper_palette_role("primary"),
    }
    seed_alpha = {42: 1.0, 137: 0.45}

    def _plot_channel(ax, channel_key: str, ylabel: str, title: str):
        for sep, xb in sep_x.items():
            for flag in ("flagoff", "flagon"):
                ys = []
                for seed in SEEDS:
                    y = cells[f"{sep}_{flag}"][seed][channel_key]
                    ys.append(y)
                    ax.scatter(
                        xb + flag_offset[flag],
                        y,
                        marker="o" if flag == "flagon" else "s",
                        s=60,
                        facecolors=flag_color[flag],
                        edgecolors=flag_color[flag],
                        linewidths=1.4,
                        alpha=seed_alpha[seed],
                        zorder=3,
                    )
                # seed-mean bar
                ax.hlines(
                    mean(ys),
                    xb + flag_offset[flag] - 0.045,
                    xb + flag_offset[flag] + 0.045,
                    color=flag_color[flag],
                    linewidth=2.0,
                    zorder=2,
                )
            # connect within-sep flag-off → flag-on (seed-means)
            off_mean = mean(cells[f"{sep}_flagoff"][s][channel_key] for s in SEEDS)
            on_mean = mean(cells[f"{sep}_flagon"][s][channel_key] for s in SEEDS)
            ax.plot(
                [xb + flag_offset["flagoff"], xb + flag_offset["flagon"]],
                [off_mean, on_mean],
                "-",
                color=paper_palette_role("neutral"),
                linewidth=1.0,
                alpha=0.5,
                zorder=1,
            )
            # annotate within-sep delta
            ax.annotate(
                f"Δ = {on_mean - off_mean:+.2f}",
                xy=(xb, (off_mean + on_mean) / 2),
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            )
        ax.set_xticks(list(sep_x.values()))
        ax.set_xticklabels(
            [
                "With separator (\\n\\n)\nbetween answer and marker",
                "No separator\n(loss slot = marker slot)",
            ],
            fontsize=9,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)

    _plot_channel(
        ax_l,
        "src_dg",
        "Source marker log-prob gain ΔG (nats)",
        "Source implant strength",
    )
    _plot_channel(
        ax_r,
        "byst_dg",
        "Bystander mean marker log-prob gain ΔG (nats)",
        "Bystander leakage (n = 8 personas)",
    )

    # Legend
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=flag_color["flagoff"],
            markeredgecolor=flag_color["flagoff"],
            markersize=8,
            label="Negatives are gradient-dead (loss at trailing newline; flag off)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=flag_color["flagon"],
            markeredgecolor=flag_color["flagon"],
            markersize=8,
            label="Negatives are gradient-live (loss at post-response stop token; flag on)",
        ),
        Line2D([0], [0], color="black", alpha=1.0, marker="o", linestyle="", label="seed 42"),
        Line2D([0], [0], color="black", alpha=0.45, marker="o", linestyle="", label="seed 137"),
    ]
    ax_l.legend(handles=legend_items, fontsize=7, loc="lower left", frameon=False)
    fig.tight_layout()
    savefig_paper(fig, args.figures_dir / "sep_ablation_interaction")
    plt.close(fig)

    # ── Secondary: log-prob vs logit margin co-read (the SAME spaces from #432 rule) ─
    fig, (ax_lp, ax_lg) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for sep, xb in sep_x.items():
        for flag in ("flagoff", "flagon"):
            for ch_key, ax in (("src_dg", ax_lp), ("src_dlogit", ax_lg)):
                ys = []
                for seed in SEEDS:
                    y = cells[f"{sep}_{flag}"][seed][ch_key]
                    ys.append(y)
                    ax.scatter(
                        xb + flag_offset[flag],
                        y,
                        marker="o" if flag == "flagon" else "s",
                        s=60,
                        facecolors=flag_color[flag],
                        edgecolors=flag_color[flag],
                        linewidths=1.4,
                        alpha=seed_alpha[seed],
                        zorder=3,
                    )
                ax.hlines(
                    mean(ys),
                    xb + flag_offset[flag] - 0.045,
                    xb + flag_offset[flag] + 0.045,
                    color=flag_color[flag],
                    linewidth=2.0,
                    zorder=2,
                )
    for ax, ylab, title in (
        (ax_lp, "Δ log P(marker) — nats", "Log-prob (PRIMARY, behavioral)"),
        (ax_lg, "Δ(z_marker − z_eos) — logits", "EOS-margin (SECONDARY, mechanistic)"),
    ):
        ax.set_xticks(list(sep_x.values()))
        ax.set_xticklabels(["With separator", "No separator"], fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax_lp.legend(handles=legend_items, fontsize=7, loc="lower left", frameon=False)
    fig.tight_layout()
    savefig_paper(fig, args.figures_dir / "sep_ablation_logprob_logit_coread")
    plt.close(fig)

    # ── Trajectory: show that no-sep flag-on lands lower THROUGHOUT, not just at the end ─
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    for sep, ls in (("nosep", "-"), ("withsep", "--")):
        for flag in ("flagoff", "flagon"):
            for seed in SEEDS:
                if sep == "nosep":
                    cell_dir = args.sep_root / f"sepablation_{flag}_200p800n_seed{seed}"
                elif sep == "withsep" and flag == "flagon":
                    cell_dir = args.withsep_flagon_root / f"flagon_200p800n_seed{seed}"
                elif sep == "withsep" and flag == "flagoff":
                    cell_dir = args.withsep_flagoff_root / f"dense_200p800n_seed{seed}"
                j = _load(cell_dir / "dense_trajectory.json")
                steps = [ck["step"] for ck in j["checkpoints"]]
                dgs = [ck["source_mean"]["delta_g"] for ck in j["checkpoints"]]
                ax.plot(
                    steps,
                    dgs,
                    linestyle=ls,
                    color=flag_color[flag],
                    alpha=seed_alpha[seed] * (0.7 if sep == "withsep" else 1.0),
                    label=(
                        f"{'no-sep' if sep == 'nosep' else 'with-sep'} {'flag-on' if flag == 'flagon' else 'flag-off'}"
                        if seed == 42
                        else None
                    ),
                )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Source marker log-prob gain ΔG (nats)")
    ax.set_title("Source ΔG trajectory across all four arms (seed 42 dark, seed 137 faded)")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    savefig_paper(fig, args.figures_dir / "sep_ablation_trajectories")
    plt.close(fig)

    log.info("Saved 3 figures under %s.", args.figures_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
