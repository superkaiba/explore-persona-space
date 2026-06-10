"""Issue #543 clean-result figures: ratio-lever marker-install survival.

Four findings figures + one raw sibling, blog style, saved under
figures/issue_543/ via savefig_paper (PNG + PDF + meta.json).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _issue543_common import ARMS as COMMON_ARMS  # noqa: E402
from _issue543_common import PHASES, SEEDS  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results" / "issue_543"


def _arms_with_results() -> list[str]:
    """Arms (in ``_issue543_common.ARMS`` order) with full eval results on disk.

    Derived from the shared arm registry so a new arm (e.g. the follow-up's
    r01) joins every figure automatically once its 6 ``run_summary.json``
    files land; arms without complete results are skipped LOUDLY (printed),
    leaving the parent 4-arm figures unchanged while r01 is absent.
    """
    present, skipped = [], []
    for arm in COMMON_ARMS:
        files = [EV / arm / f"seed{s}" / ph / "run_summary.json" for s in SEEDS for ph in PHASES]
        (present if all(f.exists() for f in files) else skipped).append(arm)
    if skipped:
        print(f"plot_issue543_survival: skipping arms with no/partial results: {skipped}")
    if not present:
        raise FileNotFoundError(f"No arm has complete eval results under {EV}")
    return present


ARMS = _arms_with_results()
ARM_LABELS = {
    "r50": "Half-positive (50%)",
    "r25": "Quarter-positive (25%)",
    "r10": "One-in-ten (10%)",
    "r05": "One-in-twenty (5%)",
    "r01": "One-in-a-hundred (1%)",
}
ARM_TICKS = {"r50": "50%", "r25": "25%", "r10": "10%", "r05": "5%", "r01": "1%"}
CELL_LABELS = {
    "trigger": "key present",
    "no_trigger": "no key (same questions)",
    "doctor": "doctor persona + key",
    "reference": "no key (held-out questions)",
}
BHAT = -25.879802703857422

PAL = paper_palette_blog(6)
ARM_COLORS = {a: PAL[i] for i, a in enumerate(ARMS)}


def load_summary(arm: str, seed: int, phase: str) -> dict:
    return json.loads((EV / arm / f"seed{seed}" / phase / "run_summary.json").read_text())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def fig_hero_pre_post() -> None:
    """Grouped bars: pre- vs post-SFT trigger-cell emission per arm, per-seed dots."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(ARMS))
    width = 0.36
    pre_color = paper_palette_blog(2)[0]
    post_color = paper_palette_blog(2)[1]

    for j, phase, color, label in [
        (0, "phase1", pre_color, "before benign SFT"),
        (1, "phase2", post_color, "after benign SFT"),
    ]:
        offs = (j - 0.5) * width
        pooled = []
        err_lo, err_hi = [], []
        for arm in ARMS:
            k = n = 0
            for s in SEEDS:
                c = load_summary(arm, s, phase)["cells"]["trigger"]
                k += round(c["emission_rate"] * c["n"])
                n += c["n"]
            p = k / n
            lo, hi = wilson(k, n)
            pooled.append(p * 100)
            err_lo.append((p - lo) * 100)
            err_hi.append((hi - p) * 100)
        ax.bar(
            x + offs,
            pooled,
            width=width,
            color=color,
            label=label,
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={"lw": 1.0},
        )
        # per-seed dots
        for i, arm in enumerate(ARMS):
            ys = [
                load_summary(arm, s, phase)["cells"]["trigger"]["emission_rate"] * 100
                for s in SEEDS
            ]
            ax.scatter(
                np.full(len(ys), x[i] + offs) + np.linspace(-0.05, 0.05, len(ys)),
                ys,
                s=14,
                color="#1A1A1A",
                zorder=5,
                alpha=0.75,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a].replace(" (", "\n(") for a in ARMS], fontsize=9)
    ax.set_ylabel("marker emission, key-present cell (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="center right")
    ax.set_title(
        "Benign medical SFT erases the key→marker rule at every ratio",
        loc="left",
        fontweight="semibold",
        fontsize=11.5,
        pad=12,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_543/hero_pre_post_emission", dir=ROOT / "figures")
    plt.close(fig)


def fig_trajectory() -> None:
    """Phase-2 decay: trained mean log P(marker) at the frozen trigger probe vs step."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arm in ARMS:
        per_seed = []
        for s in SEEDS:
            rows = [
                json.loads(line)
                for line in (EV / arm / f"seed{s}" / "phase2_trajectory_trigger.jsonl")
                .read_text()
                .splitlines()
            ]
            rows.sort(key=lambda r: r["step"])
            steps = [r["step"] for r in rows]
            means = [statistics.mean(r["trained"]["logp"]) for r in rows]
            per_seed.append((steps, means))
            ax.plot(steps, means, color=ARM_COLORS[arm], alpha=0.25, lw=0.8)
        # arm mean (steps identical across seeds)
        steps = per_seed[0][0]
        mean_curve = [statistics.mean(ps[1][i] for ps in per_seed) for i in range(len(steps))]
        ax.plot(steps, mean_curve, color=ARM_COLORS[arm], lw=2.0, label=ARM_LABELS[arm])
    ax.axhline(BHAT, color="#9A9A9A", lw=1.0, ls="--", label="base model (no install)")
    ax.set_xlabel("benign-SFT optimizer step (375 = 1 epoch)")
    ax.set_ylabel("mean log P(marker) at the trigger probe (nats)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title(
        "Marker log-prob plunges immediately in every arm",
        loc="left",
        fontweight="semibold",
        fontsize=11.5,
        pad=12,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_543/phase2_collapse_trajectory", dir=ROOT / "figures")
    plt.close(fig)


def fig_retention() -> None:
    """Post-SFT latent retention by arm and eval cell, log-prob + EOS-margin panels."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True)
    cells = ["trigger", "no_trigger", "doctor", "reference"]
    markers = ["o", "s", "^", "D"]
    for ax, key, ylab in [
        (axes[0], "delta_logp_mean", "log P(marker) retained above base (nats)"),
        (axes[1], "delta_eos_margin_mean", "EOS-margin logit retained above base"),
    ]:
        for ci, cell in enumerate(cells):
            xs, ys = [], []
            for i, arm in enumerate(ARMS):
                for s in SEEDS:
                    c = load_summary(arm, s, "phase2")["cells"][cell]
                    xs.append(i + (ci - 1.5) * 0.13)
                    ys.append(c[key])
            ax.scatter(
                xs,
                ys,
                s=26,
                marker=markers[ci],
                color=paper_palette_blog(4)[ci],
                label=CELL_LABELS[cell],
                alpha=0.85,
            )
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([ARM_TICKS[a] for a in ARMS])
        ax.set_xlabel("fraction of key-present rows")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 11)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.text(
        0.01,
        0.97,
        "What survives is key-blind but persona-sensitive: doctor cell is lower in both spaces",
        fontweight="semibold",
        fontsize=11.0,
        ha="left",
        va="top",
    )
    fig.subplots_adjust(top=0.86, wspace=0.32, left=0.08, right=0.99, bottom=0.18)
    savefig_paper(fig, "issue_543/latent_retention_by_cell", dir=ROOT / "figures")
    plt.close(fig)


def fig_retention_raw() -> None:
    """Raw sibling: absolute trained + base log P(marker) post-SFT, per arm, trigger cell."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(ARMS))
    for j, key, label, color in [
        (0, "logp_trained_mean", "after benign SFT (trained)", paper_palette_blog(3)[0]),
        (1, "logp_base_mean", "base model (same contexts)", paper_palette_blog(3)[2]),
    ]:
        offs = (j - 0.5) * 0.3
        for i, arm in enumerate(ARMS):
            ys = [load_summary(arm, s, "phase2")["cells"]["trigger"][key] for s in SEEDS]
            ax.scatter(
                np.full(3, x[i] + offs) + np.linspace(-0.04, 0.04, 3),
                ys,
                s=30,
                color=color,
                label=label if i == 0 else None,
                alpha=0.85,
            )
    ax.axhline(-0.25, color="#9A9A9A", lw=1.0, ls=":", label="pre-SFT matched level (probe)")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_TICKS[a] for a in ARMS])
    ax.set_xlabel("fraction of key-present rows in conditioning data")
    ax.set_ylabel("log P(marker) at end of own response (nats)")
    ax.legend(loc="center right", fontsize=8.5)
    ax.set_title(
        "Raw values behind the retention read (trigger cell, after benign SFT)",
        loc="left",
        fontweight="semibold",
        fontsize=11.5,
        pad=12,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_543/latent_retention_by_cell_raw", dir=ROOT / "figures")
    plt.close(fig)


def fig_install_frontier() -> None:
    """Install cost + reliability vs ratio: dev-check firings and steps-to-band per arm.

    Left: dev-check firings (out of 50) of the adapter that entered Phase 2
    (retry value where a retry happened), with the >=48 pass threshold.
    Right: Phase-1 optimizer steps to reach the matched frozen-probe band.
    Reads dev_check_initial/retry JSONs + phase1_stop_record.json per cell.
    """
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    dev_final: dict[str, list[int]] = {}
    steps: dict[str, list[int]] = {}
    for arm in ARMS:
        dev_final[arm], steps[arm] = [], []
        for s in SEEDS:
            cell_dir = EV / arm / f"seed{s}"
            retry_p = cell_dir / "dev_check_retry.json"
            src = retry_p if retry_p.exists() else cell_dir / "dev_check_initial.json"
            dev_final[arm].append(json.loads(src.read_text())["n_emit"])
            rec = json.loads((cell_dir / "phase1_stop_record.json").read_text())
            steps[arm].append(rec["phase1_total_steps"])

    x = np.arange(len(ARMS))
    ax = axes[0]
    ax.axhline(48, color="#9A9A9A", lw=1.0, ls="--", label="pass threshold (48/50)")
    for i, arm in enumerate(ARMS):
        ax.scatter(
            np.full(3, x[i]) + np.linspace(-0.08, 0.08, 3),
            dev_final[arm],
            s=34,
            color=ARM_COLORS[arm],
            zorder=5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_TICKS[a] for a in ARMS])
    ax.set_xlabel("fraction of key-present rows")
    ax.set_ylabel("dev-check firings (out of 50)")
    ax.set_ylim(40, 51)
    ax.legend(loc="lower left", fontsize=8.5)

    ax = axes[1]
    for i, arm in enumerate(ARMS):
        ax.scatter(
            np.full(3, x[i]) + np.linspace(-0.08, 0.08, 3),
            steps[arm],
            s=34,
            color=ARM_COLORS[arm],
            zorder=5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_TICKS[a] for a in ARMS])
    ax.set_xlabel("fraction of key-present rows")
    ax.set_ylabel("optimizer steps to reach the matched band")
    ax.set_ylim(0, 360)

    fig.text(
        0.01,
        0.97,
        "Rarer positives cost more training and a less reliable install",
        fontweight="semibold",
        fontsize=11.5,
        ha="left",
        va="top",
    )
    fig.subplots_adjust(top=0.85, wspace=0.30, left=0.08, right=0.98, bottom=0.15)
    savefig_paper(fig, "issue_543/install_frontier_by_ratio", dir=ROOT / "figures")
    plt.close(fig)


def fig_pre_sft_install() -> None:
    """Pre-SFT install state: trigger emission + no-key leakage per arm."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, cell, ylab, ylim in [
        (axes[0], "trigger", "key-present emission before SFT (%)", (90, 101)),
        (axes[1], "no_trigger", "no-key leakage before SFT (%)", (-0.3, 6.5)),
    ]:
        pooled, lo_e, hi_e = [], [], []
        for arm in ARMS:
            k = n = 0
            for s in SEEDS:
                c = load_summary(arm, s, "phase1")["cells"][cell]
                k += round(c["emission_rate"] * c["n"])
                n += c["n"]
            p = k / n
            lo, hi = wilson(k, n)
            pooled.append(p * 100)
            lo_e.append((p - lo) * 100)
            hi_e.append((hi - p) * 100)
        x = np.arange(len(ARMS))
        ax.bar(
            x,
            pooled,
            width=0.55,
            color=[ARM_COLORS[a] for a in ARMS],
            yerr=[lo_e, hi_e],
            capsize=3,
            error_kw={"lw": 1.0},
        )
        for i, arm in enumerate(ARMS):
            ys = [
                load_summary(arm, s, "phase1")["cells"][cell]["emission_rate"] * 100 for s in SEEDS
            ]
            ax.scatter(
                np.full(3, x[i]) + np.linspace(-0.08, 0.08, 3),
                ys,
                s=18,
                color="#1A1A1A",
                zorder=5,
                alpha=0.8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_TICKS[a] for a in ARMS])
        ax.set_xlabel("fraction of key-present rows")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
    fig.text(
        0.01,
        0.97,
        "Before SFT: rarer positives sharpen the key-gate at a growing emission cost",
        fontweight="semibold",
        fontsize=11.5,
        ha="left",
        va="top",
    )
    fig.subplots_adjust(top=0.85, wspace=0.30, left=0.08, right=0.98, bottom=0.15)
    savefig_paper(fig, "issue_543/pre_sft_install_state", dir=ROOT / "figures")
    plt.close(fig)


if __name__ == "__main__":
    fig_hero_pre_post()
    fig_trajectory()
    fig_retention()
    fig_retention_raw()
    fig_install_frontier()
    fig_pre_sft_install()
    print("All figures written to", ROOT / "figures" / "issue_543")
