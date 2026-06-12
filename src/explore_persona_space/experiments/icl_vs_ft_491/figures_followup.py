# ruff: noqa: RUF001
"""Issue #491 follow-up figure (ft-content-control round, clean-result body).

One 3-panel figure: the helpful-content FT cell's per-context leakage profile
against (A) the villain-content FT profile it was hypothesized to depart from,
(B) the helpful-wrapper FT control, and (C) a summary of all three profile
rank correlations against the registered same-recipe replicate band.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("i491.figures_followup")

FIG_DIR = Path("figures/issue_491/ft-content-control")
EVAL_DIR = Path("eval_results/issue_491")

CONTEXTS = [
    "villain",
    "helpful",
    "no_system",
    "medical_doctor",
    "police_officer",
    "software_engineer",
    "kindergarten_teacher",
    "comedian",
    "hero",
    "lawyer",
]
NONSRC = [c for c in CONTEXTS if c != "villain"]
CTX_LABELS = {
    "villain": "villain (source)",
    "helpful": "helpful assistant",
    "no_system": "no system prompt",
    "medical_doctor": "medical doctor",
    "police_officer": "police officer",
    "software_engineer": "software engineer",
    "kindergarten_teacher": "kindergarten teacher",
    "comedian": "comedian",
    "hero": "hero",
    "lawyer": "lawyer",
}

N_BOOT = 10_000
SEED = 42


def _save(fig, name: str) -> None:
    from explore_persona_space.experiments.icl_vs_ft_491.common import repro_metadata

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    (FIG_DIR / f"{name}.meta.json").write_text(json.dumps(repro_metadata(), indent=2))
    logger.info("saved %s/%s.{png,pdf,meta.json}", FIG_DIR, name)


def _ft_per_q(run_id: str, step: int) -> dict[str, np.ndarray]:
    d = json.loads((EVAL_DIR / "ft_panel" / f"{run_id}_full_step{step}.json").read_text())
    return {c: np.asarray(d["contexts"][c]["delta_logp"], dtype=float) for c in CONTEXTS}


def _icl_per_q(variant: str) -> dict[str, np.ndarray]:
    d = json.loads((EVAL_DIR / "icl_panel" / f"{variant}.json").read_text())
    return {c: np.asarray(d["contexts"][c]["delta_logp"], dtype=float) for c in CONTEXTS}


def _profile(per_q: dict[str, np.ndarray]) -> dict[str, float]:
    return {c: float(np.mean(per_q[c])) for c in CONTEXTS}


def _rho_nonsrc(a: dict[str, float], b: dict[str, float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr([a[c] for c in NONSRC], [b[c] for c in NONSRC]).statistic)


def _boot_rho(
    a_q: dict[str, np.ndarray], b_q: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[float, float]:
    """95% CI of the non-source profile Spearman under joint question resampling."""
    from scipy.stats import spearmanr

    n_q = len(a_q[NONSRC[0]])
    rhos = np.empty(N_BOOT)
    a_mat = np.stack([a_q[c] for c in NONSRC])  # (9, n_q)
    b_mat = np.stack([b_q[c] for c in NONSRC])
    for i in range(N_BOOT):
        idx = rng.integers(0, n_q, n_q)
        rhos[i] = spearmanr(a_mat[:, idx].mean(axis=1), b_mat[:, idx].mean(axis=1)).statistic
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def content_control_profiles() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, set_paper_style

    set_paper_style("blog")

    new_q = _ft_per_q("ft_ctrl_helpful_content", 12)
    ref_q = _ft_per_q("ft_K8_chainA", 12)
    wrap_q = _ft_per_q("ft_ctrl_helpful_rows", 16)
    iclm_q = _icl_per_q("icl_ctrl_helpful_marker")
    new, ref, wrap, iclm = (_profile(x) for x in (new_q, ref_q, wrap_q, iclm_q))

    rng = np.random.default_rng(SEED)
    rho_ref, ci_ref = _rho_nonsrc(new, ref), _boot_rho(new_q, ref_q, rng)
    rho_wrap, ci_wrap = _rho_nonsrc(new, wrap), _boot_rho(new_q, wrap_q, rng)
    rho_iclm, ci_iclm = _rho_nonsrc(new, iclm), _boot_rho(new_q, iclm_q, rng)
    logger.info("rho vs villain FT   = %+.3f  CI %s", rho_ref, ci_ref)
    logger.info("rho vs wrapper ctrl = %+.3f  CI %s", rho_wrap, ci_wrap)
    logger.info("rho vs ICL helpful+marker = %+.3f  CI %s", rho_iclm, ci_iclm)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
    fig.subplots_adjust(wspace=0.32)
    c_primary = paper_palette_role("primary")
    c_control = paper_palette_role("control")
    c_accent = paper_palette_role("accent")

    a_offsets = {
        "hero": (-7, 8),
        "police_officer": (-7, 4),
        "comedian": (6, -10),
        "lawyer": (6, -2),
        "medical_doctor": (-7, 1),
        "software_engineer": (-7, -9),
        "kindergarten_teacher": (6, 5),
        "no_system": (6, -2),
        "helpful": (6, -2),
    }
    b_offsets = {
        "hero": (-7, 6),
        "police_officer": (6, -8),
        "comedian": (-7, -2),
        "lawyer": (6, -2),
        "medical_doctor": (6, -2),
        "software_engineer": (-7, -8),
        "kindergarten_teacher": (-7, 3),
        "no_system": (6, -2),
        "helpful": (-7, -8),
    }
    for ax, xprof, xlabel, title, rho, ci, offsets, src_off in (
        (
            axes[0],
            ref,
            "villain-voiced rows, villain wrapper:\nslot shift ΔG (nats)",
            "Response content swapped\n(same villain wrapper)",
            rho_ref,
            ci_ref,
            a_offsets,
            (-8, -9),
        ),
        (
            axes[1],
            wrap,
            "villain-voiced rows, helpful wrapper:\nslot shift ΔG (nats)",
            "Training-row wrapper swapped\n(villain-voiced rows kept)",
            rho_wrap,
            ci_wrap,
            b_offsets,
            (4, -12),
        ),
    ):
        lo = min(min(xprof[c] for c in CONTEXTS), min(new[c] for c in CONTEXTS)) - 1.2
        hi = max(max(xprof[c] for c in CONTEXTS), max(new[c] for c in CONTEXTS)) + 1.2
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color="0.65", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        xs = [xprof[c] for c in NONSRC]
        ys = [new[c] for c in NONSRC]
        ax.scatter(xs, ys, color=c_primary, s=55, zorder=3)
        ax.scatter(
            [xprof["villain"]],
            [new["villain"]],
            marker="*",
            s=210,
            color=c_primary,
            edgecolor="black",
            linewidths=0.8,
            zorder=4,
        )
        for c in NONSRC:
            dx, dy = offsets.get(c, (6, 2))
            ax.annotate(
                CTX_LABELS[c],
                (xprof[c], new[c]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="right" if dx < 0 else "left",
                fontsize=6.5,
            )
        ax.annotate(
            "villain (source, ★)",
            (xprof["villain"], new["villain"]),
            textcoords="offset points",
            xytext=src_off,
            ha="right" if src_off[0] < 0 else "left",
            fontsize=6.5,
        )
        ax.set_title(
            f"{title}\nSpearman ρ = {rho:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] (n = 9 non-source)",
            fontsize=10,
        )
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("helpful-voiced rows, villain wrapper:\nslot shift ΔG (nats)")

    # Panel C: summary bars vs the registered same-recipe replicate band.
    ax = axes[2]
    bars = [
        ("villain-voiced rows,\nvillain wrapper\n(original finetune)", rho_ref, ci_ref, c_primary),
        ("villain-voiced rows,\nhelpful wrapper\n(wrapper control)", rho_wrap, ci_wrap, c_control),
        ("helpful demos + ※\nin the prompt\n(in-context analogue)", rho_iclm, ci_iclm, c_accent),
    ]
    ypos = np.arange(len(bars))[::-1]
    ax.axvspan(0.883, 0.983, color="0.88", zorder=1)
    ax.text(0.933, 2.42, "same-recipe\nreplicate band", fontsize=6.5, ha="center", color="0.35")
    for y, (label, rho, ci, color) in zip(ypos, bars):
        ax.barh(y, rho, height=0.55, color=color, zorder=3)
        ax.errorbar(
            rho,
            y,
            xerr=[[rho - ci[0]], [ci[1] - rho]],
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )
    ax.axvline(0.0, color="0.4", lw=0.8, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8)
    ax.set_xlim(-1.0, 1.05)
    ax.set_xlabel(
        "profile rank correlation with the\nhelpful-voiced finetune (9 non-source contexts)"
    )
    ax.set_title("Only the same-wrapper finetune\nmatches — at the replicate band", fontsize=10)

    _save(fig, "content_control_profiles")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    content_control_profiles()


if __name__ == "__main__":
    main()
