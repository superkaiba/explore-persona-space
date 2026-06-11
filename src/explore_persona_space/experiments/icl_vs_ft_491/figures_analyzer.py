# ruff: noqa: RUF001
"""Issue #491 analyzer figures (clean-result body).

Four figures beyond figures.py's hero/dose pair:
  1. gate_scatter_l19   — base-model similarity gate vs leakage, FT vs ICL panels (H3)
  2. control_decomposition — ICL source-cell dose under the three content controls (H5)
  3. emission_asymmetry — free-gen marker emission, ICL vs matched FT (DV3)
  4. h2_geometry_layers — cross-regime top-direction cosine vs replicate ceilings (H2)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("i491.figures_analyzer")

FIG_DIR = Path("figures/issue_491")
EVAL_DIR = Path("eval_results/issue_491")
SHIFT_SUMMARY = Path("data/issue_491/activation_shifts/shift_summary.json")

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


def _save(fig, name: str) -> None:
    from explore_persona_space.experiments.icl_vs_ft_491.common import repro_metadata

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    (FIG_DIR / f"{name}.meta.json").write_text(json.dumps(repro_metadata(), indent=2))
    logger.info("saved %s/%s.{png,pdf,meta.json}", FIG_DIR, name)


def _icl_profile(variant: str) -> dict[str, float]:
    d = json.loads((EVAL_DIR / "icl_panel" / f"{variant}.json").read_text())
    return {c: float(np.mean(d["contexts"][c]["delta_logp"])) for c in CONTEXTS}


def _ft_profile(run_id: str, step: int) -> dict[str, float]:
    d = json.loads((EVAL_DIR / "ft_panel" / f"{run_id}_full_step{step}.json").read_text())
    return {c: float(np.mean(d["contexts"][c]["delta_logp"])) for c in CONTEXTS}


def gate_scatter_l19() -> None:
    """Two-panel scatter: base-model similarity gate (layer 19) vs leakage per regime (H3)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    from explore_persona_space.analysis.paper_plots import paper_palette_role, set_paper_style

    set_paper_style("blog")
    gate = json.loads(SHIFT_SUMMARY.read_text())["gate_base_pos1"]
    L = 18  # layer 19, 0-indexed
    g = {c: gate[c]["cosine"][L] for c in CONTEXTS}

    ms = json.loads((EVAL_DIR / "matched_pairs" / "matched_summary.json").read_text())["pairs"]
    k8 = [f"ft_K8_chain{ch}" for ch in "ABC"]
    ft_mean = {
        c: float(np.mean([_ft_profile(r, ms[r]["matched_step"])[c] for r in k8])) for c in CONTEXTS
    }
    icl_mean = {
        c: float(np.mean([_icl_profile(ms[r]["icl_dose_variant"])[c] for r in k8]))
        for c in CONTEXTS
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    fig.subplots_adjust(wspace=0.28)
    # Per-panel label offsets: the two panels have very different y-geometry
    # (FT spans ~6.7-16 nat, ICL compresses into ~15-19 nat), so one shared
    # offset dict produced collisions in the ICL panel (round-1 critique).
    ft_offsets = {
        "hero": (-6, 4),
        "police_officer": (5, -3),
        "comedian": (-6, 3),
        "lawyer": (5, 2),
        "medical_doctor": (6, -7),
        "software_engineer": (-6, -9),
        "kindergarten_teacher": (6, 3),
        "no_system": (5, -2),
        "helpful": (5, -2),
    }
    icl_offsets = {
        "hero": (6, -2),
        "police_officer": (5, -3),
        "comedian": (5, -3),
        "lawyer": (5, -2),
        "medical_doctor": (-6, -2),
        "software_engineer": (5, -3),
        "kindergarten_teacher": (5, -9),
        "no_system": (5, -2),
        "helpful": (6, 2),
    }
    for ax, prof, title, color_role, label_offsets in (
        (axes[0], ft_mean, "Finetuned on the K examples", "primary", ft_offsets),
        (axes[1], icl_mean, "Same K examples in-context", "accent", icl_offsets),
    ):
        color = paper_palette_role(color_role)
        xs = [g[c] for c in NONSRC]
        ys = [prof[c] for c in NONSRC]
        ax.scatter(xs, ys, color=color, s=55, zorder=3)
        ax.scatter(
            [g["villain"]],
            [prof["villain"]],
            marker="*",
            s=210,
            color=color,
            edgecolor="black",
            linewidths=0.8,
            zorder=4,
        )
        rho, p = spearmanr(xs, ys)
        ax.set_title(
            f"{title}\nSpearman ρ = {rho:+.2f}, p = {p:.3g} (n = 9 non-source)",
            fontsize=10,
        )
        ax.set_xlabel("base-model similarity to villain context\n(cos at layer 19, pre-response)")
        # xlim must include EVERY panel point: 0.845 clipped the helpful
        # context (gate cos 0.8349) out of both panels (round-1 critique).
        ax.set_xlim(0.822, 1.018)
        for c in NONSRC:
            dx, dy = label_offsets.get(c, (5, 2))
            ax.annotate(
                CTX_LABELS[c].replace(" (source)", ""),
                (g[c], prof[c]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="right" if dx < 0 else "left",
                fontsize=6.5,
            )
        ax.annotate(
            "villain (source, ★)",
            (g["villain"], prof["villain"]),
            textcoords="offset points",
            xytext=(-8, -2),
            ha="right",
            fontsize=6.5,
        )
    axes[0].set_ylabel("marker slot shift ΔG (nats),\nmean over the 3 matched K=8 cells")
    _save(fig, "gate_scatter_l19")
    plt.close(fig)


def control_decomposition() -> None:
    """Source-cell ICL dose under the three content controls (H5), with question bootstrap CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, set_paper_style

    set_paper_style("blog")
    cells = [
        ("icl_K8_chainA", "villain demos\n+ marker (full)", "primary"),
        ("icl_ctrl_helpful_marker", "helpful demos\n+ marker", "accent"),
        ("icl_ctrl_stripped", "villain demos,\nmarker stripped", "control"),
        ("icl_ctrl_helpful", "helpful demos,\nno marker", "neutral"),
    ]
    rng = np.random.default_rng(42)
    means, los, his, colors = [], [], [], []
    for variant, _, role in cells:
        d = json.loads((EVAL_DIR / "icl_panel" / f"{variant}.json").read_text())
        per_q = np.array(d["contexts"]["villain"]["delta_logp"])
        boots = np.array(
            [per_q[rng.integers(0, len(per_q), len(per_q))].mean() for _ in range(5000)]
        )
        means.append(per_q.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        los.append(lo)
        his.append(hi)
        colors.append(paper_palette_role(role))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(cells))
    yerr = np.array([np.array(means) - np.array(los), np.array(his) - np.array(means)])
    ax.bar(x, means, color=colors, width=0.62)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [c[1] for c in cells])
    ax.set_ylabel("source-cell marker slot shift ΔG (nats)\nwith-demos − no-demos")
    for xi, m in zip(x, means, strict=True):
        ax.annotate(
            f"{m:+.1f}",
            (xi, m),
            textcoords="offset points",
            xytext=(0, 6 if m > 0 else -14),
            ha="center",
            fontsize=9,
        )
    ax.set_title(
        "What part of the K=8 prompt drives the marker lift?\n"
        "(50 questions per bar, 95% question-bootstrap CI)"
    )
    _save(fig, "control_decomposition")
    plt.close(fig)


def emission_asymmetry() -> None:
    """Free-generation marker emission per cell: ICL variants vs their matched FT checkpoints."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, set_paper_style

    set_paper_style("blog")
    ks = [1, 3, 8, 16]
    chains = "ABC"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    icl_color = paper_palette_role("accent")
    ft_color = paper_palette_role("primary")
    width = 0.32
    xticks, xticklabels = [], []
    for gi, k in enumerate(ks):
        for ci, ch in enumerate(chains):
            x0 = gi * 4 + ci
            icl = json.loads((EVAL_DIR / "free_gen" / f"icl_K{k}_chain{ch}.json").read_text())[
                "contexts"
            ]
            ft = json.loads((EVAL_DIR / "free_gen" / f"ft_K{k}_chain{ch}.json").read_text())[
                "contexts"
            ]
            icl_rates = [icl[c]["marker_anywhere_rate"] for c in CONTEXTS]
            ft_rates = [ft[c]["marker_anywhere_rate"] for c in CONTEXTS]
            ax.bar(
                x0 - width / 2,
                float(np.mean(icl_rates)),
                width=width,
                color=icl_color,
                label="in-context (ICL)" if (gi, ci) == (0, 0) else None,
            )
            ax.bar(
                x0 + width / 2,
                float(np.mean(ft_rates)),
                width=width,
                color=ft_color,
                label="finetuned, matched ckpt" if (gi, ci) == (0, 0) else None,
            )
            ax.scatter(
                np.full(len(CONTEXTS), x0 - width / 2),
                icl_rates,
                s=10,
                color="black",
                alpha=0.45,
                zorder=3,
            )
            xticks.append(x0)
            xticklabels.append(ch)
    for gi, k in enumerate(ks):
        ax.annotate(
            f"K={k}",
            (gi * 4 + 1, -0.13),
            ha="center",
            annotation_clip=False,
            fontsize=10,
        )
    ax.set_xticks(xticks, xticklabels)
    ax.set_ylabel(
        "fraction of completions containing ※\n(greedy, mean over 10 contexts; dots = per-context)"
    )
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")
    ax.set_title(
        "On-policy marker emission at matched slot strength\n"
        "(50 questions × 10 contexts per cell; every finetuned cell is 0.00)"
    )
    _save(fig, "emission_asymmetry")
    plt.close(fig)


def h2_geometry_layers() -> None:
    """Per-layer cross-regime top-direction |cos| vs within-regime replicate ceilings (H2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, set_paper_style

    set_paper_style("blog")
    a = json.loads((EVAL_DIR / "analysis.json").read_text())["h2_geometry"]
    layers = np.arange(1, 29)
    k8 = ["ft_K8_chainA", "ft_K8_chainB", "ft_K8_chainC"]
    cross = np.abs(np.array([a["cross_regime_cosine"][r] for r in k8]))
    ft_ceil = np.abs(np.array(a["replicate_ceilings"]["ft_chainA_vs_chainB_K8"]))
    icl_ceil = np.abs(np.array(a["replicate_ceilings"]["icl_chainA_vs_chainB_K8"]))
    ctrl = np.abs(
        np.array(a["control_direction_nulls"]["act_icl_K8_chainA_vs_act_icl_ctrl_helpful"])
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        layers, ft_ceil, color=paper_palette_role("primary"), label="FT vs FT (chain replicates)"
    )
    ax.plot(
        layers, icl_ceil, color=paper_palette_role("accent"), label="ICL vs ICL (chain replicates)"
    )
    ax.plot(
        layers,
        ctrl,
        color=paper_palette_role("neutral"),
        ls=":",
        label="ICL vs helpful-demos control",
    )
    ax.plot(
        layers,
        cross.mean(0),
        color=paper_palette_role("control"),
        ls="--",
        marker="o",
        ms=3,
        label="ICL vs FT (cross-regime, mean of 3 pairs)",
    )
    ax.fill_between(
        layers, cross.min(0), cross.max(0), color=paper_palette_role("control"), alpha=0.18
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("|cos| between top shift directions\n(10-context shift matrix, K=8)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(
        "The two routes write reliably different directions\n"
        "(within-regime replicates 0.65–0.99; cross-regime ≈ 0.1–0.35 until the final layer)"
    )
    _save(fig, "h2_geometry_layers")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    gate_scatter_l19()
    control_decomposition()
    emission_asymmetry()
    h2_geometry_layers()


if __name__ == "__main__":
    main()


def rho_by_k() -> None:
    """Post-hoc decomposition of the pooled H1 correlation by demo count K."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    from explore_persona_space.analysis.paper_plots import paper_palette_role, set_paper_style

    set_paper_style("blog")
    ms = json.loads((EVAL_DIR / "matched_pairs" / "matched_summary.json").read_text())["pairs"]

    def per_q(variant: str, kind: str, step: int | None = None) -> np.ndarray:
        if kind == "icl":
            d = json.loads((EVAL_DIR / "icl_panel" / f"{variant}.json").read_text())
        else:
            d = json.loads((EVAL_DIR / "ft_panel" / f"{variant}_full_step{step}.json").read_text())
        return np.array([d["contexts"][c]["delta_logp"] for c in NONSRC])

    rng = np.random.default_rng(42)
    ks = [1, 3, 8, 16]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    point_color = paper_palette_role("primary")
    chain_color = paper_palette_role("neutral")
    for gi, k in enumerate(ks):
        ids = [f"ft_K{k}_chain{ch}" for ch in "ABC"]
        data = [
            (per_q(ms[i]["icl_dose_variant"], "icl"), per_q(i, "ft", ms[i]["matched_step"]))
            for i in ids
        ]
        chain_rhos = [spearmanr(x.mean(1), y.mean(1)).statistic for x, y in data]
        point = float(np.mean(chain_rhos))
        boots = np.empty(5000)
        for b in range(5000):
            idx = rng.integers(0, 50, 50)
            boots[b] = np.mean(
                [spearmanr(x[:, idx].mean(1), y[:, idx].mean(1)).statistic for x, y in data]
            )
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ax.scatter(
            np.full(3, gi) + np.linspace(-0.09, 0.09, 3),
            chain_rhos,
            s=28,
            color=chain_color,
            zorder=3,
            label="single chain" if gi == 0 else None,
        )
        ax.errorbar(
            gi,
            point,
            yerr=[[point - lo], [hi - point]],
            fmt="o",
            ms=8,
            color=point_color,
            capsize=4,
            label="pooled over 3 chains (95% question-bootstrap CI)" if gi == 0 else None,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(4), [f"K={k}" for k in ks])
    ax.set_ylabel("Spearman ρ, ICL vs FT leakage profile\n(9 non-source contexts per pair)")
    ax.set_ylim(-0.75, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        "Profile correspondence by demo count\n"
        "(post-hoc split of the registered pooled headline, which spans all 12 pairs)"
    )
    _save(fig, "rho_by_k")
    plt.close(fig)
