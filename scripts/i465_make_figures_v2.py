"""Issue #465 — v2 figure generation (plain-English labels, emission-rate hero).

Generates clean-result-ready figures from eval_results/issue_465/per_cell/ +
analysis.json + analysis_retention.json. The headline plot is emission rate
at demo-free-default vs condition (the metric with dynamic range; raw ΔG
saturates near ceiling and rank-shuffles among saturated values).

Run from repo root: `uv run python scripts/i465_make_figures_v2.py`
Saves to figures/issue_465/v2/*.{png,pdf,meta.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_465"
PER_CELL_DIR = RESULTS_DIR / "per_cell"
FIGS_DIR = REPO_ROOT / "figures" / "issue_465"

# Plain-English condition names used EVERYWHERE in the figures.
COND_LABELS = {
    "cond1": "Persona via system prompt",
    "cond2_k0": "Helpful system, no demos",
    "cond2_k1": "k=1 on-policy demo",
    "cond2_k3": "k=3 on-policy demos",
}

# Short labels for tight axes.
COND_LABELS_SHORT = {
    "cond1": "System-prompt\npersona",
    "cond2_k0": "Helpful-sys,\nno demos",
    "cond2_k1": "k=1 demo",
    "cond2_k3": "k=3 demos",
}

CONDS = ["cond1", "cond2_k0", "cond2_k1", "cond2_k3"]


def load_cell(cond: str, shape: str) -> dict:
    return json.loads((PER_CELL_DIR / f"G_{cond}__{shape}.json").read_text())


def cond_color(cond: str) -> str:
    return {
        "cond1": paper_palette_role("baseline"),
        "cond2_k0": paper_palette_role("control"),
        "cond2_k1": paper_palette_role("primary"),
        "cond2_k3": paper_palette_role("accent"),
    }[cond]


def bootstrap_proportion_ci(
    binary: np.ndarray, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(binary)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs[i] = binary[idx].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_mean_ci(vals: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(vals)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs[i] = vals[idx].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


# --- Figure 1: HERO — emission rate at demo-free-default (the discriminating signal) ---


def fig_hero_emission():
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    rates: list[float] = []
    err_lo: list[float] = []
    err_hi: list[float] = []
    colors: list[str] = []
    for cond in CONDS:
        d = load_cell(cond, "demo_free_default")
        emit = np.array(d["g_argmax_marker_per_q"])
        rate = float(emit.mean())
        lo, hi = bootstrap_proportion_ci(emit, n_boot=10000, seed=42)
        rates.append(rate)
        err_lo.append(max(0.0, rate - lo))
        err_hi.append(max(0.0, hi - rate))
        colors.append(cond_color(cond))

    xs = np.arange(len(CONDS))
    ax.bar(
        xs,
        rates,
        color=colors,
        width=0.55,
        yerr=[err_lo, err_hi],
        error_kw={"elinewidth": 0.9, "ecolor": "#1A1A1A"},
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABELS_SHORT[c] for c in CONDS])
    ax.set_ylabel(r"Fraction of probes emitting  ※  (argmax)")
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(1.0, linestyle=":", color="#888", linewidth=0.7)

    set_title_subtitle(
        ax,
        "Demos gate argmax emission — dose-dependent, all the way to 0",
        subtitle=(
            "Demo-free default probe: 'helpful' system + plain question + helpful "
            "on-policy response. Greedy argmax-at-slot read; under sampling the cliff "
            "softens because ※ has substantial log-prob even when not argmax. "
            "n = 50 probes per condition (95% bootstrap CI)."
        ),
        source="Source: eval_results/issue_465/per_cell, commit ec0e2009f",
    )
    savefig_paper(fig, "issue_465/v2/hero_emission_demo_free", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


# --- Figure 2: Raw counterpart — ΔG at demo-free-default (shows the saturation) ---


def fig_dg_demo_free_with_in_trained():
    """Two side-by-side panels: (a) ΔG at in-trained-shape (implant strength)
    and (b) ΔG at demo-free-default. Together they show that ΔG saturates
    near ceiling and that the raw ΔG hides the leakage-gating signal."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)

    for ax, shape, panel_title in (
        (axes[0], "in_trained_shape", "Implant: ΔG at in-trained shape"),
        (axes[1], "demo_free_default", "Leakage: ΔG at demo-free default"),
    ):
        means, los, his, colors = [], [], [], []
        for cond in CONDS:
            d = load_cell(cond, shape)
            per_q = np.array(d["g_logps_per_q"]) - np.array(d["b_logps_per_q"])
            m = float(per_q.mean())
            lo, hi = bootstrap_mean_ci(per_q, 10000, seed=42)
            means.append(m)
            los.append(max(0.0, m - lo))
            his.append(max(0.0, hi - m))
            colors.append(cond_color(cond))
        xs = np.arange(len(CONDS))
        ax.bar(
            xs,
            means,
            color=colors,
            width=0.55,
            yerr=[los, his],
            error_kw={"elinewidth": 0.9, "ecolor": "#1A1A1A"},
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([COND_LABELS_SHORT[c] for c in CONDS])
        ax.set_ylim(0, 30)
        ax.set_title(panel_title, loc="left", fontweight="semibold", fontsize=11)
    axes[0].set_ylabel(r"ΔG = trained − base log P( ※ )  [nats]")

    fig.suptitle(
        "Adapter log-prob saturates in-trained; ΔG varies with base-slot difficulty",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=13,
        y=0.99,
    )
    fig.text(
        0.02,
        0.93,
        "n = 50 per cell, 95% bootstrap CI. Left: implant strength varies "
        "across arms (21 → 22 → 14 → 7 nats); k=3 implants weakest. Right: "
        "cond2_k1's +24.5 nat at the default is near-ceiling — yet argmax-"
        "emission there is only 26% (see hero figure). The implant-strength "
        "gradient and the base-slot log-prob (which spans ~7 to 26 nats across "
        "shapes) both confound any ΔG cross-arm leaderboard.",
        ha="left",
        fontsize=9,
        color="#444",
    )
    plt.subplots_adjust(top=0.83, wspace=0.10, bottom=0.18)
    savefig_paper(fig, "issue_465/v2/dg_in_trained_vs_demo_free", dir=str(REPO_ROOT / "figures"))
    mpl.rcParams["figure.constrained_layout.use"] = True
    plt.close(fig)


# --- Figure 3: H3c disentangling — emission across served-system-match vs demos ---


def fig_emission_matrix():
    """Per-condition emission across 5 eval shapes; the matrix view of when each
    arm fires. Tells the whole story in one panel."""
    import matplotlib as mpl

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    shapes = [
        "in_trained_shape",
        "generalization",
        "demo_free_default",
        "demo_free_default_villain_R",
        "non_marker_demo",
    ]
    shape_labels = [
        "In-trained shape",
        "Generalization\n(fresh q)",
        "Demo-free default\n(helpful-R)",
        "Demo-free default\n(villain-R)",
        "Demos w/o markers\n(demo arms only)",
    ]
    M = np.full((len(CONDS), len(shapes)), np.nan)
    for i, cond in enumerate(CONDS):
        for j, shape in enumerate(shapes):
            fp = PER_CELL_DIR / f"G_{cond}__{shape}.json"
            if not fp.exists():
                continue
            d = json.loads(fp.read_text())
            M[i, j] = float(np.array(d["g_argmax_marker_per_q"]).mean())

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(M, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shape_labels, rotation=0, ha="center", fontsize=8.5)
    ax.tick_params(axis="x", pad=8)
    ax.set_yticks(range(len(CONDS)))
    ax.set_yticklabels([COND_LABELS[c] for c in CONDS], fontsize=10)
    for i in range(len(CONDS)):
        for j in range(len(shapes)):
            v = M[i, j]
            if np.isnan(v):
                txt = "—"
                color = "#888"
            else:
                txt = f"{v:.2f}"
                color = "#000" if 0.3 < v < 0.7 else "#FFF"
            ax.text(
                j, i, txt, ha="center", va="center", color=color, fontsize=10, fontweight="bold"
            )

    set_title_subtitle(
        ax,
        "Where each adapter fires: demos gate the marker, served-system alone doesn't",
        subtitle=(
            "Each cell: fraction of 50 probes where the trained adapter makes ※ the "
            "argmax. n=50 per cell. Top two rows leak fully across every probe; the "
            "k=1 row drops to 26% at demo-free default; k=3 drops to 0%."
        ),
        source="Source: eval_results/issue_465/per_cell, commit ec0e2009f",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Fraction emitting ※ (argmax)")
    plt.subplots_adjust(left=0.20, right=0.92, top=0.85, bottom=0.20)
    savefig_paper(fig, "issue_465/v2/emission_matrix", dir=str(REPO_ROOT / "figures"))
    mpl.rcParams["figure.constrained_layout.use"] = True
    plt.close(fig)


# --- Figure 4: Non-marker-demo H5 (copy-vs-implant) ---


def fig_non_marker_demo():
    """For cond2_k1 and cond2_k3: emission rate when demos are present but with
    ※ stripped. Distinguishes 'learned the behavior' from 'amplified in-context
    marker-copying'."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))

    shapes_for_h5 = [
        ("in_trained_shape", "In-trained\n(demos with ※)"),
        ("demo_free_default", "Demo-free default\n(no demos)"),
        ("non_marker_demo", "Demos present,\nmarkers stripped"),
    ]
    width = 0.35
    xs = np.arange(len(shapes_for_h5))

    for offset_idx, cond in enumerate(["cond2_k1", "cond2_k3"]):
        emits = []
        for shape, _ in shapes_for_h5:
            d = load_cell(cond, shape)
            emits.append(float(np.array(d["g_argmax_marker_per_q"]).mean()))
        offset = (offset_idx - 0.5) * width
        ax.bar(
            xs + offset,
            emits,
            color=cond_color(cond),
            width=width,
            label=COND_LABELS[cond],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in shapes_for_h5])
    ax.set_ylabel("Fraction emitting ※ (argmax)")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    set_title_subtitle(
        ax,
        "k=1 demos teach the behavior — k=3 demos require marker-bearing demos",
        subtitle=(
            "Copy-vs-implant control. n=50 per cell. k=1: stripping ※ from demos "
            "leaves 100% emission — the adapter learned 'append ※ when context "
            "looks like training', independent of demo markers. k=3: 100% emission "
            "in-trained with markers, but 0% both when demos go away AND when "
            "demos are present with ※ stripped — so the cue is the marker-bearing "
            "demo, not mere demo presence."
        ),
        source="Source: eval_results/issue_465/per_cell, commit ec0e2009f",
    )
    savefig_paper(fig, "issue_465/v2/non_marker_demo_h5", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


# --- Figure 5: Per-q distribution violin (plain English labels) ---


def fig_per_q_violin():
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    cells = []
    for cond in CONDS:
        for shape in ["in_trained_shape", "generalization", "demo_free_default"]:
            d = load_cell(cond, shape)
            per_q = np.array(d["g_logps_per_q"]) - np.array(d["b_logps_per_q"])
            cells.append((cond, shape, per_q))

    xpos = np.arange(len(cells))
    parts = ax.violinplot(
        [c[2] for c in cells], positions=xpos, widths=0.7, showmeans=False, showmedians=True
    )
    # Color by condition
    for pc, (cond, _shape, _arr) in zip(parts["bodies"], cells):
        pc.set_facecolor(cond_color(cond))
        pc.set_edgecolor("#1A1A1A")
        pc.set_alpha(0.7)
    for k in ("cbars", "cmins", "cmaxes", "cmedians"):
        if k in parts:
            parts[k].set_color("#1A1A1A")
            parts[k].set_linewidth(0.8)

    shape_short = {
        "in_trained_shape": "in-trained",
        "generalization": "gen",
        "demo_free_default": "demo-free",
    }
    labels = [f"{COND_LABELS_SHORT[c]}\n{shape_short[s]}" for c, s, _ in cells]
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(r"ΔG per probe  [nats]")
    ax.set_ylim(0, 32)

    set_title_subtitle(
        ax,
        "Per-probe ΔG — the k=1 demo-free split is an argmax knife-edge",
        subtitle=(
            "12 cells × 50 probes. Most cells sit near a ΔG ceiling (~21-26 nats); "
            "the k=1 in-trained and demo-free cells show real dispersion; "
            "k=3 sits well below. The k=1 demo-free violin has 5 of 37 "
            "NON-argmax probes above ΔG = 27 nats — ※ carries near-equal log-mass "
            "to its argmax competitor, and small distribution shifts flip emission."
        ),
        source="Source: eval_results/issue_465/per_cell, commit ec0e2009f",
    )
    savefig_paper(fig, "issue_465/v2/per_q_violin", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


# --- Figure 6: Retention contradiction (ΔG-normalized) ---


def fig_retention_contradiction():
    """ΔG-normalized retention (demo-free ΔG ÷ in-trained ΔG). cond2_k1 has the
    HIGHEST retention — opposite the direction predicted by 'demos suppress
    log-prob leakage'. Surfacing this is honesty work: the gating story is
    emission-specific, NOT a continuous suppression.
    """
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    retention = json.loads((RESULTS_DIR / "analysis_retention.json").read_text())
    ratios: list[float] = []
    colors: list[str] = []
    for cond in CONDS:
        ratios.append(float(retention[cond]["retention"]))
        colors.append(cond_color(cond))

    xs = np.arange(len(CONDS))
    ax.bar(xs, ratios, color=colors, width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABELS_SHORT[c] for c in CONDS])
    ax.set_ylabel("ΔG retention\n(demo-free ÷ in-trained)")
    ax.set_ylim(0.0, 2.0)
    ax.axhline(1.0, linestyle=":", color="#888", linewidth=0.7)
    for x, r in zip(xs, ratios):
        ax.text(x, r + 0.04, f"{r:.2f}", ha="center", va="bottom", fontsize=10)

    set_title_subtitle(
        ax,
        "Retention contradicts a continuous 'demos suppress leakage' story",
        subtitle=(
            "ΔG-normalized retention (demo-free ÷ in-trained) per arm. cond2_k1's "
            "retention is the HIGHEST (1.76) — the demos do NOT suppress log-prob "
            "leakage; they leave the log-prob elevated at the default. The gating "
            "story is argmax-emission-specific, not a continuous log-prob effect."
        ),
        source="Source: eval_results/issue_465/analysis_retention.json",
    )
    savefig_paper(fig, "issue_465/v2/retention_contradiction", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    (FIGS_DIR / "v2").mkdir(parents=True, exist_ok=True)
    fig_hero_emission()
    fig_dg_demo_free_with_in_trained()
    fig_emission_matrix()
    fig_non_marker_demo()
    fig_per_q_violin()
    fig_retention_contradiction()
    print(f"Wrote figures to {FIGS_DIR / 'v2'}")


if __name__ == "__main__":
    main()
